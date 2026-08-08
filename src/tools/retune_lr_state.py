"""Write a copy of a PyTorch-Lightning checkpoint whose InverseLR schedule is *retuned*.

Purpose (exp_13, **decay_tail** arm): continue the 87,500-step P1 anchor on a
decaying learning-rate tail — ``InverseLR(inv_gamma=30000, power=1.0)`` instead of
the recipe's non-decaying ``InverseLR(inv_gamma=1e6, power=0.5)`` — so that late
training stops *orbiting* and settles.  The anchor checkpoint itself is never
modified: this tool only ever writes to a different path (enforced below).

Why the treatment CANNOT be delivered by the model config (the B1 blocker)
-------------------------------------------------------------------------
``InverseLR`` (``src/training/utils.py:21``) stores ``inv_gamma`` / ``power`` /
``warmup`` / ``final_lr`` as plain instance attributes.  ``torch``'s
``LRScheduler.state_dict()`` serialises *every* attribute except ``optimizer``, and
``LRScheduler.load_state_dict`` (``torch/optim/lr_scheduler.py:149``) is literally
``self.__dict__.update(state_dict)``.  On a warm resume PL therefore **overwrites**
the freshly built scheduler's hyperparameters with the ones baked into the
checkpoint.  Measured in env ``flac``: build ``InverseLR(30000, 1.0, 0.99)``, restore
the 87.5k anchor's scheduler state, and ``inv_gamma`` reverts to ``1000000`` and
``power`` to ``0.5``; the lr at 97,500 would be ``4.7727e-05``, i.e. the config-only
swap is a **silent no-op**.  ``src/tests/test_retune_lr_state.py`` pins that
reproduction as a regression test.

The rewrite (three scalars, nothing else)
-----------------------------------------
::

    lr_schedulers[0]['inv_gamma']                 1000000 -> 30000
    lr_schedulers[0]['power']                         0.5 -> 1.0
    optimizer_states[0]['param_groups'][0]['lr']  4.794633014853842e-05
                                                  -> 1.2765957446808513e-05

The third rewrite matters because PL restores ``param_groups`` *before* the first
optimizer update and the scheduler only fires *after* it
(``src/training/diffusion.py`` logs ``interval="step"``): without it, step 87,501
would still run at the anchor's ``4.79e-05``.  ``1.2765957446808513e-05`` is the
closed-form ``InverseLR(30000, 1.0, 0.99)`` value at ``last_epoch == 87500`` over
``base_lr == 5e-05``, and the tool RE-DERIVES it from the checkpoint's own
``base_lrs`` / ``warmup`` / ``final_lr`` / ``last_epoch`` and refuses to write unless
the derivation reproduces the pinned constant exactly (so a checkpoint at another
step, or with a different base lr, is rejected rather than mis-set).

Everything else is preserved verbatim: the scheduler's counters
(``last_epoch`` / ``_step_count`` / ``_last_lr``), ``base_lrs``, ``warmup``,
``final_lr``, the full Adam moment state (this is a WARM continuation),
``initial_lr`` and the other param-group fields, the model + EMA ``state_dict``,
``loops`` / ``callbacks`` / ``global_step`` / ``epoch``, and the embedded
``model_config`` (which stays BVp1 — the launcher's INITIAL-mode lineage assert
reads it).

Usage
-----
    python -m src.tools.retune_lr_state --in <anchor.ckpt> --out <retuned.ckpt> [--force]
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

OPTIMIZER_KEY = "optimizer_states"
SCHEDULER_KEY = "lr_schedulers"

# --- the exp_13 treatment (plan_decay_tail.md §1) ---
NEW_INV_GAMMA = 30000
NEW_POWER = 1.0
ANCHOR_LAST_EPOCH = 87500
TAIL_LR = 1.2765957446808513e-05
# tolerance for the self-consistency re-derivation; the arithmetic is identical to the
# scheduler's own, so the difference is expected to be exactly 0.0
_LR_ABSTOL = 1e-18


def _same_file(a: str, b: str) -> bool:
    """True when ``a`` and ``b`` name the same file (``./`` spellings, symlinks, hardlinks)."""
    if os.path.realpath(os.path.abspath(a)) == os.path.realpath(os.path.abspath(b)):
        return True
    try:
        return os.path.samefile(a, b)  # inode identity -> catches hardlinks
    except OSError:  # one of them does not exist yet -> not the same file
        return False


def inverse_lr_closed_form(last_epoch, inv_gamma, power, warmup, final_lr, base_lr):
    """``src/training/utils.py::InverseLR._get_closed_form_lr`` for a single base lr."""
    w = 1 - warmup ** (last_epoch + 1)
    lr_mult = (1 + last_epoch / inv_gamma) ** -power
    return w * max(final_lr, base_lr * lr_mult)


def _require(cond, exc, msg):
    if not cond:
        raise exc(msg)


def retune_lr_state(in_path: str, out_path: str, force: bool = False) -> str:
    """Write ``in_path`` to ``out_path`` with the InverseLR tail schedule installed.

    Rewrites exactly three scalars (see the module docstring) and preserves everything
    else.  Fail-closed on: a missing input, an output that is the same file as the
    input, an existing output (unless ``force``), a non-Lightning checkpoint, a missing
    or malformed ``optimizer_states`` / ``lr_schedulers``, more than one optimizer /
    scheduler / param_group, and a checkpoint whose scheduler state does not re-derive
    the pinned tail lr (wrong resume step or wrong base lr).  Returns ``out_path``.
    """
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"input checkpoint not found: {in_path}")
    if _same_file(in_path, out_path):
        raise ValueError(
            f"refusing to overwrite the input checkpoint: --out is the same file as --in ({in_path}). "
            "The 87.5k anchor must never be modified."
        )
    if os.path.exists(out_path) and not force:
        raise FileExistsError(f"output already exists (pass force=True/--force to overwrite): {out_path}")

    ckpt = torch.load(in_path, map_location="cpu", weights_only=False)
    _require(isinstance(ckpt, dict), TypeError,
             f"not a Lightning checkpoint dict: {in_path} (got {type(ckpt).__name__})")

    # ---------------- scheduler state ----------------
    if SCHEDULER_KEY not in ckpt:
        raise KeyError(
            f"{in_path} has no '{SCHEDULER_KEY}' key — it is not a full training checkpoint "
            "and cannot serve as a resume anchor (PL 2.1 KeyErrors on resume)."
        )
    scheds = ckpt[SCHEDULER_KEY]
    _require(isinstance(scheds, list), TypeError,
             f"'{SCHEDULER_KEY}' is not a list (got {type(scheds).__name__})")
    _require(len(scheds) == 1, ValueError,
             f"expected exactly 1 lr_scheduler entry, found {len(scheds)} — refusing to guess "
             "which one carries the diffusion schedule")
    sched = scheds[0]
    _require(isinstance(sched, dict), TypeError,
             f"'{SCHEDULER_KEY}[0]' is not a scheduler state_dict (got {type(sched).__name__})")
    for k in ("inv_gamma", "power", "warmup", "final_lr", "base_lrs", "last_epoch"):
        if k not in sched:
            raise KeyError(
                f"'{SCHEDULER_KEY}[0]' has no '{k}' key — this is not an InverseLR state_dict "
                "(exp_13 retunes InverseLR only)."
            )
    base_lrs = sched["base_lrs"]
    _require(isinstance(base_lrs, list) and len(base_lrs) == 1, ValueError,
             f"expected exactly 1 base_lr, found {base_lrs!r}")

    # ---------------- optimizer state ----------------
    if OPTIMIZER_KEY not in ckpt:
        raise KeyError(
            f"{in_path} has no '{OPTIMIZER_KEY}' key — it is not a full training checkpoint "
            "(weights-only?) and cannot serve as a resume anchor."
        )
    opt_states = ckpt[OPTIMIZER_KEY]
    _require(isinstance(opt_states, list), TypeError,
             f"'{OPTIMIZER_KEY}' is not a list (got {type(opt_states).__name__})")
    _require(len(opt_states) == 1, ValueError,
             f"expected exactly 1 optimizer entry, found {len(opt_states)}")
    entry = opt_states[0]
    _require(isinstance(entry, dict) and "state" in entry and "param_groups" in entry, TypeError,
             f"'{OPTIMIZER_KEY}[0]' is not a torch optimizer state_dict "
             "(expected keys 'state' and 'param_groups')")
    groups = entry["param_groups"]
    _require(isinstance(groups, list), TypeError,
             f"'{OPTIMIZER_KEY}[0]['param_groups']' is not a list")
    _require(len(groups) == 1, ValueError,
             f"expected exactly 1 param_group, found {len(groups)} — refusing to guess which "
             "group carries the diffusion lr")
    group = groups[0]
    _require(isinstance(group, dict), TypeError, "param_groups[0] is not a dict")
    if "lr" not in group:
        raise KeyError(f"'{OPTIMIZER_KEY}[0]['param_groups'][0]' has no 'lr' key")

    # ---------------- self-consistency: re-derive the tail lr from THIS checkpoint ----
    last_epoch = sched["last_epoch"]
    derived = inverse_lr_closed_form(
        last_epoch=last_epoch, inv_gamma=NEW_INV_GAMMA, power=NEW_POWER,
        warmup=sched["warmup"], final_lr=sched["final_lr"], base_lr=base_lrs[0],
    )
    _require(abs(derived - TAIL_LR) <= _LR_ABSTOL, ValueError,
             f"the retuned schedule does not reproduce the pinned tail lr at this checkpoint: "
             f"InverseLR({NEW_INV_GAMMA}, {NEW_POWER}, warmup={sched['warmup']}) over "
             f"base_lr={base_lrs[0]!r} at last_epoch={last_epoch!r} gives {derived!r}, "
             f"expected {TAIL_LR!r} (exp_13's constant is pinned to last_epoch "
             f"{ANCHOR_LAST_EPOCH} over base_lr 5e-05). Refusing to write a mis-set lr.")

    old_inv_gamma, old_power, old_lr = sched["inv_gamma"], sched["power"], group["lr"]

    # ---------------- the rewrite ----------------
    sched["inv_gamma"] = NEW_INV_GAMMA
    sched["power"] = NEW_POWER
    group["lr"] = TAIL_LR

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(ckpt, out_path)

    kept = [k for k in ckpt if k not in (OPTIMIZER_KEY, SCHEDULER_KEY)]
    print("=== retune_lr_state (InverseLR decay tail) ===")
    print(f"  in : {in_path}")
    print(f"  out: {out_path}")
    print(f"  REWROTE lr_schedulers[0]['inv_gamma']: {old_inv_gamma!r} -> {NEW_INV_GAMMA!r}")
    print(f"  REWROTE lr_schedulers[0]['power']    : {old_power!r} -> {NEW_POWER!r}")
    print(f"  REWROTE optimizer_states[0]['param_groups'][0]['lr']: {old_lr!r} -> {TAIL_LR!r}")
    print(f"    (re-derived from this checkpoint: InverseLR({NEW_INV_GAMMA}, {NEW_POWER}, "
          f"warmup={sched['warmup']!r}) over base_lr={base_lrs[0]!r} at "
          f"last_epoch={last_epoch!r} = {derived!r})")
    print(f"  KEPT (scheduler): last_epoch={sched['last_epoch']!r} _step_count={sched.get('_step_count')!r} "
          f"_last_lr={sched.get('_last_lr')!r} base_lrs={base_lrs!r} warmup={sched['warmup']!r} "
          f"final_lr={sched['final_lr']!r}")
    print(f"  KEPT (optimizer): {len(entry['state'])} per-parameter state entries (WARM), "
          f"initial_lr={group.get('initial_lr')!r}, "
          f"{len([k for k in group if k != 'lr'])} other param_group field(s) verbatim")
    print(f"  KEPT (top level): {', '.join(kept)}")
    print(f"  resume position: global_step={ckpt.get('global_step')!r} epoch={ckpt.get('epoch')!r}")
    sd = ckpt.get("state_dict") or {}
    ema_step = sd.get("diffusion_ema.step")
    if ema_step is not None:
        print(f"  EMA: {sum(1 for k in sd if k.startswith('diffusion_ema.'))} entries, step={int(ema_step)}")
    mc = ckpt.get("model_config")
    if isinstance(mc, dict):
        print("  embedded model_config: UNCHANGED "
              f"({len(mc)} top-level keys) — still the config the anchor was trained under")
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="in_path", required=True, help="source checkpoint (never modified)")
    ap.add_argument("--out", dest="out_path", required=True, help="destination checkpoint copy")
    ap.add_argument("--force", action="store_true", help="overwrite an existing --out")
    args = ap.parse_args(argv)
    try:
        retune_lr_state(args.in_path, args.out_path, force=args.force)
    except (FileNotFoundError, FileExistsError, ValueError, KeyError, TypeError) as e:
        print(f"retune_lr_state FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
