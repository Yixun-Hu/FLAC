"""Write a copy of a PyTorch-Lightning checkpoint with the Adam *state* cleared.

Purpose (exp_09, **F-reset** arm): resume training from the 87.5k full-parity
anchor with freshly initialised Adam moments (and per-parameter ``step`` zeroed)
while keeping every other resume-critical piece of state — model + EMA weights,
``global_step`` / ``epoch`` / loop counters, the LR-scheduler position, the
callback states **and the scheduled learning rate**. The anchor checkpoint
itself is never modified: this tool only ever writes to a different path
(enforced; see ``ValueError`` below).

Treatment: keep-entry / clear-state
-----------------------------------
Each entry of ``optimizer_states`` keeps its ``param_groups`` verbatim and has
only its per-parameter ``state`` emptied::

    {"state": {...449 entries...}, "param_groups": [...]}
      ->  {"state": {},            "param_groups": [...]}   # groups unchanged

Why not ``del checkpoint["optimizer_states"]`` and why not ``[]``
-----------------------------------------------------------------
Verified against the installed ``pytorch_lightning==2.1.0`` / ``torch`` in env
``flac`` and confirmed by direct execution (the numbers below are measured):

1. **Absent key = crash.** ``checkpoint_connector.py:361-365``
   (``_CheckpointConnector.restore_optimizers_and_schedulers``) raises
   ``KeyError("Trying to restore optimizer state but checkpoint contains only
   the model…")`` when ``optimizer_states`` is missing. The same function
   (lines 368-372) raises the analogous ``KeyError`` for a missing
   ``lr_schedulers`` — which is why this tool keeps that key intact too.

2. **Empty list = accepted by PL, but wrong LR.** ``strategies/strategy.py:365-369``
   (``Strategy.load_optimizer_state_dict``; the implementation ``DDPStrategy``
   inherits — only ``fsdp``/``deepspeed`` override it) does
   ``for optimizer, opt_state in zip(self.optimizers, checkpoint["optimizer_states"])``.
   An empty list therefore restores *nothing* — including no ``param_groups``.
   The optimizer keeps whatever ``configure_optimizers`` left in it, and that is
   **not** the base LR: ``InverseLR`` steps once at construction
   (``torch/optim/lr_scheduler.py:133-136``, ``_initial_step``), writing the
   step-0 warmup value ``(1 - 0.99**1) * 5e-5 = 5.0e-07`` into the live
   param_group (this warmup-at-construction behaviour is already pinned, for the
   5e-6 case, by ``src/tests/test_finetune_cond.py:664``).
   ``LRScheduler.load_state_dict`` (``torch/optim/lr_scheduler.py:149``) is just
   ``self.__dict__.update(state_dict)`` — it restores the scheduler's *counters*
   and never rewrites ``optimizer.param_groups``. Net effect of the empty-list
   spelling: the first optimizer update after resume runs at **5.0e-07 instead
   of the scheduled 4.794633e-05** (~96x low); only the post-step
   ``interval="step"`` scheduler call (``src/training/diffusion.py:191-201``)
   restores the schedule afterwards. That LR asymmetry between F-warm and
   F-reset is avoidable, so it is not used.

3. **Keep-entry / clear-state = correct.** ``optimizer.load_state_dict`` is still
   called with the retained entry, so ``param_groups`` (hence
   ``lr == 4.794633014853842e-05`` and ``initial_lr == 5e-5``) is restored
   exactly as in a warm resume, while the empty ``state`` mapping makes Adam
   allocate fresh ``exp_avg`` / ``exp_avg_sq`` with ``step == 1`` on the first
   update. Measured: post-restore ``lr = 4.794633e-05`` and
   ``len(optimizer.state) == 0``; first update at ``4.794633e-05``;
   ``len(optimizer.state) == 1`` after it. F-reset therefore differs from
   F-warm in *exactly* the moments, nothing else.

Usage
-----
    python -m src.tools.strip_optimizer_state --in <anchor.ckpt> --out <copy.ckpt> [--force]
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

OPTIMIZER_KEY = "optimizer_states"


def _same_file(a: str, b: str) -> bool:
    """True when ``a`` and ``b`` name the same file (``./`` spellings, symlinks, hardlinks)."""
    if os.path.realpath(os.path.abspath(a)) == os.path.realpath(os.path.abspath(b)):
        return True
    try:
        return os.path.samefile(a, b)  # inode identity -> catches hardlinks
    except OSError:  # one of them does not exist yet -> not the same file
        return False


def strip_optimizer_state(in_path: str, out_path: str, force: bool = False) -> str:
    """Write ``in_path`` to ``out_path`` with every optimizer's ``state`` emptied.

    ``param_groups`` (and therefore the scheduled LR) are preserved verbatim; see the
    module docstring for why the entry is kept rather than dropped.

    Fail-closed on: a missing input, an output that is the same file as the input, and
    an existing output (unless ``force``). Returns ``out_path``.
    """
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"input checkpoint not found: {in_path}")
    if _same_file(in_path, out_path):
        raise ValueError(
            f"refusing to overwrite the input checkpoint: --out is the same file as --in ({in_path}). "
            "The anchor checkpoint must never be modified."
        )
    if os.path.exists(out_path) and not force:
        raise FileExistsError(f"output already exists (pass force=True/--force to overwrite): {out_path}")

    ckpt = torch.load(in_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"not a Lightning checkpoint dict: {in_path} (got {type(ckpt).__name__})")
    if OPTIMIZER_KEY not in ckpt:
        raise KeyError(
            f"{in_path} has no '{OPTIMIZER_KEY}' key — it is not a full training checkpoint "
            "(weights-only?) and cannot serve as a resume anchor."
        )
    opt_states = ckpt[OPTIMIZER_KEY]
    if not isinstance(opt_states, list):
        raise TypeError(f"'{OPTIMIZER_KEY}' is not a list (got {type(opt_states).__name__})")

    n_cleared, n_groups = 0, 0
    for i, entry in enumerate(opt_states):
        if not isinstance(entry, dict) or "state" not in entry or "param_groups" not in entry:
            raise TypeError(
                f"'{OPTIMIZER_KEY}[{i}]' is not a torch optimizer state_dict "
                "(expected keys 'state' and 'param_groups')"
            )
        n_cleared += len(entry["state"])
        n_groups += len(entry["param_groups"])
        entry["state"] = {}  # moments + per-param step reset; param_groups untouched

    kept = [k for k in ckpt if k != OPTIMIZER_KEY]

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(ckpt, out_path)

    lrs = [g.get("lr") for e in opt_states for g in e["param_groups"]]
    print("=== strip_optimizer_state (keep-entry / clear-state) ===")
    print(f"  in : {in_path}")
    print(f"  out: {out_path}")
    print(
        f"  REMOVED: per-parameter optimizer state — {n_cleared} entries across "
        f"{len(opt_states)} optimizer(s) (exp_avg / exp_avg_sq / step) => Adam re-initialises fresh"
    )
    print(
        f"  KEPT (inside {OPTIMIZER_KEY}): {n_groups} param_group(s) verbatim, lr={lrs} "
        "=> the scheduled LR is preserved (no step-0 warmup-LR artifact)"
    )
    print(f"  KEPT (top level): {', '.join(kept)}")
    gs, ep = ckpt.get("global_step"), ckpt.get("epoch")
    print(f"  resume position: global_step={gs} epoch={ep}")
    scheds = ckpt.get("lr_schedulers") or []
    if scheds and isinstance(scheds[0], dict):
        print(
            f"  lr_schedulers[0]: last_epoch={scheds[0].get('last_epoch')} "
            f"_step_count={scheds[0].get('_step_count')} _last_lr={scheds[0].get('_last_lr')}"
        )
    sd = ckpt.get("state_dict") or {}
    ema_step = sd.get("diffusion_ema.step")
    if ema_step is not None:
        print(f"  EMA: {sum(1 for k in sd if k.startswith('diffusion_ema.'))} entries, step={int(ema_step)}")
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="in_path", required=True, help="source checkpoint (never modified)")
    ap.add_argument("--out", dest="out_path", required=True, help="destination checkpoint copy")
    ap.add_argument("--force", action="store_true", help="overwrite an existing --out")
    args = ap.parse_args(argv)
    try:
        strip_optimizer_state(args.in_path, args.out_path, force=args.force)
    except (FileNotFoundError, FileExistsError, ValueError, KeyError, TypeError) as e:
        print(f"strip_optimizer_state FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
