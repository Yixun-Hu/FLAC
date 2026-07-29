"""Write a copy of a PyTorch-Lightning checkpoint with the optimizer state removed.

Purpose (exp_09, **F-reset** arm): resume training from the 87.5k full-parity
anchor with a *freshly initialised* Adam (moments and per-parameter ``step``
zeroed) while keeping every other resume-critical piece of state — model + EMA
weights, ``global_step`` / ``epoch`` / loop counters, the LR-scheduler position
and the callback states. The anchor checkpoint itself is **never** modified:
this tool only ever writes to a different path (enforced, see ``ValueError``
below).

Why ``optimizer_states = []`` and not ``del checkpoint["optimizer_states"]``
---------------------------------------------------------------------------
Verified against the installed ``pytorch_lightning==2.1.0`` in env ``flac``:

* ``pytorch_lightning/trainer/connectors/checkpoint_connector.py:361-365``
  (``_CheckpointConnector.restore_optimizers_and_schedulers``)::

      if "optimizer_states" not in self._loaded_checkpoint:
          raise KeyError(
              "Trying to restore optimizer state but checkpoint contains only the model."
              ...

  i.e. an **absent** key is a hard crash on resume, not a "start fresh" signal.
  (The same function, lines 368-372, raises the analogous ``KeyError`` when
  ``lr_schedulers`` is missing — which is why this tool keeps that key intact.)

* ``pytorch_lightning/strategies/strategy.py:365-369``
  (``Strategy.load_optimizer_state_dict``, the implementation used by
  ``DDPStrategy`` — only ``fsdp``/``deepspeed`` override it)::

      optimizer_states = checkpoint["optimizer_states"]
      for optimizer, opt_state in zip(self.optimizers, optimizer_states):
          optimizer.load_state_dict(opt_state)

  With an **empty list** the ``zip`` yields nothing, so every optimizer keeps
  exactly what ``LightningModule.configure_optimizers`` built — a fresh Adam.
  No warning, no crash.

Conclusion: *present-but-empty* is the only spelling that cleanly yields fresh
optimizers under PL 2.1, so that is what this tool writes.

Caveat (documented, intentional)
--------------------------------
The stripped checkpoint no longer carries ``param_groups``, so on resume the
optimizer's ``lr`` starts at the config's base value (5e-5 for FLAC) instead of
the scheduled value. ``lr_schedulers`` *is* restored, and FLAC's scheduler runs
with ``interval="step"`` (``src/training/diffusion.py:191-201``), so the
scheduler re-imposes its own value after the first optimizer step. Net effect:
exactly **one** micro-step at the base LR (5e-5 vs the analytic InverseLR value
~4.7946e-5 at 87.5k — a ~4% one-step difference) before the schedule takes over
again. This is accepted for the F-reset probe; a warm resume (``OPT_RESET``
unset) is unaffected.

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
    """True when ``a`` and ``b`` name the same file, across path spellings/symlinks."""
    if os.path.realpath(os.path.abspath(a)) == os.path.realpath(os.path.abspath(b)):
        return True
    try:
        return os.path.samefile(a, b)
    except OSError:  # one of them does not exist yet -> not the same file
        return False


def strip_optimizer_state(in_path: str, out_path: str, force: bool = False) -> str:
    """Write ``in_path`` to ``out_path`` with ``optimizer_states`` emptied.

    Fail-closed on: a missing input, an output that resolves to the input, and an
    existing output (unless ``force``). Returns ``out_path``.
    """
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"input checkpoint not found: {in_path}")
    if _same_file(in_path, out_path):
        raise ValueError(
            f"refusing to overwrite the input checkpoint: --out resolves to --in ({in_path}). "
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

    removed = ckpt[OPTIMIZER_KEY]
    n_opt = len(removed) if isinstance(removed, list) else 0
    n_state = sum(len(o.get("state", {})) for o in removed if isinstance(o, dict))
    n_groups = sum(len(o.get("param_groups", [])) for o in removed if isinstance(o, dict))

    # PRESENT-but-EMPTY (see module docstring): absent would KeyError on resume.
    ckpt[OPTIMIZER_KEY] = []

    kept = [k for k in ckpt if k != OPTIMIZER_KEY]

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(ckpt, out_path)

    print("=== strip_optimizer_state ===")
    print(f"  in : {in_path}")
    print(f"  out: {out_path}")
    print(
        f"  REMOVED: {OPTIMIZER_KEY} -> [] "
        f"({n_opt} optimizer(s), {n_state} per-param state entries, {n_groups} param_group(s)) "
        "=> PL 2.1 restores nothing and Adam re-initialises fresh"
    )
    print(f"  KEPT   : {', '.join(kept)}")
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
