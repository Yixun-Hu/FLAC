#!/usr/bin/env python
"""exp_11 P0 profiling runner — train.py's fit with ONE extra timing callback.

PARITY NOTE (read first). The exp_11 ARMS TRAIN WITH ``train.py``; this runner
exists ONLY to instrument the P0 profiling stage and must never train an arm.
It builds the run through exactly the same factory path as ``train.py``'s
``main()`` — ``prefigure.get_all_args`` over the same ``defaults.ini`` ->
``create_dataloader_from_config`` -> ``create_model_from_config`` ->
``create_training_wrapper_from_config`` -> ``train.construct_trainer`` (the same
``build_trainer_kwargs`` boundary, so precision/strategy/SyncBN/max_steps/accum
are assembled by the tested code, not re-implemented here) — and adds a single
``pl.Callback`` plus three fail-closed guards (logger must be ``none``, no
validation, no resume). Nothing else differs.

Why it exists (round-2 review B2): full-process wall time is dominated by
imports, VAE load, DDP rendezvous and first-batch warmup. Those terms do NOT
cancel between two independent Slurm jobs, so a 10-step/30-step *job pair*
cannot measure steady state. ``P0StepTimer`` instead timestamps completed
optimizer steps inside ONE fit, after ``torch.cuda.synchronize()``, on rank
zero:

    steps/s = (30 - 10) / (t_mono(30) - t_mono(10))

Heavy imports (``train``, ``src.*``) are deferred into ``main()`` so the unit
tests can import this module cheaply.
"""
import math
import os
import sys
import time

import pytorch_lightning as pl
import torch

# The two optimizer steps that bound the measurement window.
P0_WINDOW = (10, 30)
# Exact format of the printed marks; the sbatch greps these and the collector
# receives the extracted values through the P0RESULT line.
P0STEP_RE = r"P0STEP step=(?P<step>\d+) t=(?P<t>[0-9.]+) ts=(?P<ts>[0-9.]+)"


class P0StepTimer(pl.Callback):
    """Timestamp the window steps: monotonic (for the rate) + epoch (to window
    the GPU utilisation/power poller against the same interval).

    ``trainer.global_step`` counts COMPLETED optimizer steps, so with accum 1 the
    mark at ``global_step == n`` is taken after exactly ``n`` steps. CUDA is
    synchronised first — otherwise the timestamp races queued kernels and the
    measured rate is the launch rate, not the compute rate. Marks are taken on
    every rank (the sync must not diverge across ranks) but printed only by rank
    zero."""

    def __init__(self, window=P0_WINDOW):
        super().__init__()
        self.window = tuple(int(w) for w in window)
        self.marks = {}       # step -> time.monotonic()
        self.wall_marks = {}  # step -> time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step not in self.window or step in self.marks:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        mono, wall = time.monotonic(), time.time()
        self.marks[step] = mono
        self.wall_marks[step] = wall
        if int(getattr(trainer, "global_rank", 0)) == 0:
            print(f"P0STEP step={step} t={mono:.6f} ts={wall:.6f}", flush=True)

    def complete(self):
        return all(step in self.marks for step in self.window)


def window_rate(marks, window=P0_WINDOW):
    """steps/s from a ``{step: monotonic}`` mapping over ``window``.

    Fail-closed: a missing mark, a non-finite timestamp or a non-positive delta
    raises instead of yielding a plausible-looking number."""
    lo, hi = int(window[0]), int(window[-1])
    missing = [s for s in (lo, hi) if s not in marks]
    if missing:
        raise ValueError(f"missing window mark(s) for step(s) {missing}")
    t_lo, t_hi = float(marks[lo]), float(marks[hi])
    if not (math.isfinite(t_lo) and math.isfinite(t_hi)):
        raise ValueError(f"non-finite window marks: {t_lo}, {t_hi}")
    delta = t_hi - t_lo
    if delta <= 0:
        raise ValueError(f"non-positive window delta {delta} (t{lo}={t_lo}, t{hi}={t_hi})")
    return (hi - lo) / delta


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))  # exp dir -> worklog_yixun -> worklog -> repo
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import json

    from prefigure.prefigure import get_all_args

    import train as flac_train
    from src.data.dataset import create_dataloader_from_config
    from src.models import create_model_from_config
    from src.models.utils import load_ckpt_state_dict, remove_weight_norm_from_model
    from src.training import create_training_wrapper_from_config

    torch.set_float32_matmul_precision("medium")
    torch.multiprocessing.set_sharing_strategy("file_system")
    args = get_all_args()

    # --- P0 guards: this runner may only ever do a logger-free, validation-free,
    # --- fresh-start profiling fit (anything else belongs to train.py) ---
    if str(args.logger) != "none":
        raise SystemExit(f"p0_runner requires --logger none, got {args.logger!r}")
    if args.val_dataset_config or int(args.val_every) > 0:
        raise SystemExit("p0_runner does not run validation")
    if args.ckpt_path or args.pretrained_ckpt_path:
        raise SystemExit("p0_runner profiles from-scratch steps only (no resume/finetune)")

    seed = args.seed
    if os.environ.get("SLURM_PROCID") is not None:  # verbatim train.py behaviour
        seed += int(os.environ.get("SLURM_PROCID"))
    pl.seed_everything(seed, workers=True)

    with open(args.model_config) as f:
        model_config = json.load(f)
    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    train_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 1),
    )

    model = create_model_from_config(model_config)
    if args.remove_pretransform_weight_norm == "pre_load":
        remove_weight_norm_from_model(model.pretransform)
    if args.pretransform_ckpt_path:
        model.pretransform.load_state_dict(load_ckpt_state_dict(args.pretransform_ckpt_path))
    if args.remove_pretransform_weight_norm == "post_load":
        remove_weight_norm_from_model(model.pretransform)

    training_wrapper = create_training_wrapper_from_config(model_config, model)

    checkpoint_dir = args.save_dir if args.save_dir else None
    timer = P0StepTimer()
    callbacks = [
        pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every,
                                     dirpath=checkpoint_dir, save_top_k=-1),
        flac_train.ExceptionCallback(),
        flac_train.ModelConfigEmbedderCallback(model_config),
        timer,
    ]

    if args.strategy:
        strategy = args.strategy
    else:
        strategy = "ddp_find_unused_parameters_true" if args.num_gpus > 1 else "auto"

    trainer = flac_train.construct_trainer(
        args, strategy=strategy, callbacks=callbacks, logger=None,
        checkpoint_dir=checkpoint_dir, val_args={},
    )
    print(f"P0RUNNER: cfg={args.model_config} steps={args.max_steps} "
          f"micro={args.batch_size} gpus={args.num_gpus} strategy={strategy} "
          f"world_size_env={os.environ.get('WORLD_SIZE', 'unset')}", flush=True)

    trainer.fit(training_wrapper, train_dl)

    if int(getattr(trainer, "global_rank", 0)) == 0:
        if not timer.complete():
            print(f"P0RUNNER FAIL: window marks incomplete ({sorted(timer.marks)}); "
                  f"max_steps must exceed {P0_WINDOW[-1]}", flush=True)
            return 2
        print(f"P0RUNNER: window rate {window_rate(timer.marks):.6f} steps/s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
