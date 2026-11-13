#!/usr/bin/env python3
"""exp_43 std_aug -- run ``train.py`` in-process with OBSERVE-ONLY evidence (plan_std_aug.md §5.1).

Forked from the exp_40 evidence driver (whose folder stays byte-frozen). Usage (from the run
tree root, which is the cwd)::

    EXP43_EVIDENCE_DIR=<abs dir> EXP43_EVIDENCE_MODE=verbatim|digest EXP43_PUBLISH=nas|off \
        python <runtree>/worklog/worklog_yixun/exp_43_std_aug_claude/exp43_evidence_driver.py <train.py argv>

``sys.argv[0]`` stays this file, so PyTorch Lightning's subprocess DDP launcher
(``_basic_subprocess_cmd``: ``[python, abspath(sys.argv[0])] + sys.argv[1:]``) re-executes
THIS driver for every further rank; torchrun starts it once per rank. Every wrapper below
calls the original with the original arguments and returns its result unchanged:

* ``DiffusionCondTrainingWrapper._apply_yaw_aug`` + ``src.training.diffusion.rotate_scene_metadata``
  -- the column offset every sample of every step actually received;
* ``src.training.diffusion.mirror_scene_metadata`` (the very function object of
  ``src.data.yaw_mirror``; install refuses otherwise) -- the MIRROR SPY: per step and rank, the
  indices of the samples actually passed to the mirror, resolved by object identity against
  that step's rotated samples, and the check that the returned batch is exactly the rotated
  samples with those mirrored outputs substituted. Each step's applied ``(offsets, flip bits)``
  is compared with a PRIVATE recomputation (a fresh ``torch.Generator`` seeded like
  ``_apply_yaw_aug``'s: the offsets, then ``torch.randint(0, 2, (n,))``; plan §2.3), and the
  streaming digest is taken over ``"<step>:<offsets>|<bits>\n"``;
* ``src.models.transformer.flash_attn_func`` -- the number of flash-attention calls (the
  DiT's attention branch is chosen by this module global at call time);
* ``torch.utils.checkpoint.checkpoint`` -- calls by ``use_reentrant`` value (FLAC's ViT
  layer wrapper resolves it by attribute at call time, ``conditioners.py``);
* ``DiffusionCondTrainingWrapper.on_fit_start`` (after ``strategy.setup``: SyncBN
  conversion + DDP broadcast done) -- rank, world size, GPU identity, SyncBatchNorm count,
  the wrapper's yaw_aug state, and the initial-state signature (full ``state_dict`` digest,
  python / NumPy / torch CPU / CUDA RNG digests, Sobol engine state);
* ``on_train_end`` -- the counters, written atomically as ``rank<r>.summary.json``;
* ``ModelCheckpoint._save_checkpoint`` (FULL) -- right after each synchronous save: the
  ``.saved`` marker, then the bounded, killable NAS publication + acknowledgement; the outcome
  is broadcast so every rank raises on a failure (exp_40 plan §14 B2).

Evidence is RANK-PRIVATE: rank r appends only to ``<dir>/rank<r>.jsonl``. Nothing here
draws from the python / NumPy / torch global streams (the recomputation uses its own generator),
moves a tensor the step uses, or changes a return value.

Written by the exp_43 Coder seat (Claude Opus 5.5) per plan_std_aug.md §5.1 on the exp_40 driver.
"""
import hashlib
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def _fail(msg):
    print(f"EXP43 EVIDENCE DRIVER REFUSED: {msg}", flush=True)
    sys.exit(86)


SAMPLED_STEPS = (0, 1, 2, 3, 1000, 2499, 2500, 39997, 39998, 39999)
HIST_BINS, HIST_WIDTH = 16, 32          # 16 coarse bins of the unquantised offsets: bin = d // 32


class Evidence:
    """Per-process evidence. ``verbatim`` (smokes): every step's offsets and applied flip bits as
    records. Both modes keep the streaming digest the FULL receipt is judged on (sha256 over
    ``"<step>:<d0>,<d1>,...|<b0>,<b1>,...\n"``, bit i = 1 iff sample i was passed to the
    mirror), the 16-bin offset histogram, the applied-flip count, the sampled steps' offsets,
    bits and losses, and a device-side all-losses-finite flag (exp_40 plan §14 H4: no per-step
    loss digest, no per-step host sync)."""

    def __init__(self, directory, verbatim):
        self.dir = directory
        self.verbatim = verbatim
        self.rank = None
        self.fh = None
        self.counts = {"flash_calls": 0, "ckpt_dinov3": 0, "ckpt_dit": 0, "ckpt_unclassified": 0,
                       "ckpt_reentrant": 0, "ckpt_default": 0, "apply_yaw_aug": 0,
                       "training_step": 0, "checkpoints_published": 0,
                       "mirror_calls": 0, "flip_anomalies": 0, "stream_mismatch_steps": 0}
        self.collecting = None         # inside one _apply_yaw_aug: the offsets reaching the rotation
        self.rotated = None            # ... the rotated sample dicts, in call order
        self.mirrored = None           # ... {sample index: mirrored output} of the samples mirrored
        self.stream_hash = hashlib.sha256()
        self.finite = None             # device-side all-finite flag (plan §14 H4: no per-step sync)
        self.hist = [0] * 16
        self.n_offsets = 0
        self.steps = []
        self.sampled = {}
        self.sampled_flips = {}
        self.flips_applied = 0
        self.losses_sampled = {}

    def bind_rank(self, rank):
        rank = int(rank)
        if self.rank is None:
            self.rank = rank
            path = os.path.join(self.dir, f"rank{rank}.jsonl")
            if os.path.exists(path):
                _fail(f"{path} exists: evidence files are written once per run")
            self.fh = open(path, "a", buffering=1)
        elif self.rank != rank:
            _fail(f"global rank changed from {self.rank} to {rank} inside one process")

    def write(self, record):
        self.fh.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        self.fh.flush()

    def record_step(self, step, offsets, flips, t):
        self.stream_hash.update(f"{step}:{','.join(map(str, offsets))}|{','.join(map(str, flips))}\n".encode())
        for d in offsets:
            if type(d) is int and 0 <= d < HIST_BINS * HIST_WIDTH:
                self.hist[d // HIST_WIDTH] += 1
        self.n_offsets += len(offsets)
        self.flips_applied += sum(flips)
        self.steps.append(step)
        if step in SAMPLED_STEPS:
            self.sampled[str(step)] = list(offsets)
            self.sampled_flips[str(step)] = list(flips)
        if self.verbatim:
            self.write({"kind": "offsets", "rank": self.rank, "step": step, "n": len(offsets),
                        "offsets": list(offsets), "flips": list(flips), "t": t})
        elif step in (100, 299) or step % 1000 == 0:
            self.write({"kind": "tick", "rank": self.rank, "step": step, "t": t})

    def record_loss(self, step, loss):
        """Device-side only: no ``.item()`` / ``.cpu()`` / ``float()`` on the per-step path."""
        import torch
        self.counts["training_step"] += 1
        ok = torch.isfinite(loss.detach()).all()
        self.finite = ok if self.finite is None else (self.finite & ok)
        if step in SAMPLED_STEPS:
            self.losses_sampled[str(step)] = loss.detach().clone()

    def summary(self, extra):
        steps_ok = self.steps == list(range(len(self.steps)))
        finite = None if self.finite is None else bool(self.finite.item())       # ONE transfer
        sampled_losses = {k: float(v.float().cpu()).hex() for k, v in self.losses_sampled.items()}
        payload = {"rank": self.rank, "counts": dict(self.counts),
                   "stream_digest": self.stream_hash.hexdigest(), "all_losses_finite": finite,
                   "offset_histogram": list(self.hist), "n_offsets": self.n_offsets,
                   "n_steps": len(self.steps), "steps_contiguous_from_0": steps_ok,
                   "sampled_offsets": self.sampled, "sampled_flips": self.sampled_flips,
                   "flips_applied": self.flips_applied, "sampled_losses": sampled_losses, **extra}
        path = os.path.join(self.dir, f"rank{self.rank}.summary.json")
        tmp = f"{path}.tmp-{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(payload, f, sort_keys=True, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)


def _tensor_bytes(t):
    t = t.detach().cpu().contiguous()
    if t.numel() == 0:
        return b""
    import torch
    return t.reshape(-1).view(torch.uint8).numpy().tobytes()


def state_signature(module):
    h = hashlib.sha256()
    for name, t in sorted(module.state_dict().items()):
        h.update(f"{name}|{t.dtype}|{list(t.shape)}|".encode())
        h.update(_tensor_bytes(t))
    return h.hexdigest()


def param_signature(module):
    h = hashlib.sha256()
    for name, p in module.named_parameters():
        h.update(f"{name}|{p.dtype}|{list(p.shape)}|".encode())
        h.update(_tensor_bytes(p))
    return h.hexdigest()


def rng_signature(module):
    import numpy as np
    import torch
    out = {"python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest()}
    st = np.random.get_state()
    out["numpy"] = hashlib.sha256(np.asarray(st[1], dtype=np.uint32).tobytes()
                                  + repr((st[0],) + tuple(st[2:])).encode()).hexdigest()
    out["torch_cpu"] = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    if torch.cuda.is_available():
        out["torch_cuda"] = hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()
    sob = getattr(module, "rng", None)
    if sob is not None:
        out["sobol"] = hashlib.sha256(b"".join(_tensor_bytes(getattr(sob, a))
                                               for a in ("sobolstate", "shift", "quasi"))
                                      + repr(sob.num_generated).encode()).hexdigest()
    return out


def checkpoint_owner_class(function):
    """Canonical owner of a checkpointed callable (plan §14 H6): ``functools.partial`` is
    unwrapped to ``.func``; a bound method is owned by its instance (FLAC's DINOv3 layer
    wrapper checkpoints the layer's original bound ``forward``); a callable object (the DiT
    checkpoints its ``TransformerBlock`` modules) is owned by itself; a plain function by its
    module. Returns ``dinov3`` | ``dit`` | ``unclassified``."""
    import functools
    import types
    f = function
    while isinstance(f, functools.partial):
        f = f.func
    owner = getattr(f, "__self__", None)
    if owner is not None and not isinstance(owner, types.ModuleType):
        mod = type(owner).__module__
    elif isinstance(f, (types.FunctionType, types.BuiltinFunctionType)):
        mod = getattr(f, "__module__", "") or ""
    else:
        mod = type(f).__module__
    if mod.startswith("transformers.models.dinov3_vit."):
        return "dinov3"
    if mod == "src.models.transformer":
        return "dit"
    return "unclassified"


def install_publication(ev, mode):
    """B2 (plan §14): right after each synchronous checkpoint save, rank 0 marks it and
    publishes it to the NAS (``nas``); the outcome is broadcast and EVERY rank raises if it
    failed, so training never continues past an unacknowledged checkpoint. ``off`` (smokes)
    installs nothing."""
    if mode == "off":
        return
    import pytorch_lightning as pl
    sys.path.insert(0, HERE)
    import exp43_kit as kit
    if mode != "nas":
        _fail(f"publication mode {mode!r} is not nas / off")
    run_id = os.environ.get("EXP43_RUN_ID", "")
    orig = pl.callbacks.ModelCheckpoint._save_checkpoint

    def _save_checkpoint(self, trainer, filepath):
        orig(self, trainer, filepath)
        ok, why = True, ""
        if trainer.is_global_zero:
            try:
                t0 = time.time()
                kit.mark_saved(filepath)
                kit.publish_saved(filepath, run_id, timeout_s=1800)
                ev.counts["checkpoints_published"] += 1
                ev.write({"kind": "publish", "rank": ev.rank, "path": os.path.basename(filepath),
                          "seconds": time.time() - t0})
            except Exception as err:          # noqa: BLE001 -- any failure stops the run
                ok, why = False, f"{type(err).__name__}: {err}"
        ok, why = trainer.strategy.broadcast((ok, why), src=0)
        if not ok:
            raise RuntimeError(f"exp43: checkpoint {filepath} was not acknowledged: {why}")
    pl.callbacks.ModelCheckpoint._save_checkpoint = _save_checkpoint


def recompute_step(wrapper, n):
    """The PRIVATE per-step recomputation of what ``_apply_yaw_aug`` must apply (plan §2.3): a
    FRESH generator seeded by ``(yaw_aug_seed, global_step, global_rank)``, the ``n`` offsets
    first (the exp_15 draw), then ``n`` flip bits by ``torch.randint(0, 2, (n,))``. Touches no
    global stream."""
    import torch
    import src.training.diffusion as tdiff
    from src.data.yaw_rotation import draw_yaw_offsets
    g = torch.Generator()
    g.manual_seed(tdiff._yaw_aug_step_seed(wrapper.yaw_aug_seed, int(wrapper.global_step),
                                           int(wrapper.global_rank)))
    offsets = draw_yaw_offsets(n, wrapper.yaw_aug_img_w, g).tolist()
    return offsets, torch.randint(0, 2, (n,), generator=g, dtype=torch.long).tolist()


def install(ev):
    import torch
    import torch.utils.checkpoint as tcp
    import src.models.transformer as transformer
    import src.training.diffusion as tdiff
    import src.data.yaw_mirror as yaw_mirror
    from src.data.yaw_rotation import yaw_column_shift

    # the mirror spy must observe THE function the wrapper calls: refuse before patching anything
    orig_mirror = getattr(tdiff, "mirror_scene_metadata", None)
    if orig_mirror is None or orig_mirror is not yaw_mirror.mirror_scene_metadata:
        _fail("src.training.diffusion.mirror_scene_metadata is not src.data.yaw_mirror's: the mirror "
              "spy would not observe the applied flips")

    orig_flash = transformer.flash_attn_func
    if orig_flash is not None:
        def counted_flash(*args, **kwargs):
            ev.counts["flash_calls"] += 1
            return orig_flash(*args, **kwargs)
        transformer.flash_attn_func = counted_flash

    orig_ckpt = tcp.checkpoint

    def counted_checkpoint(function, *args, use_reentrant=None, **kwargs):
        if use_reentrant is True:
            ev.counts["ckpt_reentrant"] += 1
        elif use_reentrant is None:
            ev.counts["ckpt_default"] += 1
        else:
            ev.counts[f"ckpt_{checkpoint_owner_class(function)}"] += 1
        return orig_ckpt(function, *args, use_reentrant=use_reentrant, **kwargs)
    tcp.checkpoint = counted_checkpoint

    orig_rotate = tdiff.rotate_scene_metadata

    def spied_rotate(md, alpha, img_w):
        out = orig_rotate(md, alpha, img_w)
        if ev.collecting is not None:
            ev.collecting.append(int(yaw_column_shift(alpha, img_w)))
            ev.rotated.append(out)
        return out
    tdiff.rotate_scene_metadata = spied_rotate

    def spied_mirror(md, *args, **kwargs):
        out = orig_mirror(md, *args, **kwargs)
        ev.counts["mirror_calls"] += 1
        hits = [] if ev.rotated is None else [i for i, x in enumerate(ev.rotated) if x is md]
        if len(hits) != 1 or hits[0] in ev.mirrored:
            # outside _apply_yaw_aug, not exactly one of this step's rotated samples, or twice
            ev.counts["flip_anomalies"] += 1
        else:
            ev.mirrored[hits[0]] = out
        return out
    tdiff.mirror_scene_metadata = spied_mirror

    cls = tdiff.DiffusionCondTrainingWrapper
    orig_apply = cls._apply_yaw_aug

    def apply_yaw_aug(self, metadata):
        ev.bind_rank(self.global_rank)
        ev.collecting, ev.rotated, ev.mirrored = [], [], {}
        try:
            out = orig_apply(self, metadata)
        finally:
            got, rotated, mirrored = ev.collecting, ev.rotated, ev.mirrored
            ev.collecting = ev.rotated = ev.mirrored = None
        n = len(metadata)
        flips = [1 if i in mirrored else 0 for i in range(n)]
        if not (isinstance(out, list) and len(out) == len(rotated) == len(got) == n
                and set(mirrored) <= set(range(n))
                and all(out[i] is (mirrored[i] if flips[i] else rotated[i]) for i in range(n))):
            ev.counts["flip_anomalies"] += 1      # the batch is not the rotated samples + the mirrored ones
        if (got, flips) != recompute_step(self, n):
            ev.counts["stream_mismatch_steps"] += 1
        ev.counts["apply_yaw_aug"] += 1
        ev.record_step(int(self.global_step), got, flips, time.time())
        return out
    cls._apply_yaw_aug = apply_yaw_aug

    orig_step = cls.training_step

    def training_step(self, batch, batch_idx):
        out = orig_step(self, batch, batch_idx)
        ev.bind_rank(self.global_rank)
        ev.record_loss(int(self.global_step), out["loss"] if isinstance(out, dict) else out)
        return out
    cls.training_step = training_step

    orig_fit_start = cls.on_fit_start

    def on_fit_start(self):
        result = orig_fit_start(self)
        ev.bind_rank(self.global_rank)
        dev = torch.cuda.current_device() if torch.cuda.is_available() else None
        props = torch.cuda.get_device_properties(dev) if dev is not None else None
        ev.write({
            "kind": "fit_start", "rank": int(self.global_rank),
            "world_size": int(self.trainer.world_size),
            "local_rank": int(getattr(self.trainer, "local_rank", 0)),
            "gpu_name": props.name if props is not None else None,
            "gpu_uuid": str(getattr(props, "uuid", "")) if props is not None else None,
            "syncbn_modules": sum(isinstance(m, torch.nn.SyncBatchNorm) for m in self.modules()),
            "batchnorm_modules": sum(isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                                     for m in self.modules()),
            "yaw_aug": {"enabled": self.yaw_aug_enabled, "img_w": self.yaw_aug_img_w,
                        "seed": self.yaw_aug_seed, "flip": self.yaw_aug_flip},
            "flash_attn_func_is_none": orig_flash is None,
            "state_signature": state_signature(self),
            "param_signature": param_signature(self),
            "rng_signature": rng_signature(self),
            "global_step": int(self.global_step),
            "numerics": {"matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                         "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                         "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                         "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                         "float32_matmul_precision": torch.get_float32_matmul_precision()},
        })
        return result
    cls.on_fit_start = on_fit_start

    orig_train_end = cls.on_train_end

    def on_train_end(self):
        result = orig_train_end(self)
        ev.bind_rank(self.global_rank)
        ev.summary({"global_step": int(self.global_step), "pid": os.getpid()})
        return result
    cls.on_train_end = on_train_end


def main():
    evdir = os.environ.get("EXP43_EVIDENCE_DIR", "")
    if not os.path.isabs(evdir) or not os.path.isdir(evdir):
        _fail(f"EXP43_EVIDENCE_DIR must be an existing absolute directory, got {evdir!r}")
    if not os.path.isfile(os.path.join(ROOT, "train.py")):
        _fail(f"no train.py at the run tree root {ROOT}")
    if os.path.realpath(os.getcwd()) != os.path.realpath(ROOT):
        _fail(f"cwd {os.getcwd()} is not the run tree root {ROOT}")
    mode = os.environ.get("EXP43_EVIDENCE_MODE", "")
    if mode not in ("verbatim", "digest"):
        _fail(f"EXP43_EVIDENCE_MODE must be verbatim or digest, got {mode!r}")
    publish = os.environ.get("EXP43_PUBLISH", "")
    if publish not in ("nas", "off"):
        _fail(f"EXP43_PUBLISH must be nas or off, got {publish!r}")
    sys.path.insert(0, ROOT)
    ev = Evidence(evdir, verbatim=(mode == "verbatim"))
    install(ev)
    install_publication(ev, publish)
    import train
    if os.path.realpath(train.__file__) != os.path.realpath(os.path.join(ROOT, "train.py")):
        _fail(f"train resolved to {train.__file__}, not the run tree's")
    train.main()


if __name__ == "__main__":
    main()
