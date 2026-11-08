"""Observe-only run evidence for the first optimizer step (exp_38, plan_data_efficiency.md r2 §10 B3).

Enabled ONLY by ``train.py --evidence-json <path>`` (default ``''`` -> no callback at all). On every
rank it records, at fit start, what the run is about to train (a digest of the full wrapper
``state_dict``, the timestep Sobol engine state, Python / NumPy / torch RNG digests, SyncBN vs
BatchNorm module counts, non-reentrant ViT checkpoint wrappers, whether flash-attention is wired
into ``src.models.transformer``), then — for the FIRST optimizer step only — counts, through
pass-through wrappers, flash-attention and SDPA calls in ``src.models.transformer``,
``torch.utils.checkpoint.checkpoint`` calls (with ``use_reentrant``) and the conditioner's call
shapes ``(len(batch_metadata), only_ids)``. After step 1 it writes ``<path>.rank<r>.json`` and
removes every wrapper. Nothing here draws a random number, moves a parameter or changes a value.
"""
import functools
import hashlib
import json
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch

import src.models.transformer as _T


def _tensor_bytes(t):
    t = t.detach().to("cpu").contiguous().reshape(-1)
    return t.view(torch.uint8).numpy().tobytes() if t.numel() else b""


def state_dict_digest(sd):
    """sha256 over (key, dtype, shape, raw bytes) in sorted-key order."""
    h = hashlib.sha256()
    for key in sorted(sd):
        v = sd[key]
        h.update(key.encode())
        if torch.is_tensor(v):
            h.update(f"|{v.dtype}|{tuple(v.shape)}|".encode())
            h.update(_tensor_bytes(v))
        else:
            h.update(f"|{type(v).__name__}|{v!r}|".encode())
    return h.hexdigest()


def rng_digests():
    out = {"python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest()}
    st = np.random.get_state()
    out["numpy"] = hashlib.sha256(st[1].tobytes() + repr(st[2:]).encode()).hexdigest()
    out["torch_cpu"] = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        out["torch_cuda"] = hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()
    return out


def sobol_state(engine):
    if not isinstance(engine, torch.quasirandom.SobolEngine):
        return None
    h = hashlib.sha256(_tensor_bytes(engine.quasi) + _tensor_bytes(engine.shift) + _tensor_bytes(engine.sobolstate))
    return {"dimension": engine.dimension, "num_generated": int(engine.num_generated),
            "scramble": bool(engine.scramble), "state_sha256": h.hexdigest()}


class _CountingF:
    """Stands in for ``torch.nn.functional`` inside src.models.transformer only: counts
    ``scaled_dot_product_attention`` and forwards every attribute unchanged."""
    def __init__(self, real, counter):
        self._real, self._counter = real, counter

    def scaled_dot_product_attention(self, *a, **k):
        self._counter["sdpa_calls"] += 1
        return self._real.scaled_dot_product_attention(*a, **k)

    def __getattr__(self, name):
        return getattr(self._real, name)


class RunEvidenceCallback(pl.Callback):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self._rec = None
        self._restore = []
        self._done = False

    # ---------------------------------------------------------------- wrappers
    def _install(self, pl_module):
        rec = self._rec
        rec.update({"flash_calls": 0, "sdpa_calls": 0, "checkpoint_calls": [], "conditioner_calls": []})
        if _T.flash_attn_func is not None:
            orig = _T.flash_attn_func

            @functools.wraps(orig)
            def flash(*a, **k):
                rec["flash_calls"] += 1
                return orig(*a, **k)
            _T.flash_attn_func = flash
            self._restore.append(lambda: setattr(_T, "flash_attn_func", orig))
        real_f = _T.F
        _T.F = _CountingF(real_f, rec)
        self._restore.append(lambda: setattr(_T, "F", real_f))
        orig_ckpt = torch.utils.checkpoint.checkpoint

        @functools.wraps(orig_ckpt)
        def ckpt(function, *a, **k):
            rec["checkpoint_calls"].append({"use_reentrant": k.get("use_reentrant", "UNSET")})
            return orig_ckpt(function, *a, **k)
        torch.utils.checkpoint.checkpoint = ckpt
        self._restore.append(lambda: setattr(torch.utils.checkpoint, "checkpoint", orig_ckpt))
        cond = getattr(getattr(pl_module, "diffusion", None), "conditioner", None)
        if cond is not None:
            cls_forward = cond.forward

            def cforward(batch_metadata, device, only_ids=None):
                rec["conditioner_calls"].append([len(batch_metadata), None if only_ids is None else list(only_ids)])
                return cls_forward(batch_metadata, device, only_ids=only_ids)
            cond.forward = cforward
            self._restore.append(lambda: vars(cond).pop("forward", None))

    def _uninstall(self):
        while self._restore:
            self._restore.pop()()

    # ---------------------------------------------------------------- hooks
    def on_fit_start(self, trainer, pl_module):
        mods = list(pl_module.modules())
        bn = torch.nn.modules.batchnorm._BatchNorm
        self._rec = {
            "rank": int(trainer.global_rank), "local_rank": int(getattr(trainer, "local_rank", 0)),
            "world_size": int(trainer.world_size), "num_nodes": int(getattr(trainer, "num_nodes", 1)),
            "precision": str(getattr(trainer, "precision", "")),
            "accumulate_grad_batches": getattr(trainer, "accumulate_grad_batches", None),
            "max_steps": getattr(trainer, "max_steps", None),
            "gradient_clip_val": getattr(trainer, "gradient_clip_val", None),
            "strategy": type(getattr(trainer, "strategy", None)).__name__,
            "global_step_at_fit_start": int(trainer.global_step),
            "state_dict_sha256": state_dict_digest(pl_module.state_dict()),
            "n_state_tensors": len(pl_module.state_dict()),
            "sobol": sobol_state(getattr(pl_module, "rng", None)),
            "rng_at_fit_start": rng_digests(),
            "syncbn_modules": sum(isinstance(m, torch.nn.SyncBatchNorm) for m in mods),
            "batchnorm_nonsync_modules": sum(isinstance(m, bn) and not isinstance(m, torch.nn.SyncBatchNorm) for m in mods),
            # src.models.conditioners._make_vit_ckpt_forward: functools.wraps copies __name__, the
            # code object keeps its own name
            "vit_ckpt_wrapped_layers": sum(getattr(getattr(vars(m).get("forward"), "__code__", None), "co_name", "")
                                           == "ckpt_forward" for m in mods),
            "flash_attn_func_set": _T.flash_attn_func is not None,
        }
        self._install(pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._done or self._rec is None or trainer.global_step < 1:
            return
        self._uninstall()
        rec = self._rec
        reals = batch[0] if isinstance(batch, (list, tuple)) and batch else None
        rec["micro_batch"] = int(reals.shape[0]) if torch.is_tensor(reals) else None
        rec["global_step_recorded"] = int(trainer.global_step)
        out = f"{self.path}.rank{rec['rank']}.json"
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        tmp = out + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(rec, fh, indent=1, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, out)
        print(f"EXP38 RUN EVIDENCE rank {rec['rank']} written: {out}", flush=True)
        self._done = True

    def teardown(self, trainer, pl_module, stage):
        self._uninstall()
