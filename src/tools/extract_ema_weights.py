"""Write a plain, EMA-initialised weights file from a wrapped FLAC training checkpoint.

Purpose (exp_19, HAA finetuning). The released HAA recipe initialises from
``weights/FLAC/FLAC_EMA.ckpt`` — a *plain* weights file with **bare** model keys —
via ``train.py --pretrained-ckpt-path``. exp_19's three inits are instead 40k
PyTorch-Lightning *training* checkpoints (prefixed keys plus optimizer, scheduler,
EMA bookkeeping and a loss buffer). ``unwrap_model.py``, the upstream converter,
imports ``stable_audio_tools`` and does not run in this fork. This tool is the
replacement, and it is **copy-only**: the 40k artifacts underpin five closed
experiments and are never modified (enforced below; verified by sha in the tests).

What "the EMA weights" are (measured on the real artifacts, not assumed)
-----------------------------------------------------------------------
The training wrapper builds ``EMA(self.diffusion.model, ...)``
(``src/training/diffusion.py:277``), so the EMA shadows **the DiT only**. A 40k
checkpoint's ``state_dict`` holds::

    diffusion.model.*            210   live DiT
    diffusion.conditioner.*      561   DINOv3 (shared) + RIR encoder   -- no EMA copy
    diffusion.pretransform.*     295   VAE                             -- no EMA copy
    diffusion_ema.ema_model.*    210   EMA copy of the DiT
    diffusion_ema.initted/.step    2   EMA bookkeeping, not weights
    losses.losses.0.weight         1   loss module, not the model

The released ``FLAC_EMA.ckpt`` is therefore **not** 210 keys but 1066 bare ones
(``model`` 210 / ``conditioner`` 561 / ``pretransform`` 295): ``export_model``
(``src/training/diffusion.py:911``) assigns the EMA weights *into*
``diffusion.model`` and saves the whole wrapper. This tool reproduces exactly that
artifact:

    out["state_dict"][bare(k)] = state_dict["diffusion." + bare(k)]     # carried
    out["state_dict"]["model." + t] = state_dict["diffusion_ema.ema_model." + t]

An EMA-only file would fail ``train.py:148``'s
``model.load_state_dict(weights, strict=True)`` with 856 missing keys — pinned as
a test (``test_an_EMA_only_file_would_fail_the_strict_load``).

Fail-closed conditions (each exits 2 from the CLI)
--------------------------------------------------
* input missing, or ``--out`` naming the same file as ``--ckpt-path``;
* ``--out`` already exists (never overwritten: two arms differ only by their init,
  so a silent overwrite swaps an experiment);
* no ``diffusion_ema.ema_model.*`` keys — every exp_19 arm trained with
  ``use_ema: true``, so their absence means the WRONG FILE, never a reason to fall
  back to the online weights;
* the EMA key set does not mirror the live ``diffusion.model.*`` key set;
* any unrecognised top-level key family (only ``losses``/``discriminator`` are
  registered as droppable — train.py drops them too).

``torch.load`` runs with ``map_location="cpu"`` and ``weights_only=True``. All
three exp_19 inits load under it; a checkpoint that does not is refused rather
than un-pickled with arbitrary code execution.

⚠️ The printed sha256 is a function of the content **and of the output filename**:
``torch.save`` prefixes every zip entry with the output file's basename, so the
same tensors written to ``a.ckpt`` and ``b.ckpt`` hash differently. Pin the sha
together with the path it was produced at.

Usage
-----
    python -m src.tools.extract_ema_weights --ckpt-path <wrapped.ckpt> --out <init.ckpt>
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import torch

LIVE_PREFIX = "diffusion."
EMA_PREFIX = "diffusion_ema.ema_model."
EMA_TARGET_PREFIX = "model."
EMA_BOOKKEEPING = ("diffusion_ema.initted", "diffusion_ema.step")
# The families train.py itself discards (train.py:146-147). Keeping them would make
# the output's sha a promise about bytes nobody loads.
DROP_SUBSTRINGS = ("discriminator", "losses")

_CHUNK = 1 << 20


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _same_file(a: str, b: str) -> bool:
    """True when ``a`` and ``b`` name the same file (``./`` spellings, symlinks, hardlinks)."""
    if os.path.realpath(os.path.abspath(a)) == os.path.realpath(os.path.abspath(b)):
        return True
    try:
        return os.path.samefile(a, b)
    except OSError:          # one of them does not exist yet -> not the same file
        return False


def _detached(value):
    """A standalone copy of a tensor value.

    ``clone`` rather than the loaded object so the written file can never inherit
    a view into a larger storage (which ``torch.save`` would serialise whole) or a
    stray ``requires_grad``. Non-tensor entries pass through unchanged.
    """
    if torch.is_tensor(value):
        return value.detach().clone()
    return value


def extract_ema_weights(ckpt_path: str, out_path: str) -> dict:
    """Write ``out_path`` = the bare model weights of ``ckpt_path``, EMA-initialised.

    ``ckpt_path`` is opened read-only and never modified. Returns a JSON-safe
    summary dict (counts, dropped keys, both file shas) for the launcher's
    provenance record. See the module docstring for the refusal conditions.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"input checkpoint not found: {ckpt_path}")
    if _same_file(ckpt_path, out_path):
        raise ValueError(
            f"refusing to write onto the input: --out is the same file as --ckpt-path "
            f"({ckpt_path}). The 40k checkpoints are read-only evidence."
        )
    if os.path.exists(out_path):
        raise FileExistsError(
            f"output already exists and is never overwritten: {out_path} "
            "(two arms differ only by their init — delete it deliberately or pick "
            "another name)"
        )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        raise TypeError(f"not a Lightning checkpoint dict: {ckpt_path} (got {type(ckpt).__name__})")
    if "state_dict" not in ckpt:
        raise KeyError(
            f"{ckpt_path} has no 'state_dict' key — it is not a PyTorch-Lightning "
            "training checkpoint (an already-exported weights file?)"
        )
    state_dict = ckpt["state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError(f"'state_dict' is not a dict (got {type(state_dict).__name__})")

    live, ema, dropped, unknown = {}, {}, [], []
    for key, value in state_dict.items():
        if key.startswith(EMA_PREFIX):
            ema[key[len(EMA_PREFIX):]] = value
        elif key in EMA_BOOKKEEPING:
            dropped.append(key)                       # EMA counters, not weights
        elif key.startswith(LIVE_PREFIX):
            bare = key[len(LIVE_PREFIX):]
            if any(s in bare for s in DROP_SUBSTRINGS):
                dropped.append(key)
            else:
                live[bare] = value
        elif any(s in key for s in DROP_SUBSTRINGS):
            dropped.append(key)                       # e.g. top-level losses.losses.0.weight
        else:
            unknown.append(key)

    if unknown:
        raise ValueError(
            f"{ckpt_path} carries {len(unknown)} key(s) in no recognised family "
            f"(first: {sorted(unknown)[:3]}). They could be model weights; refusing "
            "to silently drop them."
        )
    if not ema:
        raise KeyError(
            f"{ckpt_path} has no '{EMA_PREFIX}*' keys. Every exp_19 arm trains with "
            "use_ema: true, so this is the wrong file — refusing to fall back to the "
            "online weights, which would init an arm the record calls EMA from "
            "non-EMA weights."
        )

    live_dit = {k[len(EMA_TARGET_PREFIX):] for k in live if k.startswith(EMA_TARGET_PREFIX)}
    skew = sorted(set(ema) ^ live_dit)
    if skew:
        raise ValueError(
            f"the EMA copy does not mirror the live DiT: {len(skew)} key(s) differ "
            f"(first: {skew[:3]}). This checkpoint was not produced by "
            "EMA(self.diffusion.model, ...) as assumed; the substitution would emit "
            "a state dict that is neither the live nor the EMA model."
        )

    out_sd = {k: _detached(v) for k, v in live.items()}
    for tail, value in ema.items():                   # EMA overwrites the DiT subtree
        out_sd[EMA_TARGET_PREFIX + tail] = _detached(value)

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save({"state_dict": out_sd}, out_path)

    summary = {
        "in_path": ckpt_path,
        "out_path": out_path,
        "in_sha256": _sha256_file(ckpt_path),
        "out_sha256": _sha256_file(out_path),
        "n_ema": len(ema),
        "n_carried": len(out_sd) - len(ema),
        "n_total": len(out_sd),
        "n_dropped": len(dropped),
        "dropped": sorted(dropped),
        "global_step": ckpt.get("global_step"),
        "epoch": ckpt.get("epoch"),
    }

    print("=== extract_ema_weights (EMA DiT + carried conditioner/VAE, bare keys) ===")
    print(f"  in : {summary['in_path']}  sha256={summary['in_sha256']}")
    print(f"       global_step={summary['global_step']} epoch={summary['epoch']} (unmodified)")
    print(f"  EXTRACTED from {EMA_PREFIX}*: {summary['n_ema']} tensors -> {EMA_TARGET_PREFIX}*")
    print(f"  CARRIED from {LIVE_PREFIX}* (no EMA copy exists): {summary['n_carried']} tensors")
    print(f"  DROPPED ({summary['n_dropped']}): {', '.join(summary['dropped']) or '-'}")
    print(f"  out: {summary['out_path']}  ({summary['n_total']} bare keys)")
    print(f"  out sha256: {summary['out_sha256']}")
    print("  (sha is path-dependent: torch.save prefixes zip entries with the basename)")
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt-path", required=True,
                    help="wrapped PL training checkpoint (read-only, never modified)")
    ap.add_argument("--out", required=True,
                    help="destination weights file for train.py --pretrained-ckpt-path")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        extract_ema_weights(args.ckpt_path, args.out)
    except (FileNotFoundError, FileExistsError, ValueError, KeyError, TypeError) as e:
        print(f"extract_ema_weights FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
