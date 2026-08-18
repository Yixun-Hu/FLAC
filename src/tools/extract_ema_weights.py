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
* ``--out`` already exists — checked up front AND enforced **atomically** at
  publication (r1 finding 3): the payload is written to a same-directory
  temporary file and published with ``os.link``, which fails rather than replaces
  if the destination appeared in the meantime. There is no window in which a
  concurrently created file can be clobbered, and a crash mid-write leaves a
  ``.tmp`` rather than a truncated init;
* no ``diffusion_ema.ema_model.*`` keys — every exp_19 arm trained with
  ``use_ema: true``, so their absence means the WRONG FILE, never a reason to fall
  back to the online weights;
* the EMA key set does not mirror the live ``diffusion.model.*`` key set, or any
  EMA tensor disagrees with its live counterpart in **shape, dtype or layout**
  (r1 finding 2): ``load_state_dict`` is key/shape-strict but silently CASTS the
  source dtype into the target, so a bf16 EMA entry would load without complaint
  and silently change the initialisation;
* any key outside the recognised namespaces. Dropping is namespace-scoped
  (``losses.*`` / ``discriminator.*`` as the FIRST path component), not substring
  matching, so a future ``conditioner.…losses_proj.weight`` is no longer discarded
  silently (r1 non-blocking finding);
* any surviving key that ``train.py``'s own substring transforms (lines 142-147)
  would rewrite or discard — such a key would be dropped on load and then fail the
  strict load with a confusing name, so it is refused HERE, where it is legible.

``torch.load`` runs with ``map_location="cpu"`` and ``weights_only=True``. All
three exp_19 inits load under it; a checkpoint that does not is refused rather
than un-pickled with arbitrary code execution.

The printed sha256 is **content-only**: the payload is serialised through an open
file object, so ``torch.save`` writes its fixed ``archive/`` zip prefix instead of
one derived from the output filename. Copying or renaming the file therefore does
not change its sha, and the launcher's manifest pins sha and path as two
independent facts (which content, and which arm it belongs to).

Usage
-----
    python -m src.tools.extract_ema_weights --ckpt-path <wrapped.ckpt> --out <init.ckpt>
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import os
import sys
import tempfile

import torch

LIVE_PREFIX = "diffusion."
EMA_PREFIX = "diffusion_ema.ema_model."
EMA_TARGET_PREFIX = "model."
EMA_BOOKKEEPING = ("diffusion_ema.initted", "diffusion_ema.step")
# The families train.py itself discards (train.py:146-147), matched as the FIRST
# path component. Substring matching would also swallow a legitimate weight whose
# name merely CONTAINS one of these words.
DROP_NAMESPACES = ("losses", "discriminator")
# The rewrites train.py applies on load (lines 142-147). A surviving key that
# collides with any of them cannot round-trip.
TRAIN_PY_REWRITES = ("diffusion.", "autoencoder.")
TRAIN_PY_DROPS = ("discriminator", "losses")

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


def _check_substitutable(tail: str, ema_v, live_v) -> None:
    """Refuse an EMA tensor that is not interchangeable with its live counterpart.

    ``load_state_dict`` validates keys and shapes and then CASTS the source into
    the target's dtype, so a dtype difference is exactly the failure that produces
    a silently wrong initialisation instead of an error. ``layout`` is checked for
    the same reason: a sparse/strided mismatch changes what was averaged.
    """
    key = EMA_TARGET_PREFIX + tail
    if not torch.is_tensor(ema_v) or not torch.is_tensor(live_v):
        raise ValueError(
            f"{key}: EMA and live entries must both be tensors, got "
            f"{type(ema_v).__name__} (EMA) and {type(live_v).__name__} (live)")
    if tuple(ema_v.shape) != tuple(live_v.shape):
        raise ValueError(
            f"{key}: EMA shape {tuple(ema_v.shape)} != live shape {tuple(live_v.shape)}; "
            "the EMA does not shadow this checkpoint's DiT")
    if ema_v.dtype != live_v.dtype:
        raise ValueError(
            f"{key}: EMA dtype {ema_v.dtype} != live dtype {live_v.dtype}; "
            "load_state_dict would CAST it silently and the arm would start from "
            "weights nobody chose")
    if ema_v.layout != live_v.layout:
        raise ValueError(
            f"{key}: EMA layout {ema_v.layout} != live layout {live_v.layout}")


def _survives_train_py(key: str) -> bool:
    """Would ``key`` reach ``model.load_state_dict`` unchanged? (train.py:142-147)"""
    return (not any(r in key for r in TRAIN_PY_REWRITES)
            and not any(d in key for d in TRAIN_PY_DROPS))


def _publish_exclusive(tmp_path: str, out_path: str) -> None:
    """Publish ``tmp_path`` as ``out_path``, atomically refusing to replace.

    ``os.link`` is the operation that carries the guarantee: it either creates the
    destination or fails with ``EEXIST``, with no window between the two. A
    check-then-``torch.save`` cannot promise that — a file created in between is
    overwritten (r1 finding 3). Deliberately no fallback for filesystems without
    hard links: silently degrading to a non-atomic copy would return the very
    window this function exists to close.
    """
    try:
        os.link(tmp_path, out_path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            raise FileExistsError(
                f"output appeared while this run was writing and is never "
                f"overwritten: {out_path} (nothing was published; the payload was "
                "discarded)") from e
        raise OSError(
            f"could not publish {out_path} atomically via os.link ({e}); refusing "
            "to fall back to a replacing write") from e


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
            f"({ckpt_path}). The 40k checkpoints are read-only evidence.")
    # Reported early because it is the legible message; the guarantee itself is
    # the exclusive publication at the end, which needs no ordering assumption.
    if os.path.exists(out_path):
        raise FileExistsError(
            f"output already exists and is never overwritten: {out_path} "
            "(two arms differ only by their init — delete it deliberately or pick "
            "another name)")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        raise TypeError(f"not a Lightning checkpoint dict: {ckpt_path} (got {type(ckpt).__name__})")
    if "state_dict" not in ckpt:
        raise KeyError(
            f"{ckpt_path} has no 'state_dict' key — it is not a PyTorch-Lightning "
            "training checkpoint (an already-exported weights file?)")
    state_dict = ckpt["state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError(f"'state_dict' is not a dict (got {type(state_dict).__name__})")

    def _namespace(key: str) -> str:
        return key.split(".", 1)[0]

    live, ema, dropped, unknown = {}, {}, [], []
    for key, value in state_dict.items():
        if key.startswith(EMA_PREFIX):
            ema[key[len(EMA_PREFIX):]] = value
        elif key in EMA_BOOKKEEPING:
            dropped.append(key)                       # EMA counters, not weights
        elif key.startswith(LIVE_PREFIX):
            bare = key[len(LIVE_PREFIX):]
            if _namespace(bare) in DROP_NAMESPACES:
                dropped.append(key)
            else:
                live[bare] = value
        elif _namespace(key) in DROP_NAMESPACES:
            dropped.append(key)                       # e.g. losses.losses.0.weight
        else:
            unknown.append(key)

    if unknown:
        raise ValueError(
            f"{ckpt_path} carries {len(unknown)} key(s) in no recognised family "
            f"(first: {sorted(unknown)[:3]}). They could be model weights; refusing "
            "to silently drop them.")
    if not ema:
        raise KeyError(
            f"{ckpt_path} has no '{EMA_PREFIX}*' keys. Every exp_19 arm trains with "
            "use_ema: true, so this is the wrong file — refusing to fall back to the "
            "online weights, which would init an arm the record calls EMA from "
            "non-EMA weights.")

    live_dit = {k[len(EMA_TARGET_PREFIX):] for k in live if k.startswith(EMA_TARGET_PREFIX)}
    skew = sorted(set(ema) ^ live_dit)
    if skew:
        raise ValueError(
            f"the EMA copy does not mirror the live DiT: {len(skew)} key(s) differ "
            f"(first: {skew[:3]}). This checkpoint was not produced by "
            "EMA(self.diffusion.model, ...) as assumed; the substitution would emit "
            "a state dict that is neither the live nor the EMA model.")

    out_sd = {k: _detached(v) for k, v in live.items()}
    for tail, value in ema.items():                   # EMA overwrites the DiT subtree
        _check_substitutable(tail, value, live[EMA_TARGET_PREFIX + tail])
        out_sd[EMA_TARGET_PREFIX + tail] = _detached(value)

    mangled = sorted(k for k in out_sd if not _survives_train_py(k))
    if mangled:
        raise ValueError(
            f"{len(mangled)} extracted key(s) would be rewritten or dropped by "
            f"train.py's own load transforms (lines 142-147) — first: {mangled[:3]}. "
            "Such a key never reaches load_state_dict, so the strict load would "
            "fail later with a confusing name; refusing here instead.")

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # Same directory, so the publication below is a link within one filesystem.
    fd, tmp_path = tempfile.mkstemp(dir=out_dir or ".", prefix=".extract_ema_", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            # Through the FILE OBJECT, not the path: torch.save then writes its
            # fixed 'archive/' zip prefix instead of one derived from the
            # (random) temp filename, so the published sha is content-only and
            # identical to what a direct write of the same tensors would give.
            torch.save({"state_dict": out_sd}, f)
            f.flush()
            os.fsync(f.fileno())
        _publish_exclusive(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

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
    print(f"  EXTRACTED from {EMA_PREFIX}*: {summary['n_ema']} tensors -> {EMA_TARGET_PREFIX}*"
          " (shape/dtype/layout checked against the live DiT)")
    print(f"  CARRIED from {LIVE_PREFIX}* (no EMA copy exists): {summary['n_carried']} tensors")
    print(f"  DROPPED ({summary['n_dropped']}): {', '.join(summary['dropped']) or '-'}")
    print(f"  out: {summary['out_path']}  ({summary['n_total']} bare keys, published via os.link)")
    print(f"  out sha256: {summary['out_sha256']}")
    print("  (content-only sha: serialised through a file object, so it does not "
          "depend on the output filename)")
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
    except (FileNotFoundError, FileExistsError, ValueError, KeyError, TypeError, OSError) as e:
        print(f"extract_ema_weights FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
