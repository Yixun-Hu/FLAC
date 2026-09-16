"""The per-run training contract of exp_14 (plan §4 "Round D", finding r2-2).

The six data-curve runs differ only in which split file they read, and their checkpoints
all land on one NAS directory tree. A `.ckpt` therefore has to carry, in itself, the
evidence of which run wrote it: this module builds that evidence (`build_contract`),
embeds a deep copy of it in every checkpoint Lightning saves (`RunContractCallback`), and
re-checks it before a checkpoint is trusted (`validate_checkpoint`).

Two hashes, two jobs, never interchanged (finding r3-2):

* ``file_sha256(path)`` pins the *bytes of a file* the run reads (dataset config, split
  JSON). Nothing embeds those files, so their identity can only be a byte sha.
* ``canonical_digest(obj)`` pins a *parsed object*. ``ModelConfigEmbedderCallback``
  embeds the parsed model-config dict in the checkpoint, so its identity must be computed
  from a dict on both sides -- at launch from the arm's JSON file and at validation from
  the embedded dict -- and must not change when the source file is reformatted.
"""
import datetime
import hashlib
import json

CONTRACT_VERSION = 1


class CheckpointContractError(ValueError):
    """A checkpoint does not match the contract of the run it claims to belong to."""


def canonical_digest(obj):
    """sha256 over the canonical JSON serialisation of ``obj``.

    Canonical = sorted keys, no whitespace, UTF-8 text (``ensure_ascii=False``). Both
    sides of a comparison must use exactly this definition, so it lives here and nowhere
    else.
    """
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path):
    """sha256 over the bytes of the file at ``path`` (streamed; configs stay small)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fin:
        for chunk in iter(lambda: fin.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bool(value):
    """Coerce a flag to a genuine ``bool`` (prefigure yields the *string* "false"/"true"
    for the lowercase ini literal -- see ``train._as_bool`` and its tests). Duplicated
    rather than imported so this module never imports the repo-root ``train``."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "1", "yes", "on"):
            return True
        if text in ("false", "0", "no", "off", ""):
            return False
        raise ValueError(f"cannot interpret sync_batchnorm={value!r} as a boolean")
    raise TypeError(f"sync_batchnorm must be bool or str, got {type(value).__name__}")


def _utc_now():
    """Launch timestamp: second-resolution ISO-8601 in UTC (e.g. 2026-09-16T18:00:00+00:00)."""
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


def build_contract(run_id, fraction, dataset_config_path, split_json_path,
                   model_config_path, seed, micro_batch, num_gpus, accum_batches,
                   sync_batchnorm, flac_sha, package_sha, launched_at=None):
    """Assemble the contract of one run from the files and settings it is launched with.

    Reads the three pinned files (so a missing one is a fail-closed ``OSError`` at launch,
    not a silent hole in the contract) and normalises every field to a JSON-stable
    primitive: the contract is persisted as ``run_contract.json``, re-read verbatim on
    every resume and compared with ``==`` against the copy embedded in a checkpoint, so a
    value that does not survive ``json.dumps``/``json.loads`` unchanged would break
    resume continuity. ``launched_at`` defaults to now (UTC).
    """
    with open(model_config_path) as fin:
        model_config = json.load(fin)
    return {
        "run_id": str(run_id),
        "fraction": float(fraction),
        "dataset_config_sha256": file_sha256(dataset_config_path),
        "split_sha256": file_sha256(split_json_path),
        "model_config_digest": canonical_digest(model_config),
        "model_config_path": str(model_config_path),
        "seed": int(seed),
        "micro_batch": int(micro_batch),
        "num_gpus": int(num_gpus),
        "accum_batches": int(accum_batches),
        "sync_batchnorm": _as_bool(sync_batchnorm),
        "flac_sha": str(flac_sha),
        "package_sha": str(package_sha),
        "launched_at": _utc_now() if launched_at is None else str(launched_at),
        "contract_version": CONTRACT_VERSION,
    }
