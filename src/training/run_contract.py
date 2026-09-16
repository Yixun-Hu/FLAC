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
import copy
import datetime
import hashlib
import json

import pytorch_lightning as pl
import torch

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


class RunContractCallback(pl.Callback):
    """Embed the run's contract in every checkpoint Lightning writes.

    Mirrors ``train.ModelConfigEmbedderCallback`` (which writes ``checkpoint["model_config"]``
    the same way); the two together make a checkpoint self-identifying: *this* model config,
    trained by *this* run on *this* split with *this* code.

    The contract is deep-copied on the way in and on the way out, so a checkpoint never
    aliases the launcher's live dict and a later mutation of either side cannot rewrite
    what an already-saved (or a future) checkpoint claims.
    """

    def __init__(self, contract):
        if not isinstance(contract, dict):
            raise TypeError(
                "RunContractCallback needs the parsed contract dict, got "
                f"{type(contract).__name__} -- load run_contract.json before constructing it"
            )
        self.contract = copy.deepcopy(contract)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["run_contract"] = copy.deepcopy(self.contract)


_MISSING = object()


def validate_checkpoint(path, expected_contract, expected_model_config, expect_step,
                        for_resume=False):
    """Fail-closed gate: is the file at ``path`` the checkpoint this run promised?

    Returns the contract loaded from the checkpoint; raises ``CheckpointContractError``
    with a precise message otherwise. The checks, in order:

    1. the file loads at all (a partial ``.ckpt`` is a violation, never a traceback the
       launcher has to interpret -- plan §4 Round D "a partial .ckpt never counts");
    2. ``global_step == expect_step`` (the boundary the launcher is waiting for);
    3. the embedded ``model_config`` (written by ``train.ModelConfigEmbedderCallback``)
       equals ``expected_model_config`` *as parsed dicts*;
    4. its ``canonical_digest`` equals the contract's ``model_config_digest`` -- an
       independent check, so a contract carrying another arm's digest is caught even when
       the embedded config matches;
    5. the embedded ``run_contract`` equals ``expected_contract`` **exactly** (wrong
       fraction / wrong run / wrong split sha => reject);
    6. when ``for_resume``: non-empty ``optimizer_states[0]["state"]`` and a non-empty
       ``lr_schedulers``. Resuming without them silently restarts Adam at the step-0
       warmup lr and rebuilds the schedule (CLAUDE.md "Checkpoint surgery on warm
       resume"), which would be invisible in the loss curve.
    """
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as err:
        raise CheckpointContractError(
            f"checkpoint {path} cannot be loaded ({type(err).__name__}: {err})"
        ) from None
    if not isinstance(checkpoint, dict):
        raise CheckpointContractError(
            f"checkpoint {path} is not a Lightning checkpoint: it loads as "
            f"{type(checkpoint).__name__}"
        )

    step = checkpoint.get("global_step", _MISSING)
    if step != expect_step:
        found = "absent" if step is _MISSING else repr(step)
        raise CheckpointContractError(
            f"checkpoint {path} is at global_step {found}, expected {expect_step}"
        )

    if "model_config" not in checkpoint:
        raise CheckpointContractError(
            f"checkpoint {path} embeds no model_config (ModelConfigEmbedderCallback was "
            "not attached to the run that wrote it)"
        )
    embedded_config = checkpoint["model_config"]
    if embedded_config != expected_model_config:
        raise CheckpointContractError(
            f"checkpoint {path} embeds a different model config than the expected one "
            "(compared as parsed dicts)"
        )
    if "model_config_digest" not in expected_contract:
        raise CheckpointContractError(
            "the expected contract has no model_config_digest -- it was not built by "
            "build_contract"
        )
    digest = canonical_digest(embedded_config)
    if digest != expected_contract["model_config_digest"]:
        raise CheckpointContractError(
            f"checkpoint {path} embeds a model config whose canonical digest {digest} is "
            f"not the contract's model_config_digest {expected_contract['model_config_digest']}"
        )

    if "run_contract" not in checkpoint:
        raise CheckpointContractError(
            f"checkpoint {path} is a legacy checkpoint without run_contract: it cannot be "
            "attributed to a run"
        )
    contract = checkpoint["run_contract"]
    if not isinstance(contract, dict):
        raise CheckpointContractError(
            f"checkpoint {path} carries a run_contract of type {type(contract).__name__}, "
            "not a dict"
        )
    if contract != expected_contract:
        differing = sorted(
            key for key in set(contract) | set(expected_contract)
            if contract.get(key, _MISSING) != expected_contract.get(key, _MISSING)
        )
        raise CheckpointContractError(
            f"checkpoint {path} belongs to a different run: run_contract differs in "
            f"{differing} (checkpoint run_id={contract.get('run_id')!r}, expected "
            f"run_id={expected_contract.get('run_id')!r})"
        )

    if for_resume:
        optimizer_states = checkpoint.get("optimizer_states") or []
        if not optimizer_states or not optimizer_states[0].get("state"):
            raise CheckpointContractError(
                f"checkpoint {path} cannot be resumed from: its optimizer state is empty, "
                "so the resumed run would restart Adam from scratch at the step-0 warmup lr"
            )
        if not checkpoint.get("lr_schedulers"):
            raise CheckpointContractError(
                f"checkpoint {path} cannot be resumed from: it carries no lr_scheduler "
                "state, so the schedule would be rebuilt from the config"
            )
    return contract
