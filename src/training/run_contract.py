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
import argparse
import copy
import datetime
import hashlib
import json
import os
import sys

import pytorch_lightning as pl
import torch

CONTRACT_VERSION = 1
CONTRACT_FILENAME = "run_contract.json"
RESUME_LOG_FILENAME = "resume_log.json"

_MISSING = object()

#: The fields that ARE the run's identity: if any of them changes, the persisted contract
#: describes a different run and must never be inherited. Only two contract keys are left
#: out -- ``launched_at`` (precisely what a resume inherits) and ``model_config_path``,
#: which is INFORMATIONAL only: the config's identity is its content
#: (``model_config_digest``), and the same config is legitimately re-read from another
#: absolute path (another checkout, a NAS copy) on a later launch.
IDENTITY_FIELDS = (
    "run_id",
    "fraction",
    "dataset_config_sha256",
    "split_sha256",
    "model_config_digest",
    "seed",
    "micro_batch",
    "num_gpus",
    "accum_batches",
    "sync_batchnorm",
    "flac_sha",
    "package_sha",
    "contract_version",
)

#: Every key a contract must carry (identity + the two non-identity keys).
CONTRACT_FIELDS = IDENTITY_FIELDS + ("model_config_path", "launched_at")


class CheckpointContractError(ValueError):
    """A checkpoint does not match the contract of the run it claims to belong to."""


class ContractMismatchError(ValueError):
    """A run dir already holds a contract describing a *different* run than this launch."""


class ContractSchemaError(ValueError):
    """A JSON document that has to be a contract (or a model config) is not one."""


def canonical_digest(obj):
    """sha256 over the canonical JSON serialisation of ``obj``.

    The serialisation is a **frozen literal** -- ``json.dumps(obj, sort_keys=True,
    separators=(",", ":"))``, stdlib defaults for everything else (``ensure_ascii`` left
    at True, so non-ASCII becomes a ``\\uXXXX`` escape) -- encoded as UTF-8. Any future
    argument added to that call silently invalidates every digest ever recorded, so both
    sides of a comparison use this one function and nothing re-implements it.
    """
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path):
    """sha256 over the bytes of the file at ``path`` (streamed; configs stay small)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fin:
        for chunk in iter(lambda: fin.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path, what):
    """Read a JSON file that must hold an object; every failure is a ContractSchemaError.

    Missing file, invalid JSON and a non-object root are all *input* errors (the CLI maps
    them to exit 2), never checkpoint verdicts and never uncaught exceptions.
    """
    try:
        with open(path) as fin:
            parsed = json.load(fin)
    except OSError as err:
        raise ContractSchemaError(f"{what} {path} cannot be read: {err}") from None
    except ValueError as err:
        raise ContractSchemaError(f"{what} {path} is not valid JSON: {err}") from None
    if not isinstance(parsed, dict):
        raise ContractSchemaError(
            f"{what} {path} is not a JSON object: it holds a {type(parsed).__name__}"
        )
    return parsed


def require_contract_fields(contract, path):
    """Every CONTRACT_FIELDS key must be present before a contract is used to judge a
    checkpoint -- a truncated sidecar is a launcher bug, not a checkpoint verdict."""
    missing = [field for field in CONTRACT_FIELDS if field not in contract]
    if missing:
        raise ContractSchemaError(f"contract {path} is missing required field(s): {missing}")
    return contract


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

    Reads the three pinned files, so a missing or unusable one is fail-closed at launch
    rather than a silent hole in the contract (``OSError`` from the two byte-sha'd files,
    ``ContractSchemaError`` for a model config that is missing, malformed or not a JSON
    object -- the CLI maps both to exit 2). Normalises every field to a JSON-stable
    primitive: the contract is persisted as ``run_contract.json``, re-read verbatim on
    every resume and compared with ``==`` against the copy embedded in a checkpoint, so a
    value that does not survive ``json.dumps``/``json.loads`` unchanged would break
    resume continuity. ``launched_at`` defaults to now (UTC).
    """
    model_config = load_json_object(model_config_path, "model config")
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
        _require_resume_state(path, checkpoint)
    return contract


def _type_name(value):
    return "absent" if value is _MISSING else type(value).__name__


def _require_resume_state(path, checkpoint):
    """The resume half of the gate: the state containers are *type-checked*, not duck-typed.

    A malformed checkpoint must come out as a contract verdict, never as an AttributeError
    or a TypeError escaping into the launcher (codex MEDIUM 3): ``optimizer_states`` has to
    be a non-empty list of dicts whose first entry carries a non-empty dict ``state``, and
    ``lr_schedulers`` a non-empty list of dicts. Resuming without real optimizer state
    silently restarts Adam at the step-0 warmup lr and rebuilds the schedule from the
    config (CLAUDE.md "Checkpoint surgery on warm resume") -- invisible in the loss curve.
    """
    optimizer_states = checkpoint.get("optimizer_states", _MISSING)
    if not isinstance(optimizer_states, list) or not optimizer_states:
        raise CheckpointContractError(
            f"checkpoint {path} cannot be resumed from: optimizer_states is "
            f"{_type_name(optimizer_states)}, expected a non-empty list"
        )
    first = optimizer_states[0]
    if not isinstance(first, dict):
        raise CheckpointContractError(
            f"checkpoint {path} cannot be resumed from: optimizer_states[0] is a "
            f"{_type_name(first)}, expected a dict"
        )
    state = first.get("state", _MISSING)
    if not isinstance(state, dict) or not state:
        raise CheckpointContractError(
            f"checkpoint {path} cannot be resumed from: its optimizer state is "
            f"{'empty' if isinstance(state, dict) else _type_name(state)}, so the resumed "
            "run would restart Adam from scratch at the step-0 warmup lr"
        )

    schedulers = checkpoint.get("lr_schedulers", _MISSING)
    if not isinstance(schedulers, list) or not schedulers:
        raise CheckpointContractError(
            f"checkpoint {path} cannot be resumed from: lr_schedulers is "
            f"{_type_name(schedulers)}, expected a non-empty list -- the schedule would be "
            "rebuilt from the config"
        )
    for index, entry in enumerate(schedulers):
        if not isinstance(entry, dict):
            raise CheckpointContractError(
                f"checkpoint {path} cannot be resumed from: lr_schedulers[{index}] is a "
                f"{_type_name(entry)}, expected a dict"
            )


def _write_json_atomically(path, obj):
    """Write ``obj`` as indent=1 JSON, replacing ``path`` atomically (never a half file)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as fout:
        json.dump(obj, fout, indent=1)
        fout.write("\n")
    os.replace(temporary, path)


def load_or_create_contract(run_dir, build_fn):
    """The contract of a run is created **once** and re-read ever after -- but only for the
    same run.

    ``build_fn(launched_at)`` must return a freshly built contract using the launch's
    CURRENT inputs, stamped with the ``launched_at`` it is handed (``None`` => its own).

    First launch: ``build_fn(launched_at=None)`` is persisted as
    ``<run_dir>/run_contract.json`` and its JSON round trip returned, so the in-memory
    contract can never differ from the one a later launch re-reads.

    Every later launch (i.e. every resume) rebuilds a candidate with the *persisted*
    ``launched_at`` and compares every field in ``IDENTITY_FIELDS``. All equal => the
    persisted contract is returned **verbatim**, so checkpoints saved after a resume carry
    the contract of the original launch (a rebuilt one would have a new ``launched_at`` and
    would match no earlier checkpoint). Any difference => ``ContractMismatchError`` and the
    sidecar is left **untouched**: relaunching into an existing run dir with a different
    split, config, seed or batch geometry is a new run, and silently inheriting the old
    contract would make every checkpoint claim a run that never happened (codex BLOCKING 1).

    Resumes are recorded by ``append_resume_log``, never inside the contract.
    """
    path = os.path.join(run_dir, CONTRACT_FILENAME)
    if not os.path.exists(path):
        _write_json_atomically(path, build_fn(launched_at=None))
        with open(path) as fin:
            return json.load(fin)

    with open(path) as fin:
        persisted = json.load(fin)
    if not isinstance(persisted, dict):
        raise ContractMismatchError(
            f"{path} is not a contract object: it holds a {type(persisted).__name__}"
        )
    # A persisted sidecar is judged complete BEFORE it is used (codex M5): a truncated one
    # carrying every identity field but, say, no `launched_at` used to be inherited and then
    # read at `contract["launched_at"]` -- an uncaught KeyError, i.e. exit 1, which a
    # launcher cannot classify. ContractSchemaError maps to exit 2 and leaves the file
    # untouched; repairing it is a human decision, never this function's.
    require_contract_fields(persisted, path)
    candidate = build_fn(launched_at=persisted.get("launched_at"))
    differing = [
        field for field in IDENTITY_FIELDS
        if persisted.get(field, _MISSING) != candidate.get(field, _MISSING)
    ]
    if differing:
        details = "; ".join(
            f"{field}: persisted {persisted.get(field, None)!r} vs this launch "
            f"{candidate.get(field, None)!r}" for field in differing
        )
        raise ContractMismatchError(
            f"{path} describes a different run and was NOT modified -- {details}. Launch "
            "this configuration in its own --run-dir, or point the launcher back at the "
            "inputs this run was created with."
        )
    return persisted


def append_resume_log(run_dir, entry):
    """Append one ``{timestamp, from_ckpt, from_sha}`` record to ``<run_dir>/resume_log.json``.

    Kept strictly outside the contract (the contract must stay bit-stable for the life of
    the run). Returns the full log. A resume log that is not a JSON list is fail-closed:
    silently replacing it would erase the run's resume history.
    """
    path = os.path.join(run_dir, RESUME_LOG_FILENAME)
    log = []
    if os.path.exists(path):
        with open(path) as fin:
            log = json.load(fin)
        if not isinstance(log, list):
            raise ValueError(f"{path} is not a JSON list of resume records")
    log.append(entry)
    _write_json_atomically(path, log)
    return log


# ======================================================================================
# CLIs used by the round-D2 bash launcher: `python -m src.training.run_contract …`
# ======================================================================================
# The exit-code contract both subcommands honour (codex MEDIUM 2). No path may exit 1:
# an uncaught exception is the one outcome a launcher cannot classify.
EXIT_OK = 0
EXIT_INPUT_ERROR = 2          # the CLI's own inputs are wrong (a launcher bug), incl. a
                              # run dir that belongs to another run (ContractMismatchError)
EXIT_CONTRACT_VIOLATION = 3   # ONLY a CheckpointContractError: the checkpoint disagrees


def build_arg_parser():
    parser = argparse.ArgumentParser(
        prog="python -m src.training.run_contract",
        description="Create a run's contract, or validate a checkpoint against one.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser(
        "make-contract",
        help="create <run-dir>/run_contract.json if absent (else keep the persisted one) "
             "and print its path",
    )
    make.add_argument("--run-dir", required=True)
    make.add_argument("--run-id", required=True)
    make.add_argument("--fraction", required=True, type=float)
    make.add_argument("--dataset-config", required=True)
    make.add_argument("--split-json", required=True)
    make.add_argument("--model-config", required=True)
    make.add_argument("--seed", required=True, type=int)
    make.add_argument("--micro-batch", required=True, type=int)
    make.add_argument("--num-gpus", required=True, type=int)
    make.add_argument("--accum-batches", required=True, type=int)
    make.add_argument("--sync-batchnorm", required=True, type=_as_bool)
    make.add_argument("--flac-sha", required=True)
    make.add_argument("--package-sha", required=True)
    make.add_argument("--launched-at", default=None,
                      help="ISO-8601 launch timestamp (default: now, UTC). Ignored when the "
                           "contract already exists, which is what makes a resume safe.")

    validate = sub.add_parser(
        "validate", help="validate a checkpoint against a run contract (exit 3 if it fails)"
    )
    validate.add_argument("--ckpt", required=True)
    validate.add_argument("--contract", required=True, help="the run's run_contract.json")
    validate.add_argument("--model-config", required=True, help="the arm's model config JSON")
    validate.add_argument("--expect-step", required=True, type=int)
    validate.add_argument("--for-resume", action="store_true",
                          help="also require optimizer and lr-scheduler state")
    return parser


def _cmd_make_contract(args):
    """stdout is ONLY the path, so the launcher can capture it with $(...)."""
    def build_fn(launched_at=None):
        """Rebuild the contract from THIS launch's inputs; inherit a persisted timestamp."""
        return build_contract(
            run_id=args.run_id,
            fraction=args.fraction,
            dataset_config_path=args.dataset_config,
            split_json_path=args.split_json,
            model_config_path=args.model_config,
            seed=args.seed,
            micro_batch=args.micro_batch,
            num_gpus=args.num_gpus,
            accum_batches=args.accum_batches,
            sync_batchnorm=args.sync_batchnorm,
            flac_sha=args.flac_sha,
            package_sha=args.package_sha,
            launched_at=args.launched_at if launched_at is None else launched_at,
        )

    try:
        contract = load_or_create_contract(args.run_dir, build_fn)
    except ContractMismatchError as err:
        print(f"CONTRACT MISMATCH: {err}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    except (OSError, ValueError, TypeError) as err:
        print(f"make-contract failed: {err}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    except Exception as err:  # never exit 1: a launcher cannot classify an uncaught crash
        print(f"make-contract failed unexpectedly ({type(err).__name__}: {err})",
              file=sys.stderr)
        return EXIT_INPUT_ERROR
    print(f"run contract for {contract['run_id']} launched at {contract['launched_at']}",
          file=sys.stderr)
    print(os.path.join(args.run_dir, CONTRACT_FILENAME))
    return EXIT_OK


def _cmd_validate(args):
    try:
        expected_contract = require_contract_fields(
            load_json_object(args.contract, "contract"), args.contract
        )
        expected_model_config = load_json_object(args.model_config, "model config")
    except (OSError, ValueError, TypeError) as err:
        print(f"validate failed: {err}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    try:
        contract = validate_checkpoint(args.ckpt, expected_contract, expected_model_config,
                                       args.expect_step, args.for_resume)
    except CheckpointContractError as err:
        print(f"CONTRACT VIOLATION: {err}", file=sys.stderr)
        return EXIT_CONTRACT_VIOLATION
    except Exception as err:  # never exit 1 (see _cmd_make_contract)
        print(f"validate failed unexpectedly ({type(err).__name__}: {err})", file=sys.stderr)
        return EXIT_INPUT_ERROR
    print(f"OK {args.ckpt}: run_id={contract.get('run_id')} step={args.expect_step}"
          f"{' resumable' if args.for_resume else ''}")
    return EXIT_OK


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.command == "make-contract":
        return _cmd_make_contract(args)
    return _cmd_validate(args)


if __name__ == "__main__":
    raise SystemExit(main())
