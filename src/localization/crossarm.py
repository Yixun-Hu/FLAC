"""exp_20 -- cross-arm localization at matched 40k: the gates exp_18 did not need.

exp_18 evaluated ONE checkpoint, so it never had to prove that two runs are
comparable. exp_20 compares three training policies (P1 vanilla, BF frame-average
invariant, YAW yaw-augmented) at the same step, and every claim it makes rests on
facts that must be machine-checked first:

  * **B2 admission** -- the file about to spend a GPU is the pre-registered
    40,000-step checkpoint of the arm it claims: step, canonical config equality,
    a COMPLETE EMA mirror, and the arm identity the checkpoint itself embeds.
  * **B1 FA binding** -- the frame-average arm is conditioned the way it was
    trained, with the chunk plan DECLARED rather than inherited from a module
    constant (announcement 06).
  * **B3 pairing** -- two arms may only be compared per query if they scored the
    same queries in the same order with the same contexts and the same noise.

The admission semantics are PORTED from exp_15's reviewed kit
(``worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_record_control.py``) --
canonical bytes, the EMA mirror contract, the read-stability snapshot -- and a
test runs both implementations over the same fixture so the port cannot drift.
"""
import copy
import hashlib
import json
import math
import os
from datetime import datetime, timezone

import numpy as np
import torch

#: EMA weights are stored under this prefix and mirror the online DiT family.
EMA_WEIGHT_PREFIX = "diffusion_ema.ema_model."
ONLINE_MODEL_PREFIX = "diffusion.model."
CHUNK_BYTES = 1 << 20

#: the registered endpoint every exp_20 arm is compared at.
REGISTERED_STEP = 40000

#: The three arms, their committed training configs and the identity each one
#: must embed. The config paths are the files whose canonical bytes the
#: checkpoints actually equal -- verified in exp_20 r1, and NOT
#: ``FLAC_AR.json``: every trained checkpoint carries the trainer's
#: ``gradient_checkpointing`` flags, so the released inference config cannot
#: equal any of them.
ARMS = {
    "P1": {
        "config_rel": "worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_VANCKPT.json",
        "cond_method": "vanilla",
        "lineage": "exp_07 P1 vanilla; config shared with exp_11's VANL arm",
        "identity": {"cond_method": None, "frame_avg_angles": None, "yaw_aug": None},
    },
    "BF": {
        "config_rel": "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json",
        "cond_method": "fa_invariant",
        "lineage": "exp_07 B-F fa_invariant (C4 frame average); byte-equal to exp_11's "
                   "FLAC_AR_BF_C4L.json",
        "identity": {"cond_method": "fa_invariant",
                     "frame_avg_angles": [0.0, 90.0, 180.0, 270.0], "yaw_aug": None},
    },
    "YAW": {
        "config_rel": "worklog/worklog_yixun/exp_15_yaw_aug_claude/FLAC_AR_YAWAUG.json",
        "cond_method": "vanilla",
        "lineage": "exp_17 A6000 yaw-aug (recipe-matched to P1/BF; NOT exp_15's cluster arm)",
        "identity": {"cond_method": None, "frame_avg_angles": None,
                     "yaw_aug": {"enabled": True, "img_w": 512, "seed": 42}},
    },
}


# --------------------------------------------------------------------------- #
# canonicalisation (ported from exp_15's reviewed kit)
# --------------------------------------------------------------------------- #
def validate_json_domain(obj, where="$"):
    """Assert an object lives in the JSON type domain, strictly.

    Canonicalisation only stands in for the object if it is injective, and
    ``json.dumps`` is not: it coerces non-string keys and emits ``NaN``.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"{where}: JSON object key {key!r} is {type(key).__name__}, not a string "
                    "-- canonicalisation would coerce it and hide the difference")
            validate_json_domain(value, f"{where}.{key}")
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            validate_json_domain(value, f"{where}[{index}]")
    elif isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"{where}: {obj!r} is not a finite number")
    elif not isinstance(obj, (str, bool, int, type(None))):
        raise ValueError(f"{where}: {type(obj).__name__} is not a JSON type")


def canonical_bytes(obj):
    """Sorted-key, whitespace-free, TYPE-SENSITIVE JSON bytes.

    Python says ``True == 1 == 1.0``; the canonical bytes are ``true``, ``1`` and
    ``1.0`` -- three different documents, which is the point.
    """
    validate_json_domain(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def canonical_sha256(obj):
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def sha256_file(path, chunk_bytes=CHUNK_BYTES):
    """Stream a file through sha256 -- the checkpoints are 724 MB."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_load_checkpoint(path):
    """Load with mmap and the SAFE unpickler only -- no ``weights_only=False``.

    This function exists to admit a file as evidence; falling back to arbitrary
    pickle execution at the moment its trustworthiness is in question would
    defeat the gate.
    """
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except Exception as error:                       # noqa: BLE001 -- reported as a refusal
        raise ValueError(f"{path}: the safe loader (mmap=True, weights_only=True) failed: "
                         f"{error}") from error


def _sha256_fd(fd, chunk_bytes=CHUNK_BYTES):
    """Hash the inode behind an ALREADY OPEN descriptor, from offset 0.

    Hashing by path and loading by path are two lookups: between them the name
    can be re-pointed at another inode and then restored, and the record would
    bind a checkpoint that was never inspected while every later identity check
    still passes. Hashing through the held descriptor pins the object measured
    (exp_15's semantics, ported exactly -- r1 review F5).
    """
    digest = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(fd, chunk_bytes, offset)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)
        offset += len(chunk)


def load_checkpoint_from_fd(fd):
    """Deserialize from the HELD descriptor -- never a second path lookup.

    ``torch.load`` cannot mmap a file object, so this trades lazy storages for a
    binding that survives an ABA swap: replace the name, load, restore it, and a
    path reopen deserializes the impostor while every identity check compares the
    restored path and passes. Admission already builds the model for load
    integrity, so the resident cost is one it was paying anyway.
    """
    duplicate = os.dup(fd)
    try:
        with os.fdopen(duplicate, "rb") as handle:
            handle.seek(0)
            return torch.load(handle, map_location="cpu", weights_only=True)
    except Exception as error:                       # noqa: BLE001 -- a refusal
        raise ValueError(f"the safe loader (held descriptor, weights_only=True) failed: "
                         f"{error}") from error


def _identity(stat):
    return {"device": stat.st_dev, "inode": stat.st_ino, "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns}


def snapshot_checkpoint(path):
    """``(checkpoint, sha256, identity)`` for ONE stable snapshot.

    The digest and the load are separate reads, so the guarantee is detection
    after the fact: the file's identity is re-stated afterwards and a file that
    moved underneath the read is refused rather than reported.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        before = os.fstat(fd)
        digest = _sha256_fd(fd)
        checkpoint = load_checkpoint_from_fd(fd)
        after_fd, after_path = os.fstat(fd), os.stat(path)
        if _identity(after_fd) != _identity(before):
            raise ValueError(f"{path}: the checkpoint changed while it was being read "
                             "(size/mtime moved on the same inode)")
        if (after_path.st_dev, after_path.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"{path}: the path changed to a different file while it was "
                             "being read (inode replaced)")
    finally:
        os.close(fd)
    return checkpoint, digest, _identity(before)


def summarize_ema(state_dict):
    """Require the EMA family to MIRROR the online DiT family, exactly.

    Suffix-set equality between ``diffusion.model.*`` and
    ``diffusion_ema.ema_model.*`` plus matching shape and dtype per suffix. Mere
    presence is far too weak: a checkpoint carrying only the EMA bookkeeping
    buffers would be admitted and would then load silently partial weights.
    """
    online = {key[len(ONLINE_MODEL_PREFIX):]: value for key, value in state_dict.items()
              if key.startswith(ONLINE_MODEL_PREFIX)}
    ema = {key[len(EMA_WEIGHT_PREFIX):]: value for key, value in state_dict.items()
           if key.startswith(EMA_WEIGHT_PREFIX)}
    if not online:
        raise ValueError(f"no online DiT weights ({ONLINE_MODEL_PREFIX}*) in the state_dict")
    if not ema:
        raise ValueError(f"no EMA weights ({EMA_WEIGHT_PREFIX}*) in the state_dict: exp_20 "
                         "evaluates EMA weights, so a checkpoint without them cannot be "
                         "admitted (bookkeeping buffers alone do not count)")
    missing, extra = sorted(set(online) - set(ema)), sorted(set(ema) - set(online))
    if missing or extra:
        raise ValueError("EMA family does not mirror the online DiT: "
                         f"{len(missing)} missing {missing[:5]}, {len(extra)} unexpected "
                         f"{extra[:5]} (online {len(online)} keys, EMA {len(ema)} keys)")
    for suffix in sorted(online):
        a, b = online[suffix], ema[suffix]
        if tuple(a.shape) != tuple(b.shape):
            raise ValueError(f"EMA/online shape mismatch for {suffix!r}: online "
                             f"{list(a.shape)} vs EMA {list(b.shape)}")
        if a.dtype != b.dtype:
            raise ValueError(f"EMA/online dtype mismatch for {suffix!r}: online {a.dtype} "
                             f"vs EMA {b.dtype}")
    inventory = "\n".join(f"{suffix}:{list(ema[suffix].shape)}:{ema[suffix].dtype}"
                          for suffix in sorted(ema))
    return {"ema_prefix": EMA_WEIGHT_PREFIX, "ema_key_count": len(ema),
            "online_model_key_count": len(online),
            "ema_inventory_sha256": hashlib.sha256(inventory.encode()).hexdigest()}


# --------------------------------------------------------------------------- #
# M6 -- lineage, as machine fields
# --------------------------------------------------------------------------- #
#: The NAS archive the three checkpoints were verified against.
NAS_PROVENANCE = ("/media/diskstation/yixunhu/FLAC/checkpoints/ar_40k_endpoints/"
                  "PROVENANCE.md")

#: Every cross-arm conclusion rests on ONE training run per arm.
LINEAGE_CAVEAT = (
    "This arm is a single historical training run. Cross-arm differences are conditional on "
    "these particular runs; with no replicated training per arm, no causal claim about the "
    "training policy itself is licensed."
)

#: Immutable citations, machine-readable (r1 review F7 / plan M6).
LINEAGE = {
    "P1": {
        "experiment": "exp_07",
        "completion_commits": ["2e2ac021e2e926163c68c57ba1e25c1b4340d9f5",
                               "f19f377f145e80d8d706c7dd5887e5c7a50d49de"],
        "topology": {"gpus": 2, "device": "NVIDIA RTX A6000", "nodes": 1,
                     "strategy": "ddp_find_unused_parameters_true", "sync_batchnorm": True},
        "recipe": {"seed": 42, "batch_size": 32, "accum_batches": 1, "num_workers": 6,
                   "max_steps": 40000, "precision": "bf16-mixed"},
        "branch": "check-equivariance-necessity",
    },
    "BF": {
        "experiment": "exp_07",
        "completion_commits": ["2e2ac021e2e926163c68c57ba1e25c1b4340d9f5",
                               "f19f377f145e80d8d706c7dd5887e5c7a50d49de"],
        "topology": {"gpus": 2, "device": "NVIDIA RTX A6000", "nodes": 1,
                     "strategy": "ddp_find_unused_parameters_true", "sync_batchnorm": True},
        "recipe": {"seed": 42, "batch_size": 32, "accum_batches": 1, "num_workers": 6,
                   "max_steps": 40000, "precision": "bf16-mixed"},
        "branch": "check-equivariance-necessity",
    },
    "YAW": {
        "experiment": "exp_17",
        # exp_17's own completion commits: training COMPLETE + audit PASSED, and
        # the per-angle rot-seed rows that closed it (plan M6)
        "completion_commits": ["42cbddaf3dbd4015e5e343edc13eb411fa6c18d5",
                               "f378775fd5034e81df3334024343f22cd5835f88"],
        "topology": {"gpus": 2, "device": "NVIDIA RTX A6000", "nodes": 1,
                     "strategy": "ddp_find_unused_parameters_true", "sync_batchnorm": True},
        "recipe": {"seed": 42, "batch_size": 32, "accum_batches": 1, "num_workers": 6,
                   "max_steps": 40000, "precision": "bf16-mixed",
                   "checkpoint_cadence": 2500},
        "branch": "exp-17-yawaug-a6000 (A6000 sibling checkout)",
        "note": "recipe-matched to P1/BF; deliberately NOT exp_15's cluster arm 16b964ec",
    },
}


def lineage_binding(arm, nas_provenance=NAS_PROVENANCE):
    """The arm's M6 binding block: immutable citations and the recipe, as fields."""
    spec = LINEAGE.get(arm)
    if spec is None:
        raise ValueError(f"no lineage binding registered for arm {arm!r}")
    binding = copy.deepcopy(spec)
    binding["arm"] = arm
    binding["nas_provenance"] = {
        "path": str(nas_provenance),
        "sha256": (sha256_file(nas_provenance) if os.path.isfile(nas_provenance) else None),
        "available": os.path.isfile(nas_provenance),
    }
    binding["caveat"] = LINEAGE_CAVEAT
    return binding


# --------------------------------------------------------------------------- #
# B2 -- admission
# --------------------------------------------------------------------------- #
def _identity_reasons(arm, embedded):
    """Named reasons the embedded config is not this arm's.

    exp_20's arms are defined by their TRAINING policy, and every one of those
    policies leaves a trace in the checkpoint's own config: BF carries
    ``cond_method``/``frame_avg_angles``, YAW carries ``yaw_aug``, P1 carries
    neither. The conditioning method is therefore checkable against the
    checkpoint itself, not only against the manifest.
    """
    spec = ARMS.get(arm)
    if spec is None:
        return [f"unknown arm {arm!r}; registered arms are {sorted(ARMS)}"]
    training = embedded.get("training")
    if not isinstance(training, dict):
        return [f"embedded model_config has no 'training' block "
                f"({type(training).__name__})"]
    reasons = []
    for field, want in spec["identity"].items():
        got = training.get(field, None)
        if want is None and got is not None:
            reasons.append(f"embedded training.{field}={got!r}, but arm {arm} must not carry "
                           f"it ({spec['lineage']})")
        elif want is not None and got != want:
            reasons.append(f"embedded training.{field}={got!r} != the registered {want!r} for "
                           f"arm {arm}; the training policy is part of the arm's identity")
    return reasons


def load_integrity(model_config, state_dict):
    """Build the model and report ``load_state_dict(strict=False)``."""
    from src.models import create_model_from_config

    model = create_model_from_config(model_config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return {"checked": True, "missing": sorted(missing), "unexpected": sorted(unexpected)}


def classify_load_integrity(missing, unexpected):
    """The REGISTERED integrity contract: 0 missing / 0 STRAY unexpected.

    ``eval_FLAC.check_load_integrity`` tolerates exactly two unexpected prefixes
    -- ``diffusion_ema.`` (the EMA bookkeeping buffers) and ``losses.`` (the
    training loss module) -- because every wrapped PL checkpoint carries them.
    Demanding zero RAW unexpected keys would refuse all three exp_20 arms for a
    benign reason and would be stricter than the driver that actually runs them.
    """
    from eval_FLAC import LOAD_WHITELIST_PREFIXES

    missing = sorted(missing)
    unexpected = sorted(unexpected)
    stray = [key for key in unexpected if not key.startswith(LOAD_WHITELIST_PREFIXES)]
    return {"checked": True, "missing": missing, "unexpected": unexpected, "stray": stray,
            "n_missing": len(missing), "n_unexpected": len(unexpected), "n_stray": len(stray),
            "n_whitelisted": len(unexpected) - len(stray),
            "whitelist": list(LOAD_WHITELIST_PREFIXES),
            "clean": not (missing or stray)}


def admit_checkpoint(ckpt_path, arm_config_path, arm, expect_step=REGISTERED_STEP,
                     check_load_integrity=True):
    """The B2 gate: everything re-derived FROM THE FILES, as one record.

    Returns the admission record, ``admitted`` false with named ``reasons``
    rather than raising, so a caller can report every problem at once. A record
    is evidence about a file at a moment; the later gates re-derive and compare.
    """
    reasons = []
    config_bytes = open(arm_config_path, "rb").read()
    config_obj = json.loads(config_bytes)
    record = {
        "arm": arm, "ckpt_path": str(ckpt_path), "config_path": str(arm_config_path),
        "expect_step": int(expect_step),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "config_canonical_sha256": canonical_sha256(config_obj),
        "sha256": None, "bytes": None, "global_step": None,
        "embedded_config_canonical_sha256": None,
        "ema_key_count": None, "online_model_key_count": None,
        "ema_inventory_sha256": None,
        "cond_method": None, "frame_avg_angles": None,
        "load_integrity": {"checked": False},
        "lineage": (ARMS.get(arm) or {}).get("lineage"),
        "lineage_binding": lineage_binding(arm) if arm in LINEAGE else None,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    try:
        checkpoint, digest, identity = snapshot_checkpoint(ckpt_path)
    except (ValueError, OSError) as error:
        record["reasons"] = [str(error)]
        record["admitted"] = False
        return record
    record["sha256"], record["bytes"] = digest, identity["bytes"]

    step = checkpoint.get("global_step", None)
    record["global_step"] = step if isinstance(step, int) and not isinstance(step, bool) else None
    if isinstance(step, bool) or not isinstance(step, int):
        reasons.append(f"embedded global_step is {step!r} ({type(step).__name__}), expected a "
                       "plain int")
    elif step != int(expect_step):
        reasons.append(f"embedded global_step is {step}, not the pre-registered endpoint "
                       f"{expect_step}")

    state_dict = checkpoint.get("state_dict", None)
    if not isinstance(state_dict, dict) or not state_dict:
        reasons.append("no usable 'state_dict' in the checkpoint")
        state_dict = {}
    else:
        try:
            record.update(summarize_ema(state_dict))
        except ValueError as error:
            reasons.append(f"EMA/online mirror check FAILED: {error}")

    embedded = checkpoint.get("model_config", None)
    if embedded is None:
        reasons.append("no 'model_config' embedded in the checkpoint: it cannot be bound to "
                       "the config it was trained with")
    elif not isinstance(embedded, dict):
        reasons.append(f"embedded model_config is a {type(embedded).__name__}, not an object")
    else:
        try:
            embedded_bytes = canonical_bytes(embedded)
        except ValueError as error:
            reasons.append(f"embedded model_config is not canonicalisable: {error}")
        else:
            record["embedded_config_canonical_sha256"] = hashlib.sha256(
                embedded_bytes).hexdigest()
            if embedded_bytes != canonical_bytes(config_obj):
                reasons.append(f"the embedded model_config differs from {arm_config_path} "
                               "(canonical bytes) -- this checkpoint was not trained with "
                               "that config")
        training = embedded.get("training") or {}
        if isinstance(training, dict):
            record["cond_method"] = training.get("cond_method", "vanilla")
            angles = training.get("frame_avg_angles")
            record["frame_avg_angles"] = (None if angles is None
                                          else [float(a) for a in angles])
        reasons += _identity_reasons(arm, embedded)

    if check_load_integrity and state_dict:
        try:
            from eval_localization import prepare_state_dict

            prepared, _source = prepare_state_dict(
                {"state_dict": dict(state_dict)}, (embedded or {}).get("training"))
            raw = load_integrity(json.loads(json.dumps(config_obj)), prepared)
            record["load_integrity"] = classify_load_integrity(raw["missing"],
                                                               raw["unexpected"])
            if not record["load_integrity"]["clean"]:
                reasons.append(
                    f"load integrity: {record['load_integrity']['n_missing']} missing / "
                    f"{record['load_integrity']['n_stray']} stray unexpected keys "
                    f"(first stray: {record['load_integrity']['stray'][:3]}), expected 0/0 "
                    f"under the registered whitelist {record['load_integrity']['whitelist']}")
        except Exception as error:                   # noqa: BLE001 -- reported as a refusal
            record["load_integrity"] = {"checked": False, "error": str(error)}
            reasons.append(f"load integrity could not be established: {error}")

    record["reasons"] = reasons
    record["admitted"] = not reasons
    return record


# --------------------------------------------------------------------------- #
# B1 -- frame-average protocol binding
# --------------------------------------------------------------------------- #
#: the registered C4 orbit; angles[0] must be the identity pass.
FA_ANGLES = (0.0, 90.0, 180.0, 270.0)

#: The chunk plan exp_20 DECLARES for evaluation (announcement 06 §3): one
#: conditioner forward per angle. ``invariant_conditioning`` partitions the orbit
#: into chunks of ``max_fwd_samples`` rows and a chunk is whole angles, so
#: ``angles_per_chunk = max(1, cap // batch)``: setting the cap to the candidate
#: micro-batch is exactly what makes it one. The module default (64) would put
#: all four angles of a 10-candidate query in a single forward instead -- a
#: different partition, and under train-mode DINOv3 a different RoPE draw.
FA_CHUNK_PLAN = "per_angle"

#: What a BF registration manifest must lock, and the run must resolve. The
#: plan-string alone was paperwork: "per_angle" is a claim, while the cap policy,
#: the cap, the micro-batch, the orbit size, the angles per chunk, the number of
#: orbit forwards and the shared-angle count are the NUMBERS that decide what the
#: forward actually does (r1 review F4).
FA_LOCKED_FIELDS = ("cond_method", "frame_avg_angles", "rotate_deg", "cond_autocast",
                    "frame_avg_chunk_plan", "cap_policy", "frame_avg_fwd_cap",
                    "candidate_micro_batch", "orbit_size", "angles_per_chunk",
                    "n_orbit_forwards", "shared_angle_count")

#: The registered FA micro-batch: one query's whole candidate set (M = 10). The
#: driver conditions all candidates of a query in ONE call, so this -- not the
#: dataloader's --batch-size, which chunks the sampling rows -- is the batch the
#: orbit partition divides (r2 re-review nit).
REGISTERED_CANDIDATE_MICRO_BATCH = 10
CAP_POLICY = "candidate_micro_batch"


def fa_max_fwd_samples(metadata):
    """The per-angle cap for THIS conditioning call: the candidate micro-batch."""
    count = len(metadata)
    if count < 1:
        raise ValueError("frame-average conditioning needs at least one metadata item")
    return int(count)


def fa_conditioning(conditioner, metadata, device, angles=FA_ANGLES, record=None):
    """The driver's frame-average conditioning call, with the plan made explicit.

    Identical to ``eval_FLAC``'s call except that the cap is stated here rather
    than inherited: the localization driver conditions all candidates of one
    query in a single call, so the cap IS that call's batch. When ``record`` is
    given it receives the partition this call actually EXECUTED -- cheap ints the
    row carries, so a plan that drifts from its manifest is detectable
    afterwards rather than assumed (r1 review F4).
    """
    from src.data.yaw_rotation import invariant_conditioning

    cap = fa_max_fwd_samples(metadata)
    recorder = _PartitionRecorder(conditioner)
    out = invariant_conditioning(recorder, metadata, device, tuple(angles),
                                 max_fwd_samples=cap)
    if record is not None:
        record.update(observed_partition(recorder.partition, len(metadata), angles, cap))
    return out


def observed_partition(partition, batch, angles, cap):
    """The executed chunk plan as numbers, from what the conditioner was called with."""
    partition = [int(p) for p in partition]
    per_chunk = sorted({p // max(1, batch) for p in partition}) or [0]
    return {
        "cond_method": "fa_invariant", "cap_policy": CAP_POLICY,
        "frame_avg_fwd_cap": int(cap), "candidate_micro_batch": int(batch),
        "orbit_size": len(tuple(angles)), "partition": partition,
        "n_orbit_forwards": len(partition),
        "angles_per_chunk": max(per_chunk), "shared_angle_count": max(per_chunk),
    }


def fa_run_state(cond_method, frame_avg_angles=FA_ANGLES, rotate_deg=0.0,
                 cond_autocast="default", chunk_plan=FA_CHUNK_PLAN,
                 candidate_micro_batch=REGISTERED_CANDIDATE_MICRO_BATCH):
    """The FA protocol state a run resolves, in the manifest's own vocabulary.

    The numeric execution state is derived from the SAME rule the code applies
    (``angles_per_chunk = max(1, cap // batch)`` with ``cap = batch``), so the
    manifest locks the partition rather than a word describing it.
    """
    angles = [float(a) for a in frame_avg_angles]
    batch = int(candidate_micro_batch)
    cap = batch
    per_chunk = max(1, cap // batch)
    return {
        "cond_method": cond_method,
        "frame_avg_angles": angles,
        "rotate_deg": float(rotate_deg),
        "cond_autocast": cond_autocast,
        "frame_avg_chunk_plan": chunk_plan,
        "orbit_execution": chunk_plan,
        "cap_policy": CAP_POLICY,
        "frame_avg_fwd_cap": cap,
        "candidate_micro_batch": batch,
        "orbit_size": len(angles),
        "angles_per_chunk": per_chunk,
        "n_orbit_forwards": max(0, len(angles) - 1),
        "shared_angle_count": per_chunk,
    }


def fa_reasons(manifest, state, fields=FA_LOCKED_FIELDS):
    """Named mismatches between a registration manifest and the resolved FA state."""
    reasons = []
    for field in fields:
        if field not in manifest:
            reasons.append(f"registration manifest does not lock {field!r}; a frame-average "
                           "run must declare its whole conditioning protocol")
            continue
        locked, actual = manifest[field], state.get(field)
        if field in ("cap_policy", "frame_avg_fwd_cap", "candidate_micro_batch", "orbit_size",
                     "angles_per_chunk", "n_orbit_forwards", "shared_angle_count"):
            locked = locked if isinstance(locked, str) else (
                None if locked is None else int(locked))
            actual = actual if isinstance(actual, str) else (
                None if actual is None else int(actual))
        if field == "frame_avg_angles":
            locked = None if locked is None else [float(a) for a in locked]
            actual = None if actual is None else [float(a) for a in actual]
        elif field == "rotate_deg":
            locked = None if locked is None else float(locked)
            actual = None if actual is None else float(actual)
        if locked != actual:
            reasons.append(f"registered {field} is {locked!r} but this run resolves {actual!r}")
    return reasons


def cond_method_binding(checkpoint, cond_method, manifest=None, manifest_verified=False,
                        registered=False):
    """Where the conditioning method is bound: the CHECKPOINT, a verified
    MANIFEST, or nowhere at all.

    exp_20's three arms embed their training config, so ``training.cond_method``
    binds the CLI flag to the file itself. The released EMA checkpoint embeds no
    ``model_config``: there the method is not detectable, and claiming a manifest
    binding when no verified manifest exists would be a fiction. That case is
    ``unbound`` -- refused for a registered run, stamped for a smoke/dev one
    (r1 review F6).
    """
    embedded = (checkpoint or {}).get("model_config")
    training = embedded.get("training") if isinstance(embedded, dict) else None
    if isinstance(training, dict):
        found = training.get("cond_method", "vanilla")
        reasons = []
        if str(found) != str(cond_method):
            reasons.append(f"the checkpoint was trained with cond_method={found!r} but the run "
                           f"asks for {cond_method!r}; conditioning a model the way it was not "
                           "trained is catastrophic, not a variant")
        return {"binding": "checkpoint", "checkpoint_cond_method": found, "reasons": reasons,
                "stamped": False, "manifest_verified": bool(manifest_verified),
                "note": "the checkpoint embeds its training config, so the flag is bound to "
                        "the file itself"}

    detail = ("the checkpoint embeds no model_config" if not isinstance(embedded, dict)
              else "the embedded model_config carries no training block")
    if manifest is not None and manifest_verified:
        locked = (manifest or {}).get("cond_method")
        reasons = []
        if locked is None:
            reasons.append(f"{detail} and the verified manifest locks no cond_method; nothing "
                           "binds the conditioning method")
        elif str(locked) != str(cond_method):
            reasons.append(f"the verified manifest locks cond_method={locked!r} but the run "
                           f"asks for {cond_method!r}")
        return {"binding": "manifest" if not reasons else "unbound",
                "checkpoint_cond_method": None, "manifest_cond_method": locked,
                "reasons": reasons, "stamped": False, "manifest_verified": True,
                "note": f"{detail}; the binding rests on the verified registration manifest"}

    reasons = []
    if registered:
        reasons.append(f"{detail} and no verified registration manifest binds it; a registered "
                       "run may not leave the conditioning method unbound")
    return {"binding": "unbound", "checkpoint_cond_method": None,
            "manifest_cond_method": (manifest or {}).get("cond_method") if manifest else None,
            "reasons": reasons, "stamped": True,
            "manifest_verified": bool(manifest_verified),
            "note": f"{detail} and no verified manifest was supplied: the conditioning method "
                    "is UNBOUND and the row is stamped as such"}


#: PREREGISTERED parity tolerances. The caller does not get to choose: a gate
#: whose bar is an argument is a gate that can be argued down (r1 review F3).
PARITY_TOLERANCES = {"autocast_off": 0.0, "registered_autocast": 2e-2}


class _PartitionRecorder:
    """Wrap a conditioner and record the batch size of every ORBIT forward."""

    def __init__(self, inner):
        self.inner, self.partition, self.calls = inner, [], []

    def __call__(self, metadata, device, only_ids=None):
        self.calls.append({"batch": len(metadata), "only_ids": None if only_ids is None
                           else tuple(only_ids)})
        if only_ids is not None:                   # the rotated-angle forwards
            self.partition.append(len(metadata))
        return self.inner(metadata, device, only_ids=only_ids) if only_ids is not None \
            else self.inner(metadata, device)


def _tensor_facts(value):
    tensor = torch.as_tensor(value)
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "finite": bool(torch.isfinite(tensor.float()).all())}


def fa_parity_gate(conditioner_factory, metadata, device="cpu", angles=FA_ANGLES,
                   replay_angles=None, replay_cap=None, autocast=False, tolerance=None,
                   dtype=None):
    """B1(b): the driver's FA conditioning vs an ``eval_FLAC``-faithful replay.

    Fail-closed on every channel the r1 review found open: the two output key
    sets must be EQUAL (a missing or extra id is a failure, not an ignored one),
    masks are compared as well as tensors, any non-finite value anywhere fails,
    the executed partitions must match, and the tolerance is preregistered --
    bitwise with autocast off, a fixed bound under the registered autocast.
    """
    from src.data.yaw_rotation import invariant_conditioning

    bar = PARITY_TOLERANCES["registered_autocast" if autocast else "autocast_off"]
    if tolerance is not None and float(tolerance) != bar:
        raise ValueError(f"the parity tolerance is preregistered ({bar}); a caller-chosen "
                         f"{tolerance} would let the gate be argued down")
    angles = tuple(float(a) for a in angles)
    replay = tuple(float(a) for a in (replay_angles if replay_angles is not None else angles))
    driver_cap = fa_max_fwd_samples(metadata)
    reference_cap = int(replay_cap) if replay_cap is not None else driver_cap

    def _run(fn):
        if not autocast:
            return fn()
        context = (torch.amp.autocast(device) if dtype is None
                   else torch.amp.autocast(device, dtype=dtype))
        with context:
            return fn()

    # The DRIVER side must be the production path -- the same function the
    # campaign executes -- or the gate proves nothing about the campaign. Only
    # the REPLAY side is the independent reimplementation (r2 F3).
    import eval_localization as driver_module

    driver_recorder = _PartitionRecorder(conditioner_factory())
    replay_recorder = _PartitionRecorder(conditioner_factory())
    driver = _run(lambda: driver_module.conditioning_call(
        "fa_invariant", driver_recorder, metadata, device, angles))
    reference = _run(lambda: invariant_conditioning(replay_recorder, metadata, device, replay,
                                                    max_fwd_samples=reference_cap))

    reasons, per_id, worst, worst_mask = [], {}, 0.0, 0.0
    bitwise, finite = True, True

    # Per-side facts BEFORE any comparison: a non-finite value anywhere on either
    # side -- tensor or mask, shared key or not -- makes the verdict meaningless.
    per_side = {}
    for label, block in (("driver", driver), ("replay", reference)):
        facts = {}
        for name, pair in block.items():
            tensor, mask = torch.as_tensor(pair[0]), torch.as_tensor(pair[1])
            tensor_finite = bool(torch.isfinite(tensor.float()).all())
            mask_finite = bool(torch.isfinite(mask.float()).all())
            facts[name] = {"tensor_dtype": str(tensor.dtype),
                           "tensor_shape": list(tensor.shape),
                           "tensor_finite": tensor_finite,
                           "mask_dtype": str(mask.dtype), "mask_shape": list(mask.shape),
                           "mask_finite": mask_finite}
            if not (tensor_finite and mask_finite):
                finite = False
                reasons.append(f"{label}.{name}: a non-finite value is present "
                               f"(tensor finite={tensor_finite}, mask finite={mask_finite}); "
                               "a parity verdict over NaN/Inf is meaningless")
        per_side[label] = facts
    if set(driver) != set(reference):
        missing = sorted(set(reference) - set(driver))
        extra = sorted(set(driver) - set(reference))
        reasons.append(f"output key set differs: missing {missing}, extra {extra}")
        bitwise = False
    if driver_recorder.partition != replay_recorder.partition:
        reasons.append(f"executed partition differs: driver {driver_recorder.partition} vs "
                       f"replay {replay_recorder.partition}")
        bitwise = False
    if list(angles) != list(replay):
        reasons.append(f"orbit differs: driver {list(angles)} vs replay {list(replay)}")

    for name in sorted(set(driver) & set(reference)):
        # dtype equality is decided on the ORIGINAL tensors: the value comparison
        # casts to float32, which hides a float64/float32 pair with equal values
        # and a mask stored as bool (r3 re-review F3).
        for kind in ("tensor", "mask"):
            got = per_side["driver"][name][f"{kind}_dtype"]
            want = per_side["replay"][name][f"{kind}_dtype"]
            if got != want:
                reasons.append(f"{name}: {kind} dtype differs (driver {got} vs replay "
                               f"{want}); equal values in different storage types are not "
                               "parity")
        a_t, a_m = torch.as_tensor(driver[name][0]).float(), torch.as_tensor(driver[name][1])
        b_t, b_m = (torch.as_tensor(reference[name][0]).float(),
                    torch.as_tensor(reference[name][1]))
        entry = _tensor_facts(a_t)
        entry["mask_shape"] = list(torch.as_tensor(a_m).shape)
        if a_t.shape != b_t.shape or torch.as_tensor(a_m).shape != torch.as_tensor(b_m).shape:
            reasons.append(f"{name}: shape differs (tensor {list(a_t.shape)} vs "
                           f"{list(b_t.shape)}, mask {list(torch.as_tensor(a_m).shape)} vs "
                           f"{list(torch.as_tensor(b_m).shape)})")
            entry.update({"max_abs_diff": float("inf"), "bitwise": False,
                          "mask_max_abs_diff": float("inf")})
            per_id[name], bitwise = entry, False
            continue
        diff = float((a_t - b_t).abs().max())
        mask_diff = float((torch.as_tensor(a_m).float()
                           - torch.as_tensor(b_m).float()).abs().max())
        entry.update({"max_abs_diff": diff, "mask_max_abs_diff": mask_diff,
                      "bitwise": bool(torch.equal(a_t, b_t))})
        entry["finite"] = (per_side["driver"][name]["tensor_finite"]
                           and per_side["driver"][name]["mask_finite"]
                           and per_side["replay"][name]["tensor_finite"]
                           and per_side["replay"][name]["mask_finite"])
        if mask_diff > bar:
            reasons.append(f"{name}: mask differs by {mask_diff}")
        worst, worst_mask = max(worst, diff), max(worst_mask, mask_diff)
        bitwise = bitwise and entry["bitwise"]
        per_id[name] = entry

    within = worst <= bar and worst_mask <= bar
    return {
        "match": bool(finite and not reasons and (within if autocast else bitwise)),
        "bitwise": bitwise, "finite": finite, "reasons": reasons,
        "max_abs_diff": worst, "mask_max_abs_diff": worst_mask,
        "tolerance": bar, "autocast": bool(autocast),
        "ids": sorted(set(driver) | set(reference)), "per_id": per_id,
        "driver_ids": sorted(driver), "replay_ids": sorted(reference), "per_side": per_side,
        "driver_path": "eval_localization.conditioning_call",
        "replay_path": "src.data.yaw_rotation.invariant_conditioning",
        "angles": list(angles), "replay_angles": list(replay),
        "driver_cap": driver_cap, "replay_cap": reference_cap,
        "driver_partition": list(driver_recorder.partition),
        "replay_partition": list(replay_recorder.partition),
        "chunk_plan": FA_CHUNK_PLAN, "n_metadata": len(metadata), "device": str(device),
    }


#: the source files whose bytes decide what an FA forward actually does.
FA_SOURCE_FILES = ("src/data/yaw_rotation.py", "src/localization/crossarm.py",
                   "eval_localization.py")


def fa_source_shas(repo_root=None):
    """sha256 of the executable FA sources, recorded at RUN time.

    A registration commit need only be an ancestor of HEAD, so the code that
    executes an orbit can change after the manifest says ``per_angle``. Hashing
    the blobs at run time makes that drift detectable afterwards (r1 review F4).
    """
    root = repo_root or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "..", ".."))
    out = {}
    for relpath in FA_SOURCE_FILES:
        path = os.path.join(root, relpath)
        out[relpath] = sha256_file(path) if os.path.isfile(path) else None
    return out


def _build_conditioner_factory(ckpt_path, model_config_path, device):
    """A factory that builds a FRESH conditioner from the real checkpoint."""
    import copy as _copy

    from src.models import create_model_from_config
    from src.training import create_training_wrapper_from_config

    with open(model_config_path) as handle:
        model_config = json.load(handle)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    training_config = _copy.deepcopy(model_config.get("training", None))

    def factory():
        from eval_localization import prepare_state_dict

        local_config = _copy.deepcopy(model_config)
        local_training = _copy.deepcopy(training_config)
        state_dict, weights_source = prepare_state_dict(
            {"state_dict": dict(checkpoint["state_dict"])}, local_training)
        model = create_model_from_config(local_config)
        model.load_state_dict(state_dict, strict=False)
        local_config["training"] = local_training
        module = create_training_wrapper_from_config(local_config, model)
        module.eval().requires_grad_(False)
        module.to(device)
        factory.weights_source = weights_source
        return module.diffusion.conditioner

    factory.weights_source = None
    return factory, {"device": str(device), "weights_source": None}


def _load_one_query(args, device, position=0):
    """ONE real query's candidate metadata, through the driver's own loader.

    Exactly the objects a scoring pass conditions on: the loader at the pinned
    parallelism, the frozen candidate manifest, and the same
    ``candidate_metadata`` swap the query path applies -- so the parity gate
    measures the conditioning of a real query, not of a fixture.
    """
    import itertools

    from eval_localization import (build_dataloader, candidate_camera_positions,
                                   candidate_set_from_manifest, load_dataset_config,
                                   manifest_for_dataset_config, parse_ir_filename,
                                   read_model_config, room_id_from_relpath, sample_target_id,
                                   _iter_items)
    from src.localization.candidates import candidate_metadata

    dataset_config = load_dataset_config(args)
    model_config = read_model_config(args)
    manifest = manifest_for_dataset_config(dataset_config)
    loader = build_dataloader(args, model_config, dataset_config)
    item = next(itertools.islice(_iter_items(loader), int(position), None), None)
    if item is None:
        raise ValueError(f"the dataset has no query at position {position}")
    _obs_wav, md = item

    room_id = room_id_from_relpath(md["relpath"])
    gt_node, receiver_node = parse_ir_filename(md["path"])
    cand_set = candidate_set_from_manifest(manifest, room_id, gt_node, receiver_node)
    positions = candidate_camera_positions(cand_set)
    metadata = [candidate_metadata(md, positions[m]) for m in range(positions.shape[0])]
    query = {"query_id": sample_target_id(md), "room_id": room_id,
             "relpath": md["relpath"], "position": int(position),
             "n_candidates": len(metadata), "gt_node": int(gt_node),
             "receiver_node": int(receiver_node),
             "candidate_nodes": [int(n) for n in cand_set.nodes],
             "gt_index": int(cand_set.gt_index)}
    return metadata, query


def run_fa_parity(ckpt_path, model_config_path, dataset_config=None, device="cpu",
                  autocast_modes=("off", "registered"), position=0, angles=FA_ANGLES,
                  rotate_deg=0.0, repo_root=None, args=None):
    """The REAL fa-parity gate: one query, one checkpoint, the full evidence.

    Produces the record the review enumerates -- artifact digests, the query's
    identity and candidate count, requested and resolved autocast, the orbit and
    rotation, both caps and both EXECUTED partitions, exact key sets, per-key
    shapes/dtypes/finiteness/differences, the preregistered tolerances, and the
    runtime it all happened on.
    """
    import platform

    import argparse as _argparse

    factory, build_facts = _build_conditioner_factory(ckpt_path, model_config_path, device)
    if args is None:
        args = _argparse.Namespace(model_config=model_config_path,
                                   dataset_config=dataset_config, ckpt_path=ckpt_path,
                                   device=device, batch_size=4, num_workers=4, seed=42)
    metadata, query = _load_one_query(args, device, position=position)

    results = {}
    for mode in autocast_modes:
        autocast = mode != "off"
        requested = "off" if not autocast else "default"
        resolved = None
        if autocast:
            from eval_localization import resolve_cond_autocast

            _enabled, resolved_dtype = resolve_cond_autocast("default")
            resolved = str(resolved_dtype) if resolved_dtype is not None else "device default"
        verdict = fa_parity_gate(factory, metadata, device=device, angles=angles,
                                 autocast=autocast)
        verdict["requested_autocast"] = requested
        verdict["resolved_autocast_dtype"] = resolved if autocast else "n/a"
        results[mode] = verdict

    with open(model_config_path, "rb") as handle:
        model_config_sha = hashlib.sha256(handle.read()).hexdigest()
    return {
        "mode": "fa-parity", "ckpt_path": str(ckpt_path),
        "ckpt_sha256": sha256_file(ckpt_path),
        "model_config_path": str(model_config_path), "model_config_sha256": model_config_sha,
        "dataset_config": dataset_config,
        "query": query, "angles": [float(a) for a in angles], "rotate_deg": float(rotate_deg),
        "chunk_plan": FA_CHUNK_PLAN, "results": results,
        "tolerances": dict(PARITY_TOLERANCES),
        "source_shas": fa_source_shas(repo_root),
        "weights_source": build_facts.get("weights_source") or factory.weights_source,
        "runtime": {"device": str(device), "torch": torch.__version__,
                    "python": platform.python_version(), "platform": platform.platform(),
                    "cuda": torch.version.cuda},
        "passed": all(result["match"] for result in results.values()),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# --------------------------------------------------------------------------- #
# B3 -- the paired-inference gate
# --------------------------------------------------------------------------- #
#: what two arms must share EXACTLY before their queries may be paired.
PAIRING_FIELDS = ("query_ids", "context_stream_digest", "split_hash", "split_file_sha256",
                  "candidate_manifest_sha256", "loader", "noise_keys")


#: every pairing field must be present AND non-empty before arms may be paired.
REQUIRED_PAIRING_FIELDS = PAIRING_FIELDS


def registered_seeds(manifest):
    """The registered seed set, read from the manifest -- never hardcoded."""
    seeds = (manifest or {}).get("seeds")
    if not isinstance(seeds, (list, tuple)) or not seeds:
        raise ValueError("the manifest locks no non-empty 'seeds' list; the registered seed "
                         "set cannot be assumed")
    return tuple(int(seed) for seed in seeds)


def pairing_facts(rows_path, summary_path, arm, regime):
    """Read the pairing facts of ONE published cell from its own artifacts."""
    query_ids, noise_keys = [], {}
    with open(rows_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            identity = str(row["query_id"])
            query_ids.append(identity)
            if row.get("noise_keys") is None:
                raise ValueError(f"{rows_path}: query {identity!r} records no noise_keys; the "
                                 "pairing gate cannot prove the arms drew the same noise")
            noise_keys[identity] = [int(k) for k in row["noise_keys"]]
    with open(summary_path) as handle:
        provenance = json.load(handle).get("provenance") or {}
    return {
        "arm": arm, "regime": regime, "seed": provenance.get("seed"),
        "query_ids": query_ids,
        "context_stream_digest": provenance.get("context_stream_digest"),
        "split_hash": provenance.get("split_hash"),
        "split_file_sha256": provenance.get("split_file_sha256"),
        "candidate_manifest_sha256": provenance.get("candidate_manifest_sha256"),
        "loader": {"batch_size": provenance.get("batch_size"),
                   "num_workers": provenance.get("num_workers"),
                   "shuffle": provenance.get("loader_shuffle"),
                   "drop_last": provenance.get("loader_drop_last")},
        "noise_keys": noise_keys,
        "rows_path": str(rows_path), "summary_path": str(summary_path),
    }


def _describe(field, value):
    if field == "query_ids":
        return f"{len(value)} ids, first {value[:2]}, last {value[-2:]}"
    if field == "noise_keys":
        return f"{len(value)} queries keyed"
    return repr(value)


def validate_pairing(runs, fields=None, strict_fields=False):
    """B3: prove two or more arms scored the SAME queries the same way.

    Paired per-query inference is only meaningful if the arms saw one stream:
    the same identities in the same ORDER, drawn with the same contexts and the
    same noise, over the same split and candidate sets, through an identically
    configured loader. Any difference blocks paired reporting -- the fallback is
    an unpaired comparison, labelled as one, never a silent pairing.
    """
    # The mandatory set is a FLOOR, not a default: a caller may add fields but
    # never narrow them, and fields=() must not turn the gate off (r2 F2).
    requested = tuple(fields or ())
    if strict_fields:
        unknown = [field for field in requested if field not in PAIRING_FIELDS]
        if unknown:
            raise ValueError(f"{unknown} are not pairing fields; the mandatory set is "
                             f"{list(PAIRING_FIELDS)}")
    fields = tuple(PAIRING_FIELDS) + tuple(f for f in requested if f not in PAIRING_FIELDS)

    runs = list(runs)
    if len(runs) < 2:
        raise ValueError("validate_pairing needs at least two arms to pair")
    reference = sorted(runs, key=lambda r: str(r["arm"]))[0]
    mismatches = []

    # Fail-closed FIRST: absent or empty evidence is not agreement (r1 review F2).
    arms = [str(run.get("arm")) for run in runs]
    if len(set(arms)) != len(arms):
        mismatches.append({"field": "arms", "arms": sorted(arms),
                           "detail": f"the same arm appears twice: {sorted(arms)}"})
    for run in runs:
        arm = str(run.get("arm"))
        for field in REQUIRED_PAIRING_FIELDS:
            value = run.get(field)
            if value is None or (hasattr(value, "__len__") and len(value) == 0):
                mismatches.append({"field": field, "arms": [arm],
                                   "detail": f"{arm} carries no {field}; missing evidence is "
                                             "not agreement"})
        ids = run.get("query_ids") or []
        if len(set(ids)) != len(ids):
            mismatches.append({"field": "query_ids", "arms": [arm],
                               "detail": f"{arm} has duplicate query ids in its stream"})
        keys = run.get("noise_keys") or {}
        missing_keys = [identity for identity in ids if identity not in keys]
        if ids and missing_keys:
            mismatches.append({"field": "noise_keys", "arms": [arm],
                               "detail": f"{arm} has no noise array for {len(missing_keys)} "
                                         f"queries (first {missing_keys[:3]})"})
        loader = run.get("loader") or {}
        unset = sorted(k for k, v in loader.items() if v is None)
        if unset:
            mismatches.append({"field": "loader", "arms": [arm],
                               "detail": f"{arm} leaves loader settings unset: {unset}"})

    for run in runs:
        if (run.get("regime"), run.get("seed")) != (reference.get("regime"),
                                                    reference.get("seed")):
            mismatches.append({
                "field": "cell", "arms": [reference["arm"], run["arm"]],
                "detail": f"{reference['arm']} is {reference.get('regime')}/seed "
                          f"{reference.get('seed')} but {run['arm']} is {run.get('regime')}/"
                          f"seed {run.get('seed')}"})
    for field in fields:
        for run in runs:
            if run is reference:
                continue
            if run.get(field) != reference.get(field):
                mismatches.append({
                    "field": field, "arms": [reference["arm"], run["arm"]],
                    "detail": f"{reference['arm']}: {_describe(field, reference.get(field))} "
                              f"!= {run['arm']}: {_describe(field, run.get(field))}"})
    return {
        "paired": not mismatches,
        "n_arms": len(runs), "arms": sorted(str(r["arm"]) for r in runs),
        "reference_arm": str(reference["arm"]),
        "regime": reference.get("regime"), "seed": reference.get("seed"),
        "n_queries": len(reference.get("query_ids") or []),
        "fields_checked": list(fields), "mismatches": mismatches,
        "fallback": ("n/a -- the cells are paired" if not mismatches else
                     "paired reporting is BLOCKED; only an unpaired comparison may be "
                     "reported, labelled as unpaired"),
    }


def aggregate_seeds_per_query(per_seed, registered_seeds, fields=("top1", "e_loc")):
    """Seeds are REPLICATES: average each query across seeds, then cluster.

    Treating three seeds as three independent queries would triple the apparent
    sample size of a room-clustered test. The aggregate is one record per query
    carrying the mean of each field over the seeds, and a query that any seed is
    missing makes the cell incomplete rather than shorter.
    """
    seeds = sorted(per_seed)
    if not seeds:
        raise ValueError("aggregate_seeds_per_query needs at least one seed")
    if not registered_seeds:
        raise ValueError("aggregate_seeds_per_query needs the registered seed set (read it "
                         "from the manifest with registered_seeds()); replicates cannot be "
                         "aggregated against an assumed set")
    wanted = tuple(int(seed) for seed in registered_seeds)
    if tuple(int(seed) for seed in seeds) != tuple(sorted(wanted)):
        raise ValueError(f"the cell carries seeds {seeds} but the registered seed set is "
                         f"{list(sorted(wanted))}; an incomplete or extended set of "
                         "replicates may not be aggregated")
    by_seed = {}
    for seed in seeds:
        indexed = {}
        for record in per_seed[seed]:
            identity = str(record["query_id"])
            if identity in indexed:
                raise ValueError(f"seed {seed} scores query {identity!r} twice")
            indexed[identity] = record
        by_seed[seed] = indexed

    reference = by_seed[seeds[0]]
    out = []
    for identity in reference:
        rooms = set()
        values = {field: [] for field in fields}
        for seed in seeds:
            record = by_seed[seed].get(identity)
            if record is None:
                raise ValueError(f"query {identity!r} is missing from seed {seed}; the cell is "
                                 "incomplete and may not be aggregated")
            rooms.add(str(record["room_id"]))
            for field in fields:
                values[field].append(float(record[field]))
        if len(rooms) != 1:
            raise ValueError(f"query {identity!r} is in different rooms across seeds: "
                             f"{sorted(rooms)}")
        entry = {"query_id": identity, "room_id": rooms.pop(), "n_seeds": len(seeds),
                 "seeds": list(seeds)}
        entry.update({field: float(np.mean(values[field])) for field in fields})
        out.append(entry)
    for seed in seeds[1:]:
        extra = sorted(set(by_seed[seed]) - set(reference))
        if extra:
            raise ValueError(f"seed {seed} scores queries no other seed does: {extra[:5]}")
    return out


#: The confirmatory family, fixed before any number is seen: top-1 only, the two
#: treatment-vs-control contrasts, in both context regimes (plan B3).
CONFIRMATORY_CONTRASTS = (("BF", "P1", "K8"), ("BF", "P1", "K1"),
                          ("YAW", "P1", "K8"), ("YAW", "P1", "K1"))
CONFIRMATORY_ENDPOINT = "top1"


def contrast_label(treatment, control, regime):
    return f"{treatment}_vs_{control}_{regime}"


def build_holm_family(p_values, alpha=0.05):
    """Holm over EXACTLY the four registered contrasts -- no more, no fewer.

    A family that can grow or shrink after the numbers are seen is not a
    correction, so an unexpected label and a missing one are both refusals.
    """
    from src.localization.rir_metrics import holm_bonferroni

    registered = {contrast_label(*contrast) for contrast in CONFIRMATORY_CONTRASTS}
    supplied = set(p_values)
    unexpected = sorted(supplied - registered)
    if unexpected:
        raise ValueError(f"{unexpected} is not a registered contrast; the confirmatory family "
                         f"is exactly {sorted(registered)}")
    if supplied != registered:
        raise ValueError(f"the confirmatory family must carry exactly {len(registered)} "
                         f"contrasts, got {len(supplied)} (missing "
                         f"{sorted(registered - supplied)})")
    family = holm_bonferroni({label: float(value) for label, value in p_values.items()},
                             alpha=alpha)
    family["endpoint"] = CONFIRMATORY_ENDPOINT
    family["contrasts"] = [contrast_label(*c) for c in CONFIRMATORY_CONTRASTS]
    family["note"] = ("top-1 is the sole confirmatory endpoint; e_loc and every other metric "
                      "are supportive and are never added to this family")
    return family
