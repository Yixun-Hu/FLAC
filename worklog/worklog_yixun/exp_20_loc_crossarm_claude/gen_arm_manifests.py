#!/usr/bin/env python
"""exp_20 -- generate the per-arm registration manifests (plan §3, M5).

Six protocol manifests (3 arms x {R2 K_ctx=8, R2b K_ctx=1}) and three metric
manifests, all emitted from ONE place so the fields that must be identical
across arms cannot drift apart by hand:

  * the protocol manifests reuse exp_18's registered template verbatim -- the
    same split digest, candidate manifest, tau, aggregation, seeds, scorer and
    readout -- and rebind only what an arm changes: its checkpoint sha, its model
    config sha and, for BF, the whole frame-average block (announcement 06: the
    chunk plan is DECLARED).
  * the metric manifests INHERIT the scorer subdocument from exp_18's frozen
    ``loc_invert_R4_metric_registration.json`` by deep equality. Nothing is
    recalibrated per arm: the R4 constants were chosen on released-checkpoint
    seen generations, and re-tuning them per arm would make each arm's scorer a
    different instrument. The transport caveat travels in the manifest.

Usage:
    python gen_arm_manifests.py --out-dir <dir> --admissions <admission.json> [--write]
"""
import argparse
import copy
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.localization.crossarm import (ARMS, FA_ANGLES, FA_CHUNK_PLAN,  # noqa: E402
                                       REGISTERED_CANDIDATE_MICRO_BATCH, REGISTERED_STEP,
                                       canonical_sha256, fa_run_state,
                                       fa_source_shas)

STEP = REGISTERED_STEP

#: exp_18's registered templates: every non-arm field is copied from these.
EXP18 = os.path.join("worklog", "worklog_yixun", "exp_18_loc_invert_claude")
TEMPLATES = {"R2": os.path.join(EXP18, "loc_invert_R2_registration.json"),
             "R2b": os.path.join(EXP18, "loc_invert_R2b_registration.json")}
METRIC_SOURCE = os.path.join(EXP18, "loc_invert_R4_metric_registration.json")

REGIMES = ("R2", "R2b")
SEEDS = [42, 43, 44]

#: fields an ARM rebinds; everything else is inherited from the template.
ARM_BOUND_FIELDS = ("ckpt_sha256", "model_config_sha256", "cond_method", "experiment")

TRANSPORT_CAVEAT = (
    "The R4 metric constants (delta_max, the M4 mu/sigma, the T30 backend) were calibrated "
    "on the RELEASED checkpoint's seen generations. They are inherited here unchanged and "
    "used as FIXED EXTERNAL SCORERS across arms; their validity outside that calibration "
    "domain is not claimed, and no constant may be recalibrated per arm. The AGREE primary "
    "endpoint is unaffected by this caveat."
)


def _load(path):
    with open(os.path.join(_REPO_ROOT, path) if not os.path.isabs(path) else path) as handle:
        return json.load(handle)


def protocol_manifest(arm, regime, ckpt_sha256, model_config_sha256, template=None,
                      registered_at=None, admission_record_sha256=None,
                      batch_size=4, num_workers=4):
    """One arm x regime protocol manifest, exp_18's template with the arm rebound."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; registered arms are {sorted(ARMS)}")
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}; registered regimes are {REGIMES}")
    payload = copy.deepcopy(template if template is not None else _load(TEMPLATES[regime]))
    spec = ARMS[arm]
    payload["ckpt_sha256"] = ckpt_sha256
    payload["model_config_sha256"] = model_config_sha256
    payload["cond_method"] = spec["cond_method"]
    payload["seeds"] = list(SEEDS)
    payload["experiment"] = (f"exp_20 loc_crossarm {arm} {regime} "
                             f"({'K_ctx=8' if regime == 'R2' else 'K_ctx=1'}, "
                             f"matched 40k; {spec['lineage']})")
    payload["arm"] = arm
    payload["batch_size"] = int(batch_size)
    payload["num_workers"] = int(num_workers)
    payload["admission_record_sha256"] = admission_record_sha256
    payload["arm_config_rel"] = spec["config_rel"]
    payload["arm_lineage"] = spec["lineage"]
    payload["registered_at"] = registered_at or datetime.now(timezone.utc).isoformat(
        timespec="seconds")
    if spec["cond_method"] == "fa_invariant":
        # announcement 06: the whole conditioning protocol is locked, chunk plan
        # included, so a run that partitions the orbit differently is refused.
        payload["fa_source_shas"] = fa_source_shas()
        payload.update(fa_run_state(spec["cond_method"], frame_avg_angles=FA_ANGLES,
                                    rotate_deg=0.0,
                                    cond_autocast=payload.get("cond_autocast", "default"),
                                    chunk_plan=FA_CHUNK_PLAN,
                                    candidate_micro_batch=REGISTERED_CANDIDATE_MICRO_BATCH))
    return payload


#: The FROZEN scorer subdocument. Deep equality against it is not an option a
#: caller may skip: the production path checks it every time (r1 review F1).
def frozen_scorer(metric_source=METRIC_SOURCE):
    source = _load(metric_source)
    metric_config = source.get("metric_config")
    if not isinstance(metric_config, dict) or not metric_config:
        raise ValueError(f"{metric_source} carries no metric_config to inherit")
    return source, metric_config


def assert_scorer_is_frozen(metric_config, reference=None):
    """Refuse any drift from exp_18's frozen constants -- always, not on request."""
    if reference is None:
        _reference_source, reference = frozen_scorer(METRIC_SOURCE)
    if metric_config != reference:
        differing = sorted(key for key in set(metric_config) | set(reference)
                           if metric_config.get(key) != reference.get(key))
        raise ValueError(f"the scorer subdocument is not deep-equal to the frozen one; "
                         f"{differing} differ -- exp_20 inherits it and may not recalibrate")
    return metric_config


def metric_manifest(arm, metric_source=METRIC_SOURCE, ckpt_sha256=None,
                    protocol_digests=None, expect_metric_config=None, registered_at=None,
                    admission_record_sha256=None):
    """One arm's metric manifest: the scorer subdocument INHERITED, never re-tuned.

    The emitted document has the shape the FROZEN verifier requires -- top-level
    ``source_sha`` and ``r2_manifest_digests`` keyed by committed repository
    paths -- because a manifest that cannot pass ``verify_metric_registration``
    would refuse every metrics-inline unseen cell (r1 review F1).
    """
    source, metric_config = frozen_scorer(metric_source)
    assert_scorer_is_frozen(metric_config, expect_metric_config)
    return {
        "arm": arm,
        "experiment": f"exp_20 loc_crossarm {arm} metric registration (inherited scorer)",
        "registerable": copy.deepcopy(source.get("registerable")),
        "metric_config": copy.deepcopy(metric_config),
        # the verifier reads these at the TOP level
        "source_sha": source.get("source_sha"),
        "r2_manifest_digests": copy.deepcopy(source.get("r2_manifest_digests") or {}),
        "r2_identity_digest": source.get("r2_identity_digest"),
        "candidate_manifest_sha256": source.get("candidate_manifest_sha256"),
        "seeds": list(SEEDS),
        "inherited_from": {
            "path": str(metric_source),
            "metric_config_canonical_sha256": canonical_sha256(metric_config),
            "registerable_canonical_sha256": canonical_sha256(source.get("registerable")),
            "source_sha": source.get("source_sha"),
        },
        "ckpt_sha256": ckpt_sha256,
        "protocol_manifest_digests": dict(protocol_digests or {}),
        "admission_record_sha256": admission_record_sha256,
        "transport_caveat": TRANSPORT_CAVEAT,
        "registered_at": registered_at or datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
    }


#: what an admission record must SAY before a manifest may be written from it.
def verify_admission_record(arm, record):
    """Re-check the record's facts; ``admitted: true`` is a claim, not evidence."""
    reasons = []
    spec = ARMS[arm]
    if not record.get("admitted"):
        reasons.append(f"arm {arm} is not admitted: {record.get('reasons')}")
    if str(record.get("arm")) != arm:
        reasons.append(f"the record names arm {record.get('arm')!r}, not {arm!r}")
    if str(record.get("config_path")) != spec["config_rel"]:
        reasons.append(f"the record's config path {record.get('config_path')!r} is not the "
                       f"registered {spec['config_rel']!r}")
    if record.get("global_step") != STEP:
        reasons.append(f"the record's global_step is {record.get('global_step')!r}, not the "
                       f"registered endpoint {STEP}")
    ema, online = record.get("ema_key_count"), record.get("online_model_key_count")
    if not ema or ema != online:
        reasons.append(f"the record's EMA inventory is {ema!r}/{online!r}; a complete mirror "
                       "is required")
    integrity = record.get("load_integrity") or {}
    if not integrity.get("clean") or integrity.get("n_missing") or integrity.get("n_stray"):
        reasons.append(f"the record's load integrity is {integrity!r}, not 0 missing / 0 stray")
    if str(record.get("cond_method")) != spec["cond_method"]:
        reasons.append(f"the record's cond_method {record.get('cond_method')!r} is not arm "
                       f"{arm}'s registered {spec['cond_method']!r}")
    if not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64:
        reasons.append("the record carries no sha256 for the checkpoint")
    if reasons:
        raise ValueError(f"admission record for {arm} is not usable: " + "; ".join(reasons))
    return canonical_sha256(record)


def _sha256_json(payload):
    return hashlib.sha256(json.dumps(payload, indent=2, sort_keys=True).encode()
                          + b"\n").hexdigest()


def generate(out_dir, admissions, metric_source=METRIC_SOURCE, write=True):
    """Emit all nine manifests; returns the paths (written or planned)."""
    os.makedirs(out_dir, exist_ok=True)
    written, digests = [], {}
    record_digests = {}
    for arm in sorted(ARMS):
        record = admissions.get(arm)
        if record is None:
            raise ValueError(f"no admission record for arm {arm!r}; a manifest may not be "
                             "written before its checkpoint is admitted")
        record_digests[arm] = verify_admission_record(arm, record)
        digests[arm] = {}
        for regime in REGIMES:
            payload = protocol_manifest(arm, regime, ckpt_sha256=record["sha256"],
                                        model_config_sha256=record["config_sha256"],
                                        admission_record_sha256=record_digests[arm])
            path = os.path.join(out_dir, f"loc_crossarm_{arm}_{regime}_registration.json")
            digests[arm][regime] = _sha256_json(payload)
            if write:
                _write_json(path, payload)
            written.append(path)
    for arm in sorted(ARMS):
        payload = metric_manifest(arm, metric_source=metric_source,
                                  ckpt_sha256=admissions[arm]["sha256"],
                                  protocol_digests=digests[arm],
                                  admission_record_sha256=record_digests[arm])
        path = os.path.join(out_dir, f"loc_crossarm_{arm}_metric_registration.json")
        if write:
            _write_json(path, payload)
        written.append(path)
    return written


def _write_json(path, payload):
    with open(path + ".partial", "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(path + ".partial", path)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--admissions", required=True,
                        help="JSON: {arm: admission record} from crossarm.admit_checkpoint")
    parser.add_argument("--metric-source", default=METRIC_SOURCE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    admissions = _load(args.admissions)
    for arm, record in admissions.items():
        if not record.get("admitted"):
            raise SystemExit(f"arm {arm} is not admitted; refusing to write its manifests: "
                             f"{record.get('reasons')}")
    written = generate(args.out_dir, admissions, metric_source=args.metric_source,
                       write=not args.dry_run)
    for path in written:
        print(("planned " if args.dry_run else "wrote ") + path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
