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
                                       canonical_sha256, fa_run_state)

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
                      registered_at=None):
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
    payload["arm_config_rel"] = spec["config_rel"]
    payload["arm_lineage"] = spec["lineage"]
    payload["registered_at"] = registered_at or datetime.now(timezone.utc).isoformat(
        timespec="seconds")
    if spec["cond_method"] == "fa_invariant":
        # announcement 06: the whole conditioning protocol is locked, chunk plan
        # included, so a run that partitions the orbit differently is refused.
        payload.update(fa_run_state(spec["cond_method"], frame_avg_angles=FA_ANGLES,
                                    rotate_deg=0.0,
                                    cond_autocast=payload.get("cond_autocast", "default"),
                                    chunk_plan=FA_CHUNK_PLAN))
    return payload


def metric_manifest(arm, metric_source=METRIC_SOURCE, ckpt_sha256=None,
                    protocol_digests=None, expect_metric_config=None, registered_at=None):
    """One arm's metric manifest: the scorer subdocument INHERITED, never re-tuned."""
    source = _load(metric_source)
    metric_config = source.get("metric_config")
    if not isinstance(metric_config, dict) or not metric_config:
        raise ValueError(f"{metric_source} carries no metric_config to inherit")
    if expect_metric_config is not None and metric_config != expect_metric_config:
        differing = sorted(k for k in set(metric_config) | set(expect_metric_config)
                           if metric_config.get(k) != expect_metric_config.get(k))
        raise ValueError(f"the scorer subdocument is not deep-equal to the frozen one; "
                         f"{differing} differ -- exp_20 inherits it and may not recalibrate")
    return {
        "arm": arm,
        "experiment": f"exp_20 loc_crossarm {arm} metric registration (inherited scorer)",
        "registerable": copy.deepcopy(source.get("registerable")),
        "metric_config": copy.deepcopy(metric_config),
        "inherited_from": {
            "path": str(metric_source),
            "metric_config_canonical_sha256": canonical_sha256(metric_config),
            "registerable_canonical_sha256": canonical_sha256(source.get("registerable")),
            "source_sha": source.get("source_sha"),
        },
        "seeds": list(SEEDS),
        "ckpt_sha256": ckpt_sha256,
        "protocol_manifest_digests": dict(protocol_digests or {}),
        "r2_identity_digest": source.get("r2_identity_digest"),
        "candidate_manifest_sha256": source.get("candidate_manifest_sha256"),
        "transport_caveat": TRANSPORT_CAVEAT,
        "registered_at": registered_at or datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
    }


def _sha256_json(payload):
    return hashlib.sha256(json.dumps(payload, indent=2, sort_keys=True).encode()
                          + b"\n").hexdigest()


def generate(out_dir, admissions, metric_source=METRIC_SOURCE, write=True):
    """Emit all nine manifests; returns the paths (written or planned)."""
    os.makedirs(out_dir, exist_ok=True)
    written, digests = [], {}
    for arm in sorted(ARMS):
        record = admissions.get(arm)
        if record is None:
            raise ValueError(f"no admission record for arm {arm!r}; a manifest may not be "
                             "written before its checkpoint is admitted")
        digests[arm] = {}
        for regime in REGIMES:
            payload = protocol_manifest(arm, regime, ckpt_sha256=record["sha256"],
                                        model_config_sha256=record["config_sha256"])
            path = os.path.join(out_dir, f"loc_crossarm_{arm}_{regime}_registration.json")
            digests[arm][regime] = _sha256_json(payload)
            if write:
                _write_json(path, payload)
            written.append(path)
    for arm in sorted(ARMS):
        payload = metric_manifest(arm, metric_source=metric_source,
                                  ckpt_sha256=admissions[arm]["sha256"],
                                  protocol_digests=digests[arm])
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
