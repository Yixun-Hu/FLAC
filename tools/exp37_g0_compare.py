#!/usr/bin/env python3
"""exp_37 gate G0 (plan_loc_efficiency.md §10 "G0 pinned"): one G0 run vs the latency bundle's K=1-only query.

Pinned: query index 4467 (``single_channel_ir_1/Bathrooms/Bathrooms_idx_14/S009_R017_hybrid_IR.wav``, 27
candidates, candidate hash ``4f212dcb...86bf1``), bundle file ``query_04467.npz`` of ``repeat_001/<arm>``; K_gen 1,
seed 42, batch 64, tau 0.1, the bundle's timing flags. PASS needs: candidate arrays EXACTLY equal, the same argmax,
max |delta similarity| <= 1e-3 (fixed; never relaxed post hoc); exact equality is reported when it holds. YAW
(byte-identical checkpoint) runs first as the environment / backend control: P1 / B-F are REFUSED without a YAW
PASS verdict; YAW FAIL => environment / backend parity failed; YAW PASS but P1 / B-F FAIL => wrapped-vs-extracted
EMA parity suspect. Exit 0 PASS, 1 FAIL, 2 REFUSED (a malformed, tampered or non-pinned input). The verdict JSON
(canonical sha256) is written once and never overwritten. Written by the exp_37 Coder (Claude Opus 5.5), round B-a.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
sys.path.insert(0, str(REPO_ROOT))

from src.localization.pilot import canonical_sha256  # noqa: E402
from src.localization.scoring import log_mean_exp_scores, stable_argmax  # noqa: E402

BUNDLE_REPEAT = REPO_ROOT / "NeurIPS_2026_Workshop_Latency_Reproducibility" / "data" / "repeats" / "repeat_001"
AUDIT_REL = "worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/geometry_audit.json"
LAUNCHER_REL = "overlays/no_flash/exp37_no_flash_launch.py"
SHARED_FIELDS = ("model_config_sha256", "agree_checkpoint_sha256", "context_manifest_sha256", "geometry_audit_sha256",
                 "pilot_manifest_sha256", "n_context", "score_sample_counts", "tau", "sample_seed",
                 "candidate_batch_size", "synchronize_timing", "warmup_query_count", "measure_core_forward", "sampler")
C4 = [0.0, 90.0, 180.0, 270.0]


@dataclasses.dataclass(frozen=True)
class Pins:
    query_index: int
    query_id: str
    candidate_hash: str
    candidate_count: int
    tolerance: float
    overlay_sha256: str
    arms: dict


PINS = Pins(4467, "single_channel_ir_1/Bathrooms/Bathrooms_idx_14/S009_R017_hybrid_IR.wav",
            "4f212dcbf53bea4e604969f50c2bc8f3f6330e9d383375a46b0a4563efd86bf1", 27, 1e-3,
            "0d0e52ca693e236a2e0a44cd61f61539fdf995cff5132d1f621181a0f3e7c023", {
                "YAW": {"reference_dir": "yawaug_flac", "cond": "vanilla", "angles": None,
                        "local_ckpt": "ac1f26034e4f341fe0c2cb4638e2eb473959d66ddd2fd95d184dc2fd4f264de7",
                        "reference_ckpt": "ac1f26034e4f341fe0c2cb4638e2eb473959d66ddd2fd95d184dc2fd4f264de7",
                        "reference_manifest_sha256": "fc4b2c77531ddf0855668d4b59e803801eeddd18dcff17b7ab9d304fbacd50ab"},
                "P1": {"reference_dir": "vanilla_flac", "cond": "vanilla", "angles": None,
                       "local_ckpt": "c4c678826cddda37fa4977926aadee530afd037b3abb110918b52a342ce9845c",
                       "reference_ckpt": "da12748586912c5fe9683a6d27b2507ff13c0a89c458abcbdc63aecd4f35c643",
                       "reference_manifest_sha256": "b5b50134d5d727580408d7f4cd7920a066548355691d60b2da3a44e235879f26"},
                "BF": {"reference_dir": "fa_bf_flac", "cond": "fa_invariant", "angles": C4,
                       "local_ckpt": "5319feb4af874624859e87105ddd8ab06d4b449769d1e054f712b2b1c0542328",
                       "reference_ckpt": "0f61277f45367fb0e75d7ee70c0627b8948a23eb62be58f13fce91662551557a",
                       "reference_manifest_sha256": "e29f9431ec71d5962c2eeeea3835899b61365f7b258db7fdd32dcd2bd8e864b2"}})


class Refused(Exception):
    pass


def _sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hashed(path: Path, label: str) -> dict:
    if not path.is_file():
        raise Refused(f"{label}: {path} is missing")
    payload = json.loads(path.read_text())
    if payload.get("sha256") != canonical_sha256({k: v for k, v in payload.items() if k != "sha256"}):
        raise Refused(f"{label}: {path} fails its own sha256")
    return payload


def _load_run(run_dir: Path, label: str, pins: Pins) -> dict:
    """A runner output directory holding the pinned query: manifest, query JSON, arrays -- all self-consistent."""
    manifest = _hashed(run_dir / "run_manifest.json", label)
    stem = f"query_{pins.query_index:05d}"
    query = _hashed(run_dir / "queries" / f"{stem}.json", f"{label} (pinned query_index {pins.query_index})")
    npz = run_dir / "queries" / f"{stem}.npz"
    pinned = {"query_index": pins.query_index, "query_id": pins.query_id, "candidate_count": pins.candidate_count,
              "candidate_indices_sha256": pins.candidate_hash, "run_manifest_sha256": manifest["sha256"],
              "n_context": 8, "score_sample_counts": [1]}
    bad = [f"{label}: query {k} is {query.get(k)!r}, not {v!r}" for k, v in pinned.items() if query.get(k) != v]
    if not npz.is_file() or query.get("arrays_sha256") != _sha_file(npz):
        bad.append(f"{label}: {npz.name} does not match the query's arrays_sha256")
    if bad:
        raise Refused("; ".join(bad))
    with np.load(npz) as arrays:
        candidates, similarities = arrays["candidates"], arrays["similarities"]
    shapes = {"candidates": (pins.candidate_count, 3), "similarities": (pins.candidate_count, 1)}
    for name, array in (("candidates", candidates), ("similarities", similarities)):
        if array.shape != shapes[name] or array.dtype != np.float32 or not np.isfinite(array).all():
            raise Refused(f"{label}: {name} is {array.dtype}{array.shape}, not finite float32{shapes[name]}")
    scores = log_mean_exp_scores(torch.from_numpy(similarities), (1,), tau=float(manifest["identity"]["tau"]))[1]
    winner = stable_argmax(scores)
    if query.get("metrics", {}).get("1", {}).get("prediction_index") != winner:
        raise Refused(f"{label}: the recorded prediction_index differs from the argmax {winner} of its own arrays")
    ordered = np.sort(scores.numpy())[::-1]
    return {"identity": manifest["identity"], "manifest_sha256": manifest["sha256"], "candidates": candidates,
            "similarities": similarities, "argmax": winner, "margin": float(ordered[0] - ordered[1]),
            "npz_sha256": query["arrays_sha256"]}


def load_reference(arm: str, reference_root: Path, pins: Pins = PINS) -> dict:
    spec = pins.arms[arm]
    ref = _load_run(Path(reference_root) / spec["reference_dir"], f"bundle {arm}", pins)
    identity = ref["identity"]
    if ref["manifest_sha256"] != spec["reference_manifest_sha256"]:
        raise Refused(f"bundle {arm}: run manifest {ref['manifest_sha256']} is not the pinned one")
    if (identity["checkpoint_sha256"], identity["conditioning_method"]) != (spec["reference_ckpt"], spec["cond"]):
        raise Refused(f"bundle {arm}: checkpoint / conditioning_method are not the pinned ones")
    if pins.query_index not in identity["query_indices"]:
        raise Refused(f"bundle {arm}: query_indices do not hold {pins.query_index}")
    return ref


def _identity_reasons(arm, ours, ref, audit, launcher_sha256, pins) -> list[str]:
    spec, mine = pins.arms[arm], ours["identity"]
    want = {**{k: ref["identity"].get(k) for k in SHARED_FIELDS}, "query_indices": [pins.query_index],
            "checkpoint_sha256": spec["local_ckpt"], "conditioning_method": spec["cond"],
            "frame_average_angles": spec["angles"], "attention_backend": "sdpa",
            "no_flash_launch": {"overlay_sha256": pins.overlay_sha256, "launcher_sha256": launcher_sha256,
                                "sentinel": "EXP37_NO_FLASH_LAUNCHER"}}
    reasons = [f"identity {k} is {mine.get(k)!r}, not {v!r}" for k, v in want.items() if mine.get(k) != v]
    if audit["sha256"] != canonical_sha256({k: v for k, v in audit.items() if k != "sha256"}) \
            or mine.get("geometry_audit_sha256") != audit["sha256"]:
        reasons.append("the geometry audit fails its sha256 or is not the run's")
    override = mine.get("mesh_root_override") or {}
    digests = {room: row.get("mesh_sha256") for room, row in override.get("rooms", {}).items()}
    if override.get("audit_mesh_root") != audit["mesh_root"] or digests != {
            room: row["mesh_sha256"] for room, row in audit["rooms"].items()}:
        reasons.append("identity mesh_root_override is absent or its digest map is not the audit's")
    return reasons


def _yaw_reasons(arm, yaw, launcher_sha256, reference_root, pins) -> list[str]:
    if arm == "YAW":
        return []
    if yaw is None:
        return [f"{arm} needs the YAW environment-control verdict first (--yaw-verdict)"]
    want = {"gate": "exp37_G0", "arm": "YAW", "verdict": "PASS", "attention_backend": "sdpa",
            "overlay_sha256": pins.overlay_sha256, "launcher_sha256": launcher_sha256,
            "reference_root": str(reference_root)}
    bad = [f"YAW verdict {k} is {yaw.get(k)!r}, not {v!r}" for k, v in want.items() if yaw.get(k) != v]
    if yaw.get("sha256") != canonical_sha256({k: v for k, v in yaw.items() if k != "sha256"}):
        bad.append("the YAW verdict fails its own sha256")
    return bad


def compare(arm: str, run_dir: Path, reference_root: Path, *, audit: dict, launcher_sha256: str,
            yaw_verdict: dict | None = None, pins: Pins = PINS) -> dict:
    verdict = {"gate": "exp37_G0", "arm": arm, "run_dir": str(run_dir), "reference_root": str(reference_root),
               "query_index": pins.query_index, "tolerance": pins.tolerance, "attention_backend": "sdpa",
               "overlay_sha256": pins.overlay_sha256, "launcher_sha256": launcher_sha256,
               "comparator_sha256": _sha_file(Path(__file__))}
    reasons = _yaw_reasons(arm, yaw_verdict, launcher_sha256, reference_root, pins)
    try:
        ref = load_reference(arm, reference_root, pins)
        ours = _load_run(Path(run_dir), f"G0 {arm} run", pins)
        reasons += _identity_reasons(arm, ours, ref, audit, launcher_sha256, pins)
    except (Refused, KeyError, ValueError, OSError) as error:
        reasons.append(f"{type(error).__name__}: {error}")
    if reasons:
        verdict.update(verdict="REFUSED", exit_code=2, reasons=reasons,
                       diagnosis="not a valid G0 comparison; nothing is concluded")
        return {**verdict, "sha256": canonical_sha256(verdict)}
    delta = np.abs(ours["similarities"].astype(np.float64) - ref["similarities"].astype(np.float64))[:, 0]
    fails = []
    if ours["candidates"].tobytes() != ref["candidates"].tobytes():
        fails.append("candidate arrays differ (exact comparison)")
    if ours["argmax"] != ref["argmax"]:
        fails.append(f"argmax {ours['argmax']} != bundle argmax {ref['argmax']}")
    if float(delta.max()) > pins.tolerance:
        fails.append(f"max |delta similarity| {float(delta.max()):.3e} > {pins.tolerance:g}")
    if arm == "YAW":
        diagnosis = ("environment / backend parity FAILED (hard stop): the byte-identical YAW checkpoint does not "
                     "reproduce the bundle") if fails else "environment / backend parity established by the YAW control"
    else:
        diagnosis = (f"wrapped-vs-extracted EMA parity suspect for {arm} (the YAW control passed, so the environment / "
                     "backend is not the cause)") if fails else f"wrapped-checkpoint EMA parity established for {arm}"
    verdict.update(verdict="FAIL" if fails else "PASS", exit_code=1 if fails else 0, reasons=fails, diagnosis=diagnosis,
                   exact_equality=bool(np.array_equal(ours["similarities"], ref["similarities"])),
                   max_abs_delta=float(delta.max()), deltas=[float(v) for v in delta],
                   argmax={"ours": ours["argmax"], "reference": ref["argmax"]},
                   top_two_margin={"ours": ours["margin"], "reference": ref["margin"]},
                   run_manifest_sha256=ours["manifest_sha256"], reference_manifest_sha256=ref["manifest_sha256"],
                   npz_sha256={"ours": ours["npz_sha256"], "reference": ref["npz_sha256"]},
                   yaw_verdict_sha256=None if yaw_verdict is None else yaw_verdict.get("sha256"))
    return {**verdict, "sha256": canonical_sha256(verdict)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", choices=tuple(PINS.arms), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--yaw-verdict", type=Path)
    parser.add_argument("--reference-root", type=Path, default=BUNDLE_REPEAT)
    args = parser.parse_args()
    if args.output.exists():
        print(f"G0 REFUSED: {args.output} exists (verdicts are written once)", file=sys.stderr)
        return 2
    yaw = json.loads(args.yaw_verdict.read_text()) if args.yaw_verdict else None
    verdict = compare(args.arm, args.run_dir.resolve(), args.reference_root.resolve(),
                      audit=json.loads((REPO_ROOT / AUDIT_REL).read_text()),
                      launcher_sha256=_sha_file(REPO_ROOT / LAUNCHER_REL), yaw_verdict=yaw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(f"G0 {args.arm}: {verdict['verdict']} (exit {verdict['exit_code']}) -- {verdict['diagnosis']}; "
          f"max|delta|={verdict.get('max_abs_delta')} exact={verdict.get('exact_equality')} reasons={verdict['reasons']}")
    return int(verdict["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
