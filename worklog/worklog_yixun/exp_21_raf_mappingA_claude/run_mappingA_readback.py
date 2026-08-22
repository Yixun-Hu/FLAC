"""exp_21 readback rung: re-derive placements and microphone correspondence.

A thin harness over the TESTED components in data/RAF/ (mappingA_common,
prepare_data). It reads the real corpus READ-ONLY and writes exactly one artifact,
mappingA_correspondence_record.json, into this worklog folder.

What it answers (plan Rev 2 section 2, Codex M2):
  * how many PHYSICAL placements each room really has, re-derived by complete
    linkage rather than assumed from the informal 139/121 centroid-rounding counts;
  * for every tx-group, whether its 36 microphones correspond one-to-one to its
    placement's medoid template, with displacement p50/p95/max and the ambiguity
    margin recorded either way;
  * how many placements are ELIGIBLE (>= 9 passing, source-xyz-distinct groups);
  * whether the registered n_items = 16 x 36 x 2 identity is achievable
    (>= 16 eligible placements per room).

Usage:
    python worklog/.../run_mappingA_readback.py --raf-root /path/to/raf_dataset
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
_RAF_DIR = os.path.join(_REPO_ROOT, "data", "RAF")
if _RAF_DIR not in sys.path:
    sys.path.insert(0, _RAF_DIR)

import mappingA_common as mac  # noqa: E402
import prepare_data as raf_prepare  # noqa: E402

MIN_ELIGIBLE_GROUPS = 9          # per placement, source-xyz-distinct and passing
REQUIRED_PLACEMENTS = 16         # per room, for the registered 1,152-item identity


def audit_room(raf_root, room):
    room_dir = os.path.join(raf_root, "archived", room)
    print(f"[{room}] reading {room_dir}", flush=True)
    index = raf_prepare.load_room_index(room_dir)
    groups, group_report = raf_prepare.group_captures(index, allow_nonuniform=True)
    print(f"[{room}] {len(index)} captures in {len(groups)} tx-groups", flush=True)

    # Only exactly-36 groups can carry a 36-way correspondence; the others are
    # recorded as excluded rather than silently skipped (exp_19 R1's rule).
    sized, wrong_size = [], []
    for group in groups:
        (sized if group["size"] == mac.CANONICAL_ARRAY_SIZE else wrong_size).append(group)
    if wrong_size:
        print(f"[{room}] {len(wrong_size)} groups are not 36 captures -> excluded",
              flush=True)

    clusters = mac.cluster_placements(sized)
    print(f"[{room}] complete linkage (cap {mac.PLACEMENT_CAP_M} m) -> "
          f"{len(clusters)} placements", flush=True)

    by_key = {g["group_key"]: g for g in sized}
    placements, eligible = [], []
    all_p50, all_p95, all_max, all_margin = [], [], [], []
    # r2 N9: the rigid residual is recorded (never gated), and duplicated receiver
    # positions are counted -- a group holding them cannot be matched at all.
    all_rigid_rms, all_rigid_max = [], []
    n_duplicate_groups = 0
    n_pass = n_fail = 0

    for cluster in clusters:
        template = cluster["template_rx"]
        members, passing_xyz = [], {}
        for key in cluster["member_keys"]:
            group = by_key[key]
            report = mac.match_mics(template, group["rx_xyz_p"])
            members.append({
                "group_key": key,
                "passed": report["passed"],
                "reasons": report["reasons"],
                "p50_m": report["p50_m"],
                "p95_m": report["p95_m"],
                "max_m": report["max_m"],
                "min_ambiguity_margin": (None
                                         if not np.isfinite(report["min_ambiguity_margin"])
                                         else report["min_ambiguity_margin"]),
                "rigid_residual_rms_m": report["rigid_residual_rms_m"],
                "rigid_residual_max_m": report["rigid_residual_max_m"],
                "rigid_rotation_deg": report["rigid_rotation_deg"],
                "rigid_translation_m": report["rigid_translation_m"],
                "n_duplicate_positions": len(report["duplicate_positions"]["group"]),
                "evidence_sha256": report["evidence_sha256"],
                "source_xyz_key": mac.source_xyz_key(group["tx_xyz"]),
            })
            all_rigid_rms.append(report["rigid_residual_rms_m"])
            all_rigid_max.append(report["rigid_residual_max_m"])
            if report["duplicate_positions"]["group"]:
                n_duplicate_groups += 1
            all_p50.append(report["p50_m"])
            all_p95.append(report["p95_m"])
            all_max.append(report["max_m"])
            if np.isfinite(report["min_ambiguity_margin"]):
                all_margin.append(report["min_ambiguity_margin"])
            if report["passed"]:
                n_pass += 1
                passing_xyz.setdefault(mac.source_xyz_key(group["tx_xyz"]), []).append(key)
            else:
                n_fail += 1

        n_distinct = len(passing_xyz)
        entry = {
            "placement_id": cluster["placement_id"],
            "medoid_key": cluster["medoid_key"],
            "centroid_p": [float(v) for v in cluster["centroid_p"]],
            "n_groups": len(cluster["member_keys"]),
            "n_passing": sum(1 for m in members if m["passed"]),
            "n_passing_source_distinct": n_distinct,
            "eligible": n_distinct >= MIN_ELIGIBLE_GROUPS,
            "members": members,
        }
        placements.append(entry)
        if entry["eligible"]:
            eligible.append(entry["placement_id"])

    def stats(values):
        if not values:
            return None
        arr = np.asarray(values, dtype=np.float64)
        return {"n": int(arr.size), "min": float(arr.min()),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "max": float(arr.max()), "mean": float(arr.mean())}

    return {
        "room": room,
        "n_captures": len(index),
        "n_groups": len(groups),
        "n_groups_size_36": len(sized),
        "excluded_wrong_size": [{"group_key": g["group_key"], "size": g["size"]}
                                for g in wrong_size],
        "size_histogram": group_report["size_histogram"],
        "informal_placement_count": group_report["n_placements"],
        "n_placements": len(clusters),
        "n_eligible_placements": len(eligible),
        "eligible_placement_ids": eligible,
        "n_groups_passing": n_pass,
        "n_groups_failing": n_fail,
        "n_groups_with_duplicate_positions": n_duplicate_groups,
        "rigid_residual_rms_m": stats(all_rigid_rms),
        "rigid_residual_max_m": stats(all_rigid_max),
        "displacement_p50_m": stats(all_p50),
        "displacement_p95_m": stats(all_p95),
        "displacement_max_m": stats(all_max),
        "ambiguity_margin": stats(all_margin),
        "placements": placements,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="exp_21 correspondence readback")
    parser.add_argument("--raf-root", default="/media/diskstation/yixunhu/raf_dataset")
    parser.add_argument("--rooms", nargs="+", default=["EmptyRoom", "FurnishedRoom"])
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "mappingA_correspondence_record.json"))
    args = parser.parse_args(argv)

    rooms = {room: audit_room(args.raf_root, room) for room in args.rooms}
    verdict_rooms = {room: payload["n_eligible_placements"] >= REQUIRED_PLACEMENTS
                     for room, payload in rooms.items()}
    record = {
        "schema_version": 1,
        "created_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "raf_root": args.raf_root,
        "algorithm_version": mac.MATCH_ALGORITHM_VERSION,
        "tolerances": {
            "placement_cap_m": mac.PLACEMENT_CAP_M,
            "match_p95_m": mac.MATCH_P95_M,
            "match_max_m": mac.MATCH_MAX_M,
            "match_ambiguity_margin": mac.MATCH_AMBIGUITY_MARGIN,
            "duplicate_tolerance_m": mac.DUPLICATE_TOLERANCE_M,
            "min_eligible_groups_per_placement": MIN_ELIGIBLE_GROUPS,
            "required_eligible_placements_per_room": REQUIRED_PLACEMENTS,
        },
        "rooms": rooms,
        "verdict": {
            "eligible_by_room": verdict_rooms,
            "n_items_identity_achievable": all(verdict_rooms.values()),
            "registered_n_items": REQUIRED_PLACEMENTS * mac.CANONICAL_ARRAY_SIZE
            * len(args.rooms),
        },
    }
    with open(args.out, "w") as f:
        json.dump(record, f, indent=2, allow_nan=False)

    print("\n=== correspondence readback ===", flush=True)
    for room, payload in rooms.items():
        print(f"{room}: {payload['n_groups']} tx-groups "
              f"({payload['n_groups_size_36']} sized) -> {payload['n_placements']} "
              f"placements (informal key said {payload['informal_placement_count']}); "
              f"{payload['n_groups_passing']} pass / {payload['n_groups_failing']} fail; "
              f"{payload['n_eligible_placements']} eligible placements", flush=True)
        for name in ("displacement_p50_m", "displacement_p95_m", "displacement_max_m",
                     "ambiguity_margin"):
            values = payload[name]
            if values:
                print(f"   {name}: median {values['p50']:.5f}, p95 {values['p95']:.5f}, "
                      f"max {values['max']:.5f} (n={values['n']})", flush=True)
    print(f"verdict: n_items identity achievable = "
          f"{record['verdict']['n_items_identity_achievable']}", flush=True)
    print(f"record -> {args.out}", flush=True)
    return 0 if record["verdict"]["n_items_identity_achievable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
