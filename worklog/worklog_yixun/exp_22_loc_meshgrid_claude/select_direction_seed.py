#!/usr/bin/env python
"""exp_22 -- run the registered anchor-driven direction-set selection.

exp_22 is self-authoritative about the 31 directions (Yixun, 2026-08-25): they
exist only to test interior free space, the metadata anchors are known-interior
points, and the classifier must therefore agree with every one of them. The
registered rule is fixed in advance and applied here:

    the smallest generator seed s >= 0 whose build_directions(31, seed=s) set
    gives strict-majority odd parity (>= 16 of 31) for EVERY metadata source and
    receiver anchor in ALL 16 required rooms.

The sweep classifies only anchors -- about 116 points per room -- so the cost is
loading each mesh once. Prints the winning seed, its digest and the literal
array ready to paste into ``meshgrid_geometry.FROZEN_DIRECTIONS_LITERAL``.

Usage:
    python select_direction_seed.py --mesh-root <.../room_mesh_obj_format> \\
        [--metadata-root AcousticRooms/metadata] [--max-seed 64] [--out report.json]
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.localization import meshgrid_geometry as mg          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_meshgrid_geometry import REQUIRED_ROOMS, resolve_room_meshes  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mesh-root", required=True)
    parser.add_argument("--metadata-root", default=os.path.join("AcousticRooms", "metadata"))
    parser.add_argument("--max-seed", type=int, default=64)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    meshes = resolve_room_meshes(REQUIRED_ROOMS, args.mesh_root)
    print(f"resolved {len(meshes)} meshes; loading (once each)...")
    scenes = mg.anchor_scenes(REQUIRED_ROOMS, args.mesh_root, args.metadata_root,
                              resolve=lambda room, _root: meshes[room])
    total = sum(entry["sources"].shape[0] + entry["receivers"].shape[0]
                for entry in scenes.values())
    print(f"{len(scenes)} rooms, {total} anchors to classify per seed")

    # provenance first: what does the CURRENTLY pinned set do on these anchors?
    current = mg.evaluate_direction_seed(mg.FROZEN_DIRECTIONS_SEED, scenes)
    print(f"currently pinned seed {mg.FROZEN_DIRECTIONS_SEED}: ok={current['ok']} "
          f"min_votes={current['min_votes']} failures={current['n_failures']}")
    for failure in current["failures"]:
        print(f"    FAIL {failure['room_id']} {failure['kind']} {failure['point']} "
              f"{failure['odd_votes']}/31")
    if current["directions_sha256"] != mg.FROZEN_DIRECTIONS_SHA256:
        print(f"    NOTE: build_directions(31, seed={mg.FROZEN_DIRECTIONS_SEED}) hashes to "
              f"{current['directions_sha256'][:16]}..., the pinned literals to "
              f"{mg.FROZEN_DIRECTIONS_SHA256[:16]}... -- the pin is NOT that seed")

    def announce(report):
        print(f"  seed {report['seed']:>3}: ok={report['ok']} min_votes={report['min_votes']} "
              f"failures={report['n_failures']}")

    selection = mg.select_direction_seed(scenes, max_seed=args.max_seed, on_seed=announce)
    directions = selection["directions"]
    digest = selection["report"]["directions_sha256"]
    print(f"\nSELECTED seed {selection['seed']} | digest {digest}")
    print(f"min votes over all anchors: {selection['report']['min_votes']}/31")
    print("\nFROZEN_DIRECTIONS_LITERAL = (")
    for row in directions:
        print("    (%r, %r, %r)," % tuple(float(v) for v in row))
    print(")")

    payload = {
        "rule": mg.DIRECTION_SELECTION_RULE,
        "selected_seed": selection["seed"], "directions_sha256": digest,
        "directions": [[float(v) for v in row] for row in directions],
        "attempts": selection["attempts"],
        "previous_pin": {"seed": mg.FROZEN_DIRECTIONS_SEED,
                         "sha256": mg.FROZEN_DIRECTIONS_SHA256,
                         "evaluation": current},
        "rooms": sorted(scenes), "n_anchors": int(total),
        "report": selection["report"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"\nreport -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
