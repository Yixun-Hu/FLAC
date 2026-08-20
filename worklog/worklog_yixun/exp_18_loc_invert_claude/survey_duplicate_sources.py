#!/usr/bin/env python
"""exp_18 probe: which rooms of a split have two source labels at one position?

This is the Planner's R0-abort survey, promoted from a shell one-liner to
reviewed tooling (universal review coverage). It reuses the reviewed library
code only -- ``enumerate_metadata_sources`` for the authority and
``merge_position_duplicates`` for the grouping -- and touches nothing else, so it
cannot disagree with what the driver does.

Measured on 2026-08-19: 2 of 131 SEEN rooms are affected
(``Bathrooms_idx_11``: S9 == S10 at [4.87, 2.36, 1.87]; ``Bathrooms_idx_16``:
S4 == S7 at [2.33, -3.07, 1.70]); all 17 UNSEEN rooms are clean, which is why the
registered headline protocol is untouched by plan Rev 3.2.

    python worklog/worklog_yixun/exp_18_loc_invert_claude/survey_duplicate_sources.py \\
        --split data/AR/seen_eval.json --dataset-root AcousticRooms
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from src.localization.candidates import (  # noqa: E402
    enumerate_metadata_sources, merge_position_duplicates)


def survey_split(split_path, dataset_root):
    """Per room: metadata sources, unique positions, and the merged groups."""
    with open(split_path) as handle:
        split = json.load(handle)

    rooms, dirty = {}, {}
    for scene in sorted(split):
        for scene_id in sorted(split[scene]):
            room_id = f"{scene}/{scene_id}"
            meta_dir = os.path.join(dataset_root, "metadata", scene, scene_id)
            if not os.path.isdir(meta_dir):
                rooms[room_id] = {"error": f"no metadata directory at {meta_dir}"}
                continue
            try:
                sources = enumerate_metadata_sources(meta_dir, allow_duplicate_positions=True)
            except ValueError as err:                      # cross-receiver drift, not duplicates
                rooms[room_id] = {"error": str(err)}
                continue
            merged, groups = merge_position_duplicates(sources)
            duplicates = {str(canonical): members for canonical, members in groups.items()
                          if len(members) > 1}
            rooms[room_id] = {"n_sources": len(sources), "n_positions": len(merged),
                              "merge_map": duplicates}
            if duplicates:
                dirty[room_id] = {canonical: [
                    [float(v) for v in sources[int(canonical)]], members]
                    for canonical, members in duplicates.items()}
    return {"split": str(split_path), "dataset_root": str(dataset_root),
            "n_rooms": len(rooms), "n_rooms_with_duplicates": len(dirty),
            "rooms": rooms, "duplicates": dirty}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", required=True, help="split JSON, e.g. data/AR/seen_eval.json")
    parser.add_argument("--dataset-root", default="AcousticRooms")
    parser.add_argument("--out", default=None, help="write the full report as JSON")
    args = parser.parse_args(argv)

    report = survey_split(args.split, args.dataset_root)
    print(f"{report['n_rooms_with_duplicates']}/{report['n_rooms']} rooms have "
          "two source labels at one position")
    for room_id, groups in sorted(report["duplicates"].items()):
        for canonical, (xyz, members) in sorted(groups.items()):
            print(f"  {room_id}: S{members[0]} == " + " == ".join(f"S{m}" for m in members[1:])
                  + f" at {xyz} (canonical S{canonical})")
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(report, handle, sort_keys=True, indent=2)
            handle.write("\n")
        print(f"report -> {args.out}")
    return report


if __name__ == "__main__":
    main()
