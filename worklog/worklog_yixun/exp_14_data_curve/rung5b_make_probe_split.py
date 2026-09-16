#!/usr/bin/env python3
"""Ladder rung 5b fixture (exp_14): derive from a split a variant in which EVERY receiver keeps
exactly ONE source, so that under `restrict_to_split: true` every target's allowed context pool is
empty and the first batch must raise DatasetContractError (whole-DDP termination probe).

Copy-only. Usage: make_probe_one_source_split.py <in_split.json> <out_split.json> <out_dataset_config.json> <base_dataset_config.json>
"""
import json, sys, collections, os

def one_source_per_receiver(split):
    out = {}
    n_in = n_out = 0
    for scene in sorted(split):
        out[scene] = {}
        for room in sorted(split[scene]):
            files = sorted(split[scene][room]); n_in += len(files)
            first_by_rec = {}
            for f in files:                      # sorted => deterministic: lowest source id per receiver
                rec = f.split("_")[1]
                first_by_rec.setdefault(rec, f)
            kept = sorted(first_by_rec.values()); n_out += len(kept)
            out[scene][room] = kept
    return out, n_in, n_out

def main(argv):
    in_split, out_split, out_cfg, base_cfg = argv[1:5]
    split = json.load(open(in_split))
    probe, n_in, n_out = one_source_per_receiver(split)
    # contract: every receiver has exactly one source in the probe
    for scene, rooms in probe.items():
        for room, files in rooms.items():
            recs = collections.Counter(f.split("_")[1] for f in files)
            assert all(c == 1 for c in recs.values()), (scene, room)
    with open(out_split, "w") as f:
        json.dump(probe, f, indent=1)
    cfg = json.load(open(base_cfg))
    cfg["datasets"][0]["json_file_path"] = os.path.relpath(out_split)
    cfg["modalities"]["acoustic_context"]["restrict_to_split"] = True
    with open(out_cfg, "w") as f:
        json.dump(cfg, f, indent=4)
    print(f"probe split: {n_out} targets (from {n_in}); one source per receiver everywhere; "
          f"config -> {out_cfg} (json_file_path={cfg['datasets'][0]['json_file_path']}, restrict_to_split=True)")

if __name__ == "__main__":
    main(sys.argv)
