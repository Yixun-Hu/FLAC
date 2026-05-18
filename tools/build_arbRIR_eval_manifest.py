"""
Frozen-manifest pre-pass for the arbitrary-RIR eval (plan:
plan/eval_arbRIR_v0_vs_baseline_K1_K8.md).

Why this exists: `eval_FLAC.py` runs a manual dataloader loop with no
`worker_init_fn` (`src/data/dataset.py:411`), so numpy is unseeded in worker
processes and `np.random.choice` inside the metadata module is NOT reproducible
run-to-run / across the A↔B module swap. Doing the source-excluded sampling
*once*, single-threaded, with an explicit `np.random.RandomState(42)`, and
freezing the picks to disk makes "every eval run sees identical context" a
structural guarantee instead of a probabilistic hope.

Output: one JSON manifest per split, keyed by the **scene triple**
`"<scene>/<scene_id>/<filename_no_ext>"` — exactly what the eval module parses
from `info['relpath']` via `split("/")[-3:]` (see Manifest-key contract in the
plan), so build-time and dataloader-time keys match independent of root-path
conventions. Value = list of K=8 context wav filenames; K=1 is the prefix.

This replicates `AR_md_arbRIR_v0.py:sample_arbitrary_context`'s candidate
enumeration + source-exclusion + `np.random.choice`, with one deliberate
difference: `sorted(os.listdir(...))` (not the bare `os.listdir` at
AR_md_arbRIR_v0.py:137) so the build is idempotent regardless of filesystem
listing order.

Usage:
    python tools/build_arbRIR_eval_manifest.py                # both splits
    python tools/build_arbRIR_eval_manifest.py --split seen
    python tools/build_arbRIR_eval_manifest.py --force        # overwrite
"""
import argparse
import json
import os
import sys

import numpy as np

# AcousticRooms layout (matches dataset config path / folder_name and
# json_scandir's os.path.join(dir, folder_name, scene, sub_scene, fn)).
AR_ROOT = "AcousticRooms"
FOLDER = "single_channel_ir_1"
SEED = 42
K = 8

SPLIT_FILES = {
    "seen": "data/AR/seen_eval.json",
    "unseen": "data/AR/unseen_eval.json",
}
OUT_TMPL = "data/AR/arbRIR_v0_eval_manifest_{split}.json"


def src_node_of(fn):
    """'S008_R089_hybrid_IR.wav' -> 8. Mirrors AR_md_arbRIR_v0.py parsing
    (`int(parts[0][1:])`). Returns None on parse failure (skip, like the
    module's try/except)."""
    try:
        return int(fn.split("_")[0][1:])
    except (ValueError, IndexError):
        return None


def build_split(split_name, force):
    split_json = SPLIT_FILES[split_name]
    out_path = OUT_TMPL.format(split=split_name)

    if os.path.exists(out_path) and not force:
        raise SystemExit(
            f"[abort] {out_path} already exists. It is a frozen artifact — "
            f"pass --force only if you intend to regenerate it (this changes "
            f"the eval context for ALL runs)."
        )

    with open(split_json) as f:
        split = json.load(f)

    rs = np.random.RandomState(SEED)
    manifest = {}
    n_queries = 0
    n_replace_fallback = 0

    # Iterate in the split file's natural order: scene -> scene_id -> queries.
    # JSON object order is preserved by json.load, so combined with the sorted
    # candidate pool and a single sequential RandomState this is reproducible
    # and idempotent.
    for scene in split:
        for scene_id in split[scene]:
            wav_dir = os.path.join(AR_ROOT, FOLDER, scene, scene_id)
            if not os.path.isdir(wav_dir):
                raise RuntimeError(f"missing wav dir: {wav_dir}")
            all_files = sorted(os.listdir(wav_dir))  # sorted -> idempotent
            candidates = [
                fn for fn in all_files
                if fn.endswith("_hybrid_IR.wav") and src_node_of(fn) is not None
            ]

            for q_fn in split[scene][scene_id]:
                n_queries += 1
                q_src = src_node_of(q_fn)
                if q_src is None:
                    raise RuntimeError(f"cannot parse src node from query {q_fn}")
                # Source-exclusion: drop every pair sharing the query source.
                pool = [fn for fn in candidates if src_node_of(fn) != q_src]
                if len(pool) == 0:
                    raise RuntimeError(
                        f"no context candidates after excluding src={q_src} "
                        f"in {wav_dir}"
                    )
                replace = len(pool) < K
                if replace:
                    n_replace_fallback += 1
                sel = rs.choice(pool, K, replace=replace)

                key = f"{scene}/{scene_id}/{os.path.splitext(q_fn)[0]}"
                ctx = [str(x) for x in sel]

                # --- inline sanity (doubles as Verification #1) ---
                assert len(ctx) == K, (key, len(ctx))
                for cfn in ctx:
                    assert src_node_of(cfn) != q_src, (
                        f"source-exclusion violated: query src={q_src} "
                        f"context {cfn} in {key}"
                    )
                    assert os.path.exists(os.path.join(wav_dir, cfn)), (
                        f"context file missing on disk: {wav_dir}/{cfn}"
                    )
                assert key not in manifest, f"duplicate query key {key}"
                manifest[key] = ctx

    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    os.replace(tmp, out_path)
    print(
        f"[ok] {split_name}: wrote {out_path}  "
        f"keys={len(manifest)} queries={n_queries} "
        f"replace_fallback={n_replace_fallback} (K={K}, seed={SEED})"
    )
    return len(manifest), n_queries


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["seen", "unseen", "both"], default="both")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing (frozen) manifest")
    args = ap.parse_args()

    if not os.path.isdir(AR_ROOT):
        sys.exit(f"[abort] {AR_ROOT}/ not found; run from the repo root.")

    splits = ["seen", "unseen"] if args.split == "both" else [args.split]
    for s in splits:
        keys, q = build_split(s, args.force)
        assert keys == q, (
            f"{s}: manifest keys ({keys}) != queries ({q}) — duplicate or "
            f"dropped query keys; investigate before using."
        )


if __name__ == "__main__":
    main()
