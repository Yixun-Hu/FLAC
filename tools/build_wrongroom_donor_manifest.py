"""
Wrong-room donor manifest for the context-ablation control
(advisor hypothesis: context RIR = material proxy).

For every eval query, deterministically pick ONE context RIR from a
DIFFERENT scene category (≠ query's scene) → guaranteed different geometry
AND independently-randomized AcousticRooms materials. Frozen so the control
is reproducible exactly like the main eval manifest.

Output: data/AR/arbRIR_v0_eval_wrongroom_{seen,unseen}.json
        { "<scene>/<scene_id>/<filename_no_ext>": [donor_scene, donor_scene_id, donor_wav] }

The wrongroom eval module loads donor AUDIO from here but keeps the POSE
vectors from the same-room correct K=1 pick (the existing
arbRIR_v0_eval_manifest_{split}.json), so `wrongroom` differs from
`correct` (baseline_A_*_K1) in EXACTLY one tensor: context_audio.
"""
import argparse
import json
import os
import sys

import numpy as np

AR_ROOT = "AcousticRooms"
FOLDER = "single_channel_ir_1"
SEED = 42
SPLIT_FILES = {"seen": "data/AR/seen_eval.json", "unseen": "data/AR/unseen_eval.json"}
OUT_TMPL = "data/AR/arbRIR_v0_eval_wrongroom_{split}.json"


def build(split, force):
    out = OUT_TMPL.format(split=split)
    if os.path.exists(out) and not force:
        raise SystemExit(f"[abort] {out} exists; pass --force to regenerate "
                         f"(changes the wrong-room control for ALL runs).")
    sd = json.load(open(SPLIT_FILES[split]))

    rooms = [(sc, sid) for sc in sd for sid in sd[sc]]            # split order
    rs = np.random.RandomState(SEED)
    manifest, n_q = {}, 0
    listdir_cache = {}

    for scene in sd:
        for scene_id in sd[scene]:
            for q_wav in sd[scene][scene_id]:
                n_q += 1
                # Donor pool = rooms in a DIFFERENT scene category.
                pool = [r for r in rooms if r[0] != scene]
                assert pool, f"no different-category room for {scene}"
                d_scene, d_sid = pool[rs.randint(len(pool))]
                d_dir = os.path.join(AR_ROOT, FOLDER, d_scene, d_sid)
                if d_dir not in listdir_cache:
                    listdir_cache[d_dir] = sorted(
                        f for f in os.listdir(d_dir) if f.endswith("_hybrid_IR.wav")
                    )
                wavs = listdir_cache[d_dir]
                assert wavs, f"no IR wavs in donor {d_dir}"
                d_wav = wavs[rs.randint(len(wavs))]

                key = f"{scene}/{scene_id}/{os.path.splitext(q_wav)[0]}"
                assert key not in manifest, f"dup key {key}"
                # Invariant: donor is a different room AND different category.
                assert d_scene != scene, (key, d_scene, scene)
                assert os.path.exists(os.path.join(d_dir, d_wav))
                manifest[key] = [d_scene, d_sid, d_wav]

    tmp = out + ".tmp"
    json.dump(manifest, open(tmp, "w"), indent=1, sort_keys=True)
    os.replace(tmp, out)
    assert len(manifest) == n_q, (len(manifest), n_q)
    n_cat = len({tuple(v[:2]) for v in manifest.values()})
    print(f"[ok] {split}: {out}  keys={len(manifest)} queries={n_q} "
          f"distinct_donor_rooms={n_cat} (seed={SEED})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["seen", "unseen", "both"], default="both")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if not os.path.isdir(AR_ROOT):
        sys.exit(f"[abort] {AR_ROOT}/ not found; run from repo root.")
    for s in (["seen", "unseen"] if a.split == "both" else [a.split]):
        build(s, a.force)


if __name__ == "__main__":
    main()
