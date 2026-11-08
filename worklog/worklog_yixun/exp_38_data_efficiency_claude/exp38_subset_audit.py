#!/usr/bin/env python3
"""exp_38 round A4 — exhaustive, audio-free audit that exp_38 trains on EXACTLY exp_14's
subsets with exp_14's context restriction (plan_data_efficiency.md §10 A4; Codex plan r1 HIGH 2).

Per fraction (025 / 050 / 075) and for the 100 % split it records and checks:
  (i)  the complete ORDERED target list ``get_audio_filenames`` builds (= ``SampleDataset.filenames``
       for these one-entry dataset configs), its sha256 and count; the source of ``json_scandir`` /
       ``get_audio_filenames`` and of the four AR_md functions must hash to exp_14's launch blob;
  (ii) for EVERY target, AR_md's unrestricted candidate pool (same receiver, the other sources of the
       room, the ``S00{int}`` rebuild, existence on disk) and the restricted pool (∩ the split's
       basenames of that room): zero empty restricted pools, per-room 0 / 1-7 / >=8 histograms equal to
       exp_14's manifest, restricted ⊆ unrestricted, every removed candidate outside the split;
  (iii) for ``train.json`` (the 100 % runs, unrestricted) how many on-disk candidates lie outside it.
It reads the AR tree read-only (one ``os.listdir`` per room) and writes one JSON artifact + .sha256.

    python exp38_subset_audit.py --out <json>        # from the repo root (dataset path "AcousticRooms")

Written by the exp_38 Coder (Claude Opus 5.5).
"""
import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import exp38_registry as reg  # noqa: E402

AR_MD = "src/configs/dataset_configs/custom_metadata/AR_md.py"
DATASET_PY = "src/data/dataset.py"
MANIFEST = "data/AR/train_frac_manifest_s2026.json"
TRAIN_JSON = "data/AR/train.json"
TRAIN_JSON_SHA256 = "aa4e52d616fc42e88d5e4952c7e7ff266347615a60f93a0590b707f5eeaead03"
FOLDER = "single_channel_ir_1"
# sha256 of each function's source segment (ast) in exp_14's launch blobs (66901577); identical at
# the exp_38 base 2b42468c for the dataset.py functions (computed 2026-11-07).
EXP14_FN_SHA256 = {
    (DATASET_PY, "json_scandir"): "c9a44d114a7b782998e50c776f2eaf7c5ad3df7aceb344ee6f213cdbe86e6702",
    (DATASET_PY, "get_audio_filenames"): "02aaf13c08e19f23b236856716a84e0238e8345d4f4e0f5849dfed4cec1c2ab0",
    (AR_MD, "get_custom_metadata"): "af0b962ed25745fca6ef4f623d689c84ad0036d144c1d337c1db4382dea02e97",
    (AR_MD, "_load_split_room_index"): "863339faab05e5bbbe024205412769f1b5bc8a7736fbc29d4f6c15810a5be459",
    (AR_MD, "get_allowed_basenames"): "d2af83ac2ade307de1a19c526c24cc292c047a318e854a348226ccf78f341a84",
    (AR_MD, "get_ir_and_location_for_other_sources"): "a882dcfa79304153a684b2156568ee1862fc32accce538cd924489b396f55123",
}
AR_MD_EXP14_SHA256 = "fb2f68821e24f3e9d7f5c38ab2ecaee0235d4c9a36734c5bf4fe35fa86198310"
AR_MD_BASE_SHA256 = "4846b6531c686bb8d6aa3c2f5427d0f8fe9a50da3a56338c9ff6a072b7600446"   # 7bbd8aa == 2b42468c


def fn_source_sha256(text, name):
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return hashlib.sha256(ast.get_source_segment(text, node).encode()).hexdigest()
    return None


def room_index(listing):
    """(set of names, set of source indices) of one room listing (computed once per room)."""
    return frozenset(listing), frozenset(int(fn.split("_")[0][1:]) for fn in listing)


def candidate_pool(listing, target_basename, index=None):
    """AR_md.get_ir_and_location_for_other_sources' candidate list, from one room listing.

    Same rule: every source index present among the room's files (``int(name.split('_')[0][1:])``),
    minus the target's own, rebuilt as ``S00{node}_{R..}_hybrid_IR.wav`` and kept if it exists.
    Returned in AR_md's iteration order (list of a set difference: CPython small-int set order).
    """
    src_node = int(target_basename.split("_")[0][1:])
    rec_n = target_basename.split("_")[1]
    names, all_src = index if index is not None else room_index(listing)
    all_src = set(all_src)
    out = []
    for node in list(all_src.difference({src_node})):
        cand = f"S00{node}_{rec_n}_hybrid_IR.wav"
        if cand in names:
            out.append(cand)
    return out


def bucket(n):
    return "0" if n == 0 else ("1-7" if n < 8 else ">=8")


def audit_split(split_rel, restrict, dataset_root, listings=None):
    """Return (summary, per_room_hist, problems) for one split."""
    from src.data.dataset import get_audio_filenames
    split = json.load(open(split_rel))
    files = get_audio_filenames(paths=dataset_root, json_file_path=split_rel, folder_name=FOLDER)
    listings = {} if listings is None else listings
    per_room, problems = {}, []
    n_removed = n_outside_split = n_empty = 0
    order = []
    pools = hashlib.sha256()
    for scene, rooms in split.items():
        for room, names in rooms.items():
            d = os.path.join(dataset_root, FOLDER, scene, room)
            if d not in listings:
                lst = sorted(os.listdir(d))
                listings[d] = (lst, room_index(lst))
            lst, idx = listings[d]
            allowed = {os.path.basename(f) for f in names}
            hist = {"0": 0, "1-7": 0, ">=8": 0}
            for name in names:
                order.append(os.path.join(dataset_root, FOLDER, scene, room, name))
                if name not in idx[0]:
                    problems.append(f"{scene}/{room}/{name} listed by the split but absent on disk")
                    continue
                pool = candidate_pool(lst, name, idx)
                kept = [c for c in pool if c in allowed] if restrict else pool
                pools.update(f"{scene}/{room}/{name}:{','.join(kept)}\n".encode())
                n_removed += len(pool) - len(kept)
                n_outside_split += sum(1 for c in pool if c not in allowed)
                if restrict and not kept:
                    n_empty += 1
                hist[bucket(len([c for c in pool if c in allowed]))] += 1
            per_room[room] = hist
    if files != order:
        problems.append("get_audio_filenames order != the split's scene/room/file order")
    summary = {"split": split_rel, "split_sha256": reg.sha256_file(split_rel), "restricted": restrict,
               "n_targets": len(files), "rooms": len(per_room),
               "ordered_targets_sha256": hashlib.sha256("\n".join(files).encode()).hexdigest(),
               "pools_sha256": pools.hexdigest(),
               "empty_restricted_pools": n_empty, "candidates_removed_by_restriction": n_removed,
               "on_disk_candidates_outside_split": n_outside_split}
    if restrict and n_empty:
        problems.append(f"{split_rel}: {n_empty} targets with an empty restricted pool")
    if restrict and n_removed != n_outside_split:
        problems.append(f"{split_rel}: restriction removed {n_removed} != {n_outside_split} out-of-split candidates")
    return summary, per_room, problems


def root_fingerprint(dataset_root, listings=None):
    """sha256 over every room directory's sorted listing (the 243 rooms of train.json)."""
    tr = json.load(open(TRAIN_JSON))
    h = hashlib.sha256()
    for scene in sorted(tr):
        for room in sorted(tr[scene]):
            d = os.path.join(dataset_root, FOLDER, scene, room)
            lst = listings[d][0] if listings and d in listings else sorted(os.listdir(d))
            h.update(f"{scene}/{room}\0{chr(0).join(lst)}\n".encode())
    return h.hexdigest()


def bound_map(dataset_root):
    files = [AR_MD, DATASET_PY, MANIFEST, TRAIN_JSON, os.path.relpath(os.path.realpath(__file__), reg.REPO)]
    for t in reg.FRACS:
        files += [reg.FRACS[t]["split"], reg.FRACS[t]["dataset_config"]]
    out = {rel: reg.sha256_file(rel) for rel in sorted(set(files))}
    out["realpath(" + dataset_root + ")"] = os.path.realpath(dataset_root)
    return out


def verify(artifact, dataset_root="AcousticRooms", repo=None):
    """Re-check a committed audit against the CURRENT tree: sidecar, PASS, the complete bound map
    and the root fingerprint (one listing per room, seconds). Returns a list of problems."""
    os.chdir(repo or reg.REPO)
    try:
        text = open(artifact, "rb").read()
        side = open(artifact + ".sha256").read().split()[0]
    except OSError as err:
        return [f"artifact unreadable: {err}"]
    if hashlib.sha256(text).hexdigest() != side:
        return ["artifact does not match its .sha256"]
    res = json.loads(text)
    p = [] if res.get("verdict") == "PASS" and not res.get("problems") else [f"verdict {res.get('verdict')}"]
    if res.get("bound") != bound_map(dataset_root):
        cur = bound_map(dataset_root)
        p.append(f"bound map differs: {sorted(k for k in set(cur) | set(res.get('bound', {})) if cur.get(k) != res.get('bound', {}).get(k))}")
    if res.get("root_fingerprint") != root_fingerprint(dataset_root):
        p.append("the AcousticRooms room listings changed since the audit")
    return p


def run_audit(dataset_root="AcousticRooms", repo="."):
    problems, out = [], {"schema": 1, "fractions": {}}
    os.chdir(repo)
    for (path, name), want in EXP14_FN_SHA256.items():
        got = fn_source_sha256(open(path).read(), name)
        if got != want:
            problems.append(f"{path}::{name} source {got} != exp_14's {want}")
    if reg.sha256_file(AR_MD) != AR_MD_EXP14_SHA256:
        problems.append("AR_md.py is not exp_14's launch blob")
    base = subprocess.run(["git", "show", f"2b42468c:{AR_MD}"], capture_output=True)
    old = subprocess.run(["git", "show", f"7bbd8aa:{AR_MD}"], capture_output=True)
    base_ok = (base.returncode == 0 and old.returncode == 0
               and hashlib.sha256(base.stdout).hexdigest() == AR_MD_BASE_SHA256
               == hashlib.sha256(old.stdout).hexdigest())
    if not base_ok:
        problems.append("the exp_38 base AR_md.py is not byte-identical to exp_14's base (7bbd8aa)")
    out["base_ar_md_equals_exp14_base"] = base_ok
    if reg.sha256_file(TRAIN_JSON) != TRAIN_JSON_SHA256:
        problems.append("train.json is not the split exp_14 subsampled")
    manifest = json.load(open(MANIFEST))
    listings = {}
    for tag, frac_key in (("025", "0.25"), ("050", "0.5"), ("075", "0.75")):
        f = reg.FRACS[tag]
        try:
            reg.check_subset(tag)
        except reg.GateError as err:
            problems.append(str(err))
        summary, per_room, probs = audit_split(f["split"], True, dataset_root, listings)
        problems += probs
        want = manifest["fractions"][frac_key]
        if per_room != {r: v["context_histogram"] for r, v in want["per_room"].items()}:
            problems.append(f"{tag}: per-room context histograms differ from exp_14's manifest")
        total = {b: sum(h[b] for h in per_room.values()) for b in ("0", "1-7", ">=8")}
        if total != want["context_histogram"] or summary["n_targets"] != want["final"] != f["n_items"]:
            problems.append(f"{tag}: totals {total} / {summary['n_targets']} != manifest")
        summary["context_histogram"] = total
        out["fractions"][tag] = summary
    summary, per_room, probs = audit_split(TRAIN_JSON, False, dataset_root, listings)
    problems += probs
    total = {b: sum(h[b] for h in per_room.values()) for b in ("0", "1-7", ">=8")}
    if total != manifest["full_split_context_histogram"]:
        problems.append(f"100 %: in-split histogram {total} != manifest {manifest['full_split_context_histogram']}")
    summary["context_histogram_in_split"] = total
    out["full_split"] = summary
    out["bound"] = bound_map(dataset_root)
    out["root_fingerprint"] = root_fingerprint(dataset_root, listings)
    out["problems"] = problems
    out["verdict"] = "PASS" if not problems else "FAIL"
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--out")
    g.add_argument("--verify", metavar="ARTIFACT")
    p.add_argument("--dataset-root", default="AcousticRooms")
    a = p.parse_args(argv)
    if a.verify:
        problems = verify(os.path.realpath(a.verify), a.dataset_root)
        print("SUBSET AUDIT VERIFY " + ("PASS" if not problems else "FAIL: " + " | ".join(problems)))
        return 0 if not problems else 1
    if os.path.exists(a.out):
        print(f"refusing to overwrite {a.out}", file=sys.stderr)
        return 2
    res = run_audit(a.dataset_root, repo=reg.REPO)
    text = json.dumps(res, indent=1, sort_keys=True) + "\n"
    with open(a.out, "w") as fh:
        fh.write(text)
    with open(a.out + ".sha256", "w") as fh:
        fh.write(f"{hashlib.sha256(text.encode()).hexdigest()}  {os.path.basename(a.out)}\n")
    print(f"SUBSET AUDIT {res['verdict']}: {a.out}" + ("" if not res["problems"] else f" ({len(res['problems'])} problems)"))
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
