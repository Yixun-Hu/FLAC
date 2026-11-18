#!/usr/bin/env python3
"""exp_46 table: C4F at 25 / 50 / 75 / 100 % next to YAW-Aug and OrbitRIR at 25 / 50 / 75 % x K {8, 1} = 20 rows
(plan_c4flip_data_curve.md r3 §3.2 "exp46_table.py", M2, §8).

    exp46_table.py import-exp38 --commit <sha> [--generator-root <dir>]
    exp46_table.py build --exp38-commit <sha>

``import-exp38`` (Codex plan r2 finding 5, r3 findings 1 and 4): the exp_38 results bundle -- a commit on
exp-38-data-efficiency, descending from 3593f34c, that force-adds the complete local dependency set of exp_38's
offline admission at exp_38's own relative paths -- is extracted from git into a temporary detached directory (only
EXP38_SOURCE_PATHS; no git worktree metadata is created) and an ISOLATED adapter (a separate ``python -I -B``
process importing exp_38's own exp38_eval from that directory) runs exp38_eval.validate_e0_receipt(offline=True)
and exp38_eval.validate_cell(..., offline=True) for the 60 YAW-Aug / OrbitRIR fraction cells, recording every
file it opens (an audit hook). Aggregation is pure code here: exp38_collect.collect() is NOT used (no offline mode;
it opens NAS payloads). Every file the adapter opens must be exp_38 code or one of EXP38_IMPORT_FILES; every one of
EXP38_IMPORT_FILES is re-hashed locally into the record; NAS checkpoint and waveform payloads are retained
attestations (the hashes the receipts carry), never opened. The result is the sealed c4flip_data_curve_exp38_import.json
(commit sha, every admitted file's sha256, the derived rows). With ``--generator-root`` the verified files are then
copied byte for byte (hash re-checked after the copy) to their exact relative paths there, for the shared generator's
twelve exp_38 rows; a differing existing file refuses. The exp_38 commit is a RUNTIME input (recorded), never a
constant here: this file is closure code, frozen at the launch commit X.

``build``: C4F from exp_46's admitted cells (exp46_eval's offline admission, re-derived and compared with the sealed
collector JSON); YAW-Aug / OrbitRIR ONLY from the sealed import record of the given exp_38 commit (its sidecar, its
commit, its rows re-derived from its per-cell values) -- and the record is bound back to that commit (Codex code r1
finding 6): the bundle is re-extracted at the commit (git archive, read-only) and every recorded file hash, every
cell's metrics, the checkpoint / dump-manifest attestations and the E0 campaign are re-verified against those blobs.
exp_46 has no bridge (plan r3 §3.2): every C4F point is evaluated in this tree. Refuses a missing (method x
fraction x K) cell, a value that differs from its source, an import record whose hash or commit does not match. Never reads exp_38's preliminary files (plan F9) and never re-runs exp_38's closure checks against the
exp_46 tree (Codex plan r1 finding 8).
Writes c4flip_data_curve_table.md (+ the sealed c4flip_data_curve_table_evidence.json) and prints the §8 disclosures.

Written by the exp_46 Coder (Claude Opus 5.5).
"""
import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import exp46_eval as ev  # noqa: E402
import exp46_registry as reg  # noqa: E402

EXP38_BASE = "3593f34cad4e6e4415a572e93d16ee83465f8ae8"     # the merged exp_38 tip: the bundle must descend from it
EXP38_RUNS = tuple(f"de_{a}_f{t}" for a in ("yaw", "bf") for t in ("025", "050", "075"))
EXP38_FARM = "outputs_FLAC/exp38_data_efficiency/eval"
# what exp_38's offline admission reads from git (the bundle's own tracked files)
EXP38_SOURCE_PATHS = (reg.EXP38_DIR, EXP38_FARM, "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json",
                      "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json",
                      "worklog/worklog_yixun/exp_15_yaw_aug_claude/FLAC_AR_YAWAUG.json")
IMPORT_RECORD = f"{reg.EXP_DIR}/c4flip_data_curve_exp38_import.json"
IMPORT_KEYS = frozenset({"schema", "commit", "base", "campaign", "files", "code", "opened", "attestations", "cells",
                         "rows", "written_at"})
TABLE_MD = f"{reg.EXP_DIR}/c4flip_data_curve_table.md"
TABLE_EVIDENCE = f"{reg.EXP_DIR}/c4flip_data_curve_table_evidence.json"
METHODS = {"c4f": "C4 + flip (C4F), exp_46", "yaw": "YAW-Aug, exp_38", "bf": "OrbitRIR (B-F), exp_38"}
FRACS = ("025", "050", "075")                      # exp_38's imported fractions
# per-method fraction sets (plan r3 §3.2 "exp46_table.py", M2): C4F at 25 / 50 / 75 / 100 %, YAW-Aug and OrbitRIR at
# 25 / 50 / 75 % (exp_38's formal results); x K {8, 1} = 20 comparison rows -- no YAW-Aug / OrbitRIR 100 % row
METHOD_FRACS = {"c4f": ("025", "050", "075", "100"), "yaw": FRACS, "bf": FRACS}
N_ROWS = 2 * sum(len(t) for t in METHOD_FRACS.values())            # 20
HEADLINE = (("EDT", "EDT (ms) ↓", 3), ("C50", "C50 (dB) ↓", 4), ("T60", "T60 (%) ↓", 3),
            ("RIR_to_GT_RIR_R@1", "R@1 ↑", 3), ("RIR_to_GT_RIR_R@5", "R@5 ↑", 3), ("RIR_to_GT_RIR_R@10", "R@10 ↑", 3))
DISCLOSURES = reg.disclosures("table")          # the ONE source: exp46_registry.DISCLOSURES (finding 15)


class TableError(RuntimeError):
    pass


# ---------------------------------------------------------------- EXP38_IMPORT_FILES (plan r3 §3.2, r3 finding 1)
# Derived from exp_38's OFFLINE code paths at 3593f34c (file:line), i.e. every local file they open:
#  exp38_eval.validate_e0_receipt(offline=True, readmit=True)   [exp38_eval.py:395-411, _e0_finish 443-478]
#    - the E0 receipt + .sha256 (exp38_registry.read_sealed 836-856)
#    - check_refs(): ref_cells/REF_MANIFEST.sha256 + the 68 ref cells (exp38_eval.py:315-330, load_ref 333-338)
#    - load_e0a_decision(required=True) when the receipt binds it: the decision + .sha256 (270-298)
#    - re-admission of the 8 E0 cells: metrics, stream, receipt (463-469, validate_cell 532-534); their configs
#      (want_r model_config_sha256, 582): exp_07 FLAC_AR_BVp1.json / FLAC_AR_BF.json, exp_15 FLAC_AR_YAWAUG.json
#  exp38_eval.validate_cell(cell, campaign, offline=True)       [exp38_eval.py:525-617]
#    - metrics, stream, receipt (532-534); the arm config (582): exp_15 FLAC_AR_YAWAUG.json (yaw) or
#      exp_38 FLAC_AR_BF_cap32.json (bf); the dump-manifest COPY <stem>.dump_manifest.json (600-604)
#    - expected_ckpt(offline) -> exp38_registry.check_training_receipt(receipt_dir=<farm>/<run>, verify_bytes=False)
#  exp38_registry.check_training_receipt (offline)              [exp38_registry.py:1725-1852]
#    - TRAINING_COMPLETE.json + .sha256 (1741); the arm config (1748)
#    - receipts/: the retained E0 / environment / SMOKE receipts + .sha256 (1770-1772), SNAPSHOT.sha256 +
#      SNAPSHOT.links (1785-1786), smoke_log.log (1801-1805), train.log (1808-1812)
#    - evidence/run_evidence.json.rank{0,1}.json (1813-1816); CHECKPOINTS.sha256 (1820-1823)
#    - TRAINING_COMMITTED.json + .sha256 and the terminal launch record it names (1830-1834)
# The receipts/ names and the launch record are data-dependent: they are read from the run's sealed
# TRAINING_COMPLETE.json ("receipts") and TRAINING_COMMITTED.json ("launch_record").
EXP38_CONFIGS = ("worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json",
                 "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json",
                 "worklog/worklog_yixun/exp_15_yaw_aug_claude/FLAC_AR_YAWAUG.json",
                 f"{reg.EXP38_DIR}/FLAC_AR_BF_cap32.json")


def exp38_cell_stems():
    """{cell name: (run, K, seed, farm-relative stem)} of the 60 exp_38 fraction cells (exp38_eval.run_cells naming)."""
    out = {}
    for run in EXP38_RUNS:
        tag = run[-3:]
        ck = f"epoch={(reg.MAX_STEPS - 1) // reg.FRACS[tag]['steps_per_epoch']}-step={reg.MAX_STEPS}"
        for K in (8, 1):
            for s in ev.SEEDS:
                name = f"{run}_K{K}_s{s}"
                out[name] = (run, K, s, f"{EXP38_FARM}/{run}/{ck}_metrics_1_1.0_{name}"
                             + ("_fa_invariant_a4" if "_bf_" in run else ""))
    return out


def exp38_import_files(root):
    """The exact relative paths of EXP38_IMPORT_FILES in the extracted bundle ``root`` (see the derivation above)."""
    e38 = reg.EXP38_DIR
    files = [f"{e38}/data_efficiency_e0_receipt.json", f"{e38}/data_efficiency_e0_receipt.json.sha256",
             f"{reg.EXP38_DIR}/ref_cells/REF_MANIFEST.sha256", *EXP38_CONFIGS]
    files += [f"{reg.EXP38_DIR}/ref_cells/{r}" for r in reg.ref_set()]
    e0 = json.load(open(os.path.join(root, files[0])))
    if e0.get("e0a_decision_sha256") is not None:
        files += [f"{e38}/data_efficiency_e0a_decision.json", f"{e38}/data_efficiency_e0a_decision.json.sha256"]
    # exp38_eval.E0: E0a = dc_van_f025's epoch=34 endpoint (vanilla), E0b = B-F 100 % (fa_invariant), E0c / E0d =
    # YAW / P1 100 % (vanilla); exp38_eval.make_cell names the stem <ckpt stem>_metrics_1_1.0_<cell><method suffix>
    for gate, ckpt, suffix in (("E0a", "epoch=34-step=40000", ""), ("E0b", "epoch=8-step=40000", "_fa_invariant_a4"),
                               ("E0c", "epoch=8-step=40000", ""), ("E0d", "epoch=8-step=40000", "")):
        for K in (8, 1):
            stem = f"{EXP38_FARM}/{gate}/{ckpt}_metrics_1_1.0_de_{gate}_K{K}_s42{suffix}"
            files += [stem + ".json", stem + ".stream.json", stem + ".receipt.json"]
    for run in EXP38_RUNS:
        farm = f"{EXP38_FARM}/{run}"
        tr = json.load(open(os.path.join(root, farm, reg.TRAINING_RECEIPT)))
        cm = json.load(open(os.path.join(root, farm, reg.COMMIT_MARKER)))
        files += [f"{farm}/{f}" for f in (reg.TRAINING_RECEIPT, reg.TRAINING_RECEIPT + ".sha256", reg.COMMIT_MARKER,
                                          reg.COMMIT_MARKER + ".sha256", cm["launch_record"], "CHECKPOINTS.sha256")]
        names = tr["receipts"]
        for role in ("e0", "env", "smoke"):
            files += [f"{farm}/receipts/{names[role]}", f"{farm}/receipts/{names[role]}.sha256"]
        files += [f"{farm}/receipts/{names[r]}" for r in ("smoke_log", "log", "snapshot_manifest", "snapshot_links")]
        files += [f"{farm}/evidence/run_evidence.json.rank{r}.json" for r in range(2)]
    for name, (_run, _k, _s, stem) in exp38_cell_stems().items():
        files += [stem + ".json", stem + ".stream.json", stem + ".receipt.json", stem + ".dump_manifest.json"]
    return sorted(set(files))


# ---------------------------------------------------------------- the isolated adapter (runs inside the bundle)
ADAPTER = r'''
import json, os, sys
root = os.path.realpath(sys.argv[1])
opened = set()

def hook(event, args):
    if event in ("open", "os.listdir", "os.scandir") and args and args[0] is not None and not isinstance(args[0], int):
        p = os.path.realpath(os.fsdecode(args[0]))
        if "__pycache__" in p.split(os.sep):        # the import system probing bytecode caches: code, not data
            return
        if p.startswith("/media"):
            opened.add("NAS:" + p)
        elif p == root or p.startswith(root + os.sep):
            opened.add(os.path.relpath(p, root) + ("/" if event != "open" else ""))
sys.addaudithook(hook)
sys.path.insert(0, os.path.join(root, sys.argv[2]))
import exp38_eval as ev
campaign = ev.validate_e0_receipt(offline=True)
cells = {}
for run in json.loads(sys.argv[3]):
    for c in ev.run_cells(run):
        probs = ev.validate_cell(c, campaign, offline=True)
        rec = json.load(open(ev._abs(c["metrics"]))) if not probs else {}
        receipt = json.load(open(ev._abs(c["receipt"]))) if not probs else {}
        cells[c["name"]] = {"run": run, "K": c["K"], "seed": c["seed"], "problems": probs,
                            "metrics": rec.get("metrics"), "metrics_rel": os.path.relpath(ev._abs(c["metrics"]), root),
                            "ckpt_sha256": receipt.get("ckpt_sha256"), "dump_manifest_sha256": receipt.get("dump_manifest_sha256")}
print("EXP46_ADAPTER " + json.dumps({"campaign": {k: campaign[k] for k in ("head", "closure_sha256", "sha256")},
                                     "cells": cells, "opened": sorted(opened)}))
'''


def _git(repo, *args, binary=False):
    out = subprocess.run(["git", "-C", repo, *args], capture_output=True)
    if out.returncode:
        raise TableError(f"git {' '.join(args)} failed: {out.stderr.decode(errors='replace').strip()[-300:]}")
    return out.stdout if binary else out.stdout.decode()


def extract_bundle(commit, dest, repo=None):
    """The bundle's EXP38_SOURCE_PATHS at ``commit`` into ``dest`` (a detached extraction; a .git marker makes
    exp_38's _repo_root resolve there)."""
    import io
    import tarfile
    repo = repo or reg.REPO
    sha = _git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}").strip()
    if sha != commit:
        raise TableError(f"--commit must be the full 40-hex commit id (resolves to {sha})")
    if subprocess.run(["git", "-C", repo, "merge-base", "--is-ancestor", EXP38_BASE, sha]).returncode:
        raise TableError(f"{sha[:12]} does not descend from exp_38's tip {EXP38_BASE[:12]}")
    data = _git(repo, "archive", "--format=tar", sha, "--", *EXP38_SOURCE_PATHS, binary=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for m in tar.getmembers():
            if not (m.isfile() or m.isdir()) or m.name.startswith("/") or ".." in m.name.split("/"):
                raise TableError(f"bundle entry {m.name!r} is not a plain file")
        tar.extractall(dest)
    open(os.path.join(dest, ".git"), "w").write("exp46 import extraction marker\n")
    return sha


def mean_sd(values):
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def aggregate(per_seed):
    """{seed: metrics} -> {metric: [mean, sd, n]} over exactly seeds 42-46 (exp38_collect.aggregate's convention)."""
    if sorted(per_seed) != list(ev.SEEDS):
        raise TableError(f"seeds {sorted(per_seed)} != {list(ev.SEEDS)}")
    return {k: [*mean_sd([per_seed[s][k] for s in ev.SEEDS]), len(per_seed)] for k in ev.METRIC_KEYS}


def exp38_rows(cells):
    rows = {}
    for run in EXP38_RUNS:
        for K in (8, 1):
            per = {v["seed"]: v["metrics"] for v in cells.values() if v["run"] == run and v["K"] == K}
            rows[f"{run.split('_')[1]}|{run[-3:]}|{K}"] = aggregate(per)
    return rows


def import_exp38(commit, generator_root=None, repo=None, python=None):
    """Admit the exp_38 results bundle of ``commit`` offline (isolated adapter) and seal the import record."""
    repo = repo or reg.REPO
    out_path = os.path.join(repo, IMPORT_RECORD)
    if os.path.exists(out_path):
        raise TableError(f"{IMPORT_RECORD} exists (an import is sealed once; archive it first)")
    tmp = tempfile.mkdtemp(prefix="exp46_exp38_import_")
    try:
        sha = extract_bundle(commit, tmp, repo)
        try:
            files = exp38_import_files(tmp)
        except (OSError, ValueError, KeyError) as err:
            raise TableError(f"bundle incomplete: {err}")
        missing = [f for f in files if not os.path.isfile(os.path.join(tmp, f))]
        if missing:
            raise TableError(f"bundle misses {len(missing)} file(s) of EXP38_IMPORT_FILES, e.g. {missing[:3]}")
        hashes = {f: reg.sha256_file(os.path.join(tmp, f)) for f in files}
        env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "GIT_"))}
        env.update(PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1")
        res = subprocess.run([python or sys.executable, "-I", "-B", "-c", ADAPTER, tmp, reg.EXP38_DIR,
                              json.dumps(list(EXP38_RUNS))], cwd=tmp, env=env, capture_output=True, text=True,
                             timeout=3600)
        line = [ln for ln in res.stdout.split("\n") if ln.startswith("EXP46_ADAPTER ")]
        if res.returncode or len(line) != 1:
            raise TableError(f"exp_38's offline admission refused (rc {res.returncode}): {res.stderr.strip()[-600:]}")
        got = json.loads(line[0][len("EXP46_ADAPTER "):])
        bad = {n: v["problems"] for n, v in got["cells"].items() if v["problems"]}
        if bad or sorted(got["cells"]) != sorted(exp38_cell_stems()):
            raise TableError(f"exp_38 cells not admitted offline: {dict(list(bad.items())[:3])}")
        nas = [p for p in got["opened"] if p.startswith("NAS:")]
        code = {p for p in got["opened"] if p.startswith(reg.EXP38_DIR + "/") and p.endswith(".py")}
        stray = sorted(p for p in got["opened"] if not p.endswith("/") and p not in code and p not in hashes)
        if nas or stray:
            raise TableError(f"the adapter opened files outside EXP38_IMPORT_FILES: NAS {nas[:2]}, other {stray[:3]}")
        for name, v in got["cells"].items():                    # the values are the hashed files' values
            stem = exp38_cell_stems()[name][3]
            if v["metrics_rel"] != stem + ".json" or \
                    json.load(open(os.path.join(tmp, stem + ".json")))["metrics"] != v["metrics"]:
                raise TableError(f"{name}: adapter values / path differ from the hashed metrics file")
        code_hashes = {p: reg.sha256_file(os.path.join(tmp, p)) for p in sorted(code)}    # exp_38's kit, as imported
        rec = {"schema": 1, "commit": sha, "base": EXP38_BASE, "campaign": got["campaign"], "files": hashes,
               "code": code_hashes,
               "opened": sorted(p for p in got["opened"] if not p.endswith("/") and p not in code),
               "attestations": {"checkpoints": {n: v["ckpt_sha256"] for n, v in sorted(got["cells"].items())},
                                "waveform_dump_manifests": {n: v["dump_manifest_sha256"] for n, v in sorted(got["cells"].items())}},
               "cells": {n: {k: v[k] for k in ("run", "K", "seed", "metrics")} for n, v in sorted(got["cells"].items())},
               "rows": exp38_rows(got["cells"]), "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        if generator_root:
            materialise(tmp, dict(hashes, **code_hashes), generator_root)
        digest = reg.write_sealed(out_path, rec)
        return {"import_record": out_path, "sha256": digest, "commit": sha, "files": len(hashes)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def materialise(src_root, hashes, generator_root):
    """Copy every verified file -- all of EXP38_IMPORT_FILES and the exp_38 kit code its offline admission imported
    -- byte for byte to its exact relative path under ``generator_root`` (r3 finding 4; Codex code r1 finding 12: the
    generator root need not carry exp_38's folder): the hash is re-checked after the copy; an existing file must
    already be byte-identical (a tree that tracks exp_38's folder at the same bytes is left as it is)."""
    if not os.path.isdir(generator_root):
        raise TableError(f"generator root {generator_root} does not exist")
    # Codex code r2 finding 6: the generator root itself may not equal or lie inside an exp_38 / exp_45 protected root
    # (exp_46 plan r3 §3.2 "Isolation"), and no file is written through an aliased directory; checked for every file
    # BEFORE any write. The files themselves are exp_38's own verified records at their registered relative paths
    # (plan r3 §3.2; exp_45's r2-6 exception, unchanged)
    probs = reg.isolation_problems(generator_root, "the generator root", trees=[reg.REPO, generator_root],
                                   contains=False)
    if probs:
        raise TableError("isolation from exp_38 / exp_45 (exp_46 plan r3 §3.2): " + "; ".join(probs))
    for rel in sorted(hashes):
        problem = reg.alias_problem(generator_root, rel)
        if problem:
            raise TableError(f"{problem}: refused, nothing written")
    for rel, digest in sorted(hashes.items()):
        dest = os.path.join(generator_root, rel)
        if os.path.lexists(dest):                      # lexists: a dangling symlink is an entry (r2 finding 9)
            if os.path.islink(dest) or not os.path.isfile(dest) or reg.sha256_file(dest) != digest:
                raise TableError(f"{dest} exists with other bytes (or is not a regular file): refused")
            continue
        if reg.tracked_in_index(generator_root, rel):
            raise TableError(f"{dest} is tracked in the generator root's index but missing: refused, never written")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        data = open(os.path.join(src_root, rel), "rb").read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise TableError(f"{rel} changed during the import")
        try:
            reg.publish_no_clobber(dest, data)         # no replacing rename: an entry created meanwhile is kept
        except FileExistsError:
            raise TableError(f"{dest} appeared during the copy: refused, never overwritten")
        if reg.sha256_file(dest) != digest:
            raise TableError(f"{dest}: the copy does not re-hash to {digest[:12]}")


# ---------------------------------------------------------------- build
def c4f_rows(campaign):
    """exp_46's C4F rows {"c4f|<tag>|<K>": {metric: [mean, sd, n]}} through exp46_eval's offline admission."""
    rows = {}
    for run in reg.campaign_runs():
        cells = ev.run_cells(run)
        probs = [x for c in cells for x in ev.validate_cell(c, campaign, offline=True)]
        if probs:
            raise TableError(f"{run}: not every C4F cell is admitted ({probs[0]})")
        for K in (8, 1):
            rows[f"c4f|{reg.parse_run_id(run)[1]}|{K}"] = aggregate(
                {c["seed"]: json.load(open(ev._abs(c["metrics"])))["metrics"] for c in cells if c["K"] == K})
    return rows


def _same(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def verify_record_at_commit(rec, repo=None):
    """Codex code r1 finding 6: the sealed import record against the IMMUTABLE blobs of its commit. The bundle is
    re-extracted at ``rec["commit"]`` (git archive into a temporary directory; the repository is only read): the
    record's file set must be EXP38_IMPORT_FILES of that tree, every recorded sha256 must re-hash, every cell's
    (run, K, seed, metrics) must be its metrics file's, its checkpoint / dump-manifest attestations its receipt's,
    and the campaign the E0 receipt's. Raises TableError."""
    tmp = tempfile.mkdtemp(prefix="exp46_table_build_")
    try:
        commit = extract_bundle(rec["commit"], tmp, repo)
        try:
            files = exp38_import_files(tmp)
        except (OSError, ValueError, KeyError) as err:
            raise TableError(f"the bundle at {commit[:12]} is incomplete: {err}")
        if sorted(rec["files"]) != files:
            raise TableError(f"the import record's file set != EXP38_IMPORT_FILES at {commit[:12]}")
        bad = [f for f in files if not os.path.isfile(os.path.join(tmp, f))
               or reg.sha256_file(os.path.join(tmp, f)) != rec["files"][f]]
        bad += sorted(f for f, d in (rec["code"] or {}).items() if not f.startswith(reg.EXP38_DIR + "/")
                      or not f.endswith(".py") or not os.path.isfile(os.path.join(tmp, f))
                      or reg.sha256_file(os.path.join(tmp, f)) != d)
        if bad:
            raise TableError(f"{len(bad)} recorded sha256 differ from the blobs at {commit[:12]}, e.g. {bad[:3]}")
        stems = exp38_cell_stems()
        if sorted(rec["cells"]) != sorted(stems):
            raise TableError(f"the import record's cells != the 60 exp_38 fraction cells")
        att = rec["attestations"]
        for name, (run, K, seed, stem) in sorted(stems.items()):
            cell = rec["cells"][name]
            metrics = json.load(open(os.path.join(tmp, stem + ".json"))).get("metrics")
            if (cell.get("run"), cell.get("K"), cell.get("seed")) != (run, K, seed) or not _same(cell.get("metrics"), metrics):
                raise TableError(f"{name}: the recorded values differ from its metrics file at {commit[:12]}")
            receipt = json.load(open(os.path.join(tmp, stem + ".receipt.json")))
            if (att["checkpoints"].get(name), att["waveform_dump_manifests"].get(name)) != \
                    (receipt.get("ckpt_sha256"), receipt.get("dump_manifest_sha256")):
                raise TableError(f"{name}: the recorded attestations differ from its receipt at {commit[:12]}")
        e0_rel = f"{reg.EXP38_DIR}/data_efficiency_e0_receipt.json"
        e0 = json.load(open(os.path.join(tmp, e0_rel)))
        if not _same(rec["campaign"], {"head": e0.get("head"), "closure_sha256": e0.get("closure_sha256"),
                                       "sha256": rec["files"][e0_rel]}):
            raise TableError(f"the recorded campaign differs from the E0 receipt at {commit[:12]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def row_set_problems(rows):
    """The 20 registered rows exactly: a missing row OR an extra one (e.g. a YAW-Aug 100 % row) is a problem."""
    want = {f"{m}|{t}|{K}" for m in METHODS for t in METHOD_FRACS[m] for K in (8, 1)}
    assert len(want) == N_ROWS == 20
    missing, extra = sorted(want - set(rows)), sorted(set(rows) - want)
    return ([f"table rows missing {missing}"] if missing else []) + ([f"table rows extra {extra}"] if extra else [])


def build(exp38_commit, repo=None):
    repo = repo or reg.REPO
    if not (isinstance(exp38_commit, str) and len(exp38_commit) == 40):
        raise TableError("--exp38-commit must be a full 40-hex commit id")
    rec, rec_sha = reg.read_sealed(os.path.join(repo, IMPORT_RECORD), IMPORT_KEYS)
    if rec["commit"] != exp38_commit or rec["base"] != EXP38_BASE or rec["schema"] != 1:
        raise TableError(f"the import record is of commit {str(rec['commit'])[:12]}, not {exp38_commit[:12]}")
    if not _same(exp38_rows(rec["cells"]), rec["rows"]):
        raise TableError("the import record's rows differ from its own per-cell values")
    verify_record_at_commit(rec, repo)                   # ... and its values are the commit's blobs (finding 6)
    campaign = ev.validate_e0_receipt(offline=True)
    rows = dict(c4f_rows(campaign))
    # Codex code r1 finding 8: the C4F source rows and the collector's published rows are each EXACTLY the eight C4F
    # rows (no extra fraction / K / model, no duplicate, none missing), value for value
    c4f_want = {f"c4f|{t}|{K}" for t in METHOD_FRACS["c4f"] for K in (8, 1)}
    if set(rows) != c4f_want:
        raise TableError(f"the C4F source rows are not exactly the eight C4F rows (missing "
                         f"{sorted(c4f_want - set(rows))}, extra {sorted(set(rows) - c4f_want)})")
    results = os.path.join(repo, f"{reg.EXP_DIR}/c4flip_data_curve_results.json")
    entries = json.load(open(results))["rows"]
    keys = [f"{r['model']}|{r['fraction']}|{r['K']}" for r in entries]
    if len(keys) != len(set(keys)) or set(keys) != c4f_want:
        raise TableError(f"the collector's published rows are not exactly the C4F source rows (missing "
                         f"{sorted(c4f_want - set(keys))}, extra {sorted(set(keys) - c4f_want)}, "
                         f"{len(keys) - len(set(keys))} duplicate(s))")
    published = {k: {m: [s["mean"], s["sd"], s["n"]] for m, s in r["stats"].items()} for k, r in zip(keys, entries)}
    for key, st in rows.items():
        if not _same(published.get(key), st):
            raise TableError(f"{key}: the collector's published value differs from its source cells")
    rows.update(rec["rows"])
    problems = row_set_problems(rows)
    if problems:
        raise TableError("; ".join(problems))
    md = render(rows, rec, rec_sha, campaign)
    evidence = {"schema": 1, "exp38_commit": exp38_commit, "exp38_import_sha256": rec_sha,
                "exp46_campaign": {k: campaign[k] for k in ("head", "closure_sha256", "sha256")},
                "c4f_results_sha256": reg.sha256_file(results), "rows": rows,
                "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    for p in (TABLE_EVIDENCE, TABLE_EVIDENCE + ".sha256"):
        if os.path.exists(os.path.join(repo, p)):
            raise TableError(f"{p} exists (the table is built once per evidence; archive it first)")
    reg.atomic_write_bytes(os.path.join(repo, TABLE_MD), md.encode())
    reg.write_sealed(os.path.join(repo, TABLE_EVIDENCE), evidence)
    return md


def _finite(x):
    import math
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def render(rows, rec, rec_sha, campaign):
    """The table Markdown. The rendering boundary itself (Codex code r1 finding 8): exactly the 20 registered rows (an
    extra row is refused, never silently dropped) and every printed headline cell a finite [mean, sd, n]."""
    problems = row_set_problems(rows)
    bad = sorted(f"{key}.{k}" for key, st in rows.items() for k, _, _ in HEADLINE
                 if not (isinstance(st, dict) and isinstance(st.get(k), list) and len(st[k]) == 3
                         and all(_finite(x) for x in st[k])))
    if problems or bad:
        raise TableError("; ".join(problems + ([f"malformed cells {bad[:4]}"] if bad else [])))
    out = ["# exp_46 c4flip_data_curve — C4F vs YAW-Aug vs OrbitRIR (generated by exp46_table.py)", "",
           f"θ = 0, 1 Euler step, cfg 1.0, EMA 40k endpoints, item-weighted over the 6,337 unseen items, mean ± sample SD "
           f"over eval seeds 42–46. C4F: exp_46 pin {campaign['head'][:12]}; YAW-Aug / OrbitRIR: exp_38 results commit "
           f"{rec['commit'][:12]} (import record sha256 {rec_sha[:12]}).", ""]
    for K in (8, 1):
        out += [f"## K = {K}" + (" (headline)" if K == 8 else ""), "",
                "| method | data | " + " | ".join(h for _, h, _ in HEADLINE) + " |", "|---|---|" + "---|" * len(HEADLINE)]
        for m, label in METHODS.items():
            for t in METHOD_FRACS[m]:
                st = rows[f"{m}|{t}|{K}"]
                out.append(f"| {label} | {int(t)} % | " + " | ".join(f"{st[k][0]:.{d}f} ± {st[k][1]:.{d}f}"
                                                                    for k, _, d in HEADLINE) + " |")
        out.append("")
    return "\n".join(out + reg.disclosure_block("table")) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("import-exp38"); s.add_argument("--commit", required=True); s.add_argument("--generator-root")
    s = sub.add_parser("build"); s.add_argument("--exp38-commit", required=True)
    a = p.parse_args(argv)
    try:
        if a.cmd == "import-exp38":
            print(json.dumps(import_exp38(a.commit, a.generator_root)))
        else:
            print(build(a.exp38_commit))
    except (TableError, ev.EvalError, reg.GateError, OSError, ValueError, KeyError) as err:
        print(f"EXP46 TABLE REFUSED: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
