#!/usr/bin/env python3
"""exp_46 collector (forked from exp_45's exp45_collect.py at X45f, from exp_38's; plan_c4flip_data_curve.md r3 §3.2): the C4F result cells (10 per run)
and the E0 receipt -> c4flip_data_curve_results.{json,md} (exp_46 has no bridge: plan r3 §3.2).

c4f_rows() replaces exp_38's reference_rows() / EXP14_ROWS / P1-bridge path (Codex plan r2 finding 3): every cell
goes through exp46_eval's OFFLINE admission (validate_cell(..., offline=True): exact metrics / by-scene /
receipt schemas, protocol, the run's TRAINING_COMPLETE copy in the farm, the stream's input AND assignment hashes, the dump-manifest copy) under the sealed
E0 receipt (validate_e0_receipt(offline=True)), so the results re-derive from the tracked evidence alone; NAS
payloads (checkpoints, waveforms) are attested by the hashes the receipts carry, never opened. Any missing or
unadmitted cell refuses: nothing publishes partially. Publication (exp_38 Codex code r1 HIGH 10): Markdown first,
the JSON LAST (the commit point; the Markdown carries the JSON's sha256), each temp -> fsync -> rename, under the
collector lock; the JSON carries {path: sha256} of every file the admission consumed.

    python exp46_collect.py                                   # collect + publish (the default)
    python exp46_collect.py verify [--root DIR]               # sha256-verify a published results bundle in DIR
    python exp46_collect.py materialise --dest DIR            # byte-copy the verified publication into DIR

``verify`` / ``materialise`` (Codex code r1 finding 12): the results JSON names every file its admission consumed
with its sha256 and the Markdown names the JSON's sha256, so a checkout of the results commit, a publication
worktree or the neuronic generator root is verified by re-hashing exactly those files. ``materialise`` copies them,
plus what exp46_eval's offline admission reads besides the evidence (generator_static: the exp_46 kit's
exp46_eval.py / exp46_registry.py, the C4F + E0 configs, exp_38's E0a decision and reference cells), byte for byte to
the same relative paths under DIR -- an existing file must already be identical, every copy is re-hashed -- and then
verifies DIR. Run from the tree holding the publication (tree 45 -> the publication worktree on the A6000 box; the
build checkout of the results commit -> the generator root on neuronic).

Written by the exp_38 Coder (Claude Opus 5.5); exp_46 deltas by the exp_46 Coder (Claude Opus 5.5).
"""
import argparse
import fcntl
import hashlib
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import exp46_eval as ev  # noqa: E402
import exp46_registry as reg  # noqa: E402

OUT_JSON = f"{reg.EXP_DIR}/c4flip_data_curve_results.json"
OUT_MD = f"{reg.EXP_DIR}/c4flip_data_curve_results.md"
LOCK = f"{reg.EXP_DIR}/.collect.lock"
SHOW = ("EDT", "C50", "T60", "FD", "RIR_to_GT_RIR_R@1")
FRAC_LABEL = {"025": "25 %", "050": "50 %", "075": "75 %", "100": "100 %"}


def mean_sd(values):
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def aggregate(records):
    """records: {seed: metrics dict} for one (run, K) -> {metric: (mean, sd, n)}."""
    if sorted(records) != list(ev.SEEDS):
        raise ValueError(f"seeds {sorted(records)} != {list(ev.SEEDS)}")
    return {k: (*mean_sd([records[s][k] for s in ev.SEEDS]), len(records)) for k in ev.METRIC_KEYS}


def _sha_map(paths):
    """{repo-relative path: sha256} of local files (the offline admission's inputs)."""
    return {os.path.relpath(ev._abs(p), reg.REPO): reg.sha256_file(ev._abs(p)) for p in paths}


def cell_files(cell):
    files = [cell["metrics"], cell["stream"], cell["receipt"]]
    return files + ([ev.dump_manifest_copy(cell)] if cell["dump"] else [])


def farm_files(run):
    """The farm copies of a run's training records that the offline admission re-opens (exp46_eval.prepare)."""
    farm = os.path.dirname(ev._abs(ev.run_cells(run)[0]["ckpt_link"]))
    cm = json.load(open(os.path.join(farm, reg.COMMIT_MARKER)))
    out = [os.path.join(farm, f) for f in ("CHECKPOINTS.sha256", reg.TRAINING_RECEIPT, reg.TRAINING_RECEIPT + ".sha256",
                                          reg.COMMIT_MARKER, reg.COMMIT_MARKER + ".sha256", cm["launch_record"])]
    for sub in (reg.RECEIPTS_DIR, "evidence"):
        out += [os.path.join(farm, sub, f) for f in sorted(os.listdir(os.path.join(farm, sub)))]
    return out


def c4f_rows(campaign):
    """(rows, manifest runs) of the campaign runs, every cell through exp46_eval's OFFLINE admission under ``campaign``;
    refuses (EvalError) if any cell is missing or not admitted (plan r3 §3.2; no bridge in exp_46)."""
    blocks = [(run, "c4f", reg.parse_run_id(run)[1], ev.run_cells(run)) for run in reg.campaign_runs()]
    rows, runs, problems = {}, {}, []
    for run, model, frac, cells in blocks:
        probs = [x for c in cells for x in ev.validate_cell(c, campaign, offline=True)]
        if probs:
            problems += probs
            continue
        files = [f for c in cells for f in cell_files(c)]
        farm = os.path.dirname(ev._abs(cells[0]["ckpt_link"]))
        platform = json.load(open(os.path.join(farm, reg.TRAINING_RECEIPT)))["platform"]
        if platform != reg.TRAINING_PLATFORM:
            problems.append(f"{run}: trained on {platform}, not {reg.TRAINING_PLATFORM}")
            continue
        files += farm_files(run)
        source = f"exp_46 {run}, trained on {platform} (eval pin {campaign['head'][:12]})"
        for K in (8, 1):
            recs = {c["seed"]: json.load(open(ev._abs(c["metrics"])))["metrics"] for c in cells if c["K"] == K}
            rows[(model, frac, K)] = {"stats": aggregate(recs), "source": source}
        runs[run] = _sha_map(files)
    if problems:
        raise ev.EvalError(f"{len(problems)} cell problem(s), nothing published: " + "; ".join(problems[:5]))
    return rows, runs


def e0_files(campaign):
    """The E0 receipt (+ sidecar), its eight cells and the reference manifest (the offline E0 admission's inputs)."""
    return [campaign["e0_receipt"], campaign["e0_receipt"] + ".sha256", ev.REF_MANIFEST] + \
        [f for c in ev.e0_cells() for f in (c["metrics"], c["stream"], c["receipt"])]


MODELS = {"c4f": "C4 + flip (C4F)"}


def render(rows, pending):
    out = ["# exp_46 c4flip_data_curve — results (generated by exp46_collect.py; never hand-edited)", "",
           "Full unseen split (6,337 items / 17 rooms), split-level item-weighted, mean ± sample SD over eval "
           "seeds 42–46, 40k-step EMA endpoints, one training seed (42) per run. EDT ms, C50 dB, T60 %, FD, R@1 %.", ""]
    for K in (8, 1):
        out += [f"## K = {K}" + (" (headline)" if K == 8 else ""), "",
                "| model | data | " + " | ".join(SHOW) + " | source |", "|---|---|" + "---|" * len(SHOW) + "---|"]
        for model, frac in [("c4f", reg.parse_run_id(r)[1]) for r in reg.campaign_runs()]:
            st = rows[(model, frac, K)]["stats"]
            cells = [f"{st[m][0]:.3f} ± {st[m][1]:.3f}" for m in SHOW]
            out.append(f"| {MODELS[model]} | {FRAC_LABEL[frac]} | " + " | ".join(cells) + f" | {rows[(model, frac, K)]['source']} |")
        out.append("")
    return "\n".join(out + reg.disclosure_block("collect")) + "\n"            # plan r3 §8 (finding 15)


def publish(rows, pending, manifest, repo=None):
    repo = repo or reg.REPO
    payload = {"rows": [{"model": k[0], "fraction": k[1], "K": k[2], "source": v["source"],
                         "stats": {m: {"mean": s[0], "sd": s[1], "n": s[2]} for m, s in v["stats"].items()}}
                        for k, v in sorted(rows.items())], "pending": pending, "manifest": manifest}
    jtext = (json.dumps(payload, indent=1, sort_keys=True) + "\n").encode()
    mtext = (render(rows, pending) +
             f"\nresults JSON sha256: `{hashlib.sha256(jtext).hexdigest()}`\n").encode()
    lock = reg.open_below(os.path.join(repo, LOCK), "a", os.path.join(repo, reg.EXP_DIR), "the publication lock")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        reg.atomic_write_bytes(os.path.join(repo, OUT_MD), mtext, root=os.path.join(repo, reg.EXP_DIR))     # first
        reg.atomic_write_bytes(os.path.join(repo, OUT_JSON), jtext, root=os.path.join(repo, reg.EXP_DIR))   # the commit point
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    return hashlib.sha256(jtext).hexdigest()


def collect():
    campaign = ev.validate_e0_receipt(offline=True)
    rows, runs = c4f_rows(campaign)
    manifest = {"campaign": {k: campaign[k] for k in ("head", "closure_sha256", "env_receipt_sha256",
                                                      "eval_snapshot_sha256", "sha256")},
                "artifacts": _sha_map(e0_files(campaign)), "runs": runs}
    return rows, [], manifest


def generator_static():
    """What exp46_eval's OFFLINE admission reads in another root besides the published evidence (the files a
    generator root must hold for the shared table's exp_46 rows): the exp_46 kit's exp46_eval.py + exp46_registry.py,
    the C4F and E0 model configs, exp_38's E0a decision + sidecar and its reference manifest + cells."""
    return (f"{reg.EXP_DIR}/exp46_eval.py", f"{reg.EXP_DIR}/exp46_registry.py", reg.ARMS["c4f"]["config"],
            *(v["config"] for v in reg.E0_CONFIGS.values()), reg.E0A_DECISION, reg.E0A_DECISION + ".sha256",
            reg.REF_MANIFEST, *(f"{reg.REF_DIR}/{r}" for r in reg.ref_set()))


def verify(root=None):
    """sha256-verify the published exp_46 results in ``root`` (finding 12): the Markdown names the JSON's sha256 and
    every file of the JSON's manifest (the E0 artifacts, every run's cells and farm records)
    re-hashes to its recorded value. Returns {relative path: sha256} of every verified file, the JSON and Markdown
    included; raises EvalError."""
    root = os.path.realpath(root or reg.REPO)
    try:
        jtext = open(os.path.join(root, OUT_JSON), "rb").read()
        md = open(os.path.join(root, OUT_MD)).read()
        man = json.loads(jtext)["manifest"]
        want = dict(man["artifacts"])
        for files in man["runs"].values():
            want.update(files)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as err:
        raise ev.EvalError(f"no published exp_46 results in {root}: {err}")
    jsha = hashlib.sha256(jtext).hexdigest()
    if f"results JSON sha256: `{jsha}`" not in md:
        raise ev.EvalError(f"{OUT_MD} in {root} does not name its results JSON's sha256 {jsha[:12]}")
    if any(os.path.isabs(rel) or ".." in rel.split("/") for rel in want):
        raise ev.EvalError("the results manifest names a path outside its root")
    bad = sorted(rel for rel, d in want.items() if not os.path.isfile(os.path.join(root, rel))
                 or reg.sha256_file(os.path.join(root, rel)) != d)
    if bad:
        raise ev.EvalError(f"{len(bad)} published file(s) in {root} missing or not their recorded sha256: {bad[:3]}")
    return dict(want, **{OUT_JSON: jsha, OUT_MD: reg.sha256_file(os.path.join(root, OUT_MD))})


def materialise(dest, root=None):
    """Copy the verified publication of ``root`` + generator_static() byte for byte to the same relative paths under
    ``dest`` (finding 12): an existing destination file must already be identical (else refuse, nothing overwritten),
    every copy is re-hashed, and ``dest`` is verified afterwards."""
    root = os.path.realpath(root or reg.REPO)
    if not os.path.isdir(dest):
        raise ev.EvalError(f"destination {dest} does not exist")
    probs = reg.isolation_problems(dest, "the materialise destination", trees=[root, dest], contains=False)
    if probs:
        raise ev.EvalError("isolation from exp_38 / exp_45 (exp_46 plan r3 §3.2): " + "; ".join(probs))
    files = verify(root)
    files.update({rel: reg.sha256_file(os.path.join(root, rel)) for rel in generator_static()})
    # every file is checked BEFORE any write (r2 finding 6): exp_38's own files (its reference cells, its E0a decision)
    # are verify-only -- exp_46 never writes into exp_38's namespaces -- and nothing is written through an alias
    exp38_owned = {rel for rel in files if rel.startswith((reg.EXP38_DIR + "/", reg.EXP38_FARM + "/"))}
    for rel in sorted(files):
        target = os.path.join(dest, rel)
        if rel in exp38_owned:
            if os.path.islink(target) or not os.path.isfile(target) or reg.sha256_file(target) != files[rel]:
                raise ev.EvalError(f"{target}: exp_38's read-only file is not in the destination at identical bytes; "
                                   "exp_46 never writes into exp_38's namespaces (it reaches a generator root only "
                                   "through exp_38's own publication, import-exp38 --generator-root)")
            continue
        problem = reg.alias_problem(dest, rel)
        if problem:
            raise ev.EvalError(f"{problem}: refused, nothing written")
    copied = same = 0
    for rel, digest in sorted(files.items()):
        target = os.path.join(dest, rel)
        if os.path.lexists(target):
            if os.path.islink(target) or not os.path.isfile(target) or reg.sha256_file(target) != digest:
                raise ev.EvalError(f"{target} exists with other bytes: refused, nothing overwritten")
            same += 1
            continue
        if reg.tracked_in_index(dest, rel):          # tracked but missing in the destination repo (r2 finding 9)
            raise ev.EvalError(f"{target} is tracked in the destination's index but missing: refused, never written")
        data = open(os.path.join(root, rel), "rb").read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ev.EvalError(f"{rel} changed during the copy")
        try:                                          # below the destination root (Codex code r2 finding 1)
            reg.makedirs_below(os.path.dirname(target), dest, "a publication directory")
        except reg.GateError as err:
            raise ev.EvalError(str(err))
        try:
            reg.publish_no_clobber(target, data, root=dest)    # no replacing rename: an entry created meanwhile is kept
        except FileExistsError:
            raise ev.EvalError(f"{target} appeared during the copy: refused, never overwritten")
        except reg.GateError as err:
            raise ev.EvalError(str(err))
        if reg.sha256_file(target) != digest:
            raise ev.EvalError(f"{target}: the copy does not re-hash to {digest[:12]}")
        copied += 1
    verify(dest)
    return {"dest": os.path.realpath(dest), "copied": copied, "already_identical": same, "files": len(files)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("verify")
    s.add_argument("--root")
    s = sub.add_parser("materialise")
    s.add_argument("--dest", required=True)
    a = p.parse_args(argv)
    os.chdir(reg.REPO)
    try:
        if a.cmd == "verify":
            files = verify(a.root)
            print(f"verified {len(files)} published file(s) in {os.path.realpath(a.root or reg.REPO)}")
            return 0
        if a.cmd == "materialise":
            print(json.dumps(materialise(a.dest)))
            return 0
        rows, pending, manifest = collect()
    except (ev.EvalError, reg.GateError, OSError, ValueError, KeyError) as err:
        print(f"EXP46 COLLECT REFUSED: {err}", file=sys.stderr)
        return 2
    sha = publish(rows, pending, manifest)
    print(f"published {OUT_MD} / {OUT_JSON} (json sha256 {sha[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
