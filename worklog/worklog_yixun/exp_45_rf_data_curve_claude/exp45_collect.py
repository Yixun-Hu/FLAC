#!/usr/bin/env python3
"""exp_45 collector (forked from exp_38's exp38_collect.py; plan_rf_data_curve.md r3 §3.2): the 30 RF result cells,
the 10 RF-100 bridge cells and the E0 receipt -> rf_data_curve_results.{json,md}.

rf_rows() replaces exp_38's reference_rows() / EXP14_ROWS / P1-bridge path (Codex plan r2 finding 3): every cell
goes through exp45_eval's OFFLINE admission (validate_cell(..., offline=True): exact metrics / by-scene /
receipt schemas, protocol, the run's TRAINING_COMPLETE copy in the farm -- or, for a bridge cell, the
RF100_BRIDGE descriptor --, the stream's input AND assignment hashes, the dump-manifest copy) under the sealed
E0 receipt (validate_e0_receipt(offline=True)), so the results re-derive from the tracked evidence alone; NAS
payloads (checkpoints, waveforms) are attested by the hashes the receipts carry, never opened. Any missing or
unadmitted cell refuses: nothing publishes partially. The RF-100 bridge rows publish only with the sealed bridge
comparison rf_data_curve_bridge.json of THIS campaign and these ten cells (exp45_eval.validate_bridge_result; Codex
code r1 finding 4); its verdict -- equal, or an evaluator discrepancy per cell -- is carried into the rows' source,
the Markdown and the JSON manifest, never turned into an equality. Publication (exp_38 Codex code r1 HIGH 10): Markdown first,
the JSON LAST (the commit point; the Markdown carries the JSON's sha256), each temp -> fsync -> rename, under the
collector lock; the JSON carries {path: sha256} of every file the admission consumed.

    python exp45_collect.py

Written by the exp_38 Coder (Claude Opus 5.5); exp_45 deltas by the exp_45 Coder (Claude Opus 5.5).
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
import exp45_eval as ev  # noqa: E402
import exp45_registry as reg  # noqa: E402

OUT_JSON = f"{reg.EXP_DIR}/rf_data_curve_results.json"
OUT_MD = f"{reg.EXP_DIR}/rf_data_curve_results.md"
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
    """The farm copies of a run's training records that the offline admission re-opens (exp45_eval.prepare)."""
    farm = os.path.dirname(ev._abs(ev.run_cells(run)[0]["ckpt_link"]))
    cm = json.load(open(os.path.join(farm, reg.COMMIT_MARKER)))
    out = [os.path.join(farm, f) for f in ("CHECKPOINTS.sha256", reg.TRAINING_RECEIPT, reg.TRAINING_RECEIPT + ".sha256",
                                          reg.COMMIT_MARKER, reg.COMMIT_MARKER + ".sha256", cm["launch_record"])]
    for sub in (reg.RECEIPTS_DIR, "evidence"):
        out += [os.path.join(farm, sub, f) for f in sorted(os.listdir(os.path.join(farm, sub)))]
    return out


def bridge_summary(campaign):
    """(the manifest's "bridge" entry, the result's files) of the sealed bridge comparison (finding 4)."""
    rec, digest = ev.validate_bridge_result(campaign)
    disc = {n: v["differences"] for n, v in sorted(rec["cells"].items()) if v["verdict"] != "equal"}
    path = ev._abs(ev.BRIDGE_RESULT)
    return ({"result": os.path.relpath(path, reg.REPO), "sha256": digest, "verdict": rec["verdict"],
             "descriptor_sha256": rec["descriptor_sha256"], "admission_record_sha256": rec["admission_record"]["sha256"],
             "discrepancies": disc}, [path, path + ".sha256"])


def rf_rows(campaign):
    """(rows, manifest runs, bridge) of the three RF runs and the RF-100 bridge, every cell through exp45_eval's OFFLINE
    admission under ``campaign`` and the bridge rows through the sealed bridge comparison; refuses (EvalError) if any
    cell is missing or not admitted, or the comparison is absent / not this campaign's (plan r3 §3.2, finding 4)."""
    blocks = [(run, "rf", reg.parse_run_id(run)[1], ev.run_cells(run)) for run in reg.campaign_runs()]
    blocks.append((ev.BRIDGE_RUN, "rf_bridge", "100", ev.bridge_cells()))
    rows, runs, problems, bridge = {}, {}, [], None
    for run, model, frac, cells in blocks:
        probs = [x for c in cells for x in ev.validate_cell(c, campaign, offline=True)]
        if probs:
            problems += probs
            continue
        files = [f for c in cells for f in cell_files(c)]
        if model == "rf_bridge":
            bridge, bfiles = bridge_summary(campaign)
            files += bfiles
        if model == "rf":
            farm = os.path.dirname(ev._abs(cells[0]["ckpt_link"]))
            platform = json.load(open(os.path.join(farm, reg.TRAINING_RECEIPT)))["platform"]
            if platform != reg.TRAINING_PLATFORM:
                problems.append(f"{run}: trained on {platform}, not {reg.TRAINING_PLATFORM}")
                continue
            files += farm_files(run)
            source = f"exp_45 {run}, trained on {platform} (eval pin {campaign['head'][:12]})"
        else:
            verdict = ("equal on all ten cells" if bridge["verdict"] == "equal" else
                       f"EVALUATOR DISCREPANCY in {len(bridge['discrepancies'])} of 10 cells (an evaluator discrepancy "
                       "until explained, never an equality)")
            source = (f"exp_43's RF 100 % checkpoint {ev.RF100_BRIDGE['sha256'][:12]} re-evaluated at the exp_45 pin "
                      f"{campaign['head'][:12]} (RF100_BRIDGE); bridge comparison: {verdict}")
        for K in (8, 1):
            recs = {c["seed"]: json.load(open(ev._abs(c["metrics"])))["metrics"] for c in cells if c["K"] == K}
            rows[(model, frac, K)] = {"stats": aggregate(recs), "source": source}
        runs[run] = _sha_map(files)
    if problems:
        raise ev.EvalError(f"{len(problems)} cell problem(s), nothing published: " + "; ".join(problems[:5]))
    return rows, runs, bridge


def e0_files(campaign):
    """The E0 receipt (+ sidecar), its eight cells and the reference manifest (the offline E0 admission's inputs)."""
    return [campaign["e0_receipt"], campaign["e0_receipt"] + ".sha256", ev.REF_MANIFEST] + \
        [f for c in ev.e0_cells() for f in (c["metrics"], c["stream"], c["receipt"])]


MODELS = {"rf": "Roll + Flip (RF)", "rf_bridge": "Roll + Flip (RF), exp_43's 100 % checkpoint (bridge)"}


def render(rows, pending, bridge=None):
    out = ["# exp_45 rf_data_curve — results (generated by exp45_collect.py; never hand-edited)", "",
           "Full unseen split (6,337 items / 17 rooms), split-level item-weighted, mean ± sample SD over eval "
           "seeds 42–46, 40k-step EMA endpoints, one training seed (42) per run. EDT ms, C50 dB, T60 %, FD, R@1 %.", ""]
    for K in (8, 1):
        out += [f"## K = {K}" + (" (headline)" if K == 8 else ""), "",
                "| model | data | " + " | ".join(SHOW) + " | source |", "|---|---|" + "---|" * len(SHOW) + "---|"]
        for model, frac in [("rf", t) for t in ("025", "050", "075")] + [("rf_bridge", "100")]:
            st = rows[(model, frac, K)]["stats"]
            cells = [f"{st[m][0]:.3f} ± {st[m][1]:.3f}" for m in SHOW]
            out.append(f"| {MODELS[model]} | {FRAC_LABEL[frac]} | " + " | ".join(cells) + f" | {rows[(model, frac, K)]['source']} |")
        out.append("")
    if bridge is not None:                               # the sealed bridge comparison, as recorded (finding 4)
        out.append(f"**RF-100 bridge comparison** (`{os.path.basename(bridge['result'])}`, sha256 "
                   f"`{bridge['sha256'][:12]}`): " + ("equal on all ten cells." if bridge["verdict"] == "equal" else
                                                     f"**evaluator discrepancy** in {len(bridge['discrepancies'])} of 10 "
                                                     "cells (reported until explained, never as equality):"))
        for cell, diffs in sorted(bridge["discrepancies"].items()):
            out.append(f"- {cell}: " + "; ".join(f"{k} ours {v['ours']!r} / reference {v['reference']!r}"
                                                for k, v in sorted(diffs.items())))
        out.append("")
    return "\n".join(out + reg.disclosure_block("collect")) + "\n"            # plan r3 §8 (finding 15)


def publish(rows, pending, manifest, repo=None):
    repo = repo or reg.REPO
    payload = {"rows": [{"model": k[0], "fraction": k[1], "K": k[2], "source": v["source"],
                         "stats": {m: {"mean": s[0], "sd": s[1], "n": s[2]} for m, s in v["stats"].items()}}
                        for k, v in sorted(rows.items())], "pending": pending, "manifest": manifest}
    jtext = (json.dumps(payload, indent=1, sort_keys=True) + "\n").encode()
    mtext = (render(rows, pending, manifest.get("bridge")) +
             f"\nresults JSON sha256: `{hashlib.sha256(jtext).hexdigest()}`\n").encode()
    lock = open(os.path.join(repo, LOCK), "a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        reg.atomic_write_bytes(os.path.join(repo, OUT_MD), mtext)      # first
        reg.atomic_write_bytes(os.path.join(repo, OUT_JSON), jtext)    # last: the commit point
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    return hashlib.sha256(jtext).hexdigest()


def collect():
    campaign = ev.validate_e0_receipt(offline=True)
    rows, runs, bridge = rf_rows(campaign)
    manifest = {"campaign": {k: campaign[k] for k in ("head", "closure_sha256", "env_receipt_sha256",
                                                      "eval_snapshot_sha256", "sha256")},
                "artifacts": _sha_map(e0_files(campaign)), "runs": runs, "bridge": bridge}
    return rows, [], manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.parse_args(argv)
    os.chdir(reg.REPO)
    try:
        rows, pending, manifest = collect()
    except (ev.EvalError, reg.GateError, OSError, ValueError, KeyError) as err:
        print(f"EXP45 COLLECT REFUSED: {err}", file=sys.stderr)
        return 2
    sha = publish(rows, pending, manifest)
    print(f"published {OUT_MD} / {OUT_JSON} (json sha256 {sha[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
