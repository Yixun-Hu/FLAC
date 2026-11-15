#!/usr/bin/env python3
"""exp_38 collector: admitted cells -> data_efficiency_results.{json,md} (plan r2 §9.4, §10 B9).

Everything is bound to the sealed E0 receipt (``exp38_eval.validate_e0_receipt``: a PASS at THIS
closure). A cell is used only if ``exp38_eval.validate_cell`` admits it under that campaign (exact
metrics / by-scene / receipt schemas, protocol, the run's TRAINING_COMPLETE checkpoint sha + bytes,
stream pins, weight pins, the complete waveform dump); a run's block is published only when all 10
cells (K 8 and 1, seeds 42-46) are admitted, else it renders PENDING. Reference rows (dc_van, P1 /
YAW / B-F at 100 %) come from exactly the 68 registered ``ref_cells/`` files, each read once and
compared with the E0 receipt's sha. Publication (Codex code r1 HIGH 10): Markdown first, the JSON
LAST (the commit point; the Markdown carries the JSON's sha256), each temp -> fsync -> rename, then
a directory fsync, under the collector lock; the JSON carries the manifest of every admitted
artifact. Pending cells exit 1 unless ``--partial``.

    python exp38_collect.py [--bridge] [--partial]

Written by the exp_38 Coder (Claude Opus 5.5).
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
import exp38_eval as ev  # noqa: E402
import exp38_registry as reg  # noqa: E402

OUT_JSON = f"{reg.EXP_DIR}/data_efficiency_results.json"
OUT_MD = f"{reg.EXP_DIR}/data_efficiency_results.md"
LOCK = f"{reg.EXP_DIR}/.collect.lock"
SHOW = ("EDT", "C50", "T60", "FD", "RIR_to_GT_RIR_R@1")
MODELS = {"p1": "FLAC (P1)", "yaw": "Yaw-Aug FLAC (YAW)", "bf": "OrbitRIR (B-F)",
          "p1_bridge": "FLAC (P1) platform bridge (exp_38)"}
FRAC_LABEL = {"025": "25 %", "050": "50 %", "075": "75 %", "100": "100 %"}


def mean_sd(values):
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def aggregate(records):
    """records: {seed: metrics dict} for one (run, K) -> {metric: (mean, sd, n)}."""
    if sorted(records) != list(ev.SEEDS):
        raise ValueError(f"seeds {sorted(records)} != {list(ev.SEEDS)}")
    return {k: (*mean_sd([records[s][k] for s in ev.SEEDS]), len(records)) for k in ev.METRIC_KEYS}


def reference_rows(campaign):
    """Rows of the 68 registered reference cells; every file must be the one the E0 receipt bound."""
    refs = ev.check_refs()
    rec = json.load(open(campaign["e0_receipt"]))
    if rec["references"] != refs:
        raise ev.EvalError("reference cells differ from the E0 receipt's")
    rows, used = {}, {}
    for (model, frac, pattern) in [("p1", t, f"anchors/dc_van/dc_van_f{t}_K{{K}}_s{{s}}.json") for t in ("025", "050", "075")] + \
            [("p1", "100", "anchors/P1/P1_K{K}_s{s}.json"), ("yaw", "100", "anchors/YAW/YAW_K{K}_s{s}.json"),
             ("bf", "100", "anchors/BF/BF_K{K}_s{s}.json")]:
        for K in (8, 1):
            recs = {}
            for s_ in ev.SEEDS:
                rel = pattern.format(K=K, s=s_)
                recs[s_] = ev.load_ref(rel, refs)["metrics"]
                used[rel] = refs[rel]
            label = "exp_14" if (model, frac) in EXP14_ROWS and e0a_decided(campaign) else None
            rows[(model, frac, K)] = {"stats": aggregate(recs), "source": "reference (ref_cells/anchors)",
                                      "retrieval_evaluator": label}
    return rows, used


# the FLAC 25 / 50 / 75 % rows are exp_14's dc_van records: their acoustics replay exactly at our evaluator
# (E0a), their AGREE retrieval does not -- Yixun's decision (data_efficiency_e0a_decision.json): keep exp_14's
# values and LABEL their retrieval columns as exp_14-evaluator
EXP14_ROWS = {("p1", "025"), ("p1", "050"), ("p1", "075")}
RETRIEVAL_MARK = " †"


def e0a_decided(campaign):
    """True when the campaign's E0 receipt admitted E0a under the decision (retrieval deviations accepted)."""
    return json.load(open(campaign["e0_receipt"])).get("e0a_decision_sha256") is not None


def run_block(run, campaign, validate=None):
    """(rows for the run's two K, problems, {path: sha256} of every artifact the admission consumed)."""
    validate = validate or (lambda c: ev.validate_cell(c, campaign))
    arm, tag, _ = reg.parse_run_id(run)
    cells = ev.run_cells(run)
    problems = [x for c in cells for x in validate(c)]
    if problems:
        return {}, problems, {}
    model = "p1_bridge" if arm == "p1" else arm
    farm = os.path.dirname(ev._abs(cells[0]["ckpt_link"]))
    platform = json.load(open(os.path.join(farm, reg.TRAINING_RECEIPT)))["platform"]
    if platform != reg.TRAINING_PLATFORM:
        return {}, [f"{run}: trained on {platform}, D2 = {reg.TRAINING_PLATFORM} (not admitted)"], {}
    rows, manifest = {}, ev.run_artifacts(run)
    for c in cells:
        manifest.update(ev.cell_artifacts(c))
    for K in (8, 1):
        recs = {c["seed"]: json.load(open(ev._abs(c["metrics"])))["metrics"] for c in cells if c["K"] == K}
        rows[(model, tag, K)] = {"stats": aggregate(recs), "source": f"exp_38 {run}, trained on {platform} "
                                                                 f"(eval pin {campaign['head'][:12]})",
                                 "retrieval_evaluator": None}
    return rows, [], manifest


def render(rows, pending):
    out = ["# exp_38 data_efficiency — results (generated by exp38_collect.py; never hand-edited)", "",
           "Full unseen split (6,337 items / 17 rooms), split-level item-weighted, mean ± sample SD over eval "
           "seeds 42–46, 40k-step EMA endpoints, one training seed (42) per cell. EDT ms, C50 dB, T60 %, FD, R@1 %.", ""]
    for K in (8, 1):
        out += [f"## K = {K}" + (" (headline)" if K == 8 else ""), "",
                "| model | data | " + " | ".join(SHOW) + " | source |", "|---|---|" + "---|" * len(SHOW) + "---|"]
        for model in ("p1", "yaw", "bf", "p1_bridge"):
            for frac in ("025", "050", "075", "100"):
                key = (model, frac, K)
                if key in rows:
                    st = rows[key]["stats"]
                    mark = RETRIEVAL_MARK if rows[key].get("retrieval_evaluator") == "exp_14" else ""
                    cells = [f"{st[m][0]:.3f} ± {st[m][1]:.3f}" + (mark if m in ev.RETRIEVAL_KEYS else "") for m in SHOW]
                    out.append(f"| {MODELS[model]} | {FRAC_LABEL[frac]} | " + " | ".join(cells) + f" | {rows[key]['source']} |")
                elif (model in ("yaw", "bf") and frac != "100") or (model == "p1_bridge" and frac == "025"
                                                                     and "de_p1_f025" in reg.campaign_runs()):
                    out.append(f"| {MODELS[model]} | {FRAC_LABEL[frac]} | " + " | ".join(["PENDING"] * len(SHOW)) + " | — |")
        out.append("")
    if any(v.get("retrieval_evaluator") == "exp_14" for v in rows.values()):
        out += [f"{RETRIEVAL_MARK.strip()} FLAC 25 / 50 / 75 %: exp_14's own evaluator records. Their acoustic metrics "
                "(T60, Invalid T60, C50, EDT) and FD replay exactly at the exp_38 evaluator pin (E0a); their AGREE "
                "retrieval metrics -- R@1, R@5 and R@10, RIR->GT-RIR and RIR->geom, shown or not -- do not (AGREE "
                "samples encoder noise at evaluation), so ALL their retrieval values are exp_14-evaluator numbers, "
                "not comparable item-for-item with the exp_38-evaluator rows (Yixun's decision, "
                "`data_efficiency_e0a_decision.json`).", ""]
    if pending:
        out += ["## Pending / not admitted", ""] + [f"- {p}" for p in pending] + [""]
    return "\n".join(out) + "\n"


def publish(rows, pending, manifest, repo=None):
    repo = repo or reg.REPO
    payload = {"rows": [{"model": k[0], "fraction": k[1], "K": k[2], "source": v["source"],
                         "retrieval_evaluator": v.get("retrieval_evaluator") or "exp_38",
                         "exp_14_evaluator_metrics": sorted(ev.RETRIEVAL_KEYS) if v.get("retrieval_evaluator") == "exp_14" else [],
                         "stats": {m: {"mean": s[0], "sd": s[1], "n": s[2]} for m, s in v["stats"].items()}}
                        for k, v in sorted(rows.items())], "pending": pending, "manifest": manifest}
    jtext = (json.dumps(payload, indent=1, sort_keys=True) + "\n").encode()
    mtext = (render(rows, pending) + f"\nresults JSON sha256: `{hashlib.sha256(jtext).hexdigest()}`\n").encode()
    lock = open(os.path.join(repo, LOCK), "a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        reg.atomic_write_bytes(os.path.join(repo, OUT_MD), mtext)      # first
        reg.atomic_write_bytes(os.path.join(repo, OUT_JSON), jtext)    # last: the commit point
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    return hashlib.sha256(jtext).hexdigest()


def collect(bridge=None):
    campaign = ev.validate_e0_receipt()
    rows, used = reference_rows(campaign)
    manifest = {"campaign": {k: campaign[k] for k in ("head", "closure_sha256", "env_receipt_sha256",
                                                      "eval_snapshot_sha256", "sha256")},
                "artifacts": ev.campaign_artifacts(campaign), "runs": {}}
    pending = []
    runs = reg.campaign_runs() + (["de_p1_f025"] if bridge and "de_p1_f025" not in reg.campaign_runs() else [])
    for run in runs:
        r, probs, arts = run_block(run, campaign)
        rows.update(r)
        pending += probs
        if arts:
            manifest["runs"][run] = arts
    assert set(used) <= {os.path.relpath(k, os.path.join(reg.REPO, reg.EXP_DIR, "ref_cells"))
                         for k in manifest["artifacts"]}
    return rows, pending, manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--bridge", action="store_true")
    p.add_argument("--partial", action="store_true", help="publish although cells are pending (exit 0)")
    a = p.parse_args(argv)
    os.chdir(reg.REPO)
    try:
        rows, pending, manifest = collect(a.bridge)
    except (ev.EvalError, reg.GateError, OSError, ValueError, KeyError) as err:
        print(f"EXP38 COLLECT REFUSED: {err}", file=sys.stderr)
        return 2
    sha = publish(rows, pending, manifest)
    print(f"published {OUT_MD} / {OUT_JSON} (json sha256 {sha[:12]}), {len(pending)} pending item(s)")
    return 0 if (not pending or a.partial) else 1


if __name__ == "__main__":
    sys.exit(main())
