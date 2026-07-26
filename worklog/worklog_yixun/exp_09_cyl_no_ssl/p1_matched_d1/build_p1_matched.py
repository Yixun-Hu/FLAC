#!/usr/bin/env python3
"""exp_06-addendum: build the matched-mode references from RAW P1 eval JSONs and run the
adapter -> aggregate. control_stats are computed from the 10 P1 artifacts (5-seed mean +
sample std, ddof=1 — the adapter/exp_01 convention), NEVER hand-written. The measured side
is the SAME committed no-SSL d1_manifest.json (unchanged). Refuses partial coverage."""
import glob
import json
import math
import os
import subprocess
import sys
import time

WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
PDIR = os.path.join(WT, "worklog/worklog_yixun/exp_09_cyl_no_ssl/p1_matched_d1")
DR = os.path.join(WT, "worklog/worklog_yixun/exp_09_cyl_no_ssl/d_records")
IMP = os.path.join(PDIR, "p1_import")
PY = "/home/yixunhu/miniconda3/envs/flac/bin/python"
ADAPTER = os.path.join(WT, "worklog/worklog_yixun/exp_09_cyl_no_ssl/gate_thresholds_to_verdicts.py")
AGG = os.path.join(WT, "worklog/worklog_yixun/exp_09_cyl_no_ssl/aggregate_gate.py")
SEEDS = (42, 43, 44, 45, 46)
METRICS = ("T60", "C50", "EDT", "RIR_to_GT_RIR_R@1")


def stats(vals):
    n = len(vals)
    mu = sum(vals) / n
    sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return [mu, sd]


def main():
    control, manifest_note = {}, {}
    for k in ("1", "8"):
        per_metric = {m: [] for m in METRICS}
        paths = {}
        for s in SEEDS:
            hits = sorted(glob.glob(os.path.join(IMP, f"*metrics*P1D1_K{k}_s{s}*.json")))
            if len(hits) != 1:
                sys.exit(f"REFUSE: expected exactly 1 P1 artifact for K={k} s={s}, got {len(hits)}")
            paths[str(s)] = hits[0]
            m = json.load(open(hits[0]))["metrics"]
            for met in METRICS:
                v = m[met]
                if not isinstance(v, (int, float)) or not math.isfinite(v):
                    sys.exit(f"REFUSE: non-finite {met} in {hits[0]}")
                per_metric[met].append(float(v))
        control[k] = {met: stats(vs) for met, vs in per_metric.items()}
        manifest_note[k] = paths
    tpl = json.load(open(os.path.join(PDIR, "references_matched.template.json")))
    assert tpl["d1"]["control_stats"] == "__FILLED_AT_RUNTIME_FROM_RAW_P1_EVALS__"
    tpl["d1"]["control_stats"] = control
    tpl["d1"]["control_artifacts"] = manifest_note   # provenance: seed -> raw P1 JSON path
    refs = os.path.join(PDIR, "references_matched.json")
    json.dump(tpl, open(refs, "w"), indent=2)
    print("control_stats:", json.dumps(control, indent=1))
    vd = os.path.join(PDIR, f"verdicts_matched_{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    r = subprocess.run([PY, ADAPTER, "--references", refs,
                        "--d1", os.path.join(DR, "d1_manifest.json"), "--out-dir", vd],
                       capture_output=True, text=True)
    print(r.stdout + (("--- stderr ---\n" + r.stderr) if r.stderr.strip() else ""))
    print(f"adapter rc={r.returncode}")
    if r.returncode == 2:
        sys.exit("REFUSE: adapter rejected inputs")
    a = subprocess.run([PY, AGG, "--out", os.path.join(vd, "aggregate_matched.json"),
                        "--require", "d1_parity", *sorted(glob.glob(os.path.join(vd, "verdict_*.json")))],
                       capture_output=True, text=True)
    print(a.stdout + (a.stderr if a.stderr.strip() else ""))
    print(f"aggregate rc={a.returncode}")
    print("VERDICTS DIR:", vd)


if __name__ == "__main__":
    main()
