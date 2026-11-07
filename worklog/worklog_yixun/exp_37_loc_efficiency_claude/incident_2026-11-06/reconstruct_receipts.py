"""Read-only reconstruction of the 4 overwritten exp_35 receipts (.cell.json) from the intact T-dup receipt, the
reconstructed manifests, the intact dump markers and the cells' eval logs; verified against the digests the leaf
recorded before the incident. Writes ONLY into ./restore/."""
import hashlib, json, sys
from pathlib import Path

sys.path.insert(0, ".")
from reconstruct import ORIG, FARM, DUMPS, OUT  # noqa: E402  (re-runs the verified manifest reconstruction)

LOGS = {"exp35_BF_T_S40000_K1_s42": "depth_canon_2026-11-05_21-06-30_eval_exp35_BF_T_S40000_K1_s42.log",
        "exp35_BF_T_S40000_K8_s42": "depth_canon_2026-11-05_21-31-54_eval_exp35_BF_T_S40000_K8_s42.log",
        "exp35_BF_C4_S40000_K1_s42_th90": "depth_canon_2026-11-06_07-28-04_eval_exp35_BF_C4_S40000_K1_s42_th90.log",
        "exp35_BF_C4_S40000_K8_s42_th90": "depth_canon_2026-11-06_07-53-46_eval_exp35_BF_C4_S40000_K8_s42_th90.log"}
DATASET = {1: ("src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json",
               "4875374040cad2272e1df2b99809743c7bcf3fd0718df40c6ebfaf9c84c8b4e3"),
           8: ("src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json",
               "063c66c2411cde4b1f07ec7c5331150b322517cf0067a0ef3def819368423b55")}
sha = lambda b: hashlib.sha256(b).hexdigest()
template = json.loads((FARM / "epoch=8-step=40000_metrics_1_1.0_exp35_BF_T_S40000_K1_s42_dup_fa_invariant_a4.cell.json").read_text())
ok = True
for suffix, (m_sha, s_sha, r_sha, man_sha) in ORIG.items():
    stem = f"epoch=8-step=40000_metrics_1_1.0_{suffix}"
    name = suffix.split("_fa_invariant")[0]
    k = 1 if "_K1_" in name else 8
    theta = 90 if "_th90" in name else 0
    manifest_bytes = (OUT / f"{stem}.manifest.json").read_bytes()
    assert sha(manifest_bytes) == man_sha
    argv = list(template["argv"])
    argv[argv.index("--dataset-config") + 1] = DATASET[k][0]
    argv[argv.index("--rotate-deg") + 1] = str(theta)
    argv[argv.index("--eval-name") + 1] = name
    found = None
    for block in ("BF-pin", "BF-T", "BF-C4"):
        for lane in (0, 1):
            r = dict(template)
            r.update(cell=name, stem=stem, block=block, dataset_config=DATASET[k][0], dataset_config_sha256=DATASET[k][1],
                     argv=argv, artifact_sha256={"metrics": m_sha, "stream": s_sha, "canon": None},
                     dump_evidence={"manifest_sha256": man_sha, "manifest": json.loads(manifest_bytes),
                                    "marker": (DUMPS / stem / "COMPLETE").read_text(),
                                    "waveforms_sidecar": (DUMPS / stem / "waveforms.sha256").read_text()},
                     lane=lane, log=f"worklog/worklog_yixun/exp_25_depth_canon_claude/{LOGS[name]}")
            raw = json.dumps(r, indent=2).encode()
            if sha(raw) == r_sha:
                found = (block, lane, raw)
    if found:
        (OUT / f"{stem}.cell.json").write_bytes(found[2])
    print(f"{name}: receipt reconstructed byte-exact: {bool(found)}" + (f" (block {found[0]}, lane {found[1]})" if found else ""))
    ok &= bool(found)
sys.exit(0 if ok else 1)
