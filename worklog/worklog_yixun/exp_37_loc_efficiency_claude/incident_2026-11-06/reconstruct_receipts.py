"""Read-only reconstruction of the 4 overwritten exp_35 receipts (.cell.json) (the 2026-11-06 incident) from the intact
T-dup receipt, the verified manifests in restore/ (``reconstruct.py`` first), the intact dump markers and the cells'
eval logs; verified against the digests the leaf recorded before the incident.

Its sibling ``reconstruct.py`` is imported from this script's RESOLVED directory, never from the current directory,
and every write goes through its confined, exclusive ``publish`` into ``<this directory>/restore/`` (Codex code review
3, H-03). Same arguments and exit codes as ``reconstruct.py``. Written by the exp_37 Coder (Claude Opus 5.5, max
effort); final fix round: fixed after review 3, not re-reviewed (cap).
"""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("exp37_incident_reconstruct", HERE / "reconstruct.py")
rec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rec)

LOGS = {"exp35_BF_T_S40000_K1_s42": "depth_canon_2026-11-05_21-06-30_eval_exp35_BF_T_S40000_K1_s42.log",
        "exp35_BF_T_S40000_K8_s42": "depth_canon_2026-11-05_21-31-54_eval_exp35_BF_T_S40000_K8_s42.log",
        "exp35_BF_C4_S40000_K1_s42_th90": "depth_canon_2026-11-06_07-28-04_eval_exp35_BF_C4_S40000_K1_s42_th90.log",
        "exp35_BF_C4_S40000_K8_s42_th90": "depth_canon_2026-11-06_07-53-46_eval_exp35_BF_C4_S40000_K8_s42_th90.log"}
DATASET = {1: ("src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json",
               "4875374040cad2272e1df2b99809743c7bcf3fd0718df40c6ebfaf9c84c8b4e3"),
           8: ("src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json",
               "063c66c2411cde4b1f07ec7c5331150b322517cf0067a0ef3def819368423b55")}


def main(argv=None) -> int:
    farm, dumps, orig = rec.parse(argv)
    template = json.loads((farm / f"{rec.DUP}.cell.json").read_text())
    files, ok = {}, True
    for suffix, (m_sha, s_sha, r_sha, man_sha) in orig.items():
        stem = f"epoch=8-step=40000_metrics_1_1.0_{suffix}"
        name = suffix.split("_fa_invariant")[0]
        k = 1 if "_K1_" in name else 8
        theta = 90 if "_th90" in name else 0
        manifest_bytes = rec.destination(f"{stem}.manifest.json").read_bytes()
        if rec.sha(manifest_bytes) != man_sha:
            print(f"{name}: the verified manifest is not in restore/ (run reconstruct.py first)")
            ok = False
            continue
        argv = list(template["argv"])
        argv[argv.index("--dataset-config") + 1] = DATASET[k][0]
        argv[argv.index("--rotate-deg") + 1] = str(theta)
        argv[argv.index("--eval-name") + 1] = name
        found = None
        for block in ("BF-pin", "BF-T", "BF-C4"):
            for lane in (0, 1):
                receipt = dict(template)
                receipt.update(cell=name, stem=stem, block=block, dataset_config=DATASET[k][0],
                               dataset_config_sha256=DATASET[k][1], argv=argv,
                               artifact_sha256={"metrics": m_sha, "stream": s_sha, "canon": None},
                               dump_evidence={"manifest_sha256": man_sha, "manifest": json.loads(manifest_bytes),
                                              "marker": (dumps / stem / "COMPLETE").read_text(),
                                              "waveforms_sidecar": (dumps / stem / "waveforms.sha256").read_text()},
                               lane=lane, log=f"worklog/worklog_yixun/exp_25_depth_canon_claude/{LOGS[name]}")
                raw = json.dumps(receipt, indent=2).encode()
                if rec.sha(raw) == r_sha:
                    found = (block, lane, raw)
        if found:
            files[f"{stem}.cell.json"] = found[2]
        print(f"{name}: receipt reconstructed byte-exact: {bool(found)}"
              + (f" (block {found[0]}, lane {found[1]})" if found else ""))
        ok &= bool(found)
    refused = rec.publish(files)
    return refused or (0 if ok else 1)


if __name__ == "__main__":
    sys.exit(main())
