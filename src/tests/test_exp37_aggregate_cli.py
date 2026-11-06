"""exp_37 round B-c: the B2 / B3 aggregation CLI (``tools/exp37_aggregate_b2b3.py``) end to end on a synthetic world:
B2 accuracy per arm x grid over seeds (0.5 m, its nested 1.0 m and c2f readouts, the 0.75 m runs), the time axis of
every point (L_q = core + shape-matched selector; c2f = its sequential total), the internal Pareto frontier, the B3
orbit x theta table and orbit latencies, markdown, and the two-panel Pareto figure (PNG + PDF).
CPU only. Written by the exp_37 Coder (Claude Opus 5.5, max effort), round B-c.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.localization import exp37_latency as lat
from src.localization.exp37_grid import nested_subset
from src.localization.pilot import canonical_sha256
from src.localization.runner import initialize_run, save_query_result
from tools import exp37_aggregate_b2b3 as tool

C4 = [0.0, 90.0, 180.0, 270.0]
HALF = np.array([[x, y, 1.0] for x in np.arange(0.0, 3.01, 0.5) for y in np.arange(0.0, 2.01, 0.5)], dtype=np.float32)
SEVENTY_FIVE = np.array([[x, 0.0, 0.75] for x in np.arange(0.0, 3.01, 0.75)], dtype=np.float32)
HOST = {"pid": 42, "cuda_device": {"uuid": "6ac2604a-1111"}, "gpu_compute_apps": {"lines": ["GPU-6ac2604a-1111, 42, py, 1 MiB"]},
        "top_cpu_processes": {"lines": []}}


def _hashed(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**body, "sha256": canonical_sha256(body)}))
    return {**body, "sha256": canonical_sha256(body)}


class World:
    def __init__(self, root: Path):
        self.root = root
        self.pilot = self._pilot("pilot_0p50.json", len(HALF))
        self.pilot75 = self._pilot("pilot_0p75.json", len(SEVENTY_FIVE))
        index = nested_subset(HALF)
        recs = [{"index": r["index"], "query_id": r["query_id"], "room": r["room"], "candidate_count": len(index),
                 "candidate_indices_sha256": hashlib.sha256(index.astype("<u4").tobytes()).hexdigest()}
                for r in self.pilot["records"]]
        self.nested = _hashed(root / "nested.json", {"subset": "nested_1m_integer", "records": recs, "query_count": 2,
                                                     "parent_pilot_sha256": self.pilot["sha256"]})

    def _pilot(self, name, count):
        records = [{"index": i, "query_id": f"q{i}.wav", "room": f"r{i}", "candidate_count": count} for i in (1, 2)]
        return _hashed(self.root / name, {"schema_version": 1, "query_count": 2, "records": records})

    def selector(self, name, selection, seconds):
        rows = [{"query_index": r["index"], "query_id": r["query_id"], "room": r["room"],
                 "candidate_count": r["candidate_count"], "peak_memory_allocated_bytes": 5,
                 "agree": {"median_seconds": {"scoring_total_seconds": seconds}}} for r in selection["records"]]
        _hashed(self.root / name, {"record_type": "exp37_agree_selector_latency", "repeat_count": 3,
                                   "candidate_batch_size": 64, "host": HOST, "queries": rows})
        return str(self.root / name)

    def run(self, name, arm, pilot, cands, *, seed=42, core=1.0, extra=None, per_query=None, timing=True, angles=None):
        cond = lat.ARM_PROTOCOL[arm]["cond"]
        identity = {"conditioning_method": cond, "checkpoint_sha256": lat.ARM_PROTOCOL[arm]["ckpt"], "n_context": 8,
                    "pilot_manifest_sha256": pilot["sha256"], "query_indices": [1, 2], "sample_seed": seed,
                    "frame_average_angles": (angles or C4) if cond == "fa_invariant" else None, "tau": 0.1,
                    "score_sample_counts": [1], "candidate_batch_size": 64, "attention_backend": "sdpa",
                    **(lat.TIMING_IDENTITY if timing else {}), **(extra or {})}
        run_dir = self.root / name
        manifest = initialize_run(run_dir, identity)
        truth = np.array([1.0, 1.0, 1.0])
        for r in pilot["records"]:
            c = cands(r["index"]) if callable(cands) else cands
            sims = (1.0 - np.linalg.norm(c - truth, axis=1) / 10 + 0.01 * seed * (np.arange(len(c)) % 2)).astype(np.float32)
            win = int(np.argmax(sims.astype(np.float64)))
            result = {"run_manifest_sha256": manifest["sha256"], "query_index": r["index"], "query_id": r["query_id"],
                      "room": r["room"], "candidate_count": len(c), "peak_memory_bytes": 7, "source_global": truth.tolist(),
                      "metrics": {"1": {"prediction_index": win, "localization_error_m": float(np.linalg.norm(c[win] - truth))}},
                      "latency_protocol": {"name": "fem_core_aligned_kctx8_kgen1", "context_count": 8},
                      "latency_seconds": {"context_conditioning": 0.1, "candidate_conditioning_and_generation": core - 0.1,
                                          "core_forward_total": core}, **((per_query or {}).get(r["index"], {}))}
            save_query_result(run_dir, result=result, candidates=c, similarities=sims.reshape(-1, 1))
        (run_dir / "host_records").mkdir(exist_ok=True)
        (run_dir / "host_records" / "end_x_pid42.json").write_text(json.dumps({"label": "end", "snapshot": HOST}))
        return str(run_dir)


@pytest.fixture
def spec(tmp_path):
    w = World(tmp_path)
    b2 = [{"arm": arm, "grid": "0.50", "seed": s, "dir": w.run(f"{arm}_050_{s}", arm, w.pilot, HALF, seed=s, core=2.0)}
          for arm in ("P1", "BF") for s in (42, 43, 44)]
    b2 += [{"arm": arm, "grid": "0.75", "seed": s, "dir": w.run(f"{arm}_075_{s}", arm, w.pilot75, SEVENTY_FIVE, seed=s,
                                                                 core=0.6)} for arm in ("P1", "BF") for s in (42, 43, 44)]
    index = nested_subset(HALF)
    sub_sha = hashlib.sha256(index.astype("<u4").tobytes()).hexdigest()
    timing = [{"arm": arm, "kind": "nested_1m", "dir": w.run(f"{arm}_nested", arm, w.pilot, HALF[index], core=0.4,
                                                              extra={"candidate_subset": "nested_1m"},
                                                              per_query={i: {"candidate_subset": {"indices_sha256": sub_sha}}
                                                                         for i in (1, 2)})} for arm in ("P1", "BF")]
    c2f_q = {i: {"latency_protocol": {"name": "exp37_c2f_sequential_kctx8_kgen1", "context_count": 8, "top_k": 5},
                 "latency_seconds": {"c2f_total": 0.9, "core_forward_total": 0.7},
                 "c2f": {"generated_count": 12, "union_indices": list(range(12))}} for i in (1, 2)}
    timing += [{"arm": arm, "kind": "c2f", "dir": w.run(f"{arm}_c2f", arm, w.pilot, HALF[:12], per_query=c2f_q,
                                                         extra={"candidate_subset": "c2f", "c2f_top_k": 5,
                                                                "c2f_chebyshev_radius_m": 0.5})} for arm in ("P1", "BF")]
    b3 = [{"orbit": o, "theta": t, "seed": s,
           "dir": w.run(f"b3_{o}_{t}_{s}", "BF", w.pilot, HALF, seed=s, angles=angles, timing=(t == 0),
                        extra={"rotate_deg": float(t)} if t else None)}
          for o, angles in (("0", [0.0]), ("0_180", [0.0, 180.0])) for t in (0, 90) for s in (42, 43, 44)]
    lat_runs = [{"orbit": o, "dir": w.run(f"lat_{o}", "BF", w.pilot, HALF, core=3.0, angles=a)}
                for o, a in (("0", [0.0]), ("0_180", [0.0, 180.0]))]
    body = {"pilot_0p50": str(tmp_path / "pilot_0p50.json"), "pilot_0p75": str(tmp_path / "pilot_0p75.json"),
            "nested_manifest": str(tmp_path / "nested.json"), "selector_0p50": w.selector("s050.json", w.pilot, 0.5),
            "selector_0p75": w.selector("s075.json", w.pilot75, 0.2), "selector_nested": w.selector("s100.json", w.nested, 0.1),
            "b2_runs": b2, "b2_timing": timing, "b3_runs": b3,
            "b3_latency": {"selection": str(tmp_path / "pilot_0p50.json"), "selector": str(tmp_path / "s050.json"),
                           "runs": lat_runs}}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(body))
    return path


def test_the_b2_points_carry_seed_statistics_time_and_validity(spec):
    out = tool.build_summary(json.loads(spec.read_text()))
    points = {(p["arm"], p["grid"]): p for p in out["b2"]["points"]}
    assert set(points) == {(a, g) for a in ("P1", "BF") for g in ("0.50", "0.75", "1.00_nested", "c2f_k5")}
    half = points[("P1", "0.50")]
    assert half["seconds_per_query"] == pytest.approx(2.5) and half["seconds_per_query_repeats"] == {"43": 2.5, "44": 2.5}
    assert half["accuracy"]["median_localization_error_m"]["n"] == 3 and half["timing_valid"] is True
    assert points[("BF", "1.00_nested")]["seconds_per_query"] == pytest.approx(0.5)          # 0.4 core + 0.1 selector
    assert points[("BF", "c2f_k5")]["seconds_per_query"] == pytest.approx(0.9)               # the sequential total
    assert points[("P1", "0.75")]["seconds_per_query"] == pytest.approx(0.8)
    assert {s["grid"] for s in out["b2"]["sensitivity"]} == {"c2f_k1", "c2f_k3", "c2f_k10"}
    frontier = out["b2"]["frontier"]["median_localization_error_m"]
    assert frontier and all(isinstance(i, int) for i in frontier)


def test_the_b3_table_pairs_theta_per_orbit_and_reports_orbit_latency(spec):
    out = tool.build_summary(json.loads(spec.read_text()))
    rows = {(r["orbit"], r["theta"]): r for r in out["b3"]["accuracy"]}
    assert set(rows) == {(o, t) for o in ("0", "0_180") for t in (0, 90)}
    assert rows[("0", 90)]["accuracy"]["median_localization_error_m"]["n"] == 3
    latency = {r["orbit"]: r for r in out["b3"]["latency"]}
    assert latency["0"]["overall"]["mean_seconds"] == pytest.approx(3.5) and latency["0_180"]["timing_valid"] is True


def test_a_run_of_the_wrong_orbit_or_arm_is_refused(spec, tmp_path):
    body = json.loads(spec.read_text())
    body["b3_runs"][0]["orbit"] = "0_180"                                         # its identity says {0}
    with pytest.raises(RuntimeError, match="frame_average_angles"):
        tool.build_summary(body)
    body = json.loads(spec.read_text())
    body["b2_runs"][0]["arm"] = "BF"                                              # a P1 run claimed as B-F
    with pytest.raises(RuntimeError):
        tool.build_summary(body)


def test_the_cli_writes_json_markdown_and_both_figure_files(spec, tmp_path):
    import subprocess
    import sys
    stem = tmp_path / "fig" / "b2_pareto"
    argv = [sys.executable, str(Path(tool.__file__)), "--spec", str(spec), "--output-json", str(tmp_path / "o.json"),
            "--output-md", str(tmp_path / "o.md"), "--figure-stem", str(stem)]
    done = subprocess.run(argv, capture_output=True, text=True, timeout=600,
                          env={"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin",
                               "MPLCONFIGDIR": str(tmp_path / "mpl")})
    assert done.returncode == 0, done.stderr
    assert (stem.with_suffix(".png")).stat().st_size > 10000 and (stem.with_suffix(".pdf")).stat().st_size > 1000
    text = (tmp_path / "o.md").read_text()
    assert "Pareto" in text and "OrbitRIR" in text and "c2f" in text
    assert json.loads((tmp_path / "o.json").read_text())["sha256"]


def test_timing_validity_is_per_seed_headline_and_repeats(spec):
    """A co-tenant during a seed-43 repeat must not invalidate the valid seed-42 headline (and vice versa)."""
    body = json.loads(spec.read_text())
    s43 = next(item["dir"] for item in body["b2_runs"] if (item["arm"], item["grid"], item["seed"]) == ("P1", "0.50", 43))
    busy = dict(HOST, gpu_compute_apps={"lines": ["GPU-6ac2604a-1111, 42, py, 1 MiB", "GPU-6ac2604a-1111, 77, py, 9 MiB"]})
    (Path(s43) / "host_records" / "end_x_pid42.json").write_text(json.dumps({"label": "end", "snapshot": busy}))
    point = next(p for p in tool.build_summary(body)["b2"]["points"] if (p["arm"], p["grid"]) == ("P1", "0.50"))
    assert point["timing_valid"] is True and point["repeat_timing_valid"] == {"43": False, "44": True}
