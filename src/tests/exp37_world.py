"""exp_37 test support: a COMPLETE synthetic registered B2 / B3 world on CPU (Codex code review 1, BLOCKING 4).

Every registered cell exists exactly as the runner writes it: B2 = P1 / BF x {0.50, 0.75} x seeds 42-44 (seed 42 a
monitored timing pass + its bitwise artifact replay; 43 / 44 accuracy-only runs with inline dumps), the nested-1.0 m
and c2f timing passes per arm; B3 = the 15 accuracy-only orbit x theta x seed runs (C4 theta 0 = the B2 B-F 0.50 runs)
and the three orbit-latency passes; bound + attested selectors; a synthesis section; external references. Two
queries on a 5 x 3 lattice at 0.5 m (15 candidates; the nested 1.0 m subset has 6). ``World.inputs`` is the fixed
input map the spec builder consumes. Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 1.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from exp37_fakes import attest_run, attest_selector, example_identity, selector_binding, selector_fields
from src.localization import exp37_grid as grid
from src.localization import exp37_identity as ident
from src.localization import exp37_waveforms as wav
from src.localization.pilot import canonical_sha256
from src.localization.runner import initialize_run, save_query_result

HALF = np.array([[x, y, 1.0] for x in np.arange(0.0, 2.01, 0.5) for y in np.arange(0.0, 1.01, 0.5)], dtype=np.float32)
SEVENTY_FIVE = np.array([[x, 0.0, 0.75] for x in np.arange(0.0, 1.51, 0.75)], dtype=np.float32)
TRUTH = [1.0, 1.0, 1.0]
ORBIT_ANGLES = {"0": [0.0], "0_180": [0.0, 180.0], "c4": None}
ORBIT_STEM = {"0": "o0", "0_180": "o0_180", "c4": "c4"}


def hashed(path, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**body, "sha256": canonical_sha256(body)}
    path.write_text(json.dumps(payload))
    return payload


def sims(c, seed):
    return (1.0 - np.linalg.norm(np.asarray(c) - np.array(TRUTH), axis=1) / 10
            + 0.01 * seed * (np.arange(len(c)) % 2)).astype(np.float32)


class World:
    def __init__(self, root: Path):
        self.root, self.runs, self.waves = Path(root), Path(root) / "runs", Path(root) / "waves"
        self.audits = {name: hashed(self.root / f"{name}.json", {"mesh_root": f"/m/{name}", "grid": name,
                                                                 "rooms": {"R": {"mesh_sha256": "b" * 64}}})
                       for name in ("audit_0p50", "audit_0p75")}
        self.pilot = self._pilot("pilot_0p50.json", len(HALF))
        self.pilot75 = self._pilot("pilot_0p75.json", len(SEVENTY_FIVE))
        index = grid.nested_subset(HALF)
        self.nested_sha = hashlib.sha256(index.astype("<u4").tobytes()).hexdigest()
        records = [{"index": r["index"], "query_id": r["query_id"], "room": r["room"], "candidate_count": len(index),
                    "candidate_indices_sha256": self.nested_sha} for r in self.pilot["records"]]
        self.nested = hashed(self.root / "nested.json", {"subset": "nested_1m_integer", "records": records,
                                                         "query_count": 2, "parent_pilot_sha256": self.pilot["sha256"]})
        self.selectors = {"s050": self.selector("s050.json", self.pilot, 0.5),
                          "s075": self.selector("s075.json", self.pilot75, 0.2),
                          "s100": self.selector("s100.json", self.nested, 0.1)}

    def _pilot(self, name, count):
        records = [{"index": i, "query_id": f"q{i}.wav", "room": f"r{i}", "candidate_count": count,
                    "candidate_indices_sha256": f"h{i}"} for i in (1, 2)]
        return hashed(self.root / name, {"schema_version": 1, "query_count": 2, "records": records})

    def selector(self, name, selection, seconds, **attest):
        fields = selector_fields(selection, seconds)                    # complete repeat evidence (review 2)
        path = self.root / "selector" / name
        binding = selector_binding(fields, path)
        hashed(path, {**fields, "binding": binding})
        attest_selector(path, binding, **attest)
        return str(path)

    # ------------------------------------------------------------------ runs, as the runner writes them
    def make(self, stem, arm, selection, audit, *, kind, seed=42, cands=HALF, core=1.0, of=None, per_query=None,
             sims_rows=None, attest=None, **modes):
        """``kind``: timing (attested) | acc (inline dump) | replay (of the timing run ``of``, bitwise)."""
        mode = {"timing": None, "acc": "inline", "replay": "replay"}[kind]
        of_manifest = json.loads((Path(of) / "run_manifest.json").read_text()) if of else None
        want = ident.registered_identity(arm, selection=selection, audit=audit, seed=seed, timing=kind == "timing",
                                         waveform_mode=mode, replay_of_sha256=of_manifest["sha256"] if of else None,
                                         **modes)
        identity = example_identity(want, audit)
        if mode:
            identity["waveform_dump"] = {"mode": mode, "dir": str(self.waves / stem)}
        if of:
            identity["replay_of"] = {"run_manifest_sha256": of_manifest["sha256"], "run_dir": str(of)}
        run_dir = self.runs / stem
        manifest = initialize_run(run_dir, identity)
        store = wav.WaveformDump(self.waves / stem, run_stem=stem, root=self.waves) if mode else None
        for r in selection["records"]:
            if of:
                source = json.loads((Path(of) / "queries" / f"query_{r['index']:05d}.json").read_text())
                with np.load(Path(of) / source["arrays_file"]) as arrays:
                    c, s = np.array(arrays["candidates"]), np.array(arrays["similarities"])
                extra = {k: source[k] for k in ("candidate_indices_sha256", "parent_candidate_indices_sha256",
                                                "candidate_subset") if k in source}
                rows = (np.asarray(source["c2f"]["union_indices"]) if "c2f" in source else grid.nested_subset(HALF)
                        if "candidate_subset" in source else np.arange(len(c)))       # the actual row mapping
            else:
                c = cands(r) if callable(cands) else cands
                s = (sims(HALF, seed)[sims_rows] if sims_rows is not None else sims(c, seed)).reshape(-1, 1)
                extra = {"candidate_indices_sha256": r["candidate_indices_sha256"]}
                rows = np.asarray(sims_rows) if sims_rows is not None else np.arange(len(c))
            result = {"run_manifest_sha256": manifest["sha256"], "query_index": r["index"], "query_id": r["query_id"],
                      "room": r["room"], "candidate_count": len(c), "peak_memory_bytes": 7, "source_global": TRUTH,
                      "metrics": {"1": grid.recomputed_k1_metrics(c, s, TRUTH, 0.1)}, **extra}
            if kind == "timing":
                result.update(latency_protocol={"name": f"fem_core_aligned_kctx{modes.get('context_count', 8)}_kgen1",
                                                "context_count": modes.get("context_count", 8)},
                              latency_seconds={"context_conditioning": 0.1,
                                               "candidate_conditioning_and_generation": core - 0.1,
                                               "core_forward_total": core})
            result.update((per_query or {}).get(r["index"], {}))
            if store:
                entry = store.write(r["index"], np.full((len(c), 1, 10240), 0.001 * r["index"], np.float32), rows)
                result["waveforms"] = {"dump_dir": str(store.dir), **entry}
            save_query_result(run_dir, result=result, candidates=c, similarities=s)
        if store:
            store.finish(run_manifest_sha256=manifest["sha256"], identity=identity, mode=mode,
                         query_indices=[r["index"] for r in selection["records"]], replay_of=identity.get("replay_of"))
        if kind == "timing":
            attest_run(run_dir, **(attest or {}))
        return run_dir

    def c2f(self, stem, arm, *, seed=42, kind="timing", **kw):
        """A faithful c2f run over the arm's same-seed 0.5 m arrays (the registered rule applied to its scores): the
        seed-42 timing pass, or (``kind='acc'``) a seed 43 / 44 accuracy-only run with its inline dump."""
        parent = sims(HALF, seed)
        rule = grid.c2f_indices(HALF, grid.query_scores(parent.reshape(-1, 1), 0.1), k=5)
        union, new = rule["union"], np.setdiff1d(rule["refined"], rule["coarse"])
        sha = hashlib.sha256(union.astype("<u4").tobytes()).hexdigest()
        parts = {"observed_encoding": 0.05, "context_conditioning": 0.1, "coarse_generation": 0.3, "coarse_scoring": 0.1,
                 "selection": 0.01, "refined_generation": 0.25, "refined_scoring": 0.08, "union_argmax": 0.01}
        selector = ("observed_encoding", "coarse_scoring", "selection", "refined_scoring", "union_argmax")
        winner = int(union[int(np.argmax(grid.query_scores(parent[union].reshape(-1, 1), 0.1)))])
        from src.localization.exp37_latency import c2f_protocol
        core = float(parts["context_conditioning"] + parts["coarse_generation"] + parts["refined_generation"])
        timing = {"latency_protocol": c2f_protocol(5),
                  "latency_seconds": {**parts, "c2f_total": float(sum(parts.values())), "core_forward_total": core},
                  "selector_repeat_seconds": {p: [parts[p]] * 3 for p in selector}} if kind == "timing" else {}
        per_query = {i: {"candidate_indices_sha256": sha, "parent_candidate_indices_sha256": f"h{i}", **timing,
                         "c2f": {"full_candidate_count": len(HALF), "coarse_count": len(rule["coarse"]),
                                 "top_indices": rule["top"].tolist(), "refined_count": len(rule["refined"]),
                                 "new_count": len(new), "union_indices": union.tolist(),
                                 "generated_count": len(rule["coarse"]) + len(new), "winner_full_index": winner,
                                 "union_indices_sha256": sha, "top_k": 5, "chebyshev_radius_m": 0.5}} for i in (1, 2)}
        return self.make(stem, arm, self.pilot, self.audits["audit_0p50"], kind=kind, seed=seed, cands=HALF[union],
                         per_query=per_query, sims_rows=union, candidate_subset="c2f", c2f_top_k=5, **kw)

    def build(self, *, seeds_43_44_timing=False):
        a50, a75 = self.audits["audit_0p50"], self.audits["audit_0p75"]
        for arm in ("P1", "BF"):
            for g, pilot, audit, cands, core in (("0p50", self.pilot, a50, HALF, 2.0),
                                                 ("0p75", self.pilot75, a75, SEVENTY_FIVE, 0.6)):
                for seed in (42, 43, 44):
                    stem = f"b2_{g}_{arm}_s{seed}"
                    if seed == 42 or seeds_43_44_timing:
                        timing = self.make(stem, arm, pilot, audit, kind="timing", seed=seed, cands=cands, core=core)
                        self.make(f"{stem}_replay", arm, pilot, audit, kind="replay", seed=seed, of=timing)
                    else:
                        self.make(stem, arm, pilot, audit, kind="acc", seed=seed, cands=cands)
            index = grid.nested_subset(HALF)
            self.make(f"b2t_nested_{arm}_s42", arm, self.pilot, a50, kind="timing", cands=HALF[index], core=0.4,
                      candidate_subset="nested_1m",
                      per_query={i: {"candidate_subset": {"name": "nested_subset", "full_candidate_count": len(HALF),
                                                          "indices_sha256": self.nested_sha},
                                     "candidate_indices_sha256": self.nested_sha,
                                     "parent_candidate_indices_sha256": f"h{i}"} for i in (1, 2)})
            self.c2f(f"b2t_c2f_{arm}_s42", arm)
        for orbit in ("0", "0_180", "c4"):
            for theta in (0, 90):
                if (orbit, theta) == ("c4", 0):
                    continue
                for seed in (42, 43, 44):
                    self.make(f"b3_{ORBIT_STEM[orbit]}_t{theta}_s{seed}", "BF", self.pilot, a50, kind="acc", seed=seed,
                              frame_angles=ORBIT_ANGLES[orbit], rotate_deg=float(theta))
        for orbit, stem in (("0", "b3_lat_o0_s42"), ("0_180", "b3_lat_o0_180_s42"), ("c4", "b1_k8_BF_s42")):
            self.make(stem, "BF", self.pilot, a50, kind="timing", core=3.0, frame_angles=ORBIT_ANGLES[orbit])
        return self

    # ------------------------------------------------------------------ the fixed inputs
    def synthesis_cell(self, orbit, theta, k, leaf_commit, keys):
        """One synthesis cell's three artifacts exactly as the exp_25 runner writes them, and its spec entry."""
        from src.localization import exp37_synthesis as syn
        name, block, source = syn.cell_name(orbit, theta, k)
        stem, pin = syn.cell_stem(orbit, theta, k), leaf_commit if source == "new" else syn.H35_PIN
        paths = syn.cell_paths(orbit, theta, k)
        self._inside(paths.values())                               # the INCIDENT guard: never the real pinned roots
        folder = Path(paths["metrics_json"]).parent
        folder.mkdir(parents=True, exist_ok=True)
        value = 10.0 + k + theta / 90 + len(syn.ORBITS[orbit])
        metrics = {"metrics": {key: value + i for i, key in enumerate(keys)},
                   "ckpt_path": syn.CKPT_LINK, "rotate_deg": float(theta), "cond_method": "fa_invariant",
                   "frame_avg_angles": syn.ORBITS[orbit], "cond_autocast": "bf16", "orbit_execution": "frame_average",
                   "frame_avg_fwd_cap": 64, "source_sha": pin, "batch_size": 64, "n_samples": syn.STREAMS,
                   "dataset_config": syn.DATASET[k][0], "seed": 42, "cfg_scale": 1.0, "steps": 1, "eval_name": name,
                   "weights_source": "ema", "device": "cuda", "ckpt_sha256_loaded": syn.CKPT_SHA256,
                   "ckpt_bytes_loaded": 1, "by_scene": {scene: {key: value + i + j for i, key in enumerate(keys)}
                                                         for j, scene in enumerate(("Cafe", "Office"))}}
        metrics_path = Path(paths["metrics_json"])
        metrics_path.write_text(json.dumps(metrics))
        metrics_sha = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
        wave = hashlib.sha256(stem.encode()).hexdigest()
        manifest = {"schema_version": 1, "n": syn.STREAMS, "dtype": "float32", "shape": [syn.STREAMS, 10240],
                    "sha256": wave, "input_hash": "1" * 64, "metrics_stem": stem, "source_sha": pin,
                    "ckpt_path": syn.CKPT_LINK, "metrics_path": f"outputs_FLAC/exp25_depth_canon/BF/{stem}.json",
                    "metrics_sha256": metrics_sha, "stream_sidecar_path": f"outputs_FLAC/exp25_depth_canon/BF/{stem}.stream.json",
                    "stream_sidecar_sha256": "2" * 64, "canon_sidecar_path": None, "canon_sidecar_sha256": None}
        manifest_bytes = json.dumps(manifest, indent=2).encode()
        manifest_path = Path(paths["dump_manifest"])
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_bytes(manifest_bytes)
        argv = ["--model-config", syn.MODEL_CONFIG[0], "--dataset-config", syn.DATASET[k][0], "--cond-method",
                "fa_invariant", "--frame-avg-angles", ",".join(str(int(a)) for a in syn.ORBITS[orbit]),
                "--frame-avg-max-fwd-samples", "64", "--cond-autocast", "bf16", "--cfg-scale", "1.0", "--steps", "1",
                "--batch-size", "64", "--device", "cuda", "--record-per-scene", "--record-stream",
                "--expected-stream-count", str(syn.STREAMS), "--seed", "42", "--rotate-mode", "fixed", "--rotate-deg",
                str(theta), "--eval-name", name, "--ckpt-path", syn.CKPT_LINK, "--dump-scored-waveforms", syn.DUMP_ROOT]
        values = {"schema_version": 2, "cell": name, "stem": stem, "block": block, "arm": "BF", "step": 40000,
                  "ckpt_link": syn.CKPT_LINK, "ckpt_target": "/media/diskstation/x/BF/epoch=8-step=40000.ckpt",
                  "ckpt_sha256": syn.CKPT_SHA256, "ckpt_cache": f"outputs_FLAC/exp25_depth_canon/.ckpt_cache/{syn.CKPT_SHA256}.ckpt",
                  "ckpt_sha256_measured": syn.CKPT_SHA256, "ckpt_sha256_loaded": syn.CKPT_SHA256, "ckpt_bytes_loaded": 1,
                  "model_config": syn.MODEL_CONFIG[0], "model_config_sha256": syn.MODEL_CONFIG[1],
                  "dataset_config": syn.DATASET[k][0], "dataset_config_sha256": syn.DATASET[k][1],
                  "expected_stream_count": syn.STREAMS, "dump": True, "dump_root": syn.DUMP_ROOT, "argv": argv,
                  "expect_head": pin, **{key: syn.LEAF_TREE_DIGEST if source == "new" else syn.H35_TREE_DIGEST
                                         for key in ("campaign_digest", "spawn_digest", "post_digest")},
                  "artifact_sha256": {"metrics": metrics_sha, "stream": "2" * 64, "canon": None},
                  "dump_evidence": {"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "manifest": manifest,
                                    "marker": f"{wave}\n", "waveforms_sidecar": f"{wave}  waveforms.npy\n"},
                  "lane": 0, "log": f"worklog/{name}.log", "rc": 0, **syn.NO_FLASH}
        receipt_path = Path(paths["receipt"])
        receipt_path.write_text(json.dumps({key: values[key] for key in syn.RECEIPT_KEYS}, indent=2))
        sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return {"orbit": orbit, "theta": theta, "k_context": k, "seed": 42, "source": source, "cell": name,
                "metrics_json": str(metrics_path), "metrics_sha256": metrics_sha, "receipt": str(receipt_path),
                "receipt_sha256": sha(receipt_path), "dump_manifest": str(manifest_path),
                "dump_manifest_sha256": sha(manifest_path)}

    def _inside(self, paths) -> None:
        """Refuse (before any write) a target outside this world's root -- an unpatched exp37_synthesis points at the
        real leaf / exp_35 worktrees and the NAS dump root (the 2026-11-06 incident)."""
        root = self.root.resolve()
        outside = [str(p) for p in paths if not Path(p).resolve().is_relative_to(root)]
        if outside:
            raise RuntimeError(f"the synthesis fixture refuses to write outside its root {root}: {outside[:3]} "
                               "(patch exp37_synthesis with patch_synthesis first)")

    def synthesis_roots(self):
        """Where this world's synthesis artifacts live (the leaf / exp_35 worktrees and the dump root stand-ins)."""
        return {"LEAF_ROOT": str(self.root / "leaf"), "EXP35_ROOT": str(self.root / "exp35"),
                "DUMP_ROOT": str(self.root / "exp25_dumps"), "LEAF_COMMIT": "e" * 40}

    def synthesis(self, *, status="complete", leaf_commit="e" * 40):
        """The synthesis section (call with exp37_synthesis's roots patched to ``synthesis_roots``); the reused
        cells' digests are kept in ``self.reused`` for the pinned table."""
        keys = ["T60", "C50", "EDT", "FD"]                               # = exp37_build_synthesis_cells.METRICS
        from src.localization import exp37_synthesis as syn
        if status == "complete":                                         # every target checked before the first write
            self._inside(p for o in ("0", "0_180", "c4") for t in (0, 90) for k in (1, 8)
                         for p in syn.cell_paths(o, t, k).values())
        cells = [self.synthesis_cell(o, t, k, leaf_commit, keys) for o in ("0", "0_180", "c4") for t in (0, 90)
                 for k in (1, 8)] if status == "complete" else []
        self.reused = {c["cell"]: {key: c[key] for key in ("metrics_sha256", "receipt_sha256", "dump_manifest_sha256")}
                       for c in cells if c["source"] == "reused_exp35"}
        body = {"status": status, "leaf_commit": leaf_commit if status == "complete" else None,
                "runbook": "the exp_36 leaf's B3 runbook (Planner)" if status == "complete" else None,
                "metrics_keys": keys if status == "complete" else [], "cells": cells}
        if status == "pending":
            body["note"] = "pending"
        path = self.root / "synthesis.json"
        path.write_text(json.dumps(body))
        return path

    def external(self):
        per_query = [{"query_index": r["index"]} for r in self.pilot["records"]]
        raw = {"median_localization_error_m": 1.2, "success_rate_at_0_5m": 0.2, "success_rate_at_1_0m": 0.5,
               "resolution_aware_success_rate_at_0_5m": 0.3}
        summary = hashed(self.root / "exp23_summary.json", {"query_count": 2, "per_query": per_query,
                                                            "fem_coverage": {"failed_query_count": 1},
                                                            "metrics": {"few_shot": raw, "fem_omp": raw}})
        csv_path = self.root / "latency_per_query.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["query_index", "few_shot_rir_median_seconds",
                                                        "fem_omp_median_seconds"])
            writer.writeheader()
            for r in self.pilot["records"]:
                writer.writerow({"query_index": r["index"], "few_shot_rir_median_seconds": 0.05,
                                 "fem_omp_median_seconds": 9.0})
        return summary["sha256"], hashlib.sha256(csv_path.read_bytes()).hexdigest()

    @property
    def inputs(self) -> dict:
        return {"pilot_0p50": str(self.root / "pilot_0p50.json"), "pilot_0p75": str(self.root / "pilot_0p75.json"),
                "audit_0p50": str(self.root / "audit_0p50.json"), "audit_0p75": str(self.root / "audit_0p75.json"),
                "nested_manifest": str(self.root / "nested.json"), "selector_0p50": self.selectors["s050"],
                "selector_0p75": self.selectors["s075"], "selector_nested": self.selectors["s100"],
                "latency_selection": str(self.root / "pilot_0p50.json"), "latency_selector": self.selectors["s050"],
                "external_accuracy": str(self.root / "exp23_summary.json"),
                "external_latency_csv": str(self.root / "latency_per_query.csv")}


def add_timing_only_replays(world: World) -> World:
    """The optional artifact replays of the timing-only passes (nested, c2f, orbit latency)."""
    a50 = world.audits["audit_0p50"]
    for arm in ("P1", "BF"):
        world.make(f"b2t_nested_{arm}_s42_replay", arm, world.pilot, a50, kind="replay",
                   of=world.runs / f"b2t_nested_{arm}_s42", candidate_subset="nested_1m")
        world.make(f"b2t_c2f_{arm}_s42_replay", arm, world.pilot, a50, kind="replay", of=world.runs / f"b2t_c2f_{arm}_s42",
                   candidate_subset="c2f", c2f_top_k=5)
    for orbit, stem in (("0", "b3_lat_o0_s42"), ("0_180", "b3_lat_o0_180_s42"), ("c4", "b1_k8_BF_s42")):
        world.make(f"{stem}_replay", "BF", world.pilot, a50, kind="replay", of=world.runs / stem,
                   frame_angles=ORBIT_ANGLES[orbit])
    return world


def patch_synthesis(monkeypatch, world: World) -> None:
    """Point exp37_synthesis's pinned roots / leaf commit at the world's stand-ins (tests only)."""
    from src.localization import exp37_synthesis as syn
    for name, value in world.synthesis_roots().items():
        monkeypatch.setattr(syn, name, value)


def registered_world(root, monkeypatch) -> World:
    """Every registered run: the B2 / B3 runs, the mandatory replays of every timing pass, the seed 43 / 44 c2f
    accuracy runs, and the synthesis + external inputs. ``monkeypatch`` is REQUIRED: the synthesis roots are always
    redirected to the world's stand-ins (and the reused cells' digests pinned) -- the 2026-11-06 incident."""
    from src.localization import exp37_synthesis as syn
    world = World(root)
    patch_synthesis(monkeypatch, world)
    world = add_timing_only_replays(world.build())
    for arm in ("P1", "BF"):
        for seed in (43, 44):
            world.c2f(f"b2a_c2f_{arm}_s{seed}", arm, seed=seed, kind="acc")
    world.synthesis()
    monkeypatch.setattr(syn, "REUSED", world.reused)
    world.external()
    return world
