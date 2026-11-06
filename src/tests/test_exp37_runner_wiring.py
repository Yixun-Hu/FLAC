"""exp_37 round B-a items 3-4: the runner / CLI wiring -- ``--query-index`` (the one-query G0 mode), the attention
backend and the verified no-flash launch provenance in the run identity, the ``--mesh-root`` map in the identity
and on every room reconstruction, and the per-process host records. CPU only: the model, AGREE, the room
reconstruction and ``execute_query`` are stubbed (the timed path itself is pinned by test_exp37_headline_timer_guard).

Written by the exp_37 Coder (Claude Opus 5.5, max effort) per the Coder brief r1, round B-a.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import src.models.transformer as transformer
from src.localization import run_provenance, runner
from src.localization.ar_queries import load_context_manifest
from src.localization.pilot import load_pilot_manifest, resolve_pilot_records

REPO = Path(__file__).resolve().parents[2]
E9 = REPO / "worklog/worklog_yixun/exp_09_localization_grid_preflight_claude"
SELECTION_128 = REPO / "NeurIPS_2026_Workshop_Latency_Reproducibility/data/selection/frozen_16room_128.json"
ORIGINAL_IDENTITY_KEYS = {
    "model_config_sha256", "checkpoint_sha256", "agree_checkpoint_sha256", "context_manifest_sha256",
    "geometry_audit_sha256", "pilot_manifest_sha256", "query_indices", "conditioning_method", "frame_average_angles",
    "n_context", "score_sample_counts", "tau", "sample_seed", "candidate_batch_size", "synchronize_timing",
    "warmup_query_count", "measure_core_forward", "sampler"}


@pytest.fixture(scope="module")
def frozen():
    return (load_context_manifest(E9 / "context_manifest_exp01_seed42.json"),
            json.loads((E9 / "geometry_audit.json").read_text()), load_pilot_manifest(SELECTION_128))


def test_select_queries_by_index_limit_or_all(frozen):
    context, audit, pilot = frozen
    joined = resolve_pilot_records(pilot, context, audit)
    one = runner.select_queries(joined, query_index=4467)
    assert len(one) == 1 and one[0][0]["index"] == 4467 and one[0][0]["candidate_count"] == 27
    assert one[0][0]["query_id"] == "single_channel_ir_1/Bathrooms/Bathrooms_idx_14/S009_R017_hybrid_IR.wav"
    assert runner.select_queries(joined, query_limit=3) == joined[:3]
    assert runner.select_queries(joined) == joined
    with pytest.raises(ValueError, match="not in the pilot"):
        runner.select_queries(joined, query_index=999999)
    with pytest.raises(ValueError, match="mutually exclusive"):
        runner.select_queries(joined, query_limit=1, query_index=4467)
    with pytest.raises(ValueError, match="outside the pilot range"):
        runner.select_queries(joined, query_limit=0)


class _Stubs:
    """Records what the runner hands to the stubbed heavy steps."""

    def __init__(self):
        self.reconstruct, self.host, self.executed = [], [], []


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    s = _Stubs()
    monkeypatch.setattr(torch, "set_float32_matmul_precision", lambda precision: None)   # no global side effect
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", torch.backends.cuda.matmul.allow_tf32)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", torch.backends.cudnn.allow_tf32)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device=None: None)
    monkeypatch.setattr(runner, "load_flac_module", lambda *a: (object(), {}))
    monkeypatch.setattr(runner, "load_agree_retrieval", lambda *a: object())

    def reconstruct(room, audit, mesh_path=None):
        s.reconstruct.append((room, mesh_path))
        return np.zeros((0, 3))

    def execute(**kw):
        s.executed.append(kw["selected"]["index"])
        n = int(kw["selected"]["candidate_count"])
        result = {"schema_version": 1, "run_manifest_sha256": kw["run_sha256"], "query_index": kw["selected"]["index"],
                  "query_id": kw["selected"]["query_id"], "candidate_count": n, "elapsed_seconds": 0.0}
        return result, np.zeros((n, 3), dtype=np.float32), np.zeros((n, 1), dtype=np.float32)

    def host(output_dir, label, **kw):
        s.host.append((label, kw["handoff"], kw["run_manifest_sha256"]))

    monkeypatch.setattr(runner, "reconstruct_room_base_candidates", reconstruct)
    monkeypatch.setattr(runner, "execute_query", execute)
    monkeypatch.setattr(runner, "write_host_record", host)
    return s


def _ckpt(tmp_path):
    path = tmp_path / "fake.ckpt"
    path.write_bytes(b"weights")
    return path


def _run(frozen, tmp_path, **extra):
    context, audit, pilot = frozen
    return runner.run_localization(
        model_config_path=REPO / "src/configs/model_configs/FLAC/AR/FLAC_AR.json", checkpoint_path=_ckpt(tmp_path),
        agree_checkpoint_path=_ckpt(tmp_path), context_manifest=context, geometry_audit=audit, pilot_manifest=pilot,
        dataset_root=tmp_path, output_dir=tmp_path / "run", device="cuda:0", cond_method="vanilla",
        score_sample_counts=(1,), **extra)


def _identity(tmp_path):
    return json.loads((tmp_path / "run" / "run_manifest.json").read_text())["identity"]


def test_the_default_run_adds_only_the_attention_backend(frozen, tmp_path, stubbed, monkeypatch):
    monkeypatch.setattr(transformer, "flash_attn_func", object())
    out = _run(frozen, tmp_path, query_limit=2)
    identity = _identity(tmp_path)
    assert set(identity) == ORIGINAL_IDENTITY_KEYS | {"attention_backend"}
    assert identity["attention_backend"] == "flash_attn" and out["completed_queries"] == 2
    assert all(mesh_path is None for _room, mesh_path in stubbed.reconstruct)
    assert [label for label, *_ in stubbed.host] == ["start", "end"]
    assert all(handoff is None for _label, handoff, _sha in stubbed.host)


def test_the_exp37_options_reach_the_identity_the_rooms_and_the_host_records(frozen, tmp_path, stubbed, monkeypatch):
    context, audit, pilot = frozen
    monkeypatch.setattr(transformer, "flash_attn_func", None)
    launcher = REPO / "overlays/no_flash/exp37_no_flash_launch.py"
    handoff = {"attention_backend": "sdpa", "flash_attn_func_is_none": True, "pid": os.getpid(),
               "overlay_sha256": run_provenance.OVERLAY_SHA256, "sentinel": run_provenance.SENTINEL,
               "launcher_sha256": hashlib.sha256(launcher.read_bytes()).hexdigest(), "target": "localize_FLAC.py"}
    monkeypatch.setenv(run_provenance.LAUNCH_ENV, json.dumps(handoff))
    remap = {"argument": "m", "resolved": "/m", "audit_mesh_root": audit["mesh_root"],
             "rooms": {room: {"mesh_path": f"/m/{room}.obj", "mesh_sha256": row["mesh_sha256"]}
                       for room, row in audit["rooms"].items()}}
    monkeypatch.setattr(runner, "resolve_mesh_root", lambda a, root: remap if (a is audit and root == Path("m")) else None)
    _run(frozen, tmp_path, query_index=4467, mesh_root=Path("m"), require_attention_backend="sdpa",
         warmup_query_count=1)
    identity = _identity(tmp_path)
    assert identity["query_indices"] == [4467] and identity["attention_backend"] == "sdpa"
    assert identity["no_flash_launch"] == {"overlay_sha256": handoff["overlay_sha256"],
                                           "launcher_sha256": handoff["launcher_sha256"], "sentinel": handoff["sentinel"]}
    assert identity["mesh_root_override"] == remap
    assert stubbed.reconstruct == [("Bathrooms_idx_14", "/m/Bathrooms_idx_14.obj")]   # warm-up + measured: one build
    assert stubbed.executed == [4467, 4467]                                               # the warm-up, then the query
    assert [(label, h) for label, h, _ in stubbed.host] == [("start", handoff), ("end", handoff)]


def test_a_required_backend_refuses_before_anything_is_written(frozen, tmp_path, stubbed, monkeypatch):
    monkeypatch.setattr(transformer, "flash_attn_func", None)
    monkeypatch.delenv(run_provenance.LAUNCH_ENV, raising=False)
    with pytest.raises(RuntimeError, match="hand-off"):
        _run(frozen, tmp_path, query_index=4467, require_attention_backend="sdpa")
    assert not (tmp_path / "run").exists() and stubbed.executed == []


def test_a_mesh_mismatch_refuses_before_anything_is_written(frozen, tmp_path, stubbed, monkeypatch):
    def refuse(audit, root):
        raise RuntimeError("room X: SHA-256 differs")
    monkeypatch.setattr(runner, "resolve_mesh_root", refuse)
    with pytest.raises(RuntimeError, match="SHA-256"):
        _run(frozen, tmp_path, query_index=4467, mesh_root=Path("m"))
    assert not (tmp_path / "run").exists() and stubbed.executed == []


def test_the_cli_passes_the_exp37_flags_through(monkeypatch, tmp_path):
    sys.path.insert(0, str(REPO))
    import localize_FLAC

    seen = {}
    monkeypatch.setattr(localize_FLAC, "run_localization", lambda **kw: seen.update(kw) or {"ok": True})
    monkeypatch.setattr(sys, "argv", [
        "localize_FLAC.py", "--model-config", "c.json", "--ckpt-path", "k.ckpt", "--agree-ckpt", "a.pt",
        "--context-manifest", str(E9 / "context_manifest_exp01_seed42.json"),
        "--geometry-audit", str(E9 / "geometry_audit.json"), "--pilot-manifest", str(SELECTION_128),
        "--dataset-root", "AcousticRooms", "--output-dir", str(REPO / "never_created_by_this_test"),
        "--cond-method", "vanilla", "--query-index", "4467", "--mesh-root", "AcousticRooms/room_mesh_obj_format",
        "--require-attention-backend", "sdpa"])
    localize_FLAC.main()
    assert seen["query_index"] == 4467 and seen["mesh_root"] == Path("AcousticRooms/room_mesh_obj_format")
    assert seen["require_attention_backend"] == "sdpa" and seen["query_limit"] is None
    assert not (REPO / "never_created_by_this_test").exists()
