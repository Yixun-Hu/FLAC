"""exp_37 round B-a item 4: the G0 comparator (plan_loc_efficiency.md §10 "G0 pinned"; exit 0 PASS / 1 FAIL /
2 REFUSED). Synthetic artifacts are written with the runner's own ``initialize_run`` / ``save_query_result``; the
real-bundle cases need no GPU. Written by the exp_37 Coder (Claude Opus 5.5, max effort), round B-a.
"""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from src.localization.pilot import canonical_sha256
from src.localization.runner import initialize_run, save_query_result
from tools import exp37_g0_compare as g0

REPO = Path(__file__).resolve().parents[2]
CAND = np.array([[x * 0.5, 1.0, 1.0] for x in range(27)], dtype=np.float32)
SIMS = np.linspace(0.1, 0.5, 27, dtype=np.float32).reshape(27, 1)
LAUNCHER = "1" * 64
AUDIT = {"mesh_root": "/other/host", "rooms": {"Bathrooms_idx_14": {"mesh_sha256": "b" * 64}}}
AUDIT = {**AUDIT, "sha256": canonical_sha256(AUDIT)}
SHARED = {"model_config_sha256": "m" * 64, "agree_checkpoint_sha256": "a" * 64, "context_manifest_sha256": "c" * 64,
          "geometry_audit_sha256": AUDIT["sha256"], "pilot_manifest_sha256": "p" * 64, "n_context": 8,
          "score_sample_counts": [1], "tau": 0.1, "sample_seed": 42, "candidate_batch_size": 64,
          "synchronize_timing": True, "warmup_query_count": 1, "measure_core_forward": True,
          "sampler": {"type": "rectified_flow_discrete_euler", "steps": 1, "cfg_scale": 1.0}}


def _write(run_dir, identity, candidates=CAND, sims=SIMS, **query_over):
    manifest = initialize_run(run_dir, identity)
    winner = int(np.argmax(sims[:, 0]))
    result = {"schema_version": 1, "run_manifest_sha256": manifest["sha256"], "query_index": 4467,
              "query_id": g0.PINS.query_id, "candidate_count": 27, "candidate_indices_sha256": g0.PINS.candidate_hash,
              "n_context": 8, "score_sample_counts": [1], "metrics": {"1": {"prediction_index": winner}},
              "latency_protocol": {"name": "fem_core_aligned_kctx8_kgen1"}, **query_over}
    save_query_result(run_dir, result=result, candidates=candidates, similarities=sims)
    return manifest["sha256"]


@pytest.fixture
def world(tmp_path):
    """A synthetic bundle (all three arms) with its pins, and a factory for our G0 runs."""
    pins_arms, root = {}, tmp_path / "bundle"
    for arm, spec in g0.PINS.arms.items():
        identity = {**SHARED, "checkpoint_sha256": f"ref-{arm}", "conditioning_method": spec["cond"],
                    "frame_average_angles": spec["angles"], "query_indices": [5182, 4467]}
        sha = _write(root / spec["reference_dir"], identity)
        pins_arms[arm] = {**spec, "reference_ckpt": f"ref-{arm}", "local_ckpt": f"ours-{arm}",
                          "reference_manifest_sha256": sha}
    pins = dataclasses.replace(g0.PINS, arms=pins_arms)

    def ours(arm, name=None, identity_over=None, **kw):
        spec = pins.arms[arm]
        identity = {**SHARED, "checkpoint_sha256": spec["local_ckpt"], "conditioning_method": spec["cond"],
                    "frame_average_angles": spec["angles"], "query_indices": [4467], "attention_backend": "sdpa",
                    "no_flash_launch": {"overlay_sha256": pins.overlay_sha256, "launcher_sha256": LAUNCHER,
                                        "sentinel": "EXP37_NO_FLASH_LAUNCHER"},
                    "mesh_root_override": {"audit_mesh_root": "/other/host", "rooms": {
                        "Bathrooms_idx_14": {"mesh_path": "/x.obj", "mesh_sha256": "b" * 64}}}}
        identity.update(identity_over or {})
        run = tmp_path / (name or f"g0_{arm}")
        _write(run, {k: v for k, v in identity.items() if v is not None or k == "frame_average_angles"}, **kw)
        return run
    return pins, root, ours, tmp_path


def _compare(world, arm, run, yaw=None):
    pins, root, _ours, _tmp = world
    return g0.compare(arm, run, root, audit=AUDIT, launcher_sha256=LAUNCHER, yaw_verdict=yaw, pins=pins)


def _yaw_pass(world):
    verdict = _compare(world, "YAW", world[2]("YAW"))
    assert verdict["verdict"] == "PASS"
    return verdict


def test_an_identical_yaw_run_passes_with_exact_equality(world):
    verdict = _yaw_pass(world)
    assert verdict["exit_code"] == 0 and verdict["exact_equality"] is True and verdict["max_abs_delta"] == 0.0
    assert verdict["argmax"] == {"ours": 26, "reference": 26} and "environment" in verdict["diagnosis"]
    assert verdict["sha256"] == canonical_sha256({k: v for k, v in verdict.items() if k != "sha256"})


def test_a_delta_within_the_fixed_tolerance_passes_but_is_not_exact(world):
    verdict = _compare(world, "YAW", world[2]("YAW", sims=SIMS + np.float32(9e-4)))
    assert verdict["verdict"] == "PASS" and verdict["exact_equality"] is False
    assert verdict["max_abs_delta"] == pytest.approx(9e-4, rel=1e-3)


@pytest.mark.parametrize("change, needle", [
    (lambda s, c: (s + np.float32(2e-3), c), "max |delta similarity|"),
    (lambda s, c: (np.flip(s, 0).copy(), c), "argmax"),
    (lambda s, c: (s, c + np.float32(0.5)), "candidate arrays"),
], ids=["delta", "argmax", "candidates"])
def test_a_yaw_failure_is_an_environment_failure(world, change, needle):
    sims, cand = change(SIMS, CAND)
    verdict = _compare(world, "YAW", world[2]("YAW", sims=sims, candidates=cand))
    assert verdict["verdict"] == "FAIL" and verdict["exit_code"] == 1 and needle in " ".join(verdict["reasons"])
    assert "environment / backend parity FAILED" in verdict["diagnosis"]


def test_p1_and_bf_need_a_yaw_pass_and_then_diagnose_ema_parity(world):
    yaw = _yaw_pass(world)
    for arm in ("P1", "BF"):
        assert _compare(world, arm, world[2](arm))["exit_code"] == 2                  # no YAW verdict: refused
        assert _compare(world, arm, world[2](arm, name=f"{arm}b"), yaw)["verdict"] == "PASS"
        fail = _compare(world, arm, world[2](arm, name=f"{arm}c", sims=SIMS + np.float32(5e-3)), yaw)
        assert fail["verdict"] == "FAIL" and "EMA parity suspect" in fail["diagnosis"]
    bad_yaw = _compare(world, "YAW", world[2]("YAW", name="yawbad", sims=SIMS + np.float32(5e-3)))
    refused = _compare(world, "P1", world[2]("P1", name="P1d"), bad_yaw)
    assert refused["exit_code"] == 2 and "YAW" in " ".join(refused["reasons"])


@pytest.mark.parametrize("over, needle", [
    ({"attention_backend": "flash_attn"}, "attention_backend"),
    ({"no_flash_launch": None}, "no_flash_launch"),
    ({"checkpoint_sha256": "ref-YAW-other"}, "checkpoint"),
    ({"conditioning_method": "fa_invariant"}, "conditioning_method"),
    ({"query_indices": [4467, 4469]}, "query_indices"),
    ({"sample_seed": 43}, "sample_seed"),
    ({"candidate_batch_size": 32}, "candidate_batch_size"),
    ({"score_sample_counts": [1, 4, 8]}, "score_sample_counts"),
    ({"mesh_root_override": None}, "mesh_root_override"),
], ids=lambda v: v if isinstance(v, str) else next(iter(v)))
def test_a_run_that_is_not_the_pinned_g0_invocation_is_refused(world, over, needle):
    verdict = _compare(world, "YAW", world[2]("YAW", identity_over=over))
    assert verdict["exit_code"] == 2 and verdict["verdict"] == "REFUSED"
    assert needle in " ".join(verdict["reasons"])


@pytest.mark.parametrize("query_over, needle", [
    ({"query_index": 4469}, "query_index"),
    ({"candidate_indices_sha256": "0" * 64}, "candidate_indices_sha256"),
    ({"metrics": {"1": {"prediction_index": 3}}}, "prediction_index"),
])
def test_a_wrong_or_inconsistent_query_artifact_is_refused(world, query_over, needle):
    verdict = _compare(world, "YAW", world[2]("YAW", **query_over))
    assert verdict["exit_code"] == 2 and needle in " ".join(verdict["reasons"])


def test_tampered_artifacts_are_refused(world):
    pins, root, ours, _tmp = world
    run = ours("YAW")
    (run / "queries" / "query_04467.npz").write_bytes(b"corrupt")
    assert "arrays_sha256" in " ".join(_compare(world, "YAW", run)["reasons"])
    ref = root / pins.arms["YAW"]["reference_dir"] / "run_manifest.json"
    ref.write_text(ref.read_text().replace('"tau": 0.1', '"tau": 0.2'))
    assert _compare(world, "YAW", ours("YAW", name="fresh"))["exit_code"] == 2


def test_the_pins_are_the_registered_g0_query():
    assert (g0.PINS.query_index, g0.PINS.candidate_count, g0.PINS.tolerance) == (4467, 27, 1e-3)
    assert g0.PINS.query_id == "single_channel_ir_1/Bathrooms/Bathrooms_idx_14/S009_R017_hybrid_IR.wav"
    assert g0.PINS.candidate_hash == "4f212dcbf53bea4e604969f50c2bc8f3f6330e9d383375a46b0a4563efd86bf1"
    assert [g0.PINS.arms[a]["cond"] for a in ("YAW", "P1", "BF")] == ["vanilla", "vanilla", "fa_invariant"]


@pytest.mark.parametrize("arm", ["YAW", "P1", "BF"])
def test_the_real_bundle_references_validate_and_self_compare_exactly(arm, tmp_path):
    """Our-run stand-in = the bundle's own arrays under our identity: an exact PASS (no GPU involved)."""
    audit = json.loads((REPO / g0.AUDIT_REL).read_text())
    ref = g0.load_reference(arm, g0.BUNDLE_REPEAT, g0.PINS)
    identity = {**{k: ref["identity"][k] for k in g0.SHARED_FIELDS}, "query_indices": [4467],
                "checkpoint_sha256": g0.PINS.arms[arm]["local_ckpt"], "conditioning_method": g0.PINS.arms[arm]["cond"],
                "frame_average_angles": g0.PINS.arms[arm]["angles"], "attention_backend": "sdpa",
                "no_flash_launch": {"overlay_sha256": g0.PINS.overlay_sha256, "launcher_sha256": LAUNCHER,
                                    "sentinel": "EXP37_NO_FLASH_LAUNCHER"},
                "mesh_root_override": {"audit_mesh_root": audit["mesh_root"], "rooms": {
                    r: {"mesh_path": "/x", "mesh_sha256": row["mesh_sha256"]} for r, row in audit["rooms"].items()}}}
    _write(tmp_path / "run", identity, candidates=ref["candidates"], sims=ref["similarities"],
           metrics={"1": {"prediction_index": int(np.argmax(ref["similarities"][:, 0]))}})
    yaw = None
    if arm != "YAW":
        yaw = {**_fake_yaw_pass(), "launcher_sha256": LAUNCHER}
        yaw["sha256"] = canonical_sha256({k: v for k, v in yaw.items() if k != "sha256"})
    verdict = g0.compare(arm, tmp_path / "run", g0.BUNDLE_REPEAT, audit=audit, launcher_sha256=LAUNCHER,
                         yaw_verdict=yaw)
    assert verdict["verdict"] == "PASS" and verdict["exact_equality"] is True, verdict.get("reasons")


def _fake_yaw_pass():
    return {"gate": "exp37_G0", "arm": "YAW", "verdict": "PASS", "attention_backend": "sdpa",
            "overlay_sha256": g0.PINS.overlay_sha256, "reference_root": str(g0.BUNDLE_REPEAT)}
