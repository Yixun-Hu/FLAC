"""exp-09 Stage A -- convention-audit tests (TDD, CPU).

Every gate is exercised RED-first (a fabricated violation makes the named test fail) and then
green. The unit layer runs entirely on tiny synthetic fixtures (a fabricated "sample" driving the
same code paths, with W divisible by 16 and a tiny random-init cylindrical model -- the
equivariance is structural, not weight-dependent). One integration test uses the real sample and
is skipped if the AcousticRooms data is unreachable.

Mutation map (see the mutation sweep in the Coder report):
  m1 A2a comparator disabled          -> test_a2a_detects_wrong_convention        RED
  m2 undo-roll direction flipped      -> test_a2b_patch_equivariance_pass         RED
  m3 negative-control threshold flip  -> test_a2c_controls_pass                   RED
  m4 status maps control-fail wrong   -> test_classify_control_failure_is_invalid RED
  m5 atomic write bypassed            -> test_atomic_write_uses_tmp_rename         RED
"""
import json
import math
import os
import warnings

import pytest
import torch

import audit_convention as ac

warnings.filterwarnings("ignore")

H, W = 64, 128            # tiny panorama, W divisible by patch 16 (token grid 4 x 8).
J_UNIT = 1                # dj = 16 px = 1 token col; strictly < W_t/2 so roll direction matters.
DJ = ac.ROLL_MULTIPLE * J_UNIT
ALPHA = DJ * 2.0 * math.pi / W
TOKEN_ROLL = DJ // ac.PATCH_SIZE


# ---------------------------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------------------------
def _tiny_cyl_model(gauge):
    from cylindrical_dinov3 import CylindricalDINOv3ViTModel
    from cylindrical_dinov3.configuration_cylindrical_dinov3 import CylindricalDINOv3ViTConfig

    cfg = CylindricalDINOv3ViTConfig(
        hidden_size=32, num_attention_heads=2, num_hidden_layers=2, patch_size=ac.PATCH_SIZE,
        num_register_tokens=2, image_size=W, intermediate_size=64, gauge=gauge,
    )
    torch.manual_seed(1)
    return CylindricalDINOv3ViTModel(cfg).eval()


@pytest.fixture(scope="module")
def model_on():
    return _tiny_cyl_model("cylindrical_xyz")


@pytest.fixture(scope="module")
def model_off():
    return _tiny_cyl_model("none")


@pytest.fixture
def synthetic_pair():
    """A base field and its IDEAL physical-yaw transform T_k(base) = Rz(alpha).Roll_{dj}(base)."""
    torch.manual_seed(7)
    c_base = torch.randn(1, 3, H, W)
    c_rot = ac.physical_yaw_expected(c_base, DJ, ALPHA)
    return c_base, c_rot


@pytest.fixture
def synthetic_md():
    """A fabricated non-degenerate sample md for the real conditioner path."""
    torch.manual_seed(11)
    depth = torch.randn(3, H, W).double() + 3.0          # per-channel std > 0, non-constant
    source_vit = torch.tensor([[1.3, -0.7, 0.4]], dtype=torch.float32)  # radius > 0
    return {"source_vit": source_vit, "depth": depth, "scene": "synthetic"}


# =============================================================================================
# A2a -- pre-model residual (real GeometryConditioner capture + real rotate_scene_metadata).
# =============================================================================================
def test_capture_shape_and_finiteness(synthetic_md):
    mc = ac.build_geometry_conditioner(H, W, ac.GEOM_MAX_VALUE)
    field = ac.capture_encoder_input(mc, synthetic_md)
    assert tuple(field.shape) == (1, 3, H, W)
    assert torch.isfinite(field).all()


def test_a2a_residual_pass(synthetic_md):
    """The REAL rotated pipeline field equals Rz(alpha).Roll_{16j}(base) to <= 1e-6."""
    from src.data.yaw_rotation import rotate_scene_metadata

    mc = ac.build_geometry_conditioner(H, W, ac.GEOM_MAX_VALUE)
    c_base = ac.capture_encoder_input(mc, synthetic_md)
    md_rot = rotate_scene_metadata(synthetic_md, ALPHA, W)
    c_rot = ac.capture_encoder_input(mc, md_rot)
    residual = ac.a2a_residual(c_base, c_rot, DJ, ALPHA)
    assert residual <= ac.A2A_RTOL, f"A2a residual {residual:.3e} exceeds {ac.A2A_RTOL}"


def test_a2a_detects_wrong_convention(synthetic_md):
    """RED-first sensitivity (mutation m1): a WRONG-direction roll must NOT pass the A2a gate.
    If the comparator is disabled (compares c_rot to itself), this residual collapses to 0 and
    the assertion fails."""
    from src.data.yaw_rotation import rotate_scene_metadata

    mc = ac.build_geometry_conditioner(H, W, ac.GEOM_MAX_VALUE)
    c_base = ac.capture_encoder_input(mc, synthetic_md)
    md_rot = rotate_scene_metadata(synthetic_md, ALPHA, W)
    c_rot = ac.capture_encoder_input(mc, md_rot)
    # Feed a deliberately wrong "base" (rolled the opposite way) into the comparator.
    wrong_base = torch.roll(c_base, shifts=-2 * DJ, dims=-1)
    residual = ac.a2a_residual(wrong_base, c_rot, DJ, ALPHA)
    assert residual > ac.A2A_RTOL, "A2a comparator failed to detect a wrong-convention field"


# =============================================================================================
# A2b -- gauge-ON model equivariance.
# =============================================================================================
def test_a2b_pooled_invariance_pass(model_on, synthetic_pair):
    c_base, c_rot = synthetic_pair
    pooled, _ = ac.a2b_metrics(model_on, c_base, c_rot, TOKEN_ROLL, H, W)
    assert pooled <= ac.A2B_RTOL, f"pooled invariance {pooled:.3e} exceeds {ac.A2B_RTOL}"


def test_a2b_patch_equivariance_pass(model_on, synthetic_pair):
    """Patch roll-equivariance after undoing the +j token-column roll (mutation m2: flipping
    the undo direction makes this residual O(1) and the assertion fails)."""
    c_base, c_rot = synthetic_pair
    _, patch = ac.a2b_metrics(model_on, c_base, c_rot, TOKEN_ROLL, H, W)
    assert patch <= ac.A2B_RTOL, f"patch equivariance {patch:.3e} exceeds {ac.A2B_RTOL}"


def test_a2b_wrong_roll_direction_is_large(model_on, synthetic_pair):
    """Direct check that the token-roll direction is load-bearing (independent of m2)."""
    c_base, c_rot = synthetic_pair
    with torch.no_grad():
        out_base, out_rot = model_on(c_base), model_on(c_rot)
    base_grid = ac.patch_grid_from_tokens(out_base.last_hidden_state, H, W)
    rot_grid = ac.patch_grid_from_tokens(out_rot.last_hidden_state, H, W)
    wrong = ac.roll_token_columns(rot_grid, +TOKEN_ROLL)  # wrong sign
    assert ac.rel_err(wrong, base_grid) >= ac.CONTROL_MIN


# =============================================================================================
# A2c -- negative controls.
# =============================================================================================
def test_a2c_controls_pass(model_on, model_off, synthetic_pair):
    """Gauge-OFF and channel-permuted controls must each be >= 1e-2 (mutation m3: inverting the
    threshold comparison makes controls_pass reject these genuinely-large residuals)."""
    c_base, c_rot = synthetic_pair
    controls = ac.a2c_controls(model_on, model_off, c_base, c_rot, TOKEN_ROLL, H, W)
    assert all(v >= ac.CONTROL_MIN for v in controls.values()), controls
    assert ac.controls_pass(controls) is True


def test_a2c_controls_reject_small(model_on, synthetic_pair):
    """A control that is (wrongly) tiny must make controls_pass False."""
    fake = {"gauge_off_pooled": 0.5, "gauge_off_patch": 0.5,
            "channel_perm_pooled": 1e-9, "channel_perm_patch": 0.5}
    assert ac.controls_pass(fake) is False


# =============================================================================================
# A2d -- three-way status classification + exit codes.
# =============================================================================================
def test_classify_valid_pass():
    assert ac.classify_audit_status(True, True, True, True) == ac.STATUS_VALID_PASS


def test_classify_valid_convention_failure():
    # A2a + controls + all other validity pass, but A2b fails -> gauge-OFF.
    assert ac.classify_audit_status(True, True, True, False) == ac.STATUS_CONVENTION_FAILURE


def test_classify_control_failure_is_invalid():
    """Mutation m4: a negative-control failure must yield invalid_infrastructure, NEVER
    valid_convention_failure -- even when A2b also fails."""
    assert ac.classify_audit_status(True, False, True, False) == ac.STATUS_INVALID
    assert ac.classify_audit_status(True, False, True, True) == ac.STATUS_INVALID


def test_classify_a2a_failure_is_invalid():
    assert ac.classify_audit_status(False, True, True, True) == ac.STATUS_INVALID


def test_classify_other_validity_failure_is_invalid():
    assert ac.classify_audit_status(True, True, False, True) == ac.STATUS_INVALID


def test_exit_code_mapping():
    assert ac.exit_code_for(ac.STATUS_VALID_PASS) == 0
    assert ac.exit_code_for(ac.STATUS_CONVENTION_FAILURE) == 2
    assert ac.exit_code_for(ac.STATUS_INVALID) == 3


# =============================================================================================
# Finiteness + atomic serialisation.
# =============================================================================================
def test_collect_and_sanitize_non_finite():
    rec = {"a": 1.0, "b": [float("nan"), 2.0], "c": {"d": float("inf")}}
    bad = ac.collect_non_finite(rec)
    assert set(bad) == {"b[0]", "c.d"}
    safe = ac.sanitize_non_finite(rec)
    assert safe["b"][0] is None and safe["c"]["d"] is None and safe["a"] == 1.0
    assert ac.collect_non_finite(safe) == []


def test_atomic_write_uses_tmp_rename(tmp_path, monkeypatch):
    """Mutation m5: the write MUST go through a temp file + os.replace. Bypassing the rename
    (direct write) means os.replace is never called and this test fails."""
    calls = []
    real_replace = os.replace

    def spy_replace(src, dst):
        calls.append((src, dst))
        return real_replace(src, dst)

    monkeypatch.setattr(ac.os, "replace", spy_replace)
    out = tmp_path / "audit_convention.json"
    ac.atomic_write_json(str(out), {"audit_status": "valid_pass", "x": 1.5})
    assert len(calls) == 1, "atomic_write_json did not use os.replace (tmp+rename bypassed)"
    assert calls[0][1] == str(out)
    assert json.loads(out.read_text())["audit_status"] == "valid_pass"
    # No temp leftovers.
    assert [p for p in os.listdir(tmp_path) if ".tmp" in p] == []


def test_atomic_write_no_partial_on_nonfinite(tmp_path):
    """A non-finite value (allow_nan=False) raises, and the destination is left untouched:
    no partial write, no temp leftover."""
    out = tmp_path / "audit_convention.json"
    out.write_text('{"old": true}')
    with pytest.raises(ValueError):
        ac.atomic_write_json(str(out), {"bad": float("inf")})
    assert json.loads(out.read_text()) == {"old": True}  # unchanged
    assert [p for p in os.listdir(tmp_path) if ".tmp" in p] == []


# =============================================================================================
# Sample acquisition + fingerprint / non-degeneracy.
# =============================================================================================
def test_fingerprint_and_nondegeneracy(synthetic_md):
    ident = {"scene": "synthetic", "sub": "x", "wav": "S001_R0001.wav", "data_root": "/x"}
    fp = ac.sample_fingerprint(synthetic_md, ident)
    assert fp["nondegenerate"] is True
    assert len(fp["sha256"]) == 64
    assert fp["per_channel_std"] and all(s > 0 for s in fp["per_channel_std"])


def test_nondegeneracy_rejects_constant_depth():
    md = {"source_vit": torch.tensor([[1.0, 1.0, 0.0]]), "depth": torch.ones(3, H, W).double()}
    fp = ac.sample_fingerprint(md, {"scene": "c"})
    assert fp["nondegenerate"] is False  # constant depth -> per-channel std 0


def test_nondegeneracy_rejects_origin_source():
    torch.manual_seed(3)
    md = {"source_vit": torch.zeros(1, 3), "depth": torch.randn(3, H, W).double()}
    fp = ac.sample_fingerprint(md, {"scene": "c"})
    assert fp["nondegenerate"] is False  # source radius 0 -> yaw does not move it


def test_resolve_data_root_unreachable_raises():
    with pytest.raises(ac.InvalidInfrastructure):
        ac.resolve_data_root(["/nonexistent/data/root"], "Cafe", "Cafe_idx_0", "S001_R0044_hybrid_IR.wav")


def test_angle_spec_matches_plan():
    spec = ac.angle_spec(512)
    assert [s["j"] for s in spec] == [1, 4, 8, 16, 24]
    for s in spec:
        assert s["dj_pixels"] == 16 * s["j"]
        assert s["token_roll"] == s["j"]
        assert math.isclose(s["alpha_rad"], 2 * math.pi * (16 * s["j"]) / 512)
    # j=4 -> 45 deg, j=8 -> 90 deg (plan).
    assert math.isclose(spec[1]["degrees"], 45.0)
    assert math.isclose(spec[2]["degrees"], 90.0)


# =============================================================================================
# A1 static block -- convention facts + quoted source lines.
# =============================================================================================
def test_a1_static_quotes_live_source():
    a1 = ac.build_a1_static(str(ac._WORKTREE_ROOT))
    assert a1["convention"]["channels_xyz"] == {"x": 0, "y": 1, "z": 2}
    ar_quote = a1["source_quotes"]["AR_md.convert_equirect_to_camera_coord"]["text"]
    assert "theta_map" in ar_quote and "2.0 * np.pi / img_w" in ar_quote
    yaw_quote = a1["source_quotes"]["yaw_rotation.rotate_scene_metadata.roll_and_rotate"]["text"]
    assert "torch.roll(depth" in yaw_quote
    gauge_q = a1["source_quotes"]["gauge.channel_contract"]
    assert gauge_q is not None and "X_CHANNEL" in gauge_q["text"]


# =============================================================================================
# Integration -- ONE real sample; skipped if the AcousticRooms data is unreachable.
# =============================================================================================
def _real_data_root():
    try:
        return ac.resolve_data_root(
            ac.DEFAULT_DATA_ROOTS, ac.DEFAULT_SAMPLE["scene"], ac.DEFAULT_SAMPLE["sub"],
            ac.DEFAULT_SAMPLE["wav"],
        )
    except ac.InvalidInfrastructure:
        return None


@pytest.mark.skipif(_real_data_root() is None, reason="AcousticRooms data unreachable")
def test_integration_real_sample_a2a():
    """Real AR_md sample + real rotate_scene_metadata + real GeometryConditioner capture ->
    A2a residual <= 1e-6 at j=1. Exercises the full real DATA path (not the model gates, which
    are data-independent and covered by the tiny-model unit tests)."""
    from src.data.yaw_rotation import rotate_scene_metadata

    root = _real_data_root()
    loaded = ac.load_real_sample(
        str(ac._WORKTREE_ROOT), root, ac.DEFAULT_SAMPLE["scene"], ac.DEFAULT_SAMPLE["sub"],
        ac.DEFAULT_SAMPLE["wav"],
    )
    md = loaded["md"]
    fp = ac.sample_fingerprint(md, {"scene": "Cafe"})
    assert fp["nondegenerate"] is True
    width_px = int(md["depth"].shape[-1])
    dj = ac.ROLL_MULTIPLE * 1
    alpha = dj * 2.0 * math.pi / width_px
    mc = ac.build_geometry_conditioner(int(md["depth"].shape[-2]), width_px, ac.GEOM_MAX_VALUE)
    c_base = ac.capture_encoder_input(mc, md)
    c_rot = ac.capture_encoder_input(mc, rotate_scene_metadata(md, alpha, width_px))
    assert ac.a2a_residual(c_base, c_rot, dj, alpha) <= ac.A2A_RTOL
