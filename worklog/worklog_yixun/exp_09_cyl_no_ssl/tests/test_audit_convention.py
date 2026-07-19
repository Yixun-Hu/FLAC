"""exp-09 Stage A -- convention-audit tests (TDD, CPU).

Every gate is exercised RED-first (a fabricated violation makes the named test fail) and then
green. The unit layer runs entirely on tiny synthetic fixtures (a fabricated "sample" driving the
same code paths, with W divisible by 16 and a tiny random-init cylindrical model -- the
equivariance is structural, not weight-dependent). One integration test uses the real sample and
is skipped if the AcousticRooms data is unreachable.

Structural mutation map (Stage A round 1):
  m1 A2a comparator disabled          -> test_a2a_detects_wrong_convention        RED
  m2 undo-roll direction flipped      -> test_a2b_patch_equivariance_pass         RED
  m3 negative-control threshold flip  -> test_a2c_controls_pass                   RED
  m4 status maps control-fail wrong   -> test_classify_control_failure_is_invalid RED
  m5 atomic write bypassed            -> test_atomic_write_uses_tmp_rename         RED

Fix-round-2 mutation map (fail-closed gates; see the Coder report):
  M1 provenance identity -> presence-check -> test_evaluate_provenance_wrong_package_sha_fails RED
  M2 fallback writer unguarded (re-raise)  -> test_runner_serialization_failure_exits_3        RED
  M3 fingerprint gate True when absent     -> test_main_absent_fingerprint_is_invalid          RED
  M4 angle_spec rescales, not rejects      -> test_angle_spec_rejects_wrong_width              RED
  M5 A1 pattern-assert removed             -> test_a1_source_drift_is_invalid                  RED

Fix-round-3 mutation map (exact-output cleanliness exclusion; scoped package-source gate):
  M1 exclusion widened back to parent dir  -> test_cleanliness_excludes_only_output_not_runner RED
  M2 scoped package-src gate removed       -> test_evaluate_provenance_dirty_package_src_fails  RED
"""
import json
import math
import os
import shutil
import subprocess
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
    field = ac.capture_encoder_input(mc, synthetic_md, expected_hw=(H, W))
    assert tuple(field.shape) == (1, 3, H, W)
    assert torch.isfinite(field).all()


def test_a2a_residual_pass(synthetic_md):
    """The REAL rotated pipeline field equals Rz(alpha).Roll_{16j}(base) to <= 1e-6."""
    from src.data.yaw_rotation import rotate_scene_metadata

    mc = ac.build_geometry_conditioner(H, W, ac.GEOM_MAX_VALUE)
    c_base = ac.capture_encoder_input(mc, synthetic_md, expected_hw=(H, W))
    md_rot = rotate_scene_metadata(synthetic_md, ALPHA, W)
    c_rot = ac.capture_encoder_input(mc, md_rot, expected_hw=(H, W))
    residual = ac.a2a_residual(c_base, c_rot, DJ, ALPHA)
    assert residual <= ac.A2A_RTOL, f"A2a residual {residual:.3e} exceeds {ac.A2A_RTOL}"


def test_a2a_detects_wrong_convention(synthetic_md):
    """RED-first sensitivity (mutation m1): a WRONG-direction roll must NOT pass the A2a gate.
    If the comparator is disabled (compares c_rot to itself), this residual collapses to 0 and
    the assertion fails."""
    from src.data.yaw_rotation import rotate_scene_metadata

    mc = ac.build_geometry_conditioner(H, W, ac.GEOM_MAX_VALUE)
    c_base = ac.capture_encoder_input(mc, synthetic_md, expected_hw=(H, W))
    md_rot = rotate_scene_metadata(synthetic_md, ALPHA, W)
    c_rot = ac.capture_encoder_input(mc, md_rot, expected_hw=(H, W))
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
    assert a1["valid"] is True and a1["check_failures"] == []
    basis = a1["source_basis"]
    # AR_md azimuth / elevation / point-cloud, quoted with the live literals.
    assert "2.0 * np.pi / img_w" in basis["AR_md.azimuth_theta"]["text"]
    assert "np.pi / img_h" in basis["AR_md.elevation_phi"]["text"]
    assert "cos_phi * cos_theta" in basis["AR_md.point_cloud_xyz"]["text"]
    # yaw_rotation: depth roll + Rz matrix.
    assert "torch.roll(depth" in basis["yaw_rotation.roll_depth"]["text"]
    assert "[[c, -s, 0.0]" in basis["yaw_rotation.azimuth_rotation_matrix"]["text"]
    # conditioners.py:284 encoder expression (finding 4 literal).
    assert "coord[:, i, :, None, None] - depth_coord" in basis["conditioners.encoder_input"]["text"]
    # gauge channel contract.
    assert "X_CHANNEL = 0" in basis["gauge.channel_contract"]["text"]


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


# =============================================================================================
# Fix-round-2: fail-closed provenance / fingerprint / geometry gates, total boundary, A1 drift.
# =============================================================================================
FAKE_PROV = {
    "cylindrical_dinov3": {"version": "0.0.0-test",
                           "package_file": "/pkg/src/cylindrical_dinov3/__init__.py",
                           "git": {"sha": "PKGSHA0", "dirty": False},
                           "src_status": {"pathspec": "src/cylindrical_dinov3",
                                          "clean": True, "dirty_paths": []}},
    "official_weights": {"checkpoint": "/ckpt/REVX", "revision": "REVX",
                         "weights_file": "/ckpt/REVX/model.safetensors", "sha256": "WSHA0"},
    "flac_worktree": {"path": "/wt", "git": {"sha": "WTSHA0", "dirty": False},
                      "clean_excluding_output": True},
}


def _init_fixture_repo(root, files):
    """Create a REAL throwaway git repo at ``root`` with ``files`` (relpath -> content) committed
    clean. Used to drive the REAL cleanliness / scoped-status filters on REAL porcelain output."""
    root = str(root)
    subprocess.run(["git", "init", "-q", root], check=True)
    subprocess.run(["git", "-C", root, "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", root, "config", "user.name", "t"], check=True)
    subprocess.run(["git", "-C", root, "config", "commit.gpgsign", "false"], check=True)
    for rel, content in files.items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            fh.write(content)
    subprocess.run(["git", "-C", root, "add", "-A"], check=True)
    subprocess.run(["git", "-C", root, "commit", "-qm", "init"], check=True)


def _real_scoped_status(repo, pathspec="src/cylindrical_dinov3"):
    """Test-local, dependency-free scoped ``git status --porcelain -- <pathspec>`` on a real repo,
    producing the same dict shape the production seam serialises. Kept independent of any new
    audit_convention symbol so the blocker-2 RED-first proof runs against the pre-fix code too."""
    out = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--", pathspec]).decode()
    dirty = [ln[3:].strip() for ln in out.splitlines() if ln.strip()]
    return {"pathspec": pathspec, "clean": (len(dirty) == 0), "dirty_paths": dirty}


def _fresh_prov():
    return json.loads(json.dumps(FAKE_PROV))


def _good_expectations():
    return {"package_sha": "PKGSHA0", "package_path_prefix": "/pkg", "worktree_sha": "WTSHA0",
            "expect_clean_worktree": True, "checkpoint_revision": "REVX", "weights_sha256": "WSHA0"}


# ---------------------------------------------------------------------------------------------
# t1 -- fingerprint content-sensitivity (catches a constant-hash mutant).
# ---------------------------------------------------------------------------------------------
def test_fingerprint_content_sensitive(synthetic_md):
    ident = {"scene": "s", "sub": "x", "wav": "w", "data_root": "/r"}
    base = ac.sample_fingerprint(synthetic_md, ident)["sha256"]
    md_d = dict(synthetic_md); d = synthetic_md["depth"].clone(); d[0, 0, 0] += 1e-3; md_d["depth"] = d
    assert ac.sample_fingerprint(md_d, ident)["sha256"] != base, "depth byte must change fingerprint"
    md_s = dict(synthetic_md); s = synthetic_md["source_vit"].clone(); s[0, 0] += 1e-3; md_s["source_vit"] = s
    assert ac.sample_fingerprint(md_s, ident)["sha256"] != base, "source vector must change fingerprint"


# ---------------------------------------------------------------------------------------------
# Provenance IDENTITY gate (finding 1; mutation M1 target = evaluate_provenance identity compare).
# ---------------------------------------------------------------------------------------------
def test_evaluate_provenance_all_match_ok():
    ok, reasons, checks = ac.evaluate_provenance(_fresh_prov(), _good_expectations())
    assert ok is True and reasons == []
    assert all(c["ok"] for c in checks)


def test_evaluate_provenance_wrong_package_sha_fails():
    """t2 / M1: actual sha present but != expected must FAIL the identity gate (presence-only passes)."""
    exp = _good_expectations(); exp["package_sha"] = "WRONGSHA"
    ok, reasons, _ = ac.evaluate_provenance(_fresh_prov(), exp)
    assert ok is False
    assert any("package_sha" in r and "mismatch" in r for r in reasons)


def test_evaluate_provenance_absent_expectation_fails_closed():
    for key in ["package_sha", "package_path_prefix", "worktree_sha", "checkpoint_revision",
                "weights_sha256"]:
        exp = _good_expectations(); exp[key] = None
        ok, reasons, _ = ac.evaluate_provenance(_fresh_prov(), exp)
        assert ok is False, f"absent {key} must fail closed"
        assert any(key in r and "absent" in r for r in reasons)
    exp = _good_expectations(); exp["expect_clean_worktree"] = False
    assert ac.evaluate_provenance(_fresh_prov(), exp)[0] is False, "absent clean-worktree flag must fail"


def test_evaluate_provenance_fake_package_path_and_dirty_fail(tmp_path):
    """Finding 1 scenario: a fake package at /wrong/package.py must NOT pass; and a worktree whose
    cleanliness is False fails the gate. The dirty-worktree bool is DERIVED from the REAL filter on
    a real dirty fixture (no injected pre-derived boolean -- r2 flagged the old tests:409 for that)."""
    prov = _fresh_prov(); prov["cylindrical_dinov3"]["package_file"] = "/wrong/package.py"
    ok, reasons, _ = ac.evaluate_provenance(prov, _good_expectations())
    assert ok is False and any("package_path_prefix" in r for r in reasons)
    # Derive clean_excluding_output from the REAL filter on a real fixture with a dirty runner file.
    repo = tmp_path / "wt"
    _init_fixture_repo(repo, {ac.OUTPUT_RELDIR + "/audit_convention.py": "r\n",
                              ac.OUTPUT_JSON_RELPATH: "{}\n"})
    with open(str(repo / ac.OUTPUT_RELDIR / "audit_convention.py"), "w") as fh:
        fh.write("tampered\n")
    clean = ac._worktree_clean_excluding(str(repo), ac.OUTPUT_JSON_RELPATH)
    assert clean is False  # real filter: a dirty runner is not excluded
    prov2 = _fresh_prov(); prov2["flac_worktree"]["clean_excluding_output"] = clean
    assert ac.evaluate_provenance(prov2, _good_expectations())[0] is False


# ---------------------------------------------------------------------------------------------
# Geometry gates (finding 3; mutation M4 = angle_spec rescales instead of rejecting).
# ---------------------------------------------------------------------------------------------
def test_angle_spec_rejects_wrong_width():
    with pytest.raises(ac.InvalidInfrastructure):
        ac.angle_spec(128, registered_width=512)
    spec = ac.angle_spec(512, registered_width=512)  # matching width is fine
    assert [s["j"] for s in spec] == list(ac.J_VALUES)


def test_capture_rejects_wrong_geometry(synthetic_md):
    mc = ac.build_geometry_conditioner(H, W, ac.GEOM_MAX_VALUE)
    with pytest.raises(ac.InvalidInfrastructure):
        ac.capture_encoder_input(mc, synthetic_md, expected_hw=(256, 512))  # actual is 64x128


# ---------------------------------------------------------------------------------------------
# t3 -- capture sentinel: hook never fires => invalid (never a fabricated field).
# ---------------------------------------------------------------------------------------------
def test_capture_sentinel_not_fired_is_invalid(synthetic_md, monkeypatch):
    mc = ac.build_geometry_conditioner(H, W, ac.GEOM_MAX_VALUE)
    geom = mc.conditioners["source_vit"]

    def _no_vit_forward(self, coord, device="cpu"):
        raise RuntimeError("forward replaced; self.vit never called")

    monkeypatch.setattr(type(geom), "forward", _no_vit_forward)
    with pytest.raises(ac.InvalidInfrastructure):
        ac.capture_encoder_input(mc, synthetic_md, expected_hw=(H, W))


# ---------------------------------------------------------------------------------------------
# t6 -- A1 source drift => invalid (mutation M5 = pattern-assert removed).
# ---------------------------------------------------------------------------------------------
def test_a1_source_drift_is_invalid(tmp_path):
    wt = tmp_path / "wt"
    real = ac._WORKTREE_ROOT
    for rel in (ac.AR_MD_REL, ac.YAW_ROTATION_REL, ac.CONDITIONERS_REL):
        dst = wt / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(real / rel), str(dst))
    assert ac.build_a1_static(str(wt))["valid"] is True, "undoctored copy must be valid"
    cond = wt / ac.CONDITIONERS_REL
    text = cond.read_text().replace(
        "(coord[:, i, :, None, None] - depth_coord) / self.max_value",
        "(coord[:, i, :, None, None] - depth_coord) * 3.14159")
    cond.write_text(text)
    a1 = ac.build_a1_static(str(wt))
    assert a1["valid"] is False
    assert any("conditioners" in name for name in a1["check_failures"])


# ---------------------------------------------------------------------------------------------
# t4 -- persistent serialization failure => exit 3, no traceback (mutation M2 = unguarded fallback).
# ---------------------------------------------------------------------------------------------
def _always_raise(*a, **k):
    raise OSError("simulated persistent serialization failure")


def test_runner_serialization_failure_exits_3(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "atomic_write_json", _always_raise)
    rc = ac.main(["--out", str(tmp_path / "o.json"), "--data-root", "/nonexistent/root",
                  "--checkpoint", "/nonexistent/ckpt", "--expect-fingerprint", "deadbeef"])
    assert rc == 3, "a persistent write failure must still exit 3 (never propagate a traceback)"


# ---------------------------------------------------------------------------------------------
# t5 -- main()-level exit codes 0/2/3 on tiny fixtures, through the real gates.
# ---------------------------------------------------------------------------------------------
@pytest.fixture
def tiny_env(monkeypatch, synthetic_md, tmp_path):
    """Drive run_audit / main end-to-end on a tiny synthetic sample at (H,W)=(64,128) with tiny
    cylindrical models. Data/model/provenance ACQUISITION is stubbed; every GATE is production
    code (a1, provenance evaluation, fingerprint, geometry, capture, A2a/b/c, serialisation)."""
    scene, sub, wav = "synthetic", "syn_idx_0", "S001_R0001_hybrid_IR.wav"
    data_root = "/fake/root"
    ident = {"scene": scene, "sub": sub, "wav": wav, "data_root": data_root}
    fp = ac.sample_fingerprint(synthetic_md, ident)["sha256"]

    monkeypatch.setattr(ac, "build_provenance", lambda *a, **k: _fresh_prov())
    monkeypatch.setattr(ac, "resolve_data_root", lambda *a, **k: data_root)
    monkeypatch.setattr(ac, "load_real_sample", lambda *a, **k: {
        "md": synthetic_md, "rel": "syn/rel", "full": "/x", "data_root": data_root,
        "scene": scene, "sub": sub, "wav": wav})
    monkeypatch.setattr(ac, "load_cylindrical_model", lambda ckpt, gauge: _tiny_cyl_model(gauge))
    monkeypatch.setattr(ac, "J_VALUES", (1, 2, 3))  # non-wrapping at W=128 (W_t=8, rolls < W_t/2)

    out = str(tmp_path / "audit_convention.json")
    good_argv = [
        "--out", out, "--geometry", str(H), str(W),
        "--scene", scene, "--sub", sub, "--wav", wav,
        "--expect-fingerprint", fp,
        "--expect-package-sha", "PKGSHA0",
        "--expect-package-path-prefix", "/pkg",
        "--expect-worktree-sha", "WTSHA0",
        "--expect-clean-worktree",
        "--expect-checkpoint-revision", "REVX",
        "--expect-weights-sha256", "WSHA0",
    ]
    return {"monkeypatch": monkeypatch, "out": out, "fp": fp,
            "sample": {"scene": scene, "sub": sub, "wav": wav}, "geometry": (H, W),
            "expectations": _good_expectations(), "good_argv": good_argv}


def _run_main_read(env, argv):
    rc = ac.main(argv)
    rec = json.loads(open(env["out"]).read())
    return rc, rec


def test_runner_returned_record_equals_persisted(tiny_env):
    """Finding 2: on the success path the RETURNED record equals the PERSISTED record exactly."""
    env = tiny_env
    rec, code = ac.run_audit(out_path=env["out"], sample=env["sample"], geometry=env["geometry"],
                             expect_fingerprint=env["fp"], expectations=env["expectations"])
    persisted = json.loads(open(env["out"]).read())
    assert rec == persisted
    assert rec["audit_status"] == ac.STATUS_VALID_PASS and code == 0


def test_main_valid_pass_exit0(tiny_env):
    rc, rec = _run_main_read(tiny_env, tiny_env["good_argv"])
    assert rec["audit_status"] == ac.STATUS_VALID_PASS, rec.get("reasons")
    assert rec["validity"]["provenance_ok"] is True and rec["validity"]["a1_valid"] is True
    assert rc == 0


def test_main_convention_failure_exit2(tiny_env):
    """Validity holds but A2b fails: model_on is the non-equivariant 'none' model => exit 2."""
    tiny_env["monkeypatch"].setattr(ac, "load_cylindrical_model",
                                    lambda ckpt, gauge: _tiny_cyl_model("none"))
    rc, rec = _run_main_read(tiny_env, tiny_env["good_argv"])
    assert rec["audit_status"] == ac.STATUS_CONVENTION_FAILURE, rec.get("reasons")
    assert rec["validity"]["other_validity_pass"] is True and rec["a2b"]["all_pass"] is False
    assert rc == 2


def test_main_invalid_exit3(tiny_env):
    """A wrong registered weights sha => provenance identity fails => invalid => exit 3."""
    argv = list(tiny_env["good_argv"])
    i = argv.index("--expect-weights-sha256"); argv[i + 1] = "WRONGWEIGHTS"
    rc, rec = _run_main_read(tiny_env, argv)
    assert rec["audit_status"] == ac.STATUS_INVALID
    assert rec["validity"]["provenance_ok"] is False
    assert rc == 3


def test_main_provenance_mismatch_is_invalid(tiny_env):
    """t2 production-connected: wrong expected package SHA through the real run_audit gate => invalid."""
    argv = list(tiny_env["good_argv"])
    i = argv.index("--expect-package-sha"); argv[i + 1] = "WRONGPKG"
    rc, rec = _run_main_read(tiny_env, argv)
    assert rc == 3 and rec["audit_status"] == ac.STATUS_INVALID
    assert rec["validity"]["provenance_ok"] is False
    assert any("package_sha" in r for r in rec["reasons"])


def test_main_absent_fingerprint_is_invalid(tiny_env):
    """Mutation M3: dropping --expect-fingerprint must yield invalid (fail closed), not valid_pass."""
    argv = list(tiny_env["good_argv"])
    i = argv.index("--expect-fingerprint"); del argv[i:i + 2]
    rc, rec = _run_main_read(tiny_env, argv)
    assert rc == 3 and rec["audit_status"] == ac.STATUS_INVALID
    assert rec["validity"]["fingerprint_ok"] is False
    assert "fingerprint_expectation_absent" in rec["reasons"]


# =============================================================================================
# Fix-round-3 blocker 1: the clean-worktree derivation excludes ONLY the exact output JSON
# (+ atomic-writer temp siblings), NEVER the parent directory. Real filter on real porcelain.
# =============================================================================================
def test_cleanliness_excludes_only_output_not_runner(tmp_path):
    """Blocker 1 / mutation M1: a dirty ``audit_convention.py`` (or anything else under
    OUTPUT_RELDIR) must make the worktree NOT clean -- the parent directory must NOT be excluded.
    Production-connected through the REAL ``build_provenance`` -> ``_worktree_clean_excluding``
    filter on a REAL fixture repo's REAL ``git status --porcelain`` (no injected boolean).

    RED-first (pre-fix): build_provenance excluded the whole OUTPUT_RELDIR directory, so a dirty
    runner returned clean_excluding_output=True. Widening the exclusion back to the parent
    directory re-introduces exactly that defect and re-reddens this test."""
    repo = tmp_path / "wt"
    _init_fixture_repo(repo, {
        ac.OUTPUT_RELDIR + "/audit_convention.py": "print('runner')\n",
        ac.OUTPUT_RELDIR + "/tests/test_audit_convention.py": "x = 1\n",
        # Reference via OUTPUT_RELDIR (not the new OUTPUT_JSON_RELPATH constant) so this headline
        # blocker-1 test also runs against the pre-fix code for a clean-assertion RED-first proof.
        ac.OUTPUT_RELDIR + "/audit_convention.json": "{}\n",
        "readme.md": "hi\n",
    })
    # Clean repo -> clean (3-arg build_provenance uses the blessed default output path).
    prov = ac.build_provenance("/no/ckpt", str(repo), str(repo))
    assert prov["flac_worktree"]["clean_excluding_output"] is True
    # Dirty the runner itself -> the parent dir is NOT excluded -> not clean.
    with open(str(repo / ac.OUTPUT_RELDIR / "audit_convention.py"), "w") as fh:
        fh.write("print('tampered')\n")
    prov = ac.build_provenance("/no/ckpt", str(repo), str(repo))
    assert prov["flac_worktree"]["clean_excluding_output"] is False, \
        "a dirty audit_convention.py must make the worktree not-clean (parent dir must NOT be excluded)"


def test_worktree_clean_excluding_exact_output_and_tmp(tmp_path):
    """The REAL filter excludes EXACTLY the output JSON and the atomic writer's temp siblings
    (``.audit_convention.*.tmp``) in that directory, and nothing else -- driven on real porcelain."""
    repo = tmp_path / "wt"
    _init_fixture_repo(repo, {
        ac.OUTPUT_JSON_RELPATH: "{}\n",
        ac.OUTPUT_RELDIR + "/audit_convention.py": "r\n",
    })
    outp = ac.OUTPUT_JSON_RELPATH
    assert ac._worktree_clean_excluding(str(repo), outp) is True                     # clean
    with open(str(repo / ac.OUTPUT_JSON_RELPATH), "w") as fh:
        fh.write('{"x": 1}\n')                                                       # modify output
    assert ac._worktree_clean_excluding(str(repo), outp) is True                     # excluded
    tmp_sib = repo / ac.OUTPUT_RELDIR / (ac.OUTPUT_TMP_PREFIX + "AB12" + ac.OUTPUT_TMP_SUFFIX)
    with open(str(tmp_sib), "w") as fh:
        fh.write("partial\n")                                                        # atomic tmp
    assert ac._worktree_clean_excluding(str(repo), outp) is True                     # tmp excluded
    with open(str(repo / ac.OUTPUT_RELDIR / "audit_convention.py"), "w") as fh:
        fh.write("tampered\n")                                                        # runner dirty
    assert ac._worktree_clean_excluding(str(repo), outp) is False                    # NOT excluded


def test_worktree_clean_excluding_ignores_output_outside_worktree(tmp_path):
    """An out_path resolving OUTSIDE the worktree excludes nothing (any dirty path => not clean)."""
    repo = tmp_path / "wt"
    _init_fixture_repo(repo, {ac.OUTPUT_JSON_RELPATH: "{}\n"})
    with open(str(repo / ac.OUTPUT_JSON_RELPATH), "w") as fh:
        fh.write('{"x": 1}\n')
    assert ac._worktree_clean_excluding(str(repo), "/somewhere/else/audit_convention.json") is False


# =============================================================================================
# Fix-round-3 blocker 2: scoped package-source cleanliness gate (src/cylindrical_dinov3).
# =============================================================================================
def test_evaluate_provenance_dirty_package_src_fails(tmp_path):
    """Blocker 2 (a) / mutation M2: a dirty path UNDER src/cylindrical_dinov3/ must fail provenance,
    even with every registered expectation matching. Production-connected through the REAL
    ``evaluate_provenance`` gate on a scoped listing from a REAL fixture repo's REAL porcelain.

    RED-first (pre-fix): evaluate_provenance recorded cylindrical_dinov3.git.dirty but never gated
    it, so this returned ok=True. Removing the scoped gate re-reddens this test."""
    repo = tmp_path / "pkg"
    _init_fixture_repo(repo, {"src/cylindrical_dinov3/gauge.py": "X_CHANNEL = 0\n",
                              "worklog/x.md": "prep\n"})
    with open(str(repo / "src/cylindrical_dinov3/gauge.py"), "w") as fh:
        fh.write("X_CHANNEL = 0  # tampered\n")
    status = _real_scoped_status(repo)                       # REAL scoped porcelain
    assert status["clean"] is False and status["dirty_paths"]
    prov = _fresh_prov()
    prov["cylindrical_dinov3"]["git"]["dirty"] = True        # mirror the blocker exactly
    prov["cylindrical_dinov3"]["src_status"] = status
    ok, reasons, checks = ac.evaluate_provenance(prov, _good_expectations())
    assert ok is False, "a dirty package-source subtree must fail provenance"
    assert any("package_src" in r and "dirty" in r for r in reasons)
    # Prove the ONLY failing gate is the scoped-src one (all registered expectations still match).
    failed = [c["name"] for c in checks if not c["ok"]]
    assert failed == ["package_src_clean"], failed


def test_evaluate_provenance_dirty_elsewhere_in_package_passes(tmp_path):
    """Blocker 2 (b): a dirty path OUTSIDE src/cylindrical_dinov3/ (e.g. a worklog record for a
    commit being prepared) must NOT fail -- the gate is scoped both ways. Real scoped porcelain."""
    repo = tmp_path / "pkg"
    _init_fixture_repo(repo, {"src/cylindrical_dinov3/gauge.py": "X_CHANNEL = 0\n",
                              "worklog/x.md": "prep\n"})
    with open(str(repo / "worklog/x.md"), "w") as fh:
        fh.write("prep -- commits being prepared\n")          # dirty, but OUTSIDE the scope
    status = _real_scoped_status(repo)
    assert status["clean"] is True and status["dirty_paths"] == []
    prov = _fresh_prov(); prov["cylindrical_dinov3"]["src_status"] = status
    ok, reasons, _ = ac.evaluate_provenance(prov, _good_expectations())
    assert ok is True, reasons


def test_build_provenance_records_scoped_src_status(tmp_path):
    """The production ``build_provenance`` seam runs the REAL scoped git status and SERIALISES the
    dirty listing under cylindrical_dinov3.src_status (blocker 2: scoping + serialisation)."""
    repo = tmp_path / "pkg"
    _init_fixture_repo(repo, {"src/cylindrical_dinov3/gauge.py": "X_CHANNEL = 0\n",
                              "worklog/x.md": "prep\n"})
    with open(str(repo / "src/cylindrical_dinov3/gauge.py"), "w") as fh:
        fh.write("X_CHANNEL = 0  # tampered\n")
    with open(str(repo / "worklog/x.md"), "w") as fh:
        fh.write("also dirty but out of scope\n")
    prov = ac.build_provenance("/no/ckpt", str(repo), str(repo),
                               str(repo / ac.OUTPUT_JSON_RELPATH))
    ss = prov["cylindrical_dinov3"]["src_status"]
    assert ss["pathspec"] == ac.PACKAGE_SRC_PATHSPEC
    assert ss["clean"] is False
    assert any("src/cylindrical_dinov3" in p for p in ss["dirty_paths"])
    assert all("worklog" not in p for p in ss["dirty_paths"]), "scope must exclude worklog changes"
