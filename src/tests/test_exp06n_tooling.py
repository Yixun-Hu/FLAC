"""exp_06 (worklog_yixun_neuronic) — tests for the exp06n TOOLING: the pre-launch
arm-wiring gate (bitwise-legacy oracle + mean-pool sabotage), the trained-checkpoint
equivariance guard (B3: non-vacuity asserts + gauge-off negative control), and the
launch/probe/eval script contracts.

Companion to ``test_exp06n_cond_pool.py`` (which pins the model delta itself). Everything
here is CPU-only and offline; the gate/guard helpers are exercised on TINY random-weight
cylindrical backbones, never on the Hub.

Run-script identity: exp06n_launch.sh must be exp03n_launch.sh under ONE declared
arm-token substitution (``ARM_SUBSTITUTIONS``) — byte-for-byte, no functional edit. The
PROBE is the one deliberate exception (plan B5): it WRITES exp06n_frozen_min_free.txt
itself, so its diff against the renamed exp03n probe must consist EXCLUSIVELY of lines
marked ``EXP06 PROBE DIFF n/N`` (complete numbering) plus removed lines that all belong
to the old by-hand frozen-file protocol. The Slurm wrappers are pure renames.
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import copy  # noqa: E402
import difflib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_EXP09_DIR = _REPO_ROOT / "worklog" / "worklog_yixun" / "exp_09_cyl_no_ssl"
_GATE_PATH = _EXP09_DIR / "assert_arm_configs_exp06n.py"
_GUARD_PATH = _EXP09_DIR / "guard_exp06n_equivariance.py"

from src.tests.test_exp03n_cond_pool import (  # noqa: E402
    _HIDDEN,
    _build,
    _cyl_conditioning,
    _geoms,
    _save_tiny_cyl,
)
from src.tests.test_exp06n_cond_pool import _full_model_config  # noqa: E402


def _load_module(path: Path, name: str):
    if not path.exists():
        pytest.fail(f"required deliverable is missing: {path}")
    if str(_EXP09_DIR) not in sys.path:
        sys.path.insert(0, str(_EXP09_DIR))
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate():
    return _load_module(_GATE_PATH, "assert_arm_configs_exp06n")


@pytest.fixture(scope="module")
def guard():
    return _load_module(_GUARD_PATH, "guard_exp06n_equivariance")


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    return _save_tiny_cyl(tmp_path_factory.mktemp("exp06n_tooling_tiny"))


# ------------------------------------------------------------------------------------ #
# 1. gate: per-variant DUAL config contract (B1)
# ------------------------------------------------------------------------------------ #
def test_gate_accepts_both_real_config_variants(gate):
    for name, variant in (("FLAC_AR_exp06n.json", "base"),
                          ("FLAC_AR_exp06n_online_eval.json", "online")):
        cfg, resolved = gate.assert_config_contract(str(_EXP09_DIR / name))
        assert resolved == variant, (name, resolved)
        assert cfg["model"]["conditioning"]["configs"][1]["config"]["ViT"]["cond_pool"] == "max_linear"


def test_gate_variant_autodetect_is_filename_driven(gate):
    assert gate.variant_for_config("/x/FLAC_AR_exp06n_online_eval.json") == "online"
    assert gate.variant_for_config("/x/FLAC_AR_exp06n.json") == "base"


def test_gate_rejects_the_wrong_reference_for_a_variant(gate):
    with pytest.raises(RuntimeError, match="contract|reconstruct"):
        gate.assert_config_contract(str(_EXP09_DIR / "FLAC_AR_exp06n_online_eval.json"),
                                    variant="base")
    with pytest.raises(RuntimeError, match="contract|reconstruct"):
        gate.assert_config_contract(str(_EXP09_DIR / "FLAC_AR_exp06n.json"), variant="online")


def test_gate_rejects_the_exp03n_configs(gate):
    """Arm confusion is the failure this gate exists to prevent: the exp_03 configs are
    identical except the knob value + width key, so a swapped --config must be refused."""
    for name in ("FLAC_AR_exp03n.json", "FLAC_AR_exp03n_online_eval.json"):
        with pytest.raises(RuntimeError, match="cond_pool|contract"):
            gate.assert_config_contract(str(_EXP09_DIR / name))


@pytest.mark.parametrize("mutation", ["extra_key", "changed_recipe", "dropped_key",
                                      "width_key_present", "one_block_only"])
def test_gate_rejects_mutated_configs(gate, tmp_path, mutation):
    cfg = json.load(open(_EXP09_DIR / "FLAC_AR_exp06n.json"))
    vits = [c for c in cfg["model"]["conditioning"]["configs"] if c["type"] == "ViTCoordinates"]
    if mutation == "extra_key":
        vits[0]["config"]["ViT"]["cond_mlp_dropout"] = 0.1
        vits[1]["config"]["ViT"]["cond_mlp_dropout"] = 0.1
    elif mutation == "changed_recipe":
        cfg["training"]["frame_avg_angles"] = [0.0, 90.0, 180.0, 270.0]
    elif mutation == "dropped_key":
        del vits[1]["config"]["ViT"]["cond_pool"]
    elif mutation == "width_key_present":
        # the exp_03 leftover: a width key beside max_linear must be refused (B1: exactly
        # {cond_mlp_hidden} is REMOVED vs exp03n)
        vits[0]["config"]["ViT"]["cond_mlp_hidden"] = 384
        vits[1]["config"]["ViT"]["cond_mlp_hidden"] = 384
    elif mutation == "one_block_only":
        del vits[1]["config"]["ViT"]["cond_pool"]
    path = tmp_path / "FLAC_AR_exp06n.json"
    path.write_text(json.dumps(cfg, indent=4) + "\n")
    with pytest.raises(RuntimeError):
        gate.assert_config_contract(str(path))


def test_gate_sibling_contract_is_per_variant(gate):
    """B1 applies base<->base and online<->online: the gate must expose the sibling
    (exp03n) reference per variant and never mix them."""
    assert gate.VARIANT_REFERENCE["base"][2] == "FLAC_AR_exp03n.json"
    assert gate.VARIANT_REFERENCE["online"][2] == "FLAC_AR_exp03n_online_eval.json"


# ------------------------------------------------------------------------------------ #
# 2. gate: instantiated-model arm wiring (B2)
# ------------------------------------------------------------------------------------ #
def test_gate_head_wiring_accepts_a_correct_max_linear_build(gate, tiny_dir):
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="max_linear"))
    head = gate.assert_head_wiring(mc, hidden_size=_HIDDEN, cond_dim=256)
    assert head is _geoms(mc)[0].lin_proj


def test_gate_head_wiring_rejects_the_legacy_build(gate, tiny_dir):
    """The legacy arm ALSO carries a bare Linear, so the pooling selector separates it."""
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True))
    with pytest.raises(RuntimeError, match="dino_pool"):
        gate.assert_head_wiring(mc, hidden_size=_HIDDEN, cond_dim=256)


def test_gate_head_wiring_rejects_the_exp03_max_mlp_build(gate, tiny_dir):
    """Max pooling ALONE does not identify exp_06 — the exp_03 arm max-pools too. The
    bare-Linear head shape is what separates them; an MLP head must be refused."""
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="max_mlp"))
    with pytest.raises(RuntimeError, match="head|Linear"):
        gate.assert_head_wiring(mc, hidden_size=_HIDDEN, cond_dim=256)


def test_gate_head_wiring_rejects_an_unshared_head(gate, tiny_dir):
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="max_linear"))
    geoms = _geoms(mc)
    geoms[1].lin_proj = copy.deepcopy(geoms[0].lin_proj)
    with pytest.raises(RuntimeError, match="shared"):
        gate.assert_head_wiring(mc, hidden_size=_HIDDEN, cond_dim=256)


def test_gate_param_delta_must_be_exactly_zero(gate, tiny_dir):
    legacy = _build(_cyl_conditioning(tiny_dir, with_context=True))
    new = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="max_linear"))
    gate.assert_param_delta(legacy, new)          # expected == 0, must pass
    exp03 = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="max_mlp",
                                     cond_mlp_hidden=_HIDDEN))
    with pytest.raises(RuntimeError, match="trainable"):
        gate.assert_param_delta(legacy, exp03)    # the MLP arm's +hidden-layer delta != 0


def test_gate_live_forward_matches_the_external_oracle(gate, tiny_dir):
    gate.assert_live_forward(_build(_cyl_conditioning(tiny_dir, with_context=True,
                                                      cond_pool="max_linear")),
                             height=64, width=128)


def test_gate_live_forward_catches_a_mean_pooling_regression(gate, tiny_dir):
    """Sabotage: flip the served pooling to mean while everything else stays. The state
    dict is UNCHANGED (pooling has no parameters), so ONLY the live forward can catch it —
    that is why B2 makes the external oracle + sabotage mandatory."""
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="max_linear"))
    for geom in _geoms(mc):
        geom.dino_pool = "mean"
    with pytest.raises(RuntimeError, match="amax|forward"):
        gate.assert_live_forward(mc, height=64, width=128)


def test_gate_grad_flow_passes_on_a_correct_build(gate, tiny_dir):
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="max_linear"))
    gate.assert_grad_flow(mc, height=64, width=128)


def test_gate_grad_flow_catches_a_detached_conditioner(gate, tiny_dir):
    """Sabotage: detach one conditioner's forward. A single summed backward over both uses
    would hide it behind the other use of the SHARED head; the per-use check must not."""
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="max_linear"))
    geoms = _geoms(mc)
    real_forward = geoms[1].forward

    def _detached(coord, device="cpu"):
        out = real_forward(coord, device=device)
        return [out[0].detach(), out[1]]

    geoms[1].forward = _detached
    with pytest.raises(RuntimeError, match="grad"):
        gate.assert_grad_flow(mc, height=64, width=128)


# ------------------------------------------------------------------------------------ #
# 3. gate: bitwise FULL-init identity vs the legacy build (B2 1-3)
# ------------------------------------------------------------------------------------ #
def test_gate_full_init_identity_on_tiny_full_models(gate, tiny_dir):
    gate.assert_full_init_identity(_full_model_config(tiny_dir, cond_pool="max_linear"),
                                   _full_model_config(tiny_dir, cond_pool=None), seed=42)


def test_gate_full_init_identity_catches_an_extra_rng_draw(gate, tiny_dir, monkeypatch):
    """Sabotage: make the max_linear build consume ONE extra global draw. Every tensor
    drawn after the conditioner then differs AND the post-build RNG states diverge."""
    real = gate.create_model_from_config

    def _leaky(cfg, *args, **kwargs):
        model = real(cfg, *args, **kwargs)
        blocks = [c["config"]["ViT"] for c in cfg["model"]["conditioning"]["configs"]
                  if c["type"] == "ViTCoordinates"]
        if blocks and blocks[0].get("cond_pool") == "max_linear":
            torch.randn(1)      # the leak
        return model

    monkeypatch.setattr(gate, "create_model_from_config", _leaky)
    with pytest.raises(RuntimeError, match="RNG|bitwise"):
        gate.assert_full_init_identity(_full_model_config(tiny_dir, cond_pool="max_linear"),
                                       _full_model_config(tiny_dir, cond_pool=None), seed=42)


def test_gate_full_init_identity_catches_a_perturbed_tensor(gate, tiny_dir, monkeypatch):
    real = gate.create_model_from_config

    def _perturbing(cfg, *args, **kwargs):
        model = real(cfg, *args, **kwargs)
        blocks = [c["config"]["ViT"] for c in cfg["model"]["conditioning"]["configs"]
                  if c["type"] == "ViTCoordinates"]
        if blocks and blocks[0].get("cond_pool") == "max_linear":
            with torch.no_grad():
                model.conditioner.conditioners["source_vit"].lin_proj.weight.add_(1.0)
        return model

    monkeypatch.setattr(gate, "create_model_from_config", _perturbing)
    with pytest.raises(RuntimeError, match="bitwise"):
        gate.assert_full_init_identity(_full_model_config(tiny_dir, cond_pool="max_linear"),
                                       _full_model_config(tiny_dir, cond_pool=None), seed=42)


# ------------------------------------------------------------------------------------ #
# 4. gate: the worktree pin (EXPECT_EXP06_SHA, with the inherited alias; B4)
# ------------------------------------------------------------------------------------ #
class _Stop(Exception):
    """Sentinel: stop ``main`` right after the pin is resolved (no heavy work follows)."""


@pytest.fixture
def pin_probe(gate, monkeypatch):
    """Run ``gate.main`` far enough to capture the worktree pin it resolved."""
    seen = {}

    def _record(repo, sha, strict=True):
        seen["sha"], seen["strict"] = sha, strict
        raise _Stop()

    monkeypatch.setattr(gate.exp09_gate, "assert_exp09_provenance", _record)
    for name in ("EXPECT_EXP06_SHA", "EXPECT_EXP09_SHA"):
        monkeypatch.delenv(name, raising=False)

    def run(argv=()):
        with pytest.raises(_Stop):
            gate.main([*argv])
        return seen

    return run


def test_gate_reads_the_exp06_pin_from_the_env(pin_probe, monkeypatch):
    monkeypatch.setenv("EXPECT_EXP06_SHA", "aaaa1111")
    assert pin_probe()["sha"] == "aaaa1111"


def test_gate_still_accepts_the_inherited_exp09_pin_env(pin_probe, monkeypatch):
    monkeypatch.setenv("EXPECT_EXP09_SHA", "bbbb2222")
    assert pin_probe()["sha"] == "bbbb2222"


def test_gate_reads_the_pin_from_either_cli_flag(pin_probe):
    assert pin_probe(["--expect-exp06-sha", "cccc3333"])["sha"] == "cccc3333"
    assert pin_probe(["--expect-exp09-sha", "dddd4444"])["sha"] == "dddd4444"


# The four registered spellings are ALIASES OF ONE VALUE. Any two supplied with DIFFERENT
# SHAs must refuse (B4, the exp_04 lesson): a precedence chain would run pinned to the
# winner while the operator believed the other — most dangerously env/env, a stale exp_03
# export sitting beside a fresh exp_06 one.
_PIN_SPELLINGS = ("--expect-exp06-sha", "--expect-exp09-sha",
                  "EXPECT_EXP06_SHA", "EXPECT_EXP09_SHA")


def _apply_pins(monkeypatch, pins):
    argv = []
    for name, value in pins.items():
        assert name in _PIN_SPELLINGS, name
        if name.startswith("--"):
            argv += [name, value]
        else:
            monkeypatch.setenv(name, value)
    return argv


@pytest.mark.parametrize("pins", [
    # env/env — the hazard a precedence chain hides
    {"EXPECT_EXP06_SHA": "aaaa1111", "EXPECT_EXP09_SHA": "bbbb2222"},
    # env/CLI, both orientations and both spellings
    {"EXPECT_EXP06_SHA": "aaaa1111", "--expect-exp09-sha": "bbbb2222"},
    {"EXPECT_EXP09_SHA": "bbbb2222", "--expect-exp06-sha": "aaaa1111"},
    {"EXPECT_EXP06_SHA": "aaaa1111", "--expect-exp06-sha": "cccc3333"},
    {"EXPECT_EXP09_SHA": "bbbb2222", "--expect-exp09-sha": "dddd4444"},
    # CLI/CLI
    {"--expect-exp06-sha": "aaaa1111", "--expect-exp09-sha": "bbbb2222"},
    # three sources, one dissenter
    {"EXPECT_EXP06_SHA": "aaaa1111", "EXPECT_EXP09_SHA": "aaaa1111",
     "--expect-exp06-sha": "cccc3333"},
])
def test_gate_refuses_any_disagreeing_pin_spellings(gate, pin_probe, monkeypatch, pins):
    argv = _apply_pins(monkeypatch, pins)
    with pytest.raises(RuntimeError, match="disagree") as excinfo:
        gate.main(argv)
    message = str(excinfo.value)
    for name, value in pins.items():
        assert name in message and value in message, (
            f"the refusal must name every supplied source and value ({name}={value})"
        )


@pytest.mark.parametrize("pins", [
    {"EXPECT_EXP06_SHA": "eeee5555", "EXPECT_EXP09_SHA": "eeee5555"},
    {"EXPECT_EXP06_SHA": "eeee5555", "--expect-exp09-sha": "eeee5555"},
    {"EXPECT_EXP09_SHA": "eeee5555", "--expect-exp06-sha": "eeee5555"},
    {"EXPECT_EXP06_SHA": "eeee5555", "--expect-exp06-sha": "eeee5555"},
    {"EXPECT_EXP06_SHA": "eeee5555", "EXPECT_EXP09_SHA": "eeee5555",
     "--expect-exp06-sha": "eeee5555", "--expect-exp09-sha": "eeee5555"},
])
def test_gate_accepts_agreeing_duplicate_pin_spellings(pin_probe, monkeypatch, pins):
    argv = _apply_pins(monkeypatch, pins)
    assert pin_probe(argv)["sha"] == "eeee5555"


def test_gate_passes_an_absent_pin_through_strict(pin_probe):
    seen = pin_probe()
    assert seen["sha"] is None and seen["strict"] is True, seen


# ------------------------------------------------------------------------------------ #
# 5. trained-checkpoint equivariance guard (B3)
# ------------------------------------------------------------------------------------ #
_TINY_GEOM = dict(height=64, width=128)   # 4x8 tokens at patch 16 -> W_t = 8


def test_guard_passes_on_a_random_weight_mini_model(guard, tiny_dir):
    """The guard tests the ACTUAL served condition Linear(amax(tokens)) — architectural,
    so it must hold at random init, not only on trained weights."""
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="max_linear"))
    residuals = guard.check_equivariance(_geoms(mc)[0], **_TINY_GEOM)
    assert set(residuals) == {90.0, 180.0, 270.0}
    assert max(residuals.values()) <= guard.DEFAULT_BOUND, residuals


def test_guard_refuses_the_max_mlp_arm(guard, tiny_dir):
    """dino_pool == 'max' alone does NOT identify this arm — exp_03 max-pools too (B3:
    the guard must refuse max+MLP rather than silently certify the wrong arm)."""
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="max_mlp"))
    with pytest.raises(RuntimeError, match="head|Linear"):
        guard.check_equivariance(_geoms(mc)[0], **_TINY_GEOM)


def test_guard_refuses_the_legacy_mean_arm(guard, tiny_dir):
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True))
    with pytest.raises(RuntimeError, match="dino_pool|max"):
        guard.check_equivariance(_geoms(mc)[0], **_TINY_GEOM)


def test_guard_refuses_an_angle_that_is_not_a_whole_token_column(guard, tiny_dir):
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="max_linear"))
    with pytest.raises(RuntimeError, match="column"):
        guard.check_equivariance(_geoms(mc)[0], angles=(30.0,), **_TINY_GEOM)


def test_guard_yaw_op_fidelity_assert_has_teeth(guard, tiny_dir, monkeypatch):
    """B3(b): the fixture's yaw must equal the package ``physical_yaw``. Sabotage the yaw
    op to identity — the pose still rotates, so the composed input no longer equals
    physical_yaw of the base field and the guard must refuse (not measure garbage)."""
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="max_linear"))
    monkeypatch.setattr(guard, "physical_yaw", lambda field, shift: field)
    with pytest.raises(RuntimeError, match="physical_yaw|fixture"):
        guard.check_equivariance(_geoms(mc)[0], **_TINY_GEOM)


def test_guard_positive_check_fails_gauge_off(guard, tmp_path):
    """With the gauge disabled the token field no longer column-rolls, so the POSITIVE
    check must refuse (either the roll assert or the residual bound trips)."""
    gauge_off = _save_tiny_cyl(tmp_path / "tiny_gauge_off")
    mc = _build(_cyl_conditioning(gauge_off, with_context=True, cond_pool="max_linear",
                                  gauge="none"))
    with pytest.raises(RuntimeError, match="roll|residual|equivarian"):
        guard.check_equivariance(_geoms(mc)[0], **_TINY_GEOM)


def test_guard_negative_control_passes_on_a_gauge_off_build(guard, tmp_path):
    """B3(f): the gauge-off negative control must MEASURE broken invariance (>= 1e-3)."""
    gauge_off = _save_tiny_cyl(tmp_path / "tiny_gauge_off_nc")
    mc = _build(_cyl_conditioning(gauge_off, with_context=True, cond_pool="max_linear",
                                  gauge="none"))
    residuals = guard.check_gauge_off_negative_control(_geoms(mc)[0], **_TINY_GEOM)
    assert max(residuals.values()) >= guard.NEGATIVE_FLOOR, residuals


def test_guard_negative_control_refuses_a_gauge_on_build(guard, tiny_dir):
    """Non-vacuity of the control itself: fed an INVARIANT (gauge-on) model it must refuse
    — otherwise 'the control passed' could mean 'the control measured nothing'."""
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="max_linear"))
    with pytest.raises(RuntimeError, match="teeth|invarian"):
        guard.check_gauge_off_negative_control(_geoms(mc)[0], **_TINY_GEOM)


def _write_pl_checkpoint(tmp_path, tiny_dir):
    """A Lightning-shaped checkpoint of a TINY exp06n model (``diffusion.``-prefixed state
    dict), plus the model-config JSON that reproduces it."""
    from src.models import create_model_from_config

    cfg = _full_model_config(tiny_dir, cond_pool="max_linear")
    cfg_path = tmp_path / "FLAC_AR_exp06n.json"
    cfg_path.write_text(json.dumps(cfg, indent=4) + "\n")
    torch.manual_seed(3)
    model = create_model_from_config(copy.deepcopy(cfg))
    ckpt_path = tmp_path / "epoch=0-step=40.ckpt"
    torch.save({"state_dict": {f"diffusion.{k}": v for k, v in model.state_dict().items()}},
               ckpt_path)
    return str(ckpt_path), str(cfg_path), model


def test_guard_loads_a_lightning_checkpoint_and_serves_the_max_linear_head(guard, tiny_dir, tmp_path):
    ckpt_path, cfg_path, model = _write_pl_checkpoint(tmp_path, tiny_dir)
    geoms = guard.load_geometry_conditioners(ckpt_path, cfg_path, device="cpu")
    assert len(geoms) == 2
    assert all(g.dino_pool == "max" for g in geoms)
    ref = model.conditioner.conditioners["source_vit"].lin_proj
    assert torch.equal(geoms[0].lin_proj.weight, ref.weight), "the Linear head did not load"
    assert torch.equal(geoms[0].lin_proj.bias, ref.bias)


def test_guard_gauge_override_builds_the_gauge_off_model(guard, tiny_dir, tmp_path):
    ckpt_path, cfg_path, _ = _write_pl_checkpoint(tmp_path, tiny_dir)
    geoms = guard.load_geometry_conditioners(ckpt_path, cfg_path, device="cpu",
                                             gauge_override="none")
    assert geoms[0].vit.config.gauge == "none", (
        "gauge_override='none' must build the negative-control (gauge-off) backbone"
    )


def test_guard_cli_returns_zero_on_a_loadable_checkpoint(guard, tiny_dir, tmp_path, capsys):
    ckpt_path, cfg_path, _ = _write_pl_checkpoint(tmp_path, tiny_dir)
    rc = guard.main([ckpt_path, "--model-config", cfg_path, "--height", "64", "--width", "128"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "GUARD PASS" in out, out
    assert "negative control" in out.lower(), (
        "the CLI must RUN the gauge-off negative control, not only the positive check"
    )


# --- review B1: the PRODUCTION CLI pins the registered contract. NaN bounds make every
# comparison silently false (NaN > x and NaN < x are both False), so an overridable bound
# would let `--bound nan --negative-floor nan --angles 90` reach GUARD PASS while measuring
# nothing. The CLI must accept EXACTLY angles {90,180,270}, finite bound 1e-4 and finite
# negative floor 1e-3 — any other value refuses (string-pin style, like the MB=32 rung). --- #
@pytest.mark.parametrize("argv_extra, match", [
    (["--bound", "nan"], "bound"),
    (["--negative-floor", "nan"], "floor|negative"),
    (["--bound", "nan", "--negative-floor", "nan", "--angles", "90"], "angle|bound"),
    (["--angles", "90"], "angle"),
    (["--angles", "90,180"], "angle"),
    (["--angles", "90,180,270,45"], "angle"),
    (["--bound", "0.5"], "bound"),
    (["--bound", "inf"], "bound"),
    (["--negative-floor", "1e-9"], "floor|negative"),
    (["--negative-floor", "inf"], "floor|negative"),
])
def test_guard_cli_pins_the_registered_contract(guard, tiny_dir, tmp_path, argv_extra, match):
    ckpt_path, cfg_path, _ = _write_pl_checkpoint(tmp_path, tiny_dir)
    with pytest.raises(RuntimeError, match=match):
        guard.main([ckpt_path, "--model-config", cfg_path,
                    "--height", "64", "--width", "128", *argv_extra])


def test_guard_cli_accepts_the_registered_values_spelled_explicitly(guard, tiny_dir, tmp_path):
    """Re-stating the registered contract on the command line is NOT an override."""
    ckpt_path, cfg_path, _ = _write_pl_checkpoint(tmp_path, tiny_dir)
    rc = guard.main([ckpt_path, "--model-config", cfg_path, "--height", "64", "--width", "128",
                     "--angles", "90,180,270", "--bound", "1e-4", "--negative-floor", "1e-3"])
    assert rc == 0


def test_guard_library_refuses_nonfinite_bound_and_floor(guard, tiny_dir):
    """Defense in depth below the CLI: the library comparisons themselves refuse a
    non-finite or non-positive bound/floor rather than silently evaluating False
    against NaN."""
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="max_linear"))
    geom = _geoms(mc)[0]
    for bad in (float("nan"), float("inf"), 0.0, -1e-4):
        with pytest.raises(RuntimeError, match="finite|positive"):
            guard.check_equivariance(geom, bound=bad, **_TINY_GEOM)
        with pytest.raises(RuntimeError, match="finite|positive"):
            guard.check_gauge_off_negative_control(geom, floor=bad, **_TINY_GEOM)


# ------------------------------------------------------------------------------------ #
# 6. launch / probe / eval script contracts
# ------------------------------------------------------------------------------------ #
_LAUNCH = _EXP09_DIR / "exp06n_launch.sh"
_EXP03N_LAUNCH = _EXP09_DIR / "exp03n_launch.sh"
_EXP09_LAUNCH = _EXP09_DIR / "exp09_launch.sh"
_PROBE = _EXP09_DIR / "exp06n_probe.sh"
_EXP03N_PROBE = _EXP09_DIR / "exp03n_probe.sh"
# The Slurm wrappers live in the CYLINDRICAL repo (they are that repo's records); the
# contract is asserted here because this is where the trap lives. Skipped when that
# checkout is not beside this one.
_SLURM_DIR = Path("/n/fs/gatrdp/codespace/cylindrical-dinov3/slurm_neuronic")
_WORKTREE_ROOT_ABS = "/n/fs/gatrdp/codespace/exp06-maxpool-linear-cond"
_CYL_SRC_ABS = "/n/fs/gatrdp/codespace/cylindrical-dinov3/src"
_RECORDS_DIR_ABS = ("/n/fs/gatrdp/codespace/cylindrical-dinov3/worklog/worklog_yixun_neuronic/"
                    "exp_06_maxpool_linear_cond_claude")

# The ONLY differences allowed between an exp03n script/wrapper and its exp06n counterpart
# (probe excepted, below): arm identifiers. Ordered specific -> general so compound tokens
# (worktree root, records folder) resolve before the plain "exp03" rename.
ARM_SUBSTITUTIONS = (
    ("exp03-maxpool-mlp-cond", "exp06-maxpool-linear-cond"),
    ("maxpool_mlp_cond", "maxpool_linear_cond"),
    ("maxpoolmlp", "maxpoollinear"),
    ("max_mlp", "max_linear"),
    ("maxpool+MLP", "maxpool+Linear"),
    ("MAX-POOL + MLP", "MAX-POOL + BARE-LINEAR"),
    ("max-pool + MLP", "max-pool + bare-Linear"),
    ("cond_pool/cond_mlp_hidden", "cond_pool"),
    ("EXP09_SHA", "EXP06_SHA"),
    ("expect-exp09-sha", "expect-exp06-sha"),
    ("exp_03", "exp_06"),
    ("exp03", "exp06"),
    ("EXP03", "EXP06"),
)


def _rename_arm(text: str) -> str:
    for old, new in ARM_SUBSTITUTIONS:
        text = text.replace(old, new)
    return text


def _text(path: Path) -> str:
    if not path.exists():
        pytest.fail(f"required deliverable is missing: {path}")
    return path.read_text()


def _sbatch(name: str) -> str:
    path = _SLURM_DIR / name
    if not _SLURM_DIR.exists():
        pytest.skip(f"cylindrical slurm_neuronic checkout not present at {_SLURM_DIR}")
    if not path.exists():
        pytest.fail(f"required deliverable is missing: {path}")
    return path.read_text()


@pytest.mark.parametrize("script", ["exp06n_launch.sh", "exp06n_probe.sh"])
def test_shell_scripts_parse(script):
    path = _EXP09_DIR / script
    _text(path)
    rc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr


def test_launcher_is_the_exp03n_launcher_up_to_the_arm_identifiers():
    """Run identity, mechanically: every byte of exp06n_launch.sh outside ARM_SUBSTITUTIONS
    is exp_03's — same recipe, same rung pin, same gates, same 67,500 steps. Any real edit
    (a changed flag, a dropped refusal) shows up here as a diff."""
    got = _text(_LAUNCH)
    want = _rename_arm(_text(_EXP03N_LAUNCH))
    assert got == want, "\n".join(difflib.unified_diff(
        want.splitlines(), got.splitlines(), "renamed exp03n", "exp06n_launch.sh",
        lineterm=""))[:4000]


# The probe is the ONE deliberate functional deviation (plan B5): it WRITES the frozen
# free-VRAM file. Its diff against the renamed exp03n probe must be fully accounted for:
# every ADDED line marked `EXP06 PROBE DIFF n/N` with complete numbering, every REMOVED
# line part of the old by-hand frozen-file protocol prose.
_REMOVED_LINE_OK = re.compile(r"(?i)frozen|writ|by hand|produce")


def test_probe_diff_from_exp03n_is_fully_marked_and_frozen_file_scoped():
    got = _text(_PROBE).splitlines()
    want = _rename_arm(_text(_EXP03N_PROBE)).splitlines()
    added, removed = [], []
    for line in difflib.unified_diff(want, got, n=0, lineterm=""):
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:])
    assert added, "the probe is byte-identical to the renamed exp03n probe — it cannot be writing the frozen file"
    markers = []
    for line in added:
        m = re.search(r"EXP06 PROBE DIFF (\d+)/(\d+)", line)
        assert m, f"unmarked probe diff line: {line[:140]}"
        markers.append((int(m.group(1)), int(m.group(2))))
    totals = {t for _, t in markers}
    assert len(totals) == 1, f"inconsistent probe diff totals: {sorted(totals)}"
    total = totals.pop()
    assert sorted(n for n, _ in markers) == list(range(1, total + 1)), sorted(markers)
    assert len(added) == total, f"{len(added)} added lines but the markers claim {total}"
    for line in removed:
        assert _REMOVED_LINE_OK.search(line), (
            f"probe removed a line OUTSIDE the frozen-file protocol: {line[:140]}"
        )


def test_probe_writes_its_own_frozen_file_and_refuses_to_clobber():
    probe = _text(_PROBE)
    assert re.search(r'>\s*"\$FROZEN_FILE"', probe), (
        "the probe must WRITE the frozen free-VRAM file itself (plan B5)"
    )
    assert '"$RECOMMENDED"' in probe, "the written value must be the derived peak x 1.15"
    assert re.search(r'\[ ! -e "\$FROZEN_FILE" \]', probe), (
        "the probe must REFUSE when the frozen file already exists (never clobber a "
        "reviewed value)"
    )
    assert "exp06n_frozen_min_free.txt" in probe
    assert "exp03n_frozen_min_free.txt" not in probe, (
        "the exp_03 frozen VRAM value is another arm's measurement"
    )
    assert "8381" not in probe, "exp_03's frozen number must not be inherited anywhere"


def test_launcher_every_diff_from_exp09_is_marked_and_numbered():
    """House style (exp_02/exp_03): the launcher is a COPY of the reviewed exp-09 launcher
    in which every changed/added line carries a ``# EXP06 DIFF n/N`` marker."""
    exp09 = _text(_EXP09_LAUNCH).splitlines()
    exp06n = _text(_LAUNCH).splitlines()
    added = [line for line in difflib.unified_diff(exp09, exp06n, n=0, lineterm="")
             if line.startswith("+") and not line.startswith("+++")]
    assert added, "the launcher is byte-identical to exp09_launch.sh — nothing was adapted"
    markers = []
    for line in added:
        m = re.search(r"EXP06 DIFF (\d+)/(\d+)", line)
        assert m, f"unmarked diff line in exp06n_launch.sh: {line[1:][:120]}"
        markers.append((int(m.group(1)), int(m.group(2))))
    totals = {t for _, t in markers}
    assert len(totals) == 1, f"inconsistent diff totals: {sorted(totals)}"
    total = totals.pop()
    assert sorted(n for n, _ in markers) == list(range(1, total + 1)), sorted(markers)
    assert len(added) == total, f"{len(added)} changed lines but the markers claim {total}"


_SCIENTIFIC_FLAGS = (
    '--max-steps 67500 --batch-size "$MB" --accum-batches "$ACC" --num-workers 6 --seed 42',
    "--num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true",
    '--logger "$LOGGER" --checkpoint-every 2500',
    "--dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json",
    "--pretransform-ckpt-path weights/FLAC/VAE.safetensors",
)


def test_launcher_keeps_every_scientific_flag_byte_identical():
    launch, exp03n, exp09 = _text(_LAUNCH), _text(_EXP03N_LAUNCH), _text(_EXP09_LAUNCH)
    for flag_line in _SCIENTIFIC_FLAGS:
        assert flag_line in exp09, f"fixture drift: {flag_line!r} not in exp09_launch.sh"
        assert flag_line in exp03n, f"fixture drift: {flag_line!r} not in exp03n_launch.sh"
        assert flag_line in launch, f"scientific flag line changed: {flag_line!r}"
    assert '[ "$MB" = "32" ] && [ "$ACC" = "1" ]' in launch


def test_launcher_binds_the_exp06n_arm_not_exp03n_or_exp09():
    launch = _text(_LAUNCH)
    assert f"cd {_WORKTREE_ROOT_ABS}" in launch, "the launcher must cd to the absolute worktree root"
    assert f"export PYTHONPATH={_CYL_SRC_ABS}" in launch
    assert "FLAC_AR_exp06n.json" in launch
    assert "FLAC_AR_exp03n.json" not in launch and "FLAC_AR_exp09.json" not in launch
    assert "assert_arm_configs_exp06n.py" in launch
    assert "assert_arm_configs_exp03n.py" not in launch
    assert "assert_arm_configs_exp09.py" not in launch
    assert "--name FLAC_exp06n_maxpoollinear --experiment-name exp06n_maxpoollinear" in launch
    assert "--save-dir /n/fs/gatrdp/outputs/exp06n_maxpoollinear" in launch
    assert _RECORDS_DIR_ABS in launch, "the teed log must land in the exp_06 records folder"
    assert "maxpool_linear_cond_${TS}_j${SLURM_JOB_ID:-nojob}_train.log" in launch


def test_launcher_refuses_without_the_frozen_vram_file_and_the_two_pins():
    launch = _text(_LAUNCH)
    assert "exp06n_frozen_min_free.txt" in launch
    assert "exp03n_frozen_min_free.txt" not in launch, (
        "the exp_03 frozen VRAM value is another arm's measurement"
    )
    assert "c1_frozen_min_free.txt" not in launch
    assert 'FROZEN_FILE" ] || {' in launch, "the frozen-file REFUSE must survive"
    assert re.search(r'\[ -n "\$EXPECT_PACKAGE_SHA" \].*\[ -n "\$EXPECT_EXP06_SHA" \]', launch), (
        "the launcher must REFUSE when either pin is absent"
    )


def test_the_frozen_vram_file_is_probe_written_and_integer_when_present():
    """The file is created by exp06n_probe.sh ON the GPU node (plan B5) — absent until the
    probe has run; when present it must be a plain positive integer (the launcher binds
    the exact value)."""
    path = _EXP09_DIR / "exp06n_frozen_min_free.txt"
    if path.exists():
        value = path.read_text().strip()
        assert re.fullmatch(r"[0-9]+", value), f"non-integer frozen value: {value!r}"
        assert int(value) > 0


def test_launcher_wandb_identity_gate_uses_netrc_not_bashrc():
    launch = _text(_LAUNCH)
    assert ".bashrc" not in launch, "the ~/.bashrc WANDB_API_KEY grep must be dropped"
    assert "netrc" in launch
    assert "yh4742@princeton.edu" in launch, "the identity gate must still pin the account"


def test_probe_is_the_two_phase_lifecycle_probe_with_the_launcher_gates():
    probe = _text(_PROBE)
    assert "--max-steps 30" in probe and "--checkpoint-every 10" in probe
    assert "--max-steps 40" in probe and "--ckpt-path" in probe
    assert "step=30" in probe and "step=40" in probe
    assert re.search(r'SAVE="\$\{SAVE:-/n/fs/gatrdp/outputs/exp06n_maxpoollinear_PROBE\}"', probe), (
        "the probe must default to its own _PROBE save dir"
    )
    assert re.search(r'\[ ! -e "\$SAVE" \]', probe), "the probe must refuse an existing save dir"
    assert "assert_arm_configs_exp06n.py" in probe
    assert "FLAC_AR_exp06n.json" in probe
    assert f"export PYTHONPATH={_CYL_SRC_ABS}" in probe
    assert "EXPECT_PACKAGE_SHA" in probe and "EXPECT_EXP06_SHA" in probe
    assert "21900" in probe, "the provisional free-VRAM floor must be pinned"
    assert "1.15" in probe, "the probe must derive the 1.15x frozen value"
    assert ".bashrc" not in probe
    assert _RECORDS_DIR_ABS in probe, "probe logs/samples belong to the exp_06 records folder"
    assert "--batch-size 32" in probe and "--accum-batches 1" in probe
    assert "--sync-batchnorm true" in probe and "ddp_find_unused_parameters_true" in probe
    assert "--seed 42" in probe and "--num-gpus 2" in probe


# ---- Slurm wrappers (cylindrical repo) ---------------------------------------------- #
# Review B2/B3 add REAL functional lines to the wrappers, so pure byte-identity no longer
# holds; instead every diff vs the renamed exp03n wrapper must be accounted for: each
# ADDED line marked `EXP06 SBATCH DIFF n/N` (complete per-file numbering) and REMOVED
# lines allowed ONLY in the eval wrapper's artifact-copy region (the B3 rewrite).
_SBATCH_REMOVED_OK = {
    "train_exp06n.sbatch": None,   # pure additions only (the B2 env sweep)
    "probe_exp06n.sbatch": None,   # pure additions only (the B2 env sweep)
    # the artifact-copy region rewritten by B3/R2; the bare `fi` is the EMA block's closer
    # re-emitted by the differ (if/fi balance is backstopped by the bash -n parse test)
    "eval_exp06n.sbatch": re.compile(r"cp -v|sha256sum|\$ART|IMPORT_DIR|^fi$"),
}


@pytest.mark.parametrize("exp06n,exp03n", [("train_exp06n.sbatch", "train_exp03n.sbatch"),
                                           ("probe_exp06n.sbatch", "probe_exp03n.sbatch"),
                                           ("eval_exp06n.sbatch", "eval_exp03n.sbatch")])
def test_slurm_wrappers_are_renamed_exp03n_plus_marked_diffs_only(exp06n, exp03n):
    got = _sbatch(exp06n).splitlines()
    want = _rename_arm(_sbatch(exp03n)).splitlines()
    added, removed = [], []
    for line in difflib.unified_diff(want, got, n=0, lineterm=""):
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:])
    assert added, (
        f"{exp06n} is byte-identical to the renamed exp03n wrapper — the review B2/B3 "
        "hardening lines are missing"
    )
    markers = []
    for line in added:
        m = re.search(r"EXP06 SBATCH DIFF (\d+)/(\d+)", line)
        assert m, f"unmarked {exp06n} diff line: {line[:140]}"
        markers.append((int(m.group(1)), int(m.group(2))))
    totals = {t for _, t in markers}
    assert len(totals) == 1, f"inconsistent {exp06n} diff totals: {sorted(totals)}"
    total = totals.pop()
    assert sorted(n for n, _ in markers) == list(range(1, total + 1)), sorted(markers)
    assert len(added) == total, f"{len(added)} added lines but the markers claim {total}"
    removed_ok = _SBATCH_REMOVED_OK[exp06n]
    if removed_ok is None:
        assert removed == [], f"{exp06n} removed lines it must keep: {removed[:3]}"
    else:
        for line in removed:
            assert removed_ok.search(line), (
                f"{exp06n} removed a line OUTSIDE the declared copy region: {line[:140]}"
            )


# --- review B2: a stale submission export must not survive into the launcher/probe.
# The launcher honors C1_FROZEN_MIN_FREE_FILE (re-points the frozen-VRAM binding at ANY
# integer file), EXP06N_LOG_DIR (relocates the teed records log), MIN_FREE_MB (must equal
# frozen, but unset is strictly safer) and CKPT_PATH (a stale export would silently turn a
# fresh launch into a resume); the probe honors MIN_FREE_MB (would soften the provisional
# co-tenancy floor), EXP06N_LOG_DIR and SAVE. Each wrapper must unset them explicitly. --- #
def _unset_vars(text):
    names = set()
    for m in re.finditer(r"^\s*unset\s+(.+?)(?:\s*#.*)?$", text, flags=re.M):
        names.update(m.group(1).split())
    return names


def test_train_sbatch_unsets_every_inheritable_launcher_env():
    """r1 B2 + r2 R1: the direct overrides AND the bash re-entry vectors — a child bash
    re-sources $BASH_ENV (or $ENV under sh-mode), which could re-export the swept
    variables AFTER the sweep."""
    names = _unset_vars(_sbatch("train_exp06n.sbatch"))
    for var in ("C1_FROZEN_MIN_FREE_FILE", "EXP06N_LOG_DIR", "MIN_FREE_MB", "CKPT_PATH",
                "BASH_ENV", "ENV"):
        assert var in names, f"train_exp06n.sbatch must unset inheritable {var} (review B2/R1)"


def test_probe_sbatch_unsets_every_inheritable_probe_env():
    names = _unset_vars(_sbatch("probe_exp06n.sbatch"))
    for var in ("MIN_FREE_MB", "EXP06N_LOG_DIR", "SAVE", "BASH_ENV", "ENV"):
        assert var in names, f"probe_exp06n.sbatch must unset inheritable {var} (review B2/R1)"


def test_eval_sbatch_unsets_the_bash_reentry_envs():
    """The eval wrapper spawns no child bash script today, but the R1 sweep is uniform:
    a future child bash must not re-source a stale $BASH_ENV/$ENV."""
    names = _unset_vars(_sbatch("eval_exp06n.sbatch"))
    for var in ("BASH_ENV", "ENV"):
        assert var in names, f"eval_exp06n.sbatch must unset inheritable {var} (review r2 R1)"


@pytest.mark.parametrize("name,gpus", [("train_exp06n.sbatch", "2"), ("probe_exp06n.sbatch", "2")])
def test_gpu_wrappers_pin_and_assert_their_launch_shape(name, gpus):
    text = _sbatch(name)
    assert "#SBATCH --nodes=1" in text and "#SBATCH --ntasks=1" in text
    assert f"launch-shape: 1 node / {gpus}x L40 / 1 task" in text
    assert '[ "$NNODES" = "1" ]' in text and '[ "$NTASKS" = "1" ]' in text
    assert f'[ "$NGPU" = "{gpus}" ]' in text
    assert "EXPECT_EXP06_SHA" in text and "EXPECT_PACKAGE_SHA" in text


def test_train_sbatch_shape_and_preflights():
    text = _sbatch("train_exp06n.sbatch")
    assert "--gres=gpu:l40:2" in text
    assert "--cpus-per-task=16" in text and "--mem=32G" in text
    assert "--time=5-00:00:00" in text
    assert "exp06n_launch.sh" in text
    assert "100" in text, "the storage floor must be present"
    assert "/n/fs/gatrdp/outputs/exp06n_maxpoollinear" in text
    assert re.search(r"\[ ! -e /n/fs/gatrdp/outputs/exp06n_maxpoollinear \]", text), (
        "a fresh launch must refuse an existing save dir"
    )
    assert "HF_HUB_OFFLINE=1" in text
    assert _WORKTREE_ROOT_ABS in text


def test_probe_sbatch_shape():
    text = _sbatch("probe_exp06n.sbatch")
    assert "--gres=gpu:l40:2" in text
    assert "--time=00:50:00" in text
    assert "exp06n_probe.sh" in text


def test_eval_sbatch_passes_the_mandatory_fa_invariant_flags():
    """THE trap (handoff): eval_FLAC.py defaults to --cond-method vanilla and ignores the
    model config, so an eval that omits these flags silently scores the raw-pose path."""
    text = _sbatch("eval_exp06n.sbatch")
    assert "--cond-method fa_invariant" in text
    assert "--frame-avg-angles 0" in text
    assert "--cond-method vanilla" not in text


# --- per-stream DRY RUN: the wrapper renders the command it will run, and the test asserts
# the EXACT rendered argument vector (stream -> config is THE binding a swap must break). --- #
_STREAM_CONFIG = {
    "online": "worklog/worklog_yixun/exp_09_cyl_no_ssl/FLAC_AR_exp06n_online_eval.json",
    "ema": "worklog/worklog_yixun/exp_09_cyl_no_ssl/FLAC_AR_exp06n.json",
}
_EVAL_PROTOCOL_FLAGS = {
    "--cond-method": "fa_invariant",   # THE trap: never 'vanilla'
    "--frame-avg-angles": "0",
    "--steps": "1",
    "--cfg-scale": "1.0",
    "--batch-size": "64",
    "--num-workers": "4",
    "--device": "cuda",
    "--cond-autocast": "bf16",
}
_DS_FOR_K = {1: "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json",
             8: "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json"}


def _dry_run_eval(stream, *, step="40000", idx=0, cwd=None):
    path = _SLURM_DIR / "eval_exp06n.sbatch"
    if not _SLURM_DIR.exists():
        pytest.skip(f"cylindrical slurm_neuronic checkout not present at {_SLURM_DIR}")
    if not path.exists():
        pytest.fail(f"required deliverable is missing: {path}")
    env = dict(os.environ)
    env.update(DRY_RUN="1", STEP=step, STREAM=stream,
               EXPECT_PACKAGE_SHA="deadbeefdeadbeef", EXPECT_EXP06_SHA="cafebabecafebabe",
               SLURM_ARRAY_TASK_ID=str(idx))
    proc = subprocess.run(["bash", str(path)], capture_output=True, text=True, env=env,
                          cwd=str(cwd) if cwd else None)
    assert proc.returncode == 0, f"dry run failed (rc={proc.returncode}):\n{proc.stderr}"
    return proc.stdout


def _rendered(stdout, prefix):
    lines = [l for l in stdout.splitlines() if l.startswith(prefix)]
    assert len(lines) == 1, f"expected exactly one {prefix!r} line, got {lines}"
    return lines[0][len(prefix):].strip().split()


def _flag(tokens, flag):
    assert flag in tokens, f"{flag} missing from the rendered command: {' '.join(tokens)}"
    assert tokens.count(flag) == 1, f"{flag} appears {tokens.count(flag)}x"
    return tokens[tokens.index(flag) + 1]


@pytest.mark.parametrize("stream", ["online", "ema"])
@pytest.mark.parametrize("idx,k,seed", [(0, 1, 42), (4, 1, 46), (5, 8, 42), (9, 8, 46)])
def test_eval_dry_run_renders_the_exact_command_per_stream(stream, idx, k, seed, tmp_path):
    out = _dry_run_eval(stream, step="40000", idx=idx, cwd=tmp_path)
    tokens = _rendered(out, "EVAL_CMD:")
    assert tokens[:2] == ["python", "eval_FLAC.py"], tokens[:2]

    assert _flag(tokens, "--model-config") == _STREAM_CONFIG[stream], (
        f"stream {stream!r} rendered the WRONG model config: {_flag(tokens, '--model-config')}"
    )
    assert _flag(_rendered(out, "PIN_GATE:"), "--config") == _STREAM_CONFIG[stream]

    for flag, value in _EVAL_PROTOCOL_FLAGS.items():
        assert _flag(tokens, flag) == value, f"{flag} = {_flag(tokens, flag)!r}, expected {value!r}"
    assert "vanilla" not in tokens, "the raw-pose vanilla trap must never be rendered"

    assert _flag(tokens, "--dataset-config") == _DS_FOR_K[k]
    assert _flag(tokens, "--seed") == str(seed)
    assert _flag(tokens, "--eval-name") == f"exp06n_40000_{stream}_K{k}_s{seed}"
    assert _flag(tokens, "--ckpt-path").endswith("step=40000.ckpt")
    assert "exp06n_maxpoollinear" in _flag(tokens, "--ckpt-path"), (
        "the eval must resolve THIS arm's checkpoints"
    )
    assert list(tmp_path.iterdir()) == [], f"the dry run created files: {list(tmp_path.iterdir())}"


def test_eval_dry_run_streams_bind_different_configs(tmp_path):
    rendered = {s: _flag(_rendered(_dry_run_eval(s, cwd=tmp_path), "EVAL_CMD:"), "--model-config")
                for s in ("online", "ema")}
    assert rendered["online"] != rendered["ema"], rendered
    assert rendered == _STREAM_CONFIG, rendered
    assert "online_eval" in rendered["online"], "the online stream must load the use_ema:false config"
    assert not rendered["ema"].endswith("online_eval.json"), (
        "the EMA stream must load the training config (use_ema:true), not the online variant"
    )


def test_eval_dry_run_refuses_an_unknown_stream(tmp_path):
    path = _SLURM_DIR / "eval_exp06n.sbatch"
    if not _SLURM_DIR.exists():
        pytest.skip("cylindrical slurm_neuronic checkout not present")
    env = dict(os.environ)
    env.update(DRY_RUN="1", STEP="40000", STREAM="EMA", EXPECT_PACKAGE_SHA="x",
               EXPECT_EXP06_SHA="y", SLURM_ARRAY_TASK_ID="0")
    proc = subprocess.run(["bash", str(path)], capture_output=True, text=True, env=env,
                          cwd=str(tmp_path))
    assert proc.returncode != 0, "an unknown STREAM must be refused (fail-closed)"


def test_eval_dry_run_refuses_without_the_worktree_pin(tmp_path):
    path = _SLURM_DIR / "eval_exp06n.sbatch"
    if not _SLURM_DIR.exists():
        pytest.skip("cylindrical slurm_neuronic checkout not present")
    env = {k: v for k, v in os.environ.items()
           if k not in ("EXPECT_EXP06_SHA", "EXPECT_EXP09_SHA")}
    env.update(DRY_RUN="1", STEP="40000", STREAM="online", EXPECT_PACKAGE_SHA="x",
               SLURM_ARRAY_TASK_ID="0")
    proc = subprocess.run(["bash", str(path)], capture_output=True, text=True, env=env,
                          cwd=str(tmp_path))
    assert proc.returncode != 0, "a missing EXPECT_EXP06_SHA must be refused (fail-closed)"


def test_eval_sbatch_array_and_launch_shape_directives():
    text = _sbatch("eval_exp06n.sbatch")
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks=1" in text
    assert "#SBATCH --gres=gpu:l40:1" in text
    assert "--array=0-9" in text, "K{1,8} x seeds 42-46 = 10 cells"
    assert "exp06n_${STEP}_${STREAM}_K${K}_s${SEED}" in text, "stream-tagged, collision-free eval name"
    assert "assert_arm_configs_exp06n.py" in text, "each cell must run the pin gate"


def test_eval_sbatch_asserts_its_launch_shape_and_evaluation_only_status():
    text = _sbatch("eval_exp06n.sbatch")
    assert "launch-shape: 1 node / 1x L40 / 1 task / evaluation-only" in text
    assert 'WT_SHA=$(git -C "$WORKTREE" rev-parse HEAD)' in text
    assert 'PKG_SHA=$(git -C "$CYL_REPO" rev-parse HEAD)' in text
    assert '[ "$WT_SHA" = "$EXPECT_EXP06_SHA" ]' in text
    assert '[ "$PKG_SHA" = "$EXPECT_PACKAGE_SHA" ]' in text
    assert '[ "$NNODES" = "1" ]' in text and '[ "$NTASKS" = "1" ]' in text
    assert '[ "$NGPU" = "1" ]' in text
    assert "module.eval().requires_grad_(False)" in text
    assert "with torch.no_grad():" in text
    assert "EVAL-ONLY CONTRACT VIOLATED" in text
    assert 'grep -q "Found 6337 files"' in text, "the full-split criterion must be asserted post-run"
    assert 'for MARK in "optim" "backward" "training_step" "Epoch "' in text


def test_eval_sbatch_proves_the_weight_stream_it_used():
    text = _sbatch("eval_exp06n.sbatch")
    assert "diffusion_ema.ema_model." in text, "EMA cells must PRE-assert the ckpt carries EMA keys"
    assert "Using EMA model" in text, "the log line must be asserted post-run (present/absent)"


def test_eval_sbatch_retention_and_overwrite_guard():
    text = _sbatch("eval_exp06n.sbatch")
    assert "sha256sum" in text
    assert _RECORDS_DIR_ABS in text, "artifacts must be retained in the exp_06 records folder"
    assert f"WORKTREE={_WORKTREE_ROOT_ABS}" in text
    assert 'IMPORT_DIR="${WORKTREE}/outputs_FLAC/exp06n_maxpoollinear_import"' in text, (
        "EMA raws must ALSO land in the repo-rooted import dir the canonical generator globs"
    )
    assert re.search(r'if \[ "\$STREAM" = "ema" \];[\s\S]*IMPORT_DIR', text), (
        "the import-dir copy must be EMA-only"
    )
    assert 'reserve "$ART"' in text, "an existing/reserved artifact must abort the cell"


# --- review B3 + r2 R2: plain absence checks race — two identical cells can both pass a
# preflight, eval_FLAC.py opens ART with "w", and an ordinary cp has a check-then-use
# window. ALL THREE destinations are ATOMICALLY reserved (noclobber sentinel) right after
# the names are derived, and retention publishes via cp-to-tmp + mv -n + byte-verify. --- #
def test_eval_sbatch_reserves_all_three_destinations_atomically_before_compute():
    text = _sbatch("eval_exp06n.sbatch")
    assert re.search(r'\(\s*set -C;\s*: > "\$1\.reserve"\s*\)', text), (
        "the reservation must be an ATOMIC noclobber create (set -C), not an absence check"
    )
    assert "trap cleanup_reserves EXIT" in text
    art_reserve = text.index('reserve "$ART"')
    records_reserve = text.index('reserve "${RECORDS}/${ART_BASE}"')
    import_reserve = text.index('reserve "${IMPORT_DIR}/${ART_BASE}"')
    pin_gate = text.index("pin + arm-wiring gate, bound to the ACTUAL config")
    eval_run = text.index("--- eval (mandatory")
    for name, pos in (("ckpt-dir", art_reserve), ("records", records_reserve),
                      ("import", import_reserve)):
        assert pos < pin_gate < eval_run, (
            f"the {name} reservation must be taken BEFORE the pin gate and the eval "
            "(reserving after compute wastes the cell and leaves the clobber window open)"
        )
    # the reservation refuses BOTH a pre-existing final artifact and a held sentinel
    assert re.search(r'reserve\(\)\s*\{[\s\S]*?\[ ! -e "\$1" \]', text), (
        "reserve() must refuse a pre-existing final artifact"
    )
    assert "already held" in text, "a held sentinel (concurrent identical cell) must refuse"


def test_eval_sbatch_publishes_via_staged_atomic_rename_and_verifies():
    text = _sbatch("eval_exp06n.sbatch")
    # the old unconditional-overwrite forms must be gone
    assert 'cp -v "$ART" "$RECORDS/"' not in text, "records copy still clobbers (review B3)"
    assert 'cp -v "$ART" "$IMPORT_DIR/"' not in text, "import copy still clobbers (review B3)"
    assert 'cp -v "$ART"' not in text, "no direct cp of ART may remain — publish() only"
    # publish(): stage to a tmp name, atomic no-clobber rename, detect the losing race,
    # then byte-verify the published copy
    assert re.search(r'local tmp="\$2\.tmp\.\$\$"', text), "publish must stage to a tmp name"
    assert re.search(r'mv -n "\$tmp" "\$2"', text), "the rename must be no-clobber (mv -n)"
    assert re.search(r'\[ ! -e "\$tmp" \]', text), (
        "a surviving tmp file (mv -n lost the race) must abort, not be ignored"
    )
    assert re.search(r'cmp -s "\$1" "\$2"', text), "the published copy must be byte-verified"
    assert 'publish "$ART" "${RECORDS}/${ART_BASE}"' in text
    assert 'publish "$ART" "${IMPORT_DIR}/${ART_BASE}"' in text


def test_eval_sbatch_sentinels_cleared_only_after_verified_publishes():
    text = _sbatch("eval_exp06n.sbatch")
    assert "PUBLISH_OK=0" in text, "the success flag must start false"
    ok_pos = text.rindex("PUBLISH_OK=1")
    assert ok_pos > text.index('publish "$ART" "${RECORDS}/${ART_BASE}"'), (
        "PUBLISH_OK may only be set after the verified publishes"
    )
    assert ok_pos > text.index('publish "$ART" "${IMPORT_DIR}/${ART_BASE}"')
    assert re.search(r'cleanup_reserves\(\)\s*\{[\s\S]*?PUBLISH_OK[\s\S]*?rm -f', text), (
        "the EXIT trap must clear sentinels only when PUBLISH_OK is set"
    )
    assert "KEPT" in text, "on failure the sentinels must be kept (and said so) for inspection"


def test_eval_sbatch_reserve_and_publish_functions_behave(tmp_path):
    """FUNCTIONAL: extract reserve()/publish() from the wrapper and exercise them in a
    bash harness — a second reservation must refuse, publishing onto a taken destination
    must refuse WITHOUT touching it, and a clean publish must byte-match."""
    text = _sbatch("eval_exp06n.sbatch")

    def _fn(name):
        m = re.search(rf"^{name}\(\) \{{[\s\S]*?^\}}", text, flags=re.M)
        assert m, f"function {name}() not found in eval_exp06n.sbatch"
        return m.group(0)

    harness = "\n".join([
        "set -uo pipefail", "RESERVES=()", "PUBLISH_OK=0",
        _fn("reserve"), _fn("publish"),
        'cd "$1"',
        'echo payload > src.json',
        '( reserve out.json ) && echo "R1=OK" || echo "R1=REFUSED"',
        '( reserve out.json ) && echo "R2=OK" || echo "R2=REFUSED"',
        'echo squatter > taken.json',
        '( reserve taken.json ) && echo "R3=OK" || echo "R3=REFUSED"',
        '( publish src.json out.json ) && echo "P1=OK" || echo "P1=REFUSED"',
        'echo other > held.json',
        '( publish src.json held.json ) && echo "P2=OK" || echo "P2=REFUSED"',
    ])
    script = tmp_path / "harness.sh"
    script.write_text(harness + "\n")
    proc = subprocess.run(["bash", str(script), str(tmp_path)],
                          capture_output=True, text=True)
    out = proc.stdout
    assert "R1=OK" in out, f"first reservation must succeed:\n{out}\n{proc.stderr}"
    assert "R2=REFUSED" in out, f"second reservation must refuse (atomicity):\n{out}"
    assert "R3=REFUSED" in out, f"reserving over an existing final artifact must refuse:\n{out}"
    assert "P1=OK" in out, f"publishing to a reserved-free destination must succeed:\n{out}"
    assert (tmp_path / "out.json").read_text() == "payload\n"
    assert "P2=REFUSED" in out, f"publishing onto a taken destination must refuse:\n{out}"
    assert (tmp_path / "held.json").read_text() == "other\n", (
        "the losing publish must leave the existing destination untouched"
    )
    assert not list(tmp_path.glob("*.tmp.*")), "no staging tmp files may survive"


def test_eval_sbatch_appends_the_import_sha_manifest():
    """B4/B5: every EMA artifact copied into the import dir appends its sha256 line
    (repo-relative name, the established IMPORT_SHA256SUMS format) to the manifest."""
    text = _sbatch("eval_exp06n.sbatch")
    assert "IMPORT_SHA256SUMS.txt" in text
    assert re.search(r'tee -a "\$\{IMPORT_DIR\}/IMPORT_SHA256SUMS\.txt"', text), (
        "the sha256 line must be APPENDED to the import manifest, not just printed"
    )
    assert re.search(r'cd "\$WORKTREE" && sha256sum "outputs_FLAC/exp06n_maxpoollinear_import/', text), (
        "manifest lines must carry repo-relative names (sha256sum -c from the repo root)"
    )


# ------------------------------------------------------------------------------------ #
# 7. provenance deliverables (review B4): generator pending rows + import scaffolds
# ------------------------------------------------------------------------------------ #
_GEN_PATH = Path("/n/fs/gatrdp/codespace/cylindrical-dinov3/worklog/worklog_yixun_neuronic/"
                 "gen_model_comparison_neuronic.py")
_FLAC_MAIN_MANIFEST = Path("/n/fs/gatrdp/codespace/FLAC/outputs_FLAC/"
                           "exp06n_maxpoollinear_import/IMPORT_SHA256SUMS.txt")
_WORKTREE_MANIFEST = _REPO_ROOT / "outputs_FLAC" / "exp06n_maxpoollinear_import" / "IMPORT_SHA256SUMS.txt"


@pytest.fixture(scope="module")
def gen():
    if not _GEN_PATH.parent.exists():
        pytest.skip(f"cylindrical worklog checkout not present at {_GEN_PATH.parent}")
    return _load_module(_GEN_PATH, "gen_model_comparison_neuronic")


def test_generator_registers_the_full_exp06_pending_row_matrix(gen):
    """@40k/@67.5k x K{1,8} x ONLINE/EMA = 8 row specs, globbing the exp06n eval-name
    family over the exp_06 records folder, each carrying the r2-R3 provenance VALIDATOR
    as a 5th element (mirrors the exp03n/exp04n 4-tuple shape otherwise)."""
    exp06_rows = [r for r in gen.ROWS if "exp_06" in r[0]]
    assert len(exp06_rows) == 8, f"expected 8 exp_06 row specs, got {len(exp06_rows)}"
    seen = set()
    for row in exp06_rows:
        assert len(row) == 5, (
            f"exp_06 row specs must carry the provenance validator (r2 R3): {row[0]}"
        )
        label, proto, k, patterns, validate = row
        assert callable(validate), f"row validator is not callable: {label}"
        assert "max_linear" in label, label
        assert proto.startswith("fa eval, bf16, "), proto
        stream = "online" if "ONLINE" in proto else "ema"
        assert len(patterns) == 1, patterns
        pattern = patterns[0]
        assert "exp_06_maxpool_linear_cond_claude" in pattern, (
            f"row glob must scan the exp_06 records folder: {pattern}"
        )
        step = "40000" if "40000" in pattern else "67500"
        assert f"exp06n_{step}_{stream}_K{k}_s" in pattern, (
            f"row glob must match the eval-name family exp06n_<step>_<stream>_K<k>_s<seed>: "
            f"{pattern}"
        )
        assert ("@40k" in label) == (step == "40000"), (label, step)
        assert ("@67.5k" in label) == (step == "67500"), (label, step)
        seen.add((step, k, stream))
    assert seen == {(s, k, t) for s in ("40000", "67500") for k in (1, 8)
                    for t in ("online", "ema")}, sorted(seen)


def test_generator_renders_the_exp06_rows_as_pending_until_raws_exist(gen):
    """No exp06n raws are on disk yet, so every exp_06 row must render as pending —
    exactly the canonical-table convention for not-yet-measured cells."""
    exp06_rows = [r for r in gen.ROWS if "exp_06" in r[0]]
    body = gen.render_body(exp06_rows)
    data_lines = [l for l in body.splitlines() if l.startswith("| cyl no-SSL max_linear")]
    assert len(data_lines) == 8, body
    for line in data_lines:
        assert "*pending (" in line, f"exp_06 row did not render as pending: {line}"


def test_generator_exp03_exp04_row_specs_are_untouched(gen):
    """The exp06 append must not disturb the existing registered rows — same count, same
    4-tuple shape, NO validator retrofitted (r2 R3 scopes validation to the NEW rows;
    generalizing to older rows is deliberate follow-up, not this diff)."""
    for token, count in (("exp_03", 8), ("exp_04", 8)):
        rows = [r for r in gen.ROWS if token in r[0]]
        assert len(rows) == count
        assert all(len(r) == 4 for r in rows), f"{token} rows must stay 4-tuples"
    p1_rows = [r for r in gen.ROWS if r[0].startswith("P1 ")]
    assert len(p1_rows) == 4 and all(len(r) == 4 for r in p1_rows)


# --- r2 R3 functional: the validator must reject wrong-protocol / wrong-arm / bad-seed
# raws as HARD errors naming the file, accept a clean 5-seed row, and let short rows
# stay pending. Exercised on synthetic raws in tmp dirs, via the REAL aggregate path. --- #
_VALID_METRICS = {"T60": 1.0, "Invalid T60": 0.0, "C50": 0.5, "EDT": 2.0, "FD": 0.1,
                  "RIR_to_GT_RIR_R@1": 10.0, "RIR_to_GT_RIR_R@5": 20.0,
                  "RIR_to_GT_RIR_R@10": 30.0}


def _write_raw(dirpath, seed, *, step="40000", stream="online", k=1, epoch=8, **overrides):
    record = {
        "metrics": dict(_VALID_METRICS),
        "ckpt_path": (f"/n/fs/gatrdp/outputs/exp06n_maxpoollinear/FLAC_exp06n_maxpoollinear/"
                      f"exp06n_maxpoollinear/checkpoints/epoch={epoch}-step={step}.ckpt"),
        "rotate_deg": 0.0,
        "cond_method": "fa_invariant",
        "frame_avg_angles": [0.0],
        "cond_autocast": "bf16",
    }
    record.update(overrides)
    name = (f"epoch={epoch}-step={step}_metrics_1_1.0_"
            f"exp06n_{step}_{stream}_K{k}_s{seed}_fa_invariant_a1.json")
    path = Path(dirpath) / name
    path.write_text(json.dumps(record) + "\n")
    return path


def _exp06_row(gen, dirpath, *, step="40000", stream="online", k=1):
    return ("cyl no-SSL max_linear @40k (exp_06)", f"fa eval, bf16, {stream.upper()}", k,
            [str(Path(dirpath) / f"*exp06n_{step}_{stream}_K{k}_s*.json")],
            gen.make_exp06_validator(step))


def test_generator_validator_accepts_a_clean_five_seed_row(gen, tmp_path):
    for seed in range(42, 47):
        _write_raw(tmp_path, seed)
    body = gen.render_body([_exp06_row(gen, tmp_path)])
    line = [l for l in body.splitlines() if l.startswith("| cyl no-SSL max_linear")][0]
    assert "pending" not in line, line
    assert "| 5 |" in line, line


def test_generator_validator_keeps_a_short_row_pending_without_error(gen, tmp_path):
    for seed in (42, 43, 44):
        _write_raw(tmp_path, seed)
    body = gen.render_body([_exp06_row(gen, tmp_path)])
    line = [l for l in body.splitlines() if l.startswith("| cyl no-SSL max_linear")][0]
    assert "*pending (3/5" in line, line


@pytest.mark.parametrize("overrides, match", [
    ({"cond_method": "vanilla"}, "cond_method"),
    ({"cond_autocast": "default"}, "cond_autocast"),
    ({"rotate_deg": 90.0}, "rotate_deg"),
    ({"frame_avg_angles": [0.0, 90.0, 180.0, 270.0]}, "frame_avg_angles"),
    # a WRONG-ARM raw renamed into the exp06 family: recorded ckpt_path betrays it
    ({"ckpt_path": "/n/fs/gatrdp/outputs/exp03n_maxpoolmlp/FLAC_exp03n_maxpoolmlp/"
                   "exp03n_maxpoolmlp/checkpoints/epoch=8-step=40000.ckpt"}, "save dir|ckpt_path"),
    # right arm, WRONG STEP checkpoint under a 40000-named eval
    ({"ckpt_path": "/n/fs/gatrdp/outputs/exp06n_maxpoollinear/FLAC_exp06n_maxpoollinear/"
                   "exp06n_maxpoollinear/checkpoints/epoch=14-step=67500.ckpt"}, "step"),
])
def test_generator_validator_hard_errors_on_wrong_provenance(gen, tmp_path, overrides, match):
    for seed in (42, 43, 44, 45):
        _write_raw(tmp_path, seed)
    bad = _write_raw(tmp_path, 46, **overrides)
    with pytest.raises(RuntimeError, match=match) as excinfo:
        gen.render_body([_exp06_row(gen, tmp_path)])
    assert bad.name in str(excinfo.value), (
        "the hard error must NAME the violating file (never a silent skip)"
    )


def test_generator_validator_hard_errors_on_seed_violations(gen, tmp_path):
    # duplicate seed (5 files, distinct names via epoch, seeds 42,42,43,44,45)
    for seed, epoch in ((42, 8), (42, 9), (43, 8), (44, 8), (45, 8)):
        _write_raw(tmp_path, seed, epoch=epoch)
    with pytest.raises(RuntimeError, match="seed"):
        gen.render_body([_exp06_row(gen, tmp_path)])

    out_of_range = tmp_path / "range"
    out_of_range.mkdir()
    for seed in (42, 43, 44, 45, 47):
        _write_raw(out_of_range, seed)
    with pytest.raises(RuntimeError, match="seed|42"):
        gen.render_body([_exp06_row(gen, out_of_range)])

    leak = tmp_path / "leak"
    leak.mkdir()
    for seed in range(42, 47):
        _write_raw(leak, seed)
    _write_raw(leak, 46, epoch=9)   # a 6th matching file
    with pytest.raises(RuntimeError, match="more than|leak|seed"):
        gen.render_body([_exp06_row(gen, leak)])


def _assert_manifest_scaffold(path: Path):
    text = path.read_text()
    lines = text.splitlines()
    assert lines and lines[0].startswith("#"), (
        f"{path} must open with the contract header (comment lines; sha256sum -c ignores them)"
    )
    header = [l for l in lines if l.startswith("#")]
    assert any("append" in l.lower() for l in header), (
        "the header must document the APPENDING contract (eval wrapper appends sha256 lines)"
    )
    assert any("sha256" in l.lower() for l in header)
    for line in lines:
        if line and not line.startswith("#"):
            assert re.fullmatch(r"[0-9a-f]{64}  \S.*", line), (
                f"non-header manifest line is not a sha256sum line: {line[:120]}"
            )


def test_worktree_import_scaffold_manifest_exists_with_the_contract_header():
    """The dir the eval wrapper appends into (repo-rooted in THIS worktree, force-added:
    outputs_FLAC/ is ignored — the 71e77a2 lesson)."""
    if not _WORKTREE_MANIFEST.exists():
        pytest.fail(f"required deliverable is missing: {_WORKTREE_MANIFEST}")
    _assert_manifest_scaffold(_WORKTREE_MANIFEST)


def test_flac_main_import_scaffold_manifest_exists_with_the_contract_header():
    """The canonical-table-side landing zone in the main FLAC checkout (committed there
    by the Planner; created by this round)."""
    if not _FLAC_MAIN_MANIFEST.parent.parent.exists():
        pytest.skip(f"FLAC main checkout not present at {_FLAC_MAIN_MANIFEST.parent.parent}")
    if not _FLAC_MAIN_MANIFEST.exists():
        pytest.fail(f"required deliverable is missing: {_FLAC_MAIN_MANIFEST}")
    _assert_manifest_scaffold(_FLAC_MAIN_MANIFEST)
