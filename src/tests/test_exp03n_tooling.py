"""exp_03 (worklog_yixun_neuronic) — tests for the exp03n TOOLING: the pre-launch
arm-wiring gate, the trained-checkpoint equivariance guard, and the launch/probe/eval
script contracts.

Companion to ``test_exp03n_cond_pool.py`` (which pins the model delta itself). Everything
here is CPU-only and offline; the gate/guard helpers are exercised on TINY random-weight
cylindrical backbones, never on the Hub.
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import copy  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_EXP09_DIR = _REPO_ROOT / "worklog" / "worklog_yixun" / "exp_09_cyl_no_ssl"
_GATE_PATH = _EXP09_DIR / "assert_arm_configs_exp03n.py"

from src.tests.test_exp03n_cond_pool import (  # noqa: E402
    _HIDDEN,
    _build,
    _cyl_conditioning,
    _geoms,
    _save_tiny_cyl,
)


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
    return _load_module(_GATE_PATH, "assert_arm_configs_exp03n")


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    return _save_tiny_cyl(tmp_path_factory.mktemp("exp03n_tooling_tiny"))


# ------------------------------------------------------------------------------------ #
# 1. gate: per-variant config contract
# ------------------------------------------------------------------------------------ #
def test_gate_accepts_both_real_config_variants(gate):
    for name, variant in (("FLAC_AR_exp03n.json", "base"),
                          ("FLAC_AR_exp03n_online_eval.json", "online")):
        cfg, resolved = gate.assert_config_contract(str(_EXP09_DIR / name))
        assert resolved == variant, (name, resolved)
        assert cfg["model"]["conditioning"]["configs"][1]["config"]["ViT"]["cond_pool"] == "max_mlp"


def test_gate_variant_autodetect_is_filename_driven(gate):
    assert gate.variant_for_config("/x/FLAC_AR_exp03n_online_eval.json") == "online"
    assert gate.variant_for_config("/x/FLAC_AR_exp03n.json") == "base"


def test_gate_rejects_the_wrong_reference_for_a_variant(gate, tmp_path):
    """Binding the ONLINE file to the BASE delta (or vice versa) must fail: the two
    references differ in use_ema and the gradient_checkpointing keys."""
    with pytest.raises(RuntimeError, match="contract|reconstruct"):
        gate.assert_config_contract(str(_EXP09_DIR / "FLAC_AR_exp03n_online_eval.json"),
                                    variant="base")
    with pytest.raises(RuntimeError, match="contract|reconstruct"):
        gate.assert_config_contract(str(_EXP09_DIR / "FLAC_AR_exp03n.json"), variant="online")


@pytest.mark.parametrize("mutation", ["extra_key", "changed_recipe", "dropped_key",
                                      "wrong_width", "one_block_only"])
def test_gate_rejects_mutated_configs(gate, tmp_path, mutation):
    cfg = json.load(open(_EXP09_DIR / "FLAC_AR_exp03n.json"))
    vits = [c for c in cfg["model"]["conditioning"]["configs"] if c["type"] == "ViTCoordinates"]
    if mutation == "extra_key":
        vits[0]["config"]["ViT"]["cond_mlp_dropout"] = 0.1
        vits[1]["config"]["ViT"]["cond_mlp_dropout"] = 0.1
    elif mutation == "changed_recipe":
        cfg["training"]["frame_avg_angles"] = [0.0, 90.0, 180.0, 270.0]
    elif mutation == "dropped_key":
        del vits[1]["config"]["ViT"]["cond_pool"]
    elif mutation == "wrong_width":
        vits[1]["config"]["ViT"]["cond_mlp_hidden"] = 512
    elif mutation == "one_block_only":
        del vits[1]["config"]["ViT"]["cond_pool"]
        del vits[1]["config"]["ViT"]["cond_mlp_hidden"]
    path = tmp_path / "FLAC_AR_exp03n.json"
    path.write_text(json.dumps(cfg, indent=4) + "\n")
    with pytest.raises(RuntimeError):
        gate.assert_config_contract(str(path))


# ------------------------------------------------------------------------------------ #
# 2. gate: instantiated-model arm wiring
# ------------------------------------------------------------------------------------ #
def test_gate_head_wiring_accepts_a_correct_max_mlp_build(gate, tiny_dir):
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="max_mlp"))
    head = gate.assert_head_wiring(mc, hidden_size=_HIDDEN, cond_dim=256)
    assert head is _geoms(mc)[0].lin_proj


def test_gate_head_wiring_rejects_the_legacy_build(gate, tiny_dir):
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True))
    with pytest.raises(RuntimeError):
        gate.assert_head_wiring(mc, hidden_size=_HIDDEN, cond_dim=256)


def test_gate_head_wiring_rejects_an_unshared_head(gate, tiny_dir):
    """A second, independently-built head would double the new parameters and break the
    shared-head contract — the gate must notice even though every shape still matches."""
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="max_mlp"))
    geoms = _geoms(mc)
    geoms[1].lin_proj = copy.deepcopy(geoms[0].lin_proj)
    with pytest.raises(RuntimeError, match="shared"):
        gate.assert_head_wiring(mc, hidden_size=_HIDDEN, cond_dim=256)


def test_gate_head_wiring_rejects_a_mean_pooled_conditioner(gate, tiny_dir):
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="max_mlp"))
    _geoms(mc)[1].dino_pool = "mean"
    with pytest.raises(RuntimeError, match="dino_pool"):
        gate.assert_head_wiring(mc, hidden_size=_HIDDEN, cond_dim=256)


def test_gate_param_delta_is_the_hidden_layer_only(gate, tiny_dir):
    legacy = _build(_cyl_conditioning(tiny_dir, with_context=True))
    new = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="max_mlp"))
    gate.assert_param_delta(legacy, new, expected=_HIDDEN * _HIDDEN + _HIDDEN)
    with pytest.raises(RuntimeError, match="trainable"):
        gate.assert_param_delta(legacy, new, expected=147840)


def test_gate_live_forward_matches_the_external_oracle(gate, tiny_dir):
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="max_mlp"))
    gate.assert_live_forward(mc, height=64, width=128)


def test_gate_live_forward_catches_a_mean_pooling_regression(gate, tiny_dir):
    """Sabotage: flip the served pooling back to the mean while the head stays the MLP.
    Shapes and parameter counts are unchanged, so ONLY the live forward can catch it."""
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="max_mlp"))
    for geom in _geoms(mc):
        geom.dino_pool = "mean"
    with pytest.raises(RuntimeError, match="amax|forward"):
        gate.assert_live_forward(mc, height=64, width=128)


def test_gate_common_init_identity_on_tiny_full_models(gate, tiny_dir):
    from src.tests.test_exp03n_cond_pool import _full_model_config

    new_cfg = _full_model_config(tiny_dir, max_mlp=True)
    legacy_cfg = _full_model_config(tiny_dir, max_mlp=False)
    gate.assert_common_init_identity(new_cfg, legacy_cfg, seed=42)


def test_gate_common_init_identity_catches_a_perturbed_output_layer(gate, tiny_dir, monkeypatch):
    """Sabotage: draw the hidden layer from the GLOBAL stream (no fork_rng). Every module
    built AFTER the conditioner then initialises differently — exactly the one-factor
    violation this assertion exists to catch."""
    from src.tests.test_exp03n_cond_pool import _full_model_config

    import src.models.conditioners as cond_mod

    def _leaky_fork_rng(*args, **kwargs):
        import contextlib

        return contextlib.nullcontext()

    monkeypatch.setattr(cond_mod.torch.random, "fork_rng", _leaky_fork_rng)
    new_cfg = _full_model_config(tiny_dir, max_mlp=True)
    legacy_cfg = _full_model_config(tiny_dir, max_mlp=False)
    with pytest.raises(RuntimeError):
        gate.assert_common_init_identity(new_cfg, legacy_cfg, seed=42)


# ------------------------------------------------------------------------------------ #
# 3. trained-checkpoint equivariance guard (plan §4.5b)
# ------------------------------------------------------------------------------------ #
_GUARD_PATH = _EXP09_DIR / "guard_exp03n_equivariance.py"

_TINY_GEOM = dict(height=64, width=128)   # 4x8 tokens at patch 16 -> W_t = 8


@pytest.fixture(scope="module")
def guard():
    return _load_module(_GUARD_PATH, "guard_exp03n_equivariance")


def test_guard_passes_on_a_random_weight_mini_model(guard, tiny_dir):
    """The guard tests the ACTUAL served condition MLP(amax(tokens)) — architectural, so it
    must hold at random init, not only on trained weights."""
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="max_mlp"))
    residuals = guard.check_equivariance(_geoms(mc)[0], **_TINY_GEOM)
    assert set(residuals) == {90.0, 180.0, 270.0}
    assert max(residuals.values()) <= guard.DEFAULT_BOUND, residuals


def test_guard_negative_control_gauge_off_raises(guard, tiny_dir, tmp_path):
    gauge_off = _save_tiny_cyl(tmp_path / "tiny_gauge_off")
    mc = _build(_cyl_conditioning(gauge_off, with_context=True, cond_pool="max_mlp", gauge="none"))
    with pytest.raises(RuntimeError, match="equivarian|residual"):
        guard.check_equivariance(_geoms(mc)[0], **_TINY_GEOM)


def test_guard_refuses_an_angle_that_is_not_a_whole_token_column(guard, tiny_dir):
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="max_mlp"))
    with pytest.raises(RuntimeError, match="column"):
        guard.check_equivariance(_geoms(mc)[0], angles=(30.0,), **_TINY_GEOM)


def test_guard_refuses_a_mean_pooled_conditioner(guard, tiny_dir):
    """Scope guard: this file certifies the max-pool head. Pointing it at a legacy build
    must REFUSE rather than silently certify the wrong served condition."""
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True))
    with pytest.raises(RuntimeError, match="dino_pool|max"):
        guard.check_equivariance(_geoms(mc)[0], **_TINY_GEOM)


def _write_pl_checkpoint(tmp_path, tiny_dir):
    """A Lightning-shaped checkpoint of a TINY exp03n model (``diffusion.``-prefixed
    state dict), plus the model-config JSON that reproduces it."""
    from src.models import create_model_from_config
    from src.tests.test_exp03n_cond_pool import _full_model_config

    cfg = _full_model_config(tiny_dir, max_mlp=True)
    cfg_path = tmp_path / "FLAC_AR_exp03n.json"
    cfg_path.write_text(json.dumps(cfg, indent=4) + "\n")
    torch.manual_seed(3)
    model = create_model_from_config(copy.deepcopy(cfg))
    ckpt_path = tmp_path / "epoch=0-step=40.ckpt"
    torch.save({"state_dict": {f"diffusion.{k}": v for k, v in model.state_dict().items()}},
               ckpt_path)
    return str(ckpt_path), str(cfg_path), model


def test_guard_loads_a_lightning_checkpoint_and_serves_the_max_pool_head(guard, tiny_dir, tmp_path):
    ckpt_path, cfg_path, model = _write_pl_checkpoint(tmp_path, tiny_dir)
    geoms = guard.load_geometry_conditioners(ckpt_path, cfg_path, device="cpu")
    assert len(geoms) == 2
    assert all(g.dino_pool == "max" for g in geoms)
    ref = model.conditioner.conditioners["source_vit"].lin_proj
    assert torch.equal(geoms[0].lin_proj[0].weight, ref[0].weight), "hidden layer did not load"
    assert torch.equal(geoms[0].lin_proj[2].weight, ref[2].weight), "output layer did not load"


def test_guard_cli_returns_zero_on_a_loadable_checkpoint(guard, tiny_dir, tmp_path, capsys):
    ckpt_path, cfg_path, _ = _write_pl_checkpoint(tmp_path, tiny_dir)
    rc = guard.main([ckpt_path, "--model-config", cfg_path, "--height", "64", "--width", "128"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "GUARD PASS" in out, out
