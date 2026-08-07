"""exp_04 (worklog_yixun_neuronic) — MEAN-pool + the SAME MLP conditioning head.

The exp_04 delta (delta plan §1/§1b) is ONE new registered value of the exp_03 knob:

* ``"cond_pool": "mean_mlp"`` keeps the LEGACY pooling (``pooler_output``, i.e. the patch
  mean — byte-identical pooling path to the mean+Linear arm) and attaches the IDENTICAL
  MLP head ``Linear(H, H) → GELU → Linear(H, cond_dim)`` that ``"max_mlp"`` builds;
* the head construction is the SHARED code path (one ``fork_rng``-isolated hidden layer at
  the pinned CPU seed, one global-RNG output layer at the legacy code point), so a
  same-seed ``mean_mlp`` build and ``max_mlp`` build are bitwise identical — the pooling
  SELECTOR is the sole difference between exp_03 and exp_04 (delta 1b, the factor-isolation
  proof this experiment rests on);
* ``"max_mlp"`` is unchanged and genuinely unknown values (``"avg_mlp"``, …) still
  ValueError.

Fixtures, tiny backbone and geometry are imported from ``test_exp03n_cond_pool`` so both
arms are measured with the SAME harness (64x128 px at patch 16 → a 4x8 token grid, W_t = 8;
the registered angles {90, 180, 270} are exactly 2, 4 and 6 whole token columns).

CPU-only and offline; one test binds the real 384-d official snapshot when it is in the
local HF cache (skips otherwise) for the production head shape and +147,840 oracle.
"""
import os
import sys

# CPU-only + offline, hard-set before torch initialises CUDA (mandate).
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import copy  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]  # src/tests -> src -> repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cylindrical_dinov3 import physical_yaw  # noqa: E402

from src.models.conditioners import (  # noqa: E402
    create_multi_conditioner_from_conditioning_config,
)

# The exp_03 harness, reused VERBATIM: same tiny backbone, same fixture geometry, same
# non-axisymmetric scene and the same A2b statistic — the two arms must be comparable.
from src.tests.test_exp03n_cond_pool import (  # noqa: E402
    _BOUND,
    _BROKEN,
    _COND_DIM,
    _EXP09_DIR,
    _EXP09_JSON,
    _HIDDEN,
    _H_T,
    _OFFICIAL_HIDDEN,
    _PATCH,
    _REGISTERED_ANGLES_DEG,
    _W_T,
    _base_sample,
    _build,
    _cond_field,
    _cyl_conditioning,
    _geoms,
    _n_trainable,
    _official_path,
    _rel_l2,
    _save_tiny_cyl,
    _yaw_sample,
)

DEV = "cpu"


@pytest.fixture(scope="module")
def tiny_cyl_dir(tmp_path_factory):
    return _save_tiny_cyl(tmp_path_factory.mktemp("exp04n_tiny_cyl"))


def _build_with_rng_state(cond_cfg, seed: int = 42):
    """Build under a pinned global seed and return ``(mc, post-construction RNG state)``."""
    torch.manual_seed(seed)
    mc = create_multi_conditioner_from_conditioning_config(copy.deepcopy(cond_cfg))
    return mc.eval(), torch.random.get_rng_state()


def _full_model_config(tiny_cyl_dir, *, cond_pool):
    """The exp-09 full-model config re-pointed at the tiny backbone, with ``cond_pool``
    (None ⇒ the legacy mean+Linear arm) in BOTH ViT blocks and a shrunken DiT."""
    with open(_EXP09_JSON) as f:
        cfg = json.load(f)
    for c in cfg["model"]["conditioning"]["configs"]:
        if c["type"] == "ViTCoordinates":
            block = c["config"]["ViT"]
            block["hf_model_name_or_path"] = tiny_cyl_dir
            if cond_pool is not None:
                block["cond_pool"] = cond_pool
                block["cond_mlp_hidden"] = _HIDDEN
    cfg["model"]["diffusion"]["config"]["depth"] = 2   # shrink the DiT for a fast CPU build
    return cfg


def _full_model_with_rng_state(tiny_cyl_dir, *, cond_pool, seed: int = 42):
    from src.models import create_model_from_config

    torch.manual_seed(seed)
    model = create_model_from_config(_full_model_config(tiny_cyl_dir, cond_pool=cond_pool))
    return model, torch.random.get_rng_state()


# ------------------------------------------------------------------------------------ #
# 1. "mean_mlp" is a REGISTERED value that builds the declared head at the legacy pooling
# ------------------------------------------------------------------------------------ #
def test_mean_mlp_builds_the_declared_head_shapes(tiny_cyl_dir):
    mc = _build(_cyl_conditioning(tiny_cyl_dir, with_context=True,
                                  cond_pool="mean_mlp", cond_mlp_hidden=_HIDDEN))
    geoms = _geoms(mc)
    assert len(geoms) == 2
    head = geoms[0].lin_proj
    assert geoms[1].lin_proj is head, "the MLP head is not ONE shared object"
    assert geoms[0].vit is geoms[1].vit, "the backbone is not shared"
    assert isinstance(head, nn.Sequential) and len(head) == 3, head
    assert type(head[0]) is nn.Linear
    assert (head[0].in_features, head[0].out_features) == (_HIDDEN, _HIDDEN)
    assert type(head[1]) is nn.GELU, type(head[1])
    assert type(head[2]) is nn.Linear
    assert (head[2].in_features, head[2].out_features) == (_HIDDEN, _COND_DIM)
    for geom in geoms:
        assert geom.dino_pool == "mean", "exp_04 serves the LEGACY (mean) pooling"
        assert geom.model_type == "dino"


def test_mean_mlp_default_width_is_the_backbone_width(tiny_cyl_dir):
    geom = _geoms(_build(_cyl_conditioning(tiny_cyl_dir, with_context=False,
                                           cond_pool="mean_mlp")))[0]
    assert (geom.lin_proj[0].in_features, geom.lin_proj[0].out_features) == (_HIDDEN, _HIDDEN)


def test_max_mlp_regression_is_unchanged(tiny_cyl_dir):
    """The exp_03 arm must be untouched by the exp_04 knob value."""
    geom = _geoms(_build(_cyl_conditioning(tiny_cyl_dir, with_context=False,
                                           cond_pool="max_mlp")))[0]
    assert geom.dino_pool == "max"
    assert isinstance(geom.lin_proj, nn.Sequential) and len(geom.lin_proj) == 3


@pytest.mark.parametrize("value", ["avg_mlp", "mean", "meanmlp", "MEAN_MLP", "mean_MLP",
                                   "", True, 1])
def test_unknown_cond_pool_values_still_raise(tiny_cyl_dir, value):
    """Fail-closed is preserved: only the two REGISTERED values are accepted."""
    cfg = _cyl_conditioning(tiny_cyl_dir, with_context=False, cond_pool=value)
    with pytest.raises(ValueError, match="cond_pool"):
        create_multi_conditioner_from_conditioning_config(cfg)


def test_orphan_and_invalid_widths_still_raise_under_mean_mlp(tiny_cyl_dir):
    orphan = _cyl_conditioning(tiny_cyl_dir, with_context=False, cond_mlp_hidden=_HIDDEN)
    with pytest.raises(ValueError, match="cond_mlp_hidden"):
        create_multi_conditioner_from_conditioning_config(orphan)
    for width in (0, -1, True, 3.5, "64", _HIDDEN * 2):
        cfg = _cyl_conditioning(tiny_cyl_dir, with_context=False, cond_pool="mean_mlp")
        cfg["configs"][0]["config"]["ViT"]["cond_mlp_hidden"] = width
        with pytest.raises(ValueError, match="cond_mlp_hidden"):
            create_multi_conditioner_from_conditioning_config(cfg)


def test_unequal_vit_blocks_trip_the_shared_backbone_guard(tiny_cyl_dir):
    cfg = _cyl_conditioning(tiny_cyl_dir, with_context=True, cond_pool="mean_mlp",
                            cond_mlp_hidden=_HIDDEN,
                            second_block_overrides={"cond_pool": "max_mlp"})
    with pytest.raises(ValueError, match="differ"):
        create_multi_conditioner_from_conditioning_config(cfg)


# ------------------------------------------------------------------------------------ #
# 2. CROSS-ARM IDENTITY (delta 1b): mean_mlp and max_mlp differ ONLY in the selector
# ------------------------------------------------------------------------------------ #
def test_cross_arm_state_dicts_and_rng_states_are_bitwise_identical(tiny_cyl_dir):
    """Same seed, same config except the knob VALUE ⇒ every tensor (hidden AND output layer
    included) bitwise equal and the post-construction global RNG state bitwise equal. This
    is the factor-isolation proof: exp_04 vs exp_03 is the pooling selector and nothing
    else. A copy-pasted head construction (a second fork_rng, a reordered draw) breaks it."""
    mean_cfg = _cyl_conditioning(tiny_cyl_dir, with_context=True, cond_pool="mean_mlp",
                                 cond_mlp_hidden=_HIDDEN)
    max_cfg = _cyl_conditioning(tiny_cyl_dir, with_context=True, cond_pool="max_mlp",
                                cond_mlp_hidden=_HIDDEN)
    mean_mc, mean_rng = _build_with_rng_state(mean_cfg, seed=42)
    max_mc, max_rng = _build_with_rng_state(max_cfg, seed=42)

    mean_sd, max_sd = mean_mc.state_dict(), max_mc.state_dict()
    assert set(mean_sd) == set(max_sd), (
        f"state-dict key sets differ: {sorted(set(mean_sd) ^ set(max_sd))}"
    )
    head_keys = [k for k in mean_sd if ".lin_proj.0." in k or ".lin_proj.2." in k]
    assert head_keys, "no MLP head tensors in the state dict — the oracle would be vacuous"
    for key in sorted(mean_sd):
        assert torch.equal(mean_sd[key], max_sd[key]), (
            f"cross-arm tensor {key} is NOT bitwise identical — the exp_04/exp_03 comparison "
            "would not isolate the pooling selector"
        )
    assert torch.equal(mean_rng, max_rng), (
        "the two arms left DIFFERENT global RNG states — the head construction is not the "
        "shared code path"
    )
    # ... and the arms really are wired differently (the oracle is not vacuous)
    assert _geoms(mean_mc)[0].dino_pool == "mean"
    assert _geoms(max_mc)[0].dino_pool == "max"


def test_cross_arm_full_model_identity_including_downstream_modules(tiny_cyl_dir):
    """The same oracle on FULL models: every module built AFTER the conditioners (DiT,
    context encoders, …) must also be bitwise identical across the two arms."""
    mean_model, mean_rng = _full_model_with_rng_state(tiny_cyl_dir, cond_pool="mean_mlp")
    max_model, max_rng = _full_model_with_rng_state(tiny_cyl_dir, cond_pool="max_mlp")
    mean_sd, max_sd = mean_model.state_dict(), max_model.state_dict()
    assert set(mean_sd) == set(max_sd), sorted(set(mean_sd) ^ set(max_sd))
    for key in sorted(mean_sd):
        assert torch.equal(mean_sd[key], max_sd[key]), f"cross-arm tensor {key} differs"
    assert torch.equal(mean_rng, max_rng)


# ------------------------------------------------------------------------------------ #
# 3. common-init identity vs the LEGACY (mean + bare Linear) arm
# ------------------------------------------------------------------------------------ #
def test_common_init_is_bitwise_identical_including_the_mapped_output_layer(tiny_cyl_dir):
    """One-factor discipline vs the arm exp_04 is measured against: under seed 42 every
    shared tensor is bitwise equal, ONLY the hidden layer is new, and the renamed output
    layer ``lin_proj.2.*`` equals the legacy ``lin_proj.*`` for BOTH conditioner aliases."""
    legacy = _full_model_with_rng_state(tiny_cyl_dir, cond_pool=None)[0].state_dict()
    new = _full_model_with_rng_state(tiny_cyl_dir, cond_pool="mean_mlp")[0].state_dict()

    aliases = ["conditioner.conditioners.source_vit",
               "conditioner.conditioners.context_poses_vit"]
    only_new, only_legacy = set(new) - set(legacy), set(legacy) - set(new)
    assert only_new == {f"{a}.lin_proj.{i}.{t}"
                        for a in aliases for i in (0, 2) for t in ("weight", "bias")}, sorted(only_new)
    assert only_legacy == {f"{a}.lin_proj.{t}"
                           for a in aliases for t in ("weight", "bias")}, sorted(only_legacy)
    for key in sorted(set(legacy) & set(new)):
        assert torch.equal(legacy[key], new[key]), f"common tensor {key} is NOT bitwise identical"
    for alias in aliases:
        for tensor in ("weight", "bias"):
            assert torch.equal(legacy[f"{alias}.lin_proj.{tensor}"],
                               new[f"{alias}.lin_proj.2.{tensor}"]), (
                f"{alias}: the mean_mlp OUTPUT layer is not bitwise-equal to the legacy projection"
            )


def test_hidden_layer_init_is_rng_isolated_and_reproducible(tiny_cyl_dir):
    cfg = _cyl_conditioning(tiny_cyl_dir, with_context=False, cond_pool="mean_mlp")
    head_a = _geoms(_build(cfg, seed=42))[0].lin_proj
    head_b = _geoms(_build(cfg, seed=1234))[0].lin_proj
    assert torch.equal(head_a[0].weight, head_b[0].weight), (
        "hidden-layer init is not RNG-isolated (it moved with the global seed)"
    )
    assert torch.equal(head_a[0].bias, head_b[0].bias)
    assert not torch.equal(head_a[2].weight, head_b[2].weight), (
        "the OUTPUT layer must still follow the global seed (it is the legacy draw)"
    )
    torch.manual_seed(7)
    create_multi_conditioner_from_conditioning_config(copy.deepcopy(cfg))
    after_mean_mlp = torch.randn(4)
    torch.manual_seed(7)
    create_multi_conditioner_from_conditioning_config(
        copy.deepcopy(_cyl_conditioning(tiny_cyl_dir, with_context=False))
    )
    after_legacy = torch.randn(4)
    assert torch.equal(after_mean_mlp, after_legacy), (
        "the mean_mlp build consumed a different number of GLOBAL RNG draws than legacy"
    )


# ------------------------------------------------------------------------------------ #
# 4. live forward: the served condition is head(pooler_output), NOT head(amax(tokens))
# ------------------------------------------------------------------------------------ #
def test_mean_mlp_forward_is_the_mlp_of_the_pooler_output(tiny_cyl_dir):
    geom = _geoms(_build(_cyl_conditioning(tiny_cyl_dir, with_context=False,
                                           cond_pool="mean_mlp")))[0]
    batch = [_base_sample(0), _base_sample(1)]
    with torch.no_grad():
        got = geom(batch, device=DEV)[0]
        field = torch.cat([_cond_field(s, geom.max_value) for s in batch], dim=0)
        outputs = geom.vit(field)
        want = geom.lin_proj(outputs.pooler_output).unsqueeze(1)
        sabotage = geom.lin_proj(outputs.last_hidden_state.amax(dim=1)).unsqueeze(1)
    assert outputs.last_hidden_state.shape == (2, _H_T * _W_T, _HIDDEN)
    assert got.shape == (2, 1, _COND_DIM)
    assert torch.equal(got, want), (
        f"served condition != MLP(pooler_output) (max abs {(got - want).abs().max().item():.3e})"
    )
    # SABOTAGE guard: the max-pooled oracle must NOT match, or the assertion above would
    # pass for the exp_03 arm too and prove nothing about WHICH pooling is served.
    assert not torch.allclose(got, sabotage), (
        "MLP(amax(tokens)) is indistinguishable from the served condition — the mean/max "
        "distinction is not being measured"
    )


def test_mean_mlp_pooling_path_is_byte_identical_to_the_legacy_pooling(tiny_cyl_dir):
    """The pooling itself is the LEGACY path (``pooler_output``), so the pooled vector the
    head consumes is bitwise equal to the vector the legacy mean+Linear arm consumes."""
    geom = _geoms(_build(_cyl_conditioning(tiny_cyl_dir, with_context=False,
                                           cond_pool="mean_mlp")))[0]
    with torch.no_grad():
        outputs = geom.vit(_cond_field(_base_sample(0), geom.max_value))
    assert torch.equal(outputs.pooler_output, outputs.last_hidden_state.mean(dim=1))


# ------------------------------------------------------------------------------------ #
# 5. parameter-count oracles
# ------------------------------------------------------------------------------------ #
def test_trainable_param_delta_is_exactly_the_hidden_layer(tiny_cyl_dir):
    legacy = _build(_cyl_conditioning(tiny_cyl_dir, with_context=True))
    new = _build(_cyl_conditioning(tiny_cyl_dir, with_context=True,
                                   cond_pool="mean_mlp", cond_mlp_hidden=_HIDDEN))
    assert _n_trainable(new) - _n_trainable(legacy) == _HIDDEN * _HIDDEN + _HIDDEN


def test_official_head_is_384_384_256_and_the_delta_is_147840():
    """The production oracle: at the REAL ViT-S/16 the mean_mlp head is exactly
    Linear(384,384) → GELU → Linear(384,256) and the trainable-parameter delta is +147,840."""
    official = _official_path()
    legacy = _build(_cyl_conditioning(official, with_context=True))
    new = _build(_cyl_conditioning(official, with_context=True,
                                   cond_pool="mean_mlp", cond_mlp_hidden=_OFFICIAL_HIDDEN))
    head = _geoms(new)[0].lin_proj
    assert (head[0].in_features, head[0].out_features) == (384, 384)
    assert type(head[1]) is nn.GELU
    assert (head[2].in_features, head[2].out_features) == (384, 256)
    assert _n_trainable(new) - _n_trainable(legacy) == 147840


# ------------------------------------------------------------------------------------ #
# 6. yaw invariance of the SERVED condition (delta §3), end-to-end
# ------------------------------------------------------------------------------------ #
def _invariance_residuals(geom, *, angles=_REGISTERED_ANGLES_DEG):
    base = _base_sample(0)
    residuals = {}
    with torch.no_grad():
        cond_base = geom([base], device=DEV)[0]
    for deg in angles:
        k = int(round(deg / 360.0 * _W_T))
        with torch.no_grad():
            cond_rot = geom([_yaw_sample(base, k)], device=DEV)[0]
        residuals[deg] = _rel_l2(cond_rot, cond_base)
    return residuals


def test_the_yaw_fixture_actually_changes_the_backbone_input():
    """Non-vacuity (i): the yawed input genuinely differs from the base input and is
    EXACTLY the package's ``physical_yaw`` of it."""
    base = _base_sample(0)
    field = _cond_field(base)
    for deg in _REGISTERED_ANGLES_DEG:
        k = int(round(deg / 360.0 * _W_T))
        rot_field = _cond_field(_yaw_sample(base, k))
        assert not torch.allclose(rot_field, field, atol=1e-3), (
            f"the {deg}-deg yawed input is indistinguishable from the base input"
        )
        assert torch.allclose(rot_field, physical_yaw(field, k * _PATCH), atol=1e-5)


def test_mean_mlp_condition_is_invariant_at_the_registered_angles(tiny_cyl_dir):
    """The SERVED condition MLP(mean(tokens)) is yaw-invariant within the registered 1e-4
    bound: the roll permutes the token axis and the mean over that axis is
    permutation-invariant up to floating-point reduction order."""
    geom = _geoms(_build(_cyl_conditioning(tiny_cyl_dir, with_context=False,
                                           cond_pool="mean_mlp")))[0]
    residuals = _invariance_residuals(geom)
    worst = max(residuals.values())
    assert worst <= _BOUND, f"mean_mlp condition not yaw-invariant: {residuals}"
    print(f"\n[exp04n] mean_mlp condition rel-L2 residuals: "
          f"{ {k: f'{v:.3e}' for k, v in residuals.items()} }")


def test_gauge_off_negative_control_violates_invariance(tiny_cyl_dir):
    """NEGATIVE control: with the gauge DISABLED the same fixture must break invariance by
    at least 10x the bound — otherwise the positive test proves nothing."""
    geom = _geoms(_build(_cyl_conditioning(tiny_cyl_dir, with_context=False,
                                           cond_pool="mean_mlp", gauge="none")))[0]
    worst = max(_invariance_residuals(geom).values())
    assert worst >= _BROKEN, (
        f"gauge-OFF control did NOT break invariance (worst rel-L2 {worst:.3e} < {_BROKEN:.0e})"
    )


# ------------------------------------------------------------------------------------ #
# 7. gradients through BOTH conditioner uses (one backward per use)
# ------------------------------------------------------------------------------------ #
@pytest.mark.parametrize("use", [0, 1])
def test_grads_reach_both_mlp_layers_and_the_backbone_through_each_conditioner_use(
    tiny_cyl_dir, use
):
    mc = _build(_cyl_conditioning(tiny_cyl_dir, with_context=True,
                                  cond_pool="mean_mlp", cond_mlp_hidden=_HIDDEN))
    geoms = _geoms(mc)
    assert len(geoms) == 2 and geoms[0].lin_proj is geoms[1].lin_proj
    head = geoms[0].lin_proj
    backbone_param = geoms[0].vit.layer[0].mlp.up_proj.weight
    assert backbone_param.requires_grad

    mc.train()
    mc.zero_grad(set_to_none=True)
    try:
        for param in (head[0].weight, head[0].bias, head[2].weight, head[2].bias, backbone_param):
            assert param.grad is None, "grads were not cleared before the measured backward"
        geoms[use]([_base_sample(0)], device=DEV)[0].float().pow(2).mean().backward()
        for name, layer in (("hidden", head[0]), ("output", head[2])):
            for tensor in ("weight", "bias"):
                grad = getattr(layer, tensor).grad
                assert grad is not None, f"conditioner {use}: {name}.{tensor} got no grad"
                assert torch.isfinite(grad).all(), f"conditioner {use}: {name}.{tensor} not finite"
                assert grad.abs().max().item() > 0.0, f"conditioner {use}: {name}.{tensor} all zeros"
        assert backbone_param.grad is not None, f"conditioner {use}: the backbone got no grad"
        assert torch.isfinite(backbone_param.grad).all()
        assert backbone_param.grad.abs().max().item() > 0.0
    finally:
        mc.zero_grad(set_to_none=True)
        mc.eval()


# ------------------------------------------------------------------------------------ #
# 8. per-variant config contract (delta §2)
# ------------------------------------------------------------------------------------ #
_NEW_KEYS = {"cond_pool": "mean_mlp", "cond_mlp_hidden": 384}

_CONFIG_VARIANTS = (
    # (exp04n file, exp-09 reference it must reconstruct to)
    ("FLAC_AR_exp04n.json", "FLAC_AR_exp09.json"),
    ("FLAC_AR_exp04n_online_eval.json", "FLAC_AR_exp09_online_eval.json"),
)


@pytest.mark.parametrize("exp04n_name,exp09_name", _CONFIG_VARIANTS)
def test_config_contract_exp04n(exp04n_name, exp09_name):
    """Each exp04n config is its OWN exp-09 counterpart plus EXACTLY the two conditioning
    keys in BOTH ViT blocks — deleting only those keys must reconstruct the reference as a
    parsed object. The per-variant reference matters: the online-eval config intentionally
    differs (use_ema false, no gradient_checkpointing keys), so a single shared reference
    would hide a swap."""
    with open(_EXP09_DIR / exp04n_name) as f:
        exp04n = json.load(f)
    with open(_EXP09_DIR / exp09_name) as f:
        reference = json.load(f)

    vit_blocks = [c["config"]["ViT"] for c in exp04n["model"]["conditioning"]["configs"]
                  if c["type"] == "ViTCoordinates"]
    assert len(vit_blocks) == 2, f"expected 2 ViT blocks, got {len(vit_blocks)}"
    for block in vit_blocks:
        for key, value in _NEW_KEYS.items():
            assert block.get(key) == value, f"{exp04n_name}: ViT block {key} = {block.get(key)!r}"
    assert vit_blocks[0] == vit_blocks[1], f"{exp04n_name}: the two ViT blocks are not deep-equal"

    stripped = copy.deepcopy(exp04n)
    n = 0
    for c in stripped["model"]["conditioning"]["configs"]:
        if c["type"] == "ViTCoordinates":
            for key in _NEW_KEYS:
                del c["config"]["ViT"][key]
            n += 1
    assert n == 2
    assert stripped == reference, (
        f"{exp04n_name} minus the two conditioning keys != {exp09_name} (parsed-object "
        "mismatch) — something OTHER than the head changed"
    )


def test_exp04n_configs_differ_from_exp03n_ONLY_in_the_cond_pool_value():
    """The run identity of delta §4: exp_04's configs are exp_03's with ``max_mlp`` →
    ``mean_mlp`` and nothing else (the recipe, the gauge and the backbone are the control
    surface of BOTH experiments)."""
    for exp04n_name, exp03n_name in (("FLAC_AR_exp04n.json", "FLAC_AR_exp03n.json"),
                                     ("FLAC_AR_exp04n_online_eval.json",
                                      "FLAC_AR_exp03n_online_eval.json")):
        with open(_EXP09_DIR / exp04n_name) as f:
            exp04n = json.load(f)
        with open(_EXP09_DIR / exp03n_name) as f:
            exp03n = json.load(f)
        for c in exp04n["model"]["conditioning"]["configs"]:
            if c["type"] == "ViTCoordinates":
                c["config"]["ViT"]["cond_pool"] = "max_mlp"
        assert exp04n == exp03n, (
            f"{exp04n_name} differs from {exp03n_name} by more than the cond_pool value"
        )


def test_exp04n_train_config_still_carries_the_fa_invariant_training_pin():
    with open(_EXP09_DIR / "FLAC_AR_exp04n.json") as f:
        cfg = json.load(f)
    assert cfg["training"]["cond_method"] == "fa_invariant"
    assert cfg["training"]["frame_avg_angles"] == [0.0]
    assert cfg["training"]["use_ema"] is True


def test_exp04n_online_eval_config_is_the_non_ema_variant():
    with open(_EXP09_DIR / "FLAC_AR_exp04n_online_eval.json") as f:
        cfg = json.load(f)
    assert cfg["training"]["use_ema"] is False
    for c in cfg["model"]["conditioning"]["configs"]:
        if c["type"] == "ViTCoordinates":
            assert "gradient_checkpointing" not in c["config"], (
                "the online-eval variant must not carry gradient_checkpointing (exp-09 convention)"
            )
