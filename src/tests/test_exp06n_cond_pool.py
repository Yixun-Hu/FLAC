"""exp_06 (worklog_yixun_neuronic) — MAX-pool + the LEGACY bare-Linear conditioning head.

The exp_06 delta (plan §1) is ONE new registered value of the exp_03 knob and it is the
cleanest arm of the pool-x-head 2x2:

* ``"cond_pool": "max_linear"`` swaps ONLY the pooling — ``last_hidden_state.amax(dim=1)``
  instead of ``pooler_output`` — while the head stays the LEGACY ``Linear(H, cond_dim)``
  drawn at the LEGACY code point from the GLOBAL RNG stream. NO hidden layer, NO extra
  parameters, NOT ONE additional RNG draw;
* consequently the BITWISE ORACLE is stronger than exp_03/04's: a seed-42 ``max_linear``
  build and a seed-42 legacy build must have IDENTICAL full state dicts (exact key-set
  equality both ways + ``torch.equal`` per tensor) AND identical post-construction global
  RNG states;
* ``cond_mlp_hidden`` is FORBIDDEN with ``max_linear`` (no hidden layer exists to size);
  ``"max_mlp"`` is unchanged and genuinely unknown values still ValueError.

Fixtures, tiny backbone and geometry are imported from ``test_exp03n_cond_pool`` so the
arms are measured with the SAME harness (64x128 px at patch 16 → a 4x8 token grid,
W_t = 8; the registered angles {90, 180, 270} are exactly 2, 4 and 6 whole token columns).

CPU-only and offline; one test binds the real 384-d official snapshot when it is in the
local HF cache (skips otherwise) for the production head shape and the ZERO-delta oracle.
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
# non-axisymmetric scene and the same A2b statistic — the arms must be comparable.
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
    _roll_tokens,
    _save_tiny_cyl,
    _yaw_sample,
)

DEV = "cpu"


@pytest.fixture(scope="module")
def tiny_cyl_dir(tmp_path_factory):
    return _save_tiny_cyl(tmp_path_factory.mktemp("exp06n_tiny_cyl"))


def _build_with_rng_state(cond_cfg, seed: int = 42):
    """Build under a pinned global seed and return ``(mc, post-construction RNG state)``."""
    torch.manual_seed(seed)
    mc = create_multi_conditioner_from_conditioning_config(copy.deepcopy(cond_cfg))
    return mc.eval(), torch.random.get_rng_state()


def _full_model_config(tiny_cyl_dir, *, cond_pool):
    """The exp-09 full-model config re-pointed at the tiny backbone, with ``cond_pool``
    (None ⇒ the legacy mean+Linear arm) in BOTH ViT blocks and a shrunken DiT. UNLIKE the
    exp_03/04 helpers no ``cond_mlp_hidden`` is ever injected — ``max_linear`` forbids it."""
    with open(_EXP09_JSON) as f:
        cfg = json.load(f)
    for c in cfg["model"]["conditioning"]["configs"]:
        if c["type"] == "ViTCoordinates":
            block = c["config"]["ViT"]
            block["hf_model_name_or_path"] = tiny_cyl_dir
            if cond_pool is not None:
                block["cond_pool"] = cond_pool
    cfg["model"]["diffusion"]["config"]["depth"] = 2   # shrink the DiT for a fast CPU build
    return cfg


def _full_model_with_rng_state(tiny_cyl_dir, *, cond_pool, seed: int = 42):
    from src.models import create_model_from_config

    torch.manual_seed(seed)
    model = create_model_from_config(_full_model_config(tiny_cyl_dir, cond_pool=cond_pool))
    return model, torch.random.get_rng_state()


# ------------------------------------------------------------------------------------ #
# 1. "max_linear" is a REGISTERED value: max pooling + the LEGACY bare Linear head
# ------------------------------------------------------------------------------------ #
def test_max_linear_builds_the_legacy_bare_linear_head_at_max_pooling(tiny_cyl_dir):
    mc = _build(_cyl_conditioning(tiny_cyl_dir, with_context=True, cond_pool="max_linear"))
    geoms = _geoms(mc)
    assert len(geoms) == 2
    head = geoms[0].lin_proj
    assert geoms[1].lin_proj is head, "the Linear head is not ONE shared object"
    assert geoms[0].vit is geoms[1].vit, "the backbone is not shared"
    assert type(head) is nn.Linear, (
        f"the exp_06 head must be the LEGACY bare nn.Linear, got {type(head).__name__}"
    )
    assert (head.in_features, head.out_features) == (_HIDDEN, _COND_DIM)
    for geom in geoms:
        assert geom.dino_pool == "max", "exp_06 serves MAX pooling"
        assert geom.model_type == "dino"


def test_max_mlp_regression_is_unchanged(tiny_cyl_dir):
    """The exp_03 arm must be untouched by the exp_06 knob value."""
    geom = _geoms(_build(_cyl_conditioning(tiny_cyl_dir, with_context=False,
                                           cond_pool="max_mlp")))[0]
    assert geom.dino_pool == "max"
    assert isinstance(geom.lin_proj, nn.Sequential) and len(geom.lin_proj) == 3


def test_legacy_default_regression_is_unchanged(tiny_cyl_dir):
    """The absent-key path must stay the legacy mean-pool + bare-Linear arm."""
    geom = _geoms(_build(_cyl_conditioning(tiny_cyl_dir, with_context=False)))[0]
    assert geom.dino_pool == "mean"
    assert type(geom.lin_proj) is nn.Linear


@pytest.mark.parametrize("value", ["max", "maxlinear", "MAX_LINEAR", "max_Linear",
                                   "mean_linear", "linear", "", True, 1])
def test_unknown_cond_pool_values_still_raise(tiny_cyl_dir, value):
    """Fail-closed is preserved: only the REGISTERED values are accepted."""
    cfg = _cyl_conditioning(tiny_cyl_dir, with_context=False, cond_pool=value)
    with pytest.raises(ValueError, match="cond_pool"):
        create_multi_conditioner_from_conditioning_config(cfg)


def test_cond_mlp_hidden_with_max_linear_raises(tiny_cyl_dir):
    """The bare Linear head has NO hidden layer: a width key would be silently ignored,
    so it must be refused at construction (fail-closed, never a silent no-op)."""
    cfg = _cyl_conditioning(tiny_cyl_dir, with_context=False, cond_pool="max_linear",
                            cond_mlp_hidden=_HIDDEN)
    with pytest.raises(ValueError, match="cond_mlp_hidden"):
        create_multi_conditioner_from_conditioning_config(cfg)


def test_orphan_cond_mlp_hidden_still_raises(tiny_cyl_dir):
    cfg = _cyl_conditioning(tiny_cyl_dir, with_context=False, cond_mlp_hidden=_HIDDEN)
    with pytest.raises(ValueError, match="cond_mlp_hidden"):
        create_multi_conditioner_from_conditioning_config(cfg)


def test_unequal_vit_blocks_trip_the_shared_backbone_guard(tiny_cyl_dir):
    cfg = _cyl_conditioning(tiny_cyl_dir, with_context=True, cond_pool="max_linear",
                            second_block_overrides={"cond_pool": "max_mlp"})
    with pytest.raises(ValueError, match="differ"):
        create_multi_conditioner_from_conditioning_config(cfg)


# ------------------------------------------------------------------------------------ #
# 2. the BITWISE ORACLE (plan §1): full state-dict identity vs the LEGACY build
# ------------------------------------------------------------------------------------ #
def test_conditioner_state_dict_is_bitwise_the_legacy_one_with_equal_rng_states(tiny_cyl_dir):
    """Pooling has no parameters and the construction order is untouched, so at seed 42
    the FULL state dict of the max_linear build must be BITWISE EQUAL to the legacy
    mean+Linear build — exact key-set equality BOTH ways, ``torch.equal`` per tensor —
    and the post-construction global RNG states must be equal (not one extra draw)."""
    legacy_mc, legacy_rng = _build_with_rng_state(
        _cyl_conditioning(tiny_cyl_dir, with_context=True), seed=42)
    new_mc, new_rng = _build_with_rng_state(
        _cyl_conditioning(tiny_cyl_dir, with_context=True, cond_pool="max_linear"), seed=42)

    legacy_sd, new_sd = legacy_mc.state_dict(), new_mc.state_dict()
    assert set(new_sd) - set(legacy_sd) == set(), sorted(set(new_sd) - set(legacy_sd))
    assert set(legacy_sd) - set(new_sd) == set(), sorted(set(legacy_sd) - set(new_sd))
    for key in sorted(legacy_sd):
        assert torch.equal(legacy_sd[key], new_sd[key]), (
            f"tensor {key} is NOT bitwise identical to the legacy seed-42 build — the "
            "ablation would not be pooling-only"
        )
    assert torch.equal(legacy_rng, new_rng), (
        "the max_linear build left a DIFFERENT global RNG state than the legacy build — "
        "the branch consumed RNG it must not touch"
    )
    # non-vacuity: the two builds really are wired to different poolings
    assert _geoms(legacy_mc)[0].dino_pool == "mean"
    assert _geoms(new_mc)[0].dino_pool == "max"


def test_full_model_state_dict_is_bitwise_the_legacy_one(tiny_cyl_dir):
    """The same oracle on FULL models (conditioners + DiT + everything downstream): the
    key sets are equal both ways, every tensor is bitwise equal, and the post-build global
    RNG states match — so every module built AFTER the conditioner is untouched too."""
    legacy_model, legacy_rng = _full_model_with_rng_state(tiny_cyl_dir, cond_pool=None)
    new_model, new_rng = _full_model_with_rng_state(tiny_cyl_dir, cond_pool="max_linear")
    legacy_sd, new_sd = legacy_model.state_dict(), new_model.state_dict()
    assert set(new_sd) - set(legacy_sd) == set(), sorted(set(new_sd) - set(legacy_sd))
    assert set(legacy_sd) - set(new_sd) == set(), sorted(set(legacy_sd) - set(new_sd))
    for key in sorted(legacy_sd):
        assert torch.equal(legacy_sd[key], new_sd[key]), f"full-model tensor {key} differs"
    assert torch.equal(legacy_rng, new_rng)


def test_the_head_follows_the_global_seed_and_no_extra_global_draws_happen(tiny_cyl_dir):
    """The head IS the legacy projection: it must move with the global seed (it is the
    legacy draw, not a pinned side stream), and a build must consume exactly as many
    global draws as the legacy build."""
    cfg = _cyl_conditioning(tiny_cyl_dir, with_context=False, cond_pool="max_linear")
    head_a = _geoms(_build(cfg, seed=42))[0].lin_proj
    head_b = _geoms(_build(cfg, seed=1234))[0].lin_proj
    assert not torch.equal(head_a.weight, head_b.weight), (
        "the head must follow the global seed (it is the legacy projection draw)"
    )
    torch.manual_seed(7)
    create_multi_conditioner_from_conditioning_config(copy.deepcopy(cfg))
    after_max_linear = torch.randn(4)
    torch.manual_seed(7)
    create_multi_conditioner_from_conditioning_config(
        copy.deepcopy(_cyl_conditioning(tiny_cyl_dir, with_context=False))
    )
    after_legacy = torch.randn(4)
    assert torch.equal(after_max_linear, after_legacy), (
        "the max_linear build consumed a different number of GLOBAL RNG draws than legacy"
    )


# ------------------------------------------------------------------------------------ #
# 3. live forward: the served condition is Linear(amax(tokens)), NOT Linear(mean)
# ------------------------------------------------------------------------------------ #
def test_max_linear_forward_is_the_linear_of_the_token_amax(tiny_cyl_dir):
    geom = _geoms(_build(_cyl_conditioning(tiny_cyl_dir, with_context=False,
                                           cond_pool="max_linear")))[0]
    batch = [_base_sample(0), _base_sample(1)]
    with torch.no_grad():
        got = geom(batch, device=DEV)[0]
        field = torch.cat([_cond_field(s, geom.max_value) for s in batch], dim=0)
        outputs = geom.vit(field)
        want = geom.lin_proj(outputs.last_hidden_state.amax(dim=1)).unsqueeze(1)
        sabotage = geom.lin_proj(outputs.pooler_output).unsqueeze(1)
    assert outputs.last_hidden_state.shape == (2, _H_T * _W_T, _HIDDEN)
    assert got.shape == (2, 1, _COND_DIM)
    assert torch.equal(got, want), (
        f"served condition != Linear(amax(tokens)) (max abs {(got - want).abs().max().item():.3e})"
    )
    # SABOTAGE guard: the MEAN-pooled oracle must NOT match, or the assertion above would
    # pass for the legacy arm too and prove nothing about WHICH pooling is served (state-
    # dict identity and equivariance alone cannot catch a silent revert to mean).
    assert not torch.allclose(got, sabotage), (
        "Linear(pooler_output) is indistinguishable from the served condition — the "
        "mean/max distinction is not being measured"
    )


# ------------------------------------------------------------------------------------ #
# 4. parameter-count oracles: the delta is EXACTLY zero
# ------------------------------------------------------------------------------------ #
def test_trainable_param_delta_is_exactly_zero(tiny_cyl_dir):
    legacy = _build(_cyl_conditioning(tiny_cyl_dir, with_context=True))
    new = _build(_cyl_conditioning(tiny_cyl_dir, with_context=True, cond_pool="max_linear"))
    assert _n_trainable(new) - _n_trainable(legacy) == 0


def test_official_head_is_384_256_and_the_delta_is_zero():
    """The production oracle: at the REAL ViT-S/16 the max_linear head is exactly the
    legacy Linear(384,256) (98,560 params) and the trainable-parameter delta is ZERO."""
    official = _official_path()
    legacy = _build(_cyl_conditioning(official, with_context=True))
    new = _build(_cyl_conditioning(official, with_context=True, cond_pool="max_linear"))
    head = _geoms(new)[0].lin_proj
    assert type(head) is nn.Linear
    assert (head.in_features, head.out_features) == (_OFFICIAL_HIDDEN, _COND_DIM)
    assert _n_trainable(new) - _n_trainable(legacy) == 0


# ------------------------------------------------------------------------------------ #
# 5. yaw invariance of the SERVED condition, end-to-end (non-axisymmetric fixture)
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


def test_backbone_tokens_undergo_the_expected_column_roll(tiny_cyl_dir):
    """Non-vacuity (ii): under the gauge, a physical yaw of k token columns permutes the
    token field by exactly that column roll — the premise max-pool invariance rests on."""
    geom = _geoms(_build(_cyl_conditioning(tiny_cyl_dir, with_context=False,
                                           cond_pool="max_linear")))[0]
    base = _base_sample(0)
    with torch.no_grad():
        tokens = geom.vit(_cond_field(base, geom.max_value)).last_hidden_state
    for deg in _REGISTERED_ANGLES_DEG:
        k = int(round(deg / 360.0 * _W_T))
        with torch.no_grad():
            rolled = geom.vit(_cond_field(_yaw_sample(base, k), geom.max_value)).last_hidden_state
        res = _rel_l2(rolled, _roll_tokens(tokens, k))
        assert res <= _BOUND, f"token field at {deg} deg is not the k={k} column roll (rel-L2 {res:.3e})"


def test_max_linear_condition_is_invariant_at_the_registered_angles(tiny_cyl_dir):
    """The SERVED condition Linear(amax(tokens)) is yaw-invariant within the registered
    1e-4 bound: the roll permutes the token axis, amax over that axis is permutation-
    invariant, and the Linear is pointwise on the pooled vector."""
    geom = _geoms(_build(_cyl_conditioning(tiny_cyl_dir, with_context=False,
                                           cond_pool="max_linear")))[0]
    residuals = _invariance_residuals(geom)
    worst = max(residuals.values())
    assert worst <= _BOUND, f"max_linear condition not yaw-invariant: {residuals}"
    print(f"\n[exp06n] max_linear condition rel-L2 residuals: "
          f"{ {k: f'{v:.3e}' for k, v in residuals.items()} }")


def test_gauge_off_negative_control_violates_invariance(tiny_cyl_dir):
    """NEGATIVE control: with the gauge DISABLED the same fixture must break invariance by
    at least 10x the bound — otherwise the positive test proves nothing."""
    geom = _geoms(_build(_cyl_conditioning(tiny_cyl_dir, with_context=False,
                                           cond_pool="max_linear", gauge="none")))[0]
    worst = max(_invariance_residuals(geom).values())
    assert worst >= _BROKEN, (
        f"gauge-OFF control did NOT break invariance (worst rel-L2 {worst:.3e} < {_BROKEN:.0e})"
    )


# ------------------------------------------------------------------------------------ #
# 6. gradients through BOTH conditioner uses (one backward per use)
# ------------------------------------------------------------------------------------ #
@pytest.mark.parametrize("use", [0, 1])
def test_grads_reach_the_shared_linear_and_the_backbone_through_each_conditioner_use(
    tiny_cyl_dir, use
):
    """ONE backward per conditioner USE, grads zeroed in between: summing both outputs
    into a single backward would let a detached conditioner hide behind the other use of
    the SHARED head/backbone. Each use must independently deliver finite nonzero grads to
    the bare Linear (through the max-pool path) and to a backbone parameter."""
    mc = _build(_cyl_conditioning(tiny_cyl_dir, with_context=True, cond_pool="max_linear"))
    geoms = _geoms(mc)
    assert len(geoms) == 2 and geoms[0].lin_proj is geoms[1].lin_proj
    head = geoms[0].lin_proj
    backbone_param = geoms[0].vit.layer[0].mlp.up_proj.weight
    assert backbone_param.requires_grad

    mc.train()
    mc.zero_grad(set_to_none=True)
    try:
        for param in (head.weight, head.bias, backbone_param):
            assert param.grad is None, "grads were not cleared before the measured backward"
        geoms[use]([_base_sample(0)], device=DEV)[0].float().pow(2).mean().backward()
        for tensor in ("weight", "bias"):
            grad = getattr(head, tensor).grad
            assert grad is not None, f"conditioner {use}: head.{tensor} got no grad"
            assert torch.isfinite(grad).all(), f"conditioner {use}: head.{tensor} not finite"
            assert grad.abs().max().item() > 0.0, f"conditioner {use}: head.{tensor} all zeros"
        assert backbone_param.grad is not None, f"conditioner {use}: the backbone got no grad"
        assert torch.isfinite(backbone_param.grad).all()
        assert backbone_param.grad.abs().max().item() > 0.0
    finally:
        mc.zero_grad(set_to_none=True)
        mc.eval()


# ------------------------------------------------------------------------------------ #
# 7. per-variant config contracts (plan B1 — BOTH bindings, per variant)
# ------------------------------------------------------------------------------------ #
_NEW_KEYS = {"cond_pool": "max_linear"}
_SIBLING_KEYS = {"cond_pool": "max_mlp", "cond_mlp_hidden": 384}   # the exp_03 arm's knob

_CONFIG_VARIANTS = (
    # (exp06n file, exp-09 reference, exp03n sibling) — base<->base and online<->online,
    # never across (the online-eval variant intentionally differs in use_ema/grad-ckpt).
    ("FLAC_AR_exp06n.json", "FLAC_AR_exp09.json", "FLAC_AR_exp03n.json"),
    ("FLAC_AR_exp06n_online_eval.json", "FLAC_AR_exp09_online_eval.json",
     "FLAC_AR_exp03n_online_eval.json"),
)


@pytest.mark.parametrize("exp06n_name,exp09_name,exp03n_name", _CONFIG_VARIANTS)
def test_config_contract_exp06n_reconstructs_exp09(exp06n_name, exp09_name, exp03n_name):
    """B1(ii): each exp06n config is its OWN exp-09 counterpart plus EXACTLY ``cond_pool``
    in BOTH ViT blocks (``cond_mlp_hidden`` ABSENT) — deleting only ``cond_pool`` must
    reconstruct the reference parsed-object-exactly."""
    with open(_EXP09_DIR / exp06n_name) as f:
        exp06n = json.load(f)
    with open(_EXP09_DIR / exp09_name) as f:
        reference = json.load(f)

    vit_blocks = [c["config"]["ViT"] for c in exp06n["model"]["conditioning"]["configs"]
                  if c["type"] == "ViTCoordinates"]
    assert len(vit_blocks) == 2, f"expected 2 ViT blocks, got {len(vit_blocks)}"
    for block in vit_blocks:
        for key, value in _NEW_KEYS.items():
            assert block.get(key) == value, f"{exp06n_name}: ViT block {key} = {block.get(key)!r}"
        assert "cond_mlp_hidden" not in block, (
            f"{exp06n_name}: cond_mlp_hidden must be ABSENT (the bare Linear head has no width)"
        )
    assert vit_blocks[0] == vit_blocks[1], f"{exp06n_name}: the two ViT blocks are not deep-equal"

    stripped = copy.deepcopy(exp06n)
    n = 0
    for c in stripped["model"]["conditioning"]["configs"]:
        if c["type"] == "ViTCoordinates":
            for key in _NEW_KEYS:
                del c["config"]["ViT"][key]
            n += 1
    assert n == 2
    assert stripped == reference, (
        f"{exp06n_name} minus cond_pool != {exp09_name} (parsed-object mismatch) — "
        "something OTHER than the pooling changed"
    )


@pytest.mark.parametrize("exp06n_name,exp09_name,exp03n_name", _CONFIG_VARIANTS)
def test_config_contract_exp06n_vs_exp03n_delta(exp06n_name, exp09_name, exp03n_name):
    """B1(i): exp06 vs exp03, per variant — EXACTLY ``cond_pool`` differs ("max_linear" vs
    "max_mlp") and EXACTLY ``cond_mlp_hidden`` is removed; nothing else. Proven by
    transforming the parsed exp06 config into the sibling and requiring equality."""
    with open(_EXP09_DIR / exp06n_name) as f:
        exp06n = json.load(f)
    with open(_EXP09_DIR / exp03n_name) as f:
        exp03n = json.load(f)
    as_sibling = copy.deepcopy(exp06n)
    n = 0
    for c in as_sibling["model"]["conditioning"]["configs"]:
        if c["type"] == "ViTCoordinates":
            block = c["config"]["ViT"]
            assert block.get("cond_pool") == "max_linear"
            assert "cond_mlp_hidden" not in block
            block.update(_SIBLING_KEYS)
            n += 1
    assert n == 2
    assert as_sibling == exp03n, (
        f"{exp06n_name} differs from {exp03n_name} by MORE than "
        "{cond_pool value, cond_mlp_hidden removal}"
    )


def test_exp06n_train_config_still_carries_the_fa_invariant_training_pin():
    with open(_EXP09_DIR / "FLAC_AR_exp06n.json") as f:
        cfg = json.load(f)
    assert cfg["training"]["cond_method"] == "fa_invariant"
    assert cfg["training"]["frame_avg_angles"] == [0.0]
    assert cfg["training"]["use_ema"] is True


def test_exp06n_online_eval_config_is_the_non_ema_variant():
    with open(_EXP09_DIR / "FLAC_AR_exp06n_online_eval.json") as f:
        cfg = json.load(f)
    assert cfg["training"]["use_ema"] is False
    for c in cfg["model"]["conditioning"]["configs"]:
        if c["type"] == "ViTCoordinates":
            assert "gradient_checkpointing" not in c["config"], (
                "the online-eval variant must not carry gradient_checkpointing (exp-09 convention)"
            )
