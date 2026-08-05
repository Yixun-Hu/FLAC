"""exp_03 (worklog_yixun_neuronic) — max-pool + MLP conditioning head: plan §4.6 tests.

The architecture delta under test (plan §2), scoped ENTIRELY to the
``cylindrical_dinov3`` factory branch of ``src/models/conditioners.py``:

* ``"cond_pool": "max_mlp"`` (+ optional ``"cond_mlp_hidden"``) in a ViT block swaps
  the conditioning head from  mean-pool (``pooler_output``) → ``Linear(H, cond_dim)``
  to  max-pool (``last_hidden_state.amax(dim=1)``) → ``Linear(H, H) → GELU → Linear(H, cond_dim)``;
* the key being ABSENT keeps the legacy path byte-identical (module types, forward
  arithmetic, and the global RNG stream);
* construction order is pinned: the OUTPUT ``Linear(H, cond_dim)`` is drawn from the
  GLOBAL RNG at exactly the legacy code point (so every downstream module's init is
  unchanged AND the output layer is bitwise-equal to the legacy projection), while the
  hidden layer is drawn inside ``torch.random.fork_rng(devices=[])`` from a pinned
  CPU-generator seed.

Everything here is CPU-only and offline: the backbone is a tiny cylindrical DINOv3
built from a config and saved to a tmp dir (no ``from_pretrained`` against the Hub).
One test additionally binds the REAL 384-d official snapshot when it is present in the
local HF cache (skips otherwise) to pin the production head shape and the +147,840
trainable-parameter oracle.

Fixture geometry: 64x128 pixels at patch 16 → a 4x8 token grid, i.e. ``W_t = 8``, so
one token column is 360/8 = 45 deg and the registered angles {90, 180, 270} are EXACTLY
2, 4 and 6 whole token columns of THIS test grid (production is 256x512 → W_t = 32,
where the same angles are 8, 16 and 24 columns). Only whole-column yaws are exact.
"""
import os
import sys

# CPU-only + offline, hard-set before torch initialises CUDA (mandate; also dodges the
# deepspeed op-probe save_pretrained triggers in this env). Set, not setdefault.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import copy  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]  # src/tests -> src -> repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cylindrical_dinov3 import (  # noqa: E402
    CylindricalDINOv3ViTConfig,
    CylindricalDINOv3ViTModel,
    physical_yaw,
)

from src.models.conditioners import (  # noqa: E402
    create_multi_conditioner_from_conditioning_config,
)

DEV = "cpu"
_EXP09_DIR = _REPO_ROOT / "worklog" / "worklog_yixun" / "exp_09_cyl_no_ssl"
_EXP09_JSON = _EXP09_DIR / "FLAC_AR_exp09.json"

_HIDDEN = 64           # tiny backbone width (production: 384)
_COND_DIM = 256        # FLAC cond_dim (production value, kept)
_PATCH = 16
_H_PX, _W_PX = 64, 128
_H_T, _W_T = _H_PX // _PATCH, _W_PX // _PATCH   # 4 x 8 tokens
_BOUND = 1e-4          # the registered equivariance bound (plan §2 / exp_01 A2b)
_BROKEN = 10 * _BOUND  # negative control: "unmistakably broken", never a fitted number
_REGISTERED_ANGLES_DEG = (90.0, 180.0, 270.0)
_OFFICIAL_HIDDEN = 384


# ------------------------------------------------------------------------------------ #
# fixtures / helpers
# ------------------------------------------------------------------------------------ #
def _official_path():
    """The official DINOv3 ViT-S/16 snapshot in the local HF cache, else skip.

    Resolved through ``HF_HUB_CACHE`` (so an exported ``HF_HOME`` is honoured) with the
    default user cache as a fallback."""
    roots = []
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        roots.append(HF_HUB_CACHE)
    except Exception:  # noqa: BLE001 - fall back to the default location
        pass
    roots.append(os.path.expanduser("~/.cache/huggingface/hub"))
    for root in roots:
        repo = os.path.join(root, "models--facebook--dinov3-vits16-pretrain-lvd1689m")
        for snap in sorted(glob.glob(os.path.join(repo, "snapshots", "*"))):
            if os.path.exists(os.path.join(snap, "config.json")):
                return snap
    pytest.skip("official DINOv3 checkpoint is not in the local HF cache")


def _tiny_cyl_config(**overrides) -> CylindricalDINOv3ViTConfig:
    kwargs = dict(
        hidden_size=_HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=4,      # head_dim 16 -> 4 azimuth harmonics
        intermediate_size=128,
        num_register_tokens=4,
        attention_dropout=0.0,      # eval-mode determinism
        drop_path_rate=0.0,
        # NOT the DINOv3 default 0.02, and load-bearing: at 0.02 the residual stream
        # dominates and is roll-equivariant on its own, so a gauge-OFF negative control
        # has no teeth (see the cylindrical package's conftest for the measured table).
        initializer_range=0.2,
    )
    kwargs.update(overrides)
    return CylindricalDINOv3ViTConfig(**kwargs)


def _save_tiny_cyl(dirpath, seed: int = 0, **cfg_overrides) -> str:
    cfg = _tiny_cyl_config(**cfg_overrides)
    cfg._attn_implementation = "eager"
    torch.manual_seed(seed)
    model = CylindricalDINOv3ViTModel(cfg)
    model.save_pretrained(str(dirpath))
    return str(dirpath)


@pytest.fixture(scope="module")
def tiny_cyl_dir(tmp_path_factory):
    return _save_tiny_cyl(tmp_path_factory.mktemp("exp03n_tiny_cyl"))


def _vit_block(vit_path, *, gauge="cylindrical_xyz", cond_pool=None, cond_mlp_hidden=None,
               implementation="cylindrical_dinov3"):
    block = {
        "hf_model_name_or_path": vit_path,
        "ch_dim": 3,
        "freeze": False,
        "from_scratch": False,
        "img_h": 256,
        "img_w": 512,
        "gauge": gauge,
    }
    if implementation is not None:
        block["implementation"] = implementation
    if cond_pool is not None:
        block["cond_pool"] = cond_pool
    if cond_mlp_hidden is not None:
        block["cond_mlp_hidden"] = cond_mlp_hidden
    return block


def _cyl_conditioning(vit_path, *, with_context=True, cond_dim=_COND_DIM,
                      second_block_overrides=None, **vit_kw):
    """The exp-09 conditioning subtree shape (source_vit [+ context_poses_vit])."""
    vit_block = _vit_block(vit_path, **vit_kw)
    configs = [
        {"id": "source_vit", "type": "ViTCoordinates",
         "config": {"ViT": copy.deepcopy(vit_block), "max_value": 1}},
    ]
    if with_context:
        second = copy.deepcopy(vit_block)
        if second_block_overrides is not None:
            second.update(second_block_overrides)
            for key, value in list(second_block_overrides.items()):
                if value is None:
                    second.pop(key, None)
        configs.append(
            {"id": "context_poses_vit", "type": "ViTCoordinates",
             "config": {"ViT": second, "max_value": 1}}
        )
    return {"configs": configs, "cond_dim": cond_dim}


def _geoms(mc):
    return [c for c in mc.conditioners.values()
            if getattr(c, "name", None) == "GeometryConditioner"]


def _build(cond_cfg, seed: int = 42):
    torch.manual_seed(seed)
    return create_multi_conditioner_from_conditioning_config(copy.deepcopy(cond_cfg)).eval()


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """A2b statistic: ||a - b||_F / (||b||_F + 1e-12), reduced in float64."""
    a64, b64 = a.detach().to(torch.float64), b.detach().to(torch.float64)
    return (torch.linalg.vector_norm(a64 - b64)
            / (torch.linalg.vector_norm(b64) + 1e-12)).item()


def _nonaxisymmetric_depth(H: int, W: int) -> torch.Tensor:
    """A geometrically consistent, deliberately NON-axisymmetric equirectangular XYZ
    depth field [3, H, W] (azimuth-dependent radius ⇒ a yaw genuinely changes it)."""
    j = torch.arange(W, dtype=torch.float32)
    theta = (j + 0.5) * 2.0 * math.pi / W
    i = torch.arange(H, dtype=torch.float32)
    el = (i + 0.5) * math.pi / H - math.pi / 2.0
    theta_g = theta.view(1, W).expand(H, W)
    el_g = el.view(H, 1).expand(H, W)
    d = 3.0 + 1.0 * torch.sin(theta_g) + 0.5 * torch.sin(2.0 * theta_g) + 0.25 * torch.cos(3.0 * theta_g)
    x = d * torch.cos(el_g) * torch.cos(theta_g)
    y = d * torch.cos(el_g) * torch.sin(theta_g)
    z = d * torch.sin(el_g)
    return torch.stack([x, y, z], dim=0).contiguous()


def _rz(alpha: float) -> torch.Tensor:
    c, s = math.cos(alpha), math.sin(alpha)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)


def _base_sample(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return {"coord": torch.randn(1, 3, generator=g), "depth": _nonaxisymmetric_depth(_H_PX, _W_PX)}


def _yaw_sample(sample, k_tokens: int):
    """Physically yaw a sample by ``k_tokens`` whole token columns: the XYZ depth field
    through the PACKAGE's ``physical_yaw`` (roll + Rz on x,y) and the pose by the same Rz."""
    alpha = 2.0 * math.pi * k_tokens / _W_T
    depth = physical_yaw(sample["depth"].unsqueeze(0), k_tokens * _PATCH).squeeze(0)
    coord = torch.einsum("ij,...j->...i", _rz(alpha), sample["coord"])
    return {"coord": coord, "depth": depth}


def _cond_field(sample, max_value: float = 1.0) -> torch.Tensor:
    """The [1, 3, H, W] tensor GeometryConditioner.forward feeds the backbone."""
    coord = sample["coord"].float().unsqueeze(0)          # [1, 1, 3]
    depth = sample["depth"].float().unsqueeze(0)          # [1, 3, H, W]
    return (coord[:, 0, :, None, None] - depth) / max_value


def _roll_tokens(tokens: torch.Tensor, k: int) -> torch.Tensor:
    b, n, c = tokens.shape
    assert n == _H_T * _W_T, (n, _H_T, _W_T)
    return torch.roll(tokens.view(b, _H_T, _W_T, c), shifts=k, dims=2).reshape(b, n, c)


def _legacy_forward_oracle(geom, coord_list):
    """A faithful re-implementation of the PRE-change dino forward path, written from the
    pre-change source (``pooled = outputs.pooler_output; c = lin_proj(pooled).unsqueeze(1)``).
    Used as the byte-identity fixture for the absent-key path — an independent oracle rather
    than a stored tensor, so the assertion is bitwise without being machine-specific."""
    coords = torch.stack([c["coord"].float() for c in coord_list], dim=0)
    depth = torch.stack([c["depth"].float() for c in coord_list], dim=0)
    if coords.ndim == 2:
        coords = coords.unsqueeze(1)
    outs = []
    for i in range(coords.shape[1]):
        c = (coords[:, i, :, None, None] - depth) / geom.max_value
        pooled = geom.vit(c).pooler_output
        outs.append(geom.lin_proj(pooled).unsqueeze(1))
    return torch.cat(outs, dim=1)


# ------------------------------------------------------------------------------------ #
# 1. config knob: absent key == the legacy mean+Linear head, byte-identical forward
# ------------------------------------------------------------------------------------ #
def test_absent_cond_pool_keeps_the_legacy_mean_linear_head(tiny_cyl_dir):
    mc = _build(_cyl_conditioning(tiny_cyl_dir, with_context=True))
    geoms = _geoms(mc)
    assert len(geoms) == 2
    for geom in geoms:
        assert geom.model_type == "dino"
        assert type(geom.lin_proj) is nn.Linear, type(geom.lin_proj)
        assert (geom.lin_proj.in_features, geom.lin_proj.out_features) == (_HIDDEN, _COND_DIM)
        assert getattr(geom, "dino_pool", "mean") == "mean"


def test_absent_cond_pool_forward_is_bitwise_the_pre_change_arithmetic(tiny_cyl_dir):
    geom = _geoms(_build(_cyl_conditioning(tiny_cyl_dir, with_context=False)))[0]
    batch = [_base_sample(0), _base_sample(1)]
    with torch.no_grad():
        got = geom(batch, device=DEV)[0]
        want = _legacy_forward_oracle(geom, batch)
    assert got.shape == want.shape == (2, 1, _COND_DIM)
    assert torch.equal(got, want), (
        f"legacy path drifted from the pre-change arithmetic (max abs "
        f"{(got - want).abs().max().item():.3e})"
    )


# ------------------------------------------------------------------------------------ #
# 2. config knob: "max_mlp" builds the declared head, shared by BOTH conditioners
# ------------------------------------------------------------------------------------ #
def test_max_mlp_builds_the_declared_head_shapes(tiny_cyl_dir):
    mc = _build(_cyl_conditioning(tiny_cyl_dir, with_context=True,
                                  cond_pool="max_mlp", cond_mlp_hidden=_HIDDEN))
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
        assert geom.dino_pool == "max"
        assert geom.model_type == "dino"


def test_max_mlp_default_width_is_the_backbone_width(tiny_cyl_dir):
    """``cond_mlp_hidden`` is optional: absent ⇒ the backbone width (384 in production,
    which is the plan's declared default; the fixture's 64 here)."""
    geom = _geoms(_build(_cyl_conditioning(tiny_cyl_dir, with_context=False,
                                           cond_pool="max_mlp")))[0]
    assert (geom.lin_proj[0].in_features, geom.lin_proj[0].out_features) == (_HIDDEN, _HIDDEN)


def test_max_mlp_forward_is_the_mlp_of_the_token_amax(tiny_cyl_dir):
    geom = _geoms(_build(_cyl_conditioning(tiny_cyl_dir, with_context=False,
                                           cond_pool="max_mlp")))[0]
    batch = [_base_sample(0), _base_sample(1)]
    with torch.no_grad():
        got = geom(batch, device=DEV)[0]
        field = torch.cat([_cond_field(s, geom.max_value) for s in batch], dim=0)
        tokens = geom.vit(field).last_hidden_state
        want = geom.lin_proj(tokens.amax(dim=1)).unsqueeze(1)
    assert tokens.shape == (2, _H_T * _W_T, _HIDDEN)
    assert got.shape == (2, 1, _COND_DIM)
    assert torch.equal(got, want), (
        f"served condition != MLP(amax(tokens)) (max abs {(got - want).abs().max().item():.3e})"
    )


# ------------------------------------------------------------------------------------ #
# 3. fail-closed knob validation
# ------------------------------------------------------------------------------------ #
@pytest.mark.parametrize("value", ["max", "mean_mlp", "maxmlp", "MAX_MLP", "", True, 1])
def test_unknown_cond_pool_value_raises(tiny_cyl_dir, value):
    cfg = _cyl_conditioning(tiny_cyl_dir, with_context=False, cond_pool=value)
    with pytest.raises(ValueError, match="cond_pool"):
        create_multi_conditioner_from_conditioning_config(cfg)


def test_orphan_cond_mlp_hidden_without_cond_pool_raises(tiny_cyl_dir):
    cfg = _cyl_conditioning(tiny_cyl_dir, with_context=False, cond_mlp_hidden=_HIDDEN)
    with pytest.raises(ValueError, match="cond_mlp_hidden"):
        create_multi_conditioner_from_conditioning_config(cfg)


@pytest.mark.parametrize("width", [0, -1, -384, True, False, 3.5, "64", [64]])
def test_invalid_cond_mlp_hidden_width_raises(tiny_cyl_dir, width):
    cfg = _cyl_conditioning(tiny_cyl_dir, with_context=False, cond_pool="max_mlp")
    cfg["configs"][0]["config"]["ViT"]["cond_mlp_hidden"] = width
    with pytest.raises(ValueError, match="cond_mlp_hidden"):
        create_multi_conditioner_from_conditioning_config(cfg)


def test_cond_mlp_hidden_other_than_the_backbone_width_raises(tiny_cyl_dir):
    """PLAN-NOTE (minimal fail-closed reading of plan §2.1): the output layer is the
    LEGACY ``Linear(hidden_size, cond_dim)`` drawn at the legacy code point (that is what
    makes it bitwise-equal to legacy and keeps the downstream RNG stream intact), so the
    hidden layer must map hidden_size → hidden_size for the Sequential to compose at all.
    Any other width is refused at CONSTRUCTION rather than blowing up in a forward."""
    cfg = _cyl_conditioning(tiny_cyl_dir, with_context=False, cond_pool="max_mlp",
                            cond_mlp_hidden=_HIDDEN * 2)
    with pytest.raises(ValueError, match="cond_mlp_hidden"):
        create_multi_conditioner_from_conditioning_config(cfg)


def test_unequal_vit_blocks_trip_the_shared_backbone_guard(tiny_cyl_dir):
    """The knob must be equal on BOTH ViT blocks: the existing cylindrical block-equality
    guard is what enforces it (the factory would otherwise silently discard the 2nd block)."""
    cfg = _cyl_conditioning(tiny_cyl_dir, with_context=True, cond_pool="max_mlp",
                            cond_mlp_hidden=_HIDDEN,
                            second_block_overrides={"cond_pool": None, "cond_mlp_hidden": None})
    with pytest.raises(ValueError, match="differ"):
        create_multi_conditioner_from_conditioning_config(cfg)


def test_unequal_vit_blocks_width_only_also_trips_the_guard(tiny_cyl_dir):
    cfg = _cyl_conditioning(tiny_cyl_dir, with_context=True, cond_pool="max_mlp",
                            cond_mlp_hidden=_HIDDEN,
                            second_block_overrides={"cond_mlp_hidden": _HIDDEN * 2})
    with pytest.raises(ValueError, match="differ"):
        create_multi_conditioner_from_conditioning_config(cfg)


# ------------------------------------------------------------------------------------ #
# 4. common-init identity (plan §2.1 / r2-F1 / r3-N5)
# ------------------------------------------------------------------------------------ #
def _full_model_config(tiny_cyl_dir, *, max_mlp: bool):
    with open(_EXP09_JSON) as f:
        cfg = json.load(f)
    for c in cfg["model"]["conditioning"]["configs"]:
        if c["type"] == "ViTCoordinates":
            block = c["config"]["ViT"]
            block["hf_model_name_or_path"] = tiny_cyl_dir
            if max_mlp:
                block["cond_pool"] = "max_mlp"
                block["cond_mlp_hidden"] = _HIDDEN
    cfg["model"]["diffusion"]["config"]["depth"] = 2   # shrink the DiT for a fast CPU build
    return cfg


def _full_model(tiny_cyl_dir, *, max_mlp: bool, seed: int = 42):
    from src.models import create_model_from_config

    torch.manual_seed(seed)
    return create_model_from_config(_full_model_config(tiny_cyl_dir, max_mlp=max_mlp))


def test_common_init_is_bitwise_identical_including_the_mapped_output_layer(tiny_cyl_dir):
    """Full-model construction under seed 42: every tensor the two arms share must be
    BITWISE equal (so the ablation is one-factor), and the renamed output layer
    ``lin_proj.2.*`` must equal the legacy ``lin_proj.*`` for BOTH conditioner aliases —
    a bare key intersection would silently skip exactly that tensor."""
    legacy = _full_model(tiny_cyl_dir, max_mlp=False).state_dict()
    new = _full_model(tiny_cyl_dir, max_mlp=True).state_dict()

    aliases = ["conditioner.conditioners.source_vit",
               "conditioner.conditioners.context_poses_vit"]
    only_new = set(new) - set(legacy)
    only_legacy = set(legacy) - set(new)
    assert only_new == {f"{a}.lin_proj.{i}.{t}" for a in aliases for i in (0, 2) for t in ("weight", "bias")}, (
        sorted(only_new)
    )
    assert only_legacy == {f"{a}.lin_proj.{t}" for a in aliases for t in ("weight", "bias")}, (
        sorted(only_legacy)
    )

    for key in sorted(set(legacy) & set(new)):
        assert torch.equal(legacy[key], new[key]), f"common tensor {key} is NOT bitwise identical"

    # the mapped output layer: legacy .lin_proj.{weight,bias} == new .lin_proj.2.{weight,bias}
    for alias in aliases:
        for tensor in ("weight", "bias"):
            assert torch.equal(legacy[f"{alias}.lin_proj.{tensor}"],
                               new[f"{alias}.lin_proj.2.{tensor}"]), (
                f"{alias}: the max_mlp OUTPUT layer is not bitwise-equal to the legacy projection"
            )
    # only the hidden layer is new
    assert {k for k in only_new if ".lin_proj.0." in k}, "no hidden-layer tensors appeared"


def test_hidden_layer_init_is_rng_isolated_and_reproducible(tiny_cyl_dir):
    """The hidden layer is drawn inside ``fork_rng`` from a PINNED CPU-generator seed, so
    (a) it is identical across builds at different global seeds, and (b) the global RNG
    stream is untouched by it — proven by the bitwise common-init test above plus this
    draw-after-build check."""
    cfg = _cyl_conditioning(tiny_cyl_dir, with_context=False, cond_pool="max_mlp")
    head_a = _geoms(_build(cfg, seed=42))[0].lin_proj
    head_b = _geoms(_build(cfg, seed=1234))[0].lin_proj
    assert torch.equal(head_a[0].weight, head_b[0].weight), (
        "hidden-layer init is not RNG-isolated (it moved with the global seed)"
    )
    assert torch.equal(head_a[0].bias, head_b[0].bias)
    assert not torch.equal(head_a[2].weight, head_b[2].weight), (
        "the OUTPUT layer must still follow the global seed (it is the legacy draw)"
    )

    # the global stream after construction is the legacy stream (no leaked draws)
    torch.manual_seed(7)
    create_multi_conditioner_from_conditioning_config(copy.deepcopy(cfg))
    after_max_mlp = torch.randn(4)
    torch.manual_seed(7)
    create_multi_conditioner_from_conditioning_config(
        copy.deepcopy(_cyl_conditioning(tiny_cyl_dir, with_context=False))
    )
    after_legacy = torch.randn(4)
    assert torch.equal(after_max_mlp, after_legacy), (
        "the max_mlp build consumed a different number of GLOBAL RNG draws than legacy"
    )


# ------------------------------------------------------------------------------------ #
# 5. parameter-count oracle
# ------------------------------------------------------------------------------------ #
def _n_trainable(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def test_trainable_param_delta_is_exactly_the_hidden_layer(tiny_cyl_dir):
    legacy = _build(_cyl_conditioning(tiny_cyl_dir, with_context=True))
    new = _build(_cyl_conditioning(tiny_cyl_dir, with_context=True,
                                   cond_pool="max_mlp", cond_mlp_hidden=_HIDDEN))
    assert _n_trainable(new) - _n_trainable(legacy) == _HIDDEN * _HIDDEN + _HIDDEN


def test_official_head_is_384_384_256_and_the_delta_is_147840():
    """The production oracle (plan §4.3): at the REAL ViT-S/16 the head is exactly
    Linear(384,384) → GELU → Linear(384,256) and the trainable-parameter delta is
    +147,840 — the hidden layer ALONE (the output layer replaces, and is bitwise-equal
    to, the legacy 98,560-param projection)."""
    official = _official_path()
    legacy = _build(_cyl_conditioning(official, with_context=True))
    new = _build(_cyl_conditioning(official, with_context=True,
                                   cond_pool="max_mlp", cond_mlp_hidden=_OFFICIAL_HIDDEN))
    head = _geoms(new)[0].lin_proj
    assert (head[0].in_features, head[0].out_features) == (384, 384)
    assert type(head[1]) is nn.GELU
    assert (head[2].in_features, head[2].out_features) == (384, 256)
    assert _n_trainable(new) - _n_trainable(legacy) == 147840


# ------------------------------------------------------------------------------------ #
# 6. max-pool invariance under physical yaw (plan §4.6 / F2), end-to-end
# ------------------------------------------------------------------------------------ #
def _invariance_residuals(geom, *, angles=_REGISTERED_ANGLES_DEG):
    base = _base_sample(0)
    residuals = {}
    with torch.no_grad():
        cond_base = geom([base], device=DEV)[0]
    for deg in angles:
        k = int(round(deg / 360.0 * _W_T))
        rot = _yaw_sample(base, k)
        with torch.no_grad():
            cond_rot = geom([rot], device=DEV)[0]
        residuals[deg] = _rel_l2(cond_rot, cond_base)
    return residuals


def test_the_yaw_fixture_actually_changes_the_backbone_input():
    """Non-vacuity (i): the yawed input must genuinely differ from the base input, and it
    must be EXACTLY the package's ``physical_yaw`` of it (the harness is faithful)."""
    base = _base_sample(0)
    field = _cond_field(base)
    for deg in _REGISTERED_ANGLES_DEG:
        k = int(round(deg / 360.0 * _W_T))
        rot_field = _cond_field(_yaw_sample(base, k))
        assert not torch.allclose(rot_field, field, atol=1e-3), (
            f"the {deg}-deg yawed input is indistinguishable from the base input — the "
            "invariance fixture would be vacuous (is the depth field axisymmetric?)"
        )
        assert torch.allclose(rot_field, physical_yaw(field, k * _PATCH), atol=1e-5), (
            f"the {deg}-deg fixture is not the package's physical_yaw of the base field"
        )


def test_backbone_tokens_undergo_the_expected_column_roll(tiny_cyl_dir):
    """Non-vacuity (ii): under the gauge, a physical yaw of k token columns permutes the
    token field by exactly that column roll — the premise max-pool invariance rests on."""
    geom = _geoms(_build(_cyl_conditioning(tiny_cyl_dir, with_context=False,
                                           cond_pool="max_mlp")))[0]
    base = _base_sample(0)
    with torch.no_grad():
        tokens = geom.vit(_cond_field(base, geom.max_value)).last_hidden_state
    for deg in _REGISTERED_ANGLES_DEG:
        k = int(round(deg / 360.0 * _W_T))
        with torch.no_grad():
            rolled = geom.vit(_cond_field(_yaw_sample(base, k), geom.max_value)).last_hidden_state
        res = _rel_l2(rolled, _roll_tokens(tokens, k))
        assert res <= _BOUND, f"token field at {deg} deg is not the k={k} column roll (rel-L2 {res:.3e})"


def test_maxpool_condition_is_invariant_at_the_registered_angles(tiny_cyl_dir):
    """(iii) the SERVED condition — GeometryConditioner.forward end-to-end, i.e.
    MLP(amax(tokens)) — is invariant to physical yaw within the registered 1e-4 bound."""
    geom = _geoms(_build(_cyl_conditioning(tiny_cyl_dir, with_context=False,
                                           cond_pool="max_mlp")))[0]
    residuals = _invariance_residuals(geom)
    worst = max(residuals.values())
    assert worst <= _BOUND, f"max_mlp condition not yaw-invariant: {residuals}"
    print(f"\n[exp03n] max_mlp condition rel-L2 residuals: "
          f"{ {k: f'{v:.3e}' for k, v in residuals.items()} }")


def test_gauge_off_negative_control_violates_invariance(tiny_cyl_dir):
    """NEGATIVE control: with the gauge DISABLED the same fixture must break invariance by
    at least 10x the bound — otherwise the positive test above proves nothing."""
    geom = _geoms(_build(_cyl_conditioning(tiny_cyl_dir, with_context=False,
                                           cond_pool="max_mlp", gauge="none")))[0]
    residuals = _invariance_residuals(geom)
    worst = max(residuals.values())
    assert worst >= _BROKEN, (
        f"gauge-OFF control did NOT break invariance (worst rel-L2 {worst:.3e} < {_BROKEN:.0e}) — "
        "the invariance assertion has no teeth"
    )


# ------------------------------------------------------------------------------------ #
# 7. gradients (plan §2 N8)
# ------------------------------------------------------------------------------------ #
def test_grads_reach_both_mlp_layers_and_the_backbone_through_both_conditioners(tiny_cyl_dir):
    mc = _build(_cyl_conditioning(tiny_cyl_dir, with_context=True,
                                  cond_pool="max_mlp", cond_mlp_hidden=_HIDDEN))
    geoms = _geoms(mc)
    head = geoms[0].lin_proj
    backbone_param = geoms[0].vit.layer[0].mlp.up_proj.weight
    assert backbone_param.requires_grad

    mc.train()
    batch = [_base_sample(0)]
    loss = sum(geom(batch, device=DEV)[0].float().pow(2).mean() for geom in geoms)
    loss.backward()
    try:
        for name, param in (("hidden", head[0]), ("output", head[2])):
            for tensor in ("weight", "bias"):
                grad = getattr(param, tensor).grad
                assert grad is not None, f"{name}.{tensor} got no grad"
                assert torch.isfinite(grad).all(), f"{name}.{tensor} grad is not finite"
                assert grad.abs().max().item() > 0.0, f"{name}.{tensor} grad is all zeros"
        assert backbone_param.grad is not None, "the shared backbone got no grad"
        assert torch.isfinite(backbone_param.grad).all()
        assert backbone_param.grad.abs().max().item() > 0.0
    finally:
        mc.zero_grad(set_to_none=True)
        mc.eval()
