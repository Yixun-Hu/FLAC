"""exp-09 Stage B integration tests: the ``cylindrical_dinov3`` conditioner branch.

These pin the FLAC-side wiring the plan §2 prescribes, on CPU, with tiny saved
cylindrical backbones where possible and the official DINOv3 ViT-S/16 snapshot
(read-only, from the local HF cache) for the two contract tests that need real
384-d weights. Everything here is CPU-only (``CUDA_VISIBLE_DEVICES=""`` set below
before torch initialises) and offline (``HF_HUB_OFFLINE=1``).

Surfaces under test (plan §2 + the Stage-B deliverable):

* branch construction returns the exact custom class, with the XYZ gauge module
  present and eager attention pinned;
* fail-closed: an unknown ``implementation`` raises (never falls through to
  ``AutoModel``); ``from_scratch: true`` with this implementation raises;
* the ``gauge`` default is ``"cylindrical_xyz"`` (audit-blessed gauge-ON), NOT
  the package default ``"none"``;
* an ``implementation``-absent config takes the byte-identical legacy path (the
  vit is the plain ``AutoModel`` class);
* pooler contract: ``pooler_output`` equals the patch-token mean at hidden 384;
* shared-backbone semantics (source_vit-first; context_poses_vit reuses the SAME
  object; the two ViT blocks are equal);
* grad-checkpointing on the ``.layer`` ModuleList (construct + one fwd/bwd);
* complete-conditioning invariance at 45° under fa_invariant[0.0] (named test i);
* raw-pose negative control at 45° under vanilla (named test ii);
* backbone-forward counter == vanilla count for identical metadata (no extra
  frame-average passes);
* strict-load round-trip + eval-style ``check_load_integrity`` + resume-load and
  eval-checkpoint load end-to-end on the full exp-09 model;
* AGREE isolation: the retrieval-metric config subtree is untouched by the swap;
* the FLAC_AR_exp09.json config-delta vs FLAC_AR_BF.json is exactly the
  registered set.
"""
import os
import sys

# CPU-only + offline, hard-set BEFORE torch initialises CUDA (mandate; also dodges
# the deepspeed op-probe that save_pretrained triggers in this env when CUDA is
# visible-but-unusable). Set, not setdefault: an exported CUDA_VISIBLE_DEVICES=0
# must not disarm the guard.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import copy
import glob
import json
import math
from pathlib import Path

import pytest
import torch
import torch.utils.checkpoint
from torch import nn

# tests/ -> exp_09_cyl_no_ssl -> worklog_yixun -> worklog -> worktree root
_WORKTREE_ROOT = Path(__file__).resolve().parents[4]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from cylindrical_dinov3 import (  # noqa: E402
    CylindricalDINOv3ViTConfig,
    CylindricalDINOv3ViTModel,
    CylindricalXYZGauge,
)

from src.data import yaw_rotation as yr  # noqa: E402
from src.models.conditioners import (  # noqa: E402
    GeometryConditioner,
    _checkpoint_vit_layers,
    create_multi_conditioner_from_conditioning_config,
)

DEV = "cpu"
SEED = 0

_EXP07 = _WORKTREE_ROOT / "worklog" / "worklog_yixun" / "exp_07_fa_scratch_claude"
_BF_JSON = _EXP07 / "FLAC_AR_BF.json"
_EXP09_DIR = _WORKTREE_ROOT / "worklog" / "worklog_yixun" / "exp_09_cyl_no_ssl"
_EXP09_JSON = _EXP09_DIR / "FLAC_AR_exp09.json"

_OFFICIAL_REPO_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--facebook--dinov3-vits16-pretrain-lvd1689m"
)
_OFFICIAL_HIDDEN = 384  # ViT-S/16 -> FLAC's Linear(384, 256)


# ------------------------------------------------------------------------------------- #
# fixtures / helpers
# ------------------------------------------------------------------------------------- #
def _official_path() -> str:
    for snap in sorted(glob.glob(os.path.join(_OFFICIAL_REPO_DIR, "snapshots", "*"))):
        if os.path.exists(os.path.join(snap, "config.json")):
            return snap
    pytest.skip(f"official DINOv3 checkpoint not in local HF cache: {_OFFICIAL_REPO_DIR}")


def _make_tiny_cyl_config(**overrides) -> CylindricalDINOv3ViTConfig:
    kwargs = dict(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,  # head_dim 16 -> 4 azimuth harmonics
        intermediate_size=128,
        num_register_tokens=4,
        attention_dropout=0.0,
        drop_path_rate=0.0,
    )
    kwargs.update(overrides)
    return CylindricalDINOv3ViTConfig(**kwargs)


def _save_tiny_cyl(dirpath, seed: int = SEED, **cfg_overrides) -> str:
    """Save a tiny cylindrical model to ``dirpath`` (config gauge stays 'none' on
    disk; the FLAC branch supplies gauge= at from_pretrained, so this also
    exercises the load-time gauge override)."""
    cfg = _make_tiny_cyl_config(**cfg_overrides)
    cfg._attn_implementation = "eager"
    torch.manual_seed(seed)
    model = CylindricalDINOv3ViTModel(cfg)
    model.save_pretrained(dirpath)
    return str(dirpath)


@pytest.fixture(scope="module")
def tiny_cyl_dir(tmp_path_factory):
    return _save_tiny_cyl(tmp_path_factory.mktemp("tiny_cyl"))


def _vit_block(vit_path, *, implementation="cylindrical_dinov3", gauge="cylindrical_xyz",
               include_gauge=True, from_scratch=False, freeze=False, ch_dim=3):
    block = {
        "hf_model_name_or_path": vit_path,
        "ch_dim": ch_dim,
        "freeze": freeze,
        "from_scratch": from_scratch,
        "img_h": 256,
        "img_w": 512,
    }
    if implementation is not None:
        block["implementation"] = implementation
    if include_gauge:
        block["gauge"] = gauge
    return block


def _cyl_conditioning(vit_path, *, with_context=True, gradient_checkpointing=False,
                      cond_dim=256, **vit_kw):
    vit_block = _vit_block(vit_path, **vit_kw)
    configs = [
        {"id": "source", "type": "dist_embedder",
         "config": {"num_freqs": 20, "max_freq": 10, "ch_dim": 1, "include_in": True}},
        {"id": "source_vit", "type": "ViTCoordinates",
         "config": {"ViT": copy.deepcopy(vit_block), "max_value": 1,
                    "gradient_checkpointing": gradient_checkpointing}},
    ]
    if with_context:
        configs += [
            {"id": "context_poses_vit", "type": "ViTCoordinates",
             "config": {"ViT": copy.deepcopy(vit_block), "max_value": 1,
                        "gradient_checkpointing": gradient_checkpointing}},
            {"id": "context_poses", "type": "dist_embedder",
             "config": {"num_freqs": 20, "max_freq": 10, "ch_dim": 1, "include_in": True}},
            {"id": "context_audio", "type": "rir",
             "config": {"in_channels": 1, "n_fft": 124, "win_length": 31,
                        "hop_length": 62, "project_out": True}},
        ]
    return {"configs": configs, "cond_dim": cond_dim}


def _geoms(mc):
    return [c for c in mc.conditioners.values()
            if getattr(c, "name", None) == "GeometryConditioner"]


def _consistent_depth(H: int, W: int) -> torch.Tensor:
    """Geometrically consistent, non-axisymmetric equirectangular depth (adapted
    from src/tests/test_invariant_conditioning)."""
    j = torch.arange(W, dtype=torch.float32)
    theta = (j + 0.5) * 2.0 * math.pi / W - math.pi
    i = torch.arange(H, dtype=torch.float32)
    el = (i + 0.5) * math.pi / H - math.pi / 2.0
    theta_g = theta.view(1, W).expand(H, W)
    el_g = el.view(H, 1).expand(H, W)
    d = 3.0 + 1.0 * torch.sin(theta_g) + 0.5 * torch.sin(2.0 * theta_g)
    x = d * torch.cos(el_g) * torch.cos(theta_g)
    y = d * torch.cos(el_g) * torch.sin(theta_g)
    z = d * torch.sin(el_g)
    return torch.stack([x, y, z], dim=0).contiguous()


def _synthetic_sample(seed: int, *, H: int, W: int, n_context: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "source": torch.randn(3, generator=g),
        "source_vit": torch.randn(1, 3, generator=g),
        "context_poses": torch.randn(n_context, 3, generator=g),
        "context_poses_vit": torch.randn(n_context, 3, generator=g),
        "context_audio": torch.randn(n_context, 1, 256, generator=g),
        "depth": _consistent_depth(H, W),
    }


def _synthetic_batch(n: int = 2, *, H: int = 32, W: int = 128, n_context: int = 3) -> list:
    # W=128 -> a 45 deg yaw is round(128/8)=16 px = one patch column (patch 16),
    # i.e. an EXACT token roll the gauge backbone is invariant to.
    return [_synthetic_sample(s, H=H, W=W, n_context=n_context) for s in range(n)]


# ------------------------------------------------------------------------------------- #
# 1. branch construction -> exact class, gauge module, eager attention
# ------------------------------------------------------------------------------------- #
def test_branch_constructs_the_exact_custom_class_with_gauge_and_eager(tiny_cyl_dir):
    mc = create_multi_conditioner_from_conditioning_config(
        _cyl_conditioning(tiny_cyl_dir, with_context=False)
    )
    vit = _geoms(mc)[0].vit
    assert type(vit) is CylindricalDINOv3ViTModel, type(vit)
    assert isinstance(vit.gauge, CylindricalXYZGauge), "gauge module absent/wrong type"
    assert vit.config.gauge == "cylindrical_xyz"
    assert vit.config._attn_implementation == "eager"
    # model_type='dino' and lin_proj Linear(hidden, cond_dim) is what feeds pooler_output
    geom = _geoms(mc)[0]
    assert geom.model_type == "dino"
    assert isinstance(geom.lin_proj, nn.Linear)
    assert geom.lin_proj.in_features == vit.config.hidden_size
    assert geom.lin_proj.out_features == 256


# ------------------------------------------------------------------------------------- #
# 2. fail-closed: unknown implementation raises (never falls through to AutoModel)
#    [mutation-sweep (a)]
# ------------------------------------------------------------------------------------- #
def test_unknown_implementation_raises_and_never_falls_through(tiny_cyl_dir):
    cfg = _cyl_conditioning(tiny_cyl_dir, with_context=False, implementation="cylndrical_dinov3")
    with pytest.raises(ValueError, match="[Ii]mplementation"):
        create_multi_conditioner_from_conditioning_config(cfg)


# ------------------------------------------------------------------------------------- #
# 3. fail-closed: from_scratch:true with this implementation raises
# ------------------------------------------------------------------------------------- #
def test_from_scratch_true_is_rejected_for_cylindrical(tiny_cyl_dir):
    cfg = _cyl_conditioning(tiny_cyl_dir, with_context=False, from_scratch=True)
    with pytest.raises(ValueError, match="from_scratch"):
        create_multi_conditioner_from_conditioning_config(cfg)


# ------------------------------------------------------------------------------------- #
#    gauge default is 'cylindrical_xyz' when the config omits it  [mutation-sweep (b)]
# ------------------------------------------------------------------------------------- #
def test_gauge_defaults_to_cylindrical_xyz_when_absent(tiny_cyl_dir):
    """Audit-blessed gauge-ON: a FLAC ViT block with NO 'gauge' key must build a
    gauge-ON backbone, never the package's own 'none' default."""
    cfg = _cyl_conditioning(tiny_cyl_dir, with_context=False, include_gauge=False)
    vit = _geoms(create_multi_conditioner_from_conditioning_config(cfg))[0].vit
    assert vit.config.gauge == "cylindrical_xyz", vit.config.gauge
    assert isinstance(vit.gauge, CylindricalXYZGauge)


def test_explicit_gauge_none_is_honoured(tiny_cyl_dir):
    """A caller that explicitly asks for gauge='none' gets it (the default is a
    default, not a hard override) — proves the default test above is not vacuous."""
    cfg = _cyl_conditioning(tiny_cyl_dir, with_context=False, gauge="none")
    vit = _geoms(create_multi_conditioner_from_conditioning_config(cfg))[0].vit
    assert vit.config.gauge == "none"
    assert not hasattr(vit, "gauge")


# ------------------------------------------------------------------------------------- #
# 4. absent implementation -> byte-identical legacy path (plain AutoModel class)
# ------------------------------------------------------------------------------------- #
def test_absent_implementation_takes_the_legacy_automodel_path():
    """A legacy HF config (no 'implementation' key) must build the plain
    transformers AutoModel class, not the cylindrical subclass — the new field is
    the ONLY thing that routes into the exp-09 branch."""
    official = _official_path()
    cfg = _cyl_conditioning(official, with_context=False, implementation=None, include_gauge=False)
    vit = _geoms(create_multi_conditioner_from_conditioning_config(cfg))[0].vit
    assert type(vit).__name__ == "DINOv3ViTModel", type(vit)
    assert not isinstance(vit, CylindricalDINOv3ViTModel)


def test_absent_implementation_simplevit_legacy_is_untouched():
    """The non-HF legacy branches are equally undisturbed: an arch-less, hf-path-less
    ViT block still builds the in-repo SimpleViT (cheap; no checkpoint needed)."""
    from src.models.simplevit import SimpleViT
    cfg = {
        "configs": [{
            "id": "source_vit", "type": "ViTCoordinates",
            "config": {"ViT": {"img_h": 32, "img_w": 32, "patch_h": 16, "patch_w": 16,
                               "dim": 32, "depth": 1, "heads": 2}, "max_value": 1},
        }],
        "cond_dim": 256,
    }
    vit = _geoms(create_multi_conditioner_from_conditioning_config(cfg))[0].vit
    assert isinstance(vit, SimpleViT)


# ------------------------------------------------------------------------------------- #
# 5. pooler contract: pooler_output == patch-token mean, hidden 384  (official)
# ------------------------------------------------------------------------------------- #
def test_pooler_output_is_patch_mean_at_hidden_384():
    official = _official_path()
    mc = create_multi_conditioner_from_conditioning_config(
        _cyl_conditioning(official, with_context=False)
    )
    vit = _geoms(mc)[0].vit.eval()
    assert vit.config.hidden_size == _OFFICIAL_HIDDEN
    x = torch.randn(2, 3, 256, 512)
    with torch.no_grad():
        out = vit(x)
    assert out.pooler_output.shape == (2, _OFFICIAL_HIDDEN)
    assert torch.allclose(
        out.pooler_output, out.last_hidden_state.mean(dim=1), atol=1e-6
    ), "pooler_output is not the patch-token mean"


# ------------------------------------------------------------------------------------- #
# 6. shared-backbone semantics: production builds ONE shared backbone object.
#    (Config block-equality is asserted separately against the REAL JSON — see
#    test_exp09_two_vit_blocks_are_deep_equal_and_exact_in_the_real_json — NOT against a
#    test-manufactured deep copy, which would be vacuous.)
# ------------------------------------------------------------------------------------- #
def test_shared_backbone_is_one_object(tiny_cyl_dir):
    cond_cfg = _cyl_conditioning(tiny_cyl_dir, with_context=True)
    geoms = _geoms(create_multi_conditioner_from_conditioning_config(copy.deepcopy(cond_cfg)))
    assert len(geoms) == 2
    assert geoms[0].vit is geoms[1].vit, "the two ViT conditioners do not share one backbone"


# ------------------------------------------------------------------------------------- #
# 6b. dispatcher defense-in-depth (Codex blocker 2c): a cylindrical build with a
#     DIVERGENT second ViT block must raise — the factory reuses the first backbone
#     and would otherwise silently ignore the mismatch. Legacy configs are untouched.
# ------------------------------------------------------------------------------------- #
def test_dispatcher_rejects_divergent_second_cyl_block(tiny_cyl_dir):
    cond_cfg = _cyl_conditioning(tiny_cyl_dir, with_context=True)
    ctx_vit = cond_cfg["configs"][2]["config"]["ViT"]
    # the reviewer's exact bypass shape: swap implementation<->gauge on the 2nd block
    ctx_vit["implementation"], ctx_vit["gauge"] = "cylindrical_xyz", "cylindrical_dinov3"
    with pytest.raises(ValueError, match="differ"):
        create_multi_conditioner_from_conditioning_config(cond_cfg)


def test_dispatcher_allows_equal_second_cyl_block(tiny_cyl_dir):
    """Control: equal second block builds fine (no false positive), one shared backbone."""
    mc = create_multi_conditioner_from_conditioning_config(
        _cyl_conditioning(tiny_cyl_dir, with_context=True)
    )
    geoms = _geoms(mc)
    assert geoms[0].vit is geoms[1].vit


def test_dispatcher_equality_guard_is_inert_for_legacy_configs():
    """The guard is gated on the cylindrical branch only: a LEGACY (SimpleViT) config
    whose second ViT block DIFFERS still builds — the factory reuses the first backbone
    and byte-identical legacy behavior is preserved (the second block was always
    discarded)."""
    def _sv_block(dim):
        return {"img_h": 32, "img_w": 32, "patch_h": 16, "patch_w": 16,
                "dim": dim, "depth": 1, "heads": 2}
    cfg = {
        "configs": [
            {"id": "source_vit", "type": "ViTCoordinates",
             "config": {"ViT": _sv_block(32), "max_value": 1}},
            {"id": "context_poses_vit", "type": "ViTCoordinates",
             "config": {"ViT": _sv_block(64), "max_value": 1}},  # DIFFERENT dim
        ],
        "cond_dim": 256,
    }
    mc = create_multi_conditioner_from_conditioning_config(cfg)  # must NOT raise
    geoms = _geoms(mc)
    assert geoms[0].vit is geoms[1].vit


# ------------------------------------------------------------------------------------- #
# 6c. screen-script contract (Codex blocker 1): both eval invocations MUST select
#     fa_invariant[0] explicitly — eval_FLAC.py defaults to 'vanilla' and does NOT
#     read cond_method from the JSON, so an omitted flag screens the raw-pose trap.
# ------------------------------------------------------------------------------------- #
def _screen_eval_invocations():
    text = (_EXP09_DIR / "exp09_screen.sh").read_text()
    joined = text.replace("\\\n", " ")  # collapse shell line-continuations
    return [" ".join(line.split()) for line in joined.splitlines()
            if "python eval_FLAC.py" in line]


def _flag_value(tokens, flag):
    return tokens[tokens.index(flag) + 1] if flag in tokens else None


def test_screen_script_selects_fa_invariant_on_both_evals():
    invs = _screen_eval_invocations()
    assert len(invs) == 2, f"expected 2 eval_FLAC.py invocations, got {len(invs)}: {invs}"
    for inv in invs:
        tokens = inv.split()
        assert _flag_value(tokens, "--cond-method") == "fa_invariant", (
            f"screen eval omits/mis-sets --cond-method fa_invariant: {inv}"
        )
        assert _flag_value(tokens, "--frame-avg-angles") == "0", (
            f"screen eval omits/mis-sets --frame-avg-angles 0: {inv}"
        )


# ------------------------------------------------------------------------------------- #
# 7. gradient checkpointing on our .layer ModuleList  (construct + fwd/bwd)
# ------------------------------------------------------------------------------------- #
class _CountingCheckpoint:
    def __init__(self, real):
        self.real = real
        self.use_reentrant_seen = []

    def __call__(self, fn, *args, **kwargs):
        self.use_reentrant_seen.append(kwargs.get("use_reentrant", "ABSENT"))
        return self.real(fn, *args, **kwargs)

    @property
    def count(self):
        return len(self.use_reentrant_seen)


def test_checkpoint_vit_layers_runs_on_our_model(tiny_cyl_dir):
    cfg = _make_tiny_cyl_config()
    cfg._attn_implementation = "eager"
    model = CylindricalDINOv3ViTModel(cfg)
    n = _checkpoint_vit_layers(model)
    assert n == len(model.layer) == 2
    for layer in model.layer:
        assert getattr(layer, "_flac_ckpt_wrapped", False) is True


def test_gradient_checkpointing_forward_backward_on_tiny_cyl(tiny_cyl_dir, monkeypatch):
    mc = create_multi_conditioner_from_conditioning_config(
        _cyl_conditioning(tiny_cyl_dir, with_context=False, gradient_checkpointing=True)
    )
    geom = _geoms(mc)[0]
    vit = geom.vit
    assert geom._vit_ckpt_layers == len(vit.layer)

    shim = _CountingCheckpoint(torch.utils.checkpoint.checkpoint)
    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", shim)

    coord = [{"coord": torch.randn(1, 3), "depth": _consistent_depth(32, 128)} for _ in range(2)]
    try:
        geom.train()
        out = geom(coord, device=DEV)[0]
        assert shim.count == len(vit.layer), (shim.count, len(vit.layer))
        assert all(u is False for u in shim.use_reentrant_seen)
        out.float().pow(2).mean().backward()
        # backward does not re-enter checkpoint; grads flowed into every layer
        assert shim.count == len(vit.layer)
        for i, layer in enumerate(vit.layer):
            assert any(p.grad is not None for p in layer.parameters()), f"layer {i} got no grad"
    finally:
        geom.zero_grad(set_to_none=True)
        geom.eval()


# ------------------------------------------------------------------------------------- #
# 8/9. complete-conditioning invariance at 45° (test i) + raw-pose control (test ii)
# ------------------------------------------------------------------------------------- #
def _rotate_batch(md, deg, img_w):
    return [yr.rotate_scene_metadata(m, math.radians(deg), img_w) for m in md]


def test_i_complete_conditioning_invariant_at_45deg_fa_invariant(tiny_cyl_dir):
    """fa_invariant[0.0]: the FULL conditioning dict (ViT features via the gauge
    backbone + cylindrical pose features + the yaw-invariant RIR path) is invariant
    under a physical 45° yaw of the scene (an exact one-patch token roll at W=128)."""
    mc = create_multi_conditioner_from_conditioning_config(
        _cyl_conditioning(tiny_cyl_dir, with_context=True)
    ).eval()
    md = _synthetic_batch(2, H=32, W=128, n_context=3)
    angles = (0.0,)

    base = yr.invariant_conditioning(mc, md, DEV, angles)
    rot = _rotate_batch(md, 45.0, 128)
    out = yr.invariant_conditioning(mc, rot, DEV, angles)

    assert set(out.keys()) == set(base.keys())
    worst = 0.0
    for key in base:
        d = (out[key][0] - base[key][0]).abs().max().item()
        worst = max(worst, d)
        assert torch.allclose(out[key][0], base[key][0], atol=1e-4), (
            f"{key} not invariant at 45° (max abs diff {d:.3e})"
        )
    print(f"\n[test i] worst complete-conditioning drift at 45°: {worst:.3e}")


def test_ii_raw_pose_negative_control_varies_under_vanilla_at_45deg(tiny_cyl_dir):
    """vanilla (the documented-WRONG path): the raw Cartesian pose conditioners
    (source/context_poses dist-embedders) MUST change under a 45° yaw — proving the
    invariance test above actually has teeth and detects the vanilla trap."""
    mc = create_multi_conditioner_from_conditioning_config(
        _cyl_conditioning(tiny_cyl_dir, with_context=True)
    ).eval()
    md = _synthetic_batch(2, H=32, W=128, n_context=3)

    base = mc(md, DEV)                         # vanilla = single raw pass
    rot = _rotate_batch(md, 45.0, 128)
    out = mc(rot, DEV)

    changed = []
    for key in ("source", "context_poses"):
        d = (out[key][0] - base[key][0]).abs().max().item()
        changed.append(d)
        assert not torch.allclose(out[key][0], base[key][0], atol=1e-4), (
            f"vanilla {key} was unexpectedly invariant at 45° ({d:.3e}) — the trap "
            "detector is vacuous"
        )
    print(f"\n[test ii] vanilla pose-path drift at 45°: source {changed[0]:.3e}, "
          f"context_poses {changed[1]:.3e}")


# ------------------------------------------------------------------------------------- #
# 10. backbone-forward counter == vanilla count (no extra frame-average passes)
#     [mutation-sweep (c): inducing a second frame angle must break this]
# ------------------------------------------------------------------------------------- #
def _count_backbone_forwards(mc, run):
    vit = _geoms(mc)[0].vit
    state = {"n": 0}
    handle = vit.register_forward_hook(lambda *_: state.__setitem__("n", state["n"] + 1))
    try:
        run()
    finally:
        handle.remove()
    return state["n"]


def test_backbone_forward_count_fa_invariant_matches_vanilla(tiny_cyl_dir):
    n_context = 3
    mc = create_multi_conditioner_from_conditioning_config(
        _cyl_conditioning(tiny_cyl_dir, with_context=True)
    ).eval()
    md = _synthetic_batch(2, H=32, W=128, n_context=n_context)

    n_vanilla = _count_backbone_forwards(mc, lambda: mc(md, DEV))
    # fa_invariant with the exp-09 angle set [0.0]: ONE base/frame pass, no extra
    # frame-average passes (the len(angles)==1 early return).
    fa_angles = (0.0,)
    n_fa = _count_backbone_forwards(mc, lambda: yr.invariant_conditioning(mc, md, DEV, fa_angles))

    # source_vit runs the backbone once; context_poses_vit runs it once per context
    # pose -> 1 + K over the batch. fa_invariant[0.0] must match exactly.
    assert n_vanilla == 1 + n_context, (n_vanilla, n_context)
    assert n_fa == n_vanilla, (
        f"fa_invariant[0.0] ran the backbone {n_fa}x vs vanilla {n_vanilla}x — "
        "an extra frame-average pass leaked in"
    )


# ------------------------------------------------------------------------------------- #
# 11. load chain: strict round-trip + eval check_load_integrity + resume, end-to-end
# ------------------------------------------------------------------------------------- #
def _full_exp09_config(tiny_cyl_dir):
    with open(_BF_JSON) as f:
        cfg = json.load(f)
    for c in cfg["model"]["conditioning"]["configs"]:
        if c["type"] == "ViTCoordinates":
            c["config"]["ViT"]["implementation"] = "cylindrical_dinov3"
            c["config"]["ViT"]["gauge"] = "cylindrical_xyz"
            c["config"]["ViT"]["hf_model_name_or_path"] = tiny_cyl_dir
    cfg["training"]["frame_avg_angles"] = [0.0]
    cfg["model"]["diffusion"]["config"]["depth"] = 2  # shrink DiT for a fast CPU build
    return cfg


def test_strict_load_round_trip_of_full_exp09_model(tiny_cyl_dir):
    from src.models import create_model_from_config
    cfg = _full_exp09_config(tiny_cyl_dir)
    torch.manual_seed(SEED)
    model = create_model_from_config(cfg)
    state = model.state_dict()

    torch.manual_seed(SEED + 1)  # different init -> a real reload, not a no-op
    fresh = create_model_from_config(cfg)
    missing, unexpected = fresh.load_state_dict(state, strict=False)
    assert list(missing) == [], f"missing keys on strict reload: {list(missing)[:5]}"
    assert list(unexpected) == [], f"unexpected keys on strict reload: {list(unexpected)[:5]}"
    # train.py-style strict=True must not raise
    fresh.load_state_dict(state, strict=True)


def test_eval_and_resume_load_end_to_end(tiny_cyl_dir):
    from eval_FLAC import check_load_integrity
    from src.models import create_model_from_config
    from src.training import create_training_wrapper_from_config

    cfg = _full_exp09_config(tiny_cyl_dir)
    torch.manual_seed(SEED)
    model = create_model_from_config(cfg)
    wrapper = create_training_wrapper_from_config(cfg, model)

    # --- resume-load (Lightning restores the whole wrapper): strict=True full reload
    torch.manual_seed(SEED + 2)
    fresh_wrapper = create_training_wrapper_from_config(cfg, create_model_from_config(cfg))
    fresh_wrapper.load_state_dict(wrapper.state_dict(), strict=True)  # must not raise

    # --- eval-checkpoint load: replicate eval_FLAC's state_dict remap on a PL-style
    #     wrapper checkpoint and run the REAL check_load_integrity.
    state_dict = {f"diffusion.{k}": v for k, v in model.state_dict().items()}
    for k, v in wrapper.state_dict().items():
        if k.startswith("diffusion_ema.") or k.startswith("losses."):
            state_dict[k] = v
    for key in list(state_dict.keys()):
        if key.startswith("diffusion."):
            state_dict[key.replace("diffusion.", "", 1)] = state_dict.pop(key)
    if any(k.startswith("diffusion_ema.ema_model.") for k in state_dict):
        for key in list(state_dict.keys()):
            if key.startswith("diffusion_ema.ema_model."):
                state_dict[key.replace("diffusion_ema.ema_model.", "model.")] = state_dict.pop(key)

    eval_model = create_model_from_config(cfg)
    missing, unexpected = eval_model.load_state_dict(state_dict, strict=False)
    check_load_integrity(missing, unexpected)  # raises on any real mismatch
    assert list(missing) == [], f"eval load left params at random init: {list(missing)[:5]}"


# ------------------------------------------------------------------------------------- #
# 12. AGREE isolation: the retrieval-metric config subtree is untouched by the swap
# ------------------------------------------------------------------------------------- #
def test_agree_metric_config_is_untouched_by_the_swap():
    with open(_BF_JSON) as f:
        bf = json.load(f)
    with open(_EXP09_JSON) as f:
        exp09 = json.load(f)
    assert exp09["training"]["metrics"] == bf["training"]["metrics"], (
        "the AGREE / retrieval-metric config subtree diverged from B-F"
    )
    # AGREE checkpoint path in particular is byte-identical
    assert exp09["training"]["metrics"]["AGREE_ckpt"] == bf["training"]["metrics"]["AGREE_ckpt"]


# ------------------------------------------------------------------------------------- #
#     config-delta: FLAC_AR_exp09.json vs FLAC_AR_BF.json is EXACTLY the registered
#     set  [mutation-sweep (d): any extra changed key must trip this]
# ------------------------------------------------------------------------------------- #
def _diff(a, b, prefix=""):
    changed, added, removed = {}, {}, {}
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            p = f"{prefix}.{k}" if prefix else k
            if k not in a:
                added[p] = b[k]
            elif k not in b:
                removed[p] = a[k]
            else:
                c, ad, rm = _diff(a[k], b[k], p)
                changed.update(c)
                added.update(ad)
                removed.update(rm)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            c, ad, rm = _diff(x, y, f"{prefix}[{i}]")
            changed.update(c)
            added.update(ad)
            removed.update(rm)
    else:
        if a != b:
            changed[prefix] = (a, b)
    return changed, added, removed


# configs[1] = source_vit, configs[2] = context_poses_vit. EXACT path->value mappings
# (NOT membership in a value-set: a swapped implementation<->gauge on the second block
# keeps both values inside a 2-string set yet is a real bug — the reviewer's bypass).
_REGISTERED_ADDED_VALUES = {
    "model.conditioning.configs[1].config.ViT.implementation": "cylindrical_dinov3",
    "model.conditioning.configs[1].config.ViT.gauge": "cylindrical_xyz",
    "model.conditioning.configs[2].config.ViT.implementation": "cylindrical_dinov3",
    "model.conditioning.configs[2].config.ViT.gauge": "cylindrical_xyz",
}
_REGISTERED_ADDED = set(_REGISTERED_ADDED_VALUES)
_REGISTERED_CHANGED = {"training.frame_avg_angles"}


def test_config_delta_vs_bf_is_exactly_the_registered_set():
    """The delta vs B-F is EXACTLY the registered path->value set, read from the real
    FLAC_AR_exp09.json — every added ViT key is pinned to its precise value, so a
    swapped implementation<->gauge on either block (both still 'cylindrical_*') fails."""
    with open(_BF_JSON) as f:
        bf = json.load(f)
    with open(_EXP09_JSON) as f:
        exp09 = json.load(f)
    changed, added, removed = _diff(bf, exp09)
    assert set(added) == _REGISTERED_ADDED, f"unexpected added keys: {set(added) ^ _REGISTERED_ADDED}"
    assert set(changed) == _REGISTERED_CHANGED, (
        f"unexpected changed keys: {set(changed) ^ _REGISTERED_CHANGED}"
    )
    assert removed == {}, f"unexpected removed keys: {removed}"
    assert changed["training.frame_avg_angles"] == ([0.0, 90.0, 180.0, 270.0], [0.0])
    # EXACT value per path — not membership in a set
    for path, value in _REGISTERED_ADDED_VALUES.items():
        assert added[path] == value, f"{path} = {added[path]!r}, expected {value!r}"


def test_exp09_two_vit_blocks_are_deep_equal_and_exact_in_the_real_json():
    """Read from the REAL JSON (not a test-manufactured deep copy): both ViT blocks
    carry the exact registered values AND are deep-equal to EACH OTHER, so a swapped
    or drifted second block (silently discarded by the factory) is caught here."""
    with open(_EXP09_JSON) as f:
        exp09 = json.load(f)
    configs = exp09["model"]["conditioning"]["configs"]
    vit_confs = [c for c in configs if c["type"] == "ViTCoordinates"]
    assert [c["id"] for c in vit_confs] == ["source_vit", "context_poses_vit"]
    blocks = [c["config"]["ViT"] for c in vit_confs]
    assert blocks[0] == blocks[1], (
        f"the two ViT blocks are NOT deep-equal:\n  source_vit={blocks[0]}\n  "
        f"context_poses_vit={blocks[1]}"
    )
    for b in blocks:
        assert b["implementation"] == "cylindrical_dinov3", b
        assert b["gauge"] == "cylindrical_xyz", b
    for c in vit_confs:
        assert c["config"]["gradient_checkpointing"] is True
    assert exp09["training"]["cond_method"] == "fa_invariant"
    assert exp09["training"]["frame_avg_angles"] == [0.0]
