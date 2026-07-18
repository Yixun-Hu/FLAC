"""Tests for the opt-in ``gradient_checkpointing`` flag on the ViT conditioner.

Motivation: the B-F arm's C4 frame-averaging pushes a ViT batch of 128 depth
panoramas through DINOv3, which OOMs a 48 GB card at micro-batch 32. Activation
checkpointing trades extra backward-time recompute for a large activation-memory
saving with **numerically identical** weights and gradients. The conditioner-config
key ``model.conditioning.configs[i].config.gradient_checkpointing`` (bool, default
``False``/absent) is consumed by ``GeometryConditioner.__init__``.

Implementation under test (post codex P0 review of the pinned transformers==4.57.0):
the conditioner does NOT rely on HF's ``gradient_checkpointing_enable()`` (DINOv3
routes checkpointing through the ``GradientCheckpointingLayer.__call__`` base-class
hook whose enable-time ``functools.partial`` binds ``torch.utils.checkpoint
.checkpoint`` opaquely and gates only on ``.training``). Instead it wraps each
encoder layer's *instance* forward (``vit.layer`` ``nn.ModuleList``) with an
explicit non-reentrant ``torch.utils.checkpoint.checkpoint`` call, gated on
``layer.training and torch.is_grad_enabled()``. That makes the behavior pinned and
OBSERVABLE: these tests shim ``torch.utils.checkpoint.checkpoint`` and count real
invocations during forward/backward — metadata flags are never trusted.

Surfaces under test:

* default/absent: a stack built from the on-disk ``FLAC_AR_BV.json`` conditioning
  block leaves every DINOv3 encoder layer UNWRAPPED (no instance ``forward``, no
  marker) — byte-identical behavior to today.
* enabled (structural): injecting ``gradient_checkpointing: true`` in-memory (no
  JSON written) wraps every layer exactly ONCE even though the two ViTCoordinates
  conditioners share one backbone (idempotency), and records the
  ``_vit_ckpt_layers`` marker.
* enabled (EXECUTION proof): a counting shim around the real
  ``torch.utils.checkpoint.checkpoint`` observes exactly one invocation per layer
  during a train-mode grad-enabled forward (count unchanged by backward — the
  non-reentrant recompute does not re-enter ``checkpoint``), every call pinned to
  ``use_reentrant=False`` (REQUIRED for DDP find_unused_parameters), and ZERO
  invocations under ``eval()`` or ``torch.no_grad()``.
* GRADIENT identity: same seed + input, float32 CPU, forward+backward through the
  flag-ON and flag-OFF backbones -> all parameter grads ``torch.allclose``
  (atol=1e-6, rtol=1e-5; non-reentrant recompute is deterministic on CPU).
* state_dict sha256 identity ON vs OFF (same seed): the adapter registers no
  modules/params/buffers -> weights byte-identical; plus the
  ``assert_arm_configs.py`` gate-invariants canary (param names / count / hash).
* fail-closed: a backbone without a usable non-empty ``nn.ModuleList`` at
  ``.layer`` raises ``ValueError`` naming the backbone type — INCLUDING a backbone
  that offers a decorative ``gradient_checkpointing_enable`` (presence of the HF
  method must no longer be trusted).

Heavy builds are capped at two (``cond_off`` + ``cond_on`` module-scoped fixtures);
execution/gradient tests run the shared vit alone on a tiny 32x32 input (4 patches
+ cls/register tokens). DINOv3 weights load from the local HF cache with
``HF_HUB_OFFLINE=1`` (set below before importing transformers-backed code); no
network or GPU is required. The repo-root/``src/tests`` conftest prepends the
checkout to ``sys.path`` so ``src.*`` resolves here, not to a stale installed copy.
"""
import copy
import hashlib
import json
import os
import random

# Offline HF: resolve DINOv3 strictly from the local cache. Set BEFORE importing
# any transformers-backed module so from_pretrained never reaches the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pytest
import torch
import torch.utils.checkpoint
from torch import nn

from src.models.conditioners import (
    GeometryConditioner,
    create_multi_conditioner_from_conditioning_config,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_EXP07 = os.path.join(_REPO_ROOT, "worklog", "worklog_yixun", "exp_07_fa_scratch_claude")
_BV_JSON = os.path.join(_EXP07, "FLAC_AR_BV.json")

SEED = 42


# --------------------------------------------------------------------------------------
# Environment guards: skip (don't error) when the fixtures cannot be built portably.
# --------------------------------------------------------------------------------------

def _dinov3_cached() -> bool:
    """True iff the pinned DINOv3 snapshot exists in the resolved HF cache."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception:
        return False
    snap = os.path.join(
        HF_HUB_CACHE, "models--facebook--dinov3-vits16-pretrain-lvd1689m", "snapshots"
    )
    return os.path.isdir(snap) and len(os.listdir(snap)) > 0


_needs_dino = pytest.mark.skipif(
    not (os.path.isfile(_BV_JSON) and _dinov3_cached()),
    reason="requires FLAC_AR_BV.json and the DINOv3 snapshot in the local HF cache",
)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _bv_conditioning() -> dict:
    """The ``model.conditioning`` block from the on-disk B-V config (unmodified)."""
    with open(_BV_JSON) as f:
        cfg = json.load(f)
    return cfg["model"]["conditioning"]


def _with_gc(conditioning_cfg: dict, value: bool = True) -> dict:
    """Deep-copy the conditioning block and inject ``gradient_checkpointing`` into
    every ViTCoordinates conditioner config (in-memory only; no JSON is written)."""
    cfg = copy.deepcopy(conditioning_cfg)
    for c in cfg["configs"]:
        if c["type"] == "ViTCoordinates":
            c["config"]["gradient_checkpointing"] = value
    return cfg


def _build(conditioning_cfg: dict):
    """Seeded build of the conditioner stack -> same seed => byte-identical init
    weights across builds (mirrors assert_arm_configs' seeding)."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    return create_multi_conditioner_from_conditioning_config(copy.deepcopy(conditioning_cfg))


def _geometry_conditioners(mc):
    return [c for c in mc.conditioners.values() if getattr(c, "name", None) == "GeometryConditioner"]


def _shared_vit(mc):
    geoms = _geometry_conditioners(mc)
    assert geoms, "no GeometryConditioner in the built stack"
    return geoms[0].vit


def _state_hash(module: nn.Module) -> str:
    """sha256 over the full state_dict (keys + raw bytes), matching the digest the
    pre-launch gate computes for its seeded init-identity assertion."""
    h = hashlib.sha256()
    sd = module.state_dict()
    for k in sorted(sd):
        v = sd[k]
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class _CountingCheckpoint:
    """Counting shim for ``torch.utils.checkpoint.checkpoint``: records each call's
    ``use_reentrant`` and delegates to the REAL checkpoint so autograd semantics
    are exercised, not stubbed."""

    def __init__(self, real):
        self.real = real
        self.use_reentrant_seen = []

    def __call__(self, fn, *args, **kwargs):
        self.use_reentrant_seen.append(kwargs.get("use_reentrant", "ABSENT"))
        return self.real(fn, *args, **kwargs)

    @property
    def count(self):
        return len(self.use_reentrant_seen)


@pytest.fixture()
def counting_checkpoint(monkeypatch):
    shim = _CountingCheckpoint(torch.utils.checkpoint.checkpoint)
    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", shim)
    return shim


def _tiny_input():
    """Smallest legal DINOv3 input: 32x32 -> 2x2 patches (+ cls/register tokens)."""
    return torch.randn(1, 3, 32, 32)


# --------------------------------------------------------------------------------------
# Heavy fixtures: exactly two full DINOv3 builds, shared across the module.
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cond_off():
    return _build(_bv_conditioning())


@pytest.fixture(scope="module")
def cond_on():
    return _build(_with_gc(_bv_conditioning(), True))


# --------------------------------------------------------------------------------------
# default / absent -> layers untouched (byte-identical behavior to today)
# --------------------------------------------------------------------------------------

@_needs_dino
def test_default_bv_vit_layers_unwrapped(cond_off):
    """The unmodified B-V conditioning block leaves every encoder layer with its
    plain class forward (no instance-level override, no wrap marker) and sets no
    ``_vit_ckpt_layers`` marker on the conditioners."""
    vit = _shared_vit(cond_off)
    for layer in vit.layer:
        assert "forward" not in layer.__dict__
        assert not getattr(layer, "_flac_ckpt_wrapped", False)
    for geom in _geometry_conditioners(cond_off):
        assert getattr(geom, "_vit_ckpt_layers", None) is None


@_needs_dino
def test_bv_vit_conditioners_share_one_backbone(cond_off):
    """Documents the shared-backbone invariant the factory guarantees: both
    ViTCoordinates conditioners wrap the SAME vit object, so the flag on either
    conditioner checkpoints the one shared encoder."""
    geoms = _geometry_conditioners(cond_off)
    assert len(geoms) == 2
    assert geoms[0].vit is geoms[1].vit


# --------------------------------------------------------------------------------------
# enabled -> structural wiring (each layer wrapped exactly once, marker recorded)
# --------------------------------------------------------------------------------------

@_needs_dino
def test_enabled_wraps_every_layer_exactly_once(cond_on):
    """gradient_checkpointing:true wraps every encoder layer's INSTANCE forward and
    records the marker; despite BOTH ViTCoordinates conditioners carrying the flag
    over the one shared backbone, each layer is wrapped exactly once (idempotent),
    so no double checkpointing."""
    vit = _shared_vit(cond_on)
    n_layers = len(vit.layer)
    assert n_layers > 0
    for layer in vit.layer:
        assert "forward" in layer.__dict__, "layer forward not instance-wrapped"
        assert getattr(layer, "_flac_ckpt_wrapped", False) is True
        inner = getattr(layer.forward, "__wrapped__", None)
        assert inner is not None, "wrapper must expose __wrapped__ (functools.wraps)"
        # exactly one wrap: the wrapped target is the plain bound method, not
        # another checkpoint wrapper
        assert getattr(inner, "__wrapped__", None) is None, "layer wrapped twice"
    for geom in _geometry_conditioners(cond_on):
        assert geom._vit_ckpt_layers == n_layers


# --------------------------------------------------------------------------------------
# enabled -> EXECUTION proof (checkpoint really runs; metadata is never trusted)
# --------------------------------------------------------------------------------------

@_needs_dino
def test_checkpoint_executes_once_per_layer_train_grad(cond_on, counting_checkpoint):
    """train() + grad-enabled forward invokes the REAL torch.utils.checkpoint
    .checkpoint exactly once per encoder layer, every call pinned to
    use_reentrant=False; backward succeeds through the checkpointed graph (each
    layer's params receive grads) WITHOUT re-entering checkpoint (non-reentrant
    recompute happens inside the saved-tensors hooks)."""
    vit = _shared_vit(cond_on)
    n_layers = len(vit.layer)
    try:
        vit.train()
        torch.manual_seed(0)
        out = vit(_tiny_input())
        assert counting_checkpoint.count == n_layers, (
            f"expected {n_layers} checkpoint invocations during forward, "
            f"saw {counting_checkpoint.count}"
        )
        assert all(u is False for u in counting_checkpoint.use_reentrant_seen), (
            counting_checkpoint.use_reentrant_seen
        )
        loss = out.last_hidden_state.float().pow(2).mean()
        loss.backward()
        # backward must not re-enter checkpoint()
        assert counting_checkpoint.count == n_layers
        # gradients flowed through every checkpointed layer
        for i, layer in enumerate(vit.layer):
            assert any(p.grad is not None for p in layer.parameters()), (
                f"layer {i} received no gradients through checkpointing"
            )
        print(f"\nmeasured checkpoint invocations: {counting_checkpoint.count} "
              f"(= num layers {n_layers}), use_reentrant pinned False on all calls")
    finally:
        vit.zero_grad(set_to_none=True)
        vit.eval()


@_needs_dino
def test_checkpoint_not_invoked_eval_or_no_grad(cond_on, counting_checkpoint):
    """eval() (grad enabled) and train()+torch.no_grad() both bypass checkpointing
    entirely (zero invocations): inference stays zero-overhead."""
    vit = _shared_vit(cond_on)
    try:
        vit.eval()
        torch.manual_seed(0)
        vit(_tiny_input())
        assert counting_checkpoint.count == 0

        vit.train()
        with torch.no_grad():
            vit(_tiny_input())
        assert counting_checkpoint.count == 0
    finally:
        vit.zero_grad(set_to_none=True)
        vit.eval()


# --------------------------------------------------------------------------------------
# GRADIENT identity ON vs OFF (numerics untouched end-to-end, not just at init)
# --------------------------------------------------------------------------------------

def _param_grads(vit: nn.Module, seed: int = 123):
    """Seeded tiny forward+backward through the vit; returns {name: grad clone}."""
    try:
        vit.train()
        vit.zero_grad(set_to_none=True)
        torch.manual_seed(seed)
        out = vit(_tiny_input())
        loss = out.last_hidden_state.float().pow(2).mean()
        loss.backward()
        return {
            n: p.grad.detach().clone()
            for n, p in vit.named_parameters()
            if p.grad is not None
        }
    finally:
        vit.zero_grad(set_to_none=True)
        vit.eval()


@_needs_dino
def test_gradient_identity_on_vs_off(cond_off, cond_on):
    """Same seed + input, float32 CPU: parameter grads with checkpointing ON are
    torch.allclose (atol=1e-6, rtol=1e-5) to the plain path — the non-reentrant
    recompute is deterministic on CPU, so the flag changes memory, never numerics."""
    grads_off = _param_grads(_shared_vit(cond_off))
    grads_on = _param_grads(_shared_vit(cond_on))
    assert set(grads_off) == set(grads_on)
    assert len(grads_off) >= 100, f"suspiciously few grad tensors: {len(grads_off)}"
    max_diff = 0.0
    for n in grads_off:
        assert torch.allclose(grads_on[n], grads_off[n], atol=1e-6, rtol=1e-5), (
            f"grad mismatch for {n}: max abs diff "
            f"{(grads_on[n] - grads_off[n]).abs().max().item():.3e}"
        )
        max_diff = max(max_diff, (grads_on[n] - grads_off[n]).abs().max().item())
    print(f"\ngradient-identity max abs diff over {len(grads_off)} tensors: {max_diff:.3e}")


# --------------------------------------------------------------------------------------
# state_dict identity + assert_arm_configs gate-invariants canary
# --------------------------------------------------------------------------------------

@_needs_dino
def test_state_dict_identical_flag_on_vs_off(cond_off, cond_on):
    """Same seed, flag ON vs OFF -> byte-identical state_dict. The adapter replaces
    instance forwards only; it registers no modules/params/buffers."""
    assert _state_hash(cond_on) == _state_hash(cond_off)


@_needs_dino
def test_gate_architecture_invariants_hold_under_flag(cond_off, cond_on):
    """Regression canary for worklog/.../assert_arm_configs.py: that gate asserts
    (i) identical named_parameters name-sets, (ii) identical param count, and (iii)
    identical seeded state_dict sha256 across arms. All three are invariant under the
    flag, so adding gradient_checkpointing to the BF config does NOT trip the gate."""
    names_off = [n for n, _ in cond_off.named_parameters()]
    names_on = [n for n, _ in cond_on.named_parameters()]
    assert names_off == names_on
    assert sum(p.numel() for p in cond_off.parameters()) == sum(
        p.numel() for p in cond_on.parameters()
    )
    assert _state_hash(cond_off) == _state_hash(cond_on)


# --------------------------------------------------------------------------------------
# cheap adapter semantics on a stub backbone (no DINOv3 load)
# --------------------------------------------------------------------------------------

class _TinyLayerViT(nn.Module):
    """Minimal DINOv3-shaped backbone: a ``.layer`` nn.ModuleList of Linear layers."""

    def __init__(self, n_layers: int = 2, dim: int = 8):
        super().__init__()
        self.layer = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])

    def forward(self, x):
        for l in self.layer:
            x = l(x)
        return x


def _make_geom(vit, **kw):
    return GeometryConditioner(
        vit_model=vit,
        vit_proj=nn.Identity(),
        lin_proj=nn.Identity(),
        output_dim=4,
        model_type="vit",
        **kw,
    )


def test_stub_adapter_wraps_counts_and_matches_values(counting_checkpoint):
    """No-DINOv3 pin of the full adapter contract: constructing with the flag wraps
    each stub layer once (idempotent across a second conditioner sharing the same
    backbone), train+grad calls route through the real checkpoint with
    use_reentrant=False, eval calls bypass it, and checkpointed outputs equal the
    plain ones."""
    stub = _TinyLayerViT(n_layers=2)
    cond = _make_geom(stub, gradient_checkpointing=True)
    assert cond._vit_ckpt_layers == 2
    # second conditioner over the SAME backbone: no double-wrap
    cond2 = _make_geom(stub, gradient_checkpointing=True)
    assert cond2._vit_ckpt_layers == 2
    for layer in stub.layer:
        assert getattr(layer.forward.__wrapped__, "__wrapped__", None) is None

    x = torch.randn(3, 8)
    stub.eval()
    y_plain = stub(x)
    assert counting_checkpoint.count == 0

    stub.train()
    y_ckpt = stub(x)
    assert counting_checkpoint.count == 2  # once per layer
    assert all(u is False for u in counting_checkpoint.use_reentrant_seen)
    assert y_ckpt.requires_grad
    assert torch.allclose(y_ckpt.detach(), y_plain, atol=0, rtol=0)

    y_ckpt.pow(2).mean().backward()
    assert counting_checkpoint.count == 2  # backward does not re-enter checkpoint()
    assert all(l.weight.grad is not None for l in stub.layer)


# --------------------------------------------------------------------------------------
# fail-closed + absent/False no-op on incompatible backbones
# --------------------------------------------------------------------------------------

class _NoLayerViT(nn.Module):
    """Backbone with no ``.layer`` at all (e.g. a plain in-repo ViT)."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x):
        return x


class _EmptyLayerViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.ModuleList()

    def forward(self, x):
        return x


class _NonModuleListLayerViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 4)

    def forward(self, x):
        return x


class _DecorativeEnableViT(nn.Module):
    """Offers HF's method name but nothing checkpointable underneath: the presence
    of gradient_checkpointing_enable must NOT be trusted (the 4.57.0 P0 lesson)."""

    def __init__(self):
        super().__init__()
        self.layer = nn.ModuleList()

    def gradient_checkpointing_enable(self, **kwargs):  # pragma: no cover - decorative
        pass

    def forward(self, x):
        return x


@pytest.mark.parametrize(
    "vit_cls", [_NoLayerViT, _EmptyLayerViT, _NonModuleListLayerViT, _DecorativeEnableViT]
)
def test_fail_closed_without_usable_layer_modulelist(vit_cls):
    """Requesting the flag on a backbone without a non-empty nn.ModuleList at
    ``.layer`` is a hard ValueError naming the backbone type — never a silent
    no-op, and never satisfied by a decorative gradient_checkpointing_enable."""
    with pytest.raises(ValueError) as excinfo:
        _make_geom(vit_cls(), gradient_checkpointing=True)
    msg = str(excinfo.value)
    assert "gradient_checkpointing" in msg
    assert vit_cls.__name__ in msg


def test_absent_flag_is_noop_on_incompatible_backbone():
    """Absent/False -> the guard never fires even on a backbone that could not
    support checkpointing, so default behavior is byte-identical to today."""
    stub = _NoLayerViT()
    cond_default = _make_geom(stub)
    assert cond_default.vit is stub
    assert getattr(cond_default, "_vit_ckpt_layers", None) is None
    stub2 = _NoLayerViT()
    cond_false = _make_geom(stub2, gradient_checkpointing=False)
    assert cond_false.vit is stub2
    assert getattr(cond_false, "_vit_ckpt_layers", None) is None
