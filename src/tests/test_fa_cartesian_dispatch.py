"""exp_21 (bf_fa_cartesian) — training-side dispatch, guards and the arm config.

RED first. ``DiffusionCondTrainingWrapper``'s cond_method whitelist admits only
``vanilla`` / ``fa_invariant``; ``_compute_conditioning`` has no ``fa_cartesian``
branch; ``_parse_yaw_aug_config`` rejects only ``fa_invariant``; and
``FLAC_AR_BFC.json`` does not exist yet. Each test pins a contract the plan
states (§3b, §3c, §3e, §3h item 9).

The recorded red phase was **22 failed / 22 passed**, not a clean sweep (r2
review, nit 2). The 22 pre-implementation passes are deliberate no-change pins:
the factory's ``frame_avg_angles`` forwarding and ``_parse_frame_avg_cap_config``
are not cond_method-gated, so they already plumbed correctly to an
``fa_cartesian`` wrapper and there was nothing to make green — they are asserted
so a gate added LATER fails here rather than in a multi-day run. Those tests
reach the factory through a stubbed wrapper, which is what lets them run at all
while the real whitelist still rejected the method.

**Why a new file rather than more cases in ``test_cond_dispatch.py``.** That file
is exp_03's audited record of the vanilla/fa_invariant contract and is part of
this round's regression set: leaving it byte-unchanged is precisely what makes
its green run evidence that widening the whitelist and adding a dispatch branch
did not disturb the two methods already in the record. exp_14 set the precedent
of an arm owning ONE file covering its call-site, factory and config-parity
contracts together (``test_frame_avg_cap_config.py``), and it keeps
``test_fa_cartesian.py`` a pure ``yaw_rotation`` suite that imports no model
factory.

**The call-shape tests and what they are for.** They pin the literal argument
form of the dispatch: no declared cap -> the four-argument positional call; a
declared cap -> those same four arguments plus ``max_fwd_samples=`` as a
KEYWORD. The rationale is PARITY with the ``fa_invariant`` branch and API
discipline — NOT positional-slip prevention (r1 review, candidate finding (b)).
``fa_cartesian_conditioning``'s fifth positional parameter IS
``max_fwd_samples``; it has no ``vit_ids`` parameter, so a positional fifth
argument would be harmless here, whereas in ``invariant_conditioning`` it would
silently disable the frame average. The two branches issue the same call shape so
that the arms differ in the conditioning FUNCTION and in nothing about how it is
invoked — including which arm a future refactor of the shared cap plumbing would
touch.
"""
import copy
import json
import math
import types
from pathlib import Path

import pytest
import torch
from torch import nn

import src.training.diffusion as tdiff
from src.data import yaw_rotation as yr
from src.models.conditioners import MultiConditioner
from src.models.factory import create_model_from_config
from src.training.factory import create_training_wrapper_from_config

DEV = "cpu"
C4 = yr.DEFAULT_FRAME_ANGLES
CAP_KEY = "frame_avg_max_fwd_samples"

_REPO = Path(__file__).resolve().parents[2]
BF_CONFIG = _REPO / "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json"
BFC_CONFIG = (_REPO / "worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude"
              / "FLAC_AR_BFC.json")

# The arm's declared identity (plan §3e / decision D5).
BFC_COND_METHOD = "fa_cartesian"
BFC_ANGLES = [0.0, 90.0, 180.0, 270.0]
BFC_CAP = 32
BFC_MICRO_BATCH = 32          # per rank, from the launch recipe (plan §3f)

H, W, N_CTX = 2, 16, 3


@pytest.fixture(autouse=True)
def single_thread():
    """The tensors here are tiny; torch's default pool would spend its time
    launching threads (the ``test_fa_cartesian`` / ``test_invariant_conditioning``
    convention)."""
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(prev)


# --------------------------------------------------------------------------- #
# counting stand-ins for the pose/geometry stack (no DINOv3, no pretrained ckpt)
# --------------------------------------------------------------------------- #
class _CountingDist(nn.Module):
    """dist_embedder stand-in; records the batch size of every forward."""

    def __init__(self, out_dim: int = 4, seed: int = 0):
        super().__init__()
        self.name = "DistEmbedderConditioner"
        self.batch_sizes = []
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("w", torch.randn(3, out_dim, generator=g))

    def forward(self, x_list, device=DEV):
        self.batch_sizes.append(len(x_list))
        x = torch.stack(x_list, dim=0).to(device)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out = torch.tanh(x @ self.w + 0.3)
        return [out.contiguous(), torch.ones(out.shape[0], out.shape[1], device=device)]


class _CountingGeom(nn.Module):
    """GeometryConditioner stand-in (name matched so ``MultiConditioner`` feeds it
    ``{'coord', 'depth'}``); records the batch size of every forward."""

    def __init__(self, out_dim: int = 4, seed: int = 0):
        super().__init__()
        self.name = "GeometryConditioner"
        self.batch_sizes = []
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("proj", torch.randn(3, out_dim, generator=g))

    def forward(self, coord_list, device=DEV):
        self.batch_sizes.append(len(coord_list))
        coord = torch.stack([c["coord"] for c in coord_list], dim=0).to(device)
        if coord.ndim == 2:
            coord = coord.unsqueeze(1)
        depth = torch.stack([c["depth"] for c in coord_list], dim=0).to(device)
        stat = depth.mean(dim=(2, 3))                       # [B, 3]
        h = (coord + stat[:, None, :]) @ self.proj
        return [torch.tanh(h).contiguous(), torch.ones(h.shape[0], 1, device=device)]


def _pose_conditioner() -> MultiConditioner:
    """All four :data:`yr.POSE_KEYS` — ``fa_cartesian_conditioning`` is fail-closed
    on a missing one, so a partial id set could not reach the partition at all."""
    return MultiConditioner({
        "source": _CountingDist(seed=1),
        "context_poses": _CountingDist(seed=2),
        "source_vit": _CountingGeom(seed=3),
        "context_poses_vit": _CountingGeom(seed=4),
    }).eval()


def _pose_md(seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "source": torch.randn(3, generator=g),
        "source_vit": torch.randn(1, 3, generator=g),
        "context_poses": torch.randn(N_CTX, 3, generator=g),
        "context_poses_vit": torch.randn(N_CTX, 3, generator=g),
        "depth": torch.randn(3, H, W, generator=g),
    }


def _pose_batch(n: int) -> list:
    return [_pose_md(s) for s in range(n)]


class _TinyDiffusion(nn.Module):
    """The minimum surface ``DiffusionCondTrainingWrapper`` touches at
    construction, plus a conditioner ``_compute_conditioning`` can really run."""

    def __init__(self):
        super().__init__()
        self.model = nn.Linear(2, 2)
        self.conditioner = _pose_conditioner()
        self.diffusion_objective = "rectified_flow"
        self.pretransform = None


def _direct(cond_method: str = BFC_COND_METHOD, **kwargs):
    """Direct wrapper construction: the guards must fail closed on their own, not
    only behind the factory."""
    return tdiff.DiffusionCondTrainingWrapper(
        _TinyDiffusion(), lr=5e-5, use_ema=False, cond_method=cond_method, **kwargs)


# --------------------------------------------------------------------------- #
# tiny real diffusion_cond model (for the three-site dispatch tests)
# --------------------------------------------------------------------------- #
def _base_config():
    """``test_cond_dispatch``'s CPU-only model, verbatim in shape: the dispatch
    tests neutralise everything after conditioning, so the deliberately shrunken
    (not forward-consistent) DiT is exactly what is wanted here too."""
    return {
        "model_type": "diffusion_cond",
        "sample_size": 64,
        "sample_rate": 22050,
        "audio_channels": 1,
        "model": {
            "conditioning": {
                "configs": [
                    {"id": "source", "type": "dist_embedder",
                     "config": {"num_freqs": 4, "max_freq": 4, "ch_dim": 1, "include_in": True}},
                    {"id": "context_poses", "type": "dist_embedder",
                     "config": {"num_freqs": 4, "max_freq": 4, "ch_dim": 1, "include_in": True}},
                ],
                "cond_dim": 32,
            },
            "diffusion": {
                "cross_attention_cond_ids": ["context_poses"],
                "global_cond_ids": ["source"],
                "type": "dit",
                "diffusion_objective": "rectified_flow",
                "config": {
                    "io_channels": 4, "embed_dim": 32, "depth": 1, "num_heads": 2,
                    "cond_token_dim": 32, "global_cond_dim": 64,
                    "transformer_type": "continuous_transformer",
                    "global_cond_type": "adaLN",
                },
            },
            "io_channels": 4,
        },
        "training": {
            "timestep_sampler": "uniform",
            "cfg_dropout_prob": 0.0,
            "use_ema": False,
            "optimizer_configs": {
                "diffusion": {"optimizer": {"type": "AdamW",
                    "config": {"lr": 5e-6, "betas": [0.9, 0.999], "weight_decay": 1e-3}}}
            },
        },
    }


def _build_wrapper(**training_overrides):
    cfg = _base_config()
    cfg["training"].update(training_overrides)
    model = create_model_from_config(cfg)
    return create_training_wrapper_from_config(cfg, model)


def _step_md(seed):
    g = torch.Generator().manual_seed(seed)
    return {
        "source": torch.randn(3, generator=g),
        "context_poses": torch.randn(2, 3, generator=g),
        "padding_mask": torch.ones(64),                 # training_step reads this
        "scene": f"s{seed}",                            # test_step reads this
        "depth": torch.randn(3, 8, 16, generator=g),    # test_step depth path
    }


def _step_batch(n=2):
    return torch.randn(n, 1, 64), [_step_md(s) for s in range(n)]


def _neutralize_for_steps(wrapper, monkeypatch):
    """Stub everything the step methods touch AFTER conditioning, so a dispatch
    test never depends on the DiT maths, the sampler, the metric callback, a
    Lightning Trainer or the logger (``test_cond_dispatch``'s pattern)."""
    wrapper.diffusion.forward = lambda x, t, **kw: torch.zeros_like(x)
    wrapper.log_dict = lambda *a, **k: None
    wrapper.trainer = types.SimpleNamespace(
        optimizers=[types.SimpleNamespace(param_groups=[{"lr": 5e-6}])]
    )
    wrapper.samples = 64
    wrapper.steps = 1
    wrapper.cfg_scale = 1.0
    wrapper.store_predictions = False
    wrapper.preds = []
    wrapper.metric_callback = types.SimpleNamespace(update_metrics=lambda *a, **k: None)
    monkeypatch.setattr(tdiff, "sample_discrete_euler", lambda model, x, *a, **k: x)


class _Spy:
    """Records the RAW call — so the assertions are on the call SHAPE and not
    merely on its effect — and delegates to the real conditioner so the
    surrounding step code still receives valid conditioning."""

    def __init__(self):
        self.calls = []

    def __call__(self, conditioner, metadata, device, *args, **kwargs):
        self.calls.append(((conditioner, metadata, device) + args, dict(kwargs)))
        return conditioner(metadata, device)

    @property
    def n(self):
        return len(self.calls)


# --------------------------------------------------------------------------- #
# 1. the constructor whitelist
# --------------------------------------------------------------------------- #
def test_fa_cartesian_is_accepted_by_the_wrapper():
    wrapper = _build_wrapper(cond_method=BFC_COND_METHOD)
    assert wrapper.cond_method == BFC_COND_METHOD
    assert tuple(wrapper.frame_avg_angles) == tuple(yr.DEFAULT_FRAME_ANGLES)
    assert wrapper.frame_avg_max_fwd_samples is None


@pytest.mark.parametrize("method", ["vanilla", "fa_invariant"])
def test_the_existing_cond_methods_still_construct(method):
    """Widening a whitelist must not narrow it anywhere else."""
    assert _build_wrapper(cond_method=method).cond_method == method


@pytest.mark.parametrize("bad", [
    "canon", "fa-cartesian", "facartesian", "FA_CARTESIAN", "cartesian", "",
])
def test_unknown_cond_methods_are_still_rejected_at_construction(bad):
    """Fail fast at construction, not at the first step of a multi-day run."""
    with pytest.raises(ValueError) as e:
        _build_wrapper(cond_method=bad)
    assert "cond_method" in str(e.value)


def test_the_whitelist_error_enumerates_the_new_method():
    """A stale 'valid options' list sends the next reader to the wrong branch."""
    with pytest.raises(ValueError) as e:
        _build_wrapper(cond_method="nope")
    message = str(e.value)
    for method in ("vanilla", "fa_invariant", BFC_COND_METHOD):
        assert method in message, f"{method!r} missing from {message!r}"


def test_frame_avg_angles_override_reaches_an_fa_cartesian_wrapper():
    wrapper = _build_wrapper(cond_method=BFC_COND_METHOD, frame_avg_angles=[0.0, 180.0])
    assert tuple(wrapper.frame_avg_angles) == (0.0, 180.0)


# --------------------------------------------------------------------------- #
# 2. yaw_aug x fa_cartesian is rejected at BOTH sites
# --------------------------------------------------------------------------- #
def _construct_directly(**kwargs):
    """Bypass the factory entirely; the yaw_aug validation runs before the
    constructor touches the model, so a bare namespace stands in for it."""
    return tdiff.DiffusionCondTrainingWrapper(
        types.SimpleNamespace(), lr=5e-6, use_ema=False, **kwargs)


def _factory_config(**training):
    cfg = {
        "model_type": "diffusion_cond",
        "sample_rate": 22050,
        "training": {"learning_rate": 5e-5},
    }
    cfg["training"].update(training)
    return cfg


@pytest.fixture()
def stub_wrapper(monkeypatch):
    """Capture the factory's kwargs without building a model."""
    captured = {}

    class _Stub:
        def __init__(self, model, **kwargs):
            captured.clear()
            captured.update(kwargs)

    monkeypatch.setattr(tdiff, "DiffusionCondTrainingWrapper", _Stub)
    return captured


@pytest.fixture()
def wrapper_must_not_be_constructed(monkeypatch):
    """Replace the wrapper with a stub that FAILS if it is ever built.

    r2 review, BLOCKING finding. A factory-guard test that constructs the real
    wrapper does not test the factory guard: the wrapper carries its own
    yaw_aug × frame-average rejection whose message also contains ``yaw_aug`` and
    the method name, so deleting ``"fa_cartesian"`` from
    ``_parse_yaw_aug_config``'s membership tuple would leave such a test green.
    (That is also why its red-phase failure proved nothing about the factory: the
    pre-implementation wrapper rejected ``fa_cartesian`` merely as an unknown
    method.)

    ``AssertionError`` is deliberately not a ``ValueError``, so it cannot be
    absorbed by the ``pytest.raises(ValueError)`` under test: reaching
    construction fails the test loudly instead of quietly satisfying it.
    """
    class _Forbidden:
        def __init__(self, model, **kwargs):
            raise AssertionError(
                "the factory went on to CONSTRUCT the wrapper: "
                "_parse_yaw_aug_config did not reject this combination, so any "
                "ValueError this test could see would be the wrapper's guard "
                "standing in for the factory's")

    monkeypatch.setattr(tdiff, "DiffusionCondTrainingWrapper", _Forbidden)


YAW_AUG_BLOCK = {"enabled": True, "img_w": 512, "seed": 42}


@pytest.mark.parametrize("method", ["fa_invariant", BFC_COND_METHOD])
def test_wrapper_ctor_rejects_yaw_aug_with_a_frame_averaged_method(method):
    """Composing them silently would train an arm neither experiment declared.

    NOT because the orbit covers the augmentation: ``yaw_aug`` draws uniformly
    over all ``img_w`` (512) column rotations, so C4 is a four-element SUBGROUP of
    what it samples, not the same set (the production guards at
    ``training/factory.py`` and ``training/diffusion.py`` say this correctly; this
    docstring did not). The composition is an unapproved, untested distinct
    treatment, which is why it is refused rather than reasoned about.

    ``fa_invariant`` is included as the regression: exp_15's guard must not be
    weakened while it is being widened."""
    with pytest.raises(ValueError) as e:
        _construct_directly(yaw_aug_enabled=True, yaw_aug_img_w=512, yaw_aug_seed=42,
                            cond_method=method)
    message = str(e.value)
    assert "yaw_aug" in message and method in message, message


@pytest.mark.parametrize("method", ["fa_invariant", BFC_COND_METHOD])
def test_factory_rejects_yaw_aug_with_a_frame_averaged_method(
        method, wrapper_must_not_be_constructed):
    """Isolated to the FACTORY site (r2 review, BLOCKING): with construction
    forbidden, ``_parse_yaw_aug_config`` is the only thing that can raise a
    ValueError here, so the test fails if its membership tuple loses the method.
    The wrapper site keeps its own independent pin above."""
    with pytest.raises(ValueError) as e:
        create_training_wrapper_from_config(
            _factory_config(cond_method=method, yaw_aug=dict(YAW_AUG_BLOCK)), object())
    message = str(e.value)
    assert "yaw_aug" in message and method in message, message


def test_yaw_aug_is_still_allowed_with_vanilla(stub_wrapper):
    """The guard must reject the frame-averaged methods and nothing else — exp_15's
    own arm is a vanilla one."""
    create_training_wrapper_from_config(
        _factory_config(cond_method="vanilla", yaw_aug=dict(YAW_AUG_BLOCK)), object())
    assert stub_wrapper["yaw_aug_enabled"] is True
    assert stub_wrapper["yaw_aug_img_w"] == 512
    assert stub_wrapper["yaw_aug_seed"] == 42


def test_a_disabled_yaw_aug_block_is_not_rejected_for_fa_cartesian(stub_wrapper):
    """``enabled: false`` is the pre-change call: no kwargs, no rejection."""
    create_training_wrapper_from_config(
        _factory_config(cond_method=BFC_COND_METHOD, yaw_aug={"enabled": False}), object())
    assert "yaw_aug_enabled" not in stub_wrapper
    assert stub_wrapper["cond_method"] == BFC_COND_METHOD


# --------------------------------------------------------------------------- #
# 3. dispatch: one fa_cartesian call per step, at all three sites, and never the
#    fa_invariant function
# --------------------------------------------------------------------------- #
def test_fa_cartesian_dispatches_in_training_validation_and_test(monkeypatch):
    """The repo's ``flow_source`` precedent: a new conditioning mode has to reach
    training_step, validation_step AND test_step, or an arm evaluates through a
    method it was not trained with."""
    wrapper = _build_wrapper(cond_method=BFC_COND_METHOD)
    _neutralize_for_steps(wrapper, monkeypatch)
    spy = _Spy()
    monkeypatch.setattr(tdiff, "fa_cartesian_conditioning", spy)

    reals, metadata = _step_batch(2)
    for name, step in (("training_step", wrapper.training_step),
                       ("validation_step", wrapper.validation_step),
                       ("test_step", wrapper.test_step)):
        before = spy.n
        step((reals, metadata), 0)
        assert spy.n == before + 1, (
            f"{name} issued {spy.n - before} fa_cartesian_conditioning call(s), "
            "expected exactly 1")
    assert spy.n == 3


def test_fa_cartesian_never_calls_invariant_conditioning(monkeypatch):
    """The single mechanism change is the conditioning FUNCTION; a fallthrough to
    the cylindrical path would produce plausible numbers for the wrong arm."""
    wrapper = _build_wrapper(cond_method=BFC_COND_METHOD)
    _neutralize_for_steps(wrapper, monkeypatch)
    cart, inv = _Spy(), _Spy()
    monkeypatch.setattr(tdiff, "fa_cartesian_conditioning", cart)
    monkeypatch.setattr(tdiff, "invariant_conditioning", inv)

    reals, metadata = _step_batch(2)
    wrapper.training_step((reals, metadata), 0)
    wrapper.validation_step((reals, metadata), 0)
    wrapper.test_step((reals, metadata), 0)
    assert cart.n == 3
    assert inv.n == 0, "fa_cartesian must never route through invariant_conditioning"


def test_fa_invariant_never_calls_fa_cartesian_conditioning(monkeypatch):
    """The converse regression: B-F is the comparator arm and must keep dispatching
    exactly where it always did."""
    wrapper = _build_wrapper(cond_method="fa_invariant")
    _neutralize_for_steps(wrapper, monkeypatch)
    cart, inv = _Spy(), _Spy()
    monkeypatch.setattr(tdiff, "fa_cartesian_conditioning", cart)
    monkeypatch.setattr(tdiff, "invariant_conditioning", inv)

    reals, metadata = _step_batch(2)
    wrapper.training_step((reals, metadata), 0)
    assert inv.n == 1 and cart.n == 0


def test_vanilla_calls_neither_symmetrisation(monkeypatch):
    wrapper = _build_wrapper(cond_method="vanilla")
    _neutralize_for_steps(wrapper, monkeypatch)
    cart, inv = _Spy(), _Spy()
    monkeypatch.setattr(tdiff, "fa_cartesian_conditioning", cart)
    monkeypatch.setattr(tdiff, "invariant_conditioning", inv)

    reals, metadata = _step_batch(2)
    wrapper.training_step((reals, metadata), 0)
    assert cart.n == 0 and inv.n == 0


def test_the_backstop_raise_still_guards_an_unknown_method():
    """The constructor is the fail-fast gate; ``_compute_conditioning`` keeps its
    own raise so a value assigned after construction cannot fall through the new
    branch into ``vanilla``."""
    wrapper = _direct()
    wrapper.cond_method = "smuggled"
    with pytest.raises(ValueError, match="cond_method"):
        wrapper._compute_conditioning(_pose_batch(2))


# --------------------------------------------------------------------------- #
# 4. call shape: parity with the fa_invariant branch (see the module docstring)
# --------------------------------------------------------------------------- #
def test_the_no_cap_path_issues_the_four_argument_call(monkeypatch):
    spy = _Spy()
    monkeypatch.setattr(tdiff, "fa_cartesian_conditioning", spy)

    wrapper = _direct()
    wrapper._compute_conditioning(_pose_batch(2))

    args, kwargs = spy.calls[-1]
    assert kwargs == {}, (
        "the no-cap path must issue the four-argument call with NO keyword — the "
        f"same shape the fa_invariant branch issues; got keywords {sorted(kwargs)}")
    assert len(args) == 4, f"expected exactly 4 positional args, got {len(args)}"
    assert args[0] is wrapper.diffusion.conditioner
    assert tuple(args[3]) == tuple(C4), "angles must be the 4th positional argument"


def test_a_declared_cap_arrives_as_a_keyword(monkeypatch):
    """PARITY, not slip prevention (r1 candidate finding (b)):
    ``fa_cartesian_conditioning``'s fifth positional parameter IS
    ``max_fwd_samples``, so a positional argument would be correct here — it is
    the fa_invariant branch, whose fifth positional is ``vit_ids``, that requires
    the keyword. Both branches issue the same shape so the arms differ only in
    which function is called."""
    spy = _Spy()
    monkeypatch.setattr(tdiff, "fa_cartesian_conditioning", spy)

    _direct(frame_avg_max_fwd_samples=BFC_CAP)._compute_conditioning(_pose_batch(2))

    args, kwargs = spy.calls[-1]
    assert kwargs == {"max_fwd_samples": BFC_CAP}
    assert len(args) == 4, "the cap must not be filled positionally"
    assert tuple(args[3]) == tuple(C4)


def test_the_two_branches_issue_the_same_call_shape(monkeypatch):
    """Stated as an equality rather than as two separate expectations, so a future
    change to one branch's plumbing cannot leave the arms comparing different
    call conventions."""
    cart, inv = _Spy(), _Spy()
    monkeypatch.setattr(tdiff, "fa_cartesian_conditioning", cart)
    monkeypatch.setattr(tdiff, "invariant_conditioning", inv)

    for cap in (None, BFC_CAP):
        extra = {} if cap is None else {"frame_avg_max_fwd_samples": cap}
        _direct(cond_method=BFC_COND_METHOD, **extra)._compute_conditioning(_pose_batch(2))
        _direct(cond_method="fa_invariant", **extra)._compute_conditioning(_pose_batch(2))
        c_args, c_kwargs = cart.calls[-1]
        i_args, i_kwargs = inv.calls[-1]
        assert c_kwargs == i_kwargs, f"cap={cap}: keyword shapes differ"
        assert len(c_args) == len(i_args), f"cap={cap}: positional counts differ"
        assert tuple(c_args[3]) == tuple(i_args[3])


def test_the_declared_cap_reaches_the_real_partition():
    """End-to-end through the REAL ``_compute_conditioning``: cap 32 at micro-32
    must execute one angle per chunk, counted as conditioner forwards on all four
    pose ids. Arithmetic on the cap is exactly what cannot catch a threading bug."""
    wrapper = _direct(frame_avg_max_fwd_samples=BFC_CAP)
    wrapper._compute_conditioning(_pose_batch(BFC_MICRO_BATCH))
    for key in yr.POSE_KEYS:
        sizes = wrapper.diffusion.conditioner.conditioners[key].batch_sizes
        assert sizes == [BFC_MICRO_BATCH] * len(C4), f"{key} executed {sizes}"


def test_without_a_declared_cap_the_partition_is_the_module_default():
    """The complement of the test above: the cap is a real knob, not decoration."""
    wrapper = _direct()
    wrapper._compute_conditioning(_pose_batch(BFC_MICRO_BATCH))
    sizes = wrapper.diffusion.conditioner.conditioners["source"].batch_sizes
    assert sizes == [32, 64, 32], sizes            # cap 64 -> 2 angles per chunk


# --------------------------------------------------------------------------- #
# 5. factory plumbing: angles and cap reach fa_cartesian exactly as fa_invariant
# --------------------------------------------------------------------------- #
def test_factory_plumbs_angles_and_cap_identically_for_both_methods(stub_wrapper):
    """The plan's requirement stated as an equality of the whole kwarg dict: any
    cond_method-specific gate on either knob — present or added later — shows up
    here as a difference outside ``cond_method``."""
    captured = {}
    for method in ("fa_invariant", BFC_COND_METHOD):
        create_training_wrapper_from_config(
            _factory_config(cond_method=method, frame_avg_angles=BFC_ANGLES,
                            **{CAP_KEY: BFC_CAP}), object())
        captured[method] = dict(stub_wrapper)

    fa, fc = captured["fa_invariant"], captured[BFC_COND_METHOD]
    assert fc["cond_method"] == BFC_COND_METHOD
    assert fc["frame_avg_angles"] == BFC_ANGLES
    assert fc[CAP_KEY] == BFC_CAP
    fc_stripped, fa_stripped = dict(fc), dict(fa)
    fc_stripped.pop("cond_method"), fa_stripped.pop("cond_method")
    assert fc_stripped == fa_stripped, (
        "fa_cartesian is plumbed differently from fa_invariant: "
        f"{sorted(set(fc_stripped.items()) ^ set(fa_stripped.items()))}")


def test_factory_omits_the_cap_kwarg_for_fa_cartesian_when_absent(stub_wrapper):
    """Absent key -> the literal pre-change construction call (exp_14's contract),
    which is what keeps every arm already in the record byte-identical."""
    create_training_wrapper_from_config(
        _factory_config(cond_method=BFC_COND_METHOD, frame_avg_angles=BFC_ANGLES), object())
    assert CAP_KEY not in stub_wrapper
    assert stub_wrapper["cond_method"] == BFC_COND_METHOD


@pytest.mark.parametrize("bad", [True, False, 32.0, "32", None, [32], 0, -1])
def test_factory_still_rejects_bad_caps_for_fa_cartesian(bad, stub_wrapper):
    """Fail-closed on the RAW value: a coerced ``int("32")`` or a ``True`` would
    arm the wrong chunk plan for a multi-day run without a word of complaint."""
    with pytest.raises(ValueError, match=CAP_KEY):
        create_training_wrapper_from_config(
            _factory_config(cond_method=BFC_COND_METHOD, **{CAP_KEY: bad}), object())


def test_direct_construction_rejects_bad_caps_for_fa_cartesian():
    with pytest.raises(ValueError, match=CAP_KEY):
        _direct(frame_avg_max_fwd_samples=0)


# --------------------------------------------------------------------------- #
# 6. the arm config is FLAC_AR_BF.json plus exactly two declared deltas
# --------------------------------------------------------------------------- #
def _flat(obj, prefix=()):
    """Every leaf of a parsed JSON document, keyed by its full path. A dict
    comparison says only WHETHER two configs differ; the experiment's claim is
    about exactly WHICH leaves differ."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out.update(_flat(v, prefix + (k,)))
        return out
    if isinstance(obj, list):
        out = {}
        for i, v in enumerate(obj):
            out.update(_flat(v, prefix + (i,)))
        return out
    return {prefix: obj}


def _bf():
    return json.loads(BF_CONFIG.read_text())


def _bfc():
    assert BFC_CONFIG.is_file(), f"arm config not found: {BFC_CONFIG}"
    return json.loads(BFC_CONFIG.read_text())


def test_bfc_differs_from_bf_in_exactly_two_training_leaves():
    """The single-mechanism claim, pinned programmatically. ``cond_method`` is the
    mechanism; the cap is decision D5 (a declared knob that postdates B-F's
    training, chosen for draw-schedule parity, not a method change). Anything
    else — an optimizer constant, a metric flag, a ViT setting — would make the
    comparison a multi-factor one that no readout can untangle."""
    flat_bf, flat_bfc = _flat(_bf()), _flat(_bfc())
    added = set(flat_bfc) - set(flat_bf)
    removed = set(flat_bf) - set(flat_bfc)
    changed = {k for k in set(flat_bf) & set(flat_bfc) if flat_bf[k] != flat_bfc[k]}

    assert removed == set(), f"BFC drops {sorted(removed)}"
    assert added == {("training", CAP_KEY)}, f"BFC adds {sorted(added)}"
    assert changed == {("training", "cond_method")}, f"BFC changes {sorted(changed)}"


def test_bfc_is_bf_once_the_two_deltas_are_undone():
    """The same claim from the other side: undo the declared deltas and the whole
    document must be the comparator's, byte-for-byte after parsing."""
    bf, bfc = _bf(), _bfc()
    restored = copy.deepcopy(bfc)
    restored["training"].pop(CAP_KEY)
    restored["training"]["cond_method"] = bf["training"]["cond_method"]
    assert restored == bf


def test_bfc_declares_the_arm_identity():
    training = _bfc()["training"]
    assert training["cond_method"] == BFC_COND_METHOD
    assert training["frame_avg_angles"] == BFC_ANGLES
    assert training[CAP_KEY] == BFC_CAP
    assert training["use_ema"] is True


def test_bfc_keeps_the_comparators_frame_angles():
    """The orbit itself is NOT part of the treatment: BFC and B-F symmetrise over
    the same C4 subgroup, by different mechanisms."""
    assert _bfc()["training"]["frame_avg_angles"] == _bf()["training"]["frame_avg_angles"]
    assert _bfc()["training"]["frame_avg_angles"] == BFC_ANGLES


def test_bfc_builds_through_the_factory(stub_wrapper):
    create_training_wrapper_from_config(_bfc(), object())
    assert stub_wrapper["cond_method"] == BFC_COND_METHOD
    assert stub_wrapper["frame_avg_angles"] == BFC_ANGLES
    assert stub_wrapper[CAP_KEY] == BFC_CAP


def test_bfc_declares_the_intended_training_chunk_plan():
    """announcement 06: cap + micro-batch -> angles per chunk. D5 chose 32 so the
    arm reproduces B-F's per-angle draw schedule (one angle per forward at
    micro-32), asserted by EXECUTION and not by arithmetic on the cap."""
    training = _bfc()["training"]
    assert max(1, training[CAP_KEY] // BFC_MICRO_BATCH) == 1

    cond = _pose_conditioner()
    yr.fa_cartesian_conditioning(cond, _pose_batch(BFC_MICRO_BATCH), DEV,
                                 tuple(training["frame_avg_angles"]),
                                 max_fwd_samples=training[CAP_KEY])
    for key in yr.POSE_KEYS:
        sizes = cond.conditioners[key].batch_sizes[1:]        # after the base pass
        assert len(sizes) == math.ceil((len(BFC_ANGLES) - 1) / 1)
        assert max(sizes) == BFC_MICRO_BATCH, f"{key} executed {sizes}"


# --------------------------------------------------------------------------- #
# 7. the HAA finetune path must keep REJECTING fa_cartesian (plan §1)
# --------------------------------------------------------------------------- #
def test_the_haa_finetune_path_still_rejects_fa_cartesian():
    """``finetune_cond.py`` carries its OWN whitelist, deliberately not extended
    this round: HAA finetuning of this arm is a separate, unapproved experiment,
    and the AR-trained orbit's behaviour on HAA's source-position panoramas has
    not been examined. Nothing in ``test_finetune_cond.py`` would notice the
    method being admitted there — its 'bad' cases predate this arm — so the
    out-of-scope boundary is pinned here, where the widening happened.

    Imported inside the test to keep this module's import surface to the training
    package it is about."""
    import finetune_cond

    assert BFC_COND_METHOD not in finetune_cond.VALID_COND_METHODS, (
        "finetune_cond was widened to fa_cartesian without an approved HAA round")
    cfg = json.loads((_REPO / "src/configs/model_configs/FLAC/AR/FLAC_AR.json").read_text())
    with pytest.raises(ValueError, match="cond_method"):
        finetune_cond.build_finetune_training_config(cfg, BFC_COND_METHOD, 5e-6, [0.0])
