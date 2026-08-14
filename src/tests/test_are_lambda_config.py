"""exp_16 (are_port) Phase 1 — ``training.are_lambda`` is a DECLARED config key.

Seat: Opus 5 Coder (SOP §Roles); plan `worklog/worklog_yixun/exp_16_are_port_claude/
plan_are_port.md` §§1-3.

THE TREATMENT IS A TARGET REPARAMETERISATION, AND IT IS A CONFIG KEY. FLAC's
rectified flow learns ``noise -> z``; ARE-V learns ``noise -> (z - lambda*A(p))``
and adds ``+lambda*A_query`` back before the decode. ``lambda`` therefore has to
be a checkpoint-embedded property of the arm — exactly the shape exp_14 gave the
frame-average cap (announcement 06: declared, not derived) — so that an arm's
objective is auditable after the fact.

The contract pinned here, in the order the plan states it:

1. **Absent key = today, byte for byte.** No ``training.are_lambda`` -> the
   factory issues the LITERAL pre-change construction call (no kwarg at all),
   the wrapper holds ``are_lambda is None``, and the anchor function is NEVER
   invoked. Every arm already in the record — P1 included, which is exp_16's
   free lambda=0 control — must be unaffected down to the call shape.
2. **Fail closed on the RAW value.** ``bool`` is an ``int`` and ``float("1")``
   is 1.0; either would arm a six-day run at the wrong lambda without a word.
   Out-of-range, wrong type, and a declared anchor block with no lambda (or a
   lambda with no anchor block) all abort at the factory AND at direct
   construction.
3. **The algebra, at every dispatch site.** ``training_step`` and
   ``validation_step`` learn ``noise - (z - lambda*A)``; ``test_step`` adds
   ``lambda*A`` back BEFORE the decode. CLAUDE.md's rule is that a new mode
   touches every site; these tests are what makes that mechanical.
4. **Evaluation states its lambda.** ``--are-lambda`` defaults to the embedded
   model_config's value, an explicit flag overrides it (0.0 included — that is
   the AR3 sweep), and whichever wins is recorded in the metrics row. Records for
   non-ARE runs stay byte-identical.
5. **Round-trip**: the key survives into the checkpoint's embedded
   ``model_config``, which is what the launcher's resume gate compares.
"""
import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

import eval_FLAC
import src.training.diffusion as tdiff
from src.data import are_anchor as ar
from src.training.factory import create_training_wrapper_from_config

_REPO = Path(__file__).resolve().parents[2]
BVP1 = _REPO / "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json"
ARE_CONFIG = _REPO / "worklog/worklog_yixun/exp_16_are_port_claude/FLAC_AR_ARE.json"

LAM_KEY = "are_lambda"
ANCHOR_KEY = "are_anchor"
ANCHOR_BLOCK = {"delta_hat": 0.0, "a_g": 0.5}

FS, SAMPLE_SIZE, HOP = 22050, 10240, 1024        # FLAC's real time base
CH, LAT = 4, SAMPLE_SIZE // HOP
H_PANO, W_PANO = 8, 16


# --------------------------------------------------------------------------- #
# stubs
# --------------------------------------------------------------------------- #
class _StubPretransform(nn.Module):
    """Deterministic mean-encoder + a decode that records what it was given."""

    def __init__(self):
        super().__init__()
        self.downsampling_ratio = HOP
        self.enable_grad = False
        self.scale = 1.0
        self.decoded = []

    def encode(self, x):
        return x.reshape(x.shape[0], 1, -1, HOP).mean(-1).expand(x.shape[0], CH, LAT).contiguous()

    def encode_mean(self, x):                     # the anchor seam (no bottleneck sampling)
        return self.encode(x) + 2.0               # a non-zero silence bias

    def decode(self, z):
        self.decoded.append(z.detach().clone())
        return z.reshape(z.shape[0], 1, -1)[:, :, :SAMPLE_SIZE]


class _StubConditioner(nn.Module):
    def forward(self, metadata, device):
        return {"stub": [torch.zeros(len(metadata), 1, 1, device=device)]}


class _StubDiffusion(nn.Module):
    """Minimum surface ``DiffusionCondTrainingWrapper`` touches."""

    def __init__(self):
        super().__init__()
        # a REAL parameter, so the parity tests below can compare gradients and
        # post-step weights rather than only the target tensor
        self.model = nn.Conv1d(CH, CH, 1)
        self.conditioner = _StubConditioner()
        self.diffusion_objective = "rectified_flow"
        self.pretransform = _StubPretransform()
        self.dist_shift = None
        self.io_channels = CH
        self.seen_inputs = []

    def forward(self, x, t, cond=None, cfg_dropout_prob=0.0, **kwargs):
        # x IS x_t, the noised input. Recording it is what lets the algebra test
        # below check the NOISING as well as the target (r1 review finding 4:
        # inspecting `targets` alone passes the forbidden "target from residual,
        # noised input from the original z" implementation).
        self.seen_inputs.append((x.detach().clone(), t.detach().clone()))
        return self.model(x)

    def get_conditioning_inputs(self, conditioning):
        return {}


def _direction(i, j, img_h=H_PANO, img_w=W_PANO):
    import math
    theta = (j + 0.5) * 2.0 * math.pi / img_w - math.pi
    phi = (i + 0.5) * math.pi / img_h - math.pi / 2
    return torch.tensor([math.cos(phi) * math.cos(theta),
                         math.cos(phi) * math.sin(theta),
                         -math.sin(phi)], dtype=torch.float32)


def _depth_pano(value=10.0):
    pano = torch.stack([_direction(i, j) * value
                        for i in range(H_PANO) for j in range(W_PANO)])
    return pano.reshape(H_PANO, W_PANO, 3).permute(2, 0, 1).contiguous()


def _batch(n=2):
    reals = torch.arange(n * SAMPLE_SIZE, dtype=torch.float32).reshape(n, 1, SAMPLE_SIZE) / 1e4
    md = [{"source": _direction(3, 5) * (2.0 + k),
           "depth": _depth_pano(),
           "time_shift": k % 3,          # the loader's RandomTimeShift draw
           "scene": "Toy",
           "padding_mask": torch.ones(SAMPLE_SIZE)} for k in range(n)]
    return reals, md


def _wrapper(**kwargs):
    w = tdiff.DiffusionCondTrainingWrapper(
        _StubDiffusion(), lr=5e-5, use_ema=False, cfg_dropout_prob=0.0, **kwargs)
    w.log_dict = lambda *a, **k: None
    w.log = lambda *a, **k: None
    w._trainer = _FakeTrainer()
    return w


class _FakeTrainer:
    class _Opt:
        param_groups = [{"lr": 5e-5}]

    optimizers = [_Opt()]


class _CaptureLosses(nn.Module):
    """Stands in for ``MultiLoss`` so the test can read the raw ``loss_info``.

    An ``nn.Module`` rather than a closure because ``self.losses`` lives on a
    ``LightningModule``, whose ``__setattr__`` refuses to replace a child module
    with a plain callable.
    """

    def __init__(self, seen):
        super().__init__()
        self.seen = seen

    def forward(self, loss_info):
        self.seen.update(loss_info)
        return torch.zeros((), requires_grad=True), {}


def _capture_targets(wrapper):
    seen = {}
    wrapper.losses = _CaptureLosses(seen)
    return seen  # also reachable as wrapper.losses.seen


ANCHOR_KW = dict(sample_rate=FS, sample_size=SAMPLE_SIZE)


def _anchor_block(**over):
    block = dict(ANCHOR_BLOCK)
    block.update(over)
    return block


# --------------------------------------------------------------------------- #
# 1. factory parsing
# --------------------------------------------------------------------------- #
def _model_config(**training_extra):
    cfg = {
        "model_type": "diffusion_cond",
        "sample_rate": FS,
        "sample_size": SAMPLE_SIZE,
        "training": {"learning_rate": 5e-5, "use_ema": False},
    }
    cfg["training"].update(training_extra)
    return cfg


@pytest.fixture()
def stub_wrapper(monkeypatch):
    captured = {}

    class _Stub:
        def __init__(self, model, **kwargs):
            captured.clear()
            captured.update(kwargs)

    monkeypatch.setattr(tdiff, "DiffusionCondTrainingWrapper", _Stub)
    return captured


def test_absent_key_issues_the_literal_pre_change_call(stub_wrapper):
    create_training_wrapper_from_config(_model_config(), object())
    assert LAM_KEY not in stub_wrapper
    assert ANCHOR_KEY not in stub_wrapper


def test_declared_lambda_is_threaded_with_its_anchor_block(stub_wrapper):
    create_training_wrapper_from_config(
        _model_config(**{LAM_KEY: 1.0, ANCHOR_KEY: _anchor_block()}), object())
    assert stub_wrapper[LAM_KEY] == 1.0
    block = stub_wrapper[ANCHOR_KEY]
    assert block["delta_hat"] == 0.0 and block["a_g"] == 0.5
    # fs / sample_size come from the MODEL config, never from the anchor block
    assert block["sample_rate"] == FS and block["sample_size"] == SAMPLE_SIZE


@pytest.mark.parametrize("bad", [True, False, "1.0", None, [1.0], {}])
def test_factory_rejects_non_numeric_lambda(bad, stub_wrapper):
    with pytest.raises(ValueError, match=LAM_KEY):
        create_training_wrapper_from_config(
            _model_config(**{LAM_KEY: bad, ANCHOR_KEY: _anchor_block()}), object())


@pytest.mark.parametrize("bad", [-0.001, 1.001, 2.0, -1.0])
def test_factory_rejects_out_of_range_lambda(bad, stub_wrapper):
    with pytest.raises(ValueError, match="0.*1"):
        create_training_wrapper_from_config(
            _model_config(**{LAM_KEY: bad, ANCHOR_KEY: _anchor_block()}), object())


def test_factory_requires_the_anchor_block(stub_wrapper):
    with pytest.raises(ValueError, match=ANCHOR_KEY):
        create_training_wrapper_from_config(_model_config(**{LAM_KEY: 1.0}), object())


def test_factory_rejects_an_anchor_block_with_no_lambda(stub_wrapper):
    with pytest.raises(ValueError, match=LAM_KEY):
        create_training_wrapper_from_config(
            _model_config(**{ANCHOR_KEY: _anchor_block()}), object())


@pytest.mark.parametrize("missing", ["delta_hat", "a_g"])
def test_factory_requires_every_calibrated_constant(missing, stub_wrapper):
    block = _anchor_block()
    del block[missing]
    with pytest.raises(ValueError, match=missing):
        create_training_wrapper_from_config(
            _model_config(**{LAM_KEY: 1.0, ANCHOR_KEY: block}), object())


def test_factory_rejects_unknown_anchor_keys(stub_wrapper):
    with pytest.raises(ValueError, match="unknown"):
        create_training_wrapper_from_config(
            _model_config(**{LAM_KEY: 1.0, ANCHOR_KEY: _anchor_block(typo=1)}), object())


def test_anchor_block_may_not_restate_the_model_config(stub_wrapper):
    """``sample_rate``/``sample_size`` are the MODEL config's, so restating them
    in the anchor block could silently disagree with the audio the run trains on."""
    with pytest.raises(ValueError, match="sample_rate"):
        create_training_wrapper_from_config(
            _model_config(**{LAM_KEY: 1.0, ANCHOR_KEY: _anchor_block(sample_rate=16000)}),
            object())


def test_integer_lambda_is_accepted_and_normalised(stub_wrapper):
    create_training_wrapper_from_config(
        _model_config(**{LAM_KEY: 1, ANCHOR_KEY: _anchor_block()}), object())
    assert stub_wrapper[LAM_KEY] == 1.0
    assert isinstance(stub_wrapper[LAM_KEY], float)


# --------------------------------------------------------------------------- #
# 2. direct construction is as fail-closed as the config path
# --------------------------------------------------------------------------- #
def test_direct_construction_defaults_to_no_are():
    assert _wrapper().are_lambda is None


@pytest.mark.parametrize("bad", [True, False, "1.0", [1.0]])
def test_direct_construction_rejects_non_numeric_lambda(bad):
    with pytest.raises(ValueError, match=LAM_KEY):
        _wrapper(are_lambda=bad, are_anchor=_anchor_block(**ANCHOR_KW))


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_direct_construction_rejects_out_of_range_lambda(bad):
    with pytest.raises(ValueError, match=LAM_KEY):
        _wrapper(are_lambda=bad, are_anchor=_anchor_block(**ANCHOR_KW))


def test_direct_construction_requires_an_anchor_block():
    with pytest.raises(ValueError, match=ANCHOR_KEY):
        _wrapper(are_lambda=1.0)


# --------------------------------------------------------------------------- #
# 3. the algebra at every dispatch site
# --------------------------------------------------------------------------- #
def test_absent_lambda_never_calls_the_anchor_function(monkeypatch):
    """Raw spy, exp_14 F1 style: the assertion is on the CALL, not on its effect.
    A control arm that computed anchors and multiplied them by zero would be
    numerically identical and scientifically different (extra work, extra
    surface); the contract is that the anchor code is not entered at all."""
    calls = []
    monkeypatch.setattr(tdiff, "compute_are_anchors",
                        lambda *a, **k: calls.append((a, k)))
    w = _wrapper()
    seen = _capture_targets(w)
    reals, md = _batch()
    w.training_step((reals, md), 0)
    assert calls == []
    # ...and the objective is the UNMODIFIED one. Round 1 asserted
    # `targets + z == targets + z` here, which is true of anything; the real
    # statement is the rectified-flow invariant x_t - t*u == z, evaluated against
    # the raw latent rather than any residual.
    z = w.diffusion.pretransform.encode(reals)
    x_t, t = w.diffusion.seen_inputs[-1]
    assert torch.allclose(x_t - t[:, None, None] * seen["targets"], z, atol=1e-5)


def _residual_invariant(wrapper):
    """``x_t - t*u`` for the last step, which must equal the flow's start point.

    Rectified flow uses ``alphas = 1-t``, ``sigmas = t``, so with a start point
    ``r`` and noise ``n``::

        x_t = (1-t)*r + t*n        u = n - r
        x_t - t*u = (1-t)*r + t*n - t*n + t*r = r

    The noise cancels exactly, which is what makes this a JOINT statement about
    the noised input and the target: it is only ``r`` when BOTH were built from
    ``r``. If ``x_t`` came from ``z`` while ``u`` came from ``r = z - lam*A`` the
    expression collapses to ``z - t*lam*A``; if the two are swapped it collapses
    to ``z - (1-t)*lam*A``. Neither equals ``r`` except at a single degenerate
    ``t``, so one test discriminates both forbidden variants.
    """
    x_t, t = wrapper.diffusion.seen_inputs[-1]
    targets = wrapper.losses.seen["targets"]
    return x_t - t[:, None, None] * targets


def test_both_the_noised_input_and_the_target_come_from_the_same_residual(monkeypatch):
    """The load-bearing algebra check (r1 review finding 4).

    Round 1 inspected ``targets`` only, so the forbidden implementation -- target
    from the residual, noised input from the original ``z`` -- would have passed.
    This captures the MODEL INPUT and asserts the joint invariant, so it fails for
    that variant and for its mirror image.
    """
    anchors = torch.full((2, CH, LAT), 0.25)
    monkeypatch.setattr(tdiff, "compute_are_anchors", lambda *a, **k: anchors)
    reals, md = _batch()
    lam = 1.0

    w = _wrapper(are_lambda=lam, are_anchor=_anchor_block(**ANCHOR_KW))
    _capture_targets(w)
    torch.manual_seed(11)
    w.training_step((reals, md), 0)

    z = w.diffusion.pretransform.encode(reals)
    residual = z - lam * anchors
    assert torch.allclose(_residual_invariant(w), residual, atol=1e-5)
    # and it is genuinely NOT the unmodified latent, or the assertion would hold
    # for the no-treatment implementation too
    assert not torch.allclose(_residual_invariant(w), z, atol=1e-3)


def test_the_invariant_rejects_both_forbidden_variants():
    """Non-vacuity, on the criterion itself.

    Constructs the two mixed implementations by hand and shows the invariant
    refuses them. (The live mutation experiment -- editing diffusion.py to the
    forbidden variant and observing the test above go red -- is recorded in the
    round-2 review notes; this keeps the demonstration in the suite.)
    """
    torch.manual_seed(3)
    z = torch.randn(2, CH, LAT)
    a = torch.full((2, CH, LAT), 0.25)
    n = torch.randn(2, CH, LAT)
    t = torch.tensor([0.3, 0.7])
    lam = 1.0
    r = z - lam * a

    def invariant(x_t, u):
        return x_t - t[:, None, None] * u

    # correct: both from r
    assert torch.allclose(invariant((1 - t)[:, None, None] * r + t[:, None, None] * n,
                                    n - r), r, atol=1e-6)
    # forbidden A: noised input from z, target from r
    bad_a = invariant((1 - t)[:, None, None] * z + t[:, None, None] * n, n - r)
    assert not torch.allclose(bad_a, r, atol=1e-3)
    # forbidden B: noised input from r, target from z
    bad_b = invariant((1 - t)[:, None, None] * r + t[:, None, None] * n, n - z)
    assert not torch.allclose(bad_b, r, atol=1e-3)


def test_validation_noising_also_uses_the_residual(monkeypatch):
    """Dispatch site 2 gets the same joint check, not just an "it was called" spy."""
    anchors = torch.full((2, CH, LAT), 0.25)
    monkeypatch.setattr(tdiff, "compute_are_anchors", lambda *a, **k: anchors)
    reals, md = _batch()
    lam = 1.0
    w = _wrapper(are_lambda=lam, are_anchor=_anchor_block(**ANCHOR_KW))
    w.validation_timesteps = [0.5]
    w.validation_step_outputs = {"val/loss_0.5": []}
    w.validation_step((reals, md), 0)

    z = w.diffusion.pretransform.encode(reals)
    residual = z - lam * anchors
    x_t, t = w.diffusion.seen_inputs[-1]
    # validation builds `targets` inline rather than through self.losses, so the
    # noise is recovered from the noised input the model actually saw and the
    # target is rebuilt exactly as validation_step does (u = noise - start point).
    noise = (x_t - (1 - t)[:, None, None] * residual) / t[:, None, None]
    u = noise - residual
    assert torch.allclose(x_t - t[:, None, None] * u, residual, atol=1e-5)
    # the invariant would not hold against the raw latent -- i.e. validation really
    # is noising the residual, not z
    assert not torch.allclose(x_t - t[:, None, None] * u, z, atol=1e-3)


def test_training_target_is_the_anchor_residual(monkeypatch):
    anchors = torch.full((2, CH, LAT), 0.25)
    calls = []

    def _spy(bank, metadata, device):
        calls.append((bank, metadata, device))
        return anchors

    monkeypatch.setattr(tdiff, "compute_are_anchors", _spy)

    reals, md = _batch()
    lam = 1.0
    w = _wrapper(are_lambda=lam, are_anchor=_anchor_block(**ANCHOR_KW))
    seen = _capture_targets(w)

    torch.manual_seed(11)
    w.training_step((reals, md), 0)
    assert len(calls) == 1
    residual_targets = seen["targets"]

    # the same run with lambda -> 0 anchors reproduces the plain objective, so
    # the delta between the two IS lambda*A (the noise draw is seeded identically)
    monkeypatch.setattr(tdiff, "compute_are_anchors",
                        lambda *a, **k: torch.zeros_like(anchors))
    w2 = _wrapper(are_lambda=lam, are_anchor=_anchor_block(**ANCHOR_KW))
    seen2 = _capture_targets(w2)
    torch.manual_seed(11)
    w2.training_step((reals, md), 0)
    assert torch.allclose(residual_targets - seen2["targets"], lam * anchors, atol=1e-6)


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
def test_lambda_scales_the_residual_linearly(monkeypatch, lam):
    anchors = torch.full((2, CH, LAT), 0.25)
    monkeypatch.setattr(tdiff, "compute_are_anchors", lambda *a, **k: anchors)
    reals, md = _batch()

    w = _wrapper(are_lambda=lam, are_anchor=_anchor_block(**ANCHOR_KW))
    seen = _capture_targets(w)
    torch.manual_seed(3)
    w.training_step((reals, md), 0)

    monkeypatch.setattr(tdiff, "compute_are_anchors", lambda *a, **k: torch.zeros_like(anchors))
    w0 = _wrapper(are_lambda=lam, are_anchor=_anchor_block(**ANCHOR_KW))
    seen0 = _capture_targets(w0)
    torch.manual_seed(3)
    w0.training_step((reals, md), 0)

    assert torch.allclose(seen["targets"] - seen0["targets"], lam * anchors, atol=1e-6)


def test_validation_step_uses_the_same_residual_target(monkeypatch):
    anchors = torch.full((2, CH, LAT), 0.25)
    calls = []
    monkeypatch.setattr(tdiff, "compute_are_anchors",
                        lambda *a, **k: (calls.append(1), anchors)[1])
    reals, md = _batch()
    w = _wrapper(are_lambda=1.0, are_anchor=_anchor_block(**ANCHOR_KW))
    w.validation_timesteps = [0.5]
    w.validation_step_outputs = {"val/loss_0.5": []}
    w.validation_step((reals, md), 0)
    assert len(calls) == 1, "validation is a dispatch site too (CLAUDE.md)"


def test_validation_step_without_lambda_never_calls_the_anchor(monkeypatch):
    calls = []
    monkeypatch.setattr(tdiff, "compute_are_anchors", lambda *a, **k: calls.append(1))
    reals, md = _batch()
    w = _wrapper()
    w.validation_timesteps = [0.5]
    w.validation_step_outputs = {"val/loss_0.5": []}
    w.validation_step((reals, md), 0)
    assert calls == []


def _prep_test_step(w, monkeypatch, sampled):
    monkeypatch.setattr(tdiff, "sample_discrete_euler",
                        lambda model, noise, steps, **kw: sampled)
    w.set_test_config(samples=SAMPLE_SIZE, cfg_scale=1.0, steps=1, sample_rate=FS,
                      audio_channels=1, metrics={}, store_predictions=False)

    class _Stub:
        def __init__(self):
            self.calls = []

        def update_metrics(self, *a, **k):
            self.calls.append((a, k))

    w.metric_callback = _Stub()
    return w


def test_test_step_adds_the_anchor_back_before_the_decode(monkeypatch):
    """The decode must see ``residual + lambda*A`` — one decode, of one latent,
    with the anchor already restored."""
    anchors = torch.full((2, CH, LAT), 0.25)
    monkeypatch.setattr(tdiff, "compute_are_anchors", lambda *a, **k: anchors)
    sampled = torch.arange(2 * CH * LAT, dtype=torch.float32).reshape(2, CH, LAT) / 100.0

    lam = 1.0
    w = _wrapper(are_lambda=lam, are_anchor=_anchor_block(**ANCHOR_KW))
    _prep_test_step(w, monkeypatch, sampled)
    reals, md = _batch()
    w.test_step((reals, md), 0)

    assert len(w.diffusion.pretransform.decoded) == 1
    assert torch.allclose(w.diffusion.pretransform.decoded[0], sampled + lam * anchors,
                          atol=1e-6)


def test_test_step_without_lambda_decodes_the_raw_sample(monkeypatch):
    calls = []
    monkeypatch.setattr(tdiff, "compute_are_anchors", lambda *a, **k: calls.append(1))
    sampled = torch.ones(2, CH, LAT)
    w = _wrapper()
    _prep_test_step(w, monkeypatch, sampled)
    reals, md = _batch()
    w.test_step((reals, md), 0)
    assert calls == []
    assert torch.equal(w.diffusion.pretransform.decoded[0], sampled)


def test_anchor_latent_length_mismatch_is_a_hard_error(monkeypatch):
    monkeypatch.setattr(tdiff, "compute_are_anchors",
                        lambda *a, **k: torch.zeros(2, CH, LAT + 1))
    reals, md = _batch()
    w = _wrapper(are_lambda=1.0, are_anchor=_anchor_block(**ANCHOR_KW))
    _capture_targets(w)
    with pytest.raises(ValueError, match="shape"):
        w.training_step((reals, md), 0)


def test_are_requires_a_pretransform():
    w = _wrapper(are_lambda=1.0, are_anchor=_anchor_block(**ANCHOR_KW))
    w.diffusion.pretransform = None
    with pytest.raises(ValueError, match="pretransform"):
        w._are_anchor_bank()


# --------------------------------------------------------------------------- #
# 4. evaluation states its lambda
# --------------------------------------------------------------------------- #
def test_resolve_are_lambda_defaults_to_the_embedded_config():
    lam, source = eval_FLAC.resolve_are_lambda(
        {LAM_KEY: 1.0, ANCHOR_KEY: ANCHOR_BLOCK}, None)
    assert lam == 1.0 and source == "model_config"


def test_resolve_are_lambda_is_none_for_a_non_are_arm():
    assert eval_FLAC.resolve_are_lambda({}, None) == (None, None)


@pytest.mark.parametrize("cli", [0.0, 0.5, 1.0])
def test_resolve_are_lambda_cli_overrides_the_config(cli):
    lam, source = eval_FLAC.resolve_are_lambda(
        {LAM_KEY: 1.0, ANCHOR_KEY: ANCHOR_BLOCK}, cli)
    assert lam == cli and source == "cli"


def test_resolve_are_lambda_refuses_a_cli_lambda_on_a_non_are_arm():
    """An arm with no anchor block cannot compute ``A``; silently ignoring the
    flag would report a lambda that influenced nothing (announcement 05)."""
    with pytest.raises(ValueError, match=ANCHOR_KEY):
        eval_FLAC.resolve_are_lambda({}, 1.0)


@pytest.mark.parametrize("bad", [-0.5, 1.5])
def test_resolve_are_lambda_rejects_out_of_range(bad):
    with pytest.raises(ValueError, match="0.*1"):
        eval_FLAC.resolve_are_lambda({LAM_KEY: 1.0, ANCHOR_KEY: ANCHOR_BLOCK}, bad)


def test_metrics_record_is_byte_identical_without_are():
    record = eval_FLAC.build_metrics_record({"T60": 1.0}, "ck.ckpt", 0.0, "vanilla", None)
    assert LAM_KEY not in record and "are_lambda_source" not in record


def test_metrics_record_carries_the_applied_lambda():
    record = eval_FLAC.build_metrics_record(
        {"T60": 1.0}, "ck.ckpt", 0.0, "vanilla", None,
        are_lambda=0.5, are_lambda_source="cli", are_anchor=ANCHOR_BLOCK)
    assert record[LAM_KEY] == 0.5
    assert record["are_lambda_source"] == "cli"
    assert record[ANCHOR_KEY] == ANCHOR_BLOCK


def test_output_paths_are_unchanged_without_are():
    legacy = eval_FLAC.build_output_paths("d/ck.ckpt", 1, 1.0, "e")
    assert legacy == eval_FLAC.build_output_paths("d/ck.ckpt", 1, 1.0, "e", are_lambda=None)


@pytest.mark.parametrize("lam,token", [(0.0, "_are0"), (0.5, "_are0p5"), (1.0, "_are1")])
def test_output_paths_separate_the_lambda_sweep(lam, token):
    """AR3 evaluates one checkpoint at lambda in {0, 0.5, 1}; without a suffix the
    three cells would overwrite one another's metrics file."""
    p = eval_FLAC.build_output_paths("d/ck.ckpt", 1, 1.0, "e", are_lambda=lam)
    assert p["metrics"].endswith(f"{token}.json")
    assert p["predictions"].endswith(f"{token}.pt")


def test_eval_addback_algebra():
    fakes = torch.arange(2 * CH * LAT, dtype=torch.float32).reshape(2, CH, LAT)
    anchors = torch.full((2, CH, LAT), 0.5)
    out = ar.apply_anchor_addback(fakes, anchors, 0.5)
    assert torch.equal(out, fakes + 0.25)
    assert torch.equal(ar.apply_anchor_addback(fakes, anchors, 0.0), fakes)


def test_eval_addback_rejects_a_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        ar.apply_anchor_addback(torch.zeros(2, CH, LAT), torch.zeros(2, CH, LAT + 1), 1.0)


def test_cli_exposes_are_lambda():
    src = (_REPO / "eval_FLAC.py").read_text()
    assert '"--are-lambda"' in src
    assert "default=None" in src


# --------------------------------------------------------------------------- #
# 5. round-trip into the checkpoint, and the arm config itself
# --------------------------------------------------------------------------- #
def test_lambda_survives_the_model_config_checkpoint_round_trip(tmp_path):
    from train import ModelConfigEmbedderCallback

    cfg = _model_config(**{LAM_KEY: 1.0, ANCHOR_KEY: _anchor_block()})
    ckpt = {}
    ModelConfigEmbedderCallback(cfg).on_save_checkpoint(None, None, ckpt)
    p = tmp_path / "c.ckpt"
    torch.save(ckpt, p)
    reloaded = torch.load(p, map_location="cpu", weights_only=False)
    assert reloaded["model_config"]["training"][LAM_KEY] == 1.0
    assert reloaded["model_config"]["training"][ANCHOR_KEY]["a_g"] == 0.5


@pytest.mark.skipif(not ARE_CONFIG.is_file(), reason="arm config not written yet")
def test_are_config_is_bvp1_plus_only_the_are_keys():
    arm = json.loads(ARE_CONFIG.read_text())
    base = json.loads(BVP1.read_text())
    stripped = copy.deepcopy(arm)
    del stripped["training"][LAM_KEY]
    del stripped["training"][ANCHOR_KEY]
    assert stripped == base, "FLAC_AR_ARE.json is no longer BVp1 + exactly the ARE keys"
    assert arm["training"][LAM_KEY] == 1.0
    assert isinstance(arm["training"][LAM_KEY], float)
    block = arm["training"][ANCHOR_KEY]
    assert set(block) == {"delta_hat", "a_g"}
    for v in block.values():
        assert isinstance(v, float)


@pytest.mark.skipif(not ARE_CONFIG.is_file(), reason="arm config not written yet")
def test_are_config_builds_through_the_factory(stub_wrapper):
    arm = json.loads(ARE_CONFIG.read_text())
    create_training_wrapper_from_config(arm, object())
    assert stub_wrapper[LAM_KEY] == 1.0
    assert stub_wrapper[ANCHOR_KEY]["sample_rate"] == arm["sample_rate"]


@pytest.mark.skipif(not ARE_CONFIG.is_file(), reason="arm config not written yet")
def test_are_config_worst_case_frames_hold_for_ar():
    arm = json.loads(ARE_CONFIG.read_text())
    block = dict(arm["training"][ANCHOR_KEY])
    block.update(sample_rate=arm["sample_rate"], sample_size=arm["sample_size"])
    cfg = ar.anchor_config_from_dict(
        block, hop=arm["model"]["pretransform"]["config"]["downsampling_ratio"])
    rep = ar.assert_worst_case_frames(cfg)
    assert rep["ok"] is True and rep["required_frames"] <= cfg.early_frames


# --------------------------------------------------------------------------- #
# 6. the lambda=0 / key-absent parity the control claim rests on
# --------------------------------------------------------------------------- #
def _one_step(**kwargs):
    """One full optimisation step, returning everything that could differ.

    Both arms are CONSTRUCTED under the same seed (``SobolEngine(scramble=True)``
    draws at construction) and STEPPED under the same seed, so any difference is
    attributable to the ARE plumbing and nothing else.
    """
    torch.manual_seed(0)
    w = tdiff.DiffusionCondTrainingWrapper(
        _StubDiffusion(), lr=1e-3, use_ema=True, cfg_dropout_prob=0.0, **kwargs)
    w.log_dict = lambda *a, **k: None
    w.log = lambda *a, **k: None
    w._trainer = _FakeTrainer()
    opt = torch.optim.SGD(w.diffusion.parameters(), lr=0.1)

    reals, md = _batch(4)
    torch.manual_seed(1234)
    loss = w.training_step((reals, md), 0)
    loss.backward()
    grads = [p.grad.detach().clone() for p in w.diffusion.parameters()]
    opt.step()
    w.on_before_zero_grad()
    params = [p.detach().clone() for p in w.diffusion.parameters()]
    ema = [p.detach().clone() for p in w.diffusion_ema.ema_model.parameters()]
    return loss.detach().clone(), grads, params, ema, torch.randn(8)


def _assert_step_identical(a, b, what):
    la, ga, pa, ea, ra = a
    lb, gb, pb, eb, rb = b
    assert torch.equal(la, lb), f"{what}: loss differs"
    for x, y in zip(ga, gb):
        assert torch.equal(x, y), f"{what}: gradients differ"
    for x, y in zip(pa, pb):
        assert torch.equal(x, y), f"{what}: post-step weights differ"
    for x, y in zip(ea, eb):
        assert torch.equal(x, y), f"{what}: EMA weights differ"
    assert torch.equal(ra, rb), f"{what}: the global RNG stream was displaced"


def test_declared_lambda_zero_is_bit_identical_to_the_absent_key():
    """The claim exp_16's control rests on, made mechanical: an arm that DECLARES
    ``are_lambda: 0.0`` runs the entire anchor pipeline (encode, bias subtraction,
    LOS gate, truncation) and must still produce a bit-identical optimisation
    step -- same loss, same gradients, same weights, same EMA, and above all the
    same RNG stream, since ``0.0 * A`` is exactly zero and the anchor path draws
    from no generator."""
    _assert_step_identical(_one_step(),
                           _one_step(are_lambda=0.0, are_anchor=_anchor_block(**ANCHOR_KW)),
                           "lambda=0.0 vs key-absent")


def test_an_are_arm_consumes_exactly_the_same_rng_stream():
    """Even at lambda=1 the RNG stream must be untouched: the anchor is a
    deterministic function of geometry, so the treatment arm sees the SAME noise
    draws its control would have seen. (The weights legitimately differ -- that is
    the treatment.)"""
    _, _, _, _, rng_plain = _one_step()
    _, _, _, _, rng_are = _one_step(are_lambda=1.0, are_anchor=_anchor_block(**ANCHOR_KW))
    assert torch.equal(rng_plain, rng_are)


def test_lambda_one_actually_changes_the_step():
    """The mirror of the parity tests: a real lambda must NOT be inert, or the
    two tests above would be passing for the wrong reason."""
    _, _, params_plain, _, _ = _one_step()
    _, _, params_are, _, _ = _one_step(are_lambda=1.0, are_anchor=_anchor_block(**ANCHOR_KW))
    assert any(not torch.equal(x, y) for x, y in zip(params_plain, params_are))


# --------------------------------------------------------------------------- #
# 7. the add-back is bound to the CHECKPOINT, not to --model-config
#    (r1 code review, finding 2)
# --------------------------------------------------------------------------- #
def _are_cfg(a_g=0.5, lam=1.0, delta_hat=0.0):
    return {
        "model_type": "diffusion_cond", "sample_rate": FS, "sample_size": SAMPLE_SIZE,
        "training": {"use_ema": True, LAM_KEY: lam,
                     ANCHOR_KEY: {"delta_hat": delta_hat, "a_g": a_g}},
    }


def _plain_cfg():
    return {"model_type": "diffusion_cond", "sample_rate": FS,
            "sample_size": SAMPLE_SIZE, "training": {"use_ema": True}}


def test_non_are_evaluation_is_untouched_by_the_binding():
    """Byte-compat: an arm that declares no ARE resolves to None and never
    compares anything, so every row already in the record is unaffected -- even
    when the checkpoint's embedded config differs from the file (which is common
    for released checkpoints)."""
    embedded = _plain_cfg()
    embedded["training"]["something_else"] = 7
    assert eval_FLAC.resolve_are_from_checkpoint(embedded, _plain_cfg(), None) == (
        None, None, None)
    # ...and a checkpoint with no embedded config at all is still fine
    assert eval_FLAC.resolve_are_from_checkpoint(None, _plain_cfg(), None) == (
        None, None, None)


def test_are_evaluation_takes_the_anchor_from_the_checkpoint():
    lam, source, anchor = eval_FLAC.resolve_are_from_checkpoint(
        _are_cfg(), _are_cfg(), None)
    assert lam == 1.0 and source == "model_config"
    assert anchor == {"delta_hat": 0.0, "a_g": 0.5}


def test_are_evaluation_rejects_a_checkpoint_with_no_embedded_config():
    with pytest.raises(ValueError, match="embedded 'model_config'"):
        eval_FLAC.resolve_are_from_checkpoint(None, _are_cfg(), None)


def test_are_evaluation_rejects_a_mismatched_anchor():
    """THE defect this closes: a checkpoint trained against one anchor being
    evaluated against another, loading cleanly and recording false provenance."""
    with pytest.raises(ValueError, match=r"are_anchor\.a_g"):
        eval_FLAC.resolve_are_from_checkpoint(_are_cfg(a_g=0.5), _are_cfg(a_g=0.9), None)


def test_are_evaluation_rejects_a_mismatched_lambda():
    with pytest.raises(ValueError, match=LAM_KEY):
        eval_FLAC.resolve_are_from_checkpoint(_are_cfg(lam=1.0), _are_cfg(lam=0.5), None)


def test_are_evaluation_rejects_a_mismatched_delta_hat():
    with pytest.raises(ValueError, match="delta_hat"):
        eval_FLAC.resolve_are_from_checkpoint(
            _are_cfg(delta_hat=0.0), _are_cfg(delta_hat=3.0), None)


def test_are_evaluation_is_type_strict():
    """``1 == 1.0`` in Python; the factory rejects the int, so the binding must
    too or a config that could never have trained this checkpoint would pass."""
    embedded = _are_cfg()
    embedded["training"][LAM_KEY] = 1          # int, not float
    with pytest.raises(ValueError, match="type int != float"):
        eval_FLAC.resolve_are_from_checkpoint(embedded, _are_cfg(), None)


def test_an_are_checkpoint_cannot_be_evaluated_through_a_non_are_config():
    """Otherwise the add-back would be silently skipped and the RESIDUAL reported
    as if it were an impulse response."""
    with pytest.raises(ValueError, match="type-strict"):
        eval_FLAC.resolve_are_from_checkpoint(_are_cfg(), _plain_cfg(), None)


@pytest.mark.parametrize("dose", [0.0, 0.5, 1.0])
def test_the_cli_overrides_only_the_dose(dose):
    """AR3's sweep is a DOSE sweep: lambda may be overridden, the anchor may not."""
    lam, source, anchor = eval_FLAC.resolve_are_from_checkpoint(
        _are_cfg(a_g=0.5), _are_cfg(a_g=0.5), dose)
    assert lam == dose and source == "cli"
    assert anchor == {"delta_hat": 0.0, "a_g": 0.5}, (
        "the CLI must not be able to change WHICH anchor is added back")


def test_a_cli_dose_still_requires_a_matching_pair():
    with pytest.raises(ValueError, match="type-strict"):
        eval_FLAC.resolve_are_from_checkpoint(_are_cfg(a_g=0.5), _are_cfg(a_g=0.9), 0.5)


def test_evaluate_model_refuses_a_mismatched_pair_before_building_anything(tmp_path, monkeypatch):
    """End to end through ``evaluate_model``: the refusal must happen before a
    single model is constructed, so it costs nothing and cannot reach a GPU."""
    built = []
    monkeypatch.setattr(eval_FLAC, "create_model_from_config",
                        lambda cfg: built.append(cfg))

    cfg_path = tmp_path / "arm.json"
    cfg_path.write_text(json.dumps(_are_cfg(a_g=0.9)))
    ckpt_path = tmp_path / "c.ckpt"
    torch.save({"state_dict": {}, "model_config": _are_cfg(a_g=0.5)}, ckpt_path)

    with pytest.raises(ValueError, match=r"are_anchor\.a_g"):
        eval_FLAC.evaluate_model(str(cfg_path), "unused.json", str(ckpt_path),
                                 steps=1, cfg_scale=1.0, device="cpu")
    assert built == [], "a model was constructed before the mismatch was caught"
