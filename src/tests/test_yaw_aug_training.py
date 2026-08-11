"""exp_15 round 1 — training-side random-yaw augmentation (plan §§3.1, 3.3-3, 6.2, 6.3, 6.5).

exp_15 trains vanilla FLAC with an independent random yaw applied to every
training sample's *conditioning* (panorama roll + all four pose fields rotated
together; ``reals`` / ``context_audio`` untouched, because a rigid yaw of the
whole scene leaves the RIR unchanged). The treatment must be invisible when it
is switched off — the arm is compared against a historical control trained
before the hook existed — and exactly reproducible when it is switched on, at
any resume point.

This module opens with the **golden disabled-path regression** (plan §3.3-3),
captured at the pre-change commit *before* any production edit of this round:
one whole ``training_step`` on a seeded synthetic batch, recording the metadata
that reaches the conditioner, the conditioning tensors that go onward, the loss,
and the global python/NumPy/torch RNG states before and after. Everything is
stored as sha256 digests of raw tensor bytes plus the loss in ``float.hex()``
form, so the comparison is exact, not approximate, on this platform. If this
test ever goes red, the "``yaw_aug`` absent ⇒ nothing changed" claim is dead.

Harness notes:

* The model is a tiny CPU-only ``diffusion_cond`` built through the *real*
  factories (no pretransform, no EMA), following ``test_cond_dispatch.py``.
  Unlike that module the DiT forward is **real**, so the fixture covers the
  whole step; that requires ``src.models.transformer.flash_attn_func`` to be
  neutralised, since the installed flash-attn kernel is CUDA-only and this test
  is CPU-only. Both the capture and the replay disable it identically.
* Regenerate the fixture deliberately (never to "fix" a red test), from the repo
  root, at a commit where the disabled path is known-good, passing that commit
  explicitly — the writer refuses to stamp the record "unknown"::

      python src/tests/test_yaw_aug_training.py --write-golden $(git rev-parse HEAD)

  The committed fixture was captured at ``d3a0312``, the parent of the first
  exp_15 commit, i.e. before any production file of this round was edited.
"""
import hashlib
import json
import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

import src.models.transformer as _transformer
import src.training.diffusion as tdiff
from src.data import yaw_rotation as yr
from src.models.factory import create_model_from_config
from src.training.factory import create_training_wrapper_from_config


GOLDEN_PATH = Path(__file__).resolve().parent / "fixtures" / "exp15_yaw_aug_disabled_golden.json"

# Master seed of the golden capture. Seeds python/NumPy/torch before the model
# is even built (the wrapper's SobolEngine(scramble=True) consumes global torch
# randomness at construction, so construction is part of the deterministic run).
GOLDEN_SEED = 20250810


# --------------------------------------------------------------------------- #
# tiny CPU diffusion_cond model (real factories, no pretransform, no EMA)
# --------------------------------------------------------------------------- #
def _base_config():
    """A minimal but *forward-consistent* diffusion_cond config.

    ``embed_dim`` 64 is the smallest width the DiT's rotary embedding admits
    (``RotaryEmbedding(max(dim_heads // 2, 32))`` needs a head dim >= 64), so it
    is not shrunk further.
    """
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
                    "io_channels": 4, "embed_dim": 64, "depth": 1, "num_heads": 1,
                    "cond_token_dim": 32, "global_cond_dim": 32,
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


def _build_wrapper(config=None, **training_overrides):
    cfg = _base_config() if config is None else config
    cfg["training"].update(training_overrides)
    model = create_model_from_config(cfg)
    return create_training_wrapper_from_config(cfg, model)


def _attach_stub_trainer(wrapper, global_step=0, global_rank=0):
    """Everything ``training_step`` reads off a Trainer, and nothing more.

    ``LightningModule.global_step`` / ``global_rank`` proxy to the attached
    trainer, so the stub also fixes the counter inputs of the augmentation seed.
    """
    wrapper.log_dict = lambda *a, **k: None
    wrapper.trainer = types.SimpleNamespace(
        optimizers=[types.SimpleNamespace(param_groups=[{"lr": 5e-6}])],
        global_step=global_step,
        global_rank=global_rank,
    )


def _make_md(seed, img_w=16, height=4):
    g = torch.Generator().manual_seed(seed)
    return {
        "source": torch.randn(3, generator=g),
        "source_vit": torch.randn(3, generator=g),
        "context_poses": torch.randn(2, 3, generator=g),
        "context_poses_vit": torch.randn(2, 3, generator=g),
        "context_audio": torch.randn(2, 8, generator=g),
        "depth": torch.randn(3, height, img_w, generator=g),
        "padding_mask": torch.ones(64),
        "scene": f"scene_{seed}",
    }


def _batch(n=2, img_w=16):
    g = torch.Generator().manual_seed(9000)
    return torch.randn(n, 4, 64, generator=g), [_make_md(s, img_w=img_w) for s in range(n)]


@pytest.fixture
def no_flash(monkeypatch):
    """Force the CPU ``scaled_dot_product_attention`` fallback in the DiT.

    ``apply_attn`` picks flash-attn purely on module-global availability, never
    on device, and the installed kernel has no CPU implementation.
    """
    monkeypatch.setattr(_transformer, "flash_attn_func", None)


# --------------------------------------------------------------------------- #
# digests — exact, byte-level, small enough to commit
# --------------------------------------------------------------------------- #
def _tensor_digest(t):
    arr = t.detach().cpu().contiguous().numpy()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }


def _rng_digest():
    """One digest over the three global RNG streams a training step could touch."""
    h = hashlib.sha256()
    h.update(repr(random.getstate()).encode("utf-8"))
    np_state = np.random.get_state()
    h.update(str(np_state[0]).encode("utf-8"))
    h.update(np.asarray(np_state[1], dtype=np.uint32).tobytes())
    h.update(repr(tuple(np_state[2:])).encode("utf-8"))
    h.update(torch.get_rng_state().numpy().tobytes())
    return h.hexdigest()


_MD_TENSOR_KEYS = ("source", "source_vit", "context_poses", "context_poses_vit",
                   "context_audio", "depth", "padding_mask")


def _metadata_digests(metadata):
    return [
        {k: _tensor_digest(md[k]) for k in _MD_TENSOR_KEYS if k in md}
        for md in metadata
    ]


# --------------------------------------------------------------------------- #
# the capture: one whole training_step on the disabled path
# --------------------------------------------------------------------------- #
def _capture_disabled_path():
    """Run the reference step and return the record the fixture pins.

    Deterministic end to end: master seed -> model construction -> synthetic
    batch -> one ``training_step``. Returns metadata-as-seen-by-the-conditioner,
    the conditioning handed onward, the loss, and RNG snapshots either side.
    """
    random.seed(GOLDEN_SEED)
    np.random.seed(GOLDEN_SEED)
    torch.manual_seed(GOLDEN_SEED)

    wrapper = _build_wrapper()
    _attach_stub_trainer(wrapper, global_step=7, global_rank=0)

    seen = {}
    original = wrapper._compute_conditioning

    def _spy(metadata):
        seen["metadata"] = _metadata_digests(metadata)
        out = original(metadata)
        seen["conditioning"] = {
            key: _tensor_digest(value[0]) for key, value in sorted(out.items())
        }
        return out

    wrapper._compute_conditioning = _spy

    reals, metadata = _batch(2)
    rng_before = _rng_digest()
    loss = wrapper.training_step((reals, metadata), 0)
    rng_after = _rng_digest()

    return {
        "metadata_into_conditioner": seen["metadata"],
        "conditioning": seen["conditioning"],
        "loss_hex": float(loss.item()).hex(),
        "loss_repr": repr(float(loss.item())),
        "rng_before": rng_before,
        "rng_after": rng_after,
    }


def _capture_sha_from_argv(argv):
    """Parse (and insist on) the capture SHA of a regeneration run.

    The fixture's whole evidential value is that it was captured at a *named*
    pre-change commit; a record stamped "unknown" would prove nothing about what
    the disabled path did or when (review finding 5).
    """
    if "--write-golden" not in argv:
        raise SystemExit(
            "refusing to overwrite the golden fixture without --write-golden"
        )
    index = argv.index("--write-golden") + 1
    if index >= len(argv) or argv[index].startswith("-"):
        raise SystemExit(
            "refusing to write the golden fixture without an explicit capture SHA. "
            "Run: python src/tests/test_yaw_aug_training.py --write-golden "
            "$(git rev-parse HEAD)"
        )
    return argv[index]


def _write_golden(capture_sha):
    if not capture_sha or capture_sha == "unknown":
        raise ValueError(
            f"capture SHA must be an explicit commit SHA, got {capture_sha!r}"
        )
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    _transformer.flash_attn_func = None          # same neutralisation as the fixture
    record = {
        "_meta": {
            "experiment": "exp_15",
            "purpose": "golden disabled-path regression (plan §3.3-3): the whole "
                       "training_step with training.yaw_aug ABSENT",
            "capture_commit": capture_sha,
            "master_seed": GOLDEN_SEED,
            "global_step": 7,
            "global_rank": 0,
            "flash_attn_disabled": True,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "record": _capture_disabled_path(),
    }
    GOLDEN_PATH.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN_PATH}")


# --------------------------------------------------------------------------- #
# §6.5-5 golden disabled-path regression
# --------------------------------------------------------------------------- #
def test_disabled_path_matches_golden(no_flash):
    golden = json.loads(GOLDEN_PATH.read_text())["record"]
    got = _capture_disabled_path()

    assert got["metadata_into_conditioner"] == golden["metadata_into_conditioner"], (
        "metadata reaching the conditioner changed on the yaw_aug-absent path"
    )
    assert got["conditioning"] == golden["conditioning"], (
        "conditioning handed to the diffusion model changed on the yaw_aug-absent path"
    )
    assert got["loss_hex"] == golden["loss_hex"], (
        f"training loss changed on the yaw_aug-absent path: "
        f"{got['loss_repr']} != {golden['loss_repr']}"
    )
    assert got["rng_before"] == golden["rng_before"], (
        "global RNG state entering training_step changed (construction consumed "
        "different randomness)"
    )
    assert got["rng_after"] == golden["rng_after"], (
        "global RNG state leaving training_step changed (the step consumed "
        "different randomness)"
    )


# --------------------------------------------------------------------------- #
# §6.5-1 counter-based step seed: determinism, decorrelation, resume-exactness
# --------------------------------------------------------------------------- #
def _offsets(seed, step, rank, n=8, img_w=512):
    """Draw one micro-batch of offsets exactly as ``training_step`` will."""
    gen = torch.Generator()
    gen.manual_seed(tdiff._yaw_aug_step_seed(seed, step, rank))
    return yr.draw_yaw_offsets(n, img_w, gen)


def test_step_seed_is_deterministic():
    assert tdiff._yaw_aug_step_seed(42, 137, 3) == tdiff._yaw_aug_step_seed(42, 137, 3)
    assert torch.equal(_offsets(42, 137, 3), _offsets(42, 137, 3))


def test_step_seed_is_a_valid_32_bit_generator_seed():
    """32 bits, because that is all the pinned torch CPU generator keeps.

    See ``test_torch_cpu_generator_ignores_high_seed_bits``: returning a wider
    value would only *look* like more entropy.
    """
    for seed, step, rank in [(0, 0, 0), (42, 0, 0), (42, 39999, 7), (2**31, 10**6, 63)]:
        value = tdiff._yaw_aug_step_seed(seed, step, rank)
        assert isinstance(value, int)
        assert 0 <= value < 2**32


def test_torch_cpu_generator_ignores_high_seed_bits():
    """Pinned-environment documentation test (round-1 review, finding 1).

    torch 2.7.0's CPU MT19937 seeds from the low 32 bits only, so ``s`` and
    ``s + 2**32`` are the same stream. This is *why* _yaw_aug_step_seed derives a
    keyed 32-bit bijection instead of a well-avalanched 63-bit hash: only an
    injection into the 32-bit space can promise distinct streams.
    """
    a = torch.randint(0, 512, (32,), generator=torch.Generator().manual_seed(12345))
    b = torch.randint(0, 512, (32,), generator=torch.Generator().manual_seed(12345 + 2**32))
    assert torch.equal(a, b), (
        "the pinned torch CPU generator no longer aliases seeds modulo 2**32; the "
        "assumption behind the 32-bit bijection in _yaw_aug_step_seed has changed "
        "and the derivation should be revisited (a wider seed would now be usable)"
    )


def test_effective_seed_domain_is_collision_free():
    """The whole armed domain — 40,000 steps x 8 ranks — maps to distinct seeds.

    The predecessor of this function collided 10 times here (review finding 1),
    e.g. (step=526, rank=2) and (step=10156, rank=7) drew identical yaw streams.
    Distinctness is now structural, and this test checks the actual domain rather
    than sampling it.
    """
    steps, ranks = 40000, 8
    effective = {
        tdiff._yaw_aug_step_seed(42, step, rank)
        for step in range(steps)
        for rank in range(ranks)
    }
    assert len(effective) == steps * ranks
    assert max(effective) < 2**32 and min(effective) >= 0


def test_reviewer_collision_pair_now_draws_different_streams():
    """Regression on the concrete pair the reviewer exhibited."""
    assert tdiff._yaw_aug_step_seed(42, 526, 2) != tdiff._yaw_aug_step_seed(42, 10156, 7)
    assert not torch.equal(_offsets(42, 526, 2, n=64), _offsets(42, 10156, 7, n=64))


@pytest.mark.parametrize("step,rank", [(2**20, 0), (2**20 + 1, 0), (0, 2**12), (0, 2**12 + 5)])
def test_step_seed_rejects_out_of_domain_counters(step, rank):
    """The bijection is only injective inside its declared domain, so leaving it
    is a hard error rather than a silent wrap into another cell's stream."""
    with pytest.raises(ValueError):
        tdiff._yaw_aug_step_seed(42, step, rank)


def test_step_seed_distinct_across_steps():
    seeds = [tdiff._yaw_aug_step_seed(42, step, 0) for step in range(4096)]
    assert len(set(seeds)) == len(seeds), "step seeds collided within a single run"


def test_step_seed_distinct_across_ranks():
    seeds = [tdiff._yaw_aug_step_seed(42, 137, rank) for rank in range(64)]
    assert len(set(seeds)) == len(seeds), "step seeds collided across ranks"


def test_step_seed_distinct_across_run_seeds():
    seeds = [tdiff._yaw_aug_step_seed(run_seed, 137, 0) for run_seed in range(1024)]
    assert len(set(seeds)) == len(seeds), "step seeds collided across run seeds"


def test_offsets_decorrelate_across_steps():
    """Neighbouring steps must not share (or merely shift) their draws."""
    a = torch.cat([_offsets(42, step, 0, n=64) for step in range(0, 32)])
    b = torch.cat([_offsets(42, step, 0, n=64) for step in range(1, 33)])
    assert not torch.equal(a, b)
    # agreement should sit near chance (1/512), certainly nowhere near lockstep
    assert (a == b).float().mean().item() < 0.05


def test_offsets_decorrelate_across_ranks():
    """The 8 ranks of one step must see 8 independent yaw assignments."""
    per_rank = [_offsets(42, 137, rank, n=64) for rank in range(8)]
    for i in range(len(per_rank)):
        for j in range(i + 1, len(per_rank)):
            assert not torch.equal(per_rank[i], per_rank[j])
            assert (per_rank[i] == per_rank[j]).float().mean().item() < 0.05


def test_draws_are_resume_exact():
    """The property the counter-based scheme exists for (plan §3.1, review F5).

    A run killed at step N and resumed there must make exactly the draws the
    uninterrupted run would have made: there is no stream state to checkpoint,
    only the (seed, step, rank) counter, which the resumed process rebuilds from
    the checkpoint's ``global_step``.
    """
    n, img_w, N, M = 8, 512, 25, 12

    uninterrupted = [
        _offsets(42, step, 0, n=n, img_w=img_w) for step in range(N + M)
    ]
    # leg 1: steps 0..N-1, then the process dies (its generators die with it)
    leg1 = [_offsets(42, step, 0, n=n, img_w=img_w) for step in range(N)]
    # leg 2: a fresh process resumes at N with nothing but the counter
    leg2 = [_offsets(42, step, 0, n=n, img_w=img_w) for step in range(N, N + M)]

    assert all(torch.equal(x, y) for x, y in zip(uninterrupted, leg1 + leg2))


def test_step_seed_rejects_negative_counters():
    with pytest.raises(ValueError):
        tdiff._yaw_aug_step_seed(42, -1, 0)
    with pytest.raises(ValueError):
        tdiff._yaw_aug_step_seed(42, 0, -1)


# --------------------------------------------------------------------------- #
# §6.5-4 factory parsing of training.yaw_aug (+ its fail-closed guards)
# --------------------------------------------------------------------------- #
# The wrapper kwargs the factory passed BEFORE exp_15 existed. Plan §3.3-4: with
# ``training.yaw_aug`` absent the construction call must be *literally* the
# pre-change call — the control arm was trained through that call and the
# comparison rests on it being untouched.
PRE_CHANGE_WRAPPER_KWARGS = {
    "lr", "mask_padding", "mask_padding_dropout", "use_ema", "log_loss_info",
    "optimizer_configs", "pre_encoded", "cfg_dropout_prob", "timestep_sampler",
    "timestep_sampler_options", "p_one_shot", "test_param", "cond_method",
    "frame_avg_angles",
}

_SENTINEL_MODEL = object()


def _capture_wrapper_kwargs(monkeypatch, training_overrides):
    """Build through the real factory with the wrapper class stubbed out.

    ``create_training_wrapper_from_config`` imports the wrapper inside the
    function body, so patching the module attribute intercepts construction; the
    model is never touched on the diffusion_cond branch, hence the sentinel.
    """
    captured = {}

    class _Stub:
        def __init__(self, model, **kwargs):
            captured["model"] = model
            captured["kwargs"] = kwargs

    monkeypatch.setattr(tdiff, "DiffusionCondTrainingWrapper", _Stub)
    cfg = _base_config()
    cfg["training"].update(training_overrides)
    create_training_wrapper_from_config(cfg, _SENTINEL_MODEL)
    return captured


def _expect_factory_error(training_overrides, *needles):
    cfg = _base_config()
    cfg["training"].update(training_overrides)
    with pytest.raises(ValueError) as excinfo:
        create_training_wrapper_from_config(cfg, _SENTINEL_MODEL)
    message = str(excinfo.value)
    for needle in needles:
        assert needle in message, f"error message {message!r} does not name {needle!r}"


def test_absent_yaw_aug_block_passes_no_new_kwargs(monkeypatch):
    captured = _capture_wrapper_kwargs(monkeypatch, {})
    assert captured["model"] is _SENTINEL_MODEL
    assert set(captured["kwargs"]) == PRE_CHANGE_WRAPPER_KWARGS


def test_disabled_yaw_aug_block_passes_no_new_kwargs(monkeypatch):
    """``enabled: false`` is still the pre-change call — the default is False."""
    captured = _capture_wrapper_kwargs(monkeypatch, {"yaw_aug": {"enabled": False}})
    assert set(captured["kwargs"]) == PRE_CHANGE_WRAPPER_KWARGS


def test_enabled_yaw_aug_block_passes_flags(monkeypatch):
    captured = _capture_wrapper_kwargs(
        monkeypatch, {"yaw_aug": {"enabled": True, "img_w": 512, "seed": 42}}
    )
    kwargs = captured["kwargs"]
    assert set(kwargs) == PRE_CHANGE_WRAPPER_KWARGS | {
        "yaw_aug_enabled", "yaw_aug_img_w", "yaw_aug_seed"
    }
    assert kwargs["yaw_aug_enabled"] is True
    assert kwargs["yaw_aug_img_w"] == 512
    assert kwargs["yaw_aug_seed"] == 42


def test_yaw_aug_rejects_fa_invariant():
    """Augmentation on top of frame averaging is an untested combination."""
    _expect_factory_error(
        {"cond_method": "fa_invariant", "yaw_aug": {"enabled": True, "img_w": 512, "seed": 42}},
        "yaw_aug", "fa_invariant",
    )


@pytest.mark.parametrize("missing", ["seed", "img_w"])
def test_yaw_aug_enabled_requires_seed_and_img_w(missing):
    block = {"enabled": True, "img_w": 512, "seed": 42}
    block.pop(missing)
    _expect_factory_error({"yaw_aug": block}, "yaw_aug", missing)


def test_yaw_aug_rejects_unknown_keys():
    _expect_factory_error(
        {"yaw_aug": {"enabled": True, "img_w": 512, "seed": 42, "img_h": 256}},
        "yaw_aug", "img_h",
    )


@pytest.mark.parametrize("enabled", [1, 0, "true", "false", None, 1.0])
def test_yaw_aug_rejects_non_literal_bool_enabled(enabled):
    """A truthy 1 in the JSON must not silently switch the treatment on/off."""
    _expect_factory_error(
        {"yaw_aug": {"enabled": enabled, "img_w": 512, "seed": 42}},
        "yaw_aug", "enabled",
    )


@pytest.mark.parametrize("block", ["enabled", True, ["enabled"], 512])
def test_yaw_aug_rejects_non_dict_block(block):
    _expect_factory_error({"yaw_aug": block}, "yaw_aug")


@pytest.mark.parametrize("img_w", [0, -512, 512.0, "512", True])
def test_yaw_aug_rejects_bad_img_w(img_w):
    _expect_factory_error(
        {"yaw_aug": {"enabled": True, "img_w": img_w, "seed": 42}}, "yaw_aug", "img_w"
    )


@pytest.mark.parametrize("seed", [42.0, "42", True, None])
def test_yaw_aug_rejects_bad_seed(seed):
    _expect_factory_error(
        {"yaw_aug": {"enabled": True, "img_w": 512, "seed": seed}}, "yaw_aug", "seed"
    )


# --------------------------------------------------------------------------- #
# §6.5-2,3,6,7,8 the training_step hook
# --------------------------------------------------------------------------- #
def _enabled_wrapper(img_w=16, seed=42, step=0, rank=0, **overrides):
    wrapper = _build_wrapper(
        yaw_aug={"enabled": True, "img_w": img_w, "seed": seed}, **overrides
    )
    _attach_stub_trainer(wrapper, global_step=step, global_rank=rank)
    return wrapper


def _spy_conditioning(wrapper):
    """Record the metadata list each ``_compute_conditioning`` call receives."""
    seen = []
    original = wrapper._compute_conditioning

    def _spy(metadata):
        seen.append(metadata)
        return original(metadata)

    wrapper._compute_conditioning = _spy
    return seen


def _expected_offsets(seed, step, rank, n, img_w):
    gen = torch.Generator()
    gen.manual_seed(tdiff._yaw_aug_step_seed(seed, step, rank))
    return yr.draw_yaw_offsets(n, img_w, gen)


def _manual_rotate(md, d, img_w):
    """A reference rotation computed from first principles, NOT via
    ``rotate_scene_metadata`` — otherwise the test would only prove the code
    agrees with itself."""
    alpha = d * 2.0 * np.pi / img_w
    c, s = float(np.cos(alpha)), float(np.sin(alpha))
    rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    out = dict(md)
    rolled = torch.roll(md["depth"], shifts=d, dims=2)
    out["depth"] = torch.einsum("ij,jhw->ihw", rot, rolled)
    for key in ("source", "source_vit", "context_poses", "context_poses_vit"):
        if key in md:
            out[key] = torch.einsum("ij,...j->...i", rot, md[key])
    return out


# --- flags reach the wrapper ------------------------------------------------ #
def test_wrapper_defaults_have_yaw_aug_off():
    wrapper = _build_wrapper()
    assert wrapper.yaw_aug_enabled is False


def test_factory_flags_reach_the_wrapper():
    wrapper = _build_wrapper(yaw_aug={"enabled": True, "img_w": 512, "seed": 42})
    assert wrapper.yaw_aug_enabled is True
    assert wrapper.yaw_aug_img_w == 512
    assert wrapper.yaw_aug_seed == 42


def test_banner_printed_once_at_fit_start(capsys):
    wrapper = _enabled_wrapper(img_w=512, seed=42)
    wrapper.on_fit_start()
    assert capsys.readouterr().out.count("yaw_aug ENABLED img_w=512 seed=42") == 1


def test_banner_is_flushed(monkeypatch):
    """The launcher tees torchrun's stdout through a FIFO without
    PYTHONUNBUFFERED, so a buffered banner can reach the log after the launch
    gate has already given up (review finding 3)."""
    calls = []
    monkeypatch.setattr("builtins.print", lambda *a, **kw: calls.append((a, kw)))
    _enabled_wrapper(img_w=512, seed=42).on_fit_start()
    assert len(calls) == 1
    assert calls[0][1].get("flush") is True


def test_no_banner_when_disabled(capsys):
    wrapper = _build_wrapper()
    _attach_stub_trainer(wrapper)
    wrapper.on_fit_start()
    assert "yaw_aug" not in capsys.readouterr().out


# --- constructor guards reject rather than coerce (review finding 4) -------- #
def _construct_directly(**yaw_aug_kwargs):
    """Bypass the factory entirely: the constructor must fail closed on its own.

    The yaw_aug validation runs before the constructor touches the model, so a
    bare namespace stands in for it.
    """
    return tdiff.DiffusionCondTrainingWrapper(
        types.SimpleNamespace(), lr=5e-6, use_ema=False, **yaw_aug_kwargs
    )


@pytest.mark.parametrize("enabled", ["false", "true", 1, 0, 1.0, None])
def test_constructor_rejects_non_literal_bool_enabled(enabled):
    with pytest.raises(ValueError, match="yaw_aug_enabled"):
        _construct_directly(yaw_aug_enabled=enabled)


@pytest.mark.parametrize("img_w", ["512", 512.0, True, None, 0, -1])
def test_constructor_rejects_bad_img_w(img_w):
    with pytest.raises(ValueError, match="yaw_aug_img_w"):
        _construct_directly(yaw_aug_enabled=True, yaw_aug_img_w=img_w, yaw_aug_seed=42)


@pytest.mark.parametrize("seed", ["42", 42.0, True, None])
def test_constructor_rejects_bad_seed(seed):
    with pytest.raises(ValueError, match="yaw_aug_seed"):
        _construct_directly(yaw_aug_enabled=True, yaw_aug_img_w=512, yaw_aug_seed=seed)


def test_constructor_does_not_coerce_valid_values():
    wrapper = _build_wrapper(yaw_aug={"enabled": True, "img_w": 512, "seed": 42})
    assert wrapper.yaw_aug_enabled is True
    assert isinstance(wrapper.yaw_aug_img_w, int) and wrapper.yaw_aug_img_w == 512
    assert isinstance(wrapper.yaw_aug_seed, int) and wrapper.yaw_aug_seed == 42


# --- golden regeneration demands an explicit capture SHA (finding 5) -------- #
def test_write_golden_requires_an_explicit_sha():
    with pytest.raises(SystemExit, match="SHA"):
        _capture_sha_from_argv(["test_yaw_aug_training.py", "--write-golden"])
    with pytest.raises(SystemExit, match="SHA"):
        _capture_sha_from_argv(["test_yaw_aug_training.py", "--write-golden", "--force"])
    with pytest.raises(SystemExit):
        _capture_sha_from_argv(["test_yaw_aug_training.py"])


def test_write_golden_accepts_an_explicit_sha():
    argv = ["test_yaw_aug_training.py", "--write-golden", "d3a0312"]
    assert _capture_sha_from_argv(argv) == "d3a0312"


def test_write_golden_rejects_a_placeholder_sha():
    """Rejection must happen BEFORE anything is written.

    While this guard was still red, this very test rewrote the committed fixture
    with ``capture_commit: "unknown"`` (the record digests re-captured
    identically, so only the provenance stamp was lost — it was restored from
    git). The byte check below makes a future regression of the guard fail
    loudly instead of quietly overwriting the round's reference.
    """
    before = GOLDEN_PATH.read_bytes()
    with pytest.raises(ValueError, match="SHA"):
        _write_golden("unknown")
    assert GOLDEN_PATH.read_bytes() == before, "the golden fixture was overwritten"


# --- §6.5-3 exactness: the drawn offset is the applied offset --------------- #
@pytest.mark.parametrize("img_w", [16, 512])
def test_every_drawn_angle_requantises_to_its_offset(img_w):
    offsets = _expected_offsets(42, 0, 0, n=256, img_w=img_w)
    for d in offsets.tolist() + list(range(img_w)):
        alpha = yr.offsets_to_radians([d], img_w)[0]
        assert yr.yaw_column_shift(alpha, img_w) == d


# --- §6.5-2 global-RNG isolation ------------------------------------------- #
def test_augmentation_does_not_touch_global_rng():
    wrapper = _enabled_wrapper(img_w=512, step=11, rank=0)
    _, metadata = _batch(4, img_w=512)
    random.seed(3); np.random.seed(3); torch.manual_seed(3)
    before = _rng_digest()
    wrapper._apply_yaw_aug(metadata)
    assert _rng_digest() == before, (
        "the yaw draw/application consumed global randomness; the augmented arm "
        "would then differ from the control by RNG displacement as well as by "
        "the treatment"
    )


def test_enabled_step_leaves_the_same_global_rng_state_as_disabled(no_flash):
    """Black-box form of the same property, through a whole training_step."""
    digests = []
    for training in ({}, {"yaw_aug": {"enabled": True, "img_w": 16, "seed": 42}}):
        random.seed(GOLDEN_SEED); np.random.seed(GOLDEN_SEED); torch.manual_seed(GOLDEN_SEED)
        wrapper = _build_wrapper(**training)
        _attach_stub_trainer(wrapper, global_step=5)
        reals, metadata = _batch(2)
        wrapper.training_step((reals, metadata), 0)
        digests.append(_rng_digest())
    assert digests[0] == digests[1]


# --- §6.5-6 application: exactly the drawn per-sample rotation -------------- #
def test_training_step_applies_the_drawn_rotation(no_flash):
    img_w, step, rank, seed = 512, 13, 2, 42
    wrapper = _enabled_wrapper(img_w=img_w, seed=seed, step=step, rank=rank)
    seen = _spy_conditioning(wrapper)
    reals, metadata = _batch(3, img_w=img_w)
    pristine = [{k: (v.clone() if torch.is_tensor(v) else v) for k, v in md.items()}
                for md in metadata]

    wrapper.training_step((reals, metadata), 0)

    offsets = _expected_offsets(seed, step, rank, n=3, img_w=img_w).tolist()
    assert len(seen) == 1
    for md_in, md_out, d in zip(pristine, seen[0], offsets):
        expected = _manual_rotate(md_in, d, img_w)
        for key in ("depth", "source", "source_vit", "context_poses", "context_poses_vit"):
            assert torch.equal(md_out[key], expected[key]), f"{key} not rotated by d={d}"
            assert md_out[key].dtype == md_in[key].dtype

    # untouched fields pass through bit-identically
    for md_in, md_out in zip(pristine, seen[0]):
        assert torch.equal(md_out["context_audio"], md_in["context_audio"])
        assert torch.equal(md_out["padding_mask"], md_in["padding_mask"])
        assert md_out["scene"] == md_in["scene"]

    # the caller's batch is never mutated in place
    for md_in, md_now in zip(pristine, metadata):
        for key, value in md_in.items():
            if torch.is_tensor(value):
                assert torch.equal(md_now[key], value), f"input metadata {key} mutated"


def test_training_step_leaves_reals_untouched(no_flash):
    """A rigid yaw of the scene leaves the RIR unchanged — that is the whole
    reason the augmented pair is a valid training pair."""
    wrapper = _enabled_wrapper(img_w=512)
    reals, metadata = _batch(2, img_w=512)
    reference = reals.clone()
    wrapper.training_step((reals, metadata), 0)
    assert torch.equal(reals, reference)


def test_draws_advance_with_global_step(no_flash):
    """Different steps must see different yaws; the same step reproduces."""
    def seen_depths(step):
        wrapper = _enabled_wrapper(img_w=512, step=step)
        seen = _spy_conditioning(wrapper)
        reals, metadata = _batch(2, img_w=512)
        wrapper.training_step((reals, metadata), 0)
        return [md["depth"].clone() for md in seen[0]]

    a, b, a_again = seen_depths(0), seen_depths(1), seen_depths(0)
    assert all(torch.equal(x, y) for x, y in zip(a, a_again))
    assert not all(torch.equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("d", [0, 1, 128, 511])
def test_fixed_offset_integration_cases(monkeypatch, no_flash, d):
    """Pin the geometry itself at four offsets, including the identity (d=0) and
    the wrap-around neighbour (d=511)."""
    img_w = 512
    monkeypatch.setattr(
        tdiff, "draw_yaw_offsets",
        lambda n, w, generator: torch.full((n,), d, dtype=torch.long),
    )
    wrapper = _enabled_wrapper(img_w=img_w)
    seen = _spy_conditioning(wrapper)
    reals, metadata = _batch(2, img_w=img_w)
    pristine = [{k: (v.clone() if torch.is_tensor(v) else v) for k, v in md.items()}
                for md in metadata]

    wrapper.training_step((reals, metadata), 0)

    for md_in, md_out in zip(pristine, seen[0]):
        expected = _manual_rotate(md_in, d, img_w)
        assert torch.allclose(md_out["depth"], expected["depth"], atol=1e-6)
        assert md_out["depth"].dtype == md_in["depth"].dtype
        assert md_out["depth"].device == md_in["depth"].device
        for key in yr.POSE_KEYS:
            assert torch.allclose(md_out[key], expected[key], atol=1e-6), key
            assert md_out[key].dtype == md_in[key].dtype, key
            assert md_out[key].device == md_in[key].device, key
        if d == 0:
            assert torch.equal(md_out["depth"], md_in["depth"])
            for key in ("source", "source_vit", "context_poses", "context_poses_vit"):
                assert torch.allclose(md_out[key], md_in[key], atol=1e-6)


# --- §6.5-7 fail-closed schema guards -------------------------------------- #
def _run_enabled_step(metadata, img_w=512):
    wrapper = _enabled_wrapper(img_w=img_w)
    n = len(metadata) if isinstance(metadata, (list, tuple)) else 1
    reals = torch.zeros(max(n, 1), 4, 64)
    wrapper.training_step((reals, metadata), 0)


def test_guard_empty_metadata():
    with pytest.raises(ValueError, match="yaw_aug"):
        _run_enabled_step([])


@pytest.mark.parametrize("bad", [None, {}, "metadata"])
def test_guard_metadata_not_a_list(bad):
    with pytest.raises(ValueError, match="yaw_aug"):
        _run_enabled_step(bad)


def test_guard_sample_not_a_dict():
    md = _make_md(0, img_w=512)
    with pytest.raises(ValueError, match="yaw_aug"):
        _run_enabled_step([md, "not-a-dict"])


def test_guard_missing_depth():
    md = _make_md(0, img_w=512)
    del md["depth"]
    with pytest.raises(ValueError, match="depth"):
        _run_enabled_step([md])


def test_guard_wrong_depth_width():
    """The roll width is validated against the ACTUAL tensor, never trusted from
    the config (plan §3.1, review F7)."""
    md = _make_md(0, img_w=256)
    with pytest.raises(ValueError, match="256"):
        _run_enabled_step([md], img_w=512)


@pytest.mark.parametrize("shape", [(3, 512), (1, 4, 512), (3, 4, 512, 1)])
def test_guard_wrong_depth_shape(shape):
    md = _make_md(0, img_w=512)
    md["depth"] = torch.zeros(*shape)
    with pytest.raises(ValueError, match="depth"):
        _run_enabled_step([md])


@pytest.mark.parametrize("key", yr.POSE_KEYS)
def test_guard_wrong_pose_trailing_dim(key):
    md = _make_md(0, img_w=512)
    md[key] = torch.zeros(*md[key].shape[:-1], 2)
    with pytest.raises(ValueError, match=key):
        _run_enabled_step([md])


@pytest.mark.parametrize("key", yr.POSE_KEYS)
def test_guard_missing_pose_field(key):
    """All four pose fields are REQUIRED (review finding 2).

    ``rotate_scene_metadata`` skips absent keys, so a sample missing one would be
    rotated only partially — depth and three poses moved, the fourth left behind
    — which is geometric nonsense the model would silently train on. Detecting
    exactly this schema drift is what the guard is for.
    """
    md = _make_md(0, img_w=512)
    del md[key]
    with pytest.raises(ValueError, match=key):
        _run_enabled_step([md])


@pytest.mark.parametrize("key", yr.POSE_KEYS)
def test_guard_scalar_pose(key):
    """A 0-d pose must raise ValueError, not IndexError from ``shape[-1]``."""
    md = _make_md(0, img_w=512)
    md[key] = torch.tensor(1.0)
    with pytest.raises(ValueError, match=key):
        _run_enabled_step([md])


@pytest.mark.parametrize("key", yr.POSE_KEYS)
def test_guard_non_tensor_pose(key):
    md = _make_md(0, img_w=512)
    md[key] = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match=key):
        _run_enabled_step([md])


def test_guard_depth_not_a_tensor():
    md = _make_md(0, img_w=512)
    md["depth"] = [[0.0] * 512]
    with pytest.raises(ValueError, match="depth"):
        _run_enabled_step([md])


# --- §6.5-8 validation never augments -------------------------------------- #
def test_validation_step_never_augments(no_flash):
    wrapper = _enabled_wrapper(img_w=512, step=13, rank=2)
    seen = _spy_conditioning(wrapper)
    reals, metadata = _batch(2, img_w=512)
    pristine = [{k: (v.clone() if torch.is_tensor(v) else v) for k, v in md.items()}
                for md in metadata]

    wrapper.validation_step((reals, metadata), 0)

    assert len(seen) == 1
    for md_in, md_out in zip(pristine, seen[0]):
        for key in ("depth", "source", "source_vit", "context_poses", "context_poses_vit"):
            assert torch.equal(md_out[key], md_in[key]), (
                f"validation_step rotated {key}: the val loss would no longer be "
                "comparable to any earlier run"
            )


if __name__ == "__main__":
    _write_golden(_capture_sha_from_argv(sys.argv))
