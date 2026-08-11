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
* Regenerate the fixture deliberately (never to "fix" a red test) with
  ``python src/tests/test_yaw_aug_training.py --write-golden`` from the repo
  root, at a commit where the disabled path is known-good.
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


def _write_golden(capture_sha):
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


def test_step_seed_is_a_valid_63_bit_generator_seed():
    for seed, step, rank in [(0, 0, 0), (42, 0, 0), (42, 39999, 7), (2**31, 10**6, 63)]:
        value = tdiff._yaw_aug_step_seed(seed, step, rank)
        assert isinstance(value, int)
        assert 0 <= value < 2**63


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


if __name__ == "__main__":
    if "--write-golden" not in sys.argv:
        raise SystemExit("refusing to overwrite the golden fixture without --write-golden")
    sha = sys.argv[sys.argv.index("--write-golden") + 1] if len(sys.argv) > 2 else "unknown"
    _write_golden(sha)
