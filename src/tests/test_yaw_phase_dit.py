"""Round-4/5 contracts for yaw-phase DiT, CFG, and generation plumbing.

The unit tests use a tiny DiT or replace ``_forward`` with a recorder so a
failure identifies batch/dtype plumbing rather than an attention kernel.  The
last test is a CPU-cheap end-to-end control: it starts from complete raw scene
metadata, applies every C16 rotation, constructs equivariant geometry tokens,
and runs a real non-zero cross-attention branch.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.data import yaw_rotation as yr
from src.inference import generation as generation_mod
from src.models.diffusion import DiTWrapper
from src.models.dit import DiffusionTransformer


BATCH = 2
IO_CHANNELS = 4
DIM = 32
TOKENS = 34
NUM_FREQS = 8


def _tiny_dit(
    *,
    depth: int = 2,
    dtype: torch.dtype = torch.float32,
    num_freqs: int = NUM_FREQS,
):
    torch.manual_seed(20260718)
    model = DiffusionTransformer(
        io_channels=IO_CHANNELS,
        patch_size=1,
        embed_dim=DIM,
        cond_token_dim=DIM,
        project_cond_tokens=False,
        global_cond_dim=DIM,
        project_global_cond=False,
        depth=depth,
        # V0 rotates 2*M=16 dimensions, so the tiny head must still satisfy
        # the production head-dim contract (32 >= 16).
        num_heads=1,
        rotary_pos_emb=False,
        azimuth_num_freqs=num_freqs,
        diffusion_objective="rectified_flow",
    )
    return model.to(dtype)


def _tiny_dit_wrapper():
    return DiTWrapper(
        diffusion_objective="rectified_flow",
        io_channels=IO_CHANNELS,
        patch_size=1,
        embed_dim=DIM,
        cond_token_dim=DIM,
        project_cond_tokens=False,
        global_cond_dim=DIM,
        project_global_cond=False,
        depth=1,
        num_heads=1,
        rotary_pos_emb=False,
        azimuth_num_freqs=NUM_FREQS,
    )


def _dit_inputs(*, batch: int = BATCH, phase_dtype=torch.float32):
    generator = torch.Generator().manual_seed(913)
    return {
        "x": torch.randn(batch, IO_CHANNELS, 5, generator=generator),
        "t": torch.linspace(0.2, 0.7, batch),
        "cross_attn_cond": torch.randn(
            batch, TOKENS, DIM, generator=generator
        ),
        "cross_attn_cond_mask": torch.tensor(
            [[True] * TOKENS] * batch
        ),
        "cross_attn_phases": torch.linspace(-math.pi, math.pi, TOKENS)
        .view(1, TOKENS)
        .expand(batch, -1)
        .to(phase_dtype),
        "query_phase": torch.linspace(-0.6, 0.9, batch).to(phase_dtype),
        "global_embed": torch.randn(batch, DIM, generator=generator),
    }


def _expected_angles(cross_phases, query_phase, num_freqs=NUM_FREQS):
    relative = cross_phases.float() - query_phase.float()[:, None]
    frequencies = torch.arange(
        1, num_freqs + 1, device=relative.device, dtype=torch.float32
    )
    angles = relative[..., None] * frequencies
    return torch.cat([angles, angles], dim=-1)


@pytest.mark.parametrize("use_checkpointing", [False, True])
def test_relative_phases_are_float32_and_reused_by_every_block(
    monkeypatch, use_checkpointing
):
    """DiT must form relative angles once and pass the same tensor to all layers."""

    model = _tiny_dit(depth=3)
    inputs = _dit_inputs(phase_dtype=torch.float64)
    seen = []

    for layer_index, layer in enumerate(model.transformer.layers):

        def record_layer(
            x,
            *,
            context=None,
            cross_rope_phases=None,
            _index=layer_index,
            **kwargs,
        ):
            seen.append((_index, cross_rope_phases, context))
            return x

        monkeypatch.setattr(layer, "forward", record_layer)

    output = model(**inputs, cfg_scale=1.0, use_checkpointing=use_checkpointing)

    assert output.shape == inputs["x"].shape
    assert [index for index, _, _ in seen] == [0, 1, 2]
    expected = _expected_angles(
        inputs["cross_attn_phases"], inputs["query_phase"]
    )
    for _, angles, context in seen:
        assert angles is not None
        assert angles.dtype is torch.float32
        assert angles.shape == (BATCH, TOKENS, 2 * NUM_FREQS)
        torch.testing.assert_close(angles, expected)
        assert context.shape == (BATCH, TOKENS, DIM)
    # Recomputing the mathematically same angles in every block is avoidable and
    # makes it easier for block-local casting to diverge.  Lock object reuse.
    assert all(angles is seen[0][1] for _, angles, _ in seen)


def test_m_zero_phase_aware_control_runs_without_applying_cross_rope(monkeypatch):
    model = _tiny_dit(depth=2, num_freqs=0)
    inputs = _dit_inputs()
    seen = []

    for layer in model.transformer.layers:

        def record_layer(x, *, cross_rope_phases="missing", **kwargs):
            seen.append(cross_rope_phases)
            return x

        monkeypatch.setattr(layer, "forward", record_layer)

    output = model(**inputs, cfg_scale=1.0, use_checkpointing=False)

    assert output.shape == inputs["x"].shape
    assert seen == [None, None]


def test_azimuth_rope_rejects_more_dimensions_than_one_attention_head():
    with pytest.raises(ValueError, match="dimensions per head|34|32"):
        _tiny_dit(depth=1, num_freqs=17)


@pytest.mark.parametrize("cfg_scale", [1.0, 2.5])
def test_cfg_batch_alignment_covers_content_mask_global_and_phases(
    monkeypatch, cfg_scale
):
    model = _tiny_dit(depth=1)
    inputs = _dit_inputs()
    captured = {}

    def record_forward(x, t, **kwargs):
        captured["x"] = x
        captured["t"] = t
        captured.update(kwargs)
        return x

    monkeypatch.setattr(model, "_forward", record_forward)
    output = model(**inputs, cfg_scale=cfg_scale)

    assert output.shape == inputs["x"].shape
    if cfg_scale == 1.0:
        assert captured["x"].shape[0] == BATCH
        torch.testing.assert_close(
            captured["cross_attn_cond"], inputs["cross_attn_cond"]
        )
        assert torch.equal(
            captured["cross_attn_cond_mask"],
            inputs["cross_attn_cond_mask"],
        )
        assert torch.equal(
            captured["cross_attn_phases"], inputs["cross_attn_phases"]
        )
        assert torch.equal(captured["query_phase"], inputs["query_phase"])
        torch.testing.assert_close(
            captured["global_embed"], inputs["global_embed"]
        )
        return

    assert captured["x"].shape[0] == 2 * BATCH
    torch.testing.assert_close(
        captured["cross_attn_cond"][:BATCH], inputs["cross_attn_cond"]
    )
    assert torch.count_nonzero(captured["cross_attn_cond"][BATCH:]) == 0
    assert torch.equal(
        captured["cross_attn_cond_mask"],
        torch.cat(
            [
                inputs["cross_attn_cond_mask"],
                inputs["cross_attn_cond_mask"],
            ]
        ),
    )
    assert torch.equal(
        captured["cross_attn_phases"],
        torch.cat(
            [inputs["cross_attn_phases"], inputs["cross_attn_phases"]]
        ),
    )
    assert torch.equal(
        captured["query_phase"],
        torch.cat([inputs["query_phase"], inputs["query_phase"]]),
    )
    torch.testing.assert_close(
        captured["global_embed"],
        torch.cat([inputs["global_embed"], inputs["global_embed"]]),
    )


def test_cfg_dropout_zeros_only_content_and_retains_float32_phases(monkeypatch):
    model = _tiny_dit(depth=1)
    inputs = _dit_inputs(phase_dtype=torch.float64)
    captured = {}

    def record_forward(x, t, **kwargs):
        captured.update(kwargs)
        return x

    monkeypatch.setattr(model, "_forward", record_forward)
    model(**inputs, cfg_scale=1.0, cfg_dropout_prob=1.0)

    assert torch.count_nonzero(captured["cross_attn_cond"]) == 0
    assert captured["cross_attn_phases"].dtype is torch.float32
    assert captured["query_phase"].dtype is torch.float32
    torch.testing.assert_close(
        captured["cross_attn_phases"], inputs["cross_attn_phases"].float()
    )
    torch.testing.assert_close(
        captured["query_phase"], inputs["query_phase"].float()
    )
    # Existing FLAC semantics retain invariant global content during null CFG.
    torch.testing.assert_close(captured["global_embed"], inputs["global_embed"])


def test_bfloat16_content_never_downcasts_phase_or_query(monkeypatch):
    model = _tiny_dit(depth=1, dtype=torch.bfloat16)
    inputs = _dit_inputs(phase_dtype=torch.float64)
    captured = {}

    def record_forward(x, t, **kwargs):
        captured["x"] = x
        captured["t"] = t
        captured.update(kwargs)
        return x

    monkeypatch.setattr(model, "_forward", record_forward)
    output = model(**inputs, cfg_scale=1.0)

    assert output.dtype is torch.bfloat16
    for key in ("x", "t", "cross_attn_cond", "global_embed"):
        assert captured[key].dtype is torch.bfloat16
    assert captured["cross_attn_phases"].dtype is torch.float32
    assert captured["query_phase"].dtype is torch.float32
    assert captured["cross_attn_cond_mask"].dtype is torch.bool


@pytest.mark.parametrize(
    "negative_kwargs",
    [
        {"negative_cross_attn_cond": torch.randn(BATCH, TOKENS, DIM)},
        {"negative_cross_attn_phases": torch.zeros(BATCH, TOKENS)},
        {"negative_query_phase": torch.zeros(BATCH)},
    ],
    ids=("negative-content", "negative-token-phases", "negative-query-phase"),
)
def test_phase_aware_independent_negative_conditioning_is_rejected(
    monkeypatch, negative_kwargs
):
    model = _tiny_dit(depth=1)
    inputs = _dit_inputs()
    monkeypatch.setattr(model, "_forward", lambda x, t, **kwargs: x)

    with pytest.raises(
        (ValueError, NotImplementedError),
        match=r"negative|independent|phase-aware|V0",
    ):
        model(**inputs, **negative_kwargs, cfg_scale=2.0)


def test_phase_aware_ordinary_null_cfg_remains_supported(monkeypatch):
    model = _tiny_dit(depth=1)
    inputs = _dit_inputs()
    captured = {}

    def record_forward(x, t, **kwargs):
        captured.update(kwargs)
        return x

    monkeypatch.setattr(model, "_forward", record_forward)
    output = model(**inputs, cfg_scale=2.0)

    assert output.shape == inputs["x"].shape
    assert captured["cross_attn_cond"].shape[0] == 2 * BATCH
    assert captured["cross_attn_phases"].shape[0] == 2 * BATCH
    assert captured["query_phase"].shape[0] == 2 * BATCH


def test_phase_aware_path_rejects_a_mask_continuous_transformer_would_ignore():
    model = _tiny_dit(depth=1)
    inputs = _dit_inputs()
    inputs["cross_attn_cond_mask"][0, -1] = False

    with pytest.raises(ValueError, match="all-valid|mask|does not consume"):
        model(**inputs, cfg_scale=1.0)


@pytest.mark.parametrize(
    "negative_kwargs",
    [
        {"negative_global_cond": torch.ones(BATCH, DIM)},
        {"negative_input_concat_cond": torch.ones(BATCH, 1, 5)},
    ],
    ids=("wrapper-negative-global", "wrapper-negative-input-concat"),
)
def test_phase_aware_wrapper_rejects_negative_inputs_it_cannot_forward(
    negative_kwargs,
):
    wrapper = _tiny_dit_wrapper()
    inputs = _dit_inputs()

    with pytest.raises(ValueError, match="negative|phase-aware|V0"):
        wrapper(
            inputs["x"],
            inputs["t"],
            cross_attn_cond=inputs["cross_attn_cond"],
            cross_attn_mask=inputs["cross_attn_cond_mask"],
            cross_attn_phases=inputs["cross_attn_phases"],
            query_phase=inputs["query_phase"],
            global_cond=inputs["global_embed"],
            cfg_scale=2.0,
            **negative_kwargs,
        )


class _GenerationHarness:
    def __init__(self):
        self.pretransform = None
        self.io_channels = IO_CHANNELS
        self.model = nn.Linear(1, 1, bias=False).to(torch.bfloat16)
        self.diffusion_objective = "rectified_flow"
        self.dist_shift = None

    def get_conditioning_inputs(self, tensors, negative=False):
        assert not negative
        return dict(tensors)


def test_generation_casts_content_but_exempts_all_phase_keys(monkeypatch):
    harness = _GenerationHarness()
    phase_keys = (
        "cross_attn_phases",
        "query_phase",
        "negative_cross_attn_phases",
        "negative_query_phase",
    )
    conditioning_inputs = {
        "cross_attn_cond": torch.randn(1, TOKENS, DIM),
        "global_embed": torch.randn(1, DIM),
        "cross_attn_mask": torch.ones(1, TOKENS, dtype=torch.bool),
        "cross_attn_phases": torch.randn(1, TOKENS, dtype=torch.float32),
        "query_phase": torch.randn(1, dtype=torch.float32),
        # The negative phase names are cast-exemption contracts even though V0
        # rejects executing independent negative conditioning in the DiT.
        "negative_cross_attn_phases": torch.randn(
            1, TOKENS, dtype=torch.float32
        ),
        "negative_query_phase": torch.randn(1, dtype=torch.float32),
    }
    captured = {}

    def fake_sample_rf(model, noise, **kwargs):
        captured.update(kwargs)
        return noise

    monkeypatch.setattr(generation_mod, "sample_rf", fake_sample_rf)
    result = generation_mod.generate_diffusion_cond(
        harness,
        steps=1,
        conditioning_tensors=conditioning_inputs,
        batch_size=1,
        sample_size=8,
        device="cpu",
        return_latents=True,
    )

    assert result.dtype is torch.bfloat16
    assert captured["cross_attn_cond"].dtype is torch.bfloat16
    assert captured["global_embed"].dtype is torch.bfloat16
    for key in phase_keys:
        assert captured[key].dtype is torch.float32


def test_negative_cast_helper_preserves_legacy_content_dtype_but_forces_phase_fp32():
    inputs = {
        "negative_cross_attn_cond": torch.randn(1, 2, 3, dtype=torch.float64),
        "negative_cross_attn_phases": torch.randn(1, 2, dtype=torch.float64),
        "negative_query_phase": torch.randn(1, dtype=torch.float64),
    }

    cast = generation_mod._cast_conditioning_inputs(
        inputs, torch.bfloat16, cast_content=False
    )

    assert cast["negative_cross_attn_cond"].dtype is torch.float64
    assert cast["negative_cross_attn_phases"].dtype is torch.float32
    assert cast["negative_query_phase"].dtype is torch.float32


def _column_features(depth: torch.Tensor) -> torch.Tensor:
    """Rotation-invariant column content which shifts with panorama columns."""

    scalar = depth.square().sum(dim=0).sqrt().mean(dim=0)
    frequencies = torch.arange(1, DIM // 2 + 1, dtype=scalar.dtype)
    angles = scalar[:, None] * frequencies[None, :]
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


def _pose_features(content: torch.Tensor) -> torch.Tensor:
    frequencies = torch.arange(1, DIM // 4 + 1, dtype=content.dtype)
    angles = content[..., None] * frequencies
    return torch.cat(
        [
            angles[..., 0, :].sin(),
            angles[..., 0, :].cos(),
            angles[..., 1, :].sin(),
            angles[..., 1, :].cos(),
        ],
        dim=-1,
    )


def _raw_phase_inputs(metadata: dict) -> dict:
    """Small deterministic conditioner preserving the production V0 contract."""

    bundle = getattr(yr, "yaw_pose_content_and_phase")(metadata)
    geometry = _column_features(metadata["depth"])
    context_geometry = torch.cat(
        [geometry[:, DIM // 2 :], geometry[:, : DIM // 2]], dim=-1
    )
    pose = _pose_features(bundle["context_content"])[0]
    audio_scalar = metadata["context_audio"].float().square().mean().sqrt()
    audio = _pose_features(torch.stack([audio_scalar, audio_scalar + 0.25]))

    # Fixed stream identifiers are content, so they move with the corresponding
    # tokens but carry no azimuth action of their own.
    type_offsets = torch.zeros(4, DIM)
    type_offsets[0, 0] = 0.2
    type_offsets[1, 1] = 0.3
    type_offsets[2, 2] = 0.4
    type_offsets[3, 3] = 0.5
    content = torch.cat(
        [
            geometry + type_offsets[0],
            context_geometry + type_offsets[1],
            pose[None] + type_offsets[2],
            audio[None] + type_offsets[3],
        ],
        dim=0,
    ).unsqueeze(0)

    column_phases = (
        (torch.arange(16, dtype=torch.float32) + 0.5)
        * (2.0 * math.pi / 16)
        - math.pi
    )
    context_phase = bundle["context_phases"][0].float()
    phases = torch.cat(
        [
            column_phases,
            column_phases,
            context_phase[None],
            context_phase[None],
        ]
    ).unsqueeze(0)
    return {
        "cross_attn_cond": content,
        "cross_attn_cond_mask": torch.ones(1, TOKENS, dtype=torch.bool),
        "cross_attn_phases": phases,
        "query_phase": bundle["target_phase"].float().reshape(1),
        "global_embed": _pose_features(bundle["target_content"]).reshape(
            1, DIM
        ),
    }


def _make_sensitive_cross_attention(model: DiffusionTransformer) -> None:
    """Remove zero-init vacuity and make phase/content sensitivity auditable."""

    model.to_cond_embed = nn.Identity()
    model.to_global_embed = nn.Identity()
    with torch.no_grad():
        for layer in model.transformer.layers:
            attention = layer.cross_attn
            attention.to_q.weight.copy_(torch.eye(DIM))
            attention.to_kv.weight[:DIM].copy_(torch.eye(DIM))
            attention.to_kv.weight[DIM:].copy_(torch.eye(DIM))
            attention.to_out.weight.copy_(torch.eye(DIM))


def test_raw_metadata_full_c16_is_nonvacuous_phase_sensitive_and_invariant():
    """Sensitive C16 scaffold using a real DiT cross-attention computation."""

    torch.manual_seed(20260718)
    model = _tiny_dit(depth=1).eval()
    _make_sensitive_cross_attention(model)
    metadata = {
        "source": torch.tensor([1.2, -0.8, 0.35]),
        "source_vit": torch.tensor([1.2, -0.8, 0.35]),
        "context_poses": torch.tensor([[-0.4, 1.5, -0.2]]),
        "context_poses_vit": torch.tensor([[-0.4, 1.5, -0.2]]),
        "context_audio": torch.linspace(-0.7, 0.9, 32),
        "depth": torch.randn(3, 2, 16),
    }
    x = torch.randn(1, IO_CHANNELS, 5)
    t = torch.tensor([0.375])
    cross_norms = []
    hook = model.transformer.layers[0].cross_attn.register_forward_hook(
        lambda module, args, output: cross_norms.append(float(output.norm()))
    )

    with torch.no_grad():
        base_inputs = _raw_phase_inputs(metadata)
        base = model(x, t, **base_inputs, cfg_scale=1.0, use_checkpointing=False)
        identity = model(
            x, t, **base_inputs, cfg_scale=1.0, use_checkpointing=False
        )

        phase_perturbed = dict(base_inputs)
        phase_perturbed["cross_attn_phases"] = (
            base_inputs["cross_attn_phases"]
            + torch.linspace(-0.8, 0.6, TOKENS).unsqueeze(0)
        )
        phase_changed = model(
            x,
            t,
            **phase_perturbed,
            cfg_scale=1.0,
            use_checkpointing=False,
        )

        orbit_outputs = []
        for shift in range(16):
            moved = yr.rotate_scene_metadata(
                metadata,
                shift * 2.0 * math.pi / 16,
                img_w=16,
            )
            orbit_outputs.append(
                model(
                    x,
                    t,
                    **_raw_phase_inputs(moved),
                    cfg_scale=1.0,
                    use_checkpointing=False,
                )
            )

        moved = yr.rotate_scene_metadata(
            metadata, 2.0 * math.pi / 16, img_w=16
        )
        stale = _raw_phase_inputs(moved)
        stale["query_phase"] = base_inputs["query_phase"]
        broken = model(
            x, t, **stale, cfg_scale=1.0, use_checkpointing=False
        )
    hook.remove()

    identity_floor = float((identity - base).abs().max())
    phase_sensitivity = float((phase_changed - base).abs().max())
    c16_defect = max(float((output - base).abs().max()) for output in orbit_outputs)
    broken_defect = float((broken - base).abs().max())

    assert identity_floor == 0.0
    assert cross_norms and min(cross_norms) > 1.0e-4
    assert phase_sensitivity > 1.0e-5
    assert c16_defect <= 3.0e-5
    assert broken_defect > max(1.0e-5, 10.0 * c16_defect)
