"""Round-4 RED contracts for relative-yaw rotary cross-attention.

The tests stay at the smallest useful boundaries: rotary construction and
application, one cross-attention layer, and the two
``ContinuousTransformer`` execution paths.  They intentionally overwrite
cross-attention output projections with an identity matrix so phase tests
cannot pass merely because the repository's normal initialization zeros those
projections.
"""

from __future__ import annotations

from functools import reduce
import math
import types

import pytest
import torch
import torch.nn.functional as F

from src.models import transformer as transformer_mod
from src.models.transformer import (
    Attention,
    ContinuousTransformer,
    apply_rotary_pos_emb,
)


def _azimuthal_rotary_embedding(num_freqs: int):
    """Late lookup keeps the RED module collectable before the class exists."""

    cls = getattr(transformer_mod, "AzimuthalRotaryEmbedding")
    return cls(num_freqs)


def _rotate_half_reference(x: torch.Tensor) -> torch.Tensor:
    first, second = x.reshape(*x.shape[:-1], 2, x.shape[-1] // 2).unbind(-2)
    return torch.cat((-second, first), dim=-1)


def _legacy_2d_rotary_reference(
    tensor: torch.Tensor,
    freqs: torch.Tensor,
    scale: float | torch.Tensor = 1,
) -> torch.Tensor:
    """Frozen reference for the pre-V0 two-dimensional frequency path."""

    out_dtype = tensor.dtype
    dtype = reduce(
        torch.promote_types, (tensor.dtype, freqs.dtype, torch.float32)
    )
    rot_dim = freqs.shape[-1]
    seq_len = tensor.shape[-2]
    freqs = freqs.to(dtype)[-seq_len:, :]
    tensor = tensor.to(dtype)
    rotated, tail = tensor[..., :rot_dim], tensor[..., rot_dim:]
    rotated = (
        rotated * freqs.cos() * scale
        + _rotate_half_reference(rotated) * freqs.sin() * scale
    )
    return torch.cat((rotated.to(out_dtype), tail.to(out_dtype)), dim=-1)


def _batched_rotary_reference(
    tensor: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    """Reference for q/k ``[B,H,N,D]`` plus phases ``[B,N,R]``."""

    out_dtype = tensor.dtype
    dtype = reduce(
        torch.promote_types, (tensor.dtype, freqs.dtype, torch.float32)
    )
    rot_dim = freqs.shape[-1]
    values = tensor.to(dtype)
    broadcast_freqs = freqs.to(dtype).unsqueeze(1)
    rotated, tail = values[..., :rot_dim], values[..., rot_dim:]
    rotated = (
        rotated * broadcast_freqs.cos()
        + _rotate_half_reference(rotated) * broadcast_freqs.sin()
    )
    return torch.cat((rotated.to(out_dtype), tail.to(out_dtype)), dim=-1)


def _identity_output_cross_attention(
    *, qk_norm: str = "ln", differential: bool = False
) -> Attention:
    torch.manual_seed(314159)
    attention = Attention(
        dim=16,
        dim_heads=8,
        dim_context=16,
        qk_norm=qk_norm,
        differential=differential,
        zero_init_output=True,
    )
    with torch.no_grad():
        attention.to_out.weight.copy_(torch.eye(16))
    assert torch.count_nonzero(attention.to_out.weight).item() == 16
    return attention.eval()


# ---------------------------------------------------------------------------
# Azimuthal frequencies and shape-aware rotary application
# ---------------------------------------------------------------------------


def test_azimuthal_rotary_embedding_shape_dtype_and_nonpersistent_buffer():
    embedding = _azimuthal_rotary_embedding(8)
    # Float64 input makes the mandatory FP32 phase path observable on CPU.
    phase = torch.linspace(-math.pi, math.pi, 10, dtype=torch.float64).reshape(2, 5)

    angles = embedding(phase)

    assert angles.shape == (2, 5, 16)
    assert angles.dtype is torch.float32
    buffers = dict(embedding.named_buffers())
    assert len(buffers) == 1
    frequency_name, frequencies = next(iter(buffers.items()))
    torch.testing.assert_close(
        frequencies, torch.arange(1, 9, dtype=torch.float32)
    )
    assert frequency_name not in embedding.state_dict()
    assert frequency_name in embedding._non_persistent_buffers_set
    assert not tuple(embedding.parameters())


def test_azimuthal_rotary_embedding_is_two_pi_periodic_as_a_rotation():
    embedding = _azimuthal_rotary_embedding(8)
    phase = torch.tensor(
        [[-2.25, -0.125, 0.5], [1.0, 2.125, 3.0]], dtype=torch.float32
    )
    values = torch.randn(2, 3, 24, generator=torch.Generator().manual_seed(11))

    once = apply_rotary_pos_emb(values, embedding(phase))
    around_the_circle = apply_rotary_pos_emb(
        values, embedding(phase + 2 * math.pi)
    )

    torch.testing.assert_close(once, around_the_circle, rtol=1e-5, atol=2e-5)


def test_apply_rotary_pos_emb_preserves_the_legacy_2d_frequency_behavior():
    generator = torch.Generator().manual_seed(22)
    values = torch.randn(2, 3, 5, 12, generator=generator)
    # A longer frequency table fixes the old suffix-selection behavior too.
    freqs = torch.randn(9, 8, generator=generator)
    scale = 0.75

    expected = _legacy_2d_rotary_reference(values, freqs, scale)
    actual = apply_rotary_pos_emb(values, freqs, scale)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("batch,tokens", [(5, 3), (2, 5), (4, 4)])
def test_apply_rotary_pos_emb_handles_batched_phases_independent_of_axis_sizes(
    batch: int, tokens: int
):
    """The batch axis must never be mistaken for the legacy sequence axis."""

    generator = torch.Generator().manual_seed(100 + batch * 10 + tokens)
    values = torch.randn(batch, 2, tokens, 12, generator=generator)
    freqs = torch.randn(batch, tokens, 8, generator=generator)

    expected = _batched_rotary_reference(values, freqs)
    actual = apply_rotary_pos_emb(values, freqs)

    assert actual.shape == values.shape
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    # Only the first ``rot_dim`` features participate in azimuth RoPE.
    assert torch.equal(actual[..., 8:], values[..., 8:])


def test_apply_rotary_pos_emb_rejects_a_batched_phase_batch_mismatch():
    values = torch.randn(3, 2, 4, 12)
    wrong_batch = torch.randn(2, 4, 8)

    with pytest.raises((AssertionError, ValueError)):
        apply_rotary_pos_emb(values, wrong_batch)


# ---------------------------------------------------------------------------
# One cross-attention layer: rotate normalized K, never Q or V
# ---------------------------------------------------------------------------


def test_cross_rope_rotates_only_normalized_keys_and_leaves_q_and_v_unchanged():
    attention = _identity_output_cross_attention(qk_norm="ln")
    generator = torch.Generator().manual_seed(33)
    query = torch.randn(2, 3, 16, generator=generator)
    context = torch.randn(2, 4, 16, generator=generator)
    angles = _azimuthal_rotary_embedding(2)(
        torch.tensor(
            [[0.25, -0.5, 1.0, 1.5], [-1.25, 0.75, 0.5, -0.25]]
        )
    )
    captures: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def capture_attn(self, q, k, v, **kwargs):
        captures.append((q.detach().clone(), k.detach().clone(), v.detach().clone()))
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=bool(kwargs.get("causal", False))
        )

    attention.apply_attn = types.MethodType(capture_attn, attention)
    attention(query, context=context)
    attention(query, context=context, cross_rope_phases=angles)

    assert len(captures) == 2
    q_plain, k_plain, v_plain = captures[0]
    q_phase, k_phase, v_phase = captures[1]
    assert torch.equal(q_phase, q_plain), "latent Q must not receive azimuth RoPE"
    assert torch.equal(v_phase, v_plain), "cross-attention V must stay unrotated"
    expected_k = apply_rotary_pos_emb(k_plain, angles)
    torch.testing.assert_close(k_phase, expected_k, rtol=1e-6, atol=1e-6)
    assert not torch.allclose(k_phase, k_plain)


def test_cross_attention_uses_relative_phase_and_is_common_shift_invariant():
    attention = _identity_output_cross_attention(qk_norm="ln")
    embedding = _azimuthal_rotary_embedding(2)
    generator = torch.Generator().manual_seed(44)
    query = torch.randn(2, 3, 16, generator=generator)
    context = torch.randn(2, 4, 16, generator=generator)
    query_phase = torch.tensor([0.25, -0.50])
    key_phase = torch.tensor(
        [[-1.00, -0.25, 0.75, 1.50], [-1.50, -0.25, 0.50, 1.25]]
    )

    def output(keys: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        relative = keys.float() - queries.float().unsqueeze(-1)
        return attention(
            query,
            context=context,
            cross_rope_phases=embedding(relative),
        )

    baseline = output(key_phase, query_phase)
    common_shift = torch.tensor(0.75)
    shifted = output(key_phase + common_shift, query_phase + common_shift)

    assert baseline.abs().max().item() > 1e-5
    torch.testing.assert_close(shifted, baseline, rtol=1e-6, atol=1e-6)

    changed_key_phase = key_phase.clone()
    changed_key_phase[:, 0] += 0.70
    changed = output(changed_key_phase, query_phase)
    assert (changed - baseline).abs().max().item() > 1e-5


def test_cross_rope_none_preserves_the_existing_attention_path():
    attention = _identity_output_cross_attention(qk_norm="l2")
    generator = torch.Generator().manual_seed(55)
    query = torch.randn(2, 3, 16, generator=generator)
    context = torch.randn(2, 4, 16, generator=generator)

    old_call = attention(query, context=context)
    explicit_none = attention(
        query, context=context, cross_rope_phases=None
    )

    torch.testing.assert_close(explicit_none, old_call, rtol=0, atol=0)


def test_cross_rope_explicitly_rejects_differential_attention():
    attention = _identity_output_cross_attention(
        qk_norm="none", differential=True
    )
    query = torch.randn(2, 3, 16)
    context = torch.randn(2, 4, 16)
    angles = _azimuthal_rotary_embedding(2)(torch.randn(2, 4))

    with pytest.raises((AssertionError, ValueError)):
        attention(query, context=context, cross_rope_phases=angles)


# ---------------------------------------------------------------------------
# TransformerBlock / ContinuousTransformer plumbing
# ---------------------------------------------------------------------------


def _tiny_continuous_transformer(global_cond_dim: int | None):
    torch.manual_seed(2718)
    model = ContinuousTransformer(
        dim=16,
        depth=2,
        dim_heads=8,
        cross_attend=True,
        cond_token_dim=16,
        global_cond_dim=global_cond_dim,
        rotary_pos_emb=False,
        zero_init_branch_outputs=False,
        attn_kwargs={"qk_norm": "ln"},
    ).eval()
    # Make the non-vacuity condition explicit rather than relying on random
    # initialization details of ``zero_init_branch_outputs=False``.
    with torch.no_grad():
        for layer in model.layers:
            layer.cross_attn.to_out.weight.copy_(torch.eye(16))
            assert torch.count_nonzero(layer.cross_attn.to_out.weight).item() == 16
    return model


@pytest.mark.parametrize("use_checkpointing", [False, True])
@pytest.mark.parametrize("global_cond_dim", [None, 16])
def test_continuous_transformer_plumbs_cross_rope_through_every_block_branch(
    use_checkpointing: bool, global_cond_dim: int | None
):
    model = _tiny_continuous_transformer(global_cond_dim)
    generator = torch.Generator().manual_seed(66)
    x = torch.randn(2, 3, 16, generator=generator, requires_grad=True)
    context = torch.randn(2, 4, 16, generator=generator)
    relative_phase = torch.tensor(
        [[-1.0, -0.25, 0.50, 1.25], [-1.5, 0.0, 0.75, 1.50]]
    )
    angles = _azimuthal_rotary_embedding(2)(relative_phase)
    global_cond = (
        torch.randn(2, global_cond_dim, generator=generator)
        if global_cond_dim is not None
        else None
    )
    seen: list[torch.Tensor | None] = []

    def record_phase(_module, _args, kwargs):
        value = kwargs.get("cross_rope_phases")
        seen.append(None if value is None else value.detach().clone())

    handles = [
        layer.cross_attn.register_forward_pre_hook(record_phase, with_kwargs=True)
        for layer in model.layers
    ]
    try:
        baseline = model(
            x,
            context=context,
            global_cond=global_cond,
            cross_rope_phases=angles,
            use_checkpointing=use_checkpointing,
        )
    finally:
        for handle in handles:
            handle.remove()

    assert len(seen) == len(model.layers)
    for received in seen:
        assert received is not None
        assert received.dtype is torch.float32
        torch.testing.assert_close(received, angles, rtol=0, atol=0)

    changed_phase = relative_phase.clone()
    changed_phase[:, 0] += 0.70
    changed = model(
        x,
        context=context,
        global_cond=global_cond,
        cross_rope_phases=_azimuthal_rotary_embedding(2)(changed_phase),
        use_checkpointing=use_checkpointing,
    )
    assert (changed - baseline).abs().max().item() > 1e-5


def test_checkpointed_and_direct_cross_rope_paths_are_numerically_identical():
    model = _tiny_continuous_transformer(global_cond_dim=16)
    generator = torch.Generator().manual_seed(77)
    x = torch.randn(2, 3, 16, generator=generator, requires_grad=True)
    context = torch.randn(2, 4, 16, generator=generator)
    global_cond = torch.randn(2, 16, generator=generator)
    angles = _azimuthal_rotary_embedding(2)(torch.randn(2, 4, generator=generator))

    direct = model(
        x,
        context=context,
        global_cond=global_cond,
        cross_rope_phases=angles,
        use_checkpointing=False,
    )
    checkpointed = model(
        x,
        context=context,
        global_cond=global_cond,
        cross_rope_phases=angles,
        use_checkpointing=True,
    )

    torch.testing.assert_close(checkpointed, direct, rtol=1e-6, atol=1e-6)
