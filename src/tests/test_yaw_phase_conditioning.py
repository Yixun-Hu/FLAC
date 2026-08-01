"""TDD contract tests for V0 yaw-phase conditioning assembly.

These tests deliberately use tiny, deterministic conditioner outputs rather than
constructing CylViT or AudioResNet.  Geometry/pose conditioner behavior is tested
separately; this module fixes the wrapper boundary that joins those entries into
the 34-token DiT context.
"""

import types

import pytest
import torch
from torch import nn

from src.models.diffusion import ConditionedDiffusionModelWrapper


BATCH = 2
DIM = 8
CROSS_IDS = (
    "source_vit",
    "context_poses_vit",
    "context_poses",
    "context_audio",
)


class _ConditionerSpec(nn.Module):
    """Parameter-free stand-in exposing the output dimension if queried."""

    def __init__(self, output_dim=DIM):
        super().__init__()
        self.output_dim = output_dim


class _FakeMultiConditioner(nn.Module):
    def __init__(self, output_dim=DIM):
        super().__init__()
        self.output_dim = output_dim
        self.conditioners = nn.ModuleDict(
            {key: _ConditionerSpec(output_dim) for key in (*CROSS_IDS, "source")}
        )


class _FakeConditionedModel(nn.Module):
    """Small model that records exactly what the outer wrapper forwards."""

    def __init__(self, cond_token_dim=DIM):
        super().__init__()
        # The real object passed to the wrapper is DiTWrapper, whose nested
        # DiffusionTransformer exposes ``cond_token_dim`` at ``model.model``.
        self.model = types.SimpleNamespace(cond_token_dim=cond_token_dim)
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.last_kwargs = None

    def forward(self, x, t, **kwargs):
        self.last_kwargs = kwargs
        return x * self.gain


def _wrapper(*, query_phase_cond_id=None, phase_aliases=None, cross_ids=CROSS_IDS):
    kwargs = {}
    if query_phase_cond_id is not None:
        kwargs["query_phase_cond_id"] = query_phase_cond_id
    if phase_aliases is not None:
        kwargs["phase_aliases"] = phase_aliases
    return ConditionedDiffusionModelWrapper(
        model=_FakeConditionedModel(),
        conditioner=_FakeMultiConditioner(),
        io_channels=4,
        sample_rate=22_050,
        min_input_length=1,
        cross_attn_cond_ids=list(cross_ids),
        global_cond_ids=["source"],
        **kwargs,
    )


def _phase_wrapper(*, phase_aliases=None):
    if phase_aliases is None:
        phase_aliases = {"context_audio": "context_poses"}
    wrapper = _wrapper(
        query_phase_cond_id="source",
        phase_aliases=phase_aliases,
    )
    # Assembly/order assertions must not depend on random initialization.
    with torch.no_grad():
        for embedding in wrapper.cross_attn_type_embeddings.values():
            embedding.zero_()
    return wrapper


def _entry(token_count, value, phase_start, *, phase_dtype=torch.float32):
    content = torch.full((BATCH, token_count, DIM), float(value))
    mask = torch.full((BATCH, token_count), float(value))
    phase = (
        torch.arange(token_count, dtype=torch.float32)
        .view(1, token_count)
        .expand(BATCH, token_count)
        .add(float(phase_start))
        .to(phase_dtype)
    )
    return [content, mask, phase]


def _v0_entries(*, phase_dtype=torch.float32):
    source = torch.full((BATCH, 1, DIM), 5.0)
    source_mask = torch.ones(BATCH, 1)
    # Different query phases per sample ensure assembly does not accidentally
    # select only the first row or flatten the batch.
    source_phase = torch.tensor([[0.25], [-0.75]], dtype=phase_dtype)
    return {
        "source_vit": _entry(16, 1, 10, phase_dtype=phase_dtype),
        "context_poses_vit": _entry(16, 2, 30, phase_dtype=phase_dtype),
        "context_poses": _entry(1, 3, 50, phase_dtype=phase_dtype),
        # RIRConditioner is intentionally phase-free; its phase is resolved by
        # the explicit context_audio -> context_poses alias.
        "context_audio": [
            torch.full((BATCH, 1, DIM), 4.0),
            torch.full((BATCH, 1), 4.0),
        ],
        "source": [source, source_mask, source_phase],
    }


def test_old_two_item_entries_preserve_the_exact_output_contract():
    """An inactive wrapper must keep accepting every legacy two-item entry."""

    wrapper = _wrapper()
    entries = {
        key: [
            torch.full((BATCH, 1, DIM), float(i)),
            torch.full((BATCH, 1), float(i)),
        ]
        for i, key in enumerate((*CROSS_IDS, "source"), start=1)
    }

    assembled = wrapper.get_conditioning_inputs(entries)

    assert set(assembled) == {
        "cross_attn_cond",
        "cross_attn_mask",
        "global_cond",
        "input_concat_cond",
        "prepend_cond",
        "prepend_cond_mask",
    }
    assert assembled["cross_attn_cond"].shape == (BATCH, 4, DIM)
    assert assembled["cross_attn_mask"].shape == (BATCH, 4)
    assert assembled["global_cond"].shape == (BATCH, DIM)
    for index in range(4):
        assert torch.equal(
            assembled["cross_attn_cond"][:, index],
            torch.full((BATCH, DIM), float(index + 1)),
        )
    assert "cross_attn_phases" not in assembled
    assert "query_phase" not in assembled


def test_inactive_defaults_add_no_modules_or_state_dict_keys():
    """The old opt-out path must remain checkpoint-key compatible."""

    default = _wrapper()
    explicit_inactive = _wrapper(query_phase_cond_id=None, phase_aliases=None)

    assert default.query_phase_cond_id is None
    assert default.phase_aliases == {}
    assert getattr(default, "cross_attn_type_embeddings", None) is None
    assert set(default.state_dict()) == {"model.gain"}
    assert set(explicit_inactive.state_dict()) == set(default.state_dict())


def test_phase_triplets_assemble_fixed_34_token_order_and_shapes():
    wrapper = _phase_wrapper()
    entries = _v0_entries()

    assembled = wrapper.get_conditioning_inputs(entries)

    assert assembled["cross_attn_cond"].shape == (BATCH, 34, DIM)
    assert assembled["cross_attn_mask"].shape == (BATCH, 34)
    assert assembled["cross_attn_phases"].shape == (BATCH, 34)
    assert assembled["query_phase"].shape == (BATCH,)
    assert assembled["global_cond"].shape == (BATCH, DIM)

    # Fixed V0 order: 16 target geometry, 16 context geometry, one context
    # pose and one context audio token.
    expected_content = torch.cat(
        [entries[key][0] for key in CROSS_IDS], dim=1
    )
    expected_mask = torch.cat([entries[key][1] for key in CROSS_IDS], dim=1)
    expected_phase = torch.cat(
        [
            entries["source_vit"][2],
            entries["context_poses_vit"][2],
            entries["context_poses"][2],
            entries["context_poses"][2],  # context_audio alias
        ],
        dim=1,
    )
    assert torch.equal(assembled["cross_attn_cond"], expected_content)
    assert torch.equal(assembled["cross_attn_mask"], expected_mask)
    assert torch.equal(assembled["cross_attn_phases"], expected_phase)
    assert torch.equal(assembled["query_phase"], entries["source"][2][:, 0])
    assert torch.equal(assembled["global_cond"], entries["source"][0][:, 0])


def test_audio_phase_is_resolved_from_explicit_pose_alias():
    wrapper = _phase_wrapper()
    entries = _v0_entries()
    assembled = wrapper.get_conditioning_inputs(entries)

    assert torch.equal(
        assembled["cross_attn_phases"][:, -1],
        entries["context_poses"][2][:, 0],
    )


def test_query_and_cross_phases_are_forced_to_float32():
    wrapper = _phase_wrapper()
    # Float64 catches a missing explicit cast even on CPU. The implementation
    # must keep phases FP32 independently of the content/model dtype.
    assembled = wrapper.get_conditioning_inputs(
        _v0_entries(phase_dtype=torch.float64)
    )

    assert assembled["cross_attn_phases"].dtype is torch.float32
    assert assembled["query_phase"].dtype is torch.float32


def test_phase_tensors_are_forwarded_to_the_conditioned_model():
    wrapper = _phase_wrapper()
    entries = _v0_entries()
    x = torch.randn(BATCH, 4, 3)
    t = torch.rand(BATCH)

    out = wrapper(x, t, entries)

    assert torch.equal(out, x)
    assert wrapper.model.last_kwargs["cross_attn_phases"].shape == (BATCH, 34)
    assert wrapper.model.last_kwargs["query_phase"].shape == (BATCH,)


def test_type_embeddings_are_opt_in_and_assigned_by_stream():
    wrapper = _phase_wrapper()

    assert isinstance(wrapper.cross_attn_type_embeddings, nn.ParameterDict)
    assert set(wrapper.cross_attn_type_embeddings) == set(CROSS_IDS)
    for embedding in wrapper.cross_attn_type_embeddings.values():
        assert embedding.numel() == DIM

    # Make the assignment observable without relying on random init values.
    with torch.no_grad():
        for index, key in enumerate(CROSS_IDS, start=1):
            wrapper.cross_attn_type_embeddings[key].fill_(index * 10.0)

    entries = _v0_entries()
    assembled = wrapper.get_conditioning_inputs(entries)
    starts = (0, 16, 32, 33)
    for index, (key, start) in enumerate(zip(CROSS_IDS, starts), start=1):
        expected = entries[key][0] + index * 10.0
        stop = start + entries[key][0].shape[1]
        assert torch.equal(assembled["cross_attn_cond"][:, start:stop], expected)

    # Global target-pose content is not one of the four cross streams.
    assert torch.equal(assembled["global_cond"], entries["source"][0][:, 0])
    type_state_keys = {
        key for key in wrapper.state_dict() if key.startswith("cross_attn_type_embeddings.")
    }
    assert type_state_keys == {
        f"cross_attn_type_embeddings.{key}" for key in CROSS_IDS
    }


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda entries: entries.pop("context_audio"), "context_audio"),
        (
            lambda entries: entries.__setitem__(
                "context_poses", entries["context_poses"][:2]
            ),
            "phase",
        ),
        (
            lambda entries: entries["source_vit"].__setitem__(
                1, entries["source_vit"][1][:, :-1]
            ),
            "mask|length|16",
        ),
        (
            lambda entries: entries["source_vit"].__setitem__(
                2, entries["source_vit"][2][:, :-1]
            ),
            "phase|length|16",
        ),
    ],
    ids=("missing-stream", "missing-phase", "short-mask", "short-phase"),
)
def test_phase_aware_assembly_fails_fast_on_missing_or_misaligned_entries(
    mutation, match
):
    wrapper = _phase_wrapper()
    entries = _v0_entries()
    mutation(entries)

    with pytest.raises(ValueError, match=match):
        wrapper.get_conditioning_inputs(entries)


def test_missing_phase_alias_is_rejected():
    wrapper = _phase_wrapper(phase_aliases={})

    with pytest.raises(ValueError, match="context_audio|phase|alias"):
        wrapper.get_conditioning_inputs(_v0_entries())


def test_alias_source_must_exist_and_have_a_phase():
    wrapper = _phase_wrapper(phase_aliases={"context_audio": "unknown_pose"})

    with pytest.raises(ValueError, match="unknown_pose|alias|phase"):
        wrapper.get_conditioning_inputs(_v0_entries())


def test_k_greater_than_one_is_rejected_before_non_v0_assembly():
    wrapper = _phase_wrapper()
    entries = _v0_entries()
    # A realistic K=2 conditioner result has two context pose/audio tokens and
    # 2*16 context-geometry tokens. It must not silently produce 52 tokens.
    entries["context_poses_vit"] = _entry(32, 2, 30)
    entries["context_poses"] = _entry(2, 3, 50)
    entries["context_audio"] = [
        torch.full((BATCH, 2, DIM), 4.0),
        torch.full((BATCH, 2), 4.0),
    ]

    with pytest.raises(ValueError, match="K=1|34|exactly 16"):
        wrapper.get_conditioning_inputs(entries)


def test_fixed_stream_counts_reject_compensating_15_plus_17_tokens():
    wrapper = _phase_wrapper()
    entries = _v0_entries()
    entries["source_vit"] = _entry(15, 1, 10)
    entries["context_poses_vit"] = _entry(17, 2, 30)

    with pytest.raises(ValueError, match="source_vit|16|exactly"):
        wrapper.get_conditioning_inputs(entries)


def test_phase_aware_constructor_rejects_reordered_cross_stream_ids():
    reordered = (
        "context_poses_vit",
        "source_vit",
        "context_poses",
        "context_audio",
    )

    with pytest.raises(ValueError, match="fixed|order|source_vit"):
        _wrapper(
            query_phase_cond_id="source",
            phase_aliases={"context_audio": "context_poses"},
            cross_ids=reordered,
        )
