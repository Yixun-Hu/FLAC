"""Round-2 contracts for the opt-in C16 relative-phase conditioning path.

These tests intentionally exercise the public seams described in
``design_yaw_equivariant_dit_v0.md`` while keeping every legacy conditioner
entry a two-item ``[content, mask]`` list.  The new pose and azimuth-pooled
geometry entries are three-item ``[content, mask, phase]`` lists.
"""

from __future__ import annotations

import copy
import math
from unittest import mock

import pytest
import torch
from torch import nn

from src.data import yaw_rotation as yr
from src.models import conditioners as conditioner_mod
from src.models.conditioners import (
    GeometryConditioner,
    create_multi_conditioner_from_conditioning_config,
)
from src.models.cyl_vit import CylindricalViT


DEV = "cpu"


def _pose_bundle(metadata: dict, eps: float = 1e-6) -> dict:
    """Late lookup lets the RED suite collect before the helper exists."""

    return getattr(yr, "yaw_pose_content_and_phase")(metadata, eps=eps)


def _wrapped_residual(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    return yr.wrap_angle(actual - expected)


def _yaw_conditioning_config(cond_dim: int = 6) -> dict:
    common = {
        "num_freqs": 2,
        "max_freq": 3,
        "include_in": True,
        "radial_max_val": 10.0,
        "height_max_val": 5.0,
    }
    return {
        "cond_dim": cond_dim,
        "configs": [
            {
                "id": "source",
                "type": "yaw_pose",
                "config": {**common, "pose_role": "target"},
            },
            {
                "id": "context_poses",
                "type": "yaw_pose",
                "config": {**common, "pose_role": "context"},
            },
        ],
    }


def _k1_pose_batch(dtype: torch.dtype = torch.float32) -> list[dict]:
    return [
        {
            "source": torch.tensor([3.0, 4.0, 1.0], dtype=dtype),
            "context_poses": torch.tensor([[0.0, 2.0, -1.0]], dtype=dtype),
        },
        {
            "source": torch.tensor([0.0, -2.0, 3.0], dtype=dtype),
            "context_poses": torch.tensor([[-4.0, 0.0, 2.0]], dtype=dtype),
        },
    ]


# ---------------------------------------------------------------------------
# Joint pose content / phase helper
# ---------------------------------------------------------------------------


def test_joint_pose_bundle_values_and_common_yaw_action():
    metadata = {
        "source": torch.tensor([1.0, 1.0, 0.5]),
        "context_poses": torch.tensor(
            [[0.0, 2.0, 1.0], [-3.0, 0.0, -0.25]]
        ),
    }
    base = _pose_bundle(metadata)

    torch.testing.assert_close(
        base["target_content"], torch.tensor([math.sqrt(2.0), 0.5])
    )
    torch.testing.assert_close(
        base["context_content"],
        torch.tensor([[2.0, 1.0], [3.0, -0.25]]),
    )
    torch.testing.assert_close(base["target_phase"], torch.tensor(math.pi / 4))
    torch.testing.assert_close(
        base["context_phases"], torch.tensor([math.pi / 2, math.pi])
    )

    alpha = math.pi / 8
    rotated = yr.rotate_scene_metadata(
        metadata,
        alpha,
        img_w=512,
        pose_keys=("source", "context_poses"),
    )
    moved = _pose_bundle(rotated)

    torch.testing.assert_close(moved["target_content"], base["target_content"])
    torch.testing.assert_close(moved["context_content"], base["context_content"])
    torch.testing.assert_close(
        _wrapped_residual(moved["target_phase"], base["target_phase"] + alpha),
        torch.zeros_like(moved["target_phase"]),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        _wrapped_residual(
            moved["context_phases"], base["context_phases"] + alpha
        ),
        torch.zeros_like(moved["context_phases"]),
        atol=1e-6,
        rtol=0,
    )

    base_relative = yr.wrap_angle(
        base["context_phases"] - base["target_phase"]
    )
    moved_relative = yr.wrap_angle(
        moved["context_phases"] - moved["target_phase"]
    )
    torch.testing.assert_close(moved_relative, base_relative, atol=1e-6, rtol=0)


def test_joint_pose_bundle_degenerate_target_uses_largest_radius_fallback():
    metadata = {
        "source": torch.tensor([0.0, 0.0, 1.5]),
        "context_poses": torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, -3.0, 2.0],  # largest radius -> reference -pi/2
                [0.0, 0.0, -2.0],  # individually undefined -> reference
            ]
        ),
    }
    bundle = _pose_bundle(metadata)

    torch.testing.assert_close(bundle["target_phase"], torch.tensor(-math.pi / 2))
    torch.testing.assert_close(
        bundle["context_phases"],
        torch.tensor([0.0, -math.pi / 2, -math.pi / 2]),
    )
    assert float(bundle["context_phases"][2] - bundle["target_phase"]) == 0.0


def test_joint_pose_bundle_individual_degenerate_context_uses_reference():
    metadata = {
        "source": torch.tensor([0.0, 2.0, 0.5]),
        "context_poses": torch.tensor(
            [[0.0, 0.0, 3.0], [-2.0, 0.0, -1.0]]
        ),
    }
    bundle = _pose_bundle(metadata)

    torch.testing.assert_close(bundle["target_phase"], torch.tensor(math.pi / 2))
    torch.testing.assert_close(
        bundle["context_phases"], torch.tensor([math.pi / 2, math.pi])
    )


def test_joint_pose_bundle_all_degenerate_is_finite_and_zero_phase():
    metadata = {
        "source": torch.tensor([1e-9, -3e-10, 1.0]),
        "context_poses": torch.tensor(
            [[-2e-10, 7e-10, 0.5], [5e-10, 5e-10, -0.5]]
        ),
    }
    bundle = _pose_bundle(metadata, eps=1e-6)

    assert torch.isfinite(bundle["target_content"]).all()
    assert torch.isfinite(bundle["context_content"]).all()
    assert float(bundle["target_phase"]) == 0.0
    assert torch.equal(bundle["context_phases"], torch.zeros(2))


def test_joint_pose_bundle_is_nonmutating_and_preserves_dtype_device():
    marker = torch.tensor([9.0], dtype=torch.float64)
    metadata = {
        "source": torch.tensor([3.0, 4.0, 2.0], dtype=torch.float64),
        "context_poses": torch.tensor([[0.0, -2.0, 1.0]], dtype=torch.float64),
        "unrelated": marker,
    }
    before = copy.deepcopy(metadata)
    bundle = _pose_bundle(metadata)

    for key in before:
        assert torch.equal(metadata[key], before[key]), f"{key} was mutated"
    assert metadata["unrelated"] is marker
    for key in ("target_content", "context_content"):
        assert bundle[key].dtype == torch.float64
        assert bundle[key].device == metadata["source"].device
    for key in ("target_phase", "context_phases"):
        assert bundle[key].dtype == torch.float32
        assert bundle[key].device == metadata["source"].device


# ---------------------------------------------------------------------------
# CylViT phase buffer and GeometryConditioner pooling
# ---------------------------------------------------------------------------


def test_cylvit_exposes_nonpersistent_column_center_phases():
    model = CylindricalViT(
        image_size=(16, 512),
        patch_size=(16, 32),
        dim=32,
        depth=0,
        heads=1,
        dim_head=32,
    )
    expected = (
        (torch.arange(16, dtype=torch.float32) + 0.5) * (2.0 * math.pi / 16)
        - math.pi
    )

    phases = model.azimuth_phases
    assert phases.shape == (16,)
    assert phases.dtype == torch.float32
    torch.testing.assert_close(phases, expected)
    assert "azimuth_phases" in dict(model.named_buffers())
    assert "azimuth_phases" not in model.state_dict()


class _GridTokenViT(nn.Module):
    """Return deterministic h-major tokens with easily audited column order."""

    def __init__(self, h_tok: int = 2, w_tok: int = 16, dim: int = 3):
        super().__init__()
        self.h_tok = h_tok
        self.w_tok = w_tok
        self.num_tokens = h_tok * w_tok
        phases = (
            (torch.arange(w_tok, dtype=torch.float32) + 0.5)
            * (2.0 * math.pi / w_tok)
            - math.pi
        )
        self.register_buffer("azimuth_phases", phases, persistent=False)
        rows = torch.arange(h_tok, dtype=torch.float32)[:, None] * 100.0
        cols = torch.arange(w_tok, dtype=torch.float32)[None, :]
        scalar = (rows + cols).reshape(self.num_tokens, 1)
        channels = torch.cat([scalar, scalar + 1.0, scalar + 2.0], dim=-1)
        self.register_buffer("tokens", channels)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.tokens.unsqueeze(0).expand(image.shape[0], -1, -1)


def _geometry_inputs(batch: int = 2, contexts: int = 1) -> list[dict]:
    inputs = []
    for index in range(batch):
        coord = torch.arange(contexts * 3, dtype=torch.float32).reshape(contexts, 3)
        if contexts == 1:
            coord = coord[0]
        inputs.append(
            {
                "coord": coord + index,
                "depth": torch.zeros(3, 2, 2),
            }
        )
    return inputs


def _geometry_conditioner(token_pool: str, lin_proj: nn.Module | None = None):
    vit = _GridTokenViT()
    if lin_proj is None:
        lin_proj = nn.Linear(vit.num_tokens, 1, bias=False)
    return GeometryConditioner(
        vit_model=vit,
        vit_proj=nn.Identity(),
        lin_proj=lin_proj,
        output_dim=3,
        dim=3,
        model_type="vit",
        token_pool=token_pool,
    )


def test_azimuth_pooling_returns_aligned_content_mask_phase_in_h_major_order():
    conditioner = _geometry_conditioner("azimuth")
    content, mask, phase = conditioner(_geometry_inputs(batch=2), device=DEV)

    tokens = conditioner.vit.tokens.reshape(2, 16, 3)
    expected_content = tokens.mean(dim=0).unsqueeze(0).expand(2, -1, -1)
    expected_phase = conditioner.vit.azimuth_phases.unsqueeze(0).expand(2, -1)

    assert content.shape == (2, 16, 3)
    assert mask.shape == (2, 16)
    assert phase.shape == (2, 16)
    torch.testing.assert_close(content, expected_content)
    assert torch.equal(mask, torch.ones_like(mask))
    torch.testing.assert_close(phase, expected_phase)
    assert phase.dtype == torch.float32


def test_azimuth_pooling_rejects_more_than_one_context_before_encoding():
    conditioner = _geometry_conditioner("azimuth")
    with mock.patch.object(conditioner.vit, "forward", wraps=conditioner.vit.forward) as run:
        with pytest.raises(ValueError, match=r"K=1|exactly one|one context"):
            conditioner(_geometry_inputs(batch=1, contexts=2), device=DEV)
    run.assert_not_called()


@pytest.mark.parametrize("token_pool", ["mean", "linear"])
def test_legacy_geometry_pooling_remains_two_item_and_numerically_identical(token_pool):
    vit = _GridTokenViT()
    linear = nn.Linear(vit.num_tokens, 1, bias=False)
    with torch.no_grad():
        linear.weight.copy_(
            torch.linspace(-0.25, 0.75, vit.num_tokens).reshape(1, -1)
        )
    conditioner = GeometryConditioner(
        vit_model=vit,
        vit_proj=nn.Identity(),
        lin_proj=linear,
        output_dim=3,
        dim=3,
        model_type="vit",
        token_pool=token_pool,
    )
    entry = conditioner(_geometry_inputs(batch=2), device=DEV)

    assert len(entry) == 2
    content, mask = entry
    tokens = vit.tokens.unsqueeze(0).expand(2, -1, -1)
    if token_pool == "mean":
        expected = tokens.mean(dim=1, keepdim=True)
    else:
        expected = linear(tokens.permute(0, 2, 1)).squeeze(-1).unsqueeze(1)
    torch.testing.assert_close(content, expected)
    assert mask.shape == (2, 1)


def test_linear_cylvit_azimuth_pooling_is_c16_equivariant_with_destination_phases():
    """Intermediate encoder+pooling control requested by the plan review."""

    torch.manual_seed(20260718)
    vit = CylindricalViT(
        image_size=(32, 512),
        patch_size=(16, 32),
        dim=32,
        depth=0,
        heads=1,
        dim_head=32,
        patch_embed_type="linear",
    ).eval()
    conditioner = GeometryConditioner(
        vit_model=vit,
        vit_proj=nn.Identity(),
        lin_proj=nn.Linear(vit.num_tokens, 1),
        output_dim=32,
        dim=32,
        model_type="vit",
        token_pool="azimuth",
    ).eval()
    metadata = {
        "source_vit": torch.tensor([0.7, -1.2, 0.3]),
        "depth": torch.randn(3, 32, 512),
    }
    orbit = []
    for shift in range(16):
        moved = yr.rotate_scene_metadata(
            metadata,
            shift * 2.0 * math.pi / 16,
            img_w=512,
            pose_keys=("source_vit",),
        )
        orbit.append({"coord": moved["source_vit"], "depth": moved["depth"]})

    with torch.no_grad():
        content, mask, phases = conditioner(orbit, device=DEV)

    expected = torch.stack(
        [torch.roll(content[0], shifts=shift, dims=0) for shift in range(16)]
    )
    torch.testing.assert_close(content, expected, rtol=2e-5, atol=3e-5)
    assert mask.shape == (16, 16)
    torch.testing.assert_close(phases, phases[:1].expand_as(phases))


# ---------------------------------------------------------------------------
# YawPoseConditioner factory sharing and MultiConditioner joint routing
# ---------------------------------------------------------------------------


def test_yaw_pose_factory_shares_projection_and_emits_triplet_contracts():
    multi = create_multi_conditioner_from_conditioning_config(
        _yaw_conditioning_config(cond_dim=6)
    )
    target = multi.conditioners["source"]
    context = multi.conditioners["context_poses"]

    yaw_cls = getattr(conditioner_mod, "YawPoseConditioner")
    assert isinstance(target, yaw_cls)
    assert isinstance(context, yaw_cls)
    assert target.pose_role == "target"
    assert context.pose_role == "context"
    assert target.yaw_pose_proj is context.yaw_pose_proj
    assert target.yaw_pose_proj.in_features == 10  # 2 * (2*num_freqs + raw)
    assert target.yaw_pose_proj.out_features == 6

    outputs = multi(_k1_pose_batch(), DEV)
    for key in ("source", "context_poses"):
        assert len(outputs[key]) == 3
        content, mask, phase = outputs[key]
        assert content.shape == (2, 1, 6)
        assert mask.shape == (2, 1)
        assert phase.shape == (2, 1)
        assert torch.equal(mask, torch.ones_like(mask))
        assert phase.dtype == torch.float32

    torch.testing.assert_close(
        outputs["source"][2][:, 0], torch.tensor([math.atan2(4.0, 3.0), -math.pi / 2])
    )
    torch.testing.assert_close(
        outputs["context_poses"][2][:, 0], torch.tensor([math.pi / 2, math.pi])
    )


def test_yaw_pose_conditioner_rejects_unknown_role():
    yaw_cls = getattr(conditioner_mod, "YawPoseConditioner")
    projection = nn.Linear(6, 4)
    with pytest.raises(ValueError, match="pose_role"):
        yaw_cls(
            output_dim=4,
            pose_role="listener",
            num_freqs=1,
            include_in=True,
            radial_max_val=5.0,
            height_max_val=5.0,
            yaw_pose_proj=projection,
        )


def test_multiconditioner_computes_one_joint_bundle_per_sample_and_honors_only_ids():
    multi = create_multi_conditioner_from_conditioning_config(
        _yaw_conditioning_config(cond_dim=6)
    )
    batch = [
        {
            "source": torch.tensor([0.0, 0.0, 1.0]),
            "context_poses": torch.tensor([[0.0, 2.0, -1.0]]),
        }
    ]
    original = yr.yaw_pose_content_and_phase
    owner = (
        conditioner_mod
        if hasattr(conditioner_mod, "yaw_pose_content_and_phase")
        else yr
    )
    with mock.patch.object(
        owner, "yaw_pose_content_and_phase", wraps=original
    ) as joint_helper:
        output = multi(batch, DEV, only_ids=("source",))

    assert set(output) == {"source"}
    joint_helper.assert_called_once()
    # Target is degenerate, so the skipped context conditioner's raw pose still
    # has to participate in the common fallback reference.
    torch.testing.assert_close(
        output["source"][2], torch.tensor([[math.pi / 2]], dtype=torch.float32)
    )


class _CaptureGeometry(nn.Module):
    name = "GeometryConditioner"

    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, values, device=DEV):
        self.seen = values
        batch = len(values)
        return [torch.zeros(batch, 1, 2), torch.ones(batch, 1)]


def test_joint_pose_routing_does_not_replace_raw_cartesian_geometry_fields():
    multi = create_multi_conditioner_from_conditioning_config(
        _yaw_conditioning_config(cond_dim=6)
    )
    capture = _CaptureGeometry()
    multi.conditioners["source_vit"] = capture
    batch = _k1_pose_batch()
    for item in batch:
        item["source_vit"] = item["source"].clone()
        item["depth"] = torch.randn(3, 2, 2)
    before = copy.deepcopy(batch)

    _ = multi(batch, DEV)

    assert capture.seen is not None
    for index, seen in enumerate(capture.seen):
        assert torch.equal(seen["coord"], before[index]["source_vit"])
        assert torch.equal(seen["depth"], before[index]["depth"])
    for item, reference in zip(batch, before):
        for key in reference:
            assert torch.equal(item[key], reference[key]), f"{key} was mutated"


def test_phase_aware_multiconditioner_rejects_k_greater_than_one():
    multi = create_multi_conditioner_from_conditioning_config(
        _yaw_conditioning_config(cond_dim=6)
    )
    batch = [
        {
            "source": torch.tensor([1.0, 0.0, 0.0]),
            "context_poses": torch.tensor(
                [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
            ),
        }
    ]
    with pytest.raises(ValueError, match=r"K=1|exactly one|one context"):
        multi(batch, DEV)


def test_yaw_pose_phase_stays_float32_under_autocast():
    multi = create_multi_conditioner_from_conditioning_config(
        _yaw_conditioning_config(cond_dim=6)
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        outputs = multi(_k1_pose_batch(dtype=torch.float32), DEV)

    assert outputs["source"][2].dtype == torch.float32
    assert outputs["context_poses"][2].dtype == torch.float32


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_freq", 4),
        ("radial_max_val", 11.0),
        ("height_max_val", 6.0),
    ],
)
def test_shared_yaw_pose_projection_rejects_mismatched_encoder_semantics(
    field, value
):
    config = _yaw_conditioning_config(cond_dim=6)
    config["configs"][1]["config"][field] = value

    with pytest.raises(ValueError, match="identical|semantics|share"):
        create_multi_conditioner_from_conditioning_config(config)


def test_yaw_pose_explicit_bfloat16_weights_keep_content_and_phase_dtypes():
    multi = create_multi_conditioner_from_conditioning_config(
        _yaw_conditioning_config(cond_dim=6)
    ).to(dtype=torch.bfloat16)

    outputs = multi(_k1_pose_batch(dtype=torch.float32), DEV)

    assert outputs["source"][0].dtype == torch.bfloat16
    assert outputs["context_poses"][0].dtype == torch.bfloat16
    assert outputs["source"][2].dtype == torch.float32
    assert outputs["context_poses"][2].dtype == torch.float32


def test_both_pose_roles_accumulate_gradient_into_the_shared_projection():
    multi = create_multi_conditioner_from_conditioning_config(
        _yaw_conditioning_config(cond_dim=6)
    )
    shared = multi.conditioners["source"].yaw_pose_proj
    assert shared is multi.conditioners["context_poses"].yaw_pose_proj

    outputs = multi(_k1_pose_batch(), DEV)
    (outputs["source"][0].sum() + outputs["context_poses"][0].sum()).backward()

    assert shared.weight.grad is not None
    assert torch.isfinite(shared.weight.grad).all()


def test_azimuth_pooling_rejects_phase_buffer_length_mismatch():
    conditioner = _geometry_conditioner("azimuth")
    conditioner.vit.azimuth_phases = torch.zeros(15)

    with pytest.raises(ValueError, match="phase count|column grid"):
        conditioner(_geometry_inputs(batch=1), device=DEV)
