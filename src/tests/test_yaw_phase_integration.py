"""Non-vacuous raw-metadata integration checks for yaw-phase DiT V0."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch

from src.data.yaw_rotation import rotate_scene_metadata
from src.models.factory import create_model_from_config


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs/model_configs/FLAC/AR/FLAC_AR_YawPhaseDiT_C16.json"
)
C16 = tuple(shift * 2.0 * math.pi / 16 for shift in range(16))

# Pre-registered before observing the positive-path result.  The identity floor
# must be exact, the equivariance ceiling leaves room for chained FP32 kernels,
# and every broken control must clear both an absolute floor and 10x the good
# C16 defect.  BF16 has a separate quantization-aware ceiling.
FP32_C16_ATOL = 5.0e-5
BF16_C16_ATOL = 2.0e-2
SENSITIVITY_FLOOR = 1.0e-5
BROKEN_TO_GOOD_RATIO = 10.0


def _tiny_v0_config(*, azimuth_num_freqs: int = 8) -> dict:
    with CONFIG_PATH.open() as handle:
        config = copy.deepcopy(json.load(handle))

    config["sample_size"] = 8
    config["model"].pop("pretransform")
    config["model"]["io_channels"] = 4
    config["model"]["conditioning"]["cond_dim"] = 32

    conditioners = {
        item["id"]: item
        for item in config["model"]["conditioning"]["configs"]
    }
    for condition_id in ("source_vit", "context_poses_vit"):
        vit = conditioners[condition_id]["config"]["ViT"]
        vit.update(
            {
                "img_h": 2,
                "img_w": 16,
                "patch_h": 2,
                "patch_w": 1,
                "dim": 32,
                "depth": 1,
                "heads": 1,
                "mlp_dim": 32,
            }
        )
    audio = conditioners["context_audio"]["config"]
    audio.update({"n_fft": 32, "win_length": 16, "hop_length": 8})

    dit = config["model"]["diffusion"]["config"]
    dit.update(
        {
            "io_channels": 4,
            "embed_dim": 32,
            "depth": 1,
            "num_heads": 1,
            "cond_token_dim": 32,
            "global_cond_dim": 32,
            "azimuth_num_freqs": azimuth_num_freqs,
        }
    )
    return config


def _raw_metadata() -> dict:
    generator = torch.Generator().manual_seed(20260718)
    return {
        "source": torch.tensor([1.20, -0.80, 0.35]),
        "source_vit": torch.tensor([1.20, -0.80, 0.35]),
        "context_poses": torch.tensor([[-0.40, 1.50, -0.20]]),
        "context_poses_vit": torch.tensor([[-0.40, 1.50, -0.20]]),
        "context_audio": torch.randn(1, 1, 192, generator=generator),
        "depth": torch.randn(3, 2, 16, generator=generator),
    }


def _activate_cross_attention(wrapper) -> None:
    """Make the tiny model's cross branch explicitly non-zero and auditable."""

    dim = 32
    attention = wrapper.model.model.transformer.layers[0].cross_attn
    with torch.no_grad():
        attention.to_q.weight.copy_(torch.eye(dim))
        attention.to_kv.weight[:dim].copy_(torch.eye(dim))
        attention.to_kv.weight[dim:].copy_(torch.eye(dim))
        attention.to_out.weight.copy_(torch.eye(dim))


def _condition_c16(wrapper, metadata: dict):
    # RIR content is yaw invariant.  Run the real RIRConditioner once, then run
    # every yaw-dependent real conditioner for every member of the C16 orbit.
    audio = wrapper.conditioner(
        [metadata], "cpu", only_ids=("context_audio",)
    )
    orbit = []
    for angle in C16:
        moved = rotate_scene_metadata(metadata, angle, img_w=16)
        entries = wrapper.conditioner(
            [moved],
            "cpu",
            only_ids=(
                "source",
                "source_vit",
                "context_poses_vit",
                "context_poses",
            ),
        )
        entries.update(audio)
        orbit.append((moved, entries, wrapper.get_conditioning_inputs(entries)))
    return orbit


def _run(wrapper, inputs: dict, *, cfg_scale: float) -> torch.Tensor:
    x = torch.linspace(-0.8, 0.9, 28).reshape(1, 4, 7)
    t = torch.tensor([0.375])
    return wrapper.model(
        x,
        t,
        **inputs,
        cfg_scale=cfg_scale,
        use_checkpointing=False,
    )


def _max_defect(outputs: list[torch.Tensor]) -> float:
    base = outputs[0].float()
    return max(float((output.float() - base).abs().max()) for output in outputs)


def _perturb(inputs: dict, **updates) -> dict:
    result = dict(inputs)
    result.update(updates)
    return result


@pytest.fixture(scope="module")
def real_v0_case():
    torch.manual_seed(20260718)
    wrapper = create_model_from_config(_tiny_v0_config()).eval()
    _activate_cross_attention(wrapper)
    metadata = _raw_metadata()
    with torch.no_grad():
        orbit = _condition_c16(wrapper, metadata)
    return wrapper, metadata, orbit


def test_real_conditioner_assembly_has_fixed_order_alias_and_fp32_phases(
    real_v0_case,
):
    wrapper, _, orbit = real_v0_case
    _, entries, assembled = orbit[0]

    assert type(wrapper.conditioner).__name__ == "MultiConditioner"
    assert type(wrapper.conditioner.conditioners["source"]).__name__ == (
        "YawPoseConditioner"
    )
    assert type(wrapper.conditioner.conditioners["source_vit"]).__name__ == (
        "GeometryConditioner"
    )
    assert type(wrapper.conditioner.conditioners["context_audio"]).__name__ == (
        "RIRConditioner"
    )
    assert type(wrapper.model).__name__ == "DiTWrapper"
    assert wrapper.cross_attn_cond_ids == [
        "source_vit",
        "context_poses_vit",
        "context_poses",
        "context_audio",
    ]

    assert entries["source_vit"][0].shape == (1, 16, 32)
    assert entries["context_poses_vit"][0].shape == (1, 16, 32)
    assert entries["context_poses"][0].shape == (1, 1, 32)
    assert entries["context_audio"][0].shape == (1, 1, 32)
    assert len(entries["context_audio"]) == 2
    assert assembled["cross_attn_cond"].shape == (1, 34, 32)
    assert assembled["cross_attn_mask"].shape == (1, 34)
    assert assembled["cross_attn_phases"].shape == (1, 34)
    assert assembled["query_phase"].shape == (1,)
    assert assembled["cross_attn_phases"].dtype is torch.float32
    assert assembled["query_phase"].dtype is torch.float32

    # Type embeddings are zero at initialization, so exact slices prove the
    # configured stream order rather than merely checking the final length.
    expected = torch.cat(
        [
            entries["source_vit"][0],
            entries["context_poses_vit"][0],
            entries["context_poses"][0],
            entries["context_audio"][0],
        ],
        dim=1,
    )
    torch.testing.assert_close(assembled["cross_attn_cond"], expected)
    torch.testing.assert_close(
        assembled["cross_attn_phases"][:, 32:33],
        entries["context_poses"][2],
    )
    torch.testing.assert_close(
        assembled["cross_attn_phases"][:, 33:34],
        entries["context_poses"][2],
    )


@pytest.mark.parametrize("cfg_scale", [1.0, 2.5])
def test_real_raw_metadata_full_c16_is_invariant_and_nonvacuous(
    real_v0_case, cfg_scale
):
    wrapper, _, orbit = real_v0_case
    cross_norms = []
    hook = wrapper.model.model.transformer.layers[0].cross_attn.register_forward_hook(
        lambda module, args, output: cross_norms.append(float(output.float().norm()))
    )
    with torch.no_grad():
        outputs = [_run(wrapper, inputs, cfg_scale=cfg_scale) for _, _, inputs in orbit]
        identity = _run(wrapper, orbit[0][2], cfg_scale=cfg_scale)
    hook.remove()

    identity_floor = float((identity - outputs[0]).abs().max())
    good_defect = _max_defect(outputs)
    assert identity_floor == 0.0
    assert cross_norms and min(cross_norms) > SENSITIVITY_FLOOR
    assert good_defect <= FP32_C16_ATOL


def test_phase_only_and_broken_pairings_have_pre_registered_large_defects(
    real_v0_case,
):
    wrapper, _, orbit = real_v0_case
    base_inputs = orbit[0][2]
    moved_inputs = orbit[1][2]
    with torch.no_grad():
        base = _run(wrapper, base_inputs, cfg_scale=1.0)
        good_outputs = [_run(wrapper, inputs, cfg_scale=1.0) for _, _, inputs in orbit]

        offsets = torch.linspace(-0.7, 0.8, 34).reshape(1, 34)
        phase_only = _run(
            wrapper,
            _perturb(
                base_inputs,
                cross_attn_phases=base_inputs["cross_attn_phases"] + offsets,
            ),
            cfg_scale=1.0,
        )
        stale_query = _run(
            wrapper,
            _perturb(moved_inputs, query_phase=base_inputs["query_phase"]),
            cfg_scale=1.0,
        )
        wrong_phases = moved_inputs["cross_attn_phases"].clone()
        wrong_phases[:, :16] = torch.roll(wrong_phases[:, :16], shifts=3, dims=1)
        wrong_phases[:, 16:32] = torch.roll(
            wrong_phases[:, 16:32], shifts=-2, dims=1
        )
        wrong_pairing = _run(
            wrapper,
            _perturb(moved_inputs, cross_attn_phases=wrong_phases),
            cfg_scale=1.0,
        )
        wrong_sign_phases = (
            2.0 * base_inputs["query_phase"][:, None]
            - base_inputs["cross_attn_phases"]
        )
        wrong_sign = _run(
            wrapper,
            _perturb(base_inputs, cross_attn_phases=wrong_sign_phases),
            cfg_scale=1.0,
        )
        # The rotated conditioner already emits destination-column phases.
        # Adding the physical yaw again reproduces the double-phase bug.
        double_phase = _run(
            wrapper,
            _perturb(
                moved_inputs,
                cross_attn_phases=moved_inputs["cross_attn_phases"] + C16[1],
            ),
            cfg_scale=1.0,
        )

    good_defect = _max_defect(good_outputs)
    defects = {
        "phase_only": float((phase_only - base).abs().max()),
        "stale_query": float((stale_query - base).abs().max()),
        "wrong_pairing": float((wrong_pairing - base).abs().max()),
        "wrong_sign": float((wrong_sign - base).abs().max()),
        "double_phase": float((double_phase - base).abs().max()),
    }
    for name, defect in defects.items():
        assert defect > SENSITIVITY_FLOOR, (name, defect, good_defect)
        assert defect > BROKEN_TO_GOOD_RATIO * max(good_defect, 1.0e-8), (
            name,
            defect,
            good_defect,
        )


def test_bfloat16_content_keeps_fp32_phase_and_c16_invariance(real_v0_case):
    wrapper, _, orbit = real_v0_case
    wrapper.to(torch.bfloat16)
    try:
        assert all(
            inputs["cross_attn_phases"].dtype is torch.float32
            and inputs["query_phase"].dtype is torch.float32
            for _, _, inputs in orbit
        )
        with torch.no_grad():
            outputs = [_run(wrapper, inputs, cfg_scale=2.5) for _, _, inputs in orbit]
        assert all(output.dtype is torch.bfloat16 for output in outputs)
        assert _max_defect(outputs) <= BF16_C16_ATOL
    finally:
        wrapper.to(torch.float32)


def test_zero_frequency_negative_control_has_cross_path_without_phase_sensitivity(
    real_v0_case,
):
    phase_wrapper, _, orbit = real_v0_case
    torch.manual_seed(20260718)
    control = create_model_from_config(
        _tiny_v0_config(azimuth_num_freqs=0)
    ).eval()
    control.load_state_dict(phase_wrapper.state_dict(), strict=True)
    _activate_cross_attention(control)
    inputs = orbit[0][2]
    cross_norms = []
    hook = control.model.model.transformer.layers[0].cross_attn.register_forward_hook(
        lambda module, args, output: cross_norms.append(float(output.float().norm()))
    )
    with torch.no_grad():
        base = _run(control, inputs, cfg_scale=1.0)
        perturbed = _run(
            control,
            _perturb(
                inputs,
                cross_attn_phases=inputs["cross_attn_phases"]
                + torch.linspace(-1.0, 1.0, 34).reshape(1, 34),
                query_phase=inputs["query_phase"] + 0.9,
            ),
            cfg_scale=1.0,
        )
    hook.remove()

    assert cross_norms and min(cross_norms) > SENSITIVITY_FLOOR
    assert torch.equal(base, perturbed)
