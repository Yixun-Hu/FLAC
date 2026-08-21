"""exp_19: RUNTIME routing contracts for the cylindrical arms (2026-08-21).

Why this file exists: the first CYL HAA run was invalidated because the config
*said* `implementation: cylindrical_dinov3` while `conditioners.py` on this branch
had no cylindrical routing and silently built a vanilla `AutoModel`. Every gate
validated the JSON; nothing validated the constructed object. These tests close
that class of defect: they build the conditioner stack exactly the way train.py
does and assert on the RESULT.

CPU-only; loads the DINOv3 backbone once from the local HF cache (HF_HUB_OFFLINE).
"""

import json
import os
import sys

import pytest
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

CYL_PKG = "/home/yixunhu/codespace/cylindrical-dinov3/src"
if CYL_PKG not in sys.path:                      # PYTHONPATH convention, mirrored
    sys.path.insert(0, CYL_PKG)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPDIR = os.path.join(REPO, "worklog/worklog_yixun/exp_19_haa_finetune_claude")

from src.models.conditioners import create_multi_conditioner_from_conditioning_config  # noqa: E402


def _conditioning(cfg_name):
    cfg = json.load(open(os.path.join(EXPDIR, cfg_name)))
    return cfg["model"]["conditioning"]


def _vit_of(mc, cid="source_vit"):
    return mc.conditioners[cid].vit


# ---------------------------------------------------------------------------------
# THE defect test: the constructed class, not the config, is cylindrical
# ---------------------------------------------------------------------------------
def test_cyl_config_constructs_a_cylindrical_model_not_a_vanilla_one():
    mc = create_multi_conditioner_from_conditioning_config(_conditioning("FLAC_HAA_finetune_CYL.json"))
    vit = _vit_of(mc)
    assert type(vit).__name__ == "CylindricalDINOv3ViTModel", (
        f"CYL config built {type(vit).__name__} -- the vanilla-fallback defect is back")
    assert vit.config.gauge == "cylindrical_xyz"
    # default knobs on the no-SSL arm
    assert getattr(vit.config, "azimuth_mode", "full") == "full"
    assert getattr(vit.config, "prefix_mode", "strip") == "strip"


def test_cylssl_config_carries_the_exp12_arm_b_knobs():
    mc = create_multi_conditioner_from_conditioning_config(_conditioning("FLAC_HAA_finetune_CYLSSL.json"))
    vit = _vit_of(mc)
    assert type(vit).__name__ == "CylindricalDINOv3ViTModel"
    assert vit.config.azimuth_mode == "lowband"
    assert vit.config.prefix_mode == "m0_registers"
    # lowband's harmonic ladder tops out at m=2 -- the architecture, not the label
    assert int(vit.rope_embeddings.azimuth_harmonics.max().item()) == 2


def test_both_conditioners_share_one_backbone():
    mc = create_multi_conditioner_from_conditioning_config(_conditioning("FLAC_HAA_finetune_CYL.json"))
    assert _vit_of(mc, "source_vit") is _vit_of(mc, "context_poses_vit")


def test_pooled_output_is_roll_invariant_smoke():
    """The property the whole cylindrical program rests on, asserted on the real
    construction path: a 1-patch azimuth roll (with the world yawed to match)
    leaves the pooled conditioning invariant. A vanilla backbone fails this."""
    import math
    mc = create_multi_conditioner_from_conditioning_config(_conditioning("FLAC_HAA_finetune_CYL.json"))
    vit = _vit_of(mc).eval()
    torch.manual_seed(0)
    x = torch.randn(1, 3, 256, 512)
    k, W = 16, 512                                # one patch column
    ang = 2 * math.pi * k / W
    r = torch.roll(x, k, dims=-1)
    xr, yr = r[:, 0].clone(), r[:, 1].clone()
    r[:, 0] = xr * math.cos(ang) - yr * math.sin(ang)
    r[:, 1] = xr * math.sin(ang) + yr * math.cos(ang)
    with torch.no_grad():
        a = vit(x).pooler_output
        b = vit(r).pooler_output
    assert torch.max(torch.abs(a - b)).item() < 1e-4, "pooled output is NOT roll-invariant"


def test_unknown_implementation_fails_closed():
    cond = _conditioning("FLAC_HAA_finetune_CYL.json")
    cond = json.loads(json.dumps(cond))
    for c in cond["configs"]:
        if c["type"] == "ViTCoordinates":
            c["config"]["ViT"]["implementation"] = "cylindrical_dinov3_typo"
    with pytest.raises(ValueError, match="unknown ViT implementation"):
        create_multi_conditioner_from_conditioning_config(cond)


def test_mismatched_second_vit_block_fails_closed():
    cond = _conditioning("FLAC_HAA_finetune_CYL.json")
    cond = json.loads(json.dumps(cond))
    blocks = [c for c in cond["configs"] if c["type"] == "ViTCoordinates"]
    blocks[1]["config"]["ViT"]["gauge"] = "none"
    with pytest.raises(ValueError, match="differs from the one that built"):
        create_multi_conditioner_from_conditioning_config(cond)


def test_ssl_ckpt_knob_mismatch_fails_closed(tmp_path):
    blob = {"backbone": {}, "azimuth_mode": "ray12", "prefix_mode": "strip", "step": 1}
    p = tmp_path / "ssl.pt"
    torch.save(blob, p)
    cond = _conditioning("FLAC_HAA_finetune_CYLSSL.json")
    cond = json.loads(json.dumps(cond))
    for c in cond["configs"]:
        if c["type"] == "ViTCoordinates":
            c["config"]["ViT"]["ssl_ckpt"] = str(p)
    with pytest.raises(ValueError, match="pretrained with azimuth_mode='ray12'"):
        create_multi_conditioner_from_conditioning_config(cond)
