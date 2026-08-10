"""exp_12 arm B -- CPU tests. No GPU, no training, no writes outside tmp_path.

Run:  cd worklog/worklog_yixun/exp_12_arms/ssl && \
      PYTHONPATH=/home/yixunhu/codespace/cylindrical-dinov3/src:. python -m pytest tests -q
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ssl_data as D  # noqa: E402
from ssl_losses import DINOLoss, IBOTLoss, cosine_schedule, gram_loss, koleo_loss  # noqa: E402
from ssl_model import SSLModel, ema_update, make_teacher  # noqa: E402

REPO = "/home/yixunhu/codespace/exp-12-arms"
DATA = "/home/yixunhu/codespace/FLAC/AcousticRooms/"
HAS_DATA = os.path.isdir(os.path.join(DATA, "depth_map"))


def _production_points(depth_np):
    """Import AR_md and call the real production converter."""
    spec = importlib.util.spec_from_file_location(
        "AR_md", os.path.join(REPO, "src/configs/dataset_configs/custom_metadata/AR_md.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.convert_equirect_to_camera_coord(torch.from_numpy(depth_np), 256, 512).permute(2, 0, 1)


# ----------------------------------------------------------------------------------------
# 1. the field we pretrain on IS the field production conditions on
# ----------------------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_DATA, reason="AcousticRooms not present")
def test_field_matches_production_converter():
    """Difference is float-association only (~1e-6 m on a ~10 m room), not semantics."""
    p = next(
        os.path.join(r, f)
        for r, _, fs in os.walk(os.path.join(DATA, "depth_map"))
        for f in fs
        if f.endswith(".npy")
    )
    depth = np.load(p)
    mine = D.depth_to_points(torch.from_numpy(depth).float())
    prod = _production_points(depth).float()
    assert mine.shape == prod.shape == (3, 256, 512)
    assert torch.max(torch.abs(mine - prod)).item() < 1e-4

    query = torch.tensor([1.0, -2.0, 0.5])
    field = D.displacement_field(torch.from_numpy(depth).float(), query)
    expect = query.view(3, 1, 1) - prod          # == GeometryConditioner's (coord - depth)/1
    assert torch.max(torch.abs(field - expect)).item() < 1e-4


def test_ray_grid_is_unit_norm_and_seam_consistent():
    r = D.unit_ray_grid(32, 64)
    assert torch.allclose(r.pow(2).sum(0).sqrt(), torch.ones(32, 64), atol=1e-5)
    # Column centres tile the circle exactly, so rolling the grid right by k columns shows
    # the direction k columns earlier: roll(r, k)[:, j] == R_z(-2*pi*k/W) r[:, j]. This is
    # the same yaw relation the backbone is equivariant to (there the world rotates WITH
    # the content, hence the opposite sign in test_roll_is_a_literal_noop).
    k, W = 7, 64
    a = -2 * math.pi * k / W
    x, y = r[0], r[1]
    rot = torch.stack(
        [x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a), r[2]]
    )
    assert torch.max(torch.abs(torch.roll(r, k, dims=-1) - rot)).item() < 1e-5


def test_elevation_flip_is_an_exact_symmetry():
    f = torch.randn(3, 8, 16)
    g = D.elevation_flip(D.elevation_flip(f))
    assert torch.equal(f, g)
    h = D.elevation_flip(f)
    assert torch.equal(h[2], -torch.flip(f[2], dims=(-2,)))
    assert torch.equal(h[0], torch.flip(f[0], dims=(-2,)))


def test_downsample_depth_is_area_average():
    d = torch.arange(16.0).reshape(4, 4)
    out = D.downsample_depth(d, 2)
    assert out.shape == (2, 2)
    assert out[0, 0] == pytest.approx((0 + 1 + 4 + 5) / 4)


# ----------------------------------------------------------------------------------------
# 2. THE LEAK GUARD -- the 17 unseen_eval rooms must never enter pretraining
# ----------------------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_DATA, reason="AcousticRooms not present")
def test_build_index_refuses_a_manifest_containing_held_out_rooms(tmp_path):
    unseen = json.load(open(os.path.join(REPO, "data/AR/unseen_eval.json")))
    scene = sorted(unseen)[0]
    room = sorted(unseen[scene])[0]
    poisoned = {scene: {room: unseen[scene][room][:2]}}
    mpath = tmp_path / "poisoned.json"
    mpath.write_text(json.dumps(poisoned))
    with pytest.raises(ValueError, match="REFUSE"):
        D.build_index(DATA, str(mpath), os.path.join(REPO, "data/AR/unseen_eval.json"))


def test_dataset_refuses_a_forbidden_room_in_its_index():
    idx = [D.RoomIndex("Cafe", "Cafe_idx_1", [0], {0: [0, 0, 0]}, {1: [1, 0, 0]}, [(1, 0)])]
    with pytest.raises(ValueError, match="REFUSE"):
        D.RoomViewDataset(idx, DATA, forbidden_rooms={("Cafe", "Cafe_idx_1")})


@pytest.mark.skipif(not HAS_DATA, reason="AcousticRooms not present")
def test_train_and_unseen_room_ids_are_disjoint():
    tr = json.load(open(os.path.join(REPO, "data/AR/train.json")))
    ue = json.load(open(os.path.join(REPO, "data/AR/unseen_eval.json")))
    a = {(s, r) for s in tr for r in tr[s]}
    b = {(s, r) for s in ue for r in ue[s]}
    assert len(a) == 243 and len(b) == 17 and not (a & b)


def test_parse_pair_matches_ar_md_spelling():
    assert D._parse_pair("S001_R0044_hybrid_IR.wav") == (1, 44)
    assert D._parse_pair("S0070_R0075_hybrid_IR.wav") == (70, 75)


# ----------------------------------------------------------------------------------------
# 3. masking
# ----------------------------------------------------------------------------------------
def test_block_mask_respects_ratio_and_can_wrap_the_seam():
    rng = random.Random(0)
    seen_wrap = False
    for _ in range(60):
        m = D.block_mask(16, 32, (0.1, 0.5), rng)
        frac = m.float().mean().item()
        assert 0.0 <= frac <= 0.75
        if m[:, 0].any() and m[:, -1].any():
            seen_wrap = True
    assert seen_wrap, "masks never wrapped the azimuth seam"


def test_block_mask_zero_ratio_is_empty():
    assert not D.block_mask(4, 8, (0.0, 0.0), random.Random(1)).any()


# ----------------------------------------------------------------------------------------
# 4. losses
# ----------------------------------------------------------------------------------------
def test_dino_loss_skips_self_pairs_and_center_moves():
    loss = DINOLoss(16, center_momentum=0.5)
    a, b = torch.randn(4, 16), torch.randn(4, 16)
    single = loss([a], [a], 0.04)
    assert float(single) == 0.0, "a single view pair must contribute nothing (i == j skipped)"
    assert float(loss([a, b], [a, b], 0.04)) > 0
    before = loss.center.clone()
    loss.update_center([torch.ones(4, 16)])
    assert not torch.allclose(before, loss.center)
    assert torch.allclose(loss.center, 0.5 * before + 0.5 * torch.ones(1, 16))


def test_ibot_loss_is_zero_on_empty_masks_and_positive_otherwise():
    loss = IBOTLoss(8)
    assert float(loss(torch.zeros(0, 8), torch.zeros(0, 8), 0.04)) == 0.0
    s, t = torch.randn(5, 8), torch.randn(5, 8)
    assert float(loss(s, t, 0.04)) > 0


def test_gram_loss_zero_iff_same_geometry_and_rotation_invariant():
    x = torch.randn(2, 10, 6)
    assert float(gram_loss(x, x)) == pytest.approx(0.0, abs=1e-6)
    q, _ = torch.linalg.qr(torch.randn(6, 6))
    assert float(gram_loss(x, x @ q)) == pytest.approx(0.0, abs=1e-5)
    assert float(gram_loss(x, torch.randn(2, 10, 6))) > 1e-3


def test_koleo_penalises_collapse():
    spread = torch.randn(8, 16)
    collapsed = torch.randn(1, 16).repeat(8, 1) + 1e-4 * torch.randn(8, 16)
    assert float(koleo_loss(collapsed)) > float(koleo_loss(spread))


def test_cosine_schedule_endpoints_and_warmup():
    assert cosine_schedule(0, 100, 1.0, 0.0, warmup=10) == pytest.approx(0.0)
    assert cosine_schedule(10, 100, 1.0, 0.0, warmup=10) == pytest.approx(1.0)
    assert cosine_schedule(100, 100, 1.0, 0.0, warmup=10) == pytest.approx(0.0, abs=1e-6)
    assert cosine_schedule(0, 100, 0.994, 1.0) == pytest.approx(0.994)


# ----------------------------------------------------------------------------------------
# 5. model plumbing on a tiny backbone (CPU)
# ----------------------------------------------------------------------------------------
def _tiny(azimuth_mode="lowband", prefix_mode="m0_registers"):
    from cylindrical_dinov3 import CylindricalDINOv3ViTConfig, CylindricalDINOv3ViTModel

    cfg = CylindricalDINOv3ViTConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        patch_size=16,
        num_register_tokens=4,
        gauge="cylindrical_xyz",
        azimuth_mode=azimuth_mode,
        prefix_mode=prefix_mode,
    )
    torch.manual_seed(0)
    return CylindricalDINOv3ViTModel(cfg)


def test_masking_changes_only_masked_positions_paths():
    m = _tiny()
    x = torch.randn(2, 3, 64, 128)
    n = (64 // 16) * (128 // 16)
    mask = torch.zeros(2, n, dtype=torch.bool)
    mask[:, :5] = True
    a = m(x).last_hidden_state
    b = m(x, bool_masked_pos=mask).last_hidden_state
    assert a.shape == b.shape == (2, n, 64)
    assert not torch.allclose(a, b), "bool_masked_pos had no effect"


def test_roll_is_a_literal_noop_for_the_global_objective():
    """Why arm B does NOT use roll augmentation: the backbone is exactly roll-equivariant,
    so a yawed view yields a bit-comparable patch mean and zero DINO gradient."""
    m = _tiny().eval()
    x = torch.randn(2, 3, 64, 128)
    k = 3                                        # patch columns
    shift, W = k * 16, 128
    ang = 2 * math.pi * shift / W
    rolled = torch.roll(x, shift, dims=-1)
    xr, yr = rolled[:, 0].clone(), rolled[:, 1].clone()
    rolled[:, 0] = xr * math.cos(ang) - yr * math.sin(ang)
    rolled[:, 1] = xr * math.sin(ang) + yr * math.cos(ang)
    with torch.no_grad():
        p0 = m(x).pooler_output
        p1 = m(rolled).pooler_output
    assert torch.max(torch.abs(p0 - p1)).item() < 1e-4


def test_ema_update_moves_teacher_towards_student():
    s = SSLModel(_tiny(), out_dim=32, ibot_out_dim=16)
    t = make_teacher(s)
    with torch.no_grad():
        for p in s.parameters():
            p.add_(1.0)
    before = next(t.parameters()).clone()
    ema_update(t, s, 0.5)
    after = next(t.parameters())
    assert torch.allclose(after, 0.5 * before + 0.5 * (before + 1.0))
    assert all(not p.requires_grad for p in t.parameters())


def test_dino_head_prototype_norms_are_fixed_at_one():
    s = SSLModel(_tiny(), out_dim=32, ibot_out_dim=16)
    g = s.dino_head.last_layer.parametrizations.weight.original0
    assert torch.allclose(g, torch.ones_like(g))
    assert not g.requires_grad
    y = s.dino_head(torch.randn(3, 64))
    assert y.shape == (3, 32) and torch.isfinite(y).all()


def test_two_training_steps_run_end_to_end_on_cpu():
    """The shape-and-plumbing smoke: two real optimiser steps of the full objective."""
    torch.manual_seed(0)
    student = SSLModel(_tiny(), out_dim=32, ibot_out_dim=16)
    teacher = make_teacher(student)
    dino, ibot = DINOLoss(32), IBOTLoss(16)
    opt = torch.optim.AdamW(student.parameters(), lr=1e-4)
    B, n = 2, (64 // 16) * (128 // 16)
    for _ in range(2):
        g = torch.randn(B, 2, 3, 64, 128)
        loc = torch.randn(B, 2, 3, 32, 64)
        mask = torch.zeros(B, 2, n, dtype=torch.bool)
        mask[:, :, :6] = True
        with torch.no_grad():
            t_out = [teacher(g[:, i]) for i in range(2)]
            t_glob = [teacher.dino_head(o["pooled"]) for o in t_out]
        s_out = [student(g[:, i], bool_masked_pos=mask[:, i]) for i in range(2)]
        s_glob = [student.dino_head(o["pooled"]) for o in s_out]
        for i in range(loc.shape[1]):
            s_glob.append(student.dino_head(student(loc[:, i])["pooled"]))
        sp = torch.cat([student.ibot_head(s_out[i]["patch"][mask[:, i]]) for i in range(2)])
        tp = torch.cat([teacher.ibot_head(t_out[i]["patch"][mask[:, i]]) for i in range(2)])
        loss = (
            dino(s_glob, t_glob, 0.04)
            + ibot(sp, tp, 0.04)
            + gram_loss(s_out[0]["patch"], t_out[0]["patch"])
            + 0.1 * koleo_loss(s_out[0]["pooled"])
        )
        assert torch.isfinite(loss)
        opt.zero_grad()
        loss.backward()
        grads = [p.grad for p in student.backbone.parameters() if p.grad is not None]
        assert grads and any(g_.abs().sum() > 0 for g_ in grads), "no gradient reached the backbone"
        opt.step()
        ema_update(teacher, student, 0.9)
        dino.update_center(t_glob)
        ibot.update_center(tp)


def test_local_views_are_a_valid_input_size():
    m = _tiny()
    out = m(torch.randn(1, 3, 32, 64))
    assert out.last_hidden_state.shape == (1, (32 // 16) * (64 // 16), 64)


@pytest.mark.skipif(not HAS_DATA, reason="AcousticRooms not present")
def test_dataset_item_shapes_on_one_real_room():
    tr = json.load(open(os.path.join(REPO, "data/AR/train.json")))
    scene = sorted(tr)[0]
    room = sorted(tr[scene])[0]
    small = {scene: {room: tr[scene][room]}}
    tmp = "/tmp/_exp12b_one_room.json"
    json.dump(small, open(tmp, "w"))
    idx = D.build_index(DATA, tmp, os.path.join(REPO, "data/AR/unseen_eval.json"))
    assert len(idx) == 1 and idx[0].pairs
    ds = D.RoomViewDataset(idx, DATA, n_local=3)
    item = ds[0]
    assert item["globals"].shape == (2, 3, 256, 512)
    assert item["locals"].shape == (3, 3, 128, 256)
    assert item["masks"].shape == (2, 16 * 32) and item["masks"].dtype == torch.bool
    assert torch.isfinite(item["globals"]).all() and torch.isfinite(item["locals"]).all()
    os.remove(tmp)
