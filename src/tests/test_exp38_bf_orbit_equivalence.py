"""exp_38 B4a — at cap == per-rank batch, the batched C4 orbit IS the legacy per-angle loop that
trained B-F 100 %, in TRAINING mode (plan_data_efficiency.md r2 §5, §10 B4a; announcement 06).

Two independent, identical copies of the real B-F conditioner (DINOv3-S from the pinned HF cache,
train mode, the config's non-reentrant ViT checkpointing, train-mode RoPE rescale 2.0) are run
under the same seeded RNG: copy A through ``_orbit_average_loop`` (the reference loop kept in
``yaw_rotation.py``), copy B through ``_orbit_average_batched(cap=B)`` — the training path with
``training.frame_avg_max_fwd_samples = B``. Outputs, masks, post-forward and post-backward RNG
states, the loss and every parameter gradient must be bit-identical, and every checkpoint call
non-reentrant. CPU fp32; skipped when the DINOv3 snapshot is not cached.
"""
import json
import math
import os

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CFG = os.path.join(REPO, "worklog/worklog_yixun/exp_38_data_efficiency_claude/FLAC_AR_BF_cap32.json")


def _hf_cached():
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception:
        return False
    return os.path.isdir(os.path.join(HF_HUB_CACHE, "models--facebook--dinov3-vits16-pretrain-lvd1689m"))


pytestmark = pytest.mark.skipif(not _hf_cached(), reason="DINOv3 snapshot not cached")


def _metadata(batch, n_ctx, gen):
    out = []
    for _ in range(batch):
        depth = torch.rand(3, 256, 512, generator=gen) * 4.0 - 2.0
        src = torch.rand(3, generator=gen) * 2.0 - 1.0
        ctx = torch.rand(n_ctx, 3, generator=gen) * 2.0 - 1.0
        out.append({"depth": depth, "source": src, "source_vit": src.unsqueeze(0), "context_poses": ctx,
                    "context_poses_vit": ctx, "context_audio": torch.randn(n_ctx, 1, 9600, generator=gen) * 0.01})
    return out


def _run(cond, md, orbit, batch):
    from src.data import yaw_rotation as yr
    torch.manual_seed(1234)
    md_inv = [yr.cylindrical_pose_features(m) for m in md]
    base = cond(md_inv, "cpu")
    present = [i for i in ("source_vit", "context_poses_vit") if i in base]
    angles = (0.0, 90.0, 180.0, 270.0)
    if orbit == "loop":
        acc = yr._orbit_average_loop(cond, md_inv, base, present, angles, 512, "cpu")
    else:
        acc = yr._orbit_average_batched(cond, md_inv, base, present, angles, 512, "cpu",
                                        cap=batch, cap_is_explicit=True)
    for i in present:
        base[i][0] = acc[i] / float(len(angles))
    rng_fwd = torch.get_rng_state().clone()
    loss = sum((v[0].float() ** 2).mean() for v in base.values())
    loss.backward()
    return base, loss.detach(), rng_fwd, torch.get_rng_state().clone()


def test_loop_and_cap_equal_batch_are_bit_identical_in_training_mode(monkeypatch):
    from src.models.conditioners import create_multi_conditioner_from_conditioning_config
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    cfg = json.load(open(CFG))
    assert cfg["training"]["frame_avg_max_fwd_samples"] == 32
    # Two independent builds under the same seed (NOT deepcopy: the ViT checkpoint wrapper is an
    # instance-level closure over its own layer, which deepcopy would share between the copies).
    torch.manual_seed(0)
    cond_a = create_multi_conditioner_from_conditioning_config(cfg["model"]["conditioning"])
    torch.manual_seed(0)
    cond_b = create_multi_conditioner_from_conditioning_config(cfg["model"]["conditioning"])
    sa, sb = cond_a.state_dict(), cond_b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    assert not set(map(id, cond_a.parameters())) & set(map(id, cond_b.parameters()))
    cond_a.train(); cond_b.train()
    calls = []
    real_ckpt = torch.utils.checkpoint.checkpoint

    def spy(fn, *a, **k):
        calls.append(k.get("use_reentrant", "UNSET"))
        return real_ckpt(fn, *a, **k)
    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", spy)
    batch, n_ctx = 2, 2
    md = _metadata(batch, n_ctx, torch.Generator().manual_seed(7))
    out_a, loss_a, fwd_a, bwd_a = _run(cond_a, md, "loop", batch)
    n_calls_loop = len(calls)
    out_b, loss_b, fwd_b, bwd_b = _run(cond_b, md, "batched", batch)
    assert n_calls_loop > 0 and len(calls) == 2 * n_calls_loop and set(calls) == {False}
    assert set(out_a) == set(out_b)
    for key in out_a:
        assert torch.equal(out_a[key][0], out_b[key][0]), key
        ma, mb = out_a[key][1], out_b[key][1]
        assert (ma is None and mb is None) or torch.equal(ma, mb), key
    assert torch.equal(loss_a, loss_b) and math.isfinite(float(loss_a))
    assert torch.equal(fwd_a, fwd_b) and torch.equal(bwd_a, bwd_b)
    pa, pb = dict(cond_a.named_parameters()), dict(cond_b.named_parameters())
    n_grads = 0
    for name, p in pa.items():
        ga, gb = p.grad, pb[name].grad
        assert (ga is None) == (gb is None), name
        if ga is not None:
            assert torch.equal(ga, gb), name
            n_grads += 1
    assert n_grads > 100
