"""exp_25 depth_canon round 3b -- the numerical-composition control and eps_cond
(``worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_cond_probe.py``,
plan_depth_canon.md v8 §1 Tier 2 + eps_cond, §6 test 21b).

Written by the exp_25 Coder seat (Claude Opus, max effort) per plan_depth_canon.md v8 §1/§6/§8.

CPU only. The pure helpers run on synthetic tensors; the stage runner drives a real tiny
``diffusion_cond`` wrapper built by the repo factories (``test_yaw_aug_training``'s
forward-consistent toy config); the loader is checked against ``eval_FLAC.evaluate_model``
on one toy checkpoint.
"""
import importlib.util
import os
import sys

import numpy as np
import pytest
import torch

from src.tools import depth_canon_stats as stats

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PROBE_PY = os.path.join(_REPO_ROOT, "worklog", "worklog_yixun", "exp_25_depth_canon_claude",
                         "depth_canon_cond_probe.py")


def _load_probe():
    spec = importlib.util.spec_from_file_location("depth_canon_cond_probe", _PROBE_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_probe()


# --------------------------------------------------------------------------- #
# offsets and fixed noise: deterministic from the seed
# --------------------------------------------------------------------------- #
def test_offsets_are_the_registered_set_and_deterministic_from_the_seed():
    a, b, c = (probe.draw_offsets(64, 512, s) for s in (42, 42, 43))
    assert list(a) == list(probe.OFFSET_KEYS) == ["d64", "d128", "d256", "d384", "random"]
    for key, d in zip(probe.OFFSET_KEYS[:4], (64, 128, 256, 384)):
        assert a[key] == [d] * 64
    assert a == b and a["random"] != c["random"]
    assert all(isinstance(d, int) and 1 <= d <= 511 for d in a["random"])


def test_random_offsets_cover_exactly_one_to_w_minus_one():
    assert set(probe.draw_offsets(20000, 512, 7)["random"]) == set(range(1, 512))


@pytest.mark.parametrize("bad", [dict(n_items=0), dict(img_w=384), dict(img_w=True), dict(n_items=2.0),
                                 dict(seed=-1)])
def test_draw_offsets_refuses_bad_arguments(bad):
    kw = dict(n_items=4, img_w=512, seed=1)
    kw.update(bad)
    with pytest.raises(ValueError):
        probe.draw_offsets(**kw)


def test_noise_is_one_seeded_tensor_per_item_independent_of_the_item_count():
    five = probe.noise_bank(5, (4, 16), seed=42)
    assert five.shape == (5, 4, 16) and five.dtype == torch.float32
    assert torch.equal(probe.noise_bank(3, (4, 16), seed=42), five[:3])
    assert torch.equal(probe.noise_bank(5, (4, 16), seed=42), five)
    assert not torch.equal(probe.noise_bank(5, (4, 16), seed=43), five)
    assert len({tuple(r.flatten().tolist()[:4]) for r in five}) == 5


# --------------------------------------------------------------------------- #
# stagewise comparison (pinned denominators via depth_canon_stats)
# --------------------------------------------------------------------------- #
def test_identical_tensors_compare_to_exactly_zero():
    a = torch.randn(4, 3, 8, generator=torch.Generator().manual_seed(0))
    r = probe.stage_rel(a, a.clone())
    assert r["relL2"].tolist() == [0.0] * 4 and r["relLinf"].tolist() == [0.0] * 4


def test_rel_uses_the_second_argument_as_reference_with_the_floor():
    a, b = torch.ones(2, 3), torch.zeros(2, 3)
    r = probe.stage_rel(a, b)
    assert r["relL2"] == pytest.approx([3 ** 0.5 / stats.REL_FLOOR] * 2)
    assert r["relLinf"] == pytest.approx([1.0 / stats.REL_FLOOR] * 2)
    r2 = probe.stage_rel(torch.tensor([[1.0, 1.0]]), torch.tensor([[2.0, 1.0]]))
    assert r2["relL2"] == pytest.approx([1 / 5 ** 0.5])


def test_rel_is_scale_invariant_and_per_item():
    g = torch.Generator().manual_seed(1)
    a, b = (torch.randn(3, 2, 5, generator=g, dtype=torch.float64) for _ in range(2))
    r, rk = probe.stage_rel(a, b), probe.stage_rel(3.7 * a, 3.7 * b)   # float64: no rescale rounding
    assert np.allclose(r["relL2"], rk["relL2"], rtol=1e-12) and r["relL2"].shape == (3,)
    want = stats.rel_l2(a.reshape(3, -1).double().numpy(), b.reshape(3, -1).double().numpy())
    assert np.array_equal(r["relL2"], want)


def test_bf16_stages_are_compared_in_float64():
    a = torch.randn(2, 6).to(torch.bfloat16)
    r = probe.stage_rel(a, a.float().double() + 1.0)
    assert r["relL2"].dtype == np.float64 and r["relL2"].shape == (2,)


def test_conditioning_comparison_per_id_masks_and_all():
    g = torch.Generator().manual_seed(2)
    x, y = torch.randn(3, 4, 8, generator=g), torch.randn(3, 1, 8, generator=g)
    m = torch.ones(3, 4, dtype=torch.bool)
    a = {"ctx": [x, m], "src": [y, torch.ones(3, 1)]}
    same = probe.compare_cond(a, {"ctx": [x.clone(), m.clone()], "src": [y.clone(), torch.ones(3, 1)]})
    assert same["masks_equal"] is True and same["ids"] == ["ctx", "src"]
    assert all(v.tolist() == [0.0] * 3 for k in ("ctx", "src", "all") for v in same[k].values())
    m2 = m.clone()
    m2[1, 2] = False
    y2 = y.clone()
    y2[2] += 1.0
    diff = probe.compare_cond(a, {"ctx": [x, m2], "src": [y2, torch.ones(3, 1)]})
    assert diff["masks_equal"] is False and diff["ctx"]["relL2"].tolist() == [0.0] * 3
    assert diff["src"]["relL2"][2] > 0 and diff["all"]["relL2"][2] > 0
    flat = lambda c: torch.cat([c["ctx"][0].reshape(3, -1), c["src"][0].reshape(3, -1)], 1)  # noqa: E731
    assert np.array_equal(diff["all"]["relL2"], probe.stage_rel(     # (test, reference)
        flat(a), flat({"ctx": [x], "src": [y2]}))["relL2"])
    with pytest.raises(ValueError, match="conditioner ids"):
        probe.compare_cond(a, {"ctx": [x, m]})


def test_summaries_are_depth_canon_stats_paired_summary():
    v = np.array([0.0, 1e-7, 3e-6, 2e-5])
    assert probe.summarize({"relL2": v, "relLinf": 2 * v}) == {
        "relL2": stats.paired_summary(v), "relLinf": stats.paired_summary(2 * v)}


# --------------------------------------------------------------------------- #
# A = canon(md), B = canon(rot_d(md)) and the Tier-1 record of the probe items
# --------------------------------------------------------------------------- #
from src.data import yaw_rotation as yr  # noqa: E402
from test_depth_canon_eval import CLASSES, W16, _eval_md  # noqa: E402  (16 x 32 AR clouds)


def _items(n=5):
    return [_eval_md(i, CLASSES[i % len(CLASSES)]) for i in range(n)]


def test_canonical_pairs_are_the_operator_on_md_and_on_its_rotation():
    mds = _items()
    offsets = {"d3": [3] * 5, "d16": [16] * 5, "random": [1, 7, 31, 9, 22]}
    a_list, b_lists, tier1 = probe.canonical_pairs(mds, offsets, W16)
    assert list(b_lists) == list(offsets) and len(a_list) == len(tier1) == 5
    for i, md in enumerate(mds):
        ref, dj, tie_class = yr.canonicalize_scene_metadata(md, W16)
        assert all(torch.equal(a_list[i][k], ref[k]) for k in ("depth",) + yr.POSE_KEYS)
        assert tier1[i]["dj_A"] == dj and tier1[i]["class_A"] == tie_class
        assert tier1[i]["depth_sha_A"] == yr.canonical_depth_sha256(ref["depth"])
        for key, ds in offsets.items():
            rot = yr.rotate_scene_metadata(md, yr.offsets_to_radians([ds[i]], W16)[0], W16)
            want, dj_b, cls_b = yr.canonicalize_scene_metadata(rot, W16)
            assert all(torch.equal(b_lists[key][i][k], want[k]) for k in ("depth",) + yr.POSE_KEYS)
            assert tier1[i]["offsets"][key] == {"d": ds[i], "dj_B": dj_b, "class_B": cls_b,
                                                "depth_sha_equal": True, "class_equal": True,
                                                "dj_corotates": dj_b == (dj - ds[i]) % W16}
            assert tier1[i]["offsets"][key]["dj_corotates"] is True
    assert all(md["context_audio"] is a["context_audio"] for md, a in zip(mds, a_list))


def test_tier1_records_a_mismatch_it_did_not_create(monkeypatch):
    real = yr.canonical_depth_sha256
    calls = []

    def lying(depth):
        calls.append(1)
        return real(depth) if len(calls) % 2 else "0" * 64
    monkeypatch.setattr(yr, "canonical_depth_sha256", lying)
    _, _, tier1 = probe.canonical_pairs(_items(1), {"d3": [3]}, W16)
    assert tier1[0]["offsets"]["d3"]["depth_sha_equal"] is False


def test_canonical_pairs_refuses_misaligned_offsets_and_periodic_items():
    with pytest.raises(ValueError, match="one offset per item"):
        probe.canonical_pairs(_items(2), {"d3": [3]}, W16)
    with pytest.raises(yr.StabilizedTieError):
        probe.canonical_pairs([_eval_md(0, period=16)], {"d3": [3]}, W16)


# --------------------------------------------------------------------------- #
# the stage runner on a REAL tiny diffusion_cond wrapper (repo factories, CPU)
# --------------------------------------------------------------------------- #
from test_yaw_aug_training import _base_config, no_flash  # noqa: E402,F401  (fixture)


def _toy_module(seed=0):
    from src.models.factory import create_model_from_config
    from src.training.factory import create_training_wrapper_from_config
    torch.manual_seed(seed)
    cfg = _base_config()
    module = create_training_wrapper_from_config(cfg, create_model_from_config(cfg))
    return module.eval().requires_grad_(False)


def _toy_inputs(n=3):
    a_list, b_lists, _ = probe.canonical_pairs(_items(n), {"d5": [5] * n}, W16)
    return a_list, b_lists["d5"], probe.noise_bank(n, (4, 64), seed=42), torch.zeros(n, 1, 64)


def test_run_stages_is_one_eval_euler_step_with_the_given_noise(no_flash):
    module = _toy_module()
    a_list, _, noise, reals = _toy_inputs()
    keep = noise.clone()
    out = probe.run_stages(module, a_list, noise, reals, probe.PROTOCOLS["fp32"])
    assert torch.equal(noise, keep)                                       # not mutated
    with torch.no_grad():
        cond = module.diffusion.conditioner(a_list, module.device)
        v = module.diffusion.model(noise, torch.ones(3), **module.diffusion.get_conditioning_inputs(cond),
                                   cfg_scale=1.0, batch_cfg=True)
    assert sorted(out["cond"]) == sorted(cond)
    assert all(torch.equal(out["cond"][k][0], cond[k][0]) for k in cond)
    assert torch.equal(out["velocity"], v) and torch.equal(out["latent"], noise - v)
    assert torch.equal(out["waveform"], out["latent"].clamp(-1.0, 1.0))  # no pretransform here


def test_run_stages_hands_the_sampler_exactly_the_given_noise(no_flash, monkeypatch):
    import src.inference.sampling as sampling
    module, seen = _toy_module(), []
    real = sampling.sample_discrete_euler
    monkeypatch.setattr(sampling, "sample_discrete_euler",
                        lambda model, x, steps, **kw: seen.append((x, steps, kw)) or real(model, x, steps, **kw))
    a_list, b_list, noise, reals = _toy_inputs()
    probe.run_stages(module, a_list, noise, reals, probe.PROTOCOLS["eval"])
    probe.run_stages(module, b_list, noise, reals, probe.PROTOCOLS["eval"])
    assert [torch.equal(x, noise) for x, _, _ in seen] == [True, True]
    assert all(s == 1 and kw["cfg_scale"] == 1.0 and kw["batch_cfg"] is True and kw["disable_tqdm"]
               and kw["dist_shift"] is module.diffusion.dist_shift for _, s, kw in seen)


@pytest.mark.parametrize("name, autocast, dtype, precision", [
    ("fp32", False, None, "highest"), ("eval", True, torch.bfloat16, "medium")])
def test_protocols_set_cond_autocast_and_matmul_precision(no_flash, name, autocast, dtype, precision):
    module, seen = _toy_module(), []
    hook = module.diffusion.conditioner.register_forward_pre_hook(lambda m, a: seen.append(
        (torch.is_autocast_enabled("cpu"), torch.get_autocast_dtype("cpu"), torch.get_float32_matmul_precision())))
    before = torch.get_float32_matmul_precision()
    a_list, _, noise, reals = _toy_inputs()
    probe.run_stages(module, a_list, noise, reals, probe.PROTOCOLS[name])
    hook.remove()
    assert seen[0][0] is autocast and seen[0][2] == precision and (not autocast or seen[0][1] == dtype)
    assert torch.get_float32_matmul_precision() == before and not torch.is_autocast_enabled("cpu")


def test_identical_inputs_give_exactly_zero_at_every_stage(no_flash):
    module = _toy_module()
    a_list, b_list, noise, reals = _toy_inputs()
    s1, s2 = (probe.run_stages(module, a_list, noise, reals, probe.PROTOCOLS["fp32"]) for _ in range(2))
    cmp = probe.compare_stages(s1, s2)
    assert list(cmp) == list(probe.STAGES)
    assert cmp["cond"]["all"]["relL2"].max() == 0.0 and cmp["cond"]["masks_equal"] is True
    assert all(cmp[s]["relL2"].max() == 0.0 for s in probe.STAGES[1:])
    rot = probe.compare_stages(probe.run_stages(module, b_list, noise, reals, probe.PROTOCOLS["fp32"]), s1)
    assert rot["cond"]["all"]["relL2"].max() < 1e-5   # float32 pose composition only


def test_a_non_rectified_flow_objective_is_refused(no_flash):
    module = _toy_module()
    module.diffusion.model.diffusion_objective = "v"
    a_list, _, noise, reals = _toy_inputs()
    with pytest.raises(ValueError, match="rectified_flow"):
        probe.run_stages(module, a_list, noise, reals, probe.PROTOCOLS["eval"])
