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
    assert set(probe.draw_offsets(4000, 8, 7)["random"]) == set(range(1, 8))


@pytest.mark.parametrize("bad", [dict(n_items=0), dict(img_w=1), dict(img_w=True), dict(n_items=2.0)])
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
