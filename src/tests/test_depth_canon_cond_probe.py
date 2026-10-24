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
