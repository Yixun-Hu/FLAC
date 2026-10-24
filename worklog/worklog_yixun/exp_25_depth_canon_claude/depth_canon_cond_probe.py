#!/usr/bin/env python3
"""exp_25 depth_canon -- the numerical-composition control and eps_cond
(plan_depth_canon.md v8 §1 Tier 2 (ii) + eps_cond, §6 test 21b).

For the first ``--n-items`` items of an evaluation loader and each registered column
offset d in {64, 128, 256, 384, random in [1, W-1]}, A = canon(md) and
B = canon(rot_d(md)) are pushed through the SAME loaded checkpoint with the SAME fixed
per-item noise, and compared stage by stage: conditioning tensors, the one-step DiT
velocity, the latent after the single Euler step, the decoded waveform after
clamp(-1, 1). relL2 / relLinf use ``src/tools/depth_canon_stats`` (pinned
denominators, A is the reference). Tier 1 (canonical recovered-depth sha256 of A == B)
is recorded for the same items. eps_cond = max fp32 conditioning relL2; PASS <= 1e-5.

GPU for the registered run (the Planner launches it); CPU-testable on a toy stack.

Written by the exp_25 Coder seat (Claude Opus, max effort) per plan_depth_canon.md v8 §1/§6/§8.
"""
import os
import sys

import numpy as np
import torch


def _repo_root(p):
    p = os.path.abspath(p)
    while not os.path.isdir(os.path.join(p, ".git")) and not os.path.isfile(os.path.join(p, ".git")):
        parent = os.path.dirname(p)
        if parent == p:
            raise RuntimeError("repo root (.git) not found")
        p = parent
    return p


REPO = _repo_root(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)  # this checkout's src/ and eval_FLAC.py, never a stale installed copy

from src.tools import depth_canon_stats as stats  # noqa: E402

FIXED_OFFSETS = (64, 128, 256, 384)
OFFSET_KEYS = tuple(f"d{d}" for d in FIXED_OFFSETS) + ("random",)
RANDOM_OFFSET_SEED_SHIFT = 1  # the random offsets' generator is seeded with seed + 1


# --------------------------------------------------------------------------- #
# offsets and fixed noise
# --------------------------------------------------------------------------- #
def _check_int(name, v, lo):
    if isinstance(v, bool) or not isinstance(v, int) or v < lo:
        raise ValueError(f"{name} must be an int >= {lo}, got {v!r}")


def draw_offsets(n_items, img_w, seed):
    """``{key: [d_i per item]}``: the four fixed column offsets and one independent
    uniform draw in ``[1, img_w - 1]`` per item from a generator seeded with
    ``seed + RANDOM_OFFSET_SEED_SHIFT``."""
    _check_int("n_items", n_items, 1)
    _check_int("img_w", img_w, 2)
    _check_int("seed", seed, 0)
    out = {f"d{d}": [d % img_w] * n_items for d in FIXED_OFFSETS}
    g = torch.Generator().manual_seed(seed + RANDOM_OFFSET_SEED_SHIFT)
    out["random"] = [int(d) for d in torch.randint(1, img_w, (n_items,), generator=g)]
    return out


def item_noise(i, shape, seed):
    """Item ``i``'s fixed noise: its own generator, seeded from ``(seed, i)`` through
    numpy's SeedSequence, so it does not depend on how many items run."""
    s = int(np.random.SeedSequence([int(seed), int(i)]).generate_state(1, dtype=np.uint64)[0])
    return torch.randn(tuple(shape), generator=torch.Generator().manual_seed(s & (2 ** 63 - 1)))


def noise_bank(n_items, shape, seed):
    """``[n_items, *shape]`` float32: one seeded noise tensor per item, reused for A and B."""
    return torch.stack([item_noise(i, shape, seed) for i in range(n_items)])


# --------------------------------------------------------------------------- #
# stagewise comparison
# --------------------------------------------------------------------------- #
def _rows(t):
    """``[B, ...]`` tensor -> float64 CPU ``[B, prod(rest)]`` numpy (bf16 included)."""
    t = t.detach().to(device="cpu", dtype=torch.float64)
    return t.reshape(t.shape[0], -1).numpy()


def stage_rel(test, ref):
    """Per-item ``{"relL2", "relLinf"}`` of ``test`` against the reference ``ref``."""
    a, b = _rows(test), _rows(ref)
    return {"relL2": stats.rel_l2(a, b), "relLinf": stats.rel_linf(a, b)}


def compare_cond(test, ref):
    """Conditioning dicts ``{id: [tensor, mask]}``: per-id relL2 / relLinf of the
    tensors, the same over all ids concatenated per item (``"all"``, ids sorted), and
    whether every mask is identical."""
    if sorted(test) != sorted(ref):
        raise ValueError(f"conditioner ids differ: {sorted(test)} vs {sorted(ref)}")
    ids = sorted(ref)
    out = {"ids": ids, "masks_equal": all(
        (test[k][1] is None and ref[k][1] is None) or
        (test[k][1] is not None and ref[k][1] is not None and torch.equal(test[k][1].cpu(), ref[k][1].cpu()))
        for k in ids)}
    for k in ids:
        out[k] = stage_rel(test[k][0], ref[k][0])
    cat = lambda c: torch.cat([c[k][0].detach().double().cpu().reshape(c[k][0].shape[0], -1)  # noqa: E731
                               for k in ids], dim=1)
    out["all"] = stage_rel(cat(test), cat(ref))
    return out


def summarize(rel):
    """``{"relL2": paired_summary, "relLinf": paired_summary}`` over the items."""
    return {k: stats.paired_summary(rel[k]) for k in ("relL2", "relLinf")}
