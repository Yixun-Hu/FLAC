#!/usr/bin/env python3
"""exp_25 depth_canon -- the PRE-FULL tie audit (plan_depth_canon.md v8 §7 rung 3).

Every depth panorama referenced by the train / unseen / seen AR splits is run through
the canonicalization selector twice: the treatment path (AR_md's own point cloud ->
``canonicalize_scene_metadata``) and an independent raw path on the scalar ``.npy``
(numpy, written here). Gate: PASS iff every map was processed and no map is an
image-key survivor (exactly periodic -> ``StabilizedTieError``). Read-only on the data.

Written by the exp_25 Coder seat (Claude Opus, max effort) per plan_depth_canon.md v8 §7/§8.
"""
import os
import sys

import numpy as np


def _repo_root(p):
    p = os.path.abspath(p)
    while not os.path.isdir(os.path.join(p, ".git")) and not os.path.isfile(os.path.join(p, ".git")):
        parent = os.path.dirname(p)
        if parent == p:
            raise RuntimeError("repo root (.git) not found")
        p = parent
    return p


REPO = _repo_root(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


# --------------------------------------------------------------------------- #
# raw-side selector: pure numpy, deliberately NOT the proxy implementation
# --------------------------------------------------------------------------- #
def _check_raw(raw):
    if not isinstance(raw, np.ndarray) or raw.ndim != 2 or raw.size == 0 \
            or not np.issubdtype(raw.dtype, np.floating):
        raise ValueError(f"raw depth must be a non-empty floating [H, W] ndarray, got "
                         f"{getattr(raw, 'dtype', type(raw).__name__)} {getattr(raw, 'shape', '')}")
    if not np.isfinite(raw).all():
        raise ValueError("raw depth contains non-finite values")


def raw_candidates(raw):
    """Columns whose maximum equals the global maximum (exact equality), ascending."""
    _check_raw(raw)
    P = raw.max(axis=0)
    return [int(c) for c in np.flatnonzero(P == P.max())]


def raw_rooted_survivors(raw, cands):
    """The candidates whose ROOTED raw image -- columns c, c+1, ... (mod W), each read
    top to bottom -- is lexicographically largest (exact comparison); ties all kept."""
    H = raw.shape[0]
    flat = np.ascontiguousarray(raw.T).ravel()          # column w = flat[w*H:(w+1)*H]
    best, best_img = [cands[0]], np.roll(flat, -cands[0] * H)
    for c in cands[1:]:
        img = np.roll(flat, -c * H)
        diff = np.flatnonzero(img != best_img)
        if diff.size == 0:
            best.append(c)
        elif img[diff[0]] > best_img[diff[0]]:
            best, best_img = [c], img
    return best


def raw_select(raw):
    """The plan §2.1 Steps B-C rule on a raw scalar map.

    Returns ``(j, tie_class, candidates, survivors)``; ``j`` and ``tie_class`` are
    ``None`` when several candidates survive the rooted-image key (exactly periodic)."""
    cands = raw_candidates(raw)
    if len(cands) == 1:
        n_max = int((raw[:, cands[0]] == raw.max()).sum())
        return cands[0], ("same_column" if n_max > 1 else "none"), cands, list(cands)
    survivors = raw_rooted_survivors(raw, cands)
    if len(survivors) == 1:
        return survivors[0], "cross_column_image", cands, survivors
    return None, None, cands, survivors


def max_circular_sep_deg(cands, img_w):
    """Largest circular distance between two candidate columns, in degrees (0 for one)."""
    if len(cands) < 2:
        return 0.0
    cols = max(min((a - b) % img_w, (b - a) % img_w) for a in cands for b in cands if a != b)
    return cols * 360.0 / img_w
