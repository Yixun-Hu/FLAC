#!/usr/bin/env python3
"""exp_25 depth_canon -- the PRE-FULL tie audit (plan_depth_canon.md v8 §7 rung 3).

Every depth panorama referenced by the train / unseen / seen AR splits is run through
the canonicalization selector twice: the treatment path (AR_md's own point cloud ->
``canonicalize_scene_metadata``) and an independent raw path on the scalar ``.npy``
(numpy, written here). Gate: PASS iff every map was processed and no map is an
image-key survivor (exactly periodic -> ``StabilizedTieError``). Read-only on the data.

Written by the exp_25 Coder seat (Claude Opus, max effort) per plan_depth_canon.md v8 §7/§8.
"""
import hashlib
import importlib.util
import json
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
sys.path.insert(0, REPO)  # this checkout's src/, never a stale pip-installed copy

# AR_md.get_custom_metadata with every modality but the depth switched off: the depth
# branch then runs alone, verbatim (its path logic and its point-cloud construction).
DEPTH_ONLY_MODALITIES = {"acoustic_context": {"load": False}, "depth": {"load": True},
                         "poses": {"load": False}}


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


# --------------------------------------------------------------------------- #
# split item -> depth map: AR_md's own string logic (it is inline in
# get_custom_metadata, so it cannot be imported; the tests pin equality with the
# path AR_md actually np.load()s)
# --------------------------------------------------------------------------- #
def resolve_depth_path(audio_path, root_path):
    """``(path, key)``: the depth path exactly as ``AR_md.get_custom_metadata`` builds
    it for the loader item ``audio_path`` (``info['relpath']`` computed as
    ``src/data/dataset.py`` does), and its key ``depth_map/<scene>/<id>/<r>.npy``."""
    rel_path = os.path.relpath(audio_path, root_path)                          # dataset.py
    common_suffix = os.path.commonpath([audio_path[::-1], rel_path[::-1]])[::-1]  # AR_md
    dataset_folder = audio_path[: -len(common_suffix)]
    scene_name = rel_path.split("/")[-3]
    scene_id = rel_path.split("/")[-2]
    filename = rel_path.split("/")[-1].split(".")[0]
    receiver_idx = int(filename.split("_")[1][1:])
    path = os.path.join(dataset_folder + "depth_map", scene_name, scene_id, f"{receiver_idx}.npy")
    return path, os.path.join("depth_map", scene_name, scene_id, f"{receiver_idx}.npy")


def load_metadata_module(path):
    """The custom metadata module, loaded by path exactly as ``src/data/dataset.py`` does."""
    spec = importlib.util.spec_from_file_location("metadata_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# split configs -> items (the loader's own json_scandir) -> unique maps + ledger
# --------------------------------------------------------------------------- #
def _abs(p):
    """A config path as train.py / eval_FLAC.py see it from the repository root."""
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_split_config(config_path):
    """One dataset config -> the fields the loader uses, with the content hashes."""
    with open(_abs(config_path)) as f:
        cfg = json.load(f)
    entries = cfg.get("datasets")
    if not isinstance(entries, list) or len(entries) != 1:
        raise ValueError(f"{config_path}: expected exactly one dataset entry, got {entries!r:.200}")
    entry = entries[0]
    for key in ("path", "json_file_path", "custom_metadata_module", "folder_name"):
        if not isinstance(entry.get(key), str) or not entry[key]:
            raise ValueError(f"{config_path}: dataset entry lacks {key!r}")
    module = _abs(entry["custom_metadata_module"])
    return {"dataset_config": config_path, "dataset_config_sha256": _sha256_file(_abs(config_path)),
            "json_file_path": entry["json_file_path"], "json_abs": _abs(entry["json_file_path"]),
            "json_sha256": _sha256_file(_abs(entry["json_file_path"])),
            "root": entry["path"], "root_abs": _abs(entry["path"]),
            "folder_name": entry["folder_name"], "custom_metadata_module": entry["custom_metadata_module"],
            "custom_metadata_module_abs": module, "custom_metadata_module_sha256": _sha256_file(module)}


def enumerate_items(split):
    """The split's item paths, built by the loader's own ``get_audio_filenames``."""
    from src.data.dataset import get_audio_filenames
    return get_audio_filenames(paths=split["root_abs"], json_file_path=split["json_abs"],
                               folder_name=split["folder_name"])


def collect_maps(splits):
    """``{key: {"path", "rep", "counts"}}`` over the union of the splits (deduplicated by
    the resolved depth path; ``rep`` = the smallest item path that resolves to it) and
    the per-split ledger rows ``{split, ..., n_items, n_unique_maps}``."""
    roots = {s["root_abs"] for s in splits.values()}
    modules = {s["custom_metadata_module_abs"] for s in splits.values()}
    if len(roots) != 1 or len(modules) != 1:
        raise ValueError(f"every split must use the same dataset root and metadata module, got "
                         f"roots {sorted(roots)} modules {sorted(modules)}")
    root = roots.pop()
    maps, ledger = {}, []
    for name, split in splits.items():
        items = enumerate_items(split)
        keys = set()
        for item in items:
            path, key = resolve_depth_path(item, root)
            entry = maps.setdefault(key, {"path": os.path.normpath(path), "rep": item,
                                          "counts": {n: 0 for n in splits}})
            if entry["path"] != os.path.normpath(path):
                raise ValueError(f"map key {key} resolves to two paths: {entry['path']} / {path}")
            entry["counts"][name] += 1
            entry["rep"] = min(entry["rep"], item)
            keys.add(key)
        ledger.append(dict(split, split=name, n_items=len(items), n_unique_maps=len(keys)))
    return maps, ledger
