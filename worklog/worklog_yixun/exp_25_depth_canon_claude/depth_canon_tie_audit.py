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
import io
import json
import os
import subprocess
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


# --------------------------------------------------------------------------- #
# per map: treatment path vs raw path
# --------------------------------------------------------------------------- #
def _canon_md(cloud):
    """The operator's input: AR_md's cloud plus fixed finite poses. The selector reads
    no pose (plan v6), so these only satisfy canonicalize_scene_metadata's contract."""
    import torch
    src = torch.tensor([1.0, 0.0, 0.0])
    ctx = torch.tensor([[0.0, 1.0, 0.0]])
    return {"depth": cloud, "source": src, "source_vit": src[None], "context_poses": ctx,
            "context_poses_vit": ctx.clone()}


def audit_map(key, entry, md_module, root_abs):
    """One row of the audit. Never raises: a failure is recorded in ``row['error']``
    (and the map then counts as NOT processed)."""
    import torch
    from src.data import yaw_rotation as yr
    row = {"path": key, "sha256": None, "counts": dict(entry["counts"]), "img_w": None,
           "j_star": None, "dj": None,
           "tie_class": None, "survivor": False, "n_candidates": None, "candidates": None,
           "survivors": None, "max_sep_deg": None, "raw_tie_class": None, "raw_survivor": None,
           "raw_agree_set": None, "raw_agree_col": None, "error": None}
    try:
        with open(entry["path"], "rb") as f:
            blob = f.read()
        row["sha256"] = hashlib.sha256(blob).hexdigest()
        raw = np.load(io.BytesIO(blob), allow_pickle=False)
        _check_raw(raw)
        info = {"path": entry["rep"], "relpath": os.path.relpath(entry["rep"], root_abs),
                "modalities": DEPTH_ONLY_MODALITIES}
        cloud = md_module.get_custom_metadata(info, None)["depth"]      # AR_md's own cloud
        ref = md_module.convert_equirect_to_camera_coord(torch.from_numpy(raw), *raw.shape)
        if not torch.equal(cloud, ref.permute(2, 0, 1)):
            raise RuntimeError("the cloud AR_md built differs from the resolved file's content")
        W = row["img_w"] = int(cloud.shape[2])
        P = yr.column_profile(yr.recover_depth_proxy(cloud))
        cands = [int(c) for c in torch.nonzero(P == P.max()).flatten().tolist()]
        try:
            _, dj, tie_class = yr.canonicalize_scene_metadata(_canon_md(cloud), W)
            j_star = (W - dj) % W
            if j_star not in cands:
                raise RuntimeError(f"selected column {j_star} is not a candidate {cands}")
            survivors = [j_star]
            row.update(j_star=j_star, dj=dj, tie_class=tie_class)
        except yr.StabilizedTieError as err:
            survivors = list(err.candidates)
            row["survivor"] = True
        raw_j, raw_class, raw_cands, raw_survivors = raw_select(raw)
        row.update(n_candidates=len(cands), candidates=cands, survivors=survivors,
                   max_sep_deg=max_circular_sep_deg(cands, W), raw_tie_class=raw_class,
                   raw_survivor=raw_j is None, raw_agree_set=raw_cands == cands,
                   raw_agree_col=raw_survivors == survivors)
    except Exception as err:  # noqa: BLE001 -- recorded, and the gate then refuses PASS
        row["error"] = f"{type(err).__name__}: {err}"
    return row


# --------------------------------------------------------------------------- #
# the whole audit
# --------------------------------------------------------------------------- #
SCHEMA_VERSION = 1
EXPDIR = os.path.dirname(os.path.abspath(__file__))
SPLIT_CONFIGS = {"train": "src/configs/dataset_configs/AR/train/acousticroom_train.json",
                 "unseen": "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json",
                 "seen": "src/configs/dataset_configs/AR/eval/acousticroom_seeneval.json"}
EXIT = {"PASS": 0, "FAIL": 1, "INCOMPLETE": 3, "REFUSED_SUBSET": 4}
_WORKER = {}


def _init_worker(module_path, root_abs):
    import torch
    torch.set_num_threads(1)
    _WORKER.update(module=load_metadata_module(module_path), root=root_abs)


def _task(item):
    return audit_map(item[0], item[1], _WORKER["module"], _WORKER["root"])


def run_rows(maps, module_path, root_abs, workers, limit=None):
    """Rows in sorted-path order (``imap`` keeps the order whatever ``workers`` is)."""
    tasks = [(k, maps[k]) for k in sorted(maps)][:limit]
    if workers <= 1:
        _init_worker(module_path, root_abs)
        return [_task(t) for t in tasks]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(workers, _init_worker, (module_path, root_abs)) as pool:
        return list(pool.imap(_task, tasks, chunksize=8))


def summarize(rows, split_names, img_w):
    def cls(r):
        return "error" if r["error"] else ("survivor" if r["survivor"] else r["tie_class"])
    keys = ("none", "same_column", "cross_column_image", "survivor", "error")
    per = {}
    for s in split_names:
        mine = [r for r in rows if r["counts"][s]]
        p = {"maps": dict.fromkeys(keys, 0), "items": dict.fromkeys(keys, 0),
             "dj_histogram_maps": [0] * img_w, "dj_histogram_items": [0] * img_w}
        for r in mine:
            p["maps"][cls(r)] += 1
            p["items"][cls(r)] += r["counts"][s]
            if r["dj"] is not None:
                p["dj_histogram_maps"][r["dj"]] += 1
                p["dj_histogram_items"][r["dj"]] += r["counts"][s]
        per[s] = p
    agree = {f: {"true": sum(r[f] is True for r in rows), "false": sum(r[f] is False for r in rows)}
             for f in ("raw_agree_set", "raw_agree_col")}
    return per, agree


def _git(*args):
    try:
        return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_report(rows, ledger, split_names, n_union, limit, module_abs):
    survivors = [{"path": r["path"], "candidates": r["survivors"], "counts": r["counts"]}
                 for r in rows if r["survivor"]]
    errors = [{"path": r["path"], "error": r["error"]} for r in rows if r["error"]]
    subset = limit is not None or tuple(split_names) != tuple(SPLIT_CONFIGS)
    verdict = ("FAIL" if survivors else "INCOMPLETE" if errors
               else "REFUSED_SUBSET" if subset or len(rows) != n_union else "PASS")
    widths = {r["img_w"] for r in rows if r["img_w"] is not None}
    if len(widths) > 1:
        raise ValueError(f"maps of different widths {sorted(widths)}")
    img_w = widths.pop() if widths else 0
    per, agree = summarize(rows, split_names, img_w)
    code = {"script": os.path.relpath(os.path.abspath(__file__), REPO),
            "src/data/yaw_rotation.py": "src/data/yaw_rotation.py",
            "src/data/dataset.py": "src/data/dataset.py", "custom_metadata_module": module_abs}
    tool = {"script_sha256": _sha256_file(os.path.abspath(__file__)),
            "code_sha256": {k: _sha256_file(_abs(v)) for k, v in code.items()},
            "source_sha": _git("rev-parse", "HEAD"),
            "dirty_code": _git("status", "--porcelain", "--", *[_abs(v) for v in code.values()])}
    return {"schema_version": SCHEMA_VERSION, "tool": tool, "inputs": ledger,
            "coverage": {"splits": list(split_names), "limit": limit, "subset": subset,
                         "n_maps_union": n_union, "n_maps_audited": len(rows),
                         "complete": not subset and len(rows) == n_union},
            "gate": {"verdict": verdict, "exit_code": EXIT[verdict], "n_survivors": len(survivors),
                     "n_errors": len(errors)},
            "per_split": per, "agreement": agree, "survivors": survivors, "errors": errors,
            "rows": rows}


def write_report(report, out):
    """Pretty header, one row per line (grep-able), then the ``sha256sum`` sidecar."""
    head = {k: v for k, v in report.items() if k != "rows"}
    text = json.dumps(head, indent=1)[:-2] + ',\n "rows": [\n' + ",\n".join(
        "  " + json.dumps(r, separators=(",", ":")) for r in report["rows"]) + "\n ]\n}\n"
    with open(out, "w") as f:
        f.write(text)
    with open(out + ".sha256", "w") as f:
        f.write(f"{hashlib.sha256(text.encode()).hexdigest()}  {os.path.basename(out)}\n")
