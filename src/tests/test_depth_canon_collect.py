"""exp_25 depth_canon r3c: the cell verifier + collector ``depth_canon_collect.py`` (plan v9
§1 tiers, §5 atomic completion, §9). Synthetic cells are written with the REAL evaluator
functions, so a format change in eval_FLAC.py breaks these fixtures instead of silently
diverging; streams are N = 8 items (the manifest's counts lowered to N; the registered
6,337 / 6,217 are pinned by test_depth_canon_eval_run.py).
Written by the exp_25 Coder seat (Claude Opus 5.5, max effort) per plan_depth_canon.md v9.
"""
import copy
import hashlib
import importlib.util
import json
import math
import os
from unittest import mock

import pytest
import torch

import eval_FLAC
from src.data.yaw_rotation import TIE_CLASSES, draw_yaw_offsets

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPDIR = os.path.join(REPO, "worklog", "worklog_yixun", "exp_25_depth_canon_claude")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(EXPDIR, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MC = _load("depth_canon_make_cells")
C = _load("depth_canon_collect")

PIN = "a" * 40
N, W = 8, 512
EMITTED = ("T60", "C50", "EDT", "FD", "Invalid T60", "RIR_to_GT_RIR_R@1", "RIR_to_GT_RIR_R@5",
           "RIR_to_GT_RIR_R@10", "RIR_to_geom_R@1", "RIR_to_geom_R@5", "RIR_to_geom_R@10")
SCENES = ("Apartments", "Auditorium", "Bathrooms", "Bedrooms", "Cafe", "ListeningRoom",
          "LivingRoomsWithHallway", "MeetingRoom", "Office", "Restaurants")
FARM = "outputs_FLAC/exp25_depth_canon"
CANON_BASE = "epoch={e}-step={s}"


def small_manifest(n=N):
    m = copy.deepcopy(MC.build_manifest())
    for split in m["splits"].values():
        split["expected_stream_count"] = n
    for c in m["cells"]:
        c["expected_stream_count"] = n
        a = c["eval_args"]
        a[a.index("--expected-stream-count") + 1] = str(n)
    return m


MANIFEST = small_manifest()


def row(block, arm=None, K=1, seed=42, deg=None, step=40000, sub=None):
    hits = [c for c in MANIFEST["cells"] if c["block"] == block and c["K"] == K
            and c["seed"] == seed and c["step"] == step and (arm is None or c["arm"] == arm)
            and (deg is None or c["rotate_deg"] == deg) and (sub is None or c["subblock"] == sub)]
    assert len(hits) == 1, (block, arm, K, seed, deg, step, sub, len(hits))
    return hits[0]


def ckpt_base(r):  # a CANON checkpoint's epoch is only known at run time; any value names it
    return CANON_BASE.format(e=r["step"] // 4551, s=r["step"]) if r["arm"] == "CANON" else "epoch=8-step=40000"


def target_ids(split, n=N):
    return [f"AR:{split}/Room{i % 3}/S00{i % 4}_R{i:03d}" for i in range(n)]


def build_stream(r, n=N):
    if r["rotate_mode"] == "random":
        g = torch.Generator(device="cpu").manual_seed(r["rotate_seed"])
        offs = [int(d) for d in draw_yaw_offsets(n, W, g).tolist()]
    else:
        offs = [eval_FLAC.yaw_column_shift(math.radians(r["rotate_deg"]), W)] * n
    s = eval_FLAC.RotationStream()
    for i, (t, off) in enumerate(zip(target_ids(r["split"], n), offs)):
        ctx = [f"{r['split']}:K{r['K']}:s{r['seed']}:{i}:{j}" for j in range(r["K"])]
        s.rows.append(eval_FLAC.StreamRow(i, t, ctx, W, off, i, 1))
    return s


def build_canon(r, stream):
    cs = eval_FLAC.CanonStream()
    for i, srow in enumerate(stream.rows):
        cls = "cross_column_image" if i % 5 == 3 else "same_column" if i % 7 == 4 else "none"
        sha = hashlib.sha256(f"{r['split']}:{srow.target_id}".encode()).hexdigest()
        cs.rows.append([i, srow.target_id, ((37 * i + 5) - srow.offset) % W, cls, sha])
    return cs


def default_metrics(r):
    base = {"CANON": 9.0, "P1": 10.0, "YAW": 9.5}[r["arm"]] + (1.0 if r["K"] == 1 else 0.0)
    return {k: round(base + 0.01 * (r["seed"] - 42) + 0.1 * j, 6) for j, k in enumerate(EMITTED)}


def write_cell(tmp, r, *, pin=PIN, n=N, metrics=None, by_scene=None, farm=FARM):
    """Everything eval_FLAC + the driver publish for one row, under ``tmp`` (= the cwd the
    evaluator ran in). Returns the metrics path (relative, as eval_FLAC names it)."""
    link = os.path.join(farm, r["arm"], ckpt_base(r) + ".ckpt")
    paths = eval_FLAC.build_output_paths(
        link, 1, 1.0, r["cell"], cond_method="vanilla", rotate_deg=float(r["rotate_deg"] or 0),
        n_angles=4, rotate_mode=r["rotate_mode"], rotate_seed=r["rotate_seed"], are_lambda=None,
        canonicalize=r["canonicalize"], canon_mismatch=r["canon_mismatch"])
    metrics_rel = paths["metrics"]
    at = lambda p: os.path.join(str(tmp), p)  # noqa: E731
    os.makedirs(os.path.dirname(at(metrics_rel)), exist_ok=True)  # the farm, under the cwd
    plan = eval_FLAC.resolve_rotation_plan(r["rotate_mode"], float(r["rotate_deg"] or 0),
                                           r["rotate_seed"], r["seed"])
    stream = build_stream(r, n)
    canon = build_canon(r, stream) if r["canonicalize"] else None
    m = metrics or default_metrics(r)
    scenes = by_scene or {s: {k: v + 0.01 * i for k, v in m.items()} for i, s in enumerate(SCENES)}
    rot = dict(rotate_mode="random", rotate_seed=plan.rotate_seed, input_hash=stream.input_hash(),
               assignment_hash=stream.assignment_hash(), stream_count=n, img_w=W) if plan.is_random else {}
    prov = None  # build_canon_provenance exactly as evaluate_model calls it
    if r["canon_in_play"]:
        with open(os.path.join(REPO, r["model_config"])) as f:
            cfg = json.load(f)
        proto = eval_FLAC.CanonProtocol(True, r["canon_trained"], r["canonicalize"],
                                        r["canon_mismatch"], W, None, None)
        prov = eval_FLAC.build_canon_provenance(proto, canon, cfg, copy.deepcopy(cfg))
    with mock.patch.object(eval_FLAC, "source_sha", lambda: pin):
        rec = eval_FLAC.build_metrics_record(
            m, link, float(r["rotate_deg"] or 0), "vanilla", None, cond_autocast="bf16",
            batch_size=64, n_samples=n, dataset_config=r["dataset_config"], seed=r["seed"],
            cfg_scale=1.0, steps=1, eval_name=r["cell"], weights_source="ema", device="cuda",
            by_scene=scenes, frame_avg_max_fwd_samples=64, canon_provenance=prov, **rot)
        with open(at(metrics_rel), "w") as f:
            json.dump(rec, f, indent=4)
        stream_rel = eval_FLAC.stream_sidecar_path(metrics_rel)
        with open(at(stream_rel), "w") as f:
            json.dump(eval_FLAC.build_stream_record(plan, stream), f, indent=2)
        canon_rel = None
        if canon is not None:
            canon_rel = eval_FLAC.canon_sidecar_path(metrics_rel)
            with open(at(canon_rel), "w") as f:
                json.dump(eval_FLAC.build_canon_record(canon, W), f, indent=2)
    return metrics_rel


# ---- cycle 2a: the metrics record (and, next, the stream sidecar) vs the cell class ---- #
CASES = [("T", "CANON", 1, 42, 0, 40000), ("C4+45", "CANON", 8, 42, 90, 40000),
         ("R", "CANON", 1, 44, None, 40000), ("D-i", "CANON", 8, 42, 0, 40000),
         ("D-ii", "P1", 1, 42, 0, 40000), ("S", "CANON", 8, 45, 0, 40000),
         ("P1-pin", "P1", 1, 43, 0, 40000), ("YAW-pin", "YAW", 8, 42, 45, 40000),
         ("TRAJ", "CANON", 1, 42, 270, 7500), ("T-dup", "CANON", 8, 42, 0, 40000)]


@pytest.fixture(params=CASES, ids=[f"{c[0]}_{c[1]}_K{c[2]}" for c in CASES])
def case_row(request):
    block, arm, k, seed, deg, step = request.param
    return row(block, arm, K=k, seed=seed, deg=deg, step=step)


def test_record_and_stream_of_a_true_cell_verify(tmp_path, case_row):
    rel = write_cell(tmp_path, case_row)
    rec = json.load(open(tmp_path / rel))
    assert C.record_reasons(rec, case_row, pin=PIN, link=rec["ckpt_path"]) == []


C4 = dict(block="C4+45", K=1, deg=90)
EDITS = [  # (row spec, edit, needle)
    (C4, lambda d: d.update(cond_autocast="default"), "cond_autocast"),
    (C4, lambda d: d.update(cond_method="fa_invariant"), "cond_method"),
    (C4, lambda d: d.update(weights_source="online"), "weights_source"),
    (C4, lambda d: d.update(cfg_scale=1), "cfg_scale"), (C4, lambda d: d.update(steps=2), "steps"),
    (C4, lambda d: d.update(n_samples=N - 1), "n_samples"), (C4, lambda d: d.update(seed=43), "seed"),
    (C4, lambda d: d.update(eval_name="exp25_other"), "eval_name"),
    (C4, lambda d: d.update(source_sha="b" * 40), "source_sha"),
    (C4, lambda d: d.update(device="cpu"), "device"), (C4, lambda d: d.update(batch_size=32), "batch_size"),
    (C4, lambda d: d.update(dataset_config="x.json"), "dataset_config"),
    (C4, lambda d: d.update(rotate_deg=0.0), "rotate_deg"),
    (C4, lambda d: d["metrics"].update(T60=float("nan")), "T60"),
    (C4, lambda d: d["metrics"].pop("FD"), "metrics keys"), (C4, lambda d: d.pop("by_scene"), "key order"),
    (C4, lambda d: d["by_scene"].pop("Cafe"), "by_scene"),
    (C4, lambda d: d.update(canonicalize_applied=False), "canonicalize_applied"),
    (C4, lambda d: d.update(model_config_sha256_embedded="0" * 64), "model_config_sha256"),
    (C4, lambda d: d.update(canonicalize_tie_counts={"none": N, "same_column": 0,
                                                     "cross_column_image": 0}), "tie_counts"),
    (dict(block="D-i", K=1), lambda d: d.update(canonicalize_mismatch=False), "canonicalize_mismatch"),
    (dict(block="D-ii", K=8), lambda d: d.update(canonicalize_mismatch=False), "canonicalize_mismatch"),
    (dict(block="P1-pin", sub="T"), lambda d: d.update(canonicalize_trained=False), "key order"),
]


@pytest.mark.parametrize("spec, edit, needle", EDITS, ids=[f"{e[0]['block']}-{e[2]}" for e in EDITS])
def test_record_violations_are_named(tmp_path, spec, edit, needle):
    r = row(**spec)
    rec = json.load(open(tmp_path / write_cell(tmp_path, r)))
    assert C.record_reasons(rec, r, pin=PIN, link=rec["ckpt_path"]) == []
    edit(rec)
    reasons = C.record_reasons(rec, r, pin=PIN, link=rec["ckpt_path"])
    assert reasons and any(needle in x for x in reasons), reasons
