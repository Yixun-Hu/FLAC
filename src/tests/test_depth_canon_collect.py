"""exp_25 depth_canon r3c: the cell verifier + collector ``depth_canon_collect.py`` (plan v9
§1 tiers, §5 atomic completion, §9). Synthetic cells are written with the REAL evaluator
functions, so a format change in eval_FLAC.py breaks these fixtures instead of silently
diverging; streams are N = 8 items (the manifest's counts lowered to N; the registered
6,337 / 6,217 are pinned by test_depth_canon_eval_run.py).
Written by the exp_25 Coder seat (Claude Opus 5.5, max effort) per plan_depth_canon.md v9.
"""
import collections
import copy
import shutil
import hashlib
import importlib.util
import json
import math
import os
from unittest import mock

import numpy as np
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
N, W, T = 8, 512, 16
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


def write_cell(tmp, r, *, pin=PIN, n=N, metrics=None, by_scene=None, farm=FARM, waves=None,
               dump=True, receipt=True, ckpt_sha=None, link=None, dump_root=None):
    """Everything eval_FLAC + the driver publish for one row, under ``tmp`` (= the cwd the
    evaluator ran in). Returns the metrics path (relative, as eval_FLAC names it)."""
    link = link or os.path.join(farm, r["arm"], ckpt_base(r) + ".ckpt")  # = the evaluator's --ckpt-path
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
        dump_root = dump_root or os.path.join(str(tmp), "dumps")
        os.makedirs(dump_root, exist_ok=True)
        if dump:  # finalize exactly as evaluate_model calls it
            d = eval_FLAC.ScoredWaveformDump(eval_FLAC.scored_waveform_dir(dump_root, metrics_rel), n)
            d.append(waveforms_for(r, n) if waves is None else waves)
            sha = lambda p: None if p is None else eval_FLAC._file_sha256(at(p))  # noqa: E731
            d.finalize(stream.input_hash(), os.path.basename(d.final_dir), extra_meta={
                "ckpt_path": link, "metrics_path": metrics_rel, "metrics_sha256": sha(metrics_rel),
                "stream_sidecar_path": stream_rel, "stream_sidecar_sha256": sha(stream_rel),
                "canon_sidecar_path": canon_rel, "canon_sidecar_sha256": sha(canon_rel)})
    if receipt:
        write_receipt(tmp, r, link, metrics_rel, dump, pin, ckpt_sha, dump_root=dump_root)
    return metrics_rel


def waveforms_for(r, n=N):
    g = torch.Generator().manual_seed(1000 * r["K"] + r["seed"])
    return torch.rand(n, 1, T, generator=g) * 2 - 1


def canon_index():
    """What a launcher checkpoint manifest yields: step -> (target path, sha256)."""
    ns = MANIFEST["arms"]["CANON"]["ckpt"]["run_ns"]
    return {s: (f"{ns}/checkpoints/{CANON_BASE.format(e=s // 4551, s=s)}.ckpt",
                hashlib.sha256(f"canon-{s}".encode()).hexdigest()) for s in range(2500, 40001, 2500)}


def expected_target(r):
    ck = MANIFEST["arms"][r["arm"]]["ckpt"]
    return canon_index()[r["step"]] if r["arm"] == "CANON" else (ck["path"], ck["sha256"])


DIGEST = "d" * 64   # the campaign's relevant-tree content digest (rB-F1 B3)


def write_receipt(tmp, r, link, metrics_rel, dump, pin, ckpt_sha=None, dump_root=None):
    """The driver's receipt, written by the REAL writer (``depth_canon_collect.write_receipt``)
    from a real plan entry; ``ckpt_sha`` tampers the written file afterwards."""
    dump_root = dump_root or os.path.join(str(tmp), "dumps")
    entry = C.plan_cells(MANIFEST, [r], farm_root=os.path.dirname(os.path.dirname(link)), dump_root=dump_root,
                         canon_index=canon_index(), no_dump_blocks=[] if dump else [r["block"]])[0]
    assert entry["link"] == link and entry["metrics"] == metrics_rel
    path = C.write_receipt(entry, r, base_dir=str(tmp), lane=0, log="logs/x.log", rc=0, pin=pin,
                           ckpt_sha256=entry["sha256"], campaign_digest=DIGEST, spawn_digest=DIGEST,
                           post_digest=DIGEST)
    if ckpt_sha:
        edit_json(path, lambda d: d.update(ckpt_sha256=ckpt_sha))
    return path


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
    (C4, lambda d: d.update(canonicalize_tie_counts={"none": N, "same_column": 1,
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


# ---- cycle 2b: the stream sidecar and the canon sidecar ---- #
def _sidecars(tmp, r):
    rel = write_cell(tmp, r)
    load = lambda p: json.load(open(os.path.join(str(tmp), p))) if p else None  # noqa: E731
    canon = eval_FLAC.canon_sidecar_path(rel) if r["canonicalize"] else None
    return load(rel), load(eval_FLAC.stream_sidecar_path(rel)), load(canon)


def test_sidecars_of_a_true_cell_verify(tmp_path, case_row):
    rec, stream, canon = _sidecars(tmp_path, case_row)
    assert C.stream_reasons(stream, case_row, rec) == []
    assert (canon is None) == (not case_row["canonicalize"])
    if canon is not None:
        assert C.canon_reasons(canon, case_row, stream, rec) == []


STREAM_EDITS = [(lambda d: d.update(stream_count=N + 1), "stream_count"),
                (lambda d: d["input_tuples"].pop(), "input_tuples"),
                (lambda d: d.update(input_hash="0" * 64), "input_hash"),
                (lambda d: d["offsets"].__setitem__(0, 1), "offset"),
                (lambda d: d.update(img_w=256), "img_w"),
                (lambda d: d.update(rotate_mode="random"), "rotate_mode"),
                (lambda d: d["input_tuples"].__setitem__(1, [5] + d["input_tuples"][1][1:]), "position")]


@pytest.mark.parametrize("edit, needle", STREAM_EDITS, ids=[e[1] for e in STREAM_EDITS])
def test_stream_violations_are_named(tmp_path, edit, needle):
    r = row("C4+45", K=1, deg=90)
    rec, stream, _ = _sidecars(tmp_path, r)
    edit(stream)
    reasons = C.stream_reasons(stream, r, rec)
    assert reasons and any(needle in x for x in reasons), reasons


def test_random_record_and_stream_are_one_stream(tmp_path):
    r = row("R", K=8, seed=46)
    rec, stream, _ = _sidecars(tmp_path, r)
    rec["assignment_hash"] = "0" * 64
    assert any("assignment_hash" in x for x in C.stream_reasons(stream, r, rec))


def _relabel(d):  # same total, different classes
    p = next(p for p in d["positions"] if p[3] == "none")
    p[3] = "same_column"


CANON_EDITS = [(lambda c, r: _relabel(c), "tie_counts"),
               (lambda c, r: c["positions"][0].__setitem__(1, "AR:other"), "target"),
               (lambda c, r: c.update(canonical_depth_hash="0" * 64), "canonical_depth_hash"),
               (lambda c, r: c.update(canon_hash="0" * 64), "canon_hash"),
               (lambda c, r: c["positions"][1].__setitem__(2, W), "malformed"),
               (lambda c, r: c["positions"][2].__setitem__(4, "x"), "malformed"),
               (lambda c, r: c.update(stream_count=N - 1), "stream_count"),
               (lambda c, r: c.update(tiebreak="pose"), "tiebreak"),
               (lambda c, r: r.update(canonicalize_tie_counts={
                   "none": r["canonicalize_tie_counts"]["none"] - 1,
                   "same_column": r["canonicalize_tie_counts"]["same_column"] + 1,
                   "cross_column_image": r["canonicalize_tie_counts"]["cross_column_image"]}),
                "record canonicalize_tie_counts")]


@pytest.mark.parametrize("edit, needle", CANON_EDITS, ids=[e[1] for e in CANON_EDITS])
def test_canon_violations_are_named(tmp_path, edit, needle):
    r = row("R", K=1, seed=43)
    rec, stream, canon = _sidecars(tmp_path, r)
    edit(canon, rec)
    reasons = C.canon_reasons(canon, r, stream, rec)
    assert reasons and any(needle in x for x in reasons), reasons


# ---- cycle 2c: the waveform dump (announcement 08) and the driver's receipt ---- #
def _expect(tmp, rel, r):
    at = lambda p: os.path.join(str(tmp), p)  # noqa: E731
    canon = eval_FLAC.canon_sidecar_path(rel) if r["canonicalize"] else None
    return {"ckpt_path": os.path.join(FARM, r["arm"], ckpt_base(r) + ".ckpt"), "metrics_path": rel,
            "metrics_file": at(rel), "stream_sidecar_path": eval_FLAC.stream_sidecar_path(rel),
            "stream_file": at(eval_FLAC.stream_sidecar_path(rel)), "canon_sidecar_path": canon,
            "canon_file": at(canon) if canon else None}


def _dump_check(tmp, rel, r, mode="full"):
    stream = json.load(open(os.path.join(str(tmp), eval_FLAC.stream_sidecar_path(rel))))
    stem = os.path.basename(rel)[:-len(".json")]
    return C.dump_reasons(os.path.join(str(tmp), "dumps", stem), r, stem=stem, stream=stream,
                          expect=_expect(tmp, rel, r), pin=PIN, mode=mode)


@pytest.mark.parametrize("mode", ["full", "manifest"])
def test_dump_of_a_true_cell_verifies(tmp_path, case_row, mode):
    assert _dump_check(tmp_path, write_cell(tmp_path, case_row), case_row, mode) == []


def _dump_dir(tmp, rel):
    return os.path.join(str(tmp), "dumps", os.path.basename(rel)[:-len(".json")])


DUMP_EDITS = [
    (lambda t, rel: os.unlink(os.path.join(_dump_dir(t, rel), "COMPLETE")), "complete", ("full", "manifest")),
    (lambda t, rel: os.mkdir(_dump_dir(t, rel) + ".staging-77"), "staging", ("full", "manifest")),
    (lambda t, rel: np.save(os.path.join(_dump_dir(t, rel), "waveforms.npy"), np.zeros((N, T), np.float32)),
     "complete", ("full",)),
    (lambda t, rel: edit_json(os.path.join(str(t), rel), lambda d: d.update(seed=42)), "metrics_sha256",
     ("full", "manifest")),
    (lambda t, rel: edit_json(os.path.join(_dump_dir(t, rel), "manifest.json"),
                              lambda d: d.update(input_hash="0" * 64)), "input_hash", ("full", "manifest")),
    (lambda t, rel: edit_json(os.path.join(_dump_dir(t, rel), "manifest.json"),
                              lambda d: d.update(source_sha="b" * 40)), "source_sha", ("full", "manifest")),
    (lambda t, rel: edit_json(os.path.join(_dump_dir(t, rel), "manifest.json"),
                              lambda d: d.update(ckpt_path="elsewhere.ckpt")), "ckpt_path", ("full", "manifest")),
]


def edit_json(path, fn):
    with open(path) as f:
        d = json.load(f)
    fn(d)
    with open(path, "w") as f:
        json.dump(d, f, indent=2)


@pytest.mark.parametrize("edit, needle, modes", DUMP_EDITS, ids=[f"{e[1]}{i}" for i, e in enumerate(DUMP_EDITS)])
def test_dump_violations_are_named(tmp_path, edit, needle, modes):
    r = row("C4+45", K=8, deg=180)
    rel = write_cell(tmp_path, r)
    edit(tmp_path, rel)
    for mode in modes:
        reasons = _dump_check(tmp_path, rel, r, mode)
        assert reasons and any(needle in x.lower() for x in reasons), (mode, reasons)


def _receipt_check(tmp, rel, r, **over):
    receipt = json.load(open(os.path.join(str(tmp), rel[:-len(".json")] + ".cell.json")))
    target, sha = expected_target(r)
    kw = dict(stem=os.path.basename(rel)[:-len(".json")], link=_expect(tmp, rel, r)["ckpt_path"],
              target=target, ckpt_sha256=sha, dump=True, dump_root=os.path.join(str(tmp), "dumps"), pin=PIN)
    kw.update(over)
    return C.receipt_reasons(receipt, r, **kw)


def test_receipt_of_a_true_cell_verifies(tmp_path, case_row):
    assert _receipt_check(tmp_path, write_cell(tmp_path, case_row), case_row) == []


@pytest.mark.parametrize("over, needle", [
    (dict(ckpt_sha256="0" * 64), "ckpt_sha256"), (dict(pin="b" * 40), "expect_head"),
    (dict(dump=False), "dump"), (dict(dump_root="/elsewhere"), "argv"),
    (dict(target="/nas/other.ckpt"), "ckpt_target"), (dict(link="farm/x.ckpt"), "ckpt_link")])
def test_receipt_violations_are_named(tmp_path, over, needle):
    r = row("P1-pin", "P1", K=8, seed=44, sub="R")
    reasons = _receipt_check(tmp_path, write_cell(tmp_path, r), r, **over)
    assert reasons and any(needle in x for x in reasons), reasons


def _receipt_path(tmp, rel):
    return os.path.join(str(tmp), rel[:-len(".json")] + ".cell.json")


@pytest.mark.parametrize("edit, needle", [
    (lambda d: d.update(spawn_digest="e" * 64), "spawn_digest"),
    (lambda d: d.update(post_digest="e" * 64), "post_digest"),
    (lambda d: d.update(campaign_digest="nope", spawn_digest="nope", post_digest="nope"), "campaign_digest"),
    (lambda d: d.pop("post_digest"), "receipt keys")], ids=["spawn", "post", "malformed", "absent"])
def test_a_receipt_whose_tree_digest_moved_is_refused(tmp_path, edit, needle):
    """rB-F1 B3: the receipt binds the relevant-tree digest at campaign start, at the spawn and
    after the evaluator exited; all three must be one sha256."""
    r = row("C4+45", K=1, deg=180)
    rel = write_cell(tmp_path, r)
    edit_json(_receipt_path(tmp_path, rel), edit)
    reasons = _receipt_check(tmp_path, rel, r)
    assert reasons and any(needle in x for x in reasons), reasons


def test_the_skip_predicate_binds_this_campaigns_digest(tmp_path):
    r = row("T", K=8, seed=43)
    rel = write_cell(tmp_path, r)
    assert verify(tmp_path, rel, expect_digest=DIGEST) == []
    assert any("campaign_digest" in x for x in verify(tmp_path, rel, expect_digest="f" * 64))


def test_write_receipt_refuses_a_moved_tree(tmp_path):
    r = row("T", K=1, seed=44)
    entry = C.plan_cells(MANIFEST, [r], farm_root=FARM, dump_root=str(tmp_path), canon_index=canon_index(),
                         no_dump_blocks=[])[0]
    os.makedirs(os.path.join(str(tmp_path), os.path.dirname(entry["receipt"])))
    for spawn, post in ((DIGEST, "e" * 64), ("e" * 64, DIGEST)):
        with pytest.raises(ValueError, match="digest"):
            C.write_receipt(entry, r, base_dir=str(tmp_path), lane=0, log="x", rc=0, pin=PIN,
                            ckpt_sha256=entry["sha256"], campaign_digest=DIGEST, spawn_digest=spawn,
                            post_digest=post)
    assert not os.path.exists(os.path.join(str(tmp_path), entry["receipt"]))


# ---- rB-F1 B6 (Codex rB HIGH; decision A7): the receipt is offline-verifiable publication evidence ---- #
def test_the_receipt_binds_the_artifacts_and_the_verified_dump(tmp_path):
    r = row("T", K=8, seed=44)
    rel = write_cell(tmp_path, r)
    rec = json.load(open(_receipt_path(tmp_path, rel)))
    at = lambda p: os.path.join(str(tmp_path), p)  # noqa: E731
    assert rec["artifact_sha256"] == {"metrics": eval_FLAC._file_sha256(at(rel)),
                                      "stream": eval_FLAC._file_sha256(at(eval_FLAC.stream_sidecar_path(rel))),
                                      "canon": eval_FLAC._file_sha256(at(eval_FLAC.canon_sidecar_path(rel)))}
    d = _dump_dir(tmp_path, rel)
    ev = rec["dump_evidence"]
    assert ev["manifest_sha256"] == eval_FLAC._file_sha256(os.path.join(d, "manifest.json"))
    assert ev["manifest"] == json.load(open(os.path.join(d, "manifest.json")))
    assert ev["marker"] == open(os.path.join(d, "COMPLETE")).read() == ev["manifest"]["sha256"] + "\n"
    assert ev["waveforms_sidecar"] == open(os.path.join(d, "waveforms.sha256")).read()
    no_dump = row("TRAJ", K=1, deg=0, step=2500)
    rec2 = json.load(open(_receipt_path(tmp_path, write_cell(tmp_path, no_dump, dump=False))))
    assert rec2["dump_evidence"] is None and rec2["artifact_sha256"]["canon"]


def test_offline_verification_needs_no_dump_root(tmp_path):
    r = row("P1-pin", "P1", K=1, seed=46, sub="T")
    rel = write_cell(tmp_path, r)
    assert verify(tmp_path, rel, dump_check="receipt") == []
    shutil.rmtree(os.path.join(str(tmp_path), "dumps"))                 # the NAS unmounted / moved / collected
    assert verify(tmp_path, rel, dump_check="receipt") == []
    assert any("dump" in x for x in verify(tmp_path, rel, dump_check="manifest"))   # the online modes see it


OFFLINE_EDITS = [
    (lambda d: d["dump_evidence"]["manifest"].update(n=N - 1), "manifest"),
    (lambda d: d["dump_evidence"].update(marker="0" * 64 + "\n"), "marker"),
    (lambda d: d["dump_evidence"].update(manifest_sha256="0" * 64), "manifest_sha256"),
    (lambda d: d["dump_evidence"]["manifest"].update(input_hash="0" * 64), "input_hash"),
    (lambda d: d["dump_evidence"].update(waveforms_sidecar="x"), "sidecar"),
    (lambda d: d["artifact_sha256"].update(stream="0" * 64), "artifact_sha256"),
    (lambda d: d.update(dump_evidence=None), "dump_evidence"),
]


@pytest.mark.parametrize("edit, needle", OFFLINE_EDITS, ids=[e[1] + str(i) for i, e in enumerate(OFFLINE_EDITS)])
def test_a_tampered_receipt_fails_offline_verification(tmp_path, edit, needle):
    r = row("R", K=1, seed=42)
    rel = write_cell(tmp_path, r)
    edit_json(_receipt_path(tmp_path, rel), edit)
    reasons = verify(tmp_path, rel, dump_check="receipt")
    assert reasons and any(needle in x for x in reasons), reasons


def test_a_committed_artifact_edited_after_the_receipt_fails_offline(tmp_path):
    r = row("YAW-pin", "YAW", K=8, seed=45, sub="T")
    rel = write_cell(tmp_path, r)
    edit_json(os.path.join(str(tmp_path), eval_FLAC.stream_sidecar_path(rel)), lambda d: d.update(note=1))
    assert any("stream" in x for x in verify(tmp_path, rel, dump_check="receipt"))


def test_a_dump_changed_after_the_receipt_is_seen_online(tmp_path):
    r = row("T", K=1, seed=45)
    rel = write_cell(tmp_path, r)
    edit_json(os.path.join(_dump_dir(tmp_path, rel), "manifest.json"), lambda d: d.update(note=1))
    assert any("dump_evidence" in x or "manifest" in x for x in verify(tmp_path, rel, dump_check="manifest"))


def test_receipt_rc_and_wrong_checkpoint_sha_are_refused(tmp_path):
    r = row("TRAJ", K=8, deg=90, step=12500)
    rel = write_cell(tmp_path, r, ckpt_sha="1" * 64)
    assert any("ckpt_sha256" in x for x in _receipt_check(tmp_path, rel, r))
    edit_json(os.path.join(str(tmp_path), rel[:-len(".json")] + ".cell.json"), lambda d: d.update(rc=1))
    assert any("rc" in x for x in _receipt_check(tmp_path, rel, r))


# ---- cycle 2d: verify_cell (the driver's resume predicate), the CANON index, the CLI ---- #
def verify(tmp, rel, **kw):
    for key, value in (("manifest", MANIFEST), ("farm_root", FARM), ("base_dir", str(tmp)),
                       ("dump_root", os.path.join(str(tmp), "dumps")), ("pin", PIN),
                       ("canon_index", canon_index()), ("check_link", False)):
        kw.setdefault(key, value)
    return C.verify_cell(os.path.basename(rel)[:-len(".json")], **kw)


def test_verify_cell_accepts_every_true_cell_class(tmp_path, case_row):
    rel = write_cell(tmp_path, case_row)
    assert verify(tmp_path, rel) == [] and verify(tmp_path, rel, dump_check="manifest") == []


@pytest.mark.parametrize("victim", ["metrics", "stream", "receipt", "canon"])
def test_verify_cell_names_a_missing_artifact(tmp_path, victim):
    r = row("T", K=8, seed=45)
    rel = write_cell(tmp_path, r)
    path = {"metrics": rel, "stream": eval_FLAC.stream_sidecar_path(rel),
            "receipt": rel[:-len(".json")] + ".cell.json", "canon": eval_FLAC.canon_sidecar_path(rel)}[victim]
    os.unlink(os.path.join(str(tmp_path), path))
    assert any(victim in x and "missing" in x for x in verify(tmp_path, rel))


def test_verify_cell_refusals(tmp_path):
    r = row("P1-pin", "P1", K=1, seed=45, sub="T")
    rel = write_cell(tmp_path, r)
    stem = os.path.basename(rel)[:-len(".json")]
    assert verify(tmp_path, rel, pin="xyz") and "registered" in " ".join(
        C.verify_cell("nonsense_metrics_1_1.0_x", manifest=MANIFEST, farm_root=FARM, dump_root="/d",
                      pin=PIN))
    canon_name = os.path.basename(eval_FLAC.canon_sidecar_path(rel))
    open(os.path.join(str(tmp_path), os.path.dirname(rel), canon_name), "w").write("{}")
    assert any("unexpected canon sidecar" in x for x in verify(tmp_path, rel))
    t = row("T", K=1, seed=42)
    rel_t = write_cell(tmp_path, t)
    assert any("CANON" in x for x in verify(tmp_path, rel_t, canon_index=None))
    wrong_epoch = os.path.basename(rel_t).replace("epoch=8-", "epoch=7-")[:-len(".json")]
    assert any("checkpoint" in x for x in C.verify_cell(
        wrong_epoch, manifest=MANIFEST, farm_root=FARM, dump_root="/d", pin=PIN, canon_index=canon_index()))
    assert stem.startswith("epoch=8-step=40000_metrics_")


def test_verify_cell_without_a_dump(tmp_path):
    r = row("TRAJ", K=1, deg=0, step=2500)
    rel = write_cell(tmp_path, r, dump=False)
    assert verify(tmp_path, rel, expect_dump=False) == []
    assert any("dump" in x for x in verify(tmp_path, rel))           # the default expects one
    r2 = row("TRAJ", K=1, deg=90, step=2500)
    rel2 = write_cell(tmp_path, r2)
    assert any("dump" in x for x in verify(tmp_path, rel2, expect_dump=False))


def test_verify_cell_checks_the_farm_entry_is_the_cache_file(tmp_path):
    """rB-F1 B4: the farm entry is a regular file hard-linked to the content-addressed cache
    file <farm>/.ckpt_cache/<sha256>.ckpt -- never a symlink (it could be retargeted)."""
    r = row("YAW-pin", "YAW", K=1, seed=42, sub="T")
    rel = write_cell(tmp_path, r)
    sha = MANIFEST["arms"]["YAW"]["ckpt"]["sha256"]
    link = os.path.join(str(tmp_path), FARM, "YAW", "epoch=8-step=40000.ckpt")
    cache = os.path.join(str(tmp_path), FARM, ".ckpt_cache", f"{sha}.ckpt")
    os.makedirs(os.path.dirname(cache))
    open(cache, "wb").write(b"the verified copy")
    assert any("farm entry" in x for x in verify(tmp_path, rel, check_link=True))          # absent
    os.symlink(cache, link)
    assert any("symlink" in x for x in verify(tmp_path, rel, check_link=True))             # never a symlink
    os.unlink(link)
    open(link, "wb").write(b"the verified copy")
    assert any("hard link" in x for x in verify(tmp_path, rel, check_link=True))           # another inode
    os.unlink(link)
    os.link(cache, link)
    assert verify(tmp_path, rel, check_link=True) == []


@pytest.mark.parametrize("edit, needle", [
    (lambda d: d.update(ckpt_sha256_measured="1" * 64), "ckpt_sha256_measured"),
    (lambda d: d.update(ckpt_cache="outputs_FLAC/elsewhere.ckpt"), "ckpt_cache")])
def test_the_receipt_binds_the_measured_checkpoint(tmp_path, edit, needle):
    r = row("P1-pin", "P1", K=8, seed=43, sub="T")
    rel = write_cell(tmp_path, r)
    rec = json.load(open(_receipt_path(tmp_path, rel)))
    sha = MANIFEST["arms"]["P1"]["ckpt"]["sha256"]
    assert rec["ckpt_sha256_measured"] == rec["ckpt_sha256"] == sha
    assert rec["ckpt_cache"] == f"{FARM}/.ckpt_cache/{sha}.ckpt"
    edit_json(_receipt_path(tmp_path, rel), edit)
    assert any(needle in x for x in _receipt_check(tmp_path, rel, r))


def _canon_manifest_file(tmp, lines=None):
    path = os.path.join(str(tmp), "depth_canon_x_exp25_CANON_checkpoints.sha256")
    with open(path, "w") as f:
        f.writelines(lines if lines is not None else
                     [f"{sha}  {p}\n" for _, (p, sha) in sorted(canon_index().items())])
    return path


def test_canon_index_from_the_launcher_manifest(tmp_path):
    ns = MANIFEST["arms"]["CANON"]["ckpt"]["run_ns"]
    assert C.load_canon_index(_canon_manifest_file(tmp_path), ns) == canon_index()
    good = f"{'1' * 64}  {ns}/checkpoints/epoch=0-step=2500.ckpt\n"
    for bad in ([good, good], [f"{'1' * 64}  {ns}/notes.txt\n"], ["nonsense\n"],
                [f"{'1' * 64}  /other/checkpoints/epoch=0-step=2500.ckpt\n"]):
        with pytest.raises(ValueError):
            C.load_canon_index(_canon_manifest_file(tmp_path, bad), ns)


def test_cli_verify_cell(tmp_path, capsys):
    cells = os.path.join(str(tmp_path), "cells.json")
    with open(cells, "w") as f:
        json.dump(MANIFEST, f)
    r = row("C4+45", K=1, deg=270)
    rel = write_cell(tmp_path, r)
    argv = ["--verify-cell", os.path.basename(rel)[:-len(".json")], "--pin", PIN, "--cells", cells,
            "--farm-root", FARM, "--dump-root", os.path.join(str(tmp_path), "dumps"), "--base-dir",
            str(tmp_path), "--canon-ckpt-manifest", _canon_manifest_file(tmp_path), "--no-link-check"]
    assert C.main(argv) == 0 and "VALID" in capsys.readouterr().out
    os.unlink(os.path.join(str(tmp_path), eval_FLAC.stream_sidecar_path(rel)))
    assert C.main(argv) == 1 and "INVALID" in capsys.readouterr().out


def test_a_quarantined_staging_dir_is_not_a_leftover(tmp_path):
    """The driver parks a staging leftover as <name>.staging-<pid>.incomplete-<ts>: that must
    never count as a leftover again, or the cell could never complete."""
    r = row("T", K=1, seed=44)
    rel = write_cell(tmp_path, r)
    d = os.path.join(str(tmp_path), "dumps", os.path.basename(rel)[:-len(".json")])
    os.mkdir(d + ".staging-12.incomplete-2026-10-24_00-00-00")
    assert C.staging_leftovers(d) == [] and _dump_check(tmp_path, rel, r) == []
    os.mkdir(d + ".staging-13")
    assert C.staging_leftovers(d) == [os.path.basename(d) + ".staging-13"]


# ---- cycle 5a: full mode -- loading, completeness, pairing by (K, seed), Tier 1 ---- #
def write_campaign(tmp, blocks, metrics_fn=None, waves_fn=None, skip=()):
    for r in C.select_cells(MANIFEST, blocks=blocks):
        if r["cell"] not in skip:
            write_cell(tmp, r, metrics=metrics_fn(r) if metrics_fn else None,
                       waves=waves_fn(r) if waves_fn else None)


def collect(tmp, blocks, **kw):
    kw.setdefault("dump_check", "full")
    return C.collect_cells(MANIFEST, C.select_cells(MANIFEST, blocks=blocks), farm_root=FARM,
                           dump_root=os.path.join(str(tmp), "dumps"), pin=PIN,
                           canon_index=canon_index(), base_dir=str(tmp), check_link=False,
                           no_dump_blocks=(), **kw)


INVARIANCE = ["T", "T-dup", "C4+45", "R"]


def test_collect_reports_missing_and_invalid_cells(tmp_path):
    write_campaign(tmp_path, INVARIANCE, skip={"exp25_CANON_R_S40000_K8_s46"})
    bad = row("C4+45", K=8, deg=270)
    os.unlink(os.path.join(str(tmp_path), "dumps",
                           C.cell_paths(bad, os.path.join(FARM, "CANON", ckpt_base(bad) + ".ckpt"),
                                        "d").stem, "COMPLETE"))
    cells = collect(tmp_path, INVARIANCE)
    assert len(cells) == 30
    missing, invalid = C.completeness(cells)
    assert missing == ["exp25_CANON_R_S40000_K8_s46"] and invalid == [bad["cell"]]
    assert cells[bad["cell"]]["reasons"] and not cells[bad["cell"]]["valid"]
    assert all(c["valid"] for n, c in cells.items() if n not in (missing + invalid))


def test_pairs_and_tier1(tmp_path):
    write_campaign(tmp_path, INVARIANCE + ["P1-pin"])
    cells = collect(tmp_path, INVARIANCE + ["P1-pin"])
    pairs = C.make_pairs(cells)
    kinds = collections.Counter((p["arm"], p["kind"]) for p in pairs)
    assert kinds == {("CANON", "C4"): 6, ("CANON", "45"): 2, ("CANON", "R"): 10, ("CANON", "dup"): 2,
                     ("P1", "C4"): 6, ("P1", "45"): 2, ("P1", "R"): 10}
    assert all(p["input_hash_equal"] for p in pairs)
    t1 = C.tier1(pairs, cells)
    assert t1["pass"] and t1["n_pairs"] == 20 and all(
        p["n_equal"] == p["n"] == N and p["canonical_depth_hash_equal"] for p in t1["pairs"])
    # one altered per-position depth hash (sidecar rewritten consistently) -> Tier 1 FAILS
    victim = next(p for p in pairs if p["arm"] == "CANON" and p["kind"] == "R")
    canon = cells[victim["cell"]]["canon"]
    canon["positions"][3][4] = "f" * 64
    t1 = C.tier1(pairs, cells)
    bad = next(p for p in t1["pairs"] if p["cell"] == victim["cell"])
    assert not t1["pass"] and bad["n_equal"] == N - 1 and not bad["canonical_depth_hash_equal"]


def test_a_pair_with_a_different_input_hash_is_not_a_pair(tmp_path):
    write_campaign(tmp_path, ["T", "C4+45"])
    cells = collect(tmp_path, ["T", "C4+45"])
    rot = next(n for n in cells if n.endswith("K1_s42_th90"))
    cells[rot]["stream"]["input_hash"] = "0" * 64
    pairs = C.make_pairs(cells)
    p = next(p for p in pairs if p["cell"] == rot)
    assert not p["input_hash_equal"] and not C.tier1(pairs, cells)["pass"]


# ---- cycle 5b: Tier 2 against the composition control; the probe artifact ---- #
CONTROL = {"d64": 1e-3, "d128": 2e-4, "d256": 3e-4, "d384": 4e-4, "random": 5e-4, "pooled": 6e-4}


import functools  # noqa: E402


@functools.lru_cache(maxsize=1)
def _toy_probe_run():
    """The REAL composition probe (``run_probe``) on the CPU toy stack of the probe's tests:
    5 items x the five registered offsets x both protocols (computed once per session)."""
    import test_depth_canon_cond_probe as TP
    with mock.patch("src.models.transformer.flash_attn_func", None):
        mds = [TP._probe_md(i) for i in range(5)]
        rep = TP.probe.run_probe(TP._toy_module(), torch.zeros(5, 1, 64), mds, img_w=512, seed=42,
                                 batch_size=3, latent_len=64)
    stream = eval_FLAC.RotationStream()
    for md in mds:
        stream.record(md, None, 512)
    return TP.probe, rep, stream


def _probe_prov(probe, ckpt_sha, pin):
    """The provenance main() records, for a registered run of THIS campaign (toy cardinalities)."""
    canon = MANIFEST["arms"]["CANON"]
    ds = next(c for c in MANIFEST["cells"] if c["split"] == "unseen" and c["K"] == 8)
    digest = C.model_config_digest(next(c for c in MANIFEST["cells"] if c["arm"] == "CANON"))
    return {"ckpt_path": "outputs_FLAC/exp25_depth_canon/CANON/epoch=8-step=40000.ckpt",
            "ckpt_sha256": ckpt_sha or canon_index()[40000][1],
            "model_config": os.path.join(REPO, canon["model_config"]),
            "model_config_file_sha256": canon["model_config_sha256"], "dataset_config": ds["dataset_config"],
            "dataset_config_sha256": ds["dataset_config_sha256"], "split_json": "data/AR/unseen_eval.json",
            "split_json_sha256": eval_FLAC._file_sha256(os.path.join(REPO, "data/AR/unseen_eval.json")),
            "source_sha": pin, "dirty": "", "device": "cuda", "torch": torch.__version__, "cuda_device": "toy",
            "seed": 42, "random_offset_seed": 43, "noise": "per item", "n_items": 5, "batch_size": 3,
            "num_workers": 4, "img_w": 512, "latent_shape": [4, 64], "protocols": copy.deepcopy(probe.PROTOCOLS),
            "load": {"weights_source": "ema", "canonicalize_trained": True, "canonicalize_applied": True,
                     "canonicalize_mismatch": False, "img_w": 512, "model_config_sha256_file": digest,
                     "model_config_sha256_embedded": digest},
            "allow_canon_mismatch": False}


def write_probe(tmp, monkeypatch, registered=True, mutate=None, tamper=False, ckpt_sha=None, pin=PIN):
    """A registered composition-control artifact built and published by the probe's OWN
    ``run_probe`` / ``build_report`` / ``write_report`` (rB-F1 B5); the collector's probe
    contract is re-pointed at the toy cardinalities (5 items, batch 3, K = 2 contexts)."""
    probe, rep, stream = _toy_probe_run()
    monkeypatch.setattr(C, "PROBE_CONTRACT", dict(C.PROBE_CONTRACT, n_items=5, batch_size=3, K=2))
    report = probe.build_report(registered, [], _probe_prov(probe, ckpt_sha, pin), stream, copy.deepcopy(rep))
    if mutate:
        mutate(report)
    path = os.path.join(str(tmp), "depth_canon_cond_probe.json")
    probe.write_report(report, path)
    if tamper:
        open(path + ".sha256", "w").write("0" * 64 + "  depth_canon_cond_probe.json\n")
    return path


def test_control_key_mapping():
    assert [C.control_key(d) for d in (90, 180, 270, 45, None)] == ["d128", "d256", "d384", "d64", "random"]


def _load(path):
    return C.load_control(path, ckpt_sha256=canon_index()[40000][1], pin=PIN, manifest=MANIFEST)


def _relabel_eps(r, factor):
    """Scale the fp32 conditioning relL2 lists and re-derive eps_cond + verdict with the probe's
    OWN code: an internally consistent artifact (a legitimate result, PASS or FAIL)."""
    probe = _toy_probe_run()[0]
    for d in r["results"]["fp32"].values():
        for pid in d["cond"]["per_id"].values():
            pid["relL2"] = [v * factor for v in pid["relL2"]]
    r["eps_cond"] = probe._eps_cond(r["results"])
    r["verdict"]["eps_cond"] = r["eps_cond"]["pass"]
    r["verdict"]["pass"] = all(r["verdict"][k] for k in ("eps_cond", "tier1", "masks"))


def _ctx(r, i, ids):
    r["items"][i]["context_ids"] = ids


PROBE_EDITS = {
    "unregistered": (dict(registered=False), "registered"),
    "deviation": (dict(mutate=lambda r: r["deviations"].append("x")), "deviations"),
    "tampered_sidecar": (dict(tamper=True), "sidecar"),
    "other_checkpoint": (dict(ckpt_sha="1" * 64), "ckpt_sha256"),
    "other_pin": (dict(pin="b" * 40), "source_sha"),
    "dirty_tree": (dict(mutate=lambda r: r["provenance"].update(dirty=" M src/x.py")), "dirty"),
    "n_items": (dict(mutate=lambda r: r["provenance"].update(n_items=4)), "n_items"),
    "seed": (dict(mutate=lambda r: r["provenance"].update(seed=7)), "seed"),
    "device": (dict(mutate=lambda r: r["provenance"].update(device="cpu")), "device"),
    "no_bf16_stage": (dict(mutate=lambda r: r["provenance"]["protocols"]["eval"].update(cond_autocast="off")),
                      "protocols"),
    "model_config_hash": (dict(mutate=lambda r: r["provenance"].update(model_config_file_sha256="0" * 64)),
                          "model_config_file_sha256"),
    "dataset_config_hash": (dict(mutate=lambda r: r["provenance"].update(dataset_config_sha256="0" * 64)),
                            "dataset_config_sha256"),
    "split_hash": (dict(mutate=lambda r: r["provenance"].update(split_json_sha256="0" * 64)), "split_json"),
    "load_not_ema": (dict(mutate=lambda r: r["provenance"]["load"].update(weights_source="online")), "load"),
    "offsets": (dict(mutate=lambda r: r["offsets"]["random"].__setitem__(0, 7 if r["offsets"]["random"][0] != 7
                                                                           else 8)), "offsets"),
    "item_dropped": (dict(mutate=lambda r: r["items"].pop()), "items"),
    "context_count": (dict(mutate=lambda r: _ctx(r, 1, r["items"][1]["context_ids"][:1])), "context"),
    "context_empty": (dict(mutate=lambda r: _ctx(r, 2, ["", r["items"][2]["context_ids"][1]])), "context"),
    "stream_hash": (dict(mutate=lambda r: r.update(stream_input_hash="0" * 64)), "stream_input_hash"),
    "eps_missing": (dict(mutate=lambda r: r.pop("eps_cond")), "eps_cond"),
    "eps_nan": (dict(mutate=lambda r: r["eps_cond"].update(value=float("nan"))), "eps_cond"),
    "eps_inconsistent": (dict(mutate=lambda r: r["eps_cond"].update(value=r["eps_cond"]["value"] / 2)), "eps_cond"),
    "eps_threshold": (dict(mutate=lambda r: r["eps_cond"].update(threshold=1e-4)), "threshold"),
    "eps_pass_flipped": (dict(mutate=lambda r: r["eps_cond"].update({"pass": not r["eps_cond"]["pass"]})),
                         "eps_cond"),
    "control_inconsistent": (dict(mutate=lambda r: r["control_p99"]["eval"].update(d64=0.5)), "control_p99"),
    "verdict_inconsistent": (dict(mutate=lambda r: r["verdict"].update(eps_cond=not r["verdict"]["eps_cond"])),
                             "verdict"),
}


def test_a_registered_probe_of_this_campaign_is_the_control(tmp_path, monkeypatch):
    control, reasons, eps = _load(write_probe(tmp_path, monkeypatch))
    assert reasons == [] and set(control) == {"d64", "d128", "d256", "d384", "random", "pooled"}
    rep = _toy_probe_run()[1]
    assert control["d128"] == rep["control_p99"]["eval"]["d128"] and control["pooled"] == rep["control_p99"]["eval"]["pooled"]
    assert eps == {"value": rep["eps_cond"]["value"], "threshold": 1e-5, "verdict": "PASS"}


@pytest.mark.parametrize("name", list(PROBE_EDITS))
def test_an_unbound_or_inconsistent_probe_is_not_a_control(tmp_path, monkeypatch, name):
    kw, needle = PROBE_EDITS[name]
    control, reasons, eps = _load(write_probe(tmp_path, monkeypatch, **kw))
    assert control is None and eps is None and any(needle in x for x in reasons), (name, reasons)


def test_a_legitimate_eps_fail_stays_a_scientific_fail(tmp_path, monkeypatch):
    control, reasons, eps = _load(write_probe(tmp_path, monkeypatch, mutate=lambda r: _relabel_eps(r, 1e9)))
    assert reasons == [] and control is not None                       # a valid artifact, a valid control
    assert eps["verdict"] == "FAIL" and eps["value"] > eps["threshold"] == 1e-5


def test_the_probe_contract_mirrors_the_probe():
    probe = _toy_probe_run()[0]
    P = C.PROBE_CONTRACT
    assert {k: P[k] for k in ("n_items", "batch_size", "seed", "device")} == {
        k: probe.REGISTERED[k] for k in ("n_items", "batch_size", "seed", "device")}
    assert P["protocols"] == probe.PROTOCOLS and tuple(P["offset_keys"]) == probe.OFFSET_KEYS
    assert P["eps_threshold"] == probe.EPS_COND_THRESHOLD and P["K"] == P["config_K"] == 8 and P["img_w"] == 512


def _scaled(r, eps, item=None):
    w = waveforms_for(r)
    if item is None:
        return w * (1 + eps)
    w = w.clone()
    w[item] *= (1 + eps)
    return w


def _tier2(tmp, waves):
    """Tier 2's own rules, against a fixed control (the artifact checks are tested above)."""
    write_campaign(tmp, INVARIANCE, waves_fn=lambda r: waves.get(r["cell"]))
    cells = collect(tmp, INVARIANCE)
    return C.tier2(C.make_pairs(cells), cells, dict(CONTROL))


def test_tier2_invariant_campaign_passes(tmp_path):
    t2 = _tier2(tmp_path, {})
    assert t2["pass"] and t2["determinism"]["pass"] and not t2["alarm"]["suspended"]
    assert len(t2["pairs"]) == 18 and all(p["relL2"]["max"] == 0 for p in t2["pairs"])
    assert t2["pooled"]["control_p99"] == CONTROL["pooled"]
    assert {p["control_key"] for p in t2["pairs"]} == {"d64", "d128", "d256", "d384", "random"}


def test_tier2_matched_rule_cap_and_pooled_rule(tmp_path):
    th90 = row("C4+45", K=1, deg=90)
    t2 = _tier2(tmp_path, {th90["cell"]: _scaled(th90, 1e-3)})
    bad = next(p for p in t2["pairs"] if p["cell"] == th90["cell"])
    assert abs(bad["relL2"]["p99"] - 1e-3) < 1e-6 and abs(bad["relLinf"]["max"] - 1e-3) < 1e-6
    assert not bad["pass_matched"] and bad["pass_cap"] and not t2["pass"]   # 1e-3 > 3 x 2e-4
    assert all(p["pass_matched"] for p in t2["pairs"] if p["cell"] != th90["cell"])


def test_tier2_hard_cap(tmp_path, monkeypatch):
    monkeypatch.setitem(CONTROL, "d64", 1.0)
    r45 = row("C4+45", K=8, deg=45)
    t2 = _tier2(tmp_path, {r45["cell"]: _scaled(r45, 2e-2)})
    bad = next(p for p in t2["pairs"] if p["cell"] == r45["cell"])
    assert bad["pass_matched"] and not bad["pass_cap"] and not t2["pass"]


def test_tier2_alarm_names_the_offending_items(tmp_path):
    rr = row("R", K=1, seed=45)
    t2 = _tier2(tmp_path, {rr["cell"]: _scaled(rr, 0.5, item=3)})
    assert t2["alarm"]["suspended"] and "suspended pending item triage" in t2["alarm"]["note"]
    assert t2["alarm"]["items"] == {rr["cell"]: [target_ids("unseen")[3]]}


def test_tier2_duplicate_cell_must_be_exactly_reproducible(tmp_path):
    dup = row("T-dup", K=8)
    t2 = _tier2(tmp_path, {dup["cell"]: _scaled(dup, 1e-7)})
    assert not t2["determinism"]["pass"] and t2["pass"]          # a separate requirement
    d = next(p for p in t2["determinism"]["pairs"] if p["cell"] == dup["cell"])
    assert d["relL2_max"] > 0 and not d["dump_sha_equal"]


def test_tier2_without_a_control_is_not_computable(tmp_path):
    write_campaign(tmp_path, INVARIANCE)
    cells = collect(tmp_path, INVARIANCE)
    t2 = C.tier2(C.make_pairs(cells), cells, None)
    assert t2["pass"] is None and "control" in t2["note"]


# ---- cycle 5c: Tier 3 and the table block, on known numbers ---- #
SHIFT = {"CANON": {90: 0.005, 180: -0.002, 270: 0.001, 45: 0.003},
         "P1": {90: 0.5, 180: -0.2, 270: 0.1, 45: 0.3}, "YAW": {90: 0.05, 180: -0.02, 270: 0.01, 45: 0.03}}
R_SHIFT = {"CANON": 0.001, "P1": 0.4, "YAW": 0.04}
TIER3 = ["T", "C4+45", "R", "P1-pin", "YAW-pin"]


def t3_metrics(r, shift=SHIFT, r_shift=R_SHIFT):
    v = 5.0 + 0.01 * (r["seed"] - 42) + (0.5 if r["K"] == 1 else 0.0)
    if r["rotate_mode"] == "random":
        v += r_shift[r["arm"]]
    elif r["rotate_deg"]:
        v += shift[r["arm"]][r["rotate_deg"]]
    return {k: round(v + j, 6) for j, k in enumerate(EMITTED)}


def test_tier3_on_known_numbers(tmp_path):
    write_campaign(tmp_path, TIER3, metrics_fn=t3_metrics)
    t3 = C.tier3(collect(tmp_path, TIER3))
    canon = t3["arms"]["CANON"][1]["item"]["T60"]
    assert abs(canon["delta_c4"] - 0.007) < 1e-9 and abs(canon["probe45"] - 0.003) < 1e-9
    assert abs(canon["delta_r"] - 0.001) < 1e-9 and canon["n_r_pairs"] == 5
    assert abs(canon["sd_z"] - 0.0158113883) < 1e-9
    assert abs(t3["arms"]["P1"][8]["item"]["EDT"]["delta_c4"] - 0.7) < 1e-9
    scene = t3["arms"]["CANON"][1]["scene"]["T60"]                    # by_scene = metric + 0.01 i
    assert abs(scene["delta_c4"] - 0.007) < 1e-9 and abs(scene["mean_z"] - (canon["mean_z"] + 0.045)) < 1e-9
    assert t3["pass"] is True and len(t3["checks"]) == 2 * len(C.TIER3_KEYS)
    assert all(c["pass_c4"] and c["pass_r"] for c in t3["checks"])


def test_tier3_fails_on_either_rule(tmp_path):
    weak_p1 = {**SHIFT, "P1": {90: 0.05, 180: -0.01, 270: 0.0, 45: 0.0}}      # 10 % of 0.06 < 0.007
    write_campaign(tmp_path, TIER3, metrics_fn=lambda r: t3_metrics(r, shift=weak_p1,
                                                                   r_shift={**R_SHIFT, "CANON": 0.1}))
    t3 = C.tier3(collect(tmp_path, TIER3))
    assert t3["pass"] is False
    assert not any(c["pass_c4"] for c in t3["checks"]) and not any(c["pass_r"] for c in t3["checks"])


def test_table_block_mean_and_sample_sd(tmp_path):
    write_campaign(tmp_path, TIER3, metrics_fn=t3_metrics)
    tab = C.table_block(collect(tmp_path, TIER3))
    t60 = tab["CANON"][1]["item"]["T60"]
    assert t60["n"] == 5 and abs(t60["mean"] - 5.52) < 1e-9 and abs(t60["sd"] - 0.0158113883) < 1e-9
    assert set(tab) == {"CANON", "P1", "YAW", "seen"} and tab["YAW"][8]["scene"]["C50"]["n"] == 5
    assert tab["seen"]["YAW"][1]["item"]["T60"]["n"] == 5 and "CANON" not in tab["seen"]  # S not written


# ---- cycle 5d: TRAJ grid, B-F historical row, tie counts, the seen/unseen overlap ---- #
def test_traj_grid(tmp_path):
    blocks = ["T", "C4+45", "P1-pin", "TRAJ"]
    write_campaign(tmp_path, blocks, metrics_fn=t3_metrics)
    cells = collect(tmp_path, blocks)
    t3 = C.tier3(cells)
    grid = C.traj_grid(cells, t3)
    assert [g["step"] for g in grid["grid"]] == [s for s in range(2500, 40001, 2500) for _ in (1, 8)]
    for g in grid["grid"]:
        assert abs(g["delta_c4"]["T60"] - 0.007) < 1e-9 and set(g["values"]) == {"0", "90", "180", "270"}
        assert g["within_10pct_of_p1_40k"]["T60"] is True
    last = grid["grid"][-1]
    assert last["cells"]["0"] == "exp25_CANON_T_S40000_K8_s42"          # 40k = the T / C4 cells


def test_bf_row_is_quoted_with_its_provenance():
    bf = C.bf_historical()
    assert bf["rows"][1]["values"]["T60"] == "9.543 ± 0.054" and bf["rows"][8]["values"]["T60"] == "8.202 ± 0.017"
    assert bf["rows"][8]["protocol"] == "fa eval (legacy-loop)" and bf["rows"][8]["n"] == 5
    assert "not re-measured" in bf["provenance"] and "model_comparison.md" in bf["provenance"]


# rB-F1 B1 (Codex rB BLOCKING): the tie audit is a REAL audit (the audit script's own
# pipeline) of a tiny tree whose eval splits carry exactly the cells' tie classes, and the
# collector validates it with the ONE shared gate (depth_canon_audit_gate.py).
import tempfile  # noqa: E402

import test_depth_canon_audit_gate as TG  # noqa: E402
import test_depth_canon_tie_audit as TA  # noqa: E402

_OFFICE = "Office_idx_2"
AUDIT_SPLITS = {   # unseen / seen: 8 items each = 6 'none', 1 'same_column', 1 'cross_column_image'
    "train": {"Cafe": {"Cafe_idx_0": ["S001_R001_hybrid_IR.wav", "S002_R001_hybrid_IR.wav"]}},
    "unseen": {"Office": {_OFFICE: [f"S00{s}_R005_hybrid_IR.wav" for s in range(1, 7)]
                          + ["S001_R006_hybrid_IR.wav", "S001_R007_hybrid_IR.wav"]}},
    "seen": {"Office": {_OFFICE: ["S001_R005_hybrid_IR.wav", "S007_R005_hybrid_IR.wav", "S008_R005_hybrid_IR.wav",
                                  "S009_R005_hybrid_IR.wav", "S0010_R005_hybrid_IR.wav", "S0011_R005_hybrid_IR.wav",
                                  "S002_R006_hybrid_IR.wav", "S002_R007_hybrid_IR.wav"]}}}
SAME_MAP, CROSS_MAP = f"depth_map/Office/{_OFFICE}/6.npy", f"depth_map/Office/{_OFFICE}/7.npy"


def write_audit(tmp, monkeypatch, mutate=None, tamper=False, maps=None, limit=None):
    """Publish a real audit at ``tmp/depth_canon_tie_audit.json`` with the audit's own writer
    (``mutate`` edits the report first, so its sidecar still matches) and point the gate's
    registered contract at this tree."""
    tree = tempfile.mkdtemp(prefix="audit_tree_", dir=str(tmp))
    planted = {SAME_MAP: TA._fixture("s", [(10, 77), (200, 77)], None).numpy(),
               CROSS_MAP: TA._fixture("c", [(90, 5), (90, 6)], None).numpy(), **(maps or {})}
    _, rep, contract = TG.make_audit(__import__("pathlib").Path(tree), splits=AUDIT_SPLITS, maps=planted,
                                     limit=limit)
    if mutate:
        mutate(rep)
    path = os.path.join(str(tmp), "depth_canon_tie_audit.json")
    TA.audit.write_report(rep, path)
    if tamper:
        open(path + ".sha256", "w").write("0" * 64 + "  depth_canon_tie_audit.json\n")
    monkeypatch.setattr(C.audit_gate(), "REGISTERED", contract)
    return path


def _plain_map():
    return TA._fixture("u", [], None, seed=11).numpy()


def test_tie_counts_per_cell_and_per_split(tmp_path, monkeypatch):
    write_campaign(tmp_path, ["T", "R", "S"])
    cells = collect(tmp_path, ["T", "R", "S"])
    audit, summary, reasons = C.load_tie_audit(write_audit(tmp_path, monkeypatch))
    assert reasons == [] and summary["n_maps"] == 4
    ties = C.tie_summary(cells, audit, summary)
    assert len(ties["per_cell"]) == 30 and ties["mismatches"] == [] and ties["shared_maps"] == 3
    assert ties["per_split"]["unseen"]["items"]["cross_column_image"] == 1
    assert ties["per_split"]["train"]["items"]["none"] == 2
    audit, summary, _ = C.load_tie_audit(write_audit(tmp_path, monkeypatch, maps={SAME_MAP: _plain_map()}))
    assert len(C.tie_summary(cells, audit, summary)["mismatches"]) == 30      # unseen AND seen cells
    for kw, needle in ((dict(tamper=True), "sidecar"), (dict(mutate=lambda r: r["gate"].update(verdict="FAIL")),
                                                         "gate")):
        audit, summary, reasons = C.load_tie_audit(write_audit(tmp_path, monkeypatch, **kw))
        assert audit is None and summary is None and any(needle in x for x in reasons), reasons


AUDIT_CASES = {   # every one must refuse FULL collection: exit 3, neither result file written
    "drop_train_per_split": dict(mutate=lambda r: r["per_split"].pop("train")),
    "item_count": dict(mutate=lambda r: r["per_split"]["unseen"]["items"].update(none=5)),
    "cardinality": dict(mutate=lambda r: r["inputs"][1].update(n_items=7)),
    "input_hash": dict(mutate=lambda r: r["inputs"][2].update(json_sha256="0" * 64)),
    "tool_hash": dict(mutate=lambda r: r["tool"]["code_sha256"].update({"src/data/dataset.py": "0" * 64})),
    "survivor": dict(mutate=lambda r: r["survivors"].append({"path": r["rows"][0]["path"]})),
    "coverage_incomplete": dict(mutate=lambda r: r["coverage"].update(complete=False)),
    "limit_run": dict(limit=2),
    "tampered_sidecar": dict(tamper=True),
    "cell_disagrees_with_the_audit": dict(maps={SAME_MAP: _plain_map()}),
}
INVARIANCE_S = ["T", "T-dup", "C4+45", "R", "S"]


def test_a_genuine_audit_lets_full_collection_through(tmp_path, monkeypatch, capsys):
    write_campaign(tmp_path, INVARIANCE_S)
    write_probe(tmp_path, monkeypatch)
    write_audit(tmp_path, monkeypatch)
    rc, out = _collect_cli(tmp_path, blocks=",".join(INVARIANCE_S))
    assert rc == 0, capsys.readouterr().out
    res = json.load(open(out + ".json"))
    assert res["partial"] is False and res["ties"]["mismatches"] == [] and res["ties"]["shared_maps"] == 3


@pytest.mark.parametrize("name", list(AUDIT_CASES))
def test_an_invalid_audit_or_a_disagreeing_cell_refuses_full_collection(tmp_path, monkeypatch, capsys, name):
    write_campaign(tmp_path, INVARIANCE_S)
    write_probe(tmp_path, monkeypatch)
    write_audit(tmp_path, monkeypatch, **AUDIT_CASES[name])
    rc, out = _collect_cli(tmp_path, blocks=",".join(INVARIANCE_S))
    text = capsys.readouterr().out
    assert rc == 3 and not os.path.exists(out + ".md") and not os.path.exists(out + ".json"), text
    assert "tie" in text and "RESULTS REFUSED" in text, text


def test_an_invalid_audit_is_labelled_under_partial(tmp_path, monkeypatch):
    write_campaign(tmp_path, INVARIANCE_S)
    write_probe(tmp_path, monkeypatch)
    write_audit(tmp_path, monkeypatch, maps={SAME_MAP: _plain_map()})
    rc, out = _collect_cli(tmp_path, "--partial", blocks=",".join(INVARIANCE_S))
    res, md = json.load(open(out + ".json")), open(out + ".md").read()
    assert rc == 0 and res["partial"] is True and any("tie" in p for p in res["problems"])
    assert len(res["ties"]["mismatches"]) == 40 and "PARTIAL" in md        # T, T-dup, C4+45, R, S


def test_the_seen_unseen_overlap_disclosure():
    ov = C.split_overlap()
    assert (ov["unseen_items"], ov["seen_items"], ov["shared_items"]) == (6337, 6217, 1093)
    assert "not be presented as independent estimates" in ov["disclosure"]
    assert "17.2 %" in ov["disclosure"] and "17.6 %" in ov["disclosure"]


# ---- cycle 5e: results.{md,json}, the refusal and --partial ---- #
def _collect_cli(tmp, *extra, blocks=None):
    cells_json = os.path.join(str(tmp), "cells.json")
    with open(cells_json, "w") as f:
        json.dump(MANIFEST, f)
    out = os.path.join(str(tmp), "results")
    argv = ["--collect", "--pin", PIN, "--cells", cells_json, "--farm-root", FARM,
            "--dump-root", os.path.join(str(tmp), "dumps"), "--base-dir", str(tmp),
            "--canon-ckpt-manifest", _canon_manifest_file(tmp), "--no-link-check",
            "--probe-json", os.path.join(str(tmp), "depth_canon_cond_probe.json"),
            "--tie-audit-json", os.path.join(str(tmp), "depth_canon_tie_audit.json"),
            "--out-md", out + ".md", "--out-json", out + ".json"] + (["--blocks", blocks] if blocks else [])
    return C.main(argv + list(extra)), out


ALL_BLOCKS = list(MANIFEST["blocks"])


def test_full_collection_writes_both_results_files(tmp_path, monkeypatch, capsys):
    write_campaign(tmp_path, ALL_BLOCKS, metrics_fn=t3_metrics)
    write_probe(tmp_path, monkeypatch)
    write_audit(tmp_path, monkeypatch)
    rc, out = _collect_cli(tmp_path)
    assert rc == 0, capsys.readouterr().out
    res = json.load(open(out + ".json"))
    md = open(out + ".md").read()
    assert res["partial"] is False and res["missing"] == [] and res["pin"] == PIN
    assert res["tier1"]["pass"] and res["tier2"]["pass"] and res["tier3"]["pass"] is True
    assert len(res["traj"]["grid"]) == 32 and res["table"]["CANON"]["1"]["item"]["T60"]["n"] == 5
    assert "PARTIAL" not in md and "not be presented as independent estimates" in md
    assert "eps_cond (fp32 conditioning, max relL2) = " in md and "registered threshold 1e-05: PASS" in md
    for needle in ("Tier 1", "Tier 2", "Tier 3", "eps_cond", "B-F", "re-measured at the exp_25 pin",
                   "TRAJ", "tie classes", "D-i", "D-ii", "item-weighted", "per-scene"):
        assert needle in md, needle


def test_missing_cells_refuse_the_table_unless_partial(tmp_path, monkeypatch, capsys):
    gone = "exp25_YAW_C4_S40000_K8_s42_th45"
    write_campaign(tmp_path, ALL_BLOCKS, skip={gone})
    write_probe(tmp_path, monkeypatch, registered=False)
    write_audit(tmp_path, monkeypatch)
    rc, out = _collect_cli(tmp_path)
    assert rc == 3 and not os.path.exists(out + ".md") and not os.path.exists(out + ".json")
    text = capsys.readouterr().out
    assert gone in text and "registered" in text
    rc, out = _collect_cli(tmp_path, "--partial")
    assert rc == 0
    md, res = open(out + ".md").read(), json.load(open(out + ".json"))
    assert res["partial"] is True and res["missing"] == [gone]
    headings = [ln for ln in md.splitlines() if ln.startswith("## ")]
    assert headings and all("PARTIAL" in h for h in headings) and gone in md


def test_cells_of_two_campaign_digests_refuse_full_collection(tmp_path, monkeypatch, capsys):
    """rB-F1 B3: one campaign = one evaluated tree; a cell measured under another digest is
    individually consistent but the collection is not."""
    write_campaign(tmp_path, INVARIANCE_S)
    write_probe(tmp_path, monkeypatch)
    write_audit(tmp_path, monkeypatch)
    odd = row("R", K=8, seed=45)
    rel = C.cell_paths(odd, os.path.join(FARM, "CANON", ckpt_base(odd) + ".ckpt"), "d").receipt
    edit_json(os.path.join(str(tmp_path), rel),
              lambda d: d.update(campaign_digest="e" * 64, spawn_digest="e" * 64, post_digest="e" * 64))
    rc, out = _collect_cli(tmp_path, blocks=",".join(INVARIANCE_S))
    text = capsys.readouterr().out
    assert rc == 3 and "campaign digest" in text and not os.path.exists(out + ".json"), text


def test_a_block_selection_defines_the_required_set(tmp_path, monkeypatch, capsys):
    blocks = [b for b in ALL_BLOCKS if b != "YAW-pin"]                       # the A1 switch off
    write_campaign(tmp_path, blocks, metrics_fn=t3_metrics)
    write_probe(tmp_path, monkeypatch)
    write_audit(tmp_path, monkeypatch)
    rc, out = _collect_cli(tmp_path, blocks=",".join(blocks))
    assert rc == 0, capsys.readouterr().out
    res = json.load(open(out + ".json"))
    assert res["blocks"] == blocks and res["partial"] is False
