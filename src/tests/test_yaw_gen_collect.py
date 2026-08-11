"""TDD for the exp_14 collector — ``yaw_gen_collect.py`` (plan §5.6).

One test group per function of the plan's §5.6 table, written RED first. The
collector's whole job is to refuse to produce a number it cannot prove, so most
of what is pinned here is a REFUSAL: a cell that fails exp_14's own per-cell
validator is never aggregated, a block with four seeds renders PENDING rather
than a four-seed mean, and a §3.3 hash inequality renders the affected contrast
BLOCKED rather than a plausible-looking contrast between two differently-rotated
galleries.

The synthetic artifact factory below writes REAL three-file cells (metrics
record + ``.screenmeta.json`` + ``.stream.json``) exactly where
``exp14_validate_cell`` expects them, at a small stream count — the count is a
parameter of every check, so an 8-position split exercises the same code path as
the campaign's 6,337 without writing 106 × 6,337 tuples in a unit test.
"""
import copy
import importlib.util
import json
import math
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # src/tests/ -> src/ -> repo root
_EXPDIR = os.path.join(_REPO_ROOT, "worklog", "worklog_yixun", "exp_14_yaw_gen_claude")
_COLLECT_PY = os.path.join(_EXPDIR, "yaw_gen_collect.py")

# The pre-registered golden assignment (plan §4 gate G3) — imported from the
# round-1 suite rather than re-typed, so there is exactly one copy of it.
from test_yaw_random_eval import GOLDEN_SEED42_W512  # noqa: E402


def _load_collector():
    spec = importlib.util.spec_from_file_location("yaw_gen_collect", _COLLECT_PY)
    assert spec is not None and spec.loader is not None, f"cannot load {_COLLECT_PY}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load_collector()
V = C.V                                   # the round-2 validator, imported not copied

PIN = "a" * 40
COUNT = 8                                 # the test split: 8 positions, not 6,337
CKPT_SHA = {arm: f"{i}" * 64 for i, arm in enumerate(V.ARMS, start=1)}


# --------------------------------------------------------------------------- #
# synthetic artifacts
# --------------------------------------------------------------------------- #
def _targets(count=COUNT):
    """The split manifest order — identical for every cell, as the campaign is."""
    return [f"room{i % 3}/rir_{i:04d}.wav" for i in range(count)]


def _context_ids(i, k):
    """The stochastic context draw's fingerprint: same items in every arm."""
    return [f"ctx_{i:04d}_{j}" for j in range(k)]


def _offsets(cell, count=COUNT):
    """What the cell's rotation actually applied, per position."""
    if cell.cell == "rgen":
        return C.golden_offsets(int(cell.seed), count, V.IMG_W)
    if cell.cell == "vctl":
        return [V.expected_column_shift(cell.rotate_deg)] * count
    return [0] * count


def synthetic_metrics(cell):
    """Plausible metric values with a per-seed jitter and a designed V-cell story.

    The numbers exist to exercise the gates: the Cn validity controls sit ON their
    unrotated reference (in-group invariance), VANL@90 degrades far past its own
    seed noise (the positive control), and the random-yaw cells degrade in an
    order that makes the endpoint contrasts non-degenerate.
    """
    orbit = V.TRAIN_ORBIT[cell.arm]
    jitter = ((int(cell.seed) - 42) - 2) * 0.01          # -0.02 .. +0.02
    base = {"T60": 9.0, "C50": 1.0, "EDT": 43.0, "FD": 0.34,
            "RIR_to_GT_RIR_R@1": 5.0, "RIR_to_GT_RIR_R@5": 15.0,
            "RIR_to_GT_RIR_R@10": 23.0,
            "RIR_to_geom_R@1": 4.0, "RIR_to_geom_R@5": 13.0,
            "RIR_to_geom_R@10": 20.0, "Invalid T60": 0.0}
    out = {k: (v + jitter * (1.0 if k != "FD" else 0.1) if k != "Invalid T60" else v)
           for k, v in base.items()}
    if cell.cell == "rgen":                              # random yaw: everyone degrades
        penalty = 3.0 / (1.0 + orbit / 4.0) if orbit else 3.0
        out["T60"] += penalty
        out["RIR_to_GT_RIR_R@1"] -= penalty / 3.0
        out["FD"] += penalty / 50.0
    if cell.cell == "vctl":
        if cell.arm == "VANL":                           # the positive control
            out["T60"] += 3.4
            out["RIR_to_GT_RIR_R@1"] -= 1.0
        elif float(cell.rotate_deg) == 45.0:             # off-group mechanism control
            out["T60"] += 0.6
    return out


def write_cell(root, cell, *, pin=PIN, count=COUNT, metrics=None, record_patch=None,
               meta_patch=None, stream_patch=None, offsets=None, targets=None,
               ckpt_sha=None):
    """Write one registered cell's three artifacts under a fake ``--output-root``.

    Returns the metrics path. Every ``*_patch`` is applied last, so a test can
    break exactly one field and keep the rest provably well-formed.
    """
    ck_dir = os.path.join(root, f"exp11_{cell.arm}", f"FLAC_exp11_{cell.arm}",
                          f"exp11_{cell.arm}", "checkpoints")
    os.makedirs(ck_dir, exist_ok=True)
    ckpt = os.path.join(ck_dir, f"epoch=8-step={int(cell.step)}.ckpt")
    if not os.path.exists(ckpt):
        with open(ckpt, "wb") as fh:
            fh.write(b"synthetic checkpoint")
    metrics_path = V.metrics_path(ckpt, cell)
    mode, deg, rseed = V.rotation_expectation(cell)
    tgts = list(targets or _targets(count))
    offs = list(offsets if offsets is not None else _offsets(cell, count))
    inp = [[i, tgts[i], _context_ids(i, int(cell.k)), V.IMG_W] for i in range(count)]
    asg = [[i, tgts[i], offs[i]] for i in range(count)]
    stream = {
        "schema_version": V.STREAM_SCHEMA_VERSION,
        "fingerprint_schema": V.FINGERPRINT_SCHEMA,
        "rotate_mode": mode, "rotate_seed": rseed, "rotate_deg": deg,
        "img_w": V.IMG_W, "stream_count": count,
        "input_tuples": inp, "offsets": offs, "assignment_tuples": asg,
        "input_hash": V.canonical_stream_hash(inp),
        "assignment_hash": V.canonical_stream_hash(asg),
    }
    record = {
        "metrics": dict(metrics if metrics is not None else synthetic_metrics(cell)),
        "ckpt_path": ckpt, "rotate_deg": deg,
        "cond_method": V.cond_method(cell.arm),
        "frame_avg_angles": V.frame_avg_angles(cell.arm),
        "cond_autocast": V.COND_AUTOCAST,
        "orbit_execution": "n/a" if cell.arm == "VANL" else "batched",
        "frame_avg_fwd_cap": None if cell.arm == "VANL" else 64,
        "source_sha": pin, "batch_size": V.BATCH_SIZE, "n_samples": count,
        "dataset_config": "src/configs/dataset_configs/AR/eval/" + (
            V.SPLIT_K8 if int(cell.k) == 8 else V.SPLIT_K1),
        "seed": int(cell.seed), "cfg_scale": V.CFG_SCALE, "steps": V.STEPS,
        "eval_name": V.eval_name(cell), "weights_source": "ema", "device": "cuda",
    }
    if mode == "random":
        record["rotate_deg"] = None
        record.update({"rotate_mode": mode, "rotate_seed": rseed,
                       "input_hash": stream["input_hash"],
                       "assignment_hash": stream["assignment_hash"],
                       "stream_count": count, "img_w": V.IMG_W})
    meta = {
        "arm": cell.arm, "cell": cell.cell, "step": int(cell.step),
        "seed": int(cell.seed), "K": int(cell.k), "eval_name": V.eval_name(cell),
        "cfg_scale": V.CFG_SCALE, "steps": V.STEPS,
        "model_config": f"worklog/.../FLAC_AR_BF_{cell.arm}.json",
        "model_config_sha256": "c" * 64,
        "dataset_config": record["dataset_config"],
        "ckpt_path": ckpt, "ckpt_sha256": ckpt_sha or CKPT_SHA[cell.arm],
        "use_ema": True, "frame_avg_angles": V.frame_avg_angles(cell.arm),
        "cond_method": V.cond_method(cell.arm), "cond_autocast": V.COND_AUTOCAST,
        "commit": pin, "training_orbit": V.TRAIN_ORBIT[cell.arm],
        "eval_orbit": V.TRAIN_ORBIT[cell.arm],
        "rotate_mode": mode, "rotate_deg": deg, "rotate_seed": rseed,
        "expected_stream_count": count, "record_stream": True,
        "stream_sidecar": os.path.basename(metrics_path)[:-len(".json")] + ".stream.json",
        "batch_size": V.BATCH_SIZE, "num_workers": V.NUM_WORKERS,
    }
    for payload, patch in ((record, record_patch), (meta, meta_patch),
                           (stream, stream_patch)):
        if patch:
            payload.update(patch)
    with open(metrics_path, "w") as fh:
        json.dump(record, fh)
    with open(V.screenmeta_path(metrics_path), "w") as fh:
        json.dump(meta, fh)
    with open(V.stream_path(metrics_path), "w") as fh:
        json.dump(stream, fh)
    return metrics_path


def write_grid(root, cells=None, **kw):
    """Write a whole (sub)grid; returns ``{cell: metrics_path}``."""
    return {c: write_cell(root, c, **kw) for c in (cells or V.expected_grid())}


def expectation(pin=PIN, count=COUNT, sha=None):
    return C.CampaignExpectation(pin=pin, ckpt_sha_by_arm=dict(sha or CKPT_SHA),
                                 expected_count=count)


def a_cell(arm="C8", cell="zref", seed=42, k=8, deg=None):
    return V.Cell(arm, cell, V.STEP, seed, k, deg)


# --------------------------------------------------------------------------- #
# 1. expected_grid — the registered campaign, re-exported not re-derived
# --------------------------------------------------------------------------- #
def test_expected_grid_is_the_registered_106_cells():
    grid = C.expected_grid()
    assert len(grid) == 106 and len(set(grid)) == 106
    assert sum(1 for c in grid if c.cell == "rgen") == 50
    assert sum(1 for c in grid if c.cell == "zref") == 50
    assert sum(1 for c in grid if c.cell == "vctl") == 6


def test_expected_grid_is_the_validators_grid_not_a_second_copy():
    """The collector must not carry its own idea of the campaign: a grid that
    drifts from the submitter's is a grid that reports on cells nobody ran."""
    assert C.expected_grid() == V.expected_grid()


def test_expected_grid_v_tuples_are_exact():
    v = {(c.arm, float(c.rotate_deg)) for c in C.expected_grid() if c.cell == "vctl"}
    assert v == {("C4L", 90.0), ("C8", 90.0), ("C16", 90.0), ("C32", 90.0),
                 ("VANL", 90.0), ("C4L", 45.0)}
    assert all(c.seed == 42 and c.k == 8 for c in C.expected_grid() if c.cell == "vctl")


def test_expected_grid_admits_no_unregistered_combination():
    grid = set(C.expected_grid())
    assert V.Cell("VANL", "vctl", V.STEP, 42, 8, 45.0) not in grid   # no vanilla off-group
    assert V.Cell("C8", "rgen", V.STEP, 47, 8, None) not in grid     # seed 47
    assert V.Cell("C8", "zref", V.STEP, 42, 4, None) not in grid     # K=4
    assert V.Cell("C8", "vctl", V.STEP, 43, 8, 90.0) not in grid     # vctl seed 43


# --------------------------------------------------------------------------- #
# 2. parse_cell_artifact
# --------------------------------------------------------------------------- #
def test_parse_reads_a_well_formed_cell(tmp_path):
    cell = a_cell()
    path = write_cell(str(tmp_path), cell)
    art = C.parse_cell_artifact(path)
    assert art.cell == cell
    assert art.record["eval_name"] == V.eval_name(cell)
    assert art.stream["stream_count"] == COUNT
    assert art.screenmeta["arm"] == cell.arm


def test_parse_rejects_a_missing_artifact(tmp_path):
    with pytest.raises(C.ArtifactError) as exc:
        C.parse_cell_artifact(str(tmp_path / "nothing.json"))
    assert "missing" in str(exc.value)


def test_parse_rejects_malformed_json(tmp_path):
    path = write_cell(str(tmp_path), a_cell())
    with open(path, "w") as fh:
        fh.write("{not json")
    with pytest.raises(C.ArtifactError) as exc:
        C.parse_cell_artifact(path)
    assert "parse" in str(exc.value).lower()


def test_parse_rejects_a_non_object_payload(tmp_path):
    path = write_cell(str(tmp_path), a_cell())
    with open(path, "w") as fh:
        json.dump([1, 2, 3], fh)
    with pytest.raises(C.ArtifactError) as exc:
        C.parse_cell_artifact(path)
    assert "object" in str(exc.value)


def test_parse_rejects_a_record_missing_required_keys(tmp_path):
    path = write_cell(str(tmp_path), a_cell())
    rec = json.load(open(path))
    del rec["eval_name"]
    json.dump(rec, open(path, "w"))
    with pytest.raises(C.ArtifactError) as exc:
        C.parse_cell_artifact(path)
    assert "eval_name" in str(exc.value)

    path2 = write_cell(str(tmp_path), a_cell(arm="C16"))
    rec = json.load(open(path2))
    del rec["metrics"]
    json.dump(rec, open(path2, "w"))
    with pytest.raises(C.ArtifactError) as exc:
        C.parse_cell_artifact(path2)
    assert "metrics" in str(exc.value)


def test_parse_rejects_an_unregistered_eval_name(tmp_path):
    path = write_cell(str(tmp_path), a_cell())
    rec = json.load(open(path))
    rec["eval_name"] = "exp14_C8_zref_S40000_s47_K8"        # seed 47: not registered
    json.dump(rec, open(path, "w"))
    with pytest.raises(C.ArtifactError) as exc:
        C.parse_cell_artifact(path)
    assert "UNREGISTERED" in str(exc.value) or "unregistered" in str(exc.value)


def test_parse_rejects_a_stream_from_another_schema_version(tmp_path):
    path = write_cell(str(tmp_path), a_cell(),
                      stream_patch={"schema_version": V.STREAM_SCHEMA_VERSION + 1})
    with pytest.raises(C.ArtifactError) as exc:
        C.parse_cell_artifact(path)
    assert "schema" in str(exc.value)


def test_parse_rejects_a_fingerprint_from_another_schema_version(tmp_path):
    path = write_cell(str(tmp_path), a_cell(),
                      stream_patch={"fingerprint_schema": V.FINGERPRINT_SCHEMA + 9})
    with pytest.raises(C.ArtifactError) as exc:
        C.parse_cell_artifact(path)
    assert "fingerprint" in str(exc.value)


def test_parse_rejects_missing_sidecars(tmp_path):
    path = write_cell(str(tmp_path), a_cell())
    os.remove(V.screenmeta_path(path))
    with pytest.raises(C.ArtifactError) as exc:
        C.parse_cell_artifact(path)
    assert "screenmeta" in str(exc.value)

    path2 = write_cell(str(tmp_path), a_cell(arm="C32"))
    os.remove(V.stream_path(path2))
    with pytest.raises(C.ArtifactError) as exc:
        C.parse_cell_artifact(path2)
    assert "stream" in str(exc.value)


# --------------------------------------------------------------------------- #
# 3. validate_cell_provenance — every campaign pin, by name
# --------------------------------------------------------------------------- #
def test_expectation_cannot_exist_without_a_pin_or_a_checkpoint_digest():
    """Fail-closed by construction: a collector run that never learned which
    commit and which checkpoint produced its numbers cannot even ask the
    question, let alone answer it with a mean."""
    with pytest.raises(ValueError):
        C.CampaignExpectation(pin=None, ckpt_sha_by_arm=CKPT_SHA)
    with pytest.raises(ValueError):
        C.CampaignExpectation(pin=PIN, ckpt_sha_by_arm={})
    with pytest.raises(ValueError):
        C.CampaignExpectation(pin=PIN, ckpt_sha_by_arm={"C8": None})


def test_validate_accepts_a_good_cell(tmp_path):
    cell = a_cell()
    art = C.parse_cell_artifact(write_cell(str(tmp_path), cell))
    assert C.validate_cell_provenance(art, expectation().for_cell(cell)) == []


@pytest.mark.parametrize("patch,needle", [
    ({"source_sha": "f" * 40}, "source_sha"),
    ({"cond_method": "vanilla"}, "cond_method"),
    ({"frame_avg_angles": [0.0, 90.0]}, "frame_avg_angles"),
    ({"cond_autocast": "default"}, "cond_autocast"),
    ({"batch_size": 32}, "batch_size"),
    ({"n_samples": COUNT - 1}, "n_samples"),
    ({"weights_source": "online"}, "weights_source"),
    ({"cfg_scale": 3.0}, "cfg_scale"),
    ({"steps": 8}, "steps"),
])
def test_validate_names_every_broken_record_field(tmp_path, patch, needle):
    cell = a_cell()
    art = C.parse_cell_artifact(write_cell(str(tmp_path), cell, record_patch=patch))
    reasons = C.validate_cell_provenance(art, expectation().for_cell(cell))
    assert reasons and any(needle in r for r in reasons), reasons


@pytest.mark.parametrize("patch,needle", [
    ({"num_workers": 6}, "num_workers"),
    ({"batch_size": 16}, "batch_size"),
    ({"ckpt_sha256": "0" * 64}, "ckpt_sha256"),
    ({"commit": "9" * 40}, "commit"),
    ({"record_stream": False}, "record_stream"),
    ({"use_ema": False}, "use_ema"),
    ({"expected_stream_count": 3}, "expected_stream_count"),
])
def test_validate_names_every_broken_manifest_pin(tmp_path, patch, needle):
    cell = a_cell()
    art = C.parse_cell_artifact(write_cell(str(tmp_path), cell, meta_patch=patch))
    reasons = C.validate_cell_provenance(art, expectation().for_cell(cell))
    assert reasons and any(needle in r for r in reasons), reasons


def test_validate_rejects_a_short_stream(tmp_path):
    """The campaign's estimand is the FULL split; a cell that evaluated fewer
    positions is a different measurement wearing the same name."""
    cell = a_cell()
    path = write_cell(str(tmp_path), cell)
    stream = json.load(open(V.stream_path(path)))
    for key in ("input_tuples", "assignment_tuples", "offsets"):
        stream[key] = stream[key][:-1]
    stream["stream_count"] = COUNT - 1
    stream["input_hash"] = V.canonical_stream_hash(stream["input_tuples"])
    stream["assignment_hash"] = V.canonical_stream_hash(stream["assignment_tuples"])
    json.dump(stream, open(V.stream_path(path), "w"))
    art = C.parse_cell_artifact(path)
    reasons = C.validate_cell_provenance(art, expectation().for_cell(cell))
    assert reasons and any("positions" in r or "stream_count" in r for r in reasons), reasons


def test_validate_rejects_a_random_cell_whose_record_and_sidecar_disagree(tmp_path):
    cell = a_cell(cell="rgen")
    art = C.parse_cell_artifact(
        write_cell(str(tmp_path), cell, record_patch={"input_hash": "b" * 64}))
    reasons = C.validate_cell_provenance(art, expectation().for_cell(cell))
    assert reasons and any("input_hash" in r for r in reasons), reasons


def test_load_cell_is_parse_plus_validate(tmp_path):
    """The one entry point the collector actually uses: a cell that cannot be
    proven comes back as reasons, never as data."""
    good = a_cell()
    data, reasons = C.load_cell(write_cell(str(tmp_path), good), expectation())
    assert reasons == [] and data is not None and data.cell == good
    assert data.metrics["T60"] == pytest.approx(synthetic_metrics(good)["T60"])

    bad = a_cell(arm="C16")
    path = write_cell(str(tmp_path), bad, record_patch={"batch_size": 8})
    data, reasons = C.load_cell(path, expectation())
    assert data is None and reasons and any("batch_size" in r for r in reasons)


# --------------------------------------------------------------------------- #
# 4. match_assignments — the §3.3 equalities
# --------------------------------------------------------------------------- #
def _load_all(root, cells, exp=None):
    exp = exp or expectation()
    out = []
    for c in cells:
        data, reasons = C.load_cell(V.metrics_path(
            V.checkpoint_path(root, c.arm, c.step), c), exp)
        assert reasons == [], (c, reasons)
        out.append(data)
    return out


def _one_seed_slice(seed=42, k=8):
    return [c for c in V.expected_grid()
            if c.cell in ("rgen", "zref") and c.seed == seed and c.k == k]


def test_match_assignments_accepts_a_rotation_matched_slice(tmp_path):
    root = str(tmp_path)
    cells = _one_seed_slice()
    write_grid(root, cells)
    assert C.match_assignments(_load_all(root, cells)) == []


def test_match_assignments_names_a_cross_arm_input_hash_violation(tmp_path):
    """One arm evaluated a different context draw: the arms are no longer
    comparable at that (K, seed), and no cross-arm number may be printed."""
    root = str(tmp_path)
    cells = _one_seed_slice()
    write_grid(root, [c for c in cells if c.arm != "C32"])
    for c in [c for c in cells if c.arm == "C32"]:
        write_cell(root, c, targets=[f"OTHER/rir_{i}.wav" for i in range(COUNT)])
    violations = C.match_assignments(_load_all(root, cells))
    assert violations, "a differing input_hash across arms went undetected"
    assert any(v.kind == "cross_arm_input_hash" and "C32" in v.detail for v in violations)


def test_match_assignments_names_a_cross_arm_assignment_hash_violation(tmp_path):
    """Same items, different yaw draw: the rotated cells are not rotation-matched."""
    root = str(tmp_path)
    cells = _one_seed_slice()
    write_grid(root, [c for c in cells if not (c.arm == "C8" and c.cell == "rgen")])
    for c in [c for c in cells if c.arm == "C8" and c.cell == "rgen"]:
        write_cell(root, c, offsets=list(reversed(C.golden_offsets(c.seed, COUNT, V.IMG_W))))
    violations = C.match_assignments(_load_all(root, cells))
    assert any(v.kind == "cross_arm_assignment_hash" for v in violations), violations


def test_match_assignments_names_a_broken_z_r_pairing(tmp_path):
    """Z and R differ only in rotation; a differing input_hash means the pair is
    not a pair, so the arm's Δ is unmeasurable at that seed."""
    root = str(tmp_path)
    cells = _one_seed_slice()
    write_grid(root, [c for c in cells if not (c.arm == "C4L" and c.cell == "zref")])
    for c in [c for c in cells if c.arm == "C4L" and c.cell == "zref"]:
        write_cell(root, c, targets=[f"shifted/rir_{i}.wav" for i in range(COUNT)])
    violations = C.match_assignments(_load_all(root, cells))
    kinds = {v.kind for v in violations}
    assert "z_r_input_hash" in kinds, violations
    assert any("C4L" in v.detail for v in violations)


def test_match_assignments_ignores_singleton_groups(tmp_path):
    """C4L@45 is the only cell in its group — nothing to compare it against, and
    an unmatched singleton is not a violation."""
    root = str(tmp_path)
    cell = V.Cell("C4L", "vctl", V.STEP, 42, 8, 45.0)
    write_cell(root, cell)
    assert C.match_assignments(_load_all(root, [cell])) == []


# --------------------------------------------------------------------------- #
# 5. pair_seeds
# --------------------------------------------------------------------------- #
def _fake_data(cell):
    return C.CellData(cell=cell, path=f"/synthetic/{V.eval_name(cell)}.json",
                      metrics=synthetic_metrics(cell), input_hash="i" * 64,
                      assignment_hash="a" * 64, offsets=(), source_sha=PIN)


def test_pair_seeds_pairs_the_five_registered_seeds():
    z = [_fake_data(a_cell(cell="zref", seed=s)) for s in V.SEEDS]
    r = [_fake_data(a_cell(cell="rgen", seed=s)) for s in V.SEEDS]
    pairs, problems = C.pair_seeds(z, r)
    assert problems == []
    assert sorted(pairs) == list(V.SEEDS)
    assert all(pairs[s][0].cell.cell == "zref" and pairs[s][1].cell.cell == "rgen"
               for s in pairs)


def test_pair_seeds_reports_a_missing_seed():
    z = [_fake_data(a_cell(cell="zref", seed=s)) for s in V.SEEDS]
    r = [_fake_data(a_cell(cell="rgen", seed=s)) for s in V.SEEDS if s != 45]
    pairs, problems = C.pair_seeds(z, r)
    assert 45 not in pairs
    assert problems and any("45" in p for p in problems)


def test_pair_seeds_rejects_a_duplicate_seed():
    z = [_fake_data(a_cell(cell="zref", seed=42))] * 2 + \
        [_fake_data(a_cell(cell="zref", seed=s)) for s in V.SEEDS if s != 42]
    r = [_fake_data(a_cell(cell="rgen", seed=s)) for s in V.SEEDS]
    pairs, problems = C.pair_seeds(z, r)
    assert problems and any("duplicate" in p.lower() for p in problems)
    assert 42 not in pairs, "a duplicated seed must not silently pick a winner"


def test_pair_seeds_rejects_orphans_from_another_arm_or_k():
    z = [_fake_data(a_cell(cell="zref", seed=s)) for s in V.SEEDS]
    r = [_fake_data(a_cell(cell="rgen", seed=s)) for s in V.SEEDS if s != 46]
    r.append(_fake_data(a_cell(arm="C32", cell="rgen", seed=46)))     # wrong arm
    pairs, problems = C.pair_seeds(z, r)
    assert problems and any("C32" in p or "arm" in p for p in problems)
    assert 46 not in pairs


# --------------------------------------------------------------------------- #
# 6. aggregate_cell — 5/5 or PENDING, never a partial mean
# --------------------------------------------------------------------------- #
def test_aggregate_cell_requires_all_five_seeds():
    partial = [_fake_data(a_cell(cell="rgen", seed=s)) for s in (42, 43, 44, 45)]
    agg = C.aggregate_cell(partial)
    assert agg.status == "PENDING" and agg.n == 4
    assert agg.values == {} and agg.per_seed == {}, (
        "a partial block must not carry numbers at all — rendering is not the "
        "only place a four-seed mean could leak")


def test_aggregate_cell_computes_mean_and_std_over_five_seeds():
    records = [_fake_data(a_cell(cell="rgen", seed=s)) for s in V.SEEDS]
    agg = C.aggregate_cell(records)
    assert agg.status == "OK" and agg.n == 5 and sorted(agg.seeds) == list(V.SEEDS)
    vals = [synthetic_metrics(a_cell(cell="rgen", seed=s))["T60"] for s in V.SEEDS]
    mean, std = agg.values["T60"]
    assert mean == pytest.approx(sum(vals) / 5)
    assert std == pytest.approx(
        math.sqrt(sum((v - sum(vals) / 5) ** 2 for v in vals) / 4))
    assert agg.per_seed["T60"][42] == pytest.approx(vals[0])


def test_aggregate_cell_rejects_a_mixed_block():
    """One block is one (arm, cell, K); mixing arms would average two models."""
    mixed = [_fake_data(a_cell(arm="C8", cell="rgen", seed=s)) for s in (42, 43, 44)]
    mixed += [_fake_data(a_cell(arm="C16", cell="rgen", seed=s)) for s in (45, 46)]
    agg = C.aggregate_cell(mixed)
    assert agg.status != "OK", "a block spanning two arms must never aggregate"


def test_aggregate_cell_of_an_empty_block_is_pending():
    agg = C.aggregate_cell([])
    assert agg.status == "PENDING" and agg.n == 0 and agg.values == {}


# --------------------------------------------------------------------------- #
# 7. paired_t_ci — the §4 estimation convention, pinned to known values
# --------------------------------------------------------------------------- #
def test_paired_t_ci_matches_a_hand_computed_fixture():
    """Five differences, df=4, two-sided 95%: mean ± t(.975,4)·s/√5."""
    diffs = [1.0, 2.0, 3.0, 4.0, 5.0]
    res = C.paired_t_ci(diffs)
    mean, sd, n = 3.0, math.sqrt(2.5), 5
    se = sd / math.sqrt(n)
    assert res.df == 4 and res.n == 5
    assert res.mean == pytest.approx(mean)
    assert res.lo == pytest.approx(mean - 2.776445105 * se, rel=1e-6)
    assert res.hi == pytest.approx(mean + 2.776445105 * se, rel=1e-6)
    assert res.t == pytest.approx(mean / se)
    assert res.p == pytest.approx(0.01324, abs=1e-4)


def test_paired_t_critical_value_is_the_preregistered_constant():
    """§4 pins t_{0.975,4} = 2.776445 whether or not scipy is installed."""
    assert C.t_critical(0.05, 4) == pytest.approx(2.776445105, rel=1e-8)


def test_paired_t_fallback_agrees_with_scipy():
    """The pure-python survival function is the fallback when scipy is absent;
    it must not be a different test from the one the campaign pre-registered."""
    scipy_stats = pytest.importorskip("scipy.stats")
    for df in (4, 3, 9):
        for t in (0.0, 0.5, 1.234, 2.776445105, 6.0):
            assert C._student_t_sf(t, df) == pytest.approx(
                float(scipy_stats.t.sf(t, df)), rel=1e-6, abs=1e-9)
        assert C._student_t_ppf(0.975, df) == pytest.approx(
            float(scipy_stats.t.ppf(0.975, df)), rel=1e-6)


def test_paired_t_ci_handles_zero_variance():
    """Five identical differences: a real effect with no spread, and a zero
    effect with no spread, must not both come back as 'significant'."""
    same = C.paired_t_ci([2.0] * 5)
    assert same.p == pytest.approx(0.0) and same.lo == same.hi == pytest.approx(2.0)
    nothing = C.paired_t_ci([0.0] * 5)
    assert nothing.p == pytest.approx(1.0) and nothing.mean == 0.0


def test_paired_t_ci_refuses_fewer_than_two_observations():
    with pytest.raises(ValueError):
        C.paired_t_ci([1.0])


# --------------------------------------------------------------------------- #
# 8. holm_adjust
# --------------------------------------------------------------------------- #
def test_holm_adjust_known_fixture():
    """Two co-primaries (the campaign's own case) and a three-way fixture."""
    assert C.holm_adjust([0.01, 0.04]) == pytest.approx([0.02, 0.04])
    assert C.holm_adjust([0.04, 0.01]) == pytest.approx([0.04, 0.02])
    assert C.holm_adjust([0.01, 0.02, 0.03]) == pytest.approx([0.03, 0.04, 0.04])


def test_holm_adjust_is_monotone_and_capped():
    adj = C.holm_adjust([0.4, 0.5, 0.6])
    assert adj == pytest.approx([1.0, 1.0, 1.0])
    adj2 = C.holm_adjust([0.001, 0.9])
    assert adj2[0] <= adj2[1]


def test_holm_adjust_handles_ties():
    assert C.holm_adjust([0.03, 0.03]) == pytest.approx([0.06, 0.06])
    assert C.holm_adjust([]) == []


# --------------------------------------------------------------------------- #
# 9. metric_direction — the complete §4 table
# --------------------------------------------------------------------------- #
def test_metric_direction_is_the_complete_table():
    for metric in ("T60", "C50", "EDT", "FD"):
        assert C.metric_direction(metric) == "lower"
    for metric in ("RIR_to_GT_RIR_R@1", "RIR_to_GT_RIR_R@5", "RIR_to_GT_RIR_R@10",
                   "RIR_to_geom_R@1", "RIR_to_geom_R@5", "RIR_to_geom_R@10"):
        assert C.metric_direction(metric) == "higher"


def test_metric_direction_accepts_the_plans_display_names():
    assert C.metric_direction("T60%") == "lower"
    assert C.metric_direction("R@1") == "higher"


def test_metric_direction_refuses_an_unknown_metric():
    """A metric with no registered direction cannot be scored 'better' at all."""
    with pytest.raises(KeyError):
        C.metric_direction("Invalid T60")
    with pytest.raises(KeyError):
        C.metric_direction("made_up")


def test_co_primary_metrics_are_t60_and_audio_to_audio_r_at_1():
    """Round-1 review B3: the reported R@1 is RIR_to_GT_RIR_R@1 (audio-to-audio).
    Only RIR_to_geom_R@k embeds the rotated point cloud."""
    assert C.CO_PRIMARY == ("T60", "RIR_to_GT_RIR_R@1")
    assert all(m.startswith("RIR_to_geom") for m in C.CONFOUNDED_METRICS)
    assert not set(C.CO_PRIMARY) & set(C.CONFOUNDED_METRICS)


# --------------------------------------------------------------------------- #
# 10. the contrast machinery
# --------------------------------------------------------------------------- #
def test_contrast_reads_direction_from_the_metric():
    """Lower-is-better: a negative mean difference FAVOURS the first arm."""
    lower = C.contrast("T60", [-1.0, -1.1, -0.9, -1.2, -1.0], better="lower")
    assert lower.favors_first and lower.p < 0.05
    higher = C.contrast("RIR_to_GT_RIR_R@1", [1.0, 1.1, 0.9, 1.2, 1.0], better="higher")
    assert higher.favors_first and higher.p < 0.05
    reversed_ = C.contrast("T60", [1.0, 1.1, 0.9, 1.2, 1.0], better="lower")
    assert not reversed_.favors_first


def test_verdict_rules_follow_the_plan():
    """SUPPORTED = both co-primaries favour after Holm; PARTIAL = exactly one;
    NEGATIVE = neither (or reversed)."""
    assert C.verdict([True, True]) == "SUPPORTED"
    assert C.verdict([True, False]) == "PARTIAL"
    assert C.verdict([False, False]) == "NEGATIVE"


def test_endpoint_contrast_applies_holm_over_the_two_co_primaries():
    seeds = list(V.SEEDS)
    first = {"T60": {s: 8.0 for s in seeds},
             "RIR_to_GT_RIR_R@1": {s: 6.0 + 0.01 * i for i, s in enumerate(seeds)}}
    second = {"T60": {s: 9.0 + 0.01 * i for i, s in enumerate(seeds)},
              "RIR_to_GT_RIR_R@1": {s: 5.0 for s in seeds}}
    res = C.endpoint_contrast("H-P demo", first, second, seeds, better="metric")
    assert set(res["metrics"]) == set(C.CO_PRIMARY)
    for metric, row in res["metrics"].items():
        assert row["p_holm"] >= row["p"], "Holm may only make a p-value larger"
    assert res["verdict"] == "SUPPORTED", res
