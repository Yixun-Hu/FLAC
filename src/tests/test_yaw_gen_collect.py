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


def _stable_jitter(*parts, scale=0.01, spread=7):
    """A deterministic pseudo-noise term (Python's own hash() is randomised)."""
    h = 0
    for part in parts:
        for ch in str(part):
            h = (h * 131 + ord(ch)) % 1000003
    return ((h % spread) - (spread // 2)) * scale


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
        # Noise that does NOT cancel in the paired difference, so the demo
        # transcripts exercise real confidence intervals instead of the
        # degenerate zero-spread branch. Rotated cells only: the unrotated block
        # is the gates' own yardstick (σ̂) and their reference at seed 42, so
        # perturbing it would move G1's tolerance and its measurand together.
        out["T60"] += _stable_jitter(cell.arm, cell.seed, cell.k, "T60")
        out["RIR_to_GT_RIR_R@1"] += _stable_jitter(cell.arm, cell.seed, cell.k, "R1")
    if cell.cell == "vctl":
        if cell.arm == "VANL":                           # the positive control
            out["T60"] += 3.4
            out["RIR_to_GT_RIR_R@1"] -= 1.0
        elif float(cell.rotate_deg) == 45.0:             # off-group mechanism control
            out["T60"] += 0.6
    return out


SCENE_TEMPLATE = {"T60": 9.0, "C50": 1.0, "EDT": 43.0, "FD": 0.34,
                  "RIR_to_GT_RIR_R@1": 5.0, "RIR_to_GT_RIR_R@5": 15.0,
                  "RIR_to_GT_RIR_R@10": 23.0, "RIR_to_geom_R@1": 4.0,
                  "RIR_to_geom_R@5": 13.0, "RIR_to_geom_R@10": 20.0,
                  "Invalid T60": 0.0}


def _scene_block(cell, metrics=None, n=V.EXPECTED_SCENES):
    """Per-scene values that AVERAGE to the cell's metrics, exactly.

    The offsets are symmetric about the middle scene, so their mean is zero: the
    per-scene mean this campaign reports equals the value the fixture intends,
    while no two scenes carry the same number (a per-scene mean that is right for
    the wrong reason — every scene identical — would prove nothing)."""
    base = dict(metrics if metrics is not None else synthetic_metrics(cell))
    mid = (n - 1) / 2.0
    scenes = {}
    for i in range(n):
        spread = (i - mid) * 0.01
        scenes[f"Room{i}/Room{i}_idx_{i}"] = {
            k: (v + spread if k != "Invalid T60" else v) for k, v in base.items()}
    return scenes


def write_cell(root, cell, *, pin=PIN, count=COUNT, metrics=None, record_patch=None,
               meta_patch=None, stream_patch=None, offsets=None, targets=None,
               ckpt_sha=None, by_scene=_scene_block):
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
    scenes = by_scene(cell, metrics) if callable(by_scene) else by_scene
    if scenes is not None:
        record["by_scene"] = scenes
        record["per_scene_schema"] = V.PER_SCENE_SCHEMA
        record["scene_count"] = len(scenes)
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
        "record_per_scene": True,
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


def test_match_assignments_does_not_compare_two_different_fixed_angles(tmp_path):
    """C4L@45 and the four @90 controls are BOTH registered validity cells and
    their yaw assignments differ by design. Grouping them together would report
    the campaign's own design as an integrity failure — while the items they
    evaluated (input_hash) must still agree across every one of them."""
    root = str(tmp_path)
    vctl = [c for c in V.expected_grid() if c.cell == "vctl"]
    write_grid(root, vctl)
    assert C.match_assignments(_load_all(root, vctl)) == []


def test_match_assignments_still_compares_two_cells_at_the_same_angle(tmp_path):
    """...but two @90 cells ARE still compared. Note what such a violation can and
    cannot be: a fixed cell whose offsets are not the angle's own column shift is
    already refused per-cell by the validator, so at one angle the only reachable
    disagreement is a different item stream — which must surface on BOTH hashes,
    scoped to that angle."""
    root = str(tmp_path)
    vctl = [c for c in V.expected_grid() if c.cell == "vctl"]
    write_grid(root, [c for c in vctl if c.arm != "C8"])
    for c in [c for c in vctl if c.arm == "C8"]:
        write_cell(root, c, targets=[f"elsewhere/rir_{i}.wav" for i in range(COUNT)])
    violations = C.match_assignments(_load_all(root, vctl))
    assert any(v.kind == "cross_arm_assignment_hash" and "90" in v.scope
               for v in violations), violations
    assert any(v.kind == "cross_arm_input_hash" for v in violations), violations


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
                      metrics=synthetic_metrics(cell),
                      flat_metrics={m: synthetic_metrics(cell)[m] for m in C.G5_METRICS},
                      input_hash="i" * 64, assignment_hash="a" * 64, offsets=(),
                      source_sha=PIN)


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


# --------------------------------------------------------------------------- #
# 11. evaluate_gates — G1..G4 executable, G5 a check and never a gate
# --------------------------------------------------------------------------- #
def _full_grid_store(tmp_path, **kw):
    root = str(tmp_path)
    write_grid(root, **kw)
    return root, C.collect_cells(root, expectation())


def test_gates_pass_on_a_complete_well_formed_campaign(tmp_path):
    root, store = _full_grid_store(tmp_path)
    gates = C.evaluate_gates(store)
    assert gates["all_passed"] is True, gates
    for name in ("G1", "G2", "G3", "G4"):
        assert gates[name]["status"] == "PASS", (name, gates[name])


def test_g1_fails_when_an_in_group_rotation_moves_the_metric(tmp_path):
    """The floor gate: rotating a Cn arm BY ITS OWN GROUP ANGLE must not move
    the metric by more than half the arm's own seed noise."""
    root = str(tmp_path)
    grid = [c for c in V.expected_grid()
            if not (c.arm == "C16" and c.cell == "vctl")]
    write_grid(root, grid)
    broken = V.Cell("C16", "vctl", V.STEP, 42, 8, 90.0)
    metrics = synthetic_metrics(broken)
    metrics["T60"] += 5.0                     # far past 0.5 sigma
    write_cell(root, broken, metrics=metrics)
    gates = C.evaluate_gates(C.collect_cells(root, expectation()))
    assert gates["G1"]["status"] == "FAIL"
    assert any("C16" in f for f in gates["G1"]["failures"]), gates["G1"]
    assert gates["all_passed"] is False


def test_g2_fails_when_the_positive_control_does_not_degrade(tmp_path):
    """If VANL@90 looks like VANL@0, the harness is not detecting non-invariance
    at all and every 'flat' reading below it is uninterpretable."""
    root = str(tmp_path)
    grid = [c for c in V.expected_grid() if not (c.arm == "VANL" and c.cell == "vctl")]
    write_grid(root, grid)
    flat = V.Cell("VANL", "vctl", V.STEP, 42, 8, 90.0)
    write_cell(root, flat, metrics=synthetic_metrics(a_cell(arm="VANL", cell="zref")))
    gates = C.evaluate_gates(C.collect_cells(root, expectation()))
    assert gates["G2"]["status"] == "FAIL", gates["G2"]
    assert gates["all_passed"] is False


def test_g3_recomputes_the_golden_assignment_rather_than_trusting_it(tmp_path):
    root = str(tmp_path)
    grid = [c for c in V.expected_grid()
            if not (c.arm == "C32" and c.cell == "rgen" and c.seed == 42 and c.k == 8)]
    write_grid(root, grid)
    tampered = V.Cell("C32", "rgen", V.STEP, 42, 8, None)
    offs = C.golden_offsets(42, COUNT, V.IMG_W)
    offs[3] = (offs[3] + 1) % V.IMG_W              # one position off by one column
    write_cell(root, tampered, offsets=offs)
    gates = C.evaluate_gates(C.collect_cells(root, expectation()))
    assert gates["G3"]["status"] == "FAIL", gates["G3"]
    assert any("C32" in f for f in gates["G3"]["failures"])


def test_g3_golden_sequence_is_the_preregistered_seed42_stream():
    """The collector's recomputation must reproduce the round-1 constant."""
    assert C.golden_offsets(42, len(GOLDEN_SEED42_W512), 512) == GOLDEN_SEED42_W512
    assert C.golden_offsets(43, 16, 512) != GOLDEN_SEED42_W512


def test_g4_is_the_section_3_3_hash_equalities(tmp_path):
    root = str(tmp_path)
    grid = [c for c in V.expected_grid()
            if not (c.arm == "C8" and c.cell == "rgen" and c.seed == 44 and c.k == 8)]
    write_grid(root, grid)
    odd = V.Cell("C8", "rgen", V.STEP, 44, 8, None)
    write_cell(root, odd, targets=[f"other/rir_{i}.wav" for i in range(COUNT)])
    gates = C.evaluate_gates(C.collect_cells(root, expectation()))
    assert gates["G4"]["status"] == "FAIL"
    assert gates["G4"]["violations"], gates["G4"]
    assert ("cross_arm", 8) in {tuple(b) for b in gates["blocked_scopes"]}


def test_g4_on_an_empty_campaign_is_pending_not_pass(tmp_path):
    """Nothing to compare is not the same as everything agreeing. With no cells
    on disk there are no hash equalities to check, and a PASS there would report
    a campaign that has not started as having satisfied its integrity gate."""
    store = C.collect_cells(str(tmp_path), expectation())
    assert store.cells == []
    gates = C.evaluate_gates(store)
    assert gates["G4"]["status"] == "PENDING", gates["G4"]
    assert gates["all_passed"] is False


def test_g4_is_pass_only_once_comparisons_were_actually_made(tmp_path):
    root = str(tmp_path)
    cells = _one_seed_slice()
    write_grid(root, cells)
    gates = C.evaluate_gates(C.collect_cells(root, expectation()))
    assert gates["G4"]["status"] == "PASS"
    assert gates["G4"]["comparisons"] > 0, gates["G4"]


def test_g5_is_reported_but_never_gates(tmp_path):
    """The exp_11 comparison is cross-pin: informative, never a halt."""
    root, store = _full_grid_store(tmp_path)
    gates = C.evaluate_gates(store, exp11_root=str(tmp_path / "no_exp11_here"))
    assert gates["G5"]["status"] in ("CHECK", "UNAVAILABLE")
    assert gates["all_passed"] is True, "G5 must not participate in the gate verdict"
    assert "G5" not in gates["gate_names"]


def test_gates_report_pending_when_the_z_block_is_incomplete(tmp_path):
    """G1's tolerance is a std over five Z seeds; with four, there is no gate to
    evaluate — and 'no gate evaluated' may never read as 'gate passed'."""
    root = str(tmp_path)
    grid = [c for c in V.expected_grid()
            if not (c.arm == "C8" and c.cell == "zref" and c.seed == 46 and c.k == 8)]
    write_grid(root, grid)
    gates = C.evaluate_gates(C.collect_cells(root, expectation()))
    assert gates["G1"]["status"] == "PENDING", gates["G1"]
    assert gates["all_passed"] is False


# --------------------------------------------------------------------------- #
# 12. collect_cells — discovery is fail-closed too
# --------------------------------------------------------------------------- #
def test_collect_cells_separates_valid_missing_and_rejected(tmp_path):
    root = str(tmp_path)
    grid = list(V.expected_grid())
    write_grid(root, grid[:-3])
    bad = grid[-3]
    write_cell(root, bad, record_patch={"weights_source": "online"})
    store = C.collect_cells(root, expectation())
    assert len(store.cells) == len(grid) - 3
    assert len(store.missing) == 2 and len(store.rejected) == 1
    assert store.rejected[0]["cell"] == V.eval_name(bad)
    assert any("weights_source" in r for r in store.rejected[0]["reasons"])
    assert bad not in {c.cell for c in store.cells}, "a rejected cell must not be data"


# --------------------------------------------------------------------------- #
# 13. suppress_validity_cells — the V block is QA, never a headline
# --------------------------------------------------------------------------- #
def test_suppress_validity_cells_removes_every_v_cell():
    cells = [_fake_data(a_cell(cell="rgen", seed=42)),
             _fake_data(a_cell(cell="zref", seed=42)),
             _fake_data(V.Cell("C4L", "vctl", V.STEP, 42, 8, 90.0)),
             _fake_data(V.Cell("C4L", "vctl", V.STEP, 42, 8, 45.0))]
    kept = C.suppress_validity_cells(cells)
    assert len(kept) == 2 and all(c.cell.cell != "vctl" for c in kept)


def test_suppress_validity_cells_also_filters_rendered_rows():
    rows = [{"cell_type": "rgen", "arm": "C8"}, {"cell_type": "vctl", "arm": "C8"}]
    assert C.suppress_validity_cells(rows) == [rows[0]]


# --------------------------------------------------------------------------- #
# 14. render_tables — golden markdown for the two load-bearing sections
# --------------------------------------------------------------------------- #
_GOLDEN_BLOCK = """\
| arm | K | n | T60 (scene-mean) ↓ | RIR_to_GT_RIR_R@1 (split) ↑ |
|---|---|---|---|---|
| VANL | 8 | 5 | 12.000 ± 0.100 | 4.000 ± 0.050 |
| C4L | 8 | — | *PENDING (3/5 seeds)* | |
| C8 | 8 | 5 | **BLOCKED — arms are not rotation-matched at K=8** | |
"""


def test_render_block_table_is_byte_stable():
    rows = [
        {"arm": "VANL", "K": 8, "status": "OK", "n": 5,
         "values": {"T60": (12.0, 0.1), "RIR_to_GT_RIR_R@1": (4.0, 0.05)}},
        {"arm": "C4L", "K": 8, "status": "PENDING", "n": 3, "values": {},
         "reasons": ["3/5 seeds on disk; missing [45, 46]"]},
        {"arm": "C8", "K": 8, "status": "BLOCKED", "n": 5, "values": {},
         "reasons": ["arms are not rotation-matched at K=8"]},
    ]
    assert C.render_block_table(rows, ("T60", "RIR_to_GT_RIR_R@1")) == _GOLDEN_BLOCK


_GOLDEN_GATES = """\
| gate | status | definition |
|---|---|---|
| G1 | PASS | in-group floor |
| G2 | FAIL | positive control |
| G3 | PENDING | golden assignment |
| G4 | PASS | assignment integrity |

**G2 FAIL** — VANL T60: degradation 0.0100 < 5·σ̂=0.0790
**G3 PENDING** — no rotated (rgen) cell has landed yet
"""


def test_render_gate_report_is_byte_stable():
    gates = {
        "gate_names": ("G1", "G2", "G3", "G4"),
        "G1": {"status": "PASS", "definition": "in-group floor", "failures": [],
               "pending": []},
        "G2": {"status": "FAIL", "definition": "positive control",
               "failures": ["VANL T60: degradation 0.0100 < 5·σ̂=0.0790"], "pending": []},
        "G3": {"status": "PENDING", "definition": "golden assignment", "failures": [],
               "pending": ["no rotated (rgen) cell has landed yet"]},
        "G4": {"status": "PASS", "definition": "assignment integrity", "failures": [],
               "pending": []},
    }
    assert C.render_gate_report(gates) == _GOLDEN_GATES


# --------------------------------------------------------------------------- #
# 15. end to end: a complete grid, and three deliberately-broken ones
# --------------------------------------------------------------------------- #
def _results(root, **kw):
    store = C.collect_cells(root, expectation())
    return C.build_results(store, generated_at="2026-08-11T00:00:00-04:00", **kw)


def test_end_to_end_complete_grid_renders_every_section(tmp_path):
    root = str(tmp_path)
    write_grid(root)
    results = _results(root)
    text = C.render_tables(results)
    for heading in ("Cell inventory", "Validity gates", "Absolute robustness",
                    "Paired degradation", "Endpoint contrasts", "Adjacent",
                    "geometry retrieval", "Validity cells"):
        assert heading.lower() in text.lower(), f"missing section: {heading}"
    assert results["gates"]["all_passed"] is True
    assert results["hypotheses"]["suppressed"] is False
    for name in ("H-P", "H-M", "H-S"):
        assert results["hypotheses"][name]["verdict"] in (
            "SUPPORTED", "PARTIAL", "NEGATIVE"), name
    # R3F7: the complete-grid invariant, asserted rather than waved through
    assert "PENDING" not in text and "BLOCKED" not in text.split("## 9")[0], (
        "a complete, well-formed campaign rendered a refusal")


def test_headline_tables_never_contain_a_validity_cell_or_geom_retrieval(tmp_path):
    root = str(tmp_path)
    write_grid(root)
    text = C.render_tables(_results(root))
    headline, _, tail = text.partition("## 7. Geometry retrieval")
    # TABLE ROWS only: the aggregation ruling in the header necessarily NAMES the
    # geometry family (it is the thing being routed to the split-level source),
    # so the invariant is that none of its NUMBERS reaches a headline table.
    headline_tables = [ln for ln in headline.splitlines() if ln.startswith("|")]
    assert not any("vctl" in ln for ln in headline_tables), (
        "a validity cell reached a headline table")
    assert not any("RIR_to_geom" in ln for ln in headline_tables), (
        "the rotated-gallery retrieval metric reached a headline table")
    assert "RIR_to_geom" in tail and "confounded" in tail.lower()


def test_a_missing_seed_renders_pending_and_never_a_number(tmp_path):
    root = str(tmp_path)
    grid = [c for c in V.expected_grid()
            if not (c.arm == "C32" and c.cell == "rgen" and c.seed == 46 and c.k == 8)]
    write_grid(root, grid)
    results = _results(root)
    block = results["blocks"]["R"]["C32"]["8"]
    assert block["status"] == "PENDING" and block["values"] == {}
    text = C.render_tables(results)
    assert "PENDING (4/5 seeds)" in text
    assert results["hypotheses"]["H-P"]["status"] == "PENDING", (
        "an endpoint contrast whose input block is incomplete cannot be a number")


def test_a_hash_mismatch_blocks_the_affected_contrast(tmp_path):
    root = str(tmp_path)
    grid = [c for c in V.expected_grid()
            if not (c.arm == "C32" and c.cell == "rgen" and c.seed == 43 and c.k == 8)]
    write_grid(root, grid)
    write_cell(root, V.Cell("C32", "rgen", V.STEP, 43, 8, None),
               targets=[f"wrong/rir_{i}.wav" for i in range(COUNT)])
    results = _results(root)
    text = C.render_tables(results)
    assert results["gates"]["G4"]["status"] == "FAIL"
    assert results["hypotheses"]["H-P"]["status"] == "BLOCKED", results["hypotheses"]["H-P"]
    assert "BLOCKED" in text
    assert results["hypotheses"]["suppressed"] is True


def test_a_failing_gate_suppresses_the_h_readouts(tmp_path):
    """Plan §4: H-readouts render only when G1–G4 pass. A failing gate is not a
    footnote under a number; the number does not appear."""
    root = str(tmp_path)
    grid = [c for c in V.expected_grid() if not (c.arm == "VANL" and c.cell == "vctl")]
    write_grid(root, grid)
    write_cell(root, V.Cell("VANL", "vctl", V.STEP, 42, 8, 90.0),
               metrics=synthetic_metrics(a_cell(arm="VANL", cell="zref")))
    results = _results(root)
    text = C.render_tables(results)
    assert results["gates"]["G2"]["status"] == "FAIL"
    assert results["hypotheses"]["suppressed"] is True
    assert "SUPPRESSED" in text
    body = text.split("Endpoint contrasts")[1].split("##")[0]
    assert "SUPPORTED" not in body and "PARTIAL" not in body, body


def test_json_bundle_is_machine_readable_and_complete(tmp_path):
    root = str(tmp_path)
    write_grid(root)
    results = _results(root)
    encoded = json.dumps(results)                    # must be JSON-safe end to end
    back = json.loads(encoded)
    for key in ("campaign", "aggregation", "inventory", "blocks", "paired", "gates",
                "hypotheses", "adjacent", "geometry_retrieval", "validity_cells"):
        assert key in back, key
    assert back["campaign"]["pin"] == PIN
    assert back["campaign"]["expected_count"] == COUNT


def test_cli_writes_markdown_and_json(tmp_path):
    root = str(tmp_path / "outputs")
    os.makedirs(root)
    write_grid(root)
    expect_file = tmp_path / "ckpt_expect.json"
    expect_file.write_text(json.dumps({
        "step": V.STEP, "arms": {a: {"sha256": s} for a, s in CKPT_SHA.items()}}))
    out_md, out_json = tmp_path / "r.md", tmp_path / "r.json"
    rc = C.main(["--output-root", root, "--pin", PIN, "--expected-count", str(COUNT),
                 "--ckpt-expect", str(expect_file), "--out", str(out_md),
                 "--json", str(out_json)])
    assert rc == 0
    assert "Absolute robustness" in out_md.read_text()
    assert json.loads(out_json.read_text())["campaign"]["pin"] == PIN


def test_cli_refuses_to_run_without_a_pin(tmp_path):
    with pytest.raises(SystemExit):
        C.main(["--output-root", str(tmp_path)])


def test_markdown_cells_escape_the_pipes_inside_them():
    """G1's definition is literally |m(V@90°) − m(Z)| ≤ 0.5·σ̂ — four bars that
    would otherwise split the gate row into extra columns."""
    gates = {"gate_names": ("G1",),
             "G1": {"status": "PASS", "definition": "|m(V)−m(Z)| ≤ 0.5·σ̂",
                    "failures": [], "pending": []}}
    row = [ln for ln in C.render_gate_report(gates).splitlines()
           if ln.startswith("| G1")][0]
    assert row.count("|") - row.count("\\|") == 4, row      # exactly 3 columns


def test_cli_exit_code_distinguishes_incomplete_readouts_from_failed_gates(tmp_path):
    """Gates can pass over a campaign that simply has not finished; 0 must mean
    'the readouts exist', so an incomplete grid gets its own code."""
    root = str(tmp_path / "outputs")
    os.makedirs(root)
    grid = [c for c in V.expected_grid()
            if not (c.arm == "C32" and c.cell == "rgen" and c.seed == 46 and c.k == 8)]
    write_grid(root, grid)
    expect_file = tmp_path / "e.json"
    expect_file.write_text(json.dumps({
        "step": V.STEP, "arms": {a: {"sha256": s} for a, s in CKPT_SHA.items()}}))
    rc = C.main(["--output-root", root, "--pin", PIN, "--expected-count", str(COUNT),
                 "--ckpt-expect", str(expect_file), "--out", str(tmp_path / "o.md")])
    assert rc == 4, "an incomplete campaign must not exit 0, nor look like a gate failure"


def test_cli_exit_code_reports_that_gates_did_not_pass(tmp_path):
    """A collector run over an incomplete campaign must not exit 0 as if it had
    produced the readouts: the exit code is what a wrapper script reads."""
    root = str(tmp_path / "outputs")
    os.makedirs(root)
    write_grid(root, [c for c in V.expected_grid() if c.cell == "zref"])
    expect_file = tmp_path / "e.json"
    expect_file.write_text(json.dumps({
        "step": V.STEP, "arms": {a: {"sha256": s} for a, s in CKPT_SHA.items()}}))
    rc = C.main(["--output-root", root, "--pin", PIN, "--expected-count", str(COUNT),
                 "--ckpt-expect", str(expect_file), "--out", str(tmp_path / "o.md")])
    assert rc != 0


# --------------------------------------------------------------------------- #
# 16. round-3 fixes — the per-scene estimand and consumer-level payloads
# --------------------------------------------------------------------------- #
def test_the_observation_is_the_mean_over_scenes(tmp_path):
    """R3F1 (review B1): the plan's estimand is the PER-SCENE mean. The flat
    metrics block in the record is the split-level (item-weighted) number, which
    is a DIFFERENT quantity — the collector must read by_scene."""
    cell = a_cell()
    scenes = {f"Room{i}/Room{i}_idx_{i}": dict(SCENE_TEMPLATE, T60=float(i))
              for i in range(V.EXPECTED_SCENES)}
    path = write_cell(str(tmp_path), cell, by_scene=scenes,
                      record_patch={"metrics": dict(synthetic_metrics(cell), T60=999.0)})
    data, reasons = C.load_cell(path, expectation())
    assert reasons == [], reasons
    assert data.metrics["T60"] == pytest.approx(sum(range(V.EXPECTED_SCENES))
                                                / V.EXPECTED_SCENES)
    assert data.metrics["T60"] != 999.0, "the collector read the item-weighted number"


def test_a_cell_without_per_scene_results_is_invalid_with_no_fallback(tmp_path):
    cell = a_cell()
    path = write_cell(str(tmp_path), cell, by_scene=None)
    data, reasons = C.load_cell(path, expectation())
    assert data is None
    assert any("by_scene" in r for r in reasons), reasons


def test_a_scene_missing_a_reported_metric_is_rejected(tmp_path):
    """R3F5: a nonempty payload missing R@1 used to pass validation and raise
    KeyError later, in the middle of a contrast."""
    cell = a_cell()
    scenes = _scene_block(cell)
    del scenes["Room3/Room3_idx_3"]["C50"]          # an ACOUSTIC metric: scene-mean
    data, reasons = C.load_cell(write_cell(str(tmp_path), cell, by_scene=scenes),
                                expectation())
    assert data is None
    assert any("C50" in r and "Room3" in r for r in reasons), reasons


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "9.0", None])
def test_a_non_finite_or_non_numeric_metric_is_rejected(tmp_path, bad):
    """R3F5: NaN must never reach a mean, a CI or a verdict."""
    cell = a_cell()
    scenes = _scene_block(cell)
    scenes["Room0/Room0_idx_0"]["T60"] = bad
    data, reasons = C.load_cell(write_cell(str(tmp_path), cell, by_scene=scenes),
                                expectation())
    assert data is None
    assert any("T60" in r for r in reasons), reasons


def test_the_geometry_metrics_are_optional_but_checked_when_present(tmp_path):
    """The confounded block is descriptive, so its absence is not a campaign
    failure — but a NaN in it must still never be published."""
    cell = a_cell()
    flat = {k: v for k, v in synthetic_metrics(cell).items()
            if k not in C.CONFOUNDED_METRICS}
    data, reasons = C.load_cell(
        write_cell(str(tmp_path), cell, record_patch={"metrics": flat}), expectation())
    assert reasons == [] and data is not None
    assert not any(m in data.metrics for m in C.CONFOUNDED_METRICS)

    flat2 = dict(synthetic_metrics(cell))
    flat2["RIR_to_geom_R@1"] = float("nan")
    data2, reasons2 = C.load_cell(
        write_cell(str(tmp_path), a_cell(arm="C16"), record_patch={"metrics": flat2}),
        expectation())
    assert data2 is None and any("RIR_to_geom_R@1" in r for r in reasons2), reasons2


def test_no_aggregation_deviation_language_survives():
    """R3F1: the estimand is measured now, so the deviation disclosure and the
    'contrasts are unaffected' claim must be gone — not softened."""
    source = open(C.__file__).read()
    for phrase in ("DEVIATION", "contrasts are unaffected", "item-weighted rather than",
                   "not recoverable from the committed artifacts"):
        assert phrase not in source, f"stale aggregation language: {phrase!r}"
    assert C.AGGREGATION["scene_mean"] == list(C.ACOUSTIC_METRICS)
    assert C.AGGREGATION["split_level"] == list(C.SPLIT_LEVEL_METRICS)
    assert "per-scene mean applies to the ACOUSTIC" in C.AGGREGATION["ruling"]


def test_paired_pending_rows_carry_the_matched_pair_count(tmp_path):
    """N6: a paired block with four matched pairs reported 0/5, because the count
    was derived from the (deliberately empty) metrics dict."""
    root = str(tmp_path)
    grid = [c for c in V.expected_grid()
            if not (c.arm == "C32" and c.cell == "rgen" and c.seed == 46 and c.k == 8)]
    write_grid(root, grid)
    results = _results(root)
    entry = results["paired"]["C32"]["8"]
    assert entry["status"] == "PENDING"
    assert entry["pairs"] == 4, entry
    text = C.render_tables(results)
    assert "PENDING (4/5 pairs)" in text, "the Δ table still reports 0/5"


# --------------------------------------------------------------------------- #
# 17. r3-fix2 — the Planner's per-metric aggregation ruling
#
# Per-scene applies to the ACOUSTIC-PARAMETER family only. Retrieval and FD are
# read from the split-level metrics, because a within-scene gallery is a
# different, easier task and a one-room Frechet is small-sample biased.
# --------------------------------------------------------------------------- #
def test_aggregation_source_routes_each_metric():
    for metric in ("T60", "C50", "EDT", "Invalid T60"):
        assert C.aggregation_source(metric) == "scene-mean", metric
    for metric in ("FD", "RIR_to_GT_RIR_R@1", "RIR_to_GT_RIR_R@5",
                   "RIR_to_GT_RIR_R@10", "RIR_to_geom_R@1", "RIR_to_geom_R@5",
                   "RIR_to_geom_R@10"):
        assert C.aggregation_source(metric) == "split", metric
    with pytest.raises(KeyError):
        C.aggregation_source("made_up")


def test_the_co_primaries_come_from_different_sources():
    """T60% is the scene-mean; R@1 is the split-level quantity exp_01's noise
    floor was calibrated against."""
    assert C.CO_PRIMARY == ("T60", "RIR_to_GT_RIR_R@1")
    assert C.aggregation_source(C.CO_PRIMARY[0]) == "scene-mean"
    assert C.aggregation_source(C.CO_PRIMARY[1]) == "split"


def test_cell_metrics_are_routed_per_metric(tmp_path):
    """The decisive test: build a cell whose two sources DISAGREE for both
    families, and check each metric came from the ruled one."""
    cell = a_cell()
    flat = dict(synthetic_metrics(cell))
    flat["T60"] = 100.0                      # split-level T60: must NOT be used
    flat["RIR_to_GT_RIR_R@1"] = 7.5          # split-level R@1: MUST be used
    flat["FD"] = 0.99                        # split-level FD: MUST be used
    scenes = _scene_block(cell)              # scene means: T60 = the synthetic value
    for payload in scenes.values():
        payload["RIR_to_GT_RIR_R@1"] = 1.0   # scene retrieval: must NOT be used
        payload["FD"] = 0.11                 # scene FD: must NOT be used
    path = write_cell(str(tmp_path), cell, by_scene=scenes,
                      record_patch={"metrics": flat})
    data, reasons = C.load_cell(path, expectation())
    assert reasons == [], reasons
    assert data.metrics["T60"] == pytest.approx(synthetic_metrics(cell)["T60"])
    assert data.metrics["T60"] != 100.0, "T60 was taken from the split-level block"
    assert data.metrics["RIR_to_GT_RIR_R@1"] == pytest.approx(7.5)
    assert data.metrics["FD"] == pytest.approx(0.99)


def test_a_missing_split_level_metric_is_rejected(tmp_path):
    """The split side gets the same consumer-level payload check as the scene one."""
    cell = a_cell()
    flat = dict(synthetic_metrics(cell))
    del flat["RIR_to_GT_RIR_R@1"]
    data, reasons = C.load_cell(
        write_cell(str(tmp_path), cell, record_patch={"metrics": flat}), expectation())
    assert data is None and any("RIR_to_GT_RIR_R@1" in r for r in reasons), reasons


def test_a_non_finite_split_level_metric_is_rejected(tmp_path):
    cell = a_cell()
    flat = dict(synthetic_metrics(cell), FD=float("nan"))
    data, reasons = C.load_cell(
        write_cell(str(tmp_path), cell, record_patch={"metrics": flat}), expectation())
    assert data is None and any("FD" in r for r in reasons), reasons


def test_by_scene_is_still_required_even_though_retrieval_is_split(tmp_path):
    """The acoustic family needs it, so the validator's demand is unchanged."""
    cell = a_cell()
    data, reasons = C.load_cell(write_cell(str(tmp_path), cell, by_scene=None),
                                expectation())
    assert data is None and any("by_scene" in r for r in reasons), reasons


def test_report_labels_every_metrics_aggregation(tmp_path):
    root = str(tmp_path)
    write_grid(root)
    text = C.render_tables(_results(root))
    assert "T60 (scene-mean)" in text, "the acoustic metric is not labelled"
    assert "RIR_to_GT_RIR_R@1 (split)" in text, "the retrieval metric is not labelled"
    assert "FD (split)" in text
    assert "C50 (scene-mean)" in text


def test_the_planner_ruling_is_recorded_verbatim():
    """Pre-registered before any cell ran; it must be readable in the artifact,
    not only in a worklog entry."""
    source = open(C.__file__).read()
    for phrase in ("different, easier task", "noise-floor calibration",
                   "small-sample"):
        assert phrase in source, f"the ruling's reasoning is missing: {phrase!r}"
    rendered = C.render_tables(C.build_results(
        C.CellStore("/nowhere", expectation(), [], [], []),
        generated_at="2026-08-11T00:00:00-04:00"))
    assert "scene-mean" in rendered and "split-level" in rendered


def test_gate_definitions_name_the_source_they_read(tmp_path):
    root = str(tmp_path)
    write_grid(root)
    gates = C.evaluate_gates(C.collect_cells(root, expectation()))
    assert "scene-mean" in gates["G2"]["definition"].lower(), gates["G2"]["definition"]
    g1 = gates["G1"]["definition"].lower()
    assert "source" in g1 and "scene-mean" in g1 and "split" in g1, g1


def test_g2_follows_the_scene_mean_not_the_split_t60(tmp_path):
    """G2's measurand is T60, so it must move with the SCENE-MEAN T60. Here the
    split-level T60 shows a huge degradation and the scene-mean shows none: the
    gate must fail."""
    root = str(tmp_path)
    grid = [c for c in V.expected_grid() if not (c.arm == "VANL" and c.cell == "vctl")]
    write_grid(root, grid)
    control = V.Cell("VANL", "vctl", V.STEP, 42, 8, 90.0)
    reference = synthetic_metrics(a_cell(arm="VANL", cell="zref"))
    flat = dict(reference, T60=reference["T60"] + 50.0)   # split says: huge degradation
    write_cell(root, control, metrics=reference, record_patch={"metrics": flat})
    gates = C.evaluate_gates(C.collect_cells(root, expectation()))
    assert gates["G2"]["status"] == "FAIL", (
        "G2 read the split-level T60 instead of the ruled scene-mean")


# --------------------------------------------------------------------------- #
# 18. round-3 CLOSURE fix FX5 (finding B6) — G5 must compare LIKE estimands
#
# After the per-metric ruling, exp_14's T60 observation is the scene-mean while
# exp_11's committed rows only ever carried the flat split-level number. G5 is a
# REPRODUCTION check, so it has to read exp_14's flat T60 too — otherwise its
# difference and 3σ threshold compare two different quantities and would flag (or
# excuse) a discrepancy that is really just the aggregation.
# --------------------------------------------------------------------------- #
def _exp11_conf_rows(root, arm, k, value, seeds=V.SEEDS, step=V.STEP):
    """Five committed exp_11 conf rows for one (arm, K), at a chosen T60."""
    orbit = V.TRAIN_ORBIT[arm]
    suffix = "" if orbit == 0 else f"_fa_invariant_a{orbit}"
    d = os.path.join(root, f"exp11_{arm}", f"FLAC_exp11_{arm}", f"exp11_{arm}",
                     "checkpoints")
    os.makedirs(d, exist_ok=True)
    for i, seed in enumerate(seeds):
        name = (f"epoch=8-step={step}_metrics_1_1.0_exp11_{arm}_conf_S{step}"
                f"_s{seed}_K{k}{suffix}.json")
        with open(os.path.join(d, name), "w") as fh:
            json.dump({"metrics": {"T60": value + i * 0.01,
                                   "RIR_to_GT_RIR_R@1": 5.0}}, fh)


def test_g5_compares_flat_to_flat_not_scene_mean_to_flat(tmp_path):
    root = str(tmp_path)
    # every exp_14 cell: scene-mean T60 = 9.0-ish, flat (split) T60 = 20.0
    for cell in V.expected_grid():
        flat = dict(synthetic_metrics(cell), T60=20.0)
        write_cell(root, cell, record_patch={"metrics": flat})
    _exp11_conf_rows(root, "C8", 8, 20.0)          # exp_11 agrees with the FLAT value
    gates = C.evaluate_gates(C.collect_cells(root, expectation()), exp11_root=root)
    rows = [r for r in gates["G5"]["rows"] if r["arm"] == "C8" and r["metric"] == "T60"]
    assert rows, gates["G5"]
    row = rows[0]
    assert row["exp14_mean"] == pytest.approx(20.0, abs=0.05), (
        "G5 read the scene-mean T60 against exp_11's split-level number")
    assert not row["beyond"], row
    assert "flat" in gates["G5"]["definition"].lower() or "split" in row.get("source", "")


def test_g5_flags_a_real_flat_discrepancy(tmp_path):
    root = str(tmp_path)
    for cell in V.expected_grid():
        write_cell(root, cell, record_patch={"metrics": dict(synthetic_metrics(cell),
                                                             T60=20.0)})
    _exp11_conf_rows(root, "C8", 8, 30.0)          # a genuine 10-point difference
    gates = C.evaluate_gates(C.collect_cells(root, expectation()), exp11_root=root)
    row = [r for r in gates["G5"]["rows"] if r["arm"] == "C8" and r["metric"] == "T60"][0]
    assert row["beyond"], row
    assert gates["all_passed"] is True, "G5 must never gate, however loudly it reports"


def test_g5_labels_the_source_it_read(tmp_path):
    root = str(tmp_path)
    for cell in V.expected_grid():
        write_cell(root, cell)
    _exp11_conf_rows(root, "C8", 8, synthetic_metrics(a_cell(arm="C8"))["T60"])
    results = _results(root, exp11_root=root)
    text = C.render_tables(results)
    assert "split-level" in text.split("G5")[1][:600], (
        "the G5 line does not say which estimand it compared")
