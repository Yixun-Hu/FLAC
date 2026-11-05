"""exp_35 round 5b (ii-a): the exp35 results pipeline ``exp35_results`` (plan_orbitrir_yaw_evidence.md r2 §1
deliverables (a)/(b)/(c), §4, §5 items 1-7), frozen before any phase-(i) data exists (§11 R8):

- §5's fail-closed list refuses even with ``partial``: the external exp_25 JSON or the registry not as pinned, a cell
  that fails verification (receipt v2), more than one campaign digest, a stream off the registry, a CTL check that
  is not "equal" (item 7 -- also when the control cells have not run), an external exp_25 cell that does not verify,
  an A6 CSV that is not the pinned one, a B-F probe file that is not a valid control;
- cells not run yet, the B-F probe not run yet and an E2 that did not PASS are labelled gaps (``partial``);
- reused exp_25 rows stay under ``external_reference.exp25`` and the tables point at them (item 3);
- the A6 CSV is read by E2 only (item 6): its 45-degree rows reach table (b) on an E2 PASS, never Tier 2 / Tier 3;
- on an E2 FAIL the 45-degree FLAC / OrbitRIR points are the s42 one-pin cells (§4 E2).

The world: the shared fixture's exp_35 cells at PIN and the exp_25 CTL references in the run tree, the external
exp_25 cells in tmp/ext (the exp-25 worktree), a fixture registry, a finalized fixture B-F probe and an A6 CSV.

Written by the exp_35 Coder (Claude Opus 5.5, max effort) per plan_orbitrir_yaw_evidence.md r2/r3.
"""
import csv
import hashlib
import json
import os
import statistics

import pytest

import test_depth_canon_collect as T
import test_exp35_bf_probe_contract as TB
import test_exp35_collect_part1 as P1

C, MANIFEST, PIN = T.C, T.MANIFEST, T.PIN
REF_PIN = P1.REF_PIN
SEEDS = (42, 43, 44, 45, 46)
PHASE = ["CTL", "BF-pin", "BF-dup", "CANON-45"]
EXT = sorted([f"exp25_{a}_R_S40000_K{k}_s{s}" for a in ("P1", "CANON") for k in (1, 8) for s in SEEDS]
             + [f"exp25_{a}_C4_S40000_K{k}_s42_th45" for a in ("P1", "CANON") for k in (1, 8)])


def _row(name):
    return next(c for c in MANIFEST["cells"] if c["cell"] == name)


def _paths(name):
    row = _row(name)
    target, _ = C.expected_checkpoint(row, MANIFEST, T.canon_index())
    return C.cell_paths(row, os.path.join(T.FARM, row["arm"], os.path.basename(target)), "d")


def _flat(base, name):
    return json.load(open(os.path.join(str(base), _paths(name).metrics)))["metrics"]


def fixture_registry():
    """What exp35_make_stream_registry.py freezes, for the fixture's streams (N = 8)."""
    base = _row("exp25_P1_T_S40000_K1_s42")
    stream = lambda **kw: T.build_stream(dict(base, **kw))  # noqa: E731
    return {"input_hash": {str(k): {str(s): stream(K=k, seed=s).input_hash() for s in SEEDS} for k in (1, 8)},
            "assignment_hash_random": {str(s): stream(rotate_mode="random", rotate_seed=s, rotate_deg=None, seed=s)
                                       .assignment_hash() for s in SEEDS},
            "assignment_hash_fixed": {str(d): stream(rotate_deg=d).assignment_hash()
                                      for d in (0, 90, 180, 270, 45, 135, 225)}}


def a6_csv(tmp, monkeypatch, shift45=0.0, shift0=0.0):
    """A pinned fixture A6 CSV: theta = 0 rows = the cells' metrics (4 dp; E2 PASSES unless ``shift0`` moves the
    B-F rows), 45-degree rows offset by ``shift45`` (+0.25 for B-F)."""
    rows = []
    for method in ("Vanilla P1@40k", "FA B-F@40k"):
        for k in (1, 8):
            for angle in (0, 45):
                for s in SEEDS:
                    bf = method == "FA B-F@40k"
                    m = T.default_metrics(_row(f"exp35_BF_T_S40000_K{k}_s{s}" if bf else f"exp25_P1_T_S40000_K{k}_s{s}"))
                    off = (shift45 + 0.25 * bf) if angle else (shift0 if bf else 0.0)
                    rows.append(dict({"method": method, "K": k, "angle_deg": angle, "seed": s, "source_json": "x"},
                                     **{key: round(m[key], 4) + off for key in P1.CSV_COLUMNS[5:]}))
    path = tmp / f"yaw_metrics_raw_{shift45:g}_{shift0:g}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=P1.CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    monkeypatch.setattr(C, "A6_CSV_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    return str(path)


def world(tmp, monkeypatch, blocks=("BF-pin", "BF-dup", "CANON-45"), skip=(), ctl=True, probe=True, **csv_kw):
    """Everything the exp35 mode reads, under tmp (the run worktree) and tmp/ext (the exp-25 worktree)."""
    T.write_campaign(tmp, list(blocks), skip=skip)
    for k in (1, 8) if ctl else ():
        P1._ctl_world(tmp, monkeypatch, k)
    ext = tmp / "ext"
    for name in EXT:
        T.write_cell(ext, _row(name), pin=REF_PIN, dump_root=str(ext / "dumps"))
    monkeypatch.setattr(C, "load_registry", lambda path=None: (fixture_registry(), []))
    path = str(tmp / "orbitrir_bf_cond_probe.json")
    if probe:
        path = TB.write_bf_probe(tmp, monkeypatch, mutate=lambda r: r["provenance"].update(ckpt_bytes_loaded=T.CKPT_BYTES))
        monkeypatch.setattr(C, "PROBE_CONTRACT_BF", dict(C.PROBE_CONTRACT_BF, ckpt_bytes=T.CKPT_BYTES))
        C.finalize_bf_probe(path, TB.write_log(tmp), campaign_digest=T.DIGEST, base_dir=str(tmp))
    return {"probe": path, "csv": a6_csv(tmp, monkeypatch, **csv_kw), "ext": str(ext)}


def results(tmp, w, blocks=PHASE, **kw):
    res = C.exp35_results(MANIFEST, pin=PIN, blocks=blocks, base_dir=str(tmp), farm_root=T.FARM,
                          dump_root=os.path.join(str(tmp), "dumps"), canon_index=T.canon_index(),
                          exp25_base_dir=w["ext"], exp25_dump_root=os.path.join(w["ext"], "dumps"),
                          bf_probe_json=w["probe"], a6_csv=w["csv"], check_link=False, **kw)
    return json.loads(json.dumps(res))                       # what the results file holds


@pytest.fixture(scope="module")
def exp25():
    doc, reasons = C.load_exp25_reference()
    assert reasons == []
    return doc


def test_the_external_cells_are_the_registered_ones():
    assert sorted(C.EXP25_EXTERNAL) == EXT and C.A6_METHODS == {"FLAC": "Vanilla P1@40k", "OrbitRIR": "FA B-F@40k"}


def test_a_complete_world_gives_the_registered_results(tmp_path, monkeypatch, exp25):
    w = world(tmp_path, monkeypatch)
    r = results(tmp_path, w)
    assert r["partial"] is False and r["gaps"] == [] and r["pin"] == PIN and r["campaign_digest"] == T.DIGEST
    assert r["blocks"] == PHASE and r["e1_ctl"] == {"1": {"status": "equal", "reasons": []},
                                                    "8": {"status": "equal", "reasons": []}}
    assert r["e2"]["status"] == "PASS" and r["tier2_c4"]["pass"] is True
    assert r["tier3"]["BF"]["pass"] is True                  # the fixture's metrics do not move with yaw
    ext = r["external_reference"]
    assert ext["exp25"]["sha256"] == C.EXP25_RESULTS_SHA256 and ext["exp25"]["pin"] == C.EXP25_PIN
    assert ext["exp25"]["table"] == {a: exp25["table"][a] for a in ("P1", "CANON")}          # full precision
    assert ext["exp25"]["tier3"] == {a: exp25["tier3"]["arms"][a] for a in ("CANON", "P1", "YAW")}
    assert ext["exp25_cells"]["pin"] == C.EXP25_PIN and sorted(ext["exp25_cells"]["metrics"]) == EXT
    assert ext["exp25_cells"]["metrics"]["exp25_CANON_R_S40000_K8_s45"] == _flat(w["ext"], "exp25_CANON_R_S40000_K8_s45")
    assert ext["a6_csv"]["sha256"] == C.A6_CSV_SHA256 and ext["registry_sha256"] == C.EXP35_REGISTRY_SHA256


def test_the_tables_point_at_the_reused_rows_and_name_every_source(tmp_path, monkeypatch, exp25):
    w = world(tmp_path, monkeypatch)
    r = results(tmp_path, w)
    a, b, c = r["table_a"], r["table_b"], r["table_c"]
    assert a["FLAC"] == {"ref": "external_reference.exp25.table.P1"} and C.resolve(r, a["FLAC"]) == exp25["table"]["P1"]
    assert a["CANON"] == {"ref": "external_reference.exp25.table.CANON"}
    orbit = [_flat(tmp_path, f"exp35_BF_T_S40000_K8_s{s}")["T60"] for s in SEEDS]
    assert a["OrbitRIR"]["8"]["T60"]["values"] == orbit and a["OrbitRIR"]["source"].startswith("exp_35 BF-pin T")
    assert set(a["OrbitRIR"]["1"]) == set(C.TABLE_KEYS) and "historical_bf" in a
    assert c == {"CANON": {"ref": "external_reference.exp25.tier3.CANON"}, "P1": {"ref": "external_reference.exp25.tier3.P1"},
                 "YAW": {"ref": "external_reference.exp25.tier3.YAW"}, "BF": {"ref": "tier3.BF.arms.BF"}}
    assert C.resolve(r, c["P1"])["1"]["item"]["FD"]["delta_c4"] == exp25["tier3"]["arms"]["P1"]["1"]["item"]["FD"]["delta_c4"]
    assert C.resolve(r, c["BF"])["8"]["item"]["T60"]["n_z"] == 5
    with open(w["csv"]) as f:
        rows = list(csv.DictReader(f))
    want = statistics.mean(float(x["T60"]) for x in rows if (x["method"], x["K"], x["angle_deg"]) == ("Vanilla P1@40k", "1", "45"))
    assert abs(b["45"]["FLAC"]["1"]["T60"]["mean"] - want) < 1e-12 and b["45"]["FLAC"]["source"].startswith("A6 CSV")
    assert b["45"]["OrbitRIR"]["8"]["C50"]["n"] == 5 and b["45"]["OrbitRIR"]["source"].startswith("A6 CSV")
    canon45 = [_flat(w["ext"], "exp25_CANON_C4_S40000_K1_s42_th45")["EDT"]] + [
        _flat(tmp_path, f"exp35_CANON_OFF_S40000_K1_s{s}_th45")["EDT"] for s in SEEDS[1:]]
    assert b["45"]["CANON"]["1"]["EDT"]["values"] == canon45 and "CANON-45" in b["45"]["CANON"]["source"]
    assert b["R"]["FLAC"]["8"]["T60"]["values"] == [_flat(w["ext"], f"exp25_P1_R_S40000_K8_s{s}")["T60"] for s in SEEDS]
    assert b["R"]["CANON"]["1"]["RIR_to_GT_RIR_R@5"]["n"] == 5 and b["R"]["OrbitRIR"]["1"]["T60"]["n"] == 5
    assert all("source" in b[x][m] for x in ("45", "R") for m in ("FLAC", "CANON", "OrbitRIR"))


def test_missing_cells_are_a_labelled_gap_only_under_partial(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch, blocks=("BF-pin", "BF-dup"))           # CANON-45 not run yet
    with pytest.raises(ValueError, match="pass --partial"):
        results(tmp_path, w)
    r = results(tmp_path, w, partial=True)
    assert r["partial"] is True and any("not run" in g for g in r["gaps"]) and len(r["missing"]) == 8
    assert r["table_b"]["45"]["CANON"]["1"]["T60"]["n"] == 1                # the exp_25 s42 cell only


def test_a_b_f_probe_not_yet_run_is_a_gap(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch, probe=False)
    with pytest.raises(ValueError, match="pass --partial"):
        results(tmp_path, w)
    r = results(tmp_path, w, partial=True)
    assert r["tier2_c4"]["pass"] is None and any("probe" in g for g in r["gaps"])


def test_a_poisoned_a6_csv_leaves_tier2_and_tier3_unchanged(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch)
    r = results(tmp_path, w)
    w2 = dict(w, csv=a6_csv(tmp_path, monkeypatch, shift45=3.0))            # other 45-degree values, re-pinned
    r2 = results(tmp_path, w2)
    assert r2["tier2_c4"] == r["tier2_c4"] and r2["tier3"] == r["tier3"] and r2["e2"]["rows"] == r["e2"]["rows"]
    assert abs(r2["table_b"]["45"]["FLAC"]["1"]["T60"]["mean"] - r["table_b"]["45"]["FLAC"]["1"]["T60"]["mean"] - 3) < 1e-9


def _receipt(base, name):
    return os.path.join(str(base), _paths(name).receipt)


INTEGRITY = {
    "registry": ("stream registry", lambda tmp, w, mp: mp.setattr(C, "load_registry", lambda path=None: (
        dict(fixture_registry(), input_hash={"1": fixture_registry()["input_hash"]["1"],
                                             "8": dict(fixture_registry()["input_hash"]["8"], **{"43": "0" * 64})}), []))),
    "registry_file": ("stream registry", lambda tmp, w, mp: mp.setattr(C, "load_registry", lambda path=None: (
        None, ["exp35_exp25_stream_registry.json hashes to x"]))),
    "two_digests": ("campaign digests", lambda tmp, w, mp: T.edit_json(_receipt(tmp, "exp35_BF_R_S40000_K1_s44"),
                    lambda d: d.update(campaign_digest="e" * 64, spawn_digest="e" * 64, post_digest="e" * 64))),
    "v1_receipt": ("failed verification", lambda tmp, w, mp: T.edit_json(_receipt(tmp, "exp35_BF_T_S40000_K8_s45"),
                   lambda d: d.update(schema_version=1))),
    "ctl_mismatch": ("CTL K8 mismatch", lambda tmp, w, mp: mp.setattr(
        C, "exp35_ctl_check", lambda k, **kw: ("mismatch", ["waveform"]) if k == 8 else ("equal", []))),
    "exp25_reference": ("exp_25 results", lambda tmp, w, mp: mp.setattr(C, "EXP25_RESULTS_SHA256", "0" * 64)),
    "external_cell": ("external exp_25 cells", lambda tmp, w, mp: T.edit_json(
        _receipt(w["ext"], "exp25_P1_R_S40000_K1_s44"), lambda d: d.update(rc=1))),
    "a6_csv": ("A6 CSV", lambda tmp, w, mp: mp.setattr(C, "A6_CSV_SHA256", "0" * 64)),
    "probe": ("B-F probe", lambda tmp, w, mp: T.edit_json(w["probe"], lambda d: d.update(registered=False))),
}


@pytest.mark.parametrize("case", sorted(INTEGRITY))
def test_an_integrity_problem_refuses_even_under_partial(tmp_path, monkeypatch, case):
    needle, spoil = INTEGRITY[case]
    w = world(tmp_path, monkeypatch)
    spoil(tmp_path, w, monkeypatch)
    with pytest.raises(ValueError, match="RESULTS REFUSED") as err:
        results(tmp_path, w, partial=True)
    assert needle in str(err.value) and "pass --partial" not in str(err.value)


def test_control_cells_that_have_not_run_refuse_the_results(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch, ctl=False)                             # §5 item 7: E1 must be "equal"
    with pytest.raises(ValueError, match="CTL K1 refused"):
        results(tmp_path, w, partial=True)


def test_the_exp35_mode_takes_exp35_blocks_only(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="exp_35 blocks only"):
        results(tmp_path, w, blocks=["BF-pin", "T"], partial=True)


def test_an_e2_failure_takes_the_s42_one_pin_cells(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch, shift0=0.01)                           # |dT60| 0.01 > 0.001
    with pytest.raises(ValueError, match="pass --partial"):
        results(tmp_path, w)
    r = results(tmp_path, w, partial=True)
    b45 = r["table_b"]["45"]
    assert r["e2"]["status"] == "FAIL" and r["e2"]["rows45"] == [] and any("E2 FAIL" in g for g in r["gaps"])
    assert b45["FLAC"]["1"]["T60"]["values"] == [_flat(w["ext"], "exp25_P1_C4_S40000_K1_s42_th45")["T60"]]
    assert b45["OrbitRIR"]["8"]["T60"]["values"] == [_flat(tmp_path, "exp35_BF_C4_S40000_K8_s42_th45")["T60"]]
    assert "E2 FAIL" in b45["FLAC"]["source"] and "contingency" in b45["OrbitRIR"]["source"]
    assert r["tier2_c4"]["pass"] is True and r["tier3"]["BF"]["pass"] is True      # E2 decides the 45-degree source only


def test_an_e2_without_its_cells_leaves_the_45_degree_points_pending(tmp_path, monkeypatch):
    w = world(tmp_path, monkeypatch, skip={"exp35_BF_T_S40000_K8_s42"})
    r = results(tmp_path, w, partial=True)
    assert r["e2"]["status"] == "REFUSED" and r["table_b"]["45"]["FLAC"]["source"].startswith("pending")
    assert r["table_b"]["45"]["OrbitRIR"]["1"] is None and any("E2 REFUSED" in g for g in r["gaps"])
    assert r["tier3"]["BF"]["pass"] is None and r["table_b"]["45"]["CANON"]["8"]["T60"]["n"] == 5
