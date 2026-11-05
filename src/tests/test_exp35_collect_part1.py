"""exp_35 round 5a (ii): the exp35 collect mode, part 1 (plan_orbitrir_yaw_evidence.md r2 §3c E1, §4 E2, §5 items
1-5): the hash-pinned external exp_25 results (``ec3fcd08…``), the hash-pinned stream registry, the CTL hard stop
(``--exp35-ctl-check K``: exit 0 equal / 1 mismatch / 2 refused) and E2 (``--exp35-e2``: the 45-degree source).

Synthetic cells come from the shared fixture (real evaluator functions); the pinned constants are re-pointed at
the fixtures where a test needs it, and the REAL committed exp_25 JSON / registry are checked as they are.

Written by the exp_35 Coder (Claude Opus 5.5, max effort) per plan_orbitrir_yaw_evidence.md r2/r3.
"""
import copy
import csv
import hashlib
import json
import math
import os

import pytest

import test_depth_canon_collect as T
import test_exp35_stream_registry as TR

C, MANIFEST, PIN = T.C, T.MANIFEST, T.PIN
REF_PIN = "e" * 40                    # the fixture's "exp_25 pin" for the CTL reference cells


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _row(name):
    return next(c for c in MANIFEST["cells"] if c["cell"] == name)


def _paths(tmp, name):
    r = _row(name)
    return C.cell_paths(r, os.path.join(T.FARM, r["arm"], T.ckpt_base(r) + ".ckpt"), os.path.join(str(tmp), "dumps"))


# --------------------------------------------------------------------------- #
# §5 items 1-3: the external exp_25 results, hash-pinned
# --------------------------------------------------------------------------- #
def test_the_committed_exp25_results_are_the_registered_external_reference():
    doc, reasons = C.load_exp25_reference()
    assert reasons == [] and doc["pin"] == C.EXP25_PIN == "d6ca307a85b9b0a75ec9d47f4d237e63a3db527e"
    assert C.EXP25_RESULTS_SHA256 == "ec3fcd08e3d210524b50246d9bfdf084e72835432a5d06450085e0cad12eb6e0"
    assert len(C.EXP25_MISSING) == 20 and doc["missing"] == list(C.EXP25_MISSING)
    p1 = doc["tier3"]["arms"]["P1"]["1"]["item"]["T60"]["delta_c4"]
    assert p1 == 0.9222999999999999                                        # full precision (§11 N7)


def test_a_changed_exp25_results_file_is_refused(tmp_path):
    path = tmp_path / "r.json"
    path.write_bytes(open(os.path.join(T.EXPDIR, "depth_canon_results.json"), "rb").read() + b" ")
    doc, reasons = C.load_exp25_reference(str(path))
    assert doc is None and any("sha256" in x for x in reasons)


def _ref_doc():
    with open(os.path.join(T.EXPDIR, "depth_canon_results.json")) as f:
        return json.load(f)


REF_EDITS = {
    "schema": (lambda d: d.update(schema_version=2), "schema_version"),
    "pin": (lambda d: d.update(pin="b" * 40), "pin"),
    "not_partial": (lambda d: d.update(partial=False), "partial"),
    "missing": (lambda d: d["missing"].pop(), "missing"),
    "invalid": (lambda d: d["invalid"].update(x=["y"]), "invalid"),
    "nan": (lambda d: d["tier3"]["arms"]["CANON"]["8"]["item"]["EDT"].update(delta_r=float("nan")), "delta_r"),
    "null": (lambda d: d["tier3"]["arms"]["YAW"]["1"]["item"]["FD"].update(probe45=None), "probe45"),
    "n_z": (lambda d: d["tier3"]["arms"]["P1"]["8"]["item"]["C50"].update(n_z=4), "n_z"),
    "n_r": (lambda d: d["tier3"]["arms"]["P1"]["1"]["item"]["T60"].update(n_r_pairs=4), "n_r_pairs"),
    "table_n": (lambda d: d["table"]["CANON"]["1"]["item"]["RIR_to_GT_RIR_R@5"].update(n=4), "table"),
    "table_sd": (lambda d: d["table"]["P1"]["8"]["item"]["T60"].update(sd=None), "table"),
}


@pytest.mark.parametrize("name", list(REF_EDITS))
def test_every_reused_exp25_field_is_checked(name):
    doc = _ref_doc()
    assert C.exp25_reference_reasons(doc) == []
    edit, needle = REF_EDITS[name]
    edit(doc)
    reasons = C.exp25_reference_reasons(doc)
    assert reasons and any(needle in x for x in reasons), (name, reasons)


# --------------------------------------------------------------------------- #
# §5 item 5: the stream registry
# --------------------------------------------------------------------------- #
def test_the_committed_registry_is_pinned():
    reg, reasons = C.load_registry()
    assert reasons == [] and C.EXP35_REGISTRY_SHA256 == _sha(C.EXP35_REGISTRY)
    assert reg["input_hash"]["8"]["42"].startswith("4c950c1e2ed0") and reg["assignment_hash_fixed"]["0"].startswith(
        "4894d0e119f7")
    assert C.load_registry(C.EXP35_REGISTRY + ".absent")[1]                 # a missing file is refused


@pytest.fixture()
def registry(tmp_path):
    TR._campaign(tmp_path)
    return TR._build(tmp_path)


EXP35_SAMPLE = ["exp35_P1_T_S40000_K1_s42", "exp35_BF_T_S40000_K8_s44", "exp35_BF_R_S40000_K1_s45",
                "exp35_BF_C4_S40000_K8_s42_th270", "exp35_CANON_OFF_S40000_K1_s43_th225",
                "exp35_P1_OFF_S40000_K8_s46_th135", "exp35_CANON_OFF_S40000_K8_s44_th45"]


@pytest.mark.parametrize("name", EXP35_SAMPLE)
def test_exp35_streams_reproduce_the_registry(tmp_path, registry, name):
    r = _row(name)
    T.write_cell(tmp_path, r, receipt=False)
    stream = json.load(open(os.path.join(str(tmp_path), _paths(tmp_path, name).stream)))
    assert C.registry_reasons(r, stream, registry) == []
    stream["input_hash"] = "0" * 64
    assert any("input_hash" in x for x in C.registry_reasons(r, stream, registry))


def test_another_draw_is_a_registry_mismatch(tmp_path, registry):
    r = _row("exp35_BF_R_S40000_K8_s43")
    T.write_cell(tmp_path, r, receipt=False, repeats=(1, 2))                 # another context draw
    stream = json.load(open(os.path.join(str(tmp_path), _paths(tmp_path, r["cell"]).stream)))
    assert any("input_hash" in x for x in C.registry_reasons(r, stream, registry))
    stream["assignment_hash"] = registry["assignment_hash_random"]["44"]      # another seed's assignment
    assert any("assignment_hash" in x for x in C.registry_reasons(r, stream, registry))


# --------------------------------------------------------------------------- #
# §3c E1: the CTL hard stop
# --------------------------------------------------------------------------- #
def _ctl_world(tmp, monkeypatch, k=1, ctl=None, ref=None):
    """The exp_25 reference (at REF_PIN) and the exp_35 control (at PIN) in one tree, the constants pinned to them."""
    ref_row, ctl_row = _row(f"exp25_P1_T_S40000_K{k}_s42"), _row(f"exp35_P1_T_S40000_K{k}_s42")
    T.write_cell(tmp, ref_row, pin=REF_PIN, **(ref or {}))
    T.write_cell(tmp, ctl_row, **(ctl or {}))
    rp = _paths(tmp, ref_row["cell"])
    receipt = json.load(open(os.path.join(str(tmp), rp.receipt)))
    monkeypatch.setattr(C, "EXP25_PIN", REF_PIN)
    monkeypatch.setattr(C, "CTL_REFERENCE", {**C.CTL_REFERENCE, k: {
        "receipt_sha256": _sha(os.path.join(str(tmp), rp.receipt)),
        "waveform_sha256": receipt["dump_evidence"]["manifest"]["sha256"],
        "stream_sha256": _sha(os.path.join(str(tmp), rp.stream))}})
    monkeypatch.setattr(C, "CTL_LOADED", (MANIFEST["arms"]["P1"]["ckpt"]["sha256"], T.CKPT_BYTES))
    monkeypatch.setattr(C, "CTL_DUMP_SHAPE", [T.N, T.T])


def _ctl(tmp, k=1):
    return C.exp35_ctl_check(k, pin=PIN, manifest=MANIFEST, base_dir=str(tmp), farm_root=T.FARM,
                             dump_root=os.path.join(str(tmp), "dumps"))


@pytest.mark.parametrize("k", [1, 8])
def test_an_identical_control_is_equal(tmp_path, monkeypatch, k):
    _ctl_world(tmp_path, monkeypatch, k)
    status, reasons = _ctl(tmp_path, k)
    assert (status, reasons) == ("equal", [])


def test_the_registered_ctl_reference_constants():
    assert C.CTL_REFERENCE[1]["receipt_sha256"].startswith("fa1ea729") and C.CTL_REFERENCE[8]["receipt_sha256"].startswith(
        "54490e2a")
    assert C.CTL_REFERENCE[1]["waveform_sha256"].startswith("8cb47c945d55") and C.CTL_REFERENCE[8][
        "waveform_sha256"].endswith("0fd3")
    assert C.CTL_REFERENCE[1]["stream_sha256"].endswith("793d") and C.CTL_REFERENCE[8]["stream_sha256"].endswith("4af6")
    assert C.CTL_LOADED == ("c4c678826cddda37fa4977926aadee530afd037b3abb110918b52a342ce9845c", 723927531)
    assert C.CTL_DUMP_SHAPE == [6337, 10240]
    for k in (1, 8):                                                          # the force-added exp_25 receipts
        rel = f"outputs_FLAC/exp25_depth_canon/P1/epoch=8-step=40000_metrics_1_1.0_exp25_P1_T_S40000_K{k}_s42"
        assert _sha(os.path.join(T.REPO, rel + ".cell.json")) == C.CTL_REFERENCE[k]["receipt_sha256"]
        assert _sha(os.path.join(T.REPO, rel + ".stream.json")) == C.CTL_REFERENCE[k]["stream_sha256"]


def _scaled_waves(name, eps):
    return T.waveforms_for(_row(name)) * (1 + eps)


def test_a_control_with_other_waveforms_is_a_mismatch(tmp_path, monkeypatch):
    _ctl_world(tmp_path, monkeypatch, 1, ctl=dict(waves=_scaled_waves("exp35_P1_T_S40000_K1_s42", 1e-6)))
    status, reasons = _ctl(tmp_path, 1)
    assert status == "mismatch" and any("waveform" in x for x in reasons), reasons


def test_a_control_with_other_metrics_is_a_mismatch(tmp_path, monkeypatch):
    m = dict(T.default_metrics(_row("exp35_P1_T_S40000_K8_s42")))
    m["T60"] += 1e-9
    _ctl_world(tmp_path, monkeypatch, 8, ctl=dict(metrics=m))
    status, reasons = _ctl(tmp_path, 8)
    assert status == "mismatch" and any("metrics" in x for x in reasons), reasons


def test_a_control_with_other_scene_values_is_a_mismatch(tmp_path, monkeypatch):
    m = T.default_metrics(_row("exp35_P1_T_S40000_K1_s42"))
    scenes = {s: {k: v + 0.01 * i + (1e-9 if s == "Cafe" else 0) for k, v in m.items()} for i, s in enumerate(T.SCENES)}
    _ctl_world(tmp_path, monkeypatch, 1, ctl=dict(by_scene=scenes))
    status, reasons = _ctl(tmp_path, 1)
    assert status == "mismatch" and any("by_scene" in x for x in reasons), reasons


@pytest.mark.parametrize("case", ["no_control", "reference_receipt_changed", "control_unverified"])
def test_an_undecidable_control_check_is_refused(tmp_path, monkeypatch, case):
    _ctl_world(tmp_path, monkeypatch, 1)
    if case == "no_control":
        os.unlink(os.path.join(str(tmp_path), _paths(tmp_path, "exp35_P1_T_S40000_K1_s42").metrics))
    elif case == "reference_receipt_changed":
        T.edit_json(os.path.join(str(tmp_path), _paths(tmp_path, "exp25_P1_T_S40000_K1_s42").receipt),
                    lambda d: d.update(lane=1))
    else:
        T.edit_json(os.path.join(str(tmp_path), _paths(tmp_path, "exp35_P1_T_S40000_K1_s42").receipt),
                    lambda d: d.update(attention_backend="flash"))
    status, reasons = _ctl(tmp_path, 1)
    assert status == "refused" and reasons, (case, reasons)


def test_cli_exp35_ctl_check_exit_codes(tmp_path, monkeypatch, capsys):
    _ctl_world(tmp_path, monkeypatch, 1)
    cells = tmp_path / "cells.json"
    cells.write_text(json.dumps(MANIFEST))
    argv = ["--exp35-ctl-check", "1", "--pin", PIN, "--cells", str(cells), "--base-dir", str(tmp_path),
            "--farm-root", T.FARM, "--dump-root", os.path.join(str(tmp_path), "dumps")]
    assert C.main(argv) == 0 and "CTL K1 EQUAL" in capsys.readouterr().out
    monkeypatch.setattr(C, "exp35_ctl_check", lambda *a, **k: ("mismatch", ["x"]))
    assert C.main(argv) == 1 and "MISMATCH" in capsys.readouterr().out
    monkeypatch.setattr(C, "exp35_ctl_check", lambda *a, **k: ("refused", ["y"]))
    assert C.main(argv) == 2 and "REFUSED" in capsys.readouterr().out
    assert C.main(argv[:1] + ["3"] + argv[2:]) == 2


# --------------------------------------------------------------------------- #
# §4 E2: the B-F theta = 0 cells vs the A6 CSV row (the 45-degree source only)
# --------------------------------------------------------------------------- #
CSV_COLUMNS = ["method", "K", "angle_deg", "seed", "source_json", "T60", "C50", "EDT", "FD", "RIR_to_GT_RIR_R@1",
               "RIR_to_GT_RIR_R@5", "RIR_to_GT_RIR_R@10", "RIR_to_geom_R@1", "RIR_to_geom_R@5", "RIR_to_geom_R@10"]


def _e2_world(tmp, monkeypatch, offsets=None, drop=None):
    rows = []
    for k in (1, 8):
        r = _row(f"exp35_BF_T_S40000_K{k}_s42")
        T.write_cell(tmp, r)
        m = T.default_metrics(r)
        vals = {key: round(m[key], 4) + (offsets or {}).get((k, key), 0.0) for key in C.E2_TOLERANCE}
        rows.append(dict({"method": "FA B-F@40k", "K": k, "angle_deg": 0, "seed": 42, "source_json": "x.json",
                          "RIR_to_geom_R@1": 1, "RIR_to_geom_R@5": 1, "RIR_to_geom_R@10": 1}, **vals))
        rows.append(dict(rows[-1], method="Vanilla P1@40k", T60=0.0))        # another method's row: never read
    rows = [x for x in rows if not (drop and (x["method"], x["K"]) == drop)]
    path = tmp / "yaw_metrics_raw.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    monkeypatch.setattr(C, "A6_CSV_SHA256", _sha(path))
    return str(path)


def _e2(tmp, path):
    return C.exp35_e2(pin=PIN, manifest=MANIFEST, base_dir=str(tmp), csv_path=path, farm_root=T.FARM,
                      dump_root=os.path.join(str(tmp), "dumps"))


def test_e2_passes_when_all_fourteen_differences_are_inside_the_tolerances(tmp_path, monkeypatch):
    res = _e2(tmp_path, _e2_world(tmp_path, monkeypatch, offsets={(1, "T60"): 0.0009, (8, "RIR_to_GT_RIR_R@10"): -0.19}))
    assert res["status"] == "PASS" and res["pass"] is True and len(res["rows"]) == 14
    assert all(x["pass"] for x in res["rows"]) and res["csv_sha256"] == C.A6_CSV_SHA256


@pytest.mark.parametrize("k, key, off", [(1, "T60", 0.0011), (8, "C50", 0.0006), (1, "EDT", -0.0051),
                                         (8, "FD", 0.00051), (1, "RIR_to_GT_RIR_R@1", 0.21)])
def test_e2_fails_on_any_difference_beyond_its_tolerance(tmp_path, monkeypatch, k, key, off):
    res = _e2(tmp_path, _e2_world(tmp_path, monkeypatch, offsets={(k, key): off}))
    bad = [x for x in res["rows"] if not x["pass"]]
    assert res["status"] == "FAIL" and res["pass"] is False and [(x["K"], x["metric"]) for x in bad] == [(k, key)]


def test_e2_is_refused_without_its_inputs(tmp_path, monkeypatch):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    for d in (a, b, c):
        d.mkdir()
    assert _e2(a, _e2_world(a, monkeypatch, drop=("FA B-F@40k", 8)))["status"] == "REFUSED"   # no K8 reference row
    path = _e2_world(b, monkeypatch)
    monkeypatch.setattr(C, "A6_CSV_SHA256", "0" * 64)
    assert _e2(b, path)["status"] == "REFUSED"                                              # another CSV
    path = _e2_world(c, monkeypatch)
    os.unlink(os.path.join(str(c), _paths(c, "exp35_BF_T_S40000_K1_s42").receipt))
    assert _e2(c, path)["status"] == "REFUSED"                                              # an unverified cell


def test_the_registered_e2_constants():
    assert C.E2_TOLERANCE == {"T60": 0.001, "C50": 0.0005, "EDT": 0.005, "FD": 0.0005, "RIR_to_GT_RIR_R@1": 0.2,
                              "RIR_to_GT_RIR_R@5": 0.2, "RIR_to_GT_RIR_R@10": 0.2}
    assert _sha(C.A6_CSV) == "0ae554c09f45ce89b8957cf75a3d15e44f4173f4f8987497dbbebf16bbfb23da"
    with open(C.A6_CSV) as f:
        ref = {int(r["K"]): r for r in csv.DictReader(f) if (r["method"], r["angle_deg"], r["seed"]) == (
            "FA B-F@40k", "0", "42")}
    assert (float(ref[1]["T60"]), float(ref[8]["T60"]), float(ref[8]["RIR_to_GT_RIR_R@1"])) == (9.4859, 8.189, 5.3495)


def test_cli_exp35_e2_exit_codes(tmp_path, monkeypatch, capsys):
    path = _e2_world(tmp_path, monkeypatch)
    cells = tmp_path / "cells.json"
    cells.write_text(json.dumps(MANIFEST))
    argv = ["--exp35-e2", "--pin", PIN, "--cells", str(cells), "--base-dir", str(tmp_path), "--farm-root", T.FARM,
            "--dump-root", os.path.join(str(tmp_path), "dumps"), "--a6-csv", path]
    assert C.main(argv) == 0 and "E2 PASS" in capsys.readouterr().out
    monkeypatch.setattr(C, "exp35_e2", lambda **k: {"status": "FAIL", "pass": False, "rows": [], "reasons": [],
                                                    "rule": "r"})
    assert C.main(argv) == 1
    monkeypatch.setattr(C, "exp35_e2", lambda **k: {"status": "REFUSED", "pass": None, "rows": [], "reasons": ["z"],
                                                    "rule": "r"})
    assert C.main(argv) == 2
