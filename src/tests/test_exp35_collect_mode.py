"""exp_35 round 5b (ii-b): the page and the ``--exp35`` CLI (plan_orbitrir_yaw_evidence.md r2 §1 deliverables:
``orbitrir_yaw_results.{md,json}`` in the exp_35 folder; §5). The JSON is exactly ``exp35_results``; the page
resolves every pointer into ``external_reference`` and recomputes nothing; a refusal exits 3 and writes nothing.

The world is round 5b (ii-a)'s (``test_exp35_collect_results.world``).

Written by the exp_35 Coder (Claude Opus 5.5, max effort) per plan_orbitrir_yaw_evidence.md r2/r3.
"""
import json
import os

import pytest

import test_depth_canon_collect as T
import test_exp35_collect_results as R

C, MANIFEST, PIN = T.C, T.MANIFEST, T.PIN


def argv(tmp, w, *extra, blocks="CTL,BF-pin,BF-dup,CANON-45"):
    cells = tmp / "cells.json"
    cells.write_text(json.dumps(MANIFEST))
    return ["--exp35", "--pin", PIN, "--cells", str(cells), "--base-dir", str(tmp), "--farm-root", T.FARM,
            "--dump-root", os.path.join(str(tmp), "dumps"), "--canon-ckpt-manifest", T._canon_manifest_file(tmp),
            "--exp25-base-dir", w["ext"], "--exp25-dump-root", os.path.join(w["ext"], "dumps"),
            "--bf-probe-json", w["probe"], "--a6-csv", w["csv"], "--no-link-check", "--blocks", blocks] + list(extra)


def _out(tmp):
    os.makedirs(tmp / "out", exist_ok=True)
    return str(tmp / "out" / "orbitrir_yaw_results")


def test_the_cli_writes_the_pipeline_s_results_and_a_page_that_resolves_the_pointers(tmp_path, monkeypatch, capsys):
    w = R.world(tmp_path, monkeypatch)
    out = _out(tmp_path)
    assert C.main(argv(tmp_path, w, "--exp35-out", out)) == 0 and "complete" in capsys.readouterr().out
    res = json.load(open(out + ".json"))
    assert res == R.results(tmp_path, w)
    md = open(out + ".md").read()
    doc = C.load_exp25_reference()[0]
    for needle in ("(a) theta = 0", "(b) 45 degrees", "(b) random yaw", "(c) Tier 3", "Tier 2-C4", "E1", "E2",
                   "B-F HISTORICAL", "external_reference", "Descriptive only", "**Tier 3 B-F: PASS**", "**PASS**",
                   C._fmt(doc["table"]["P1"]["1"]["item"]["T60"], "T60"),                    # (a) FLAC, by pointer
                   C._fmt(res["table_a"]["OrbitRIR"]["8"]["C50"], "C50"),
                   C._num(doc["tier3"]["arms"]["P1"]["8"]["item"]["FD"]["delta_c4"]),       # (c) P1, by pointer
                   res["table_b"]["45"]["FLAC"]["source"], res["table_b"]["R"]["CANON"]["source"]):
        assert needle in md, needle
    assert "PARTIAL" not in md


def test_the_default_output_is_the_exp35_folder(tmp_path, monkeypatch):
    w = R.world(tmp_path, monkeypatch)
    monkeypatch.setattr(C, "EXP35_RESULTS", _out(tmp_path))
    assert C.main(argv(tmp_path, w)) == 0 and os.path.exists(_out(tmp_path) + ".md")


def test_the_registered_output_prefix():
    assert C.EXP35_RESULTS == os.path.join(C.EXP35_DIR, "orbitrir_yaw_results")
    assert C.EXP35_DIR.endswith(os.path.join("worklog", "worklog_yixun", "exp_35_orbitrir_yaw_evidence_claude"))


def test_a_refusal_exits_3_and_writes_nothing(tmp_path, monkeypatch, capsys):
    w = R.world(tmp_path, monkeypatch, ctl=False)
    out = _out(tmp_path)
    assert C.main(argv(tmp_path, w, "--exp35-out", out, "--partial")) == 3
    assert "RESULTS REFUSED" in capsys.readouterr().out and not os.listdir(tmp_path / "out")


def test_a_partial_page_is_labelled(tmp_path, monkeypatch, capsys):
    w = R.world(tmp_path, monkeypatch, blocks=("BF-pin", "BF-dup"), shift0=0.01)     # CANON-45 not run; E2 FAIL
    out = _out(tmp_path)
    assert C.main(argv(tmp_path, w, "--exp35-out", out)) == 3 and "pass --partial" in capsys.readouterr().out
    assert C.main(argv(tmp_path, w, "--exp35-out", out, "--partial")) == 0
    md, res = open(out + ".md").read(), json.load(open(out + ".json"))
    assert res["partial"] and md.startswith("# exp_35 orbitrir_yaw_evidence — results (PARTIAL)")
    assert "**PARTIAL:**" in md and "## PARTIAL — (a)" in md and all(g in md for g in res["gaps"])
    assert "E2 FAIL: the s42 one-pin SDPA cell" in md


def test_the_exp35_mode_refuses_exp25_blocks_and_the_exp25_outputs(tmp_path, monkeypatch, capsys):
    w = R.world(tmp_path, monkeypatch)
    assert C.main(argv(tmp_path, w, "--exp35-out", _out(tmp_path), blocks="BF-pin,T")) == 3
    assert "exp_35 blocks only" in capsys.readouterr().out
    for extra in (["--out-json", str(tmp_path / "x.json")], ["--out-md", str(tmp_path / "x.md")], ["--collect"]):
        with pytest.raises(SystemExit) as err:
            C.main(argv(tmp_path, w, *extra))
        assert err.value.code == 2
