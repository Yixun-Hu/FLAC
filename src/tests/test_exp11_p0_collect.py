"""Tests for the exp_11 P0 profiling collector (``p0_collect.py``).

Each P0 cell is ONE 30-step job that prints one ``P0RESULT`` line carrying the
in-fit window timestamps produced by ``p0_runner``'s callback; the collector
turns those into ``steps/s = 20 / (t30_mono - t10_mono)`` plus peak VRAM, the
in-window utilisation/power summary read back from the poller CSV, and the
derived attribution columns.

Round-2 review shape (all asserted below):
* collection is bound to ONE submission manifest (runid + sha + expected job
  ids + per-cell config sha); cross-run or mislabelled rows are refused (B4),
* every expected cell yields a row — PENDING/MISSING/MALFORMED/INVALID/OOM/
  FAILED/OK — and anything short of all-OK is a nonzero exit *before* the
  derived tables are computed (B4),
* the orbit fit needs the exact {VAN, C4L, C8} set, rejects non-finite values,
  and flags implausible outputs as AMBIGUOUS instead of interpreting them (B6),
* ``rc=5`` / ``valid=0`` classify as INVALID, not generic FAILED (N9).

Everything is a pure function over parsed rows; the only IO uses ``tmp_path``.
"""
import importlib.util
import os

import pytest


_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # src/tests/ -> src/ -> repo root
_COLLECT_PY = os.path.join(
    _REPO_ROOT, "worklog", "worklog_yixun", "exp_11_fa_orbit_claude", "p0_collect.py"
)

RUNID = "43a4d5b-1754400000"
SHA = "43a4d5b33553022720d9b80aaaf115947898ce51"
CFG_SHA = "a" * 64


def _load_module():
    spec = importlib.util.spec_from_file_location("exp11_p0_collect", _COLLECT_PY)
    assert spec is not None and spec.loader is not None, f"cannot load {_COLLECT_PY}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load_module()


def _line(cell="C4L_32x2", jobid=1001, ngpu=2, mb=32, rc=0, t10=1000.0, t30=1100.0,
          peak=20000, valid=1, runid=RUNID, sha=SHA, config_sha=CFG_SHA,
          per_uuid=None, vram_csv="p0_C4L_32x2_vram.csv", maxsteps=30):
    per_uuid = per_uuid or "GPU-aaa:20000,GPU-bbb:19000"
    return (
        f"P0RESULT runid={runid} sha={sha} jobid={jobid} cell={cell} maxsteps={maxsteps} "
        f"ngpu={ngpu} mb={mb} rc={rc} config_sha={config_sha} wall_fit={t30 - t10 + 300:.3f} "
        f"t10={t10 + 1.7e9:.3f} t30={t30 + 1.7e9:.3f} t10_mono={t10:.3f} t30_mono={t30:.3f} "
        f"peak_overall_mib={peak} peak_per_uuid={per_uuid} vram_csv={vram_csv} valid={valid}"
    )


def _row(**kw):
    row = P.parse_p0_line(_line(**kw))
    assert row is not None
    return row


def _manifest_text(cells=(("C4L_32x2", 1001), ("C8_32x2", 1002)), runid=RUNID, sha=SHA):
    lines = ["# exp_11 P0 submission manifest", f"runid {runid}", f"sha {sha}",
             "submitted_at 1754400000"]
    lines += [f"cell {cell} 30 {jobid} {CFG_SHA}" for cell, jobid in cells]
    return "\n".join(lines) + "\n"


def _manifest(**kw):
    return P.parse_manifest(_manifest_text(**kw))


# --------------------------------------------------------------------------- #
# 1. P0RESULT parsing
# --------------------------------------------------------------------------- #
def test_parse_valid_line():
    row = P.parse_p0_line(_line())
    assert row["runid"] == RUNID and row["sha"] == SHA and row["jobid"] == 1001
    assert row["cell"] == "C4L_32x2" and row["ngpu"] == 2 and row["mb"] == 32
    assert row["rc"] == 0 and row["valid"] == 1 and row["config_sha"] == CFG_SHA
    assert row["t30_mono"] - row["t10_mono"] == pytest.approx(100.0)
    assert row["t10"] > 1.7e9 and row["t30"] > row["t10"]
    assert row["peak_overall_mib"] == 20000
    assert row["peak_per_uuid"] == {"GPU-aaa": 20000, "GPU-bbb": 19000}
    assert row["vram_csv"] == "p0_C4L_32x2_vram.csv"


def test_parse_non_p0_line_returns_none():
    for junk in ("", "Epoch 0: 100%|##| 30/30", "P0STEP step=10 t=1.0 ts=2.0", " P0RESULT-ish"):
        assert P.parse_p0_line(junk) is None


def test_parse_missing_field_raises():
    for field in ("peak_overall_mib", "runid", "config_sha", "t30_mono", "jobid"):
        bad = " ".join(t for t in _line().split() if not t.startswith(f"{field}="))
        with pytest.raises(ValueError) as e:
            P.parse_p0_line(bad)
        assert field in str(e.value)


def test_parse_rejects_malformed_and_nonfinite_values():
    bads = [
        _line().replace("t10_mono=1000.000", "t10_mono=abc"),
        _line().replace("ngpu=2", "ngpu=two"),
        _line().replace("peak_per_uuid=GPU-aaa:20000,GPU-bbb:19000", "peak_per_uuid=GPU-aaa:x"),
        _line().replace("t30_mono=1100.000", "t30_mono=nan"),      # B6: NaN must not parse
        _line().replace("t10_mono=1000.000", "t10_mono=inf"),
        _line(peak=-5),                                            # negative peak
        _line(valid=7),                                            # valid must be 0/1
    ]
    for bad in bads:
        with pytest.raises(ValueError):
            P.parse_p0_line(bad)


# --------------------------------------------------------------------------- #
# 2. manifest parsing and row admission (B4)
# --------------------------------------------------------------------------- #
def test_parse_manifest():
    man = _manifest()
    assert man["runid"] == RUNID and man["sha"] == SHA
    assert [e["cell"] for e in man["expected"]] == ["C4L_32x2", "C8_32x2"]
    assert man["expected"][0]["jobid"] == 1001
    assert man["expected"][0]["config_sha"] == CFG_SHA
    assert man["expected"][0]["maxsteps"] == 30


def test_parse_manifest_rejects_broken_manifests():
    with pytest.raises(ValueError):
        P.parse_manifest("runid x\n")                       # no sha, no cells
    with pytest.raises(ValueError):
        P.parse_manifest(_manifest_text().replace("runid", "runidx", 1))
    dup = _manifest_text(cells=(("C4L_32x2", 1), ("C4L_32x2", 2)))
    with pytest.raises(ValueError):
        P.parse_manifest(dup)


def test_admit_rows_accepts_matching_rows():
    man = _manifest()
    rows = [_row(cell="C4L_32x2", jobid=1001), _row(cell="C8_32x2", jobid=1002)]
    admitted, rejected = P.admit_rows(rows, man)
    assert sorted(admitted) == ["C4L_32x2", "C8_32x2"] and rejected == []


def test_admit_rows_refuses_cross_run_and_mislabelled_rows():
    man = _manifest()
    cases = {
        "runid": _row(cell="C4L_32x2", jobid=1001, runid="deadbee-1"),
        "sha": _row(cell="C4L_32x2", jobid=1001, sha="f" * 40),
        "jobid": _row(cell="C4L_32x2", jobid=9999),
        "cell": _row(cell="VAN_8x8", jobid=1001),
        "config_sha": _row(cell="C4L_32x2", jobid=1001, config_sha="b" * 64),
    }
    for label, row in cases.items():
        admitted, rejected = P.admit_rows([row], man)
        assert admitted == {}, f"{label} mismatch was admitted"
        assert len(rejected) == 1 and label in rejected[0][1]


def test_admit_rows_refuses_duplicate_cell():
    man = _manifest()
    rows = [_row(cell="C4L_32x2", jobid=1001), _row(cell="C4L_32x2", jobid=1001, t30=1200.0)]
    admitted, rejected = P.admit_rows(rows, man)
    assert "C4L_32x2" not in admitted
    assert any("duplicate" in msg for _, msg in rejected)


# --------------------------------------------------------------------------- #
# 3. in-fit rate
# --------------------------------------------------------------------------- #
def test_steps_per_second():
    assert P.steps_per_second(1000.0, 1100.0) == pytest.approx(0.2)
    assert P.steps_per_second(0.0, 10.0) == pytest.approx(2.0)


def test_steps_per_second_rejects_nonpositive_and_nonfinite():
    for lo, hi in ((200.0, 200.0), (300.0, 200.0), (float("nan"), 200.0),
                   (100.0, float("inf"))):
        with pytest.raises(ValueError):
            P.steps_per_second(lo, hi)


# --------------------------------------------------------------------------- #
# 4. one row per EXPECTED cell, with the right status (B4, N9)
# --------------------------------------------------------------------------- #
def test_summarize_ok_row():
    man = _manifest(cells=(("C4L_32x2", 1001),))
    admitted, _ = P.admit_rows([_row(cell="C4L_32x2", jobid=1001)], man)
    (s,) = P.summarize(man, admitted)
    assert s["status"] == "OK" and s["steps_s"] == pytest.approx(0.2)
    assert s["peak_overall_mib"] == 20000 and s["peak_per_gpu_max_mib"] == 20000
    assert s["ngpu"] == 2 and s["mb"] == 32 and s["rung"] == "32x2"


def test_summarize_status_classes():
    cells = (("C4L_32x2", 1), ("C8_32x2", 2), ("VAN_32x2", 3), ("C4L_16x4", 4),
             ("C8_16x4", 5), ("VAN_16x4", 6), ("C4L_8x8", "SUBMIT_FAILED"))
    man = _manifest(cells=cells)
    rows = [
        _row(cell="C4L_32x2", jobid=1, rc=3),                     # OOM
        _row(cell="C8_32x2", jobid=2, rc=5),                      # measurement invalid
        _row(cell="VAN_32x2", jobid=3, valid=0),                  # measurement invalid
        _row(cell="C4L_16x4", jobid=4, rc=1),                     # generic failure
        _row(cell="C8_16x4", jobid=5, t10=100.0, t30=100.0),      # zero window
        # VAN_16x4 (jobid 6): no result at all -> PENDING
    ]
    admitted, _ = P.admit_rows(rows, man)
    by_cell = {s["cell"]: s for s in P.summarize(man, admitted)}

    assert set(by_cell) == {c for c, _ in cells}, "every expected cell needs a row"
    assert by_cell["C4L_32x2"]["status"] == "OOM"
    assert by_cell["C8_32x2"]["status"] == "INVALID"   # rc=5 before the generic branch
    assert by_cell["VAN_32x2"]["status"] == "INVALID"
    assert by_cell["C4L_16x4"]["status"] == "FAILED"
    assert by_cell["C8_16x4"]["status"] == "INVALID"
    assert by_cell["VAN_16x4"]["status"] == "PENDING"
    assert by_cell["C4L_8x8"]["status"] == "MISSING"
    for s in by_cell.values():
        assert s["steps_s"] is None
    assert "measurement" in by_cell["C8_32x2"]["note"].lower()
    assert by_cell["C4L_32x2"]["peak_overall_mib"] == 20000  # OOM peak still reported


def test_summarize_marks_malformed_files():
    man = _manifest(cells=(("C4L_32x2", 1001),))
    summaries = P.summarize(man, {}, malformed=[("C4L_32x2", "two P0RESULT lines")])
    assert summaries[0]["status"] == "MALFORMED"
    assert "two P0RESULT" in summaries[0]["note"]


def test_all_ok_gate():
    man = _manifest(cells=(("C4L_32x2", 1001),))
    admitted, _ = P.admit_rows([_row(cell="C4L_32x2", jobid=1001)], man)
    assert P.all_ok(P.summarize(man, admitted)) is True
    assert P.all_ok(P.summarize(man, {})) is False


# --------------------------------------------------------------------------- #
# 5. poller window summary (B3): util/power over t10 -> t30 only
# --------------------------------------------------------------------------- #
def test_summarize_poller_window(tmp_path):
    csv = tmp_path / "p0_vram.csv"
    rows = []
    for tick, ts, mem, util, power in (
        (0, 100.0, 1000, 5, 70.0),      # before the window -> excluded
        (1, 150.0, 30000, 90, 300.0),   # in window
        (2, 160.0, 31000, 80, 280.0),   # in window
        (3, 400.0, 500, 0, 60.0),       # after the window -> excluded
    ):
        for uuid in ("GPU-aaa", "GPU-bbb"):
            rows.append(f"tick={tick} uuid={uuid} mem={mem} util={util} power={power} ts={ts}")
    csv.write_text("\n".join(rows) + "\n")

    summary = P.summarize_poller(str(csv), 140.0, 200.0, ("GPU-aaa", "GPU-bbb"))
    assert sorted(summary) == ["GPU-aaa", "GPU-bbb"]
    for per in summary.values():
        assert per["ticks"] == 2
        assert per["mem_max_mib"] == 31000
        assert per["util_mean"] == pytest.approx(85.0)
        assert per["power_mean_w"] == pytest.approx(290.0)


def test_summarize_poller_flags_missing_uuid_and_empty_window(tmp_path):
    csv = tmp_path / "p0_vram.csv"
    csv.write_text("tick=0 uuid=GPU-aaa mem=100 util=10 power=90.0 ts=150.0\n")
    summary = P.summarize_poller(str(csv), 140.0, 200.0, ("GPU-aaa", "GPU-bbb"))
    assert summary["GPU-bbb"]["ticks"] == 0        # rank placement evidence missing
    assert summary["GPU-bbb"]["mem_max_mib"] is None
    empty = P.summarize_poller(str(csv), 1000.0, 2000.0, ("GPU-aaa",))
    assert empty["GPU-aaa"]["ticks"] == 0


# --------------------------------------------------------------------------- #
# 6. derived attribution (B3 wording, B6 gating)
# --------------------------------------------------------------------------- #
def _summary(cell, ngpu, mb, steps_s, status="OK", peak=20000):
    return {"cell": cell, "family": P.family_of(cell), "ngpu": ngpu, "mb": mb,
            "rung": f"{mb}x{ngpu}", "steps_s": steps_s, "peak_overall_mib": peak,
            "peak_per_gpu_max_mib": peak, "status": status, "note": ""}


def _triple(rung="32x2", ngpu=2, mb=32, base=0.5, per_pass=0.25):
    return [_summary(f"{fam}_{rung}", ngpu, mb, 1.0 / (base + per_pass * n))
            for fam, n in (("VAN", 1), ("C4L", 4), ("C8", 8))]


def test_orbit_pass_fit_recovers_slope_and_unattributed_residual():
    fit = P.orbit_pass_fit(_triple())
    assert set(fit) == {"32x2"}
    assert fit["32x2"]["slope_s_per_pass"] == pytest.approx(0.25)
    assert fit["32x2"]["unattributed_residual_s"] == pytest.approx(0.5)
    assert fit["32x2"]["n_points"] == 3 and fit["32x2"]["ambiguous"] is False


def test_orbit_pass_fit_requires_the_exact_van_c4l_c8_set():
    two_points = [s for s in _triple() if s["family"] != "VAN"]
    assert P.orbit_pass_fit(two_points) == {}, "two points must not be fitted (B6)"
    with_oom = two_points + [_summary("VAN_32x2", 2, 32, None, status="OOM")]
    assert P.orbit_pass_fit(with_oom) == {}
    extra = _triple() + [_summary("CKPT4_32x2", 2, 32, 0.4)]
    assert set(P.orbit_pass_fit(extra)) == {"32x2"}  # CKPT4 never enters the fit


def test_orbit_pass_fit_flags_implausible_output_as_ambiguous():
    faster_with_more_passes = [
        _summary("VAN_32x2", 2, 32, 0.10),
        _summary("C4L_32x2", 2, 32, 0.20),
        _summary("C8_32x2", 2, 32, 0.40),
    ]
    fit = P.orbit_pass_fit(faster_with_more_passes)
    assert fit["32x2"]["ambiguous"] is True         # negative slope: model does not hold
    assert P.attribution_ok(fit) is False
    assert P.attribution_ok(P.orbit_pass_fit(_triple())) is True


def test_marginal_contrast_is_reported_separately():
    contrast = P.marginal_contrast(_triple())
    # C8 - C4L = 4 extra passes x 0.25 s
    assert contrast["32x2"]["delta_s"] == pytest.approx(1.0)
    assert contrast["32x2"]["s_per_pass"] == pytest.approx(0.25)
    assert P.marginal_contrast([_summary("C4L_32x2", 2, 32, 0.2)]) == {}


def test_ddp_scaling_efficiency():
    summaries = [_summary("C4L_32x2", 2, 32, 0.10),
                 _summary("C4L_16x4", 4, 16, 0.20),
                 _summary("C4L_8x8", 8, 8, 0.20)]
    eff = {e["ngpu"]: e["efficiency"] for e in P.ddp_scaling(summaries)["C4L"]}
    assert eff[2] == pytest.approx(1.0)
    assert eff[4] == pytest.approx(1.0)
    assert eff[8] == pytest.approx(0.5)


def test_grad_ckpt_cost():
    summaries = [_summary("C4L_32x2", 2, 32, 0.20, peak=20000),
                 _summary("CKPT4_32x2", 2, 32, 0.10, peak=30000)]
    gc = P.grad_ckpt_cost(summaries)
    assert gc["no_ckpt_speedup"] == pytest.approx(2.0)
    assert gc["delta_s_per_step"] == pytest.approx(5.0)
    assert gc["delta_peak_mib"] == -10000            # peak(no-ckpt) - peak(ckpt-on)
    assert P.grad_ckpt_cost([_summary("C4L_32x2", 2, 32, 0.2)]) is None


# --------------------------------------------------------------------------- #
# 7. markdown: deterministic, and derived tables withheld while incomplete
# --------------------------------------------------------------------------- #
def test_render_markdown_deterministic_and_labels_residual():
    summaries = _triple()
    md = P.render_markdown(summaries, complete=True)
    assert md == P.render_markdown(list(reversed(summaries)), complete=True)
    assert md.index("VAN_32x2") < md.index("C4L_32x2") < md.index("C8_32x2")
    assert "unattributed" in md.lower()
    assert "residual step cost" not in md.lower(), "B3: do not call the intercept a diagnosis"
    assert md.count("\n| ") >= 3


def test_render_markdown_withholds_derived_when_incomplete():
    summaries = _triple() + [_summary("C4L_8x8", 8, 8, None, status="PENDING")]
    md = P.render_markdown(summaries, complete=False)
    assert "C4L_8x8" in md and "PENDING" in md
    assert "WITHHELD" in md.upper()
    assert "s per orbit pass" not in md, "derived attribution must not render on a partial run"


# --------------------------------------------------------------------------- #
# 8. scanning and the end-to-end exit contract
# --------------------------------------------------------------------------- #
def test_scan_dir_treats_duplicate_result_lines_as_malformed(tmp_path):
    good = _line(cell="C4L_32x2", jobid=1001)
    (tmp_path / "slurm_p0_p0-C4L_32x2_1001.out").write_text("boot\n" + good + "\n")
    (tmp_path / "slurm_p0_p0-C8_32x2_1002.out").write_text(
        _line(cell="C8_32x2", jobid=1002) + "\n" + _line(cell="C8_32x2", jobid=1002) + "\n"
    )
    (tmp_path / "slurm_p0_p0-VAN_32x2_1003.out").write_text("died early\n")

    rows, malformed = P.scan_dir(str(tmp_path))
    assert [r["cell"] for r in rows] == ["C4L_32x2"], "duplicates must not be last-wins"
    assert any("C8_32x2" in where or "1002" in where for where, _ in malformed)
    assert any("duplicate" in msg.lower() for _, msg in malformed)


def test_main_requires_manifest_and_exits_nonzero_when_incomplete(tmp_path):
    man_path = tmp_path / "p0_manifest_test.txt"
    man_path.write_text(_manifest_text(cells=(("C4L_32x2", 1001), ("C8_32x2", 1002))))
    (tmp_path / "slurm_p0_p0-C4L_32x2_1001.out").write_text(_line(cell="C4L_32x2", jobid=1001) + "\n")
    out = tmp_path / "p0_report.md"

    rc = P.main(["--manifest", str(man_path), "--dir", str(tmp_path), "--out", str(out)])
    assert rc != 0, "a pending cell must fail the collection"
    body = out.read_text()
    assert "C8_32x2" in body and "PENDING" in body
    assert "WITHHELD" in body.upper()

    with pytest.raises(SystemExit):
        P.main(["--dir", str(tmp_path)])          # --manifest is required
