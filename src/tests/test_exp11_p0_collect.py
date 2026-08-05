"""Tests for the exp_11 P0 profiling collector (``p0_collect.py``).

The P0 stage runs each profiling cell TWICE (``--max-steps`` 10 and 30) and each
job prints exactly one machine-parseable ``P0RESULT`` line. The collector turns
those lines into steady-state throughput --- ``steps/s = 20 / (wall_30 -
wall_10)``, which cancels the (large, variable) startup cost --- plus peak VRAM
and the derived attribution columns (per-orbit-pass cost, DDP scaling, grad-ckpt
cost).

Everything under test is a pure function over parsed rows: no clock, no network,
no GPU, no Slurm. The only IO test uses pytest's ``tmp_path``. Failure modes are
asserted explicitly, because a profiling row that is silently dropped (missing
half a pair, OOM, invalid measurement, non-monotone walls) would otherwise turn
into a *missing* table row and a wrong rung decision.
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


def _load_module():
    spec = importlib.util.spec_from_file_location("exp11_p0_collect", _COLLECT_PY)
    assert spec is not None and spec.loader is not None, f"cannot load {_COLLECT_PY}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load_module()


def _line(cell="C4L_32x2", maxsteps=30, ngpu=2, mb=32, rc=0, wall=200.0,
          peak=20000, per_uuid="GPU-aaa:20000,GPU-bbb:19000", valid=1):
    return (
        f"P0RESULT cell={cell} maxsteps={maxsteps} ngpu={ngpu} mb={mb} rc={rc} "
        f"wall_fit={wall} peak_overall_mib={peak} peak_per_uuid={per_uuid} valid={valid}"
    )


def _row(**kw):
    row = P.parse_p0_line(_line(**kw))
    assert row is not None
    return row


def _pair_rows(cell, ngpu, mb, w10, w30, **kw):
    return [
        _row(cell=cell, maxsteps=10, ngpu=ngpu, mb=mb, wall=w10, **kw),
        _row(cell=cell, maxsteps=30, ngpu=ngpu, mb=mb, wall=w30, **kw),
    ]


# --------------------------------------------------------------------------- #
# 1. P0RESULT line parsing
# --------------------------------------------------------------------------- #
def test_parse_valid_line():
    row = P.parse_p0_line(_line())
    assert row["cell"] == "C4L_32x2"
    assert row["maxsteps"] == 30 and row["ngpu"] == 2 and row["mb"] == 32
    assert row["rc"] == 0 and row["valid"] == 1
    assert row["wall_fit"] == pytest.approx(200.0)
    assert row["peak_overall_mib"] == 20000
    assert row["peak_per_uuid"] == {"GPU-aaa": 20000, "GPU-bbb": 19000}


def test_parse_non_p0_line_returns_none():
    for junk in ("", "Epoch 0: 100%|##| 30/30", "srun: job 1 queued", "  P0RESULT-ish"):
        assert P.parse_p0_line(junk) is None


def test_parse_missing_field_raises():
    bad = _line().replace(" peak_overall_mib=20000", "")
    with pytest.raises(ValueError) as e:
        P.parse_p0_line(bad)
    assert "peak_overall_mib" in str(e.value)


def test_parse_malformed_value_raises():
    for bad in (_line().replace("wall_fit=200.0", "wall_fit=abc"),
                _line().replace("ngpu=2", "ngpu=two"),
                _line().replace("peak_per_uuid=GPU-aaa:20000,GPU-bbb:19000",
                                "peak_per_uuid=GPU-aaa:notanumber")):
        with pytest.raises(ValueError):
            P.parse_p0_line(bad)


# --------------------------------------------------------------------------- #
# 2. pairing the 10/30-step runs
# --------------------------------------------------------------------------- #
def test_pair_rows_by_cell():
    rows = _pair_rows("C4L_32x2", 2, 32, 120.0, 220.0) + _pair_rows("C8_32x2", 2, 32, 130.0, 330.0)
    pairs, problems = P.pair_rows(rows)
    assert problems == []
    assert [c for c, _ in pairs] == ["C4L_32x2", "C8_32x2"]  # deterministic order
    assert pairs[0][1]["lo"]["maxsteps"] == 10 and pairs[0][1]["hi"]["maxsteps"] == 30


def test_pair_rows_reports_incomplete_pair():
    rows = [_row(cell="C8_8x8", maxsteps=30, ngpu=8, mb=8)]
    pairs, problems = P.pair_rows(rows)
    assert pairs == []
    assert len(problems) == 1 and problems[0][0] == "C8_8x8"
    assert "10" in problems[0][1]  # names the missing half, never silently dropped


def test_pair_rows_reports_duplicate_and_shape_mismatch():
    dup = _pair_rows("C4L_32x2", 2, 32, 100.0, 200.0) + [_row(cell="C4L_32x2", maxsteps=30)]
    _, problems = P.pair_rows(dup)
    assert any("duplicate" in msg for _, msg in problems)

    mixed = [_row(cell="C8_16x4", maxsteps=10, ngpu=4, mb=16),
             _row(cell="C8_16x4", maxsteps=30, ngpu=2, mb=32)]
    _, problems = P.pair_rows(mixed)
    assert any("ngpu" in msg or "mb" in msg for _, msg in problems)


# --------------------------------------------------------------------------- #
# 3. steady-state steps/s
# --------------------------------------------------------------------------- #
def test_steps_per_second():
    assert P.steps_per_second(100.0, 200.0) == pytest.approx(0.2)
    assert P.steps_per_second(50.0, 100.0, steps_lo=10, steps_hi=30) == pytest.approx(0.4)


def test_steps_per_second_rejects_nonpositive_delta():
    for w10, w30 in ((200.0, 200.0), (300.0, 200.0)):
        with pytest.raises(ValueError):
            P.steps_per_second(w10, w30)


# --------------------------------------------------------------------------- #
# 4. per-cell summary: peaks, statuses, no silent drops
# --------------------------------------------------------------------------- #
def test_summarize_peak_is_max_over_pair():
    rows = _pair_rows("C4L_32x2", 2, 32, 100.0, 200.0)
    rows[0]["peak_overall_mib"] = 31000
    rows[0]["peak_per_uuid"] = {"GPU-aaa": 31000, "GPU-bbb": 12000}
    rows[1]["peak_overall_mib"] = 25000
    rows[1]["peak_per_uuid"] = {"GPU-aaa": 25000, "GPU-bbb": 24000}
    (summary,) = P.summarize(*P.pair_rows(rows))
    assert summary["peak_overall_mib"] == 31000
    assert summary["peak_per_gpu_max_mib"] == 31000
    assert summary["steps_s"] == pytest.approx(0.2)
    assert summary["status"] == "OK"


def test_summarize_marks_failures_without_dropping():
    oom = _pair_rows("C4L_32x2", 2, 32, 100.0, 120.0, rc=3)
    failed = _pair_rows("C8_8x8", 8, 8, 100.0, 200.0, rc=1)
    invalid = _pair_rows("VAN_16x4", 4, 16, 100.0, 200.0, valid=0)
    nonmono = _pair_rows("C8_16x4", 4, 16, 200.0, 200.0)

    summaries = P.summarize(*P.pair_rows(oom + failed + invalid + nonmono))
    by_cell = {s["cell"]: s for s in summaries}
    assert set(by_cell) == {"C4L_32x2", "C8_8x8", "VAN_16x4", "C8_16x4"}
    assert by_cell["C4L_32x2"]["status"] == "OOM"
    assert by_cell["C8_8x8"]["status"] == "FAILED"
    assert by_cell["VAN_16x4"]["status"] == "INVALID"
    assert by_cell["C8_16x4"]["status"] == "INVALID"  # nonpositive wall delta
    for cell in ("C4L_32x2", "C8_8x8", "VAN_16x4", "C8_16x4"):
        assert by_cell[cell]["steps_s"] is None
        assert by_cell[cell]["peak_overall_mib"] == 20000  # peak still reported


def test_summarize_carries_pairing_problems_as_rows():
    rows = [_row(cell="C32_32x2", maxsteps=10, ngpu=2, mb=32)]
    summaries = P.summarize(*P.pair_rows(rows))
    assert [s["cell"] for s in summaries] == ["C32_32x2"]
    assert summaries[0]["status"] == "INCOMPLETE"
    assert summaries[0]["steps_s"] is None


# --------------------------------------------------------------------------- #
# 5. derived attribution columns (pure functions over summaries)
# --------------------------------------------------------------------------- #
def _summary(cell, ngpu, mb, steps_s, status="OK", peak=20000):
    return {"cell": cell, "family": P.family_of(cell), "ngpu": ngpu, "mb": mb,
            "rung": f"{mb}x{ngpu}", "steps_s": steps_s, "peak_overall_mib": peak,
            "peak_per_gpu_max_mib": peak, "status": status}


def test_orbit_pass_fit_recovers_slope_and_intercept():
    # step time = 0.5 s + 0.25 s per ViT orbit pass; VAN=1, C4L=4, C8=8 passes.
    summaries = [
        _summary("VAN_32x2", 2, 32, 1.0 / (0.5 + 0.25 * 1)),
        _summary("C4L_32x2", 2, 32, 1.0 / (0.5 + 0.25 * 4)),
        _summary("C8_32x2", 2, 32, 1.0 / (0.5 + 0.25 * 8)),
    ]
    fit = P.orbit_pass_fit(summaries)
    assert set(fit) == {"32x2"}
    assert fit["32x2"]["slope_s_per_pass"] == pytest.approx(0.25)
    assert fit["32x2"]["intercept_s"] == pytest.approx(0.5)
    assert fit["32x2"]["n_points"] == 3


def test_orbit_pass_fit_skips_rungs_without_enough_points():
    assert P.orbit_pass_fit([_summary("C4L_8x8", 8, 8, 0.1)]) == {}
    assert P.orbit_pass_fit([_summary("VAN_8x8", 8, 8, None, status="OOM"),
                             _summary("C4L_8x8", 8, 8, 0.1)]) == {}


def test_ddp_scaling_efficiency():
    summaries = [
        _summary("C4L_32x2", 2, 32, 0.10),
        _summary("C4L_16x4", 4, 16, 0.20),   # perfect strong scaling
        _summary("C4L_8x8", 8, 8, 0.20),     # half of ideal (0.40)
    ]
    scal = P.ddp_scaling(summaries)
    eff = {e["ngpu"]: e["efficiency"] for e in scal["C4L"]}
    assert eff[2] == pytest.approx(1.0)      # reference rung
    assert eff[4] == pytest.approx(1.0)
    assert eff[8] == pytest.approx(0.5)


def test_grad_ckpt_cost():
    summaries = [_summary("C4L_32x2", 2, 32, 0.20, peak=20000),
                 _summary("CKPT4_32x2", 2, 32, 0.10, peak=30000)]
    gc = P.grad_ckpt_cost(summaries)
    assert gc["no_ckpt_speedup"] == pytest.approx(2.0)
    assert gc["delta_s_per_step"] == pytest.approx(5.0)   # 10 s vs 5 s per step
    assert gc["delta_peak_mib"] == -10000                 # peak(no-ckpt) - peak(ckpt-on)
    assert P.grad_ckpt_cost([_summary("C4L_32x2", 2, 32, 0.2)]) is None


# --------------------------------------------------------------------------- #
# 6. markdown emission is deterministic
# --------------------------------------------------------------------------- #
def test_render_markdown_deterministic_and_complete():
    summaries = P.summarize(*P.pair_rows(
        _pair_rows("C8_32x2", 2, 32, 100.0, 300.0)
        + _pair_rows("VAN_32x2", 2, 32, 100.0, 150.0)
        + _pair_rows("C4L_32x2", 2, 32, 100.0, 200.0, rc=3)
    ))
    md = P.render_markdown(summaries, sources=["b.out", "a.out"])
    shuffled = P.render_markdown(list(reversed(summaries)), sources=["a.out", "b.out"])
    assert md == shuffled, "table ordering must not depend on input order"
    for cell in ("VAN_32x2", "C4L_32x2", "C8_32x2"):
        assert cell in md
    assert "OOM" in md          # failed cells appear in the table
    assert md.count("\n| ") >= 3
    # canonical family order VAN < C4L < C8 regardless of input order
    assert md.index("VAN_32x2") < md.index("C4L_32x2") < md.index("C8_32x2")


# --------------------------------------------------------------------------- #
# 7. directory scan (the only IO; tmp_path, no clock)
# --------------------------------------------------------------------------- #
def test_scan_dir_reads_results_and_flags_resultless_logs(tmp_path):
    (tmp_path / "slurm_p0_p0-C4L_32x2-s10_1.out").write_text(
        "boot\n" + _line(cell="C4L_32x2", maxsteps=10, wall=100.0) + "\n"
    )
    (tmp_path / "slurm_p0_p0-C4L_32x2-s30_2.out").write_text(
        _line(cell="C4L_32x2", maxsteps=30, wall=200.0) + "\n"
    )
    (tmp_path / "slurm_p0_p0-C8_32x2-s10_3.out").write_text("died before the result line\n")
    (tmp_path / "unrelated.out").write_text(_line(cell="NOPE_1x1") + "\n")

    rows, problems = P.scan_dir(str(tmp_path))
    assert len(rows) == 2
    assert {r["maxsteps"] for r in rows} == {10, 30}
    assert len(problems) == 1 and "slurm_p0_p0-C8_32x2-s10_3.out" in problems[0][1]

    summaries = P.summarize(*P.pair_rows(rows))
    assert [s["cell"] for s in summaries] == ["C4L_32x2"]
    assert summaries[0]["steps_s"] == pytest.approx(0.2)
