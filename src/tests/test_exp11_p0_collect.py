"""Tests for the exp_11 P0 profiling collector (``p0_collect.py``).

Each P0 cell is ONE 30-step job printing one ``P0RESULT`` line carrying the in-fit
window marks from ``p0_runner``'s callback; the collector turns those into
``steps/s = 20 / (t30_mono - t10_mono)``, verifies the poller artifact, and emits
the derived attribution.

Re-review shape (all asserted below):
* the manifest declares a MODE — an attribution fit is required only for
  ``matrix``; ``spot``/``workers`` collections succeed on their own rows (B1),
* poller evidence is mandatory provenance: the CSV must exist, hash-match, and
  carry complete in-window ticks with finite util/power per UUID, else the cell
  is INVALID (B2),
* admission binds runid/sha/jobid/cell/config_sha AND the exact execution shape
  ``maxsteps`` / ``mb`` / ``ngpu`` / ``workers`` from the manifest row, never
  reconstructed from the label (B3),
* rows are keyed ``(cell, workers)`` so the 0-vs-6 worker pair lives in ONE
  manifest, and the contrast needs both halves (B4),
* the orbit fit is over the EXACT {FA1, C4L, C8} set — FA1 (single-angle
  fa_invariant) shares C4L/C8's cylindrical pose path, so the slope is the cost
  of an ADDITIONAL ViT pass; canonical VAN is a separately reported contrast,
  never inside the fit (B6).

Pure functions over parsed rows; the only IO uses ``tmp_path``.
"""
import hashlib
import importlib.util
import os

import pytest


_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # src/tests/ -> src/ -> repo root
_COLLECT_PY = os.path.join(
    _REPO_ROOT, "worklog", "worklog_yixun", "exp_11_fa_orbit_claude", "p0_collect.py"
)

RUNID = "ec0250d-1754400000123456789-3f9a1c2b"
SHA = "ec0250d294368eafdfaa953f059e17d6faa00284"
CFG_SHA = "a" * 64
CSV_SHA = "b" * 64


def _load_module():
    spec = importlib.util.spec_from_file_location("exp11_p0_collect", _COLLECT_PY)
    assert spec is not None and spec.loader is not None, f"cannot load {_COLLECT_PY}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load_module()


def _line(cell="C4L_32x2", jobid=1001, ngpu=2, mb=32, rc=0, t10=1000.0, t30=1100.0,
          peak=20000, valid=1, runid=RUNID, sha=SHA, config_sha=CFG_SHA, workers=6,
          per_uuid=None, vram_csv="p0_C4L_32x2_vram.csv", pollcsv_sha=CSV_SHA,
          maxsteps=30, wall_fit=None):
    per_uuid = per_uuid or ",".join(f"GPU-{cell}-{i}:{peak}" for i in range(ngpu))
    wall_fit = (t30 - t10 + 300.0) if wall_fit is None else wall_fit
    return (
        f"P0RESULT runid={runid} sha={sha} jobid={jobid} cell={cell} maxsteps={maxsteps} "
        f"ngpu={ngpu} mb={mb} workers={workers} rc={rc} config_sha={config_sha} "
        f"wall_fit={wall_fit:.3f} t10={t10 + 1.7e9:.3f} t30={t30 + 1.7e9:.3f} "
        f"t10_mono={t10:.3f} t30_mono={t30:.3f} peak_overall_mib={peak} "
        f"peak_per_uuid={per_uuid} vram_csv={vram_csv} pollcsv_sha256={pollcsv_sha} "
        f"valid={valid}"
    )


def _row(**kw):
    row = P.parse_p0_line(_line(**kw))
    assert row is not None
    return row


def _manifest_text(cells=(("C4L_32x2", 1001), ("C8_32x2", 1002)), runid=RUNID, sha=SHA,
                   mode="matrix", workers=6, maxsteps=30):
    lines = ["# exp_11 P0 submission manifest", f"runid {runid}", f"sha {sha}",
             f"mode {mode}", "submitted_at 1754400000"]
    for entry in cells:
        cell, jobid = entry[0], entry[1]
        w = entry[2] if len(entry) > 2 else workers
        _, mb, ngpu = P.rung_of(cell)
        lines.append(f"cell {cell} {maxsteps} {jobid} {CFG_SHA} {mb} {ngpu} {w} 00:40:00")
    return "\n".join(lines) + "\n"


def _manifest(**kw):
    return P.parse_manifest(_manifest_text(**kw))


def _write_csv(tmp_path, name, uuids, ticks=((0, 999.0), (1, 1050.0), (2, 1100.0)),
               mem=20000, util=93, power=295.5, base=1.7e9):
    body = "".join(
        f"tick={t} uuid={u} mem={mem} util={util} power={power} ts={base + ts:.3f}\n"
        for t, ts in ticks for u in uuids
    )
    (tmp_path / name).write_text(body)
    return hashlib.sha256(body.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# 1. P0RESULT parsing
# --------------------------------------------------------------------------- #
def test_parse_valid_line():
    row = P.parse_p0_line(_line())
    assert row["runid"] == RUNID and row["sha"] == SHA and row["jobid"] == 1001
    assert row["cell"] == "C4L_32x2" and row["ngpu"] == 2 and row["mb"] == 32
    assert row["workers"] == 6 and row["maxsteps"] == 30
    assert row["rc"] == 0 and row["valid"] == 1 and row["config_sha"] == CFG_SHA
    assert row["pollcsv_sha256"] == CSV_SHA and row["vram_csv"] == "p0_C4L_32x2_vram.csv"
    assert row["t30_mono"] - row["t10_mono"] == pytest.approx(100.0)


def test_parse_non_p0_line_returns_none():
    for junk in ("", "Epoch 0: 100%|##| 30/30", "P0STEP step=10 t=1.0 ts=2.0", " P0RESULT-ish"):
        assert P.parse_p0_line(junk) is None


def test_parse_missing_field_raises():
    for field in ("peak_overall_mib", "runid", "config_sha", "t30_mono", "jobid",
                  "workers", "pollcsv_sha256"):
        bad = " ".join(t for t in _line().split() if not t.startswith(f"{field}="))
        with pytest.raises(ValueError) as e:
            P.parse_p0_line(bad)
        assert field in str(e.value)


def test_parse_rejects_malformed_and_nonfinite_values():
    bads = [
        _line().replace("t10_mono=1000.000", "t10_mono=abc"),
        _line().replace("ngpu=2", "ngpu=two"),
        _line().replace("peak_per_uuid=GPU-C4L_32x2-0:20000,GPU-C4L_32x2-1:20000",
                        "peak_per_uuid=GPU-aaa:x"),
        _line().replace("t30_mono=1100.000", "t30_mono=nan"),
        _line().replace("t10_mono=1000.000", "t10_mono=inf"),
        _line(peak=-5),
        _line(valid=7),
        _line(wall_fit=0.0),        # a completed job cannot take zero wall time
        _line(wall_fit=-3.0),
    ]
    for bad in bads:
        with pytest.raises(ValueError):
            P.parse_p0_line(bad)


# --------------------------------------------------------------------------- #
# 2. manifest: mode + exact execution shape (B1, B3, B4)
# --------------------------------------------------------------------------- #
def test_parse_manifest():
    man = _manifest()
    assert man["runid"] == RUNID and man["sha"] == SHA and man["mode"] == "matrix"
    first = man["expected"][0]
    assert first["cell"] == "C4L_32x2" and first["jobid"] == 1001
    assert first["maxsteps"] == 30 and first["mb"] == 32 and first["ngpu"] == 2
    assert first["workers"] == 6 and first["timelimit"] == "00:40:00"
    assert first["key"] == ("C4L_32x2", 6)


def test_parse_manifest_rejects_broken_manifests():
    with pytest.raises(ValueError):
        P.parse_manifest("runid x\n")                                  # no sha/mode/cells
    with pytest.raises(ValueError):
        P.parse_manifest(_manifest_text().replace("mode matrix", "mode wobble"))
    with pytest.raises(ValueError):
        P.parse_manifest(_manifest_text().replace("mode matrix\n", ""))
    dup = _manifest_text(cells=(("C4L_32x2", 1), ("C4L_32x2", 2)))      # same (cell, workers)
    with pytest.raises(ValueError):
        P.parse_manifest(dup)


def test_manifest_allows_one_cell_with_two_worker_variants():
    man = _manifest(mode="workers", cells=(("C4L_32x2", 1, 0), ("C4L_32x2", 2, 6)))
    assert [e["key"] for e in man["expected"]] == [("C4L_32x2", 0), ("C4L_32x2", 6)]


# --------------------------------------------------------------------------- #
# 3. admission binds the exact execution shape (B3)
# --------------------------------------------------------------------------- #
def test_admit_rows_accepts_matching_rows():
    man = _manifest()
    rows = [_row(cell="C4L_32x2", jobid=1001), _row(cell="C8_32x2", jobid=1002)]
    admitted, rejected = P.admit_rows(rows, man)
    assert sorted(admitted) == [("C4L_32x2", 6), ("C8_32x2", 6)] and rejected == []


def test_admit_rows_refuses_cross_run_mislabelled_and_reshaped_rows():
    man = _manifest()
    cases = {
        "runid": _row(cell="C4L_32x2", jobid=1001, runid="deadbee-1-2"),
        "sha": _row(cell="C4L_32x2", jobid=1001, sha="f" * 40),
        "jobid": _row(cell="C4L_32x2", jobid=9999),
        "cell": _row(cell="VAN_8x8", jobid=1001),
        "config_sha": _row(cell="C4L_32x2", jobid=1001, config_sha="c" * 64),
        "maxsteps": _row(cell="C4L_32x2", jobid=1001, maxsteps=31),
        "mb": _row(cell="C4L_32x2", jobid=1001, mb=8),
        "ngpu": _row(cell="C4L_32x2", jobid=1001, ngpu=8),
        "workers": _row(cell="C4L_32x2", jobid=1001, workers=0),
    }
    for label, row in cases.items():
        admitted, rejected = P.admit_rows([row], man)
        assert admitted == {}, f"{label} mismatch was admitted"
        assert len(rejected) == 1 and label in rejected[0][1], f"{label}: {rejected}"


def test_admit_rows_refuses_duplicate_key():
    man = _manifest()
    rows = [_row(cell="C4L_32x2", jobid=1001), _row(cell="C4L_32x2", jobid=1001, t30=1200.0)]
    admitted, rejected = P.admit_rows(rows, man)
    assert ("C4L_32x2", 6) not in admitted
    assert any("duplicate" in msg for _, msg in rejected)


# --------------------------------------------------------------------------- #
# 4. rate + status classes
# --------------------------------------------------------------------------- #
def test_steps_per_second():
    assert P.steps_per_second(1000.0, 1100.0) == pytest.approx(0.2)


def test_steps_per_second_rejects_nonpositive_and_nonfinite():
    for lo, hi in ((200.0, 200.0), (300.0, 200.0), (float("nan"), 200.0), (100.0, float("inf"))):
        with pytest.raises(ValueError):
            P.steps_per_second(lo, hi)


def test_summarize_status_classes():
    cells = (("C4L_32x2", 1), ("C8_32x2", 2), ("VAN_32x2", 3), ("C4L_16x4", 4),
             ("C8_16x4", 5), ("VAN_16x4", 6), ("FA1_8x8", "SUBMIT_FAILED"))
    man = _manifest(cells=cells)
    rows = [
        _row(cell="C4L_32x2", jobid=1, rc=3),                       # OOM
        _row(cell="C8_32x2", jobid=2, rc=5),                        # measurement invalid
        _row(cell="VAN_32x2", jobid=3, valid=0),
        _row(cell="C4L_16x4", jobid=4, ngpu=4, mb=16, rc=1),        # generic failure
        _row(cell="C8_16x4", jobid=5, ngpu=4, mb=16, t10=100.0, t30=100.0),
        # VAN_16x4 (jobid 6): no result -> PENDING
    ]
    admitted, _ = P.admit_rows(rows, man)
    by_cell = {s["cell"]: s for s in P.summarize(man, admitted)}
    assert set(by_cell) == {c[0] for c in cells}
    assert by_cell["C4L_32x2"]["status"] == "OOM"
    assert by_cell["C8_32x2"]["status"] == "INVALID"
    assert by_cell["VAN_32x2"]["status"] == "INVALID"
    assert by_cell["C4L_16x4"]["status"] == "FAILED"
    assert by_cell["C8_16x4"]["status"] == "INVALID"
    assert by_cell["VAN_16x4"]["status"] == "PENDING"
    assert by_cell["FA1_8x8"]["status"] == "MISSING"
    assert all(s["steps_s"] is None for s in by_cell.values())


def test_all_ok_gate():
    man = _manifest(cells=(("C4L_32x2", 1001),))
    admitted, _ = P.admit_rows([_row(cell="C4L_32x2", jobid=1001)], man)
    assert P.all_ok(P.summarize(man, admitted)) is True
    assert P.all_ok(P.summarize(man, {})) is False


# --------------------------------------------------------------------------- #
# 5. poller evidence is MANDATORY provenance (B2)
# --------------------------------------------------------------------------- #
def _ok_cell(tmp_path, cell="C4L_32x2", jobid=1001, ngpu=2, **csv_kw):
    uuids = [f"GPU-{cell}-{i}" for i in range(ngpu)]
    name = f"p0_{cell}_vram.csv"
    digest = _write_csv(tmp_path, name, uuids, **csv_kw)
    man = _manifest(cells=((cell, jobid),))
    row = _row(cell=cell, jobid=jobid, ngpu=ngpu, vram_csv=name, pollcsv_sha=digest)
    admitted, _ = P.admit_rows([row], man)
    return man, admitted


def test_poller_evidence_accepted_when_complete(tmp_path):
    man, admitted = _ok_cell(tmp_path)
    summaries, poller = P.apply_poller_evidence(P.summarize(man, admitted), str(tmp_path))
    assert summaries[0]["status"] == "OK"
    per = poller[("C4L_32x2", 6)]
    assert all(v["ticks"] >= 1 for v in per.values())
    assert all(v["util_mean"] == pytest.approx(93.0) for v in per.values())
    assert all(v["power_mean_w"] == pytest.approx(295.5) for v in per.values())


def test_poller_evidence_missing_file_is_invalid(tmp_path):
    man, admitted = _ok_cell(tmp_path)
    os.remove(tmp_path / "p0_C4L_32x2_vram.csv")
    summaries, _ = P.apply_poller_evidence(P.summarize(man, admitted), str(tmp_path))
    assert summaries[0]["status"] == "INVALID" and "missing" in summaries[0]["note"].lower()


def test_poller_evidence_hash_mismatch_is_invalid(tmp_path):
    man, admitted = _ok_cell(tmp_path)
    (tmp_path / "p0_C4L_32x2_vram.csv").write_text("tick=0 uuid=GPU-x mem=1 util=1 power=1 ts=1\n")
    summaries, _ = P.apply_poller_evidence(P.summarize(man, admitted), str(tmp_path))
    assert summaries[0]["status"] == "INVALID" and "hash" in summaries[0]["note"].lower()


def test_poller_evidence_requires_every_uuid_in_window(tmp_path):
    # every tick lies outside the measured window -> no rank-placement evidence
    man, admitted = _ok_cell(tmp_path, ticks=((0, 10.0), (1, 20.0)))
    summaries, _ = P.apply_poller_evidence(P.summarize(man, admitted), str(tmp_path))
    assert summaries[0]["status"] == "INVALID"
    assert "window" in summaries[0]["note"].lower() or "tick" in summaries[0]["note"].lower()


def test_poller_evidence_requires_finite_util_and_power(tmp_path):
    man, admitted = _ok_cell(tmp_path, util="[N/A]")
    summaries, _ = P.apply_poller_evidence(P.summarize(man, admitted), str(tmp_path))
    assert summaries[0]["status"] == "INVALID"
    assert "util" in summaries[0]["note"].lower()


# --------------------------------------------------------------------------- #
# 6. derived attribution: FA1-anchored fit, VAN as a separate contrast (B6)
# --------------------------------------------------------------------------- #
def _summary(cell, ngpu, mb, steps_s, status="OK", peak=20000, workers=6):
    return {"cell": cell, "family": P.family_of(cell), "ngpu": ngpu, "mb": mb,
            "rung": f"{mb}x{ngpu}", "steps_s": steps_s, "peak_overall_mib": peak,
            "peak_per_gpu_max_mib": peak, "status": status, "note": "", "workers": workers,
            "key": (cell, workers)}


def _triple(rung="32x2", ngpu=2, mb=32, base=0.5, per_pass=0.25, van=None):
    out = [_summary(f"{fam}_{rung}", ngpu, mb, 1.0 / (base + per_pass * n))
           for fam, n in (("FA1", 1), ("C4L", 4), ("C8", 8))]
    if van is not None:
        out.append(_summary(f"VAN_{rung}", ngpu, mb, 1.0 / van))
    return out


def test_orbit_pass_fit_uses_fa1_c4l_c8():
    fit = P.orbit_pass_fit(_triple())
    assert set(fit) == {"32x2"}
    assert fit["32x2"]["slope_s_per_pass"] == pytest.approx(0.25)
    assert fit["32x2"]["unattributed_residual_s"] == pytest.approx(0.5)
    assert fit["32x2"]["families"] == ["C4L", "C8", "FA1"]
    assert fit["32x2"]["ambiguous"] is False


def test_orbit_pass_fit_excludes_van_and_requires_the_exact_set():
    van_only = [s for s in _triple(van=0.75) if s["family"] != "FA1"]
    assert P.orbit_pass_fit(van_only) == {}, "VAN must never substitute for FA1 (B6)"
    two = [s for s in _triple() if s["family"] != "C8"]
    assert P.orbit_pass_fit(two) == {}
    with_oom = two + [_summary("C8_32x2", 2, 32, None, status="OOM")]
    assert P.orbit_pass_fit(with_oom) == {}
    extra = _triple(van=0.75) + [_summary("CKPT4_32x2", 2, 32, 0.4)]
    assert set(P.orbit_pass_fit(extra)) == {"32x2"}


def test_orbit_pass_fit_flags_implausible_output_as_ambiguous():
    faster_with_more_passes = [_summary("FA1_32x2", 2, 32, 0.10),
                               _summary("C4L_32x2", 2, 32, 0.20),
                               _summary("C8_32x2", 2, 32, 0.40)]
    fit = P.orbit_pass_fit(faster_with_more_passes)
    assert fit["32x2"]["ambiguous"] is True
    assert P.attribution_ok(fit) is False
    assert P.attribution_ok(P.orbit_pass_fit(_triple())) is True
    assert P.attribution_ok({}) is False


def test_vanilla_contrast_is_reported_separately():
    contrast = P.vanilla_contrast(_triple(van=0.60))
    # VAN step 0.60 s vs FA1 0.75 s: the fa dispatch + cylindrical pose path costs 0.15 s
    assert contrast["32x2"]["delta_s"] == pytest.approx(0.15)
    assert contrast["32x2"]["van_s_per_step"] == pytest.approx(0.60)
    assert contrast["32x2"]["fa1_s_per_step"] == pytest.approx(0.75)
    assert P.vanilla_contrast(_triple()) == {}      # no VAN -> nothing to contrast


def test_marginal_contrast_c8_minus_c4l():
    contrast = P.marginal_contrast(_triple())
    assert contrast["32x2"]["delta_s"] == pytest.approx(1.0)
    assert contrast["32x2"]["s_per_pass"] == pytest.approx(0.25)


def test_worker_contrast_needs_both_halves():
    both = [_summary("C4L_32x2", 2, 32, 0.20, workers=6),
            _summary("C4L_32x2", 2, 32, 0.10, workers=0)]
    wc = P.worker_contrast(both)
    assert wc["C4L_32x2"]["steps_s_w6"] == pytest.approx(0.20)
    assert wc["C4L_32x2"]["steps_s_w0"] == pytest.approx(0.10)
    assert wc["C4L_32x2"]["speedup_w6_over_w0"] == pytest.approx(2.0)
    assert P.worker_contrast([both[0]]) == {}


def test_ddp_scaling_and_grad_ckpt_cost():
    summaries = [_summary("C4L_32x2", 2, 32, 0.10), _summary("C4L_16x4", 4, 16, 0.20),
                 _summary("C4L_8x8", 8, 8, 0.20)]
    eff = {e["ngpu"]: e["efficiency"] for e in P.ddp_scaling(summaries)["C4L"]}
    assert (eff[2], eff[4], eff[8]) == (pytest.approx(1.0), pytest.approx(1.0), pytest.approx(0.5))
    gc = P.grad_ckpt_cost([_summary("C4L_32x2", 2, 32, 0.20, peak=20000),
                           _summary("CKPT4_32x2", 2, 32, 0.10, peak=30000)])
    assert gc["no_ckpt_speedup"] == pytest.approx(2.0)
    assert gc["delta_s_per_step"] == pytest.approx(5.0)
    assert gc["delta_peak_mib"] == -10000


# --------------------------------------------------------------------------- #
# 7. markdown
# --------------------------------------------------------------------------- #
def test_render_markdown_deterministic_and_labels_residual():
    summaries = _triple(van=0.60)
    md = P.render_markdown(summaries, mode="matrix", complete=True)
    assert md == P.render_markdown(list(reversed(summaries)), mode="matrix", complete=True)
    assert md.index("VAN_32x2") < md.index("FA1_32x2") < md.index("C4L_32x2")
    assert "unattributed" in md.lower()
    assert "residual step cost" not in md.lower()
    assert md.count("\n| ") >= 3


def test_render_markdown_withholds_derived_when_incomplete():
    summaries = _triple() + [_summary("C4L_8x8", 8, 8, None, status="PENDING")]
    md = P.render_markdown(summaries, mode="matrix", complete=False)
    assert "PENDING" in md and "WITHHELD" in md.upper()
    assert "s per orbit pass" not in md


# --------------------------------------------------------------------------- #
# 8. scanning and the mode-aware exit contract (B1)
# --------------------------------------------------------------------------- #
def test_scan_dir_treats_duplicate_result_lines_as_malformed(tmp_path):
    (tmp_path / "slurm_p0_p0-C4L_32x2_1001.out").write_text("boot\n" + _line(jobid=1001) + "\n")
    (tmp_path / "slurm_p0_p0-C8_32x2_1002.out").write_text(
        _line(cell="C8_32x2", jobid=1002) + "\n" + _line(cell="C8_32x2", jobid=1002) + "\n")
    (tmp_path / "slurm_p0_p0-VAN_32x2_1003.out").write_text("died early\n")
    rows, malformed = P.scan_dir(str(tmp_path))
    assert [r["cell"] for r in rows] == ["C4L_32x2"]
    assert any("duplicate" in msg.lower() for _, msg in malformed)


def _land_cell(tmp_path, cell, jobid, ngpu, mb, steps_s, workers=6):
    """Write one cell's slurm log plus a poller CSV whose ticks straddle that
    cell's own measured window (the collector requires in-window evidence)."""
    uuids = [f"GPU-{cell}-{i}" for i in range(ngpu)]
    name = f"p0_{cell}_w{workers}_vram.csv"
    t10, t30 = 1000.0, 1000.0 + 20.0 / steps_s
    ticks = ((0, t10 - 5.0), (1, t10 + 0.1), (2, (t10 + t30) / 2), (3, t30), (4, t30 + 5.0))
    digest = _write_csv(tmp_path, name, uuids, ticks=ticks)
    (tmp_path / f"slurm_p0_p0-{cell}_{jobid}.out").write_text(
        _line(cell=cell, jobid=jobid, ngpu=ngpu, mb=mb, workers=workers, t10=t10,
              t30=t30, vram_csv=name, pollcsv_sha=digest) + "\n")


def test_main_matrix_requires_the_fit_and_reports_run_specific_path(tmp_path):
    cells, jid = [], 2000
    for fam, n in (("FA1", 1), ("C4L", 4), ("C8", 8)):
        jid += 1
        cells.append((f"{fam}_32x2", jid))
        _land_cell(tmp_path, f"{fam}_32x2", jid, 2, 32, 1.0 / (0.5 + 0.25 * n))
    man_path = tmp_path / "p0_manifest_x.txt"
    man_path.write_text(_manifest_text(cells=tuple(cells), mode="matrix"))

    rc = P.main(["--manifest", str(man_path), "--dir", str(tmp_path)])
    assert rc == 0
    report = tmp_path / f"p0_report_{RUNID}.md"      # run-specific, no clobbering
    assert report.exists()
    body = report.read_text()
    assert "s per orbit pass" in body and "unattributed" in body.lower()


def test_main_spot_mode_succeeds_without_a_fit(tmp_path):
    _land_cell(tmp_path, "C16_16x4", 3001, 4, 16, 0.05)
    _land_cell(tmp_path, "C32_16x4", 3002, 4, 16, 0.02)
    man_path = tmp_path / "p0_manifest_spot.txt"
    man_path.write_text(_manifest_text(cells=(("C16_16x4", 3001), ("C32_16x4", 3002)),
                                       mode="spot"))
    assert P.main(["--manifest", str(man_path), "--dir", str(tmp_path)]) == 0


def test_main_workers_mode_pairs_in_one_manifest(tmp_path):
    _land_cell(tmp_path, "C4L_32x2", 4001, 2, 32, 0.20, workers=6)
    _land_cell(tmp_path, "C4L_32x2", 4002, 2, 32, 0.10, workers=0)
    man_path = tmp_path / "p0_manifest_w.txt"
    man_path.write_text(_manifest_text(cells=(("C4L_32x2", 4001, 6), ("C4L_32x2", 4002, 0)),
                                       mode="workers"))
    assert P.main(["--manifest", str(man_path), "--dir", str(tmp_path)]) == 0
    body = (tmp_path / f"p0_report_{RUNID}.md").read_text()
    assert "worker" in body.lower() and "2.000" in body      # the 0-vs-6 speedup


def test_main_requires_manifest_and_fails_on_a_pending_cell(tmp_path):
    _land_cell(tmp_path, "C4L_32x2", 1001, 2, 32, 0.2)
    man_path = tmp_path / "p0_manifest_p.txt"
    man_path.write_text(_manifest_text(cells=(("C4L_32x2", 1001), ("C8_32x2", 1002))))
    out = tmp_path / "custom_report.md"
    rc = P.main(["--manifest", str(man_path), "--dir", str(tmp_path), "--out", str(out)])
    assert rc != 0
    body = out.read_text()
    assert "C8_32x2" in body and "PENDING" in body and "WITHHELD" in body.upper()
    with pytest.raises(SystemExit):
        P.main(["--dir", str(tmp_path)])
