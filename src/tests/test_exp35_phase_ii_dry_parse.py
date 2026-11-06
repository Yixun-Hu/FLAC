"""exp_35 phase (ii) kit, part 4a (plan_orbitrir_yaw_evidence.md §15 item 7; Codex plan §15 r1 S4, S6 "exact
dry-run parsing: 68 IDs, 68 argv records, no E2-FAIL / SKIP / refusal / quarantine"): the exact runner command
(the 68 cells, the frozen lanes, the CANON manifest, the launch environment, nothing shared, no --no-dump-blocks)
and the line-by-line parser of the runner's DRY_RUN transcript -- anchored on the REAL H35 transcript of phase (i)
and exercised on the sandbox stub runner's phase-(ii) transcript with every fault it can inject.

Written by the exp_35 phase-(ii) kit Coder (Claude Opus 5.5, max effort).
"""
import json
import os
import subprocess

import pytest

import exp35_phase_ii_sandbox as SB
from exp35_phase_ii_sandbox import X25, X35, make_box

K = SB.load_kit()
REAL = os.path.join(SB.REPO, X35, "dry_run_artifacts_5b065801")


def test_the_exact_runner_command():
    cfg = K.registered_config()
    argv, env, cwd = K.runner_command(cfg, (0,), dry=True)
    assert argv == ["bash", f"{X25}/depth_canon_eval_run.sh", "--blocks", "OFF-P1,OFF-BF,OFF-CANON,CANON-45", "--lanes",
                    "0", "--canon-ckpt-manifest", f"{X25}/depth_canon_2026-10-26_07-44-41_exp25_CANON_checkpoints.sha256"]
    assert env == dict(K.LAUNCH_ENV, EXPECT_HEAD=SB.H35_REAL, DRY_RUN="1") and cwd == cfg.run
    argv2, env2, _ = K.runner_command(cfg, (0, 1), dry=False)
    assert argv2[5] == "0,1" and env2 == dict(K.LAUNCH_ENV, EXPECT_HEAD=SB.H35_REAL)
    assert not set(K.SHARING) & (set(env) | set(env2)) and "--no-dump-blocks" not in argv + argv2
    assert "GPU_MAX_POLLS" not in env2 and "PYTHONPATH" not in env2 and "E2-FAIL" not in " ".join(argv2)


def test_the_pinned_preflight_and_warning_lines_are_the_real_ones():
    raw = open(os.path.join(REAL, "depth_canon_2026-11-05_19-24-07_no_flash_preflight.log"), "rb").read()
    assert raw == K.PREFLIGHT.encode() == SB.PREFLIGHT.encode()
    text = open(os.path.join(REAL, "depth_canon_2026-11-05_19-24-07_eval_run.log")).read()
    assert text.splitlines().count(K.BENIGN) == 2 and K.BENIGN == SB.BENIGN


def _phase_i(cfg):
    m = json.loads(SB.real_cells())
    rows = [r for r in m["cells"] if r["block"] in ("CTL", "BF-pin", "BF-dup")]
    return m, rows, cfg._replace(h35=SB.H35_REAL, digest=SB.DIGEST_REAL, run="/home/yixunhu/codespace/exp-35-orbitrir-yaw")


def test_the_parser_reads_the_real_h35_runner_transcript_exactly():
    m, rows, cfg = _phase_i(K.registered_config())
    text = open(os.path.join(REAL, "depth_canon_2026-11-05_19-24-07_eval_run.log")).read()
    why, info = K.parse_dry_run(text, cfg=cfg, m=m, rows=rows, lanes=(1,))
    assert why == [] and info["ts"] == "2026-11-05_19-24-07" and info["ids"] == [r["cell"] for r in rows]
    assert len(info["argv"]) == 32 and info["benign_lines"] == 2 and info["lines"] == 112
    plan = json.load(open(os.path.join(REAL, "depth_canon_2026-11-05_19-24-07_eval_plan.json")))
    assert info["argv"] == [e["argv"] for e in plan["cells"]]


@pytest.mark.parametrize("edit, reason", [
    (lambda t: t.replace("lanes 1\n", "lanes 0,1\n"), "not the expected"),
    (lambda t: "\n".join(t.splitlines()[:-1]) + "\n", "ends after"),
    (lambda t: t + "trailing\n", "follows the summary"),
    (lambda t: t.replace("evaluator tree digest da82", "evaluator tree digest 0a82"), "not the expected"),
    (lambda t: t.replace("dump root OK under DRY_RUN: /media/diskstation/yixunhu/FLAC/exp25_pred_waveforms (exists)",
                         "dump root OK under DRY_RUN: /media/diskstation/yixunhu/FLAC/exp25_pred_waveforms (would be "
                         "created)"), "not the expected"),
    (lambda t: t.replace("eval_plan.json\n", "eval_plan.json\n" + K.BENIGN + "\n", 1), None),     # a third warning
    (lambda t: t.replace("| BF-pin | BF | would run\n", "| BF-pin | BF | would run\n" + K.BENIGN + "\n", 1),
     "not the expected"),                                                          # a warning among the cells
    (lambda t: t.replace("2026-11-05_19-24-07_eval_plan", "2026-11-05_19-24-08_eval_plan"), "timestamps")])
def test_every_deviation_of_the_real_transcript_is_named(edit, reason):
    m, rows, cfg = _phase_i(K.registered_config())
    text = open(os.path.join(REAL, "depth_canon_2026-11-05_19-24-07_eval_run.log")).read()
    why, _ = K.parse_dry_run(edit(text), cfg=cfg, m=m, rows=rows, lanes=(1,))
    assert (why == []) if reason is None else any(reason in x for x in why), why


# --------------------------------------------------------------------------- #
# the phase-(ii) transcript of the sandbox stub runner, and every fault it injects
# --------------------------------------------------------------------------- #
def _stub_dry(tmp_path, **dry):
    stub = SB.install_stub(tmp_path)
    b = make_box(tmp_path, runner=stub)
    SB.stub_control(tmp_path, dry=dry)
    argv, env, cwd = K.runner_command(b.cfg, (0,), dry=True)
    p = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, timeout=120)
    _, m, rows = K.selection(SB.real_cells())
    return b, m, rows, p


def test_the_stub_transcript_of_the_68_cells_parses_exactly(tmp_path):
    b, m, rows, p = _stub_dry(tmp_path)
    assert p.returncode == 0, p.stdout.decode()[-2000:]
    why, info = K.parse_dry_run(p.stdout.decode(), cfg=b.cfg, m=m, rows=rows, lanes=(0,))
    assert why == [] and info["ids"] == [r["cell"] for r in rows] and len(info["ids"]) == 68 and len(info["argv"]) == 68
    assert info["argv"] == [e["argv"] for e in K.expected_plan(m, rows, run=b.run, digest=b.digest)["cells"]]
    assert K.names_sha256(info["ids"]) == K.NAMES_SHA256 and info["benign_lines"] == 2
    assert all(a[-2:] == ["--dump-scored-waveforms", K.DUMP_ROOT] for a in info["argv"])


@pytest.mark.parametrize("fault, reason", [
    ({"drop": "exp35_BF_OFF_S40000_K8_s44_th225"}, "not the expected"),
    ({"skip": "exp35_P1_OFF_S40000_K1_s42_th135"}, "a SKIP line"),
    ({"quarantine": "exp35_CANON_OFF_S40000_K1_s45_th135"}, "a quarantine line"),
    ({"e2fail": True}, "an E2-FAIL cell"),
    ({"argv_tamper": "exp35_CANON_OFF_S40000_K8_s43_th45"}, "argv"),
    ({"extra": "an unexplained line"}, "not the expected"),
    ({"warning_late": True}, "not the expected"),
    ({"abort": True}, "a refusal"),
    ({"digest": "0" * 64}, "not the expected"),
    ({"lanes": "1"}, "not the expected"),
    ({"summary_n": 67}, "not the expected"),
    ({"warning": False}, None)])
def test_every_stub_fault_is_refused_by_the_parser(tmp_path, fault, reason):
    b, m, rows, p = _stub_dry(tmp_path, **fault)
    why, info = K.parse_dry_run(p.stdout.decode(), cfg=b.cfg, m=m, rows=rows, lanes=(0,))
    assert (why == []) if reason is None else (why and any(reason in x for x in why)), why


def test_an_e2_fail_cell_id_alone_is_named(tmp_path):
    m = json.loads(SB.real_cells())
    e2 = next(c["cell"] for c in m["cells"] if c["block"] == "E2-FAIL")
    why, _ = K.parse_dry_run(f"DRYRUN {e2} argv: []\n", cfg=K.registered_config(), m=m, rows=[], lanes=(0,))
    assert any("an E2-FAIL cell" in x for x in why)
