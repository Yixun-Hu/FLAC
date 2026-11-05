"""exp_35 round 6a (i): the training guard ``exp35_training_guard.py`` (plan_orbitrir_yaw_evidence.md r2 §7 "Guard",
r3 §12 B2, r4 §13 NIT 1) -- a READ-ONLY parser of Yixun's s43 training log, tested on synthetic PL tqdm logs shaped
like the real one (``\\r``-separated bar lines, each step printed twice, the epoch-end validation bar, the re-print of
the last step after it, the elapsed field restarting with each epoch).

Thresholds exact and strict: any completed 500-step window > 9.712 s/step, any completed 100-step window > 9.950, the
mean since our first spawn > 8.521 -- judged only over windows that end after the baseline step. Integrity: inode,
size, the content anchor (the sha256 of the last 4 KiB parsed), steps / epochs / totals monotone.

Written by the exp_35 Coder (Claude Opus 5.5, max effort) per plan_orbitrir_yaw_evidence.md r2/r3/r4.
"""
import importlib.util
import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
GUARD = os.path.join(REPO, "worklog", "worklog_yixun", "exp_35_orbitrir_yaw_evidence_claude", "exp35_training_guard.py")


def _load():
    spec = importlib.util.spec_from_file_location("exp35_training_guard", GUARD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _load()
T = 1200                                   # the fixture's bar total (the real one is 4550)


def _fmt(s):
    s = int(round(s))                       # integer seconds, exact at the window boundaries
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


def bar(epoch, n, elapsed, total=T, loss="0.512"):
    pct = int(100 * n / total)
    return (f"Epoch {epoch}: {pct:3d}%|{'#' * (10 * n // total):<10}| {n}/{total} [{_fmt(elapsed)}<{_fmt(max(0, 9999 - elapsed))}, "
            f" 0.24it/s, v_num=0, train/loss={loss}, train/lr=4.94e-5]")


def epoch_lines(epoch, rates, total=T, start_n=0):
    """Every step of one epoch printed twice (a postfix refresh), elapsed = cumulative integer seconds."""
    out, t = [bar(epoch, start_n, 0, total)], 0.0
    for n in range(start_n + 1, total + 1):
        t += rates(n) if callable(rates) else rates
        out += [bar(epoch, n, t, total), bar(epoch, n, t, total, loss="0.498")]
    return out, t


def validation(epoch, after, total=T):
    v = [f"Validation DataLoader 0: {i * 10:3d}%|{'#' * i:<10}| {i * 10}/98 [00:{i * 2:02d}<02:00,  0.70it/s]\x1b[A"
         for i in range(10)]
    return ["", "Validation: |          | 0/? [00:00<?, ?it/s]\x1b[A"] + v + [bar(epoch, total, after + 160, total),
                                                                         bar(epoch, 0, 0, total)]   # PL's reset, old label


def write_log(path, epochs, sep="\r", tail=None):
    """``epochs``: one ``rates`` per epoch (a float or a callable of the step) -- validation between epochs."""
    lines = []
    for e, rates in enumerate(epochs):
        ls, t = epoch_lines(e, rates)
        lines += ls + (validation(e, t) if e < len(epochs) - 1 else [])
    text = sep.join(lines) + sep + (tail or "")
    with open(path, "w") as f:
        f.write(text)
    return text


def obs(path, **kw):
    return G.observe(str(path), **kw)


# --------------------------------------------------------------------------- #
# the registered thresholds and what is judged
# --------------------------------------------------------------------------- #
def test_the_registered_thresholds_are_exact():
    assert (G.W500, G.W100, G.MEAN, G.ANCHOR_BYTES) == (9.712, 9.950, 8.521, 4096)
    assert (G.EXIT_OK, G.EXIT_STOP, G.EXIT_HARD, G.EXIT_REFUSED) == (0, 1, 2, 3)


def test_a_healthy_log_is_ok_and_reports_the_step(tmp_path):
    p = tmp_path / "train.log"
    write_log(p, [4.23, 4.23, 4.23])
    r = obs(p, baseline=T + 40)
    assert (r["status"], r["exit"], r["reasons"]) == ("OK", 0, [])
    assert r["step"] == {"epoch": 2, "n": T, "total": T, "global": 3 * T}
    assert r["windows"]["500"]["judged"] == 4 and r["windows"]["100"]["judged"] == 24   # windows ending after T + 40
    assert abs(r["mean_since_baseline"]["s_per_step"] - 4.23) < 0.01 and r["log"]["size"] == os.path.getsize(p)


def test_before_the_baseline_nothing_is_judged(tmp_path):
    p = tmp_path / "train.log"
    write_log(p, [12.0, 4.23])                                # epoch 0 ran slow -- before our first spawn
    assert obs(p)["status"] == "OK" and obs(p)["windows"]["500"]["judged"] == 0
    assert obs(p, baseline=T + 1)["status"] == "OK"
    assert obs(p, baseline=0)["status"] == "STOP"


@pytest.mark.parametrize("seconds, status", [(4856, "OK"), (4857, "STOP")])   # 500 x 9.712 = 4856 s exactly
def test_the_500_step_window_threshold_is_strict(tmp_path, seconds, status):
    p = tmp_path / "train.log"
    write_log(p, [4.23, lambda n: 4.23 if not 500 < n <= 1000 else (seconds / 500)])
    r = obs(p, baseline=T)
    assert r["status"] == status, r["reasons"]
    if status == "STOP":
        assert any("500-step window" in x and "epoch 1 steps 500-1000" in x for x in r["reasons"])


def test_a_100_step_window_alone_stops(tmp_path):
    p = tmp_path / "train.log"
    write_log(p, [4.23, lambda n: 10.0 if 300 < n <= 400 else 4.23])          # 500-window: (100 x 10 + 400 x 4.23)/500
    r = obs(p, baseline=T)
    assert r["status"] == "STOP" and r["exit"] == 1
    assert any("100-step window epoch 1 steps 300-400" in x for x in r["reasons"]) and not any(
        "500-step" in x for x in r["reasons"])


def test_the_mean_since_the_baseline_stops_while_every_window_is_inside(tmp_path):
    p = tmp_path / "train.log"
    write_log(p, [4.23, 8.6, 8.6])
    r = obs(p, baseline=T)
    assert r["status"] == "STOP" and any("mean" in x for x in r["reasons"])
    assert not any("window" in x for x in r["reasons"])


def test_the_validation_between_epochs_is_in_no_window(tmp_path):
    """The 160 s validation pause and the re-print of the last step after it (a larger elapsed) are excluded."""
    p = tmp_path / "train.log"
    write_log(p, [9.5, 9.5, 9.5])                              # every window just inside 9.712 / 9.950
    r = obs(p, baseline=0)
    assert r["status"] == "STOP" and r["reasons"] == [r for r in r["reasons"] if "mean" in r]   # 9.5 > 8.521 only
    assert r["windows"]["500"]["worst"]["s_per_step"] == pytest.approx(9.5, abs=0.01)


def test_a_partial_last_line_is_not_parsed(tmp_path):
    p = tmp_path / "train.log"
    write_log(p, [4.23, 4.23], tail="Epoch 2:   0%|          | 0/1200 [00:0")
    r = obs(p, baseline=T)
    assert r["status"] == "OK" and r["step"]["global"] == 2 * T
    assert r["anchor"]["offset"] == os.path.getsize(p) - len("Epoch 2:   0%|          | 0/1200 [00:0")


def test_newline_separated_logs_parse_the_same(tmp_path):
    a, b = tmp_path / "a.log", tmp_path / "b.log"
    write_log(a, [4.23, 5.0])
    write_log(b, [4.23, 5.0], sep="\n")
    ra, rb = obs(a, baseline=T), obs(b, baseline=T)
    assert ra["windows"] == rb["windows"] and ra["step"] == rb["step"] and ra["mean_since_baseline"] == rb[
        "mean_since_baseline"]


# --------------------------------------------------------------------------- #
# integrity: rotation, truncation, copy-truncate-and-regrow, restarts
# --------------------------------------------------------------------------- #
def _anchored(tmp_path, epochs=(4.23, 4.23)):
    p = tmp_path / "train.log"
    write_log(p, list(epochs))
    return p, obs(p, baseline=T)["anchor"]


def test_an_unchanged_or_growing_log_keeps_its_anchor(tmp_path):
    p, anchor = _anchored(tmp_path)
    assert obs(p, baseline=T, anchor=anchor)["status"] == "OK"
    with open(p, "a") as f:
        f.write("\r".join(epoch_lines(2, 4.23)[0][:50]) + "\r")
    r = obs(p, baseline=T, anchor=anchor)
    assert r["status"] == "OK" and r["anchor"]["offset"] > anchor["offset"]


def test_a_rotated_log_is_a_hard_stop(tmp_path):
    p, anchor = _anchored(tmp_path)
    os.rename(p, tmp_path / "train.log.1")
    write_log(p, [4.23, 4.23])                                # same content, another inode
    r = obs(p, baseline=T, anchor=anchor)
    assert (r["status"], r["exit"]) == ("HARD_STOP", 2) and any("inode" in x for x in r["reasons"])


def test_a_shrunk_log_is_a_hard_stop(tmp_path):
    p, anchor = _anchored(tmp_path)
    with open(p, "r+") as f:
        f.truncate(anchor["size"] - 10)
    r = obs(p, baseline=T, anchor=anchor)
    assert r["status"] == "HARD_STOP" and any("shrank" in x for x in r["reasons"])


def test_a_copy_truncate_and_regrow_is_a_hard_stop(tmp_path):
    """§13 NIT 1: same inode, the size grew again past the anchor -- but the bytes before the anchor changed."""
    p, anchor = _anchored(tmp_path)
    text = open(p).read()
    with open(p, "r+") as f:                                  # rewrite in place: same inode, longer, other bytes
        f.seek(0)
        f.write(text.replace("loss=0.498", "loss=0.497") + "\r".join(epoch_lines(2, 4.23)[0][:20]) + "\r")
    r = obs(p, baseline=T, anchor=anchor)
    assert os.stat(p).st_ino == anchor["inode"] and os.path.getsize(p) > anchor["size"]
    assert r["status"] == "HARD_STOP" and any("anchor" in x for x in r["reasons"])


@pytest.mark.parametrize("tail, needle", [
    (bar(1, 300, 10), "backwards"),                          # a step behind the last one, within the epoch
    (bar(1, T, 5000) + "\r" + bar(1, 1150, 4000), "backwards"),
    (bar(0, 10, 40), "epoch 0 after epoch 1"),                # an epoch number going back: a restart
    (bar(2, 5, 20, total=900), "total"),                      # another bar total: another run
])
def test_a_restart_or_another_run_in_the_same_log_is_a_hard_stop(tmp_path, tail, needle):
    p = tmp_path / "train.log"
    write_log(p, [4.23, 4.23], tail=tail + "\r")
    r = obs(p, baseline=T)
    assert r["status"] == "HARD_STOP" and any(needle in x for x in r["reasons"]), r["reasons"]


def test_a_log_without_a_training_bar_is_a_hard_stop(tmp_path):
    p = tmp_path / "train.log"
    p.write_text("Validation: |          | 0/? [00:00<?, ?it/s]\r")
    assert obs(p)["status"] == "HARD_STOP"


def test_an_unreadable_log_is_refused(tmp_path):
    r = obs(tmp_path / "absent.log")
    assert (r["status"], r["exit"]) == ("REFUSED", 3)


# --------------------------------------------------------------------------- #
# the CLI: a JSON report, the documented exit codes, read-only
# --------------------------------------------------------------------------- #
def test_the_cli_reports_json_with_the_exit_code_and_writes_nothing(tmp_path, capsys):
    p = tmp_path / "train.log"
    write_log(p, [4.23, 8.6, 8.6])
    anchor = tmp_path / "anchor.json"
    anchor.write_text(json.dumps(obs(p)["anchor"]))
    before = {q: (os.stat(q).st_mtime_ns, os.path.getsize(q)) for q in tmp_path.iterdir()}
    assert G.main(["--log", str(p), "--baseline-step", str(T), "--anchor", str(anchor)]) == 1
    r = json.loads(capsys.readouterr().out)
    assert r["status"] == "STOP" and r["thresholds"] == {"w500": 9.712, "w100": 9.950, "mean": 8.521}
    assert {q: (os.stat(q).st_mtime_ns, os.path.getsize(q)) for q in tmp_path.iterdir()} == before
    assert G.main(["--log", str(tmp_path / "absent.log")]) == 3
    assert G.main(["--log", str(p), "--anchor", str(tmp_path / "absent.json")]) == 3


def test_the_guard_never_writes_or_signals():
    src = open(GUARD).read()
    for banned in ("os.kill", "signal.", ".terminate(", ".kill(", "subprocess", "open(path, \"w", "os.remove",
                   "os.unlink", "os.rename", "shutil"):
        assert banned not in src, banned


def test_the_bar_reset_under_the_old_label_after_a_completed_epoch_is_accepted(tmp_path):
    """The real s43 log: after the validation PL resets the bar BEFORE the next epoch's label ("Epoch 0: 0/4550
    [00:00<") -- accepted after the epoch's last step; the same line in the middle of an epoch is a restart."""
    p = tmp_path / "train.log"
    text = write_log(p, [4.23, 4.23])
    assert "\r" + bar(0, 0, 0) + "\r" + bar(1, 0, 0) + "\r" in text and obs(p, baseline=T)["status"] == "OK"
    write_log(p, [4.23], tail=bar(0, 0, 0) + "\r")            # 0/T again in epoch 0 -- after its last step: fine
    assert obs(p)["status"] == "OK"
    lines, _ = epoch_lines(0, 4.23)
    p.write_text("\r".join(lines[:401] + [bar(0, 0, 0)] + lines[401:]) + "\r")   # mid-epoch: a restart
    r = obs(p)
    assert r["status"] == "HARD_STOP" and any("backwards" in x for x in r["reasons"])
