"""exp_25 depth_canon round 3d: the unattended SMOKE -> FULL watcher
``depth_canon_autolaunch.sh`` (Yixun pre-approved the training GO, 2026-10-24).

Every case runs the REAL script as a subprocess from a throwaway ``git init`` repo under
/tmp (one commit, so its HEAD checks are real), in harness mode: a stub ``nvidia-smi`` on
PATH that plays a per-poll scenario file, and a fake launcher under /tmp that records each
call's MODE / EXPECT_HEAD and exits per its own scenario. Nothing touches a GPU, the NAS or
the real launcher.

Written by the exp_25 Coder seat (Claude Opus 5.5, max effort), round 3d.
"""
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPREL = os.path.join("worklog", "worklog_yixun", "exp_25_depth_canon_claude")
WATCHER = os.path.join(REPO, EXPREL, "depth_canon_autolaunch.sh")
FLAC_BIN = "/home/yixunhu/miniconda3/envs/flac/bin"
UUID0, UUID1 = "GPU-test-0", "GPU-test-1"

STUB_NVSMI = r'''#!/usr/bin/env bash
# stub: one scenario line per POLL (the index,uuid query opens a poll); the last line repeats.
# rA A3 scenarios exit NONZERO after plausible output (uuidrc / memrc / appsrc), print two memory
# rows (memtwo), or name a process on a uuid outside the map (unknown); GPU 2 is a known other card.
# rC-F2: ONLY the nounits compute-app query is answered (bare MiB numbers; na / na_other = a
# non-numeric used_memory); any other compute-app query is logged to $H/nvsmi_unexpected and exits 9.
H="$DEPTH_CANON_HARNESS"; n="$(cat "$H/poll_n" 2>/dev/null || echo 0)"
case "$*" in *index,uuid*) n=$((n + 1)); echo "$n" > "$H/poll_n" ;; esac
total="$(wc -l < "$H/gpu_scenario")"; [ "$n" -le "$total" ] || n="$total"; [ "$n" -ge 1 ] || n=1
line="$(sed -n "${n}p" "$H/gpu_scenario")"
case "$*" in *index,uuid*) echo "$*" >> "$H/nvsmi_calls" ;; esac
case "$line" in
  fail) exit 1 ;;
  commit*) case "$*" in *index,uuid*) git -C "$AUTOLAUNCH_TEST_REPO" -c user.name=t -c user.email=t@l \
                                          commit -q --allow-empty -m "a commit after arming" ;; esac ;;
esac
case "$*" in
  *index,uuid*) printf '0, GPU-test-0\n1, GPU-test-1\n2, GPU-test-2\n'; [ "$line" = uuidrc ] && exit 4; true ;;
  "--query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits")
                        [ "$line" = busy ] && echo "GPU-test-0, 4242, python, 32470"
                        [ "$line" = busy1 ] && echo "GPU-test-1, 4343, python, 16000"
                        [ "$line" = other ] && echo "GPU-test-2, 4444, python, 100"
                        [ "$line" = unknown ] && echo "GPU-test-9, 4545, python, 100"
                        [ "$line" = garbage ] && echo "garbage"
                        [ "$line" = na ] && echo "GPU-test-0, 4242, python, [N/A]"
                        [ "$line" = na_other ] && echo "GPU-test-2, 4444, python, [N/A]"
                        [ "$line" = appsrc ] && exit 5
                        true ;;
  *query-compute-apps*) echo "$*" >> "$H/nvsmi_unexpected"; echo "stub: unexpected compute-app query"; exit 9 ;;
  *memory.free*) [ "$line" = lowmem ] && echo 1000 || echo 40000
                 [ "$line" = memtwo ] && echo 40000
                 [ "$line" = memrc ] && exit 3
                 true ;;
esac
'''
FAKE_LAUNCHER = r'''#!/usr/bin/env bash
# fake launcher: records MODE / EXPECT_HEAD; behaviour = the next "MODE rc text" scenario line
H="$DEPTH_CANON_HARNESS"
echo "MODE=${MODE} EXPECT_HEAD=${EXPECT_HEAD} DRY_RUN=${DRY_RUN:-unset} MB=${MB:-unset} ACC=${ACC:-unset}" >> "$H/launch_calls"
env | grep -E '^(GO_TOKEN|AUTOLAUNCH_(LAUNCHER|POLL_S|STABLE_POLLS|MAX_POLLS))=' >> "$H/launch_env" || true
n="$(( $(cat "$H/launch_n" 2>/dev/null || echo 0) + 1 ))"; echo "$n" > "$H/launch_n"
total="$(wc -l < "$H/launch_scenario")"; [ "$n" -le "$total" ] || n="$total"
read -r mode rc text <<<"$(sed -n "${n}p" "$H/launch_scenario")"
[ "$mode" = "$MODE" ] || { echo "fake launcher: expected MODE=$mode, got $MODE"; exit 99; }
sleep "${FAKE_SLEEP:-0}"
echo "$text"
exit "$rc"
'''


def _git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@l", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture()
def rig(tmp_path):
    """A throwaway repo (watcher at its pinned path, a pins file, a placeholder launcher) +
    a harness dir with the marker, the stubs and the scenario files."""
    repo, h = tmp_path / "repo", tmp_path / "h"
    exp = repo / EXPREL
    exp.mkdir(parents=True)
    (h / "bin").mkdir(parents=True)
    (h / ".depth_canon_harness").write_text("")
    shutil.copy(WATCHER, exp / "depth_canon_autolaunch.sh")
    (exp / "depth_canon_pins.sha256").write_text("0" * 64 + "  train.py\n")
    (exp / "depth_canon_launch.sh").write_text("#!/usr/bin/env bash\necho placeholder; exit 9\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    (h / "bin" / "nvidia-smi").write_text(STUB_NVSMI)
    (h / "fake_launch.sh").write_text(FAKE_LAUNCHER)
    for f in (h / "bin" / "nvidia-smi", h / "fake_launch.sh"):
        os.chmod(f, 0o755)
    r = {"repo": repo, "exp": exp, "h": h, "head": _git(repo, "rev-parse", "HEAD")}
    gpu(r, ["free"])
    launches(r, ["SMOKE 0 smoke ok", "FULL 0 full ok"])
    return r


def gpu(r, lines):
    (r["h"] / "gpu_scenario").write_text("\n".join(lines) + "\n")


def launches(r, lines):
    (r["h"] / "launch_scenario").write_text("\n".join(lines) + "\n")


def pins_sha(r):
    return hashlib.sha256((r["exp"] / "depth_canon_pins.sha256").read_bytes()).hexdigest()


def token(r, head=None, manifest=None, by="planner", path=None):
    text = (f"head={head or r['head']}\nmanifest_sha256={manifest or pins_sha(r)}\n"
            f"issued=2026-10-24T02:00:00-04:00\nby={by}\n")
    (path or r["exp"] / "depth_canon_GO.token").write_text(text)


def run(r, env=None, max_polls=12, harness=True, timeout=120, path_prefix=None):
    e = {k: v for k, v in os.environ.items()
         if not k.startswith(("AUTOLAUNCH_", "DEPTH_CANON_", "GO_TOKEN", "MODE", "EXPECT_HEAD"))}
    e["PATH"] = f"{r['h'] / 'bin'}:{path_prefix or FLAC_BIN}:{os.environ['PATH']}"
    e["AUTOLAUNCH_TEST_REPO"] = str(r["repo"])
    if harness:
        e.update(DEPTH_CANON_HARNESS=str(r["h"]), AUTOLAUNCH_LAUNCHER=str(r["h"] / "fake_launch.sh"),
                 AUTOLAUNCH_POLL_S="0", AUTOLAUNCH_MAX_POLLS=str(max_polls))
    e.update(env or {})
    p = subprocess.run(["bash", str(r["exp"] / "depth_canon_autolaunch.sh")], env=e, capture_output=True,
                       text=True, timeout=timeout, cwd=str(r["h"]))
    return p.returncode, p.stdout + p.stderr


def state(r):
    return json.loads((r["exp"] / "depth_canon_autolaunch_state.json").read_text())


def calls(r):
    path = r["h"] / "launch_calls"
    return path.read_text().splitlines() if path.exists() else []


# --------------------------------------------------------------------------- #
# the GO token
# --------------------------------------------------------------------------- #
def test_a_no_token_waits_and_never_launches(rig):
    rc, out = run(rig, max_polls=3)
    st = state(rig)
    assert rc == 3 and "poll budget exhausted" in out and calls(rig) == []
    assert st["phase"] == "WAITING_TOKEN" and "token" in st["reason"] and st["kind"] == "depth_canon_autolaunch_state"


def test_b_a_token_for_another_head_is_refused(rig):
    token(rig, head="1" * 40)
    rc, out = run(rig, max_polls=3)
    assert rc == 3 and calls(rig) == [] and state(rig)["phase"] == "WAITING_TOKEN"
    assert "head" in state(rig)["reason"] and "1" * 40 in state(rig)["reason"]


def test_c_a_token_for_another_manifest_is_refused(rig):
    token(rig, manifest="2" * 64)
    rc, _ = run(rig, max_polls=3)
    assert rc == 3 and calls(rig) == [] and "manifest" in state(rig)["reason"]


@pytest.mark.parametrize("text", ["head=abc\n", "head=" + "a" * 40 + "\nby=planner\n", "garbage\n"])
def test_a_malformed_token_is_not_a_token(rig, text):
    (rig["exp"] / "depth_canon_GO.token").write_text(text)
    rc, _ = run(rig, max_polls=2)
    assert rc == 3 and calls(rig) == [] and state(rig)["phase"] == "WAITING_TOKEN"


# --------------------------------------------------------------------------- #
# the GPU condition
# --------------------------------------------------------------------------- #
def test_d_shared_policy_busy_gpus_are_clean_and_the_cotenant_is_logged(rig):
    """rC-F1 (plan v14, Yixun 2026-10-25 shared launch): a foreign compute process on GPU 0 does
    not make the poll dirty -- the memory floor decides; the co-tenant is logged ONCE, as one
    `co-tenants:` log line per change of the pid set, never through write_state."""
    token(rig)
    gpu(rig, ["busy"])
    rc, out = run(rig, max_polls=12)
    st = state(rig)
    assert rc == 0, out
    assert [c.split()[0] for c in calls(rig)] == ["MODE=SMOKE", "MODE=FULL"] and st["phase"] == "DONE"
    assert out.count("pid 4242") == 1 and "32470 MiB" in out                  # once per change
    line = next(ln for ln in out.splitlines() if "pid 4242" in ln)
    assert "co-tenants: " in line and "GPU 0" in line and "cwd " in line, line
    log = open(os.path.join(rig["repo"], st["log"])).read()
    assert log.count("pid 4242") == 1 and "4242" not in st["reason"]


def test_d2_a_changed_cotenant_set_is_logged_once_per_change(rig):
    """rC-F1: the `co-tenants:` line is printed when the set of pids changes between polls,
    not on every poll (and the changed set is still a clean poll)."""
    token(rig)
    gpu(rig, ["busy", "busy", "busy1"])
    rc, out = run(rig, max_polls=12)
    assert rc == 0, out
    assert out.count("pid 4242") == 1 and out.count("pid 4343") == 1
    assert len([ln for ln in out.splitlines() if "co-tenants: " in ln]) == 2


def test_e_a_dirty_poll_resets_the_counter(rig):
    token(rig)
    gpu(rig, ["free", "free", "lowmem", "free", "free"])                # rC-F1: lowmem is the dirty poll
    rc, _ = run(rig, max_polls=5)
    assert rc == 3 and calls(rig) == [] and state(rig)["clean_polls"] == 2


def test_f_three_clean_polls_fire_smoke_then_full(rig):
    token(rig)
    rc, out = run(rig)
    st = state(rig)
    assert rc == 0, out
    assert calls(rig) == [f"MODE=SMOKE EXPECT_HEAD={rig['head']} DRY_RUN=0 MB=32 ACC=1",
                          f"MODE=FULL EXPECT_HEAD={rig['head']} DRY_RUN=0 MB=32 ACC=1"]
    assert (st["phase"], st["smoke_rc"], st["full_rc"], st["token_head"]) == ("DONE", 0, 0, rig["head"])
    assert st["clean_polls"] >= 3 and st["watcher_pid"] > 0 and st["log"].endswith("_autolaunch.log")
    assert "smoke ok" in open(os.path.join(rig["repo"], st["log"])).read()     # launcher output tee'd
    assert len((rig["h"] / "nvsmi_calls").read_text().splitlines()) >= 3
    assert not (rig["h"] / "nvsmi_unexpected").exists()        # rC-F2: only the nounits query was asked


def test_another_card_and_low_memory(rig):
    token(rig)
    gpu(rig, ["other", "other", "other"])
    assert run(rig)[0] == 0                                    # a process on GPU 9 does not block
    for f in ("poll_n", "launch_n", "launch_calls"):
        (rig["h"] / f).unlink(missing_ok=True)
    gpu(rig, ["lowmem"])
    rc, out = run(rig, max_polls=3)
    assert rc == 3 and calls(rig) == [] and "1000 MiB" in out


def test_g_a_transient_gpu_refusal_retries_after_restabilizing(rig):
    token(rig)
    launches(rig, ["SMOKE 7 GPU GATE FAILED: GPU 1 free 11289 MiB < 12288 MiB (12 GiB) - abort",
                   "SMOKE 0 smoke ok", "FULL 0 full ok"])
    rc, out = run(rig, max_polls=20)
    assert rc == 0 and [c.split()[0] for c in calls(rig)] == ["MODE=SMOKE", "MODE=SMOKE", "MODE=FULL"]
    assert state(rig)["gpu_retries"] == 1
    polls = len((rig["h"] / "nvsmi_calls").read_text().splitlines())
    assert polls >= 6                                          # 3 clean polls before each SMOKE


def test_g2_five_retries_then_failed(rig):
    token(rig)
    launches(rig, ["SMOKE 7 GPU GATE FAILED: GPU 0 free 100 MiB < 12288 MiB (12 GiB) - abort"])
    rc, out = run(rig, max_polls=60)
    st = state(rig)
    assert rc == 1 and [c.split()[0] for c in calls(rig)] == ["MODE=SMOKE"] * 6
    assert (st["phase"], st["gpu_retries"]) == ("FAILED", 6) and "SMOKE" in st["reason"]


def test_h_any_other_smoke_failure_is_final(rig):
    token(rig)
    launches(rig, ["SMOKE 4 SMOKE VERDICT: FAIL - projected 130.0 h > MAX_PROJECTED_HOURS=120"])
    rc, _ = run(rig)
    st = state(rig)
    assert rc == 1 and calls(rig) == [f"MODE=SMOKE EXPECT_HEAD={rig['head']} DRY_RUN=0 MB=32 ACC=1"]
    assert (st["phase"], st["smoke_rc"], st["full_rc"]) == ("FAILED", 4, None) and "SMOKE" in st["reason"]


def test_i_a_full_failure_is_final(rig):
    token(rig)
    launches(rig, ["SMOKE 0 smoke ok", "FULL 6 ENDPOINT NOT REACHED"])
    rc, _ = run(rig)
    st = state(rig)
    assert rc == 1 and (st["phase"], st["smoke_rc"], st["full_rc"]) == ("FAILED", 0, 6)
    assert "FULL" in st["reason"] and "6" in st["reason"]


@pytest.mark.parametrize("scenario", ["fail", "garbage", "uuidrc", "memrc", "memtwo", "appsrc", "unknown"])
def test_j_an_unusable_nvidia_smi_is_dirty_and_never_fires(rig, scenario):
    token(rig)
    gpu(rig, [scenario])
    rc, _ = run(rig, max_polls=5)
    assert rc == 3 and calls(rig) == [] and state(rig)["phase"] == "WAITING_GPU"


@pytest.mark.parametrize("scenario, row", [("na", "GPU-test-0, 4242, python, [N/A]"),
                                           ("na_other", "GPU-test-2, 4444, python, [N/A]")])
def test_j2_a_non_numeric_used_memory_is_a_dirty_poll_that_waits(rig, scenario, row):
    """rC-F2 (Codex rC NIT, C4/C5): the launcher refuses a non-numeric used_memory on ANY row (rc 7);
    the watcher mirrors it -- a DIRTY poll whose reason names the row, the clean counter reset, no
    launcher call (no retry burnt), phase WAITING_GPU, never SMOKE, never FAILED."""
    token(rig)
    gpu(rig, ["free", "free", scenario, "free"])
    rc, out = run(rig, max_polls=4)
    st = state(rig)
    why = f"non-numeric used_memory in compute-app row '{row}' - waiting"
    assert rc == 3 and calls(rig) == [] and "SMOKE_RUNNING" not in out and "FAILED" not in out, out
    assert (st["phase"], st["clean_polls"], st["gpu_retries"]) == ("WAITING_GPU", 1, 0), st
    assert f"phase WAITING_GPU: GPUs not free: {why}" in out, out
    assert why in open(os.path.join(rig["repo"], st["log"])).read()


def test_k_a_commit_after_arming_is_refused(rig):
    token(rig)
    gpu(rig, ["free", "free", "commit-and-free", "free"])         # the commit lands on the firing poll
    rc, out = run(rig, max_polls=6)
    assert rc == 3 and calls(rig) == [], out                      # the pre-launch re-check caught it
    assert state(rig)["phase"] == "WAITING_TOKEN" and "head" in state(rig)["reason"]


# --------------------------------------------------------------------------- #
# preconditions (exit 2)
# --------------------------------------------------------------------------- #
def test_l_a_second_instance_is_refused(rig):
    token(rig)
    lock = open(rig["exp"] / ".depth_canon_autolaunch.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        rc, out = run(rig, max_polls=2)
    finally:
        lock.close()
    assert rc == 2 and "lock" in out and calls(rig) == []


@pytest.mark.parametrize("env, needle", [
    ({"AUTOLAUNCH_LAUNCHER": "/tmp/x.sh"}, "AUTOLAUNCH_LAUNCHER"),
    ({"AUTOLAUNCH_POLL_S": "0"}, "AUTOLAUNCH_POLL_S"),
    ({"GO_TOKEN": "/tmp/t"}, "GO_TOKEN"),
    ({"AUTOLAUNCH_STABLE_POLLS": "1"}, "AUTOLAUNCH_STABLE_POLLS"),
    ({"AUTOLAUNCH_MAX_POLLS": "1"}, "AUTOLAUNCH_MAX_POLLS")])
def test_m_overrides_without_the_harness_are_refused(rig, env, needle):
    rc, out = run(rig, env=env, harness=False, timeout=20)
    assert rc == 2 and needle in out and calls(rig) == []


def test_m2_a_harness_launcher_outside_tmp_is_refused(rig):
    rc, out = run(rig, env={"AUTOLAUNCH_LAUNCHER": os.path.join(REPO, "train.py")})
    assert rc == 2 and "AUTOLAUNCH_LAUNCHER" in out


def test_n_a_non_flac_python_first_on_path_is_refused(rig, tmp_path):
    other = tmp_path / "otherpy"
    other.mkdir()
    (other / "python").write_text("#!/usr/bin/env bash\nexit 0\n")
    os.chmod(other / "python", 0o755)
    rc, out = run(rig, path_prefix=f"{other}:{FLAC_BIN}")
    assert rc == 2 and "flac" in out and calls(rig) == []


def test_the_production_launcher_must_exist(rig):
    (rig["exp"] / "depth_canon_launch.sh").unlink()
    rc, out = run(rig, harness=False, timeout=20)
    assert rc == 2 and "launcher" in out


def test_a_harness_go_token_override(rig, tmp_path):
    tok = tmp_path / "GO.token"
    token(rig, path=tok)
    rc, _ = run(rig, env={"GO_TOKEN": str(tok)})
    assert rc == 0 and len(calls(rig)) == 2


# --------------------------------------------------------------------------- #
# (o) the state file is valid JSON at every observation and replaced atomically
# --------------------------------------------------------------------------- #
def test_o_the_state_file_is_always_whole_json(rig):
    token(rig)
    path = rig["exp"] / "depth_canon_autolaunch_state.json"
    seen, bad, stop = [], [], threading.Event()

    def reader():
        while not stop.is_set():
            try:
                text = path.read_text()
            except FileNotFoundError:
                continue
            try:
                seen.append(json.loads(text)["phase"])
            except ValueError:
                bad.append(text)
    t = threading.Thread(target=reader)
    t.start()
    try:
        rc, _ = run(rig, env={"AUTOLAUNCH_POLL_S": "0.05", "FAKE_SLEEP": "0.3"})
    finally:
        stop.set()
        t.join()
    assert rc == 0 and bad == []
    order = [p for i, p in enumerate(seen) if i == 0 or p != seen[i - 1]]
    assert order[-3:] == ["SMOKE_RUNNING", "FULL_RUNNING", "DONE"] and "WAITING_GPU" in order
    assert not [p for p in os.listdir(rig["exp"]) if "autolaunch_state.json." in p]   # no temp left


def test_the_watcher_never_signals_and_writes_only_inside_expdir(rig):
    """Contract: no `kill` anywhere in the script; a complete run leaves new files only in $EXPDIR."""
    text = open(WATCHER).read()
    assert "kill" not in text and "pkill" not in text and "/media/diskstation" not in text
    token(rig)
    assert run(rig)[0] == 0
    status = _git(rig["repo"], "status", "--porcelain", "--untracked-files=all").splitlines()
    assert status and all(line[3:].startswith(EXPREL.replace(os.sep, "/") + "/") for line in status), status
    new = sorted(p[3:].rsplit("/", 1)[1] for p in status)
    assert "depth_canon_autolaunch_state.json" in new and any(p.endswith("_autolaunch.log") for p in new)



# --------------------------------------------------------------------------- #
# rB-F1 block 2: rA A4 (literal rung + scrubbed state), A6a/b/d/e/f
# --------------------------------------------------------------------------- #
def test_the_launcher_gets_the_literal_rung_and_no_harness_overrides(rig):
    """A4: an inherited MB / ACC / DRY_RUN can never reach the launcher (a stray shell variable
    must not turn the unattended window into a final failure), nor any harness-only override."""
    token(rig)
    rc, out = run(rig, env={"MB": "8", "ACC": "4", "DRY_RUN": "1"})
    assert rc == 0, out
    assert all(c.endswith("DRY_RUN=0 MB=32 ACC=1") for c in calls(rig)) and len(calls(rig)) == 2
    assert (rig["h"] / "launch_env").read_text() == ""           # GO_TOKEN / AUTOLAUNCH_* scrubbed
    text = open(WATCHER).read()
    assert "env -u DEPTH_CANON_HARNESS -u GO_TOKEN -u AUTOLAUNCH_LAUNCHER" in text   # production scrub


def test_only_rc_7_is_a_transient_gpu_refusal(rig):
    """A6e: the launcher's dedicated rc 7 is the ONE transient class; the old text match is gone
    (rc 2 carrying a GPU string is a final failure)."""
    token(rig)
    launches(rig, ["SMOKE 2 GPU GATE FAILED: GPU 0 free 100 MiB < 12288 MiB (12 GiB) - abort"])
    rc, _ = run(rig, max_polls=20)
    assert rc == 1 and len(calls(rig)) == 1 and state(rig)["phase"] == "FAILED"
    assert "GPU_GATE_STR" not in open(WATCHER).read()


def _launcher_rc7_remap():
    """The launcher's terminal code for a NON-GPU-gate exit 7 (train.py's own status), from its source."""
    import re
    src = open(os.path.join(REPO, EXPREL, "depth_canon_launch.sh")).read()
    m = re.search(r"^RC_TRAINER7=(\d+)\b", src, re.M)
    assert m, "the launcher defines no RC_TRAINER7 (rB-F2 F2-1: rc 7 is reserved for the GPU gate)"
    return int(m.group(1)), src


def test_a_trainer_exit_7_remapped_by_the_launcher_is_final(rig):
    """rB-F2 F2-1 (Codex A2 HIGH 1): train.py exiting 7 must never look like the GPU gate's
    transient refusal -- the launcher remaps it to a terminal code, documented in its header,
    and the watcher makes ONE attempt and enters FAILED."""
    remap, src = _launcher_rc7_remap()
    assert remap not in (0, 7) and f"{remap} " in src.split("set -euo pipefail")[0]
    token(rig)
    launches(rig, [f"SMOKE {remap} train.py exited rc=7; exit 7 is reserved for the GPU gate"])
    rc, _ = run(rig, max_polls=20)
    st = state(rig)
    assert rc == 1 and len(calls(rig)) == 1 and (st["phase"], st["smoke_rc"], st["full_rc"]) == ("FAILED", remap, None)


def test_a_near_matching_interpreter_is_refused(rig, tmp_path):
    """A6a: the resolved path must BE the launcher's pinned interpreter, not match a glob."""
    fake = tmp_path / "x" / "envs" / "flac" / "bin"
    fake.mkdir(parents=True)
    (fake / "python").write_text("#!/usr/bin/env bash\nexit 0\n")
    os.chmod(fake / "python", 0o755)
    token(rig)
    rc, out = run(rig, path_prefix=f"{fake}:{FLAC_BIN}")
    assert rc == 2 and "/home/yixunhu/miniconda3/envs/flac/bin/python" in out and calls(rig) == []


@pytest.mark.parametrize("issued, blank, end", [("2026-10-24", False, "\n"), ("2026-10-24T02:00", False, "\n"),
                                                ("2026-10-24T02:00:00", False, "\n"),
                                                ("2026-10-24T02:00:00-04:00", True, "\n"),
                                                ("2026-10-24T02:00:00-04:00", True, ""),
                                                ("2026-10-24T02:00:00-04:00", False, "")],
                         ids=["date_only", "no_seconds", "no_offset", "blank_line",
                              "blank_line_no_final_newline", "no_final_newline"])
def test_a_token_must_be_exactly_four_lines_with_a_full_iso_timestamp(rig, issued, blank, end):
    """A6b; rB-F2 F2-5 (Codex A2 NIT 5): every logical line counts, blanks included -- four keys +
    a blank line + no final newline has four newline bytes but five lines and is NOT a token."""
    text = (f"head={rig['head']}\nmanifest_sha256={pins_sha(rig)}\nissued={issued}\n" + ("\n" if blank else "")
            + "by=planner" + end)
    (rig["exp"] / "depth_canon_GO.token").write_text(text)
    rc, _ = run(rig, max_polls=2)
    assert rc == 3 and calls(rig) == [] and state(rig)["phase"] == "WAITING_TOKEN"


def test_a_dirty_pinned_path_waits_for_a_clean_tree(rig):
    """A6d: a dirty pinned file is WAITING_TOKEN (the launcher would refuse it), never a fired
    launch that fails terminally."""
    (rig["repo"] / "train.py").write_text("print('pinned')\n")
    _git(rig["repo"], "add", "-A")
    _git(rig["repo"], "commit", "-q", "-m", "a pinned file")
    rig["head"] = _git(rig["repo"], "rev-parse", "HEAD")
    token(rig)
    (rig["repo"] / "train.py").write_text("print('edited')\n")
    rc, _ = run(rig, max_polls=3)
    assert rc == 3 and calls(rig) == [] and state(rig)["phase"] == "WAITING_TOKEN"
    assert "dirty" in state(rig)["reason"] and "train.py" in state(rig)["reason"]


def test_the_log_is_complete_when_the_watcher_exits(rig):
    """A6f: the outer tee is drained before exit (the last state line and FULL's output are in
    the log the moment the watcher returns)."""
    token(rig)
    launches(rig, ["SMOKE 0 smoke ok", "FULL 0 full ok " + "x" * 2000])
    rc, _ = run(rig)
    text = open(os.path.join(rig["repo"], state(rig)["log"])).read()
    assert rc == 0 and "full ok" in text and text.rstrip().endswith("phase DONE: SMOKE and FULL both exited 0")


# --------------------------------------------------------------------------- #
# rC-F1 (plan v14): the shared-GPU policy, mirrored from the launcher
# --------------------------------------------------------------------------- #
def _literal(text, name):
    import re
    found = re.findall(rf"^{name}=(\S+)", text, re.M)
    assert len(found) == 1, f"{name}: expected exactly one top-level assignment, found {found}"
    return found[0]


def test_constants_mirror_the_launcher():
    """The watcher's floor and policy are the launcher's pinned literals (12 GiB, shared)."""
    watcher = open(WATCHER).read()
    launcher = open(os.path.join(REPO, EXPREL, "depth_canon_launch.sh")).read()
    for name, want in (("MIN_GPU_FREE_MIB", "12288"), ("GPU_OCCUPANCY_POLICY", "shared")):
        assert _literal(watcher, name) == _literal(launcher, name) == want, name


def _watcher_with_policy(r, policy):
    """Commit a copy of the watcher whose policy literal is <policy> into the rig's repo."""
    path = r["exp"] / "depth_canon_autolaunch.sh"
    text = path.read_text()
    assert text.count("\nGPU_OCCUPANCY_POLICY=shared\n") == 1
    path.write_text(text.replace("\nGPU_OCCUPANCY_POLICY=shared\n", f"\nGPU_OCCUPANCY_POLICY={policy}\n"))
    _git(r["repo"], "add", "-A")
    _git(r["repo"], "commit", "-q", "-m", f"policy {policy}")
    r["head"] = _git(r["repo"], "rev-parse", "HEAD")


def test_idle_policy_still_refuses_occupied_cards(rig):
    """GPU_OCCUPANCY_POLICY=idle restores the v10 predicate: any compute process on GPU 0/1 is dirty."""
    _watcher_with_policy(rig, "idle")
    token(rig)
    gpu(rig, ["busy"])
    rc, out = run(rig, max_polls=4)
    st = state(rig)
    assert rc == 3 and calls(rig) == [] and st["phase"] == "WAITING_GPU", out
    assert "foreign compute process" in st["reason"] and "pid 4242" in st["reason"]
    assert out.count("pid 4242") == 1 and "co-tenants: " not in out


def test_an_unknown_policy_literal_is_refused(rig):
    """Only idle | shared exist: any other literal is a precondition refusal (exit 2), never a launch."""
    _watcher_with_policy(rig, "relaxed")
    token(rig)
    rc, out = run(rig, max_polls=4)
    assert rc == 2 and "GPU_OCCUPANCY_POLICY" in out and calls(rig) == []
