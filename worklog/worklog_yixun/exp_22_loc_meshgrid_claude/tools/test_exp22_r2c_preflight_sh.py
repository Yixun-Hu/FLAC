"""exp_22 A7 (plan §9 A7 r2 (v)(2), r3 (b) (f) (h) (i)): the Planner-run launcher, capture -> seal -> launch, in a
sandbox checkout whose tools differ from the reviewed ones ONLY in their registered-config blocks: the hard checks,
the launcher's own sealed polls, the seal's bindings and its 10-min freshness, the single-instance lock, the
reservation, the DETACHED supervisor (setsid + nohup + env -i, the lock fd inherited) and its confirmed identity;
refusals, and faults at the detachment (a supervisor dying at its start, a launcher killed after the reservation)."""
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
import types

import pytest

from conftest import C, GPU0, GPU1, HOLDER, MAP, cfg_for, load, sandbox_repo

E = load("exp22_r2c_evidence")
S = load("exp22_r2c_supervisor")
SCRIPT = os.path.join(os.path.dirname(S.__file__), "exp22_r2c_preflight.sh")
DONE = "EVAL_DONE ACEattn s43 cells=40 2026-11-06T15:00:00-05:00\n"
LAUNCH_ENV = ["HOME=/home/yixunhu", "USER=yixunhu", "LOGNAME=yixunhu", "LANG=en_US.UTF-8",
              "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "PYTHONDONTWRITEBYTECODE=1"]


WEIGHT_ASSETS = ("weights/exp20/YAW_40k.ckpt", "weights/exp20/BF_40k.ckpt", "weights/exp20/P1_40k.ckpt",
                 "weights/FLAC/FLAC_EMA.ckpt", "weights/FLAC/VAE.ckpt", "weights/AGREE/AGREE_fullAR.pt")
TRACKED_WEIGHT = "weights/FLAC/FLAC_EMA_metrics_tracked_stand_in.json"   # PIN2 tracks files under weights/FLAC/
WEIGHT_LINKS = {"weights/AGREE": "weights/AGREE", "weights/exp20": "weights/exp20",   # the Planner's layout
                "weights/FLAC/FLAC_EMA.ckpt": "weights/FLAC/FLAC_EMA.ckpt", "weights/FLAC/VAE.ckpt": "weights/FLAC/VAE.ckpt"}


def make_lab(tmp_path, poll_s=0.05, fresh_s=600, py=None):
    cfg = cfg_for(tmp_path)
    with open(cfg.queue_log, "a") as f:
        f.write(DONE)
    nv = tmp_path / "nvidia-smi"
    (tmp_path / "apps.txt").write_text("")
    nv.write_text(f"#!/bin/sh\ncase \"$1\" in --query-gpu=*) printf '%s' '{MAP}';; *) cat {tmp_path / 'apps.txt'};; esac\n")
    nv.chmod(0o755)
    w, main = tmp_path / "W", tmp_path / "main"
    w, pin2 = sandbox_repo(tmp_path, tracked={TRACKED_WEIGHT: "{}"}, common=dict(
        RESERVATIONS=cfg.reservations, EXP35_LOCK=cfg.exp35_lock, QUEUE_LOG=cfg.queue_log, PROC="/proc",
        NVIDIA_SMI=str(nv), POLL_S=poll_s, WINDOW_POLLS=4, POLL_SLACK_S=5, FRESH_S=fresh_s, PY=sys.executable,
        PYTEST_MAIN=C.PYTEST_MAIN, CHILD_POLL_S=0.05), shells={
        "exp22_r2c_preflight.sh": dict(PY=py or sys.executable, PYTEST_MAIN=C.PYTEST_MAIN,
                                       OUTPUTS_TARGET=str(w / "outputs_loc"), WEIGHTS_TARGET=str(main / "weights"),
                                       ROOMS_TARGET=str(w / "AcousticRooms"), DINO_CACHE=str(tmp_path / "dinov3"),
                                       CONFIRM_TIMEOUT_S=30),
        "exp22_free_card_detect.sh": dict(PY=sys.executable)})
    for d in ("outputs_loc/exp22/g1_audit", "outputs_loc/exp22/i1_P1_CRN_br256_20260825_194053_merged",
              "AcousticRooms/metadata", "AcousticRooms/depth_map", "AcousticRooms/single_channel_ir_1"):
        os.makedirs(os.path.join(w, d), exist_ok=True)
    for f in ("outputs_loc/exp22/d1_context_manifest.json", "outputs_loc/exp22/g1_audit/geometry_audit_report.json"):
        open(os.path.join(w, f), "w").write("{}")
    for rel in WEIGHT_ASSETS:                                           # the main checkout's untracked weights
        os.makedirs(os.path.dirname(main / rel), exist_ok=True)
        (main / rel).write_text(f"stand-in {rel}\n")
    for rel, target in WEIGHT_LINKS.items():                            # one link per untracked entry (the layout)
        os.symlink(main / target, os.path.join(w, rel))
    os.makedirs(tmp_path / "dinov3")                                    # the stand-in DINOv3 HF cache
    script = os.path.join(w, "worklog", "worklog_yixun", "exp_22_loc_meshgrid_claude", "tools", "exp22_r2c_preflight.sh")
    sha = E.sha256_bytes(open(script, "rb").read())

    def run(*args, expect=sha, popen=False):
        env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"], "EXPECT_SCRIPT_SHA256": expect}
        argv = ["bash", script, *args]
        if popen:
            return subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return subprocess.run(argv, env=env, capture_output=True, text=True, timeout=600)
    lab = types.SimpleNamespace(w=str(w), pin2=pin2, cfg=cfg, run=run, sha=sha, tmp=tmp_path, main=main,
                                ex=os.path.join(str(w), "outputs_loc", "exp22"),
                                res=lambda u=GPU0: os.path.join(cfg.reservations, f"{u}.json"))
    lab.control = lambda d: open(os.path.join(lab.w, "a7_control.json"), "w").write(json.dumps(d))
    return lab


def captured(lab, card="GPU0"):
    p = lab.run("capture", lab.pin2, card)
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-2000:]
    m = re.search(r"^SEAL (\S+) sha256 ([0-9a-f]{64})$", p.stdout, re.M)
    return m[1], m[2], p.stdout


def wait_terminal(orch, timeout=240):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            st = json.load(open(os.path.join(orch, "state.json")))
            if st["phase"] in ("released", "stopped"):
                return st
        except (OSError, ValueError):
            pass
        time.sleep(0.2)
    raise AssertionError("the detached supervisor did not finish")


def test_capture_seal_launch_run_detached_to_the_release(tmp_path):
    lab = make_lab(tmp_path)
    seal, ssha, out = captured(lab)
    orch = os.path.dirname(seal)
    assert sorted(os.listdir(orch)) == ["capture.log", "capture_polls.jsonl", "seal.txt"]
    text = open(seal).read()
    for key in ("pin2 " + lab.pin2, "run " + orch.rsplit("_", 1)[1], f"card GPU0 {GPU0} 0", "capture_record ",
                "first_poll ", "last_poll ", "capture_log ", "sealed_at ", f"script_sha256 {lab.sha}"):
        assert key in text, key
    assert text.count("\ntool ") == 6 and text.count("\npin ") == 4
    log = open(os.path.join(orch, "capture.log")).read()
    assert "CAPTURE CHECKS OK" in log and "CHECK FAILED" not in log and log.count("CHECK OK") > 20
    lab.control({"vanilla_cache_parity": {"sleep": 3}})                 # keep the supervisor alive for a look
    p = lab.run("launch", lab.pin2, seal, ssha)
    assert p.returncode == 0 and "LAUNCHED: supervisor pid" in p.stdout, p.stdout[-4000:]
    pid = int(re.search(r"LAUNCHED: supervisor pid (\d+)", p.stdout)[1])
    env = sorted(x for x in open(f"/proc/{pid}/environ", "rb").read().decode().split("\0") if x)
    stat = open(f"/proc/{pid}/stat").read()
    assert env == sorted(LAUNCH_ENV)                                       # env -i: exactly the launch environment
    assert int(stat[stat.rindex(")") + 2:].split()[3]) == pid             # setsid: its own session
    assert os.readlink(f"/proc/{pid}/fd/0") == "/dev/null" and os.readlink(f"/proc/{pid}/cwd") == lab.w
    assert C.lock_status(S.lock_path(lab.w)) == "held"                     # the inherited single-instance lock
    st = wait_terminal(orch)
    assert st["phase"] == "released" and len(st["lines"]) == 15 and all(x["admitted"] for x in st["lines"])
    assert st["supervisor"]["pid"] == pid and not os.path.lexists(lab.res())
    deadline = time.time() + 60
    while os.path.exists(f"/proc/{pid}") and time.time() < deadline:      # "released" is written just before it exits
        time.sleep(0.05)
    assert C.lock_status(S.lock_path(lab.w)) == "unheld"
    assert sorted(f for f in os.listdir(orch) if f.startswith("dispatch_"))[0] == "dispatch_01_T.json"


@pytest.mark.parametrize("args, rc", [([], 2), (["capture", "x" * 40, "GPU0"], 2), (["capture", "a" * 40, "GPU2"], 2),
                                      (["launch", "a" * 40, "seal"], 2), (["nonsense", "a" * 40], 2)])
def test_bad_arguments_are_refused(tmp_path, args, rc):
    lab = make_lab(tmp_path)
    assert lab.run(*args).returncode == rc


def test_the_capture_refuses_without_the_reviewed_script_sha256(tmp_path):
    lab = make_lab(tmp_path)
    p = lab.run("capture", lab.pin2, "GPU0", expect="0" * 64)
    assert p.returncode == 1 and "CHECK FAILED: EXPECT_SCRIPT_SHA256" in p.stdout and "SEAL " not in p.stdout


@pytest.mark.parametrize("breakage", ["busy-card", "head", "dirty", "asset", "active-exp22", "pin", "gpu1-table-job",
                                      "t-checkpoint", "t-agree", "t-depth-map", "t-dinov3-cache", "t-vae", "t-wavs",
                                      "weights-dir-link", "asset-copy", "asset-wrong-link"])
def test_the_capture_refuses_and_seals_nothing(tmp_path, breakage):
    lab = make_lab(tmp_path)
    card = "GPU1" if breakage == "gpu1-table-job" else "GPU0"
    if breakage == "busy-card":
        (lab.tmp / "apps.txt").write_text(f"4242, {GPU0}\n")
    elif breakage == "head":
        subprocess.run(["git", "-C", lab.w, "-c", "user.email=x@y", "-c", "user.name=x", "commit", "-q", "--allow-empty",
                        "-m", "moved"], check=True)
    elif breakage == "dirty":
        open(os.path.join(lab.w, "localize_meshgrid.py"), "a").write("# edited\n")
    elif breakage == "asset":
        os.unlink(lab.main / "weights" / "exp20" / "BF_40k.ckpt")
    elif breakage in ("t-checkpoint", "t-vae"):                          # a missing per-entry link
        os.unlink(os.path.join(lab.w, "weights", "FLAC", "FLAC_EMA.ckpt" if breakage == "t-checkpoint" else "VAE.ckpt"))
    elif breakage == "t-agree":                                          # the T tests would skip without it
        os.unlink(lab.main / "weights" / "AGREE" / "AGREE_fullAR.pt")
    elif breakage == "t-wavs":
        os.rmdir(os.path.join(lab.w, "AcousticRooms", "single_channel_ir_1"))
    elif breakage == "weights-dir-link":                                 # the whole directory as one link
        shutil.rmtree(os.path.join(lab.w, "weights"))
        os.symlink(lab.main / "weights", os.path.join(lab.w, "weights"))
    elif breakage == "asset-copy":                                       # byte-identical, another inode
        link = os.path.join(lab.w, "weights", "FLAC", "VAE.ckpt")
        os.unlink(link)
        shutil.copy(lab.main / "weights" / "FLAC" / "VAE.ckpt", link)
    elif breakage == "asset-wrong-link":                                 # a link to another file
        link = os.path.join(lab.w, "weights", "FLAC", "FLAC_EMA.ckpt")
        os.unlink(link)
        os.symlink(lab.main / "weights" / "FLAC" / "VAE.ckpt", link)
    elif breakage == "t-depth-map":
        os.rmdir(os.path.join(lab.w, "AcousticRooms", "depth_map"))
    elif breakage == "t-dinov3-cache":
        os.rmdir(lab.tmp / "dinov3")
    elif breakage == "active-exp22":
        open(lab.res(GPU1), "w").write(json.dumps({"holder": "exp22", "run": "20261101T000000"}))
    elif breakage == "pin":
        subprocess.run(["git", "-C", lab.w, "rm", "-q", "--cached", "pyproject.toml"], check=True)
        open(os.path.join(lab.w, "pyproject.toml"), "a").write("# x\n")
    else:
        open(lab.cfg.queue_log, "w").write("2026-11-04T06:12:55-05:00 START training ACEattn s43\n")
    p = lab.run("capture", lab.pin2, card)
    assert p.returncode == 1 and "CHECK FAILED" in p.stdout and "SEAL " not in p.stdout, p.stdout[-3000:]
    assert not [d for d in os.listdir(lab.ex) if d.startswith("r2_preflight_") and
                os.path.exists(os.path.join(lab.ex, d, "seal.txt"))]
    expect = {"weights-dir-link": ["CHECK FAILED: the tracked tree is clean",
                                   "CHECK FAILED: weights/ is this worktree's own directory"],
              "asset-copy": ["CHECK FAILED: asset weights/FLAC/VAE.ckpt is the main checkout's file"],
              "asset-wrong-link": ["CHECK FAILED: asset weights/FLAC/FLAC_EMA.ckpt is the main checkout's file"],
              "t-vae": ["CHECK FAILED: asset weights/FLAC/VAE.ckpt"],
              "t-wavs": ["CHECK FAILED: asset AcousticRooms/single_channel_ir_1/"]}.get(breakage, [])
    assert all(e in p.stdout for e in expect), (expect, p.stdout[-3000:])


def test_the_per_entry_layout_passes_the_capture_with_tracked_weights_files(tmp_path):
    lab = make_lab(tmp_path)
    tracked = subprocess.run(["git", "-C", lab.w, "ls-files", "weights"], capture_output=True, text=True).stdout.split()
    assert tracked == [TRACKED_WEIGHT] and os.path.isdir(os.path.join(lab.w, "weights"))
    seal, ssha, out = captured(lab)
    assert "CHECK OK: weights/ is this worktree's own directory" in out
    assert all(f"CHECK OK: asset {rel} is the main checkout's file" in out for rel in WEIGHT_ASSETS), out[-3000:]


def _script_array(name):
    return re.search(name + r"=\(([^)]*)\)", open(SCRIPT).read())[1].split()


def test_pin2_tracks_files_under_weights_so_the_per_asset_form_is_required():
    """The defect A7-C3 fixes: at PIN2 weights/ holds tracked files, so it can never be one link (git would read them
    as deleted); every registered weights asset is an untracked file reached through a per-entry link."""
    repo = os.path.abspath(os.path.join(os.path.dirname(SCRIPT), "..", "..", "..", ".."))
    tracked = subprocess.run(["git", "-C", repo, "ls-files", "weights"], capture_output=True, text=True,
                             check=True).stdout.split()
    assert tracked and all(t.startswith("weights/FLAC/") for t in tracked)
    assert tuple(_script_array("WEIGHT_ASSETS")) == WEIGHT_ASSETS and not set(WEIGHT_ASSETS) & set(tracked)


def test_the_weight_assets_cover_every_weights_file_the_lines_and_t_open():
    """Statically: the gates' and the probe's --ckpt-path, the model config's AGREE checkpoint (the scorer every line
    builds), the AGREE dinoV3 audio tower's pretrained VAE (opened while that scorer is built) and T's released
    FLAC checkpoint (test_eval_localization._FLAC_CKPT); T's other files are not under weights/."""
    repo = os.path.abspath(os.path.join(os.path.dirname(SCRIPT), "..", "..", "..", ".."))
    lines = json.load(open(os.path.join(os.path.dirname(SCRIPT), "exp22_r2c_preflight_lines.json")))["lines"]
    ckpts = {ln["argv"][ln["argv"].index("--ckpt-path") + 1] for ln in lines if "--ckpt-path" in ln["argv"]}
    agree = json.load(open(os.path.join(repo, "src/configs/model_configs/FLAC/AR/FLAC_AR.json")))["training"]["metrics"]
    vae = json.load(open(os.path.join(repo, "AGREE/AGREE/model_configs/dinoV3.json")))["audio_cfg"]["pretrained"]
    text = open(os.path.join(repo, "src/tests/test_eval_localization.py")).read()
    flac = re.search(r'^_FLAC_CKPT = os\.path\.join\(_REPO_ROOT, "weights", "FLAC", "FLAC_EMA\.ckpt"\)$', text, re.M)
    want = ckpts | {agree["AGREE_ckpt"], vae, "weights/FLAC/FLAC_EMA.ckpt" if flac else "?"}
    assert ckpts == {"weights/exp20/YAW_40k.ckpt", "weights/exp20/BF_40k.ckpt", "weights/exp20/P1_40k.ckpt"}
    assert want == set(WEIGHT_ASSETS), want


def test_a_card_that_turns_busy_during_the_sealed_polls_is_not_sealed(tmp_path):
    lab = make_lab(tmp_path, poll_s=0.5)
    p = lab.run("capture", lab.pin2, "GPU0", popen=True)
    deadline = time.time() + 60
    while time.time() < deadline and not [d for d in os.listdir(lab.ex) if d.startswith("r2_preflight_") and
                                          os.path.exists(os.path.join(lab.ex, d, "capture_polls.jsonl"))]:
        time.sleep(0.01)
    (lab.tmp / "apps.txt").write_text(f"4242, {GPU0}\n")                   # a process arrives mid-window
    out, _ = p.communicate(timeout=120)
    assert p.returncode == 1 and "CHECK FAILED: every sealed poll on time and free" in out and "SEAL " not in out


def test_the_launch_refuses_a_stale_capture(tmp_path):
    lab = make_lab(tmp_path, fresh_s=2)
    seal, ssha, _ = captured(lab)
    time.sleep(2.5)
    p = lab.run("launch", lab.pin2, seal, ssha)
    assert p.returncode == 1 and "stale" in p.stdout and not os.path.exists(lab.res())
    assert not os.path.exists(os.path.join(os.path.dirname(seal), "state.json"))


@pytest.mark.parametrize("tamper", ["seal-sha", "record", "capture-log", "tool", "foreign-reservation", "lock-held",
                                    "wrong-pin2", "wrong-script"])
def test_the_launch_refuses_and_reserves_nothing(tmp_path, tamper):
    lab = make_lab(tmp_path)
    seal, ssha, _ = captured(lab)
    orch, args, expect, holder = os.path.dirname(seal), ["launch", lab.pin2, seal, ssha], lab.sha, None
    if tamper == "seal-sha":
        args[3] = "0" * 64
    elif tamper == "record":
        open(os.path.join(orch, "capture_polls.jsonl"), "a").write("{}\n")
    elif tamper == "capture-log":
        open(os.path.join(orch, "capture.log"), "a").write("edited\n")
    elif tamper == "tool":
        path = os.path.join(lab.w, "worklog", "worklog_yixun", "exp_22_loc_meshgrid_claude", "tools",
                            "exp22_r2c_preflight_lines.json")
        open(path, "a").write(" ")                                          # (also dirties the tracked tree)
    elif tamper == "foreign-reservation":
        open(lab.res(), "w").write(json.dumps({"holder": "exp35"}))
    elif tamper == "lock-held":
        holder = subprocess.Popen([sys.executable, "-c", HOLDER, S.lock_path(lab.w)], stdout=subprocess.PIPE, text=True)
        assert holder.stdout.readline().strip() == "locked"
    elif tamper == "wrong-pin2":
        args[1] = "b" * 40
    else:
        expect = "1" * 64
    try:
        p = lab.run(*args, expect=expect)
    finally:
        if holder:
            holder.kill()
            holder.wait()
    assert p.returncode == 1 and "LAUNCHED" not in p.stdout, p.stdout[-3000:]
    assert not os.path.exists(os.path.join(orch, "state.json"))
    if tamper == "foreign-reservation":
        assert json.load(open(lab.res())) == {"holder": "exp35"}             # untouched
    else:
        assert not os.path.exists(lab.res())


def test_a_supervisor_dying_at_its_start_is_not_confirmed_and_everything_stays(tmp_path):
    wrapper = tmp_path / "py_wrapper.sh"
    wrapper.write_text(f"#!/bin/sh\nfor a in \"$@\"; do [ \"$a\" = run ] && exit 7; done\nexec {sys.executable} \"$@\"\n")
    wrapper.chmod(0o755)
    lab = make_lab(tmp_path, py=str(wrapper))
    seal, ssha, _ = captured(lab)
    p = lab.run("launch", lab.pin2, seal, ssha)
    orch = os.path.dirname(seal)
    assert p.returncode == 1 and "LAUNCH NOT CONFIRMED" in p.stdout
    st = json.load(open(os.path.join(orch, "state.json")))
    assert st["phase"] == "reserved" and os.path.exists(lab.res())          # the reservation and the state stay
    assert C.lock_status(S.lock_path(lab.w)) == "unheld"                    # nothing lives
    p = lab.run("launch", lab.pin2, seal, ssha)
    assert p.returncode == 1 and "LAUNCHED" not in p.stdout                 # a restart refuses
    assert json.load(open(os.path.join(orch, "state.json")))["phase"] == "reserved"


def test_a_launch_after_a_launcher_killed_past_its_reservation_refuses(tmp_path):
    lab = make_lab(tmp_path)
    seal, ssha, _ = captured(lab)
    orch = os.path.dirname(seal)
    sup = os.path.join(lab.w, "worklog", "worklog_yixun", "exp_22_loc_meshgrid_claude", "tools", "exp22_r2c_supervisor.py")
    r = subprocess.run([sys.executable, "-B", sup, "reserve", "--orch", orch, "--pin2", lab.pin2, "--card", "GPU0",
                        "--index", "0", "--seal", seal, "--seal-sha256", ssha], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout                                      # what a killed launcher leaves behind
    p = lab.run("launch", lab.pin2, seal, ssha)
    assert p.returncode == 1 and "LAUNCHED" not in p.stdout and os.path.exists(lab.res())
    out = lab.run("capture", lab.pin2, "GPU1")
    assert out.returncode == 1 and "CHECK FAILED: no exp22 run is active" in out.stdout   # nor can a new run start


def test_the_shell_config_and_literals_agree_with_the_python_ones():
    text = open(SCRIPT).read()
    block = text[text.index("# --- BEGIN REGISTERED CONFIG"):text.index("# --- END REGISTERED CONFIG ---")]
    vals = dict(ln.split("=", 1) for ln in block.splitlines()[1:] if ln and not ln.startswith("#"))
    assert vals == {"PY": C.PY, "PYTEST_MAIN": C.PYTEST_MAIN, "OUTPUTS_TARGET": "/home/yixunhu/codespace/FLAC/outputs_loc",
                    "WEIGHTS_TARGET": "/home/yixunhu/codespace/FLAC/weights",
                    "ROOMS_TARGET": "/media/diskstation/yixunhu/FLAC/AcousticRooms", "CONFIRM_TIMEOUT_S": "60",
                    "DINO_CACHE": "/home/yixunhu/.cache/huggingface/hub/models--facebook--dinov3-vits16-pretrain-lvd1689m"}
    assert f"GPU0={GPU0}; GPU1={GPU1}" in text
    for rel, sha in E.PIN_FILES.items():
        assert f'"{rel}|{sha}"' in text
    env = re.search(r"LAUNCH_ENV=\(([^)]*)\)", text)[1].split()
    assert env == LAUNCH_ENV and dict(x.split("=", 1) for x in env[:5]) == E.BASE_ENV
    tools = re.search(r"TOOL_FILES=\(([^)]*)\)", text)[1].split()
    assert tools == list(S.TOOLS)


def test_the_launcher_never_signals_any_process():
    code = [ln.split("#", 1)[0] for ln in open(SCRIPT).read().splitlines()]
    assert not [ln for ln in code if re.search(r"(?<![-\w])(kill|pkill|killall|skill|timeout)\b", ln)]   # no command
    assert "setsid" in "".join(code) and "nohup" in "".join(code) and "env -i" in "".join(code)
