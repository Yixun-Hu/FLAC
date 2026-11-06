"""exp_35 phase (ii) kit tests: the shared sandbox (plan_orbitrir_yaw_evidence.md §15; Codex plan §15 r1 S6 "a
completely sandboxed harness"). Everything here lives under pytest's tmp_path: never the real runner, a GPU
command, the NAS, a real reservation file, a production process identity or the run worktree. The real H35
bytes the kit pins (the cells manifest, the CANON manifest, the supervisor) are read from git OBJECTS only.

Written by the exp_35 phase-(ii) kit Coder (Claude Opus 5.5, max effort).
"""
import copy
import functools
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
X35 = "worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude"
X25 = "worklog/worklog_yixun/exp_25_depth_canon_claude"
KIT_PY = os.path.join(REPO, X35, "exp35_phase_ii.py")
KIT_SH = os.path.join(REPO, X35, "exp35_phase_ii.sh")
H35_REAL = "5b065801e9d1c1844461ec84cd0558702c714326"
DIGEST_REAL = "da82d0121413b792e2f42b94026b884d371d995c17e2dcc111c7edd58f8650c1"
SNAPSHOT = os.path.join(REPO, X35, "orbitrir_yaw_evidence_2026-11-06_01-54-17_phase_i_state_snapshot.json")
GPU0, GPU1 = "GPU-0a1aaa90-3043-0321-2b0c-c35b9036624e", "GPU-6ac2604a-e84e-0615-1221-ccbed7c268da"
FINAL_EVENT = "phase (i) complete; the reservation released"


@functools.lru_cache(maxsize=None)
def load_kit():
    """The kit helper under test, imported by path (one copy per process; no bytecode written)."""
    spec = importlib.util.spec_from_file_location("exp35_phase_ii", KIT_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@functools.lru_cache(maxsize=None)
def git_show(rev, path):
    """The bytes of ``path`` at ``rev`` from the shared object store (read-only; no worktree is touched)."""
    return subprocess.run(["git", "-C", REPO, "show", f"{rev}:{path}"], capture_output=True, check=True).stdout


def real_cells():
    return git_show(H35_REAL, f"{X25}/depth_canon_cells.json")


def real_canon():
    return git_show(H35_REAL, f"{X25}/depth_canon_2026-10-26_07-44-41_exp25_CANON_checkpoints.sha256")


def real_supervisor():
    return git_show(H35_REAL, f"{X35}/exp35_supervisor.py")


def snapshot_state():
    """The archived REAL phase-(i) state of 01:54 EST Nov 6 (13 / 33 gates_verified, chunk 14 in flight)."""
    with open(SNAPSHOT) as f:
        return json.load(f)


def completed_state(**over):
    """The real snapshot carried to a completed phase (i) exactly as the supervisor ends it: every chunk
    gates_verified (the plain cell chunks shaped like the real chunk 13), the lane and the reservation None, the
    final release event last. ``over`` replaces top-level keys (h35, campaign_digest of every chunk, ...)."""
    doc = snapshot_state()
    proto = copy.deepcopy(doc["chunks"][12])                                # a real plain gates_verified cell chunk
    for c in doc["chunks"]:
        if c["state"] != "gates_verified":
            n, cell = c["n"], c["cell"]
            c.clear()
            c.update(copy.deepcopy(proto), n=n, cell=cell)
    doc["lane"] = doc["reservation"] = None
    t = doc["events"][-1]["t"] + 50000.0
    doc["events"].append({"t": t, "time": "2026-11-06T11:20:00-05:00", "what": FINAL_EVENT})
    digest = over.pop("campaign_digest", None)
    if digest is not None:
        for c in doc["chunks"]:
            if "campaign_digest" in c:
                c["campaign_digest"] = digest
    doc.update(over)
    return doc


# --------------------------------------------------------------------------- #
# part 2: the sandbox world -- a fake clock + nvidia-smi answers, a fake /proc tree, sandbox paths
# --------------------------------------------------------------------------- #
MAP = f"0, {GPU0}\n1, {GPU1}\n"
EVAL_DONE_S43 = "EVAL_DONE ACEattn s43 cells=40 2026-11-06T22:00:00-05:00\n"
END_S43 = "2026-11-06T14:31:02-05:00 END training ACEattn s43 exit=0\n"
HOLDER = r'''
import fcntl, os, sys, time
fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o644)
fcntl.flock(fd, fcntl.LOCK_EX)
print("locked", flush=True)
time.sleep(float(sys.argv[2]) if len(sys.argv) > 2 else 60)
'''


class FakeWorld:
    """nvidia-smi answers from a dict (a callable answer may raise); a fake clock whose sleep advances it."""

    def __init__(self, gpu_map=MAP, apps="", t=1_800_000_000.0):
        self.answers = {"map": gpu_map, "apps": apps}
        self.t, self.slept, self.queries = t, [], []

    def now(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += max(0.0, s)

    def nvsmi(self, query):
        self.queries.append(query)
        a = self.answers[query]
        return a() if callable(a) else a


def fake_proc(root, pid, argv, ppid=1, start=1000, cwd="/somewhere"):
    """One /proc/<pid> entry: cmdline, stat (field 22 = ``start``), cwd and fd/1, fd/2 links."""
    d = os.path.join(str(root), str(pid))
    os.makedirs(os.path.join(d, "fd"))
    with open(os.path.join(d, "cmdline"), "wb") as f:
        f.write(b"".join(a.encode() + b"\0" for a in argv))
    fields = ["S", str(ppid)] + ["0"] * 17 + [str(start)] + ["0"] * 10
    with open(os.path.join(d, "stat"), "w") as f:
        f.write(f"{pid} ({os.path.basename(argv[0])}) " + " ".join(fields) + "\n")
    os.symlink(cwd, os.path.join(d, "cwd"))
    os.symlink("/dev/null", os.path.join(d, "fd", "1"))
    os.symlink("/dev/null", os.path.join(d, "fd", "2"))
    return d


def exp30_files(root):
    """Stand-ins for Yixun's queue script, evaluator and admission helper, and his queue log (s43 ended)."""
    d = os.path.join(str(root), "exp30")
    os.makedirs(os.path.join(d, "logs"), exist_ok=True)
    pins = {}
    for name in ("exp30_attn_queue.sh", "exp30_attn_eval.sh", "exp30_attn_queue_state.py"):
        with open(os.path.join(d, name), "w") as f:
            f.write(f"# stand-in {name}\n")
        with open(os.path.join(d, name), "rb") as f:
            pins[os.path.join(d, name)] = hashlib.sha256(f.read()).hexdigest()
    log = os.path.join(d, "logs", "queue.log")
    with open(log, "w") as f:
        f.write("2026-11-04T06:12:55-05:00 START training ACEattn s43 gpus 0,1\n" + END_S43)
    return pins, log


def cfg_for(tmp_path, **over):
    """A Config naming only sandbox paths (the production block is never used by a test)."""
    K = load_kit()
    root = str(tmp_path)
    for d in ("run", "reservations", "proc", "bin"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    pins, log = exp30_files(root)
    cfg = K.Config(run=os.path.join(root, "run"), h35="a" * 40, digest="d" * 64,
                   reservations=os.path.join(root, "reservations"), exp30_pins=pins, queue_log=log,
                   nvidia_smi=os.path.join(root, "bin", "nvidia-smi"), proc="/proc",
                   scan_proc=os.path.join(root, "proc"), py=sys.executable, gap_s=60, confirm_s=10, identify_s=5.0,
                   dry_timeout_s=120, dry_max_age_s=10800, kit_dir=os.path.dirname(KIT_PY), helper=KIT_PY)
    cfg = cfg._replace(**over)
    assert all(str(v).startswith(root) for v in (cfg.run, cfg.reservations, cfg.queue_log, cfg.scan_proc,
                                                   cfg.nvidia_smi, *cfg.exp30_pins)), "a sandbox path escaped tmp_path"
    return cfg


# --------------------------------------------------------------------------- #
# part 3: a sandbox run worktree (git, on its branch) and a sandbox kit checkout (git) with the committed report
# --------------------------------------------------------------------------- #
RUNNER = f"{X25}/depth_canon_eval_run.sh"
CANON_REL = f"{X25}/depth_canon_2026-10-26_07-44-41_exp25_CANON_checkpoints.sha256"
REPORT_REL = f"{X35}/orbitrir_yaw_evidence_phase_i_report_to_yixun.md"


def git(w, *args):
    return subprocess.run(["git", "-C", str(w), "-c", "user.email=p2kit@sandbox", "-c", "user.name=p2kit", *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def _write(root, rel, data):
    p = os.path.join(str(root), rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(data if isinstance(data, bytes) else data.encode())
    return p


def make_run(root, runner=None):
    """The sandbox run worktree <root>/run: the runner-relevant tree (stand-ins + the REAL cells manifest), the REAL
    CANON manifest and supervisor, committed on branch exp-35-orbitrir-yaw. Returns (path, head, digest)."""
    K = load_kit()
    w = os.path.join(str(root), "run")
    m = json.loads(real_cells())
    files = {"eval_FLAC.py": "# stand-in\n", "src/__init__.py": "", "src/models/transformer.py": "# stand-in\n",
             "src/tests/test_x.py": "# excluded by the pathspec\n", "data/AR/unseen_eval.json": "{}\n",
             f"{X25}/depth_canon_collect.py": "# stand-in\n", f"{X25}/depth_canon_make_cells.py": "# stand-in\n",
             f"{X25}/depth_canon_audit_gate.py": "# stand-in\n", "overlays/no_flash/flash_attn.py": "raise ImportError\n",
             f"{X35}/exp35_gpu_identity.py": "# stand-in\n", "README.md": "sandbox run worktree\n",
             RUNNER: runner or "#!/usr/bin/env bash\necho 'no stub runner in this sandbox - abort'\nexit 2\n",
             f"{X25}/depth_canon_cells.json": real_cells(), CANON_REL: real_canon(),
             f"{X35}/exp35_supervisor.py": real_supervisor()}
    files.update({a["model_config"]: "{}\n" for a in m["arms"].values()})
    for rel, data in files.items():
        _write(w, rel, data)
    os.chmod(os.path.join(w, RUNNER), 0o755)
    git(w, "init", "-q", "-b", "exp-35-orbitrir-yaw")
    git(w, "add", "-A")
    git(w, "commit", "-q", "-m", "sandbox H35")
    return w, git(w, "rev-parse", "HEAD"), K.tree_digest(w, K.relevant(m))


def write_phase_i(run, head, digest, **over):
    """The completed phase-(i) state (+ its unheld lock file) in the sandbox run worktree; returns its sha256."""
    raw = (json.dumps(completed_state(h35=head, campaign_digest=digest, **over), indent=1)).encode()
    p = _write(run, f"{X35}/exp35_supervisor_state.json", raw)
    open(p + ".lock", "w").close()
    return hashlib.sha256(raw).hexdigest()


def report_text(h35, state_sha, lanes="0", decision="none", prior="none", eta="68 cells on GPU 0, about 19 h"):
    body = "".join(f"exp35-phase-ii {k} {v}\n" for k, v in (
        ("h35", h35), ("phase-i-state-sha256", state_sha), ("lanes", lanes), ("second-lane-decision", decision),
        ("eta", eta), ("prior-runs", prior)))
    return "# exp_35 phase (i) report to Yixun (sandbox)\n\nThe measured slowdown, the per-cell time, the plan.\n\n```\n" \
        + body + "```\n"


def make_kit(root, report, files=None):
    """The sandbox kit checkout <root>/kit: the kit files (``files`` {name: text}, else the real ones) and the report,
    committed. Returns (kit_dir, commit)."""
    k = os.path.join(str(root), "kit")
    for name in ("exp35_phase_ii.py", "exp35_phase_ii.sh"):
        src = os.path.join(REPO, X35, name)
        text = (files or {}).get(name)
        if text is None and os.path.exists(src):
            with open(src, "rb") as f:
                text = f.read()
        text = b"# not written yet\n" if text is None else text
        _write(k, f"{X35}/{name}", text)
    _write(k, REPORT_REL, report)
    git(k, "init", "-q", "-b", "localization-exp")
    git(k, "add", "-A")
    git(k, "commit", "-q", "-m", "sandbox kit")
    return os.path.join(k, X35), git(k, "rev-parse", "HEAD")


def make_box(tmp_path, lanes="0", decision="none", runner=None, prior="none", **cfg_over):
    """A complete sandbox for the preconditions and the reserve: the run worktree, the completed phase (i), Yixun's
    files, the kit checkout with a matching committed report, and a Config naming only these paths."""
    import types
    run, head, digest = make_run(tmp_path, runner)
    state_sha = write_phase_i(run, head, digest)
    report = report_text(head, state_sha, lanes=lanes, decision=decision, prior=prior)
    kit_dir, commit = make_kit(tmp_path, report)
    cfg = cfg_for(tmp_path, run=run, h35=head, digest=digest, kit_dir=kit_dir,
                  helper=os.path.join(kit_dir, "exp35_phase_ii.py"), **cfg_over)
    for rel in (f"{X25}/.depth_canon_eval.lock",):
        open(os.path.join(run, rel), "w").close()
    rsha = hashlib.sha256(report.encode()).hexdigest()
    return types.SimpleNamespace(cfg=cfg, run=run, head=head, digest=digest, state_sha=state_sha, kit_dir=kit_dir,
                                 commit=commit, report=REPORT_REL, report_sha=rsha, tmp=str(tmp_path),
                                 res=lambda u=GPU0: os.path.join(cfg.reservations, f"{u}.json"))
