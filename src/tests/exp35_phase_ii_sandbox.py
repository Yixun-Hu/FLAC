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
                   dry_max_age_s=10800, kit_dir=os.path.dirname(KIT_PY), helper=KIT_PY)
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


BEGIN, END = "# --- BEGIN REGISTERED CONFIG", "# --- END REGISTERED CONFIG ---"


def replace_block(text, values):
    """``text`` with ONLY its registered-config block replaced by ``values`` (exactly the block's names)."""
    i = text.index(BEGIN)
    j = text.index(END, i)
    first, body = text[i:text.index("\n", i)], text[text.index("\n", i) + 1:j]
    import ast
    names = sorted(t.id for node in ast.parse(body).body for t in (
        node.targets[0].elts if isinstance(node.targets[0], ast.Tuple) else node.targets))
    assert sorted(values) == names, (sorted(values), names)
    return text[:i] + first + "\n" + "".join(f"{k} = {v!r}\n" for k, v in values.items()) + text[j:]


def replace_sh_block(text, values):
    """The shell script ``text`` with ONLY its registered-config block replaced (exactly the block's names)."""
    i = text.index(BEGIN)
    j = text.index(END, i)
    first, body = text[i:text.index("\n", i)], text[text.index("\n", i) + 1:j]
    names = sorted(ln.split("=", 1)[0] for ln in body.splitlines() if ln and not ln.startswith("#"))
    assert sorted(values) == names, (sorted(values), names)
    return text[:i] + first + "\n" + "".join(f"{k}={v}\n" for k, v in values.items()) + text[j:]


def fake_nvidia_smi(root, gpu_map=MAP, apps=""):
    """<root>/bin/nvidia-smi answering the two queries from <root>/bin/{map,apps}.txt (the tests edit them)."""
    d = os.path.join(str(root), "bin")
    os.makedirs(d, exist_ok=True)
    for name, text in (("map.txt", gpu_map), ("apps.txt", apps)):
        with open(os.path.join(d, name), "w") as f:
            f.write(text)
    with open(os.path.join(d, "nvidia-smi"), "w") as f:
        f.write(f"#!/bin/sh\ncase \"$1\" in --query-gpu=index,uuid) cat '{d}/map.txt';; "
                f"--query-compute-apps=pid,gpu_uuid) cat '{d}/apps.txt';; *) exit 9;; esac\n")
    os.chmod(os.path.join(d, "nvidia-smi"), 0o755)
    return os.path.join(d, "nvidia-smi")


def sandbox_block(cfg):
    """The helper's registered-config values naming the sandbox ``cfg`` (the copy a detached wrapper runs)."""
    exp30 = os.path.dirname(next(iter(cfg.exp30_pins)))
    return {"RUN": cfg.run, "H35": cfg.h35, "CAMPAIGN_DIGEST": cfg.digest, "RESERVATIONS": cfg.reservations,
            "EXP30": exp30, "EXP30_PINS": {os.path.basename(k): v for k, v in cfg.exp30_pins.items()},
            "QUEUE_LOG": os.path.relpath(cfg.queue_log, exp30), "NVIDIA_SMI": cfg.nvidia_smi, "PROC": cfg.proc,
            "SCAN_PROC": cfg.scan_proc, "PY": cfg.py, "GAP_S": cfg.gap_s, "CONFIRM_S": cfg.confirm_s,
            "IDENTIFY_S": cfg.identify_s, "DRY_MAX_AGE_S": cfg.dry_max_age_s}


def make_box(tmp_path, lanes="0", decision="none", runner=None, prior="none", **cfg_over):
    """A complete sandbox: the run worktree, the completed phase (i), Yixun's files, the kit checkout -- its helper
    copy's config block naming THIS sandbox -- with a matching committed report, and the Config of these paths."""
    import types
    run, head, digest = make_run(tmp_path, runner)
    state_sha = write_phase_i(run, head, digest)
    report = report_text(head, state_sha, lanes=lanes, decision=decision, prior=prior)
    kit_dir = os.path.join(str(tmp_path), "kit", X35)
    cfg = cfg_for(tmp_path, run=run, h35=head, digest=digest, kit_dir=kit_dir,
                  helper=os.path.join(kit_dir, "exp35_phase_ii.py"), **cfg_over)
    with open(KIT_PY) as f:
        helper = replace_block(f.read(), sandbox_block(cfg))
    files = {"exp35_phase_ii.py": helper}
    if os.path.exists(KIT_SH):
        with open(KIT_SH) as f:
            files["exp35_phase_ii.sh"] = replace_sh_block(f.read(), {"PY": cfg.py})
    kit_dir, commit = make_kit(tmp_path, report, files=files)
    for rel in (f"{X25}/.depth_canon_eval.lock",):
        open(os.path.join(run, rel), "w").close()
    rsha = hashlib.sha256(report.encode()).hexdigest()
    return types.SimpleNamespace(cfg=cfg, run=run, head=head, digest=digest, state_sha=state_sha, kit_dir=kit_dir,
                                 commit=commit, report=REPORT_REL, report_sha=rsha, tmp=str(tmp_path),
                                 res=lambda u=GPU0: os.path.join(cfg.reservations, f"{u}.json"))


# --------------------------------------------------------------------------- #
# part 4: the stub runner (bash, so its identity is a bash process like the real one; its body in Python). It
# reproduces depth_canon_eval_run.sh's interface and output at H35 -- the lock, the tee'd log, the plan JSON, the
# no-flash pre-flight log, DRY_RUN's exact lines -- and a real-run imitation (receipts, DONE / FAILED lines, the
# summary, rc 0 / 1 / 2 / 4). Faults come from <stub>/control.json. It never touches a GPU or the NAS.
# --------------------------------------------------------------------------- #
BENIGN = ("/home/yixunhu/miniconda3/envs/flac/lib/python3.10/site-packages/lightning_fabric/__init__.py:40: "
          "pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The "
          "pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin "
          "to Setuptools<81.")
PREFLIGHT = ("exp35 no-flash overlay: flash_attn disabled, attention backend = torch SDPA\n"
             "flash_attn not installed, disabling Flash Attention\n"
             "EXP35_NO_FLASH_LAUNCHER attention_backend=sdpa "
             "overlay_sha256=0d0e52ca693e236a2e0a44cd61f61539fdf995cff5132d1f621181a0f3e7c023 "
             "launcher_sha256=bbd2847d3a50685420750d066195c8656a1509b4e905883886c0568d244043de target=--preflight\n")
STUB_PY = r'''
import json, os, subprocess, sys, time
X25 = "worklog/worklog_yixun/exp_25_depth_canon_claude"
X35 = "worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude"
STUB = os.path.dirname(os.path.abspath(__file__))
BENIGN, PREFLIGHT = json.load(open(os.path.join(STUB, "texts.json")))
FARM, CANON = "outputs_FLAC/exp25_depth_canon", json.load(open(os.path.join(STUB, "canon.json")))
pid, args = int(sys.argv[1]), sys.argv[2:]
ctl = json.load(open(os.path.join(STUB, "control.json"))) if os.path.exists(os.path.join(STUB, "control.json")) else {}
dry = os.environ.get("DRY_RUN") == "1"
with open(os.path.join(STUB, "calls.jsonl"), "a") as f:
    f.write(json.dumps({"pid": pid, "argv": args, "dry": dry, "cwd": os.getcwd()}) + "\n")
opts = dict(zip(args[::2], args[1::2]))
if len(args) % 2 or set(opts) - {"--blocks", "--lanes", "--canon-ckpt-manifest"}:
    print("unknown argument - abort"); sys.exit(2)
c = ctl.get("dry" if dry else "run", {})
ts = time.strftime("%Y-%m-%d_%H-%M-%S")
log = open(f"{X25}/depth_canon_{ts}_eval_run.log", "ab")
def say(line, internal=True):
    b = (line + "\n").encode()
    sys.stdout.buffer.write(b); sys.stdout.flush()
    if internal:
        log.write(b); log.flush()
head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
m = json.load(open(f"{X25}/depth_canon_cells.json"))
cfgs = [a["model_config"] for a in m["arms"].values()]
rel = ["eval_FLAC.py", "src", ":(exclude)src/tests", "data/AR", *cfgs, f"{X25}/depth_canon_eval_run.sh",
       f"{X25}/depth_canon_collect.py", f"{X25}/depth_canon_make_cells.py", f"{X25}/depth_canon_cells.json",
       f"{X25}/depth_canon_audit_gate.py", "overlays/no_flash", f"{X35}/exp35_gpu_identity.py"]
digest = c.get("digest") or subprocess.run(["bash", "-c", 'git ls-files -z --cached --others --exclude-standard -- "$@" '
    "| LC_ALL=C sort -zu | xargs -0r sha256sum -- | sha256sum | cut -c1-64", "_", *rel],
    capture_output=True, text=True).stdout.strip()
blocks = opts.get("--blocks", "").split(",")
rows = [r for r in m["cells"] if r["block"] in blocks]
if c.get("e2fail"):
    rows += [r for r in m["cells"] if r["block"] == "E2-FAIL"][:1]
n, lanes = len(rows), c.get("lanes", opts.get("--lanes"))
def entry(r):
    ck = m["arms"][r["arm"]]["ckpt"]
    tgt, sha = (CANON[0], CANON[1]) if ck["kind"] == "canon_run" else (ck["path"], ck["sha256"])
    name = os.path.basename(tgt); stem = name[:-5] + "_metrics_" + r["metrics_suffix"]; link = f"{FARM}/{r['arm']}/{name}"
    return tgt, sha, link, f"{FARM}/{r['arm']}/{stem}.cell.json", r["eval_args"] + ["--ckpt-path", link,
        "--dump-scored-waveforms", m["dump_root"]]
say(f"=== exp_25 depth_canon eval campaign -- {ts} -- HEAD {head} -- dry_run={int(dry)} -- harness=false ===")
if c.get("abort"):
    say("PLAN REFUSED: unknown, duplicate or empty block selection - abort"); sys.exit(2)
say(f"evaluator pin OK: {head}; evaluator tree digest {digest}")
say(f"no-flash overlay + launcher OK: 0d0e52ca693e / bbd2847d3a50 ({n} exp_35 cell(s))")
if c.get("warning", True):
    say(BENIGN)
say("cells manifest OK")
if c.get("warning", True):
    say(BENIGN)
say(f"plan: {n} cell(s) -> {X25}/depth_canon_{ts}_eval_plan.json")
import importlib.util
spec = importlib.util.spec_from_file_location("kit_for_the_stub_plan", json.load(open(os.path.join(STUB, "kit.json"))))
kit = importlib.util.module_from_spec(spec); spec.loader.exec_module(kit)
plan = kit.expected_plan(m, rows, run=os.path.realpath(os.getcwd()), digest=digest)   # the real runner's plan JSON
if c.get("plan_tamper"):
    plan["cells"][3]["argv"] = plan["cells"][3]["argv"][:-2]
json.dump(plan, open(f"{X25}/depth_canon_{ts}_eval_plan.json", "w"), indent=1)
say(f"plan: {n} cell(s); dump root {m['dump_root']}; farm {FARM}; lanes {lanes}")
say("configs OK: every model / dataset config of the selection matches its manifest pin")
open(f"{X25}/depth_canon_{ts}_no_flash_preflight.log", "w").write(PREFLIGHT + c.get("preflight_extra", ""))
say(f"BACKEND OK {X25}/depth_canon_{ts}_no_flash_preflight.log")
say(f"no-flash pre-flight OK: {n} exp_35 cell(s) spawn through overlays/no_flash/exp35_no_flash_launch.py "
    f"(log {X25}/depth_canon_{ts}_no_flash_preflight.log)")
say(f"dump root OK under DRY_RUN: {m['dump_root']} (exists)" if dry else
    f"dump root OK: {m['dump_root']} is cifs //10.249.7.28/shared_data, write probe created + removed")
say("checkpoint cache space: 1 checkpoint(s) to stage = 723927531 B + 2147483648 B margin = 2871411179 B needed; "
    "714080342016 B available on outputs_FLAC/exp25_depth_canon/.ckpt_cache")
if c.get("extra"):
    say(c["extra"])
if c.get("internal_extra"):
    log.write((c["internal_extra"] + "\n").encode()); log.flush()
root, seen, done, failed = os.path.realpath(os.getcwd()), set(), 0, []
for r in rows:
    tgt, sha, link, receipt, argv = entry(r)
    if c.get("argv_tamper") == r["cell"]:
        argv = argv[:-2]
    if c.get("drop") == r["cell"]:
        continue
    if r["arm"] not in seen:
        seen.add(r["arm"])
        say(f"checkpoint OK (DRY_RUN, not staged): {tgt} ({sha})" if dry else f"checkpoint OK: {tgt} -> cache ({sha})")
    if c.get("skip") == r["cell"]:
        say(f"CELL {r['cell']} | {r['block']} | SKIP (complete and verified)"); continue
    q = c.get("quarantine") == r["cell"]
    say(f"CELL {r['cell']} | {r['block']} | {r['arm']} | {'would ' if dry else ''}"
        + ("QUARANTINE 1 artifact(s), then run" if q else "run"))
    if q:
        say(f"  would quarantine {receipt}")
    if dry:
        say(f"DRYRUN {r['cell']} argv: " + json.dumps(argv, separators=(",", ":")))
        say(f"  (exp_35: spawned through overlays/no_flash/exp35_no_flash_launch.py, PYTHONPATH={root}/overlays/no_flash)")
        if c.get("warning_late"):
            say(BENIGN)
        continue
    hold = c.get("hold")
    while hold and os.path.exists(hold):
        time.sleep(0.05)
    say(f"  RUN {r['cell']} on GPU {lanes.split(',')[0]} (log {X25}/depth_canon_{ts}_eval_{r['cell']}.log)")
    time.sleep(c.get("sleep", 0))
    if r["cell"] in c.get("fail", []):
        failed.append(r["cell"]); say(f"  !! FAILED {r['cell']}: the evaluator exited rc=1 (log x)"); continue
    os.makedirs(os.path.dirname(receipt), exist_ok=True)
    open(receipt, "w").write(json.dumps({"cell": r["cell"], "stub": True}))
    done += 1; say(f"  DONE {r['cell']} on GPU {lanes.split(',')[0]}: complete and verified")
if dry:
    say(f"=== DRY_RUN summary: {c.get('summary_n', n)} cell(s) planned; nothing evaluated, linked, created or moved ===")
    sys.exit(c.get("rc", 0))
if c.get("child"):
    subprocess.Popen(["sleep", str(c["child"])])                        # an evaluator descendant outliving the runner
say(f"=== summary: selected {n} | skipped (complete and verified) 0 | ran ok {done} | failed {len(failed)} | not run 0 ===")
for x in failed:
    say(f"  FAILED {x}")
rc = c.get("rc", 1 if failed else 0)
say({0: "CAMPAIGN SELECTION COMPLETE", 1: "CAMPAIGN SELECTION INCOMPLETE - re-run to resume"}.get(
    rc, f"ABORTED: {'PIN DRIFT' if rc == 4 else 'GPU GATE FAILED'} - abort"))
sys.exit(rc)
'''
STUB_SH = '''#!/usr/bin/env bash
# sandbox stand-in for depth_canon_eval_run.sh (the exp_35 phase-(ii) kit tests): never the real runner
cd "$(dirname "$0")/../../.." || exit 2
exec 9>"worklog/worklog_yixun/exp_25_depth_canon_claude/.depth_canon_eval.lock" || exit 2
flock -n 9 || { echo "another evaluation campaign holds worklog/worklog_yixun/exp_25_depth_canon_claude/.depth_canon_eval.lock - abort"; exit 2; }
tr '\\0' '\\n' < /proc/$$/environ > "@STUB@/env_$$.txt"
"@PY@" -B "@STUB@/stub_runner.py" "$$" "$@"
exit $?
'''


def install_stub(root):
    """Write the stub runner's body under <root>/stub; return the bash runner's text (committed at RUNNER)."""
    stub = os.path.join(str(root), "stub")
    os.makedirs(stub, exist_ok=True)
    K = load_kit()
    for name, data in (("stub_runner.py", STUB_PY), ("texts.json", json.dumps([BENIGN, PREFLIGHT])),
                       ("canon.json", json.dumps(list(K.CANON_CKPT))), ("kit.json", json.dumps(KIT_PY))):
        with open(os.path.join(stub, name), "w") as f:
            f.write(data)
    return STUB_SH.replace("@STUB@", stub).replace("@PY@", sys.executable)


def stub_control(root, **sections):
    with open(os.path.join(str(root), "stub", "control.json"), "w") as f:
        json.dump(sections, f)


def stub_calls(root):
    p = os.path.join(str(root), "stub", "calls.jsonl")
    return [json.loads(ln) for ln in open(p)] if os.path.exists(p) else []


def stub_env(root, pid):
    return dict(ln.split("=", 1) for ln in open(os.path.join(str(root), "stub", f"env_{pid}.txt")).read().splitlines()
                if ln)
