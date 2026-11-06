#!/usr/bin/env python3
"""exp_35 phase (ii): the launch kit's helper (plan_orbitrir_yaw_evidence.md §15 and its r1 clarifications (a)-(c);
Codex plan §15 r1 "S6 -- binding kit and test contract"). ``exp35_phase_ii.sh`` runs it in the phase-(i) launch
environment for the Planner's five modes -- reserve | dry-run | launch | status | release -- and ``launch`` starts
it once more, detached, as the wrapper that runs the H35 runner and durably records its exact exit code.

Part 1a: the registered pins, the durable writers and the pure checks -- the 68-cell selection of the cells manifest
at H35 and its name digest, the CANON re-pin, the runner's plan JSON re-derived row by row, the lane list.
Part 1b: the runner's relevant-tree digest, the phase-(i) completion contract and the report evidence.
Part 2a: the read-only world (nvidia-smi, lock probes, process scans, his table-job rule, /proc identities).
Part 2b: the O_EXCL reservation records with their exact authentication, and the one card observation every mode makes.
Part 3a: the preconditions every mutating mode re-checks -- the kit's self-pin, the report evidence, the run worktree at
H35 with its campaign digest, Yixun's pinned files and the end of his s43 training -- and the evidence record.
Part 3b: the orchestration directory, its durable state machine, the kit lock and the reserve transaction.
Part 4a: the exact runner command and the exact parser of its DRY_RUN transcript.
Part 4b: the dry-run transaction (from 'reserved' only; the runner is never given a timeout: the kit never kills).
Part 5a: the detached wrapper -- the kit lock, both complete identities and environment proofs, the exact rc.
Part 5b: the launch (the matching fresh dry run, the recheck, launch_pending, the last look, the detachment).
Part 6a: the authenticated release and the read-only status.
Part 6b: the CLI dispatch (exp35_phase_ii.sh beside this file is the Planner's entry point).

Never: moves or modifies H35, installs anything, selects E2-FAIL, suppresses dumps, enables GPU sharing,
overwrites / adopts / removes another holder's reservation, signals / kills / attaches to any process,
relaunches, clears a failure, changes a lane or starts exp_22.

Written by the exp_35 phase-(ii) kit Coder (Claude Opus 5.5, max effort).
"""
import argparse
import collections
import datetime
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))

# --- BEGIN REGISTERED CONFIG (the sandbox tests replace this block, and only this block) ---
RUN = "/home/yixunhu/codespace/exp-35-orbitrir-yaw"
H35 = "5b065801e9d1c1844461ec84cd0558702c714326"
CAMPAIGN_DIGEST = "da82d0121413b792e2f42b94026b884d371d995c17e2dcc111c7edd58f8650c1"
RESERVATIONS = "/home/yixunhu/codespace/.gpu_reservations"
EXP30 = "/home/yixunhu/codespace/flac-fork-attn/worklog/worklog_yixun/exp_30_attn"
EXP30_PINS = {"exp30_attn_queue.sh": "9394ca903931de7778f9665e36b355d9cfab1d61e693703a29e02d43c505caaa",
              "exp30_attn_eval.sh": "94fcc9e17da578049f0c4a3a05798d00f7d312426300b1f20786367b7b863d6c",
              "exp30_attn_queue_state.py": "8572bb0305bcd1f1f245a9d2b31ca9760ba7ed036bccf5bf612d2b2208b33907"}
QUEUE_LOG = "logs/queue_20261101T233607_741947.log"
NVIDIA_SMI = "/usr/bin/nvidia-smi"
PROC = "/proc"
SCAN_PROC = "/proc"
PY = "/home/yixunhu/miniconda3/envs/flac/bin/python"
GAP_S, CONFIRM_S, IDENTIFY_S, DRY_MAX_AGE_S = 60, 60, 5.0, 10800
# --- END REGISTERED CONFIG ---

BRANCH = "exp-35-orbitrir-yaw"
X25 = "worklog/worklog_yixun/exp_25_depth_canon_claude"
X35 = "worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude"
RUNNER, CELLS = f"{X25}/depth_canon_eval_run.sh", f"{X25}/depth_canon_cells.json"
CANON_MANIFEST = f"{X25}/depth_canon_2026-10-26_07-44-41_exp25_CANON_checkpoints.sha256"
CELLS_SHA256, N_ROWS = "440eb1befb3445e0b26b01b8e8fa12cb2c37fed76c900864e7efbed5e9e7037d", 346
BLOCKS = ("OFF-P1", "OFF-BF", "OFF-CANON", "CANON-45")                       # §15 item 1; E2-FAIL never selected
COUNTS = {"OFF-P1": 20, "OFF-BF": 20, "OFF-CANON": 20, "CANON-45": 8}
BLOCK_ARM = {"OFF-P1": "P1", "OFF-BF": "BF", "OFF-CANON": "CANON", "CANON-45": "CANON"}
ORDERS, E2_FAIL_ORDERS = tuple(range(262, 330)), tuple(range(330, 346))
NAMES_SHA256 = "6a88112f9cceeb9de3a8056e621b68bdc7c5d1d484aebb2e05a4624c8406d427"   # the 68 names, newline-joined
EXP35_BLOCKS = ("CTL", "BF-pin", "BF-dup", "OFF-P1", "OFF-BF", "OFF-CANON", "CANON-45", "E2-FAIL")   # the collector's
DUMP_ROOT, FARM = "/media/diskstation/yixunhu/FLAC/exp25_pred_waveforms", "outputs_FLAC/exp25_depth_canon"
CANON_MANIFEST_SHA256 = "1e4814472b72375d6f3f42b3773f7bdb5d137719f882d00aef0387b770218cba"
CANON_CKPT = ("/media/diskstation/yixunhu/FLAC/checkpoints/exp25_depth_canon/FLAC_exp25_CANON/exp25_CANON/checkpoints/"
              "epoch=8-step=40000.ckpt", "52c64b525c8a34f1721b73abd5c4aad67295ea2f729426e9330104b6d1441ace")
CARDS = {0: "GPU-0a1aaa90-3043-0321-2b0c-c35b9036624e", 1: "GPU-6ac2604a-e84e-0615-1221-ccbed7c268da"}
LAUNCH_ENV = {"HOME": "/home/yixunhu", "USER": "yixunhu", "LOGNAME": "yixunhu", "SHELL": "/bin/bash",   # §15 item 7
              "LANG": "en_US.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
              "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"}
SHARING = ("ALLOW_SHARED_GPU", "EXP35_APPROVED_COTENANTS", "EXP35_APPROVED_COTENANTS_SHA256")   # absent, always
SUP_STATE, SUP_FILE = f"{X35}/exp35_supervisor_state.json", f"{X35}/exp35_supervisor.py"
SUP_LOCK, RUNNER_LOCK, KIT_LOCK = f"{SUP_STATE}.lock", f"{X25}/.depth_canon_eval.lock", f"{X35}/exp35_phase_ii.lock"
FINAL_EVENT = "phase (i) complete; the reservation released"                 # exp35_supervisor.py's last event
KIT_FILES = ("exp35_phase_ii.sh", "exp35_phase_ii.py")
Config = collections.namedtuple("Config", "run h35 digest reservations exp30_pins queue_log nvidia_smi proc scan_proc "
                                "py gap_s confirm_s identify_s dry_max_age_s kit_dir helper")


def registered_config():
    return Config(RUN, H35, CAMPAIGN_DIGEST, RESERVATIONS, {f"{EXP30}/{k}": v for k, v in EXP30_PINS.items()},
                  f"{EXP30}/{QUEUE_LOG}", NVIDIA_SMI, PROC, SCAN_PROC, PY, GAP_S, CONFIRM_S, IDENTIFY_S,
                  DRY_MAX_AGE_S, HERE, os.path.abspath(__file__))


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def iso(t):
    return datetime.datetime.fromtimestamp(t).astimezone().isoformat(timespec="seconds")


Snap = collections.namedtuple("Snap", "raw sha256 dev inode")


def snapshot(path):
    """ONE no-follow read of the regular file ``path``: its bytes, their sha256, its (device, inode); None when it
    is absent, a symlink, not a regular file or unreadable."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except (OSError, TypeError):
        return None
    try:
        st, chunks = os.fstat(fd), []
        if not stat.S_ISREG(st.st_mode):
            return None
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
    except OSError:
        return None
    finally:
        os.close(fd)
    raw = b"".join(chunks)
    return Snap(raw, sha256_bytes(raw), st.st_dev, st.st_ino)


def fsync_dir(d):
    fd = os.open(d, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_exclusive(path, raw):
    """O_CREAT | O_EXCL (any existing name, a dangling link included, refuses), fully written, the file and its
    directory fsynced: the Snap of exactly the bytes written."""
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        view = memoryview(raw)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
        st = os.fstat(fd)
    finally:
        os.close(fd)
    fsync_dir(os.path.dirname(os.path.abspath(path)))
    return Snap(raw, sha256_bytes(raw), st.st_dev, st.st_ino)


def dumps(doc):
    return (json.dumps(doc, indent=1) + "\n").encode()


def write_json_atomic(path, doc):
    """A temp file beside ``path``, fsync, rename over it, fsync the directory: the old or the new state, whole."""
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".exp35_phase_ii_state.", dir=d)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(dumps(doc))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    fsync_dir(d)


def names_sha256(names):
    return sha256_bytes(("\n".join(names) + "\n").encode())


def selection(raw):
    """(reasons, manifest, rows): the cells manifest at its pin and its 68 phase-(ii) rows in manifest order."""
    out = [] if sha256_bytes(raw) == CELLS_SHA256 else [
        f"the cells manifest hashes to {sha256_bytes(raw)}, not the pin {CELLS_SHA256}"]
    try:
        m = json.loads(raw)
        cells = m["cells"]
        rows = [c for c in cells if c["block"] in BLOCKS]
        e2 = [c["order"] for c in cells if c["block"] == "E2-FAIL"]
        names = [c["cell"] for c in rows]
    except (ValueError, KeyError, TypeError) as err:
        return out + [f"the cells manifest is not the generator's schema ({err!r:.120})"], None, []
    counts = dict(collections.Counter(c["block"] for c in rows))
    out += [why for why, bad in (
        (f"the cells manifest has {len(cells)} rows, not {N_ROWS}", len(cells) != N_ROWS),
        (f"the manifest's dump root {m.get('dump_root')!r} is not {DUMP_ROOT}", m.get("dump_root") != DUMP_ROOT),
        (f"the manifest's farm root {m.get('farm_root')!r} is not {FARM}", m.get("farm_root") != FARM),
        (f"the selected blocks count {counts}, not {COUNTS}", counts != COUNTS),
        (f"the selected rows' orders are not {ORDERS[0]}..{ORDERS[-1]}", [c.get("order") for c in rows] != list(ORDERS)),
        (f"the E2-FAIL rows' orders are {e2}, not {E2_FAIL_ORDERS[0]}..{E2_FAIL_ORDERS[-1]}", e2 != list(E2_FAIL_ORDERS)),
        ("the selected names are not distinct", len(set(names)) != len(names)),
        (f"the selected names digest to {names_sha256(names)}, not {NAMES_SHA256}", names_sha256(names) != NAMES_SHA256))
        if bad]
    out += [f"{c['cell']}: dump is {c.get('dump')!r}, not True" for c in rows if c.get("dump") is not True]
    out += [f"{c['cell']}: arm {c.get('arm')} is not the {c['block']} arm {BLOCK_ARM[c['block']]}" for c in rows
            if c.get("arm") != BLOCK_ARM[c["block"]]]
    return out, m, rows


def canon_reasons(raw):
    """§15 item 3: the CANON manifest at its pin, carrying exactly ONE step-40000 entry -- the pinned one."""
    out = [] if sha256_bytes(raw) == CANON_MANIFEST_SHA256 else [
        f"the CANON manifest hashes to {sha256_bytes(raw)}, not the pin {CANON_MANIFEST_SHA256}"]
    want = f"{CANON_CKPT[1]}  {CANON_CKPT[0]}"
    hits = [ln for ln in raw.decode(errors="replace").splitlines() if ln.endswith("  " + CANON_CKPT[0])]
    return out + ([] if hits == [want] else [f"the CANON manifest's step-40000 entries are {hits}, not [{want!r}]"])


def plan_entry(m, row, canon_index, no_dump=()):
    """One entry of the runner's plan JSON (depth_canon_collect.py plan_cells at H35), re-derived from its row."""
    ck = m["arms"][row["arm"]]["ckpt"]
    target, sha = canon_index[row["step"]] if ck["kind"] == "canon_run" else (ck["path"], ck["sha256"])
    name = os.path.basename(target)
    link, stem = f"{FARM}/{row['arm']}/{name}", name[:-len(".ckpt")] + "_metrics_" + row["metrics_suffix"]
    base, dump = f"{FARM}/{row['arm']}/{stem}", bool(row["dump"]) and row["block"] not in no_dump
    e = {"cell": row["cell"], "block": row["block"], "arm": row["arm"], "step": row["step"], "K": row["K"],
         "seed": row["seed"], "target": target, "sha256": sha, "nas_manifest": ck.get("nas_manifest"),
         "nas_manifest_key": ck.get("nas_manifest_key"), "link": link, "cache": f"{FARM}/.ckpt_cache/{sha}.ckpt",
         "metrics": base + ".json", "stream": base + ".stream.json",
         "canon": base + ".canon.json" if row["canonicalize"] else None, "canon_probe": base + ".canon.json",
         "receipt": base + ".cell.json", "dump": dump, "dump_root": DUMP_ROOT, "dump_dir": f"{DUMP_ROOT}/{stem}",
         "stem": stem, "expected_stream_count": row["expected_stream_count"],
         "argv": list(row["eval_args"]) + ["--ckpt-path", link] + (["--dump-scored-waveforms", DUMP_ROOT] if dump else [])}
    return dict(e, no_flash=True) if row["block"] in EXP35_BLOCKS else e


def expected_plan(m, rows, *, run, digest, no_dump=(), canon_index=None):
    """The runner's whole plan JSON for ``rows`` (the exact planned set and every argv)."""
    index = canon_index or {40000: CANON_CKPT}
    return {"schema_version": 1, "cells_manifest": f"{run}/{CELLS}", "campaign_digest": digest, "farm_root": FARM,
            "dump_root": DUMP_ROOT, "n_selected": len(rows), "cells": [plan_entry(m, r, index, no_dump) for r in rows]}



def parse_lanes(text):
    """The frozen lane list: one or two distinct registered GPU indices, comma-separated ('0', '0,1', '1,0')."""
    if not isinstance(text, str) or not re.fullmatch(r"[0-9](,[0-9])?", text):
        raise ValueError(f"the lane list {text!r} is not one or two comma-separated GPU indices")
    lanes = tuple(int(x) for x in text.split(","))
    if len(set(lanes)) != len(lanes) or any(i not in CARDS for i in lanes):
        raise ValueError(f"the lane list {text!r} repeats a GPU or names an index outside {sorted(CARDS)}")
    return lanes


def lanes_text(lanes):
    return ",".join(str(i) for i in lanes)


def relevant(m):
    """depth_canon_eval_run.sh's RELEVANT for an exp_35 selection (every arm's model config, the overlay, the identity)."""
    return ["eval_FLAC.py", "src", ":(exclude)src/tests", "data/AR", *[a["model_config"] for a in m["arms"].values()],
            RUNNER, f"{X25}/depth_canon_collect.py", f"{X25}/depth_canon_make_cells.py", CELLS,
            f"{X25}/depth_canon_audit_gate.py", "overlays/no_flash", f"{X35}/exp35_gpu_identity.py"]


def git(repo, *args, raw=False):
    """A read-only git query (no optional locks: a status never rewrites the index), waited for without a timeout (a
    timeout would kill it); ValueError when it fails."""
    r = subprocess.run(["git", "--no-optional-locks", "-C", repo, *args], capture_output=True)
    if r.returncode != 0:
        raise ValueError(f"git {' '.join(args[:2])} in {repo} failed (rc {r.returncode}): "
                         f"{r.stderr.decode(errors='replace').strip()[:200]}")
    return r.stdout if raw else r.stdout.decode()


def tree_digest(repo, paths):
    """The runner's tree_digest: the sha256 of the sorted ``sha256sum`` lines of every tracked + untracked (unignored)
    file under ``paths``. A name the runner's sha256sum would escape (a backslash, a newline) is refused."""
    out = git(repo, "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *paths, raw=True)
    names = sorted({n for n in out.split(b"\0") if n})
    if not names or any(b"\\" in n or b"\n" in n for n in names):
        raise ValueError(f"the relevant tree of {repo} is empty or holds a name the runner's sha256sum escapes")
    lines = []
    for n in names:
        s = snapshot(os.path.join(repo, os.fsdecode(n)))
        if s is None:
            raise ValueError(f"the relevant file {os.fsdecode(n)} is not a readable regular file")
        lines.append(f"{s.sha256}  {os.fsdecode(n)}")
    return sha256_bytes(("\n".join(lines) + "\n").encode())


def tree_dirty(repo, paths):
    """The runner's tree_dirty: '' iff every file under ``paths`` is committed and clean (untracked files count)."""
    return git(repo, "status", "--porcelain", "--untracked-files=all", "--", *paths)


def phase_i_chunks():
    """exp35_supervisor.py chunk_plan() at H35 as (n, kind, cell, gate): the frozen 33-chunk order."""
    t = lambda k, s, x="": ("cell", f"exp35_BF_T_S40000_K{k}_s{s}{x}", None)  # noqa: E731
    out = [("cell", "exp35_P1_T_S40000_K1_s42", "E1"), ("cell", "exp35_P1_T_S40000_K8_s42", "E1"), t(1, 42),
           ("cell", "exp35_BF_T_S40000_K8_s42", "E2"), ("probe", None, "PROBE")]
    out += [t(k, s) for s in (43, 44, 45, 46) for k in (1, 8)]
    out += [("cell", f"exp35_BF_R_S40000_K{k}_s{s}", None) for s in (42, 43, 44, 45, 46) for k in (1, 8)]
    out += [t(k, 42, "_dup") for k in (1, 8)]
    out += [("cell", f"exp35_BF_C4_S40000_K{k}_s42_th{d}", None) for d in (90, 180, 270, 45) for k in (1, 8)]
    return tuple((i + 1, *c) for i, c in enumerate(out))


PHASE_I_CHUNKS = phase_i_chunks()
PHASE_I_KEYS = ("schema_version", "campaign", "h35", "cells_sha256", "supervisor_sha256", "approved_sha256", "baseline",
                "his_log_anchor", "lane", "reservation", "chunks", "stop_latch", "events")
GATES = {"E1": ("e1", "equal"), "E2": ("e2", "PASS")}


def phase_i_reasons(doc, *, h35, digest, supervisor_sha256):
    """§15 item 4: [] iff ``doc`` is the supervisor's state of a COMPLETE phase (i) -- schema v1 at H35, the 33 frozen
    chunks gates_verified with rc 0 (or adopted at a restart), E1 equal, E2 PASS, the probe contract rc 0, ONE
    campaign digest, no stop latch, no lane and no reservation, the final release event last."""
    if not isinstance(doc, dict) or list(doc) != list(PHASE_I_KEYS):
        return [f"the phase-(i) state's keys are not the supervisor's schema-v1 keys {list(PHASE_I_KEYS)}"]
    out = [f"phase (i) {k} {doc[k]!r} != {v!r}" for k, v in (
        ("schema_version", 1), ("campaign", "exp35"), ("h35", h35), ("cells_sha256", CELLS_SHA256),
        ("supervisor_sha256", supervisor_sha256)) if doc[k] != v]
    ch = doc["chunks"]
    if not (isinstance(ch, list) and all(isinstance(c, dict) for c in ch)
            and [tuple(c.get(k) for k in ("n", "kind", "cell", "gate")) for c in ch] == list(PHASE_I_CHUNKS)):
        return out + ["the phase-(i) chunks are not the frozen 33-chunk order"]
    for c in ch:
        n, rc_key = c["n"], "probe_rc" if c["kind"] == "probe" else "runner_rc"
        if c.get("state") != "gates_verified":
            out.append(f"chunk {n} is {c.get('state')}, not gates_verified")
            continue
        rc, adopted = c.get(rc_key), c.get("adopted_at_restart") is True
        if not ((type(rc) is int and rc == 0) or (adopted and rc is None)):
            out.append(f"chunk {n}: {rc_key} {rc!r} (not 0, nor a chunk adopted at a restart)")
        if c["gate"] in GATES:
            key, want = GATES[c["gate"]]
            r = c.get(key) if isinstance(c.get(key), dict) else {}
            if not (r.get("rc") == 0 and r.get("verdict") == want):
                out.append(f"chunk {n}: {c['gate']} is {r.get('verdict')!r} (rc {r.get('rc')!r}), not {want}")
        if c["gate"] == "PROBE" and not (isinstance(c.get("probe_contract"), dict) and c["probe_contract"].get("rc") == 0):
            out.append(f"chunk {n}: the probe contract is {c.get('probe_contract')!r:.80}, not rc 0")
    digests = sorted({str(c.get("campaign_digest")) for c in ch if c["kind"] == "cell"})
    out += [] if digests == [digest] else [f"the cell chunks carry campaign digests {digests}, not exactly [{digest}]"]
    out += [why for why, bad in (
        (f"the stop latch is set: {doc['stop_latch']!r:.120}", doc["stop_latch"] is not None),
        (f"the phase-(i) lane is still set: {doc['lane']!r}", doc["lane"] is not None),
        (f"the phase-(i) reservation is still held: {doc['reservation']!r:.160}", doc["reservation"] is not None)) if bad]
    ev = doc["events"]
    last = ev[-1].get("what") if isinstance(ev, list) and ev and isinstance(ev[-1], dict) else None
    return out + ([] if last == FINAL_EVENT else [
        f"the last phase-(i) event is {last!r}, not the final release event {FINAL_EVENT!r}"])


REPORT_KEYS = ("h35", "phase-i-state-sha256", "lanes", "second-lane-decision", "eta", "prior-runs")
REPORT_LINE = re.compile(r"exp35-phase-ii ([a-z0-9-]+) (\S.*)")


def report_fields(raw):
    """(fields, reasons) of the Planner's committed phase-(i) report: each line holding ``exp35-phase-ii`` must be one
    unindented ``exp35-phase-ii <key> <value>`` field; the six REPORT_KEYS exactly once each, nothing else."""
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as err:
        return {}, [f"the report is not UTF-8 ({err})"]
    fields, out = {}, []
    for ln in lines:
        if "exp35-phase-ii" not in ln:
            continue
        m = REPORT_LINE.fullmatch(ln)
        if m is None or m[1] not in REPORT_KEYS or m[1] in fields:
            out.append(f"a malformed, unknown or repeated report line {ln!r}")
            continue
        fields[m[1]] = m[2]
    return fields, out + [f"the report carries no exp35-phase-ii {k} line" for k in REPORT_KEYS if k not in fields]


def report_reasons(fields, *, h35, lanes, state_sha256, prior_runs):
    """The report binds this launch: H35, the final phase-(i) state, the frozen lane list (a second lane only with a
    recorded decision), and every earlier launched orchestration (a relaunch only after a report naming it)."""
    dec, two = fields.get("second-lane-decision"), len(lanes) > 1
    out = [f"the report's {k} is {fields.get(k)!r}, not {want!r}" for k, want in (
        ("h35", h35), ("phase-i-state-sha256", state_sha256), ("lanes", lanes_text(lanes)),
        ("prior-runs", ",".join(sorted(prior_runs)) or "none")) if fields.get(k) != want]
    return out + ([f"the report's second-lane-decision {dec!r} does not fit {len(lanes)} lane(s): a second lane needs "
                   "a recorded decision, one lane exactly 'none'"] if (dec == "none") == two else [])


# --------------------------------------------------------------------------- #
# part 2a: the read-only world -- nvidia-smi, locks, processes, his table job, identities
# --------------------------------------------------------------------------- #
class Unreadable(ValueError):
    """An input of a check could not be read: never taken as free, absent or gone."""


class World:
    """The real world: wall time, sleep and nvidia-smi (every failure is Unreadable)."""
    QUERIES = {"map": "--query-gpu=index,uuid", "apps": "--query-compute-apps=pid,gpu_uuid"}

    def __init__(self, nvidia_smi):
        self.nvidia_smi = nvidia_smi

    def now(self):
        return time.time()

    def sleep(self, s):
        time.sleep(max(0.0, s))

    def nvsmi(self, query):
        if query not in self.QUERIES:
            raise ValueError(f"unknown nvidia-smi query {query!r}")
        argv = [self.nvidia_smi, self.QUERIES[query], "--format=csv,noheader"]
        try:
            r = subprocess.run(argv, capture_output=True, text=True)     # no timeout: a timeout would kill it
        except (OSError, subprocess.SubprocessError) as err:
            raise Unreadable(f"nvidia-smi {argv[1]}: {err}") from err
        if r.returncode != 0:
            raise Unreadable(f"nvidia-smi {argv[1]}: rc {r.returncode}: {r.stderr.strip()[:200]}")
        return r.stdout


ROW = re.compile(r"\s*(\d+),\s*(GPU-[0-9A-Za-z-]+)\s*")


def gpu_map(text):
    """``{index: uuid}`` from ``nvidia-smi --query-gpu=index,uuid``; ValueError when malformed, duplicate or empty."""
    out = {}
    for line in [ln for ln in text.splitlines() if ln.strip()]:
        m = ROW.fullmatch(line)
        if not m or int(m[1]) in out or m[2] in out.values():
            raise ValueError(f"malformed or duplicate nvidia-smi index,uuid row {line!r}")
        out[int(m[1])] = m[2]
    if not out:
        raise ValueError("nvidia-smi lists no GPU")
    return out


def compute_apps(text):
    """``[(pid, uuid)]`` from ``nvidia-smi --query-compute-apps=pid,gpu_uuid``; ValueError when a row is malformed."""
    out = []
    for line in [ln for ln in text.splitlines() if ln.strip()]:
        m = ROW.fullmatch(line)
        if not m:
            raise ValueError(f"malformed nvidia-smi compute-app row {line!r}")
        out.append((int(m[1]), m[2]))
    return out


def lock_status(path):
    """'absent' | 'unheld' | 'held': opened O_RDONLY (never created), LOCK_SH | LOCK_NB, closed at once (no lock
    outlives the probe); Unreadable on any other error."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return "absent"
    except OSError as err:
        raise Unreadable(f"the lock {path} is unreadable ({err})") from err
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        return "unheld"
    except BlockingIOError:
        return "held"
    except OSError as err:
        raise Unreadable(f"the lock {path} cannot be probed ({err})") from err
    finally:
        os.close(fd)


WATCHED = ("exp35_supervisor.py", "depth_canon_eval_run.sh")                        # alive anywhere: refused
IN_RUN = ("exp35_no_flash_launch.py", "eval_FLAC.py", "depth_canon_cond_probe.py")  # alive in the run worktree


def _cmdlines(proc):
    """(pid, argv) of every process but this one; a vanished entry is skipped, an unreadable one is Unreadable."""
    try:
        names = os.listdir(proc)
    except OSError as err:
        raise Unreadable(f"{proc} is unreadable ({err})") from err
    for n in names:
        if not n.isdigit() or int(n) == os.getpid():
            continue
        try:
            with open(os.path.join(proc, n, "cmdline"), "rb") as f:
                raw = f.read()
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError as err:
            raise Unreadable(f"{proc}/{n}/cmdline is unreadable ({err})") from err
        yield int(n), [a.decode(errors="replace") for a in raw.split(b"\0") if a]


def _in_run(proc, pid, run):
    try:
        cwd = os.path.realpath(os.readlink(os.path.join(proc, str(pid), "cwd")))
    except FileNotFoundError:
        return False
    except OSError as err:
        raise Unreadable(f"the cwd of process {pid} is unreadable ({err})") from err
    root = os.path.realpath(run)
    return cwd == root or cwd.startswith(root + os.sep)


def live_procs(cfg):
    """[(pid, why)]: every process that may belong to an exp_35 run -- a supervisor or a runner anywhere, a phase-(ii)
    wrapper, an evaluator or probe in the run worktree (S6: no live phase-(i) identity; release: nothing of ours)."""
    out = []
    for pid, argv in _cmdlines(cfg.scan_proc):
        base = [os.path.basename(a) for a in argv]
        hit = next((b for b in base if b in WATCHED), None)
        if hit is None and "exp35_phase_ii.py" in base and "wrap" in argv:
            hit = "a phase-(ii) wrapper (exp35_phase_ii.py wrap)"
        if hit is None and any(b in IN_RUN for b in base) and _in_run(cfg.scan_proc, pid, cfg.run):
            hit = next(b for b in base if b in IN_RUN) + " in the run worktree"
        if hit:
            out.append((pid, f"{hit} runs"))
    return sorted(out)


EVAL_DONE, TABLE_JOB = "EVAL_DONE ACEattn s43 ", "exp30_attn_eval.sh"


def table_job_reasons(cfg):
    """GPU 1 is a lane only once his s43 table job is terminally complete (exp_22's card rule (d)): his queue log
    carries an ``EVAL_DONE ACEattn s43 `` line and no exp30_attn_eval.sh process runs."""
    try:
        with open(cfg.queue_log, "rb") as f:
            lines = f.read().decode(errors="replace").splitlines()
    except OSError as err:
        raise Unreadable(f"his queue log {cfg.queue_log} is unreadable ({err})") from err
    out = [] if any(ln.startswith(EVAL_DONE) for ln in lines) else [
        f"his queue log has no {EVAL_DONE!r} line: his s43 table job is not terminally complete (GPU 1)"]
    running = sorted(p for p, argv in _cmdlines(cfg.scan_proc) if TABLE_JOB in (os.path.basename(a) for a in argv))
    return out + ([f"{TABLE_JOB} runs (pids {running}): his table job"] if running else [])


ID_KEYS = ("pid", "start_time", "cwd", "cmdline_sha256")


def _stat_identity(text):
    """(pid, start time): fields 1 and 22 of /proc/<pid>/stat (counted after the LAST ')' of the comm)."""
    return int(text.split(" ", 1)[0]), int(text[text.rindex(")") + 2:].split()[19])


def proc_identity(pid, proc):
    """The identity of a LIVE process -- pid, start time, cwd, cmdline sha256, fd 1 / 2 -- read through ONE directory
    fd of /proc/<pid> between two stat reads that must agree; ValueError when gone, unreadable or reused."""
    pid = int(pid)
    try:
        fd = os.open(os.path.join(proc, str(pid)), os.O_RDONLY | os.O_DIRECTORY)
    except OSError as err:
        raise ValueError(f"process {pid}: its /proc identity is unreadable ({type(err).__name__}: {err})") from err
    opener = lambda path, flags: os.open(path, flags, dir_fd=fd)  # noqa: E731
    try:
        with open("stat", opener=opener) as f:
            first = _stat_identity(f.read())
        with open("cmdline", "rb", opener=opener) as f:
            cmdline = f.read()
        links = [os.readlink(p, dir_fd=fd) for p in ("cwd", "fd/1", "fd/2")]
        with open("stat", opener=opener) as f:
            second = _stat_identity(f.read())
    except (OSError, ValueError, IndexError) as err:
        raise ValueError(f"process {pid}: its /proc identity is unreadable ({type(err).__name__}: {err})") from err
    finally:
        os.close(fd)
    if not first == second == (pid, first[1]):
        raise ValueError(f"process {pid}: its /proc entry changed during the sample ({first} -> {second}): PID reuse")
    return {"pid": pid, "start_time": first[1], "cwd": links[0], "cmdline_sha256": sha256_bytes(cmdline),
            "fd1": links[1], "fd2": links[2]}


def identity_status(ident, proc):
    """'alive' (its pid runs with its recorded start time), 'gone', or 'reused' (the pid names another process) --
    never by a pid alone; ValueError for an incomplete record, Unreadable for an unreadable live entry."""
    if not (isinstance(ident, dict) and all(k in ident for k in ID_KEYS) and type(ident["pid"]) is int
            and ident["pid"] > 0 and type(ident["start_time"]) is int):
        raise ValueError(f"an incomplete identity {ident!r:.120} identifies no process")
    path = os.path.join(proc, str(ident["pid"]))
    if not os.path.lexists(path):
        return "gone"
    try:
        live = proc_identity(ident["pid"], proc)
    except ValueError as err:
        if not os.path.lexists(path):
            return "gone"
        raise Unreadable(str(err)) from err
    return "alive" if live["start_time"] == ident["start_time"] else "reused"


# --------------------------------------------------------------------------- #
# part 2b: the O_EXCL reservation records and the card observation
# --------------------------------------------------------------------------- #
RES_KEYS = ("holder", "phase", "h35", "gpu_uuid", "gpu_index", "created", "run")


def reservation_path(cfg, uuid):
    return os.path.join(cfg.reservations, f"{uuid}.json")


def reservation_status(cfg, uuid):
    """'absent', 'present (holder X)' or 'present (...)' for any other file (malformed, a dangling link);
    Unreadable when the directory is missing or unreadable or the file cannot be read."""
    try:
        os.listdir(cfg.reservations)
    except OSError as err:
        raise Unreadable(f"the reservation directory {cfg.reservations} is missing or unreadable ({err})") from err
    path = reservation_path(cfg, uuid)
    try:
        os.lstat(path)
    except FileNotFoundError:
        return "absent"
    except OSError as err:
        raise Unreadable(f"the reservation file {path} is unreadable ({err})") from err
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except FileNotFoundError:
        return "present (a dangling link, or removed while it was read)"
    except OSError as err:
        raise Unreadable(f"the reservation file {path} is unreadable ({err})") from err
    try:
        doc = json.loads(raw)
    except ValueError:
        return "present (malformed)"
    return f"present (holder {doc['holder']})" if isinstance(doc, dict) and isinstance(doc.get("holder"), str) else (
        "present (malformed)")


def create_reservation(cfg, index, *, h35, run, now):
    """§15 item 6: the O_EXCL record (holder exp35, phase ii, H35, the card's uuid and index, created, the run id),
    fully written and fsynced with its directory; the held record: path, exact bytes' sha256, (device, inode)."""
    uuid = CARDS[index]
    raw = json.dumps(dict(zip(RES_KEYS, ("exp35", "ii", h35, uuid, index, now, run)))).encode()
    s = write_exclusive(reservation_path(cfg, uuid), raw)
    return {"index": index, "uuid": uuid, "path": reservation_path(cfg, uuid), "sha256": s.sha256, "dev": s.dev,
            "inode": s.inode}


def held_reasons(held):
    """[] iff the file at ``held['path']`` IS our record: its bytes' sha256 and its (device, inode), never via a link."""
    s = snapshot(held["path"])
    if s is None:
        return [f"our reservation {held['path']} is gone, a link or unreadable"]
    return ([] if s.sha256 == held["sha256"] else [f"{held['path']} is not our record (other bytes)"]) + (
        [] if (s.dev, s.inode) == (held["dev"], held["inode"]) else [f"{held['path']} is not our record (another inode)"])


def release_reservation(held):
    """Unlink ``held['path']`` only while it IS our record, through the inode just authenticated, then fsync the
    directory; [] once removed, else the reasons -- and nothing is unlinked."""
    why = held_reasons(held)
    if why:
        return why
    try:
        st = os.lstat(held["path"])
        if (st.st_dev, st.st_ino) != (held["dev"], held["inode"]):
            return [f"{held['path']} was replaced while it was authenticated"]
        os.unlink(held["path"])
    except OSError as err:
        return [f"{held['path']} could not be removed ({err})"]
    fsync_dir(os.path.dirname(held["path"]))
    return []


def observe(world, cfg, lanes, held=()):
    """ONE observation (§15 item 6; S6): the GPU map is exactly the registered index <-> uuid pair; per lane no compute
    process and no reservation file of any holder -- with ``held``: exactly our record -- and on GPU 1 his s43 table
    job terminally complete; no exp35 or unexplained file on another card; no supervisor, runner, wrapper or
    run-worktree evaluator alive; the supervisor and runner locks unheld. Unreadable inputs raise Unreadable."""
    t, map_raw, apps_raw = world.now(), world.nvsmi("map"), world.nvsmi("apps")
    try:
        m, apps = gpu_map(map_raw), compute_apps(apps_raw)
    except ValueError as err:
        raise Unreadable(f"the nvidia-smi tables: {err}") from err
    mine, cards = {h["uuid"]: h for h in held}, {}
    out = [] if m == CARDS else [f"the GPU map {m} is not the registered {CARDS}"]
    out += [f"compute process {p} sits on an unknown uuid {u}" for p, u in apps if u not in CARDS.values()]
    for i, uuid in CARDS.items():
        pids = sorted(p for p, u in apps if u == uuid)
        res = "held" if uuid in mine else reservation_status(cfg, uuid)
        if i in lanes:
            out += [f"GPU {i} ({uuid}) holds compute process(es) {pids}"] if pids else []
            out += held_reasons(mine[uuid]) if uuid in mine else (
                [] if res == "absent" else [f"GPU {i}: a reservation file exists ({res})"])
            out += table_job_reasons(cfg) if i == 1 else []
        elif res == "present (holder exp35)":
            out.append(f"GPU {i}: an exp35 reservation exists ({res}): a phase-(i) leftover or another phase-(ii) record")
        elif res != "absent" and not res.startswith("present (holder "):
            out.append(f"GPU {i}: an unexplained reservation file ({res})")
        cards[str(i)] = {"uuid": uuid, "pids": pids, "reservation": res}
    locks = {k: lock_status(os.path.join(cfg.run, rel)) for k, rel in (("supervisor", SUP_LOCK), ("runner", RUNNER_LOCK))}
    out += [f"the {k} lock is held" for k, v in locks.items() if v == "held"]
    out += [f"pid {p}: {why}" for p, why in live_procs(cfg)]
    return {"t": t, "iso": iso(t), "map": {str(k): v for k, v in m.items()}, "map_raw": map_raw, "apps_raw": apps_raw,
            "apps_sha256": sha256_bytes(apps_raw.encode()), "cards": cards, "locks": locks, "reasons": out}


# --------------------------------------------------------------------------- #
# part 3a: the preconditions every mutating mode re-checks (§15 items 1-5; S6 "checks and durable records")
# --------------------------------------------------------------------------- #
HEX40 = re.compile(r"[0-9a-f]{40}")
S43_LINE = re.compile(r"\S+ (START|END) training ACEattn s43( .*)?")


def _repo_of(path):
    return os.path.realpath(git(path, "rev-parse", "--show-toplevel").strip())


def kit_evidence(cfg, kit_commit):
    """S6 self-pin: both kit files tracked, clean and byte-identical to their blobs at the reviewed ``kit_commit``, the
    kit outside the run worktree. (reasons, the record: commit, checkout, HEAD, the files' sha256)."""
    if not (isinstance(kit_commit, str) and HEX40.fullmatch(kit_commit)):
        return [f"the kit commit {kit_commit!r} is not a 40-hex commit (EXPECT_KIT_COMMIT)"], None
    try:
        repo = _repo_of(cfg.kit_dir)
        head = git(repo, "rev-parse", "HEAD").strip()
    except ValueError as err:
        return [f"the kit checkout: {err}"], None
    kit, root = os.path.realpath(cfg.kit_dir), os.path.realpath(cfg.run)
    out = [f"the kit {kit} lives inside the run worktree {root}"] if (kit + os.sep).startswith(root + os.sep) else []
    files = {}
    for name in KIT_FILES:
        rel, s = os.path.relpath(os.path.join(kit, name), repo), snapshot(os.path.join(kit, name))
        files[name] = s.sha256 if s else None
        try:
            blob = git(repo, "cat-file", "blob", f"{kit_commit}:{rel}", raw=True)
            dirty = git(repo, "status", "--porcelain", "--untracked-files=all", "--", rel)
        except ValueError as err:
            out.append(f"{rel} is not in the kit commit {kit_commit} ({err})")
            continue
        if s is None or s.raw != blob or dirty:
            out.append(f"{rel} is not the committed, clean bytes of the kit commit {kit_commit}")
    return out, {"commit": kit_commit, "repo": repo, "head": head, "files": files}


def report_evidence(cfg, path, sha256):
    """The brief: the Planner's phase-(i) report to Yixun is a file of the kit's checkout, committed at HEAD, clean,
    hashing to the given ``sha256``. (reasons, {path, sha256} or None, its bytes or None)."""
    try:
        repo = _repo_of(cfg.kit_dir)
    except ValueError as err:
        return [f"the kit checkout: {err}"], None, None
    given = os.path.normpath(path if os.path.isabs(path) else os.path.join(repo, path))
    full = os.path.join(os.path.realpath(os.path.dirname(given)), os.path.basename(given))
    rel = os.path.relpath(full, repo)
    if rel == os.curdir or rel.startswith(os.pardir):
        return [f"the report {path} lies outside the kit checkout {repo}"], None, None
    s = None if os.path.islink(full) else snapshot(full)
    if s is None:
        return [f"the report {rel} is a link, or not a readable regular file"], None, None
    out = [] if s.sha256 == sha256 else [f"the report {rel} hashes to {s.sha256}, not the given {sha256}"]
    try:
        blob = git(repo, "cat-file", "blob", f"HEAD:{rel}", raw=True)
        dirty = git(repo, "status", "--porcelain", "--untracked-files=all", "--", rel)
    except ValueError as err:
        return out + [f"the report {rel} is not committed at the kit checkout's HEAD ({err})"], None, None
    out += [] if blob == s.raw and dirty == "" else [f"the report {rel} is not the committed, clean bytes of HEAD"]
    return out, {"path": rel, "sha256": s.sha256}, s.raw


def worktree_reasons(cfg, m):
    """§15 item 2 + r1 (b): the run worktree's root at H35 on its branch, its tracked tree clean, the runner-relevant
    tree committed and clean, and its digest exactly the campaign digest."""
    try:
        top = _repo_of(cfg.run)
        if top != os.path.realpath(cfg.run):
            return [f"{cfg.run} is not a worktree root (its root is {top})"]
        head, branch = git(cfg.run, "rev-parse", "HEAD").strip(), git(cfg.run, "rev-parse", "--abbrev-ref", "HEAD").strip()
        tracked = git(cfg.run, "status", "--porcelain", "--untracked-files=no")
        paths = relevant(m)
        dirty, digest = tree_dirty(cfg.run, paths), tree_digest(cfg.run, paths)
    except ValueError as err:
        return [f"the run worktree: {err}"]
    return [why for why, bad in (
        (f"the run worktree's HEAD {head} is not H35 {cfg.h35}", head != cfg.h35),
        (f"the run worktree is on {branch!r}, not {BRANCH}", branch != BRANCH),
        (f"the run worktree's tracked tree is dirty: {tracked.strip()[:300]!r}", tracked != ""),
        (f"the runner-relevant tree is not committed and clean: {dirty.strip()[:300]!r}", dirty != ""),
        (f"the runner-relevant tree digests to {digest}, not the campaign digest {cfg.digest}", digest != cfg.digest))
        if bad]


def s43_end(cfg):
    """(his s43 END line or None, reasons): his s43 training ended iff the LAST s43 training line of his queue log is
    its END line (exp30_attn_queue.sh writes 'END training <arm> s<seed> exit=<rc>' once the trainer is gone)."""
    try:
        with open(cfg.queue_log, "rb") as f:
            lines = f.read().decode(errors="replace").splitlines()
    except OSError as err:
        return None, [f"his queue log {cfg.queue_log} is unreadable ({err})"]
    last = ([ln for ln in lines if S43_LINE.fullmatch(ln)] or [None])[-1]
    if last is not None and re.fullmatch(r"\S+ END training ACEattn s43 exit=-?\d+", last):
        return last, []
    return None, [f"his s43 training has not ended (its last training line: {last!r})"]


def external_reasons(cfg):
    """§13 NIT 3 / S6 'stale queue pin': his queue script, evaluator and admission helper at their pins, and the end
    of his s43 training."""
    out = []
    for p, want in cfg.exp30_pins.items():
        s = snapshot(p)
        out += [] if s is not None and s.sha256 == want else [f"{p} hashes to {s.sha256 if s else None}, not its pin {want}"]
    return out + s43_end(cfg)[1]


def evidence(cfg, *, lanes, report, report_sha256, prior_runs):
    """Every precondition of §15 items 1-5 and S6, recomputed now: (reasons, the record of what was checked)."""
    cells, canon = snapshot(os.path.join(cfg.run, CELLS)), snapshot(os.path.join(cfg.run, CANON_MANIFEST))
    out, m, rows = selection(cells.raw if cells else b"")
    out += canon_reasons(canon.raw) if canon else [f"the CANON manifest {CANON_MANIFEST} is not a readable file"]
    out += (worktree_reasons(cfg, m) if m is not None else []) + external_reasons(cfg)
    sup, st = snapshot(os.path.join(cfg.run, SUP_FILE)), snapshot(os.path.join(cfg.run, SUP_STATE))
    try:
        doc = json.loads(st.raw) if st else None
    except ValueError:
        doc = None
    out += phase_i_reasons(doc, h35=cfg.h35, digest=cfg.digest, supervisor_sha256=sup.sha256 if sup else None)
    why, rep, raw = report_evidence(cfg, report, report_sha256)
    fields, why2 = report_fields(raw) if raw is not None else ({}, [])
    out += why + why2 + report_reasons(fields, h35=cfg.h35, lanes=lanes, state_sha256=st.sha256 if st else None,
                                       prior_runs=prior_runs)
    ck = {a: [m["arms"][a]["ckpt"]["path"], m["arms"][a]["ckpt"]["sha256"]] for a in ("P1", "BF")} if m else {}
    return out, {"h35": cfg.h35, "run_worktree": cfg.run, "campaign_digest": cfg.digest,
                 "cells_sha256": cells.sha256 if cells else None, "names_sha256": names_sha256([r["cell"] for r in rows]),
                 "n_cells": len(rows), "blocks": list(BLOCKS), "dump_root": m.get("dump_root") if m else None,
                 "canon_manifest": CANON_MANIFEST, "canon_manifest_sha256": canon.sha256 if canon else None,
                 "checkpoints": dict(ck, CANON=list(CANON_CKPT)), "exp30": dict(cfg.exp30_pins),
                 "queue_log": cfg.queue_log, "s43_end": s43_end(cfg)[0], "phase_i_state_sha256": st.sha256 if st else None,
                 "supervisor_sha256": sup.sha256 if sup else None, "report": rep,
                 "lanes": [{"index": i, "uuid": CARDS[i]} for i in lanes], "eta": fields.get("eta"),
                 "second_lane_decision": fields.get("second-lane-decision"), "prior_runs": sorted(prior_runs)}


# --------------------------------------------------------------------------- #
# part 3b: the orchestration, its durable state machine, the kit lock and the reserve transaction
# --------------------------------------------------------------------------- #
ORCH_PREFIX, RUN_RE = "exp35_phase_ii_", re.compile(r"[0-9]{8}T[0-9]{6}")
STATE_KEYS = ("schema_version", "campaign", "run", "kit", "evidence", "lanes", "reservations", "phase", "observations",
              "dry_run", "launch", "wrapper", "runner", "exit", "failure", "released", "events")
RESOLVED = ("released", "reserve_refused")                          # nothing of ours is held in these phases


def state_path(orch):
    return os.path.join(orch, "state.json")


def load_state(orch):
    """The state of an orchestration directory; ValueError when unreadable or not this kit's schema-v1 own-run state."""
    try:
        with open(state_path(orch), "rb") as f:
            doc = json.loads(f.read())
    except (OSError, ValueError) as err:
        raise ValueError(f"the state of {orch} is unreadable ({err})") from err
    if not (isinstance(doc, dict) and list(doc) == list(STATE_KEYS) and doc["schema_version"] == 1
            and doc["campaign"] == "exp35_phase_ii" and os.path.basename(orch) == f"{ORCH_PREFIX}{doc['run']}"):
        raise ValueError(f"the state of {orch} is not this kit's schema-v1 state of its own run")
    return doc


def orchestrations(cfg):
    """[(directory, state or why it is unreadable)] of every phase-(ii) orchestration in the run worktree."""
    base = os.path.join(cfg.run, X35)
    try:
        names = sorted(os.listdir(base))
    except OSError as err:
        raise Unreadable(f"the exp_35 folder {base} is unreadable ({err})") from err
    out = []
    for n in names:
        if n.startswith(ORCH_PREFIX):
            try:
                out.append((os.path.join(base, n), load_state(os.path.join(base, n))))
            except ValueError as err:
                out.append((os.path.join(base, n), str(err)))
    return out


def prior(cfg, exclude=None):
    """(reasons, run ids): every OTHER orchestration must be resolved; those that reached launch_pending are the
    prior runs the report must name (S5: no relaunch without a review and a report)."""
    why, runs = [], []
    for d, st in orchestrations(cfg):
        if exclude is not None and os.path.realpath(d) == os.path.realpath(exclude):
            continue
        if not isinstance(st, dict) or st["phase"] not in RESOLVED:
            why.append(f"the orchestration {d} is unresolved ({st if not isinstance(st, dict) else st['phase']})")
        if isinstance(st, dict) and st["launch"] is not None:
            runs.append(st["run"])
    return why, runs


def event(st, now, what, **kw):
    st["events"].append(dict({"t": now, "time": iso(now), "what": what}, **kw))


def kit_lock(cfg):
    """The kit's single-instance lock (LOCK_EX | LOCK_NB): its fd, or None while another process holds it -- a kit
    mode, our detached wrapper, or a runner / evaluator descendant that inherited it."""
    fd = os.open(os.path.join(cfg.run, KIT_LOCK), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def reserve(world, cfg, *, kit_commit, h35, lanes, report, report_sha256):
    """§15 item 6 + S6: every precondition at TWO observations ``gap_s`` apart, then a fresh orchestration with its
    state 'reserving' (O_EXCL), each lane's O_EXCL record (all or nothing), the observation again AFTER the
    acquisition, then 'reserved'; else 'reserve_refused' (only our exact records released) or 'reserve_failed' (a
    record no longer ours: retained). (rc, message)."""
    why, kit = kit_evidence(cfg, kit_commit)
    why += [] if h35 == cfg.h35 else [f"the given H35 {h35} is not the registered {cfg.h35}"]
    if why:
        return 1, "RESERVE REFUSED (nothing reserved): " + "; ".join(why)
    lock = kit_lock(cfg)
    if lock is None:
        return 1, f"RESERVE REFUSED (nothing reserved): the kit lock {KIT_LOCK} is held (a kit mode or a run lives)"
    try:
        return _reserve(world, cfg, kit, lanes, report, report_sha256)
    finally:
        os.close(lock)


def _reserve(world, cfg, kit, lanes, report, report_sha256):
    try:
        why, runs = prior(cfg)
        bad, ev = evidence(cfg, lanes=lanes, report=report, report_sha256=report_sha256, prior_runs=runs)
        obs = [observe(world, cfg, lanes)]
        why += bad + obs[0]["reasons"]
        if not why:
            world.sleep(cfg.gap_s)
            obs.append(observe(world, cfg, lanes))
            why += obs[1]["reasons"] + ([] if obs[1]["map"] == obs[0]["map"] else ["the GPU map changed between the "
                                                                                 "two observations"])
    except Unreadable as err:
        why = [f"unreadable: {err}"]
    if why:
        return 1, "RESERVE REFUSED (nothing reserved): " + "; ".join(why)
    run = time.strftime("%Y%m%dT%H%M%S", time.localtime(world.now()))
    orch = os.path.join(cfg.run, X35, ORCH_PREFIX + run)
    st = dict(zip(STATE_KEYS, (1, "exp35_phase_ii", run, kit, ev, ev["lanes"], [], "reserving", obs) + (None,) * 7 + ([],)))
    event(st, world.now(), "reserving: the preconditions held at both observations; the O_EXCL records next")
    os.mkdir(orch)
    write_exclusive(state_path(orch), dumps(st))
    held = []
    try:
        for i in lanes:
            held.append(create_reservation(cfg, i, h35=cfg.h35, run=run, now=world.now()))
        st["observations"].append(observe(world, cfg, lanes, held=held))
        why = st["observations"][-1]["reasons"]
    except FileExistsError:
        why = ["another holder created a lane's reservation at the O_EXCL (a lost race; its file is untouched)"]
    except Unreadable as err:
        why = [f"unreadable: {err}"]
    if not why:
        st.update(phase="reserved", reservations=held)
        event(st, world.now(), "reserved", records=[h["path"] for h in held])
        write_json_atomic(state_path(orch), st)
        return 0, f"RESERVED {orch}: " + "; ".join(f"GPU {h['index']} {h['path']} sha256 {h['sha256']} inode "
                                                   f"{h['inode']}" for h in held) + "; state reserved"
    kept = [dict(h, unreleased=r) for h in held for r in [release_reservation(h)] if r]
    st.update(phase="reserve_failed" if kept else "reserve_refused", reservations=kept,
              failure={"reasons": why, "time": iso(world.now())},
              released={"by": "reserve", "records": [h["path"] for h in held if h["path"] not in
                                                     [k["path"] for k in kept]], "time": iso(world.now())})
    event(st, world.now(), st["phase"], reasons=why)
    write_json_atomic(state_path(orch), st)
    return 1, (f"RESERVE REFUSED after the acquisition ({st['phase']}; our own records released, "
               f"{len(kept)} retained): " + "; ".join(why))


# --------------------------------------------------------------------------- #
# part 4a: the exact runner command and the exact parser of its DRY_RUN transcript (§15 item 7; S4, S6)
# --------------------------------------------------------------------------- #
TS_RE = r"[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}"
# the pinned flac env's one warning line (printed by the runner's own collector calls; the real H35 transcript has two)
BENIGN = ("/home/yixunhu/miniconda3/envs/flac/lib/python3.10/site-packages/lightning_fabric/__init__.py:40: "
          "pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The "
          "pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin "
          "to Setuptools<81.")
PREFLIGHT = ("exp35 no-flash overlay: flash_attn disabled, attention backend = torch SDPA\n"
             "flash_attn not installed, disabling Flash Attention\n"
             "EXP35_NO_FLASH_LAUNCHER attention_backend=sdpa "
             "overlay_sha256=0d0e52ca693e236a2e0a44cd61f61539fdf995cff5132d1f621181a0f3e7c023 "
             "launcher_sha256=bbd2847d3a50685420750d066195c8656a1509b4e905883886c0568d244043de target=--preflight\n")
LAUNCHER_LINE = "  (exp_35: spawned through overlays/no_flash/exp35_no_flash_launch.py, PYTHONPATH={}/overlays/no_flash)"


def runner_command(cfg, lanes, dry):
    """§15 item 7: (argv, env, cwd) -- the 68 cells' blocks, the frozen lanes, the CANON manifest; no --no-dump-blocks,
    no sharing variable; the phase-(i) launch environment + EXPECT_HEAD (+ DRY_RUN=1); cwd = the run worktree."""
    argv = ["bash", RUNNER, "--blocks", ",".join(BLOCKS), "--lanes", lanes_text(lanes), "--canon-ckpt-manifest",
            CANON_MANIFEST]
    return argv, dict(LAUNCH_ENV, EXPECT_HEAD=cfg.h35, **({"DRY_RUN": "1"} if dry else {})), cfg.run


def _expected_lines(cfg, m, rows, lanes):
    """The runner's DRY_RUN transcript at H35 for ``rows``: its 11 preamble lines as patterns, then per arm the one
    checkpoint line and per cell exactly 'would run', its argv and the launcher line, then the summary."""
    x, e, n = re.escape(X25), re.escape, len(rows)
    want = [("re", p) for p in (
        rf"=== exp_25 depth_canon eval campaign -- (?P<ts>{TS_RE}) -- HEAD {cfg.h35} -- dry_run=1 -- harness=false ===",
        rf"evaluator pin OK: {cfg.h35}; evaluator tree digest {cfg.digest}",
        rf"no-flash overlay \+ launcher OK: 0d0e52ca693e / bbd2847d3a50 \({n} exp_35 cell\(s\)\)", r"cells manifest OK",
        rf"plan: {n} cell\(s\) -> {x}/depth_canon_(?P<ts>{TS_RE})_eval_plan\.json",
        rf"plan: {n} cell\(s\); dump root {e(DUMP_ROOT)}; farm {e(FARM)}; lanes {e(lanes_text(lanes))}",
        r"configs OK: every model / dataset config of the selection matches its manifest pin",
        rf"BACKEND OK {x}/depth_canon_(?P<ts>{TS_RE})_no_flash_preflight\.log",
        rf"no-flash pre-flight OK: {n} exp_35 cell\(s\) spawn through overlays/no_flash/exp35_no_flash_launch\.py "
        rf"\(log {x}/depth_canon_(?P<ts>{TS_RE})_no_flash_preflight\.log\)",
        rf"dump root OK under DRY_RUN: {e(DUMP_ROOT)} \(exists\)",
        r"checkpoint cache space: \d+ checkpoint\(s\) to stage = \d+ B \+ \d+ B margin = \d+ B needed; \d+ B "
        r"available on \S+")]
    seen = set()
    for r in rows:
        p = plan_entry(m, r, {40000: CANON_CKPT})
        if r["arm"] not in seen:
            seen.add(r["arm"])
            want.append(("eq", f"checkpoint OK (DRY_RUN, not staged): {p['target']} ({p['sha256']})"))
        want += [("eq", f"CELL {r['cell']} | {r['block']} | {r['arm']} | would run"), ("argv", r["cell"], p["argv"]),
                 ("eq", LAUNCHER_LINE.format(os.path.realpath(cfg.run)))]
    return want + [("eq", f"=== DRY_RUN summary: {n} cell(s) planned; nothing evaluated, linked, created or moved ===")]


def _json(raw):
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return None


def parse_dry_run(text, *, cfg, m, rows, lanes):
    """S6 'exact dry-run parsing': (reasons, info). Every line must be the next expected one -- the pinned env's one
    warning line only between preamble lines -- and none may be a SKIP, quarantine, E2-FAIL or refusal line. info: the
    run's timestamp, the cell ids and the argv records in order, the warning count, the line count."""
    lines, e2 = text.splitlines(), {c["cell"] for c in m["cells"] if c["block"] == "E2-FAIL"}
    out = [f"the transcript holds {k}: {v[0][:160]!r}" for k, v in (
        ("a SKIP line", [ln for ln in lines if "| SKIP" in ln]),
        ("a quarantine line", [ln for ln in lines if "quarantine" in ln.lower()]),
        ("an E2-FAIL cell", [ln for ln in lines if "E2-FAIL" in ln or (ln.startswith(("CELL ", "DRYRUN "))
                                                                       and ln.split()[1:2] and ln.split()[1] in e2)]),
        ("a refusal", [ln for ln in lines if "abort" in ln or "REFUSED" in ln or "FAILED" in ln])) if v]
    want, k, ts, ids, argv, benign, broke = _expected_lines(cfg, m, rows, lanes), 0, set(), [], [], 0, False
    for i, ln in enumerate(lines, 1):
        if ln == BENIGN and k < 11:
            benign += 1
            continue
        if k == len(want):
            out, broke = out + [f"line {i} follows the summary: {ln[:160]!r}"], True
            break
        w = want[k]
        if w[0] == "re":
            hit = re.fullmatch(w[1], ln)
            ts.update([hit["ts"]] if hit is not None and "ts" in hit.re.groupindex else [])
        elif w[0] == "eq":
            hit = ln == w[1]
            ids += [w[1].split()[1]] if hit and ln.startswith("CELL ") else []
        else:
            pre = f"DRYRUN {w[1]} argv: "
            hit = ln.startswith(pre) and _json(ln[len(pre):]) == w[2]
            argv += [w[2]] if hit else []
        if not hit:
            want_text = f"DRYRUN {w[1]} argv {w[2]}" if w[0] == "argv" else w[1]
            out, broke = out + [f"line {i} {ln[:200]!r} is not the expected {want_text[:300]!r}"], True
            break
        k += 1
    out += [] if broke or k == len(want) else [f"the transcript ends after {k} of its {len(want)} expected lines"]
    out += [] if len(ts) == 1 else [f"the transcript names {len(ts)} run timestamps {sorted(ts)}"]
    out += [] if len(ids) == len(argv) == len(rows) else [f"{len(ids)} cell ids, {len(argv)} argv, not {len(rows)}"]
    return out, {"ts": min(ts) if len(ts) == 1 else None, "ids": ids, "argv": argv, "benign_lines": benign,
                 "lines": len(lines)}


# --------------------------------------------------------------------------- #
# part 4b: the dry-run transaction (§15 item 7; S6 "record the dry-run environment, cwd, argv, outer / internal log
# hashes, rc, exact planned set, and all 68 argv records")
# --------------------------------------------------------------------------- #
def open_mode(cfg, kit_commit, h35, orch, phases):
    """dry-run / launch / release: the self-pin, the H35 argument, an orchestration of THIS run worktree, the kit lock,
    its state in ``phases`` and reserved with this very kit commit. ((lock fd, state), []) or (None, reasons)."""
    why = kit_evidence(cfg, kit_commit)[0] + (
        [] if h35 == cfg.h35 else [f"the given H35 {h35} is not the registered {cfg.h35}"])
    base = os.path.realpath(os.path.join(cfg.run, X35))
    why += [] if os.path.dirname(os.path.realpath(orch)) == base else [f"{orch} is not an orchestration of {base}"]
    lock = None if why else kit_lock(cfg)
    if lock is None:
        return None, why or [f"the kit lock {KIT_LOCK} is held (a kit mode, our wrapper or a runner child lives)"]
    try:
        st = load_state(orch)
        why = ([] if st["phase"] in phases else [f"the state is {st['phase']}, not {' or '.join(phases)}"]) + (
            [] if st["kit"]["commit"] == kit_commit else [
                f"the orchestration was reserved with the kit {st['kit']['commit']}"])
    except ValueError as err:
        why = [str(err)]
    if why:
        os.close(lock)
        return None, why
    return (lock, st), []


def recheck(world, cfg, st, orch):
    """Every precondition again, the evidence unchanged since the reservation, the held records ours: (reasons, obs)."""
    lanes = tuple(x["index"] for x in st["lanes"])
    why, runs = prior(cfg, exclude=orch)
    bad, ev = evidence(cfg, lanes=lanes, report=st["evidence"]["report"]["path"],
                       report_sha256=st["evidence"]["report"]["sha256"], prior_runs=runs)
    o = observe(world, cfg, lanes, held=st["reservations"])
    return why + bad + ([] if ev == st["evidence"] else ["the evidence changed since the reservation"]) + o["reasons"], o


def dry_run(world, cfg, *, kit_commit, h35, orch):
    """§15 item 7: from 'reserved' only -- the recheck, the EXACT runner command under DRY_RUN=1, its transcript parsed
    exactly and its own artifacts checked; then 'dry_run_passed', or 'dry_run_failed' (retained). (rc, message)."""
    opened, why = open_mode(cfg, kit_commit, h35, orch, ("reserved",))
    if opened is None:
        return 1, "DRY RUN REFUSED (nothing changed): " + "; ".join(why)
    lock, st = opened
    try:
        argv, env, cwd = runner_command(cfg, tuple(x["index"] for x in st["lanes"]), dry=True)
        rec = {"argv": argv, "env": env, "cwd": cwd, "started": world.now()}
        try:
            why, rec["before"] = recheck(world, cfg, st, orch)
            if not why:
                p = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT)
                raw = p.communicate()[0]                                         # no timeout: never killed
                path = os.path.join(orch, "dry_run_runner.log")
                rec.update(rc=p.returncode, ended=world.now(), outer_log={
                    "path": path, "sha256": write_exclusive(path, raw).sha256, "bytes": len(raw)})
                why = [f"the dry run exited rc {p.returncode}"] if p.returncode else []
                why += artifacts(cfg, raw, st, rec)
                more, rec["after"] = recheck(world, cfg, st, orch)
                why += more
        except Unreadable as err:
            why = [f"unreadable: {err}"]
        except OSError as err:                           # the runner could not be spawned, or its log written
            why = [f"the dry run could not be spawned or its outer log written ({err})"]
        st.update(phase="dry_run_failed" if why else "dry_run_passed", dry_run=dict(rec, reasons=why),
                  failure={"reasons": why, "time": iso(world.now())} if why else None)
        event(st, world.now(), st["phase"], reasons=why[:5])
        write_json_atomic(state_path(orch), st)
    finally:
        os.close(lock)
    if why:
        return 1, "DRY RUN FAILED (retained; release, then reserve anew): " + "; ".join(why)
    return 0, (f"DRY RUN PASSED: {rec['n_ids']} cells, {len(rec['argv_records'])} argv, the runner's log, plan and "
               "pre-flight exact")


def artifacts(cfg, raw, st, rec):
    """The transcript parsed exactly; the runner's own log = it, byte for byte; its plan JSON = the re-derived plan;
    its pre-flight log = the pinned three lines. Fills ``rec`` (paths + sha256, the planned set, the 68 argv)."""
    cells = snapshot(os.path.join(cfg.run, CELLS))
    _, m, rows = selection(cells.raw if cells else b"")
    if m is None:
        return ["the cells manifest is unreadable"]
    lanes = tuple(x["index"] for x in st["lanes"])
    why, info = parse_dry_run(raw.decode(errors="replace"), cfg=cfg, m=m, rows=rows, lanes=lanes)
    rec.update(ts=info["ts"], n_ids=len(info["ids"]), ids=info["ids"], ids_sha256=names_sha256(info["ids"]),
               argv_records=info["argv"], benign_lines=info["benign_lines"])
    want = {"internal_log": raw, "plan": expected_plan(m, rows, run=os.path.realpath(cfg.run), digest=cfg.digest),
            "preflight": PREFLIGHT.encode()}
    for key, name in (("internal_log", "eval_run.log"), ("plan", "eval_plan.json"),
                      ("preflight", "no_flash_preflight.log")):
        path = os.path.join(cfg.run, X25, f"depth_canon_{info['ts']}_{name}")
        s = snapshot(path) if info["ts"] else None
        rec[key] = {"path": path if info["ts"] else None, "sha256": s.sha256 if s else None}
        ok = s is not None and (_json(s.raw) if key == "plan" else s.raw) == want[key]
        why += [] if ok else [f"the runner's {key.replace('_', ' ')} {path if info['ts'] else '(no timestamp)'} is not "
                              "the exact expected bytes"]
    return why


# --------------------------------------------------------------------------- #
# part 5a: the detached wrapper (S6 "launch through a detached wrapper / supervisor that durably records the runner's
# exact rc; record complete identities for the detached wrapper and actual runner")
# --------------------------------------------------------------------------- #
SETSID, NOHUP = "/usr/bin/setsid", "/usr/bin/nohup"


def launch_spec(cfg, st, orch, lock_fd):
    """What 'launch_pending' commits to BEFORE any spawn: the runner's exact command and its outer log, and the
    detached wrapper's (setsid + nohup, the launch environment, cwd = the run worktree, the kit lock's fd)."""
    argv, env, cwd = runner_command(cfg, tuple(x["index"] for x in st["lanes"]), dry=False)
    wargv = [SETSID, NOHUP, cfg.py, "-B", cfg.helper, "wrap", "--orch", orch, "--lock-fd", str(lock_fd)]
    return {"runner": {"argv": argv, "env": env, "cwd": cwd, "log": os.path.join(orch, "runner.log")},
            "wrapper": {"argv": wargv, "env": dict(LAUNCH_ENV), "cwd": cfg.run, "log": os.path.join(orch, "wrapper.log")}}


def environ_of(pid, proc):
    """The environment a live process was STARTED with (/proc/<pid>/environ); ValueError when unreadable."""
    try:
        with open(os.path.join(proc, str(pid), "environ"), "rb") as f:
            raw = f.read()
    except OSError as err:
        raise ValueError(f"process {pid}: its environment is unreadable ({err})") from err
    return dict(x.decode(errors="replace").split("=", 1) for x in raw.split(b"\0") if b"=" in x)


def _proof(ident, env, want, label):
    return dict({k: ident[k] for k in ID_KEYS}, environ_sha256=sha256_bytes(json.dumps(env, sort_keys=True).encode()),
                environ_keys=sorted(env), sharing_absent=not set(SHARING) & set(env), **{label: env == want})


def identify(p, spec, cfg):
    """(the runner's complete identity + the proof of its /proc environment, None) once /proc shows the spawned command
    -- re-read for up to ``identify_s`` s, as the exec may lag the fork -- else (None, why): never a pid alone."""
    expect = sha256_bytes(b"".join(a.encode() + b"\0" for a in spec["argv"]))
    deadline, last = time.monotonic() + cfg.identify_s, ""
    while True:
        try:
            ident = proc_identity(p.pid, cfg.proc)
            if ident["cmdline_sha256"] == expect:
                return _proof(ident, environ_of(p.pid, cfg.proc), spec["env"], "environ_is_the_spec"), None
            last = "its /proc cmdline is not the spawned command"
        except ValueError as err:
            last = str(err)
        if p.poll() is not None:
            return None, f"the runner {p.pid} ended (rc {p.returncode}) before its identity was captured ({last})"
        if time.monotonic() > deadline:
            return None, f"the runner {p.pid} did not show its command within {cfg.identify_s} s ({last})"
        time.sleep(0.01)


def spawn(argv, *, cwd, env, log, lock_fd):
    """One child: its exact argv, environment and cwd, stdin /dev/null, stdout + stderr into ``log`` (created O_EXCL),
    the kit lock's fd passed on (so every descendant holds the lock while it lives), every other fd closed."""
    fd = os.open(log, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        return subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=fd, stderr=subprocess.STDOUT,
                                pass_fds=(lock_fd,), close_fds=True)
    finally:
        os.close(fd)


def wait_for(p):
    """The runner's exit code, waited for: never a timeout, never a signal (S5)."""
    return p.wait()


def runner_logs(cfg, spec):
    """The outer log (the wrapper's capture of the runner) and the runner's own tee'd log its first line names."""
    outer = snapshot(spec["log"])
    first = outer.raw.split(b"\n", 1)[0].decode(errors="replace") if outer else ""
    m = re.fullmatch(rf"=== exp_25 depth_canon eval campaign -- ({TS_RE}) -- .*", first)
    path = os.path.join(cfg.run, X25, f"depth_canon_{m[1]}_eval_run.log") if m else None
    inner = snapshot(path) if path else None
    return {"outer": {"path": spec["log"], "sha256": outer.sha256 if outer else None, "bytes": len(outer.raw) if outer else 0},
            "internal": {"path": path, "sha256": inner.sha256 if inner else None}}


def wrap(cfg, *, orch, lock_fd, world=None):
    """The detached wrapper: it must hold the kit lock through ``lock_fd`` and find a fresh 'launch_pending' (else it
    refuses and writes nothing); its own identity and environment ('wrapper_refused' unless exactly the launch
    environment); 'spawning' -> the runner, the lock fd passed on -> 'running' with the runner's identity -> waits,
    never signalling -> 'exited' with the EXACT rc (then the logs). 0 once the rc is durable, 3 when refused."""
    world = world or World(cfg.nvidia_smi)
    try:
        fst, lst = os.fstat(lock_fd), os.stat(os.path.join(cfg.run, KIT_LOCK))
        if (fst.st_dev, fst.st_ino) != (lst.st_dev, lst.st_ino):
            raise OSError(f"fd {lock_fd} is not the kit lock {KIT_LOCK}")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        st = load_state(orch)
        if st["phase"] != "launch_pending" or st["wrapper"] is not None:
            raise ValueError(f"the state is {st['phase']} (wrapper {st['wrapper']!r:.60}), not a fresh launch_pending")
    except (OSError, ValueError) as err:
        print(f"WRAPPER REFUSED (nothing written): {err}", flush=True)
        return 3

    def commit(phase, **kw):
        st.update(phase=phase, **kw)
        event(st, world.now(), phase, **({"reasons": kw["failure"]["reasons"]} if kw.get("failure") else {}))
        write_json_atomic(state_path(orch), st)
        print(f"[{iso(world.now())}] {phase}", flush=True)
    try:
        st["wrapper"] = _proof(proc_identity(os.getpid(), cfg.proc), environ_of(os.getpid(), cfg.proc), LAUNCH_ENV,
                               "environ_is_the_launch_env")
        why = [] if st["wrapper"]["environ_is_the_launch_env"] else [
            f"the wrapper's environment {st['wrapper']['environ_keys']} is not the launch environment"]
    except ValueError as err:
        why = [f"the wrapper's own identity: {err}"]
    if why:
        commit("wrapper_refused", failure={"reasons": why, "time": iso(world.now())})
        return 3
    spec = st["launch"]["runner"]
    commit("spawning")
    try:
        p = spawn(spec["argv"], cwd=spec["cwd"], env=spec["env"], log=spec["log"], lock_fd=lock_fd)
    except OSError as err:
        commit("runner_spawn_failed", failure={"reasons": [f"the runner could not be spawned ({err})"],
                                               "time": iso(world.now())})
        return 3
    ident, why = identify(p, spec, cfg)
    commit("running", runner=ident or {"pid": p.pid, "unidentified": why})
    rc = wait_for(p)
    commit("exited", exit={"rc": rc, "signal": -rc if rc < 0 else None, "ended": world.now(), "time": iso(world.now()),
                           "logs": None})
    deadline = time.monotonic() + 30                     # its tee holds the runner lock until the logs are complete
    try:
        while lock_status(os.path.join(cfg.run, RUNNER_LOCK)) == "held" and time.monotonic() < deadline:
            time.sleep(0.2)
    except Unreadable:
        pass
    st["exit"]["logs"] = runner_logs(cfg, spec)
    write_json_atomic(state_path(orch), st)
    return 0



# --------------------------------------------------------------------------- #
# part 5b: the launch (S6 "before launch, reauthenticate every reservation, recheck the cards and runner lock, and
# require the matching successful dry run") and its read-only confirmation
# --------------------------------------------------------------------------- #
def launch(world, cfg, *, kit_commit, h35, orch):
    """§15 item 7: from 'dry_run_passed' only -- the dry run of THIS exact command, fresh; the recheck; 'launch_pending'
    BEFORE any spawn; the last look; then the wrapper DETACHED with the kit lock's fd, and its identity confirmed.
    A refusal is retained ('launch_refused'). Never waits for, signals or controls the runner. (rc, message)."""
    opened, why = open_mode(cfg, kit_commit, h35, orch, ("dry_run_passed",))
    if opened is None:
        return 1, "LAUNCH REFUSED (nothing changed): " + "; ".join(why)
    lock, st = opened
    try:
        spec, d, now = launch_spec(cfg, st, orch, lock), st["dry_run"], world.now()
        r = spec["runner"]
        why = [] if (d["argv"], d["env"], d["cwd"]) == (r["argv"], dict(r["env"], DRY_RUN="1"), r["cwd"]) else [
            "the passed dry run is not this exact command"]
        why += [] if 0 <= now - d["ended"] <= cfg.dry_max_age_s else [
            f"the dry run ended {now - d['ended']:.0f} s ago: older than {cfg.dry_max_age_s} s, or in the future"]
        try:
            more, before = recheck(world, cfg, st, orch)
        except Unreadable as err:
            more, before = [f"unreadable: {err}"], None
        if not why + more:
            st.update(phase="launch_pending", launch=dict(spec, started=now, before=before))
            event(st, now, "launch_pending: the detached wrapper is spawned next")
            write_json_atomic(state_path(orch), st)
            try:
                last = observe(world, cfg, tuple(x["index"] for x in st["lanes"]), held=st["reservations"])["reasons"]
            except Unreadable as err:
                last = [f"unreadable: {err}"]
            more = [f"at the last look: {x}" for x in last]
        if why + more:
            st.update(phase="launch_refused", failure={"reasons": why + more, "time": iso(world.now())})
            event(st, world.now(), "launch_refused", reasons=(why + more)[:5])
            write_json_atomic(state_path(orch), st)
            return 1, "LAUNCH REFUSED (retained; release, then reserve anew): " + "; ".join(why + more)
        w = spec["wrapper"]
        try:
            p = spawn(w["argv"], cwd=w["cwd"], env=w["env"], log=w["log"], lock_fd=lock)
        except OSError as err:
            st.update(phase="wrapper_spawn_failed", failure={"reasons": [f"the wrapper could not be spawned ({err})"],
                                                             "time": iso(world.now())})
            write_json_atomic(state_path(orch), st)
            return 1, f"LAUNCH FAILED (retained): the wrapper could not be spawned ({err})"
    finally:
        os.close(lock)                                  # the detached wrapper keeps the lock through its inherited fd
    return confirm(cfg, orch, p.pid)


def confirm(cfg, orch, pid):
    """Read-only, after the detachment: within ``confirm_s`` the state must name the live ``pid`` as the wrapper with
    its complete identity and the runner alive -- or report what the wrapper recorded instead. Never signals."""
    deadline, st = time.monotonic() + cfg.confirm_s, None
    while st is None or st["phase"] not in ("running", "exited", "wrapper_refused", "runner_spawn_failed"):
        if time.monotonic() > deadline:
            return 1, (f"LAUNCH NOT CONFIRMED within {cfg.confirm_s} s (phase {st['phase'] if st else None}); the "
                       f"wrapper pid {pid} is never signalled: inspect {orch} with status")
        time.sleep(0.1)
        try:
            st = load_state(orch)
        except ValueError:
            st = None
    if st["phase"] != "running":
        return 1, f"LAUNCHED, BUT {st['phase']}: {st['exit'] or st['failure']} -- the Planner reviews; no relaunch"
    w, r = st["wrapper"], st["runner"]
    try:
        ok = w["pid"] == pid and identity_status(w, cfg.proc) == "alive" and identity_status(r, cfg.proc) == "alive"
    except ValueError as err:                            # an unidentified runner, or an unreadable identity
        ok, w = False, dict(w, why=str(err))
    if not ok:
        return 1, f"LAUNCH NOT CONFIRMED: the recorded wrapper {w} / runner {r} are not both alive as pid {pid}'s run"
    return 0, (f"LAUNCHED: wrapper pid {pid} (start time {w['start_time']}), runner pid {r['pid']} (start time "
               f"{r['start_time']}); state running; {orch}")


# --------------------------------------------------------------------------- #
# part 6a: the authenticated release and the read-only status (S6)
# --------------------------------------------------------------------------- #
RELEASABLE = ("reserved", "dry_run_passed", "dry_run_failed", "launch_refused", "wrapper_spawn_failed", "wrapper_refused",
              "runner_spawn_failed", "exited")       # durable exit / refusal states, or a reservation never launched
MEANING = {0: "every selected cell complete and verified", 1: "a cell failed (later cells may still have run)",
           2: "a gate, backend or lane refusal, or another abort", 4: "PIN DRIFT: HEAD or the relevant tree changed"}


def release(world, cfg, *, kit_commit, h35, orch):
    """S6: from a durable exit / refusal state only (or a reservation never launched: withdrawn), under the kit lock
    (held by no wrapper, runner or descendant -- they all inherited it); no recorded wrapper / runner alive (complete
    identities), the runner lock unheld, no exp_35 process alive; ALL records authenticated, then each one again right
    before its unlink and the directory fsynced. 'released', or 'release_failed' (retained). (rc, message)."""
    opened, why = open_mode(cfg, kit_commit, h35, orch, RELEASABLE)
    if opened is None:
        return 1, "RELEASE REFUSED (nothing unlinked): " + "; ".join(why)
    lock, st = opened
    try:
        try:
            why = [f"the recorded {k} {st[k]['pid']} may be alive ({s})" for k in ("wrapper", "runner")
                   if st[k] and "start_time" in st[k] for s in [identity_status(st[k], cfg.proc)] if s == "alive"]
            why += ["the runner was never identified and no exit is recorded"] if (
                st["runner"] and "start_time" not in st["runner"] and st["exit"] is None) else []
            why += ["the runner lock is held (a runner or an evaluator descendant lives)"] if lock_status(
                os.path.join(cfg.run, RUNNER_LOCK)) == "held" else []
            why += [f"pid {p}: {w}" for p, w in live_procs(cfg)] + [x for h in st["reservations"] for x in held_reasons(h)]
        except Unreadable as err:
            why = [f"unreadable: {err}"]
        if why:
            return 1, "RELEASE REFUSED (nothing unlinked): " + "; ".join(why)
        done, kept, prior_phase = [], [], st["phase"]
        for h in st["reservations"]:
            r = release_reservation(h)                    # authenticated again right before its unlink
            kept += [dict(h, unreleased=r)] if r else []
            done += [] if r else [h["path"]]
        st.update(phase="release_failed" if kept else "released", reservations=kept, released={
            "by": "planner", "prior_phase": prior_phase, "records": done, "time": iso(world.now())})
        event(st, world.now(), st["phase"], records=done)
        write_json_atomic(state_path(orch), st)
    finally:
        os.close(lock)
    if kept:
        return 1, f"RELEASE FAILED (retained): {len(kept)} record(s) not ours at the unlink: {kept}"
    return 0, f"RELEASED {orch}: " + ", ".join(done) + f" (prior phase {prior_phase})"


def _status_of(cfg, orch, m, rows):
    try:
        st = load_state(orch)
    except ValueError as err:
        return [f"{orch}: {err}"]
    out = [f"{orch}: phase {st['phase']}; lanes {[x['index'] for x in st['lanes']]}; failure {st['failure']}"]
    out += [f"  record {h['path']}: " + ("; ".join(held_reasons(h)) or "our exact record") for h in st["reservations"]]
    for k in ("wrapper", "runner"):
        ident = st[k]
        try:
            s = "none" if not ident else ident.get("unidentified") and "unidentified" or identity_status(ident, cfg.proc)
        except ValueError as err:
            s = f"unreadable ({err})"
        out.append(f"  {k}: {s}" + (f" (pid {ident['pid']}, start time {ident.get('start_time')})" if ident else ""))
    log = snapshot(((st["launch"] or {}).get("runner") or {}).get("log") or "")
    if log is not None:
        lines = log.raw.decode(errors="replace").splitlines()
        out.append(f"  runner log: {len(log.raw)} B, sha256 {log.sha256}; DONE {sum(ln.startswith('  DONE ') for ln in lines)}"
                   f", FAILED {sum(ln.startswith('  !! FAILED ') for ln in lines)}; last line {(lines or [''])[-1]!r}")
    plan = expected_plan(m, rows, run=os.path.realpath(cfg.run), digest=cfg.digest)["cells"] if m else []
    got = sum(os.path.isfile(os.path.join(cfg.run, e["receipt"])) for e in plan)
    out.append(f"  receipts: {got}/{len(plan)} of the exact cells")
    rc = (st["exit"] or {}).get("rc")
    return out + ([f"  exit: rc {rc} ({MEANING.get(rc, 'an unexpected exit code')})"] if st["exit"] else [])


def status(world, cfg, *, orch=None, kit_commit=None):
    """S6: READ-ONLY -- no lock taken or created, no state written, no git index refreshed: the kit pin, the
    preconditions, the locks, the cards, every orchestration, and for ``orch`` (else the newest) its records,
    identities, runner log, receipts of the exact 68 cells and final rc. (0, report)."""
    out = [f"exp_35 phase (ii) status at {iso(world.now())} (read-only)"]
    if kit_commit:
        why = kit_evidence(cfg, kit_commit)[0]
        out.append("kit: " + ("; ".join(why) or f"the reviewed commit {kit_commit}, clean"))
    cells, st_i, sup = (snapshot(os.path.join(cfg.run, rel)) for rel in (CELLS, SUP_STATE, SUP_FILE))
    why, m, rows = selection(cells.raw if cells else b"")
    out.append(f"cells: {len(rows)} selected, names sha256 {names_sha256([r['cell'] for r in rows])}; " + ("; ".join(why) or "OK"))
    doc = _json(st_i.raw) if st_i else None
    why = phase_i_reasons(doc, h35=cfg.h35, digest=cfg.digest, supervisor_sha256=sup.sha256 if sup else None)
    out.append(f"phase (i): state sha256 {st_i.sha256 if st_i else None}; " + ("; ".join(why) or "complete"))
    out.append("run worktree: " + ("; ".join(worktree_reasons(cfg, m) if m else ["no manifest"]) or "at H35, clean, digest OK"))
    out.append("his files: " + ("; ".join(external_reasons(cfg)) or "at their pins; his s43 training ended"))
    try:
        locks = {k: lock_status(os.path.join(cfg.run, rel)) for k, rel in (("supervisor", SUP_LOCK),
                                                                           ("runner", RUNNER_LOCK), ("kit", KIT_LOCK))}
        out.append("locks: " + "; ".join(f"{k} lock {v}" for k, v in locks.items()))
        apps = compute_apps(world.nvsmi("apps"))
        out += [f"GPU {i} {u}: compute pids {sorted(p for p, x in apps if x == u)}; reservation "
                f"{reservation_status(cfg, u)}" for i, u in CARDS.items()]
        out.append(f"exp_35 processes: {live_procs(cfg) or 'none'}")
    except (ValueError, Unreadable) as err:
        out.append(f"unreadable: {err}")
    try:
        orchs = orchestrations(cfg)
    except Unreadable as err:
        orchs, out = [], out + [f"orchestrations: unreadable: {err}"]
    else:
        out.append("orchestrations: " + (", ".join(f"{os.path.basename(d)} ({s['phase'] if isinstance(s, dict) else s})"
                                                   for d, s in orchs) or "none"))
    target = orch or (orchs[-1][0] if orchs else None)
    return 0, "\n".join(out + (_status_of(cfg, target, m, rows) if target else []))

def main(argv=None):
    """The helper's CLI (exp35_phase_ii.sh calls it under env -i; ``wrap`` is started by ``launch`` only): 0 done,
    1 refused or failed, 2 bad arguments, 3 a refused wrapper."""
    ap = argparse.ArgumentParser(description="exp_35 phase (ii): the launch kit's helper (run by exp35_phase_ii.sh)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("reserve", "dry-run", "launch", "release"):
        p = sub.add_parser(name)
        p.add_argument("--kit-commit", required=True)
        p.add_argument("--h35", required=True)
        for opt in (("--lanes", "--report", "--report-sha256") if name == "reserve" else ("--orch",)):
            p.add_argument(opt, required=True)
    s = sub.add_parser("status")
    s.add_argument("--kit-commit", default="")
    s.add_argument("--orch", default=None)
    w = sub.add_parser("wrap", help="the detached wrapper (started by launch only)")
    w.add_argument("--orch", required=True)
    w.add_argument("--lock-fd", type=int, required=True)
    args = ap.parse_args(argv)
    cfg = registered_config()
    world, orch = World(cfg.nvidia_smi), os.path.abspath(args.orch) if getattr(args, "orch", None) else None
    if args.cmd == "wrap":
        return wrap(cfg, orch=orch, lock_fd=args.lock_fd)
    if args.cmd == "status":
        rc, msg = status(world, cfg, orch=orch, kit_commit=args.kit_commit or None)
    elif args.cmd == "reserve":
        try:
            lanes = parse_lanes(args.lanes)
        except ValueError as err:
            print(f"RESERVE REFUSED (bad arguments): {err}", flush=True)
            return 2
        rc, msg = reserve(world, cfg, kit_commit=args.kit_commit, h35=args.h35, lanes=lanes, report=args.report,
                          report_sha256=args.report_sha256)
    else:
        rc, msg = {"dry-run": dry_run, "launch": launch, "release": release}[args.cmd](
            world, cfg, kit_commit=args.kit_commit, h35=args.h35, orch=orch)
    print(msg, flush=True)
    return rc

if __name__ == "__main__":
    sys.exit(main())
