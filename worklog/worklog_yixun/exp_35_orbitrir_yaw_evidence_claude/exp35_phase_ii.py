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

Never: moves or modifies H35, installs anything, selects E2-FAIL, suppresses dumps, enables GPU sharing,
overwrites / adopts / removes another holder's reservation, signals / kills / attaches to any process,
relaunches, clears a failure, changes a lane or starts exp_22.

Written by the exp_35 phase-(ii) kit Coder (Claude Opus 5.5, max effort).
"""
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
GAP_S, CONFIRM_S, IDENTIFY_S, DRY_TIMEOUT_S, DRY_MAX_AGE_S = 60, 60, 5.0, 1800, 10800
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
HEX64 = re.compile(r"[0-9a-f]{64}")
Config = collections.namedtuple("Config", "run h35 digest reservations exp30_pins queue_log nvidia_smi proc scan_proc "
                                "py gap_s confirm_s identify_s dry_timeout_s dry_max_age_s kit_dir helper")


def registered_config():
    return Config(RUN, H35, CAMPAIGN_DIGEST, RESERVATIONS, {f"{EXP30}/{k}": v for k, v in EXP30_PINS.items()},
                  f"{EXP30}/{QUEUE_LOG}", NVIDIA_SMI, PROC, SCAN_PROC, PY, GAP_S, CONFIRM_S, IDENTIFY_S,
                  DRY_TIMEOUT_S, DRY_MAX_AGE_S, HERE, os.path.abspath(__file__))


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
    """A read-only git query (no optional locks: a status never rewrites the index); ValueError when it fails."""
    r = subprocess.run(["git", "--no-optional-locks", "-C", repo, *args], capture_output=True, timeout=600)
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
            r = subprocess.run(argv, capture_output=True, text=True, timeout=120)
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
