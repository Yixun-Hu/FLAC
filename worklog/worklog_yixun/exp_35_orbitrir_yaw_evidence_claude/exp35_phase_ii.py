#!/usr/bin/env python3
"""exp_35 phase (ii): the launch kit's helper (plan_orbitrir_yaw_evidence.md §15 and its r1 clarifications (a)-(c);
Codex plan §15 r1 "S6 -- binding kit and test contract"). ``exp35_phase_ii.sh`` runs it in the phase-(i) launch
environment for the Planner's five modes -- reserve | dry-run | launch | status | release -- and ``launch`` starts
it once more, detached, as the wrapper that runs the H35 runner and durably records its exact exit code.

Part 1a: the registered pins, the durable writers and the pure checks -- the 68-cell selection of the cells manifest
at H35 and its name digest, the CANON re-pin, the runner's plan JSON re-derived row by row, the lane list.

Never: moves or modifies H35, installs anything, selects E2-FAIL, suppresses dumps, enables GPU sharing,
overwrites / adopts / removes another holder's reservation, signals / kills / attaches to any process,
relaunches, clears a failure, changes a lane or starts exp_22.

Written by the exp_35 phase-(ii) kit Coder (Claude Opus 5.5, max effort).
"""
import collections
import datetime
import hashlib
import json
import os
import re
import stat
import sys
import tempfile

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
