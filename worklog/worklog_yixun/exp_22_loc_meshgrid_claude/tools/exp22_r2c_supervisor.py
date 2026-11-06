#!/usr/bin/env python3
"""exp_22 R2-C pre-flight: the reservation, the durable state and the detached supervisor (plan_loc_meshgrid_r2_bf_yaw.md
§9 A7 r2 (iv) (v)(2), r3 (b) (e) (f) (g), r4, r5).

  reserve --orch D --pin2 P --card C --index I --seal F --seal-sha256 X   the launcher, holding the exp22 lock: the card
        re-checked, the O_EXCL reservation (holder exp22, PIN2, the card, epoch 1, created, run; its exact bytes' sha256
        and device / inode kept), the card re-checked AFTER the acquisition (any change: only our own authenticated
        record is released), then the first durable state ``reserved`` (O_EXCL)
  run --orch D --lock-fd N        the detached supervisor (holds the inherited exp22 lock for its whole life)
  confirm --orch D --pid P        the launcher: the supervisor's recorded identity is the live pid's, the lock held
  active [--except-orch D]        exit 0 iff no exp22 run is active or unresolved
  release --orch D                the Planner, by hand, after a stop or a crash: only with no live supervisor or line
        child (pid + start time, never a pid alone) and only our exact record (bytes + inode) is unlinked

The state machine (A7 r3 (e)): reserved -> supervisor_started -> line_pending -> child_identified -> line_complete
(-> line_pending ...) -> released, or -> stopped (the reservation kept until the Planner releases it); every
transition written atomically and fsynced (file + directory) BEFORE the next action; no crash path releases or adopts.
A restart refuses while its state exists. Never signals, kills or edits any process.

Written by the exp_22 A7 Coder (Claude Opus 5.5, max effort).
"""
import argparse
import importlib.util
import json
import os
import re
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
WORKTREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))


def _load(name):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    return sys.modules[name]


C, E = _load("exp22_r2c_common"), _load("exp22_r2c_evidence")
HOLDER, SUPERVISOR = "exp22", "exp22_r2c_supervisor.py"
TOOLS = ("exp22_r2c_common.py", "exp22_r2c_evidence.py", "exp22_r2c_supervisor.py", "exp22_free_card_detect.sh",
         "exp22_r2c_preflight.sh", "exp22_r2c_preflight_lines.json")
LINES_JSON = os.path.join(HERE, "exp22_r2c_preflight_lines.json")
PHASES = ("reserved", "supervisor_started", "line_pending", "child_identified", "line_complete", "released", "stopped")
TERMINAL = ("released", "stopped")
STATE_KEYS = ("schema_version", "campaign", "run", "pin2", "worktree", "card", "reservation", "seal", "lines_sha256",
              "tools_sha256", "supervisor", "phase", "current", "lines", "stop", "released", "events")
ID_KEYS = ("pid", "start_time", "cwd", "cmdline_sha256")
ORCH_RE, HEX40, HEX64 = re.compile(r"r2_preflight_([0-9]{8}T[0-9]{6})"), re.compile("[0-9a-f]{40}"), re.compile("[0-9a-f]{64}")


def lock_path(worktree):
    return os.path.join(worktree, "outputs_loc", "exp22", "r2_preflight.lock")


def run_of(worktree, orch):
    """The run id iff ``orch`` is the directory <worktree>/outputs_loc/exp22/r2_preflight_<RUN> (no link), else None."""
    real, root = os.path.realpath(orch), os.path.realpath(os.path.join(worktree, "outputs_loc", "exp22"))
    m = ORCH_RE.fullmatch(os.path.basename(real))
    return m[1] if m and os.path.dirname(real) == root and os.path.isdir(real) and not os.path.islink(orch) else None


def _sha_file(path):
    try:
        with open(path, "rb") as f:
            return E.sha256_bytes(f.read())
    except OSError:
        return None


def tools_sha256():
    return {n: _sha_file(os.path.join(HERE, n)) for n in TOOLS}


def create_reservation(dirpath, uuid, *, pin2, run, now):
    """O_CREAT | O_EXCL (``FileExistsError`` when any file holds the name); the held record: path, sha256, dev, inode."""
    raw = json.dumps({"holder": HOLDER, "pin2": pin2, "gpu_uuid": uuid, "epoch": 1, "created": now, "run": run}).encode()
    path = C.reservation_path(dirpath, uuid)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        view = memoryview(raw)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
        st = os.fstat(fd)
    finally:
        os.close(fd)
    E.fsync_dir(dirpath)
    return {"path": path, "sha256": E.sha256_bytes(raw), "dev": st.st_dev, "inode": st.st_ino}


def release_reservation(held):
    """Unlink ``held['path']`` only while it IS our record (bytes + device / inode), through the inode just
    authenticated; [] once removed, else the reasons -- and nothing is unlinked."""
    why = C.held_reasons(held)
    if why:
        return why
    try:
        st = os.lstat(held["path"])
        if [st.st_dev, st.st_ino] != [held["dev"], held["inode"]]:
            return [f"{held['path']} was replaced while it was authenticated"]
        os.unlink(held["path"])
    except OSError as err:
        return [f"{held['path']} could not be removed ({err})"]
    E.fsync_dir(os.path.dirname(held["path"]))
    return []


def _ident_ok(ident):
    return isinstance(ident, dict) and type(ident.get("pid")) is int and ident["pid"] > 0 and (
        "start_time" not in ident or type(ident["start_time"]) is int)


def state_reasons(doc, cfg):
    """The schema of a state file (the run, PIN2, the card, our reservation in THE directory, the phase, identities)."""
    if not isinstance(doc, dict) or list(doc) != list(STATE_KEYS):
        return [f"the state's keys are not {list(STATE_KEYS)}"]
    card, held, sup, cur = doc["card"], doc["reservation"], doc["supervisor"], doc["current"]
    out = [k for k, ok in (
        ("schema / campaign", doc["schema_version"] == 1 and doc["campaign"] == "exp22_r2c_preflight"),
        ("run", isinstance(doc["run"], str) and E.RUN_RE.fullmatch(doc["run"])),
        ("pin2", isinstance(doc["pin2"], str) and HEX40.fullmatch(doc["pin2"])),
        ("card", isinstance(card, dict) and list(card) == ["name", "uuid", "index"] and card["name"] in C.CARDS
         and C.CARDS[card["name"]] == card["uuid"] and type(card["index"]) is int),
        ("reservation", isinstance(held, dict) and list(held) == ["path", "sha256", "dev", "inode"] and isinstance(card, dict)
         and held["path"] == C.reservation_path(cfg.reservations, card.get("uuid")) and HEX64.fullmatch(str(held["sha256"]))
         and type(held["dev"]) is int and type(held["inode"]) is int),
        ("phase", doc["phase"] in PHASES), ("lines / events", isinstance(doc["lines"], list) and isinstance(doc["events"], list)),
        ("supervisor", (sup is None) == (doc["phase"] == "reserved") and (sup is None or _ident_ok(sup))),
        ("current", cur is None or (isinstance(cur, dict) and ("child" not in cur or _ident_ok(cur["child"]))))) if not ok]
    return [f"the state's {k} is malformed or inconsistent" for k in out]


def new_state(*, run, pin2, worktree, card, index, held, seal, seal_sha256, now):
    return {"schema_version": 1, "campaign": "exp22_r2c_preflight", "run": run, "pin2": pin2, "worktree": worktree,
            "card": {"name": card, "uuid": C.CARDS[card], "index": index}, "reservation": held,
            "seal": {"path": seal, "sha256": seal_sha256}, "lines_sha256": _sha_file(LINES_JSON),
            "tools_sha256": tools_sha256(), "supervisor": None, "phase": "reserved", "current": None, "lines": [],
            "stop": None, "released": None,
            "events": [{"t": now, "time": C.iso(now), "what": "reserved (O_EXCL; re-checked before and after)"}]}


def _seal_fields(raw):
    out = {}
    for ln in raw.decode(errors="replace").splitlines():
        k, _, v = ln.partition(" ")
        out.setdefault(k, v)
    return out


def reserve(world, cfg, *, worktree, orch, pin2, card, index, seal, seal_sha256):
    """``(0, msg)`` once the reservation and the state ``reserved`` are durable; ``(3, why)`` otherwise."""
    run, state = run_of(worktree, orch), os.path.join(orch, "state.json")
    if run is None or card not in C.CARDS:
        return 3, f"RESERVE REFUSED: {orch} is not a <worktree>/outputs_loc/exp22/r2_preflight_<RUN> directory / {card!r}"
    if os.path.lexists(state):
        return 3, "RESERVE REFUSED: the state exists (a restart refuses while its state exists)"
    try:
        with open(seal, "rb") as f:
            raw = f.read()
    except OSError as err:
        return 3, f"RESERVE REFUSED: the seal is unreadable ({err})"
    f = _seal_fields(raw)
    bad = [k for k, v in (("pin2", pin2), ("run", run), ("card", f"{card} {C.CARDS[card]} {index}")) if f.get(k) != v]
    if E.sha256_bytes(raw) != seal_sha256 or bad or os.path.dirname(os.path.realpath(seal)) != os.path.realpath(orch):
        return 3, f"RESERVE REFUSED: the seal does not bind this reservation ({bad or 'its sha256 or its location'})"
    try:
        o = C.observe(world, cfg, card)
    except C.Unreadable as err:
        return 3, f"RESERVE REFUSED before the acquisition: {err}"
    if not o["free"] or o["index"] != index:
        why = o["reasons"] or [f"the card is at index {o['index']}, not {index}"]
        return 3, f"RESERVE REFUSED before the acquisition: {'; '.join(why)}"
    try:
        held = create_reservation(cfg.reservations, C.CARDS[card], pin2=pin2, run=run, now=world.now())
    except FileExistsError:
        return 3, "RESERVE REFUSED: another holder reserved the card at the O_EXCL (a lost race; nothing touched)"
    try:
        o = C.observe(world, cfg, card, held=held)
        for uuid in C.CARDS.values():
            if uuid != C.CARDS[card]:
                C.reservation_status(cfg.reservations, uuid)          # every reservation readable, or a refusal
        why = o["reasons"] + ([] if o["index"] == index else [f"the card is at index {o['index']}, not {index}"])
    except C.Unreadable as err:
        why = [str(err)]
    if why:
        rel = release_reservation(held)
        return 3, (f"RESERVE REFUSED after the acquisition: {'; '.join(why)} -- " + (
            "released our own record" if not rel else f"our record does not authenticate, nothing unlinked: {'; '.join(rel)}"))
    doc = new_state(run=run, pin2=pin2, worktree=worktree, card=card, index=index, held=held, seal=seal,
                    seal_sha256=seal_sha256, now=world.now())
    E.write_exclusive(state, E.dumps(doc))
    return 0, f"RESERVED {held['path']} sha256 {held['sha256']} inode {held['inode']}; state {state} phase reserved"


def alive(ident, proc):
    """True when the process ``ident`` names may still run: its pid lives with the recorded start time (and cmdline);
    an identity that cannot be read while the pid exists counts as alive (unprovable)."""
    try:
        live = C.proc_identity(ident["pid"], proc)
    except ValueError:
        return os.path.exists(os.path.join(proc, str(ident["pid"])))
    return "start_time" not in ident or (live["start_time"] == ident["start_time"] and ident.get(
        "cmdline_sha256", live["cmdline_sha256"]) == live["cmdline_sha256"])


def active_reasons(cfg, worktree, except_orch=None):
    """Why an exp22 run is active or unresolved: the exp22 lock held, a supervisor process, an exp22 reservation on
    either card, or a state (other than ``except_orch``'s) neither released nor stopped; unreadable = a reason."""
    out = []
    try:
        if C.lock_status(lock_path(worktree)) == "held":
            out.append(f"the exp22 lock {lock_path(worktree)} is held: a supervisor or a line child lives")
        pids = C.procs_named(SUPERVISOR, cfg.proc)
        out += [f"exp22 supervisor process(es) {pids} run"] if pids else []
        out += [f"a reservation file of holder exp22 exists for {c} ({C.reservation_path(cfg.reservations, u)})"
                for c, u in C.CARDS.items() if C.reservation_status(cfg.reservations, u) == "present (holder exp22)"]
    except C.Unreadable as err:
        out.append(f"unreadable: {err}")
    root = os.path.join(worktree, "outputs_loc", "exp22")
    try:
        names = sorted(os.listdir(root))
    except OSError as err:
        return out + [f"{root} is unreadable ({err})"]
    for n in names:
        d, sp = os.path.join(root, n), os.path.join(root, n, "state.json")
        if not ORCH_RE.fullmatch(n) or not os.path.lexists(sp) or (
                except_orch and os.path.realpath(d) == os.path.realpath(except_orch)):
            continue
        try:
            with open(sp) as f:
                phase = json.load(f)["phase"]
        except (OSError, ValueError, KeyError, TypeError) as err:
            out.append(f"{sp} is unreadable ({err}): unresolved")
            continue
        out += [] if phase in TERMINAL else [f"{d}: phase {phase} (an exp22 run is active or unresolved)"]
    return out


def planner_release(cfg, worktree, orch):
    """The Planner's release after a stop or a crash (by hand): ``(0, msg)`` or ``(1, why)`` -- nothing unlinked."""
    sp = os.path.join(orch, "state.json")
    try:
        if run_of(worktree, orch) is None or C.lock_status(lock_path(worktree)) == "held":
            return 1, "RELEASE REFUSED: not an orchestration directory, or the exp22 lock is held (something lives)"
        with open(sp) as f:
            doc = json.load(f)
    except (OSError, ValueError, C.Unreadable) as err:
        return 1, f"RELEASE REFUSED: {err}"
    why = state_reasons(doc, cfg) or (["already released"] if doc["phase"] == "released" else [])
    live = [k for k, ident in (("supervisor", doc["supervisor"]), ("line child", (doc["current"] or {}).get("child")))
            if not why and ident and alive(ident, cfg.proc)]
    if why or live:
        return 1, "RELEASE REFUSED: " + "; ".join(why + [f"the recorded {k} may be alive (pid + start time)" for k in live])
    note = "the reservation was already absent"
    if os.path.lexists(doc["reservation"]["path"]):
        rel = release_reservation(doc["reservation"])
        if rel:
            return 1, "RELEASE REFUSED (nothing unlinked): " + "; ".join(rel)
        note = "our exact record released"
    now = time.time()
    doc.update(phase="released", released={"t": now, "time": C.iso(now), "by": "planner", "note": note})
    doc["events"].append({"t": now, "time": C.iso(now), "what": f"released by the Planner: {note}"})
    E.write_json_atomic(sp, doc)
    return 0, f"RELEASED by the Planner: {note}"


def confirm(cfg, worktree, orch, pid, timeout=60.0):
    """The launcher's check after the detachment: the state's supervisor IS the live ``pid`` (pid, start time, cwd,
    cmdline sha256) and the exp22 lock is held; waits up to ``timeout`` s for the identity to be written."""
    deadline, sup, doc = time.time() + timeout, None, {}
    while sup is None and time.time() < deadline:
        try:
            with open(os.path.join(orch, "state.json")) as f:
                doc = json.load(f)
            sup = doc.get("supervisor")
        except (OSError, ValueError):
            pass
        if sup is None:
            time.sleep(0.2)
    try:
        live = C.proc_identity(pid, cfg.proc) if sup else None
        held = C.lock_status(lock_path(worktree)) == "held"
    except (ValueError, C.Unreadable) as err:
        return 1, f"NOT CONFIRMED: {err}"
    if not sup or {k: live[k] for k in ID_KEYS} != sup or not held:
        return 1, f"NOT CONFIRMED: recorded {sup}, live {live}, the exp22 lock {'held' if held else 'NOT held'}"
    return 0, (f"CONFIRMED supervisor pid {pid} start time {sup['start_time']} cwd {sup['cwd']} cmdline sha256 "
               f"{sup['cmdline_sha256']}; the exp22 lock held; phase {doc['phase']}")
