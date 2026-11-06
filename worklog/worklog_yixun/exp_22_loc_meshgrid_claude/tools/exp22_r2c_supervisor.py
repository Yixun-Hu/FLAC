#!/usr/bin/env python3
"""exp_22 R2-C pre-flight: the reservation, the durable state and the detached supervisor (plan_loc_meshgrid_r2_bf_yaw.md
§9 A7 r2 (iv) (v)(2), r3 (b) (e) (f) (g), r4, r5).

  reserve --orch D --pin2 P --card C --index I --seal F --seal-sha256 X   the launcher, holding the exp22 lock: the card
        re-checked, the O_EXCL reservation (holder exp22, PIN2, the card, epoch 1, created, run; its exact bytes' sha256
        and device / inode kept), the card re-checked AFTER the acquisition (any change: only our own authenticated
        record is released), then the first durable state ``reserved`` (O_EXCL)
  run --orch D --lock-fd N        the detached supervisor (holds the inherited exp22 lock for its whole life)
  confirm --orch D --pid P        the launcher: the supervisor's recorded identity is the live pid's, the lock held
  active [--except-orch D] [--holding-lock]   exit 0 iff no exp22 run is active or unresolved (--holding-lock: the
        caller -- the launcher's last look -- holds the exclusive exp22 lock itself)
  release --orch D                the Planner, by hand, after a stop or a crash: only with no live supervisor or line
        child (pid + start time, never a pid alone) and only our exact record (bytes + inode) is unlinked

The state machine (A7 r3 (e)): reserved -> supervisor_started -> line_pending -> child_identified -> line_complete
(-> line_pending ...) -> released, or -> stopped (the reservation kept until the Planner releases it); every
transition written atomically and fsynced (file + directory) BEFORE the next action; no crash path releases or adopts.
A restart refuses while its state exists. Never signals, kills or edits any process.

Written by the exp_22 A7 Coder (Claude Opus 5.5, max effort).
"""
import argparse
import fcntl
import importlib.util
import json
import os
import re
import subprocess
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


def active_reasons(cfg, worktree, except_orch=None, holding_lock=False):
    """Why an exp22 run is active or unresolved: the exp22 lock held, a supervisor process, an exp22 reservation on
    either card, or a state (other than ``except_orch``'s) neither released nor stopped; unreadable = a reason.
    ``holding_lock``: the caller (the launcher's last look) holds the exclusive lock itself, so no other one does."""
    out = []
    try:
        if not holding_lock and C.lock_status(lock_path(worktree)) == "held":
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


def spawn_line(argv, env, cwd, log_fd, lock_fd):
    """One line's child: its exact environment and cwd, stdout + stderr into the line's log, stdin from /dev/null;
    the exp22 lock fd passed on, so a child that outlives a crashed supervisor keeps the lock held."""
    return subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=log_fd,
                            stderr=subprocess.STDOUT, pass_fds=(lock_fd,), close_fds=True)


class World(C.World):
    def spawn(self, argv, env, cwd, log_fd, lock_fd):
        return spawn_line(argv, env, cwd, log_fd, lock_fd)


class Supervisor:
    """The detached pre-flight over the registered lines (A7 r2 (iii)-(v), r3 (a) (c) (e)-(g), r4, r5); never signals."""

    def __init__(self, world, cfg, orch, lock_fd, worktree=WORKTREE, child_poll_s=None, relist=(3, 5.0)):
        self.w, self.cfg, self.orch, self.lock_fd, self.wt, self.relist = world, cfg, orch, lock_fd, worktree, relist
        self.poll = C.CHILD_POLL_S if child_poll_s is None else child_poll_s
        self.sp, self.hb = os.path.join(orch, "state.json"), os.path.join(orch, "heartbeat.json")
        self.st = self.lines = self.latch = self.t_info = None

    def path(self, rel):
        return os.path.join(self.wt, rel)

    def save(self):
        E.write_json_atomic(self.sp, self.st)

    def event(self, what, **kw):
        t = self.w.now()
        self.st["events"].append(dict({"t": t, "time": C.iso(t), "what": what}, **kw))
        self.save()
        print(f"[{C.iso(t)}] {what} {json.dumps(kw) if kw else ''}", flush=True)

    def heartbeat(self):
        t = self.w.now()
        E.write_json_atomic(self.hb, {"t": t, "time": C.iso(t), "pid": os.getpid(), "phase": self.st["phase"],
                                      "line": (self.st["current"] or {}).get("name")})

    def stop(self, reason):
        self.st.update(phase="stopped", stop={"reason": reason, "time": C.iso(self.w.now())})
        self.event("STOPPED (the reservation is kept until the Planner releases it)", reason=reason)
        self.heartbeat()
        return 3

    def run(self):
        """0: every line admitted and our record released; 3: refused (nothing changed) or stopped."""
        try:
            fst, lst = os.fstat(self.lock_fd), os.stat(lock_path(self.wt))
            if (fst.st_dev, fst.st_ino) != (lst.st_dev, lst.st_ino):
                raise OSError("the lock fd is not the exp22 lock file")
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with open(self.sp) as f:
                self.st = json.load(f)
        except (OSError, ValueError, TypeError) as err:
            print(f"SUPERVISOR REFUSED (nothing changed): the inherited exp22 lock or the state: {err}")
            return 3
        why = state_reasons(self.st, self.cfg) or [x for x, bad in (
            (f"the state is {self.st['phase']}: a restart refuses while its state exists (the Planner resolves)",
             self.st["phase"] != "reserved" or self.st["lines"] or self.st["current"] is not None),
            ("the orchestration directory is not the state's run", run_of(self.wt, self.orch) != self.st["run"]),
            ("the tools or the line list changed since the reservation",
             tools_sha256() != self.st["tools_sha256"] or _sha_file(LINES_JSON) != self.st["lines_sha256"])) if bad]
        if why:
            print("SUPERVISOR REFUSED (nothing changed): " + "; ".join(why))
            return 3
        me = C.proc_identity(os.getpid())
        self.st.update(supervisor={k: me[k] for k in ID_KEYS}, phase="supervisor_started")
        self.event("supervisor started", pid=os.getpid())
        try:
            with open(LINES_JSON) as f:
                self.lines = E.expand_lines(json.load(f), run=self.st["run"], index=self.st["card"]["index"],
                                            worktree=self.wt, py=C.PY, pytest_main=C.PYTEST_MAIN)
        except (OSError, ValueError) as err:
            return self.stop(f"the line list: {err}")
        for ln in self.lines:
            rc = self.line(ln)
            if rc is not None:
                return rc
        return self.finish()

    def _read(self, rel):
        try:
            with open(self.path(rel), "rb") as f:
                return f.read()
        except (OSError, TypeError):
            return b""

    def checks(self):
        """HEAD = PIN2 on a clean tracked tree, the four pins, the tools, every earlier log / output / record intact."""
        out = []
        try:
            head = subprocess.run(["git", "-C", self.wt, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=60)
            dirty = subprocess.run(["git", "-C", self.wt, "status", "--porcelain", "--untracked-files=no"],
                                   capture_output=True, text=True, timeout=300)
            out += [] if head.returncode == 0 and head.stdout.strip() == self.st["pin2"] else [
                f"HEAD {head.stdout.strip()!r} is not PIN2 {self.st['pin2']}"]
            out += [] if dirty.returncode == 0 and dirty.stdout == "" else ["the tracked tree is dirty (or git failed)"]
        except (OSError, subprocess.SubprocessError) as err:
            out.append(f"git: {err}")
        out += [f"{rel} does not hash to its pin" for rel, sha in E.PIN_FILES.items() if _sha_file(self.path(rel)) != sha]
        out += [] if tools_sha256() == self.st["tools_sha256"] else ["the tools changed since the reservation"]
        return out + [f"{rel} changed since its line {done['name']}" for done in self.st["lines"]
                      for rel, sha in done["files"].items() if _sha_file(self.path(rel)) != sha]

    def input_reasons(self, ln):
        made = {rel: sha for done in self.st["lines"] for rel, sha in done["files"].items()}
        return [f"the input {rel} is not its line's recorded output" for rel in ln["inputs"]
                if rel not in made or _sha_file(self.path(rel)) != made[rel]]

    def card(self):
        """Our card under the card rule with our held record; a process already gone may linger in the compute table
        for a moment after its line, so a table of only vanished pids is re-queried (relist)."""
        name, index = self.st["card"]["name"], self.st["card"]["index"]
        for attempt in range(self.relist[0]):
            try:
                o = C.observe(self.w, self.cfg, name, held=self.st["reservation"])
            except C.Unreadable as err:
                return [str(err)]
            gone = [p for p in o["pids_on_card"] if not os.path.exists(os.path.join(self.cfg.proc, str(p)))]
            if not o["pids_on_card"] or gone != o["pids_on_card"] or attempt + 1 == self.relist[0]:
                break
            self.w.sleep(self.relist[1])
        return o["reasons"] + ([] if o["index"] == index else [f"{name} is now at index {o['index']}, not {index}"])

    def preline(self, ln):
        fresh = [ln["log"], ln["record"]] + ln["outputs"] + ([ln["junit"]] if ln["junit"] else [])
        if ln["name"] == "GV" or ln["kind"] == "probe":           # the scientific directory: created by its driver
            fresh.append(os.path.dirname(ln["outputs"][0]))
        return self.checks() + [f"{rel} exists before its line (never clobbered)" for rel in fresh
                                if os.path.lexists(self.path(rel))] + self.input_reasons(ln) + self.card()

    def line(self, ln):
        why = self.preline(ln)
        if why:
            return self.stop(f"before line {ln['n']} {ln['name']}: " + "; ".join(why))
        cur = {k: ln[k] for k in ("n", "name", "kind", "argv", "env", "log", "record", "outputs", "inputs", "junit")}
        cur.update(cwd=self.wt, card=self.st["card"], reservation=self.st["reservation"], started=self.w.now())
        self.st.update(current=cur, phase="line_pending")
        self.save()
        self.t_info = None
        fd = os.open(self.path(ln["log"]), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        try:
            proc = self.w.spawn(ln["argv"], ln["env"], self.wt, fd, self.lock_fd)
        except OSError as err:
            return self.stop(f"line {ln['n']} {ln['name']} could not be spawned: {err}")
        finally:
            os.close(fd)
        cur["child"] = self.identify(proc, ln)
        self.st["phase"] = "child_identified"
        self.save()
        self.watch(proc)
        cur.update(rc=proc.returncode, ended=self.w.now())
        self.st["phase"] = "line_complete"                      # the exit recorded BEFORE any judgement or stop
        self.save()
        refusals = self.evidence(ln, proc.returncode) + ([self.latch] if self.latch else [])
        sha = self.record(ln, cur, refusals)
        files = {rel: _sha_file(self.path(rel)) for rel in [ln["log"], ln["record"]] + ln["outputs"]
                 + ([ln["junit"]] if ln["junit"] else [])}
        self.st["lines"].append({"n": ln["n"], "name": ln["name"], "rc": proc.returncode, "admitted": not refusals,
                                 "record": ln["record"], "record_sha256": sha,
                                 "files": {k: v for k, v in files.items() if v is not None}})
        self.st["current"] = None
        self.save()
        return self.stop(f"line {ln['n']} {ln['name']}: " + "; ".join(refusals)) if refusals else None

    def identify(self, proc, ln):
        """The child's /proc identity right after the spawn (pid, start time, cwd, cmdline sha256, fd 1 / 2). The kernel
        closes the spawn's close-on-exec error pipe (Popen returns) BEFORE it maps the new argv, so an empty or other
        cmdline is re-read for up to 5 s until it is the spawned command or the child has ended."""
        expect, deadline = E.sha256_bytes(b"".join(a.encode() + b"\0" for a in ln["argv"])), time.monotonic() + 5.0
        while True:
            try:
                ident = C.proc_identity(proc.pid, self.cfg.proc)
            except ValueError as err:
                ident = {"pid": proc.pid, "unreadable": str(err)}
            done = proc.poll() is not None
            if ident.get("cmdline_sha256") == expect or done or time.monotonic() > deadline:
                break
            time.sleep(0.01)
        ident.update(expected_cmdline_sha256=expect, exited_before_identification=done)
        if ident.get("cmdline_sha256") != expect and not ident["exited_before_identification"]:
            self.latch = f"the child {proc.pid} runs but its /proc identity is not the spawned command"
        return ident

    def watch(self, proc):
        """Wait for the child, NEVER signalling it: a heartbeat and the card re-checked every ``poll`` s."""
        while True:
            self.heartbeat()
            try:
                proc.wait(timeout=self.poll)
                return
            except subprocess.TimeoutExpired:
                pass
            if self.latch is None:
                why = self.during(proc.pid)
                if why:
                    self.latch = "during the line: " + "; ".join(why)
                    self.event("LATCH: no further line; the running child is never signalled", reason=self.latch)

    def during(self, pid):
        try:
            before = C.descendants(pid, self.cfg.proc)
            o = C.observe(self.w, self.cfg, self.st["card"]["name"], held=self.st["reservation"], ours=before)
            after = C.descendants(pid, self.cfg.proc)
        except (C.Unreadable, OSError) as err:
            return [str(err)]
        foreign = [p for p in o["pids_on_card"] if p not in after]
        return ([f"foreign compute process(es) {foreign} on the card"] if foreign else []) + o["rule_reasons"] + (
            [] if o["index"] == self.st["card"]["index"] else ["the card's index changed"])

    def evidence(self, ln, rc):
        """The line's admission (A7 r2 (iv), r3 (a) (c), r4, r5): its exit, the backend rule, T's three layers, the
        outputs, the inputs bound to their lines' records, V's RE-VERIFIED and A's WITHIN THE CAP."""
        text = self._read(ln["log"]).decode(errors="replace")
        out = ([] if rc == 0 else [f"exit code {rc}"]) + (E.backend_log_reasons(text, ln["target"]) if ln["wrapped"] else [])
        if ln["kind"] == "pytest":
            junit = self._read(ln["junit"])
            out += E.junit_reasons(junit, ln["node_ids"]) if junit else ["the JUnit file is missing"]
            why, self.t_info = E.pytest_summary_reasons(text, ln["node_ids"])
            out += why
        out += [f"the output {rel} is missing or not a regular file" for rel in ln["outputs"]
                if not os.path.isfile(self.path(rel)) or os.path.islink(self.path(rel))] + self.input_reasons(ln)
        lines = text.splitlines()
        if ln["kind"] == "reverify" and lines.count("RE-VERIFIED") != 1:
            out.append("the re-verification did not print RE-VERIFIED exactly once")
        if ln["kind"] == "arith" and not any(x.startswith("A5 one-card probe: makespan_proj = ")
                                             and x.endswith("-> WITHIN THE CAP") for x in lines):
            out.append("the A5 arithmetic did not print WITHIN THE CAP (above the cap: STOP AND ASK YIXUN)")
        return out

    def record(self, ln, cur, refusals):
        """The O_EXCL backend-dispatch record of the line (A7 r2 (iv), r3 (g), r4) in the orchestration directory."""
        raw = self._read(ln["log"])
        sent = [x for x in raw.decode(errors="replace").splitlines() if x.startswith(E.SENTINEL + " ")]
        t, info = ln["kind"] == "pytest", self.t_info or {}
        doc = {"schema_version": 1, "kind": "exp22_r2c_dispatch_record", "run": self.st["run"], "n": ln["n"],
               "name": ln["name"], "pin2": self.st["pin2"], "cwd": self.wt, "argv": ln["argv"], "env": ln["env"],
               "card": self.st["card"], "wrapped": ln["wrapped"], "target": ln["target"], "started": cur["started"],
               "started_iso": C.iso(cur["started"]), "ended": cur["ended"], "ended_iso": C.iso(cur["ended"]),
               "exit_code": cur["rc"], "sentinel": sent[0] if len(sent) == 1 else None,
               "log": {"path": ln["log"], "sha256": E.sha256_bytes(raw), "bytes": len(raw)},
               "output_paths": [{"path": r, "sha256": _sha_file(self.path(r))} for r in ln["outputs"]],
               "input_paths": [{"path": r, "sha256": _sha_file(self.path(r))} for r in ln["inputs"]],
               "selected_node_ids": ln["node_ids"], "pytest_summary": info.get("pytest_summary") if t else None,
               "pytest_passed_lines_sha256": info.get("passed_lines_sha256") if t else None,
               "junit_xml": {"path": ln["junit"], "sha256": _sha_file(self.path(ln["junit"]))} if t else None,
               "launcher_sha256": E.LAUNCHER_SHA256, "overlay_sha256": E.OVERLAY_SHA256, "child": cur["child"],
               "reservation": self.st["reservation"], "admitted": not refusals, "refusals": refusals}
        return E.write_exclusive(self.path(ln["record"]), E.dumps(doc))

    def finish(self):
        why = self.checks()
        if why:
            return self.stop("before the release: " + "; ".join(why))
        why = release_reservation(self.st["reservation"])
        if why:
            return self.stop("the final release (nothing unlinked): " + "; ".join(why))
        t = self.w.now()
        self.st.update(phase="released", released={"t": t, "time": C.iso(t), "by": "supervisor",
                                                   "note": "every line admitted; our exact record released"})
        self.event("RELEASED: T -> the gates -> the BF probe -> A5 complete; our exact record released")
        self.heartbeat()
        return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="exp_22 R2-C pre-flight: reservation, supervisor, release")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reserve")
    for a in ("--orch", "--pin2", "--card", "--seal", "--seal-sha256"):
        r.add_argument(a, required=True)
    r.add_argument("--index", type=int, required=True)
    u = sub.add_parser("run")
    u.add_argument("--orch", required=True)
    u.add_argument("--lock-fd", type=int, required=True)
    c = sub.add_parser("confirm")
    c.add_argument("--orch", required=True)
    c.add_argument("--pid", type=int, required=True)
    c.add_argument("--timeout", type=float, default=60.0)
    a = sub.add_parser("active")
    a.add_argument("--except-orch")
    a.add_argument("--holding-lock", action="store_true", help="the caller holds the exp22 lock itself")
    sub.add_parser("release").add_argument("--orch", required=True)
    args = ap.parse_args(argv)
    cfg, orch = C.registered_config(), os.path.abspath(args.orch) if getattr(args, "orch", None) else None
    if args.cmd == "run":
        return Supervisor(World(C.NVIDIA_SMI), cfg, orch, args.lock_fd).run()
    if args.cmd == "reserve":
        rc, msg = reserve(World(C.NVIDIA_SMI), cfg, worktree=WORKTREE, orch=orch, pin2=args.pin2, card=args.card,
                          index=args.index, seal=os.path.abspath(args.seal), seal_sha256=args.seal_sha256)
    elif args.cmd == "confirm":
        rc, msg = confirm(cfg, WORKTREE, orch, args.pid, args.timeout)
    elif args.cmd == "release":
        rc, msg = planner_release(cfg, WORKTREE, orch)
    else:
        why = active_reasons(cfg, WORKTREE, except_orch=args.except_orch, holding_lock=args.holding_lock)
        rc, msg = (1, "EXP22 RUN ACTIVE OR UNRESOLVED: " + "; ".join(why)) if why else (0, "no exp22 run is active")
    print(msg, flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
