#!/usr/bin/env python3
"""exp_22 R2-C pre-flight: the evidence rules and writers (plan_loc_meshgrid_r2_bf_yaw.md §9 A7 r2 (iv), r3 (a) (c)
(g), r4, r5).

* The backend rule (r3 (a); the reviewed exp_35 collector's ``backend_log_reasons`` with the line's own target):
  in a wrapped line's complete log no line contains ``No module named 'flash_attn'`` or ``NO-FLASH LAUNCH REFUSED``;
  exactly ONE line starts ``EXP35_NO_FLASH_LAUNCHER `` and it equals the pinned sentinel for THIS target; before it,
  exactly one back-to-back pair (the overlay's ImportError line, then transformer.py's fallback line); after it,
  every occurrence of either line belongs to such a pair (a child interpreter importing the transformer).
* T's admission (r3 (c) + r4 + r5), three layers: (1) pytest exit 0; (2) the fresh JUnit file: one testsuite with
  tests=11, failures=errors=skipped=0, over exactly the eleven registered identities, no failure / error / skipped
  element; (3) the -rA short summary is exactly eleven ``PASSED <nodeid>`` lines naming the eleven ids, followed by
  the quiet-form terminal summary, a whole line matching SUMMARY_RE. ``-o xfail_strict=true`` turns an XPASS of a
  marker WITHOUT an explicit ``strict=`` into a FAILED report; an explicit ``strict=False`` XPASS stays an XPASS
  (a JUnit pass under pytest 9.1.1) and is refused by layer (3).
* The pre-flight line list (``exp22_r2c_preflight_lines.json``, A7 r2 (iii)): validated strictly -- the registered
  fifteen lines in order, the wrapped ones through the launcher, every gate's own mode flag, the out-dirs only the
  fresh @GATES@ / @PROBE@, no shard / merge / report / crossarm / resume token -- and expanded for one run id and card.
* The launcher's sealed admission evidence (A7 r3 (b)): WINDOW_POLLS polls of the chosen card POLL_S s apart, each
  appended to an O_EXCL capture record (the poll's time, the raw nvidia-smi tables and their sha256), and the
  verifier the seal and the launch apply (the count, the cadence, every poll free, every table re-parsed, one index).
* Writers: an O_EXCL record (never overwritten; file and directory fsynced) and the atomic state write (temp,
  fsync, rename, directory fsync).

Written by the exp_22 A7 Coder (Claude Opus 5.5, max effort).
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True

PIN_FILES = {   # PIN2's four files = exp_35 H35 5b065801, byte-identical (A7 r1 (ii), r2 (ii))
    "overlays/no_flash/flash_attn.py": "0d0e52ca693e236a2e0a44cd61f61539fdf995cff5132d1f621181a0f3e7c023",
    "overlays/no_flash/exp35_no_flash_launch.py": "bbd2847d3a50685420750d066195c8656a1509b4e905883886c0568d244043de",
    "src/tests/test_exp35_no_flash_launch.py": "d61a3c02a7e1295a48b3edb60e8abccab508cadb6f4c11bdca9872234bd961dd",
    "pyproject.toml": "6422eae5482773db252372760825749f41ac061487be892e350fda997b619a74"}
OVERLAY_SHA256 = PIN_FILES["overlays/no_flash/flash_attn.py"]
LAUNCHER_SHA256 = PIN_FILES["overlays/no_flash/exp35_no_flash_launch.py"]
LAUNCHER = "overlays/no_flash/exp35_no_flash_launch.py"
SENTINEL = "EXP35_NO_FLASH_LAUNCHER"
OVERLAY_LINE = "exp35 no-flash overlay: flash_attn disabled, attention backend = torch SDPA"
FALLBACK_LINE = "flash_attn not installed, disabling Flash Attention"            # src/models/transformer.py:16
MISSING_LINE = "No module named 'flash_attn'"
REFUSED_LINE = "NO-FLASH LAUNCH REFUSED"
SUMMARY_RE = re.compile(r"11 passed(?:, [1-9][0-9]* warnings?)? in [0-9]+\.[0-9]{2}s(?: \([^()\r\n]+\))?")   # r5
HEADER_RE = re.compile(r"=+ short test summary info =+")
OUTCOME_RE = re.compile(r"(?:PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS) ")


T_NODE_IDS = tuple(f"src/tests/test_{f}.py::test_{t}" for f, t in (
    ("eval_localization", "integration_generation_is_reproducible_for_a_fixed_noise_key"),
    ("eval_localization", "integration_parity_one_query_matches_the_eval_flac_path"),
    ("eval_localization", "integration_real_mk_batch_split_is_bitwise_irrelevant"),
    ("eval_localization", "integration_real_mk_parity_is_exact_without_conditioning_autocast"),
    ("eval_localization", "run_query_matches_a_candidate_major_replay_with_full_conditioning[4]"),
    ("eval_localization", "run_query_matches_a_candidate_major_replay_with_full_conditioning[64]"),
    ("eval_localization", "run_query_mk_replay_is_stable_across_the_batch_split"),
    ("loc_meshgrid_queries", "our_call_graph_matches_an_eval_flac_faithful_reference"),
    ("loc_meshgrid_queries", "real_bounded_slice_materializes_through_the_released_loader"),
    ("loc_meshgrid_queries", "release_stack_resolves_and_records_the_agree_checkpoint"),
    ("loc_meshgrid_queries", "skipping_the_metric_stack_changes_the_draw")))
T_FLAGS = ("-x", "-q", "--color=no", "-o", "xfail_strict=true", "-rA", "-p", "no:cacheprovider", "--junitxml",
           "@ORCH@/T_junit.xml")
GATE_OUT = {"GV": ("--cache-parity-check", "vanilla_cache_parity"), "G1": ("--fa-split-parity-check", "fa_split_parity"),
            "G2": ("--cache-parity-check", "fa_cache_parity"), "G3": ("--fa-production-parity-check", "fa_production_parity"),
            "G3D": ("--fa-production-drift-measurement", "fa_production_drift"),
            "G0": ("--replay-check", "vanilla_replay_regression")}
GATE_FLAGS = {flag for flag, _ in GATE_OUT.values()}
KIND = dict({"T": "pytest", "P": "probe", "A": "arith"}, **{g: "gate" for g in GATE_OUT},
            **{f"V_{g}": "reverify" for g in GATE_OUT})
LINE_NAMES = ("T",) + tuple(x for g in GATE_OUT for x in (g, f"V_{g}")) + ("P", "A")
WRAPPED = ("pytest", "gate", "probe")
SHARD_TOKENS = {"--emit-launch-record", "--launch-record", "--merge-shards", "--merge-out", "--rooms", "--resume", "-m"}
PROBE_OUT = "@PROBE@/diagnostics_BF_bathrooms14.json"
PLACEHOLDERS = ("@PY@", "@L@", "@PYTEST_MAIN@", "@ORCH@", "@GATES@", "@PROBE@")
BASE_ENV = {"HOME": "/home/yixunhu", "USER": "yixunhu", "LOGNAME": "yixunhu", "LANG": "en_US.UTF-8",
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"}
RUN_RE = re.compile(r"[0-9]{8}T[0-9]{6}")
HERE = os.path.dirname(os.path.abspath(__file__))


def common():
    """The shared observation module beside this file (one copy per process)."""
    if "exp22_r2c_common" not in sys.modules:
        spec = importlib.util.spec_from_file_location("exp22_r2c_common", os.path.join(HERE, "exp22_r2c_common.py"))
        sys.modules["exp22_r2c_common"] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules["exp22_r2c_common"])
    return sys.modules["exp22_r2c_common"]


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sentinel_line(target):
    return (f"{SENTINEL} attention_backend=sdpa overlay_sha256={OVERLAY_SHA256} launcher_sha256={LAUNCHER_SHA256} "
            f"target={target}")


def backend_log_reasons(text, target):
    """[] iff ``text`` (a wrapped line's complete log) proves the SDPA backend for ``target`` (A7 r3 (a))."""
    lines, out = text.splitlines(), []
    if any(MISSING_LINE in ln for ln in lines):
        out.append(f"the log reports {MISSING_LINE!r}: an import ran without the overlay")
    if any(REFUSED_LINE in ln for ln in lines):
        out.append(f"the log carries {REFUSED_LINE!r}: the launcher refused")
    sent = [i for i, ln in enumerate(lines) if ln.startswith(SENTINEL + " ")]
    if len(sent) != 1:
        return out + [f"the log holds {len(sent)} lines starting {SENTINEL!r}, not exactly one"]
    s = sent[0]
    if lines[s] != sentinel_line(target):
        out.append(f"the sentinel {lines[s]!r} is not the expected {sentinel_line(target)!r}")
    ov = [i for i in range(s) if lines[i] == OVERLAY_LINE]
    fb = [i for i in range(s) if lines[i] == FALLBACK_LINE]
    if not (len(ov) == len(fb) == 1 and fb[0] == ov[0] + 1):
        out.append(f"before the sentinel: {len(ov)} overlay and {len(fb)} fallback line(s), not one back-to-back pair")
    for i in range(s + 1, len(lines)):
        if lines[i] == OVERLAY_LINE and (i + 1 == len(lines) or lines[i + 1] != FALLBACK_LINE):
            out.append(f"log line {i + 1}: an overlay line not followed by the fallback line")
        if lines[i] == FALLBACK_LINE and lines[i - 1] != OVERLAY_LINE:
            out.append(f"log line {i + 1}: a fallback line not preceded by the overlay line")
    return out


def junit_identity(nodeid):
    """(classname, name) of ``nodeid`` in the JUnit XML -- pytest 9.1.1 ``junitxml.mangle_test_address``."""
    path, bracket, params = nodeid.partition("[")
    names = path.split("::")
    names[0] = re.sub(r"\.py$", "", names[0].replace("/", "."))
    names[-1] += bracket + params
    return ".".join(names[:-1]), names[-1]


def junit_reasons(raw, node_ids):
    """Layer (2): [] iff ``raw`` is pytest's JUnit file of exactly ``node_ids``, every one a plain pass."""
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as err:
        return [f"the JUnit file is not XML ({err})"]
    suites = list(root)
    if root.tag != "testsuites" or len(suites) != 1 or suites[0].tag != "testsuite":
        return [f"the JUnit root <{root.tag}> does not hold exactly one <testsuite>"]
    suite, n = suites[0], str(len(node_ids))
    out = [f"JUnit {k}={suite.get(k)!r}, not {v!r}" for k, v in
           (("tests", n), ("failures", "0"), ("errors", "0"), ("skipped", "0")) if suite.get(k) != v]
    cases = [c for c in suite if c.tag == "testcase"]
    got = sorted((c.get("classname"), c.get("name")) for c in cases)
    if got != sorted(junit_identity(i) for i in node_ids):
        out.append(f"the JUnit cases {got} are not exactly the {n} registered identities")
    bad = sorted({c.get("name") for c in cases for x in c if x.tag in ("failure", "error", "skipped")})
    return out + ([f"JUnit cases with a failure / error / skipped element: {bad}"] if bad else [])


def pytest_summary_reasons(text, node_ids):
    """Layer (3): ``(reasons, info)`` -- the ONE short summary is exactly the PASSED lines of ``node_ids``, and the
    line right after it is the quiet-form terminal summary (SUMMARY_RE, the whole line)."""
    lines = text.splitlines()
    heads = [i for i, ln in enumerate(lines) if HEADER_RE.fullmatch(ln)]
    if len(heads) != 1:
        return [f"the log holds {len(heads)} 'short test summary info' headers, not exactly one"], None
    i, block = heads[0] + 1, []
    while i < len(lines) and OUTCOME_RE.match(lines[i]):
        block.append(lines[i])
        i += 1
    summary = lines[i] if i < len(lines) else None
    out = [] if summary is not None and SUMMARY_RE.fullmatch(summary) else [
        f"the terminal summary {summary!r} is not the quiet form '11 passed[, N warning(s)] in S.SSs[ (...)]'"]
    passed = [ln for ln in block if ln.startswith("PASSED ")]
    if len(passed) != len(block):
        out.append(f"the short summary holds other outcomes: {[ln for ln in block if not ln.startswith('PASSED ')][:5]}")
    if sorted(ln[len("PASSED "):] for ln in passed) != sorted(node_ids):
        out.append(f"the short summary's {len(passed)} PASSED lines are not exactly the {len(node_ids)} registered ids")
    return out, {"pytest_summary": summary, "passed_lines": passed,
                 "passed_lines_sha256": sha256_bytes(("\n".join(passed) + "\n").encode())}


def _line_reasons(ln, prev):
    name, kind, argv = ln["name"], ln["kind"], ln["argv"]
    out = [f"{name}: kind {kind!r}, not {KIND[name]!r}"] if kind != KIND[name] else []
    keys = ("n", "name", "kind", "argv", "outputs", "inputs") + (("junit",) if name == "T" else ())
    if sorted(ln) != sorted(keys) or not all(isinstance(a, str) for a in argv + ln["outputs"] + ln["inputs"]):
        return out + [f"{name}: keys {sorted(ln)} / non-string entries"]
    out += [f"{name}: {a!r} is a shard / merge / report / crossarm / resume token" for a in argv
            if a in SHARD_TOKENS or "meshgrid_report" in a or "meshgrid_crossarm" in a]
    out += [f"{name}: unknown placeholder in {a!r}" for a in argv + ln["outputs"] + ln["inputs"]
            if set(re.findall(r"@[A-Z_0-9]+@", a)) - set(PLACEHOLDERS)]
    dirs = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "--out-dir"]
    want_dir = {"gate": ["@GATES@"], "probe": ["@PROBE@"]}.get(kind, [])
    if dirs != want_dir or argv.count("--out-dir") != len(want_dir):
        out.append(f"{name}: --out-dir {dirs}, not {want_dir}")
    modes = [a for a in argv if a in GATE_FLAGS or a == "--probe"]
    if kind in WRAPPED and argv[:2] != ["@PY@", "@L@"]:
        out.append(f"{name}: not run through the launcher")
    if kind == "pytest" and (tuple(argv[2:13]) != ("@PYTEST_MAIN@",) + T_FLAGS or tuple(argv[13:]) != T_NODE_IDS
                             or ln["junit"] != "@ORCH@/T_junit.xml" or ln["outputs"] or ln["inputs"]):
        out.append("T: not the registered pytest line on the eleven ids")
    if kind == "gate" and (argv[2] != "localize_meshgrid.py" or modes != [GATE_OUT[name][0]]
                           or ln["outputs"] != [f"@GATES@/{GATE_OUT[name][1]}.json"] or ln["inputs"]):
        out.append(f"{name}: not the registered {GATE_OUT[name][0]} gate")
    if kind == "probe" and (argv[2] != "localize_meshgrid.py" or modes != ["--probe"]
                            or argv[argv.index("--probe") + 1] != "1" or ln["outputs"] != [PROBE_OUT] or ln["inputs"]):
        out.append("P: not the registered one-card probe")
    if kind == "reverify" and (argv[:2] != ["@PY@", "-c"] or len(argv) != 4 or "reverify_gate_evidence" not in argv[2]
                               or argv[3:] != prev["outputs"] or ln["inputs"] != prev["outputs"] or ln["outputs"]):
        out.append(f"{name}: not the re-verification of {prev['name']}'s evidence")
    if kind == "arith" and (argv[:2] != ["@PY@", "-c"] or len(argv) != 3 or f"'{PROBE_OUT}'" not in argv[2]
                            or "1.5 * h * 0.5145" not in argv[2] or ln["inputs"] != [PROBE_OUT] or ln["outputs"]):
        out.append("A: not the registered one-card A5 arithmetic")
    return out


def lines_reasons(doc):
    """[] iff ``doc`` is the registered pre-flight list: no shard, merge, report or crossarm line is reachable."""
    if not (isinstance(doc, dict) and doc.get("schema_version") == 1 and isinstance(doc.get("lines"), list)
            and all(isinstance(ln, dict) and isinstance(ln.get("argv"), list) for ln in doc["lines"])):
        return ["not a schema-1 line list"]
    if [(ln.get("n"), ln.get("name")) for ln in doc["lines"]] != list(enumerate(LINE_NAMES, 1)):
        return [f"the lines are not the registered {len(LINE_NAMES)} in order"]
    return [x for i, ln in enumerate(doc["lines"]) for x in _line_reasons(ln, doc["lines"][i - 1])]


def expand_lines(doc, *, run, index, worktree, py, pytest_main):
    """The concrete lines of run ``run`` on the card of index ``index`` (paths relative to the worktree)."""
    why = lines_reasons(doc)
    if why or not (isinstance(run, str) and RUN_RE.fullmatch(run)) or type(index) is not int:
        raise ValueError(f"refused: {why or f'run id {run!r} / index {index!r}'}")
    orch = f"outputs_loc/exp22/r2_preflight_{run}"
    subs = {"@PY@": py, "@L@": LAUNCHER, "@PYTEST_MAIN@": pytest_main, "@ORCH@": orch,
            "@GATES@": f"outputs_loc/exp22/r2_gates_{run}", "@PROBE@": f"outputs_loc/exp22/r2_probe_BF_{run}"}

    def sub(text):
        for k, v in subs.items():
            text = text.replace(k, v)
        return text
    out = []
    for ln in doc["lines"]:
        wrapped, argv, tag = ln["kind"] in WRAPPED, [sub(a) for a in ln["argv"]], f"{ln['n']:02d}_{ln['name']}"
        env = (dict(BASE_ENV, CUDA_VISIBLE_DEVICES=str(index), HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
                    PYTHONPATH=os.path.join(worktree, "overlays", "no_flash")) if wrapped
               else dict(BASE_ENV, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"))
        out.append({"n": ln["n"], "name": ln["name"], "kind": ln["kind"], "wrapped": wrapped,
                    "target": argv[2] if wrapped else None, "argv": argv, "env": env,
                    "log": f"{orch}/line_{tag}.log", "record": f"{orch}/dispatch_{tag}.json",
                    "outputs": [sub(p) for p in ln["outputs"]], "inputs": [sub(p) for p in ln["inputs"]],
                    "junit": sub(ln["junit"]) if "junit" in ln else None,
                    "node_ids": list(T_NODE_IDS) if ln["kind"] == "pytest" else None})
    return out


def fsync_dir(d):
    fd = os.open(d, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


Snap = collections.namedtuple("Snap", "raw sha256 dev inode")


def snapshot(path):
    """ONE read of the regular file at ``path`` through a no-follow, non-blocking descriptor: its bytes, their sha256,
    its (device, inode); None when it is absent, a symlink, not a regular file or unreadable. What a line is judged
    by, recorded with and later authenticated against is this one snapshot (Codex A7-C1 BLOCKING 1)."""
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


def identity(snap):
    return {"sha256": snap.sha256, "dev": snap.dev, "inode": snap.inode}


def write_exclusive_snap(path, raw):
    """O_CREAT | O_EXCL (a symlink or any existing name refuses): never overwritten; the file and its directory
    fsynced; returns the Snap of exactly the bytes written and the descriptor's (device, inode)."""
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


def write_exclusive(path, raw):
    """``write_exclusive_snap`` returning the sha256 of the bytes written."""
    return write_exclusive_snap(path, raw).sha256


def dumps(doc):
    return (json.dumps(doc, indent=1) + "\n").encode()


def write_json_atomic(path, doc):
    """A temp file beside ``path``, fsync, rename over it, fsync the directory: the old or the new state, whole."""
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".exp22_state.", dir=d)
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


def _append_row(fd, row):
    os.write(fd, (json.dumps(row) + "\n").encode())
    os.fsync(fd)


def capture_polls(world, cfg, card, record):
    """A7 r3 (b): WINDOW_POLLS polls of ``card`` POLL_S s apart (from the first), each appended -- one JSON line,
    fsynced -- to the O_EXCL record ``record``; ``(0, info)`` only when every poll was on time (<= POLL_SLACK_S
    late) and free; else ``(1, why)`` at the first poll that was not (kept in the record)."""
    C = common()
    fd = os.open(record, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_APPEND, 0o644)
    try:
        fsync_dir(os.path.dirname(os.path.abspath(record)))
        t0 = world.now()
        for k in range(C.WINDOW_POLLS):
            due = t0 + k * C.POLL_S
            world.sleep(due - world.now())
            try:
                o = C.observe(world, cfg, card)
            except C.Unreadable as err:
                _append_row(fd, {"k": k, "due": due, "unreadable": str(err)})
                return 1, f"poll {k}: unreadable: {err}"
            _append_row(fd, dict({"k": k, "due": due}, **o))
            if o["t"] - due > C.POLL_SLACK_S:
                return 1, f"poll {k} ran {o['t'] - due:.1f} s late (> {C.POLL_SLACK_S} s): the cadence is not attested"
            if not o["free"]:
                return 1, f"poll {k}: {card} is busy: {'; '.join(o['reasons'])}"
        return 0, {"card": card, "first": t0, "last": o["t"], "polls": C.WINDOW_POLLS}
    finally:
        os.close(fd)


def _row_reasons(r, k, card, t0, C):
    uuid, rules = C.CARDS[card], ("exp35 lock absent", "exp35 lock unheld") if card == "GPU0" else (
        "his s43 table job terminally complete",)
    if not isinstance(r, dict) or r.get("k") != k or "unreadable" in r or not isinstance(t0, (int, float)):
        return [f"poll {k}: malformed or unreadable"]
    out = [] if isinstance(r.get("due"), (int, float)) and abs(r["due"] - (t0 + k * C.POLL_S)) < 1e-6 else [
        f"poll {k}: due {r.get('due')!r} is off the {C.POLL_S} s cadence"]
    lag = r.get("t") - r["due"] if isinstance(r.get("t"), (int, float)) and not out else None
    out += [] if lag is not None and 0 <= lag <= C.POLL_SLACK_S else [f"poll {k}: observed {lag} s after its due time"]
    if (r.get("card"), r.get("uuid"), r.get("free"), r.get("reasons"), r.get("reservation"), r.get("pids_on_card")) != (
            card, uuid, True, [], "absent", []) or r.get("rule") not in rules:
        out.append(f"poll {k}: not a free observation of {card} under its rule")
    try:
        if (C.sha256_bytes(r["apps_raw"].encode()) != r.get("apps_sha256")
                or any(u == uuid for _, u in C.compute_apps(r["apps_raw"]))):
            out.append(f"poll {k}: the compute table lists {card} or does not hash to its sha256")
        m = C.gpu_map(r["gpu_map_raw"])
        if sorted(m.values()) != sorted(C.CARDS.values()) or m.get(r.get("index")) != uuid:
            out.append(f"poll {k}: the GPU map does not put {uuid} at index {r.get('index')!r}")
    except (AttributeError, KeyError, TypeError, ValueError) as err:
        out.append(f"poll {k}: unparsable tables ({err})")
    return out


def capture_reasons(raw, card):
    """``([], info)`` iff ``raw`` -- the capture record's bytes -- attests WINDOW_POLLS on-time free polls of ``card``
    at one index; else ``(reasons, None)``."""
    C = common()
    try:
        rows = [json.loads(ln) for ln in raw.decode().splitlines()]
    except (UnicodeDecodeError, ValueError) as err:
        return [f"the capture record is not JSON lines ({err})"], None
    if card not in C.CARDS or len(rows) != C.WINDOW_POLLS:
        return [f"{len(rows)} polls of {card!r}, not {C.WINDOW_POLLS} of a registered card"], None
    t0 = rows[0].get("due") if isinstance(rows[0], dict) else None
    out = [x for k, r in enumerate(rows) for x in _row_reasons(r, k, card, t0, C)]
    if out or len({r["index"] for r in rows}) != 1:
        return out or ["the card's index changed between polls"], None
    return [], {"card": card, "uuid": C.CARDS[card], "index": rows[0]["index"], "first": rows[0]["t"],
                "last": rows[-1]["t"], "polls": len(rows)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="exp_22 R2-C capture polls / capture check")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("capture-polls", "check-capture"):
        p = sub.add_parser(name)
        p.add_argument("--card", required=True, choices=("GPU0", "GPU1"))
        p.add_argument("--record", required=True)
    sub.choices["check-capture"].add_argument("--sha256", required=True)
    sub.choices["check-capture"].add_argument("--fresh", action="store_true", help="refuse a last poll > FRESH_S old")
    args = ap.parse_args(argv)
    C = common()
    if args.cmd == "capture-polls":
        try:
            rc, info = capture_polls(C.World(C.NVIDIA_SMI), C.registered_config(), args.card, args.record)
        except FileExistsError:
            print(f"CAPTURE POLLS REFUSED: the capture record {args.record} exists (never overwritten)")
            return 1
        print(f"CAPTURE POLLS OK card {info['card']} first {info['first']!r} last {info['last']!r} polls {info['polls']}"
              if rc == 0 else f"CAPTURE POLLS REFUSED: {info}")
        return rc
    try:
        with open(args.record, "rb") as f:
            raw = f.read()
    except OSError as err:
        print(f"CAPTURE REFUSED: the capture record is unreadable ({err})")
        return 1
    why, info = capture_reasons(raw, args.card) if sha256_bytes(raw) == args.sha256 else (
        [f"the capture record hashes to {sha256_bytes(raw)}, not the sealed sha256 {args.sha256}"], None)
    age = time.time() - info["last"] if info else None
    if info and args.fresh and not 0 <= age <= C.FRESH_S:
        why = [f"the last poll is {age:.1f} s old: stale (> {C.FRESH_S} s); capture again"]
    print("CAPTURE REFUSED: " + "; ".join(why) if why else
          f"CAPTURE OK card {info['card']} uuid {info['uuid']} index {info['index']} first {info['first']!r} "
          f"last {info['last']!r} polls {info['polls']} age {age:.1f}")
    return 1 if why else 0


if __name__ == "__main__":
    sys.exit(main())
