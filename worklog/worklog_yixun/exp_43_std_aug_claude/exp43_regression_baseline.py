#!/usr/bin/env python3
"""exp_43 plan §4.2 — regression baseline of the full CPU test suite (``record``) and the
post-change acceptance check against it (``compare``).

    record  --junit J.xml --terminal-log L.log --commit SHA40 --worktree W --command "..." --out B.json
            [--python PY]
    compare --baseline B.json --commit SHA40 --junit J.xml --terminal-log L.log --worktree W
            [--new-test-file REL.py]... [--edited-node NODEID]... [--allow-new-skip NODEID]... --out C.json

Exit codes: 0 = record written / compare PASS; 1 = compare FAIL (report written, reasons listed);
2 = refused (nothing written).

Inputs are one pytest run: its JUnit XML (xunit2, the pytest 9.1.1 default) AND its teed terminal
output made with ``-rA``. Both are needed: JUnit writes a non-strict XPASS as a plain pass, so
``xpassed`` is visible only in the terminal's short test summary. The two are cross-checked node by
node and against both sets of totals; any disagreement is a refusal.

Node ids are rebuilt from JUnit's ``classname`` + ``name`` (pytest's ``mangle_test_address``
inverse): the longest dotted prefix of ``classname`` that is an existing ``<prefix>.py`` file in the
worktree is the module, the rest are classes. Two candidate modules → refused (ambiguous); none →
refused. An empty ``classname`` is a collection-level report whose ``name`` is the module.

Outcome per node: ``error`` if it has an <error> element, ``failed`` if a <failure>, ``xfailed`` if a
<skipped type="pytest.xfail">, ``skipped`` if another <skipped>, else ``xpassed`` when the terminal
summary says XPASS, else ``passed``. The one duplicate pytest writes on purpose — a call <failure>
followed by a separate testcase holding the teardown <error> — is merged into one ``failed`` node
(``double_reported``); every other duplicate node id is refused.

Error signature of every <failure>/<error>: ``phase`` (call / setup / teardown / collection),
``exc_type`` (the exception named by the last ``path:line: Type`` line of the report text, else the
``Type:`` prefix of the message, else ``<unknown>``) and the NORMALISED message (JUnit ``message``
attribute; the ``failed on setup/teardown with "..."`` wrapper is removed; for collection errors the
``E   `` lines of the text). Normalisation, applied in this order (``rules_description()``; its
sha256 is stored in every record and ``compare`` refuses a baseline made with other rules):
  ansi              ANSI colour escapes removed
  roots             the worktree (as given and its realpath) -> <REPO>; the interpreter's
                    sys.prefix (and realpath) -> <PYENV>; the home directory -> <HOME>; longest first,
                    only at a path-component boundary (first, so a worktree under /tmp is still <REPO>)
  pytest_tmp        /tmp/pytest-of-<user>/pytest-<n|current>        -> <PYTEST_TMP>
  pytest_tmp_n      the trailing counter of the next path component    -> <N>   (test_x0 -> test_x<N>)
  tmp_entry         /tmp/<entry>, /var/tmp/<entry>, /dev/shm/<entry>   -> <TMP>
  abs_path          any other absolute path                             -> <ABS>/<basename>
  hex               0x + >= 4 hex digits                                -> 0x<HEX>
  timestamp         ISO-like date + time                                -> <TIMESTAMP>
  pid               pid / PID / Pid / process / Process + number        -> ... <PID>
  duration_clock    H:MM:SS[.f]                                         -> <DUR>
  duration          number + ns|us|µs|ms|s|sec(s)|second(s)|min(s)|minute(s)|h|hr(s)|hour(s) -> <DUR>
  line_ref          <.ext>:<number>                                     -> <.ext>:<L>
  line_word         line / lineno / Line + number                       -> ... <L>
  whitespace        trailing whitespace of every line and of the text removed
Skip and xfail reasons are normalised the same way.

compare verdict PASS iff all of: no baseline node is missing; every node not in the baseline belongs
to a declared ``--new-test-file`` (which must be absent from the baseline and collect >= 1 node);
every baseline node except the declared ``--edited-node`` s keeps its outcome, and failures / errors
keep their identical signature list, skips / xfails their identical normalised reason (so: no
pass -> fail/error, no new skip or xfail, every pre-existing failure still fails the same way); every
new-file node passes, except a node declared with ``--allow-new-skip`` may skip; every edited node
exists in both runs and passes in the new run; and the count of passing pytest-9 subtests is unchanged
(JUnit carries no per-subtest identity: a passing subtest only adds one 'passed' stat to its parent's
testcase, so JUnit tests= = testcase elements + the terminal's 'N subtests passed'; failed / skipped
subtests are refused because pytest merges them into the parent element).
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SCHEMA = "exp43-regression-baseline/1"
COMPARE_SCHEMA = "exp43-regression-compare/1"
OUTCOMES = ("passed", "failed", "error", "skipped", "xfailed", "xpassed")
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_PATH_STOP = r"\s/'\":,;()\[\]{}<>=|*?"


class Refusal(Exception):
    """A precondition failed; nothing is written."""


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
_RULES = [
    ("ansi", r"\x1b\[[0-9;]*[A-Za-z]", "", "ANSI colour escapes removed"),
    ("roots", None, None, "worktree -> <REPO>, interpreter prefix -> <PYENV>, home -> <HOME> (longest first)"),
    ("pytest_tmp", r"/tmp/pytest-of-[^/\s]+/pytest-(?:\d+|current)", "<PYTEST_TMP>",
     "pytest's per-session base temp dir"),
    ("pytest_tmp_n", r"(<PYTEST_TMP>/[^/\s'\":]*?)\d+(?=[/\s'\":]|$)", r"\1<N>",
     "the counter pytest appends to a test's tmp_path component"),
    ("tmp_entry", r"(?<![\w.<>])/(?:tmp|var/tmp|dev/shm)/[^" + _PATH_STOP + r"]+", "<TMP>",
     "any first-level entry of a system temp directory"),
    ("abs_path", r"(?<![\w.<>:/~-])/(?:[^" + _PATH_STOP + r"]+/)+([^" + _PATH_STOP + r"]*)", r"<ABS>/\1",
     "any other absolute path, basename kept"),
    ("hex", r"\b0x[0-9a-fA-F]{4,}\b", "0x<HEX>", "hex addresses"),
    ("timestamp", r"\b\d{4}-\d{2}-\d{2}[ T_]\d{2}[:\-]\d{2}[:\-]\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?",
     "<TIMESTAMP>", "wall-clock timestamps"),
    ("pid", r"\b(pid|PID|Pid|process|Process)(\s*[=:#]?\s*)\d+\b", r"\1\2<PID>", "process ids"),
    ("duration_clock", r"\b\d+:\d{2}:\d{2}(?:\.\d+)?\b", "<DUR>", "H:MM:SS timings"),
    ("duration", r"\b\d+(?:\.\d+)?\s?(?:ns|us|µs|ms|s|secs?|seconds?|mins?|minutes?|h|hrs?|hours?)\b",
     "<DUR>", "timings with a unit"),
    ("line_ref", r"(\.[A-Za-z0-9_]+):\d+\b", r"\1:<L>", "file.ext:line references"),
    ("line_word", r"\b(line|lineno|Line)(\s*[=:]?\s*)\d+\b", r"\1\2<L>", "'line N' references"),
    ("whitespace", None, None, "trailing whitespace of every line and of the text removed"),
]
_COMPILED = [(n, re.compile(p) if p else None, r) for n, p, r, _d in _RULES]


def rules_description():
    return [{"name": n, "pattern": p, "replacement": r, "description": d} for n, p, r, d in _RULES]


def rules_sha256():
    return hashlib.sha256(json.dumps(rules_description(), sort_keys=True).encode()).hexdigest()


def path_roots(worktree):
    """The substitution table for the 'roots' step: token -> distinct absolute spellings."""
    roots = {"<REPO>": [], "<PYENV>": [], "<HOME>": []}
    if worktree is not None:
        roots["<REPO>"] = [os.path.abspath(worktree), os.path.realpath(worktree)]
    roots["<PYENV>"] = [os.path.abspath(sys.prefix), os.path.realpath(sys.prefix)]
    home = os.path.expanduser("~")
    roots["<HOME>"] = [os.path.abspath(home), os.path.realpath(home)]
    return {k: sorted({p.rstrip("/") for p in v if p and p != "/"}) for k, v in roots.items()}


def _apply_roots(text, roots):
    pairs = sorted(((p, tok) for tok, ps in roots.items() for p in ps), key=lambda x: -len(x[0]))
    for p, tok in pairs:
        text = re.sub(re.escape(p) + r"(?=[" + _PATH_STOP.replace("<>", "") + r"]|$)", tok, text)
    return text


def normalise(text, roots):
    text = str(text)
    for name, rx, rep in _COMPILED:
        if name == "roots":
            text = _apply_roots(text, roots)
        elif name == "whitespace":
            text = "\n".join(line.rstrip() for line in text.split("\n")).strip()
        else:
            text = rx.sub(rep, text)
    return text


_LOC_TYPE = re.compile(r"^\S.*?:\d+: ([A-Za-z_][\w.]*)\s*$", re.M)
_MSG_TYPE = re.compile(r"^([A-Za-z_][\w.]*):(?:\s|$)")
_WRAP = re.compile(r'^failed on (setup|teardown) with "(.*)"$', re.S)


def signature(tag, message, text, roots):
    """Normalised (phase, exc_type, message) of one <failure> / <error> element."""
    message = message or ""
    text = text or ""
    if tag == "failure":
        phase, msg = "call", message
    elif tag == "error":
        m = _WRAP.match(message)
        if m:
            phase, msg = m.group(1), m.group(2)
        elif message == "collection failure":
            phase = "collection"
            msg = "\n".join(line[1:].strip() for line in text.split("\n") if line.startswith("E "))
        else:
            raise Refusal(f"unrecognised <error> message format: {message[:120]!r}")
    else:
        raise Refusal(f"not a failure/error element: {tag}")
    locs = _LOC_TYPE.findall(text)
    if locs:
        exc = locs[-1]
    else:
        m = _MSG_TYPE.match(msg.strip())
        exc = m.group(1) if m else "<unknown>"
    sig = {"phase": phase, "exc_type": exc, "message": normalise(msg, roots)}
    sig["sha256"] = hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()
    return sig


# --------------------------------------------------------------------------- #
# node ids
# --------------------------------------------------------------------------- #
_IDENT = re.compile(r"^[A-Za-z_][\w]*$")


def resolve_nodeid(classname, name, worktree):
    """Inverse of pytest's mangle_test_address, resolved against the worktree's files."""
    wt = Path(worktree)
    if classname == "pytest" and name == "internal":
        raise Refusal("pytest reported an internal error")
    if not name or "\n" in name:
        raise Refusal(f"malformed test name {name!r} (classname {classname!r})")
    if classname == "":
        parts, tail = name.split("."), []
    else:
        parts = classname.split(".")
        base, sep, params = name.partition("[")
        if not _IDENT.match(base) or (sep and not params.endswith("]")):
            raise Refusal(f"malformed test name {name!r} (classname {classname!r})")
        tail = [name]
    if not all(_IDENT.match(p) for p in parts):
        raise Refusal(f"malformed classname {classname!r} (name {name!r})")
    hits = [i for i in range(len(parts), 0, -1) if (wt / ("/".join(parts[:i]) + ".py")).is_file()]
    if not hits:
        raise Refusal(f"no module in the worktree for classname {classname!r} name {name!r}")
    if len(hits) > 1:
        raise Refusal(f"ambiguous module for classname {classname!r}: "
                      + ", ".join("/".join(parts[:i]) + ".py" for i in hits))
    i = hits[0]
    if classname == "" and i != len(parts):
        raise Refusal(f"collection-level report {name!r} does not name a module")
    return "::".join(["/".join(parts[:i]) + ".py", *parts[i:], *tail])


# --------------------------------------------------------------------------- #
# JUnit + terminal log -> one run
# --------------------------------------------------------------------------- #
_CASE_CHILDREN = {"properties", "failure", "error", "skipped", "system-out", "system-err"}


def _int_attr(suite, key):
    v = suite.get(key)
    if v is None or not re.fullmatch(r"\d+", v):
        raise Refusal(f"JUnit testsuite attribute {key}={v!r} is not a count")
    return int(v)


def _elements(path, worktree, roots, subtests_passed):
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        raise Refusal(f"JUnit XML does not parse: {e}")
    if root.tag == "testsuites":
        suites = [c for c in root if c.tag == "testsuite"]
        if len(suites) != 1 or len(root) != 1:
            raise Refusal("JUnit XML must hold exactly one <testsuite>")
        suite = suites[0]
    elif root.tag == "testsuite":
        suite = root
    else:
        raise Refusal(f"JUnit root element is <{root.tag}>")
    elements, n_el = [], {"failure": 0, "error": 0, "skipped": 0}
    for tc in suite:
        if tc.tag == "properties":
            continue
        if tc.tag != "testcase":
            raise Refusal(f"unexpected <{tc.tag}> in the testsuite")
        kids = list(tc)
        bad = {k.tag for k in kids} - _CASE_CHILDREN
        if bad:
            raise Refusal(f"unexpected children {sorted(bad)} in a testcase")
        for k in kids:
            if k.tag in n_el:
                n_el[k.tag] += 1
        fails = [k for k in kids if k.tag == "failure"]
        errs = [k for k in kids if k.tag == "error"]
        skips = [k for k in kids if k.tag == "skipped"]
        if len(fails) > 1 or len(skips) > 1 or (fails and skips):
            raise Refusal(f"testcase {tc.get('classname')}::{tc.get('name')} has an impossible child set")
        nodeid = resolve_nodeid(tc.get("classname", ""), tc.get("name", ""), worktree)
        el = {"nodeid": nodeid, "signatures": [], "skip": None}
        for k in fails + errs:
            el["signatures"].append(signature(k.tag, k.get("message"), k.text, roots))
            el.setdefault("raw", []).append((k.get("message") or "")[:2000])
        if skips:
            k = skips[0]
            typ, msg = k.get("type"), k.get("message") or ""
            if typ == "pytest.xfail":
                el["skip"] = ("xfailed", msg)
            elif typ == "pytest.skip":
                el["skip"] = ("skipped", msg)
            elif typ is None and msg == "collection skipped":
                el["skip"] = ("skipped", k.text or msg)
            else:
                raise Refusal(f"unrecognised <skipped> element (type={typ!r} message={msg[:80]!r}) on {nodeid}")
        el["kinds"] = (bool(fails), [s["phase"] for s in el["signatures"] if s["phase"] != "call"])
        elements.append(el)
    totals = {k: _int_attr(suite, k) for k in ("tests", "failures", "errors", "skipped")}
    # pytest 9 counts every passing subtest as one more 'passed' stat of its parent's testcase
    # element, so JUnit tests= exceeds the element count by the terminal's 'N subtests passed'.
    seen = {"tests": len(elements) + subtests_passed, "failures": n_el["failure"], "errors": n_el["error"],
            "skipped": n_el["skipped"]}
    if totals != seen:
        raise Refusal(f"JUnit totals {totals} disagree with its own testcases {seen} "
                      f"({len(elements)} testcase elements + {subtests_passed} passing subtests)")
    meta = {**totals, "time": suite.get("time"), "timestamp": suite.get("timestamp"),
            "hostname": suite.get("hostname"), "name": suite.get("name")}
    return elements, meta


def _merge(elements):
    nodes, order = {}, []
    for el in elements:
        nid = el["nodeid"]
        if nid not in nodes:
            nodes[nid] = dict(el, double_reported=False)
            order.append(nid)
            continue
        prev = nodes[nid]
        has_fail, others = prev["kinds"]
        new_fail, new_others = el["kinds"]
        if (prev["double_reported"] or not has_fail or others or prev["skip"]
                or new_fail or new_others != ["teardown"] or el["skip"]):
            raise Refusal(f"duplicate node id {nid} (not pytest's call-failure + teardown-error pair)")
        prev["signatures"] = prev["signatures"] + el["signatures"]
        prev["raw"] = prev.get("raw", []) + el.get("raw", [])
        prev["kinds"] = (True, ["teardown"])
        prev["double_reported"] = True
    return nodes, order


_SEP = re.compile(r"^=+ short test summary info =+$")
_FINAL = re.compile(r"^(?:=+ )?(?P<body>\d+ [a-z]+(?: [a-z]+)?(?:, \d+ [a-z]+(?: [a-z]+)?)*) in (?P<secs>[\d.]+)s"
                    r"(?: \((?P<clock>[\d:.]+)\))?(?: =+)?$")
_WORDS = {"failed": "failed", "passed": "passed", "skipped": "skipped", "xfailed": "xfailed",
          "xpassed": "xpassed", "error": "error", "errors": "error", "deselected": "deselected",
          "warning": "warnings", "warnings": "warnings", "rerun": "rerun",
          "subtests passed": "subtests_passed"}
_KINDS = ("PASSED", "FAILED", "ERROR", "XFAIL", "XPASS", "SKIPPED")
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _summary_node(rest, known):
    cands = [rest] + [rest[:m.start()] for m in re.finditer(r" - ", rest)]
    hits = [c for c in cands if c in known]
    if not hits:
        raise Refusal(f"terminal summary names a node the JUnit XML does not hold: {rest[:160]!r}")
    return max(hits, key=len)


def _totals_line(path):
    lines = [_ANSI.sub("", ln.rstrip("\n")) for ln in Path(path).read_text(errors="replace").split("\n")]
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        raise Refusal("terminal log is empty")
    final = lines[-1].strip()
    m = _FINAL.match(final)
    if not m:
        raise Refusal(f"terminal log does not end with pytest's totals line: {final[:160]!r}")
    counts = {}
    for part in m.group("body").split(", "):
        n, word = part.split(" ", 1)
        if word not in _WORDS:
            raise Refusal(f"unknown or unsupported word {word!r} in the totals line")
        counts[_WORDS[word]] = counts.get(_WORDS[word], 0) + int(n)
    return lines, counts, final


_SUB = re.compile(r"^SUB(PASSED|FAILED|SKIPPED|XFAIL)\b")


def _terminal(lines, known):
    seps = [i for i, ln in enumerate(lines) if _SEP.match(ln.strip())]
    if not seps:
        raise Refusal("terminal log has no 'short test summary info' section (run pytest with -rA)")
    entries = {k: [] for k in _KINDS}
    n_skipped, last = 0, None
    for ln in lines[seps[-1] + 1:-1]:
        if _SUB.match(ln):
            raise Refusal(f"failed / skipped / xfailed subtest reports are not supported: {ln[:160]!r}")
        word, _sp, rest = ln.partition(" ")
        if word in _KINDS and rest:
            if word == "SKIPPED":
                sm = re.match(r"^\[(\d+)\] ", rest)
                if not sm:
                    raise Refusal(f"malformed SKIPPED summary line {ln[:160]!r}")
                n_skipped += int(sm.group(1))
            else:
                entries[word].append(_summary_node(rest, known))
            last = word
        elif last in ("FAILED", "ERROR", "XFAIL", "XPASS"):
            continue                                    # continuation of a multi-line message
        else:
            raise Refusal(f"unexpected line in the short test summary: {ln[:160]!r}")
    return entries, n_skipped


def _counter(xs):
    c = {}
    for x in xs:
        c[x] = c.get(x, 0) + 1
    return c


def parse_run(junit, terminal_log, worktree):
    """One pytest run → {nodes: {nodeid: {...}}, totals, junit_meta, terminal}; refuses on any
    inconsistency between the two files."""
    roots = path_roots(worktree)
    lines, counts, final = _totals_line(terminal_log)
    subtests_passed = counts.get("subtests_passed", 0)
    elements, meta = _elements(junit, worktree, roots, subtests_passed)
    nodes, order = _merge(elements)
    entries, n_skipped = _terminal(lines, set(nodes))
    plain = {n for n, e in nodes.items() if not e["signatures"] and not e["skip"]}
    td_only = {n for n, e in nodes.items() if e["kinds"] == (False, ["teardown"]) and not e["skip"]}
    xpass = _counter(entries["XPASS"])
    if any(v > 1 for v in xpass.values()) or not set(xpass) <= plain:
        raise Refusal("XPASS summary lines disagree with the JUnit XML")
    want = {
        "PASSED": _counter((plain - set(xpass)) | td_only),
        "FAILED": _counter(n for n, e in nodes.items() if e["kinds"][0]),
        "ERROR": _counter(n for n, e in nodes.items() for p in e["kinds"][1]),
        "XFAIL": _counter(n for n, e in nodes.items() if e["skip"] and e["skip"][0] == "xfailed"),
    }
    for kind, exp in want.items():
        if _counter(entries[kind]) != exp:
            raise Refusal(f"terminal {kind} lines disagree with the JUnit XML")
    junit_skips = sum(1 for e in nodes.values() if e["skip"] and e["skip"][0] == "skipped")
    if n_skipped != junit_skips:
        raise Refusal(f"terminal SKIPPED count {n_skipped} disagrees with the JUnit XML ({junit_skips})")
    term_expect = {"failed": len(entries["FAILED"]), "passed": len(entries["PASSED"]),
                   "error": len(entries["ERROR"]), "skipped": n_skipped,
                   "xfailed": len(entries["XFAIL"]), "xpassed": len(entries["XPASS"])}
    got = {k: counts.get(k, 0) for k in term_expect}
    if got != term_expect:
        raise Refusal(f"terminal totals line {got} disagrees with its own summary / the JUnit XML {term_expect}")
    out = {}
    for nid in sorted(order):
        e = nodes[nid]
        if e["kinds"][1]:
            outcome = "failed" if e["double_reported"] else "error"
        elif e["kinds"][0]:
            outcome = "failed"
        elif e["skip"]:
            outcome = e["skip"][0]
        else:
            outcome = "xpassed" if nid in xpass else "passed"
        rec = {"outcome": outcome}
        if e["signatures"]:
            rec["signatures"] = e["signatures"]
        if e["skip"]:
            rec["reason"] = normalise(e["skip"][1], roots)
        if e["double_reported"]:
            rec["double_reported"] = True
        out[nid] = (rec, e.get("raw", []), e["skip"][1] if e["skip"] else None)
    totals = {k: 0 for k in OUTCOMES}
    for rec, _r, _s in out.values():
        totals[rec["outcome"]] += 1
    return {"nodes": {k: v[0] for k, v in out.items()}, "raw": {k: (v[1], v[2]) for k, v in out.items()},
            "totals": totals, "junit_meta": meta, "terminal": {"final_line": final, "counts": counts},
            "roots": roots, "subtests_passed": subtests_passed}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_sha256(rec):
    body = {k: v for k, v in rec.items() if k != "content_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def _git_head(worktree):
    return subprocess.run(["git", "-C", str(worktree), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()


def _git_tracked_clean(worktree):
    out = subprocess.run(["git", "-C", str(worktree), "status", "--porcelain", "--untracked-files=no"],
                         capture_output=True, text=True, check=True).stdout
    return out.strip() == ""


_PROBE = ("import json, sys, pytest, torch; print(json.dumps({'python': sys.version.split()[0], "
          "'pytest': pytest.__version__, 'torch': torch.__version__, 'executable': sys.executable}))")


def _probe_versions(python):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    out = subprocess.run([python, "-I", "-c", _PROBE], capture_output=True, text=True, check=True, env=env)
    return json.loads(out.stdout.strip().splitlines()[-1])


def _write_new(path, obj):
    with open(path, "x") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")


def _file_block(path):
    return {"path": os.path.abspath(path), "sha256": _sha256(path)}


def _tool_block():
    return {"path": os.path.abspath(__file__), "sha256": _sha256(__file__)}


# --------------------------------------------------------------------------- #
# record
# --------------------------------------------------------------------------- #
def cmd_record(a):
    if not _SHA40.match(a.commit):
        raise Refusal(f"--commit must be the full 40-hex sha (got {a.commit!r})")
    if os.path.lexists(a.out):
        raise Refusal(f"refusing to overwrite {a.out}")
    wt = os.path.abspath(a.worktree)
    head = _git_head(wt)
    if head != a.commit:
        raise Refusal(f"worktree HEAD {head} != --commit {a.commit}")
    if not _git_tracked_clean(wt):
        raise Refusal("the worktree's tracked tree is not clean")
    run = parse_run(a.junit, a.terminal_log, wt)
    failures = []
    for nid, rec in run["nodes"].items():
        if rec["outcome"] in ("failed", "error"):
            raw_msgs, _ = run["raw"][nid]
            failures.append({"nodeid": nid, "outcome": rec["outcome"], "signatures": rec["signatures"],
                             "raw_messages": raw_msgs})
    rec = {
        "schema": SCHEMA, "commit": a.commit, "worktree": wt, "worktree_head": head,
        "tracked_tree_clean": True, "command": a.command, "versions": _probe_versions(a.python),
        "junit": {**_file_block(a.junit), "testsuite": run["junit_meta"]},
        "terminal_log": {**_file_block(a.terminal_log), **run["terminal"]},
        "collection_count": len(run["nodes"]), "totals": run["totals"],
        "subtests_passed": run["subtests_passed"], "nodes": run["nodes"],
        "failures": failures,
        "skips": {n: r["reason"] for n, r in run["nodes"].items() if r["outcome"] in ("skipped", "xfailed")},
        "normalisation": {"rules": rules_description(), "rules_sha256": rules_sha256(), "roots": run["roots"]},
        "tool": _tool_block(),
    }
    rec["content_sha256"] = content_sha256(rec)
    _write_new(a.out, rec)
    print(f"recorded {rec['collection_count']} nodes {rec['totals']} -> {a.out}")
    return 0


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #
_NODEID = re.compile(r"^[^\s:/][^\s:]*\.py(::[A-Za-z_]\w*(\[.*\])?)+$")


def _check_rel(p):
    if os.path.isabs(p) or not p.endswith(".py") or any(x in ("", ".", "..") for x in p.split("/")):
        raise Refusal(f"--new-test-file must be a normalised repo-relative .py path (got {p!r})")
    return p


def _in_files(nid, files):
    return any(nid == f or nid.startswith(f + "::") for f in files)


def _load_baseline(path, commit):
    try:
        b = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise Refusal(f"baseline unreadable: {e}")
    if not isinstance(b, dict) or b.get("schema") != SCHEMA:
        raise Refusal("baseline schema mismatch")
    if b.get("content_sha256") != content_sha256(b):
        raise Refusal("baseline content_sha256 does not verify (edited after recording?)")
    if b.get("commit") != commit:
        raise Refusal(f"baseline commit {b.get('commit')} != --commit {commit}")
    if b.get("normalisation", {}).get("rules_sha256") != rules_sha256():
        raise Refusal("baseline was normalised with other rules")
    return b


def _strip(sigs):
    return [(s["phase"], s["exc_type"], s["message"]) for s in sigs or []]


def compare(base, new, new_files, edited, allow_skip):
    reasons = []
    bn, nn = base["nodes"], new["nodes"]
    for f in new_files:
        if any(_in_files(n, [f]) for n in bn):
            reasons.append(f"declared new test file {f} already has nodes in the baseline")
        if not any(_in_files(n, [f]) for n in nn):
            reasons.append(f"declared new test file {f} collected no node")
    if base.get("subtests_passed", 0) != new.get("subtests_passed", 0):
        reasons.append(f"passing subtests changed: {base.get('subtests_passed', 0)} -> {new.get('subtests_passed', 0)} "
                       "(not attributable to a node id)")
    removed = sorted(set(bn) - set(nn))
    added = sorted(set(nn) - set(bn))
    for n in removed:
        reasons.append(f"baseline node missing: {n}")
    for n in added:
        if not _in_files(n, new_files):
            reasons.append(f"new node not in a declared new test file: {n}")
    baseline_failures = {}
    for n in sorted(set(bn) & set(nn)):
        if n in edited:
            continue
        b, w = bn[n], nn[n]
        same = b["outcome"] == w["outcome"]
        if b["outcome"] in ("failed", "error"):
            same = same and _strip(b.get("signatures")) == _strip(w.get("signatures"))
            baseline_failures[n] = "unchanged" if same else "changed"
            if not same:
                reasons.append(f"pre-existing {b['outcome']} changed: {n} ({b['outcome']} -> {w['outcome']}"
                               + (", signature differs)" if b["outcome"] == w["outcome"] else ")"))
        elif b["outcome"] in ("skipped", "xfailed"):
            if not (same and b.get("reason") == w.get("reason")):
                reasons.append(f"pre-existing {b['outcome']} changed: {n} -> {w['outcome']} {w.get('reason', '')!r}")
        elif not same:
            label = {"failed": "pass -> fail", "error": "pass -> error", "skipped": "new skip",
                     "xfailed": "new xfail"}.get(w["outcome"], "outcome changed")
            reasons.append(f"{label}: {n} ({b['outcome']} -> {w['outcome']})")
    allowed = {}
    for n in added:
        if not _in_files(n, new_files):
            continue
        o = nn[n]["outcome"]
        if o == "skipped" and n in allow_skip:
            allowed[n] = nn[n].get("reason")
        elif o != "passed":
            reasons.append(f"new-file node not passing ({o}"
                           + (", undeclared new skip" if o == "skipped" else "") + f"): {n}")
    edited_rep = {}
    for n in sorted(edited):
        rep = {"baseline": bn.get(n, {}).get("outcome"), "new": nn.get(n, {}).get("outcome")}
        rep["ok"] = rep["baseline"] is not None and rep["new"] == "passed"
        edited_rep[n] = rep
        if rep["baseline"] is None:
            reasons.append(f"edited node not in the baseline: {n}")
        elif rep["new"] != "passed":
            reasons.append(f"edited node does not pass: {n} ({rep['new']})")
    return {"verdict": "PASS" if not reasons else "FAIL", "reasons": reasons, "removed_nodes": removed,
            "added_nodes": added, "baseline_failures": baseline_failures, "edited_nodes": edited_rep,
            "allowed_new_skips": allowed}


def cmd_compare(a):
    if not _SHA40.match(a.commit):
        raise Refusal(f"--commit must be the full 40-hex sha (got {a.commit!r})")
    if os.path.lexists(a.out):
        raise Refusal(f"refusing to overwrite {a.out}")
    new_files = [_check_rel(p) for p in a.new_test_file]
    for n in a.edited_node + a.allow_new_skip:
        if not _NODEID.match(n):
            raise Refusal(f"malformed node id {n!r}")
    for n in a.edited_node:
        if _in_files(n, new_files):
            raise Refusal(f"edited node {n} belongs to a declared new test file")
    for n in a.allow_new_skip:
        if not _in_files(n, new_files):
            raise Refusal(f"--allow-new-skip {n} is not in a declared new test file")
    base = _load_baseline(a.baseline, a.commit)
    wt = os.path.abspath(a.worktree)
    new = parse_run(a.junit, a.terminal_log, wt)
    res = compare(base, new, new_files, set(a.edited_node), set(a.allow_new_skip))
    try:
        head = _git_head(wt)
    except (OSError, subprocess.CalledProcessError):
        head = None
    rep = {"schema": COMPARE_SCHEMA, **res,
           "baseline": {"path": os.path.abspath(a.baseline), "sha256": _sha256(a.baseline),
                        "commit": base["commit"], "content_sha256": base["content_sha256"],
                        "totals": base["totals"], "collection_count": base["collection_count"]},
           "new": {"worktree": wt, "worktree_head": head, "junit": _file_block(a.junit),
                   "terminal_log": {**_file_block(a.terminal_log), **new["terminal"]},
                   "totals": new["totals"], "collection_count": len(new["nodes"]),
                   "subtests_passed": new["subtests_passed"]},
           "declared": {"new_test_files": new_files, "edited_nodes": sorted(a.edited_node),
                        "allow_new_skip": sorted(a.allow_new_skip)},
           "new_failures": {n: r for n, r in new["nodes"].items() if r["outcome"] in ("failed", "error")},
           "tool": _tool_block()}
    _write_new(a.out, rep)
    print(f"verdict {rep['verdict']}" + "".join(f"\n  - {r}" for r in rep["reasons"]))
    return 0 if rep["verdict"] == "PASS" else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--junit", required=True)
    r.add_argument("--terminal-log", required=True)
    r.add_argument("--commit", required=True)
    r.add_argument("--worktree", required=True)
    r.add_argument("--command", required=True)
    r.add_argument("--python", default=sys.executable)
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--baseline", required=True)
    c.add_argument("--commit", required=True)
    c.add_argument("--junit", required=True)
    c.add_argument("--terminal-log", required=True)
    c.add_argument("--worktree", required=True)
    c.add_argument("--new-test-file", action="append", default=[])
    c.add_argument("--edited-node", action="append", default=[])
    c.add_argument("--allow-new-skip", action="append", default=[])
    c.add_argument("--out", required=True)
    a = p.parse_args(argv)
    try:
        return cmd_record(a) if a.cmd == "record" else cmd_compare(a)
    except Refusal as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
