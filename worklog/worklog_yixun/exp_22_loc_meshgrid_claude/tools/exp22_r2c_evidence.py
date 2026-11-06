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
* Writers: an O_EXCL record (never overwritten; file and directory fsynced) and the atomic state write (temp,
  fsync, rename, directory fsync).

Written by the exp_22 A7 Coder (Claude Opus 5.5, max effort).
"""
import hashlib
import json
import os
import re
import sys
import tempfile
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


def fsync_dir(d):
    fd = os.open(d, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_exclusive(path, raw):
    """O_CREAT | O_EXCL (a symlink or any existing name refuses): never overwritten; the file and its directory
    fsynced; returns the sha256 of the bytes written."""
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        view = memoryview(raw)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    fsync_dir(os.path.dirname(os.path.abspath(path)))
    return sha256_bytes(raw)


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
