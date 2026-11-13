"""exp_43 plan §4.2 — the regression-baseline tool, test first.

The tool (``worklog/worklog_yixun/exp_43_std_aug_claude/exp43_regression_baseline.py``) is loaded by
path, as ``test_exp40_kit.py`` loads its kit. Every input is synthetic and lives under ``tmp_path``:
a fake worktree (only the files node-id resolution needs), pytest-shaped JUnit XML (xunit2, the
pytest 9.1.1 default) and a pytest-shaped ``-rA`` terminal log. Git and the interpreter-version probe
are monkeypatched; nothing here runs git or touches a real checkout.
"""
import hashlib
import importlib.util
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TOOL = _REPO / "worklog/worklog_yixun/exp_43_std_aug_claude/exp43_regression_baseline.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rb = _load("exp43_regression_baseline", _TOOL)

SHA = "6a51f7413a91a507badb7d8a795936d107ef63cb"
OTHER = "e1288a92f7536e579840fdfab255f3120eaf9605"
CMD = 'CUDA_VISIBLE_DEVICES="" python -m pytest -q src/tests -rf -rA --junitxml=j.xml'
VERSIONS = {"python": "3.10.18", "pytest": "9.1.1", "torch": "2.7.0", "executable": "/x/python"}


# --------------------------------------------------------------------------- #
# synthetic inputs
# --------------------------------------------------------------------------- #
def _worktree(root: Path, extra=()):
    for rel in ("src/tests/test_alpha.py", "src/tests/test_beta.py", "src/tests/test_new.py", *extra):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# synthetic\n")
    return root


def F(msg, text):
    return ("failure", {"message": msg}, text)


def E(msg, text):
    return ("error", {"message": msg}, text)


def S(reason):
    return ("skipped", {"type": "pytest.skip", "message": reason}, f"src/tests/test_beta.py:3: {reason}")


def X(reason):
    return ("skipped", {"type": "pytest.xfail", "message": reason}, None)


def write_junit(path, cases, **override):
    n_el = {"failure": 0, "error": 0, "skipped": 0}
    for _c, _n, children in cases:
        for tag, _a, _t in children:
            n_el[tag] += 1
    attrs = {"name": "pytest", "errors": str(n_el["error"]), "failures": str(n_el["failure"]),
             "skipped": str(n_el["skipped"]), "tests": str(len(cases)), "time": "12.345",
             "timestamp": "2026-11-13T01:04:33.000000-05:00", "hostname": "box"}
    attrs.update({k: str(v) for k, v in override.items()})
    suites = ET.Element("testsuites")
    suite = ET.SubElement(suites, "testsuite", attrs)
    for classname, name, children in cases:
        tc = ET.SubElement(suite, "testcase", {"classname": classname, "name": name, "time": "0.010"})
        for tag, a, text in children:
            el = ET.SubElement(tc, tag, a)
            if text is not None:
                el.text = text
        ET.SubElement(tc, "system-out").text = "captured"
    Path(path).write_bytes(b'<?xml version="1.0" encoding="utf-8"?>' + ET.tostring(suites))
    return Path(path)


def write_log(path, summary, final):
    body = ["." * 72 + " [ 50%]", "....F.E" + " [100%]", "=" * 20 + " PASSES " + "=" * 20,
            "____ test_ok ____", "captured stdout", "=" * 30 + " short test summary info " + "=" * 30,
            *summary, final]
    Path(path).write_text("\n".join(body) + "\n")
    return Path(path)


A = "src.tests.test_alpha"
B = "src.tests.test_beta"
N_OK = "src/tests/test_alpha.py::test_ok"
N_PARAM = "src/tests/test_alpha.py::TestGroup::test_param[a.b-1]"
N_XP = "src/tests/test_alpha.py::test_xp"
N_FAIL = "src/tests/test_beta.py::test_fail"
N_ERR = "src/tests/test_beta.py::test_err"
N_SKIP = "src/tests/test_beta.py::test_skip"
N_XF = "src/tests/test_beta.py::test_xf"


def scenario(wt: Path, *, hexaddr="0x7f3a2b1c9d10", pid="4242", secs="3.21", line="12",
             tmpn="0", fail_type="AssertionError"):
    """Seven nodes, one of every outcome; machine-specific detail is parameterised so two runs
    from two worktrees differ only in what the normalisation removes."""
    fail_msg = (f"{fail_type}: {wt}/data/x.json missing; object at {hexaddr}; "
                f"tmp /tmp/pytest-of-yixunhu/pytest-71{tmpn}/test_fail{tmpn}/out.txt")
    fail_text = (f"def test_fail():\n>       assert False\nE       {fail_type}: boom\n\n"
                 f"src/tests/test_beta.py:{line}: {fail_type}")
    err_msg = f'failed on setup with "RuntimeError: pid {pid} timed out after {secs}s"'
    err_text = f"fixture setup\nE   RuntimeError: x\n\nsrc/tests/test_beta.py:{int(line) + 8}: RuntimeError"
    cases = [
        (A, "test_ok", ()),
        (A + ".TestGroup", "test_param[a.b-1]", ()),
        (A, "test_xp", ()),
        (B, "test_fail", (F(fail_msg, fail_text),)),
        (B, "test_err", (E(err_msg, err_text),)),
        (B, "test_skip", (S("needs a GPU"),)),
        (B, "test_xf", (X("known numerical issue"),)),
    ]
    summary = [f"PASSED {N_OK}", f"PASSED {N_PARAM}", "SKIPPED [1] src/tests/test_beta.py:3: needs a GPU",
               f"XFAIL {N_XF} - known numerical issue", f"XPASS {N_XP} - flaky marker",
               f"ERROR {N_ERR} - RuntimeError: pid {pid} timed...", f"FAILED {N_FAIL} - {fail_type}: ..."]
    final = f"1 failed, 2 passed, 1 skipped, 1 xfailed, 1 xpassed, 1 error in {secs}s (0:00:03)"
    return cases, summary, final


@pytest.fixture
def git_ok(monkeypatch):
    monkeypatch.setattr(rb, "_git_head", lambda wt: SHA)
    monkeypatch.setattr(rb, "_git_tracked_clean", lambda wt: True)
    monkeypatch.setattr(rb, "_probe_versions", lambda python: dict(VERSIONS))


def _inputs(tmp_path, name="run", wt=None, cases=None, summary=None, final=None, **kw):
    wt = wt or _worktree(tmp_path / f"wt_{name}")
    c, s, f = scenario(wt, **kw)
    j = write_junit(tmp_path / f"{name}.xml", cases if cases is not None else c)
    lg = write_log(tmp_path / f"{name}.log", summary if summary is not None else s, final or f)
    return wt, j, lg


def _record(tmp_path, wt, j, lg, out=None, commit=SHA):
    out = out or tmp_path / "baseline.json"
    rc = rb.main(["record", "--junit", str(j), "--terminal-log", str(lg), "--commit", commit,
                  "--worktree", str(wt), "--command", CMD, "--out", str(out)])
    return rc, out


def _baseline(tmp_path, **kw):
    wt, j, lg = _inputs(tmp_path, "base", **kw)
    rc, out = _record(tmp_path, wt, j, lg)
    assert rc == 0
    return out


def _compare(tmp_path, base, wt, j, lg, *extra, out_name="cmp.json", commit=SHA):
    out = tmp_path / out_name
    pre_existing = out.exists()
    rc = rb.main(["compare", "--baseline", str(base), "--commit", commit, "--junit", str(j),
                  "--terminal-log", str(lg), "--worktree", str(wt), *extra, "--out", str(out)])
    if rc == 2 and not pre_existing:
        assert not out.exists(), "a refusal must write nothing"
    return rc, (json.loads(out.read_text()) if rc in (0, 1) else None)


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw, expected", [
    ("at 0x7f3a2b1c9d10 end", "at 0x<HEX> end"),
    ("pid 4242 died", "pid <PID> died"),
    ("Process(pid=91) x", "Process(pid=<PID>) x"),
    ("took 3.21s", "took <DUR>"),
    ("after 30 seconds", "after <DUR>"),
    ("in 0:55:11 wall", "in <DUR> wall"),
    ("at 2026-11-13 01:04:33.123 now", "at <TIMESTAMP> now"),
    ("x.py:123: AssertionError", "x.py:<L>: AssertionError"),
    ('File "x.py", line 77, in f', 'File "x.py", line <L>, in f'),
    ("/tmp/pytest-of-bob/pytest-7142/test_case_three0/out", "<PYTEST_TMP>/test_case_three<N>/out"),
    ("/tmp/exp40.ST6IgE/wandb/run", "<TMP>/wandb/run"),
    ("/tmp/tmpab12cd_x/f.json", "<TMP>/f.json"),
    ("see /usr/lib/x86/libfoo.so.1 now", "see <ABS>/libfoo.so.1 now"),
    ("\x1b[31mred\x1b[0m", "red"),
    ("trailing   \nlines  ", "trailing\nlines"),
], ids=["hex", "pid", "pid_kwarg", "dur_s", "dur_word", "dur_clock", "timestamp", "line_ref", "line_word",
        "pytest_tmp", "tmp_entry", "tmp_mkdtemp", "abs_path", "ansi", "whitespace"])
def test_each_normalisation_rule(raw, expected):
    assert rb.normalise(raw, rb.path_roots(None)) == expected


def test_worktree_root_and_its_realpath_become_repo(tmp_path):
    real = _worktree(tmp_path / "real")
    link = tmp_path / "link"
    link.symlink_to(real)
    roots = rb.path_roots(link)
    assert rb.normalise(f"{link}/src/a.py and {real}/data/b.json", roots) == "<REPO>/src/a.py and <REPO>/data/b.json"
    # a sibling directory sharing the prefix is not the worktree
    assert "<REPO>" not in rb.normalise(f"{real}_other/x/y.py", roots)
    assert rb.normalise("/home/someone/other_repo/x/y.py", rb.path_roots(real)) == "<ABS>/y.py"


def test_interpreter_prefix_and_home_are_tokens():
    roots = rb.path_roots(None)
    assert rb.normalise(f"{sys.prefix}/lib/python3.10/site.py", roots) == "<PYENV>/lib/python3.10/site.py"
    home = str(Path.home())
    assert rb.normalise(f"{home}/.cache/hub/x", roots).startswith("<HOME>/")


def test_normalisation_is_idempotent_and_hash_is_stable():
    roots = rb.path_roots(None)
    raw = "E 0x7f3a2b1c9d10 pid 9 3.2s /tmp/pytest-of-u/pytest-3/t0/a.py:12 line 4 /a/b/c"
    once = rb.normalise(raw, roots)
    assert rb.normalise(once, roots) == once
    assert rb.rules_sha256() == rb.rules_sha256() and len(rb.rules_sha256()) == 64
    assert [r["name"] for r in rb.rules_description()][0] == "ansi"


def test_same_failure_from_two_worktrees_has_one_signature_and_type_matters(tmp_path):
    w1, w2 = _worktree(tmp_path / "w1"), _worktree(tmp_path / "w2")
    c1 = scenario(w1)[0][3][2][0]
    c2 = scenario(w2, hexaddr="0x55aa11223344", tmpn="9", line="99")[0][3][2][0]
    c3 = scenario(w2, fail_type="KeyError")[0][3][2][0]
    s1 = rb.signature(c1[0], c1[1]["message"], c1[2], rb.path_roots(w1))
    s2 = rb.signature(c2[0], c2[1]["message"], c2[2], rb.path_roots(w2))
    s3 = rb.signature(c3[0], c3[1]["message"], c3[2], rb.path_roots(w2))
    assert s1 == s2
    assert s1["phase"] == "call" and s1["exc_type"] == "AssertionError"
    assert "<REPO>/data/x.json" in s1["message"] and "0x<HEX>" in s1["message"]
    assert s3["exc_type"] == "KeyError" and s3 != s1


def test_error_signature_strips_the_phase_wrapper(tmp_path):
    wt = _worktree(tmp_path / "w")
    s = rb.signature("error", 'failed on teardown with "OSError: [Errno 2] /tmp/x1/y"',
                     "t\n\nsrc/tests/test_beta.py:3: OSError", rb.path_roots(wt))
    assert s == {"phase": "teardown", "exc_type": "OSError", "message": "OSError: [Errno 2] <TMP>/y",
                 "sha256": s["sha256"]}
    c = rb.signature("error", "collection failure", "src/tests/test_beta.py:1: in <module>\n"
                     "    import nope\nE   ModuleNotFoundError: No module named 'nope'", rb.path_roots(wt))
    assert c["phase"] == "collection" and c["exc_type"] == "ModuleNotFoundError"
    with pytest.raises(rb.Refusal):
        rb.signature("error", "something unexpected", "x", rb.path_roots(wt))


# --------------------------------------------------------------------------- #
# node ids
# --------------------------------------------------------------------------- #
def test_node_id_resolution(tmp_path):
    wt = _worktree(tmp_path / "w", extra=("src/tests/sub/test_deep.py",))
    r = lambda c, n: rb.resolve_nodeid(c, n, wt)  # noqa: E731
    assert r(A, "test_ok") == N_OK
    assert r(A + ".TestGroup", "test_param[a.b-1]") == N_PARAM
    assert r(A + ".TestOuter.TestInner", "test_x[p::q]") == "src/tests/test_alpha.py::TestOuter::TestInner::test_x[p::q]"
    assert r("src.tests.sub.test_deep", "test_d") == "src/tests/sub/test_deep.py::test_d"
    assert r("", "src.tests.test_beta") == "src/tests/test_beta.py"          # a collection-level report


@pytest.mark.parametrize("classname, name", [
    ("src..tests.test_alpha", "test_ok"),        # empty component
    (A, ""),                                     # empty name
    (A, "test ok"),                              # whitespace outside the parameter brackets
    ("src.tests.test_missing", "test_ok"),       # no such module in the worktree
    ("pytest", "internal"),                      # pytest internal error
])
def test_malformed_node_ids_are_refused(tmp_path, classname, name):
    wt = _worktree(tmp_path / "w")
    with pytest.raises(rb.Refusal):
        rb.resolve_nodeid(classname, name, wt)


def test_ambiguous_module_resolution_is_refused(tmp_path):
    wt = _worktree(tmp_path / "w", extra=("src/tests/test_alpha/TestGroup.py",))
    with pytest.raises(rb.Refusal, match="ambiguous"):
        rb.resolve_nodeid(A + ".TestGroup", "test_param", wt)


# --------------------------------------------------------------------------- #
# record
# --------------------------------------------------------------------------- #
def test_record_writes_every_field(tmp_path, git_ok):
    wt, j, lg = _inputs(tmp_path)
    rc, out = _record(tmp_path, wt, j, lg)
    assert rc == 0
    rec = json.loads(out.read_text())
    assert rec["schema"] == rb.SCHEMA and rec["commit"] == SHA and rec["worktree_head"] == SHA
    assert rec["tracked_tree_clean"] is True and rec["command"] == CMD
    assert rec["versions"] == VERSIONS
    assert rec["collection_count"] == 7
    assert rec["totals"] == {"passed": 2, "failed": 1, "error": 1, "skipped": 1, "xfailed": 1, "xpassed": 1}
    outcomes = {k: v["outcome"] for k, v in rec["nodes"].items()}
    assert outcomes == {N_OK: "passed", N_PARAM: "passed", N_XP: "xpassed", N_FAIL: "failed",
                        N_ERR: "error", N_SKIP: "skipped", N_XF: "xfailed"}
    fsig = rec["nodes"][N_FAIL]["signatures"][0]
    assert fsig["exc_type"] == "AssertionError" and "<REPO>/data/x.json" in fsig["message"]
    assert "<PYTEST_TMP>/test_fail<N>/out.txt" in fsig["message"] and str(wt) not in json.dumps(fsig)
    esig = rec["nodes"][N_ERR]["signatures"][0]
    assert esig["phase"] == "setup" and esig["message"] == "RuntimeError: pid <PID> timed out after <DUR>"
    assert rec["nodes"][N_SKIP]["reason"] == "needs a GPU"
    assert rec["nodes"][N_XF]["reason"] == "known numerical issue"
    assert [f["nodeid"] for f in rec["failures"]] == [N_ERR, N_FAIL]
    assert rec["junit"]["sha256"] == _sha(j) and rec["terminal_log"]["sha256"] == _sha(lg)
    assert rec["junit"]["testsuite"]["tests"] == 7
    assert rec["tool"]["sha256"] == _sha(_TOOL)
    assert rec["normalisation"]["rules_sha256"] == rb.rules_sha256()
    assert rec["content_sha256"] == rb.content_sha256(rec)


@pytest.mark.parametrize("head, clean, match", [(OTHER, True, "HEAD"), (SHA, False, "tracked")])
def test_record_refuses_wrong_head_or_dirty_tree(tmp_path, monkeypatch, git_ok, head, clean, match, capsys):
    monkeypatch.setattr(rb, "_git_head", lambda wt: head)
    monkeypatch.setattr(rb, "_git_tracked_clean", lambda wt: clean)
    wt, j, lg = _inputs(tmp_path)
    rc, out = _record(tmp_path, wt, j, lg)
    assert rc == 2 and not out.exists() and match in capsys.readouterr().err


def test_record_refuses_to_overwrite(tmp_path, git_ok):
    wt, j, lg = _inputs(tmp_path)
    out = tmp_path / "baseline.json"
    out.write_text("keep")
    rc, _ = _record(tmp_path, wt, j, lg, out=out)
    assert rc == 2 and out.read_text() == "keep"


def test_record_refuses_a_malformed_commit(tmp_path, git_ok):
    wt, j, lg = _inputs(tmp_path)
    rc, out = _record(tmp_path, wt, j, lg, commit="6a51f741")
    assert rc == 2 and not out.exists()


def test_duplicate_node_ids_are_refused(tmp_path, git_ok, capsys):
    wt = _worktree(tmp_path / "w")
    cases, summary, final = scenario(wt)
    cases = cases + [(A, "test_ok", ())]
    summary = [f"PASSED {N_OK}"] + summary
    final = final.replace("2 passed", "3 passed")
    wt, j, lg = _inputs(tmp_path, wt=wt, cases=cases, summary=summary, final=final)
    rc, out = _record(tmp_path, wt, j, lg)
    assert rc == 2 and not out.exists() and "duplicate" in capsys.readouterr().err


def test_documented_failure_plus_teardown_error_pair_is_merged(tmp_path, git_ok):
    wt = _worktree(tmp_path / "w")
    cases, summary, final = scenario(wt)
    cases = cases + [(A, "test_dbl", (F("ValueError: v", "x\n\nsrc/tests/test_alpha.py:5: ValueError"),)),
                     (A, "test_dbl", (E('failed on teardown with "OSError: t"', "y\n\nsrc/tests/test_alpha.py:9: OSError"),))]
    nd = "src/tests/test_alpha.py::test_dbl"
    summary = summary + [f"ERROR {nd} - OSError: t", f"FAILED {nd} - ValueError: v"]
    final = final.replace("1 failed", "2 failed").replace("1 error", "2 errors")
    wt, j, lg = _inputs(tmp_path, wt=wt, cases=cases, summary=summary, final=final)
    rc, out = _record(tmp_path, wt, j, lg)
    assert rc == 0
    rec = json.loads(out.read_text())
    node = rec["nodes"][nd]
    assert node["outcome"] == "failed" and node["double_reported"] is True
    assert [s["phase"] for s in node["signatures"]] == ["call", "teardown"]
    assert rec["collection_count"] == 8


@pytest.mark.parametrize("override", [{"tests": 8}, {"failures": 0}, {"errors": 2}, {"skipped": 1}])
def test_junit_whose_totals_disagree_with_its_testcases_is_refused(tmp_path, git_ok, override, capsys):
    wt = _worktree(tmp_path / "w")
    cases, summary, final = scenario(wt)
    j = write_junit(tmp_path / "bad.xml", cases, **override)
    lg = write_log(tmp_path / "bad.log", summary, final)
    rc, out = _record(tmp_path, wt, j, lg)
    assert rc == 2 and not out.exists() and "disagree" in capsys.readouterr().err


def _drop(summary, prefix):
    return [s for s in summary if not s.startswith(prefix)]


@pytest.mark.parametrize("mutate", [
    lambda s, f: (_drop(s, "XPASS"), f),                                       # XPASS line missing
    lambda s, f: (s + [f"FAILED {N_OK} - x"], f),                               # FAILED names a passed node
    lambda s, f: (s, f.replace("2 passed", "3 passed")),                        # final counts wrong
    lambda s, f: (s + ["FAILED src/tests/test_alpha.py::test_ghost - x"], f),  # unknown node
    lambda s, f: (_drop(s, "SKIPPED"), f),                                      # skip count wrong
    lambda s, f: (s, "no final line here"),                                     # no totals line
])
def test_terminal_log_disagreeing_with_junit_is_refused(tmp_path, git_ok, mutate):
    wt = _worktree(tmp_path / "w")
    cases, summary, final = scenario(wt)
    summary, final = mutate(summary, final)
    wt, j, lg = _inputs(tmp_path, wt=wt, cases=cases, summary=summary, final=final)
    rc, out = _record(tmp_path, wt, j, lg)
    assert rc == 2 and not out.exists()


def test_terminal_log_without_summary_section_is_refused(tmp_path, git_ok):
    wt, j, _ = _inputs(tmp_path)
    lg = tmp_path / "plain.log"
    lg.write_text(".......\n1 failed, 2 passed, 1 skipped, 1 xfailed, 1 xpassed, 1 error in 3.21s\n")
    rc, out = _record(tmp_path, wt, j, lg)
    assert rc == 2 and not out.exists()


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #
def _new_run(tmp_path, cases_fn=None, summary_fn=None, final_fn=None, **kw):
    wt = _worktree(tmp_path / "wt_new")
    cases, summary, final = scenario(wt, hexaddr="0x55aa11223344", pid="777", secs="9.87", line="40",
                                     tmpn="5", **kw)
    if cases_fn:
        cases = cases_fn(cases, wt)
    if summary_fn:
        summary = summary_fn(summary)
    if final_fn:
        final = final_fn(final)
    j = write_junit(tmp_path / "new.xml", cases)
    lg = write_log(tmp_path / "new.log", summary, final)
    return wt, j, lg


def test_identical_run_from_another_worktree_passes(tmp_path, git_ok):
    base = _baseline(tmp_path)
    wt, j, lg = _new_run(tmp_path)
    rc, rep = _compare(tmp_path, base, wt, j, lg)
    assert rc == 0 and rep["verdict"] == "PASS" and rep["reasons"] == []
    assert rep["baseline_failures"] == {N_ERR: "unchanged", N_FAIL: "unchanged"}
    assert rep["baseline"]["content_sha256"] == json.loads(base.read_text())["content_sha256"]


def _add(cases, *new):
    return cases + list(new)


def test_nodes_of_declared_new_test_files_are_allowed(tmp_path, git_ok):
    base = _baseline(tmp_path)
    wt, j, lg = _new_run(tmp_path, cases_fn=lambda c, w: _add(c, ("src.tests.test_new", "test_n1", ())),
                         summary_fn=lambda s: s + ["PASSED src/tests/test_new.py::test_n1"],
                         final_fn=lambda f: f.replace("2 passed", "3 passed"))
    rc, rep = _compare(tmp_path, base, wt, j, lg, "--new-test-file", "src/tests/test_new.py")
    assert rc == 0 and rep["verdict"] == "PASS"
    assert rep["added_nodes"] == ["src/tests/test_new.py::test_n1"]
    rc2, rep2 = _compare(tmp_path, base, wt, j, lg, out_name="cmp2.json")
    assert rc2 == 1 and any("not in a declared new test file" in r for r in rep2["reasons"])


def test_removed_node_fails(tmp_path, git_ok):
    base = _baseline(tmp_path)
    wt, j, lg = _new_run(tmp_path, cases_fn=lambda c, w: [x for x in c if x[1] != "test_ok"],
                         summary_fn=lambda s: _drop(s, f"PASSED {N_OK}"),
                         final_fn=lambda f: f.replace("2 passed", "1 passed"))
    rc, rep = _compare(tmp_path, base, wt, j, lg)
    assert rc == 1 and rep["removed_nodes"] == [N_OK]


def _set_children(name, children):
    def fn(cases, wt):
        return [(c, n, children if n == name else ch) for c, n, ch in cases]
    return fn


def test_fixed_baseline_failure_fails_the_comparison(tmp_path, git_ok):
    base = _baseline(tmp_path)
    wt, j, lg = _new_run(tmp_path, cases_fn=_set_children("test_fail", ()),
                         summary_fn=lambda s: _drop(s, "FAILED") + [f"PASSED {N_FAIL}"],
                         final_fn=lambda f: f.replace("1 failed, 2 passed", "3 passed"))
    rc, rep = _compare(tmp_path, base, wt, j, lg)
    assert rc == 1 and rep["baseline_failures"][N_FAIL] == "changed"
    assert any(N_FAIL in r for r in rep["reasons"])


def test_changed_failure_signature_fails(tmp_path, git_ok):
    base = _baseline(tmp_path)
    wt, j, lg = _new_run(tmp_path, fail_type="KeyError")
    rc, rep = _compare(tmp_path, base, wt, j, lg)
    assert rc == 1 and rep["baseline_failures"][N_FAIL] == "changed"


@pytest.mark.parametrize("children, word, final_fn", [
    ((F("AssertionError: new", "x\n\nsrc/tests/test_alpha.py:2: AssertionError"),), "FAILED",
     lambda f: f.replace("1 failed, 2 passed", "2 failed, 1 passed")),
    ((S("now skipped"),), None, lambda f: f.replace("2 passed, 1 skipped", "1 passed, 2 skipped")),
    ((X("now xfail"),), "XFAIL", lambda f: f.replace("2 passed", "1 passed").replace("1 xfailed", "2 xfailed")),
])
def test_pass_turning_into_fail_skip_or_xfail_fails(tmp_path, git_ok, children, word, final_fn):
    base = _baseline(tmp_path)

    def summ(s):
        s = _drop(s, f"PASSED {N_OK}")
        if word:
            s = s + [f"{word} {N_OK} - x"]
        else:
            s = [x.replace("SKIPPED [1]", "SKIPPED [2]") for x in s]
        return s
    wt, j, lg = _new_run(tmp_path, cases_fn=_set_children("test_ok", children), summary_fn=summ, final_fn=final_fn)
    rc, rep = _compare(tmp_path, base, wt, j, lg)
    assert rc == 1 and any(N_OK in r for r in rep["reasons"])


def _new_skip_run(tmp_path):
    return _new_run(tmp_path, cases_fn=lambda c, w: _add(c, ("src.tests.test_new", "test_opt", (S("run absent"),))),
                    summary_fn=lambda s: [x.replace("SKIPPED [1]", "SKIPPED [2]") for x in s],
                    final_fn=lambda f: f.replace("1 skipped", "2 skipped"))


def test_new_skip_in_a_new_file_must_be_declared(tmp_path, git_ok):
    base = _baseline(tmp_path)
    wt, j, lg = _new_skip_run(tmp_path)
    nd = "src/tests/test_new.py::test_opt"
    rc, rep = _compare(tmp_path, base, wt, j, lg, "--new-test-file", "src/tests/test_new.py")
    assert rc == 1 and any("skip" in r and nd in r for r in rep["reasons"])
    rc2, rep2 = _compare(tmp_path, base, wt, j, lg, "--new-test-file", "src/tests/test_new.py",
                         "--allow-new-skip", nd, out_name="cmp2.json")
    assert rc2 == 0 and rep2["allowed_new_skips"] == {nd: "run absent"}


def test_failing_new_test_fails(tmp_path, git_ok):
    base = _baseline(tmp_path)
    wt, j, lg = _new_run(
        tmp_path, cases_fn=lambda c, w: _add(c, ("src.tests.test_new", "test_bad",
                                                 (F("AssertionError: no", "x\n\nsrc/tests/test_new.py:1: AssertionError"),))),
        summary_fn=lambda s: s + ["FAILED src/tests/test_new.py::test_bad - x"],
        final_fn=lambda f: f.replace("1 failed", "2 failed"))
    rc, rep = _compare(tmp_path, base, wt, j, lg, "--new-test-file", "src/tests/test_new.py")
    assert rc == 1


def test_edited_node_is_listed_separately_and_must_pass(tmp_path, git_ok):
    base = _baseline(tmp_path)
    wt, j, lg = _new_run(tmp_path)
    rc, rep = _compare(tmp_path, base, wt, j, lg, "--edited-node", N_PARAM)
    assert rc == 0 and rep["edited_nodes"] == {N_PARAM: {"baseline": "passed", "new": "passed", "ok": True}}
    wt2, j2, lg2 = _new_run(tmp_path, cases_fn=_set_children(
        "test_param[a.b-1]", (F("AssertionError: e", "x\n\nsrc/tests/test_alpha.py:3: AssertionError"),)),
        summary_fn=lambda s: _drop(s, f"PASSED {N_PARAM}") + [f"FAILED {N_PARAM} - e"],
        final_fn=lambda f: f.replace("1 failed, 2 passed", "2 failed, 1 passed"))
    rc2, rep2 = _compare(tmp_path, base, wt2, j2, lg2, "--edited-node", N_PARAM, out_name="cmp2.json")
    assert rc2 == 1 and rep2["edited_nodes"][N_PARAM]["ok"] is False


def test_edited_node_absent_from_baseline_fails(tmp_path, git_ok):
    base = _baseline(tmp_path)
    wt, j, lg = _new_run(tmp_path)
    rc, rep = _compare(tmp_path, base, wt, j, lg, "--edited-node", "src/tests/test_alpha.py::test_nope")
    assert rc == 1


def test_declared_new_file_already_in_baseline_fails(tmp_path, git_ok):
    base = _baseline(tmp_path)
    wt, j, lg = _new_run(tmp_path)
    rc, rep = _compare(tmp_path, base, wt, j, lg, "--new-test-file", "src/tests/test_alpha.py")
    assert rc == 1 and any("already" in r for r in rep["reasons"])


def test_baseline_commit_mismatch_is_refused(tmp_path, git_ok, capsys):
    base = _baseline(tmp_path)
    wt, j, lg = _new_run(tmp_path)
    rc, rep = _compare(tmp_path, base, wt, j, lg, commit=OTHER)
    assert rc == 2 and rep is None and "commit" in capsys.readouterr().err


def test_tampered_baseline_is_refused(tmp_path, git_ok):
    base = _baseline(tmp_path)
    rec = json.loads(base.read_text())
    rec["nodes"][N_FAIL]["outcome"] = "passed"
    base.write_text(json.dumps(rec))
    wt, j, lg = _new_run(tmp_path)
    rc, rep = _compare(tmp_path, base, wt, j, lg)
    assert rc == 2 and rep is None


@pytest.mark.parametrize("extra", [
    ["--edited-node", "src/tests/test_new.py::test_n1", "--new-test-file", "src/tests/test_new.py"],
    ["--allow-new-skip", N_OK],
    ["--new-test-file", "/abs/src/tests/test_new.py"],
    ["--new-test-file", "src/tests/../tests/test_new.py"],
    ["--edited-node", "not a node id"],
])
def test_inconsistent_declarations_are_refused(tmp_path, git_ok, extra):
    base = _baseline(tmp_path)
    wt, j, lg = _new_run(tmp_path)
    rc, rep = _compare(tmp_path, base, wt, j, lg, *extra)
    assert rc == 2 and rep is None


def test_compare_refuses_to_overwrite(tmp_path, git_ok):
    base = _baseline(tmp_path)
    wt, j, lg = _new_run(tmp_path)
    (tmp_path / "cmp.json").write_text("keep")
    rc, _ = _compare(tmp_path, base, wt, j, lg)
    assert rc == 2 and (tmp_path / "cmp.json").read_text() == "keep"


# --------------------------------------------------------------------------- #
# real pytest output (pytest 9.1.1 of the running interpreter, in a subprocess)
# --------------------------------------------------------------------------- #
_MINI = '''
import os, pytest

def test_pass():
    pass

def test_fail(tmp_path):
    assert str(tmp_path) == "x", f"pid {os.getpid()} at {hex(id(object()))}"

@pytest.fixture
def bad_setup():
    raise RuntimeError("setup broke")

def test_setup_error(bad_setup):
    pass

@pytest.fixture
def bad_teardown():
    yield
    raise OSError("teardown broke")

def test_teardown_error(bad_teardown):
    pass

def test_fail_and_teardown(bad_teardown):
    raise ValueError("call broke")

@pytest.mark.skip(reason="needs GPU")
def test_skip():
    pass

@pytest.mark.xfail(reason="known")
def test_xfail():
    assert False

@pytest.mark.xfail(reason="flaky")
def test_xpass():
    pass

@pytest.mark.xfail(reason="strict one", strict=True)
def test_xpass_strict():
    pass

class TestK:
    @pytest.mark.parametrize("v", ["a.b", "c::d", 1.5])
    def test_p(self, v):
        assert v != 1.5

def test_imperative_skip():
    pytest.skip("runtime skip")
'''


def test_real_pytest_junit_and_terminal_log_round_trip(tmp_path):
    import subprocess
    proj = tmp_path / "proj"
    (proj / "src/tests").mkdir(parents=True)
    (proj / "pyproject.toml").write_text("")
    (proj / "src/tests/test_mini.py").write_text(_MINI)
    junit = tmp_path / "j.xml"
    env = {k: v for k, v in __import__("os").environ.items() if not k.startswith("PYTEST")}
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:warnings",
                          "src/tests", "-rf", "-rA", f"--junitxml={junit}", f"--basetemp={tmp_path / 'bt'}"],
                         cwd=proj, capture_output=True, text=True, env=env)
    log = tmp_path / "run.log"
    log.write_text(out.stdout + out.stderr)
    run = rb.parse_run(junit, log, proj)
    p = "src/tests/test_mini.py::"
    assert {k[len(p):]: v["outcome"] for k, v in run["nodes"].items()} == {
        "test_pass": "passed", "test_fail": "failed", "test_setup_error": "error",
        "test_teardown_error": "error", "test_fail_and_teardown": "failed", "test_skip": "skipped",
        "test_xfail": "xfailed", "test_xpass": "xpassed", "test_xpass_strict": "failed",
        "TestK::test_p[a.b]": "passed", "TestK::test_p[c::d]": "passed", "TestK::test_p[1.5]": "failed",
        "test_imperative_skip": "skipped"}
    assert run["totals"] == {"passed": 3, "failed": 4, "error": 2, "skipped": 2, "xfailed": 1, "xpassed": 1}
    dbl = run["nodes"][p + "test_fail_and_teardown"]
    assert dbl["double_reported"] and [s["exc_type"] for s in dbl["signatures"]] == ["ValueError", "OSError"]
    assert run["nodes"][p + "test_teardown_error"]["signatures"][0]["phase"] == "teardown"
    fail = run["nodes"][p + "test_fail"]["signatures"][0]["message"]
    assert "pid <PID> at 0x<HEX>" in fail and str(tmp_path) not in fail
    assert run["nodes"][p + "test_skip"]["reason"] == "needs GPU"


# --------------------------------------------------------------------------- #
# pytest 9 subtests: a passing subtest adds one 'passed' stat (JUnit tests=) but no testcase element,
# and appears only as 'N subtests passed' in the totals line (found in the real 6a51f741 baseline)
# --------------------------------------------------------------------------- #
def test_passing_subtests_reconcile_junit_tests_with_the_totals_line(tmp_path, git_ok):
    wt = _worktree(tmp_path / "w")
    cases, summary, final = scenario(wt)
    j = write_junit(tmp_path / "s.xml", cases, tests=len(cases) + 2)
    final2 = final.replace(" in ", ", 2 subtests passed in ")
    lg = write_log(tmp_path / "s.log", summary, final2)
    rc, out = _record(tmp_path, wt, j, lg)
    assert rc == 0
    rec = json.loads(out.read_text())
    assert rec["subtests_passed"] == 2 and rec["collection_count"] == 7
    assert rec["junit"]["testsuite"]["tests"] == 9
    lg2 = write_log(tmp_path / "s2.log", summary, final)          # same JUnit, no subtests in the log
    rc2, out2 = _record(tmp_path, wt, j, lg2, out=tmp_path / "b2.json")
    assert rc2 == 2 and not out2.exists()


@pytest.mark.parametrize("line", ["SUBFAILED(filename='a') src/tests/test_alpha.py::test_ok - boom",
                                  "SUBSKIPPED(x=1) src/tests/test_alpha.py::test_ok - why"])
def test_failed_or_skipped_subtest_reports_are_refused(tmp_path, git_ok, line, capsys):
    wt = _worktree(tmp_path / "w")
    cases, summary, final = scenario(wt)
    wt, j, lg = _inputs(tmp_path, wt=wt, cases=cases, summary=summary + [line], final=final)
    rc, out = _record(tmp_path, wt, j, lg)
    assert rc == 2 and not out.exists() and "subtest" in capsys.readouterr().err


@pytest.mark.parametrize("word", ["3 subtests failed", "1 subtests skipped", "2 weird"])
def test_unknown_or_unsupported_totals_words_are_refused(tmp_path, git_ok, word):
    wt = _worktree(tmp_path / "w")
    cases, summary, final = scenario(wt)
    wt, j, lg = _inputs(tmp_path, wt=wt, cases=cases, summary=summary, final=final.replace(" in ", f", {word} in "))
    rc, out = _record(tmp_path, wt, j, lg)
    assert rc == 2 and not out.exists()


_MINI_SUB = '''
import unittest

class TestU(unittest.TestCase):
    def test_sub(self):
        for v in (1, 2):
            with self.subTest(v=v):
                self.assertTrue(v > 0)

def test_plain():
    pass
'''


def _run_mini(tmp_path, source):
    import subprocess
    proj = tmp_path / "proj"
    (proj / "src/tests").mkdir(parents=True)
    (proj / "pyproject.toml").write_text("")
    (proj / "src/tests/test_mini.py").write_text(source)
    junit = tmp_path / "j.xml"
    env = {k: v for k, v in __import__("os").environ.items() if not k.startswith("PYTEST")}
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:warnings",
                          "src/tests", "-rf", "-rA", f"--junitxml={junit}", f"--basetemp={tmp_path / 'bt'}"],
                         cwd=proj, capture_output=True, text=True, env=env)
    log = tmp_path / "run.log"
    log.write_text(out.stdout + out.stderr)
    return proj, junit, log


def test_real_pytest_passing_unittest_subtests_are_accepted(tmp_path):
    proj, junit, log = _run_mini(tmp_path, _MINI_SUB)
    assert "2 subtests passed" in log.read_text()
    run = rb.parse_run(junit, log, proj)
    assert run["subtests_passed"] == 2
    assert {k: v["outcome"] for k, v in run["nodes"].items()} == {
        "src/tests/test_mini.py::TestU::test_sub": "passed", "src/tests/test_mini.py::test_plain": "passed"}


def test_real_pytest_failing_subtest_is_refused(tmp_path):
    proj, junit, log = _run_mini(tmp_path, _MINI_SUB.replace("v > 0", "v > 1"))
    with pytest.raises(rb.Refusal):
        rb.parse_run(junit, log, proj)


def test_changed_subtest_count_fails_the_comparison(tmp_path, git_ok):
    wt = _worktree(tmp_path / "wt_base")
    cases, summary, final = scenario(wt)
    j = write_junit(tmp_path / "b.xml", cases, tests=len(cases) + 2)
    lg = write_log(tmp_path / "b.log", summary, final.replace(" in ", ", 2 subtests passed in "))
    rc, base = _record(tmp_path, wt, j, lg)
    assert rc == 0
    wt2, j2, lg2 = _new_run(tmp_path)
    rc2, rep = _compare(tmp_path, base, wt2, j2, lg2)
    assert rc2 == 1 and any("subtests" in r for r in rep["reasons"])


# --------------------------------------------------------------------------- #
# node-id canonicalisation at compare time (found on the real L2 run: 17 parametrised ids embed the
# absolute worktree path, 1 embeds the wall clock). Only the text inside the parametrisation brackets
# is canonicalised, symmetrically, each side with its own worktree root.
# --------------------------------------------------------------------------- #
COMMITTED_BASELINE = _REPO / "worklog/worklog_yixun/exp_43_std_aug_claude/exp43_regression_baseline.json"
MESSAGE_RULES_SHA256 = "09268be54b832c29768cd02b83637f286078575e97a34abaac26f7fb5f23706c"
NODEID_RULES_SHA256 = "135c47b35d0559d5e505c8ebf94ef0449ac6037acbfddf3a1543f2c12d11bec8"
PA = "src/tests/test_alpha.py::"
NEW_KW = dict(hexaddr="0x55aa11223344", pid="777", secs="9.87", line="40", tmpn="5")


def _extra_run(tmp_path, name, wt, extra, secs="3.21", **kw):
    """scenario() plus extra nodes ``(classname, test name, outcome)``, outcome passed / failed / skipped."""
    cases, summary, _final = scenario(wt, secs=secs, **kw)
    n = {"passed": 0, "failed": 0, "skipped": 0}
    for cls, tname, outcome in extra:
        nid = cls.replace(".", "/") + ".py::" + tname
        n[outcome] += 1
        if outcome == "failed":
            cases = cases + [(cls, tname, (F("AssertionError: p", "x\n\nsrc/tests/test_alpha.py:7: AssertionError"),))]
            summary = summary + [f"FAILED {nid} - AssertionError: p"]
        elif outcome == "skipped":
            cases = cases + [(cls, tname, (S("run absent"),))]
        else:
            cases = cases + [(cls, tname, ())]
            summary = summary + [f"PASSED {nid}"]
    summary = [x.replace("SKIPPED [1]", f"SKIPPED [{1 + n['skipped']}]") for x in summary]
    final = (f"{1 + n['failed']} failed, {2 + n['passed']} passed, {1 + n['skipped']} skipped, 1 xfailed, "
             f"1 xpassed, 1 error in {secs}s (0:00:03)")
    return write_junit(tmp_path / f"{name}.xml", cases), write_log(tmp_path / f"{name}.log", summary, final)


def _canon_runs(tmp_path, base_extra, new_extra, new_link=False, new_embed_real=False):
    """A baseline recorded in wt_base and a new run in wt_new (optionally passed as a symlink); the
    extra nodes are ``fn(root) -> [(classname, name, outcome)]`` with the root each run embeds."""
    wb = _worktree(tmp_path / "wt_base")
    j, lg = _extra_run(tmp_path, "base", wb, base_extra(wb))
    rc, base = _record(tmp_path, wb, j, lg)
    assert rc == 0
    wn = _worktree(tmp_path / "wt_new")
    arg = wn
    if new_link:
        arg = tmp_path / "wt_new_link"
        arg.symlink_to(wn)
    embed = wn if (new_embed_real or not new_link) else arg
    j2, lg2 = _extra_run(tmp_path, "new", arg, new_extra(embed), **NEW_KW)
    return base, wb, wn, arg, j2, lg2


def _path_node(root, rel="src/tests/test_alpha.py", outcome="passed"):
    return (A, f"test_path[{root}/{rel}]", outcome)


@pytest.mark.parametrize("nodeid, roots, expected", [
    # (a) this run's own worktree root -> <REPO>, only as a whole path
    ("f.py::t[/x/repo/a.py]", ["/x/repo"], "f.py::t[<REPO>/a.py]"),
    ("f.py::t[/x/repo]", ["/x/repo"], "f.py::t[<REPO>]"),
    ("f.py::t[k=/x/repo/a, /x/repo]", ["/x/repo"], "f.py::t[k=<REPO>/a, <REPO>]"),
    ("f.py::t[/x/repo2/a.py]", ["/x/repo"], "f.py::t[/x/repo2/a.py]"),          # shares a prefix only
    ("f.py::t[/x/repo.bak/a]", ["/x/repo"], "f.py::t[/x/repo.bak/a]"),
    ("f.py::t[/x/repo-old/a]", ["/x/repo"], "f.py::t[/x/repo-old/a]"),
    ("f.py::t[/y/x/repo/a]", ["/x/repo"], "f.py::t[/y/x/repo/a]"),              # a deeper path
    ("f.py::t[a/x/repo/a]", ["/x/repo"], "f.py::t[a/x/repo/a]"),                # a relative path
    ("f.py::t[1-/x/repo/a]", ["/x/repo"], "f.py::t[1-/x/repo/a]"),              # '-' is no start boundary
    ("f.py::t[/b/root/a]", ["/a/root"], "f.py::t[/b/root/a]"),                  # another run's root
    ("f.py::t[/x/repo/sub/a]", ["/x/repo", "/x/repo/sub"], "f.py::t[<REPO>/a]"),  # longest first
    ("f.py::t[/x/re.po+(1)/a]", ["/x/re.po+(1)"], "f.py::t[<REPO>/a]"),         # the root is literal
    ("f.py::t[/x/reXpo+(1)/a]", ["/x/re.po+(1)"], "f.py::t[/x/reXpo+(1)/a]"),
    ("f.py::t[/x/run-1794553485.5/a]", ["/x/run-1794553485.5"], "f.py::t[<REPO>/a]"),  # roots before epochs
    # (b) an epoch-like float -> <EPOCH>
    ("f.py::t[t-1794553485.4892116-non-future]", [], "f.py::t[t-<EPOCH>-non-future]"),
    ("f.py::t[123456789.0]", [], "f.py::t[<EPOCH>]"),
    ("f.py::t[a1794553485.5]", [], "f.py::t[a<EPOCH>]"),
    ("f.py::t[12345678.9]", [], "f.py::t[12345678.9]"),                        # 8 integer digits
    ("f.py::t[1794553485]", [], "f.py::t[1794553485]"),                        # no fraction
    ("f.py::t[1794553485.]", [], "f.py::t[1794553485.]"),
    ("f.py::t[1.1794553485.5]", [], "f.py::t[1.1794553485.5]"),                # bounded by a dot
    ("f.py::t[1794553485.5.1]", [], "f.py::t[1794553485.5.1]"),
    ("f.py::t[2.109375]", [], "f.py::t[2.109375]"),
    ("f.py::C::t[p::q[1794553485.5]]", [], "f.py::C::t[p::q[<EPOCH>]]"),
    # text outside the brackets is never touched
    ("/x/repo/f.py::t[/x/repo/a]", ["/x/repo"], "/x/repo/f.py::t[<REPO>/a]"),
    ("d/1794553485.4892116/f.py::t", [], "d/1794553485.4892116/f.py::t"),
    ("d/1794553485.4892116/f.py::t[1794553485.4892116]", [], "d/1794553485.4892116/f.py::t[<EPOCH>]"),
    ("/x/repo/f.py::t", ["/x/repo"], "/x/repo/f.py::t"),
    ("src/tests/test_a.py", ["/x/repo"], "src/tests/test_a.py"),
], ids=["root_subpath", "root_alone", "root_after_eq_and_comma", "prefix_only_digit", "prefix_only_dot",
        "prefix_only_dash", "deeper_path", "relative_path", "dash_before_root", "other_runs_root", "longest_first",
        "root_escaped", "root_escaped_no_match", "root_before_epoch",
        "epoch_real", "epoch_9_digits", "epoch_after_letter", "digits_8", "no_fraction", "no_fraction_digit",
        "dot_before", "dot_after", "small_float", "epoch_nested",
        "outside_root", "outside_epoch_no_brackets", "outside_epoch_with_brackets", "outside_root_no_brackets",
        "collection_level"])
def test_canonical_nodeid_rules(nodeid, roots, expected):
    assert rb.canonical_nodeid(nodeid, roots) == expected


@pytest.mark.parametrize("nodeid", ["f.py::t[abc", "f.py::t[<EPOCH>]", "f.py::t[<REPO>/a]", "f.py::t[x<REPO>]"],
                         ids=["unbalanced", "epoch_token", "repo_token", "repo_token_inside"])
def test_canonical_nodeid_refuses_unbalanced_brackets_or_a_preexisting_token(nodeid):
    with pytest.raises(rb.Refusal):
        rb.canonical_nodeid(nodeid, ["/x/repo"])


def test_nodeid_rule_set_is_described_and_hashed():
    d = rb.nodeid_rules_description()
    assert [r["name"] for r in d["rules"]] == ["nodeid_repo_root", "nodeid_epoch"]
    assert all(r["pattern"] and r["replacement"] and r["description"] for r in d["rules"])
    assert d["scope"] and d["guards"]
    assert rb.nodeid_rules_sha256() == hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
    assert rb.nodeid_rules_sha256() == NODEID_RULES_SHA256          # any change to the rules is visible


def test_message_rules_and_the_committed_baseline_are_unchanged():
    assert rb.rules_sha256() == MESSAGE_RULES_SHA256
    b = rb._load_baseline(COMMITTED_BASELINE, SHA)
    assert b["content_sha256"] == rb.content_sha256(b) and b["normalisation"]["rules_sha256"] == MESSAGE_RULES_SHA256
    roots = rb.baseline_repo_roots(b)
    assert roots == ["/home/yixunhu/codespace/exp-43-baseline-6a51f741"]
    canon, rewritten = rb.canonicalise_nodes(b["nodes"], roots, "baseline")
    assert len(canon) == len(b["nodes"]) == 6966 and len(rewritten) == 18
    assert sum(c.endswith("[t-<EPOCH>-non-future]") for c in rewritten.values()) == 1
    assert sum("[<REPO>/" in c for c in rewritten.values()) == 17


def test_worktree_path_param_ids_from_two_roots_compare_equal(tmp_path, git_ok):
    base, wb, wn, arg, j, lg = _canon_runs(tmp_path, lambda r: [_path_node(r), (A, f"test_root[{r}]", "passed")],
                                           lambda r: [_path_node(r), (A, f"test_root[{r}]", "passed")])
    rc, rep = _compare(tmp_path, base, arg, j, lg)
    assert rc == 0 and rep["verdict"] == "PASS" and rep["reasons"] == []
    assert rep["removed_nodes"] == [] and rep["added_nodes"] == []
    assert rep["schema"] == "exp43-regression-compare/2"
    nc = rep["node_id_canonicalisation"]
    assert nc["canonicalised"] == [
        {"canonical": PA + "test_path[<REPO>/src/tests/test_alpha.py]",
         "baseline": PA + f"test_path[{wb}/src/tests/test_alpha.py]",
         "new": PA + f"test_path[{wn}/src/tests/test_alpha.py]"},
        {"canonical": PA + "test_root[<REPO>]", "baseline": PA + f"test_root[{wb}]", "new": PA + f"test_root[{wn}]"}]
    assert nc["roots"] == {"baseline": [str(wb)], "new": [str(wn)]}


@pytest.mark.parametrize("embed_real", [False, True])
def test_new_run_worktree_as_given_and_its_realpath_are_both_canonicalised(tmp_path, git_ok, embed_real):
    base, wb, wn, arg, j, lg = _canon_runs(tmp_path, lambda r: [_path_node(r)], lambda r: [_path_node(r)],
                                           new_link=True, new_embed_real=embed_real)
    rc, rep = _compare(tmp_path, base, arg, j, lg)
    assert rc == 0 and rep["verdict"] == "PASS"
    assert rep["node_id_canonicalisation"]["roots"]["new"] == sorted([str(arg), str(wn)])


def test_epoch_float_param_ids_compare_equal(tmp_path, git_ok):
    base, wb, wn, arg, j, lg = _canon_runs(
        tmp_path, lambda r: [(A, "test_t[t-1794553485.4892116-non-future]", "passed")],
        lambda r: [(A, "test_t[t-1794562367.1640806-non-future]", "passed")])
    rc, rep = _compare(tmp_path, base, arg, j, lg)
    assert rc == 0 and rep["verdict"] == "PASS"
    assert rep["node_id_canonicalisation"]["canonicalised"] == [
        {"canonical": PA + "test_t[t-<EPOCH>-non-future]", "baseline": PA + "test_t[t-1794553485.4892116-non-future]",
         "new": PA + "test_t[t-1794562367.1640806-non-future]"}]


def test_the_other_runs_root_is_not_canonicalised(tmp_path, git_ok):
    """The new run embeds the BASELINE's root: only each run's own root is a token, so this differs."""
    base, wb, wn, arg, j, lg = _canon_runs(tmp_path, lambda r: [_path_node(r)],
                                           lambda r: [_path_node(tmp_path / "wt_base")])
    rc, rep = _compare(tmp_path, base, arg, j, lg)
    assert rc == 1
    assert rep["removed_nodes"] == [PA + "test_path[<REPO>/src/tests/test_alpha.py]"]
    assert rep["added_nodes"] == [PA + f"test_path[{wb}/src/tests/test_alpha.py]"]


def test_a_path_that_only_shares_the_roots_prefix_is_not_canonicalised(tmp_path, git_ok):
    base, wb, wn, arg, j, lg = _canon_runs(tmp_path, lambda r: [(A, f"test_path[{r}2/a.py]", "passed")],
                                           lambda r: [(A, f"test_path[{r}2/a.py]", "passed")])
    rc, rep = _compare(tmp_path, base, arg, j, lg)
    assert rc == 1 and rep["removed_nodes"] == [PA + f"test_path[{wb}2/a.py]"]
    assert rep["node_id_canonicalisation"]["canonicalised"] == []


@pytest.mark.parametrize("base_name, new_name", [
    ("test_path[{r}/src/a.py]", "test_path[{r}/src/b.py]"),                 # a different relative file
    ("test_t[17945534.5]", "test_t[17945535.5]"),                           # 8 integer digits: no epoch
    ("test_t[1794553485]", "test_t[1794562367]"),                           # no fraction: no epoch
    ("test_t[1.5]", "test_t[2.5]"),
    ("test_t[a-1794553485.5]", "test_t[b-1794553485.5]"),                   # text next to an epoch
], ids=["other_file", "digits_8", "no_fraction", "small_float", "text_next_to_epoch"])
def test_a_real_difference_inside_the_brackets_still_fails(tmp_path, git_ok, base_name, new_name):
    base, wb, wn, arg, j, lg = _canon_runs(tmp_path, lambda r: [(A, base_name.format(r=r), "passed")],
                                           lambda r: [(A, new_name.format(r=r), "passed")])
    rc, rep = _compare(tmp_path, base, arg, j, lg)
    assert rc == 1 and len(rep["removed_nodes"]) == 1 and len(rep["added_nodes"]) == 1


@pytest.mark.parametrize("side, names", [
    ("new", ["test_t[1794553485.1]", "test_t[1794553486.2]"]),
    ("baseline", ["test_t[1794553485.1]", "test_t[1794553486.2]"]),
    ("new", ["test_path[{link}/a]", "test_path[{real}/a]"]),
])
def test_a_canonical_id_collision_is_refused(tmp_path, git_ok, side, names, capsys):
    def ex(r):
        return [(A, n.format(link=tmp_path / "wt_new_link", real=tmp_path / "wt_new"), "passed") for n in names]
    base, wb, wn, arg, j, lg = _canon_runs(tmp_path, ex if side == "baseline" else (lambda r: []),
                                           ex if side == "new" else (lambda r: []), new_link=True)
    rc, rep = _compare(tmp_path, base, arg, j, lg)
    assert rc == 2 and rep is None and not (tmp_path / "cmp.json").exists()
    assert "collide" in capsys.readouterr().err


@pytest.mark.parametrize("side", ["new", "baseline"])
def test_a_node_id_already_holding_a_token_is_refused(tmp_path, git_ok, side, capsys):
    ex = lambda r: [(A, "test_tok[<EPOCH>]", "passed")]  # noqa: E731
    base, wb, wn, arg, j, lg = _canon_runs(tmp_path, ex if side == "baseline" else (lambda r: []),
                                           ex if side == "new" else (lambda r: []))
    rc, rep = _compare(tmp_path, base, arg, j, lg)
    assert rc == 2 and rep is None and "<EPOCH>" in capsys.readouterr().err


@pytest.mark.parametrize("base_out, new_out, reason", [
    ("passed", "failed", "pass -> fail: {n} (passed -> failed)"),
    ("failed", "passed", "pre-existing failed changed: {n} (failed -> passed)"),
])
def test_an_outcome_change_on_a_canonicalised_node_is_still_caught(tmp_path, git_ok, base_out, new_out, reason):
    base, wb, wn, arg, j, lg = _canon_runs(
        tmp_path, lambda r: [_path_node(r, outcome=base_out), (A, "test_t[1794553485.5]", "passed")],
        lambda r: [_path_node(r, outcome=new_out), (A, "test_t[1794562367.5]", "passed")])
    rc, rep = _compare(tmp_path, base, arg, j, lg)
    n = PA + "test_path[<REPO>/src/tests/test_alpha.py]"
    assert rc == 1 and rep["reasons"] == [reason.format(n=n)]
    assert rep["removed_nodes"] == [] and rep["added_nodes"] == []


def test_the_report_lists_every_canonicalised_node_and_the_rule_set(tmp_path, git_ok):
    base, wb, wn, arg, j, lg = _canon_runs(
        tmp_path, lambda r: [_path_node(r), (A, "test_t[t-1794553485.4892116-x]", "passed")],
        lambda r: [_path_node(r), (A, "test_t[t-1794562367.1640806-x]", "passed"),
                   ("src.tests.test_new", "test_n[1794562367.9892623]", "passed")])
    rc, rep = _compare(tmp_path, base, arg, j, lg, "--new-test-file", "src/tests/test_new.py")
    assert rc == 0 and rep["verdict"] == "PASS"
    assert rep["added_nodes"] == ["src/tests/test_new.py::test_n[<EPOCH>]"]
    nc = rep["node_id_canonicalisation"]
    assert nc["canonicalised"] == [
        {"canonical": PA + "test_path[<REPO>/src/tests/test_alpha.py]",
         "baseline": PA + f"test_path[{wb}/src/tests/test_alpha.py]",
         "new": PA + f"test_path[{wn}/src/tests/test_alpha.py]"},
        {"canonical": PA + "test_t[t-<EPOCH>-x]", "baseline": PA + "test_t[t-1794553485.4892116-x]",
         "new": PA + "test_t[t-1794562367.1640806-x]"},
        {"canonical": "src/tests/test_new.py::test_n[<EPOCH>]", "baseline": None,
         "new": "src/tests/test_new.py::test_n[1794562367.9892623]"}]
    assert nc["counts"] == {"baseline_ids_rewritten": 2, "new_ids_rewritten": 3, "pairs": 2}
    d = rb.nodeid_rules_description()
    assert nc["rules"] == d["rules"] and nc["scope"] == d["scope"] and nc["guards"] == d["guards"]
    assert nc["rules_sha256"] == rb.nodeid_rules_sha256()
    # the message-normalisation rule set is reported elsewhere and is not part of this one
    assert "ansi" not in json.dumps(nc["rules"])


def test_declarations_name_canonical_ids_and_raw_spellings_are_refused(tmp_path, git_ok):
    base, wb, wn, arg, j, lg = _canon_runs(
        tmp_path, lambda r: [_path_node(r)],
        lambda r: [_path_node(r), ("src.tests.test_new", "test_opt[1794562367.5]", "skipped")])
    canon = PA + "test_path[<REPO>/src/tests/test_alpha.py]"
    skip = "src/tests/test_new.py::test_opt[<EPOCH>]"
    nf = ["--new-test-file", "src/tests/test_new.py"]
    rc, rep = _compare(tmp_path, base, arg, j, lg, *nf, "--edited-node", canon, "--allow-new-skip", skip)
    assert rc == 0 and rep["edited_nodes"] == {canon: {"baseline": "passed", "new": "passed", "ok": True}}
    assert rep["allowed_new_skips"] == {skip: "run absent"}
    for k, extra in enumerate([["--edited-node", PA + f"test_path[{wn}/src/tests/test_alpha.py]"],
                               ["--edited-node", PA + f"test_path[{wb}/src/tests/test_alpha.py]"],
                               ["--allow-new-skip", "src/tests/test_new.py::test_opt[1794562367.5]"]]):
        rc2, rep2 = _compare(tmp_path, base, arg, j, lg, *nf, *extra, out_name=f"cmp_raw{k}.json")
        assert rc2 == 2 and rep2 is None


@pytest.mark.parametrize("mutate", [
    lambda rec: rec["normalisation"]["roots"].__setitem__("<REPO>", []),
    lambda rec: rec["normalisation"]["roots"].pop("<REPO>"),
    lambda rec: rec["normalisation"]["roots"].__setitem__("<REPO>", ["relative/root"]),
    lambda rec: rec.__setitem__("worktree", "/somewhere/else"),
])
def test_a_baseline_whose_recorded_root_is_unusable_is_refused(tmp_path, git_ok, mutate):
    base, wb, wn, arg, j, lg = _canon_runs(tmp_path, lambda r: [_path_node(r)], lambda r: [_path_node(r)])
    rec = json.loads(base.read_text())
    mutate(rec)
    rec["content_sha256"] = rb.content_sha256(rec)
    base.write_text(json.dumps(rec))
    rc, rep = _compare(tmp_path, base, arg, j, lg)
    assert rc == 2 and rep is None


def test_the_baselines_recorded_realpath_root_is_canonicalised(tmp_path, git_ok):
    """A baseline recorded through a symlinked worktree: its recorded <REPO> roots hold both spellings."""
    real = _worktree(tmp_path / "wt_base_real")
    link = tmp_path / "wt_base_link"
    link.symlink_to(real)
    j, lg = _extra_run(tmp_path, "base", link, [_path_node(real), (A, f"test_l[{link}/x]", "passed")])
    rc, base = _record(tmp_path, link, j, lg)
    assert rc == 0
    wn = _worktree(tmp_path / "wt_new")
    j2, lg2 = _extra_run(tmp_path, "new", wn, [_path_node(wn), (A, f"test_l[{wn}/x]", "passed")], **NEW_KW)
    rc, rep = _compare(tmp_path, base, wn, j2, lg2)
    assert rc == 0 and rep["verdict"] == "PASS"
    assert rep["node_id_canonicalisation"]["roots"]["baseline"] == sorted([str(link), str(real)])
    assert rep["node_id_canonicalisation"]["counts"]["pairs"] == 2


@pytest.mark.parametrize("roots", [[""], ["/"], ["relative/root"], ["/x/repo/"], ["/x/repo", "rel"]])
def test_an_unusable_root_is_refused(roots):
    with pytest.raises(rb.Refusal, match="unusable"):
        rb.canonical_nodeid("f.py::t[/x/repo/a]", roots)


@pytest.mark.parametrize("rec", [
    {"worktree": "rel", "normalisation": {"roots": {"<REPO>": ["rel"]}}},
    {"worktree": "/x/repo/", "normalisation": {"roots": {"<REPO>": ["/x/repo/"]}}},
    {"worktree": "/x/repo", "normalisation": {"roots": {"<REPO>": "/x/repo"}}},
    {"worktree": "/x/repo", "normalisation": {"roots": []}},
    {"worktree": "/x/repo"},
])
def test_baseline_repo_roots_requires_usable_recorded_roots(rec):
    with pytest.raises(rb.Refusal, match="usable"):
        rb.baseline_repo_roots(rec)
    assert rb.baseline_repo_roots({"worktree": "/x/repo", "normalisation": {"roots": {"<REPO>": ["/y", "/x/repo"]}}}) \
        == ["/x/repo", "/y"]


# --------------------------------------------------------------------------- #
# this file's own node ids must be inert for the tool (found on the L2r2 run: the compare refused the
# id ``...preexisting_token[f.py::t[<EPOCH>]]`` of this very file). Every parametrize set whose auto ids
# would hold a token, a nested bracket, an absolute path or an epoch-like float carries explicit ids.
# --------------------------------------------------------------------------- #
def test_this_files_own_node_ids_are_inert_for_the_tool(request):
    here = Path(__file__).resolve()
    items = [it for it in request.session.items if Path(str(it.path)).resolve() == here]
    expected = {}
    for name, obj in list(globals().items()):
        marks = [m for m in getattr(obj, "pytestmark", []) if m.name == "parametrize"] if name.startswith("test_") else []
        if marks:
            expected[name] = 1
            for m in marks:
                expected[name] *= len(m.args[1])
    got = {}
    for it in items:
        k = getattr(it, "originalname", it.name)
        got[k] = got.get(k, 0) + 1
    if not expected or any(got.get(k, 0) != v for k, v in expected.items()):
        pytest.skip("needs this whole file collected (a -k / --deselect run collects only part of it)")
    roots = rb.path_roots(Path(os.path.abspath(__file__)).parents[2])["<REPO>"]
    params = [it.nodeid for it in items if "[" in it.nodeid]
    assert len(params) == sum(expected.values())
    bad = []
    for nid in (it.nodeid for it in items):
        try:
            changed = rb.canonical_nodeid(nid, roots) != nid
        except rb.Refusal as e:
            bad.append((nid, f"refused: {e}"))
            continue
        if changed:
            bad.append((nid, "rewritten by a node-id rule"))
        if "[" in nid:
            p = nid[nid.index("[") + 1:-1]
            if "[" in p or "]" in p:
                bad.append((nid, "nested bracket"))
            if re.search(r"(?:^|[\s'\"=(,:-])/", p):
                bad.append((nid, "absolute path"))
    assert bad == []
