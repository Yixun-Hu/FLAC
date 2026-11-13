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
])
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
