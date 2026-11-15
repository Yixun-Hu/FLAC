"""exp_45 L1 — the two-parent regression comparison (exp45_regression_compare.py; plan_rf_data_curve.md r3 §3.2,
Codex plan r2 finding 6): synthetic junit records of the merged tree and both parents, one rule per case.

Written by the exp_45 Coder (Claude Opus 5.5).
"""
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "worklog", "worklog_yixun", "exp_45_rf_data_curve_claude"))
import exp45_regression_compare as rc  # noqa: E402

TIE = sorted(rc.TIE_AUDIT)
COMPAT = sorted(rc.COMPAT_EDITED)


def _xml(path, cases):
    """cases: {node: outcome or (outcome, message)} -> a pytest-shaped junit file."""
    out = ['<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest">']
    for node, res in cases.items():
        outcome, msg = (res, "AssertionError: assert 1 == 2") if isinstance(res, str) else res
        mod, name = node.split("::", 1)
        cls = mod[:-3].replace("/", ".")
        if "::" in name:
            k, name = name.split("::", 1)
            cls += "." + k
        name = name.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
        msg = msg.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
        body = {"passed": "", "skipped": '<skipped message="NAS not mounted"/>',
                "failed": f'<failure message="{msg}">trace</failure>', "error": f'<error message="{msg}">trace</error>'}[outcome]
        out.append(f'<testcase classname="{cls}" name="{name}" time="0.1">{body}</testcase>')
    out.append("</testsuite></testsuites>")
    open(path, "w").write("".join(out))
    return str(path)


BASE = {"src/tests/test_a.py::test_x": "passed", "src/tests/test_b.py::test_y[1-2]": "passed",
        "src/tests/test_c.py::TestK::test_z": "passed", COMPAT[0]: "passed", COMPAT[1]: "passed"}


def _run(tmp_path, merged, p38=None, p43=None):
    files = [_xml(tmp_path / f"{n}.xml", c) for n, c in (("m", merged), ("p38", p38 or BASE), ("p43", p43 or BASE))]
    return rc.compare(*(rc.load(f) for f in files))


def test_node_ids_follow_pytests_classname():
    assert rc.node_id({"classname": "src.tests.test_b", "name": "test_y[1-2]"}) == "src/tests/test_b.py::test_y[1-2]"
    assert rc.node_id({"classname": "src.tests.test_c.TestK", "name": "test_z"}) == "src/tests/test_c.py::TestK::test_z"
    with pytest.raises(rc.CompareError):
        rc.node_id({"classname": "nothing.here", "name": "x"})


def test_an_identical_merged_tree_passes(tmp_path):
    problems, report = _run(tmp_path, dict(BASE))
    assert problems == [] and report["verdict"] == "PASS" and report["counts"]["merged"]["passed"] == 5


def test_a_node_lost_from_either_parent_refuses(tmp_path):
    merged = {k: v for k, v in BASE.items() if k != "src/tests/test_b.py::test_y[1-2]"}
    problems, _ = _run(tmp_path, merged)
    assert len(problems) == 2 and all("missing in the merged tree" in p for p in problems)
    only43 = dict(BASE, **{"src/tests/test_exp43_kit.py::test_new": "passed"})
    problems, _ = _run(tmp_path, dict(BASE), p43=only43)
    assert problems == ["src/tests/test_exp43_kit.py::test_new: present in 4a7e3b3a, missing in the merged tree"]


def test_a_new_failure_refuses_and_an_inherited_one_with_the_same_signature_is_allowed(tmp_path):
    node = "src/tests/test_a.py::test_x"
    problems, _ = _run(tmp_path, dict(BASE, **{node: "failed"}))
    assert len(problems) == 1 and node in problems[0]
    problems, report = _run(tmp_path, dict(BASE, **{node: "failed"}), p38=dict(BASE, **{node: "failed"}))
    assert problems == [] and "inherited from 3593f34c" in report["allowed_failures"][0]
    other = ("failed", "KeyError: 'x'")                                             # another signature: refuse
    problems, _ = _run(tmp_path, dict(BASE, **{node: "failed"}), p43=dict(BASE, **{node: other}))
    assert len(problems) == 1
    problems, _ = _run(tmp_path, dict(BASE, **{node: "error"}), p38=dict(BASE, **{node: "failed"}))
    assert len(problems) == 1                                                       # error vs failure: another kind


def test_signatures_ignore_tmp_paths_and_addresses(tmp_path):
    node = "src/tests/test_a.py::test_x"
    m = ("failed", "OSError: /n/fs/x/.tmp/pytest-of-u/pytest-41/test_x0/f at 0x7f00aa: boom")
    p = ("failed", "OSError: /tmp/pytest-of-u/pytest-7/test_x0/f at 0x7fbbcc: boom")
    problems, _ = _run(tmp_path, dict(BASE, **{node: m}), p38=dict(BASE, **{node: p}))
    assert problems == []


def test_the_three_tie_audit_tests_may_fail(tmp_path):
    base = dict(BASE, **{t: "passed" for t in TIE})
    problems, report = _run(tmp_path, dict(base, **{t: "failed" for t in TIE}), p38=base, p43=base)
    assert problems == [] and len(report["allowed_failures"]) == 3
    near = "src/tests/test_depth_canon_collect.py::test_something_else"
    problems, _ = _run(tmp_path, dict(base, **{near: "failed"}), p38=dict(base, **{near: "passed"}), p43=dict(base, **{near: "passed"}))
    assert len(problems) == 1


def test_the_compat_edited_nodes_must_pass_even_if_a_parent_failed_them(tmp_path):
    for outcome in ("failed", "error", "skipped"):
        merged = dict(BASE, **{COMPAT[1]: outcome})
        problems, _ = _run(tmp_path, merged, p43=dict(BASE, **{COMPAT[1]: outcome if outcome != "skipped" else "passed"}))
        assert len(problems) == 1 and COMPAT[1] in problems[0], outcome
    problems, _ = _run(tmp_path, {k: v for k, v in BASE.items() if k != COMPAT[0]},
                       p38={k: v for k, v in BASE.items() if k != COMPAT[0]}, p43={k: v for k, v in BASE.items() if k != COMPAT[0]})
    assert problems == [f"{COMPAT[0]}: the compat-edited node is missing"]


def test_every_new_exp45_node_must_pass(tmp_path):
    new = {"src/tests/test_exp45_registry.py::test_a[rf]": "passed", "src/tests/test_gen_model_comparison_exp45.py::test_b": "passed",
           "src/tests/test_exp45_receipts.py::test_c": "skipped"}
    problems, report = _run(tmp_path, dict(BASE, **new))
    assert problems == [] and report["new_exp45_skips"] == ["src/tests/test_exp45_receipts.py::test_c"]
    for outcome in ("failed", "error"):
        problems, _ = _run(tmp_path, dict(BASE, **dict(new, **{"src/tests/test_exp45_registry.py::test_a[rf]": outcome})))
        assert len(problems) == 1 and "new exp_45 node" in problems[0]


def test_the_cli_exit_codes(tmp_path, capsys):
    good = [_xml(tmp_path / f"{n}.xml", BASE) for n in ("m", "a", "b")]
    out = tmp_path / "report.json"
    assert rc.main(["--merged", good[0], "--exp38", good[1], "--exp43", good[2], "--out", str(out)]) == 0
    assert json.load(open(out))["verdict"] == "PASS" and "REGRESSION POLICY PASS" in capsys.readouterr().out
    bad = _xml(tmp_path / "bad.xml", dict(BASE, **{"src/tests/test_a.py::test_x": "failed"}))
    assert rc.main(["--merged", bad, "--exp38", good[1], "--exp43", good[2]]) == 1
    (tmp_path / "broken.xml").write_text("<testsuites><testcase")
    assert rc.main(["--merged", str(tmp_path / "broken.xml"), "--exp38", good[1], "--exp43", good[2]]) == 2
    (tmp_path / "empty.xml").write_text("<testsuites/>")
    assert rc.main(["--merged", str(tmp_path / "empty.xml"), "--exp38", good[1], "--exp43", good[2]]) == 2
