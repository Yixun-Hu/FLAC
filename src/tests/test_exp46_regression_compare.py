"""exp_46 L1 — the two-parent regression comparison (exp46_regression_compare.py; plan_c4flip_data_curve.md r3 §3.2,
forked from exp_45's test of exp45_regression_compare.py at X45f, plan_rf_data_curve.md r3 §3.2, Codex plan r2
finding 6): synthetic junit records of the merged tree and both parents, one rule per case -- plus exp_46's
extension: the declared edited nodes E1-E5 must pass, every new exp_46 node (the generator contract included) must
pass, the exp_46 copies of exp_45's environment skips keep the same reasons, and a parent node may go only as a
registered replacement. L1 option (A) (Yixun 2026-11-19): pytest's collection-error records (the tests at the end).

Written by the exp_45 Coder (Claude Opus 5.5); exp_46 extension by the exp_46 Coder (Claude Opus 5.5).
"""
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "worklog", "worklog_yixun", "exp_46_c4flip_data_curve_claude"))
import exp46_regression_compare as rc  # noqa: E402

TIE = sorted(rc.TIE_AUDIT)
COMPAT = sorted(rc.COMPAT_EDITED)


def _xml(path, cases, extra=()):
    """cases: {node: outcome or (outcome, message)} -> a pytest-shaped junit file; ``extra``: raw testcase records
    (pytest's collection records, below) written after them as given."""
    out = ['<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest">']
    for node, res in cases.items():
        outcome, msg = (res, "NAS not mounted" if res == "skipped" else "AssertionError: assert 1 == 2") \
            if isinstance(res, str) else res
        mod, name = node.split("::", 1)
        cls = mod[:-3].replace("/", ".")
        if "::" in name:
            k, name = name.split("::", 1)
            cls += "." + k
        name = name.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
        msg = msg.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
        body = {"passed": "", "skipped": f'<skipped type="pytest.skip" message="{msg}">{mod}:1: {msg}</skipped>',
                "failed": f'<failure message="{msg}">trace</failure>', "error": f'<error message="{msg}">trace</error>'}[outcome]
        out.append(f'<testcase classname="{cls}" name="{name}" time="0.1">{body}</testcase>')
    out.extend(extra)
    out.append("</testsuite></testsuites>")
    open(path, "w").write("".join(out))
    return str(path)


DECLARED = sorted(rc.DECLARED_EDITED)
BASE = {"src/tests/test_a.py::test_x": "passed", "src/tests/test_b.py::test_y[1-2]": "passed",
        "src/tests/test_c.py::TestK::test_z": "passed", COMPAT[0]: "passed", COMPAT[1]: "passed",
        **{n: "passed" for n in DECLARED}}


ROOTS = ("/n/fs/gatrdp/codespace/exp-45-rf-data-curve", "/n/fs/gatrdp/codespace/_l1_parent_3593f34c",
         "/n/fs/gatrdp/codespace/_l1_parent_4a7e3b3a")


def _run(tmp_path, merged, p38=None, p43=None, roots=ROOTS):
    files = [_xml(tmp_path / f"{n}.xml", c) for n, c in (("m", merged), ("p38", p38 or BASE), ("p43", p43 or BASE))]
    return rc.compare(*(rc.load(f, roots=roots) for f in files))


def test_node_ids_follow_pytests_classname():
    assert rc.node_id({"classname": "src.tests.test_b", "name": "test_y[1-2]"}) == "src/tests/test_b.py::test_y[1-2]"
    assert rc.node_id({"classname": "src.tests.test_c.TestK", "name": "test_z"}) == "src/tests/test_c.py::TestK::test_z"
    with pytest.raises(rc.CompareError):
        rc.node_id({"classname": "nothing.here", "name": "x"})


def test_an_identical_merged_tree_passes(tmp_path):
    problems, report = _run(tmp_path, dict(BASE))
    assert problems == [] and report["verdict"] == "PASS" and report["counts"]["merged"]["passed"] == 5 + len(DECLARED)


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


def test_signatures_normalise_the_checkout_roots(tmp_path):
    """Codex code r1 finding 8: the inherited validator-pin failure names the checkout it ran in; the same failure in
    the three checkouts has ONE signature once their roots are normalised -- the failing operation and the error
    text stay part of it."""
    node = "src/tests/test_exp40_eval.py::test_authoritative_validator_pins_are_the_source_blobs"
    msg = ("exp40_kit.Refused: git rev-parse 6be49f46:worklog/x.py failed in {root}: fatal: path 'worklog/x.py' "
           "exists on disk, but not in '6be49f46'")
    per = [dict(BASE, **{node: ("failed", msg.format(root=r))}) for r in ROOTS]
    problems, report = _run(tmp_path, *per)
    assert problems == [] and "inherited from 3593f34c, 4a7e3b3a" in report["allowed_failures"][0]
    assert "<checkout>" in report["allowed_failures"][0] and ROOTS[0] not in report["allowed_failures"][0]
    # a root given with a trailing slash or as a symlink resolves to the same placeholder
    problems, _ = _run(tmp_path, *per, roots=tuple(r + "/" for r in ROOTS))
    assert problems == []
    # another failing operation or another error is another signature, roots normalised or not
    for other in (msg.replace("rev-parse", "cat-file"), msg.replace("exists on disk", "is missing")):
        problems, _ = _run(tmp_path, per[0], p38=dict(BASE, **{node: ("failed", other.format(root=ROOTS[1]))}),
                           p43=dict(BASE, **{node: ("failed", other.format(root=ROOTS[2]))}))
        assert len(problems) == 1 and node in problems[0]
    # without the roots the reviewer's case reproduces: three different signatures
    problems, _ = _run(tmp_path, *per, roots=())
    assert len(problems) == 1


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
    new = {"src/tests/test_exp45_registry.py::test_a[rf]": "passed", "src/tests/test_gen_model_comparison_exp45.py::test_b": "passed"}
    problems, report = _run(tmp_path, dict(BASE, **new))
    assert problems == [] and report["new_node_skips"] == []
    for outcome in ("failed", "error"):
        problems, _ = _run(tmp_path, dict(BASE, **dict(new, **{"src/tests/test_exp45_registry.py::test_a[rf]": outcome})))
        assert len(problems) == 1 and "new exp_45 / exp_46 node" in problems[0]
    # Codex code r1 finding 9: an UNREGISTERED skip of a new node is a FAIL, whatever its reason
    problems, _ = _run(tmp_path, dict(BASE, **dict(new, **{"src/tests/test_exp45_receipts.py::test_c": "skipped"})))
    assert len(problems) == 1 and "test_c" in problems[0] and "skip" in problems[0]


def test_only_the_registered_environment_skips_of_new_nodes_pass(tmp_path):
    """Codex code r1 finding 9: skip reasons are kept; a new exp_45 node may be skipped only as an exactly registered
    (node, reason) environment exception -- the same node with another reason, or the reason on another node, FAILs."""
    assert rc.ENV_SKIPS and all(rc.NEW_FILE.match(n) and isinstance(r, str) and r for n, r in rc.ENV_SKIPS)
    reg_skips = {n: ("skipped", r) for n, r in sorted(rc.ENV_SKIPS)}
    problems, report = _run(tmp_path, dict(BASE, **reg_skips))
    assert problems == [] and report["new_node_skips"] == [{"node": n, "reason": r} for n, r in sorted(rc.ENV_SKIPS)]
    node, reason = sorted(rc.ENV_SKIPS)[0]
    problems, _ = _run(tmp_path, dict(BASE, **{node: ("skipped", reason + " (or not)")}))
    assert len(problems) == 1 and node in problems[0]
    problems, _ = _run(tmp_path, dict(BASE, **{"src/tests/test_exp45_registry.py::test_unregistered": ("skipped", reason)}))
    assert len(problems) == 1 and "test_unregistered" in problems[0]
    # an old (non-exp_45) node may still be skipped: the policy covers new nodes only
    problems, _ = _run(tmp_path, dict(BASE, **{"src/tests/test_a.py::test_x": "skipped"}))
    assert problems == []


def test_the_cli_exit_codes(tmp_path, capsys):
    good = [_xml(tmp_path / f"{n}.xml", BASE) for n in ("m", "a", "b")]
    out = tmp_path / "report.json"
    roots = ["--merged-root", ROOTS[0], "--exp38-root", ROOTS[1], "--exp43-root", ROOTS[2]]
    with pytest.raises(SystemExit):
        rc.main(["--merged", good[0], "--exp38", good[1], "--exp43", good[2]])             # the roots are required
    assert rc.main(["--merged", good[0], "--exp38", good[1], "--exp43", good[2], "--out", str(out)] + roots) == 0
    assert json.load(open(out))["verdict"] == "PASS" and "REGRESSION POLICY PASS" in capsys.readouterr().out
    bad = _xml(tmp_path / "bad.xml", dict(BASE, **{"src/tests/test_a.py::test_x": "failed"}))
    assert rc.main(["--merged", bad, "--exp38", good[1], "--exp43", good[2]] + roots) == 1
    (tmp_path / "broken.xml").write_text("<testsuites><testcase")
    assert rc.main(["--merged", str(tmp_path / "broken.xml"), "--exp38", good[1], "--exp43", good[2]] + roots) == 2
    (tmp_path / "empty.xml").write_text("<testsuites/>")
    assert rc.main(["--merged", str(tmp_path / "empty.xml"), "--exp38", good[1], "--exp43", good[2]] + roots) == 2


# ------------------------------------------------------------------ exp_46's extension (plan r3 §3.2 "exp46_regression_compare.py")
def test_the_declared_edited_nodes_are_exactly_e1_to_e5_and_each_must_pass_and_be_present(tmp_path):
    """plan r3 §3.1 E1-E5 (node ids kept): each declared edited node must PASS in the merged tree -- a failure, an
    error or a skip refuses even if a parent failed it the same way -- and its absence refuses."""
    assert DECLARED == sorted([
        "src/tests/test_yaw_aug_flip.py::test_constructor_refuses_flip_with_offset_step",
        *(f"src/tests/test_yaw_aug_flip_factory.py::test_true_flip_with_offset_step_is_refused[{s}]" for s in (32, 1, 4)),
        "src/tests/test_exp43_kit.py::test_config_contract_through_the_real_factory",
        "src/tests/test_exp45_merge_provenance.py::test_later_exp45_commits_touch_only_the_allowed_paths",
        "src/tests/test_gen_model_comparison_depth_canon.py::test_every_pre_exp25_row_renders_byte_identically", *E7_E8,
        E9])
    assert rc.EDITED == rc.COMPAT_EDITED | rc.DECLARED_EDITED and rc.REPLACED == {}
    problems, _ = _run(tmp_path, BASE)
    assert problems == []
    for node in DECLARED:
        for outcome in ("failed", "error", "skipped"):
            parent = dict(BASE, **{node: outcome if outcome != "skipped" else "passed"})
            problems, _ = _run(tmp_path, dict(BASE, **{node: outcome}), p38=parent, p43=parent)
            assert len(problems) == 1 and problems[0].startswith(f"{node}: must pass"), (node, outcome, problems)
        gone = {k: v for k, v in BASE.items() if k != node}
        problems, _ = _run(tmp_path, gone, p38=gone, p43=gone)
        assert problems == [f"{node}: the declared-edited / replacement node is missing"], node


def test_every_new_exp46_node_must_pass_the_generator_contract_included(tmp_path):
    new = {"src/tests/test_exp46_registry.py::test_a[c4f]": "passed",
           "src/tests/test_gen_model_comparison_exp46.py::test_b": "passed",
           "src/tests/test_gen_model_comparison_exp45.py::test_dormant": "passed"}
    problems, report = _run(tmp_path, dict(BASE, **new))
    assert problems == [] and report["new_node_skips"] == []
    for node in new:
        for outcome in ("failed", "error"):
            problems, _ = _run(tmp_path, dict(BASE, **dict(new, **{node: outcome})))
            assert len(problems) == 1 and node in problems[0] and "new exp_45 / exp_46 node" in problems[0]
        problems, _ = _run(tmp_path, dict(BASE, **dict(new, **{node: ("skipped", "NAS not mounted")})))
        assert len(problems) == 1 and "not a registered environment skip" in problems[0], node
    assert rc.NEW_FILE.match("src/tests/test_exp46_x.py::t") and rc.NEW_FILE.match("src/tests/test_exp45_x.py::t")
    assert not rc.NEW_FILE.match("src/tests/test_exp47_x.py::t") and not rc.NEW_FILE.match("src/tests/test_exp4_x.py::t")
    assert not rc.NEW_FILE.match("src/tests/test_gen_model_comparison_exp38.py::t")


def test_the_exp46_environment_skips_are_exp45s_on_the_exp46_copies_with_unchanged_reasons():
    """Codex plan r2 finding 4: ENV_SKIPS keeps every original exp_45 entry and gains, for each copied kit node that
    skips on neuronic for the same reason, its exp_46 node id with the UNCHANGED reason -- nothing else."""
    exp45 = {(n, r) for n, r in rc.ENV_SKIPS if n.startswith("src/tests/test_exp45_")}
    exp46 = {(n, r) for n, r in rc.ENV_SKIPS if n.startswith("src/tests/test_exp46_")}
    assert len(exp45) == 5 and len(exp46) == 4 and exp45 | exp46 == set(rc.ENV_SKIPS)
    assert exp46 <= {(n.replace("test_exp45_", "test_exp46_", 1), r) for n, r in exp45}
    assert {n for n, _ in exp45} - {n.replace("test_exp46_", "test_exp45_", 1) for n, _ in exp46} == {
        "src/tests/test_exp45_registry.py::test_the_real_rf_100_percent_endpoint_satisfies_the_exact_schema"}


def test_a_parent_node_may_go_only_as_a_registered_replacement_that_passes(tmp_path, monkeypatch):
    old, new = "src/tests/test_a.py::test_x", "src/tests/test_a.py::test_x_replaced"
    merged = {k: v for k, v in BASE.items() if k != old}
    problems, _ = _run(tmp_path, merged)
    assert problems == [f"{old}: present in 3593f34c, missing in the merged tree",
                        f"{old}: present in 4a7e3b3a, missing in the merged tree"]
    monkeypatch.setattr(rc, "REPLACED", {old: new})
    problems, report = _run(tmp_path, dict(merged, **{new: "passed"}))
    assert problems == [] and report["allowed_failures"] == [f"{old}: replaced by the registered {new}"] * 2
    problems, _ = _run(tmp_path, dict(merged, **{new: "failed"}))
    assert len(problems) == 1 and problems[0].startswith(f"{new}: must pass")
    problems, _ = _run(tmp_path, merged)
    assert f"{new}: the declared-edited / replacement node is missing" in problems and len(problems) == 3


def test_the_wandb_environment_failures_are_inherited_only_as_the_parents_exact_node_outcome_and_signature(tmp_path):
    """Fix round 1 (Codex code r1 open item 1): on neuronic the flac env's wandb 0.26.1 makes exp_43's kit nodes fail
    in EVERY tree. L1 accepts such a failure as inherited only when a parent junit shows the same node with the same
    outcome and the same normalised signature (checkout roots normalised) -- and L1 carries no L0 deselection forward:
    a node absent from the merged junit (deselected) while a parent ran it refuses."""
    node = "src/tests/test_exp43_kit.py::test_admission_validator_accepts_the_exact_record_and_rehashes"
    msg = "exp43_kit.Refused: the history extractor exited 2: REFUSED: wandb 0.26.1 != the pinned 0.15.4 ({root}/x.py)"
    merged = dict(BASE, **{node: ("error", msg.format(root=ROOTS[0]))})
    same = dict(BASE, **{node: ("error", msg.format(root=ROOTS[2]))})               # the exp_43 parent, its own root
    problems, report = _run(tmp_path, merged, p43=same)
    assert problems == [] and report["allowed_failures"] == [
        f"{node}: inherited from 4a7e3b3a (error: {msg.format(root='<checkout>')})"]
    for parent in (dict(BASE, **{node: "passed"}), dict(BASE, **{node: ("failed", msg.format(root=ROOTS[2]))}),
                   dict(BASE, **{node: ("error", msg.format(root=ROOTS[2]).replace("0.26.1", "0.27.0"))}),
                   dict(BASE, **{node: ("skipped", "wandb absent")})):
        problems, _ = _run(tmp_path, merged, p43=parent)
        assert len(problems) == 1 and problems[0].startswith(f"{node}: error in the merged tree"), parent[node]
    deselected = {k: v for k, v in BASE.items() if k != node}                       # L0's deselection, carried over
    problems, _ = _run(tmp_path, deselected, p43=same)
    assert problems == [f"{node}: present in 4a7e3b3a, missing in the merged tree"]


# ------------------------------------------------------------------ D1 (PREPARED): E7 / E8 registered as declared edits
E7_E8 = ["src/tests/test_exp45_sampler.py::test_both_launch_environments_give_the_same_rank_and_world",
         "src/tests/test_exp38_sampler.py::test_both_launch_environments_give_the_same_rank_and_world"]
# L1 attempt 2 (Yixun 2026-11-19): exp_38's supervisor refusal node restores the signal handlers sv.main() installs
E9 = "src/tests/test_exp38_supervise.py::test_unregistered_runs_are_refused_before_any_work"


def test_e7_e8_must_be_present_and_pass_even_if_absent_from_both_parents(tmp_path):
    """Codex code r3's L1 rule for E7 / E8: each node must PASS in the merged tree, also when NEITHER parent has it; a
    failure, an error or a skip refuses even when a parent has the identical outcome (no inherited-failure exception);
    its absence refuses."""
    for node in E7_E8:
        assert node in rc.DECLARED_EDITED
        parents = {k: v for k, v in BASE.items() if k != node}                    # absent from both parents
        problems, _ = _run(tmp_path, BASE, p38=parents, p43=parents)
        assert problems == [], (node, problems)
        for outcome in ("failed", "error", "skipped"):
            for p in (parents, dict(parents, **{node: outcome})):                 # absent / identical in the parents
                problems, _ = _run(tmp_path, dict(BASE, **{node: outcome}), p38=p, p43=p)
                assert len(problems) == 1 and problems[0].startswith(f"{node}: must pass"), (node, outcome, problems)
        problems, _ = _run(tmp_path, parents, p38=parents, p43=parents)
        assert problems == [f"{node}: the declared-edited / replacement node is missing"], (node, problems)


# ------------------------------------------------------------------ L1 option (A): pytest's collection-error records
# Yixun 2026-11-19 ("Fix as recommended"): the L1 SUITE runs with --continue-on-collection-errors, so every junit holds
# pytest's record of each module that failed to collect. The shape is copied from attempt 1's
# l1/attempt1/merged.junit.xml (read only): an EMPTY classname, the dotted module path as the name, and
# <error message="collection failure"> whose text is the short traceback with the cause on its last "E   " line.
# RAISED / IMPORT are attempt 1's two kinds of text (an exception raised while the module ran; pytest's ImportError
# text); ROOT, MODULE and CAUSE stand for the checkout the suite ran in, the file and the cause.
H35 = ("subprocess.CalledProcessError: Command '['git', '-C', 'ROOT', 'show', '5b065801e9d1c1844461ec84cd0558702c714326:"
       "worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_supervisor.py']' returned non-zero exit status 128.")
OPEN3D = "ModuleNotFoundError: No module named 'open3d'"
RAISED = ("MODULE:87: in <module>\n    SUP_SHA = SHA(SB.real_supervisor())\n"
          "src/tests/exp35_phase_ii_sandbox.py:55: in real_supervisor\n"
          '    return git_show(H35_REAL, f"{X35}/exp35_supervisor.py")\n'
          "src/tests/exp35_phase_ii_sandbox.py:43: in git_show\n"
          '    return subprocess.run(["git", "-C", REPO, "show", f"{rev}:{path}"], capture_output=True, check=True).stdout\n'
          "../../envs/flac/lib/python3.10/subprocess.py:526: in run\n    raise CalledProcessError(retcode, process.args,\n"
          "E   CAUSE")
IMPORT = ("ImportError while importing test module 'ROOT/MODULE'.\n"
          "Hint: make sure your test modules/packages have valid Python names.\nTraceback:\n"
          "../../envs/flac/lib/python3.10/importlib/__init__.py:126: in import_module\n"
          "    return _bootstrap._gcd_import(name[level:], package, level)\n"
          "MODULE:25: in <module>\n    import render_depth as raf_render  # noqa: E402\n"
          "data/RAF/render_depth.py:22: in <module>\n    import open3d as o3d\nE   CAUSE")
ATTEMPT1 = {"src/tests/test_exp35_phase_ii_checks.py": H35, "src/tests/test_exp35_phase_ii_schema.py": H35,
            "src/tests/test_raf_publish.py": OPEN3D, "src/tests/test_raf_readback.py": OPEN3D,
            "src/tests/test_raf_render_depth.py": OPEN3D}
MUST = "must pass (an edited / replacement node or a new exp_45 / exp_46 node), got error"


def _collect(module, cause, root, text=None):
    """One collection-error record as pytest's junit writes it (text escaped as pytest escapes it)."""
    body = (text or (RAISED if cause.startswith("subprocess.") else IMPORT)).replace("CAUSE", cause)
    body = body.replace("ROOT", root).replace("MODULE", module)
    body = body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (f'<testcase classname="" name="{module[:-3].replace("/", ".")}" time="0.000">'
            f'<error message="collection failure">{body}</error></testcase>')


def _runc(tmp_path, merged, p38=None, p43=None, base=BASE, roots=ROOTS):
    """_run with collection records: merged / p38 / p43 map a module to its cause, or to (cause, text); each tree's
    records name its own checkout root; ``base`` are the collected nodes of all three trees."""
    files = []
    for n, tree, root in (("m", merged, roots[0]), ("p38", p38, roots[1]), ("p43", p43, roots[2])):
        pairs = {m: c if isinstance(c, tuple) else (c, None) for m, c in (tree or {}).items()}
        files.append(_xml(tmp_path / f"{n}.xml", base, [_collect(m, c, root, t) for m, (c, t) in pairs.items()]))
    return rc.compare(*(rc.load(f, roots=roots) for f in files))


def _sig(cause):
    return "error: " + cause.replace("ROOT", "<checkout>")


def test_a_collection_record_is_its_module_and_any_other_empty_classname_still_refuses():
    """(a) an empty classname with a dotted test-module name is that module's node id; (e) any other name with an
    empty classname still refuses (exit 2 in the CLI test below); a classname-ful node is unchanged."""
    assert rc.node_id({"classname": "", "name": "src.tests.test_x"}) == "src/tests/test_x.py"
    assert rc.node_id({"classname": "", "name": "src.tests.test_raf_publish"}) == "src/tests/test_raf_publish.py"
    assert rc.node_id({"classname": "src.tests.test_x", "name": "test_y"}) == "src/tests/test_x.py::test_y"
    for name in ("", "conftest", "src.tests", "src.tests.helpers", "src.tests.test_x.TestK", "src..test_x",
                 "src.tests.test_x.py"):
        with pytest.raises(rc.CompareError, match="names no test module"):
            rc.node_id({"classname": "", "name": name})


def test_a_collection_failures_signature_is_its_last_e_line_and_an_identical_parent_record_is_inherited(tmp_path):
    """(b) pytest's message ("collection failure") is the same for every collection error: the signature is the
    record's last "E   " line with the checkout roots normalised; the traceback above it is not part of it. A merged
    record is inherited when a parent holds the identical node, outcome and signature."""
    two = {"src/tests/test_exp35_phase_ii_checks.py": H35, "src/tests/test_raf_publish.py": OPEN3D}
    for i, root in enumerate(ROOTS):                                # each tree names its own checkout
        f = _xml(tmp_path / f"t{i}.xml", {}, [_collect(m, c, root) for m, c in two.items()])
        assert rc.load(f, roots=ROOTS) == {m: ("error", _sig(c)) for m, c in two.items()}
    last = IMPORT.replace("E   CAUSE", "E   ImportError: first\nE   CAUSE")    # several E lines: the last one
    f = _xml(tmp_path / "last.xml", {}, [_collect("src/tests/test_x.py", OPEN3D, ROOTS[0], last)])
    assert rc.load(f, roots=ROOTS) == {"src/tests/test_x.py": ("error", _sig(OPEN3D))}
    problems, report = _runc(tmp_path, ATTEMPT1, p38=ATTEMPT1, p43=ATTEMPT1)  # attempt 1's five, in all three trees
    assert problems == [] and report["verdict"] == "PASS" and report["counts"]["merged"]["error"] == 5
    assert report["allowed_failures"] == [f"{m}: inherited from 3593f34c, 4a7e3b3a ({_sig(ATTEMPT1[m])})"
                                          for m in sorted(ATTEMPT1)]
    x = "src/tests/test_raf_publish.py"
    moved = IMPORT.replace("MODULE:25:", "MODULE:31:")                        # another traceback, the same cause
    problems, report = _runc(tmp_path, {x: OPEN3D}, p43={x: (OPEN3D, moved)})
    assert problems == [] and report["allowed_failures"] == [f"{x}: inherited from 4a7e3b3a ({_sig(OPEN3D)})"]


def test_a_merged_only_collection_error_another_cause_or_another_module_is_a_problem(tmp_path):
    """(c) each of these is a problem: a merged-only collection error, a parent record with another "E   " cause, and
    the same cause on another module."""
    x, y = "src/tests/test_raf_publish.py", "src/tests/test_raf_readback.py"
    problems, _ = _runc(tmp_path, {x: OPEN3D})
    assert len(problems) == 1 and problems[0].startswith(f"{x}: error in the merged tree ({_sig(OPEN3D)})"), problems
    for other in ("ModuleNotFoundError: No module named 'trimesh'", "ImportError: cannot import name 'o3d'", H35):
        problems, _ = _runc(tmp_path, {x: OPEN3D}, p38={x: other}, p43={x: other})
        assert len(problems) == 1 and problems[0].startswith(f"{x}: error in the merged tree"), (other, problems)
    for other in (H35.replace("status 128", "status 1"), H35.replace("exp35_supervisor.py", "exp35_kit.py")):
        problems, _ = _runc(tmp_path, {x: H35}, p38={x: other}, p43={x: other})
        assert len(problems) == 1 and problems[0].startswith(f"{x}: error in the merged tree"), (other, problems)
    problems, report = _runc(tmp_path, {x: OPEN3D, y: OPEN3D}, p38={y: OPEN3D}, p43={y: OPEN3D})
    assert len(problems) == 1 and problems[0].startswith(f"{x}: error in the merged tree"), problems
    assert report["allowed_failures"] == [f"{y}: inherited from 3593f34c, 4a7e3b3a ({_sig(OPEN3D)})"]


def test_a_collection_error_in_a_must_pass_or_new_exp45_exp46_file_is_never_inherited(tmp_path):
    """(d) must_pass is keyed by node, a collection record names a whole FILE: the record of a file that holds a
    must-pass node (compat-edited / declared-edited / replacement) or of a new exp_45 / exp_46 file is never admitted
    by inheritance -- a problem even when both parents hold the identical record."""
    kit = "src/tests/test_exp43_kit.py"                                   # holds a compat-edited and a declared node
    assert any(n.startswith(kit + "::") for n in rc.COMPAT_EDITED) and any(n.startswith(kit + "::") for n in DECLARED)
    nowhere = {k: v for k, v in BASE.items() if not k.startswith(kit + "::")}   # the file collected in no tree
    problems, report = _runc(tmp_path, {kit: OPEN3D}, p38={kit: OPEN3D}, p43={kit: OPEN3D}, base=nowhere)
    assert f"{kit}: {MUST} {_sig(OPEN3D)}" in problems, problems
    assert not [a for a in report["allowed_failures"] if a.startswith(kit)]
    for new in ("src/tests/test_exp46_l1_probe.py", "src/tests/test_exp45_l1_probe.py",
                "src/tests/test_gen_model_comparison_exp46.py"):
        problems, report = _runc(tmp_path, {new: OPEN3D}, p38={new: OPEN3D}, p43={new: OPEN3D})
        assert problems == [f"{new}: {MUST} {_sig(OPEN3D)}"] and report["allowed_failures"] == [], (new, problems)
        problems, _ = _runc(tmp_path, {new: OPEN3D})
        assert problems == [f"{new}: {MUST} {_sig(OPEN3D)}"], (new, problems)
    old = "src/tests/test_exp38_kit_l1_probe.py"                          # an old file: judged by inheritance
    problems, report = _runc(tmp_path, {old: OPEN3D}, p38={old: OPEN3D})
    assert problems == [] and report["allowed_failures"] == [f"{old}: inherited from 3593f34c ({_sig(OPEN3D)})"]
    skipped = ('<testcase classname="" name="src.tests.test_exp46_l1_probe" time="0.000"><skipped message="collection '
               "skipped\">('src/tests/test_exp46_l1_probe.py', 3, 'Skipped: no wandb')</skipped></testcase>")
    files = [_xml(tmp_path / "s.xml", BASE, [skipped]), _xml(tmp_path / "s38.xml", BASE), _xml(tmp_path / "s43.xml", BASE)]
    problems, _ = rc.compare(*(rc.load(f, roots=ROOTS) for f in files))      # an exp_46 file skipped whole
    assert len(problems) == 1 and problems[0].startswith("src/tests/test_exp46_l1_probe.py: a new exp_45 / exp_46 node "
                                                         "was skipped ('collection skipped')"), problems


def test_the_cli_reads_the_collection_records_and_still_refuses_an_unreadable_one(tmp_path, capsys):
    """Attempt 1's five records in all three trees: exit 0 with five inherited failures. (e) A record with an empty
    classname whose name is not a dotted test module, or a collection failure with no "E   " line, refuses: exit 2."""
    roots = ["--merged-root", ROOTS[0], "--exp38-root", ROOTS[1], "--exp43-root", ROOTS[2]]
    trees = [_xml(tmp_path / f"{n}.xml", BASE, [_collect(m, c, r) for m, c in ATTEMPT1.items()])
             for n, r in zip(("m", "a", "b"), ROOTS)]
    assert rc.main(["--merged", trees[0], "--exp38", trees[1], "--exp43", trees[2]] + roots) == 0
    assert "REGRESSION POLICY PASS: 0 problem(s), 5 allowed failure(s)" in capsys.readouterr().out
    no_module = _collect("src/tests/test_x.py", OPEN3D, ROOTS[0]).replace('name="src.tests.test_x"', 'name="conftest"')
    no_cause = _collect("src/tests/test_x.py", OPEN3D, ROOTS[0], text="src/tests/test_x.py:1: in <module>\n    boom()")
    for record, why in ((no_module, "names no test module"), (no_cause, "no 'E   ' line")):
        bad = _xml(tmp_path / "bad.xml", BASE, [record])
        assert rc.main(["--merged", bad, "--exp38", trees[1], "--exp43", trees[2]] + roots) == 2
        assert why in capsys.readouterr().err, why
