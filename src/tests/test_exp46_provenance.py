"""exp_46 c4flip_data_curve -- provenance of the exp_46 commits (plan_c4flip_data_curve.md r3 §3.2 "Provenance",
§3.1, M1).

exp_46 = ``X45f`` (exp_45's frozen tip) + exp_46 commits. This module audits ONLY the exp_46 commits; exp_45's own
history stays with exp_45's tests (test_exp45_merge_provenance.py, its endpoints fixed at ``X45f`` by E4).

* commit A is the verbatim rename-copy of the exp_45 kit and its kit-behaviour test files at ``X45f``: every copied
  path's blob and mode equal the ``X45f`` source under the declared rename map, and commit A adds nothing else but
  this module.
* every commit of ``X45f..HEAD`` (and every uncommitted tracked edit) touches only the allowed paths: ``EXPREL46``,
  ``src/tests/test_exp46_*.py``, ``exp46_fixtures.py``, ``test_gen_model_comparison_exp46.py``, the files of the
  declared edits E1-E5, the generator and the two training sources; the results-publication paths only under the
  results subject; the E0 publication exception (Codex plan r1 finding 6) only under its exact subject and only for
  the 24 E0-cell files (outside the closure, so closure_digest(Y46) == closure_digest(X46)).
* content checks against ``X45f``: E1-E5 change only their named functions / lines; factory.py and diffusion.py
  remove exactly the two refusals and change only the named comment / docstring lines; the generator gains the exp_46
  contract only (no line removed). ``EXPREL45`` and the exp_38 / exp_40 / exp_43 folders are byte-frozen.
"""
import ast
import fnmatch
import os
import re
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
X45F = "1e37ead2a352bd78d50cdda9ed02e95738a4ad04"          # exp_45's frozen tip, the base of exp_46 (plan §3.0)
EXPREL45 = "worklog/worklog_yixun/exp_45_rf_data_curve_claude"
EXPREL46 = "worklog/worklog_yixun/exp_46_c4flip_data_curve_claude"
COMMIT_A_SUBJECT = ("exp_46 L0 commit A: verbatim rename-copy of the exp_45 kit and its kit-behaviour tests "
                    "at X45f")

KIT = ("registry.py", "supervise.py", "submit_neuronic.py", "train.sbatch", "init_digest.py", "eval.py",
       "collect.py", "table.py", "regression_compare.py")
# test files with at least one kit-behaviour node (the node-level disposition: commit A message, worklog); the two
# exp_45-history files test_exp45_merge_provenance.py and test_exp45_command_sheets.py are NOT copied
KIT_TESTS = ("ddp_evidence", "eval", "isolation", "launchers", "receipts", "registry", "regression_compare",
             "sampler", "storage", "supervise", "table")
RENAME_MAP = {**{f"{EXPREL45}/exp45_{n}": f"{EXPREL46}/exp46_{n}" for n in KIT},
              **{f"src/tests/test_exp45_{n}.py": f"src/tests/test_exp46_{n}.py" for n in KIT_TESTS},
              "src/tests/exp45_fixtures.py": "src/tests/exp46_fixtures.py"}
NOT_COPIED = ("src/tests/test_exp45_merge_provenance.py", "src/tests/test_exp45_command_sheets.py")

GEN = "worklog/worklog_yixun/gen_model_comparison.py"
FACTORY, DIFFUSION = "src/training/factory.py", "src/training/diffusion.py"
FROZEN = (EXPREL45, "worklog/worklog_yixun/exp_38_data_efficiency_claude", "worklog/worklog_yixun/exp_40_yaw_aug_c16_claude",
          "worklog/worklog_yixun/exp_43_std_aug_claude")
# the declared test edits (plan r3 §3.1): {file: (its named functions, its named module-level lines)}
E_EDITS = {
    "src/tests/test_yaw_aug_flip.py": (("test_constructor_refuses_flip_with_offset_step",), ()),               # E1
    "src/tests/test_yaw_aug_flip_factory.py": (("test_true_flip_with_offset_step_is_refused",),               # E2
                                               ("module docstring line 10",)),
    "src/tests/test_exp43_kit.py": (("test_config_contract_through_the_real_factory",), ()),                  # E3
    "src/tests/test_exp45_merge_provenance.py": (("merge_commit",                                           # E4
                                                  "test_later_exp45_commits_touch_only_the_allowed_paths"),
                                                 ("the X45F constant",)),
    "src/tests/test_gen_model_comparison_depth_canon.py": (("test_every_pre_exp25_row_renders_byte_identically",),
                                                           ()),                                              # E5
}
ALWAYS = (f"{EXPREL46}/*", "src/tests/test_exp46_*.py", "src/tests/exp46_fixtures.py",
          "src/tests/test_gen_model_comparison_exp46.py", *E_EDITS, GEN, FACTORY, DIFFUSION)
RESULTS_ONLY = ("outputs_FLAC/exp46_c4flip_data_curve/eval/*", "worklog/worklog_yixun/model_comparison.md")
RESULTS_SUBJECT = "exp_46 results publication"            # the subject prefix of a results-publication commit
# the E0 publication commit Y46 (Codex plan r1 finding 6): ITS exact subject, and besides EXPREL46 ONLY the 24 files of
# the eight E0 cells -- each gate's registered checkpoint epoch and conditioning suffix (exp46_eval.e0_cells naming,
# test-checked); the results-publication rule is not widened
E0_SUBJECT = re.compile(r"exp_46 records: E0 PASS at [0-9a-f]{40}")
E0_GATES = {"E0a": ("epoch=34-step=40000", ""), "E0b": ("epoch=8-step=40000", "_fa_invariant_a4"),
            "E0c": ("epoch=8-step=40000", ""), "E0d": ("epoch=8-step=40000", "")}
E0_FARM = "outputs_FLAC/exp46_c4flip_data_curve/eval"
E0_FILES = frozenset(f"{E0_FARM}/{g}/{ck}_metrics_1_1.0_de_{g}_K{K}_s42{sfx}{ext}"
                     for g, (ck, sfx) in E0_GATES.items() for K in (8, 1)
                     for ext in (".json", ".stream.json", ".receipt.json"))


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def exp46_commits():
    """The first-parent exp_46 commits after X45f, oldest first: (sha, subject)."""
    out = git("log", "--first-parent", "--reverse", "--format=%H%x00%s", f"{X45F}..HEAD")
    return [tuple(line.split("\0", 1)) for line in out.splitlines() if line]


def tree_entry(rev, path):
    """(mode, blob sha) of ``path`` at ``rev``."""
    out = git("ls-tree", rev, "--", path).split()
    assert len(out) == 4 and out[1] == "blob" and out[3] == path, (rev, path, out)
    return out[0], out[2]


def commit_a():
    hits = [sha for sha, subject in exp46_commits() if subject == COMMIT_A_SUBJECT]
    assert len(hits) == 1, f"exactly one commit A expected after X45f, found {hits}"
    return hits[0]


def test_commit_a_is_the_x45f_sources_under_the_declared_rename_map():
    a = commit_a()
    changes = [line.split("\t") for line in git("diff-tree", "-r", "--no-commit-id", "--no-renames",
                                                  "--name-status", f"{a}^", a).splitlines()]
    assert all(status == "A" for status, _ in changes), changes            # commit A only ADDS files
    added = {path for _, path in changes}
    assert added == set(RENAME_MAP.values()) | {"src/tests/test_exp46_provenance.py"}
    for src, dst in sorted(RENAME_MAP.items()):
        assert tree_entry(a, dst) == tree_entry(X45F, src), (src, dst)   # blob AND mode equal
    for path in NOT_COPIED:
        assert path.replace("test_exp45_", "test_exp46_") not in added


def test_the_rename_map_covers_the_nine_kit_files_and_every_kit_behaviour_test_file():
    kit45 = sorted(p for p in git("ls-tree", "--name-only", X45F, f"{EXPREL45}/").split()
                   if os.path.basename(p).startswith("exp45_") and not p.endswith((".json", ".md", ".sha256")))
    assert kit45 == sorted(k for k in RENAME_MAP if k.startswith(EXPREL45)), kit45
    tests45 = sorted(p for p in git("ls-tree", "--name-only", X45F, "src/tests/").split()
                     if os.path.basename(p).startswith("test_exp45_"))
    assert sorted(set(tests45) - set(NOT_COPIED)) == sorted(k for k in RENAME_MAP if k.startswith("src/tests/test_"))


# ------------------------------------------------------------------ the allowed paths of every exp_46 commit
def matches(path, patterns):
    return any(fnmatch.fnmatchcase(path, pat) for pat in patterns)


def path_allowed(path, subject):
    """May an exp_46 commit with this ``subject`` touch ``path``? (plan r3 §3.2 "Provenance" + the E0 exception)"""
    if matches(path, ALWAYS):
        return True
    if matches(path, RESULTS_ONLY) and subject.startswith(RESULTS_SUBJECT):
        return True
    return E0_SUBJECT.fullmatch(subject) is not None and path in E0_FILES


def test_every_exp46_commit_and_the_working_tree_touch_only_the_allowed_paths():
    assert subprocess.run(["git", "merge-base", "--is-ancestor", X45F, "HEAD"], cwd=REPO).returncode == 0
    assert git("rev-list", "--merges", f"{X45F}..HEAD").strip() == "", "no merge after X45f"
    commits = exp46_commits()
    assert commits and commit_a() in {sha for sha, _ in commits}
    problems = []
    for sha, subject in commits:
        for line in git("diff-tree", "-r", "--no-commit-id", "--no-renames", "--name-status", sha).splitlines():
            status, path = line.split("\t", 1)
            ok = path_allowed(path, subject)
            if ok and E0_SUBJECT.fullmatch(subject) and path in E0_FILES and status != "A":
                ok = False                                         # the E0 exception ADDS its files
            if not ok:
                problems.append(f"{sha[:10]} ({subject[:50]}): {status} {path}")
    for path in git("diff", "--name-only", "HEAD").split("\n"):          # uncommitted tracked edits too
        if path and not matches(path, ALWAYS):
            problems.append(f"working tree: {path}")
    assert not problems, problems[:10]


def test_exp45_and_the_exp38_exp40_exp43_folders_are_byte_frozen_at_x45f():
    for folder in FROZEN:
        for args in (("diff", "--quiet", X45F, "HEAD", "--", folder), ("diff", "--quiet", X45F, "--", folder)):
            assert subprocess.run(["git", *args], cwd=REPO).returncode == 0, f"{folder} differs from X45f"


def _e0_cell_files():
    """The 24 files of the eight E0 cells (metrics, stream, receipt) at their farm paths, by exp46_eval's own naming."""
    import importlib.util
    sys.path.insert(0, os.path.join(REPO, EXPREL46))
    spec = importlib.util.spec_from_file_location("exp46_eval_provenance", os.path.join(REPO, EXPREL46, "exp46_eval.py"))
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev, sorted(c[k] for c in ev.e0_cells() for k in ("metrics", "stream", "receipt"))


def test_the_e0_publication_rule_admits_exactly_the_24_registered_e0_cell_files():
    """Codex plan r1 finding 6: the exact subject ``exp_46 records: E0 PASS at <40-hex>`` may add exactly the 24
    E0-cell files (each gate's registered epoch and suffix) and nothing else outside EXPREL46 -- a wrong epoch, a wrong
    suffix, a wrong seed, an extra file or another subject is refused; the results rule is not widened; every E0
    path lies outside the training closure, so the E0 publication commit keeps closure_digest(Y46) == closure_digest(X46)."""
    ev, e0 = _e0_cell_files()
    assert sorted(E0_FILES) == e0 and len(e0) == 24 and all(p.startswith(E0_FARM + "/E0") for p in e0)
    subject = "exp_46 records: E0 PASS at " + "0123456789abcdef" * 2 + "01234567"
    assert all(path_allowed(p, subject) for p in e0)
    combos = [f"{E0_FARM}/{g}/epoch={ep}-step=40000_metrics_1_1.0_de_{g}_K{K}_s42{sfx}{ext}"
              for g in ("E0a", "E0b", "E0c", "E0d") for ep in (34, 8) for K in (8, 1)
              for sfx in ("", "_fa_invariant_a4") for ext in (".json", ".stream.json", ".receipt.json")]
    assert len(combos) == 96 and sorted(p for p in combos if path_allowed(p, subject)) == e0
    for wrong in (f"{E0_FARM}/E0a/epoch=8-step=40000_metrics_1_1.0_de_E0a_K8_s42.json",          # E0a's epoch is 34
                  f"{E0_FARM}/E0b/epoch=8-step=40000_metrics_1_1.0_de_E0b_K8_s42.json",          # E0b needs its suffix
                  e0[0].replace("_s42", "_s43"), f"{E0_FARM}/E0a/extra.json",                    # seed / extra file
                  e0[0][:-len(".json")] + ".dump_manifest.json", re.sub(r"_K[18]_", "_K4_", e0[0]),
                  "worklog/worklog_yixun/model_comparison.md", f"{E0_FARM}/exp46_C4Ff025_l40_2x32/x.json",
                  "src/data/dataset.py", e0[0].replace("exp46_c4flip_data_curve", "exp45_rf_data_curve")):
        assert not path_allowed(wrong, subject), wrong
    for other in ("exp_46 records: E0 PASS at X", "exp_46 records: E0 PASS at " + "a" * 39,
                  "exp_45 records: E0 PASS at " + "a" * 40, "exp_46 records: notes",
                  subject + " (amended)", " " + subject):
        assert not any(path_allowed(p, other) for p in e0), other           # the E0 rule needs ITS subject
    # the results-publication rule is unchanged: the farm + the table under its own subject, nothing else
    assert path_allowed(f"{E0_FARM}/exp46_C4Ff025_l40_2x32/x.json", RESULTS_SUBJECT + ": 40 cells")
    assert not path_allowed("src/data/dataset.py", RESULTS_SUBJECT + ": x")
    reg = ev.reg
    assert not [p for p in e0 + [reg.E0_RECEIPT, reg.E0_RECEIPT + ".sha256"] if reg._in_closure(p)]


# ------------------------------------------------------------------ content checks against X45f
def _hunks(path):
    """[(old_start, old_count, new_start, new_count)] of ``git diff -U0 X45f HEAD -- path``."""
    out = []
    for line in git("diff", "-U0", "--no-color", X45F, "HEAD", "--", path).splitlines():
        m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
        if m:
            a, b, c, d = m.groups()
            out.append((int(a), 1 if b is None else int(b), int(c), 1 if d is None else int(d)))
    return out


def _show(rev, path):
    return git("show", f"{rev}:{path}")


def _function_lines(src, names):
    """{line numbers} of the named top-level functions (decorators included)."""
    lines = set()
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            lines |= set(range(start, node.end_lineno + 1))
    return lines


def _named_lines(src, what):
    tree = ast.parse(src)
    if what == "module docstring line 10":
        doc = tree.body[0]
        assert isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant) and doc.end_lineno >= 10
        return {10}
    if what == "the X45F constant":
        hits = [n for n in tree.body if isinstance(n, ast.Assign) and [getattr(t, "id", None) for t in n.targets] == ["X45F"]]
        return set(range(hits[0].lineno, hits[0].end_lineno + 1)) if hits else set()
    raise AssertionError(what)


def edit_problems(old, new, hunks, functions, extra):
    """Problems of one declared edit: a removed (X45f-side) or added (HEAD-side) line outside the declared functions /
    module-level lines of that version, a declared function missing, or no change at all."""
    if not (_function_lines(old, functions) and _function_lines(new, functions)):
        return ["a declared function is missing"]
    allowed_old = _function_lines(old, functions) | {ln for w in extra for ln in _named_lines(old, w)}
    allowed_new = _function_lines(new, functions) | {ln for w in extra for ln in _named_lines(new, w)}
    out = [] if hunks else ["the declared edit is missing"]
    for a, b, c, d in hunks:
        if not set(range(a, a + b)) <= allowed_old:
            out.append(f"removed lines {a}..{a + b - 1} outside {functions} / {extra}")
        if not set(range(c, c + d)) <= allowed_new:
            out.append(f"added lines {c}..{c + d - 1} outside {functions} / {extra}")
    return out


@pytest.mark.parametrize("path", sorted(E_EDITS))
def test_each_declared_edit_changes_only_its_named_functions_and_lines(path):
    """plan r3 §3.1 E1-E5, content-checked: every line removed from the X45f file and every line added at HEAD lies in
    the declared functions (or the declared module-level lines) of that version; the edit exists."""
    functions, extra = E_EDITS[path]
    assert edit_problems(_show(X45F, path), _show("HEAD", path), _hunks(path), functions, extra) == [], path


def test_the_declared_edit_check_refuses_a_change_outside_the_named_functions():
    old = '"""doc"""\nimport os\n\n\ndef keep():\n    return 1\n\n\ndef test_named():\n    assert 1\n'
    new = '"""doc"""\nimport os\nimport sys\n\n\ndef keep():\n    return 1\n\n\ndef test_named():\n    assert 2\n'
    ok = edit_problems(old, new.replace("import sys\n", ""), [(10, 1, 10, 1)], ("test_named",), ())
    assert ok == []
    assert edit_problems(old, new, [(2, 0, 3, 1), (10, 1, 11, 1)], ("test_named",), ()) == [
        "added lines 3..3 outside ('test_named',) / ()"]
    assert edit_problems(old, old.replace("return 1", "return 2"), [(6, 1, 6, 1)], ("test_named",), ()) == [
        "removed lines 6..6 outside ('test_named',) / ()", "added lines 6..6 outside ('test_named',) / ()"]
    assert edit_problems(old, old, [], ("test_named",), ()) == ["the declared edit is missing"]
    assert edit_problems(old, old.replace("test_named", "renamed"), [(9, 1, 9, 1)], ("test_named",), ()) == [
        "a declared function is missing"]


def test_paths_outside_the_allowed_set_are_refused_under_an_ordinary_subject():
    for path in ("src/data/dataset.py", "README.md", f"{EXPREL45}/exp45_registry.py", "src/tests/test_exp45_eval.py",
                 "src/tests/conftest.py", "worklog/worklog_yixun/model_comparison.md", f"{E0_FARM}/E0a/x.json",
                 "outputs_FLAC/exp45_rf_data_curve/eval/x.json", "worklog/worklog_yixun/HANDOFF.md",
                 "worklog/worklog_yixun/exp_43_std_aug_claude/FLAC_AR_YAWAUG_FLIP.json", "src/data/yaw_rotation.py"):
        assert not path_allowed(path, "exp_46 L0 X: anything"), path
    for path in (f"{EXPREL46}/exp46_registry.py", f"{EXPREL46}/c4flip_data_curve_worklog.md",
                 "src/tests/test_exp46_x.py", "src/tests/exp46_fixtures.py", GEN, FACTORY, DIFFUSION, *E_EDITS):
        assert path_allowed(path, "exp_46 L0 X: anything"), path


def _lines(src, start, count):
    return src.splitlines()[start - 1:start - 1 + count]


def test_the_two_source_edits_remove_exactly_the_two_refusals_and_change_only_the_named_comment_lines():
    """plan r3 §3.1 / §3.2 "Provenance": factory.py and diffusion.py against X45f -- exactly the registered hunks:
    each refusal of the flip + offset_step pair removed whole, the comment / docstring lines in their named ranges
    (diffusion.py 294-298 and 752-758 at X45f), nothing else."""
    want = {FACTORY: [(90, 0, 91, 2), (98, 4, 99, 0)],
            DIFFUSION: [(296, 3, 296, 3), (305, 4, 304, 0), (758, 1, 754, 2)]}
    for path, hunks in want.items():
        assert _hunks(path) == hunks, path
    old_f, new_f = _show(X45F, FACTORY), _show("HEAD", FACTORY)
    refusal = "\n".join(_lines(old_f, 98, 4))
    assert refusal.lstrip().startswith('if "offset_step" in block:') and "raise ValueError(" in refusal \
        and "scope for exp_43" in refusal
    assert all(ln.lstrip().startswith("#") for ln in _lines(new_f, 91, 2))
    old_d, new_d = _show(X45F, DIFFUSION), _show("HEAD", DIFFUSION)
    refusal = "\n".join(_lines(old_d, 305, 4))
    assert refusal.lstrip().startswith("if yaw_aug_flip and yaw_aug_offset_step is not None:") \
        and "raise ValueError(" in refusal and "scope for exp_43" in refusal
    assert all(ln.lstrip().startswith("#") for ln in _lines(old_d, 296, 3) + _lines(new_d, 296, 3))
    assert 758 in _docstring_lines(old_d) and {754, 755} <= _docstring_lines(new_d)   # docstring lines, both sides
    assert "scope for exp_43" not in new_f + new_d                         # both refusals are gone


def _docstring_lines(src):
    """{line numbers} of every function / class docstring of ``src``."""
    out = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body \
                and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant) \
                and isinstance(n.body[0].value.value, str):
            out |= set(range(n.body[0].lineno, n.body[0].end_lineno + 1))
    return out


def test_the_generator_gains_the_exp46_contract_only():
    """plan r3 §3.2 "Generator": additions only -- no line of the X45f generator removed -- and every added hunk is
    exp_46's (it names exp_46 / exp46 / EXP46)."""
    num = git("diff", "--numstat", X45F, "HEAD", "--", GEN).split()
    assert num and num[1] == "0", f"{GEN}: {num[1] if num else '?'} line(s) of the X45f generator removed"
    new = _show("HEAD", GEN)
    for _, _, c, d in _hunks(GEN):
        text = "\n".join(_lines(new, c, d))
        assert re.search(r"exp_?46|EXP46", text), text[:200]
