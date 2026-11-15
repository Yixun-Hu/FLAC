"""exp_45 rf_data_curve — merge provenance and the later-commit path audit (plan_rf_data_curve.md r3 §3.1, §3.3).

The merge commit M of exp_38's tip ``3593f34c`` into ``exp-45-rf-data-curve`` (first parent ``738a5de7``) is
checked path by path against a REGISTERED mapping: exp_38-side paths (data side, ``run_evidence.py``,
``train.py``, ``defaults.ini``, the exp_38 folder and tests) equal ``3593f34c``; exp_43-side paths (the yaw-aug
path, the exp_40 / exp_43 folders and tests) equal ``4a7e3b3a``; the exp_45 folder equals the first parent;
paths no side changed equal every parent; the two declared conflict files equal their declared union
resolutions. Membership, file modes and deletions are compared over the union of the trees. Commits after M are
audited as a diff against M (the allowed exp_45 paths only), and the exp_38 / exp_40 / exp_43 folders stay
byte-frozen. CPU-only; reads git objects only.

Written by the exp_45 Coder (Claude Opus 5.5).
"""
import ast
import difflib
import fnmatch
import os
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = "2b42468c8f429b1d93da93b3e4ddb7bd6f8f1932"           # merge base of the two lineages
EXP38_TIP = "3593f34cad4e6e4415a572e93d16ee83465f8ae8"      # origin/exp-38-data-efficiency (the only commit merged)
EXP43_PIN = "4a7e3b3a948388434bc102b8bd7e7aba94648d71"      # the exp_43 training pin (exp_45's base)
FIRST_PARENT = "738a5de77809dd8c4fffbea22354060ce00f50db"   # exp-45-rf-data-curve before the merge (plan r3 approved)
EXPREL = "worklog/worklog_yixun/exp_45_rf_data_curve_claude"
EXP38REL = "worklog/worklog_yixun/exp_38_data_efficiency_claude"
EXP40REL = "worklog/worklog_yixun/exp_40_yaw_aug_c16_claude"
EXP43REL = "worklog/worklog_yixun/exp_43_std_aug_claude"
GEN = "worklog/worklog_yixun/gen_model_comparison.py"
CANON_TEST = "src/tests/test_gen_model_comparison_depth_canon.py"

# ---- the registered per-path mapping (fnmatch patterns; '*' also matches '/')
EXP38_SIDE = (
    "data/AR/train_frac0[257][05]_s2026.json", "data/AR/train_frac_manifest_s2026.json",
    "data/AR/train_frac_manifest_s2026.sha256", "defaults.ini", "train.py",
    "src/configs/dataset_configs/AR/train/acousticroom_train_frac0[257][05].json",
    "src/configs/dataset_configs/custom_metadata/AR_md.py", "src/data/dataset.py", "src/training/run_evidence.py",
    "src/tests/exp38_fixtures.py", "src/tests/fixtures/exp38_e0a_replay_bfc4ea4b/*", "src/tests/test_exp38_*.py",
    "src/tests/test_ar_md_restrict_context.py", "src/tests/test_run_evidence.py",
    "src/tests/test_gen_model_comparison_exp38.py", f"{EXP38REL}/*")
EXP43_SIDE = (
    "CLAUDE.md", "outputs_FLAC/exp40_yaw_aug_c16/farm/*", "src/data/yaw_mirror.py", "src/data/yaw_offset_step.py",
    "src/training/diffusion.py", "src/training/factory.py", "src/tests/exp40_yaw_aug_golden_capture.py",
    "src/tests/exp43_yaw_aug_flip_golden_capture.py", "src/tests/fixtures/exp40_yaw_aug_enabled_golden.json",
    "src/tests/fixtures/exp43_yaw_aug_flip_golden.json", "src/tests/test_exp40_*.py", "src/tests/test_exp43_*.py",
    "src/tests/test_yaw_aug_c16*.py", "src/tests/test_yaw_aug_flip*.py", "src/tests/test_yaw_mirror.py",
    f"{EXP40REL}/*", f"{EXP43REL}/*", "worklog/worklog_yixun/HANDOFF.md", "worklog/worklog_yixun/issue_report.md",
    "worklog/worklog_yixun/master_experiment_tracker.md")
EXP45_SIDE = (f"{EXPREL}/*",)
# the two declared conflict resolutions (union of both parents' edits), as git blob ids
CONFLICTS = {GEN: "9c60c21194cfc6c8eec0d91d7a08f7a285081e1d",
             CANON_TEST: "624080472cb80ecda3dc468220969f7c0488b606"}
# the one line the union writes itself: both parents replaced the same base line, each with its own version
SYNTHESIZED = {CANON_TEST: {"    rest = [ln for ln in cur if ln not in exp25 and ln not in data_curve "
                            "and ln not in exp40 and ln not in data_eff]"}}
SUPERSEDED = {CANON_TEST: {"    rest = [ln for ln in cur if ln not in exp25 and ln not in data_curve and ln not in exp40]",
                           "    rest = [ln for ln in cur if ln not in exp25 and ln not in data_curve and ln not in data_eff]"}}

# ---- paths a commit after the merge may touch (plan §3.1)
ALLOWED_AFTER_MERGE = (f"{EXPREL}/*", "src/tests/test_exp45_*.py", "src/tests/exp45_fixtures.py",
                       "src/tests/test_exp40_kit.py", "src/tests/test_exp43_kit.py", GEN,
                       "src/tests/test_gen_model_comparison_exp45.py")
RESULTS_ONLY = ("outputs_FLAC/exp45_rf_data_curve/eval/*", "worklog/worklog_yixun/model_comparison.md")
RESULTS_SUBJECT = "exp_45 results publication"            # the subject prefix of a results-publication commit
COMPAT_EDITS = {"src/tests/test_exp40_kit.py": "test_prefigure_parse_matches_the_expected_values_type_strict",
                "src/tests/test_exp43_kit.py": "test_prefigure_parse_matches_the_expected_values_type_strict"}


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True, check=True).stdout


def tree(commit):
    """{path: (mode, type, object id)} of every entry of ``commit``'s tree."""
    out = {}
    for entry in git("ls-tree", "-r", "-z", "--full-tree", commit).split("\0"):
        if entry:
            meta, path = entry.split("\t", 1)
            mode, kind, oid = meta.split()
            out[path] = (mode, kind, oid)
    return out


def changed(a, b):
    ta, tb = tree(a), tree(b)
    return {p for p in set(ta) | set(tb) if ta.get(p) != tb.get(p)}


def matches(path, patterns):
    return any(fnmatch.fnmatchcase(path, pat) for pat in patterns)


def side_of(path):
    sides = [name for name, pats in (("exp38", EXP38_SIDE), ("exp43", EXP43_SIDE), ("exp45", EXP45_SIDE))
             if matches(path, pats)]
    return sides


def merge_commit():
    """The single merge commit after FIRST_PARENT; its parents must be exactly (FIRST_PARENT, EXP38_TIP)."""
    rows = [ln.split() for ln in git("rev-list", "--merges", "--parents", "--ancestry-path",
                                     f"{FIRST_PARENT}..HEAD").split("\n") if ln.strip()]
    assert len(rows) == 1, f"expected exactly one merge after {FIRST_PARENT[:8]}, found {len(rows)}"
    assert rows[0][1:] == [FIRST_PARENT, EXP38_TIP], f"merge parents {rows[0][1:]}"
    return rows[0][0]


def blob(commit, path):
    return subprocess.run(["git", "-C", REPO, "show", f"{commit}:{path}"], capture_output=True, check=True).stdout


# --------------------------------------------------------------------------- the merge commit
def test_the_merge_is_exp38s_tip_into_the_exp45_branch_and_nothing_else():
    m = merge_commit()
    assert git("merge-base", FIRST_PARENT, EXP38_TIP).strip() == BASE
    for anc in (BASE, EXP38_TIP, EXP43_PIN, FIRST_PARENT):
        assert subprocess.run(["git", "-C", REPO, "merge-base", "--is-ancestor", anc, m]).returncode == 0
    # no later exp_38 commit (records or otherwise) entered the branch: M's second-parent history is 3593f34c's
    assert git("rev-list", f"{EXP38_TIP}..{m}^2").strip() == ""


def test_the_registered_mapping_classifies_every_changed_path_exactly_once():
    d38, d43, d45 = changed(BASE, EXP38_TIP), changed(BASE, EXP43_PIN), changed(EXP43_PIN, FIRST_PARENT)
    assert d38 & d43 == set(CONFLICTS), sorted(d38 & d43)
    assert not d45 & (d38 | d43)
    for paths, want in ((d38 - set(CONFLICTS), "exp38"), (d43 - set(CONFLICTS), "exp43"), (d45, "exp45")):
        bad = sorted(p for p in paths if side_of(p) != [want])
        assert not bad, f"{want}: unregistered / ambiguous paths {bad[:5]}"
    for pats, paths in ((EXP38_SIDE, d38), (EXP43_SIDE, d43), (EXP45_SIDE, d45)):
        stale = [pat for pat in pats if not any(fnmatch.fnmatchcase(p, pat) for p in paths)]
        assert not stale, f"registered patterns that match no changed path: {stale}"
    for path in CONFLICTS:
        assert side_of(path) == [], f"{path} is a conflict file and must not belong to a side"


def test_every_path_of_the_merge_commit_comes_from_its_registered_parent():
    m = merge_commit()
    tm, tb, t38, t43, t45 = (tree(c) for c in (m, BASE, EXP38_TIP, EXP43_PIN, FIRST_PARENT))
    problems = []
    for path in sorted(set(tm) | set(tb) | set(t38) | set(t43) | set(t45)):
        got = tm.get(path)
        if path in CONFLICTS:
            want = ("100644", "blob", CONFLICTS[path])
        else:
            sides = side_of(path)
            if sides == ["exp38"]:
                want = t38.get(path)
            elif sides == ["exp43"]:
                want = t43.get(path)
            elif sides == ["exp45"]:
                want = t45.get(path)
            else:
                want = tb.get(path)
                if not (t38.get(path) == t43.get(path) == t45.get(path) == want):
                    problems.append(f"{path}: changed by a parent but in no registered side")
                    continue
        if got != want:                                  # mode, object id, and presence / deletion
            problems.append(f"{path}: merge has {got}, registered parent has {want}")
    assert not problems, problems[:10]


def _inserted(old, new):
    sm = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    return [ln for tag, _, _, j1, j2 in sm.get_opcodes() if tag in ("insert", "replace") for ln in new[j1:j2]]


@pytest.mark.parametrize("path", sorted(CONFLICTS))
def test_the_conflict_resolutions_are_the_declared_unions(path):
    m = merge_commit()
    text = {c: blob(c, path).decode().split("\n") for c in (BASE, EXP38_TIP, FIRST_PARENT, m)}
    res = text[m]
    for side in (EXP38_TIP, FIRST_PARENT):                     # nothing either side added is lost
        missing = [ln for ln in _inserted(text[BASE], text[side])
                   if ln not in res and ln not in SUPERSEDED.get(path, set())]
        assert not missing, (side[:8], missing[:3])
    for ln in SUPERSEDED.get(path, set()):                     # ... except the replaced line, which the union replaces
        assert ln not in res
    known = set(text[BASE]) | set(text[EXP38_TIP]) | set(text[FIRST_PARENT]) | SYNTHESIZED.get(path, set())
    assert [ln for ln in res if ln not in known] == []
    assert git("rev-parse", f"{m}:{path}").strip() == CONFLICTS[path]


# --------------------------------------------------------------------------- after the merge
def _function_free(src, name):
    """``src`` with the top-level function ``name`` replaced by a marker (the rest must stay byte-equal)."""
    lines = src.split("\n")
    nodes = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1, name
    start = min([nodes[0].lineno] + [d.lineno for d in nodes[0].decorator_list]) - 1
    return "\n".join(lines[:start] + ["<compat edit>"] + lines[nodes[0].end_lineno:])


def test_later_exp45_commits_touch_only_the_allowed_paths():
    m = merge_commit()
    assert git("rev-list", "--merges", f"{m}..HEAD").strip() == "", "no merge after the provenance merge"
    problems = []
    for commit in git("rev-list", f"{m}..HEAD").split():
        subject = git("log", "-1", "--format=%s", commit).strip()
        for path in git("diff-tree", "--no-commit-id", "-r", "--name-only", commit).split("\n"):
            if not path:
                continue
            if matches(path, ALLOWED_AFTER_MERGE):
                continue
            if matches(path, RESULTS_ONLY) and subject.startswith(RESULTS_SUBJECT):
                continue
            problems.append(f"{commit[:10]} ({subject[:40]}): {path}")
    for path in git("diff", "--name-only", "HEAD").split("\n"):      # uncommitted tracked edits too
        if path and not matches(path, ALLOWED_AFTER_MERGE):
            problems.append(f"working tree: {path}")
    assert not problems, problems[:10]
    # the generator gains the exp_45 contract only: additions, no line of the merged generator removed
    num = git("diff", "--numstat", m, "--", GEN).split()
    assert not num or num[1] == "0", f"{GEN}: {num[1]} merged line(s) removed after the merge"
    for path, fn in COMPAT_EDITS.items():                            # the compat edits stay inside their test
        assert _function_free(blob("HEAD", path).decode(), fn) == _function_free(blob(m, path).decode(), fn)


@pytest.mark.parametrize("folder,parent", [(EXP38REL, EXP38_TIP), (EXP40REL, EXP43_PIN), (EXP43REL, EXP43_PIN)])
def test_the_exp38_exp40_exp43_folders_are_byte_frozen(folder, parent):
    for args in (("diff", "--quiet", parent, "HEAD", "--", folder), ("diff", "--quiet", parent, "--", folder)):
        assert subprocess.run(["git", "-C", REPO, *args]).returncode == 0, f"{folder} differs from {parent[:8]}"
