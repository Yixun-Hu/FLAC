"""exp_46 c4flip_data_curve -- provenance of the exp_46 commits (plan_c4flip_data_curve.md r3 §3.2 "Provenance",
§3.1, M1).

exp_46 = ``X45f`` (exp_45's frozen tip) + exp_46 commits. This module audits ONLY the exp_46 commits; exp_45's own
history stays with exp_45's tests (test_exp45_merge_provenance.py, its endpoints fixed at ``X45f`` by E4).

* commit A is the verbatim rename-copy of the exp_45 kit and its kit-behaviour test files at ``X45f``: every copied
  path's blob and mode equal the ``X45f`` source under the declared rename map, and commit A adds nothing else but
  this module.
"""
import os
import subprocess

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
