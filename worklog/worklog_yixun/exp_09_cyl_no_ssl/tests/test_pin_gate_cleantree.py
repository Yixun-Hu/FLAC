"""exp-09 pin-gate scoped clean-tree + SHA-override tests (integrative-review finding 3).

The real package repo is READ-ONLY and the real worktree state is not a test fixture, so the
scoped-cleanliness gate is exercised against THROWAWAY fixture git repos (the same approach the
Stage-A audit-gate tests use). Both repos, both directions:

* package scope (src/cylindrical_dinov3): dirty -> not clean; clean -> clean; dirty OUTSIDE the
  scope -> still clean (scoping is real);
* worktree executable scope (src/ + exp09 dir): a dirty tracked file blocks; a NEW untracked
  non-output file blocks; an untracked run log/JSON is EXCLUDED; a tracked-modified source
  .json blocks (not excluded); git-unavailable -> None (caller fails closed);
* --expect-package-sha pins the accepted set to exactly one SHA.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")

_EXP09_DIR = Path(__file__).resolve().parents[1]
if str(_EXP09_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP09_DIR))

import assert_arm_configs_exp09 as gate  # noqa: E402


# ------------------------------------------------------------------------------------- #
# fixture git repo with the scoped subtrees populated + committed
# ------------------------------------------------------------------------------------- #
def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _make_repo(tmp_path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src" / "cylindrical_dinov3").mkdir(parents=True)
    (repo / "src" / "cylindrical_dinov3" / "modeling.py").write_text("# pkg code\n")
    (repo / "src" / "conditioners.py").write_text("# flac src code\n")
    exp09 = repo / "worklog" / "worklog_yixun" / "exp_09_cyl_no_ssl"
    exp09.mkdir(parents=True)
    (exp09 / "exp09_launch.sh").write_text("# launcher\n")
    (exp09 / "FLAC_AR_exp09.json").write_text('{"a": 1}\n')
    (repo / "README.md").write_text("root file, outside both scopes\n")
    _git(repo, "init")
    _git(repo, "config", "user.email", "t@t.t")
    _git(repo, "config", "user.name", "t")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "init")
    return repo


# ------------------------------------------------------------------------------------- #
# package scope: src/cylindrical_dinov3
# ------------------------------------------------------------------------------------- #
def test_package_src_clean_when_clean(tmp_path):
    repo = _make_repo(tmp_path)
    assert gate.package_src_clean(str(repo)) is True


def test_package_src_dirty_blocks(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "src" / "cylindrical_dinov3" / "modeling.py").write_text("# MUTATED pkg code\n")
    assert gate.package_src_clean(str(repo)) is False


def test_package_src_untracked_file_blocks(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "src" / "cylindrical_dinov3" / "sneaky.py").write_text("# new pkg file\n")
    assert gate.package_src_clean(str(repo)) is False


def test_package_scope_ignores_dirt_outside_it(tmp_path):
    """A dirty file OUTSIDE src/cylindrical_dinov3 does NOT make the PACKAGE scope dirty."""
    repo = _make_repo(tmp_path)
    (repo / "src" / "conditioners.py").write_text("# changed flac src, not package\n")
    (repo / "README.md").write_text("changed root\n")
    assert gate.package_src_clean(str(repo)) is True


def test_package_src_clean_none_on_git_error(tmp_path):
    assert gate.package_src_clean(str(tmp_path / "not_a_repo")) is None


# ------------------------------------------------------------------------------------- #
# worktree executable scope: src/ + exp09 dir
# ------------------------------------------------------------------------------------- #
def test_exp09_tree_clean_when_clean(tmp_path):
    repo = _make_repo(tmp_path)
    assert gate.exp09_tree_clean(str(repo)) is True


def test_exp09_tree_dirty_tracked_src_blocks(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "src" / "conditioners.py").write_text("# MUTATED flac src\n")
    assert gate.exp09_tree_clean(str(repo)) is False


def test_exp09_tree_dirty_tracked_launcher_blocks(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "worklog" / "worklog_yixun" / "exp_09_cyl_no_ssl" / "exp09_launch.sh").write_text("# edited\n")
    assert gate.exp09_tree_clean(str(repo)) is False


def test_exp09_tree_modified_source_json_blocks(tmp_path):
    """A TRACKED source .json (config) that is MODIFIED blocks — the .json exclusion only applies
    to brand-new UNTRACKED run outputs, never a tracked config edit."""
    repo = _make_repo(tmp_path)
    (repo / "worklog" / "worklog_yixun" / "exp_09_cyl_no_ssl" / "FLAC_AR_exp09.json").write_text('{"a": 2}\n')
    assert gate.exp09_tree_clean(str(repo)) is False


def test_exp09_tree_untracked_log_is_excluded(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "worklog" / "worklog_yixun" / "exp_09_cyl_no_ssl" / "exp09_run.log").write_text("run output\n")
    assert gate.exp09_tree_clean(str(repo)) is True


def test_exp09_tree_untracked_json_output_is_excluded(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "worklog" / "worklog_yixun" / "exp_09_cyl_no_ssl" / "peak_out.json").write_text('{"peak": 1}\n')
    assert gate.exp09_tree_clean(str(repo)) is True


def test_exp09_tree_untracked_non_output_blocks(tmp_path):
    """A NEW untracked NON-output file (e.g. a .py) is executable code — it blocks, not excluded."""
    repo = _make_repo(tmp_path)
    (repo / "worklog" / "worklog_yixun" / "exp_09_cyl_no_ssl" / "sneaky.py").write_text("# new code\n")
    assert gate.exp09_tree_clean(str(repo)) is False


def test_exp09_tree_ignores_dirt_outside_scope(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "README.md").write_text("changed root, outside src/ and exp09/\n")
    assert gate.exp09_tree_clean(str(repo)) is True


def test_exp09_tree_none_on_git_error(tmp_path):
    assert gate.exp09_tree_clean(str(tmp_path / "not_a_repo")) is None


# ------------------------------------------------------------------------------------- #
# _entry_is_output_artifact unit cases
# ------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("status,dst,expect", [
    ("??", "a/b/run.log", True),        # untracked log -> excluded
    ("??", "a/b/out.json", True),       # untracked json -> excluded
    ("??", "a/b/code.py", False),       # untracked non-output -> blocks
    (" M", "a/b/run.log", False),       # TRACKED modified log -> blocks (only untracked excluded)
    ("A ", "a/b/out.json", False),      # staged-added json -> blocks
])
def test_entry_is_output_artifact(status, dst, expect):
    entry = {"status": status, "dst": dst, "is_untracked": status == "??", "is_rename": False}
    assert gate._entry_is_output_artifact(entry) is expect


def test_rename_is_never_an_output_artifact():
    entry = {"status": "R ", "dst": "a/out.json", "is_untracked": False, "is_rename": True}
    assert gate._entry_is_output_artifact(entry) is False


# ------------------------------------------------------------------------------------- #
# --expect-package-sha selection
# ------------------------------------------------------------------------------------- #
def test_accepted_shas_default_is_the_registered_set():
    assert gate.accepted_shas() == set(gate.CYL_ACCEPTED_SHAS)


def test_accepted_shas_override_pins_exactly_one():
    one = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
    assert gate.accepted_shas(one) == {one}
    assert gate.accepted_shas(one) != set(gate.CYL_ACCEPTED_SHAS)


def test_parse_porcelain_z_rename_records_both_sides():
    # `git status --porcelain=v1 -z` renders a rename as `<dst>\0<src>`
    entries = gate._parse_porcelain_z("R  new/path.py\0old/path.py\0")
    assert len(entries) == 1
    assert entries[0]["is_rename"] is True
    assert set(entries[0]["paths"]) == {"new/path.py", "old/path.py"}
