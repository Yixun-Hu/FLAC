"""Tests for the exp_11 launcher's W&B run-identity readback.

Job 3646734 trained perfectly — 8 ranks, 30 steps, checkpoint written, dual logs
byte-identical — and still classified 7, because the readback looked for the run
under ``$WANDB_DIR/wandb`` while wandb had written it under ``$REPO/wandb``:
``train.py:165`` builds ``WandbLogger(project=, name=)`` with no ``save_dir``, so
PL passes its default ``save_dir='.'`` into ``wandb.init`` and that argument wins
over the exported ``WANDB_DIR``.

The fix locates the run by the collision-proof id the launcher generated, which
wandb embeds in the run directory name, across every candidate root. These tests
pin that contract: the real layout is found, both roots are searched, exactly one
match is required, and a directory whose id does not match is rejected.
"""
import importlib.util
import json
import os

import pytest


_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # src/tests/ -> src/ -> repo root
_READBACK_PY = os.path.join(
    _REPO_ROOT, "worklog", "worklog_yixun", "exp_11_fa_orbit_claude",
    "fa_orbit_wandb_readback.py",
)

RUN_ID = "exp11-C4L-1786049318048844980-bd40da20"


def _load_module():
    spec = importlib.util.spec_from_file_location("exp11_wandb_readback", _READBACK_PY)
    assert spec is not None and spec.loader is not None, f"cannot load {_READBACK_PY}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load_module()


def _make_run(root, run_id=RUN_ID, ts="20260806_164917", meta=None):
    """Create the on-disk layout wandb actually produced in job 3646734."""
    run_dir = os.path.join(root, "wandb", f"run-{ts}-{run_id}")
    os.makedirs(os.path.join(run_dir, "files"), exist_ok=True)
    if meta is not None:
        with open(os.path.join(run_dir, "files", "wandb-metadata.json"), "w") as fh:
            json.dump(meta, fh)
    return run_dir


# --------------------------------------------------------------------------- #
# 1. locating the run
# --------------------------------------------------------------------------- #
def test_finds_the_run_under_the_repo_root_where_pl_actually_writes_it(tmp_path):
    repo, wandb_dir = tmp_path / "repo", tmp_path / "savedir"
    (wandb_dir / "wandb").mkdir(parents=True)          # exported WANDB_DIR: empty
    want = _make_run(str(repo))                        # PL's save_dir='.' wins
    got, problems = R.locate_run_dir([str(repo), str(wandb_dir)], RUN_ID)
    assert problems == [] and got == want


def test_finds_the_run_under_wandb_dir_when_that_is_where_it_landed(tmp_path):
    repo, wandb_dir = tmp_path / "repo", tmp_path / "savedir"
    (repo / "wandb").mkdir(parents=True)
    want = _make_run(str(wandb_dir))
    got, problems = R.locate_run_dir([str(repo), str(wandb_dir)], RUN_ID)
    assert problems == [] and got == want


def test_missing_run_is_a_failure_not_a_shrug(tmp_path):
    got, problems = R.locate_run_dir([str(tmp_path)], RUN_ID)
    assert got is None
    assert problems and RUN_ID in problems[0]


def test_ambiguous_id_across_roots_is_rejected(tmp_path):
    repo, wandb_dir = tmp_path / "repo", tmp_path / "savedir"
    _make_run(str(repo), ts="20260806_164917")
    _make_run(str(wandb_dir), ts="20260806_170000")
    got, problems = R.locate_run_dir([str(repo), str(wandb_dir)], RUN_ID)
    assert got is None and any("ambiguous" in p for p in problems)


def test_other_runs_in_the_same_root_are_ignored(tmp_path):
    root = str(tmp_path)
    _make_run(root, run_id="exp11-C8-someotherrun-aaaaaaaa")
    want = _make_run(root)
    got, problems = R.locate_run_dir([root], RUN_ID)
    assert problems == [] and got == want


def test_empty_run_id_is_refused(tmp_path):
    got, problems = R.locate_run_dir([str(tmp_path)], "")
    assert got is None and problems


# --------------------------------------------------------------------------- #
# 2. verifying the identity
# --------------------------------------------------------------------------- #
def test_identity_matches(tmp_path):
    run = _make_run(str(tmp_path), meta={"entity": "ent", "project": "FLAC_exp11_C4L",
                                         "name": "exp11_C4L"})
    assert R.verify_identity(run, RUN_ID, "ent", "FLAC_exp11_C4L", "exp11_C4L") == []


@pytest.mark.parametrize("field,bad", [("entity", "someone-else"),
                                       ("project", "FLAC_exp11_C8"),
                                       ("name", "exp11_C8")])
def test_identity_contradiction_is_a_failure(tmp_path, field, bad):
    meta = {"entity": "ent", "project": "FLAC_exp11_C4L", "name": "exp11_C4L"}
    meta[field] = bad
    run = _make_run(str(tmp_path), meta=meta)
    problems = R.verify_identity(run, RUN_ID, "ent", "FLAC_exp11_C4L", "exp11_C4L")
    assert problems and field in problems[0]


def test_absent_metadata_field_is_not_a_contradiction(tmp_path):
    """wandb-metadata does not always carry every field; only a CONTRADICTION fails."""
    run = _make_run(str(tmp_path), meta={"entity": "ent"})
    assert R.verify_identity(run, RUN_ID, "ent", "FLAC_exp11_C4L", "exp11_C4L") == []
    run2 = _make_run(str(tmp_path / "b"))                  # no metadata file at all
    assert R.verify_identity(run2, RUN_ID, "ent", "P", "N") == []


def test_run_dir_not_carrying_the_id_is_rejected(tmp_path):
    run = _make_run(str(tmp_path), run_id="a-different-id", meta={})
    problems = R.verify_identity(run, RUN_ID, None, None, None)
    assert problems and "does not carry id" in problems[0]


def test_missing_directory_is_rejected(tmp_path):
    assert R.verify_identity(str(tmp_path / "nope"), RUN_ID, None, None, None)


# --------------------------------------------------------------------------- #
# 3. the CLI the launcher actually calls
# --------------------------------------------------------------------------- #
def test_cli_returns_zero_on_the_real_layout(tmp_path, capsys):
    repo, wandb_dir = tmp_path / "repo", tmp_path / "savedir"
    (wandb_dir / "wandb").mkdir(parents=True)
    _make_run(str(repo), meta={"entity": "ent", "project": "P", "name": "N"})
    rc = R.main(["--run-id", RUN_ID, "--root", str(repo), "--root", str(wandb_dir),
                 "--entity", "ent", "--project", "P", "--name", "N"])
    assert rc == 0 and "wandb run identity OK" in capsys.readouterr().out


def test_cli_returns_nonzero_when_the_run_is_absent(tmp_path, capsys):
    rc = R.main(["--run-id", RUN_ID, "--root", str(tmp_path)])
    assert rc == 1 and "WANDB IDENTITY" in capsys.readouterr().out
