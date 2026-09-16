"""Tests for exp_14's fail-closed launch verifier (plan §4 Round D, ``verify_exp14.py``).

The verifier is the last thing that runs before ~4.5 GPU-days per pair are committed, so
every check is tested twice: once on a fixture that must PASS and once on a deliberately
broken copy that must FAIL. Nothing here touches the real AcousticRooms tree -- the
dataset root, the split directory, the kit and the expected shas are all injected, and the
fixtures are built with the round-A tool itself so a fixture can never drift from the
algorithm it is supposed to certify.

The two checks that read real repository state (the committed pins and ``git rev-parse
HEAD``) run against this worktree, which is cheap and has no side effects.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

import pytest

from src.tools import make_ar_train_subsets as subsets
from src.tools.data_curve import verify

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _sha(path):
    with open(path, "rb") as fin:
        return hashlib.sha256(fin.read()).hexdigest()


def _ok(results, name):
    matching = [r for r in results if r.name == name]
    assert matching, f"no check named {name!r} in {[r.name for r in results]}"
    return all(r.ok for r in matching), matching


# ------------------------------------------------------------------ the two arm configs
CYL = {
    "model": {"conditioning": {"configs": [
        {"id": "source", "config": {"dim": 4}},
        {"id": "source_vit", "config": {"ViT": {"model": "dinoV3", "implementation": "cyl",
                                                "gauge": "cylindrical_xyz"}}},
        {"id": "context_poses_vit", "config": {"ViT": {"model": "dinoV3",
                                                       "implementation": "cyl",
                                                       "gauge": "cylindrical_xyz"}}},
    ]}},
    "training": {"lr": 5e-5, "cond_method": "fa_invariant", "frame_avg_angles": [0]},
}


def _van_from(cyl):
    van = json.loads(json.dumps(cyl))
    for entry in van["model"]["conditioning"]["configs"][1:]:
        entry["config"]["ViT"].pop("implementation", None)
        entry["config"]["ViT"].pop("gauge", None)
    van["training"].pop("cond_method", None)
    van["training"].pop("frame_avg_angles", None)
    return van


def _kit(tmp_path, cyl=None, van=None):
    """Write a kit ``configs/`` dir and return ``(kit_dir, {basename: sha})``."""
    kit = tmp_path / "kit" / "configs"
    kit.mkdir(parents=True)
    cyl = CYL if cyl is None else cyl
    van = _van_from(CYL) if van is None else van
    shas = {}
    for name, payload in (("FLAC_AR_exp14_cylS.json", cyl), ("FLAC_AR_exp14_vanS.json", van)):
        path = kit / name
        path.write_text(json.dumps(payload, indent=2))
        shas[name] = _sha(str(path))
    return str(tmp_path / "kit"), shas


def test_arm_configs_pass_when_shas_match_and_only_the_backbone_differs(tmp_path):
    kit, shas = _kit(tmp_path)
    results = verify.check_arm_configs(kit, expected_shas=shas)
    assert all(r.ok for r in results), [r.line() for r in results]
    assert any("sha" in r.name for r in results)


def test_arm_configs_fail_on_a_sha_mismatch(tmp_path):
    kit, shas = _kit(tmp_path)
    shas["FLAC_AR_exp14_cylS.json"] = "0" * 64
    results = verify.check_arm_configs(kit, expected_shas=shas)
    assert not all(r.ok for r in results)


def test_arm_configs_fail_when_van_differs_by_more_than_the_backbone(tmp_path):
    van = _van_from(CYL)
    van["training"]["lr"] = 1e-4          # a real recipe difference between the arms
    kit, shas = _kit(tmp_path, van=van)
    ok, results = _ok(verify.check_arm_configs(kit, expected_shas=shas),
                      "arm configs differ only in the geometry backbone")
    assert not ok, [r.line() for r in results]
    assert "/training/lr" in results[0].detail


def test_arm_configs_fail_when_a_cyl_only_key_is_missing(tmp_path):
    cyl = json.loads(json.dumps(CYL))
    cyl["model"]["conditioning"]["configs"][2]["config"]["ViT"].pop("gauge")
    kit, shas = _kit(tmp_path, cyl=cyl, van=_van_from(cyl))
    ok, _ = _ok(verify.check_arm_configs(kit, expected_shas=shas),
                "arm configs differ only in the geometry backbone")
    assert not ok


def test_arm_configs_fail_when_a_config_is_absent(tmp_path):
    kit, shas = _kit(tmp_path)
    os.remove(os.path.join(kit, "configs", "FLAC_AR_exp14_vanS.json"))
    assert not all(r.ok for r in verify.check_arm_configs(kit, expected_shas=shas))


def test_the_real_kit_configs_match_the_pinned_shas():
    """The shas the plan pins (8df8a811… / 733ca52b…), against the real kit."""
    kit = os.path.join(os.path.dirname(REPO_ROOT),
                       "cylindrical-dinov3", "worklog", "worklog_yixun",
                       "exp_14_data_curve_claude")
    if not os.path.isdir(os.path.join(kit, "configs")):
        pytest.skip(f"kit not available at {kit}")
    results = verify.check_arm_configs(kit)
    assert all(r.ok for r in results), [r.line() for r in results]


# --------------------------------------------------------------- the dataset configs
def test_dataset_configs_pass_on_this_worktree():
    results = verify.check_dataset_configs(REPO_ROOT)
    assert all(r.ok for r in results), [r.line() for r in results]
    assert len(results) == 3


def test_dataset_configs_fail_on_a_third_difference(tmp_path):
    root = tmp_path / "wt"
    cfg_dir = root / "src/configs/dataset_configs/AR/train"
    cfg_dir.mkdir(parents=True)
    for name in ("acousticroom_train.json", "acousticroom_train_frac025.json",
                 "acousticroom_train_frac050.json", "acousticroom_train_frac075.json"):
        shutil.copy(os.path.join(REPO_ROOT, "src/configs/dataset_configs/AR/train", name),
                    cfg_dir / name)
    tampered = json.loads((cfg_dir / "acousticroom_train_frac050.json").read_text())
    tampered["augs"] = False
    (cfg_dir / "acousticroom_train_frac050.json").write_text(json.dumps(tampered))

    results = verify.check_dataset_configs(str(root))
    assert not all(r.ok for r in results)
    assert any("augs" in r.detail for r in results if not r.ok)


# ----------------------------------------------------------------- the file-level pins
def test_pinned_files_pass_on_this_worktree():
    results = verify.check_pinned_files(REPO_ROOT)
    assert all(r.ok for r in results), [r.line() for r in results]
    assert len(results) == len(verify.PINNED_FILE_SHAS)


def test_pinned_files_fail_on_a_wrong_sha_and_on_an_absent_file(tmp_path):
    bad = dict.fromkeys(verify.PINNED_FILE_SHAS, "0" * 64)
    assert not any(r.ok for r in verify.check_pinned_files(REPO_ROOT, expected=bad))
    missing = verify.check_pinned_files(str(tmp_path), expected=verify.PINNED_FILE_SHAS)
    assert not any(r.ok for r in missing)


def test_the_vae_pin_follows_the_weights_symlink():
    """``weights/`` is a symlink; the pin is the sha of the file it resolves to."""
    assert os.path.islink(os.path.join(REPO_ROOT, "weights"))
    result = [r for r in verify.check_pinned_files(REPO_ROOT) if "VAE" in r.name]
    assert result and result[0].ok


# ------------------------------------------------------------------------ git identity
def _head(repo):
    return subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def test_git_head_passes_on_the_expected_sha():
    result = verify.check_git_head(REPO_ROOT, _head(REPO_ROOT), "FLAC worktree")
    assert result.ok, result.line()


def test_git_head_fails_on_another_sha_and_outside_a_repo(tmp_path):
    assert not verify.check_git_head(REPO_ROOT, "0" * 40, "FLAC worktree").ok
    assert not verify.check_git_head(str(tmp_path), "0" * 40, "package").ok
