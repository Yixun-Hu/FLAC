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
from pathlib import Path

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


def _tracked_dirty(repo):
    """True while tracked files are modified -- the normal state mid-round. The launch
    contract legitimately FAILs then (finding B2), so tests that run against this live
    worktree must expect that outcome rather than assume a clean tree."""
    return subprocess.run(["git", "-C", repo, "diff", "--quiet", "HEAD"],
                          capture_output=True).returncode != 0


def test_git_head_fails_on_another_sha_and_outside_a_repo(tmp_path):
    assert not verify.check_git_head(REPO_ROOT, "0" * 40, "FLAC worktree").ok
    assert not verify.check_git_head(str(tmp_path), "0" * 40, "package").ok


def _git_repo(tmp_path, name="repo"):
    """A one-commit git repository, and its HEAD."""
    repo = tmp_path / name
    repo.mkdir()
    (repo / "code.py").write_text("x = 1\n")
    env = dict(os.environ, GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null")
    for args in (["init", "-q"], ["add", "code.py"],
                 ["-c", "user.email=t@t", "-c", "user.name=T", "commit", "-qm", "c"]):
        done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                              text=True, env=env)
        assert done.returncode == 0, done.stderr
    return str(repo), _head(str(repo))


def test_git_head_fails_when_tracked_files_are_modified(tmp_path):
    """Finding B2: HEAD equality alone let dirty train.py / evaluator / package code run
    while every check passed. A modified tracked file is now a FAIL."""
    repo, head = _git_repo(tmp_path)
    assert verify.check_git_head(repo, head, "package").ok        # clean first
    open(os.path.join(repo, "code.py"), "a").write("x = 2  # uncommitted\n")
    result = verify.check_git_head(repo, head, "package")
    assert not result.ok, result.line()
    assert "modified" in result.detail and head in result.detail


def test_git_head_fails_when_a_tracked_file_is_deleted(tmp_path):
    repo, head = _git_repo(tmp_path)
    os.remove(os.path.join(repo, "code.py"))
    assert not verify.check_git_head(repo, head, "package").ok


def test_git_head_still_passes_with_untracked_files_only(tmp_path):
    """The FLAC worktree legitimately carries untracked AcousticRooms / weights symlinks."""
    repo, head = _git_repo(tmp_path)
    (Path(repo) / "AcousticRooms").mkdir()
    (Path(repo) / "notes.txt").write_text("scratch\n")
    result = verify.check_git_head(repo, head, "package")
    assert result.ok, result.line()
    assert "untracked" in result.detail or "clean" in result.detail


def test_git_head_distinguishes_a_git_error_from_a_mismatch(tmp_path):
    outside = verify.check_git_head(str(tmp_path), "0" * 40, "package")
    assert not outside.ok
    assert "not a git repository" in outside.detail
    repo, head = _git_repo(tmp_path)
    mismatch = verify.check_git_head(repo, "0" * 40, "package")
    assert not mismatch.ok and "!=" in mismatch.detail


# ------------------------------------------------------- the committed split artifacts
TOY_SPLIT = {
    "Toy": {
        "Toy_idx_0": [f"S{s:03d}_R{r:03d}_hybrid_IR.wav"
                      for s in range(4) for r in range(4)],
        "Toy_idx_1": [f"S{s:03d}_R{r:03d}_hybrid_IR.wav"
                      for s in range(3) for r in range(5)],
    },
    "Toy2": {
        "Toy2_idx_0": [f"S{s:03d}_R{r:03d}_hybrid_IR.wav"
                       for s in range(2) for r in range(6)],
    },
}


def _splits(tmp_path, split=None):
    """A committed-artifact fixture built by the round-A tool itself."""
    data_dir = tmp_path / "data" / "AR"
    data_dir.mkdir(parents=True)
    train_json = data_dir / "train.json"
    train_json.write_text(json.dumps(TOY_SPLIT if split is None else split))
    built, manifest = subsets.build_subsets(json.loads(train_json.read_text()),
                                            [0.25, 0.5, 0.75], verify.SPLIT_SEED,
                                            str(train_json))
    subsets.write_outputs(str(data_dir), built, manifest, verify.SPLIT_SEED)
    return str(data_dir)


def test_split_checksums_pass_on_a_freshly_written_set(tmp_path):
    results = verify.check_split_checksums(_splits(tmp_path))
    assert all(r.ok for r in results), [r.line() for r in results]


def test_split_checksums_fail_when_a_split_file_is_edited(tmp_path):
    data_dir = _splits(tmp_path)
    path = os.path.join(data_dir, "train_frac050_s2026.json")
    edited = json.loads(open(path).read())
    edited["Toy"]["Toy_idx_0"] = edited["Toy"]["Toy_idx_0"][:-1]
    open(path, "w").write(json.dumps(edited))
    results = verify.check_split_checksums(data_dir)
    assert not all(r.ok for r in results)
    assert any("train_frac050" in r.detail or "train_frac050" in r.name
               for r in results if not r.ok)


def test_split_checksums_fail_when_the_manifest_itself_is_edited(tmp_path):
    data_dir = _splits(tmp_path)
    path = os.path.join(data_dir, "train_frac_manifest_s2026.json")
    manifest = json.loads(open(path).read())
    manifest["fractions"]["0.25"]["final"] += 1
    open(path, "w").write(json.dumps(manifest))
    assert not all(r.ok for r in verify.check_split_checksums(data_dir))


def test_split_contents_pass_on_a_freshly_written_set(tmp_path):
    results = verify.check_split_contents(_splits(tmp_path))
    assert all(r.ok for r in results), [r.line() for r in results]
    names = {r.name for r in results}
    assert {"splits are nested", "no starved targets", "split sizes match the manifest",
            "eligible-context histograms match the manifest"} <= names


def test_split_contents_fail_on_broken_nesting(tmp_path):
    data_dir = _splits(tmp_path)
    smaller = json.loads(open(os.path.join(data_dir, "train_frac025_s2026.json")).read())
    inherited = smaller["Toy"]["Toy_idx_0"][0]
    path = os.path.join(data_dir, "train_frac050_s2026.json")
    broken = json.loads(open(path).read())
    broken["Toy"]["Toy_idx_0"] = [f for f in broken["Toy"]["Toy_idx_0"] if f != inherited]
    open(path, "w").write(json.dumps(broken))
    results = verify.check_split_contents(data_dir)
    nested_ok, nested = _ok(results, "splits are nested")
    assert not nested_ok, [r.line() for r in results]
    assert inherited in nested[0].detail
    assert not _ok(results, "split sizes match the manifest")[0]


def test_split_contents_fail_on_a_starved_target(tmp_path):
    data_dir = _splits(tmp_path)
    path = os.path.join(data_dir, "train_frac025_s2026.json")
    broken = json.loads(open(path).read())
    # keep exactly one source at one receiver -> that target has an empty context pool
    broken["Toy2"]["Toy2_idx_0"] = ["S000_R000_hybrid_IR.wav"]
    open(path, "w").write(json.dumps(broken))
    results = verify.check_split_contents(data_dir)
    ok, starved = _ok(results, "no starved targets")
    assert not ok, [r.line() for r in results]
    assert "1" in starved[0].detail


def test_split_contents_report_the_full_splits_own_eligibility(tmp_path):
    """Round E: the fractions are judged against the sampler's reachability, so the verifier
    must also state the baseline the fractions are compared against — the FULL split's dead
    targets (which must be 0) and its ``< 8``-reachable-context count."""
    data_dir = _splits(tmp_path)
    results = verify.check_split_contents(data_dir)
    ok, dead = _ok(results, "full split has no dead targets")
    assert ok, [r.line() for r in results]
    train = json.loads(open(os.path.join(data_dir, "train.json")).read())
    full = {"0": 0, "1-7": 0, ">=8": 0}
    for rooms in train.values():
        for files in rooms.values():
            for bin_name, n in subsets.context_histogram(set(files)).items():
                full[bin_name] += n
    assert full["0"] == 0
    assert str(full["1-7"]) in dead[0].detail          # the "< 8" count is reported
    assert subsets.ELIGIBILITY_RULE in dead[0].detail  # ... and the rule it was judged under


def test_split_contents_fail_when_the_full_split_has_a_dead_target(tmp_path):
    """A target the pinned sampler can never reach a context for is unrepairable by any
    subset, so it invalidates the whole construction rather than one fraction. The extra
    entry sits at its own receiver, so every committed split stays a subset of train.json
    and only this check fires."""
    data_dir = _splits(tmp_path)
    path = os.path.join(data_dir, "train.json")
    train = json.loads(open(path).read())
    train["Toy2"]["Toy2_idx_0"].append("S010_R099_hybrid_IR.wav")   # rebuilds to S0010_* : dead
    open(path, "w").write(json.dumps(train))
    results = verify.check_split_contents(data_dir)
    ok, dead = _ok(results, "full split has no dead targets")
    assert not ok, [r.line() for r in results]
    assert "S010_R099_hybrid_IR.wav" in dead[0].detail
    assert _ok(results, "splits are subsets of train.json")[0]      # nothing else disturbed
    assert _ok(results, "no starved targets")[0]


def test_verifier_and_generator_share_one_eligibility_definition():
    """No second implementation (brief, round E): the verifier's reachability must be the
    generator's function object, not a copy that can drift from it."""
    import src.tools.data_curve.verify as verify_module

    source = open(verify_module.__file__, encoding="utf-8").read()
    assert "sampler_candidates" in source
    assert "S00{" not in source and verify.__dict__.get("SAMPLER_TAIL") is None


def test_split_contents_fail_when_a_split_leaves_train_json(tmp_path):
    data_dir = _splits(tmp_path)
    path = os.path.join(data_dir, "train_frac075_s2026.json")
    broken = json.loads(open(path).read())
    broken["Toy"]["Toy_idx_0"].append("S099_R099_hybrid_IR.wav")
    open(path, "w").write(json.dumps(broken))
    ok, _ = _ok(verify.check_split_contents(data_dir), "splits are subsets of train.json")
    assert not ok


# ------------------------------------------------------------- the anchor audit (D3)
def _dataset_root(tmp_path, listed, on_disk):
    root = tmp_path / "AR"
    for scene, rooms in on_disk.items():
        for room, files in rooms.items():
            room_dir = root / "single_channel_ir_1" / scene / room
            room_dir.mkdir(parents=True)
            for fname in files:
                (room_dir / fname).write_bytes(b"")
    train_json = tmp_path / "train_listed.json"
    train_json.write_text(json.dumps(listed))
    return str(root), str(train_json)


def test_anchor_audit_passes_when_extra_files_sit_at_unlisted_receivers(tmp_path):
    listed = {"Toy": {"Toy_idx_0": ["S000_R000_hybrid_IR.wav", "S001_R000_hybrid_IR.wav"]}}
    on_disk = {"Toy": {"Toy_idx_0": ["S000_R000_hybrid_IR.wav", "S001_R000_hybrid_IR.wav",
                                     "S000_R777_hybrid_IR.wav", "S001_R777_hybrid_IR.wav"]}}
    root, train_json = _dataset_root(tmp_path, listed, on_disk)
    result = verify.check_anchor_audit(root, train_json)
    assert result.ok, result.line()
    assert "2" in result.detail            # two extra files, zero violations


def test_anchor_audit_fails_on_an_extra_file_at_a_training_receiver(tmp_path):
    listed = {"Toy": {"Toy_idx_0": ["S000_R000_hybrid_IR.wav", "S001_R000_hybrid_IR.wav"]}}
    on_disk = {"Toy": {"Toy_idx_0": ["S000_R000_hybrid_IR.wav", "S001_R000_hybrid_IR.wav",
                                     "S002_R000_hybrid_IR.wav"]}}
    root, train_json = _dataset_root(tmp_path, listed, on_disk)
    result = verify.check_anchor_audit(root, train_json)
    assert not result.ok
    assert "S002_R000_hybrid_IR.wav" in result.detail


def test_anchor_audit_fails_when_a_room_is_missing_on_disk(tmp_path):
    listed = {"Toy": {"Toy_idx_0": ["S000_R000_hybrid_IR.wav"]},
              "Gone": {"Gone_idx_0": ["S000_R000_hybrid_IR.wav"]}}
    on_disk = {"Toy": {"Toy_idx_0": ["S000_R000_hybrid_IR.wav"]}}
    root, train_json = _dataset_root(tmp_path, listed, on_disk)
    assert not verify.check_anchor_audit(root, train_json).ok


# ------------------------------------------------------------------ report and CLI
def test_run_all_reports_every_check_and_summarises(tmp_path, capsys):
    kit, shas = _kit(tmp_path)
    results = verify.run_all(
        flac_wt=REPO_ROOT, kit_dir=kit, pkg_dir=REPO_ROOT,
        expect_flac_sha=_head(REPO_ROOT), expect_pkg_sha="0" * 40,
        data_dir=_splits(tmp_path), dataset_root=None, expected_arm_shas=shas,
    )
    failed = [r for r in results if not r.ok]
    expected_failures = ["package HEAD"]
    if _tracked_dirty(REPO_ROOT):   # mid-round: the FLAC identity check must fail too
        expected_failures = ["FLAC worktree HEAD", "package HEAD"]
    assert sorted(r.name for r in failed) == sorted(expected_failures), \
        [r.line() for r in results]
    assert verify.report(results) == verify.EXIT_FAILED
    out = capsys.readouterr().out
    assert out.count("PASS ") == len(results) - len(expected_failures)
    assert "FAIL package HEAD" in out


def test_run_all_skips_the_anchor_audit_only_when_no_root_is_given(tmp_path):
    kit, shas = _kit(tmp_path)
    common = dict(flac_wt=REPO_ROOT, kit_dir=kit, pkg_dir=REPO_ROOT,
                  expect_flac_sha=_head(REPO_ROOT), expect_pkg_sha=_head(REPO_ROOT),
                  data_dir=_splits(tmp_path), expected_arm_shas=shas)
    without = verify.run_all(dataset_root=None, **common)
    tolerated = {"FLAC worktree HEAD", "package HEAD"} if _tracked_dirty(REPO_ROOT) else set()
    assert all(r.ok for r in without if r.name not in tolerated), [r.line() for r in without]
    assert not any("anchor" in r.name for r in without)

    root, train_json = _dataset_root(
        tmp_path, {"Toy": {"Toy_idx_0": ["S000_R000_hybrid_IR.wav"]}},
        {"Toy": {"Toy_idx_0": ["S000_R000_hybrid_IR.wav", "S001_R000_hybrid_IR.wav"]}})
    with_root = verify.run_all(dataset_root=root, train_json=train_json, **common)
    anchor = [r for r in with_root if "anchor" in r.name]
    assert len(anchor) == 1 and not anchor[0].ok      # S001 sits at a listed receiver


def _toy_dataset_root(tmp_path):
    """The RIR tree TOY_SPLIT describes, so the CLI's anchor audit has something to walk."""
    root = tmp_path / "toy_AR"
    for scene, rooms in TOY_SPLIT.items():
        for room, files in rooms.items():
            room_dir = root / "single_channel_ir_1" / scene / room
            room_dir.mkdir(parents=True, exist_ok=True)
            for fname in files:
                (room_dir / fname).write_bytes(b"")
    return str(root)


def test_cli_exits_3_when_a_check_fails(tmp_path):
    kit, _ = _kit(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-m", "src.tools.data_curve.verify",
         "--expect-flac-sha", "0" * 40, "--expect-pkg-sha", "0" * 40,
         "--pkg-dir", REPO_ROOT, "--kit-dir", kit, "--flac-wt", REPO_ROOT,
         "--data-dir", _splits(tmp_path), "--dataset-root", _toy_dataset_root(tmp_path)],
        cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "FAIL FLAC worktree HEAD" in proc.stdout
    assert "PASS receiver-level anchor audit" in proc.stdout   # never skippable (N7)


def test_cli_has_no_flag_that_skips_the_anchor_audit(tmp_path):
    """Finding N7: a production run must not be able to report success without the audit."""
    kit, _ = _kit(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-m", "src.tools.data_curve.verify",
         "--expect-flac-sha", "0" * 40, "--expect-pkg-sha", "0" * 40,
         "--pkg-dir", REPO_ROOT, "--kit-dir", kit, "--skip-anchor-audit"],
        cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 2                     # argparse: unrecognized argument
    assert "--skip-anchor-audit" in proc.stderr


def test_cli_exits_0_on_the_real_launch_contract():
    """The full contract, as the launcher runs it, against this worktree and the kit.

    The RIR tree is reached only through the injected ``--dataset-root``, and the verifier
    never writes: this is a read-only integration check of the real launch gate."""
    kit = os.path.join(os.path.dirname(REPO_ROOT), "cylindrical-dinov3", "worklog",
                       "worklog_yixun", "exp_14_data_curve_claude")
    dataset_root = os.path.join(REPO_ROOT, "AcousticRooms")
    if not os.path.isdir(os.path.join(kit, "configs")):
        pytest.skip("kit not available")
    if not os.path.isdir(os.path.join(dataset_root, "single_channel_ir_1")):
        pytest.skip("AcousticRooms not available")
    if _tracked_dirty(REPO_ROOT):
        pytest.skip("tracked files are modified: the launch contract cannot pass (B2) "
                    "until the round's commits have landed")
    manifest_path = os.path.join(REPO_ROOT, "data", "AR",
                                 f"train_frac_manifest_s{verify.SPLIT_SEED}.json")
    with open(manifest_path) as fin:
        eligibility = json.load(fin).get("eligibility")
    if eligibility != subsets.ELIGIBILITY_RULE:
        pytest.skip(
            f"the committed splits were built under eligibility {eligibility!r}, not "
            f"{subsets.ELIGIBILITY_RULE!r}: round E redefined an eligible context as one the "
            "PINNED sampler can actually reconstruct, and under that rule these files have "
            "229 / 25 / 0 starved targets. The Planner regenerates and commits them after "
            "review; this check un-skips itself the moment the manifest names the new rule."
        )
    head = _head(REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, "-m", "src.tools.data_curve.verify",
         "--expect-flac-sha", head, "--expect-pkg-sha", head,
         "--pkg-dir", REPO_ROOT, "--kit-dir", kit, "--dataset-root", dataset_root],
        cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "FAIL" not in proc.stdout
