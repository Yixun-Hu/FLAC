"""Tests for exp_14's run-name / argv helper (plan §4 "Round D", naming rule r3-3).

Every string that identifies an exp_14 data-curve cell is generated here, never typed
into a shell: the run ids, the ``--eval-name``s, the two argv arrays of the frozen recipe
(plan §2) and the metric/prediction paths ``eval_FLAC`` will write for them.

Two hazards these tests exist for:

* **The naming rule (finding r3-3).** FLAC's ``gen_model_comparison.py::is_exp14_row``
  claims *any* row whose glob contains ``exp14_`` for the exp_14 **yaw** campaign, which
  would label this experiment's rows with the wrong protocol and run the wrong validator.
  So no run id, eval name or metric basename may contain that substring.
* **The eval protocol (announcement 05, CLAUDE.md "Eval-protocol flags").** A cyl
  checkpoint scored with the vanilla path (or vice versa) produces plausible but
  catastrophically wrong numbers. The arm ⇄ ``--cond-method`` and K ⇄ dataset-config
  mappings are therefore pinned token by token.

CPU-only and filesystem-free except for the ``check-bundle`` cases, which write small
tensors into ``tmp_path``.
"""
import hashlib
import os
import subprocess
import sys

import pytest

from src.tools.data_curve import names

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ALL_CELLS = [
    (arm, tag, k, seed)
    for arm in ("cyl", "van")
    for tag in ("025", "050", "075")
    for k in (1, 8)
    for seed in (42, 43, 44, 45, 46)
]


# ----------------------------------------------------------------- the fraction table
def test_fractions_table_is_the_three_planned_fractions():
    assert names.FRACTIONS == {"025": 0.25, "050": 0.5, "075": 0.75}


def test_arms_and_grid_constants():
    assert names.ARMS == ("cyl", "van")
    assert names.SEEDS == (42, 43, 44, 45, 46)
    assert names.K_VALUES == (1, 8)


# ------------------------------------------------------------------ ids and eval names
@pytest.mark.parametrize("arm,tag,expected", [
    ("cyl", "025", "dc_cyl_f025"),
    ("van", "025", "dc_van_f025"),
    ("cyl", "050", "dc_cyl_f050"),
    ("van", "075", "dc_van_f075"),
])
def test_run_id_format(arm, tag, expected):
    assert names.run_id(arm, tag) == expected


@pytest.mark.parametrize("arm,tag,k,seed,expected", [
    ("cyl", "025", 8, 42, "dc_cyl_f025_K8_s42"),
    ("van", "050", 1, 46, "dc_van_f050_K1_s46"),
])
def test_eval_name_format(arm, tag, k, seed, expected):
    assert names.eval_name(arm, tag, k, seed) == expected


@pytest.mark.parametrize("bad", ["cylindrical", "CYL", "", "dc", None, 8])
def test_run_id_rejects_an_unknown_arm(bad):
    with pytest.raises(ValueError):
        names.run_id(bad, "025")


@pytest.mark.parametrize("bad", ["25", "0.25", "100", "", None, 25])
def test_run_id_rejects_an_unknown_fraction_tag(bad):
    with pytest.raises(ValueError):
        names.run_id("cyl", bad)


@pytest.mark.parametrize("bad", [0, 4, 9, "8", None])
def test_eval_name_rejects_an_unplanned_K(bad):
    with pytest.raises(ValueError):
        names.eval_name("cyl", "025", bad, 42)


@pytest.mark.parametrize("bad", [41, 47, "42", None, True])
def test_eval_name_rejects_an_unplanned_seed(bad):
    with pytest.raises(ValueError):
        names.eval_name("cyl", "025", 8, bad)


# ------------------------------------------------------- the forbidden-substring rule
def test_assert_no_forbidden_substring_returns_a_clean_string():
    assert names.assert_no_forbidden_substring("dc_cyl_f025") == "dc_cyl_f025"


@pytest.mark.parametrize("bad", [
    "exp14_cyl_f025",
    "outputs_FLAC/exp14_data_curve/x.json",
    "dc_cyl_f025_exp14_K8",
])
def test_assert_no_forbidden_substring_raises(bad):
    with pytest.raises(ValueError) as err:
        names.assert_no_forbidden_substring(bad)
    assert "exp14_" in str(err.value)


def test_no_generated_name_carries_the_forbidden_substring():
    """Run ids, eval names and metric/prediction basenames of every planned cell."""
    ckpt = "/media/diskstation/yixunhu/FLAC/checkpoints/exp14_data_curve/dc_cyl_f025/" \
           "epoch=8-step=40000.ckpt"
    for arm, tag, k, seed in ALL_CELLS:
        assert names.FORBIDDEN_SUBSTRING not in names.run_id(arm, tag)
        assert names.FORBIDDEN_SUBSTRING not in names.eval_name(arm, tag, k, seed)
        for path in (names.metrics_json_path(ckpt, arm, tag, k, seed),
                     names.predictions_pt_path(ckpt, arm, tag, k, seed)):
            # the NAS *directory* legitimately carries it (announcement 07); the basename,
            # which is what a model_comparison row globs, must not.
            assert names.FORBIDDEN_SUBSTRING not in os.path.basename(path)


# -------------------------------------------------------- output paths == eval_FLAC's
def test_metrics_and_prediction_paths_mirror_eval_FLAC_build_output_paths():
    """The launcher's skip logic must look exactly where ``eval_FLAC`` writes."""
    import eval_FLAC  # heavy, side-effect-free at import (see test_eval_paths.py)

    ckpt = "/nas/dc_cyl_f025/epoch=8-step=40000.ckpt"
    for arm, tag, k, seed in ALL_CELLS:
        expected = eval_FLAC.build_output_paths(
            ckpt, 1, 1.0, names.eval_name(arm, tag, k, seed),
            cond_method=names.ARM_COND_METHOD[arm], rotate_deg=0.0, n_angles=1,
        )
        assert names.metrics_json_path(ckpt, arm, tag, k, seed) == expected["metrics"]
        assert names.predictions_pt_path(ckpt, arm, tag, k, seed) == expected["predictions"]


def test_metrics_basename_carries_the_arm_suffix():
    ckpt = "/nas/dc_cyl_f025/epoch=8-step=40000.ckpt"
    cyl = os.path.basename(names.metrics_json_path(ckpt, "cyl", "025", 8, 42))
    van = os.path.basename(names.metrics_json_path(ckpt, "van", "025", 8, 42))
    assert cyl == "epoch=8-step=40000_metrics_1_1.0_dc_cyl_f025_K8_s42_fa_invariant_a1.json"
    assert van == "epoch=8-step=40000_metrics_1_1.0_dc_van_f025_K8_s42.json"


# ================================================================== the training argv
KIT_CYL = "/kit/configs/FLAC_AR_exp14_cylS.json"
KIT_VAN = "/kit/configs/FLAC_AR_exp14_vanS.json"
NAS = "/media/diskstation/yixunhu/FLAC/checkpoints/exp14_data_curve"


def test_train_argv_is_the_frozen_exp13_recipe_token_by_token():
    argv = names.train_argv(
        "cyl", "025", KIT_CYL,
        "src/configs/dataset_configs/AR/train/acousticroom_train_frac025.json",
        f"{NAS}/dc_cyl_f025", f"{NAS}/dc_cyl_f025/run_contract.json",
    )
    assert argv == [
        "python", "train.py",
        "--model-config", KIT_CYL,
        "--dataset-config",
        "src/configs/dataset_configs/AR/train/acousticroom_train_frac025.json",
        "--pretransform-ckpt-path", "weights/FLAC/VAE.safetensors",
        "--max-steps", "40000",
        "--batch-size", "32",
        "--accum-batches", "1",
        "--num-workers", "6",
        "--seed", "42",
        "--num-gpus", "2",
        "--strategy", "ddp_find_unused_parameters_true",
        "--sync-batchnorm", "true",
        "--logger", "none",
        "--checkpoint-every", "2500",
        "--name", "dc_cyl_f025",
        "--experiment-name", "dc_cyl_f025",
        "--save-dir", f"{NAS}/dc_cyl_f025",
        "--run-contract-json", f"{NAS}/dc_cyl_f025/run_contract.json",
    ]


def test_train_argv_appends_ckpt_path_only_when_resuming():
    base = names.train_argv(
        "van", "050", KIT_VAN, names.TRAIN_DATASET_CONFIGS["050"],
        f"{NAS}/dc_van_f050", f"{NAS}/dc_van_f050/run_contract.json",
    )
    assert "--ckpt-path" not in base
    resumed = names.train_argv(
        "van", "050", KIT_VAN, names.TRAIN_DATASET_CONFIGS["050"],
        f"{NAS}/dc_van_f050", f"{NAS}/dc_van_f050/run_contract.json",
        ckpt_path=f"{NAS}/dc_van_f050/epoch=1-step=25000.ckpt",
    )
    assert resumed == base + ["--ckpt-path", f"{NAS}/dc_van_f050/epoch=1-step=25000.ckpt"]


def test_train_argv_rejects_a_dataset_config_of_another_fraction():
    """The one substitution the run contract exists to prevent (plan §4 Round D)."""
    with pytest.raises(ValueError) as err:
        names.train_argv(
            "cyl", "025", KIT_CYL, names.TRAIN_DATASET_CONFIGS["050"],
            f"{NAS}/dc_cyl_f025", f"{NAS}/dc_cyl_f025/run_contract.json",
        )
    assert "acousticroom_train_frac025.json" in str(err.value)


def test_train_argv_rejects_the_other_arms_model_config():
    with pytest.raises(ValueError):
        names.train_argv(
            "cyl", "025", KIT_VAN, names.TRAIN_DATASET_CONFIGS["025"],
            f"{NAS}/dc_cyl_f025", f"{NAS}/dc_cyl_f025/run_contract.json",
        )


@pytest.mark.parametrize("save_dir,contract", [
    (f"{NAS}/dc_van_f025", f"{NAS}/dc_cyl_f025/run_contract.json"),   # another run's contract
    (f"{NAS}/dc_cyl_f050", f"{NAS}/dc_cyl_f050/run_contract.json"),   # save dir != run id
    (f"{NAS}/dc_cyl_f025", f"{NAS}/dc_cyl_f025/contract.json"),       # not the sidecar name
])
def test_train_argv_rejects_a_mismatched_run_dir_or_contract(save_dir, contract):
    with pytest.raises(ValueError):
        names.train_argv("cyl", "025", KIT_CYL, names.TRAIN_DATASET_CONFIGS["025"],
                         save_dir, contract)


# ==================================================================== the eval argv
CKPT = f"{NAS}/dc_cyl_f025/epoch=8-step=40000.ckpt"


def test_eval_argv_cyl_K8_is_the_frozen_protocol():
    assert names.eval_argv("cyl", "025", 8, 42, KIT_CYL, CKPT) == [
        "python", "eval_FLAC.py",
        "--model-config", KIT_CYL,
        "--dataset-config", "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json",
        "--ckpt-path", CKPT,
        "--cond-method", "fa_invariant",
        "--frame-avg-angles", "0",
        "--rotate-deg", "0",
        "--cond-autocast", "bf16",
        "--seed", "42",
        "--steps", "1",
        "--cfg-scale", "1.0",
        "--eval-name", "dc_cyl_f025_K8_s42",
        "--store_predictions",
    ]


def test_eval_argv_van_K1_is_the_frozen_protocol():
    ckpt = f"{NAS}/dc_van_f075/epoch=8-step=40000.ckpt"
    assert names.eval_argv("van", "075", 1, 46, KIT_VAN, ckpt) == [
        "python", "eval_FLAC.py",
        "--model-config", KIT_VAN,
        "--dataset-config", "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json",
        "--ckpt-path", ckpt,
        "--cond-method", "vanilla",
        "--frame-avg-angles", "0",
        "--rotate-deg", "0",
        "--cond-autocast", "bf16",
        "--seed", "46",
        "--steps", "1",
        "--cfg-scale", "1.0",
        "--eval-name", "dc_van_f075_K1_s46",
        "--store_predictions",
    ]


def test_eval_argv_maps_K_to_the_full_unseen_config_and_arm_to_cond_method():
    for arm, tag, k, seed in ALL_CELLS:
        cfg = KIT_CYL if arm == "cyl" else KIT_VAN
        argv = names.eval_argv(arm, tag, k, seed, cfg,
                               f"{NAS}/{names.run_id(arm, tag)}/epoch=8-step=40000.ckpt")
        ds = argv[argv.index("--dataset-config") + 1]
        assert ds == names.EVAL_DATASET_CONFIGS[k]
        assert "unseeneval_1.json" in ds if k == 1 else ds.endswith("unseeneval.json")
        assert argv[argv.index("--cond-method") + 1] == names.ARM_COND_METHOD[arm]
        # announcement 05: all four conditioning flags are explicit for BOTH arms
        for flag in ("--cond-method", "--frame-avg-angles", "--rotate-deg", "--cond-autocast"):
            assert argv.count(flag) == 1


def test_eval_argv_rejects_a_checkpoint_from_another_run():
    with pytest.raises(ValueError):
        names.eval_argv("cyl", "025", 8, 42, KIT_CYL,
                        f"{NAS}/dc_van_f025/epoch=8-step=40000.ckpt")


# ============================================================ the check-bundle contract
def _write_bundle(tmp_path, *, n=4, arm="cyl", k=8, seed=42, tag="025",
                  predictions=None, **meta_overrides):
    """A prediction bundle in exactly the shape ``eval_FLAC --store_predictions`` writes."""
    import torch

    import eval_FLAC

    meta = eval_FLAC.build_predictions_meta(
        names.EVAL_DATASET_CONFIGS[k], seed, n, names.ARM_COND_METHOD[arm],
        [0.0] if arm == "cyl" else None, 0.0, 64, "bf16",
        ckpt_path=f"{NAS}/{names.run_id(arm, tag)}/epoch=8-step=40000.ckpt",
        eval_name=names.eval_name(arm, tag, k, seed), steps=1, cfg_scale=1.0,
    )
    meta.update(meta_overrides)
    if predictions is None:
        predictions = torch.zeros(n, 1, names.SAMPLE_LEN)
    path = tmp_path / "bundle.pt"
    torch.save({"predictions": predictions, "meta": meta}, path)
    return str(path)


def _real_ckpt(tmp_path, arm="cyl", tag="025", content=b"checkpoint bytes"):
    """A stand-in for the validated final checkpoint, and its digest."""
    import hashlib as _hashlib

    run_dir = tmp_path / "nas" / names.run_id(arm, tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "epoch=8-step=40000.ckpt"
    path.write_bytes(content)
    return str(path), _hashlib.sha256(content).hexdigest()


def test_check_bundle_accepts_a_matching_bundle(tmp_path):
    path = _write_bundle(tmp_path)
    assert names.check_bundle(path, 4, 42, 8, "cyl",
                              expect_eval_name="dc_cyl_f025_K8_s42") == []


# ------------------------------------- the artifact ⇄ checkpoint binding (codex round F)
def test_the_artifact_contract_constant_mirrors_the_evaluator(tmp_path):
    """``names`` must stay importable without torch, so the contract string is mirrored --
    and a mirror that drifts would silently stop rejecting foreign artifacts."""
    import eval_FLAC

    assert names.PREDICTIONS_ARTIFACT_CONTRACT == eval_FLAC.PREDICTIONS_ARTIFACT_CONTRACT


def test_check_bundle_binds_the_artifact_to_the_validated_checkpoint(tmp_path):
    ckpt, digest = _real_ckpt(tmp_path)
    path = _write_bundle(tmp_path, ckpt_path=ckpt)
    assert names.check_bundle(path, 4, 42, 8, "cyl", expect_ckpt=ckpt,
                              expect_ckpt_sha256=digest) == []
    # an unnormalised spelling of the same file is the same file
    odd = os.path.join(os.path.dirname(ckpt), ".", "", os.path.basename(ckpt))
    assert names.check_bundle(path, 4, 42, 8, "cyl", expect_ckpt=odd) == []


def test_check_bundle_rejects_a_bundle_from_a_foreign_checkpoint(tmp_path):
    """Codex round-F finding 2: a stale bundle from another checkpoint parses, carries the
    right protocol, and used to be counted (and skipped) as this cell's result."""
    ckpt, digest = _real_ckpt(tmp_path)
    foreign, _ = _real_ckpt(tmp_path, arm="van", content=b"another run")
    path = _write_bundle(tmp_path, ckpt_path=foreign)
    violations = names.check_bundle(path, 4, 42, 8, "cyl", expect_ckpt=ckpt,
                                    expect_ckpt_sha256=digest)
    assert any("ckpt_path" in v for v in violations), violations
    assert any(ckpt in v for v in violations), violations


def test_check_bundle_rejects_a_checkpoint_replaced_after_validation(tmp_path):
    ckpt, digest = _real_ckpt(tmp_path)
    path = _write_bundle(tmp_path, ckpt_path=ckpt)
    open(ckpt, "wb").write(b"a rerun overwrote it")
    violations = names.check_bundle(path, 4, 42, 8, "cyl", expect_ckpt=ckpt,
                                    expect_ckpt_sha256=digest)
    assert any("replaced after validation" in v for v in violations), violations

    os.remove(ckpt)
    gone = names.check_bundle(path, 4, 42, 8, "cyl", expect_ckpt=ckpt,
                              expect_ckpt_sha256=digest)
    assert any("cannot be hashed" in v for v in gone), gone


@pytest.mark.parametrize("override,needle", [
    ({"steps": 4}, "steps"),
    ({"cfg_scale": 3.0}, "cfg_scale"),
    ({"artifact_contract": "raw decoder output"}, "artifact_contract"),
])
def test_check_bundle_rejects_a_foreign_sampling_protocol(tmp_path, override, needle):
    path = _write_bundle(tmp_path, **override)
    violations = names.check_bundle(path, 4, 42, 8, "cyl")
    assert any(needle in v for v in violations), violations


def test_check_bundle_accepts_the_van_protocol(tmp_path):
    path = _write_bundle(tmp_path, arm="van", k=1, seed=46, tag="075")
    assert names.check_bundle(path, 4, 46, 1, "van") == []


@pytest.mark.parametrize("kwargs,args,needle", [
    ({}, (5, 42, 8, "cyl"), "n_items"),                              # wrong item count
    ({}, (4, 43, 8, "cyl"), "seed"),                                 # wrong seed
    ({}, (4, 42, 1, "cyl"), "dataset_config"),                       # K1 asked, K8 bundle
    ({}, (4, 42, 8, "van"), "cond_method"),                          # other arm's protocol
    ({"cond_autocast": "default"}, (4, 42, 8, "cyl"), "cond_autocast"),
    ({"rotate_deg": 45.0}, (4, 42, 8, "cyl"), "rotate_deg"),
    ({"frame_avg_angles": [0.0, 90.0, 180.0, 270.0]}, (4, 42, 8, "cyl"), "frame_avg_angles"),
    ({"stored_after_clamp_pad": False}, (4, 42, 8, "cyl"), "stored_after_clamp_pad"),
])
def test_check_bundle_rejects_a_protocol_mismatch(tmp_path, kwargs, args, needle):
    path = _write_bundle(tmp_path, **kwargs)
    violations = names.check_bundle(path, *args)
    assert violations, "expected a violation"
    assert any(needle in v for v in violations), violations


def test_check_bundle_accepts_an_unnormalised_but_identical_config_path(tmp_path):
    path = _write_bundle(
        tmp_path,
        dataset_config="./src/configs/dataset_configs/AR/eval/./acousticroom_unseeneval.json")
    assert names.check_bundle(path, 4, 42, 8, "cyl") == []


def test_check_bundle_rejects_a_same_named_config_from_another_directory(tmp_path):
    """codex M4: a basename match let a custom/reduced config with 6,337 different entries
    pass as the full unseen eval (announcement 01). The whole relative path is pinned."""
    path = _write_bundle(tmp_path,
                         dataset_config="/tmp/mine/acousticroom_unseeneval.json")
    violations = names.check_bundle(path, 4, 42, 8, "cyl")
    assert any("dataset_config" in v for v in violations), violations
    assert any(names.EVAL_DATASET_CONFIGS[8] in v for v in violations), violations


def test_eval_dataset_config_sha256_hashes_the_committed_config():
    for k, rel in names.EVAL_DATASET_CONFIGS.items():
        expected = hashlib.sha256(open(os.path.join(REPO_ROOT, rel), "rb").read()).hexdigest()
        assert names.eval_dataset_config_sha256(k) == expected


def test_check_bundle_fails_when_the_expected_config_cannot_be_hashed(tmp_path):
    """The sha is the cell's full-split provenance until eval_FLAC embeds one, so a config
    root without it cannot certify anything."""
    path = _write_bundle(tmp_path)
    violations = names.check_bundle(path, 4, 42, 8, "cyl", config_root=str(tmp_path))
    assert any("sha256" in v or "readable" in v for v in violations), violations


def test_check_bundle_rejects_a_wrong_eval_name(tmp_path):
    path = _write_bundle(tmp_path)
    violations = names.check_bundle(path, 4, 42, 8, "cyl",
                                    expect_eval_name="dc_cyl_f050_K8_s42")
    assert any("eval_name" in v for v in violations), violations


def test_check_bundle_rejects_a_wrong_shape_and_non_finite_values(tmp_path):
    import torch

    short = _write_bundle(tmp_path, predictions=torch.zeros(4, 1, 8000))
    assert any("shape" in v for v in names.check_bundle(short, 4, 42, 8, "cyl"))

    nans = torch.zeros(4, 1, names.SAMPLE_LEN)
    nans[2, 0, 7] = float("nan")
    path = _write_bundle(tmp_path, predictions=nans)
    assert any("finite" in v for v in names.check_bundle(path, 4, 42, 8, "cyl"))


def test_check_bundle_rejects_an_unreadable_bundle(tmp_path):
    broken = tmp_path / "truncated.pt"
    broken.write_bytes(b"not a torch file")
    assert any("load" in v for v in names.check_bundle(str(broken), 4, 42, 8, "cyl"))
    assert any("load" in v for v in names.check_bundle(str(tmp_path / "absent.pt"),
                                                       4, 42, 8, "cyl"))


# ================================================= the check-metrics contract (round F)
def _write_metrics(tmp_path, ckpt, *, arm="cyl", name="metrics.json", **overrides):
    """A metrics JSON in exactly the shape ``eval_FLAC`` writes for one cell."""
    import json as _json

    import eval_FLAC

    method = names.ARM_COND_METHOD[arm]
    record = eval_FLAC.build_metrics_record(
        {"T60": 0.1, "C50": 1.2}, ckpt, 0.0, method,
        [0.0] if method == "fa_invariant" else None, "bf16")
    record.update(overrides)
    path = tmp_path / name
    path.write_text(_json.dumps(record, indent=4))
    return str(path)


def _metrics_args(ckpt, arm="cyl"):
    return dict(expect_ckpt=ckpt, expect_cond_method=names.ARM_COND_METHOD[arm],
                expect_angles=names.FRAME_AVG_ANGLES, expect_rotate=names.ROTATE_DEG,
                expect_autocast=names.COND_AUTOCAST)


def test_check_metrics_accepts_both_arms_own_records(tmp_path):
    ckpt, _ = _real_ckpt(tmp_path)
    for arm in names.ARMS:
        path = _write_metrics(tmp_path, ckpt, arm=arm, name=f"{arm}.json")
        assert names.check_metrics(path, **_metrics_args(ckpt, arm)) == []


def test_check_metrics_rejects_a_record_from_a_foreign_checkpoint(tmp_path):
    ckpt, _ = _real_ckpt(tmp_path)
    foreign, _ = _real_ckpt(tmp_path, arm="van", content=b"another run")
    path = _write_metrics(tmp_path, foreign)
    violations = names.check_metrics(path, **_metrics_args(ckpt))
    assert any("ckpt_path" in v for v in violations), violations


@pytest.mark.parametrize("override,needle", [
    ({"metrics": {}}, "metrics"),
    ({"metrics": None}, "metrics"),
    ({"cond_method": "vanilla"}, "cond_method"),
    ({"frame_avg_angles": [0.0, 90.0, 180.0, 270.0]}, "frame_avg_angles"),
    ({"rotate_deg": 45.0}, "rotate_deg"),
    ({"cond_autocast": "default"}, "cond_autocast"),
])
def test_check_metrics_rejects_a_protocol_mismatch(tmp_path, override, needle):
    ckpt, _ = _real_ckpt(tmp_path)
    path = _write_metrics(tmp_path, ckpt, **override)
    violations = names.check_metrics(path, **_metrics_args(ckpt))
    assert any(needle in v for v in violations), violations


def test_check_metrics_treats_an_unreadable_record_as_not_done(tmp_path):
    ckpt, _ = _real_ckpt(tmp_path)
    half = tmp_path / "half.json"
    half.write_text('{"metrics": {"T60": 1.0}, "ckpt_pa')
    assert names.check_metrics(str(half), **_metrics_args(ckpt))
    assert names.check_metrics(str(tmp_path / "absent.json"), **_metrics_args(ckpt))


def test_check_metrics_rejects_an_unplanned_cond_method(tmp_path):
    ckpt, _ = _real_ckpt(tmp_path)
    path = _write_metrics(tmp_path, ckpt)
    with pytest.raises(ValueError):
        names.check_metrics(path, ckpt, "frame_avg", "0", 0.0, "bf16")


def _cli(*args):
    return subprocess.run([sys.executable, "-m", "src.tools.data_curve.names", *args],
                          cwd=REPO_ROOT, capture_output=True, text=True)


def test_check_bundle_cli_exit_codes(tmp_path):
    ckpt, digest = _real_ckpt(tmp_path)
    good = _write_bundle(tmp_path, ckpt_path=ckpt)
    bind = ["--expect-ckpt", ckpt, "--expect-ckpt-sha256", digest]
    ok = _cli("check-bundle", "--pt", good, "--expect-n", "4", "--expect-seed", "42",
              "--expect-K", "8", "--expect-arm", "cyl",
              "--expect-eval-name", "dc_cyl_f025_K8_s42", *bind)
    assert ok.returncode == 0, ok.stderr
    assert "PASS" in ok.stdout
    # the launcher records these tokens as the cell's provenance (codex M4, round-F 2)
    sha = hashlib.sha256(open(os.path.join(REPO_ROOT, names.EVAL_DATASET_CONFIGS[8]),
                              "rb").read()).hexdigest()
    assert f"dataset_config_sha256={sha}" in ok.stdout
    assert f"ckpt_sha256={digest}" in ok.stdout

    bad = _cli("check-bundle", "--pt", good, "--expect-n", "4", "--expect-seed", "42",
               "--expect-K", "8", "--expect-arm", "van", *bind)
    assert bad.returncode == 3, bad.stderr        # the artifact disagrees
    assert "FAIL" in bad.stdout or "FAIL" in bad.stderr

    argerr = _cli("check-bundle", "--pt", good, "--expect-n", "4", "--expect-seed", "7",
                  "--expect-K", "8", "--expect-arm", "cyl", *bind)
    assert argerr.returncode == 2, argerr.stderr  # unplanned seed: a launcher bug


def test_check_bundle_cli_refuses_an_unbound_artifact(tmp_path):
    """A launcher that forgets the binding must not get a verdict at all (round-F 2)."""
    good = _write_bundle(tmp_path)
    unbound = _cli("check-bundle", "--pt", good, "--expect-n", "4", "--expect-seed", "42",
                   "--expect-K", "8", "--expect-arm", "cyl")
    assert unbound.returncode == 2, unbound.stdout
    assert "--expect-ckpt" in unbound.stderr


def test_check_bundle_cli_refuses_a_foreign_checkpoint(tmp_path):
    ckpt, digest = _real_ckpt(tmp_path)
    foreign, _ = _real_ckpt(tmp_path, arm="van", content=b"another run")
    stale = _write_bundle(tmp_path, ckpt_path=foreign)
    proc = _cli("check-bundle", "--pt", stale, "--expect-n", "4", "--expect-seed", "42",
                "--expect-K", "8", "--expect-arm", "cyl", "--expect-ckpt", ckpt,
                "--expect-ckpt-sha256", digest)
    assert proc.returncode == 3, proc.stdout
    assert "ckpt_path" in proc.stdout


def test_check_metrics_cli_exit_codes(tmp_path):
    ckpt, _ = _real_ckpt(tmp_path)
    good = _write_metrics(tmp_path, ckpt)
    flags = ("--expect-ckpt", ckpt, "--expect-cond-method", "fa_invariant",
             "--expect-angles", "0", "--expect-rotate", "0", "--expect-autocast", "bf16")
    ok = _cli("check-metrics", "--json", good, *flags)
    assert ok.returncode == 0, ok.stderr
    assert "PASS" in ok.stdout and ckpt in ok.stdout

    foreign, _ = _real_ckpt(tmp_path, arm="van", content=b"another run")
    stale = _write_metrics(tmp_path, foreign, name="stale.json")
    bad = _cli("check-metrics", "--json", stale, *flags)
    assert bad.returncode == 3, bad.stdout
    assert "ckpt_path" in bad.stdout

    argerr = _cli("check-metrics", "--json", good, "--expect-ckpt", ckpt)
    assert argerr.returncode == 2, argerr.stdout    # every protocol flag is required
