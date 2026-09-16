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
