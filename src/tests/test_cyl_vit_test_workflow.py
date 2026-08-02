"""Regression tests for the cyl-vit-test train/inference launch surface."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
EXPDIR = ROOT / "worklog/worklog_zhixuan/cyl_vit_test"


def load_script_module(name: str):
    path = EXPDIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_script(name: str, **environment: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update({key: str(value) for key, value in environment.items()})
    env["PYTHON"] = sys.executable
    return subprocess.run(
        ["bash", str(EXPDIR / name)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_registered_config_is_only_the_cylindrical_vit_substitution():
    verify = load_script_module("verify_config.py")
    config = verify.validate_configs()
    conditioners = {
        item["id"]: item["config"]
        for item in config["model"]["conditioning"]["configs"]
    }
    assert conditioners["source_vit"] == conditioners["context_poses_vit"]
    assert conditioners["source_vit"]["ViT"] == verify.CYL_VIT_BLOCK
    assert conditioners["source_vit"]["token_pool"] == "mean"


def test_registered_config_builds_one_shared_cylindrical_vit():
    verify = load_script_module("verify_config.py")
    verify.instantiate_and_validate(verify.validate_configs())


@pytest.mark.parametrize(
    "script",
    (
        "run_preflight.sh",
        "run_train.sh",
        "run_predict.sh",
        "run_eval_suite.sh",
        "run_yaw_suite.sh",
        "run_pipeline.sh",
    ),
)
def test_shell_launchers_parse(script):
    subprocess.run(["bash", "-n", str(EXPDIR / script)], check=True)


def test_train_dry_run_pins_standard_flac_recipe():
    result = run_script(
        "run_train.sh",
        DRY_RUN="1",
        GPU_IDS="0,1",
        BATCH_SIZE="8",
        GLOBAL_BATCH_SIZE="64",
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert "effective=64" in output
    assert "--max-steps 67500" in output
    assert "--num-gpus 2" in output
    assert "--accum-batches 4" in output
    assert "--checkpoint-every 2500" in output
    assert "--pretransform-ckpt-path weights/FLAC/VAE.safetensors" in output
    assert "--pretrained-ckpt-path" not in output
    assert "FLAC_AR_CylViT.json" in output


def test_train_refuses_non_integral_effective_batch():
    result = run_script(
        "run_train.sh",
        DRY_RUN="1",
        GPU_IDS="0,1",
        BATCH_SIZE="7",
        GLOBAL_BATCH_SIZE="64",
    )
    assert result.returncode == 2
    assert "not divisible" in result.stderr


@pytest.mark.parametrize(
    ("split", "k_value", "dataset_name"),
    (
        ("unseen", "1", "acousticroom_unseeneval_1.json"),
        ("unseen", "8", "acousticroom_unseeneval.json"),
        ("seen", "1", "acousticroom_seeneval_1.json"),
        ("seen", "8", "acousticroom_seeneval.json"),
    ),
)
def test_predict_dry_run_selects_the_exact_context_protocol(split, k_value, dataset_name):
    result = run_script(
        "run_predict.sh",
        DRY_RUN="1",
        CKPT_PATH="/tmp/example-step=67500.ckpt",
        SPLIT=split,
        K=k_value,
        SEED="43",
        YAW="22.5",
        STORE_PREDICTIONS="1",
    )
    assert result.returncode == 0, result.stderr
    assert dataset_name in result.stdout
    assert "--seed 43" in result.stdout
    assert "--rotate-deg 22.5" in result.stdout
    assert "--cond-method vanilla" in result.stdout
    assert "--store_predictions" in result.stdout


def test_checkpoint_resolver_requires_an_unambiguous_match(tmp_path):
    finder = load_script_module("find_checkpoint.py")
    checkpoint = tmp_path / "nested" / "epoch=14-step=67500.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    assert finder.find_checkpoint(tmp_path, 67_500) == checkpoint.resolve()

    duplicate = tmp_path / "other" / "epoch=15-step=67500.ckpt"
    duplicate.parent.mkdir()
    duplicate.touch()
    with pytest.raises(RuntimeError, match="ambiguous"):
        finder.find_checkpoint(tmp_path, 67_500)


def test_metric_summary_groups_seeds(tmp_path):
    summary = load_script_module("summarize_metrics.py")
    checkpoint = tmp_path / "epoch=14-step=67500.ckpt"
    checkpoint.touch()
    for seed, value in ((42, 10.0), (43, 12.0)):
        path = tmp_path / (
            f"{checkpoint.stem}_metrics_1_1.0_cylvit_unseen_K1_s{seed}_yaw0.json"
        )
        path.write_text(json.dumps({"metrics": {"T60": value, "C50": 1.0}}))
    records = summary.load_records(checkpoint)
    rendered = summary.build_summary(checkpoint, records)
    assert "42,43" in rendered
    assert "unseen" in rendered
    assert "11.0000 +/- 1.4142" in rendered
    assert "Matched metric files: 2" in rendered


def test_complete_pipeline_has_a_gpu_free_dry_run():
    result = run_script(
        "run_pipeline.sh",
        DRY_RUN="1",
        RUN_YAW="0",
        GPU_IDS="0",
        BATCH_SIZE="8",
        SEEDS="42",
        KS="1 8",
    )
    assert result.returncode == 0, result.stderr
    assert "--max-steps 67500" in result.stdout
    assert "cylvit_unseen_K1_s42_yaw0" in result.stdout
    assert "cylvit_unseen_K8_s42_yaw0" in result.stdout
