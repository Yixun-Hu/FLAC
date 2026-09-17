"""The launcher's contract with ``src.tools.data_curve.names`` -- frozen while runs are live.

``scripts/exp14_launch.sh`` shells out to ``python -m src.tools.data_curve.names
check-bundle`` and ``check-metrics`` for all twenty cells of a pair, from THIS worktree,
during the eval phase of a run that is already training. Anything the analysis tooling
wants from that module therefore has to be additive: same argv, same exit code, same
printed line, same number of checkpoint hashes.

This file is that contract, written down. It pins the CLI's observable behaviour on
fixtures, proves the CLI passes exactly the nine arguments it always passed (so the new
``precomputed_ckpt_sha256`` keyword cannot reach it), and pins the signature of
``check_bundle`` parameter by parameter -- the one place a well-meant refactor would
otherwise break a live run silently.

CPU-only; bundles are 2-item stand-ins for the 6,337-item split.
"""
import hashlib
import inspect
import json
import os

import pytest
import torch

from src.tools.data_curve import names

N_ITEMS = 2
CKPT_BYTES = b"a checkpoint, for the purposes of argument"
#: check_bundle's parameters as the launcher has always called them, in order.
FROZEN_CHECK_BUNDLE_PARAMS = [
    ("path", inspect.Parameter.empty), ("expect_n", inspect.Parameter.empty),
    ("expect_seed", inspect.Parameter.empty), ("expect_K", inspect.Parameter.empty),
    ("expect_arm", inspect.Parameter.empty), ("expect_eval_name", None),
    ("config_root", None), ("expect_ckpt", None), ("expect_ckpt_sha256", None),
]


def make_cell(tmp_path, arm="cyl", tag="025", K=8, seed=42, meta_patch=None,
              record_patch=None):
    """One cell's checkpoint, metrics JSON and prediction bundle, exactly as scored."""
    run_dir = tmp_path / names.run_id(arm, tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = str(run_dir / f"epoch=8-step={names.MAX_STEPS}.ckpt")
    with open(ckpt, "wb") as fout:
        fout.write(CKPT_BYTES)
    digest = hashlib.sha256(CKPT_BYTES).hexdigest()
    angles = [0.0] if arm == "cyl" else None
    record = {"metrics": {"T60": 9.0}, "ckpt_path": ckpt, "rotate_deg": names.ROTATE_DEG,
              "cond_method": names.ARM_COND_METHOD[arm], "frame_avg_angles": angles,
              "cond_autocast": names.COND_AUTOCAST, "ckpt_sha256": digest}
    record.update(record_patch or {})
    with open(names.metrics_json_path(ckpt, arm, tag, K, seed), "w") as fout:
        json.dump(record, fout)
    meta = {"dataset_config": names.EVAL_DATASET_CONFIGS[K], "seed": seed,
            "n_samples": N_ITEMS, "n_items": N_ITEMS, "batch_size": 8,
            "cond_method": names.ARM_COND_METHOD[arm], "frame_avg_angles": angles,
            "rotate_deg": names.ROTATE_DEG, "cond_autocast": names.COND_AUTOCAST,
            "ckpt_path": ckpt, "ckpt_sha256": digest,
            "eval_name": names.eval_name(arm, tag, K, seed), "steps": names.EVAL_STEPS,
            "cfg_scale": names.EVAL_CFG_SCALE, "stored_after_clamp_pad": True,
            "artifact_contract": names.PREDICTIONS_ARTIFACT_CONTRACT}
    meta.update(meta_patch or {})
    bundle = names.predictions_pt_path(ckpt, arm, tag, K, seed)
    torch.save({"predictions": torch.zeros(N_ITEMS, 1, names.SAMPLE_LEN), "meta": meta},
               bundle)
    return ckpt, digest, bundle, names.metrics_json_path(ckpt, arm, tag, K, seed)


def count_hashes_of(monkeypatch, target):
    """Count ``names.file_sha256`` calls against one path, leaving its behaviour intact."""
    calls = []
    real = names.file_sha256

    def counting(path):
        if os.path.normpath(path) == os.path.normpath(target):
            calls.append(path)
        return real(path)

    monkeypatch.setattr(names, "file_sha256", counting)
    return calls


def bundle_argv(bundle, ckpt, digest, arm="cyl", tag="025", K=8, seed=42):
    return ["check-bundle", "--pt", bundle, "--expect-n", str(N_ITEMS),
            "--expect-seed", str(seed), "--expect-K", str(K), "--expect-arm", arm,
            "--expect-eval-name", names.eval_name(arm, tag, K, seed),
            "--expect-ckpt", ckpt, "--expect-ckpt-sha256", digest]


# ------------------------------------------------------ the CLI, exactly as it was
def test_check_bundle_cli_passes_a_good_cell_and_hashes_the_checkpoint_once(
        tmp_path, capsys, monkeypatch):
    ckpt, digest, bundle, _ = make_cell(tmp_path)
    hashes = count_hashes_of(monkeypatch, ckpt)
    assert names.main(bundle_argv(bundle, ckpt, digest)) == names.EXIT_OK
    out = capsys.readouterr().out
    assert out.startswith(f"PASS {bundle}: n=2 seed=42 K=8 arm=cyl (fa_invariant, ")
    assert f"ckpt={ckpt} ckpt_sha256={digest}" in out
    assert len(hashes) == 1                    # the CLI still hashes, exactly once


def test_check_bundle_cli_still_catches_a_checkpoint_replaced_after_validation(
        tmp_path, capsys, monkeypatch):
    ckpt, digest, bundle, _ = make_cell(tmp_path)
    with open(ckpt, "wb") as fout:
        fout.write(b"different bytes at the same pathname")
    hashes = count_hashes_of(monkeypatch, ckpt)
    assert names.main(bundle_argv(bundle, ckpt, digest)) == names.EXIT_BUNDLE_VIOLATION
    assert "it was replaced after validation" in capsys.readouterr().out
    assert len(hashes) == 1


def test_check_metrics_cli_is_unchanged(tmp_path, capsys):
    ckpt, digest, _, metrics = make_cell(tmp_path)
    argv = ["check-metrics", "--json", metrics, "--expect-ckpt", ckpt,
            "--expect-ckpt-sha256", digest, "--expect-cond-method", "fa_invariant",
            "--expect-angles", "0", "--expect-rotate", "0", "--expect-autocast", "bf16"]
    assert names.main(argv) == names.EXIT_OK
    assert capsys.readouterr().out.startswith(f"PASS {metrics}:")
    ckpt2, digest2, _, metrics2 = make_cell(tmp_path / "other", arm="van",
                                            record_patch={"cond_method": "fa_invariant"})
    argv[argv.index("--json") + 1] = metrics2
    argv[argv.index("--expect-ckpt") + 1] = ckpt2
    argv[argv.index("--expect-ckpt-sha256") + 1] = digest2
    argv[argv.index("--expect-cond-method") + 1] = "vanilla"
    assert names.main(argv) == names.EXIT_BUNDLE_VIOLATION
    assert "cond_method" in capsys.readouterr().out


def test_the_cli_calls_check_bundle_with_the_nine_arguments_it_always_did(
        tmp_path, monkeypatch):
    ckpt, digest, bundle, _ = make_cell(tmp_path)
    seen = {}

    def spy(*args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        return []

    monkeypatch.setattr(names, "check_bundle", spy)
    assert names.main(bundle_argv(bundle, ckpt, digest)) == names.EXIT_OK
    assert len(seen["args"]) == 9 and seen["kwargs"] == {}
    assert "precomputed_ckpt_sha256" not in seen["kwargs"]


def test_check_bundle_signature_only_ever_grew_at_the_end():
    params = list(inspect.signature(names.check_bundle).parameters.values())
    assert [(p.name, p.default) for p in params[:9]] == FROZEN_CHECK_BUNDLE_PARAMS
    assert [p.kind for p in params] == [inspect.Parameter.POSITIONAL_OR_KEYWORD] * len(params)
    assert [(p.name, p.default) for p in params[9:]] == [("precomputed_ckpt_sha256", None)]


# ------------------------------------------- the new keyword, for the analysis tooling
def test_a_precomputed_digest_binds_the_bundle_without_touching_the_file(
        tmp_path, monkeypatch):
    ckpt, digest, bundle, _ = make_cell(tmp_path)
    hashes = count_hashes_of(monkeypatch, ckpt)
    assert names.check_bundle(bundle, N_ITEMS, 42, 8, "cyl", expect_ckpt=ckpt,
                              expect_ckpt_sha256=digest,
                              precomputed_ckpt_sha256=digest) == []
    assert hashes == []


@pytest.mark.parametrize("precomputed, needle", [
    ("b" * 64, "it was replaced after validation"),      # the file is not what we validated
])
def test_a_precomputed_digest_that_disagrees_is_still_a_violation(tmp_path, monkeypatch,
                                                                  precomputed, needle):
    ckpt, digest, bundle, _ = make_cell(tmp_path)
    hashes = count_hashes_of(monkeypatch, ckpt)
    violations = names.check_bundle(bundle, N_ITEMS, 42, 8, "cyl", expect_ckpt=ckpt,
                                    expect_ckpt_sha256=digest,
                                    precomputed_ckpt_sha256=precomputed)
    assert any(needle in v for v in violations)
    assert hashes == []


def test_a_precomputed_digest_does_not_excuse_a_bundle_that_carries_the_wrong_one(tmp_path):
    ckpt, digest, bundle, _ = make_cell(tmp_path, meta_patch={"ckpt_sha256": "c" * 64})
    violations = names.check_bundle(bundle, N_ITEMS, 42, 8, "cyl", expect_ckpt=ckpt,
                                    expect_ckpt_sha256=digest,
                                    precomputed_ckpt_sha256=digest)
    assert any("meta.ckpt_sha256" in v for v in violations)
