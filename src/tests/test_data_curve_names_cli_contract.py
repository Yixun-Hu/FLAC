"""The launcher's contract with ``src.tools.data_curve.names`` -- frozen while runs are live.

``scripts/exp14_launch.sh`` shells out to ``python -m src.tools.data_curve.names
check-bundle`` and ``check-metrics`` for all twenty cells of a pair, from THIS worktree,
during the eval phase of a run that is already training. Anything the analysis tooling
wants from that module therefore has to be additive: same argv, same exit code, the same
bytes on stdout and stderr, same number of checkpoint hashes.

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
#: What someone swaps in at the same pathname after the launcher validated the original.
REPLACEMENT_BYTES = b"different bytes at the same pathname"
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


def captured(capsys, tmp_path):
    """``(stdout, stderr)`` with the one varying substring -- the tmp directory -- named.

    Both streams, in full, for every frozen case (codex D3-fix2 finding 4). A
    ``startswith`` or ``in`` check waves through an extra line, a changed suffix and
    altered provenance text alike, and a launcher parsing this output survives none of
    those.
    """
    out = capsys.readouterr()
    return out.out.replace(str(tmp_path), "<tmp>"), out.err.replace(str(tmp_path), "<tmp>")


# ------------------------------------------------- the frozen bytes, spelled out in full
#: Literals, deliberately: deriving them from ``names`` would ask the module under contract
#: what its own contract is. ``test_the_frozen_strings_are_what_the_fixtures_hold`` below
#: says where each one comes from, so a legitimate change fails there by name as well.
CKPT_SHA256 = "a7ef7cb1d3dfdc756c4da6dd8167b1c9f152858e8b3c18f667ea6b5929fec05c"
REPLACEMENT_SHA256 = "2a8c5ea6f994bbaa5cd3cf19cc86e80d9c1551d76eba3c1d5bb3945f067a0db9"
K8_CONFIG = "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json"
K8_CONFIG_SHA256 = "063c66c2411cde4b1f07ec7c5331150b322517cf0067a0ef3def819368423b55"
CKPT = "<tmp>/dc_cyl_f025/epoch=8-step=40000.ckpt"
BUNDLE = ("<tmp>/dc_cyl_f025/"
          "epoch=8-step=40000_predictions_1_1.0_dc_cyl_f025_K8_s42_fa_invariant_a1.pt")
METRICS = ("<tmp>/dc_cyl_f025/"
           "epoch=8-step=40000_metrics_1_1.0_dc_cyl_f025_K8_s42_fa_invariant_a1.json")
VAN_METRICS = ("<tmp>/other/dc_van_f025/"
               "epoch=8-step=40000_metrics_1_1.0_dc_van_f025_K8_s42.json")
EXPECT_BUNDLE_PASS = (
    f"PASS {BUNDLE}: n=2 seed=42 K=8 arm=cyl (fa_invariant, autocast bf16, rotate 0) "
    f"dataset_config={K8_CONFIG} dataset_config_sha256={K8_CONFIG_SHA256} "
    f"ckpt={CKPT} ckpt_sha256={CKPT_SHA256}\n")
EXPECT_BUNDLE_REPLACED = (
    f"FAIL {BUNDLE}: the checkpoint '{CKPT}' now hashes to {REPLACEMENT_SHA256}, not the "
    f"{CKPT_SHA256} the launcher validated: it was replaced after validation\n")
EXPECT_METRICS_PASS = (
    f"PASS {METRICS}: cond_method=fa_invariant angles=0 rotate=0.0 autocast=bf16 "
    f"ckpt={CKPT} ckpt_sha256={CKPT_SHA256}\n")
EXPECT_METRICS_FAIL = (
    f"FAIL {VAN_METRICS}: cond_method is 'fa_invariant', expected 'vanilla'\n")


def test_the_frozen_strings_are_what_the_fixtures_hold():
    assert CKPT_SHA256 == hashlib.sha256(CKPT_BYTES).hexdigest()
    assert REPLACEMENT_SHA256 == hashlib.sha256(REPLACEMENT_BYTES).hexdigest()
    assert K8_CONFIG == names.EVAL_DATASET_CONFIGS[8]
    assert K8_CONFIG_SHA256 == names.eval_dataset_config_sha256(8)


# ------------------------------------------------------ the CLI, exactly as it was
def test_check_bundle_cli_passes_a_good_cell_and_hashes_the_checkpoint_once(
        tmp_path, capsys, monkeypatch):
    ckpt, digest, bundle, _ = make_cell(tmp_path)
    hashes = count_hashes_of(monkeypatch, ckpt)
    assert names.main(bundle_argv(bundle, ckpt, digest)) == names.EXIT_OK
    assert captured(capsys, tmp_path) == (EXPECT_BUNDLE_PASS, "")
    assert len(hashes) == 1                    # the CLI still hashes, exactly once


def test_check_bundle_cli_still_catches_a_checkpoint_replaced_after_validation(
        tmp_path, capsys, monkeypatch):
    ckpt, digest, bundle, _ = make_cell(tmp_path)
    with open(ckpt, "wb") as fout:
        fout.write(REPLACEMENT_BYTES)
    hashes = count_hashes_of(monkeypatch, ckpt)
    assert names.main(bundle_argv(bundle, ckpt, digest)) == names.EXIT_BUNDLE_VIOLATION
    assert captured(capsys, tmp_path) == (EXPECT_BUNDLE_REPLACED, "")
    assert len(hashes) == 1


def test_check_metrics_cli_is_unchanged(tmp_path, capsys):
    ckpt, digest, _, metrics = make_cell(tmp_path)
    argv = ["check-metrics", "--json", metrics, "--expect-ckpt", ckpt,
            "--expect-ckpt-sha256", digest, "--expect-cond-method", "fa_invariant",
            "--expect-angles", "0", "--expect-rotate", "0", "--expect-autocast", "bf16"]
    assert names.main(argv) == names.EXIT_OK
    assert captured(capsys, tmp_path) == (EXPECT_METRICS_PASS, "")
    ckpt2, digest2, _, metrics2 = make_cell(tmp_path / "other", arm="van",
                                            record_patch={"cond_method": "fa_invariant"})
    argv[argv.index("--json") + 1] = metrics2
    argv[argv.index("--expect-ckpt") + 1] = ckpt2
    argv[argv.index("--expect-ckpt-sha256") + 1] = digest2
    argv[argv.index("--expect-cond-method") + 1] = "vanilla"
    assert names.main(argv) == names.EXIT_BUNDLE_VIOLATION
    assert captured(capsys, tmp_path) == (EXPECT_METRICS_FAIL, "")


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
