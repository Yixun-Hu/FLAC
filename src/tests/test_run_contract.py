"""Tests for exp_14 round D — the per-run training contract (plan §4 "Round D", D11-D17).

Finding r2-2: six data-curve runs differ *only* in which split file they read, and their
checkpoints all land on the same NAS under near-identical names. Nothing in a `.ckpt`
says which fraction, which split, which code and which launch produced it, so a
mis-pointed `--dataset-config`, a resumed-from-the-wrong-run checkpoint or a stale
evaluation would be indistinguishable from the real thing *after the fact*.

Round D closes that: every checkpoint carries a `run_contract` dict written by a
Lightning callback, and `validate_checkpoint` is the fail-closed gate the launcher uses
before it calls a checkpoint "done" (step, embedded model config, contract identity, and
optimizer/scheduler state when the checkpoint is about to be resumed from).

Two hashes with two different jobs, never mixed (finding r3-2):

* ``file_sha256`` — the bytes of a *file* on disk (the dataset config, the split JSON).
  These pin inputs the run reads but never embeds.
* ``canonical_digest`` — a sha256 over the canonical JSON serialisation of a *parsed
  object*. The model config is compared this way because the checkpoint embeds the parsed
  dict (``ModelConfigEmbedderCallback``), never the file, so a byte-sha of the source
  file could never be checked against it. The digest must therefore be independent of the
  source file's formatting and key order.

CPU-only: the Lightning runs below are `accelerator="cpu"` on a two-parameter module.
"""
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import types

import pytest
import pytorch_lightning as pl
import torch

import train
from prefigure.prefigure import get_all_args
from train import ModelConfigEmbedderCallback
from src.training.run_contract import (
    CONTRACT_FILENAME,
    IDENTITY_FIELDS,
    CONTRACT_VERSION,
    RESUME_LOG_FILENAME,
    CheckpointContractError,
    ContractMismatchError,
    RunContractCallback,
    append_resume_log,
    build_contract,
    canonical_digest,
    file_sha256,
    load_or_create_contract,
    validate_checkpoint,
)
from src.training.run_contract import main as run_contract_main

CONTRACT_KEYS = {
    "run_id",
    "fraction",
    "dataset_config_sha256",
    "split_sha256",
    "model_config_digest",
    "model_config_path",
    "seed",
    "micro_batch",
    "num_gpus",
    "accum_batches",
    "sync_batchnorm",
    "flac_sha",
    "package_sha",
    "launched_at",
    "contract_version",
}

LAUNCHED_AT = "2026-09-16T18:00:00+00:00"

MODEL_CONFIG = {
    "model_type": "diffusion_cond",
    "sample_size": 65536,
    "model": {"diffusion": {"cross_attention_cond_ids": ["context_poses", "context_audio"]}},
    "training": {"learning_rate": 5e-5},
}


def write_json(path, obj, **dump_kwargs):
    with open(path, "w") as fout:
        json.dump(obj, fout, **dump_kwargs)
    return str(path)


@pytest.fixture
def launch_files(tmp_path):
    """The three files a launch pins: model config, dataset config, split JSON."""
    return {
        "model_config": write_json(tmp_path / "FLAC_AR_exp14_cylS.json", MODEL_CONFIG, indent=4),
        "dataset_config": write_json(
            tmp_path / "acousticroom_train_frac025.json",
            {"datasets": [{"json_file_path": "data/AR/train_frac025_s2026.json"}]},
        ),
        "split": write_json(
            tmp_path / "train_frac025_s2026.json", {"Scene": {"Scene_idx_0": ["a.wav"]}}
        ),
    }


def make_contract(launch_files, **overrides):
    kwargs = dict(
        run_id="dc_cyl_f025",
        fraction=0.25,
        dataset_config_path=launch_files["dataset_config"],
        split_json_path=launch_files["split"],
        model_config_path=launch_files["model_config"],
        seed=42,
        micro_batch=32,
        num_gpus=2,
        accum_batches=1,
        sync_batchnorm=True,
        flac_sha="a" * 40,
        package_sha="b" * 40,
        launched_at=LAUNCHED_AT,
    )
    kwargs.update(overrides)
    return build_contract(**kwargs)


# ======================================================================================
# D11 — canonical_digest: stable across key order and source-file formatting
# ======================================================================================
def test_D11_canonical_digest_is_the_documented_sha256():
    """The definition is the contract (both sides of the check must compute it the same
    way): sha256 of the FROZEN literal ``json.dumps(obj, sort_keys=True,
    separators=(',', ':'))`` -- stdlib defaults otherwise, ``ensure_ascii`` included -- 
    encoded as UTF-8. Anything added to that call changes every digest ever computed."""
    payload = json.dumps(MODEL_CONFIG, sort_keys=True, separators=(",", ":"))
    assert canonical_digest(MODEL_CONFIG) == hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_D11_canonical_digest_is_key_order_independent():
    """Dict insertion order is not part of the model config's identity."""
    forwards = {"a": 1, "b": {"x": 1, "y": 2}, "c": [1, 2, 3]}
    backwards = {"c": [1, 2, 3], "b": {"y": 2, "x": 1}, "a": 1}
    assert forwards == backwards  # sanity: the same mapping, different insertion order
    assert canonical_digest(forwards) == canonical_digest(backwards)


def test_D11_canonical_digest_independent_of_json_file_formatting(tmp_path):
    """The same config written compact / indented / with reordered keys parses to one
    digest. This is why the model config is digested from the PARSED dict and never
    byte-sha'd: the checkpoint embeds the dict, not the file."""
    compact = write_json(tmp_path / "compact.json", MODEL_CONFIG, separators=(",", ":"))
    indented = write_json(tmp_path / "indented.json", MODEL_CONFIG, indent=4)
    sorted_keys = write_json(tmp_path / "sorted.json", MODEL_CONFIG, indent=2, sort_keys=True)

    digests = {canonical_digest(json.load(open(p))) for p in (compact, indented, sorted_keys)}
    assert len(digests) == 1
    # ... while the *byte* shas of those three files are all different.
    assert len({file_sha256(p) for p in (compact, indented, sorted_keys)}) == 3


def test_D11_canonical_digest_is_order_sensitive_inside_lists():
    """Lists are ordered data (``cross_attention_cond_ids`` order matters), so reordering
    one is a different config."""
    assert canonical_digest({"ids": ["a", "b"]}) != canonical_digest({"ids": ["b", "a"]})


@pytest.mark.parametrize(
    "changed",
    [
        {"model_type": "autoencoder"},                       # top-level value
        {"sample_size": 65535},                              # numeric value
        {"training": {"learning_rate": 5.1e-5}},             # nested value
        {"extra": 1},                                        # an added key
    ],
)
def test_D11_canonical_digest_detects_any_change(changed):
    altered = dict(MODEL_CONFIG)
    altered.update(changed)
    assert canonical_digest(altered) != canonical_digest(MODEL_CONFIG)


def test_D11_canonical_digest_escapes_non_ascii():
    """(codex NIT 4) The frozen literal keeps the stdlib default ``ensure_ascii=True``, so
    a non-ASCII value is serialised as its ``\\uXXXX`` escape. Both real arm configs are
    pure ASCII (pinned below), so this is a question of *which literal is frozen*, not of
    any digest changing."""
    obj = {"note": "45\u00b0 control"}
    expected = hashlib.sha256(r'{"note":"45\u00b0 control"}'.encode("utf-8")).hexdigest()
    assert canonical_digest(obj) == expected


KIT_CONFIG_DIR = (
    "/home/yixunhu/codespace/cylindrical-dinov3/worklog/worklog_yixun/"
    "exp_14_data_curve_claude/configs"
)


@pytest.mark.skipif(not os.path.isdir(KIT_CONFIG_DIR), reason="exp_14 kit configs live in the sibling cylindrical-dinov3 checkout")
@pytest.mark.parametrize("config,expected_prefix", [
    ("FLAC_AR_exp14_cylS.json", "83c9119e"),
    ("FLAC_AR_exp14_vanS.json", "2023ccc6"),
])
def test_D11_canonical_digest_of_the_kit_configs_is_pinned(config, expected_prefix):
    """Regression pin on the two configs the campaign will actually launch with: their
    canonical digests are the values the round-D1 reviewer computed. Both files are pure
    ASCII, so switching the literal to ``ensure_ascii``-default left them unchanged -- the
    test proves it rather than asserting it."""
    with open(os.path.join(KIT_CONFIG_DIR, config)) as fin:
        parsed = json.load(fin)
    digest = canonical_digest(parsed)
    assert digest.startswith(expected_prefix)
    assert json.dumps(parsed, sort_keys=True, separators=(",", ":")).isascii()


def test_D11_file_sha256_is_the_bytes_of_the_file(tmp_path):
    path = tmp_path / "blob.bin"
    path.write_bytes(b"\x00\xff" * 1024)
    assert file_sha256(str(path)) == hashlib.sha256(b"\x00\xff" * 1024).hexdigest()


# ======================================================================================
# D11 — build_contract: exactly the documented keys, JSON-round-trip stable
# ======================================================================================
def test_build_contract_has_exactly_the_documented_keys(launch_files):
    assert set(make_contract(launch_files)) == CONTRACT_KEYS


def test_build_contract_pins_the_launch_inputs(launch_files):
    contract = make_contract(launch_files)
    assert contract["run_id"] == "dc_cyl_f025"
    assert contract["fraction"] == 0.25
    assert contract["dataset_config_sha256"] == file_sha256(launch_files["dataset_config"])
    assert contract["split_sha256"] == file_sha256(launch_files["split"])
    assert contract["model_config_digest"] == canonical_digest(MODEL_CONFIG)
    assert contract["model_config_path"] == launch_files["model_config"]
    assert contract["seed"] == 42
    assert contract["micro_batch"] == 32
    assert contract["num_gpus"] == 2
    assert contract["accum_batches"] == 1
    assert contract["sync_batchnorm"] is True
    assert contract["flac_sha"] == "a" * 40
    assert contract["package_sha"] == "b" * 40
    assert contract["launched_at"] == "2026-09-16T18:00:00+00:00"
    assert contract["contract_version"] == CONTRACT_VERSION == 1


def test_build_contract_digests_the_parsed_model_config_not_the_file(tmp_path, launch_files):
    """Re-writing the arm's config with different formatting changes its byte sha but must
    not change the contract's ``model_config_digest`` (finding r3-2)."""
    reformatted = write_json(tmp_path / "reformatted.json", MODEL_CONFIG, separators=(",", ":"))
    assert file_sha256(reformatted) != file_sha256(launch_files["model_config"])
    assert (
        make_contract(launch_files, model_config_path=reformatted)["model_config_digest"]
        == make_contract(launch_files)["model_config_digest"]
    )


def test_build_contract_survives_the_json_round_trip(launch_files):
    """The contract is persisted as ``run_contract.json`` and re-read on every resume, and
    then compared with ``==`` against the copy embedded in a checkpoint. Every value must
    therefore be a JSON-stable primitive: ``json.loads(json.dumps(c)) == c``."""
    contract = make_contract(launch_files)
    assert json.loads(json.dumps(contract)) == contract


@pytest.mark.parametrize("given,expected", [(True, True), ("true", True), (False, False), ("false", False)])
def test_build_contract_normalises_sync_batchnorm_to_a_real_bool(launch_files, given, expected):
    """prefigure yields the *string* "false"/"true" for the lowercase ini literal
    (see ``test_train_sync_batchnorm``); the contract stores a genuine bool so a contract
    built from the launcher's argv equals one built from a parsed args namespace."""
    contract = make_contract(launch_files, sync_batchnorm=given)
    assert contract["sync_batchnorm"] is expected


def test_build_contract_rejects_an_unreadable_input(tmp_path, launch_files):
    """Fail-closed at build time: a missing split file is a launch bug, not a contract."""
    with pytest.raises(OSError):
        make_contract(launch_files, split_json_path=str(tmp_path / "nope.json"))


def test_build_contract_defaults_launched_at_to_an_utc_timestamp(launch_files):
    contract = make_contract(launch_files, launched_at=None)
    assert isinstance(contract["launched_at"], str)
    assert contract["launched_at"].endswith("+00:00")


def test_checkpoint_contract_error_is_a_value_error():
    """The launcher may catch ``ValueError``; the dedicated type keeps messages precise."""
    assert issubclass(CheckpointContractError, ValueError)


# ======================================================================================
# D12 — RunContractCallback: the contract is in every checkpoint Lightning writes
# ======================================================================================
class TinyModule(pl.LightningModule):
    """Two-parameter CPU stand-in for FLAC: enough to produce real optimizer and
    scheduler state in a checkpoint, cheap enough to fit a unit test."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)

    def training_step(self, batch, batch_idx):
        return self.layer(batch).square().mean()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


def run_training(ckpt_dir, contract, model_config, max_steps, resume_from=None):
    """Run ``max_steps`` CPU steps with the three callbacks a data-curve run uses and
    return the path of the checkpoint saved at step ``max_steps``."""
    torch.manual_seed(0)
    loader = torch.utils.data.DataLoader(torch.randn(8, 2), batch_size=2)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=max_steps,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        default_root_dir=str(ckpt_dir),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=str(ckpt_dir), every_n_train_steps=1, save_top_k=-1
            ),
            ModelConfigEmbedderCallback(model_config),
            RunContractCallback(contract),
        ],
    )
    trainer.fit(TinyModule(), loader, ckpt_path=resume_from)
    saved = glob.glob(os.path.join(str(ckpt_dir), f"*step={max_steps}*.ckpt"))
    assert len(saved) == 1, saved
    return saved[0]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """One real (CPU, 1-step) training run with the round-D callbacks attached."""
    root = tmp_path_factory.mktemp("run")
    files = {
        "model_config": write_json(root / "model.json", MODEL_CONFIG, indent=4),
        "dataset_config": write_json(root / "dataset.json", {"datasets": []}),
        "split": write_json(root / "split.json", {"Scene": {"Scene_idx_0": ["a.wav"]}}),
    }
    contract = make_contract(files)
    ckpt = run_training(root / "checkpoints", contract, MODEL_CONFIG, max_steps=1)
    return {"root": root, "files": files, "contract": contract, "ckpt": ckpt}


def test_D12_checkpoint_carries_the_run_contract_and_the_model_config(trained):
    """The whole point of round D: a checkpoint identifies its own run."""
    checkpoint = torch.load(trained["ckpt"], map_location="cpu", weights_only=False)
    assert checkpoint["run_contract"] == trained["contract"]
    assert checkpoint["model_config"] == MODEL_CONFIG   # ModelConfigEmbedderCallback's key
    assert checkpoint["global_step"] == 1
    assert checkpoint["optimizer_states"][0]["state"]   # real optimizer state
    assert checkpoint["lr_schedulers"]


def test_D12_embedded_contract_is_a_deep_copy_of_the_callbacks_contract():
    """A checkpoint must not alias the launcher's live dict: mutating the embedded copy
    (or the dict handed to the constructor) may never change what later checkpoints say."""
    contract = {"run_id": "dc_cyl_f025", "nested": {"fraction": 0.25}}
    callback = RunContractCallback(contract)

    contract["nested"]["fraction"] = 0.5          # the caller mutates its own dict
    first = {}
    callback.on_save_checkpoint(None, None, first)
    assert first["run_contract"] == {"run_id": "dc_cyl_f025", "nested": {"fraction": 0.25}}

    first["run_contract"]["nested"]["fraction"] = 0.75   # something mutates a checkpoint
    second = {}
    callback.on_save_checkpoint(None, None, second)
    assert second["run_contract"]["nested"]["fraction"] == 0.25


def test_D12_callback_keeps_every_other_checkpoint_key(trained):
    """The callback only adds its key; the Lightning checkpoint is otherwise untouched."""
    checkpoint = torch.load(trained["ckpt"], map_location="cpu", weights_only=False)
    assert {"state_dict", "optimizer_states", "lr_schedulers", "global_step", "epoch"} <= set(checkpoint)


@pytest.mark.parametrize("bad", [None, "run_contract.json", ["run_id"], 1])
def test_D12_callback_refuses_a_non_dict_contract(bad):
    """Fail-closed at construction: a path or a None would embed a contract that can never
    match, and the failure would only surface hours later at validation time."""
    with pytest.raises(TypeError):
        RunContractCallback(bad)


# ======================================================================================
# D14/D15/D16 — validate_checkpoint: the fail-closed gate before a checkpoint is trusted
# ======================================================================================
def rewrite_checkpoint(source, dest, mutate):
    """Load a checkpoint, apply ``mutate`` to the dict, save the result under ``dest``."""
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    mutate(checkpoint)
    torch.save(checkpoint, dest)
    return str(dest)


def test_D14_validate_accepts_the_matching_checkpoint(trained):
    """Right step, right model config, right contract -> the loaded contract comes back."""
    returned = validate_checkpoint(
        trained["ckpt"], trained["contract"], MODEL_CONFIG, expect_step=1, for_resume=False
    )
    assert returned == trained["contract"]


def test_D14_validate_accepts_a_reformatted_but_identical_model_config(trained, tmp_path):
    """The expected config is compared as a parsed dict, so the launcher may re-read the
    arm's JSON from anywhere and in any formatting."""
    reformatted = json.load(open(write_json(tmp_path / "compact.json", MODEL_CONFIG,
                                            separators=(",", ":"), sort_keys=True)))
    assert validate_checkpoint(
        trained["ckpt"], trained["contract"], reformatted, expect_step=1, for_resume=False
    ) == trained["contract"]


def test_D14_validate_for_resume_accepts_real_optimizer_and_scheduler_state(trained):
    """D12's run really stepped the optimizer, so the checkpoint carries Adam state and a
    scheduler entry -- the extra conditions for resuming from it."""
    assert validate_checkpoint(
        trained["ckpt"], trained["contract"], MODEL_CONFIG, expect_step=1, for_resume=True
    ) == trained["contract"]


def test_D15_rejects_another_fractions_contract(trained, tmp_path):
    """The failure round D exists for: a 50 % checkpoint validated as the 25 % run's."""
    other = dict(trained["contract"], run_id="dc_cyl_f050", fraction=0.5,
                 split_sha256="c" * 64)
    with pytest.raises(CheckpointContractError) as excinfo:
        validate_checkpoint(trained["ckpt"], other, MODEL_CONFIG, expect_step=1, for_resume=False)
    message = str(excinfo.value)
    assert "run_contract" in message
    assert "fraction" in message and "run_id" in message and "split_sha256" in message


def test_D15_rejects_a_wrong_step(trained):
    with pytest.raises(CheckpointContractError) as excinfo:
        validate_checkpoint(trained["ckpt"], trained["contract"], MODEL_CONFIG,
                            expect_step=2500, for_resume=False)
    assert "global_step" in str(excinfo.value)
    assert "2500" in str(excinfo.value)


def test_D15_rejects_a_legacy_checkpoint_without_a_contract(trained, tmp_path):
    """Checkpoints written before round D (the 100 % anchors) carry no contract; they are
    not silently accepted, they are named as legacy."""
    legacy = rewrite_checkpoint(trained["ckpt"], tmp_path / "legacy.ckpt",
                                lambda ckpt: ckpt.pop("run_contract"))
    with pytest.raises(CheckpointContractError) as excinfo:
        validate_checkpoint(legacy, trained["contract"], MODEL_CONFIG, expect_step=1,
                            for_resume=False)
    assert "legacy checkpoint without run_contract" in str(excinfo.value)


def test_D15_rejects_an_embedded_model_config_that_differs_from_the_expected_dict(trained):
    """(finding r3-2) The embedded dict is compared to the expected *parsed* config, not
    only through the digest -- a contract could otherwise be reused across arms."""
    expected = dict(MODEL_CONFIG, sample_size=32768)
    with pytest.raises(CheckpointContractError) as excinfo:
        validate_checkpoint(trained["ckpt"], trained["contract"], expected, expect_step=1,
                            for_resume=False)
    assert "model config" in str(excinfo.value)


def test_D15_rejects_a_digest_mismatch(trained):
    """Separately from the dict comparison: the contract's model_config_digest must be the
    canonical digest of the config the checkpoint embeds (the two checks fail
    independently, so a contract carrying another arm's digest is caught)."""
    tampered = dict(trained["contract"], model_config_digest=canonical_digest({"other": 1}))
    with pytest.raises(CheckpointContractError) as excinfo:
        validate_checkpoint(trained["ckpt"], tampered, MODEL_CONFIG, expect_step=1,
                            for_resume=False)
    assert "digest" in str(excinfo.value)


def test_D15_rejects_a_contract_without_a_digest(trained):
    tampered = {k: v for k, v in trained["contract"].items() if k != "model_config_digest"}
    with pytest.raises(CheckpointContractError) as excinfo:
        validate_checkpoint(trained["ckpt"], tampered, MODEL_CONFIG, expect_step=1,
                            for_resume=False)
    assert "model_config_digest" in str(excinfo.value)


def test_D15_rejects_a_checkpoint_without_an_embedded_model_config(trained, tmp_path):
    stripped = rewrite_checkpoint(trained["ckpt"], tmp_path / "no_config.ckpt",
                                  lambda ckpt: ckpt.pop("model_config"))
    with pytest.raises(CheckpointContractError) as excinfo:
        validate_checkpoint(stripped, trained["contract"], MODEL_CONFIG, expect_step=1,
                            for_resume=False)
    assert "model_config" in str(excinfo.value)


@pytest.mark.parametrize("kind", ["missing", "truncated", "not_a_checkpoint"])
def test_D15_rejects_an_unloadable_file(trained, tmp_path, kind):
    """A partial .ckpt never counts (plan §4 Round D): a file the loader cannot read is a
    contract violation, not a traceback the launcher has to interpret."""
    if kind == "missing":
        path = str(tmp_path / "does_not_exist.ckpt")
    elif kind == "truncated":
        path = str(tmp_path / "partial.ckpt")
        with open(trained["ckpt"], "rb") as src, open(path, "wb") as dst:
            dst.write(src.read(4096))
    else:
        path = str(tmp_path / "text.ckpt")
        with open(path, "w") as fout:
            fout.write("not a checkpoint")

    with pytest.raises(CheckpointContractError) as excinfo:
        validate_checkpoint(path, trained["contract"], MODEL_CONFIG, expect_step=1,
                            for_resume=False)
    assert path in str(excinfo.value)


@pytest.mark.parametrize("mutate,needle", [
    (lambda ckpt: ckpt["optimizer_states"][0].__setitem__("state", {}), "optimizer"),
    (lambda ckpt: ckpt.__setitem__("optimizer_states", []), "optimizer"),
    (lambda ckpt: ckpt.pop("optimizer_states"), "optimizer"),
    (lambda ckpt: ckpt.__setitem__("lr_schedulers", []), "lr_scheduler"),
    (lambda ckpt: ckpt.pop("lr_schedulers"), "lr_scheduler"),
])
def test_D16_for_resume_rejects_missing_optimizer_or_scheduler_state(trained, tmp_path, mutate, needle):
    """Resuming from a checkpoint with no optimizer state silently restarts Adam at the
    step-0 warmup lr (CLAUDE.md "Checkpoint surgery"); for_resume makes that fail-closed."""
    path = rewrite_checkpoint(trained["ckpt"], tmp_path / f"stripped_{needle}_{id(mutate)}.ckpt", mutate)
    with pytest.raises(CheckpointContractError) as excinfo:
        validate_checkpoint(path, trained["contract"], MODEL_CONFIG, expect_step=1, for_resume=True)
    assert needle in str(excinfo.value)


def test_D16_the_same_checkpoint_is_valid_when_not_resuming(trained, tmp_path):
    """The optimizer/scheduler conditions apply ONLY to a resume: a completed run's final
    checkpoint is still a valid run artifact for evaluation."""
    stripped = rewrite_checkpoint(trained["ckpt"], tmp_path / "no_opt.ckpt",
                                  lambda ckpt: ckpt["optimizer_states"][0].__setitem__("state", {}))
    assert validate_checkpoint(stripped, trained["contract"], MODEL_CONFIG, expect_step=1,
                               for_resume=False) == trained["contract"]


# ======================================================================================
# D13 — the --run-contract-json flag: default off is byte-identical to before
# ======================================================================================
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULTS_INI = os.path.join(_REPO_ROOT, "defaults.ini")
_TRAIN_PY = os.path.join(_REPO_ROOT, "train.py")

BASE_TRAINER_KWARGS = {
    "devices", "accelerator", "num_nodes", "strategy", "precision",
    "accumulate_grad_batches", "callbacks", "logger", "log_every_n_steps",
    "max_steps", "default_root_dir", "gradient_clip_val",
    "reload_dataloaders_every_n_epochs", "num_sanity_val_steps",
}


def stub_args(**overrides):
    base = dict(
        num_gpus=1, num_nodes=1, precision="bf16-mixed", accum_batches=1,
        max_steps=1_000_000, gradient_clip_val=0.0, sync_batchnorm="false",
        run_contract_json="",
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def test_D13_defaults_ini_declares_run_contract_json_empty():
    with open(_DEFAULTS_INI) as fin:
        ini = fin.read()
    assert re.search(r"(?m)^\s*run_contract_json\s*=\s*''\s*$", ini), (
        "defaults.ini must declare `run_contract_json = ''` (default off)"
    )


def test_D13_prefigure_default_is_empty(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py"])
    args = get_all_args(defaults_file=_DEFAULTS_INI)
    assert hasattr(args, "run_contract_json")
    assert args.run_contract_json == ""


def test_D13_cli_flag_overrides_the_path(monkeypatch):
    """Prefigure maps the ini key to ``--run-contract-json`` (``_`` -> ``-``)."""
    monkeypatch.setattr(sys, "argv", ["train.py", "--run-contract-json", "/nas/run_contract.json"])
    args = get_all_args(defaults_file=_DEFAULTS_INI)
    assert args.run_contract_json == "/nas/run_contract.json"


def test_D13_default_leaves_the_callbacks_list_identical():
    """Default off => the callbacks list is exactly the three objects main() built, in
    order: every existing recipe behaves as before round D."""
    base = [object(), object(), object()]
    callbacks = train.build_callbacks(stub_args(), base)
    assert callbacks == base
    assert [id(cb) for cb in callbacks] == [id(cb) for cb in base]
    assert callbacks is not base       # a copy, so the caller's list is never mutated


@pytest.mark.parametrize("missing", [True, False])
def test_D13_default_leaves_the_trainer_kwargs_identical(missing):
    """The flag never becomes a Trainer kwarg; with it empty (or absent from args
    entirely, e.g. an older recipe) the kwargs dict is the pre-change dict."""
    args = stub_args()
    if missing:
        del args.run_contract_json
    base = [object(), object(), object()]
    kwargs = train.build_trainer_kwargs(
        args, strategy="auto", callbacks=train.build_callbacks(args, base), logger=None,
        checkpoint_dir=None, val_args={},
    )
    assert set(kwargs) == BASE_TRAINER_KWARGS
    assert kwargs["callbacks"] == base


def test_D13_flag_appends_the_contract_callback_after_the_existing_three(tmp_path, launch_files):
    """Enabled => exactly one extra callback, appended LAST, carrying the parsed contract."""
    contract = make_contract(launch_files)
    path = write_json(tmp_path / "run_contract.json", contract, indent=1)
    base = [object(), object(), object()]

    callbacks = train.build_callbacks(stub_args(run_contract_json=path), base)

    assert len(callbacks) == 4
    assert [id(cb) for cb in callbacks[:3]] == [id(cb) for cb in base]
    assert isinstance(callbacks[3], RunContractCallback)
    assert callbacks[3].contract == contract
    assert base == [base[0], base[1], base[2]] and len(base) == 3   # caller's list untouched


def test_D13_missing_contract_file_is_fail_closed(tmp_path):
    """A run launched with a contract path that does not exist must not train contract-less."""
    with pytest.raises(OSError):
        train.build_callbacks(stub_args(run_contract_json=str(tmp_path / "absent.json")), [])


def test_D13_contract_json_that_is_not_an_object_is_fail_closed(tmp_path):
    path = write_json(tmp_path / "list.json", ["dc_cyl_f025"])
    with pytest.raises(TypeError):
        train.build_callbacks(stub_args(run_contract_json=path), [])


def test_D13_revert_guard_train_py_wires_the_flag():
    """Cheap source-level revert guard: main() must route its three callbacks through
    build_callbacks, and build_callbacks must construct a RunContractCallback."""
    with open(_TRAIN_PY) as fin:
        source = fin.read()
    assert "RunContractCallback" in source
    assert "run_contract_json" in source
    assert re.search(
        r"callbacks=build_callbacks\(\s*args,\s*\[ckpt_callback,\s*exc_callback,\s*"
        r"save_model_config_callback\]", source
    ), "main() must pass its three callbacks through build_callbacks"


# ======================================================================================
# D17 — persistence and resume continuity
# ======================================================================================
def contract_builder(launch_files, timestamp=LAUNCHED_AT, **overrides):
    """A ``build_fn`` of the shape ``load_or_create_contract`` calls: it is handed the
    persisted ``launched_at`` when one exists, so the candidate it returns differs from the
    persisted contract only where the run's *identity* really changed."""
    def build_fn(launched_at=None):
        return make_contract(launch_files, launched_at=launched_at or timestamp, **overrides)
    return build_fn


def test_load_or_create_contract_creates_the_sidecar_once(tmp_path, launch_files):
    """First launch: the contract is built, written to <run-dir>/run_contract.json and
    returned. The run dir is created if the launcher has not made it yet."""
    run_dir = tmp_path / "dc_cyl_f025"
    calls = []

    def build_fn(launched_at=None):
        calls.append(launched_at)
        return make_contract(launch_files)

    contract = load_or_create_contract(str(run_dir), build_fn)

    assert calls == [None]                          # nothing persisted yet to inherit
    path = run_dir / CONTRACT_FILENAME
    assert json.load(open(path)) == contract
    assert open(path).read().startswith("{\n ")     # indent=1, readable in a worklog


def test_D17_load_or_create_contract_returns_the_persisted_contract_verbatim(tmp_path, launch_files):
    """Resume with unchanged inputs: the sidecar wins, even though this launch would have
    minted a later ``launched_at``. A rebuilt contract would no longer match any earlier
    checkpoint -- the rejection in test_D17_resumed_checkpoint... shows exactly that."""
    run_dir = tmp_path / "dc_cyl_f025"
    original = load_or_create_contract(str(run_dir), contract_builder(launch_files))

    later = load_or_create_contract(
        str(run_dir), contract_builder(launch_files, timestamp="2026-09-18T02:00:00+00:00")
    )

    assert later == original
    assert later["launched_at"] == LAUNCHED_AT


def test_load_or_create_contract_returns_what_the_file_holds(tmp_path, launch_files):
    """The returned dict is the JSON round trip of the file (not the in-memory build), so
    the in-memory contract can never differ from the one a later launch re-reads."""
    run_dir = tmp_path / "dc_cyl_f025"
    contract = load_or_create_contract(str(run_dir), contract_builder(launch_files))
    assert contract == json.load(open(run_dir / CONTRACT_FILENAME))


# --- codex BLOCKING 1: a persisted contract is reused only if the run's identity matches
IDENTITY_OVERRIDES = {
    "run_id": {"run_id": "dc_van_f025"},
    "fraction": {"fraction": 0.5},
    "seed": {"seed": 43},
    "micro_batch": {"micro_batch": 16},
    "num_gpus": {"num_gpus": 1},
    "accum_batches": {"accum_batches": 2},
    "sync_batchnorm": {"sync_batchnorm": False},
    "flac_sha": {"flac_sha": "c" * 40},
    "package_sha": {"package_sha": "d" * 40},
}


@pytest.fixture
def persisted_run(tmp_path, launch_files):
    """A run whose contract sidecar already exists (the state every resume starts from)."""
    run_dir = tmp_path / "dc_cyl_f025"
    contract = load_or_create_contract(str(run_dir), contract_builder(launch_files))
    path = run_dir / CONTRACT_FILENAME
    return {"run_dir": run_dir, "path": path, "contract": contract,
            "bytes": path.read_bytes()}


@pytest.mark.parametrize("field", sorted(IDENTITY_OVERRIDES))
def test_BLOCKING1_a_changed_identity_field_is_refused(persisted_run, launch_files, field):
    """The failure Codex blocked on: relaunching into an existing run dir with different
    inputs must NOT silently inherit the old contract (every checkpoint would then claim a
    run that never happened). One field at a time, each rejected by name."""
    builder = contract_builder(launch_files, **IDENTITY_OVERRIDES[field])

    with pytest.raises(ContractMismatchError) as excinfo:
        load_or_create_contract(str(persisted_run["run_dir"]), builder)

    assert field in str(excinfo.value)
    assert persisted_run["path"].read_bytes() == persisted_run["bytes"]   # never rewritten


def test_BLOCKING1_identity_fields_are_the_whole_contract_minus_two():
    """Exactly two keys are excluded from the identity comparison: ``launched_at`` (it is
    what a resume inherits) and ``model_config_path`` (informational, see above)."""
    assert set(IDENTITY_FIELDS) == CONTRACT_KEYS - {"launched_at", "model_config_path"}
    assert len(IDENTITY_FIELDS) == 13


def test_BLOCKING1_contract_mismatch_error_is_a_value_error():
    assert issubclass(ContractMismatchError, ValueError)
    assert not issubclass(ContractMismatchError, CheckpointContractError)


def test_append_resume_log_records_resumes_outside_the_contract(tmp_path, launch_files):
    """Each resume appends {timestamp, from_ckpt, from_sha} to a SEPARATE resume_log.json;
    the contract itself never changes, or every resumed checkpoint would stop matching."""
    run_dir = tmp_path / "dc_cyl_f025"
    contract = load_or_create_contract(str(run_dir), contract_builder(launch_files))

    first = {"timestamp": "2026-09-18T02:00:00+00:00", "from_ckpt": "step=2500.ckpt", "from_sha": "d" * 64}
    second = {"timestamp": "2026-09-19T02:00:00+00:00", "from_ckpt": "step=5000.ckpt", "from_sha": "e" * 64}
    append_resume_log(str(run_dir), first)
    append_resume_log(str(run_dir), second)

    assert json.load(open(run_dir / RESUME_LOG_FILENAME)) == [first, second]
    assert json.load(open(run_dir / CONTRACT_FILENAME)) == contract


def test_append_resume_log_refuses_a_file_that_is_not_a_list(tmp_path):
    run_dir = tmp_path / "dc_cyl_f025"
    os.makedirs(run_dir)
    write_json(run_dir / RESUME_LOG_FILENAME, {"timestamp": "2026-09-18T02:00:00+00:00"})
    with pytest.raises(ValueError):
        append_resume_log(str(run_dir), {"timestamp": "x"})


def test_D17_resumed_checkpoint_carries_the_original_contract(trained, tmp_path):
    """The continuity proof. A second Trainer resumes from the D12 checkpoint with the
    contract RE-READ from the sidecar; the checkpoint it saves at step 2 validates against
    the ORIGINAL contract, while a freshly minted one (new launched_at) is rejected -- so
    the launcher must re-read, never rebuild."""
    run_dir = trained["root"]
    persisted = load_or_create_contract(str(run_dir), lambda launched_at=None: trained["contract"])
    assert persisted == trained["contract"]

    resumed_ckpt = run_training(run_dir / "checkpoints", persisted, MODEL_CONFIG,
                                max_steps=2, resume_from=trained["ckpt"])

    assert validate_checkpoint(resumed_ckpt, trained["contract"], MODEL_CONFIG,
                               expect_step=2, for_resume=True) == trained["contract"]

    fresh = make_contract(trained["files"], launched_at="2026-09-18T02:00:00+00:00")
    assert fresh != trained["contract"]
    with pytest.raises(CheckpointContractError) as excinfo:
        validate_checkpoint(resumed_ckpt, fresh, MODEL_CONFIG, expect_step=2, for_resume=True)
    assert "launched_at" in str(excinfo.value)


# ======================================================================================
# CLIs — what the round-D2 bash launcher calls (`python -m src.training.run_contract …`)
# ======================================================================================
def make_contract_argv(run_dir, launch_files, **overrides):
    argv = {
        "--run-dir": str(run_dir),
        "--run-id": "dc_cyl_f025",
        "--fraction": "0.25",
        "--dataset-config": launch_files["dataset_config"],
        "--split-json": launch_files["split"],
        "--model-config": launch_files["model_config"],
        "--seed": "42",
        "--micro-batch": "32",
        "--num-gpus": "2",
        "--accum-batches": "1",
        "--sync-batchnorm": "true",
        "--flac-sha": "a" * 40,
        "--package-sha": "b" * 40,
        "--launched-at": "2026-09-16T18:00:00+00:00",
    }
    argv.update(overrides)
    return ["make-contract"] + [token for pair in argv.items() for token in pair]


def test_cli_make_contract_creates_the_sidecar_and_prints_its_path(tmp_path, launch_files, capsys):
    run_dir = tmp_path / "dc_cyl_f025"

    assert run_contract_main(make_contract_argv(run_dir, launch_files)) == 0

    printed = capsys.readouterr().out.strip()
    assert printed == str(run_dir / CONTRACT_FILENAME)     # stdout is the path, for $(...)
    assert json.load(open(printed)) == make_contract(launch_files)


def test_cli_make_contract_is_idempotent_and_never_rebuilds(tmp_path, launch_files, capsys):
    """A resume re-runs the same launcher line: the persisted contract is returned verbatim,
    even though the second invocation passes a different --launched-at."""
    run_dir = tmp_path / "dc_cyl_f025"
    run_contract_main(make_contract_argv(run_dir, launch_files))
    capsys.readouterr()

    assert run_contract_main(make_contract_argv(
        run_dir, launch_files, **{"--launched-at": "2026-09-18T02:00:00+00:00"})) == 0

    assert capsys.readouterr().out.strip() == str(run_dir / CONTRACT_FILENAME)
    assert json.load(open(run_dir / CONTRACT_FILENAME))["launched_at"] == "2026-09-16T18:00:00+00:00"


def test_cli_make_contract_parses_sync_batchnorm_as_a_bool(tmp_path, launch_files):
    run_dir = tmp_path / "dc_van_f025"
    run_contract_main(make_contract_argv(run_dir, launch_files, **{"--sync-batchnorm": "false"}))
    assert json.load(open(run_dir / CONTRACT_FILENAME))["sync_batchnorm"] is False


def validate_argv(ckpt, contract_path, model_config_path, expect_step, for_resume=False):
    argv = ["validate", "--ckpt", str(ckpt), "--contract", str(contract_path),
            "--model-config", str(model_config_path), "--expect-step", str(expect_step)]
    return argv + (["--for-resume"] if for_resume else [])


@pytest.fixture
def validate_inputs(trained, tmp_path):
    return {
        "ckpt": trained["ckpt"],
        "contract": write_json(tmp_path / "run_contract.json", trained["contract"], indent=1),
        "model_config": write_json(tmp_path / "model.json", MODEL_CONFIG, indent=4),
    }


def test_cli_validate_exits_zero_on_a_matching_checkpoint(validate_inputs, capsys):
    code = run_contract_main(validate_argv(validate_inputs["ckpt"], validate_inputs["contract"],
                                           validate_inputs["model_config"], 1, for_resume=True))
    assert code == 0
    assert "dc_cyl_f025" in capsys.readouterr().out


def test_cli_validate_exits_three_and_prints_the_failure(validate_inputs, capsys):
    """Exit 3 is the launcher's "this checkpoint does not count" signal; the reason goes to
    stderr so it lands in the teed log."""
    code = run_contract_main(validate_argv(validate_inputs["ckpt"], validate_inputs["contract"],
                                           validate_inputs["model_config"], 40000))
    assert code == 3
    assert "global_step" in capsys.readouterr().err


def test_cli_validate_exits_two_when_its_own_inputs_are_unreadable(validate_inputs, tmp_path, capsys):
    """A missing contract sidecar is a launcher bug, not a checkpoint verdict: a distinct
    non-zero exit code so the two are never confused."""
    code = run_contract_main(validate_argv(validate_inputs["ckpt"], tmp_path / "absent.json",
                                           validate_inputs["model_config"], 1))
    assert code == 2
    assert capsys.readouterr().err


def test_cli_module_entry_point_runs(validate_inputs):
    """`python -m src.training.run_contract` is the form the round-D2 bash launcher uses,
    so the module-level __main__ wiring (exit codes included) is exercised for real."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=_REPO_ROOT)
    base = [sys.executable, "-m", "src.training.run_contract"]

    ok = subprocess.run(base + validate_argv(validate_inputs["ckpt"], validate_inputs["contract"],
                                             validate_inputs["model_config"], 1),
                        cwd=_REPO_ROOT, env=env, capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr

    bad = subprocess.run(base + validate_argv(validate_inputs["ckpt"], validate_inputs["contract"],
                                              validate_inputs["model_config"], 2),
                         cwd=_REPO_ROOT, env=env, capture_output=True, text=True)
    assert bad.returncode == 3
    assert "global_step" in bad.stderr

