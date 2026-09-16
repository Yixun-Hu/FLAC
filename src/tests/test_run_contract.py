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

import pytest
import pytorch_lightning as pl
import torch

from train import ModelConfigEmbedderCallback
from src.training.run_contract import (
    CONTRACT_VERSION,
    CheckpointContractError,
    RunContractCallback,
    build_contract,
    canonical_digest,
    file_sha256,
)

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
        launched_at="2026-09-16T18:00:00+00:00",
    )
    kwargs.update(overrides)
    return build_contract(**kwargs)


# ======================================================================================
# D11 — canonical_digest: stable across key order and source-file formatting
# ======================================================================================
def test_D11_canonical_digest_is_the_documented_sha256():
    """The definition is the contract (both sides of the check must compute it the same
    way): sha256 of ``json.dumps(obj, sort_keys=True, separators=(',', ':'),
    ensure_ascii=False)`` encoded as UTF-8."""
    payload = json.dumps(MODEL_CONFIG, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
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


def test_D11_canonical_digest_keeps_non_ascii_verbatim():
    """``ensure_ascii=False``: the payload is the UTF-8 text, not an escaped ASCII form."""
    obj = {"note": "45° control"}
    expected = hashlib.sha256('{"note":"45° control"}'.encode("utf-8")).hexdigest()
    assert canonical_digest(obj) == expected


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
