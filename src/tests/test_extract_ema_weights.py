"""exp_19 — ``src/tools/extract_ema_weights.py``: the HAA finetune's *initial* weights.

The released HAA recipe finetunes from ``weights/FLAC/FLAC_EMA.ckpt``, a plain
weights file with **bare** model keys. Our three exp_19 inits are 40k *training*
checkpoints written by PyTorch-Lightning, whose ``state_dict`` is prefixed and
carries the optimizer, the scheduler, the EMA bookkeeping and a loss buffer.
``unwrap_model.py`` — the upstream converter — imports ``stable_audio_tools`` and
does not run in this fork. This tool is the replacement, and it is *copy-only*:
the 40k artifacts are the evidence base of five closed experiments and must come
out byte-identical.

⚠️ **What "the EMA weights" actually are** (measured, not assumed — this
contradicts the round-1 brief and is reported as such). The training wrapper
builds ``EMA(self.diffusion.model, ...)`` (``src/training/diffusion.py:277``), so
the EMA shadows the **DiT only**. In a real 40k checkpoint::

    diffusion.model.*             210 tensors   (live DiT)
    diffusion.conditioner.*       561 tensors   (DINOv3 x1 shared, RIR encoder)
    diffusion.pretransform.*      295 tensors   (VAE)
    diffusion_ema.ema_model.*     210 tensors   (EMA copy of the DiT — and nothing else)
    diffusion_ema.initted/.step     2 tensors   (bookkeeping, not weights)
    losses.losses.0.weight          1 tensor    (loss module, not the model)

A file containing only the 210 bare EMA keys is therefore **not** loadable by
``train.py``: line 148 is ``model.load_state_dict(weights, strict=True)`` against
a model that has 1066 keys, and 856 of them would be missing. It is also not what
the released artifact is — ``weights/FLAC/FLAC_EMA.ckpt`` holds exactly 1066 bare
keys (``model`` 210 / ``conditioner`` 561 / ``pretransform`` 295), because
``export_model`` (``src/training/diffusion.py:911``) assigns the EMA weights *into*
``diffusion.model`` and then saves the whole wrapper. So the contract pinned here
is the released one: **every bare key, with the ``model.*`` subtree taken from the
EMA copy and the rest carried from the live weights**, and a test below proves the
EMA-only spelling would fail the strict load.

The failure modes each test guards are stated in its docstring. Written by the
exp_19 coder seat (Claude Opus 5, max effort).
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.models.utils import load_ckpt_state_dict
from src.tools.extract_ema_weights import (
    EMA_PREFIX,
    EMA_TARGET_PREFIX,
    LIVE_PREFIX,
    extract_ema_weights,
    main,
)


_REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# fixtures: a TINY synthetic wrapped checkpoint with the real key topology
# --------------------------------------------------------------------------- #
LIVE_DIT = {
    "blocks.0.weight": torch.arange(12.0).reshape(4, 3),
    "blocks.0.bias": torch.arange(4.0),
    "norm.scale": torch.full((4,), 2.0),
}
# Deliberately different values: every "did you take the EMA and not the live
# copy?" assertion below is void if the two agree.
EMA_DIT = {k: v + 100.0 for k, v in LIVE_DIT.items()}

CONDITIONER_KEY = "conditioner.conditioners.source.proj_out.weight"
PRETRANSFORM_KEY = "pretransform.model.encoder.layers.0.bias"
CARRIED = {
    CONDITIONER_KEY: torch.full((2, 3), 7.0),
    PRETRANSFORM_KEY: torch.full((2,), -1.0),
}

EXPECTED_KEYS = {EMA_TARGET_PREFIX + k for k in EMA_DIT} | set(CARRIED)


def _wrapped_state_dict():
    sd = {}
    for k, v in LIVE_DIT.items():
        sd[LIVE_PREFIX + "model." + k] = v.clone()
    for k, v in CARRIED.items():
        sd[LIVE_PREFIX + k] = v.clone()
    # The two families train.py drops (line 146-147). They must never reach the
    # output either: a file that claims to be "the weights that were loaded" and
    # silently carries a discriminator is a lie about the initialisation.
    sd[LIVE_PREFIX + "losses.x"] = torch.zeros(1)
    sd[LIVE_PREFIX + "discriminator.y"] = torch.zeros(1)
    for k, v in EMA_DIT.items():
        sd[EMA_PREFIX + k] = v.clone()
    sd["diffusion_ema.initted"] = torch.tensor(True)
    sd["diffusion_ema.step"] = torch.tensor(40000)
    return sd


def _write_ckpt(path, state_dict, **extra):
    ckpt = {
        "epoch": 8,
        "global_step": 40000,
        "pytorch-lightning_version": "2.1.0",
        "state_dict": state_dict,
        "optimizer_states": [
            {"state": {0: {"exp_avg": torch.full((4, 3), 3.0)}},
             "param_groups": [{"lr": 5e-06, "betas": [0.9, 0.999]}]}
        ],
        "lr_schedulers": [{"last_epoch": 40000, "_step_count": 40001}],
        "model_config": {"training": {"use_ema": True}},
    }
    ckpt.update(extra)
    torch.save(ckpt, path)
    return path


@pytest.fixture()
def wrapped(tmp_path):
    return _write_ckpt(tmp_path / "epoch=8-step=40000.ckpt", _wrapped_state_dict())


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _out_state_dict(path):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    return obj["state_dict"]


def train_py_load_transforms(weights):
    """train.py:142-147, replicated verbatim — the consumer this file must satisfy.

    Replicated rather than imported because ``train.py`` executes argument parsing
    and dataloader construction at import time; the transforms themselves are five
    lines and are pinned here so a drift in either copy is visible in review.
    """
    weights = {k.replace('diffusion.', ''): v for k, v in weights.items()}   # 142
    weights = {k.replace('autoencoder.', ''): v for k, v in weights.items()}  # 143
    weights = {k: v for k, v in weights.items() if 'discriminator' not in k}  # 146
    weights = {k: v for k, v in weights.items() if 'losses' not in k}         # 147
    return weights


def module_with_keys(spec):
    """An ``nn.Module`` whose ``state_dict`` key set is exactly ``spec``'s.

    Lets the round-trip test perform the REAL ``load_state_dict(..., strict=True)``
    of train.py:148 rather than merely comparing key sets — strict loading is what
    turns a missing subtree into a crash, and it is the whole reason this tool
    emits the carried weights as well as the EMA ones.
    """
    root = torch.nn.Module()
    for key, value in spec.items():
        *path, leaf = key.split(".")
        mod = root
        for part in path:
            if part not in mod._modules:
                mod.add_module(part, torch.nn.Module())
            mod = mod._modules[part]
        mod.register_buffer(leaf, torch.zeros_like(value))
    return root


# --------------------------------------------------------------------------- #
# 1. what comes out
# --------------------------------------------------------------------------- #
def test_the_fixture_distinguishes_ema_from_live_weights():
    """Non-vacuity guard for every EMA-vs-live assertion in this file."""
    for k in LIVE_DIT:
        assert not torch.equal(LIVE_DIT[k], EMA_DIT[k]), k


def test_the_output_is_the_full_bare_key_set(wrapped, tmp_path):
    """The key set is the contract with ``train.py``'s strict load.

    Not "the EMA keys": the EMA shadows the DiT only (module docstring), so an
    EMA-only file is missing the conditioner and the VAE.
    """
    out = tmp_path / "init.ckpt"
    extract_ema_weights(str(wrapped), str(out))
    assert set(_out_state_dict(out)) == EXPECTED_KEYS


def test_the_model_subtree_carries_the_EMA_tensors_not_the_live_ones(wrapped, tmp_path):
    """The one thing the tool exists for.

    Silently exporting the live weights would produce a finetune that *runs*,
    reports plausible numbers, and answers a different question than the one the
    paper recipe asks (it inits from EMA).
    """
    out = tmp_path / "init.ckpt"
    extract_ema_weights(str(wrapped), str(out))
    sd = _out_state_dict(out)
    for k, v in EMA_DIT.items():
        assert torch.equal(sd[EMA_TARGET_PREFIX + k], v), k
        assert not torch.equal(sd[EMA_TARGET_PREFIX + k], LIVE_DIT[k]), k


def test_the_non_EMA_subtrees_are_carried_from_the_live_weights(wrapped, tmp_path):
    """Conditioner and VAE have no EMA copy; they must come through unchanged."""
    out = tmp_path / "init.ckpt"
    extract_ema_weights(str(wrapped), str(out))
    sd = _out_state_dict(out)
    for k, v in CARRIED.items():
        assert torch.equal(sd[k], v), k


def test_training_state_and_dropped_families_never_leak(wrapped, tmp_path):
    """Optimizer moments, scheduler counters, losses, discriminator, EMA bookkeeping.

    Any of them in the output would either be silently dropped by train.py (making
    the file's sha a promise about bytes nobody loads) or, worse, restored as
    weights. The output is a weights file and nothing else.
    """
    out = tmp_path / "init.ckpt"
    extract_ema_weights(str(wrapped), str(out))
    obj = torch.load(out, map_location="cpu", weights_only=True)
    assert set(obj) == {"state_dict"}, obj.keys()
    for key in obj["state_dict"]:
        assert "losses" not in key, key
        assert "discriminator" not in key, key
        assert "optimizer" not in key, key
        assert not key.startswith("diffusion"), key
        assert key not in ("initted", "step"), key


def test_no_bare_key_contains_a_substring_train_py_would_rewrite(wrapped, tmp_path):
    """train.py's strip is ``str.replace``, not a prefix strip.

    A bare key containing ``diffusion.`` or ``autoencoder.`` anywhere would be
    mangled on load and then fail the strict load with a confusing name. Verified
    to hold for the real released file too (0 such keys in FLAC_EMA.ckpt).
    """
    out = tmp_path / "init.ckpt"
    extract_ema_weights(str(wrapped), str(out))
    for key in _out_state_dict(out):
        assert "diffusion." not in key, key
        assert "autoencoder." not in key, key


# --------------------------------------------------------------------------- #
# 2. the round trip that actually matters
# --------------------------------------------------------------------------- #
def test_round_trip_through_load_ckpt_state_dict_and_train_py(wrapped, tmp_path):
    """The real consumer path: ``load_ckpt_state_dict`` -> train.py:142-148.

    Uses the repo's own loader (which is where the ``["state_dict"]`` and
    ``weights_only=True`` expectations live), then the real strict load. If any
    piece of the contract is wrong, this is where it shows.
    """
    out = tmp_path / "init.ckpt"
    extract_ema_weights(str(wrapped), str(out))

    weights = load_ckpt_state_dict(str(out))
    weights = train_py_load_transforms(weights)
    assert set(weights) == EXPECTED_KEYS

    model = module_with_keys({k: v for k, v in weights.items()})
    model.load_state_dict(weights, strict=True)          # train.py:148
    loaded = model.state_dict()
    for k, v in EMA_DIT.items():
        assert torch.equal(loaded[EMA_TARGET_PREFIX + k], v), k


def test_an_EMA_only_file_would_fail_the_strict_load(wrapped, tmp_path):
    """Why the tool does not emit "just the EMA keys" (the brief's literal reading).

    This is the measured consequence, executed: 856 of 1066 keys missing in the
    real case, 2 of 5 here. Keeping it as a test means the design decision is
    re-derived on every run rather than trusted from a comment.
    """
    out = tmp_path / "init.ckpt"
    extract_ema_weights(str(wrapped), str(out))
    full = load_ckpt_state_dict(str(out))
    ema_only = {k: v for k, v in full.items() if k.startswith(EMA_TARGET_PREFIX)}
    assert 0 < len(ema_only) < len(full), "fixture must have non-EMA weights too"

    model = module_with_keys(full)
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(ema_only, strict=True)


# --------------------------------------------------------------------------- #
# 3. copy-only
# --------------------------------------------------------------------------- #
def test_the_input_checkpoint_is_byte_identical_afterwards(wrapped, tmp_path):
    """The 40k artifacts back five closed experiments; the tool may only read them."""
    before = _sha256(wrapped)
    extract_ema_weights(str(wrapped), str(tmp_path / "init.ckpt"))
    assert _sha256(wrapped) == before


def test_writing_onto_the_input_is_refused(wrapped):
    """The one spelling that would destroy the artifact: ``--out`` == ``--in``."""
    with pytest.raises(ValueError, match="same file"):
        extract_ema_weights(str(wrapped), str(wrapped))


def test_an_existing_output_is_never_overwritten(wrapped, tmp_path):
    """Two arms differ only by their init; a silent overwrite swaps an experiment."""
    out = tmp_path / "init.ckpt"
    out.write_bytes(b"previous arm's init")
    with pytest.raises(FileExistsError):
        extract_ema_weights(str(wrapped), str(out))
    assert out.read_bytes() == b"previous arm's init"


def test_a_missing_input_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_ema_weights(str(tmp_path / "nope.ckpt"), str(tmp_path / "out.ckpt"))
    assert not (tmp_path / "out.ckpt").exists(), "a failed run must leave no output"


# --------------------------------------------------------------------------- #
# 4. fail-closed on the wrong file
# --------------------------------------------------------------------------- #
def test_a_checkpoint_without_EMA_keys_is_refused(tmp_path):
    """Every exp_19 arm trained with ``use_ema: true``.

    A checkpoint with no ``diffusion_ema.ema_model.*`` is therefore the WRONG
    FILE, not a checkpoint to fall back to the live weights on. Falling back
    would produce an arm initialised from online weights while every record says
    EMA.
    """
    sd = {k: v for k, v in _wrapped_state_dict().items() if not k.startswith("diffusion_ema.")}
    ckpt = _write_ckpt(tmp_path / "noema.ckpt", sd)
    with pytest.raises(KeyError, match="diffusion_ema.ema_model"):
        extract_ema_weights(str(ckpt), str(tmp_path / "out.ckpt"))
    assert not (tmp_path / "out.ckpt").exists()


def test_an_EMA_that_does_not_mirror_the_live_DiT_is_refused(tmp_path):
    """The substitution is only meaningful if the two key sets agree exactly.

    An EMA carrying a key the live model lacks (or missing one it has) means the
    checkpoint was not produced by the ``EMA(self.diffusion.model)`` wrapper this
    tool assumes — mixing them would emit a state dict that is neither.
    """
    sd = _wrapped_state_dict()
    sd[EMA_PREFIX + "blocks.99.weight"] = torch.zeros(2)
    ckpt = _write_ckpt(tmp_path / "skew.ckpt", sd)
    with pytest.raises(ValueError, match="blocks.99.weight"):
        extract_ema_weights(str(ckpt), str(tmp_path / "out.ckpt"))


def test_a_file_without_a_state_dict_is_refused(tmp_path):
    """A model-only export or a stray .pt is not a training checkpoint."""
    p = tmp_path / "bare.ckpt"
    torch.save({"weights": {"a": torch.zeros(1)}}, p)
    with pytest.raises(KeyError, match="state_dict"):
        extract_ema_weights(str(p), str(tmp_path / "out.ckpt"))


def test_an_unrecognised_top_level_family_is_refused(tmp_path):
    """Unknown key families fail closed rather than being silently dropped.

    ``losses.*`` is the one known non-``diffusion.`` family (it is dropped by
    train.py anyway and is registered as droppable). Anything else could be model
    weights, and discarding weights without saying so is how an init quietly
    becomes partial.
    """
    sd = _wrapped_state_dict()
    sd["mystery.module.weight"] = torch.zeros(3)
    ckpt = _write_ckpt(tmp_path / "mystery.ckpt", sd)
    with pytest.raises(ValueError, match="mystery.module.weight"):
        extract_ema_weights(str(ckpt), str(tmp_path / "out.ckpt"))


def test_the_known_losses_family_is_dropped_without_complaint(tmp_path):
    """Real 40k checkpoints carry a top-level ``losses.losses.0.weight``.

    It is a loss-module buffer, not a model weight; train.py drops it. If this
    raised, the tool could not run on any of our three inits.
    """
    sd = _wrapped_state_dict()
    sd["losses.losses.0.weight"] = torch.zeros(1)
    ckpt = _write_ckpt(tmp_path / "withlosses.ckpt", sd)
    out = tmp_path / "out.ckpt"
    extract_ema_weights(str(ckpt), str(out))
    assert set(_out_state_dict(out)) == EXPECTED_KEYS


# --------------------------------------------------------------------------- #
# 5. the CLI and what it reports
# --------------------------------------------------------------------------- #
def test_cli_success_returns_zero_and_prints_the_output_sha(wrapped, tmp_path, capsys):
    """The launcher pins the printed sha; it must be the sha of the file on disk."""
    out = tmp_path / "init.ckpt"
    rc = main(["--ckpt-path", str(wrapped), "--out", str(out)])
    assert rc == 0
    printed = capsys.readouterr().out
    assert _sha256(out) in printed, printed
    assert str(len(EMA_DIT)) in printed, "the extracted-tensor count must be reported"


@pytest.mark.parametrize(
    "case, why",
    [
        ("missing_input", "wrong path typed"),
        ("existing_output", "second arm about to overwrite the first"),
        ("no_ema", "wrong checkpoint file"),
    ],
)
def test_cli_refusals_exit_2_and_write_nothing(tmp_path, case, why):
    """Exit code 2 is what the launcher's ``set -e`` gate keys on."""
    out = tmp_path / "out.ckpt"
    if case == "missing_input":
        argv = ["--ckpt-path", str(tmp_path / "nope.ckpt"), "--out", str(out)]
    elif case == "existing_output":
        src = _write_ckpt(tmp_path / "in.ckpt", _wrapped_state_dict())
        out.write_bytes(b"occupied")
        argv = ["--ckpt-path", str(src), "--out", str(out)]
    else:
        sd = {k: v for k, v in _wrapped_state_dict().items()
              if not k.startswith("diffusion_ema.")}
        src = _write_ckpt(tmp_path / "noema.ckpt", sd)
        argv = ["--ckpt-path", str(src), "--out", str(out)]

    assert main(argv) == 2, why
    if case != "existing_output":
        assert not out.exists()


def test_the_output_is_reproducible_for_a_fixed_path(wrapped, tmp_path):
    """Re-running must reproduce the pinned sha, or the pin is not a pin.

    Scoped to a FIXED output path on purpose: ``torch.save`` prefixes every zip
    entry with the output file's basename, so the same tensors written to
    ``a.ckpt`` and ``b.ckpt`` hash differently. The tool documents this.
    """
    first = tmp_path / "init.ckpt"
    extract_ema_weights(str(wrapped), str(first))
    sha1 = _sha256(first)
    first.unlink()
    extract_ema_weights(str(wrapped), str(first))
    assert _sha256(first) == sha1


def test_the_module_runs_as_a_script(wrapped, tmp_path):
    """``python -m src.tools.extract_ema_weights`` is the documented invocation."""
    out = tmp_path / "init.ckpt"
    proc = subprocess.run(
        [sys.executable, "-m", "src.tools.extract_ema_weights",
         "--ckpt-path", str(wrapped), "--out", str(out)],
        cwd=str(_REPO), capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.is_file()
    assert _sha256(out) in proc.stdout


# --------------------------------------------------------------------------- #
# 6. the same algebra, on the real artifacts (skipped where they are absent)
# --------------------------------------------------------------------------- #
REAL_INITS = {
    "exp07_BF": _REPO / "outputs_FLAC/exp07_BF/FLAC_exp07_BF/exp07_BF/checkpoints/epoch=8-step=40000.ckpt",
    "exp07_P1": _REPO / "outputs_FLAC/exp07_P1/FLAC_exp07_P1/exp07_P1/checkpoints/epoch=8-step=40000.ckpt",
    "exp17_YAW": _REPO / "outputs_FLAC/exp17_YAWAUG_roteval/epoch=8-step=40000.ckpt",
}
RELEASED_EMA = _REPO / "weights/FLAC/FLAC_EMA.ckpt"


@pytest.mark.parametrize("arm", sorted(REAL_INITS))
def test_the_real_inits_produce_the_released_artifacts_key_set(arm):
    """The synthetic fixture asserts the algebra; this asserts the PREMISE.

    Namely: that a real 40k checkpoint decomposes the way the module docstring
    says (210 EMA / 856 carried / 3 dropped / 0 unknown) and that the resulting
    bare key set is byte-for-byte the released ``FLAC_EMA.ckpt``'s — the file the
    published HAA recipe finetunes from. Key sets only, memory-mapped: no tensor
    is materialised and nothing is written.

    Skipped rather than failed where the artifacts are absent, so the suite still
    runs on a fresh checkout.
    """
    ckpt_path = REAL_INITS[arm]
    if not ckpt_path.is_file() or not RELEASED_EMA.is_file():
        pytest.skip(f"artifact not present on this machine: {ckpt_path}")

    released = set(torch.load(RELEASED_EMA, map_location="cpu", mmap=True,
                              weights_only=True)["state_dict"])
    state_dict = torch.load(ckpt_path, map_location="cpu", mmap=True,
                            weights_only=True)["state_dict"]

    live, ema, dropped, unknown = set(), set(), [], []
    for key in state_dict:
        if key.startswith(EMA_PREFIX):
            ema.add(key[len(EMA_PREFIX):])
        elif key.startswith("diffusion_ema."):
            dropped.append(key)
        elif key.startswith(LIVE_PREFIX):
            bare = key[len(LIVE_PREFIX):]
            (dropped.append(key) if ("losses" in bare or "discriminator" in bare)
             else live.add(bare))
        elif "losses" in key or "discriminator" in key:
            dropped.append(key)
        else:
            unknown.append(key)

    assert unknown == [], unknown
    assert len(ema) == 210, len(ema)
    assert ema == {k[len(EMA_TARGET_PREFIX):] for k in live
                   if k.startswith(EMA_TARGET_PREFIX)}, "EMA does not mirror the live DiT"
    assert sorted(dropped) == ["diffusion_ema.initted", "diffusion_ema.step",
                               "losses.losses.0.weight"], sorted(dropped)
    assert live | {EMA_TARGET_PREFIX + t for t in ema} == released, (
        "the extracted key set differs from the released FLAC_EMA.ckpt's"
    )


def test_the_summary_it_returns_is_machine_readable(wrapped, tmp_path):
    """The launcher records the provenance; a dict beats scraping stdout."""
    out = tmp_path / "init.ckpt"
    summary = extract_ema_weights(str(wrapped), str(out))
    assert summary["n_ema"] == len(EMA_DIT)
    assert summary["n_total"] == len(EXPECTED_KEYS)
    assert summary["out_sha256"] == _sha256(out)
    assert summary["in_sha256"] == _sha256(wrapped)
    json.dumps(summary)          # must survive being written to a manifest
