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
keys, because ``export_model`` (``src/training/diffusion.py:911``) assigns the EMA
weights *into* ``diffusion.model`` and then saves the whole wrapper. So the
contract pinned here is the released one: **every bare key, with the ``model.*``
subtree taken from the EMA copy and the rest carried from the live weights**.

**Codex r1 fixes covered here.** The round trip is no longer self-fulfilling: it
strict-loads into a model built by ``create_model_from_config`` from the STOCK HAA
config — an independent target with its own shapes and dtypes — rather than into a
module generated from the extracted weights' own keys (finding 1). Dtype- and
shape-drifted EMA tensors have explicit mutation guards (finding 2), the atomic
publication has a concurrent-writer guard (finding 3), and the real-artifact test
now compares shapes, dtypes and a sample of VALUES rather than key sets alone.

Written by the exp_19 coder seat (Claude Opus 5, max effort).
"""
import hashlib
import json
import os

# Must precede any import that pulls in huggingface_hub: the real-model fixtures
# below build DINOv3 from the LOCAL cache and must never reach the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import subprocess                                                    # noqa: E402
import sys                                                           # noqa: E402
from pathlib import Path                                             # noqa: E402

import pytest                                                        # noqa: E402
import torch                                                         # noqa: E402

from src.models.utils import load_ckpt_state_dict                     # noqa: E402
from src.tools.extract_ema_weights import (                           # noqa: E402
    EMA_PREFIX,
    EMA_TARGET_PREFIX,
    LIVE_PREFIX,
    extract_ema_weights,
    main,
)


_REPO = Path(__file__).resolve().parents[2]
STOCK_HAA_CONFIG = _REPO / "src/configs/model_configs/FLAC/HAA/FLAC_HAA_finetune.json"


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


# --------------------------------------------------------------------------- #
# the REAL model — the independent load target (Codex r1 finding 1)
# --------------------------------------------------------------------------- #
def build_real_flac_model():
    """``create_model_from_config`` on the stock HAA config, or skip.

    This is the actual thing ``train.py`` strict-loads into, with its own key set,
    shapes and dtypes. Building a module out of the extracted weights' own keys —
    which is what this file used to do — cannot fail for a wrong shape or a wrong
    dtype, because it copies both from the very object under test.

    DINOv3 is read from the local HF cache (``HF_HUB_OFFLINE=1``, set at import).
    A machine without that cache SKIPS rather than fakes a target.
    """
    try:
        from src.models.factory import create_model_from_config
        config = json.loads(STOCK_HAA_CONFIG.read_text())
        return create_model_from_config(config)
    except Exception as e:                                  # noqa: BLE001
        pytest.skip(f"cannot construct the real FLAC model here "
                    f"({type(e).__name__}: {e}); HF cache missing?")


@pytest.fixture(scope="module")
def real_model_spec():
    """``{key: (shape, dtype)}`` of the real model — cheap to reuse, safe to share."""
    model = build_real_flac_model()
    return {k: (tuple(v.shape), v.dtype) for k, v in model.state_dict().items()}


@pytest.fixture(scope="module")
def real_wrapped_ckpt(real_model_spec, tmp_path_factory):
    """A wrapped PL checkpoint with the REAL model's topology.

    Live weights are zeros, EMA weights are ones, so "did you take the EMA?" is
    answerable by inspection after the strict load. Dtypes and shapes come from
    the real model, so the extracted file must satisfy a target it did not define.
    """
    sd = {}
    for k, (shape, dtype) in real_model_spec.items():
        sd[LIVE_PREFIX + k] = torch.zeros(shape, dtype=dtype)
        if k.startswith(EMA_TARGET_PREFIX):
            sd[EMA_PREFIX + k[len(EMA_TARGET_PREFIX):]] = torch.ones(shape, dtype=dtype)
    sd["diffusion_ema.initted"] = torch.tensor(True)
    sd["diffusion_ema.step"] = torch.tensor(40000)
    sd["losses.losses.0.weight"] = torch.zeros(1)
    path = tmp_path_factory.mktemp("realckpt") / "epoch=8-step=40000.ckpt"
    return _write_ckpt(path, sd)


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
    mangled on load and then fail the strict load with a confusing name. Now an
    enforced invariant of the tool, not merely an observation.
    """
    out = tmp_path / "init.ckpt"
    extract_ema_weights(str(wrapped), str(out))
    for key in _out_state_dict(out):
        assert "diffusion." not in key, key
        assert "autoencoder." not in key, key


# --------------------------------------------------------------------------- #
# 2. the round trip that actually matters — into the REAL model
# --------------------------------------------------------------------------- #
def test_the_extracted_init_strict_loads_into_the_real_model(real_wrapped_ckpt, tmp_path):
    """``load_ckpt_state_dict`` -> train.py:142-147 -> ``load_state_dict(strict=True)``.

    The load target is built by ``create_model_from_config`` from the stock HAA
    config — the same call ``train.py:137`` makes — so a wrong key, a wrong shape
    or a missing subtree is a failure here, not a passing tautology. The values
    are then read BACK out of the model to prove the EMA copy (ones), not the live
    copy (zeros), is what ended up in the DiT.
    """
    model = build_real_flac_model()
    out = tmp_path / "init.ckpt"
    summary = extract_ema_weights(str(real_wrapped_ckpt), str(out))

    weights = train_py_load_transforms(load_ckpt_state_dict(str(out)))
    model.load_state_dict(weights, strict=True)                       # train.py:148

    loaded = model.state_dict()
    dit = [k for k in loaded if k.startswith(EMA_TARGET_PREFIX)]
    carried = [k for k in loaded if not k.startswith(EMA_TARGET_PREFIX)]
    assert len(dit) == summary["n_ema"] and len(carried) == summary["n_carried"]
    for k in dit:
        assert torch.equal(loaded[k], torch.ones_like(loaded[k])), f"{k} is not the EMA copy"
    for k in carried:
        assert torch.equal(loaded[k], torch.zeros_like(loaded[k])), f"{k} is not the carried copy"


def test_the_real_target_would_reject_a_short_init(real_wrapped_ckpt, tmp_path):
    """Non-vacuity for the test above: the real target must be able to FAIL.

    Dropping the conditioner subtree from an otherwise valid init has to raise —
    which is precisely what a module generated from the init's own keys could
    never do (Codex r1 finding 1).
    """
    model = build_real_flac_model()
    out = tmp_path / "init.ckpt"
    extract_ema_weights(str(real_wrapped_ckpt), str(out))
    weights = train_py_load_transforms(load_ckpt_state_dict(str(out)))
    short = {k: v for k, v in weights.items() if not k.startswith("conditioner.")}
    assert 0 < len(short) < len(weights)

    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(short, strict=True)


def test_an_EMA_only_file_would_fail_the_strict_load(real_wrapped_ckpt, tmp_path):
    """Why the tool does not emit "just the EMA keys" (the round-1 brief's reading).

    Executed against the real model: 856 of 1066 keys missing. Keeping it as a
    test means the design decision is re-derived on every run.
    """
    model = build_real_flac_model()
    out = tmp_path / "init.ckpt"
    extract_ema_weights(str(real_wrapped_ckpt), str(out))
    full = train_py_load_transforms(load_ckpt_state_dict(str(out)))
    ema_only = {k: v for k, v in full.items() if k.startswith(EMA_TARGET_PREFIX)}
    assert 0 < len(ema_only) < len(full)

    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(ema_only, strict=True)


# --------------------------------------------------------------------------- #
# 3. copy-only, and the publication is atomic
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


def test_an_output_that_appears_DURING_the_write_is_refused(wrapped, tmp_path, monkeypatch):
    """The TOCTOU window the up-front existence check cannot close (r1 finding 3).

    ``torch.save`` is wrapped so that a competing writer creates the destination
    after the check has passed and while the payload is being written. The
    publication must fail rather than replace, the competitor's bytes must survive
    intact, and no partial output or stray temp file may be left behind.
    """
    out = tmp_path / "init.ckpt"
    import src.tools.extract_ema_weights as mod
    real_save = torch.save

    def racing_save(obj, f, *a, **kw):
        real_save(obj, f, *a, **kw)
        out.write_bytes(b"the other writer got here first")   # concurrent publisher

    monkeypatch.setattr(mod.torch, "save", racing_save)

    with pytest.raises(FileExistsError, match="appeared while this run was writing"):
        extract_ema_weights(str(wrapped), str(out))

    assert out.read_bytes() == b"the other writer got here first", "our payload replaced theirs"
    leftovers = list(tmp_path.glob(".extract_ema_*"))
    assert leftovers == [], f"temp files left behind: {leftovers}"


def test_the_publication_is_a_no_replace_link_not_a_write(wrapped, tmp_path):
    """Non-vacuity for the guard above: prove the atomic step is what publishes.

    If ``os.link`` were replaced by a plain write, this raises nothing.
    """
    import src.tools.extract_ema_weights as mod
    calls = []
    real_link = os.link

    def counting_link(src, dst, *a, **kw):
        calls.append((src, dst))
        return real_link(src, dst, *a, **kw)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(mod.os, "link", counting_link)
    try:
        extract_ema_weights(str(wrapped), str(tmp_path / "init.ckpt"))
    finally:
        monkeypatch.undo()
    assert len(calls) == 1 and calls[0][1] == str(tmp_path / "init.ckpt")


def test_a_missing_input_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_ema_weights(str(tmp_path / "nope.ckpt"), str(tmp_path / "out.ckpt"))
    assert not (tmp_path / "out.ckpt").exists(), "a failed run must leave no output"


# --------------------------------------------------------------------------- #
# 4. fail-closed on the wrong file, the wrong tensors, the wrong namespaces
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
    """The substitution is only meaningful if the two key sets agree exactly."""
    sd = _wrapped_state_dict()
    sd[EMA_PREFIX + "blocks.99.weight"] = torch.zeros(2)
    ckpt = _write_ckpt(tmp_path / "skew.ckpt", sd)
    with pytest.raises(ValueError, match="blocks.99.weight"):
        extract_ema_weights(str(ckpt), str(tmp_path / "out.ckpt"))


def test_a_dtype_drifted_EMA_tensor_is_refused(tmp_path):
    """r1 finding 2 — the failure a key/shape check cannot see.

    ``load_state_dict`` CASTS the source into the target's dtype, so a bf16 EMA
    entry loads without a word and the arm silently starts from rounded weights.
    """
    sd = _wrapped_state_dict()
    key = EMA_PREFIX + "blocks.0.weight"
    sd[key] = sd[key].to(torch.bfloat16)
    ckpt = _write_ckpt(tmp_path / "dtype.ckpt", sd)
    with pytest.raises(ValueError, match="dtype"):
        extract_ema_weights(str(ckpt), str(tmp_path / "out.ckpt"))
    assert not (tmp_path / "out.ckpt").exists()


def test_a_shape_drifted_EMA_tensor_is_refused(tmp_path):
    """Same guard, the shape axis: an EMA that does not shadow THIS DiT."""
    sd = _wrapped_state_dict()
    sd[EMA_PREFIX + "blocks.0.weight"] = torch.zeros(3, 4)
    ckpt = _write_ckpt(tmp_path / "shape.ckpt", sd)
    with pytest.raises(ValueError, match="shape"):
        extract_ema_weights(str(ckpt), str(tmp_path / "out.ckpt"))
    assert not (tmp_path / "out.ckpt").exists()


def test_the_dtype_guard_is_not_vacuous(tmp_path):
    """The same fixture WITHOUT the drift must extract cleanly.

    Otherwise the two tests above would pass for any reason at all.
    """
    ckpt = _write_ckpt(tmp_path / "clean.ckpt", _wrapped_state_dict())
    out = tmp_path / "out.ckpt"
    extract_ema_weights(str(ckpt), str(out))
    assert set(_out_state_dict(out)) == EXPECTED_KEYS


def test_a_file_without_a_state_dict_is_refused(tmp_path):
    """A model-only export or a stray .pt is not a training checkpoint."""
    p = tmp_path / "bare.ckpt"
    torch.save({"weights": {"a": torch.zeros(1)}}, p)
    with pytest.raises(KeyError, match="state_dict"):
        extract_ema_weights(str(p), str(tmp_path / "out.ckpt"))


def test_an_unrecognised_top_level_family_is_refused(tmp_path):
    """Unknown key families fail closed rather than being silently dropped."""
    sd = _wrapped_state_dict()
    sd["mystery.module.weight"] = torch.zeros(3)
    ckpt = _write_ckpt(tmp_path / "mystery.ckpt", sd)
    with pytest.raises(ValueError, match="mystery.module.weight"):
        extract_ema_weights(str(ckpt), str(tmp_path / "out.ckpt"))


def test_the_known_losses_family_is_dropped_without_complaint(tmp_path):
    """Real 40k checkpoints carry a top-level ``losses.losses.0.weight``."""
    sd = _wrapped_state_dict()
    sd["losses.losses.0.weight"] = torch.zeros(1)
    ckpt = _write_ckpt(tmp_path / "withlosses.ckpt", sd)
    out = tmp_path / "out.ckpt"
    extract_ema_weights(str(ckpt), str(out))
    assert set(_out_state_dict(out)) == EXPECTED_KEYS


def test_a_weight_whose_NAME_merely_contains_losses_is_not_silently_dropped(tmp_path):
    """r1 non-blocking finding: dropping is namespace-scoped, not substring-scoped.

    ``conditioner.…losses_proj.weight`` is a model weight, not a loss module, so
    the tool no longer discards it. It cannot be shipped either — train.py's OWN
    substring filter (line 147) would drop it on load and the strict load would
    then fail with a confusing name — so the tool refuses HERE, naming the key
    and the line, instead of writing an init that quietly cannot be loaded.
    """
    sd = _wrapped_state_dict()
    sd[LIVE_PREFIX + "conditioner.losses_proj.weight"] = torch.zeros(2)
    ckpt = _write_ckpt(tmp_path / "namelike.ckpt", sd)
    with pytest.raises(ValueError, match="conditioner.losses_proj.weight"):
        extract_ema_weights(str(ckpt), str(tmp_path / "out.ckpt"))


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
        ("dtype_drift", "an EMA that would be cast on load"),
    ],
)
def test_cli_refusals_exit_2_and_write_nothing(tmp_path, case, why):
    """Exit code 2 is what the launcher's gate keys on."""
    out = tmp_path / "out.ckpt"
    if case == "missing_input":
        argv = ["--ckpt-path", str(tmp_path / "nope.ckpt"), "--out", str(out)]
    elif case == "existing_output":
        src = _write_ckpt(tmp_path / "in.ckpt", _wrapped_state_dict())
        out.write_bytes(b"occupied")
        argv = ["--ckpt-path", str(src), "--out", str(out)]
    elif case == "dtype_drift":
        sd = _wrapped_state_dict()
        k = EMA_PREFIX + "blocks.0.bias"
        sd[k] = sd[k].to(torch.float64)
        src = _write_ckpt(tmp_path / "dtype.ckpt", sd)
        argv = ["--ckpt-path", str(src), "--out", str(out)]
    else:
        sd = {k: v for k, v in _wrapped_state_dict().items()
              if not k.startswith("diffusion_ema.")}
        src = _write_ckpt(tmp_path / "noema.ckpt", sd)
        argv = ["--ckpt-path", str(src), "--out", str(out)]

    assert main(argv) == 2, why
    if case != "existing_output":
        assert not out.exists()


def test_the_output_sha_depends_on_content_only_not_on_the_filename(wrapped, tmp_path):
    """Re-running must reproduce the pinned sha, or the pin is not a pin.

    Since the payload is serialised through a FILE OBJECT, ``torch.save`` writes
    its fixed ``archive/`` zip prefix rather than one derived from the output
    basename — so the same tensors hash the same under any name, and moving an
    init does not invalidate its manifest line.
    """
    a = tmp_path / "init.ckpt"
    b = tmp_path / "renamed_init.ckpt"
    extract_ema_weights(str(wrapped), str(a))
    extract_ema_weights(str(wrapped), str(b))
    assert _sha256(a) == _sha256(b)


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
def test_the_real_inits_match_the_released_artifact_in_keys_shapes_and_dtypes(arm):
    """The synthetic fixtures assert the algebra; this asserts the PREMISE.

    A real 40k checkpoint must decompose the way the module docstring says
    (210 EMA / 856 carried / 3 dropped / 0 unknown), and the resulting bare
    entries must agree with the released ``FLAC_EMA.ckpt`` — the file the
    published HAA recipe finetunes from — in KEY SET, SHAPE and DTYPE, not merely
    in names (Codex r1 finding 1). Memory-mapped: nothing is written.

    Skipped rather than failed where the artifacts are absent, so the suite still
    runs on a fresh checkout.
    """
    ckpt_path = REAL_INITS[arm]
    if not ckpt_path.is_file() or not RELEASED_EMA.is_file():
        pytest.skip(f"artifact not present on this machine: {ckpt_path}")

    released = torch.load(RELEASED_EMA, map_location="cpu", mmap=True,
                          weights_only=True)["state_dict"]
    state_dict = torch.load(ckpt_path, map_location="cpu", mmap=True,
                            weights_only=True)["state_dict"]

    live, ema, dropped, unknown = {}, {}, [], []
    for key, value in state_dict.items():
        if key.startswith(EMA_PREFIX):
            ema[key[len(EMA_PREFIX):]] = value
        elif key.startswith("diffusion_ema."):
            dropped.append(key)
        elif key.startswith(LIVE_PREFIX):
            bare = key[len(LIVE_PREFIX):]
            (dropped.append(key) if bare.split(".", 1)[0] in ("losses", "discriminator")
             else live.__setitem__(bare, value))
        elif key.split(".", 1)[0] in ("losses", "discriminator"):
            dropped.append(key)
        else:
            unknown.append(key)

    assert unknown == [], unknown
    assert len(ema) == 210, len(ema)
    assert sorted(dropped) == ["diffusion_ema.initted", "diffusion_ema.step",
                               "losses.losses.0.weight"], sorted(dropped)

    extracted = dict(live)
    for tail, value in ema.items():
        target = EMA_TARGET_PREFIX + tail
        assert target in live, target
        assert tuple(value.shape) == tuple(live[target].shape), target
        assert value.dtype == live[target].dtype, target
        extracted[target] = value

    assert set(extracted) == set(released), "key set differs from the released artifact"
    for key in extracted:
        assert tuple(extracted[key].shape) == tuple(released[key].shape), f"{key} shape"
        assert extracted[key].dtype == released[key].dtype, f"{key} dtype"

    # Values: the EMA subtree must be the EMA copy, and the carried subtree the
    # live one. Sampled — comparing 1066 real tensors would read ~1 GB per arm.
    sample = sorted(k for k in extracted if k.startswith(EMA_TARGET_PREFIX))[:5]
    assert sample
    for key in sample:
        tail = key[len(EMA_TARGET_PREFIX):]
        assert torch.equal(extracted[key], ema[tail]), f"{key} is not the EMA tensor"
    carried_sample = sorted(k for k in extracted if k.startswith("conditioner."))[:5]
    for key in carried_sample:
        assert torch.equal(extracted[key], live[key]), f"{key} is not the live tensor"


def test_the_summary_it_returns_is_machine_readable(wrapped, tmp_path):
    """The launcher records the provenance; a dict beats scraping stdout."""
    out = tmp_path / "init.ckpt"
    summary = extract_ema_weights(str(wrapped), str(out))
    assert summary["n_ema"] == len(EMA_DIT)
    assert summary["n_total"] == len(EXPECTED_KEYS)
    assert summary["out_sha256"] == _sha256(out)
    assert summary["in_sha256"] == _sha256(wrapped)
    json.dumps(summary)          # must survive being written to a manifest
