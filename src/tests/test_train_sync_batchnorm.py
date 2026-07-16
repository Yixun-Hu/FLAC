"""Tests for the ``--sync-batchnorm`` training flag on ``train.py``.

Motivation: multi-GPU BatchNorm layers need PyTorch Lightning's
``Trainer(sync_batchnorm=True)`` to synchronise running statistics across ranks. This
adds an opt-in ``--sync-batchnorm`` flag (ini key ``sync_batchnorm``, default ``false``)
that forwards straight to the Trainer, with a **fail-closed** guard: SyncBatchNorm is a
multi-GPU-only feature (Yixun mandate), so requesting it with ``num_gpus < 2`` is a hard
``ValueError`` rather than a silently-ignored no-op.

Surfaces under test:

* ``defaults.ini`` + prefigure plumbing — the ini key ``sync_batchnorm`` must surface as
  ``args.sync_batchnorm`` and default to a falsy value. NOTE the prefigure boolean
  convention in this repo: the lowercase ini literal ``false`` is NOT parsed to a real
  ``bool``. ``ast.literal_eval("false")`` raises (lowercase ``false``/``true`` are not
  Python literals), so prefigure registers the flag as ``type=str`` and yields the
  *string* ``"false"``; only the capitalized ``False``/``True`` would yield a real bool.
  The CLI override form is therefore ``--sync-batchnorm true`` / ``--sync-batchnorm
  false`` (a value is required; a bare ``--sync-batchnorm`` errors). ``train._as_bool``
  coerces whatever prefigure yields (str or bool) into a genuine ``bool`` so the trainer
  kwarg is always a real boolean.
* ``train.build_trainer_kwargs`` — must thread a *resolved bool* ``sync_batchnorm`` into
  the Trainer kwargs and enforce the fail-closed guard, without disturbing any other
  kwarg (``test_train_max_steps`` is the byte-for-byte non-regression canary).
* ``train.construct_trainer`` — the pl.Trainer boundary, pinned by monkeypatching
  ``pl.Trainer`` so the resolved bool actually reaches the constructed Trainer and the
  guard fires before any Trainer is built.

Importing ``train`` must have no side effects (``get_all_args()`` lives inside
``main()``, guarded by ``if __name__ == '__main__'``). The repo-root ``conftest.py``
prepends the checkout to ``sys.path`` so ``import train`` resolves here, not to a stale
``pip install .`` copy.
"""
import os
import re
import sys
import types

import pytest

import train
from prefigure.prefigure import get_all_args

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULTS_INI = os.path.join(_REPO_ROOT, "defaults.ini")
_TRAIN_PY = os.path.join(_REPO_ROOT, "train.py")


def _stub_args(**overrides):
    """Minimal stand-in for the prefigure Namespace consumed by build_trainer_kwargs.

    Defaults mirror a safe single-GPU run with sync off. ``sync_batchnorm`` defaults to
    the string ``"false"`` to match exactly what prefigure yields from the lowercase ini
    literal (see module docstring)."""
    base = dict(
        num_gpus=1,
        num_nodes=1,
        precision="bf16-mixed",
        accum_batches=1,
        max_steps=1_000_000,
        gradient_clip_val=0.0,
        sync_batchnorm="false",
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


# --------------------------------------------------------------------------------------
# 1. defaults.ini + prefigure: sync_batchnorm surfaces (falsy) as args.sync_batchnorm
# --------------------------------------------------------------------------------------

def test_defaults_ini_has_sync_batchnorm_key():
    """(e) ini-key presence: defaults.ini declares ``sync_batchnorm = false`` near the
    trainer keys, so every recipe inherits a safe (disabled) default."""
    with open(_DEFAULTS_INI) as f:
        ini = f.read()
    assert re.search(r"(?m)^\s*sync_batchnorm\s*=\s*false\s*$", ini), (
        "defaults.ini must declare `sync_batchnorm = false` (default off)"
    )


def test_defaults_ini_sync_batchnorm_defaults_false(monkeypatch):
    """A clean argv means prefigure applies only defaults.ini; the flag surfaces on args
    and, once coerced by train._as_bool, is exactly ``False``."""
    monkeypatch.setattr(sys, "argv", ["train.py"])
    args = get_all_args(defaults_file=_DEFAULTS_INI)
    assert hasattr(args, "sync_batchnorm")
    assert train._as_bool(args.sync_batchnorm) is False


def test_prefigure_yields_string_false_for_lowercase_ini(monkeypatch):
    """(f) prefigure bool-parsing sanity: document + pin the convention. The lowercase ini
    literal ``false`` is NOT a Python literal, so prefigure keeps it as the *string*
    ``"false"`` (flag registered ``type=str``). This is exactly why ``_as_bool`` exists,
    and it must always yield a real ``bool``."""
    monkeypatch.setattr(sys, "argv", ["train.py"])
    args = get_all_args(defaults_file=_DEFAULTS_INI)
    # Document the actual convention: prefigure yields the *string* "false" here.
    assert args.sync_batchnorm == "false"
    assert isinstance(args.sync_batchnorm, str)
    # Whatever prefigure yields, _as_bool must turn it into a genuine bool.
    assert isinstance(train._as_bool(args.sync_batchnorm), bool)


def test_cli_flag_overrides_sync_batchnorm(monkeypatch):
    """CLI override form: ``--sync-batchnorm true`` (ini key ``sync_batchnorm`` -> flag
    ``--sync-batchnorm`` per prefigure's ``_`` -> ``-`` mapping; a value is required)."""
    monkeypatch.setattr(sys, "argv", ["train.py", "--sync-batchnorm", "true"])
    args = get_all_args(defaults_file=_DEFAULTS_INI)
    assert train._as_bool(args.sync_batchnorm) is True


# --------------------------------------------------------------------------------------
# 2. _as_bool coercion: str/bool -> genuine bool (the prefigure-convention shim)
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (False, False), (True, True),           # already a real bool
    ("false", False), ("true", True),       # prefigure's lowercase-ini / CLI form
    ("False", False), ("True", True),       # capitalized CLI string
    ("0", False), ("1", True),              # numeric string
])
def test_as_bool_coerces_to_real_bool(value, expected):
    out = train._as_bool(value)
    assert out is expected              # identity: returns the bool singletons
    assert isinstance(out, bool)


@pytest.mark.parametrize("bad", [1, 0, None, 1.0])
def test_as_bool_rejects_unsupported_types(bad):
    """Review hardening: only bool/str are legal carriers for the flag; anything else is
    a wiring bug and must fail fast rather than be coerced."""
    with pytest.raises(TypeError):
        train._as_bool(bad)


# --------------------------------------------------------------------------------------
# 3. build_trainer_kwargs: resolved bool threaded through; fail-closed guard
# --------------------------------------------------------------------------------------

def test_build_trainer_kwargs_default_false():
    """(a) default false -> the key is ABSENT: the kwargs are byte-identical to the
    pre-change dict (PL's own default False applies), so existing recipes are untouched."""
    kw = train.build_trainer_kwargs(
        _stub_args(), strategy="auto", callbacks=[], logger=None,
        checkpoint_dir=None, val_args={},
    )
    assert "sync_batchnorm" not in kw


def test_build_trainer_kwargs_true_multi_gpu():
    """(b) true + num_gpus 2 -> kwargs carry sync_batchnorm == True (a real bool)."""
    kw = train.build_trainer_kwargs(
        _stub_args(sync_batchnorm="true", num_gpus=2), strategy="ddp_find_unused_parameters_true",
        callbacks=[], logger=None, checkpoint_dir=None, val_args={},
    )
    assert kw["sync_batchnorm"] is True
    assert isinstance(kw["sync_batchnorm"], bool)


def test_build_trainer_kwargs_true_bool_multi_gpu():
    """A real-bool ``True`` (capitalized ini / programmatic) is honoured too."""
    kw = train.build_trainer_kwargs(
        _stub_args(sync_batchnorm=True, num_gpus=4), strategy="ddp_find_unused_parameters_true",
        callbacks=[], logger=None, checkpoint_dir=None, val_args={},
    )
    assert kw["sync_batchnorm"] is True


@pytest.mark.parametrize("truthy", ["true", "True", True, "1"])
def test_build_trainer_kwargs_fail_closed_single_gpu(truthy):
    """(c) true + num_gpus 1 -> ValueError (fail-closed; SyncBatchNorm is multi-GPU only).

    Covers every truthy spelling that can reach the guard (prefigure string, capitalized
    string, real bool, numeric string) so no path silently enables it on one device."""
    with pytest.raises(ValueError, match=r"(?i)sync.?batchnorm"):
        train.build_trainer_kwargs(
            _stub_args(sync_batchnorm=truthy, num_gpus=1), strategy="auto", callbacks=[],
            logger=None, checkpoint_dir=None, val_args={},
        )


def test_build_trainer_kwargs_false_single_gpu_ok():
    """The guard only bites when enabled: sync off on a single GPU stays valid (no raise),
    key absent, so every existing single-GPU recipe is unaffected."""
    kw = train.build_trainer_kwargs(
        _stub_args(sync_batchnorm="false", num_gpus=1), strategy="auto", callbacks=[],
        logger=None, checkpoint_dir=None, val_args={},
    )
    assert "sync_batchnorm" not in kw


def test_build_trainer_kwargs_adds_only_sync_batchnorm_key():
    """Minimal-diff pin on the ENABLED path: turning the flag on adds ``sync_batchnorm``
    and nothing else -- every other Trainer kwarg is left exactly as the max-steps
    canary pins it."""
    kw = train.build_trainer_kwargs(
        _stub_args(sync_batchnorm="true", num_gpus=2), strategy="ddp_find_unused_parameters_true",
        callbacks=[], logger=None, checkpoint_dir=None, val_args={},
    )
    assert kw["sync_batchnorm"] is True
    assert set(kw) - {"sync_batchnorm"} == {
        "devices", "accelerator", "num_nodes", "strategy", "precision",
        "accumulate_grad_batches", "callbacks", "logger", "log_every_n_steps",
        "max_steps", "default_root_dir", "gradient_clip_val",
        "reload_dataloaders_every_n_epochs", "num_sanity_val_steps",
    }


def test_build_trainer_kwargs_val_args_cannot_smuggle_sync_batchnorm():
    """Regression (review finding): ``**val_args`` is merged into the kwargs, so without
    a guard a direct caller could smuggle ``sync_batchnorm`` past the fail-closed check.
    It must be rejected regardless of the args-level flag."""
    with pytest.raises(ValueError, match=r"(?i)val_args"):
        train.build_trainer_kwargs(
            _stub_args(num_gpus=1), strategy="auto", callbacks=[], logger=None,
            checkpoint_dir=None, val_args={"sync_batchnorm": True},
        )


# --------------------------------------------------------------------------------------
# 4. construct_trainer: the pl.Trainer boundary (resolved bool reaches Trainer)
# --------------------------------------------------------------------------------------

def test_construct_trainer_passes_sync_batchnorm_to_pl_trainer(monkeypatch):
    """(d) Trainer-boundary pin: a stub replaces pl.Trainer in the train module namespace;
    construct_trainer with sync_batchnorm true + num_gpus 2 must call it exactly once with
    ``kwargs['sync_batchnorm'] is True`` (a real bool) and return its instance. A revert
    that bypasses the helper or forwards the raw string would fail here."""
    calls = []
    dummy = object()

    def stub_trainer(**kwargs):
        calls.append(kwargs)
        return dummy

    monkeypatch.setattr(train.pl, "Trainer", stub_trainer)
    trainer = train.construct_trainer(
        _stub_args(sync_batchnorm="true", num_gpus=2), strategy="ddp_find_unused_parameters_true",
        callbacks=[], logger=None, checkpoint_dir=None, val_args={},
    )
    assert len(calls) == 1                       # exactly one Trainer constructed
    assert calls[0]["sync_batchnorm"] is True    # resolved bool, not the raw "true" string
    assert isinstance(calls[0]["sync_batchnorm"], bool)
    assert trainer is dummy


def test_construct_trainer_fail_closed_before_building(monkeypatch):
    """The fail-closed guard fires *before* any Trainer is constructed: construct_trainer
    with sync on a single GPU raises and never touches pl.Trainer."""
    calls = []
    monkeypatch.setattr(train.pl, "Trainer", lambda **kw: calls.append(kw) or object())
    with pytest.raises(ValueError, match=r"(?i)sync.?batchnorm"):
        train.construct_trainer(
            _stub_args(sync_batchnorm="true", num_gpus=1), strategy="auto", callbacks=[],
            logger=None, checkpoint_dir=None, val_args={},
        )
    assert calls == []  # fail-closed: no Trainer built


# --------------------------------------------------------------------------------------
# 5. Source-level revert guard
# --------------------------------------------------------------------------------------

def test_revert_guard_source_wires_sync_batchnorm_and_guard():
    """Cheap source-level revert-guard: train.py must (a) forward a ``sync_batchnorm``
    kwarg into the Trainer kwargs dict and (b) keep a fail-closed ``ValueError`` guard on
    ``num_gpus < 2`` so a revert that drops either bites."""
    with open(_TRAIN_PY) as f:
        source = f.read()
    assert re.search(r"""kwargs\[["']sync_batchnorm["']\]\s*=\s*True""", source), (
        "build_trainer_kwargs must conditionally add the 'sync_batchnorm' kwarg when enabled"
    )
    assert re.search(r"num_gpus\s*<\s*2", source), (
        "a fail-closed guard on num_gpus < 2 must exist"
    )
    assert "raise ValueError" in source


# --------------------------------------------------------------------------------------
# 6. Import safety
# --------------------------------------------------------------------------------------

def test_import_has_no_side_effects():
    """Importing train (done at module top) must not run training: the helpers exist and
    nothing executed."""
    assert callable(train._as_bool)
    assert callable(train.build_trainer_kwargs)
    assert callable(train.construct_trainer)
    assert callable(train.main)
