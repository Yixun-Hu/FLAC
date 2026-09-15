"""Tests for the ``--max-steps`` training-budget flag on ``train.py`` (exp_07).

Motivation: ``train.py`` previously hard-coded ``max_steps=1000000`` inside the
``pl.Trainer(...)`` construction (with a ``#1000 for HAA`` comment), so enforcing a
pre-registered budget (67_500 optimizer steps) meant editing code. These tests pin the
contract of the flag *before* it exists (TDD RED). Two surfaces are under test:

* ``defaults.ini`` + prefigure plumbing — the ini key ``max_steps`` must surface as
  ``args.max_steps`` (an int) and default to 1_000_000, preserving the current behaviour and
  the CLAUDE.md ``#1000 for HAA`` convention (HAA overrides via ``--max-steps 1000``, never a
  code edit). The CLI flag ``--max-steps`` must override that default.
* ``train.build_trainer_kwargs`` — a pure, side-effect-free extraction of the exact keyword
  dict previously inlined into ``pl.Trainer``. It threads ``args.max_steps`` through to the
  trainer while reproducing every other kwarg unchanged (precision, accumulate_grad_batches,
  gradient_clip_val, devices/num_nodes/strategy/callbacks/logger/default_root_dir, the fixed
  log_every_n_steps=100 / num_sanity_val_steps=0 / reload_dataloaders_every_n_epochs=0, and
  the conditional val_check_interval merge). This is the non-regression guard.
* ``train.construct_trainer`` — the Trainer boundary (codex review, Medium finding): the tiny
  wrapper ``main()`` actually calls, pinned by monkeypatching ``pl.Trainer`` in the train
  module namespace so a revert that bypasses the helper or restores a hard-coded ``max_steps``
  literal fails loudly. A source-level regex revert-guard backs it up.

Importing ``train`` must have no side effects: ``get_all_args()`` is called inside ``main()``
(guarded by ``if __name__ == '__main__'``), never at import time.
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


# --------------------------------------------------------------------------------------
# 1. defaults.ini + prefigure: max_steps surfaces as args.max_steps (int), default 1e6
# --------------------------------------------------------------------------------------

def test_defaults_ini_max_steps_is_one_million(monkeypatch):
    """Default budget MUST stay 1_000_000 — guards the current behaviour and the CLAUDE.md
    ``#1000 for HAA`` convention (HAA overrides through the flag, never a code edit). A clean
    argv means prefigure applies only defaults.ini."""
    monkeypatch.setattr(sys, "argv", ["train.py"])
    args = get_all_args(defaults_file=_DEFAULTS_INI)
    assert args.max_steps == 1_000_000
    assert isinstance(args.max_steps, int)  # literal_eval -> int, not the string "1000000"


def test_cli_flag_overrides_max_steps(monkeypatch):
    """The pre-registered budget (67_500) reaches ``args`` via the ``--max-steps`` CLI flag
    (ini key ``max_steps`` -> flag ``--max-steps``, per prefigure's ``_`` -> ``-`` mapping)."""
    monkeypatch.setattr(sys, "argv", ["train.py", "--max-steps", "67500"])
    args = get_all_args(defaults_file=_DEFAULTS_INI)
    assert args.max_steps == 67_500
    assert isinstance(args.max_steps, int)


# --------------------------------------------------------------------------------------
# 2. build_trainer_kwargs: max_steps threaded through; every other kwarg reproduced
# --------------------------------------------------------------------------------------

def _stub_args(**overrides):
    """A minimal stand-in for the prefigure Namespace consumed by build_trainer_kwargs."""
    base = dict(
        num_gpus=1,
        num_nodes=1,
        precision="bf16-mixed",
        accum_batches=1,
        max_steps=1_000_000,
        gradient_clip_val=0.0,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def test_build_trainer_kwargs_default_max_steps():
    """With the default budget, the trainer kwargs carry max_steps == 1_000_000."""
    kw = train.build_trainer_kwargs(
        _stub_args(), strategy="auto", callbacks=[], logger=None,
        checkpoint_dir=None, val_args={},
    )
    assert kw["max_steps"] == 1_000_000


def test_build_trainer_kwargs_override_reaches_trainer():
    """A 67_500 override on args reaches the pl.Trainer kwargs dict."""
    kw = train.build_trainer_kwargs(
        _stub_args(max_steps=67_500), strategy="auto", callbacks=[], logger=None,
        checkpoint_dir=None, val_args={},
    )
    assert kw["max_steps"] == 67_500


def test_build_trainer_kwargs_reproduces_pre_change_literal():
    """Byte-for-byte non-regression pin of the kwargs that were previously inlined into
    ``pl.Trainer(...)``. Distinctive values (num_gpus=2, num_nodes=3, precision="16-mixed",
    accum_batches=4, gradient_clip_val=0.5) so a mis-wiring (e.g. swapping accum/precision, or
    dropping the fixed log_every_n_steps / num_sanity_val_steps / reload_dataloaders keys) is
    caught. This pins precision, accumulate_grad_batches wiring and gradient_clip_val
    passthrough (the important ones called out in the brief) at once."""
    args = _stub_args(
        num_gpus=2, num_nodes=3, precision="16-mixed",
        accum_batches=4, max_steps=1_000_000, gradient_clip_val=0.5,
    )
    logger = object()
    callbacks = ["ckpt", "exc", "cfg"]
    kw = train.build_trainer_kwargs(
        args,
        strategy="ddp_find_unused_parameters_true",
        callbacks=callbacks,
        logger=logger,
        checkpoint_dir="/some/dir",
        val_args={},
    )
    assert kw == {
        "devices": 2,
        "accelerator": "gpu",
        "num_nodes": 3,
        "strategy": "ddp_find_unused_parameters_true",
        "precision": "16-mixed",
        "accumulate_grad_batches": 4,
        "callbacks": callbacks,
        "logger": logger,
        "log_every_n_steps": 100,
        "max_steps": 1_000_000,
        "default_root_dir": "/some/dir",
        "gradient_clip_val": 0.5,
        "reload_dataloaders_every_n_epochs": 0,
        "num_sanity_val_steps": 0,
    }
    # Passthroughs are the same objects (not copies), matching the original inline wiring.
    assert kw["callbacks"] is callbacks
    assert kw["logger"] is logger


def test_build_trainer_kwargs_merges_val_args_when_present():
    """When val_every > 0 the caller passes a non-empty val_args; both keys must appear
    exactly as ``**val_args`` did in the original inline call."""
    val_args = {"check_val_every_n_epoch": None, "val_check_interval": 500}
    kw = train.build_trainer_kwargs(
        _stub_args(), strategy="auto", callbacks=[], logger=None,
        checkpoint_dir=None, val_args=val_args,
    )
    assert kw["check_val_every_n_epoch"] is None
    assert kw["val_check_interval"] == 500


def test_build_trainer_kwargs_omits_val_keys_when_val_args_empty():
    """Default path (val_every <= 0 -> empty val_args): the validation keys stay absent,
    matching the pre-change behaviour where ``**{}`` added nothing."""
    kw = train.build_trainer_kwargs(
        _stub_args(), strategy="auto", callbacks=[], logger=None,
        checkpoint_dir=None, val_args={},
    )
    assert "check_val_every_n_epoch" not in kw
    assert "val_check_interval" not in kw


# --------------------------------------------------------------------------------------
# 3. construct_trainer: the pl.Trainer boundary (codex review, Medium) + revert guard
# --------------------------------------------------------------------------------------

_TRAIN_PY = os.path.join(_REPO_ROOT, "train.py")


def test_construct_trainer_passes_override_to_pl_trainer(monkeypatch):
    """The Trainer-boundary pin: a stub replaces pl.Trainer in the train module namespace;
    construct_trainer with args.max_steps=67_500 must call it exactly once with
    kwargs['max_steps'] == 67_500 and return its instance. All build_trainer_kwargs-only
    tests would keep passing if main()'s construction bypassed the helper — this one
    exercises the wrapper main() actually calls."""
    calls = []
    dummy = object()

    def stub_trainer(**kwargs):
        calls.append(kwargs)
        return dummy

    monkeypatch.setattr(train.pl, "Trainer", stub_trainer)
    trainer = train.construct_trainer(
        _stub_args(max_steps=67_500), strategy="auto", callbacks=[], logger=None,
        checkpoint_dir=None, val_args={},
    )
    assert len(calls) == 1  # exactly one Trainer constructed
    assert calls[0]["max_steps"] == 67_500
    assert trainer is dummy  # the constructed Trainer is returned as-is


def test_revert_guard_main_uses_construct_trainer_and_no_hardcoded_literal():
    """Cheap source-level revert-guard (codex review): (a) main()'s body must contain a
    construct_trainer( call — the lookbehind rejects a def line, so moving the def after
    main() cannot satisfy this vacuously; (b) no numeric max_steps literal anywhere in
    train.py, in either keyword form (max_steps=1000000) or dict-key form
    ("max_steps": 1000000) — whitespace-tolerant, so a reformatted revert still bites.
    The correct wiring ("max_steps": args.max_steps) starts with 'a', not a digit."""
    with open(_TRAIN_PY) as f:
        source = f.read()

    main_body = re.search(r"def main\(\):(.*)", source, re.DOTALL)
    assert main_body is not None, "train.py must define main()"
    assert re.search(r"(?<!def )\bconstruct_trainer\s*\(", main_body.group(1)), (
        "main() must build the Trainer via construct_trainer(...)"
    )

    hardcode = re.search(r"""max_steps["']?\s*[=:]\s*\d""", source)
    assert hardcode is None, (
        f"hard-coded numeric max_steps found in train.py: {hardcode.group(0)!r}; "
        "the budget must come from args.max_steps (--max-steps / defaults.ini)"
    )


# --------------------------------------------------------------------------------------
# 4. Import safety
# --------------------------------------------------------------------------------------

def test_import_has_no_side_effects():
    """Importing train (done at module top) must not run training: main() is __main__-guarded
    and get_all_args() lives inside it, so the helper exists but nothing executed."""
    assert hasattr(train, "build_trainer_kwargs")
    assert callable(train.build_trainer_kwargs)
    assert hasattr(train, "construct_trainer")
    assert callable(train.construct_trainer)
    assert hasattr(train, "main")
