"""Tests for ``src/tools/strip_optimizer_state.py`` (exp_09, F-reset arm).

TDD RED-first. The hook produces a *copy* of a PyTorch-Lightning training
checkpoint with the Adam optimizer state removed, so that a ``--ckpt-path``
resume re-initialises the optimizer from scratch (fresh moments, per-param step
zeroed) while retaining every other resume-critical piece of state: model +
EMA weights (``state_dict``), loop counters / ``global_step`` / ``epoch``, the
LR-scheduler position (``lr_schedulers``) and the callback states.

Contract pinned here:

* ``optimizer_states`` is **emptied in place** (``[]``), *not* deleted — PL 2.1
  raises ``KeyError`` when the key is absent (see the module docstring of the
  hook for the source reference). Everything else is byte-identical.
* fail-closed on a missing ``--in`` file (non-zero exit, no output written);
* fail-closed when ``--in`` and ``--out`` resolve to the same path (the anchor
  checkpoint must never be mutated);
* idempotent: stripping an already-stripped checkpoint is a no-op.

Synthetic checkpoint dicts only — the real anchor is ~700 MB and is never
touched by the test-suite.
"""
import os
import subprocess
import sys

import pytest
import torch

import src.tools.strip_optimizer_state as sos

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _fake_ckpt():
    """A miniature PL-2.1-shaped training checkpoint (same top-level keys as the anchor)."""
    return {
        "epoch": 19,
        "global_step": 87500,
        "pytorch-lightning_version": "2.1.0",
        "state_dict": {
            "diffusion.model.w": torch.ones(4),
            "diffusion_ema.initted": torch.tensor(True),
            "diffusion_ema.step": torch.tensor(87500),
            "diffusion_ema.ema_model.model.w": torch.full((4,), 0.5),
        },
        "loops": {"fit_loop": {"epoch_loop.batch_progress": {"total": {"completed": 87500}}}},
        "callbacks": {"ModelCheckpoint{'monitor': None}": {"best_model_score": None}},
        "optimizer_states": [
            {
                "state": {0: {"step": torch.tensor(87500.0),
                              "exp_avg": torch.ones(4),
                              "exp_avg_sq": torch.ones(4)}},
                "param_groups": [{"lr": 4.7946e-05, "betas": (0.9, 0.999),
                                  "initial_lr": 5e-05, "params": [0]}],
            }
        ],
        "lr_schedulers": [
            {"inv_gamma": 1000000, "power": 0.5, "warmup": 0.99, "final_lr": 0.0,
             "base_lrs": [5e-05], "last_epoch": 87500, "_step_count": 87501,
             "_last_lr": [4.7946e-05]}
        ],
        "model_config": {"model_type": "diffusion_cond"},
    }


def _write(tmp_path, name="anchor.ckpt"):
    p = tmp_path / name
    torch.save(_fake_ckpt(), p)
    return str(p)


def _assert_same(a, b, path="ckpt"):
    """Recursive structural equality that understands tensors."""
    assert type(a) is type(b), f"{path}: type {type(a)} != {type(b)}"
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), f"{path}: tensor mismatch"
    elif isinstance(a, dict):
        assert list(a.keys()) == list(b.keys()), f"{path}: key mismatch"
        for k in a:
            _assert_same(a[k], b[k], f"{path}[{k!r}]")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{path}: length mismatch"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_same(x, y, f"{path}[{i}]")
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


# ---------------------------------------------------------------------------------------
# (a) preserves everything except the optimizer internals
# ---------------------------------------------------------------------------------------

def test_strip_preserves_all_but_optimizer_state(tmp_path):
    src_p = _write(tmp_path)
    out_p = str(tmp_path / "stripped.ckpt")

    sos.strip_optimizer_state(src_p, out_p)

    orig = _fake_ckpt()
    out = torch.load(out_p, map_location="cpu", weights_only=False)

    # the key must still be PRESENT (PL 2.1 KeyErrors when it is absent) but empty
    assert "optimizer_states" in out
    assert out["optimizer_states"] == []

    # everything else identical, in the same key order
    assert list(out.keys()) == list(orig.keys())
    for k in orig:
        if k == "optimizer_states":
            continue
        _assert_same(out[k], orig[k], f"ckpt[{k!r}]")

    # explicit spot-checks of the resume-critical fields
    assert out["global_step"] == 87500
    assert out["epoch"] == 19
    assert out["lr_schedulers"][0]["last_epoch"] == 87500
    assert out["lr_schedulers"][0]["_step_count"] == 87501
    assert int(out["state_dict"]["diffusion_ema.step"]) == 87500
    assert torch.equal(out["state_dict"]["diffusion_ema.ema_model.model.w"], torch.full((4,), 0.5))
    assert out["callbacks"] == orig["callbacks"]
    assert out["loops"] == orig["loops"]


def test_input_file_is_not_modified(tmp_path):
    src_p = _write(tmp_path)
    out_p = str(tmp_path / "stripped.ckpt")
    before = os.path.getsize(src_p)

    sos.strip_optimizer_state(src_p, out_p)

    reread = torch.load(src_p, map_location="cpu", weights_only=False)
    assert len(reread["optimizer_states"]) == 1
    assert reread["optimizer_states"][0]["state"], "anchor optimizer state was mutated"
    assert os.path.getsize(src_p) == before


# ---------------------------------------------------------------------------------------
# (b) fail-closed on a missing input
# ---------------------------------------------------------------------------------------

def test_missing_input_raises(tmp_path):
    out_p = str(tmp_path / "stripped.ckpt")
    with pytest.raises(FileNotFoundError):
        sos.strip_optimizer_state(str(tmp_path / "nope.ckpt"), out_p)
    assert not os.path.exists(out_p)


def test_missing_input_cli_exits_nonzero(tmp_path):
    out_p = tmp_path / "stripped.ckpt"
    r = subprocess.run(
        [sys.executable, "-m", "src.tools.strip_optimizer_state",
         "--in", str(tmp_path / "nope.ckpt"), "--out", str(out_p)],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert not out_p.exists()


# ---------------------------------------------------------------------------------------
# (c) never writes over --in
# ---------------------------------------------------------------------------------------

def test_refuses_same_path(tmp_path):
    src_p = _write(tmp_path)
    with pytest.raises(ValueError):
        sos.strip_optimizer_state(src_p, src_p)
    # unchanged
    reread = torch.load(src_p, map_location="cpu", weights_only=False)
    assert reread["optimizer_states"][0]["state"]


def test_refuses_same_path_via_indirection(tmp_path):
    """A different spelling of the same file (``./x/../anchor.ckpt``) is still the anchor."""
    src_p = _write(tmp_path)
    alias = os.path.join(str(tmp_path), ".", "anchor.ckpt")
    with pytest.raises(ValueError):
        sos.strip_optimizer_state(src_p, alias)
    reread = torch.load(src_p, map_location="cpu", weights_only=False)
    assert reread["optimizer_states"][0]["state"]


# ---------------------------------------------------------------------------------------
# (d) idempotency
# ---------------------------------------------------------------------------------------

def test_idempotent(tmp_path):
    src_p = _write(tmp_path)
    once = str(tmp_path / "once.ckpt")
    twice = str(tmp_path / "twice.ckpt")

    sos.strip_optimizer_state(src_p, once)
    sos.strip_optimizer_state(once, twice)

    a = torch.load(once, map_location="cpu", weights_only=False)
    b = torch.load(twice, map_location="cpu", weights_only=False)
    _assert_same(a, b)
    assert b["optimizer_states"] == []


# ---------------------------------------------------------------------------------------
# CLI surface + summary reporting
# ---------------------------------------------------------------------------------------

def test_cli_writes_output_and_prints_summary(tmp_path):
    src_p = _write(tmp_path)
    out_p = tmp_path / "stripped.ckpt"
    r = subprocess.run(
        [sys.executable, "-m", "src.tools.strip_optimizer_state",
         "--in", src_p, "--out", str(out_p)],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert out_p.exists()
    out = torch.load(str(out_p), map_location="cpu", weights_only=False)
    assert out["optimizer_states"] == []
    assert out["global_step"] == 87500
    # a human-readable removed/kept summary is part of the contract
    txt = r.stdout
    assert "optimizer_states" in txt
    for kept in ("state_dict", "lr_schedulers", "global_step", "loops", "callbacks"):
        assert kept in txt, f"summary does not mention kept key {kept!r}"


def test_refuses_to_clobber_existing_output(tmp_path):
    src_p = _write(tmp_path)
    out_p = tmp_path / "stripped.ckpt"
    out_p.write_text("pre-existing")
    with pytest.raises(FileExistsError):
        sos.strip_optimizer_state(src_p, str(out_p))
    assert out_p.read_text() == "pre-existing"

    # ...unless --force / force=True is given
    sos.strip_optimizer_state(src_p, str(out_p), force=True)
    out = torch.load(str(out_p), map_location="cpu", weights_only=False)
    assert out["optimizer_states"] == []


def test_missing_optimizer_states_key_is_rejected(tmp_path):
    """A weights-only / already-mangled checkpoint is not a valid resume anchor."""
    ck = _fake_ckpt()
    del ck["optimizer_states"]
    p = tmp_path / "weights_only.ckpt"
    torch.save(ck, p)
    with pytest.raises(KeyError):
        sos.strip_optimizer_state(str(p), str(tmp_path / "out.ckpt"))
