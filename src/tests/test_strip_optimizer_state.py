"""Tests for ``src/tools/strip_optimizer_state.py`` (exp_09, F-reset arm).

TDD, revised after codex review r1 (BLOCKER: the first draft's ``optimizer_states = []``
spelling silently ran the first post-resume update at the step-0 warmup LR). The hook
produces a *copy* of a PyTorch-Lightning training checkpoint with the Adam **state**
cleared — moments and per-parameter ``step`` — while keeping everything else, including
the optimizer's ``param_groups`` and hence the **scheduled LR**.

Contract pinned here:

* ``optimizer_states`` keeps its entry; only ``entry["state"]`` is emptied.
  ``param_groups`` are identical to the input (the removed LR asymmetry between F-warm
  and F-reset is what this buys). Deleting the key is not an option: PL 2.1 raises
  ``KeyError`` when it is absent — see the hook's module docstring for source refs.
* a **restore-level** test drives the real PL restore sequence on a synthetic
  AdamW+InverseLR pair and proves: post-restore ``lr == 4.794633014853842e-05`` with an
  empty Adam state, first update at that same LR, fresh moments afterwards — and that
  the rejected empty-list spelling would instead have updated at ``5e-07``;
* fail-closed on a missing ``--in``;
* never writes over ``--in`` — including via ``./`` spellings, **symlinks** and
  **hardlinks**;
* idempotent.

Synthetic checkpoint dicts only — the real anchor is ~700 MB and is never touched by the
test-suite.
"""
import copy
import os
import subprocess
import sys

import pytest
import torch

import src.tools.strip_optimizer_state as sos

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# the anchor's scheduled LR at 87,500 steps (InverseLR(1e6, 0.5, 0.99) over base 5e-5),
# read off the real ckpt: lr_schedulers[0]['_last_lr'] == [4.794633014853842e-05]
SCHEDULED_LR = 4.794633014853842e-05


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
                "param_groups": [{"lr": SCHEDULED_LR, "betas": (0.9, 0.999), "eps": 1e-8,
                                  "weight_decay": 1e-3, "amsgrad": False, "maximize": False,
                                  "initial_lr": 5e-05, "params": [0]}],
            }
        ],
        "lr_schedulers": [
            {"inv_gamma": 1000000, "power": 0.5, "warmup": 0.99, "final_lr": 0.0,
             "base_lrs": [5e-05], "last_epoch": 87500, "_step_count": 87501,
             "_get_lr_called_within_step": False, "_last_lr": [SCHEDULED_LR]}
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
# (a) preserves everything except the per-parameter optimizer state
# ---------------------------------------------------------------------------------------

def test_strip_clears_state_but_keeps_entry_and_param_groups(tmp_path):
    src_p = _write(tmp_path)
    out_p = str(tmp_path / "stripped.ckpt")

    sos.strip_optimizer_state(src_p, out_p)

    orig = _fake_ckpt()
    out = torch.load(out_p, map_location="cpu", weights_only=False)

    # the key AND its entry survive (an absent key KeyErrors on PL-2.1 resume; an empty
    # list would drop param_groups and hence the scheduled LR)
    assert "optimizer_states" in out
    assert len(out["optimizer_states"]) == 1, "the optimizer entry must be retained"
    entry = out["optimizer_states"][0]
    assert entry["state"] == {}, "per-parameter Adam state must be cleared"
    _assert_same(entry["param_groups"], orig["optimizer_states"][0]["param_groups"],
                 "optimizer_states[0]['param_groups']")
    assert entry["param_groups"][0]["lr"] == SCHEDULED_LR
    assert entry["param_groups"][0]["initial_lr"] == 5e-05

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


def test_multiple_optimizers_all_cleared(tmp_path):
    ck = _fake_ckpt()
    ck["optimizer_states"].append(copy.deepcopy(ck["optimizer_states"][0]))
    ck["optimizer_states"][1]["param_groups"][0]["lr"] = 1.23e-4
    p = tmp_path / "two_opts.ckpt"
    torch.save(ck, p)

    out_p = str(tmp_path / "stripped.ckpt")
    sos.strip_optimizer_state(str(p), out_p)

    out = torch.load(out_p, map_location="cpu", weights_only=False)
    assert len(out["optimizer_states"]) == 2
    assert all(e["state"] == {} for e in out["optimizer_states"])
    assert [e["param_groups"][0]["lr"] for e in out["optimizer_states"]] == [SCHEDULED_LR, 1.23e-4]


def test_input_file_is_not_modified(tmp_path):
    src_p = _write(tmp_path)
    out_p = str(tmp_path / "stripped.ckpt")
    before = os.path.getsize(src_p)

    sos.strip_optimizer_state(src_p, out_p)

    reread = torch.load(src_p, map_location="cpu", weights_only=False)
    assert reread["optimizer_states"][0]["state"], "anchor optimizer state was mutated"
    assert os.path.getsize(src_p) == before


# ---------------------------------------------------------------------------------------
# restore-level behaviour: the actual reason keep-entry/clear-state was chosen
# ---------------------------------------------------------------------------------------

def _fresh_opt_and_sched():
    """Mimic FLAC's ``configure_optimizers``: AdamW(5e-5) + InverseLR(1e6, 0.5, 0.99)@step."""
    from src.training.utils import InverseLR
    param = torch.nn.Parameter(torch.zeros(3))
    opt = torch.optim.AdamW([param], lr=5e-5, betas=(0.9, 0.999), weight_decay=1e-3)
    sched = InverseLR(opt, inv_gamma=1000000, power=0.5, warmup=0.99)
    return param, opt, sched


def _pl_restore(opt, sched, ckpt):
    """The PL 2.1 restore sequence, verbatim in behaviour.

    ``Strategy.load_optimizer_state_dict`` (strategies/strategy.py:365-369) zips the
    optimizers against ``checkpoint['optimizer_states']``; ``restore_lr_schedulers``
    (checkpoint_connector.py:383-391) calls ``scheduler.load_state_dict``.
    """
    for optimizer, opt_state in zip([opt], ckpt["optimizer_states"]):
        optimizer.load_state_dict(opt_state)
    sched.load_state_dict(copy.deepcopy(ckpt["lr_schedulers"][0]))


def test_restore_keeps_scheduled_lr_with_empty_adam_state(tmp_path):
    src_p = _write(tmp_path)
    out_p = str(tmp_path / "stripped.ckpt")
    sos.strip_optimizer_state(src_p, out_p)
    stripped = torch.load(out_p, map_location="cpu", weights_only=False)

    param, opt, sched = _fresh_opt_and_sched()
    # InverseLR steps once at construction (torch/optim/lr_scheduler.py:133-136), writing
    # the step-0 warmup LR into the live param_group:
    warmup_lr = (1 - 0.99 ** 1) * 5e-5
    assert opt.param_groups[0]["lr"] == pytest.approx(warmup_lr, rel=1e-12)
    assert opt.param_groups[0]["lr"] == pytest.approx(5e-07, rel=1e-9)

    _pl_restore(opt, sched, stripped)

    # param_groups came back -> the scheduled LR is restored ...
    assert opt.param_groups[0]["lr"] == SCHEDULED_LR
    assert opt.param_groups[0]["initial_lr"] == 5e-05
    # ... while Adam has NO moments yet
    assert len(opt.state) == 0, "Adam state must be empty before the first update"
    assert sched.last_epoch == 87500 and sched._step_count == 87501

    # first update runs at the scheduled LR and creates fresh moments with step == 1
    param.grad = torch.ones(3)
    opt.step()
    assert opt.param_groups[0]["lr"] == SCHEDULED_LR, "first update must use the scheduled LR"
    assert len(opt.state) == 1
    st = opt.state[param]
    assert float(st["step"]) == 1.0, "per-parameter step must restart from 0"
    assert torch.count_nonzero(st["exp_avg"]) > 0 and torch.count_nonzero(st["exp_avg_sq"]) > 0

    # the schedule continues unbroken from 87,500
    sched.step()
    assert sched.last_epoch == 87501
    assert opt.param_groups[0]["lr"] < SCHEDULED_LR  # InverseLR decays monotonically here


def test_warm_resume_restores_moments_for_contrast():
    """Sanity contrast: the untouched checkpoint restores moments AND the same LR."""
    ck = _fake_ckpt()
    param, opt, sched = _fresh_opt_and_sched()
    ck["optimizer_states"][0]["state"] = {0: {"step": torch.tensor(87500.0),
                                              "exp_avg": torch.ones(3),
                                              "exp_avg_sq": torch.ones(3)}}
    _pl_restore(opt, sched, ck)
    assert opt.param_groups[0]["lr"] == SCHEDULED_LR
    assert len(opt.state) == 1
    assert float(opt.state[param]["step"]) == 87500.0


def test_rejected_empty_list_spelling_would_use_the_warmup_lr(tmp_path):
    """Regression guard for the codex-r1 BLOCKER.

    Documents *why* ``optimizer_states = []`` was rejected: PL's zip restores nothing, so
    the first update would run at the step-0 warmup LR (~5e-07), ~96x below the schedule.
    """
    param, opt, sched = _fresh_opt_and_sched()
    ck = _fake_ckpt()
    ck["optimizer_states"] = []          # the rejected spelling
    _pl_restore(opt, sched, ck)
    assert opt.param_groups[0]["lr"] == pytest.approx(5e-07, rel=1e-9)
    assert opt.param_groups[0]["lr"] != SCHEDULED_LR
    assert SCHEDULED_LR / opt.param_groups[0]["lr"] > 90

    # and the shipped hook does NOT produce that spelling
    src_p = _write(tmp_path)
    out_p = str(tmp_path / "stripped.ckpt")
    sos.strip_optimizer_state(src_p, out_p)
    assert torch.load(out_p, map_location="cpu", weights_only=False)["optimizer_states"] != []


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
# (c) never writes over --in (path spelling, symlink, hardlink)
# ---------------------------------------------------------------------------------------

def _assert_intact(p):
    reread = torch.load(p, map_location="cpu", weights_only=False)
    assert reread["optimizer_states"][0]["state"], f"{p} was mutated"


def test_refuses_same_path(tmp_path):
    src_p = _write(tmp_path)
    with pytest.raises(ValueError):
        sos.strip_optimizer_state(src_p, src_p)
    _assert_intact(src_p)


def test_refuses_same_path_via_dot_spelling(tmp_path):
    src_p = _write(tmp_path)
    alias = os.path.join(str(tmp_path), ".", "anchor.ckpt")
    with pytest.raises(ValueError):
        sos.strip_optimizer_state(src_p, alias)
    _assert_intact(src_p)


def test_refuses_symlink_to_input(tmp_path):
    src_p = _write(tmp_path)
    link = str(tmp_path / "link.ckpt")
    os.symlink(src_p, link)
    with pytest.raises(ValueError):
        sos.strip_optimizer_state(src_p, link)
    _assert_intact(src_p)
    # ...and the reverse direction (read through the symlink, write the real path)
    with pytest.raises(ValueError):
        sos.strip_optimizer_state(link, src_p)
    _assert_intact(src_p)


def test_refuses_symlinked_parent_directory(tmp_path):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    src_p = _write(real_dir)
    link_dir = tmp_path / "alias"
    os.symlink(str(real_dir), str(link_dir))
    with pytest.raises(ValueError):
        sos.strip_optimizer_state(src_p, os.path.join(str(link_dir), "anchor.ckpt"))
    _assert_intact(src_p)


def test_refuses_hardlink_to_input(tmp_path):
    src_p = _write(tmp_path)
    hard = str(tmp_path / "hard.ckpt")
    os.link(src_p, hard)  # same inode, different path -> realpath differs, samefile catches it
    assert os.path.realpath(hard) != os.path.realpath(src_p)
    with pytest.raises(ValueError):
        sos.strip_optimizer_state(src_p, hard)
    _assert_intact(src_p)
    _assert_intact(hard)


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
    assert b["optimizer_states"][0]["state"] == {}
    assert b["optimizer_states"][0]["param_groups"][0]["lr"] == SCHEDULED_LR


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
    assert out["optimizer_states"][0]["state"] == {}
    assert out["global_step"] == 87500
    # a human-readable removed/kept summary is part of the contract
    txt = r.stdout
    assert "param_group" in txt and "REMOVED" in txt and "KEPT" in txt
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
    assert out["optimizer_states"][0]["state"] == {}


def test_missing_optimizer_states_key_is_rejected(tmp_path):
    """A weights-only / already-mangled checkpoint is not a valid resume anchor."""
    ck = _fake_ckpt()
    del ck["optimizer_states"]
    p = tmp_path / "weights_only.ckpt"
    torch.save(ck, p)
    with pytest.raises(KeyError):
        sos.strip_optimizer_state(str(p), str(tmp_path / "out.ckpt"))


def test_malformed_optimizer_entry_is_rejected(tmp_path):
    ck = _fake_ckpt()
    ck["optimizer_states"] = [{"not": "an optimizer state_dict"}]
    p = tmp_path / "bad.ckpt"
    torch.save(ck, p)
    with pytest.raises(TypeError):
        sos.strip_optimizer_state(str(p), str(tmp_path / "out.ckpt"))
