"""Tests for ``src/tools/retune_lr_state.py`` (exp_13, decay_tail arm).

TDD.  The **red** test here is the *clobber reproduction* that motivates the whole
tool — ``test_config_swap_alone_is_clobbered_by_the_checkpoint`` — the B1 BLOCKER of
``decay_tail_plan_review_opus_fallback.md``:

    ``InverseLR`` stores ``inv_gamma`` / ``power`` / ``warmup`` / ``final_lr`` as plain
    instance attributes; ``torch``'s ``LRScheduler.state_dict()`` serialises every
    attribute except ``optimizer``, and ``load_state_dict`` is literally
    ``self.__dict__.update(state_dict)`` (``torch/optim/lr_scheduler.py:149``).  A warm
    resume therefore **overwrites** a freshly configured ``InverseLR(30000, 1.0)`` with
    the checkpoint's ``InverseLR(1e6, 0.5)`` — the config-only treatment is a silent
    no-op.

The treatment must consequently be delivered by a **checkpoint rewrite**:
``retune_lr_state.py`` writes a COPY of the 87.5k anchor whose

  * ``lr_schedulers[0]['inv_gamma']`` is ``30000`` and ``['power']`` is ``1.0``, and
  * ``optimizer_states[0]['param_groups'][0]['lr']`` is ``1.2765957446808513e-05``
    (so step 87,501 *itself* runs on the tail curve rather than at 4.79e-05),

with **everything else** — counters (``last_epoch`` / ``_step_count``), ``base_lrs``,
``warmup``, ``final_lr``, model + EMA weights, ``loops`` / ``callbacks`` and the embedded
``model_config`` — byte-for-byte preserved.  Copy-only: the anchor is never written.

Contract pinned here:

* the clobber reproduction (red) — documents the mechanism the tool exists to defeat;
* the retuned **round-trip**: after the real PL restore sequence the *live*
  ``optimizer.param_groups[0]['lr']`` is exactly ``1.2765957446808513e-05``, and one
  ``optimizer.step()`` + ``scheduler.step()`` later the lr follows the **new** curve
  (closed form of the repo's ``InverseLR`` at 87,501, asserted to 1e-12);
* everything-else preservation, incl. the embedded ``model_config``;
* fail-closed on: a missing input, ``--out`` being the same file as ``--in`` (``./``
  spellings, symlinks, symlinked parents, hardlinks), an existing ``--out`` without
  ``--force``, and missing / malformed ``optimizer_states`` / ``lr_schedulers`` keys;
* idempotency, and that the input file's sha256 is unchanged by a run.

Synthetic checkpoint dicts only — the real 690 MB anchor is never touched here.
"""
import copy
import hashlib
import json
import os
import subprocess
import sys

import pytest
import torch

import src.tools.retune_lr_state as rls

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- the anchor's measured state at 87,500 steps (read off the real checkpoint) ---
ANCHOR_STEP = 87500
ANCHOR_LR = 4.794633014853842e-05          # lr_schedulers[0]['_last_lr'][0]
BASE_LR = 5e-05                            # base_lrs[0] / param_groups[0]['initial_lr']
OLD_INV_GAMMA, OLD_POWER, WARMUP = 1000000, 0.5, 0.99

# --- the exp_13 treatment (plan_decay_tail.md §1) ---
NEW_INV_GAMMA, NEW_POWER = 30000, 1.0
TAIL_LR = 1.2765957446808513e-05           # InverseLR(30000, 1.0, 0.99) at last_epoch 87500


def _inverse_lr_closed_form(last_epoch, inv_gamma, power, warmup=WARMUP,
                            final_lr=0.0, base_lr=BASE_LR):
    """``src/training/utils.py::InverseLR._get_closed_form_lr``, transcribed."""
    w = 1 - warmup ** (last_epoch + 1)
    lr_mult = (1 + last_epoch / inv_gamma) ** -power
    return w * max(final_lr, base_lr * lr_mult)


def _real_old_scheduler_state():
    """An 87.5k scheduler state dict produced by the REAL ``InverseLR(1e6, 0.5, 0.99)``.

    Constructed with ``last_epoch=87499`` so torch's ``_initial_step`` advances it to
    87,500 (the anchor's value); ``_step_count`` is then set to the anchor's 87,501.
    Verified field-by-field against the real checkpoint: the resulting dict is
    ``{'inv_gamma': 1000000, 'power': 0.5, 'warmup': 0.99, 'final_lr': 0.0,
    'base_lrs': [5e-05], 'last_epoch': 87500, '_step_count': 87501,
    '_get_lr_called_within_step': False, '_last_lr': [4.794633014853842e-05]}``.
    """
    from src.training.utils import InverseLR
    param = torch.nn.Parameter(torch.zeros(3))
    opt = torch.optim.AdamW([param], lr=BASE_LR, betas=(0.9, 0.999), weight_decay=1e-3)
    opt.param_groups[0]["initial_lr"] = BASE_LR
    sched = InverseLR(opt, inv_gamma=OLD_INV_GAMMA, power=OLD_POWER, warmup=WARMUP,
                      last_epoch=ANCHOR_STEP - 1)
    assert sched.last_epoch == ANCHOR_STEP
    sched._step_count = ANCHOR_STEP + 1        # match the anchor exactly
    st = sched.state_dict()
    assert st["_last_lr"] == [ANCHOR_LR], st["_last_lr"]
    return st


def _fake_ckpt():
    """A miniature PL-2.1-shaped 87.5k training checkpoint (anchor key layout)."""
    return {
        "epoch": 19,
        "global_step": ANCHOR_STEP,
        "pytorch-lightning_version": "2.1.0",
        "state_dict": {
            "diffusion.model.w": torch.ones(4),
            "diffusion_ema.initted": torch.tensor(True),
            "diffusion_ema.step": torch.tensor(ANCHOR_STEP),
            "diffusion_ema.ema_model.model.w": torch.full((4,), 0.5),
        },
        "loops": {"fit_loop": {"epoch_loop.batch_progress": {"total": {"completed": ANCHOR_STEP}}}},
        "callbacks": {"ModelCheckpoint{'monitor': None}": {"best_model_score": None}},
        "optimizer_states": [
            {
                # shaped for the 3-element parameter used by ``_fresh_opt_and_sched``
                # so ``optimizer.load_state_dict`` + ``step()`` really run
                "state": {0: {"step": torch.tensor(float(ANCHOR_STEP)),
                              "exp_avg": torch.ones(3),
                              "exp_avg_sq": torch.ones(3)}},
                "param_groups": [{"lr": ANCHOR_LR, "betas": [0.9, 0.999], "eps": 1e-8,
                                  "weight_decay": 0.001, "amsgrad": False, "maximize": False,
                                  "foreach": None, "capturable": False, "differentiable": False,
                                  "fused": None, "decoupled_weight_decay": True,
                                  "initial_lr": BASE_LR, "params": [0]}],
            }
        ],
        "lr_schedulers": [_real_old_scheduler_state()],
        # the anchor embeds FLAC_AR_BVp1.json; the tool must not touch it
        "model_config": {"model_type": "diffusion_cond",
                         "training": {"optimizer_configs": {"diffusion": {"scheduler": {
                             "type": "InverseLR",
                             "config": {"inv_gamma": OLD_INV_GAMMA, "power": OLD_POWER,
                                        "warmup": WARMUP}}}}}},
    }


def _write(tmp_path, name="anchor.ckpt"):
    p = tmp_path / name
    torch.save(_fake_ckpt(), p)
    return str(p)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _fresh_opt_and_sched(inv_gamma, power, warmup=WARMUP):
    """Mimic FLAC's ``configure_optimizers``: AdamW(5e-5) + InverseLR(...)@step."""
    from src.training.utils import InverseLR
    param = torch.nn.Parameter(torch.zeros(3))
    opt = torch.optim.AdamW([param], lr=BASE_LR, betas=(0.9, 0.999), weight_decay=1e-3)
    sched = InverseLR(opt, inv_gamma=inv_gamma, power=power, warmup=warmup)
    return param, opt, sched


def _pl_restore(opt, sched, ckpt):
    """The PL 2.1 restore sequence, verbatim in behaviour.

    ``Strategy.load_optimizer_state_dict`` (strategies/strategy.py:365-369) zips the
    optimizers against ``checkpoint['optimizer_states']``; ``restore_lr_schedulers``
    (checkpoint_connector.py:383-391) calls ``scheduler.load_state_dict``.
    """
    for optimizer, opt_state in zip([opt], ckpt["optimizer_states"]):
        optimizer.load_state_dict(copy.deepcopy(opt_state))
    sched.load_state_dict(copy.deepcopy(ckpt["lr_schedulers"][0]))


# =======================================================================================
# (a) THE RED TEST — B1 clobber reproduction: a config-only swap is a silent no-op
# =======================================================================================

def test_config_swap_alone_is_clobbered_by_the_checkpoint():
    """Configuring ``InverseLR(30000, 1.0)`` and warm-resuming REVERTS to (1e6, 0.5).

    This is the measured BLOCKER that makes ``retune_lr_state.py`` necessary: the
    checkpoint, not the config, owns the scheduler's hyperparameters on resume.
    """
    param, opt, sched = _fresh_opt_and_sched(NEW_INV_GAMMA, NEW_POWER)
    # the fresh (config-side) scheduler really is the decay-tail one ...
    assert (sched.inv_gamma, sched.power, sched.warmup) == (NEW_INV_GAMMA, NEW_POWER, WARMUP)

    ck = _fake_ckpt()          # an 87.5k anchor-shaped state saved from InverseLR(1e6, 0.5, 0.99)
    _pl_restore(opt, sched, ck)

    # ... and the restore CLOBBERS it back to the anchor's hyperparameters
    assert sched.inv_gamma == OLD_INV_GAMMA, "inv_gamma must revert (this is the bug)"
    assert sched.power == OLD_POWER, "power must revert (this is the bug)"
    assert sched.warmup == WARMUP
    assert sched.last_epoch == ANCHOR_STEP and sched._step_count == ANCHOR_STEP + 1
    # the live lr is the anchor's, ~3.76x above the intended tail lr
    assert opt.param_groups[0]["lr"] == ANCHOR_LR
    assert ANCHOR_LR / TAIL_LR == pytest.approx(3.7558, rel=1e-4)

    # and the very next scheduled step continues on the OLD curve, not the tail curve
    param.grad = torch.ones(3)
    opt.step()
    sched.step()
    lr_after = opt.param_groups[0]["lr"]
    old_curve = _inverse_lr_closed_form(ANCHOR_STEP + 1, OLD_INV_GAMMA, OLD_POWER)
    new_curve = _inverse_lr_closed_form(ANCHOR_STEP + 1, NEW_INV_GAMMA, NEW_POWER)
    assert lr_after == pytest.approx(old_curve, abs=1e-12)
    assert abs(lr_after - new_curve) > 3e-05, "the config-only swap is a NO-OP by ~3.5e-05"


# =======================================================================================
# (b) the retuned round-trip — the tool's reason to exist
# =======================================================================================

def test_retuned_ckpt_restores_the_tail_lr_and_the_tail_curve(tmp_path):
    src_p = _write(tmp_path)
    out_p = str(tmp_path / "retuned.ckpt")

    rls.retune_lr_state(src_p, out_p)
    retuned = torch.load(out_p, map_location="cpu", weights_only=False)

    # the rewritten fields, as stored
    assert retuned["lr_schedulers"][0]["inv_gamma"] == NEW_INV_GAMMA
    assert retuned["lr_schedulers"][0]["power"] == NEW_POWER
    assert retuned["optimizer_states"][0]["param_groups"][0]["lr"] == TAIL_LR

    # drive the REAL PL restore sequence with the dtail config's scheduler
    param, opt, sched = _fresh_opt_and_sched(NEW_INV_GAMMA, NEW_POWER)
    _pl_restore(opt, sched, retuned)

    # the live optimizer lr is EXACTLY the tail lr (this is what step 87,501 runs at)
    assert opt.param_groups[0]["lr"] == TAIL_LR
    assert opt.param_groups[0]["initial_lr"] == BASE_LR
    # ... and the restored scheduler now carries the tail hyperparameters (no clobber)
    assert (sched.inv_gamma, sched.power, sched.warmup) == (NEW_INV_GAMMA, NEW_POWER, WARMUP)
    assert sched.final_lr == 0.0
    assert sched.base_lrs == [BASE_LR]
    assert sched.last_epoch == ANCHOR_STEP and sched._step_count == ANCHOR_STEP + 1

    # one training step at the tail lr, then the schedule advances on the NEW curve
    param.grad = torch.ones(3)
    opt.step()
    assert opt.param_groups[0]["lr"] == TAIL_LR, "step 87,501 must run at the tail lr"
    sched.step()
    assert sched.last_epoch == ANCHOR_STEP + 1
    expected = _inverse_lr_closed_form(ANCHOR_STEP + 1, NEW_INV_GAMMA, NEW_POWER)
    assert expected == pytest.approx(1.2765848801286797e-05, abs=1e-18)
    assert opt.param_groups[0]["lr"] == pytest.approx(expected, abs=1e-12)
    # and it is NOT the untreated curve
    assert abs(opt.param_groups[0]["lr"]
               - _inverse_lr_closed_form(ANCHOR_STEP + 1, OLD_INV_GAMMA, OLD_POWER)) > 3e-05


def test_tail_curve_reaches_the_pre_registered_budget_end_lr():
    """Pin the plan's arithmetic: lr(97,500) on the tail curve is 1.1764706e-05."""
    assert _inverse_lr_closed_form(97500, NEW_INV_GAMMA, NEW_POWER) == pytest.approx(
        1.1764705882352942e-05, abs=1e-18)
    assert TAIL_LR / ANCHOR_LR == pytest.approx(0.2662, rel=1e-3)   # "0.266x", not 1/4


# =======================================================================================
# (c) preservation: everything except the three rewritten scalars
# =======================================================================================

def test_everything_else_is_preserved(tmp_path):
    src_p = _write(tmp_path)
    out_p = str(tmp_path / "retuned.ckpt")
    rls.retune_lr_state(src_p, out_p)

    orig = _fake_ckpt()
    out = torch.load(out_p, map_location="cpu", weights_only=False)

    assert list(out.keys()) == list(orig.keys())
    for k in orig:
        if k in ("optimizer_states", "lr_schedulers"):
            continue
        _assert_same(out[k], orig[k], f"ckpt[{k!r}]")

    # scheduler: only inv_gamma / power differ
    s_out, s_orig = out["lr_schedulers"][0], orig["lr_schedulers"][0]
    assert list(s_out.keys()) == list(s_orig.keys())
    for k in s_orig:
        if k in ("inv_gamma", "power"):
            continue
        assert s_out[k] == s_orig[k], f"lr_schedulers[0][{k!r}] changed: {s_orig[k]!r} -> {s_out[k]!r}"
    assert s_out["warmup"] == WARMUP
    assert s_out["final_lr"] == 0.0
    assert s_out["base_lrs"] == [BASE_LR]
    assert s_out["last_epoch"] == ANCHOR_STEP
    assert s_out["_step_count"] == ANCHOR_STEP + 1
    assert s_out["_last_lr"] == [ANCHOR_LR], "_last_lr is a counter, not a control input"

    # optimizer: Adam moments intact (this is a WARM continuation), only lr rewritten
    e_out, e_orig = out["optimizer_states"][0], orig["optimizer_states"][0]
    _assert_same(e_out["state"], e_orig["state"], "optimizer_states[0]['state']")
    assert len(e_out["state"]) == 1
    pg_out, pg_orig = e_out["param_groups"][0], e_orig["param_groups"][0]
    assert list(pg_out.keys()) == list(pg_orig.keys())
    for k in pg_orig:
        if k == "lr":
            continue
        assert pg_out[k] == pg_orig[k], f"param_groups[0][{k!r}] changed"
    assert pg_out["initial_lr"] == BASE_LR

    # resume-critical spot checks
    assert out["global_step"] == ANCHOR_STEP and out["epoch"] == 19
    assert int(out["state_dict"]["diffusion_ema.step"]) == ANCHOR_STEP
    assert torch.equal(out["state_dict"]["diffusion_ema.ema_model.model.w"], torch.full((4,), 0.5))
    assert out["loops"] == orig["loops"] and out["callbacks"] == orig["callbacks"]


def test_embedded_model_config_is_untouched(tmp_path):
    """The retuned copy still embeds BVp1's scheduler block — the tool rewrites STATE only.

    (The launcher's INITIAL mode asserts ``embedded model_config == FLAC_AR_BVp1.json``
    against the *anchor*; the retuned copy inherits it verbatim so the same reading holds.)
    """
    src_p = _write(tmp_path)
    out_p = str(tmp_path / "retuned.ckpt")
    rls.retune_lr_state(src_p, out_p)
    out = torch.load(out_p, map_location="cpu", weights_only=False)
    _assert_same(out["model_config"], _fake_ckpt()["model_config"], "model_config")
    sched_cfg = out["model_config"]["training"]["optimizer_configs"]["diffusion"]["scheduler"]
    assert sched_cfg["config"]["inv_gamma"] == OLD_INV_GAMMA


def test_input_file_is_not_modified(tmp_path):
    src_p = _write(tmp_path)
    out_p = str(tmp_path / "retuned.ckpt")
    before_sha = _sha256(src_p)
    before_size = os.path.getsize(src_p)

    rls.retune_lr_state(src_p, out_p)

    assert _sha256(src_p) == before_sha, "the anchor's sha256 changed — the tool wrote its input"
    assert os.path.getsize(src_p) == before_size
    reread = torch.load(src_p, map_location="cpu", weights_only=False)
    assert reread["lr_schedulers"][0]["inv_gamma"] == OLD_INV_GAMMA
    assert reread["lr_schedulers"][0]["power"] == OLD_POWER
    assert reread["optimizer_states"][0]["param_groups"][0]["lr"] == ANCHOR_LR


# =======================================================================================
# (d) idempotency
# =======================================================================================

def test_idempotent(tmp_path):
    src_p = _write(tmp_path)
    once = str(tmp_path / "once.ckpt")
    twice = str(tmp_path / "twice.ckpt")

    rls.retune_lr_state(src_p, once)
    once_sha = _sha256(once)
    rls.retune_lr_state(once, twice)

    a = torch.load(once, map_location="cpu", weights_only=False)
    b = torch.load(twice, map_location="cpu", weights_only=False)
    _assert_same(a, b)
    assert b["lr_schedulers"][0]["inv_gamma"] == NEW_INV_GAMMA
    assert b["lr_schedulers"][0]["power"] == NEW_POWER
    assert b["optimizer_states"][0]["param_groups"][0]["lr"] == TAIL_LR
    assert _sha256(once) == once_sha, "the second run mutated its input"


# =======================================================================================
# (e) fail-closed: missing input
# =======================================================================================

def test_missing_input_raises(tmp_path):
    out_p = str(tmp_path / "retuned.ckpt")
    with pytest.raises(FileNotFoundError):
        rls.retune_lr_state(str(tmp_path / "nope.ckpt"), out_p)
    assert not os.path.exists(out_p)


def test_missing_input_cli_exits_nonzero(tmp_path):
    out_p = tmp_path / "retuned.ckpt"
    r = subprocess.run(
        [sys.executable, "-m", "src.tools.retune_lr_state",
         "--in", str(tmp_path / "nope.ckpt"), "--out", str(out_p)],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert not out_p.exists()


# =======================================================================================
# (f) fail-closed: never writes over --in (path spelling, symlink, symlinked parent, hardlink)
# =======================================================================================

def _assert_intact(p):
    reread = torch.load(p, map_location="cpu", weights_only=False)
    assert reread["lr_schedulers"][0]["inv_gamma"] == OLD_INV_GAMMA, f"{p} was mutated"


def test_refuses_same_path(tmp_path):
    src_p = _write(tmp_path)
    with pytest.raises(ValueError):
        rls.retune_lr_state(src_p, src_p)
    _assert_intact(src_p)


def test_refuses_same_path_via_dot_spelling(tmp_path):
    src_p = _write(tmp_path)
    alias = os.path.join(str(tmp_path), ".", "anchor.ckpt")
    with pytest.raises(ValueError):
        rls.retune_lr_state(src_p, alias)
    _assert_intact(src_p)


def test_refuses_symlink_to_input(tmp_path):
    src_p = _write(tmp_path)
    link = str(tmp_path / "link.ckpt")
    os.symlink(src_p, link)
    with pytest.raises(ValueError):
        rls.retune_lr_state(src_p, link)
    _assert_intact(src_p)
    # ...and the reverse direction (read through the symlink, write the real path)
    with pytest.raises(ValueError):
        rls.retune_lr_state(link, src_p)
    _assert_intact(src_p)


def test_refuses_symlinked_parent_directory(tmp_path):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    src_p = _write(real_dir)
    link_dir = tmp_path / "alias"
    os.symlink(str(real_dir), str(link_dir))
    with pytest.raises(ValueError):
        rls.retune_lr_state(src_p, os.path.join(str(link_dir), "anchor.ckpt"))
    _assert_intact(src_p)


def test_refuses_hardlink_to_input(tmp_path):
    src_p = _write(tmp_path)
    hard = str(tmp_path / "hard.ckpt")
    os.link(src_p, hard)   # same inode, different path -> realpath differs, samefile catches it
    assert os.path.realpath(hard) != os.path.realpath(src_p)
    with pytest.raises(ValueError):
        rls.retune_lr_state(src_p, hard)
    _assert_intact(src_p)
    _assert_intact(hard)


# =======================================================================================
# (g) fail-closed: existing output, missing / malformed keys
# =======================================================================================

def test_refuses_to_clobber_existing_output(tmp_path):
    src_p = _write(tmp_path)
    out_p = tmp_path / "retuned.ckpt"
    out_p.write_text("pre-existing")
    with pytest.raises(FileExistsError):
        rls.retune_lr_state(src_p, str(out_p))
    assert out_p.read_text() == "pre-existing"

    # ...unless --force / force=True is given
    rls.retune_lr_state(src_p, str(out_p), force=True)
    out = torch.load(str(out_p), map_location="cpu", weights_only=False)
    assert out["lr_schedulers"][0]["inv_gamma"] == NEW_INV_GAMMA


def test_missing_lr_schedulers_key_is_rejected(tmp_path):
    ck = _fake_ckpt()
    del ck["lr_schedulers"]
    p = tmp_path / "no_sched.ckpt"
    torch.save(ck, p)
    with pytest.raises(KeyError):
        rls.retune_lr_state(str(p), str(tmp_path / "out.ckpt"))
    assert not (tmp_path / "out.ckpt").exists()


def test_missing_optimizer_states_key_is_rejected(tmp_path):
    ck = _fake_ckpt()
    del ck["optimizer_states"]
    p = tmp_path / "no_opt.ckpt"
    torch.save(ck, p)
    with pytest.raises(KeyError):
        rls.retune_lr_state(str(p), str(tmp_path / "out.ckpt"))


def test_missing_inv_gamma_in_scheduler_state_is_rejected(tmp_path):
    """A non-InverseLR scheduler state has no inv_gamma/power to retune."""
    ck = _fake_ckpt()
    del ck["lr_schedulers"][0]["inv_gamma"]
    p = tmp_path / "wrong_sched.ckpt"
    torch.save(ck, p)
    with pytest.raises(KeyError):
        rls.retune_lr_state(str(p), str(tmp_path / "out.ckpt"))


def test_missing_lr_in_param_group_is_rejected(tmp_path):
    ck = _fake_ckpt()
    del ck["optimizer_states"][0]["param_groups"][0]["lr"]
    p = tmp_path / "no_lr.ckpt"
    torch.save(ck, p)
    with pytest.raises(KeyError):
        rls.retune_lr_state(str(p), str(tmp_path / "out.ckpt"))


def test_multiple_schedulers_rejected(tmp_path):
    ck = _fake_ckpt()
    ck["lr_schedulers"].append(copy.deepcopy(ck["lr_schedulers"][0]))
    p = tmp_path / "two_scheds.ckpt"
    torch.save(ck, p)
    with pytest.raises(ValueError):
        rls.retune_lr_state(str(p), str(tmp_path / "out.ckpt"))


def test_multiple_optimizers_rejected(tmp_path):
    ck = _fake_ckpt()
    ck["optimizer_states"].append(copy.deepcopy(ck["optimizer_states"][0]))
    p = tmp_path / "two_opts.ckpt"
    torch.save(ck, p)
    with pytest.raises(ValueError):
        rls.retune_lr_state(str(p), str(tmp_path / "out.ckpt"))


def test_multiple_param_groups_rejected(tmp_path):
    ck = _fake_ckpt()
    ck["optimizer_states"][0]["param_groups"].append(
        copy.deepcopy(ck["optimizer_states"][0]["param_groups"][0]))
    p = tmp_path / "two_groups.ckpt"
    torch.save(ck, p)
    with pytest.raises(ValueError):
        rls.retune_lr_state(str(p), str(tmp_path / "out.ckpt"))


def test_malformed_optimizer_entry_is_rejected(tmp_path):
    ck = _fake_ckpt()
    ck["optimizer_states"] = [{"not": "an optimizer state_dict"}]
    p = tmp_path / "bad.ckpt"
    torch.save(ck, p)
    with pytest.raises(TypeError):
        rls.retune_lr_state(str(p), str(tmp_path / "out.ckpt"))


def test_not_a_lightning_checkpoint_is_rejected(tmp_path):
    p = tmp_path / "tensor.ckpt"
    torch.save(torch.ones(3), p)
    with pytest.raises(TypeError):
        rls.retune_lr_state(str(p), str(tmp_path / "out.ckpt"))


def test_wrong_resume_step_is_rejected(tmp_path):
    """``TAIL_LR`` is the tail-curve value at 87,500 — a different step would mis-set it."""
    ck = _fake_ckpt()
    ck["lr_schedulers"][0]["last_epoch"] = 50000
    p = tmp_path / "wrong_step.ckpt"
    torch.save(ck, p)
    with pytest.raises(ValueError):
        rls.retune_lr_state(str(p), str(tmp_path / "out.ckpt"))


# =======================================================================================
# (h) CLI surface + summary reporting
# =======================================================================================

def test_cli_writes_output_and_prints_summary(tmp_path):
    src_p = _write(tmp_path)
    out_p = tmp_path / "retuned.ckpt"
    r = subprocess.run(
        [sys.executable, "-m", "src.tools.retune_lr_state",
         "--in", src_p, "--out", str(out_p)],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert out_p.exists()
    out = torch.load(str(out_p), map_location="cpu", weights_only=False)
    assert out["lr_schedulers"][0]["inv_gamma"] == NEW_INV_GAMMA
    assert out["lr_schedulers"][0]["power"] == NEW_POWER
    assert out["optimizer_states"][0]["param_groups"][0]["lr"] == TAIL_LR
    txt = r.stdout
    assert "REWROTE" in txt and "KEPT" in txt
    assert "inv_gamma" in txt and "power" in txt
    assert repr(TAIL_LR) in txt or f"{TAIL_LR!r}" in txt
    for kept in ("state_dict", "global_step", "loops", "callbacks", "model_config"):
        assert kept in txt, f"summary does not mention kept key {kept!r}"


def test_cli_force_flag(tmp_path):
    src_p = _write(tmp_path)
    out_p = tmp_path / "retuned.ckpt"
    out_p.write_text("pre-existing")
    r = subprocess.run(
        [sys.executable, "-m", "src.tools.retune_lr_state",
         "--in", src_p, "--out", str(out_p)],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert out_p.read_text() == "pre-existing"

    r = subprocess.run(
        [sys.executable, "-m", "src.tools.retune_lr_state",
         "--in", src_p, "--out", str(out_p), "--force"],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    out = torch.load(str(out_p), map_location="cpu", weights_only=False)
    assert out["optimizer_states"][0]["param_groups"][0]["lr"] == TAIL_LR


# =======================================================================================
# (i) the shipped dtail config agrees with the tool's constants
# =======================================================================================

def test_dtail_config_scheduler_block_matches_the_tool(tmp_path):
    cfg_path = os.path.join(_REPO_ROOT, "worklog", "worklog_yixun",
                            "exp_13_decay_tail_claude", "FLAC_AR_BVp1_dtail.json")
    cfg = json.load(open(cfg_path))
    block = cfg["training"]["optimizer_configs"]["diffusion"]["scheduler"]
    assert block == {"type": "InverseLR",
                     "config": {"inv_gamma": NEW_INV_GAMMA, "power": NEW_POWER,
                                "warmup": WARMUP}}
    assert "final_lr_ratio" not in json.dumps(cfg), "final_lr_ratio is not an InverseLR kwarg"
    assert rls.NEW_INV_GAMMA == block["config"]["inv_gamma"]
    assert rls.NEW_POWER == block["config"]["power"]
    assert rls.TAIL_LR == TAIL_LR

    # negative: dtail == BVp1 once the scheduler subtrees are deleted
    bvp1 = json.load(open(os.path.join(
        _REPO_ROOT, "worklog", "worklog_yixun", "exp_07_fa_scratch_claude", "FLAC_AR_BVp1.json")))
    a, b = copy.deepcopy(cfg), copy.deepcopy(bvp1)
    for d in (a, b):
        del d["training"]["optimizer_configs"]["diffusion"]["scheduler"]
    assert a == b, "dtail differs from BVp1 somewhere other than the scheduler block"
