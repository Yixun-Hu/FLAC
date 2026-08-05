"""Tests for the exp_11 P0 profiling runner's in-fit timing callback.

Round-2 review B2: full-process wall time is dominated by imports, VAE load, DDP
rendezvous and first-batch warmup, and those terms do NOT cancel between two
independent Slurm jobs. The steady-state rate is therefore measured *inside one
fit* by ``P0StepTimer``, which timestamps completed optimizer steps 10 and 30:

    steps/s = (30 - 10) / (t_mono(30) - t_mono(10))

Only the callback bookkeeping and the window arithmetic are unit-tested here
(CPU, no Lightning Trainer, no GPU, no data): the trainer is a duck-typed stub.
``p0_runner`` keeps its heavy imports (train.py, src.*) inside ``main()`` so this
import stays cheap.
"""
import importlib.util
import math
import os
import re

import pytest


_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # src/tests/ -> src/ -> repo root
_RUNNER_PY = os.path.join(
    _REPO_ROOT, "worklog", "worklog_yixun", "exp_11_fa_orbit_claude", "p0_runner.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("exp11_p0_runner", _RUNNER_PY)
    assert spec is not None and spec.loader is not None, f"cannot load {_RUNNER_PY}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load_module()


class _Trainer:
    """Duck-typed stand-in for the bits of pl.Trainer the callback reads."""

    def __init__(self, global_rank=0):
        self.global_step = 0
        self.global_rank = global_rank


def _feed(cb, trainer, steps):
    """Drive on_train_batch_end once per completed optimizer step."""
    for idx, step in enumerate(steps):
        trainer.global_step = step
        cb.on_train_batch_end(trainer, None, None, None, idx)


# --------------------------------------------------------------------------- #
# 1. window arithmetic
# --------------------------------------------------------------------------- #
def test_window_rate():
    assert R.window_rate({10: 100.0, 30: 200.0}) == pytest.approx(0.2)
    assert R.window_rate({10: 5.0, 30: 15.0}) == pytest.approx(2.0)


def test_window_rate_requires_both_marks():
    for marks in ({10: 100.0}, {30: 200.0}, {}):
        with pytest.raises(ValueError):
            R.window_rate(marks)


def test_window_rate_rejects_nonpositive_and_nonfinite():
    for marks in ({10: 200.0, 30: 200.0},          # zero delta
                  {10: 300.0, 30: 200.0},          # time went backwards
                  {10: 100.0, 30: float("nan")},
                  {10: float("inf"), 30: 200.0}):
        with pytest.raises(ValueError):
            R.window_rate(marks)


# --------------------------------------------------------------------------- #
# 2. the callback marks exactly the window steps, once each
# --------------------------------------------------------------------------- #
def test_callback_marks_only_window_steps():
    cb = R.P0StepTimer()
    trainer = _Trainer()
    _feed(cb, trainer, [1, 5, 9, 10, 11, 20, 29, 30, 31])
    assert sorted(cb.marks) == [10, 30]
    assert sorted(cb.wall_marks) == [10, 30]
    assert cb.marks[30] > cb.marks[10]
    assert R.window_rate(cb.marks) > 0


def test_callback_mark_is_idempotent():
    cb = R.P0StepTimer()
    trainer = _Trainer()
    _feed(cb, trainer, [10, 10, 10])
    first = cb.marks[10]
    _feed(cb, trainer, [10])
    assert cb.marks[10] == first, "a repeated global_step must not re-stamp the mark"


def test_callback_window_is_configurable():
    cb = R.P0StepTimer(window=(2, 4))
    _feed(cb, _Trainer(), [1, 2, 3, 4, 5])
    assert sorted(cb.marks) == [2, 4]


# --------------------------------------------------------------------------- #
# 3. rank-zero-only printing, in the exact format the sbatch greps
# --------------------------------------------------------------------------- #
def test_prints_only_on_rank_zero(capsys):
    _feed(R.P0StepTimer(), _Trainer(global_rank=1), [10, 30])
    assert capsys.readouterr().out == "", "non-zero ranks must stay silent"

    cb = R.P0StepTimer()
    _feed(cb, _Trainer(global_rank=0), [10, 30])
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("P0STEP")]
    assert len(lines) == 2
    parsed = [re.fullmatch(R.P0STEP_RE, l) for l in lines]
    assert all(parsed), f"printed lines do not match P0STEP_RE: {lines}"
    got = {int(m.group("step")): (float(m.group("t")), float(m.group("ts"))) for m in parsed}
    assert sorted(got) == [10, 30]
    for step, (mono, epoch) in got.items():
        assert math.isfinite(mono) and math.isfinite(epoch)
        assert epoch > 1.7e9, "ts must be an epoch timestamp (time.time)"
        assert mono == pytest.approx(cb.marks[step], abs=1e-6)


# --------------------------------------------------------------------------- #
# 4. CUDA is synchronised before every timestamp (else the mark races the GPU)
# --------------------------------------------------------------------------- #
def test_synchronizes_cuda_before_marking(monkeypatch):
    calls = []
    monkeypatch.setattr(R.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(R.torch.cuda, "synchronize", lambda *a, **k: calls.append(1))
    _feed(R.P0StepTimer(), _Trainer(), [9, 10, 11, 30])
    assert len(calls) == 2, "synchronize must run exactly at the two window steps"


def test_no_cuda_sync_when_unavailable(monkeypatch):
    monkeypatch.setattr(R.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(R.torch.cuda, "synchronize",
                        lambda *a, **k: pytest.fail("synchronize called without CUDA"))
    cb = R.P0StepTimer()
    _feed(cb, _Trainer(), [10, 30])
    assert sorted(cb.marks) == [10, 30]


# --------------------------------------------------------------------------- #
# 5. parity guard: the runner must not drift from train.py's factory path
# --------------------------------------------------------------------------- #
def test_runner_documents_parity_and_defers_heavy_imports():
    with open(_RUNNER_PY) as fh:
        src = fh.read()
    assert "PARITY" in src, "the runner must carry the train.py parity note"
    head = src.split("def main", 1)[0]
    for heavy in ("import train", "from src."):
        assert heavy not in head, (
            f"{heavy!r} must be deferred into main() so the unit test import stays cheap"
        )
    assert "construct_trainer" in src, "the trainer must come from train.py's factory"
