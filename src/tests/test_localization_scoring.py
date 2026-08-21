import numpy as np
import pytest
import torch

from src.localization.scoring import (
    deterministic_random_candidate,
    localization_metrics,
    log_mean_exp_scores,
    stable_argmax,
)


def test_nested_log_mean_exp_and_stable_tie():
    similarities = torch.tensor([[0.2] * 8, [0.1, 0.3] * 4], dtype=torch.float32)
    scores = log_mean_exp_scores(similarities)
    assert tuple(scores) == (1, 4, 8)
    assert scores[1].tolist() == pytest.approx([0.2, 0.1])
    assert scores[8][0].item() == pytest.approx(0.2)
    assert stable_argmax(torch.tensor([1.0, 1.0, 0.5])) == 0


def test_log_mean_exp_is_stable_and_rejects_invalid_inputs():
    values = torch.tensor([[1000.0, 999.0]], dtype=torch.float32)
    assert torch.isfinite(log_mean_exp_scores(values, (2,), tau=0.01)[2]).all()
    with pytest.raises(ValueError):
        log_mean_exp_scores(torch.tensor([[float("nan")]]), (1,))
    with pytest.raises(ValueError):
        log_mean_exp_scores(torch.ones(1, 1), (2,))
    with pytest.raises(ValueError):
        log_mean_exp_scores(torch.ones(1, 1), (1,), tau=0)


def test_localization_and_random_baseline_metrics():
    candidates = np.array([[0, 0, 0], [0.5, 0, 0], [2, 0, 0]], dtype=float)
    metrics = localization_metrics(candidates, np.array([0.4, 0, 0]), 2)
    assert metrics["localization_error_m"] == pytest.approx(1.6)
    assert metrics["oracle_error_m"] == pytest.approx(0.1)
    assert metrics["excess_error_m"] == pytest.approx(1.5)
    assert metrics["success_1_0m"] == 0
    assert deterministic_random_candidate(7, 10) == deterministic_random_candidate(7, 10)
    assert 0 <= deterministic_random_candidate(7, 10) < 10
