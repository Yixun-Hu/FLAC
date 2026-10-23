"""exp_25 depth_canon round 3a -- the evaluator (``eval_FLAC.py``): filename token,
record provenance, checkpoint protocol binding, canonical sidecar, metadata
preparation and the scored-waveform dump (plan_depth_canon.md v8 §2.2, §5, §6
tests 16-21, 21b).

Written by the exp_25 Coder seat (Claude Opus, max effort) per plan_depth_canon.md v8 §2.2/§5/§6/§8.

CPU only. Synthetic depth goes through the REAL AR_md converter
(``test_depth_canon``'s helpers); the legacy byte pins are the exp_14 golden
fixture and ``test_eval_paths``'s exp_01 names, reused verbatim.
"""
import itertools

import pytest

import eval_FLAC
from test_exp14_fixed_mode_snapshot import MATRIX, META

CKPT = "weights/FLAC/FLAC_EMA.ckpt"


# --------------------------------------------------------------------------- #
# 16. canon_suffix + build_output_paths
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("applied, mismatch, token", [
    (False, False, ""), (True, False, "_canon"), (False, True, "_canonmis"),
    (True, True, "_canon_canonmis")])
def test_canon_suffix_tokens(applied, mismatch, token):
    assert eval_FLAC.canon_suffix(applied, mismatch) == token


@pytest.mark.parametrize("bad", [1, 0, None, "true"])
def test_canon_suffix_takes_literal_bools_only(bad):
    with pytest.raises(ValueError, match="bool"):
        eval_FLAC.canon_suffix(bad, False)
    with pytest.raises(ValueError, match="bool"):
        eval_FLAC.canon_suffix(False, bad)


@pytest.mark.parametrize("case", MATRIX, ids=lambda c: f'{c["cond_method"]}_{c["rotate_deg"]}')
def test_explicit_non_canonical_paths_are_the_golden_bytes(case):
    paths = eval_FLAC.build_output_paths(
        META["ckpt_path"], META["steps"], META["cfg_scale"], META["eval_name"],
        cond_method=case["cond_method"], rotate_deg=case["rotate_deg"],
        n_angles=case["n_angles"], canonicalize=False, canon_mismatch=False)
    assert paths == case["output_paths"]


def test_exp01_names_are_unchanged_with_explicit_defaults():
    paths = eval_FLAC.build_output_paths(CKPT, 1, 1.0, "exp01_unseen_K1_seed42",
                                         canonicalize=False, canon_mismatch=False)
    assert paths == {
        "metrics": "weights/FLAC/FLAC_EMA_metrics_1_1.0_exp01_unseen_K1_seed42.json",
        "predictions": "weights/FLAC/FLAC_EMA_predictions_1_1.0_exp01_unseen_K1_seed42.pt"}


def test_canon_token_sits_after_the_method_and_before_rotation_and_are():
    paths = eval_FLAC.build_output_paths(
        CKPT, 1, 1.0, "e", cond_method="fa_invariant", rotate_deg=90.0, are_lambda=0.5,
        canonicalize=True, canon_mismatch=True)
    assert paths["metrics"] == \
        "weights/FLAC/FLAC_EMA_metrics_1_1.0_e_fa_invariant_a4_canon_canonmis_rot90_are0p5.json"
    paths = eval_FLAC.build_output_paths(CKPT, 1, 1.0, "e", rotate_mode="random",
                                         rotate_seed=42, canon_mismatch=True)
    assert paths["predictions"] == "weights/FLAC/FLAC_EMA_predictions_1_1.0_e_canonmis_rotrand42.pt"


ROTATIONS = [dict(rotate_deg=0.0), dict(rotate_deg=90.0),
             dict(rotate_mode="random", rotate_seed=42)]
COMBOS = list(itertools.product([False, True], [False, True], range(len(ROTATIONS)),
                                ["vanilla", "fa_invariant"], [None, 0.0, 0.5]))


def _combo_paths(applied, mismatch, rot, cond_method, lam):
    return eval_FLAC.build_output_paths(
        CKPT, 1, 1.0, "cell", cond_method=cond_method, are_lambda=lam,
        canonicalize=applied, canon_mismatch=mismatch, **ROTATIONS[rot])


def test_names_are_injective_over_every_protocol_combination():
    """canonical x mismatch x fixed/random x vanilla/FA x ARE: 72 cells, 72 names each."""
    paths = [_combo_paths(*c) for c in COMBOS]
    for kind in ("metrics", "predictions"):
        assert len({p[kind] for p in paths}) == len(COMBOS) == 72
    for p in paths:  # metrics and predictions differ only by kind + extension
        assert p["metrics"][:-5].replace("_metrics_", "_") == \
            p["predictions"][:-3].replace("_predictions_", "_")
