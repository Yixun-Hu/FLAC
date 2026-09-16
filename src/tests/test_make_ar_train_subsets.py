"""Tests for ``src/tools/make_ar_train_subsets.py`` — exp_14 round A ("data curve").

The tool builds the *nested, per-room* random subsamples of the AcousticRooms training
split (25 / 50 / 75 %) that the exp_14 data-efficiency curve trains on. It is copy-only:
it reads ``data/AR/train.json`` and writes new split files plus a manifest, never touching
its input. Because six 40k-step trainings are pinned to those files, the algorithm itself
(not merely its properties) is under test: the frozen rule of ``plan_data_curve.md`` §3 is

  1. iterate scenes sorted, rooms sorted, files sorted;
  2. ONE ``random.Random(seed)`` for the whole build; per room exactly one
     ``rng.sample(sorted_files, n)`` permutation — the only RNG consumption anywhere;
  3. raw prefix for fraction f = first ``max(1, round(f * n))`` entries of that permutation
     (Python's built-in ``round``, i.e. ties-to-even);
  4. deterministic top-up (no RNG) of every target whose receiver would otherwise hold no
     other retained source;
  5. nesting: S_f = raw_prefix(perm, f) ∪ S_{previous smaller f}, top-up applied to the union.

Test ids follow the plan's A1-A7 contract list.
"""
import random

import pytest

from src.tools import make_ar_train_subsets as mas


# ======================================================================================
# A1 — parse_nodes
# ======================================================================================
def test_A1_parse_nodes_returns_both_raw_tokens():
    assert mas.parse_nodes("S0012_R0077_hybrid_IR.wav") == ("S0012", "R0077")
    assert mas.parse_nodes("S001_R001_hybrid_IR.wav") == ("S001", "R001")


def test_A1_parse_nodes_compares_raw_strings_not_integers():
    """"S001" and "S0012" are DIFFERENT nodes: the tokens are never int-parsed."""
    assert mas.parse_nodes("S001_R008_hybrid_IR.wav")[0] == "S001"
    assert mas.parse_nodes("S0012_R008_hybrid_IR.wav")[0] == "S0012"
    assert mas.parse_nodes("S001_R008_hybrid_IR.wav")[0] != mas.parse_nodes("S0012_R008_hybrid_IR.wav")[0]
    assert mas.parse_nodes("S006_R008_hybrid_IR.wav")[1] != mas.parse_nodes("S006_R0080_hybrid_IR.wav")[1]


@pytest.mark.parametrize(
    "bad",
    [
        "",                              # empty
        "S0012",                         # no separator at all
        "S0012_R0077.wav",               # only two underscore-separated segments
        "X0012_R0077_hybrid_IR.wav",     # source token does not start with S
        "S0012_X0077_hybrid_IR.wav",     # receiver token does not start with R
        "R0077_S0012_hybrid_IR.wav",     # swapped order
        "S_R0077_hybrid_IR.wav",         # empty source id
        "S0012_R_hybrid_IR.wav",         # empty receiver id
        "_R0077_hybrid_IR.wav",          # empty source token
        "S0012__hybrid_IR.wav",          # empty receiver token
        "S0012_R0077_",                  # empty tail
    ],
)
def test_A1_parse_nodes_malformed_raises(bad):
    with pytest.raises(ValueError):
        mas.parse_nodes(bad)


def test_A1_parse_nodes_rejects_non_string():
    with pytest.raises(ValueError):
        mas.parse_nodes(None)
