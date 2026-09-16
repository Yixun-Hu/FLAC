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


# ======================================================================================
# Shared toy split (A2-A7)
#
# Three rooms in two scenes, sized 2 / 6 / 10 files so that all three ties-to-even cases of
# ``round`` are exercised (0.25*2=0.5 -> 0 -> clamped to 1; 0.25*6=1.5 -> 2; 0.25*10=2.5 -> 2;
# 0.75*6=4.5 -> 4; 0.75*10=7.5 -> 8). Every receiver carries >= 2 sources in the FULL room
# list, as the top-up contract requires. Scene keys, room keys and file lists are stored in
# DELIBERATELY SCRAMBLED order so the sorted-iteration contract is under test.
# ======================================================================================
def _f(src, rec):
    return f"{src}_{rec}_hybrid_IR.wav"


TOY_ROOM_A0 = [_f("S002", "R001"), _f("S001", "R001")]
TOY_ROOM_A1 = [_f(s, r) for r in ("R012", "R010", "R011") for s in ("S002", "S001")]
TOY_ROOM_B0 = (
    [_f(s, r) for r in ("R021", "R020") for s in ("S003", "S001", "S002")]
    + [_f(s, r) for r in ("R023", "R022") for s in ("S002", "S001")]
)
TOY_SPLIT = {
    "Beta": {"Beta_idx_0": list(TOY_ROOM_B0)},
    "Alpha": {"Alpha_idx_1": list(TOY_ROOM_A1), "Alpha_idx_0": list(TOY_ROOM_A0)},
}


class SpyRandom:
    """``random.Random`` wrapper that records RNG consumption and forbids every other draw."""

    def __init__(self, seed):
        self._rng = random.Random(seed)
        self.calls = []

    def sample(self, population, k):
        self.calls.append(("sample", tuple(population), k))
        return self._rng.sample(population, k)

    def __getattr__(self, name):  # pragma: no cover - only reached on a contract violation
        raise AssertionError(f"unexpected RNG consumption: rng.{name}")


# ======================================================================================
# A2 — room_permutation
# ======================================================================================
def test_A2_room_permutation_matches_reference_draw_and_is_a_permutation():
    files = sorted(TOY_ROOM_B0)
    perm = mas.room_permutation(files, random.Random(7))
    assert perm == random.Random(7).sample(files, len(files))
    assert sorted(perm) == files
    assert files == sorted(TOY_ROOM_B0)  # input list untouched


def test_A2_room_permutation_is_deterministic_for_a_seed():
    files = sorted(TOY_ROOM_B0)
    assert mas.room_permutation(files, random.Random(7)) == mas.room_permutation(files, random.Random(7))
    assert mas.room_permutation(files, random.Random(7)) != mas.room_permutation(files, random.Random(8))


def test_A2_room_permutation_consumes_exactly_one_sample_call():
    files = sorted(TOY_ROOM_B0)
    spy = SpyRandom(7)
    perm = mas.room_permutation(files, spy)
    assert spy.calls == [("sample", tuple(files), len(files))]
    assert perm == random.Random(7).sample(files, len(files))


# ======================================================================================
# A3 — raw_prefix
# ======================================================================================
@pytest.mark.parametrize(
    "n, frac, expected_k",
    [
        (1, 0.25, 1),    # round(0.25) = 0 -> clamped to the 1-file minimum
        (2, 0.25, 1),    # round(0.5)  = 0 (ties-to-even) -> clamped to 1
        (6, 0.25, 2),    # round(1.5)  = 2 (ties-to-even)
        (10, 0.25, 2),   # round(2.5)  = 2 (ties-to-even)
        (4, 0.25, 1),    # exact 1.0
        (2, 0.5, 1),
        (6, 0.5, 3),
        (10, 0.5, 5),
        (2, 0.75, 2),    # round(1.5) = 2
        (6, 0.75, 4),    # round(4.5) = 4 (ties-to-even)
        (10, 0.75, 8),   # round(7.5) = 8 (ties-to-even)
        (1600, 0.25, 400),
    ],
)
def test_A3_raw_prefix_sizes(n, frac, expected_k):
    perm = [f"S{i:04d}_R0001_hybrid_IR.wav" for i in range(n)]
    out = mas.raw_prefix(perm, frac)
    assert len(out) == expected_k
    assert out == perm[:expected_k]           # it is a prefix, in permutation order
    assert perm == [f"S{i:04d}_R0001_hybrid_IR.wav" for i in range(n)]  # input untouched


def test_A3_raw_prefix_is_nested_across_ascending_fractions():
    perm = mas.room_permutation(sorted(TOY_ROOM_B0), random.Random(7))
    p25 = mas.raw_prefix(perm, 0.25)
    p50 = mas.raw_prefix(perm, 0.5)
    p75 = mas.raw_prefix(perm, 0.75)
    assert p50[: len(p25)] == p25
    assert p75[: len(p50)] == p50
