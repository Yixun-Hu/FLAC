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
import copy
import hashlib
import json
import os
import platform
import random
import types

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
        "Sabc_Rxyz_hybrid_IR.wav",       # non-numeric ids
        "SS_RR_tail",                    # non-numeric ids
        "S_R1_x",                        # empty source id
        "S1_R_x",                        # empty receiver id
        "S12a_R34_x",                    # trailing non-digit in the source id
        "S12_R3-4_x",                    # non-digit in the receiver id
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


# ======================================================================================
# A4 — topup_zero_context
#
# ``PERM_B0`` is the seed-7 permutation of the 10-file toy room (see A7 for the derivation);
# it is hard-coded here so these tests exercise the top-up rule alone, with no RNG involved.
# Receivers in that room: R020 {S001,S002,S003}, R021 {S001,S002,S003}, R022 {S001,S002},
# R023 {S001,S002}.
# ======================================================================================
PERM_B0 = [
    _f("S003", "R021"),  # 0
    _f("S001", "R020"),  # 1
    _f("S001", "R023"),  # 2
    _f("S003", "R020"),  # 3
    _f("S002", "R022"),  # 4
    _f("S002", "R023"),  # 5
    _f("S002", "R020"),  # 6
    _f("S002", "R021"),  # 7
    _f("S001", "R022"),  # 8
    _f("S001", "R021"),  # 9
]


def test_A4_topup_adds_the_earliest_same_receiver_other_source_entry():
    """Both prefix entries are starved; each gets the EARLIEST permutation entry that shares
    its receiver and carries a different source — for R020 that is index 3 (S003), not the
    lower-numbered source S002 which sits at index 6."""
    selected = {PERM_B0[0], PERM_B0[1]}
    out, added = mas.topup_zero_context(selected, PERM_B0)
    assert added == [_f("S002", "R021"), _f("S003", "R020")]  # walk order = permutation order
    assert out == {PERM_B0[0], PERM_B0[1], _f("S002", "R021"), _f("S003", "R020")}
    assert selected == {PERM_B0[0], PERM_B0[1]}  # caller's set never mutated


def test_A4_topup_leaves_non_starved_selection_unchanged():
    perm = [_f("S002", "R001"), _f("S001", "R001")]
    selected = {perm[0], perm[1]}
    out, added = mas.topup_zero_context(selected, perm)
    assert added == []
    assert out == selected


def test_A4_topup_never_re_adds_an_already_selected_entry():
    out, added = mas.topup_zero_context(set(PERM_B0), PERM_B0)
    assert added == []
    assert out == set(PERM_B0)
    # a mixed case: only the starved target triggers an addition, nothing already present
    selected = {_f("S001", "R020"), _f("S003", "R020"), _f("S002", "R022")}
    out, added = mas.topup_zero_context(selected, PERM_B0)
    assert added == [_f("S001", "R022")]
    assert not set(added) & selected
    assert out == selected | {_f("S001", "R022")}


def test_A4_topup_single_pass_resolves_later_targets_via_earlier_additions():
    """A receiver holding two selected entries of the SAME source is fixed by one addition:
    the second entry must not trigger a second (duplicate) top-up."""
    perm = ["S001_R001_a.wav", "S001_R001_b.wav", "S002_R001_c.wav"]
    out, added = mas.topup_zero_context({perm[0], perm[1]}, perm)
    assert added == ["S002_R001_c.wav"]
    assert out == set(perm)


def test_A4_topup_raises_when_the_full_room_has_no_other_source_at_that_receiver():
    """Contract violation (cannot happen on AR): fail loudly instead of emitting a starved
    target whose acoustic-context pool would be empty."""
    perm = [_f("S001", "R001"), _f("S001", "R002")]
    with pytest.raises(ValueError):
        mas.topup_zero_context({perm[0]}, perm)


def test_A4_topup_rejects_selected_entries_absent_from_the_permutation():
    with pytest.raises(ValueError):
        mas.topup_zero_context({_f("S009", "R099")}, PERM_B0)


# ======================================================================================
# A5a — context_histogram (eligible-context bins; the "< 8" bins are where FLAC's sampler
# has to draw the K=8 references WITH replacement)
# ======================================================================================
def test_A5_context_histogram_bins_by_number_of_other_sources():
    assert mas.context_histogram(set()) == {"0": 0, "1-7": 0, ">=8": 0}
    assert mas.context_histogram({_f("S001", "R001")}) == {"0": 1, "1-7": 0, ">=8": 0}
    two = {_f("S001", "R001"), _f("S002", "R001")}
    assert mas.context_histogram(two) == {"0": 0, "1-7": 2, ">=8": 0}
    eight = {_f(f"S{i:03d}", "R001") for i in range(1, 9)}
    assert mas.context_histogram(eight) == {"0": 0, "1-7": 8, ">=8": 0}   # 7 others each
    nine = {_f(f"S{i:03d}", "R001") for i in range(1, 10)}
    assert mas.context_histogram(nine) == {"0": 0, "1-7": 0, ">=8": 9}    # 8 others each


def test_A5_context_histogram_counts_distinct_sources_and_mixes_receivers():
    # same source twice at one receiver (different tails) is ONE other source, not two
    room = {"S001_R001_a.wav", "S001_R001_b.wav", "S002_R001_c.wav"}
    assert mas.context_histogram(room) == {"0": 0, "1-7": 3, ">=8": 0}
    mixed = {_f("S001", "R001"), _f("S002", "R001"), _f("S001", "R002")}
    assert mas.context_histogram(mixed) == {"0": 1, "1-7": 2, ">=8": 0}


# ======================================================================================
# A5 — build_subsets (properties; A7 pins the exact algorithm output)
# ======================================================================================
FRACS = [0.25, 0.5, 0.75]


def _rooms(split):
    return [(scene, room) for scene in split for room in split[scene]]


def test_A5_output_keys_match_the_input_split_in_sorted_order():
    subsets, _ = mas.build_subsets(TOY_SPLIT, FRACS, seed=7)
    assert sorted(subsets) == FRACS
    for frac in FRACS:
        out = subsets[frac]
        assert list(out) == sorted(TOY_SPLIT)                       # scenes, sorted
        for scene in out:
            assert list(out[scene]) == sorted(TOY_SPLIT[scene])     # rooms, sorted
        assert set(_rooms(out)) == set(_rooms(TOY_SPLIT))


def test_A5_every_retained_file_comes_from_the_input_and_rooms_are_sorted():
    subsets, _ = mas.build_subsets(TOY_SPLIT, FRACS, seed=7)
    for frac in FRACS:
        for scene, room in _rooms(subsets[frac]):
            files = subsets[frac][scene][room]
            assert files == sorted(files)
            assert len(files) == len(set(files))
            assert set(files) <= set(TOY_SPLIT[scene][room])
            assert len(files) >= 1


def test_A5_subsets_are_nested_and_grow_with_the_fraction():
    subsets, _ = mas.build_subsets(TOY_SPLIT, FRACS, seed=7)
    for scene, room in _rooms(TOY_SPLIT):
        s25 = set(subsets[0.25][scene][room])
        s50 = set(subsets[0.5][scene][room])
        s75 = set(subsets[0.75][scene][room])
        assert s25 <= s50 <= s75


def test_A5_manifest_counts_reconcile_per_fraction_and_per_room():
    subsets, manifest = mas.build_subsets(TOY_SPLIT, FRACS, seed=7)
    assert manifest["seed"] == 7
    assert list(manifest["fractions"]) == ["0.25", "0.5", "0.75"]
    for frac in FRACS:
        entry = manifest["fractions"][str(frac)]
        assert entry["final"] == entry["raw_prefix"] + entry["new_topup"] + entry["inherited_topup"]
        assert entry["final"] == sum(
            len(subsets[frac][scene][room]) for scene, room in _rooms(subsets[frac])
        )
        assert entry["effective_epochs_at_40k_x64"] == pytest.approx(40_000 * 64 / entry["final"])
        per_room = entry["per_room"]
        assert set(per_room) == {room for _, room in _rooms(TOY_SPLIT)}
        for scene, room in _rooms(TOY_SPLIT):
            rc = per_room[room]
            assert rc["final"] == rc["raw_prefix"] + rc["new_topup"] + rc["inherited_topup"]
            assert rc["final"] == len(subsets[frac][scene][room])
            assert rc["raw_prefix"] == len(
                mas.raw_prefix(
                    mas.room_permutation(sorted(TOY_SPLIT[scene][room]), random.Random(0)), frac
                )
            )
        for key in ("raw_prefix", "new_topup", "inherited_topup", "final"):
            assert entry[key] == sum(rc[key] for rc in per_room.values())


def test_A5_no_starved_target_survives_in_any_fraction():
    subsets, manifest = mas.build_subsets(TOY_SPLIT, FRACS, seed=7)
    for frac in FRACS:
        assert manifest["fractions"][str(frac)]["context_histogram"]["0"] == 0
        total = {"0": 0, "1-7": 0, ">=8": 0}
        for scene, room in _rooms(subsets[frac]):
            for bin_name, n in mas.context_histogram(set(subsets[frac][scene][room])).items():
                total[bin_name] += n
        assert total == manifest["fractions"][str(frac)]["context_histogram"]
        assert sum(total.values()) == manifest["fractions"][str(frac)]["final"]


def test_A5_consumes_exactly_one_sample_draw_per_room_and_no_other_rng(monkeypatch):
    spies = []

    def factory(seed):
        spy = SpyRandom(seed)
        spies.append(spy)
        return spy

    monkeypatch.setattr(mas, "random", types.SimpleNamespace(Random=factory))
    subsets, _ = mas.build_subsets(TOY_SPLIT, FRACS, seed=7)
    assert len(spies) == 1                       # ONE PRNG for the whole build
    expected = [
        ("sample", tuple(sorted(TOY_SPLIT[scene][room])), len(TOY_SPLIT[scene][room]))
        for scene in sorted(TOY_SPLIT)
        for room in sorted(TOY_SPLIT[scene])
    ]
    assert spies[0].calls == expected            # one draw per room, in sorted order
    assert subsets[0.25] == mas.build_subsets(TOY_SPLIT, FRACS, seed=7)[0][0.25]


def test_A5_is_deterministic_seed_dependent_and_order_independent():
    a, ma = mas.build_subsets(TOY_SPLIT, FRACS, seed=7)
    b, mb = mas.build_subsets(TOY_SPLIT, list(reversed(FRACS)), seed=7)
    assert a == b and ma == mb                   # fractions are sorted ascending internally
    c, _ = mas.build_subsets(TOY_SPLIT, FRACS, seed=8)
    assert c[0.25] != a[0.25]


def test_A5_does_not_mutate_the_input_split():
    before = copy.deepcopy(TOY_SPLIT)
    mas.build_subsets(TOY_SPLIT, FRACS, seed=7)
    assert TOY_SPLIT == before


def test_A5_manifest_records_provenance_placeholders():
    _, manifest = mas.build_subsets(TOY_SPLIT, FRACS, seed=7)
    assert manifest["python_version"] == platform.python_version()
    assert manifest["source_train_json_sha256"] is None   # filled in by write_outputs
    assert manifest["files"] == {}                        # ditto


@pytest.mark.parametrize(
    "split",
    [
        {"Alpha": {"Alpha_idx_0": []}},                                     # empty room
        {"Alpha": {"Alpha_idx_0": [_f("S001", "R001")] * 2}},               # duplicate file
        {"Alpha": {"R": [_f("S001", "R001"), _f("S002", "R001")]},
         "Beta": {"R": [_f("S001", "R001"), _f("S002", "R001")]}},          # room name clash
    ],
)
def test_A5_rejects_malformed_splits(split):
    with pytest.raises(ValueError):
        mas.build_subsets(split, FRACS, seed=7)


@pytest.mark.parametrize("fractions", [[], [0.0], [-0.25], [1.5], [0.25, 0.25]])
def test_A5_rejects_malformed_fractions(fractions):
    with pytest.raises(ValueError):
        mas.build_subsets(TOY_SPLIT, fractions, seed=7)


# ======================================================================================
# A6 — write_outputs (+ the CLI round trip, below)
# ======================================================================================
def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_toy_train_json(tmp_path):
    src = tmp_path / "train.json"
    src.write_text(json.dumps(TOY_SPLIT, indent=4), encoding="utf-8")
    return src


def test_A6_write_outputs_emits_named_splits_manifest_and_detached_checksums(tmp_path):
    src = _write_toy_train_json(tmp_path)
    subsets, manifest = mas.build_subsets(TOY_SPLIT, FRACS, seed=7)
    before = copy.deepcopy(manifest)
    out_dir = tmp_path / "out" / "nested"          # must be created by the tool
    paths = mas.write_outputs(str(out_dir), subsets, manifest, 7, str(src))

    names = ["train_frac025_s7.json", "train_frac050_s7.json", "train_frac075_s7.json"]
    assert sorted(p.name for p in out_dir.iterdir()) == sorted(
        names + ["train_frac_manifest_s7.json", "train_frac_manifest_s7.sha256"]
    )
    assert manifest == before                      # caller's manifest not mutated
    assert [os.path.basename(paths["splits"][f]) for f in FRACS] == names
    for frac, name in zip(FRACS, names):
        assert json.loads((out_dir / name).read_text(encoding="utf-8")) == subsets[frac]


def test_A6_manifest_hashes_every_emitted_split_but_not_itself(tmp_path):
    src = _write_toy_train_json(tmp_path)
    subsets, manifest = mas.build_subsets(TOY_SPLIT, FRACS, seed=7)
    mas.write_outputs(str(tmp_path / "out"), subsets, manifest, 7, str(src))
    out_dir = tmp_path / "out"
    written = json.loads((out_dir / "train_frac_manifest_s7.json").read_text(encoding="utf-8"))

    assert written["source_train_json_sha256"] == _sha256(src)
    assert set(written["files"]) == {
        "train_frac025_s7.json", "train_frac050_s7.json", "train_frac075_s7.json"
    }
    for name, digest in written["files"].items():
        assert digest == _sha256(out_dir / name)
    assert "train_frac_manifest_s7.json" not in written["files"]


def test_A6_detached_checksum_file_verifies_manifest_and_splits(tmp_path):
    src = _write_toy_train_json(tmp_path)
    subsets, manifest = mas.build_subsets(TOY_SPLIT, FRACS, seed=7)
    out_dir = tmp_path / "out"
    mas.write_outputs(str(out_dir), subsets, manifest, 7, str(src))

    lines = (out_dir / "train_frac_manifest_s7.sha256").read_text(encoding="utf-8").splitlines()
    covered = {}
    for line in lines:                             # `sha256sum -c` format: "<hash>  <name>"
        digest, name = line.split("  ", 1)
        assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
        covered[name] = digest
    assert set(covered) == {
        "train_frac025_s7.json", "train_frac050_s7.json", "train_frac075_s7.json",
        "train_frac_manifest_s7.json",
    }
    for name, digest in covered.items():           # recomputed here, not via the sha256sum binary
        assert digest == _sha256(out_dir / name)


def test_A6_rerun_is_byte_identical_and_seed_changes_the_bytes(tmp_path):
    src = _write_toy_train_json(tmp_path)
    blobs = []
    for run in ("a", "b"):
        subsets, manifest = mas.build_subsets(TOY_SPLIT, FRACS, seed=7)
        out_dir = tmp_path / run
        mas.write_outputs(str(out_dir), subsets, manifest, 7, str(src))
        blobs.append({p.name: p.read_bytes() for p in sorted(out_dir.iterdir())})
    assert blobs[0] == blobs[1]

    big = {"Scene": {"Scene_idx_0": [_f(f"S{s:03d}", f"R{r:03d}") for s in range(1, 5) for r in range(1, 11)]}}
    outs = {}
    for seed in (7, 8):
        subsets, manifest = mas.build_subsets(big, FRACS, seed=seed)
        out_dir = tmp_path / f"seed{seed}"
        mas.write_outputs(str(out_dir), subsets, manifest, seed, str(src))
        outs[seed] = {p.name: p.read_bytes() for p in sorted(out_dir.iterdir())}
    assert set(outs[7]) & set(outs[8]) == set()    # the seed is part of every filename
    assert outs[7]["train_frac025_s7.json"] != outs[8]["train_frac025_s8.json"]


def test_A6_write_outputs_rejects_fractions_without_a_whole_percent_tag(tmp_path):
    subsets, manifest = mas.build_subsets(TOY_SPLIT, [0.333], seed=7)
    with pytest.raises(ValueError):
        mas.write_outputs(str(tmp_path / "out"), subsets, manifest, 7, None)


# ======================================================================================
# A6 (continued) — the CLI round trip
# ======================================================================================
def test_A6_cli_round_trip_writes_the_same_artifacts_and_summarises(tmp_path, capsys):
    src = _write_toy_train_json(tmp_path)
    out_dir = tmp_path / "cli"
    rc = mas.main([
        "--train-json", str(src), "--out-dir", str(out_dir),
        "--fractions", "0.25,0.5,0.75", "--seed", "7",
    ])
    assert rc == 0

    subsets, manifest = mas.build_subsets(TOY_SPLIT, FRACS, seed=7)
    reference = tmp_path / "reference"
    mas.write_outputs(str(reference), subsets, manifest, 7, str(src))
    assert {p.name: p.read_bytes() for p in sorted(out_dir.iterdir())} == {
        p.name: p.read_bytes() for p in sorted(reference.iterdir())
    }

    printed = capsys.readouterr().out.splitlines()
    summary = [ln for ln in printed if "final=" in ln]
    assert len(summary) == 3                                   # one line per fraction
    for frac, line in zip(FRACS, summary):
        entry = manifest["fractions"][str(frac)]
        for token in (
            f"raw_prefix={entry['raw_prefix']}",
            f"new_topup={entry['new_topup']}",
            f"inherited_topup={entry['inherited_topup']}",
            f"final={entry['final']}",
            str(entry["context_histogram"]["0"]),
            str(entry["context_histogram"]["1-7"]),
        ):
            assert token in line


def test_A6_cli_defaults_are_the_frozen_fractions_and_seed(tmp_path):
    src = _write_toy_train_json(tmp_path)
    out_dir = tmp_path / "defaults"
    assert mas.main(["--train-json", str(src), "--out-dir", str(out_dir)]) == 0
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "train_frac025_s2026.json", "train_frac050_s2026.json", "train_frac075_s2026.json",
        "train_frac_manifest_s2026.json", "train_frac_manifest_s2026.sha256",
    ]


def test_A6_cli_never_touches_its_input(tmp_path):
    src = _write_toy_train_json(tmp_path)
    before = src.read_bytes()
    mas.main(["--train-json", str(src), "--out-dir", str(tmp_path / "o1"), "--seed", "7"])
    assert src.read_bytes() == before


# ======================================================================================
# A7 — hand-derived regression fixture (pins the ALGORITHM, not just its properties)
# ======================================================================================
EXPECTED_S7 = {
    0.25: {
        "Alpha": {
            "Alpha_idx_0": [_f("S001", "R001"), _f("S002", "R001")],
            "Alpha_idx_1": [_f("S001", "R010"), _f("S002", "R010")],
        },
        "Beta": {
            "Beta_idx_0": [
                _f("S001", "R020"), _f("S002", "R021"), _f("S003", "R020"), _f("S003", "R021"),
            ],
        },
    },
    0.5: {
        "Alpha": {
            "Alpha_idx_0": [_f("S001", "R001"), _f("S002", "R001")],
            "Alpha_idx_1": [
                _f("S001", "R010"), _f("S001", "R011"), _f("S002", "R010"), _f("S002", "R011"),
            ],
        },
        "Beta": {
            "Beta_idx_0": [
                _f("S001", "R020"), _f("S001", "R022"), _f("S001", "R023"), _f("S002", "R021"),
                _f("S002", "R022"), _f("S002", "R023"), _f("S003", "R020"), _f("S003", "R021"),
            ],
        },
    },
    0.75: {
        "Alpha": {
            "Alpha_idx_0": [_f("S001", "R001"), _f("S002", "R001")],
            "Alpha_idx_1": [
                _f("S001", "R010"), _f("S001", "R011"), _f("S001", "R012"),
                _f("S002", "R010"), _f("S002", "R011"), _f("S002", "R012"),
            ],
        },
        "Beta": {
            "Beta_idx_0": [
                _f("S001", "R020"), _f("S001", "R022"), _f("S001", "R023"), _f("S002", "R020"),
                _f("S002", "R021"), _f("S002", "R022"), _f("S002", "R023"), _f("S003", "R020"),
                _f("S003", "R021"),
            ],
        },
    },
}

# (raw_prefix, new_topup, inherited_topup, final) per room, hand-derived below.
EXPECTED_S7_PER_ROOM = {
    0.25: {"Alpha_idx_0": (1, 1, 0, 2), "Alpha_idx_1": (2, 0, 0, 2), "Beta_idx_0": (2, 2, 0, 4)},
    0.5: {"Alpha_idx_0": (1, 0, 1, 2), "Alpha_idx_1": (3, 1, 0, 4), "Beta_idx_0": (5, 2, 1, 8)},
    0.75: {"Alpha_idx_0": (2, 0, 0, 2), "Alpha_idx_1": (4, 1, 1, 6), "Beta_idx_0": (8, 0, 1, 9)},
}
EXPECTED_S7_TOTALS = {0.25: (5, 3, 0, 8), 0.5: (9, 3, 2, 14), 0.75: (14, 1, 2, 17)}
EXPECTED_S7_HIST = {
    0.25: {"0": 0, "1-7": 8, ">=8": 0},
    0.5: {"0": 0, "1-7": 14, ">=8": 0},
    0.75: {"0": 0, "1-7": 17, ">=8": 0},
}


def test_A7_regression_fixture_seed_7_matches_the_hand_derived_algorithm():
    """Hand derivation (only the three permutations come from the PRNG; everything after them
    is worked out by applying plan §3 by hand). Names abbreviated ``Sxxx_Ryyy``.

    Iteration order: scenes sorted -> Alpha, Beta; rooms sorted -> Alpha_idx_0, Alpha_idx_1,
    Beta_idx_0; one ``random.Random(7)`` consumed in that order, so

      perm(Alpha_idx_0) = [S002_R001, S001_R001]
      perm(Alpha_idx_1) = [S002_R010, S001_R010, S002_R011, S001_R012, S002_R012, S001_R011]
      perm(Beta_idx_0)  = [S003_R021, S001_R020, S001_R023, S003_R020, S002_R022,
                           S002_R023, S002_R020, S002_R021, S001_R022, S001_R021]

    Alpha_idx_0 (n=2; receiver R001 carries S001, S002)
      f=.25  k=max(1, round(0.5))=1 -> prefix [S002_R001]. S002_R001 is the only entry at R001
             -> starved; earliest R001 entry with another source = index 1 (S001_R001) -> add.
             raw 1, new 1, inherited 0, final 2.
      f=.50  k=round(1.0)=1 -> prefix [S002_R001]; union with S_25 = both; nobody starved.
             raw 1, new 0, inherited 1 (S001_R001 comes only from S_25), final 2.
      f=.75  k=round(1.5)=2 (ties-to-even) -> prefix = both. raw 2, new 0, inherited 0, final 2.

    Alpha_idx_1 (n=6; R010/R011/R012 each carry S001, S002)
      f=.25  k=round(1.5)=2 -> prefix [S002_R010, S001_R010]; R010 already has two sources ->
             no top-up. raw 2, new 0, inherited 0, final 2.
      f=.50  k=3 -> prefix [S002_R010, S001_R010, S002_R011]; union adds nothing new.
             Walk: S002_R011 is alone at R011 -> earliest other source at R011 is index 5
             (S001_R011) -> add; when the walk reaches index 5 that entry is no longer starved.
             raw 3, new 1, inherited 0, final 4.
      f=.75  k=round(4.5)=4 (ties-to-even) -> prefix [S002_R010, S001_R010, S002_R011,
             S001_R012]; union with S_50 re-adds S001_R011 (inherited). Walk: S001_R012 alone at
             R012 -> earliest other source at R012 is index 4 (S002_R012) -> add.
             raw 4, new 1, inherited 1, final 6 (the whole room).

    Beta_idx_0 (n=10; R020 {S001,S002,S003}, R021 {S001,S002,S003}, R022 {S001,S002},
                R023 {S001,S002})
      f=.25  k=round(2.5)=2 (ties-to-even) -> prefix [S003_R021, S001_R020]. Walk in
             permutation order: S003_R021 alone at R021 -> earliest other source at R021 is
             index 7 (S002_R021) -> add. S001_R020 alone at R020 -> earliest other source at
             R020 is index 3 (S003_R020), NOT index 6 (S002_R020) -> add.
             raw 2, new 2, inherited 0, final 4.
      f=.50  k=5 -> prefix indices 0-4 [S003_R021, S001_R020, S001_R023, S003_R020, S002_R022];
             union with S_25 inherits S002_R021 (index 7). Walk: S001_R023 alone at R023 ->
             add index 5 (S002_R023); S002_R022 alone at R022 -> add index 8 (S001_R022).
             raw 5, new 2, inherited 1, final 8.
      f=.75  k=round(7.5)=8 (ties-to-even) -> prefix indices 0-7; union with S_50 inherits
             S001_R022 (index 8). Every receiver now holds >= 2 sources -> no top-up. Only
             index 9 (S001_R021) is left out. raw 8, new 0, inherited 1, final 9.

    Totals: .25 -> 5/3/0/8, .50 -> 9/3/2/14, .75 -> 14/1/2/17. Every retained target has 1 or 2
    other retained sources at its receiver, so the eligible-context histogram is entirely in the
    "1-7" bin: 8 / 14 / 17 entries, and the "0" bin is empty at every fraction.
    """
    subsets, manifest = mas.build_subsets(TOY_SPLIT, FRACS, seed=7)

    for frac in FRACS:
        assert subsets[frac] == EXPECTED_S7[frac]
        assert list(subsets[frac]) == ["Alpha", "Beta"]
        assert list(subsets[frac]["Alpha"]) == ["Alpha_idx_0", "Alpha_idx_1"]

        entry = manifest["fractions"][str(frac)]
        assert tuple(entry[key] for key in mas.COUNT_KEYS) == EXPECTED_S7_TOTALS[frac]
        assert entry["context_histogram"] == EXPECTED_S7_HIST[frac]
        for room, expected in EXPECTED_S7_PER_ROOM[frac].items():
            assert tuple(entry["per_room"][room][key] for key in mas.COUNT_KEYS) == expected


def test_A7_regression_fixture_nesting_and_full_coverage():
    """The hand-derived expectation is itself nested, and at 75 % the two small rooms are fully
    covered while Beta_idx_0 keeps 9 of its 10 entries."""
    for scene, room in _rooms(TOY_SPLIT):
        s25 = set(EXPECTED_S7[0.25][scene][room])
        s50 = set(EXPECTED_S7[0.5][scene][room])
        s75 = set(EXPECTED_S7[0.75][scene][room])
        assert s25 <= s50 <= s75 <= set(TOY_SPLIT[scene][room])
    assert set(EXPECTED_S7[0.75]["Alpha"]["Alpha_idx_1"]) == set(TOY_ROOM_A1)
    assert set(TOY_ROOM_B0) - set(EXPECTED_S7[0.75]["Beta"]["Beta_idx_0"]) == {_f("S001", "R021")}
