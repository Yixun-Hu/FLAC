"""Placement clustering + microphone correspondence (exp_21, contract A, cycle 1).

This is the scientific foundation of Mapping A: an item claims that ONE microphone
heard several sources, so "the same microphone across tx-groups" has to be a
measured fact, not an assumption. The inherited `_placement_key` (a receiver
centroid rounded to 1 cm, explicitly informational) can split one re-occupation
across adjacent rounding bins or merge two placements, and a distance cutoff alone
never proves a one-to-one 36-way correspondence -- Codex M2.

The oracle below is a rigid 36-mic lattice permuted by a KNOWN permutation and
perturbed by KNOWN noise, so both the recovered assignment and the ambiguity margin
are checkable by hand.
"""
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RAF_DIR = os.path.join(_REPO_ROOT, "data", "RAF")
if _RAF_DIR not in sys.path:
    sys.path.insert(0, _RAF_DIR)

import mappingA_common as mac  # noqa: E402

assert os.path.dirname(os.path.abspath(mac.__file__)) == _RAF_DIR


# --------------------------------------------------------------------------- #
# fixtures: a rigid array with hand-known geometry
# --------------------------------------------------------------------------- #
def rigid_array():
    """36 mics on a 3 x 3 x 4 lattice, spacing 0.5 / 0.5 / 0.4 m.

    Nearest-neighbour spacing is 0.4 m, which is what makes the ambiguity margin
    hand-checkable: a 3 mm displacement sits ~133x below it.
    """
    pts = [(x, y, z) for x in (0.0, 0.5, 1.0) for y in (0.0, 0.5, 1.0)
           for z in (0.0, 0.4, 0.8, 1.2)]
    assert len(pts) == 36
    return np.array(pts, dtype=np.float64)


def permuted(array, permutation, noise=None):
    """``group_rx[i] = template_rx[permutation[i]] + noise[i]``."""
    out = array[list(permutation)].copy()
    if noise is not None:
        out = out + noise
    return out


def _rng_noise(n, scale, seed=0):
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return direction * scale


# --------------------------------------------------------------------------- #
# match_mics -- the correspondence itself
# --------------------------------------------------------------------------- #
def test_hungarian_recovers_a_known_permutation_exactly():
    """The whole claim in one test: shuffle the array, perturb it by 3 mm, and the
    assignment must undo the shuffle for all 36 mics."""
    template = rigid_array()
    rng = np.random.default_rng(21)
    permutation = rng.permutation(36)
    group = permuted(template, permutation, _rng_noise(36, 0.003, seed=1))

    report = mac.match_mics(template, group)
    assert report["passed"] is True
    assert report["assignment"] == [int(np.where(permutation == i)[0][0])
                                    for i in range(36)]
    # every template slot maps to the group row holding ITS point
    for slot, row in enumerate(report["assignment"]):
        assert permutation[row] == slot


def test_the_assignment_is_a_permutation():
    template = rigid_array()
    group = permuted(template, np.random.default_rng(3).permutation(36),
                     _rng_noise(36, 0.002, seed=2))
    report = mac.match_mics(template, group)
    assert sorted(report["assignment"]) == list(range(36))


def test_displacements_are_reported_per_slot_and_summarised():
    template = rigid_array()
    group = permuted(template, range(36), _rng_noise(36, 0.003, seed=4))
    report = mac.match_mics(template, group)
    assert len(report["displacements_m"]) == 36
    assert all(abs(d - 0.003) < 1e-9 for d in report["displacements_m"])
    assert report["p50_m"] == pytest.approx(0.003)
    assert report["p95_m"] == pytest.approx(0.003)
    assert report["max_m"] == pytest.approx(0.003)


def test_identical_arrays_match_with_zero_displacement():
    template = rigid_array()
    report = mac.match_mics(template, template.copy())
    assert report["passed"] is True
    assert report["max_m"] == pytest.approx(0.0)
    assert report["assignment"] == list(range(36))
    # a zero displacement is infinitely unambiguous, not a division by zero
    assert report["min_ambiguity_margin"] == float("inf")


def test_the_ambiguity_margin_is_hand_checkable():
    """Three collinear mics 0.30 m apart, all displaced +0.03 m along the axis, so
    group points sit at x = 0.03 / 0.33 / 0.63.

    Slot 0 (x=0):    nearest 0.03, second 0.33 -> margin 11
    Slot 1 (x=0.30): nearest 0.03, second 0.27 -> margin  9   <- the INTERIOR mic
    Slot 2 (x=0.60): nearest 0.03, second 0.27 -> margin  9
    The reported minimum is therefore 9, from a mic with a neighbour on each side --
    an end mic always looks less ambiguous than it is.

    The 3 cm displacement is itself outside the hard cap, so this group is refused
    on THAT ground while the margin stays comfortable: the two gates are
    independent, and this test is about the margin arithmetic.
    """
    template = np.array([[0.0, 0, 0], [0.30, 0, 0], [0.60, 0, 0]])
    group = template + np.array([0.03, 0.0, 0.0])
    report = mac.match_mics(template, group, expected_n=3)
    assert report["assignment"] == [0, 1, 2]
    assert report["displacements_m"][0] == pytest.approx(0.03)
    assert report["ambiguity_margins"][0] == pytest.approx(11.0)
    assert report["ambiguity_margins"][1] == pytest.approx(9.0)
    assert report["min_ambiguity_margin"] == pytest.approx(9.0)
    assert report["passed"] is False
    assert any("max displacement" in r for r in report["reasons"])
    assert not any("ambiguity" in r for r in report["reasons"])


def test_an_ambiguous_pair_fails_even_with_a_small_displacement():
    """Two mics only 6 mm apart, displaced 3 mm: group points at 0.003 / 0.009.

    Slot 0 (x=0):     nearest 0.003, second 0.009 -> margin 3
    Slot 1 (x=0.006): the two group points are BOTH 0.003 away -> margin 1
    The displacement is well inside every distance tolerance, yet which mic is
    which is a coin flip -- only the margin sees that.
    """
    template = np.array([[0.0, 0, 0], [0.006, 0, 0], [0.60, 0, 0]])
    group = template + np.array([0.003, 0.0, 0.0])
    report = mac.match_mics(template, group, expected_n=3)
    assert report["ambiguity_margins"][0] == pytest.approx(3.0)
    assert report["min_ambiguity_margin"] == pytest.approx(1.0)
    assert report["passed"] is False
    assert any("ambiguity" in r for r in report["reasons"])


@pytest.mark.parametrize("scale,needle", [
    (0.015, "p95"),      # 1.5 cm everywhere: p95 above the 1 cm bound
    (0.05, "p95"),       # 5 cm: far outside
])
def test_a_displaced_array_fails_the_p95_bound(scale, needle):
    template = rigid_array()
    group = permuted(template, range(36), _rng_noise(36, scale, seed=5))
    report = mac.match_mics(template, group)
    assert report["passed"] is False
    assert any(needle in r for r in report["reasons"])


def test_a_single_anomalous_mic_fails_the_hard_cap():
    """p95 can stay clean while ONE mic is 2.5 cm out; the hard cap catches it."""
    template = rigid_array()
    noise = _rng_noise(36, 0.002, seed=6)
    noise[7] = np.array([0.025, 0.0, 0.0])
    group = permuted(template, range(36), noise)
    report = mac.match_mics(template, group)
    assert report["p95_m"] < 0.01
    assert report["max_m"] == pytest.approx(0.025)
    assert report["passed"] is False
    assert any("max" in r for r in report["reasons"])


def test_the_registered_tolerances_are_the_planned_ones():
    assert mac.MATCH_P95_M == 0.01
    assert mac.MATCH_MAX_M == 0.02
    assert mac.MATCH_AMBIGUITY_MARGIN == 3.0
    assert mac.PLACEMENT_CAP_M == 0.05
    assert mac.MATCH_ALGORITHM_VERSION


@pytest.mark.parametrize("bad", [
    (np.zeros((35, 3)), np.zeros((36, 3))),
    (np.zeros((36, 3)), np.zeros((36, 2))),
    (np.zeros((36, 3)), np.full((36, 3), np.nan)),
])
def test_match_mics_rejects_malformed_input(bad):
    with pytest.raises(ValueError):
        mac.match_mics(bad[0], bad[1])


def test_match_mics_is_deterministic():
    template = rigid_array()
    group = permuted(template, np.random.default_rng(9).permutation(36),
                     _rng_noise(36, 0.004, seed=7))
    first = mac.match_mics(template, group)
    np.random.seed(1234)
    second = mac.match_mics(template, group)
    assert first["assignment"] == second["assignment"]
    assert first["displacements_m"] == second["displacements_m"]


# --------------------------------------------------------------------------- #
# cluster_placements -- which tx-groups share one physical placement
# --------------------------------------------------------------------------- #
def _group(key, centre, permutation=None, noise_scale=0.0, seed=0):
    array = rigid_array() + np.asarray(centre, dtype=np.float64)
    if permutation is not None:
        array = permuted(array, permutation,
                         _rng_noise(36, noise_scale, seed) if noise_scale else None)
    return {"group_key": key, "rx_xyz_p": array,
            "rx_centroid_p": array.mean(axis=0)}


def test_re_occupations_of_one_placement_cluster_together():
    groups = [_group("g0", (0, 0, 0)),
              _group("g1", (0.01, 0, 0)),      # 1 cm away
              _group("g2", (0.02, 0.01, 0)),   # ~2.2 cm from g0
              _group("g3", (5.0, 0, 0))]       # a different placement entirely
    clusters = mac.cluster_placements(groups)
    assert len(clusters) == 2
    members = {c["placement_id"]: sorted(c["member_keys"]) for c in clusters}
    assert sorted(members.values()) == [["g0", "g1", "g2"], ["g3"]]


def test_complete_linkage_refuses_transitive_chaining():
    """g0-g1 and g1-g2 are each 4 cm apart, but g0-g2 is 8 cm: single linkage would
    chain them into one placement, complete linkage must not."""
    groups = [_group("g0", (0, 0, 0)),
              _group("g1", (0.04, 0, 0)),
              _group("g2", (0.08, 0, 0))]
    clusters = mac.cluster_placements(groups)
    assert len(clusters) == 2
    sizes = sorted(len(c["member_keys"]) for c in clusters)
    assert sizes == [1, 2]


def test_the_cap_is_the_registered_five_centimetres():
    just_inside = [_group("a", (0, 0, 0)), _group("b", (0.049, 0, 0))]
    just_outside = [_group("a", (0, 0, 0)), _group("b", (0.051, 0, 0))]
    assert len(mac.cluster_placements(just_inside)) == 1
    assert len(mac.cluster_placements(just_outside)) == 2


def test_each_cluster_gets_a_deterministic_medoid_template():
    groups = [_group("g2", (0.02, 0, 0)), _group("g0", (0, 0, 0)),
              _group("g1", (0.01, 0, 0))]
    cluster = mac.cluster_placements(groups)[0]
    assert cluster["medoid_key"] == "g1"          # the middle occupation
    assert cluster["template_rx"].shape == (36, 3)
    np.testing.assert_allclose(cluster["template_rx"],
                               next(g["rx_xyz_p"] for g in groups
                                    if g["group_key"] == "g1"))
    # re-running on a shuffled input gives the same answer
    again = mac.cluster_placements(list(reversed(groups)))[0]
    assert again["medoid_key"] == cluster["medoid_key"]
    assert again["placement_id"] == cluster["placement_id"]


def test_placement_ids_are_stable_and_ordered():
    groups = [_group("z", (5.0, 0, 0)), _group("a", (0, 0, 0))]
    clusters = mac.cluster_placements(groups)
    assert [c["placement_id"] for c in clusters] == ["p000", "p001"]
    assert clusters[0]["member_keys"] == ["a"]     # ordered by first member key


def test_clustering_rejects_groups_without_receiver_arrays():
    with pytest.raises(ValueError):
        mac.cluster_placements([{"group_key": "g0"}])


# --------------------------------------------------------------------------- #
# the two together: a placement whose members are permuted re-occupations
# --------------------------------------------------------------------------- #
def test_a_placement_of_permuted_re_occupations_matches_end_to_end():
    rng = np.random.default_rng(11)
    permutations = [rng.permutation(36) for _ in range(3)]
    # Re-occupations of one placement sit sub-centimetre apart (exp_19 measured
    # "re-occupied to sub-cm"), which is what the 1 cm p95 bound is calibrated for.
    groups = [_group("g0", (0, 0, 0))]
    for i, permutation in enumerate(permutations, start=1):
        groups.append(_group(f"g{i}", (0.002 * i, 0, 0), permutation=permutation,
                             noise_scale=0.002, seed=20 + i))
    cluster = mac.cluster_placements(groups)[0]
    assert len(cluster["member_keys"]) == 4

    template = cluster["template_rx"]
    for group in groups:
        report = mac.match_mics(template, group["rx_xyz_p"])
        assert report["passed"] is True, group["group_key"]
        assert sorted(report["assignment"]) == list(range(36))
        assert report["max_m"] < mac.MATCH_MAX_M


# --------------------------------------------------------------------------- #
# cycle 2: target selection (hash-uniform) and per-item context
# --------------------------------------------------------------------------- #
import hashlib  # noqa: E402


def _pose(key, xyz, quat=(0.0, 0.9, 0.0, 0.1)):
    return {"group_key": key, "tx_xyz": np.array(xyz, dtype=np.float64),
            "quat_canon": np.array(quat, dtype=np.float64)}


def test_target_digest_matches_an_independent_sha256_golden():
    """Golden from GNU coreutils, outside this codebase:

        $ printf 'mappingA-target|0|EmptyRoom|p003|g7f2a1' | sha256sum
        5e6b2e8ad7d21d30...
    """
    assert mac.target_digest("EmptyRoom", "p003", "g7f2a1").startswith(
        "5e6b2e8ad7d21d30")


def test_select_target_is_hash_uniform_and_stable():
    poses = [_pose(f"g{i:04d}", (i * 1.0, 1.5, 0.0)) for i in range(20)]
    chosen = mac.select_target("EmptyRoom", "p003", poses)
    assert chosen in poses
    # the winner is the smallest digest -- recomputed here from the rule, not from
    # the implementation's own bookkeeping
    expected = min(poses, key=lambda p: hashlib.sha256(
        f"mappingA-target|0|EmptyRoom|p003|{p['group_key']}".encode()).hexdigest())
    assert chosen["group_key"] == expected["group_key"]
    assert mac.select_target("EmptyRoom", "p003", list(reversed(poses)))[
        "group_key"] == chosen["group_key"]


def test_select_target_is_not_a_positional_or_spatial_rule():
    """M8: the estimand is general unseen-source performance, so the target must
    not be the first, the last, or the most extreme pose."""
    poses = [_pose(f"g{i:04d}", (i * 1.0, 1.5, 0.0)) for i in range(20)]
    chosen = {mac.select_target("EmptyRoom", f"p{i:03d}", poses)["group_key"]
              for i in range(30)}
    assert len(chosen) > 5                       # varies across placements
    assert chosen != {"g0000"} and chosen != {"g0019"}


def test_select_target_varies_with_room_placement_and_seed():
    poses = [_pose(f"g{i:04d}", (i * 1.0, 1.5, 0.0)) for i in range(40)]
    base = mac.select_target("EmptyRoom", "p003", poses)["group_key"]
    assert mac.select_target("FurnishedRoom", "p003", poses)["group_key"] != base
    assert mac.select_target("EmptyRoom", "p004", poses)["group_key"] != base
    assert mac.select_target("EmptyRoom", "p003", poses, seed=1)["group_key"] != base


def test_select_target_needs_a_pose():
    with pytest.raises(ValueError):
        mac.select_target("EmptyRoom", "p003", [])


def test_context_excludes_every_group_sharing_the_target_source_xyz():
    """M5: unseen SOURCE POSITION, not merely unseen pose -- a quaternion-only
    duplicate is the same loudspeaker in the same place."""
    target = _pose("gT", (1.0, 1.5, 2.0), quat=(0.1, 0.9, 0.0, 0.1))
    twin = _pose("gTWIN", (1.0, 1.5, 2.0), quat=(0.9, 0.1, 0.0, 0.1))  # same xyz
    others = [_pose(f"g{i}", (float(i), 1.5, 0.0)) for i in range(12)]
    context = mac.stable_item_context("EmptyRoom", "p003", 17, target,
                                      [target, twin] + others, k=8)
    keys = [c["group_key"] for c in context["context"]]
    assert "gT" not in keys
    assert "gTWIN" not in keys
    assert len(keys) == 8 and len(set(keys)) == 8
    assert context["n_excluded_same_xyz"] == 2


def test_context_is_deterministic_per_item_and_ignores_ambient_rng():
    target = _pose("gT", (1.0, 1.5, 2.0))
    others = [_pose(f"g{i}", (float(i), 1.5, 0.0)) for i in range(12)]
    pool = [target] + others
    np.random.seed(0)
    first = mac.stable_item_context("EmptyRoom", "p003", 17, target, pool, k=8)
    np.random.seed(99)
    second = mac.stable_item_context("EmptyRoom", "p003", 17, target,
                                     list(reversed(pool)), k=8)
    assert [c["group_key"] for c in first["context"]] == \
        [c["group_key"] for c in second["context"]]


def test_context_differs_per_mic_slot_and_per_target():
    target = _pose("gT", (1.0, 1.5, 2.0))
    others = [_pose(f"g{i}", (float(i), 1.5, 0.0)) for i in range(20)]
    pool = [target] + others
    a = mac.stable_item_context("EmptyRoom", "p003", 17, target, pool, k=8)
    b = mac.stable_item_context("EmptyRoom", "p003", 18, target, pool, k=8)
    c = mac.stable_item_context("EmptyRoom", "p003", 17, others[0],
                                pool, k=8)
    assert [x["group_key"] for x in a["context"]] != [x["group_key"] for x in b["context"]]
    assert [x["group_key"] for x in a["context"]] != [x["group_key"] for x in c["context"]]


def test_context_seed_matches_an_independent_sha256_golden():
    """    $ printf 'mappingA-context|0|EmptyRoom|p003|17|gTARGET' | sha256sum
        f8500499fa60973d...
    """
    assert mac.context_digest("EmptyRoom", "p003", 17, "gTARGET").startswith(
        "f8500499fa60973d")


def test_context_pool_too_small_fails_closed():
    target = _pose("gT", (1.0, 1.5, 2.0))
    others = [_pose(f"g{i}", (float(i), 1.5, 0.0)) for i in range(5)]
    with pytest.raises(ValueError) as exc:
        mac.stable_item_context("EmptyRoom", "p003", 17, target, [target] + others, k=8)
    assert "5" in str(exc.value)


def test_source_xyz_key_is_the_six_decimal_rendering():
    assert mac.source_xyz_key([1.0, -0.0, 2.5]) == "1.000000,0.000000,2.500000"
    assert mac.source_xyz_key([1.0, 0.0, 2.5]) == mac.source_xyz_key([1.0, -0.0, 2.5])
    assert mac.source_xyz_key([1.0, 0.0, 2.5]) != mac.source_xyz_key([1.000001, 0.0, 2.5])


# --------------------------------------------------------------------------- #
# r2 N9: rigid residual, per-slot evidence, duplicates, degenerate margins
# --------------------------------------------------------------------------- #
def _rotation_z(degrees):
    angle = np.radians(degrees)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_a_rigid_motion_has_no_residual():
    """One rotation and one translation explain the whole array -- the definition
    of the re-occupied rig the same-microphone claim assumes."""
    template = rigid_array()
    rotation = _rotation_z(7.0)
    moved = (rotation @ template.T).T + np.array([1.0, -2.0, 0.5])
    residual = mac.rigid_residual(template, moved)
    assert residual["rms_m"] < 1e-12 and residual["max_m"] < 1e-12
    assert abs(residual["rotation_deg"] - 7.0) < 1e-9
    assert abs(residual["translation_m"] - np.linalg.norm([1.0, -2.0, 0.5])) < 1e-9


def test_a_deformed_array_has_a_residual_the_per_mic_gates_cannot_see():
    """Every displacement is inside tolerance, yet no rigid motion maps the set
    onto the template: that is exactly what the residual is for."""
    template = rigid_array()
    group = template.copy()
    group[5] += np.array([0.0, 0.0, 0.006])       # 6 mm on one mic only
    report = mac.match_mics(template, group)
    assert report["max_m"] <= mac.MATCH_MAX_M     # per-mic gates are clean
    assert report["p95_m"] <= mac.MATCH_P95_M
    assert report["rigid_residual_rms_m"] > 1e-4
    assert report["rigid_residual_max_m"] > 5e-3
    assert report["passed"] is True               # RECORDED, not gated


def test_a_mirror_image_is_not_reported_as_a_rigid_fit():
    template = rigid_array()
    mirrored = template * np.array([1.0, 1.0, -1.0])
    residual = mac.rigid_residual(template, mirrored)
    assert residual["rms_m"] > 0.1


def test_duplicate_receiver_positions_are_refused():
    template = rigid_array()
    group = template.copy()
    group[7] = group[3]                            # two mics at one point
    report = mac.match_mics(template, group)
    assert report["passed"] is False
    assert any("duplicate" in reason for reason in report["reasons"])
    assert report["duplicate_positions"]["group"][0]["i"] == 3
    assert report["duplicate_positions"]["group"][0]["j"] == 7
    assert report["duplicate_positions"]["template"] == []


def test_a_duplicated_template_mic_is_refused_too():
    template = rigid_array()
    template[2] = template[1]
    report = mac.match_mics(template, rigid_array())
    assert report["passed"] is False
    assert any("template holds" in reason for reason in report["reasons"])


def test_two_mics_on_one_slot_is_the_most_ambiguous_case_not_the_least():
    """matched == second == 0 used to read as an infinite margin -- the most
    decisive verdict available -- for the one configuration that is a coin flip."""
    template = rigid_array()
    group = template.copy()
    group[9] = group[8]                            # both sit on template slot 8
    report = mac.match_mics(template, group)
    assert report["min_ambiguity_margin"] == 0.0
    assert report["passed"] is False
    reasons = " ".join(report["reasons"])
    assert "duplicate" in reasons and "ambiguity margin" in reasons


def test_an_exact_hit_with_a_distant_neighbour_is_still_unambiguous():
    report = mac.match_mics(rigid_array(), rigid_array())
    assert report["min_ambiguity_margin"] == float("inf")
    assert report["passed"] is True


def test_the_evidence_digest_names_the_assignment_not_its_summary():
    """Two matches can share p95, max and margin exactly and still be different
    correspondences; the digest is over the per-slot assignment itself."""
    template = rigid_array()
    identity = mac.match_mics(template, template)

    permutation = list(range(36))
    permutation[0], permutation[1] = permutation[1], permutation[0]
    relabelled = mac.match_mics(template, permuted(template, permutation))
    assert relabelled["p95_m"] == identity["p95_m"]
    assert relabelled["max_m"] == identity["max_m"]
    assert relabelled["min_ambiguity_margin"] == identity["min_ambiguity_margin"]
    assert relabelled["assignment"] != identity["assignment"]
    assert relabelled["evidence_sha256"] != identity["evidence_sha256"]
    # ... and it is stable: the same inputs digest to the same value
    assert mac.match_mics(template, template)["evidence_sha256"] == \
        identity["evidence_sha256"]
    assert len(identity["evidence_sha256"]) == 64
    assert len(identity["per_slot_evidence"]) == 36


def test_the_algorithm_version_moved_with_the_algorithm():
    """A record produced under -1 describes a different computation: no rigid
    residual, no evidence digest, duplicates matched and zero/zero margins read as
    infinite."""
    assert mac.MATCH_ALGORITHM_VERSION == "mappingA-correspondence-2"
