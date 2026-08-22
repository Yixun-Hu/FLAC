"""Two publication FLAVORS on one tree (exp_21, contract B5, cycle 6).

Mapping A publishes its own audio, metadata and depth into disjoint roots, so the
same corpus directory can carry a Mapping-H publication and a Mapping-A one at the
same time. exp_19 r4-T4 is the cautionary history: a shared marker name let a
prepare rerun rename the depth attestation aside and orphan evidence it never
regenerated. Flavor-scoped kinds keep the two independent, and the four
composition cases below are the proof.
"""
import json
import os
import sys
from unittest import mock

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RAF_DIR = os.path.join(_REPO_ROOT, "data", "RAF")
if _RAF_DIR not in sys.path:
    sys.path.insert(0, _RAF_DIR)

import publish as raf_publish  # noqa: E402


# --------------------------------------------------------------------------- #
# kinds and identities
# --------------------------------------------------------------------------- #
def test_the_flavors_have_their_own_marker_kinds():
    assert set(raf_publish.MARKER_KINDS) == {"prepare", "depth",
                                             "mappingA_prepare", "mappingA_depth"}
    names = {raf_publish.marker_name(k) for k in raf_publish.MARKER_KINDS}
    assert len(names) == 4                      # no two kinds share a file name
    assert raf_publish.marker_name("mappingA_prepare") == \
        "raf_publish_commit.mappingA_prepare.json"


def test_every_kind_carries_a_registered_identity():
    for kind in raf_publish.MARKER_KINDS:
        assert kind in raf_publish.CANONICAL_IDENTITIES
        assert isinstance(raf_publish.CANONICAL_IDENTITIES[kind], dict)


def test_the_mappingA_identities_pin_the_planned_fields():
    prepare = raf_publish.CANONICAL_IDENTITIES["mappingA_prepare"]
    assert prepare["rooms"] == list(raf_publish.CANONICAL_ROOMS)
    assert prepare["n_placements"] == 16
    assert prepare["k"] == 8
    assert prepare["n_items"] == 1152
    assert prepare["match_algorithm_version"]
    assert prepare["match_p95_m"] == 0.01
    assert prepare["match_max_m"] == 0.02
    assert prepare["match_ambiguity_margin"] == 3.0
    assert prepare["placement_cap_m"] == 0.05
    # Amendment 4: Mapping A publishes its COMPLETE union at x2.0 (its clip clamp
    # binds at 2.0401); Mapping H stays at x3.0 and the difference is disclosed.
    assert prepare["amplitude_scalar"] == 2.0
    assert raf_publish.CANONICAL_PREPARE_PARAMS["amplitude_scalar"] == 3.0
    # r2 N5 + r5 pin: every digest in this identity is now an exact value. The
    # audio union comes from the clean dry run (generation 5fc096147bec).
    assert len(prepare["correspondence_sha256"]) == 64
    assert prepare["correspondence_sha256"] != raf_publish.SHA256_SHAPE
    assert prepare["readback_record_sha256"] == raf_publish.canonical_record_digest()
    assert prepare["audio_union_sha256"] == (
        "b19eff06c7a13e0aaeafcdf95ad58f7f4f24bb3def794889102440437a220a21")
    assert raf_publish.SHA256_SHAPE not in prepare.values()

    depth = raf_publish.CANONICAL_IDENTITIES["mappingA_depth"]
    assert depth["positions_from"] == "mappingA"
    assert depth["img_h"] == 256 and depth["img_w"] == 512
    assert depth["max_miss_rate"] == 0.0025
    assert depth["n_maps"] == 1152


def test_the_mappingH_identities_are_untouched():
    """The inherited publication must keep verifying exactly as before."""
    assert raf_publish.CANONICAL_IDENTITIES["prepare"] is \
        raf_publish.CANONICAL_PREPARE_PARAMS
    assert raf_publish.CANONICAL_IDENTITIES["depth"] is \
        raf_publish.CANONICAL_RENDER_PARAMS
    assert raf_publish.CANONICAL_PREPARE_PARAMS["amplitude_scalar"] == 3.0
    assert raf_publish.CANONICAL_RENDER_PARAMS["max_miss_rate"] == 0.0025


# --------------------------------------------------------------------------- #
# publishing both flavors on one tree
# --------------------------------------------------------------------------- #
def _publish(marker_root, kind, roots, payload="x"):
    with raf_publish.PublishTransaction(str(marker_root), kind=kind) as txn:
        for root in roots:
            staged = txn.stage(str(root))
            with open(staged.path("payload.txt"), "w") as f:
                f.write(payload)
        return txn.commit(extra={"canonical": False, "taint": [], "flavor": kind})


def _tree(tmp_path):
    """The ACTUAL default topology of the two CLIs (N2).

    Mapping H: split root data/RAF, runtime RAF/. Mapping A: split root
    data/RAF_mappingA (registered as disjoint), runtime RAF/mappingA/. The r1
    composition tests invented a separate A split root and so could not see that
    the shipped default pointed both flavors at data/RAF.
    """
    import prepare_data as raf_prepare_mod
    import prepare_mappingA as prep_a_mod

    assert os.path.basename(raf_prepare_mod.build_parser().get_default("split_dir")) \
        != os.path.basename(prep_a_mod.build_parser().get_default("split_dir"))
    return {
        "split_h": tmp_path / raf_prepare_mod.build_parser().get_default("split_dir"),
        "runtime": tmp_path / "runtime" / "RAF",
        "depth_h": [tmp_path / "runtime" / "RAF" / room / "depth_images"
                    for room in raf_publish.CANONICAL_ROOMS],
        "split_a": tmp_path / prep_a_mod.build_parser().get_default("split_dir"),
        "runtime_a": tmp_path / "runtime" / "RAF" / "mappingA",
        "depth_a": [tmp_path / "runtime" / "RAF" / "mappingA" / room / "depth_images"
                    for room in raf_publish.CANONICAL_ROOMS],
    }


def test_the_two_cli_default_split_roots_are_disjoint():
    """N2's root cause in one assertion: the shipped defaults, not a test topology."""
    import prepare_data as raf_prepare_mod
    import prepare_mappingA as prep_a_mod

    h_root = raf_prepare_mod.build_parser().get_default("split_dir")
    a_root = prep_a_mod.build_parser().get_default("split_dir")
    assert h_root == "data/RAF"
    assert a_root == "data/RAF_mappingA" == prep_a_mod.MAPPINGA_SPLIT_ROOT
    assert os.path.abspath(h_root) != os.path.abspath(a_root)
    assert not os.path.abspath(a_root).startswith(os.path.abspath(h_root) + os.sep)


def test_a_shared_split_root_would_break_both_flavors(tmp_path):
    """Why the disjointness matters: one manifest per directory, so publishing the
    second flavor into the same root invalidates the first flavor's attestation."""
    shared = tmp_path / "data_RAF"
    _publish(shared, "prepare", [tmp_path / "runtime", shared])
    before = raf_publish.verify_publication(str(shared), kind="prepare")["published"]
    assert before is True
    _publish(shared, "mappingA_prepare", [tmp_path / "runtime_a", shared])
    assert raf_publish.verify_publication(str(shared), kind="prepare")["published"] \
        is False


def _publish_h(t):
    _publish(t["split_h"], "prepare", [t["runtime"], t["split_h"]])
    _publish(t["runtime"], "depth", t["depth_h"])


def _publish_a(t):
    _publish(t["split_a"], "mappingA_prepare", [t["runtime_a"], t["split_a"]])
    _publish(t["runtime_a"], "mappingA_depth", t["depth_a"])


def _h_valid(t):
    return raf_publish.verify_combined_publication(
        str(t["split_h"]), str(t["runtime"]), canonical=False,
        rooms=list(raf_publish.CANONICAL_ROOMS))["published"]


def _a_valid(t):
    return raf_publish.verify_combined_publication(
        str(t["split_a"]), str(t["runtime_a"]), canonical=False,
        rooms=list(raf_publish.CANONICAL_ROOMS), flavor="mappingA")["published"]


def test_composition_h_then_a(tmp_path):
    """Publishing Mapping A must not disturb an existing Mapping-H publication."""
    t = _tree(tmp_path)
    _publish_h(t)
    assert _h_valid(t) is True
    _publish_a(t)
    assert _h_valid(t) is True
    assert _a_valid(t) is True


def test_composition_a_then_h(tmp_path):
    """... and the other order, since the H roots are ancestors of the A ones."""
    t = _tree(tmp_path)
    _publish_a(t)
    assert _a_valid(t) is True
    _publish_h(t)
    assert _a_valid(t) is True
    assert _h_valid(t) is True


def test_composition_republish_one_flavor(tmp_path):
    """Re-cutting Mapping A leaves Mapping H's generation untouched."""
    t = _tree(tmp_path)
    _publish_h(t)
    _publish_a(t)
    h_marker = json.loads(
        (t["split_h"] / raf_publish.marker_name("prepare")).read_text())

    _publish_a(t)                                    # republish A only
    assert _a_valid(t) is True and _h_valid(t) is True
    assert json.loads(
        (t["split_h"] / raf_publish.marker_name("prepare")).read_text()) == h_marker


def test_composition_injected_crash_during_a_leaves_h_valid(tmp_path):
    """A crash mid-A must leave A unpublished and H exactly as it was."""
    t = _tree(tmp_path)
    _publish_h(t)
    _publish_a(t)
    h_before = json.loads(
        (t["split_h"] / raf_publish.marker_name("prepare")).read_text())
    a_before = json.loads(
        (t["split_a"] / raf_publish.marker_name("mappingA_prepare")).read_text())

    real_replace = os.replace
    calls = {"n": 0}

    def failing_replace(src, dst):
        if str(dst).endswith("payload.txt"):
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("NAS went away mid Mapping-A publish")
        return real_replace(src, dst)

    with mock.patch("os.replace", failing_replace):
        with pytest.raises(OSError):
            _publish(t["split_a"], "mappingA_prepare",
                     [t["runtime_a"], t["split_a"]], payload="second")

    assert _a_valid(t) is False                      # A reads as unpublished
    assert _h_valid(t) is True                       # H untouched
    assert json.loads(
        (t["split_h"] / raf_publish.marker_name("prepare")).read_text()) == h_before
    assert a_before["generation"]


def test_the_two_flavors_use_different_marker_files(tmp_path):
    t = _tree(tmp_path)
    _publish_h(t)
    _publish_a(t)
    assert (t["split_h"] / raf_publish.marker_name("prepare")).exists()
    assert (t["split_a"] / raf_publish.marker_name("mappingA_prepare")).exists()
    assert not (t["split_h"] / raf_publish.marker_name("mappingA_prepare")).exists()


def test_a_flavor_cannot_verify_with_the_other_flavors_kind(tmp_path):
    t = _tree(tmp_path)
    _publish_a(t)
    wrong = raf_publish.verify_publication(str(t["split_a"]), kind="prepare")
    assert wrong["published"] is False
    assert "marker" in wrong["reason"]


def test_verify_combined_publication_rejects_an_unknown_flavor(tmp_path):
    with pytest.raises(ValueError):
        raf_publish.verify_combined_publication(str(tmp_path), str(tmp_path),
                                                flavor="mappingZ", canonical=False,
                                                rooms=["EmptyRoom"])
