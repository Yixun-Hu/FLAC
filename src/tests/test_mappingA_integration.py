"""Mapping A end to end on a synthetic corpus (exp_21, cycle 11).

Each earlier cycle proved one link. This walks the whole chain on one fixture --
placement clustering, microphone correspondence, item construction, audio union,
amplitude audit, writing, manifest validation, publication, listener rendering, and
finally loading through the REAL dataloader with the publication gate live -- so
the pieces are shown to compose, not merely to work apart.
"""
import json
import os
import sys

import numpy as np
import pytest
import soundfile as sf

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RAF_DIR = os.path.join(_REPO_ROOT, "data", "RAF")
if _RAF_DIR not in sys.path:
    sys.path.insert(0, _RAF_DIR)

import mappingA_common as mac  # noqa: E402
import prepare_mappingA as prep_a  # noqa: E402
import publish as raf_publish  # noqa: E402
import render_depth as raf_render  # noqa: E402

from test_mappingA_md import load_md  # noqa: E402
from test_raf_prepare_data import write_room  # noqa: E402
from test_raf_render_depth import _box_mesh_raf, _readback  # noqa: E402

ROOM = "EmptyRoom"
N_GROUPS = 12
K = 8


def _one_placement_groups(n=N_GROUPS):
    """n tx-poses at DISTINCT source positions over ONE array placement.

    That is the Mapping-A population: the array stays put and the loudspeaker
    moves, which is what lets one microphone hear many sources.
    """
    return [((round(0.1 * (g + 1), 6), 0.9, 0.0, 0.1),          # quaternion
             (round(1.0 + g, 6), 1.5, round(0.5 * g, 6)),        # distinct tx xyz
             # the same placement, re-occupied to sub-cm (exp_19 measured exactly
             # that); a wider spread would legitimately fail the 1 cm p95 gate
             # y=0.6 lifts the array off the floor: a listener-positioned render
             # needs the camera strictly inside the room, and the lattice's lowest
             # mic sits at the placement height itself
             (2.0, 0.6, round(-1.0 + 0.0005 * g, 6)))
            for g in range(n)]


@pytest.fixture
def corpus(tmp_path):
    write_room(str(tmp_path), ROOM, groups=_one_placement_groups(), rir_peak=0.2)
    return os.path.join(str(tmp_path), "archived", ROOM)


def _poses_from_corpus(corpus_dir):
    """Re-derive the exp_19 group structures the Mapping-A stack consumes."""
    import prepare_data as raf_prepare

    index = raf_prepare.load_room_index(corpus_dir)
    groups, _report = raf_prepare.group_captures(index)
    return groups


def test_the_whole_chain_composes(corpus, tmp_path):
    # 1. placements + correspondence -------------------------------------------
    groups = _poses_from_corpus(corpus)
    assert len(groups) == N_GROUPS
    clusters = mac.cluster_placements(groups)
    assert len(clusters) == 1, "the array never moved, so this is ONE placement"
    placement = clusters[0]

    assignment, match = {}, {}
    for group in groups:
        report = mac.match_mics(placement["template_rx"], group["rx_xyz_p"])
        assert report["passed"] is True, group["group_key"]
        assignment[group["group_key"]] = report["assignment"]
        match[group["group_key"]] = {"p95_m": report["p95_m"],
                                     "max_m": report["max_m"],
                                     "min_ambiguity_margin":
                                         report["min_ambiguity_margin"],
                                     # r3 P2: the full evidence travels with it
                                     "evidence_sha256": report["evidence_sha256"],
                                     "rigid_residual_rms_m":
                                         report["rigid_residual_rms_m"]}

    # 2. items ------------------------------------------------------------------
    poses = [dict(g, tx_xyz=g["tx_xyz"], tx_xyz_p=g["tx_xyz_p"]) for g in groups]
    items = prep_a.build_items(ROOM, placement["placement_id"], poses, assignment,
                               match, k=K)
    assert len(items) == 36
    report = prep_a.validate_manifest({"items": items, "k": K}, expected_items=36,
                                      k=K, assignments={ROOM: assignment})
    assert report["passed"] is True
    assert report["assignments_attested"] is True

    # 3. audio union + amplitude policy ----------------------------------------
    union = prep_a.enumerate_audio_union(items)
    counts = prep_a.union_report(union, items)
    assert counts["n_items"] == 36
    assert counts["n_captures"] <= 36 * (K + 1)          # deduplicated
    audit = prep_a.audit_amplitude_union(corpus, union[ROOM], scalar=3.0)
    assert audit["passed"] is True

    # 4. write ------------------------------------------------------------------
    runtime = tmp_path / "runtime" / "mappingA"
    written = prep_a.write_union(corpus, str(runtime / ROOM), union[ROOM], scalar=3.0)
    assert written["n_files"] == len(union[ROOM])
    assert written["roundtrip_max_abs_error"] == 0.0

    # 5. runtime metadata -------------------------------------------------------
    meta_dir = runtime / ROOM / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "mappingA_metadata.json", "w") as f:
        json.dump({item["target_capture_id"]: item for item in items}, f)

    # 6. listener renders -------------------------------------------------------
    raf_root = tmp_path / "raf"
    (raf_root / "3d_models" / ROOM).mkdir(parents=True)
    import open3d as o3d

    o3d.io.write_triangle_mesh(
        str(raf_root / "3d_models" / ROOM / "mesh.obj"),
        _box_mesh_raf(x0=-10.0, x1=10.0, y0=0.0, y1=3.0, z0=-10.0, z1=10.0,
                      to_pipeline=False))
    raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(runtime),
                     "--rooms", ROOM, "--positions-from", "mappingA",
                     "--img-h", "64", "--img-w", "128", "--non-canonical",
                     "--readback-record", _readback(tmp_path)])
    with open(runtime / ROOM / "depth_images" / "raf_depth_qa.json") as f:
        qa = json.load(f)
    assert qa["positions_from"] == "mappingA"
    assert all(e["positioned_at"] == "listener" for e in qa["maps"].values())
    assert all(e["real_mesh"]["vertical_axis"]["ok"] for e in qa["maps"].values())

    # 7. the manifest the dataloader reads --------------------------------------
    split_dir = tmp_path / "splits_mappingA"
    split_dir.mkdir()
    manifest = {ROOM: sorted(f"{item['target_capture_id']}.wav" for item in items)}
    with open(split_dir / "mappingA_eval.json", "w") as f:
        json.dump(manifest, f)
    assert len(manifest[ROOM]) == 36

    # 8. load through the REAL dataloader ---------------------------------------
    from src.data.dataset import LocalDatasetConfig, SampleDataset, collation_fn

    config = LocalDatasetConfig(
        id="RAF", path=str(runtime), custom_metadata_fn=load_md().get_custom_metadata,
        json_file_path=str(split_dir / "mappingA_eval.json"),
        folder_name="mono_rirs_22050Hz",
        conditioning={"acoustic_context": {"load": True, "max_context": K,
                                           "max_len": 9600, "deterministic": True},
                      "depth": {"load": False},          # 64x128 maps by design here
                      "poses": {"load": True}})
    dataset = SampleDataset([config], sample_size=10240, sample_rate=22050,
                            random_crop=False, force_channels="mono", augs=False)
    assert len(dataset) == 36
    audio, metadata = collation_fn([dataset[i] for i in range(4)])
    assert audio.shape == (4, 1, 10240)
    for md in metadata:
        assert md["source"].shape == (3,)
        assert md["context_poses"].shape == (K, 3)
        assert md["context_audio"].shape == (K, 1, 9600)
        assert md["context_capture_ids"].shape == (K,)
        assert int(md["sample_target_id"]) not in md["context_capture_ids"].tolist()

    # 9. and the items really are unseen-source ---------------------------------
    by_capture = {item["target_capture_id"]: item for item in items}
    for md in metadata:
        item = by_capture[f"{int(md['sample_target_id']):06d}"]
        assert all(c["xyz_key"] != item["target_xyz_key"] for c in item["context"])


def test_the_chain_stops_at_the_amplitude_gate(corpus, tmp_path):
    """One loud capture anywhere in the union stops the run BEFORE any write."""
    from test_raf_prepare_data import _rir

    groups = _poses_from_corpus(corpus)
    clusters = mac.cluster_placements(groups)
    assignment = {g["group_key"]: mac.match_mics(clusters[0]["template_rx"],
                                                 g["rx_xyz_p"])["assignment"]
                  for g in groups}
    match = {g["group_key"]: {"p95_m": 0.001, "max_m": 0.002,
                              "min_ambiguity_margin": 50.0} for g in groups}
    items = prep_a.build_items(ROOM, "p000", groups, assignment, match, k=K)
    union = prep_a.enumerate_audio_union(items)

    loud = union[ROOM][5]
    sf.write(os.path.join(corpus, "data", loud, "rir.wav"), _rir(7, peak=0.9), 48000,
             subtype="FLOAT")
    with pytest.raises(prep_a.AmplitudePolicyError) as exc:
        prep_a.audit_amplitude_union(corpus, union[ROOM], scalar=3.0)
    assert exc.value.report["decision_required"] is True
    assert not (tmp_path / "runtime").exists()     # nothing was written


def test_a_failed_correspondence_removes_a_group_before_eligibility(corpus, tmp_path):
    """A tx-group whose array moved is excluded BEFORE items exist, so the item
    count never shrinks silently afterwards."""
    groups = _poses_from_corpus(corpus)
    clusters = mac.cluster_placements(groups)
    template = clusters[0]["template_rx"]

    moved = dict(groups[3])
    moved["rx_xyz_p"] = groups[3]["rx_xyz_p"] + np.array([0.03, 0.0, 0.0])
    eligible = []
    for group in groups[:3] + [moved] + groups[4:]:
        if mac.match_mics(template, group["rx_xyz_p"])["passed"]:
            eligible.append(group)
    assert len(eligible) == N_GROUPS - 1
    assert all(g["group_key"] != moved["group_key"] for g in eligible)
    # eligibility still holds (>= 9 source-distinct groups), so items can be built
    assert len(eligible) >= 9
