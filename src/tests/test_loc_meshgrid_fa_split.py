"""exp_22 R2-B -- the frame-average-aware mesh-grid conditioning (plan r2 §2 table, §3 R2-B).

The BF arm (OrbitRIR, C4 frame-averaged) conditions every candidate through the
released ``invariant_conditioning``. A mesh pass cannot afford that per
candidate, so R2-B splits it five ways (Codex plan review B5): ``context_audio``
once per query, ``context_poses_vit`` C4-averaged once per query,
``context_poses`` recomputed per scored candidate batch (cylindrical against the
CANDIDATE's reference, never averaged), ``source_vit`` C4-averaged and
``source`` cylindrical base cached per (receiver, candidate).

A wrong cache silently changes every BF number, so these tests pin, cell by
cell, which inputs each tensor reads and which batch it is computed in -- with
fake conditioners whose outputs are functions of exactly the inputs they are
supposed to read -- and they pin the VANILLA path's numbers to digests frozen at
the untouched R2-A HEAD, so the change cannot move the P1/YAW arms.
"""
import json
import math
import os

import numpy as np
import pytest
import torch

import test_loc_meshgrid_engine as ENG                    # noqa: E402  (src/tests on path)
from src.localization import meshgrid_engine as me
from src.localization.crossarm import canonical_sha256

WORKLOG = os.path.join("worklog", "worklog_yixun", "exp_22_loc_meshgrid_claude")


# --------------------------------------------------------------------------- #
# the vanilla no-regression fixture: digests frozen BEFORE any R2-B change
# --------------------------------------------------------------------------- #
#: computed at the untouched R2-A HEAD 75628305 (tracked-clean), twice, by the
#: scratch script logged in FROZEN_LOG -- the same computation as
#: :func:`_vanilla_digests` below. A change to the vanilla path that moves ANY
#: number, ANY conditioner call (batch size or requested ids) or ANY sampler
#: batch shape moves one of these.
FROZEN_LOG = os.path.join(WORKLOG, "loc_meshgrid_2026-11-02_22:42:36_r2b_frozen_vanilla_at_head.log")
FROZEN_VANILLA = {
    "conditioner_calls": "65cdc32430b3d3b03a46b49c446783083bb15b3ec9af32d8752d465380975560",
    "n_conditioner_rows": 254,
    "n_sampler_rows": 1160,
    "parity": "5790e2bd63da98dbfdc8c0d7dd1d547946a398fa8955c28c118e813a8e76d6eb",
    "replay": "f9180f34ab02288ada519ca652049f13a852ec90ac1e5f86900ac40d131c86be",
    "sampler_batches": "e8620612433bbf26f9965ea8511de4419bacf32617a073b66b4e68c31830b738",
    "score_fingerprints": "6fe4a71cc5798db1dea01ea876721e5bde0f4f8ebb053c6ef0c1f069aa414c10",
    "sims_sha256": "fd7dd260acb8ef63caf1564dccbeb3214d1fc05e909edb6add4d04d601626e56",
}


def _vanilla_digests(tmp_path):
    """The vanilla pass, replay and cache-parity outputs on the engine fixture, digested."""
    plan, records, items = ENG._aligned(tmp_path)
    engine = ENG.SyntheticEngine()
    batches, sample = [], engine.sampler
    # the fixture's sampler is batch-invariant, so the batch SHAPES it is called
    # with are recorded explicitly -- otherwise a re-chunked pass would be invisible
    engine.sampler = lambda noise, cond: (batches.append(int(noise.shape[0])),
                                          sample(noise, cond))[1]
    out = str(tmp_path / "run")
    summary = me.run_pass(engine, items, records, plan, out, num_samples=4,
                          prefixes=(1, 4), batch_rows=8, source_chunk=3)
    rows = me.read_rows(out)
    room = me.load_room_plan(plan, "A/A_idx_1")
    query = room.queries[0]
    replay = me.replay_check(ENG.SyntheticEngine(), query, me.GuardedMetadata(items[0][1]),
                             items[0][0], num_samples=4, prefixes=(1, 4), batch_rows=8,
                             source_chunk=3)
    parity = me.cache_parity_check(ENG.SyntheticEngine(), query, items[0][1], n_candidates=3,
                                   source_chunk=2)
    return {
        "score_fingerprints": canonical_sha256([me.score_fingerprint(r) for r in rows]),
        "sims_sha256": canonical_sha256([r["sims_sha256"] for r in rows]),
        "conditioner_calls": canonical_sha256(engine.conditioner.calls),
        "n_sampler_rows": engine.n_sampler_rows,
        "sampler_batches": canonical_sha256(batches),
        "n_conditioner_rows": summary["n_conditioner_rows"],
        "replay": canonical_sha256({k: replay[k] for k in ("max_abs_delta", "bit_exact",
                                                           "fingerprint_equal",
                                                           "n_candidates")}),
        "parity": canonical_sha256({k: parity[k] for k in ("memoization", "batched",
                                                           "counter_test", "dtypes",
                                                           "match")}),
    }


def test_the_vanilla_path_reproduces_its_frozen_digests(tmp_path):
    """Plan r2 §3 R2-B: "the vanilla path is numerically untouched" -- on the fixture,
    every score, every sidecar, every conditioner call, the replay and the cache
    parity equal what the R2-A HEAD produced."""
    assert _vanilla_digests(tmp_path) == FROZEN_VANILLA


def test_the_frozen_digests_were_logged_at_the_untouched_head():
    """The pinned values are the ones the record says were computed before the change."""
    text = open(FROZEN_LOG).read()
    assert "HEAD: 75628305ac40eb0375c7d09702b8b8c66ac9d8a4; tracked-clean: yes" in text
    runs = [json.loads(block.split("\n", 1)[1]) for block in text.split("## run ")[1:]]
    assert len(runs) == 2 and runs[0] == runs[1] == FROZEN_VANILLA


# --------------------------------------------------------------------------- #
# the dependency table, cell by cell, on a fake whose outputs read EXACTLY the
# inputs the real conditioners read
# --------------------------------------------------------------------------- #
C4 = (0.0, 90.0, 180.0, 270.0)
FA_IDS = ("context_poses_vit", "context_poses", "context_audio", "source", "source_vit")


class FaFakeConditioner:
    """``MultiConditioner``'s seam, with each id a function of exactly its real inputs.

    ``context_audio`` reads the context audio; ``source`` and ``context_poses``
    read their pose vectors (absolute or cylindrical -- whatever they are
    handed); the two ViT ids read their pose MINUS the depth panorama under a
    column-asymmetric weight, so a rotated frame is a genuinely different input
    and a frame average is genuinely not the angle-0 value. Every output row is
    computed from its own sample only, so batch composition cannot leak into a
    value -- which is what makes bit-identity at identical batching meaningful.
    """

    def __init__(self):
        self.calls = []

    @staticmethod
    def _pose(vectors):
        rows = torch.as_tensor(vectors).float().reshape(-1, 3)
        s = rows @ torch.tensor([1.0, 2.0, 3.0])
        index = torch.arange(rows.shape[0], dtype=torch.float32)
        return torch.stack([torch.sin(s + index).sum(), torch.cos(s).sum(),
                            torch.sin(2.0 * s).sum()])

    @staticmethod
    def _vit(vectors, depth):
        depth = torch.as_tensor(depth).float()
        weight = torch.arange(depth.numel(), dtype=torch.float32).reshape(depth.shape)
        weight = weight / depth.numel()
        total = []
        for pose in torch.as_tensor(vectors).float().reshape(-1, 3):
            s = ((pose[:, None, None] - depth) * weight).sum()
            total.append(torch.stack([torch.sin(s), torch.cos(0.5 * s), torch.sin(0.25 * s)]))
        return torch.stack(total).sum(dim=0)

    def _row(self, key, md):
        if key == "context_audio":
            s = torch.as_tensor(md["context_audio"]).float().sum()
            return torch.stack([torch.sin(s), torch.cos(s), torch.sin(2.0 * s)])
        if key in ("source", "context_poses"):
            return self._pose(md[key])
        return self._vit(md[key], md["depth"])

    def __call__(self, batch_metadata, device, only_ids=None):
        self.calls.append({"n": len(batch_metadata),
                           "ids": None if only_ids is None else sorted(only_ids)})
        out = {}
        for key in FA_IDS:
            if only_ids is not None and key not in only_ids:
                continue
            rows = [self._row(key, md) for md in batch_metadata]
            out[key] = [torch.stack(rows).to(device), torch.ones(len(batch_metadata), 1)]
        return out


def _fa_md(n_contexts=3):
    """One query's loader item: a [3, H, W] panorama whose W makes 90 deg an exact roll."""
    generator = torch.Generator().manual_seed(7)
    poses = torch.tensor([[1.0, 0.5, 0.2], [-0.7, 1.3, 0.1], [0.4, -1.1, -0.3],
                          [-0.2, -0.6, 0.4]], dtype=torch.float32)[:n_contexts]
    return {"depth": torch.rand((3, 2, 8), generator=generator, dtype=torch.float64) * 4.0 - 2.0,
            "context_audio": torch.linspace(0.0, 1.0, n_contexts * 16).reshape(n_contexts, 1, 16),
            "context_poses": poses, "context_poses_vit": poses,
            "source": torch.tensor([0.9, -0.4, 0.3]),
            "source_vit": torch.tensor([[0.9, -0.4, 0.3]])}


def _positions(n=40, seed=3):
    """Camera-frame candidate positions with distinct azimuths."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-2.0, 2.0, size=(n, 3)).astype(np.float64)


def _released(conditioner, md, positions):
    """The released path over a candidate batch: ``invariant_conditioning`` at cap = batch."""
    from src.data.yaw_rotation import invariant_conditioning
    from src.localization.candidates import candidate_metadata

    free = {key: value for key, value in md.items() if key not in ("source", "source_vit")}
    batch = [candidate_metadata(free, position) for position in positions]
    return invariant_conditioning(conditioner, batch, "cpu", C4, max_fwd_samples=len(batch))


def _split():
    return me.FaSplit(angles=C4, recorder=me.FaPartitionRecorder())


def test_the_split_is_the_dependency_table_and_partitions_the_five_ids():
    assert me.FA_SPLIT == "fa_five_way"
    assert me.FA_BRANCH_IDS == {"context_vit": ("context_poses_vit", "context_audio"),
                                "context_scored_batch": ("context_poses",),
                                "source_cache": ("source", "source_vit")}
    ids = [key for branch in me.FA_BRANCH_IDS.values() for key in branch]
    assert sorted(ids) == sorted(FA_IDS) and len(ids) == len(set(ids))
    # the same five ids the released config declares -- nothing dropped, nothing added
    assert set(ids) == set(me.CONTEXT_COND_IDS) | set(me.SOURCE_COND_IDS)
    assert me.FA_ORBIT_BRANCHES == ("context_vit", "source_cache")


def test_context_audio_is_computed_once_per_query_and_never_orbit_averaged():
    fake = FaFakeConditioner()
    md = me.GuardedMetadata(_fa_md())                 # the target stays unreadable
    out = me.fa_query_conditioning(_split(), fake, md, "cpu")
    assert sorted(out) == ["context_audio", "context_poses_vit"]
    # ONE base pass over one row carries context_audio; the three rotated passes
    # re-run the ViT id ONLY
    assert fake.calls == [{"n": 1, "ids": ["context_audio", "context_poses_vit"]},
                          {"n": 1, "ids": ["context_poses_vit"]},
                          {"n": 1, "ids": ["context_poses_vit"]},
                          {"n": 1, "ids": ["context_poses_vit"]}]
    assert sum("context_audio" in (call["ids"] or FA_IDS) for call in fake.calls) == 1
    direct = FaFakeConditioner()(([_fa_md()]), "cpu", only_ids=["context_audio"])
    assert torch.equal(out["context_audio"][0], direct["context_audio"][0])
    assert out["context_audio"][0].shape[0] == 1 and out["context_poses_vit"][0].shape[0] == 1


def test_context_poses_vit_is_the_c4_average_of_the_rotated_scene():
    from src.data.yaw_rotation import rotate_scene_metadata

    fake, md = FaFakeConditioner(), _fa_md()
    out = me.fa_query_conditioning(_split(), fake, me.GuardedMetadata(md), "cpu")
    frames = [rotate_scene_metadata(md, math.radians(angle), 8,
                                    pose_keys=("context_poses_vit",)) for angle in C4]
    by_hand = sum(FaFakeConditioner()._vit(f["context_poses_vit"], f["depth"]) for f in frames) / 4
    assert torch.allclose(out["context_poses_vit"][0][0], by_hand, atol=1e-6)
    angle0 = FaFakeConditioner()._vit(md["context_poses_vit"], md["depth"])
    assert not torch.allclose(out["context_poses_vit"][0][0], angle0, atol=1e-3)


def test_context_poses_varies_with_the_candidate_and_not_with_the_angle():
    fake, md = FaFakeConditioner(), _fa_md()
    positions = _positions(4)
    out = me.fa_candidate_conditioning(_split(), fake, md["context_poses"], positions, "cpu")
    assert sorted(out) == ["context_poses"]
    rows = out["context_poses"][0]
    assert rows.shape[0] == 4
    # candidate-dependent: dphi is measured against EACH candidate's azimuth
    assert all(not torch.allclose(rows[i], rows[j]) for i in range(4) for j in range(i + 1, 4))
    # angle-INDEPENDENT: no rotated pass is ever made for it ...
    assert fake.calls == [{"n": 4, "ids": ["context_poses"]}]
    # ... because rotating the whole scene (candidates and contexts) does not move it,
    # at a C4 angle and at an arbitrary one
    for angle in (90.0, 33.0):
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        rotated = me.fa_candidate_conditioning(
            _split(), FaFakeConditioner(),
            torch.as_tensor(md["context_poses"].double().numpy() @ rot.T, dtype=torch.float32),
            positions @ rot.T, "cpu")
        assert torch.allclose(rotated["context_poses"][0], rows, atol=1e-4)
    # ... and it is NOT the absolute pose (the cylindrical conversion happened)
    absolute = FaFakeConditioner()._pose(md["context_poses"])
    assert not any(torch.allclose(row, absolute, atol=1e-3) for row in rows)


def test_source_vit_is_averaged_over_exactly_four_angles_and_source_is_the_cylindrical_base():
    from src.data.yaw_rotation import cylindrical_pose_features, rotate_scene_metadata
    from src.localization.candidates import candidate_metadata

    fake, md = FaFakeConditioner(), _fa_md()
    positions = _positions(5)
    out = me.source_conditioning(fake, {"depth": md["depth"]}, positions, "cpu", chunk=5,
                                 fa=_split())
    assert sorted(out) == ["source", "source_vit"]
    # one base pass (both ids), then EXACTLY three rotated passes of source_vit alone
    assert fake.calls == [{"n": 5, "ids": ["source", "source_vit"]}] + \
        [{"n": 5, "ids": ["source_vit"]}] * 3
    for row, position in enumerate(positions):
        candidate = candidate_metadata({"depth": md["depth"]}, position)
        frames = [rotate_scene_metadata(cylindrical_pose_features(candidate),
                                        math.radians(angle), 8, pose_keys=("source_vit",))
                  for angle in C4]
        by_hand = sum(FaFakeConditioner()._vit(f["source_vit"], f["depth"]) for f in frames) / 4
        assert torch.allclose(out["source_vit"][0][row], by_hand, atol=1e-6)
        # source: the cylindrical (r, z, 0), read once -- not averaged, not absolute
        cylindrical = cylindrical_pose_features(candidate)["source"]
        assert torch.equal(out["source"][0][row], FaFakeConditioner()._pose(cylindrical))
        r = math.hypot(position[0], position[1])
        assert torch.allclose(cylindrical, torch.tensor([r, position[2], 0.0]).float(), atol=1e-6)


def test_the_masks_are_the_base_pass_masks():
    class Marked(FaFakeConditioner):
        def __call__(self, batch_metadata, device, only_ids=None):
            out = super().__call__(batch_metadata, device, only_ids=only_ids)
            mark = 2.0 if only_ids is not None and len(only_ids) == 1 else 1.0
            return {key: [value[0], value[1] * mark] for key, value in out.items()}

    md = _fa_md()
    out = me.fa_query_conditioning(_split(), Marked(), me.GuardedMetadata(md), "cpu")
    assert torch.equal(out["context_poses_vit"][1], torch.ones(1, 1))


def test_the_split_equals_the_released_path_bit_for_bit_at_identical_batching():
    """The decomposition itself (gate 1's core): each branch at the released call's batch."""
    from src.localization.candidates import candidate_metadata

    md = _fa_md()
    for n in (1, 3, 5):
        positions = _positions(n, seed=n)
        released = _released(FaFakeConditioner(), md, positions)
        fake, split = FaFakeConditioner(), _split()
        query_row = {key: md[key] for key in me.FA_QUERY_KEYS}
        parts = {}
        parts.update(me.fa_branch(fake, "context_vit", [query_row] * n, "cpu", C4))
        parts.update(me.fa_branch(fake, "context_scored_batch",
                                  [candidate_metadata({"context_poses": md["context_poses"]}, p)
                                   for p in positions], "cpu", C4))
        parts.update(me.fa_branch(fake, "source_cache",
                                  [candidate_metadata({"depth": md["depth"]}, p)
                                   for p in positions], "cpu", C4))
        assert sorted(parts) == sorted(released)
        for key in FA_IDS:
            assert torch.equal(parts[key][0], released[key][0]), (n, key)
            assert torch.equal(parts[key][1], released[key][1]), (n, key)
        # the comparison bites: context_poses taken from the QUERY row (no candidate
        # reference) is the r1 plan's withdrawn error, and it is visible
        wrong = me.fa_branch(fake, "context_scored_batch",
                             [{"context_poses": md["context_poses"]}] * n, "cpu", C4)
        assert not torch.equal(wrong["context_poses"][0], released["context_poses"][0])


def test_a_branch_conditioner_answers_only_its_own_ids():
    md = _fa_md()
    with pytest.raises(ValueError, match="asked for"):
        me._BranchConditioner(FaFakeConditioner(), ("context_poses",))(
            [md], "cpu", only_ids=["source_vit"])

    class Extra(FaFakeConditioner):
        def __call__(self, batch_metadata, device, only_ids=None):
            out = super().__call__(batch_metadata, device, only_ids=None)
            return out

    with pytest.raises(ValueError, match="unrequested"):
        me.fa_branch(Extra(), "context_vit", [{key: md[key] for key in me.FA_QUERY_KEYS}
                                              | {"source": md["source"],
                                                 "context_poses": md["context_poses"],
                                                 "source_vit": md["source_vit"]}], "cpu", C4)


# --------------------------------------------------------------------------- #
# the executed partition: runtime evidence, derived from the calls that ran
# --------------------------------------------------------------------------- #
DECLARED = me.declared_fa_partition(list(C4), batch_rows=256, num_samples=8, source_chunk=16)


def _run_production_shapes(split, fake=None, source_chunk=16, scored=32, n_source=20,
                           n_scored=40):
    fake = fake or FaFakeConditioner()
    md = _fa_md()
    me.fa_query_conditioning(split, fake, me.GuardedMetadata(md), "cpu")
    me.source_conditioning(fake, {"depth": md["depth"]}, _positions(n_source), "cpu",
                           chunk=source_chunk, fa=split)
    positions = _positions(n_scored, seed=5)
    for start in range(0, n_scored, scored):
        me.fa_candidate_conditioning(split, fake, md["context_poses"],
                                     positions[start:start + scored], "cpu", configured=scored)
    return fake


def test_the_recorder_derives_the_declared_partition_from_what_ran():
    split = _split()
    _run_production_shapes(split)
    report = split.recorder.report()
    assert report["executed"] == DECLARED
    assert report["violations"] == []
    branches = report["branches"]
    # full AND tail micro-batches, each one forward per angle
    assert branches["source_cache"]["base_batch_histogram"] == {"16": 1, "4": 1}
    assert branches["source_cache"]["orbit_forward_histogram"] == {"16": 3, "4": 3}
    assert branches["context_scored_batch"]["base_batch_histogram"] == {"32": 1, "8": 1}
    assert branches["context_scored_batch"]["orbit_forward_histogram"] == {}
    assert branches["context_vit"]["base_batch_histogram"] == {"1": 1}
    assert branches["context_vit"]["orbit_forward_histogram"] == {"1": 3}
    assert all(entry["full_micro_batch_observed"] for entry in branches.values())
    assert me.assert_fa_partition_executed(DECLARED, split.recorder)["executed"] == DECLARED


@pytest.mark.parametrize("shapes, paths", [
    ({"source_chunk": 8}, ["source_cache.candidate_micro_batch", "source_cache.frame_avg_fwd_cap"]),
    ({"scored": 16}, ["context_scored_batch.candidate_micro_batch"]),
])
def test_a_partition_that_is_not_the_declared_one_is_refused_by_path(shapes, paths):
    split = _split()
    _run_production_shapes(split, **shapes)
    assert me.fa_partition_differences(DECLARED, split.recorder.executed()) == paths
    with pytest.raises(ValueError, match=paths[0].replace(".", r"\.")):
        me.assert_fa_partition_executed(DECLARED, split.recorder)


def test_mixed_micro_batches_two_angles_per_forward_and_cap_violations_are_refused():
    split = _split()
    _run_production_shapes(split)
    _run_production_shapes(split, source_chunk=12)            # a second, different chunking
    assert "source_cache.candidate_micro_batch" in me.fa_partition_differences(
        DECLARED, split.recorder.executed())

    recorder = me.FaPartitionRecorder()
    # an orbit that put two angles in one forward, then one: not the per-angle plan
    recorder.observe("source_cache", [(4, False), (8, True), (4, True)], cap=8, configured=16,
                     orbit_size=4)
    executed = recorder.executed()
    assert executed["angles_per_chunk"] == [1, 2] and executed["n_orbit_forwards"] == 2
    assert executed["cap_policy"] == "violated"
    assert any("cap 8" in reason for reason in recorder.violations())

    recorder = me.FaPartitionRecorder()
    recorder.observe("context_scored_batch", [(40, False)], cap=40, configured=32, orbit_size=4)
    assert any("exceeds" in reason for reason in recorder.violations())
    with pytest.raises(ValueError, match="violations"):
        me.assert_fa_partition_executed(DECLARED, recorder)


def test_an_empty_or_undeclared_partition_is_refused():
    with pytest.raises(ValueError):
        me.assert_fa_partition_executed(DECLARED, me.FaPartitionRecorder())
    split = _split()
    _run_production_shapes(split)
    with pytest.raises(ValueError, match="declare"):
        me.assert_fa_partition_executed(None, split.recorder)
