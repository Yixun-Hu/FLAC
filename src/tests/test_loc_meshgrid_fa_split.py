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


# --------------------------------------------------------------------------- #
# the split wired through the pass: caches, scored batches, refusals
# --------------------------------------------------------------------------- #
class FaSyntheticEngine(ENG.SyntheticEngine):
    """The engine fixture's synthetic stack, conditioned through the engaged split."""

    def __init__(self, angles=C4):
        super().__init__()
        self.conditioner = FaFakeConditioner()
        self.cond_method = "fa_invariant"
        self.split = me.split_for("fa_invariant", angles)


def _fa_aligned(tmp_path):
    """The engine fixture with a real ``[3, H, W]`` panorama per receiver (the fa path rolls it)."""
    plan, records, items = ENG._aligned(tmp_path)
    for _obs, md in items:
        seed = int(round(1000 * float(md["depth"].reshape(-1)[0])))     # one per receiver
        generator = torch.Generator().manual_seed(seed)
        md["depth"] = torch.rand((3, 2, 8), generator=generator, dtype=torch.float64) - 0.5
    return plan, records, items


#: the fixture's batching: 8 generated rows of K = 4 -> 2 candidates per scored batch
FA_SHAPES = {"num_samples": 4, "prefixes": (1, 4), "batch_rows": 8, "source_chunk": 3}
FA_DECLARED = me.declared_fa_partition(list(C4), batch_rows=8, num_samples=4, source_chunk=3)


def _fa_pass(tmp_path, engine=None, **overrides):
    plan, records, items = _fa_aligned(tmp_path)
    engine = engine or FaSyntheticEngine()
    out = str(tmp_path / "run")
    kwargs = dict(FA_SHAPES, fa_partition=FA_DECLARED)
    kwargs.update(overrides)
    summary = me.run_pass(engine, items, records, plan, out, **kwargs)
    return engine, summary, out, (plan, records, items)


def test_the_cache_refuses_every_method_and_split_but_the_registered_two():
    split = me.split_for("fa_invariant", C4)
    assert me.assert_cacheable("vanilla") is True
    assert me.assert_cacheable("vanilla", None) is True
    assert me.assert_cacheable("fa_invariant", split) is True
    for method, engaged in (("fa_invariant", None), ("vanilla", split),
                            ("fa_invariant", me.FaSplit(angles=C4, name="other")),
                            ("bogus", None), ("bogus", split)):
        with pytest.raises(ValueError, match="fa_invariant"):
            me.assert_cacheable(method, engaged)
    assert me.split_for("vanilla", [0.0]) is None
    for angles in (None, [], [90.0, 0.0]):
        with pytest.raises(ValueError, match="fa_invariant"):
            me.split_for("fa_invariant", angles)
    assert isinstance(split.recorder, me.FaPartitionRecorder) and split.angles == C4


def test_a_synthetic_fa_pass_scores_every_query_under_its_declared_partition(tmp_path):
    engine, summary, out, _ = _fa_pass(tmp_path)
    assert summary["n_scored"] == 4
    done, rejected = me.completed_queries(out)
    assert len(done) == 4 and rejected == []
    executed = summary["fa_partition_executed"]
    assert executed["executed"] == FA_DECLARED and executed["violations"] == []
    assert executed["note"] == me.FA_EXECUTION_NOTE
    # the source branch is still one conditioner row per (receiver, candidate)
    assert summary["n_conditioner_rows"] == ENG._fixture_totals()[1]
    assert engine.n_sampler_rows == ENG._fixture_totals()[0] * 4


def test_the_pass_computes_context_audio_exactly_once_per_query(tmp_path):
    engine, summary, _out, _ = _fa_pass(tmp_path)
    audio = [call for call in engine.conditioner.calls if "context_audio" in (call["ids"] or [])]
    assert len(audio) == summary["n_contexts_conditioned"] == 4
    assert all(call["n"] == 1 for call in audio)


def test_the_pass_recomputes_context_poses_once_per_scored_batch_and_never_per_angle(tmp_path):
    engine, _summary, _out, (plan, _records, _items) = _fa_pass(tmp_path)
    expected = []
    for room_id in ("A/A_idx_1", "B/B_idx_2"):
        for group in me.receiver_groups(me.load_room_plan(plan, room_id)):
            for query in group.queries:
                m = query.n_candidates
                expected += [min(2, m - start) for start in range(0, m, 2)]
    poses = [call["n"] for call in engine.conditioner.calls if call["ids"] == ["context_poses"]]
    assert poses == expected
    # the ViT ids are the only ones ever re-run on a rotated frame
    rotated = {tuple(call["ids"]) for call in engine.conditioner.calls
               if call["ids"] and len(call["ids"]) == 1 and call["ids"] != ["context_poses"]}
    assert rotated == {("context_poses_vit",), ("source_vit",)}


def test_source_vit_is_averaged_over_four_angles_per_receiver_chunk_in_the_pass(tmp_path):
    engine, _summary, _out, (plan, _records, _items) = _fa_pass(tmp_path)
    chunks = []
    for room_id in ("A/A_idx_1", "B/B_idx_2"):
        for group in me.receiver_groups(me.load_room_plan(plan, room_id)):
            u = len(group.union)
            chunks += [min(3, u - start) for start in range(0, u, 3)]
    base = [call["n"] for call in engine.conditioner.calls
            if call["ids"] == ["source", "source_vit"]]
    rotated = [call["n"] for call in engine.conditioner.calls if call["ids"] == ["source_vit"]]
    assert base == chunks and rotated == [n for n in chunks for _ in range(3)]


def test_every_scored_row_is_conditioned_exactly_as_the_released_path_conditions_it(tmp_path):
    """End to end: every generated row the DiT sees equals ``invariant_conditioning``
    over that scored batch's candidates (the fake is per-row, so batch shapes
    cannot excuse a difference)."""
    engine = FaSyntheticEngine()
    captured, inner = [], engine.cond_inputs_fn
    engine.cond_inputs_fn = lambda merged: (captured.append(
        {key: [value[0].clone(), value[1].clone()] for key, value in merged.items()}),
        inner(merged))[1]
    _engine, _summary, _out, (plan, _records, items) = _fa_pass(tmp_path, engine=engine)
    expected = []
    for room_id in ("A/A_idx_1", "B/B_idx_2"):
        for group in me.receiver_groups(me.load_room_plan(plan, room_id)):
            for query in group.queries:
                md = items[query.position][1]
                positions = (np.asarray(query.coordinates, dtype=np.float64)
                             - np.asarray(query.receiver_xyz, dtype=np.float64))
                for start in range(0, query.n_candidates, 2):
                    expected.append(_released(FaFakeConditioner(), md, positions[start:start + 2]))
    assert len(captured) == len(expected)
    for got, want in zip(captured, expected):
        assert sorted(got) == sorted(FA_IDS)
        for key in FA_IDS:
            assert torch.equal(got[key][0], want[key][0].repeat_interleave(4, dim=0)), key
            assert torch.equal(got[key][1], want[key][1].repeat_interleave(4, dim=0)), key


def test_a_declared_partition_the_pass_does_not_execute_is_refused_before_any_row(tmp_path):
    declared = me.declared_fa_partition(list(C4), batch_rows=8, num_samples=4, source_chunk=16)
    with pytest.raises(ValueError, match=r"source_cache\.candidate_micro_batch"):
        _fa_pass(tmp_path, fa_partition=declared)
    assert not os.path.isdir(os.path.join(str(tmp_path / "run"), me.ROWS_DIRNAME))


def test_an_fa_pass_without_a_declared_partition_is_refused_before_it_conditions(tmp_path):
    engine = FaSyntheticEngine()
    with pytest.raises(ValueError, match="declare"):
        _fa_pass(tmp_path, engine=engine, fa_partition=None)
    assert engine.conditioner.calls == []


def test_a_vanilla_pass_carries_no_executed_partition(tmp_path):
    plan, records, items = ENG._aligned(tmp_path)
    summary = me.run_pass(ENG.SyntheticEngine(), items, records, plan, str(tmp_path / "run"),
                          num_samples=4, prefixes=(1, 4), batch_rows=8)
    assert "fa_partition_executed" not in summary


def test_the_fa_replay_check_is_bit_exact_and_bites(tmp_path):
    plan, _records, items = _fa_aligned(tmp_path)
    query = me.load_room_plan(plan, "A/A_idx_1").queries[0]
    md = me.GuardedMetadata(items[0][1])
    report = me.replay_check(FaSyntheticEngine(), query, md, items[0][0], num_samples=4,
                             prefixes=(1, 4), batch_rows=8, source_chunk=3)
    assert report["bit_exact"] is True and report["max_abs_delta"] == 0.0

    class Drifting(FaFakeConditioner):
        def __call__(self, batch_metadata, device, only_ids=None):
            out = super().__call__(batch_metadata, device, only_ids=only_ids)
            return {key: [value[0] + 1e-2 * len(self.calls), value[1]]
                    for key, value in out.items()}

    engine = FaSyntheticEngine()
    engine.conditioner = Drifting()
    report = me.replay_check(engine, query, md, items[0][0], num_samples=4, prefixes=(1, 4),
                             batch_rows=8, source_chunk=3)
    assert report["bit_exact"] is False


def test_an_fa_dump_regenerates_its_selection_outside_the_scored_partition(tmp_path):
    query_id = "0|ir/A/A_idx_1/S001_R002_hybrid_IR.wav"
    engine, summary, out, _ = _fa_pass(tmp_path, dump_queries={query_id}, dump_top_n=2)
    assert summary["n_dumped"] == 1
    # the dump's one-forward regeneration is not a scored batch: the executed
    # partition is unchanged by it
    assert summary["fa_partition_executed"]["executed"] == FA_DECLARED
    row = json.load(open(me.query_artifact_paths(out, "A/A_idx_1", 0)["row"]))
    payload = np.load(os.path.join(out, row["waveform_path"]))
    sims = np.load(me.query_artifact_paths(out, "A/A_idx_1", 0)["sims"])
    embedder = ENG.SyntheticEngine()._embed
    for slot, index in enumerate(row["waveform_candidate_indices"]):
        position = row["candidate_indices"].index(index)
        wavs = torch.as_tensor(payload["waveforms"][slot]).unsqueeze(1)
        obs = torch.as_tensor(payload["observation"]).reshape(1, 1, -1)
        got = (embedder(wavs) @ embedder(obs)[0]).numpy()
        assert np.allclose(got, sims[position].astype(np.float32), atol=2e-3)


def test_an_fa_probe_publishes_its_executed_partition_and_no_score(tmp_path):
    engine, summary, out, _ = _fa_pass(tmp_path, probe=1)
    assert summary["n_scored"] == 0 and summary["probe_records"]
    assert summary["fa_partition_executed"]["executed"] == FA_DECLARED
    assert not os.path.isdir(os.path.join(out, me.ROWS_DIRNAME))


def test_a_resume_that_conditions_nothing_reports_an_empty_partition_without_refusing(tmp_path):
    _engine, _summary, out, (plan, records, items) = _fa_pass(tmp_path)
    done, _rejected = me.completed_queries(out)
    engine = FaSyntheticEngine()
    summary = me.run_pass(engine, items, records, plan, out, done=done,
                          fa_partition=FA_DECLARED, **FA_SHAPES)
    assert summary["n_scored"] == 0 and engine.conditioner.calls == []
    assert summary["fa_partition_executed"]["branches"]["source_cache"]["invocations"] == 0


# --------------------------------------------------------------------------- #
# the real stack's builder, the driver and the probe artifact
# --------------------------------------------------------------------------- #
class _ReachedTheCheckpoint(Exception):
    """Raised by a stubbed checkpoint read: the builder got past every gate."""


def test_the_engine_builder_admits_fa_with_its_orbit_and_goes_on_to_the_checkpoint(monkeypatch):
    model_config = json.load(open(os.path.join("src", "configs", "model_configs", "FLAC", "AR",
                                               "FLAC_AR.json")))
    monkeypatch.setattr(torch, "set_float32_matmul_precision", lambda _value: None)

    def _stop(*_args, **_kwargs):
        raise _ReachedTheCheckpoint()

    monkeypatch.setattr(torch, "load", _stop)
    with pytest.raises(_ReachedTheCheckpoint):
        me.build_mesh_engine("weights/exp20/BF_40k.ckpt", model_config, None,
                             cond_method="fa_invariant", frame_avg_angles=list(C4))


def _r2_binding(**overrides):
    payload = dict(ENG._binding(), frame_avg_angles=list(C4), rotate_deg=0.0,
                   cond_method="fa_invariant", fa_partition=DECLARED)
    payload.update(overrides)
    return payload


def test_the_probe_artifact_serializes_the_whole_r2_binding_and_the_executed_partition(tmp_path):
    """Codex rA A8 / rA-F1 F6(b): the probe artifact used to keep only the R1 fields."""
    record = me.probe_record("q", "R", 3, 8, {"conditioning": 0.5, "sampling": 1.0},
                             receiver_id="r", n_union=3)
    split = _split()
    _run_production_shapes(split)
    executed = split.recorder.report()
    binding = _r2_binding()
    path = me.write_probe_records(str(tmp_path), [record], binding=binding,
                                  binding_sha256=me.binding_sha256(binding),
                                  fa_partition_executed=executed)
    payload = json.load(open(path))
    assert payload["binding"] == {field: binding[field] for field in me.binding_fields(binding)}
    for field in me.RUN_BINDING_PROTOCOL_FIELDS:
        assert payload["binding"][field] == binding[field]
    assert payload["fa_partition_executed"]["executed"] == DECLARED
    # a legacy binding serializes exactly as before, and no executed block appears
    legacy = ENG._binding()
    path = me.write_probe_records(str(tmp_path / "legacy"), [record], binding=legacy,
                                  binding_sha256=me.binding_sha256(legacy))
    payload = json.load(open(path))
    assert payload["binding"] == {field: legacy[field] for field in me.RUN_BINDING_FIELDS}
    assert "fa_partition_executed" not in payload


def test_the_driver_preflight_engages_the_split_it_will_build(tmp_path, monkeypatch):
    import localize_meshgrid as driver

    seen = []
    real = me.assert_cacheable
    monkeypatch.setattr(me, "assert_cacheable",
                        lambda method, split=None: (seen.append((method, split)),
                                                    real(method, split))[1])
    out_dir, report_path, _base = ENG._fixture_audit(tmp_path)
    manifest = tmp_path / "d1.json"
    manifest.write_text(json.dumps({"records": [], "protocol_facts": {}}))
    args = driver.parse_args(["--ckpt-path", "x.ckpt", "--audit-report", report_path,
                              "--context-manifest", str(manifest), "--probe", "1"]
                             + ENG._FA_PROTOCOL)
    driver.validate_args(args)
    model_config = json.load(open(os.path.join("src", "configs", "model_configs", "FLAC", "AR",
                                               "FLAC_AR.json")))
    from src.localization import meshgrid_report as mr

    driver.preflight(args, me.load_audit_plan(report_path), model_config,
                     {"model_config": {"training": {"cond_method": "fa_invariant",
                                                    "frame_avg_angles": list(C4)}}},
                     ckpt_sha256=mr.REGISTERED_CKPT_SHA256["BF"],
                     agree_sha256=mr.REGISTERED_ARTIFACT_SHA256["agree_ckpt_sha256"],
                     model_config_sha256=mr.REGISTERED_ARTIFACT_SHA256["model_config_sha256"])
    (method, split), = seen
    assert method == "fa_invariant" and isinstance(split, me.FaSplit) and split.angles == C4


def test_the_driver_builds_the_split_and_hands_the_pass_the_declared_partition(tmp_path,
                                                                              monkeypatch):
    """The BF throughput probe's wiring, end to end through ``main`` with the stack stubbed."""
    import types

    import localize_meshgrid as driver
    from src.localization import agree_embed
    from src.localization import meshgrid_queries as mq

    out_dir, report_path, _base = ENG._fixture_audit(tmp_path)
    manifest = tmp_path / "d1.json"
    manifest.write_text(json.dumps({"records": [], "protocol_facts": {}}))
    ckpt, agree = tmp_path / "bf.ckpt", tmp_path / "agree.pt"
    ckpt.write_bytes(b"bf")
    agree.write_bytes(b"agree")
    probe_out = tmp_path / "probe"
    argv = (["--ckpt-path", str(ckpt), "--agree-ckpt", str(agree), "--audit-report", report_path,
             "--context-manifest", str(manifest), "--out-dir", str(probe_out), "--probe", "1",
             "--probe-room", "B/B_idx_2"] + ENG._FA_PROTOCOL)
    built, passed = {}, {}
    monkeypatch.setattr(mq, "load_manifest", lambda _path: {"records": []})
    monkeypatch.setattr(mq, "with_resolved_agree", lambda _config: {
        "training": {"metrics": {"AGREE_ckpt": str(agree)}}})
    monkeypatch.setattr(torch, "load", lambda *_a, **_k: {
        "model_config": {"training": {"cond_method": "fa_invariant",
                                      "frame_avg_angles": list(C4)}}})
    monkeypatch.setattr(agree_embed, "load_agree_audio", lambda path, device: types.SimpleNamespace(
        ckpt_sha256=me.file_sha256(path)))

    def _build(ckpt_path, model_config, scorer, **kwargs):
        built.update(kwargs)
        return FaSyntheticEngine(), {"weights_source": "stub", "latent_shape": (2, 4)}

    def _pass(engine, stream, records, plan, out, **kwargs):
        passed.update(kwargs)
        return {"probe_records": [], "n_generated": 0, "n_candidate_query_pairs": 0,
                "timings_s": {}, "fa_partition_executed": {"executed": DECLARED}}

    monkeypatch.setattr(me, "build_mesh_engine", _build)
    monkeypatch.setattr(me, "run_pass", _pass)
    monkeypatch.setattr(mq, "build_release_stack", lambda *_a: ([], {"call_graph": "stub"}))
    monkeypatch.setattr(me, "assert_release_rng_state", lambda _manifest: True)
    assert driver.main(argv) == 0
    assert built["cond_method"] == "fa_invariant" and built["frame_avg_angles"] == list(C4)
    assert passed["fa_partition"] == DECLARED
    payload = json.load(open(probe_out / "diagnostics_probe.json"))
    assert payload["binding"]["fa_partition"] == DECLARED
    assert payload["binding"]["frame_avg_angles"] == list(C4)
    assert payload["fa_partition_executed"] == {"executed": DECLARED}


# --------------------------------------------------------------------------- #
# gate 1 -- the split vs the released invariant_conditioning at IDENTICAL batching
# --------------------------------------------------------------------------- #
from src.localization import meshgrid_fa_parity as fp                    # noqa: E402

GATE_SCHEMA = {"gate", "name", "verdict", "match", "reasons", "tolerance", "inputs",
               "executed_partition", "provenance", "created_utc", "note"}


def _gate_query(tmp_path):
    plan, _records, items = _fa_aligned(tmp_path)
    query = me.load_room_plan(plan, "A/A_idx_1").queries[0]
    return query, me.GuardedMetadata(items[0][1])


class BatchSensitiveFa(FaFakeConditioner):
    """An output that moves with the BATCH SIZE, as a real backbone under autocast does."""

    OFFSET = 1e-4

    def __call__(self, batch_metadata, device, only_ids=None):
        out = super().__call__(batch_metadata, device, only_ids=only_ids)
        return {key: [value[0] + self.OFFSET * len(batch_metadata), value[1]]
                for key, value in out.items()}


def test_gate1_passes_bit_for_bit_and_records_full_and_tail_batches(tmp_path):
    query, md = _gate_query(tmp_path)
    report = fp.split_parity(FaSyntheticEngine(), query, md, micro_batch=3, n_candidates=7)
    assert report["gate"] == "G1" and report["name"] == "fa_split_parity"
    assert report["verdict"] == "PASS" and report["match"] is True and report["reasons"] == []
    assert report["tolerance"] == 0.0
    assert [chunk["n"] for chunk in report["chunks"]] == [3, 3, 1]
    assert report["coverage"] == {"full": 2, "tail": 1, "complete": True}
    for chunk in report["chunks"]:
        assert chunk["released_partition"]["partition"] == [chunk["n"]] * 3
        for key in FA_IDS:
            entry = chunk["conditioning"][key]
            assert entry["tensor"]["bitwise"] and entry["tensor"]["max_abs_diff"] == 0.0
            assert entry["mask"]["bitwise"]
            assert entry["tensor"]["digest"][0] == entry["tensor"]["digest"][1]
        assert all(entry["bitwise"] for entry in chunk["dit_inputs"].values())
    assert set(report["per_key"]) == set(FA_IDS) | {"dit_inputs.cond"}
    assert report["inputs"]["candidate_indices"] == [int(i) for i in
                                                     query.candidate_indices[:7]]
    assert set(report["executed_partition"]) >= {"executed", "violations", "branches"}


def test_gate1_holds_at_identical_batching_even_for_a_batch_sensitive_backbone(tmp_path):
    query, md = _gate_query(tmp_path)
    engine = FaSyntheticEngine()
    engine.conditioner = BatchSensitiveFa()
    report = fp.split_parity(engine, query, md, micro_batch=3, n_candidates=7)
    assert report["verdict"] == "PASS"


def test_gate1_fails_on_a_wrong_decomposition_and_names_the_key(tmp_path):
    query, md = _gate_query(tmp_path)

    class Unfaithful(FaFakeConditioner):
        """The only_ids seam computes context_poses differently from the full call."""

        def __call__(self, batch_metadata, device, only_ids=None):
            out = super().__call__(batch_metadata, device, only_ids=only_ids)
            if only_ids is not None and list(only_ids) == ["context_poses"]:
                out["context_poses"] = [out["context_poses"][0] + 0.5, out["context_poses"][1]]
            return out

    engine = FaSyntheticEngine()
    engine.conditioner = Unfaithful()
    report = fp.split_parity(engine, query, md, micro_batch=3, n_candidates=7)
    assert report["verdict"] == "FAIL" and report["match"] is False
    assert any("context_poses" in reason for reason in report["reasons"])
    assert report["per_key"]["context_poses"]["max_abs_diff"] == pytest.approx(0.5)
    assert report["per_key"]["source"]["max_abs_diff"] == 0.0


def test_gate1_without_full_and_tail_coverage_does_not_pass(tmp_path):
    query, md = _gate_query(tmp_path)
    report = fp.split_parity(FaSyntheticEngine(), query, md, micro_batch=3, n_candidates=6)
    assert report["coverage"] == {"full": 2, "tail": 0, "complete": False}
    assert report["verdict"] == "FAIL" and any("tail" in r for r in report["reasons"])


def test_the_gates_refuse_an_engine_without_the_split(tmp_path):
    query, md = _gate_query(tmp_path)
    with pytest.raises(ValueError, match="split"):
        fp.split_parity(ENG.SyntheticEngine(), query, md, micro_batch=3)


def test_gate_evidence_carries_its_provenance_and_is_written_atomically(tmp_path):
    query, md = _gate_query(tmp_path)
    report = fp.split_parity(FaSyntheticEngine(), query, md, micro_batch=3, n_candidates=7)
    environment = {"git_sha": "a" * 40, "git_status_dirty": False, "hostname": "box",
                   "gpus": []}
    evidence = fp.finalize_evidence(report, device="cpu", environment=environment,
                                    inputs={"ckpt_sha256": "5" * 64, "cond_autocast": "off"})
    assert GATE_SCHEMA <= set(evidence)
    assert evidence["provenance"]["git_sha"] == "a" * 40
    assert evidence["provenance"]["tracked_clean"] is True
    assert evidence["provenance"]["hostname"] == "box"
    assert evidence["provenance"]["gpu_uuid"] is None            # a CPU run names no card
    assert evidence["inputs"]["ckpt_sha256"] == "5" * 64
    assert evidence["inputs"]["query_id"] == query.query_id
    path = fp.write_evidence(str(tmp_path / "evidence"), evidence)
    assert os.path.basename(path) == "fa_split_parity.json"
    assert json.load(open(path)) == json.loads(json.dumps(evidence))
    assert not os.path.exists(path + ".tmp")


def test_a_bfloat16_tensor_is_digested_without_numpy():
    value = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    facts = fp.tensor_pair(value, value.clone(), 0.0)
    assert facts["bitwise"] and facts["dtype"] == ["torch.bfloat16", "torch.bfloat16"]
    assert facts["digest"][0] == facts["digest"][1] and len(facts["digest"][0]) == 64
    other = fp.tensor_pair(value, value.float(), 0.0)
    assert other["bitwise"] is False and any("dtype" in r for r in other["reasons"])
