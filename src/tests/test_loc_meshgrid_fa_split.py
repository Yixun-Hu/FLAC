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
    # the executed micro-batch is read off the calls (16 is still the largest observed), so a
    # second chunking is a VIOLATION of the one-stride rule rather than a moved headline
    assert any("more than one configured micro-batch" in reason
               for reason in split.recorder.violations())
    with pytest.raises(ValueError, match="violations"):
        me.assert_fa_partition_executed(DECLARED, split.recorder)

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
    """The engine fixture's synthetic stack, conditioned through the engaged split. Its DiT
    inputs are the three REGISTERED ones (``fp.REGISTERED_DIT_INPUTS``, Codex rB-F2 finding
    4): ``get_conditioning_inputs``' concatenation for FLAC_AR's id lists over the fake's
    tensors, with a CONSTANT cross-attention mask so that a mask corruption stays in the mask
    clause and a DiT-input corruption in the DiT-input clause, each isolated (Codex rB finding
    7). The vanilla SyntheticEngine, whose digests are frozen, keeps its own inputs."""

    def __init__(self, angles=C4):
        super().__init__()
        self.conditioner = FaFakeConditioner()
        self.cond_method = "fa_invariant"
        self.split = me.split_for("fa_invariant", angles)

    @staticmethod
    def _cond_inputs(conditioning):
        cross = ("context_poses_vit", "context_poses", "context_audio")
        return {"cross_attn_cond": torch.cat([conditioning[k][0] for k in cross], dim=-1),
                "cross_attn_mask": torch.ones(conditioning["context_audio"][0].shape[0], 3),
                "global_cond": torch.cat([conditioning[k][0] for k in ("source", "source_vit")],
                                         dim=-1)}

    def _sample(self, noise, cond):
        self.n_sampler_rows += int(noise.shape[0])
        level = torch.cat([cond["cross_attn_cond"], cond["global_cond"]], dim=-1).mean(dim=-1)
        return noise + level.reshape(-1, 1, 1)


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


def test_a_resume_that_conditions_nothing_keeps_the_runs_executed_partition(tmp_path):
    """Codex rB finding 5: a final no-op resume used to replace the summary's block with an
    empty recorder; the run's executed partition is its durable ledger's, every segment."""
    _engine, first, out, (plan, records, items) = _fa_pass(tmp_path)
    done, _rejected = me.completed_queries(out)
    engine = FaSyntheticEngine()
    summary = me.run_pass(engine, items, records, plan, out, done=done,
                          fa_partition=FA_DECLARED, **FA_SHAPES)
    assert summary["n_scored"] == 0 and engine.conditioner.calls == []
    executed = summary["fa_partition_executed"]
    assert executed["executed"] == FA_DECLARED and executed["violations"] == []
    assert executed["ledger"]["counts_sha256"] == first["fa_partition_executed"]["ledger"][
        "counts_sha256"]
    assert executed["ledger"]["n_segments"] == 2


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
    assert set(report["per_key"]) == set(FA_IDS) | {"dit_inputs.cross_attn_cond",
                                                    "dit_inputs.cross_attn_mask",
                                                    "dit_inputs.global_cond"}
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


# --------------------------------------------------------------------------- #
# gate 2 -- cached vs uncached memoization at matched batching, with counter-tests
# --------------------------------------------------------------------------- #
COUNTERS = {"moved_candidates": {"source", "source_vit", "context_poses"},
            "per_query_context_poses": {"context_poses"},
            "unaveraged_vit": {"source_vit", "context_poses_vit"},
            "swapped_candidates": {"source", "source_vit", "context_poses"}}


def test_gate2_memoization_is_bit_exact_and_every_counter_test_bites(tmp_path):
    query, md = _gate_query(tmp_path)
    report = fp.cache_parity(FaSyntheticEngine(), query, md, n_candidates=5, source_chunk=2,
                             micro_batch=3)
    assert report["gate"] == "G2" and report["name"] == "fa_cache_parity"
    assert report["verdict"] == "PASS" and report["reasons"] == []
    memo = report["memoization"]
    assert memo["match"] is True
    for key in FA_IDS:
        assert memo["conditioning"][key]["tensor"]["bitwise"]
        assert memo["conditioning"][key]["mask"]["bitwise"]
    assert all(entry["bitwise"] for entry in memo["dit_inputs"].values())
    assert set(report["counter_tests"]) == set(COUNTERS)
    for name, keys in COUNTERS.items():
        counter = report["counter_tests"][name]
        assert set(counter["keys"]) == keys and counter["detected"] is True, name
        assert all(value > 0.0 for value in counter["keys"].values()), name
    # matched batching IS one candidate per call, on both sides
    executed = report["executed_partition"]["executed"]
    assert executed["context_vit"]["candidate_micro_batch"] == 1
    assert executed["context_scored_batch"]["candidate_micro_batch"] == 1
    assert executed["source_cache"]["candidate_micro_batch"] == 1
    assert report["batched"]["informational"] is True
    assert report["inputs"]["source_chunk"] == 2 and report["inputs"]["n_candidates"] == 5


def test_gate2_separates_batch_nondeterminism_from_a_cache_error(tmp_path):
    query, md = _gate_query(tmp_path)
    engine = FaSyntheticEngine()
    engine.conditioner = BatchSensitiveFa()
    report = fp.cache_parity(engine, query, md, n_candidates=5, source_chunk=2, micro_batch=3)
    assert report["verdict"] == "PASS"                         # matched batching is exact
    assert report["batched"]["max_abs_diff"] > 0.0             # production shapes are not


def test_gate2_fails_on_a_cache_that_serves_something_else(tmp_path):
    query, md = _gate_query(tmp_path)

    class Unfaithful(FaFakeConditioner):
        """The source batch's rotated pass computes source_vit differently."""

        def __call__(self, batch_metadata, device, only_ids=None):
            out = super().__call__(batch_metadata, device, only_ids=only_ids)
            if only_ids is not None and list(only_ids) == ["source_vit"]:
                out["source_vit"] = [out["source_vit"][0] + 0.5, out["source_vit"][1]]
            return out

    engine = FaSyntheticEngine()
    engine.conditioner = Unfaithful()
    report = fp.cache_parity(engine, query, md, n_candidates=5, source_chunk=2, micro_batch=3)
    assert report["verdict"] == "FAIL" and report["memoization"]["match"] is False
    assert any(reason.startswith("memoization: source_vit") for reason in report["reasons"])


def test_gate2_fails_when_a_counter_test_cannot_bite(tmp_path):
    query, md = _gate_query(tmp_path)

    class Blind(FaFakeConditioner):
        """A backbone that ignores every pose: no comparison over it can fail."""

        def _row(self, key, row):
            return torch.zeros(3)

    engine = FaSyntheticEngine()
    engine.conditioner = Blind()
    report = fp.cache_parity(engine, query, md, n_candidates=5, source_chunk=2, micro_batch=3)
    assert report["memoization"]["match"] is True
    assert report["verdict"] == "FAIL"
    assert any("did not bite" in reason for reason in report["reasons"])


def test_the_vanilla_parity_refuses_an_engine_with_the_split(tmp_path):
    query, md = _gate_query(tmp_path)
    with pytest.raises(ValueError, match="fa_cache_parity"):
        me.cache_parity_check(FaSyntheticEngine(), query, md, n_candidates=3, source_chunk=2)


# --------------------------------------------------------------------------- #
# gate 3 -- production batching vs a faithful replay, scores and argmax included
# --------------------------------------------------------------------------- #
def _g3_cases(tmp_path, positions=(0, 3)):
    """Gate-3 cases for the fixture queries at ``positions`` (each with its receiver union)."""
    plan, _records, items = _fa_aligned(tmp_path)
    cases = []
    for room_id in ("A/A_idx_1", "B/B_idx_2"):
        room = me.load_room_plan(plan, room_id)
        for query in room.queries:
            if query.position in positions:
                obs, md = items[query.position]
                cases.append(fp.probe_case(room, query, me.GuardedMetadata(md), obs))
    return cases


G3_KW = {"batch_rows": 8, "source_chunk": 3, "num_samples": 4, "prefixes": (1, 4)}


def _ids(cases):
    """The registered probe-query set a gate over ``cases`` must consume (in case order)."""
    return [case["query"].query_id for case in cases]


#: the pinned anchor's registered constants -- its bytes' digest and (Codex rB-F2 finding 2)
#: its content digests -- which a fixture probe set re-registers for its own test
ANCHOR_CONSTANTS = ("COMMITTED_PROBE_REPORT_SHA256", "COMMITTED_PROBE_QUERY_IDS_SHA256",
                    "COMMITTED_PROBE_FROZEN_DIGESTS_SHA256")


@pytest.fixture(autouse=True)
def _registered_anchor(monkeypatch):
    """Every test gets the COMMITTED anchor constants back afterwards, whatever a fixture
    probe set re-registered (``_anchor_for`` assigns them directly; monkeypatch restores)."""
    for name in ANCHOR_CONSTANTS:
        monkeypatch.setattr(fp, name, getattr(fp, name, None), raising=False)


def _anchor_for(ids):
    """A pinned probe-report anchor over ``ids``, shaped as ``fp.load_probe_anchor`` returns it
    (gate 3's registered set is the pinned committed report's -- Codex rB-F1 finding 2). The
    fixture's ids and frozen map are RE-REGISTERED as the pinned report's content (Codex
    rB-F2 finding 2) for this test only -- the autouse ``_registered_anchor`` restores them."""
    ids = [str(query) for query in ids]
    digests = {query: {"row_sha256": "1" * 64, "sims_sha256": "2" * 64} for query in ids}
    fp.COMMITTED_PROBE_QUERY_IDS_SHA256 = canonical_sha256(ids)
    fp.COMMITTED_PROBE_FROZEN_DIGESTS_SHA256 = canonical_sha256(digests)
    return {"path": "fixture_probe_report.json", "sha256": fp.COMMITTED_PROBE_REPORT_SHA256,
            "pinned_sha256": fp.COMMITTED_PROBE_REPORT_SHA256, "query_ids": ids,
            "query_ids_sha256": canonical_sha256(ids), "n_queries": len(ids),
            "frozen_digests_sha256": canonical_sha256(digests), "digests": digests}


def _g3(engine, cases, **kwargs):
    """Gate 3 over ``cases`` with exactly those cases registered, anchored by a pinned report."""
    kwargs.setdefault("expected_queries", _ids(cases))
    kwargs.setdefault("anchor", _anchor_for(kwargs["expected_queries"] or []))
    return fp.production_parity(engine, cases, **kwargs)


def _quality_keys(payload, found=None):
    found = set() if found is None else found
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in ("prediction_index", "prediction_row", "prediction_xyz", "scores_hex",
                       "margin", "scores", "sims", "e_loc", "mean_scores_hex"):
                found.add(key)
            _quality_keys(value, found)
    elif isinstance(payload, list):
        for value in payload:
            _quality_keys(value, found)
    return found


def test_a_probe_case_carries_the_receivers_whole_union_in_camera_frame(tmp_path):
    plan, _records, items = _fa_aligned(tmp_path)
    room = me.load_room_plan(plan, "A/A_idx_1")
    query = room.queries[0]
    case = fp.probe_case(room, query, me.GuardedMetadata(items[0][1]), items[0][0])
    group = next(g for g in me.receiver_groups(room) if query in g.queries)
    assert case["union"] == [int(i) for i in group.union]
    assert np.array_equal(case["union_positions"],
                          room.base[np.asarray(group.union)] - group.receiver_xyz)
    assert case["query"] is query


def test_gate3_passes_on_a_batch_invariant_stack_and_covers_full_and_tail_batches(tmp_path):
    report = _g3(FaSyntheticEngine(), _g3_cases(tmp_path), **G3_KW,
                                  declared_partition=FA_DECLARED)
    assert report["gate"] == "G3" and report["name"] == "fa_production_parity"
    assert report["tolerance"] == 1e-3 and report["score_tolerance"] == me.SCORE_TOLERANCE
    assert report["verdict"] == "PASS", report["reasons"]
    assert report["coverage"]["scored"]["complete"] and report["coverage"]["source"]["complete"]
    assert len(report["queries"]) == 2
    for entry in report["queries"]:
        for k in ("1", "4"):
            assert entry["score_stability"][k]["max_abs_delta"] == 0.0
            assert entry["score_stability"][k]["argmax_equal"] is True
    assert report["score_stability"]["4"]["n_argmax_equal"] == 2
    assert report["executed_partition"]["executed"] == FA_DECLARED
    assert set(report["per_key"]) >= set(FA_IDS)
    # shapes, dtypes and masks per key; the replay's own orbit partition per batch
    for key in FA_IDS:
        assert report["per_key"][key]["shapes"] and report["per_key"][key]["dtypes"]
        assert report["per_key"][key]["mask_max_abs_diff"] == 0.0
    for entry in report["queries"]:
        assert all(chunk["replay_partition"]["partition"] == [chunk["n"]] * 3
                   for chunk in entry["chunks"])


def test_gate3_tolerates_backbone_batch_noise_within_the_registered_bound(tmp_path):
    engine = FaSyntheticEngine()
    engine.conditioner = BatchSensitiveFa()               # 1e-4 per row of batch size
    report = _g3(engine, _g3_cases(tmp_path), **G3_KW,
                                  declared_partition=FA_DECLARED)
    assert report["verdict"] == "PASS", report["reasons"]
    worst = max(entry["max_abs_diff"] for key, entry in report["per_key"].items()
                if not key.startswith("dit_inputs"))
    assert 0.0 < worst <= 1e-3
    assert any(not entry["bitwise"] for entry in report["per_key"].values())


def test_gate3_fails_above_the_registered_bound_and_names_the_keys(tmp_path):
    class Noisy(BatchSensitiveFa):
        OFFSET = 1e-2

    engine = FaSyntheticEngine()
    engine.conditioner = Noisy()
    report = _g3(engine, _g3_cases(tmp_path), **G3_KW,
                                  declared_partition=FA_DECLARED)
    assert report["verdict"] == "FAIL"
    assert any("context_audio" in reason for reason in report["reasons"])


def test_gate3_refuses_an_executed_partition_that_is_not_the_declared_one(tmp_path):
    declared = me.declared_fa_partition(list(C4), batch_rows=8, num_samples=4, source_chunk=16)
    report = _g3(FaSyntheticEngine(), _g3_cases(tmp_path), **G3_KW,
                                  declared_partition=declared)
    assert report["verdict"] == "FAIL"
    assert any("source_cache.candidate_micro_batch" in reason for reason in report["reasons"])


def test_gate3_without_a_tail_batch_does_not_pass(tmp_path):
    cases = _g3_cases(tmp_path)
    sizes = [case["query"].n_candidates for case in cases]
    # one candidate per scored batch: every scored batch is full, none is a tail
    report = _g3(FaSyntheticEngine(), cases, batch_rows=4, source_chunk=3,
                 num_samples=4, prefixes=(1, 4))
    assert report["coverage"]["scored"] == {"full": sum(sizes), "tail": 0, "complete": False}
    assert report["verdict"] == "FAIL" and any("coverage" in r for r in report["reasons"])


def test_gate3_evidence_reads_no_localization_quality(tmp_path):
    report = _g3(FaSyntheticEngine(), _g3_cases(tmp_path), **G3_KW,
                                  declared_partition=FA_DECLARED)
    evidence = fp.finalize_evidence(report, device="cpu",
                                    environment={"git_sha": "a" * 40, "git_status_dirty": False,
                                                 "hostname": "box"})
    assert _quality_keys(json.loads(json.dumps(evidence))) == set()


# --------------------------------------------------------------------------- #
# the driver's gate modes: record-free diagnostics that write evidence
# --------------------------------------------------------------------------- #
def _gate_args(tmp_path, *mode, protocol=None):
    import localize_meshgrid as driver

    argv = (["--ckpt-path", "x.ckpt", "--out-dir", str(tmp_path / "gates")]
            + list(ENG._FA_PROTOCOL if protocol is None else protocol) + list(mode))
    return driver.parse_args(argv)


def _off(protocol):
    swapped = list(protocol)
    swapped[swapped.index("--cond-autocast") + 1] = "off"
    return swapped


def test_the_gate_modes_validate_their_registered_conditions(tmp_path):
    import localize_meshgrid as driver

    g1 = _gate_args(tmp_path, "--fa-split-parity-check", protocol=_off(ENG._FA_PROTOCOL))
    assert driver.validate_args(g1) and driver.writes_query_artifacts(g1) is False
    # gate 1 is autocast OFF, and frame-averaged
    with pytest.raises(SystemExit, match="--cond-autocast off"):
        driver.validate_args(_gate_args(tmp_path, "--fa-split-parity-check"))
    with pytest.raises(SystemExit, match="fa_invariant"):
        driver.validate_args(_gate_args(tmp_path, "--fa-split-parity-check",
                                        protocol=_off(ENG._VANILLA_PROTOCOL)))
    # gate 3 is the PRODUCTION batching with autocast OFF (fp32; plan r2 amendment A4 i)
    g3 = _gate_args(tmp_path, "--fa-production-parity-check", protocol=_off(ENG._FA_PROTOCOL))
    assert driver.validate_args(g3) and driver.writes_query_artifacts(g3) is False
    for protocol, flag in ((ENG._FA_PROTOCOL, "--cond-autocast off"),
                           ([t if t != "256" else "64" for t in _off(ENG._FA_PROTOCOL)],
                            "--batch-rows"),
                           ([t if t != "16" else "8" for t in _off(ENG._FA_PROTOCOL)],
                            "--source-chunk")):
        with pytest.raises(SystemExit, match=flag):
            driver.validate_args(_gate_args(tmp_path, "--fa-production-parity-check",
                                            protocol=protocol))
    with pytest.raises(SystemExit, match="fa_invariant"):
        driver.validate_args(_gate_args(tmp_path, "--fa-production-parity-check",
                                        protocol=ENG._VANILLA_PROTOCOL))
    # one diagnostic at a time; gates are record-free
    with pytest.raises(SystemExit, match="one diagnostic"):
        driver.validate_args(_gate_args(tmp_path, "--cache-parity-check",
                                        "--fa-production-parity-check"))
    with pytest.raises(SystemExit, match="record-free"):
        driver.validate_args(_gate_args(tmp_path, "--fa-production-parity-check",
                                        "--launch-record", "l.json",
                                        protocol=_off(ENG._FA_PROTOCOL)))


def _driver_stack(tmp_path):
    plan, records, items = _fa_aligned(tmp_path)
    # the released loader yields a BATCHED observation tensor and a metadata list
    return plan, records, [(obs, [md]) for obs, md in items]


def test_the_driver_runs_gate1_and_writes_its_evidence(tmp_path, capsys, clean_env):
    import localize_meshgrid as driver

    plan, records, loader = _driver_stack(tmp_path)
    args = _gate_args(tmp_path, "--fa-split-parity-check", "--parity-candidates", "48",
                      protocol=_off(ENG._FA_PROTOCOL))
    binding = _registered_binding("BF", cond_autocast="off")
    code = driver._run_fa_gate(args, FaSyntheticEngine(), plan, records, loader, "G1", binding)
    assert code == 0
    evidence = json.load(open(tmp_path / "gates" / "fa_split_parity.json"))
    assert evidence["verdict"] == "PASS" and evidence["gate"] == "G1"
    assert evidence["inputs"]["binding"] == binding
    # the RESOLVED autocast state, never the flag's word (Codex rB finding 6; A4 iii)
    assert "cond_autocast" not in evidence["inputs"]
    assert evidence["inputs"]["cond_autocast_requested"] == "off"
    assert evidence["inputs"]["cond_autocast_enabled"] is False
    assert evidence["inputs"]["cond_autocast_dtype"] == "float32"
    assert evidence["coverage"]["complete"] is True             # 48 = 32 + a tail of 16
    assert evidence["provenance"]["record_free"] is True
    assert evidence["inputs"]["identity"]["arm"] == "BF"
    assert evidence["inputs"]["identity"]["deviations"] == {"cond_autocast": "off"}
    assert "GATE G1 fa_split_parity: PASS" in capsys.readouterr().out


def test_the_drivers_cache_parity_runs_gate2_under_fa_and_the_vanilla_proof_otherwise(
        tmp_path, capsys, clean_env):
    import localize_meshgrid as driver

    plan, records, loader = _driver_stack(tmp_path)
    args = _gate_args(tmp_path, "--cache-parity-check", "--parity-candidates", "5")
    assert driver._run_cache_parity(args, FaSyntheticEngine(), plan, records, loader,
                                    binding=_registered_binding("BF")) == 0
    evidence = json.load(open(tmp_path / "gates" / "fa_cache_parity.json"))
    assert evidence["verdict"] == "PASS" and evidence["inputs"]["n_candidates"] == 5
    assert "GATE G2 fa_cache_parity: PASS" in capsys.readouterr().out
    # vanilla: the original proof, printed as before, no fa evidence
    plan, records, items = ENG._aligned(tmp_path / "vanilla")
    args = driver.parse_args(["--ckpt-path", "x.ckpt", "--source-chunk", "2",
                              "--out-dir", str(tmp_path / "vanilla_gates")])
    assert driver._run_cache_parity(args, ENG.SyntheticEngine(), plan, records,
                                    [([obs], [md]) for obs, md in items]) == 0
    assert "MEMOIZATION" in capsys.readouterr().out
    assert not os.path.exists(tmp_path / "vanilla_gates" / "fa_cache_parity.json")


def _fixture_anchor(tmp_path, monkeypatch, plan):
    """The fixture's registered probe queries pinned as the committed probe report."""
    probes = me.registered_probe_queries(plan)
    anchor = _fake_anchor(tmp_path, monkeypatch, {
        probes[room]: {"row_sha256": "1" * 64, "sims_sha256": "2" * 64} for room in sorted(probes)},
        name="fixture_probe_report.json")
    return anchor


def test_the_driver_runs_gate3_over_the_registered_probe_queries(tmp_path, capsys, monkeypatch):
    import localize_meshgrid as driver

    plan, records, loader = _driver_stack(tmp_path)
    anchor = _fixture_anchor(tmp_path, monkeypatch, plan)
    args = _gate_args(tmp_path, "--fa-production-parity-check", "--replay-probe-report",
                      anchor["path"], protocol=_off(ENG._FA_PROTOCOL))
    args.batch_rows, args.source_chunk, args.num_samples, args.k_prefixes = 8, 3, 4, [1, 4]
    binding = {"fa_partition": FA_DECLARED, "cond_method": "fa_invariant"}
    code = driver._run_fa_gate(args, FaSyntheticEngine(), plan, records, loader, "G3", binding)
    evidence = json.load(open(tmp_path / "gates" / "fa_production_parity.json"))
    probes = me.registered_probe_queries(plan)
    assert evidence["inputs"]["queries"] == [probes[room] for room in sorted(probes)]
    assert evidence["declared_partition"] == FA_DECLARED
    assert evidence["executed_partition"]["executed"] == FA_DECLARED
    assert code == (0 if evidence["verdict"] == "PASS" else 1)
    assert f"GATE G3 fa_production_parity: {evidence['verdict']}" in capsys.readouterr().out
    # the exact ordered registered set and its digest travel with the verdict
    ordered = [probes[room] for room in sorted(probes)]
    assert evidence["probe_queries"]["registered"] == ordered
    assert evidence["probe_queries"]["registered_sha256"] == canonical_sha256(ordered)
    assert evidence["probe_queries"]["complete"] is True


# --------------------------------------------------------------------------- #
# the vanilla no-regression gate: the P1 probe queries' published rows, replayed
# --------------------------------------------------------------------------- #
PREREGISTERED = os.path.join(WORKLOG, "loc_meshgrid_preregistered_digests.json")
PROBE_REPORT = os.path.join(WORKLOG, "offgrid_probe_report.json")


def test_the_frozen_vanilla_digests_are_the_committed_p1_registration():
    assert fp.FROZEN_OBSERVATION_BANK_SHA256 == json.load(open(PREREGISTERED))[
        "observation_bank_sha256"]
    assert fp.FROZEN_OBSERVATION_BANK_SHA256.startswith("ee2ba80a")
    assert fp.COMMITTED_PROBE_REPORT == PROBE_REPORT
    report = json.load(open(PROBE_REPORT))
    assert report["run_gate"]["observation_bank_sha256"] == fp.FROZEN_OBSERVATION_BANK_SHA256
    frozen = fp.frozen_grid_digests(PROBE_REPORT)
    assert len(frozen) == 16
    for digests in frozen.values():
        assert set(digests) == {"row_sha256", "sims_sha256"}
        assert all(len(value) == 64 for value in digests.values())


def _published_vanilla_run(tmp_path, binding=None):
    """A published vanilla run of the fixture (the P1 stand-in) and its frozen grid digests."""
    plan, records, items = ENG._aligned(tmp_path)
    binding = ENG._binding(k_prefixes=[1, 4], num_samples=4) if binding is None else binding
    out = str(tmp_path / "published")
    os.makedirs(out)
    me.write_binding(out, binding, advisory={"batch_rows": 8, "source_chunk": 3})
    me.run_pass(ENG.SyntheticEngine(), items, records, plan, out,
                num_samples=binding["num_samples"], prefixes=tuple(binding["k_prefixes"]),
                batch_rows=8, source_chunk=3, binding_sha256=me.binding_sha256(binding))
    cases = []
    for room_id, position in (("A/A_idx_1", 0), ("B/B_idx_2", 3)):
        room = me.load_room_plan(plan, room_id)
        query = next(q for q in room.queries if q.position == position)
        obs, md = items[position]
        cases.append(fp.probe_case(room, query, me.GuardedMetadata(md), obs))
    probes = {case["query"].query_id for case in cases}
    frozen = {row["query_id"]: {"row_sha256": row["row_sha256"],
                                "sims_sha256": row["sims_sha256"]}
              for row in me.read_rows(out) if row["query_id"] in probes}
    return out, binding, frozen, cases


def _fake_anchor(tmp_path, monkeypatch, frozen, name="probe_report.json"):
    """A probe report over ``frozen`` written to disk and PINNED in place of the committed
    one (the fixture's stand-in for 8c5fd5e8...), read back through the real loader."""
    path = tmp_path / name
    path.write_text(json.dumps({"records": [
        {"query_id": query_id, "grid_row_sha256": digests["row_sha256"],
         "grid_sims_sha256": digests["sims_sha256"]} for query_id, digests in frozen.items()]}))
    import hashlib

    monkeypatch.setattr(fp, "COMMITTED_PROBE_REPORT_SHA256",
                        hashlib.sha256(path.read_bytes()).hexdigest())
    # ... and its CONTENT registered with it (Codex rB-F2 finding 2)
    monkeypatch.setattr(fp, "COMMITTED_PROBE_QUERY_IDS_SHA256",
                        canonical_sha256([str(query_id) for query_id in frozen]), raising=False)
    monkeypatch.setattr(fp, "COMMITTED_PROBE_FROZEN_DIGESTS_SHA256",
                        canonical_sha256({str(query_id): {"row_sha256": d["row_sha256"],
                                                          "sims_sha256": d["sims_sha256"]}
                                          for query_id, d in frozen.items()}), raising=False)
    return fp.load_probe_anchor(str(path))


G0_KW = {"seed": 42, "num_samples": 4, "tau": me.TAU}


def test_gate0_the_vanilla_path_reproduces_the_published_rows_exactly(tmp_path, monkeypatch):
    out, binding, frozen, cases = _published_vanilla_run(tmp_path)
    anchor = _fake_anchor(tmp_path, monkeypatch, frozen)
    report = fp.vanilla_replay_regression(
        ENG.SyntheticEngine(), cases, out, anchor=anchor, observation_bank_sha256="o" * 64,
        expected_observation_bank_sha256="o" * 64, run_binding=binding,
        expected_queries=_ids(cases), **G0_KW)
    assert report["gate"] == "G0" and report["name"] == "vanilla_replay_regression"
    assert report["verdict"] == "PASS", report["reasons"]
    assert len(report["queries"]) == 2
    for entry in report["queries"]:
        assert entry["frozen_digests_equal"] is True
        assert entry["matched"]["float16_bit_exact"] is True
        assert entry["matched"]["aggregate"]["max_abs_delta"] == 0.0
        assert (entry["matched"]["batch_rows"], entry["matched"]["source_chunk"]) == (8, 3)
    assert _quality_keys(json.loads(json.dumps(report))) == set()


def test_gate0_fails_when_the_vanilla_path_moves(tmp_path, monkeypatch):
    out, binding, frozen, cases = _published_vanilla_run(tmp_path)
    anchor = _fake_anchor(tmp_path, monkeypatch, frozen)

    class Shifted(ENG.FakeConditioner):
        def __call__(self, batch_metadata, device, only_ids=None):
            out_ = super().__call__(batch_metadata, device, only_ids=only_ids)
            return {key: [value[0] + 1e-2, value[1]] for key, value in out_.items()}

    engine = ENG.SyntheticEngine()
    engine.conditioner = Shifted()
    report = fp.vanilla_replay_regression(
        engine, cases, out, anchor=anchor, observation_bank_sha256="o" * 64,
        expected_observation_bank_sha256="o" * 64, run_binding=binding, **G0_KW)
    assert report["verdict"] == "FAIL"
    assert any("float16" in reason for reason in report["reasons"])


def test_gate0_fails_on_moved_frozen_digests_a_moved_bank_or_another_protocol(tmp_path,
                                                                              monkeypatch):
    out, binding, frozen, cases = _published_vanilla_run(tmp_path)
    first = cases[0]["query"].query_id
    tampered = dict(frozen, **{first: dict(frozen[first], sims_sha256="0" * 64)})
    report = fp.vanilla_replay_regression(
        ENG.SyntheticEngine(), cases, out, anchor=_fake_anchor(tmp_path, monkeypatch, tampered),
        observation_bank_sha256="o" * 64, expected_observation_bank_sha256="o" * 64,
        run_binding=binding, **G0_KW)
    assert report["verdict"] == "FAIL" and any("frozen" in r for r in report["reasons"])
    anchor = _fake_anchor(tmp_path, monkeypatch, frozen)
    report = fp.vanilla_replay_regression(
        ENG.SyntheticEngine(), cases, out, anchor=anchor, observation_bank_sha256="x" * 64,
        run_binding=binding, **G0_KW)
    assert report["verdict"] == "FAIL" and any("observation bank" in r for r in report["reasons"])
    report = fp.vanilla_replay_regression(
        ENG.SyntheticEngine(), cases, out, anchor=anchor, observation_bank_sha256="o" * 64,
        expected_observation_bank_sha256="o" * 64, run_binding=dict(binding, tau=0.05), **G0_KW)
    assert report["verdict"] == "FAIL" and any("tau" in r for r in report["reasons"])


def test_gate0_refuses_an_engine_with_the_split(tmp_path, monkeypatch):
    out, binding, frozen, cases = _published_vanilla_run(tmp_path)
    with pytest.raises(ValueError, match="vanilla"):
        fp.vanilla_replay_regression(FaSyntheticEngine(), cases, out,
                                     anchor=_fake_anchor(tmp_path, monkeypatch, frozen),
                                     observation_bank_sha256="o" * 64, **G0_KW)


def test_the_driver_runs_the_vanilla_regression_replay(tmp_path, monkeypatch, capsys, clean_env):
    import localize_meshgrid as driver
    from src.localization import meshgrid_offgrid_probe as mop

    out, binding, frozen, _cases = _published_vanilla_run(tmp_path, _registered_binding("P1"))
    plan, records, items = ENG._aligned(tmp_path / "stream")
    anchor = _fake_anchor(tmp_path, monkeypatch, frozen)
    report_path = tmp_path / "probe_report.json"
    monkeypatch.setattr(mop, "compute_observation_bank_digest",
                        lambda *a, **k: {"observation_bank_sha256":
                                         fp.FROZEN_OBSERVATION_BANK_SHA256})
    args = driver.parse_args(["--ckpt-path", "x.ckpt", "--replay-check", "--replay-against", out,
                              "--replay-probe-report", str(report_path), "--out-dir",
                              str(tmp_path / "gates")] + ENG._VANILLA_PROTOCOL)
    assert driver.validate_args(args) and driver.writes_query_artifacts(args) is False
    code = driver._run_replay_check(args, ENG.SyntheticEngine(), plan, records,
                                    [(obs, [md]) for obs, md in items], binding=binding)
    evidence = json.load(open(tmp_path / "gates" / "vanilla_replay_regression.json"))
    assert code == 0 and evidence["verdict"] == "PASS", evidence["reasons"]
    probes = me.registered_probe_queries(plan)
    assert [entry["query_id"] for entry in evidence["queries"]] == [probes[r] for r in
                                                                     sorted(probes)]
    assert "GATE G0 vanilla_replay_regression: PASS" in capsys.readouterr().out
    # the anchor's path, digest and exact key-set verification are published
    published = evidence["inputs"]["probe_report"]
    assert published["path"] == str(report_path)
    assert published["sha256"] == published["pinned_sha256"] == anchor["sha256"]
    assert published["query_ids"] == anchor["query_ids"]
    assert published["query_ids_sha256"] == canonical_sha256(anchor["query_ids"])
    assert evidence["inputs"]["anchor_key_set_equal"] is True
    assert evidence["inputs"]["identity"]["arm"] == "P1"
    assert evidence["verdict_clauses"] == {"comparison": True, "autocast": True,
                                           "identity": True, "provenance": True}
    # --replay-against is the VANILLA no-regression form of --replay-check only
    with pytest.raises(SystemExit, match="vanilla"):
        driver.validate_args(driver.parse_args(["--ckpt-path", "x.ckpt", "--replay-check",
                                                "--replay-against", out] + ENG._FA_PROTOCOL))
    with pytest.raises(SystemExit, match="--replay-check"):
        driver.validate_args(driver.parse_args(["--ckpt-path", "x.ckpt", "--probe", "1",
                                                "--replay-against", out]))


def test_gate3_publishes_a_digest_chain_per_key_and_side(tmp_path):
    """Per-chunk digests of 5,000-candidate queries would bloat the evidence; an ordered
    sha256 chain of them per key and side is compact and re-derivable by a re-run."""
    first = _g3(FaSyntheticEngine(), _g3_cases(tmp_path / "a"), **G3_KW)
    second = _g3(FaSyntheticEngine(), _g3_cases(tmp_path / "b"), **G3_KW)
    chains = first["digest_chains"]
    assert set(chains) >= set(FA_IDS)
    for key in FA_IDS:
        production, replay = chains[key]
        assert len(production) == len(replay) == 64
        assert production == replay                          # bit-identical on this stack
    assert chains == second["digest_chains"]                  # deterministic

    engine = FaSyntheticEngine()
    engine.conditioner = BatchSensitiveFa()
    noisy = _g3(engine, _g3_cases(tmp_path / "c"), **G3_KW)
    assert noisy["digest_chains"]["context_audio"][0] != noisy["digest_chains"]["context_audio"][1]


# --------------------------------------------------------------------------- #
# the REAL conditioning classes, on CPU: MultiConditioner + DistEmbedder + RIR
# encoder + GeometryConditioner over a tiny locally-built ViT (no weights read)
# --------------------------------------------------------------------------- #
def _tiny_real_conditioner():
    """FLAC_AR.json's conditioning block, the ViT swapped for a tiny SimpleViT."""
    import copy

    from src.models.conditioners import create_multi_conditioner_from_conditioning_config

    config = json.load(open(os.path.join("src", "configs", "model_configs", "FLAC", "AR",
                                         "FLAC_AR.json")))
    conditioning = copy.deepcopy(config["model"]["conditioning"])
    conditioning["cond_dim"] = 16
    for entry in conditioning["configs"]:
        if entry["type"] == "ViTCoordinates":
            entry["config"]["ViT"] = {"ch_dim": 3, "img_h": 8, "img_w": 16, "patch_h": 4,
                                      "patch_w": 4, "dim": 16, "depth": 1, "heads": 2,
                                      "mlp_dim": 16}
    torch.manual_seed(0)
    module = create_multi_conditioner_from_conditioning_config(conditioning).eval()
    cross = ("context_poses_vit", "context_poses", "context_audio")

    def conditioner(metadata, device, only_ids=None):
        with torch.no_grad():
            return module(metadata, device, only_ids=only_ids)

    def cond_inputs(conditioning_tensors):
        """``get_conditioning_inputs``' concatenation for FLAC_AR's id lists."""
        return {"cross_attn_cond": torch.cat([conditioning_tensors[k][0] for k in cross], dim=1),
                "cross_attn_mask": torch.cat([conditioning_tensors[k][1] for k in cross], dim=1),
                "global_cond": torch.cat([conditioning_tensors[k][0]
                                          for k in ("source", "source_vit")], dim=-1).squeeze(1)}
    return conditioner, cond_inputs


class TinyRealEngine(ENG.SyntheticEngine):
    """The synthetic generation stack, conditioned by the REAL conditioner classes."""

    def __init__(self):
        super().__init__()
        self.conditioner, self.cond_inputs_fn = _tiny_real_conditioner()
        self.cond_method = "fa_invariant"
        self.split = me.split_for("fa_invariant", C4)

    def _sample(self, noise, cond):
        self.n_sampler_rows += int(noise.shape[0])
        level = cond["cross_attn_cond"].mean(dim=(1, 2)) + cond["global_cond"].mean(dim=-1)
        return noise + level.reshape(-1, 1, 1)


def _real_cases(tmp_path):
    plan, _records, items = _fa_aligned(tmp_path)
    generator = torch.Generator().manual_seed(11)
    for _obs, md in items:
        md["depth"] = torch.rand((3, 8, 16), generator=generator, dtype=torch.float64) - 0.5
        md["context_audio"] = torch.randn((8, 1, 256), generator=generator)
    return plan, items


@pytest.mark.parametrize("n, micro", [(5, 2)])
def test_gate1_is_bit_exact_on_the_real_conditioning_classes(tmp_path, n, micro):
    plan, items = _real_cases(tmp_path)
    query = me.load_room_plan(plan, "A/A_idx_1").queries[0]
    report = fp.split_parity(TinyRealEngine(), query, me.GuardedMetadata(items[0][1]),
                             micro_batch=micro, n_candidates=n)
    assert report["verdict"] == "PASS", report["reasons"]
    assert set(report["per_key"]) == set(FA_IDS) | {"dit_inputs.cross_attn_cond",
                                                    "dit_inputs.cross_attn_mask",
                                                    "dit_inputs.global_cond"}
    assert all(entry["bitwise"] for entry in report["per_key"].values())


def test_gate2_is_bit_exact_and_bites_on_the_real_conditioning_classes(tmp_path):
    plan, items = _real_cases(tmp_path)
    query = me.load_room_plan(plan, "A/A_idx_1").queries[0]
    report = fp.cache_parity(TinyRealEngine(), query, me.GuardedMetadata(items[0][1]),
                             n_candidates=4, source_chunk=2, micro_batch=2)
    assert report["verdict"] == "PASS", report["reasons"]
    assert all(counter["detected"] for counter in report["counter_tests"].values())


def test_gate3_holds_on_the_real_conditioning_classes(tmp_path):
    plan, items = _real_cases(tmp_path)
    room = me.load_room_plan(plan, "B/B_idx_2")
    query = room.queries[0]
    obs, md = items[query.position]
    case = fp.probe_case(room, query, me.GuardedMetadata(md), obs)
    # a bounded slice keeps the CPU cost small: the first 9 candidates of the query
    query.candidate_indices = query.candidate_indices[:9]
    # source chunk 4: the union of 99 leaves a tail of 3 (chunk 3 would leave none)
    report = _g3(TinyRealEngine(), [case], batch_rows=8, source_chunk=4,
                 num_samples=4, prefixes=(1, 4))
    assert report["verdict"] == "PASS", report["reasons"]
    assert report["executed_partition"]["violations"] == []


def test_the_gate_cases_stop_the_stream_after_the_last_wanted_query(tmp_path):
    import localize_meshgrid as driver

    plan, records, loader = _driver_stack(tmp_path)
    args = _gate_args(tmp_path, "--fa-production-parity-check")
    pulled = []

    def stream():
        for position, item in enumerate(list(loader) + [loader[-1]]):   # one item past the end
            pulled.append(position)
            yield item

    cases = list(driver._probe_cases(args, plan, records, stream()))
    assert [case["query"].position for case in cases] == [0, 3]
    assert pulled == [0, 1, 2, 3]              # nothing after the last wanted query is loaded


def test_gate2_on_a_single_candidate_cannot_pass_and_does_not_crash(tmp_path):
    query, md = _gate_query(tmp_path)
    query.candidate_indices = query.candidate_indices[:1]
    report = fp.cache_parity(FaSyntheticEngine(), query, md, source_chunk=2, micro_batch=3)
    assert report["inputs"]["n_candidates"] == 1
    assert report["memoization"]["match"] is True
    # candidate dependence cannot be shown on one candidate: the gate says so
    assert report["verdict"] == "FAIL"
    assert any("per_query_context_poses did not bite" in r for r in report["reasons"])


# --------------------------------------------------------------------------- #
# exp_22 rB-F1 (Codex code review rB): gate integrity
# --------------------------------------------------------------------------- #
COMMITTED_PROBE_REPORT_SHA256 = "8c5fd5e8b35e380fe6bc66a328eb8353ce5ef75b56fc02a587ddb5c85957ffd3"


def test_the_g0_anchor_is_pinned_to_the_committed_probe_report_digest():
    """Codex rB finding 2: the frozen grid anchor is the committed report, by digest."""
    assert fp.COMMITTED_PROBE_REPORT_SHA256 == COMMITTED_PROBE_REPORT_SHA256
    anchor = fp.load_probe_anchor(PROBE_REPORT)
    assert anchor["sha256"] == anchor["pinned_sha256"] == COMMITTED_PROBE_REPORT_SHA256
    assert anchor["path"] == PROBE_REPORT
    assert anchor["n_queries"] == len(anchor["query_ids"]) == len(set(anchor["query_ids"])) == 16
    assert anchor["query_ids_sha256"] == canonical_sha256(anchor["query_ids"])
    assert anchor["digests"] == fp.frozen_grid_digests(PROBE_REPORT)
    assert sorted(anchor["digests"]) == sorted(anchor["query_ids"])


def test_a_replaced_probe_report_is_refused_as_the_anchor(tmp_path):
    payload = json.load(open(PROBE_REPORT))
    payload["records"][0]["grid_sims_sha256"] = "0" * 64
    replaced = tmp_path / "offgrid_probe_report.json"
    replaced.write_text(json.dumps(payload))
    for loader in (fp.load_probe_anchor, fp.frozen_grid_digests):
        with pytest.raises(ValueError, match="8c5fd5e8"):
            loader(str(replaced))
    # the same BYTES elsewhere are the same anchor (the path is published, not trusted)
    moved = tmp_path / "moved.json"
    moved.write_bytes(open(PROBE_REPORT, "rb").read())
    assert fp.load_probe_anchor(str(moved))["sha256"] == COMMITTED_PROBE_REPORT_SHA256


def test_the_driver_refuses_a_replaced_g0_anchor_before_any_replay(tmp_path, monkeypatch):
    import localize_meshgrid as driver
    from src.localization import meshgrid_offgrid_probe as mop

    out, binding, frozen, _cases = _published_vanilla_run(tmp_path)
    plan, records, items = ENG._aligned(tmp_path / "stream")
    replaced = tmp_path / "replaced.json"
    replaced.write_text(json.dumps({"records": [
        {"query_id": query_id, "grid_row_sha256": digests["row_sha256"],
         "grid_sims_sha256": digests["sims_sha256"]} for query_id, digests in frozen.items()]}))
    banked = []
    monkeypatch.setattr(mop, "compute_observation_bank_digest",
                        lambda *a, **k: banked.append(1) or {
                            "observation_bank_sha256": fp.FROZEN_OBSERVATION_BANK_SHA256})
    args = driver.parse_args(["--ckpt-path", "x.ckpt", "--replay-check", "--replay-against", out,
                              "--replay-probe-report", str(replaced), "--out-dir",
                              str(tmp_path / "gates"), "--num-samples", "4", "--k-prefixes",
                              "1", "4"] + ENG._VANILLA_PROTOCOL)
    engine = ENG.SyntheticEngine()
    with pytest.raises(SystemExit, match="8c5fd5e8"):
        driver._run_replay_check(args, engine, plan, records,
                                 [(obs, [md]) for obs, md in items], binding=binding)
    assert banked == [] and engine.n_sampler_rows == 0
    assert not os.path.exists(tmp_path / "gates" / "vanilla_replay_regression.json")


def test_gate0_requires_the_anchor_key_set_to_be_exactly_the_replayed_queries(tmp_path,
                                                                              monkeypatch):
    out, binding, frozen, cases = _published_vanilla_run(tmp_path)
    extra = dict(frozen, **{"9|ir/Z/Z_idx_9/S001_R001_hybrid_IR.wav": {
        "row_sha256": "1" * 64, "sims_sha256": "2" * 64}})
    report = fp.vanilla_replay_regression(
        ENG.SyntheticEngine(), cases, out, anchor=_fake_anchor(tmp_path, monkeypatch, extra),
        observation_bank_sha256="o" * 64, expected_observation_bank_sha256="o" * 64,
        run_binding=binding, **G0_KW)
    assert report["verdict"] == "FAIL" and report["inputs"]["anchor_key_set_equal"] is False
    assert any("key set" in reason for reason in report["reasons"])


def test_gate0_refuses_an_anchor_that_is_not_the_pinned_report(tmp_path, monkeypatch):
    out, binding, frozen, cases = _published_vanilla_run(tmp_path)
    anchor = _fake_anchor(tmp_path, monkeypatch, frozen)
    monkeypatch.setattr(fp, "COMMITTED_PROBE_REPORT_SHA256", "f" * 64)
    report = fp.vanilla_replay_regression(
        ENG.SyntheticEngine(), cases, out, anchor=anchor, observation_bank_sha256="o" * 64,
        expected_observation_bank_sha256="o" * 64, run_binding=binding, **G0_KW)
    assert report["verdict"] == "FAIL"
    assert any("pinned" in reason for reason in report["reasons"])


@pytest.mark.parametrize("deleted", ["early", "late"])
def test_the_gate_cases_refuse_a_stream_missing_a_registered_probe_query(tmp_path, deleted):
    """Codex rB finding 1: a partial stream used to reach the gate."""
    import localize_meshgrid as driver

    plan, records, loader = _driver_stack(tmp_path)
    probes = me.registered_probe_queries(plan)
    ordered = [probes[room] for room in sorted(probes)]
    gone = ordered[0] if deleted == "early" else ordered[-1]
    kept = [record for record in records if record["query_id"] != gone]
    args = _gate_args(tmp_path, "--fa-production-parity-check")
    with pytest.raises(SystemExit, match="registered probe quer"):
        list(driver._probe_cases(args, plan, kept, iter(loader)))


@pytest.mark.parametrize("deleted", ["early", "late"])
def test_gate3_and_gate0_fail_on_a_partial_probe_query_set(tmp_path, monkeypatch, deleted):
    cases = _g3_cases(tmp_path / "g3")
    registered = _ids(cases)
    partial = cases[1:] if deleted == "early" else cases[:-1]
    report = fp.production_parity(FaSyntheticEngine(), partial, **G3_KW,
                                  declared_partition=FA_DECLARED, expected_queries=registered)
    assert report["verdict"] == "FAIL" and report["probe_queries"]["complete"] is False
    missing = registered[0] if deleted == "early" else registered[-1]
    assert report["probe_queries"]["missing"] == [missing]
    assert any("registered probe quer" in reason for reason in report["reasons"])
    # an unstated set is never a complete one
    report = fp.production_parity(FaSyntheticEngine(), cases, **G3_KW,
                                  declared_partition=FA_DECLARED)
    assert report["verdict"] == "FAIL" and report["probe_queries"]["complete"] is False

    out, binding, frozen, g0_cases = _published_vanilla_run(tmp_path / "g0")
    anchor = _fake_anchor(tmp_path, monkeypatch, frozen)
    partial = g0_cases[1:] if deleted == "early" else g0_cases[:-1]
    report = fp.vanilla_replay_regression(
        ENG.SyntheticEngine(), partial, out, anchor=anchor, observation_bank_sha256="o" * 64,
        expected_observation_bank_sha256="o" * 64, run_binding=binding,
        expected_queries=_ids(g0_cases), **G0_KW)
    assert report["verdict"] == "FAIL" and report["probe_queries"]["complete"] is False
    assert any("registered probe quer" in reason for reason in report["reasons"])


def test_the_probe_query_set_records_the_exact_ordered_keys_and_their_digest():
    block, reasons = fp.probe_query_set(["b", "a", "c"], ["a", "b", "c"])
    assert reasons == [] and block["complete"] is True
    assert block["registered"] == ["b", "a", "c"] and block["consumed"] == ["a", "b", "c"]
    assert block["registered_sha256"] == canonical_sha256(["b", "a", "c"])
    assert block["consumed_sha256"] == canonical_sha256(["a", "b", "c"])
    assert block["n_registered"] == block["n_consumed"] == 3
    for consumed, field in ((["a", "b"], "missing"), (["a", "b", "c", "d"], "unexpected"),
                            (["a", "b", "c", "a"], "repeated")):
        block, reasons = fp.probe_query_set(["b", "a", "c"], consumed)
        assert block["complete"] is False and block[field] and reasons, field


def test_a_gate_may_not_be_bounded_to_one_room(tmp_path):
    """Plan r2 amendment A4 (v): G3 and G0 consume all 16 registered probe queries."""
    import localize_meshgrid as driver

    with pytest.raises(SystemExit, match="16 registered"):
        driver.validate_args(_gate_args(tmp_path, "--fa-production-parity-check",
                                        "--probe-room", "B/B_idx_2"))
    with pytest.raises(SystemExit, match="16 registered"):
        driver.validate_args(driver.parse_args(
            ["--ckpt-path", "x.ckpt", "--replay-check", "--replay-against", "run",
             "--probe-room", "B/B_idx_2"] + ENG._VANILLA_PROTOCOL))


def test_gate3_refuses_an_audit_whose_probe_set_is_not_the_committed_one(tmp_path, monkeypatch):
    import localize_meshgrid as driver

    plan, records, loader = _driver_stack(tmp_path)
    probes = me.registered_probe_queries(plan)
    first = probes[sorted(probes)[0]]
    anchor = _fake_anchor(tmp_path, monkeypatch, {first: {"row_sha256": "1" * 64,
                                                          "sims_sha256": "2" * 64}})
    args = _gate_args(tmp_path, "--fa-production-parity-check", "--replay-probe-report",
                      anchor["path"])
    args.batch_rows, args.source_chunk, args.num_samples, args.k_prefixes = 8, 3, 4, [1, 4]
    engine = FaSyntheticEngine()
    with pytest.raises(SystemExit, match="committed probe report"):
        driver._run_fa_gate(args, engine, plan, records, loader, "G3",
                            {"fa_partition": FA_DECLARED, "cond_method": "fa_invariant"})
    assert engine.n_sampler_rows == 0



# --------------------------------------------------------------------------- #
# plan r2 amendment A4 (i)-(iii): G3 is the fp32 LOGIC gate; the production
# autocast run is a disclosed drift MEASUREMENT; evidence names the resolved state
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode, enabled, dtype", [("off", False, "float32"),
                                                  ("default", True, "bfloat16"),
                                                  ("bf16", True, "bfloat16")])
def test_the_autocast_state_is_read_back_from_the_context_the_conditioner_runs_in(mode, enabled,
                                                                                  dtype):
    """On CPU torch's default autocast dtype is bfloat16 (on CUDA, float16) -- the state is
    whatever the engine's own context resolves to on this device, read back from torch."""
    state = me.cond_autocast_state("cpu", mode)
    assert state == {"cond_autocast_requested": mode, "cond_autocast_enabled": enabled,
                     "cond_autocast_dtype": dtype, "cond_autocast_device_type": "cpu"}
    with me.cond_autocast_context("cpu", *__import__("eval_FLAC").resolve_cond_autocast(mode)):
        assert torch.is_autocast_enabled("cpu") is enabled


def test_the_registered_default_resolves_to_float16_on_cuda_not_bf16():
    """The plan-text erratum (A4 iii): `default` is torch's per-device default -- fp16 on cuda."""
    from eval_FLAC import resolve_cond_autocast

    assert resolve_cond_autocast("default") == (True, None)
    assert torch.get_autocast_dtype("cuda") == torch.float16


def test_the_drift_measurement_mode_is_the_production_autocast_and_batching(tmp_path):
    import localize_meshgrid as driver

    drift = _gate_args(tmp_path, "--fa-production-drift-measurement")
    assert driver.validate_args(drift) and driver.writes_query_artifacts(drift) is False
    for protocol, flag in ((_off(ENG._FA_PROTOCOL), "--cond-autocast default"),
                           ([t if t != "256" else "64" for t in ENG._FA_PROTOCOL], "--batch-rows"),
                           ([t if t != "16" else "8" for t in ENG._FA_PROTOCOL], "--source-chunk"),
                           (ENG._VANILLA_PROTOCOL, "fa_invariant")):
        with pytest.raises(SystemExit, match=flag):
            driver.validate_args(_gate_args(tmp_path, "--fa-production-drift-measurement",
                                            protocol=protocol))
    with pytest.raises(SystemExit, match="one diagnostic"):
        driver.validate_args(_gate_args(tmp_path, "--fa-production-drift-measurement",
                                        "--fa-production-parity-check"))


class Noisy(BatchSensitiveFa):
    OFFSET = 1e-2


def test_the_drift_measurement_publishes_deltas_without_a_verdict(tmp_path):
    engine = FaSyntheticEngine()
    engine.conditioner = Noisy()                       # far above the gate's 1e-3
    cases = _g3_cases(tmp_path)
    report = _g3(engine, cases, **G3_KW, declared_partition=FA_DECLARED, measurement=True)
    assert report["gate"] == "G3D" and report["name"] == "fa_production_drift"
    assert report["verdict"] == "measurement" and report["match"] is None
    assert report["tolerance"] is None and report["complete"] is True
    assert report["reasons"] == []                     # no threshold is applied to a measurement
    assert max(entry["max_abs_diff"] for entry in report["per_key"].values()) > 1e-3
    assert report["context"]["vanilla_batching_caveat_max_abs_diff"] == 3.9e-3
    assert report["context"]["batching_caveat"] == me.BATCHING_CAVEAT
    # the same comparison as the gate: per-key deltas, score/argmax agreement, chains
    gate = _g3(engine, _g3_cases(tmp_path / "gate"), **G3_KW, declared_partition=FA_DECLARED)
    assert gate["verdict"] == "FAIL" and set(gate["per_key"]) == set(report["per_key"])
    assert set(report["score_stability"]) == set(gate["score_stability"])
    # structure still decides whether the measurement is COMPLETE (here: no source tail)
    report = _g3(engine, cases[1:], **G3_KW, declared_partition=FA_DECLARED, measurement=True)
    assert report["complete"] is False and report["verdict"] == "measurement"
    assert any("coverage" in reason for reason in report["reasons"])
    evidence = fp.finalize_evidence(report, device="cpu", environment={
        "git_sha": "a" * 40, "git_status_dirty": False, "hostname": "box"})
    assert _quality_keys(json.loads(json.dumps(evidence))) == set()


def test_the_evidence_holds_each_gate_to_its_registered_autocast(tmp_path):
    report = _g3(FaSyntheticEngine(), _g3_cases(tmp_path), **G3_KW,
                 declared_partition=FA_DECLARED)
    environment = {"git_sha": "a" * 40, "git_status_dirty": False, "hostname": "box"}
    identity = {"identity": fp.gate_identity("G3", _registered_binding("BF",
                                                                       cond_autocast="off"))[0]}
    passing = fp.finalize_evidence(report, device="cpu", environment=environment,
                                   inputs=dict(me.cond_autocast_state("cpu", "off"), **identity))
    assert passing["verdict"] == "PASS"
    failing = fp.finalize_evidence(report, device="cpu", environment=environment,
                                   inputs=dict(me.cond_autocast_state("cpu", "default"),
                                               **identity))
    assert failing["verdict"] == "FAIL" and failing["match"] is False
    assert any("autocast" in reason for reason in failing["reasons"])
    unstated = fp.finalize_evidence(report, device="cpu", environment=environment)
    assert unstated["verdict"] == "FAIL"


def test_the_driver_runs_the_drift_measurement_and_writes_its_own_evidence(tmp_path, monkeypatch,
                                                                         capsys, clean_env):
    import localize_meshgrid as driver

    plan, records, loader = _driver_stack(tmp_path)
    anchor = _fixture_anchor(tmp_path, monkeypatch, plan)
    args = _gate_args(tmp_path, "--fa-production-drift-measurement", "--replay-probe-report",
                      anchor["path"])                     # the REGISTERED 256 / 16 / K = 8
    engine = FaSyntheticEngine()
    engine.conditioner = Noisy()
    code = driver._run_fa_gate(args, engine, plan, records, loader, "G3D",
                               _registered_binding("BF"))
    evidence = json.load(open(tmp_path / "gates" / "fa_production_drift.json"))
    assert evidence["verdict"] == "measurement" and evidence["complete"] is True
    assert evidence["inputs"]["cond_autocast_requested"] == "default"
    assert evidence["inputs"]["cond_autocast_enabled"] is True
    assert "cond_autocast" not in evidence["inputs"]
    assert code == 0
    assert not os.path.exists(tmp_path / "gates" / "fa_production_parity.json")
    assert "fa_production_drift: measurement" in capsys.readouterr().out



# --------------------------------------------------------------------------- #
# Codex rB finding 3: a PASS needs the registered identity and captured provenance
# --------------------------------------------------------------------------- #
from src.localization import meshgrid_report as mr                       # noqa: E402

CLEAN_ENVIRONMENT = {"git_sha": "a" * 40, "git_status_dirty": False, "hostname": "box",
                     "gpus": []}


@pytest.fixture
def clean_env(monkeypatch):
    """A captured, tracked-clean environment for the driver's evidence (the worktree under
    test is usually dirty, which -- correctly -- makes every real gate non-PASS)."""
    from src.localization import meshgrid_offgrid_probe as mop

    monkeypatch.setattr(mop, "current_environment", lambda: dict(CLEAN_ENVIRONMENT))
    return CLEAN_ENVIRONMENT


def _registered_binding(arm, **overrides):
    """A run binding that IS arm ``arm``'s registered protocol (R2 generation)."""
    payload = dict(ENG._binding(), **mr.REGISTERED_PROTOCOL, **mr.REGISTERED_ARTIFACT_SHA256,
                   **mr.REGISTERED_ARM_PROTOCOL[arm], ckpt_sha256=mr.REGISTERED_CKPT_SHA256[arm])
    payload.update(overrides)
    return payload


def test_the_registered_binding_helper_is_the_registered_protocol():
    for arm in ("P1", "YAW", "BF"):
        verdict = mr.assert_registered_protocol(_registered_binding(arm), expect_arm=arm)
        assert verdict["is_registered"] and verdict["arm"] == arm


@pytest.mark.parametrize("gate, arm, overrides, expect_arm, ok", [
    ("G1", "BF", {"cond_autocast": "off"}, None, True),
    ("G1", "BF", {}, None, False),                     # G1 runs its registered autocast off
    ("G3", "BF", {"cond_autocast": "off"}, None, True),
    ("G3", "YAW", {"cond_autocast": "off"}, None, False),
    ("G3D", "BF", {}, None, True),
    ("G3D", "BF", {"cond_autocast": "off"}, None, False),
    ("G2", "BF", {}, None, True),
    ("G2", "BF", {"tau": 0.05}, None, False),
    ("G2", "BF", {"ckpt_sha256": "5" * 64}, None, False),
    ("G0", "P1", {}, None, True),
    ("G0", "BF", {}, None, False),
    ("GV", "YAW", {}, "YAW", True),
    ("GV", "P1", {}, "YAW", False),
    ("GV", "BF", {}, "BF", False),
])
def test_each_gate_requires_its_registered_arm_identity(gate, arm, overrides, expect_arm, ok):
    block, reasons = fp.gate_identity(gate, _registered_binding(arm, **overrides), expect_arm)
    assert block["registered_for_gate"] is ok and (reasons == []) is ok, reasons
    assert all(reason.startswith("identity:") for reason in reasons)


def _preflight(args, arm):
    import localize_meshgrid as driver

    # each arm's checkpoint embeds its own training policy (crossarm.ARMS identities)
    training = {"P1": {}, "YAW": {"yaw_aug": {"enabled": True, "img_w": 512, "seed": 42}},
                "BF": {"cond_method": "fa_invariant", "frame_avg_angles": list(C4)}}[arm]
    model_config = json.load(open(os.path.join("src", "configs", "model_configs", "FLAC", "AR",
                                               "FLAC_AR.json")))
    return driver.preflight(args, me.load_audit_plan(args.audit_report), model_config,
                            {"model_config": {"training": training}},
                            ckpt_sha256=mr.REGISTERED_CKPT_SHA256[arm],
                            agree_sha256=mr.REGISTERED_ARTIFACT_SHA256["agree_ckpt_sha256"],
                            model_config_sha256=mr.REGISTERED_ARTIFACT_SHA256[
                                "model_config_sha256"])


def _preflight_args(tmp_path, *mode, protocol):
    import localize_meshgrid as driver

    _out, report_path, _base = ENG._fixture_audit(tmp_path)
    manifest = tmp_path / "d1.json"
    manifest.write_text(json.dumps({"records": [], "protocol_facts": {}}))
    args = driver.parse_args(["--ckpt-path", "x.ckpt", "--audit-report", report_path,
                              "--context-manifest", str(manifest), "--out-dir",
                              str(tmp_path / "gates")] + list(protocol) + list(mode))
    driver.validate_args(args)
    return args


def test_the_driver_refuses_a_gate_on_an_unregistered_identity_before_any_device_work(tmp_path):
    """Preflight is CPU-only: a gate on the wrong arm (or a moved protocol) never starts."""
    g1 = _preflight_args(tmp_path / "g1", "--fa-split-parity-check",
                         protocol=_off(ENG._FA_PROTOCOL))
    binding, verdict = _preflight(g1, "BF")
    assert verdict["gate_identity"]["registered_for_gate"] is True
    with pytest.raises(SystemExit, match="identity"):
        _preflight(g1, "YAW")
    moved = _preflight_args(tmp_path / "g2", "--cache-parity-check", "--tau", "0.05",
                            protocol=ENG._FA_PROTOCOL)
    with pytest.raises(SystemExit, match="identity"):
        _preflight(moved, "BF")
    g0 = _preflight_args(tmp_path / "g0", "--replay-check", "--replay-against", "run",
                         protocol=ENG._VANILLA_PROTOCOL)
    assert _preflight(g0, "P1")[1]["gate_identity"]["arm"] == "P1"
    with pytest.raises(SystemExit, match="identity"):
        _preflight(g0, "YAW")
    # the YAW parity gate is the vanilla cache parity WITH its registered arm named
    yaw = _preflight_args(tmp_path / "yaw", "--cache-parity-check", "--expect-arm", "YAW",
                          protocol=ENG._VANILLA_PROTOCOL)
    assert _preflight(yaw, "YAW")[1]["gate_identity"]["arm"] == "YAW"
    with pytest.raises(SystemExit):
        _preflight(yaw, "P1")
    # without --expect-arm the vanilla cache parity is the R1 diagnostic: no gate, no verdict
    plain = _preflight_args(tmp_path / "plain", "--cache-parity-check",
                            protocol=ENG._VANILLA_PROTOCOL)
    assert _preflight(plain, "YAW")[1]["gate_identity"] is None


def _g1_report(tmp_path):
    query, md = _gate_query(tmp_path)
    return fp.split_parity(FaSyntheticEngine(), query, md, micro_batch=3, n_candidates=7)


def _g1_inputs(device="cpu"):
    return dict(me.cond_autocast_state(device, "off"),
                identity=fp.gate_identity("G1", _registered_binding("BF", cond_autocast="off"))[0])


@pytest.mark.parametrize("broken, why", [({"git_status_dirty": True}, "tracked tree"),
                                         ({"git_status_dirty": None}, "tracked tree"),
                                         ({"git_sha": None}, "git SHA"),
                                         ({"git_sha": "abc"}, "git SHA"),
                                         ({"hostname": None}, "hostname")])
def test_failed_provenance_makes_a_numerical_pass_non_pass(tmp_path, broken, why):
    report = _g1_report(tmp_path)
    assert report["verdict"] == "PASS"
    clean = fp.finalize_evidence(report, device="cpu", environment=CLEAN_ENVIRONMENT,
                                 inputs=_g1_inputs())
    assert clean["verdict"] == "PASS" and clean["provenance"]["complete"] is True
    evidence = fp.finalize_evidence(report, device="cpu",
                                    environment=dict(CLEAN_ENVIRONMENT, **broken),
                                    inputs=_g1_inputs())
    assert evidence["verdict"] == "FAIL" and evidence["match"] is False
    assert evidence["provenance"]["complete"] is False
    assert any(why in reason for reason in evidence["provenance"]["reasons"])
    assert any(reason.startswith("provenance:") for reason in evidence["reasons"])
    assert evidence["verdict_clauses"] == {"comparison": True, "autocast": True,
                                           "identity": True, "provenance": False}


def test_a_failed_cuda_uuid_capture_on_a_gpu_makes_the_gate_non_pass(tmp_path):
    report = _g1_report(tmp_path)
    failed = {"ok": False, "uuid": None, "error": "the runtime could not describe device 0"}
    evidence = fp.finalize_evidence(report, device="cuda:0", environment=CLEAN_ENVIRONMENT,
                                    runtime=failed, inputs=_g1_inputs("cuda:0"))
    assert evidence["verdict"] == "FAIL"
    assert any("CUDA" in reason and "could not describe" in reason
               for reason in evidence["provenance"]["reasons"])
    captured = {"ok": True, "uuid": "GPU-1234", "error": None}
    evidence = fp.finalize_evidence(report, device="cuda:0", environment=CLEAN_ENVIRONMENT,
                                    runtime=captured, inputs=_g1_inputs("cuda:0"))
    assert evidence["verdict"] == "PASS" and evidence["provenance"]["gpu_uuid"] == "GPU-1234"


def test_evidence_without_a_registered_identity_is_not_a_pass(tmp_path):
    report = _g1_report(tmp_path)
    bare = fp.finalize_evidence(report, device="cpu", environment=CLEAN_ENVIRONMENT,
                                inputs=me.cond_autocast_state("cpu", "off"))
    assert bare["verdict"] == "FAIL" and bare["verdict_clauses"]["identity"] is False
    wrong = fp.finalize_evidence(report, device="cpu", environment=CLEAN_ENVIRONMENT, inputs=dict(
        me.cond_autocast_state("cpu", "off"),
        identity=fp.gate_identity("G1", _registered_binding("YAW", cond_autocast="off"))[0]))
    assert wrong["verdict"] == "FAIL"
    assert any(reason.startswith("identity:") for reason in wrong["reasons"])


def test_the_driver_exits_non_zero_when_provenance_fails(tmp_path, monkeypatch):
    import localize_meshgrid as driver
    from src.localization import meshgrid_offgrid_probe as mop

    monkeypatch.setattr(mop, "current_environment",
                        lambda: dict(CLEAN_ENVIRONMENT, git_status_dirty=True))
    plan, records, loader = _driver_stack(tmp_path)
    args = _gate_args(tmp_path, "--fa-split-parity-check", "--parity-candidates", "48",
                      protocol=_off(ENG._FA_PROTOCOL))
    code = driver._run_fa_gate(args, FaSyntheticEngine(), plan, records, loader, "G1",
                               _registered_binding("BF", cond_autocast="off"))
    evidence = json.load(open(tmp_path / "gates" / "fa_split_parity.json"))
    assert code == 1 and evidence["verdict"] == "FAIL"
    assert evidence["verdict_clauses"]["comparison"] is True
    assert evidence["verdict_clauses"]["provenance"] is False


def test_the_yaw_parity_gate_publishes_its_evidence_under_the_yaw_identity(tmp_path, capsys,
                                                                          clean_env):
    import localize_meshgrid as driver

    plan, records, items = ENG._aligned(tmp_path)
    loader = [([obs], [md]) for obs, md in items]
    args = driver.parse_args(["--ckpt-path", "x.ckpt", "--cache-parity-check", "--expect-arm",
                              "YAW", "--out-dir", str(tmp_path / "gates")]
                             + ENG._VANILLA_PROTOCOL)
    assert driver.validate_args(args)
    assert driver._run_cache_parity(args, ENG.SyntheticEngine(), plan, records, loader,
                                    binding=_registered_binding("YAW")) == 0
    evidence = json.load(open(tmp_path / "gates" / "vanilla_cache_parity.json"))
    assert evidence["gate"] == "GV" and evidence["verdict"] == "PASS"
    assert evidence["inputs"]["identity"]["arm"] == "YAW"
    assert evidence["vanilla_cache_parity"]["match"] is True
    assert "MEMOIZATION" in capsys.readouterr().out
    # the same gate on another arm's binding is not a PASS
    assert driver._run_cache_parity(args, ENG.SyntheticEngine(), plan, records, loader,
                                    binding=_registered_binding("P1")) == 1


def test_gate3_passes_end_to_end_at_the_registered_numbers_and_identity(tmp_path, monkeypatch,
                                                                       capsys, clean_env):
    import localize_meshgrid as driver

    plan, records, loader = _driver_stack(tmp_path)
    anchor = _fixture_anchor(tmp_path, monkeypatch, plan)
    args = _gate_args(tmp_path, "--fa-production-parity-check", "--replay-probe-report",
                      anchor["path"], protocol=_off(ENG._FA_PROTOCOL))
    assert driver.validate_args(args)
    code = driver._run_fa_gate(args, FaSyntheticEngine(), plan, records, loader, "G3",
                               _registered_binding("BF", cond_autocast="off"))
    evidence = json.load(open(tmp_path / "gates" / "fa_production_parity.json"))
    assert code == 0 and evidence["verdict"] == "PASS", evidence["reasons"]
    assert evidence["verdict_clauses"] == {"comparison": True, "autocast": True,
                                           "identity": True, "provenance": True}
    assert evidence["executed_partition"]["executed"] == mr.REGISTERED_ARM_PROTOCOL["BF"][
        "fa_partition"]
    assert evidence["probe_queries"]["complete"] is True


# --------------------------------------------------------------------------- #
# Codex rB finding 7: one isolated negative per G3 (and G1) verdict clause --
# each mutation breaks exactly ONE clause, and the verdict must say which
# --------------------------------------------------------------------------- #
def _only(reasons, needle):
    """``reasons`` is non-empty and every reason belongs to the clause ``needle`` names."""
    assert reasons, "the mutation was not detected"
    assert all(needle in reason for reason in reasons), reasons


def _g3_registered(tmp_path, cases=None, **kwargs):
    kwargs = dict(G3_KW, declared_partition=FA_DECLARED, **kwargs)
    return _g3(FaSyntheticEngine(), _g3_cases(tmp_path) if cases is None else cases, **kwargs)


def test_gate3_fails_on_score_drift_alone(tmp_path, monkeypatch):
    """Identical conditioning, every replay similarity shifted by 0.01: |dS| = 0.01 at every
    K with the argmax unmoved -- only the score clause can catch it."""
    from src.localization import scoring

    state, real_sims = {"replay": False}, scoring.cosine_sims
    real_score, real_released = me._score_one_query, fp.released_conditioning
    monkeypatch.setattr(me, "_score_one_query", lambda *a, **k: (
        state.update(replay=False), real_score(*a, **k))[1])
    monkeypatch.setattr(fp, "released_conditioning", lambda *a, **k: (
        real_released(*a, **k), state.update(replay=True))[0])
    monkeypatch.setattr(scoring, "cosine_sims", lambda *a, **k: real_sims(*a, **k) + (
        0.01 if state["replay"] else 0.0))
    report = _g3_registered(tmp_path)
    assert report["verdict"] == "FAIL"
    _only(report["reasons"], "score |diff|")
    assert all(entry["n_argmax_equal"] == entry["n_queries"]
               for entry in report["score_stability"].values())


def test_gate3_fails_on_argmax_movement_alone(tmp_path, monkeypatch):
    """Identical conditioning and scores; the replay side's argmax reported elsewhere."""
    calls, real = [], me.argmax_by_global_index

    def moved(scores, indices):
        calls.append(1)
        found = real(scores, indices)
        return found if len(calls) % 2 else found + 1      # the second of each pair moves

    monkeypatch.setattr(me, "argmax_by_global_index", moved)
    report = _g3_registered(tmp_path)
    assert report["verdict"] == "FAIL"
    _only(report["reasons"], "argmax moved")
    assert all(entry["max_abs_delta"] == 0.0 for entry in report["score_stability"].values())


def test_gate3_fails_without_a_source_tail_alone(tmp_path):
    """Query 3's union of 99 at source chunk 3: 33 full source batches and no tail, while
    its 99 candidates at micro 2 still leave a scored tail."""
    cases = [case for case in _g3_cases(tmp_path) if case["query"].position == 3]
    assert len(cases[0]["union"]) == 99
    report = _g3_registered(tmp_path, cases=cases)
    assert report["coverage"]["source"] == {"full": 33, "tail": 0, "complete": False}
    assert report["coverage"]["scored"]["complete"] is True
    assert report["verdict"] == "FAIL"
    _only(report["reasons"], "coverage: source")


def test_gate3_and_gate1_fail_on_a_replay_orbit_that_is_not_per_angle(tmp_path, monkeypatch):
    """The replay batched two angles per forward (cap = 2 x batch): per-row values are
    unchanged, so only the replay-partition clause can catch it."""
    from src.localization import crossarm

    monkeypatch.setattr(crossarm, "fa_max_fwd_samples", lambda metadata: 2 * len(metadata))
    report = _g3_registered(tmp_path)
    assert report["verdict"] == "FAIL"
    _only(report["reasons"], "replay orbit partition")
    assert all(entry["max_abs_diff"] == 0.0 for entry in report["per_key"].values())
    query, md = _gate_query(tmp_path / "g1")
    g1 = fp.split_parity(FaSyntheticEngine(), query, md, micro_batch=3, n_candidates=7)
    assert g1["verdict"] == "FAIL"
    _only(g1["reasons"], "replay orbit partition")


def _corrupt_the_compared_production_block(monkeypatch, mutate):
    """Mutate ONLY the production block gate 3 compares (``scored_conditioning(record=False)``):
    the generated waveforms on both sides stay untouched."""
    real = me.scored_conditioning

    def corrupted(*args, **kwargs):
        block = real(*args, **kwargs)
        return block if kwargs.get("record", True) else mutate(block)

    monkeypatch.setattr(me, "scored_conditioning", corrupted)


@pytest.mark.parametrize("mutation", ["values", "dtype", "absent", "shape"])
def test_gate3_fails_on_a_mask_only_corruption(tmp_path, monkeypatch, mutation):
    def mutate(block):
        tensor, mask = block["context_audio"]
        mask = {"values": lambda m: 1.0 - m, "dtype": lambda m: m.bool(),
                "absent": lambda m: None, "shape": lambda m: torch.cat([m, m], dim=1)}[
                    mutation](mask)
        return dict(block, context_audio=[tensor, mask])

    _corrupt_the_compared_production_block(monkeypatch, mutate)
    report = _g3_registered(tmp_path)
    assert report["verdict"] == "FAIL"
    _only(report["reasons"], "context_audio.mask")
    assert report["per_key"]["context_audio"]["max_abs_diff"] == 0.0     # the tensor is clean
    assert all(entry["max_abs_delta"] == 0.0 for entry in report["score_stability"].values())


def test_gate3_fails_on_a_dit_input_only_corruption(tmp_path, monkeypatch):
    """The five blocks agree, but what the DiT is handed for the production block does not."""

    class Tagged(dict):
        pass

    _corrupt_the_compared_production_block(monkeypatch, Tagged)
    engine = FaSyntheticEngine()
    inner = engine.cond_inputs_fn
    engine.cond_inputs_fn = lambda block: {key: value + (0.5 if isinstance(block, Tagged) else 0.0)
                                           for key, value in inner(block).items()}
    report = _g3(engine, _g3_cases(tmp_path), **G3_KW, declared_partition=FA_DECLARED)
    assert report["verdict"] == "FAIL"
    _only(report["reasons"], "dit_inputs.")
    assert all(report["per_key"][key]["max_abs_diff"] == 0.0 for key in FA_IDS)


def test_gate3_chains_and_per_key_facts_carry_the_masks(tmp_path, monkeypatch):
    report = _g3_registered(tmp_path)
    for key in FA_IDS:
        production, replay = report["digest_chains"][f"{key}.mask"]
        assert len(production) == 64 and production == replay
        assert report["per_key"][key]["mask_dtypes"] == ["torch.float32/torch.float32"]
        assert report["per_key"][key]["mask_shapes"]

    def drop(block):
        return dict(block, source=[block["source"][0], None])

    _corrupt_the_compared_production_block(monkeypatch, drop)
    report = _g3_registered(tmp_path / "absent")
    production, replay = report["digest_chains"]["source.mask"]
    assert production != replay                     # mask presence is part of the chain



# --------------------------------------------------------------------------- #
# Codex rB finding 5 (part 1): the executed partition is CALL-derived and compared
# over the exact schema both ways; NIT 8: assert_cacheable checks the whole split
# --------------------------------------------------------------------------- #
def _observe_production(recorder, *, scored=(32, 8), source=(16, 4), vit=(1,), configured=None,
                        orbit_size=4):
    configured = configured or {}
    for n in vit:
        recorder.observe("context_vit", [(n, False)] + [(n, True)] * 3, cap=n,
                         configured=configured.get("context_vit", 1), orbit_size=orbit_size)
    for n in scored:
        recorder.observe("context_scored_batch", [(n, False)], cap=n,
                         configured=configured.get("context_scored_batch", 32),
                         orbit_size=orbit_size)
    for n in source:
        recorder.observe("source_cache", [(n, False)] + [(n, True)] * 3, cap=n,
                         configured=configured.get("source_cache", 16), orbit_size=orbit_size)
    return recorder


def test_the_executed_partition_is_read_off_the_calls_not_the_configured_values():
    full = _observe_production(me.FaPartitionRecorder())
    assert full.executed() == DECLARED
    assert all(entry["saturated"] for entry in full.report()["branches"].values())
    # the loop SAYS 32 and 16, but every call it made was smaller: that is what ran
    tails = _observe_production(me.FaPartitionRecorder(), scored=(20, 7), source=(9,))
    executed = tails.executed()
    assert executed["context_scored_batch"]["candidate_micro_batch"] == 20
    assert executed["source_cache"] == {"candidate_micro_batch": 9, "frame_avg_fwd_cap": 9}
    branches = tails.report()["branches"]
    assert branches["context_scored_batch"]["saturated"] is False
    assert branches["context_scored_batch"]["observed_max_base_batch"] == 20
    assert branches["context_vit"]["saturated"] is True
    # the orbit size is counted from the passes made, not copied from the argument
    short = me.FaPartitionRecorder()
    short.observe("source_cache", [(16, False), (16, True), (16, True)], cap=16, configured=16,
                  orbit_size=4)
    assert short.executed()["orbit_size"] == 3


def test_declared_vs_executed_compares_the_exact_schema_both_ways():
    executed = _observe_production(me.FaPartitionRecorder()).executed()
    assert me.fa_partition_differences(DECLARED, executed) == []
    extra_top = dict(executed, surprise=1)
    extra_nested = dict(executed, context_scored_batch=dict(executed["context_scored_batch"],
                                                            frame_avg_fwd_cap=32))
    missing_nested = dict(executed, source_cache={"candidate_micro_batch": 16})
    flattened = dict(executed, context_vit=1)
    missing_top = {key: value for key, value in executed.items() if key != "orbit_size"}
    for found, path in ((extra_top, "surprise"),
                        (extra_nested, "context_scored_batch.frame_avg_fwd_cap"),
                        (missing_nested, "source_cache.frame_avg_fwd_cap"),
                        (flattened, "context_vit"), (missing_top, "orbit_size")):
        assert me.fa_partition_differences(DECLARED, found) == [path], path
        # and the other way round: a declaration with the extra key is not this execution
        assert me.fa_partition_differences(found, executed) == [path], path


def test_a_tail_only_prefix_passes_the_per_row_check_but_not_the_final_one():
    """Bathrooms_idx_14's z_band queries all have < 32 candidates: a per-row check over a
    prefix that has not yet run a full scored batch is consistent, not a difference; the
    pass's FINAL check needs every declared micro-batch observed at full size."""
    tails = _observe_production(me.FaPartitionRecorder(), scored=(24, 31))
    assert me.assert_fa_partition_executed(DECLARED, tails, final=False)["violations"] == []
    with pytest.raises(ValueError, match=r"context_scored_batch\.candidate_micro_batch"):
        me.assert_fa_partition_executed(DECLARED, tails)
    # a LARGER batch than declared is never a prefix, nor is a loop at another stride
    over = _observe_production(me.FaPartitionRecorder(), scored=(40,),
                               configured={"context_scored_batch": 40})
    with pytest.raises(ValueError, match=r"context_scored_batch\.candidate_micro_batch"):
        me.assert_fa_partition_executed(DECLARED, over, final=False)
    rechunked = _observe_production(me.FaPartitionRecorder(), source=(8, 8, 4),
                                    configured={"source_cache": 8})
    with pytest.raises(ValueError, match=r"source_cache\.candidate_micro_batch"):
        me.assert_fa_partition_executed(DECLARED, rechunked, final=False)


def test_a_non_orbit_branch_never_makes_rotated_forwards():
    recorder = _observe_production(me.FaPartitionRecorder())
    recorder.observe("context_scored_batch", [(32, False), (32, True)], cap=32, configured=32,
                     orbit_size=4)
    assert any("context_scored_batch" in reason and "rotated" in reason
               for reason in recorder.violations())


def test_a_pass_whose_scored_batches_never_fill_refuses_at_the_end_but_a_probe_reports_it(
        tmp_path):
    """Micro-batch 200 exceeds every fixture query: each row passes the prefix rule, the
    pass's final check refuses, and the no-quality probe publishes what ran instead."""
    declared = me.declared_fa_partition(list(C4), batch_rows=800, num_samples=4, source_chunk=3)
    with pytest.raises(ValueError, match=r"context_scored_batch\.candidate_micro_batch"):
        _fa_pass(tmp_path / "pass", batch_rows=800, fa_partition=declared)
    _engine, summary, _out, _ = _fa_pass(tmp_path / "probe", batch_rows=800,
                                         fa_partition=declared, probe=1)
    executed = summary["fa_partition_executed"]
    assert executed["executed"]["context_scored_batch"]["candidate_micro_batch"] < 200
    assert executed["branches"]["context_scored_batch"]["saturated"] is False


@pytest.mark.parametrize("split, why", [
    (me.FaSplit(angles=C4), "recorder"),
    (me.FaSplit(angles=C4, recorder=object()), "recorder"),
    (me.FaSplit(angles=(), recorder=me.FaPartitionRecorder()), "orbit"),
    (me.FaSplit(angles=(90.0, 0.0, 180.0, 270.0), recorder=me.FaPartitionRecorder()), "identity"),
    (me.FaSplit(angles=(0.0, 90.0, 90.0), recorder=me.FaPartitionRecorder()), "repeats"),
    (me.FaSplit(angles=(0.0, float("nan")), recorder=me.FaPartitionRecorder()), "finite"),
])
def test_assert_cacheable_refuses_a_malformed_split(split, why):
    """Codex rB NIT 8: a hand-built FaSplit is checked whole, not by name alone."""
    with pytest.raises(ValueError, match=why):
        me.assert_cacheable("fa_invariant", split)
    engine = FaSyntheticEngine()
    engine.split = split
    with pytest.raises(ValueError, match=why):
        fp.require_split(engine)



# --------------------------------------------------------------------------- #
# Codex rB finding 5 (part 2): the executed partition is DURABLE -- a per-query
# ledger across segments -- and every shard's is re-derived at the merge
# --------------------------------------------------------------------------- #
import test_loc_meshgrid_launch_record as LR                             # noqa: E402


def test_recorder_counts_round_trip_exactly():
    recorder = _observe_production(me.FaPartitionRecorder(), scored=(32, 32, 8))
    again = me.FaPartitionRecorder.from_counts(recorder.to_counts())
    assert again.counts == recorder.counts
    both = me.FaPartitionRecorder.from_counts(recorder.to_counts(), recorder.to_counts())
    assert both.report()["branches"]["context_scored_batch"]["invocations"] == 6
    with pytest.raises(ValueError, match="branch"):
        me.FaPartitionRecorder.from_counts([dict(recorder.to_counts()[0], branch="nowhere")])


def _ledger(out):
    return json.load(open(os.path.join(out, me.FA_LEDGER_FILENAME)))


def test_the_ledger_is_written_before_every_row_and_survives_a_crash(tmp_path, monkeypatch):
    real, written = me.write_query_artifact, []

    def crash_on_the_third(*args, **kwargs):
        if len(written) == 2:
            raise RuntimeError("the card fell over")
        written.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(me, "write_query_artifact", crash_on_the_third)
    plan, records, items = _fa_aligned(tmp_path)
    out = str(tmp_path / "run")
    with pytest.raises(RuntimeError, match="fell over"):
        me.run_pass(FaSyntheticEngine(), items, records, plan, out, fa_partition=FA_DECLARED,
                    **FA_SHAPES)
    ledger = _ledger(out)
    assert ledger["declared"] == FA_DECLARED and len(ledger["segments"]) == 1
    assert ledger["segments"][0]["n_queries"] == 3          # the crashed query's calls too
    monkeypatch.setattr(me, "write_query_artifact", real)
    done, _rejected = me.completed_queries(out)
    assert len(done) == 2
    summary = me.run_pass(FaSyntheticEngine(), items, records, plan, out, done=done,
                          fa_partition=FA_DECLARED, **FA_SHAPES)
    ledger = _ledger(out)
    assert [segment["n_queries"] for segment in ledger["segments"]] == [3, 2]
    merged = me.FaPartitionRecorder.from_counts(*[s["counts"] for s in ledger["segments"]])
    assert summary["fa_partition_executed"]["executed"] == merged.executed() == FA_DECLARED
    assert summary["fa_partition_executed"]["ledger"]["counts_sha256"] == canonical_sha256(
        merged.to_counts()) == ledger["merged"]["counts_sha256"]


def test_a_ledger_for_another_declaration_is_refused(tmp_path):
    _engine, _summary, out, (plan, records, items) = _fa_pass(tmp_path)
    other = me.declared_fa_partition(list(C4), batch_rows=8, num_samples=4, source_chunk=5)
    with pytest.raises(ValueError, match="declar"):
        me.run_pass(FaSyntheticEngine(), items, records, plan, out, fa_partition=other,
                    **dict(FA_SHAPES, source_chunk=5))


def test_a_probe_writes_no_ledger(tmp_path):
    _engine, _summary, out, _ = _fa_pass(tmp_path, probe=1)
    assert not os.path.exists(os.path.join(out, me.FA_LEDGER_FILENAME))


def _claim_fa(tmp_path, out, binding, rooms, record=None):
    """The driver's claim, in its order: the ledger FIRST (Codex rB-F2 finding 3), then the
    binding, then -- with a ``record`` name -- the launch segment. ``(ledger, sha)``."""
    from src.localization import meshgrid_segments as ms

    path = admission = None
    if record is not None:
        argv = [token if token != "out" else os.path.basename(out) for token in LR.ARGV]
        path = LR.emit(tmp_path, argv=argv, name=record)
        admission = LR.admit(path, argv=argv)
    sha = None if admission is None else admission["sha256"]
    ledger = me.open_fa_ledger(out, binding["fa_partition"], sha)
    me.write_binding(out, binding, advisory={"batch_rows": 8, "source_chunk": 3},
                     declared_rooms=rooms)
    if admission is not None:
        ms.open_segment(out, path, admission, resume=False)
    return ledger, sha


def _fa_shards(tmp_path):
    """Two record-carrying fa shards of the fixture, published like the real ones."""
    plan, records, items = _fa_aligned(tmp_path)
    binding = _r2_binding(fa_partition=FA_DECLARED)
    dirs = {}
    for name, rooms in (("a", ["A/A_idx_1"]), ("b", ["B/B_idx_2"])):
        out = str(tmp_path / name)
        ledger, sha = _claim_fa(tmp_path, out, binding, rooms, record=f"{name}.json")
        summary = me.run_pass(FaSyntheticEngine(), items, records, plan, out, rooms=rooms,
                              binding_sha256=me.binding_sha256(binding), fa_ledger=ledger,
                              launch_record_sha256=sha, fa_partition=FA_DECLARED, **FA_SHAPES)
        me.write_json(os.path.join(out, "run_summary.json"), summary)
        dirs[name] = out
    pairs, unions = ENG._fixture_totals()
    totals = {"rooms": 2, "queries": len(records), "candidate_query_pairs": pairs,
              "source_rows": unions, "generated_waveforms": pairs * 4}
    return plan, records, dirs, totals


def test_an_fa_merge_re_derives_every_shards_executed_partition_and_keeps_it(tmp_path):
    plan, records, dirs, totals = _fa_shards(tmp_path)
    merged = str(tmp_path / "merged")
    report = me.merge_shards([dirs["a"], dirs["b"]], merged, plan, records, totals=totals)
    block = report["fa_partition_executed"]
    assert block["executed"] == FA_DECLARED and block["violations"] == []
    shards = [me.FaPartitionRecorder.from_counts(*[s["counts"] for s in _ledger(d)["segments"]])
              for d in (dirs["a"], dirs["b"])]
    union = me.FaPartitionRecorder.from_counts(*[r.to_counts() for r in shards])
    assert block["counts_sha256"] == canonical_sha256(union.to_counts())
    assert [entry["counts_sha256"] for entry in block["per_shard"]] == [
        canonical_sha256(r.to_counts()) for r in shards]
    kept = _ledger(merged)
    assert kept["merged"]["counts_sha256"] == block["counts_sha256"]
    assert {segment["shard"] for segment in kept["segments"]} == {dirs["a"], dirs["b"]}


@pytest.mark.parametrize("damage, match", [
    ("ledger_missing", "fa_partition_ledger"),
    ("ledger_counts_edited", "violations"),
    ("summary_disagrees", "run summary"),
    ("summary_missing_block", "run summary"),
])
def test_an_fa_merge_refuses_a_shard_whose_executed_partition_does_not_hold(tmp_path, damage,
                                                                           match):
    plan, records, dirs, totals = _fa_shards(tmp_path)
    shard = dirs["b"]
    ledger_path = os.path.join(shard, me.FA_LEDGER_FILENAME)
    summary_path = os.path.join(shard, "run_summary.json")
    if damage == "ledger_missing":
        os.remove(ledger_path)
    elif damage == "ledger_counts_edited":
        ledger = _ledger(shard)
        entry = next(e for e in ledger["segments"][0]["counts"]
                     if e["branch"] == "context_scored_batch")
        entry["base"] = [entry["configured"] + 1]                  # a batch above its stride
        me.write_json(ledger_path, ledger)
    else:
        summary = json.load(open(summary_path))
        if damage == "summary_disagrees":
            summary["fa_partition_executed"]["ledger"]["counts_sha256"] = "0" * 64
        else:
            summary.pop("fa_partition_executed")
        me.write_json(summary_path, summary)
    merged = str(tmp_path / "merged")
    with pytest.raises(ValueError, match=match):
        me.merge_shards([dirs["a"], dirs["b"]], merged, plan, records, totals=totals)
    assert not os.path.exists(merged)


def test_a_vanilla_merge_refuses_a_shard_carrying_a_frame_average_partition(tmp_path):
    plan, records, dirs, totals, _binding = LR._r2_shards(tmp_path)
    me.write_json(os.path.join(dirs["a"], me.FA_LEDGER_FILENAME), {"segments": []})
    with pytest.raises(ValueError, match="vanilla"):
        me.merge_shards([dirs["a"], dirs["b"]], str(tmp_path / "m"), plan, records,
                        totals=totals)
    os.remove(os.path.join(dirs["a"], me.FA_LEDGER_FILENAME))
    report = me.merge_shards([dirs["a"], dirs["b"]], str(tmp_path / "m2"), plan, records,
                             totals=totals)
    assert report["fa_partition_executed"] is None


# --------------------------------------------------------------------------- #
# exp_22 rB-F2 (Codex code review rB-F1), finding 1: the drift measurement G3D
# carries no tolerance of any kind (plan r2 amendment A6 iii)
# --------------------------------------------------------------------------- #
def test_g3_carries_its_numeric_tolerances_and_g3d_carries_none(tmp_path):
    """A6 (iii): ``tolerance``, ``score_tolerance`` and ``match`` are the registered numbers
    (and a boolean) for the G3 logic gate and null for G3D -- in the gate's report, in its
    finalized evidence and in the JSON a reader re-loads."""
    gate = _g3_registered(tmp_path / "g3")
    measurement = _g3_registered(tmp_path / "g3d", measurement=True)
    for report in (gate, fp.finalize_evidence(gate, device="cpu", environment=CLEAN_ENVIRONMENT)):
        for payload in (report, json.loads(json.dumps(report))):
            assert payload["tolerance"] == 1e-3
            assert payload["score_tolerance"] == me.SCORE_TOLERANCE
            assert isinstance(payload["match"], bool)
    for report in (measurement, fp.finalize_evidence(measurement, device="cpu",
                                                     environment=CLEAN_ENVIRONMENT)):
        for payload in (report, json.loads(json.dumps(report))):
            assert payload["verdict"] == "measurement"
            assert payload["tolerance"] is None and payload["score_tolerance"] is None
            assert payload["match"] is None


# --------------------------------------------------------------------------- #
# Codex rB-F1 finding 2 (part 1): the G3 / G3D evidence carries what its verdict was
# judged on -- the pinned anchor block and, per chunk, the COMPLETE replay-partition
# record and the per-key facts -- so a reader can re-verify it from the JSON alone
# --------------------------------------------------------------------------- #
#: every field ``replay_partition_reasons`` reads, and the rest of the released record
REPLAY_RECORD_FIELDS = {"partition", "frame_avg_fwd_cap", "candidate_micro_batch",
                        "angles_per_chunk", "n_orbit_forwards", "orbit_size", "cap_policy",
                        "shared_angle_count", "cond_method"}


def _registered_gate3(tmp_path, monkeypatch, gate="G3", conditioner=None):
    """A driver-run G3 (fp32 logic gate) or G3D (drift measurement) at the REGISTERED
    256 / 16 / K = 8 over the fixture's registered probe queries -- the real evidence."""
    import localize_meshgrid as driver

    plan, records, loader = _driver_stack(tmp_path)
    anchor = _fixture_anchor(tmp_path, monkeypatch, plan)
    mode = "--fa-production-parity-check" if gate == "G3" else "--fa-production-drift-measurement"
    protocol = _off(ENG._FA_PROTOCOL) if gate == "G3" else ENG._FA_PROTOCOL
    args = _gate_args(tmp_path, mode, "--replay-probe-report", anchor["path"], protocol=protocol)
    assert driver.validate_args(args)
    binding = _registered_binding("BF", **({"cond_autocast": "off"} if gate == "G3" else {}))
    engine = FaSyntheticEngine()
    engine.conditioner = conditioner or engine.conditioner
    code = driver._run_fa_gate(args, engine, plan, records, loader, gate, binding)
    name = "fa_production_parity.json" if gate == "G3" else "fa_production_drift.json"
    return code, json.load(open(tmp_path / "gates" / name)), anchor


@pytest.mark.parametrize("gate", ["G3", "G3D"])
def test_gate3_and_its_measurement_publish_the_pinned_anchor_block(tmp_path, monkeypatch,
                                                                   clean_env, gate):
    """The driver used to verify the pinned report and then discard it (localize_meshgrid.py
    ~890): the evidence could not show that its registered set is the pinned 8c5fd5e8...'s."""
    code, evidence, anchor = _registered_gate3(tmp_path, monkeypatch, gate)
    assert code == 0
    published = evidence["inputs"]["probe_report"]
    assert published == {key: anchor[key] for key in fp.ANCHOR_EVIDENCE}
    assert published["sha256"] == published["pinned_sha256"] == fp.COMMITTED_PROBE_REPORT_SHA256
    assert published["query_ids_sha256"] == canonical_sha256(published["query_ids"])
    assert sorted(published["query_ids"]) == sorted(evidence["probe_queries"]["registered"])
    assert evidence["inputs"]["anchor_key_set_equal"] is True


def test_gate3_without_its_pinned_anchor_does_not_pass(tmp_path):
    cases = _g3_cases(tmp_path)
    ids = _ids(cases)
    kwargs = dict(G3_KW, declared_partition=FA_DECLARED, expected_queries=ids)
    assert fp.production_parity(FaSyntheticEngine(), cases, anchor=_anchor_for(ids),
                                **kwargs)["verdict"] == "PASS"
    unanchored = fp.production_parity(FaSyntheticEngine(), cases, **kwargs)
    assert unanchored["verdict"] == "FAIL"
    _only(unanchored["reasons"], "anchor")
    for anchor, needle in ((dict(_anchor_for(ids), sha256="0" * 64), "pinned"),
                           (_anchor_for(ids + ["9|ir/Z/Z_idx_9/S001_R001_hybrid_IR.wav"]),
                            "key set")):
        report = fp.production_parity(FaSyntheticEngine(), cases, anchor=anchor, **kwargs)
        assert report["verdict"] == "FAIL" and report["inputs"]["probe_report"]["path"]
        _only(report["reasons"], "anchor")
        assert any(needle in reason for reason in report["reasons"]), needle


@pytest.mark.parametrize("measurement", [False, True])
def test_each_gate3_chunk_serializes_its_complete_replay_record_and_per_key_facts(tmp_path,
                                                                                  measurement):
    """Codex rB-F1 finding 2: a chunk published only ``record['partition']``, not the cap,
    micro-batch, angle and forward counts ``replay_partition_reasons`` judged."""
    report = json.loads(json.dumps(_g3_registered(tmp_path, measurement=measurement)))
    for entry in report["queries"]:
        for chunk in entry["chunks"]:
            record = chunk["replay_partition"]
            assert REPLAY_RECORD_FIELDS <= set(record), sorted(record)
            assert fp.replay_partition_reasons(record, chunk["n"], len(C4)) == []
            assert record["partition"] == [chunk["n"]] * 3
            assert set(chunk["conditioning"]) == set(FA_IDS)
            for facts in chunk["conditioning"].values():
                assert {"shape", "dtype", "digest", "finite", "max_abs_diff",
                        "bitwise"} <= set(facts["tensor"]) and "digest" in facts["mask"]
            assert chunk["dit_inputs"] and all("max_abs_diff" in facts
                                               for facts in chunk["dit_inputs"].values())
    executed = report["executed_partition"]
    again = me.FaPartitionRecorder.from_counts(executed["counts"]).report()
    assert {key: again[key] for key in ("executed", "violations", "branches")} == {
        key: executed[key] for key in ("executed", "violations", "branches")}


# --------------------------------------------------------------------------- #
# Codex rB-F1 finding 2 (part 2): ``reverify_gate_evidence`` recomputes every PASS
# clause of a gate FROM ITS SERIALIZED JSON ALONE and names what does not hold
# --------------------------------------------------------------------------- #
import copy                                                              # noqa: E402


def _tamper(evidence, path, value=None, delete=False, scale=None):
    """A deep copy of ``evidence`` with the field at ``path`` (keys / list indices) replaced,
    scaled or deleted -- the reader's JSON, edited."""
    edited = copy.deepcopy(json.loads(json.dumps(evidence)))
    target = edited
    for step in path[:-1]:
        target = target[step]
    if delete:
        del target[path[-1]]
    elif scale is not None:
        target[path[-1]] = target[path[-1]] * scale
    else:
        target[path[-1]] = value
    return edited


@pytest.mark.parametrize("gate", ["G3", "G3D"])
def test_a_real_gate3_evidence_re_verifies_from_its_json_alone(tmp_path, monkeypatch, clean_env,
                                                               gate):
    code, evidence, _anchor = _registered_gate3(tmp_path, monkeypatch, gate)
    assert code == 0 and evidence["verdict"] == ("PASS" if gate == "G3" else "measurement")
    assert fp.reverify_gate_evidence(evidence) == []
    assert fp.reverify_gate_evidence(json.loads(json.dumps(evidence))) == []


G3_TAMPERS = [
    # (name, path, edit, the reason's needle)
    ("per-key delta above tolerance", ["queries", 0, "chunks", 0, "conditioning", "source",
                                       "tensor", "max_abs_diff"], {"value": 2e-3},
     "source.tensor: max |diff| 0.002 > 0.001"),
    ("per-key summary edited", ["per_key", "source", "max_abs_diff"], {"value": 2e-3}, "per_key"),
    ("missing chunk", ["queries", 1, "chunks", 1], {"delete": True}, "do not tile"),
    ("wrong replay-partition field", ["queries", 0, "chunks", 0, "replay_partition",
                                      "frame_avg_fwd_cap"], {"scale": 2}, "replay orbit partition"),
    ("anchor digest mismatch", ["inputs", "probe_report", "sha256"], {"value": "0" * 64},
     "pinned committed probe report"),
    ("anchor ids not their digest", ["inputs", "probe_report", "query_ids_sha256"],
     {"value": "0" * 64}, "digest or count"),
    ("missing probe query", ["queries", 1], {"delete": True},
     "registered probe queries were not consumed"),
    ("provenance clause flipped", ["provenance", "tracked_clean"], {"value": False},
     "provenance: the tracked tree"),
    ("verdict clause flipped", ["verdict_clauses", "provenance"], {"value": False},
     "verdict_clauses"),
    ("score drift", ["queries", 0, "score_stability", "8", "max_abs_delta"], {"value": 0.5},
     "score |diff|"),
    ("argmax moved", ["queries", 0, "score_stability", "1", "argmax_equal"], {"value": False},
     "argmax moved"),
    ("executed counts edited", ["executed_partition", "counts", 0, "base"], {"value": [7]},
     "executed partition"),
    ("declared partition edited", ["declared_partition", "source_cache",
                                   "candidate_micro_batch"], {"value": 8}, "declared"),
    ("binding moved", ["inputs", "binding", "tau"], {"value": 0.05}, "identity"),
    ("autocast edited", ["inputs", "cond_autocast_enabled"], {"value": True}, "autocast"),
    ("tolerance raised", ["tolerance"], {"value": 1.0}, "tolerance"),
    ("coverage edited", ["coverage", "source", "tail"], {"value": 99}, "coverage"),
    ("verdict without its clauses", ["reasons"], {"value": ["anything"]}, "verdict"),
]


@pytest.mark.parametrize("name, path, edit, needle", G3_TAMPERS, ids=[t[0] for t in G3_TAMPERS])
def test_a_tampered_gate3_evidence_names_what_does_not_re_verify(tmp_path, monkeypatch, clean_env,
                                                                 name, path, edit, needle):
    _code, evidence, _anchor = _registered_gate3(tmp_path, monkeypatch, "G3")
    reasons = fp.reverify_gate_evidence(_tamper(evidence, path, **edit))
    assert any(needle in reason for reason in reasons), (name, reasons)


def test_a_drift_measurement_re_verifies_without_judging_its_deltas(tmp_path, monkeypatch,
                                                                   clean_env):
    """G3D publishes deltas far above 1e-3 and still re-verifies: there is no tolerance (A6
    iii). Its structure is re-verified, and a threshold smuggled into it is named."""
    _code, evidence, _anchor = _registered_gate3(tmp_path, monkeypatch, "G3D", conditioner=Noisy())
    assert max(entry["max_abs_diff"] for entry in evidence["per_key"].values()) > 1e-3
    assert fp.reverify_gate_evidence(evidence) == []
    for path, edit, needle in ((["score_tolerance"], {"value": 1e-3}, "score_tolerance"),
                               (["match"], {"value": True}, "match"),
                               (["complete"], {"value": False}, "complete"),
                               (["queries", 1, "chunks", 1], {"delete": True}, "do not tile"),
                               (["inputs", "probe_report", "pinned_sha256"], {"value": "1" * 64},
                                "pinned committed probe report")):
        reasons = fp.reverify_gate_evidence(_tamper(evidence, path, **edit))
        assert any(needle in reason for reason in reasons), (path, reasons)


def test_an_unknown_or_unregistered_evidence_does_not_re_verify():
    assert fp.reverify_gate_evidence({}) and fp.reverify_gate_evidence({"gate": "G9"})


# --------------------------------------------------------------------------- #
# ... and every other gate whose evidence allows it: G1, G2, GV and G0 (G0 now
# publishes the frozen digests it compared and the published binding it replayed)
# --------------------------------------------------------------------------- #
def _real_gate_evidence(tmp_path, monkeypatch, gate):
    """A driver-run PASS evidence of ``gate`` on the fixture (``clean_env`` active)."""
    import localize_meshgrid as driver
    from src.localization import meshgrid_offgrid_probe as mop

    if gate in ("G1", "G2"):
        plan, records, loader = _driver_stack(tmp_path)
        if gate == "G1":
            args = _gate_args(tmp_path, "--fa-split-parity-check", "--parity-candidates", "48",
                              protocol=_off(ENG._FA_PROTOCOL))
            code = driver._run_fa_gate(args, FaSyntheticEngine(), plan, records, loader, "G1",
                                       _registered_binding("BF", cond_autocast="off"))
        else:
            args = _gate_args(tmp_path, "--cache-parity-check", "--parity-candidates", "5")
            code = driver._run_cache_parity(args, FaSyntheticEngine(), plan, records, loader,
                                            binding=_registered_binding("BF"))
    elif gate == "GV":
        plan, records, items = ENG._aligned(tmp_path)
        args = driver.parse_args(["--ckpt-path", "x.ckpt", "--cache-parity-check", "--expect-arm",
                                  "YAW", "--out-dir", str(tmp_path / "gates")]
                                 + ENG._VANILLA_PROTOCOL)
        code = driver._run_cache_parity(args, ENG.SyntheticEngine(), plan, records,
                                        [([obs], [md]) for obs, md in items],
                                        binding=_registered_binding("YAW"))
    else:
        out, binding, frozen, _cases = _published_vanilla_run(tmp_path, _registered_binding("P1"))
        plan, records, items = ENG._aligned(tmp_path / "stream")
        anchor = _fake_anchor(tmp_path, monkeypatch, frozen)
        monkeypatch.setattr(mop, "compute_observation_bank_digest",
                            lambda *a, **k: {"observation_bank_sha256":
                                             fp.FROZEN_OBSERVATION_BANK_SHA256})
        args = driver.parse_args(["--ckpt-path", "x.ckpt", "--replay-check", "--replay-against",
                                  out, "--replay-probe-report", anchor["path"], "--out-dir",
                                  str(tmp_path / "gates")] + ENG._VANILLA_PROTOCOL)
        code = driver._run_replay_check(args, ENG.SyntheticEngine(), plan, records,
                                        [(obs, [md]) for obs, md in items], binding=binding)
    evidence = json.load(open(tmp_path / "gates" / f"{fp.GATE_NAMES[gate]}.json"))
    assert code == 0 and evidence["verdict"] == "PASS", evidence["reasons"]
    return evidence


@pytest.mark.parametrize("gate", ["G1", "G2", "GV", "G0"])
def test_every_other_real_gate_evidence_re_verifies_from_its_json_alone(tmp_path, monkeypatch,
                                                                        clean_env, gate):
    assert fp.reverify_gate_evidence(_real_gate_evidence(tmp_path, monkeypatch, gate)) == []


OTHER_TAMPERS = [
    ("G1", ["chunks", 0, "conditioning", "source_vit", "tensor", "max_abs_diff"],
     {"value": 1e-9}, "source_vit.tensor: max |diff| 1e-09 > 0.0"),
    ("G1", ["chunks", 1, "released_partition", "frame_avg_fwd_cap"], {"scale": 2},
     "replay orbit partition"),
    ("G1", ["chunks", 1], {"delete": True}, "do not tile"),
    ("G1", ["chunks", 0, "dit_inputs", "cross_attn_cond", "dtype", 1],
     {"value": "torch.float16"}, "dit_inputs.cross_attn_cond: dtype"),
    ("G1", ["inputs", "identity", "arm"], {"value": "YAW"}, "identity"),
    ("G2", ["memoization", "conditioning", "context_poses", "mask", "max_abs_diff"],
     {"value": 1e-9}, "memoization: context_poses.mask: max |diff|"),
    ("G2", ["counter_tests", "unaveraged_vit", "keys", "source_vit"], {"value": 0.0},
     "did not bite"),
    ("G2", ["counter_tests", "swapped_candidates"], {"delete": True}, "counter-test"),
    ("G2", ["memoization", "match"], {"value": False}, "memoization"),
    ("GV", ["vanilla_cache_parity", "memoization", "keys", "source", "equal"], {"value": False},
     "memoization"),
    ("GV", ["vanilla_cache_parity", "counter_test", "max_abs_diff"], {"value": 0.0},
     "counter-test"),
    ("GV", ["inputs", "identity", "arm"], {"value": "P1"}, "identity"),
    ("GV", ["inputs", "binding", "ckpt_sha256"], {"value": "5" * 64}, "identity"),
    ("G0", ["queries", 0, "matched", "float16_bit_exact"], {"value": False}, "float16"),
    ("G0", ["queries", 1, "matched", "aggregate", "max_abs_delta"], {"value": 1e-7},
     "aggregate"),
    ("G0", ["queries", 0, "frozen_digests", "sims_sha256"], {"value": "0" * 64}, "frozen"),
    ("G0", ["inputs", "observation_bank_sha256"], {"value": "0" * 64}, "observation bank"),
    ("G0", ["inputs", "published_binding", "tau"], {"value": 0.05}, "published"),
    ("G0", ["inputs", "probe_report", "pinned_sha256"], {"value": "0" * 64},
     "pinned committed probe report"),
    ("G0", ["queries", 1], {"delete": True}, "registered probe queries were not consumed"),
    ("G0", ["provenance", "hostname"], {"value": None}, "provenance: the hostname"),
]


@pytest.mark.parametrize("gate, path, edit, needle", OTHER_TAMPERS,
                         ids=[f"{t[0]}-{'.'.join(map(str, t[1]))}" for t in OTHER_TAMPERS])
def test_a_tampered_gate_evidence_names_what_does_not_re_verify(tmp_path, monkeypatch, clean_env,
                                                                gate, path, edit, needle):
    evidence = _real_gate_evidence(tmp_path, monkeypatch, gate)
    reasons = fp.reverify_gate_evidence(_tamper(evidence, path, **edit))
    assert any(needle in reason for reason in reasons), reasons


def test_gate0_publishes_what_its_frozen_and_protocol_clauses_compared(tmp_path, monkeypatch,
                                                                       clean_env):
    evidence = _real_gate_evidence(tmp_path, monkeypatch, "G0")
    for entry in evidence["queries"]:
        assert entry["frozen_digests"] == entry["published_digests"]
    published = evidence["inputs"]["published_binding"]
    assert me.binding_sha256(published) == evidence["inputs"]["published_binding_sha256"]


def test_every_finalized_evidence_tells_its_reader_how_to_re_verify_it(tmp_path):
    evidence = fp.finalize_evidence(_g1_report(tmp_path), device="cpu",
                                    environment=CLEAN_ENVIRONMENT, inputs=_g1_inputs())
    assert evidence["reverify_note"] == fp.REVERIFY_NOTE
    assert "reverify_gate_evidence" in fp.REVERIFY_NOTE


# --------------------------------------------------------------------------- #
# Codex rB-F1 finding 3: ONE validating ledger reader, used by every reopen (resume,
# the driver's preflight) and by the merge -- the ledger is tamper-evident across both
# --------------------------------------------------------------------------- #
def _fa_files(out):
    ledger_path = os.path.join(out, me.FA_LEDGER_FILENAME)
    summary_path = os.path.join(out, "run_summary.json")
    return ledger_path, summary_path


def _two_segment_run(out, stack, rooms=None):
    """A finished fa run ``out`` plus a final no-op resume: two ledger segments and the
    run summary the driver writes after each invocation."""
    plan, records, items = stack
    done, _rejected = me.completed_queries(out)
    bound = os.path.join(out, me.BINDING_FILENAME)
    summary = me.run_pass(FaSyntheticEngine(), items, records, plan, out, done=done, rooms=rooms,
                          binding_sha256=(json.load(open(bound))["binding_sha256"]
                                          if os.path.isfile(bound) else None),
                          fa_partition=FA_DECLARED, **FA_SHAPES)
    assert summary["n_scored"] == 0                          # a true no-op resume
    me.write_json(_fa_files(out)[1], summary)
    assert len(_ledger(out)["segments"]) == 2
    return done


def _resummed(ledger):
    """Recompute every stored digest of ``ledger`` from its (edited) counts, so that only
    the check under test can catch the edit."""
    for segment in ledger["segments"]:
        segment["counts_sha256"] = me.counts_sha256(
            me.FaPartitionRecorder.from_counts(segment["counts"]))
    merged = me.FaPartitionRecorder.from_counts(*[s["counts"] for s in ledger["segments"]])
    ledger["merged"] = {"counts_sha256": me.counts_sha256(merged),
                        "executed": merged.executed(), "violations": merged.violations()}
    return ledger


def _bump(ledger):
    ledger["segments"][0]["counts"][0]["count"] += 1


def _negative(ledger):
    ledger["segments"][0]["counts"][0]["count"] = -1
    _resummed(ledger)


LEDGER_CORRUPTIONS = {
    # name: (edit the ledger, edit the run summary, the refusal's needle)
    "tampered count": (_bump, None, "segment 0's counts do not hash to its counts_sha256"),
    "tampered segment digest": (lambda l: l["segments"][1].update(counts_sha256="0" * 64), None,
                                "segment 1's counts do not hash"),
    "tampered merged block": (lambda l: l["merged"]["executed"].update(orbit_size=5), None,
                              "merged block"),
    "missing segment index": (lambda l: l["segments"].pop(0), None, "segment indices"),
    "duplicated segment index": (lambda l: l["segments"][1].update(segment_index=0), None,
                                 "segment indices"),
    "negative count": (_negative, None, "not a non-negative integer"),
    "summary digest mismatch": (None, lambda s: s["fa_partition_executed"]["ledger"].update(
        counts_sha256="0" * 64), "run summary"),
    "summary block mismatch": (None, lambda s: s["fa_partition_executed"]["executed"].update(
        orbit_size=5), "run summary"),
    "summary names a missing ledger": ("delete", None, "not on disk"),
    "garbled ledger": ("garble", None, "is not valid JSON"),
    "unknown branch": (lambda l: l["segments"][0]["counts"][0].update(branch="nowhere"), None,
                       "segment 0: a ledger entry names branch 'nowhere'"),
}


def _corrupt(out, name):
    edit_ledger, edit_summary, needle = LEDGER_CORRUPTIONS[name]
    ledger_path, summary_path = _fa_files(out)
    if edit_ledger == "delete":
        os.remove(ledger_path)
    elif edit_ledger == "garble":
        open(ledger_path, "w").write('{"declared": ')
    elif edit_ledger is not None:
        ledger = _ledger(out)
        edit_ledger(ledger)
        me.write_json(ledger_path, ledger)
    if edit_summary is not None:
        summary = json.load(open(summary_path))
        edit_summary(summary)
        me.write_json(summary_path, summary)
    return needle


def _snapshot(directory):
    """Every file under ``directory`` and the sha256 of its bytes."""
    import hashlib

    found = {}
    for root, _dirs, files in os.walk(directory):
        for name in files:
            path = os.path.join(root, name)
            found[os.path.relpath(path, directory)] = hashlib.sha256(
                open(path, "rb").read()).hexdigest()
    return found


@pytest.mark.parametrize("name", sorted(LEDGER_CORRUPTIONS))
def test_a_corrupted_ledger_is_refused_at_reopen_before_a_no_op_resume_writes_anything(tmp_path,
                                                                                      name):
    _engine, summary, out, stack = _fa_pass(tmp_path)
    me.write_json(_fa_files(out)[1], summary)
    done = _two_segment_run(out, stack)
    assert me.read_fa_ledger(out, FA_DECLARED)["n_segments"] == 2       # the honest run reads
    needle = _corrupt(out, name)
    before = _snapshot(out)
    engine = FaSyntheticEngine()
    plan, records, items = stack
    with pytest.raises(ValueError, match=re_escape(needle)):
        me.run_pass(engine, items, records, plan, out, done=done, fa_partition=FA_DECLARED,
                    **FA_SHAPES)
    assert _snapshot(out) == before and engine.conditioner.calls == []
    with pytest.raises(ValueError, match=re_escape(needle)):
        me.read_fa_ledger(out, FA_DECLARED)


def re_escape(text):
    import re

    return re.escape(text)


@pytest.mark.parametrize("name", sorted(LEDGER_CORRUPTIONS) + ["stale summary"])
def test_a_corrupted_shard_ledger_is_refused_at_the_merge_with_nothing_written(tmp_path, name):
    plan, records, dirs, totals = _fa_shards(tmp_path)
    shard = dirs["b"]
    _two_segment_run(shard, _fa_aligned(tmp_path / "stack"), rooms=["B/B_idx_2"])
    if name == "stale summary":            # a reopen admits it (a crash after the segment);
        summary_path = _fa_files(shard)[1]  # the merge needs the summary of EVERY segment
        summary = json.load(open(summary_path))
        prefix = me.FaPartitionRecorder.from_counts(_ledger(shard)["segments"][0]["counts"])
        summary["fa_partition_executed"] = me.fa_executed_block(FA_DECLARED, prefix, 1)
        me.write_json(summary_path, summary)
        assert me.read_fa_ledger(shard, FA_DECLARED)["n_segments"] == 2
        needle = "run summary"
    else:
        needle = _corrupt(shard, name)
    merged = str(tmp_path / "merged")
    with pytest.raises(ValueError, match=re_escape(needle)):
        me.merge_shards([dirs["a"], dirs["b"]], merged, plan, records, totals=totals)
    assert not os.path.exists(merged)


def test_the_driver_refuses_a_corrupted_ledger_before_it_writes_or_touches_a_device(tmp_path):
    """The driver's preflight reads the out-dir through the same reader, BEFORE the engine is
    built and before claim_out_dir writes the binding or opens a launch segment."""
    # the fixture run at the REGISTERED shapes, so its declaration is the driver's binding's
    _engine, summary, out, stack = _fa_pass(tmp_path / "run", batch_rows=256, num_samples=8,
                                            prefixes=(1, 4, 8), source_chunk=16,
                                            fa_partition=DECLARED)
    me.write_json(_fa_files(out)[1], summary)
    args = _preflight_args(tmp_path / "pre", "--resume", "--launch-record", "x.json",
                           protocol=ENG._FA_PROTOCOL)
    args.out_dir = out
    assert _preflight(args, "BF")[0]["fa_partition"] == DECLARED       # the honest run reopens
    _corrupt(out, "tampered count")
    before = _snapshot(out)
    with pytest.raises(SystemExit, match="counts do not hash to its counts_sha256"):
        _preflight(args, "BF")
    assert _snapshot(out) == before


# --------------------------------------------------------------------------- #
# Codex rB-F1 NIT 5; plan r2 amendment A6 (iv): a shard whose executed-partition check
# failed is NON-ADMISSIBLE -- it says so on disk, and the merge, the per-arm report and
# every reopen refuse it by name (restart in a fresh out-dir, never resume)
# --------------------------------------------------------------------------- #
NEVER_FILLS = me.declared_fa_partition(list(C4), batch_rows=800, num_samples=4, source_chunk=3)


def _failed_pass(tmp_path, declared=NEVER_FILLS, **overrides):
    plan, records, items = _fa_aligned(tmp_path)
    out = str(tmp_path / "run")
    kwargs = dict(FA_SHAPES, fa_partition=declared, **overrides)
    with pytest.raises(ValueError, match="not the declared one"):
        me.run_pass(FaSyntheticEngine(), items, records, plan, out, **kwargs)
    return out, (plan, records, items), kwargs


def test_a_pass_whose_final_partition_check_fails_marks_itself_non_admissible(tmp_path):
    """Micro-batch 200 is never filled: every row passes the prefix rule, the final exact
    check fails -- the rows stay on disk as INCOMPLETE artifacts, and the shard says so."""
    out, (plan, records, items), kwargs = _failed_pass(tmp_path, batch_rows=800)
    marker = json.load(open(os.path.join(out, me.FA_FAILURE_FILENAME)))
    assert marker["check"] == "final" and marker["note"] == me.FA_FAILURE_NOTE
    assert "A6 (iv)" in marker["note"] and "never" in marker["note"]
    assert marker["declared"] == NEVER_FILLS
    assert marker["differences"] == ["context_scored_batch.candidate_micro_batch"]
    assert marker["executed"]["context_scored_batch"]["candidate_micro_batch"] < 200
    assert marker["violations"] == [] and "context_scored_batch" in marker["error"]
    assert marker["ledger"] == {"file": me.FA_LEDGER_FILENAME, "n_segments": 1}
    # the judged counts are the run's summed ledger, and they re-derive the executed block
    assert marker["counts_sha256"] == _ledger(out)["merged"]["counts_sha256"]
    assert me.FaPartitionRecorder.from_counts(marker["counts"]).executed() == marker["executed"]
    assert me.read_rows(out) and not os.path.exists(os.path.join(out, "run_summary.json"))
    # never resumed: a reopen is refused by name before anything is written
    before = _snapshot(out)
    done, _rejected = me.completed_queries(out)
    with pytest.raises(ValueError, match="NON-ADMISSIBLE"):
        me.run_pass(FaSyntheticEngine(), items, records, plan, out, done=done, **kwargs)
    for check in (lambda: me.read_fa_ledger(out, NEVER_FILLS),
                  lambda: me.assert_admissible(out)):
        with pytest.raises(ValueError, match=r"NON-ADMISSIBLE.*A6 \(iv\)"):
            check()
    assert _snapshot(out) == before


def test_a_pass_refused_before_a_row_marks_itself_non_admissible_too(tmp_path):
    """The before-row check refuses a partition the pass does not execute (source chunk 16
    declared, 3 run): the same non-admissible state, named by the check that caught it."""
    declared = me.declared_fa_partition(list(C4), batch_rows=8, num_samples=4, source_chunk=16)
    out, _stack, _kwargs = _failed_pass(tmp_path, declared=declared)
    marker = json.load(open(os.path.join(out, me.FA_FAILURE_FILENAME)))
    assert marker["check"] == "before_row"
    assert "source_cache.candidate_micro_batch" in marker["differences"]


def test_a_probe_never_writes_a_failure_marker(tmp_path):
    _engine, summary, out, _stack = _fa_pass(tmp_path, batch_rows=800, fa_partition=NEVER_FILLS,
                                             probe=1)
    assert summary["fa_partition_executed"]["branches"]["context_scored_batch"]["saturated"] is False
    assert not os.path.exists(os.path.join(out, me.FA_FAILURE_FILENAME))


@pytest.mark.parametrize("when", ["a later invocation", "the only invocation"])
def test_the_merge_refuses_a_non_admissible_shard_by_name_with_nothing_written(tmp_path,
                                                                             monkeypatch, when):
    """The pass itself marks the shard when its final exact check fails -- after an earlier
    invocation finished (a stale run summary is on disk) or in its only invocation (no run
    summary at all) -- and the merge names it in both cases, before anything else."""
    plan, records, dirs, totals = _fa_shards(tmp_path)
    real = me.assert_fa_partition_executed

    def fails_finally(declared, recorder, final=True):
        if final:
            raise ValueError("the executed frame-average partition is not the declared one "
                             "(simulated)")
        return real(declared, recorder, final=False)

    monkeypatch.setattr(me, "assert_fa_partition_executed", fails_finally)
    stack = _fa_aligned(tmp_path / "stack")
    shard = dirs["b"] if when == "a later invocation" else str(tmp_path / "b_failed")
    with pytest.raises(ValueError, match="not the declared one"):
        if when == "a later invocation":
            _two_segment_run(shard, stack, rooms=["B/B_idx_2"])
        else:
            binding = _r2_binding(fa_partition=FA_DECLARED)
            ledger, _sha = _claim_fa(tmp_path, shard, binding, ["B/B_idx_2"])
            me.run_pass(FaSyntheticEngine(), stack[2], stack[1], stack[0], shard,
                        rooms=["B/B_idx_2"], binding_sha256=me.binding_sha256(binding),
                        fa_ledger=ledger, fa_partition=FA_DECLARED, **FA_SHAPES)
    monkeypatch.setattr(me, "assert_fa_partition_executed", real)
    assert os.path.exists(os.path.join(shard, me.FA_FAILURE_FILENAME))
    assert os.path.exists(_fa_files(shard)[1]) is (when == "a later invocation")
    merged = str(tmp_path / "merged")
    with pytest.raises(ValueError, match=r"NON-ADMISSIBLE.*A6 \(iv\)"):
        me.merge_shards([dirs["a"], shard], merged, plan, records, totals=totals)
    assert not os.path.exists(merged)


@pytest.mark.parametrize("merged", [False, True])
def test_the_per_arm_report_refuses_a_non_admissible_run_by_name(tmp_path, merged):
    import test_loc_meshgrid_report as RPT

    fixture = RPT.build_fixture_run(tmp_path, merged=merged, arm="BF",
                                    launch=RPT.fixture_launch(tmp_path, "BF") if merged else None)
    assert RPT.evaluate_fixture(fixture)["registered_protocol"]["arm"] == "BF"   # admissible
    declared = fixture["binding"]["fa_partition"]
    me.write_partition_failure(fixture["run_dir"], declared, me.FaPartitionRecorder(),
                               check="final", error="simulated")
    with pytest.raises(ValueError, match=r"NON-ADMISSIBLE.*A6 \(iv\)"):
        RPT.evaluate_fixture(fixture)


# --------------------------------------------------------------------------- #
# exp_22 rB-F3 (Codex code review rB-F2), finding 2: the pinned anchor's CONTENT -- its
# ordered probe query ids and its frozen grid-digest map -- is registered beside the
# digest of its bytes, and every clause that reads an anchor authenticates it
# --------------------------------------------------------------------------- #
def _committed_content():
    """The committed report's ordered ids and frozen map, read WITHOUT the module's loader."""
    records = json.load(open(PROBE_REPORT))["records"]
    ids = [str(record["query_id"]) for record in records]
    frozen = {str(record["query_id"]): {"row_sha256": record["grid_row_sha256"],
                                        "sims_sha256": record["grid_sims_sha256"]}
              for record in records}
    return ids, frozen


def test_the_registered_anchor_content_is_the_committed_reports():
    """The two content constants are recomputed here from the committed file's bytes with
    the module's canonical digest, so they cannot drift from the pinned report."""
    ids, frozen = _committed_content()
    assert len(ids) == 16
    assert fp.COMMITTED_PROBE_QUERY_IDS_SHA256 == canonical_sha256(ids) == (
        "4f5743a8c7dcead5486281560a60e709ff93e44e315acf16722962790ab82382")
    assert fp.COMMITTED_PROBE_FROZEN_DIGESTS_SHA256 == canonical_sha256(frozen) == (
        "9ecef47d94165e375c87c98597815a8b0c1ef3b8e83fe75112519ab20f31011c")
    anchor = fp.load_probe_anchor(PROBE_REPORT)
    assert anchor["query_ids"] == ids and anchor["digests"] == frozen
    assert anchor["query_ids_sha256"] == fp.COMMITTED_PROBE_QUERY_IDS_SHA256
    assert anchor["frozen_digests_sha256"] == fp.COMMITTED_PROBE_FROZEN_DIGESTS_SHA256
    assert "frozen_digests_sha256" in fp.ANCHOR_EVIDENCE
    assert fp.anchor_reasons({key: anchor[key] for key in fp.ANCHOR_EVIDENCE}, True) == []


def test_an_anchor_claiming_the_pinned_digest_over_other_content_is_refused():
    """Codex rB-F2's probe: an arbitrary query id, its digest recomputed, the pinned sha
    strings retained -- ``anchor_reasons(..., True)`` returned [] at rB-F2."""
    anchor = fp.load_probe_anchor(PROBE_REPORT)
    block = {key: anchor[key] for key in fp.ANCHOR_EVIDENCE}
    for ids in (["0|forged/Q/Q_idx_0/S001_R001_hybrid_IR.wav"],
                anchor["query_ids"][:-1] + ["0|forged/Q/Q_idx_0/S001_R001_hybrid_IR.wav"],
                list(reversed(anchor["query_ids"]))):
        forged = dict(block, query_ids=ids, query_ids_sha256=canonical_sha256(ids),
                      n_queries=len(ids))
        reasons = fp.anchor_reasons(forged, True)
        assert any("query ids are not the registered content" in r for r in reasons), reasons
    moved = dict(block, frozen_digests_sha256="0" * 64)
    assert any("frozen grid-digest map is not the registered content" in reason
               for reason in fp.anchor_reasons(moved, True))
    assert fp.anchor_reasons(dict(block, frozen_digests_sha256=None), True)


def test_a_report_with_the_pinned_bytes_digest_but_other_content_is_refused_at_load(tmp_path,
                                                                                  monkeypatch):
    """The bytes check alone vouches for the content only while the constants agree with the
    file; the loader holds the parsed content to the registered digests too."""
    import hashlib

    path = tmp_path / "other.json"
    path.write_text(json.dumps({"records": [{"query_id": "0|x", "grid_row_sha256": "1" * 64,
                                             "grid_sims_sha256": "2" * 64}]}))
    monkeypatch.setattr(fp, "COMMITTED_PROBE_REPORT_SHA256",
                        hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="registered content"):
        fp.load_probe_anchor(str(path))


def test_gate0_fails_on_a_frozen_map_that_is_not_the_registered_content(tmp_path, monkeypatch):
    out, binding, frozen, cases = _published_vanilla_run(tmp_path)
    anchor = _fake_anchor(tmp_path, monkeypatch, frozen)
    monkeypatch.setattr(fp, "COMMITTED_PROBE_FROZEN_DIGESTS_SHA256", "0" * 64)
    report = fp.vanilla_replay_regression(
        ENG.SyntheticEngine(), cases, out, anchor=anchor, observation_bank_sha256="o" * 64,
        expected_observation_bank_sha256="o" * 64, run_binding=binding,
        expected_queries=_ids(cases), **G0_KW)
    assert report["verdict"] == "FAIL"
    assert any("frozen grid-digest map" in reason for reason in report["reasons"])


def test_a_coordinated_swap_of_g0s_frozen_and_published_maps_does_not_re_verify(
        tmp_path, monkeypatch, clean_env):
    """Codex rB-F2: G0's re-verification compared frozen == published only, so replacing both
    maps together re-verified. The frozen map must hash to the registered content."""
    evidence = _real_gate_evidence(tmp_path, monkeypatch, "G0")
    assert fp.reverify_gate_evidence(evidence) == []
    swapped = copy.deepcopy(evidence)
    for entry in swapped["queries"]:
        entry["frozen_digests"] = {"row_sha256": "3" * 64, "sims_sha256": "4" * 64}
        entry["published_digests"] = dict(entry["frozen_digests"])
    reasons = fp.reverify_gate_evidence(swapped)
    assert any("frozen grid-digest map" in reason for reason in reasons), reasons
    # the anchor block's own claim is held to the registration as well
    claimed = _tamper(evidence, ["inputs", "probe_report", "frozen_digests_sha256"], "0" * 64)
    assert any("frozen grid-digest map is not the registered content" in reason
               for reason in fp.reverify_gate_evidence(claimed))


# --------------------------------------------------------------------------- #
# exp_22 rB-F3 (Codex code review rB-F2), finding 4 (part 1): ``reverify_gate_evidence``
# always returns named reasons, and its autocast / G3D / DiT-input / executed-count
# clauses are complete
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("evidence", [
    {"gate": "G3", "verdict_clauses": "malformed"}, {"gate": "G3D", "verdict_clauses": 7},
    {"gate": "G1", "chunks": 7, "inputs": "x"}, {"gate": ["G3"]}, {"gate": {"G0": 1}},
    {"gate": "G0", "queries": "x", "inputs": []}, {"gate": "GV", "provenance": "x"},
    "not a dict", None, []])
def test_a_malformed_evidence_is_named_never_raised(evidence):
    """Codex rB-F2: ``_verdict_reasons`` ran outside the MALFORMED guard, so
    ``{"gate": "G3", "verdict_clauses": "malformed"}`` raised AttributeError."""
    reasons = fp.reverify_gate_evidence(evidence)
    assert reasons and all(isinstance(reason, str) for reason in reasons)


def test_the_registered_autocast_resolution_is_torchs():
    """``default`` is torch's per-device autocast default (float16 on CUDA, bfloat16 on CPU),
    ``off`` the float32 parameters -- read from torch and from the engine's own context."""
    assert fp.RESOLVED_AUTOCAST[("default", "cuda")] == (
        True, str(torch.get_autocast_dtype("cuda")).replace("torch.", "")) == (True, "float16")
    for mode in ("default", "off"):
        state = me.cond_autocast_state("cpu", mode)
        assert fp.RESOLVED_AUTOCAST[(mode, "cpu")] == (state["cond_autocast_enabled"],
                                                        state["cond_autocast_dtype"])
    assert fp.RESOLVED_AUTOCAST[("off", "cuda")] == (False, "float32")
    for token, enabled in fp.GATE_AUTOCAST.values():        # the two registrations agree
        assert all(fp.RESOLVED_AUTOCAST[(token, kind)][0] is enabled for kind in ("cpu", "cuda"))


def _autocast(gate, on, **overrides):
    """A gate's inputs carrying the autocast state its registered token resolves to on ``on``,
    with ``overrides`` (``"<deleted>"`` drops a field)."""
    requested = fp.GATE_AUTOCAST[gate][0]
    kind = on.partition(":")[0]
    enabled, dtype = {"off": (False, "float32"), "default": (
        True, "float16" if kind == "cuda" else "bfloat16")}[requested]
    inputs = {"device": on, "cond_autocast_requested": requested,
              "cond_autocast_enabled": enabled, "cond_autocast_dtype": dtype,
              "cond_autocast_device_type": kind}
    inputs.update(overrides)
    return {key: value for key, value in inputs.items() if value != "<deleted>"}


@pytest.mark.parametrize("gate", ["G1", "G3", "G3D"])
@pytest.mark.parametrize("device", ["cpu", "cuda:0", "cuda:1"])
def test_the_autocast_clause_holds_the_exact_resolved_state_on_the_evidence_device(gate, device):
    assert fp.autocast_reasons(gate, _autocast(gate, device)) == []
    kind = device.partition(":")[0]
    wrong_dtype = {"float32": "float16", "float16": "bfloat16", "bfloat16": "float16"}
    broken = [{"cond_autocast_device_type": "<deleted>", "cond_autocast_dtype": "banana"},
              {"cond_autocast_device_type": "<deleted>"},
              {"cond_autocast_dtype": "banana"},
              {"cond_autocast_dtype": wrong_dtype[_autocast(gate, device)[
                  "cond_autocast_dtype"]]},
              {"cond_autocast_device_type": "cpu" if kind == "cuda" else "cuda"},
              {"device": "mps:0", "cond_autocast_device_type": "mps"},
              {"device": None}, {"device": "<deleted>"},
              {"cond_autocast_enabled": int(_autocast(gate, device)["cond_autocast_enabled"])},
              {"cond_autocast_requested": "bf16"}]
    for overrides in broken:
        assert fp.autocast_reasons(gate, _autocast(gate, device, **overrides)), overrides


def test_codex_rbf2s_autocast_probe_does_not_re_verify(tmp_path, monkeypatch, clean_env):
    """Codex rB-F2: a missing device type and dtype ``banana`` returned []."""
    _code, evidence, _anchor = _registered_gate3(tmp_path, monkeypatch, "G3")
    probe = _tamper(_tamper(evidence, ["inputs", "cond_autocast_device_type"], delete=True),
                    ["inputs", "cond_autocast_dtype"], "banana")
    assert any(reason.startswith("autocast:") for reason in fp.reverify_gate_evidence(probe))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, "0.0", None, True])
def test_a_drift_measurement_with_a_delta_that_is_not_a_finite_number_is_not_complete(
        tmp_path, monkeypatch, clean_env, value):
    """Codex rB-F2: G3D 'complete' did not require its score deltas -- nor, threshold-free,
    its per-key deltas -- to be finite numeric measurements."""
    _code, evidence, _anchor = _registered_gate3(tmp_path, monkeypatch, "G3D", conditioner=Noisy())
    assert fp.reverify_gate_evidence(evidence) == []
    for path in (["queries", 0, "chunks", 0, "conditioning", "source", "tensor", "max_abs_diff"],
                 ["queries", 1, "chunks", 0, "dit_inputs", "global_cond", "max_abs_diff"],
                 ["queries", 0, "score_stability", "4", "max_abs_delta"]):
        reasons = fp.reverify_gate_evidence(_tamper(evidence, path, value))
        assert any("not a finite number" in reason for reason in reasons), (path, reasons)
    # the live measurement holds its own deltas to the same rule
    assert fp.measurement_reasons("8", {"max_abs_delta": value, "argmax_equal": True,
                                        "at_risk": False})
    assert fp.measurement_reasons("8", {"max_abs_delta": 0.5, "argmax_equal": True,
                                        "at_risk": False}) == []


def test_the_registered_dit_inputs_are_flac_ars_conditioning_inputs():
    """The registered set is what ``get_conditioning_inputs`` returns non-None for FLAC_AR's id
    lists -- derived from the model config, not copied."""
    from src.models.diffusion import ConditionedDiffusionModelWrapper

    config = json.load(open(os.path.join("src", "configs", "model_configs", "FLAC", "AR",
                                         "FLAC_AR.json")))["model"]["diffusion"]
    holder = type("Ids", (), {"cross_attn_cond_ids": config["cross_attention_cond_ids"],
                              "global_cond_ids": config["global_cond_ids"],
                              "input_concat_ids": config.get("input_concat_ids", []),
                              "prepend_cond_ids": config.get("prepend_cond_ids", [])})()
    tensors = {key: [torch.zeros(2, 4), torch.ones(2, 4)] for key in FA_IDS}
    produced = ConditionedDiffusionModelWrapper.get_conditioning_inputs(holder, tensors)
    assert sorted(name for name, value in produced.items() if value is not None) == sorted(
        fp.REGISTERED_DIT_INPUTS)


@pytest.mark.parametrize("edit", ["drop", "extra", "rename"])
def test_the_dit_input_clause_requires_exactly_the_registered_inputs(tmp_path, monkeypatch,
                                                                     clean_env, edit):
    """Codex rB-F2: ``inputs_reasons`` required a non-empty DiT-input map, not the registered
    key set -- in the live gates and in the re-verification."""
    _code, evidence, _anchor = _registered_gate3(tmp_path, monkeypatch, "G3")
    chunk = ["queries", 0, "chunks", 0, "dit_inputs"]
    facts = evidence["queries"][0]["chunks"][0]["dit_inputs"]
    edited = {"drop": {k: v for k, v in facts.items() if k != "global_cond"},
              "extra": dict(facts, prepend_cond=facts["global_cond"]),
              "rename": {("cond" if k == "global_cond" else k): v for k, v in facts.items()}}
    reasons = fp.reverify_gate_evidence(_tamper(evidence, chunk, edited[edit]))
    assert any("registered DiT inputs" in reason for reason in reasons), reasons
    engine = FaSyntheticEngine()
    inner = engine.cond_inputs_fn
    engine.cond_inputs_fn = lambda block: {k: v for k, v in inner(block).items()
                                           if k != "global_cond"}
    query, md = _gate_query(tmp_path / "live")
    live = fp.split_parity(engine, query, md, micro_batch=3, n_candidates=7)
    assert live["verdict"] == "FAIL" and any("registered DiT inputs" in reason
                                             for reason in live["reasons"])


@pytest.mark.parametrize("edit, needle", [
    (lambda counts: counts[0].update(count=True), "not a non-negative integer"),
    (lambda counts: counts[0].update(count=-1), "not a non-negative integer"),
    (lambda counts: counts[0].update(configured="32"), "not a non-negative integer"),
    (lambda counts: counts[0].update(count=0), "zero count"),
    (lambda counts: counts[0].update(extra=1), "rows"),
    (lambda counts: counts[0].update(base=7), "rows"),
    (lambda counts: counts[0].update(branch="nowhere"), "names branch 'nowhere'"),
    (lambda counts: counts.reverse(), "not canonical"),
    (lambda counts: counts.append(dict(counts[0])), "not canonical"),
])
def test_g3_executed_counts_go_through_the_ledgers_canonical_counts_validator(
        tmp_path, monkeypatch, clean_env, edit, needle):
    """Codex rB-F2: G3 / G3D raw executed counts were not checked for canonical shape,
    non-negative integers or booleans as the ledger's are -- one validator now serves both."""
    _code, evidence, _anchor = _registered_gate3(tmp_path, monkeypatch, "G3")
    edited = copy.deepcopy(evidence)
    edit(edited["executed_partition"]["counts"])
    reasons = fp.reverify_gate_evidence(edited)
    assert any(reason.startswith("executed partition") and needle in reason
               for reason in reasons), reasons
    # the very validator the ledger reader applies to every segment's counts
    assert any(needle in reason for reason in me.counts_reasons(
        edited["executed_partition"]["counts"], "executed partition")[0])


# --------------------------------------------------------------------------- #
# exp_22 rB-F3 (Codex code review rB-F2), finding 4 (part 2): the FULL evidence schema --
# every top-level and inputs field of each gate, exactly, each of its JSON type -- so a
# systematic type substitution is always a named reason, never an exception
# --------------------------------------------------------------------------- #
SUBSTITUTES = [None, "x", 7, 2.5, True, [], {}]


def _kind(value):
    return {bool: "bool", int: "int", float: "float", str: "str", list: "list", dict: "dict",
            type(None): "null"}[type(value)]


def _real_evidence(tmp_path, monkeypatch, gate):
    if gate in ("G3", "G3D"):
        return _registered_gate3(tmp_path, monkeypatch, gate)[1]
    return _real_gate_evidence(tmp_path, monkeypatch, gate)


@pytest.mark.parametrize("gate", ["G1", "G2", "G3", "G3D", "G0", "GV"])
def test_a_systematic_type_substitution_is_always_a_named_reason(tmp_path, monkeypatch,
                                                                 clean_env, gate):
    """Every top-level and ``inputs`` field of a real evidence, replaced by a value of each
    other JSON type, and deleted: ``reverify_gate_evidence`` names a reason every time and
    never raises (Codex rB-F2: "validate the full evidence schema")."""
    evidence = json.loads(json.dumps(_real_evidence(tmp_path, monkeypatch, gate)))
    assert fp.reverify_gate_evidence(evidence) == []
    missed = []
    for path in [[field] for field in evidence] + [["inputs", field]
                                                   for field in evidence["inputs"]]:
        original = evidence[path[0]] if len(path) == 1 else evidence["inputs"][path[1]]
        for value in [s for s in SUBSTITUTES if _kind(s) != _kind(original)] + ["<deleted>"]:
            edited = (_tamper(evidence, path, delete=True) if value == "<deleted>"
                      else _tamper(evidence, path, value))
            reasons = fp.reverify_gate_evidence(edited)
            if not (reasons and all(isinstance(reason, str) for reason in reasons)):
                missed.append((".".join(path), value))
    assert missed == []
    top, inputs = fp.EVIDENCE_SCHEMA[gate]
    assert set(evidence) == set(top) and set(evidence["inputs"]) == set(inputs)


def test_an_unregistered_field_or_an_edited_registered_text_does_not_re_verify(
        tmp_path, monkeypatch, clean_env):
    evidence = _real_evidence(tmp_path, monkeypatch, "G3")
    for path, value, needle in ((["tolerance_override"], 1.0, "unregistered fields"),
                                (["inputs", "smuggled"], 1, "unregistered fields"),
                                (["note"], "a different note", "registered text"),
                                (["reverify_note"], "trust me", "registered text")):
        reasons = fp.reverify_gate_evidence(_tamper(evidence, path, value))
        assert any(needle in reason for reason in reasons), (path, reasons)


# --------------------------------------------------------------------------- #
# exp_22 rB-F3 (Codex code review rB-F2), finding 3 (part 1): a missing ledger is admitted
# only in a FRESH out-dir; every ledger integer is a real int; the ledger is opened -- and on
# disk -- before the binding, a launch segment or a row is written
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("deleted", [("ledger", "summary"), ("ledger",)])
def test_a_missing_ledger_is_refused_in_an_out_dir_that_already_holds_a_run(tmp_path, deleted):
    """Codex rB-F2: deleting the ledger and the run summary let a reopen bootstrap a
    replacement ledger over existing rows."""
    _engine, summary, out, (plan, records, items) = _fa_pass(tmp_path)
    me.write_json(_fa_files(out)[1], summary)
    for path, name in zip(_fa_files(out), ("ledger", "summary")):
        if name in deleted:
            os.remove(path)
    before = _snapshot(out)
    done, _rejected = me.completed_queries(out)
    engine = FaSyntheticEngine()
    with pytest.raises(ValueError, match="not a fresh out-dir"):
        me.run_pass(engine, items, records, plan, out, done=done, fa_partition=FA_DECLARED,
                    **FA_SHAPES)
    assert _snapshot(out) == before and engine.conditioner.calls == []
    with pytest.raises(ValueError, match="not a fresh out-dir"):
        me.read_fa_ledger(out, FA_DECLARED)


RUN_ARTIFACTS = {
    "binding": lambda out: me.write_binding(out, _registered_binding("BF")),
    "run summary": lambda out: me.write_json(os.path.join(out, "run_summary.json"), {}),
    "launch segments": lambda out: me.write_json(os.path.join(out, "launch_segments.json"),
                                                 {"segments": []}),
    "launch records dir": lambda out: os.makedirs(os.path.join(out, "launch_records")),
    "a row": lambda out: (os.makedirs(os.path.join(out, "rows", "A_A_idx_1")),
                          me.write_json(os.path.join(out, "rows", "A_A_idx_1", "q00000.json"),
                                        {})),
    "merge report": lambda out: me.write_json(os.path.join(out, "merge_report.json"), {}),
    "a half-written ledger": lambda out: open(os.path.join(
        out, me.FA_LEDGER_FILENAME + ".tmp"), "w").write("{"),
    "a probe record": lambda out: me.write_json(os.path.join(out, "diagnostics_probe.json"), {}),
    "anything else": lambda out: open(os.path.join(out, "notes.txt"), "w").write("x"),
}


@pytest.mark.parametrize("artifact", sorted(RUN_ARTIFACTS))
def test_a_missing_ledger_is_admitted_only_in_a_fresh_out_dir(tmp_path, artifact):
    """Fresh = absent, or holding only the operator's emitted launch records (top-level
    *.json files the run never writes) -- the registered launch lines emit the record into
    the shard directory before its first run."""
    out = str(tmp_path / "shard")
    assert me.read_fa_ledger(out, FA_DECLARED) is None                       # absent
    os.makedirs(out)
    assert me.read_fa_ledger(out, FA_DECLARED) is None                       # empty
    for name in ("launch_record.json", "launch_record_resume1.json"):
        me.write_json(os.path.join(out, name), {"argv": []})
    assert me.read_fa_ledger(out, FA_DECLARED) is None                       # emitted records
    RUN_ARTIFACTS[artifact](out)
    with pytest.raises(ValueError, match="not a fresh out-dir"):
        me.read_fa_ledger(out, FA_DECLARED)
    assert me.read_fa_ledger(out, None) is None or artifact == "merge report"


@pytest.mark.parametrize("indices", [[False, True], [0, True], [0.0, 1], ["0", 1]])
def test_every_ledger_segment_index_is_a_real_integer(tmp_path, indices):
    """Codex rB-F2: segment indices accepted False / True as 0 / 1."""
    _engine, summary, out, stack = _fa_pass(tmp_path)
    me.write_json(_fa_files(out)[1], summary)
    _two_segment_run(out, stack)
    ledger = _ledger(out)
    for segment, index in zip(ledger["segments"], indices):
        segment["segment_index"] = index
    me.write_json(_fa_files(out)[0], ledger)
    with pytest.raises(ValueError, match="segment indices"):
        me.read_fa_ledger(out, FA_DECLARED)


def test_the_driver_opens_the_ledger_before_it_writes_the_binding(tmp_path, monkeypatch):
    """claim_out_dir opens (and publishes) this invocation's ledger segment FIRST, so the
    binding and the launch segment never exist in an out-dir without a ledger."""
    import localize_meshgrid as driver

    # claim_out_dir's own inputs (a record-free namespace: the record is the driver's business)
    args = driver.parse_args(["--ckpt-path", "x.ckpt", "--out-dir", str(tmp_path / "shard")]
                             + ENG._FA_PROTOCOL)
    os.makedirs(args.out_dir)
    me.write_json(os.path.join(args.out_dir, "launch_record.json"), {"argv": []})
    binding, advisory = _registered_binding("BF"), {"source_chunk": 16, "batch_rows": 256}
    seen, real = [], me.write_binding
    monkeypatch.setattr(me, "write_binding", lambda out, *a, **k: (seen.append(os.path.isfile(
        os.path.join(out, me.FA_LEDGER_FILENAME))), real(out, *a, **k))[1])
    _done, _segment, ledger = driver.claim_out_dir(args, binding, advisory, None)
    assert seen == [True]
    on_disk = _ledger(args.out_dir)
    assert [(s["segment_index"], s["n_queries"], s["counts"]) for s in on_disk["segments"]] == [
        (0, 0, [])] and on_disk == ledger["payload"]
    args.resume = True                                  # a reopen appends ITS segment first
    _done, _segment, ledger = driver.claim_out_dir(args, binding, advisory, None)
    assert [s["segment_index"] for s in _ledger(args.out_dir)["segments"]] == [0, 1]
    # a vanilla claim opens no ledger, exactly as before
    args.out_dir, args.resume = str(tmp_path / "vanilla"), False
    assert driver.claim_out_dir(args, _registered_binding("YAW"), advisory, None)[2] is None
    assert not os.path.exists(os.path.join(args.out_dir, me.FA_LEDGER_FILENAME))


def test_a_pass_adopts_only_the_ledger_opened_for_its_out_dir_declaration_and_record(tmp_path):
    plan, records, items = _fa_aligned(tmp_path)
    out = str(tmp_path / "run")
    ledger = me.open_fa_ledger(out, FA_DECLARED, None)
    assert os.path.isfile(os.path.join(out, me.FA_LEDGER_FILENAME))          # on disk at once
    for kwargs in ({"out_dir": str(tmp_path / "elsewhere")},
                   {"fa_partition": me.declared_fa_partition(list(C4), batch_rows=8, num_samples=4,
                                                            source_chunk=5)},
                   {"launch_record_sha256": "f" * 64}):
        call = dict({"out_dir": out, "fa_partition": FA_DECLARED, "launch_record_sha256": None},
                    **kwargs)
        with pytest.raises(ValueError, match="not the one just opened"):
            me.run_pass(FaSyntheticEngine(), items, records, plan, call["out_dir"],
                        fa_ledger=ledger, fa_partition=call["fa_partition"],
                        launch_record_sha256=call["launch_record_sha256"], **FA_SHAPES)
    summary = me.run_pass(FaSyntheticEngine(), items, records, plan, out, fa_ledger=ledger,
                          fa_partition=FA_DECLARED, **FA_SHAPES)
    assert summary["fa_partition_executed"]["executed"] == FA_DECLARED
    assert len(_ledger(out)["segments"]) == 1                    # adopted, not a second one


# --------------------------------------------------------------------------- #
# exp_22 rB-F3 (Codex code review rB-F2), finding 1 (BLOCKING): the merged BF ledger is
# valid under its own reader -- one index space, each segment keeping its shard origin --
# and the canonical per-arm report re-reads it and joins it to merge_report.json's receipt
# --------------------------------------------------------------------------- #
def _noop_resume(out, stack, rooms):
    """A no-op resume of a finished fa shard: one more (empty) ledger segment + its summary."""
    plan, records, items = stack
    done, _rejected = me.completed_queries(out)
    bound = json.load(open(os.path.join(out, me.BINDING_FILENAME)))["binding_sha256"]
    summary = me.run_pass(FaSyntheticEngine(), items, records, plan, out, done=done, rooms=rooms,
                          binding_sha256=bound, fa_partition=FA_DECLARED, **FA_SHAPES)
    assert summary["n_scored"] == 0
    me.write_json(_fa_files(out)[1], summary)


@pytest.mark.parametrize("extra", [0, 1, 2])
def test_the_merged_ledger_is_valid_under_its_own_reader(tmp_path, extra):
    """Codex rB-F2: two one-segment shards merged to segment indices [0, 0], which the reader
    itself refuses. The merge renumbers 0..n-1 (shard order, then each shard's own order) and
    keeps each segment's origin; the merged ledger reads EXACTLY."""
    plan, records, dirs, totals = _fa_shards(tmp_path)
    for name, rooms in (("a", ["A/A_idx_1"]), ("b", ["B/B_idx_2"])):
        for _ in range(extra):
            _noop_resume(dirs[name], _fa_aligned(tmp_path / f"stack_{name}"), rooms)
    origins = [(dirs[name], segment) for name in ("a", "b")
               for segment in _ledger(dirs[name])["segments"]]
    merged = str(tmp_path / "merged")
    report = me.merge_shards([dirs["a"], dirs["b"]], merged, plan, records, totals=totals)
    kept = _ledger(merged)
    assert [s["segment_index"] for s in kept["segments"]] == list(range(2 * (1 + extra)))
    for segment, (shard, origin) in zip(kept["segments"], origins):
        assert (segment["shard"], segment["shard_segment_index"]) == (shard,
                                                                      origin["segment_index"])
        assert {k: v for k, v in segment.items()
                if k not in ("segment_index", "shard", "shard_segment_index")} == {
            k: v for k, v in origin.items() if k != "segment_index"}
    found = me.read_fa_ledger(merged, FA_DECLARED, exact=True)
    assert found["n_segments"] == 2 * (1 + extra)
    assert found["counts_sha256"] == report["fa_partition_executed"]["counts_sha256"]
    assert [entry["n_segments"] for entry in report["fa_partition_executed"]["per_shard"]] == [
        1 + extra, 1 + extra]
    with pytest.raises(ValueError, match="never reopened"):              # a merged run is final
        me.read_fa_ledger(merged, FA_DECLARED)


def _bf_merged(tmp_path):
    import test_loc_meshgrid_report as RPT

    fixture = RPT.build_fixture_run(tmp_path, merged=True, arm="BF",
                                    launch=RPT.fixture_launch(tmp_path, "BF"))
    assert RPT.evaluate_fixture(fixture)["registered_protocol"]["arm"] == "BF"     # admissible
    return RPT, fixture


def _edit_json(path, edit):
    payload = json.load(open(path))
    edit(payload)
    me.write_json(path, payload)


def _resign_merged(ledger):
    """An edited merged ledger with every SELF-digest recomputed, so only the join to the
    receipt (or the origin rule) can catch the edit."""
    _resummed(ledger)
    return ledger


MERGED_CORRUPTIONS = {
    "ledger deleted": ("ledger", None, "publishes no fa_partition_ledger.json"),
    "receipt deleted": ("receipt", lambda r: r.pop("fa_partition_executed"), "receipt"),
    "receipt null": ("receipt", lambda r: r.update(fa_partition_executed=None), "receipt"),
    "receipt digest moved": ("receipt", lambda r: r["fa_partition_executed"].update(
        counts_sha256="0" * 64), "receipt"),
    "receipt executed moved": ("receipt", lambda r: r["fa_partition_executed"]["executed"].update(
        orbit_size=5), "receipt"),
    "receipt census moved": ("receipt", lambda r: r["fa_partition_executed"]["per_shard"][0].update(
        n_segments=2), "receipt"),
    "a count edited, self-digests recomputed": ("ledger", lambda l: (
        l["segments"][0]["counts"][0].update(count=l["segments"][0]["counts"][0]["count"] + 1),
        _resign_merged(l)), "receipt"),
    "indices duplicated": ("ledger", lambda l: l["segments"][1].update(segment_index=0),
                           "segment indices"),
    "an origin dropped": ("ledger", lambda l: l["segments"][1].pop("shard_segment_index"),
                          "shard"),
    "shard blocks swapped": ("ledger", lambda l: (l["segments"].reverse(), [
        s.update(segment_index=i) for i, s in enumerate(l["segments"])]), "receipt"),
}


@pytest.mark.parametrize("name", sorted(MERGED_CORRUPTIONS))
def test_the_per_arm_report_refuses_a_merged_bf_run_whose_ledger_or_receipt_does_not_join(
        tmp_path, name):
    """Codex rB-F2: the report checked only the failure marker, so the merged ledger or the
    receipt could change after the merge without invalidating the per-arm (or crossarm)
    report. It now re-reads the ledger exactly and joins it to the receipt."""
    from src.localization import meshgrid_crossarm as mx

    RPT, fixture = _bf_merged(tmp_path)
    run = fixture["run_dir"]
    where, edit, needle = MERGED_CORRUPTIONS[name]
    if edit is None:
        os.remove(os.path.join(run, me.FA_LEDGER_FILENAME))
    else:
        _edit_json(os.path.join(run, me.FA_LEDGER_FILENAME if where == "ledger"
                                else "merge_report.json"), edit)
    with pytest.raises(ValueError, match=re_escape(needle)):
        RPT.evaluate_fixture(fixture)
    with pytest.raises(ValueError, match=re_escape(needle)):              # the crossarm inherits
        mx.admit_run(run, "BF", audit_report=fixture["audit_report"],
                     context_manifest=fixture["context_manifest"],
                     metadata_root=fixture["metadata_root"],
                     expect_metadata_bank_sha256=fixture["metadata_bank_sha256"],
                     totals=fixture["totals"], require_manifest_census=False)


def test_a_single_shard_bf_report_re_reads_its_ledger_against_its_run_summary(tmp_path):
    import test_loc_meshgrid_report as RPT

    fixture = RPT.build_fixture_run(tmp_path, merged=False, arm="BF")
    assert RPT.evaluate_fixture(fixture)["registered_protocol"]["arm"] == "BF"
    _edit_json(os.path.join(fixture["run_dir"], "run_summary.json"),
               lambda s: s["fa_partition_executed"]["ledger"].update(counts_sha256="0" * 64))
    with pytest.raises(ValueError, match="run summary"):
        RPT.evaluate_fixture(fixture)


@pytest.mark.parametrize("state, refused", [("no block, legacy shape", False),
                                            ("null receipt", False),
                                            ("a ledger file", True),
                                            ("a non-null receipt", True)])
def test_a_vanilla_report_admits_no_fa_ledger_and_no_fa_receipt(tmp_path, state, refused):
    """The vanilla side of the join: an fa ledger or a non-null receipt in a P1 / YAW run is
    refused; absent (the Aug-2026 P1 merged run carries neither) or null is admitted."""
    import test_loc_meshgrid_report as RPT

    fixture = RPT.build_fixture_run(tmp_path)                        # the P1 fixture, merged
    report_path = os.path.join(fixture["run_dir"], "merge_report.json")
    if state == "no block, legacy shape":
        _edit_json(report_path, lambda r: r.pop("fa_partition_executed"))
    elif state == "a ledger file":
        me.write_json(os.path.join(fixture["run_dir"], me.FA_LEDGER_FILENAME), {"segments": []})
    elif state == "a non-null receipt":
        _edit_json(report_path, lambda r: r.update(fa_partition_executed={"executed": {}}))
    if refused:
        with pytest.raises(ValueError, match="vanilla run carries no frame-average"):
            RPT.evaluate_fixture(fixture)
    else:
        assert RPT.evaluate_fixture(fixture)["registered_protocol"]["arm"] == "P1"


# --------------------------------------------------------------------------- #
# exp_22 rB-F3 (Codex code review rB-F2), finding 3 (part 2): every ledger state is bound to
# the published rows -- each fa row carries the state persisted just before it, and every
# reader (reopen, merge, report) holds each segment's stored state to its own last row
# --------------------------------------------------------------------------- #
def _crashed_pass(tmp_path, monkeypatch, crash_at=3):
    """A pass killed between the ledger write and the row write of its ``crash_at``-th row:
    the ledger is left exactly one query AHEAD of the rows (it is written first)."""
    real, written = me.write_query_artifact, []

    def crash(*args, **kwargs):
        if len(written) == crash_at - 1:
            raise RuntimeError("killed between the ledger write and the row write")
        written.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(me, "write_query_artifact", crash)
    plan, records, items = _fa_aligned(tmp_path)
    out = str(tmp_path / "run")
    with pytest.raises(RuntimeError, match="killed"):
        me.run_pass(FaSyntheticEngine(), items, records, plan, out, fa_partition=FA_DECLARED,
                    **FA_SHAPES)
    monkeypatch.setattr(me, "write_query_artifact", real)
    return out, (plan, records, items)


def _stamps(out):
    return sorted((me.read_fa_stamp(out, row)[1] for row in me.read_rows(out)),
                  key=lambda stamp: stamp["n_queries"])


def _stamp_file(out, row):
    return me.fa_stamp_path(out, row["room_id"], row["position"])


def test_every_fa_row_carries_the_ledger_state_it_was_published_under(tmp_path):
    _engine, _summary, out, _stack = _fa_pass(tmp_path)
    segment = _ledger(out)["segments"][0]
    stamps = _stamps(out)
    assert [s["n_queries"] for s in stamps] == list(range(1, segment["n_queries"] + 1))
    assert {s["segment_index"] for s in stamps} == {0}
    assert stamps[-1]["counts_sha256"] == segment["counts_sha256"]          # exact at the end
    assert sorted(stamps[0]) == ["counts_sha256", "n_queries", "query_id", "segment_index"]
    # the stamp lives BESIDE the row: an fa row keeps exactly the vanilla row schema, so the
    # crossarm's pairing of BF rows with P1 / YAW rows is unchanged
    plan, records, items = ENG._aligned(tmp_path / "vanilla")
    me.run_pass(ENG.SyntheticEngine(), items, records, plan, str(tmp_path / "vanilla_run"),
                **{k: v for k, v in FA_SHAPES.items()})
    assert {frozenset(row) for row in me.read_rows(out)} == {
        frozenset(row) for row in me.read_rows(str(tmp_path / "vanilla_run"))}
    # the delta binds the state at n - 1: counts minus delta hashes to the previous row's stamp
    prior = me.FaPartitionRecorder.from_counts(segment["counts"])
    for entry in segment["delta_counts"]:
        key = (entry["configured"], entry["cap"], tuple(entry["base"]), tuple(entry["orbit"]),
               entry["orbit_size"])
        bucket = prior.counts[entry["branch"]]
        bucket[key] -= entry["count"]
        if not bucket[key]:
            del bucket[key]
    assert me.counts_sha256(prior) == stamps[-2]["counts_sha256"]


def _truncate_tail(ledger, dropped, delta_policy):
    """Codex rB-F2's probe: drop call rows of the tail segment -- its full-size source-cache
    rows (an unsaturated, tail-only prefix) or every tail-size row (the executed block
    unchanged: lost call counts only) -- and recompute every self-digest; ``delta_policy``
    keeps the delta inside the truncated counts, or also claims one query ahead."""
    segment = ledger["segments"][-1]
    full = {"full-size source calls": lambda e: e["branch"] == "source_cache"
                                                and e["base"] == [e["configured"]],
            "tail-size calls": lambda e: e["base"] != [e["configured"]]}[dropped]
    segment["counts"] = [entry for entry in segment["counts"] if not full(entry)]
    segment["delta_counts"] = []
    if delta_policy == "one ahead":
        segment["n_queries"] += 1
    return _resummed(ledger)


@pytest.mark.parametrize("dropped", ["full-size source calls", "tail-size calls"])
@pytest.mark.parametrize("delta_policy", ["emptied", "one ahead"])
def test_codex_rbf2s_truncated_tail_probe_is_refused_at_reopen(tmp_path, dropped, delta_policy):
    """A truncated tail segment with recomputed self-digests and no summary passed the
    prefix rule at rB-F2 (and full-size calls of a resume could then saturate the maxima):
    its rows' stamps now bind the state, so the reopen is refused before anything is
    written -- the reader alone already refuses it."""
    _engine, _summary, out, (plan, records, items) = _fa_pass(tmp_path)
    assert not os.path.exists(_fa_files(out)[1])                    # no summary, as in the probe
    ledger = _truncate_tail(_ledger(out), dropped, delta_policy)
    me.write_json(_fa_files(out)[0], ledger)
    with pytest.raises(ValueError, match="last row was published under"):
        me.read_fa_ledger(out, FA_DECLARED)
    before = _snapshot(out)
    done, _rejected = me.completed_queries(out)
    engine = FaSyntheticEngine()
    with pytest.raises(ValueError, match="last row was published under"):
        me.run_pass(engine, items, records, plan, out, done=done, fa_partition=FA_DECLARED,
                    **FA_SHAPES)
    assert _snapshot(out) == before and engine.conditioner.calls == []


def test_a_state_one_query_ahead_is_admitted_at_reopen_and_bound_by_its_prefix(tmp_path,
                                                                              monkeypatch):
    out, (plan, records, items) = _crashed_pass(tmp_path, monkeypatch)
    segment = _ledger(out)["segments"][0]
    assert segment["n_queries"] == 3 and len(me.read_rows(out)) == 2         # one ahead
    assert me.read_fa_ledger(out, FA_DECLARED)["n_segments"] == 1              # admitted
    for edit, needle in ((lambda s: s.update(delta_counts=[]), "last row was published under"),
                         (lambda s: s.update(n_queries=4), "its rows show 2"),
                         (lambda s: s.update(n_queries=1), "beyond its segment's 1 queries")):
        ledger = _ledger(out)
        edit(ledger["segments"][0])
        me.write_json(_fa_files(out)[0], _resummed(ledger))
        with pytest.raises(ValueError, match=re_escape(needle)):
            me.read_fa_ledger(out, FA_DECLARED)
        me.write_json(_fa_files(out)[0], _restore(out, segment))


def _restore(out, segment):
    ledger = _ledger(out)
    ledger["segments"] = [segment]
    return _resummed(ledger)


def test_at_a_merge_the_last_invocation_is_exact_and_a_lost_attempt_was_executed_again(
        tmp_path, monkeypatch):
    """A crashed segment stays one query ahead forever (its lost row is regenerated by the
    resume): admitted at the merge only if it is not the shard's last invocation and every
    call shape its lost attempt made was executed again by a later segment."""
    out, (plan, records, items) = _crashed_pass(tmp_path, monkeypatch)
    done, _rejected = me.completed_queries(out)
    summary = me.run_pass(FaSyntheticEngine(), items, records, plan, out, done=done,
                          fa_partition=FA_DECLARED, **FA_SHAPES)
    me.write_json(_fa_files(out)[1], summary)
    assert me.read_fa_ledger(out, FA_DECLARED, exact=True)["n_segments"] == 2   # merge-ready
    # a call shape (a source chunk of 2) that no invocation of this shard ever executed
    phantom = {"branch": "source_cache", "configured": 3, "cap": 2, "base": [2],
               "orbit": [2, 2, 2], "orbit_size": 4, "count": 1}
    assert all(entry["base"] != [2] for segment in _ledger(out)["segments"]
               for entry in segment["counts"] if entry["branch"] == "source_cache")
    for edit, needle in (
            (lambda l: [l["segments"][0][field].append(dict(phantom))
                        for field in ("counts", "delta_counts")], "executed again"),
            # the last invocation one query ahead, its prefix consistent (an empty lost attempt)
            (lambda l: l["segments"][1].update(n_queries=l["segments"][1]["n_queries"] + 1,
                                               delta_counts=[]), "last invocation")):
        ledger = _ledger(out)
        edit(ledger)
        for segment in ledger["segments"]:
            for field in ("counts", "delta_counts"):
                segment[field] = me.FaPartitionRecorder.from_counts(segment[field]).to_counts()
        _resummed(ledger)
        summary["fa_partition_executed"] = me.fa_executed_block(
            FA_DECLARED, me.FaPartitionRecorder.from_counts(
                *[s["counts"] for s in ledger["segments"]]), 2)
        me.write_json(_fa_files(out)[0], ledger)
        me.write_json(_fa_files(out)[1], summary)
        with pytest.raises(ValueError, match=needle):
            me.read_fa_ledger(out, FA_DECLARED, exact=True)


@pytest.mark.parametrize("edit, needle", [
    (lambda stamp: stamp.update(counts_sha256="0" * 64), "published under"),
    (lambda stamp: stamp.update(segment_index=7), "does not hold"),
    (lambda stamp: stamp.update(n_queries=True), "not a"),
    (lambda stamp: stamp.update(query_id="9|another"), "not a"),
    (lambda stamp: stamp.pop("counts_sha256"), "not a"),
    (None, "not a"),                                            # the stamp file deleted
])
def test_a_row_whose_stamp_does_not_bind_its_ledger_is_refused(tmp_path, edit, needle):
    """A moved stamp of the last row (the coordinated edit a ledger self-digest cannot catch)
    no longer binds its segment's state -- refused at reopen and at merge."""
    _engine, summary, out, _stack = _fa_pass(tmp_path)
    me.write_json(_fa_files(out)[1], summary)
    last = max(me.read_rows(out), key=lambda row: me.read_fa_stamp(out, row)[1]["n_queries"])
    if edit is None:
        os.remove(_stamp_file(out, last))
    else:
        _edit_json(_stamp_file(out, last), edit)
    for exact in (False, True):
        with pytest.raises(ValueError, match=needle):
            me.read_fa_ledger(out, FA_DECLARED, exact=exact)


def test_two_rows_may_not_carry_the_same_ledger_state(tmp_path):
    _engine, summary, out, _stack = _fa_pass(tmp_path)
    rows = sorted(me.read_rows(out), key=lambda row: me.read_fa_stamp(out, row)[1]["n_queries"])
    me.write_json(_stamp_file(out, rows[0]), dict(me.read_fa_stamp(out, rows[1])[1],
                                                  query_id=rows[0]["query_id"]))
    with pytest.raises(ValueError, match="twice"):
        me.read_fa_ledger(out, FA_DECLARED)


def test_the_merged_report_binds_every_merged_row_to_its_shards_segment(tmp_path):
    RPT, fixture = _bf_merged(tmp_path)
    run = fixture["run_dir"]
    rows = me.read_rows(run)
    assert all(sorted(me.read_fa_stamp(run, row)[1]) == ["counts_sha256", "n_queries",
                                                         "query_id", "segment_index"]
               for row in rows)
    # the LAST row of a segment carries the state its segment's stored state is bound to
    last = max(rows, key=lambda row: me.read_fa_stamp(run, row)[1]["n_queries"])
    _edit_json(_stamp_file(run, last), lambda stamp: stamp.update(counts_sha256="0" * 64))
    with pytest.raises(ValueError, match="published under"):
        RPT.evaluate_fixture(fixture)
    os.remove(_stamp_file(run, rows[0]))
    with pytest.raises(ValueError, match="not a"):
        RPT.evaluate_fixture(fixture)


# --------------------------------------------------------------------------- #
# exp_22 rB-F3 (Codex code review rB-F2), NIT 5: the kill window between an executed-
# partition check and its verdict is closed -- a pending-check sentinel is published before
# each check and removed only once it passed; a directory still carrying it is refused
# --------------------------------------------------------------------------- #
PENDING = "partition_check_pending.json"


def test_the_sentinel_is_on_disk_exactly_while_each_partition_check_runs(tmp_path, monkeypatch):
    real, seen = me.assert_fa_partition_executed, []

    def watched(declared, recorder, final=True):
        seen.append((final, os.path.exists(os.path.join(str(tmp_path / "run"), PENDING))))
        return real(declared, recorder, final=final)

    monkeypatch.setattr(me, "assert_fa_partition_executed", watched)
    _engine, _summary, out, _stack = _fa_pass(tmp_path)
    rows = len(me.read_rows(out))
    # every before-row check and the final one ran with the sentinel published
    assert [final for final, _ in seen].count(False) == rows and seen[-1][0] is True
    assert all(pending for _final, pending in seen)
    assert not os.path.exists(os.path.join(out, PENDING))           # removed once each passed
    me.assert_admissible(out)
    # a probe judges nothing durable and publishes no sentinel
    seen.clear()
    _engine, _summary, probe_out, _stack = _fa_pass(tmp_path / "probe", probe=1)
    assert not os.path.exists(os.path.join(probe_out, PENDING))


class _Killed(BaseException):
    """A hard kill: not an Exception, so no handler of the pass can run."""


@pytest.mark.parametrize("where", ["inside the before-row check", "inside the final check",
                                   "after a check passed, before its sentinel is removed"])
def test_a_shard_killed_inside_a_partition_check_is_non_admissible(tmp_path, monkeypatch, where):
    """Codex rB-F2 NIT 5: a kill between a failed check and the failure marker left the
    before-row ledger holding the previous good prefix -- fail-open. The sentinel makes any
    kill inside the window conservative: the shard is refused and restarted (A6 iv)."""
    real_check, real_end = me.assert_fa_partition_executed, me.end_partition_check
    if where == "inside the before-row check":
        state = {"n": 0}

        def killed(declared, recorder, final=True):
            state["n"] += 1
            if state["n"] == 3:
                raise _Killed()
            return real_check(declared, recorder, final=final)
        monkeypatch.setattr(me, "assert_fa_partition_executed", killed)
    elif where == "inside the final check":
        def killed(declared, recorder, final=True):
            if final:
                raise _Killed()
            return real_check(declared, recorder, final=final)
        monkeypatch.setattr(me, "assert_fa_partition_executed", killed)
    else:
        def killed(out_dir):
            raise _Killed()
        monkeypatch.setattr(me, "end_partition_check", killed)
    plan, records, items = _fa_aligned(tmp_path)
    out = str(tmp_path / "run")
    with pytest.raises(_Killed):
        me.run_pass(FaSyntheticEngine(), items, records, plan, out, fa_partition=FA_DECLARED,
                    **FA_SHAPES)
    monkeypatch.setattr(me, "assert_fa_partition_executed", real_check)
    monkeypatch.setattr(me, "end_partition_check", real_end)
    assert json.load(open(os.path.join(out, PENDING)))["check"] in ("before_row", "final")
    before = _snapshot(out)
    done, _rejected = me.completed_queries(out)
    for refuse in (lambda: me.run_pass(FaSyntheticEngine(), items, records, plan, out, done=done,
                                       fa_partition=FA_DECLARED, **FA_SHAPES),
                   lambda: me.read_fa_ledger(out, FA_DECLARED),
                   lambda: me.assert_admissible(out)):
        with pytest.raises(ValueError, match=r"NON-ADMISSIBLE.*A6 \(iv\)"):
            refuse()
    assert _snapshot(out) == before


def test_the_merge_and_the_report_refuse_a_directory_carrying_the_sentinel(tmp_path):
    import test_loc_meshgrid_report as RPT

    plan, records, dirs, totals = _fa_shards(tmp_path)
    me.write_json(os.path.join(dirs["b"], PENDING), {"check": "before_row"})
    merged = str(tmp_path / "merged")
    with pytest.raises(ValueError, match=r"NON-ADMISSIBLE.*A6 \(iv\)"):
        me.merge_shards([dirs["a"], dirs["b"]], merged, plan, records, totals=totals)
    assert not os.path.exists(merged)
    fixture = RPT.build_fixture_run(tmp_path / "report", merged=False, arm="BF")
    assert RPT.evaluate_fixture(fixture)["registered_protocol"]["arm"] == "BF"
    me.write_json(os.path.join(fixture["run_dir"], PENDING), {"check": "final"})
    with pytest.raises(ValueError, match=r"NON-ADMISSIBLE.*A6 \(iv\)"):
        RPT.evaluate_fixture(fixture)


# --------------------------------------------------------------------------- #
# exp_22 rB-F3: ISOLATED negatives for re-verifier paths that were tamper-tested only through
# edits another clause also catches (Codex rB-F2 C8) -- each edit below is named by exactly
# one clause, so deleting that clause is observable
# --------------------------------------------------------------------------- #
ISOLATED_TAMPERS = [
    # (gate, path, edit, the only clause's needle)
    ("G3", ["inputs", "tau"], {"value": 0.05}, "inputs: tau"),
    ("G3", ["digest_chains", "source", 0], {"value": "0" * 64}, "digest_chains"),
    ("G3", ["score_stability", "8", "max_abs_delta"], {"value": 0.5}, "score_stability"),
    ("G3", ["executed_partition", "executed", "orbit_size"], {"value": 5},
     "the published block is not the one its call counts yield"),
    ("G0", ["inputs", "published_binding_sha256"], {"value": "0" * 64},
     "does not hash to its published_binding_sha256"),
    ("G0", ["inputs", "binding", "tau"], {"value": 0.05},
     "the replay does not run the published protocol"),
    ("G0", ["queries", 0, "published_digests", "row_sha256"], {"value": "0" * 64},
     "the published row's grid digests are not the frozen ones"),
    ("G1", ["inputs", "angles"], {"value": [0.0, 90.0, 180.0, 270.0, 45.0]}, "compared orbit"),
    ("G1", ["coverage", "tail"], {"value": 9}, "coverage"),
    ("G1", ["per_key", "source", "max_abs_diff"], {"value": 1.0}, "per_key"),
    ("GV", ["vanilla_cache_parity", "match"], {"value": False}, "flags are not re-verified"),
    ("GV", ["inputs", "query_id"], {"value": "0|another"}, "parity report covers"),
    ("G2", ["counter_tests", "extra"], {"value": {"keys": {"source": 1.0}, "detected": True}},
     "counter-test: published"),
    ("G2", ["per_key", "source", "max_abs_diff"], {"value": 1.0}, "per_key"),
]


@pytest.mark.parametrize("gate, path, edit, needle", ISOLATED_TAMPERS,
                         ids=[f"{t[0]}-{'.'.join(map(str, t[1]))}" for t in ISOLATED_TAMPERS])
def test_an_isolated_tamper_is_named_by_its_own_clause(tmp_path, monkeypatch, clean_env, gate,
                                                       path, edit, needle):
    evidence = _real_evidence(tmp_path, monkeypatch, gate)
    reasons = fp.reverify_gate_evidence(_tamper(evidence, path, **edit))
    assert any(needle in reason for reason in reasons), reasons


def test_the_merge_refuses_a_merged_ledger_its_reader_would_refuse(tmp_path, monkeypatch):
    """The census verifies the merged ledger and the receipt as the reader will, and the
    merge binds the merged ledger to the rows it publishes -- both in memory, before any
    write (defence in depth: a census bug would otherwise publish a ledger the per-arm report
    refuses)."""
    plan, records, dirs, totals = _fa_shards(tmp_path)
    real_merged, real_binding = me._merged_reasons, me._row_binding_reasons
    monkeypatch.setattr(me, "_merged_reasons", lambda *a: (["a simulated census bug"], None))
    with pytest.raises(ValueError, match="would not verify under its own reader"):
        me.merge_shards([dirs["a"], dirs["b"]], str(tmp_path / "m1"), plan, records,
                        totals=totals)
    monkeypatch.setattr(me, "_merged_reasons", real_merged)

    def merged_only(segments, stamps, exact, blocks=None):    # the shards' own reads still pass
        return (["a simulated binding bug"] if blocks is not None
                else real_binding(segments, stamps, exact, blocks))

    monkeypatch.setattr(me, "_row_binding_reasons", merged_only)
    with pytest.raises(ValueError, match="do not bind the merged ledger"):
        me.merge_shards([dirs["a"], dirs["b"]], str(tmp_path / "m2"), plan, records,
                        totals=totals)
    assert not os.path.exists(tmp_path / "m1") and not os.path.exists(tmp_path / "m2")


def test_a_ledger_segment_naming_another_launch_record_than_its_rows_is_refused(tmp_path):
    """A segment's launch record is in no digest; the rows it is bound to name theirs."""
    _engine, summary, out, _stack = _fa_pass(tmp_path)
    me.write_json(_fa_files(out)[1], summary)
    assert me.read_fa_ledger(out, FA_DECLARED, exact=True)
    _edit_json(_fa_files(out)[0],
               lambda ledger: ledger["segments"][0].update(launch_record_sha256="f" * 64))
    with pytest.raises(ValueError, match="which the ledger does not hold"):
        me.read_fa_ledger(out, FA_DECLARED, exact=True)


def test_a_segment_whose_delta_counts_are_outside_its_counts_is_refused(tmp_path):
    """The delta binds a one-ahead state's prefix only if it is inside the counts: a delta row
    the counts do not hold (every self-digest recomputed) is refused by the reader."""
    _engine, summary, out, _stack = _fa_pass(tmp_path)
    me.write_json(_fa_files(out)[1], summary)
    phantom = {"branch": "source_cache", "configured": 3, "cap": 2, "base": [2],
               "orbit": [2, 2, 2], "orbit_size": 4, "count": 1}
    ledger = _ledger(out)
    ledger["segments"][0]["delta_counts"] = me.FaPartitionRecorder.from_counts(
        ledger["segments"][0]["delta_counts"] + [phantom]).to_counts()
    me.write_json(_fa_files(out)[0], _resummed(ledger))
    for exact in (False, True):
        with pytest.raises(ValueError, match="delta_counts are not inside its counts"):
            me.read_fa_ledger(out, FA_DECLARED, exact=exact)


# --------------------------------------------------------------------------- #
# exp_22 rB-F4 (Codex code review rB-F3, HIGH 1): the delete-the-check proof made
# exhaustive. Every atomic refusal / verdict condition the rB-F3 production diff added or
# changed (383a1ab5..701926e5) gets a targeted negative asserting THAT atom's OWN named
# reason, built -- where feasible -- so that the atom is the only clause the input violates.
# The coverage table and the run that deletes each atom in turn:
# loc_meshgrid_<ts>_r2bf4_delete_the_check_mutations.log. Part 1: the re-verifier's clauses.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", [-1.0, False], ids=["negative", "bool"])
def test_a_negative_or_boolean_delta_is_never_within_a_registered_tolerance(tmp_path, monkeypatch,
                                                                            clean_env, value):
    """``pair_reasons`` under a tolerance: ``found <= tolerance`` alone admits a negative
    |diff| and a bool (Python orders ``False`` as 0) -- the measured-number conjunct refuses
    both. The fixture's |diff|s are all 0.0, so neither the per-key fold (a max from 0.0) nor
    the chunk headline moves: the tensor clause is the only one the edit violates."""
    facts = {"shape": [[2], [2]], "dtype": ["torch.float32"] * 2, "finite": True,
             "max_abs_diff": value}
    for tolerance in (1e-3, 0.0):
        assert fp.pair_reasons(facts, tolerance) == [f"max |diff| {fp._fmt(value)} > {tolerance}"]
    for gate, path in (("G3", ["queries", 0, "chunks", 0, "conditioning", "source", "tensor",
                               "max_abs_diff"]),
                       ("G1", ["chunks", 0, "conditioning", "source", "tensor", "max_abs_diff"])):
        (tmp_path / gate).mkdir()
        evidence = _real_evidence(tmp_path / gate, monkeypatch, gate)
        needle = f"source.tensor: max |diff| {fp._fmt(value)} > {fp.GATE_TOLERANCES[gate]}"
        reasons = fp.reverify_gate_evidence(_tamper(evidence, path, value))
        assert any(needle in reason for reason in reasons), (gate, reasons)


def test_gate0_fails_on_ordered_query_ids_that_are_not_the_registered_content(tmp_path,
                                                                             monkeypatch):
    """``_content_reasons``' ordered-ids comparison, which only G0's LIVE replay applies (its
    re-verification passes ``None``): the bytes pin and the frozen map are the registered
    ones, only the ordered ids' registration moved -- and that is the gate's one reason."""
    out, binding, frozen, cases = _published_vanilla_run(tmp_path)
    anchor = _fake_anchor(tmp_path, monkeypatch, frozen)
    monkeypatch.setattr(fp, "COMMITTED_PROBE_QUERY_IDS_SHA256", "0" * 64)
    report = fp.vanilla_replay_regression(
        ENG.SyntheticEngine(), cases, out, anchor=anchor, observation_bank_sha256="o" * 64,
        expected_observation_bank_sha256="o" * 64, run_binding=binding,
        expected_queries=_ids(cases), **G0_KW)
    assert report["verdict"] == "FAIL"
    assert report["reasons"] == ["the frozen anchor: its ordered query ids are not the registered "
                                 "content of the pinned probe report"]


AUTOCAST_COMPONENTS = [
    # (component, an edit of the resolved state that violates ONLY that component)
    ("requested", lambda s: {"cond_autocast_requested": {"off": "default", "default": "off"}[
        s["cond_autocast_requested"]]}),
    ("enabled_flipped", lambda s: {"cond_autocast_enabled": not s["cond_autocast_enabled"]}),
    ("enabled_an_int", lambda s: {"cond_autocast_enabled": int(s["cond_autocast_enabled"])}),
    ("dtype", lambda s: {"cond_autocast_dtype": "float64"}),
    ("device_type", lambda s: {"cond_autocast_device_type": {"cpu": "cuda", "cuda": "cpu"}[
        s["cond_autocast_device_type"]]}),
    ("unregistered_device_null_state", lambda s: {
        "device": "mps:0", "cond_autocast_enabled": None, "cond_autocast_dtype": None,
        "cond_autocast_device_type": "mps"}),
    ("no_device_null_state", lambda s: {
        "device": None, "cond_autocast_enabled": None, "cond_autocast_dtype": None,
        "cond_autocast_device_type": None}),
    ("device_not_a_string", lambda s: {"device": 7}),
]


@pytest.mark.parametrize("component, edit", AUTOCAST_COMPONENTS,
                         ids=[c[0] for c in AUTOCAST_COMPONENTS])
@pytest.mark.parametrize("gate", ["G1", "G3", "G3D"])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_each_autocast_component_alone_is_refused_by_the_autocast_clause(device, gate,
                                                                         component, edit):
    """``autocast_reasons``, one conjunct at a time: the token, a real ``enabled`` bool of the
    resolved value, the resolved dtype, the device type of ``inputs.device``, a REGISTERED
    device type (an unregistered one -- or none -- resolves to no state, so even a published
    null state is refused) and a string device (anything else is named, never raised)."""
    state = _autocast(gate, device)
    assert fp.autocast_reasons(gate, state) == []
    reasons = fp.autocast_reasons(gate, dict(state, **edit(state)))
    assert len(reasons) == 1 and reasons[0].startswith(
        f"autocast: {fp.GATE_NAMES[gate]} is registered at --cond-autocast"), reasons


def test_the_live_drift_measurement_names_a_score_delta_that_is_not_a_measurement(monkeypatch):
    """``_score_deltas``' measurement branch: ``nested_scores`` and ``argmax_by_global_index``
    refuse non-finite scores, so a live delta is a finite measurement unless two FINITE score
    vectors overflow float64 -- this defence in depth is observed by simulating that upstream
    defect (production scores +1e308, replay scores -1e308: the |difference| is inf)."""
    real, calls = me.nested_scores, []

    def defective(sims, tau, prefixes):
        found = real(sims, tau=tau, prefixes=prefixes)
        calls.append(1)
        sign = 1.0 if len(calls) % 2 else -1.0
        return {k: dict(v, scores=v["scores"].double() * 0.0 + sign * 1e308)
                for k, v in found.items()}

    sims = torch.rand(3, 4, generator=torch.Generator().manual_seed(0))
    assert fp._score_deltas(sims, sims.clone(), [0, 1, 2], me.TAU, (1, 4), enforce=False)[1] == []
    monkeypatch.setattr(me, "nested_scores", defective)
    out, why = fp._score_deltas(sims, sims.clone(), [0, 1, 2], me.TAU, (1, 4), enforce=False)
    assert why == [f"K={k}: score |diff| inf is not a finite number >= 0" for k in (1, 4)], why
    assert all(isinstance(out[str(k)][name], bool) for k in (1, 4)
               for name in ("argmax_equal", "at_risk"))


@pytest.mark.parametrize("flag", ["argmax_equal", "at_risk"])
def test_a_drift_measurement_agreement_flag_that_is_not_a_boolean_is_named(tmp_path, monkeypatch,
                                                                           clean_env, flag):
    """``measurement_reasons``, one flag at a time -- live and re-verified. The re-verified edit
    publishes the SAME agreement as an int (``True`` -> 1, ``False`` -> 0), so the per-K summary
    (which counts ``int(flag)``) is unchanged: only this clause can name it."""
    facts = {"max_abs_delta": 0.5, "argmax_equal": True, "at_risk": False}
    assert fp.measurement_reasons("8", facts) == []
    assert fp.measurement_reasons("8", dict(facts, **{flag: int(facts[flag])})) == [
        f"K=8: {flag} is not a boolean"]
    _code, evidence, _anchor = _registered_gate3(tmp_path, monkeypatch, "G3D", conditioner=Noisy())
    original = evidence["queries"][0]["score_stability"]["4"][flag]
    reasons = fp.reverify_gate_evidence(
        _tamper(evidence, ["queries", 0, "score_stability", "4", flag], int(original)))
    assert any(reason.endswith(f"K=4: {flag} is not a boolean") for reason in reasons), reasons
    assert not any(reason.startswith("score_stability") for reason in reasons), reasons


@pytest.mark.parametrize("moved", ["the_ordered_ids", "the_frozen_map"])
def test_the_loader_holds_the_ordered_ids_and_the_frozen_map_each_to_its_registration(
        tmp_path, monkeypatch, moved):
    """``load_probe_anchor``'s content check is two comparisons: the ordered ids (the committed
    records REVERSED keep the frozen map, which is keyed by id) and the frozen map (one row
    digest moved keeps the ordered ids) -- each input violates exactly one of them."""
    import hashlib

    records = json.load(open(PROBE_REPORT))["records"]
    records = (list(reversed(records)) if moved == "the_ordered_ids"
               else [dict(records[0], grid_row_sha256="0" * 64)] + records[1:])
    ids = [str(record["query_id"]) for record in records]
    frozen = {str(record["query_id"]): {"row_sha256": record["grid_row_sha256"],
                                        "sims_sha256": record["grid_sims_sha256"]}
              for record in records}
    assert (canonical_sha256(ids) == fp.COMMITTED_PROBE_QUERY_IDS_SHA256) is (
        moved == "the_frozen_map")
    assert (canonical_sha256(frozen) == fp.COMMITTED_PROBE_FROZEN_DIGESTS_SHA256) is (
        moved == "the_ordered_ids")
    path = tmp_path / "relocated.json"
    path.write_text(json.dumps({"records": records}))
    monkeypatch.setattr(fp, "COMMITTED_PROBE_REPORT_SHA256",
                        hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="is not the registered content of the pinned report"):
        fp.load_probe_anchor(str(path))


@pytest.mark.parametrize("error", ["ZeroDivisionError", "RecursionError"])
def test_an_arithmetic_or_recursion_error_in_the_evidence_is_named_never_raised(
        tmp_path, monkeypatch, clean_env, error):
    """``MALFORMED`` names an ArithmeticError (a bound num_samples of 0.5 divides the registered
    256 rows by int(0.5) = 0) and a RecursionError (a frozen digest nested 100,000 deep, which
    the canonical digest recurses into) instead of letting either escape."""
    if error == "ZeroDivisionError":
        evidence = _tamper(_real_evidence(tmp_path, monkeypatch, "G3"),
                           ["inputs", "binding", "num_samples"], 0.5)
        reasons = fp.reverify_gate_evidence(evidence)
    else:
        evidence = copy.deepcopy(_real_evidence(tmp_path, monkeypatch, "G0"))
        deep = []
        for _ in range(100000):
            deep = [deep]
        evidence["queries"][0]["frozen_digests"] = deep
        reasons = fp.reverify_gate_evidence(evidence)
        evidence["queries"][0]["frozen_digests"] = deep = None     # never in a failure's repr
    assert any(reason.startswith(f"comparison: the evidence is malformed ({error}")
               for reason in reasons), reasons


def _one_sided():
    """A DiT input compared on one side only, as ``compare_inputs`` serializes it."""
    return {"bitwise": False, "max_abs_diff": float("inf"), "finite": True,
            "reasons": ["present on one side only"]}


@pytest.mark.parametrize("value, needle", [
    ("a string", "dit_inputs: the inputs the DiT consumes were not compared"),
    (7, "dit_inputs: the inputs the DiT consumes were not compared"),
    ({}, "dit_inputs: the inputs the DiT consumes were not compared"),
    ("one_sided", "dit_inputs.global_cond: present on one side only"),
], ids=["a_string", "an_int", "empty", "one_sided"])
def test_the_dit_input_clause_names_an_uncompared_or_one_sided_input(tmp_path, monkeypatch,
                                                                     clean_env, value, needle):
    """``inputs_reasons``: a block that is not a dict, an empty block (each: "not compared")
    and a one-sided input (its own reason, not a |diff| one), live and re-verified."""
    _code, evidence, _anchor = _registered_gate3(tmp_path, monkeypatch, "G3")
    facts = evidence["queries"][0]["chunks"][0]["dit_inputs"]
    edited = dict(facts, global_cond=_one_sided()) if value == "one_sided" else value
    assert fp.inputs_reasons(copy.deepcopy(facts), 1e-3) == []
    assert fp.inputs_reasons(copy.deepcopy(edited), 1e-3) == [needle]
    reasons = fp.reverify_gate_evidence(
        _tamper(evidence, ["queries", 0, "chunks", 0, "dit_inputs"], edited))
    assert any(reason.endswith(needle) for reason in reasons), reasons


@pytest.mark.parametrize("part", ["query", "coverage", "score_stability"])
def test_a_malformed_part_is_named_by_its_own_collector_and_hides_no_other_reason(
        tmp_path, monkeypatch, clean_env, part):
    """``_reverify_production`` wraps each part that can raise on a malformed fact in its own
    ``_collect``: a malformed query (its chunks a string), a scored chunk size that is a string
    (``coverage`` orders it) and a score delta that is a string (``score_summary`` maxes it)
    are each named by THEIR collector, and an independent clause (``inputs: tau``) survives."""
    evidence = _tamper(_real_evidence(tmp_path, monkeypatch, "G3"), ["inputs", "tau"], 0.05)
    path, value = {"query": (["queries", 1, "chunks"], "x"),
                   "coverage": (["queries", 1, "chunks", 0, "n"], "32"),
                   "score_stability": (["queries", 0, "score_stability", "4", "max_abs_delta"],
                                       "x")}[part]
    reasons = fp.reverify_gate_evidence(_tamper(evidence, path, value))
    assert any(reason.startswith(f"{part}: the evidence is malformed") for reason in reasons), reasons
    assert any(reason.startswith("inputs: tau") for reason in reasons), reasons


def test_a_query_whose_candidate_index_list_is_not_its_count_does_not_tile(tmp_path, monkeypatch,
                                                                           clean_env):
    """The tiling clause's second disjunct: the chunks still tile ``n_candidates``; only the
    published candidate-index list is one short -- and nothing else reads it."""
    evidence = _real_evidence(tmp_path, monkeypatch, "G3")
    query = evidence["queries"][0]["inputs"]
    reasons = fp.reverify_gate_evidence(_tamper(evidence, ["queries", 0, "inputs",
                                                           "candidate_indices"],
                                                query["candidate_indices"][:-1]))
    assert any(reason.startswith(f"{query['query_id']}: its chunks do not tile")
               for reason in reasons), reasons


def test_a_query_whose_score_stability_is_not_published_for_exactly_the_registered_k(
        tmp_path, monkeypatch, clean_env):
    """The per-query K-set clause: K = 1 dropped from one query AND the published per-K summary
    re-derived to match, so the K-set clause is the only one the edit violates."""
    evidence = copy.deepcopy(_real_evidence(tmp_path, monkeypatch, "G3"))
    evidence["queries"][0]["score_stability"].pop("1")
    evidence["score_stability"] = fp.score_summary(evidence["queries"])
    reasons = fp.reverify_gate_evidence(evidence)
    qid = evidence["queries"][0]["inputs"]["query_id"]
    assert any(reason.startswith(f"{qid}: its score stability is not published for exactly K")
               for reason in reasons), reasons
    assert not any(reason.startswith("score_stability") for reason in reasons), reasons


def test_a_chunk_headline_that_is_not_its_facts_is_named(tmp_path, monkeypatch, clean_env):
    """The chunk-headline clause: only the published headline moved -- the per-key facts it
    summarizes (and so the per-key fold, the chains and the coverage) are untouched."""
    evidence = _real_evidence(tmp_path, monkeypatch, "G3")
    reasons = fp.reverify_gate_evidence(
        _tamper(evidence, ["queries", 0, "chunks", 0, "max_abs_diff"], 0.5))
    assert any(reason.endswith("chunk@0: its max |diff| / bitwise headline is not its facts'")
               for reason in reasons), reasons


def test_the_coverage_clause_requires_both_a_full_and_a_tail_batch():
    """``_coverage_reasons``' completeness clause, alone: the published coverage IS the one the
    chunks yield, and it has no tail batch."""
    full_only = fp.coverage([4, 4, 4], 4)
    assert full_only == {"full": 3, "tail": 0, "complete": False}
    covered = {"scored": full_only, "source": fp.coverage([3, 1], 3)}
    assert fp._coverage_reasons({"coverage": covered}, covered) == [
        "coverage: scored batches 3 full / 0 tail; a gate covers both"]
    whole = {"scored": fp.coverage([4, 2], 4), "source": fp.coverage([3, 1], 3)}
    assert fp._coverage_reasons({"coverage": whole}, whole) == []


@pytest.mark.parametrize("gate", ["G1", "G2", "G3", "G3D", "G0", "GV"])
def test_every_schema_entry_names_its_own_substitution_deletion_and_extra(tmp_path, monkeypatch,
                                                                          clean_env, gate):
    """The substitution sweep above proves "never raised, always a reason"; this one proves each
    EVIDENCE_SCHEMA entry's OWN reason. Every top-level and inputs field of a real evidence,
    replaced by each value of another JSON type (by the TEST's ``_kind``, never the module's),
    deleted, and joined by an unregistered field: ``schema_reasons`` names exactly that. A
    non-dict ``inputs`` is named once, at the top level; the entry point returns the schema's
    reasons."""
    evidence = json.loads(json.dumps(_real_evidence(tmp_path, monkeypatch, gate)))
    assert fp.schema_reasons(gate, evidence) == []
    top, inputs = fp.EVIDENCE_SCHEMA[gate]
    missed = []
    for where, schema in (("evidence", top), ("inputs", inputs)):
        prefix = [] if where == "evidence" else ["inputs"]
        for field, kind in schema.items():
            for value in [substitute for substitute in SUBSTITUTES if _kind(substitute) != kind]:
                wanted = f"schema: {where}.{field} is {type(value).__name__}, not {kind}"
                if wanted not in fp.schema_reasons(gate, _tamper(evidence, prefix + [field], value)):
                    missed.append(wanted)
            if f"schema: {where} lacks ['{field}']" not in fp.schema_reasons(
                    gate, _tamper(evidence, prefix + [field], delete=True)):
                missed.append(f"{where}.{field} deleted")
        if f"schema: {where} carries unregistered fields ['smuggled']" not in fp.schema_reasons(
                gate, _tamper(evidence, prefix + ["smuggled"], 1)):
            missed.append(f"{where}.smuggled")
    assert missed == []
    for value in ("x", 7, [], None):
        reasons = fp.schema_reasons(gate, _tamper(evidence, ["inputs"], value))
        assert f"schema: evidence.inputs is {type(value).__name__}, not dict" in reasons
        assert not any(reason.startswith("schema: inputs") for reason in reasons), reasons
    assert "schema: inputs carries unregistered fields ['smuggled']" in fp.reverify_gate_evidence(
        _tamper(evidence, ["inputs", "smuggled"], 1))


def test_an_unknown_gate_is_named_by_the_gate_clause_alone():
    """``reverify_gate_evidence``'s gate clause, second disjunct: a STRING that is not a
    re-verifiable gate is named by it, and nothing else runs (the first disjunct, a non-string
    gate, is ``test_a_malformed_evidence_is_named_never_raised``)."""
    for gate in ("G9", "g3", ""):
        assert fp.reverify_gate_evidence({"gate": gate}) == [
            f"gate: {gate!r} is not one of the re-verifiable gates {sorted(fp.REVERIFIERS)}"]


def test_a_finalized_evidence_carries_the_device_its_autocast_clause_resolves_on(tmp_path):
    """``finalize_evidence`` publishes the device the gate ran on as ``inputs.device`` when its
    report and its inputs name none -- the autocast clause resolves the registered token ON it,
    so without it no evidence could PASS."""
    report = _g1_report(tmp_path)
    assert "device" not in report["inputs"] and "device" not in _g1_inputs()
    evidence = fp.finalize_evidence(report, device="cpu", environment=CLEAN_ENVIRONMENT,
                                    inputs=_g1_inputs())
    assert evidence["inputs"]["device"] == "cpu"
    assert evidence["verdict_clauses"]["autocast"] is True and evidence["verdict"] == "PASS"


# --------------------------------------------------------------------------- #
# rB-F4 part 2: the executed-partition ledger -- THE counts validator, each segment and its
# delta, the accumulate-only rewrite, the adopted handle, freshness, and the reader's and the
# census's disjuncts, each atom alone
# --------------------------------------------------------------------------- #
from src.localization import meshgrid_segments as ms                     # noqa: E402


def _canonical_counts():
    """Canonical counts rows over three branches, orbit rows included."""
    return _observe_production(me.FaPartitionRecorder(), scored=(32, 8)).to_counts()


def _with(counts, index, **fields):
    edited = copy.deepcopy(counts)
    edited[index].update(fields)
    return edited


def _orbit_row(counts):
    return next(index for index, entry in enumerate(counts) if entry["orbit"])


_NOT_ROWS, _NOT_INT = "x's counts are not", "x holds a count that is not a non-negative integer"
COUNT_MALFORMATIONS = [
    # (id, an edit of the canonical counts, the validator's OWN reason (its start))
    ("not_a_list_but_a_dict", lambda c: {}, _NOT_ROWS),
    ("not_a_list_but_null", lambda c: None, _NOT_ROWS),
    ("an_entry_not_a_dict", lambda c: [list(me.LEDGER_ENTRY_FIELDS)] + c[1:], _NOT_ROWS),
    ("an_extra_field", lambda c: _with(c, 0, extra=1), _NOT_ROWS),
    ("a_missing_field", lambda c: [{k: v for k, v in c[0].items() if k != "cap"}] + c[1:],
     _NOT_ROWS),
    ("base_not_a_list", lambda c: _with(c, 0, base=7), _NOT_ROWS),
    ("orbit_not_a_list", lambda c: _with(c, 0, orbit=7), _NOT_ROWS),
    # the same VALUE as a float / bool passes the canonical comparison (from_counts coerces
    # with int(), and 1 == 1.0 == True): only the real-int clause of its family refuses it
    ("configured_a_float", lambda c: _with(c, 0, configured=float(c[0]["configured"])), _NOT_INT),
    ("cap_a_float", lambda c: _with(c, 0, cap=float(c[0]["cap"])), _NOT_INT),
    ("orbit_size_a_float", lambda c: _with(c, 0, orbit_size=float(c[0]["orbit_size"])), _NOT_INT),
    ("count_a_bool", lambda c: _with(c, 0, count=True), _NOT_INT),
    ("a_base_element_a_float", lambda c: _with(c, 0, base=[float(n) for n in c[0]["base"]]),
     _NOT_INT),
    ("an_orbit_element_a_float", lambda c: _with(
        c, _orbit_row(c), orbit=[float(n) for n in c[_orbit_row(c)]["orbit"]]), _NOT_INT),
    ("a_zero_count", lambda c: _with(c, 0, count=0), "x holds a zero count"),
    ("an_unknown_branch", lambda c: _with(c, 0, branch="nowhere"),
     "x: a ledger entry names branch 'nowhere'"),
    ("rows_reversed", lambda c: list(reversed(c)), "x's counts are not canonical"),
    ("a_row_repeated", lambda c: c + [dict(c[0])], "x's counts are not canonical"),
]


@pytest.mark.parametrize("name, edit, needle", COUNT_MALFORMATIONS,
                         ids=[m[0] for m in COUNT_MALFORMATIONS])
def test_the_counts_validator_names_each_malformation_by_its_own_reason(name, edit, needle):
    """``counts_reasons`` -- THE validator of segment counts, deltas and gate executed counts --
    one conjunct at a time, directly: a list of dict rows of exactly the seven fields, base /
    orbit lists, every value family a real non-negative int, a count >= 1, a known branch, the
    canonical order. Each is named by its own reason, never raised."""
    counts = _canonical_counts()
    assert len(counts) >= 2 and me.counts_reasons(copy.deepcopy(counts), "x")[0] == []
    why, recorder = me.counts_reasons(edit(copy.deepcopy(counts)), "x")
    assert len(why) == 1 and why[0].startswith(needle), why
    assert (recorder is None) is (needle != "x's counts are not canonical")


@pytest.mark.parametrize("exact", [False, True], ids=["reopen", "merge"])
def test_a_segment_query_count_that_is_not_a_real_integer_is_refused(tmp_path, exact):
    """``_segment_reasons``' query count: n_queries 4.0 for 4 binds exactly like 4
    (``4.0 in (4, 5)``), so only the real-int clause can refuse it."""
    _engine, summary, out, _stack = _fa_pass(tmp_path)
    me.write_json(_fa_files(out)[1], summary)
    n = _ledger(out)["segments"][0]["n_queries"]
    _edit_json(_fa_files(out)[0],
               lambda ledger: ledger["segments"][0].update(n_queries=float(n)))
    with pytest.raises(ValueError, match=re_escape(
            f"segment 0's n_queries {float(n)!r} is not a non-negative integer")):
        me.read_fa_ledger(out, FA_DECLARED, exact=exact)


DELTA_MALFORMATIONS = [
    ("rows_reversed", lambda d: list(reversed(d)), "segment 0's delta's counts are not canonical"),
    ("missing", None, "segment 0's delta's counts are not"),
    ("a_bool_count", lambda d: [dict(d[0], count=True)] + d[1:],
     "segment 0's delta holds a count that is not a non-negative integer"),
    ("an_unknown_branch", lambda d: [dict(d[0], branch="nowhere")] + d[1:],
     "segment 0's delta: a ledger entry names branch 'nowhere'"),
]


@pytest.mark.parametrize("name, edit, needle", DELTA_MALFORMATIONS,
                         ids=[m[0] for m in DELTA_MALFORMATIONS])
def test_a_segment_delta_that_is_not_canonical_counts_is_refused_by_its_own_reason(
        tmp_path, name, edit, needle):
    """``_segment_reasons``' delta clauses: the delta is held to THE counts validator, and the
    inside-the-counts check runs only on a valid delta (a malformed one is named, never raised).
    A reversed or a bool-count delta keeps every subtraction -- so every binding -- intact."""
    _engine, _summary, out, _stack = _fa_pass(tmp_path)
    ledger = _ledger(out)
    segment = ledger["segments"][0]
    assert len(segment["delta_counts"]) >= 2 and segment["delta_counts"][0]["count"] == 1
    if edit is None:
        segment.pop("delta_counts")
    else:
        segment["delta_counts"] = edit(segment["delta_counts"])
    me.write_json(_fa_files(out)[0], ledger)
    with pytest.raises(ValueError, match=re_escape(needle)):
        me.read_fa_ledger(out, FA_DECLARED)


def test_the_ledger_only_ever_accumulates(tmp_path):
    """``persist_fa_ledger`` refuses a recorder holding fewer calls than the segment it would
    overwrite already published (its delta would be negative)."""
    ledger = me.open_fa_ledger(str(tmp_path / "run"), FA_DECLARED)
    me.persist_fa_ledger(ledger, _observe_production(me.FaPartitionRecorder(), scored=(2, 2)))
    with pytest.raises(ValueError, match="only ever accumulates"):
        me.persist_fa_ledger(ledger, me.FaPartitionRecorder())


@pytest.mark.parametrize("ran", ["a_query", "a_call"])
def test_a_ledger_handle_whose_segment_already_ran_is_not_adopted(tmp_path, ran):
    """``adopt_fa_ledger``'s fresh-segment conjuncts, one at a time: a handle whose segment holds
    a query (and no call) or a call (and no query) is not the one just opened for this pass."""
    out = str(tmp_path / "run")
    ledger = me.open_fa_ledger(out, FA_DECLARED, None)
    assert me.adopt_fa_ledger(ledger, out, FA_DECLARED, None) is ledger
    ledger["payload"]["segments"][-1].update(
        **({"n_queries": 1} if ran == "a_query" else {"counts": _canonical_counts()}))
    with pytest.raises(ValueError, match="not the one just opened"):
        me.adopt_fa_ledger(ledger, out, FA_DECLARED, None)


@pytest.mark.parametrize("state", ["a_regular_file", "a_json_named_directory"])
def test_an_out_dir_that_is_a_file_or_holds_a_json_named_directory_is_not_fresh(tmp_path, state):
    """``fresh_out_dir``: an out-dir that is not a directory, and a sub-DIRECTORY whose name ends
    in .json (only top-level json FILES can be emitted launch records)."""
    out = str(tmp_path / "shard")
    if state == "a_regular_file":
        with open(out, "w") as handle:
            handle.write("x")
    else:
        os.makedirs(os.path.join(out, "stale.json"))
    assert me.fresh_out_dir(out)[0] is False
    with pytest.raises(ValueError, match="not a fresh out-dir"):
        me.read_fa_ledger(out, FA_DECLARED)


@pytest.mark.parametrize("name", [me.BINDING_FILENAME, me.RUN_SUMMARY_FILENAME,
                                  me.MERGE_REPORT_FILENAME, me.FA_LEDGER_FILENAME,
                                  me.FA_FAILURE_FILENAME, me.FA_PENDING_FILENAME,
                                  ms.SEGMENTS_FILENAME])
def test_a_file_a_pass_writes_is_never_taken_for_an_emitted_launch_record(tmp_path, name):
    """``fresh_out_dir``'s ``written`` names, one at a time and directly. The ledger, the failure
    marker and the sentinel are unreachable through the reader (a ledger takes the reading
    path; a marker or a sentinel is refused first), so only this test observes those three."""
    out = tmp_path / "shard"
    out.mkdir()
    me.write_json(str(out / "launch_record.json"), {"argv": []})
    assert me.fresh_out_dir(str(out)) == (True, [])
    me.write_json(str(out / name), {})
    assert me.fresh_out_dir(str(out)) == (False, [name])


def test_a_vanilla_run_carrying_an_fa_summary_block_is_refused(tmp_path):
    """``read_fa_ledger``'s vanilla clause, the summary disjunct: no ledger file, no receipt --
    only a run summary carrying an executed-partition block (a null one is admitted)."""
    out = str(tmp_path / "vanilla")
    os.makedirs(out)
    me.write_json(os.path.join(out, me.RUN_SUMMARY_FILENAME), {"fa_partition_executed": {}})
    with pytest.raises(ValueError, match="a vanilla run carries no frame-average executed"):
        me.read_fa_ledger(out, None)
    me.write_json(os.path.join(out, me.RUN_SUMMARY_FILENAME), {"fa_partition_executed": None})
    assert me.read_fa_ledger(out, None) is None


def test_a_merge_or_report_reading_no_ledger_refuses_even_a_fresh_directory(tmp_path):
    """``read_fa_ledger``'s missing-ledger clause, the ``exact`` disjunct: every directory a
    merge or a report reads holds a binding, so it is never fresh -- this disjunct is defence in
    depth, observed on an absent and on an empty directory (where a reopen may START)."""
    for out in (str(tmp_path / "absent"), str(tmp_path / "empty")):
        if out.endswith("empty"):
            os.makedirs(out)
        assert me.read_fa_ledger(out, FA_DECLARED) is None
        with pytest.raises(ValueError, match="publishes no fa_partition_ledger.json"):
            me.read_fa_ledger(out, FA_DECLARED, exact=True)


def test_a_summary_block_without_its_ledger_is_refused_even_if_freshness_were_misjudged(
        tmp_path, monkeypatch):
    """... the summary disjunct: a run summary makes a directory non-fresh, so ``not fresh``
    already refuses this input -- the summary disjunct is defence in depth, observed by
    simulating a freshness defect."""
    _engine, summary, out, _stack = _fa_pass(tmp_path)
    me.write_json(_fa_files(out)[1], summary)
    os.remove(_fa_files(out)[0])
    monkeypatch.setattr(me, "fresh_out_dir", lambda out_dir: (True, []))
    with pytest.raises(ValueError, match="publishes no fa_partition_ledger.json"):
        me.read_fa_ledger(out, FA_DECLARED)


def test_a_shard_that_executed_nothing_is_refused_at_a_merge_by_its_partition_check(tmp_path):
    """``read_fa_ledger``'s partition check, the ``exact`` disjunct: nothing ran (one empty
    segment and the summary block it yields) is a consistent prefix at a reopen, but never the
    declared partition at a merge."""
    out = str(tmp_path / "shard")
    me.open_fa_ledger(out, FA_DECLARED)
    me.write_json(os.path.join(out, me.RUN_SUMMARY_FILENAME), {
        "fa_partition_executed": me.fa_executed_block(FA_DECLARED, me.FaPartitionRecorder(), 1)})
    assert me.read_fa_ledger(out, FA_DECLARED)["n_segments"] == 1
    with pytest.raises(ValueError, match="the executed frame-average partition is not the declared"):
        me.read_fa_ledger(out, FA_DECLARED, exact=True)


def test_a_ledger_whose_declaration_was_rewritten_is_refused_at_reopen_by_what_it_executed(
        tmp_path):
    """``read_fa_ledger``'s partition check, the ran-anything disjunct: at a reopen the executed
    prefix is held to the declaration -- a ledger whose ``declared`` was rewritten together with
    the reader's declaration passes every integrity clause before it."""
    _engine, _summary, out, _stack = _fa_pass(tmp_path)
    other = me.declared_fa_partition(list(C4), batch_rows=8, num_samples=4, source_chunk=5)
    assert other != FA_DECLARED and not os.path.exists(_fa_files(out)[1])
    _edit_json(_fa_files(out)[0], lambda ledger: ledger.update(declared=other))
    with pytest.raises(ValueError, match="the executed frame-average partition is not the declared"):
        me.read_fa_ledger(out, other)


def test_the_census_verifies_the_merged_ledgers_own_integrity_before_any_write(tmp_path,
                                                                              monkeypatch):
    """``fa_partition_census`` holds the merged ledger to ``_ledger_reasons`` in memory (a census
    bug -- simulated: the merged ledger alone refused, every shard's own read intact -- is
    refused before anything is written)."""
    plan, records, dirs, totals = _fa_shards(tmp_path)
    real = me._ledger_reasons

    def merged_only(payload, declared):
        if any("shard" in segment for segment in (payload or {}).get("segments") or []):
            return ["a simulated integrity bug"], None
        return real(payload, declared)

    monkeypatch.setattr(me, "_ledger_reasons", merged_only)
    with pytest.raises(ValueError, match="would not verify under its own reader: a simulated "
                                         "integrity bug"):
        me.merge_shards([dirs["a"], dirs["b"]], str(tmp_path / "merged"), plan, records,
                        totals=totals)
    assert not os.path.exists(tmp_path / "merged")
