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
