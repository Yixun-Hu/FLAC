"""exp_22 R1 -- the mesh-grid localization report (inherited plan §2).

The report is a JOIN: the I1 rows, the G1 candidate geometry, the D1 context
manifest and the dataset's own pair metadata all have to describe the same
5,337 queries before a single number is taken out of them. Almost every test
here is therefore a gate test -- census, digest, protocol, cross-check -- and the
arithmetic tests are deliberately small and hand-checkable.

The fixture is a REAL run: a synthetic generation stack is driven through the
engine's own ``run_pass``, so the rows, the sidecars, the digests and the binding
are produced by exactly the code the production pass uses. No GPU is involved.
"""
import json
import math
import os

import numpy as np
import pytest
import torch

from src.localization import meshgrid_engine as me
from src.localization import meshgrid_geometry as mg
from src.localization import meshgrid_queries as mq
from src.localization import meshgrid_report as mr


# --------------------------------------------------------------------------- #
# the fixture: a two-room subset with a real candidate geometry
# --------------------------------------------------------------------------- #
FIXTURE_LATTICE = np.array([[x, y, z]
                            for x in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
                            for y in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
                            for z in (0.5, 1.0, 1.5)], dtype=np.float64)

_NEAR = [(1.0, 1.0), (1.5, 1.0), (1.0, 1.5), (1.5, 1.5),
         (2.0, 1.0), (2.0, 1.5), (1.0, 2.0), (1.5, 2.0)]
_FAR = [(0.0, 2.5), (0.5, 2.5), (2.5, 0.0), (2.5, 0.5),
        (0.0, 2.0), (0.5, 2.0), (2.0, 2.5), (2.5, 2.0)]


def _ctx(points, z):
    return [[float(x), float(y), float(z)] for x, y in points]


#: ``(position, S<src>_R<rec> stem, receiver, context globals, continuous TRUTH)``.
FIXTURE_QUERIES = {
    "A/A_idx_1": [
        (0, "S001_R002", [0.0, 0.0, 0.5], _ctx(_NEAR, 0.5), [1.1, 1.1, 0.6]),
        (1, "S003_R004", [2.5, 2.5, 1.5], _ctx(_FAR, 0.5), [0.4, 2.4, 0.6]),
        (2, "S005_R002", [0.0, 0.0, 0.5], _ctx(_NEAR, 1.5), [1.6, 1.6, 1.4]),
    ],
    "B/B_idx_2": [
        (3, "S001_R009", [1.0, 1.0, 1.0], _ctx(_FAR, 1.0), [2.4, 0.4, 1.1]),
    ],
}

FIXTURE_TAU = 0.1
FIXTURE_PREFIXES = (1, 4, 8)
FIXTURE_SAMPLES = 8


def fixture_relpath(room_id, name):
    scene, scene_id = room_id.split("/")
    return f"ir/{scene}/{scene_id}/{name}_hybrid_IR.wav"


def fixture_query_id(room_id, entry):
    return f"{entry[0]}|{fixture_relpath(room_id, entry[1])}"


def fixture_receiver_id(room_id, receiver):
    return f"{room_id}|" + ",".join(f"{float(v):.6f}" for v in receiver)


def _write_room(out_dir, room_id, base, branch="z_band"):
    """A verifier-valid room manifest whose queries are filtered, not invented."""
    scene, scene_id = room_id.split("/")
    stem = f"candidates_{scene}_{scene_id}"
    np.savez(os.path.join(out_dir, stem + ".npz"), base_candidates=base)
    payload = {"room_id": room_id, "chosen_branch": branch, "spacing": 0.5,
               "coordinates_npz": stem + ".npz", "n_base_valid": int(base.shape[0]),
               "base_candidates_sha256": mg.coordinates_digest(base),
               "directions_seed": 1, "queries": []}
    for entry in FIXTURE_QUERIES[room_id]:
        position, _name, receiver, contexts, truth = entry
        full = mg.filter_query_candidates(base, receiver=receiver, context_sources=contexts)
        band_bounds = mg.context_z_band(contexts)
        banded = mg.filter_query_candidates(base, receiver=receiver,
                                            context_sources=contexts, z_band=band_bounds)
        full_idx = np.flatnonzero(full["mask"])
        band_idx = np.flatnonzero(banded["mask"])
        payload["queries"].append({
            "position": position, "query_id": fixture_query_id(room_id, entry),
            "receiver": [float(v) for v in receiver],
            "receiver_id": fixture_receiver_id(room_id, receiver),
            "candidate_indices": [int(i) for i in full_idx],
            "n_candidates": int(full_idx.size),
            "candidate_indices_z_band": [int(i) for i in band_idx],
            "n_candidates_z_band": int(band_idx.size),
            "candidate_coordinates_sha256": mg.coordinates_digest(full["candidates"]),
            "n_dropped_receiver": full["n_dropped_receiver"],
            "n_dropped_context": full["n_dropped_context"],
            "n_contexts": len(contexts),
            "z_band": [float(band_bounds[0]), float(band_bounds[1])],
            "oracle": {"full_height": mg.grid_oracle_error(full["candidates"], truth),
                       "z_band": mg.grid_oracle_error(banded["candidates"], truth)}})
    path = os.path.join(out_dir, stem + ".json")
    with open(path, "w") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path, payload


def _fixture_audit(tmp_path, branch="z_band", stamp=None):
    """A verifier-valid G1 audit. ``stamp`` makes a second, DIFFERENT valid one.

    The fixture is deterministic, so rebuilding it byte-for-byte would hash the
    same and could not exercise the artifact-hash join at all; ``stamp`` adds a
    harmless top-level key that ``verify_report_chain`` does not read.
    """
    out_dir = str(tmp_path / "g1")
    os.makedirs(out_dir, exist_ok=True)
    written = {room_id: _write_room(out_dir, room_id, FIXTURE_LATTICE, branch=branch)
               for room_id in sorted(FIXTURE_QUERIES)}
    report = {"experiment": "exp_22 loc_meshgrid G1 geometry audit",
              "n_queries": sum(len(FIXTURE_QUERIES[room]) for room in written),
              "n_rooms": len(written), "status": "accepted", "diagnostics_only": False,
              "branch": {"branch": branch, "n_new_over_threshold": 0},
              "directions_seed": 1, "spacing": 0.5,
              "fixture_stamp": stamp,
              "rooms": {room: {"candidate_manifest": os.path.basename(path),
                               "candidate_manifest_sha256": mg.manifest_json_sha256(payload),
                               "mesh": {"path": f"{room}.obj", "sha256": "0" * 64}}
                        for room, (path, payload) in written.items()}}
    report_path = os.path.join(out_dir, "geometry_audit_report.json")
    with open(report_path, "w") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report_path


def _stream_md(room_id, entry):
    """One loader item, built from the SAME fixture geometry as the manifest."""
    position, name, receiver, contexts, truth = entry
    poses = torch.tensor(np.asarray(contexts, dtype=np.float64)
                         - np.asarray(receiver, dtype=np.float64), dtype=torch.float32)
    source = torch.tensor(np.asarray(truth, dtype=np.float64)
                          - np.asarray(receiver, dtype=np.float64), dtype=torch.float32)
    relpath = fixture_relpath(room_id, name)
    # every context RIR differs from every other one AND varies along time, so
    # the digests, the conditioner and the calibration cosines all bite
    context_audio = (torch.arange(8 * 16, dtype=torch.float32).reshape(8, 1, 16) * 0.01
                     + 0.1 * (position + 1))
    return {"depth": torch.full((2, 4), float(sum(receiver))),
            "context_audio": context_audio.contiguous(),
            "context_poses": poses, "context_poses_vit": poses,
            "source": source, "source_vit": source.unsqueeze(0),
            "relpath": relpath, "path": "AcousticRooms/" + relpath, "idx": position}


def _write_metadata(tmp_path):
    """The dataset pair metadata the report resolves the continuous truth from."""
    root = str(tmp_path / "metadata")
    from src.localization.candidates import parse_ir_filename

    for room_id, entries in FIXTURE_QUERIES.items():
        scene, scene_id = room_id.split("/")
        directory = os.path.join(root, scene, scene_id)
        os.makedirs(directory, exist_ok=True)
        for entry in entries:
            src_node, rec_node = parse_ir_filename(fixture_relpath(room_id, entry[1]))
            with open(os.path.join(directory,
                                   f"S{src_node:03d}_R{rec_node:03d}.json"), "w") as handle:
                json.dump({"src_loc": [float(v) for v in entry[4]],
                           "rec_loc": [float(v) for v in entry[2]]}, handle)
    return root


class FakeConditioner:
    """A deterministic ``MultiConditioner`` stand-in with its ``only_ids`` seam."""

    def __call__(self, batch_metadata, device, only_ids=None):
        out = {}
        for key in me.CONTEXT_COND_IDS + me.SOURCE_COND_IDS:
            if only_ids is not None and key not in only_ids:
                continue
            rows = []
            for md in batch_metadata:
                if key in me.CONTEXT_COND_IDS:
                    seed = float(torch.as_tensor(md["context_audio"]).sum()) + len(key)
                else:
                    seed = (float(torch.as_tensor(md["source"]).sum())
                            + float(torch.as_tensor(md["depth"]).sum()) + len(key))
                rows.append(torch.full((3,), math.sin(seed), dtype=torch.float32))
            out[key] = [torch.stack(rows).to(device),
                        torch.ones(len(batch_metadata), dtype=torch.bool)]
        return out


class SyntheticEngine(me.MeshEngine):
    """A deterministic stand-in for the whole generation + scoring stack."""

    def __init__(self):
        super().__init__(device="cpu", latent_shape=(2, 4), conditioner=FakeConditioner(),
                         cond_inputs_fn=self._cond_inputs, sampler=self._sample,
                         decoder=self._decode, embedder=self._embed, cond_method="vanilla")

    @staticmethod
    def _cond_inputs(conditioning):
        return {"cond": torch.cat([conditioning[key][0] for key in sorted(conditioning)],
                                  dim=-1)}

    @staticmethod
    def _sample(noise, cond):
        return noise + cond["cond"].mean(dim=-1).reshape(-1, 1, 1)

    @staticmethod
    def _decode(latents):
        return latents.reshape(latents.shape[0], 1, -1)

    @staticmethod
    def _embed(wavs):
        flat = wavs.reshape(wavs.shape[0], -1)[:, :4].float()
        return torch.nn.functional.normalize(flat + 1e-3, dim=-1)


FIXTURE_ADVISORY = {"source_chunk": 2, "batch_rows": 8}


def _fixture_binding(plan, context_manifest, **overrides):
    """The fixture's binding, hashed over the fixture's OWN artifacts.

    The r9 fixture used placeholder digests, which meant the artifact-hash join
    could not have failed there even when it was absent (Codex r9 review, finding
    2 names this). These are the real file digests, and the protocol fields are
    the registered constants, so both new gates actually bite in the tests.
    """
    binding = {
        "model_config_sha256": mr.REGISTERED_ARTIFACT_SHA256["model_config_sha256"],
        # the registered P1 arm, so the admissible-arm registry passes (r9d M6)
        "ckpt_sha256": mr.REGISTERED_CKPT_SHA256["P1"],
        "agree_ckpt_sha256": mr.REGISTERED_ARTIFACT_SHA256["agree_ckpt_sha256"],
        "d1_manifest_sha256": me.file_sha256(context_manifest),
        "g1_report_sha256": plan.report_sha256,
        "room_manifest_sha256": {room: me.file_sha256(path)
                                 for room, path in plan.rooms.items()},
        "branch": "z_band", "k_prefixes": list(FIXTURE_PREFIXES),
        "num_samples": FIXTURE_SAMPLES, "tau": FIXTURE_TAU, "seed": me.SEED,
        "noise_policy": me.REGISTERED_NOISE_POLICY, "steps": me.STEPS,
        "cfg_scale": me.CFG_SCALE, "cond_method": "vanilla",
        "scorer_readout": me.SCORER_READOUT, "cond_autocast": "default",
        "dataset_config_sha256":
            mr.REGISTERED_ARTIFACT_SHA256["dataset_config_sha256"],
        "dump_cases_sha256": None,
        "dataset_config": "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json",
    }
    binding.update(overrides)
    return binding


def fixture_totals(plan=None, branch="z_band"):
    """The census the fixture implies -- derived, never hand-counted."""
    pairs = 0
    for room_id in sorted(FIXTURE_QUERIES):
        for entry in FIXTURE_QUERIES[room_id]:
            _position, _name, receiver, contexts, _truth = entry
            band = mg.context_z_band(contexts)
            kept = mg.filter_query_candidates(
                FIXTURE_LATTICE, receiver=receiver, context_sources=contexts,
                z_band=None if branch == "full_height" else band)
            pairs += int(kept["mask"].sum())
    source_rows = None
    if plan is not None:
        source_rows = sum(len(group.union) for room_id in sorted(plan.rooms)
                          for group in me.receiver_groups(me.load_room_plan(plan, room_id)))
    return {"rooms": len(FIXTURE_QUERIES),
            "queries": sum(len(v) for v in FIXTURE_QUERIES.values()),
            "candidate_query_pairs": pairs,
            "source_rows": source_rows,
            "generated_waveforms": pairs * FIXTURE_SAMPLES}


def build_fixture_run(tmp_path, merged=True):
    """A complete, self-consistent fixture: G1 audit, D1 manifest, MERGED run.

    ``merged=True`` (the default) scores each room as its own shard and combines
    them with the engine's own ``merge_shards``, so the fixture run directory is
    a real census-gated merge -- merge report, declared rooms, pinned advisory
    and all. A report that skipped the merge gates could not be caught by a
    hand-assembled fixture, which is exactly what the r9 review found.
    """
    audit_report = _fixture_audit(tmp_path)
    plan = me.load_audit_plan(audit_report)
    entries = sorted(((room_id, entry) for room_id in sorted(FIXTURE_QUERIES)
                      for entry in FIXTURE_QUERIES[room_id]), key=lambda pair: pair[1][0])
    records, items = [], []
    for room_id, entry in entries:
        md = _stream_md(room_id, entry)
        records.append(mq.context_record(md, entry[0], eligible=8))
        items.append((torch.full((1, 1, 16), 0.5 + 0.01 * entry[0]), md))

    manifest_path = str(tmp_path / "d1_context_manifest.json")
    mq.write_manifest(manifest_path, {
        "experiment": "exp_22 loc_meshgrid D1 context manifest (fixture)",
        "protocol": dict(mq.EXP01_LOADER), "context_width": mq.CONTEXT_WIDTH,
        "n_full": len(records), "n_filtered": len(records),
        "filtered_stream_sha256": mq.stream_hash(records),
        "census_verified": False, "records": records,
    }, require_census=False)

    binding = _fixture_binding(plan, manifest_path)
    binding_sha = me.binding_sha256(binding)
    totals = fixture_totals(plan)

    def _score(out_dir, rooms=None):
        me.write_binding(out_dir, binding, advisory=FIXTURE_ADVISORY, declared_rooms=rooms)
        summary = me.run_pass(SyntheticEngine(), items, records, plan, out_dir,
                              tau=FIXTURE_TAU, num_samples=FIXTURE_SAMPLES,
                              prefixes=FIXTURE_PREFIXES, batch_rows=8, source_chunk=2,
                              rooms=rooms, binding_sha256=binding_sha)
        me.write_json(os.path.join(out_dir, "run_summary.json"), summary)
        return out_dir

    if merged:
        shards = [_score(str(tmp_path / f"shard_{me.room_stem(room)}"), rooms=[room])
                  for room in sorted(FIXTURE_QUERIES)]
        run_dir = str(tmp_path / "merged")
        me.merge_shards(shards, run_dir, plan, records, totals=totals)
    else:
        run_dir = _score(str(tmp_path / "run"))
        shards = [run_dir]

    metadata_root = _write_metadata(tmp_path)
    # the PRE-REGISTRATION step, run exactly as the CLI runs it: the digest is
    # computed from the D1 manifest and the tree alone, before any evaluation
    preregistered = mr.compute_metadata_bank_digest(manifest_path, metadata_root,
                                                    require_manifest_census=False)
    return {"run_dir": run_dir, "audit_report": audit_report, "plan": plan,
            "context_manifest": manifest_path, "metadata_root": metadata_root,
            "metadata_bank_sha256": preregistered["metadata_bank_sha256"],
            "binding": binding, "binding_sha256": binding_sha, "merged": merged,
            "shards": shards, "records": records, "items": items, "totals": totals}


def fixture_source_provider(fixture):
    """``query_id -> md['source']`` from the fixture stream -- the injective witness."""
    sources = {}
    for _obs, md in fixture["items"]:
        sources[f"{md['idx']}|{md['relpath']}"] = md["source"]
    return sources.get


def evaluate_fixture(fixture, **kwargs):
    """Evaluate the fixture as a CANONICAL run unless the caller says otherwise."""
    kwargs.setdefault("single_shard", not fixture["merged"])
    kwargs.setdefault("expect_metadata_bank_sha256", fixture["metadata_bank_sha256"])
    kwargs.setdefault("totals", fixture["totals"])
    return mr.evaluate_run(fixture["run_dir"], fixture["audit_report"],
                           fixture["context_manifest"], fixture["metadata_root"],
                           require_manifest_census=False, **kwargs)


def evaluate_unpinned(fixture, **kwargs):
    """Evaluate WITHOUT the pre-registered bank -- explicitly non-canonical."""
    return evaluate_fixture(fixture, expect_metadata_bank_sha256=None,
                            allow_unpinned_metadata_bank=True, **kwargs)


# --------------------------------------------------------------------------- #
# the registered constants
# --------------------------------------------------------------------------- #
def test_the_registered_reporting_settings_are_pinned():
    assert mr.BOOTSTRAP_SEED == 20260825
    assert mr.BOOTSTRAP_N == 10000
    assert mr.BOOTSTRAP_ALPHA == 0.05
    assert mr.SUCCESS_RADII == (0.5, 1.0)
    assert mr.RANDOM_BASELINE_SEEDS == (101, 102, 103, 104, 105)
    assert mr.AGGREGATORS == ("lme", "mean")
    assert (mr.HEADLINE_AGGREGATOR, mr.HEADLINE_K) == ("lme", 8)
    assert mr.ORACLE_THRESHOLD == 0.5
    assert "5,337/16 rooms" in mr.SUBSET_LABEL
    assert "canonical-heading diagnostic only" in mr.SUBSET_LABEL


def test_the_diagnostic_aggregator_is_labelled_as_one():
    assert "HEADLINE" in mr.AGGREGATOR_ROLES["lme"]
    assert "DIAGNOSTIC" in mr.AGGREGATOR_ROLES["mean"]


# --------------------------------------------------------------------------- #
# the gates
# --------------------------------------------------------------------------- #
def test_a_binding_is_recomputed_from_its_own_content(tmp_path):
    fixture = build_fixture_run(tmp_path)
    published, digest = mr.load_published_binding(fixture["run_dir"])
    assert digest == fixture["binding_sha256"]
    assert published["branch"] == "z_band"


def test_an_edited_binding_cannot_vouch_for_itself(tmp_path):
    fixture = build_fixture_run(tmp_path)
    path = os.path.join(fixture["run_dir"], me.BINDING_FILENAME)
    payload = json.load(open(path))
    payload["ckpt_sha256"] = "0" * 64                    # a different checkpoint
    with open(path, "w") as handle:
        json.dump(payload, handle)
    with pytest.raises(ValueError, match="does not match its own content"):
        mr.load_published_binding(fixture["run_dir"])


def test_a_binding_missing_a_registered_field_is_refused(tmp_path):
    fixture = build_fixture_run(tmp_path)
    path = os.path.join(fixture["run_dir"], me.BINDING_FILENAME)
    payload = json.load(open(path))
    payload.pop("cond_autocast")
    with open(path, "w") as handle:
        json.dump(payload, handle)
    with pytest.raises(ValueError, match="cond_autocast"):
        mr.load_published_binding(fixture["run_dir"])


def test_a_tampered_row_is_refused_not_aggregated(tmp_path):
    fixture = build_fixture_run(tmp_path)
    path = me.query_artifact_paths(fixture["run_dir"], "A/A_idx_1", 0)["row"]
    row = json.load(open(path))
    row["e_oracle"] = 0.0                                # a better-looking oracle
    with open(path, "w") as handle:
        json.dump(row, handle)
    with pytest.raises(ValueError, match="do not re-verify"):
        mr.verify_rows(fixture["run_dir"], fixture["binding_sha256"])


def test_a_tampered_similarity_sidecar_is_refused(tmp_path):
    fixture = build_fixture_run(tmp_path)
    sims_path = me.query_artifact_paths(fixture["run_dir"], "A/A_idx_1", 0)["sims"]
    array = np.load(sims_path)
    array[0, 0] = np.float16(0.999)
    np.save(sims_path, array)
    with pytest.raises(ValueError, match="do not re-verify"):
        mr.verify_rows(fixture["run_dir"], fixture["binding_sha256"])


def test_rows_from_another_binding_are_never_reported(tmp_path):
    fixture = build_fixture_run(tmp_path)
    with pytest.raises(ValueError, match="do not re-verify"):
        mr.verify_rows(fixture["run_dir"], "0" * 64)


def _census_rows(fixture):
    return [{"query_id": record["query_id"], "room_id": record["room_id"],
             "position": record["position"], "n_candidates": 3, "num_samples": 8}
            for record in fixture["records"]]


def test_the_census_refuses_a_missing_query(tmp_path):
    fixture = build_fixture_run(tmp_path)
    rows = _census_rows(fixture)
    dropped = rows.pop(1)
    with pytest.raises(ValueError, match="have no published row"):
        mr.assert_census(rows, fixture["records"], totals={"queries": len(rows)})
    assert dropped["query_id"]


def test_the_census_refuses_a_duplicate_query(tmp_path):
    fixture = build_fixture_run(tmp_path)
    rows = _census_rows(fixture)
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="published more than once"):
        mr.assert_census(rows, fixture["records"], totals=None)


def test_the_census_refuses_a_query_outside_the_subset(tmp_path):
    fixture = build_fixture_run(tmp_path)
    rows = _census_rows(fixture)
    rows.append(dict(rows[0], query_id="99|ir/Z/Z_idx_9/S001_R001_hybrid_IR.wav",
                     room_id="Z/Z_idx_9", position=99))
    with pytest.raises(ValueError, match="not in the registered subset"):
        mr.assert_census(rows, fixture["records"], totals=None)


def test_the_census_refuses_a_row_filed_under_the_wrong_room(tmp_path):
    fixture = build_fixture_run(tmp_path)
    rows = _census_rows(fixture)
    rows[0]["room_id"] = "B/B_idx_2"
    with pytest.raises(ValueError, match="but the context manifest registers"):
        mr.assert_census(rows, fixture["records"], totals=None)


def test_the_census_refuses_the_wrong_query_room_or_pair_totals(tmp_path):
    fixture = build_fixture_run(tmp_path)
    rows = _census_rows(fixture)
    with pytest.raises(ValueError, match="registered census is 5,337"):
        mr.assert_census(rows, fixture["records"], totals={"queries": 5337})
    with pytest.raises(ValueError, match="registered census is 16"):
        mr.assert_census(rows, fixture["records"], totals={"rooms": 16})
    with pytest.raises(ValueError, match="candidate_query_pairs"):
        mr.assert_census(rows, fixture["records"],
                         totals={"candidate_query_pairs": 8896540})


def test_the_census_passes_on_the_real_fixture_run(tmp_path):
    fixture = build_fixture_run(tmp_path)
    rows = mr.verify_rows(fixture["run_dir"], fixture["binding_sha256"])
    census = mr.assert_census(rows, fixture["records"], totals=fixture["totals"])
    assert census["n_queries"] == fixture["totals"]["queries"]
    assert census["n_rooms"] == 2
    assert census["candidate_query_pairs"] == fixture["totals"]["candidate_query_pairs"]
    assert census["excluded_room"] == mq.EXCLUDED_ROOM


def test_a_row_produced_under_a_different_protocol_is_refused(tmp_path):
    fixture = build_fixture_run(tmp_path)
    rows = mr.verify_rows(fixture["run_dir"], fixture["binding_sha256"])
    with pytest.raises(ValueError, match="tau"):
        mr.assert_row_protocol(rows, dict(fixture["binding"], tau=0.2))
    with pytest.raises(ValueError, match="noise_policy"):
        mr.assert_row_protocol(rows, dict(fixture["binding"],
                                          noise_policy="per_candidate"))
    assert mr.assert_row_protocol(rows, fixture["binding"])["tau"] == FIXTURE_TAU


# --------------------------------------------------------------------------- #
# ground truth
# --------------------------------------------------------------------------- #
def test_the_truth_resolver_reads_the_pair_metadata_the_audit_reads(tmp_path):
    import importlib.util

    fixture = build_fixture_run(tmp_path)
    spec = importlib.util.spec_from_file_location(
        "_agm", "worklog/worklog_yixun/exp_22_loc_meshgrid_claude/audit_meshgrid_geometry.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)

    resolver = mr.TruthResolver(fixture["metadata_root"])
    for record in fixture["records"]:
        # the same seam, resolved two ways: the report's and the G1 audit tool's
        assert resolver.resolve(record)[0].tolist() == \
            audit._metadata_for(record, fixture["metadata_root"])[0].tolist()
        assert resolver.resolve(record)[1].tolist() == \
            audit._metadata_for(record, fixture["metadata_root"])[1].tolist()


def test_a_receiver_that_is_not_the_manifests_is_refused():
    assert mr.assert_receiver_matches("q", [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    with pytest.raises(ValueError, match="is not resolving the same query"):
        mr.assert_receiver_matches("q", [1.0, 2.0, 3.0], [1.0, 2.0, 3.5])


def test_a_truth_whose_oracle_disagrees_with_g1_is_refused(tmp_path):
    fixture = build_fixture_run(tmp_path)
    _position, _name, receiver, contexts, truth = FIXTURE_QUERIES["A/A_idx_1"][0]
    banded = mg.filter_query_candidates(FIXTURE_LATTICE, receiver=receiver,
                                        context_sources=contexts,
                                        z_band=mg.context_z_band(contexts))
    moved = [1.11, 1.11, 0.61]
    # the substitution has to BITE: a different source with the same oracle would
    # prove nothing about the gate
    assert abs(mg.grid_oracle_error(banded["candidates"], moved)
               - mg.grid_oracle_error(banded["candidates"], truth)) > mr.ORACLE_TOLERANCE

    scene, scene_id = "A/A_idx_1".split("/")
    path = os.path.join(fixture["metadata_root"], scene, scene_id, "S001_R002.json")
    payload = json.load(open(path))
    payload["src_loc"] = moved
    with open(path, "w") as handle:
        json.dump(payload, handle)
    with pytest.raises(ValueError, match="is not the one the audit measured"):
        evaluate_fixture(fixture)


def test_a_row_whose_published_oracle_is_not_its_own_geometry_is_refused():
    coordinates = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    sims = _f16([[0.9, 0.9], [0.1, 0.1]])
    row = _hand_row(sims, coordinates)
    row["e_oracle"] = 0.0                                # the truth is 1 m from the grid
    with pytest.raises(ValueError, match="is not the one the audit measured"):
        mr.evaluate_query(row, sims, coordinates, [4.0, 0.0, 0.0])


# --------------------------------------------------------------------------- #
# one query: the arithmetic and the cross-checks
# --------------------------------------------------------------------------- #
def _f16(sims):
    """The sidecar as the engine publishes it: float16, exactly (r9c M7)."""
    return np.asarray(sims, dtype=np.float16)


def _hand_row(sims, coordinates, tau=0.1, prefixes=(1, 2), scored_from=None):
    """A row built by the ENGINE's own scorer over hand-chosen similarities.

    ``scored_from`` lets a fixture score the row from values the float16 sidecar
    cannot represent, which is how the sub-ulp flip case is constructed; by
    default the row is scored from exactly what the sidecar carries.
    """
    indices = list(range(len(coordinates)))
    source = np.asarray(sims, dtype=np.float16).astype(np.float32) \
        if scored_from is None else np.asarray(scored_from, dtype=np.float32)
    scored = me.score_query(torch.as_tensor(source), indices,
                            np.asarray(coordinates, dtype=np.float64), tau=tau,
                            prefixes=prefixes)
    return {"query_id": "0|ir/A/A_idx_1/S001_R002_hybrid_IR.wav", "room_id": "A/A_idx_1",
            "position": 0, "receiver_id": "A/A_idx_1|0,0,0",
            "n_candidates": len(indices), "num_samples": int(np.asarray(sims).shape[1]),
            "tau": float(tau), "k_prefixes": [int(k) for k in prefixes],
            "candidate_indices": indices, "e_oracle": None,
            "sims_dtype": me.SIMS_DTYPE,
            "by_k": {str(k): block for k, block in scored["by_k"].items()},
            "timings_s": {name: 1.0 for name in mr.ROW_TIMING_COMPONENTS}}


def test_e_loc_e_oracle_and_e_excess_are_the_registered_arithmetic():
    coordinates = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    sims = _f16([[0.1, 0.1], [0.9, 0.9]])                 # candidate 1 wins
    truth = [4.0, 0.0, 0.0]
    row = _hand_row(sims, coordinates)
    row["e_oracle"] = 1.0                                 # min(4.0, 1.0)
    result = mr.evaluate_query(row, sims, coordinates, truth)
    assert result["e_oracle"] == pytest.approx(1.0)
    entry = result["by"]["lme"][2]
    assert entry["prediction_index"] == 1
    assert entry["e_loc"] == pytest.approx(1.0)
    assert entry["e_excess"] == pytest.approx(0.0)       # the prediction IS the oracle cell
    # and a losing prediction carries the whole excess
    sims_flipped = _f16([[0.9, 0.9], [0.1, 0.1]])
    row2 = _hand_row(sims_flipped, coordinates)
    row2["e_oracle"] = 1.0
    entry2 = mr.evaluate_query(row2, sims_flipped, coordinates, truth)["by"]["lme"][2]
    assert entry2["e_loc"] == pytest.approx(4.0)
    assert entry2["e_excess"] == pytest.approx(3.0)


def test_the_success_boundary_counts_as_a_success():
    coordinates = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
    sims = _f16([[0.9, 0.9], [0.1, 0.1]])                 # candidate 0 wins
    truth = [0.5, 0.0, 0.0]                               # e_loc is exactly 0.5
    row = _hand_row(sims, coordinates)
    row["e_oracle"] = 0.0
    entry = mr.evaluate_query(row, sims, coordinates, truth)["by"]["lme"][2]
    assert entry["e_loc"] == pytest.approx(0.5)
    assert entry["success_raw"]["0.5"] == 1.0
    assert entry["success_raw"]["1.0"] == 1.0
    assert entry["success_oracle_normalized"]["0.5"] == 1.0
    # ... and one micron past it does not
    moved = [0.500001, 0.0, 0.0]
    row2 = _hand_row(sims, coordinates)
    row2["e_oracle"] = float(np.linalg.norm(np.asarray(coordinates, dtype=np.float64)
                                            - np.asarray(moved), axis=1).min())
    entry2 = mr.evaluate_query(row2, sims, coordinates, moved)["by"]["lme"][2]
    assert entry2["e_loc"] > 0.5
    assert entry2["success_raw"]["0.5"] == 0.0
    assert entry2["success_raw"]["1.0"] == 1.0


def test_lme_and_s_mean_are_both_recomputed_and_both_cross_checked():
    coordinates = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    # candidate 1 has the best MEAN, candidate 2 the best log-mean-exp peak
    sims = _f16([[0.10, 0.10], [0.60, 0.60], [0.95, 0.20]])
    row = _hand_row(sims, coordinates, tau=0.01)
    row["e_oracle"] = 0.0
    result = mr.evaluate_query(row, sims, coordinates, [0.0, 0.0, 0.0], tau=0.01)
    assert result["by"]["lme"][2]["prediction_index"] == 2
    assert result["by"]["mean"][2]["prediction_index"] == 1
    for aggregator in mr.AGGREGATORS:
        assert result["sidecar"][aggregator][2]["argmax_agrees"] is True


def test_a_row_whose_argmax_is_not_its_own_scores_is_refused():
    coordinates = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    sims = _f16([[0.1, 0.1], [0.9, 0.9]])
    row = _hand_row(sims, coordinates)
    row["e_oracle"] = 0.0
    row["by_k"]["2"]["prediction_row"] = 0
    row["by_k"]["2"]["prediction_index"] = 0
    with pytest.raises(ValueError, match="internally inconsistent"):
        mr.evaluate_query(row, sims, coordinates, [0.0, 0.0, 0.0])


def test_a_row_whose_prediction_xyz_is_not_the_g1_coordinate_is_refused():
    coordinates = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    sims = _f16([[0.1, 0.1], [0.9, 0.9]])
    row = _hand_row(sims, coordinates)
    row["e_oracle"] = 0.0
    row["by_k"]["2"]["prediction_xyz"] = [1.0, 0.0, 0.5]
    with pytest.raises(ValueError, match="different candidate array"):
        mr.evaluate_query(row, sims, coordinates, [0.0, 0.0, 0.0])


def test_an_inflated_margin_cannot_excuse_an_argmax_that_could_flip():
    coordinates = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    sims = _f16([[0.1, 0.1], [0.9, 0.9]])
    row = _hand_row(sims, coordinates)
    row["e_oracle"] = 0.0
    row["by_k"]["2"]["margin"] = 10.0
    with pytest.raises(ValueError, match="inflated margin"):
        mr.evaluate_query(row, sims, coordinates, [0.0, 0.0, 0.0])


def _float16_flip_case():
    """A row whose float16 sidecar reverses a sub-ulp ordering.

    At 0.5 the float16 spacing is 2^-11, so 0.5 and 0.50005 both round to 0.5:
    the row's float32 scores put candidate 1 ahead, and the sidecar ties them,
    which the registered tie-break resolves towards the SMALLER global index.
    """
    coordinates = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    stored = [[0.5, 0.5], [0.50005, 0.50005]]
    sidecar = _f16(stored)
    row = _hand_row(sidecar, coordinates, scored_from=stored)
    row["e_oracle"] = 0.0
    assert sidecar[0, 0] == sidecar[1, 0]                 # the flip is real
    return row, sidecar, coordinates


def test_a_float16_argmax_flip_is_counted_and_named_not_absorbed():
    row, sidecar, coordinates = _float16_flip_case()
    result = mr.evaluate_query(row, sidecar, coordinates, [0.0, 0.0, 0.0])
    entry = result["sidecar"]["lme"][2]
    assert entry["argmax_agrees"] is False
    # named for what the inequality states, not for a cause it cannot establish
    assert entry["argmax_flip_within_2dev"] is True
    assert "explained_by_precision" not in entry
    assert entry["within_float16_bound"] is True
    assert entry["max_abs_delta"] <= entry["float16_bound"]
    assert entry["sims_dtype"] == me.SIMS_DTYPE
    # the PUBLISHED prediction still comes from the row's float32 score
    assert result["by"]["lme"][2]["prediction_index"] == 1


def test_every_argmax_flip_satisfies_the_2dev_inequality_by_arithmetic():
    # the flag can never be False on a flip: the leader must lose and the
    # runner-up gain at least the margin between them, so max deviation >= m / 2
    row, sidecar, coordinates = _float16_flip_case()
    entry = mr.evaluate_query(row, sidecar, coordinates,
                              [0.0, 0.0, 0.0])["sidecar"]["lme"][2]
    assert entry["margin"] <= me.ARGMAX_STABILITY_FACTOR * entry["max_abs_delta"]


def test_a_sidecar_that_is_not_a_float16_quantization_is_refused_outright():
    """The ABSOLUTE bound: no argmax agreement can excuse an impossible deviation."""
    coordinates = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    sidecar = _f16([[0.10, 0.10], [0.90, 0.90]])
    # the row was scored from similarities the sidecar is nowhere near, yet the
    # ORDER is unchanged, so the r9 argmax rule would have said nothing
    row = _hand_row(sidecar, coordinates, scored_from=[[0.30, 0.30], [0.95, 0.95]])
    row["e_oracle"] = 0.0
    with pytest.raises(ValueError, match="not a quantization of what this row was scored"):
        mr.evaluate_query(row, sidecar, coordinates, [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="not a quantization of what this row was scored"):
        mr.evaluate_query(row, sidecar, coordinates, [0.0, 0.0, 0.0],
                          sidecar_argmax_policy="strict")


def test_the_float16_bound_is_the_half_ulp_of_the_stored_samples():
    near_one = mr.float16_quantization_bound(_f16([[0.9, 0.9]]))
    near_zero = mr.float16_quantization_bound(_f16([[0.001, 0.001]]))
    assert near_one > near_zero                      # ulps grow with magnitude
    assert near_one == pytest.approx(
        0.5 * float(np.spacing(np.float16(0.9))) + mr.SIDECAR_FLOAT32_SLACK)
    with pytest.raises(ValueError, match="at least one sample"):
        mr.float16_quantization_bound(np.zeros((0, 2), dtype=np.float16))


def test_a_widened_or_undeclared_sidecar_is_refused():
    coordinates = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    sims = _f16([[0.1, 0.1], [0.9, 0.9]])
    row = _hand_row(sims, coordinates)
    row["e_oracle"] = 0.0
    with pytest.raises(ValueError, match="the sidecar array is float32"):
        mr.evaluate_query(row, sims.astype(np.float32), coordinates, [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="declares sims_dtype"):
        mr.evaluate_query(dict(row, sims_dtype="float32"), sims, coordinates,
                          [0.0, 0.0, 0.0])


def test_the_strict_policy_refuses_the_same_flip():
    row, sidecar, coordinates = _float16_flip_case()
    with pytest.raises(ValueError, match="recomputed from the float16 sidecar"):
        mr.evaluate_query(row, sidecar, coordinates, [0.0, 0.0, 0.0],
                          sidecar_argmax_policy="strict")


def test_an_unknown_sidecar_policy_is_refused():
    row, sidecar, coordinates = _float16_flip_case()
    with pytest.raises(ValueError, match="unknown sidecar_argmax_policy"):
        mr.evaluate_query(row, sidecar, coordinates, [0.0, 0.0, 0.0],
                          sidecar_argmax_policy="lenient")


def test_a_sidecar_of_the_wrong_shape_is_refused():
    coordinates = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    sims = _f16([[0.1, 0.1], [0.9, 0.9]])
    row = _hand_row(sims, coordinates)
    row["e_oracle"] = 0.0
    with pytest.raises(ValueError, match="the sidecar is"):
        mr.evaluate_query(row, _f16([[0.1], [0.9]]), coordinates, [0.0, 0.0, 0.0])


# --------------------------------------------------------------------------- #
# room-first aggregation and the room bootstrap
# --------------------------------------------------------------------------- #
def _synthetic_results(by_room, aggregator="lme", k=8):
    results, position = [], 0
    for room, errors in sorted(by_room.items()):
        for error in errors:
            results.append({
                "query_id": f"{position}|ir/{room}/x_hybrid_IR.wav", "room_id": room,
                "position": position, "n_candidates": 10, "num_samples": 8,
                "e_oracle": 0.0, "truth_xyz": [0.0, 0.0, 0.0],
                "oracle_candidate_index": 0, "oracle_candidate_xyz": [0.0, 0.0, 0.0],
                "latency_s": {"sampling": 1.0},
                "by": {aggregator: {k: {"e_loc": float(error), "e_excess": float(error),
                                        "best_score": 0.5, "margin": 0.1,
                                        "prediction_index": 0,
                                        "prediction_xyz": [0.0, 0.0, 0.0],
                                        "success_raw": {}, "success_oracle_normalized": {}}}},
                "sidecar": {aggregator: {k: {"max_abs_delta": 0.0, "argmax_agrees": True,
                                             "margin": 0.1,
                                             "explained_by_precision": False}}},
            })
            position += 1
    return results


def test_aggregation_is_room_first_and_not_pooled():
    # room A: nine queries at 1 m; room B: one query at 100 m
    results = _synthetic_results({"A/A_idx_1": [1.0] * 9, "B/B_idx_2": [100.0]})
    cell = mr.build_cell(results, "lme", 8, n=64)
    assert cell["per_room"]["A/A_idx_1"]["median_e_loc"] == pytest.approx(1.0)
    assert cell["per_room"]["B/B_idx_2"]["median_e_loc"] == pytest.approx(100.0)
    # room-first: (1 + 100) / 2, not the pooled median of 1.0
    assert cell["across_rooms"]["median_e_loc"]["point"] == pytest.approx(50.5)
    assert cell["pooled"]["median_e_loc"] == pytest.approx(1.0)
    assert "secondary" in cell["pooled"]["label"]


def test_the_room_bootstrap_is_deterministic_under_the_registered_seed():
    first = mr.room_bootstrap_draws(16, seed=mr.BOOTSTRAP_SEED, n=1000)
    second = mr.room_bootstrap_draws(16, seed=mr.BOOTSTRAP_SEED, n=1000)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, mr.room_bootstrap_draws(16, seed=1, n=1000))
    assert first.shape == (1000, 16)
    assert first.min() >= 0 and first.max() <= 15


def test_the_bootstrap_interval_is_reproducible_and_brackets_the_point():
    results = _synthetic_results({f"R{i}/R{i}_idx_1": [float(i)] for i in range(1, 9)})
    left = mr.build_cell(results, "lme", 8)
    right = mr.build_cell(results, "lme", 8)
    for name in mr.flat_stat_names():
        assert left["across_rooms"][name] == right["across_rooms"][name]
    entry = left["across_rooms"]["mean_e_loc"]
    assert entry["ci_lo"] <= entry["point"] <= entry["ci_hi"]
    assert left["across_rooms"]["_settings"]["bootstrap_seed"] == mr.BOOTSTRAP_SEED
    assert left["across_rooms"]["_settings"]["n_boot"] == mr.BOOTSTRAP_N


def test_a_single_room_collapses_the_interval_onto_the_point():
    results = _synthetic_results({"A/A_idx_1": [1.0, 3.0]})
    cell = mr.build_cell(results, "lme", 8, n=128)
    entry = cell["across_rooms"]["mean_e_loc"]
    assert entry["ci_lo"] == pytest.approx(entry["point"])
    assert entry["ci_hi"] == pytest.approx(entry["point"])


def test_the_percentile_interval_pins_its_interpolation():
    low, high = mr.percentile_ci(np.arange(101, dtype=np.float64), alpha=0.05)
    assert (low, high) == pytest.approx((2.5, 97.5))
    with pytest.raises(ValueError, match="alpha must be in"):
        mr.percentile_ci([1.0, 2.0], alpha=1.5)


# --------------------------------------------------------------------------- #
# the random baseline
# --------------------------------------------------------------------------- #
def test_the_baseline_draw_is_deterministic_per_seed_and_query():
    assert mr.draw_baseline_candidate(101, "q|a.wav", 1000) == \
        mr.draw_baseline_candidate(101, "q|a.wav", 1000)
    draws = {seed: mr.draw_baseline_candidate(seed, "q|a.wav", 1000)
             for seed in mr.RANDOM_BASELINE_SEEDS}
    assert len(set(draws.values())) > 1                   # the seeds are independent
    assert mr.draw_baseline_candidate(101, "q|a.wav", 1000) != \
        mr.draw_baseline_candidate(101, "q|b.wav", 1000)


def test_the_baseline_key_is_not_pythons_salted_hash():
    assert mr.baseline_key(101, "q|a.wav") == mr.baseline_key(101, "q|a.wav")
    assert mr.baseline_key(101, "q|a.wav") != mr.baseline_key(102, "q|a.wav")
    assert 0 <= mr.baseline_key(101, "q|a.wav") < (1 << 63)


def test_the_baseline_draw_does_not_depend_on_iteration_order():
    queries = [f"{i}|ir/A/A_idx_1/S00{i}_R001_hybrid_IR.wav" for i in range(20)]
    forward = [mr.draw_baseline_candidate(101, q, 37) for q in queries]
    backward = [mr.draw_baseline_candidate(101, q, 37) for q in reversed(queries)]
    assert forward == list(reversed(backward))


def test_the_baseline_draw_stays_inside_the_candidate_set():
    for size in (1, 2, 37, 5295):
        for seed in mr.RANDOM_BASELINE_SEEDS:
            assert 0 <= mr.draw_baseline_candidate(seed, "q|a.wav", size) < size
    with pytest.raises(ValueError, match="non-empty candidate set"):
        mr.draw_baseline_candidate(101, "q|a.wav", 0)


def test_the_baseline_reports_every_repetition_and_the_pooled_summary():
    results = _synthetic_results({"A/A_idx_1": [1.0, 2.0], "B/B_idx_2": [3.0]})
    for index, result in enumerate(results):
        result["baseline_e_loc"] = {seed: float(seed) / 100.0 + index
                                    for seed in mr.RANDOM_BASELINE_SEEDS}
    report = mr.baseline_report(results, bootstrap={"n": 64})
    assert [rep["seed"] for rep in report["repetitions"]] == list(mr.RANDOM_BASELINE_SEEDS)
    assert report["summary_over_repetitions"]["mean_e_loc"]["sd"] > 0.0
    assert len(report["summary_over_repetitions"]["mean_e_loc"]["per_seed"]) == 5
    assert report["all_draws"]["per_room"]["A/A_idx_1"]["n_queries"] == 10


# --------------------------------------------------------------------------- #
# associations
# --------------------------------------------------------------------------- #
def test_correlation_is_exact_on_a_monotone_pair_and_names_the_degenerate_case():
    x = [1.0, 2.0, 3.0, 4.0]
    assert mr.correlation(x, [2.0, 4.0, 6.0, 8.0])["pearson"] == pytest.approx(1.0)
    assert mr.correlation(x, [8.0, 4.0, 2.0, 1.0])["spearman"] == pytest.approx(-1.0)
    assert mr.correlation(x, [1.0, 1.0, 1.0, 1.0])["pearson"] is None
    assert mr.correlation([1.0], [1.0])["pearson"] is None
    with pytest.raises(ValueError, match="paired inputs"):
        mr.correlation([1.0, 2.0], [1.0])


def test_the_rank_transform_averages_ties():
    assert mr._rankdata([10.0, 20.0, 20.0, 30.0]).tolist() == [1.0, 2.5, 2.5, 4.0]


def test_the_association_report_covers_the_registered_pairs():
    results = _synthetic_results({"A/A_idx_1": [1.0, 2.0, 3.0, 4.0],
                                  "B/B_idx_2": [5.0, 6.0, 7.0, 8.0]})
    for index, result in enumerate(results):
        result["n_candidates"] = 10 + index
        result["by"]["lme"][8]["best_score"] = 0.1 * index
    report = mr.association_report(results)
    assert report["pooled"]["n_candidates_vs_best_score"]["spearman"] == pytest.approx(1.0)
    assert set(report["pooled"]) >= {"n_candidates_vs_best_score", "n_candidates_vs_e_loc",
                                     "n_candidates_vs_e_excess", "n_candidates_vs_e_oracle"}
    assert sum(bucket["n_queries"] for bucket in report["by_candidate_count_quantile"]) == 8
    assert "diagnostic only" in report["note"]


# --------------------------------------------------------------------------- #
# latency
# --------------------------------------------------------------------------- #
def _timed(results, seconds=2.0, n_candidates=10, num_samples=8):
    for result in results:
        result["latency_s"] = {name: seconds / len(mr.ROW_TIMING_COMPONENTS)
                               for name in mr.ROW_TIMING_COMPONENTS}
        result["n_candidates"] = n_candidates
        result["num_samples"] = num_samples
    return results


def test_latency_is_reported_per_query_candidate_and_generated_rir():
    results = _timed(_synthetic_results({"A/A_idx_1": [1.0, 2.0]}))
    report = mr.latency_report(results, n=64)
    assert report["total_seconds"] == pytest.approx(4.0)
    assert report["per_room"]["A/A_idx_1"]["mean_seconds_per_query"] == pytest.approx(2.0)
    assert report["pooled"]["seconds_per_candidate"] == pytest.approx(4.0 / 20)
    assert report["pooled"]["seconds_per_generated_rir"] == pytest.approx(4.0 / 160)
    assert "context branch" in report["scope_note"]


def test_latency_is_aggregated_room_first_with_a_room_bootstrap():
    # room A is ten times slower than room B; the pooled number is dominated by
    # whichever room has more queries, the room-first one is not
    results = _timed(_synthetic_results({"A/A_idx_1": [1.0]}), seconds=10.0)
    results += _timed(_synthetic_results({"B/B_idx_2": [1.0, 2.0, 3.0]}), seconds=1.0)
    for index, result in enumerate(results):
        result["position"] = index
    report = mr.latency_report(results, n=128)
    across = report["across_rooms"]
    assert across["mean_seconds_per_query"]["point"] == pytest.approx((10.0 + 1.0) / 2)
    assert report["pooled"]["seconds_per_query"]["mean"] == pytest.approx(13.0 / 4)
    assert across["mean_seconds_per_query"]["ci_lo"] <= \
        across["mean_seconds_per_query"]["point"] <= \
        across["mean_seconds_per_query"]["ci_hi"]
    assert across["_settings"]["n_rooms"] == 2
    assert across["_settings"]["bootstrap_seed"] == mr.BOOTSTRAP_SEED
    assert sorted(name for name in across if not name.startswith("_")) == \
        sorted(mr.LATENCY_STAT_NAMES)


def test_a_missing_timing_component_is_named_and_counted_never_zeroed():
    results = _timed(_synthetic_results({"A/A_idx_1": [1.0, 2.0]}))
    results[0]["latency_s"].pop("decode")
    report = mr.latency_report(results, n=64)
    # the incomplete row is excluded from the headline, not folded in at zero
    assert report["n_queries"] == 1
    assert report["n_incomplete"] == 1
    assert report["missing_components"]["decode"]["n_rows"] == 1
    assert report["missing_components"]["decode"]["query_ids"] == [results[0]["query_id"]]
    assert report["incomplete_rows"][0]["missing"] == ["decode"]
    assert report["total_seconds"] == pytest.approx(2.0)      # the complete row only
    assert "counted" in report["completeness_note"]


def test_a_row_without_timings_refuses_rather_than_reporting_zero():
    results = _synthetic_results({"A/A_idx_1": [1.0]})
    results[0]["latency_s"] = {}
    with pytest.raises(ValueError, match="no row carries all of"):
        mr.latency_report(results)


# --------------------------------------------------------------------------- #
# the pre-registered visualization cases
# --------------------------------------------------------------------------- #
def test_the_quantile_selection_is_the_registered_rule():
    results = _synthetic_results({"A/A_idx_1": [5.0, 1.0, 9.0, 3.0, 7.0]})
    selection = mr.select_visualization_cases(results)
    labels = [case["quantile"] for case in selection["cases"]]
    assert labels == ["lowest_e_loc", "median_e_loc", "highest_e_loc"]
    assert [case["e_loc"] for case in selection["cases"]] == [1.0, 5.0, 9.0]
    assert selection["aggregator"] == "lme" and selection["k"] == 8


def test_every_quantile_breaks_ties_by_the_smallest_position_including_the_highest():
    results = _synthetic_results({"A/A_idx_1": [1.0, 1.0, 1.0, 1.0]})
    selection = mr.select_visualization_cases(results)
    positions = [case["position"] for case in selection["cases"]]
    # every error is equal, so all three quantiles name the SAME value and the
    # tie-break decides -- and it is the smallest position in all three, which is
    # what the printed rule promises (Codex r9 review, finding 10)
    assert positions == [0, 0, 0]
    assert [case["n_attaining"] for case in selection["cases"]] == [4, 4, 4]
    assert mr.select_visualization_cases(list(reversed(results)))["cases"] == \
        selection["cases"]


def test_the_highest_error_case_no_longer_takes_the_largest_tied_position():
    # two queries share the maximum error; the rule names the earlier one
    results = _synthetic_results({"A/A_idx_1": [1.0, 9.0, 5.0, 9.0]})
    highest = mr.select_visualization_cases(results)["cases"][-1]
    assert highest["quantile"] == "highest_e_loc"
    assert highest["e_loc"] == 9.0
    assert highest["position"] == 1
    assert highest["n_attaining"] == 2


def test_the_median_case_is_the_lower_median_for_an_even_set():
    results = _synthetic_results({"A/A_idx_1": [1.0, 2.0, 3.0, 4.0]})
    selection = mr.select_visualization_cases(results)
    assert [case["e_loc"] for case in selection["cases"]] == [1.0, 2.0, 4.0]


def test_the_selection_is_a_pure_function_of_the_results():
    results = _synthetic_results({"A/A_idx_1": [5.0, 1.0], "B/B_idx_2": [9.0]})
    assert mr.select_visualization_cases(results) == mr.select_visualization_cases(results)
    with pytest.raises(ValueError, match="at least one scored query"):
        mr.select_visualization_cases([])


# --------------------------------------------------------------------------- #
# the whole report, end to end on the fixture run
# --------------------------------------------------------------------------- #
def test_the_report_gates_then_evaluates_the_whole_fixture_run(tmp_path):
    fixture = build_fixture_run(tmp_path)
    evaluated = evaluate_fixture(fixture)
    assert evaluated["binding_sha256"] == fixture["binding_sha256"]
    assert len(evaluated["results"]) == fixture["totals"]["queries"]
    assert [r["position"] for r in evaluated["results"]] == [0, 1, 2, 3]
    for result in evaluated["results"]:
        assert result["e_oracle"] > 0.0
        assert result["e_oracle_delta"] <= mr.ORACLE_TOLERANCE
        assert set(result["by"]) == {"lme", "mean"}
        assert sorted(result["by"]["lme"]) == list(FIXTURE_PREFIXES)
        assert set(result["baseline_e_loc"]) == set(mr.RANDOM_BASELINE_SEEDS)


def test_the_published_report_carries_every_registered_readout(tmp_path):
    fixture = build_fixture_run(tmp_path)
    evaluated = evaluate_fixture(fixture)
    report = mr.build_report(evaluated, fixture["run_dir"], fixture["audit_report"],
                             fixture["context_manifest"], fixture["metadata_root"], n_boot=256)
    for aggregator in mr.AGGREGATORS:
        for k in FIXTURE_PREFIXES:
            cell = report["metrics"][aggregator][str(k)]
            assert sorted(cell["per_room"]) == sorted(FIXTURE_QUERIES)
            for name in mr.flat_stat_names():
                entry = cell["across_rooms"][name]
                assert entry["ci_lo"] <= entry["point"] <= entry["ci_hi"]
    assert report["oracle"]["threshold_m"] == 0.5
    assert report["latency"]["n_queries"] == 4
    assert len(report["random_baseline"]["repetitions"]) == 5
    assert report["associations"]["k"] == mr.HEADLINE_K
    assert report["crosscheck"]["oracle"]["max_abs_delta_m"] <= mr.ORACLE_TOLERANCE
    assert report["census"]["n_rooms"] == 2
    assert report["gates"]["binding_recomputed_from_content"] is True


def test_the_report_names_the_registered_controls_it_does_not_contain(tmp_path):
    fixture = build_fixture_run(tmp_path)
    evaluated = evaluate_fixture(fixture)
    report = mr.build_report(evaluated, fixture["run_dir"], fixture["audit_report"],
                             fixture["context_manifest"], fixture["metadata_root"], n_boot=128)
    elsewhere = report["controls_elsewhere"]
    assert "meshgrid_offgrid_probe.py" in elsewhere["off_grid_truth_probe"]
    assert "meshgrid_offgrid_probe.py" in \
        elsewhere["real_vs_generated_agree_calibration"]
    # the sparse/metadata-bank control: named with its tool and its state, so
    # silence is read neither as a null nor as a published number (r9b)
    sparse = elsewhere["agree_oracle_retrieval_over_the_metadata_bank"]
    assert "meshgrid_retrieval_control.py" in sparse
    assert "built (r9b), run pending" in sparse
    assert "sparse/metadata-bank" in sparse and "retrieval_control_handoff.json" in sparse
    markdown = mr.render_markdown(report)
    assert "controls that are NOT in this report" in markdown


def test_every_emitted_artifact_carries_the_leakage_caveat_and_the_subset_label(tmp_path):
    fixture = build_fixture_run(tmp_path)
    evaluated = evaluate_fixture(fixture)
    report = mr.build_report(evaluated, fixture["run_dir"], fixture["audit_report"],
                             fixture["context_manifest"], fixture["metadata_root"], n_boot=128)
    cases = mr.build_case_payload(report["visualization_cases"], evaluated["rows_by_id"],
                                  evaluated["plans_by_id"],
                                  {r["query_id"]: r for r in evaluated["results"]},
                                  fixture["run_dir"], evaluated["plan"])
    published = mr.write_report(str(tmp_path / "r1"), report, cases)

    payload = json.load(open(published["paths"]["json"]))
    assert payload["labels"]["agree_leakage_caveat"] == me.AGREE_LEAKAGE_CAVEAT
    assert payload["labels"]["subset"] == mr.SUBSET_LABEL
    assert payload["labels"]["sims_precision_caveat"] == me.SIMS_PRECISION_CAVEAT
    assert payload["labels"]["scorer_readout_deviation"] == me.SCORER_READOUT_DEVIATION

    markdown = open(published["paths"]["markdown"]).read()
    assert me.AGREE_LEAKAGE_CAVEAT in markdown
    assert mr.SUBSET_LABEL in markdown
    assert fixture["binding_sha256"] in markdown
    # every table is stamped, not only the header
    assert markdown.count(fixture["binding_sha256"][:16]) >= 6

    case_payload = json.load(open(published["paths"]["cases"]))
    assert case_payload["agree_leakage_caveat"] == me.AGREE_LEAKAGE_CAVEAT
    assert case_payload["subset"] == mr.SUBSET_LABEL


def test_the_case_file_is_a_registered_dump_list_a_renderer_can_read(tmp_path):
    fixture = build_fixture_run(tmp_path)
    evaluated = evaluate_fixture(fixture)
    report = mr.build_report(evaluated, fixture["run_dir"], fixture["audit_report"],
                             fixture["context_manifest"], fixture["metadata_root"], n_boot=128)
    cases = mr.build_case_payload(report["visualization_cases"], evaluated["rows_by_id"],
                                  evaluated["plans_by_id"],
                                  {r["query_id"]: r for r in evaluated["results"]},
                                  fixture["run_dir"], evaluated["plan"])
    published = mr.write_report(str(tmp_path / "r1"), report, cases)

    # the engine's own dump-case loader must accept it, digest and all
    loaded = me.load_dump_cases(published["paths"]["cases"],
                                expected_sha256=published["sha256"]["cases"])
    assert loaded["query_ids"] == list(
        dict.fromkeys(case["query_id"] for case in cases["cases"]))
    assert len(cases["cases"]) == 3
    assert me.assert_dump_allowed(loaded["query_ids"], set(loaded["query_ids"])) is True

    case = cases["cases"][0]
    assert len(case["candidate_xyz"]) == case["n_candidates"] == len(case["scores"])
    assert case["score_softmax"] == pytest.approx(
        (np.exp(np.asarray(case["scores"]) / mr.VISUALIZATION_T)
         / np.exp(np.asarray(case["scores"]) / mr.VISUALIZATION_T).sum()).tolist(), rel=1e-5)
    assert sum(case["score_softmax"]) == pytest.approx(1.0)
    for key in ("receiver_xyz", "truth_xyz", "prediction_xyz", "oracle_candidate_xyz",
                "e_loc", "e_excess", "e_oracle", "z_band", "branch", "mesh", "sims_path"):
        assert key in case
    assert "uncalibrated" in case["score_softmax_label"]


def test_the_report_refuses_a_run_whose_rooms_are_not_the_audits(tmp_path):
    fixture = build_fixture_run(tmp_path)
    room_dir = os.path.join(fixture["run_dir"], me.ROWS_DIRNAME,
                            me.room_stem("B/B_idx_2"))
    for name in os.listdir(room_dir):
        os.remove(os.path.join(room_dir, name))
    os.rmdir(room_dir)
    # the merge receipt now claims totals the surviving rows cannot supply, and
    # that is caught before the census even runs (r9d B1)
    with pytest.raises(ValueError, match="but the rows themselves yield"):
        evaluate_fixture(fixture)
    # ... and with the receipt out of the way the identity join still catches it
    with pytest.raises(ValueError, match="have no published row"):
        evaluate_fixture(fixture, single_shard=True)


def test_the_report_refuses_a_manifest_that_is_not_the_registered_census(tmp_path):
    fixture = build_fixture_run(tmp_path)
    with pytest.raises(ValueError, match="the registered census is"):
        mr.evaluate_run(fixture["run_dir"], fixture["audit_report"],
                        fixture["context_manifest"], fixture["metadata_root"],
                        totals=fixture["totals"])


def test_the_cli_defaults_are_the_registered_settings():
    args = mr.parse_args(["--run-dir", "run", "--out-dir", "out"])
    assert args.bootstrap_seed == mr.BOOTSTRAP_SEED
    assert args.n_boot == mr.BOOTSTRAP_N
    assert args.baseline_seeds == list(mr.RANDOM_BASELINE_SEEDS)
    assert args.sidecar_argmax_policy == mr.SIDECAR_ARGMAX_POLICY
    assert args.metadata_root == os.path.join("AcousticRooms", "metadata")
    with pytest.raises(SystemExit):
        mr.validate_args(mr.parse_args(["--run-dir", "r", "--out-dir", "o", "--n-boot", "0"]))


# --------------------------------------------------------------------------- #
# r9c: the blockers the Codex r9 review found, each with its exploit
# --------------------------------------------------------------------------- #
def test_a_second_valid_g1_audit_cannot_be_swapped_in(tmp_path):
    """B1: the binding authenticated only itself; the FILES were never joined."""
    fixture = build_fixture_run(tmp_path)
    other = _fixture_audit(tmp_path / "other", stamp="a second valid audit")
    assert me.load_audit_plan(other).branch == "z_band"
    with pytest.raises(ValueError, match="g1_report_sha256"):
        mr.evaluate_run(fixture["run_dir"], other, fixture["context_manifest"],
                        fixture["metadata_root"], totals=fixture["totals"],
                        require_manifest_census=False)


def test_a_second_valid_d1_manifest_cannot_be_swapped_in(tmp_path):
    fixture = build_fixture_run(tmp_path)
    other = str(tmp_path / "other_d1.json")
    payload = json.load(open(fixture["context_manifest"]))
    payload["experiment"] = "a different, equally valid manifest"
    mq.write_manifest(other, payload, require_census=False)
    with pytest.raises(ValueError, match="d1_manifest_sha256"):
        mr.evaluate_run(fixture["run_dir"], fixture["audit_report"], other,
                        fixture["metadata_root"], totals=fixture["totals"],
                        require_manifest_census=False)


def test_an_edited_room_manifest_is_caught_by_the_hash_join(tmp_path):
    fixture = build_fixture_run(tmp_path)
    plan = fixture["plan"]
    with pytest.raises(ValueError, match="room_manifest_sha256"):
        mr.assert_artifact_hashes(dict(fixture["binding"],
                                       room_manifest_sha256={room: "0" * 64
                                                             for room in plan.rooms}),
                                  plan, fixture["context_manifest"])
    assert mr.assert_artifact_hashes(fixture["binding"], plan,
                                     fixture["context_manifest"])["n_room_manifests"] == 2


def test_a_never_merged_directory_is_refused_as_the_canonical_pass(tmp_path):
    """B1: a shard-local directory passed every per-row check and was reported."""
    fixture = build_fixture_run(tmp_path, merged=False)
    assert not os.path.isfile(os.path.join(fixture["run_dir"], "merge_report.json"))
    with pytest.raises(ValueError, match="publishes no merge_report.json"):
        mr.evaluate_run(fixture["run_dir"], fixture["audit_report"],
                        fixture["context_manifest"], fixture["metadata_root"],
                        totals=fixture["totals"], require_manifest_census=False)


def test_single_shard_relaxes_only_the_merge_report_never_a_hash_join(tmp_path):
    fixture = build_fixture_run(tmp_path, merged=False)
    evaluated = evaluate_fixture(fixture)                 # single_shard=True
    assert evaluated["single_shard"] is True
    assert evaluated["merge"] is None
    # ... and the hash joins are still enforced in that mode
    other = _fixture_audit(tmp_path / "other", stamp="a second valid audit")
    with pytest.raises(ValueError, match="g1_report_sha256"):
        mr.evaluate_run(fixture["run_dir"], other, fixture["context_manifest"],
                        fixture["metadata_root"], totals=fixture["totals"],
                        require_manifest_census=False, single_shard=True)


def test_the_merge_report_must_authenticate_this_directory(tmp_path):
    fixture = build_fixture_run(tmp_path)
    path = os.path.join(fixture["run_dir"], "merge_report.json")
    pristine = json.load(open(path))
    for field, value, pattern in (
            ("binding_sha256", "0" * 64, "written under binding"),
            ("declared_rooms", ["A/A_idx_1"], "declared rooms are not the audit's"),
            ("n_rows", 3, "rows for a registered census"),
            ("ok", False, "does not claim success"),
            ("advisory", {"batch_rows": 999, "source_chunk": 1}, "advisory batching"),
            ("totals", dict(pristine["totals"], source_rows=1), "source_rows census is")):
        me.write_json(path, dict(pristine, **{field: value}))
        with pytest.raises(ValueError, match=pattern):
            evaluate_fixture(fixture)
    me.write_json(path, pristine)
    assert evaluate_fixture(fixture)["merge"]["n_rows"] == 4


def test_the_merged_run_reports_its_merge_gates(tmp_path):
    fixture = build_fixture_run(tmp_path)
    evaluated = evaluate_fixture(fixture)
    assert evaluated["single_shard"] is False
    assert evaluated["merge"]["declared_rooms"] == sorted(FIXTURE_QUERIES)
    assert evaluated["merge"]["advisory"] == FIXTURE_ADVISORY
    assert evaluated["merge"]["totals"]["source_rows"] == fixture["totals"]["source_rows"]
    assert evaluated["merge"]["source_rows_derived_from"] == "g1_plan"


def _reduced_audit(tmp_path, drop_query_id=None, duplicate_query_id=None):
    """A verifier-valid audit whose room manifest lost -- or repeated -- a query."""
    import shutil

    source = os.path.dirname(_fixture_audit(tmp_path / "src_audit"))
    out_dir = str(tmp_path / "reduced")
    shutil.copytree(source, out_dir)
    report_path = os.path.join(out_dir, "geometry_audit_report.json")
    report = json.load(open(report_path))
    for room, entry in report["rooms"].items():
        path = os.path.join(out_dir, entry["candidate_manifest"])
        payload = json.load(open(path))
        queries = payload["queries"]
        if drop_query_id and any(q["query_id"] == drop_query_id for q in queries):
            payload["queries"] = [q for q in queries if q["query_id"] != drop_query_id]
        if duplicate_query_id and any(q["query_id"] == duplicate_query_id for q in queries):
            twin = next(q for q in queries if q["query_id"] == duplicate_query_id)
            payload["queries"] = queries + [dict(twin)]
        with open(path, "w") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        entry["candidate_manifest_sha256"] = mg.manifest_json_sha256(payload)
    with open(report_path, "w") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report_path


def test_a_g1_plan_that_lost_a_query_cannot_produce_canonical_metrics(tmp_path):
    """B2: only room NAMES were compared, so a 5,336-query plan looked complete."""
    fixture = build_fixture_run(tmp_path)
    dropped = fixture["records"][0]["query_id"]
    plan = me.load_audit_plan(_reduced_audit(tmp_path, drop_query_id=dropped))
    rows = mr.verify_rows(fixture["run_dir"], fixture["binding_sha256"])
    # the census still passes -- it never looked at the audit's query set
    assert mr.assert_census(rows, fixture["records"],
                            totals=fixture["totals"])["n_queries"] == 4
    with pytest.raises(ValueError, match="G1 audit does not cover exactly"):
        mr.assert_identity_join(plan, fixture["records"], rows)


def test_a_g1_plan_that_names_a_query_twice_is_refused(tmp_path):
    fixture = build_fixture_run(tmp_path)
    twin = fixture["records"][0]["query_id"]
    plan = me.load_audit_plan(_reduced_audit(tmp_path, duplicate_query_id=twin))
    with pytest.raises(ValueError, match="names .* twice"):
        mr.plan_query_identities(plan)


def test_the_identity_join_refuses_a_missing_or_misplaced_row(tmp_path):
    fixture = build_fixture_run(tmp_path)
    rows = mr.verify_rows(fixture["run_dir"], fixture["binding_sha256"])
    assert mr.assert_identity_join(fixture["plan"], fixture["records"],
                                   rows)["n_queries"] == 4
    with pytest.raises(ValueError, match="published rows does not cover exactly"):
        mr.assert_identity_join(fixture["plan"], fixture["records"], rows[:-1])
    moved = [dict(row) for row in rows]
    moved[0]["position"] = 99
    with pytest.raises(ValueError, match="identities disagree on room or stream position"):
        mr.assert_identity_join(fixture["plan"], fixture["records"], moved)


def test_the_shared_grid_oracle_rule_agrees_with_the_retrieval_controls(tmp_path):
    """One rule, two callers: r9b's QueryPlan form and the shared helper."""
    from src.localization import meshgrid_retrieval_control as rc

    fixture = build_fixture_run(tmp_path)
    room = me.load_room_plan(fixture["plan"], "A/A_idx_1")
    query = room.queries[0]
    truth = FIXTURE_QUERIES["A/A_idx_1"][0][4]
    assert rc.assert_grid_oracle(query, truth) == pytest.approx(
        mr.assert_grid_oracle(query.query_id, query.coordinates, query.oracle, truth))
    moved = [1.11, 1.11, 0.61]
    with pytest.raises(ValueError):
        rc.assert_grid_oracle(query, moved)
    with pytest.raises(ValueError):
        mr.assert_grid_oracle(query.query_id, query.coordinates, query.oracle, moved)


def _mirrored_truth(room_id="A/A_idx_1", index=0):
    """A truth reflected through its nearest candidate: SAME oracle, new place."""
    _position, _name, receiver, contexts, truth = FIXTURE_QUERIES[room_id][index]
    kept = mg.filter_query_candidates(FIXTURE_LATTICE, receiver=receiver,
                                      context_sources=contexts,
                                      z_band=mg.context_z_band(contexts))
    candidates = kept["candidates"]
    distances = np.linalg.norm(candidates - np.asarray(truth), axis=1)
    nearest = candidates[int(distances.argmin())]
    mirrored = (2.0 * nearest - np.asarray(truth, dtype=np.float64)).tolist()
    # the spoof is only interesting if the scalar oracle cannot see it
    assert abs(mg.grid_oracle_error(candidates, mirrored)
               - mg.grid_oracle_error(candidates, truth)) <= mr.ORACLE_TOLERANCE
    assert mirrored != list(truth)
    return mirrored


def _spoof_truth(fixture, room_id="A/A_idx_1", stem="S001_R002"):
    mirrored = _mirrored_truth(room_id)
    scene, scene_id = room_id.split("/")
    path = os.path.join(fixture["metadata_root"], scene, scene_id, f"{stem}.json")
    payload = json.load(open(path))
    payload["src_loc"] = mirrored
    with open(path, "w") as handle:
        json.dump(payload, handle)
    return mirrored


def test_the_scalar_oracle_cannot_see_a_mirrored_in_cell_truth(tmp_path):
    """B3: the r9 checks were rec_loc + a scalar, and both are non-injective."""
    fixture = build_fixture_run(tmp_path)
    honest = evaluate_unpinned(fixture)
    mirrored = _spoof_truth(fixture)
    spoofed = evaluate_unpinned(fixture)
    # the oracle gate is silent -- which is precisely why it is not the whole check
    assert spoofed["results"][0]["truth_xyz"] == pytest.approx(mirrored)
    assert spoofed["results"][0]["e_oracle"] == pytest.approx(honest["results"][0]["e_oracle"])
    # ... and the e_loc it would have published is a different number
    assert spoofed["results"][0]["by"]["lme"][8]["e_loc"] != \
        pytest.approx(honest["results"][0]["by"]["lme"][8]["e_loc"])


def test_the_truth_vector_check_refuses_the_mirrored_truth(tmp_path):
    fixture = build_fixture_run(tmp_path)
    provider = fixture_source_provider(fixture)
    assert evaluate_fixture(fixture, source_provider=provider)["truth_vector_checked"]
    _spoof_truth(fixture)
    with pytest.raises(ValueError, match="is not the one the query was held out from"):
        evaluate_fixture(fixture, source_provider=provider)


def test_a_pinned_metadata_bank_refuses_the_mirrored_truth(tmp_path):
    fixture = build_fixture_run(tmp_path)
    registered = evaluate_fixture(fixture)["metadata_bank"]["metadata_bank_sha256"]
    assert evaluate_fixture(fixture, expect_metadata_bank_sha256=registered
                            )["metadata_bank"]["pinned"] is True
    _spoof_truth(fixture)
    with pytest.raises(ValueError, match="not the registered ones"):
        evaluate_fixture(fixture, expect_metadata_bank_sha256=registered)


def test_a_canonical_report_requires_the_pre_registered_bank_digest(tmp_path):
    """B3 RULING 2: trust-on-first-use is not a canonical mode."""
    fixture = build_fixture_run(tmp_path)
    with pytest.raises(ValueError, match="requires the PRE-REGISTERED pair-metadata bank"):
        mr.evaluate_run(fixture["run_dir"], fixture["audit_report"],
                        fixture["context_manifest"], fixture["metadata_root"],
                        totals=fixture["totals"], require_manifest_census=False)
    bank = evaluate_unpinned(fixture)["metadata_bank"]
    assert len(bank["metadata_bank_sha256"]) == 64
    assert bank["pinned"] is False
    assert "PRE-REGISTRATION" in bank["note"]
    assert "--print-metadata-bank-digest" in bank["preregistration_note"]


def test_the_digest_cli_computes_the_value_a_canonical_run_needs(tmp_path):
    """The pre-registration entry point: no run, no results, just the tree."""
    fixture = build_fixture_run(tmp_path)
    verdict = mr.compute_metadata_bank_digest(fixture["context_manifest"],
                                              fixture["metadata_root"],
                                              require_manifest_census=False)
    assert verdict["metadata_bank_sha256"] == fixture["metadata_bank_sha256"]
    assert verdict["n_pair_files"] == verdict["n_records"] == fixture["totals"]["queries"]
    # deterministic, and it is exactly what evaluate_run then demands back
    assert mr.compute_metadata_bank_digest(
        fixture["context_manifest"], fixture["metadata_root"],
        require_manifest_census=False)["metadata_bank_sha256"] == \
        verdict["metadata_bank_sha256"]
    assert evaluate_fixture(fixture)["metadata_bank"]["pinned"] is True

    # and an edit to the tree after registration changes it
    _spoof_truth(fixture)
    assert mr.compute_metadata_bank_digest(
        fixture["context_manifest"], fixture["metadata_root"],
        require_manifest_census=False)["metadata_bank_sha256"] != \
        verdict["metadata_bank_sha256"]


def test_the_digest_mode_needs_no_run_directory():
    args = mr.parse_args(["--print-metadata-bank-digest"])
    assert args.run_dir is None and args.out_dir is None
    assert mr.validate_args(args) is True
    # ... while a report still requires both
    with pytest.raises(SystemExit, match="--run-dir is required"):
        mr.validate_args(mr.parse_args(["--out-dir", "o", "--non-canonical"]))
    with pytest.raises(SystemExit, match="--out-dir is required"):
        mr.validate_args(mr.parse_args(["--run-dir", "r", "--non-canonical"]))


def test_the_cli_refuses_a_canonical_run_without_the_pre_registered_digest():
    with pytest.raises(SystemExit, match="requires the PRE-REGISTERED"):
        mr.validate_args(mr.parse_args(["--run-dir", "r", "--out-dir", "o"]))
    assert mr.validate_args(mr.parse_args(
        ["--run-dir", "r", "--out-dir", "o", "--non-canonical"])) is True
    assert mr.validate_args(mr.parse_args(
        ["--run-dir", "r", "--out-dir", "o",
         "--expect-metadata-bank-sha256", "a" * 64])) is True


def test_the_truth_vector_check_is_the_injective_one():
    assert mr.assert_truth_vector("q", [1.0, 2.0, 3.0], [1.0, 1.0, 1.0],
                                  [0.0, 1.0, 2.0]) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="is not the one the query was held out from"):
        mr.assert_truth_vector("q", [1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [0.0, 1.0, 2.5])


def test_the_binding_must_be_the_registered_protocol(tmp_path):
    """M6: row == binding was proven; binding == REGISTERED never was."""
    fixture = build_fixture_run(tmp_path)
    assert mr.assert_registered_protocol(fixture["binding"])["is_registered"] is True
    for field, value in (("tau", 0.2), ("seed", 7), ("num_samples", 4),
                         ("noise_policy", "per_candidate"), ("steps", 4),
                         ("cfg_scale", 3.0), ("scorer_readout", "sample"),
                         ("cond_autocast", "off"), ("k_prefixes", [1, 2, 4]),
                         ("agree_ckpt_sha256", "0" * 64),
                         ("model_config_sha256", "0" * 64)):
        with pytest.raises(ValueError, match="not the registered protocol"):
            mr.assert_registered_protocol(dict(fixture["binding"], **{field: value}))
        verdict = mr.assert_registered_protocol(dict(fixture["binding"], **{field: value}),
                                                allow_deviation=True)
        assert verdict["is_registered"] is False
        assert field in verdict["deviations"]


def test_the_checkpoint_must_be_one_of_the_registered_admissible_arms(tmp_path):
    """M6 RULING 1: r9c let ANY checkpoint count as registered."""
    fixture = build_fixture_run(tmp_path)
    verdict = mr.assert_registered_protocol(fixture["binding"])
    assert verdict["arm"] == "P1"
    assert verdict["ckpt_sha256_pinned"] is True
    assert verdict["registered_arms"] == mr.REGISTERED_CKPT_SHA256
    # every admissible arm passes ...
    for arm, digest in mr.REGISTERED_CKPT_SHA256.items():
        assert mr.assert_registered_protocol(
            dict(fixture["binding"], ckpt_sha256=digest))["arm"] == arm
    # ... and nothing else does
    with pytest.raises(ValueError, match="ckpt_sha256"):
        mr.assert_registered_protocol(dict(fixture["binding"], ckpt_sha256="b" * 64))
    relaxed = mr.assert_registered_protocol(dict(fixture["binding"], ckpt_sha256="b" * 64),
                                            allow_deviation=True)
    assert relaxed["arm"] is None and relaxed["is_registered"] is False
    # a narrower expectation can still single out one arm
    with pytest.raises(ValueError, match="ckpt_sha256"):
        mr.assert_registered_protocol(fixture["binding"],
                                      expect_ckpt_sha256=mr.REGISTERED_CKPT_SHA256["BF"])


def test_the_registered_arm_registry_is_the_real_exp20_checkpoints():
    """The pinned digests must still BE the files on disk (Planner RULING 1)."""
    for arm, digest in mr.REGISTERED_CKPT_SHA256.items():
        path = os.path.join("weights", "exp20", f"{arm}_40k.ckpt")
        if not os.path.isfile(path):
            pytest.skip(f"{path} is not on this machine")
        assert me.file_sha256(path) == digest
    # P1 is the checkpoint the published P1 binding names
    assert mr.REGISTERED_CKPT_SHA256["P1"] == \
        "c4c678826cddda37fa4977926aadee530afd037b3abb110918b52a342ce9845c"


def test_a_deviating_run_refuses_and_then_stamps_itself_a_sensitivity_check(tmp_path):
    fixture = build_fixture_run(tmp_path)
    with pytest.raises(ValueError, match="not the registered protocol"):
        mr.assert_registered_protocol(dict(fixture["binding"], tau=0.25))
    evaluated = evaluate_fixture(fixture)
    report = mr.build_report(evaluated, fixture["run_dir"], fixture["audit_report"],
                             fixture["context_manifest"], fixture["metadata_root"],
                             n_boot=128, bootstrap_seed=7)
    assert report["protocol"]["bootstrap"]["is_registered"] is False
    markdown = mr.render_markdown(report)
    assert "SENSITIVITY CHECK, not the registered bootstrap" in markdown


def test_deviating_seeds_refuse_at_the_cli_unless_declared():
    base = ["--run-dir", "r", "--out-dir", "o", "--non-canonical"]
    with pytest.raises(SystemExit, match="not the pre-registered settings"):
        mr.validate_args(mr.parse_args(base + ["--n-boot", "10"]))
    with pytest.raises(SystemExit, match="not the pre-registered settings"):
        mr.validate_args(mr.parse_args(base + ["--baseline-seeds", "1", "2"]))
    assert mr.validate_args(mr.parse_args(
        base + ["--n-boot", "10", "--allow-protocol-deviation"])) is True


def test_a_deviating_baseline_stops_calling_itself_pre_registered():
    results = _synthetic_results({"A/A_idx_1": [1.0, 2.0], "B/B_idx_2": [3.0]})
    for index, result in enumerate(results):
        result["baseline_e_loc"] = {seed: 1.0 + index for seed in (7, 8)}
    report = mr.baseline_report(results, seeds=(7, 8), bootstrap={"n": 32})
    assert report["seeds_are_registered"] is False
    assert "SENSITIVITY CHECK" in report["rule"]
    assert report["registered_seeds"] == list(mr.RANDOM_BASELINE_SEEDS)


def test_the_published_report_carries_the_new_gates_and_banners(tmp_path):
    fixture = build_fixture_run(tmp_path)
    evaluated = evaluate_fixture(fixture, source_provider=fixture_source_provider(fixture))
    # the REGISTERED bootstrap, so nothing in this report is a sensitivity check
    report = mr.build_report(evaluated, fixture["run_dir"], fixture["audit_report"],
                             fixture["context_manifest"], fixture["metadata_root"])
    gates = report["gates"]
    assert gates["supplied_artifacts_match_the_binding_hashes"] is True
    assert gates["binding_matches_the_registered_protocol"] is True
    assert gates["merge_report_gates_applied"] is True
    assert gates["sidecar_dtype_and_float16_bound_checked"] is True
    assert gates["truth_vector_checked_against_the_loader"] is True
    assert gates["d1_g1_rows_identity_join"]["n_queries"] == 4
    assert report["provenance"]["metadata_bank"]["metadata_bank_sha256"]
    assert report["provenance"]["merge"]["source_rows_derived_from"] == "g1_plan"

    markdown = mr.render_markdown(report)
    assert "mean e_excess (m)" in markdown             # the omitted model column
    assert markdown.count("success@1.0") >= 2          # model AND baseline tables
    assert "Latency — room-first" in markdown
    assert "NON-CANONICAL" not in markdown             # every gate was met
    assert report["canonical_status"]["canonical"] is True
    assert report["canonical_status"]["reasons"] == []


def test_the_case_file_and_the_probe_outputs_carry_the_latency_scope(tmp_path):
    fixture = build_fixture_run(tmp_path)
    evaluated = evaluate_fixture(fixture)
    report = mr.build_report(evaluated, fixture["run_dir"], fixture["audit_report"],
                             fixture["context_manifest"], fixture["metadata_root"], n_boot=128)
    cases = mr.build_case_payload(report["visualization_cases"], evaluated["rows_by_id"],
                                  evaluated["plans_by_id"],
                                  {r["query_id"]: r for r in evaluated["results"]},
                                  fixture["run_dir"], evaluated["plan"])
    assert cases["latency_scope_note"] == mr.LATENCY_SCOPE_NOTE
    assert cases["controls_elsewhere"] == mr.CONTROLS_ELSEWHERE
    assert cases["sims_precision_caveat"] == me.SIMS_PRECISION_CAVEAT


def test_the_registered_artifact_digests_are_the_ones_on_disk():
    """The pinned model-config digest must still BE the tracked config's.

    §1.4 pins f3eafef4...; if the config were edited the pin would silently start
    refusing every real run, so the literal is checked against the repository
    rather than trusted.
    """
    assert mr.REGISTERED_ARTIFACT_SHA256["model_config_sha256"] == \
        me.file_sha256(os.path.join("src", "configs", "model_configs", "FLAC", "AR",
                                    "FLAC_AR.json"))
    # the AGREE pin is §1.4's literal; the weights are not in the repository, so
    # it is asserted as a REGISTERED CONSTANT and joined to a run by the binding
    assert mr.REGISTERED_ARTIFACT_SHA256["agree_ckpt_sha256"] == \
        "3a13243d6c6a11082697592c2c5db84790d37859451df2963eb51d655b23c787"
    assert "ckpt_sha256" not in mr.REGISTERED_ARTIFACT_SHA256


# --------------------------------------------------------------------------- #
# r9d: the residuals the consolidated Codex r9c re-review left open
# --------------------------------------------------------------------------- #
def test_the_merge_receipt_is_re_derived_from_the_rows_not_believed(tmp_path):
    """B1: a receipt copies, so every number in it is rebuilt from the rows."""
    fixture = build_fixture_run(tmp_path)
    rows = mr.verify_rows(fixture["run_dir"], fixture["binding_sha256"])
    derived = mr.derive_run_facts(rows)
    assert derived["candidate_query_pairs"] == fixture["totals"]["candidate_query_pairs"]
    assert derived["generated_waveforms"] == fixture["totals"]["generated_waveforms"]
    # the row-derived source-row census is the per-receiver union of the rows'
    # OWN candidate index lists, and it must equal the G1 plan's derivation
    assert derived["source_rows"] == fixture["totals"]["source_rows"]
    assert derived["source_rows"] == mr.plan_source_rows(fixture["plan"])
    assert derived["n_receivers"] >= 1

    evaluated = evaluate_fixture(fixture)
    assert evaluated["merge"]["receipt_cross_checked_against_rows"] is True
    assert evaluated["merge"]["source_rows_from_rows"] == derived["source_rows"]
    assert evaluated["merge"]["source_rows_from_g1_plan"] == derived["source_rows"]


def test_a_receipt_that_the_rows_do_not_support_is_refused(tmp_path):
    """A hand-assembled directory can carry a genuine-looking receipt."""
    fixture = build_fixture_run(tmp_path)
    path = os.path.join(fixture["run_dir"], "merge_report.json")
    pristine = json.load(open(path))
    for field, pattern in (("source_rows", "source_rows"),
                           ("candidate_query_pairs", "candidate_query_pairs"),
                           ("generated_waveforms", "generated_waveforms")):
        totals = dict(pristine["totals"])
        totals[field] = int(totals[field]) + 1
        me.write_json(path, dict(pristine, totals=totals))
        # the census against the REGISTERED totals fires first for some fields,
        # so accept either refusal as long as the field is named
        with pytest.raises(ValueError, match=pattern):
            evaluate_fixture(fixture, totals=dict(fixture["totals"], **{field: totals[field]}))
    me.write_json(path, pristine)
    assert evaluate_fixture(fixture)["merge"]["n_rows"] == 4


def test_mixed_effective_batching_is_refused_however_good_the_receipt(tmp_path):
    """B1: a directory stitched from shards run at different batch_rows."""
    fixture = build_fixture_run(tmp_path)
    rows = mr.verify_rows(fixture["run_dir"], fixture["binding_sha256"])
    assert mr.assert_uniform_batching(rows, FIXTURE_ADVISORY)["batching"] == FIXTURE_ADVISORY

    mixed = [dict(row) for row in rows]
    mixed[0]["batching"] = {"batch_rows": 512, "source_chunk": 2}
    with pytest.raises(ValueError, match="different batchings"):
        mr.assert_uniform_batching(mixed, FIXTURE_ADVISORY)
    # uniform, but not the batching the run pins
    other = [dict(row, batching={"batch_rows": 512, "source_chunk": 2}) for row in rows]
    with pytest.raises(ValueError, match="not the ones the pass ran at"):
        mr.assert_uniform_batching(other, FIXTURE_ADVISORY)


def test_a_row_stamped_with_another_batching_is_caught_end_to_end(tmp_path):
    fixture = build_fixture_run(tmp_path)
    path = me.query_artifact_paths(fixture["run_dir"], "A/A_idx_1", 0)["row"]
    row = json.load(open(path))
    row["batching"] = {"batch_rows": 512, "source_chunk": 2}
    row["row_sha256"] = me.row_digest(row)               # re-signed, so digests pass
    me.write_json(path, row)
    assert me.verify_query_artifact(path, binding_sha256=fixture["binding_sha256"])["ok"]
    with pytest.raises(ValueError, match="different batchings"):
        evaluate_fixture(fixture)


def test_the_float16_bound_uses_both_adjacent_representables(tmp_path):
    """M7: np.spacing follows the away-from-zero gap, which is the SMALLER one
    at a negative binade boundary, halving the bound and refusing honest
    roundoff (Codex r9c review, M7)."""
    boundary = _f16([[-0.5, -0.5]])
    naive = 0.5 * float(np.abs(np.spacing(np.asarray(boundary))).max())
    honest = mr.float16_half_ulp(boundary)
    assert honest == pytest.approx(2.0 * naive)          # exactly twice
    assert honest == pytest.approx(
        0.5 * abs(float(np.float16(-0.5) - np.nextafter(np.float16(-0.5),
                                                        np.float16(-np.inf)))))
    # the positive boundary was already right, and stays right
    assert mr.float16_half_ulp(_f16([[0.5, 0.5]])) == pytest.approx(honest)


def test_a_negative_binade_boundary_no_longer_refuses_honest_roundoff():
    """The regression the naive bound would have caused, end to end."""
    coordinates = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    # values just below -0.5 quantize to -0.5, moving by ~3.7e-4 -- inside the
    # true half-ulp of 2.44e-4 + slack, but ABOVE the toward-zero half-ulp
    stored = [[-0.50018, -0.50018], [-0.26, -0.26]]
    sidecar = _f16(stored)
    assert float(sidecar[0, 0]) == -0.5
    row = _hand_row(sidecar, coordinates, scored_from=stored)
    row["e_oracle"] = 0.0
    naive = 0.5 * float(np.abs(np.spacing(np.asarray(sidecar))).max()) \
        + mr.SIDECAR_FLOAT32_SLACK
    honest = mr.float16_quantization_bound(sidecar)
    assert honest > naive
    # with the honest bound this is accepted; with the naive one it would refuse
    entry = mr.evaluate_query(row, sidecar, coordinates, [0.0, 0.0, 0.0])["sidecar"]["lme"][2]
    assert entry["within_float16_bound"] is True
    assert entry["max_abs_delta"] <= honest


def test_latency_is_never_a_clean_canonical_block_when_rows_were_dropped():
    """M8: exclusions can be selective, so the endpoint says so itself."""
    complete = _timed(_synthetic_results({"A/A_idx_1": [1.0, 2.0],
                                          "B/B_idx_2": [3.0, 4.0]}))
    for index, result in enumerate(complete):
        result["position"] = index
    clean = mr.latency_report(complete, n=64)
    assert clean["canonical"] is True
    assert clean["non_canonical_note"] is None
    assert clean["n_incomplete"] == 0

    complete[0]["latency_s"].pop("decode")
    dirty = mr.latency_report(complete, n=64)
    assert dirty["canonical"] is False
    assert "NON-CANONICAL LATENCY" in dirty["non_canonical_note"]
    assert dirty["n_rows_offered"] == 4 and dirty["n_queries"] == 3
    # named per component AND per room (r9d M8)
    assert dirty["missing_components"]["decode"]["by_room"] == {"A/A_idx_1": 1}


def test_a_room_that_loses_every_row_is_named_not_silently_dropped():
    results = _timed(_synthetic_results({"A/A_idx_1": [1.0], "B/B_idx_2": [2.0, 3.0]}))
    for index, result in enumerate(results):
        result["position"] = index
    results[0]["latency_s"].pop("embed")                 # room A's only row
    report = mr.latency_report(results, n=64)
    assert report["canonical"] is False
    assert report["rooms_without_a_complete_row"] == ["A/A_idx_1"]
    assert sorted(report["per_room"]) == ["B/B_idx_2"]
    assert report["missing_components"]["embed"]["by_room"] == {"A/A_idx_1": 1}


def test_the_canonical_status_names_every_reason_it_is_not_canonical(tmp_path):
    fixture = build_fixture_run(tmp_path)
    evaluated = evaluate_fixture(fixture)
    report = mr.build_report(evaluated, fixture["run_dir"], fixture["audit_report"],
                             fixture["context_manifest"], fixture["metadata_root"])
    assert report["canonical_status"]["canonical"] is True

    # each relaxation adds exactly one named reason
    unpinned = evaluate_unpinned(fixture)
    status = mr.canonical_status(unpinned)
    assert [reason["gate"] for reason in status["reasons"]] == ["metadata_bank"]
    assert status["canonical"] is False
    assert "NON-CANONICAL" in status["note"]

    shard = build_fixture_run(tmp_path / "shard", merged=False)
    shard_status = mr.canonical_status(evaluate_fixture(shard))
    assert "merge_report" in [reason["gate"] for reason in shard_status["reasons"]]


def test_a_non_canonical_report_says_so_at_the_top_of_its_markdown(tmp_path):
    fixture = build_fixture_run(tmp_path)
    evaluated = evaluate_unpinned(fixture)
    report = mr.build_report(evaluated, fixture["run_dir"], fixture["audit_report"],
                             fixture["context_manifest"], fixture["metadata_root"], n_boot=128)
    assert report["canonical_status"]["canonical"] is False
    markdown = mr.render_markdown(report)
    header = markdown.split("## ")[0]
    assert "NON-CANONICAL" in header
    assert "metadata_bank" in header
    assert "PRE-REGISTERED" in markdown


def test_a_non_canonical_latency_block_is_labelled_in_the_markdown(tmp_path):
    fixture = build_fixture_run(tmp_path)
    evaluated = evaluate_fixture(fixture)
    evaluated["results"][0]["latency_s"].pop("decode")
    report = mr.build_report(evaluated, fixture["run_dir"], fixture["audit_report"],
                             fixture["context_manifest"], fixture["metadata_root"], n_boot=128)
    assert report["latency"]["canonical"] is False
    assert "latency_completeness" in [reason["gate"]
                                      for reason in report["canonical_status"]["reasons"]]
    markdown = mr.render_markdown(report)
    assert "## Latency — room-first (NON-CANONICAL)" in markdown
    assert "| component | rows missing it | by room |" in markdown


def test_the_dataset_config_identity_is_part_of_the_registered_set(tmp_path):
    """M6: the observed-RIR loader is built from it, so it decides every score."""
    fixture = build_fixture_run(tmp_path)
    assert "dataset_config_sha256" in mr.REGISTERED_ARTIFACT_SHA256
    assert mr.REGISTERED_ARTIFACT_SHA256["dataset_config_sha256"] == \
        me.file_sha256(os.path.join("src", "configs", "dataset_configs", "AR", "eval",
                                    "acousticroom_unseeneval.json"))
    with pytest.raises(ValueError, match="dataset_config_sha256"):
        mr.assert_registered_protocol(dict(fixture["binding"],
                                           dataset_config_sha256="0" * 64))


# --------------------------------------------------------------------------- #
# r9g: the residuals the Codex r9f verify pass left open
# --------------------------------------------------------------------------- #
def test_a_stripped_batching_stamp_is_refused_not_skipped(tmp_path):
    """B1 residual: `if found` let an empty stamp through untouched."""
    fixture = build_fixture_run(tmp_path)
    rows = mr.verify_rows(fixture["run_dir"], fixture["binding_sha256"])
    assert mr.assert_uniform_batching(rows, FIXTURE_ADVISORY)["batching"] == FIXTURE_ADVISORY

    for stripped in ({}, None):
        edited = [dict(row) for row in rows]
        edited[0]["batching"] = stripped
        with pytest.raises(ValueError, match="no complete batching stamp"):
            mr.assert_uniform_batching(edited, FIXTURE_ADVISORY)


def test_a_partial_batching_stamp_is_refused(tmp_path):
    """B1 residual: comparing only the keys present let source_chunk vanish."""
    fixture = build_fixture_run(tmp_path)
    rows = mr.verify_rows(fixture["run_dir"], fixture["binding_sha256"])
    for missing in me.RUN_BINDING_ADVISORY:
        partial = {key: value for key, value in FIXTURE_ADVISORY.items() if key != missing}
        edited = [dict(row, batching=dict(partial)) for row in rows]
        with pytest.raises(ValueError, match="no complete batching stamp"):
            mr.assert_uniform_batching(edited, FIXTURE_ADVISORY)
    # an extra key is not a valid stamp either
    edited = [dict(row, batching=dict(FIXTURE_ADVISORY, extra=1)) for row in rows]
    with pytest.raises(ValueError, match="no complete batching stamp"):
        mr.assert_uniform_batching(edited, FIXTURE_ADVISORY)


def test_a_re_signed_row_with_a_stripped_stamp_cannot_canonicalize(tmp_path):
    """The exploit the r9f review named, end to end."""
    fixture = build_fixture_run(tmp_path)
    path = me.query_artifact_paths(fixture["run_dir"], "A/A_idx_1", 0)["row"]
    row = json.load(open(path))
    row["batching"] = {}                                 # stripped, then re-signed
    row["row_sha256"] = me.row_digest(row)
    me.write_json(path, row)
    assert me.verify_query_artifact(path, binding_sha256=fixture["binding_sha256"])["ok"]
    with pytest.raises(ValueError, match="no complete batching stamp"):
        evaluate_fixture(fixture)


def test_a_run_that_does_not_pin_its_advisory_batching_is_refused(tmp_path):
    fixture = build_fixture_run(tmp_path)
    rows = mr.verify_rows(fixture["run_dir"], fixture["binding_sha256"])
    with pytest.raises(ValueError, match="does not pin the advisory batching"):
        mr.assert_uniform_batching(rows, {"batch_rows": 8, "source_chunk": None})
    with pytest.raises(ValueError, match="does not pin the advisory batching"):
        mr.assert_uniform_batching(rows, None)


def test_the_bank_digest_can_reuse_records_a_caller_already_loaded(tmp_path):
    fixture = build_fixture_run(tmp_path)
    from_file = mr.compute_metadata_bank_digest(fixture["context_manifest"],
                                                fixture["metadata_root"],
                                                require_manifest_census=False)
    from_records = mr.compute_metadata_bank_digest(fixture["context_manifest"],
                                                   fixture["metadata_root"],
                                                   records=fixture["records"])
    assert from_records["metadata_bank_sha256"] == from_file["metadata_bank_sha256"]
    assert from_records["n_pair_files"] == from_file["n_pair_files"]


# --------------------------------------------------------------------------- #
# r9j: the last items from the Codex r9i verify pass
# --------------------------------------------------------------------------- #
def test_the_truth_is_parsed_out_of_the_very_bytes_that_were_hashed(tmp_path):
    """Item 1: hashing one read and parsing another leaves a swap window."""
    fixture = build_fixture_run(tmp_path)
    resolver = mr.TruthResolver(fixture["metadata_root"])
    record = fixture["records"][0]

    opened = []
    real_open = open

    def _spy(path, *args, **kwargs):
        if str(path).endswith(".json") and "metadata" in str(path):
            opened.append(str(path))
        return real_open(path, *args, **kwargs)

    import builtins
    builtins.open = _spy
    try:
        receiver, truth = resolver.resolve(record)
    finally:
        builtins.open = real_open

    # ONE read of the pair file: the digest and the coordinates share a buffer,
    # so no swap can sit between them
    assert len(opened) == 1, opened
    assert resolver.pair_files[record["query_id"]]["sha256"] == \
        me.file_sha256(opened[0])
    assert truth.tolist() == list(FIXTURE_QUERIES[record["room_id"]][0][4])
    assert receiver.tolist() == list(FIXTURE_QUERIES[record["room_id"]][0][2])


def test_the_recorded_digest_is_of_the_bytes_that_produced_the_truth(tmp_path):
    """A reader that returns different bytes on a second read cannot split them."""
    fixture = build_fixture_run(tmp_path)
    record = fixture["records"][0]
    scene, scene_id = record["room_id"].split("/")
    path = os.path.join(fixture["metadata_root"], scene, scene_id, "S001_R002.json")
    honest = json.load(open(path))
    mirrored = _mirrored_truth(record["room_id"])

    calls = {"n": 0}
    real_open = open
    import builtins

    def _swapping(target, *args, **kwargs):
        # a reader that serves the honest bytes first and the mirrored bytes
        # second -- the r9i attack. With one read it can only serve one.
        if str(target) == path:
            calls["n"] += 1
            payload = honest if calls["n"] == 1 else dict(honest, src_loc=mirrored)
            import io
            data = json.dumps(payload).encode()
            return io.BytesIO(data) if "b" in (args[0] if args else kwargs.get("mode", "r")) \
                else io.StringIO(data.decode())
        return real_open(target, *args, **kwargs)

    builtins.open = _swapping
    try:
        resolver = mr.TruthResolver(fixture["metadata_root"])
        _receiver, truth = resolver.resolve(record)
    finally:
        builtins.open = real_open

    assert calls["n"] == 1                       # only one chance to serve bytes
    digest = resolver.pair_files[record["query_id"]]["sha256"]
    # the digest IS of the bytes the truth came out of
    served = json.dumps(honest).encode()
    import hashlib as _h
    assert digest == _h.sha256(served).hexdigest()
    assert truth.tolist() == honest["src_loc"]


def test_an_unparseable_pair_file_is_refused_by_the_single_read_path(tmp_path):
    fixture = build_fixture_run(tmp_path)
    record = fixture["records"][0]
    scene, scene_id = record["room_id"].split("/")
    path = os.path.join(fixture["metadata_root"], scene, scene_id, "S001_R002.json")
    with open(path, "wb") as handle:
        handle.write(b"\xff\xfe not json at all")
    with pytest.raises(ValueError, match="not readable as JSON"):
        mr.TruthResolver(fixture["metadata_root"]).resolve(record)


# --------------------------------------------------------------------------- #
# r9m: verify-and-parse from one buffer, all the way through the metrics
# --------------------------------------------------------------------------- #
class _ReadSpy:
    """Counts opens of the paths it is told to watch, by real path."""

    def __init__(self, paths):
        self.wanted = {os.path.realpath(str(p)) for p in paths}
        self.opened = []
        self._real = open

    def __enter__(self):
        import builtins

        def _spy(path, *args, **kwargs):
            try:
                real = os.path.realpath(str(path))
            except TypeError:                        # a file descriptor, not a path
                real = None
            if real in self.wanted:
                self.opened.append(real)
            return self._real(path, *args, **kwargs)

        builtins.open = _spy
        return self

    def __exit__(self, *exc):
        import builtins

        builtins.open = self._real
        return False

    def count(self, path):
        return self.opened.count(os.path.realpath(str(path)))


def _artifact_paths(fixture):
    paths = []
    for row in mr.verify_rows(fixture["run_dir"], fixture["binding_sha256"]):
        entry = me.query_artifact_paths(fixture["run_dir"], row["room_id"], row["position"])
        paths += [entry["row"], entry["sims"]]
    return paths


def test_a_full_evaluation_reads_every_row_and_sidecar_exactly_once(tmp_path):
    """Item 3: r9j verified a row, reopened it, then reopened its sidecar."""
    fixture = build_fixture_run(tmp_path)
    paths = _artifact_paths(fixture)
    assert len(paths) == 2 * fixture["totals"]["queries"]

    with _ReadSpy(paths) as spy:
        evaluated = evaluate_fixture(fixture)
    assert len(evaluated["results"]) == fixture["totals"]["queries"]
    for path in paths:
        assert spy.count(path) == 1, f"{path} was opened {spy.count(path)} times"


def test_the_verified_row_and_sidecar_are_the_objects_the_metrics_consume(tmp_path):
    fixture = build_fixture_run(tmp_path)
    rows, sims, snapshot = mr.verify_rows_with_sidecars(fixture["run_dir"],
                                                        fixture["binding_sha256"])
    assert sorted(sims) == sorted(row["query_id"] for row in rows)
    assert sorted(snapshot) == sorted(sims)
    for query_id, entry in snapshot.items():
        assert len(entry["row_bytes_sha256"]) == len(entry["sims_bytes_sha256"]) == 64
        assert entry["row_bytes_sha256"] == me.file_sha256(entry["row_path"])
    for row in rows:
        block = sims[row["query_id"]]
        assert list(block.shape) == list(row["sims_shape"])
        assert block.dtype == np.dtype(me.SIMS_DTYPE)

    evaluated = evaluate_fixture(fixture)
    # the very arrays the verification parsed are what evaluate_run carried
    for query_id, block in evaluated["sims_by_id"].items():
        assert np.array_equal(block, sims[query_id])
    assert "read EXACTLY ONCE" in evaluated["single_read_note"]


def test_a_coordinated_row_and_sidecar_substitution_cannot_be_staged(tmp_path):
    """The exploit: verify the honest pair, then serve a coherent forged pair."""
    fixture = build_fixture_run(tmp_path)
    entry = me.query_artifact_paths(fixture["run_dir"], "A/A_idx_1", 0)
    honest_row = open(entry["row"], "rb").read()
    honest_sims = open(entry["sims"], "rb").read()

    calls = {"row": 0, "sims": 0}
    real_open = open
    import builtins

    def _swapping(path, *args, **kwargs):
        # a reader that serves the honest artifacts first and forged ones after:
        # with one read each it never gets the second chance
        target = os.path.realpath(str(path)) if isinstance(path, (str, bytes, os.PathLike)) \
            else None
        if target == os.path.realpath(entry["row"]):
            calls["row"] += 1
        elif target == os.path.realpath(entry["sims"]):
            calls["sims"] += 1
        return real_open(path, *args, **kwargs)

    builtins.open = _swapping
    try:
        evaluate_fixture(fixture)
    finally:
        builtins.open = real_open
    assert calls == {"row": 1, "sims": 1}
    # the bytes on disk are untouched, and they are the ones that were used
    assert open(entry["row"], "rb").read() == honest_row
    assert open(entry["sims"], "rb").read() == honest_sims


def test_the_single_buffer_verdict_agrees_with_the_engines_own(tmp_path):
    """Cross-pin: the report's reader and the engine's verifier, one verdict."""
    fixture = build_fixture_run(tmp_path)
    entry = me.query_artifact_paths(fixture["run_dir"], "A/A_idx_1", 0)
    sha = fixture["binding_sha256"]
    pristine_row = open(entry["row"], "rb").read()
    pristine_sims = open(entry["sims"], "rb").read()

    def _verdicts():
        return (mr.read_verified_query_artifact(entry["row"], binding_sha256=sha),
                me.verify_query_artifact(entry["row"], binding_sha256=sha))

    mine, theirs = _verdicts()
    assert mine["ok"] is theirs["ok"] is True
    assert mine["query_id"] == theirs["query_id"]

    def _edit_oracle():
        row = json.loads(pristine_row)
        row["e_oracle"] = 0.0
        me.write_json(entry["row"], row)

    def _edit_binding():
        row = json.loads(pristine_row)
        row["binding_sha256"] = "0" * 64
        row["row_sha256"] = me.row_digest(row)
        me.write_json(entry["row"], row)

    def _append_to_sims():
        with open(entry["sims"], "ab") as handle:
            handle.write(b"\x00")

    def _reshape_sims():
        np.save(entry["sims"], np.zeros((1, 1), dtype=np.float16))

    def _drop_sims():
        os.remove(entry["sims"])

    for name, mutate in (("edited row", _edit_oracle),
                         ("foreign binding", _edit_binding),
                         ("appended sidecar", _append_to_sims),
                         ("reshaped sidecar", _reshape_sims),
                         ("missing sidecar", _drop_sims)):
        mutate()
        mine, theirs = _verdicts()
        assert mine["ok"] is False and theirs["ok"] is False, name
        # the SAME reason, so the two readers cannot drift apart
        assert mine["reason"] == theirs["reason"], name
        with open(entry["row"], "wb") as handle:
            handle.write(pristine_row)
        with open(entry["sims"], "wb") as handle:
            handle.write(pristine_sims)

    mine, theirs = _verdicts()
    assert mine["ok"] is theirs["ok"] is True          # the restore is complete


def test_the_row_plan_join_agrees_with_the_engines_own(tmp_path):
    """Cross-pin: assert_row_matches_plan and assert_published_matches."""
    fixture = build_fixture_run(tmp_path)
    room = me.load_room_plan(fixture["plan"], "A/A_idx_1")
    query = room.queries[0]
    rows = {row["query_id"]: row
            for row in mr.verify_rows(fixture["run_dir"], fixture["binding_sha256"])}
    assert mr.assert_row_matches_plan(rows[query.query_id], query) is True
    assert me.assert_published_matches(fixture["run_dir"], query,
                                       binding_sha256=fixture["binding_sha256"]) is True

    other = next(q for q in room.queries if q.query_id != query.query_id)
    with pytest.raises(ValueError, match="does not match the candidate manifest"):
        mr.assert_row_matches_plan(rows[query.query_id], other)


def test_the_resolver_can_be_held_to_the_pre_registered_pair_digests(tmp_path):
    """Item 1's shared half: a pair file the bank never covered supplies nothing."""
    fixture = build_fixture_run(tmp_path)
    bank = mr.compute_metadata_bank_digest(fixture["context_manifest"],
                                           fixture["metadata_root"],
                                           require_manifest_census=False)
    expected = {qid: entry["sha256"] for qid, entry in bank["queries"].items()}
    assert sorted(expected) == sorted(record["query_id"] for record in fixture["records"])

    resolver = mr.TruthResolver(fixture["metadata_root"], expected=expected)
    for record in fixture["records"]:
        resolver.resolve(record)

    # an edit after the freeze is refused, on the buffer the truth would come from
    _spoof_truth(fixture)
    with pytest.raises(ValueError, match="the truth being read is not the registered one"):
        mr.TruthResolver(fixture["metadata_root"],
                         expected=expected).resolve(fixture["records"][0])
    # ... and a query the bank never covered cannot supply one either
    with pytest.raises(ValueError, match="does not cover this query"):
        mr.TruthResolver(fixture["metadata_root"], expected={}).resolve(fixture["records"][0])
