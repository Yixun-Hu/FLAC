"""exp_37 fix round (Codex BLOCKING 1 / HIGH 3): an accuracy number comes only from a run whose scored waveforms exist.

``exp37_sources.accuracy_source`` accepts an accuracy-only run with its registered inline-dump identity and its own
dump (bound to its manifest, covering exactly its queries, each query record naming its file), or a TIMING pass with
its artifact replay (registered replay identity bound to the timing manifest, its dump, per query the stored arrays
bitwise equal); it returns the inventory entry (manifest / dump / per-query artifact sha256). CPU, synthetic runs.
Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 1.
"""
import shutil

import numpy as np
import pytest

from exp37_fakes import example_identity
from src.localization import exp37_grid as grid
from src.localization import exp37_identity as ident
from src.localization import exp37_sources as sources
from src.localization import exp37_waveforms as wav
from src.localization.pilot import canonical_sha256
from src.localization.runner import initialize_run, save_query_result

AUDIT = {"sha256": "a" * 64, "mesh_root": "/m", "rooms": {"R": {"mesh_sha256": "b" * 64}}}


def _pilot(counts):
    records = [{"index": i, "query_id": f"q{i}.wav", "room": "r", "candidate_count": c, "candidate_indices_sha256": f"h{i}"}
               for i, c in counts.items()]
    body = {"schema_version": 1, "query_count": len(records), "records": records}
    return {**body, "sha256": canonical_sha256(body)}


def make_run(tmp_path, name, arm, pilot, *, timing, seed=42, replay_of=None, nudge=None, **modes):
    """A synthetic run as the runner writes it; an untimed one dumps (inline, or as the replay of ``replay_of``)."""
    mode = None if timing else ("replay" if replay_of else "inline")
    want = ident.registered_identity(arm, selection=pilot, audit=AUDIT, seed=seed, timing=timing, waveform_mode=mode,
                                     replay_of_sha256=replay_of["sha256"] if replay_of else None, **modes)
    identity, root = example_identity(want, AUDIT), tmp_path / "waves"
    if mode:
        identity["waveform_dump"] = {"mode": mode, "dir": str(root / name)}
    if replay_of:
        identity["replay_of"] = {"run_manifest_sha256": replay_of["sha256"], "run_dir": "/runs/timing"}
    manifest = initialize_run(tmp_path / name, identity)
    store = wav.WaveformDump(root / name, run_stem=name, root=root) if mode else None
    for record in pilot["records"]:
        n = int(record["candidate_count"])
        candidates = np.array([[0.75 * k, 0.0, 1.0] for k in range(n)], dtype=np.float32)
        sims = np.linspace(0.1, 0.9, n, dtype=np.float32).reshape(n, 1)
        if nudge == record["index"]:
            sims[0, 0] = np.nextafter(sims[0, 0], np.float32(2.0))
        result = {"run_manifest_sha256": manifest["sha256"], "query_index": record["index"], "query_id": record["query_id"],
                  "room": "r", "candidate_count": n, "candidate_indices_sha256": record["candidate_indices_sha256"],
                  "source_global": [0.0, 0.0, 1.0], "metrics": {"1": grid.recomputed_k1_metrics(candidates, sims,
                                                                                               [0.0, 0.0, 1.0], 0.1)}}
        if store:
            waves = np.full((n, 1, 10240), 0.01 * record["index"], dtype=np.float32)
            result["waveforms"] = {"dump_dir": str(store.dir), **store.write(record["index"], waves, np.arange(n))}
        save_query_result(tmp_path / name, result=result, candidates=candidates, similarities=sims)
    if store:
        store.finish(run_manifest_sha256=manifest["sha256"], identity=identity, mode=mode,
                     query_indices=[r["index"] for r in pilot["records"]], replay_of=identity.get("replay_of"))
    return tmp_path / name, manifest


def test_an_accuracy_only_run_is_accepted_with_its_inline_dump(tmp_path):
    pilot = _pilot({1: 3, 2: 4})
    run, manifest = make_run(tmp_path, "acc_s43", "P1", pilot, timing=False, seed=43)
    entry = sources.accuracy_source(run, arm="P1", selection=pilot, audit=AUDIT, seed=43, timing=False)
    assert entry["run_manifest_sha256"] == manifest["sha256"] and entry["timing"] is False
    assert entry["waveforms"]["mode"] == "inline" and len(entry["waveforms"]["dump_manifest_sha256"]) == 64
    assert set(entry["queries"]) == {"1", "2"} and all(len(q["arrays_sha256"]) == 64 for q in entry["queries"].values())
    with pytest.raises(RuntimeError, match="sample_seed"):                       # the claimed seed is checked
        sources.accuracy_source(run, arm="P1", selection=pilot, audit=AUDIT, seed=44, timing=False)
    with pytest.raises(RuntimeError, match="waveform_dump|measure_core_forward"):   # not a timing pass
        sources.accuracy_source(run, arm="P1", selection=pilot, audit=AUDIT, seed=43, timing=True, replay_dir=run)


def test_a_tampered_or_unnamed_waveform_file_is_refused(tmp_path):
    pilot = _pilot({1: 3})
    run, _ = make_run(tmp_path, "acc", "P1", pilot, timing=False, seed=43)
    np.save(tmp_path / "waves" / "acc" / "query_00001.npy", np.zeros((3, 1, 10240), np.float32))
    with pytest.raises(RuntimeError, match="sha256"):
        sources.accuracy_source(run, arm="P1", selection=pilot, audit=AUDIT, seed=43, timing=False)
    (tmp_path / "waves" / "acc" / "dump_manifest.json").unlink()
    with pytest.raises((RuntimeError, FileNotFoundError)):
        sources.accuracy_source(run, arm="P1", selection=pilot, audit=AUDIT, seed=43, timing=False)


def test_a_timing_pass_needs_its_bitwise_replay(tmp_path):
    pilot = _pilot({1: 3, 2: 5})
    timing, manifest = make_run(tmp_path, "t42", "BF", pilot, timing=True)
    with pytest.raises(RuntimeError, match="replay"):
        sources.accuracy_source(timing, arm="BF", selection=pilot, audit=AUDIT, seed=42, timing=True)
    replay, replay_manifest = make_run(tmp_path, "t42_replay", "BF", pilot, timing=False, replay_of=manifest)
    entry = sources.accuracy_source(timing, arm="BF", selection=pilot, audit=AUDIT, seed=42, timing=True,
                                    replay_dir=replay)
    assert entry["replay"]["run_manifest_sha256"] == replay_manifest["sha256"]
    assert entry["replay"]["waveforms"]["mode"] == "replay" and "waveforms" not in entry
    drifted, _ = make_run(tmp_path, "t42_drift", "BF", pilot, timing=False, replay_of=manifest, nudge=2)
    with pytest.raises(RuntimeError, match="bitwise"):
        sources.accuracy_source(timing, arm="BF", selection=pilot, audit=AUDIT, seed=42, timing=True,
                                replay_dir=drifted)
    other, other_manifest = make_run(tmp_path, "t42b", "BF", pilot, timing=True, frame_angles=[0.0])
    foreign, _ = make_run(tmp_path, "t42b_replay", "BF", pilot, timing=False, replay_of=other_manifest, frame_angles=[0.0])
    with pytest.raises(RuntimeError, match="replay_of|frame_average_angles"):     # the replay of ANOTHER run
        sources.accuracy_source(timing, arm="BF", selection=pilot, audit=AUDIT, seed=42, timing=True,
                                replay_dir=foreign)


def test_subset_and_split_passes_are_replayed_with_their_own_identity(tmp_path):
    pilot = _pilot({1: 4})
    split, manifest = make_run(tmp_path, "split", "P1", pilot, timing=True, stage_split=True)
    replay, _ = make_run(tmp_path, "split_replay", "P1", pilot, timing=False, replay_of=manifest)   # untimed: no split
    entry = sources.accuracy_source(split, arm="P1", selection=pilot, audit=AUDIT, seed=42, timing=True,
                                    replay_dir=replay, stage_split=True)
    assert entry["replay"]["waveforms"]["mode"] == "replay"


def test_an_unknown_identity_field_is_a_deviation():
    pilot = _pilot({1: 4})
    want = ident.registered_identity("P1", selection=pilot, audit=AUDIT, seed=42, timing=True)
    identity = example_identity(want, AUDIT)
    assert ident.identity_reasons(identity, want, AUDIT) == []
    assert ident.identity_reasons({**identity, "surprise": 1}, want, AUDIT) == [
        "identity field surprise is not registered"]


def test_a_dump_must_be_named_after_its_own_run(tmp_path):
    """The dump directory is <root>/<run stem>: a run cannot claim another run's dump."""
    pilot = _pilot({1: 3})
    run, _ = make_run(tmp_path, "acc_a", "P1", pilot, timing=False, seed=43)
    other = tmp_path / "acc_b"
    shutil.copytree(run, other)
    with pytest.raises(RuntimeError, match="named after"):
        sources.accuracy_source(other, arm="P1", selection=pilot, audit=AUDIT, seed=43, timing=False)
