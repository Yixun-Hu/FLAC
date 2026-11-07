"""exp_37 accuracy sources and their waveform artifacts (announcement 08; Codex code review 1, BLOCKING 1 / HIGH 3).

An accuracy number is publishable only from a run whose scored waveforms exist:

- an ACCURACY-ONLY run: its registered identity (untimed, ``waveform_dump`` mode ``inline``) and its own dump -- the
  dump manifest bound to the run manifest (checkpoint, seed, backend), covering exactly its queries, each hashed
  query record naming its waveform file (path, sha256, full-grid index mapping);
- a TIMING pass: its registered timing identity plus its ARTIFACT REPLAY -- the replay's registered identity
  (untimed, mode ``replay``, ``replay_of`` = the timing manifest's sha256; a split pass's replay is the plain pass),
  the replay's own dump, and per query the stored candidates and similarities bitwise equal to the timing run's.

``accuracy_source`` returns the inventory entry the summaries publish (run / replay / dump manifest sha256 and every
query record's and array file's sha256). Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 1.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.localization import exp37_identity as ident
from src.localization import exp37_queries as queries
from src.localization import exp37_waveforms as wav
from src.localization.pilot import canonical_sha256


def _hashed(path) -> dict:
    payload = json.loads(Path(path).read_text())
    if payload.get("sha256") != canonical_sha256({k: v for k, v in payload.items() if k != "sha256"}):
        raise RuntimeError(f"stale or corrupt sha256 in {path}")
    return payload


def query_record(run_dir, manifest: dict, index: int):
    """One hash-verified query record of this run and its (hash-verified) array file."""
    query = _hashed(Path(run_dir) / "queries" / f"query_{int(index):05d}.json")
    arrays = Path(run_dir) / query["arrays_file"]
    if query.get("run_manifest_sha256") != manifest["sha256"] or int(query.get("query_index", -1)) != int(index) or \
            hashlib.sha256(arrays.read_bytes()).hexdigest() != query.get("arrays_sha256"):
        raise RuntimeError(f"{run_dir}: query {index} is not this run's, or its arrays changed")
    return query, arrays


def check_against_selection(run_dir, manifest: dict, selection: dict, *, subset: bool) -> dict:
    """Review 2 HIGH 4: every query record of a source against its selection record (full: the selection's count and
    candidate hash; subset: the selection as parent; the active hash is bound to the dumped index vector)."""
    records = {}
    for record in selection["records"]:
        query = query_record(run_dir, manifest, record["index"])[0]
        queries.check_query(query, record, run_sha256=manifest["sha256"], label=f"{run_dir}: ",
                            **({"active_sha256": query.get("candidate_indices_sha256"),
                                "count": query.get("candidate_count")} if subset else {}))
        records[int(record["index"])] = query
    return records


def query_inventory(run_dir, manifest: dict, indices) -> dict:
    return {str(int(i)): {"record_sha256": q["sha256"], "arrays_sha256": q["arrays_sha256"]}
            for i in indices for q in (query_record(run_dir, manifest, i)[0],)}


def dump_coverage(run_dir, manifest: dict, counts: dict) -> dict:
    """The run's own dump: named in its identity, bound to its manifest, covering exactly ``counts`` (query -> rows)
    and named by every query record."""
    identity = manifest["identity"]
    dump = identity.get("waveform_dump")
    if not isinstance(dump, dict) or not dump.get("dir"):
        raise RuntimeError(f"{run_dir}: no waveform dump (announcement 08)")
    if Path(dump["dir"]).name != Path(run_dir).name:
        raise RuntimeError(f"{run_dir}: its dump {dump['dir']} is not named after the run (<root>/<run stem>)")
    payload = wav.load_dump_manifest(dump["dir"], run_manifest_sha256=manifest["sha256"], query_counts=counts)
    bound = all(payload.get(k) == identity.get(k) for k in ("checkpoint_sha256", "sample_seed", "attention_backend"))
    if payload["mode"] != dump["mode"] or payload["replay_of"] != identity.get("replay_of") or not bound:
        raise RuntimeError(f"{run_dir}: the dump manifest's mode / replay / checkpoint / seed / backend is not the run's")
    subset = identity.get("candidate_subset") is not None
    for index in counts:
        query = query_record(run_dir, manifest, index)[0]
        named = query.get("waveforms") or {}
        entry = payload["coverage"]["files"][str(int(index))]
        if any(named.get(k) != entry.get(k) for k in ("path", "sha256", "indices_path", "indices_sha256",
                                                      "full_indices_sha256")):
            raise RuntimeError(f"{run_dir}: query {index} does not name its dumped waveform / index files")
        rows = wav.dump_indices(dump["dir"], entry).astype(np.int64)    # review 2 BLOCKING 2: the actual mapping
        if (subset and entry["full_indices_sha256"] != query.get("candidate_indices_sha256")) or (
                not subset and not np.array_equal(rows, np.arange(int(counts[index])))):
            raise RuntimeError(f"{run_dir}: query {index}'s dumped index vector is not its candidate mapping")
    return {"dump_dir": dump["dir"], "dump_manifest_sha256": payload["sha256"], "mode": payload["mode"]}


def replay_coverage(replay_dir, timing_dir, timing_manifest: dict, *, expected: dict, audit: dict) -> dict:
    replay_dir = Path(replay_dir)
    replay = _hashed(replay_dir / "run_manifest.json")
    bad = ident.identity_reasons(replay["identity"], expected, audit)
    if bad:
        raise RuntimeError(f"{replay_dir}: {bad} (the registered replay of {timing_dir})")
    counts = {}
    for index in replay["identity"]["query_indices"]:
        timed, timed_arrays = query_record(timing_dir, timing_manifest, index)
        again, again_arrays = query_record(replay_dir, replay, index)
        with np.load(timed_arrays) as a, np.load(again_arrays) as b:
            same = all(a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]) for k in ("candidates", "similarities"))
        if not same or any(timed.get(k) != again.get(k) for k in ("candidate_count", "candidate_indices_sha256")):
            raise RuntimeError(f"{replay_dir}: query {index} does not reproduce the timing run bitwise")
        counts[int(index)] = int(timed["candidate_count"])
    return {"run_dir": str(replay_dir), "run_manifest_sha256": replay["sha256"],
            "waveforms": dump_coverage(replay_dir, replay, counts), "queries": query_inventory(replay_dir, replay, counts)}


def accuracy_source(run_dir, *, arm: str, selection: dict, audit: dict, seed: int, timing: bool, replay_dir=None,
                    **modes) -> dict:
    """The validated inventory entry of one accuracy source (see the module docstring); RuntimeError otherwise."""
    run_dir = Path(run_dir)
    manifest = _hashed(run_dir / "run_manifest.json")
    want = ident.registered_identity(arm, selection=selection, audit=audit, seed=seed, timing=timing,
                                     waveform_mode=None if timing else "inline", **modes)
    bad = ident.identity_reasons(manifest["identity"], want, audit)
    if bad:
        raise RuntimeError(f"{run_dir}: {bad} (the {arm} seed-{seed} {'timing' if timing else 'accuracy-only'} source)")
    indices = [int(r["index"]) for r in selection["records"]]
    check_against_selection(run_dir, manifest, selection, subset=bool(modes.get("candidate_subset")))
    entry = {"run_dir": str(run_dir), "run_manifest_sha256": manifest["sha256"], "timing": bool(timing),
             "queries": query_inventory(run_dir, manifest, indices)}
    if not timing:
        if replay_dir is not None:
            raise ValueError(f"{run_dir}: an accuracy-only run dumped inline; it has no replay")
        counts = {i: int(query_record(run_dir, manifest, i)[0]["candidate_count"]) for i in indices}
        return {**entry, "waveforms": dump_coverage(run_dir, manifest, counts)}
    if replay_dir is None:
        raise RuntimeError(f"{run_dir}: a timing pass whose accuracy is reported needs its artifact replay (--replay-of)")
    replay_modes = {k: v for k, v in modes.items() if k != "stage_split"}
    expected = ident.registered_identity(arm, selection=selection, audit=audit, seed=seed, timing=False,
                                         waveform_mode="replay", replay_of_sha256=manifest["sha256"], **replay_modes)
    replay = replay_coverage(replay_dir, run_dir, manifest, expected=expected, audit=audit)
    check_against_selection(Path(replay_dir), _hashed(Path(replay_dir) / "run_manifest.json"), selection,
                            subset=bool(modes.get("candidate_subset")))
    return {**entry, "replay": replay}
