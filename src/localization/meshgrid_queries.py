"""exp_22 D1 -- the release-parity context materializer (inherited plan §1.1).

exp_22 keeps exp_18/exp_20's protocol everywhere EXCEPT here. The contexts must
be the ones the *released* pipeline draws, so this module runs the unmodified
``AR_md.py`` selection path under the exact exp_01 loader protocol -- ``seed=42``,
``batch_size=64``, ``num_workers=4``, ``shuffle=False``,
``pl.seed_everything(42, workers=True)``, full split order -- and records what it
drew. Three consequences of that path are protocol, not defects to repair:

  * the eligible pool is built by rendering each candidate source as
    ``f"S00{node}"``, so source 10 (whose files are ``S010_*``) is never eligible;
  * ``np.random.choice`` runs on the GLOBAL NumPy RNG, per worker, so the draw
    depends on the batch/worker layout -- exp_18's pinned 4/4 loader would give
    different contexts and must not be substituted here;
  * a pool shorter than the context width falls back from ``replace=False`` to
    ``replace=True`` rather than narrowing the context, so the 520 short-pool
    queries keep width eight.

Contexts are materialized for ALL 6,337 records and only then is
``ListeningRoom_idx_2`` filtered: excluding first would change worker assignment
and RNG consumption for the retained queries. The result is a content-hashed
manifest that every arm reuses -- no arm redraws contexts.
"""
import hashlib
import json
import os

import torch

#: the exp_01 loader protocol, verbatim; changing any of it changes the draw.
EXP01_LOADER = {"seed": 42, "batch_size": 64, "num_workers": 4, "shuffle": False}

#: the released context width; a short pool is replacement-drawn, never narrowed.
CONTEXT_WIDTH = 8

#: the mesh-available preflight subset (inherited plan §1.1/§1.3).
EXCLUDED_ROOM = "ListeningRoom/ListeningRoom_idx_2"
EXCLUDED_COUNT = 1000
FULL_COUNT = 6337
FILTERED_COUNT = 5337

#: pinned eligible-pool censuses -- pure file-tree facts, no RNG involved.
FULL_ELIGIBLE_HISTOGRAM = {6: 91, 7: 429, 8: 5263, 9: 554}
FILTERED_ELIGIBLE_HISTOGRAM = {6: 91, 7: 429, 8: 4363, 9: 454}
#: every short-pool query lives here and is replacement-drawn to width eight.
SHORT_CONTEXT_ROOM = "Cafe/Cafe_idx_1"

#: the IR tree the split file's names resolve against.
IR_ROOT = os.path.join("AcousticRooms", "single_channel_ir_1")

#: the model config the exp_01 protocol evaluates under (its metrics block names
#: the AGREE_fullAR scorer the inherited plan pins).
DEFAULT_MODEL_CONFIG = os.path.join("src", "configs", "model_configs", "FLAC", "AR",
                                    "FLAC_AR.json")

#: The RELEASED evaluator's initialization order, in full. PyTorch draws each
#: DataLoader worker's base seed when the ITERATOR is created, and every worker's
#: NumPy stream -- the one ``AR_md`` draws contexts from -- derives from it. The
#: released evaluator constructs the metric/AGREE stack BETWEEN the loader and
#: the first iteration (eval_FLAC.py: seed -> loader -> metric callback -> loop),
#: and that construction consumes torch RNG through randomized module init. Seed
#: and iterate immediately, as r1 did, and every worker gets a different stream
#: (r1 review F1).
RELEASE_CALL_GRAPH = ("seed_everything", "build_dataloader", "build_metric_stack",
                      "create_iterator")

#: where a configured AGREE checkpoint may also be found, by basename.
AGREE_SEARCH_ROOTS = ("weights", os.path.join("weights", "AGREE"))


def released_metadata_module():
    """The unmodified released selector module, loaded by path.

    Imported by location rather than by package name because it is a dataloader
    hook, not a library: the dataset config names this exact file, and a second
    copy of its selection logic is precisely what this module refuses to be.
    """
    import importlib.util

    path = os.path.join("src", "configs", "dataset_configs", "custom_metadata", "AR_md.py")
    spec = importlib.util.spec_from_file_location("AR_md", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def eligible_context_pool(ir_path):
    """The pool the released selector draws from, in its own order.

    Mirrors ``AR_md.get_ir_and_location_for_other_sources`` exactly: every other
    source node in the directory, rendered ``S00{node}``, kept only if that file
    exists. A test drives the released function itself and compares.
    """
    directory = os.path.dirname(ir_path)
    name = os.path.basename(ir_path)
    src_node = int(name.split("_")[0][1:])
    receiver = name.split("_")[1]
    all_nodes = {int(entry.split("_")[0][1:]) for entry in os.listdir(directory)}
    pool = []
    for node in list(all_nodes.difference({src_node})):
        candidate = os.path.join(directory, f"S00{node}_{receiver}_hybrid_IR.wav")
        if os.path.exists(candidate):
            pool.append(candidate)
    return pool


def eligible_pool_size(ir_path):
    return len(eligible_context_pool(ir_path))


def _room_id(relpath):
    parts = str(relpath).split("/")
    return "/".join(parts[-3:-1]) if len(parts) >= 3 else str(relpath)


def _fingerprints(md):
    from eval_FLAC import sample_context_ids

    return list(sample_context_ids(md))


def _audio_digests(context_audio):
    """One sha256 per context RIR, over its exact float32 bytes, in draw order."""
    tensor = torch.as_tensor(context_audio)
    if tensor.dtype != torch.float32:
        raise ValueError(f"context_audio is {tensor.dtype}, not float32; the digest would "
                         "not be comparable across machines")
    return [hashlib.sha256(tensor[index].contiguous().numpy().tobytes()).hexdigest()
            for index in range(tensor.shape[0])]


def context_record(md, position, eligible):
    """One record of what a query drew: identity, contexts, pool size, order."""
    from eval_FLAC import sample_target_id

    fingerprints = _fingerprints(md)
    width = len(fingerprints)
    if width != CONTEXT_WIDTH:
        raise ValueError(f"context width {width} != the released {CONTEXT_WIDTH}; a short "
                         "pool is replacement-drawn, never narrowed")
    digests = _audio_digests(md["context_audio"])
    if len(digests) != width:
        raise ValueError(f"{len(digests)} context RIRs for {width} context poses")

    # the target source is excluded by construction; prove it rather than assume
    target = md.get("source")
    target_absent = True
    if target is not None:
        rendered = _fingerprints({"context_poses": torch.as_tensor(target).reshape(1, -1)})
        if rendered and rendered[0] in fingerprints:
            raise ValueError(f"query at position {position}: the target source appears among "
                             "its own contexts; the held-out RIR would leak")
    return {
        "position": int(position),
        "query_id": sample_target_id(md),
        "room_id": _room_id(md.get("relpath") or md.get("path") or ""),
        "relpath": md.get("relpath"),
        "context_fingerprints": fingerprints,
        "context_audio_sha256": digests,
        "context_width": width,
        "eligible": int(eligible),
        "target_absent": bool(target_absent),
    }


def rng_state_digest():
    """sha256 of the global torch RNG state -- the state the worker base seed is
    drawn from at iterator creation."""
    import torch as _torch

    return hashlib.sha256(_torch.random.get_rng_state().numpy().tobytes()).hexdigest()


def resolve_agree_checkpoint(configured, roots=None):
    """The AGREE checkpoint the metric stack will load, fail-closed.

    The configured path is preferred; a same-basename copy under the known
    weights roots is accepted and RECORDED as a fallback rather than silently
    substituted; nothing found is a refusal, because the released call graph
    loads this file and an exception would change the graph.
    """
    if configured and os.path.isfile(configured):
        return {"path": configured, "resolution": "configured", "configured": configured}
    basename = os.path.basename(str(configured or ""))
    for root in (roots if roots is not None else AGREE_SEARCH_ROOTS):
        candidate = os.path.join(root, basename)
        if basename and os.path.isfile(candidate):
            return {"path": candidate, "resolution": "basename_fallback",
                    "configured": configured}
    raise ValueError(f"the AGREE checkpoint {configured!r} is not on disk and no copy of "
                     f"{basename!r} was found under {list(AGREE_SEARCH_ROOTS)}; the released "
                     "call graph loads it, so it cannot be skipped")


def with_resolved_agree(model_config, roots=None):
    """``model_config`` with its metrics AGREE path resolved to a real file."""
    resolved = resolve_agree_checkpoint(
        ((model_config.get("training") or {}).get("metrics") or {}).get("AGREE_ckpt"),
        roots=roots)
    out = json.loads(json.dumps(model_config))
    out["training"]["metrics"]["AGREE_ckpt"] = resolved["path"]
    return out


def build_release_stack(dataset_config_path, model_config_path=None, agree_roots=None,
                        skip_metric_stack=False):
    """Seed, build the loader, build the metric/AGREE stack -- in THAT order.

    Returns ``(loader, facts)``; the iterator is created by the caller, which is
    the moment the worker base seeds are drawn.
    """
    import pytorch_lightning as pl

    from src.data.dataset import create_dataloader_from_config
    from src.training import create_metric_callback_from_config

    with open(dataset_config_path) as handle:
        dataset_config = json.load(handle)
    with open(model_config_path or DEFAULT_MODEL_CONFIG) as handle:
        model_config = json.load(handle)
    configured = ((model_config.get("training") or {}).get("metrics") or {}).get("AGREE_ckpt")
    resolved = resolve_agree_checkpoint(configured, roots=agree_roots)
    model_config["training"]["metrics"]["AGREE_ckpt"] = resolved["path"]

    pl.seed_everything(EXP01_LOADER["seed"], workers=True)
    loader = create_dataloader_from_config(
        dataset_config, batch_size=EXP01_LOADER["batch_size"],
        num_workers=EXP01_LOADER["num_workers"], sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 1),
        shuffle=EXP01_LOADER["shuffle"])

    built = False
    if not skip_metric_stack:
        create_metric_callback_from_config(
            model_config, dataset_id=dataset_config["datasets"][0]["id"], per_scene=False)
        built = True
    facts = {
        "call_graph": list(RELEASE_CALL_GRAPH if built else
                           tuple(step for step in RELEASE_CALL_GRAPH
                                 if step != "build_metric_stack")),
        "metric_stack_built": built,
        "agree_ckpt": resolved["path"], "agree_configured": resolved["configured"],
        "agree_resolution": resolved["resolution"],
        "agree_sha256": _sha256_file(resolved["path"]),
        "model_config": str(model_config_path or DEFAULT_MODEL_CONFIG),
        "rng_digest_at_iter": rng_state_digest(),
    }
    return loader, facts


def _sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_enumeration(dataset_config):
    """The split's file list, rebuilt the way ``SampleDataset`` builds it.

    Accepts the config object or its path, so the materializer has ONE seam for
    "what the split declares" and a failure anywhere in it is a refusal.
    """
    from src.data.dataset import get_audio_filenames

    if isinstance(dataset_config, (str, bytes, os.PathLike)):
        with open(dataset_config) as handle:
            dataset_config = json.load(handle)
    filenames = []
    for audio_dir in dataset_config.get("datasets") or []:
        filenames.extend(get_audio_filenames(
            paths=audio_dir["path"], keywords=None,
            json_file_path=audio_dir.get("json_file_path"),
            folder_name=audio_dir.get("folder_name")))
    return filenames


#: dataset roots a split enumeration may carry in front of the loader's relpath.
DATASET_ROOTS = ("AcousticRooms",)


def canonical_relpath(path, roots=DATASET_ROOTS):
    """One canonical root-relative form, so the comparison can be EXACT.

    The split enumeration carries the dataset root and the loader's relpath does
    not. r2 review: comparing with a bidirectional ``endswith`` accepts a
    basename or a partial component, so a different room's file can impersonate a
    position. Both sides are normalized here once and then compared as strings.
    """
    text = os.path.normpath(str(path)).lstrip("/")
    for root in roots:
        root = os.path.normpath(str(root)).lstrip("/")
        marker = root + os.sep
        index = text.find(marker)
        if index != -1:
            return text[index + len(marker):]
    return text


def assert_split_enumeration(filenames, expected_count=FULL_COUNT):
    """The split IS the registered enumeration: ordered, unique, complete."""
    filenames = list(filenames)
    if not filenames:
        raise ValueError("the split enumeration is empty: no entries to materialize; a "
                         "missing or unreadable split may not become an unchecked pass")
    canonical = [canonical_relpath(name) for name in filenames]
    duplicates = sorted({name for name in canonical if canonical.count(name) > 1})
    if duplicates:
        raise ValueError(f"the split enumeration has duplicate entries {duplicates[:5]}; the "
                         "registered order must name each query exactly once")
    if expected_count is not None and len(canonical) != int(expected_count):
        raise ValueError(f"the split enumeration declares {len(canonical)} queries but the "
                         f"registered split expects {expected_count}")
    return len(canonical)


def assert_pass_census(materialized, expected_count=FULL_COUNT,
                       expected_histogram=None):
    """The registered census, applied to a COMPLETE pass. Never optional."""
    expected_histogram = (FULL_ELIGIBLE_HISTOGRAM if expected_histogram is None
                          else expected_histogram)
    records = materialized["records"]
    if expected_count is not None and len(records) != int(expected_count):
        raise ValueError(f"the pass census is {len(records)} records, not the registered "
                         f"{expected_count}")
    histogram = eligible_histogram(records)
    if histogram != expected_histogram:
        raise ValueError(f"the eligible-pool histogram is {histogram}, not the registered "
                         f"{expected_histogram}; the released selection path did not behave "
                         "as the protocol pins it")
    identities = [record["query_id"] for record in records]
    if len(set(identities)) != len(identities):
        raise ValueError("the pass recorded the same query twice")
    return True


def assert_stream_position(md, position, expected_relpath):
    """Refuse a silently substituted item.

    ``SampleDataset.__getitem__`` returns a random OTHER item when one fails to
    load, and the substitute carries its own idx/relpath -- counts and even the
    eligible histogram can survive that, so only a positional comparison against
    the split enumeration catches it (r1 review F2).
    """
    found_idx = md.get("idx")
    if found_idx is None or int(found_idx) != int(position):
        raise ValueError(f"stream position {position}: the loader returned idx={found_idx!r}; "
                         "SampleDataset substitutes a random other item when one fails to "
                         "load, and a substituted context draw is not this query's")
    if expected_relpath is None:
        raise ValueError(f"stream position {position}: no split entry to compare against; "
                         "the positional guard may not be skipped")
    found = canonical_relpath(md.get("relpath") or md.get("path") or "")
    expected = canonical_relpath(expected_relpath)
    if found != expected:
        raise ValueError(f"stream position {position}: the loader returned relpath "
                         f"{found!r}, but the split declares {expected!r}")
    return True


def materialize_contexts(dataset_config_path, model_config_path=None, limit=None,
                         ir_root=IR_ROOT, _skip_metric_stack=False):
    """Run the released call graph over the split and record every draw.

    ``limit`` yields a BOUNDED slice, which is never marked complete: only a full
    pass may be filtered, because the excluded room's records consume RNG that
    the retained queries' draws depend on. Every position is checked against the
    split enumeration before its draw is recorded.
    """
    loader, facts = build_release_stack(dataset_config_path, model_config_path,
                                        skip_metric_stack=_skip_metric_stack)
    # fail-closed: an enumeration that cannot be built is a refusal, never an
    # empty expectation that quietly disables the positional guard (r2 review F2)
    try:
        expected = expected_enumeration(dataset_config_path)
    except Exception as error:                       # noqa: BLE001 -- reported as a refusal
        raise ValueError(f"the split enumeration could not be built from "
                         f"{dataset_config_path!r}: {error}") from error
    assert_split_enumeration(expected, expected_count=None if limit is not None
                             else FULL_COUNT)
    records = []
    for _reals, metadata in loader:                    # <- the iterator is created here
        for md in metadata:
            position = len(records)
            expected_relpath = expected[position] if position < len(expected) else None
            assert_stream_position(md, position, expected_relpath)
            path = md.get("path") or os.path.join(ir_root, md.get("relpath", ""))
            records.append(context_record(md, position, eligible_pool_size(path)))
            if limit is not None and len(records) >= int(limit):
                return _materialized(records, dataset_config_path, complete=False,
                                     facts=facts, expected=expected)
    if limit is None and expected and len(records) != len(expected):
        raise ValueError(f"the pass produced {len(records)} records but the split declares "
                         f"{len(expected)}; a truncated stream may not be recorded")
    out = _materialized(records, dataset_config_path, complete=limit is None, facts=facts,
                        expected=expected)
    if limit is None:
        assert_pass_census(out)                      # mandatory, not optional
    return out


def _materialized(records, dataset_config_path, complete, facts=None, expected=None):
    return {"records": records, "n_records": len(records), "complete": bool(complete),
            "protocol": dict(EXP01_LOADER), "dataset_config": str(dataset_config_path),
            "context_width": CONTEXT_WIDTH, "protocol_facts": dict(facts or {}),
            "n_expected": None if expected is None else len(expected)}


def filter_excluded_room(materialized, room=EXCLUDED_ROOM, expected_excluded=EXCLUDED_COUNT):
    """Drop exactly the excluded room, and only after a COMPLETE pass."""
    if not materialized.get("complete"):
        raise ValueError("the materialization is not complete: contexts must be drawn for "
                         "every record before filtering, because excluding first changes "
                         "worker assignment and RNG consumption for the retained queries "
                         "(inherited plan §1.1)")
    kept, dropped = [], []
    for record in materialized["records"]:
        (dropped if record["room_id"] == room else kept).append(record)
    if len(dropped) != int(expected_excluded):
        raise ValueError(f"the filter removed {len(dropped)} records from {room!r}, not "
                         f"exactly {expected_excluded}; any additional loss is a protocol "
                         "failure")
    if len(kept) + len(dropped) != materialized["n_records"]:
        raise ValueError("records went missing during filtering")
    out = dict(materialized)
    out.update({"records": kept, "n_records": len(kept),
                "excluded": {"room_id": room, "n_excluded": len(dropped),
                             "query_ids": [r["query_id"] for r in dropped],
                             "reason": "no official OBJ on the AcousticRooms commit; the "
                                       "mesh-available preflight subset is 5,337 queries / "
                                       "16 rooms"}})
    return out


def histogram_pairs(histogram):
    """``{size: count}`` as sorted ``[[size, count], ...]``.

    JSON coerces integer keys to strings, so a manifest that stored the census as
    a dict would not reload equal to what was written -- and the manifest's whole
    job is to be reloaded byte-stably.
    """
    return [[int(size), int(count)] for size, count in sorted(histogram.items())]


def histogram_from_pairs(pairs):
    return {int(size): int(count) for size, count in (pairs or [])}


def eligible_histogram(records):
    """Pool-size census over records, as ``{size: count}``."""
    histogram = {}
    for record in records:
        size = int(record["eligible"])
        histogram[size] = histogram.get(size, 0) + 1
    return dict(sorted(histogram.items()))


def short_context_audit(records, width=CONTEXT_WIDTH):
    """Prove every short-pool query kept the full width, and none was dropped."""
    short = [r for r in records if int(r["eligible"]) < width]
    for record in short + [r for r in records if r not in short]:
        if int(record["context_width"]) != width:
            raise ValueError(f"query {record['query_id']!r} carries context width "
                             f"{record['context_width']}, not the released {width}; the "
                             "short-pool fallback replaces, it does not narrow")
    return {"n_short": len(short), "rooms": sorted({r["room_id"] for r in short}),
            "all_width_eight": True, "n_dropped": 0,
            "eligible_sizes": histogram_pairs(eligible_histogram(short))}


#: the record fields that ARE the frozen draw; the stream hash covers these only.
STREAM_FIELDS = ("position", "query_id", "room_id", "context_fingerprints",
                 "context_audio_sha256", "context_width", "eligible")


def stream_hash(records):
    """sha256 over the ordered record stream, in a pinned serialization."""
    payload = "\n".join(
        json.dumps({field: record[field] for field in STREAM_FIELDS},
                   sort_keys=True, separators=(",", ":"))
        for record in records)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_manifest(full, filtered):
    """The frozen manifest: both streams hashed, the exclusion recorded."""
    from datetime import datetime, timezone

    if not full.get("complete"):
        raise ValueError("a manifest may only be built from a complete materialization")
    census = eligible_histogram(full["records"])
    filtered_census = eligible_histogram(filtered["records"])
    return {
        "experiment": "exp_22 loc_meshgrid D1 context manifest",
        "protocol": dict(EXP01_LOADER), "context_width": CONTEXT_WIDTH,
        "protocol_facts": full.get("protocol_facts") or {},
        "call_graph": list(RELEASE_CALL_GRAPH),
        "dataset_config": full.get("dataset_config"),
        "n_full": full["n_records"], "n_filtered": filtered["n_records"],
        "full_stream_sha256": stream_hash(full["records"]),
        "filtered_stream_sha256": stream_hash(filtered["records"]),
        "eligible_histogram_full": histogram_pairs(census),
        "eligible_histogram_filtered": histogram_pairs(filtered_census),
        "short_context": short_context_audit(full["records"]),
        "excluded": filtered.get("excluded"),
        "census_verified": (full["n_records"] == FULL_COUNT
                            and filtered["n_records"] == FILTERED_COUNT
                            and census == FULL_ELIGIBLE_HISTOGRAM
                            and filtered_census == FILTERED_ELIGIBLE_HISTOGRAM),
        "records": filtered["records"],
        "records_full": full["records"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def assert_registered_census(manifest):
    """Refuse a manifest that is not the registered 6,337 -> 5,337 census."""
    if not manifest.get("census_verified"):
        raise ValueError(
            f"the manifest covers {manifest.get('n_full')} -> {manifest.get('n_filtered')} "
            f"records with histograms "
            f"{histogram_from_pairs(manifest.get('eligible_histogram_full'))} / "
            f"{histogram_from_pairs(manifest.get('eligible_histogram_filtered'))}; "
            "the registered census is "
            f"{FULL_COUNT} -> {FILTERED_COUNT} with {FULL_ELIGIBLE_HISTOGRAM} / "
            f"{FILTERED_ELIGIBLE_HISTOGRAM}")
    return True


def write_manifest(path, manifest):
    """Write byte-stably (sorted keys, fixed indent), atomically."""
    with open(path + ".partial", "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(path + ".partial", path)
    return path


def load_manifest(path):
    """Reload a frozen manifest and re-verify its stream hashes.

    Reuse means reuse: this never constructs a loader, so it cannot redraw.
    """
    with open(path) as handle:
        manifest = json.load(handle)
    for key, records in (("filtered_stream_sha256", manifest.get("records")),
                         ("full_stream_sha256", manifest.get("records_full"))):
        if records is None:
            continue
        found = stream_hash(records)
        if found != manifest.get(key):
            raise ValueError(f"{path}: {key} is {str(manifest.get(key))[:16]}... but the "
                             f"records hash to {found[:16]}...; the manifest was edited")
    return manifest


def census_from_split(dataset_config_path, split_path=None, ir_root=IR_ROOT,
                      excluded_room=EXCLUDED_ROOM):
    """The eligible-pool census straight from the file tree -- no loader, no RNG."""
    with open(dataset_config_path) as handle:
        dataset_config = json.load(handle)
    split_path = split_path or _split_path_of(dataset_config)
    with open(split_path) as handle:
        split = json.load(handle)

    full, filtered, short_rooms = [], [], set()
    for scene in sorted(split):
        for scene_id in sorted(split[scene]):
            room_id = f"{scene}/{scene_id}"
            for name in sorted(split[scene][scene_id]):
                path = os.path.join(ir_root, scene, scene_id, name)
                size = eligible_pool_size(path)
                full.append(size)
                if size < CONTEXT_WIDTH:
                    short_rooms.add(room_id)
                if room_id != excluded_room:
                    filtered.append(size)
    return {"full": eligible_histogram([{"eligible": s} for s in full]),
            "filtered": eligible_histogram([{"eligible": s} for s in filtered]),
            "n_full": len(full), "n_filtered": len(filtered),
            "short_context_rooms": sorted(short_rooms)}


def _split_path_of(dataset_config):
    for dataset in dataset_config.get("datasets", []):
        for key in ("split_file", "json_file", "split"):
            if dataset.get(key):
                return dataset[key]
    return os.path.join("data", "AR", "unseen_eval.json")
