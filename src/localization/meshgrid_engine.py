"""exp_22 I1 -- the frozen mesh-grid localization engine (inherited plan §1.4/§1.5).

One query is an observed RIR ``h_obs``; the engine regenerates one RIR per
mesh-valid grid candidate under the frozen Vanilla FLAC checkpoint, scores each
against ``h_obs`` in AGREE's audio embedding space, and predicts the argmax
candidate. Everything that decides a number -- the noise a candidate is drawn
from, the context tensors it is conditioned on, the candidate set it competes in
-- is bound to a registered artifact (the D1 context manifest, the G1 candidate
manifests, the checkpoint, the scorer) and refused when it does not match.

Two protocol points are RECORDED DEVIATIONS from the inherited plan text and are
stamped into every run's provenance rather than being silently taken:

* ``SCORER_READOUT`` -- §1.4 pins ``encode_audio(..., normalize=True)``. That
  path samples AGREE's VAE bottleneck, which exp_18 measured as ~7e-5 of cosine
  noise per call and which consumes the global RNG stream. The registered
  scorer here is exp_18/exp_20's deterministic mean readout, which is the same
  arithmetic with the bottleneck's mean substituted for its sample.
* ``NOISE_KEY_POLICY`` -- §1.1 says candidates of a query share their seeds
  (common random numbers); the dispatched contract keys the draw by
  ``(seed, query_id, candidate_index, k)``. The dispatched key is the default,
  and the shared-across-candidates alternative is implemented and selectable so
  a ruling either way costs no code round.
"""
import hashlib
import json
import math
import os

import numpy as np
import torch

from src.localization.reaggregate import encode_sims
from src.localization.scoring import aggregate
from src.localization import scoring as _scoring

#: the three reported nested prefixes; K=1 is sample 0, K=4 samples 0-3 (§1.4).
K_PREFIXES = (1, 4, 8)
#: length of the one generated sequence all three prefixes are read from.
NUM_SAMPLES = 8
#: the fixed score temperature (§1.4; Yixun 2026-08-21).
TAU = 0.1
#: the registered noise seed.
SEED = 42
#: sampler settings mirroring the release evaluation path (§1.4).
STEPS = 1
CFG_SCALE = 1.0

#: conditioning branches. The context branch is computed once per QUERY and the
#: source branch once per (receiver, candidate) -- the split of §1.5.
CONTEXT_COND_IDS = ("context_poses_vit", "context_poses", "context_audio")
SOURCE_COND_IDS = ("source", "source_vit")

#: the registered readout and the measurement that justifies the deviation.
SCORER_READOUT = "mean"
SCORER_READOUT_DEVIATION = (
    "inherited plan §1.4 names encode_audio(..., normalize=True); this run uses the "
    "deterministic VAE-mean readout of exp_18/exp_20 (src/localization/agree_embed.py), "
    "because the sampled path draws from AGREE's VAE bottleneck -- measured by exp_18 at "
    "~7e-5 cosine noise per call -- and consumes the global RNG stream")

#: the scorer's leakage caveat (Yixun 2026-08-24 decision 2c), stamped everywhere.
AGREE_LEAKAGE_CAVEAT = (
    "AGREE_fullAR saw the full dataset including the unseen rooms; acceptable here because "
    "the scorer is frozen, identical across arms and candidates, and pinned by the approved "
    "exp_09 protocol -- but absolute levels are NOT leak-free and must never be compared "
    "against AGREE_AR-scored exp_18/exp_20 rows without this label")

NOISE_KEY_POLICIES = ("per_candidate", "shared_across_candidates")
#: the dispatched default (see the module docstring's deviation note).
NOISE_KEY_POLICY = "per_candidate"


# --------------------------------------------------------------------------- #
# noise
# --------------------------------------------------------------------------- #
def noise_key(seed, query_id, candidate_index, k):
    """Deterministic generator seed for ``(query, candidate, draw)``.

    sha256 over a canonical JSON payload -- never Python's salted ``hash()`` --
    so a resumed pass, a second machine and an offline replay all draw the same
    latent for the same candidate.
    """
    payload = json.dumps(["loc_meshgrid_noise_key", int(seed), str(query_id),
                          int(candidate_index), int(k)], separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def noise_key_for(policy, seed, query_id, candidate_index, k):
    """The draw's generator seed under the selected keying policy."""
    if policy == "per_candidate":
        return noise_key(seed, query_id, candidate_index, k)
    if policy == "shared_across_candidates":
        return _scoring.noise_key(seed, query_id, k)
    raise ValueError(f"unknown noise policy {policy!r} (expected one of {list(NOISE_KEY_POLICIES)})")


def noise_block(seed, query_id, candidate_indices, num_samples, latent_shape,
                policy=NOISE_KEY_POLICY, device="cpu"):
    """The candidate-major latent noise for one query slice: ``[M * K, C, T]``.

    Row ``m * K + k`` is candidate ``candidate_indices[m]`` with draw ``k``, each
    from its own CPU ``torch.Generator`` -- never the global stream, which the
    loader and conditioners also advance. Because every row is keyed
    independently, generating a slice of candidates yields exactly the rows the
    whole block would have carried, which is what makes the pass batch-size and
    chunking invariant.
    """
    num_samples = int(num_samples)
    if num_samples < 1:
        raise ValueError(f"num_samples (K) must be >= 1, got {num_samples}")
    shape = tuple(int(s) for s in latent_shape)
    if len(shape) != 2 or any(s < 1 for s in shape):
        raise ValueError(f"latent_shape must be [channels, samples] with positive dims, got {shape}")
    indices = [int(i) for i in candidate_indices]
    if not indices:
        raise ValueError("candidate_indices must be a non-empty sequence")
    if len(set(indices)) != len(indices):
        raise ValueError("candidate_indices must be unique: a repeated candidate would be "
                         "generated twice under the same key and scored twice")

    draws = []
    for index in indices:
        for k in range(num_samples):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(noise_key_for(policy, seed, query_id, index, k))
            draws.append(torch.randn(shape, generator=generator))
    return torch.stack(draws).to(device)


# --------------------------------------------------------------------------- #
# scoring: the nested prefixes and the registered prediction rule
# --------------------------------------------------------------------------- #
def nested_scores(sims, tau=TAU, prefixes=K_PREFIXES):
    """``{K: {scores, mean_scores}}`` over nested prefixes of one ``[M, K]`` block.

    All three settings read the SAME generated sequence (§1.4), so they cost one
    K=8 execution and cannot disagree about a sample.
    """
    if not isinstance(sims, torch.Tensor) or sims.ndim != 2:
        raise ValueError(f"sims must be an [M, K] tensor, got {type(sims).__name__}")
    if not bool(torch.isfinite(sims).all()):
        raise ValueError("sims must be finite (no NaN or Inf)")
    prefixes = tuple(int(k) for k in prefixes)
    biggest = max(prefixes)
    if sims.shape[-1] < biggest:
        raise ValueError(f"sims carries {sims.shape[-1]} samples but the registered prefixes "
                         f"{list(prefixes)} need {biggest}")
    out = {}
    for k in prefixes:
        head = sims[:, :k].contiguous()
        out[k] = {"scores": aggregate(head, method="lme", tau=tau),
                  "mean_scores": head.mean(dim=-1)}
    return out


def argmax_by_global_index(scores, candidate_indices):
    """Highest score, ties broken by the SMALLEST global candidate index (§1.4).

    Returns the ROW of the winner. The manifest's indices are ascending, so this
    is normally the first maximal row -- but the rule is written on the global
    index so that a re-ordered candidate slice cannot change the prediction.
    """
    if not isinstance(scores, torch.Tensor):
        scores = torch.as_tensor(scores)
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError(f"scores must be a non-empty [M] tensor, got shape {tuple(scores.shape)}")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must be finite (no NaN or Inf)")
    indices = [int(i) for i in candidate_indices]
    if len(indices) != scores.numel():
        raise ValueError(f"candidate_indices has {len(indices)} entries for {scores.numel()} "
                         "scores")
    best = float(scores.max())
    winners = [row for row in range(len(indices)) if float(scores[row]) == best]
    return min(winners, key=lambda row: indices[row])


def score_query(sims, candidate_indices, coordinates, tau=TAU, prefixes=K_PREFIXES):
    """One query's full score block: every prefix's scores, prediction and mean.

    ``scores_hex`` is the exact float32 hex codec exp_18 published its
    similarities in, so an offline re-aggregation reproduces the online numbers
    bit for bit.
    """
    coordinates = np.asarray(coordinates, dtype=np.float64)
    indices = [int(i) for i in candidate_indices]
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(f"coordinates must be [M, 3], got shape {coordinates.shape}")
    if coordinates.shape[0] != len(indices):
        raise ValueError(f"{coordinates.shape[0]} coordinates for {len(indices)} candidates")

    by_k = {}
    for k, block in nested_scores(sims, tau=tau, prefixes=prefixes).items():
        row = argmax_by_global_index(block["scores"], indices)
        mean_row = argmax_by_global_index(block["mean_scores"], indices)
        by_k[k] = {
            "prediction_row": row,
            "prediction_index": indices[row],
            "prediction_xyz": coordinates[row].tolist(),
            "scores_hex": encode_sims(block["scores"].reshape(1, -1))[0],
            "mean_prediction_row": mean_row,
            "mean_prediction_index": indices[mean_row],
            "mean_prediction_xyz": coordinates[mean_row].tolist(),
            "mean_scores_hex": encode_sims(block["mean_scores"].reshape(1, -1))[0],
        }
    return {"by_k": by_k, "n_candidates": len(indices),
            "num_samples": int(sims.shape[-1]), "tau": float(tau)}


# --------------------------------------------------------------------------- #
# the conditioning split and its caches (§1.5)
# --------------------------------------------------------------------------- #
def _branch(conditioner, metadata, device, ids):
    """One conditioning branch, refusing anything the conditioner did not answer.

    ``only_ids`` is the released ``MultiConditioner`` seam; a branch that comes
    back short would silently drop a conditioning input, which the DiT would
    happily run without.
    """
    out = conditioner(metadata, device, only_ids=list(ids))
    missing = [key for key in ids if key not in out]
    if missing:
        raise ValueError(f"the conditioner returned no {missing} for the requested branch "
                         f"{list(ids)}; a missing conditioning input is not a variant")
    extra = [key for key in out if key not in ids]
    if extra:
        raise ValueError(f"the conditioner returned unrequested ids {extra}; the branch split "
                         "must be exact so the two caches cannot overlap")
    return {key: list(out[key]) for key in ids}


def context_conditioning(conditioner, md, device, ids=CONTEXT_COND_IDS):
    """The query's context branch, computed ONCE over a single row (§1.5)."""
    return _branch(conditioner, [md], device, ids)


def source_conditioning(conditioner, md, positions_cam, device, chunk=256,
                        ids=SOURCE_COND_IDS):
    """The source branch over ``[U, 3]`` camera-frame candidate positions.

    Chunked so a receiver's whole union never has to fit in one forward, and
    concatenated in the union's own order -- ``chunk`` therefore changes the
    batching and nothing else, which the chunk-invariance test pins.
    """
    from src.localization.candidates import candidate_metadata

    positions = np.asarray(positions_cam, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions_cam must be [U, 3], got shape {positions.shape}")
    chunk = max(1, int(chunk))
    parts = []
    for start in range(0, positions.shape[0], chunk):
        batch = [candidate_metadata(md, positions[row])
                 for row in range(start, min(start + chunk, positions.shape[0]))]
        parts.append(_branch(conditioner, batch, device, ids))
    if not parts:
        raise ValueError("positions_cam is empty: a receiver union must have candidates")
    out = {}
    for key in ids:
        out[key] = [torch.cat([part[key][0] for part in parts], dim=0),
                    None if parts[0][key][1] is None
                    else torch.cat([part[key][1] for part in parts], dim=0)]
    return out


def _select(entry, rows):
    tensor, mask = entry[0], entry[1]
    return [tensor.index_select(0, rows),
            None if mask is None else mask.index_select(0, rows)]


def expand_conditioning(context, source, rows, device="cpu"):
    """Assemble the per-generated-row conditioning from the two caches.

    ``rows`` selects one cached source row per generated row; the query's single
    context row is repeated across all of them, which is exactly the statement
    that the context branch does not depend on the candidate.
    """
    rows = torch.as_tensor(rows, dtype=torch.long).reshape(-1)
    if rows.numel() == 0:
        raise ValueError("rows must select at least one generated row")
    zeros = torch.zeros_like(rows)
    merged = {}
    for key, entry in context.items():
        if entry[0].shape[0] != 1:
            raise ValueError(f"the cached context branch {key!r} has batch "
                             f"{entry[0].shape[0]}, not 1; it is computed once per query")
        merged[key] = _select(entry, zeros.to(entry[0].device))
    for key, entry in source.items():
        merged[key] = _select(entry, rows.to(entry[0].device))
    return merged


def receiver_union(index_lists):
    """The ascending, deduplicated union of a receiver's candidate index sets.

    This is the set the G1 cost gate counted conditioner calls over: a receiver
    whose queries ask for ``{0,1,2}`` and ``{0,1,3}`` needs four calls, not six.
    """
    union = set()
    for indices in index_lists:
        union.update(int(i) for i in indices)
    return sorted(union)


def tensor_digest(value):
    """sha256 over a tensor's exact bytes, dtype and shape."""
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    payload = json.dumps([str(tensor.dtype), list(tensor.shape)], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8") + tensor.numpy().tobytes()).hexdigest()


class ReceiverCache:
    """The source branch over ONE receiver's candidate union (§1.5).

    Bounded on purpose: the engine holds a single instance, builds it when the
    receiver's first query is reached and drops it when its last one is done, so
    the resident footprint is one receiver group however many receivers a room
    has.
    """

    def __init__(self, receiver_id, indices, conditioning, depth_digest):
        self.receiver_id = str(receiver_id)
        self.indices = [int(i) for i in indices]
        self.row_of_index = {index: row for row, index in enumerate(self.indices)}
        self.conditioning = conditioning
        self.depth_digest = depth_digest

    @classmethod
    def build(cls, conditioner, receiver_id, base_md, indices, positions_cam, device,
              chunk=256):
        indices = [int(i) for i in indices]
        if len(set(indices)) != len(indices):
            raise ValueError(f"receiver {receiver_id!r}: the union repeats a candidate index")
        positions = np.asarray(positions_cam, dtype=np.float64)
        if positions.shape[0] != len(indices):
            raise ValueError(f"receiver {receiver_id!r}: {positions.shape[0]} positions for "
                             f"{len(indices)} candidates")
        conditioning = source_conditioning(conditioner, base_md, positions, device, chunk=chunk)
        return cls(receiver_id, indices, conditioning, tensor_digest(base_md["depth"]))

    @property
    def n_candidates(self):
        return len(self.indices)

    @property
    def n_conditioner_rows(self):
        return int(self.conditioning[SOURCE_COND_IDS[0]][0].shape[0])

    def rows_for(self, indices):
        """The cache rows serving one query's candidate list, in the query's order."""
        rows = []
        for index in indices:
            row = self.row_of_index.get(int(index))
            if row is None:
                raise ValueError(f"candidate {int(index)} is not in the receiver union of "
                                 f"{self.receiver_id!r}; the cache was built from a different "
                                 "candidate manifest")
            rows.append(row)
        return torch.tensor(rows, dtype=torch.long)

    def assert_same_depth(self, md):
        """The receiver's panorama is what makes the cache reusable -- prove it.

        ``source_vit`` reads ``depth``; if two queries of the same receiver
        carried different panoramas the cached rows would be wrong for one of
        them, and nothing downstream would notice.
        """
        found = tensor_digest(md["depth"])
        if found != self.depth_digest:
            raise ValueError(f"receiver {self.receiver_id!r}: this query's depth panorama "
                             f"({found[:12]}...) is not the one the source cache was built "
                             f"from ({self.depth_digest[:12]}...)")
        return True


# --------------------------------------------------------------------------- #
# the G1 binding: what a query is allowed to be scored against
# --------------------------------------------------------------------------- #
def file_sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


class AuditPlan:
    """The verified G1 audit: its branch, its rooms and their manifest paths."""

    def __init__(self, report_path, report, rooms, branch):
        self.report_path = str(report_path)
        self.report = report
        self.rooms = dict(rooms)
        self.branch = str(branch)
        self.report_sha256 = file_sha256(report_path)
        self.out_dir = os.path.dirname(os.path.abspath(str(report_path)))

    @property
    def n_queries(self):
        return int(self.report.get("n_queries", 0))


def load_audit_plan(report_path, branch=None):
    """Re-accept the whole published G1 audit before a single query is scored.

    The chain verifier reconstructs every room's candidate coordinates from the
    npz and re-derives every digest, so a manifest edited after publication --
    or a sidecar that does not belong to it -- is a refusal here rather than a
    quietly different candidate set at generation time.
    """
    from src.localization.meshgrid_geometry import verify_report_chain

    with open(str(report_path)) as handle:
        report = json.load(handle)
    if report.get("diagnostics_only"):
        raise ValueError(f"{report_path} is a diagnostics-only report; it carries no candidate "
                         "manifests and may not bind a scored run")
    verdict = verify_report_chain(str(report_path))
    if not verdict["ok"]:
        raise ValueError(f"the G1 audit does not re-verify: {verdict['reasons'][0]}")

    registered = ((report.get("branch") or {}).get("branch"))
    if branch is not None and str(branch) != str(registered):
        raise ValueError(f"the audit selected the {registered!r} branch but this run asks for "
                         f"{branch!r}; the branch is decided by geometry before generation "
                         "(inherited plan §1.2) and is not a run-time choice")
    out_dir = os.path.dirname(os.path.abspath(str(report_path)))
    rooms = {room: os.path.join(out_dir, entry["candidate_manifest"])
             for room, entry in (report.get("rooms") or {}).items()}
    return AuditPlan(report_path, report, rooms, registered)


class QueryPlan:
    """One query's candidate set, as the audit published it."""

    def __init__(self, position, query_id, room_id, receiver_id, receiver_xyz,
                 candidate_indices, coordinates, oracle, branch, z_band, n_contexts):
        self.position = int(position)
        self.query_id = str(query_id)
        self.room_id = str(room_id)
        self.receiver_id = str(receiver_id)
        self.receiver_xyz = np.asarray(receiver_xyz, dtype=np.float64)
        self.candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
        self.coordinates = np.asarray(coordinates, dtype=np.float64)
        self.oracle = float(oracle)
        self.branch = str(branch)
        self.z_band = list(z_band) if z_band is not None else None
        self.n_contexts = int(n_contexts)

    @property
    def n_candidates(self):
        return int(self.candidate_indices.size)


class RoomPlan:
    """One room's queries, in stream order, on the audit's chosen branch."""

    def __init__(self, room_id, branch, base, queries, manifest_path, manifest_sha256):
        self.room_id = str(room_id)
        self.branch = str(branch)
        self.base = base
        self.queries = list(queries)
        self.manifest_path = str(manifest_path)
        self.manifest_sha256 = str(manifest_sha256)


#: the manifest keys each branch's indices and count live under.
BRANCH_KEYS = {"full_height": ("candidate_indices", "n_candidates"),
               "z_band": ("candidate_indices_z_band", "n_candidates_z_band")}


def load_room_plan(plan, room_id):
    """One room's candidate manifest, resolved to coordinates (bounded to a room).

    Kept per room on purpose: the biggest room's manifest is 137 MB of index
    lists, so the engine reads one at a time and turns the indices into int64
    arrays immediately.
    """
    path = plan.rooms.get(room_id)
    if path is None:
        raise ValueError(f"the audit publishes no candidate manifest for room {room_id!r}")
    with open(path) as handle:
        manifest = json.load(handle)
    if manifest.get("chosen_branch") != plan.branch:
        raise ValueError(f"{room_id}: the manifest was published on the "
                         f"{manifest.get('chosen_branch')!r} branch but the audit report "
                         f"selects {plan.branch!r}")
    npz_path = os.path.join(plan.out_dir, manifest["coordinates_npz"])
    with np.load(npz_path) as data:
        base = np.asarray(data["base_candidates"], dtype=np.float64)
    index_key, count_key = BRANCH_KEYS[plan.branch]

    queries = []
    for entry in manifest["queries"]:
        indices = np.asarray(entry[index_key], dtype=np.int64)
        if indices.size != int(entry[count_key]) or indices.size == 0:
            raise ValueError(f"{entry['query_id']}: the {plan.branch} branch carries "
                             f"{indices.size} indices for a declared {entry[count_key]}")
        queries.append(QueryPlan(
            position=entry["position"], query_id=entry["query_id"], room_id=room_id,
            receiver_id=entry["receiver_id"], receiver_xyz=entry["receiver"],
            candidate_indices=indices, coordinates=base[indices],
            oracle=(entry.get("oracle") or {})[plan.branch], branch=plan.branch,
            z_band=entry.get("z_band"), n_contexts=entry.get("n_contexts", 0)))
    queries.sort(key=lambda query: query.position)
    from src.localization.meshgrid_geometry import manifest_json_sha256

    return RoomPlan(room_id, plan.branch, base, queries, path, manifest_json_sha256(manifest))


class ReceiverGroup:
    """One receiver's queries and the candidate union they are served from."""

    def __init__(self, receiver_id, receiver_xyz, queries, union):
        self.receiver_id = str(receiver_id)
        self.receiver_xyz = np.asarray(receiver_xyz, dtype=np.float64)
        self.queries = list(queries)
        self.union = list(union)


def receiver_groups(room_plan):
    """The room's receiver groups, first-appearance ordered, stream-ordered inside.

    Grouping is what turns 8.9 M candidate-query pairs into 966 k conditioner
    calls; the order is derived from the manifest so two runs build the same
    groups in the same sequence.
    """
    order, buckets = [], {}
    for query in room_plan.queries:
        if query.receiver_id not in buckets:
            order.append(query.receiver_id)
            buckets[query.receiver_id] = []
        buckets[query.receiver_id].append(query)
    groups = []
    for receiver_id in order:
        queries = sorted(buckets[receiver_id], key=lambda query: query.position)
        union = receiver_union([query.candidate_indices for query in queries])
        groups.append(ReceiverGroup(receiver_id, queries[0].receiver_xyz, queries, union))
    return groups


# --------------------------------------------------------------------------- #
# the D1 binding: the contexts a query is generated from
# --------------------------------------------------------------------------- #
def verify_context_record(md, record, position):
    """The loader's draw IS the registered one, checked before it conditions anything.

    The D1 manifest froze each query's context fingerprints and the sha256 of
    every context RIR's exact float32 bytes. Recomputing them from the live
    stream is what makes the manifest binding executable rather than
    documentary: a different worker count, a re-ordered split or a substituted
    item all show up here.
    """
    from src.localization import meshgrid_queries as mq

    found = mq.context_record(md, position, eligible=record.get("eligible", 0))
    if found["query_id"] != record["query_id"]:
        raise ValueError(f"stream position {position}: the loader delivered query_id "
                         f"{found['query_id']!r} where the context manifest registers "
                         f"{record['query_id']!r}")
    if int(record["position"]) != int(position):
        raise ValueError(f"stream position {position}: the context manifest registers this "
                         f"query at position {record['position']}; the pass is out of order")
    if found["context_fingerprints"] != list(record["context_fingerprints"]):
        raise ValueError(f"{record['query_id']}: the context fingerprint set differs from the "
                         "frozen D1 manifest; this query would be conditioned on a different "
                         "draw than every other arm")
    if found["context_audio_sha256"] != list(record["context_audio_sha256"]):
        raise ValueError(f"{record['query_id']}: a context audio digest differs from the frozen "
                         "D1 manifest; the context RIR bytes are not the registered ones")
    return True


def assert_room_blocks(records):
    """Each room must arrive as ONE contiguous block of the stream.

    The engine buffers a room's per-query context branch and then walks that
    room's receiver groups, so a room split across the stream would either
    blow the bound or silently drop a group. It holds on the registered split;
    it is asserted rather than assumed.
    """
    order, spans = [], {}
    for record in records:
        room = record["room_id"]
        position = int(record["position"])
        if room not in spans:
            order.append(room)
            spans[room] = [position, position, 0]
        spans[room][1] = max(spans[room][1], position)
        spans[room][0] = min(spans[room][0], position)
        spans[room][2] += 1
    for room in order:
        low, high, count = spans[room]
        if high - low + 1 != count:
            raise ValueError(f"room {room!r} is not contiguous in the stream: {count} queries "
                             f"spread over positions {low}..{high}")
    return order


def assert_receiver_consistent(md, receiver_xyz, coordinates, oracle, tol=1e-4):
    """The candidate coordinates belong to THIS query's receiver.

    ``md['source']`` is the ground-truth source in the receiver's own frame, so
    ``receiver + md['source']`` is the global truth G1 measured its oracle
    against. Recomputing that minimum from the published coordinates is a cheap
    end-to-end check that the manifest row, the receiver and the loader item are
    the same query -- an offset receiver moves it immediately.
    """
    receiver = np.asarray(receiver_xyz, dtype=np.float64).reshape(3)
    source = torch.as_tensor(md["source"]).detach().cpu().to(torch.float64).reshape(-1).numpy()
    if source.size != 3:
        raise ValueError(f"md['source'] must be 3 coordinates, got {source.size}")
    truth = receiver + source
    found = float(np.linalg.norm(np.asarray(coordinates, dtype=np.float64) - truth,
                                 axis=1).min())
    if abs(found - float(oracle)) > float(tol):
        raise ValueError(f"the oracle recomputed from the loader's own geometry is {found:.6f} m "
                         f"but the candidate manifest records {float(oracle):.6f} m; the "
                         "candidate set does not belong to this receiver")
    return True
