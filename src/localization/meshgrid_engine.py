"""exp_22 I1 -- the frozen mesh-grid localization engine (inherited plan §1.4/§1.5).

One query is an observed RIR ``h_obs``; the engine regenerates one RIR per
mesh-valid grid candidate under the frozen Vanilla FLAC checkpoint, scores each
against ``h_obs`` in AGREE's audio embedding space, and predicts the argmax
candidate. Everything that decides a number -- the noise a candidate is drawn
from, the context tensors it is conditioned on, the candidate set it competes in
-- is bound to a registered artifact (the D1 context manifest, the G1 candidate
manifests, the checkpoint, the scorer) and refused when it does not match.

Five points are RECORDED DEVIATIONS from the inherited plan text, stamped into
the run binding and the published rows rather than silently taken:

* ``SCORER_READOUT`` -- §1.4 pins ``encode_audio(..., normalize=True)``. That
  path samples AGREE's VAE bottleneck, which exp_18 measured as ~7e-5 of cosine
  noise per call and which consumes the global RNG stream. The registered
  scorer here is exp_18/exp_20's deterministic mean readout, which is the same
  arithmetic with the bottleneck's mean substituted for its sample.
* ``NOISE_KEY_POLICY`` -- resolved: §1.1's common random numbers is the
  registered policy and a registered pass refuses anything else. The
  per-candidate key of the r7 dispatch remains implemented and reachable only by
  an explicit opt-in, so the r7 evidence stays reproducible.
* ``SIMS_PRECISION_CAVEAT`` -- per-sample similarities are a float16 sidecar per
  QUERY, not per room: the atomic-resume contract is per query, and a room-level
  pack would lose finished queries on a mid-room kill. Every aggregate the
  protocol reads stays float32.
* ``DUMP_CONTENT_RULE`` -- exp_18 dumped every candidate because M was ~10; here
  M averages 1,667, so a dump is a bounded, score-derived selection.
* ``BATCHING_CAVEAT`` -- the caches are a proven-exact memoization at equal
  batching (``cache_parity_check``), but the production chunking makes a run
  differ from an unchunked one by about one float16 ulp. That is the backbone's
  own batch nondeterminism, disclosed rather than claimed away.
"""
import hashlib
import json
import math
import os

from collections.abc import Mapping
from dataclasses import dataclass

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

#: The registered scope, as the published G1 audit measured it. A probe's
#: projection and the shard merge's census are both stated against these, so
#: they are pinned here rather than recomputed from whatever is on disk.
REGISTERED_TOTALS = {
    "rooms": 16,
    "queries": 5337,
    "candidate_query_pairs": 8896540,
    "source_rows": 966147,
    "generated_waveforms": 8896540 * 8,
}

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

#: The registered bound on |S_a - S_b| between two passes of the SAME protocol
#: that differ only in batching. It is the acceptance criterion for a
#: changed-batching replay, and it is provisional in exactly one sense: the
#: number below is the reviewed bound, and the ladder's real changed-batching
#: replay measures the value -- a measurement above this refuses rather than
#: relaxes.
SCORE_TOLERANCE = 1e-3

DETERMINISM_CONTRACT = (
    "At fixed batching every stage is deterministic and a replay is bit-exact through "
    "scoring: the noise is keyed, the K prefixes are slices of one sequence, the caches are "
    "a proven-exact memoization (cache_parity_check), the AGREE readout is the deterministic "
    "VAE mean, and nothing in the pass consumes the global RNG. Two passes over the same "
    "artifacts at the same batch_rows/source_chunk must therefore produce identical score "
    "fingerprints. Changed batching is a different question: the backbones' GEMM tiling "
    "moves an output by about one float16 ulp, so the passes are compared against "
    "SCORE_TOLERANCE and every query whose top-1 margin is inside that tolerance -- i.e. "
    "whose argmax COULD flip -- is counted and named, never silently accepted")

NOISE_KEY_POLICIES = ("per_candidate", "shared_across_candidates")
#: The REGISTERED policy: common random numbers -- inherited plan §1.1, "All
#: candidates for a query share receiver, depth panorama, context RIRs, context
#: poses, sample count, and seeds". The r7 dispatch keyed the draw per candidate
#: and the r7 review ruled for CRN, which is also the variance-reduction the
#: candidate comparison relies on: with a shared draw, a difference between two
#: candidates' scores is a difference between the CANDIDATES.
REGISTERED_NOISE_POLICY = "shared_across_candidates"
NOISE_KEY_POLICY = REGISTERED_NOISE_POLICY


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


def assert_registered_noise_policy(policy, allow_unregistered=False):
    """A registered pass draws under common random numbers or refuses.

    The alternative is not deleted -- a ruling can still be re-examined and the
    r7 evidence re-derived -- but it cannot be reached by a default, only by an
    explicit opt-in that no production entry point offers.
    """
    if str(policy) == REGISTERED_NOISE_POLICY:
        return True
    if not allow_unregistered:
        raise ValueError(
            f"noise policy {policy!r} is not the registered one: inherited plan §1.1 fixes "
            "common random numbers across a query's candidates, so every candidate is scored "
            f"against the SAME K draws ({REGISTERED_NOISE_POLICY!r}). Scoring candidates "
            "under different noise makes a score difference partly a sampling difference")
    if str(policy) not in NOISE_KEY_POLICIES:
        raise ValueError(f"unknown noise policy {policy!r}")
    return True


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


def top1_margin(scores):
    """``S[best] - S[runner-up]`` -- how far the argmax is from flipping.

    A tie is a zero margin (the tie-break decided it, not the score), and a
    single candidate has nothing to flip to.
    """
    if not isinstance(scores, torch.Tensor):
        scores = torch.as_tensor(scores)
    values = scores.reshape(-1).float()
    if values.numel() < 2:
        return float("inf")
    top = torch.topk(values, 2).values
    return float(top[0] - top[1])


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
        margin = top1_margin(block["scores"])
        by_k[k] = {
            "margin": margin,
            "argmax_stable": bool(margin > SCORE_TOLERANCE),
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


#: candidates per source-branch forward. Deliberately small: ``source_vit`` is a
#: GeometryConditioner, so EVERY candidate is a full ViT forward over a
#: ``[3, 256, 512]`` coordinate-minus-depth map (conditioners.py:284-296) -- the
#: 966 k calls the G1 gate counted. A chunk of 256 would stage 400 MB of input
#: before the backbone even runs.
SOURCE_CHUNK = 16


def source_conditioning(conditioner, md, positions_cam, device, chunk=SOURCE_CHUNK,
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
              chunk=SOURCE_CHUNK):
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
        return self.assert_same_depth_digest(tensor_digest(md["depth"]))

    def assert_same_depth_digest(self, found):
        """The digest form: the pass records one digest per query but keeps only
        one panorama TENSOR per receiver (a per-query copy is 1.5 MB)."""
        if found != self.depth_digest:
            raise ValueError(f"receiver {self.receiver_id!r}: this query's depth panorama "
                             f"({found[:12]}...) is not the one the source cache was built "
                             f"from ({self.depth_digest[:12]}...)")
        return True


# --------------------------------------------------------------------------- #
# the leakage guard
# --------------------------------------------------------------------------- #
class LeakageError(RuntimeError):
    """Raised when engine code reaches for the held-out target."""


class GuardedMetadata(Mapping):
    """A loader item with the TARGET fields made unreadable.

    The engine localizes a hidden source: it may read the observation, the
    contexts and the panorama, and it may not read where the source actually is.
    ``md['source']`` is right there in every loader item, so the protection is
    made structural rather than editorial -- any read, including a wholesale
    ``dict(md)`` copy, raises. The keys stay VISIBLE in iteration, so this is a
    guard and not a quiet deletion (r7 review BLOCKER GT).
    """

    BLOCKED = ("source", "source_vit")

    def __init__(self, md, blocked=BLOCKED):
        self._md = md
        self._blocked = frozenset(blocked)

    def __getitem__(self, key):
        if key in self._blocked:
            raise LeakageError(
                f"the engine read {key!r}: that is the held-out target, and exp_22 localizes "
                "it. Every geometry check the engine needs is derivable from the manifest "
                "receiver and the context poses (assert_query_geometry_consistent)")
        return self._md[key]

    def __iter__(self):
        return iter(self._md)

    def __len__(self):
        return len(self._md)

    def without_target(self):
        """A plain dict copy with the target fields DROPPED, not read."""
        return {key: self._md[key] for key in self._md if key not in self._blocked}


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
                 candidate_indices, base, oracle, branch, z_band, n_contexts,
                 n_dropped_receiver=None, n_dropped_context=None):
        self.position = int(position)
        self.query_id = str(query_id)
        self.room_id = str(room_id)
        self.receiver_id = str(receiver_id)
        self.receiver_xyz = np.asarray(receiver_xyz, dtype=np.float64)
        self.candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
        #: the room's base candidate array, shared by reference -- materializing
        #: one coordinate copy per query would hold 117 MB for the largest room.
        self.base = base
        self.oracle = float(oracle)
        self.branch = str(branch)
        self.z_band = list(z_band) if z_band is not None else None
        self.n_contexts = int(n_contexts)
        #: the FULL-HEIGHT drop counts, as the audit records them
        self.n_dropped_receiver = (None if n_dropped_receiver is None
                                   else int(n_dropped_receiver))
        self.n_dropped_context = (None if n_dropped_context is None
                                  else int(n_dropped_context))

    @property
    def coordinates(self):
        """The query's candidate coordinates ``[M, 3]``, resolved on demand."""
        return self.base[self.candidate_indices]

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
            candidate_indices=indices, base=base,
            oracle=(entry.get("oracle") or {})[plan.branch], branch=plan.branch,
            z_band=entry.get("z_band"), n_contexts=entry.get("n_contexts", 0),
            n_dropped_receiver=entry.get("n_dropped_receiver"),
            n_dropped_context=entry.get("n_dropped_context")))
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

    # prove_target_absent=False: the ENGINE may not read md['source'] (r7 review
    # BLOCKER GT). D1 materialization already proved it and froze the verdict.
    found = mq.context_record(md, position, eligible=record.get("eligible", 0),
                              prove_target_absent=False)
    if record.get("target_absent") is not True:
        raise ValueError(
            f"{record.get('query_id')!r}: the context manifest does not record "
            "target_absent=True. The engine cannot re-derive it without reading the "
            "held-out target, so a draw whose target-absence was never proven at "
            "materialization time may not be scored")
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


#: the audit's own recovery-join tolerance. The engine rebuilds each context's
#: global position as ``receiver + md['context_poses']`` in float32, while G1
#: read it from the float64 metadata anchors; only a candidate sitting within
#: microns of a guard boundary can be decided differently by that difference.
CONTEXT_JOIN_TOLERANCE = 1e-3


def context_globals(md, receiver_xyz):
    """``[N, 3]`` global context-source positions -- GT-free by construction."""
    poses = torch.as_tensor(md["context_poses"]).detach().cpu().to(torch.float64).numpy()
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
    return poses + np.asarray(receiver_xyz, dtype=np.float64).reshape(1, 3)


def _boundary_slack(point, receiver, contexts, z_band, eps):
    """How close a candidate sits to the nearest guard it could be decided by."""
    from src.localization.meshgrid_geometry import (CONTEXT_GUARD_RADIUS,
                                                    RECEIVER_MIN_DISTANCE)

    slacks = [abs(float(np.linalg.norm(point - receiver)) + eps - RECEIVER_MIN_DISTANCE)]
    if len(contexts):
        distances = np.linalg.norm(contexts - point.reshape(1, 3), axis=1)
        slacks.append(float(np.abs(distances - eps - CONTEXT_GUARD_RADIUS).min()))
    if z_band is not None:
        slacks.append(min(abs(float(point[2]) + eps - float(z_band[0])),
                          abs(float(point[2]) - eps - float(z_band[1]))))
    return min(slacks)


def assert_query_geometry_consistent(md, query, tol=CONTEXT_JOIN_TOLERANCE):
    """Re-derive this query's candidate set from GT-FREE inputs, or refuse.

    The r7 engine proved a query and its manifest row belonged together by
    recomputing G1's oracle -- which reads the target. This does strictly more
    without it: from the manifest's receiver, the live context poses and the
    room's base bank it re-derives the z-band, the two drop counts and the whole
    candidate index set, and requires them to match. A receiver from another
    query, contexts from another draw, or a manifest row attached to the wrong
    position all change that reconstruction.

    A candidate whose membership differs is tolerated only when it sits within
    ``tol`` of the guard boundary that decides it -- the float32-vs-float64
    context-anchor difference can move nothing else -- and every tolerated case
    is COUNTED and returned rather than silently absorbed.
    """
    from src.localization import meshgrid_geometry as mg

    receiver = np.asarray(query.receiver_xyz, dtype=np.float64).reshape(3)
    contexts = context_globals(md, receiver)
    if query.n_contexts and contexts.shape[0] != int(query.n_contexts):
        raise ValueError(f"{query.query_id}: the loader delivered {contexts.shape[0]} context "
                         f"poses but the candidate manifest records {query.n_contexts}")

    band = mg.context_z_band(contexts)
    if query.z_band is not None:
        drift = max(abs(band[0] - float(query.z_band[0])), abs(band[1] - float(query.z_band[1])))
        if drift > tol:
            raise ValueError(
                f"{query.query_id}: the z-band derived from this query's contexts is "
                f"[{band[0]:.6f}, {band[1]:.6f}] but the manifest records "
                f"{query.z_band}; these contexts are not the ones the candidate set was "
                "built from")

    try:
        full = mg.filter_query_candidates(query.base, receiver=receiver,
                                          context_sources=contexts)
        branch_band = None if query.branch == "full_height" else query.z_band
        kept = (full if branch_band is None else
                mg.filter_query_candidates(query.base, receiver=receiver,
                                           context_sources=contexts, z_band=branch_band))
    except ValueError as error:
        raise ValueError(f"{query.query_id}: the candidate set could not be reconstructed "
                         f"from the manifest receiver and this query's contexts: {error}") from error

    rebuilt = set(int(i) for i in np.flatnonzero(kept["mask"]))
    published = set(int(i) for i in query.candidate_indices)
    tolerated, hard = [], []
    for index in sorted(rebuilt ^ published):
        slack = _boundary_slack(np.asarray(query.base[index], dtype=np.float64), receiver,
                                contexts, branch_band, mg.EPS)
        entry = {"index": int(index), "slack": float(slack),
                 "side": "rebuilt_only" if index in rebuilt else "published_only"}
        (tolerated if slack <= tol else hard).append(entry)
    if hard:
        raise ValueError(
            f"{query.query_id}: the candidate set reconstructed from the manifest receiver and "
            f"this query's contexts differs on {len(hard)} candidate(s) that are not near any "
            f"guard boundary (first {hard[:3]}); the manifest row, the receiver and the loader "
            "item are not the same query")

    for name, rebuilt_count, published_count in (
            ("receiver", full["n_dropped_receiver"], query.n_dropped_receiver),
            ("context", full["n_dropped_context"], query.n_dropped_context)):
        if published_count is None:
            continue
        if abs(int(rebuilt_count) - int(published_count)) > len(tolerated):
            raise ValueError(
                f"{query.query_id}: the {name} guard drops {rebuilt_count} candidates here but "
                f"the manifest records {published_count}; that is more than the "
                f"{len(tolerated)} boundary-grazing candidate(s) this reconstruction tolerated")
    return {"reconstructed": len(rebuilt), "published": len(published),
            "n_tolerated": len(tolerated), "tolerated": tolerated,
            "z_band": [float(band[0]), float(band[1])]}


# --------------------------------------------------------------------------- #
# the run binding: what a resume is allowed to continue
# --------------------------------------------------------------------------- #
#: every quantity that decides a number. A resume that changes ANY of these is
#: a different experiment sharing a directory, so it is refused rather than
#: mixed into one artifact set.
RUN_BINDING_FIELDS = (
    "model_config_sha256", "ckpt_sha256", "agree_ckpt_sha256", "d1_manifest_sha256",
    "g1_report_sha256", "room_manifest_sha256", "branch", "k_prefixes", "num_samples",
    "tau", "seed", "noise_policy", "steps", "cfg_scale", "cond_method", "scorer_readout",
    # the conditioner's ARITHMETIC and the split's BYTES, not just its pathname:
    # without them a resume could mix autocast modes or an edited dataset config
    "cond_autocast", "dataset_config_sha256", "dataset_config")

#: recorded and compared, but NOT part of the strict digest. These change only
#: the batch SHAPES the backbones see; under the registered autocast that moves
#: an output by about one float16 ulp (measured on the real conditioner), which
#: is the model's own batch nondeterminism rather than a different protocol.
#: Refusing a resume over them would make an OOM unrecoverable, so a change is
#: reported instead.
RUN_BINDING_ADVISORY = ("source_chunk", "batch_rows")

BATCHING_CAVEAT = (
    "source_chunk and batch_rows change the batch shapes the ViT, the DiT, the VAE and the "
    "AGREE tower are called with. The registered --cond-autocast default runs the "
    "conditioners in float16 on CUDA, where a changed batch shape perturbs an output by "
    "about one ulp (measured: max |diff| 3.9e-3 between the batch-1 context call and an "
    "8-candidate call; the source branch was bit-identical at equal batching). A pass "
    "re-chunked mid-run is therefore NOT bit-identical to one chunked uniformly -- it is the "
    "same protocol at the backbone's own numerical noise. Within a run every query of a "
    "receiver still shares bit-identical source tokens, because they are served from one "
    "cache, and cache_parity_check proves the cache itself is exact at equal batching")

BINDING_FILENAME = "run_binding.json"


def binding_sha256(binding):
    """Canonical, type-sensitive digest of a complete run binding."""
    from src.localization.crossarm import canonical_sha256

    missing = [field for field in RUN_BINDING_FIELDS if field not in binding]
    if missing:
        raise ValueError(f"the run binding is missing {missing}; every registered field must be "
                         "pinned before a query is generated")
    extra = [key for key in binding if key not in RUN_BINDING_FIELDS]
    if extra:
        raise ValueError(f"the run binding carries unregistered fields {extra}")
    return canonical_sha256({field: binding[field] for field in sorted(RUN_BINDING_FIELDS)})


def write_binding(out_dir, binding, advisory=None):
    """Publish the binding beside the artifacts it authorizes."""
    os.makedirs(str(out_dir), exist_ok=True)
    payload = {field: binding[field] for field in RUN_BINDING_FIELDS}
    payload["binding_sha256"] = binding_sha256(binding)
    payload["advisory"] = {field: (advisory or {}).get(field)
                           for field in RUN_BINDING_ADVISORY}
    payload["advisory_history"] = []
    payload["batching_caveat"] = BATCHING_CAVEAT
    path = os.path.join(str(out_dir), BINDING_FILENAME)
    write_json(path, payload)
    return path


def record_advisory_change(out_dir, changed, advisory=None, at_utc=None):
    """Persist an advisory (batching) change into the published binding.

    Printing it to stdout loses it the moment the terminal scrolls; the run's
    own artifact has to say which batching produced which part of the pass.
    Appends to ``advisory_history`` and adopts the new values, leaving the
    strict binding digest untouched.
    """
    from datetime import datetime, timezone

    path = os.path.join(str(out_dir), BINDING_FILENAME)
    with open(path) as handle:
        published = json.load(handle)
    history = list(published.get("advisory_history") or [])
    history.append({"at_utc": at_utc or datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "changed": {field: dict(value) for field, value in changed.items()},
                    "batching_caveat": BATCHING_CAVEAT})
    published["advisory_history"] = history
    published["advisory"] = {field: (advisory or {}).get(field)
                             for field in RUN_BINDING_ADVISORY}
    write_json(path, published)
    return path


def assert_binding(out_dir, binding, advisory=None):
    """A resume continues the SAME run or refuses, naming the fields that moved.

    Returns ``True`` when everything matches, or a dict of the ADVISORY fields
    that moved -- those are reported to the operator, not refused.
    """
    path = os.path.join(str(out_dir), BINDING_FILENAME)
    if not os.path.isfile(path):
        raise ValueError(f"{out_dir} holds no {BINDING_FILENAME}; a resume may not adopt "
                         "artifacts whose provenance is unknown")
    with open(path) as handle:
        published = json.load(handle)
    differing = [field for field in RUN_BINDING_FIELDS
                 if published.get(field) != binding.get(field)]
    if differing:
        raise ValueError(f"this run does not continue the published one: {differing} differ "
                         f"(published binding {str(published.get('binding_sha256'))[:12]}..., "
                         f"this run {binding_sha256(binding)[:12]}...)")
    was = published.get("advisory") or {}
    moved = {field: {"published": was.get(field), "this_run": (advisory or {}).get(field)}
             for field in RUN_BINDING_ADVISORY
             if was.get(field) != (advisory or {}).get(field)}
    return moved or True


# --------------------------------------------------------------------------- #
# per-query artifacts and resume
# --------------------------------------------------------------------------- #
#: declared precision of the per-sample similarity sidecar.
SIMS_DTYPE = "float16"
SIMS_PRECISION_CAVEAT = (
    "per-sample similarities s[x, k] are stored as float16 (~3 decimal digits) to keep the "
    "sidecars at 2 bytes per generated waveform; every AGGREGATE the protocol reads -- S at "
    "each K and S_mean -- is published at full float32 precision in the row, so the float16 "
    "array is a diagnostic, never the source of a headline number")

ROWS_DIRNAME = "rows"

#: fields that are ABOUT the row rather than part of what it claims.
_ROW_DIGEST_EXCLUDED = ("row_sha256",)


def row_digest(row):
    """Canonical digest over everything a row CLAIMS.

    The sidecar digest authenticates the similarities; this authenticates the
    predictions, the oracle, the candidate indices, the receiver and the binding
    identity, so an edited row cannot be adopted by a resume (r7 review BLOCKER
    RESUME).
    """
    from src.localization.crossarm import canonical_sha256

    return canonical_sha256({key: value for key, value in row.items()
                             if key not in _ROW_DIGEST_EXCLUDED})


def write_json(path, payload):
    """Publish one JSON artifact atomically (tmp file, then rename)."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as handle:
        handle.write(json.dumps(payload, sort_keys=True, indent=None) + "\n")
    os.replace(tmp, path)
    return path


def room_stem(room_id):
    return str(room_id).replace("/", "_")


def query_artifact_paths(out_dir, room_id, position):
    room_dir = os.path.join(str(out_dir), ROWS_DIRNAME, room_stem(room_id))
    stem = f"q{int(position):05d}"
    return {"dir": room_dir, "row": os.path.join(room_dir, stem + ".json"),
            "sims": os.path.join(room_dir, stem + "_sims.npy")}


def write_query_artifact(out_dir, row, sims, binding_sha256=None):
    """One query's artifacts, published atomically and sidecar-first.

    The sidecar lands before the row, and the row carries its digest, so the row
    is the completion marker: a killed pass can leave an orphan sidecar (which
    the next pass overwrites) but never a row whose sidecar is missing or stale.
    """
    paths = query_artifact_paths(out_dir, row["room_id"], row["position"])
    os.makedirs(paths["dir"], exist_ok=True)
    array = np.asarray(torch.as_tensor(sims).detach().cpu().numpy(), dtype=np.float16)
    if array.ndim != 2 or array.shape != (int(row["n_candidates"]), int(row["num_samples"])):
        raise ValueError(f"sims shape {array.shape} does not match the row's "
                         f"({row['n_candidates']}, {row['num_samples']})")
    tmp_sims = paths["sims"] + ".tmp"
    with open(tmp_sims, "wb") as handle:
        np.save(handle, array)
    os.replace(tmp_sims, paths["sims"])

    published = dict(row)
    published.update({"sims_path": os.path.relpath(paths["sims"], str(out_dir)),
                      "sims_sha256": file_sha256(paths["sims"]),
                      "sims_dtype": SIMS_DTYPE,
                      "sims_shape": [int(array.shape[0]), int(array.shape[1])],
                      "sims_precision_caveat": SIMS_PRECISION_CAVEAT})
    if binding_sha256 is not None:
        published["binding_sha256"] = str(binding_sha256)
    published["row_sha256"] = row_digest(published)
    write_json(paths["row"], published)
    return paths


def verify_query_artifact(row_path, binding_sha256=None):
    """Re-accept one published query from its own bytes -- row AND sidecars."""
    row_path = str(row_path)
    try:
        with open(row_path) as handle:
            row = json.load(handle)
    except (OSError, ValueError) as error:                # noqa: BLE001 -- reported as a verdict
        return {"ok": False, "reason": f"the row is unreadable: {error}", "query_id": None}
    out_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(row_path))))
    sims_path = os.path.join(out_dir, str(row.get("sims_path", "")))
    verdict = {"ok": False, "query_id": row.get("query_id"), "row": row_path}
    if not os.path.isfile(sims_path):
        return dict(verdict, reason=f"the sims sidecar {row.get('sims_path')!r} is missing")
    if file_sha256(sims_path) != row.get("sims_sha256"):
        return dict(verdict, reason="the sims sidecar does not match the digest the row records")
    array = np.load(sims_path)
    shape = [int(array.shape[0]), int(array.shape[1])] if array.ndim == 2 else list(array.shape)
    if shape != list(row.get("sims_shape", [])):
        return dict(verdict, reason=f"the sims sidecar is {shape}, not the recorded "
                                    f"{row.get('sims_shape')}")
    if shape != [int(row.get("n_candidates", -1)), int(row.get("num_samples", -1))]:
        return dict(verdict, reason=f"the sims sidecar is {shape} but the row declares "
                                    f"{row.get('n_candidates')} candidates x "
                                    f"{row.get('num_samples')} samples")
    # the row's OWN registered prefix set: whether it is the protocol's (1, 4, 8)
    # is the run binding's business, not this artifact's
    prefixes = row.get("k_prefixes") or K_PREFIXES
    if sorted(str(k) for k in (row.get("by_k") or {})) != sorted(str(k) for k in prefixes):
        return dict(verdict, reason=f"the row publishes prefixes "
                                    f"{sorted(row.get('by_k') or {})} for a declared "
                                    f"{list(prefixes)}")
    if "row_sha256" not in row:
        return dict(verdict, reason="the row carries no row_sha256; its predictions, oracle "
                                    "and candidate indices are unauthenticated")
    if row_digest(row) != row.get("row_sha256"):
        return dict(verdict, reason="the row does not match its own row_sha256; a prediction, "
                                    "oracle, candidate index, receiver or binding field was "
                                    "edited after publication")
    if binding_sha256 is not None and row.get("binding_sha256") != str(binding_sha256):
        return dict(verdict, reason=f"the row was produced under binding "
                                    f"{str(row.get('binding_sha256'))[:12]}... but this run is "
                                    f"{str(binding_sha256)[:12]}...")
    if row.get("waveform_path"):
        dump_path = os.path.join(out_dir, str(row["waveform_path"]))
        if not os.path.isfile(dump_path):
            return dict(verdict, reason=f"the waveform dump {row['waveform_path']!r} the row "
                                        "names is missing")
        if file_sha256(dump_path) != row.get("waveform_sha256"):
            return dict(verdict, reason="the waveform dump does not match the digest the row "
                                        "records")
    return {"ok": True, "query_id": row.get("query_id"), "row": row_path, "reason": None,
            "position": row.get("position"), "room_id": row.get("room_id")}


def assert_published_matches(out_dir, query, binding_sha256=None):
    """A SKIPPED query's published row must be the row for THIS query.

    A row can be internally consistent and still describe another query's
    candidates; nothing regenerates a skipped one, so the identity is checked
    against the loaded G1 plan here or the resume refuses (r7 review BLOCKER
    RESUME).
    """
    paths = query_artifact_paths(out_dir, query.room_id, query.position)
    verdict = verify_query_artifact(paths["row"], binding_sha256=binding_sha256)
    if not verdict["ok"]:
        raise ValueError(f"{query.query_id}: the published row cannot be adopted: "
                         f"{verdict['reason']}")
    with open(paths["row"]) as handle:
        row = json.load(handle)
    published = [int(i) for i in row.get("candidate_indices", [])]
    expected = [int(i) for i in query.candidate_indices]
    mismatches = [name for name, left, right in (
        ("query_id", row.get("query_id"), query.query_id),
        ("room_id", row.get("room_id"), query.room_id),
        ("position", int(row.get("position", -1)), query.position),
        ("receiver_id", row.get("receiver_id"), query.receiver_id),
        ("branch", row.get("branch"), query.branch),
        ("n_candidates", int(row.get("n_candidates", -1)), query.n_candidates),
        ("candidate_indices", published, expected)) if left != right]
    if mismatches:
        raise ValueError(f"{query.query_id}: the published row does not match the candidate "
                         f"manifest on {mismatches}; a resume may not adopt a row that "
                         "describes a different query")
    return True


#: the parts of a row that a replay must reproduce exactly -- everything the
#: protocol reads, and nothing that measures the machine it ran on.
SCORE_FINGERPRINT_FIELDS = ("query_id", "room_id", "position", "receiver_id", "branch",
                            "n_candidates", "num_samples", "tau", "seed", "noise_policy",
                            "k_prefixes", "candidate_indices", "by_k", "e_oracle",
                            "sims_sha256")


def score_fingerprint(row):
    """Digest of everything a fixed-batching replay must reproduce bit for bit."""
    from src.localization.crossarm import canonical_sha256

    return canonical_sha256({field: row[field] for field in SCORE_FINGERPRINT_FIELDS
                             if field in row})


def read_rows(out_dir):
    """Every published row of a run, in room/position order."""
    rows = []
    root = os.path.join(str(out_dir), ROWS_DIRNAME)
    if not os.path.isdir(root):
        return rows
    for room in sorted(os.listdir(root)):
        room_dir = os.path.join(root, room)
        if not os.path.isdir(room_dir):
            continue
        for name in sorted(os.listdir(room_dir)):
            if name.endswith(".json"):
                with open(os.path.join(room_dir, name)) as handle:
                    rows.append(json.load(handle))
    return sorted(rows, key=lambda row: int(row["position"]))


def compare_scored_runs(rows_a, rows_b, tolerance=SCORE_TOLERANCE):
    """Compare two passes of the same protocol -- the registered replay report.

    Reports the largest score difference per prefix, whether the two are
    bit-exact, and every query whose argmax MOVED, plus how many sat inside the
    tolerance and could therefore have moved. Nothing is absorbed silently: the
    caller decides, on numbers.
    """
    from src.localization.reaggregate import decode_scores

    left = {row["query_id"]: row for row in rows_a}
    right = {row["query_id"]: row for row in rows_b}
    missing = sorted(set(left) ^ set(right))
    if missing:
        raise ValueError(f"the two runs do not score the same queries; {len(missing)} differ "
                         f"(first {missing[:3]})")
    by_k, flipped, at_risk = {}, [], 0
    fingerprints_equal = True
    for query_id in sorted(left):
        row_a, row_b = left[query_id], right[query_id]
        if score_fingerprint(row_a) != score_fingerprint(row_b):
            fingerprints_equal = False
        for key in sorted(row_a["by_k"], key=int):
            k = int(key)
            block_a, block_b = row_a["by_k"][key], row_b["by_k"][key]
            delta = float((decode_scores(block_a["scores_hex"])
                           - decode_scores(block_b["scores_hex"])).abs().max())
            entry = by_k.setdefault(k, {"max_abs_delta": 0.0, "n_argmax_agree": 0,
                                        "n_queries": 0, "min_margin": float("inf")})
            entry["max_abs_delta"] = max(entry["max_abs_delta"], delta)
            entry["n_queries"] += 1
            entry["min_margin"] = min(entry["min_margin"], float(block_a.get("margin", 0.0)))
            if block_a["prediction_index"] == block_b["prediction_index"]:
                entry["n_argmax_agree"] += 1
            else:
                flipped.append({"query_id": query_id, "k": k,
                                "a": block_a["prediction_index"],
                                "b": block_b["prediction_index"],
                                "margin": float(block_a.get("margin", 0.0))})
            if float(block_a.get("margin", 0.0)) <= float(tolerance):
                at_risk += 1
    max_delta = max((entry["max_abs_delta"] for entry in by_k.values()), default=0.0)
    return {"n_queries": len(left), "by_k": by_k, "max_abs_delta": max_delta,
            "tolerance": float(tolerance), "within_tolerance": max_delta <= float(tolerance),
            "bit_exact": bool(fingerprints_equal and max_delta == 0.0),
            "n_flipped": len(flipped), "flipped": flipped,
            "n_argmax_at_risk": at_risk, "contract": DETERMINISM_CONTRACT}


def completed_queries(out_dir, binding_sha256=None):
    """``(verified query ids, rejected verdicts)`` for a resume.

    Only a query whose row AND sidecar re-verify is skipped; anything else is
    reported and regenerated, so a half-written artifact can never be adopted as
    a finished one.
    """
    done, rejected = set(), []
    root = os.path.join(str(out_dir), ROWS_DIRNAME)
    if not os.path.isdir(root):
        return done, rejected
    for room in sorted(os.listdir(root)):
        room_dir = os.path.join(root, room)
        if not os.path.isdir(room_dir):
            continue
        for name in sorted(os.listdir(room_dir)):
            if not name.endswith(".json"):
                continue
            verdict = verify_query_artifact(os.path.join(room_dir, name),
                                            binding_sha256=binding_sha256)
            if verdict["ok"]:
                done.add(verdict["query_id"])
            else:
                rejected.append(verdict)
    return done, rejected


# --------------------------------------------------------------------------- #
# bounded waveform dumps (announcement-08 exemption) and the no-quality probe
# --------------------------------------------------------------------------- #
def registered_probe_queries(plan):
    """The off-grid probe set: one query per room, the lexicographically first.

    Inherited plan §2 registers "the lexicographically first query from each of
    the 16 included rooms". The ordering is on the query's RELPATH, which is the
    stable identity in the split; the position prefix of a ``query_id`` would
    order ``10|`` before ``2|``.
    """
    probes = {}
    for room_id in sorted(plan.rooms):
        room = load_room_plan(plan, room_id)
        chosen = min(room.queries, key=lambda query: query.query_id.split("|", 1)[-1])
        probes[room_id] = chosen.query_id
    return probes


def assert_dump_allowed(requested, allowed):
    """Waveform dumps are bounded to the REGISTERED set or refused."""
    allowed = set(allowed.values()) if isinstance(allowed, dict) else set(allowed)
    outside = sorted(set(str(q) for q in requested) - allowed)
    if outside:
        raise ValueError(f"waveform dumps are bounded by the announcement 08 exemption to the "
                         f"registered off-grid probe queries and the registered visualization "
                         f"cases; {outside[:3]} are in neither list")
    return True


def load_dump_cases(path):
    """A registered case list, carried with the digest of the file it came from."""
    with open(str(path)) as handle:
        payload = json.load(handle)
    ids = payload.get("query_ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(q, str) for q in ids):
        raise ValueError(f"{path} must carry a non-empty list of query_ids")
    return {"query_ids": list(ids), "sha256": file_sha256(path), "path": str(path)}


#: candidates kept in a bounded waveform dump, beyond the predictions themselves.
DUMP_TOP_N = 8
DUMP_CONTENT_RULE = (
    "a dumped query keeps the union of (a) every prefix's predicted candidate, (b) every "
    "prefix's S_mean-predicted candidate and (c) the DUMP_TOP_N best-scoring candidates at "
    "the largest prefix, plus the observation. exp_18 dumped every candidate because M was "
    "~10; here M averages 1,667, so the full dump would be 546 MB per query and 1.7 TB per "
    "pass. The selected rows are REGENERATED from their own noise keys after scoring, so a "
    "dump costs its own rows again and nothing is held in memory for it")


def dump_selection(scored, prefixes=None, top_n=DUMP_TOP_N):
    """The bounded, score-derived set of candidate rows a dump may carry."""
    from src.localization.reaggregate import decode_scores

    blocks = scored["by_k"]
    prefixes = tuple(blocks) if prefixes is None else tuple(prefixes)
    rows = set()
    for k in prefixes:
        block = blocks[k] if k in blocks else blocks[str(k)]
        rows.add(int(block["prediction_row"]))
        rows.add(int(block["mean_prediction_row"]))
    largest = max(prefixes, key=lambda k: int(k))
    block = blocks[largest] if largest in blocks else blocks[str(largest)]
    scores = decode_scores(block["scores_hex"])
    best = torch.topk(scores, min(int(top_n), scores.numel())).indices.tolist()
    rows.update(int(row) for row in best)
    return sorted(rows)


def write_query_waveforms(out_dir, row, candidate_indices, waveforms, observation):
    """One bounded waveform dump, published atomically beside the query's row."""
    paths = query_artifact_paths(out_dir, row["room_id"], row["position"])
    os.makedirs(paths["dir"], exist_ok=True)
    path = os.path.join(paths["dir"], f"q{int(row['position']):05d}_waveforms.npz")
    tmp = path + ".tmp"
    with open(tmp, "wb") as handle:
        np.savez(handle,
                 candidate_indices=np.asarray(candidate_indices, dtype=np.int64),
                 waveforms=np.asarray(torch.as_tensor(waveforms).detach().cpu().numpy(),
                                      dtype=np.float32),
                 observation=np.asarray(torch.as_tensor(observation).detach().cpu()
                                        .reshape(-1).numpy(), dtype=np.float32))
    os.replace(tmp, path)
    return {"waveform_path": os.path.relpath(path, str(out_dir)),
            "waveform_sha256": file_sha256(path),
            "waveform_candidate_indices": [int(i) for i in candidate_indices],
            "waveform_content_rule": DUMP_CONTENT_RULE}


def probe_record(query_id, room_id, n_candidates, num_samples, timings,
                 receiver_id=None, n_union=None):
    """One throughput-probe record: cost only, by construction.

    The probe exists to project GPU hours before the launch decision, so it may
    not read a score -- the record carries no similarity, no aggregate and no
    prediction, and :func:`assert_no_scores` is the gate that says so.
    """
    return {"query_id": str(query_id), "room_id": str(room_id),
            # the source cache is billed to the GROUP; without its identity the
            # per-query records cannot be summed without double counting it
            "receiver_id": None if receiver_id is None else str(receiver_id),
            "n_union": None if n_union is None else int(n_union),
            "n_candidates": int(n_candidates), "num_samples": int(num_samples),
            "n_generated": int(n_candidates) * int(num_samples),
            "timings_s": {str(k): float(v) for k, v in dict(timings).items()},
            "scores_written": False}


#: anything that would carry localization quality out of a no-quality probe.
_SCORE_KEYS = ("by_k", "scores_hex", "sims", "sims_path", "prediction_index",
               "prediction_xyz", "mean_scores_hex", "e_loc")


def assert_no_scores(record):
    """The no-quality rule, enforced on the record rather than promised."""
    found = sorted(key for key in _SCORE_KEYS if key in record)
    if found or record.get("scores_written"):
        raise ValueError(f"the throughput probe is no-quality by protocol but this record "
                         f"carries {found or ['scores_written']}; timing may be measured "
                         "before the launch decision, localization quality may not")
    return True


#: timings that scale with GENERATED WAVEFORMS.
_GENERATION_COMPONENTS = ("conditioning", "sampling", "decode", "embed", "scoring")


def project_cost(records, totals=None):
    """Project the registered pass from measured rates -- three separately.

    Generation scales with waveforms, the source branch with unique (receiver,
    candidate) rows, and the context branch with queries. Summing them into one
    per-query number would bill a receiver's cache once per query and a
    context's conditioning once per candidate, so each is measured against its
    own denominator.
    """
    totals = dict(REGISTERED_TOTALS if totals is None else totals)
    generation = sum(float(record["timings_s"].get(name, 0.0))
                     for record in records for name in _GENERATION_COMPONENTS)
    waveforms = sum(int(record["n_generated"]) for record in records)
    context = sum(float(record["timings_s"].get("context", 0.0)) for record in records)
    groups = {}
    for record in records:
        key = record.get("receiver_id")
        if key is None:
            continue
        groups[key] = (float(record["timings_s"].get("source_cache_group", 0.0)),
                       int(record.get("n_union") or 0))
    source_seconds = sum(seconds for seconds, _rows in groups.values())
    source_rows = sum(rows for _seconds, rows in groups.values())

    per_waveform = generation / waveforms if waveforms else 0.0
    per_source_row = source_seconds / source_rows if source_rows else 0.0
    per_context = context / len(records) if records else 0.0
    hours = {
        "generation": totals["generated_waveforms"] * per_waveform / 3600.0,
        "source_conditioning": totals["source_rows"] * per_source_row / 3600.0,
        "context": totals["queries"] * per_context / 3600.0,
    }
    hours["total"] = sum(hours.values())
    return {"n_records": len(records), "waveforms_measured": waveforms,
            "seconds_per_waveform": per_waveform,
            "source_rows_measured": source_rows,
            "seconds_per_source_row": per_source_row,
            "contexts_measured": len(records), "seconds_per_context": per_context,
            "projected_gpu_hours": hours, "totals": totals,
            "note": "generation scales with waveforms, the source branch with unique "
                    "(receiver, candidate) rows and the context branch with queries; the "
                    "source cache is counted ONCE per receiver group"}


def write_probe_records(out_dir, records, stem="probe", binding=None, binding_sha256=None,
                        advisory=None, totals=None):
    """Publish probe timings under a diagnostics stem, never as query artifacts."""
    for record in records:
        assert_no_scores(record)
    if binding is not None:
        assert_registered_noise_policy(binding.get("noise_policy"))
    os.makedirs(str(out_dir), exist_ok=True)
    path = os.path.join(str(out_dir), f"diagnostics_{stem}.json")
    write_json(path, {
        "experiment": "exp_22 loc_meshgrid I1 throughput probe",
        "scores_written": False, "n_queries": len(records),
        # immutable provenance: a cost decision has to be checkable against the
        # artifacts it was measured on, not against the shell history
        "binding": None if binding is None else {field: binding[field]
                                                 for field in RUN_BINDING_FIELDS
                                                 if field in binding},
        "binding_sha256": binding_sha256,
        "advisory": dict(advisory or {}),
        "noise_policy": None if binding is None else binding.get("noise_policy"),
        "projection": project_cost(records, totals=totals),
        "determinism_contract": DETERMINISM_CONTRACT,
        "agree_leakage_caveat": AGREE_LEAKAGE_CAVEAT,
        "scorer_readout_deviation": SCORER_READOUT_DEVIATION,
        "records": list(records)})
    return path


# --------------------------------------------------------------------------- #
# the pass
# --------------------------------------------------------------------------- #
def room_of_relpath(relpath):
    """``'<scene>/<scene_id>'`` from a split-relative IR path."""
    parts = str(relpath).replace(os.sep, "/").strip("/").split("/")
    if len(parts) < 3:
        raise ValueError(f"relpath must contain <scene>/<scene_id>/<file>: {relpath!r}")
    return f"{parts[-3]}/{parts[-2]}"


def _sync(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device if ":" in str(device) else None)


class _Timer:
    """Component timings with a leading and trailing device drain."""

    def __init__(self, device):
        self.device = device
        self.totals = {}

    def __call__(self, name):
        import contextlib
        import time

        @contextlib.contextmanager
        def _scope():
            _sync(self.device)
            started = time.perf_counter()
            try:
                yield
            finally:
                _sync(self.device)
                self.totals[name] = self.totals.get(name, 0.0) + (time.perf_counter() - started)
        return _scope()


@dataclass
class MeshEngine:
    """The generation + scoring stack, as callables.

    Keeping the stack behind one seam is what makes the whole per-room /
    per-receiver / per-query layout testable without a GPU: a run builds it from
    the frozen checkpoint (``build_mesh_engine``), the tests build it from
    recording fakes.
    """
    device: str
    latent_shape: tuple
    conditioner: object
    cond_inputs_fn: object
    sampler: object
    decoder: object
    embedder: object
    cond_method: str = "vanilla"


def assert_cacheable(cond_method):
    """The §1.5 split is only valid for vanilla conditioning.

    Under the released C4 frame average the cylindrical transform makes
    ``context_poses.dphi`` candidate-dependent, so a per-query context cache
    would serve the wrong tokens. §1.5 prescribes a narrower split for that arm;
    until it is implemented, an FA run is refused rather than mis-cached.
    """
    if str(cond_method) != "vanilla":
        raise ValueError(f"the per-query context cache is registered for vanilla conditioning "
                         f"only, but this engine runs cond_method={cond_method!r}; under "
                         "fa_invariant the released cylindrical transform makes context_poses "
                         "candidate-dependent (inherited plan §1.5) and only context_poses_vit "
                         "and context_audio may be cached")
    return True


def probe_groups(plan, budget, room_order=None, room=None):
    """Whole receiver groups, in pass order, until ``budget`` queries are covered.

    The probe measures cost, and the cost of a query is only meaningful once its
    receiver's cache is amortized over that receiver's whole group -- measuring
    three queries of a ten-query receiver would bill the entire union to three.

    ``room`` bounds the probe to one room. The registered subset's first room is
    Cafe, whose smallest receiver group is already ~9 queries x 5,295 candidates
    x 8 draws = 380 k waveforms, so without this there is no affordable smoke of
    the real stack at all.
    """
    budget = int(budget)
    chosen, covered = [], 0
    if room is not None:
        if room not in plan.rooms:
            raise ValueError(f"room {room!r} is not in the audit's {len(plan.rooms)} rooms")
        room_order = [room]
    for room_id in (room_order or sorted(plan.rooms)):
        for group in receiver_groups(load_room_plan(plan, room_id)):
            chosen.append((room_id, group.receiver_id))
            covered += len(group.queries)
            if covered >= budget:
                return chosen, covered
    return chosen, covered


def _score_one_query(engine, query, context, cache, obs_embedding, *, seed, num_samples,
                     noise_policy, batch_rows, timer):
    """Generate and score every candidate of one query -> sims ``[M, K]``."""
    from src.localization.scoring import cosine_sims

    indices = [int(i) for i in query.candidate_indices]
    rows_all = cache.rows_for(indices)
    per_chunk = max(1, int(batch_rows) // int(num_samples))
    parts = []
    for start in range(0, len(indices), per_chunk):
        slice_indices = indices[start:start + per_chunk]
        noise = noise_block(seed, query.query_id, slice_indices, num_samples,
                            engine.latent_shape, policy=noise_policy, device=engine.device)
        rows = rows_all[start:start + per_chunk].repeat_interleave(int(num_samples))
        with timer("conditioning"):
            merged = expand_conditioning(context, cache.conditioning, rows, engine.device)
            cond_inputs = engine.cond_inputs_fn(merged)
        with timer("sampling"):
            latents = engine.sampler(noise, cond_inputs)
        with timer("decode"):
            wavs = engine.decoder(latents).clamp(-1.0, 1.0)
        with timer("embed"):
            embeddings = engine.embedder(wavs)
        with timer("scoring"):
            parts.append(cosine_sims(
                obs_embedding,
                embeddings.reshape(len(slice_indices), int(num_samples), -1)).float().cpu())
    return torch.cat(parts, dim=0)


def _dump_query(engine, query, context, cache, out_dir, row, scored, *, seed, num_samples,
                noise_policy, top_n):
    """Regenerate the selected candidates and publish the bounded dump."""
    rows = dump_selection(scored, top_n=top_n)
    indices = [int(query.candidate_indices[row_index]) for row_index in rows]
    cache_rows = cache.rows_for(indices)
    noise = noise_block(seed, query.query_id, indices, num_samples, engine.latent_shape,
                        policy=noise_policy, device=engine.device)
    merged = expand_conditioning(context["context"], cache.conditioning,
                                 cache_rows.repeat_interleave(int(num_samples)),
                                 engine.device)
    wavs = engine.decoder(engine.sampler(noise, engine.cond_inputs_fn(merged))).clamp(-1.0, 1.0)
    wavs = wavs.reshape(len(indices), int(num_samples), -1)
    return write_query_waveforms(out_dir, row, indices, wavs, context["obs_wav"])


def replay_check(engine, query, md, obs_wav, *, seed=SEED, tau=TAU, num_samples=NUM_SAMPLES,
                 prefixes=K_PREFIXES, noise_policy=NOISE_KEY_POLICY, batch_rows=64,
                 source_chunk=SOURCE_CHUNK):
    """Score ONE query twice at identical batching and compare -- the registered
    fixed-batching determinism claim, executable (r7 review BLOCKER DETERMINISM).

    Everything is rebuilt on each pass, caches included, so a stack that drifts
    between two identical calls is caught rather than assumed away.
    """
    positions = np.asarray(query.coordinates, dtype=np.float64) \
        - np.asarray(query.receiver_xyz, dtype=np.float64)
    indices = [int(i) for i in query.candidate_indices]
    base_md = {"depth": md["depth"]}
    runs = []
    for _attempt in (0, 1):
        context = context_conditioning(engine.conditioner, md, engine.device)
        cache = ReceiverCache.build(engine.conditioner, query.receiver_id, base_md, indices,
                                    positions, engine.device, chunk=source_chunk)
        obs_embedding = engine.embedder(
            torch.as_tensor(obs_wav).to(engine.device))[0].float().cpu()
        sims = _score_one_query(engine, query, context, cache, obs_embedding, seed=seed,
                                num_samples=num_samples, noise_policy=noise_policy,
                                batch_rows=batch_rows, timer=_Timer(engine.device))
        scored = score_query(sims, indices, query.coordinates, tau=tau, prefixes=prefixes)
        runs.append((sims, scored))
    delta = float((runs[0][0] - runs[1][0]).abs().max())
    fingerprints = [{str(k): block for k, block in run[1]["by_k"].items()} for run in runs]
    return {"query_id": query.query_id, "n_candidates": len(indices),
            "batch_rows": int(batch_rows), "source_chunk": int(source_chunk),
            "max_abs_delta": delta,
            "fingerprint_equal": bool(fingerprints[0] == fingerprints[1]),
            "bit_exact": bool(torch.equal(runs[0][0], runs[1][0])
                              and fingerprints[0] == fingerprints[1]),
            "contract": DETERMINISM_CONTRACT}


def _build_row(query, scored, *, seed, noise_policy, prefixes, timings, n_contexts,
               batching=None):
    return {
        "query_id": query.query_id, "room_id": query.room_id, "position": query.position,
        "receiver_id": query.receiver_id,
        "receiver_xyz": [float(v) for v in query.receiver_xyz],
        "branch": query.branch, "z_band": query.z_band, "e_oracle": query.oracle,
        "n_candidates": scored["n_candidates"], "num_samples": scored["num_samples"],
        "tau": scored["tau"], "seed": int(seed), "noise_policy": str(noise_policy),
        "k_prefixes": [int(k) for k in prefixes],
        "candidate_indices": [int(i) for i in query.candidate_indices],
        "by_k": {str(k): block for k, block in scored["by_k"].items()},
        "scorer_readout": SCORER_READOUT,
        "scorer_readout_deviation": SCORER_READOUT_DEVIATION,
        "agree_leakage_caveat": AGREE_LEAKAGE_CAVEAT,
        "n_contexts": int(n_contexts), "timings_s": dict(timings),
        # which batching produced THIS query (advisory tier; see BATCHING_CAVEAT)
        "batching": dict(batching or {}),
    }


def run_pass(engine, stream, records, plan, out_dir, *, seed=SEED, tau=TAU,
             num_samples=NUM_SAMPLES, prefixes=K_PREFIXES, noise_policy=NOISE_KEY_POLICY,
             batch_rows=64, source_chunk=SOURCE_CHUNK, done=(), probe=None, on_row=None,
             excluded_room=None, dump_queries=(), dump_top_n=DUMP_TOP_N,
             probe_room=None, allow_unregistered_noise_policy=False,
             geometry_tol=CONTEXT_JOIN_TOLERANCE, binding_sha256=None):
    """Score the whole registered subset, room block by room block.

    The stream is the released loader in D1 order and is walked ONCE: every
    query's draw is verified against the frozen context manifest, its context
    branch is conditioned once, and its observation is embedded once. A room's
    queries are then run receiver group by receiver group, so the source branch
    costs one conditioner call per (receiver, candidate) in that receiver's
    union and exactly one group's cache is resident at a time.
    """
    from src.localization import meshgrid_queries as mq

    assert_cacheable(engine.cond_method)
    assert_registered_noise_policy(noise_policy, allow_unregistered_noise_policy)
    excluded_room = mq.EXCLUDED_ROOM if excluded_room is None else excluded_room
    done = set(done)
    by_position = {int(record["position"]): record for record in records}
    room_order = assert_room_blocks(records)

    selected = None
    if probe is not None:
        selected = set(probe_groups(plan, probe, room_order=room_order, room=probe_room)[0])

    dump_queries = set(dump_queries or ())
    state = {"n_scored": 0, "n_skipped": 0, "n_conditioner_rows": 0,
             "n_candidate_query_pairs": 0, "n_generated": 0, "n_dumped": 0,
             "n_geometry_tolerated": 0, "n_contexts_conditioned": 0,
             "rooms": [], "argmax_stability": {},
             "batching": {"batch_rows": int(batch_rows), "source_chunk": int(source_chunk)},
             "probe_records": [], "timings_s": {}}
    buffer, depths, receiver_of = {}, {}, {}
    room_plan, current_room = None, None
    timer = _Timer(engine.device)

    def flush():
        if room_plan is None:
            return
        _run_room(engine, room_plan, buffer, depths, out_dir, state, selected=selected,
                  done=done, seed=seed, tau=tau, num_samples=num_samples,
                  prefixes=prefixes, noise_policy=noise_policy, batch_rows=batch_rows,
                  source_chunk=source_chunk, on_row=on_row, probe=probe is not None,
                  geometry_tol=geometry_tol, dump_queries=dump_queries,
                  dump_top_n=dump_top_n, binding_sha256=binding_sha256)
        buffer.clear()
        depths.clear()
        receiver_of.clear()

    for position, (obs_wav, raw_md) in enumerate(stream):
        # from here on the target is unreadable, structurally (r7 review BLOCKER GT)
        md = GuardedMetadata(raw_md)
        record = by_position.get(position)
        if record is None:
            found = room_of_relpath(md.get("relpath") or md.get("path") or "")
            if found != excluded_room:
                raise ValueError(f"stream position {position} is in room {found!r}, which the "
                                 "registered subset neither excludes nor scores")
            continue
        verify_context_record(md, record, position)
        room_id = record["room_id"]
        if room_id != current_room:
            flush()
            current_room = room_id
            # a probe still STREAMS and verifies every earlier room -- the draws
            # depend on the whole pass -- but never parses a candidate manifest it
            # will not use; the largest is 137 MB of index lists
            if selected is not None and not any(room == room_id for room, _ in selected):
                room_plan, receiver_of = None, {}
            else:
                room_plan = load_room_plan(plan, room_id)
                receiver_of = {query.query_id: query.receiver_id
                               for query in room_plan.queries}
                state["rooms"].append(room_id)
        if room_plan is None:
            continue
        receiver_id = receiver_of.get(record["query_id"])
        if receiver_id is None:
            raise ValueError(f"{record['query_id']} is in the context manifest but not in "
                             f"{room_id}'s candidate manifest; the two registrations disagree")
        if selected is not None and (room_id, receiver_id) not in selected:
            continue
        if record["query_id"] in done:
            buffer[record["query_id"]] = None
            continue
        if obs_wav is None:
            raise ValueError(f"stream position {position}: the loader returned no observed "
                             "waveform; there is nothing to score against")
        query_context_timer = _Timer(engine.device)
        with query_context_timer("context"):
            context = context_conditioning(engine.conditioner, md, engine.device)
            obs_embedding = engine.embedder(
                torch.as_tensor(obs_wav).to(engine.device))[0].float().cpu()
        timer.totals["context"] = timer.totals.get("context", 0.0) + \
            query_context_timer.totals["context"]
        state["n_contexts_conditioned"] += 1
        # one panorama TENSOR per receiver, one digest per query: the tensor is
        # 1.5 MB and a room can hold 922 queries over ~93 receivers
        depths.setdefault(receiver_id, md["depth"])
        buffer[record["query_id"]] = {
            "context": context, "obs_embedding": obs_embedding,
            # only a dumped query keeps its observation waveform in memory
            "obs_wav": (torch.as_tensor(obs_wav).detach().cpu().clone()
                        if record["query_id"] in dump_queries else None),
            "depth_digest": tensor_digest(md["depth"]),
            "context_seconds": query_context_timer.totals["context"],
            # the context poses are what the GT-free geometry check re-derives from
            "context_poses": torch.as_tensor(md["context_poses"]).detach().cpu().clone(),
            "n_contexts": record["context_width"]}
    flush()
    for name, value in timer.totals.items():
        state["timings_s"][name] = state["timings_s"].get(name, 0.0) + value
    return state


def _run_room(engine, room_plan, buffer, depths, out_dir, state, *, selected, done, seed,
              tau, num_samples, prefixes, noise_policy, batch_rows, source_chunk, on_row,
              probe, geometry_tol, dump_queries=(), dump_top_n=DUMP_TOP_N,
              binding_sha256=None):
    """One room's receiver groups, one resident cache at a time."""
    missing = [] if selected is not None else [
        query.query_id for query in room_plan.queries
        if query.query_id not in buffer and query.query_id not in done]
    if missing:
        raise ValueError(f"{room_plan.room_id}: the stream did not deliver "
                         f"{len(missing)} of the room's registered queries "
                         f"(first {missing[:3]}); a partial room may not be published")

    for group in receiver_groups(room_plan):
        if selected is not None and (room_plan.room_id, group.receiver_id) not in selected:
            continue
        runnable = [query for query in group.queries
                    if buffer.get(query.query_id) is not None and query.query_id not in done]
        for query in group.queries:
            if query.query_id in done:
                # nothing will regenerate this one: prove the published row is ITS row
                assert_published_matches(out_dir, query, binding_sha256=binding_sha256)
                state["n_skipped"] += 1
        if not runnable:
            continue
        positions_cam = room_plan.base[np.asarray(group.union, dtype=np.int64)] \
            - np.asarray(group.receiver_xyz, dtype=np.float64)
        timer = _Timer(engine.device)
        with timer("source_cache"):
            cache = ReceiverCache.build(engine.conditioner, group.receiver_id,
                                        {"depth": depths[group.receiver_id]}, group.union,
                                        positions_cam, engine.device, chunk=source_chunk)
        state["n_conditioner_rows"] += cache.n_conditioner_rows
        try:
            for query in runnable:
                context = buffer[query.query_id]
                cache.assert_same_depth_digest(context["depth_digest"])
                geometry = assert_query_geometry_consistent(
                    {"context_poses": context["context_poses"]}, query, tol=geometry_tol)
                state["n_geometry_tolerated"] += geometry["n_tolerated"]
                query_timer = _Timer(engine.device)
                sims = _score_one_query(
                    engine, query, context["context"], cache, context["obs_embedding"],
                    seed=seed, num_samples=num_samples, noise_policy=noise_policy,
                    batch_rows=batch_rows, timer=query_timer)
                state["n_candidate_query_pairs"] += query.n_candidates
                state["n_generated"] += query.n_candidates * int(num_samples)
                for name, value in query_timer.totals.items():
                    state["timings_s"][name] = state["timings_s"].get(name, 0.0) + value
                if probe:
                    # the cache time is the GROUP's, billed once and reported under a
                    # name that says so: summing it per query would count it k times
                    state["probe_records"].append(probe_record(
                        query.query_id, query.room_id, query.n_candidates, num_samples,
                        dict(query_timer.totals,
                             source_cache_group=timer.totals["source_cache"],
                             context=context.get("context_seconds", 0.0),
                             group_size=len(runnable)),
                        receiver_id=group.receiver_id, n_union=len(group.union)))
                    continue
                scored = score_query(sims, query.candidate_indices.tolist(),
                                     query.coordinates, tau=tau, prefixes=prefixes)
                row = _build_row(query, scored, seed=seed, noise_policy=noise_policy,
                                 prefixes=prefixes, timings=query_timer.totals,
                                 n_contexts=context["n_contexts"],
                                 batching={"batch_rows": int(batch_rows),
                                           "source_chunk": int(source_chunk)})
                if query.query_id in dump_queries:
                    row.update(_dump_query(engine, query, context, cache, out_dir, row,
                                           scored, seed=seed, num_samples=num_samples,
                                           noise_policy=noise_policy, top_n=dump_top_n))
                    state["n_dumped"] += 1
                write_query_artifact(out_dir, row, sims, binding_sha256=binding_sha256)
                for key, block in row["by_k"].items():
                    entry = state["argmax_stability"].setdefault(
                        int(key), {"n_queries": 0, "n_unstable": 0,
                                   "min_margin": float("inf")})
                    entry["n_queries"] += 1
                    entry["n_unstable"] += int(not block["argmax_stable"])
                    entry["min_margin"] = min(entry["min_margin"], float(block["margin"]))
                state["n_scored"] += 1
                if on_row is not None:
                    on_row(row)
        finally:
            del cache


# --------------------------------------------------------------------------- #
# the real stack
# --------------------------------------------------------------------------- #
def assert_release_rng_state(manifest):
    """The global RNG must be exactly where the D1 pass created ITS iterator.

    Worker base seeds are drawn when the iterator is created, so anything that
    consumes the global stream between ``seed_everything`` and the first batch
    changes every query's context draw. D1 recorded the state digest at that
    moment; comparing it here turns a silent re-draw into a startup refusal
    instead of 5,337 digest mismatches.
    """
    from src.localization import meshgrid_queries as mq

    registered = ((manifest or {}).get("protocol_facts") or {}).get("rng_digest_at_iter")
    if not registered:
        raise ValueError("the context manifest records no rng_digest_at_iter; the released "
                         "call graph cannot be proven to have been reproduced")
    found = mq.rng_state_digest()
    if found != registered:
        raise ValueError(f"the global RNG state at iterator creation is {found[:12]}... but the "
                         f"D1 pass recorded {registered[:12]}...; something consumed the global "
                         "stream after seed_everything, so the worker base seeds -- and every "
                         "context draw -- would differ from the frozen manifest")
    return True


def build_mesh_engine(ckpt_path, model_config, agree, device="cpu", cond_method="vanilla",
                      cond_autocast="default", steps=STEPS, cfg_scale=CFG_SCALE, ckpt=None,
                      readout=SCORER_READOUT):
    """Build the frozen generator and wrap it as a :class:`MeshEngine`.

    Follows ``eval_FLAC.evaluate_model``'s lines of record through
    ``eval_localization`` -- matmul precision, ARE refusal, EMA remap,
    load-integrity check, wrapper construction, eval/no-grad, latent length from
    the pretransform ratio -- so exp_22 scores the same generative process the
    release evaluation runs. The only addition is the ``only_ids`` seam the two
    conditioning caches need.
    """
    import contextlib
    import copy

    from eval_FLAC import check_load_integrity, resolve_cond_autocast
    from eval_localization import assert_no_are, assert_rectified_flow, prepare_state_dict
    from src.inference.sampling import sample_discrete_euler
    from src.localization.agree_embed import embed_rirs
    from src.models.factory import create_model_from_config
    from src.training.factory import create_training_wrapper_from_config

    assert_cacheable(cond_method)
    torch.set_float32_matmul_precision("medium")
    model_config = copy.deepcopy(model_config)
    file_model_config = copy.deepcopy(model_config)
    assert_rectified_flow(model_config)

    training_config = model_config.get("training", None)
    if ckpt is None:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    assert_no_are(ckpt.get("model_config"), file_model_config)
    state_dict, weights_source = prepare_state_dict(ckpt, training_config)

    model_obj = create_model_from_config(model_config)
    missing, unexpected = model_obj.load_state_dict(state_dict, strict=False)
    check_load_integrity(missing, unexpected, False)

    model_config["training"] = training_config
    module = create_training_wrapper_from_config(model_config, model_obj)
    module.eval().requires_grad_(False)
    module.to(device)
    with torch.amp.autocast(device):
        model = module.diffusion.model

    if module.diffusion.pretransform is not None:
        latent_samples = model_config["sample_size"] // module.diffusion.pretransform.downsampling_ratio
    else:
        latent_samples = model_config["sample_size"]
    ac_enabled, ac_dtype = resolve_cond_autocast(cond_autocast)

    def cond_autocast_ctx():
        if not ac_enabled:
            return contextlib.nullcontext()
        if ac_dtype is None:
            return torch.amp.autocast(device)
        return torch.amp.autocast(device, dtype=ac_dtype)

    def conditioner(metadata, _device, only_ids=None):
        with cond_autocast_ctx():
            with torch.no_grad():
                return module.diffusion.conditioner(metadata, module.device, only_ids=only_ids)

    def sampler(noise, cond_inputs):
        with torch.no_grad():
            return sample_discrete_euler(model, noise, steps, **cond_inputs,
                                         cfg_scale=cfg_scale,
                                         dist_shift=module.diffusion.dist_shift,
                                         batch_cfg=True, disable_tqdm=True)

    def decoder(latents):
        with torch.no_grad():
            if module.diffusion.pretransform is not None:
                return module.diffusion.pretransform.decode(latents)
            return latents

    def embedder(wavs):
        if agree is None:
            raise ValueError("no AGREE scorer was loaded; embedding is unavailable")
        return embed_rirs(agree.model, wavs, device, readout=readout)

    engine = MeshEngine(device=device, latent_shape=(module.diffusion.io_channels, latent_samples),
                        conditioner=conditioner,
                        cond_inputs_fn=module.diffusion.get_conditioning_inputs,
                        sampler=sampler, decoder=decoder, embedder=embedder,
                        cond_method=cond_method)
    context = {"module": module, "model": model, "model_config": model_config,
               "weights_source": weights_source, "device": device,
               "latent_shape": engine.latent_shape}
    return engine, context

def _pair_diff(left, right):
    """``{equal, max_abs_diff}`` for two tensors compared exactly."""
    return {"equal": bool(torch.equal(left, right)),
            "max_abs_diff": float((left.float() - right.float()).abs().max())}


def cache_parity_check(engine, query, md, n_candidates=None, source_chunk=SOURCE_CHUNK):
    """§1.5's bit-identity proof, with the two questions kept apart.

    **memoization** -- the contract. Both sides are computed at the SAME
    batching (one candidate per call, cache chunk 1), so any difference means
    the cache serves something other than what the direct call computes. This
    must be exact.

    **batched** -- informational. One uncached call over all candidates against
    the cache at its production chunk. Under autocast a backbone's GEMM tiling
    changes with the batch shape, so a nonzero difference here is the model's
    own batch nondeterminism -- present in any batched inference and not
    introduced by the cache. It is reported, never asserted.

    **counter_test** -- the comparison must be capable of failing: the same
    comparison against a cache built from perturbed positions has to differ,
    otherwise a vacuous check would read as a pass.
    """
    from src.localization.candidates import candidate_metadata

    # the source branch needs depth and a candidate pose -- never the target
    md = md.without_target() if isinstance(md, GuardedMetadata) else md
    # default to MORE candidates than one production chunk, so the batched half
    # actually spans a chunk boundary instead of collapsing to a single call
    if n_candidates is None:
        n_candidates = 2 * int(source_chunk)
    indices = [int(i) for i in query.candidate_indices][:max(1, int(n_candidates))]
    positions = query.coordinates[:len(indices)] - np.asarray(query.receiver_xyz,
                                                              dtype=np.float64)
    context = context_conditioning(engine.conditioner, md, engine.device)

    # (1) matched batching: one candidate per call on BOTH sides
    per_candidate = [engine.conditioner([candidate_metadata(md, positions[row])],
                                        engine.device) for row in range(len(indices))]
    unit_cache = ReceiverCache.build(engine.conditioner, query.receiver_id, md, indices,
                                     positions, engine.device, chunk=1)
    unit = expand_conditioning(context, unit_cache.conditioning,
                               unit_cache.rows_for(indices), engine.device)
    memo = {}
    for key in CONTEXT_COND_IDS + SOURCE_COND_IDS:
        direct = torch.cat([per_candidate[row][key][0] for row in range(len(indices))], dim=0)
        memo[key] = _pair_diff(unit[key][0], direct)

    # (2) production batching, against one uncached call over all candidates
    whole = engine.conditioner([candidate_metadata(md, positions[row])
                                for row in range(len(indices))], engine.device)
    cache = ReceiverCache.build(engine.conditioner, query.receiver_id, md, indices,
                                positions, engine.device, chunk=source_chunk)
    batched_side = expand_conditioning(context, cache.conditioning,
                                       cache.rows_for(indices), engine.device)
    batched = {key: _pair_diff(batched_side[key][0], whole[key][0])
               for key in CONTEXT_COND_IDS + SOURCE_COND_IDS}

    # (3) the comparison must bite
    moved_cache = ReceiverCache.build(engine.conditioner, query.receiver_id, md, indices,
                                      positions + 1.0, engine.device, chunk=1)
    moved = expand_conditioning(context, moved_cache.conditioning,
                                moved_cache.rows_for(indices), engine.device)
    counter = max(_pair_diff(moved[key][0], unit[key][0])["max_abs_diff"]
                  for key in SOURCE_COND_IDS)

    memo_ok = all(entry["equal"] for entry in memo.values())
    return {"query_id": query.query_id, "n_candidates": len(indices),
            "source_chunk": int(source_chunk),
            "memoization": {"match": memo_ok, "keys": memo},
            "batched": {"keys": batched,
                        "max_abs_diff": max(entry["max_abs_diff"]
                                            for entry in batched.values()),
                        "note": "informational: a nonzero value is the backbone's own "
                                "batch-shape nondeterminism under autocast, not a cache "
                                "error"},
            "counter_test": {"detected": bool(counter > 0.0), "max_abs_diff": float(counter)},
            "dtypes": {key: str(unit[key][0].dtype)
                       for key in CONTEXT_COND_IDS + SOURCE_COND_IDS},
            "match": bool(memo_ok and counter > 0.0)}
