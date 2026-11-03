"""exp_22 I1 -- the frozen mesh-grid localization engine (inherited plan §1.4/§1.5).

One query is an observed RIR ``h_obs``; the engine regenerates one RIR per
mesh-valid grid candidate under the frozen Vanilla FLAC checkpoint, scores each
against ``h_obs`` in AGREE's audio embedding space, and predicts the argmax
candidate. Everything that decides a number -- the noise a candidate is drawn
from, the context tensors it is conditioned on, the candidate set it competes in
-- is bound to a registered artifact (the D1 context manifest, the G1 candidate
manifests, the checkpoint, the scorer) and refused when it does not match.

The engine is also, by construction, unable to read the thing it localizes: the
loader item is wrapped in :class:`GuardedMetadata`, so ``md['source']`` raises
rather than returning the answer, and every geometry check is derived from the
manifest receiver and the context poses instead
(:func:`assert_query_geometry_consistent`).

Four points are RECORDED DEVIATIONS from the inherited plan text, stamped into
the run binding and the published rows rather than silently taken:

* ``SCORER_READOUT`` -- §1.4 pins ``encode_audio(..., normalize=True)``. That
  path samples AGREE's VAE bottleneck, which exp_18 measured as ~7e-5 of cosine
  noise per call and which consumes the global RNG stream. The registered
  scorer here is exp_18/exp_20's deterministic mean readout, which is the same
  arithmetic with the bottleneck's mean substituted for its sample.
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

#: A per-score bound eps moves a GAP by up to 2 eps -- the leader can lose eps
#: while the runner-up gains it -- so an argmax is only safe when the top-1
#: margin exceeds TWICE the tolerance. r8 used eps and called margins in
#: (eps, 2 eps] stable, which the r8 review rejected.
ARGMAX_STABILITY_FACTOR = 2


def argmax_stability_bound(tolerance=None):
    """The margin an argmax must exceed to be safe under a per-score bound."""
    return ARGMAX_STABILITY_FACTOR * float(SCORE_TOLERANCE if tolerance is None
                                           else tolerance)


def is_argmax_stable(margin, tolerance=None):
    """Whether a top-1 margin survives the worst case of the per-score bound."""
    return bool(float(margin) > argmax_stability_bound(tolerance))


DETERMINISM_CONTRACT = (
    "At fixed batching every stage is deterministic and a replay is bit-exact through "
    "scoring: the noise is keyed, the K prefixes are slices of one sequence, the caches are "
    "a proven-exact memoization (cache_parity_check), the AGREE readout is the deterministic "
    "VAE mean, and nothing in the pass consumes the global RNG. Two passes over the same "
    "artifacts at the same batch_rows/source_chunk must therefore produce identical score "
    "fingerprints. Changed batching is a different question: the backbones' GEMM tiling "
    "moves an output by about one float16 ulp, so the passes are compared against "
    "SCORE_TOLERANCE, and every query whose top-1 margin is within TWICE that bound -- the "
    "worst case in which the leader loses eps while the runner-up gains it, i.e. every "
    "argmax that COULD flip -- is counted and named, never silently accepted")

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
            "argmax_stable": is_argmax_stable(margin),
            "stability_bound": argmax_stability_bound(),
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
                        ids=SOURCE_COND_IDS, fa=None):
    """The source branch over ``[U, 3]`` camera-frame candidate positions.

    Chunked so a receiver's whole union never has to fit in one forward, and
    concatenated in the union's own order -- ``chunk`` therefore changes the
    batching and nothing else, which the chunk-invariance test pins. ``fa`` (an
    engaged :class:`FaSplit`, exp_22 R2-B) runs each chunk through the released
    frame-average path instead: ``source`` cylindrical, ``source_vit`` C4-averaged.
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
        parts.append(_branch(conditioner, batch, device, ids) if fa is None else
                     fa.branch(conditioner, "source_cache", batch, device, configured=chunk))
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
              chunk=SOURCE_CHUNK, fa=None):
        indices = [int(i) for i in indices]
        if len(set(indices)) != len(indices):
            raise ValueError(f"receiver {receiver_id!r}: the union repeats a candidate index")
        positions = np.asarray(positions_cam, dtype=np.float64)
        if positions.shape[0] != len(indices):
            raise ValueError(f"receiver {receiver_id!r}: {positions.shape[0]} positions for "
                             f"{len(indices)} candidates")
        conditioning = source_conditioning(conditioner, base_md, positions, device, chunk=chunk,
                                           fa=fa)
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
# exp_22 R2-B: the five-way frame-average split (plan r2 §2 table; Codex B5, rA A8)
# --------------------------------------------------------------------------- #
#: the split's registered name -- the only one under which fa_invariant is cached
FA_SPLIT = "fa_five_way"

#: The dependency table of plan r2 §2 as the three batches the split runs. Each
#: goes through the RELEASED ``invariant_conditioning`` with the conditioner
#: restricted to the batch's ids, so the cylindrical conversion, the ViT-only
#: rotated passes, the accumulation order, the divisor and the base-pass masks
#: are the released ones; only WHICH ids a call computes and ON WHICH batch differ.
#: ``context_vit`` -- one query row: ``context_audio`` from the base pass only
#: (never averaged), ``context_poses_vit`` C4-averaged. ``context_scored_batch``
#: -- ``context_poses``, cylindrical against EACH candidate's reference azimuth
#: (candidate-dependent, angle-invariant, never averaged). ``source_cache`` --
#: ``source`` cylindrical base, ``source_vit`` C4-averaged, per receiver/candidate.
FA_BRANCH_IDS = {"context_vit": ("context_poses_vit", "context_audio"),
                 "context_scored_batch": ("context_poses",),
                 "source_cache": SOURCE_COND_IDS}
FA_BRANCHES = tuple(FA_BRANCH_IDS)
#: the batches that carry a frame-averaged (ViT) id, i.e. that make rotated passes
FA_ORBIT_BRANCHES = ("context_vit", "source_cache")
#: what the per-query context batch reads -- the held-out target is never one
FA_QUERY_KEYS = ("depth", "context_poses_vit", "context_audio")

FA_EXECUTION_NOTE = (
    "EXECUTED PARTITION (Codex rA A8, rB finding 5; announcement 06): derived at run time from "
    "the calls the five-way split actually made -- per batch, the LARGEST base batch observed "
    "(the micro-batch), the largest rotated forward (the cap), the passes counted into the "
    "orbit size, every rotated forward -- never from the command line or the loop's declared "
    "size, which is kept only as a consistency check (one stride per batch, no call above it). "
    "Before every row a prefix whose batches have not yet filled a declared micro-batch is "
    "consistent; the pass's final check, the merge and gate 3 need every declared micro-batch "
    "observed at full size. A pass whose executed partition differs from the declared "
    "run_binding.json fa_partition stops; the rows it already wrote stay on disk as "
    "INCOMPLETE, NON-ADMISSIBLE artifacts -- the shard writes partition_check_failed.json and "
    "every merge, report and reopen refuses it (plan r2 amendment A6 iv)")
#: the micro-batch fields a tail-only prefix may leave below their declared value
FA_BATCH_FIELDS = ("candidate_micro_batch", "frame_avg_fwd_cap")


class _BranchConditioner:
    """The conditioner restricted to ONE split batch's ids, recording every call it serves.

    ``invariant_conditioning`` calls it for a base pass (no ``only_ids``: here,
    exactly the batch's ids) and for rotated passes (the present ViT ids). Each
    call goes through :func:`_branch`, so an answer short or long is refused.
    """

    def __init__(self, conditioner, ids):
        self.conditioner, self.ids, self.calls = conditioner, tuple(ids), []

    def __call__(self, metadata, device, only_ids=None):
        wanted = self.ids if only_ids is None else tuple(only_ids)
        outside = [key for key in wanted if key not in self.ids]
        if outside:
            raise ValueError(f"the {list(self.ids)} batch was asked for {outside}; a split batch "
                             "computes its own ids only")
        self.calls.append((len(metadata), only_ids is not None))
        return _branch(self.conditioner, metadata, device, wanted)


def fa_branch(conditioner, branch, metadata, device, angles, recorder=None, configured=None):
    """One split batch through the released frame-average path, restricted to its ids.

    The cap is the call's own batch (exp_20's ``candidate_micro_batch`` policy), so
    every rotated angle is its own forward; ``configured`` is the micro-batch the
    caller's loop runs at, recorded beside what was observed.
    """
    from src.data.yaw_rotation import invariant_conditioning

    metadata = list(metadata)
    restricted = _BranchConditioner(conditioner, FA_BRANCH_IDS[branch])
    out = invariant_conditioning(restricted, metadata, device,
                                 tuple(float(angle) for angle in angles),
                                 max_fwd_samples=len(metadata))
    if recorder is not None:
        recorder.observe(branch, restricted.calls, cap=len(metadata),
                         configured=len(metadata) if configured is None else configured,
                         orbit_size=len(tuple(angles)))
    return {key: out[key] for key in FA_BRANCH_IDS[branch]}


@dataclass
class FaSplit:
    """The engaged five-way split: the orbit it averages over and what it executed."""
    angles: tuple
    recorder: object = None
    name: str = FA_SPLIT

    def branch(self, conditioner, branch, metadata, device, configured=None, record=True):
        return fa_branch(conditioner, branch, metadata, device, self.angles,
                         self.recorder if record else None, configured)


def fa_query_conditioning(split, conditioner, md, device):
    """The per-query fa context, ONE row: ``context_audio`` once, ``context_poses_vit`` averaged."""
    row = {key: md[key] for key in FA_QUERY_KEYS}
    return split.branch(conditioner, "context_vit", [row], device, configured=1)


def fa_candidate_conditioning(split, conditioner, context_poses, positions_cam, device,
                              configured=None, record=True):
    """``context_poses`` for one scored candidate batch, cylindrical against EACH candidate."""
    from src.localization.candidates import candidate_metadata

    base = {"context_poses": context_poses}
    batch = [candidate_metadata(base, position)
             for position in np.asarray(positions_cam, dtype=np.float64).reshape(-1, 3)]
    return split.branch(conditioner, "context_scored_batch", batch, device,
                        configured=configured, record=record)


def _one(values):
    """The single value observed, else the sorted list (never equal to a declared int)."""
    values = sorted(set(values))
    return values[0] if len(values) == 1 else (values or None)


class FaPartitionRecorder:
    """What the split EXECUTED, read off its conditioner calls (Codex rA A8).

    Counted, not logged: a full pass makes ~350 k batch invocations, so each
    distinct ``(configured, cap, base batches, orbit batches, orbit size)`` is
    one counter. :meth:`executed` derives ``declared_fa_partition``'s schema.
    """

    def __init__(self):
        self.counts = {branch: {} for branch in FA_BRANCHES}

    def observe(self, branch, calls, *, cap, configured, orbit_size):
        key = (int(configured), int(cap), tuple(n for n, rotated in calls if not rotated),
               tuple(n for n, rotated in calls if rotated), int(orbit_size))
        self.counts[branch][key] = self.counts[branch].get(key, 0) + 1

    def violations(self):
        found = set()
        for branch, bucket in self.counts.items():
            strides = sorted({key[0] for key in bucket})
            if len(strides) > 1:
                found.add(f"{branch}: ran at more than one configured micro-batch {strides}")
            for configured, cap, base, orbit, _size in bucket:
                if orbit and branch not in FA_ORBIT_BRANCHES:
                    found.add(f"{branch}: made rotated forwards {list(orbit)}, but its ids are "
                              "never orbit-averaged")
                if len(base) != 1:
                    found.add(f"{branch}: an invocation made {len(base)} base passes")
                    continue
                if base[0] > configured:
                    found.add(f"{branch}: a batch of {base[0]} exceeds its micro-batch "
                              f"{configured}")
                if cap != base[0]:
                    found.add(f"{branch}: cap {cap} is not the call's batch {base[0]} "
                              "(cap_policy candidate_micro_batch)")
                if any(rows % base[0] for rows in orbit):
                    found.add(f"{branch}: an orbit forward of {list(orbit)} rows splits an angle")
        return sorted(found)

    def observed(self, branch):
        """``(largest base batch, largest rotated forward)`` the branch's calls made."""
        bucket = self.counts[branch]
        base = [rows for key in bucket for rows in key[2]]
        orbit = [rows for key in bucket for rows in key[3]]
        return (max(base) if base else None), (max(orbit) if orbit else None)

    def executed(self):
        """Declared_fa_partition's schema, every value read off the calls (Codex rB 5)."""
        keys = {branch: list(bucket) for branch, bucket in self.counts.items()}
        every = [key for branch in FA_BRANCHES for key in keys[branch]]
        orbit = [key for branch in FA_ORBIT_BRANCHES for key in keys[branch]
                 if len(key[2]) == 1]
        per_chunk = _one(rows // key[2][0] for key in orbit for rows in key[3])
        caps_ok = bool(every) and all(len(key[2]) == 1 and key[1] == key[2][0] for key in every)
        seen = {branch: self.observed(branch) for branch in FA_BRANCHES}
        return {"frame_avg_chunk_plan": "per_angle" if per_chunk == 1 else "batched",
                "cap_policy": "candidate_micro_batch" if caps_ok else "violated",
                "orbit_size": _one(1 + sum(key[3]) // key[2][0] for key in orbit),
                "angles_per_chunk": per_chunk,
                "n_orbit_forwards": _one(len(key[3]) for key in orbit),
                "shared_angle_count": per_chunk,
                "context_vit": {"candidate_micro_batch": seen["context_vit"][0],
                                "frame_avg_fwd_cap": seen["context_vit"][1]},
                "context_scored_batch": {
                    "candidate_micro_batch": seen["context_scored_batch"][0]},
                "source_cache": {"candidate_micro_batch": seen["source_cache"][0],
                                 "frame_avg_fwd_cap": seen["source_cache"][1]}}

    def to_counts(self):
        """The counts as JSON rows, one per distinct call shape (the durable ledger's unit)."""
        return [{"branch": branch, "configured": key[0], "cap": key[1], "base": list(key[2]),
                 "orbit": list(key[3]), "orbit_size": key[4], "count": int(count)}
                for branch in FA_BRANCHES for key, count in sorted(self.counts[branch].items())]

    @classmethod
    def from_counts(cls, *count_lists):
        """A recorder holding the SUM of ``count_lists`` (segments, or shards)."""
        recorder = cls()
        for entries in count_lists:
            for entry in entries:
                if entry.get("branch") not in recorder.counts:
                    raise ValueError(f"a ledger entry names branch {entry.get('branch')!r}, not "
                                     f"one of {list(FA_BRANCHES)}")
                key = (int(entry["configured"]), int(entry["cap"]),
                       tuple(int(n) for n in entry["base"]),
                       tuple(int(n) for n in entry["orbit"]), int(entry["orbit_size"]))
                bucket = recorder.counts[entry["branch"]]
                bucket[key] = bucket.get(key, 0) + int(entry["count"])
        return recorder

    def report(self):
        """The executed block, its violations and the per-batch evidence it rests on."""
        branches = {}
        for branch, bucket in self.counts.items():
            base, orbit = {}, {}
            for (_configured, _cap, bases, orbits, _size), count in bucket.items():
                for rows in bases:
                    base[rows] = base.get(rows, 0) + count
                for rows in orbits:
                    orbit[rows] = orbit.get(rows, 0) + count
            largest, _ = self.observed(branch)
            branches[branch] = {
                "invocations": int(sum(bucket.values())),
                "configured": sorted({key[0] for key in bucket}),
                "base_batch_histogram": {str(k): v for k, v in sorted(base.items())},
                "orbit_forward_histogram": {str(k): v for k, v in sorted(orbit.items())},
                "observed_max_base_batch": largest,
                # a full micro-batch was observed: the loop's stride is evidenced by a call
                "saturated": bool(bucket) and largest == max(key[0] for key in bucket),
                "full_micro_batch_observed": any(len(key[2]) == 1 and key[2][0] == key[0]
                                                 for key in bucket)}
        return {"executed": self.executed(), "violations": self.violations(),
                "branches": branches, "note": FA_EXECUTION_NOTE}


def _union(left, right):
    """Keys of both sides, ``left``'s order first: a key on one side only is a difference."""
    return list(left) + [key for key in right if key not in left]


def fa_partition_differences(declared, executed):
    """The dotted paths at which an executed partition is not the declared one, over the
    EXACT schema both ways: a key missing or extra on either side is a difference."""
    declared, executed, paths = declared or {}, executed or {}, []
    for field in _union(declared, executed):
        wanted, found = declared.get(field), executed.get(field)
        if field not in declared or field not in executed:
            paths.append(field)
        elif isinstance(wanted, dict) and isinstance(found, dict):
            paths += [f"{field}.{key}" for key in _union(wanted, found)
                      if key not in wanted or key not in found or found[key] != wanted[key]]
        elif isinstance(wanted, dict) or isinstance(found, dict) or found != wanted:
            paths.append(field)
    return paths


def _tail_prefix(path, declared, report):
    """A micro-batch not yet observed at full size: the loop runs at the DECLARED stride (its
    one configured value) and every call so far was smaller -- tails only, not a re-chunking."""
    branch, _, field = path.partition(".")
    wanted = (declared.get(branch) or {}).get(field)
    found = (report["executed"].get(branch) or {}).get(field)
    stride = (report["branches"].get(branch) or {}).get("configured")
    return (field in FA_BATCH_FIELDS and isinstance(wanted, int) and isinstance(found, int)
            and 0 < found < wanted and stride == [wanted])


def assert_fa_partition_executed(declared, recorder, final=True):
    """The split ran the DECLARED partition, or nothing is published (Codex rA A8, rB 5).

    ``final=False`` is the check before every row: a micro-batch only ever observed
    BELOW its declared size is a tail-only prefix (a room whose queries are all
    smaller than the batch), consistent so far. The final check is exact.
    """
    if not isinstance(declared, dict) or not declared:
        raise ValueError("a frame-averaged pass must declare its announcement-06 partition "
                         "(run_binding.json fa_partition) before anything runs against it")
    report = recorder.report()
    differing = [path for path in fa_partition_differences(declared, report["executed"])
                 if final or not _tail_prefix(path, declared, report)]
    if differing or report["violations"]:
        raise ValueError(f"the executed frame-average partition is not the declared one: "
                         f"{differing} differ (executed {report['executed']}); violations "
                         f"{report['violations'][:3]}. A scored pass that fails this check is "
                         f"NON-ADMISSIBLE ({FA_FAILURE_FILENAME}; plan r2 amendment A6 iv)")
    return report


FA_FAILURE_FILENAME = "partition_check_failed.json"
FA_FAILURE_NOTE = (
    "NON-ADMISSIBLE SHARD (plan r2 amendment A6 (iv); Codex rB-F1 NIT 5): an executed-partition "
    "check of this scored pass failed, so the frame-average partition it ran is not the one its "
    "run_binding.json declares. The rows it wrote are INCOMPLETE ARTIFACTS: every merge, "
    "per-arm report, crossarm admission and reopen refuses this directory by name. Fix the "
    "cause, then RESTART the shard in a fresh out-dir with fresh launch records -- never resume")


def write_partition_failure(out_dir, declared, recorder, check, error, n_segments=None,
                            launch_record_sha256=None):
    """Mark ``out_dir`` NON-ADMISSIBLE (A6 iv) with what failed: which check, the declared and
    executed blocks, the differing paths, the violations and the judged call counts."""
    from datetime import datetime, timezone

    report = recorder.report()
    os.makedirs(str(out_dir), exist_ok=True)           # a pass refused before its first row
    write_json(os.path.join(str(out_dir), FA_FAILURE_FILENAME), {
        "note": FA_FAILURE_NOTE, "check": str(check), "error": str(error), "declared": declared,
        "executed": report["executed"], "violations": report["violations"],
        "differences": fa_partition_differences(declared, report["executed"]),
        "branches": report["branches"], "counts": recorder.to_counts(),
        "counts_sha256": counts_sha256(recorder),
        "ledger": {"file": FA_LEDGER_FILENAME, "n_segments": n_segments},
        "launch_record_sha256": launch_record_sha256,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})


def assert_admissible(run_dir):
    """A6 (iv): a shard whose executed-partition check failed is refused by name -- by the
    merge, the per-arm report and every reopen (it is restarted, never resumed)."""
    if os.path.exists(os.path.join(str(run_dir), FA_FAILURE_FILENAME)):
        raise ValueError(f"{run_dir} is NON-ADMISSIBLE (plan r2 amendment A6 (iv)): an "
                         f"executed-partition check of its pass failed ({FA_FAILURE_FILENAME}); "
                         "its rows are incomplete artifacts that no merge, report or resume "
                         "admits -- restart the shard in a fresh out-dir once the cause is fixed")
    return True


FA_LEDGER_FILENAME = "fa_partition_ledger.json"
FA_LEDGER_NOTE = (
    "EXECUTED-PARTITION LEDGER (Codex rB finding 5): the split's call counts, one segment per "
    "invocation of a scored pass, rewritten atomically after each query's partition check and "
    "BEFORE its row is written -- a crash leaves every published row's calls on disk, and a "
    "resume adds a segment instead of replacing the evidence. The run's executed partition is "
    "the sum of its segments' counts; the merge re-derives each shard's, and the sum over all "
    "shards, from these counts and refuses any disagreement with the declaration or with the "
    "shard's own run summary. Each segment also keeps delta_counts (the calls since its state "
    "at n_queries - 1), and every published row has a stamp (fa_ledger_stamps/) of the state "
    "persisted just before it: each segment's state is its last row's, or exactly one query "
    "ahead of it after a crash (Codex rB-F2 finding 3)")


def counts_sha256(recorder):
    from src.localization.crossarm import canonical_sha256

    return canonical_sha256(recorder.to_counts())


RUN_SUMMARY_FILENAME = "run_summary.json"
MERGE_REPORT_FILENAME = "merge_report.json"
#: one row of a ledger segment's counts (``FaPartitionRecorder.to_counts``)
LEDGER_ENTRY_FIELDS = ("branch", "configured", "cap", "base", "orbit", "orbit_size", "count")


def _count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def counts_reasons(counts, what):
    """``(reasons, recorder or None)``: ``counts`` are a recorder's canonical call-count rows --
    exactly its seven fields, real non-negative integers (bools refused) with a count of at
    least one, a known branch, sorted, once each. THE validator the ledger's segments and the
    gate evidence's executed counts share (Codex rB-F2 finding 4)."""
    if not (isinstance(counts, list) and all(
            isinstance(entry, dict) and sorted(entry) == sorted(LEDGER_ENTRY_FIELDS)
            and isinstance(entry["base"], list) and isinstance(entry["orbit"], list)
            for entry in counts)):
        return [f"{what}'s counts are not {list(LEDGER_ENTRY_FIELDS)} rows"], None
    if not all(_count(value) for entry in counts for value in (
            entry["configured"], entry["cap"], entry["orbit_size"], entry["count"],
            *entry["base"], *entry["orbit"])):
        return [f"{what} holds a count that is not a non-negative integer"], None
    if not all(entry["count"] >= 1 for entry in counts):
        return [f"{what} holds a zero count (a call shape no recorder observed)"], None
    try:
        recorder = FaPartitionRecorder.from_counts(counts)
    except ValueError as error:                        # an unknown branch
        return [f"{what}: {error}"], None
    return ([] if recorder.to_counts() == counts else [
        f"{what}'s counts are not canonical (repeated or unsorted rows)"]), recorder


def _segment_reasons(index, segment):
    """``(reasons, recorder)`` for one segment: a non-negative query count, canonical counts
    (:func:`counts_reasons`) and the digest it stores of them, and a canonical
    ``delta_counts`` INSIDE them (the calls since its state at n_queries - 1)."""
    reasons = [] if _count(segment.get("n_queries")) else [
        f"segment {index}'s n_queries {segment.get('n_queries')!r} is not a non-negative integer"]
    why, recorder = counts_reasons(segment.get("counts"), f"segment {index}")
    if recorder is None:
        return reasons + why, None
    if segment.get("counts_sha256") != counts_sha256(recorder):
        why.append(f"segment {index}'s counts do not hash to its counts_sha256")
    delta, _delta = counts_reasons(segment.get("delta_counts"), f"segment {index}'s delta")
    if not delta and _counts_minus(segment["counts"], segment["delta_counts"]) is None:
        delta = [f"segment {index}'s delta_counts are not inside its counts"]
    return reasons + why + delta, recorder


def _counts_minus(counts, delta):
    """``counts`` minus ``delta`` (canonical rows) as a recorder, or ``None`` when ``delta`` is
    not inside ``counts``."""
    left = FaPartitionRecorder.from_counts(counts)
    for entry in delta:
        bucket = left.counts[entry["branch"]]
        key = (entry["configured"], entry["cap"], tuple(entry["base"]), tuple(entry["orbit"]),
               entry["orbit_size"])
        rest = bucket.get(key, 0) - entry["count"]
        if rest < 0:
            return None
        if rest:
            bucket[key] = rest
        else:
            bucket.pop(key, None)
    return left


#: the ledger state an fa row is published under (Codex rB-F2 finding 3), kept in a stamp file
#: BESIDE the rows -- ``<out-dir>/fa_ledger_stamps/<room>/q<position>.json`` -- so a BF row's
#: bytes keep the vanilla row schema the crossarm pairs arms on (its row whitelist is fixed)
STAMP_FIELDS = ("segment_index", "n_queries", "counts_sha256")
FA_STAMPS_DIRNAME = "fa_ledger_stamps"


def fa_ledger_state(ledger):
    """The state of this invocation's segment, persisted just before a row: the row's stamp."""
    segment = ledger["payload"]["segments"][-1]
    return {field: segment[field] for field in STAMP_FIELDS}


def fa_stamp_path(out_dir, room_id, position):
    return os.path.join(str(out_dir), FA_STAMPS_DIRNAME, room_stem(room_id),
                        f"q{int(position):05d}.json")


def write_fa_stamp(out_dir, query, ledger):
    """Publish the stamp of ``query``'s row, after the ledger write and BEFORE the row."""
    path = fa_stamp_path(out_dir, query.room_id, query.position)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return write_json(path, dict(fa_ledger_state(ledger), query_id=query.query_id))


def read_fa_stamp(out_dir, row):
    """``(query id, stamp or None, launch record)`` of one published row (rB-F2 finding 3)."""
    path = fa_stamp_path(out_dir, row["room_id"], row["position"])
    return (row["query_id"], _read_object(path) if os.path.isfile(path) else None,
            row.get("launch_record_sha256"))


def _shapes(counts):
    return {(e["branch"], e["configured"], e["cap"], tuple(e["base"]), tuple(e["orbit"]),
             e["orbit_size"]) for e in counts}


def _row_binding_reasons(segments, stamps, exact, blocks=None):
    """Every segment's stored state against its OWN rows' stamps (Codex rB-F2 finding 3).
    ``stamps`` = ``(query id, stamp, row launch record)`` of every self-verifying row;
    ``blocks`` = a merged ledger's shard blocks (rows then map by launch record and shard-local
    index). With N the segment's n_queries and M its rows' highest: N == M -> its counts are
    the state its last row was stamped with (empty when M is 0); N == M + 1 -> ONE query
    ahead (the ledger is written before the row) -> its counts minus its delta are; anything
    else is refused. At a merge a one-ahead state is admitted only off its shard's last
    invocation, and only if every call shape of the lost attempt was executed again later."""
    merged_mode, blocks = blocks is not None, blocks or [(None, list(segments))]
    if merged_mode:
        lookup = {(s.get("launch_record_sha256"), s["shard_segment_index"]): s
                  for s in segments if s.get("launch_record_sha256") is not None}
    else:
        lookup = {s["segment_index"]: s for s in segments}
    reasons, top, seen = [], {}, set()
    for query_id, stamp, launch in stamps:
        if not (isinstance(stamp, dict) and sorted(stamp) == sorted(STAMP_FIELDS + ("query_id",))
                and stamp["query_id"] == query_id and _count(stamp["segment_index"])
                and _count(stamp["n_queries"]) and stamp["n_queries"] >= 1
                and isinstance(stamp["counts_sha256"], str)):
            reasons.append(f"row {query_id}: its ledger stamp {stamp!r} is not a "
                           f"{list(STAMP_FIELDS)} state")
            continue
        segment = lookup.get((launch, stamp["segment_index"]) if merged_mode
                             else stamp["segment_index"])
        if segment is None or segment.get("launch_record_sha256") != launch:
            reasons.append(f"row {query_id}: its stamp names segment {stamp['segment_index']} "
                           f"(launch record {launch!r}), which the ledger does not hold")
            continue
        key = (id(segment), stamp["n_queries"])
        if key in seen or stamp["n_queries"] > segment["n_queries"]:
            reasons.append(f"row {query_id}: its stamp claims a state twice or beyond its "
                           f"segment's {segment['n_queries']} queries")
            continue
        seen.add(key)
        if stamp["n_queries"] > top.get(id(segment), (0, None))[0]:
            top[id(segment)] = (stamp["n_queries"], stamp["counts_sha256"])
    for _shard, block in blocks:
        for position, segment in enumerate(block):
            reasons += _segment_binding(segment, top.get(id(segment), (0, None)), exact,
                                        last=position == len(block) - 1,
                                        later=set().union(*[_shapes(s["counts"])
                                                            for s in block[position + 1:]]))
    return reasons


def _segment_binding(segment, top, exact, last, later):
    name, (m, stamped), n = f"segment {segment['segment_index']}", top, segment["n_queries"]
    stamped = counts_sha256(FaPartitionRecorder()) if m == 0 else stamped
    if n not in (m, m + 1):
        return [f"{name} holds {n} queries but its rows show {m}: a ledger state is at most "
                "one query ahead of its last row"]
    state = (segment["counts_sha256"] if n == m
             else counts_sha256(_counts_minus(segment["counts"], segment["delta_counts"])))
    if state != stamped:
        return [f"{name}'s state is not the one its last row was published under"]
    if n == m + 1 and exact and last:
        return [f"{name} is one query ahead of its rows, but it is its shard's last invocation, "
                "which completed"]
    if n == m + 1 and exact and not _shapes(segment["delta_counts"]) <= later:
        return [f"{name}'s lost attempt made call shapes no later segment executed again"]
    return []


def _stamp_reasons(out_dir, segments, exact, blocks=None):
    """The row binding over the rows ON DISK whose own digest verifies (an unverifiable row is
    regenerated by a resume and refused by the merge, never trusted here), each with its stamp."""
    try:
        stamps = [read_fa_stamp(out_dir, row) for row in read_rows(out_dir)
                  if row.get("row_sha256") == row_digest(row)]
    except (KeyError, TypeError, ValueError) as error:
        return [f"its rows cannot be read for their ledger stamps ({type(error).__name__}: "
                f"{error})"]
    return _row_binding_reasons(segments, stamps, exact, blocks)


def _ledger_reasons(payload, declared):
    """``(reasons, summed recorder or None)``: the ledger's own integrity -- its declaration,
    segment indices 0..n-1 once each, every segment, and the stored merged block."""
    if not isinstance(payload, dict) or payload.get("declared") != declared:
        return ["it is the ledger of another declaration (or of none)"], None
    segments = payload.get("segments")
    if not (isinstance(segments, list) and segments
            and all(isinstance(segment, dict) for segment in segments)):
        return ["it holds no list of segments"], None
    indices = [segment.get("segment_index") for segment in segments]
    # real integers only: False / True would pass as 0 / 1 (Codex rB-F2 finding 3)
    reasons = [] if all(map(_count, indices)) and indices == list(range(len(segments))) else [
        f"its segment indices {indices} are not 0..{len(segments) - 1} once each (a segment is "
        "missing or duplicated)"]
    found = [_segment_reasons(index, segment) for index, segment in enumerate(segments)]
    reasons += [reason for why, _recorder in found for reason in why]
    if any(recorder is None for _why, recorder in found):
        return reasons, None
    merged = FaPartitionRecorder.from_counts(*[segment["counts"] for segment in segments])
    wanted = {"counts_sha256": counts_sha256(merged), "executed": merged.executed(),
              "violations": merged.violations()}
    stored = payload.get("merged") if isinstance(payload.get("merged"), dict) else {}
    moved = sorted(key for key in set(wanted) | set(stored) if stored.get(key) != wanted.get(key))
    if moved:
        reasons.append(f"its stored merged block differs from the sum of its segments on {moved}")
    return reasons, merged


def _summary_reasons(summary, segments, merged, exact):
    """The run summary's ``fa_partition_executed`` against the ledger: at a merge, exactly the
    whole ledger; at a reopen absent, or exactly its first ``n_segments`` (an invocation
    finished, a later one crashed after adding a segment)."""
    block = (summary or {}).get("fa_partition_executed")
    if not isinstance(block, dict):
        return (["the run summary carries no fa_partition_executed block for it"]
                if exact or summary is not None else [])
    k = (block.get("ledger") or {}).get("n_segments")
    if not (_count(k) and 1 <= k <= len(segments)) or (exact and k != len(segments)):
        return [f"the run summary's ledger block covers {k!r} of its {len(segments)} segments"]
    prefix = merged if k == len(segments) else FaPartitionRecorder.from_counts(
        *[segment["counts"] for segment in segments[:k]])
    wanted = dict(prefix.report(), ledger={"file": FA_LEDGER_FILENAME, "n_segments": k,
                                           "counts_sha256": counts_sha256(prefix)})
    moved = sorted(key for key in set(wanted) | set(block) if block.get(key) != wanted.get(key))
    return [f"the run summary's fa_partition_executed is not its ledger's first {k} "
            f"segment(s) (differs on {moved})"] if moved else []


def _merged_reasons(segments, receipt, merged):
    """``(reasons, shard blocks)`` of a MERGED ledger (Codex rB-F2 finding 1): every segment
    names its shard and its shard-local index; each shard is one contiguous block numbered
    0..k-1 in its own order; and the merge report's receipt is exactly what the segments
    yield -- the summed partition, its digest, and each shard's census in shard order."""
    blocks = []
    for segment in segments:
        shard, local = segment.get("shard"), segment.get("shard_segment_index")
        if not (isinstance(shard, str) and shard and _count(local)):
            return [f"merged segment {segment.get('segment_index')!r} does not name its shard "
                    "and its shard-local index"], None
        if blocks and blocks[-1][0] == shard:
            blocks[-1][1].append(segment)
        elif shard in [name for name, _block in blocks]:
            return [f"the merged segments of shard {shard} are not one contiguous block"], None
        else:
            blocks.append((shard, [segment]))
    reasons = [f"shard {shard}'s merged segments are not its segments 0..{len(block) - 1} in order"
               for shard, block in blocks
               if [s["shard_segment_index"] for s in block] != list(range(len(block)))]
    wanted = {"executed": merged.executed(), "violations": merged.violations(),
              "counts_sha256": counts_sha256(merged), "per_shard": [
                  {"dir": shard, "counts_sha256": counts_sha256(FaPartitionRecorder.from_counts(
                      *[s["counts"] for s in block])), "n_segments": len(block)}
                  for shard, block in blocks]}
    if not isinstance(receipt, dict):
        return reasons + ["the merge report carries no fa_partition_executed receipt"], blocks
    moved = sorted(key for key in set(wanted) | set(receipt) if receipt.get(key) != wanted.get(key))
    return reasons + ([f"the merge report's fa_partition_executed receipt is not its merged "
                       f"ledger's (differs on {moved})"] if moved else []), blocks


def _read_object(path):
    """A JSON object from ``path``, or a refusal that names the file."""
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except ValueError as error:
        raise ValueError(f"{path} is not valid JSON ({error})") from None
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def fresh_out_dir(out_dir):
    """``(fresh, entries)``: an out-dir that is absent, or holds nothing but the operator's
    emitted launch records (top-level ``*.json`` files no pass writes) -- the only place a
    ledger may START (Codex rB-F2 finding 3); ``entries`` names everything else in it."""
    from src.localization import meshgrid_segments as ms

    out_dir = str(out_dir)
    if not os.path.exists(out_dir):
        return True, []
    if not os.path.isdir(out_dir):
        return False, [out_dir]
    written = (BINDING_FILENAME, RUN_SUMMARY_FILENAME, MERGE_REPORT_FILENAME, FA_LEDGER_FILENAME,
               FA_FAILURE_FILENAME, ms.SEGMENTS_FILENAME)
    found = sorted(name for name in os.listdir(out_dir)
                   if not (name.endswith(".json") and os.path.isfile(os.path.join(out_dir, name)))
                   or name in written or name.startswith("diagnostics_"))
    return not found, found


def read_fa_ledger(out_dir, declared, exact=False):
    """THE validating reader of a run's executed-partition ledger (Codex rB-F1 finding 3) --
    the one every reopen (``open_fa_ledger``, the driver's preflight) and the merge
    (``fa_partition_census``, ``exact=True``) go through. It refuses, naming every broken
    check: another declaration; segment indices not 0..n-1 once each; a count or query count
    that is not a non-negative integer; counts not canonical or not hashing to their
    segment's ``counts_sha256``; a stored merged block (digest, executed, violations) that
    is not the sum of the segments; a run summary block that is not the whole ledger (merge)
    or a prefix of it (reopen); an executed partition not the declared one (exactly at a
    merge, as a consistent prefix at a reopen). A vanilla run carries no ledger and no block.
    Returns ``None`` when no ledger is on disk and none is required."""
    out_dir = str(out_dir)
    assert_admissible(out_dir)                         # A6 (iv), before anything is read
    path = os.path.join(out_dir, FA_LEDGER_FILENAME)
    summary_path = os.path.join(out_dir, RUN_SUMMARY_FILENAME)
    summary = _read_object(summary_path) if os.path.isfile(summary_path) else None
    # a MERGED run (it publishes the merge report) is joined to the merge report's receipt
    merge_path = os.path.join(out_dir, MERGE_REPORT_FILENAME)
    merge_report = _read_object(merge_path) if os.path.isfile(merge_path) else None
    receipt = (merge_report or {}).get("fa_partition_executed")
    if declared is None:
        if (os.path.exists(path) or (summary or {}).get("fa_partition_executed") is not None
                or receipt is not None):
            raise ValueError(f"{out_dir}: a vanilla run carries no frame-average executed "
                             f"partition, but it carries a {FA_LEDGER_FILENAME}, a summary "
                             "block or a non-null merge-report receipt")
        return None
    if not os.path.isfile(path):
        fresh, found = fresh_out_dir(out_dir)
        if exact or (summary or {}).get("fa_partition_executed") is not None or not fresh:
            raise ValueError(f"{out_dir} publishes no {FA_LEDGER_FILENAME}: its executed "
                             "frame-average partition is not on disk, so it can be neither "
                             "resumed nor merged" + ("" if fresh else (
                                 f"; and it is not a fresh out-dir (it holds {found[:4]}), the "
                                 "only place a ledger may start -- restart the shard in a "
                                 "fresh out-dir (Codex rB-F2 finding 3)")))
        return None
    payload = _read_object(path)
    reasons, merged = _ledger_reasons(payload, declared)
    blocks = None
    if not reasons and merge_report is not None:
        reasons, blocks = (_merged_reasons(payload["segments"], receipt, merged) if exact else (
            ["it is a merged run's ledger, which is final: a merged directory is never "
             "reopened"], None))
    elif not reasons:
        reasons = _summary_reasons(summary, payload["segments"], merged, exact)
    if not reasons:           # every ledger state bound to the published rows (rB-F2 3)
        reasons = _stamp_reasons(out_dir, payload["segments"], exact, blocks)
    if not reasons and (exact or any(merged.counts.values())):   # nothing ran yet: a prefix
        try:
            assert_fa_partition_executed(declared, merged, final=exact)
        except ValueError as error:
            reasons.append(str(error))
    if reasons:
        raise ValueError(f"{path} does not verify: " + "; ".join(reasons))
    return {"path": path, "payload": payload, "merged": merged,
            "counts_sha256": counts_sha256(merged), "n_segments": len(payload["segments"])}


def open_fa_ledger(out_dir, declared, launch_record_sha256=None):
    """This invocation's (new) segment of the run's ledger, opened only on a ledger that
    verifies (:func:`read_fa_ledger`) or, in a fresh out-dir, a new one -- and PUBLISHED at
    once, so the binding, a launch segment or a row never exists without it (rB-F2 3)."""
    found = read_fa_ledger(out_dir, declared)
    path = os.path.join(str(out_dir), FA_LEDGER_FILENAME)
    payload = (found["payload"] if found is not None
               else {"declared": declared, "segments": [], "note": FA_LEDGER_NOTE})
    payload["segments"].append({"segment_index": len(payload["segments"]), "n_queries": 0,
                                "launch_record_sha256": launch_record_sha256, "counts": []})
    ledger = {"path": path, "payload": payload}
    persist_fa_ledger(ledger, FaPartitionRecorder(), query=False)
    return ledger


def adopt_fa_ledger(ledger, out_dir, declared, launch_record_sha256):
    """The ledger the driver opened for THIS invocation before it wrote anything else (its
    out-dir, declaration and launch record, its segment still empty), or a refusal."""
    segment = (((ledger or {}).get("payload") or {}).get("segments") or [{}])[-1]
    if (os.path.abspath(str(ledger.get("path"))) != os.path.abspath(
            os.path.join(str(out_dir), FA_LEDGER_FILENAME))
            or ledger["payload"].get("declared") != declared
            or (segment.get("launch_record_sha256"), segment.get("n_queries"),
                segment.get("counts")) != (launch_record_sha256, 0, [])):
        raise ValueError("the executed-partition ledger handed to this pass is not the one just "
                         "opened for its out-dir, declaration and launch record")
    return ledger


def persist_fa_ledger(ledger, recorder, query=True):
    """Rewrite the ledger with this segment's counts so far; the run's summed recorder."""
    segment = ledger["payload"]["segments"][-1]
    # the state at n_queries - 1: the last query state, or (a non-query rewrite) the same base
    prior = (segment["counts"] if query else _counts_minus(
        segment["counts"], segment.get("delta_counts", [])).to_counts())
    delta = _counts_minus(recorder.to_counts(), prior)
    if delta is None:
        raise ValueError("the split's recorder holds fewer calls than its ledger segment already "
                         "published; the executed partition only ever accumulates")
    segment.update(counts=recorder.to_counts(), counts_sha256=counts_sha256(recorder),
                   n_queries=segment["n_queries"] + int(query), delta_counts=delta.to_counts())
    merged = FaPartitionRecorder.from_counts(*[s["counts"] for s in ledger["payload"]["segments"]])
    ledger["payload"]["merged"] = {"counts_sha256": counts_sha256(merged),
                                   "executed": merged.executed(),
                                   "violations": merged.violations()}
    os.makedirs(os.path.dirname(ledger["path"]) or ".", exist_ok=True)
    write_json(ledger["path"], ledger["payload"])
    return merged


def fa_executed_block(declared, merged, n_segments):
    """A scored fa pass's ``fa_partition_executed``: the run's summed partition held to the
    declaration (exactly) and the digest of the ledger counts it was derived from."""
    report = (assert_fa_partition_executed(declared, merged) if any(merged.counts.values())
              else merged.report())
    return dict(report, ledger={"file": FA_LEDGER_FILENAME, "counts_sha256": counts_sha256(merged),
                                "n_segments": int(n_segments)})


def fa_partition_census(shards, declared):
    """Each shard's ledger read through THE validating reader (:func:`read_fa_ledger`,
    exactly: integrity, a run summary covering every segment, the exact declared partition),
    then the sum over shards: ``{"ledger", "receipt"}`` -- the merged ledger (ONE index space
    0..n-1 in shard order, each segment keeping its ``shard`` and ``shard_segment_index``)
    and the merge report's receipt, both verified as the reader verifies a merged run
    (Codex rB-F2 finding 1) -- or ``None`` for a vanilla run, whose shards carry none."""
    per_shard, segments, recorders = [], [], []
    for shard in shards:
        # THE reader, exactly: integrity, the whole-ledger summary, the exact partition
        ledger = read_fa_ledger(shard["dir"], declared, exact=True)
        if declared is None:
            continue
        per_shard.append({"dir": shard["dir"], "counts_sha256": ledger["counts_sha256"],
                          "n_segments": ledger["n_segments"]})
        for segment in ledger["payload"]["segments"]:
            segments.append(dict(segment, segment_index=len(segments), shard=shard["dir"],
                                 shard_segment_index=segment["segment_index"]))
        recorders.append(ledger["merged"])
    if declared is None:
        return None
    union = FaPartitionRecorder.from_counts(*[recorder.to_counts() for recorder in recorders])
    report = assert_fa_partition_executed(declared, union)
    receipt = {"executed": report["executed"], "violations": report["violations"],
               "counts_sha256": counts_sha256(union), "per_shard": per_shard}
    merged = {"declared": declared, "segments": segments, "note": FA_LEDGER_NOTE,
              "merged": {"counts_sha256": receipt["counts_sha256"],
                         "executed": union.executed(), "violations": union.violations()}}
    reasons, summed = _ledger_reasons(merged, declared)
    blocks = None
    if not reasons:
        reasons, blocks = _merged_reasons(segments, receipt, summed)
    if reasons:
        raise ValueError("the merged executed-partition ledger would not verify under its own "
                         "reader: " + "; ".join(reasons))
    return {"ledger": merged, "receipt": receipt, "blocks": blocks}


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
    "cond_autocast", "dataset_config_sha256",
    # the DUMP AUTHORITY: a resume that arrives with a different (or a new) case
    # list would skip an already-complete query without ever producing the dump
    # it now asks for, so changing it is a different run
    "dump_cases_sha256", "dataset_config")

#: exp_22 R2 (plan r2 §3 R2-A; Codex B6/H2): the eval-protocol flags every R2
#: launch declares explicitly instead of inheriting -- the frame-average orbit
#: (a vanilla arm binds the explicit canonical ``[0.0]``), the global rotation
#: (registered 0) and the frame-average chunk plan (announcement 06; ``None`` for
#: a vanilla arm). A binding carries ALL of them or NONE: none is the legacy R1
#: generation -- exactly the published Aug-2026 P1 comparator, whose digest must
#: not move -- and a partial set is refused rather than read as either.
RUN_BINDING_PROTOCOL_FIELDS = ("frame_avg_angles", "rotate_deg", "fa_partition")


def declared_fa_partition(angles, *, batch_rows, num_samples, source_chunk):
    """The announcement-06 chunk plan a frame-averaged mesh pass DECLARES.

    exp_20's per-angle plan (``cap_policy = candidate_micro_batch``, so every
    rotated angle is its own forward) at the three batch shapes the R2 split
    conditions with (plan r2 §2 table): the per-query context ViT orbit (one
    query), the scored candidate batch ``context_poses`` is recomputed in
    (``batch_rows // num_samples`` candidates; angle-invariant, never averaged)
    and the receiver/source-cache chunk (``source_chunk``). R2-B executes the
    split and must record the EXECUTED partition against this declaration.
    """
    orbit = len(list(angles))
    source = int(source_chunk)
    return {"frame_avg_chunk_plan": "per_angle", "cap_policy": "candidate_micro_batch",
            "orbit_size": orbit, "angles_per_chunk": 1, "n_orbit_forwards": orbit - 1,
            "shared_angle_count": 1,
            "context_vit": {"candidate_micro_batch": 1, "frame_avg_fwd_cap": 1},
            "context_scored_batch": {
                "candidate_micro_batch": max(1, int(batch_rows) // int(num_samples))},
            "source_cache": {"candidate_micro_batch": source, "frame_avg_fwd_cap": source}}


def binding_fields(binding):
    """The strict fields ``binding`` is digested over: legacy R1, or R1 + R2 protocol."""
    present = [field for field in RUN_BINDING_PROTOCOL_FIELDS if field in binding]
    if not present:
        return RUN_BINDING_FIELDS
    if len(present) != len(RUN_BINDING_PROTOCOL_FIELDS):
        missing = [field for field in RUN_BINDING_PROTOCOL_FIELDS if field not in binding]
        raise ValueError(f"the run binding declares the R2 protocol fields {present} but not "
                         f"{missing}; they are bound all or none -- a partial set is neither the "
                         "legacy R1 binding nor a complete R2 one")
    return RUN_BINDING_FIELDS + RUN_BINDING_PROTOCOL_FIELDS

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

    fields = binding_fields(binding)
    missing = [field for field in RUN_BINDING_FIELDS if field not in binding]
    if missing:
        raise ValueError(f"the run binding is missing {missing}; every registered field must be "
                         "pinned before a query is generated")
    extra = [key for key in binding if key not in fields]
    if extra:
        raise ValueError(f"the run binding carries unregistered fields {extra}")
    return canonical_sha256({field: binding[field] for field in sorted(fields)})


def write_binding(out_dir, binding, advisory=None, declared_rooms=None):
    """Publish the binding beside the artifacts it authorizes."""
    os.makedirs(str(out_dir), exist_ok=True)
    payload = {field: binding[field] for field in binding_fields(binding)}
    payload["binding_sha256"] = binding_sha256(binding)
    payload["advisory"] = {field: (advisory or {}).get(field)
                           for field in RUN_BINDING_ADVISORY}
    payload["advisory_history"] = []
    # the shard's own scope: pinned for ITS resume, and deliberately outside the
    # strict digest so a merge can require two shards' base bindings to be equal
    payload["declared_rooms"] = None if declared_rooms is None else list(declared_rooms)
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


def assert_binding(out_dir, binding, advisory=None, declared_rooms=None):
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
    # the union of both generations' fields: a legacy pass is not continued by
    # an R2 run (or vice versa) just because the R1 fields agree
    fields = list(RUN_BINDING_FIELDS) + [field for field in RUN_BINDING_PROTOCOL_FIELDS
                                         if field in published or field in binding]
    differing = [field for field in fields if published.get(field) != binding.get(field)]
    if differing:
        raise ValueError(f"this run does not continue the published one: {differing} differ "
                         f"(published binding {str(published.get('binding_sha256'))[:12]}..., "
                         f"this run {binding_sha256(binding)[:12]}...)")
    published_rooms = published.get("declared_rooms")
    wanted_rooms = None if declared_rooms is None else list(declared_rooms)
    if published_rooms != wanted_rooms:
        raise ValueError(f"this run sets declared_rooms={wanted_rooms} but the published pass "
                         f"declared {published_rooms}; a shard resumes its OWN rooms, and "
                         "mixing scopes in one directory would break the merge census")
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
            "position": row.get("position"), "room_id": row.get("room_id"),
            "launch_record_sha256": row.get("launch_record_sha256")}


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
            if not is_argmax_stable(block_a.get("margin", 0.0), tolerance):
                at_risk += 1
    max_delta = max((entry["max_abs_delta"] for entry in by_k.values()), default=0.0)
    return {"n_queries": len(left), "by_k": by_k, "max_abs_delta": max_delta,
            "tolerance": float(tolerance), "within_tolerance": max_delta <= float(tolerance),
            "bit_exact": bool(fingerprints_equal and max_delta == 0.0),
            "n_flipped": len(flipped), "flipped": flipped,
            "n_argmax_at_risk": at_risk,
            "stability_bound": argmax_stability_bound(tolerance),
            "contract": DETERMINISM_CONTRACT}


def completed_queries(out_dir, binding_sha256=None, launch_records=None):
    """``(verified query ids, rejected verdicts)`` for a resume.

    Only a query whose row AND sidecar re-verify is skipped; anything else is
    reported and regenerated, so a half-written artifact can never be adopted as
    a finished one. ``launch_records`` (exp_22 R2-A, canonical passes) is the set
    of record digests this run's segment ledger knows: a row naming any other --
    or none -- would break the merge census, so it is regenerated, not adopted.
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
            if verdict["ok"] and launch_records is not None:
                # the stamp of the very row that verified -- no second read
                stamp = verdict.get("launch_record_sha256")
                if stamp not in set(launch_records):
                    verdict = dict(verdict, ok=False,
                                   reason=f"the row names launch record {stamp!r}, which this "
                                          "run's segment ledger does not hold")
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


def load_dump_cases(path, expected_sha256=None):
    """A REGISTERED case list: its digest is the authority, not its existence.

    Without this any JSON file could extend the announcement-08 exemption simply
    by being passed on the command line, and an edited list would still pass
    (r7 review MAJOR DUMP). The digest is supplied separately -- from the
    registration -- and the file must equal it.
    """
    found = file_sha256(path)
    if not expected_sha256:
        raise ValueError(f"{path} was offered as a dump case list without a registered "
                         f"digest. Its sha256 is {found}; pass it explicitly so the list is "
                         "authorized by the registration and not by its own existence")
    if str(expected_sha256) != found:
        raise ValueError(f"{path} hashes to {found[:12]}... but its registered digest is "
                         f"{str(expected_sha256)[:12]}...; the list changed after registration")
    with open(str(path)) as handle:
        payload = json.load(handle)
    ids = payload.get("query_ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(q, str) for q in ids):
        raise ValueError(f"{path} must carry a non-empty list of query_ids")
    return {"query_ids": list(ids), "sha256": found, "path": str(path)}


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
                        advisory=None, totals=None, fa_partition_executed=None):
    """Publish probe timings under a diagnostics stem, never as query artifacts.

    The binding is serialized over ``binding_fields`` -- the R2 protocol fields
    included (Codex rA A8) -- and an fa probe carries its executed partition.
    """
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
                                                 for field in binding_fields(binding)
                                                 if field in binding},
        "binding_sha256": binding_sha256,
        "advisory": dict(advisory or {}),
        "noise_policy": None if binding is None else binding.get("noise_policy"),
        "projection": project_cost(records, totals=totals),
        "determinism_contract": DETERMINISM_CONTRACT,
        "agree_leakage_caveat": AGREE_LEAKAGE_CAVEAT,
        "scorer_readout_deviation": SCORER_READOUT_DEVIATION,
        "records": list(records),
        **({} if fa_partition_executed is None
           else {"fa_partition_executed": fa_partition_executed})})
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
    #: exp_22 R2-B: the engaged :class:`FaSplit` (fa_invariant only), else ``None``
    split: object = None


def split_reasons(split):
    """Why ``split`` is not a complete engaged five-way split (Codex rB NIT 8), or ``[]``."""
    if not isinstance(split, FaSplit):
        return [f"{type(split).__name__} is not a FaSplit"]
    angles = tuple(split.angles or ())
    reasons = [] if split.name == FA_SPLIT else [f"the split is named {split.name!r}"]
    if not isinstance(split.recorder, FaPartitionRecorder):
        reasons.append("the split has no engaged executed-partition recorder")
    if not angles or not all(isinstance(a, (int, float)) and np.isfinite(a) for a in angles):
        reasons.append(f"the orbit {list(angles)} is empty or not finite")
    elif float(angles[0]) != 0.0:
        reasons.append(f"the orbit {list(angles)} does not start at the identity 0.0")
    if len(set(angles)) != len(angles):
        reasons.append(f"the orbit {list(angles)} repeats an angle")
    return reasons


def assert_cacheable(cond_method, split=None):
    """The §1.5 caches are valid for vanilla conditioning with no split, and for
    fa_invariant ONLY under the engaged five-way split (exp_22 R2-B, plan r2 §2).

    Under the released C4 frame average the cylindrical transform makes
    ``context_poses.dphi`` candidate-dependent, so the vanilla per-query context
    cache would serve the wrong tokens; the split recomputes it per scored
    candidate batch instead. Any other pairing is refused rather than mis-cached.
    """
    method = str(cond_method)
    if method == "vanilla" and split is None:
        return True
    malformed = split_reasons(split) if method == "fa_invariant" and split is not None else []
    if method == "fa_invariant" and split is not None and not malformed:
        return True
    raise ValueError(f"the per-query context cache is registered for vanilla conditioning with "
                     f"no split and for fa_invariant under the {FA_SPLIT!r} split only, but this "
                     f"engine runs cond_method={method!r} with split "
                     f"{getattr(split, 'name', split)!r} {malformed}; under fa_invariant the released "
                     "cylindrical transform makes context_poses candidate-dependent (inherited "
                     "plan §1.5), so it may only be cached the way the split caches it")


def split_for(cond_method, frame_avg_angles=None):
    """The split a method is cached under: none for vanilla; for fa_invariant the
    five-way split over its DECLARED orbit, with a fresh executed-partition recorder."""
    if str(cond_method) != "fa_invariant":
        return None
    angles = tuple(float(angle) for angle in (frame_avg_angles or ()))
    if not angles or angles[0] != 0.0:
        raise ValueError(f"fa_invariant needs its explicit frame-average orbit, starting at the "
                         f"identity 0.0; got {list(angles)}")
    return FaSplit(angles=angles, recorder=FaPartitionRecorder())


def query_context(engine, md):
    """The per-query cached branch: vanilla's three context ids, or the split's two."""
    if engine.split is None:
        return context_conditioning(engine.conditioner, md, engine.device)
    return fa_query_conditioning(engine.split, engine.conditioner, md, engine.device)


def candidate_inputs(engine, query, source):
    """What the split recomputes per scored batch -- the raw context poses (read
    from ``source``, the query's metadata or its buffer entry) and the query's
    camera-frame candidate positions -- or ``None`` under vanilla conditioning."""
    if engine.split is None:
        return None
    positions = (np.asarray(query.coordinates, dtype=np.float64)
                 - np.asarray(query.receiver_xyz, dtype=np.float64))
    return {"context_poses": source["context_poses"], "positions": positions}


def candidate_rows(engine, candidate, start, stop, num_samples, configured, record=True):
    """The split's ``context_poses`` for candidates ``[start, stop)``, one row per draw."""
    positions = candidate["positions"][start:stop]
    poses = fa_candidate_conditioning(engine.split, engine.conditioner,
                                      candidate["context_poses"], positions, engine.device,
                                      configured=configured, record=record)
    rows = torch.arange(positions.shape[0]).repeat_interleave(int(num_samples))
    return {key: _select(entry, rows.to(entry[0].device)) for key, entry in poses.items()}


def scored_conditioning(engine, context, cache, rows, candidate=None, *, start=0, stop=None,
                        num_samples=1, configured=None, record=True):
    """What the DiT is conditioned on for one scored batch: the two caches expanded
    over ``rows`` and -- under the split -- the batch's own ``context_poses``."""
    merged = expand_conditioning(context, cache.conditioning, rows, engine.device)
    if candidate is not None:
        merged.update(candidate_rows(engine, candidate, start, stop, num_samples, configured,
                                     record=record))
    return merged


def assert_declared_rooms(rooms, plan):
    """The canonical, ordered room set a shard declares -- or a refusal.

    A shard that silently drops an unknown room, or counts one twice, would
    produce a merge census that adds up while missing queries, so every failure
    mode is a refusal here rather than a filter that quietly matches nothing.
    """
    declared = [str(room) for room in (rooms or [])]
    if not declared:
        raise ValueError("a room shard may not be empty: pass the canonical room ids it owns, "
                         "or omit --rooms to run the whole registered subset")
    seen = set()
    for room in declared:
        if room in seen:
            raise ValueError(f"room {room!r} is declared twice; a shard owns each of its rooms "
                             "exactly once")
        seen.add(room)
    unknown = sorted(seen - set(plan.rooms))
    if unknown:
        raise ValueError(f"{unknown} are not in the audit's {len(plan.rooms)} rooms; a shard "
                         "can only own rooms the G1 audit published")
    return sorted(seen)


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
                     noise_policy, batch_rows, timer, candidate=None):
    """Generate and score every candidate of one query -> sims ``[M, K]``.

    ``candidate`` (:func:`candidate_inputs`, the fa split only) adds each scored
    batch's own ``context_poses``; ``None`` is the vanilla path, unchanged.
    """
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
            merged = scored_conditioning(engine, context, cache, rows, candidate, start=start,
                                         stop=start + per_chunk, num_samples=num_samples,
                                         configured=per_chunk)
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
    candidate = candidate_inputs(engine, query, context)
    if candidate is not None:
        # the dump regenerates its selection in ONE forward: not a scored batch,
        # so it is kept out of the executed partition
        candidate = dict(candidate, positions=candidate["positions"][rows])
        merged.update(candidate_rows(engine, candidate, 0, len(rows), num_samples, len(rows),
                                     record=False))
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
        context = query_context(engine, md)
        cache = ReceiverCache.build(engine.conditioner, query.receiver_id, base_md, indices,
                                    positions, engine.device, chunk=source_chunk,
                                    fa=engine.split)
        obs_embedding = engine.embedder(
            torch.as_tensor(obs_wav).to(engine.device))[0].float().cpu()
        sims = _score_one_query(engine, query, context, cache, obs_embedding, seed=seed,
                                num_samples=num_samples, noise_policy=noise_policy,
                                batch_rows=batch_rows, timer=_Timer(engine.device),
                                candidate=candidate_inputs(engine, query, md))
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
             geometry_tol=CONTEXT_JOIN_TOLERANCE, binding_sha256=None, rooms=None,
             launch_record_sha256=None, fa_partition=None, fa_ledger=None):
    """Score the whole registered subset, room block by room block.

    The stream is the released loader in D1 order and is walked ONCE: every
    query's draw is verified against the frozen context manifest, its context
    branch is conditioned once, and its observation is embedded once. A room's
    queries are then run receiver group by receiver group, so the source branch
    costs one conditioner call per (receiver, candidate) in that receiver's
    union and exactly one group's cache is resident at a time.

    Under the fa split (exp_22 R2-B) ``fa_partition`` is the binding's DECLARED
    partition: the executed one is checked against it before every row, and
    published as ``fa_partition_executed`` in the returned summary.
    """
    from src.localization import meshgrid_queries as mq

    assert_cacheable(engine.cond_method, engine.split)
    if engine.split is not None and not isinstance(fa_partition, dict):
        raise ValueError("a frame-averaged pass must declare its announcement-06 partition "
                         "(run_binding.json fa_partition) before anything runs against it")
    assert_registered_noise_policy(noise_policy, allow_unregistered_noise_policy)
    # a scored fa pass keeps its executed partition on disk, one segment per invocation --
    # opened here, or by the driver before it wrote the binding (``fa_ledger``, adopted)
    if engine.split is None or probe is not None:
        if fa_ledger is not None:
            raise ValueError("only a scored frame-averaged pass carries an executed-partition "
                             "ledger")
    elif fa_ledger is None:
        fa_ledger = open_fa_ledger(out_dir, fa_partition, launch_record_sha256)
    else:
        adopt_fa_ledger(fa_ledger, out_dir, fa_partition, launch_record_sha256)
    excluded_room = mq.EXCLUDED_ROOM if excluded_room is None else excluded_room
    done = set(done)
    by_position = {int(record["position"]): record for record in records}
    room_order = assert_room_blocks(records)

    declared = None if rooms is None else assert_declared_rooms(rooms, plan)
    selected = None
    if probe is not None:
        selected = set(probe_groups(plan, probe, room_order=room_order, room=probe_room)[0])
    # a shard OWNS rooms; it still walks and verifies the whole stream, because
    # every query's draw depends on the complete pass (inherited plan §1.1)
    owned = (None if declared is None else set(declared))
    if owned is not None and selected is not None:
        selected = {pair for pair in selected if pair[0] in owned}

    dump_queries = set(dump_queries or ())
    state = {"n_scored": 0, "n_skipped": 0, "n_conditioner_rows": 0,
             "n_candidate_query_pairs": 0, "n_generated": 0, "n_dumped": 0,
             "n_geometry_tolerated": 0, "n_contexts_conditioned": 0,
             "rooms": [], "declared_rooms": declared, "argmax_stability": {},
             "batching": {"batch_rows": int(batch_rows), "source_chunk": int(source_chunk)},
             "probe_records": [], "timings_s": {},
             "launch_record_sha256": launch_record_sha256}
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
                  dump_top_n=dump_top_n, binding_sha256=binding_sha256,
                  launch_record_sha256=launch_record_sha256, fa_partition=fa_partition,
                  fa_ledger=fa_ledger)
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
            unselected = (selected is not None
                          and not any(room == room_id for room, _ in selected))
            if (owned is not None and room_id not in owned) or unselected:
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
            context = query_context(engine, md)
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
    if fa_ledger is not None:
        # the RUN's executed partition: every segment's counts, this one included (a
        # finished shard resumed conditions nothing and still publishes it -- rB 5)
        merged = persist_fa_ledger(fa_ledger, engine.split.recorder, query=False)
        n_segments = len(fa_ledger["payload"]["segments"])
        try:
            state["fa_partition_executed"] = fa_executed_block(fa_partition, merged, n_segments)
        except ValueError as error:                    # A6 (iv): non-admissible, on disk
            write_partition_failure(out_dir, fa_partition, merged, "final", error, n_segments,
                                    launch_record_sha256)
            raise
    elif engine.split is not None:
        # the no-quality probe: what ran, prefix-consistent, published not judged
        ran = any(engine.split.recorder.counts.values())
        state["fa_partition_executed"] = (
            assert_fa_partition_executed(fa_partition, engine.split.recorder, final=False)
            if ran else engine.split.recorder.report())
    return state


def _run_room(engine, room_plan, buffer, depths, out_dir, state, *, selected, done, seed,
              tau, num_samples, prefixes, noise_policy, batch_rows, source_chunk, on_row,
              probe, geometry_tol, dump_queries=(), dump_top_n=DUMP_TOP_N,
              binding_sha256=None, launch_record_sha256=None, fa_partition=None,
              fa_ledger=None):
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
                                        positions_cam, engine.device, chunk=source_chunk,
                                        fa=engine.split)
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
                    batch_rows=batch_rows, timer=query_timer,
                    candidate=candidate_inputs(engine, query, context))
                if engine.split is not None:
                    # Codex rA A8: the executed partition is the declared one before
                    # this query's row (or probe record, or dump) can exist -- a
                    # prefix that has not yet filled a micro-batch is consistent
                    try:
                        assert_fa_partition_executed(fa_partition, engine.split.recorder,
                                                     final=False)
                    except ValueError as error:        # A6 (iv): a scored shard says so
                        if fa_ledger is not None:
                            write_partition_failure(
                                out_dir, fa_partition, engine.split.recorder, "before_row",
                                error, len(fa_ledger["payload"]["segments"]),
                                launch_record_sha256)
                        raise
                    if fa_ledger is not None:          # on disk BEFORE the row (rB 5)
                        persist_fa_ledger(fa_ledger, engine.split.recorder)
                        write_fa_stamp(out_dir, query, fa_ledger)       # rB-F2 finding 3
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
                if launch_record_sha256 is not None:
                    # the segment that produced this row (exp_22 R2-A); absent
                    # without a record, so a record-free row is byte-identical
                    row["launch_record_sha256"] = str(launch_record_sha256)
                if query.query_id in dump_queries:
                    row.update(_dump_query(engine, query, context, cache, out_dir, row,
                                           scored, seed=seed, num_samples=num_samples,
                                           noise_policy=noise_policy, top_n=dump_top_n))
                    state["n_dumped"] += 1
                write_query_artifact(out_dir, row, sims, binding_sha256=binding_sha256)
                for key, block in row["by_k"].items():
                    entry = state["argmax_stability"].setdefault(
                        int(key), {"n_queries": 0, "n_unstable": 0,
                                   "min_margin": float("inf"),
                                   "stability_bound": argmax_stability_bound()})
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


def cond_autocast_context(device, enabled, dtype):
    """The conditioners' autocast context, exactly as :func:`build_mesh_engine` enters it."""
    import contextlib

    if not enabled:
        return contextlib.nullcontext()
    if dtype is None:
        return torch.amp.autocast(device)
    return torch.amp.autocast(device, dtype=dtype)


def cond_autocast_state(device, mode):
    """The conditioners' autocast AS EXECUTED (Codex rB finding 6; plan r2 amendment A4 iii).

    ``eval_FLAC.resolve_cond_autocast`` maps the flag; the engine's own context is
    then ENTERED and torch is asked what it runs -- so ``default`` is published as
    its resolved state (float16 on CUDA, bfloat16 on CPU), never as the word.
    Disabled autocast runs the conditioners in their float32 parameters.
    """
    from eval_FLAC import resolve_cond_autocast

    enabled, dtype = resolve_cond_autocast(mode)
    kind = torch.device(str(device)).type
    with cond_autocast_context(device, enabled, dtype):
        live = bool(torch.is_autocast_enabled(kind))
        resolved = torch.get_autocast_dtype(kind) if live else torch.float32
    return {"cond_autocast_requested": str(mode), "cond_autocast_enabled": live,
            "cond_autocast_dtype": str(resolved).replace("torch.", ""),
            "cond_autocast_device_type": kind}


def build_mesh_engine(ckpt_path, model_config, agree, device="cpu", cond_method="vanilla",
                      cond_autocast="default", steps=STEPS, cfg_scale=CFG_SCALE, ckpt=None,
                      readout=SCORER_READOUT, frame_avg_angles=None):
    """Build the frozen generator and wrap it as a :class:`MeshEngine`.

    Follows ``eval_FLAC.evaluate_model``'s lines of record through
    ``eval_localization`` -- matmul precision, ARE refusal, EMA remap,
    load-integrity check, wrapper construction, eval/no-grad, latent length from
    the pretransform ratio -- so exp_22 scores the same generative process the
    release evaluation runs. The only addition is the ``only_ids`` seam the two
    conditioning caches need. fa_invariant is built under the five-way split over
    ``frame_avg_angles`` (exp_22 R2-B); without that orbit it is refused.
    """
    import copy

    from eval_FLAC import check_load_integrity, resolve_cond_autocast
    from eval_localization import assert_no_are, assert_rectified_flow, prepare_state_dict
    from src.inference.sampling import sample_discrete_euler
    from src.localization.agree_embed import embed_rirs
    from src.models.factory import create_model_from_config
    from src.training.factory import create_training_wrapper_from_config

    split = split_for(cond_method, frame_avg_angles)
    assert_cacheable(cond_method, split)
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

    def conditioner(metadata, _device, only_ids=None):
        with cond_autocast_context(device, ac_enabled, ac_dtype):
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
                        cond_method=cond_method, split=split)
    context = {"module": module, "model": model, "model_config": model_config,
               "weights_source": weights_source, "device": device,
               "latent_shape": engine.latent_shape,
               "cond_autocast": cond_autocast_state(device, cond_autocast)}
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

    if getattr(engine, "split", None) is not None:
        raise ValueError("cache_parity_check is the VANILLA proof; an engine with the fa split "
                         "is gated by meshgrid_fa_parity.cache_parity (fa_cache_parity, gate 2)")
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


# --------------------------------------------------------------------------- #
# the shard merge and its census
# --------------------------------------------------------------------------- #
def _read_shard(shard_dir):
    """One shard's published binding and summary, or a refusal."""
    assert_admissible(shard_dir)                       # A6 (iv): named before anything else
    binding_path = os.path.join(str(shard_dir), BINDING_FILENAME)
    if not os.path.isfile(binding_path):
        raise ValueError(f"{shard_dir} publishes no {BINDING_FILENAME}; a shard that cannot "
                         "say what produced it may not enter a merge")
    with open(binding_path) as handle:
        binding = json.load(handle)
    # RECOMPUTE the digest from the content: a stored string is what a tampered
    # shard would keep saying, so it is evidence of nothing (r8 review, gate 7)
    missing = [field for field in RUN_BINDING_FIELDS if field not in binding]
    if missing:
        raise ValueError(f"{shard_dir}'s binding is missing {missing}; it cannot be "
                         "recomputed, so it cannot be trusted")
    recomputed = binding_sha256({field: binding[field] for field in binding_fields(binding)})
    if binding.get("binding_sha256") != recomputed:
        raise ValueError(f"{shard_dir}'s published binding does not match its own content: it "
                         f"stores {str(binding.get('binding_sha256'))[:12]}... but hashes to "
                         f"{recomputed[:12]}...; it was edited after publication")
    summary_path = os.path.join(str(shard_dir), "run_summary.json")
    if not os.path.isfile(summary_path):
        raise ValueError(f"{shard_dir} publishes no run_summary.json; the merge census needs "
                         "its conditioner-row count, which only the pass can report")
    with open(summary_path) as handle:
        summary = json.load(handle)
    return {"dir": str(shard_dir), "binding": binding, "summary": summary,
            "declared_rooms": list(binding.get("declared_rooms") or []),
            "advisory": dict(binding.get("advisory") or {}),
            "binding_sha256": recomputed}


def merge_shards(shard_dirs, out_dir, plan, records, totals=None, expected_rooms=None):
    """Publish a merged run into a FRESH directory, and only after the census.

    The census is the whole point: two shards that each finished happily can
    still, between them, miss a query, score one twice, or have been produced by
    different protocols. Nothing is copied until every one of those is excluded.
    """
    import shutil

    totals = dict(REGISTERED_TOTALS if totals is None else totals)
    shards = [_read_shard(directory) for directory in shard_dirs]
    if len(shards) < 2:
        raise ValueError("a merge combines at least two shards")

    out_dir = str(out_dir)
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        raise ValueError(f"{out_dir} is not fresh: a merge publishes into an empty directory "
                         "so that a partial or older result cannot survive inside it")

    # (1) identical strict bindings
    digests = {shard["binding_sha256"] for shard in shards}
    if len(digests) != 1 or None in digests:
        raise ValueError(f"the shards were not produced under the same binding: "
                         f"{sorted(str(d)[:12] for d in digests)}")
    binding_sha256 = shards[0]["binding_sha256"]
    base = {field: shards[0]["binding"][field]
            for field in binding_fields(shards[0]["binding"])}

    # (2) pinned production advisory values, in the bindings AND in every row
    advisories = {json.dumps(shard["advisory"], sort_keys=True) for shard in shards}
    if len(advisories) != 1:
        raise ValueError(f"the shards pin different advisory batching values "
                         f"{sorted(advisories)}; a merged run must state one batching")
    advisory = shards[0]["advisory"]

    # (2b) exp_22 rB-F1 (Codex rB finding 5): every shard's executed frame-average
    # partition re-derived from its durable ledger, and their sum, before any write
    fa = fa_partition_census(shards, base.get("fa_partition"))

    # (3) disjoint declared rooms whose union is the registered set
    declared, overlap = [], set()
    for shard in shards:
        if not shard["declared_rooms"]:
            raise ValueError(f"{shard['dir']} declares no rooms; a merge needs each shard's "
                             "scope to prove the union is complete")
        overlap |= set(declared) & set(shard["declared_rooms"])
        declared.extend(shard["declared_rooms"])
    if overlap:
        raise ValueError(f"the shards' room sets are not disjoint ({sorted(overlap)}); the "
                         "same room scored twice would double a query")
    expected_rooms = sorted(plan.rooms if expected_rooms is None else expected_rooms)
    missing = sorted(set(expected_rooms) - set(declared))
    extra = sorted(set(declared) - set(expected_rooms))
    if missing or extra:
        raise ValueError(f"the shards do not cover the registered room set: missing {missing}, "
                         f"unexpected {extra}")
    if totals.get("rooms") is not None and len(expected_rooms) != int(totals["rooms"]):
        raise ValueError(f"the audit publishes {len(expected_rooms)} rooms but the census "
                         f"expects {totals['rooms']}")

    # (4) exactly the registered query identities and positions, once each
    expected = {record["query_id"]: (int(record["position"]), record["room_id"])
                for record in records}
    if totals.get("queries") is not None and len(expected) != int(totals["queries"]):
        raise ValueError(f"the context manifest carries {len(expected)} queries but the census "
                         f"expects {totals['queries']}")
    owner, rows = {}, {}
    for shard in shards:
        for row in read_rows(shard["dir"]):
            query_id = row["query_id"]
            if query_id in rows:
                raise ValueError(f"{query_id} is published by both {owner[query_id]} and "
                                 f"{shard['dir']}; a merged run scores each query once")
            if query_id not in expected:
                raise ValueError(f"{query_id} is not in the registered subset but was scored "
                                 f"by {shard['dir']}")
            if row["room_id"] not in shard["declared_rooms"]:
                raise ValueError(f"{shard['dir']} scored {query_id} from {row['room_id']}, "
                                 f"which it does not declare")
            position, room_id = expected[query_id]
            if int(row["position"]) != position or row["room_id"] != room_id:
                raise ValueError(f"{query_id} is published at position {row['position']} in "
                                 f"{row['room_id']} but the context manifest registers "
                                 f"position {position} in {room_id}")
            rows[query_id], owner[query_id] = row, shard["dir"]
    absent = sorted(set(expected) - set(rows))
    if absent:
        raise ValueError(f"{len(absent)} registered queries are missing from the shards "
                         f"(first {absent[:3]}); a merged run is the complete subset")

    # (4b) exp_22 R2-A launch provenance (Codex B4): an R2-generation binding is
    # produced only by the record-carrying driver, so every row must resolve to a
    # present, hash-verified, environment-verified segment record. The legacy R1
    # generation (the published Aug-2026 P1 comparator) predates launch records.
    from src.localization import meshgrid_segments as ms

    recorded = binding_fields(base) != RUN_BINDING_FIELDS
    rows_by_shard = {shard["dir"]: [row for query_id, row in rows.items()
                                    if owner[query_id] == shard["dir"]] for shard in shards}
    if recorded:
        ms.merge_ledgers(rows_by_shard, None, dry_run=True)

    # (5) every digest, every row's identity against the G1 plan -- and the
    # source-row census DERIVED from that plan, so a shard that was resumed
    # after finishing receiver groups still merges (r8 review, gate 7)
    derived_source_rows = 0
    for room_id in expected_rooms:
        room = load_room_plan(plan, room_id)
        derived_source_rows += sum(len(group.union) for group in receiver_groups(room))
        for query in room.queries:
            shard_dir = owner[query.query_id]
            assert_published_matches(shard_dir, query, binding_sha256=binding_sha256)
            row = rows[query.query_id]
            if row.get("batching") and row["batching"] != {
                    key: advisory.get(key) for key in row["batching"]}:
                raise ValueError(
                    f"{query.query_id} was produced with batching {row['batching']} but its "
                    f"shard pins {advisory}; the advisory values are not pinned across the run")

    # (6) the three registered totals
    pairs = sum(int(row["n_candidates"]) for row in rows.values())
    waveforms = sum(int(row["n_candidates"]) * int(row["num_samples"])
                    for row in rows.values())
    # the observed count is whatever the FINAL invocation of each shard reported,
    # which a restart legitimately reduces; the census is the derived one
    observed_source_rows = sum(int(shard["summary"].get("n_conditioner_rows", 0))
                               for shard in shards)
    source_rows = derived_source_rows
    for name, found, wanted in (("candidate_query_pairs", pairs,
                                 totals["candidate_query_pairs"]),
                                ("generated_waveforms", waveforms,
                                 totals["generated_waveforms"]),
                                ("source_rows", source_rows, totals["source_rows"])):
        if int(found) != int(wanted):
            raise ValueError(f"the merged census fails on {name}: the shards account for "
                             f"{found:,} but the registered total is {int(wanted):,}")

    if fa is not None:        # the merged ledger binds the very rows it is published with
        unbound = _row_binding_reasons(fa["ledger"]["segments"], [
            read_fa_stamp(owner[query_id], row) for query_id, row in rows.items()], True,
            fa["blocks"])
        if unbound:
            raise ValueError("the merged rows do not bind the merged ledger: " + "; ".join(unbound))

    # only now does anything reach disk
    os.makedirs(out_dir, exist_ok=True)
    for query_id, row in rows.items():
        shard_dir = owner[query_id]
        paths = query_artifact_paths(shard_dir, row["room_id"], row["position"])
        target = query_artifact_paths(out_dir, row["room_id"], row["position"])
        os.makedirs(target["dir"], exist_ok=True)
        shutil.copy2(paths["row"], target["row"])
        shutil.copy2(paths["sims"], target["sims"])
        if fa is not None:                    # each fa row's ledger stamp travels with it
            stamp = fa_stamp_path(out_dir, row["room_id"], row["position"])
            os.makedirs(os.path.dirname(stamp), exist_ok=True)
            shutil.copy2(fa_stamp_path(shard_dir, row["room_id"], row["position"]), stamp)
        if row.get("waveform_path"):
            source = os.path.join(shard_dir, row["waveform_path"])
            destination = os.path.join(out_dir, row["waveform_path"])
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copy2(source, destination)

    write_binding(out_dir, base, advisory=advisory, declared_rooms=sorted(declared))
    if fa is not None:
        write_json(os.path.join(out_dir, FA_LEDGER_FILENAME), fa["ledger"])
    provenance = (ms.merge_ledgers(rows_by_shard, out_dir) if recorded else
                  {"generation": "legacy_r1_binding", "complete": False, "records": [],
                   "note": "the shards carry the legacy R1 binding, which predates launch "
                           "records; only the published P1 comparator may be canonical "
                           "without them (plan r2 §1 legacy-provenance exception)"})
    report = {"experiment": "exp_22 loc_meshgrid I1 shard merge", "ok": True,
              "launch_provenance": provenance,
              "fa_partition_executed": None if fa is None else fa["receipt"],
              "binding_sha256": binding_sha256, "advisory": advisory,
              "declared_rooms": sorted(declared), "n_rows": len(rows),
              "shards": [{"dir": shard["dir"], "rooms": shard["declared_rooms"],
                          "n_scored": shard["summary"].get("n_scored"),
                          "n_conditioner_rows": shard["summary"].get("n_conditioner_rows"),
                          "advisory_history": shard["binding"].get("advisory_history") or []}
                         for shard in shards],
              "totals": {"candidate_query_pairs": pairs, "source_rows": source_rows,
                         "generated_waveforms": waveforms},
              "source_rows_derived_from": "g1_plan",
              "source_rows_observed": observed_source_rows,
              "source_rows_note": "the census counts one conditioner row per (receiver, "
                                  "candidate) in each receiver's union, derived from the G1 "
                                  "plan; the observed number is what the shards' final "
                                  "invocations reported and is lower whenever a shard was "
                                  "resumed after completing receiver groups",
              "registered_totals": totals,
              "agree_leakage_caveat": AGREE_LEAKAGE_CAVEAT,
              "batching_caveat": BATCHING_CAVEAT}
    write_json(os.path.join(out_dir, MERGE_REPORT_FILENAME), report)
    return report
