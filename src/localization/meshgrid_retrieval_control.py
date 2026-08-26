"""exp_22 r9b -- the sparse/metadata-bank AGREE retrieval control (§2).

The inherited plan §2 registers four controls under the identical candidate
manifest. r9 published two of them (the off-grid truth probe and the
real-vs-generated calibration, ``meshgrid_offgrid_probe.py``) and named the
third as outstanding. This module is that third one:

    "AGREE oracle retrieval using real candidate-bank RIRs only where an exact
    dataset RIR exists, labelled sparse/metadata-bank and not confused with the
    dense-grid model oracle."

**What it answers.** For each of the registered 5,337 queries: where would pure
AGREE nearest-neighbour retrieval place the source if it could only answer with
REAL dataset RIRs that actually exist for this room? The bank of one query is
the room's real RIRs AT THE QUERY'S OWN RECEIVER from OTHER sources -- the same
receiver, so the acoustic difference between two entries is a difference in
SOURCE POSITION and nothing else -- with the query's own observation excluded
(:data:`SELF_PAIR_RULE`). Each entry is embedded with the same frozen,
deterministic AGREE readout and the same preprocessing the engine used
(``agree_embed``: clamp to [-1, 1] -> first 8,000 samples -> pad to 10,240),
scored by ``cos(E(h_obs), E(h_real))``, and the argmax entry's source position
is the prediction.

**What it is NOT.** Three facts have to travel with every number:

* the candidate set is the SPARSE metadata bank -- at most nine real positions
  per query -- not the dense half-metre mesh-valid grid the engine searched;
* its oracle floor is therefore the SPARSE BANK's own
  (:data:`SPARSE_ORACLE_LABEL`), and an oracle-normalized success here is
  measured against a much coarser denominator than the engine's;
* the scorer is ``AGREE_fullAR`` (:data:`AGREE_LEAKAGE_CAVEAT`, copied verbatim
  from the engine), so absolute levels are not leak-free.

**What it generates.** Nothing. There is no FLAC forward pass anywhere in this
control: it reads real RIRs and embeds them. That is why its binding gate checks
the fields that decide ITS numbers -- the scorer, its readout, the D1 context
manifest, the G1 audit and the dataset config -- and records, rather than
requires, the generation-only fields (:data:`BINDING_SCOPE_NOTE`).

Everything that decides a number is still gated before it is computed: the
published run binding must hash to its own content and agree with this control's
inputs, every query's context draw must re-verify against the frozen D1 manifest
(``meshgrid_engine.verify_context_record``), the pair metadata's receiver must be
the G1 manifest's receiver, the dense-grid oracle re-derived from that room's
candidate block must equal the one G1 published -- which is what pins the
continuous truth this control measures its own errors against
(:func:`assert_grid_oracle`) -- and the result set must be exactly the registered
census. Ground truth is resolved post hoc from the dataset's own pair metadata
through the SAME seam the r9 report uses
(``meshgrid_report.TruthResolver``) -- there is no second resolver -- and the
loader item stays wrapped in ``GuardedMetadata``, so this control cannot read
``md['source']`` either.
"""
import argparse
import json
import os

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import torch

from src.localization import meshgrid_engine as me
from src.localization import meshgrid_queries as mq
from src.localization import meshgrid_report as mr
from src.localization import scoring as sc
from src.localization.candidates import parse_ir_filename
# ONE implementation of the numeric-identity pair listing (and of its ambiguity
# refusal). It is a sibling module of the same package; a second copy of the
# S008_R0089-vs-S008_R089 convention is exactly what this import avoids.
from src.localization.candidates import _pair_files as pair_index

#: the key the R1 report's ``controls_elsewhere`` names this control under. The
#: two must agree, or the report would point at nothing.
CONTROL_KEY = "agree_oracle_retrieval_over_the_metadata_bank"

CONTROL_LABEL = (
    "SPARSE / METADATA-BANK AGREE RETRIEVAL CONTROL -- the prediction is chosen from the real "
    "dataset RIRs that actually exist for this room at THIS query's receiver, from other "
    "sources, with the query's own observation excluded. Its candidate set is the sparse "
    "metadata bank (at most nine real source positions per query), NOT the dense half-metre "
    "mesh-valid grid the engine searched, and its oracle floor is the sparse bank's own -- "
    "never the dense-grid model oracle. A number from this control may not be placed in a "
    "table beside a dense-grid number without both labels")

SPARSE_ORACLE_LABEL = (
    "SPARSE-BANK ORACLE -- min over the query's REAL bank entries of ||src_loc - x*_s||, i.e. "
    "the best a retrieval restricted to existing dataset RIRs at this receiver could possibly "
    "do. It is a different and far coarser denominator than the dense-grid oracle the R1 "
    "report normalizes by, and the two are never interchangeable")

SELF_PAIR_RULE = (
    "the query's own observation is excluded from its own bank: the (source, receiver) pair "
    "the query IS can never be retrieved, so a cosine of 1.0 against itself is impossible by "
    "construction. Every other source at the same receiver is eligible")

#: what the bank shares with the model's own conditioning -- said out loud,
#: because it decides how the comparison may be read.
CONTEXT_OVERLAP_NOTE = (
    "the sparse bank OVERLAPS the query's own D1 conditioning contexts by construction: the "
    "released selector draws those eight context RIRs from the same same-receiver/other-source "
    "pool this bank is built from. That is deliberate -- retrieval and the model then answer "
    "from the same real evidence -- but it means the control is NOT independent of the model's "
    "conditioning, and a retrieval hit may be a hit on an RIR the model was itself conditioned "
    "on. It is a comparison of what is DONE with that evidence, not of who had more of it")

#: How the bank is enumerated.
#:
#: ``numeric_identity`` (registered) matches files by their PARSED integer node
#: ids over the directory listing -- the convention ``candidates.py`` fixes for
#: the whole localization stack, because the wav namespace
#: (``S008_R089_hybrid_IR.wav``) and the metadata namespace (``S008_R0089.json``)
#: pad differently.
#:
#: ``released_eligible_pool`` is the pool the RELEASED context selector draws
#: from (``meshgrid_queries.eligible_context_pool``, mirroring
#: ``AR_md.get_ir_and_location_for_other_sources``). It is kept reachable because
#: it is what the model's conditioning could see, and it is NOT the default,
#: because the released renderer builds names as ``f"S00{node}"`` and therefore
#: never finds source node 10.
BANK_RULES = ("numeric_identity", "released_eligible_pool")
REGISTERED_BANK_RULE = "numeric_identity"
BANK_RULE_NOTE = (
    "the registered bank rule is numeric_identity: every real RIR that EXISTS at this "
    "receiver from another source, matched by parsed integer node id. The released context "
    "selector renders candidate names as f\"S00{node}\", so it never finds S010 -- measured "
    "on the registered subset, its eligible pool is exactly one smaller than this bank for "
    "4,593 of the 5,337 queries. The retrieval bank is therefore a SUPERSET of the pool the "
    "model's own conditioning was drawn from, which favours retrieval slightly and is "
    "disclosed rather than removed; --bank-rule released_eligible_pool reproduces the "
    "selector's pool instead")

#: the caveats this control copies verbatim rather than paraphrasing.
AGREE_LEAKAGE_CAVEAT = me.AGREE_LEAKAGE_CAVEAT
SCORER_READOUT_DEVIATION = me.SCORER_READOUT_DEVIATION
SUBSET_LABEL = mr.SUBSET_LABEL

#: the §2 metric family, taken from the R1 report so the two cannot drift.
SUCCESS_RADII = mr.SUCCESS_RADII
BOOTSTRAP_SEED = mr.BOOTSTRAP_SEED
BOOTSTRAP_N = mr.BOOTSTRAP_N
BOOTSTRAP_ALPHA = mr.BOOTSTRAP_ALPHA
RECEIVER_TOLERANCE = mr.RECEIVER_TOLERANCE

#: a bank entry closer than this to the truth would BE the held-out target under
#: another node id (``candidates.SRC_LOC_TOL`` is the same number, for the same
#: reason: two labels at one position are one hypothesis).
TRUTH_COINCIDENCE_TOLERANCE = 1e-6

#: the released IR sample rate ``AR_md`` asserts on every context read.
RELEASED_SAMPLE_RATE = 22050

#: the run-binding fields that decide THIS control's numbers.
RETRIEVAL_BINDING_FIELDS = ("agree_ckpt_sha256", "scorer_readout", "d1_manifest_sha256",
                            "g1_report_sha256", "room_manifest_sha256", "branch",
                            "dataset_config", "dataset_config_sha256")
#: the rest of the run binding: recorded from the published run, never required.
RETRIEVAL_BINDING_NOT_CHECKED = tuple(field for field in me.RUN_BINDING_FIELDS
                                      if field not in RETRIEVAL_BINDING_FIELDS)
BINDING_SCOPE_NOTE = (
    "this control generates nothing -- there is no FLAC forward pass in it -- so the fields "
    "that decide a GENERATION (the checkpoint, the model config, the sampler steps, the CFG "
    "scale, the noise policy and its seed, the sample count and the nested prefixes, the "
    "conditioning method and its autocast, the dump authority) cannot change any number here. "
    "They are recorded from the published run binding and reported, and they do not refuse the "
    "control. tau is in that list too: at K = 1 the registered log-mean-exp aggregate is the "
    "cosine itself, so tau cancels. What IS checked is everything that decides a cosine: the "
    "AGREE checkpoint, its readout, the D1 context manifest (the stream this control walks), "
    "the G1 audit and room manifests (the receivers and the dense-grid oracle it contrasts "
    "against) and the dataset config")

REPORT_JSON = "retrieval_control_report.json"
REPORT_MARKDOWN = "retrieval_control_report.md"
#: the compact artifact the R1 report ingests to fill its ``controls_elsewhere``
#: entry. Named in ``meshgrid_report.CONTROLS_ELSEWHERE[CONTROL_KEY]``.
HANDOFF_JSON = "retrieval_control_handoff.json"


def flat_stat_names(radii=SUCCESS_RADII):
    """The §2 statistic names, exactly as the R1 report publishes them."""
    return mr.flat_stat_names(radii)


def radius_key(radius):
    return mr.radius_key(radius)


# --------------------------------------------------------------------------- #
# the bank: what a query is allowed to retrieve from
# --------------------------------------------------------------------------- #
@dataclass
class BankEntry:
    """One real dataset RIR that may be retrieved for a query.

    ``src_node``/``rec_node`` are the PARSED integer identities (the only stable
    identity across the two padding conventions), ``src_xyz`` is the source
    position the pair metadata declares, and ``rec_xyz`` is the receiver position
    it declares -- kept so the bank can prove every entry really is at the
    query's receiver.
    """
    src_node: int
    rec_node: int
    ir_path: str
    pair_path: str
    src_xyz: np.ndarray
    rec_xyz: np.ndarray

    @property
    def identity(self):
        return (int(self.src_node), int(self.rec_node))


def read_rir(path):
    """One real dataset RIR -> ``[1, 1, T]`` float32, the released read.

    ``torchaudio.load`` plus the release's own two invariants (``AR_md``:
    22,050 Hz, single-channel ``single_channel_ir_*``). No crop and no pad is
    applied on purpose: the scorer's preprocessing
    (``agree_embed.preprocess_for_scoring``) truncates to the first 8,000 samples
    and pads to 10,240, so a raw read and the loader's 10,240-sample PadCrop
    reach the AGREE tower as the SAME tensor. Introducing a second crop
    convention here could only make them differ.
    """
    import torchaudio

    wave, rate = torchaudio.load(str(path))
    if int(rate) != RELEASED_SAMPLE_RATE:
        raise ValueError(f"{path}: IR sampling rate must be {RELEASED_SAMPLE_RATE}, got {rate} "
                         "(the released AR_md context read asserts the same thing)")
    if wave.ndim != 2 or wave.shape[0] != 1:
        raise ValueError(f"{path}: expected a single-channel RIR, got shape {tuple(wave.shape)}")
    if wave.shape[-1] == 0:
        raise ValueError(f"{path}: the RIR is empty")
    wave = wave.float()
    if not bool(torch.isfinite(wave).all()):
        raise ValueError(f"{path}: the RIR carries a non-finite sample")
    return wave.reshape(1, 1, -1)


def ir_index(ir_room_dir):
    """``{(src, rec): path}`` for every IR file in a room's wav directory.

    The mirror of ``candidates._pair_files`` on the wav namespace, matched by
    ``parse_ir_filename`` so the one naming authority is reused, with the same
    fail-closed rule: two files that parse to one identity are unresolvable.
    """
    directory = str(ir_room_dir)
    if not os.path.isdir(directory):
        raise ValueError(f"IR room directory not found: {directory}")
    found = {}
    for name in sorted(os.listdir(directory)):
        try:
            key = parse_ir_filename(name)
        except ValueError:
            continue
        if key in found:
            raise ValueError(f"ambiguous IR files for S{key[0]}_R{key[1]} in {directory}: "
                             f"{os.path.basename(found[key])} and {name}")
        found[key] = os.path.join(directory, name)
    if not found:
        raise ValueError(f"no IR files (S*_R*_hybrid_IR.wav) in {directory}")
    return found


class RoomBank:
    """One room's pair metadata and IR files, listed once and reused.

    A room is walked by up to a thousand queries, so the two directory listings
    and every pair payload are read once. Nothing here is query-specific.
    """

    def __init__(self, room_id, meta_dir, ir_dir):
        self.room_id = str(room_id)
        self.meta_dir = str(meta_dir)
        self.ir_dir = str(ir_dir)
        self.pairs = pair_index(self.meta_dir)
        self.irs = ir_index(self.ir_dir)
        self._payloads = {}
        # grouped ONCE: a room is walked by up to a thousand queries, and each
        # of them asks the same two questions about its own receiver
        self._pairs_at, self._irs_at = {}, {}
        for src, rec in self.pairs:
            self._pairs_at.setdefault(int(rec), []).append(int(src))
        for rec in self._pairs_at:
            self._pairs_at[rec].sort()
        for (src, rec), path in self.irs.items():
            self._irs_at.setdefault(int(rec), {})[int(src)] = path

    def payload(self, src_node, rec_node):
        key = (int(src_node), int(rec_node))
        if key not in self._payloads:
            path = self.pairs[key]
            with open(path) as handle:
                payload = json.load(handle)
            source = np.asarray(payload["src_loc"], dtype=np.float64).reshape(3)
            receiver = np.asarray(payload["rec_loc"], dtype=np.float64).reshape(3)
            if not (np.isfinite(source).all() and np.isfinite(receiver).all()):
                raise ValueError(f"{path} carries a non-finite coordinate")
            self._payloads[key] = (source, receiver, path)
        return self._payloads[key]

    def at_receiver(self, rec_node):
        """Every source node the metadata declares at this receiver, ascending."""
        return list(self._pairs_at.get(int(rec_node), []))

    def irs_at_receiver(self, rec_node):
        """``{src_node: ir_path}`` for the IR files that exist at this receiver."""
        return dict(self._irs_at.get(int(rec_node), {}))


def room_bank(metadata_root, dataset_root, room_id, relpath, cache=None):
    """The room tables for the query at ``relpath``, memoized in ``cache``.

    The IR directory is derived from the D1 record's own ``relpath`` -- the
    identity the manifest pins -- rather than from a second room-to-directory
    convention.
    """
    scene, scene_id = str(room_id).split("/")
    meta_dir = os.path.join(str(metadata_root), scene, scene_id)
    ir_dir = os.path.dirname(os.path.join(str(dataset_root), str(relpath)))
    key = (str(room_id), meta_dir, ir_dir)
    if cache is None:
        return RoomBank(room_id, meta_dir, ir_dir)
    if key not in cache:
        cache[key] = RoomBank(room_id, meta_dir, ir_dir)
    return cache[key]


def _released_pool_identities(dataset_root, relpath):
    """The released selector's own pool, as parsed identities.

    Driven through ``meshgrid_queries.eligible_context_pool``, which mirrors the
    released ``AR_md.get_ir_and_location_for_other_sources`` -- including the
    ``f"S00{node}"`` rendering that never finds node 10.
    """
    pool = mq.eligible_context_pool(os.path.join(str(dataset_root), str(relpath)))
    return {parse_ir_filename(os.path.basename(path)) for path in pool}


def build_query_bank(metadata_root, dataset_root, room_id, relpath,
                     rule=REGISTERED_BANK_RULE, cache=None):
    """The sparse bank of one query: same receiver, other sources, real files only.

    Fail-closed, in order: the query's own pair metadata must exist (otherwise
    the receiver identity is unknown), every IR file at that receiver must have
    pair metadata (otherwise its source position is unknown), every entry's
    ``rec_loc`` must be the query's own ``rec_loc``, and the bank must be
    non-empty -- with the three counts that made it empty named in the message.

    Returns ``{"entries", "counts", "missing_ir", "rule", "receiver_xyz", ...}``
    with ``entries`` in ascending parsed numeric identity.
    """
    if rule not in BANK_RULES:
        raise ValueError(f"unknown bank rule {rule!r} (expected one of {list(BANK_RULES)})")
    src_node, rec_node = parse_ir_filename(os.path.basename(str(relpath)))
    tables = room_bank(metadata_root, dataset_root, room_id, relpath, cache=cache)

    if (src_node, rec_node) not in tables.pairs:
        raise ValueError(f"{room_id}: the query S{src_node:03d}_R{rec_node:03d} has no pair "
                         f"metadata of its own under {tables.meta_dir!r}; without it the "
                         "receiver the bank is built at is unknown")
    _query_source, receiver_xyz, _path = tables.payload(src_node, rec_node)

    at_receiver = tables.at_receiver(rec_node)
    irs_here = tables.irs_at_receiver(rec_node)
    # an IR file whose position nothing declares cannot be a candidate
    undeclared = sorted(set(irs_here) - set(at_receiver))
    if undeclared:
        raise ValueError(f"{room_id}: "
                         f"{[os.path.basename(irs_here[src]) for src in undeclared[:3]]} have "
                         f"no pair metadata under {tables.meta_dir!r}, so the source position "
                         "they would be retrieved as is unknown; the bank refuses rather than "
                         "guesses")

    allowed = None
    if rule == "released_eligible_pool":
        allowed = _released_pool_identities(dataset_root, relpath)

    entries, missing, n_self, n_rule_dropped = [], [], 0, 0
    for candidate_src in at_receiver:
        identity = (int(candidate_src), int(rec_node))
        if int(candidate_src) == int(src_node):
            n_self += 1
            continue
        if identity not in tables.irs:
            missing.append([int(candidate_src), int(rec_node)])
            continue
        if allowed is not None and identity not in allowed:
            n_rule_dropped += 1
            continue
        source, entry_receiver, pair_path = tables.payload(candidate_src, rec_node)
        drift = float(np.abs(entry_receiver - receiver_xyz).max())
        if drift > RECEIVER_TOLERANCE:
            raise ValueError(
                f"{room_id}: S{candidate_src:03d}_R{rec_node:03d} and the query's own "
                f"S{src_node:03d}_R{rec_node:03d} disagree about the receiver by {drift:.6g} m "
                f"({entry_receiver.tolist()} vs {receiver_xyz.tolist()}); a bank entry recorded "
                "at another receiver is not the same listening position")
        entries.append(BankEntry(src_node=int(candidate_src), rec_node=int(rec_node),
                                 ir_path=tables.irs[identity], pair_path=pair_path,
                                 src_xyz=source, rec_xyz=entry_receiver))

    counts = {"n_pairs_at_receiver": len(at_receiver), "n_self_excluded": int(n_self),
              "n_missing_ir": len(missing), "n_rule_dropped": int(n_rule_dropped),
              "n_bank": len(entries),
              "n_released_eligible": mq.eligible_pool_size(
                  os.path.join(str(dataset_root), str(relpath)))}
    if not entries:
        raise ValueError(
            f"{room_id} R{rec_node:03d}: the sparse metadata bank of "
            f"S{src_node:03d}_R{rec_node:03d} is EMPTY "
            f"(n_pairs_at_receiver={counts['n_pairs_at_receiver']}, "
            f"n_self_excluded={counts['n_self_excluded']}, "
            f"n_missing_ir={counts['n_missing_ir']}, "
            f"n_rule_dropped={counts['n_rule_dropped']}, rule={rule!r}). A retrieval control "
            "cannot answer a query with no real RIR to retrieve, and a silently skipped query "
            "would break the registered census")
    assert_bank_order(entries)
    return {"entries": entries, "counts": counts, "missing_ir": missing, "rule": str(rule),
            "receiver_xyz": receiver_xyz, "query_identity": [int(src_node), int(rec_node)],
            "self_pair_rule": SELF_PAIR_RULE}


def assert_bank_order(entries):
    """The bank is ordered by parsed numeric identity -- the tie-break's authority."""
    identities = [entry.identity for entry in entries]
    if identities != sorted(identities):
        raise ValueError(f"the bank must be in ascending numeric identity order so the argmax "
                         f"tie-break is deterministic; got {identities[:4]}")
    if len(set(identities)) != len(identities):
        raise ValueError(f"the bank names one (source, receiver) twice: {identities[:4]}")
    return True


def assert_bank_excludes_the_target(entries, truth, query_id,
                                    tolerance=TRUTH_COINCIDENCE_TOLERANCE):
    """No bank entry may sit ON the held-out source position.

    The query's own pair is already excluded by identity; this catches the other
    way in: a DIFFERENT node declared at the same position would hand the control
    the target's own acoustics under another name. Measured over the sixteen
    included rooms, no room has two sources at one position, so this refuses
    rather than merges (``candidates.enumerate_metadata_sources`` refuses the
    same thing for the same reason).
    """
    truth = np.asarray(truth, dtype=np.float64).reshape(3)
    for entry in entries:
        distance = float(np.linalg.norm(entry.src_xyz - truth))
        if distance <= float(tolerance):
            raise ValueError(
                f"{query_id}: bank entry S{entry.src_node:03d}_R{entry.rec_node:03d} sits on "
                f"the held-out target ({distance:.3g} m <= {float(tolerance):g} m). A second "
                "source label at the target's position would let the control retrieve the "
                "target's own acoustics under another id")
    return True


# --------------------------------------------------------------------------- #
# the score and the prediction
# --------------------------------------------------------------------------- #
def embed_bank(embedder, entries, reader=read_rir, cache=None):
    """``[M, D]`` embeddings of the bank's real RIRs, one read per file.

    ``cache`` is keyed by the IR path, so a receiver's bank is embedded once and
    reused by every query of that receiver -- the same memoization argument the
    engine makes for its source cache, on a much smaller object.
    """
    rows = []
    for entry in entries:
        path = str(entry.ir_path)
        if cache is not None and path in cache:
            rows.append(cache[path])
            continue
        embedding = torch.as_tensor(embedder(reader(path))).float().reshape(-1)
        if cache is not None:
            cache[path] = embedding
        rows.append(embedding)
    return torch.stack(rows)


def bank_sims(obs_embedding, bank_embeddings):
    """``cos(E(h_obs), E(h_real))`` for every bank entry -> ``[M]``.

    Goes through ``scoring.cosine_sims`` -- the engine's own cosine, including
    its L2-normalization refusal -- with the bank shaped as ``[M, 1, D]``,
    because a real RIR is one sample, not K draws.
    """
    obs = torch.as_tensor(obs_embedding).float().reshape(-1)
    bank = torch.as_tensor(bank_embeddings).float()
    if bank.ndim != 2:
        raise ValueError(f"bank embeddings must be [M, D], got {tuple(bank.shape)}")
    return sc.cosine_sims(obs, bank.reshape(bank.shape[0], 1, bank.shape[1]))[:, 0]


def bank_scores(sims, tau=me.TAU):
    """The registered aggregate of a one-sample bank -> ``[M]``.

    Read through ``meshgrid_engine.nested_scores`` at K = 1, so the control uses
    the engine's aggregator rather than a second scoring rule. At K = 1 the
    plan's ``S = tau * (logsumexp(s / tau) - log K)`` is the cosine itself, which
    is why tau decides nothing here.
    """
    sims = torch.as_tensor(sims).float().reshape(-1)
    return me.nested_scores(sims.reshape(-1, 1), tau=tau, prefixes=(1,))[1]["scores"]


def predict_row(scores, entries):
    """The retrieved row: highest score, ties broken by numeric identity.

    Delegated to ``meshgrid_engine.argmax_by_global_index`` -- one argmax rule in
    exp_22 -- with the bank's ROW ORDER as the global index. The bank is sorted
    by ``(src_node, rec_node)`` (``assert_bank_order``), so the tie goes to the
    lexicographically smallest parsed identity. Ordering on the parsed identity
    rather than on the file NAME is deliberate: as strings, ``"S0010"`` sorts
    before ``"S003"``, so a name order would depend on the padding convention.
    """
    assert_bank_order(entries)
    return me.argmax_by_global_index(torch.as_tensor(scores).float().reshape(-1),
                                     list(range(len(entries))))


def evaluate_bank_query(entries, sims, truth, *, query_id, room_id, position, receiver_id,
                        receiver_xyz, tau=me.TAU, radii=SUCCESS_RADII, bank=None,
                        grid=None, bank_rule=REGISTERED_BANK_RULE):
    """Every §2 readout for one query, against the SPARSE bank."""
    truth = np.asarray(truth, dtype=np.float64).reshape(3)
    sims = torch.as_tensor(sims).float().reshape(-1)
    if sims.numel() != len(entries):
        raise ValueError(f"{query_id}: {sims.numel()} similarities for {len(entries)} bank "
                         "entries")
    assert_bank_excludes_the_target(entries, truth, query_id)
    scores = bank_scores(sims, tau=tau)
    row = predict_row(scores, entries)

    positions = np.stack([np.asarray(entry.src_xyz, dtype=np.float64) for entry in entries])
    distances = np.linalg.norm(positions - truth.reshape(1, 3), axis=1)
    oracle_row = int(distances.argmin())
    e_loc = float(distances[row])
    e_oracle = float(distances[oracle_row])
    e_excess = max(0.0, e_loc - e_oracle)

    counts = dict((bank or {}).get("counts") or {})
    out = {
        "control_label": CONTROL_LABEL,
        "bank_rule": str(bank_rule),
        "query_id": str(query_id), "room_id": str(room_id), "position": int(position),
        "receiver_id": receiver_id,
        "receiver_xyz": [float(v) for v in np.asarray(receiver_xyz, dtype=np.float64)],
        "truth_xyz": truth.tolist(),
        # the census reads these two names; a bank entry is one real RIR, so a
        # "candidate" here is a bank entry and there is exactly one sample of it
        "n_candidates": len(entries), "num_samples": 1,
        "bank_identities": [[entry.src_node, entry.rec_node] for entry in entries],
        "bank_counts": counts,
        "missing_ir": [list(pair) for pair in (bank or {}).get("missing_ir") or []],
        "sims": [float(v) for v in sims],
        "scores": [float(v) for v in scores],
        "prediction_row": int(row),
        "prediction_src_node": int(entries[row].src_node),
        "prediction_rec_node": int(entries[row].rec_node),
        "prediction_xyz": positions[row].tolist(),
        "prediction_ir": os.path.basename(str(entries[row].ir_path)),
        "best_score": float(scores[row]),
        "margin": me.top1_margin(scores),
        "e_loc": e_loc,
        "e_oracle_sparse": e_oracle,
        "oracle_src_node": int(entries[oracle_row].src_node),
        "oracle_xyz": positions[oracle_row].tolist(),
        "e_excess": e_excess,
        "success_raw": {radius_key(r): float(e_loc <= float(r)) for r in radii},
        "success_oracle_normalized": {radius_key(r): float(e_excess <= float(r)) for r in radii},
    }
    if grid is not None:
        # recorded as a CONTRAST only: a different candidate set with a different
        # oracle, never this control's denominator
        out["e_oracle_grid"] = float(grid["e_oracle_grid"])
        out["n_grid_candidates"] = int(grid["n_grid_candidates"])
    return out


# --------------------------------------------------------------------------- #
# walking the registered stream
# --------------------------------------------------------------------------- #
def query_index(room_plan):
    """``{query_id: QueryPlan}`` for one room, refusing a duplicated identity.

    A G1 manifest that named one query twice would silently decide which
    receiver and which oracle the control checked against, so the duplicate is a
    refusal rather than a last-one-wins.
    """
    index = {}
    for query in room_plan.queries:
        if query.query_id in index:
            raise ValueError(f"{room_plan.room_id}: the candidate manifest names "
                             f"{query.query_id!r} twice; the control cannot tell which entry "
                             "authenticates the query")
        index[query.query_id] = query
    return index


def assert_grid_oracle(query, truth, tolerance=mr.ORACLE_TOLERANCE):
    """The truth this control resolved is the one G1 measured against this grid.

    The continuous truth is not pinned by any artifact -- the engine is
    structurally unable to read it and G1 publishes only the oracle DISTANCE --
    so it is authenticated the same way the r9 report and the off-grid probe
    authenticate it: re-derive ``min_c ||c - x*_s||`` from the room's candidate
    block and require it to equal the value the audit published. Be precise about
    the strength of that check: the oracle distance is a scalar and is therefore
    NOT injective, so it catches a truth that moved off the query's own
    neighbourhood, not every possible substitution (Codex r9 review, finding 3).
    Without it, an edited ``src_loc`` would silently move every e_loc here.
    """
    coordinates = np.asarray(query.coordinates, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64).reshape(3)
    derived = float(np.linalg.norm(coordinates - truth.reshape(1, 3), axis=1).min())
    delta = abs(derived - float(query.oracle))
    if delta > float(tolerance):
        raise ValueError(
            f"{query.query_id}: the dense-grid oracle re-derived from the G1 candidate block "
            f"and the pair metadata is {derived:.9f} m but the manifest published "
            f"{float(query.oracle):.9f} m (|delta| = {delta:.3g} > {float(tolerance):g}); the "
            "control is not looking at the same query the audit measured")
    return delta


def run_retrieval(embedder, stream, records, plan, *, metadata_root, dataset_root,
                  bank_rule=REGISTERED_BANK_RULE, tau=me.TAU, radii=SUCCESS_RADII,
                  reader=read_rir, on_record=None):
    """Walk the registered stream and score every query against its sparse bank.

    The stream is the released loader in D1 order and is walked ONCE, exactly as
    the scored pass walks it, because every query's context draw depends on the
    complete pass -- and because verifying that draw
    (``meshgrid_engine.verify_context_record``) is what proves this control is
    looking at the registered stream rather than a re-materialized one.

    The G1 plan is read ROOM BY ROOM as the stream reaches each room (the D1
    stream is room-contiguous, which ``assert_room_blocks`` asserts rather than
    assumes), so the largest room's 137 MB of index lists is held once and
    released -- and every query is authenticated against its own room's
    candidate block: its receiver, and the dense-grid oracle that pins the truth
    this control resolved.
    """
    if bank_rule not in BANK_RULES:
        raise ValueError(f"unknown bank rule {bank_rule!r} (expected one of {list(BANK_RULES)})")
    by_position = {int(record["position"]): record for record in records}
    me.assert_room_blocks(records)
    resolver = mr.TruthResolver(metadata_root)
    rooms, embeddings, results, seen = {}, {}, [], set()
    current_room, current_index, finished_rooms, oracle_deltas = None, {}, set(), []

    for position, (obs_wav, raw_md) in enumerate(stream):
        record = by_position.get(position)
        if record is None:
            continue
        md = me.GuardedMetadata(raw_md)
        me.verify_context_record(md, record, position)
        query_id = str(record["query_id"])
        room_id = str(record["room_id"])
        if query_id in seen:
            raise ValueError(f"{query_id} arrives twice in the stream; a control scores each "
                             "query once")
        if room_id != current_room:
            if room_id in finished_rooms:
                raise ValueError(f"room {room_id!r} reappears in the stream after it was left; "
                                 "the D1 stream is room-contiguous and the control walks it once")
            if current_room is not None:
                finished_rooms.add(current_room)
            if room_id not in plan.rooms:
                raise ValueError(f"the G1 audit publishes no candidate manifest for room "
                                 f"{room_id!r}; the control joins one audit to one manifest")
            current_room = room_id
            current_index = query_index(me.load_room_plan(plan, room_id))
        if query_id not in current_index:
            raise ValueError(f"{query_id} is in the D1 manifest but not in the G1 audit's "
                             f"queries for {room_id!r}; the control joins one audit to one "
                             "manifest")
        query = current_index[query_id]
        if obs_wav is None:
            raise ValueError(f"stream position {position}: the loader returned no observed "
                             "waveform; there is nothing to retrieve against")

        metadata_receiver, truth = resolver.resolve(record)
        mr.assert_receiver_matches(query_id, metadata_receiver, query.receiver_xyz)
        oracle_deltas.append(assert_grid_oracle(query, truth))
        grid = {"e_oracle_grid": float(query.oracle),
                "n_grid_candidates": int(query.n_candidates)}
        bank = build_query_bank(metadata_root, dataset_root, room_id,
                                str(record["relpath"]), rule=bank_rule, cache=rooms)
        # the truth resolver and the bank read the same pair file through two
        # different listings (find_pair_metadata / pair_index); this asserts they
        # agree, so the bank is built at the receiver the truth was resolved at
        mr.assert_receiver_matches(query_id, bank["receiver_xyz"], query.receiver_xyz)

        obs_embedding = torch.as_tensor(embedder(torch.as_tensor(obs_wav)))[0].float()
        sims = bank_sims(obs_embedding,
                         embed_bank(embedder, bank["entries"], reader=reader,
                                    cache=embeddings))
        result = evaluate_bank_query(
            bank["entries"], sims, truth, query_id=query_id, room_id=room_id,
            position=position, receiver_id=query.receiver_id,
            receiver_xyz=query.receiver_xyz, tau=tau, radii=radii, bank=bank, grid=grid,
            bank_rule=bank_rule)
        result["e_oracle_grid_delta"] = float(oracle_deltas[-1])
        results.append(result)
        seen.add(query_id)
        if on_record is not None:
            on_record(result)

    absent = sorted(str(record["query_id"]) for record in records
                    if str(record["query_id"]) not in seen)
    if absent:
        raise ValueError(f"the stream ended before {len(absent)} registered queries were "
                         f"reached (first {absent[:3]}); a partial control may not be published")
    results.sort(key=lambda result: int(result["position"]))
    return results


# --------------------------------------------------------------------------- #
# the census and the aggregation
# --------------------------------------------------------------------------- #
def retrieval_totals(rooms=None, queries=None):
    """The census this control is held to.

    The registered room and query counts are the run's; the candidate-query-pair
    and generated-waveform totals are deliberately ``None``, because this control
    generates nothing and its "candidates" are real files whose count is a
    measured property of the dataset -- published as the bank-size distribution
    instead of pinned as a total.
    """
    return {"rooms": int(me.REGISTERED_TOTALS["rooms"] if rooms is None else rooms),
            "queries": int(me.REGISTERED_TOTALS["queries"] if queries is None else queries),
            "candidate_query_pairs": None, "generated_waveforms": None}


def assert_retrieval_census(results, records, totals=None):
    """The result set IS the registered subset -- the R1 report's own census.

    Reused rather than re-implemented: a result carries ``query_id``, ``room_id``,
    ``position``, ``n_candidates`` and ``num_samples``, which is exactly what
    ``meshgrid_report.assert_census`` reads.
    """
    return mr.assert_census(results, records,
                            totals=retrieval_totals() if totals is None else totals)


def bank_report(results):
    """The bank-size distribution §2's sparse control has to publish."""
    results = list(results)
    if not results:
        raise ValueError("a bank-size distribution needs at least one scored query")
    sizes = np.asarray([int(result["n_candidates"]) for result in results], dtype=np.float64)
    histogram = {}
    for size in sizes:
        histogram[str(int(size))] = histogram.get(str(int(size)), 0) + 1
    by_room = {}
    for result in results:
        by_room.setdefault(str(result["room_id"]), []).append(int(result["n_candidates"]))
    per_room = {}
    for room in sorted(by_room):
        values = np.asarray(by_room[room], dtype=np.float64)
        per_room[room] = {"n_queries": int(values.size), "min": int(values.min()),
                          "max": int(values.max()), "mean": float(values.mean()),
                          "median": float(np.median(values)), "total": int(values.sum())}
    counts = {}
    for name in ("n_pairs_at_receiver", "n_self_excluded", "n_missing_ir", "n_rule_dropped",
                 "n_released_eligible"):
        counts[name] = int(sum(int((result.get("bank_counts") or {}).get(name, 0))
                               for result in results))
    rules = sorted({str(result["bank_rule"]) for result in results})
    if len(rules) != 1:
        raise ValueError(f"the results were scored under more than one bank rule ({rules}); "
                         "one distribution cannot describe two banks")
    return {
        "rule": rules[0],
        "rule_note": BANK_RULE_NOTE,
        "self_pair_rule": SELF_PAIR_RULE,
        "pooled": {"n_queries": int(sizes.size), "min": int(sizes.min()),
                   "max": int(sizes.max()), "mean": float(sizes.mean()),
                   "median": float(np.median(sizes)), "total": int(sizes.sum())},
        "per_room": per_room,
        "per_query_histogram": histogram,
        "n_missing_ir_total": counts["n_missing_ir"],
        "counts_total": counts,
        "n_released_eligible_total": counts["n_released_eligible"],
        "note": "one bank entry is ONE real dataset RIR (num_samples = 1); the bank size is a "
                "property of the dataset at this receiver, not of any model",
    }


def sparse_oracle_report(results, draws=None, **bootstrap):
    """The sparse-bank oracle distribution, labelled as the bank's own.

    Computed by the R1 report's own ``oracle_report`` over a projection whose
    ``e_oracle`` is the SPARSE oracle, so the arithmetic and the bootstrap are
    literally the same code; the label is what keeps the two apart.
    """
    projected = [{"room_id": result["room_id"], "e_oracle": float(result["e_oracle_sparse"])}
                 for result in results]
    block = mr.oracle_report(projected, draws=draws, **bootstrap)
    block["label"] = SPARSE_ORACLE_LABEL
    block["note"] = ("the sparse-bank oracle: min over the query's REAL bank entries of "
                     "||src_loc - x*_s||. It is NOT the dense-grid oracle the R1 report "
                     "normalizes by")
    return block


def grid_oracle_crosscheck(results):
    """How far the truth-pinning oracle re-derivation sat from what G1 published."""
    deltas = np.asarray([float(result.get("e_oracle_grid_delta", 0.0)) for result in results],
                        dtype=np.float64)
    return {"n_queries": int(deltas.size), "max_abs_delta_m": float(deltas.max()),
            "mean_abs_delta_m": float(deltas.mean()),
            "tolerance_m": float(mr.ORACLE_TOLERANCE),
            "note": "the control re-derives the DENSE-GRID oracle min_c ||c - x*_s|| from the "
                    "G1 candidate block and the pair metadata's src_loc and requires it to "
                    "equal the value the audit published. That is what pins the continuous "
                    "truth this control measures its own errors against; it is a scalar and "
                    "therefore not injective, so it catches a truth that moved, not every "
                    "possible substitution"}


def oracle_contrast(results):
    """Sparse-bank oracle vs the dense-grid oracle, side by side and named."""
    paired = [(float(r["e_oracle_sparse"]), float(r["e_oracle_grid"])) for r in results
              if "e_oracle_grid" in r]
    if not paired:
        return {"n_queries": 0, "note": "no dense-grid oracle was joined to these results"}
    sparse = np.asarray([value for value, _ in paired], dtype=np.float64)
    grid = np.asarray([value for _, value in paired], dtype=np.float64)
    return {
        "n_queries": int(sparse.size),
        "sparse_bank": {"median": float(np.median(sparse)), "mean": float(sparse.mean()),
                        "min": float(sparse.min()), "max": float(sparse.max())},
        "dense_grid": {"median": float(np.median(grid)), "mean": float(grid.mean()),
                       "min": float(grid.min()), "max": float(grid.max())},
        "median_gap_sparse_minus_grid": float(np.median(sparse - grid)),
        "n_sparse_below_grid": int((sparse < grid).sum()),
        "note": "the two oracles are floors of two DIFFERENT candidate sets and are not "
                "comparable as a quality statement: the dense grid is a half-metre lattice of "
                "thousands of points, the sparse bank is at most nine real source positions. "
                "They are printed together only so an oracle-normalized success from this "
                "control is never read against the R1 report's",
    }


def build_report(results, context, *, radii=SUCCESS_RADII, bootstrap_seed=BOOTSTRAP_SEED,
                 n_boot=BOOTSTRAP_N, alpha=BOOTSTRAP_ALPHA, totals=None):
    """The machine-readable control report: every §2 readout, every label.

    ``context`` is the run context mapping -- ``records`` and ``binding_sha256``
    are required, and ``binding``, ``run_dir``, ``audit_report``,
    ``context_manifest``, ``metadata_root``, ``dataset_root``, ``totals`` and
    ``gate`` are recorded as provenance when present.
    """
    if not results:
        raise ValueError("a retrieval control report needs at least one scored query")
    if not context.get("binding_sha256"):
        raise ValueError("the report must name the run binding this control was gated against; "
                         "a control number with no binding cannot be placed beside that run's")
    census = assert_retrieval_census(results, context["records"],
                                     totals=totals or context.get("totals"))
    draws = mr.room_bootstrap_draws(census["n_rooms"], seed=bootstrap_seed, n=n_boot)
    bootstrap = {"seed": bootstrap_seed, "n": n_boot, "alpha": alpha}

    by_room, all_loc, all_excess = {}, [], []
    for result in results:
        bucket = by_room.setdefault(str(result["room_id"]), {"e_loc": [], "e_excess": []})
        bucket["e_loc"].append(float(result["e_loc"]))
        bucket["e_excess"].append(float(result["e_excess"]))
        all_loc.append(float(result["e_loc"]))
        all_excess.append(float(result["e_excess"]))
    per_room = {room: mr.room_block(values, radii=radii) for room, values in by_room.items()}

    bank = bank_report(results)
    binding = dict(context.get("binding") or {})
    return {
        "experiment": "exp_22 loc_meshgrid R1 sparse/metadata-bank AGREE retrieval control",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "control_key": CONTROL_KEY,
        "control_label": CONTROL_LABEL,
        "labels": {
            "control": CONTROL_LABEL,
            "sparse_oracle": SPARSE_ORACLE_LABEL,
            "self_pair_rule": SELF_PAIR_RULE,
            "context_overlap": CONTEXT_OVERLAP_NOTE,
            "bank_rule_note": BANK_RULE_NOTE,
            "subset": SUBSET_LABEL,
            "agree_leakage_caveat": AGREE_LEAKAGE_CAVEAT,
            "scorer_readout_deviation": SCORER_READOUT_DEVIATION,
            "binding_scope": BINDING_SCOPE_NOTE,
        },
        "provenance": {
            "run_dir": context.get("run_dir"),
            "binding_sha256": context.get("binding_sha256"),
            "binding_checked": {field: binding.get(field) for field in RETRIEVAL_BINDING_FIELDS
                                if field in binding},
            "binding_recorded_not_checked": {field: binding.get(field)
                                             for field in RETRIEVAL_BINDING_NOT_CHECKED
                                             if field in binding},
            "audit_report": context.get("audit_report"),
            "context_manifest": context.get("context_manifest"),
            "metadata_root": context.get("metadata_root"),
            "dataset_root": context.get("dataset_root"),
            "agree_ckpt": context.get("agree_ckpt"),
            "device": context.get("device"),
        },
        "protocol": {
            "scorer_readout": me.SCORER_READOUT,
            "preprocessing": "clamp to [-1, 1] -> first 8,000 samples -> pad to 10,240 "
                             "(src/localization/agree_embed.preprocess_for_scoring), the same "
                             "function the engine scored with",
            "score": "cos(E(h_obs), E(h_real)); read through the engine's nested_scores at "
                     "K = 1, where the registered log-mean-exp aggregate IS the cosine",
            "num_samples": 1,
            "generated_waveforms": 0,
            "tie_break": "highest score, ties broken by the smallest parsed (src_node, "
                         "rec_node) identity (meshgrid_engine.argmax_by_global_index over the "
                         "bank's identity order)",
            "bank_rule": bank["rule"],
            # stated, not implied: an artifact may not call a setting
            # pre-registered when it is not (Codex r9 review, finding 6)
            "bank_rule_is_registered": bool(bank["rule"] == REGISTERED_BANK_RULE),
            "success_radii_m": [float(r) for r in radii],
            "aggregation": "room-first, then averaged over rooms (§2)",
            "bootstrap": {"seed": int(bootstrap_seed), "n_boot": int(n_boot),
                          "alpha": float(alpha), "unit": "room",
                          "method": "percentile (linear interpolation)",
                          "is_registered": bool(int(bootstrap_seed) == BOOTSTRAP_SEED
                                                and int(n_boot) == BOOTSTRAP_N),
                          "registered": {"seed": BOOTSTRAP_SEED, "n_boot": BOOTSTRAP_N}},
        },
        "census": census,
        "gates": {
            "note": "every entry below is a gate the control refuses on; a published report is "
                    "proof they passed, not a claim that they did",
            "binding_checked_against_published_run": bool(context.get("gate") is not None),
            "d1_context_draw_reverified_per_query": True,
            "pair_metadata_receiver_matches_g1": True,
            "grid_oracle_rederived_pins_the_truth": True,
            "bank_excludes_the_query_own_pair": True,
            "bank_excludes_any_entry_on_the_target": True,
            "census_is_the_registered_subset": True,
        },
        "metrics": {
            "per_room": per_room,
            "across_rooms": mr.across_rooms(per_room, names=flat_stat_names(radii), draws=draws,
                                            **bootstrap),
            "pooled": mr.pooled_block(all_loc, all_excess, radii=radii),
        },
        "sparse_oracle": sparse_oracle_report(results, draws=draws, **bootstrap),
        "oracle_contrast": oracle_contrast(results),
        "crosscheck": {"grid_oracle": grid_oracle_crosscheck(results)},
        "bank": bank,
        "results": results,
    }


# --------------------------------------------------------------------------- #
# the artifacts
# --------------------------------------------------------------------------- #
def build_handoff(report, report_sha256=None):
    """The compact record the R1 report ingests for its ``controls_elsewhere``."""
    across = report["metrics"]["across_rooms"]
    oracle = report["sparse_oracle"]["across_rooms"]
    return {
        "control_key": CONTROL_KEY,
        "control_label": CONTROL_LABEL,
        "status": "run",
        "created_utc": report["created_utc"],
        "report_json": REPORT_JSON,
        "report_markdown": REPORT_MARKDOWN,
        "report_sha256": report_sha256,
        "binding_sha256": report["provenance"]["binding_sha256"],
        "subset": SUBSET_LABEL,
        "agree_leakage_caveat": AGREE_LEAKAGE_CAVEAT,
        "context_overlap": CONTEXT_OVERLAP_NOTE,
        "census": {"n_queries": report["census"]["n_queries"],
                   "n_rooms": report["census"]["n_rooms"]},
        "headline": {name: {"point": across[name]["point"], "ci_lo": across[name]["ci_lo"],
                            "ci_hi": across[name]["ci_hi"]}
                     for name in flat_stat_names()},
        "bank": {"rule": report["bank"]["rule"],
                 "per_query_histogram": report["bank"]["per_query_histogram"],
                 "pooled": report["bank"]["pooled"],
                 "rule_note": BANK_RULE_NOTE},
        "sparse_oracle": {"label": SPARSE_ORACLE_LABEL,
                          "median_e_oracle": oracle["median_e_oracle"]["point"],
                          "mean_e_oracle": oracle["mean_e_oracle"]["point"]},
        "oracle_contrast": report["oracle_contrast"],
        "not_the_dense_grid": [
            "the candidate set is the sparse metadata bank (real dataset RIRs at this "
            "receiver), not the dense half-metre mesh-valid grid the engine searched",
            "the oracle-normalized success is measured against the SPARSE-BANK oracle, a much "
            "coarser floor than the dense-grid oracle of the R1 report",
        ],
    }


def render_markdown(report):
    """The human-readable summary; the JSON carries everything."""
    provenance, protocol, census = report["provenance"], report["protocol"], report["census"]
    stamp = (f"_binding_ `{provenance['binding_sha256']}` · _subset_ {report['labels']['subset']}"
             f" · _AGREE leakage_ {report['labels']['agree_leakage_caveat']}")
    lines = ["# exp_22 R1 — sparse/metadata-bank AGREE retrieval control", ""]
    lines.append(f"Generated {report['created_utc']}.")
    lines.append("")
    lines.append(f"> **{report['control_label']}**")
    lines.append("")
    lines.append(f"- **Scope:** {report['labels']['subset']}")
    lines.append(f"- **Run binding:** `{provenance['binding_sha256']}`")
    lines.append(f"- **Self-pair rule:** {report['labels']['self_pair_rule']}")
    lines.append(f"- **Overlap with the model's conditioning:** "
                 f"{report['labels']['context_overlap']}")
    lines.append(f"- **Bank rule:** `{protocol['bank_rule']}` — {report['labels']['bank_rule_note']}")
    lines.append(f"- **AGREE leakage caveat:** {report['labels']['agree_leakage_caveat']}")
    lines.append(f"- **Scorer readout deviation:** "
                 f"{report['labels']['scorer_readout_deviation']}")
    lines.append(f"- **Binding scope:** {report['labels']['binding_scope']}")
    lines.append("")
    lines.append(f"Census: {census['n_queries']:,} queries over {census['n_rooms']} rooms; "
                 f"{report['bank']['pooled']['total']:,} real bank entries scored, "
                 f"{protocol['generated_waveforms']} waveforms generated.")
    lines.append("")

    lines.append("## Localization — room-first")
    lines.append("")
    lines.append(stamp)
    lines.append("")
    across = report["metrics"]["across_rooms"]
    lines.append("| statistic | point [95% room bootstrap] |")
    lines.append("|---|---|")
    for name in flat_stat_names():
        entry = across[name]
        lines.append(f"| {name} | {mr.format_number(entry['point'], 4)} "
                     f"[{mr.format_number(entry['ci_lo'], 4)}, "
                     f"{mr.format_number(entry['ci_hi'], 4)}] |")
    lines.append("")
    lines.append(f"Intervals are 95% percentile intervals from "
                 f"{protocol['bootstrap']['n_boot']:,} room resamples at seed "
                 f"{protocol['bootstrap']['seed']}. The oracle-normalized rows are measured "
                 f"against the SPARSE-BANK oracle.")
    if not protocol["bootstrap"]["is_registered"]:
        lines.append("")
        lines.append(f"> **SENSITIVITY CHECK, not the registered bootstrap:** the pre-registered "
                     f"settings are seed {protocol['bootstrap']['registered']['seed']} x "
                     f"{protocol['bootstrap']['registered']['n_boot']:,} resamples.")
    if not protocol["bank_rule_is_registered"]:
        lines.append("")
        lines.append(f"> **SENSITIVITY CHECK, not the registered bank:** the registered rule is "
                     f"`{REGISTERED_BANK_RULE}`.")
    lines.append("")

    lines.append("## Per room")
    lines.append("")
    lines.append(stamp)
    lines.append("")
    lines.append("| room | n | bank min/median/max | median e_loc | success@0.5 | "
                 "oracle-norm@0.5 | median sparse e_oracle |")
    lines.append("|---|---|---|---|---|---|---|")
    banks, oracle_rooms = report["bank"]["per_room"], report["sparse_oracle"]["per_room"]
    for room in sorted(report["metrics"]["per_room"]):
        block, bank, oracle = (report["metrics"]["per_room"][room], banks[room],
                               oracle_rooms[room])
        lines.append(
            f"| {room} | {block['n_queries']:,} | "
            f"{bank['min']}/{mr.format_number(bank['median'], 1)}/{bank['max']} | "
            f"{mr.format_number(block['median_e_loc'], 3)} | "
            f"{mr.format_number(block['success_raw@0.5'], 3)} | "
            f"{mr.format_number(block['success_oracle_normalized@0.5'], 3)} | "
            f"{mr.format_number(oracle['median_e_oracle'], 3)} |")
    lines.append("")

    lines.append("## The sparse-bank oracle")
    lines.append("")
    lines.append(stamp)
    lines.append("")
    lines.append(f"> {report['labels']['sparse_oracle']}")
    lines.append("")
    contrast = report["oracle_contrast"]
    if contrast.get("n_queries"):
        lines.append("| oracle | median | mean | min | max |")
        lines.append("|---|---|---|---|---|")
        for name, key in (("sparse bank (this control)", "sparse_bank"),
                          ("dense grid (R1 report)", "dense_grid")):
            block = contrast[key]
            lines.append(f"| {name} | {mr.format_number(block['median'], 4)} | "
                         f"{mr.format_number(block['mean'], 4)} | "
                         f"{mr.format_number(block['min'], 4)} | "
                         f"{mr.format_number(block['max'], 4)} |")
        lines.append("")
        lines.append(contrast["note"])
        lines.append("")

    lines.append("## Bank sizes")
    lines.append("")
    lines.append(stamp)
    lines.append("")
    pooled = report["bank"]["pooled"]
    lines.append(f"- per query: min {pooled['min']}, median "
                 f"{mr.format_number(pooled['median'], 1)}, max {pooled['max']} "
                 f"({pooled['total']:,} real RIRs scored in total)")
    histogram = report["bank"]["per_query_histogram"]
    lines.append("- size histogram: " + ", ".join(
        f"{size}: {histogram[size]:,}" for size in sorted(histogram, key=int)))
    lines.append(f"- pairs at the receiver with no IR file (dropped): "
                 f"{report['bank']['n_missing_ir_total']:,}")
    lines.append(f"- {report['bank']['note']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_report(out_dir, report):
    """Publish the three artifacts atomically and return their paths + digests."""
    os.makedirs(str(out_dir), exist_ok=True)
    paths = {"json": os.path.join(str(out_dir), REPORT_JSON),
             "markdown": os.path.join(str(out_dir), REPORT_MARKDOWN),
             "handoff": os.path.join(str(out_dir), HANDOFF_JSON)}
    me.write_json(paths["json"], mr.jsonable(report))
    tmp = paths["markdown"] + ".tmp"
    with open(tmp, "w") as handle:
        handle.write(render_markdown(report))
    os.replace(tmp, paths["markdown"])
    # the handoff carries the digest of the report it summarizes, so the R1 side
    # can prove the two describe one run
    me.write_json(paths["handoff"],
                  mr.jsonable(build_handoff(report, report_sha256=me.file_sha256(paths["json"]))))
    return {"paths": paths, "sha256": {name: me.file_sha256(path)
                                       for name, path in paths.items()}}


# --------------------------------------------------------------------------- #
# the binding gate
# --------------------------------------------------------------------------- #
def assert_retrieval_binding(run_dir, binding, fields=RETRIEVAL_BINDING_FIELDS):
    """This control scores under the same scorer and stream the run was scored with.

    The published binding is recomputed from its own content first (a hand-edited
    file cannot vouch for itself), then compared field by field -- but only on
    the fields that decide a cosine here (:data:`BINDING_SCOPE_NOTE`). The
    generation-only fields are returned so the report can record what the run
    was generated under.
    """
    published, published_sha = mr.load_published_binding(run_dir)
    differing = {}
    for field in fields:
        if field not in binding:
            raise ValueError(f"the retrieval control binding is missing the registered field "
                             f"{field!r}; every quantity that decides a cosine must be pinned "
                             "before a control is scored")
        if published.get(field) != binding.get(field):
            differing[field] = {"published": published.get(field), "control": binding.get(field)}
    if differing:
        raise ValueError(
            f"the control does not score under the protocol the published run was scored "
            f"under: {sorted(differing)} differ (published binding {published_sha[:12]}...). "
            f"First mismatch: {sorted(differing)[0]} = "
            f"{differing[sorted(differing)[0]]!r}. A retrieval control scored with a different "
            "AGREE checkpoint, readout, context stream or candidate manifest is not comparable "
            f"to that run. {BINDING_SCOPE_NOTE}")
    return {"binding_sha256": published_sha, "checked": list(fields),
            "published": {field: published[field] for field in fields},
            "recorded_not_checked": {field: published.get(field)
                                     for field in RETRIEVAL_BINDING_NOT_CHECKED},
            "scope_note": BINDING_SCOPE_NOTE}


def build_retrieval_binding(args, plan, agree_sha256):
    """The checked binding fields, built by the DRIVER's own binding builder.

    ``localize_meshgrid.build_run_binding`` is the single definition of what each
    field means; this slices the ones this control is bound to out of it, so a
    change there cannot leave a second copy behind.
    """
    from localize_meshgrid import build_run_binding

    full = build_run_binding(args, plan, ckpt_sha256=None, agree_sha256=agree_sha256,
                             model_config_sha256=None)
    return {field: full[field] for field in RETRIEVAL_BINDING_FIELDS}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", required=True,
                        help="the MERGED I1 run directory this control is reported beside")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--context-manifest",
                        default=os.path.join("outputs_loc", "exp22",
                                             "d1_context_manifest.json"))
    parser.add_argument("--audit-report",
                        default=os.path.join("outputs_loc", "exp22", "g1_audit",
                                             "geometry_audit_report.json"))
    parser.add_argument("--metadata-root", default=os.path.join("AcousticRooms", "metadata"),
                        help="the dataset metadata root the bank positions and the continuous "
                             "truth are read from")
    parser.add_argument("--dataset-root", default="AcousticRooms",
                        help="the dataset root the D1 relpaths resolve against")
    parser.add_argument("--model-config",
                        default=os.path.join("src", "configs", "model_configs", "FLAC", "AR",
                                             "FLAC_AR.json"))
    parser.add_argument("--dataset-config",
                        default=os.path.join("src", "configs", "dataset_configs", "AR", "eval",
                                             "acousticroom_unseeneval.json"))
    parser.add_argument("--agree-ckpt", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--branch", default=None)
    parser.add_argument("--bank-rule", default=REGISTERED_BANK_RULE, choices=list(BANK_RULES))
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--n-boot", type=int, default=BOOTSTRAP_N)
    parser.add_argument("--tau", type=float, default=me.TAU)
    # registered protocol fields the shared build_run_binding reads. They decide
    # nothing here (BINDING_SCOPE_NOTE) and exist so the binding builder is
    # reused unchanged rather than copied.
    parser.add_argument("--seed", type=int, default=me.SEED, help=argparse.SUPPRESS)
    parser.add_argument("--num-samples", type=int, default=me.NUM_SAMPLES,
                        help=argparse.SUPPRESS)
    parser.add_argument("--k-prefixes", type=int, nargs="+", default=list(me.K_PREFIXES),
                        help=argparse.SUPPRESS)
    parser.add_argument("--noise-policy", default=me.NOISE_KEY_POLICY, help=argparse.SUPPRESS)
    parser.add_argument("--steps", type=int, default=me.STEPS, help=argparse.SUPPRESS)
    parser.add_argument("--cfg-scale", type=float, default=me.CFG_SCALE, help=argparse.SUPPRESS)
    parser.add_argument("--cond-method", default="vanilla", help=argparse.SUPPRESS)
    parser.add_argument("--cond-autocast", default="default", help=argparse.SUPPRESS)
    parser.add_argument("--dump-cases-sha256", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _refuse(message):
    raise SystemExit(f"REFUSED: {message}")


def validate_args(args):
    """Startup refusals -- before a checkpoint is read or a device is touched."""
    if args.bank_rule not in BANK_RULES:
        _refuse(f"unknown bank rule {args.bank_rule!r} (expected one of {list(BANK_RULES)})")
    if args.bank_rule != REGISTERED_BANK_RULE:
        print(f"NOTE: --bank-rule {args.bank_rule!r} is not the registered "
              f"{REGISTERED_BANK_RULE!r}; this run is a sensitivity check, not the registered "
              f"control. {BANK_RULE_NOTE}")
    if int(args.n_boot) < 1:
        _refuse("--n-boot must be at least 1")
    if int(args.bootstrap_seed) != BOOTSTRAP_SEED or int(args.n_boot) != BOOTSTRAP_N:
        print(f"NOTE: the bootstrap is being run at seed {args.bootstrap_seed} x "
              f"{args.n_boot} resamples, not the pre-registered {BOOTSTRAP_SEED} x "
              f"{BOOTSTRAP_N}")
    if float(args.tau) <= 0.0:
        _refuse(f"--tau must be > 0, got {args.tau}")
    if os.path.abspath(str(args.out_dir)) == os.path.abspath(str(args.run_dir)):
        _refuse("--out-dir may not be the scored run directory: a control never writes into "
                "the artifact set it reports against")
    return True


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    print(f"{CONTROL_LABEL}\n")
    print(f"SELF-PAIR RULE: {SELF_PAIR_RULE}")
    print(f"BANK RULE: {args.bank_rule} -- {BANK_RULE_NOTE}")
    print(f"AGREE LEAKAGE CAVEAT: {AGREE_LEAKAGE_CAVEAT}")
    print(f"SPARSE ORACLE: {SPARSE_ORACLE_LABEL}\n")

    from localize_meshgrid import _iter_items as iter_stream_items

    with open(args.model_config) as handle:
        model_config = json.load(handle)
    resolved = mq.with_resolved_agree(model_config)
    agree_path = args.agree_ckpt or resolved["training"]["metrics"]["AGREE_ckpt"]

    plan = me.load_audit_plan(args.audit_report, branch=args.branch)
    manifest = mq.load_manifest(args.context_manifest)
    records = manifest["records"]

    # the gate runs on FILE digests, before the scorer is built or a device is
    # touched: nothing is loaded until the control is known to continue this run
    binding = build_retrieval_binding(args, plan, me.file_sha256(agree_path))
    gate = assert_retrieval_binding(args.run_dir, binding)
    print(f"binding gate passed against {args.run_dir}: {gate['binding_sha256'][:12]}... "
          f"({len(gate['checked'])} fields checked, "
          f"{len(gate['recorded_not_checked'])} recorded)")

    from src.localization.agree_embed import embed_rirs, load_agree_audio

    agree = load_agree_audio(agree_path, args.device)

    def embedder(wavs):
        return embed_rirs(agree.model, wavs, args.device, readout=me.SCORER_READOUT)

    print(f"G1 audit re-verified: {len(plan.rooms)} rooms, branch {plan.branch}, "
          f"{plan.n_queries} queries")

    loader, facts = mq.build_release_stack(args.dataset_config, args.model_config)
    me.assert_release_rng_state(manifest)
    print(f"release call graph reproduced: {facts['call_graph']}")

    seen = {"n": 0}

    def _announce(record):
        seen["n"] += 1
        if seen["n"] % 250 == 0 or seen["n"] == 1:
            print(f"  {seen['n']:5d}/{len(records)}  {record['room_id']}  bank "
                  f"{record['n_candidates']}  e_loc {record['e_loc']:.3f} m "
                  f"(sparse oracle {record['e_oracle_sparse']:.3f} m)", flush=True)

    results = run_retrieval(embedder, iter_stream_items(loader), records, plan,
                            metadata_root=args.metadata_root, dataset_root=args.dataset_root,
                            bank_rule=args.bank_rule, tau=args.tau, on_record=_announce)
    report = build_report(results, {
        "records": records, "binding": dict(binding, **gate["recorded_not_checked"]),
        "binding_sha256": gate["binding_sha256"], "run_dir": str(args.run_dir),
        "audit_report": str(args.audit_report),
        "context_manifest": str(args.context_manifest),
        "metadata_root": str(args.metadata_root), "dataset_root": str(args.dataset_root),
        "agree_ckpt": str(agree_path), "device": str(args.device), "gate": gate,
    }, bootstrap_seed=args.bootstrap_seed, n_boot=args.n_boot)
    published = write_report(args.out_dir, report)

    across = report["metrics"]["across_rooms"]
    print(f"\nSPARSE/METADATA-BANK RETRIEVAL, room-first over "
          f"{report['census']['n_rooms']} rooms:")
    for name in flat_stat_names():
        entry = across[name]
        print(f"  {name:36s} {mr.format_number(entry['point'], 4)} "
              f"[{mr.format_number(entry['ci_lo'], 4)}, {mr.format_number(entry['ci_hi'], 4)}]")
    print(f"  sparse-bank oracle (median, room-first): "
          f"{mr.format_number(report['sparse_oracle']['across_rooms']['median_e_oracle']['point'], 4)} m")
    print(f"  bank size per query: min {report['bank']['pooled']['min']}, median "
          f"{report['bank']['pooled']['median']}, max {report['bank']['pooled']['max']}")
    for name, path in published["paths"].items():
        print(f"  {name:9s} -> {path}  sha256 {published['sha256'][name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
