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

Everything that decides a number is gated before it is computed: the sparse bank
itself -- every pair-metadata JSON and every RIR waveform it is built from -- must
hash to a digest frozen BEFORE these artifacts existed (:func:`bank_digest`,
:func:`assert_bank_digest`; PLANNER RULING 2), which is also the only thing here
that pins the continuous truth (:data:`TRUTH_INTEGRITY_NOTE`), the published run
binding must hash to its own content and agree with this control's
inputs, every query's context draw must re-verify against the frozen D1 manifest
(``meshgrid_engine.verify_context_record``), the pair metadata's receiver must be
the G1 manifest's receiver, the dense-grid oracle re-derived from that room's
candidate block must equal the one G1 published (:func:`assert_grid_oracle`: a
scalar consistency check saying the control and the audit describe the same
query, and NOT a proof of the truth vector -- that claim reduces to the bank
digest alone), and the result set must be exactly the registered
census. After the gate, the bytes stay bound: every digested file this control
re-reads is hashed again on the read that consumes it, and the observation must
be the file the loader opened AND decode to the tensor the loader delivered
(:func:`assert_observation_bytes`). Ground truth is resolved post hoc from the dataset's own pair metadata
through the SAME seam the r9 report uses
(``meshgrid_report.TruthResolver``) -- there is no second resolver -- and the
loader item stays wrapped in ``GuardedMetadata``, so this control cannot read
``md['source']`` either.
"""
import argparse
import hashlib
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
#:
#: ``model_config_sha256`` is here since r9e: it is the config the released
#: dataloader is built from, so its ``sample_rate``/``sample_size`` decide the
#: observed waveform and therefore every cosine (Codex r9c review, BLOCKER 3).
RETRIEVAL_BINDING_FIELDS = ("agree_ckpt_sha256", "scorer_readout", "model_config_sha256",
                            "d1_manifest_sha256", "g1_report_sha256", "room_manifest_sha256",
                            "branch", "dataset_config", "dataset_config_sha256")
#: run-binding fields this control gates against their REGISTERED value instead
#: of against the run's. ``tau`` is the whole list and the reason is exact: it is
#: inert here (the K = 1 score is the raw cosine), so matching the run would make
#: a stamped tau sensitivity check impossible to express, while matching the
#: registered constant is what the comparison against the R1 cell needs.
RETRIEVAL_BINDING_REGISTERED_ONLY = ("tau",)
#: the rest of the run binding: recorded from the published run, never required.
RETRIEVAL_BINDING_NOT_CHECKED = tuple(field for field in me.RUN_BINDING_FIELDS
                                      if field not in RETRIEVAL_BINDING_FIELDS
                                      and field not in RETRIEVAL_BINDING_REGISTERED_ONLY)
BINDING_SCOPE_NOTE = (
    "this control generates nothing -- there is no FLAC forward pass in it -- so the fields "
    "that decide a GENERATION (the checkpoint, the sampler steps, the CFG scale, the noise "
    "policy and its seed, the sample count and the nested prefixes, the conditioning method and "
    "its autocast, the dump authority) cannot change any number here. They are recorded from "
    "the published run binding and reported, and they do not refuse the control. What IS "
    "checked is everything that decides a cosine: the AGREE checkpoint, its readout, the MODEL "
    "CONFIG the observed-RIR loader is built from, the D1 context manifest (the stream this "
    "control walks), the G1 audit and room manifests (the receivers and the dense-grid oracle "
    "it contrasts against) and the dataset config. tau is in a THIRD class "
    "(RETRIEVAL_BINDING_REGISTERED_ONLY): it is gated against its REGISTERED value rather than "
    "against the run's, because it is inert here -- the K = 1 score is the raw cosine -- and "
    "gating it against the run would make a stamped tau sensitivity check inexpressible")

#: The AGREE checkpoint §1.4 pins BY DIGEST. Duplicated from
#: ``meshgrid_report.REGISTERED_ARTIFACT_SHA256['agree_ckpt_sha256']`` rather than
#: imported this round: r9d is editing that file in parallel, so the two are held
#: equal by a cross-pin test (``test_the_registered_values_are_the_ones_the_r1_
#: report_enforces``) and collapse into one constant in a later round.
REGISTERED_AGREE_SHA256 = "3a13243d6c6a11082697592c2c5db84790d37859451df2963eb51d655b23c787"
#: the registered model config's digest (inherited plan §1.4), duplicated on the
#: same terms and cross-pinned by the same test.
REGISTERED_MODEL_CONFIG_SHA256 = "f3eafef4456666e4705ddaf35540f6b9f1f746189814cec000bac794ba2a7ec9"

#: The loader settings that decide the OBSERVED waveform, with the values the
#: registered configs carry. Bound by VALUE and not merely by the run's config
#: digest (Codex r9f review): a random crop or a different sample size changes
#: obs_wav -- and therefore every cosine -- while every digest still matches.
REGISTERED_LOADER = {
    "loader_sample_rate": 22050,
    "loader_sample_size": 10240,
    "loader_force_channels": "mono",
    "loader_random_crop": False,
    "loader_augs": False,
}

#: the device TYPE the scored run used; compared as a type, so ``cuda:1`` is
#: ``cuda`` and a CPU control is a declared sensitivity check.
REGISTERED_DEVICE = "cuda"


def device_type(device):
    """``'cuda:1'`` -> ``'cuda'``: the part that can move a cosine's last bits."""
    if device is None:
        return None
    return torch.device(str(device)).type

#: How each result-affecting input is held.
#:
#: ``gated_against_registered`` -- compared with a value pinned in code; a
#: difference is a refusal unless the run is declared non-canonical, and is then
#: stamped everywhere. ``gated_against_run_binding`` -- compared field by field
#: with the published run's binding. ``gated_against_pre_registration`` --
#: compared with a digest the operator froze BEFORE the artifacts existed
#: (PLANNER RULING 2). ``stamped_not_checked`` -- recorded with the reason it
#: cannot move a number here.
INPUT_CLASSES = ("gated_against_registered", "gated_against_run_binding",
                 "gated_against_pre_registration", "stamped_not_checked")

_GENERATION_ONLY = ("this control generates nothing, so a generation setting cannot move any "
                    "number in it; recorded from the published run binding")


def _surface_entry(input_class, why, registered=None, in_run_binding=False):
    entry = {"class": input_class, "why": why, "in_run_binding": bool(in_run_binding)}
    if input_class == "gated_against_registered":
        entry["registered"] = registered
    return entry


#: EVERY input that can move a number in this control, and how each is held.
#:
#: The r9c review's MAJOR was that a partition of ``RUN_BINDING_FIELDS`` is
#: syntactically complete and semantically empty: the inputs that decide a
#: retrieval score mostly live OUTSIDE the run binding. This table is the full
#: surface, and a test asserts the run binding is a strict subset of it.
RESULT_AFFECTING_INPUTS = {
    # --- gated against a value pinned in code -----------------------------
    "agree_ckpt_sha256": _surface_entry(
        "gated_against_registered",
        "the scorer itself: a different AGREE checkpoint is a different embedding space and "
        "therefore a different ranking", registered=REGISTERED_AGREE_SHA256, in_run_binding=True),
    "scorer_readout": _surface_entry(
        "gated_against_registered",
        "the deterministic VAE-mean readout; the sampled path draws from AGREE's bottleneck and "
        "would make the cosines jitter", registered=me.SCORER_READOUT, in_run_binding=True),
    "tau": _surface_entry(
        "gated_against_registered",
        "INERT in this control since r9e -- the K = 1 score is the raw cosine, so tau cannot "
        "enter the arithmetic -- but the R1 cell these numbers are read beside was computed at "
        "tau = 0.1, so a different tau means a different protocol and is stamped",
        registered=me.TAU, in_run_binding=True),
    "bank_rule": _surface_entry(
        "gated_against_registered",
        "decides bank MEMBERSHIP (the released selector's f\"S00{node}\" pool omits S010)",
        registered=REGISTERED_BANK_RULE),
    "bootstrap_seed": _surface_entry(
        "gated_against_registered", "decides every published interval",
        registered=BOOTSTRAP_SEED),
    "n_boot": _surface_entry(
        "gated_against_registered", "decides every published interval",
        registered=BOOTSTRAP_N),
    "bootstrap_alpha": _surface_entry(
        "gated_against_registered",
        "the interval's LEVEL: at another alpha the published [ci_lo, ci_hi] are a different "
        "statement, and nothing else in the artifact would say so",
        registered=BOOTSTRAP_ALPHA),
    "success_radii_m": _surface_entry(
        "gated_against_registered", "decides the success columns",
        registered=[float(r) for r in SUCCESS_RADII]),
    "device": _surface_entry(
        "gated_against_registered",
        "the AGREE tower's GEMMs are the model's own batch/backend nondeterminism across device "
        "TYPES (the engine's BATCHING_CAVEAT measures ~one float16 ulp), so a cosine's last bits "
        "-- and, at a small enough top-1 margin, an argmax -- can move between CPU and CUDA. The "
        "scored run was CUDA; the device TYPE is compared (cuda:1 is cuda), and a CPU control is "
        "a stamped sensitivity check rather than a silent one",
        registered="cuda"),
    "model_config_sha256": _surface_entry(
        "gated_against_registered",
        "builds the released dataloader: sample_rate and sample_size decide the observed "
        "waveform and therefore every cosine (Codex r9c review, BLOCKER 3). ALSO matched "
        "field-by-field against the published run binding (RETRIEVAL_BINDING_FIELDS)",
        registered=REGISTERED_MODEL_CONFIG_SHA256, in_run_binding=True),
    # --- gated against the published run binding ---------------------------
    "dataset_config": _surface_entry(
        "gated_against_run_binding", "names the split the stream is walked over",
        in_run_binding=True),
    "dataset_config_sha256": _surface_entry(
        "gated_against_run_binding", "the split's exact bytes", in_run_binding=True),
    "d1_manifest_sha256": _surface_entry(
        "gated_against_run_binding", "the registered stream, its order and its context draws",
        in_run_binding=True),
    "g1_report_sha256": _surface_entry(
        "gated_against_run_binding", "the audit whose receivers and oracles are joined here",
        in_run_binding=True),
    "room_manifest_sha256": _surface_entry(
        "gated_against_run_binding", "the per-room candidate blocks the oracle is re-derived "
        "from", in_run_binding=True),
    "branch": _surface_entry(
        "gated_against_run_binding", "which candidate block the manifests resolve to",
        in_run_binding=True),
    # --- gated against a pre-registered digest -----------------------------
    "sparse_bank_sha256": _surface_entry(
        "gated_against_pre_registration",
        "the BYTES the control reads: every observed RIR, every pair file a truth or a bank "
        "position comes from, and every bank waveform -- plus bank membership. Pre-registered "
        "before the merged run exists, which is what makes post-hoc selection impossible "
        "(PLANNER RULING 2)"),
    # --- recorded, with the reason they cannot move a number ---------------
    "metadata_root": _surface_entry(
        "stamped_not_checked",
        "a PATH decides nothing; the bytes under it are bound by sparse_bank_sha256, which "
        "digests every pair file this control reads"),
    "dataset_root": _surface_entry(
        "stamped_not_checked",
        "a PATH decides nothing; the waveform bytes under it are bound by sparse_bank_sha256, "
        "and assert_roots_agree refuses a root the released loader would not read from"),
}
#: the loader settings that decide the observed waveform, bound BY VALUE.
RESULT_AFFECTING_INPUTS.update({
    name: _surface_entry(
        "gated_against_registered",
        "decides the observed waveform the cosines are taken against: the released eval read is "
        "torchaudio.load -> PadCrop(sample_size, randomize=random_crop) -> force_channels, so "
        "a different rate, size, channel fold, random crop or augmentation is a different "
        "h_obs. Bound by VALUE, not only by the run's config digest (Codex r9f review)",
        registered=value)
    for name, value in REGISTERED_LOADER.items()})
RESULT_AFFECTING_INPUTS.update({
    field: _surface_entry("stamped_not_checked", _GENERATION_ONLY, in_run_binding=True)
    for field in RETRIEVAL_BINDING_NOT_CHECKED})

#: what a run that deviates from any registered setting must say, everywhere.
NON_CANONICAL_NOTE = (
    "NON-CANONICAL RUN: at least one registered input of this control deviates from its "
    "pre-registered value, so these numbers are a SENSITIVITY CHECK and not the canonical "
    "sparse/metadata-bank result. The deviations are listed beside this note; a canonical run "
    "has none")

NON_CANONICAL_BANK_NOTE = (
    "the sparse-bank digest was RECORDED, not gated: no --expect-bank-sha256 was supplied, so "
    "the bytes behind every cosine, every bank position and every continuous truth are "
    "authorized only by their own existence. A digest that vouches for itself vouches for "
    "nothing (PLANNER RULING 2), so this run is non-canonical; freeze this value before the "
    "artifacts exist and pass it back to make the record a gate")

#: exactly what the scalar oracle re-derivation does and does not establish.
GRID_ORACLE_ESTABLISHES = (
    "ESTABLISHES that the continuous truth this control resolved is consistent with the "
    "distance the G1 audit published for this query's candidate block -- i.e. that the report "
    "and the audit are describing the same query. It does NOT establish that the truth VECTOR "
    "is the registered one: the oracle is a scalar and is not injective, so two truths mirrored "
    "inside one lattice cell share it (Codex r9 review, finding 3). That integrity claim is "
    "made by sparse_bank_sha256 and by nothing else here")

TRUTH_INTEGRITY_NOTE = (
    "the continuous truth x*_s is pinned by no run artifact -- the engine is structurally unable "
    "to read it and G1 publishes only the oracle DISTANCE -- so in this control truth integrity "
    "REDUCES TO the sparse-bank digest: every truth is read from a pair-metadata JSON, and every "
    "such file is digested into sparse_bank_sha256 by path and by exact bytes. When that digest "
    "is pre-registered, an edited src_loc anywhere in the bank changes it and the run refuses; "
    "when it is not, no gate here pins the truth vector and the artifact says so")

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


def read_bytes(path):
    """The file's exact bytes -- ONE read, so hashing and decoding cannot diverge."""
    with open(str(path), "rb") as handle:
        return handle.read()


def decode_rir(data, where=""):
    """Decoded RIR bytes -> ``[1, 1, T]`` float32, the released read.

    ``torchaudio.load`` plus the release's own two invariants (``AR_md``:
    22,050 Hz, single-channel ``single_channel_ir_*``). No crop and no pad is
    applied on purpose: the scorer's preprocessing
    (``agree_embed.preprocess_for_scoring``) truncates to the first 8,000 samples
    and pads to 10,240, so a raw read and the loader's 10,240-sample PadCrop
    reach the AGREE tower as the SAME tensor. Introducing a second crop
    convention here could only make them differ.

    Decoding takes BYTES rather than a path on purpose: the caller hashes the
    same ``bytes`` object it decodes, so the bytes that were verified are exactly
    the bytes that were consumed -- there is no second read to disagree with the
    first (Codex r9f review, byte continuity).
    """
    import io
    import torchaudio

    wave, rate = torchaudio.load(io.BytesIO(data))
    if int(rate) != RELEASED_SAMPLE_RATE:
        raise ValueError(f"{where}: IR sampling rate must be {RELEASED_SAMPLE_RATE}, got {rate} "
                         "(the released AR_md context read asserts the same thing)")
    if wave.ndim != 2 or wave.shape[0] != 1:
        raise ValueError(f"{where}: expected a single-channel RIR, got shape "
                         f"{tuple(wave.shape)}")
    if wave.shape[-1] == 0:
        raise ValueError(f"{where}: the RIR is empty")
    wave = wave.float()
    if not bool(torch.isfinite(wave).all()):
        raise ValueError(f"{where}: the RIR carries a non-finite sample")
    return wave.reshape(1, 1, -1)


def read_rir(path):
    """One real dataset RIR -> ``[1, 1, T]`` float32 (read then decode)."""
    return decode_rir(read_bytes(path), where=str(path))


def assert_file_bytes(path, expected_sha256, what, found=None):
    """The file still carries the bytes the digest covered, at the moment of use.

    The digest is taken before the pass; this is what closes the window between
    the gate and the read. Every post-gate read of a digested file goes through
    it, and the hash is taken over the SAME bytes the caller then consumes.
    """
    found = hashlib.sha256(read_bytes(path)).hexdigest() if found is None else str(found)
    if not expected_sha256:
        raise ValueError(f"{what} {path}: the sparse-bank digest carries no hash for this file, "
                         "so its bytes cannot be verified; the digest and the pass do not cover "
                         "the same files")
    if found != str(expected_sha256):
        raise ValueError(
            f"{what} {path} changed after the sparse-bank digest was taken: it now hashes to "
            f"{found[:16]}... but the digest covered {str(expected_sha256)[:16]}.... The bytes "
            "behind this query's cosines, positions or truth are not the pre-registered ones")
    return found


def verified_rir(path, expected_sha256, decoder=None):
    """A digested RIR, read once: hash the bytes, refuse, then decode THOSE bytes."""
    data = read_bytes(path)
    assert_file_bytes(path, expected_sha256, "bank waveform",
                      found=hashlib.sha256(data).hexdigest())
    decoder = decode_rir if decoder is None else decoder
    return decoder(data, str(path))


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


# --------------------------------------------------------------------------- #
# the sparse-bank digest: the bytes every number here is made of
# --------------------------------------------------------------------------- #
BANK_DIGEST_ALGORITHM = (
    "loc_meshgrid_sparse_bank/v1: canonical_sha256 (sorted-key, type-sensitive, whitespace-free "
    "JSON -- crossarm.canonical_bytes) over {'algorithm', 'rule', 'n_queries', 'queries'}, where "
    "'queries' maps each registered query_id to {'position', 'relpath', 'observation_sha256', "
    "'truth_pair': [metadata-root-relative pair path, sha256 of its bytes], 'bank': [[src_node, "
    "rec_node, pair path, pair sha256, wav path, wav sha256], ... in ascending numeric "
    "identity], 'missing_ir': [[src, rec], ...]}. Every byte this control reads is in the "
    "document: the observed RIR, the pair file the continuous truth is read from, and each bank "
    "entry's position file and waveform -- and so is MEMBERSHIP, because the per-query "
    "enumeration is part of the digested document, so a file appearing or disappearing changes "
    "the digest even when no byte of an existing file moved")


def observation_path(dataset_root, record):
    """The observed RIR's path -- ONE definition, used by the digest and the pass.

    The digest hashes this file and the loader opens this file; the r9f review's
    blocker was that those were two independent resolutions, so a pristine
    alternate ``--dataset-root`` could satisfy the frozen digest while the loader
    consumed different observation bytes. Everything downstream now derives from
    this function, and :func:`assert_observation_is_the_digested_file` asserts
    per query that the file the LOADER opened is this one.
    """
    return os.path.join(str(dataset_root), str(record["relpath"]))


def dataset_root_of_config(dataset_config_path):
    """The root the released dataloader will resolve ``relpath`` against.

    ``create_dataloader_from_config`` walks ``datasets[].path``; more than one
    entry would give the stream two roots and the digest one, so it is refused
    rather than guessed.
    """
    with open(str(dataset_config_path)) as handle:
        config = json.load(handle)
    datasets = config.get("datasets") or []
    roots = [entry.get("path") for entry in datasets if entry.get("path")]
    if len(roots) != 1:
        raise ValueError(
            f"{dataset_config_path} declares {len(roots)} dataset root(s) {roots}; this control "
            "digests exactly one root, so a config with none or several is refused rather than "
            "guessed")
    return str(roots[0])


def assert_roots_agree(dataset_root, dataset_config_path):
    """The digested root IS the root the loader will read from.

    Compared by ``realpath``, so a symlink or a trailing slash is not a
    difference, and a genuinely different tree is.
    """
    configured = dataset_root_of_config(dataset_config_path)
    left, right = os.path.realpath(str(dataset_root)), os.path.realpath(configured)
    if left != right:
        raise ValueError(
            f"--dataset-root {dataset_root!r} resolves to {left!r} but the dataset config "
            f"{dataset_config_path!r} resolves its files under {configured!r} ({right!r}). The "
            "digest would then cover different bytes than the loader consumes -- a pristine "
            "alternate root could satisfy the frozen sparse-bank digest while the observed RIRs "
            "came from somewhere else (Codex r9f review, BLOCKER). One root, or a refusal")
    return {"dataset_root": str(dataset_root), "configured": configured, "realpath": left}


def loader_values(model_config, dataset_config):
    """The observed-waveform settings the released stack will be built with.

    Read from the two configs exactly where ``meshgrid_queries.build_release_stack``
    reads them, so the values checked are the values used.
    """
    return {
        "loader_sample_rate": int(model_config["sample_rate"]),
        "loader_sample_size": int(model_config["sample_size"]),
        "loader_force_channels": str(dataset_config.get("force_channels", "stereo")),
        "loader_random_crop": bool(dataset_config.get("random_crop", True)),
        "loader_augs": bool(dataset_config.get("augs", False)),
    }


def observation_from_bytes(data, sample_size, where=""):
    """The loader's own observation tensor, rebuilt from the DIGESTED bytes.

    Reproduces the released read for an eval item exactly:
    ``load_file`` (``torchaudio.load``; no resample, the release asserts 22,050),
    then ``PadCrop_Normalized_T(sample_size, randomize=False)`` -- copy the first
    ``sample_size`` samples into a zero chunk -- then ``Mono()``, then no
    augmentations (the registered eval config sets ``augs: false``,
    ``random_crop: false``). Measured bit-identical to the loader's tensor on the
    real split.
    """
    wave = decode_rir(data, where=where).reshape(1, -1)
    sample_size = int(sample_size)
    chunk = wave.new_zeros([1, sample_size])
    chunk[:, :min(wave.shape[-1], sample_size)] = wave[:, :sample_size]
    return chunk


def assert_observation_is_the_digested_file(query_id, md, expected_path):
    """The file the LOADER opened is the file the digest covered."""
    loader_path = md.get("path") if hasattr(md, "get") else None
    if not loader_path:
        raise ValueError(f"{query_id}: the loader item carries no 'path', so the file it read "
                         "cannot be joined to the digested one")
    left, right = os.path.realpath(str(loader_path)), os.path.realpath(str(expected_path))
    if left != right:
        raise ValueError(
            f"{query_id}: the loader read {left!r} but the sparse-bank digest covers {right!r}; "
            "the observation being scored is not the observation that was digested")
    return right


def assert_observation_bytes(query_id, obs_wav, path, expected_sha256, sample_size):
    """The scored observation IS the digested bytes -- hash AND content.

    Two claims, in order: the file still hashes to what the digest covered, and
    decoding THOSE bytes through the released eval read reproduces the tensor the
    loader handed this control, exactly. The second is what makes the first bind
    the number rather than the file name: it is the digested bytes that produced
    the cosine.
    """
    data = read_bytes(path)
    assert_file_bytes(path, expected_sha256, "observation",
                      found=hashlib.sha256(data).hexdigest())
    rebuilt = observation_from_bytes(data, sample_size, where=str(path))
    observed = torch.as_tensor(obs_wav).detach().cpu().float().reshape(1, -1)
    if observed.shape != rebuilt.shape or not torch.equal(observed, rebuilt):
        raise ValueError(
            f"{query_id}: the observation the loader delivered is not what the digested bytes "
            f"of {path} decode to (shapes {tuple(observed.shape)} vs {tuple(rebuilt.shape)}, "
            f"max |delta| "
            f"{float((observed - rebuilt[:, :observed.shape[-1]]).abs().max()) if observed.shape[-1] <= rebuilt.shape[-1] else float('nan'):.3g}). "
            "The scored waveform did not come from the digested file")
    return True


def bank_digest(metadata_root, dataset_root, records, rule=REGISTERED_BANK_RULE, cache=None,
                file_digest=None, on_query=None):
    """Digest the sparse bank of the whole registered subset -> the gate's value.

    Pure, GPU-free and side-effect-free: it reads the same files the pass will
    read and hashes them, so the operator can freeze the value BEFORE the merged
    run exists (PLANNER RULING 2) and the pass can be refused when the tree it
    finds is a different one.

    Returns the digest, the document it was taken over and the file counts. Each
    distinct file is hashed once, however many banks it appears in.
    """
    from src.localization.crossarm import canonical_sha256

    file_digest = me.file_sha256 if file_digest is None else file_digest
    cache = {} if cache is None else cache
    digests, queries = {}, {}
    pair_files, wav_files, n_entries = set(), set(), 0

    def _sha(path):
        path = str(path)
        if path not in digests:
            digests[path] = file_digest(path)
        return digests[path]

    for record in sorted(records, key=lambda record: int(record["position"])):
        query_id = str(record["query_id"])
        if query_id in queries:
            raise ValueError(f"the record set names {query_id!r} twice; one query is one bank")
        relpath = str(record["relpath"])
        room_id = str(record["room_id"])
        bank = build_query_bank(metadata_root, dataset_root, room_id, relpath, rule=rule,
                                cache=cache)
        tables = room_bank(metadata_root, dataset_root, room_id, relpath, cache=cache)
        src_node, rec_node = parse_ir_filename(os.path.basename(relpath))
        truth_pair = tables.pairs[(src_node, rec_node)]
        observation = observation_path(dataset_root, record)

        rows = []
        for entry in bank["entries"]:
            rows.append([int(entry.src_node), int(entry.rec_node),
                         os.path.relpath(entry.pair_path, str(metadata_root)),
                         _sha(entry.pair_path),
                         os.path.relpath(entry.ir_path, str(dataset_root)),
                         _sha(entry.ir_path)])
            pair_files.add(entry.pair_path)
            wav_files.add(entry.ir_path)
        n_entries += len(rows)
        pair_files.add(truth_pair)
        wav_files.add(observation)
        queries[query_id] = {
            "position": int(record["position"]),
            "relpath": relpath,
            "observation_sha256": _sha(observation),
            "truth_pair": [os.path.relpath(truth_pair, str(metadata_root)), _sha(truth_pair)],
            "bank": rows,
            "missing_ir": [[int(src), int(rec)] for src, rec in bank["missing_ir"]],
        }
        if on_query is not None:
            on_query(query_id, queries[query_id])

    document = {"algorithm": BANK_DIGEST_ALGORITHM, "rule": str(rule),
                "n_queries": len(queries), "queries": queries}
    return {"sha256": canonical_sha256(document), "algorithm": BANK_DIGEST_ALGORITHM,
            "rule": str(rule), "n_queries": len(queries), "n_bank_entries": int(n_entries),
            "n_pair_files": len(pair_files), "n_wav_files": len(wav_files),
            "queries": queries}


def assert_bank_digest(found, expected=None, allow_non_canonical=False):
    """The bank this run reads is the PRE-REGISTERED one, or the run is not canonical.

    Three outcomes, and only three: the digest matches a value frozen before the
    artifacts existed (canonical); no value was frozen and the operator has
    explicitly asked for a non-canonical run (recorded, stamped); or a refusal.
    A supplied value that does not match is ALWAYS a refusal -- ``--non-canonical``
    excuses the absence of a pre-registration, never a contradiction of one.
    """
    found = str(found)
    if expected:
        if str(expected) != found:
            raise ValueError(
                f"the sparse bank this run reads hashes to {found[:16]}... but the registered "
                f"bank is {str(expected)[:16]}...; this is not the pre-registered sparse bank, "
                "so the bytes behind every cosine, every bank position and every continuous "
                "truth are not the registered ones. --non-canonical cannot excuse this: it "
                "excuses the ABSENCE of a pre-registration, not a contradiction of one")
        return {"sparse_bank_sha256": found, "canonical": True, "mode": "pre_registered",
                "expected": str(expected),
                "note": "the sparse bank matches the digest frozen before these artifacts "
                        "existed"}
    if not allow_non_canonical:
        raise ValueError(
            "a canonical run requires the pre-registered sparse-bank digest: pass "
            f"--expect-bank-sha256 (this tree hashes to {found}). Compute and freeze it with "
            "--print-bank-sha256 before the artifacts exist, or pass --non-canonical to record "
            f"it instead of gating on it. {NON_CANONICAL_BANK_NOTE}")
    return {"sparse_bank_sha256": found, "canonical": False,
            "mode": "recorded_not_pre_registered", "expected": None,
            "note": NON_CANONICAL_BANK_NOTE}


def assert_bank_unchanged(query_id, entries, document):
    """The bank built during the pass is the one the digest was taken over.

    The digest is a gate taken before the walk; this closes the window between
    them for MEMBERSHIP, which is the part a moved file changes without touching
    any byte the pass re-reads.
    """
    recorded = (document or {}).get("queries", {}).get(str(query_id))
    if recorded is None:
        raise ValueError(f"{query_id} is not in the sparse-bank digest document; the digest and "
                         "the pass do not cover the same subset")
    found = [[int(entry.src_node), int(entry.rec_node)] for entry in entries]
    wanted = [[int(row[0]), int(row[1])] for row in recorded["bank"]]
    if found != wanted:
        raise ValueError(
            f"{query_id}: the sparse bank changed after the sparse-bank digest was taken "
            f"(now {found[:4]}, digested {wanted[:4]}); the files under the dataset root moved "
            "between the gate and the pass")
    return True


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
def embed_bank(embedder, entries, decoder=None, cache=None, expected_sha256=None):
    """``[M, D]`` embeddings of the bank's real RIRs, one verified read per file.

    ``cache`` is keyed by the IR path, so a receiver's bank is embedded once and
    reused by every query of that receiver -- the same memoization argument the
    engine makes for its source cache, on a much smaller object. The file is
    hashed on that one read and refused if its bytes are not the digested ones
    (``expected_sha256`` maps path -> sha256); a cache HIT is a file this run
    already verified.

    ``expected_sha256=None`` means there is NO digest to verify against and the
    read is unverified -- the state the report publishes as
    ``bank_membership_rechecked_against_the_digest: false``. A MAP with a path
    missing from it is a refusal, not a skip: a file the digest did not cover has
    no business in a bank the digest is supposed to describe.
    """
    rows = []
    for entry in entries:
        path = str(entry.ir_path)
        if cache is not None and path in cache:
            rows.append(cache[path])
            continue
        if expected_sha256 is None:
            data = read_bytes(path)
            waveform = (decode_rir if decoder is None else decoder)(data, path)
        else:
            waveform = verified_rir(path, expected_sha256.get(path), decoder=decoder)
        embedding = torch.as_tensor(embedder(waveform)).float().reshape(-1)
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


def bank_scores(sims):
    """The score of a one-sample bank: the cosine itself -> ``[M]``.

    The plan's aggregate at K = 1 is ``S = tau * (logsumexp(s / tau) - log 1)``,
    which is ALGEBRAICALLY the cosine -- and r9b computed it that way, through
    the engine's aggregator. The r9c review's MAJOR is that the algebra is not
    the arithmetic: a float32 divide, an exp/log and a multiply can move a score
    by an ulp, and two adjacent cosines that differ can come back equal, handing
    the decision to the tie-break. The score is therefore taken directly, so it
    is the cosine bit for bit; the equivalence stays true and stays a comment.

    A copy, never a view: the caller keeps the similarities as published.
    """
    values = torch.as_tensor(sims).float().reshape(-1)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("bank similarities must be finite (no NaN or Inf)")
    return values.clone()


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
                        receiver_xyz, radii=SUCCESS_RADII, bank=None,
                        grid=None, bank_rule=REGISTERED_BANK_RULE):
    """Every §2 readout for one query, against the SPARSE bank."""
    truth = np.asarray(truth, dtype=np.float64).reshape(3)
    sims = torch.as_tensor(sims).float().reshape(-1)
    if sims.numel() != len(entries):
        raise ValueError(f"{query_id}: {sims.numel()} similarities for {len(entries)} bank "
                         "entries")
    assert_bank_excludes_the_target(entries, truth, query_id)
    scores = bank_scores(sims)
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


def _digest_hashes(bank_document, metadata_root, dataset_root, query_id):
    """``{path: sha256}`` for everything the digest covered for ONE query.

    The document stores root-relative paths (so the digest is machine-portable);
    this rejoins them to the roots the pass is actually reading from, which is
    the only place the two representations meet.
    """
    recorded = (bank_document or {}).get("queries", {}).get(str(query_id))
    if recorded is None:
        raise ValueError(f"{query_id} is not in the sparse-bank digest document; the digest and "
                         "the pass do not cover the same subset")
    hashes = {os.path.join(str(metadata_root), recorded["truth_pair"][0]):
                  recorded["truth_pair"][1]}
    for row in recorded["bank"]:
        hashes[os.path.join(str(metadata_root), row[2])] = row[3]
        hashes[os.path.join(str(dataset_root), row[4])] = row[5]
    return hashes, recorded


def run_retrieval(embedder, stream, records, plan, *, metadata_root, dataset_root,
                  bank_rule=REGISTERED_BANK_RULE, radii=SUCCESS_RADII, sample_size=None,
                  decoder=None, bank_document=None, on_record=None):
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
    candidate block: its receiver, and the dense-grid oracle that must equal the
    one the audit published (:func:`assert_grid_oracle` -- a scalar consistency
    check, NOT a proof of the truth vector; see :data:`TRUTH_INTEGRITY_NOTE`).

    ``bank_document`` is the enumeration :func:`bank_digest` was taken over, and
    the driver always supplies it. With it, three things are re-checked per query
    AFTER the gate, because membership alone is not byte continuity (Codex r9f
    review): the bank's membership, the BYTES of every file this control reads
    from that bank (each pair JSON and each waveform, hashed on the same read
    that consumes them), and the observation -- whose file must be the one the
    loader opened AND must decode to the tensor the loader delivered
    (``sample_size`` is the registered loader crop). When the document is absent
    the report says so
    (``gates.bank_membership_rechecked_against_the_digest``), because a gate that
    did not run may not be reported as one that did.
    """
    if bank_rule not in BANK_RULES:
        raise ValueError(f"unknown bank rule {bank_rule!r} (expected one of {list(BANK_RULES)})")
    by_position = {int(record["position"]): record for record in records}
    me.assert_room_blocks(records)
    resolver = mr.TruthResolver(metadata_root)
    rooms, embeddings, results, seen = {}, {}, [], set()
    current_room, current_index, finished_rooms, oracle_deltas = None, {}, set(), []
    #: pair files whose bytes this run already checked against the digest. A file
    #: is verified on its first use and not on every one of the ten queries that
    #: share it; the run is one window, not ten.
    verified_files = set()

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

        # the bytes the digest covered, rejoined to the roots being read now.
        # Taken BEFORE the truth is resolved, because the truth is read out of one
        # of these very files.
        hashes, recorded = ({}, None)
        if bank_document is not None:
            hashes, recorded = _digest_hashes(bank_document, metadata_root, dataset_root,
                                              query_id)
            truth_pair = os.path.join(str(metadata_root), recorded["truth_pair"][0])
            if truth_pair not in verified_files:
                assert_file_bytes(truth_pair, recorded["truth_pair"][1],
                                  "the truth-carrying pair file")
                verified_files.add(truth_pair)

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
        if bank_document is not None:
            assert_bank_unchanged(query_id, bank["entries"], bank_document)
            # every POSITION file this bank read, once per run
            for entry in bank["entries"]:
                pair_path = str(entry.pair_path)
                if pair_path not in verified_files:
                    assert_file_bytes(pair_path, hashes.get(pair_path),
                                      "a bank entry's pair file")
                    verified_files.add(pair_path)
            # and the observation: the loader's file, and the loader's tensor
            expected_observation = observation_path(dataset_root, record)
            assert_observation_is_the_digested_file(query_id, md, expected_observation)
            assert_observation_bytes(query_id, obs_wav, expected_observation,
                                     recorded["observation_sha256"],
                                     obs_wav.shape[-1] if sample_size is None else sample_size)

        obs_embedding = torch.as_tensor(embedder(torch.as_tensor(obs_wav)))[0].float()
        sims = bank_sims(obs_embedding,
                         embed_bank(embedder, bank["entries"], decoder=decoder,
                                    cache=embeddings,
                                    expected_sha256=hashes if bank_document is not None
                                    else None))
        result = evaluate_bank_query(
            bank["entries"], sims, truth, query_id=query_id, room_id=room_id,
            position=position, receiver_id=query.receiver_id,
            receiver_xyz=query.receiver_xyz, radii=radii, bank=bank, grid=grid,
            bank_rule=bank_rule)
        result["e_oracle_grid_delta"] = float(oracle_deltas[-1])
        # what this query's pass ACTUALLY did, so the report's gate flags are
        # derived from the walk rather than claimed by its caller
        result["digest_verified"] = bool(bank_document is not None)
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
    """How far the re-derived dense-grid oracle sat from the one G1 published."""
    deltas = np.asarray([float(result.get("e_oracle_grid_delta", 0.0)) for result in results],
                        dtype=np.float64)
    return {"n_queries": int(deltas.size), "max_abs_delta_m": float(deltas.max()),
            "mean_abs_delta_m": float(deltas.mean()),
            "tolerance_m": float(mr.ORACLE_TOLERANCE),
            "establishes": GRID_ORACLE_ESTABLISHES,
            "note": "the control re-derives the DENSE-GRID oracle min_c ||c - x*_s|| from the "
                    "G1 candidate block and the pair metadata's src_loc and requires it to "
                    "equal the value the audit published. It is a scalar and therefore not "
                    "injective; what pins the truth VECTOR is the pre-registered sparse-bank "
                    "digest, which covers the pair file the truth is read from "
                    "(TRUTH_INTEGRITY_NOTE)"}


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


# --------------------------------------------------------------------------- #
# canonicality: which registered inputs this run actually used
# --------------------------------------------------------------------------- #
def observed_inputs(context, *, bank_rule, bootstrap_seed, n_boot, alpha=BOOTSTRAP_ALPHA,
                    radii=SUCCESS_RADII):
    """What each named input WAS on this run, for the surface report and the gate."""
    binding = dict(context.get("binding") or {})
    observed = {
        "agree_ckpt_sha256": context.get("agree_ckpt_sha256"),
        "scorer_readout": context.get("scorer_readout"),
        # the config THIS control hashed; the binding gate separately proves it
        # is the run's, so a difference here is a registered-value deviation
        "model_config_sha256": (context.get("model_config_sha256")
                                or (context.get("binding") or {}).get("model_config_sha256")),
        "tau": context.get("tau"),
        "bank_rule": bank_rule,
        "bootstrap_seed": int(bootstrap_seed),
        "n_boot": int(n_boot),
        "bootstrap_alpha": float(alpha),
        "success_radii_m": [float(r) for r in radii],
        "sparse_bank_sha256": (context.get("bank_gate") or {}).get("sparse_bank_sha256"),
        "metadata_root": context.get("metadata_root"),
        "dataset_root": context.get("dataset_root"),
        # the TYPE, because that is what can move a cosine's last bits
        "device": device_type(context.get("device")),
    }
    observed.update({name: (context.get("loader") or {}).get(name)
                     for name in REGISTERED_LOADER})
    for field in me.RUN_BINDING_FIELDS:
        observed.setdefault(field, binding.get(field))
    return observed


def assess_canonicality(context, *, bank_rule, bootstrap_seed, n_boot, alpha=BOOTSTRAP_ALPHA,
                        radii=SUCCESS_RADII):
    """Every registered input that deviates, named -- or an empty list.

    The rule is one sentence: a canonical run of this control uses the registered
    value of every ``gated_against_registered`` input AND a pre-registered sparse
    bank. Anything else is a sensitivity check that must say so in every
    artifact.
    """
    observed = observed_inputs(context, bank_rule=bank_rule, bootstrap_seed=bootstrap_seed,
                               n_boot=n_boot, alpha=alpha, radii=radii)
    deviations = []
    for name, entry in sorted(RESULT_AFFECTING_INPUTS.items()):
        if entry["class"] != "gated_against_registered":
            continue
        used, registered = observed.get(name), entry["registered"]
        if used is None:
            raise ValueError(f"the report cannot state canonicality: the run did not record "
                             f"which {name!r} it used, and this input is gated against its "
                             f"registered value ({registered!r})")
        if used != registered:
            deviations.append({"input": name, "used": used, "registered": registered,
                               "why_it_matters": entry["why"]})
    gate = dict(context.get("bank_gate") or {})
    if not gate:
        raise ValueError("the report cannot state canonicality: no sparse-bank gate was "
                         "recorded, so nothing pins the bytes behind these numbers "
                         "(assert_bank_digest)")
    if not gate.get("canonical"):
        deviations.append({"input": "sparse_bank_sha256",
                           "used": gate.get("sparse_bank_sha256"), "registered": None,
                           "why_it_matters": NON_CANONICAL_BANK_NOTE})
    return {"canonical": not deviations, "deviations": deviations, "observed": observed,
            "note": None if not deviations else NON_CANONICAL_NOTE}


def assert_canonical(assessment, allow_non_canonical=False):
    """A deviation is a refusal unless the operator declared a sensitivity run."""
    if assessment["canonical"] or allow_non_canonical:
        return assessment
    first = assessment["deviations"][0]
    raise ValueError(
        f"{first['input']} = {first['used']!r} is not the registered value "
        f"({first['registered']!r}), and {len(assessment['deviations'])} registered input(s) "
        f"deviate in total ({sorted(d['input'] for d in assessment['deviations'])}). A canonical "
        f"run uses the registered value of every one of them. {NON_CANONICAL_NOTE}. Pass "
        "--non-canonical (allow_non_canonical=True) to publish this as a stamped sensitivity "
        "check instead")


def input_surface(assessment):
    """The full result-affecting input surface, with what this run used."""
    observed = assessment["observed"]
    deviating = {deviation["input"] for deviation in assessment["deviations"]}
    surface = {}
    for name, entry in sorted(RESULT_AFFECTING_INPUTS.items()):
        row = {"class": entry["class"], "why": entry["why"],
               "in_run_binding": bool(entry["in_run_binding"]),
               "used": observed.get(name)}
        if entry["class"] == "gated_against_registered":
            row["registered"] = entry["registered"]
            row["matches_registered"] = name not in deviating
        if entry["class"] == "gated_against_pre_registration":
            row["pre_registered"] = name not in deviating
        surface[name] = row
    return surface


def build_report(results, context, *, radii=SUCCESS_RADII, bootstrap_seed=BOOTSTRAP_SEED,
                 n_boot=BOOTSTRAP_N, alpha=BOOTSTRAP_ALPHA, totals=None):
    """The machine-readable control report: every §2 readout, every label.

    ``context`` is the run context mapping. ``records``, ``binding_sha256``,
    ``bank_gate`` (from :func:`assert_bank_digest`) and the observed values of
    every ``gated_against_registered`` input (``agree_ckpt_sha256``,
    ``scorer_readout``, ``tau``) are REQUIRED, because the report has to state
    canonicality and cannot state it about inputs it was not told. ``binding``,
    ``run_dir``, ``audit_report``, ``context_manifest``, ``metadata_root``,
    ``dataset_root``, ``device`` and ``totals`` are recorded as provenance when
    present. ``allow_non_canonical`` turns a deviation from a refusal into a
    stamped sensitivity check.
    """
    if not results:
        raise ValueError("a retrieval control report needs at least one scored query")
    if not context.get("binding_sha256"):
        raise ValueError("the report must name the run binding this control was gated against; "
                         "a control number with no binding cannot be placed beside that run's")
    census = assert_retrieval_census(results, context["records"],
                                     totals=totals or context.get("totals"))
    bank = bank_report(results)
    assessment = assert_canonical(
        assess_canonicality(context, bank_rule=bank["rule"], bootstrap_seed=bootstrap_seed,
                            n_boot=n_boot, alpha=alpha, radii=radii),
        allow_non_canonical=bool(context.get("allow_non_canonical")))
    bank_gate = dict(context.get("bank_gate") or {})
    digest_verified = all(bool(result.get("digest_verified")) for result in results)
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
            "truth_integrity": TRUTH_INTEGRITY_NOTE,
            "bank_digest_algorithm": BANK_DIGEST_ALGORITHM,
            # present ONLY when it applies, so its presence is the signal
            **({"non_canonical": NON_CANONICAL_NOTE} if not assessment["canonical"] else {}),
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
            "score": "cos(E(h_obs), E(h_real)), taken DIRECTLY. The plan's K = 1 aggregate "
                     "tau * (logsumexp(s / tau) - log 1) is algebraically the same number, but "
                     "computing it that way is a float32 divide, exp/log and multiply that can "
                     "move a score by an ulp and manufacture a tie, so the cosine is the score "
                     "bit for bit (Codex r9c review, MAJOR)",
            "tau": context.get("tau"),
            "tau_is_registered": bool(context.get("tau") == me.TAU),
            "tau_note": RESULT_AFFECTING_INPUTS["tau"]["why"],
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
            # the sparse bank's identity, and whether it was GATED or merely read
            "sparse_bank_sha256": bank_gate.get("sparse_bank_sha256"),
            "sparse_bank_pre_registered": bool(bank_gate.get("canonical")),
            "sparse_bank_mode": bank_gate.get("mode"),
            "bank_digest_algorithm": BANK_DIGEST_ALGORITHM,
            # the one-line verdict every table is read under
            "canonical": bool(assessment["canonical"]),
            "deviations": assessment["deviations"],
        },
        "census": census,
        "input_surface": input_surface(assessment),
        "gates": {
            "note": "every entry below is a gate the control refuses on; a published report is "
                    "proof they passed, not a claim that they did",
            "binding_checked_against_published_run": bool(context.get("gate") is not None),
            "d1_context_draw_reverified_per_query": True,
            "pair_metadata_receiver_matches_g1": True,
            # renamed in r9e: the scalar check establishes agreement with the
            # audit, NOT the identity of the truth vector (Codex r9c, BLOCKER 2)
            "grid_oracle_rederived_matches_the_g1_scalar": True,
            "truth_bytes_pinned_by_pre_registered_bank_digest":
                bool(bank_gate.get("canonical")),
            # derived from the ROWS, not from the caller: every query stamps
            # whether it was walked against the digest document
            "bank_membership_rechecked_against_the_digest": digest_verified,
            # membership alone is not byte continuity (Codex r9f review): every
            # digested file re-read after the gate is hashed on the read that
            # consumes it, and the observation must decode to the scored tensor
            "digested_bytes_reverified_on_every_post_gate_read": digest_verified,
            "observation_is_the_digested_file_and_decodes_to_the_scored_tensor":
                digest_verified,
            "one_dataset_root_for_the_digest_and_the_loader":
                bool(context.get("roots_agree", False)),
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
    canonical = bool(report["protocol"]["canonical"])
    return {
        "control_key": CONTROL_KEY,
        "control_label": CONTROL_LABEL,
        "status": "run (canonical)" if canonical else "run (NON-CANONICAL)",
        "canonical": canonical,
        "deviations": report["protocol"]["deviations"],
        "non_canonical_note": None if canonical else NON_CANONICAL_NOTE,
        "sparse_bank_sha256": report["protocol"]["sparse_bank_sha256"],
        "sparse_bank_pre_registered": report["protocol"]["sparse_bank_pre_registered"],
        "truth_integrity": TRUTH_INTEGRITY_NOTE,
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
    # the verdict is the FIRST thing on the page: every table below is read under
    # it, so it cannot sit at the bottom
    if protocol["canonical"]:
        lines.append(f"> **CANONICAL RUN** — every registered input matches its pre-registered "
                     f"value and the sparse bank was gated against the pre-registered digest "
                     f"`{str(protocol['sparse_bank_sha256'])[:16]}...`.")
    else:
        lines.append(f"> **NON-CANONICAL RUN** — {report['labels']['non_canonical']}.")
        for deviation in protocol["deviations"]:
            lines.append(f">   - `{deviation['input']}` = `{deviation['used']}` "
                         f"(registered: `{deviation['registered']}`) — "
                         f"{deviation['why_it_matters']}")
    lines.append("")
    lines.append(f"Generated {report['created_utc']}.")
    lines.append("")
    lines.append(f"> **{report['control_label']}**")
    lines.append("")
    lines.append(f"- **Scope:** {report['labels']['subset']}")
    lines.append(f"- **Run binding:** `{provenance['binding_sha256']}`")
    lines.append(f"- **Sparse bank:** `{protocol['sparse_bank_sha256']}` "
                 f"({protocol['sparse_bank_mode']})")
    lines.append(f"- **Truth integrity:** {report['labels']['truth_integrity']}")
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
                             model_config_sha256=me.file_sha256(args.model_config))
    return {field: full[field] for field in RETRIEVAL_BINDING_FIELDS}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # NOT required at parse time: --print-bank-sha256 runs BEFORE the merged run
    # exists (PLANNER RULING 2), so it cannot be made to name one. validate_args
    # requires them for every mode that scores anything.
    parser.add_argument("--run-dir", default=None,
                        help="the MERGED I1 run directory this control is reported beside")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--print-bank-sha256", action="store_true",
                        help="compute the sparse-bank digest over the registered subset, print "
                             "it and exit -- the pre-registration step; touches no run, no "
                             "scorer and no device")
    parser.add_argument("--expect-bank-sha256", default=None,
                        help="the PRE-REGISTERED sparse-bank digest; required for a canonical "
                             "run")
    parser.add_argument("--non-canonical", action="store_true",
                        help="publish a stamped SENSITIVITY CHECK: allows a missing "
                             "--expect-bank-sha256 and any deviation from a registered input, "
                             "and marks every artifact non-canonical")
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
    """Startup refusals -- before a checkpoint is read or a device is touched.

    Every deviation from a registered input is refused here unless
    ``--non-canonical`` declares the run a sensitivity check; the same rule is
    then applied again by :func:`build_report`, from the values actually used, so
    a flag alone cannot make an artifact claim canonicality.
    """
    if args.bank_rule not in BANK_RULES:
        _refuse(f"unknown bank rule {args.bank_rule!r} (expected one of {list(BANK_RULES)})")
    if int(args.n_boot) < 1:
        _refuse("--n-boot must be at least 1")
    if float(args.tau) <= 0.0:
        _refuse(f"--tau must be > 0, got {args.tau}")

    # ONE root, in every mode: the digest must cover the bytes the loader will
    # consume, so a --dataset-root the dataset config would not read from is
    # refused before anything is hashed (Codex r9f review, BLOCKER)
    try:
        assert_roots_agree(args.dataset_root, args.dataset_config)
    except (ValueError, OSError) as error:
        _refuse(str(error))

    if args.print_bank_sha256:
        # the pre-registration mode: it names no run and publishes no number
        return True
    for name, value in (("--run-dir", args.run_dir), ("--out-dir", args.out_dir)):
        if not value:
            _refuse(f"{name} is required for a scoring run (only --print-bank-sha256 may omit "
                    "it)")
    if os.path.abspath(str(args.out_dir)) == os.path.abspath(str(args.run_dir)):
        _refuse("--out-dir may not be the scored run directory: a control never writes into "
                "the artifact set it reports against")

    deviations = []
    if args.bank_rule != REGISTERED_BANK_RULE:
        deviations.append(f"--bank-rule {args.bank_rule!r} is not the registered "
                          f"{REGISTERED_BANK_RULE!r} ({BANK_RULE_NOTE})")
    if float(args.tau) != float(me.TAU):
        deviations.append(f"--tau {args.tau} is not the registered {me.TAU} "
                          f"({RESULT_AFFECTING_INPUTS['tau']['why']})")
    if int(args.bootstrap_seed) != BOOTSTRAP_SEED or int(args.n_boot) != BOOTSTRAP_N:
        deviations.append(f"the bootstrap {args.bootstrap_seed} x {args.n_boot} is not the "
                          f"registered {BOOTSTRAP_SEED} x {BOOTSTRAP_N}")
    if device_type(args.device) != REGISTERED_DEVICE:
        deviations.append(f"--device {args.device!r} is a {device_type(args.device)!r} device "
                          f"and the scored run was {REGISTERED_DEVICE!r} "
                          f"({RESULT_AFFECTING_INPUTS['device']['why']})")
    if not args.expect_bank_sha256:
        deviations.append("no --expect-bank-sha256 was supplied, so the sparse bank would be "
                          f"recorded rather than gated ({NON_CANONICAL_BANK_NOTE})")
    if deviations and not args.non_canonical:
        _refuse("this run deviates from the registered protocol and does not declare it:\n  - "
                + "\n  - ".join(deviations)
                + f"\n{NON_CANONICAL_NOTE}. Pass --non-canonical to publish it as a stamped "
                  "sensitivity check")
    for deviation in deviations:
        print(f"NOTE (non-canonical): {deviation}")
    return True


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)

    if args.print_bank_sha256:
        # PRE-REGISTRATION MODE (PLANNER RULING 2). No run, no scorer, no device:
        # this is what the main session freezes BEFORE the merged run exists, so
        # a post-hoc choice of bank is impossible.
        manifest = mq.load_manifest(args.context_manifest)
        digest = bank_digest(args.metadata_root, args.dataset_root, manifest["records"],
                             rule=args.bank_rule)
        print(f"{BANK_DIGEST_ALGORITHM}\n")
        print(f"queries        {digest['n_queries']:,}")
        print(f"bank entries   {digest['n_bank_entries']:,}")
        print(f"pair files     {digest['n_pair_files']:,}")
        print(f"waveform files {digest['n_wav_files']:,}")
        print(f"bank rule      {digest['rule']}")
        print(f"\nsparse_bank_sha256 {digest['sha256']}")
        print(f"\nfreeze it, then run the control with "
              f"--expect-bank-sha256 {digest['sha256']}")
        return 0

    print(f"{CONTROL_LABEL}\n")
    print(f"SELF-PAIR RULE: {SELF_PAIR_RULE}")
    print(f"BANK RULE: {args.bank_rule} -- {BANK_RULE_NOTE}")
    print(f"AGREE LEAKAGE CAVEAT: {AGREE_LEAKAGE_CAVEAT}")
    print(f"SPARSE ORACLE: {SPARSE_ORACLE_LABEL}")
    print(f"TRUTH INTEGRITY: {TRUTH_INTEGRITY_NOTE}\n")

    from localize_meshgrid import _iter_items as iter_stream_items

    with open(args.model_config) as handle:
        model_config = json.load(handle)
    with open(args.dataset_config) as handle:
        dataset_config = json.load(handle)
    resolved = mq.with_resolved_agree(model_config)
    agree_path = args.agree_ckpt or resolved["training"]["metrics"]["AGREE_ckpt"]
    # the settings that decide obs_wav, read where the release stack reads them
    loader = loader_values(model_config, dataset_config)
    print(f"loader: {loader}")

    plan = me.load_audit_plan(args.audit_report, branch=args.branch)
    manifest = mq.load_manifest(args.context_manifest)
    records = manifest["records"]

    # the gates run on FILE digests, before the scorer is built or a device is
    # touched: nothing is loaded until the control is known to continue this run
    agree_sha256 = me.file_sha256(agree_path)
    binding = build_retrieval_binding(args, plan, agree_sha256)
    gate = assert_retrieval_binding(args.run_dir, binding)
    print(f"binding gate passed against {args.run_dir}: {gate['binding_sha256'][:12]}... "
          f"({len(gate['checked'])} fields checked, "
          f"{len(gate['recorded_not_checked'])} recorded)")

    print("digesting the sparse bank (every pair file and every waveform it reads)...",
          flush=True)
    document = bank_digest(args.metadata_root, args.dataset_root, records,
                           rule=args.bank_rule)
    bank_gate = assert_bank_digest(document["sha256"], expected=args.expect_bank_sha256,
                                   allow_non_canonical=bool(args.non_canonical))
    print(f"sparse bank {document['sha256'][:16]}... ({bank_gate['mode']}): "
          f"{document['n_bank_entries']:,} entries over {document['n_wav_files']:,} waveforms "
          f"and {document['n_pair_files']:,} pair files")

    from src.localization.agree_embed import embed_rirs, load_agree_audio

    agree = load_agree_audio(agree_path, args.device)

    def embedder(wavs):
        return embed_rirs(agree.model, wavs, args.device, readout=me.SCORER_READOUT)

    print(f"G1 audit re-verified: {len(plan.rooms)} rooms, branch {plan.branch}, "
          f"{plan.n_queries} queries")

    stream_loader, facts = mq.build_release_stack(args.dataset_config, args.model_config)
    me.assert_release_rng_state(manifest)
    print(f"release call graph reproduced: {facts['call_graph']}")

    seen = {"n": 0}

    def _announce(record):
        seen["n"] += 1
        if seen["n"] % 250 == 0 or seen["n"] == 1:
            print(f"  {seen['n']:5d}/{len(records)}  {record['room_id']}  bank "
                  f"{record['n_candidates']}  e_loc {record['e_loc']:.3f} m "
                  f"(sparse oracle {record['e_oracle_sparse']:.3f} m)", flush=True)

    results = run_retrieval(embedder, iter_stream_items(stream_loader), records, plan,
                            metadata_root=args.metadata_root, dataset_root=args.dataset_root,
                            bank_rule=args.bank_rule, bank_document=document,
                            sample_size=loader["loader_sample_size"], on_record=_announce)
    report = build_report(results, {
        "records": records, "binding": dict(binding, **gate["recorded_not_checked"]),
        "binding_sha256": gate["binding_sha256"], "run_dir": str(args.run_dir),
        "audit_report": str(args.audit_report),
        "context_manifest": str(args.context_manifest),
        "metadata_root": str(args.metadata_root), "dataset_root": str(args.dataset_root),
        "agree_ckpt": str(agree_path), "agree_ckpt_sha256": agree_sha256,
        "model_config_sha256": binding["model_config_sha256"],
        "scorer_readout": me.SCORER_READOUT, "tau": float(args.tau),
        "device": str(args.device), "loader": loader,
        "gate": gate, "bank_gate": bank_gate,
        "roots_agree": True,
        "allow_non_canonical": bool(args.non_canonical),
    }, bootstrap_seed=args.bootstrap_seed, n_boot=args.n_boot)
    published = write_report(args.out_dir, report)

    across = report["metrics"]["across_rooms"]
    verdict = "CANONICAL" if report["protocol"]["canonical"] else "NON-CANONICAL"
    print(f"\n{verdict}: {sorted(d['input'] for d in report['protocol']['deviations']) or 'no'} "
          f"deviation(s) from the registered protocol")
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
