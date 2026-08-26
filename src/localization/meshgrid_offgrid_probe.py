"""exp_22 R1 controls -- the off-grid truth probe and the AGREE calibration (§2).

Two of the §2 controls cannot be read off the published rows, because both need
generations the registered pass deliberately never made. They live here, in one
tool, because they share every input: the same sixteen queries, the same frozen
stack, the same observation embedding.

**Off-grid truth probe.** For exactly the lexicographically first query of each
of the sixteen included rooms, generate ``K`` RIRs at the CONTINUOUS ground-truth
source position ``x*_s`` -- the position the half-metre lattice can only
approximate -- score them against the observation and report where that score
would have ranked among the query's grid candidates. It answers one question:
how much of the residual error is the grid's quantization rather than the
model's.

This control READS THE GROUND TRUTH, by design and by registration. That is
legitimate here and nowhere else in exp_22: the truth position is used only to
place a generation whose score is REPORTED, never to place a candidate. It never
enters any candidate set, any argmax, any prediction or any published
localization metric -- see :data:`CONTROL_LABEL`, which is stamped into every
record and every artifact this module writes.

**Real-vs-generated AGREE calibration.** On the same sixteen queries, compare
``cos(E(h_obs), E(h_real, other))`` against ``cos(E(h_obs), E(h_generated))``.
The real bank is the query's own frozen D1 context RIRs: real measured RIRs of
the same room, at the same receiver, from other sources -- the only real-RIR bank
the registered pass materializes, and one whose identity is already pinned by the
D1 manifest's per-context sha256. If the generated distribution sits far below
the real one, the scorer is being asked to rank inside a domain gap, and that is
a property of the embedding rather than of the localization.

Nothing runs until the artifacts agree: the published run binding must hash to
its own content and must match, field by field, the binding this probe builds
from its own checkpoint, scorer, D1 manifest, G1 report and sampler settings.
The probe is otherwise a normal announcement-08 citizen: every generated
waveform is saved with its sha256 and a manifest.
"""
import argparse
import hashlib
import json
import os

from datetime import datetime, timezone

import numpy as np
import torch

from src.localization import meshgrid_engine as me
from src.localization import meshgrid_queries as mq
from src.localization import meshgrid_report as mr
from src.localization import scoring as sc
from src.localization.reaggregate import decode_scores

#: stamped into every record, every JSON and the markdown. The label is the
#: control's containment: it says what the probe is allowed to do with the truth.
CONTROL_LABEL = (
    "OFF-GRID TRUTH CONTROL -- this probe generates at the CONTINUOUS ground-truth source "
    "position x*_s and therefore READS THE HELD-OUT TARGET, by design and by registration "
    "(inherited plan §2). Its generation is NEVER inserted into any candidate set, never "
    "competes in any argmax, never becomes a prediction and never enters any published "
    "localization metric; it exists only to report how the truth position would have SCORED "
    "against the grid the engine actually searched")

CALIBRATION_LABEL = (
    "REAL-VS-GENERATED AGREE CALIBRATION -- cos(E(h_obs), E(h_real,other)) over the query's "
    "frozen D1 context RIRs (real measured RIRs of the same room, same receiver, other sources; "
    "their bytes are pinned by the D1 manifest's per-context sha256) against cos(E(h_obs), "
    "E(h_generated)) over the off-grid truth generations. Both distributions are reported; "
    "neither is a localization metric, and the comparison diagnoses the embedding's domain gap "
    "only")

#: the binding fields the probe must agree with the published run on.
#:
#: Everything the run binding pins EXCEPT ``dump_cases_sha256``: that field is
#: the localization pass's dump AUTHORITY, and this probe is not a localization
#: pass -- it dumps exactly the sixteen registered off-grid probe queries, which
#: the announcement-08 exemption names directly rather than through a case list.
PROBE_BINDING_FIELDS = tuple(field for field in me.RUN_BINDING_FIELDS
                             if field != "dump_cases_sha256")

#: the sentinel candidate index the off-grid draw is keyed with.
#:
#: Under the registered common-random-numbers policy the key does not depend on
#: the candidate at all, so the truth generation is drawn from EXACTLY the same K
#: latents as every grid candidate of that query -- which is what makes the rank
#: comparison a comparison between POSITIONS. A per-candidate policy has no key
#: for a point that is not a candidate, so it is refused rather than invented.
OFFGRID_CANDIDATE_SENTINEL = -1

WAVEFORM_DIRNAME = "waveforms"
PROBE_REPORT_JSON = "offgrid_probe_report.json"
PROBE_REPORT_MARKDOWN = "offgrid_probe_report.md"

#: What binds the probe's LIVE observation to the frozen grid rows -- and, just
#: as importantly, what does not.
#:
#: Surveyed across the published artifacts: the D1 record pins the eight CONTEXT
#: RIRs (``context_audio_sha256``, verified per query by
#: ``meshgrid_engine.verify_context_record``) and the context poses; the engine
#: row pins its own claims and the similarity sidecar (``sims_sha256``). NOTHING
#: digests the observed RIR -- not the manifest, not the row, not the binding
#: (checked over all 1,566 published rows). No field is invented here to pretend
#: otherwise (Codex r9i review, item 2).
#:
#: What IS available is a functional tie, and it is stronger than a digest of a
#: file nobody registered: one of the row's OWN stored similarities is
#: re-derived from the live observation, using the same keyed noise and the same
#: candidate, and must reproduce the frozen value. s[x, k] = cos(E(h_obs),
#: E(h_hat[x, k])), so a different observation moves it directly and by O(1),
#: while the only admissible difference is the batch-shape noise the engine
#: already registers a bound for.
OBSERVATION_BINDING_NOTE = (
    "no registered artifact digests the observed RIR: the D1 record pins the eight CONTEXT RIRs "
    "(verified per query by the engine's verify_context_record) and the row pins its own claims "
    "and its similarity sidecar, but the observation itself is pinned nowhere (Codex r9i review, "
    "item 2). Rather than invent a field, the live observation is tied to the frozen rows "
    "FUNCTIONALLY: one candidate of the query is regenerated from the same keyed noise and the "
    "same conditioning, and its cosine against the LIVE observation must reproduce the "
    "similarity the row already published for that candidate, to within the engine's registered "
    "SCORE_TOLERANCE plus the float16 half-ulp of the sidecar. Because s[x, k] = cos(E(h_obs), "
    "E(h_hat[x, k])), a substituted observation moves that number directly. What this pins: that "
    "the observation being scored here is the observation those rows were scored against. What "
    "it does NOT pin: the observation's provenance in any absolute sense -- its bytes are "
    "recorded below so a later round can pre-register them, exactly as the pair-metadata bank "
    "was")

#: the intent record that survives a hard crash.
PUBLICATION_JOURNAL = "offgrid_publication_journal.json"

JOURNAL_NOTE = (
    "an in-process rollback cannot survive SIGKILL, a power loss, or an interrupt landing in the "
    "gap between a rename and the bookkeeping that records it (Codex r9i review, item 3). So "
    "every intended rename is journalled and fsynced BEFORE the first one runs, and the journal "
    "is marked complete only after all of them have landed and been re-verified. A journal found "
    "incomplete at startup means a previous attempt died mid-move: every final it names is "
    "moved back to quarantine before anything else happens, so the directory returns to the one "
    "state a partial publication may leave behind")

#: the publish contract, stated inside the artifact that depends on it.
PUBLICATION_ORDER_NOTE = (
    "the manifest is written and fsynced BEFORE any dump leaves quarantine, so a crash can "
    "never leave a finalized file no manifest names; the dumps are then moved with every rename "
    "inside a rollback handler, so a failure mid-move returns the whole set to quarantine; and "
    "the manifest is rewritten once publication is complete and every file has been re-verified "
    "against the digest it was staged with. publication.completed = false therefore means the "
    "dumps are in quarantine (or the process died between the move and the rewrite, which "
    "verify_published_probe resolves), and true means the published set is complete")


# --------------------------------------------------------------------------- #
# the binding gate
# --------------------------------------------------------------------------- #
def assert_probe_binding(run_dir, binding, fields=PROBE_BINDING_FIELDS):
    """The probe continues the SAME experiment as the run it reports against.

    The published binding is recomputed from its own content first (so a hand-
    edited file cannot vouch for itself), then compared field by field against
    the binding this probe built from the checkpoint, the scorer, the D1 manifest,
    the G1 report and the sampler settings it was actually given.
    """
    published, published_sha = mr.load_published_binding(run_dir)
    differing = {}
    for field in fields:
        if field not in binding:
            raise ValueError(f"the probe binding is missing the registered field {field!r}; "
                             "every quantity that decides a score must be pinned before a "
                             "control is generated")
        if published.get(field) != binding.get(field):
            differing[field] = {"published": published.get(field), "probe": binding.get(field)}
    if differing:
        raise ValueError(
            f"the probe does not run the protocol the published run was scored under: "
            f"{sorted(differing)} differ (published binding {published_sha[:12]}...). "
            f"First mismatch: {sorted(differing)[0]} = "
            f"{differing[sorted(differing)[0]]!r}. A control generated under a different "
            "checkpoint, scorer, context draw, candidate manifest or sampler setting cannot be "
            "compared against the run's scores")
    return {"binding_sha256": published_sha, "fields_checked": list(fields),
            "published": published,
            "published_checked": {field: published[field] for field in fields}}


def assert_probe_run_census(run_dir, binding, binding_sha256, plan, records,
                            context_manifest, totals=None, single_shard=False,
                            expect_ckpt_sha256=None, allow_protocol_deviation=False):
    """The run this control ranks against IS the complete, merged, registered pass.

    Every shard of a run shares the strict binding digest, so the binding alone
    cannot tell a finished 5,337-query merge from one shard of it -- and a rank
    "of 5,337 queries" taken against a partial directory would be a different
    claim than the one the report makes (Codex r9 review, finding 4). The full
    artifact ladder the R1 report applies is applied here too, reusing it rather
    than re-deriving a weaker copy: the supplied D1/G1/room manifests must be the
    bound ones, the binding must be the registered protocol, the directory must
    carry its merge report, every row and sidecar must re-verify, the census must
    hold and the D1/G1/row identities must be one set.
    """
    artifacts = mr.assert_artifact_hashes(binding, plan, context_manifest)
    registered = mr.assert_registered_protocol(binding, expect_ckpt_sha256=expect_ckpt_sha256,
                                               allow_deviation=allow_protocol_deviation)
    rows = mr.verify_rows(run_dir, binding_sha256)
    # the receipt is checked against the ROWS here too. r9d handed
    # assert_merge_report derived=None from this path, so the control accepted a
    # receipt no row supported and never looked at the batching stamps at all
    # (Codex r9f review, B4) -- the one ladder, applied the one way.
    derived = mr.derive_run_facts(rows)
    batching = mr.assert_uniform_batching(rows, binding.get("advisory"))
    merge = (None if single_shard
             else mr.assert_merge_report(run_dir, binding, binding_sha256, plan, totals=totals,
                                         derived=derived))
    census = mr.assert_census(rows, records, totals=totals)
    mr.assert_row_protocol(rows, binding)
    identity_join = mr.assert_identity_join(plan, records, rows)
    return {"artifacts": artifacts, "registered_protocol": registered, "merge": merge,
            "derived": derived, "batching": batching,
            "census": census, "identity_join": identity_join,
            "single_shard": bool(single_shard),
            "single_shard_note": mr.SINGLE_SHARD_NOTE if single_shard else None}


def assert_registered_probe_set(probes, plan):
    """The probe set IS ``one lexicographically first query per included room``.

    ``registered_probe_queries`` derives it from the manifests; this asserts the
    derived set covers every audited room exactly once, so a probe cannot quietly
    run on fifteen rooms.
    """
    rooms = sorted(plan.rooms)
    if sorted(probes) != rooms:
        missing = sorted(set(rooms) - set(probes))
        extra = sorted(set(probes) - set(rooms))
        raise ValueError(f"the off-grid probe set does not cover the audited rooms: missing "
                         f"{missing}, unexpected {extra}; §2 registers exactly one probe query "
                         "per included room")
    identities = [probes[room] for room in rooms]
    if len(set(identities)) != len(identities):
        raise ValueError("the off-grid probe set names the same query for two rooms")
    return identities


# --------------------------------------------------------------------------- #
# the generation at the continuous truth
# --------------------------------------------------------------------------- #
def assert_offgrid_noise_policy(policy):
    """Only common random numbers can key a draw at a non-candidate point."""
    if str(policy) != me.REGISTERED_NOISE_POLICY:
        raise ValueError(
            f"the off-grid truth probe needs the registered noise policy "
            f"{me.REGISTERED_NOISE_POLICY!r}, not {policy!r}: under common random numbers the "
            "truth generation is drawn from exactly the K latents every grid candidate of that "
            "query was drawn from, which is what makes its score comparable to theirs. A "
            "per-candidate key has no value for a point that is not a candidate")
    return True


def truth_noise(seed, query_id, num_samples, latent_shape, policy=me.NOISE_KEY_POLICY,
                device="cpu"):
    """The ``[K, C, T]`` latent noise the truth position is generated from."""
    assert_offgrid_noise_policy(policy)
    block = me.noise_block(seed, query_id, [OFFGRID_CANDIDATE_SENTINEL], int(num_samples),
                           latent_shape, policy=policy, device=device)
    return block


def generate_at_truth(engine, md, receiver_xyz, truth_xyz, *, query_id, seed=me.SEED,
                      num_samples=me.NUM_SAMPLES, noise_policy=me.NOISE_KEY_POLICY,
                      source_chunk=1, context=None):
    """Generate ``K`` RIRs at the continuous truth -> ``[K, 1, T]`` waveforms.

    The conditioning is assembled through the engine's own two branches, so the
    truth generation differs from a candidate generation in exactly one input --
    the source pose -- and in nothing else. The truth is taken from the caller
    (the pair metadata), never from ``md``, so the loader item may stay guarded.
    """
    assert_offgrid_noise_policy(noise_policy)
    receiver = np.asarray(receiver_xyz, dtype=np.float64).reshape(3)
    truth = np.asarray(truth_xyz, dtype=np.float64).reshape(3)
    if not (np.isfinite(receiver).all() and np.isfinite(truth).all()):
        raise ValueError(f"{query_id}: the receiver and the continuous truth must be finite")
    position_cam = (truth - receiver).reshape(1, 3)

    context = (me.context_conditioning(engine.conditioner, md, engine.device)
               if context is None else context)
    source = me.source_conditioning(engine.conditioner, {"depth": md["depth"]}, position_cam,
                                    engine.device, chunk=int(source_chunk))
    noise = truth_noise(seed, query_id, num_samples, engine.latent_shape,
                        policy=noise_policy, device=engine.device)
    rows = torch.zeros(int(num_samples), dtype=torch.long)
    merged = me.expand_conditioning(context, source, rows, engine.device)
    latents = engine.sampler(noise, engine.cond_inputs_fn(merged))
    return engine.decoder(latents).clamp(-1.0, 1.0)


def observation_digests(obs_wav, source_path=None):
    """Record what the observation IS, so a later round can pre-register it."""
    tensor = torch.as_tensor(obs_wav).detach().cpu().float().contiguous()
    out = {"tensor_sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
           "shape": [int(v) for v in tensor.shape],
           "source_path": None if source_path is None else str(source_path),
           "source_sha256": None,
           "pinned": False,
           "note": OBSERVATION_BINDING_NOTE}
    if source_path and os.path.isfile(str(source_path)):
        out["source_sha256"] = me.file_sha256(str(source_path))
    return out


def observation_continuity_tolerance(stored):
    """The only admissible difference between the two derivations of one cosine.

    The engine's registered bound on two passes of the same protocol that differ
    only in batching, plus the half-ulp the float16 sidecar itself introduces.
    Both are registered constants, not numbers chosen here.
    """
    return float(me.SCORE_TOLERANCE) + mr.float16_half_ulp(np.asarray(stored,
                                                                      dtype=np.float16))


def assert_observation_continuity(engine, query, md, context, row, sims, obs_embedding, *,
                                  seed=me.SEED, num_samples=me.NUM_SAMPLES,
                                  noise_policy=me.NOISE_KEY_POLICY, source_chunk=1,
                                  aggregator=mr.HEADLINE_AGGREGATOR):
    """The live observation IS the one the frozen rows were scored against.

    One of the query's own candidates -- the row's headline prediction, so the
    check lands exactly where the result does -- is regenerated from the same
    keyed noise and the same conditioning, and its cosine against the LIVE
    observation must reproduce the similarity the row already published. See
    :data:`OBSERVATION_BINDING_NOTE` for why this is the binding rather than a
    digest, and for what it does and does not pin.
    """
    from src.localization.scoring import cosine_sims

    largest = str(max(int(k) for k in row["by_k"]))
    block = row["by_k"][largest]
    candidate_row = int(block["prediction_row"] if aggregator == "lme"
                        else block["mean_prediction_row"])
    candidate_index = int(row["candidate_indices"][candidate_row])
    num_samples = int(num_samples)

    stored = np.asarray(sims, dtype=np.float16)[candidate_row, :num_samples]
    coordinates = np.asarray(query.coordinates, dtype=np.float64)
    position_cam = (coordinates[candidate_row]
                    - np.asarray(query.receiver_xyz, dtype=np.float64)).reshape(1, 3)

    source = me.source_conditioning(engine.conditioner, {"depth": md["depth"]}, position_cam,
                                    engine.device, chunk=int(source_chunk))
    noise = me.noise_block(seed, query.query_id, [candidate_index], num_samples,
                           engine.latent_shape, policy=noise_policy, device=engine.device)
    merged = me.expand_conditioning(context, source,
                                    torch.zeros(num_samples, dtype=torch.long), engine.device)
    wavs = engine.decoder(engine.sampler(noise, engine.cond_inputs_fn(merged))).clamp(-1.0, 1.0)
    embeddings = torch.as_tensor(engine.embedder(wavs)).float().reshape(1, num_samples, -1)
    rederived = cosine_sims(torch.as_tensor(obs_embedding).float().reshape(-1),
                            embeddings)[0].double().numpy()

    delta = float(np.abs(rederived - stored.astype(np.float64)).max())
    tolerance = observation_continuity_tolerance(stored)
    verdict = {"ok": bool(delta <= tolerance), "max_abs_delta": delta,
               "tolerance": float(tolerance), "k": int(largest),
               "candidate_index": candidate_index, "candidate_row": candidate_row,
               "num_samples": num_samples,
               "stored": [float(v) for v in stored],
               "rederived": [float(v) for v in rederived],
               "note": OBSERVATION_BINDING_NOTE}
    if not verdict["ok"]:
        raise ValueError(
            f"{query.query_id}: the observation this control loaded does not reproduce the "
            f"similarities the frozen row published for candidate {candidate_index} -- max "
            f"|delta| {delta:.3g} against a tolerance of {tolerance:.3g} (the engine's "
            f"SCORE_TOLERANCE plus the sidecar's float16 half-ulp). s[x, k] = cos(E(h_obs), "
            "E(h_hat)), so the observation being scored here is not the observation those rows "
            f"were scored against. {OBSERVATION_BINDING_NOTE}")
    return verdict


def truth_scores(embedder, obs_embedding, waveforms, tau=me.TAU, prefixes=me.K_PREFIXES):
    """``(sims [1, K], {K: {lme, mean}})`` for the truth generations."""
    embeddings = embedder(waveforms)
    embeddings = torch.as_tensor(embeddings).float().reshape(1, int(waveforms.shape[0]), -1)
    sims = sc.cosine_sims(torch.as_tensor(obs_embedding).float().reshape(-1), embeddings)
    blocks = me.nested_scores(sims, tau=tau, prefixes=prefixes)
    return sims, {int(k): {"lme": float(block["scores"][0]),
                           "mean": float(block["mean_scores"][0])}
                  for k, block in blocks.items()}


def rank_against_grid(row, scores_by_k, aggregator=mr.HEADLINE_AGGREGATOR):
    """Where the truth's score sits among the query's GRID candidate scores.

    The grid scores are the row's own float32 ``scores_hex`` -- the published
    numbers, not a recomputation -- so the rank is against exactly what the
    engine ranked. ``rank = 1`` means the truth would have beaten every candidate.
    """
    key = "scores_hex" if aggregator == "lme" else "mean_scores_hex"
    out = {}
    for k, block in sorted(row["by_k"].items(), key=lambda item: int(item[0])):
        k = int(k)
        if k not in scores_by_k:
            continue
        grid = decode_scores(block[key]).double()
        score = float(scores_by_k[k][aggregator])
        n_better = int((grid > score).sum())
        n_tied = int((grid == score).sum())
        best = float(grid.max())
        out[k] = {
            "truth_score": score,
            "rank": n_better + 1,
            "n_candidates": int(grid.numel()),
            "n_grid_better": n_better,
            "n_grid_tied": n_tied,
            "percentile": float((grid < score).double().mean()),
            "best_grid_score": best,
            "truth_minus_best_grid": score - best,
            "grid_prediction_index": int(block["prediction_index"] if aggregator == "lme"
                                         else block["mean_prediction_index"]),
        }
    return out


def calibration_record(embedder, obs_embedding, context_audio, generated_waveforms):
    """The two cosine distributions of the §2 real-vs-generated calibration."""
    obs = torch.as_tensor(obs_embedding).float().reshape(-1)
    real_wavs = torch.as_tensor(context_audio).float()
    if real_wavs.ndim == 2:                                   # [N, T] -> [N, 1, T]
        real_wavs = real_wavs.unsqueeze(1)
    if real_wavs.ndim != 3 or real_wavs.shape[1] != 1:
        raise ValueError(f"the real context bank must be [N, 1, T], got "
                         f"{tuple(real_wavs.shape)}")
    real_emb = torch.as_tensor(embedder(real_wavs)).float()
    real = sc.cosine_sims(obs, real_emb.reshape(1, real_emb.shape[0], -1))[0]

    gen_emb = torch.as_tensor(embedder(generated_waveforms)).float()
    generated = sc.cosine_sims(obs, gen_emb.reshape(1, gen_emb.shape[0], -1))[0]
    real_summary, generated_summary = _distribution(real), _distribution(generated)
    return {"label": CALIBRATION_LABEL,
            "real": [float(v) for v in real],
            "generated": [float(v) for v in generated],
            "real_summary": real_summary,
            "generated_summary": generated_summary,
            # taken from the float64 summaries, so the gap is exactly the
            # difference of the two means the report publishes
            "gap_mean_real_minus_generated": float(real_summary["mean"]
                                                   - generated_summary["mean"])}


def _distribution(values):
    array = np.asarray([float(v) for v in values], dtype=np.float64)
    if array.size == 0:
        raise ValueError("a calibration distribution must be non-empty")
    return {"n": int(array.size), "mean": float(array.mean()),
            "sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
            "min": float(array.min()), "median": float(np.median(array)),
            "max": float(array.max())}


# --------------------------------------------------------------------------- #
# artifacts (announcement 08)
# --------------------------------------------------------------------------- #
#: where a query's dump waits until the whole control has succeeded.
WAVEFORM_STAGING_DIRNAME = os.path.join(WAVEFORM_DIRNAME, ".partial")

WAVEFORM_NOTE = ("off-grid truth generations [K, 1, T], the observation and the real context "
                 "bank they are calibrated against (announcement 08 exp_22 exemption: the "
                 "sixteen registered probe queries)")

#: what a dump says about its own standing when the caller did not say.
CANONICAL_STATUS_UNKNOWN = ("status not declared by the caller; consult the probe report's "
                            "canonical_status")
CANONICAL_STATUS_CANONICAL = "CANONICAL: every registered gate of the run and this control passed"
CANONICAL_STATUS_NON_CANONICAL = (
    "NON-CANONICAL: a gate was relaxed or unmet (see canonical_status in the probe report); "
    "these generations are a diagnostic and may not be quoted as the registered result")


def write_probe_waveforms(out_dir, room_id, position, waveforms, observation, context_audio,
                          truth_xyz, receiver_xyz, query_id=None, status_label=None):
    """One probe query's dump, STAGED with its digest and its own labels.

    Staged, not published: a dump finalized the moment its query finished would
    survive a later failure as an unmanifested file holding generations at the
    ground truth (Codex r9 review, finding 9). :func:`publish_probe_waveforms`
    moves the whole set into place only after all sixteen have succeeded and the
    manifest exists; anything left behind stays under ``waveforms/.partial/``,
    quarantined and obviously incomplete.

    The labels travel INSIDE the npz as well, because a waveform file read on its
    own -- which is exactly how a dump gets used -- would otherwise carry
    generations at the held-out truth with nothing saying so.
    """
    staging = os.path.join(str(out_dir), WAVEFORM_STAGING_DIRNAME)
    os.makedirs(staging, exist_ok=True)
    name = f"offgrid_{me.room_stem(room_id)}_q{int(position):05d}.npz"
    path = os.path.join(staging, name)
    tmp = path + ".tmp"
    with open(tmp, "wb") as handle:
        np.savez(handle,
                 waveforms=np.asarray(torch.as_tensor(waveforms).detach().cpu().numpy(),
                                      dtype=np.float32),
                 observation=np.asarray(torch.as_tensor(observation).detach().cpu()
                                        .reshape(-1).numpy(), dtype=np.float32),
                 context_audio=np.asarray(torch.as_tensor(context_audio).detach().cpu()
                                          .numpy(), dtype=np.float32),
                 truth_xyz=np.asarray(truth_xyz, dtype=np.float64).reshape(3),
                 receiver_xyz=np.asarray(receiver_xyz, dtype=np.float64).reshape(3),
                 query_id=np.array(str(query_id or "")),
                 room_id=np.array(str(room_id)),
                 control_label=np.array(CONTROL_LABEL),
                 calibration_label=np.array(CALIBRATION_LABEL),
                 subset=np.array(mr.SUBSET_LABEL),
                 agree_leakage_caveat=np.array(me.AGREE_LEAKAGE_CAVEAT),
                 scorer_readout_deviation=np.array(me.SCORER_READOUT_DEVIATION),
                 # a dump gets read on its own, so it carries the same
                 # disclosures the JSON does (Codex r9c review, disclosure minor)
                 latency_scope_note=np.array(mr.LATENCY_SCOPE_NOTE),
                 truth_binding_note=np.array(mr.TRUTH_BINDING_NOTE),
                 controls_elsewhere=np.array(json.dumps(mr.CONTROLS_ELSEWHERE,
                                                        sort_keys=True)),
                 sensitivity_status=np.array(str(status_label or CANONICAL_STATUS_UNKNOWN)),
                 waveform_note=np.array(WAVEFORM_NOTE))
    os.replace(tmp, path)
    return {"waveform_path": os.path.join(WAVEFORM_DIRNAME, name),
            "sensitivity_status": str(status_label or CANONICAL_STATUS_UNKNOWN),
            "waveform_staged_path": os.path.relpath(path, str(out_dir)),
            "waveform_sha256": me.file_sha256(path),
            "waveform_published": False,
            "waveform_note": WAVEFORM_NOTE}


def _fsync_dir(path):
    """Flush a directory entry, so a rename survives a crash. Best effort."""
    try:
        handle = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        os.fsync(handle)
        return True
    except OSError:
        return False
    finally:
        os.close(handle)


def _rollback_published(moved):
    """Put every already-moved dump back in quarantine, and make it durable.

    A half-moved set is the one state the publish contract forbids: either the
    quarantine holds everything, or the published set is complete (Codex r9c
    review, M9). Rolling a rename back is another rename, so the recovery is as
    reliable as the move was -- and each reversal is followed by a directory
    fsync, so a crash during the recovery cannot leave the reversal itself
    half-durable (Codex r9f review, M9).
    """
    for target, source in reversed(moved):
        try:
            os.makedirs(os.path.dirname(source), exist_ok=True)
            os.replace(target, source)
            _fsync_dir(os.path.dirname(source))
            _fsync_dir(os.path.dirname(target))
        except OSError:                       # noqa: PERF203 -- best effort by design
            pass
    return len(moved)


def publish_probe_waveforms(out_dir, records):
    """Move every staged dump into place -- only once the control is complete.

    The digest was taken at staging time and the bytes do not change, so the
    published file's sha256 is the one the manifest already records. Any failure
    rolls every completed move back into quarantine, so the directory is never
    left holding a partial published set.
    """
    # the crash-safe half: an fsynced statement of every rename about to happen,
    # written before the first one (Codex r9i review, item 3)
    write_publication_journal(out_dir, records)
    published, moved = [], []
    # EVERY step of the loop is inside the handler, the rename included. r9d left
    # os.replace outside it, so a rename that failed part-way through -- a full
    # disk, a permission change, a cross-device target -- kept the already-moved
    # subset published (Codex r9f review, M9). The journal covers what no handler
    # can: the process not reaching the handler at all.
    try:
        for record in records:
            source = os.path.join(str(out_dir), record["waveform_staged_path"])
            target = os.path.join(str(out_dir), record["waveform_path"])
            if not os.path.isfile(source):
                raise ValueError(f"{record['query_id']}: the staged dump {source!r} is gone; "
                                 "the control may not publish a manifest naming a file it "
                                 "cannot move")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            os.replace(source, target)
            moved.append((target, source))
            _fsync_dir(os.path.dirname(target))
            if me.file_sha256(target) != record["waveform_sha256"]:
                raise ValueError(f"{record['query_id']}: the published dump does not match the "
                                 "digest recorded at staging time")
            record["waveform_published"] = True
            published.append(record["waveform_path"])
        complete_publication_journal(out_dir, len(published))
    except BaseException:
        # BaseException, not Exception: a KeyboardInterrupt mid-loop must not be
        # the one way to leave a partial published set behind
        for record in records:
            record["waveform_published"] = False
        _rollback_published(moved)
        raise
    staging = os.path.join(str(out_dir), WAVEFORM_STAGING_DIRNAME)
    if os.path.isdir(staging) and not os.listdir(staging):
        os.rmdir(staging)
    return published


def journal_path(out_dir):
    return os.path.join(str(out_dir), PUBLICATION_JOURNAL)


def write_publication_journal(out_dir, records):
    """State every intended rename, durably, BEFORE the first one happens."""
    from datetime import datetime, timezone

    payload = {
        "experiment": "exp_22 loc_meshgrid off-grid publication journal",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "completed": False,
        "note": JOURNAL_NOTE,
        "moves": [{"query_id": record["query_id"],
                   "final": record["waveform_path"],
                   "staged": record["waveform_staged_path"],
                   "sha256": record["waveform_sha256"]}
                  for record in records],
    }
    return _write_json_durably(journal_path(out_dir), payload)


def complete_publication_journal(out_dir, n_published):
    """Mark the journal complete -- only after every move landed and verified."""
    path = journal_path(out_dir)
    with open(path) as handle:
        payload = json.load(handle)
    payload.update({"completed": True, "n_published": int(n_published)})
    return _write_json_durably(path, payload)


def recover_publication(out_dir):
    """Quarantine the finals of an interrupted publication, before anything else.

    The one state a partial publication may leave behind is "everything is in
    quarantine". A journal that exists and is not complete says a previous
    attempt died between the first rename and the last verification, so every
    final it names is moved back to its staged path -- and only then may a caller
    proceed. A complete journal, or none at all, is a no-op.
    """
    path = journal_path(out_dir)
    if not os.path.isfile(path):
        return {"recovered": False, "reason": "no journal", "n_quarantined": 0}
    with open(path) as handle:
        payload = json.load(handle)
    if payload.get("completed"):
        return {"recovered": False, "reason": "the journal is complete", "n_quarantined": 0}

    quarantined, missing = [], []
    for move in payload.get("moves") or []:
        final = os.path.join(str(out_dir), move["final"])
        staged = os.path.join(str(out_dir), move["staged"])
        if not os.path.isfile(final):
            if not os.path.isfile(staged):
                missing.append(move["final"])
            continue
        os.makedirs(os.path.dirname(staged), exist_ok=True)
        os.replace(final, staged)
        _fsync_dir(os.path.dirname(staged))
        _fsync_dir(os.path.dirname(final))
        quarantined.append(move["final"])
    os.remove(path)
    _fsync_dir(os.path.dirname(os.path.abspath(path)))
    return {"recovered": True, "reason": "an interrupted publication was rolled back",
            "n_quarantined": len(quarantined), "quarantined": quarantined,
            "missing": missing, "note": JOURNAL_NOTE}


#: what the publication path needs out of the verified gate. r9d hand-listed a
#: subset here and dropped ``metadata_bank_expected``, so the JSON and the
#: Markdown always said non-canonical while the NPZ labels said canonical
#: (Codex r9f review). One definition now, and a test pins that it carries
#: everything ``probe_canonical_status`` reads.
PUBLICATION_GATE_FIELDS = ("census", "identity_join", "merge", "derived", "batching",
                           "single_shard", "single_shard_note", "registered_protocol",
                           "metadata_bank", "metadata_bank_sha256", "metadata_bank_expected",
                           "observation_continuity",
                           "non_canonical", "non_canonical_declared")


def publication_gate(gate):
    """The verified gate, sliced for publication without losing a verdict."""
    return {field: gate[field] for field in PUBLICATION_GATE_FIELDS if field in gate}


def probe_canonical_status(gate):
    """Whether this control may be quoted as the registered off-grid result.

    Mirrors ``meshgrid_report.canonical_status``: one authority, read by the
    JSON, the Markdown and the embedded NPZ label alike.
    """
    gate = gate or {}
    reasons = []
    if gate.get("single_shard"):
        reasons.append({"gate": "merge_report",
                        "why": "the run this control ranks against publishes no census-gated "
                               "merge receipt",
                        "note": mr.SINGLE_SHARD_NOTE})
    registered = gate.get("registered_protocol") or {}
    if registered and not registered.get("is_registered", True):
        reasons.append({"gate": "registered_protocol",
                        "why": f"the run binding deviates on "
                               f"{sorted(registered.get('deviations') or {})}",
                        "note": mr.CKPT_SHA256_NOTE})
    if not gate.get("metadata_bank_expected"):
        reasons.append({"gate": "metadata_bank",
                        "why": "no pre-registered pair-metadata bank digest was supplied",
                        "note": mr.METADATA_BANK_PREREGISTRATION_NOTE})
    if gate.get("observation_continuity") and not gate["observation_continuity"].get("ok", True):
        reasons.append({"gate": "observation_continuity",
                        "why": gate["observation_continuity"].get(
                            "why", "the live observation could not be tied to the frozen rows"),
                        "note": OBSERVATION_BINDING_NOTE})
    # The operator's own declaration. r9d propagated it and r9g put it in the
    # publication slice, but nothing READ it, so a valid pin plus --non-canonical
    # produced canonical JSON/Markdown beside non-canonical NPZs (Codex r9i
    # review, item 4). Read explicitly, and then joined fail-closed below.
    if gate.get("non_canonical_declared"):
        reasons.append({"gate": "declared_non_canonical",
                        "why": "the operator ran this control with --non-canonical",
                        "note": mr.NON_CANONICAL_NOTE})
    # FAIL-CLOSED: whatever the derived flag says, the status may never come out
    # more canonical than it. A reason we failed to enumerate is still a reason.
    if gate.get("non_canonical") and not reasons:
        reasons.append({"gate": "non_canonical_flag",
                        "why": "the verified gate carries non_canonical = True without naming a "
                               "reason this function knows how to enumerate; the status refuses "
                               "to be more canonical than the gate it was handed",
                        "note": mr.NON_CANONICAL_NOTE})
    return {"canonical": not reasons, "reasons": reasons,
            "note": None if not reasons else mr.NON_CANONICAL_NOTE}


def write_probe_report(out_dir, records, binding, binding_sha256, provenance,
                       tau=me.TAU, prefixes=me.K_PREFIXES, gate=None):
    """Publish the probe's JSON + markdown, both stamped with every caveat.

    The order is the contract (Codex r9c review, M9): the summary is computed,
    then the JSON and the Markdown are written and flushed to disk, and only
    then do the staged dumps leave quarantine -- with any failure during the move
    rolling every completed one back. A crash therefore leaves either a
    quarantine with no manifest, or a manifest whose every named file is present
    and digest-verified; never a scatter of unmanifested finals.
    """
    os.makedirs(str(out_dir), exist_ok=True)
    records = list(records)
    # a previous attempt may have died mid-move; put its finals back in
    # quarantine before this one writes a thing (Codex r9i review, item 3)
    recovery = recover_publication(out_dir)
    # summarize FIRST: a refusal in here must not leave published dumps behind
    summary = summarize_probe(records, prefixes=prefixes)
    report = {
        "experiment": "exp_22 loc_meshgrid R1 off-grid truth probe + AGREE calibration",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "control_label": CONTROL_LABEL,
        "calibration_label": CALIBRATION_LABEL,
        "subset": mr.SUBSET_LABEL,
        "agree_leakage_caveat": me.AGREE_LEAKAGE_CAVEAT,
        "scorer_readout_deviation": me.SCORER_READOUT_DEVIATION,
        "batching_caveat": me.BATCHING_CAVEAT,
        "binding_sha256": binding_sha256,
        "binding": {field: binding[field] for field in PROBE_BINDING_FIELDS
                    if field in binding},
        "provenance": dict(provenance or {}),
        # the run-level gates this control was admitted under; None only when the
        # caller ran the pass directly, which the CLI never does
        "run_gate": gate,
        "single_shard": bool((gate or {}).get("single_shard")),
        "single_shard_note": (gate or {}).get("single_shard_note"),
        "canonical_status": probe_canonical_status(gate),
        # every artifact of this round carries the same disclosures, so an
        # off-grid output read on its own cannot lose them (r9 finding 10)
        "latency_scope_note": mr.LATENCY_SCOPE_NOTE,
        "truth_binding_note": mr.TRUTH_BINDING_NOTE,
        "controls_elsewhere": mr.CONTROLS_ELSEWHERE,
        "protocol": {"tau": float(tau), "k_prefixes": [int(k) for k in prefixes],
                     "noise_policy": me.REGISTERED_NOISE_POLICY,
                     "noise_note": "the truth generation is keyed by the query, not by a "
                                   "candidate, so it is drawn from exactly the K latents every "
                                   "grid candidate of that query was drawn from"},
        "n_queries": len(records),
        "records": list(records),
        "summary": summary,
        "publication": {"completed": False, "note": PUBLICATION_ORDER_NOTE,
                        "journal_note": JOURNAL_NOTE, "recovery": recovery},
    }
    path = os.path.join(str(out_dir), PROBE_REPORT_JSON)
    markdown = os.path.join(str(out_dir), PROBE_REPORT_MARKDOWN)

    # pass 1 -- the SAFETY NET. Every file is named with the digest it will have,
    # and publication is honestly recorded as not yet done, so a crash during the
    # move leaves a manifest that says so rather than one that lies.
    _write_json_durably(path, mr.jsonable(report))
    _write_text_durably(markdown, render_markdown(report))

    # the manifest is on disk and fsynced; only now do the finals leave quarantine
    publish_probe_waveforms(out_dir, records)
    verified = verify_published_probe(out_dir, records)

    # pass 2 -- the COMPLETION RECORD. r9d serialized the records before the
    # publication flag was set, so a successful run persisted
    # waveform_published=false (Codex r9f review, nit). The same payload is
    # rewritten once publication is complete and verified.
    report["records"] = list(records)
    report["publication"] = {"completed": True, "n_published": verified["n_published"],
                             "verified": True, "note": PUBLICATION_ORDER_NOTE,
                             "journal_note": JOURNAL_NOTE, "recovery": recovery}
    _write_json_durably(path, mr.jsonable(report))
    _write_text_durably(markdown, render_markdown(report))
    return {"json": path, "markdown": markdown, "publication": report["publication"],
            "sha256": {"json": me.file_sha256(path), "markdown": me.file_sha256(markdown)}}


def _fsync_path(path):
    """Flush a written file, and the directory entry that names it, to disk."""
    with open(path, "rb") as handle:
        os.fsync(handle.fileno())
    directory = os.open(os.path.dirname(os.path.abspath(path)), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return path


def _write_json_durably(path, payload):
    """``meshgrid_engine.write_json`` plus the fsync the publish order needs."""
    me.write_json(path, payload)
    return _fsync_path(path)


def _write_text_durably(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return _fsync_path(path)


def verify_published_probe(out_dir, records, recover=False):
    """Every file the manifest names is present, and is the file it names.

    ``recover=True`` makes this a startup path: an incomplete journal is rolled
    back into quarantine first, so a caller that verifies after a crash is told
    the truth about a quarantined set rather than about a half-published one.
    """
    if recover:
        recover_publication(out_dir)
    missing, wrong = [], []
    for record in records:
        path = os.path.join(str(out_dir), record["waveform_path"])
        if not os.path.isfile(path):
            missing.append(record["waveform_path"])
        elif me.file_sha256(path) != record["waveform_sha256"]:
            wrong.append(record["waveform_path"])
    if missing or wrong:
        raise ValueError(
            f"the published dump set does not match the manifest just written: {len(missing)} "
            f"file(s) missing (first {missing[:3]}) and {len(wrong)} with a different digest "
            f"(first {wrong[:3]}); the control publishes a complete set or none")
    return {"n_published": len(records), "verified": True}


def summarize_probe(records, prefixes=me.K_PREFIXES):
    """Rank and calibration distributions over the sixteen probe queries."""
    records = list(records)
    if not records:
        raise ValueError("the probe summary needs at least one record")
    by_k = {}
    for k in (int(p) for p in prefixes):
        ranks = np.asarray([record["rank_lme"][str(k)]["rank"] for record in records],
                           dtype=np.float64)
        deltas = np.asarray([record["rank_lme"][str(k)]["truth_minus_best_grid"]
                             for record in records], dtype=np.float64)
        percentiles = np.asarray([record["rank_lme"][str(k)]["percentile"]
                                  for record in records], dtype=np.float64)
        # rank 1 covers both "beat everything" and "tied the best", and those are
        # different claims: a tie means the truth position scored no better than
        # some grid candidate (Codex r9 review, finding 10)
        ties = np.asarray([record["rank_lme"][str(k)]["n_grid_tied"] for record in records],
                          dtype=np.int64)
        strictly_better = int(((ranks == 1.0) & (ties == 0)).sum())
        by_k[str(k)] = {
            "n_queries": len(records),
            "rank": {"mean": float(ranks.mean()), "median": float(np.median(ranks)),
                     "min": float(ranks.min()), "max": float(ranks.max())},
            "n_truth_beats_every_candidate": strictly_better,
            "n_truth_ties_the_best": int(((ranks == 1.0) & (ties > 0)).sum()),
            "n_rank_one": int((ranks == 1.0).sum()),
            "rank_one_note": "rank 1 means no grid candidate scored HIGHER; "
                             "n_truth_beats_every_candidate counts only the strictly better "
                             "cases, and n_truth_ties_the_best the rest",
            "truth_minus_best_grid": {"mean": float(deltas.mean()),
                                      "median": float(np.median(deltas)),
                                      "min": float(deltas.min()), "max": float(deltas.max())},
            "percentile": {"mean": float(percentiles.mean()),
                           "median": float(np.median(percentiles))}}
    real = np.concatenate([np.asarray(record["calibration"]["real"], dtype=np.float64)
                           for record in records])
    generated = np.concatenate([np.asarray(record["calibration"]["generated"],
                                           dtype=np.float64) for record in records])
    return {"by_k": by_k,
            "calibration": {"real": _distribution(real), "generated": _distribution(generated),
                            "gap_mean_real_minus_generated": float(real.mean()
                                                                   - generated.mean()),
                            "label": CALIBRATION_LABEL}}


def render_markdown(report):
    """A short human-readable summary; the JSON carries everything."""
    lines = ["# exp_22 R1 — off-grid truth probe + real-vs-generated AGREE calibration", ""]
    lines.append(f"Generated {report['created_utc']}.")
    lines.append("")
    lines.append(f"> **{report['control_label']}**")
    lines.append("")
    lines.append(f"- **Scope:** {report['subset']}")
    lines.append(f"- **Run binding:** `{report['binding_sha256']}`")
    lines.append(f"- **AGREE leakage caveat:** {report['agree_leakage_caveat']}")
    lines.append(f"- **Scorer readout deviation:** {report['scorer_readout_deviation']}")
    lines.append(f"- **Truth binding:** {report['truth_binding_note']}")
    lines.append(f"- **Latency scope:** {report['latency_scope_note']}")
    lines.append("")
    status = report.get("canonical_status") or {}
    if status.get("reasons"):
        lines.append(f"> **{status['note']}**")
        lines.append(">")
        for reason in status["reasons"]:
            lines.append(f"> - `{reason['gate']}` — {reason['why']}")
        lines.append("")
    if report.get("single_shard"):
        lines.append(f"> **{report['single_shard_note']}**")
        lines.append("")
    lines.append("## Off-grid truth rank against the grid (log-mean-exp)")
    lines.append("")
    lines.append("| K | median rank | min | max | truth strictly beats every candidate | "
                 "ties the best | median (truth − best grid) |")
    lines.append("|---|---|---|---|---|---|---|")
    for k, block in sorted(report["summary"]["by_k"].items(), key=lambda item: int(item[0])):
        lines.append(f"| {k} | {mr.format_number(block['rank']['median'], 1)} | "
                     f"{mr.format_number(block['rank']['min'], 0)} | "
                     f"{mr.format_number(block['rank']['max'], 0)} | "
                     f"{block['n_truth_beats_every_candidate']}/{block['n_queries']} | "
                     f"{block['n_truth_ties_the_best']}/{block['n_queries']} | "
                     f"{mr.format_number(block['truth_minus_best_grid']['median'], 5)} |")
    lines.append("")
    lines.append(f"_{report['summary']['by_k'][str(max(int(k) for k in report['summary']['by_k']))]['rank_one_note']}_")
    lines.append("")
    lines.append("## Real vs generated AGREE cosine")
    lines.append("")
    lines.append(f"> {report['summary']['calibration']['label']}")
    lines.append("")
    calibration = report["summary"]["calibration"]
    lines.append("| bank | n | mean | sd | min | median | max |")
    lines.append("|---|---|---|---|---|---|---|")
    for name in ("real", "generated"):
        block = calibration[name]
        lines.append(f"| {name} | {block['n']} | {mr.format_number(block['mean'], 4)} | "
                     f"{mr.format_number(block['sd'], 4)} | {mr.format_number(block['min'], 4)} | "
                     f"{mr.format_number(block['median'], 4)} | {mr.format_number(block['max'], 4)} |")
    lines.append("")
    lines.append(f"Mean gap (real − generated): "
                 f"{mr.format_number(calibration['gap_mean_real_minus_generated'], 4)}")
    lines.append("")
    largest_k = max(int(k) for k in report["protocol"]["k_prefixes"])
    lines.append("## Per-query")
    lines.append("")
    lines.append(f"| room | query | e_oracle (m) | rank @K={largest_k} | "
                 f"truth − best grid @K={largest_k} | mean real cos | mean generated cos |")
    lines.append("|---|---|---|---|---|---|---|")
    for record in report["records"]:
        largest = str(max(int(k) for k in record["rank_lme"]))
        block = record["rank_lme"][largest]
        lines.append(
            f"| {record['room_id']} | `{record['query_id'].split('|')[0]}` | "
            f"{mr.format_number(record['e_oracle'], 3)} | {block['rank']} | "
            f"{mr.format_number(block['truth_minus_best_grid'], 5)} | "
            f"{mr.format_number(record['calibration']['real_summary']['mean'], 4)} | "
            f"{mr.format_number(record['calibration']['generated_summary']['mean'], 4)} |")
    lines.append("")
    lines.append("## §2 controls that are NOT in this report")
    lines.append("")
    for name, where in sorted(report["controls_elsewhere"].items()):
        lines.append(f"- **{name}** — {where}")
    lines.append("")
    lines.append(f"_Latency scope:_ {report['latency_scope_note']}")
    lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# the pass
# --------------------------------------------------------------------------- #
def load_grid_row(run_dir, query, binding_sha256=None, binding=None):
    """The published row the truth score is ranked against, fully joined first.

    A generic digest check proves a row is intact; it does not prove it is THIS
    query's row. Rows are addressed by ``(room, position)``, so a same-binding row
    left at the expected path by another query would have silently supplied its
    grid scores to the rank (Codex r9 review, finding 4). The engine's own
    identity join answers that question -- it compares the row's query id,
    receiver, branch and full candidate index list against the G1 plan -- and the
    row's protocol is checked against the binding on top of it.
    """
    paths = me.query_artifact_paths(str(run_dir), query.room_id, int(query.position))
    verdict = me.verify_query_artifact(paths["row"], binding_sha256=binding_sha256)
    if not verdict["ok"]:
        raise ValueError(f"{query.room_id} q{int(query.position):05d}: the published row cannot "
                         f"be used as the probe's grid reference: {verdict['reason']}")
    me.assert_published_matches(str(run_dir), query, binding_sha256=binding_sha256)
    with open(paths["row"]) as handle:
        row = json.load(handle)
    if binding is not None:
        mr.assert_row_protocol([row], binding)
    return row


def run_probe(engine, stream, records, plan, run_dir, out_dir, *, metadata_root,
              binding_sha256=None, binding=None, seed=me.SEED, tau=me.TAU,
              num_samples=me.NUM_SAMPLES, prefixes=me.K_PREFIXES,
              noise_policy=me.NOISE_KEY_POLICY, source_chunk=1, non_canonical=None,
              verify_observation=True, on_record=None):
    """Walk the registered stream and run both controls on the sixteen queries.

    The stream is the released loader in D1 order and is walked ONCE, exactly as
    the scored pass walks it, because every query's context draw depends on the
    complete pass; only the sixteen registered positions are generated.
    """
    assert_offgrid_noise_policy(noise_policy)
    probes = me.registered_probe_queries(plan)
    assert_registered_probe_set(probes, plan)
    wanted = {probes[room]: room for room in probes}

    by_id = {record["query_id"]: record for record in records}
    by_position = {int(record["position"]): record for record in records}
    missing = sorted(query_id for query_id in wanted if query_id not in by_id)
    if missing:
        raise ValueError(f"the probe queries {missing[:3]} are not in the context manifest; the "
                         "probe set and the registered subset disagree")

    # the probe SET comes from the engine's registered rule and the query PLANS
    # are then looked up by identity. That reads every room manifest a second
    # time (~330 MB in production) rather than re-implementing the selection rule
    # here, where a second copy could drift from the registered one.
    query_plans = {}
    for room_id in sorted(plan.rooms):
        room_plan = me.load_room_plan(plan, room_id)
        query_id = probes[room_id]
        query_plans[query_id] = next(query for query in room_plan.queries
                                     if query.query_id == query_id)

    status_label = (CANONICAL_STATUS_UNKNOWN if non_canonical is None else
                    (CANONICAL_STATUS_NON_CANONICAL if non_canonical
                     else CANONICAL_STATUS_CANONICAL))
    resolver = mr.TruthResolver(metadata_root)
    out, seen = [], set()
    for position, (obs_wav, raw_md) in enumerate(stream):
        record = by_position.get(position)
        if record is None or record["query_id"] not in wanted:
            continue
        md = me.GuardedMetadata(raw_md)
        me.verify_context_record(md, record, position)
        query = query_plans[record["query_id"]]
        if obs_wav is None:
            raise ValueError(f"stream position {position}: the loader returned no observed "
                             "waveform; there is nothing to calibrate against")

        metadata_receiver, truth = resolver.resolve(record)
        mr.assert_receiver_matches(query.query_id, metadata_receiver, query.receiver_xyz)
        coordinates = np.asarray(query.coordinates, dtype=np.float64)
        distances = np.linalg.norm(coordinates - truth.reshape(1, 3), axis=1)
        e_oracle = float(distances.min())
        mr.assert_grid_oracle(query.query_id, coordinates, query.oracle, truth)
        # the injective check: this control HAS the stream, so it can compare the
        # truth as a VECTOR against the loader's own target instead of relying on
        # the scalar oracle, which two truths mirrored inside one lattice cell
        # would share (Codex r9 review, finding 3)
        truth_vector_drift = mr.assert_truth_vector(query.query_id, truth, query.receiver_xyz,
                                                    raw_md["source"])

        row = load_grid_row(run_dir, query, binding_sha256=binding_sha256, binding=binding)
        obs_embedding = torch.as_tensor(
            engine.embedder(torch.as_tensor(obs_wav).to(engine.device))
        )[0].float().cpu()

        # the query's context branch, computed once and reused by both the truth
        # generation and the observation-continuity check
        query_context = me.context_conditioning(engine.conditioner, md, engine.device)

        # BEFORE anything is scored: prove the observation just loaded is the one
        # these frozen rows were scored against (Codex r9i review, item 2)
        continuity = None
        if verify_observation:
            sims_path = os.path.join(str(run_dir), str(row["sims_path"]))
            continuity = assert_observation_continuity(
                engine, query, md, query_context, row, np.load(sims_path), obs_embedding,
                seed=seed, num_samples=num_samples, noise_policy=noise_policy,
                source_chunk=source_chunk)

        waveforms = generate_at_truth(engine, md, query.receiver_xyz, truth,
                                      query_id=query.query_id, seed=seed,
                                      num_samples=num_samples, noise_policy=noise_policy,
                                      source_chunk=source_chunk, context=query_context)
        sims, scores = truth_scores(engine.embedder, obs_embedding, waveforms, tau=tau,
                                    prefixes=prefixes)
        calibration = calibration_record(engine.embedder, obs_embedding,
                                         md["context_audio"], waveforms)
        dump = write_probe_waveforms(out_dir, query.room_id, query.position, waveforms,
                                     obs_wav, md["context_audio"], truth, query.receiver_xyz,
                                     query_id=query.query_id, status_label=status_label)

        record_out = {
            "control_label": CONTROL_LABEL,
            "query_id": query.query_id, "room_id": query.room_id,
            "position": int(query.position), "receiver_id": query.receiver_id,
            "receiver_xyz": [float(v) for v in query.receiver_xyz],
            "truth_xyz": [float(v) for v in truth],
            "truth_vector_drift_m": float(truth_vector_drift),
            "observation_continuity": continuity,
            "observation": observation_digests(obs_wav, raw_md.get("path")),
            "n_candidates": int(query.n_candidates), "num_samples": int(num_samples),
            "e_oracle": e_oracle,
            "truth_is_a_candidate": bool(distances.min() == 0.0),
            "truth_sims": [float(v) for v in sims.reshape(-1)],
            "truth_scores": {str(k): value for k, value in scores.items()},
            "rank_lme": {str(k): value for k, value in
                         rank_against_grid(row, scores, aggregator="lme").items()},
            "rank_mean": {str(k): value for k, value in
                          rank_against_grid(row, scores, aggregator="mean").items()},
            "calibration": calibration,
            "grid_row_sha256": row.get("row_sha256"),
            "grid_sims_sha256": row.get("sims_sha256"),
        }
        record_out.update(dump)
        out.append(record_out)
        seen.add(query.query_id)
        if on_record is not None:
            on_record(record_out)

    if verify_observation:
        deltas = [record["observation_continuity"]["max_abs_delta"] for record in out]
        continuity_summary = {"ok": True, "checked": len(out),
                              "max_abs_delta": (max(deltas) if deltas else 0.0),
                              "note": OBSERVATION_BINDING_NOTE}
    else:
        continuity_summary = {"ok": False, "checked": 0,
                              "why": "the observation-continuity check was disabled, so nothing "
                                     "ties the loaded observation to the frozen rows",
                              "note": OBSERVATION_BINDING_NOTE}
    for record in out:
        record["observation_continuity_summary"] = continuity_summary

    absent = sorted(set(wanted) - seen)
    if absent:
        # the staged dumps stay in waveforms/.partial/: quarantined and obviously
        # incomplete, never a finalized file no manifest names (r9 finding 9)
        raise ValueError(f"the stream ended before {len(absent)} probe queries were reached "
                         f"(first {absent[:3]}); a partial control may not be published. "
                         f"{len(out)} staged dump(s) remain under "
                         f"{WAVEFORM_STAGING_DIRNAME}/ and are not manifested")
    out.sort(key=lambda record: int(record["position"]))
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ckpt-path", default=None,
                        help="the frozen FLAC checkpoint the run was scored under; required "
                             "except in --print-metadata-bank-digest mode")
    parser.add_argument("--run-dir", default=None,
                        help="the MERGED I1 run directory the probe ranks against")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--model-config",
                        default=os.path.join("src", "configs", "model_configs", "FLAC", "AR",
                                             "FLAC_AR.json"))
    parser.add_argument("--dataset-config",
                        default=os.path.join("src", "configs", "dataset_configs", "AR", "eval",
                                             "acousticroom_unseeneval.json"))
    parser.add_argument("--context-manifest",
                        default=os.path.join("outputs_loc", "exp22",
                                             "d1_context_manifest.json"))
    parser.add_argument("--audit-report",
                        default=os.path.join("outputs_loc", "exp22", "g1_audit",
                                             "geometry_audit_report.json"))
    parser.add_argument("--metadata-root",
                        default=os.path.join("AcousticRooms", "metadata"))
    parser.add_argument("--agree-ckpt", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--branch", default=None)
    parser.add_argument("--cond-method", default="vanilla", choices=["vanilla", "fa_invariant"])
    parser.add_argument("--cond-autocast", default="default",
                        choices=["default", "bf16", "off"])
    parser.add_argument("--seed", type=int, default=me.SEED)
    parser.add_argument("--tau", type=float, default=me.TAU)
    parser.add_argument("--num-samples", type=int, default=me.NUM_SAMPLES)
    parser.add_argument("--k-prefixes", type=int, nargs="+", default=list(me.K_PREFIXES))
    parser.add_argument("--noise-policy", default=me.NOISE_KEY_POLICY,
                        choices=list(me.NOISE_KEY_POLICIES))
    parser.add_argument("--steps", type=int, default=me.STEPS)
    parser.add_argument("--cfg-scale", type=float, default=me.CFG_SCALE)
    parser.add_argument("--source-chunk", type=int, default=1)
    parser.add_argument("--single-shard", action="store_true",
                        help="rank against a directory that carries no merge_report.json. "
                             "Relaxes only the merge-only gates; the artifact-hash joins, the "
                             "row census, the identity join and every digest still apply, and "
                             "the control is stamped as non-canonical")
    parser.add_argument("--expect-ckpt-sha256", default=None,
                        help="enforce the run binding's ckpt_sha256 against this value")
    parser.add_argument("--allow-protocol-deviation", action="store_true",
                        help="run even though the run binding is not the registered protocol; "
                             "the artifacts are then stamped as a sensitivity check")
    parser.add_argument("--expect-metadata-bank-sha256", default=None,
                        help="the PRE-REGISTERED pair-metadata bank digest the continuous "
                             "truths must come out of. Required for a canonical control; "
                             "obtain it with --print-metadata-bank-digest and commit it before "
                             "any result exists")
    parser.add_argument("--non-canonical", action="store_true",
                        help="run without a pre-registered metadata-bank digest. "
                             "Trust-on-first-use is not a canonical mode, so the report, the "
                             "markdown and every NPZ are stamped NON-CANONICAL")
    parser.add_argument("--print-metadata-bank-digest", action="store_true",
                        help="PRE-REGISTRATION MODE: compute the pair-metadata bank digest from "
                             "--context-manifest and --metadata-root, print it and exit")
    # the probe registers no dump case list: announcement 08 names its sixteen
    # queries directly. Present so build_run_binding can be reused unchanged.
    parser.add_argument("--dump-cases-sha256", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _refuse(message):
    raise SystemExit(f"REFUSED: {message}")


def validate_args(args):
    """Startup refusals -- before a checkpoint is read or a GPU is touched."""
    if args.print_metadata_bank_digest:
        return True
    for name in ("ckpt_path", "run_dir", "out_dir"):
        if not getattr(args, name):
            _refuse(f"--{name.replace('_', '-')} is required to run the control")
    if not args.expect_metadata_bank_sha256 and not args.non_canonical:
        _refuse("a canonical off-grid control requires the PRE-REGISTERED pair-metadata bank "
                f"digest. {mr.METADATA_BANK_PREREGISTRATION_NOTE}. Pass --non-canonical to run "
                "a diagnostic instead")
    if args.noise_policy != me.REGISTERED_NOISE_POLICY:
        _refuse(f"--noise-policy {args.noise_policy!r} cannot key an off-grid draw; "
                f"{me.REGISTERED_NOISE_POLICY!r} is the registered policy and the only one "
                "under which the truth generation shares the grid candidates' latents")
    if args.cond_method != "vanilla":
        _refuse(f"cond_method={args.cond_method!r}: the registered exp_22 arm is vanilla")
    prefixes = [int(k) for k in args.k_prefixes]
    if sorted(set(prefixes)) != sorted(prefixes) or min(prefixes) < 1:
        _refuse(f"--k-prefixes must be distinct positive integers, got {prefixes}")
    if max(prefixes) != int(args.num_samples):
        _refuse(f"--num-samples must equal the largest prefix ({max(prefixes)}), not "
                f"{args.num_samples}: the prefixes are nested reads of ONE sequence")
    if float(args.tau) <= 0.0:
        _refuse(f"--tau must be > 0, got {args.tau}")
    if os.path.abspath(str(args.out_dir)) == os.path.abspath(str(args.run_dir)):
        _refuse("--out-dir may not be the scored run directory: a control never writes into "
                "the artifact set it reports against")
    return True


def _load_and_validate_checkpoint(args, model_config):
    """Read the checkpoint to CPU and validate it -- AFTER the artifact gates.

    Isolated in its own function because ``validate_checkpoint`` is the first
    thing in this tool that reaches ``eval_FLAC``, whose module-level function
    defaults call ``torch.cuda.is_available()``. No allocation happens, but the
    r9c review is right that a refused run should not have gone near the device
    layer at all, so nothing calls this until every gate has passed.
    """
    from localize_meshgrid import validate_checkpoint

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    validate_checkpoint(args, model_config, ckpt)
    return ckpt


def gate_run(args, model_config, agree_path, totals=None,
             require_manifest_census=True):
    """Every artifact gate, on CPU, BEFORE anything reaches a device.

    The r9 probe built its binding out of a LOADED AGREE model, which put the
    scorer on ``--device`` before the gate that decides whether this control may
    run at all (Codex r9 review, finding 5). Nothing here needs a device: the
    AGREE identity is a file digest, the checkpoint is read to CPU, and the whole
    artifact ladder -- audit chain, context manifest, binding, hash joins,
    registered protocol, merge report, row census, identity join -- is applied.
    Only after this returns does the caller load a model.

    ``totals`` and ``require_manifest_census`` exist for fixtures, exactly as
    ``evaluate_run``'s do, and are never passed by ``main``: a real run is always
    held to the registered census.
    """
    # localize_meshgrid itself imports only torch + src.localization, so the
    # binding builder is safe to reach for here; validate_checkpoint is NOT --
    # its body imports eval_localization -> eval_FLAC, whose function defaults
    # evaluate torch.cuda.is_available() at import time (Codex r9c review, B5).
    # It is therefore called only after every artifact gate has passed.
    from localize_meshgrid import build_run_binding

    plan = me.load_audit_plan(args.audit_report, branch=args.branch)
    manifest = mq.load_manifest(args.context_manifest,
                                require_census=require_manifest_census)
    binding = build_run_binding(args, plan, ckpt_sha256=me.file_sha256(args.ckpt_path),
                                agree_sha256=me.file_sha256(agree_path),
                                model_config_sha256=me.file_sha256(args.model_config))
    gate = assert_probe_binding(args.run_dir, binding)
    gate.update(assert_probe_run_census(
        args.run_dir, gate["published"], gate["binding_sha256"], plan,
        manifest["records"], args.context_manifest,
        totals=totals, single_shard=args.single_shard,
        expect_ckpt_sha256=args.expect_ckpt_sha256,
        allow_protocol_deviation=args.allow_protocol_deviation))
    # r9d only STORED the expected string here, so the control never read the
    # tree it takes its truths from and any nonempty value passed (Codex r9f
    # review, B3). The bank is computed over the same records the run is bound
    # to and compared against the pre-registered digest -- and it happens here,
    # before _load_and_validate_checkpoint, so a bank mismatch refuses without
    # eval_FLAC ever being imported (B5 ordering preserved).
    bank = mr.compute_metadata_bank_digest(args.context_manifest, args.metadata_root,
                                           records=manifest["records"])
    gate["metadata_bank"] = mr.assert_metadata_bank(
        bank["metadata_bank_sha256"], expected=args.expect_metadata_bank_sha256,
        allow_unpinned=args.non_canonical)
    gate["metadata_bank_sha256"] = bank["metadata_bank_sha256"]
    gate["metadata_bank_expected"] = args.expect_metadata_bank_sha256
    gate["non_canonical_declared"] = bool(args.non_canonical)
    gate["non_canonical"] = bool(args.non_canonical
                                 or not gate["metadata_bank"]["pinned"]
                                 or args.single_shard
                                 or not gate["registered_protocol"]["is_registered"])
    ckpt = _load_and_validate_checkpoint(args, model_config)
    return plan, manifest, binding, gate, ckpt


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    if args.print_metadata_bank_digest:
        verdict = mr.compute_metadata_bank_digest(args.context_manifest, args.metadata_root)
        print(json.dumps(mr.jsonable(verdict), indent=2, sort_keys=True))
        print(f"\nmetadata_bank_sha256 = {verdict['metadata_bank_sha256']}")
        print(f"\n{mr.METADATA_BANK_PREREGISTRATION_NOTE}")
        return 0
    print(f"{CONTROL_LABEL}\n")
    print(f"AGREE LEAKAGE CAVEAT: {me.AGREE_LEAKAGE_CAVEAT}")
    if args.single_shard:
        print(f"\n{mr.SINGLE_SHARD_NOTE}\n")

    # the driver's own item unpacker -- one implementation, not a second copy
    from localize_meshgrid import _iter_items as iter_stream_items

    with open(args.model_config) as handle:
        model_config = json.load(handle)
    # resolve the configured scorer only when the operator did not name one
    agree_path = args.agree_ckpt or \
        mq.with_resolved_agree(model_config)["training"]["metrics"]["AGREE_ckpt"]

    # EVERY gate first, on CPU. Nothing below this line may run if one refuses.
    plan, manifest, binding, gate, ckpt = gate_run(args, model_config, agree_path)
    records = manifest["records"]
    print(f"binding gate passed against {args.run_dir}: {gate['binding_sha256'][:12]}... "
          f"({len(gate['fields_checked'])} fields); run census "
          f"{gate['census']['n_queries']:,} queries / {gate['census']['n_rooms']} rooms, "
          f"identity join over {gate['identity_join']['n_queries']:,} identities")

    from src.localization.agree_embed import load_agree_audio

    agree = load_agree_audio(agree_path, args.device)
    engine, context = me.build_mesh_engine(
        args.ckpt_path, model_config, agree, device=args.device,
        cond_method=args.cond_method, cond_autocast=args.cond_autocast,
        steps=args.steps, cfg_scale=args.cfg_scale, ckpt=ckpt)
    print(f"weights: {context['weights_source']}, latent {context['latent_shape']}")

    loader, facts = mq.build_release_stack(args.dataset_config, args.model_config)
    me.assert_release_rng_state(manifest)
    print(f"release call graph reproduced: {facts['call_graph']}")

    def _announce(record):
        largest = str(max(int(k) for k in record["rank_lme"]))
        block = record["rank_lme"][largest]
        print(f"  {record['room_id']}: the truth position ranks {block['rank']} of "
              f"{block['n_candidates']} grid candidates at K={largest} "
              f"(truth - best grid = {block['truth_minus_best_grid']:+.5f})", flush=True)

    probe_records = run_probe(
        engine, iter_stream_items(loader), records, plan, args.run_dir, args.out_dir,
        metadata_root=args.metadata_root, binding_sha256=gate["binding_sha256"],
        binding=gate["published"], seed=args.seed, tau=args.tau,
        num_samples=args.num_samples,
        prefixes=tuple(int(k) for k in args.k_prefixes), noise_policy=args.noise_policy,
        source_chunk=args.source_chunk, non_canonical=gate["non_canonical"],
        on_record=_announce)
    gate["observation_continuity"] = (probe_records[0]["observation_continuity_summary"]
                                      if probe_records else
                                      {"ok": False, "checked": 0,
                                       "why": "no probe query was reached",
                                       "note": OBSERVATION_BINDING_NOTE})
    published = write_probe_report(args.out_dir, probe_records, binding,
                                   gate["binding_sha256"],
                                   provenance={"run_dir": str(args.run_dir),
                                               "audit_report": str(args.audit_report),
                                               "audit_report_sha256": plan.report_sha256,
                                               "context_manifest": str(args.context_manifest),
                                               "agree_ckpt": agree_path,
                                               "agree_ckpt_sha256": agree.ckpt_sha256,
                                               "device": str(args.device)},
                                   tau=args.tau,
                                   prefixes=tuple(int(k) for k in args.k_prefixes),
                                   gate=publication_gate(gate))
    status = json.load(open(published["json"]))["canonical_status"]
    print(f"\n{len(probe_records)} probe queries -> {published['json']}")
    print(f"  markdown -> {published['markdown']}")
    if status["canonical"]:
        print("  CANONICAL: every registered gate of the run and this control passed")
    else:
        print(f"  {status['note']}")
        for reason in status["reasons"]:
            print(f"    - {reason['gate']}: {reason['why']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
