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
    merge = (None if single_shard
             else mr.assert_merge_report(run_dir, binding, binding_sha256, plan, totals=totals))
    rows = mr.verify_rows(run_dir, binding_sha256)
    census = mr.assert_census(rows, records, totals=totals)
    mr.assert_row_protocol(rows, binding)
    identity_join = mr.assert_identity_join(plan, records, rows)
    return {"artifacts": artifacts, "registered_protocol": registered, "merge": merge,
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
                      source_chunk=1):
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

    context = me.context_conditioning(engine.conditioner, md, engine.device)
    source = me.source_conditioning(engine.conditioner, {"depth": md["depth"]}, position_cam,
                                    engine.device, chunk=int(source_chunk))
    noise = truth_noise(seed, query_id, num_samples, engine.latent_shape,
                        policy=noise_policy, device=engine.device)
    rows = torch.zeros(int(num_samples), dtype=torch.long)
    merged = me.expand_conditioning(context, source, rows, engine.device)
    latents = engine.sampler(noise, engine.cond_inputs_fn(merged))
    return engine.decoder(latents).clamp(-1.0, 1.0)


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


def write_probe_waveforms(out_dir, room_id, position, waveforms, observation, context_audio,
                          truth_xyz, receiver_xyz, query_id=None):
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
                 waveform_note=np.array(WAVEFORM_NOTE))
    os.replace(tmp, path)
    return {"waveform_path": os.path.join(WAVEFORM_DIRNAME, name),
            "waveform_staged_path": os.path.relpath(path, str(out_dir)),
            "waveform_sha256": me.file_sha256(path),
            "waveform_published": False,
            "waveform_note": WAVEFORM_NOTE}


def publish_probe_waveforms(out_dir, records):
    """Move every staged dump into place -- only once the control is complete.

    The digest was taken at staging time and the bytes do not change, so the
    published file's sha256 is the one the manifest already records.
    """
    published = []
    for record in records:
        source = os.path.join(str(out_dir), record["waveform_staged_path"])
        target = os.path.join(str(out_dir), record["waveform_path"])
        if not os.path.isfile(source):
            raise ValueError(f"{record['query_id']}: the staged dump {source!r} is gone; the "
                             "control may not publish a manifest naming a file it cannot move")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        os.replace(source, target)
        if me.file_sha256(target) != record["waveform_sha256"]:
            raise ValueError(f"{record['query_id']}: the published dump does not match the "
                             "digest recorded at staging time")
        record["waveform_published"] = True
        published.append(record["waveform_path"])
    staging = os.path.join(str(out_dir), WAVEFORM_STAGING_DIRNAME)
    if os.path.isdir(staging) and not os.listdir(staging):
        os.rmdir(staging)
    return published


def write_probe_report(out_dir, records, binding, binding_sha256, provenance,
                       tau=me.TAU, prefixes=me.K_PREFIXES, gate=None):
    """Publish the probe's JSON + markdown, both stamped with every caveat.

    The staged waveform dumps are moved into place FIRST and only here, so a
    manifest and its files appear together or not at all.
    """
    os.makedirs(str(out_dir), exist_ok=True)
    records = list(records)
    # summarize FIRST: a refusal in here must not leave published dumps behind
    summary = summarize_probe(records, prefixes=prefixes)
    publish_probe_waveforms(out_dir, records)
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
    }
    path = os.path.join(str(out_dir), PROBE_REPORT_JSON)
    me.write_json(path, mr.jsonable(report))
    markdown = os.path.join(str(out_dir), PROBE_REPORT_MARKDOWN)
    tmp = markdown + ".tmp"
    with open(tmp, "w") as handle:
        handle.write(render_markdown(report))
    os.replace(tmp, markdown)
    return {"json": path, "markdown": markdown,
            "sha256": {"json": me.file_sha256(path), "markdown": me.file_sha256(markdown)}}


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
              noise_policy=me.NOISE_KEY_POLICY, source_chunk=1, on_record=None):
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
        waveforms = generate_at_truth(engine, md, query.receiver_xyz, truth,
                                      query_id=query.query_id, seed=seed,
                                      num_samples=num_samples, noise_policy=noise_policy,
                                      source_chunk=source_chunk)
        sims, scores = truth_scores(engine.embedder, obs_embedding, waveforms, tau=tau,
                                    prefixes=prefixes)
        calibration = calibration_record(engine.embedder, obs_embedding,
                                         md["context_audio"], waveforms)
        dump = write_probe_waveforms(out_dir, query.room_id, query.position, waveforms,
                                     obs_wav, md["context_audio"], truth, query.receiver_xyz,
                                     query_id=query.query_id)

        record_out = {
            "control_label": CONTROL_LABEL,
            "query_id": query.query_id, "room_id": query.room_id,
            "position": int(query.position), "receiver_id": query.receiver_id,
            "receiver_xyz": [float(v) for v in query.receiver_xyz],
            "truth_xyz": [float(v) for v in truth],
            "truth_vector_drift_m": float(truth_vector_drift),
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
    parser.add_argument("--ckpt-path", required=True,
                        help="the frozen FLAC checkpoint the run was scored under")
    parser.add_argument("--run-dir", required=True,
                        help="the MERGED I1 run directory the probe ranks against")
    parser.add_argument("--out-dir", required=True)
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
    # the probe registers no dump case list: announcement 08 names its sixteen
    # queries directly. Present so build_run_binding can be reused unchanged.
    parser.add_argument("--dump-cases-sha256", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _refuse(message):
    raise SystemExit(f"REFUSED: {message}")


def validate_args(args):
    """Startup refusals -- before a checkpoint is read or a GPU is touched."""
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


def gate_run(args, model_config, agree_path, totals=None):
    """Every artifact gate, on CPU, BEFORE anything reaches a device.

    The r9 probe built its binding out of a LOADED AGREE model, which put the
    scorer on ``--device`` before the gate that decides whether this control may
    run at all (Codex r9 review, finding 5). Nothing here needs a device: the
    AGREE identity is a file digest, the checkpoint is read to CPU, and the whole
    artifact ladder -- audit chain, context manifest, binding, hash joins,
    registered protocol, merge report, row census, identity join -- is applied.
    Only after this returns does the caller load a model.

    ``totals`` exists for fixtures, exactly as ``evaluate_run``'s does, and is
    never passed by ``main``: a real run is always held to the registered census.
    """
    from localize_meshgrid import build_run_binding, validate_checkpoint

    plan = me.load_audit_plan(args.audit_report, branch=args.branch)
    manifest = mq.load_manifest(args.context_manifest)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    validate_checkpoint(args, model_config, ckpt)
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
    return plan, manifest, binding, gate, ckpt


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
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
        source_chunk=args.source_chunk, on_record=_announce)
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
                                   gate={key: gate[key] for key in
                                         ("census", "identity_join", "merge", "single_shard",
                                          "single_shard_note", "registered_protocol")})
    print(f"\n{len(probe_records)} probe queries -> {published['json']}")
    print(f"  markdown -> {published['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
