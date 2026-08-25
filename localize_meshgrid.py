#!/usr/bin/env python
"""exp_22 (loc_meshgrid): localize a hidden source on a mesh-valid 3-D grid.

For each held-out query RIR the driver takes the room's physically valid
half-metre candidate lattice (published by the G1 audit), regenerates one RIR
per candidate under the frozen Vanilla FLAC checkpoint and the frozen context
draw (published by the D1 manifest), scores each against the observed RIR in
AGREE's audio embedding space, and predicts the argmax candidate at K = 1, 4
and 8 from ONE nested sequence.

It is an evaluation driver only. Every protocol quantity it applies is
registered in ``worklog/worklog_yixun/exp_22_loc_meshgrid_claude/
loc_meshgrid_inherited_exp09_plan.md``, and the two recorded deviations from
that text -- the deterministic AGREE mean readout and the per-candidate noise
key -- are stamped into the run binding and into every published row.

Reuse boundary: the engine, caches, artifact codec and manifest verifiers live
in ``src.localization.meshgrid_engine``; the model build follows
``eval_FLAC.evaluate_model``'s lines of record through ``eval_localization``.
This file is the protocol wiring around them.

  # the registered pass (P1 arm, z_band branch chosen by the G1 audit)
  python localize_meshgrid.py \
      --ckpt-path weights/exp20/P1_40k.ckpt \
      --context-manifest outputs_loc/exp22/d1_context_manifest.json \
      --audit-report outputs_loc/exp22/g1_audit/geometry_audit_report.json \
      --out-dir outputs_loc/exp22/i1_P1 --device cuda:0

  # the pre-registered no-quality throughput probe (writes timings only).
  # --probe covers WHOLE receiver groups so the cache is amortized honestly, and
  # --probe-room keeps that affordable: the split's first room is Cafe, whose
  # smallest group is already ~9 queries x 5,295 candidates x 8 draws.
  python localize_meshgrid.py ... --probe 1 \
      --probe-room Bathrooms/Bathrooms_idx_14 --out-dir outputs_loc/exp22/i1_probe

  # the §1.5 cached-vs-uncached proof on the real conditioner, then stop
  python localize_meshgrid.py ... --cache-parity-check --device cuda:0

Both diagnostics modes publish no query artifacts and no run binding.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import torch

from src.localization import meshgrid_engine as me
from src.localization import meshgrid_queries as mq

DEFAULT_MODEL_CONFIG = os.path.join("src", "configs", "model_configs", "FLAC", "AR",
                                    "FLAC_AR.json")
DEFAULT_DATASET_CONFIG = os.path.join("src", "configs", "dataset_configs", "AR", "eval",
                                      "acousticroom_unseeneval.json")
DEFAULT_CONTEXT_MANIFEST = os.path.join("outputs_loc", "exp22", "d1_context_manifest.json")
DEFAULT_AUDIT_REPORT = os.path.join("outputs_loc", "exp22", "g1_audit",
                                    "geometry_audit_report.json")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ckpt-path", required=True,
                        help="the frozen FLAC checkpoint (wrapped; EMA resolved at load)")
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--context-manifest", default=DEFAULT_CONTEXT_MANIFEST,
                        help="the frozen D1 context manifest every arm shares")
    parser.add_argument("--audit-report", default=DEFAULT_AUDIT_REPORT,
                        help="the published G1 geometry audit report")
    parser.add_argument("--out-dir", default=os.path.join("outputs_loc", "exp22", "i1"))
    parser.add_argument("--agree-ckpt", default=None,
                        help="override the AGREE scorer; default = the model config's")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--branch", default=None,
                        help="assert the audit's branch; the branch itself is not selectable")
    parser.add_argument("--cond-method", default="vanilla", choices=["vanilla", "fa_invariant"])
    parser.add_argument("--cond-autocast", default="default", choices=["default", "bf16", "off"])
    parser.add_argument("--seed", type=int, default=me.SEED)
    parser.add_argument("--tau", type=float, default=me.TAU)
    parser.add_argument("--num-samples", type=int, default=me.NUM_SAMPLES)
    parser.add_argument("--k-prefixes", type=int, nargs="+", default=list(me.K_PREFIXES))
    parser.add_argument("--noise-policy", default=me.NOISE_KEY_POLICY,
                        choices=list(me.NOISE_KEY_POLICIES))
    parser.add_argument("--steps", type=int, default=me.STEPS)
    parser.add_argument("--cfg-scale", type=float, default=me.CFG_SCALE)
    parser.add_argument("--batch-rows", type=int, default=64,
                        help="generated rows per forward (candidates x K)")
    parser.add_argument("--source-chunk", type=int, default=me.SOURCE_CHUNK,
                        help="candidates per source-branch forward; each row is a full ViT "
                             "pass over a [3, 256, 512] map, so this is the memory knob")
    parser.add_argument("--resume", action="store_true",
                        help="skip queries whose published artifacts still verify")
    parser.add_argument("--probe", type=int, default=None,
                        help="no-quality throughput probe over whole receiver groups")
    parser.add_argument("--probe-room", default=None,
                        help="bound the probe to one room; the split's first room (Cafe) has "
                             "no receiver group small enough for a smoke")
    parser.add_argument("--probe-stem", default="probe")
    parser.add_argument("--dump-waveforms", nargs="*", default=None,
                        help="query ids to dump; bounded to the registered probe/case lists")
    parser.add_argument("--dump-top-n", type=int, default=me.DUMP_TOP_N,
                        help="best-scoring candidates kept in a dump, beyond the predictions")
    parser.add_argument("--dump-cases", default=None,
                        help="a registered visualization case list (JSON with query_ids)")
    parser.add_argument("--replay-check", action="store_true",
                        help="score one query twice at fixed batching and require bit-exact "
                             "agreement (the registered determinism claim), then exit")
    parser.add_argument("--cache-parity-check", action="store_true",
                        help="run the §1.5 cached-vs-uncached bit-identity proof and exit")
    return parser.parse_args(argv)


def _refuse(message):
    raise SystemExit(f"REFUSED: {message}")


def validate_args(args):
    """Startup refusals -- before a checkpoint is read or a GPU is touched."""
    if args.probe is None and args.probe_room:
        _refuse("--probe-room only bounds the throughput probe; pass --probe too")
    if args.probe is not None:
        if int(args.probe) < 1:
            _refuse("--probe must cover at least one query")
        if args.dump_waveforms:
            _refuse("the throughput probe is no-quality by protocol: it measures cost and "
                    "writes neither scores nor waveforms, so --dump-waveforms may not be "
                    "combined with --probe")
        if args.resume:
            _refuse("--resume continues a scored pass; the probe publishes no query artifacts")
    prefixes = [int(k) for k in args.k_prefixes]
    if sorted(set(prefixes)) != sorted(prefixes) or min(prefixes) < 1:
        _refuse(f"--k-prefixes must be distinct positive integers, got {prefixes}")
    if max(prefixes) != int(args.num_samples):
        _refuse(f"the registered protocol reads K in {prefixes} as nested prefixes of ONE "
                f"generated sequence, so --num-samples must equal the largest prefix "
                f"({max(prefixes)}), not {args.num_samples}")
    if args.noise_policy != me.REGISTERED_NOISE_POLICY:
        _refuse(f"--noise-policy {args.noise_policy!r} is not the registered draw: inherited "
                "plan §1.1 fixes common random numbers across a query's candidates "
                f"({me.REGISTERED_NOISE_POLICY!r}), so a score difference between two "
                "candidates is a difference between the candidates and not between their "
                "noise")
    if float(args.tau) <= 0.0:
        _refuse(f"--tau must be > 0, got {args.tau}")
    if args.cond_method != "vanilla":
        _refuse(f"cond_method={args.cond_method!r}: the registered exp_22 arm is vanilla; the "
                "frame-average arm needs §1.5's narrower context cache, which this engine "
                "refuses rather than mis-caches")
    return True


def validate_checkpoint(args, model_config, ckpt):
    """Every checkpoint refusal, BEFORE the scorer or the model is built.

    exp_18's r3 review finding 9: an ARE artifact used to be refused inside the
    engine build, i.e. after AGREE had already been constructed on the target
    device. Both refusals -- the ARE one and the conditioning binding -- are
    CPU-only reads of the file, so they belong here.
    """
    import copy

    from eval_localization import assert_no_are
    from src.localization.crossarm import cond_method_binding

    try:
        assert_no_are(ckpt.get("model_config"), copy.deepcopy(model_config))
    except SystemExit as error:
        _refuse(f"ARE check: {error}")
    binding = cond_method_binding(ckpt, args.cond_method)
    for reason in binding["reasons"]:
        _refuse(reason)
    return binding


def dump_allowance(args, plan):
    """The query ids this run may dump, computed ONLY if a dump was requested.

    ``registered_probe_queries`` reads every room's candidate manifest -- 328 MB
    of index lists -- so a run that dumps nothing must not pay for it.
    """
    if not args.dump_waveforms:
        return set()
    cases = me.load_dump_cases(args.dump_cases) if args.dump_cases else {"query_ids": []}
    allowed = set(me.registered_probe_queries(plan).values()) | set(cases["query_ids"])
    me.assert_dump_allowed(args.dump_waveforms, allowed)
    return {str(query) for query in args.dump_waveforms}


def assert_resumable(args, out_dir):
    """A resume CONTINUES a published pass; it may not create the thing that
    authorizes it (r7 review BLOCKER RESUME)."""
    path = os.path.join(str(out_dir), me.BINDING_FILENAME)
    if not os.path.isfile(path):
        _refuse(f"--resume was asked for but {path} does not exist. A resume continues a "
                "published pass; writing the binding first would legitimize whatever rows "
                "happen to be in that directory. Start the pass without --resume")
    return path


def writes_query_artifacts(args):
    """Only a scored pass claims the output directory.

    The throughput probe and the parity check are diagnostics: they must not
    publish a run binding into a directory a scored pass will later resume, and
    must not read one either.
    """
    return args.probe is None and not args.cache_parity_check and not args.replay_check


def build_run_binding(args, plan, ckpt_sha256, agree_sha256, model_config_sha256):
    """The seventeen registered quantities a resume must reproduce exactly."""
    return {
        "model_config_sha256": model_config_sha256,
        "ckpt_sha256": ckpt_sha256,
        "agree_ckpt_sha256": agree_sha256,
        "d1_manifest_sha256": me.file_sha256(args.context_manifest),
        "g1_report_sha256": plan.report_sha256,
        "room_manifest_sha256": {room: me.file_sha256(path)
                                 for room, path in plan.rooms.items()},
        "branch": plan.branch,
        "k_prefixes": [int(k) for k in args.k_prefixes],
        "num_samples": int(args.num_samples),
        "tau": float(args.tau),
        "seed": int(args.seed),
        "noise_policy": str(args.noise_policy),
        "steps": int(args.steps),
        "cfg_scale": float(args.cfg_scale),
        "cond_method": str(args.cond_method),
        "scorer_readout": me.SCORER_READOUT,
        "cond_autocast": str(args.cond_autocast),
        "dataset_config_sha256": me.file_sha256(args.dataset_config),
        "dataset_config": str(args.dataset_config),
    }


def _iter_items(loader):
    """Yield ``(reals_i, md_i)`` one query at a time from the batched loader."""
    for reals, metadata in loader:
        for index, md in enumerate(metadata):
            yield (None if reals is None else reals[index:index + 1]), md


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)

    with open(args.model_config) as handle:
        model_config = json.load(handle)
    resolved = mq.with_resolved_agree(model_config)
    agree_path = args.agree_ckpt or resolved["training"]["metrics"]["AGREE_ckpt"]

    plan = me.load_audit_plan(args.audit_report, branch=args.branch)
    manifest = mq.load_manifest(args.context_manifest)
    records = manifest["records"]
    print(f"G1 audit re-verified: {len(plan.rooms)} rooms, branch {plan.branch}, "
          f"{plan.n_queries} queries")
    print(f"D1 manifest: {len(records)} in-scope queries, filtered stream "
          f"{str(manifest.get('filtered_stream_sha256'))[:12]}...")
    print(f"AGREE scorer: {agree_path}\n  LEAKAGE CAVEAT: {me.AGREE_LEAKAGE_CAVEAT}")
    print(f"scorer readout: {me.SCORER_READOUT} -- DECLARED DEVIATION: "
          f"{me.SCORER_READOUT_DEVIATION}")
    print(f"noise key policy: {args.noise_policy}")

    # CPU-only validation first: no ARE artifact, and the conditioning method
    # is bound to the file itself -- both before anything reaches a device
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    binding_verdict = validate_checkpoint(args, model_config, ckpt)
    print(f"cond_method binding: {binding_verdict['binding']} "
          f"({binding_verdict.get('checkpoint_cond_method')!r})")

    from src.localization.agree_embed import load_agree_audio

    agree = load_agree_audio(agree_path, args.device)
    engine, context = me.build_mesh_engine(
        args.ckpt_path, model_config, agree, device=args.device,
        cond_method=args.cond_method, cond_autocast=args.cond_autocast,
        steps=args.steps, cfg_scale=args.cfg_scale, ckpt=ckpt)
    print(f"weights: {context['weights_source']}, latent {context['latent_shape']}")

    run_binding = build_run_binding(args, plan, ckpt_sha256=me.file_sha256(args.ckpt_path),
                                    agree_sha256=agree.ckpt_sha256,
                                    model_config_sha256=me.file_sha256(args.model_config))
    advisory = {"source_chunk": int(args.source_chunk), "batch_rows": int(args.batch_rows)}
    done = set()
    if writes_query_artifacts(args):
        if args.resume:
            assert_resumable(args, args.out_dir)
        if os.path.isfile(os.path.join(args.out_dir, me.BINDING_FILENAME)):
            moved = me.assert_binding(args.out_dir, run_binding, advisory=advisory)
            if moved is not True:
                me.record_advisory_change(args.out_dir, moved, advisory=advisory)
                print(f"NOTE: batching changed since the published pass: {moved}\n  "
                      f"recorded in {me.BINDING_FILENAME}; {me.BATCHING_CAVEAT}")
        else:
            me.write_binding(args.out_dir, run_binding, advisory=advisory)
        if args.resume:
            done, rejected = me.completed_queries(
                args.out_dir, binding_sha256=me.binding_sha256(run_binding))
            print(f"resume: {len(done)} verified queries skipped, {len(rejected)} rejected "
                  f"and regenerated")
            for verdict in rejected[:5]:
                print(f"  rejected {verdict['query_id']}: {verdict['reason']}")

    dump_queries = dump_allowance(args, plan)
    if dump_queries:
        print(f"bounded dumps admitted for {len(dump_queries)} queries "
              f"(announcement 08 exemption)\n  {me.DUMP_CONTENT_RULE}")

    # the released call graph, in order: seed -> loader -> metric stack -> iterator.
    # NOTHING may consume the global RNG between build_release_stack and the first batch.
    loader, facts = mq.build_release_stack(args.dataset_config, args.model_config)
    me.assert_release_rng_state(manifest)
    print(f"release call graph reproduced: {facts['call_graph']}, RNG state matches the "
          "D1 pass at iterator creation")

    if args.cache_parity_check:
        return _run_cache_parity(args, engine, plan, records, loader)
    if args.replay_check:
        return _run_replay_check(args, engine, plan, records, loader)

    progress = _progress_printer(len(records) - len(done))
    summary = me.run_pass(engine, _iter_items(loader), records, plan, args.out_dir,
                          on_row=progress,
                          seed=args.seed, tau=args.tau, num_samples=args.num_samples,
                          prefixes=tuple(int(k) for k in args.k_prefixes),
                          noise_policy=args.noise_policy, batch_rows=args.batch_rows,
                          source_chunk=args.source_chunk, done=done,
                          probe=args.probe, probe_room=args.probe_room,
                          dump_queries=dump_queries, dump_top_n=args.dump_top_n,
                          binding_sha256=me.binding_sha256(run_binding))
    summary["created_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary["binding_sha256"] = me.binding_sha256(run_binding)
    summary["agree_leakage_caveat"] = me.AGREE_LEAKAGE_CAVEAT
    summary["scorer_readout_deviation"] = me.SCORER_READOUT_DEVIATION
    summary["sims_precision_caveat"] = me.SIMS_PRECISION_CAVEAT

    if args.probe is not None:
        path = me.write_probe_records(args.out_dir, summary.pop("probe_records"),
                                      stem=args.probe_stem)
        print(f"probe: {summary['n_generated']} generated waveforms over "
              f"{summary['n_candidate_query_pairs']} candidate-query pairs; NO scores written")
        print(f"  timings (s): {summary['timings_s']}")
        print(f"  -> {path}")
        return 0

    summary.pop("probe_records", None)
    me.write_json(os.path.join(args.out_dir, "run_summary.json"), summary)
    print(f"scored {summary['n_scored']} queries ({summary['n_skipped']} skipped), "
          f"{summary['n_conditioner_rows']} source-conditioner rows, "
          f"{summary['n_generated']} generated waveforms")
    return 0


def _progress_printer(total, every=25):
    """Per-query progress with a live rate -- this pass runs for days."""
    import time

    state = {"n": 0, "t0": time.time()}

    def _on_row(row):
        state["n"] += 1
        if state["n"] % every and state["n"] != total:
            return
        elapsed = time.time() - state["t0"]
        rate = state["n"] / max(elapsed, 1e-9)
        remaining = (total - state["n"]) / max(rate, 1e-9)
        print(f"[{state['n']}/{total}] {row['room_id']} {row['query_id'].split('|')[0]} "
              f"| {elapsed / 3600:.2f} h elapsed, {rate * 3600:.1f} queries/h, "
              f"~{remaining / 3600:.1f} h left", flush=True)

    return _on_row


def _run_replay_check(args, engine, plan, records, loader):
    """The registered fixed-batching determinism claim, on the first real query."""
    first = records[0]
    room = me.load_room_plan(plan, first["room_id"])
    query = next(q for q in room.queries if q.query_id == first["query_id"])
    for position, (obs_wav, raw_md) in enumerate(_iter_items(loader)):
        if position != int(first["position"]):
            continue
        md = me.GuardedMetadata(raw_md)
        me.verify_context_record(md, first, position)
        report = me.replay_check(engine, query, md, obs_wav, seed=args.seed, tau=args.tau,
                                 num_samples=args.num_samples,
                                 prefixes=tuple(int(k) for k in args.k_prefixes),
                                 noise_policy=args.noise_policy,
                                 batch_rows=args.batch_rows, source_chunk=args.source_chunk)
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"\nFIXED-BATCHING REPLAY: {'BIT-EXACT' if report['bit_exact'] else 'DRIFTED'} "
              f"(max |diff| {report['max_abs_delta']:.3g} over {report['n_candidates']} "
              f"candidates at batch_rows={report['batch_rows']}, "
              f"source_chunk={report['source_chunk']})")
        return 0 if report["bit_exact"] else 1
    _refuse("the stream ended before the first registered query")


def _run_cache_parity(args, engine, plan, records, loader):
    """§1.5's bit-identity proof on the first query of the pass, then stop."""
    first = records[0]
    room = me.load_room_plan(plan, first["room_id"])
    query = next(q for q in room.queries if q.query_id == first["query_id"])
    for position, (_obs, md) in enumerate(_iter_items(loader)):
        if position != int(first["position"]):
            continue
        me.verify_context_record(md, first, position)
        report = me.cache_parity_check(engine, query, md,
                                       source_chunk=args.source_chunk)
        print(json.dumps(report, indent=2, sort_keys=True))
        print("\nMEMOIZATION (the contract): "
              f"{'MATCH' if report['memoization']['match'] else 'MISMATCH'}; "
              f"counter-test {'bit' if report['counter_test']['detected'] else 'DID NOT BITE'}")
        print(f"BATCHED (informational): max |diff| = {report['batched']['max_abs_diff']:.3g} "
              f"at dtypes {sorted(set(report['dtypes'].values()))} -- "
              f"{report['batched']['note']}")
        return 0 if report["match"] else 1
    _refuse("the stream ended before the first registered query")


if __name__ == "__main__":
    sys.exit(main())
