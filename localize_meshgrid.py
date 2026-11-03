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
loc_meshgrid_inherited_exp09_plan.md``. Candidates share one draw per query
(common random numbers, §1.1), and the recorded deviation from the plan text --
the deterministic AGREE mean readout in place of its sampled bottleneck -- is
stamped into the run binding and into every published row.

Reuse boundary: the engine, caches, artifact codec and manifest verifiers live
in ``src.localization.meshgrid_engine``; the model build follows
``eval_FLAC.evaluate_model``'s lines of record through ``eval_localization``.
This file is the protocol wiring around them.

  # a registered R2 shard (plan r2 §3 R2-C): every protocol and batching flag
  # explicit; emit the launch record first, then run the IDENTICAL command with it
  python localize_meshgrid.py --emit-launch-record <out>/launch_record.json \
      --ckpt-path weights/exp20/YAW_40k.ckpt --cond-method vanilla --frame-avg-angles 0 \
      --rotate-deg 0 --cond-autocast default --batch-rows 256 --source-chunk 16 \
      --context-manifest outputs_loc/exp22/d1_context_manifest.json \
      --audit-report outputs_loc/exp22/g1_audit/geometry_audit_report.json \
      --rooms Cafe/Cafe_idx_1 --device cuda:0 --out-dir <out>
  python localize_meshgrid.py --launch-record <out>/launch_record.json <same flags>

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


#: exp_22 R2-A (Codex B6; announcement 05): the protocol and batching flags a
#: canonical full pass must DECLARE on its command line -- none may be inherited
#: from a default, because the default is not the registered value for every arm
#: (``--batch-rows`` defaults to 64; the registered pass ran 256).
REGISTERED_EXPLICIT_FLAGS = ("cond_method", "frame_avg_angles", "rotate_deg", "cond_autocast",
                             "batch_rows", "source_chunk")
#: the explicit no-op orbit a vanilla arm binds, and the registered C4 orbit
VANILLA_FRAME_AVG_ANGLES = [0.0]
FA_FRAME_AVG_ANGLES = [0.0, 90.0, 180.0, 270.0]
REGISTERED_ROTATE_DEG = 0.0


class _Explicit(argparse.Action):
    """Store the value AND remember the flag was given; a default never runs an action."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        namespace.explicit_flags = frozenset(set(namespace.explicit_flags) | {self.dest})


def resolved_protocol(args):
    """``(frame_avg_angles, rotate_deg)`` as bound; diagnostics resolve the vanilla no-op."""
    angles = args.frame_avg_angles
    if angles is None and str(args.cond_method) == "vanilla":
        angles = VANILLA_FRAME_AVG_ANGLES
    rotate = REGISTERED_ROTATE_DEG if args.rotate_deg is None else args.rotate_deg
    # `+ 0.0` folds -0.0 into 0.0: equal in Python, different canonical digests
    return ((None if angles is None else [float(a) + 0.0 for a in angles]),
            float(rotate) + 0.0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.set_defaults(explicit_flags=frozenset())
    parser.add_argument("--ckpt-path", default=None,
                        help="the frozen FLAC checkpoint (wrapped; EMA resolved at load)")
    parser.add_argument("--merge-shards", nargs="+", default=None,
                        help="merge these published shard directories into --merge-out, "
                             "publishing only after the full census passes")
    parser.add_argument("--merge-out", default=None,
                        help="the FRESH directory a merge publishes into")
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
    parser.add_argument("--rooms", nargs="+", default=None,
                        help="run only these canonical room ids (a shard); the whole D1 "
                             "stream is still walked and verified")
    parser.add_argument("--branch", default=None,
                        help="assert the audit's branch; the branch itself is not selectable")
    parser.add_argument("--cond-method", default="vanilla", choices=["vanilla", "fa_invariant"],
                        action=_Explicit)
    parser.add_argument("--cond-autocast", default="default", choices=["default", "bf16", "off"],
                        action=_Explicit)
    parser.add_argument("--frame-avg-angles", type=float, nargs="+", default=None,
                        action=_Explicit,
                        help="the frame-average orbit, explicit on every canonical launch: "
                             "vanilla arms pass exactly 0 (bound as [0.0]); fa_invariant "
                             "passes the registered C4 orbit 0 90 180 270")
    parser.add_argument("--rotate-deg", type=float, default=None, action=_Explicit,
                        help="global panorama rotation; registered 0, anything else refused")
    parser.add_argument("--seed", type=int, default=me.SEED)
    parser.add_argument("--tau", type=float, default=me.TAU)
    parser.add_argument("--num-samples", type=int, default=me.NUM_SAMPLES)
    parser.add_argument("--k-prefixes", type=int, nargs="+", default=list(me.K_PREFIXES))
    parser.add_argument("--noise-policy", default=me.NOISE_KEY_POLICY,
                        choices=list(me.NOISE_KEY_POLICIES))
    parser.add_argument("--steps", type=int, default=me.STEPS)
    parser.add_argument("--cfg-scale", type=float, default=me.CFG_SCALE)
    parser.add_argument("--batch-rows", type=int, default=64, action=_Explicit,
                        help="generated rows per forward (candidates x K)")
    parser.add_argument("--source-chunk", type=int, default=me.SOURCE_CHUNK, action=_Explicit,
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
    parser.add_argument("--dump-cases-sha256", default=None,
                        help="the REGISTERED sha256 of --dump-cases; without it the list "
                             "would authorize itself")
    parser.add_argument("--replay-check", action="store_true",
                        help="score one query twice at fixed batching and require bit-exact "
                             "agreement (the registered determinism claim), then exit")
    parser.add_argument("--cache-parity-check", action="store_true",
                        help="run the §1.5 cached-vs-uncached bit-identity proof and exit; "
                             "under fa_invariant this is GATE 2 (fa_cache_parity evidence)")
    parser.add_argument("--fa-split-parity-check", action="store_true",
                        help="GATE 1 (exp_22 R2-B): the five-way split vs the released "
                             "invariant_conditioning at identical batching, --cond-autocast off; "
                             "writes <out-dir>/fa_split_parity.json and exits")
    parser.add_argument("--fa-production-parity-check", action="store_true",
                        help="GATE 3 (exp_22 R2-B): production batching (256/16, default "
                             "autocast) vs a faithful replay over the registered probe queries "
                             "(--probe-room bounds it to one room); writes "
                             "<out-dir>/fa_production_parity.json and exits")
    parser.add_argument("--replay-against", default=None,
                        help="VANILLA NO-REGRESSION (exp_22 R2-B): with --replay-check, replay "
                             "the registered probe queries of this PUBLISHED run at their rows' "
                             "own batching and require them reproduced exactly; writes "
                             "<out-dir>/vanilla_replay_regression.json")
    parser.add_argument("--replay-probe-report", default=None,
                        help="the probe report whose per-query grid digests are frozen "
                             "(default: the committed canonical P1 off-grid probe report)")
    parser.add_argument("--dataset-root", default="AcousticRooms",
                        help="where the observed RIRs the frozen observation bank hashes live")
    parser.add_argument("--parity-candidates", type=int, default=None,
                        help="candidates gates 1 and 2 compare (default: a full and a tail "
                             "micro-batch for gate 1, two source chunks for gate 2)")
    parser.add_argument("--launch-record", default=None,
                        help="PATH to this invocation's launch record (argv, git SHA, tracked-"
                             "clean, hostname, designated GPU UUID), written BEFORE the run by "
                             "--emit-launch-record. Required for every canonical full pass; one "
                             "record per invocation (a --resume opens a new segment)")
    parser.add_argument("--emit-launch-record", default=None,
                        help="PROVENANCE MODE: write the launch record for THIS command to PATH "
                             "and exit; then run the identical command with --launch-record PATH")
    parser.add_argument("--expect-arm", default=None, choices=["P1", "BF", "YAW"],
                        help="refuse unless the checkpoint is this registered arm")
    return parser.parse_args(argv)


def _refuse(message):
    raise SystemExit(f"REFUSED: {message}")


def validate_protocol_flags(args):
    """The explicit eval protocol (announcement 05): orbit and rotation, per method."""
    angles, rotate = resolved_protocol(args)
    if str(args.cond_method) == "vanilla" and angles != VANILLA_FRAME_AVG_ANGLES:
        _refuse(f"--frame-avg-angles {angles} with --cond-method vanilla: a vanilla arm binds "
                "the explicit no-op orbit, so it passes exactly `--frame-avg-angles 0`")
    if str(args.cond_method) == "fa_invariant" and angles != FA_FRAME_AVG_ANGLES:
        _refuse(f"--frame-avg-angles {angles} with --cond-method fa_invariant: the registered "
                f"orbit is the C4 group in order, `--frame-avg-angles 0 90 180 270`")
    if rotate != REGISTERED_ROTATE_DEG:
        _refuse(f"--rotate-deg {rotate:g}: the registered rotation is 0; a rotated pass is a "
                "different experiment, not a variant of this one")
    return angles, rotate


def validate_gate_flags(args):
    """The fa parity gates' registered conditions (plan r2 §3 R2-B, Codex H3)."""
    from src.localization import meshgrid_report as mr

    gate = args.fa_split_parity_check or args.fa_production_parity_check
    if gate and str(args.cond_method) != "fa_invariant":
        _refuse("the fa parity gates compare the five-way split: pass --cond-method fa_invariant "
                "--frame-avg-angles 0 90 180 270")
    if args.fa_split_parity_check and args.cond_autocast != "off":
        _refuse("gate 1 is registered bit-exact with autocast OFF: pass --cond-autocast off")
    if args.fa_production_parity_check:
        for flag, registered, found in (("--cond-autocast", "default", args.cond_autocast),
                                        ("--batch-rows", mr.REGISTERED_BATCH_ROWS,
                                         args.batch_rows),
                                        ("--source-chunk", mr.REGISTERED_SOURCE_CHUNK,
                                         args.source_chunk)):
            if found != registered:
                _refuse(f"gate 3 runs the PRODUCTION batching and autocast: {flag} {registered}, "
                        f"not {found}")
    return True


def validate_launch_provenance(args):
    """Canonical full passes carry a record and declare every protocol flag; diagnostics don't."""
    recorded = args.launch_record or args.emit_launch_record
    if args.launch_record and args.emit_launch_record:
        _refuse("--emit-launch-record writes a record and exits; --launch-record runs the pass "
                "it describes. Pass one of them")
    if not writes_query_artifacts(args):
        if recorded:
            _refuse("launch records describe canonical full passes; --probe, the parity checks, "
                    "--replay-check and --merge-shards are diagnostics and stay record-free")
        return True
    if not recorded:
        _refuse("a canonical full pass requires --launch-record PATH, written BEFORE the run by "
                "the identical command with --emit-launch-record PATH (plan r2 §3 R2-A, B4)")
    implicit = [flag for flag in REGISTERED_EXPLICIT_FLAGS
                if flag not in set(args.explicit_flags)]
    if implicit:
        _refuse(f"a canonical full pass may not rely on a default: declare "
                f"{['--' + flag.replace('_', '-') for flag in implicit]} explicitly "
                "(announcement 05; the registered batching is --batch-rows 256 "
                "--source-chunk 16)")
    return True


def validate_args(args):
    """Startup refusals -- before a checkpoint is read or a GPU is touched."""
    if args.merge_shards is not None:
        validate_launch_provenance(args)
        if not args.merge_out:
            _refuse("--merge-shards needs --merge-out: a merge publishes into a fresh "
                    "directory, never back into a shard")
        if len(args.merge_shards) < 2:
            _refuse("--merge-shards combines at least two shard directories")
        return True
    if not args.ckpt_path:
        _refuse("--ckpt-path is required for every run except --merge-shards")
    if args.rooms is not None and args.probe is not None:
        _refuse("--rooms shards a scored pass and --probe is a no-quality diagnostic; use "
                "--probe-room to bound a probe")
    modes = [flag for flag, on in (("--probe", args.probe is not None),
                                   ("--cache-parity-check", args.cache_parity_check),
                                   ("--replay-check", args.replay_check),
                                   ("--fa-split-parity-check", args.fa_split_parity_check),
                                   ("--fa-production-parity-check",
                                    args.fa_production_parity_check)) if on]
    if len(modes) > 1:
        _refuse(f"{modes} are separate diagnostics: run one diagnostic per invocation")
    if args.replay_against and not args.replay_check:
        _refuse("--replay-against is the vanilla no-regression form of --replay-check")
    if args.replay_against and str(args.cond_method) != "vanilla":
        _refuse("--replay-against replays the published P1 rows through the vanilla path")
    if args.probe is None and args.probe_room and not (args.fa_production_parity_check
                                                       or args.replay_against):
        _refuse("--probe-room only bounds the throughput probe (or gate 3); pass --probe too")
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
    if args.dump_cases and not args.dump_cases_sha256:
        _refuse("--dump-cases needs --dump-cases-sha256: the announcement-08 exemption is "
                "extended by a REGISTERED list, and a list that authorizes itself is not one")
    if args.dump_cases_sha256 and not args.dump_cases:
        _refuse("--dump-cases-sha256 was given without --dump-cases")
    if float(args.tau) <= 0.0:
        _refuse(f"--tau must be > 0, got {args.tau}")
    validate_protocol_flags(args)
    validate_gate_flags(args)
    validate_launch_provenance(args)
    # fa_invariant is no longer refused here: the engine's own refusal
    # (assert_cacheable) fires in preflight, AFTER every binding check (R2-A)
    return True


def emit_launch_record(args, argv, environment=None, runtime=None):
    """PROVENANCE MODE (B4): the probe module's record, for THIS command; no gates, no GPU work."""
    from src.localization import meshgrid_segments as ms

    try:
        record = ms.build_launch_record(argv, device=args.device, environment=environment,
                                        runtime=runtime)
    except ValueError as error:
        _refuse(str(error))
    # the registered launch lines emit into the shard directory before it exists
    parent = os.path.dirname(os.path.abspath(str(args.emit_launch_record)))
    os.makedirs(parent, exist_ok=True)
    me.write_json(args.emit_launch_record, record)
    print(f"launch record -> {args.emit_launch_record}\nnow run the identical command with "
          f"--launch-record {args.emit_launch_record}")
    return 0


def admit_launch_record(args, argv, environment=None, runtime=None):
    """The record must describe THIS invocation on THIS machine, every axis (probe mechanism)."""
    from src.localization import meshgrid_segments as ms

    try:
        return ms.read_verified_launch_record(args.launch_record, argv, device=args.device,
                                              environment=environment, runtime=runtime)
    except (OSError, ValueError) as error:
        _refuse(f"launch record: {error}")


def preflight(args, plan, model_config, checkpoint, *, ckpt_sha256, agree_sha256,
              model_config_sha256):
    """Every CPU-side refusal about WHAT will run, in order, before any device work.

    The binding; the registered arm and its complete method (the same gate the
    report applies last -- strict for a canonical pass, a stamped deviation for a
    diagnostic); the file's embedded training protocol; the ARE / cond-method
    binding; and only THEN the engine's own cacheability refusal, so a protocol
    mismatch is named as one rather than as a cache limitation (plan r2 §3 R2-A).
    """
    from src.localization import meshgrid_report as mr

    binding = build_run_binding(args, plan, ckpt_sha256=ckpt_sha256, agree_sha256=agree_sha256,
                                model_config_sha256=model_config_sha256)
    try:
        registered = mr.assert_registered_protocol(
            binding, allow_deviation=not writes_query_artifacts(args),
            expect_arm=args.expect_arm)
    except ValueError as error:
        _refuse(f"arm/protocol check: {error}")
    embedded = mr.checkpoint_protocol_reasons(checkpoint, registered["arm"], binding)
    for reason in embedded["reasons"]:
        _refuse(f"embedded training protocol: {reason}")
    cond_binding = validate_checkpoint(args, model_config, checkpoint)
    # the split the engine will be built under (R2-B): fa is admitted only with it
    me.assert_cacheable(args.cond_method,
                        me.split_for(args.cond_method, resolved_protocol(args)[0]))
    return binding, {"registered": registered, "embedded": embedded,
                     "cond_binding": cond_binding}


def claim_out_dir(args, run_binding, advisory, launch):
    """Publish or continue the binding, open this invocation's segment, list what to skip."""
    from src.localization import meshgrid_segments as ms

    if args.resume:
        assert_resumable(args, args.out_dir)
    if os.path.isfile(os.path.join(args.out_dir, me.BINDING_FILENAME)):
        moved = me.assert_binding(args.out_dir, run_binding, advisory=advisory,
                                  declared_rooms=args.rooms)
        if moved is not True:
            me.record_advisory_change(args.out_dir, moved, advisory=advisory)
            print(f"NOTE: batching changed since the published pass: {moved}\n  "
                  f"recorded in {me.BINDING_FILENAME}; {me.BATCHING_CAVEAT}")
    else:
        me.write_binding(args.out_dir, run_binding, advisory=advisory,
                         declared_rooms=args.rooms)
    segment = (ms.open_segment(args.out_dir, args.launch_record, launch, resume=args.resume)
               if launch else None)
    done = set()
    if args.resume:
        done, rejected = me.completed_queries(
            args.out_dir, binding_sha256=me.binding_sha256(run_binding),
            launch_records=set(ms.ledger_shas(args.out_dir)) if launch else None)
        print(f"resume: {len(done)} verified queries skipped, {len(rejected)} rejected "
              f"and regenerated")
        for verdict in rejected[:5]:
            print(f"  rejected {verdict['query_id']}: {verdict['reason']}")
    return done, segment


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
    """What this run may dump, and WHERE that authority came from.

    ``registered_probe_queries`` reads every room's candidate manifest -- 328 MB
    of index lists -- so a run that dumps nothing must not pay for it.
    """
    if not args.dump_waveforms:
        return {"query_ids": set(), "case_list": None, "probe_queries": {}}
    cases = (me.load_dump_cases(args.dump_cases, expected_sha256=args.dump_cases_sha256)
             if args.dump_cases else None)
    probes = me.registered_probe_queries(plan)
    allowed = set(probes.values()) | set((cases or {}).get("query_ids", []))
    me.assert_dump_allowed(args.dump_waveforms, allowed)
    return {"query_ids": {str(query) for query in args.dump_waveforms},
            "case_list": cases, "probe_queries": probes}


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
    return (args.probe is None and not args.cache_parity_check and not args.replay_check
            and not args.fa_split_parity_check and not args.fa_production_parity_check
            and args.merge_shards is None)


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
        "dump_cases_sha256": args.dump_cases_sha256 or None,
        "dataset_config": str(args.dataset_config),
        **protocol_binding(args),
    }


def protocol_binding(args):
    """The R2 protocol fields this namespace binds, or ``{}`` for a legacy namespace.

    The off-grid probe and the retrieval control reuse :func:`build_run_binding`
    to verify the PUBLISHED P1 binding, which predates these flags; their parsers
    define no ``--frame-avg-angles``, so they keep building the legacy R1 binding.
    """
    if not hasattr(args, "frame_avg_angles"):
        return {}
    angles, rotate = resolved_protocol(args)
    partition = (me.declared_fa_partition(angles, batch_rows=args.batch_rows,
                                          num_samples=args.num_samples,
                                          source_chunk=args.source_chunk)
                 if str(args.cond_method) == "fa_invariant" else None)
    return {"frame_avg_angles": angles, "rotate_deg": rotate, "fa_partition": partition}


def _iter_items(loader):
    """Yield ``(reals_i, md_i)`` one query at a time from the batched loader."""
    for reals, metadata in loader:
        for index, md in enumerate(metadata):
            yield (None if reals is None else reals[index:index + 1]), md


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    validate_args(args)
    if args.merge_shards is not None:
        return _run_merge(args)
    if args.emit_launch_record:
        return emit_launch_record(args, argv)
    # provenance first: a record that does not describe this invocation on this
    # machine refuses before a single artifact is read (B4)
    launch = admit_launch_record(args, argv) if args.launch_record else None

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

    # CPU-only validation first: the binding, the arm and its method, the file's
    # own training protocol, no ARE, and the engine's refusal -- all before
    # anything reaches a device (plan r2 §3 R2-A; the report is the LAST gate)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    run_binding, verdict = preflight(args, plan, model_config, ckpt,
                                     ckpt_sha256=me.file_sha256(args.ckpt_path),
                                     agree_sha256=me.file_sha256(agree_path),
                                     model_config_sha256=me.file_sha256(args.model_config))
    registered = verdict["registered"]
    status = ("registered protocol" if registered["is_registered"]
              else f"DEVIATES on {sorted(registered['deviations'])}")
    print(f"arm {registered['arm']} ({status}); cond_method binding: "
          f"{verdict['cond_binding']['binding']} "
          f"({verdict['cond_binding'].get('checkpoint_cond_method')!r})")
    if launch:
        print(f"launch record {launch['sha256'][:12]}... git {launch['git_sha'][:12]} on "
              f"{launch['hostname']}, card {launch['execution_gpu_uuid']}")

    from src.localization.agree_embed import load_agree_audio

    agree = load_agree_audio(agree_path, args.device)
    if agree.ckpt_sha256 != run_binding["agree_ckpt_sha256"]:
        _refuse("the AGREE scorer that loaded is not the file the binding was built from")
    engine, context = me.build_mesh_engine(
        args.ckpt_path, model_config, agree, device=args.device,
        cond_method=args.cond_method, cond_autocast=args.cond_autocast,
        steps=args.steps, cfg_scale=args.cfg_scale, ckpt=ckpt,
        frame_avg_angles=resolved_protocol(args)[0])
    print(f"weights: {context['weights_source']}, latent {context['latent_shape']}")

    advisory = {"source_chunk": int(args.source_chunk), "batch_rows": int(args.batch_rows)}
    done, segment = set(), None
    if writes_query_artifacts(args):
        done, segment = claim_out_dir(args, run_binding, advisory, launch)

    admitted = dump_allowance(args, plan)
    dump_queries = admitted["query_ids"]
    if dump_queries:
        authority = ("the registered off-grid probe list" if admitted["case_list"] is None
                     else f"the probe list + case list {admitted['case_list']['sha256'][:12]}...")
        print(f"bounded dumps admitted for {len(dump_queries)} queries via {authority} "
              f"(announcement 08 exemption)\n  {me.DUMP_CONTENT_RULE}")

    # the released call graph, in order: seed -> loader -> metric stack -> iterator.
    # NOTHING may consume the global RNG between build_release_stack and the first batch.
    loader, facts = mq.build_release_stack(args.dataset_config, args.model_config)
    me.assert_release_rng_state(manifest)
    print(f"release call graph reproduced: {facts['call_graph']}, RNG state matches the "
          "D1 pass at iterator creation")

    if args.cache_parity_check:
        return _run_cache_parity(args, engine, plan, records, loader, binding=run_binding)
    if args.fa_split_parity_check:
        return _run_fa_gate(args, engine, plan, records, loader, "G1", run_binding)
    if args.fa_production_parity_check:
        return _run_fa_gate(args, engine, plan, records, loader, "G3", run_binding)
    if args.replay_check:
        return _run_replay_check(args, engine, plan, records, loader, binding=run_binding)

    progress = _progress_printer(len(records) - len(done))
    summary = me.run_pass(engine, _iter_items(loader), records, plan, args.out_dir,
                          on_row=progress,
                          seed=args.seed, tau=args.tau, num_samples=args.num_samples,
                          prefixes=tuple(int(k) for k in args.k_prefixes),
                          noise_policy=args.noise_policy, batch_rows=args.batch_rows,
                          source_chunk=args.source_chunk, done=done,
                          probe=args.probe, probe_room=args.probe_room,
                          dump_queries=dump_queries, dump_top_n=args.dump_top_n,
                          binding_sha256=me.binding_sha256(run_binding), rooms=args.rooms,
                          launch_record_sha256=(launch or {}).get("sha256"),
                          fa_partition=run_binding.get("fa_partition"))
    summary["created_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary["binding_sha256"] = me.binding_sha256(run_binding)
    summary["agree_leakage_caveat"] = me.AGREE_LEAKAGE_CAVEAT
    summary["scorer_readout_deviation"] = me.SCORER_READOUT_DEVIATION
    summary["sims_precision_caveat"] = me.SIMS_PRECISION_CAVEAT
    summary["dump_cases"] = admitted["case_list"]
    summary["dump_queries"] = sorted(dump_queries)

    if args.probe is not None:
        path = me.write_probe_records(args.out_dir, summary.pop("probe_records"),
                                      stem=args.probe_stem, binding=run_binding,
                                      binding_sha256=me.binding_sha256(run_binding),
                                      advisory=advisory,
                                      fa_partition_executed=summary.get("fa_partition_executed"))
        projection = json.load(open(path))["projection"]
        hours = projection["projected_gpu_hours"]
        print(f"probe: {summary['n_generated']} generated waveforms over "
              f"{summary['n_candidate_query_pairs']} candidate-query pairs; NO scores written")
        print(f"  timings (s): {summary['timings_s']}")
        print(f"  rates: {projection['seconds_per_waveform'] * 1e3:.3f} ms/waveform, "
              f"{projection['seconds_per_source_row'] * 1e3:.3f} ms/source row, "
              f"{projection['seconds_per_context'] * 1e3:.3f} ms/context")
        print(f"  PROJECTION over {me.REGISTERED_TOTALS['generated_waveforms']:,} waveforms / "
              f"{me.REGISTERED_TOTALS['source_rows']:,} source rows / "
              f"{me.REGISTERED_TOTALS['queries']:,} contexts: "
              f"generation {hours['generation']:.1f} h + source {hours['source_conditioning']:.1f} h "
              f"+ context {hours['context']:.1f} h = {hours['total']:.1f} GPU-hours")
        print(f"  -> {path}")
        return 0

    summary.pop("probe_records", None)
    if segment is not None:
        from src.localization import meshgrid_segments as ms

        summary["launch_segment"] = ms.close_segment(args.out_dir,
                                                     segment["launch_record_sha256"], summary)
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


def _run_merge(args):
    """Combine published shards into one verified run -- no model, no GPU."""
    plan = me.load_audit_plan(args.audit_report, branch=args.branch)
    records = mq.load_manifest(args.context_manifest)["records"]
    report = me.merge_shards(args.merge_shards, args.merge_out, plan, records)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("agree_leakage_caveat", "batching_caveat")},
                     indent=2, sort_keys=True))
    print(f"\nMERGE CENSUS PASSED: {report['n_rows']:,} queries over "
          f"{len(report['declared_rooms'])} rooms; "
          f"{report['totals']['candidate_query_pairs']:,} candidate-query pairs, "
          f"{report['totals']['source_rows']:,} source rows, "
          f"{report['totals']['generated_waveforms']:,} generated waveforms "
          f"-> {args.merge_out}")
    return 0


def _run_replay_check(args, engine, plan, records, loader, binding=None):
    """The registered fixed-batching determinism claim, on the first real query -- or,
    with --replay-against, the vanilla no-regression replay of the probe queries."""
    if args.replay_against:
        from src.localization import meshgrid_fa_parity as fp
        from src.localization import meshgrid_offgrid_probe as mop

        bank = mop.compute_observation_bank_digest(args.audit_report, args.context_manifest,
                                                   dataset_root=args.dataset_root)
        report = fp.vanilla_replay_regression(
            engine, _probe_cases(args, plan, records, loader), args.replay_against,
            frozen_digests=fp.frozen_grid_digests(args.replay_probe_report
                                                  or fp.COMMITTED_PROBE_REPORT),
            observation_bank_sha256=bank["observation_bank_sha256"], run_binding=binding,
            seed=args.seed, num_samples=args.num_samples, noise_policy=args.noise_policy,
            tau=args.tau)
        return _publish_gate(args, report, binding)
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


def _first_query(plan, records, loader):
    """The pass's first registered query, its verified metadata and its observation."""
    first = records[0]
    room = me.load_room_plan(plan, first["room_id"])
    query = next(q for q in room.queries if q.query_id == first["query_id"])
    for position, (obs, md) in enumerate(_iter_items(loader)):
        if position == int(first["position"]):
            me.verify_context_record(md, first, position)
            return query, md, obs
    _refuse("the stream ended before the first registered query")


def _probe_cases(args, plan, records, loader):
    """Gate 3's cases: the registered probe queries (one room with --probe-room), streamed."""
    from src.localization.meshgrid_fa_parity import probe_case

    probes = me.registered_probe_queries(plan)
    rooms = [args.probe_room] if args.probe_room else sorted(probes)
    wanted = {probes[room]: room for room in rooms}
    by_position = {int(record["position"]): record for record in records}
    for position, (obs, md) in enumerate(_iter_items(loader)):
        record = by_position.get(position)
        if record is None or record["query_id"] not in wanted:
            continue
        guarded = me.GuardedMetadata(md)
        me.verify_context_record(guarded, record, position)
        room = me.load_room_plan(plan, wanted[record["query_id"]])
        query = next(q for q in room.queries if q.query_id == record["query_id"])
        yield probe_case(room, query, guarded, obs)


def _gate_inputs(args, binding):
    return {"binding": dict(binding or {}), "cond_autocast": args.cond_autocast,
            "batch_rows": int(args.batch_rows), "source_chunk": int(args.source_chunk),
            "ckpt_path": args.ckpt_path, "device": args.device}


def _publish_gate(args, report, binding):
    from src.localization import meshgrid_fa_parity as fp

    evidence = fp.finalize_evidence(report, args.device, inputs=_gate_inputs(args, binding))
    path = fp.write_evidence(args.out_dir, evidence)
    print(f"GATE {report['gate']} {report['name']}: {report['verdict']} -> {path}")
    for reason in report["reasons"][:10]:
        print(f"  {reason}")
    return 0 if report["match"] else 1


def _run_fa_gate(args, engine, plan, records, loader, gate, binding):
    """Gate 1 on the pass's first query, or gate 3 over the registered probe queries."""
    from src.localization import meshgrid_fa_parity as fp

    micro = max(1, int(args.batch_rows) // int(args.num_samples))
    if gate == "G1":
        query, md, _obs = _first_query(plan, records, loader)
        report = fp.split_parity(engine, query, me.GuardedMetadata(md), micro_batch=micro,
                                 n_candidates=args.parity_candidates)
    else:
        report = fp.production_parity(
            engine, _probe_cases(args, plan, records, loader), batch_rows=args.batch_rows,
            source_chunk=args.source_chunk, num_samples=args.num_samples, seed=args.seed,
            noise_policy=args.noise_policy, tau=args.tau,
            prefixes=tuple(int(k) for k in args.k_prefixes),
            declared_partition=(binding or {}).get("fa_partition"))
    return _publish_gate(args, report, binding)


def _run_cache_parity(args, engine, plan, records, loader, binding=None):
    """§1.5's bit-identity proof on the first query of the pass, then stop (gate 2 under fa)."""
    query, md, _obs = _first_query(plan, records, loader)
    if getattr(engine, "split", None) is not None:
        from src.localization import meshgrid_fa_parity as fp

        return _publish_gate(args, fp.cache_parity(
            engine, query, me.GuardedMetadata(md), n_candidates=args.parity_candidates,
            source_chunk=args.source_chunk,
            micro_batch=max(1, int(args.batch_rows) // int(args.num_samples))), binding)
    report = me.cache_parity_check(engine, query, md, source_chunk=args.source_chunk)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("\nMEMOIZATION (the contract): "
          f"{'MATCH' if report['memoization']['match'] else 'MISMATCH'}; "
          f"counter-test {'bit' if report['counter_test']['detected'] else 'DID NOT BITE'}")
    print(f"BATCHED (informational): max |diff| = {report['batched']['max_abs_diff']:.3g} "
          f"at dtypes {sorted(set(report['dtypes'].values()))} -- "
          f"{report['batched']['note']}")
    return 0 if report["match"] else 1


if __name__ == "__main__":
    sys.exit(main())
