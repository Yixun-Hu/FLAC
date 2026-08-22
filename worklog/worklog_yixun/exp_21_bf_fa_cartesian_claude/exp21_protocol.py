#!/usr/bin/env python3
"""exp_21 (bf_fa_cartesian): the evaluation CAMPAIGN, stated once, for all arms.

``exp21_validate_cell.py`` says what a registered BFC *record* must contain.
This module says what *commands* produce the campaign — the BFC arm and the two
D6 comparators — and it is the single place any of them is spelled out. The eval
driver (``bfc_eval_driver.sh``) and the model-comparison generator both read it,
so a flag cannot be right in one and wrong in the other. Every constant it does
not own is IMPORTED from ``exp21_validate_cell``; nothing is restated.

WHY THREE ARMS (plan §5, D6 — APPROVED by Yixun). The historical B-F@40k and
P1@40k rows were measured at a different evaluator pin, under the legacy
per-angle orbit executor, before ``--cond-autocast bf16`` and the per-scene and
stream provenance existed. ``model_comparison.md`` itself marks legacy-loop and
batched rows non-interchangeable. Reading BFC as a paired delta against them
would fold the whole evaluator shift into the arm's effect. D6 therefore
re-evaluates BOTH comparators at THIS pin, five seeds x both K, every flag
identical to BFC's except the one that must differ:

    BFC   --cond-method fa_cartesian   (+ C4 angles, eval cap 64)
    BFre  --cond-method fa_invariant   (+ C4 angles, eval cap 64)   [B-F @40k]
    P1re  --cond-method vanilla        (no orbit)                   [P1  @40k]

TRAINED-AS, PER ARM (r5 full review, findings 1 and 4). Each arm declares the
embedded training config its checkpoint must carry, and ``check_embedded_training``
holds the checkpoint to it. The three expectations are NOT the same shape, and
that is the point — they were read off the artifacts on disk:

    BFC   cond_method 'fa_cartesian', angles [0,90,180,270], training cap 32
    BFre  cond_method 'fa_invariant', angles [0,90,180,270], NO cap key
          (the knob postdates B-F's training — absent is its real historical shape)
    P1re  NO cond_method key at all (the factory default, i.e. vanilla), no
          angles, no cap

``eval_FLAC`` enforces the BFC expectation itself, before any model is built.
The comparator arms are held to theirs by ``preflight``, which the driver runs
once per arm before that arm's cells; ``eval_FLAC`` also RECORDS what it found in
every row (``trained_cond_method``), so the table gate re-checks it per cell.

WHERE THE ARTIFACTS LAND. ``eval_FLAC`` writes its metrics JSON to
``dirname(--ckpt-path)``. For the comparators that is exp_07's checkpoint
directories, which are this checkout's own outputs tree. Verified before
registering these names: neither directory holds a single file matching
``*exp21*`` (144 and 215 files respectively), and no historical row's glob
matches the ``exp21_BFre_`` / ``exp21_P1re_`` stems, so nothing is clobbered and
no legacy row silently absorbs a new cell.
"""
import argparse
import glob
import importlib.util
import os
import sys
from collections import namedtuple

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_validator():
    """exp_21's record contract — imported, never re-typed."""
    target = os.path.join(_HERE, "exp21_validate_cell.py")
    spec = importlib.util.spec_from_file_location("exp21_validate_cell", target)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the exp_21 validator from {target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V = _load_validator()

# Everything below is V's, by reference: STEP, SEEDS, KS, the dataset configs,
# the sampling protocol, the eval cap, the split size.
STEP, SEEDS, KS = V.STEP, V.SEEDS, V.KS
EXPECTED_COUNT = V.EXPECTED_COUNT

ABSENT = object()   # "this key must not be present", distinct from a null value

#: The four extra angles of the §5 invariance grid. The 0-degree member is the
#: REGISTERED K=8 / seed-42 cell itself — it now carries --record-stream like
#: every registered cell, so re-running it under a second name would buy nothing
#: and publish two measurements of one thing (plan §5: "14 unique BFC cells —
#: 10 registered + 4 extra grid angles, the K8/s42/0 cell shared").
GRID_ANGLES = (45.0, 90.0, 180.0, 270.0)
GRID_K, GRID_SEED = 8, 42

REPO_RELATIVE_MODEL_CONFIGS = {
    "BFC": "worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/FLAC_AR_BFC.json",
    "BFre": "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json",
    "P1re": "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json",
}

#: Per arm: the checkpoint, the conditioning, and the training contract its
#: embedded model_config must satisfy.
ARMS = {
    "BFC": {
        "eval_prefix": "exp21_BFC",
        "cond_method": "fa_cartesian",
        # BFC has not trained yet, so the epoch number is unknown: the path is a
        # glob, resolved (to exactly one file) at run time.
        "ckpt": ("outputs_FLAC/exp21_BFC/FLAC_exp21_BFC/exp21_BFC/checkpoints/"
                 f"epoch=*-step={STEP}.ckpt"),
        "expected_training": {"cond_method": "fa_cartesian",
                              "frame_avg_angles": list(V.FRAME_AVG_ANGLES),
                              "frame_avg_max_fwd_samples": 32},
        # BFC has not trained yet, so there is no reviewed artifact to pin.
        # ``None`` states that absence; the gate still requires ONE digest across
        # all ten of its cells and both its K rows.
        "ckpt_sha256": None,
        "row_label": "BFC C4-Cartesian FA @40k (exp_21)",
    },
    "BFre": {
        "eval_prefix": "exp21_BFre",
        "cond_method": "fa_invariant",
        "ckpt": ("outputs_FLAC/exp07_BF/FLAC_exp07_BF/exp07_BF/checkpoints/"
                 f"epoch=8-step={STEP}.ckpt"),
        "expected_training": {"cond_method": "fa_invariant",
                              "frame_avg_angles": list(V.FRAME_AVG_ANGLES),
                              "frame_avg_max_fwd_samples": ABSENT},
        # The REVIEWED artifact, digested on disk in round 5 (exp_07 B-F @40k).
        # A comparator produced from any other bytes is not the comparator
        # this campaign approved, however well-formed its digest is.
        "ckpt_sha256": "5319feb4af874624859e87105ddd8ab06d4b449769d1e054f712b2b1c0542328",
        "row_label": "B-F @40k re-eval at the exp_21 pin (D6 paired comparator)",
    },
    "P1re": {
        "eval_prefix": "exp21_P1re",
        "cond_method": "vanilla",
        "ckpt": ("outputs_FLAC/exp07_P1/FLAC_exp07_P1/exp07_P1/checkpoints/"
                 f"epoch=8-step={STEP}.ckpt"),
        "expected_training": {"cond_method": "vanilla",
                              "frame_avg_angles": ABSENT,
                              "frame_avg_max_fwd_samples": ABSENT},
        # The REVIEWED artifact, digested on disk in round 5 (exp_07 P1 @40k).
        # A comparator produced from any other bytes is not the comparator
        # this campaign approved, however well-formed its digest is.
        "ckpt_sha256": "c4c678826cddda37fa4977926aadee530afd037b3abb110918b52a342ce9845c",
        "row_label": "P1 @40k re-eval at the exp_21 pin (D6 paired comparator)",
    },
}
ARM_ORDER = ("BFC", "BFre", "P1re")
FRAME_AVERAGED = ("fa_cartesian", "fa_invariant")


def repo_root(start=None):
    return V._repo_root(start)


def arm_profile(arm):
    """The record contract for ``arm``: the five things that vary between arms.

    Built from the SAME registry the commands are built from, so a row cannot be
    admitted under a protocol no command would have produced.
    """
    spec = ARMS[arm]
    return V.profile(spec["eval_prefix"], spec["cond_method"],
                     spec["expected_training"]["cond_method"],
                     spec["cond_method"] in FRAME_AVERAGED)


def eval_name(arm, k, seed, rotate_deg=0.0):
    """The ``--eval-name`` of one cell. It is the ONLY thing separating cells'
    artifacts: ``build_output_paths`` adds neither K nor the seed itself."""
    name = f"{ARMS[arm]['eval_prefix']}_S{STEP}_K{int(k)}_s{int(seed)}"
    return name if not rotate_deg else f"{name}_rot{_deg_token(rotate_deg)}"


def _deg_token(deg):
    """``45.0 -> '45'`` — eval_FLAC's integer rendering (rot_token). This campaign
    uses only whole degrees, and a fractional one would be a different grid."""
    if float(deg) != int(deg):
        raise ValueError(f"rotate_deg {deg!r} is not a whole degree; the §5 grid is "
                         f"{list(GRID_ANGLES)}")
    return str(int(deg))


def _rot_suffix(deg):
    return "" if not deg else f"_rot{_deg_token(deg)}"


def _method_suffix(cond_method, n_angles=4):
    """``build_output_paths``' rule (eval_FLAC.py:397), mirrored and pinned by test."""
    return "" if cond_method == "vanilla" else f"_{cond_method}_a{n_angles}"


def resolve_ckpt(arm, root=None, placeholder=None):
    """The arm's checkpoint path. Globs resolve to EXACTLY one file.

    ``placeholder`` is for dry runs before the arm has trained: it renders a
    stand-in basename rather than failing, and the caller is expected to say so.
    """
    root = root or repo_root()
    pattern = ARMS[arm]["ckpt"]
    if "*" not in pattern:
        path = os.path.join(root, pattern)
        if not os.path.isfile(path) and placeholder is None:
            raise FileNotFoundError(f"{arm}: no checkpoint at {path}")
        return path
    hits = sorted(glob.glob(os.path.join(root, pattern)))
    if len(hits) == 1:
        return hits[0]
    if not hits:
        if placeholder is None:
            raise FileNotFoundError(
                f"{arm}: no checkpoint matches {pattern} — has the arm trained to "
                f"step {STEP}?")
        return os.path.join(root, os.path.dirname(pattern),
                            os.path.basename(pattern).replace("*", placeholder))
    raise RuntimeError(
        f"{arm}: {len(hits)} checkpoints match {pattern} ({[os.path.basename(h) for h in hits]}): "
        f"a registered cell evaluates ONE step-{STEP} checkpoint")


def metrics_path(arm, k, seed, rotate_deg=0.0, root=None, placeholder=None):
    """Where ``eval_FLAC`` will write this cell's metrics JSON.

    Mirrors ``build_output_paths``: same directory as the checkpoint, named
    ``<ckpt stem>_metrics_<steps>_<cfg>_<eval name><method><rot>.json``. Pinned
    equal to eval_FLAC's own function by test.
    """
    ckpt = resolve_ckpt(arm, root=root, placeholder=placeholder)
    stem = os.path.basename(ckpt)[: -len(".ckpt")]
    name = (f"{stem}_metrics_{V.STEPS}_{V.CFG_SCALE}_{eval_name(arm, k, seed, rotate_deg)}"
            f"{_method_suffix(ARMS[arm]['cond_method'], len(V.FRAME_AVG_ANGLES))}"
            f"{_rot_suffix(rotate_deg)}.json")
    return os.path.join(os.path.dirname(ckpt), name)


def command(arm, k, seed, rotate_deg=0.0, root=None, placeholder=None):
    """The full argv of one cell, as a token list.

    ONE definition of the announcement-05 flag set. Frame-average flags are
    emitted only for the arms that run an orbit: passing ``--frame-avg-angles``
    to a vanilla run would record a protocol it did not execute, and passing a
    cap it never applies would suggest a chunk plan that does not exist.
    """
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}; the campaign is {list(ARM_ORDER)}")
    if int(k) not in KS:
        raise ValueError(f"K={k} is not one of the registered context sizes {list(KS)}")
    if int(seed) not in SEEDS:
        raise ValueError(f"seed {seed} is not one of the registered eval seeds {list(SEEDS)}")
    spec = ARMS[arm]
    argv = [
        "python", "eval_FLAC.py",
        "--model-config", REPO_RELATIVE_MODEL_CONFIGS[arm],
        "--dataset-config", V.DATASET_CONFIG[int(k)],
        "--ckpt-path", os.path.relpath(
            resolve_ckpt(arm, root=root, placeholder=placeholder), root or repo_root()),
        "--cond-method", spec["cond_method"],
    ]
    if spec["cond_method"] in FRAME_AVERAGED:
        argv += ["--frame-avg-angles",
                 ",".join(_deg_token(a) for a in V.FRAME_AVG_ANGLES),
                 "--frame-avg-max-fwd-samples", str(V.FRAME_AVG_FWD_CAP)]
    argv += [
        "--rotate-mode", "fixed", "--rotate-deg", _deg_token(rotate_deg),
        "--cond-autocast", V.COND_AUTOCAST,
        "--batch-size", str(V.BATCH_SIZE),
        "--cfg-scale", str(V.CFG_SCALE),
        "--steps", str(V.STEPS),
        "--record-per-scene",
        "--record-stream", "--expected-stream-count", str(EXPECTED_COUNT),
        "--seed", str(int(seed)),
        "--eval-name", eval_name(arm, k, seed, rotate_deg),
    ]
    return argv


Cell = namedtuple("Cell", "arm k seed rotate_deg kind")


def inventory():
    """Every cell of the campaign, in execution order.

    34 cells: 10 registered BFC (5 seeds x 2 K) + 4 invariance-grid angles at
    K=8/seed 42 + 10 BFre + 10 P1re. The grid's 0-degree member is the registered
    K8/s42 cell — shared, not re-run (plan §5).
    """
    cells = []
    for k in KS:
        for seed in SEEDS:
            cells.append(Cell("BFC", k, seed, 0.0, "registered"))
    for deg in GRID_ANGLES:
        cells.append(Cell("BFC", GRID_K, GRID_SEED, deg, "grid"))
    for arm in ("BFre", "P1re"):
        for k in KS:
            for seed in SEEDS:
                cells.append(Cell(arm, k, seed, 0.0, "comparator"))
    return cells


# --------------------------------------------------------------------------- #
# the per-arm trained-as expectation
# --------------------------------------------------------------------------- #
def _type_strict_equal(got, want):
    """Equality that does NOT let ``1`` pass for ``1.0`` — element-wise.

    Python compares ``[0, 90, 180, 270] == [0.0, 90.0, 180.0, 270.0]`` as True and
    both are ``list``, so a container-level type check misses exactly the drift
    this is here to catch (caught by test, not by inspection). The factory
    distinguishes the two, so this must as well.
    """
    if type(got) is not type(want):
        return False
    if isinstance(want, list):
        return (len(got) == len(want)
                and all(_type_strict_equal(g, w) for g, w in zip(got, want)))
    if isinstance(want, dict):
        return (set(got) == set(want)
                and all(_type_strict_equal(got[k], want[k]) for k in want))
    return got == want


def check_embedded_training(embedded_model_config, arm):
    """``[reason, ...]`` naming every way this checkpoint is not ``arm``'s.

    Pure, so it is testable without a 700 MB file. ``eval_FLAC`` enforces the BFC
    case itself before any model is built; this is what holds the COMPARATOR
    checkpoints to their own (different) shapes, and it is why the expectations
    are declared per arm rather than as one template.
    """
    if arm not in ARMS:
        return [f"unknown arm {arm!r}"]
    if not isinstance(embedded_model_config, dict) or not embedded_model_config:
        return [f"{arm}: the checkpoint carries no embedded 'model_config' "
                f"({type(embedded_model_config).__name__}), so the arm it was TRAINED "
                "as cannot be proven (announcement 05)"]
    training = embedded_model_config.get("training")
    if not isinstance(training, dict):
        return [f"{arm}: the embedded model_config has no 'training' block "
                f"(found {training!r})"]

    reasons = []
    for key, want in sorted(ARMS[arm]["expected_training"].items()):
        present = key in training
        if want is ABSENT:
            if present:
                reasons.append(
                    f"{arm}: embedded training.{key} is {training[key]!r}, but this "
                    "arm's checkpoint predates that knob and must not declare it — a "
                    "checkpoint that does is not the artifact this row publishes")
            continue
        if key == "cond_method":
            # an absent cond_method IS the factory default, and that is what those
            # weights trained under (exp_07 P1's embedded config is this shape)
            got = training.get(key, "vanilla")
        elif not present:
            reasons.append(f"{arm}: embedded training.{key} is absent, required {want!r}")
            continue
        else:
            got = training[key]
        if not _type_strict_equal(got, want):
            reasons.append(
                f"{arm}: embedded training.{key} is {got!r} ({type(got).__name__}), "
                f"required {want!r} ({type(want).__name__}) — type-strictly, element "
                "by element, which is the distinction the factory itself enforces")
    return reasons


def _load_embedded_config(path):
    """The ``model_config`` train.py embedded in a checkpoint. CPU, torch lazily."""
    import torch                                  # noqa: PLC0415 - lazy on purpose
    return torch.load(path, map_location="cpu", weights_only=False).get("model_config")


def _file_sha256(path, chunk=1 << 20):
    """Streamed sha256 -- eval_FLAC.file_sha256's rule, without importing torch."""
    import hashlib                                # noqa: PLC0415
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def preflight(arm, root=None):
    """Hold ``arm``'s checkpoint to its contract -- BOTH halves. Returns its sha256.

    A cheap, GPU-free gate the driver runs ONCE per arm, before that arm's cells,
    so a wrong-arm campaign costs one checkpoint read rather than ten evaluations.

    Two independent things are checked, and a failure of either ABORTS (r5
    re-review, BLOCKING 3 -- it used to print whatever digest it encountered and
    continue, which made it a label rather than a gate):

    * the embedded training config IS this arm's (what the weights were trained as);
    * the file's bytes ARE the reviewed artifact's, where one is pinned. BFC pins
      nothing because it has not trained yet; its uniformity is enforced across
      its own cells by the table gate instead.
    """
    root = root or repo_root()
    path = resolve_ckpt(arm, root=root)
    reasons = check_embedded_training(_load_embedded_config(path), arm)
    if reasons:
        raise SystemExit("TRAINED-AS PREFLIGHT FAILED for " + arm + ":\n  - "
                         + "\n  - ".join(reasons))
    digest = _file_sha256(path)
    want = ARMS[arm].get("ckpt_sha256")
    if want is not None and digest != want:
        raise SystemExit(
            f"CHECKPOINT DIGEST PREFLIGHT FAILED for {arm}: {path} hashes to\n"
            f"  {digest}\nbut the reviewed comparator artifact is\n  {want}\n"
            "These are not the same bytes, so this is not the comparator the "
            "campaign approved -- refusing to evaluate it under that row's name.")
    return digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_list = sub.add_parser("commands", help="print every cell's command, one per line")
    p_list.add_argument("--arm", choices=list(ARM_ORDER), default=None)
    p_list.add_argument("--placeholder", default=None,
                        help="stand-in for an unresolved checkpoint epoch (dry runs)")
    p_list.add_argument("--with-paths", action="store_true",
                        help="prefix each line with the metrics JSON it produces")
    p_pre = sub.add_parser("preflight", help="hold an arm's checkpoint to its contract")
    p_pre.add_argument("arm", choices=list(ARM_ORDER))
    args = parser.parse_args(argv)

    if args.cmd == "preflight":
        print(f"{args.arm} trained-as preflight OK; ckpt_sha256 {preflight(args.arm)}")
        return 0

    for cell in inventory():
        if args.arm and cell.arm != args.arm:
            continue
        argv_tokens = command(cell.arm, cell.k, cell.seed, cell.rotate_deg,
                              placeholder=args.placeholder)
        line = " ".join(argv_tokens)
        if args.with_paths:
            line = (os.path.relpath(metrics_path(cell.arm, cell.k, cell.seed,
                                                 cell.rotate_deg,
                                                 placeholder=args.placeholder),
                                    repo_root()) + "\t" + line)
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
