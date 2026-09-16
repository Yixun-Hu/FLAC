"""Every name and command line of the exp_14 data curve, generated in one place.

The experiment trains six runs (two backbones x three data fractions) that differ *only*
in which split file they read, and evaluates each on 10 cells (K in {1, 8} x seeds
42-46). Both halves are error-prone in ways that are invisible after the fact, so nothing
here is ever typed into a shell:

* **Naming (plan §4 Round D, finding r3-3).** FLAC's ``gen_model_comparison.py``
  classifies a results row by substring: ``is_exp14_row`` claims *any* row whose glob
  contains ``exp14_`` for the exp_14 **yaw** campaign (wrong protocol label, wrong
  validator). Run ids are therefore ``dc_<arm>_f<tag>`` and eval names
  ``dc_<arm>_f<tag>_K<K>_s<seed>``; ``assert_no_forbidden_substring`` is applied to every
  generated identifier. The NAS *directory* keeps the announcement-07 form
  ``checkpoints/exp14_data_curve/<run id>/`` -- it never appears in a row glob.
* **Protocol flags (announcement 05, CLAUDE.md "Eval-protocol flags").** The cyl arm is
  trained and scored with ``--cond-method fa_invariant``, the van arm with ``vanilla``;
  a mismatch reads as a plausible but catastrophically wrong number. ``eval_argv`` derives
  the flags from the arm and the dataset config from K, so a cell cannot be scored under
  the other arm's protocol.

``metrics_json_path`` / ``predictions_pt_path`` mirror ``eval_FLAC.build_output_paths``
(pinned by a test that compares against the real function) so the launcher's skip logic
looks exactly where the evaluator writes.
"""
import os

#: fraction tag -> fraction. The tag is the filename/run-id form ("025"), the value the
#: number that goes into the run contract.
FRACTIONS = {"025": 0.25, "050": 0.5, "075": 0.75}
ARMS = ("cyl", "van")
SEEDS = (42, 43, 44, 45, 46)
K_VALUES = (1, 8)

#: The substring that would misclassify a model_comparison row (finding r3-3).
FORBIDDEN_SUBSTRING = "exp14_"

#: arm -> ``--cond-method``. ``fa_invariant`` with a single angle is what the cyl arm is
#: trained with (announcement 06: orbit size 1, no chunking on this pin).
ARM_COND_METHOD = {"cyl": "fa_invariant", "van": "vanilla"}

#: K -> the existing full unseen-eval dataset config (announcement 01: never subsampled).
EVAL_DATASET_CONFIGS = {
    1: "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json",
    8: "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json",
}
#: fraction tag -> the training dataset config that reads that fraction's split.
TRAIN_DATASET_CONFIGS = {
    tag: f"src/configs/dataset_configs/AR/train/acousticroom_train_frac{tag}.json"
    for tag in FRACTIONS
}
#: fraction tag -> the split file itself (the run contract's ``split_sha256`` input).
SPLIT_FILES = {tag: f"data/AR/train_frac{tag}_s2026.json" for tag in FRACTIONS}

# The frozen recipe (plan §2): identical to exp_13 tier B / exp_07 P1.
PRETRANSFORM_CKPT = "weights/FLAC/VAE.safetensors"
MAX_STEPS = 40000
MICRO_BATCH = 32
ACCUM_BATCHES = 1
NUM_WORKERS = 6
TRAIN_SEED = 42
NUM_GPUS = 2
STRATEGY = "ddp_find_unused_parameters_true"
SYNC_BATCHNORM = "true"
LOGGER = "none"
CHECKPOINT_EVERY = 2500

# The frozen eval protocol (plan §2).
EVAL_STEPS = 1
EVAL_CFG_SCALE = 1.0
FRAME_AVG_ANGLES = "0"
ROTATE_DEG = 0.0
COND_AUTOCAST = "bf16"
#: The full unseen-eval split, and the decoded RIR length of the FLAC configs -- the
#: shape ``--store_predictions`` must produce for a complete cell.
N_ITEMS_UNSEEN = 6337
SAMPLE_LEN = 10240


def assert_no_forbidden_substring(s):
    """Return ``s`` unless it carries ``exp14_`` (finding r3-3), else ``ValueError``."""
    if not isinstance(s, str):
        raise ValueError(f"expected a string, got {type(s).__name__}")
    if FORBIDDEN_SUBSTRING in s:
        raise ValueError(
            f"{s!r} contains {FORBIDDEN_SUBSTRING!r}: gen_model_comparison.py would "
            "classify this row as the exp_14 yaw campaign (wrong protocol label and "
            "validator). Use the dc_* naming of plan §4 Round D."
        )
    return s


def _check_arm(arm):
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    return arm


def _check_tag(tag):
    if not isinstance(tag, str) or tag not in FRACTIONS:
        raise ValueError(f"unknown fraction tag {tag!r}; expected one of {sorted(FRACTIONS)}")
    return tag


def _check_k(k):
    if isinstance(k, bool) or k not in K_VALUES:
        raise ValueError(f"unplanned K={k!r}; the experiment evaluates {K_VALUES}")
    return k


def _check_seed(seed):
    if isinstance(seed, bool) or seed not in SEEDS:
        raise ValueError(f"unplanned seed {seed!r}; the experiment evaluates {SEEDS}")
    return seed


def run_id(arm, tag):
    """``dc_cyl_f025`` -- the training run's ``--name``/``--experiment-name`` and NAS dir."""
    return assert_no_forbidden_substring(f"dc_{_check_arm(arm)}_f{_check_tag(tag)}")


def eval_name(arm, tag, K, seed):
    """``dc_cyl_f025_K8_s42`` -- one evaluation cell's ``--eval-name``."""
    return assert_no_forbidden_substring(
        f"{run_id(arm, tag)}_K{_check_k(K)}_s{_check_seed(seed)}"
    )


def _output_paths(ckpt_path, arm, tag, K, seed):
    """Mirror of ``eval_FLAC.build_output_paths`` for this experiment's fixed protocol.

    ``steps=1``, ``cfg_scale=1.0``, ``rotate_deg=0`` (no ``_rot`` suffix) and one frame
    angle, so the cyl suffix is ``_fa_invariant_a1`` and the van suffix is empty.
    """
    arm = _check_arm(arm)
    stem_name = eval_name(arm, tag, K, seed)
    method = ARM_COND_METHOD[arm]
    suffix = "" if method == "vanilla" else f"_{method}_a1"
    ckpt_name = os.path.basename(ckpt_path).replace(".ckpt", "")
    directory = os.path.dirname(ckpt_path)
    stem = f"{EVAL_STEPS}_{EVAL_CFG_SCALE}_{stem_name}{suffix}"
    return {
        "metrics": os.path.join(directory, f"{ckpt_name}_metrics_{stem}.json"),
        "predictions": os.path.join(directory, f"{ckpt_name}_predictions_{stem}.pt"),
    }


def metrics_json_path(ckpt_path, arm, tag, K, seed):
    """Where ``eval_FLAC`` writes this cell's metrics JSON (next to the checkpoint)."""
    path = _output_paths(ckpt_path, arm, tag, K, seed)["metrics"]
    assert_no_forbidden_substring(os.path.basename(path))
    return path


def predictions_pt_path(ckpt_path, arm, tag, K, seed):
    """Where ``eval_FLAC --store_predictions`` writes this cell's prediction bundle."""
    path = _output_paths(ckpt_path, arm, tag, K, seed)["predictions"]
    assert_no_forbidden_substring(os.path.basename(path))
    return path


#: arm -> the kit model config's basename. Both files are sha-pinned by
#: ``src.tools.data_curve.verify``; requiring the exact basename here means an arm can
#: never be launched or scored with the other arm's architecture.
ARM_MODEL_CONFIG_BASENAMES = {
    "cyl": "FLAC_AR_exp14_cylS.json",
    "van": "FLAC_AR_exp14_vanS.json",
}


def _check_model_config(arm, model_config):
    expected = ARM_MODEL_CONFIG_BASENAMES[arm]
    if not isinstance(model_config, str) or os.path.basename(model_config) != expected:
        raise ValueError(
            f"arm {arm!r} must be run with {expected}, got {model_config!r}"
        )
    return model_config


def _check_run_dir(run, run_dir, what):
    if not isinstance(run_dir, str) or not run_dir:
        raise ValueError(f"{what} must be a non-empty path, got {run_dir!r}")
    if os.path.basename(run_dir.rstrip("/")) != run:
        raise ValueError(
            f"{what} {run_dir!r} is not the run directory of {run!r}: checkpoints and the "
            "contract of one run never live under another run's name"
        )
    return run_dir.rstrip("/")


def train_argv(arm, tag, model_config, dataset_config, save_dir, run_contract_json,
               ckpt_path=None):
    """The frozen training command of one run (plan §2; == exp_13 tier B == exp_07 P1).

    Every value is fixed by the recipe; the four paths are inputs because they differ
    between the kit, the worktree and the NAS. Three substitutions are fail-closed,
    because each produces a run that looks healthy and means something else:

    * ``dataset_config`` must be *this* fraction's config (a wrong one trains on another
      fraction while every record says otherwise),
    * ``model_config`` must be this arm's config,
    * ``save_dir`` must be the run's own directory and ``run_contract_json`` the
      ``run_contract.json`` sidecar inside it (a foreign contract would be embedded in
      every checkpoint this run writes).

    ``ckpt_path`` is appended **only** when resuming -- and a resume is a fresh stochastic
    continuation (PL restores no RNG or dataloader position), which the launcher discloses.
    """
    run = run_id(arm, tag)
    _check_model_config(arm, model_config)
    expected_ds = TRAIN_DATASET_CONFIGS[tag]
    if os.path.basename(str(dataset_config)) != os.path.basename(expected_ds):
        raise ValueError(
            f"fraction {tag} must be trained with {os.path.basename(expected_ds)}, got "
            f"{dataset_config!r}"
        )
    save_dir = _check_run_dir(run, save_dir, "--save-dir")
    if (os.path.dirname(str(run_contract_json)) != save_dir
            or os.path.basename(str(run_contract_json)) != "run_contract.json"):
        raise ValueError(
            f"--run-contract-json must be {save_dir}/run_contract.json, got "
            f"{run_contract_json!r}"
        )
    argv = [
        "python", "train.py",
        "--model-config", model_config,
        "--dataset-config", dataset_config,
        "--pretransform-ckpt-path", PRETRANSFORM_CKPT,
        "--max-steps", str(MAX_STEPS),
        "--batch-size", str(MICRO_BATCH),
        "--accum-batches", str(ACCUM_BATCHES),
        "--num-workers", str(NUM_WORKERS),
        "--seed", str(TRAIN_SEED),
        "--num-gpus", str(NUM_GPUS),
        "--strategy", STRATEGY,
        "--sync-batchnorm", SYNC_BATCHNORM,
        "--logger", LOGGER,
        "--checkpoint-every", str(CHECKPOINT_EVERY),
        "--name", run,
        "--experiment-name", run,
        "--save-dir", save_dir,
        "--run-contract-json", run_contract_json,
    ]
    if ckpt_path:
        argv += ["--ckpt-path", ckpt_path]
    return argv


def eval_argv(arm, tag, K, seed, model_config, ckpt_path):
    """The frozen evaluation command of one cell (plan §2, announcement 05).

    All four conditioning flags are explicit for **both** arms, K selects one of the two
    existing full unseen-eval configs (announcement 01: never a subsampled eval), and the
    checkpoint must live in this run's directory -- scoring one arm's checkpoint under the
    other's protocol is the exp_09 protocol error, and it is not detectable in the numbers.
    """
    run = run_id(arm, tag)
    _check_model_config(arm, model_config)
    _check_run_dir(run, os.path.dirname(str(ckpt_path)), "--ckpt-path's directory")
    return [
        "python", "eval_FLAC.py",
        "--model-config", model_config,
        "--dataset-config", EVAL_DATASET_CONFIGS[_check_k(K)],
        "--ckpt-path", ckpt_path,
        "--cond-method", ARM_COND_METHOD[arm],
        "--frame-avg-angles", FRAME_AVG_ANGLES,
        "--rotate-deg", str(int(ROTATE_DEG)),
        "--cond-autocast", COND_AUTOCAST,
        "--seed", str(_check_seed(seed)),
        "--steps", str(EVAL_STEPS),
        "--cfg-scale", str(EVAL_CFG_SCALE),
        "--eval-name", eval_name(arm, tag, K, seed),
        "--store_predictions",
    ]


# ======================================================================================
# check-bundle: is a stored prediction bundle really this cell's, exactly as scored?
# ======================================================================================
# A cell counts as complete only when its metrics JSON *and* a prediction bundle that
# proves the protocol exist (plan §2, announcement 08). The metrics JSON carries neither
# the seed nor the dataset config, so the bundle's meta is the only full-split provenance
# there is -- and the launcher skips an existing cell on the strength of this check.
EXIT_OK = 0
EXIT_INPUT_ERROR = 2        # the caller's own arguments are wrong (a launcher bug)
EXIT_BUNDLE_VIOLATION = 3   # the artifact disagrees with the cell it claims to be


def check_bundle(path, expect_n, expect_seed, expect_K, expect_arm, expect_eval_name=None):
    """Return the list of violations (empty == the bundle is this cell's, exactly as scored).

    Loads on CPU with ``weights_only=False`` (the bundle is a dict of a tensor and a meta
    dict written by ``torch.save``). An unreadable, truncated or malformed file is a
    violation, not an exception: a half-written bundle simply means the cell is not done.
    ``ValueError`` is raised only for an unplanned arm/K/seed -- that is a bug in the
    caller, not a verdict about the artifact.
    """
    import torch  # deferred: the name/argv helpers must stay importable without torch

    _check_arm(expect_arm)
    _check_k(expect_K)
    _check_seed(expect_seed)
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as err:
        return [f"{path}: cannot be loaded ({type(err).__name__}: {err})"]
    if not isinstance(bundle, dict) or "meta" not in bundle or "predictions" not in bundle:
        return [f"{path}: not a prediction bundle (no 'predictions'/'meta' keys)"]
    meta = bundle["meta"]
    if not isinstance(meta, dict):
        return [f"{path}: meta is a {type(meta).__name__}, not a dict"]

    violations = []

    def expect(key, value, wanted):
        if value != wanted:
            violations.append(f"meta.{key} is {value!r}, expected {wanted!r}")

    expect("n_items", meta.get("n_items"), expect_n)
    if "n_samples" in meta:
        expect("n_samples", meta.get("n_samples"), expect_n)
    expect("seed", meta.get("seed"), expect_seed)
    got_ds = os.path.basename(str(meta.get("dataset_config")))
    want_ds = os.path.basename(EVAL_DATASET_CONFIGS[expect_K])
    if got_ds != want_ds:
        violations.append(
            f"meta.dataset_config is {got_ds!r}, expected {want_ds!r} for K={expect_K}"
        )
    expect("cond_method", meta.get("cond_method"), ARM_COND_METHOD[expect_arm])
    want_angles = [float(FRAME_AVG_ANGLES)] if expect_arm == "cyl" else None
    angles = meta.get("frame_avg_angles")
    if angles is not None:
        try:
            angles = [float(a) for a in angles]
        except (TypeError, ValueError):
            pass
    expect("frame_avg_angles", angles, want_angles)
    rotate = meta.get("rotate_deg")
    if not isinstance(rotate, (int, float)) or float(rotate) != ROTATE_DEG:
        violations.append(f"meta.rotate_deg is {rotate!r}, expected {ROTATE_DEG}")
    expect("cond_autocast", meta.get("cond_autocast"), COND_AUTOCAST)
    # Round C: only a bundle stored *after* the clamp/pad is the tensor that was scored.
    expect("stored_after_clamp_pad", meta.get("stored_after_clamp_pad"), True)
    if expect_eval_name is not None:
        expect("eval_name", meta.get("eval_name"), expect_eval_name)

    predictions = bundle["predictions"]
    if not torch.is_tensor(predictions):
        violations.append(f"predictions is a {type(predictions).__name__}, not a tensor")
    else:
        wanted_shape = (expect_n, 1, SAMPLE_LEN)
        if tuple(predictions.shape) != wanted_shape:
            violations.append(
                f"predictions shape is {tuple(predictions.shape)}, expected {wanted_shape}"
            )
        if not bool(torch.isfinite(predictions.float()).all()):
            violations.append("predictions contain non-finite values")
    return violations


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.tools.data_curve.names",
        description="Check that a stored prediction bundle is the cell it claims to be.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check-bundle", help="validate one --store_predictions bundle")
    check.add_argument("--pt", required=True, help="the *_predictions_*.pt bundle")
    check.add_argument("--expect-n", required=True, type=int)
    check.add_argument("--expect-seed", required=True, type=int)
    check.add_argument("--expect-K", required=True, type=int)
    check.add_argument("--expect-arm", required=True, choices=list(ARMS))
    check.add_argument("--expect-eval-name", default=None)
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    try:
        violations = check_bundle(args.pt, args.expect_n, args.expect_seed, args.expect_K,
                                  args.expect_arm, args.expect_eval_name)
    except ValueError as err:
        print(f"check-bundle called with bad arguments: {err}")
        return EXIT_INPUT_ERROR
    if violations:
        for violation in violations:
            print(f"FAIL {args.pt}: {violation}")
        return EXIT_BUNDLE_VIOLATION
    print(f"PASS {args.pt}: n={args.expect_n} seed={args.expect_seed} K={args.expect_K} "
          f"arm={args.expect_arm} ({ARM_COND_METHOD[args.expect_arm]}, autocast "
          f"{COND_AUTOCAST}, rotate {int(ROTATE_DEG)})")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
