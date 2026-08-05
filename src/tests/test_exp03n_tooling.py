"""exp_03 (worklog_yixun_neuronic) — tests for the exp03n TOOLING: the pre-launch
arm-wiring gate, the trained-checkpoint equivariance guard, and the launch/probe/eval
script contracts.

Companion to ``test_exp03n_cond_pool.py`` (which pins the model delta itself). Everything
here is CPU-only and offline; the gate/guard helpers are exercised on TINY random-weight
cylindrical backbones, never on the Hub.
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import copy  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_EXP09_DIR = _REPO_ROOT / "worklog" / "worklog_yixun" / "exp_09_cyl_no_ssl"
_GATE_PATH = _EXP09_DIR / "assert_arm_configs_exp03n.py"

from src.tests.test_exp03n_cond_pool import (  # noqa: E402
    _HIDDEN,
    _build,
    _cyl_conditioning,
    _geoms,
    _save_tiny_cyl,
)


def _load_module(path: Path, name: str):
    if not path.exists():
        pytest.fail(f"required deliverable is missing: {path}")
    if str(_EXP09_DIR) not in sys.path:
        sys.path.insert(0, str(_EXP09_DIR))
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate():
    return _load_module(_GATE_PATH, "assert_arm_configs_exp03n")


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    return _save_tiny_cyl(tmp_path_factory.mktemp("exp03n_tooling_tiny"))


# ------------------------------------------------------------------------------------ #
# 1. gate: per-variant config contract
# ------------------------------------------------------------------------------------ #
def test_gate_accepts_both_real_config_variants(gate):
    for name, variant in (("FLAC_AR_exp03n.json", "base"),
                          ("FLAC_AR_exp03n_online_eval.json", "online")):
        cfg, resolved = gate.assert_config_contract(str(_EXP09_DIR / name))
        assert resolved == variant, (name, resolved)
        assert cfg["model"]["conditioning"]["configs"][1]["config"]["ViT"]["cond_pool"] == "max_mlp"


def test_gate_variant_autodetect_is_filename_driven(gate):
    assert gate.variant_for_config("/x/FLAC_AR_exp03n_online_eval.json") == "online"
    assert gate.variant_for_config("/x/FLAC_AR_exp03n.json") == "base"


def test_gate_rejects_the_wrong_reference_for_a_variant(gate, tmp_path):
    """Binding the ONLINE file to the BASE delta (or vice versa) must fail: the two
    references differ in use_ema and the gradient_checkpointing keys."""
    with pytest.raises(RuntimeError, match="contract|reconstruct"):
        gate.assert_config_contract(str(_EXP09_DIR / "FLAC_AR_exp03n_online_eval.json"),
                                    variant="base")
    with pytest.raises(RuntimeError, match="contract|reconstruct"):
        gate.assert_config_contract(str(_EXP09_DIR / "FLAC_AR_exp03n.json"), variant="online")


@pytest.mark.parametrize("mutation", ["extra_key", "changed_recipe", "dropped_key",
                                      "wrong_width", "one_block_only"])
def test_gate_rejects_mutated_configs(gate, tmp_path, mutation):
    cfg = json.load(open(_EXP09_DIR / "FLAC_AR_exp03n.json"))
    vits = [c for c in cfg["model"]["conditioning"]["configs"] if c["type"] == "ViTCoordinates"]
    if mutation == "extra_key":
        vits[0]["config"]["ViT"]["cond_mlp_dropout"] = 0.1
        vits[1]["config"]["ViT"]["cond_mlp_dropout"] = 0.1
    elif mutation == "changed_recipe":
        cfg["training"]["frame_avg_angles"] = [0.0, 90.0, 180.0, 270.0]
    elif mutation == "dropped_key":
        del vits[1]["config"]["ViT"]["cond_pool"]
    elif mutation == "wrong_width":
        vits[1]["config"]["ViT"]["cond_mlp_hidden"] = 512
    elif mutation == "one_block_only":
        del vits[1]["config"]["ViT"]["cond_pool"]
        del vits[1]["config"]["ViT"]["cond_mlp_hidden"]
    path = tmp_path / "FLAC_AR_exp03n.json"
    path.write_text(json.dumps(cfg, indent=4) + "\n")
    with pytest.raises(RuntimeError):
        gate.assert_config_contract(str(path))


# ------------------------------------------------------------------------------------ #
# 2. gate: instantiated-model arm wiring
# ------------------------------------------------------------------------------------ #
def test_gate_head_wiring_accepts_a_correct_max_mlp_build(gate, tiny_dir):
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="max_mlp"))
    head = gate.assert_head_wiring(mc, hidden_size=_HIDDEN, cond_dim=256)
    assert head is _geoms(mc)[0].lin_proj


def test_gate_head_wiring_rejects_the_legacy_build(gate, tiny_dir):
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True))
    with pytest.raises(RuntimeError):
        gate.assert_head_wiring(mc, hidden_size=_HIDDEN, cond_dim=256)


def test_gate_head_wiring_rejects_an_unshared_head(gate, tiny_dir):
    """A second, independently-built head would double the new parameters and break the
    shared-head contract — the gate must notice even though every shape still matches."""
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="max_mlp"))
    geoms = _geoms(mc)
    geoms[1].lin_proj = copy.deepcopy(geoms[0].lin_proj)
    with pytest.raises(RuntimeError, match="shared"):
        gate.assert_head_wiring(mc, hidden_size=_HIDDEN, cond_dim=256)


def test_gate_head_wiring_rejects_a_mean_pooled_conditioner(gate, tiny_dir):
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="max_mlp"))
    _geoms(mc)[1].dino_pool = "mean"
    with pytest.raises(RuntimeError, match="dino_pool"):
        gate.assert_head_wiring(mc, hidden_size=_HIDDEN, cond_dim=256)


def test_gate_param_delta_is_the_hidden_layer_only(gate, tiny_dir):
    legacy = _build(_cyl_conditioning(tiny_dir, with_context=True))
    new = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="max_mlp"))
    gate.assert_param_delta(legacy, new, expected=_HIDDEN * _HIDDEN + _HIDDEN)
    with pytest.raises(RuntimeError, match="trainable"):
        gate.assert_param_delta(legacy, new, expected=147840)


def test_gate_live_forward_matches_the_external_oracle(gate, tiny_dir):
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="max_mlp"))
    gate.assert_live_forward(mc, height=64, width=128)


def test_gate_live_forward_catches_a_mean_pooling_regression(gate, tiny_dir):
    """Sabotage: flip the served pooling back to the mean while the head stays the MLP.
    Shapes and parameter counts are unchanged, so ONLY the live forward can catch it."""
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="max_mlp"))
    for geom in _geoms(mc):
        geom.dino_pool = "mean"
    with pytest.raises(RuntimeError, match="amax|forward"):
        gate.assert_live_forward(mc, height=64, width=128)


def test_gate_common_init_identity_on_tiny_full_models(gate, tiny_dir):
    from src.tests.test_exp03n_cond_pool import _full_model_config

    new_cfg = _full_model_config(tiny_dir, max_mlp=True)
    legacy_cfg = _full_model_config(tiny_dir, max_mlp=False)
    gate.assert_common_init_identity(new_cfg, legacy_cfg, seed=42)


def test_gate_common_init_identity_catches_a_perturbed_output_layer(gate, tiny_dir, monkeypatch):
    """Sabotage: draw the hidden layer from the GLOBAL stream (no fork_rng). Every module
    built AFTER the conditioner then initialises differently — exactly the one-factor
    violation this assertion exists to catch."""
    from src.tests.test_exp03n_cond_pool import _full_model_config

    import src.models.conditioners as cond_mod

    def _leaky_fork_rng(*args, **kwargs):
        import contextlib

        return contextlib.nullcontext()

    monkeypatch.setattr(cond_mod.torch.random, "fork_rng", _leaky_fork_rng)
    new_cfg = _full_model_config(tiny_dir, max_mlp=True)
    legacy_cfg = _full_model_config(tiny_dir, max_mlp=False)
    with pytest.raises(RuntimeError):
        gate.assert_common_init_identity(new_cfg, legacy_cfg, seed=42)


# ------------------------------------------------------------------------------------ #
# 3. trained-checkpoint equivariance guard (plan §4.5b)
# ------------------------------------------------------------------------------------ #
_GUARD_PATH = _EXP09_DIR / "guard_exp03n_equivariance.py"

_TINY_GEOM = dict(height=64, width=128)   # 4x8 tokens at patch 16 -> W_t = 8


@pytest.fixture(scope="module")
def guard():
    return _load_module(_GUARD_PATH, "guard_exp03n_equivariance")


def test_guard_passes_on_a_random_weight_mini_model(guard, tiny_dir):
    """The guard tests the ACTUAL served condition MLP(amax(tokens)) — architectural, so it
    must hold at random init, not only on trained weights."""
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="max_mlp"))
    residuals = guard.check_equivariance(_geoms(mc)[0], **_TINY_GEOM)
    assert set(residuals) == {90.0, 180.0, 270.0}
    assert max(residuals.values()) <= guard.DEFAULT_BOUND, residuals


def test_guard_negative_control_gauge_off_raises(guard, tiny_dir, tmp_path):
    gauge_off = _save_tiny_cyl(tmp_path / "tiny_gauge_off")
    mc = _build(_cyl_conditioning(gauge_off, with_context=True, cond_pool="max_mlp", gauge="none"))
    with pytest.raises(RuntimeError, match="equivarian|residual"):
        guard.check_equivariance(_geoms(mc)[0], **_TINY_GEOM)


def test_guard_refuses_an_angle_that_is_not_a_whole_token_column(guard, tiny_dir):
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="max_mlp"))
    with pytest.raises(RuntimeError, match="column"):
        guard.check_equivariance(_geoms(mc)[0], angles=(30.0,), **_TINY_GEOM)


def test_guard_refuses_a_mean_pooled_conditioner(guard, tiny_dir):
    """Scope guard: this file certifies the max-pool head. Pointing it at a legacy build
    must REFUSE rather than silently certify the wrong served condition."""
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True))
    with pytest.raises(RuntimeError, match="dino_pool|max"):
        guard.check_equivariance(_geoms(mc)[0], **_TINY_GEOM)


def _write_pl_checkpoint(tmp_path, tiny_dir):
    """A Lightning-shaped checkpoint of a TINY exp03n model (``diffusion.``-prefixed
    state dict), plus the model-config JSON that reproduces it."""
    from src.models import create_model_from_config
    from src.tests.test_exp03n_cond_pool import _full_model_config

    cfg = _full_model_config(tiny_dir, max_mlp=True)
    cfg_path = tmp_path / "FLAC_AR_exp03n.json"
    cfg_path.write_text(json.dumps(cfg, indent=4) + "\n")
    torch.manual_seed(3)
    model = create_model_from_config(copy.deepcopy(cfg))
    ckpt_path = tmp_path / "epoch=0-step=40.ckpt"
    torch.save({"state_dict": {f"diffusion.{k}": v for k, v in model.state_dict().items()}},
               ckpt_path)
    return str(ckpt_path), str(cfg_path), model


def test_guard_loads_a_lightning_checkpoint_and_serves_the_max_pool_head(guard, tiny_dir, tmp_path):
    ckpt_path, cfg_path, model = _write_pl_checkpoint(tmp_path, tiny_dir)
    geoms = guard.load_geometry_conditioners(ckpt_path, cfg_path, device="cpu")
    assert len(geoms) == 2
    assert all(g.dino_pool == "max" for g in geoms)
    ref = model.conditioner.conditioners["source_vit"].lin_proj
    assert torch.equal(geoms[0].lin_proj[0].weight, ref[0].weight), "hidden layer did not load"
    assert torch.equal(geoms[0].lin_proj[2].weight, ref[2].weight), "output layer did not load"


def test_guard_cli_returns_zero_on_a_loadable_checkpoint(guard, tiny_dir, tmp_path, capsys):
    ckpt_path, cfg_path, _ = _write_pl_checkpoint(tmp_path, tiny_dir)
    rc = guard.main([ckpt_path, "--model-config", cfg_path, "--height", "64", "--width", "128"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "GUARD PASS" in out, out


# ------------------------------------------------------------------------------------ #
# 4. launch / probe / eval script contracts (plan §4.4, §4.5, §5.1)
# ------------------------------------------------------------------------------------ #
import difflib  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402

_LAUNCH = _EXP09_DIR / "exp03n_launch.sh"
_EXP09_LAUNCH = _EXP09_DIR / "exp09_launch.sh"
_PROBE = _EXP09_DIR / "exp03n_probe.sh"
# The Slurm wrappers live in the CYLINDRICAL repo (they are that repo's records); the
# contract is asserted here because this is where the trap lives. Skipped when that
# checkout is not beside this one.
_SLURM_DIR = Path("/n/fs/gatrdp/codespace/cylindrical-dinov3/slurm_neuronic")
_WORKTREE_ROOT_ABS = "/n/fs/gatrdp/codespace/exp03-maxpool-mlp-cond"
_CYL_SRC_ABS = "/n/fs/gatrdp/codespace/cylindrical-dinov3/src"
_RECORDS_DIR_ABS = ("/n/fs/gatrdp/codespace/cylindrical-dinov3/worklog/worklog_yixun_neuronic/"
                    "exp_03_maxpool_mlp_cond_claude")


def _text(path: Path) -> str:
    if not path.exists():
        pytest.fail(f"required deliverable is missing: {path}")
    return path.read_text()


def _sbatch(name: str) -> str:
    path = _SLURM_DIR / name
    if not _SLURM_DIR.exists():
        pytest.skip(f"cylindrical slurm_neuronic checkout not present at {_SLURM_DIR}")
    if not path.exists():
        pytest.fail(f"required deliverable is missing: {path}")
    return path.read_text()


@pytest.mark.parametrize("script", ["exp03n_launch.sh", "exp03n_probe.sh"])
def test_shell_scripts_parse(script):
    path = _EXP09_DIR / script
    _text(path)
    rc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr


def test_launcher_every_diff_from_exp09_is_marked_and_numbered():
    """House style (exp_02's p1_rerun_launch.sh): the launcher is a COPY of the reviewed
    exp-09 launcher in which every changed/added line carries a ``# EXP03 DIFF n/N`` marker,
    so a reviewer can diff the two files and account for each line."""
    exp09 = _text(_EXP09_LAUNCH).splitlines()
    exp03n = _text(_LAUNCH).splitlines()
    added = [line for line in difflib.unified_diff(exp09, exp03n, n=0, lineterm="")
             if line.startswith("+") and not line.startswith("+++")]
    assert added, "the launcher is byte-identical to exp09_launch.sh — nothing was adapted"
    markers = []
    for line in added:
        m = re.search(r"EXP03 DIFF (\d+)/(\d+)", line)
        assert m, f"unmarked diff line in exp03n_launch.sh: {line[1:][:120]}"
        markers.append((int(m.group(1)), int(m.group(2))))
    totals = {t for _, t in markers}
    assert len(totals) == 1, f"inconsistent diff totals: {sorted(totals)}"
    total = totals.pop()
    assert sorted(n for n, _ in markers) == list(range(1, total + 1)), sorted(markers)
    assert len(added) == total, f"{len(added)} changed lines but the markers claim {total}"


_SCIENTIFIC_FLAGS = (
    '--max-steps 67500 --batch-size "$MB" --accum-batches "$ACC" --num-workers 6 --seed 42',
    "--num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true",
    '--logger "$LOGGER" --checkpoint-every 2500',
    "--dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json",
    "--pretransform-ckpt-path weights/FLAC/VAE.safetensors",
)


def test_launcher_keeps_every_scientific_flag_byte_identical():
    launch, exp09 = _text(_LAUNCH), _text(_EXP09_LAUNCH)
    for flag_line in _SCIENTIFIC_FLAGS:
        assert flag_line in exp09, f"fixture drift: {flag_line!r} not in exp09_launch.sh"
        assert flag_line in launch, f"scientific flag line changed: {flag_line!r}"
    # the BN-compliant rung pin survives
    assert '[ "$MB" = "32" ] && [ "$ACC" = "1" ]' in launch


def test_launcher_binds_the_exp03n_arm_not_exp09():
    launch = _text(_LAUNCH)
    assert f"cd {_WORKTREE_ROOT_ABS}" in launch, "the launcher must cd to the absolute worktree root"
    assert f"export PYTHONPATH={_CYL_SRC_ABS}" in launch
    assert "FLAC_AR_exp03n.json" in launch
    assert "FLAC_AR_exp09.json" not in launch
    assert "assert_arm_configs_exp03n.py" in launch
    assert "assert_arm_configs_exp09.py" not in launch
    assert "--name FLAC_exp03n_maxpoolmlp --experiment-name exp03n_maxpoolmlp" in launch
    assert "--save-dir /n/fs/gatrdp/outputs/exp03n_maxpoolmlp" in launch
    assert _RECORDS_DIR_ABS in launch, "the teed log must land in the exp_03 records folder"
    assert "maxpool_mlp_cond_${TS}_j${SLURM_JOB_ID:-nojob}_train.log" in launch


def test_launcher_refuses_without_the_frozen_vram_file_and_the_two_pins():
    launch = _text(_LAUNCH)
    assert "exp03n_frozen_min_free.txt" in launch
    assert "c1_frozen_min_free.txt" not in launch
    assert 'FROZEN_FILE" ] || {' in launch, "the frozen-file REFUSE must survive"
    assert 'EXPECT_PACKAGE_SHA' in launch and 'EXPECT_EXP09_SHA' in launch
    # both pins REQUIRED: an empty value must abort, never default
    assert re.search(r'\[ -n "\$EXPECT_PACKAGE_SHA" \].*\[ -n "\$EXPECT_EXP09_SHA" \]', launch), (
        "the launcher must REFUSE when either pin is absent"
    )


def test_launcher_wandb_identity_gate_uses_netrc_not_bashrc():
    launch = _text(_LAUNCH)
    assert ".bashrc" not in launch, "the ~/.bashrc WANDB_API_KEY grep must be dropped"
    assert "netrc" in launch
    assert "yh4742@princeton.edu" in launch, "the identity gate must still pin the account"


def test_probe_is_the_two_phase_lifecycle_probe_with_the_launcher_gates():
    probe = _text(_PROBE)
    assert "--max-steps 30" in probe and "--checkpoint-every 10" in probe
    assert "--max-steps 40" in probe and "--ckpt-path" in probe
    assert "step=30" in probe and "step=40" in probe
    assert re.search(r'SAVE="\$\{SAVE:-/n/fs/gatrdp/outputs/exp03n_maxpoolmlp_PROBE\}"', probe), (
        "the probe must default to its own _PROBE save dir"
    )
    assert re.search(r'\[ ! -e "\$SAVE" \]', probe), "the probe must refuse an existing save dir"
    # same gates as the launcher EXCEPT the frozen-VRAM file (this run PRODUCES it)
    assert "assert_arm_configs_exp03n.py" in probe
    assert "FLAC_AR_exp03n.json" in probe
    assert f"export PYTHONPATH={_CYL_SRC_ABS}" in probe
    assert "EXPECT_PACKAGE_SHA" in probe and "EXPECT_EXP09_SHA" in probe
    assert "exp03n_frozen_min_free.txt" in probe, "the probe must name the file it feeds"
    assert "21900" in probe, "the provisional free-VRAM floor must be pinned"
    assert "1.15" in probe, "the probe must print the recommended 1.15x frozen value"
    assert ".bashrc" not in probe
    # the probe's scientific shape matches the arm
    assert "--batch-size 32" in probe and "--accum-batches 1" in probe
    assert "--sync-batchnorm true" in probe and "ddp_find_unused_parameters_true" in probe
    assert "--seed 42" in probe and "--num-gpus 2" in probe


# ---- Slurm wrappers (cylindrical repo) ---------------------------------------------- #
@pytest.mark.parametrize("name,gpus", [("train_exp03n.sbatch", "2"), ("probe_exp03n.sbatch", "2")])
def test_gpu_wrappers_pin_and_assert_their_launch_shape(name, gpus):
    """Plan §5.1: 1 node / 2x L40 / 1 task, declared in the directives AND asserted
    fail-closed on the worker (review fix F1, applied to the training-side wrappers too)."""
    text = _sbatch(name)
    assert "#SBATCH --nodes=1" in text and "#SBATCH --ntasks=1" in text
    assert f"launch-shape: 1 node / {gpus}x L40 / 1 task" in text
    assert '[ "$NNODES" = "1" ]' in text and '[ "$NTASKS" = "1" ]' in text
    assert f'[ "$NGPU" = "{gpus}" ]' in text


def test_train_sbatch_shape_and_preflights():
    text = _sbatch("train_exp03n.sbatch")
    assert "--gres=gpu:l40:2" in text
    assert "--cpus-per-task=16" in text and "--mem=32G" in text
    assert "--time=5-00:00:00" in text
    assert "exp03n_launch.sh" in text
    assert "100" in text, "the storage floor must be present"
    assert "/n/fs/gatrdp/outputs/exp03n_maxpoolmlp" in text
    assert re.search(r"\[ ! -e /n/fs/gatrdp/outputs/exp03n_maxpoolmlp \]", text), (
        "a fresh launch must refuse an existing save dir"
    )
    assert "HF_HUB_OFFLINE=1" in text


def test_probe_sbatch_shape():
    text = _sbatch("probe_exp03n.sbatch")
    assert "--gres=gpu:l40:2" in text
    assert "--time=00:50:00" in text
    assert "exp03n_probe.sh" in text


def test_eval_sbatch_passes_the_mandatory_fa_invariant_flags():
    """THE trap (handoff): eval_FLAC.py defaults to --cond-method vanilla and ignores the
    model config, so an eval that omits these flags silently scores the raw-pose path."""
    text = _sbatch("eval_exp03n.sbatch")
    assert "--cond-method fa_invariant" in text
    assert "--frame-avg-angles 0" in text
    assert "--cond-method vanilla" not in text


# --- per-stream DRY RUN (review fix F2): the wrapper renders the command it will run, and the
# test asserts the EXACT rendered argument vector. A word search over the script would pass with
# the two stream->config assignments SWAPPED; these assertions bind stream -> config, so a swap
# fails on both cells. ---------------------------------------------------------------------- #
_STREAM_CONFIG = {
    "online": "worklog/worklog_yixun/exp_09_cyl_no_ssl/FLAC_AR_exp03n_online_eval.json",
    "ema": "worklog/worklog_yixun/exp_09_cyl_no_ssl/FLAC_AR_exp03n.json",
}
_EVAL_PROTOCOL_FLAGS = {
    "--cond-method": "fa_invariant",   # THE trap: never 'vanilla'
    "--frame-avg-angles": "0",
    "--steps": "1",
    "--cfg-scale": "1.0",
    "--batch-size": "64",
    "--num-workers": "4",
    "--device": "cuda",
    "--cond-autocast": "bf16",
}
_DS_FOR_K = {1: "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json",
             8: "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json"}


def _dry_run_eval(stream, *, step="40000", idx=0, cwd=None):
    """Execute eval_exp03n.sbatch's DRY_RUN path and return its stdout. Touches no GPU, no
    checkpoint and no file (asserted by the caller via an empty cwd)."""
    path = _SLURM_DIR / "eval_exp03n.sbatch"
    if not _SLURM_DIR.exists():
        pytest.skip(f"cylindrical slurm_neuronic checkout not present at {_SLURM_DIR}")
    if not path.exists():
        pytest.fail(f"required deliverable is missing: {path}")
    env = dict(os.environ)
    env.update(DRY_RUN="1", STEP=step, STREAM=stream,
               EXPECT_PACKAGE_SHA="deadbeefdeadbeef", EXPECT_EXP09_SHA="cafebabecafebabe",
               SLURM_ARRAY_TASK_ID=str(idx))
    proc = subprocess.run(["bash", str(path)], capture_output=True, text=True, env=env,
                          cwd=str(cwd) if cwd else None)
    assert proc.returncode == 0, f"dry run failed (rc={proc.returncode}):\n{proc.stderr}"
    return proc.stdout


def _rendered(stdout, prefix):
    lines = [l for l in stdout.splitlines() if l.startswith(prefix)]
    assert len(lines) == 1, f"expected exactly one {prefix!r} line, got {lines}"
    return lines[0][len(prefix):].strip().split()


def _flag(tokens, flag):
    assert flag in tokens, f"{flag} missing from the rendered command: {' '.join(tokens)}"
    assert tokens.count(flag) == 1, f"{flag} appears {tokens.count(flag)}x"
    return tokens[tokens.index(flag) + 1]


@pytest.mark.parametrize("stream", ["online", "ema"])
@pytest.mark.parametrize("idx,k,seed", [(0, 1, 42), (4, 1, 46), (5, 8, 42), (9, 8, 46)])
def test_eval_dry_run_renders_the_exact_command_per_stream(stream, idx, k, seed, tmp_path):
    out = _dry_run_eval(stream, step="40000", idx=idx, cwd=tmp_path)
    tokens = _rendered(out, "EVAL_CMD:")
    assert tokens[:2] == ["python", "eval_FLAC.py"], tokens[:2]

    # stream -> config: THE binding a swap must break
    assert _flag(tokens, "--model-config") == _STREAM_CONFIG[stream], (
        f"stream {stream!r} rendered the WRONG model config: {_flag(tokens, '--model-config')}"
    )
    # the pin gate must be bound to the SAME config the eval loads
    assert _flag(_rendered(out, "PIN_GATE:"), "--config") == _STREAM_CONFIG[stream]

    for flag, value in _EVAL_PROTOCOL_FLAGS.items():
        assert _flag(tokens, flag) == value, f"{flag} = {_flag(tokens, flag)!r}, expected {value!r}"
    assert "vanilla" not in tokens, "the raw-pose vanilla trap must never be rendered"

    assert _flag(tokens, "--dataset-config") == _DS_FOR_K[k]
    assert _flag(tokens, "--seed") == str(seed)
    assert _flag(tokens, "--eval-name") == f"exp03n_40000_{stream}_K{k}_s{seed}"
    assert _flag(tokens, "--ckpt-path").endswith("step=40000.ckpt")
    # the dry run is side-effect free
    assert list(tmp_path.iterdir()) == [], f"the dry run created files: {list(tmp_path.iterdir())}"


def test_eval_dry_run_streams_bind_different_configs(tmp_path):
    """Explicit swap detector: the two streams must render DIFFERENT configs, each the one
    registered for it - so exchanging the two case-branch assignments fails here too."""
    rendered = {s: _flag(_rendered(_dry_run_eval(s, cwd=tmp_path), "EVAL_CMD:"), "--model-config")
                for s in ("online", "ema")}
    assert rendered["online"] != rendered["ema"], rendered
    assert rendered == _STREAM_CONFIG, rendered
    assert "online_eval" in rendered["online"], "the online stream must load the use_ema:false config"
    assert not rendered["ema"].endswith("online_eval.json"), (
        "the EMA stream must load the training config (use_ema:true), not the online variant"
    )


def test_eval_dry_run_refuses_an_unknown_stream(tmp_path):
    path = _SLURM_DIR / "eval_exp03n.sbatch"
    if not _SLURM_DIR.exists():
        pytest.skip("cylindrical slurm_neuronic checkout not present")
    env = dict(os.environ)
    env.update(DRY_RUN="1", STEP="40000", STREAM="EMA", EXPECT_PACKAGE_SHA="x",
               EXPECT_EXP09_SHA="y", SLURM_ARRAY_TASK_ID="0")
    proc = subprocess.run(["bash", str(path)], capture_output=True, text=True, env=env,
                          cwd=str(tmp_path))
    assert proc.returncode != 0, "an unknown STREAM must be refused (fail-closed)"


def test_eval_sbatch_array_and_launch_shape_directives():
    """Launch shape (plan §5.1 / review fix F1): 1 node, 1 task, 1 L40, and the 10-cell array."""
    text = _sbatch("eval_exp03n.sbatch")
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks=1" in text
    assert "#SBATCH --gres=gpu:l40:1" in text
    assert "--array=0-9" in text, "K{1,8} x seeds 42-46 = 10 cells"
    assert "exp03n_${STEP}_${STREAM}_K${K}_s${SEED}" in text, "stream-tagged, collision-free eval name"
    assert "assert_arm_configs_exp03n.py" in text, "each cell must run the pin gate"


def test_eval_sbatch_asserts_its_launch_shape_and_evaluation_only_status():
    """The §5.1 evidence must be MECHANICAL, not decorative: worker-reported SHAs compared to
    the pins, node/task/GPU counts asserted fail-closed, an eval-only source pre-flight, and
    post-run full-split + no-training-marker assertions on the eval's own stdout."""
    text = _sbatch("eval_exp03n.sbatch")
    assert "launch-shape: 1 node / 1x L40 / 1 task / evaluation-only" in text
    assert 'WT_SHA=$(git -C "$WORKTREE" rev-parse HEAD)' in text
    assert 'PKG_SHA=$(git -C "$CYL_REPO" rev-parse HEAD)' in text
    assert '[ "$WT_SHA" = "$EXPECT_EXP09_SHA" ]' in text
    assert '[ "$PKG_SHA" = "$EXPECT_PACKAGE_SHA" ]' in text
    assert '[ "$NNODES" = "1" ]' in text and '[ "$NTASKS" = "1" ]' in text
    assert '[ "$NGPU" = "1" ]' in text
    assert 'NTASKS="${SLURM_NTASKS:-1}"' in text, "task count must default fail-closed to 1 and be asserted"
    # evaluation-only evidence
    assert "module.eval().requires_grad_(False)" in text
    assert "with torch.no_grad():" in text
    assert "EVAL-ONLY CONTRACT VIOLATED" in text
    assert 'grep -q "Found 6337 files"' in text, "the full-split criterion must be asserted post-run"
    assert 'for MARK in "optim" "backward" "training_step" "Epoch "' in text, (
        "the eval stdout must be asserted free of training markers"
    )


def test_eval_sbatch_proves_the_weight_stream_it_used():
    text = _sbatch("eval_exp03n.sbatch")
    assert "diffusion_ema.ema_model." in text, "EMA cells must PRE-assert the ckpt carries EMA keys"
    assert "Using EMA model" in text, "the log line must be asserted post-run (present/absent)"


def test_eval_sbatch_retention_and_overwrite_guard():
    text = _sbatch("eval_exp03n.sbatch")
    assert "sha256sum" in text
    assert _RECORDS_DIR_ABS in text, "artifacts must be retained in the exp_03 records folder"
    assert f"WORKTREE={_WORKTREE_ROOT_ABS}" in text
    assert 'IMPORT_DIR="${WORKTREE}/outputs_FLAC/exp03n_maxpoolmlp_import"' in text, (
        "EMA raws must ALSO land in the repo-rooted import dir the canonical generator globs"
    )
    assert re.search(r'if \[ "\$STREAM" = "ema" \];[\s\S]*IMPORT_DIR', text), (
        "the import-dir copy must be EMA-only"
    )
    assert re.search(r"\[ ! -e .*ART", text), "an existing artifact must abort the cell"
