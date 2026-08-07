"""exp_04 (worklog_yixun_neuronic) — tests for the exp04n TOOLING: the pre-launch
arm-wiring gate (including the cross-arm identity oracle), the trained-checkpoint
equivariance guard, and the launch/probe/eval script contracts.

Companion to ``test_exp04n_cond_pool.py`` (which pins the model delta itself). Everything
here is CPU-only and offline; the gate/guard helpers are exercised on TINY random-weight
cylindrical backbones, never on the Hub.

The run scripts and Slurm wrappers are asserted to be their exp03n counterparts under ONE
declared arm-token substitution (``ARM_SUBSTITUTIONS``) — the strongest available statement
of delta §4 "run identity": the recipe, the gates and every scientific flag are the exp_03
bytes, and only the arm identifiers differ.
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import copy  # noqa: E402
import difflib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_EXP09_DIR = _REPO_ROOT / "worklog" / "worklog_yixun" / "exp_09_cyl_no_ssl"
_GATE_PATH = _EXP09_DIR / "assert_arm_configs_exp04n.py"
_GUARD_PATH = _EXP09_DIR / "guard_exp04n_equivariance.py"

from src.tests.test_exp03n_cond_pool import (  # noqa: E402
    _HIDDEN,
    _build,
    _cyl_conditioning,
    _geoms,
    _save_tiny_cyl,
)
from src.tests.test_exp04n_cond_pool import _full_model_config  # noqa: E402


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
    return _load_module(_GATE_PATH, "assert_arm_configs_exp04n")


@pytest.fixture(scope="module")
def guard():
    return _load_module(_GUARD_PATH, "guard_exp04n_equivariance")


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    return _save_tiny_cyl(tmp_path_factory.mktemp("exp04n_tooling_tiny"))


# ------------------------------------------------------------------------------------ #
# 1. gate: per-variant config contract
# ------------------------------------------------------------------------------------ #
def test_gate_accepts_both_real_config_variants(gate):
    for name, variant in (("FLAC_AR_exp04n.json", "base"),
                          ("FLAC_AR_exp04n_online_eval.json", "online")):
        cfg, resolved = gate.assert_config_contract(str(_EXP09_DIR / name))
        assert resolved == variant, (name, resolved)
        assert cfg["model"]["conditioning"]["configs"][1]["config"]["ViT"]["cond_pool"] == "mean_mlp"


def test_gate_variant_autodetect_is_filename_driven(gate):
    assert gate.variant_for_config("/x/FLAC_AR_exp04n_online_eval.json") == "online"
    assert gate.variant_for_config("/x/FLAC_AR_exp04n.json") == "base"


def test_gate_rejects_the_wrong_reference_for_a_variant(gate):
    with pytest.raises(RuntimeError, match="contract|reconstruct"):
        gate.assert_config_contract(str(_EXP09_DIR / "FLAC_AR_exp04n_online_eval.json"),
                                    variant="base")
    with pytest.raises(RuntimeError, match="contract|reconstruct"):
        gate.assert_config_contract(str(_EXP09_DIR / "FLAC_AR_exp04n.json"), variant="online")


def test_gate_rejects_the_exp03n_configs(gate):
    """Arm confusion is the failure this gate exists to prevent: the exp_03 configs are
    identical except the knob VALUE, so a swapped --config must be refused."""
    for name in ("FLAC_AR_exp03n.json", "FLAC_AR_exp03n_online_eval.json"):
        with pytest.raises(RuntimeError, match="cond_pool|contract"):
            gate.assert_config_contract(str(_EXP09_DIR / name))


@pytest.mark.parametrize("mutation", ["extra_key", "changed_recipe", "dropped_key",
                                      "wrong_width", "one_block_only"])
def test_gate_rejects_mutated_configs(gate, tmp_path, mutation):
    cfg = json.load(open(_EXP09_DIR / "FLAC_AR_exp04n.json"))
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
    path = tmp_path / "FLAC_AR_exp04n.json"
    path.write_text(json.dumps(cfg, indent=4) + "\n")
    with pytest.raises(RuntimeError):
        gate.assert_config_contract(str(path))


# ------------------------------------------------------------------------------------ #
# 2. gate: instantiated-model arm wiring
# ------------------------------------------------------------------------------------ #
def test_gate_head_wiring_accepts_a_correct_mean_mlp_build(gate, tiny_dir):
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="mean_mlp"))
    head = gate.assert_head_wiring(mc, hidden_size=_HIDDEN, cond_dim=256)
    assert head is _geoms(mc)[0].lin_proj


def test_gate_head_wiring_rejects_the_legacy_build(gate, tiny_dir):
    """The legacy arm ALSO mean-pools, so only the head shape separates it from exp_04."""
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True))
    with pytest.raises(RuntimeError, match="head"):
        gate.assert_head_wiring(mc, hidden_size=_HIDDEN, cond_dim=256)


def test_gate_head_wiring_rejects_the_exp03_max_pool_build(gate, tiny_dir):
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="max_mlp"))
    with pytest.raises(RuntimeError, match="dino_pool"):
        gate.assert_head_wiring(mc, hidden_size=_HIDDEN, cond_dim=256)


def test_gate_head_wiring_rejects_an_unshared_head(gate, tiny_dir):
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="mean_mlp"))
    geoms = _geoms(mc)
    geoms[1].lin_proj = copy.deepcopy(geoms[0].lin_proj)
    with pytest.raises(RuntimeError, match="shared"):
        gate.assert_head_wiring(mc, hidden_size=_HIDDEN, cond_dim=256)


def test_gate_param_delta_is_the_hidden_layer_only(gate, tiny_dir):
    legacy = _build(_cyl_conditioning(tiny_dir, with_context=True))
    new = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="mean_mlp"))
    gate.assert_param_delta(legacy, new, expected=_HIDDEN * _HIDDEN + _HIDDEN)
    with pytest.raises(RuntimeError, match="trainable"):
        gate.assert_param_delta(legacy, new, expected=147840)


def test_gate_live_forward_matches_the_external_oracle(gate, tiny_dir):
    gate.assert_live_forward(_build(_cyl_conditioning(tiny_dir, with_context=True,
                                                      cond_pool="mean_mlp")),
                             height=64, width=128)


def test_gate_live_forward_catches_a_max_pooling_regression(gate, tiny_dir):
    """Sabotage: flip the served pooling to max while the head stays the MLP. Shapes and
    parameter counts are unchanged, so ONLY the live forward can catch it."""
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="mean_mlp"))
    for geom in _geoms(mc):
        geom.dino_pool = "max"
    with pytest.raises(RuntimeError, match="pooler_output|forward"):
        gate.assert_live_forward(mc, height=64, width=128)


def test_gate_common_init_identity_on_tiny_full_models(gate, tiny_dir):
    gate.assert_common_init_identity(_full_model_config(tiny_dir, cond_pool="mean_mlp"),
                                     _full_model_config(tiny_dir, cond_pool=None), seed=42)


def test_gate_common_init_identity_catches_a_leaked_rng_stream(gate, tiny_dir, monkeypatch):
    """Sabotage: draw the hidden layer from the GLOBAL stream (no fork_rng). Every module
    built AFTER the conditioner then initialises differently."""
    import contextlib

    import src.models.conditioners as cond_mod

    monkeypatch.setattr(cond_mod.torch.random, "fork_rng",
                        lambda *a, **kw: contextlib.nullcontext())
    with pytest.raises(RuntimeError):
        gate.assert_common_init_identity(_full_model_config(tiny_dir, cond_pool="mean_mlp"),
                                         _full_model_config(tiny_dir, cond_pool=None), seed=42)


# ------------------------------------------------------------------------------------ #
# 3. gate: CROSS-ARM IDENTITY ORACLE (delta 1b)
# ------------------------------------------------------------------------------------ #
def test_gate_cross_arm_identity_holds_for_the_real_arms(gate, tiny_dir):
    """mean_mlp vs max_mlp at one seed: FULL state dicts and the post-construction global
    RNG state bitwise equal — the pooling selector is the sole difference."""
    gate.assert_cross_arm_identity(_full_model_config(tiny_dir, cond_pool="mean_mlp"), seed=42)


def test_gate_cross_arm_identity_refuses_a_config_that_is_not_the_mean_arm(gate, tiny_dir):
    with pytest.raises(RuntimeError, match="cross-arm"):
        gate.assert_cross_arm_identity(_full_model_config(tiny_dir, cond_pool="max_mlp"), seed=42)


def test_gate_cross_arm_identity_catches_a_divergent_sibling_build(gate, tiny_dir, monkeypatch):
    """Sabotage: make the exp_03 (sibling) build's head differ by one weight — i.e. simulate
    a per-arm COPY of the head construction that drifted. The oracle must refuse; without it
    the two experiments would silently differ by more than the pooling."""
    real = gate.create_multi_conditioner_from_conditioning_config

    def _perturbing(cond_cfg, *args, **kwargs):
        mc = real(cond_cfg, *args, **kwargs)
        blocks = [c["config"]["ViT"] for c in cond_cfg["configs"]
                  if c["type"] == "ViTCoordinates"]
        if blocks and blocks[0].get("cond_pool") == gate.SIBLING_COND_POOL:
            with torch.no_grad():
                gate._geoms(mc)[0].lin_proj[0].weight.add_(1.0)
        return mc

    monkeypatch.setattr(gate, "create_multi_conditioner_from_conditioning_config", _perturbing)
    with pytest.raises(RuntimeError, match="cross-arm"):
        gate.assert_cross_arm_identity(_full_model_config(tiny_dir, cond_pool="mean_mlp"), seed=42)


# ------------------------------------------------------------------------------------ #
# 4. gate: the worktree pin (EXPECT_EXP04_SHA, with the inherited alias)
# ------------------------------------------------------------------------------------ #
class _Stop(Exception):
    """Sentinel: stop ``main`` right after the pin is resolved (no heavy work follows)."""


@pytest.fixture
def pin_probe(gate, monkeypatch):
    """Run ``gate.main`` far enough to capture the worktree pin it resolved."""
    seen = {}

    def _record(repo, sha, strict=True):
        seen["sha"], seen["strict"] = sha, strict
        raise _Stop()

    monkeypatch.setattr(gate.exp09_gate, "assert_exp09_provenance", _record)
    for name in ("EXPECT_EXP04_SHA", "EXPECT_EXP09_SHA"):
        monkeypatch.delenv(name, raising=False)

    def run(argv=()):
        with pytest.raises(_Stop):
            gate.main([*argv])
        return seen

    return run


def test_gate_reads_the_exp04_pin_from_the_env(pin_probe, monkeypatch):
    monkeypatch.setenv("EXPECT_EXP04_SHA", "aaaa1111")
    assert pin_probe()["sha"] == "aaaa1111"


def test_gate_still_accepts_the_inherited_exp09_pin_env(pin_probe, monkeypatch):
    monkeypatch.setenv("EXPECT_EXP09_SHA", "bbbb2222")
    assert pin_probe()["sha"] == "bbbb2222"


def test_gate_cli_flags_win_over_the_env(pin_probe, monkeypatch):
    monkeypatch.setenv("EXPECT_EXP04_SHA", "aaaa1111")
    assert pin_probe(["--expect-exp04-sha", "cccc3333"])["sha"] == "cccc3333"
    assert pin_probe(["--expect-exp09-sha", "dddd4444"])["sha"] == "dddd4444"


def test_gate_refuses_disagreeing_pin_spellings(gate, pin_probe):
    with pytest.raises(RuntimeError, match="disagree"):
        gate.main(["--expect-exp04-sha", "aaaa1111", "--expect-exp09-sha", "bbbb2222"])


def test_gate_passes_an_absent_pin_through_strict(pin_probe):
    seen = pin_probe()
    assert seen["sha"] is None and seen["strict"] is True, seen


# ------------------------------------------------------------------------------------ #
# 5. trained-checkpoint equivariance guard (delta §3)
# ------------------------------------------------------------------------------------ #
_TINY_GEOM = dict(height=64, width=128)   # 4x8 tokens at patch 16 -> W_t = 8


def test_guard_passes_on_a_random_weight_mini_model(guard, tiny_dir):
    """The guard tests the ACTUAL served condition MLP(mean(tokens)) — architectural, so it
    must hold at random init, not only on trained weights."""
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="mean_mlp"))
    residuals = guard.check_equivariance(_geoms(mc)[0], **_TINY_GEOM)
    assert set(residuals) == {90.0, 180.0, 270.0}
    assert max(residuals.values()) <= guard.DEFAULT_BOUND, residuals


def test_guard_negative_control_gauge_off_raises(guard, tmp_path):
    gauge_off = _save_tiny_cyl(tmp_path / "tiny_gauge_off")
    mc = _build(_cyl_conditioning(gauge_off, with_context=True, cond_pool="mean_mlp",
                                  gauge="none"))
    with pytest.raises(RuntimeError, match="equivarian|residual"):
        guard.check_equivariance(_geoms(mc)[0], **_TINY_GEOM)


def test_guard_refuses_an_angle_that_is_not_a_whole_token_column(guard, tiny_dir):
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="mean_mlp"))
    with pytest.raises(RuntimeError, match="column"):
        guard.check_equivariance(_geoms(mc)[0], angles=(30.0,), **_TINY_GEOM)


def test_guard_refuses_the_max_pooled_arm(guard, tiny_dir):
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True, cond_pool="max_mlp"))
    with pytest.raises(RuntimeError, match="dino_pool|mean"):
        guard.check_equivariance(_geoms(mc)[0], **_TINY_GEOM)


def test_guard_refuses_the_legacy_bare_linear_arm(guard, tiny_dir):
    """Mean pooling ALONE does not identify exp_04 — the legacy arm reports dino_pool
    'mean' too. Certifying it would attest a condition the exp04n checkpoint never serves."""
    mc = _build(_cyl_conditioning(tiny_dir, with_context=True))
    with pytest.raises(RuntimeError, match="head|Linear"):
        guard.check_equivariance(_geoms(mc)[0], **_TINY_GEOM)


def _write_pl_checkpoint(tmp_path, tiny_dir):
    """A Lightning-shaped checkpoint of a TINY exp04n model (``diffusion.``-prefixed state
    dict), plus the model-config JSON that reproduces it."""
    from src.models import create_model_from_config

    cfg = _full_model_config(tiny_dir, cond_pool="mean_mlp")
    cfg_path = tmp_path / "FLAC_AR_exp04n.json"
    cfg_path.write_text(json.dumps(cfg, indent=4) + "\n")
    torch.manual_seed(3)
    model = create_model_from_config(copy.deepcopy(cfg))
    ckpt_path = tmp_path / "epoch=0-step=40.ckpt"
    torch.save({"state_dict": {f"diffusion.{k}": v for k, v in model.state_dict().items()}},
               ckpt_path)
    return str(ckpt_path), str(cfg_path), model


def test_guard_loads_a_lightning_checkpoint_and_serves_the_mean_pool_head(guard, tiny_dir, tmp_path):
    ckpt_path, cfg_path, model = _write_pl_checkpoint(tmp_path, tiny_dir)
    geoms = guard.load_geometry_conditioners(ckpt_path, cfg_path, device="cpu")
    assert len(geoms) == 2
    assert all(g.dino_pool == "mean" for g in geoms)
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
# 6. launch / probe / eval script contracts
# ------------------------------------------------------------------------------------ #
_LAUNCH = _EXP09_DIR / "exp04n_launch.sh"
_EXP03N_LAUNCH = _EXP09_DIR / "exp03n_launch.sh"
_EXP09_LAUNCH = _EXP09_DIR / "exp09_launch.sh"
_PROBE = _EXP09_DIR / "exp04n_probe.sh"
_EXP03N_PROBE = _EXP09_DIR / "exp03n_probe.sh"
# The Slurm wrappers live in the CYLINDRICAL repo (they are that repo's records); the
# contract is asserted here because this is where the trap lives. Skipped when that
# checkout is not beside this one.
_SLURM_DIR = Path("/n/fs/gatrdp/codespace/cylindrical-dinov3/slurm_neuronic")
_WORKTREE_ROOT_ABS = "/n/fs/gatrdp/codespace/exp04-meanpool-mlp-cond"
_CYL_SRC_ABS = "/n/fs/gatrdp/codespace/cylindrical-dinov3/src"
_RECORDS_DIR_ABS = ("/n/fs/gatrdp/codespace/cylindrical-dinov3/worklog/worklog_yixun_neuronic/"
                    "exp_04_meanpool_mlp_cond_claude")

# The ONLY differences allowed between an exp03n script and its exp04n counterpart: arm
# identifiers. ``EXPECT_EXP09_SHA``/``--expect-exp09-sha`` become the exp_04 spelling of the
# worktree pin (delta 0); the gate still accepts the inherited spelling.
ARM_SUBSTITUTIONS = (
    ("exp_03", "exp_04"),
    ("exp03", "exp04"),
    ("EXP03", "EXP04"),
    ("maxpool", "meanpool"),
    ("max-pool", "mean-pool"),
    ("max_mlp", "mean_mlp"),
    ("MAX-POOL", "MEAN-POOL"),
    ("EXP09_SHA", "EXP04_SHA"),
    ("expect-exp09-sha", "expect-exp04-sha"),
)


def _rename_arm(text: str) -> str:
    for old, new in ARM_SUBSTITUTIONS:
        text = text.replace(old, new)
    return text


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


@pytest.mark.parametrize("script", ["exp04n_launch.sh", "exp04n_probe.sh"])
def test_shell_scripts_parse(script):
    path = _EXP09_DIR / script
    _text(path)
    rc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr


@pytest.mark.parametrize("exp04n,exp03n", [("exp04n_launch.sh", "exp03n_launch.sh"),
                                           ("exp04n_probe.sh", "exp03n_probe.sh")])
def test_run_scripts_are_the_exp03n_scripts_up_to_the_arm_identifiers(exp04n, exp03n):
    """Delta §4 run identity, mechanically: every byte outside ARM_SUBSTITUTIONS is exp_03's
    — same recipe, same rung pin, same gates, same 67,500 steps. Any real edit (a changed
    flag, a dropped refusal) shows up here as a diff."""
    got = _text(_EXP09_DIR / exp04n)
    want = _rename_arm(_text(_EXP09_DIR / exp03n))
    assert got == want, "\n".join(difflib.unified_diff(
        want.splitlines(), got.splitlines(), "renamed exp03n", exp04n, lineterm=""))[:4000]


@pytest.mark.parametrize("exp04n,exp03n", [("train_exp04n.sbatch", "train_exp03n.sbatch"),
                                           ("probe_exp04n.sbatch", "probe_exp03n.sbatch"),
                                           ("eval_exp04n.sbatch", "eval_exp03n.sbatch")])
def test_slurm_wrappers_are_the_exp03n_wrappers_up_to_the_arm_identifiers(exp04n, exp03n):
    got = _sbatch(exp04n)
    want = _rename_arm(_sbatch(exp03n))
    assert got == want, "\n".join(difflib.unified_diff(
        want.splitlines(), got.splitlines(), "renamed exp03n", exp04n, lineterm=""))[:4000]


def test_launcher_every_diff_from_exp09_is_marked_and_numbered():
    """House style (exp_02/exp_03): the launcher is a COPY of the reviewed exp-09 launcher in
    which every changed/added line carries a ``# EXP04 DIFF n/N`` marker, so a reviewer can
    diff the two files and account for each line."""
    exp09 = _text(_EXP09_LAUNCH).splitlines()
    exp04n = _text(_LAUNCH).splitlines()
    added = [line for line in difflib.unified_diff(exp09, exp04n, n=0, lineterm="")
             if line.startswith("+") and not line.startswith("+++")]
    assert added, "the launcher is byte-identical to exp09_launch.sh — nothing was adapted"
    markers = []
    for line in added:
        m = re.search(r"EXP04 DIFF (\d+)/(\d+)", line)
        assert m, f"unmarked diff line in exp04n_launch.sh: {line[1:][:120]}"
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
    launch, exp03n, exp09 = _text(_LAUNCH), _text(_EXP03N_LAUNCH), _text(_EXP09_LAUNCH)
    for flag_line in _SCIENTIFIC_FLAGS:
        assert flag_line in exp09, f"fixture drift: {flag_line!r} not in exp09_launch.sh"
        assert flag_line in exp03n, f"fixture drift: {flag_line!r} not in exp03n_launch.sh"
        assert flag_line in launch, f"scientific flag line changed: {flag_line!r}"
    # the BN-compliant rung pin survives
    assert '[ "$MB" = "32" ] && [ "$ACC" = "1" ]' in launch


def test_launcher_binds_the_exp04n_arm_not_exp03n_or_exp09():
    launch = _text(_LAUNCH)
    assert f"cd {_WORKTREE_ROOT_ABS}" in launch, "the launcher must cd to the absolute worktree root"
    assert f"export PYTHONPATH={_CYL_SRC_ABS}" in launch
    assert "FLAC_AR_exp04n.json" in launch
    assert "FLAC_AR_exp03n.json" not in launch and "FLAC_AR_exp09.json" not in launch
    assert "assert_arm_configs_exp04n.py" in launch
    assert "assert_arm_configs_exp03n.py" not in launch
    assert "assert_arm_configs_exp09.py" not in launch
    assert "--name FLAC_exp04n_meanpoolmlp --experiment-name exp04n_meanpoolmlp" in launch
    assert "--save-dir /n/fs/gatrdp/outputs/exp04n_meanpoolmlp" in launch
    assert _RECORDS_DIR_ABS in launch, "the teed log must land in the exp_04 records folder"
    assert "meanpool_mlp_cond_${TS}_j${SLURM_JOB_ID:-nojob}_train.log" in launch


def test_launcher_refuses_without_the_frozen_vram_file_and_the_two_pins():
    launch = _text(_LAUNCH)
    assert "exp04n_frozen_min_free.txt" in launch
    assert "exp03n_frozen_min_free.txt" not in launch, (
        "the exp_03 frozen VRAM value is another arm's measurement"
    )
    assert "c1_frozen_min_free.txt" not in launch
    assert 'FROZEN_FILE" ] || {' in launch, "the frozen-file REFUSE must survive"
    assert re.search(r'\[ -n "\$EXPECT_PACKAGE_SHA" \].*\[ -n "\$EXPECT_EXP04_SHA" \]', launch), (
        "the launcher must REFUSE when either pin is absent"
    )


def test_the_frozen_vram_file_is_not_committed_yet():
    """The launcher REFUSES while this file is absent; it is written by hand, after review,
    from THIS arm's probe (never inherited from exp_03's L40 measurement)."""
    assert not (_EXP09_DIR / "exp04n_frozen_min_free.txt").exists(), (
        "exp04n_frozen_min_free.txt exists before this arm's probe has produced it"
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
    assert re.search(r'SAVE="\$\{SAVE:-/n/fs/gatrdp/outputs/exp04n_meanpoolmlp_PROBE\}"', probe), (
        "the probe must default to its own _PROBE save dir"
    )
    assert re.search(r'\[ ! -e "\$SAVE" \]', probe), "the probe must refuse an existing save dir"
    # same gates as the launcher EXCEPT the frozen-VRAM file (this run PRODUCES it)
    assert "assert_arm_configs_exp04n.py" in probe
    assert "FLAC_AR_exp04n.json" in probe
    assert f"export PYTHONPATH={_CYL_SRC_ABS}" in probe
    assert "EXPECT_PACKAGE_SHA" in probe and "EXPECT_EXP04_SHA" in probe
    assert "exp04n_frozen_min_free.txt" in probe, "the probe must name the file it feeds"
    assert "21900" in probe, "the provisional free-VRAM floor must be pinned"
    assert "1.15" in probe, "the probe must print the recommended 1.15x frozen value"
    assert ".bashrc" not in probe
    assert _RECORDS_DIR_ABS in probe, "probe logs/samples belong to the exp_04 records folder"
    # the probe's scientific shape matches the arm
    assert "--batch-size 32" in probe and "--accum-batches 1" in probe
    assert "--sync-batchnorm true" in probe and "ddp_find_unused_parameters_true" in probe
    assert "--seed 42" in probe and "--num-gpus 2" in probe


# ---- Slurm wrappers (cylindrical repo) ---------------------------------------------- #
@pytest.mark.parametrize("name,gpus", [("train_exp04n.sbatch", "2"), ("probe_exp04n.sbatch", "2")])
def test_gpu_wrappers_pin_and_assert_their_launch_shape(name, gpus):
    text = _sbatch(name)
    assert "#SBATCH --nodes=1" in text and "#SBATCH --ntasks=1" in text
    assert f"launch-shape: 1 node / {gpus}x L40 / 1 task" in text
    assert '[ "$NNODES" = "1" ]' in text and '[ "$NTASKS" = "1" ]' in text
    assert f'[ "$NGPU" = "{gpus}" ]' in text
    assert "EXPECT_EXP04_SHA" in text and "EXPECT_PACKAGE_SHA" in text


def test_train_sbatch_shape_and_preflights():
    text = _sbatch("train_exp04n.sbatch")
    assert "--gres=gpu:l40:2" in text
    assert "--cpus-per-task=16" in text and "--mem=32G" in text
    assert "--time=5-00:00:00" in text
    assert "exp04n_launch.sh" in text
    assert "100" in text, "the storage floor must be present"
    assert "/n/fs/gatrdp/outputs/exp04n_meanpoolmlp" in text
    assert re.search(r"\[ ! -e /n/fs/gatrdp/outputs/exp04n_meanpoolmlp \]", text), (
        "a fresh launch must refuse an existing save dir"
    )
    assert "HF_HUB_OFFLINE=1" in text
    assert _WORKTREE_ROOT_ABS in text


def test_probe_sbatch_shape():
    text = _sbatch("probe_exp04n.sbatch")
    assert "--gres=gpu:l40:2" in text
    assert "--time=00:50:00" in text
    assert "exp04n_probe.sh" in text


def test_eval_sbatch_passes_the_mandatory_fa_invariant_flags():
    """THE trap (handoff): eval_FLAC.py defaults to --cond-method vanilla and ignores the
    model config, so an eval that omits these flags silently scores the raw-pose path."""
    text = _sbatch("eval_exp04n.sbatch")
    assert "--cond-method fa_invariant" in text
    assert "--frame-avg-angles 0" in text
    assert "--cond-method vanilla" not in text


# --- per-stream DRY RUN: the wrapper renders the command it will run, and the test asserts the
# EXACT rendered argument vector. A word search over the script would pass with the two
# stream->config assignments SWAPPED; these assertions bind stream -> config. ----------------- #
_STREAM_CONFIG = {
    "online": "worklog/worklog_yixun/exp_09_cyl_no_ssl/FLAC_AR_exp04n_online_eval.json",
    "ema": "worklog/worklog_yixun/exp_09_cyl_no_ssl/FLAC_AR_exp04n.json",
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
    """Execute eval_exp04n.sbatch's DRY_RUN path and return its stdout. Touches no GPU, no
    checkpoint and no file (asserted by the caller via an empty cwd)."""
    path = _SLURM_DIR / "eval_exp04n.sbatch"
    if not _SLURM_DIR.exists():
        pytest.skip(f"cylindrical slurm_neuronic checkout not present at {_SLURM_DIR}")
    if not path.exists():
        pytest.fail(f"required deliverable is missing: {path}")
    env = dict(os.environ)
    env.update(DRY_RUN="1", STEP=step, STREAM=stream,
               EXPECT_PACKAGE_SHA="deadbeefdeadbeef", EXPECT_EXP04_SHA="cafebabecafebabe",
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
    assert _flag(tokens, "--eval-name") == f"exp04n_40000_{stream}_K{k}_s{seed}"
    assert _flag(tokens, "--ckpt-path").endswith("step=40000.ckpt")
    assert "exp04n_meanpoolmlp" in _flag(tokens, "--ckpt-path"), (
        "the eval must resolve THIS arm's checkpoints"
    )
    # the dry run is side-effect free
    assert list(tmp_path.iterdir()) == [], f"the dry run created files: {list(tmp_path.iterdir())}"


def test_eval_dry_run_streams_bind_different_configs(tmp_path):
    """Explicit swap detector: the two streams must render DIFFERENT configs, each the one
    registered for it — so exchanging the two case-branch assignments fails here too."""
    rendered = {s: _flag(_rendered(_dry_run_eval(s, cwd=tmp_path), "EVAL_CMD:"), "--model-config")
                for s in ("online", "ema")}
    assert rendered["online"] != rendered["ema"], rendered
    assert rendered == _STREAM_CONFIG, rendered
    assert "online_eval" in rendered["online"], "the online stream must load the use_ema:false config"
    assert not rendered["ema"].endswith("online_eval.json"), (
        "the EMA stream must load the training config (use_ema:true), not the online variant"
    )


def test_eval_dry_run_refuses_an_unknown_stream(tmp_path):
    path = _SLURM_DIR / "eval_exp04n.sbatch"
    if not _SLURM_DIR.exists():
        pytest.skip("cylindrical slurm_neuronic checkout not present")
    env = dict(os.environ)
    env.update(DRY_RUN="1", STEP="40000", STREAM="EMA", EXPECT_PACKAGE_SHA="x",
               EXPECT_EXP04_SHA="y", SLURM_ARRAY_TASK_ID="0")
    proc = subprocess.run(["bash", str(path)], capture_output=True, text=True, env=env,
                          cwd=str(tmp_path))
    assert proc.returncode != 0, "an unknown STREAM must be refused (fail-closed)"


def test_eval_dry_run_refuses_without_the_worktree_pin(tmp_path):
    """EXPECT_EXP04_SHA is REQUIRED at submission: an unpinned eval must never render."""
    path = _SLURM_DIR / "eval_exp04n.sbatch"
    if not _SLURM_DIR.exists():
        pytest.skip("cylindrical slurm_neuronic checkout not present")
    env = {k: v for k, v in os.environ.items()
           if k not in ("EXPECT_EXP04_SHA", "EXPECT_EXP09_SHA")}
    env.update(DRY_RUN="1", STEP="40000", STREAM="online", EXPECT_PACKAGE_SHA="x",
               SLURM_ARRAY_TASK_ID="0")
    proc = subprocess.run(["bash", str(path)], capture_output=True, text=True, env=env,
                          cwd=str(tmp_path))
    assert proc.returncode != 0, "a missing EXPECT_EXP04_SHA must be refused (fail-closed)"


def test_eval_sbatch_array_and_launch_shape_directives():
    text = _sbatch("eval_exp04n.sbatch")
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks=1" in text
    assert "#SBATCH --gres=gpu:l40:1" in text
    assert "--array=0-9" in text, "K{1,8} x seeds 42-46 = 10 cells"
    assert "exp04n_${STEP}_${STREAM}_K${K}_s${SEED}" in text, "stream-tagged, collision-free eval name"
    assert "assert_arm_configs_exp04n.py" in text, "each cell must run the pin gate"


def test_eval_sbatch_asserts_its_launch_shape_and_evaluation_only_status():
    text = _sbatch("eval_exp04n.sbatch")
    assert "launch-shape: 1 node / 1x L40 / 1 task / evaluation-only" in text
    assert 'WT_SHA=$(git -C "$WORKTREE" rev-parse HEAD)' in text
    assert 'PKG_SHA=$(git -C "$CYL_REPO" rev-parse HEAD)' in text
    assert '[ "$WT_SHA" = "$EXPECT_EXP04_SHA" ]' in text
    assert '[ "$PKG_SHA" = "$EXPECT_PACKAGE_SHA" ]' in text
    assert '[ "$NNODES" = "1" ]' in text and '[ "$NTASKS" = "1" ]' in text
    assert '[ "$NGPU" = "1" ]' in text
    assert 'NTASKS="${SLURM_NTASKS:-1}"' in text
    assert "module.eval().requires_grad_(False)" in text
    assert "with torch.no_grad():" in text
    assert "EVAL-ONLY CONTRACT VIOLATED" in text
    assert 'grep -q "Found 6337 files"' in text, "the full-split criterion must be asserted post-run"
    assert 'for MARK in "optim" "backward" "training_step" "Epoch "' in text


def test_eval_sbatch_proves_the_weight_stream_it_used():
    text = _sbatch("eval_exp04n.sbatch")
    assert "diffusion_ema.ema_model." in text, "EMA cells must PRE-assert the ckpt carries EMA keys"
    assert "Using EMA model" in text, "the log line must be asserted post-run (present/absent)"


def test_eval_sbatch_retention_and_overwrite_guard():
    text = _sbatch("eval_exp04n.sbatch")
    assert "sha256sum" in text
    assert _RECORDS_DIR_ABS in text, "artifacts must be retained in the exp_04 records folder"
    assert f"WORKTREE={_WORKTREE_ROOT_ABS}" in text
    assert 'IMPORT_DIR="${WORKTREE}/outputs_FLAC/exp04n_meanpoolmlp_import"' in text, (
        "EMA raws must ALSO land in the repo-rooted import dir the canonical generator globs"
    )
    assert re.search(r'if \[ "\$STREAM" = "ema" \];[\s\S]*IMPORT_DIR', text), (
        "the import-dir copy must be EMA-only"
    )
    assert re.search(r"\[ ! -e .*ART", text), "an existing artifact must abort the cell"
