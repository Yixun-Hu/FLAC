"""exp_16 (are_port) — the PINNED launch schedule and the post-run readback verdict.

Seat: Opus 5 Coder (SOP §Roles). Added in the r1 code-review round to close
findings 1 and 3.

WHAT WENT WRONG IN ROUND 1. `40000` and `2500` were shell DEFAULTS
(`DEFAULT_MAXSTEPS=40000`), so `MODE=FULL MAXSTEPS=1000 CHECKPOINT_EVERY=1`
passed every gate and would have produced a 1,000-step arm carrying production
identity, production W&B name and production provenance. The post-run readback
accepted any `global_step` in `(0, MAXSTEPS]`, so it would have blessed it. The
only guard that rejected a short FULL run rejected it for the wrong reason — its
chosen cadence happened to write no checkpoint.

Both defects were unreachable from a test because they lived inside shell
strings. The endpoint, the cadence and the readback verdict now live in
``worklog/worklog_yixun/exp_16_are_port_claude/readback.py``; ``are_launch.sh``
imports them, and this file exercises them.
"""
import copy
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_EXPDIR = _REPO / "worklog/worklog_yixun/exp_16_are_port_claude"
ARE_CONFIG = _EXPDIR / "FLAC_AR_ARE.json"
LAUNCHER = _EXPDIR / "are_launch.sh"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"exp16_{name}", _EXPDIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rb():
    return _load("readback")


@pytest.fixture(scope="module")
def stamp():
    return _load("stamp_evidence")


@pytest.fixture(scope="module")
def arm_config():
    return json.loads(ARE_CONFIG.read_text())


# --------------------------------------------------------------------------- #
# 1. the schedule is PINNED for FULL/RESTART and FREE for PROBE
# --------------------------------------------------------------------------- #
def test_the_endpoint_and_cadence_are_the_preregistered_ones(rb):
    assert rb.ENDPOINT_STEPS == 40000
    assert rb.PRODUCTION_CHECKPOINT_EVERY == 2500


@pytest.mark.parametrize("mode", ["FULL", "RESTART"])
def test_production_modes_are_pinned(rb, mode):
    assert rb.pinned_schedule(mode) == (40000, 2500)
    assert rb.schedule_problems(mode, 40000, 2500) == []


def test_probe_is_free(rb):
    assert rb.pinned_schedule("PROBE") is None
    assert rb.schedule_problems("PROBE", 15, 5) == []
    assert rb.schedule_problems("PROBE", 250, 25) == []


@pytest.mark.parametrize("mode", ["FULL", "RESTART"])
@pytest.mark.parametrize("steps", [1, 1000, 39999, 40001, 67500, 80000])
def test_any_other_endpoint_is_rejected(rb, mode, steps):
    problems = rb.schedule_problems(mode, steps, 2500)
    assert problems and "pinned to the pre-registered endpoint" in problems[0]


@pytest.mark.parametrize("every", [1, 5, 1250, 2499, 2501, 5000])
def test_any_other_cadence_is_rejected(rb, every):
    problems = rb.schedule_problems("FULL", 40000, every)
    assert problems and any("CHECKPOINT_EVERY=2500" in p for p in problems)


def test_the_review_s_exact_counterexample_is_rejected(rb):
    """``MODE=FULL MAXSTEPS=1000 CHECKPOINT_EVERY=1`` — the invocation the r1
    review demonstrated passing. It must now fail on BOTH counts, and on the
    endpoint specifically rather than incidentally on the cadence."""
    problems = rb.schedule_problems("FULL", 1000, 1)
    assert len(problems) == 2
    assert "pre-registered endpoint MAXSTEPS=40000" in problems[0]
    assert "CHECKPOINT_EVERY=2500" in problems[1]


def test_unknown_modes_raise(rb):
    for bad in ("full", "FIT", "", None):
        with pytest.raises(ValueError):
            rb.pinned_schedule(bad)


# --------------------------------------------------------------------------- #
# 2. the post-run readback verdict
# --------------------------------------------------------------------------- #
def _embedded(arm_config, **training_over):
    cfg = copy.deepcopy(arm_config)
    cfg["training"].update(training_over)
    return cfg


def test_full_readback_requires_the_endpoint_exactly(rb, arm_config):
    assert rb.readback_problems("FULL", arm_config, 40000, arm_config, 1.0) == []
    for gs in (2500, 37500, 39999, 40001):
        problems = rb.readback_problems("FULL", arm_config, gs, arm_config, 1.0)
        assert problems and "pinned endpoint 40000" in problems[0], gs


def test_restart_readback_requires_the_endpoint_exactly(rb, arm_config):
    assert rb.readback_problems("RESTART", arm_config, 40000, arm_config, 1.0,
                                expected_step=20000, max_steps=40000) == []
    problems = rb.readback_problems("RESTART", arm_config, 30000, arm_config, 1.0,
                                    expected_step=20000, max_steps=40000)
    assert problems and "pinned endpoint 40000" in problems[0]


def test_probe_readback_only_checks_its_own_window(rb, arm_config):
    assert rb.readback_problems("PROBE", arm_config, 15, arm_config, 1.0,
                                expected_step=0, max_steps=15) == []
    problems = rb.readback_problems("PROBE", arm_config, 20, arm_config, 1.0,
                                    expected_step=0, max_steps=15)
    assert problems and "outside the run window" in problems[0]


def test_readback_rejects_a_checkpoint_with_no_embedded_config(rb, arm_config):
    problems = rb.readback_problems("FULL", None, 40000, arm_config, 1.0)
    assert problems == ["checkpoint carries no embedded 'model_config' dict"]


def test_readback_rejects_a_missing_or_mistyped_lambda(rb, arm_config):
    for bad in (0.5, 1, True, None):
        embedded = _embedded(arm_config, are_lambda=bad)
        problems = rb.readback_problems("FULL", embedded, 40000, arm_config, 1.0)
        assert any("did NOT reach the artifact" in p for p in problems), bad


def test_readback_rejects_an_empty_anchor_block(rb, arm_config):
    embedded = _embedded(arm_config, are_anchor={})
    problems = rb.readback_problems("FULL", embedded, 40000, arm_config, 1.0)
    assert any("calibrated constants did NOT reach the artifact" in p for p in problems)


def test_readback_is_type_strict_against_the_arm_json(rb, arm_config):
    embedded = copy.deepcopy(arm_config)
    embedded["training"]["are_anchor"]["a_g"] = 0.9
    problems = rb.readback_problems("FULL", embedded, 40000, arm_config, 1.0)
    assert any("are_anchor.a_g" in p for p in problems)


def test_the_type_strict_rule_distinguishes_int_from_float(rb):
    assert rb.type_strict_diff({"a": 1}, {"a": 1.0}) is not None
    assert rb.type_strict_diff({"a": 1.0}, {"a": 1.0}) is None
    assert rb.type_strict_diff({"a": True}, {"a": 1}) is not None
    # the labels only name the operands; the rule is the same object everywhere
    msg = rb.type_strict_diff({"x": 1}, {}, label_a="the checkpoint", label_b="the config")
    assert "present in the checkpoint, absent from the config" in msg


# --------------------------------------------------------------------------- #
# 3. the launcher really applies the module (no second copy of the rule)
# --------------------------------------------------------------------------- #
def test_the_launcher_imports_the_schedule_and_the_readback():
    src = LAUNCHER.read_text()
    assert "from readback import pinned_schedule, schedule_problems" in src
    assert "readback_problems" in src
    assert "from readback import type_strict_diff" in src
    # ...and does not re-declare the rule it imports
    assert 'def strict_diff(a, b, path="model_config"):' not in src


def test_the_launcher_has_no_production_dirty_bypass():
    """r1 review finding 3: ``ALLOW_DIRTY_TREATMENT=1`` let any caller relax the
    gate and then launch. It is gone; the only tolerance is a flag that also
    forces an abort before train.py."""
    src = LAUNCHER.read_text()
    assert 'ALLOW_DIRTY_TREATMENT="${ALLOW_DIRTY_TREATMENT' not in src
    assert '$ALLOW_DIRTY_TREATMENT' not in src
    assert 'ARE_GUARD_DRYRUN' in src
    # the dry-run stop must precede the training invocation in the file
    assert src.index('if [ "$ARE_GUARD_DRYRUN" = "1" ]; then') < src.index("python train.py")


def test_the_dry_run_flag_cannot_reach_training():
    """The flag's whole safety argument: everything after its stop is unreachable."""
    src = LAUNCHER.read_text()
    stop = src.index('ARE_GUARD_DRYRUN=1 -> all pre-launch gates executed')
    exit_line = src.index("  exit 2\nfi", stop)
    assert exit_line - stop < 400, "the dry-run stop must exit immediately"
    assert src.index("python train.py") > exit_line


def test_bash_syntax_is_clean():
    for script in ("are_launch.sh", "are_launch_guardtests.sh"):
        rc = subprocess.run(["bash", "-n", str(_EXPDIR / script)],
                            capture_output=True, text=True)
        assert rc.returncode == 0, f"{script}: {rc.stderr}"


# --------------------------------------------------------------------------- #
# 4. the treatment fingerprint covers every launch-defining input
# --------------------------------------------------------------------------- #
def test_fingerprint_covers_the_launch_defining_inputs(stamp):
    """r1 review finding 3: a dirty edit to the launcher, the dataset config or
    the train split changes the run without touching ``src/``, and round 1's
    fingerprint would not have noticed."""
    for rel in ("worklog/worklog_yixun/exp_16_are_port_claude/are_launch.sh",
                "worklog/worklog_yixun/exp_16_are_port_claude/readback.py",
                "src/configs/dataset_configs/AR/train/acousticroom_train.json",
                "data/AR/train.json",
                "src/configs/dataset_configs/custom_metadata/AR_md.py",
                "src/data/are_anchor.py", "src/data/dataset.py", "src/data/utils.py",
                "src/training/diffusion.py", "src/training/factory.py", "train.py"):
        assert rel in stamp.TREATMENT_PATHS, rel
    for rel in stamp.TREATMENT_PATHS:
        assert (_REPO / rel).is_file(), f"fingerprinted path missing: {rel}"


def test_fingerprint_changes_when_any_covered_file_changes(stamp, tmp_path):
    """Demonstrated, not asserted: the digest must actually move."""
    import shutil

    base = stamp.treatment_fingerprint(str(_REPO))
    work = tmp_path / "repo"
    work.mkdir()
    (work / ".git").mkdir()
    for rel in stamp.TREATMENT_PATHS:
        dst = work / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(_REPO / rel, dst)
    assert stamp.treatment_fingerprint(str(work)) == base

    target = work / "worklog/worklog_yixun/exp_16_are_port_claude/are_launch.sh"
    target.write_text(target.read_text() + "\n# a dirty edit\n")
    assert stamp.treatment_fingerprint(str(work)) != base


def test_the_evidence_record_carries_every_hash(stamp):
    rec = stamp.build_record("are_fit", "AREV", "PASS", "CLAUDE.md", "unit test",
                             str(_REPO))
    for field in ("treatment_sha256", "model_config_sha256", "calibration_sha256",
                  "vae_sha256", "source_sha"):
        assert isinstance(rec.get(field), str) and rec[field], field
    assert rec["are_lambda"] == 1.0
    assert stamp.REQUIRED["AREV"] == (("are_fit", "AREV"),)


# --------------------------------------------------------------------------- #
# 5. eval_FLAC must not become import-fragile because of this experiment
# --------------------------------------------------------------------------- #
def test_eval_flac_imports_without_the_exp16_worklog_folder(tmp_path):
    """``eval_FLAC`` is imported by consumers that carry no exp_16 folder at all.

    Two regressions this pins, both found while fixing r1 finding 2:

    * a module-scope ``sys.path.insert(<exp_16 dir>)`` put that directory ahead of
      everything for the whole process, so a later top-level ``import
      stamp_evidence`` resolved to exp_16's file instead of exp_14's;
    * a module-scope load of ``readback.py`` made ``import eval_FLAC`` FAIL in any
      tree without the exp_16 folder — including the model-comparison gate's
      synthetic repo, which copies ``eval_FLAC.py`` and the exp_11 validator and
      nothing else. Every validated row became "cannot derive the expected
      filename".

    Reproduced here with a synthetic tree of exactly those two files.
    """
    import shutil
    import subprocess
    import sys
    import textwrap

    fake = tmp_path / "maintree"
    (fake / "src").mkdir(parents=True)
    shutil.copy(_REPO / "eval_FLAC.py", fake / "eval_FLAC.py")
    assert not (fake / "worklog").exists()

    driver = tmp_path / "drive.py"
    driver.write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(_REPO)!r})     # for src.*
        sys.path.insert(0, {str(fake)!r})      # the partial tree wins for eval_FLAC
        before = list(sys.path)
        import eval_FLAC
        assert eval_FLAC.__file__.startswith({str(fake)!r}), eval_FLAC.__file__
        # the exp_16 directory must not have been pushed onto the path
        assert not any("exp_16_are_port_claude" in p for p in sys.path), sys.path[:4]
        # and the legacy artifact name must still be computable
        p = eval_FLAC.build_output_paths("d/ck.ckpt", 1, 1.0, "e")
        assert p["metrics"] == "d/ck_metrics_1_1.0_e.json", p
        print("OK")
    """))
    rc = subprocess.run([sys.executable, str(driver)], capture_output=True, text=True,
                        cwd=str(tmp_path))
    assert rc.returncode == 0, rc.stdout + rc.stderr
    assert "OK" in rc.stdout


def test_are_evaluation_says_so_when_the_module_is_missing(tmp_path, monkeypatch):
    """...and when an ARE run really does need it, the error names the reason."""
    import eval_FLAC

    monkeypatch.setattr(eval_FLAC, "_EXP16_READBACK", None)
    monkeypatch.setattr(eval_FLAC.os.path, "isfile", lambda p: False)
    with pytest.raises(RuntimeError, match="readback module"):
        eval_FLAC._load_exp16_readback()
