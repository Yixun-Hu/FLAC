"""exp_17 — post-run completion audit for a FULL 40k training run.

Why this exists. The FULL run we are relying on is the one launched from the
``exp-17-yawaug-a6000`` worktree at 2026-08-15 14:54:52. Its launcher binds an
impressive set of *pre*-run pins (commit, defaults.ini, the AR split, the VAE,
the ViT snapshot, even the seed-42 initial ``state_dict`` hash) and runs a
banner watchdog that fails if a training step appears before the treatment
banner — stronger than a post-hoc grep. What it does not do is verify, after
``train.py`` returns, that the run actually **finished**: its FULL path ends at
``exit "$RC"``.

That gap is not hypothetical. The installed Lightning catches
``KeyboardInterrupt`` without re-raising, so a graceful interruption exits 0;
and the treatment banner is printed from ``on_fit_start``, i.e. *before* step 0.
So "rc=0 and the banner is present" is satisfied by a run that trained for one
minute.

**Rev 2 — after the Codex review, which broke rev 1 with a single input.**
Rev 1 treated its evidence as an unordered bag of substrings, so one crafted log
could hold a stale banner, the string ``Not Starting with 2 processes``, a
*diagnostic quoting* the termination marker, ``train/loss=Infinity``, and
sixteen checkpoint names including ``step=42500`` — and audit clean. Rev 1 also
failed OPEN on fit health: its own happy-path fixture contained no loss line at
all, yet the tool printed "finite loss". The rules that follow are therefore:

* every marker is matched as a WHOLE LINE, never as a substring;
* the checkpoint set must be exactly the registered cadence, not merely 16 of
  something;
* loss is PARSED and required — absence of evidence is not evidence of health;
* the log is BOUND to the checkpoint directory it is used to bless;
* ``rc`` must be supplied — an unknown exit status is not a zero one.

Fixture strings are the LITERAL text Lightning and the launcher emit, copied
from the real logs. That matters: an earlier gate searched for
``max_steps=40000 reached`` and could never match, because Lightning writes
```max_steps=40000` reached`` with backticks around the assignment.

Written by the main session seat (Claude Opus 5, max effort).
"""
import pytest

from src.tools.exp17_full_audit import audit_full_run, BANNER, SAVE_DIR_FLAG


# The literal terminal text, as it appears in the logs on disk.
TERMINATION = "`Trainer.fit` stopped: `max_steps=40000` reached."
TOPOLOGY = "All distributed processes registered. Starting with 2 processes"
SAVE_DIR = "outputs_FLAC/exp17_YAWAUG"
ARGV = f"ARGV: python train.py --max-steps 40000 {SAVE_DIR_FLAG} {SAVE_DIR}"


def good_log(n_loss: int = 200) -> str:
    body = [
        "=== exp_17 FULL control stage=CONTROL 2026-08-15T14:54:31-04:00 ===",
        "commit binding OK: ba57facedf53e209344ab523dc490b6453a96f0a",
        ARGV,
        TOPOLOGY,
        BANNER,
    ]
    body += [f"Epoch 0: {i}/4550 [00:10<00:00, train/loss=0.6{i % 10}]"
             for i in range(n_loss)]
    body.append(TERMINATION)
    return "\n".join(body)


def good_ckpts() -> list[str]:
    return [f"epoch=0-step={s}.ckpt" for s in range(2500, 40001, 2500)]


def audit(log=None, ckpts=None, rc=0, save_dir=SAVE_DIR):
    return audit_full_run(
        log=good_log() if log is None else log,
        ckpt_names=good_ckpts() if ckpts is None else ckpts,
        rc=rc,
        save_dir=save_dir,
    )


# --------------------------------------------------------------------------- #
# happy path + non-vacuity
# --------------------------------------------------------------------------- #
def test_a_complete_run_produces_no_problems():
    assert audit() == []


def test_the_fixture_is_not_trivially_passing():
    assert audit(log="", ckpts=[]), "an empty run must produce problems"


# --------------------------------------------------------------------------- #
# the failure this module exists for
# --------------------------------------------------------------------------- #
def test_an_interrupted_run_that_exited_zero_is_caught():
    """rc=0, banner present, but truncated — the exact shape of the danger."""
    log = good_log().replace(TERMINATION, "")
    ckpts = [f"epoch=0-step={s}.ckpt" for s in range(2500, 32501, 2500)]
    problems = audit(log=log, ckpts=ckpts)
    assert any("did not reach" in p for p in problems)
    assert any("40000" in p for p in problems)


def test_the_endpoint_marker_is_matched_with_its_backticks():
    log = good_log().replace(
        TERMINATION, "`Trainer.fit` stopped: max_steps=40000 reached.")
    assert any("did not reach" in p for p in audit(log=log))


def test_a_different_endpoint_does_not_satisfy_the_pin():
    assert any("did not reach" in p
               for p in audit(log=good_log().replace("max_steps=40000",
                                                     "max_steps=25")))


# --------------------------------------------------------------------------- #
# Codex finding 2: substring evidence let a DIAGNOSTIC counterfeit the fact
# --------------------------------------------------------------------------- #
def test_a_line_merely_quoting_the_marker_does_not_count_as_the_marker():
    """The audit's own failure message used to reproduce the marker verbatim.

    Anything that quotes it — a diagnostic, a shell transcript, a review doc
    pasted into the log — must not satisfy the completion check.
    """
    quoted = good_log().replace(
        TERMINATION,
        f"ENDPOINT NOT REACHED: the marker '{TERMINATION}' is absent")
    assert any("did not reach" in p for p in audit(log=quoted))


def test_a_negated_topology_line_does_not_satisfy_topology():
    """'Not Starting with 2 processes' contains 'Starting with 2 processes'."""
    negated = good_log().replace(TOPOLOGY, f"Not {TOPOLOGY}")
    assert any("rank" in p.lower() or "process" in p.lower()
               for p in audit(log=negated))


# --------------------------------------------------------------------------- #
# Codex finding 1: the checkpoint set must be the registered CADENCE
# --------------------------------------------------------------------------- #
def test_sixteen_checkpoints_at_the_wrong_steps_are_rejected():
    """Rev 1 accepted any 16 distinct steps that included 40000."""
    ckpts = [c.replace("step=37500", "step=42500") for c in good_ckpts()]
    problems = audit(ckpts=ckpts)
    assert any("37500" in p for p in problems), problems


def test_an_off_cadence_extra_checkpoint_is_reported():
    assert any("42500" in p for p in audit(ckpts=good_ckpts() + ["epoch=9-step=42500.ckpt"]))


def test_a_missing_endpoint_checkpoint_is_caught():
    assert any("40000" in p
               for p in audit(ckpts=[c for c in good_ckpts() if "step=40000" not in c]))


def test_a_short_checkpoint_series_is_caught():
    assert audit(ckpts=["epoch=0-step=2500.ckpt", "epoch=8-step=40000.ckpt"])


def test_the_step_number_is_read_as_a_whole_token():
    """'step=140000' must not satisfy the step=40000 requirement."""
    ckpts = [c for c in good_ckpts() if "step=40000" not in c] + ["epoch=9-step=140000.ckpt"]
    assert any("40000" in p for p in audit(ckpts=ckpts))


# --------------------------------------------------------------------------- #
# Codex finding 3+4: fit health must be PARSED, and must not fail open
# --------------------------------------------------------------------------- #
def test_a_log_with_no_loss_evidence_at_all_is_not_declared_healthy():
    """Rev 1's own happy fixture had no loss line, yet it printed 'finite loss'."""
    no_loss = "\n".join([ARGV, TOPOLOGY, BANNER, TERMINATION])
    assert any("loss" in p.lower() for p in audit(log=no_loss))


@pytest.mark.parametrize("bad", ["nan", "NaN", "inf", "-inf", "Infinity", "INF"])
def test_a_nonfinite_loss_value_is_caught(bad):
    log = good_log() + f"\nEpoch 3: 100/4550 [00:10<00:00, train/loss={bad}]"
    assert any("finite" in p.lower() for p in audit(log=log))


@pytest.mark.parametrize("benign", [
    "wandb: saving to /tmp/nan_loss_probe/run.json",
    "checkpoint written to /runs/inf/epoch=0-step=2500.ckpt",
    "loading /data/nan/manifest.json for loss weighting",
])
def test_a_path_containing_nan_or_inf_does_not_fail_a_healthy_run(benign):
    """Rev 1 scanned free text on any line containing 'loss' — a path was enough."""
    assert audit(log=good_log() + "\n" + benign) == []


def test_loss_values_are_parsed_not_pattern_matched():
    """A finite value is fine even beside scary-looking neighbours."""
    assert audit(log=good_log() + "\nEpoch 3: 1/4550 [00:10<00:00, train/loss=0.0001]") == []


# --------------------------------------------------------------------------- #
# Codex finding 5: an unknown exit status is not a zero one
# --------------------------------------------------------------------------- #
def test_rc_is_required_and_nonzero_is_reported():
    assert any("rc=1" in p for p in audit(rc=1))


def test_rc_none_means_unknown_and_is_refused():
    assert any("exit status" in p.lower() for p in audit(rc=None))


# --------------------------------------------------------------------------- #
# Codex finding 2b: the log must be BOUND to the directory it blesses
# --------------------------------------------------------------------------- #
def test_a_log_from_a_different_run_cannot_bless_this_directory():
    problems = audit(save_dir="outputs_FLAC/exp17_YAWAUG_smoke")
    assert any("save-dir" in p.lower() or "does not belong" in p.lower()
               for p in problems), problems


def test_a_log_that_never_records_its_save_dir_is_refused():
    assert audit(log=good_log().replace(ARGV, "ARGV: python train.py --max-steps 40000"))


# --------------------------------------------------------------------------- #
# remaining invariants
# --------------------------------------------------------------------------- #
def test_a_missing_banner_is_caught():
    assert any("banner" in p.lower()
               for p in audit(log=good_log().replace(BANNER, "")))


def test_a_banner_substring_does_not_count_as_the_banner():
    paraphrase = good_log().replace(
        BANNER, f"preflight: {BANNER} was requested")
    assert any("banner" in p.lower() for p in audit(log=paraphrase))


def test_carriage_returns_do_not_hide_the_markers():
    """tqdm writes \\r, not \\n; a line-based check must still see the markers."""
    assert audit(log=good_log().replace("\n", "\r")) == []
