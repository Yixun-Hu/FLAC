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
minute. Nothing downstream would notice: the checkpoint directory would simply
hold fewer files than expected, and a 32,500-step checkpoint looks exactly like
a 40,000-step one to a glob.

This module is the missing check, as a pure function over (log text, checkpoint
filenames) so it can be unit-tested without a 21-hour run and applied read-only
to a run this session did not launch. It never writes, never kills, never
touches the running job.

Fixture strings below are the LITERAL text Lightning and the launcher emit —
copied from the real logs, not paraphrased. That matters: an earlier version of
this check searched for ``max_steps=40000 reached`` and could never match,
because Lightning writes ```max_steps=40000` reached`` with backticks
around the assignment. It failed closed, so it cost a re-run rather than a false
pass, but a gate that cannot match is not a gate.

Written by the main session seat (Claude Opus 5, max effort).
"""
import pytest

from src.tools.exp17_full_audit import audit_full_run, BANNER


# The literal terminal text, as it appears in the logs on disk.
TERMINATION = "`Trainer.fit` stopped: `max_steps=40000` reached."
TOPOLOGY = "All distributed processes registered. Starting with 2 processes"
PROGRESS = "Epoch 8:  99%|#########9| 4550/4550 [2:28:37<00:00,  0.51it/s]"


def good_log() -> str:
    return "\n".join([
        "=== exp_17 FULL control stage=CONTROL 2026-08-15T14:54:31-04:00 ===",
        "commit binding OK: ba57facedf53e209344ab523dc490b6453a96f0a",
        TOPOLOGY,
        BANNER,
        PROGRESS,
        TERMINATION,
    ])


def good_ckpts() -> list[str]:
    return [f"epoch=0-step={s}.ckpt" for s in range(2500, 40001, 2500)]


# --------------------------------------------------------------------------- #
# the happy path, and its non-vacuity
# --------------------------------------------------------------------------- #
def test_a_complete_run_produces_no_problems():
    assert audit_full_run(log=good_log(), ckpt_names=good_ckpts(), rc=0) == []


def test_the_fixture_is_not_trivially_passing():
    """If the audit ignored its inputs, every test below would also pass."""
    assert audit_full_run(log="", ckpt_names=[], rc=0), (
        "an empty run must produce problems; the audit is vacuous otherwise"
    )


# --------------------------------------------------------------------------- #
# the failure this whole module exists for
# --------------------------------------------------------------------------- #
def test_an_interrupted_run_that_exited_zero_is_caught():
    """The exact shape of the danger: rc=0, banner present, but truncated.

    Lightning swallows KeyboardInterrupt, so rc is 0; the banner was printed
    before step 0, so it is present; only the endpoint evidence is missing.
    """
    log = good_log().replace(TERMINATION, "")
    ckpts = [f"epoch=0-step={s}.ckpt" for s in range(2500, 32501, 2500)]

    problems = audit_full_run(log=log, ckpt_names=ckpts, rc=0)

    assert any("did not reach" in p for p in problems)
    assert any("step=40000" in p for p in problems)


def test_the_endpoint_marker_is_matched_with_its_backticks():
    """Regression pin for the bug that made an earlier gate unmatchable."""
    without_backticks = good_log().replace(
        TERMINATION, "`Trainer.fit` stopped: max_steps=40000 reached."
    )
    problems = audit_full_run(log=without_backticks, ckpt_names=good_ckpts(), rc=0)
    assert any("did not reach" in p for p in problems), (
        "the marker must be matched as Lightning literally writes it"
    )


def test_a_different_endpoint_does_not_satisfy_the_pin():
    """A 25-step smoke log must never audit as a completed FULL run."""
    smoke = good_log().replace("max_steps=40000", "max_steps=25")
    assert any("did not reach" in p for p in audit_full_run(
        log=smoke, ckpt_names=good_ckpts(), rc=0))


# --------------------------------------------------------------------------- #
# the remaining invariants
# --------------------------------------------------------------------------- #
def test_a_missing_endpoint_checkpoint_is_caught():
    ckpts = [c for c in good_ckpts() if "step=40000" not in c]
    assert any("step=40000" in p for p in audit_full_run(
        log=good_log(), ckpt_names=ckpts, rc=0))


def test_a_short_checkpoint_series_is_caught():
    """40,000 / 2,500 = 16 checkpoints; fewer means gaps in the trajectory."""
    ckpts = ["epoch=0-step=2500.ckpt", "epoch=8-step=40000.ckpt"]
    problems = audit_full_run(log=good_log(), ckpt_names=ckpts, rc=0)
    assert any("16" in p for p in problems)


def test_a_missing_banner_is_caught():
    """The treatment must be provably live; this is exp_17's cardinal failure."""
    assert any("banner" in p.lower() for p in audit_full_run(
        log=good_log().replace(BANNER, ""), ckpt_names=good_ckpts(), rc=0))


def test_a_banner_substring_does_not_count_as_the_banner():
    """Whole-line match: a preflight paraphrase must not satisfy it."""
    paraphrase = good_log().replace(BANNER, "preflight: yaw_aug enabled img_w=512")
    assert any("banner" in p.lower() for p in audit_full_run(
        log=paraphrase, ckpt_names=good_ckpts(), rc=0))


def test_single_rank_topology_is_caught():
    """BN=64 needs 2 ranks; one rank silently halves the BN batch."""
    single = good_log().replace(TOPOLOGY, "Starting with 1 processes")
    assert any("rank" in p.lower() or "process" in p.lower()
               for p in audit_full_run(log=single, ckpt_names=good_ckpts(), rc=0))


def test_nonzero_rc_is_reported_even_when_everything_else_looks_complete():
    assert any("rc=1" in p for p in audit_full_run(
        log=good_log(), ckpt_names=good_ckpts(), rc=1))


@pytest.mark.parametrize("bad", ["nan", "NaN", "inf", "Inf"])
def test_nonfinite_loss_is_caught(bad):
    log = good_log() + f"\nEpoch 3: train/loss={bad}\n"
    assert any("finite" in p.lower() or "nan" in p.lower()
               for p in audit_full_run(log=log, ckpt_names=good_ckpts(), rc=0))


def test_carriage_returns_do_not_hide_the_markers():
    """tqdm writes \\r, not \\n; a line-based grep must still see the markers."""
    crlf = good_log().replace("\n", "\r")
    assert audit_full_run(log=crlf, ckpt_names=good_ckpts(), rc=0) == []
