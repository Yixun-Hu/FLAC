"""exp_35 phase (ii) kit, part 3a (plan_orbitrir_yaw_evidence.md §15 items 1-5 + r1 clarifications (b) (c); Codex
plan §15 r1 S6 "Checks and durable records"): the preconditions every mutating mode re-checks, on a sandbox run
worktree (a git repo on its branch, the REAL cells / CANON manifests and supervisor) and a sandbox kit checkout --
the kit's self-pin (its bytes = the reviewed commit's, clean, outside the run worktree), the report evidence (a
committed, clean file of the kit checkout hashing to the given sha256), the run worktree at H35 with its campaign
digest, Yixun's pinned files and the end of his s43 training, and the whole evidence record.

Written by the exp_35 phase-(ii) kit Coder (Claude Opus 5.5, max effort).
"""
import json
import os

import pytest

import exp35_phase_ii_sandbox as SB
from exp35_phase_ii_sandbox import GPU0, GPU1, X25, X35, git, make_box

K = SB.load_kit()


# --------------------------------------------------------------------------- #
# the kit's self-pin (S6: "self-pin the reviewed kit bytes / commit ... do not add or commit the kit inside the
# pinned worktree")
# --------------------------------------------------------------------------- #
def test_the_sandbox_kit_passes_its_self_pin_and_is_recorded(tmp_path):
    b = make_box(tmp_path)
    why, rec = K.kit_evidence(b.cfg, b.commit)
    assert why == [] and rec["commit"] == b.commit and rec["head"] == b.commit
    assert rec["repo"] == os.path.realpath(os.path.join(b.tmp, "kit"))
    assert rec["files"] == {n: K.sha256_bytes(open(os.path.join(b.kit_dir, n), "rb").read()) for n in K.KIT_FILES}


@pytest.mark.parametrize("case, reason", [("malformed", "40-hex"), ("foreign-commit", "not in the kit commit"),
                                          ("modified", "committed, clean"), ("untracked-identical", "committed, clean"),
                                          ("inside-run", "inside the run worktree")])
def test_a_kit_that_is_not_the_reviewed_commit_is_refused(tmp_path, case, reason):
    b = make_box(tmp_path)
    cfg, commit = b.cfg, b.commit
    if case == "malformed":
        commit = "abc"
    elif case == "foreign-commit":
        commit = b.head                                                      # the run worktree's commit: not the kit's
    elif case == "modified":
        with open(os.path.join(b.kit_dir, "exp35_phase_ii.py"), "a") as f:
            f.write("# edited after the review\n")
    elif case == "untracked-identical":                                      # the reviewed bytes, but not tracked
        git(os.path.join(b.tmp, "kit"), "rm", "-q", "--cached", f"{X35}/exp35_phase_ii.py")
    else:
        cfg = cfg._replace(run=os.path.join(b.tmp, "kit"))
    why, _ = K.kit_evidence(cfg, commit)
    assert why and any(reason in x for x in why), why


def test_a_later_kit_head_with_identical_kit_bytes_passes(tmp_path):
    b = make_box(tmp_path)
    with open(os.path.join(b.tmp, "kit", "notes.md"), "w") as f:
        f.write("a later records commit\n")
    git(os.path.join(b.tmp, "kit"), "add", "-A")
    git(os.path.join(b.tmp, "kit"), "commit", "-q", "-m", "records")
    why, rec = K.kit_evidence(b.cfg, b.commit)
    assert why == [] and rec["head"] != b.commit and rec["commit"] == b.commit


# --------------------------------------------------------------------------- #
# the report evidence (the brief: a committed file path + its sha256 given on the command line)
# --------------------------------------------------------------------------- #
def test_the_committed_report_is_the_evidence(tmp_path):
    b = make_box(tmp_path)
    why, rec, raw = K.report_evidence(b.cfg, b.report, b.report_sha)
    assert why == [] and rec == {"path": b.report, "sha256": b.report_sha} and K.sha256_bytes(raw) == b.report_sha
    why, rec, _ = K.report_evidence(b.cfg, os.path.join(b.tmp, "kit", b.report), b.report_sha)    # absolute: same
    assert why == [] and rec["path"] == b.report


@pytest.mark.parametrize("case, reason", [("sha", "hashes to"), ("untracked", "is not committed"),
                                          ("modified", "committed, clean"), ("staged", "committed, clean"),
                                          ("outside", "lies outside the kit checkout"), ("symlink", "is a link"),
                                          ("missing", "readable")])
def test_a_report_that_is_not_the_committed_bytes_is_refused(tmp_path, case, reason):
    b = make_box(tmp_path)
    path, sha, kit = b.report, b.report_sha, os.path.join(b.tmp, "kit")
    if case == "sha":
        sha = "0" * 64
    elif case == "untracked":
        path = f"{X35}/an_uncommitted_report.md"
        open(os.path.join(kit, path), "w").write(open(os.path.join(kit, b.report)).read())
    elif case == "modified":
        with open(os.path.join(kit, b.report), "a") as f:
            f.write("an edit after the commit\n")
        sha = K.sha256_bytes(open(os.path.join(kit, b.report), "rb").read())
    elif case == "staged":                                                    # the index differs; the file is HEAD's
        head = open(os.path.join(kit, b.report)).read()
        open(os.path.join(kit, b.report), "w").write(head + "a staged edit\n")
        git(kit, "add", b.report)
        open(os.path.join(kit, b.report), "w").write(head)
    elif case == "outside":                                                   # the right bytes, the wrong place
        path = os.path.join(b.tmp, "elsewhere.md")
        open(path, "w").write(open(os.path.join(kit, b.report)).read())
    elif case == "symlink":                                                   # a COMMITTED link to the report
        path = f"{X35}/pointer.md"
        os.symlink(os.path.join(kit, b.report), os.path.join(kit, path))
        git(kit, "add", path)
        git(kit, "commit", "-q", "-m", "a link")
    else:
        path = f"{X35}/absent.md"
    why, _, _ = K.report_evidence(b.cfg, path, sha)
    assert why and any(reason in x for x in why), why


# --------------------------------------------------------------------------- #
# the run worktree (§15 item 2; r1 (b): the runner-relevant tree committed and clean, the exact campaign digest)
# --------------------------------------------------------------------------- #
def _m(b):
    return json.load(open(os.path.join(b.run, K.CELLS)))


def test_the_sandbox_run_worktree_holds(tmp_path):
    b = make_box(tmp_path)
    assert K.worktree_reasons(b.cfg, _m(b)) == []
    open(os.path.join(b.run, X35, "untracked_phase_i_evidence.log"), "w").write("x\n")   # untracked, not relevant: fine
    assert K.worktree_reasons(b.cfg, _m(b)) == []


@pytest.mark.parametrize("case, reason", [
    ("head", "is not H35"), ("branch", "not exp-35-orbitrir-yaw"), ("tracked-dirty", "tracked tree is dirty"),
    ("relevant-untracked", "not committed and clean"), ("digest", "not the campaign digest"),
    ("not-root", "not a worktree root")])
def test_every_wrong_run_worktree_is_refused(tmp_path, case, reason):
    b = make_box(tmp_path)
    cfg = b.cfg
    if case == "head":
        git(b.run, "commit", "-q", "--allow-empty", "-m", "moved")
    elif case == "branch":
        git(b.run, "checkout", "-q", "-b", "elsewhere")
    elif case == "tracked-dirty":
        with open(os.path.join(b.run, "README.md"), "a") as f:
            f.write("edited\n")
    elif case == "relevant-untracked":
        open(os.path.join(b.run, "src", "new_module.py"), "w").write("x = 1\n")
    elif case == "digest":
        cfg = cfg._replace(digest="0" * 64)
    else:
        cfg = cfg._replace(run=os.path.join(b.run, "src"))
    why = K.worktree_reasons(cfg, _m(b))
    assert why and any(reason in x for x in why), why


# --------------------------------------------------------------------------- #
# Yixun's pinned files and the end of his s43 training (§15 item 4; S6 "stale queue pin")
# --------------------------------------------------------------------------- #
def test_his_files_at_their_pins_and_his_s43_training_ended(tmp_path):
    b = make_box(tmp_path)
    assert K.external_reasons(b.cfg) == []


@pytest.mark.parametrize("case, reason", [("pin", "not its pin"), ("running", "has not ended"),
                                          ("restarted", "has not ended"), ("unreadable", "unreadable")])
def test_a_stale_queue_pin_or_a_running_s43_is_refused(tmp_path, case, reason):
    b = make_box(tmp_path)
    if case == "pin":
        with open(next(iter(b.cfg.exp30_pins)), "a") as f:
            f.write("TRAIN_JOBS=('ACEattn 45')\n")
    elif case == "running":
        open(b.cfg.queue_log, "w").write("2026-11-04T06:12:55-05:00 START training ACEattn s43 gpus 0,1\n")
    elif case == "restarted":
        with open(b.cfg.queue_log, "a") as f:
            f.write("2026-11-06T15:00:00-05:00 START training ACEattn s43 gpus 0,1 fork=x\n")
    else:
        os.unlink(b.cfg.queue_log)
    why = K.external_reasons(b.cfg)
    assert why and any(reason in x for x in why), why


# --------------------------------------------------------------------------- #
# the whole evidence record (S6 "record H35, manifest hash and exact 68-cell-name digest, CANON-manifest /
# checkpoint hashes, campaign digest, dump root, queue / evaluator hashes, lane indices / UUIDs, report evidence,
# ETA, and any second-lane decision")
# --------------------------------------------------------------------------- #
def _ev(b, lanes=(0,), prior=()):
    return K.evidence(b.cfg, lanes=lanes, report=b.report, report_sha256=b.report_sha, prior_runs=prior)


def test_the_evidence_holds_and_records_every_registered_fact(tmp_path):
    b = make_box(tmp_path)
    why, rec = _ev(b)
    assert why == []
    m = _m(b)
    assert rec == {
        "h35": b.head, "run_worktree": b.run, "campaign_digest": b.digest, "cells_sha256": K.CELLS_SHA256,
        "names_sha256": K.NAMES_SHA256, "n_cells": 68, "blocks": list(K.BLOCKS), "dump_root": K.DUMP_ROOT,
        "canon_manifest": K.CANON_MANIFEST, "canon_manifest_sha256": K.CANON_MANIFEST_SHA256,
        "checkpoints": {"P1": [m["arms"]["P1"]["ckpt"]["path"], m["arms"]["P1"]["ckpt"]["sha256"]],
                        "BF": [m["arms"]["BF"]["ckpt"]["path"], m["arms"]["BF"]["ckpt"]["sha256"]],
                        "CANON": list(K.CANON_CKPT)},
        "exp30": dict(b.cfg.exp30_pins), "queue_log": b.cfg.queue_log, "s43_end": SB.END_S43.strip(),
        "phase_i_state_sha256": b.state_sha, "supervisor_sha256": K.sha256_bytes(SB.real_supervisor()),
        "report": {"path": b.report, "sha256": b.report_sha}, "lanes": [{"index": 0, "uuid": GPU0}],
        "eta": "68 cells on GPU 0, about 19 h", "second_lane_decision": "none", "prior_runs": []}


def test_a_recorded_second_lane_is_evidence(tmp_path):
    b = make_box(tmp_path, lanes="0,1", decision="Planner 2026-11-06T11:40-05:00: Yixun chose two cards")
    why, rec = _ev(b, lanes=(0, 1))
    assert why == [] and rec["lanes"] == [{"index": 0, "uuid": GPU0}, {"index": 1, "uuid": GPU1}]
    assert rec["second_lane_decision"].startswith("Planner")


@pytest.mark.parametrize("case, reason", [
    ("phase-i-incomplete", "not gates_verified"), ("state-after-report", "phase-i-state-sha256"),
    ("cells", "cells manifest hashes to"), ("canon", "CANON manifest hashes to"), ("lanes", "lanes"),
    ("unrecorded-second-lane", "second lane"), ("prior", "prior-runs"), ("state-missing", "keys"),
    ("supervisor-file", "supervisor_sha256")])
def test_every_failed_precondition_is_named(tmp_path, case, reason):
    b = make_box(tmp_path)
    lanes, prior = (0,), ()
    if case == "phase-i-incomplete":
        SB.write_phase_i(b.run, b.head, b.digest, stop_latch={"reason": "x", "time": "t"})
        reason = "stop latch"
    elif case == "state-after-report":
        SB.write_phase_i(b.run, b.head, b.digest, approved_sha256="f" * 64)
    elif case == "cells":
        with open(os.path.join(b.run, K.CELLS), "a") as f:
            f.write(" ")
    elif case == "canon":
        with open(os.path.join(b.run, K.CANON_MANIFEST), "a") as f:
            f.write("\n")
    elif case == "lanes":
        lanes = (1,)
    elif case == "unrecorded-second-lane":
        b = make_box(tmp_path / "two", lanes="0,1")
        lanes = (0, 1)
    elif case == "prior":
        prior = ("20261106T143000",)
    elif case == "state-missing":
        os.unlink(os.path.join(b.run, K.SUP_STATE))
    else:
        os.unlink(os.path.join(b.run, K.SUP_FILE))                            # the supervisor's bytes cannot be hashed
    why, _ = _ev(b, lanes=lanes, prior=prior)
    assert why and any(reason in x for x in why), why
