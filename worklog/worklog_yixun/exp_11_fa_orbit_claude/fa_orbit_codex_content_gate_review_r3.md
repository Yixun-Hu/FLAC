Reading additional input from stdin...
OpenAI Codex v0.146.0
--------
workdir: /n/fs/gatrdp/codespace/FLAC
model: gpt-5.6-sol
provider: openai
approval: on-request
sandbox: danger-full-access
reasoning effort: xhigh
reasoning summaries: none
session id: 019ff595-741d-7751-a25d-104f006b5084
--------
user
Round-3 delta review (read-only; do NOT install anything or modify environments/files; output = review text only). Repo /n/fs/gatrdp/codespace/FLAC. You APPROVED the content-scoped commit-binding gate in worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch in round 2. In production it then aborted 4 legs on a src/tests-ONLY commit (test files land continuously from concurrent TDD sessions; src/tests is pytest-only and never imported by train.py at runtime). Delta under review (diff vs HEAD at /tmp/claude-374349/-n-fs-gatrdp-codespace-FLAC/66305eb9-88c7-41df-95da-dde30a426dc2/scratchpad/content_gate_r3.diff): ':(exclude)src/tests' added to BOTH the closure diff pathspec and the drift-gate pathspec, plus a new synthetic guard case (src/tests-only dangling commit must PASS the gate; all 5 gate cases pass). Check: (1) is it true that nothing under src/tests is imported at training runtime (conftest side effects? packages importing src.tests?); (2) git pathspec exclude-magic semantics in BOTH git diff and git status with the other pathspecs present (does the exclude apply as intended, any version caveat for git on this host); (3) the new guard fixture's plumbing (reuses SYN_BLOB from the earlier fixture — ordering dependency OK?); (4) any fresh hole. Verdict: APPROVED or REVISE.
codex
I’ll inspect the round-3 patch, the batch script’s full gate context, runtime import graph, and the host Git behavior. I’ll keep this strictly read-only and return only the review verdict/text.
exec
/usr/bin/bash -lc "rg -n --hidden --glob '"'!'"\\.git/**' '(src\\.tests|from[[:space:]]+src[.]tests|import[[:space:]]+src[.]tests|tests[.]|conftest)' ." in /n/fs/gatrdp/codespace/FLAC
 exited 127 in 0ms:
/usr/bin/bash: line 1: rg: command not found

exec
/usr/bin/bash -lc "sed -n '1,260p' /tmp/claude-374349/-n-fs-gatrdp-codespace-FLAC/66305eb9-88c7-41df-95da-dde30a426dc2/scratchpad/content_gate_r3.diff" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
diff --git a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
index f039a3c..185d911 100644
--- a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
+++ b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
@@ -206,7 +206,7 @@ EXPREL="${EXPDIR#"$REPO"/}"; EXP07REL="${EXP07#"$REPO"/}"
 # git, not the shell, expands them — a tracked file deleted from the worktree
 # still matches (content-gate review B2) — data/AR (the split JSONs the
 # dataloader opens) is covered, and a failing git status is fail-closed.
-DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src data/AR \
+DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src ":(exclude)src/tests" data/AR \
           "$EXPREL/*.json" "$EXPREL/*.py" "$EXPREL/*.sbatch" "$EXPREL/*.sh" \
           "$EXP07REL/FLAC_AR_BF.json" 2>&1)" \
   || die "git status for the drift gate failed: ${DRIFT} - abort"
@@ -223,8 +223,12 @@ DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini sr
 # EXPECT_SHA must be the full 40-hex commit OID (a symbolic ref like HEAD
 # would defeat the binding), the diff runs against the CAPTURED HEAD OID,
 # and HEAD is re-read afterwards to close the mid-gate-commit race.
+# src/tests is EXCLUDED from both gates: pytest-only code, never imported by
+# train.py at runtime, and the TDD sessions land test files continuously —
+# gating them reproduced the exact kill this gate exists to prevent
+# (legs 3680875-78 died on a src/tests-only commit, 2026-08-12).
 surface_diff_vs_expect() {
-  git diff --name-only "${EXPECT_SHA}" "${HEAD_SHA}" -- train.py defaults.ini src data/AR \
+  git diff --name-only "${EXPECT_SHA}" "${HEAD_SHA}" -- train.py defaults.ini src ":(exclude)src/tests" data/AR \
       "$EXPDIR"/FLAC_AR_BF_C4L.json "$EXPDIR"/FLAC_AR_BF_C8.json \
       "$EXPDIR"/FLAC_AR_BF_C16.json "$EXPDIR"/FLAC_AR_BF_C32.json \
       "$EXPDIR"/FLAC_AR_VANCKPT.json "$EXPDIR"/fa_orbit_train.sbatch \
diff --git a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
index a7d0f5b..b487055 100755
--- a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
+++ b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
@@ -178,6 +178,20 @@ else
 fi
 case_run "symbolic EXPECT_SHA refused" 2 "not a full lowercase 40-hex" \
   -- ARM=C4L SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=99000000 EXPECT_SHA=HEAD SLURM_JOB_ID=999999
+# src/tests is excluded from the closure (pytest-only; TDD sessions land tests
+# continuously — a test-only commit must NOT kill a queued leg: 3680875-78).
+SYN_TIDX="${TMP}/syntidx"; SYN_TESTS=""
+if GIT_INDEX_FILE="$SYN_TIDX" git read-tree HEAD 2>/dev/null && [ -n "${SYN_BLOB:-}" ] \
+   && GIT_INDEX_FILE="$SYN_TIDX" git update-index --add --cacheinfo 100644 "$SYN_BLOB" src/tests/test_guardtest_synthetic.py 2>/dev/null; then
+  SYN_TTREE="$(GIT_INDEX_FILE="$SYN_TIDX" git write-tree 2>/dev/null)"
+  [ -n "$SYN_TTREE" ] && SYN_TESTS="$(git commit-tree "$SYN_TTREE" -p HEAD -m 'guardtest synthetic: src/tests only' 2>/dev/null)"
+fi
+if [ -n "$SYN_TESTS" ]; then
+  case_run "src/tests-only change -> gate passes" 2 "commit binding OK (content)" \
+    -- ARM=C4L SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=99000000 "EXPECT_SHA=${SYN_TESTS}" SLURM_JOB_ID=999999
+else
+  echo "FAIL  could not synthesize the src/tests-only fixture"; FAIL=$((FAIL+1))
+fi
 
 echo "--- E. semantic gate on a mislabelled config (temp copy; tracked tree untouched) ---"
 FAKE_EXP="${TMP}/fakeexp"; mkdir -p "$FAKE_EXP"

exec
/usr/bin/bash -lc "sed -n '1,320p' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
#!/bin/bash
# ============================================================================
# exp_11 ARM TRAINING LAUNCHER — one orbit arm at the ONE pinned P0 rung.
#
# Round-3 review (fa_orbit_codex_code_r3_review.md) rebuilt this script: the
# recipe is now literally pinned rather than operator-supplied, restarts get
# exp_10's torch.load depth, the run directory is locked, the exit taxonomy is a
# tested function, and the environment/W&B destinations are gated.
#
# RECIPE (post-P0, Yixun 2026-08-05): every arm runs WITH ViT gradient
# checkpointing. P0 measured the no-ckpt recipe INFEASIBLE for C8 and richer
# orbits (OOM at micro-8, 45,457 MiB) while checkpointed peaks at ~9.4 GB, so all
# four arms share one checkpointed rung and the ONLY delta between them is the
# averaging orbit.
#
# SUBMIT: never by hand — `fa_orbit_submit.sh <ARM>` derives every Slurm flag
# from the pins below, so an operator cannot mis-enter --gres/--mem/--time.
#   ARM                C4L | C8 | C16 | C32
#   EXPECT_SHA         full 40-hex reviewed commit OID (required). Binding is
#                      by CONTENT of the training surfaces, not HEAD identity:
#                      a launch is accepted when HEAD == EXPECT_SHA, or when
#                      the training closure is byte-identical between the two
#                      (two writers commit to this checkout; worklog/record
#                      commits must not kill a queued leg).
#   RESUME_CKPT/EXPECTED_STEP   crash restart only (see LINEAGE)
#   SMOKE=1            the reviewed multi-GPU smoke (see SMOKE MODE)
# RUNG / MAXSTEPS / MIN_FREE_MB / time limit are NOT operator inputs any more.
#
# LINEAGE (fail-closed, exactly two stories):
#   INITIAL  no RESUME_CKPT, EXPECTED_STEP unset/0, run directory absent.
#   RESTART  EXPECTED_STEP > 0 AND RESUME_CKPT inside this arm's OWN
#            <RUNDIR>/checkpoints/ AND the checkpoint passes
#            fa_orbit_ckpt_preflight.py (embedded step/config/optimizer/
#            scheduler/EMA + binding to the original launch manifest).
#
# WORLD SIZE: no absence timer (round-3 B4 — a cold start with W&B has no
# measured bound, and `scancel` bypassed classification). Instead: a watcher that
# terminates the torchrun process group the moment Lightning reports the WRONG
# rank count, plus the post-hoc classification in fa_orbit_classify.py.
#
# torchrun: PL 2.1.0 elects TorchElastic before SLURMEnvironment, so the ranks
# torchrun starts are used as-is; the SLURM rank variables are unset so
# SLURMEnvironment cannot claim the job. train.py is unmodified and rank-safe:
# WandbLogger.experiment is @rank_zero_experiment, and ModelCheckpoint.setup
# broadcasts rank 0's dirpath to every rank.
#
# SMOKE MODE (SMOKE=1): the reviewed pre-launch smoke. Bypasses ONLY the "pins
# must be pinned" gate; every other gate still runs. Uses SMOKE_RUNG,
# SMOKE_MAXSTEPS (small), SMOKE_MIN_FREE_MB, its own identity
# (FLAC_exp11_smoke_<ARM> / exp11_smoke_<ARM>) and its own save-dir prefix, so a
# smoke can never touch or resume an arm's real lineage.
#
# TEST HOOK: OUTPUT_ROOT (default outputs_FLAC) relocates the output namespace so
# the guard tests never write under a production prefix. It changes no gate.
# ============================================================================
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_train_%x_%j.out
# TRANSCRIPT POLICY. This file is written by Slurm for the whole life of the run.
# During the run it is deliberately UNTRACKED (the job removes it from the index
# at launch, see the untrack block below): a tracked file that a running job
# appends to is one a git checkout/stash can unlink out from under the job's file
# descriptor, freezing the visible transcript while the run continues. Completed
# transcripts are committed by the OPERATOR at run closure with `git add -f`.

set -uo pipefail

# ============================ PINNED RECIPE =================================
# Filled from the reviewed P0 report; until then every value is the literal
# placeholder and the launcher refuses to run (except under SMOKE=1).
PIN_PLACEHOLDER="TO-PIN-AFTER-P0"
PINNED_RUNG="8x8"                          # P0 run 1334933 + spot 9bf1936: fastest uniform rung where ALL arms fit (C32 peak 30,817 MiB)
PINNED_MB="8"                              # micro-batch per GPU (8 x 8 = 64 = eff = BN batch)
PINNED_NGPU="8"                            # ranks
PINNED_MAXSTEPS=100000                     # Q10: extended budget (was 40000, the
                                           # plan §2 primary matched step, which
                                           # remains the TABLE step — the extension
                                           # adds trajectory, it does not move the
                                           # registered comparison point)
PINNED_CHECKPOINT_EVERY=2500               # exp_07 cadence
PINNED_MIN_FREE_MB="36500"                 # batched C32 peak 32,063 MiB + ~4.4 GB margin (max-across-arms floor)
PINNED_TIME_LIMIT_C4L="24:00:00"           # batched 40k/0.6598 = 16.8 h x1.3 + startup
PINNED_TIME_LIMIT_C8="35:00:00"            # batched 40k/0.4351 = 25.5 h x1.3 + startup
PINNED_TIME_LIMIT_C16="60:00:00"           # batched 40k/0.2454 = 45.3 h x1.3 + startup
PINNED_TIME_LIMIT_C32="112:00:00"          # batched 40k/0.1308 = 84.9 h x1.3 + startup — SINGLE segment (no wall-split needed)
# VANL is the vanilla-conditioning arm of the SAME recipe (Q9): its cost comes
# from the official P0 VAN_8x8 rate, not from an orbit slope, because it makes no
# orbit passes at all — 40k/1.07 steps/s = 10.4 h x1.3 + startup.
PINNED_TIME_LIMIT_VANL="14:00:00"
# Q10 RESTART legs: 40k -> 100k is 60,000 further steps at the batched rates,
# x1.3 + startup. Each must sit under the 168 h partition cap, and each does.
PINNED_TIME_LIMIT_RESTART_C4L="34:00:00"    # 60k/0.6598 = 25.3 h
PINNED_TIME_LIMIT_RESTART_C8="51:00:00"     # 60k/0.4351 = 38.3 h
PINNED_TIME_LIMIT_RESTART_C16="89:00:00"    # 60k/0.2454 = 67.9 h
PINNED_TIME_LIMIT_RESTART_C32="160:00:00"   # 60k/0.1308 = 127.4 h (cap 168 h)
PINNED_TIME_LIMIT_RESTART_VANL="19:00:00"   # 60k/1.0722 = 15.5 h
PINNED_P0_MANIFEST_SHA256="72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b"  # batched matrix manifest bd96575-…-a3ed28eb; spot manifest sha in the commit message
# Environment pins (round-3 B6) — measured on the reviewed environment:
PINNED_PYTHON="/n/fs/gatrdp/envs/flac/bin/python"
PINNED_PL_VERSION="2.1.0"
PINNED_TORCH_VERSION="2.7.0+cu126"
PINNED_VAE_SHA256="8d82159eec35210198246f449bec6561fc19b514922f340a17515050daf7f0b9"
# ============================================================================

REPO=/n/fs/gatrdp/codespace/FLAC
# TEST HOOK (guard tests only): sbatch copies this script to a spool dir, so the
# repo path must be absolute; FA_ORBIT_REPO_OVERRIDE lets the guard suite point a
# dry run at a worktree. It is honoured ONLY outside a Slurm job and scrubbed
# immediately, so it can never influence a real launch.
if [ -n "${FA_ORBIT_REPO_OVERRIDE:-}" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
  REPO="$FA_ORBIT_REPO_OVERRIDE"
fi
unset FA_ORBIT_REPO_OVERRIDE
EXPDIR="$REPO/worklog/worklog_yixun/exp_11_fa_orbit_claude"
EXP07="$REPO/worklog/worklog_yixun/exp_07_fa_scratch_claude"
cd "$REPO" || exit 3
unset PYTHONPATH PYTHONOPTIMIZE
export PATH=/n/fs/gatrdp/envs/flac/bin:$PATH
export PYTHONNOUSERSITE=1
export HF_HOME=/n/fs/gatrdp/hf_cache
export HF_HUB_OFFLINE=1

DRYRUN="${DRYRUN:-0}"
SMOKE="${SMOKE:-0}"
# NEW-2: the production output namespace is not operator state. Inside a Slurm
# job it is the literal below; an ambient value that disagrees aborts. The
# override exists only for non-Slurm guard dry runs.
PRODUCTION_OUTPUT_ROOT="outputs_FLAC"
if [ -n "${SLURM_JOB_ID:-}" ]; then
  if [ -n "${OUTPUT_ROOT:-}" ] && [ "$OUTPUT_ROOT" != "$PRODUCTION_OUTPUT_ROOT" ]; then
    echo "ambient OUTPUT_ROOT='${OUTPUT_ROOT}' != the production literal '${PRODUCTION_OUTPUT_ROOT}' - abort"; exit 2
  fi
  OUTPUT_ROOT="$PRODUCTION_OUTPUT_ROOT"
else
  OUTPUT_ROOT="${OUTPUT_ROOT:-$PRODUCTION_OUTPUT_ROOT}"
fi
RESUME_CKPT="${RESUME_CKPT:-}"
EXPECTED_STEP="${EXPECTED_STEP:-0}"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"

die() { echo "$1"; exit "${2:-2}"; }

# --- A. parameters ------------------------------------------------------------
[ -n "${ARM:-}" ] || die "ARM must be exported (C4L|C8|C16|C32|VANL) - abort"
[ -n "${EXPECT_SHA:-}" ] || die "EXPECT_SHA (full reviewed commit sha) must be exported - abort"
case "$ARM" in
  C4L|C8|C16|C32|VANL) ;;
  *) die "ARM '${ARM}' is not a legal exp_11 arm — C4L|C8|C16|C32 only (FA1/VAN/CKPT4 are P0 profiling cells, never arms) - abort" ;;
esac
case "$EXPECTED_STEP" in ''|*[!0-9]*) die "EXPECTED_STEP '${EXPECTED_STEP}' must be a non-negative integer - abort";; esac

# --- B. the pins decide the recipe (round-3 B1) -------------------------------
if [ "$SMOKE" = "1" ]; then
  RUNG="${SMOKE_RUNG:-}"; MAXSTEPS="${SMOKE_MAXSTEPS:-30}"; MIN_FREE_MB="${SMOKE_MIN_FREE_MB:-}"
  CHECKPOINT_EVERY="${SMOKE_CHECKPOINT_EVERY:-10}"
  [ -n "$RUNG" ] || die "SMOKE=1 requires SMOKE_RUNG (32x2|16x4|8x8) - abort"
  [ -n "$MIN_FREE_MB" ] || die "SMOKE=1 requires SMOKE_MIN_FREE_MB (per-GPU floor) - abort"
  TIME_LIMIT="${SMOKE_TIME:-00:30:00}"; TIME_PIN_NAME="SMOKE_TIME"
  NAME="FLAC_exp11_smoke_${ARM}"; EXPNAME="exp11_smoke_${ARM}"
  SAVEDIR="${OUTPUT_ROOT}/exp11_smoke/${ARM}"
  echo "=== SMOKE MODE: pins bypassed, EVERY other gate active; identity ${EXPNAME} ==="
else
  # Q10 / re-pin fix 1: the wall pin follows the LEG, not the arm. A restart leg
  # is 60,000 further steps, not 40,000 from scratch, so the submitter allocates
  # PINNED_TIME_LIMIT_RESTART_<ARM>. The job selected PINNED_TIME_LIMIT_<ARM>
  # regardless and then rejected its own (correct) allocation in gate H — the
  # third hard-abort path the re-pin review found on jobs 3662828-30. The JOB now
  # selects the same pin the submitter did and enforces THAT one.
  if [ "$EXPECTED_STEP" -gt 0 ]; then
    TIME_PIN_NAME="PINNED_TIME_LIMIT_RESTART_${ARM}"
  else
    TIME_PIN_NAME="PINNED_TIME_LIMIT_${ARM}"
  fi
  for PIN_NAME in PINNED_RUNG PINNED_MB PINNED_NGPU PINNED_MIN_FREE_MB PINNED_P0_MANIFEST_SHA256 \
                  "$TIME_PIN_NAME"; do
    eval "PIN_VAL=\${$PIN_NAME}"
    [ "$PIN_VAL" != "$PIN_PLACEHOLDER" ] || die "${PIN_NAME} is still '${PIN_PLACEHOLDER}': the P0 report has not been pinned into this launcher yet — no arm may launch (use SMOKE=1 for the pre-launch smoke) - abort"
  done
  RUNG="$PINNED_RUNG"; MAXSTEPS="$PINNED_MAXSTEPS"; MIN_FREE_MB="$PINNED_MIN_FREE_MB"
  CHECKPOINT_EVERY="$PINNED_CHECKPOINT_EVERY"
  eval "TIME_LIMIT=\${${TIME_PIN_NAME}}"
  NAME="FLAC_exp11_${ARM}"; EXPNAME="exp11_${ARM}"; SAVEDIR="${OUTPUT_ROOT}/exp11_${ARM}"
fi

case "$RUNG" in
  32x2|16x4|8x8) ;;
  *) die "rung '${RUNG}' must be 32x2, 16x4 or 8x8 - abort" ;;
esac
MB="${RUNG%x*}"; NGPU="${RUNG#*x}"
[ "$((MB * NGPU))" -eq 64 ] || die "rung ${RUNG}: MB*NGPU = $((MB*NGPU)) != 64 (micro x N pin, plan §10) - abort"
if [ "$SMOKE" != "1" ]; then
  [ "$MB" = "$PINNED_MB" ] && [ "$NGPU" = "$PINNED_NGPU" ] || die "pin inconsistency: rung ${RUNG} vs PINNED_MB=${PINNED_MB}/PINNED_NGPU=${PINNED_NGPU} - abort"
  [ "$MAXSTEPS" = "100000" ] || die "PINNED_MAXSTEPS is ${MAXSTEPS}, the registered budget is 100000 - abort"
fi
RUNDIR="${SAVEDIR}/${NAME}/${EXPNAME}"
echo "=== exp_11 arm ${ARM} @ rung ${RUNG} (MB ${MB} x ${NGPU} GPU, grad-ckpt ON) — ${TS} — host $(hostname) ==="

# --- C. commit binding + tracked-surface drift --------------------------------
HEAD_SHA="$(git rev-parse HEAD 2>/dev/null)" || HEAD_SHA=""
EXPREL="${EXPDIR#"$REPO"/}"; EXP07REL="${EXP07#"$REPO"/}"
# The drift gate is scoped to CODE surfaces, not the whole exp folder: the four
# arms are running and Slurm appends to their tracked *.out logs continuously, so
# a folder-wide check would abort every screen on a live-log write. Configs,
# drivers and validators are still fully covered. The patterns are QUOTED so
# git, not the shell, expands them — a tracked file deleted from the worktree
# still matches (content-gate review B2) — data/AR (the split JSONs the
# dataloader opens) is covered, and a failing git status is fail-closed.
DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src ":(exclude)src/tests" data/AR \
          "$EXPREL/*.json" "$EXPREL/*.py" "$EXPREL/*.sbatch" "$EXPREL/*.sh" \
          "$EXP07REL/FLAC_AR_BF.json" 2>&1)" \
  || die "git status for the drift gate failed: ${DRIFT} - abort"
# Commit binding is CONTENT-scoped: HEAD identity is sufficient but not
# necessary. Two sessions commit to this checkout, so a pending leg must
# survive commits that leave the training closure untouched — and abort on
# any commit that changes it. The closure is what the job actually loads:
# train.py, defaults.ini, src/, the data/AR split JSONs, the five arm
# configs (enumerated — a shell glob would silently drop a config deleted
# since EXPECT_SHA), this launcher, the four runtime helper scripts it
# invokes, and exp_07's FLAC_AR_BF.json (C4L parity baseline).
# Record/analysis files (registry, manifests, gen_*/validators, worklog)
# are deliberately OUTSIDE the closure. Fail-closed on every edge:
# EXPECT_SHA must be the full 40-hex commit OID (a symbolic ref like HEAD
# would defeat the binding), the diff runs against the CAPTURED HEAD OID,
# and HEAD is re-read afterwards to close the mid-gate-commit race.
# src/tests is EXCLUDED from both gates: pytest-only code, never imported by
# train.py at runtime, and the TDD sessions land test files continuously —
# gating them reproduced the exact kill this gate exists to prevent
# (legs 3680875-78 died on a src/tests-only commit, 2026-08-12).
surface_diff_vs_expect() {
  git diff --name-only "${EXPECT_SHA}" "${HEAD_SHA}" -- train.py defaults.ini src ":(exclude)src/tests" data/AR \
      "$EXPDIR"/FLAC_AR_BF_C4L.json "$EXPDIR"/FLAC_AR_BF_C8.json \
      "$EXPDIR"/FLAC_AR_BF_C16.json "$EXPDIR"/FLAC_AR_BF_C32.json \
      "$EXPDIR"/FLAC_AR_VANCKPT.json "$EXPDIR"/fa_orbit_train.sbatch \
      "$EXPDIR"/fa_orbit_ckpt_preflight.py "$EXPDIR"/assert_arm_configs_exp11.py \
      "$EXPDIR"/fa_orbit_wandb_readback.py "$EXPDIR"/fa_orbit_classify.py \
      "$EXP07/FLAC_AR_BF.json"
}
GATE_FAIL=""; GATE_OK_MSG=""
if [ -z "$HEAD_SHA" ]; then
  GATE_FAIL="cannot resolve HEAD"
elif ! printf '%s\n' "$EXPECT_SHA" | grep -qE '^[0-9a-f]{40}$'; then
  GATE_FAIL="EXPECT_SHA '${EXPECT_SHA}' is not a full lowercase 40-hex commit id"
elif [ "$(git rev-parse --verify -q "${EXPECT_SHA}^{commit}" 2>/dev/null)" != "$EXPECT_SHA" ]; then
  GATE_FAIL="EXPECT_SHA ${EXPECT_SHA} is not a commit known to this repo"
elif [ "$HEAD_SHA" = "$EXPECT_SHA" ]; then
  GATE_OK_MSG="commit binding OK: ${HEAD_SHA}"
elif SD="$(surface_diff_vs_expect 2>&1)"; then
  if [ -z "$SD" ]; then
    GATE_OK_MSG="commit binding OK (content): training surfaces identical, EXPECT_SHA ${EXPECT_SHA} HEAD ${HEAD_SHA}"
  else
    GATE_FAIL="training surfaces changed since EXPECT_SHA ${EXPECT_SHA} (HEAD ${HEAD_SHA}): ${SD}"
  fi
else
  GATE_FAIL="surface diff vs EXPECT_SHA failed: ${SD}"
fi
if [ -z "$GATE_FAIL" ] && [ "$(git rev-parse HEAD 2>/dev/null)" != "$HEAD_SHA" ]; then
  GATE_FAIL="HEAD moved during the commit-binding check (was ${HEAD_SHA})"
fi
if [ "$DRYRUN" = "1" ]; then
  [ -z "$GATE_FAIL" ] && echo "${GATE_OK_MSG} (dry run)" \
    || echo "DRY-RUN ADVISORY: ${GATE_FAIL} (a real launch aborts here)"
  [ -z "$DRIFT" ] || echo "DRY-RUN ADVISORY: tracked measurement surfaces are modified (a real launch aborts here)"
else
  [ -n "${SLURM_JOB_ID:-}" ] || die "a real launch must run under sbatch (no SLURM_JOB_ID) - abort"
  [ -z "$GATE_FAIL" ] || die "${GATE_FAIL} - abort"
  echo "$GATE_OK_MSG"
  [ -z "$DRIFT" ] || { echo "tracked measurement surfaces modified since review - abort:"; echo "$DRIFT"; exit 2; }
fi

# --- D. arm -> config (single source) + semantic gate -------------------------
arm_config_for() {
  case "$1" in
    C4L|C8|C16|C32) echo "$EXPDIR/FLAC_AR_BF_$1.json" ;;
    VANL)           echo "$EXPDIR/FLAC_AR_VANCKPT.json" ;;
    *) return 1 ;;
  esac
}
MODEL_CONFIG="$(arm_config_for "$ARM")" || die "no config mapped for arm '${ARM}' - abort"
MODEL_CONFIG_ABS="$(readlink -f "$MODEL_CONFIG" 2>/dev/null)"
[ -n "$MODEL_CONFIG_ABS" ] && [ -f "$MODEL_CONFIG_ABS" ] || die "arm config '${MODEL_CONFIG}' does not exist - abort"
CONFIG_SHA="$(sha256sum "$MODEL_CONFIG_ABS" | awk '{print $1}')"
echo "config for ${ARM}: ${MODEL_CONFIG_ABS} sha256 ${CONFIG_SHA}"

python3 - "$MODEL_CONFIG_ABS" "$ARM" <<'PY' || die "arm/config semantic gate FAILED - abort"
import json, sys
cfg = json.load(open(sys.argv[1])); arm = sys.argv[2]
t = cfg.get("training", {}); bad = []
# VANL is the same recipe with the conditioning removed, so its gate is the
# MIRROR IMAGE of the orbit arms': the orbit keys must be ABSENT, not merely
# different. A vanilla config that carried a stray frame_avg_angles would be a
# silently fa-flavoured baseline, which would destroy the single-delta claim.
if arm == "VANL":
    cm = t.get("cond_method")
    if cm not in (None, "vanilla"):
        bad.append(f"cond_method={cm!r} (want absent or 'vanilla')")
    if "frame_avg_angles" in t:
        bad.append(f"frame_avg_angles is present ({t['frame_avg_angles']!r}) — a vanilla arm has no orbit")
    want = None
else:
    want = {"C4L": 4, "C8": 8, "C16": 16, "C32": 32}[arm]
    angles = t.get("frame_avg_angles")
    if t.get("cond_method") != "fa_invariant":
        bad.append(f"cond_method={t.get('cond_method')!r} (want fa_invariant)")
    if not isinstance(angles, list) or len(angles) != want:
        bad.append(f"frame_avg_angles has {angles and len(angles)} entries (want {want})")
    elif angles != [k * 360.0 / want for k in range(want)]:
        bad.append(f"frame_avg_angles are not the uniform C{want} orbit")
if t.get("use_ema") is not True:
    bad.append(f"use_ema={t.get('use_ema')!r} (want True)")
vits = [c for c in cfg["model"]["conditioning"]["configs"] if c["type"] == "ViTCoordinates"]
if sorted(c["id"] for c in vits) != ["context_poses_vit", "source_vit"]:
    bad.append(f"ViT conditioner ids {sorted(c['id'] for c in vits)} != the expected two")
# Post-P0: grad-ckpt ON for every arm; the KEY must exist and be literally True
for c in vits:
    if "gradient_checkpointing" not in c["config"]:
        bad.append(f"{c['id']}: gradient_checkpointing key absent (want literal true)")
    elif c["config"]["gradient_checkpointing"] is not True:
        bad.append(f"{c['id']}: gradient_checkpointing={c['config']['gradient_checkpointing']!r} (want True)")
if bad:

exec
/usr/bin/bash -lc 'git --version && git status --short --branch' in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
git version 2.52.0
## check-equivariance-necessity...origin/check-equivariance-necessity [ahead 15]
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
 M worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
 M worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_command.md
 M worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_worklog.md
 M worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_worklog.md
 M worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_yixun_query.md
?? AGENTS.md
?? AcousticRooms
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_15-18-49_C32_screen_S2500_s42_K1_jid3662408_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_15-18-49_C32_screen_S2500_s42_K8_jid3662407_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_15-19-48_C32_screen_S22500_s42_K8_jid3662416_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_15-19-49_C32_screen_S12500_s42_K1_jid3662413_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_15-19-49_C32_screen_S12500_s42_K8_jid3662412_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_15-19-49_C32_screen_S17500_s42_K1_jid3662415_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_15-19-49_C32_screen_S17500_s42_K8_jid3662414_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_15-19-49_C32_screen_S7500_s42_K1_jid3662411_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_15-19-49_C32_screen_S7500_s42_K8_jid3662410_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_15-20-48_C32_screen_S22500_s42_K1_jid3662417_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_15-20-49_C32_screen_S27500_s42_K1_jid3662419_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_15-20-49_C32_screen_S27500_s42_K8_jid3662418_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_15-20-49_C32_screen_S30000_s42_K1_jid3662421_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_15-20-49_C32_screen_S30000_s42_K8_jid3662420_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_17-06-16_screen_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_17-10-17_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_17-16-44_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_17-24-24_screen_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_20-20-54_C32_screen_S32500_s42_K1_jid3665920_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-09_20-20-54_C32_screen_S32500_s42_K8_jid3665919_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_01-27-05_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_01-31-51_C32_screen_S35000_s42_K8_jid3668010_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_01-39-54_C32_screen_S35000_s42_K1_jid3668011_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_01-40-02_screen_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_02-00-01_screen_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_06-47-34_C32_screen_S37500_s42_K8_jid3668648_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_06-48-34_C32_screen_S37500_s42_K1_jid3668649_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_11-57-38_C32_screen_S40000_s42_K1_jid3670799_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_11-57-38_C32_screen_S40000_s42_K8_jid3670798_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_18-20-00_C32_conf_S40000_s42_K8_jid3672838_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_18-23-02_C32_conf_S40000_s42_K1_jid3672839_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_18-23-02_C32_conf_S40000_s43_K8_jid3672840_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_18-25-01_C32_conf_S40000_s43_K1_jid3672841_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_18-26-02_C32_conf_S40000_s44_K1_jid3672843_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_18-26-02_C32_conf_S40000_s44_K8_jid3672842_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_18-29-03_C32_conf_S40000_s45_K8_jid3672844_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_18-37-06_C32_conf_S40000_s45_K1_jid3672845_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_18-41-06_C32_conf_S40000_s46_K8_jid3672846_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-10_18-41-07_C32_conf_S40000_s46_K1_jid3672847_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_00-46-17_VANL_screen_S2500_s42_K8_jid3674679_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_00-50-18_VANL_q9_S40000_s42_K8_jid3674658_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_00-56-20_VANL_q9_S40000_s42_K1_jid3674659_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_00-59-20_VANL_q9_S40000_s44_K8_jid3674662_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_00-59-21_VANL_q9_S40000_s43_K1_jid3674661_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_00-59-21_VANL_q9_S40000_s43_K8_jid3674660_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-03-23_VANL_q9_S40000_s44_K1_jid3674663_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-03-23_VANL_q9_S40000_s45_K8_jid3674664_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-04-23_VANL_screen_S2500_s42_K1_jid3674680_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-05-23_VANL_q9_S40000_s45_K1_jid3674665_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-05-23_VANL_q9_S40000_s46_K8_jid3674666_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-06-24_VANL_screen_S5000_s42_K8_jid3674681_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-07-23_C4L_q9_S40000_s43_K1_jid3674671_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-07-23_C4L_q9_S40000_s43_K8_jid3674670_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-07-23_C4L_q9_S40000_s44_K1_jid3674673_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-07-24_C4L_q9_S40000_s42_K8_jid3674668_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-07-24_C4L_q9_S40000_s44_K8_jid3674672_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-07-24_VANL_q9_S40000_s46_K1_jid3674667_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-07-25_C4L_q9_S40000_s42_K1_jid3674669_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-08-23_VANL_screen_S10000_s42_K8_jid3674685_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-08-23_VANL_screen_S7500_s42_K1_jid3674684_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-08-24_C4L_q9_S40000_s45_K1_jid3674675_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-08-24_C4L_q9_S40000_s45_K8_jid3674674_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-08-24_C4L_q9_S40000_s46_K1_jid3674677_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-08-24_VANL_screen_S5000_s42_K1_jid3674682_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-08-24_VANL_screen_S7500_s42_K8_jid3674683_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-08-25_C4L_q9_S40000_s46_K8_jid3674676_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-09-24_VANL_screen_S10000_s42_K1_jid3674686_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-10-25_VANL_screen_S12500_s42_K1_jid3674688_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-10-25_VANL_screen_S12500_s42_K8_jid3674687_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-10-25_VANL_screen_S15000_s42_K8_jid3674689_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-11-25_VANL_screen_S15000_s42_K1_jid3674690_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-12-24_VANL_screen_S17500_s42_K1_jid3674692_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-12-25_VANL_screen_S17500_s42_K8_jid3674691_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-13-25_VANL_screen_S20000_s42_K8_jid3674693_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-14-26_VANL_screen_S20000_s42_K1_jid3674694_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-14-26_VANL_screen_S22500_s42_K1_jid3674696_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-14-26_VANL_screen_S22500_s42_K8_jid3674695_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-14-26_VANL_screen_S25000_s42_K8_jid3674697_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-16-27_VANL_screen_S25000_s42_K1_jid3674698_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-16-27_VANL_screen_S27500_s42_K1_jid3674700_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-16-27_VANL_screen_S27500_s42_K8_jid3674699_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-17-27_VANL_screen_S30000_s42_K1_jid3674702_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-17-27_VANL_screen_S30000_s42_K8_jid3674701_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-17-27_VANL_screen_S32500_s42_K1_jid3674704_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-17-27_VANL_screen_S32500_s42_K8_jid3674703_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-17-27_VANL_screen_S35000_s42_K1_jid3674706_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-17-27_VANL_screen_S35000_s42_K8_jid3674705_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-17-27_VANL_screen_S37500_s42_K8_jid3674707_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-18-26_VANL_screen_S40000_s42_K8_jid3674709_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-18-27_VANL_screen_S37500_s42_K1_jid3674708_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_01-18-27_VANL_screen_S40000_s42_K1_jid3674710_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_14-55-12_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_14-57-42_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_15-15-45_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_18-31-38_C4L_r3_rot0_S40000_s42_K8_jid3680738_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_18-31-38_C4L_r3_rot5p625_S40000_s42_K8_jid3680739_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_18-32-38_C4L_r3_rot11p25_S40000_s42_K8_jid3680740_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_18-33-38_C4L_r3_rot22p5_S40000_s42_K8_jid3680741_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_18-33-38_C4L_r3_rot45_S40000_s42_K8_jid3680742_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_18-34-38_C8_r3_rot0_S40000_s42_K8_jid3680743_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_18-35-38_C8_r3_rot5p625_S40000_s42_K8_jid3680744_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_18-35-39_C8_r3_rot11p25_S40000_s42_K8_jid3680745_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_18-36-40_C16_r3_rot0_S40000_s42_K8_jid3680748_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_18-36-40_C16_r3_rot5p625_S40000_s42_K8_jid3680749_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_18-36-40_C8_r3_rot22p5_S40000_s42_K8_jid3680746_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_18-36-41_C8_r3_rot45_S40000_s42_K8_jid3680747_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_18-37-40_C16_r3_rot11p25_S40000_s42_K8_jid3680750_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_18-39-41_C16_r3_rot22p5_S40000_s42_K8_jid3680751_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_18-43-43_C16_r3_rot45_S40000_s42_K8_jid3680752_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_18-43-43_C32_r3_rot0_S40000_s42_K8_jid3680753_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_18-43-43_C32_r3_rot5p625_S40000_s42_K8_jid3680754_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_18-44-42_C32_r3_rot11p25_S40000_s42_K8_jid3680755_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_18-44-42_C32_r3_rot22p5_S40000_s42_K8_jid3680756_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_18-44-43_C32_r3_rot45_S40000_s42_K8_jid3680757_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_18-45-42_C4L_cross_a32_S40000_s42_K8_jid3680760_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_18-45-42_C8_cross_a4_S40000_s42_K8_jid3680761_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_18-45-43_C4L_cross_a16_S40000_s42_K8_jid3680759_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_18-45-43_C4L_cross_a8_S40000_s42_K8_jid3680758_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_18-46-44_C16_cross_a4_S40000_s42_K8_jid3680762_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_18-47-44_C32_cross_a4_S40000_s42_K8_jid3680763_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_18-47-44_C4BACKFILL_cross_a8_S40000_s42_K8_jid3680764_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_18-49-44_C4BACKFILL_cross_a16_S40000_s42_K8_jid3680765_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-11_18-49-45_C4BACKFILL_cross_a32_S40000_s42_K8_jid3680766_screen.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-12_06-45-52_guardtests.log
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_codex_content_gate_review_r3.md
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_1786310422371467848-a776b47c.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_cross_S40000_s42_K8_jid3680762.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_r3_S40000_s42_K8_jid3680748.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_r3_S40000_s42_K8_jid3680749.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_r3_S40000_s42_K8_jid3680750.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_r3_S40000_s42_K8_jid3680751.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C16_r3_S40000_s42_K8_jid3680752.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_conf_S40000_s42_K1_jid3672839.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_conf_S40000_s42_K8_jid3672838.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_conf_S40000_s43_K1_jid3672841.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_conf_S40000_s43_K8_jid3672840.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_conf_S40000_s44_K1_jid3672843.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_conf_S40000_s44_K8_jid3672842.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_conf_S40000_s45_K1_jid3672845.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_conf_S40000_s45_K8_jid3672844.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_conf_S40000_s46_K1_jid3672847.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_conf_S40000_s46_K8_jid3672846.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_cross_S40000_s42_K8_jid3680763.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_r3_S40000_s42_K8_jid3680753.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_r3_S40000_s42_K8_jid3680754.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_r3_S40000_s42_K8_jid3680755.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_r3_S40000_s42_K8_jid3680756.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_r3_S40000_s42_K8_jid3680757.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S12500_s42_K1_jid3662413.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S12500_s42_K8_jid3662412.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S17500_s42_K1_jid3662415.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S17500_s42_K8_jid3662414.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S22500_s42_K1_jid3662417.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S22500_s42_K8_jid3662416.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S2500_s42_K1_jid3662408.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S2500_s42_K8_jid3662407.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S27500_s42_K1_jid3662419.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S27500_s42_K8_jid3662418.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S30000_s42_K1_jid3662421.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S30000_s42_K8_jid3662420.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S32500_s42_K1_jid3665920.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S32500_s42_K8_jid3665919.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S35000_s42_K1_jid3668011.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S35000_s42_K8_jid3668010.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S37500_s42_K1_jid3668649.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S37500_s42_K8_jid3668648.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S40000_s42_K1_jid3670799.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S40000_s42_K8_jid3670798.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S7500_s42_K1_jid3662411.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C32_screen_S7500_s42_K8_jid3662410.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4BACKFILL_cross_S40000_s42_K8_jid3680764.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4BACKFILL_cross_S40000_s42_K8_jid3680765.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4BACKFILL_cross_S40000_s42_K8_jid3680766.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_1786310422143759413-7d512809.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_cross_S40000_s42_K8_jid3680758.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_cross_S40000_s42_K8_jid3680759.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_cross_S40000_s42_K8_jid3680760.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_q9_S40000_s42_K1_jid3674669.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_q9_S40000_s42_K8_jid3674668.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_q9_S40000_s43_K1_jid3674671.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_q9_S40000_s43_K8_jid3674670.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_q9_S40000_s44_K1_jid3674673.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_q9_S40000_s44_K8_jid3674672.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_q9_S40000_s45_K1_jid3674675.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_q9_S40000_s45_K8_jid3674674.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_q9_S40000_s46_K1_jid3674677.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_q9_S40000_s46_K8_jid3674676.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_r3_S40000_s42_K8_jid3680738.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_r3_S40000_s42_K8_jid3680739.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_r3_S40000_s42_K8_jid3680740.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_r3_S40000_s42_K8_jid3680741.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C4L_r3_S40000_s42_K8_jid3680742.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_1786310422260085470-2e58ce21.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_cross_S40000_s42_K8_jid3680761.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_r3_S40000_s42_K8_jid3680743.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_r3_S40000_s42_K8_jid3680744.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_r3_S40000_s42_K8_jid3680745.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_r3_S40000_s42_K8_jid3680746.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_C8_r3_S40000_s42_K8_jid3680747.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_1786473966640260607-09fab791.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_q9_S40000_s42_K1_jid3674659.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_q9_S40000_s42_K8_jid3674658.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_q9_S40000_s43_K1_jid3674661.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_q9_S40000_s43_K8_jid3674660.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_q9_S40000_s44_K1_jid3674663.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_q9_S40000_s44_K8_jid3674662.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_q9_S40000_s45_K1_jid3674665.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_q9_S40000_s45_K8_jid3674664.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_q9_S40000_s46_K1_jid3674667.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_q9_S40000_s46_K8_jid3674666.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S10000_s42_K1_jid3662406.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S10000_s42_K1_jid3662812.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S10000_s42_K1_jid3674686.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S10000_s42_K8_jid3662405.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S10000_s42_K8_jid3662811.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S10000_s42_K8_jid3674685.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S12500_s42_K1_jid3662814.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S12500_s42_K1_jid3674688.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S12500_s42_K8_jid3662813.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S12500_s42_K8_jid3674687.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S15000_s42_K1_jid3662816.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S15000_s42_K1_jid3674690.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S15000_s42_K8_jid3662815.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S15000_s42_K8_jid3674689.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S17500_s42_K1_jid3662818.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S17500_s42_K1_jid3674692.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S17500_s42_K8_jid3662817.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S17500_s42_K8_jid3674691.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S20000_s42_K1_jid3674694.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S20000_s42_K8_jid3674693.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S22500_s42_K1_jid3674696.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S22500_s42_K8_jid3674695.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S25000_s42_K1_jid3674698.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S25000_s42_K8_jid3674697.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S2500_s42_K1_jid3662400.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S2500_s42_K1_jid3662806.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S2500_s42_K1_jid3674680.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S2500_s42_K8_jid3662399.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S2500_s42_K8_jid3662805.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S2500_s42_K8_jid3674679.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S27500_s42_K1_jid3674700.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S27500_s42_K8_jid3674699.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S30000_s42_K1_jid3674702.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S30000_s42_K8_jid3674701.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S32500_s42_K1_jid3674704.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S32500_s42_K8_jid3674703.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S35000_s42_K1_jid3674706.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S35000_s42_K8_jid3674705.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S37500_s42_K1_jid3674708.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S37500_s42_K8_jid3674707.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S40000_s42_K1_jid3674710.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S40000_s42_K8_jid3674709.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S5000_s42_K1_jid3662402.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S5000_s42_K1_jid3662808.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S5000_s42_K1_jid3674682.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S5000_s42_K8_jid3662401.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S5000_s42_K8_jid3662807.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S5000_s42_K8_jid3674681.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S7500_s42_K1_jid3662404.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S7500_s42_K1_jid3662810.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S7500_s42_K1_jid3674684.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S7500_s42_K8_jid3662403.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S7500_s42_K8_jid3662809.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submission_VANL_screen_S7500_s42_K8_jid3674683.txt
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C16-cross-40000-s42-K8_3680762.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C16-r3-40000-s42-K8_3680748.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C16-r3-40000-s42-K8_3680749.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C16-r3-40000-s42-K8_3680750.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C16-r3-40000-s42-K8_3680751.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C16-r3-40000-s42-K8_3680752.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-conf-40000-s42-K1_3672839.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-conf-40000-s42-K8_3672838.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-conf-40000-s43-K1_3672841.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-conf-40000-s43-K8_3672840.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-conf-40000-s44-K1_3672843.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-conf-40000-s44-K8_3672842.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-conf-40000-s45-K1_3672845.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-conf-40000-s45-K8_3672844.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-conf-40000-s46-K1_3672847.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-conf-40000-s46-K8_3672846.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-cross-40000-s42-K8_3680763.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-r3-40000-s42-K8_3680753.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-r3-40000-s42-K8_3680754.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-r3-40000-s42-K8_3680755.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-r3-40000-s42-K8_3680756.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-r3-40000-s42-K8_3680757.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-12500-s42-K1_3662413.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-12500-s42-K8_3662412.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-17500-s42-K1_3662415.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-17500-s42-K8_3662414.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-22500-s42-K1_3662417.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-22500-s42-K8_3662416.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-2500-s42-K1_3662408.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-2500-s42-K8_3662407.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-27500-s42-K1_3662419.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-27500-s42-K8_3662418.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-30000-s42-K1_3662421.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-30000-s42-K8_3662420.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-32500-s42-K1_3665920.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-32500-s42-K8_3665919.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-35000-s42-K1_3668011.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-35000-s42-K8_3668010.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-37500-s42-K1_3668649.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-37500-s42-K8_3668648.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-40000-s42-K1_3670799.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-40000-s42-K8_3670798.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-7500-s42-K1_3662411.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C32-screen-7500-s42-K8_3662410.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C4BACKFILL-cross-40000-s42-K8_3680764.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C4BACKFILL-cross-40000-s42-K8_3680765.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C4BACKFILL-cross-40000-s42-K8_3680766.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C4L-cross-40000-s42-K8_3680758.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C4L-cross-40000-s42-K8_3680759.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C4L-cross-40000-s42-K8_3680760.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C4L-q9-40000-s42-K1_3674669.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C4L-q9-40000-s42-K8_3674668.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C4L-q9-40000-s43-K1_3674671.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C4L-q9-40000-s43-K8_3674670.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C4L-q9-40000-s44-K1_3674673.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C4L-q9-40000-s44-K8_3674672.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C4L-q9-40000-s45-K1_3674675.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C4L-q9-40000-s45-K8_3674674.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C4L-q9-40000-s46-K1_3674677.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C4L-q9-40000-s46-K8_3674676.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C4L-r3-40000-s42-K8_3680738.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C4L-r3-40000-s42-K8_3680739.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C4L-r3-40000-s42-K8_3680740.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C4L-r3-40000-s42-K8_3680741.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C4L-r3-40000-s42-K8_3680742.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C8-cross-40000-s42-K8_3680761.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C8-r3-40000-s42-K8_3680743.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C8-r3-40000-s42-K8_3680744.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C8-r3-40000-s42-K8_3680745.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C8-r3-40000-s42-K8_3680746.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-C8-r3-40000-s42-K8_3680747.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-q9-40000-s42-K1_3674659.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-q9-40000-s42-K8_3674658.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-q9-40000-s43-K1_3674661.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-q9-40000-s43-K8_3674660.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-q9-40000-s44-K1_3674663.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-q9-40000-s44-K8_3674662.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-q9-40000-s45-K1_3674665.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-q9-40000-s45-K8_3674664.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-q9-40000-s46-K1_3674667.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-q9-40000-s46-K8_3674666.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-10000-s42-K1_3662406.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-10000-s42-K1_3674686.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-10000-s42-K8_3662405.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-10000-s42-K8_3674685.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-12500-s42-K1_3674688.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-12500-s42-K8_3674687.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-15000-s42-K1_3674690.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-15000-s42-K8_3674689.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-17500-s42-K1_3674692.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-17500-s42-K8_3674691.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-20000-s42-K1_3674694.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-20000-s42-K8_3674693.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-22500-s42-K1_3674696.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-22500-s42-K8_3674695.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-2500-s42-K1_3662400.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-2500-s42-K1_3662806.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-2500-s42-K1_3674680.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-2500-s42-K8_3662399.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-2500-s42-K8_3662805.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-2500-s42-K8_3674679.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-25000-s42-K1_3674698.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-25000-s42-K8_3674697.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-27500-s42-K1_3674700.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-27500-s42-K8_3674699.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-30000-s42-K1_3674702.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-30000-s42-K8_3674701.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-32500-s42-K1_3674704.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-32500-s42-K8_3674703.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-35000-s42-K1_3674706.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-35000-s42-K8_3674705.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-37500-s42-K1_3674708.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-37500-s42-K8_3674707.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-40000-s42-K1_3674710.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-40000-s42-K8_3674709.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-5000-s42-K1_3662402.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-5000-s42-K1_3662808.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-5000-s42-K1_3674682.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-5000-s42-K8_3662401.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-5000-s42-K8_3662807.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-5000-s42-K8_3674681.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-7500-s42-K1_3662404.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-7500-s42-K1_3674684.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-7500-s42-K8_3662403.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-7500-s42-K8_3662809.out
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_exp11-screen-VANL-screen-7500-s42-K8_3674683.out
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/rung1_pin16fe13d_archive/
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/slurm_screen_exp14-screen-C16-vctl-rot90-40000-s42-K8_3683162.out
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/slurm_screen_exp14-screen-C32-rgen-rotrand42-40000-s42-K8_3683585.out
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/slurm_screen_exp14-screen-C32-vctl-rot90-40000-s42-K8_3683163.out
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/slurm_screen_exp14-screen-C4L-vctl-rot45-40000-s42-K8_3683165.out
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/slurm_screen_exp14-screen-C4L-vctl-rot90-40000-s42-K8_3683160.out
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/slurm_screen_exp14-screen-C8-vctl-rot90-40000-s42-K8_3683161.out
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/slurm_screen_exp14-screen-VANL-vctl-rot90-40000-s42-K8_3683164.out
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/slurm_screen_exp14-screen-VANL-zref-40000-s42-K1_3683894.out
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/slurm_screen_exp14-screen-VANL-zref-40000-s43-K1_3683895.out
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/slurm_screen_exp14-screen-VANL-zref-40000-s44-K1_3683896.out
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/slurm_screen_exp14-screen-VANL-zref-40000-s45-K1_3683897.out
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/slurm_screen_exp14-screen-VANL-zref-40000-s46-K1_3683898.out
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-12_01-23-35_C4L_vctl_rot90_S40000_s42_K8_jid3683160_screen.log
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-12_02-35-11_C8_vctl_rot90_S40000_s42_K8_jid3683161_screen.log
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-12_02-36-12_C16_vctl_rot90_S40000_s42_K8_jid3683162_screen.log
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-12_02-57-22_C32_vctl_rot90_S40000_s42_K8_jid3683163_screen.log
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-12_03-05-27_VANL_vctl_rot90_S40000_s42_K8_jid3683164_screen.log
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-12_03-19-35_C4L_vctl_rot45_S40000_s42_K8_jid3683165_screen.log
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-12_04-29-07_C32_rgen_rotrand42_S40000_s42_K8_jid3683585_screen.log
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-12_05-48-44_VANL_zref_S40000_s42_K1_jid3683894_screen.log
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-12_06-05-51_VANL_zref_S40000_s43_K1_jid3683895_screen.log
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-12_06-17-57_VANL_zref_S40000_s44_K1_jid3683896_screen.log
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-12_06-30-02_VANL_zref_S40000_s45_K1_jid3683897_screen.log
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-12_06-36-05_VANL_zref_S40000_s46_K1_jid3683898_screen.log
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submission_C16_vctl_S40000_s42_K8_jid3683162.txt
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submission_C32_rgen_S40000_s42_K8_jid3683585.txt
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submission_C32_vctl_S40000_s42_K8_jid3683163.txt
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submission_C4L_vctl_S40000_s42_K8_jid3683160.txt
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submission_C4L_vctl_S40000_s42_K8_jid3683165.txt
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submission_C4L_zref_S40000_s42_K1_jid3683906.txt
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submission_C4L_zref_S40000_s42_K8_jid3683911.txt
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submission_C4L_zref_S40000_s43_K1_jid3683907.txt
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submission_C4L_zref_S40000_s43_K8_jid3684053.txt
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submission_C4L_zref_S40000_s44_K1_jid3683908.txt
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submission_C4L_zref_S40000_s44_K8_jid3684067.txt
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submission_C4L_zref_S40000_s45_K1_jid3683909.txt
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submission_C4L_zref_S40000_s45_K8_jid3684084.txt
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submission_C4L_zref_S40000_s46_K1_jid3683910.txt
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submission_C4L_zref_S40000_s46_K8_jid3684116.txt
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submission_C8_vctl_S40000_s42_K8_jid3683161.txt
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submission_VANL_vctl_S40000_s42_K8_jid3683164.txt
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submission_VANL_zref_S40000_s42_K1_jid3683894.txt
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submission_VANL_zref_S40000_s42_K8_jid3683901.txt
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submission_VANL_zref_S40000_s43_K1_jid3683895.txt
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submission_VANL_zref_S40000_s43_K8_jid3683902.txt
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submission_VANL_zref_S40000_s44_K1_jid3683896.txt
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submission_VANL_zref_S40000_s44_K8_jid3683903.txt
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submission_VANL_zref_S40000_s45_K1_jid3683897.txt
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submission_VANL_zref_S40000_s45_K8_jid3683904.txt
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submission_VANL_zref_S40000_s46_K1_jid3683898.txt
?? worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submission_VANL_zref_S40000_s46_K8_jid3683905.txt

codex
The patch is narrowly limited to the two exclusions and one guard case. The host is on Git 2.52.0; I’m now checking the exclusion against actual repository history/status and tracing any route from `train.py` or package initializers into `src/tests`, including dynamic-import patterns.
exec
/usr/bin/bash -lc "sed -n '1,320p' train.py" in /n/fs/gatrdp/codespace/FLAC
exec
/usr/bin/bash -lc 'git status --porcelain --untracked-files=no -- src/tests' in /n/fs/gatrdp/codespace/FLAC
exec
/usr/bin/bash -lc 'find src/tests -maxdepth 3 -type f -print | sort' in /n/fs/gatrdp/codespace/FLAC
exec
/usr/bin/bash -lc 'git status --porcelain --untracked-files=no -- train.py defaults.ini src ":(exclude)src/tests" data/AR' in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
 succeeded in 0ms:
import torch
import json
import os
import pytorch_lightning as pl

from prefigure.prefigure import get_all_args, push_wandb_config
from src.data.dataset import create_dataloader_from_config
from src.models import create_model_from_config
from src.models.utils import load_ckpt_state_dict, remove_weight_norm_from_model
from src.training import create_training_wrapper_from_config

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')

class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config

def _as_bool(value):
    """Coerce a prefigure-parsed flag to a genuine ``bool``.

    prefigure parses the lowercase ini literal ``false``/``true`` as the *string*
    "false"/"true" (ast.literal_eval rejects lowercase, so the flag is registered
    ``type=str``); only the capitalized ``False``/``True`` yield a real bool. Accept
    str or bool so a forwarded Trainer kwarg is always a genuine boolean.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off", ""):
            return False
        raise ValueError(f"cannot interpret sync_batchnorm={value!r} as a boolean")
    raise TypeError(f"sync_batchnorm must be bool or str, got {type(value).__name__}")

def build_trainer_kwargs(args, strategy, callbacks, logger, checkpoint_dir, val_args):
    """Assemble the pl.Trainer keyword arguments (side-effect free; unit-testable).

    Reproduces the kwargs that were previously inlined into pl.Trainer(...) exactly;
    the only behavioral change is that max_steps is now sourced from args.max_steps
    (default 1000000 via defaults.ini) instead of a hard-coded literal, so a training
    budget can be set without editing code. strategy / callbacks / logger /
    checkpoint_dir / val_args are the values main() derives and passes straight through.

    sync_batchnorm (default off -> key ABSENT, so the kwargs are byte-identical to the
    pre-change dict and PL's own default False applies) forwards to
    Trainer(sync_batchnorm=True) only when enabled. It is a multi-GPU-only feature, so
    enabling it with num_gpus < 2 is a fail-closed ValueError (Yixun mandate) rather than
    a silently-ignored no-op. The guard lives here so both construct_trainer/main() and
    any direct caller hit it; val_args may NOT smuggle the key past the guard.
    """
    if "sync_batchnorm" in val_args:
        raise ValueError("sync_batchnorm must come from args (guarded), not val_args")
    sync_batchnorm = _as_bool(getattr(args, "sync_batchnorm", False))
    if sync_batchnorm and args.num_gpus < 2:
        raise ValueError(
            "sync_batchnorm=True requires multi-GPU training (num_gpus >= 2); got "
            f"num_gpus={args.num_gpus}. SyncBatchNorm synchronises BatchNorm statistics "
            "across ranks and is a no-op / unsupported on a single device -- set "
            "--num-gpus >= 2 or drop --sync-batchnorm."
        )
    kwargs = {
        "devices": args.num_gpus,
        "accelerator": "gpu",
        "num_nodes": args.num_nodes,
        "strategy": strategy,
        "precision": args.precision,
        "accumulate_grad_batches": args.accum_batches,
        "callbacks": callbacks,
        "logger": logger,
        "log_every_n_steps": 100,
        "max_steps": args.max_steps, # HAA finetune recipe: --max-steps 1000
        "default_root_dir": checkpoint_dir,
        "gradient_clip_val": args.gradient_clip_val,
        "reload_dataloaders_every_n_epochs": 0,
        "num_sanity_val_steps": 0, # If you need to debug validation, change this line
        **val_args,
    }
    if sync_batchnorm:
        kwargs["sync_batchnorm"] = True  # multi-GPU only; guarded above (fail-closed)
    return kwargs

def construct_trainer(args, strategy, callbacks, logger, checkpoint_dir, val_args):
    """Construct the pl.Trainer from the assembled kwargs (the tested Trainer boundary)."""
    return pl.Trainer(**build_trainer_kwargs(args, strategy, callbacks, logger, checkpoint_dir, val_args))

def main():
    torch.set_float32_matmul_precision('medium') 
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = get_all_args()
    seed = args.seed

    # Set a different seed for each process if using SLURM
    if os.environ.get("SLURM_PROCID") is not None:
        seed += int(os.environ.get("SLURM_PROCID"))

    pl.seed_everything(seed, workers=True)

    #Get JSON config from args.model_config
    with open(args.model_config) as f:
        model_config = json.load(f)

    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    train_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 1),
    )

    val_dl = None
    val_dataset_config = None
    if args.val_dataset_config:
        with open(args.val_dataset_config) as f:
            val_dataset_config = json.load(f)

        val_dl = create_dataloader_from_config(
            val_dataset_config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sample_rate=model_config["sample_rate"],
            sample_size=model_config["sample_size"],
            audio_channels=model_config.get("audio_channels", 1),
            shuffle=False
        )

    model = create_model_from_config(model_config)

    if args.pretrained_ckpt_path:
        print('Loading pretrained model...')
        weights = load_ckpt_state_dict(args.pretrained_ckpt_path)
        weights = {k.replace('diffusion.', ''): v for k, v in weights.items()} # For diffusion
        weights = {k.replace('autoencoder.', ''): v for k, v in weights.items()} # For VAE
        disc_weights = {k: v for k, v in weights.items() if 'discriminator' in k}
        disc_weights = {k.replace('discriminator.', ''): v for k, v in disc_weights.items()}
        weights = {k: v for k, v in weights.items() if 'discriminator' not in k}
        weights = {k: v for k, v in weights.items() if 'losses' not in k}
        model.load_state_dict(weights, strict=True)

    if args.remove_pretransform_weight_norm == "pre_load":
        remove_weight_norm_from_model(model.pretransform)

    if args.pretransform_ckpt_path:
        model.pretransform.load_state_dict(load_ckpt_state_dict(args.pretransform_ckpt_path))

    # Remove weight_norm from the pretransform if specified
    if args.remove_pretransform_weight_norm == "post_load":
        remove_weight_norm_from_model(model.pretransform)

    training_wrapper = create_training_wrapper_from_config(model_config, model)

    exc_callback = ExceptionCallback()

    if args.logger == 'wandb':
        logger = pl.loggers.WandbLogger(project=args.name, name=args.experiment_name)
        logger.watch(training_wrapper)
    
        if args.save_dir and isinstance(logger.experiment.id, str):
            checkpoint_dir = os.path.join(args.save_dir, logger.experiment.project, logger.experiment.name, "checkpoints") 
        else:
            checkpoint_dir = None
    elif args.logger == 'comet':
        logger = pl.loggers.CometLogger(project_name=args.name)
        if args.save_dir and isinstance(logger.version, str):
            checkpoint_dir = os.path.join(args.save_dir, logger.name, logger.version, "checkpoints") 
        else:
            checkpoint_dir = args.save_dir if args.save_dir else None
    else:
        logger = None
        checkpoint_dir = args.save_dir if args.save_dir else None
        
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, dirpath=checkpoint_dir, save_top_k=-1)
    save_model_config_callback = ModelConfigEmbedderCallback(model_config)
        
    #Combine args and config dicts
    args_dict = vars(args)
    args_dict.update({"model_config": model_config})
    args_dict.update({"dataset_config": dataset_config})
    args_dict.update({"val_dataset_config": val_dataset_config})

    # Logger 
    if args.logger == 'wandb':
        push_wandb_config(logger, args_dict)
    elif args.logger == 'comet':
        logger.log_hyperparams(args_dict)

    #Set multi-GPU strategy if specified
    if args.strategy:
        if args.strategy == "deepspeed":
            from pytorch_lightning.strategies import DeepSpeedStrategy
            strategy = DeepSpeedStrategy(stage=2,
                                        contiguous_gradients=True,
                                        overlap_comm=True,
                                        reduce_scatter=True,
                                        reduce_bucket_size=5e8,
                                        allgather_bucket_size=5e8,
                                        load_full_weights=True)
        else:
            strategy = args.strategy
    else:
        strategy = 'ddp_find_unused_parameters_true' if args.num_gpus > 1 else "auto"

    val_args = {}
    
    if args.val_every > 0:
        val_args.update({
            "check_val_every_n_epoch": None,
            "val_check_interval": args.val_every,
        })

    trainer = construct_trainer(
        args,
        strategy=strategy,
        callbacks=[ckpt_callback, exc_callback, save_model_config_callback],
        logger=logger,
        checkpoint_dir=checkpoint_dir,
        val_args=val_args,
    )

    trainer.fit(training_wrapper, train_dl, val_dl, ckpt_path=args.ckpt_path if args.ckpt_path else None)

if __name__ == '__main__':
    main()

 succeeded in 0ms:
 succeeded in 0ms:
src/tests/__pycache__/conftest.cpython-310-pytest-9.1.1.pyc
src/tests/__pycache__/conftest.cpython-311-pytest-7.4.0.pyc
src/tests/__pycache__/test_bn_drift_probe.cpython-310-pytest-9.1.1.pyc
src/tests/__pycache__/test_cond_dispatch.cpython-310-pytest-9.1.1.pyc
src/tests/__pycache__/test_cyl_vit_patch_embedding.cpython-310-pytest-9.1.1.pyc
src/tests/__pycache__/test_eval_paths.cpython-310-pytest-9.1.1.pyc
src/tests/__pycache__/test_exp11_equiv_probe.cpython-310-pytest-9.1.1.pyc
src/tests/__pycache__/test_exp11_orbit_configs.cpython-310-pytest-9.1.1.pyc
src/tests/__pycache__/test_exp11_orbit_configs.cpython-310.pyc
src/tests/__pycache__/test_exp11_p0_collect.cpython-310-pytest-9.1.1.pyc
src/tests/__pycache__/test_exp11_p0_runner.cpython-310-pytest-9.1.1.pyc
src/tests/__pycache__/test_exp11_restart_record.cpython-310-pytest-9.1.1.pyc
src/tests/__pycache__/test_exp11_restart_record.cpython-310.pyc
src/tests/__pycache__/test_exp11_trajectory_figures.cpython-310-pytest-9.1.1.pyc
src/tests/__pycache__/test_exp11_trajectory_figures.cpython-310.pyc
src/tests/__pycache__/test_exp11_validate_rows.cpython-310-pytest-9.1.1.pyc
src/tests/__pycache__/test_exp11_wandb_readback.cpython-310-pytest-9.1.1.pyc
src/tests/__pycache__/test_exp14_fixed_mode_snapshot.cpython-310-pytest-9.1.1.pyc
src/tests/__pycache__/test_exp14_fixed_mode_snapshot.cpython-310.pyc
src/tests/__pycache__/test_exp14_fixed_mode_snapshot.cpython-311-pytest-7.4.0.pyc
src/tests/__pycache__/test_exp14_validate_cell.cpython-310-pytest-9.1.1.pyc
src/tests/__pycache__/test_exp14_validate_cell.cpython-310.pyc
src/tests/__pycache__/test_exp14_validate_cell.cpython-311-pytest-7.4.0.pyc
src/tests/__pycache__/test_finetune_cond.cpython-310-pytest-9.1.1.pyc
src/tests/__pycache__/test_gen_model_comparison_gate.cpython-310-pytest-9.1.1.pyc
src/tests/__pycache__/test_gen_model_comparison_gate.cpython-310.pyc
src/tests/__pycache__/test_invariant_conditioning.cpython-310-pytest-9.1.1.pyc
src/tests/__pycache__/test_invariant_conditioning.cpython-310.pyc
src/tests/__pycache__/test_phase3_vit_training.cpython-310-pytest-9.1.1.pyc
src/tests/__pycache__/test_retune_lr_state.cpython-310-pytest-9.1.1.pyc
src/tests/__pycache__/test_strip_optimizer_state.cpython-310-pytest-9.1.1.pyc
src/tests/__pycache__/test_train_max_steps.cpython-310-pytest-9.1.1.pyc
src/tests/__pycache__/test_train_sync_batchnorm.cpython-310-pytest-9.1.1.pyc
src/tests/__pycache__/test_vit_gradient_checkpointing.cpython-310-pytest-9.1.1.pyc
src/tests/__pycache__/test_yaw_aug_arm_config.cpython-310-pytest-9.1.1.pyc
src/tests/__pycache__/test_yaw_aug_arm_config.cpython-310.pyc
src/tests/__pycache__/test_yaw_aug_record_control.cpython-310-pytest-9.1.1.pyc
src/tests/__pycache__/test_yaw_aug_record_control.cpython-310.pyc
src/tests/__pycache__/test_yaw_aug_training.cpython-310-pytest-9.1.1.pyc
src/tests/__pycache__/test_yaw_aug_training.cpython-310.pyc
src/tests/__pycache__/test_yaw_gen_collect.cpython-310-pytest-9.1.1.pyc
src/tests/__pycache__/test_yaw_gen_collect.cpython-310.pyc
src/tests/__pycache__/test_yaw_random_eval.cpython-310-pytest-9.1.1.pyc
src/tests/__pycache__/test_yaw_random_eval.cpython-310.pyc
src/tests/__pycache__/test_yaw_random_eval.cpython-311-pytest-7.4.0.pyc
src/tests/__pycache__/test_yaw_symmetry.cpython-310-pytest-9.1.1.pyc
src/tests/__pycache__/test_yaw_symmetry.cpython-311-pytest-7.4.0.pyc
src/tests/conftest.py
src/tests/exp14_fixed_mode_golden.json
src/tests/fixtures/exp15_yaw_aug_disabled_golden.json
src/tests/test_bn_drift_probe.py
src/tests/test_cond_dispatch.py
src/tests/test_cyl_vit_patch_embedding.py
src/tests/test_eval_paths.py
src/tests/test_exp11_equiv_probe.py
src/tests/test_exp11_orbit_configs.py
src/tests/test_exp11_p0_collect.py
src/tests/test_exp11_p0_runner.py
src/tests/test_exp11_restart_record.py
src/tests/test_exp11_trajectory_figures.py
src/tests/test_exp11_validate_rows.py
src/tests/test_exp11_wandb_readback.py
src/tests/test_exp14_fixed_mode_snapshot.py
src/tests/test_exp14_validate_cell.py
src/tests/test_finetune_cond.py
src/tests/test_gen_model_comparison_gate.py
src/tests/test_invariant_conditioning.py
src/tests/test_phase3_vit_training.py
src/tests/test_retune_lr_state.py
src/tests/test_strip_optimizer_state.py
src/tests/test_train_max_steps.py
src/tests/test_train_sync_batchnorm.py
src/tests/test_vit_gradient_checkpointing.py
src/tests/test_yaw_aug_arm_config.py
src/tests/test_yaw_aug_record_control.py
src/tests/test_yaw_aug_training.py
src/tests/test_yaw_gen_collect.py
src/tests/test_yaw_random_eval.py
src/tests/test_yaw_symmetry.py

exec
/usr/bin/bash -lc "find src -name '__init__.py' -type f -print -exec sed -n '1,220p' {} \\;" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
src/inference/__init__.py
src/training/losses/__init__.py
from .losses import *
from .auraloss import *
src/training/__init__.py
from .factory import create_training_wrapper_from_config, create_metric_callback_from_config
src/models/__init__.py
from .factory import create_model_from_config, create_model_from_config_pathsrc/data/__init__.py
src/__init__.py
from .models.factory import create_model_from_config, create_model_from_config_path
from .models.pretrained import get_pretrained_modelsrc/tools/__init__.py

exec
/usr/bin/bash -lc "git grep -n -I -E '(importlib|__import__|pkgutil|walk_packages|find_namespace_packages|pytest|PYTHONPATH|sys[.]path)' -- train.py src ':"'!src/tests/**'"' pyproject.toml setup.cfg setup.py 2>/dev/null" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
src/data/dataset.py:1:import importlib
src/data/dataset.py:370:            spec = importlib.util.spec_from_file_location("metadata_module", custom_metadata_module_path)
src/data/dataset.py:371:            metadata_module = importlib.util.module_from_spec(spec)

exec
/usr/bin/bash -lc "git grep -n -I -E '("'^|[''^[:alnum:]_])(src[.]tests|tests[.]|conftest|test_[[:alnum:]_]+)([''^[:alnum:]_]|$)'"' -- '*.py' '*.sh' '*.sbatch' '*.ini' '*.json' ':(exclude)src/tests/**'" in /n/fs/gatrdp/codespace/FLAC
exec
/usr/bin/bash -lc 'git log --oneline --decorate --name-status -n 25 -- src/tests' in /n/fs/gatrdp/codespace/FLAC
 succeeded in 274ms:
1ef87b0 exp_14 r3-fix: R3F1b — the kit records per-scene results and the validator demands them
M	src/tests/test_exp14_validate_cell.py
32581d9 exp_14 r3-fix: R3F1a — eval_FLAC records per-scene metrics on request
M	src/tests/test_yaw_random_eval.py
fa66330 exp_14 r3: G4 fail-open closed — nothing compared is not everything agreeing
M	src/tests/test_yaw_gen_collect.py
825b7fc exp_14 r3: model_comparison gains the exp_14 Z row contract (amended §5.7)
M	src/tests/test_gen_model_comparison_gate.py
10aa981 exp_14 r3: gates G1-G4 (blocking) and the G5 check (never blocking)
M	src/tests/test_yaw_gen_collect.py
170aeee exp_14 r3: the §4 estimation conventions — paired-t, Holm, directions
M	src/tests/test_yaw_gen_collect.py
24fea16 exp_14 r3: collector core — artifacts, provenance, §3.3 equalities, blocks
A	src/tests/test_yaw_gen_collect.py
f51578c exp_15 r2-fix: pin the REAL admission record against drift (finding 6, MINOR)
M	src/tests/test_yaw_aug_record_control.py
1e7ef90 exp_15 r2-fix: exclusive record creation (F5) + a truly byte-level arm diff (F4)
M	src/tests/test_yaw_aug_arm_config.py
M	src/tests/test_yaw_aug_record_control.py
a3719f9 exp_15 r2-fix: type-strict config and step comparison (finding 3, MAJOR)
M	src/tests/test_yaw_aug_record_control.py
7e7c512 exp_15 r2-fix: the EMA family must MIRROR the online DiT (finding 2, MAJOR)
M	src/tests/test_yaw_aug_record_control.py
30d5065 exp_15 r2-fix: one stable, safely loaded snapshot (review finding 1, BLOCKING)
M	src/tests/test_yaw_aug_record_control.py
389d0d8 exp_15 r2: bind the checkpoint to the config EMBEDDED inside it (plan §3.3-1)
M	src/tests/test_yaw_aug_record_control.py
d2c11db exp_15 r2: yaw_aug_record_control.py — the control-admission recorder (TDD)
A	src/tests/test_yaw_aug_record_control.py
075b787 exp_15 r2: FLAC_AR_YAWAUG.json — the control's config plus exactly one block
A	src/tests/test_yaw_aug_arm_config.py
a201e3c exp_14 r2-fix: FB3 + FB5 + FB7 + FB8a/b — the submitters' safety rails
M	src/tests/test_exp14_validate_cell.py
1ab22a0 exp_14 r2-fix: FB2 — the Slurm job name must identify the CELL, not the arm
M	src/tests/test_exp14_validate_cell.py

 succeeded in 728ms:
data/HAA/prepare_data.py:113:        test_indices = compute_complement_indices(train_indices + valid_indices, n_total)
data/HAA/prepare_data.py:117:        test_indices = [int(x) for x in test_indices]
data/HAA/prepare_data.py:124:            'test_indices': test_indices,
data/HAA/prepare_data.py:141:    test_json = {}
data/HAA/prepare_data.py:150:        test_indices = data['test_indices']
data/HAA/prepare_data.py:183:        test_files = [f"{i}.wav" for i in test_indices]
data/HAA/prepare_data.py:187:        test_json[scene] = test_files
data/HAA/prepare_data.py:203:    with open(os.path.join(output_dir, 'test_base.json'), 'w') as f:
data/HAA/prepare_data.py:204:        json.dump(test_json, f, indent=4)
eval_pl.py:55:    model_config['test_setup'] = {
eval_pl.py:65:    model_config['test_setup']['metrics'] = model_config['training']['metrics']
src/configs/dataset_configs/HAA/eval/haa_test.json:7:            "json_file_path": "data/HAA/test_base.json",
src/configs/dataset_configs/HAA/eval/haa_test_1.json:7:            "json_file_path": "data/HAA/test_base.json",
src/configs/dataset_configs/custom_metadata/HAA_md.py:96:    scene_context = metadata_scenes[scene_name]['train_indices']#['train_indices']# # test_indices / valid_indices / train_indices
src/models/conditioners.py:186:    tests. ``use_reentrant=False`` is REQUIRED for DDP find_unused_parameters
src/tools/retune_lr_state.py:20:swap is a **silent no-op**.  ``src/tests/test_retune_lr_state.py`` pins that
src/tools/strip_optimizer_state.py:41:   5e-6 case, by ``src/tests/test_finetune_cond.py:664``).
src/training/autoencoders.py:281:        self.test_step_outputs = []
src/training/autoencoders.py:379:    def test_step(self, batch, batch_idx):
src/training/autoencoders.py:402:            test_loss_dict = {}
src/training/autoencoders.py:409:                test_loss_dict[eval_key] = loss_value
src/training/autoencoders.py:415:        self.test_step_outputs.append(test_loss_dict)
src/training/autoencoders.py:416:        return test_loss_dict
src/training/autoencoders.py:420:        for loss_dict in self.test_step_outputs:
src/training/autoencoders.py:428:            test_loss = value / len(self.test_step_outputs)
src/training/autoencoders.py:429:            test_loss = self.all_gather(test_loss).mean().item()
src/training/autoencoders.py:430:            # log_metric(self.logger, f"test/{key}", test_loss)
src/training/autoencoders.py:440:        self.test_step_outputs.clear()  # free memory
src/training/diffusion.py:173:            test_param: tp.Optional[tp.Dict[str, tp.Any]] = None,
src/training/diffusion.py:296:        if test_param is not None:
src/training/diffusion.py:298:                samples=test_param.get("samples", 10240),
src/training/diffusion.py:299:                cfg_scale=test_param.get("cfg_scale", 1.0),
src/training/diffusion.py:300:                steps=int(test_param.get("steps", 1)),
src/training/diffusion.py:301:                sample_rate=test_param.get("sample_rate", 22050),
src/training/diffusion.py:302:                audio_channels=test_param.get("audio_channels", 1),
src/training/diffusion.py:303:                metrics=test_param.get("metrics", {}), 
src/training/diffusion.py:304:                store_predictions = test_param.get("store_predictions", False),
src/training/diffusion.py:464:        training_step, validation_step AND test_step so all inference/training
src/training/diffusion.py:715:    def test_step(self, batch, batch_idx):
src/training/factory.py:151:            test_param = model_config.get("test_setup", None),
tools/__init__.py:6:divergent sibling checkout (see ``src/tests/conftest.py`` for the same guard).
tools/bn_drift_probe.py:10:Public surface (unit-pinned in ``src/tests/test_bn_drift_probe.py``):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_equiv_probe.py:75:# (test_invariant_conditioning.py::test_batched_orbit_maps_every_angle_to_its_slice).
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_equiv_probe.py:119:# pure functions (unit-tested in src/tests/test_exp11_equiv_probe.py)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_equiv_probe.py:138:    test_invariant_conditioning.py::test_batched_orbit_maps_every_angle_to_its_slice.)
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:753:  # instead of HEAD, which is correct behaviour and wrong for these tests. Park
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:185:   && GIT_INDEX_FILE="$SYN_TIDX" git update-index --add --cacheinfo 100644 "$SYN_BLOB" src/tests/test_guardtest_synthetic.py 2>/dev/null; then
worklog/worklog_yixun/exp_14_yaw_gen_claude/exp14_validate_cell.py:28:and the metrics-path shape — and ``src/tests/test_exp14_validate_cell.py`` pins
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect_selftest.py:16:The fixtures are the ones the test-suite builds (``src/tests/test_yaw_gen_collect``
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect_selftest.py:37:import test_yaw_gen_collect as T                                    # noqa: E402
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:149:    test_record "sync $1"
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:403:test_record() { [ -n "${YAW_GEN_TEST_RECORD:-}" ] && printf '%s\n' "$*" >> "$YAW_GEN_TEST_RECORD"; return 0; }
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:407:    test_record "sbatch $*"
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:418:    test_record "scontrol release $1"
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:420:      && test_record "release saw a VALID lease for $1"
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh:431:    test_record "scancel $1"
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:261:test_record() { [ -n "${YAW_GEN_TEST_RECORD:-}" ] && printf '%s\n' "$*" >> "$YAW_GEN_TEST_RECORD"; return 0; }
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:269:    test_record "sync ${f}"
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:425:      test_record "submit $*"
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:427:        test_record "  launching-line-present"
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh:429:        test_record "  launching-line-MISSING"
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:650:# plus the training.yaw_aug block (src/tests/test_yaw_aug_arm_config.py asserts

exec
/usr/bin/bash -lc "sed -n '1,240p' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
#!/usr/bin/env bash
# ============================================================================
# fa_orbit_train_guardtests.sh — guard-branch exercise for the exp_11 arm
# launcher (round-3 review B8 rebuilt this suite).
#
# SAFETY (the old suite violated all three):
#   * it never writes under a production output prefix — every case runs with
#     OUTPUT_ROOT pointed at a mktemp directory;
#   * it never mutates a tracked config — the mislabel case copies the tree into
#     the temp root and points the launcher at the copy via OUTPUT_ROOT-style
#     isolation, and any file it does touch is restored by an EXIT trap;
#   * it submits nothing and touches no GPU.
#
# Vehicles:
#   DRYRUN=1        every cheap gate (pins, arm, rung, config map, semantic
#                   gate, lineage, argv parity), then exit before Slurm/GPU.
#   real mode       with a fake SLURM_JOB_ID: proves the commit/drift and
#                   sbatch-only gates are fail-closed.
#   mocked logs     fa_orbit_classify.py is driven directly over synthetic logs
#                   to prove every exit class (0/3/4/6/7).
#   synthetic ckpt  fa_orbit_ckpt_preflight.py is driven over torch.save'd
#                   Lightning-shaped checkpoints to prove the restart depth.
#
# Usage:  bash worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
# Exit 0 = every case behaved as specified.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR="worklog/worklog_yixun/exp_11_fa_orbit_claude"
LAUNCHER="${EXPDIR}/fa_orbit_train.sbatch"
SUBMITTER="${EXPDIR}/fa_orbit_submit.sh"
CLASSIFY="${EXPDIR}/fa_orbit_classify.py"
PREFLIGHT="${EXPDIR}/fa_orbit_ckpt_preflight.py"
PY=/n/fs/gatrdp/envs/flac/bin/python
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR}/fa_orbit_${TS}_guardtests.log"
HEAD_SHA="$(git rev-parse HEAD)"

exec > >(tee -a "$LOG") 2>&1
echo "=== fa_orbit_train guard exercise — ${TS} — $(git rev-parse --short HEAD) ==="
for f in "$LAUNCHER" "$SUBMITTER" "$CLASSIFY" "$PREFLIGHT"; do
  [ -f "$f" ] || { echo "missing ${f} - abort"; exit 3; }
done

TRACKED_BEFORE="$(git status --porcelain -- "$EXPDIR" src | sort)"
TMP="$(mktemp -d)"
OUT_ROOT="${TMP}/outputs"            # never a production prefix
mkdir -p "$OUT_ROOT"
trap 'rm -rf "$TMP"' EXIT
PASS=0; FAIL=0

case_run() {  # <name> <want-rc> <want-substring> -- <env...>   (runs the launcher)
  local name="$1" want_rc="$2" want_txt="$3"; shift 3; [ "$1" = "--" ] && shift
  local out rc
  out="$(env "$@" bash "$LAUNCHER" 2>&1)"; rc=$?
  if [ "$rc" -eq "$want_rc" ] && echo "$out" | grep -qF -- "$want_txt"; then
    echo "PASS  ${name}  (rc=${rc})"; PASS=$((PASS + 1))
  else
    echo "FAIL  ${name}: want rc=${want_rc} + '${want_txt}', got rc=${rc}"
    echo "$out" | tail -5 | sed 's/^/        | /'; FAIL=$((FAIL + 1))
  fi
}

expect_cmd() {  # <name> <want-rc> <want-substring> -- <command...>
  local name="$1" want_rc="$2" want_txt="$3"; shift 3; [ "$1" = "--" ] && shift
  local out rc
  out="$("$@" 2>&1)"; rc=$?
  if [ "$rc" -eq "$want_rc" ] && echo "$out" | grep -qF -- "$want_txt"; then
    echo "PASS  ${name}  (rc=${rc})"; PASS=$((PASS + 1))
  else
    echo "FAIL  ${name}: want rc=${want_rc} + '${want_txt}', got rc=${rc}"
    echo "$out" | tail -5 | sed 's/^/        | /'; FAIL=$((FAIL + 1))
  fi
}

REPO_ENV=("FA_ORBIT_REPO_OVERRIDE=$PWD")   # dry runs read THIS tree, not the production checkout
SMOKE_ENV=(DRYRUN=1 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}" SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=14000)

echo "--- A. the pin mechanism refuses to launch un-pinned (round-3 B1) ---"
# RETIRED: this asserted that an UNPINNED arm refuses, but every pin landed in
# ea94995, so the placeholder no longer appears in any value and the case could
# never fire. Replaced by the end state it was protecting, plus proof that the
# refusal mechanism itself is still present to catch a future unpinned value.
if grep -qE '^PINNED_[A-Z_]+="TO-PIN-AFTER-P0"' "$LAUNCHER"; then
  echo "FAIL  a launcher pin is still the placeholder"; FAIL=$((FAIL+1))
else
  echo "PASS  every launcher pin holds a concrete value"; PASS=$((PASS+1))
fi
if grep -q 'PIN_PLACEHOLDER="TO-PIN-AFTER-P0"' "$LAUNCHER" \
   && grep -q 'PIN_PLACEHOLDER' "$LAUNCHER"; then
  echo "PASS  the launcher still refuses a placeholder pin if one returns"; PASS=$((PASS+1))
else
  echo "FAIL  the placeholder refusal mechanism is gone"; FAIL=$((FAIL+1))
fi
case_run "SMOKE bypasses the pins" 0 "ARGV PARITY OK" -- "${SMOKE_ENV[@]}" ARM=C8
case_run "SMOKE needs a rung" 2 "SMOKE_RUNG" \
  -- DRYRUN=1 SMOKE=1 ARM=C8 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}"
case_run "SMOKE needs a VRAM floor" 2 "SMOKE_MIN_FREE_MB" \
  -- DRYRUN=1 SMOKE=1 SMOKE_RUNG=16x4 ARM=C8 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}"
case_run "SMOKE identity is separate" 0 "exp11_smoke_C8" -- "${SMOKE_ENV[@]}" ARM=C8

echo "--- B. parameter / arm / rung gates ---"
case_run "missing ARM" 2 "ARM" -- DRYRUN=1 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}"
case_run "missing EXPECT_SHA" 2 "EXPECT_SHA" -- DRYRUN=1 ARM=C8 "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}"
for BAD in C7 FA1 VAN CKPT4; do
  case_run "arm ${BAD} rejected" 2 "not a legal exp_11 arm" -- "${SMOKE_ENV[@]}" ARM=$BAD
done
case_run "bogus rung rejected" 2 "must be 32x2, 16x4 or 8x8" \
  -- DRYRUN=1 SMOKE=1 SMOKE_RUNG=64x1 SMOKE_MIN_FREE_MB=1 ARM=C8 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}"
for R in 32x2 16x4 8x8; do   # all three rungs are feasible now that grad-ckpt is on
  case_run "rung ${R} accepted" 0 "ARGV PARITY OK" \
    -- DRYRUN=1 SMOKE=1 "SMOKE_RUNG=${R}" SMOKE_MIN_FREE_MB=14000 ARM=C4L "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}"
done

echo "--- C. lineage gates ---"
: > "${TMP}/foreign.ckpt"
case_run "initial + RESUME_CKPT" 2 "INITIAL launch must not carry" \
  -- "${SMOKE_ENV[@]}" ARM=C8 "RESUME_CKPT=${TMP}/foreign.ckpt"
case_run "restart w/o ckpt" 2 "RESTART requires RESUME_CKPT" \
  -- "${SMOKE_ENV[@]}" ARM=C8 EXPECTED_STEP=5000
case_run "restart ckpt missing" 2 "not found" \
  -- "${SMOKE_ENV[@]}" ARM=C8 EXPECTED_STEP=5000 "RESUME_CKPT=${TMP}/nope.ckpt"
case_run "restart foreign ckpt" 2 "may only resume a checkpoint from" \
  -- "${SMOKE_ENV[@]}" ARM=C8 EXPECTED_STEP=5000 "RESUME_CKPT=${TMP}/foreign.ckpt"
# a ckpt in the arm's own checkpoints dir but NOT named .ckpt / one level up
SMOKE_RUN="${OUT_ROOT}/exp11_smoke/C8/FLAC_exp11_smoke_C8/exp11_smoke_C8"
mkdir -p "${SMOKE_RUN}/checkpoints"
: > "${SMOKE_RUN}/checkpoints/epoch=1-step=5000.ckpt"
: > "${SMOKE_RUN}/notes.txt"
case_run "restart from the arm's own ckpt dir" 0 "ARGV PARITY OK" \
  -- "${SMOKE_ENV[@]}" ARM=C8 EXPECTED_STEP=5000 SMOKE_MAXSTEPS=6000 \
     "RESUME_CKPT=${SMOKE_RUN}/checkpoints/epoch=1-step=5000.ckpt"
case_run "restart from a non-ckpt sibling" 2 "may only resume a checkpoint from" \
  -- "${SMOKE_ENV[@]}" ARM=C8 EXPECTED_STEP=5000 SMOKE_MAXSTEPS=6000 "RESUME_CKPT=${SMOKE_RUN}/notes.txt"
case_run "restart MAXSTEPS<=step" 2 "must exceed the resume step" \
  -- "${SMOKE_ENV[@]}" ARM=C8 EXPECTED_STEP=5000 SMOKE_MAXSTEPS=30 \
     "RESUME_CKPT=${SMOKE_RUN}/checkpoints/epoch=1-step=5000.ckpt"
case_run "initial refuses an existing run dir" 2 "already exists" -- "${SMOKE_ENV[@]}" ARM=C8

echo "--- D. commit-binding / sbatch-only gates (REAL mode) ---"
# NOTE: no OUTPUT_ROOT here — under a (fake) SLURM_JOB_ID the launcher forces the
# production literal, and the commit gate aborts long before anything is written.
case_run "wrong EXPECT_SHA aborts" 2 "EXPECT_SHA" \
  -- ARM=C4L SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=99000000 \
     EXPECT_SHA=0000000000000000000000000000000000000000 SLURM_JOB_ID=999999
case_run "real mode needs sbatch" 2 "must run under sbatch" \
  -- ARM=C4L SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=99000000 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}"

# Content-scoped binding (content-gate review B5): deterministic SYNTHETIC
# fixtures — dangling commits built with git plumbing. No ref moves, the
# tracked tree is untouched (only unreferenced objects are written; gc prunes
# them), and a missing fixture is a FAILURE, never a SKIP: the identical-tree
# case is the proof that record-only commits cannot kill a queued leg.
# The gate acceptance text is asserted; the run then aborts at a later gate
# (dirty-tree drift today, run-dir/allocation gates on a clean tree) with
# rc=2 and nothing written.
SYN_SAME="$(git commit-tree "$(git rev-parse 'HEAD^{tree}')" -p HEAD -m 'guardtest synthetic: identical tree' 2>/dev/null)"
SYN_IDX="${TMP}/synidx"; SYN_CHG=""
if GIT_INDEX_FILE="$SYN_IDX" git read-tree HEAD 2>/dev/null; then
  SYN_BLOB="$(printf 'guardtest synthetic drift\n' | git hash-object -w --stdin 2>/dev/null)"
  if [ -n "$SYN_BLOB" ] && GIT_INDEX_FILE="$SYN_IDX" git update-index --cacheinfo 100644 "$SYN_BLOB" train.py 2>/dev/null; then
    SYN_TREE="$(GIT_INDEX_FILE="$SYN_IDX" git write-tree 2>/dev/null)"
    [ -n "$SYN_TREE" ] && SYN_CHG="$(git commit-tree "$SYN_TREE" -p HEAD -m 'guardtest synthetic: train.py changed' 2>/dev/null)"
  fi
fi
if [ -n "$SYN_SAME" ]; then
  case_run "moved HEAD, surfaces identical -> gate passes" 2 "commit binding OK (content)" \
    -- ARM=C4L SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=99000000 "EXPECT_SHA=${SYN_SAME}" SLURM_JOB_ID=999999
else
  echo "FAIL  could not synthesize the identical-tree fixture"; FAIL=$((FAIL+1))
fi
if [ -n "$SYN_CHG" ]; then
  case_run "moved HEAD, surfaces changed -> aborts" 2 "training surfaces changed since EXPECT_SHA" \
    -- ARM=C4L SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=99000000 "EXPECT_SHA=${SYN_CHG}" SLURM_JOB_ID=999999
else
  echo "FAIL  could not synthesize the changed-surface fixture"; FAIL=$((FAIL+1))
fi
case_run "symbolic EXPECT_SHA refused" 2 "not a full lowercase 40-hex" \
  -- ARM=C4L SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=99000000 EXPECT_SHA=HEAD SLURM_JOB_ID=999999
# src/tests is excluded from the closure (pytest-only; TDD sessions land tests
# continuously — a test-only commit must NOT kill a queued leg: 3680875-78).
SYN_TIDX="${TMP}/syntidx"; SYN_TESTS=""
if GIT_INDEX_FILE="$SYN_TIDX" git read-tree HEAD 2>/dev/null && [ -n "${SYN_BLOB:-}" ] \
   && GIT_INDEX_FILE="$SYN_TIDX" git update-index --add --cacheinfo 100644 "$SYN_BLOB" src/tests/test_guardtest_synthetic.py 2>/dev/null; then
  SYN_TTREE="$(GIT_INDEX_FILE="$SYN_TIDX" git write-tree 2>/dev/null)"
  [ -n "$SYN_TTREE" ] && SYN_TESTS="$(git commit-tree "$SYN_TTREE" -p HEAD -m 'guardtest synthetic: src/tests only' 2>/dev/null)"
fi
if [ -n "$SYN_TESTS" ]; then
  case_run "src/tests-only change -> gate passes" 2 "commit binding OK (content)" \
    -- ARM=C4L SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=99000000 "EXPECT_SHA=${SYN_TESTS}" SLURM_JOB_ID=999999
else
  echo "FAIL  could not synthesize the src/tests-only fixture"; FAIL=$((FAIL+1))
fi

echo "--- E. semantic gate on a mislabelled config (temp copy; tracked tree untouched) ---"
FAKE_EXP="${TMP}/fakeexp"; mkdir -p "$FAKE_EXP"
cp "${EXPDIR}/FLAC_AR_BF_C4L.json" "${FAKE_EXP}/FLAC_AR_BF_C32.json"      # C4 orbit under the C32 name
expect_cmd "orbit mismatch rejected" 1 "ARM/CONFIG GATE" -- \
  $PY - "${FAKE_EXP}/FLAC_AR_BF_C32.json" C32 <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1])); arm = sys.argv[2]
t = cfg.get("training", {}); bad = []
want = {"C4L": 4, "C8": 8, "C16": 16, "C32": 32}[arm]
angles = t.get("frame_avg_angles")
if not isinstance(angles, list) or len(angles) != want:
    bad.append(f"frame_avg_angles has {angles and len(angles)} entries (want {want})")
if bad:
    sys.exit("ARM/CONFIG GATE: " + "; ".join(bad))
PY
TRACKED_AFTER="$(git status --porcelain -- "$EXPDIR" src | sort)"
if [ "$TRACKED_BEFORE" = "$TRACKED_AFTER" ]; then
  echo "PASS  tracked tree unchanged by the suite (snapshot before == after)"; PASS=$((PASS+1))
else
  echo "FAIL  the suite changed tracked state:"; diff <(echo "$TRACKED_BEFORE") <(echo "$TRACKED_AFTER") | sed 's/^/        | /'
  FAIL=$((FAIL+1))
fi

echo "--- F. exit taxonomy, mocked (round-3 B5) ---"
mk_log() {  # $1 dest, $2 world size (0 = absent), $3 marker?, $4 oom?
  : > "$1"
  [ "$2" != "0" ] && echo "All distributed processes registered. Starting with $2 processes" >> "$1"
  [ "$3" = "yes" ] && echo '`Trainer.fit` stopped: `max_steps=40000` reached.' >> "$1"
  [ "$4" = "yes" ] && echo "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 98.00 MiB" >> "$1"
  return 0
}
A="${TMP}/a.log"; B="${TMP}/b.log"
mk_log "$A" 4 yes no; cp "$A" "$B"
expect_cmd "class 0 complete" 0 "COMPLETE" -- $PY "$CLASSIFY" --rc 0 --tee-rc 0 --ngpu 4 --maxsteps 40000 --log "$A" --log-copy "$B"
mk_log "$A" 0 no no; cp "$A" "$B"
expect_cmd "class 6 world-size absent" 6 "WORLD-SIZE" -- $PY "$CLASSIFY" --rc 0 --tee-rc 0 --ngpu 4 --maxsteps 40000 --log "$A" --log-copy "$B"
mk_log "$A" 1 yes no; cp "$A" "$B"
expect_cmd "class 6 wrong world-size" 6 "reported [1]" -- $PY "$CLASSIFY" --rc 0 --tee-rc 0 --ngpu 4 --maxsteps 40000 --log "$A" --log-copy "$B"
mk_log "$A" 4 no yes; cp "$A" "$B"
expect_cmd "class 3 OOM on nonzero rc" 3 "OOM" -- $PY "$CLASSIFY" --rc 1 --tee-rc 0 --ngpu 4 --maxsteps 40000 --log "$A" --log-copy "$B"
mk_log "$A" 4 no no; cp "$A" "$B"
expect_cmd "class 4 missing marker" 4 "NO-MARKER" -- $PY "$CLASSIFY" --rc 0 --tee-rc 0 --ngpu 4 --maxsteps 40000 --log "$A" --log-copy "$B"
mk_log "$A" 4 yes no; cp "$A" "$B"; echo "divergent tail" >> "$B"
expect_cmd "class 7 logs differ" 7 "LOG-PROVENANCE" -- $PY "$CLASSIFY" --rc 0 --tee-rc 0 --ngpu 4 --maxsteps 40000 --log "$A" --log-copy "$B"
mk_log "$A" 4 yes no; cp "$A" "$B"; rm -f "$B"

exec
/usr/bin/bash -lc "sed -n '1,240p' src/tests/conftest.py && git ls-files 'src/tests/**' | sed -n '1,240p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
"""Pytest configuration for the FLAC test-suite.

Ensures ``src.*`` imports resolve to *this* repository checkout rather than a
stale ``pip install .`` copy that may live in site-packages (a known pitfall on
the development machine). We prepend the repo root (three directory levels up
from this file: ``src/tests/`` -> ``src/`` -> repo root) to the front of
``sys.path`` so it wins over any installed copy.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # src/tests/ -> src/ -> repo root
sys.path.insert(0, _REPO_ROOT)
src/tests/conftest.py
src/tests/exp14_fixed_mode_golden.json
src/tests/fixtures/exp15_yaw_aug_disabled_golden.json
src/tests/test_bn_drift_probe.py
src/tests/test_cond_dispatch.py
src/tests/test_cyl_vit_patch_embedding.py
src/tests/test_eval_paths.py
src/tests/test_exp11_equiv_probe.py
src/tests/test_exp11_orbit_configs.py
src/tests/test_exp11_p0_collect.py
src/tests/test_exp11_p0_runner.py
src/tests/test_exp11_restart_record.py
src/tests/test_exp11_trajectory_figures.py
src/tests/test_exp11_validate_rows.py
src/tests/test_exp11_wandb_readback.py
src/tests/test_exp14_fixed_mode_snapshot.py
src/tests/test_exp14_validate_cell.py
src/tests/test_finetune_cond.py
src/tests/test_gen_model_comparison_gate.py
src/tests/test_invariant_conditioning.py
src/tests/test_phase3_vit_training.py
src/tests/test_retune_lr_state.py
src/tests/test_strip_optimizer_state.py
src/tests/test_train_max_steps.py
src/tests/test_train_sync_batchnorm.py
src/tests/test_vit_gradient_checkpointing.py
src/tests/test_yaw_aug_arm_config.py
src/tests/test_yaw_aug_record_control.py
src/tests/test_yaw_aug_training.py
src/tests/test_yaw_gen_collect.py
src/tests/test_yaw_random_eval.py
src/tests/test_yaw_symmetry.py

exec
/usr/bin/bash -lc "sed -n '1,430p' src/data/dataset.py | sed -n '330,405p'; grep -RInE '(src[.]tests|src/tests|from[[:space:]]+tests|import[[:space:]]+tests|test_guardtest_synthetic)' --exclude-dir=.git --exclude-dir=src/tests . 2>/dev/null | head -n 300" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
        result = []
        for b in batched:
            if isinstance(b[0], (int, float)):
                b = np.array(b)
            elif isinstance(b[0], torch.Tensor):
                b = torch.stack(b)
            elif isinstance(b[0], np.ndarray):
                b = np.array(b)
            else:
                b = b
            result.append(b)
        return result


def create_dataloader_from_config(dataset_config, batch_size, sample_size, sample_rate, audio_channels=2, num_workers=4, shuffle = True):
    dataset_type = dataset_config.get("dataset_type", None)

    assert dataset_type is not None, "Dataset type must be specified in dataset config"

    if audio_channels == 1:
        force_channels = "mono"
    else:
        force_channels = dataset_config.get("force_channels", "stereo") 

    assert dataset_type == "audio_dir", f"Unsupported dataset type: {dataset_type}"

    audio_dir_configs = dataset_config.get("datasets", None)

    assert audio_dir_configs is not None, "Directory configuration must be specified in datasets[\"dataset\"]"

    configs = []

    for audio_dir_config in audio_dir_configs:
        audio_dir_path = audio_dir_config.get("path", None)
        assert audio_dir_path is not None, "Path must be set for local audio directory configuration"

        custom_metadata_fn = None
        custom_metadata_module_path = audio_dir_config.get("custom_metadata_module", None)

        if custom_metadata_module_path is not None:
            spec = importlib.util.spec_from_file_location("metadata_module", custom_metadata_module_path)
            metadata_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(metadata_module)                

            custom_metadata_fn = metadata_module.get_custom_metadata

        scenes = audio_dir_config.get("scenes", None)
        is_eval = dataset_config.get("is_eval", False)
        json_file_path = audio_dir_config.get("json_file_path", None)
        folder_name = audio_dir_config.get("folder_name", None)

        configs.append(
            LocalDatasetConfig(
                id=audio_dir_config["id"],
                path=audio_dir_path,
                custom_metadata_fn=custom_metadata_fn, 
                json_file_path=json_file_path, 
                folder_name=folder_name,
                scenes=scenes,
                is_eval=is_eval, 
                unseeneval = dataset_config.get("unseeneval", False),
                seeneval = dataset_config.get("seeneval", False),
                conditioning = dataset_config.get("modalities", None),
            )
        )

    train_set = SampleDataset(
        configs,
        sample_rate=sample_rate,
        sample_size=sample_size,
        random_crop=dataset_config.get("random_crop", True),
        force_channels=force_channels, 
        augs=dataset_config.get("augs", False),
    )

    return torch.utils.data.DataLoader(train_set, batch_size, shuffle=shuffle,
./.pytest_cache/v/cache/lastfailed:2:  "src/tests/test_yaw_random_eval.py::test_apply_rotation_plan_fixed_ignores_generator_and_stream": true,
./.pytest_cache/v/cache/lastfailed:3:  "src/tests/test_gen_model_comparison_gate.py::test_exp14_gate_admits_a_row_whose_stream_sidecar_is_not_in_this_checkout": true
./.pytest_cache/v/cache/nodeids:2:  "src/tests/test_bn_drift_probe.py::test_bn_drift_metrics_known",
./.pytest_cache/v/cache/nodeids:3:  "src/tests/test_bn_drift_probe.py::test_bn_drift_metrics_zero",
./.pytest_cache/v/cache/nodeids:4:  "src/tests/test_bn_drift_probe.py::test_probe_report_provenance",
./.pytest_cache/v/cache/nodeids:5:  "src/tests/test_bn_drift_probe.py::test_probe_variant_hook",
./.pytest_cache/v/cache/nodeids:6:  "src/tests/test_bn_drift_probe.py::test_recorder_conv_bn_stem",
./.pytest_cache/v/cache/nodeids:7:  "src/tests/test_bn_drift_probe.py::test_recorder_cuda_smoke",
./.pytest_cache/v/cache/nodeids:8:  "src/tests/test_bn_drift_probe.py::test_recorder_finds_all_bns",
./.pytest_cache/v/cache/nodeids:9:  "src/tests/test_bn_drift_probe.py::test_recorder_welford_unbiased",
./.pytest_cache/v/cache/nodeids:10:  "src/tests/test_bn_drift_probe.py::test_snapshot_buffers_catches_mutation",
./.pytest_cache/v/cache/nodeids:11:  "src/tests/test_cond_dispatch.py::test_default_cond_method_is_vanilla",
./.pytest_cache/v/cache/nodeids:12:  "src/tests/test_cond_dispatch.py::test_dispatch_all_three_sites",
./.pytest_cache/v/cache/nodeids:13:  "src/tests/test_cond_dispatch.py::test_frame_avg_angles_override",
./.pytest_cache/v/cache/nodeids:14:  "src/tests/test_cond_dispatch.py::test_known_cond_methods_construct[fa_invariant]",
./.pytest_cache/v/cache/nodeids:15:  "src/tests/test_cond_dispatch.py::test_known_cond_methods_construct[vanilla]",
./.pytest_cache/v/cache/nodeids:16:  "src/tests/test_cond_dispatch.py::test_unknown_cond_method_raises[canon]",
./.pytest_cache/v/cache/nodeids:17:  "src/tests/test_cond_dispatch.py::test_unknown_cond_method_raises[fa-invariant]",
./.pytest_cache/v/cache/nodeids:18:  "src/tests/test_cond_dispatch.py::test_vanilla_no_symmetrization",
./.pytest_cache/v/cache/nodeids:19:  "src/tests/test_cyl_vit_patch_embedding.py::CylindricalPatchEmbeddingTest::test_cnn_stem_contract_and_token_shape",
./.pytest_cache/v/cache/nodeids:20:  "src/tests/test_cyl_vit_patch_embedding.py::CylindricalPatchEmbeddingTest::test_cnn_variant_is_c16_equivariant_after_gauge_alignment",
./.pytest_cache/v/cache/nodeids:21:  "src/tests/test_cyl_vit_patch_embedding.py::CylindricalPatchEmbeddingTest::test_conditioner_factory_forwards_patch_embed_type",
./.pytest_cache/v/cache/nodeids:22:  "src/tests/test_cyl_vit_patch_embedding.py::CylindricalPatchEmbeddingTest::test_experiment_configs_only_change_declared_vit_initialization_fields",
./.pytest_cache/v/cache/nodeids:23:  "src/tests/test_cyl_vit_patch_embedding.py::CylindricalPatchEmbeddingTest::test_linear_remains_default_and_is_parameter_matched_to_cnn",
./.pytest_cache/v/cache/nodeids:24:  "src/tests/test_eval_paths.py::test_build_output_paths_n_angles_reflected",
./.pytest_cache/v/cache/nodeids:25:  "src/tests/test_eval_paths.py::test_comparator_guards_orbit_execution_provenance[frame_avg_fwd_cap-8]",
./.pytest_cache/v/cache/nodeids:26:  "src/tests/test_eval_paths.py::test_comparator_guards_orbit_execution_provenance[orbit_execution-loop]",
./.pytest_cache/v/cache/nodeids:27:  "src/tests/test_eval_paths.py::test_comparator_guards_orbit_execution_provenance[source_sha-0000000000000000000000000000000000000000]",
./.pytest_cache/v/cache/nodeids:28:  "src/tests/test_eval_paths.py::test_comparator_loads_dict_returns_tensor",
./.pytest_cache/v/cache/nodeids:29:  "src/tests/test_eval_paths.py::test_comparator_loads_legacy_bare",
./.pytest_cache/v/cache/nodeids:30:  "src/tests/test_eval_paths.py::test_comparator_meta_match_proceeds",
./.pytest_cache/v/cache/nodeids:31:  "src/tests/test_eval_paths.py::test_comparator_meta_mismatch_raises[batch_size-1]",
./.pytest_cache/v/cache/nodeids:32:  "src/tests/test_eval_paths.py::test_comparator_meta_mismatch_raises[cond_autocast-bf16]",
./.pytest_cache/v/cache/nodeids:33:  "src/tests/test_eval_paths.py::test_comparator_meta_mismatch_raises[cond_method-fa_invariant]",
./.pytest_cache/v/cache/nodeids:34:  "src/tests/test_eval_paths.py::test_comparator_meta_mismatch_raises[dataset_config-other.json]",
./.pytest_cache/v/cache/nodeids:35:  "src/tests/test_eval_paths.py::test_comparator_meta_mismatch_raises[frame_avg_angles-bad4]",
./.pytest_cache/v/cache/nodeids:36:  "src/tests/test_eval_paths.py::test_comparator_meta_mismatch_raises[seed-999]",
./.pytest_cache/v/cache/nodeids:37:  "src/tests/test_eval_paths.py::test_comparator_missing_cond_autocast_treated_as_default",
./.pytest_cache/v/cache/nodeids:38:  "src/tests/test_eval_paths.py::test_comparator_rotate_deg_mismatch_allowed",
./.pytest_cache/v/cache/nodeids:39:  "src/tests/test_eval_paths.py::test_comparator_single_sided_meta_warns_not_raises",
./.pytest_cache/v/cache/nodeids:40:  "src/tests/test_eval_paths.py::test_evaluate_model_save_path_flows_through_build_output_paths",
./.pytest_cache/v/cache/nodeids:41:  "src/tests/test_eval_paths.py::test_evaluate_model_unknown_cond_autocast_raises_fast",
./.pytest_cache/v/cache/nodeids:42:  "src/tests/test_eval_paths.py::test_evaluate_model_unknown_cond_method_raises_fast",
./.pytest_cache/v/cache/nodeids:43:  "src/tests/test_eval_paths.py::test_fa_invariant_paths",
./.pytest_cache/v/cache/nodeids:44:  "src/tests/test_eval_paths.py::test_fa_invariant_rot0_has_method_no_rot_suffix",
./.pytest_cache/v/cache/nodeids:45:  "src/tests/test_eval_paths.py::test_legacy_rot_metrics_path",
./.pytest_cache/v/cache/nodeids:46:  "src/tests/test_eval_paths.py::test_legacy_vanilla_paths",
./.pytest_cache/v/cache/nodeids:47:  "src/tests/test_eval_paths.py::test_load_integrity_real_state_dict",
./.pytest_cache/v/cache/nodeids:48:  "src/tests/test_eval_paths.py::test_load_integrity_stray_and_escape_hatch",
./.pytest_cache/v/cache/nodeids:49:  "src/tests/test_eval_paths.py::test_metrics_and_predictions_share_suffix",
./.pytest_cache/v/cache/nodeids:50:  "src/tests/test_eval_paths.py::test_metrics_json_records_method_fa_invariant",
./.pytest_cache/v/cache/nodeids:51:  "src/tests/test_eval_paths.py::test_metrics_json_records_method_vanilla",
./.pytest_cache/v/cache/nodeids:52:  "src/tests/test_eval_paths.py::test_metrics_record_carries_the_runtime_protocol",
./.pytest_cache/v/cache/nodeids:53:  "src/tests/test_eval_paths.py::test_metrics_record_defaults_stay_none_not_wrong",
./.pytest_cache/v/cache/nodeids:54:  "src/tests/test_eval_paths.py::test_metrics_record_makes_the_batch_schedule_reconstructible",
./.pytest_cache/v/cache/nodeids:55:  "src/tests/test_eval_paths.py::test_parser_cond_autocast_choices",
./.pytest_cache/v/cache/nodeids:56:  "src/tests/test_eval_paths.py::test_predictions_meta_includes_cond_autocast",
./.pytest_cache/v/cache/nodeids:57:  "src/tests/test_eval_paths.py::test_records_carry_orbit_execution_provenance",
./.pytest_cache/v/cache/nodeids:58:  "src/tests/test_eval_paths.py::test_resolve_cond_autocast_modes[bf16-expected1]",
./.pytest_cache/v/cache/nodeids:59:  "src/tests/test_eval_paths.py::test_resolve_cond_autocast_modes[default-expected0]",
./.pytest_cache/v/cache/nodeids:60:  "src/tests/test_eval_paths.py::test_resolve_cond_autocast_modes[off-expected2]",
./.pytest_cache/v/cache/nodeids:61:  "src/tests/test_eval_paths.py::test_resolve_cond_autocast_unknown_raises",
./.pytest_cache/v/cache/nodeids:62:  "src/tests/test_eval_paths.py::test_rot_suffix_is_injective_for_fractional_angles",
./.pytest_cache/v/cache/nodeids:63:  "src/tests/test_eval_paths.py::test_source_sha_falls_back_when_git_is_unavailable",
./.pytest_cache/v/cache/nodeids:64:  "src/tests/test_eval_paths.py::test_vanilla_rows_are_not_labelled_as_batched_orbit",
./.pytest_cache/v/cache/nodeids:65:  "src/tests/test_eval_paths.py::test_weights_source_is_resolved_from_the_checkpoint_not_asserted",
./.pytest_cache/v/cache/nodeids:66:  "src/tests/test_exp11_equiv_probe.py::test_deviation_floor_tames_a_near_zero_reference",
./.pytest_cache/v/cache/nodeids:67:  "src/tests/test_exp11_equiv_probe.py::test_deviation_is_nan_proof",
./.pytest_cache/v/cache/nodeids:68:  "src/tests/test_exp11_equiv_probe.py::test_deviation_is_zero_for_identical_tensors",
./.pytest_cache/v/cache/nodeids:69:  "src/tests/test_exp11_equiv_probe.py::test_deviation_reports_absolute_and_normwise_error",
./.pytest_cache/v/cache/nodeids:70:  "src/tests/test_exp11_equiv_probe.py::test_expected_cells_cover_the_review_matrix",
./.pytest_cache/v/cache/nodeids:71:  "src/tests/test_exp11_equiv_probe.py::test_gate_runs_in_true_fp32_and_only_train_mirrors_trainpy",
./.pytest_cache/v/cache/nodeids:72:  "src/tests/test_exp11_equiv_probe.py::test_matmul_precision_context_sets_and_restores_both_flags",
./.pytest_cache/v/cache/nodeids:73:  "src/tests/test_exp11_equiv_probe.py::test_orbit_angles",
./.pytest_cache/v/cache/nodeids:74:  "src/tests/test_exp11_equiv_probe.py::test_record_id_falls_back_through_path_then_index",
./.pytest_cache/v/cache/nodeids:75:  "src/tests/test_exp11_equiv_probe.py::test_record_id_uses_the_relative_path_not_the_scene_label",
./.pytest_cache/v/cache/nodeids:76:  "src/tests/test_exp11_equiv_probe.py::test_record_ids_are_distinct_across_records_of_one_scene",
./.pytest_cache/v/cache/nodeids:77:  "src/tests/test_exp11_equiv_probe.py::test_summarize_separates_gated_from_recorded",
./.pytest_cache/v/cache/nodeids:78:  "src/tests/test_exp11_equiv_probe.py::test_tolerances_are_the_registered_ones",
./.pytest_cache/v/cache/nodeids:79:  "src/tests/test_exp11_equiv_probe.py::test_train_qualification_excludes_c32",
./.pytest_cache/v/cache/nodeids:80:  "src/tests/test_exp11_equiv_probe.py::test_verdict_enforces_both_bounds_on_gated_cells",
./.pytest_cache/v/cache/nodeids:81:  "src/tests/test_exp11_equiv_probe.py::test_verdict_fails_on_a_missing_cell",
./.pytest_cache/v/cache/nodeids:82:  "src/tests/test_exp11_equiv_probe.py::test_verdict_fails_on_an_empty_result_set",
./.pytest_cache/v/cache/nodeids:83:  "src/tests/test_exp11_equiv_probe.py::test_verdict_fails_on_non_finite_forward_tensors",
./.pytest_cache/v/cache/nodeids:84:  "src/tests/test_exp11_equiv_probe.py::test_verdict_fails_when_a_vit_id_is_absent",
./.pytest_cache/v/cache/nodeids:85:  "src/tests/test_exp11_equiv_probe.py::test_verdict_passes_a_complete_clean_run",
./.pytest_cache/v/cache/nodeids:86:  "src/tests/test_exp11_equiv_probe.py::test_verdict_records_but_does_not_gate_train_mode_divergence",
./.pytest_cache/v/cache/nodeids:87:  "src/tests/test_exp11_orbit_configs.py::test_allowed_diff_leaves[C16]",
./.pytest_cache/v/cache/nodeids:88:  "src/tests/test_exp11_orbit_configs.py::test_allowed_diff_leaves[C32]",
./.pytest_cache/v/cache/nodeids:89:  "src/tests/test_exp11_orbit_configs.py::test_allowed_diff_leaves[C4L]",
./.pytest_cache/v/cache/nodeids:90:  "src/tests/test_exp11_orbit_configs.py::test_allowed_diff_leaves[C8]",
./.pytest_cache/v/cache/nodeids:91:  "src/tests/test_exp11_orbit_configs.py::test_allowed_diff_leaves[FA1]",
./.pytest_cache/v/cache/nodeids:92:  "src/tests/test_exp11_orbit_configs.py::test_angle_lists[C16]",
./.pytest_cache/v/cache/nodeids:93:  "src/tests/test_exp11_orbit_configs.py::test_angle_lists[C32]",
./.pytest_cache/v/cache/nodeids:94:  "src/tests/test_exp11_orbit_configs.py::test_angle_lists[C4L]",
./.pytest_cache/v/cache/nodeids:95:  "src/tests/test_exp11_orbit_configs.py::test_angle_lists[C8]",
./.pytest_cache/v/cache/nodeids:96:  "src/tests/test_exp11_orbit_configs.py::test_angle_lists[FA1]",
./.pytest_cache/v/cache/nodeids:97:  "src/tests/test_exp11_orbit_configs.py::test_c4l_bridge_identity",
./.pytest_cache/v/cache/nodeids:98:  "src/tests/test_exp11_orbit_configs.py::test_c4l_is_byte_identical_to_exp07_bf",
./.pytest_cache/v/cache/nodeids:99:  "src/tests/test_exp11_orbit_configs.py::test_n_matches_filename[C16]",
./.pytest_cache/v/cache/nodeids:100:  "src/tests/test_exp11_orbit_configs.py::test_n_matches_filename[C32]",
./.pytest_cache/v/cache/nodeids:101:  "src/tests/test_exp11_orbit_configs.py::test_n_matches_filename[C4L]",
./.pytest_cache/v/cache/nodeids:102:  "src/tests/test_exp11_orbit_configs.py::test_n_matches_filename[C8]",
./.pytest_cache/v/cache/nodeids:103:  "src/tests/test_exp11_orbit_configs.py::test_n_matches_filename[FA1]",
./.pytest_cache/v/cache/nodeids:104:  "src/tests/test_exp11_orbit_configs.py::test_non_boolean_gc_leaf_is_rejected",
./.pytest_cache/v/cache/nodeids:105:  "src/tests/test_exp11_orbit_configs.py::test_vanckpt_adds_only_grad_checkpointing",
./.pytest_cache/v/cache/nodeids:106:  "src/tests/test_exp11_orbit_configs.py::test_vanl_config_is_orbit_free",
./.pytest_cache/v/cache/nodeids:107:  "src/tests/test_exp11_orbit_configs.py::test_vanl_is_a_launchable_arm",
./.pytest_cache/v/cache/nodeids:108:  "src/tests/test_exp11_orbit_configs.py::test_vanl_is_registered_in_the_init_identity_gate",
./.pytest_cache/v/cache/nodeids:109:  "src/tests/test_exp11_orbit_configs.py::test_vanl_maps_to_the_vanckpt_config",
./.pytest_cache/v/cache/nodeids:110:  "src/tests/test_exp11_orbit_configs.py::test_vanl_semantic_gate_is_the_mirror_image",
./.pytest_cache/v/cache/nodeids:111:  "src/tests/test_exp11_orbit_configs.py::test_vanl_time_limit_is_pinned_from_the_vanilla_rate",
./.pytest_cache/v/cache/nodeids:112:  "src/tests/test_exp11_p0_collect.py::test_admit_rows_accepts_matching_rows",
./.pytest_cache/v/cache/nodeids:113:  "src/tests/test_exp11_p0_collect.py::test_admit_rows_refuses_cross_run_and_mislabelled_rows",
./.pytest_cache/v/cache/nodeids:114:  "src/tests/test_exp11_p0_collect.py::test_admit_rows_refuses_cross_run_mislabelled_and_reshaped_rows",
./.pytest_cache/v/cache/nodeids:115:  "src/tests/test_exp11_p0_collect.py::test_admit_rows_refuses_duplicate_cell",
./.pytest_cache/v/cache/nodeids:116:  "src/tests/test_exp11_p0_collect.py::test_admit_rows_refuses_duplicate_key",
./.pytest_cache/v/cache/nodeids:117:  "src/tests/test_exp11_p0_collect.py::test_all_ok_gate",
./.pytest_cache/v/cache/nodeids:118:  "src/tests/test_exp11_p0_collect.py::test_ddp_scaling",
./.pytest_cache/v/cache/nodeids:119:  "src/tests/test_exp11_p0_collect.py::test_ddp_scaling_efficiency",
./.pytest_cache/v/cache/nodeids:120:  "src/tests/test_exp11_p0_collect.py::test_grad_ckpt_comparison_is_retired",
./.pytest_cache/v/cache/nodeids:121:  "src/tests/test_exp11_p0_collect.py::test_grad_ckpt_cost",
./.pytest_cache/v/cache/nodeids:122:  "src/tests/test_exp11_p0_collect.py::test_main_matrix_requires_the_fit_and_reports_run_specific_path",
./.pytest_cache/v/cache/nodeids:123:  "src/tests/test_exp11_p0_collect.py::test_main_requires_manifest_and_exits_nonzero_when_incomplete",
./.pytest_cache/v/cache/nodeids:124:  "src/tests/test_exp11_p0_collect.py::test_main_requires_manifest_and_fails_on_a_pending_cell",
./.pytest_cache/v/cache/nodeids:125:  "src/tests/test_exp11_p0_collect.py::test_main_spot_mode_succeeds_without_a_fit",
./.pytest_cache/v/cache/nodeids:126:  "src/tests/test_exp11_p0_collect.py::test_main_workers_mode_pairs_in_one_manifest",
./.pytest_cache/v/cache/nodeids:127:  "src/tests/test_exp11_p0_collect.py::test_manifest_allows_one_cell_with_two_worker_variants",
./.pytest_cache/v/cache/nodeids:128:  "src/tests/test_exp11_p0_collect.py::test_marginal_contrast_c8_minus_c4l",
./.pytest_cache/v/cache/nodeids:129:  "src/tests/test_exp11_p0_collect.py::test_marginal_contrast_is_reported_separately",
./.pytest_cache/v/cache/nodeids:130:  "src/tests/test_exp11_p0_collect.py::test_orbit_pass_fit_excludes_van_and_requires_the_exact_set",
./.pytest_cache/v/cache/nodeids:131:  "src/tests/test_exp11_p0_collect.py::test_orbit_pass_fit_flags_implausible_output_as_ambiguous",
./.pytest_cache/v/cache/nodeids:132:  "src/tests/test_exp11_p0_collect.py::test_orbit_pass_fit_recovers_slope_and_unattributed_residual",
./.pytest_cache/v/cache/nodeids:133:  "src/tests/test_exp11_p0_collect.py::test_orbit_pass_fit_requires_the_exact_van_c4l_c8_set",
./.pytest_cache/v/cache/nodeids:134:  "src/tests/test_exp11_p0_collect.py::test_orbit_pass_fit_uses_fa1_c4l_c8",
./.pytest_cache/v/cache/nodeids:135:  "src/tests/test_exp11_p0_collect.py::test_parse_manifest",
./.pytest_cache/v/cache/nodeids:136:  "src/tests/test_exp11_p0_collect.py::test_parse_manifest_rejects_broken_manifests",
./.pytest_cache/v/cache/nodeids:137:  "src/tests/test_exp11_p0_collect.py::test_parse_missing_field_raises",
./.pytest_cache/v/cache/nodeids:138:  "src/tests/test_exp11_p0_collect.py::test_parse_non_p0_line_returns_none",
./.pytest_cache/v/cache/nodeids:139:  "src/tests/test_exp11_p0_collect.py::test_parse_rejects_malformed_and_nonfinite_values",
./.pytest_cache/v/cache/nodeids:140:  "src/tests/test_exp11_p0_collect.py::test_parse_valid_line",
./.pytest_cache/v/cache/nodeids:141:  "src/tests/test_exp11_p0_collect.py::test_poller_evidence_accepted_when_complete",
./.pytest_cache/v/cache/nodeids:142:  "src/tests/test_exp11_p0_collect.py::test_poller_evidence_hash_mismatch_is_invalid",
./.pytest_cache/v/cache/nodeids:143:  "src/tests/test_exp11_p0_collect.py::test_poller_evidence_missing_file_is_invalid",
./.pytest_cache/v/cache/nodeids:144:  "src/tests/test_exp11_p0_collect.py::test_poller_evidence_requires_every_uuid_in_window",
./.pytest_cache/v/cache/nodeids:145:  "src/tests/test_exp11_p0_collect.py::test_poller_evidence_requires_finite_util_and_power",
./.pytest_cache/v/cache/nodeids:146:  "src/tests/test_exp11_p0_collect.py::test_render_markdown_deterministic_and_labels_residual",
./.pytest_cache/v/cache/nodeids:147:  "src/tests/test_exp11_p0_collect.py::test_render_markdown_withholds_derived_when_incomplete",
./.pytest_cache/v/cache/nodeids:148:  "src/tests/test_exp11_p0_collect.py::test_scan_dir_treats_duplicate_result_lines_as_malformed",
./.pytest_cache/v/cache/nodeids:149:  "src/tests/test_exp11_p0_collect.py::test_steps_per_second",
./.pytest_cache/v/cache/nodeids:150:  "src/tests/test_exp11_p0_collect.py::test_steps_per_second_rejects_nonpositive_and_nonfinite",
./.pytest_cache/v/cache/nodeids:151:  "src/tests/test_exp11_p0_collect.py::test_summarize_marks_malformed_files",
./.pytest_cache/v/cache/nodeids:152:  "src/tests/test_exp11_p0_collect.py::test_summarize_ok_row",
./.pytest_cache/v/cache/nodeids:153:  "src/tests/test_exp11_p0_collect.py::test_summarize_poller_flags_missing_uuid_and_empty_window",
./.pytest_cache/v/cache/nodeids:154:  "src/tests/test_exp11_p0_collect.py::test_summarize_poller_window",
./.pytest_cache/v/cache/nodeids:155:  "src/tests/test_exp11_p0_collect.py::test_summarize_status_classes",
./.pytest_cache/v/cache/nodeids:156:  "src/tests/test_exp11_p0_collect.py::test_vanilla_contrast_is_reported_separately",
./.pytest_cache/v/cache/nodeids:157:  "src/tests/test_exp11_p0_collect.py::test_worker_contrast_needs_both_halves",
./.pytest_cache/v/cache/nodeids:158:  "src/tests/test_exp11_p0_runner.py::test_callback_mark_is_idempotent",
./.pytest_cache/v/cache/nodeids:159:  "src/tests/test_exp11_p0_runner.py::test_callback_marks_only_window_steps",
./.pytest_cache/v/cache/nodeids:160:  "src/tests/test_exp11_p0_runner.py::test_callback_window_is_configurable",
./.pytest_cache/v/cache/nodeids:161:  "src/tests/test_exp11_p0_runner.py::test_no_cuda_sync_when_unavailable",
./.pytest_cache/v/cache/nodeids:162:  "src/tests/test_exp11_p0_runner.py::test_prints_only_on_rank_zero",
./.pytest_cache/v/cache/nodeids:163:  "src/tests/test_exp11_p0_runner.py::test_runner_documents_parity_and_defers_heavy_imports",
./.pytest_cache/v/cache/nodeids:164:  "src/tests/test_exp11_p0_runner.py::test_synchronizes_cuda_before_marking",
./.pytest_cache/v/cache/nodeids:165:  "src/tests/test_exp11_p0_runner.py::test_window_rate",
./.pytest_cache/v/cache/nodeids:166:  "src/tests/test_exp11_p0_runner.py::test_window_rate_rejects_nonpositive_and_nonfinite",
./.pytest_cache/v/cache/nodeids:167:  "src/tests/test_exp11_p0_runner.py::test_window_rate_requires_both_marks",
./.pytest_cache/v/cache/nodeids:168:  "src/tests/test_exp11_restart_record.py::test_a_checkpoint_moved_out_of_the_published_path_is_refused",
./.pytest_cache/v/cache/nodeids:169:  "src/tests/test_exp11_restart_record.py::test_a_foreign_checkpoint_at_a_produced_step_is_refused",
./.pytest_cache/v/cache/nodeids:170:  "src/tests/test_exp11_restart_record.py::test_a_leg_is_recordable_once_its_arm_is_anchored",
./.pytest_cache/v/cache/nodeids:171:  "src/tests/test_exp11_restart_record.py::test_a_missing_producer_manifest_admits_nothing",
./.pytest_cache/v/cache/nodeids:172:  "src/tests/test_exp11_restart_record.py::test_a_producer_manifest_from_another_leg_is_refused",
./.pytest_cache/v/cache/nodeids:173:  "src/tests/test_exp11_restart_record.py::test_a_published_step_may_never_change",
./.pytest_cache/v/cache/nodeids:174:  "src/tests/test_exp11_restart_record.py::test_a_refused_record_writes_nothing",
./.pytest_cache/v/cache/nodeids:175:  "src/tests/test_exp11_restart_record.py::test_a_step_the_leg_never_produced_is_refused",
./.pytest_cache/v/cache/nodeids:176:  "src/tests/test_exp11_restart_record.py::test_a_vanished_restart_manifest_admits_nothing",
./.pytest_cache/v/cache/nodeids:177:  "src/tests/test_exp11_restart_record.py::test_add_anchor_audits_the_checkpoint_itself[kw0-embedded global_step]",
./.pytest_cache/v/cache/nodeids:178:  "src/tests/test_exp11_restart_record.py::test_add_anchor_audits_the_checkpoint_itself[kw1-optimizer state]",
./.pytest_cache/v/cache/nodeids:179:  "src/tests/test_exp11_restart_record.py::test_add_anchor_audits_the_checkpoint_itself[kw2-lr_schedulers]",
./.pytest_cache/v/cache/nodeids:180:  "src/tests/test_exp11_restart_record.py::test_add_anchor_audits_the_checkpoint_itself[kw3-no EMA weights]",
./.pytest_cache/v/cache/nodeids:181:  "src/tests/test_exp11_restart_record.py::test_add_anchor_dry_run_writes_nothing",
./.pytest_cache/v/cache/nodeids:182:  "src/tests/test_exp11_restart_record.py::test_add_anchor_is_idempotent",
./.pytest_cache/v/cache/nodeids:183:  "src/tests/test_exp11_restart_record.py::test_add_anchor_records_the_audited_final_checkpoint",
./.pytest_cache/v/cache/nodeids:184:  "src/tests/test_exp11_restart_record.py::test_add_anchor_refuses_a_drifted_launch_manifest",
./.pytest_cache/v/cache/nodeids:185:  "src/tests/test_exp11_restart_record.py::test_add_anchor_refuses_a_foreign_config",
./.pytest_cache/v/cache/nodeids:186:  "src/tests/test_exp11_restart_record.py::test_add_anchor_refuses_an_ambiguous_step",
./.pytest_cache/v/cache/nodeids:187:  "src/tests/test_exp11_restart_record.py::test_add_anchor_refuses_an_arm_with_no_launch",
./.pytest_cache/v/cache/nodeids:188:  "src/tests/test_exp11_restart_record.py::test_add_anchor_refuses_to_re_anchor_a_different_checkpoint",
./.pytest_cache/v/cache/nodeids:189:  "src/tests/test_exp11_restart_record.py::test_add_anchor_step_crosscheck_must_match_the_registered_budget",
./.pytest_cache/v/cache/nodeids:190:  "src/tests/test_exp11_restart_record.py::test_an_arm_with_no_recorded_leg_admits_nothing",
./.pytest_cache/v/cache/nodeids:191:  "src/tests/test_exp11_restart_record.py::test_arm_without_an_audited_anchor_is_refused",
./.pytest_cache/v/cache/nodeids:192:  "src/tests/test_exp11_restart_record.py::test_dry_run_publishes_nothing",
./.pytest_cache/v/cache/nodeids:193:  "src/tests/test_exp11_restart_record.py::test_duplicate_legs_are_refused",
./.pytest_cache/v/cache/nodeids:194:  "src/tests/test_exp11_restart_record.py::test_every_identity_field_is_validated[over0-not RESTART]",
./.pytest_cache/v/cache/nodeids:195:  "src/tests/test_exp11_restart_record.py::test_every_identity_field_is_validated[over1-records no job]",
./.pytest_cache/v/cache/nodeids:196:  "src/tests/test_exp11_restart_record.py::test_every_identity_field_is_validated[over10-RESTART wall pin]",
./.pytest_cache/v/cache/nodeids:197:  "src/tests/test_exp11_restart_record.py::test_every_identity_field_is_validated[over11-manifest config_sha256]",
./.pytest_cache/v/cache/nodeids:198:  "src/tests/test_exp11_restart_record.py::test_every_identity_field_is_validated[over12-manifest vae_sha256]",
./.pytest_cache/v/cache/nodeids:199:  "src/tests/test_exp11_restart_record.py::test_every_identity_field_is_validated[over13-manifest p0_manifest_sha256]",
./.pytest_cache/v/cache/nodeids:200:  "src/tests/test_exp11_restart_record.py::test_every_identity_field_is_validated[over14-manifest save_dir]",
./.pytest_cache/v/cache/nodeids:201:  "src/tests/test_exp11_restart_record.py::test_every_identity_field_is_validated[over15-!= the file's actual]",
./.pytest_cache/v/cache/nodeids:202:  "src/tests/test_exp11_restart_record.py::test_every_identity_field_is_validated[over2-IS the INITIAL job]",
./.pytest_cache/v/cache/nodeids:203:  "src/tests/test_exp11_restart_record.py::test_every_identity_field_is_validated[over3-records no launch_uuid]",
./.pytest_cache/v/cache/nodeids:204:  "src/tests/test_exp11_restart_record.py::test_every_identity_field_is_validated[over4-records no commit]",
./.pytest_cache/v/cache/nodeids:205:  "src/tests/test_exp11_restart_record.py::test_every_identity_field_is_validated[over5-manifest arm]",
./.pytest_cache/v/cache/nodeids:206:  "src/tests/test_exp11_restart_record.py::test_every_identity_field_is_validated[over6-manifest rung]",
./.pytest_cache/v/cache/nodeids:207:  "src/tests/test_exp11_restart_record.py::test_every_identity_field_is_validated[over7-manifest micro]",
./.pytest_cache/v/cache/nodeids:208:  "src/tests/test_exp11_restart_record.py::test_every_identity_field_is_validated[over8-Q10 budget pin]",
./.pytest_cache/v/cache/nodeids:209:  "src/tests/test_exp11_restart_record.py::test_every_identity_field_is_validated[over9-audited final step]",
./.pytest_cache/v/cache/nodeids:210:  "src/tests/test_exp11_restart_record.py::test_extend_appends_new_steps_only",
./.pytest_cache/v/cache/nodeids:211:  "src/tests/test_exp11_restart_record.py::test_missing_resume_file_is_refused_not_trusted",
./.pytest_cache/v/cache/nodeids:212:  "src/tests/test_exp11_restart_record.py::test_non_canonical_resume_path_is_refused",
./.pytest_cache/v/cache/nodeids:213:  "src/tests/test_exp11_restart_record.py::test_producer_manifest_holds_the_legs_own_checkpoints",
./.pytest_cache/v/cache/nodeids:214:  "src/tests/test_exp11_restart_record.py::test_publication_is_atomic_and_leaves_no_debris",
./.pytest_cache/v/cache/nodeids:215:  "src/tests/test_exp11_restart_record.py::test_records_a_clean_leg",
./.pytest_cache/v/cache/nodeids:216:  "src/tests/test_exp11_restart_record.py::test_resume_file_is_always_rehashed",
./.pytest_cache/v/cache/nodeids:217:  "src/tests/test_exp11_restart_record.py::test_steps_past_the_budget_are_not_the_legs_output",
./.pytest_cache/v/cache/nodeids:218:  "src/tests/test_exp11_restart_record.py::test_the_produced_checkpoint_is_admissible",
./.pytest_cache/v/cache/nodeids:219:  "src/tests/test_exp11_restart_record.py::test_the_registered_restart_manifest_is_rehashed_by_the_screen",
./.pytest_cache/v/cache/nodeids:220:  "src/tests/test_exp11_restart_record.py::test_the_screen_revalidates_every_leg_field[chains_to-0000000000000000000000000000000000000000000000000000000000000000-chains_to]",
./.pytest_cache/v/cache/nodeids:221:  "src/tests/test_exp11_restart_record.py::test_the_screen_revalidates_every_leg_field[commit--records no commit]",
./.pytest_cache/v/cache/nodeids:222:  "src/tests/test_exp11_restart_record.py::test_the_screen_revalidates_every_leg_field[config_sha256-0000000000000000000000000000000000000000000000000000000000000000-config_sha256]",
./.pytest_cache/v/cache/nodeids:223:  "src/tests/test_exp11_restart_record.py::test_the_screen_revalidates_every_leg_field[expected_step-30000-resumed at step]",
./.pytest_cache/v/cache/nodeids:224:  "src/tests/test_exp11_restart_record.py::test_the_screen_revalidates_every_leg_field[mode-INITIAL-leg mode]",
./.pytest_cache/v/cache/nodeids:225:  "src/tests/test_exp11_restart_record.py::test_the_screen_revalidates_every_leg_field[producer_manifest--records no producer_manifest]",
./.pytest_cache/v/cache/nodeids:226:  "src/tests/test_exp11_restart_record.py::test_the_screen_revalidates_every_leg_field[resume_ckpt_sha256-0000000000000000000000000000000000000000000000000000000000000000-resume_ckpt_sha256]",
./.pytest_cache/v/cache/nodeids:227:  "src/tests/test_exp11_restart_record.py::test_the_screen_revalidates_every_leg_field[save_dir-outputs/exp11_C16-save_dir]",
./.pytest_cache/v/cache/nodeids:228:  "src/tests/test_exp11_trajectory_figures.py::test_a_clean_five_seed_cell_becomes_a_band",
./.pytest_cache/v/cache/nodeids:229:  "src/tests/test_exp11_trajectory_figures.py::test_a_mixed_pin_block_is_refused",
./.pytest_cache/v/cache/nodeids:230:  "src/tests/test_exp11_trajectory_figures.py::test_a_mixed_provenance_block_is_refused",
./.pytest_cache/v/cache/nodeids:231:  "src/tests/test_exp11_trajectory_figures.py::test_a_partial_block_is_refused_not_narrowed",
./.pytest_cache/v/cache/nodeids:232:  "src/tests/test_exp11_trajectory_figures.py::test_band_cell_handles_a_missing_metric",
./.pytest_cache/v/cache/nodeids:233:  "src/tests/test_exp11_trajectory_figures.py::test_band_cell_renders_mean_sd_and_the_extrema",
./.pytest_cache/v/cache/nodeids:234:  "src/tests/test_exp11_trajectory_figures.py::test_five_files_that_are_not_five_seeds_are_refused",
./.pytest_cache/v/cache/nodeids:235:  "src/tests/test_exp11_trajectory_figures.py::test_rows_that_validate_but_carry_no_metric_are_not_invented",
./.pytest_cache/v/cache/nodeids:236:  "src/tests/test_exp11_trajectory_figures.py::test_the_default_validator_is_the_row_validator",
./.pytest_cache/v/cache/nodeids:237:  "src/tests/test_exp11_trajectory_figures.py::test_the_grid_floor_and_ceiling_are_enforced_through_the_validator[102500-budget]",
./.pytest_cache/v/cache/nodeids:238:  "src/tests/test_exp11_trajectory_figures.py::test_the_grid_floor_and_ceiling_are_enforced_through_the_validator[40000-strictly above]",
./.pytest_cache/v/cache/nodeids:239:  "src/tests/test_exp11_trajectory_figures.py::test_the_grid_floor_and_ceiling_are_enforced_through_the_validator[42501-checkpoint grid]",
./.pytest_cache/v/cache/nodeids:240:  "src/tests/test_exp11_trajectory_figures.py::test_the_harvest_asks_for_the_traj_contract",
./.pytest_cache/v/cache/nodeids:241:  "src/tests/test_exp11_trajectory_figures.py::test_the_png_generator_scales_from_the_same_extent",
./.pytest_cache/v/cache/nodeids:242:  "src/tests/test_exp11_trajectory_figures.py::test_the_svg_panel_keeps_the_band_inside_the_plot_area",
./.pytest_cache/v/cache/nodeids:243:  "src/tests/test_exp11_trajectory_figures.py::test_value_extent_includes_the_band_envelope",
./.pytest_cache/v/cache/nodeids:244:  "src/tests/test_exp11_trajectory_figures.py::test_value_extent_includes_the_conf_whiskers",
./.pytest_cache/v/cache/nodeids:245:  "src/tests/test_exp11_trajectory_figures.py::test_value_extent_is_empty_safe",
./.pytest_cache/v/cache/nodeids:246:  "src/tests/test_exp11_validate_rows.py::test_a_filename_that_does_not_match_the_schema_is_rejected",
./.pytest_cache/v/cache/nodeids:247:  "src/tests/test_exp11_validate_rows.py::test_a_full_q9_vanl_cell_validates",
./.pytest_cache/v/cache/nodeids:248:  "src/tests/test_exp11_validate_rows.py::test_a_good_vanl_row_passes",
./.pytest_cache/v/cache/nodeids:249:  "src/tests/test_exp11_validate_rows.py::test_a_k1_screen_row_validates",
./.pytest_cache/v/cache/nodeids:250:  "src/tests/test_exp11_validate_rows.py::test_a_rotated_evaluation_cannot_masquerade_as_a_screen_row",
./.pytest_cache/v/cache/nodeids:251:  "src/tests/test_exp11_validate_rows.py::test_absent_evaluator_field_is_a_failure_not_a_skip[batch_size]",
./.pytest_cache/v/cache/nodeids:252:  "src/tests/test_exp11_validate_rows.py::test_absent_evaluator_field_is_a_failure_not_a_skip[cfg_scale]",
./.pytest_cache/v/cache/nodeids:253:  "src/tests/test_exp11_validate_rows.py::test_absent_evaluator_field_is_a_failure_not_a_skip[ckpt_path]",
./.pytest_cache/v/cache/nodeids:254:  "src/tests/test_exp11_validate_rows.py::test_absent_evaluator_field_is_a_failure_not_a_skip[cond_autocast]",
./.pytest_cache/v/cache/nodeids:255:  "src/tests/test_exp11_validate_rows.py::test_absent_evaluator_field_is_a_failure_not_a_skip[cond_method]",
./.pytest_cache/v/cache/nodeids:256:  "src/tests/test_exp11_validate_rows.py::test_absent_evaluator_field_is_a_failure_not_a_skip[dataset_config]",
./.pytest_cache/v/cache/nodeids:257:  "src/tests/test_exp11_validate_rows.py::test_absent_evaluator_field_is_a_failure_not_a_skip[device]",
./.pytest_cache/v/cache/nodeids:258:  "src/tests/test_exp11_validate_rows.py::test_absent_evaluator_field_is_a_failure_not_a_skip[eval_name]",
./.pytest_cache/v/cache/nodeids:259:  "src/tests/test_exp11_validate_rows.py::test_absent_evaluator_field_is_a_failure_not_a_skip[frame_avg_angles]",
./.pytest_cache/v/cache/nodeids:260:  "src/tests/test_exp11_validate_rows.py::test_absent_evaluator_field_is_a_failure_not_a_skip[frame_avg_fwd_cap]",
./.pytest_cache/v/cache/nodeids:261:  "src/tests/test_exp11_validate_rows.py::test_absent_evaluator_field_is_a_failure_not_a_skip[n_samples]",
./.pytest_cache/v/cache/nodeids:262:  "src/tests/test_exp11_validate_rows.py::test_absent_evaluator_field_is_a_failure_not_a_skip[orbit_execution]",
./.pytest_cache/v/cache/nodeids:263:  "src/tests/test_exp11_validate_rows.py::test_absent_evaluator_field_is_a_failure_not_a_skip[seed]",
./.pytest_cache/v/cache/nodeids:264:  "src/tests/test_exp11_validate_rows.py::test_absent_evaluator_field_is_a_failure_not_a_skip[source_sha]",
./.pytest_cache/v/cache/nodeids:265:  "src/tests/test_exp11_validate_rows.py::test_absent_evaluator_field_is_a_failure_not_a_skip[steps]",
./.pytest_cache/v/cache/nodeids:266:  "src/tests/test_exp11_validate_rows.py::test_absent_evaluator_field_is_a_failure_not_a_skip[weights_source]",
./.pytest_cache/v/cache/nodeids:267:  "src/tests/test_exp11_validate_rows.py::test_all_six_table_metrics_are_required",
./.pytest_cache/v/cache/nodeids:268:  "src/tests/test_exp11_validate_rows.py::test_batch_size_and_device_are_validated",
./.pytest_cache/v/cache/nodeids:269:  "src/tests/test_exp11_validate_rows.py::test_boolean_metrics_are_not_numbers",
./.pytest_cache/v/cache/nodeids:270:  "src/tests/test_exp11_validate_rows.py::test_cell_missing_a_seed_fails",
./.pytest_cache/v/cache/nodeids:271:  "src/tests/test_exp11_validate_rows.py::test_cell_rejects_a_row_from_another_arm",
./.pytest_cache/v/cache/nodeids:272:  "src/tests/test_exp11_validate_rows.py::test_cell_rejects_two_different_checkpoint_hashes",
./.pytest_cache/v/cache/nodeids:273:  "src/tests/test_exp11_validate_rows.py::test_cell_requires_one_identical_checkpoint_and_code_identity",
./.pytest_cache/v/cache/nodeids:274:  "src/tests/test_exp11_validate_rows.py::test_cell_with_a_duplicated_seed_fails",
./.pytest_cache/v/cache/nodeids:275:  "src/tests/test_exp11_validate_rows.py::test_cell_with_all_five_seeds_passes",
./.pytest_cache/v/cache/nodeids:276:  "src/tests/test_exp11_validate_rows.py::test_ckpt_must_live_in_the_arms_own_run_dir",
./.pytest_cache/v/cache/nodeids:277:  "src/tests/test_exp11_validate_rows.py::test_ckpt_step_must_match_the_claimed_step",
./.pytest_cache/v/cache/nodeids:278:  "src/tests/test_exp11_validate_rows.py::test_contracts_are_registered_not_supplied",
./.pytest_cache/v/cache/nodeids:279:  "src/tests/test_exp11_validate_rows.py::test_cross_and_r3_are_registered_at_the_40k_endpoint_only",
./.pytest_cache/v/cache/nodeids:280:  "src/tests/test_exp11_validate_rows.py::test_cross_cell_needs_every_orbit_but_the_arms_own",
./.pytest_cache/v/cache/nodeids:281:  "src/tests/test_exp11_validate_rows.py::test_cross_contract_is_registered_and_never_table_admissible",
./.pytest_cache/v/cache/nodeids:282:  "src/tests/test_exp11_validate_rows.py::test_cross_record_angles_must_match_the_named_orbit",
./.pytest_cache/v/cache/nodeids:283:  "src/tests/test_exp11_validate_rows.py::test_cross_rejects_the_arms_own_training_orbit",
./.pytest_cache/v/cache/nodeids:284:  "src/tests/test_exp11_validate_rows.py::test_cross_sidecar_must_record_both_orbits",
./.pytest_cache/v/cache/nodeids:285:  "src/tests/test_exp11_validate_rows.py::test_duplicate_json_keys_are_rejected",
./.pytest_cache/v/cache/nodeids:286:  "src/tests/test_exp11_validate_rows.py::test_empty_metrics_are_rejected",
./.pytest_cache/v/cache/nodeids:287:  "src/tests/test_exp11_validate_rows.py::test_eval_names_are_injective_across_both_new_cell_types",
./.pytest_cache/v/cache/nodeids:288:  "src/tests/test_exp11_validate_rows.py::test_evaluator_fields_are_type_checked[batch_size-64]",
./.pytest_cache/v/cache/nodeids:289:  "src/tests/test_exp11_validate_rows.py::test_evaluator_fields_are_type_checked[cfg_scale-1.0]",
./.pytest_cache/v/cache/nodeids:290:  "src/tests/test_exp11_validate_rows.py::test_evaluator_fields_are_type_checked[device-0]",
./.pytest_cache/v/cache/nodeids:291:  "src/tests/test_exp11_validate_rows.py::test_evaluator_fields_are_type_checked[eval_name-7]",
./.pytest_cache/v/cache/nodeids:292:  "src/tests/test_exp11_validate_rows.py::test_evaluator_fields_are_type_checked[n_samples-6337.0]",
./.pytest_cache/v/cache/nodeids:293:  "src/tests/test_exp11_validate_rows.py::test_evaluator_fields_are_type_checked[seed-42]",
./.pytest_cache/v/cache/nodeids:294:  "src/tests/test_exp11_validate_rows.py::test_evaluator_fields_are_type_checked[steps-1.0]",
./.pytest_cache/v/cache/nodeids:295:  "src/tests/test_exp11_validate_rows.py::test_filename_must_be_exactly_what_build_output_paths_generates",
./.pytest_cache/v/cache/nodeids:296:  "src/tests/test_exp11_validate_rows.py::test_full_split_item_count_is_required",
./.pytest_cache/v/cache/nodeids:297:  "src/tests/test_exp11_validate_rows.py::test_futility_contract_rejects_a_second_seed",
./.pytest_cache/v/cache/nodeids:298:  "src/tests/test_exp11_validate_rows.py::test_good_row_passes",
./.pytest_cache/v/cache/nodeids:299:  "src/tests/test_exp11_validate_rows.py::test_hashes_are_recomputed_not_trusted",

exec
/usr/bin/bash -lc "find /usr/share/doc -maxdepth 3 -type f \\( -iname '*glossary*' -o -iname '*pathspec*' -o -iname 'git*.txt*' \\) -print 2>/dev/null | head -n 100; git help -a | sed -n '1,240p'" in /n/fs/gatrdp/codespace/FLAC
exec
/usr/bin/bash -lc 'git diff --name-only 3c2bcbe''^ 3c2bcbe -- train.py defaults.ini src data/AR; git diff --name-only 3c2bcbe''^ 3c2bcbe -- train.py defaults.ini src '"':(exclude)src/tests' data/AR; git diff --name-only 3c2bcbe"'^ 3c2bcbe -- train.py defaults.ini '"':(exclude)src/tests' src data/AR; git ls-files -- src ':(exclude)src/tests' | grep '"'^src/tests/'"' | head" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
src/tests/test_exp14_validate_cell.py
src/tests/test_yaw_gen_collect.py

 succeeded in 208ms:
/usr/share/doc/git/glossary-content.adoc
/usr/share/doc/git/gitglossary.html
/usr/share/doc/git/gitglossary.adoc
/usr/share/doc/qt5/qtdoc/qml-glossary.html
/usr/share/doc/pam-devel/html/adg-glossary.html
/usr/share/doc/latexmk/example_rcfiles/glossary_latexmkrc
See 'git help <command>' to read about a specific subcommand

Main Porcelain Commands
   add                     Add file contents to the index
   am                      Apply a series of patches from a mailbox
   archive                 Create an archive of files from a named tree
   backfill                Download missing objects in a partial clone
   bisect                  Use binary search to find the commit that introduced a bug
   branch                  List, create, or delete branches
   bundle                  Move objects and refs by archive
   checkout                Switch branches or restore working tree files
   cherry-pick             Apply the changes introduced by some existing commits
   citool                  Graphical alternative to git-commit
   clean                   Remove untracked files from the working tree
   clone                   Clone a repository into a new directory
   commit                  Record changes to the repository
   describe                Give an object a human readable name based on an available ref
   diff                    Show changes between commits, commit and working tree, etc
   fetch                   Download objects and refs from another repository
   format-patch            Prepare patches for e-mail submission
   gc                      Cleanup unnecessary files and optimize the local repository
   gitk                    The Git repository browser
   grep                    Print lines matching a pattern
   gui                     A portable graphical interface to Git
   init                    Create an empty Git repository or reinitialize an existing one
   log                     Show commit logs
   maintenance             Run tasks to optimize Git repository data
   merge                   Join two or more development histories together
   mv                      Move or rename a file, a directory, or a symlink
   notes                   Add or inspect object notes
   pull                    Fetch from and integrate with another repository or a local branch
   push                    Update remote refs along with associated objects
   range-diff              Compare two commit ranges (e.g. two versions of a branch)
   rebase                  Reapply commits on top of another base tip
   reset                   Reset current HEAD to the specified state
   restore                 Restore working tree files
   revert                  Revert some existing commits
   rm                      Remove files from the working tree and from the index
   scalar                  A tool for managing large Git repositories
   shortlog                Summarize 'git log' output
   show                    Show various types of objects
   sparse-checkout         Reduce your working tree to a subset of tracked files
   stash                   Stash the changes in a dirty working directory away
   status                  Show the working tree status
   submodule               Initialize, update or inspect submodules
   switch                  Switch branches
   tag                     Create, list, delete or verify tags
   worktree                Manage multiple working trees

Ancillary Commands / Manipulators
   config                  Get and set repository or global options
   fast-export             Git data exporter
   fast-import             Backend for fast Git data importers
   filter-branch           Rewrite branches
   mergetool               Run merge conflict resolution tools to resolve merge conflicts
   pack-refs               Pack heads and tags for efficient repository access
   prune                   Prune all unreachable objects from the object database
   reflog                  Manage reflog information
   refs                    Low-level access to refs
   remote                  Manage set of tracked repositories
   repack                  Pack unpacked objects in a repository
   replace                 Create, list, delete refs to replace objects

Ancillary Commands / Interrogators
   annotate                Annotate file lines with commit information
   blame                   Show what revision and author last modified each line of a file
   bugreport               Collect information for user to file a bug report
   count-objects           Count unpacked number of objects and their disk consumption
   diagnose                Generate a zip archive of diagnostic information
   difftool                Show changes using common diff tools
   fsck                    Verifies the connectivity and validity of the objects in the database
   gitweb                  Git web interface (web frontend to Git repositories)
   help                    Display help information about Git
   instaweb                Instantly browse your working repository in gitweb
   merge-tree              Perform merge without touching index or working tree
   rerere                  Reuse recorded resolution of conflicted merges
   show-branch             Show branches and their commits
   verify-commit           Check the GPG signature of commits
   verify-tag              Check the GPG signature of tags
   version                 Display version information about Git
   whatchanged             Show logs with differences each commit introduces

Interacting with Others
   imap-send               Send a collection of patches from stdin to an IMAP folder
   quiltimport             Applies a quilt patchset onto the current branch
   request-pull            Generates a summary of pending changes
   send-email              Send a collection of patches as emails
   svn                     Bidirectional operation between a Subversion repository and Git

Low-level Commands / Manipulators
   apply                   Apply a patch to files and/or to the index
   checkout-index          Copy files from the index to the working tree
   commit-graph            Write and verify Git commit-graph files
   commit-tree             Create a new commit object
   hash-object             Compute object ID and optionally create an object from a file
   index-pack              Build pack index file for an existing packed archive
   merge-file              Run a three-way file merge
   merge-index             Run a merge for files needing merging
   mktag                   Creates a tag object with extra validation
   mktree                  Build a tree-object from ls-tree formatted text
   multi-pack-index        Write and verify multi-pack-indexes
   pack-objects            Create a packed archive of objects
   prune-packed            Remove extra objects that are already in pack files
   read-tree               Reads tree information into the index
   replay                  EXPERIMENTAL: Replay commits on a new base, works with bare repos too
   symbolic-ref            Read, modify and delete symbolic refs
   unpack-objects          Unpack objects from a packed archive
   update-index            Register file contents in the working tree to the index
   update-ref              Update the object name stored in a ref safely
   write-tree              Create a tree object from the current index

Low-level Commands / Interrogators
   cat-file                Provide contents or details of repository objects
   cherry                  Find commits yet to be applied to upstream
   diff-files              Compares files in the working tree and the index
   diff-index              Compare a tree to the working tree or index
   diff-pairs              Compare the content and mode of provided blob pairs
   diff-tree               Compares the content and mode of blobs found via two tree objects
   for-each-ref            Output information on each ref
   for-each-repo           Run a Git command on a list of repositories
   get-tar-commit-id       Extract commit ID from an archive created using git-archive
   last-modified           EXPERIMENTAL: Show when files were last modified
   ls-files                Show information about files in the index and the working tree
   ls-remote               List references in a remote repository
   ls-tree                 List the contents of a tree object
   merge-base              Find as good common ancestors as possible for a merge
   name-rev                Find symbolic names for given revs
   pack-redundant          Find redundant pack files
   repo                    Retrieve information about the repository
   rev-list                Lists commit objects in reverse chronological order
   rev-parse               Pick out and massage parameters
   show-index              Show packed archive index
   show-ref                List references in a local repository
   unpack-file             Creates a temporary file with a blob's contents
   var                     Show a Git logical variable
   verify-pack             Validate packed Git archive files

Low-level Commands / Syncing Repositories
   daemon                  A really simple server for Git repositories
   fetch-pack              Receive missing objects from another repository
   http-backend            Server side implementation of Git over HTTP
   send-pack               Push objects over Git protocol to another repository
   update-server-info      Update auxiliary info file to help dumb servers

Low-level Commands / Internal Helpers
   check-attr              Display gitattributes information
   check-ignore            Debug gitignore / exclude files
   check-mailmap           Show canonical names and email addresses of contacts
   check-ref-format        Ensures that a reference name is well formed
   column                  Display data in columns
   credential              Retrieve and store user credentials
   credential-cache        Helper to temporarily store passwords in memory
   credential-store        Helper to store credentials on disk
   fmt-merge-msg           Produce a merge commit message
   hook                    Run git hooks
   interpret-trailers      Add or parse structured information in commit messages
   mailinfo                Extracts patch and authorship from a single e-mail message
   mailsplit               Simple UNIX mbox splitter program
   merge-one-file          The standard helper program to use with git-merge-index
   patch-id                Compute unique ID for a patch
   sh-i18n                 Git's i18n setup code for shell scripts
   sh-setup                Common Git shell script setup code
   stripspace              Remove unnecessary whitespace

User-facing repository, command and file interfaces
   attributes              Defining attributes per path
   cli                     Git command-line interface and conventions
   hooks                   Hooks used by Git
   ignore                  Specifies intentionally untracked files to ignore
   mailmap                 Map author/committer names and/or E-Mail addresses
   modules                 Defining submodule properties
   repository-layout       Git Repository Layout
   revisions               Specifying revisions and ranges for Git

Developer-facing file formats, protocols and other interfaces
   format-bundle           The bundle file format
   format-chunk            Chunk-based file formats
   format-commit-graph     Git commit-graph format
   format-index            Git index format
   format-pack             Git pack format
   format-signature        Git cryptographic signature formats
   protocol-capabilities   Protocol v0 and v1 capabilities
   protocol-common         Things common to various protocols
   protocol-http           Git HTTP-based protocols
   protocol-pack           How packs are transferred over-the-wire
   protocol-v2             Git Wire Protocol, Version 2

External commands
   lfs

exec
/usr/bin/bash -lc "grep -n -A45 -B12 -E '(exclude|Pathspec|pathspec)' /usr/share/doc/git/glossary-content.adoc | sed -n '1,260p'" in /n/fs/gatrdp/codespace/FLAC
exec
/usr/bin/bash -lc "git status -h 2>&1 | sed -n '1,220p'; git diff -h 2>&1 | sed -n '1,260p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
337-	contrast, no-overlay mode also deletes tracked files not
338-	present in the source, similar to 'rsync --delete'.
339-
340-[[def_pack]]pack::
341-	A set of objects which have been compressed into one file (to save space
342-	or to transmit them efficiently).
343-
344-[[def_pack_index]]pack index::
345-	The list of identifiers, and other information, of the objects in a
346-	<<def_pack,pack>>, to assist in efficiently accessing the contents of a
347-	pack.
348-
349:[[def_pathspec]]pathspec::
350-	Pattern used to limit paths in Git commands.
351-+
352:Pathspecs are used on the command line of "git ls-files", "git
353-ls-tree", "git add", "git grep", "git diff", "git checkout",
354-and many other commands to
355-limit the scope of operations to some subset of the tree or
356-working tree.  See the documentation of each command for whether
357-paths are relative to the current directory or toplevel.  The
358:pathspec syntax is as follows:
359-+
360---
361-
362-* any path matches itself
363:* the pathspec up to the last slash represents a
364:  directory prefix.  The scope of that pathspec is
365-  limited to that subtree.
366:* the rest of the pathspec is a pattern for the remainder
367-  of the pathname.  Paths relative to the directory
368-  prefix will be matched against that pattern using fnmatch(3);
369-  in particular, '*' and '?' _can_ match directory separators.
370-
371---
372-+
373-For example, Documentation/*.jpg will match all .jpg files
374-in the Documentation subtree,
375-including Documentation/chapter_1/figure_1.jpg.
376-+
377:A pathspec that begins with a colon `:` has special meaning.  In the
378-short form, the leading colon `:` is followed by zero or more "magic
379-signature" letters (which optionally is terminated by another colon `:`),
380-and the remainder is the pattern to match against the path.
381-The "magic signature" consists of ASCII symbols that are neither
382-alphanumeric, glob, regex special characters nor colon.
383-The optional colon that terminates the "magic signature" can be
384-omitted if the pattern begins with a character that does not belong to
385-"magic signature" symbol set and is not a colon.
386-+
387-In the long form, the leading colon `:` is followed by an open
388-parenthesis `(`, a comma-separated list of zero or more "magic words",
389-and a close parentheses `)`, and the remainder is the pattern to match
390-against the path.
391-+
392:A pathspec with only a colon means "there is no pathspec". This form
393:should not be combined with other pathspec.
394-+
395---
396-top;;
397-	The magic word `top` (magic signature: `/`) makes the pattern
398-	match from the root of the working tree, even when you are
399-	running the command from inside a subdirectory.
400-
401-literal;;
402-	Wildcards in the pattern such as `*` or `?` are treated
403-	as literal characters.
404-
405-icase;;
406-	Case insensitive match.
407-
408-glob;;
409-	Git treats the pattern as a shell glob suitable for
410-	consumption by fnmatch(3) with the FNM_PATHNAME flag:
411-	wildcards in the pattern will not match a / in the pathname.
412-	For example, "Documentation/{asterisk}.html" matches
413-	"Documentation/git.html" but not "Documentation/ppc/ppc.html"
414-	or "tools/perf/Documentation/perf.html".
415-+
416-Two consecutive asterisks ("`**`") in patterns matched against
417-full pathname may have special meaning:
418-
419- - A leading "`**`" followed by a slash means match in all
420-   directories. For example, "`**/foo`" matches file or directory
421-   "`foo`" anywhere. "`**/foo/bar`" matches file or directory "`bar`"
422-   anywhere that is directly under directory "`foo`".
423-
424- - A trailing "`/**`" matches everything inside. For example,
425-   "`abc/**`" matches all files inside directory "abc", relative
426-   to the location of the `.gitignore` file, with infinite depth.
427-
428- - A slash followed by two consecutive asterisks then a slash
429-   matches zero or more directories. For example, "`a/**/b`"
430-   matches "`a/b`", "`a/x/b`", "`a/x/y/b`" and so on.
431-
432- - Other consecutive asterisks are considered invalid.
433-+
434-Glob magic is incompatible with literal magic.
435-
436-attr;;
437-After `attr:` comes a space separated list of "attribute
438-requirements", all of which must be met in order for the
439-path to be considered a match; this is in addition to the
440:usual non-magic pathspec pattern matching.
441-See linkgit:gitattributes[5].
442-+
443-Each of the attribute requirements for the path takes one of
444-these forms:
445-
446-- "`ATTR`" requires that the attribute `ATTR` be set.
447-
448-- "`-ATTR`" requires that the attribute `ATTR` be unset.
449-
450-- "`ATTR=VALUE`" requires that the attribute `ATTR` be
451-  set to the string `VALUE`.
452-
453-- "`!ATTR`" requires that the attribute `ATTR` be
454-  unspecified.
455-+
456-Note that when matching against a tree object, attributes are still
457-obtained from working tree, not from the given tree object.
458-
459:exclude;;
460:	After a path matches any non-exclude pathspec, it will be run
461:	through all exclude pathspecs (magic signature: `!` or its
462-	synonym `^`). If it matches, the path is ignored.  When there
463:	is no non-exclude pathspec, the exclusion is applied to the
464:	result set as if invoked without any pathspec.
465---
466-
467-[[def_parent]]parent::
468-	A <<def_commit_object,commit object>> contains a (possibly empty) list
469-	of the logical predecessor(s) in the line of development, i.e. its
470-	parents.
471-
472-[[def_peel]]peel::
473-	The action of recursively <<def_dereference,dereferencing>> a
474-	<<def_tag_object,tag object>>.
475-
476-[[def_pickaxe]]pickaxe::
477-	The term <<def_pickaxe,pickaxe>> refers to an option to the diffcore
478-	routines that help select changes that add or delete a given text
479-	string. With the `--pickaxe-all` option, it can be used to view the full
480-	<<def_changeset,changeset>> that introduced or removed, say, a
481-	particular line of text. See linkgit:git-diff[1].
482-
483-[[def_plumbing]]plumbing::
484-	Cute name for <<def_core_git,core Git>>.
485-
486-[[def_porcelain]]porcelain::
487-	Cute name for programs and program suites depending on
488-	<<def_core_git,core Git>>, presenting a high level access to
489-	core Git. Porcelains expose more of a <<def_SCM,SCM>>
490-	interface than the <<def_plumbing,plumbing>>.
491-
492-[[def_per_worktree_ref]]per-worktree ref::
493-	Refs that are per-<<def_worktree,worktree>>, rather than
494-	global.  This is presently only <<def_HEAD,HEAD>> and any refs
495-	that start with `refs/bisect/`, but might later include other
496-	unusual refs.
497-
498-[[def_pseudoref]]pseudoref::
499-	A ref that has different semantics than normal refs. These refs can be
500-	read via normal Git commands, but cannot be written to by commands like
501-	linkgit:git-update-ref[1].
502-+
503-The following pseudorefs are known to Git:
504-
505- - `FETCH_HEAD` is written by linkgit:git-fetch[1] or linkgit:git-pull[1]. It
506-   may refer to multiple object IDs. Each object ID is annotated with metadata
507-   indicating where it was fetched from and its fetch status.
508-
509- - `MERGE_HEAD` is written by linkgit:git-merge[1] when resolving merge

 succeeded in 0ms:
usage: git status [<options>] [--] [<pathspec>...]

    -v, --[no-]verbose    be verbose
    -s, --[no-]short      show status concisely
    -b, --[no-]branch     show branch information
    --[no-]show-stash     show stash information
    --[no-]ahead-behind   compute full ahead/behind values
    --[no-]porcelain[=<version>]
                          machine-readable output
    --[no-]long           show status in long format (default)
    -z, --[no-]null       terminate entries with NUL
    -u, --[no-]untracked-files[=<mode>]
                          show untracked files, optional modes: all, normal, no. (Default: all)
    --[no-]ignored[=<mode>]
                          show ignored files, optional modes: traditional, matching, no. (Default: traditional)
    --[no-]ignore-submodules[=<when>]
                          ignore changes to submodules, optional when: all, dirty, untracked. (Default: all)
    --[no-]column[=<style>]
                          list untracked files in columns
    --no-renames          do not detect renames
    --renames             opposite of --no-renames
    -M, --find-renames[=<n>]
                          detect renames, optionally set similarity index

usage: git diff [<options>] [<commit>] [--] [<path>...]
   or: git diff [<options>] --cached [--merge-base] [<commit>] [--] [<path>...]
   or: git diff [<options>] [--merge-base] <commit> [<commit>...] <commit> [--] [<path>...]
   or: git diff [<options>] <commit>...<commit> [--] [<path>...]
   or: git diff [<options>] <blob> <blob>
   or: git diff [<options>] --no-index [--] <path> <path> [<pathspec>...]

common diff options:
  -z            output diff-raw with lines terminated with NUL.
  -p            output patch format.
  -u            synonym for -p.
  --patch-with-raw
                output both a patch and the diff-raw format.
  --stat        show diffstat instead of patch.
  --numstat     show numeric diffstat instead of patch.
  --patch-with-stat
                output a patch and prepend its diffstat.
  --name-only   show only names of changed files.
  --name-status show names and status of changed files.
  --full-index  show full object name on index lines.
  --abbrev=<n>  abbreviate object names in diff-tree header and diff-raw.
  -R            swap input file pairs.
  -B            detect complete rewrites.
  -M            detect renames.
  -C            detect copies.
  --find-copies-harder
                try unchanged files as candidate for copy detection.
  -l<n>         limit rename attempts up to <n> paths.
  -O<file>      reorder diffs according to the <file>.
  -S<string>    find filepair whose only one side contains the string.
  --pickaxe-all
                show all files diff when -S is used and hit is found.
  -a  --text    treat all files as text.


exec
/usr/bin/bash -lc "grep -nE '(train.py|torchrun|PYTHONPATH|pytest|src/tests|python )' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch | sed -n '1,260p'; grep -nE '(DATASET_CONFIG|dataset.config|AR_)' worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch | sed -n '1,220p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
38:# terminates the torchrun process group the moment Lightning reports the WRONG
41:# torchrun: PL 2.1.0 elects TorchElastic before SLURMEnvironment, so the ranks
42:# torchrun starts are used as-is; the SLURM rank variables are unset so
43:# SLURMEnvironment cannot claim the job. train.py is unmodified and rank-safe:
118:unset PYTHONPATH PYTHONOPTIMIZE
209:DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src ":(exclude)src/tests" data/AR \
217:# train.py, defaults.ini, src/, the data/AR split JSONs, the five arm
226:# src/tests is EXCLUDED from both gates: pytest-only code, never imported by
227:# train.py at runtime, and the TDD sessions land test files continuously —
229:# (legs 3680875-78 died on a src/tests-only commit, 2026-08-12).
231:  git diff --name-only "${EXPECT_SHA}" "${HEAD_SHA}" -- train.py defaults.ini src ":(exclude)src/tests" data/AR \
350:# --- F. the exact train.py argv ----------------------------------------------
422:print("--- train.py argv ---")
474:[ "$(readlink -f "$REAL_PY")" = "$(readlink -f "$PINNED_PYTHON")" ] || die "python resolves to ${REAL_PY}, pinned ${PINNED_PYTHON} - abort"
641:  echo "command torchrun --standalone --nnodes=1 --nproc_per_node=${NGPU} train.py ${ARGV[*]}"
648:# --- P. training: torchrun + FIFO tee (both statuses captured) ----------------
657:echo "=== launching ${ARM} ${RUNG}: torchrun --standalone --nproc_per_node=${NGPU} train.py (${MODE}) ==="
658:torchrun --standalone --nnodes=1 --nproc_per_node="$NGPU" train.py "${ARGV[@]}" > "$FIFO" 2>&1 &
668:        echo "WORLD-SIZE WATCHER: Lightning started with ${GOT} processes, expected ${NGPU} — terminating torchrun"
685:# train.py:165 builds WandbLogger(project=, name=) with NO save_dir, so PL passes
712:# save-dir too, so the durable record is not just torchrun's stdout.
715:  echo "torchrun rc=${rc} tee rc=${tee_rc} wandb_identity_rc=${WANDB_CHECK_RC} classified rc=${final_rc}"
211:          "$EXP07REL/FLAC_AR_BF.json" 2>&1)" \
220:# invokes, and exp_07's FLAC_AR_BF.json (C4L parity baseline).
232:      "$EXPDIR"/FLAC_AR_BF_C4L.json "$EXPDIR"/FLAC_AR_BF_C8.json \
233:      "$EXPDIR"/FLAC_AR_BF_C16.json "$EXPDIR"/FLAC_AR_BF_C32.json \
234:      "$EXPDIR"/FLAC_AR_VANCKPT.json "$EXPDIR"/fa_orbit_train.sbatch \
237:      "$EXP07/FLAC_AR_BF.json"
274:    C4L|C8|C16|C32) echo "$EXPDIR/FLAC_AR_BF_$1.json" ;;
275:    VANL)           echo "$EXPDIR/FLAC_AR_VANCKPT.json" ;;
353:  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
358:  --val-every -1 --val-dataset-config ''
371:REF = """--model-config worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json
372:--dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
384:               "--val-dataset-config": "", "--gradient-clip-val": "0.0", "--ckpt-path": None}

exec
/usr/bin/bash -lc "git grep -n -I -E '("'^|[''^[:alnum:]_])(src[.]tests|from[[:space:]]+tests([[:space:].]|$)|import[[:space:]]+tests([[:space:].]|$)|src/tests)'"' HEAD -- ':"'!src/tests/**'"' ':"'!worklog/**'"' ':"'!*.md'"'; git grep -n -I -E '(custom_metadata_module|dataset_config)' -- worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch worklog/worklog_yixun/exp_11_fa_orbit_claude/*.json data/AR/*.json" in /n/fs/gatrdp/codespace/FLAC
exec
/usr/bin/bash -lc "git log --format='%H %s' --name-only -n 80 | awk 'BEGIN{RS=\"\\n\\n\"} /src\\/tests\\// {print "'$0 "'"\\n---\"}' | sed -n '1,320p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 208ms:
src/tests/test_exp14_validate_cell.py
src/tests/test_yaw_gen_collect.py
worklog/worklog_yixun/exp_14_yaw_gen_claude/exp14_validate_cell.py
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-12_r3fix4_battery.log
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-12_r3fix4_guardtests.log
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-12_r3fix4_red_realartifact.log
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect.py
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect_selftest_blocked.txt
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect_selftest_complete.txt
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect_selftest_gatefail.txt
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect_selftest_pending.txt
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_params_set_up.md
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen.sbatch
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh
58b030faf5a0d80594b35b2a3b709993a653883e exp_14: campaign pin 16fe13d (reviewed READY-TO-PIN state; remotely reachable via exp14-campaign-pin branch)
---
src/tests/test_yaw_gen_collect.py
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_r3fix3_red_FX5.log
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect.py
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect_selftest_blocked.txt
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect_selftest_complete.txt
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect_selftest_gatefail.txt
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect_selftest_pending.txt
e36c7bec9f101c7ff434689500c0fc15f06a7622 exp_14 r3-fix3: FX4 — a stray OUTPUT_ROOT cannot split classification from execution
---
src/tests/test_exp14_validate_cell.py
worklog/worklog_yixun/exp_14_yaw_gen_claude/exp14_validate_cell.py
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh
3ee16bb458dc462e317ddea31a2d125eb92e3ed8 exp_14 r3-fix3: FX2 — the two-K transaction consumes validation, not file counts
---
src/tests/test_gen_model_comparison_gate.py
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_r3fix3_red_FX2.log
worklog/worklog_yixun/gen_model_comparison.py
a2ad7ab18139f0eb34a3cc90613bffcf76aeebcb exp_14 r3-fix3: FX1 — per-scene lifting is genuinely opt-in
---
eval_FLAC.py
src/tests/test_exp14_fixed_mode_snapshot.py
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_r3fix3_red_FX1.log
87404b1bb43fc911da92886566c8778081ee0a9d exp_14 r3-fix2: ledger entry for the aggregation ruling
---
src/tests/test_yaw_gen_collect.py
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_r3fix2_battery.log
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_r3fix2_red.log
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect.py
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect_selftest_blocked.txt
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect_selftest_complete.txt
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect_selftest_gatefail.txt
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect_selftest_pending.txt
e82c54bf0abe270dc2ca155d4b105931418f0389 exp_14 r3-fix: ledger, batteries, regenerated transcripts
---
src/tests/test_gen_model_comparison_gate.py
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_r3fix_red_R3F2345.log
worklog/worklog_yixun/gen_model_comparison.py
ce9003a79eae40d29540cb3ff7c9a499fecc9c42 standing rules (Yixun 2026-08-11): status+ETD for ALL experiments every response; ask before choosing when confirmation is wanted. A5 v2 artifacts.
---
src/tests/test_yaw_gen_collect.py
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_r3fix_red_R3F1c.log
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect.py
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect_selftest_blocked.txt
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect_selftest_complete.txt
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect_selftest_gatefail.txt
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect_selftest_pending.txt
1ef87b02a0e9fcea8cd10d03ebb6756ed018a4d1 exp_14 r3-fix: R3F1b — the kit records per-scene results and the validator demands them
---
src/tests/test_exp14_validate_cell.py
worklog/worklog_yixun/exp_14_yaw_gen_claude/exp14_validate_cell.py
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_19-59-50_screen_guardtests.log
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_r3fix_guardtests.log
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_r3fix_red_R3F1b.log
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen.sbatch
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh
32581d91187cf57eed8b89286e1a71034dc51ac1 exp_14 r3-fix: R3F1a — eval_FLAC records per-scene metrics on request
---
eval_FLAC.py
src/tests/test_yaw_random_eval.py
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_r3fix_red_R3F1a.log
c745be877ff81ce42ff928ce8f996bec86239dc7 exp_14 r3: ledger completion — last two SHAs, commit-boundary disclosure
---
src/tests/test_yaw_gen_collect.py
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_r3_final.log
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_r3_red_g4.log
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect.py
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect_selftest_blocked.txt
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect_selftest_complete.txt
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect_selftest_gatefail.txt
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect_selftest_pending.txt
608303e28185d88186f87be75199f1d487819d16 canonical table: exp_06 max_linear rows (8, @40k/@67.5k x online/EMA x K) + restore exp_03/04 rows (16 globs regressed to suffix-less in a refactor; raws unchanged); exp06 raws imported w/ sha manifest
---
src/tests/test_gen_model_comparison_gate.py
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_r3_red_table.log
worklog/worklog_yixun/gen_model_comparison.py
f8747586fb014d699b42a0957a578bf2e27f52c5 exp_14 r3: readouts, rendering, CLI + four self-test transcripts
---
src/tests/test_yaw_gen_collect.py
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_r3_red_group3.log
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect.py
170aeee148fba7558c36c8f21423b01f93e6b5e6 exp_14 r3: the §4 estimation conventions — paired-t, Holm, directions
---
src/tests/test_yaw_gen_collect.py
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_r3_red_group2.log
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect.py
24fea1666127b1c6bf3113d237a24867b86e5c56 exp_14 r3: collector core — artifacts, provenance, §3.3 equalities, blocks
---
src/tests/test_yaw_gen_collect.py
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-11_r3_red_group1.log
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect.py
5d4951e5099e72bb2321b9a48797c98144938500 exp_14 docs: ROUND 2 CLOSED — full review trail (6 checks), ledger through d200e1b, issue_report leg-binding note
---
src/tests/test_yaw_aug_record_control.py
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-56-28_pytest_r2fix.log
ae7a924dff5116de5e6d1f0663a70d61d39af33d exp_15 r2-fix: regenerate the admission record under the corrected recorder
---
src/tests/test_yaw_aug_arm_config.py
src/tests/test_yaw_aug_record_control.py
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-53-02_pytest_r2fix.log
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-53-32_pytest_r2fix.log
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-54-10_pytest_r2fix.log
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-54-50_pytest_r2fix.log
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_record_control.py
a3719f921c0a9d6d3ed6f115208657f90fcc93c9 exp_15 r2-fix: type-strict config and step comparison (finding 3, MAJOR)
---
src/tests/test_yaw_aug_record_control.py
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-51-23_pytest_r2fix.log
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-52-21_pytest_r2fix.log
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_record_control.py
7e7c512c2c9eaba9ad31bf35f160141dd5671cd6 exp_15 r2-fix: the EMA family must MIRROR the online DiT (finding 2, MAJOR)
---
src/tests/test_yaw_aug_record_control.py
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-49-46_pytest_r2fix.log
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-50-40_pytest_r2fix.log
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_record_control.py
30d5065774bf1bbc18d5d764cc3ab8eb3cd8f0b8 exp_15 r2-fix: one stable, safely loaded snapshot (review finding 1, BLOCKING)
---
src/tests/test_yaw_aug_record_control.py
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-47-17_pytest_r2fix.log
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-48-19_pytest_r2fix.log
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-48-42_pytest_r2fix.log
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_record_control.py
54b5a0c3ebfdea2bec40a8164a1af5bb3174fb71 exp_14 docs: worklog through r2 fix batch, r2 review artifact, ledger
---
src/tests/test_yaw_aug_record_control.py
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-26-15_pytest_r2.log
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-27-04_pytest_r2.log
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_record_control.py
d2c11db3f09f79ba5b26a94755721ac4ed94d325 exp_15 r2: yaw_aug_record_control.py — the control-admission recorder (TDD)
---
src/tests/test_yaw_aug_record_control.py
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-22-50_pytest_r2.log
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-23-50_pytest_r2.log
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_record_control.py
075b7871f605b61dab4d8d522899c44b799a08a3 exp_15 r2: FLAC_AR_YAWAUG.json — the control's config plus exactly one block
---
src/tests/test_yaw_aug_arm_config.py
worklog/worklog_yixun/exp_15_yaw_aug_claude/FLAC_AR_YAWAUG.json
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-20-39_pytest_r2.log
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-21-13_pytest_r2.log
a201e3c39bea2ae33c03f43530a2604b5fdaec5d exp_14 r2-fix: FB3 + FB5 + FB7 + FB8a/b — the submitters' safety rails
---
src/tests/test_exp14_validate_cell.py
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen.sbatch
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh
58d0d631b45d4510fd1e450389299c005724cb32 exp_15 r1: Codex review (REVISE→fixed) + worklog round entries — round 1 closed
---
src/tests/test_exp14_validate_cell.py
worklog/worklog_yixun/exp_14_yaw_gen_claude/exp14_validate_cell.py
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_submit.sh
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_submit_grid.sh
aa3a66fb0e0290cb6b7a69f5b0b251ff40f6b012 exp_15 r1-fix: make the golden-writer guard test unable to corrupt the fixture
---
src/tests/test_yaw_aug_training.py
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-12-55_pytest_r1fix.log
06b66b0a7c5f330f78d3611af1e4375ceaca55c3 exp_14 r2-fix: FB4 — a dedup SKIP now rests on checkpoint identity
---
src/tests/test_exp14_validate_cell.py
worklog/worklog_yixun/exp_14_yaw_gen_claude/exp14_ckpt_expect.json
worklog/worklog_yixun/exp_14_yaw_gen_claude/exp14_hash_ckpts.py
worklog/worklog_yixun/exp_14_yaw_gen_claude/exp14_validate_cell.py
a10e40971ed74071ed37f29e0572d3f489226643 exp_15 r1-fix: flush the banner, stop coercing ctor args, demand a capture SHA
---
src/tests/test_yaw_aug_training.py
src/training/diffusion.py
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-10-00_pytest_r1fix.log
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-11-16_pytest_r1fix.log
8a332340498bc66250afb8a1a72605451f44c3b6 exp_15 r1-fix: require all four pose fields in the schema guard (finding 2, MAJOR)
---
src/tests/test_yaw_aug_training.py
src/training/diffusion.py
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-07-53_pytest_r1fix.log
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-08-43_pytest_r1fix.log
076e11acdf6173a074b13e556ee2d5e6b813afb9 exp_15 r1-fix: keyed 32-bit bijection for the yaw seed (review finding 1, MAJOR)
---
src/tests/test_yaw_aug_training.py
src/training/diffusion.py
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-05-38_pytest_r1fix.log
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_13-06-41_pytest_r1fix.log
98e276ce732cdd7c8ae0c23629e3bdc819f80053 exp_14 r2-fix: FB6 — six ways a VALID verdict meant "not checked"
---
src/tests/test_exp14_validate_cell.py
worklog/worklog_yixun/exp_14_yaw_gen_claude/exp14_validate_cell.py
5d6e349ce9e276da092f029acdd78dfd77eb0d5a exp_14 r2-fix: FB1 — the --rotate-deg 0 that would have failed 100 of 106 cells
---
src/tests/test_exp14_validate_cell.py
worklog/worklog_yixun/exp_14_yaw_gen_claude/exp14_validate_cell.py
worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen.sbatch
28f66d571bbcaa5a8daba016e7caf23f492dd47d exp_15 r1: round-1 verification logs — 397 passed across the affected suites
---
src/tests/test_yaw_aug_training.py
src/training/diffusion.py
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_12-38-25_pytest_r1.log
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_12-39-38_pytest_r1.log
worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_12-40-24_pytest_r1.log
05e6c6d240f99090403f36c47cdc10cd51088587 exp_14 r2: guard suite — the exp_14 contract, end to end (162 cases)
---

 succeeded in 639ms:
HEAD:src/tools/retune_lr_state.py:20:swap is a **silent no-op**.  ``src/tests/test_retune_lr_state.py`` pins that
HEAD:src/tools/strip_optimizer_state.py:41:   5e-6 case, by ``src/tests/test_finetune_cond.py:664``).
HEAD:tools/__init__.py:6:divergent sibling checkout (see ``src/tests/conftest.py`` for the same guard).
HEAD:tools/bn_drift_probe.py:10:Public surface (unit-pinned in ``src/tests/test_bn_drift_probe.py``):
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:353:  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:372:--dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json

exec
/usr/bin/bash -lc 'command -v git; PATH=/n/fs/gatrdp/envs/flac/bin:$PATH command -v git; PATH=/n/fs/gatrdp/envs/flac/bin:$PATH git --version; if [ -x /n/fs/gatrdp/envs/flac/bin/git ]; then /n/fs/gatrdp/envs/flac/bin/git --version; fi' in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
/usr/bin/git
/usr/bin/git
git version 2.52.0

exec
/usr/bin/bash -lc 'git diff --unified=80 -- worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh' in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
diff --git a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
index f039a3c..185d911 100644
--- a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
+++ b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
@@ -129,179 +129,183 @@ SMOKE="${SMOKE:-0}"
 PRODUCTION_OUTPUT_ROOT="outputs_FLAC"
 if [ -n "${SLURM_JOB_ID:-}" ]; then
   if [ -n "${OUTPUT_ROOT:-}" ] && [ "$OUTPUT_ROOT" != "$PRODUCTION_OUTPUT_ROOT" ]; then
     echo "ambient OUTPUT_ROOT='${OUTPUT_ROOT}' != the production literal '${PRODUCTION_OUTPUT_ROOT}' - abort"; exit 2
   fi
   OUTPUT_ROOT="$PRODUCTION_OUTPUT_ROOT"
 else
   OUTPUT_ROOT="${OUTPUT_ROOT:-$PRODUCTION_OUTPUT_ROOT}"
 fi
 RESUME_CKPT="${RESUME_CKPT:-}"
 EXPECTED_STEP="${EXPECTED_STEP:-0}"
 TS="$(date '+%Y-%m-%d_%H-%M-%S')"
 
 die() { echo "$1"; exit "${2:-2}"; }
 
 # --- A. parameters ------------------------------------------------------------
 [ -n "${ARM:-}" ] || die "ARM must be exported (C4L|C8|C16|C32|VANL) - abort"
 [ -n "${EXPECT_SHA:-}" ] || die "EXPECT_SHA (full reviewed commit sha) must be exported - abort"
 case "$ARM" in
   C4L|C8|C16|C32|VANL) ;;
   *) die "ARM '${ARM}' is not a legal exp_11 arm — C4L|C8|C16|C32 only (FA1/VAN/CKPT4 are P0 profiling cells, never arms) - abort" ;;
 esac
 case "$EXPECTED_STEP" in ''|*[!0-9]*) die "EXPECTED_STEP '${EXPECTED_STEP}' must be a non-negative integer - abort";; esac
 
 # --- B. the pins decide the recipe (round-3 B1) -------------------------------
 if [ "$SMOKE" = "1" ]; then
   RUNG="${SMOKE_RUNG:-}"; MAXSTEPS="${SMOKE_MAXSTEPS:-30}"; MIN_FREE_MB="${SMOKE_MIN_FREE_MB:-}"
   CHECKPOINT_EVERY="${SMOKE_CHECKPOINT_EVERY:-10}"
   [ -n "$RUNG" ] || die "SMOKE=1 requires SMOKE_RUNG (32x2|16x4|8x8) - abort"
   [ -n "$MIN_FREE_MB" ] || die "SMOKE=1 requires SMOKE_MIN_FREE_MB (per-GPU floor) - abort"
   TIME_LIMIT="${SMOKE_TIME:-00:30:00}"; TIME_PIN_NAME="SMOKE_TIME"
   NAME="FLAC_exp11_smoke_${ARM}"; EXPNAME="exp11_smoke_${ARM}"
   SAVEDIR="${OUTPUT_ROOT}/exp11_smoke/${ARM}"
   echo "=== SMOKE MODE: pins bypassed, EVERY other gate active; identity ${EXPNAME} ==="
 else
   # Q10 / re-pin fix 1: the wall pin follows the LEG, not the arm. A restart leg
   # is 60,000 further steps, not 40,000 from scratch, so the submitter allocates
   # PINNED_TIME_LIMIT_RESTART_<ARM>. The job selected PINNED_TIME_LIMIT_<ARM>
   # regardless and then rejected its own (correct) allocation in gate H — the
   # third hard-abort path the re-pin review found on jobs 3662828-30. The JOB now
   # selects the same pin the submitter did and enforces THAT one.
   if [ "$EXPECTED_STEP" -gt 0 ]; then
     TIME_PIN_NAME="PINNED_TIME_LIMIT_RESTART_${ARM}"
   else
     TIME_PIN_NAME="PINNED_TIME_LIMIT_${ARM}"
   fi
   for PIN_NAME in PINNED_RUNG PINNED_MB PINNED_NGPU PINNED_MIN_FREE_MB PINNED_P0_MANIFEST_SHA256 \
                   "$TIME_PIN_NAME"; do
     eval "PIN_VAL=\${$PIN_NAME}"
     [ "$PIN_VAL" != "$PIN_PLACEHOLDER" ] || die "${PIN_NAME} is still '${PIN_PLACEHOLDER}': the P0 report has not been pinned into this launcher yet — no arm may launch (use SMOKE=1 for the pre-launch smoke) - abort"
   done
   RUNG="$PINNED_RUNG"; MAXSTEPS="$PINNED_MAXSTEPS"; MIN_FREE_MB="$PINNED_MIN_FREE_MB"
   CHECKPOINT_EVERY="$PINNED_CHECKPOINT_EVERY"
   eval "TIME_LIMIT=\${${TIME_PIN_NAME}}"
   NAME="FLAC_exp11_${ARM}"; EXPNAME="exp11_${ARM}"; SAVEDIR="${OUTPUT_ROOT}/exp11_${ARM}"
 fi
 
 case "$RUNG" in
   32x2|16x4|8x8) ;;
   *) die "rung '${RUNG}' must be 32x2, 16x4 or 8x8 - abort" ;;
 esac
 MB="${RUNG%x*}"; NGPU="${RUNG#*x}"
 [ "$((MB * NGPU))" -eq 64 ] || die "rung ${RUNG}: MB*NGPU = $((MB*NGPU)) != 64 (micro x N pin, plan §10) - abort"
 if [ "$SMOKE" != "1" ]; then
   [ "$MB" = "$PINNED_MB" ] && [ "$NGPU" = "$PINNED_NGPU" ] || die "pin inconsistency: rung ${RUNG} vs PINNED_MB=${PINNED_MB}/PINNED_NGPU=${PINNED_NGPU} - abort"
   [ "$MAXSTEPS" = "100000" ] || die "PINNED_MAXSTEPS is ${MAXSTEPS}, the registered budget is 100000 - abort"
 fi
 RUNDIR="${SAVEDIR}/${NAME}/${EXPNAME}"
 echo "=== exp_11 arm ${ARM} @ rung ${RUNG} (MB ${MB} x ${NGPU} GPU, grad-ckpt ON) — ${TS} — host $(hostname) ==="
 
 # --- C. commit binding + tracked-surface drift --------------------------------
 HEAD_SHA="$(git rev-parse HEAD 2>/dev/null)" || HEAD_SHA=""
 EXPREL="${EXPDIR#"$REPO"/}"; EXP07REL="${EXP07#"$REPO"/}"
 # The drift gate is scoped to CODE surfaces, not the whole exp folder: the four
 # arms are running and Slurm appends to their tracked *.out logs continuously, so
 # a folder-wide check would abort every screen on a live-log write. Configs,
 # drivers and validators are still fully covered. The patterns are QUOTED so
 # git, not the shell, expands them — a tracked file deleted from the worktree
 # still matches (content-gate review B2) — data/AR (the split JSONs the
 # dataloader opens) is covered, and a failing git status is fail-closed.
-DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src data/AR \
+DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src ":(exclude)src/tests" data/AR \
           "$EXPREL/*.json" "$EXPREL/*.py" "$EXPREL/*.sbatch" "$EXPREL/*.sh" \
           "$EXP07REL/FLAC_AR_BF.json" 2>&1)" \
   || die "git status for the drift gate failed: ${DRIFT} - abort"
 # Commit binding is CONTENT-scoped: HEAD identity is sufficient but not
 # necessary. Two sessions commit to this checkout, so a pending leg must
 # survive commits that leave the training closure untouched — and abort on
 # any commit that changes it. The closure is what the job actually loads:
 # train.py, defaults.ini, src/, the data/AR split JSONs, the five arm
 # configs (enumerated — a shell glob would silently drop a config deleted
 # since EXPECT_SHA), this launcher, the four runtime helper scripts it
 # invokes, and exp_07's FLAC_AR_BF.json (C4L parity baseline).
 # Record/analysis files (registry, manifests, gen_*/validators, worklog)
 # are deliberately OUTSIDE the closure. Fail-closed on every edge:
 # EXPECT_SHA must be the full 40-hex commit OID (a symbolic ref like HEAD
 # would defeat the binding), the diff runs against the CAPTURED HEAD OID,
 # and HEAD is re-read afterwards to close the mid-gate-commit race.
+# src/tests is EXCLUDED from both gates: pytest-only code, never imported by
+# train.py at runtime, and the TDD sessions land test files continuously —
+# gating them reproduced the exact kill this gate exists to prevent
+# (legs 3680875-78 died on a src/tests-only commit, 2026-08-12).
 surface_diff_vs_expect() {
-  git diff --name-only "${EXPECT_SHA}" "${HEAD_SHA}" -- train.py defaults.ini src data/AR \
+  git diff --name-only "${EXPECT_SHA}" "${HEAD_SHA}" -- train.py defaults.ini src ":(exclude)src/tests" data/AR \
       "$EXPDIR"/FLAC_AR_BF_C4L.json "$EXPDIR"/FLAC_AR_BF_C8.json \
       "$EXPDIR"/FLAC_AR_BF_C16.json "$EXPDIR"/FLAC_AR_BF_C32.json \
       "$EXPDIR"/FLAC_AR_VANCKPT.json "$EXPDIR"/fa_orbit_train.sbatch \
       "$EXPDIR"/fa_orbit_ckpt_preflight.py "$EXPDIR"/assert_arm_configs_exp11.py \
       "$EXPDIR"/fa_orbit_wandb_readback.py "$EXPDIR"/fa_orbit_classify.py \
       "$EXP07/FLAC_AR_BF.json"
 }
 GATE_FAIL=""; GATE_OK_MSG=""
 if [ -z "$HEAD_SHA" ]; then
   GATE_FAIL="cannot resolve HEAD"
 elif ! printf '%s\n' "$EXPECT_SHA" | grep -qE '^[0-9a-f]{40}$'; then
   GATE_FAIL="EXPECT_SHA '${EXPECT_SHA}' is not a full lowercase 40-hex commit id"
 elif [ "$(git rev-parse --verify -q "${EXPECT_SHA}^{commit}" 2>/dev/null)" != "$EXPECT_SHA" ]; then
   GATE_FAIL="EXPECT_SHA ${EXPECT_SHA} is not a commit known to this repo"
 elif [ "$HEAD_SHA" = "$EXPECT_SHA" ]; then
   GATE_OK_MSG="commit binding OK: ${HEAD_SHA}"
 elif SD="$(surface_diff_vs_expect 2>&1)"; then
   if [ -z "$SD" ]; then
     GATE_OK_MSG="commit binding OK (content): training surfaces identical, EXPECT_SHA ${EXPECT_SHA} HEAD ${HEAD_SHA}"
   else
     GATE_FAIL="training surfaces changed since EXPECT_SHA ${EXPECT_SHA} (HEAD ${HEAD_SHA}): ${SD}"
   fi
 else
   GATE_FAIL="surface diff vs EXPECT_SHA failed: ${SD}"
 fi
 if [ -z "$GATE_FAIL" ] && [ "$(git rev-parse HEAD 2>/dev/null)" != "$HEAD_SHA" ]; then
   GATE_FAIL="HEAD moved during the commit-binding check (was ${HEAD_SHA})"
 fi
 if [ "$DRYRUN" = "1" ]; then
   [ -z "$GATE_FAIL" ] && echo "${GATE_OK_MSG} (dry run)" \
     || echo "DRY-RUN ADVISORY: ${GATE_FAIL} (a real launch aborts here)"
   [ -z "$DRIFT" ] || echo "DRY-RUN ADVISORY: tracked measurement surfaces are modified (a real launch aborts here)"
 else
   [ -n "${SLURM_JOB_ID:-}" ] || die "a real launch must run under sbatch (no SLURM_JOB_ID) - abort"
   [ -z "$GATE_FAIL" ] || die "${GATE_FAIL} - abort"
   echo "$GATE_OK_MSG"
   [ -z "$DRIFT" ] || { echo "tracked measurement surfaces modified since review - abort:"; echo "$DRIFT"; exit 2; }
 fi
 
 # --- D. arm -> config (single source) + semantic gate -------------------------
 arm_config_for() {
   case "$1" in
     C4L|C8|C16|C32) echo "$EXPDIR/FLAC_AR_BF_$1.json" ;;
     VANL)           echo "$EXPDIR/FLAC_AR_VANCKPT.json" ;;
     *) return 1 ;;
   esac
 }
 MODEL_CONFIG="$(arm_config_for "$ARM")" || die "no config mapped for arm '${ARM}' - abort"
 MODEL_CONFIG_ABS="$(readlink -f "$MODEL_CONFIG" 2>/dev/null)"
 [ -n "$MODEL_CONFIG_ABS" ] && [ -f "$MODEL_CONFIG_ABS" ] || die "arm config '${MODEL_CONFIG}' does not exist - abort"
 CONFIG_SHA="$(sha256sum "$MODEL_CONFIG_ABS" | awk '{print $1}')"
 echo "config for ${ARM}: ${MODEL_CONFIG_ABS} sha256 ${CONFIG_SHA}"
 
 python3 - "$MODEL_CONFIG_ABS" "$ARM" <<'PY' || die "arm/config semantic gate FAILED - abort"
 import json, sys
 cfg = json.load(open(sys.argv[1])); arm = sys.argv[2]
 t = cfg.get("training", {}); bad = []
 # VANL is the same recipe with the conditioning removed, so its gate is the
 # MIRROR IMAGE of the orbit arms': the orbit keys must be ABSENT, not merely
 # different. A vanilla config that carried a stray frame_avg_angles would be a
 # silently fa-flavoured baseline, which would destroy the single-delta claim.
 if arm == "VANL":
     cm = t.get("cond_method")
     if cm not in (None, "vanilla"):
         bad.append(f"cond_method={cm!r} (want absent or 'vanilla')")
     if "frame_avg_angles" in t:
         bad.append(f"frame_avg_angles is present ({t['frame_avg_angles']!r}) — a vanilla arm has no orbit")
     want = None
 else:
     want = {"C4L": 4, "C8": 8, "C16": 16, "C32": 32}[arm]
     angles = t.get("frame_avg_angles")
     if t.get("cond_method") != "fa_invariant":
         bad.append(f"cond_method={t.get('cond_method')!r} (want fa_invariant)")
     if not isinstance(angles, list) or len(angles) != want:
         bad.append(f"frame_avg_angles has {angles and len(angles)} entries (want {want})")
     elif angles != [k * 360.0 / want for k in range(want)]:
         bad.append(f"frame_avg_angles are not the uniform C{want} orbit")
 if t.get("use_ema") is not True:
     bad.append(f"use_ema={t.get('use_ema')!r} (want True)")
 vits = [c for c in cfg["model"]["conditioning"]["configs"] if c["type"] == "ViTCoordinates"]
diff --git a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
index a7d0f5b..b487055 100755
--- a/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
+++ b/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
@@ -101,160 +101,174 @@ case_run "SMOKE needs a VRAM floor" 2 "SMOKE_MIN_FREE_MB" \
 case_run "SMOKE identity is separate" 0 "exp11_smoke_C8" -- "${SMOKE_ENV[@]}" ARM=C8
 
 echo "--- B. parameter / arm / rung gates ---"
 case_run "missing ARM" 2 "ARM" -- DRYRUN=1 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}"
 case_run "missing EXPECT_SHA" 2 "EXPECT_SHA" -- DRYRUN=1 ARM=C8 "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}"
 for BAD in C7 FA1 VAN CKPT4; do
   case_run "arm ${BAD} rejected" 2 "not a legal exp_11 arm" -- "${SMOKE_ENV[@]}" ARM=$BAD
 done
 case_run "bogus rung rejected" 2 "must be 32x2, 16x4 or 8x8" \
   -- DRYRUN=1 SMOKE=1 SMOKE_RUNG=64x1 SMOKE_MIN_FREE_MB=1 ARM=C8 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}"
 for R in 32x2 16x4 8x8; do   # all three rungs are feasible now that grad-ckpt is on
   case_run "rung ${R} accepted" 0 "ARGV PARITY OK" \
     -- DRYRUN=1 SMOKE=1 "SMOKE_RUNG=${R}" SMOKE_MIN_FREE_MB=14000 ARM=C4L "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}"
 done
 
 echo "--- C. lineage gates ---"
 : > "${TMP}/foreign.ckpt"
 case_run "initial + RESUME_CKPT" 2 "INITIAL launch must not carry" \
   -- "${SMOKE_ENV[@]}" ARM=C8 "RESUME_CKPT=${TMP}/foreign.ckpt"
 case_run "restart w/o ckpt" 2 "RESTART requires RESUME_CKPT" \
   -- "${SMOKE_ENV[@]}" ARM=C8 EXPECTED_STEP=5000
 case_run "restart ckpt missing" 2 "not found" \
   -- "${SMOKE_ENV[@]}" ARM=C8 EXPECTED_STEP=5000 "RESUME_CKPT=${TMP}/nope.ckpt"
 case_run "restart foreign ckpt" 2 "may only resume a checkpoint from" \
   -- "${SMOKE_ENV[@]}" ARM=C8 EXPECTED_STEP=5000 "RESUME_CKPT=${TMP}/foreign.ckpt"
 # a ckpt in the arm's own checkpoints dir but NOT named .ckpt / one level up
 SMOKE_RUN="${OUT_ROOT}/exp11_smoke/C8/FLAC_exp11_smoke_C8/exp11_smoke_C8"
 mkdir -p "${SMOKE_RUN}/checkpoints"
 : > "${SMOKE_RUN}/checkpoints/epoch=1-step=5000.ckpt"
 : > "${SMOKE_RUN}/notes.txt"
 case_run "restart from the arm's own ckpt dir" 0 "ARGV PARITY OK" \
   -- "${SMOKE_ENV[@]}" ARM=C8 EXPECTED_STEP=5000 SMOKE_MAXSTEPS=6000 \
      "RESUME_CKPT=${SMOKE_RUN}/checkpoints/epoch=1-step=5000.ckpt"
 case_run "restart from a non-ckpt sibling" 2 "may only resume a checkpoint from" \
   -- "${SMOKE_ENV[@]}" ARM=C8 EXPECTED_STEP=5000 SMOKE_MAXSTEPS=6000 "RESUME_CKPT=${SMOKE_RUN}/notes.txt"
 case_run "restart MAXSTEPS<=step" 2 "must exceed the resume step" \
   -- "${SMOKE_ENV[@]}" ARM=C8 EXPECTED_STEP=5000 SMOKE_MAXSTEPS=30 \
      "RESUME_CKPT=${SMOKE_RUN}/checkpoints/epoch=1-step=5000.ckpt"
 case_run "initial refuses an existing run dir" 2 "already exists" -- "${SMOKE_ENV[@]}" ARM=C8
 
 echo "--- D. commit-binding / sbatch-only gates (REAL mode) ---"
 # NOTE: no OUTPUT_ROOT here — under a (fake) SLURM_JOB_ID the launcher forces the
 # production literal, and the commit gate aborts long before anything is written.
 case_run "wrong EXPECT_SHA aborts" 2 "EXPECT_SHA" \
   -- ARM=C4L SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=99000000 \
      EXPECT_SHA=0000000000000000000000000000000000000000 SLURM_JOB_ID=999999
 case_run "real mode needs sbatch" 2 "must run under sbatch" \
   -- ARM=C4L SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=99000000 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}"
 
 # Content-scoped binding (content-gate review B5): deterministic SYNTHETIC
 # fixtures — dangling commits built with git plumbing. No ref moves, the
 # tracked tree is untouched (only unreferenced objects are written; gc prunes
 # them), and a missing fixture is a FAILURE, never a SKIP: the identical-tree
 # case is the proof that record-only commits cannot kill a queued leg.
 # The gate acceptance text is asserted; the run then aborts at a later gate
 # (dirty-tree drift today, run-dir/allocation gates on a clean tree) with
 # rc=2 and nothing written.
 SYN_SAME="$(git commit-tree "$(git rev-parse 'HEAD^{tree}')" -p HEAD -m 'guardtest synthetic: identical tree' 2>/dev/null)"
 SYN_IDX="${TMP}/synidx"; SYN_CHG=""
 if GIT_INDEX_FILE="$SYN_IDX" git read-tree HEAD 2>/dev/null; then
   SYN_BLOB="$(printf 'guardtest synthetic drift\n' | git hash-object -w --stdin 2>/dev/null)"
   if [ -n "$SYN_BLOB" ] && GIT_INDEX_FILE="$SYN_IDX" git update-index --cacheinfo 100644 "$SYN_BLOB" train.py 2>/dev/null; then
     SYN_TREE="$(GIT_INDEX_FILE="$SYN_IDX" git write-tree 2>/dev/null)"
     [ -n "$SYN_TREE" ] && SYN_CHG="$(git commit-tree "$SYN_TREE" -p HEAD -m 'guardtest synthetic: train.py changed' 2>/dev/null)"
   fi
 fi
 if [ -n "$SYN_SAME" ]; then
   case_run "moved HEAD, surfaces identical -> gate passes" 2 "commit binding OK (content)" \
     -- ARM=C4L SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=99000000 "EXPECT_SHA=${SYN_SAME}" SLURM_JOB_ID=999999
 else
   echo "FAIL  could not synthesize the identical-tree fixture"; FAIL=$((FAIL+1))
 fi
 if [ -n "$SYN_CHG" ]; then
   case_run "moved HEAD, surfaces changed -> aborts" 2 "training surfaces changed since EXPECT_SHA" \
     -- ARM=C4L SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=99000000 "EXPECT_SHA=${SYN_CHG}" SLURM_JOB_ID=999999
 else
   echo "FAIL  could not synthesize the changed-surface fixture"; FAIL=$((FAIL+1))
 fi
 case_run "symbolic EXPECT_SHA refused" 2 "not a full lowercase 40-hex" \
   -- ARM=C4L SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=99000000 EXPECT_SHA=HEAD SLURM_JOB_ID=999999
+# src/tests is excluded from the closure (pytest-only; TDD sessions land tests
+# continuously — a test-only commit must NOT kill a queued leg: 3680875-78).
+SYN_TIDX="${TMP}/syntidx"; SYN_TESTS=""
+if GIT_INDEX_FILE="$SYN_TIDX" git read-tree HEAD 2>/dev/null && [ -n "${SYN_BLOB:-}" ] \
+   && GIT_INDEX_FILE="$SYN_TIDX" git update-index --add --cacheinfo 100644 "$SYN_BLOB" src/tests/test_guardtest_synthetic.py 2>/dev/null; then
+  SYN_TTREE="$(GIT_INDEX_FILE="$SYN_TIDX" git write-tree 2>/dev/null)"
+  [ -n "$SYN_TTREE" ] && SYN_TESTS="$(git commit-tree "$SYN_TTREE" -p HEAD -m 'guardtest synthetic: src/tests only' 2>/dev/null)"
+fi
+if [ -n "$SYN_TESTS" ]; then
+  case_run "src/tests-only change -> gate passes" 2 "commit binding OK (content)" \
+    -- ARM=C4L SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=99000000 "EXPECT_SHA=${SYN_TESTS}" SLURM_JOB_ID=999999
+else
+  echo "FAIL  could not synthesize the src/tests-only fixture"; FAIL=$((FAIL+1))
+fi
 
 echo "--- E. semantic gate on a mislabelled config (temp copy; tracked tree untouched) ---"
 FAKE_EXP="${TMP}/fakeexp"; mkdir -p "$FAKE_EXP"
 cp "${EXPDIR}/FLAC_AR_BF_C4L.json" "${FAKE_EXP}/FLAC_AR_BF_C32.json"      # C4 orbit under the C32 name
 expect_cmd "orbit mismatch rejected" 1 "ARM/CONFIG GATE" -- \
   $PY - "${FAKE_EXP}/FLAC_AR_BF_C32.json" C32 <<'PY'
 import json, sys
 cfg = json.load(open(sys.argv[1])); arm = sys.argv[2]
 t = cfg.get("training", {}); bad = []
 want = {"C4L": 4, "C8": 8, "C16": 16, "C32": 32}[arm]
 angles = t.get("frame_avg_angles")
 if not isinstance(angles, list) or len(angles) != want:
     bad.append(f"frame_avg_angles has {angles and len(angles)} entries (want {want})")
 if bad:
     sys.exit("ARM/CONFIG GATE: " + "; ".join(bad))
 PY
 TRACKED_AFTER="$(git status --porcelain -- "$EXPDIR" src | sort)"
 if [ "$TRACKED_BEFORE" = "$TRACKED_AFTER" ]; then
   echo "PASS  tracked tree unchanged by the suite (snapshot before == after)"; PASS=$((PASS+1))
 else
   echo "FAIL  the suite changed tracked state:"; diff <(echo "$TRACKED_BEFORE") <(echo "$TRACKED_AFTER") | sed 's/^/        | /'
   FAIL=$((FAIL+1))
 fi
 
 echo "--- F. exit taxonomy, mocked (round-3 B5) ---"
 mk_log() {  # $1 dest, $2 world size (0 = absent), $3 marker?, $4 oom?
   : > "$1"
   [ "$2" != "0" ] && echo "All distributed processes registered. Starting with $2 processes" >> "$1"
   [ "$3" = "yes" ] && echo '`Trainer.fit` stopped: `max_steps=40000` reached.' >> "$1"
   [ "$4" = "yes" ] && echo "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 98.00 MiB" >> "$1"
   return 0
 }
 A="${TMP}/a.log"; B="${TMP}/b.log"
 mk_log "$A" 4 yes no; cp "$A" "$B"
 expect_cmd "class 0 complete" 0 "COMPLETE" -- $PY "$CLASSIFY" --rc 0 --tee-rc 0 --ngpu 4 --maxsteps 40000 --log "$A" --log-copy "$B"
 mk_log "$A" 0 no no; cp "$A" "$B"
 expect_cmd "class 6 world-size absent" 6 "WORLD-SIZE" -- $PY "$CLASSIFY" --rc 0 --tee-rc 0 --ngpu 4 --maxsteps 40000 --log "$A" --log-copy "$B"
 mk_log "$A" 1 yes no; cp "$A" "$B"
 expect_cmd "class 6 wrong world-size" 6 "reported [1]" -- $PY "$CLASSIFY" --rc 0 --tee-rc 0 --ngpu 4 --maxsteps 40000 --log "$A" --log-copy "$B"
 mk_log "$A" 4 no yes; cp "$A" "$B"
 expect_cmd "class 3 OOM on nonzero rc" 3 "OOM" -- $PY "$CLASSIFY" --rc 1 --tee-rc 0 --ngpu 4 --maxsteps 40000 --log "$A" --log-copy "$B"
 mk_log "$A" 4 no no; cp "$A" "$B"
 expect_cmd "class 4 missing marker" 4 "NO-MARKER" -- $PY "$CLASSIFY" --rc 0 --tee-rc 0 --ngpu 4 --maxsteps 40000 --log "$A" --log-copy "$B"
 mk_log "$A" 4 yes no; cp "$A" "$B"; echo "divergent tail" >> "$B"
 expect_cmd "class 7 logs differ" 7 "LOG-PROVENANCE" -- $PY "$CLASSIFY" --rc 0 --tee-rc 0 --ngpu 4 --maxsteps 40000 --log "$A" --log-copy "$B"
 mk_log "$A" 4 yes no; cp "$A" "$B"; rm -f "$B"
 expect_cmd "class 7 copy missing" 7 "missing log copy" -- $PY "$CLASSIFY" --rc 0 --tee-rc 0 --ngpu 4 --maxsteps 40000 --log "$A" --log-copy "$B"
 mk_log "$A" 4 yes no; cp "$A" "$B"
 expect_cmd "class 7 tee failed" 7 "tee exited" -- $PY "$CLASSIFY" --rc 0 --tee-rc 1 --ngpu 4 --maxsteps 40000 --log "$A" --log-copy "$B"
 mk_log "$A" 4 no no; cp "$A" "$B"
 expect_cmd "raw rc preserved" 9 "RUNTIME" -- $PY "$CLASSIFY" --rc 9 --tee-rc 0 --ngpu 4 --maxsteps 40000 --log "$A" --log-copy "$B"
 
 echo "--- G. restart preflight depth, mocked checkpoints (round-3 B2) ---"
 $PY - "$TMP" "${EXPDIR}/FLAC_AR_BF_C8.json" <<'PY'
 import json, os, sys, torch
 tmp, cfg_path = sys.argv[1], sys.argv[2]
 cfg = json.load(open(cfg_path))
 def ck(step=5000, config=cfg, opt=True, sched=True, ema=True):
     d = {"global_step": step, "epoch": 1, "model_config": config,
          "state_dict": {"diffusion.x": torch.zeros(1)},
          "optimizer_states": [{"state": {0: {"step": 1}} if opt else {},
                                "param_groups": [{"lr": 1e-5}]}],
          "lr_schedulers": [{"last_epoch": step}] if sched else []}
     if ema:
         d["state_dict"]["diffusion_ema.x"] = torch.zeros(1)
     return d
 torch.save(ck(), os.path.join(tmp, "good.ckpt"))
 torch.save(ck(step=4999), os.path.join(tmp, "wrongstep.ckpt"))
 c4 = json.loads(json.dumps(cfg)); c4["training"]["frame_avg_angles"] = [0.0, 90.0, 180.0, 270.0]
 torch.save(ck(config=c4), os.path.join(tmp, "wrongorbit.ckpt"))
 torch.save(ck(opt=False), os.path.join(tmp, "stripped.ckpt"))
 torch.save(ck(ema=False), os.path.join(tmp, "noema.ckpt"))
 torch.save(ck(sched=False), os.path.join(tmp, "nosched.ckpt"))
 torch.save(ck(step=45000), os.path.join(tmp, "past.ckpt"))
 open(os.path.join(tmp, "empty.ckpt"), "wb").close()
 print("synthetic checkpoints written")
 PY
 PRE=($PY "$PREFLIGHT" --config "${EXPDIR}/FLAC_AR_BF_C8.json" --max-steps 40000 --arm C8 --rung 16x4)
 expect_cmd "preflight accepts a good ckpt" 0 "CKPT_SHA256" -- "${PRE[@]}" --ckpt "${TMP}/good.ckpt" --expected-step 5000
 expect_cmd "preflight rejects a step mismatch" 2 "global_step" -- "${PRE[@]}" --ckpt "${TMP}/wrongstep.ckpt" --expected-step 5000

exec
/usr/bin/bash -lc "sed -n '1,240p' src/configs/dataset_configs/AR/train/acousticroom_train.json; git grep -n -I -E '(src/tests|src[.]tests|/tests/|conftest[.]py)' -- train.py defaults.ini src ':"'!src/tests/**'"' data/AR worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_BF_C4L.json worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_BF_C8.json worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_BF_C16.json worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_BF_C32.json worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_VANCKPT.json worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
{
    "dataset_type": "audio_dir",
    "datasets": [
        {
            "id": "AcousticRooms",
            "path": "AcousticRooms",
            "json_file_path": "data/AR/train.json",
            "custom_metadata_module": "src/configs/dataset_configs/custom_metadata/AR_md.py",
            "folder_name": "single_channel_ir_1"
        }

    ],
    "random_crop": false,
    "augs": true,
    "force_channels": "mono", 

    "modalities": {
        "acoustic_context": {
            "load": true,
            "max_context": 8, 
            "max_len": 9600 
        }, 
        "depth": {
            "load": true
        }, 
        "poses": {
            "load": true
        }
    }  
}src/tools/retune_lr_state.py:20:swap is a **silent no-op**.  ``src/tests/test_retune_lr_state.py`` pins that
src/tools/strip_optimizer_state.py:41:   5e-6 case, by ``src/tests/test_finetune_cond.py:664``).

exec
/usr/bin/bash -lc 'git diff --check -- worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh; bash -n worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch; bash -n worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh' in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
exec
/usr/bin/bash -lc "git log --format='%H %P %s' --name-only -n 40 -- src/tests | awk 'BEGIN{RS=\"\\n\\n\"} {n=split("'$0,a,"'"\\n\"); only=1; for(i=2;i<=n;i++){if(a[i]"'!="" && a[i]!~ /''^src'"\\/tests\\//) only=0} if(only) print "'$0 "'"\\n---\"}' | sed -n '1,260p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 4ms:
3c2bcbee95845f6934da43bc6c815a27086fecdf 58b030faf5a0d80594b35b2a3b709993a653883e exp_14 r3-fix4: per-scene means 10 ROOM FAMILIES, not 17 rooms
---
src/tests/test_yaw_random_eval.py

---

codex
The runtime audit is clean: `train.py`, `src` package initializers, the selected AR dataset config, and the one dynamic metadata loader have no route into `src/tests`; `conftest.py` is pytest-discovered only and its sole effect is a pytest-time `sys.path` insertion. The synthetic child commit also uses a fresh alternate index, so it does not inherit the earlier `train.py` mutation.
exec
/usr/bin/bash -lc "git status --short --ignored --untracked-files=all -- src | grep '"'^!! src/tests/'"' | sed -n '1,40p'; git status --short --ignored --untracked-files=all -- src ':(exclude)src/tests' | grep '"'^!! src/tests/'"' | sed -n '1,40p'" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
!! src/tests/__pycache__/conftest.cpython-310-pytest-9.1.1.pyc
!! src/tests/__pycache__/conftest.cpython-311-pytest-7.4.0.pyc
!! src/tests/__pycache__/test_bn_drift_probe.cpython-310-pytest-9.1.1.pyc
!! src/tests/__pycache__/test_cond_dispatch.cpython-310-pytest-9.1.1.pyc
!! src/tests/__pycache__/test_cyl_vit_patch_embedding.cpython-310-pytest-9.1.1.pyc
!! src/tests/__pycache__/test_eval_paths.cpython-310-pytest-9.1.1.pyc
!! src/tests/__pycache__/test_exp11_equiv_probe.cpython-310-pytest-9.1.1.pyc
!! src/tests/__pycache__/test_exp11_orbit_configs.cpython-310-pytest-9.1.1.pyc
!! src/tests/__pycache__/test_exp11_orbit_configs.cpython-310.pyc
!! src/tests/__pycache__/test_exp11_p0_collect.cpython-310-pytest-9.1.1.pyc
!! src/tests/__pycache__/test_exp11_p0_runner.cpython-310-pytest-9.1.1.pyc
!! src/tests/__pycache__/test_exp11_restart_record.cpython-310-pytest-9.1.1.pyc
!! src/tests/__pycache__/test_exp11_restart_record.cpython-310.pyc
!! src/tests/__pycache__/test_exp11_trajectory_figures.cpython-310-pytest-9.1.1.pyc
!! src/tests/__pycache__/test_exp11_trajectory_figures.cpython-310.pyc
!! src/tests/__pycache__/test_exp11_validate_rows.cpython-310-pytest-9.1.1.pyc
!! src/tests/__pycache__/test_exp11_wandb_readback.cpython-310-pytest-9.1.1.pyc
!! src/tests/__pycache__/test_exp14_fixed_mode_snapshot.cpython-310-pytest-9.1.1.pyc
!! src/tests/__pycache__/test_exp14_fixed_mode_snapshot.cpython-310.pyc
!! src/tests/__pycache__/test_exp14_fixed_mode_snapshot.cpython-311-pytest-7.4.0.pyc
!! src/tests/__pycache__/test_exp14_validate_cell.cpython-310-pytest-9.1.1.pyc
!! src/tests/__pycache__/test_exp14_validate_cell.cpython-310.pyc
!! src/tests/__pycache__/test_exp14_validate_cell.cpython-311-pytest-7.4.0.pyc
!! src/tests/__pycache__/test_finetune_cond.cpython-310-pytest-9.1.1.pyc
!! src/tests/__pycache__/test_gen_model_comparison_gate.cpython-310-pytest-9.1.1.pyc
!! src/tests/__pycache__/test_gen_model_comparison_gate.cpython-310.pyc
!! src/tests/__pycache__/test_invariant_conditioning.cpython-310-pytest-9.1.1.pyc
!! src/tests/__pycache__/test_invariant_conditioning.cpython-310.pyc
!! src/tests/__pycache__/test_phase3_vit_training.cpython-310-pytest-9.1.1.pyc
!! src/tests/__pycache__/test_retune_lr_state.cpython-310-pytest-9.1.1.pyc
!! src/tests/__pycache__/test_strip_optimizer_state.cpython-310-pytest-9.1.1.pyc
!! src/tests/__pycache__/test_train_max_steps.cpython-310-pytest-9.1.1.pyc
!! src/tests/__pycache__/test_train_sync_batchnorm.cpython-310-pytest-9.1.1.pyc
!! src/tests/__pycache__/test_vit_gradient_checkpointing.cpython-310-pytest-9.1.1.pyc
!! src/tests/__pycache__/test_yaw_aug_arm_config.cpython-310-pytest-9.1.1.pyc
!! src/tests/__pycache__/test_yaw_aug_arm_config.cpython-310.pyc
!! src/tests/__pycache__/test_yaw_aug_record_control.cpython-310-pytest-9.1.1.pyc
!! src/tests/__pycache__/test_yaw_aug_record_control.cpython-310.pyc
!! src/tests/__pycache__/test_yaw_aug_training.cpython-310-pytest-9.1.1.pyc
!! src/tests/__pycache__/test_yaw_aug_training.cpython-310.pyc

exec
/usr/bin/bash -lc 'git diff --numstat -- worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh; git diff --check -- worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh' in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
6	2	worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
14	0	worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh

exec
/usr/bin/bash -lc "git grep -n -I -E '("'^|[''^[:alnum:]_])(src[.]tests|from[[:space:]]+tests([[:space:].]|$)|import[[:space:]]+tests([[:space:].]|$)|conftest)([''^[:alnum:]_]|$)'"' HEAD -- train.py src ':"'!src/tests/**'"' defaults.ini; git grep -n -I -E '(pytest_plugins|PYTEST_ADDOPTS|addopts|python_files|testpaths)' -- pyproject.toml pytest.ini setup.cfg tox.ini defaults.ini 2>/dev/null" in /n/fs/gatrdp/codespace/FLAC
 exited 1 in 0ms:
exec
/usr/bin/bash -lc 'git show --format=fuller --name-status --stat 3c2bcbe; git diff --name-only 3c2bcbe''^ 3c2bcbe -- train.py defaults.ini src '"':(exclude)src/tests' data/AR 'worklog/worklog_yixun/exp_11_fa_orbit_claude/*.json' 'worklog/worklog_yixun/exp_11_fa_orbit_claude/*.py' 'worklog/worklog_yixun/exp_11_fa_orbit_claude/*.sbatch' 'worklog/worklog_yixun/exp_11_fa_orbit_claude/*.sh' worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
commit 3c2bcbee95845f6934da43bc6c815a27086fecdf
Author:     Yixun Hu <yh4742@princeton.edu>
AuthorDate: Wed Aug 12 00:58:20 2026 -0400
Commit:     Yixun Hu <yh4742@princeton.edu>
CommitDate: Wed Aug 12 00:58:20 2026 -0400

    exp_14 r3-fix4: per-scene means 10 ROOM FAMILIES, not 17 rooms
    
    Rung 1 (C4L@90, jid 3682720) evaluated correctly and was then refused by
    its own validator: "by_scene covers 10 scene(s), not the split's 17".
    The artifact was right and the constant was wrong.
    
    Root cause, established from real data: the released metric callback
    groups on md['scene'], and AR_md.py:27 sets that to the room FAMILY
    (rel_path[-3], e.g. "Cafe"); the per-room id (rel_path[-2],
    "Cafe_idx_0") never reaches the callback. So the release convention's
    per-scene mean is over the split's 10 FAMILIES — "6,337 items / 17
    rooms" describes the split's CONTENT, not its grouping. The landed
    record's by_scene keys are exactly those ten.
    
    The check is now STRONGER than the one it replaces: EXPECTED_SCENE_KEYS
    pins the ten family names and the validator compares the KEY SET, naming
    what is missing or unexpected. Ten differently-named groups would have
    satisfied the old count check and been a different estimand.
    
    Propagated to every place the expectation lives: validator (+ its
    contract line, which now says "expected_scenes 10 (release grouping: AR
    room families)"), sbatch pin and header, collector's rendered grouping
    disclosure, guard fixture and the three intent cases, and the params
    doc's protocol line. Test fixtures now use the REAL family names, so the
    key-set rule is exercised rather than sidestepped by synthetic keys.
    
    RED/GREEN on the real artifact, read-only, committed as the log: INVALID
    pre-fix with that one reason, VALID post-fix across all three files
    (metrics + screenmeta + stream). Two new tests use it directly as a
    fixture and skip where it is absent. **Rung 1 is retroactively complete**
    — the wave classifier now reports it VALID and the other five vctl cells
    MISSING, so WAVE=vctl will skip it and land the rest.
    
    Battery: 641 pytest passed (eight suites); guard 238 passed / 0 failed,
    suite_rc=0; DRYRUN grid diff empty (106 cells); transcripts regenerated.
    
    Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>

M	src/tests/test_exp14_validate_cell.py
M	src/tests/test_yaw_gen_collect.py
M	worklog/worklog_yixun/exp_14_yaw_gen_claude/exp14_validate_cell.py
A	worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-12_r3fix4_battery.log
A	worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-12_r3fix4_guardtests.log
A	worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_2026-08-12_r3fix4_red_realartifact.log
M	worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect.py
M	worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect_selftest_blocked.txt
M	worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect_selftest_complete.txt
M	worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect_selftest_gatefail.txt
M	worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_collect_selftest_pending.txt
M	worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_params_set_up.md
M	worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen.sbatch
M	worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_screen_guardtests.sh

exec
/usr/bin/bash -lc "sed -n '1,280p' pyproject.toml; git grep -n -I -E '(find_namespace|find_packages|packages[[:space:]]*=|package_data|src/tests|src[.]tests)' -- pyproject.toml setup.py setup.cfg MANIFEST.in 2>/dev/null" in /n/fs/gatrdp/codespace/FLAC
 exited 1 in 0ms:
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "FLAC"
version = "1.0"
description = "Few-shot RIR generation with flow matching"
authors = [
    { name="Amandine Brunetto" }
]
readme = "README.md"
requires-python = ">=3.10"

dependencies = [
    "torch==2.7.0",
    "torchaudio==2.7.0",
    "torchvision==0.22.0",
    "torchmetrics==0.11.4",
    "pytorch_lightning==2.1.0",
    "transformers==4.57.0",
    "huggingface_hub==0.35.3",
    "numpy==1.23.5",
    "scipy==1.15.3",
    "librosa==0.11.0",
    "einops==0.8.2",
    "ema-pytorch==0.2.3",
    "safetensors==0.7.0",
    "tqdm",
    "wandb==0.15.4",
    "deepspeed==0.18.6",
    "pyroomacoustics==0.8.3",
    "alias-free-torch==0.0.6",
    "k-diffusion==0.1.1",
    "interface==2.11.1",
    "plotting==0.0.7",
    "prefigure==0.0.9", 
]

[tool.setuptools.packages.find]
where = ["."]
codex
APPROVED

No blocking findings.

- `src/tests` is not imported by the training runtime. `train.py`, `src` initializers, and tracked non-test modules contain no such imports. `conftest.py` only adjusts `sys.path` when pytest discovers it; training never loads it. The AR dynamic metadata loader targets `src/configs/dataset_configs/custom_metadata/AR_md.py`.
- Git 2.52.0 on this host correctly applies `:(exclude)src/tests` after unioning all positive pathspecs. This works identically for `git diff` and `git status`; later positive pathspecs do not re-add excluded paths. The quoting and repository-root working directory are correct. No host-version caveat applies.
- The guard fixture plumbing is sound. `SYN_TIDX` is a fresh index initialized from `HEAD`, so it does not inherit the earlier synthetic `train.py` change. Reusing `SYN_BLOB` is harmless, `${SYN_BLOB:-}` is safe under `set -u`, and fixture-construction failure remains fail-closed.
- A commit changing both tests and runtime surfaces still fails through the runtime changes. No fresh gate hole found. The exclusion remains valid while `src/tests` retains its current pytest-only contract.
tokens used
90,278
APPROVED

No blocking findings.

- `src/tests` is not imported by the training runtime. `train.py`, `src` initializers, and tracked non-test modules contain no such imports. `conftest.py` only adjusts `sys.path` when pytest discovers it; training never loads it. The AR dynamic metadata loader targets `src/configs/dataset_configs/custom_metadata/AR_md.py`.
- Git 2.52.0 on this host correctly applies `:(exclude)src/tests` after unioning all positive pathspecs. This works identically for `git diff` and `git status`; later positive pathspecs do not re-add excluded paths. The quoting and repository-root working directory are correct. No host-version caveat applies.
- The guard fixture plumbing is sound. `SYN_TIDX` is a fresh index initialized from `HEAD`, so it does not inherit the earlier synthetic `train.py` change. Reusing `SYN_BLOB` is harmless, `${SYN_BLOB:-}` is safe under `set -u`, and fixture-construction failure remains fail-closed.
- A commit changing both tests and runtime surfaces still fails through the runtime changes. No fresh gate hole found. The exclusion remains valid while `src/tests` retains its current pytest-only contract.
