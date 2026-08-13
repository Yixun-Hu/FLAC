#!/usr/bin/env bash
# ============================================================================
# yaw_aug_train_guardtests.sh — guard-branch exercise for the exp_15 YAWAUG
# launcher. Derived from exp_11's fa_orbit_train_guardtests.sh (round-3 B8) and
# re-aimed at exp_15's gates.
#
# SAFETY (inherited, and non-negotiable):
#   * nothing is submitted — no sbatch, no scancel, no GPU is touched;
#   * every case runs with OUTPUT_ROOT pointed at a mktemp directory, never a
#     production prefix;
#   * no tracked file is ever mutated. Cases that need a mutated launcher, config
#     or allowlist use the SPOOL pattern: the file is copied into the temp root
#     and edited there, exactly as Slurm spools a submitted script, and the copy
#     is pointed back at this repo with YAW_AUG_REPO_OVERRIDE. A before/after
#     snapshot of `git status` asserts the tracked tree never moved.
#
# Vehicles:
#   DRYRUN=1        every cheap gate (pins, arm, config map, semantic gate,
#                   allowlist, lineage, accumulation, argv parity), then exit
#                   before Slurm/GPU.
#   real mode       with a fake SLURM_JOB_ID: proves the commit/drift, allowlist
#                   and sbatch-only gates are fail-closed.
#   spooled copies  a launcher with ONE pin edited, to reach gates that a correct
#                   launcher can never fail.
#   sourced helper  the banner-verdict function is extracted from the launcher
#                   and driven over synthetic logs.
#   mocked logs     exp_11's fa_orbit_classify.py over synthetic logs (the exit
#                   taxonomy is shared and arm-agnostic).
#
# Usage:  bash worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train_guardtests.sh
# Exit 0 = every case behaved as specified.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR="worklog/worklog_yixun/exp_15_yaw_aug_claude"
EXP11DIR="worklog/worklog_yixun/exp_11_fa_orbit_claude"
LAUNCHER="${EXPDIR}/yaw_aug_train.sbatch"
SUBMITTER="${EXPDIR}/yaw_aug_submit.sh"
ALLOWLIST="${EXPDIR}/yaw_aug_pin_allowlist.txt"
ARM_CONFIG="${EXPDIR}/FLAC_AR_YAWAUG.json"
CLASSIFY="${EXP11DIR}/fa_orbit_classify.py"
PREFLIGHT="${EXP11DIR}/fa_orbit_ckpt_preflight.py"
PY=/n/fs/gatrdp/envs/flac/bin/python
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR}/yaw_aug_${TS}_guardtests_${GUARD_TAG:-r3fix}.log"
HEAD_SHA="$(git rev-parse HEAD)"
CONTROL_COMMIT="81ddac372076ea92751ae09cbaf371df70f396e5"

# Machine-readable ledger: one CASE line per case, so union coverage across the
# two environments can be CHECKED rather than claimed (chain review, finding 8).
# Created BEFORE the tracked-state snapshots below, so the suite's own evidence
# files are not mistaken for the suite mutating the tree.
LEDGER="${LOG%.log}.ledger"
: > "$LEDGER"
exec > >(tee -a "$LOG") 2>&1
echo "=== yaw_aug_train guard exercise — ${TS} — $(git rev-parse --short HEAD) ==="
for f in "$LAUNCHER" "$SUBMITTER" "$ALLOWLIST" "$ARM_CONFIG" "$CLASSIFY" "$PREFLIGHT"; do
  [ -f "$f" ] || { echo "missing ${f} - abort"; exit 3; }
done

TRACKED_BEFORE="$(git status --porcelain --untracked-files=no -- "$EXPDIR" "$EXP11DIR" src data/AR | sort)"
# ...plus untracked files in OUR folder only: another session writes into
# exp_11's during a run, and that is not this suite's doing.
UNTRACKED_BEFORE="$(git status --porcelain --untracked-files=all -- "$EXPDIR" | sort)"
TMP="$(mktemp -d)"
OUT_ROOT="${TMP}/outputs"            # never a production prefix
mkdir -p "$OUT_ROOT"
trap 'rm -rf "$TMP"' EXIT
PASS=0; FAIL=0
STRICT="${STRICT:-0}"
ledger() { printf '%s\t%s\n' "$1" "$2" >> "$LEDGER"; }
skip_case() {   # <name> <reason> — a SKIP is a FAILURE under STRICT=1
  if [ "$STRICT" = "1" ]; then
    echo "FAIL  ${1} (STRICT: ${2})"; ledger FAIL "$1"; FAIL=$((FAIL + 1))
  else
    echo "SKIP  ${1} (${2})"; ledger SKIP "$1"
  fi
}
skip_env() {    # <name> <reason> — the case CANNOT run in this environment
  # e.g. outputs_FLAC is gitignored, so a worktree has no control manifest. STRICT
  # does not fail these; the union checker still demands a PASS in the OTHER
  # environment, so they cannot quietly go uncovered either.
  echo "SKIP  ${1} (environment: ${2})"; ledger SKIP "$1"
}

case_run() {  # <name> <want-rc> <want-substring> -- <env...>   (runs the REAL launcher)
  local name="$1" want_rc="$2" want_txt="$3"; shift 3; [ "$1" = "--" ] && shift
  local out rc
  out="$(env "$@" bash "$LAUNCHER" 2>&1)"; rc=$?
  if [ "$rc" -eq "$want_rc" ] && echo "$out" | grep -qF -- "$want_txt"; then
    echo "PASS  ${name}  (rc=${rc})"; ledger PASS "$name"; PASS=$((PASS + 1))
  else
    echo "FAIL  ${name}: want rc=${want_rc} + '${want_txt}', got rc=${rc}"
    echo "$out" | tail -5 | sed 's/^/        | /'; ledger FAIL "$name"; FAIL=$((FAIL + 1))
  fi
}

case_spool() {  # <name> <launcher> <want-rc> <want-substring> -- <env...>
  local name="$1" launcher="$2" want_rc="$3" want_txt="$4"; shift 4; [ "$1" = "--" ] && shift
  local out rc
  out="$(env "$@" bash "$launcher" 2>&1)"; rc=$?
  if [ "$rc" -eq "$want_rc" ] && echo "$out" | grep -qF -- "$want_txt"; then
    echo "PASS  ${name}  (rc=${rc})"; ledger PASS "$name"; PASS=$((PASS + 1))
  else
    echo "FAIL  ${name}: want rc=${want_rc} + '${want_txt}', got rc=${rc}"
    echo "$out" | tail -5 | sed 's/^/        | /'; ledger FAIL "$name"; FAIL=$((FAIL + 1))
  fi
}

expect_cmd() {  # <name> <want-rc> <want-substring> -- <command...>
  local name="$1" want_rc="$2" want_txt="$3"; shift 3; [ "$1" = "--" ] && shift
  local out rc
  out="$("$@" 2>&1)"; rc=$?
  if [ "$rc" -eq "$want_rc" ] && echo "$out" | grep -qF -- "$want_txt"; then
    echo "PASS  ${name}  (rc=${rc})"; ledger PASS "$name"; PASS=$((PASS + 1))
  else
    echo "FAIL  ${name}: want rc=${want_rc} + '${want_txt}', got rc=${rc}"
    echo "$out" | tail -5 | sed 's/^/        | /'; ledger FAIL "$name"; FAIL=$((FAIL + 1))
  fi
}

check() {  # <name> <condition-rc> — for grep-style structural assertions
  if [ "$2" -eq 0 ]; then echo "PASS  $1"; ledger PASS "$1"; PASS=$((PASS + 1))
  else echo "FAIL  $1"; ledger FAIL "$1"; FAIL=$((FAIL + 1)); fi
}

spool() {  # <tag> [<sed-expr>...] -> path to a spooled launcher copy
  # Every copy/substitution failure is fatal, and each substitution must actually
  # CHANGE the file: a sed that silently matched nothing would hand the caller a
  # pristine launcher and turn its case into a false green (review finding 7).
  local tag="$1"; shift
  local dst="${TMP}/spool_${tag}.sbatch"
  cp "$LAUNCHER" "$dst" || { echo "spool ${tag}: cp failed" >&2; return 3; }
  local expr before
  for expr in "$@"; do
    before="$(md5sum < "$dst")"
    sed -i "$expr" "$dst" || { echo "spool ${tag}: sed '${expr}' failed" >&2; return 3; }
    if [ "$before" = "$(md5sum < "$dst")" ]; then
      echo "spool ${tag}: sed '${expr}' matched nothing — the case would test an unmodified launcher" >&2
      return 3
    fi
  done
  echo "$dst"
}

# The submitter's acceptance-record path is redirected into the temp root for
# every gate case: yaw_aug_smoke_acceptance.json is a TRACKED artifact of the
# real smoke, and a suite that rewrote it would both violate its own no-mutation
# rule and dirty the closure for every later case.
SUB_SPOOL="${TMP}/yaw_aug_submit_spooled.sh"
ACC="${TMP}/spooled_acceptance.json"
cp "$SUBMITTER" "$SUB_SPOOL" || { echo "could not spool the submitter"; exit 3; }
sed -i "s|ACCEPT_FILE=\"\${EXPDIR}/yaw_aug_smoke_acceptance.json\"|ACCEPT_FILE=\"${ACC}\"|" "$SUB_SPOOL"
# A spooled copy lives outside the repo, so its `cd $(git rev-parse --show-toplevel)`
# would land in $HOME and every later gate would read the wrong tree. Pin it to
# the tree under test, exactly as sbatch's spool keeps the launcher's absolute REPO.
sed -i "s@^cd \"\$(git -C .*rev-parse --show-toplevel)\" .*@cd \"${PWD}\" || exit 3@" "$SUB_SPOOL"
grep -qF "cd \"${PWD}\" || exit 3" "$SUB_SPOOL" || { echo "submitter spool did not pin its repo"; exit 3; }
grep -q "ACCEPT_FILE=\"${ACC}\"" "$SUB_SPOOL" || { echo "submitter spool did not redirect ACCEPT_FILE"; exit 3; }

# Cases that must reach the submitter's LATER gates need a clean training
# closure. Another session commits to this checkout continuously, so a dirty
# closure is reported as a SKIP with its reason, never as a failure of this kit.
closure_clean() {
  # stderr is captured and the exit status is honoured: a git failure is NOT
  # "clean" (chain review, finding 8).
  local out
  out="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src \
           ":(exclude)src/tests" data/AR "$EXPDIR" "$EXP11DIR/fa_orbit_ckpt_preflight.py" \
           "$EXP11DIR/fa_orbit_classify.py" "$EXP11DIR/fa_orbit_wandb_readback.py" \
           "$EXP11DIR/FLAC_AR_VANCKPT.json" 2>&1)" || {
    echo "closure_clean: git status FAILED: ${out}" >&2; return 2; }
  [ -z "$out" ]
}
if closure_clean; then CLOSURE_STATE="clean"; else CLOSURE_STATE="dirty"; fi
echo "training closure at suite start: ${CLOSURE_STATE}"
sub_case() {  # like expect_cmd, but skipped (not failed) when the closure is dirty
  local name="$1"
  if ! closure_clean; then
    skip_case "$name" "training closure dirty — another session is mid-edit"; return 0
  fi
  expect_cmd "$@"
}

REPO_ENV=("YAW_AUG_REPO_OVERRIDE=$PWD")   # dry runs read THIS tree, not the production checkout
BASE_ENV=(DRYRUN=1 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}")
SMOKE_ENV=("${BASE_ENV[@]}" SMOKE=1 SMOKE_RUNG=8x8 SMOKE_MIN_FREE_MB=14000)

echo "--- A. pins are concrete and the refusal mechanism survives ---"
grep -qE '^PINNED_[A-Z_]+="TO-PIN-AFTER-P0"' "$LAUNCHER"; check "every launcher pin holds a concrete value" $((1 - $?))
grep -q 'PIN_PLACEHOLDER="TO-PIN-AFTER-P0"' "$LAUNCHER"; check "the placeholder refusal mechanism is still present" $?
grep -q '^PINNED_MAXSTEPS=40000' "$LAUNCHER"; check "budget re-pinned to the registered 40000" $?
grep -q '\[ "\$MAXSTEPS" = "40000" \]' "$LAUNCHER"; check "the production assertion enforces 40000" $?
grep -q '^PINNED_TIME_LIMIT_YAWAUG="24:00:00"' "$LAUNCHER"; check "INITIAL time pin is 24:00:00" $?
grep -q '^PINNED_TIME_LIMIT_RESTART_YAWAUG="24:00:00"' "$LAUNCHER"; check "RESTART time pin is 24:00:00" $?
grep -q 'export PYTHONUNBUFFERED=1' "$LAUNCHER"; check "PYTHONUNBUFFERED is exported for the banner" $?
grep -q 'PRE_ARGS+=(--extension' "$LAUNCHER"; check "exp_11's Q10 extension mode is not invoked" $((1 - $?))

echo "--- B. the happy path: a full DRYRUN passes every reachable gate ---"
case_run "INITIAL dry run passes" 0 "ARGV PARITY OK" -- "${BASE_ENV[@]}" ARM=YAWAUG
case_run "  ... and names the 40k budget" 0 "max_steps 40000" -- "${BASE_ENV[@]}" ARM=YAWAUG
case_run "  ... and the accumulation gate reports 1" 0 "accumulation gate OK: --accum-batches 1" \
  -- "${BASE_ENV[@]}" ARM=YAWAUG
case_run "  ... and the config gate accepts the arm config" 0 "gate OK: YAWAUG is vanilla" \
  -- "${BASE_ENV[@]}" ARM=YAWAUG
case_run "  ... and the allowlist gate passes on the real diff" 0 "launch-pin allowlist OK" \
  -- "${BASE_ENV[@]}" ARM=YAWAUG
case_run "  ... and the INITIAL time pin is selected" 0 "time pin PINNED_TIME_LIMIT_YAWAUG=24:00:00" \
  -- "${BASE_ENV[@]}" ARM=YAWAUG

echo "--- C. parameter / arm gates ---"
case_run "missing ARM" 2 "ARM must be exported" -- DRYRUN=1 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}"
case_run "missing EXPECT_SHA" 2 "EXPECT_SHA" -- DRYRUN=1 ARM=YAWAUG "OUTPUT_ROOT=${OUT_ROOT}" "${REPO_ENV[@]}"
for BAD in VANL C4L C32 yawaug; do
  case_run "arm ${BAD} rejected" 2 "is not a legal exp_15 arm" -- "${BASE_ENV[@]}" ARM=$BAD
done

echo "--- D. NEW: the budget pin (delta 2) ---"
# A production launch whose pin is not 40000 must die. Reached only via a spooled
# copy, because the real launcher cannot carry another value.
BAD_STEPS="$(spool badsteps 's/^PINNED_MAXSTEPS=40000/PINNED_MAXSTEPS=50000/')"
case_spool "production MAXSTEPS != 40000 dies" "$BAD_STEPS" 2 "the registered budget is 40000" \
  -- "${BASE_ENV[@]}" ARM=YAWAUG
BAD_STEPS_LOW="$(spool badsteps_low 's/^PINNED_MAXSTEPS=40000/PINNED_MAXSTEPS=30000/')"
case_spool "a SHORTER budget dies too (not just a longer one)" "$BAD_STEPS_LOW" 2 "the registered budget is 40000" \
  -- "${BASE_ENV[@]}" ARM=YAWAUG

echo "--- E. NEW: RESTART is capped at the pre-registered endpoint (delta 2) ---"
RUN="${OUT_ROOT}/exp15_YAWAUG/FLAC_exp15_YAWAUG/exp15_YAWAUG"
mkdir -p "${RUN}/checkpoints"
: > "${RUN}/checkpoints/epoch=8-step=40000.ckpt"
: > "${RUN}/checkpoints/epoch=2-step=12500.ckpt"
case_run "a RESTART resuming AT 40000 dies (the extension attempt)" 2 "no extension mode" \
  -- "${BASE_ENV[@]}" ARM=YAWAUG EXPECTED_STEP=40000 \
     "RESUME_CKPT=${RUN}/checkpoints/epoch=8-step=40000.ckpt"
case_run "a RESTART resuming PAST 40000 dies" 2 "at/past the pre-registered 40000-step endpoint" \
  -- "${BASE_ENV[@]}" ARM=YAWAUG EXPECTED_STEP=45000 \
     "RESUME_CKPT=${RUN}/checkpoints/epoch=8-step=40000.ckpt"
# ...and with the budget assertion neutralised (spool), the cap itself fires:
CAP="$(spool cap 's/\[ "\$MAXSTEPS" = "40000" \]/[ "$MAXSTEPS" = "50000" ]/' \
                 's/^PINNED_MAXSTEPS=40000/PINNED_MAXSTEPS=50000/')"
case_spool "a RESTART TARGETING beyond 40000 dies" "$CAP" 2 "may not target beyond the pre-registered" \
  -- "${BASE_ENV[@]}" ARM=YAWAUG EXPECTED_STEP=12500 \
     "RESUME_CKPT=${RUN}/checkpoints/epoch=2-step=12500.ckpt"
case_run "a legitimate mid-run RESTART is accepted" 0 "ARGV PARITY OK" \
  -- "${BASE_ENV[@]}" ARM=YAWAUG EXPECTED_STEP=12500 \
     "RESUME_CKPT=${RUN}/checkpoints/epoch=2-step=12500.ckpt"
case_run "  ... and selects the RESTART time pin" 0 "time pin PINNED_TIME_LIMIT_RESTART_YAWAUG=24:00:00" \
  -- "${BASE_ENV[@]}" ARM=YAWAUG EXPECTED_STEP=12500 \
     "RESUME_CKPT=${RUN}/checkpoints/epoch=2-step=12500.ckpt"

echo "--- F. lineage gates (inherited) ---"
: > "${TMP}/foreign.ckpt"
case_run "initial + RESUME_CKPT" 2 "INITIAL launch must not carry" \
  -- "${BASE_ENV[@]}" ARM=YAWAUG "RESUME_CKPT=${TMP}/foreign.ckpt"
case_run "restart w/o ckpt" 2 "RESTART requires RESUME_CKPT" -- "${BASE_ENV[@]}" ARM=YAWAUG EXPECTED_STEP=5000
case_run "restart ckpt missing" 2 "not found" \
  -- "${BASE_ENV[@]}" ARM=YAWAUG EXPECTED_STEP=5000 "RESUME_CKPT=${TMP}/nope.ckpt"
case_run "restart foreign ckpt" 2 "may only resume a checkpoint from" \
  -- "${BASE_ENV[@]}" ARM=YAWAUG EXPECTED_STEP=5000 "RESUME_CKPT=${TMP}/foreign.ckpt"
case_run "initial refuses an existing run dir" 2 "already exists" -- "${BASE_ENV[@]}" ARM=YAWAUG
# ...and now clear it, so the INITIAL cases that follow start from a clean root.
rm -rf "${OUT_ROOT}/exp15_YAWAUG"

echo "--- G. NEW: the accumulation gate (delta 4, round-1 finding) ---"
ACCUM2="$(spool accum2 's/^PINNED_ACCUM_BATCHES=1/PINNED_ACCUM_BATCHES=2/')"
case_spool "accum != 1 dies" "$ACCUM2" 2 "would draw the SAME yaws" -- "${SMOKE_ENV[@]}" ARM=YAWAUG
ACCUM_GONE="$(spool accumgone 's/  --max-steps "\$MAXSTEPS" --batch-size "\$MB" --accum-batches "\$PINNED_ACCUM_BATCHES"/  --max-steps "$MAXSTEPS" --batch-size "$MB"/')"
case_spool "an argv with no --accum-batches dies" "$ACCUM_GONE" 2 "carries no --accum-batches" \
  -- "${SMOKE_ENV[@]}" ARM=YAWAUG

echo "--- H. NEW: the semantic config gate rejects mutated configs (delta 3) ---"
# The REAL gate is exercised: a spooled launcher is pointed at a mutated copy of
# the arm config. No tracked config is touched.
mutate_config() {  # <tag> <python-mutation> -> path
  # NB: separate declarations — bash expands every word of a `local` before it
  # assigns any of them, so `local tag="$1" dst="...${tag}..."` trips `set -u`.
  local tag="$1" mut="$2"
  local dst="${TMP}/cfg_${tag}.json"
  $PY - "$ARM_CONFIG" "$dst" "$mut" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
exec(sys.argv[3])
json.dump(cfg, open(sys.argv[2], "w"), indent=4)
PY
  echo "$dst"
}
cfg_case() {  # <name> <tag> <mutation> <want-substring>
  local cfg; cfg="$(mutate_config "$2" "$3")"
  local l; l="$(spool "cfg_$2" "s|echo \"\$EXPDIR/FLAC_AR_YAWAUG.json\"|echo \"${cfg}\"|")"
  case_spool "$1" "$l" 2 "$4" -- "${BASE_ENV[@]}" ARM=YAWAUG
}
cfg_case "config with cond_method rejected" condmethod \
  'cfg["training"]["cond_method"] = "fa_invariant"' "cond_method is present"
cfg_case "config with frame_avg_angles rejected" angles \
  'cfg["training"]["frame_avg_angles"] = [0.0, 90.0, 180.0, 270.0]' "frame_avg_angles is present"
cfg_case "yaw_aug disabled rejected" yawoff \
  'cfg["training"]["yaw_aug"]["enabled"] = False' "!= the registered block"
cfg_case "yaw_aug seed changed rejected" yawseed \
  'cfg["training"]["yaw_aug"]["seed"] = 43' "!= the registered block"
cfg_case "yaw_aug img_w changed rejected" yawimgw \
  'cfg["training"]["yaw_aug"]["img_w"] = 256' "!= the registered block"
cfg_case "yaw_aug block absent rejected" yawgone \
  'cfg["training"].pop("yaw_aug")' "!= the registered block"
cfg_case "extra yaw_aug key rejected" yawextra \
  'cfg["training"]["yaw_aug"]["stride"] = 1' "!= the registered block"
cfg_case "grad-ckpt false rejected" gradckpt \
  '[c["config"].update(gradient_checkpointing=False) for c in cfg["model"]["conditioning"]["configs"] if c["type"] == "ViTCoordinates"]' \
  "gradient_checkpointing=False"
cfg_case "EMA off rejected" noema 'cfg["training"]["use_ema"] = False' "use_ema=False"
cfg_case "a ViT at another resolution rejected" vitres \
  '[c["config"]["ViT"].update(img_w=1024) for c in cfg["model"]["conditioning"]["configs"] if c["type"] == "ViTCoordinates"]' \
  "!= yaw_aug.img_w"

echo "--- I. NEW: the launch-pin allowlist gate (delta 6, plan §3.3-2) ---"
# A planted surprise: a RESTRICTIVE allowlist (temp file) that no longer covers
# the training hook. Nothing tracked is modified — the launcher copy is pointed
# at the temp allowlist, exactly as if a reviewer had forgotten an entry.
printf 'src/tests/*\n' > "${TMP}/restrictive_allowlist.txt"
SURPRISE="$(spool allowlist "s|ALLOWLIST_FILE=\"\${EXPDIR}/yaw_aug_pin_allowlist.txt\"|ALLOWLIST_FILE=\"${TMP}/restrictive_allowlist.txt\"|")"
case_spool "DRYRUN names the unreviewed file" "$SURPRISE" 0 "src/training/diffusion.py" \
  -- "${BASE_ENV[@]}" ARM=YAWAUG
case_spool "  ... and says a real launch would abort" "$SURPRISE" 0 "a real launch aborts here" \
  -- "${BASE_ENV[@]}" ARM=YAWAUG
# real mode (fake job id, clean tree required): the same surprise must be fatal
# Under a Slurm job id the launcher ignores YAW_AUG_REPO_OVERRIDE by design, so a
# spooled copy would read the PRODUCTION checkout. The spool therefore rewrites
# the literal REPO= line to THIS tree — which in the clean-worktree run is the
# worktree — so the case exercises the gate instead of skipping on someone else's
# in-flight edit (chain review, finding 8).
SURPRISE_REAL="$(spool allowlist_real \
  "s|ALLOWLIST_FILE=\"\${EXPDIR}/yaw_aug_pin_allowlist.txt\"|ALLOWLIST_FILE=\"${TMP}/restrictive_allowlist.txt\"|" \
  "s|^REPO=/n/fs/gatrdp/codespace/FLAC$|REPO=${PWD}   # guardtest: read THIS tree|")"
if closure_clean; then
  case_spool "a REAL launch dies on an unreviewed file" "$SURPRISE_REAL" 2 "unreviewed production-surface changes" \
    -- ARM=YAWAUG "EXPECT_SHA=${HEAD_SHA}" SLURM_JOB_ID=999999
else
  skip_case "a REAL launch dies on an unreviewed file" \
    "the drift gate fires first on this dirty tree; the strict clean-worktree run covers it"
fi
printf '' > "${TMP}/empty_allowlist.txt"
EMPTY="$(spool allowempty "s|ALLOWLIST_FILE=\"\${EXPDIR}/yaw_aug_pin_allowlist.txt\"|ALLOWLIST_FILE=\"${TMP}/empty_allowlist.txt\"|")"
case_spool "an empty allowlist is refused" "$EMPTY" 2 "allowlist is empty" -- "${BASE_ENV[@]}" ARM=YAWAUG
GONE="$(spool allowgone "s|ALLOWLIST_FILE=\"\${EXPDIR}/yaw_aug_pin_allowlist.txt\"|ALLOWLIST_FILE=\"${TMP}/no_such_allowlist.txt\"|")"
case_spool "a missing allowlist is refused" "$GONE" 2 "allowlist" -- "${BASE_ENV[@]}" ARM=YAWAUG
# the committed allowlist really does cover the real diff, entry by entry
$PY - "$ALLOWLIST" "$CONTROL_COMMIT" <<'PY'
import fnmatch, subprocess, sys
allow = [l.split("#")[0].strip() for l in open(sys.argv[1])]
allow = [a for a in allow if a]
changed = subprocess.run(["git", "diff", "--name-only", f"{sys.argv[2]}..HEAD", "--",
                          "train.py", "defaults.ini", "src"],
                         capture_output=True, text=True, check=True).stdout.split()
unmatched = [c for c in changed if not any(fnmatch.fnmatch(c, a) for a in allow)]
print(f"changed production files: {len(changed)}; unmatched: {unmatched}")
sys.exit(1 if unmatched else 0)
PY
check "the committed allowlist covers today's real diff" $?

echo "--- J. NEW: the post-launch banner watcher (delta 5) ---"
# The verdict function is EXTRACTED from the launcher and sourced, so the logic
# under test is the shipped logic, not a copy of it.
sed -n '/--- BEGIN banner-watcher-helper/,/--- END banner-watcher-helper/p' "$LAUNCHER" > "${TMP}/banner_helper.sh"
grep -q 'yaw_aug_banner_verdict()' "${TMP}/banner_helper.sh"; check "banner helper extracted from the launcher" $?
# shellcheck disable=SC1090
. "${TMP}/banner_helper.sh"
BANNER="yaw_aug ENABLED img_w=512 seed=42"
STEP_RE="Epoch [0-9]+:"
mk_banner_log() { : > "$1"; echo "All distributed processes registered. Starting with 8 processes" >> "$1"
                  [ "$2" = "yes" ] && echo "$BANNER" >> "$1"
                  [ "$3" = "yes" ] && echo "Epoch 0:   0%|          | 1/4550 [00:07<9:12:31,  7.28s/it]" >> "$1"; return 0; }
mk_banner_log "${TMP}/b1.log" yes no
expect_cmd "banner present, no steps yet -> OK" 0 "OK" -- yaw_aug_banner_verdict "${TMP}/b1.log" "$BANNER" "$STEP_RE"
mk_banner_log "${TMP}/b2.log" yes yes
expect_cmd "banner present and training started -> OK" 0 "OK" -- yaw_aug_banner_verdict "${TMP}/b2.log" "$BANNER" "$STEP_RE"
mk_banner_log "${TMP}/b3.log" no yes
expect_cmd "steps WITHOUT the banner -> MISSING (kill path)" 1 "MISSING" -- yaw_aug_banner_verdict "${TMP}/b3.log" "$BANNER" "$STEP_RE"
mk_banner_log "${TMP}/b4.log" no no
expect_cmd "neither yet -> PENDING (keep waiting)" 2 "PENDING" -- yaw_aug_banner_verdict "${TMP}/b4.log" "$BANNER" "$STEP_RE"
printf '%s\n' "yaw_aug ENABLED img_w=512 seed=43" "Epoch 0:  10%%" > "${TMP}/b5.log"
expect_cmd "a banner with the WRONG seed does not satisfy the gate" 1 "MISSING" -- \
  yaw_aug_banner_verdict "${TMP}/b5.log" "$BANNER" "$STEP_RE"
printf '%s\n' "yaw_aug ENABLED img_w=256 seed=42" "Epoch 0:  10%%" > "${TMP}/b6.log"
expect_cmd "a banner with the WRONG img_w does not satisfy the gate" 1 "MISSING" -- \
  yaw_aug_banner_verdict "${TMP}/b6.log" "$BANNER" "$STEP_RE"
grep -q 'final_rc=8' "$LAUNCHER"; check "a missing banner forces exit class 8" $?
grep -q 'pkill -TERM -P "\$TR_PID"' "$LAUNCHER"; check "the watcher terminates the torchrun group" $?

echo "--- K. commit-binding / sbatch-only gates (REAL mode) ---"
case_run "wrong EXPECT_SHA aborts" 2 "EXPECT_SHA" \
  -- ARM=YAWAUG SMOKE=1 SMOKE_RUNG=8x8 SMOKE_MIN_FREE_MB=99000000 \
     EXPECT_SHA=0000000000000000000000000000000000000000 SLURM_JOB_ID=999999
case_run "real mode needs sbatch" 2 "must run under sbatch" \
  -- ARM=YAWAUG SMOKE=1 SMOKE_RUNG=8x8 SMOKE_MIN_FREE_MB=99000000 "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}"
case_run "ambient OUTPUT_ROOT rejected under Slurm" 2 "!= the production literal" \
  -- ARM=YAWAUG SMOKE=1 SMOKE_RUNG=8x8 SMOKE_MIN_FREE_MB=99000000 "EXPECT_SHA=${HEAD_SHA}" \
     SLURM_JOB_ID=999999 "OUTPUT_ROOT=${OUT_ROOT}"
grep -q 'PRODUCTION_OUTPUT_ROOT="outputs_FLAC"' "$LAUNCHER"; check "launcher pins the production root literally" $?
grep -q 'OUTPUT_ROOT=outputs_FLAC' "$SUBMITTER"; check "submitter exports the fixed root, not ambient state" $?

echo "--- L. env cross-check against the control's manifest (delta 7) ---"
grep -q 'PINNED_CONTROL_MANIFEST_SHA256="113d06a284c6198cf9487e99a2efb7ccde94ae13e656a403fe2af0281d3de8b1"' "$LAUNCHER"
check "the control manifest is sha-pinned to exp_11's registry value" $?
if [ -f outputs_FLAC/exp11_VANL/launch_manifest.txt ]; then
  ACTUAL="$(sha256sum outputs_FLAC/exp11_VANL/launch_manifest.txt | awk '{print $1}')"
  [ "$ACTUAL" = "113d06a284c6198cf9487e99a2efb7ccde94ae13e656a403fe2af0281d3de8b1" ]
  check "the control manifest on disk still hashes to that pin" $?
  grep -q '^torch_version 2.7.0+cu126' outputs_FLAC/exp11_VANL/launch_manifest.txt
  check "the control manifest records our pinned torch" $?
  grep -q '^vae_sha256 8d82159eec35210198246f449bec6561fc19b514922f340a17515050daf7f0b9' outputs_FLAC/exp11_VANL/launch_manifest.txt
  check "the control manifest records our pinned VAE" $?
else
  # Name the cases this stands in for, so the union checker can match them
  # against the environment where they DO run (a summary name matches nothing).
  for MISSING_CASE in "the control manifest on disk still hashes to that pin" \
                      "the control manifest records our pinned torch" \
                      "the control manifest records our pinned VAE"; do
    skip_env "$MISSING_CASE" "outputs_FLAC is gitignored: no manifest in this tree"
  done
fi

echo "--- M. the submitter ---"
expect_cmd "submitter rejects a bad arm" 2 "must be YAWAUG" -- env DRYRUN=1 bash "$SUBMITTER" C8
# A production submission now requires a passing smoke record (full-fix F3), so
# these resource-derivation cases carry an explicit waiver; the gate itself is
# exercised in section W.
SUB=(env DRYRUN=1 "SMOKE_WAIVER=guardtest: resource-derivation cases" bash "$SUB_SPOOL" YAWAUG)
sub_case "submitter derives 8x8 resources" 0 "--gres=gpu:l40:8" -- "${SUB[@]}"
sub_case "  ... cpus from the rung" 0 "--cpus-per-task=64" -- "${SUB[@]}"
sub_case "  ... mem from the rung" 0 "--mem=108G" -- "${SUB[@]}"
sub_case "  ... the INITIAL time pin" 0 "time 24:00:00" -- "${SUB[@]}"
sub_case "  ... and submits NOTHING in DRYRUN" 0 "DRYRUN sbatch" -- "${SUB[@]}"
INTENT_BEFORE="$(ls "${EXPDIR}"/yaw_aug_submission_*.txt 2>/dev/null | wc -l)"
"${SUB[@]}" >/dev/null 2>&1
INTENT_AFTER="$(ls "${EXPDIR}"/yaw_aug_submission_*.txt 2>/dev/null | wc -l)"
[ "$INTENT_BEFORE" = "$INTENT_AFTER" ]; check "a dry run leaves no submission manifest behind" $?
awk '/^OUT=.*sbatch/{sb=NR} /^mv -n "\$TMP" "\$MANIFEST"/{mf=NR} END{exit !(mf && sb && mf < sb)}' "$SUBMITTER"
check "intent manifest is published before the sbatch call" $?
grep -q 'scancel "\$JID"' "$SUBMITTER"; check "an unrecordable job is cancelled" $?

echo "--- N. plumbing that the launcher inherited and must keep ---"
grep -q 'mktemp -u' "$LAUNCHER"; check "no race-prone 'mktemp -u' FIFO" $((1 - $?))
grep -q "trap 'rm -f \"\$FIFO\"' EXIT" "$LAUNCHER"; check "FIFO removal is in the exit trap" $?
grep -q 'pip freeze > "\$PIPFREEZE_FILE"' "$LAUNCHER"; check "pip freeze status is checked before hashing" $?
grep -q 'final_tee_rc' "$LAUNCHER"; check "the final record's tee status is captured" $?
grep -q 'flock -n 9' "$LAUNCHER"; check "launcher uses flock for run ownership" $?
grep -q 'fa_orbit_wandb_readback.py' "$LAUNCHER"; check "the wandb readback still runs" $?
grep -q 'LAUNCH_REGISTRY' "$LAUNCHER"; check "the launch registry is written" $?
grep -q 'final_ckpt_sha256' "$LAUNCHER"; check "the registry records final_ckpt_sha256 at completion" $?

echo "--- O. exit taxonomy, mocked (exp_11's classifier, arm-agnostic) ---"
mk_log() {  # $1 dest, $2 world size (0 = absent), $3 marker?, $4 oom?
  : > "$1"
  [ "$2" != "0" ] && echo "All distributed processes registered. Starting with $2 processes" >> "$1"
  [ "$3" = "yes" ] && echo '`Trainer.fit` stopped: `max_steps=40000` reached.' >> "$1"
  [ "$4" = "yes" ] && echo "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 98.00 MiB" >> "$1"
  return 0
}
A="${TMP}/a.log"; B="${TMP}/b.log"
mk_log "$A" 8 yes no; cp "$A" "$B"
expect_cmd "class 0 complete" 0 "COMPLETE" -- $PY "$CLASSIFY" --rc 0 --tee-rc 0 --ngpu 8 --maxsteps 40000 --log "$A" --log-copy "$B"
mk_log "$A" 0 no no; cp "$A" "$B"
expect_cmd "class 6 world-size absent" 6 "WORLD-SIZE" -- $PY "$CLASSIFY" --rc 0 --tee-rc 0 --ngpu 8 --maxsteps 40000 --log "$A" --log-copy "$B"
mk_log "$A" 1 yes no; cp "$A" "$B"
expect_cmd "class 6 wrong world-size" 6 "reported [1]" -- $PY "$CLASSIFY" --rc 0 --tee-rc 0 --ngpu 8 --maxsteps 40000 --log "$A" --log-copy "$B"
mk_log "$A" 8 no yes; cp "$A" "$B"
expect_cmd "class 3 OOM on nonzero rc" 3 "OOM" -- $PY "$CLASSIFY" --rc 1 --tee-rc 0 --ngpu 8 --maxsteps 40000 --log "$A" --log-copy "$B"
mk_log "$A" 8 no no; cp "$A" "$B"
expect_cmd "class 4 missing marker" 4 "NO-MARKER" -- $PY "$CLASSIFY" --rc 0 --tee-rc 0 --ngpu 8 --maxsteps 40000 --log "$A" --log-copy "$B"
mk_log "$A" 8 yes no; cp "$A" "$B"; echo "divergent tail" >> "$B"
expect_cmd "class 7 logs differ" 7 "LOG-PROVENANCE" -- $PY "$CLASSIFY" --rc 0 --tee-rc 0 --ngpu 8 --maxsteps 40000 --log "$A" --log-copy "$B"

echo "--- P. restart preflight depth (exp_11's tool, our config) ---"
$PY - "$TMP" "$ARM_CONFIG" <<'PY'
import json, os, sys, torch
tmp, cfg_path = sys.argv[1], sys.argv[2]
cfg = json.load(open(cfg_path))
def ck(step=12500, config=cfg, opt=True, sched=True, ema=True):
    d = {"global_step": step, "epoch": 2, "model_config": config,
         "state_dict": {"diffusion.model.x": torch.zeros(1)},
         "optimizer_states": [{"state": {0: {"step": 1}} if opt else {},
                               "param_groups": [{"lr": 1e-5}]}],
         "lr_schedulers": [{"last_epoch": step}] if sched else []}
    if ema:
        d["state_dict"]["diffusion_ema.ema_model.x"] = torch.zeros(1)
    return d
torch.save(ck(), os.path.join(tmp, "good.ckpt"))
torch.save(ck(step=12499), os.path.join(tmp, "wrongstep.ckpt"))
noaug = json.loads(json.dumps(cfg)); noaug["training"].pop("yaw_aug")
torch.save(ck(config=noaug), os.path.join(tmp, "noaug.ckpt"))
torch.save(ck(opt=False), os.path.join(tmp, "stripped.ckpt"))
torch.save(ck(ema=False), os.path.join(tmp, "noema.ckpt"))
print("synthetic checkpoints written")
PY
PRE=($PY "$PREFLIGHT" --config "$ARM_CONFIG" --max-steps 40000 --arm YAWAUG --rung 8x8)
expect_cmd "preflight accepts a good ckpt" 0 "CKPT_SHA256" -- "${PRE[@]}" --ckpt "${TMP}/good.ckpt" --expected-step 12500
expect_cmd "preflight rejects a step mismatch" 2 "global_step" -- "${PRE[@]}" --ckpt "${TMP}/wrongstep.ckpt" --expected-step 12500
expect_cmd "preflight rejects a ckpt trained WITHOUT yaw_aug" 2 "embedded model_config" -- \
  "${PRE[@]}" --ckpt "${TMP}/noaug.ckpt" --expected-step 12500
expect_cmd "preflight rejects a stripped optimizer" 2 "optimizer state is CLEARED" -- "${PRE[@]}" --ckpt "${TMP}/stripped.ckpt" --expected-step 12500
expect_cmd "preflight rejects a missing EMA" 2 "no EMA weights" -- "${PRE[@]}" --ckpt "${TMP}/noema.ckpt" --expected-step 12500

echo "--- R. NEW (r3-fix F1): the smoke runs the real rung and its OWN registry ---"
case_run "smoke at 16x4 dies" 2 "!= the pinned production rung" \
  -- "${BASE_ENV[@]}" SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=14000 ARM=YAWAUG
case_run "smoke at 32x2 dies" 2 "!= the pinned production rung" \
  -- "${BASE_ENV[@]}" SMOKE=1 SMOKE_RUNG=32x2 SMOKE_MIN_FREE_MB=14000 ARM=YAWAUG
case_run "smoke at the pinned 8x8 is accepted" 0 "ARGV PARITY OK" -- "${SMOKE_ENV[@]}" ARM=YAWAUG
expect_cmd "submitter refuses a 16x4 smoke" 2 "must be 8x8" -- \
  env DRYRUN=1 SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=14000 bash "$SUB_SPOOL" YAWAUG
grep -q 'LAUNCH_REGISTRY="${EXPDIR}/yaw_aug_smoke_registry.json"' "$LAUNCHER"
check "a smoke registers in its own registry file" $?
# ...and functionally, with the launcher's OWN registry-init code extracted:
sed -n '/--- BEGIN registry-init-python/,/--- END registry-init-python/p' "$LAUNCHER" > "${TMP}/reg_init.py"
grep -q 'reg\["arms"\]\[arm\]' "${TMP}/reg_init.py"; check "registry-init code extracted from the launcher" $?
SMOKE_REG="${TMP}/yaw_aug_smoke_registry.json"; PROD_REG="${TMP}/yaw_aug_launch_registry.json"
: > "${TMP}/man.txt"; echo "arm YAWAUG" >> "${TMP}/man.txt"
reg_init() {  # <registry> <job> [chain] [cap] [leg-steps]
  $PY "${TMP}/reg_init.py" "$1" YAWAUG "${TMP}/man.txt" "$2" uuid-$2 "$HEAD_SHA" \
      8x8 40000 cfgsha vaesha "${TMP}/save" wandb-$2 "${3:-0}" "${4:-40000}" "${5:-2500}"
}
expect_cmd "a SMOKE registers into the smoke registry" 0 "registered YAWAUG" -- reg_init "$SMOKE_REG" 111
expect_cmd "  ... and a CHAIN INITIAL records its chain metadata" 0 "registered YAWAUG" -- \
  reg_init "${TMP}/chain_init_registry.json" 444 1 40000 2500
$PY -c "
import json,sys
e=json.load(open(sys.argv[1]))['arms']['YAWAUG']
sys.exit(0 if e['chain'] is True and e['chain_cap']==40000 and e['chain_leg_steps']==2500
         and e['chain_initial_target']==40000 else 1)" "${TMP}/chain_init_registry.json"
check "  ... chain flag, cap, leg size and initial target" $?
$PY -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if 'YAWAUG' in d['arms'] else 1)" "$SMOKE_REG"
check "  ... the smoke registry now holds arms.YAWAUG" $?
[ ! -e "$PROD_REG" ]; check "  ... and the PRODUCTION registry was never created" $?
expect_cmd "the production INITIAL then registers successfully" 0 "registered YAWAUG" -- reg_init "$PROD_REG" 222
expect_cmd "  ... and a second production INITIAL is refused" 1 "already registered" -- reg_init "$PROD_REG" 333

echo "--- S. NEW (r3-fix F2): the completion audit is round-2-rigorous, fail-closed ---"
sed -n '/--- BEGIN completion-audit-python/,/--- END completion-audit-python/p' "$LAUNCHER" > "${TMP}/completion.py"
grep -q 'snapshot_checkpoint' "${TMP}/completion.py"; check "completion audit extracted, and it imports the round-2 recorder" $?
grep -q 'summarize_ema' "${TMP}/completion.py"; check "  ... and reuses summarize_ema" $?
grep -q 'canonical_bytes' "${TMP}/completion.py"; check "  ... and reuses canonical_bytes" $?
CK_DIR="${TMP}/ckpts"; mkdir -p "$CK_DIR"
$PY - "$CK_DIR" "$ARM_CONFIG" <<'PY'
import json, os, sys, torch
ck_dir, cfg_path = sys.argv[1], sys.argv[2]
cfg = json.load(open(cfg_path))
def ck(step=40000, config=None, ema="mirror"):
    state = {"diffusion.model.layer.weight": torch.zeros(2, 2),
             "diffusion.model.layer.bias": torch.zeros(2),
             "diffusion.conditioner.e.weight": torch.zeros(3)}
    if ema == "mirror":
        state["diffusion_ema.ema_model.layer.weight"] = torch.zeros(2, 2)
        state["diffusion_ema.ema_model.layer.bias"] = torch.zeros(2)
    elif ema == "partial":
        state["diffusion_ema.ema_model.layer.weight"] = torch.zeros(2, 2)
    return {"global_step": step, "epoch": 8,
            "model_config": cfg if config is None else config,
            "state_dict": state, "optimizer_states": [{}], "lr_schedulers": [{}]}
torch.save(ck(), os.path.join(ck_dir, "good", "epoch=8-step=40000.ckpt")
           if False else os.path.join(ck_dir, "epoch=8-step=40000.ckpt"))
for name, obj in [("wrongstep", ck(step=39000)),
                  ("wrongcfg", ck(config={**cfg, "sample_size": 4096})),
                  ("wrongema", ck(ema="partial"))]:
    d = os.path.join(ck_dir, name); os.makedirs(d, exist_ok=True)
    torch.save(obj, os.path.join(d, "epoch=8-step=40000.ckpt"))
os.makedirs(os.path.join(ck_dir, "empty"), exist_ok=True)
d = os.path.join(ck_dir, "double"); os.makedirs(d, exist_ok=True)
torch.save(ck(), os.path.join(d, "epoch=8-step=40000.ckpt"))
torch.save(ck(), os.path.join(d, "epoch=9-step=40000.ckpt"))
print("completion fixtures written")
PY
fresh_reg() { $PY -c "
import json,sys
json.dump({'arms': {'YAWAUG': {'final_ckpt_sha256': None}}, 'restarts': {}}, open(sys.argv[1],'w'))" "$1"; }
completion() {  # <ckpt-dir> <registry> — monolithic shape: target == cap, chain off
  $PY "${TMP}/completion.py" "${EXPDIR}/yaw_aug_record_control.py" "$2" YAWAUG "$1" 40000 \
      "$ARM_CONFIG" 40000 0
}
AUD_REG="${TMP}/audit_registry.json"
fresh_reg "$AUD_REG"
expect_cmd "a correct 40k checkpoint is audited and recorded" 0 "completion audit OK" -- completion "$CK_DIR" "$AUD_REG"
$PY -c "
import json,sys
e=json.load(open(sys.argv[1]))['arms']['YAWAUG']
sys.exit(0 if e['final_step']==40000 and len(e['final_ckpt_sha256'])==64
         and e['final_ckpt_audit']['ema_key_count']==2
         and e['final_ckpt_audit']['online_model_key_count']==2 else 1)" "$AUD_REG"
check "  ... final_step comes from the checkpoint and the EMA audit is recorded" $?
for CASE in wrongstep wrongcfg wrongema empty double; do
  fresh_reg "$AUD_REG"
  case "$CASE" in
    wrongstep) WANT="!= the pinned endpoint" ;;
    wrongcfg)  WANT="not trained with the arm config" ;;
    wrongema)  WANT="EMA" ;;
    empty)     WANT="found 0" ;;
    double)    WANT="found 2" ;;
  esac
  expect_cmd "completion rejects: ${CASE}" 1 "$WANT" -- completion "${CK_DIR}/${CASE}" "$AUD_REG"
done
expect_cmd "completion fails when the registry cannot be written" 1 "" -- \
  completion "$CK_DIR" "${TMP}/no_such_dir/registry.json"
grep -q 'final_rc=9' "$LAUNCHER"; check "a failed completion audit forces exit class 9" $?
grep -q 'COMPLETION AUDIT FAILED' "$LAUNCHER"; check "  ... and says so loudly" $?

echo "--- T. NEW (r3-fix F3/F4): the closure covers the split and fails CLOSED ---"
grep -q 'data/AR' "$LAUNCHER"; check "the worker closure covers data/AR (the training split)" $?
grep -q 'data/AR' "$SUBMITTER"; check "the submitter closure covers data/AR" $?
grep -q ':(exclude)src/tests' "$LAUNCHER"; check "src/tests is excluded from the worker closure" $?
grep -q ':(exclude)src/tests' "$SUBMITTER"; check "src/tests is excluded from the submitter closure" $?
grep -q 'git status for the drift gate failed' "$LAUNCHER"; check "a failing git status is fatal in the worker" $?
grep -q 'git status for the drift gate failed' "$SUBMITTER"; check "a failing git status is fatal in the submitter" $?
case_run "a broken git environment is fatal, not 'clean'" 2 "git status for the drift gate failed" \
  -- "${BASE_ENV[@]}" ARM=YAWAUG GIT_DIR=/nonexistent/nope.git
sub_case "  ... and the submitter refuses too" 2 "git status for the drift gate failed" -- \
  env DRYRUN=1 GIT_DIR=/nonexistent/nope.git "SMOKE_WAIVER=guardtest: git-failure case" bash "$SUB_SPOOL" YAWAUG
# End-to-end in a throwaway worktree: the real gates, over real mutations, with
# the shared checkout untouched.
WT="${TMP}/wt"
if git worktree add --detach --quiet "$WT" HEAD 2>/dev/null; then
  # DRYRUN, deliberately: YAW_AUG_REPO_OVERRIDE is honoured ONLY outside a Slurm
  # job (so a test hook can never steer a real launch), which means a worktree
  # case must run as a dry run. The advisory names the offending files, and the
  # real-mode fail-closed path is covered by sections I and K on this checkout.
  WT_DRY=(ARM=YAWAUG "OUTPUT_ROOT=${OUT_ROOT}/wt" "YAW_AUG_REPO_OVERRIDE=${WT}" DRYRUN=1)
  WT_HEAD="$(git -C "$WT" rev-parse HEAD)"
  printf '\n' >> "${WT}/data/AR/train.json"
  case_spool "a mutated data/AR/train.json is caught" "${WT}/${EXPDIR}/yaw_aug_train.sbatch" 0 \
    "data/AR/train.json" -- "${WT_DRY[@]}" "EXPECT_SHA=${WT_HEAD}"
  git -C "$WT" checkout -- data/AR/train.json
  rm -f "${WT}/data/AR/train.json"
  case_spool "a DELETED data/AR/train.json is caught (quoted pathspec)" "${WT}/${EXPDIR}/yaw_aug_train.sbatch" 0 \
    "data/AR/train.json" -- "${WT_DRY[@]}" "EXPECT_SHA=${WT_HEAD}"
  git -C "$WT" checkout -- data/AR/train.json
  # a TEST-ONLY commit must NOT abort a pending job (exp_11 2b75036)
  echo "# guardtest scratch" >> "${WT}/src/tests/test_yaw_aug_training.py"
  git -C "$WT" -c user.email=g@t -c user.name=g commit -qam "test-only commit" >/dev/null 2>&1
  case_spool "a test-only commit does NOT abort a pending job" "${WT}/${EXPDIR}/yaw_aug_train.sbatch" 0 \
    "commit binding OK (content)" -- "${WT_DRY[@]}" "EXPECT_SHA=${WT_HEAD}"
  # ...but a runtime-surface commit MUST
  echo "# guardtest scratch" >> "${WT}/src/training/diffusion.py"
  git -C "$WT" -c user.email=g@t -c user.name=g commit -qam "runtime commit" >/dev/null 2>&1
  case_spool "a src/training commit DOES abort a pending job" "${WT}/${EXPDIR}/yaw_aug_train.sbatch" 0 \
    "training surfaces changed since EXPECT_SHA" -- "${WT_DRY[@]}" "EXPECT_SHA=${WT_HEAD}"
  git worktree remove --force "$WT" >/dev/null 2>&1 || true
  git worktree prune >/dev/null 2>&1 || true
else
  skip_case "worktree closure cases" "git worktree add unavailable"
fi

echo "--- U. NEW (r3-fix F5/F6): banner exactness+taxonomy, manifest snapshot ---"
printf '%s\n' "yaw_aug ENABLED img_w=512 seed=420" "Epoch 0:  10%%" > "${TMP}/b7.log"
expect_cmd "seed=420 does NOT satisfy the seed=42 gate" 1 "MISSING" -- \
  yaw_aug_banner_verdict "${TMP}/b7.log" "$BANNER" "$STEP_RE"
printf '%s\n' "INFO: yaw_aug ENABLED img_w=512 seed=42" "Epoch 0:  10%%" > "${TMP}/b8.log"
expect_cmd "a PREFIXED banner line does not satisfy the gate" 1 "MISSING" -- \
  yaw_aug_banner_verdict "${TMP}/b8.log" "$BANNER" "$STEP_RE"
printf '%s\n' "yaw_aug ENABLED img_w=512 seed=42 (rank 0)" "Epoch 0:  10%%" > "${TMP}/b9.log"
expect_cmd "a SUFFIXED banner line does not satisfy the gate" 1 "MISSING" -- \
  yaw_aug_banner_verdict "${TMP}/b9.log" "$BANNER" "$STEP_RE"
printf '%s\n' "Epoch 0:   0%%|  | 1/4550" "$BANNER" > "${TMP}/b10.log"
expect_cmd "a banner printed AFTER step evidence is MISSING" 1 "MISSING" -- \
  yaw_aug_banner_verdict "${TMP}/b10.log" "$BANNER" "$STEP_RE"
grep -q 'if \[ "\$BANNER_VERDICT" = "MISSING" \]' "$LAUNCHER"; check "class 8 is driven by a definite MISSING" $?
grep -q 'which keeps its own taxonomy' "$LAUNCHER"; check "a PENDING verdict preserves an existing failure class (OOM keeps class 3)" $?
grep -q 'CONTROL_MANIFEST_SHA=' "$LAUNCHER"; check "the manifest is no longer hashed by a separate sha256sum pass" $((1 - $?))
sed -n '/--- BEGIN control-env-python/,/--- END control-env-python/p' "$LAUNCHER" > "${TMP}/control_env.py"
grep -c 'fh.read()' "${TMP}/control_env.py" >/dev/null
[ "$(grep -c 'raw = fh.read()' "${TMP}/control_env.py")" = "1" ]
check "the manifest is read exactly ONCE, and hashed+parsed from those bytes" $?
CTRL_MAN=outputs_FLAC/exp11_VANL/launch_manifest.txt
PIN=113d06a284c6198cf9487e99a2efb7ccde94ae13e656a403fe2af0281d3de8b1
if [ -f "$CTRL_MAN" ]; then
  ctrl_env() { $PY "${TMP}/control_env.py" "$1" "$2" "$3" 2.7.0+cu126 2.1.0 \
      8d82159eec35210198246f449bec6561fc19b514922f340a17515050daf7f0b9; }
  expect_cmd "the real control manifest passes the snapshot gate" 0 "control env cross-check OK" -- \
    ctrl_env "$CTRL_MAN" "$PIN" "${EXP11DIR}/arm_launch_registry.json"
  cp "$CTRL_MAN" "${TMP}/tampered_manifest.txt"; printf 'tamper\n' >> "${TMP}/tampered_manifest.txt"
  expect_cmd "a tampered manifest fails the snapshot gate" 1 "!= the pin" -- \
    ctrl_env "${TMP}/tampered_manifest.txt" "$PIN" "${EXP11DIR}/arm_launch_registry.json"
  $PY -c "
import json,sys
d=json.load(open(sys.argv[1])); d['arms']['VANL']['manifest_sha256']='0'*64
json.dump(d, open(sys.argv[2],'w'))" "${EXP11DIR}/arm_launch_registry.json" "${TMP}/reg_moved.json"
  expect_cmd "a registry disagreeing with the reviewed pin is refused" 1 "the control's identity moved" -- \
    ctrl_env "$CTRL_MAN" "$PIN" "${TMP}/reg_moved.json"
else
  for MISSING_CASE in "the real control manifest passes the snapshot gate" \
                      "a tampered manifest fails the snapshot gate" \
                      "a registry disagreeing with the reviewed pin is refused"; do
    skip_env "$MISSING_CASE" "outputs_FLAC is gitignored: no manifest in this tree"
  done
fi

echo "--- V. NEW (full-fix F1): end-of-run code is bound to a run-owned snapshot ---"
grep -q 'CODE_SNAPSHOT="${SAVEDIR}/code_snapshot_jid${SLURM_JOB_ID}"' "$LAUNCHER"
check "the job snapshots its end-of-run code into a run-owned dir" $?
for TOOL in yaw_aug_record_control.py FLAC_AR_YAWAUG.json fa_orbit_classify.py \
            fa_orbit_wandb_readback.py fa_orbit_ckpt_preflight.py; do
  grep -q "\"\$EXPDIR/${TOOL}\"\|\"\$EXP11DIR/${TOOL}\"" "$LAUNCHER"
  check "  ... ${TOOL} is in the snapshot set" $?
done
# every late invocation goes through snap(), and NONE reaches past it to the tree
for LATE in fa_orbit_classify.py fa_orbit_wandb_readback.py yaw_aug_record_control.py; do
  grep -q "snap ${LATE}" "$LAUNCHER"; check "  ... ${LATE} is invoked from the snapshot" $?
done
grep -nE 'python3 "\$EXP11DIR/(fa_orbit_classify|fa_orbit_wandb_readback)\.py"' "$LAUNCHER" >/dev/null
check "  ... and no late path still runs the live copy" $((1 - $?))
grep -q 'echo "snapshot_sha256' "$LAUNCHER"; check "  ... snapshot hashes are written into the manifest" $?
# FUNCTIONAL: mutate the live recorder AFTER the snapshot and prove the
# completion path still runs the snapshotted bytes.
SNAPDIR="${TMP}/code_snapshot"; mkdir -p "$SNAPDIR"
cp "${EXPDIR}/yaw_aug_record_control.py" "$SNAPDIR/"
cp "$ARM_CONFIG" "$SNAPDIR/"
SNAP_SHA_BEFORE="$(sha256sum "${SNAPDIR}/yaw_aug_record_control.py" | awk '{print $1}')"
LIVE_COPY="${TMP}/live_recorder.py"      # stands in for the shared checkout's copy
cp "${EXPDIR}/yaw_aug_record_control.py" "$LIVE_COPY"
printf '\ndef summarize_ema(state_dict):\n    raise SystemExit("MUTATED LIVE RECORDER RAN")\n' >> "$LIVE_COPY"
fresh_reg "${TMP}/snap_registry.json"
expect_cmd "the completion path runs the SNAPSHOT, not the mutated live copy" 0 "completion audit OK" -- \
  $PY "${TMP}/completion.py" "${SNAPDIR}/yaw_aug_record_control.py" "${TMP}/snap_registry.json" \
      YAWAUG "$CK_DIR" 40000 "${SNAPDIR}/FLAC_AR_YAWAUG.json" 40000 0
expect_cmd "  ... while the mutated live copy WOULD have failed the run" 1 "MUTATED LIVE RECORDER RAN" -- \
  $PY "${TMP}/completion.py" "$LIVE_COPY" "${TMP}/snap_registry.json" \
      YAWAUG "$CK_DIR" 40000 "${SNAPDIR}/FLAC_AR_YAWAUG.json" 40000 0
[ "$SNAP_SHA_BEFORE" = "$(sha256sum "${SNAPDIR}/yaw_aug_record_control.py" | awk '{print $1}')" ]
check "  ... and the snapshot's recorded hash still matches its bytes" $?

echo "--- W. NEW (full-fix F3): storage-light smoke + the promotion gate ---"
# (record builders shared with section W2 below)
GATE_SHA="$HEAD_SHA"
gate() { $PY "${TMP}/gate.py" "$1" "${EXPDIR}/yaw_aug_record_control.py" \
             "${2:-$GATE_SHA}" "${3:-8x8}" "${4:-8}" "${5:-30}"; }
mk_record() {  # <path> <python mutation over `r`>
  $PY - "$1" "$2" "$GATE_SHA" <<'PY'
import json, sys
path, mutation, commit = sys.argv[1:4]
r = {
    "_meta": {"experiment": "exp_15", "kind": "smoke acceptance record",
              "purpose": "gates the production submission", "job": "4242",
              "commit": commit, "rung": "8x8", "ngpu": 8, "max_steps": 30,
              "wall_seconds": 31.0},
    "measured": {"banner_verdict": "OK", "steps_observed": 30,
                 "steps_per_second": 1.03, "peak_vram_mb": 9600.0,
                 "peak_vram_source": "nvidia-smi sampler, 3 samples over 8 GPU(s)",
                 "peak_vram_per_gpu_mb": {"0": 9600.0}, "checkpoints_written": False,
                 "exit_class": 0},
    "thresholds": {"rate_floor_steps_per_second": 0.945,
                   "vanl_reference_steps_per_second": 1.05,
                   "peak_vram_ceiling_mb": 36500.0},
    "checks": {"banner_seen": True, "no_checkpoint_written": True,
               "steps_completed": True, "rate_at_least_0.9x_VANL": True,
               "peak_vram_measured": True, "peak_vram_within_rung_floor": True,
               "exit_class_zero": True, "torchrun_ok": True, "tee_ok": True,
               "wandb_provenance_ok": True, "preflight_transcript_ok": True},
    "verdict": "PASS",
}
if mutation:
    exec(mutation)
json.dump(r, open(path, "w"), indent=2, sort_keys=True)
PY
}
grep -q 'CHECKPOINT_EVERY="${SMOKE_CHECKPOINT_EVERY:-$((MAXSTEPS + 1))}"' "$LAUNCHER"
check "the smoke's checkpoint interval defaults beyond its last step" $?
case_run "a smoke whose interval would write a checkpoint dies" 2 "must write no checkpoints" \
  -- "${SMOKE_ENV[@]}" ARM=YAWAUG SMOKE_MAXSTEPS=30 SMOKE_CHECKPOINT_EVERY=10
grep -q 'SMOKE STORAGE VIOLATION' "$LAUNCHER"; check "the epilogue fails a smoke that wrote a checkpoint" $?
grep -q 'yaw_aug_smoke_acceptance.json' "$LAUNCHER"; check "the smoke writes an acceptance record" $?
grep -q 'rate_at_least_0.9x_VANL' "$LAUNCHER"; check "  ... scored against the 0.9x VANL rate floor" $?
grep -q 'peak_vram_within_rung_floor' "$LAUNCHER"; check "  ... and the rung's VRAM ceiling" $?
rm -f "$ACC"
sub_case "production is REFUSED with no acceptance record" 2 "no readable smoke acceptance record" -- \
  env DRYRUN=1 bash "$SUB_SPOOL" YAWAUG
mk_record "$ACC" 'r["verdict"] = "FAIL"; r["checks"]["banner_seen"] = False'
sub_case "production is REFUSED on a FAIL record" 2 "SMOKE ACCEPTANCE GATE" -- \
  env DRYRUN=1 bash "$SUB_SPOOL" YAWAUG
mk_record "$ACC" 'r["_meta"]["commit"] = "0"*40'
sub_case "production is REFUSED on a record bound to another commit" 2 "_meta.commit" -- \
  env DRYRUN=1 bash "$SUB_SPOOL" YAWAUG
mk_record "$ACC" ""
sub_case "production is ACCEPTED on a bound PASS record" 0 "bound to this submission" -- \
  env DRYRUN=1 bash "$SUB_SPOOL" YAWAUG
sub_case "  ... and the manifest line carries the record hash" 0 "record sha256" -- \
  env DRYRUN=1 bash "$SUB_SPOOL" YAWAUG
rm -f "$ACC"
sub_case "an explicit waiver submits and LOGS the reason" 0 "SMOKE WAIVER" -- \
  env DRYRUN=1 SMOKE_WAIVER="guardtest: no GPUs on this host" bash "$SUB_SPOOL" YAWAUG
sub_case "  ... and the waived run still prints its sbatch line" 0 "DRYRUN sbatch" -- \
  env DRYRUN=1 SMOKE_WAIVER="guardtest: no GPUs on this host" bash "$SUB_SPOOL" YAWAUG
grep -q 'smoke_acceptance ${ACCEPT_FILE:-<n/a>} sha256' "$SUBMITTER"
check "the submission manifest records the record path, its hash and any waiver" $?
sub_case "a SMOKE submission needs no acceptance record" 0 "DRYRUN sbatch" -- \
  env DRYRUN=1 SMOKE=1 SMOKE_RUNG=8x8 SMOKE_MIN_FREE_MB=14000 bash "$SUB_SPOOL" YAWAUG

echo "--- W2. NEW (f3-fix 1): the promotion gate PARSES and BINDS the record ---"
sed -n '/--- BEGIN acceptance-gate-python/,/--- END acceptance-gate-python/p' "$SUBMITTER" > "${TMP}/gate.py"
grep -q 'validate_json_domain' "${TMP}/gate.py"; check "the gate is extracted and imports the recorder's type-domain helper" $?
grep -q "grep -q '\"verdict\": \"PASS\"'" "$SUBMITTER"; check "  ... and the old substring test is gone" $((1 - $?))
REC="${TMP}/acceptance.json"
mk_record "$REC" ""
expect_cmd "a well-formed, bound PASS record is accepted" 0 "" -- gate "$REC"
[ "$(gate "$REC")" = "$(sha256sum "$REC" | awk '{print $1}')" ]
check "  ... and the gate emits the record's same-bytes sha256" $?
printf 'this is not json at all { "verdict": "PASS"\n' > "${TMP}/malformed.json"
expect_cmd "malformed JSON is refused (the substring test would have passed it)" 1 "not parseable JSON" -- \
  gate "${TMP}/malformed.json"
printf '{"x": {"verdict": "PASS"}}\n' > "${TMP}/nested.json"
expect_cmd "a NESTED verdict is refused" 1 "top-level verdict" -- gate "${TMP}/nested.json"
printf '{"verdict": "PASS", "checks": {"a": true}}\n' > "${TMP}/thin.json"
expect_cmd "a record missing whole sections is refused" 1 "is missing or not an object" -- \
  gate "${TMP}/thin.json"
expect_cmd "a missing record file is refused" 1 "no readable smoke acceptance record" -- \
  gate "${TMP}/absent.json"
mk_record "$REC" 'r["_meta"]["commit"] = "0"*40'
expect_cmd "a STALE record from another commit is refused" 1 "_meta.commit" -- gate "$REC"
mk_record "$REC" 'r["_meta"]["rung"] = "16x4"'
expect_cmd "a record from another rung is refused" 1 "_meta.rung" -- gate "$REC"
mk_record "$REC" 'r["_meta"]["ngpu"] = 4'
expect_cmd "a record from another world size is refused" 1 "_meta.ngpu" -- gate "$REC"
mk_record "$REC" 'r["_meta"]["max_steps"] = 500'
expect_cmd "a record from another step budget is refused" 1 "_meta.max_steps" -- gate "$REC"
mk_record "$REC" 'del r["checks"]["banner_seen"]; r["checks"]["banner_seen"] = False'
expect_cmd "any FALSE check is refused" 1 "not true" -- gate "$REC"
mk_record "$REC" 'r["checks"]["tee_ok"] = "true"'
expect_cmd "a STRING 'true' check is refused (literal booleans only)" 1 "not true" -- gate "$REC"
mk_record "$REC" 'r["checks"]["tee_ok"] = 1'
expect_cmd "an INT 1 check is refused" 1 "not true" -- gate "$REC"
mk_record "$REC" 'del r["checks"]'
expect_cmd "a record with no checks at all is refused" 1 "'checks' is missing" -- gate "$REC"
mk_record "$REC" 'r["verdict"] = "FAIL"'
expect_cmd "a FAIL verdict is refused" 1 "not 'PASS'" -- gate "$REC"
mk_record "$REC" 'r["measured"]["peak_vram_mb"] = None'
expect_cmd "a record with no measured VRAM is refused" 1 "peak_vram_mb" -- gate "$REC"
mk_record "$REC" 'r["measured"]["peak_vram_mb"] = float("nan")'
expect_cmd "a non-finite VRAM value is refused by the type domain" 1 "finite" -- gate "$REC"
mk_record "$REC" 'del r["_meta"]["wall_seconds"]'
expect_cmd "a record missing a _meta field is refused" 1 "_meta is missing" -- gate "$REC"
grep -q 'smoke_acceptance ${ACCEPT_FILE:-<n/a>} sha256 ${ACCEPT_SHA256:-<none>}' "$SUBMITTER"
check "the submission manifest pins the record's sha256, not just its path" $?
grep -q 'PINNED_SMOKE_MAXSTEPS' "$LAUNCHER"; check "the smoke's step budget is a launcher pin the gate binds to" $?

echo "--- W3. NEW (f3-fix 2): the producer supersedes, measures and publishes last ---"
sed -n '/--- BEGIN smoke-acceptance-python/,/--- END smoke-acceptance-python/p' "$LAUNCHER" > "${TMP}/producer.py"
grep -q 'peak_vram_measured' "${TMP}/producer.py"; check "the producer is extracted and requires measured VRAM" $?
grep -q 'superseded' "$LAUNCHER"; check "a stale record is superseded before the smoke runs" $?
grep -q 'nvidia-smi --query-gpu=index,memory.used' "$LAUNCHER"; check "a VRAM sampler runs for the life of torchrun" $?
grep -q 'os.replace(tmp, path)' "${TMP}/producer.py"; check "the record is published atomically" $?
T_LINE="$(grep -n -- '--- T\. SMOKE acceptance record' "$LAUNCHER" | cut -d: -f1)"
P_LINE="$(grep -nF 'printf' "$LAUNCHER" | grep 'FINAL_RECORD' | tail -1 | cut -d: -f1)"
CLASSIFY_LINE="$(grep -n 'CLASSIFY_OUT="\$(python3' "$LAUNCHER" | cut -d: -f1)"
[ -n "$T_LINE" ] && [ -n "$P_LINE" ] && [ "$T_LINE" -lt "$P_LINE" ] && [ "$CLASSIFY_LINE" -lt "$T_LINE" ]
check "  ... at the very END of the job, after the classifier and every status" $?
produce() {  # <out> <log> <vram-csv> <final_rc> <torchrun_rc> <tee_rc> <wandb_rc> <preflight_rc> [banner]
  $PY "${TMP}/producer.py" "$1" "$2" "$3" 30 8 36500 "${9:-OK}" 4242 "$HEAD_SHA" 8x8 0 31 \
      "$4" "$5" "$6" "$7" "$8"
}
printf 'Epoch 0: 100%%|##| 30/30 [00:28<00:00,  1.05it/s]\n' > "${TMP}/good_smoke.log"
printf '0, 9400\n1, 9600\n0, 9500\n' > "${TMP}/vram_ok.csv"
: > "${TMP}/vram_empty.csv"
OUT="${TMP}/produced.json"
rm -f "$OUT"
expect_cmd "a clean smoke publishes PASS" 0 "smoke acceptance: PASS" -- \
  produce "$OUT" "${TMP}/good_smoke.log" "${TMP}/vram_ok.csv" 0 0 0 0 0
$PY -c "
import json,sys
r=json.load(open(sys.argv[1]))
sys.exit(0 if r['verdict']=='PASS' and r['measured']['peak_vram_mb']==9600.0
         and r['measured']['peak_vram_per_gpu_mb']=={'0':9500.0,'1':9600.0} else 1)" "$OUT"
check "  ... with the per-GPU peak taken as the max over all samples and ranks" $?
expect_cmd "  ... and the gate accepts what the producer wrote" 0 "" -- gate "$OUT"
rm -f "$OUT"
expect_cmd "NO measured VRAM => FAIL, never pass-by-default" 11 '"peak_vram_measured": false' -- \
  produce "$OUT" "${TMP}/good_smoke.log" "${TMP}/vram_empty.csv" 0 0 0 0 0
expect_cmd "  ... and the gate refuses that record" 1 "" -- gate "$OUT"
rm -f "$OUT"
expect_cmd "a W&B provenance failure => FAIL" 11 '"wandb_provenance_ok": false' -- \
  produce "$OUT" "${TMP}/good_smoke.log" "${TMP}/vram_ok.csv" 0 0 0 7 0
rm -f "$OUT"
expect_cmd "a classifier class (nonzero exit class) => FAIL" 11 '"exit_class_zero": false' -- \
  produce "$OUT" "${TMP}/good_smoke.log" "${TMP}/vram_ok.csv" 4 0 0 0 0
rm -f "$OUT"
expect_cmd "a tee failure => FAIL" 11 '"tee_ok": false' -- \
  produce "$OUT" "${TMP}/good_smoke.log" "${TMP}/vram_ok.csv" 0 0 1 0 0
rm -f "$OUT"
expect_cmd "a MISSING banner => FAIL" 11 '"banner_seen": false' -- \
  produce "$OUT" "${TMP}/good_smoke.log" "${TMP}/vram_ok.csv" 0 0 0 0 0 MISSING
rm -f "$OUT"
printf 'Epoch 0:  33%%|#| 10/30 [00:30<01:00,  3.00s/it]\n' > "${TMP}/slow_smoke.log"
expect_cmd "an unfinished / too-slow smoke => FAIL" 11 '"rate_at_least_0.9x_VANL": false' -- \
  produce "$OUT" "${TMP}/slow_smoke.log" "${TMP}/vram_ok.csv" 0 0 0 0 0
expect_cmd "a record that cannot be written fails loudly" 1 "" -- \
  produce "${TMP}/no_such_dir/rec.json" "${TMP}/good_smoke.log" "${TMP}/vram_ok.csv" 0 0 0 0 0
grep -q 'SMOKE_RECORD_RC" -eq 11' "$LAUNCHER"; check "a FAIL verdict forces class 10" $?
grep -q 'a smoke whose evidence cannot be produced has not passed (class 10)' "$LAUNCHER"
check "  ... and so does a record-write failure" $?

echo "--- X. NEW (full-fix F6): a RESTART leg is recorded in the registry ---"
sed -n '/--- BEGIN registry-restart-python/,/--- END registry-restart-python/p' "$LAUNCHER" > "${TMP}/reg_restart.py"
grep -q 'restarts' "${TMP}/reg_restart.py"; check "restart-registry code extracted from the launcher" $?
RREG="${TMP}/restart_registry.json"
reg_restart() {  # <registry> <job> <resume-step>
  $PY "${TMP}/reg_restart.py" "$1" YAWAUG "${TMP}/man.txt" "$2" "uuid-$2" \
      "${TMP}/resume.ckpt" "deadbeef$2" "$3" 40000 "$HEAD_SHA"
}
: > "${TMP}/resume.ckpt"
$PY -c "
import json,sys
json.dump({'arms':{'YAWAUG':{'final_ckpt_sha256':None}},'restarts':{}}, open(sys.argv[1],'w'))" "$RREG"
expect_cmd "a restart leg is appended" 0 "restart leg recorded" -- reg_restart "$RREG" 777 12500
$PY -c "
import json,sys
legs=json.load(open(sys.argv[1]))['restarts']['YAWAUG']
e=legs[0]
sys.exit(0 if len(legs)==1 and e['job']=='777' and e['resume_step']==12500
         and e['launch_uuid']=='uuid-777' and len(e['manifest_sha256'])==64
         and e['resume_ckpt_sha256'].startswith('deadbeef') else 1)" "$RREG"
check "  ... with job, uuid, manifest sha, source ckpt sha and step" $?
expect_cmd "a second, distinct leg is appended too" 0 "restart leg recorded" -- reg_restart "$RREG" 888 25000
expect_cmd "the same job is never recorded twice" 1 "already recorded" -- reg_restart "$RREG" 888 25000
$PY -c "
import json,sys
json.dump({'arms':{},'restarts':{}}, open(sys.argv[1],'w'))" "${TMP}/orphan_registry.json"
expect_cmd "an orphan restart (no INITIAL) is refused" 1 "orphan restart" -- \
  reg_restart "${TMP}/orphan_registry.json" 999 12500
expect_cmd "a restart with no registry at all is refused" 1 "does not exist" -- \
  reg_restart "${TMP}/no_such_registry.json" 999 12500

echo "--- Y. NEW (chain): per-leg budgets, guards, and byte-compatible monolith ---"
CHAIN_ROOT="${OUT_ROOT}/chain"
CHAIN_RUN="${CHAIN_ROOT}/exp15_YAWAUG/FLAC_exp15_YAWAUG/exp15_YAWAUG/checkpoints"
mkdir -p "$CHAIN_RUN"
for STEP in 2500 12500 37500; do : > "${CHAIN_RUN}/epoch=0-step=${STEP}.ckpt"; done
CH_ENV=(DRYRUN=1 CHAIN=1 ARM=YAWAUG "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${CHAIN_ROOT}" "${REPO_ENV[@]}")
# An INITIAL leg refuses a pre-existing run dir (by design), and the boundary
# checkpoints above live in exactly such a dir — so leg 1 gets a fresh root.
CH_INIT=(DRYRUN=1 CHAIN=1 ARM=YAWAUG "EXPECT_SHA=${HEAD_SHA}" "OUTPUT_ROOT=${OUT_ROOT}/chain_init" "${REPO_ENV[@]}")
# leg math
case_run "chain INITIAL is 0 -> 2500" 0 "chain leg: 0 -> 2500 of 40000" -- "${CH_INIT[@]}"
case_run "  ... and passes --max-steps 2500 to train.py" 0 "--max-steps 2500" -- "${CH_INIT[@]}"
case_run "  ... under the per-leg wall pin" 0 "time pin PINNED_TIME_LIMIT_LEG=1:30:00" -- "${CH_INIT[@]}"
case_run "a mid chain leg is 12500 -> 15000" 0 "chain leg: 12500 -> 15000 of 40000" \
  -- "${CH_ENV[@]}" EXPECTED_STEP=12500 "RESUME_CKPT=${CHAIN_RUN}/epoch=0-step=12500.ckpt"
case_run "the FINAL leg is 37500 -> 40000" 0 "chain leg: 37500 -> 40000 of 40000" \
  -- "${CH_ENV[@]}" EXPECTED_STEP=37500 "RESUME_CKPT=${CHAIN_RUN}/epoch=0-step=37500.ckpt"
# LEG_STEPS is PINNED (chain review F6): one wall pin means one leg size.
case_run "a larger aligned LEG_STEPS is now REFUSED" 2 "!= the pinned 2500" \
  -- "${CH_ENV[@]}" LEG_STEPS=10000 EXPECTED_STEP=2500 "RESUME_CKPT=${CHAIN_RUN}/epoch=0-step=2500.ckpt"
case_run "  ... because the 1:30 wall pin is sized for 2500 steps" 2 "reviewed time pin" \
  -- "${CH_INIT[@]}" LEG_STEPS=5000
case_run "the last leg still clamps to the cap" 0 "chain leg: 37500 -> 40000 of 40000" \
  -- "${CH_ENV[@]}" EXPECTED_STEP=37500 "RESUME_CKPT=${CHAIN_RUN}/epoch=0-step=37500.ckpt"
# guards
case_run "a misaligned LEG_STEPS dies" 2 "!= the pinned 2500" -- "${CH_INIT[@]}" LEG_STEPS=1000
case_run "a zero LEG_STEPS dies" 2 "!= the pinned 2500" -- "${CH_INIT[@]}" LEG_STEPS=0
case_run "an absurdly long LEG_STEPS is refused before any arithmetic" 2 "absurdly long" \
  -- "${CH_INIT[@]}" LEG_STEPS=999999999999999999999
case_run "a misaligned --expected-step dies" 2 "not on the 2500-step cadence" \
  -- "${CH_ENV[@]}" EXPECTED_STEP=3000 "RESUME_CKPT=${CHAIN_RUN}/epoch=0-step=2500.ckpt"
case_run "resuming AT the cap still dies" 2 "at/past the pre-registered" \
  -- "${CH_ENV[@]}" EXPECTED_STEP=40000 "RESUME_CKPT=${CHAIN_RUN}/epoch=0-step=37500.ckpt"
CAPBUST="$(spool capbust 's/^PINNED_CHAIN_CAP=40000/PINNED_CHAIN_CAP=50000/')"
case_spool "a chain cap that is not the registered endpoint dies" "$CAPBUST" 2 \
  "the pre-registered endpoint is 40000" -- "${CH_INIT[@]}"
# the monolith is untouched
case_run "CHAIN unset still trains the full 40000" 0 "max_steps 40000" \
  -- "${BASE_ENV[@]}" ARM=YAWAUG
case_run "  ... under the monolithic wall pin" 0 "time pin PINNED_TIME_LIMIT_YAWAUG=24:00:00" \
  -- "${BASE_ENV[@]}" ARM=YAWAUG
# The golden is captured from the PRE-CHAIN reviewed launcher (44df1a2), not from
# HEAD — comparing HEAD against HEAD proves nothing (chain review, finding 7).
# OUTPUT_ROOT is a fixed literal and the repo path is normalised, so the only
# thing that can differ is the argv itself. Emptiness is asserted: an empty
# comparison is the classic vacuous green.
GOLDEN_ARGV_FILE="${EXPDIR}/yaw_aug_monolith_argv_golden.txt"
capture_argv() {  # <launcher> [chain-value]
  env DRYRUN=1 ARM=YAWAUG "EXPECT_SHA=${HEAD_SHA}" OUTPUT_ROOT=/GOLDEN \
      "YAW_AUG_REPO_OVERRIDE=${PWD}" ${2:+CHAIN=$2} bash "$1" 2>&1 \
    | sed -n '/^--model-config/p' | sed "s|${PWD}|<REPO>|g"
}
GOLDEN_ARGV="$(cat "$GOLDEN_ARGV_FILE" 2>/dev/null)"
[ -n "$GOLDEN_ARGV" ]; check "the pre-chain monolith argv golden is present and non-empty" $?
MONO_UNSET="$(capture_argv "$LAUNCHER")"
[ -n "$MONO_UNSET" ] && [ "$MONO_UNSET" = "$GOLDEN_ARGV" ]
check "CHAIN unset builds the PRE-CHAIN argv byte-for-byte (44df1a2 golden)" $?
MONO_ZERO="$(capture_argv "$LAUNCHER" 0)"
[ -n "$MONO_ZERO" ] && [ "$MONO_ZERO" = "$GOLDEN_ARGV" ]
check "CHAIN=0 builds the same pre-chain argv" $?
CHAIN_ARGV="$(capture_argv "$LAUNCHER" 1)"
[ -n "$CHAIN_ARGV" ] && [ "$CHAIN_ARGV" != "$GOLDEN_ARGV" ]
check "  ... while CHAIN=1 genuinely differs (the comparison can detect change)" $?

echo "--- Z. NEW (chain): the self-chaining epilogue ---"
# The unlocked next-leg helper is GONE: submission is one transaction inside
# yaw_aug_chain_state.py (section AE drives it, including the concurrency race).
grep -q 'BEGIN next-leg-helper' "$LAUNCHER"; check "the unlocked next-leg helper is retired" $((1 - $?))
grep -q 'transact-submit' "$LAUNCHER"; check "  ... replaced by the locked transaction" $?
grep -q 'chain END: this leg reached the pre-registered cap' "$LAUNCHER"
check "the FINAL leg submits no successor" $?
grep -q -- '--submitter "${REPO}/${SUBMITTER_REL}"' "$LAUNCHER"
check "the successor goes through the LIVE submitter (it re-gates at then-current HEAD)" $?
# leg-aware completion audit: only the cap leg closes the run out
fresh_reg "${TMP}/chain_registry.json"
expect_cmd "a mid-chain leg records a boundary, not a final checkpoint" 0 "leg boundary at step 40000" -- \
  $PY "${TMP}/completion.py" "${EXPDIR}/yaw_aug_record_control.py" "${TMP}/chain_registry.json" \
      YAWAUG "$CK_DIR" 40000 "$ARM_CONFIG" 50000 1
$PY -c "
import json,sys
d=json.load(open(sys.argv[1]))
legs=d['legs']['YAWAUG']
sys.exit(0 if len(legs)==1 and legs[0]['step']==40000 and legs[0]['chain'] is True
         and d['arms']['YAWAUG'].get('final_ckpt_sha256') is None else 1)" "${TMP}/chain_registry.json"
check "  ... so final_ckpt_sha256 stays unset until the cap leg" $?
fresh_reg "${TMP}/chain_registry2.json"
expect_cmd "the CAP leg ends the chain" 0 "CHAIN END" -- \
  $PY "${TMP}/completion.py" "${EXPDIR}/yaw_aug_record_control.py" "${TMP}/chain_registry2.json" \
      YAWAUG "$CK_DIR" 40000 "$ARM_CONFIG" 40000 1
$PY -c "
import json,sys
e=json.load(open(sys.argv[1]))['arms']['YAWAUG']
sys.exit(0 if e['final_step']==40000 and len(e['final_ckpt_sha256'])==64 else 1)" "${TMP}/chain_registry2.json"
check "  ... writing final_ckpt_sha256/final_step from the checkpoint" $?

echo "--- Z2. NEW (chain): the submitter's chain surface ---"
sed -n '/--- BEGIN chain-initial-manifest-python/,/--- END chain-initial-manifest-python/p' "$SUBMITTER" > "${TMP}/chain_initial.sh"
grep -q 'chain_initial_manifest()' "${TMP}/chain_initial.sh"; check "the initial-manifest reader is extracted" $?
initial_manifest() { ( . "${TMP}/chain_initial.sh"; chain_initial_manifest "$1" YAWAUG 40000 2500 ); }
# A valid INITIAL: readable manifest, registered sha, mode INITIAL on BOTH sides,
# and chain metadata describing THIS chain.
INIT_MAN="${TMP}/valid_initial_manifest.txt"
printf '%s\n' "# exp_15 arm launch manifest" \
  "job 1 host h mode INITIAL launch_uuid u" \
  "arm YAWAUG rung 8x8 micro 8 ngpu 8 max_steps 2500 ckpt_every 2500" \
  "chain 1 leg_steps 2500 leg_start 0 leg_target 2500 cap 40000" > "$INIT_MAN"
$PY -c "
import hashlib, json, sys
man = sys.argv[2]
json.dump({'arms': {'YAWAUG': {'manifest_path': man,
           'manifest_sha256': hashlib.sha256(open(man,'rb').read()).hexdigest(),
           'mode': 'INITIAL', 'chain': True, 'chain_cap': 40000,
           'chain_leg_steps': 2500, 'chain_initial_target': 2500}}, 'restarts': {}},
          open(sys.argv[1], 'w'))" "${TMP}/chain_reg_ok.json" "$INIT_MAN"
expect_cmd "a RESTART leg finds the validated INITIAL manifest" 0 "valid_initial_manifest" -- \
  initial_manifest "${TMP}/chain_reg_ok.json"
$PY -c "
import json,sys
d=json.load(open(sys.argv[1])); d['arms']['YAWAUG']['mode']='RESTART'
json.dump(d, open(sys.argv[2],'w'))" "${TMP}/chain_reg_ok.json" "${TMP}/chain_reg_mode.json"
expect_cmd "  ... and refuses when the registry mode is not INITIAL" 1 "both must be INITIAL" -- \
  initial_manifest "${TMP}/chain_reg_mode.json"
$PY -c "
import json,sys
d=json.load(open(sys.argv[1])); del d['arms']['YAWAUG']['chain_cap']
json.dump(d, open(sys.argv[2],'w'))" "${TMP}/chain_reg_ok.json" "${TMP}/chain_reg_nometa.json"
expect_cmd "  ... refuses missing chain metadata" 1 "this chain is" -- \
  initial_manifest "${TMP}/chain_reg_nometa.json"
printf 'tamper\n' >> "$INIT_MAN"
expect_cmd "  ... and refuses a manifest tampered after registration" 1 "no longer hashes" -- \
  initial_manifest "${TMP}/chain_reg_ok.json"
$PY -c "
import hashlib,json,sys
man=sys.argv[2]
d=json.load(open(sys.argv[1]))
d['arms']['YAWAUG']['manifest_sha256']=hashlib.sha256(open(man,'rb').read()).hexdigest()
json.dump(d, open(sys.argv[1],'w'))" "${TMP}/chain_reg_ok.json" "$INIT_MAN"
$PY -c "
import json,sys
json.dump({'arms':{},'restarts':{}}, open(sys.argv[1],'w'))" "${TMP}/chain_reg_empty.json"
expect_cmd "a RESTART leg with no registered INITIAL is refused" 1 "presupposes a registered first leg" -- \
  initial_manifest "${TMP}/chain_reg_empty.json"
expect_cmd "a missing registry is refused" 1 "cannot read the launch registry" -- \
  initial_manifest "${TMP}/no_such_registry.json"
grep -q 'chain_standing_waiver ${STANDING_WAIVER_REF}' "$SUBMITTER"
check "every chain manifest cites the standing waiver" $?
grep -q 'chain_initial_manifest ${CHAIN_INITIAL_MANIFEST' "$SUBMITTER"
check "  ... and a RESTART leg names the INITIAL manifest" $?
grep -q 'CHAIN=1,LEG_STEPS=' "$SUBMITTER"; check "CHAIN and LEG_STEPS are exported to the job" $?
sub_case "the submitter refuses a misaligned LEG_STEPS" 2 "checkpoint cadence" -- \
  env DRYRUN=1 CHAIN=1 LEG_STEPS=1234 bash "$SUB_SPOOL" YAWAUG
sub_case "the submitter refuses a misaligned --expected-step" 2 "no boundary checkpoint can exist" -- \
  env DRYRUN=1 CHAIN=1 bash "$SUB_SPOOL" YAWAUG --resume /x/c.ckpt --expected-step 3000
sub_case "the submitter refuses a resume at the cap" 2 "chain is already complete" -- \
  env DRYRUN=1 CHAIN=1 bash "$SUB_SPOOL" YAWAUG --resume /x/c.ckpt --expected-step 40000

echo "--- AA. NEW (chain-fix F3): advancement happens LAST, and only when green ---"
sed -n '/--- BEGIN chain-advance-helper/,/--- END chain-advance-helper/p' "$LAUNCHER" > "${TMP}/advance.sh"
grep -q 'chain_advance()' "${TMP}/advance.sh"; check "the advancement block is extracted from the launcher" $?
# Functional: drive the REAL block with stubbed collaborators and assert whether
# a successor was submitted. Static line-order checks cannot see a later class.
cat > "${TMP}/drive_advance.sh" <<'EOS'
set -uo pipefail
# chain_advance captures the transaction's output via $(...), so a stub variable
# assignment dies in that subshell — the stub signals through a MARKER FILE.
rm -f "${WORKDIR}/tx_submit.marker"
snap() { echo "${TMP_HELPERS}/$1"; }
chain_state() {
  case "$1" in
    transact-submit) : > "${WORKDIR}/tx_submit.marker"; echo "SUBMITTED 4242"; return "${SUBMIT_RC:-0}" ;;
    *) echo "chain_state $*"; return "${CHAIN_STATE_RC:-0}" ;;
  esac
}
ARM=YAWAUG; SAVEDIR="$WORKDIR"; LEG_STEPS=2500; PINNED_CHAIN_CAP=40000
SUBMITTER_REL="x/submit.sh"; REPO="."; SLURM_JOB_ID=777; TRAINLOG="${TRAINLOG:-/dev/null}"
MAXSTEPS="${MAXSTEPS:-2500}"; EXPECTED_STEP="${EXPECTED_STEP:-0}"
AUDITED_SHA256="${AUDITED_SHA256-abc}"; AUDITED_CKPT=/x/c.ckpt; AUDITED_PARENT_STEP=0
final_rc="$RC"
. "$ADVANCE"
chain_advance > "${WORKDIR}/advance.out" 2>&1
SUBMITTED=0; [ -e "${WORKDIR}/tx_submit.marker" ] && SUBMITTED=1
echo "final_rc=${final_rc} submitted=${SUBMITTED}"
EOS
drive() {  # <rc> [env...] -> "final_rc=<n> submitted=<0|1>"
  local rc="$1"; shift
  env RC="$rc" ADVANCE="${TMP}/advance.sh" WORKDIR="$TMP" TMP_HELPERS="$EXPDIR" "$@" \
      bash "${TMP}/drive_advance.sh"
}
for RC in 7 8 9; do
  expect_cmd "a class-${RC} leg submits NO successor" 0 "submitted=0" -- drive "$RC"
done
expect_cmd "  ... and keeps its own exit class (7)" 0 "final_rc=7 submitted=0" -- drive 7
expect_cmd "the FINAL (cap) leg submits no successor" 0 "submitted=0" -- \
  drive 0 MAXSTEPS=40000 EXPECTED_STEP=37500
grep -q "reached the pre-registered cap" "${TMP}/advance.out"
check "  ... and says the chain ENDed" $?
expect_cmd "a leg with no audited checkpoint submits nothing (class 12)" 0 "final_rc=12 submitted=0" -- \
  drive 0 AUDITED_SHA256=
# green mid-chain leg WITH a passing rate gate -> exactly one submission
printf 'Epoch 0:   2%%| | 100/4550 [01:00<10:00,  1.00it/s]\nEpoch 0:   6%%| | 300/4550 [04:32<10:00,  1.00it/s]\nEpoch 0:  22%%| | 1000/4550 [17:32<10:00,  1.00it/s]\n' > "${TMP}/fast.log"
expect_cmd "a green leg with a passing rate gate submits its successor" 0 "submitted=1" -- \
  drive 0 "TRAINLOG=${TMP}/fast.log"
grep -q "SUBMITTED 4242" "${TMP}/advance.out"; check "  ... exactly once, through the transaction" $?
grep -q -- "--dependency=\${CHAIN_DEPENDENCY}" "$SUBMITTER"
check "the successor carries a Slurm afterok dependency" $?
grep -q -- '--dependency "afterok:${SLURM_JOB_ID}"' "$LAUNCHER"
check "  ... naming the parent job" $?

echo "--- AB. NEW (chain-fix F5): the waiver's post-hoc rate gate ---"
RG="${EXPDIR}/yaw_aug_rate_gate.py"
rate_gate() { $PY "$RG" --log "$1" --out "${TMP}/rate.json" --leg-target "${2:-2500}"; }
expect_cmd "clean windows PASS" 0 "rate gate: PASS" -- rate_gate "${TMP}/fast.log"
$PY -c "
import json,sys
w=json.load(open(sys.argv[1]))['windows']
sys.exit(0 if abs(w[0]['rate_steps_per_second']-0.943)<0.01 else 1)" "${TMP}/rate.json"
check "  ... reproducing VANL's own 0.943 for window 1" $?
printf 'Epoch 0:   2%%| | 100/4550 [01:00<10:00,  1.00it/s]\nEpoch 0:   6%%| | 300/4550 [06:00<10:00,  1.00it/s]\nEpoch 0:  22%%| | 1000/4550 [19:00<10:00,  1.00it/s]\n' > "${TMP}/slow1.log"
expect_cmd "a breach in window 1 (100->300) refuses" 1 "BREACH" -- rate_gate "${TMP}/slow1.log"
printf 'Epoch 0:   2%%| | 100/4550 [01:00<10:00,  1.00it/s]\nEpoch 0:   6%%| | 300/4550 [04:32<10:00,  1.00it/s]\nEpoch 0:  22%%| | 1000/4550 [21:00<10:00,  1.00it/s]\n' > "${TMP}/slow2.log"
expect_cmd "a breach in window 2 (300->1000) refuses" 1 "BREACH" -- rate_gate "${TMP}/slow2.log"
printf 'Epoch 0:   2%%| | 100/4550 [01:00<10:00,  1.00it/s]\n' > "${TMP}/thin.log"
expect_cmd "missing bar entries REFUSE (never pass by default)" 2 "INSUFFICIENT DATA" -- \
  rate_gate "${TMP}/thin.log"
: > "${TMP}/empty.log"
expect_cmd "an empty log refuses" 2 "INSUFFICIENT DATA" -- rate_gate "${TMP}/empty.log"
grep -q 'final_rc=13' "$LAUNCHER"; check "a rate-gate refusal stops the chain on class 13" $?
grep -q '#  13  CHAIN only' "$LAUNCHER"; check "  ... documented in the exit taxonomy" $?

echo "--- AC. NEW (chain-fix F4): submission is idempotent ---"
CS="${EXPDIR}/yaw_aug_chain_state.py"
st() { $PY "$CS" --state "${TMP}/chain_state.json" --arm YAWAUG "$@"; }
rm -f "${TMP}/chain_state.json"
expect_cmd "a boundary is recorded AUDITED" 0 "AUDITED" -- \
  st record-audit --target 2500 --ckpt-sha256 aa --ckpt-path /x/c.ckpt --parent-step 0 --job 1
expect_cmd "  ... idempotently on replay" 0 "already AUDITED" -- \
  st record-audit --target 2500 --ckpt-sha256 aa --ckpt-path /x/c.ckpt --parent-step 0 --job 1
expect_cmd "  ... but a DIFFERENT checkpoint at that boundary is refused" 1 "different checkpoint" -- \
  st record-audit --target 2500 --ckpt-sha256 bb --ckpt-path /x/c.ckpt --parent-step 0 --job 1
TOKEN1="$(st intend --target 2500 --next-target 5000 --command CMD | awk '{print $2}')"
TOKEN2="$(st intend --target 2500 --next-target 5000 --command CMD | awk '{print $2}')"
[ -n "$TOKEN1" ] && [ "$TOKEN1" = "$TOKEN2" ]
check "an INTENDED boundary replays the SAME intent token" $?
expect_cmd "intending an unaudited boundary is refused" 1 "has not been audited" -- \
  st intend --target 7500 --next-target 10000 --command CMD
st mark-submitted --target 2500 --jid 900001 >/dev/null
expect_cmd "a SUBMITTED boundary refuses to be intended again" 3 "ALREADY_SUBMITTED 900001" -- \
  st intend --target 2500 --next-target 5000 --command CMD
expect_cmd "  ... and marking it with a different jid is refused" 1 "already SUBMITTED as job" -- \
  st mark-submitted --target 2500 --jid 900002
expect_cmd "  ... while re-marking the same jid is idempotent" 0 "already SUBMITTED" -- \
  st mark-submitted --target 2500 --jid 900001
expect_cmd "status reports the boundary" 0 "SUBMITTED" -- st status --target 2500
grep -q 'find_job_by_name' "$CS"; check "a post-sbatch crash is recovered by finding the intent token in the queue" $?
grep -q 'CHAIN_INTENT_TOKEN' "$SUBMITTER"; check "  ... which the submitter puts in the job name" $?
# registry legs: tip-bound, monotonic, idempotent, CHAIN-only
fresh_reg "${TMP}/legs_reg.json"
leg_audit() { $PY "${TMP}/completion.py" "${EXPDIR}/yaw_aug_record_control.py" "$1" YAWAUG \
                 "$CK_DIR" 40000 "$ARM_CONFIG" "$2" "$3"; }
expect_cmd "a chain leg appends a tip-bound entry" 0 "leg boundary at step 40000" -- \
  leg_audit "${TMP}/legs_reg.json" 50000 1
expect_cmd "  ... idempotently for the same checkpoint" 0 "already audited" -- \
  leg_audit "${TMP}/legs_reg.json" 50000 1
$PY -c "
import json,sys
legs=json.load(open(sys.argv[1]))['legs']['YAWAUG']
sys.exit(0 if len(legs)==1 and legs[0]['parent_step']==0 and 'parent_ckpt_sha256' in legs[0] else 1)" \
  "${TMP}/legs_reg.json"
check "  ... naming the parent step and sha it continues" $?
fresh_reg "${TMP}/mono_reg.json"
expect_cmd "a MONOLITHIC run writes no legs and no chain wording" 0 "final_ckpt_sha256 recorded" -- \
  leg_audit "${TMP}/mono_reg.json" 40000 0
$PY -c "
import json,sys
d=json.load(open(sys.argv[1]))
sys.exit(0 if 'legs' not in d else 1)" "${TMP}/mono_reg.json"
check "  ... keeping the reviewed monolith registry shape (F7)" $?
grep -q "CHAIN END" "${TMP}/mono_reg.json" 2>/dev/null; check "  ... and no CHAIN END in its output" $((1 - $?))

echo "--- AD. NEW (chain-fix F1/F2): chain preflight and W&B lineage ---"
CP="${EXPDIR}/yaw_aug_chain_preflight.py"
$PY - "$TMP" "$ARM_CONFIG" <<'PY'
import hashlib, json, os, sys, torch
tmp, cfg_path = sys.argv[1], sys.argv[2]
cfg = json.load(open(cfg_path))
state = {"diffusion.model.layer.weight": torch.zeros(2, 2),
         "diffusion.model.layer.bias": torch.zeros(2),
         "diffusion_ema.ema_model.layer.weight": torch.zeros(2, 2),
         "diffusion_ema.ema_model.layer.bias": torch.zeros(2)}
ck = {"global_step": 2500, "epoch": 0, "model_config": cfg, "state_dict": state,
      "optimizer_states": [{"state": {0: {"step": 1}}, "param_groups": [{"lr": 1e-5}]}],
      "lr_schedulers": [{"last_epoch": 2500}]}
path = os.path.join(tmp, "boundary.ckpt")
torch.save(ck, path)
sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
cfg_sha = hashlib.sha256(open(cfg_path, "rb").read()).hexdigest()
man = os.path.join(tmp, "initial_manifest.txt")
open(man, "w").write(
    "# exp_15 arm launch manifest\njob 1 host h mode INITIAL launch_uuid u\n"
    "arm YAWAUG rung 8x8 micro 8 ngpu 8 max_steps 2500 ckpt_every 2500\n"
    "chain 1 leg_steps 2500 leg_start 0 leg_target 2500 cap 40000\n"
    f"config_sha256 {cfg_sha}\nvae_sha256 VAESHA\nwandb_run_id r\n")
reg = {"arms": {"YAWAUG": {"manifest_path": man,
                           "manifest_sha256": hashlib.sha256(open(man, "rb").read()).hexdigest(),
                           "mode": "INITIAL", "rung": "8x8", "training_seed": 42,
                           "config_sha256": cfg_sha, "vae_sha256": "VAESHA",
                           "chain": True, "chain_cap": 40000, "chain_leg_steps": 2500,
                           "chain_initial_target": 2500}},
       "legs": {"YAWAUG": [{"step": 2500, "ckpt_sha256": sha}]}, "restarts": {}}
json.dump(reg, open(os.path.join(tmp, "chain_registry.json"), "w"), indent=2)
bad = dict(reg); bad = json.loads(json.dumps(reg))
bad["legs"]["YAWAUG"][0]["ckpt_sha256"] = "c" * 64
json.dump(bad, open(os.path.join(tmp, "chain_registry_badsha.json"), "w"), indent=2)
old = json.loads(json.dumps(reg)); old["legs"]["YAWAUG"][0]["step"] = 5000
json.dump(old, open(os.path.join(tmp, "chain_registry_badtip.json"), "w"), indent=2)
print("chain preflight fixture written")
PY
pre() { $PY "$CP" --ckpt "${TMP}/boundary.ckpt" --expected-step 2500 --target "${2:-5000}" \
          --cap 40000 --config "$ARM_CONFIG" --arm YAWAUG --rung 8x8 --vae-sha256 VAESHA \
          --launch-manifest "${TMP}/initial_manifest.txt" --registry "$1" \
          --recorder "${EXPDIR}/yaw_aug_record_control.py"; }
expect_cmd "the chain preflight admits a leg whose budget GROWS" 0 "CKPT_SHA256" -- \
  pre "${TMP}/chain_registry.json" 5000
expect_cmd "  ... binding the original launch identity" 0 "launch identity bound" -- \
  pre "${TMP}/chain_registry.json" 5000
expect_cmd "a checkpoint that is not the audited tip's sha is refused" 1 "not the checkpoint the chain recorded" -- \
  pre "${TMP}/chain_registry_badsha.json" 5000
expect_cmd "resuming a step that is not the tip is refused" 1 "the chain would fork" -- \
  pre "${TMP}/chain_registry_badtip.json" 5000
expect_cmd "a target beyond the cap is refused" 1 "does not advance within the cap" -- \
  pre "${TMP}/chain_registry.json" 42500
expect_cmd "a misaligned target is refused" 1 "not on the 2500-step cadence" -- \
  pre "${TMP}/chain_registry.json" 6000
$PY -c "
import json,sys
d=json.load(open(sys.argv[1])); d['arms']['YAWAUG']['mode']='RESTART'
json.dump(d, open(sys.argv[2],'w'), indent=2)" "${TMP}/chain_registry.json" "${TMP}/chain_registry_mode.json"
expect_cmd "a registry entry not marked INITIAL is refused (BOTH must agree)" 1 "not INITIAL" -- \
  pre "${TMP}/chain_registry_mode.json" 5000
$PY -c "
import json,sys
d=json.load(open(sys.argv[1])); del d['arms']['YAWAUG']['chain_leg_steps']
json.dump(d, open(sys.argv[2],'w'), indent=2)" "${TMP}/chain_registry.json" "${TMP}/chain_registry_nometa.json"
expect_cmd "missing chain metadata is refused" 1 "this chain is" -- \
  pre "${TMP}/chain_registry_nometa.json" 5000
grep -q 'snap yaw_aug_chain_preflight.py' "$LAUNCHER"
check "the launcher uses the exp_15 preflight for chain legs" $?
grep -q 'WANDB_RESUME=must' "$LAUNCHER"; check "no chain leg resumes a W&B run (WANDB_RESUME is gone)" $((1 - $?))
grep -q 'WANDB_RUN_ID="exp15_${ARM}_leg${MAXSTEPS}-' "$LAUNCHER"
check "every RESTART leg mints a fresh lineage-tagged W&B id" $?
grep -q 'parent_wandb_run_id' "$LAUNCHER"; check "  ... recording the parent id in the manifest" $?

echo "--- AE. NEW (chain-fix2): submission is a single transaction ---"
TX="${TMP}/tx"; mkdir -p "$TX"
cat > "${TX}/stub_slow_submit.sh" <<'EOS'
#!/usr/bin/env bash
sleep 2
echo "submitted YAWAUG -> job 5150$$" >> "$SUBMIT_LOG"
echo "submitted YAWAUG -> job 5150$$"
EOS
cat > "${TX}/squeue_none.sh" <<'EOS'
#!/usr/bin/env bash
exit 0
EOS
cat > "${TX}/sacct_none.sh" <<'EOS'
#!/usr/bin/env bash
exit 0
EOS
cat > "${TX}/squeue_fail.sh" <<'EOS'
#!/usr/bin/env bash
echo "slurm_load_jobs error: Unable to contact slurm controller" >&2
exit 1
EOS
cat > "${TX}/squeue_pending.sh" <<'EOS'
#!/usr/bin/env bash
echo "9001 PENDING"
EOS
cat > "${TX}/squeue_running.sh" <<'EOS'
#!/usr/bin/env bash
echo "9002 RUNNING"
EOS
cat > "${TX}/sacct_failed.sh" <<'EOS'
#!/usr/bin/env bash
echo "FAILED"
EOS
cat > "${TX}/sacct_completed.sh" <<'EOS'
#!/usr/bin/env bash
echo "COMPLETED"
EOS
chmod +x "${TX}"/*.sh
tx_state() { $PY "$CS" --state "$1" --arm YAWAUG "${@:2}"; }
tx_seed() { rm -f "$1"; tx_state "$1" record-audit --target 2500 --ckpt-sha256 aa \
              --ckpt-path /x/c.ckpt --parent-step 0 --parent-ckpt-sha256 pp --job 1 >/dev/null; }
tx_submit() {  # <state> <squeue> <sacct> [extra env...]
  local st="$1" sq="$2" sa="$3"; shift 3
  env YAW_AUG_SQUEUE_CMD="$sq" YAW_AUG_SACCT_CMD="$sa" "$@" \
      $PY "$CS" --state "$st" --arm YAWAUG transact-submit --target 2500 --next-target 5000 \
        --submitter "${TX}/stub_slow_submit.sh" --resume /x/c.ckpt --leg-steps 2500 \
        --dependency afterok:999
}
# THE race: two concurrent transactions against a deliberately slow submitter.
# wait is PID-scoped: under bash >= 5.1 a bare `wait` also waits for the suite's
# own `exec > >(tee ...)` process substitution, which never exits until the suite
# does — a guaranteed deadlock (found the hard way: four hung runs at this line).
tx_seed "${TX}/race.json"
: > "${TX}/submits.log"
RACE_PIDS=()
for _ in 1 2; do
  ( tx_submit "${TX}/race.json" "${TX}/squeue_none.sh" "${TX}/sacct_none.sh" \
      "SUBMIT_LOG=${TX}/submits.log" >> "${TX}/race.out" 2>&1 ) & RACE_PIDS+=("$!")
done
wait "${RACE_PIDS[@]}"
[ "$(grep -c . "${TX}/submits.log")" = "1" ]
check "two concurrent transactions produce EXACTLY ONE submission" $?
grep -q "ALREADY_SUBMITTED" "${TX}/race.out"
check "  ... and the loser returns the winner's job id, not a second submit" $?
RACE_JID="$(awk '/^SUBMITTED /{print $2}' "${TX}/race.out")"
grep -q "ALREADY_SUBMITTED ${RACE_JID}" "${TX}/race.out"
check "  ... the SAME job id both times" $?
# scheduler failure is fatal
tx_seed "${TX}/schedfail.json"
: > "${TX}/submits2.log"
expect_cmd "a FAILING squeue refuses (never 'no job, submit')" 1 "SCHEDULER QUERY FAILED" -- \
  tx_submit "${TX}/schedfail.json" "${TX}/squeue_fail.sh" "${TX}/sacct_none.sh" \
    "SUBMIT_LOG=${TX}/submits2.log"
[ ! -s "${TX}/submits2.log" ]; check "  ... and nothing was submitted" $?
# a stranded child is reported, not adopted, not resubmitted, not cancelled
tx_seed "${TX}/stranded.json"
: > "${TX}/submits3.log"
expect_cmd "a child whose afterok parent FAILED is STRANDED" 4 "STRANDED 9001" -- \
  tx_submit "${TX}/stranded.json" "${TX}/squeue_pending.sh" "${TX}/sacct_failed.sh" \
    "SUBMIT_LOG=${TX}/submits3.log"
[ ! -s "${TX}/submits3.log" ]; check "  ... nothing is resubmitted" $?
$PY -c "
import json,sys
b=json.load(open(sys.argv[1]))['boundaries']['2500']
sys.exit(0 if b['state']=='INTENDED' and not b.get('next_leg_jid') else 1)" "${TX}/stranded.json"
check "  ... and the boundary is NOT marked SUBMITTED" $?
grep -q 'NOT cancelling it' "$CS"; check "  ... the helper never cancels a job itself" $?
# a live/complete child IS adopted
tx_seed "${TX}/adopt.json"
: > "${TX}/submits4.log"
expect_cmd "a RUNNING token-job is adopted instead of resubmitted" 0 "ADOPTED 9002" -- \
  tx_submit "${TX}/adopt.json" "${TX}/squeue_running.sh" "${TX}/sacct_completed.sh" \
    "SUBMIT_LOG=${TX}/submits4.log"
[ ! -s "${TX}/submits4.log" ]; check "  ... with no new submission" $?
# replay of a SUBMITTED boundary
expect_cmd "replaying a SUBMITTED boundary returns the same jid" 3 "ALREADY_SUBMITTED 9002" -- \
  tx_submit "${TX}/adopt.json" "${TX}/squeue_none.sh" "${TX}/sacct_none.sh" \
    "SUBMIT_LOG=${TX}/submits4.log"
$PY -c "
import json,sys
b=json.load(open(sys.argv[1]))['boundaries']['2500']
cmd=b.get('next_leg_command') or ''
sys.exit(0 if 'transact-submit' in cmd and 'sbatch' not in cmd else 1)" "${TX}/adopt.json"
check "the advertised recovery command is the state-aware operation, not a raw sbatch" $?
$PY -c "
import json,sys
b=json.load(open(sys.argv[1]))['boundaries']['2500']
sys.exit(0 if b.get('parent_ckpt_sha256')=='pp' else 1)" "${TX}/adopt.json"
check "parent_ckpt_sha256 is persisted in the boundary record" $?
grep -q 'chain_state transact-submit' "$LAUNCHER"
check "chain_advance submits ONLY through the transaction" $?
grep -q 'submit_next_leg' "$LAUNCHER"; check "  ... and the old unlocked helper is gone" $((1 - $?))
# ...and a real chain_advance drive persists the parent sha
: > "${TMP}/marks2.txt"
cat > "${TMP}/drive_parent.sh" <<'EOS'
set -uo pipefail
snap() { echo "${HELPERS}/$1"; }
chain_state() { $PY "${HELPERS}/yaw_aug_chain_state.py" --state "$CSF" --arm YAWAUG --cap 40000 "$@"; }
ARM=YAWAUG; SAVEDIR="$WORKDIR"; LEG_STEPS=2500; PINNED_CHAIN_CAP=40000
SUBMITTER_REL="x"; REPO="."; SLURM_JOB_ID=42; TRAINLOG="$FASTLOG"; MAXSTEPS=5000
EXPECTED_STEP=2500; AUDITED_SHA256=bb; AUDITED_CKPT=/x/d.ckpt; AUDITED_PARENT_STEP=2500
AUDITED_PARENT_SHA256=aa; final_rc=0
YAW_AUG_SQUEUE_CMD="$SQ" YAW_AUG_SACCT_CMD="$SA"
export YAW_AUG_SQUEUE_CMD YAW_AUG_SACCT_CMD
. "$ADVANCE"
chain_advance >/dev/null 2>&1
EOS
env PY="$PY" HELPERS="$EXPDIR" CSF="${TX}/advance_state.json" WORKDIR="$TX" \
    FASTLOG="${TMP}/fast.log" ADVANCE="${TMP}/advance.sh" SQ="${TX}/squeue_none.sh" \
    SA="${TX}/sacct_none.sh" SUBMIT_LOG="${TX}/submits5.log" bash "${TMP}/drive_parent.sh" || true
$PY -c "
import json,sys
b=json.load(open(sys.argv[1]))['boundaries']['5000']
sys.exit(0 if b.get('parent_ckpt_sha256')=='aa' and b.get('parent_step')==2500 else 1)" \
  "${TX}/advance_state.json" 2>/dev/null
check "a real chain_advance drive records the parent step AND sha" $?

echo "--- AF. NEW (chain-fix2): terminal status supersedes the stale final record ---"
grep -q 'TERMINAL_RECORD=' "$LAUNCHER"; check "the launcher publishes a terminal status record" $?
awk '/chain_advance$/{a=NR} /TERMINAL_RECORD=/{t=NR} END{exit !(a && t && a < t)}' "$LAUNCHER"
check "  ... AFTER advancement, so it sees classes 12 and 13" $?
grep -q 'os.replace(tmp, path)' "$LAUNCHER"; check "  ... written atomically (tmp + rename)" $?
grep -q 'this supersedes any earlier' "$LAUNCHER"; check "  ... and the printed summary says so" $?
# Drive the launcher's REAL record writer, not a copy: the block is extracted by
# its markers (they are python comments inside the heredoc, so the extract runs).
sed -n "/--- BEGIN terminal-record-python/,/--- END terminal-record-python/p" "$LAUNCHER" > "${TMP}/terminal.py"
[ -s "${TMP}/terminal.py" ] && grep -q 'sys.argv\[1:17\]' "${TMP}/terminal.py"
check "the terminal-record python is extracted from the launcher (non-empty)" $?
term() {  # <out> <final_rc> <banner> <successor> <no-submit-reason> <rate-verdict>
  $PY "${TMP}/terminal.py" "$1" "$2" 0 0 0 "$3" 1 0 2500 40000 "$4" "$5" "$6" "AUDITED" 42 /x/man.txt
}
expect_cmd "a rate-breach leg writes its terminal record" 0 "exit class 13" -- \
  term "${TX}/term_breach.json" 13 MISSING "" "rate gate breach per waiver" "BREACH"
$PY -c "
import json,sys
r=json.load(open(sys.argv[1]))
sys.exit(0 if r['exit_class']==13 and r['chain']['successor_jid'] is None
         and r['chain']['rate_gate_verdict']=='BREACH' else 1)" "${TX}/term_breach.json"
check "  ... class 13, NO successor, verdict BREACH — from the real writer" $?
expect_cmd "a green leg writes its terminal record" 0 "exit class 0" -- \
  term "${TX}/term_green.json" 0 OK 4242 "" "PASS"
$PY -c "
import json,sys
r=json.load(open(sys.argv[1]))
sys.exit(0 if r['exit_class']==0 and r['chain']['successor_jid']=='4242'
         and r['chain']['rate_gate_verdict']=='PASS' else 1)" "${TX}/term_green.json"
check "  ... class 0 WITH the successor jid — from the real writer" $?

echo "--- AG. NEW (chain-fix3): dead children, unconditional identity, terminal truth, parser completeness ---"
# (1) a TERMINAL-FAILED child is NEVER adopted, whatever its parent's state.
cat > "${TX}/sacct_ctl.sh" <<EOS
#!/usr/bin/env bash
if [ -e "${TX}/child_dead.flag" ]; then
  for a in "\$@"; do [ "\$a" = "--name" ] && { echo "9003 CANCELLED"; exit 0; }; done
  echo "COMPLETED"
fi
exit 0
EOS
chmod +x "${TX}/sacct_ctl.sh"
tx_seed "${TX}/dead.json"
: > "${TX}/submits_dead.log"; touch "${TX}/child_dead.flag"
expect_cmd "a CANCELLED child with a HEALTHY parent is CHILD_DEAD, never adopted" 5 "CHILD_DEAD 9003" -- \
  tx_submit "${TX}/dead.json" "${TX}/squeue_none.sh" "${TX}/sacct_ctl.sh" \
    "SUBMIT_LOG=${TX}/submits_dead.log"
[ ! -s "${TX}/submits_dead.log" ]; check "  ... nothing was submitted" $?
$PY -c "
import json,sys
b=json.load(open(sys.argv[1]))['boundaries']['2500']
dead=b.get('dead_children') or []
sys.exit(0 if b['state']=='INTENDED' and not b.get('next_leg_jid')
         and dead and dead[0]['jid']=='9003' and dead[0]['state']=='CANCELLED'
         and b['intent_token'] != dead[0]['token'] else 1)" "${TX}/dead.json"
check "  ... not SUBMITTED; corpse on record; intent token ROTATED" $?
rm -f "${TX}/child_dead.flag"
expect_cmd "re-running after CHILD_DEAD submits a fresh successor" 0 "SUBMITTED" -- \
  tx_submit "${TX}/dead.json" "${TX}/squeue_none.sh" "${TX}/sacct_ctl.sh" \
    "SUBMIT_LOG=${TX}/submits_dead.log"
[ "$(grep -c . "${TX}/submits_dead.log")" = "1" ]; check "  ... exactly once" $?
# (2) identity completeness is UNCONDITIONAL, both at submit time and in preflight.
AG_MAN="${TMP}/ag_initial_manifest.txt"
ag_build() {  # <manifest-out> <registry-out> [max_steps] [leg_start] [leg_target] [init_target]
  printf '%s\n' "# exp_15 arm launch manifest" \
    "job 1 host h mode INITIAL launch_uuid u" \
    "arm YAWAUG rung 8x8 micro 8 ngpu 8 max_steps ${3:-2500} ckpt_every 2500" \
    "chain 1 leg_steps 2500 leg_start ${4:-0} leg_target ${5:-2500} cap 40000" > "$1"
  $PY -c "
import hashlib, json, sys
man, out, tgt = sys.argv[1], sys.argv[2], int(sys.argv[3])
json.dump({'arms': {'YAWAUG': {'manifest_path': man,
           'manifest_sha256': hashlib.sha256(open(man,'rb').read()).hexdigest(),
           'mode': 'INITIAL', 'chain': True, 'chain_cap': 40000,
           'chain_leg_steps': 2500, 'chain_initial_target': tgt}}, 'restarts': {}},
          open(out, 'w'))" "$1" "$2" "${6:-2500}"
}
ag_build "$AG_MAN" "${TMP}/ag_reg_ok.json"
expect_cmd "the strengthened submit-time validator still admits a valid INITIAL" 0 "ag_initial_manifest" -- \
  initial_manifest "${TMP}/ag_reg_ok.json"
$PY -c "
import json,sys
d=json.load(open(sys.argv[1])); del d['arms']['YAWAUG']['manifest_sha256']
json.dump(d, open(sys.argv[2],'w'))" "${TMP}/ag_reg_ok.json" "${TMP}/ag_reg_nosha.json"
expect_cmd "an INITIAL with NO registered manifest_sha256 is refused (absence != skip)" 1 "carries no manifest_sha256" -- \
  initial_manifest "${TMP}/ag_reg_nosha.json"
ag_build "${TMP}/ag_man_ms.txt" "${TMP}/ag_reg_ms.json" 5000
expect_cmd "manifest max_steps != leg size is refused at submit time" 1 "manifest max_steps" -- \
  initial_manifest "${TMP}/ag_reg_ms.json"
ag_build "${TMP}/ag_man_ls.txt" "${TMP}/ag_reg_ls.json" 2500 100
expect_cmd "manifest leg_start != 0 is refused at submit time" 1 "manifest leg_start" -- \
  initial_manifest "${TMP}/ag_reg_ls.json"
ag_build "${TMP}/ag_man_lt.txt" "${TMP}/ag_reg_lt.json" 2500 0 5000
expect_cmd "manifest leg_target != leg size is refused at submit time" 1 "manifest leg_target" -- \
  initial_manifest "${TMP}/ag_reg_lt.json"
ag_build "${TMP}/ag_man_it.txt" "${TMP}/ag_reg_it.json" 2500 0 2500 5000
expect_cmd "registry chain_initial_target != leg size is refused at submit time" 1 "registry chain_initial_target" -- \
  initial_manifest "${TMP}/ag_reg_it.json"
# ...and the worker preflight applies the same unconditional rules.
pre2() {  # <registry> <manifest>
  $PY "$CP" --ckpt "${TMP}/boundary.ckpt" --expected-step 2500 --target 5000 \
    --cap 40000 --config "$ARM_CONFIG" --arm YAWAUG --rung 8x8 --vae-sha256 VAESHA \
    --launch-manifest "$2" --registry "$1" --recorder "${EXPDIR}/yaw_aug_record_control.py"
}
$PY -c "
import json,sys
d=json.load(open(sys.argv[1])); del d['arms']['YAWAUG']['manifest_sha256']
json.dump(d, open(sys.argv[2],'w'), indent=2)" "${TMP}/chain_registry.json" "${TMP}/ag_pre_nosha.json"
expect_cmd "preflight refuses a registry entry with no manifest_sha256" 1 "carries no manifest_sha256" -- \
  pre2 "${TMP}/ag_pre_nosha.json" "${TMP}/initial_manifest.txt"
ag_prefix() {  # <new-manifest> <registry-out> <sed-expr>: mutate AD's manifest + re-hash
  sed "$3" "${TMP}/initial_manifest.txt" > "$1"
  $PY -c "
import hashlib, json, sys
d=json.load(open(sys.argv[1]))
d['arms']['YAWAUG']['manifest_path']=sys.argv[3]
d['arms']['YAWAUG']['manifest_sha256']=hashlib.sha256(open(sys.argv[3],'rb').read()).hexdigest()
json.dump(d, open(sys.argv[2],'w'), indent=2)" "${TMP}/chain_registry.json" "$2" "$1"
}
ag_prefix "${TMP}/ag_pre_ms.txt" "${TMP}/ag_pre_ms.json" 's/max_steps 2500/max_steps 5000/'
expect_cmd "preflight refuses manifest max_steps != leg size" 1 "manifest max_steps" -- \
  pre2 "${TMP}/ag_pre_ms.json" "${TMP}/ag_pre_ms.txt"
ag_prefix "${TMP}/ag_pre_lsx.txt" "${TMP}/ag_pre_lsx.json" 's/leg_start 0/leg_start 100/'
expect_cmd "preflight refuses manifest leg_start != 0" 1 "manifest leg_start" -- \
  pre2 "${TMP}/ag_pre_lsx.json" "${TMP}/ag_pre_lsx.txt"
# (3) terminal truth: a failed record write fails the JOB, and names its stranded child.
expect_cmd "the REAL terminal writer signals failure on an unwritable path" 1 "No such file or directory" -- \
  $PY "${TMP}/terminal.py" "${TMP}/no_such_dir/t.json" 0 0 0 0 OK 1 0 2500 40000 4242 "" "PASS" "AUDITED" 42 /x/m
sed -n '/--- BEGIN terminal-write-gate/,/--- END terminal-write-gate/p' "$LAUNCHER" > "${TMP}/twgate.sh"
grep -q 'final_rc=14' "${TMP}/twgate.sh"; check "the terminal-write gate is extracted (class 14)" $?
tw() {  # <write-rc> <successor> <final-rc-in> -> prints final_rc=<out>
  ( TERMINAL_WRITE_RC="$1" CHAIN_SUCCESSOR_JID="$2" final_rc="$3"
    . "${TMP}/twgate.sh"; echo "final_rc=${final_rc}" )
}
expect_cmd "write failure turns a green job into class 14" 0 "final_rc=14" -- tw 1 "" 0
tw 1 7777 0 > "${TMP}/tw.out"
grep -q "CHAIN STALLED: successor 7777 is queued with afterok" "${TMP}/tw.out"
check "  ... naming the already-queued successor and its unsatisfiable dependency" $?
grep -q "scancel 7777, fix the record path, then re-run the recorded" "${TMP}/tw.out"
check "  ... and the exact scancel + transact-submit re-run recovery" $?
expect_cmd "an existing failure class is PRESERVED (truth, not overwrite)" 0 "final_rc=9" -- tw 1 "" 9
expect_cmd "a clean write leaves a green job green" 0 "final_rc=0" -- tw 0 "" 0
grep -q '#  14  the TERMINAL status record' "$LAUNCHER"
check "class 14 is documented in the exit taxonomy" $?
expect_cmd "chain_advance routes a CHILD_DEAD transaction to class 12" 0 "final_rc=12" -- \
  drive 0 "TRAINLOG=${TMP}/fast.log" SUBMIT_RC=5
grep -q "DEAD" "${TMP}/advance.out"; check "  ... and says the successor was found dead" $?
# (4) manifest_fields() is complete against the REAL writer.
if $PY -c "import torch" 2>/dev/null; then
  sed -n '/--- BEGIN manifest-writer/,/--- END manifest-writer/p' "$LAUNCHER" > "${TMP}/manwriter.sh"
  [ -s "${TMP}/manwriter.sh" ]; check "the manifest writer is extracted from the launcher" $?
  ( export PATH="$(dirname "$PY"):$PATH"
    SLURM_JOB_ID=1; LAUNCH_UUID=u; MODE=INITIAL; ARM=YAWAUG; RUNG=8x8; MB=8; NGPU=8
    MAXSTEPS=2500; CHECKPOINT_EVERY=2500; HEAD_SHA=h; PINNED_P0_MANIFEST_SHA256=p
    MODEL_CONFIG_ABS=/x/c.json; CONFIG_SHA=cs; VAE_SHA=vs; ENV_SHA=es; CUDA_VER=12.6
    DRIVER=555; UUID_CSV=g; TIME_LIMIT=1:30:00; MIN_FREE_MB=14000; RESUME_CKPT=
    EXPECTED_STEP=0; CKPT_SHA=; SAVEDIR=/x; SLURM_OUT_AT_LAUNCH=/x/o; UNTRACK_STATE=n
    CHAIN=1; LEG_STEPS=2500; PINNED_CHAIN_CAP=40000; CODE_SNAPSHOT=/x/s; TRAINLOG=/x/t
    SAVEDIR_LOG=/x/tl; WANDB_ENTITY_SEEN=e; NAME=proj; EXPNAME=run; WANDB_RUN_ID=r
    PARENT_WANDB_RUN_ID=; SNAPSHOT_SHAS=(); ARGV=(--a 1)
    . "${TMP}/manwriter.sh" ) > "${TMP}/ag_real_manifest.txt" 2>/dev/null
  $PY -c "
import sys; sys.path.insert(0, '$EXPDIR')
from yaw_aug_chain_preflight import manifest_fields
f = manifest_fields(open('${TMP}/ag_real_manifest.txt').read())
need = ['job','host','mode','launch_uuid','arm','rung','micro','ngpu','max_steps',
        'ckpt_every','commit','p0_manifest_sha256','model_config','config_sha256',
        'vae_sha256','env_pip_freeze_sha256','torch_version','cuda','driver',
        'gpu_uuids','time_limit','min_free_mb','resume_ckpt','expected_step',
        'resume_ckpt_sha256','save_dir','slurm_transcript','untrack','chain',
        'leg_steps','leg_start','leg_target','cap','train_log','wandb_entity',
        'wandb_project','wandb_name','wandb_run_id']
missing = [k for k in need if k not in f]
assert not missing, f'unretrievable keys: {missing}'
assert f['cuda']=='12.6' and f['driver']=='555' and f['min_free_mb']=='14000'
assert f['expected_step']=='0' and f['untrack']=='n' and f['wandb_project']=='proj'
assert f['wandb_name']=='run' and f['cap']=='40000' and f['leg_target']=='2500'
print('every launcher-written key is retrievable, later pairs included')"
  check "manifest_fields() retrieves EVERY key the real writer emits (later pairs included)" $?
else
  skip_env "manifest_fields() retrieves EVERY key the real writer emits (later pairs included)" \
    "no torch for the writer's torch_version line; covered by the flac-env run"
fi

echo "--- Q. the suite touched nothing tracked, and submitted nothing ---"
TRACKED_AFTER="$(git status --porcelain --untracked-files=no -- "$EXPDIR" "$EXP11DIR" src data/AR | sort)"
UNTRACKED_AFTER="$(git status --porcelain --untracked-files=all -- "$EXPDIR" | sort)"
if [ "$TRACKED_BEFORE" = "$TRACKED_AFTER" ] && [ "$UNTRACKED_BEFORE" = "$UNTRACKED_AFTER" ]; then
  echo "PASS  tracked tree unchanged by the suite (snapshot before == after)"; PASS=$((PASS+1))
else
  echo "FAIL  the suite changed tracked state:"; diff <(echo "$TRACKED_BEFORE") <(echo "$TRACKED_AFTER") | sed 's/^/        | /'
  FAIL=$((FAIL+1))
fi
grep -qE '^[[:space:]]*(sbatch|scancel) ' "$0" && { echo "FAIL  this suite invokes sbatch/scancel"; FAIL=$((FAIL+1)); } \
  || { echo "PASS  this suite never invokes sbatch or scancel"; PASS=$((PASS+1)); }

echo
echo "=== guard tests: ${PASS} passed, ${FAIL} failed ($(grep -c "^SKIP" "$LEDGER" 2>/dev/null || echo 0) skipped, STRICT=${STRICT}) ==="
echo "ledger: ${LEDGER}"
[ "$FAIL" -eq 0 ] || exit 1
echo "log: ${LOG}"
