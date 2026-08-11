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
LOG="${EXPDIR}/yaw_aug_${TS}_guardtests_r3.log"
HEAD_SHA="$(git rev-parse HEAD)"
CONTROL_COMMIT="81ddac372076ea92751ae09cbaf371df70f396e5"

exec > >(tee -a "$LOG") 2>&1
echo "=== yaw_aug_train guard exercise — ${TS} — $(git rev-parse --short HEAD) ==="
for f in "$LAUNCHER" "$SUBMITTER" "$ALLOWLIST" "$ARM_CONFIG" "$CLASSIFY" "$PREFLIGHT"; do
  [ -f "$f" ] || { echo "missing ${f} - abort"; exit 3; }
done

TRACKED_BEFORE="$(git status --porcelain -- "$EXPDIR" "$EXP11DIR" src | sort)"
TMP="$(mktemp -d)"
OUT_ROOT="${TMP}/outputs"            # never a production prefix
mkdir -p "$OUT_ROOT"
trap 'rm -rf "$TMP"' EXIT
PASS=0; FAIL=0

case_run() {  # <name> <want-rc> <want-substring> -- <env...>   (runs the REAL launcher)
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

case_spool() {  # <name> <launcher> <want-rc> <want-substring> -- <env...>
  local name="$1" launcher="$2" want_rc="$3" want_txt="$4"; shift 4; [ "$1" = "--" ] && shift
  local out rc
  out="$(env "$@" bash "$launcher" 2>&1)"; rc=$?
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

check() {  # <name> <condition-rc> — for grep-style structural assertions
  if [ "$2" -eq 0 ]; then echo "PASS  $1"; PASS=$((PASS + 1))
  else echo "FAIL  $1"; FAIL=$((FAIL + 1)); fi
}

spool() {  # <tag> [<sed-expr>...] -> path to a spooled launcher copy
  local tag="$1"; shift
  local dst="${TMP}/spool_${tag}.sbatch"
  cp "$LAUNCHER" "$dst"
  for expr in "$@"; do sed -i "$expr" "$dst"; done
  echo "$dst"
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
CAP="$(spool cap -e 's/\[ "\$MAXSTEPS" = "40000" \]/[ "$MAXSTEPS" = "50000" ]/' \
                  -e 's/^PINNED_MAXSTEPS=40000/PINNED_MAXSTEPS=50000/')"
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
if [ -z "$(git status --porcelain --untracked-files=no -- train.py defaults.ini src "$EXPDIR" 2>/dev/null)" ]; then
  case_spool "a REAL launch dies on an unreviewed file" "$SURPRISE" 2 "unreviewed production-surface changes" \
    -- ARM=YAWAUG "EXPECT_SHA=${HEAD_SHA}" SLURM_JOB_ID=999999
else
  echo "SKIP  real-mode allowlist case (working tree dirty; the drift gate fires first)"
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
  echo "SKIP  control-manifest cross-check (manifest not present on this machine)"
fi

echo "--- M. the submitter ---"
expect_cmd "submitter rejects a bad arm" 2 "must be YAWAUG" -- env DRYRUN=1 bash "$SUBMITTER" C8
expect_cmd "submitter derives 8x8 resources" 0 "--gres=gpu:l40:8" -- env DRYRUN=1 bash "$SUBMITTER" YAWAUG
expect_cmd "  ... cpus from the rung" 0 "--cpus-per-task=64" -- env DRYRUN=1 bash "$SUBMITTER" YAWAUG
expect_cmd "  ... mem from the rung" 0 "--mem=108G" -- env DRYRUN=1 bash "$SUBMITTER" YAWAUG
expect_cmd "  ... the INITIAL time pin" 0 "time 24:00:00" -- env DRYRUN=1 bash "$SUBMITTER" YAWAUG
expect_cmd "  ... and submits NOTHING in DRYRUN" 0 "DRYRUN sbatch" -- env DRYRUN=1 bash "$SUBMITTER" YAWAUG
INTENT_BEFORE="$(ls "${EXPDIR}"/yaw_aug_submission_*.txt 2>/dev/null | wc -l)"
env DRYRUN=1 bash "$SUBMITTER" YAWAUG >/dev/null 2>&1
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

echo "--- Q. the suite touched nothing tracked, and submitted nothing ---"
TRACKED_AFTER="$(git status --porcelain -- "$EXPDIR" "$EXP11DIR" src | sort)"
if [ "$TRACKED_BEFORE" = "$TRACKED_AFTER" ]; then
  echo "PASS  tracked tree unchanged by the suite (snapshot before == after)"; PASS=$((PASS+1))
else
  echo "FAIL  the suite changed tracked state:"; diff <(echo "$TRACKED_BEFORE") <(echo "$TRACKED_AFTER") | sed 's/^/        | /'
  FAIL=$((FAIL+1))
fi
grep -qE '^[[:space:]]*(sbatch|scancel) ' "$0" && { echo "FAIL  this suite invokes sbatch/scancel"; FAIL=$((FAIL+1)); } \
  || { echo "PASS  this suite never invokes sbatch or scancel"; PASS=$((PASS+1)); }

echo
echo "=== guard tests: ${PASS} passed, ${FAIL} failed ==="
[ "$FAIL" -eq 0 ] || exit 1
echo "log: ${LOG}"
