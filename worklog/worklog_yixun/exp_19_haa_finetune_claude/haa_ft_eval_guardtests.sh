#!/usr/bin/env bash
# ============================================================================
# haa_ft_eval_guardtests.sh — exercise every gate in haa_ft_eval.sh.
#
# Same conventions as the launcher's guard suite (fix batch 2):
#  * nothing here can start an evaluation — every case runs under DRY_RUN=1, and
#    the production cases are poisoned (no EXPECT_SHA), so if the refusal under
#    test were deleted the revision gate still refuses;
#  * checkpoints and metric JSONs are FIXTURES under a private mktemp CKPT_ROOT;
#    no production namespace is read or written;
#  * the queue lock is held exclusively for the suite's duration, and the
#    children are told so — a second queue must never schedule the same cells.
#
# Usage:  bash worklog/worklog_yixun/exp_19_haa_finetune_claude/haa_ft_eval_guardtests.sh
#
# Written by the exp_19 coder seat (Claude Opus 5, max effort).
# ============================================================================
set -uo pipefail

cd "$(dirname "$0")/../../.." || exit 2
EXPDIR="worklog/worklog_yixun/exp_19_haa_finetune_claude"
EVALSH="${EXPDIR}/haa_ft_eval.sh"
LOCKFILE="${EXPDIR}/.haa_eval.lock"
K8_CFG="src/configs/dataset_configs/HAA/eval/haa_test.json"
K1_CFG="src/configs/dataset_configs/HAA/eval/haa_test_1.json"
STOCK_CFG="src/configs/model_configs/FLAC/HAA/FLAC_HAA_finetune.json"
BF_CFG="${EXPDIR}/FLAC_HAA_finetune_BF.json"
YAW_CFG="${EXPDIR}/FLAC_HAA_finetune_YAW.json"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
GUARDLOG="${EXPDIR}/haa_ft_eval_${TS}_guardtests.log"
PASS=0; FAIL=0

exec 9>"$LOCKFILE" || { echo "!! cannot open ${LOCKFILE}"; exit 2; }
flock -n 9 || { echo "!! ${LOCKFILE} is held: an eval queue may be running. Refusing."; exit 2; }

TMPROOT="$(mktemp -d)" || exit 2
DRYTMP="${TMPROOT}/dry"; mkdir -p "$DRYTMP"
CK="${TMPROOT}/ckpts"
cleanup() { rm -rf "$TMPROOT"; }
trap cleanup EXIT

# --- fixture: one checkpoint per (arm, endpoint), PL-named ------------------- #
mk_ckpts() {
  local root="$1"; shift
  for arm in P1 BF YAW; do
    local d="${root}/exp19_HAA_${arm}/FLAC_exp19_HAA_${arm}/exp19_HAA_${arm}/checkpoints"
    mkdir -p "$d" || exit 2
    printf 'x' > "${d}/epoch=409-step=410.ckpt"
    printf 'x' > "${d}/epoch=999-step=1000.ckpt"
  done
}
mk_ckpts "$CK"
CKDIR_P1="${CK}/exp19_HAA_P1/FLAC_exp19_HAA_P1/exp19_HAA_P1/checkpoints"
CKDIR_BF="${CK}/exp19_HAA_BF/FLAC_exp19_HAA_BF/exp19_HAA_BF/checkpoints"

# A record for one cell, with the fields the resume check parses.
write_record() {   # <dir> <stem> <name> <suffix> <method> <seed> <dscfg> [angles-json] [cap]
  python - "$@" <<'PY'
import json, sys
d, stem, name, suffix, method, seed, dscfg = sys.argv[1:8]
angles = json.loads(sys.argv[8]) if len(sys.argv) > 8 else None
cap = int(sys.argv[9]) if len(sys.argv) > 9 else None
rec = {"metrics": {"T60": 1.0, "C50": 2.0}, "ckpt_path": f"{d}/{stem}.ckpt",
       "rotate_deg": 0.0, "cond_method": method, "frame_avg_angles": angles,
       "cond_autocast": "bf16", "frame_avg_fwd_cap": cap, "seed": int(seed),
       "dataset_config": dscfg, "steps": 1, "cfg_scale": 1.0, "eval_name": name}
json.dump(rec, open(f"{d}/{stem}_metrics_1_1.0_{name}{suffix}.json", "w"), indent=2)
PY
}

# --- helpers ----------------------------------------------------------------- #
OKENV=(DRY_RUN=1 CKPT_ROOT="$CK" TRAIN_PGREP='[z]zz_no_such_train_process'
       GATE_LOCK_HELD_BY_CALLER=1 TMPDIR="$DRYTMP")
POISON=(DRY_RUN=0)          # production, no EXPECT_SHA: the revision gate is the backstop

case_run() {
  local name="$1" want_rc="$2" want_txt="$3"; shift 3
  local out rc
  out="$(env "$@" bash "$EVALSH" 2>&1)"; rc=$?
  if [ "$rc" = "$want_rc" ] && grep -qF -- "$want_txt" <<<"$out"; then
    echo "  PASS  ${name}  (rc=${rc})"; PASS=$((PASS+1))
  else
    echo "  FAIL  ${name}: want rc=${want_rc} and text '${want_txt}'; got rc=${rc}"
    echo "        last line: $(tail -1 <<<"$out")"; FAIL=$((FAIL+1))
  fi
}
case_lacks() {   # the queue must NOT contain some text
  local name="$1" want_rc="$2" bad_txt="$3"; shift 3
  local out rc
  out="$(env "$@" bash "$EVALSH" 2>&1)"; rc=$?
  if [ "$rc" = "$want_rc" ] && ! grep -qF -- "$bad_txt" <<<"$out"; then
    echo "  PASS  ${name}  (rc=${rc})"; PASS=$((PASS+1))
  else
    echo "  FAIL  ${name}: want rc=${want_rc} and NO '${bad_txt}'; got rc=${rc}"; FAIL=$((FAIL+1))
  fi
}
expect() {
  local name="$1" ok="$2"
  if [ "$ok" = "1" ]; then echo "  PASS  ${name}"; PASS=$((PASS+1))
  else echo "  FAIL  ${name}"; FAIL=$((FAIL+1)); fi
}
eq() { [ "$1" = "$2" ] && echo 1; }
queue_of() {   # <ARMS> -> the CELL: lines
  local out; out="$(env "${OKENV[@]}" ARMS="$1" bash "$EVALSH" 2>&1)"
  grep '^CELL: ' <<<"$out"
}
cell_line() { grep -F -- "--eval-name $1 " <<<"$2" | head -1; }
argval() { awk -v f="$2" '{for(i=1;i<=NF;i++) if($i==f){print $(i+1); exit}}' <<<"$1"; }

exec > >(tee -a "$GUARDLOG") 2>&1
echo "=== exp_19 haa_ft_eval.sh guardtests — ${TS} — HEAD $(git rev-parse --short HEAD) ==="
echo "fixtures: ${CK} (private; no production namespace is read or written)"

echo "--- A. inputs and the production overrides ---"
case_run "A1 a bad ARMS token is refused"   2 "ARMS must be a subset"  "${OKENV[@]}" ARMS="P1 BFX"
case_run "A2 a bad DRY_RUN is refused"      2 "DRY_RUN must be exactly 0 or 1"  "${OKENV[@]}" DRY_RUN=maybe
case_run "A3 EVAL_CMD refused in production"    2 "EVAL_CMD is a DRY_RUN-only test override"    "${POISON[@]}" EVAL_CMD=true
case_run "A4 CKPT_ROOT refused in production"   2 "CKPT_ROOT is a DRY_RUN-only test override"   "${POISON[@]}" CKPT_ROOT="$CK"
case_run "A5 TRAIN_PGREP refused in production" 2 "TRAIN_PGREP is a DRY_RUN-only test override" "${POISON[@]}" TRAIN_PGREP='[z]zz'
case_run "A6 caller-held lock refused in production" 2 "GATE_LOCK_HELD_BY_CALLER is a DRY_RUN-only test override" "${POISON[@]}" GATE_LOCK_HELD_BY_CALLER=1
case_run "A7 THE POISON ITSELF: production without EXPECT_SHA is refused" 2 \
         "EXPECT_SHA is REQUIRED for a production eval queue" "${POISON[@]}"
case_run "A8 a wrong EXPECT_SHA is refused"  2 "is not the revision you reviewed" \
         DRY_RUN=0 EXPECT_SHA=0000000000000000000000000000000000000000

echo "--- B. the queue lock ---"
case_run "B1 a second queue is refused while the lock is held" 2 "another exp_19 eval queue holds" \
         DRY_RUN=1 CKPT_ROOT="$CK" TRAIN_PGREP='[z]zz_no_such_train_process' TMPDIR="$DRYTMP"
case_run "B2 the caller-held path proceeds" 0 "DRY_RUN: all gates passed" "${OKENV[@]}"

echo "--- C. a training run of OURS blocks the queue ---"
# A real process this suite owns, matched by a pattern that cannot match pgrep
# itself. Its cwd IS this worktree, which is what makes it 'ours'.
bash -c 'exec -a guardtest_fake_train_marker sleep 12' & FAKEPID=$!
sleep 1
case_run "C1 our own train.py blocks the queue" 2 "train.py from THIS worktree is still running" \
         "${OKENV[@]}" TRAIN_PGREP='[g]uardtest_fake_train_marker'
kill "$FAKEPID" 2>/dev/null; wait "$FAKEPID" 2>/dev/null
case_run "C2 and the queue proceeds once it is gone" 0 "no train.py from this worktree is running" "${OKENV[@]}"
# Driven with the REAL pattern, so whatever is on the box right now is what gets
# classified. The assertion holds either way and is not vacuous: the queue must
# PROCEED (rc 0 — a sibling checkout's run is never ours to block on, CLAUDE.md)
# and must say exactly one of the two co-tenancy lines, never both and never
# neither. (The first spelling of this case was `&& echo 1 || echo 1`, which
# passed unconditionally — a test that cannot fail.)
C3_OUT="$(env "${OKENV[@]}" TRAIN_PGREP='[t]rain\.py' bash "$EVALSH" 2>&1)"; C3_RC=$?
C3_LINES=$(grep -c -E '^co-tenancy (DISCLOSED: foreign train\.py|: no foreign train\.py)' <<<"$C3_OUT")
expect "C3 a FOREIGN train.py is classified and disclosed, never fatal (CLAUDE.md: sibling checkouts)" \
  "$([ "$C3_RC" = 0 ] && [ "$C3_LINES" = 1 ] && echo 1)"
expect "C4 the foreign branch is a disclosure, never a refusal" \
  "$(grep -qF 'co-tenancy DISCLOSED: foreign train.py' "$EVALSH" \
     && ! grep -q 'FOREIGN.*exit 2' "$EVALSH" && echo 1)"

echo "--- D. the checkpoint glob must match EXACTLY one ---"
CK0="${TMPROOT}/ckpts_zero"; mk_ckpts "$CK0"; rm -f "${CK0}/exp19_HAA_BF"/*/*/checkpoints/*step=1000.ckpt
case_run "D1 zero matches is refused"  2 "CHECKPOINT GLOB FAILED for BF step=1000: 0 match" \
         "${OKENV[@]}" CKPT_ROOT="$CK0"
CK2="${TMPROOT}/ckpts_two"; mk_ckpts "$CK2"
printf 'x' > "${CK2}/exp19_HAA_BF/FLAC_exp19_HAA_BF/exp19_HAA_BF/checkpoints/epoch=1000-step=1000.ckpt"
case_run "D2 two matches is refused"   2 "CHECKPOINT GLOB FAILED for BF step=1000: 2 match" \
         "${OKENV[@]}" CKPT_ROOT="$CK2"
case_run "D3 the resolved checkpoint is reported" 0 "ckpt P1 step=410: ${CKDIR_P1}/epoch=409-step=410.ckpt" \
         "${OKENV[@]}"

echo "--- E. the grid is the registered one ---"
FULLQ="$(queue_of "P1 BF YAW")"
expect "E1 exactly 60 cells"          "$(eq "$(grep -c '^CELL: ' <<<"$FULLQ")" 60)"
expect "E2 every cell name is unique" \
  "$(eq "$(grep -o -- '--eval-name [^ ]*' <<<"$FULLQ" | sort -u | wc -l)" 60)"
expect "E3 both endpoints appear 30x each" \
  "$(a=$(grep -c -- '_S410_' <<<"$FULLQ"); b=$(grep -c -- '_S1000_' <<<"$FULLQ"); \
     [ "$a" = 30 ] && [ "$b" = 30 ] && echo 1)"
expect "E4 five seeds x 12 cells each" \
  "$(ok=1; for s in 42 43 44 45 46; do [ "$(grep -c -- "_s${s} " <<<"$FULLQ")" = 12 ] || ok=0; done; [ $ok = 1 ] && echo 1)"
expect "E5 K is carried by the dataset config, 30 cells each" \
  "$(a=$(grep -cF -- "--dataset-config ${K8_CFG}" <<<"$FULLQ"); \
     b=$(grep -cF -- "--dataset-config ${K1_CFG}" <<<"$FULLQ"); \
     [ "$a" = 30 ] && [ "$b" = 30 ] && echo 1)"
expect "E6 K8 cells pair the K8 config with the K8 label (and never the K1 one)" \
  "$(grep -- '_K8_' <<<"$FULLQ" | grep -qvF -- "--dataset-config ${K8_CFG}" && echo 0 || echo 1)"
expect "E7 K1 cells pair the K1 config with the K1 label" \
  "$(grep -- '_K1_' <<<"$FULLQ" | grep -qvF -- "--dataset-config ${K1_CFG}" && echo 0 || echo 1)"

echo "--- F. per-arm protocol (announcement 05), asserted on the real argv ---"
P1_CELL="$(cell_line exp19_HAA_P1_S410_K8_s42 "$FULLQ")"
BF_CELL="$(cell_line exp19_HAA_BF_S1000_K1_s46 "$FULLQ")"
YAW_CELL="$(cell_line exp19_HAA_YAW_S410_K1_s44 "$FULLQ")"
expect "F1 P1 is vanilla"                "$(eq "$(argval "$P1_CELL" --cond-method)" vanilla)"
expect "F2 YAW is vanilla"               "$(eq "$(argval "$YAW_CELL" --cond-method)" vanilla)"
expect "F3 BF is fa_invariant"           "$(eq "$(argval "$BF_CELL" --cond-method)" fa_invariant)"
expect "F4 BF carries B-F's C4 orbit"    "$(eq "$(argval "$BF_CELL" --frame-avg-angles)" 0,90,180,270)"
expect "F5 BF declares the orbit chunk plan (announcement 06)" \
  "$(eq "$(argval "$BF_CELL" --frame-avg-max-fwd-samples)" 64)"
expect "F6 a vanilla cell carries NO orbit flags at all" \
  "$(grep -q -- '--frame-avg' <<<"$P1_CELL$YAW_CELL" && echo 0 || echo 1)"
expect "F7 every arm uses its OWN model config" \
  "$(a="$(argval "$P1_CELL" --model-config)"; b="$(argval "$BF_CELL" --model-config)"; \
     c="$(argval "$YAW_CELL" --model-config)"; \
     [ "$a" = "$STOCK_CFG" ] && [ "$b" = "$BF_CFG" ] && [ "$c" = "$YAW_CFG" ] && echo 1)"
expect "F8 the shared protocol is on every cell" \
  "$(n=$(grep -c -- '--cond-autocast bf16 --record-per-scene' <<<"$FULLQ"); [ "$n" = 60 ] && echo 1)"
expect "F9 cfg-scale and steps are pinned on every cell" \
  "$(n=$(grep -c -- '--cfg-scale 1.0 --steps 1 ' <<<"$FULLQ"); [ "$n" = 60 ] && echo 1)"
expect "F10 the seed flag matches the cell's name" \
  "$(eq "$(argval "$BF_CELL" --seed)" 46)"
expect "F11 each cell reads its own endpoint checkpoint" \
  "$(eq "$(argval "$BF_CELL" --ckpt-path)" "${CKDIR_BF}/epoch=999-step=1000.ckpt")"

echo "--- G. resume is decided by PARSING the record, not by the filename ---"
write_record "$CKDIR_P1" "epoch=409-step=410" "exp19_HAA_P1_S410_K8_s42" "" vanilla 42 "$K8_CFG"
case_run "G1 a matching record is skipped" 0 "SKIP: exp19_HAA_P1_S410_K8_s42" "${OKENV[@]}" ARMS=P1
case_lacks "G2 ...and that cell is not queued" 0 "--eval-name exp19_HAA_P1_S410_K8_s42 " "${OKENV[@]}" ARMS=P1
# Same filename, wrong protocol: this is the exp_09 failure mode (a row produced
# under the other conditioning) and must be re-run, not trusted.
write_record "$CKDIR_P1" "epoch=409-step=410" "exp19_HAA_P1_S410_K8_s43" "" fa_invariant 43 "$K8_CFG" '[0.0,90.0,180.0,270.0]' 64
case_run "G3 a record with the WRONG cond_method is re-run" 0 "RERUN: exp19_HAA_P1_S410_K8_s43" "${OKENV[@]}" ARMS=P1
write_record "$CKDIR_P1" "epoch=409-step=410" "exp19_HAA_P1_S410_K8_s44" "" vanilla 99 "$K8_CFG"
case_run "G4 a record with the WRONG seed is re-run" 0 "RERUN: exp19_HAA_P1_S410_K8_s44" "${OKENV[@]}" ARMS=P1
write_record "$CKDIR_P1" "epoch=409-step=410" "exp19_HAA_P1_S410_K8_s45" "" vanilla 45 "$K1_CFG"
case_run "G5 a record from the WRONG K split is re-run" 0 "RERUN: exp19_HAA_P1_S410_K8_s45" "${OKENV[@]}" ARMS=P1
: > "${CKDIR_P1}/epoch=409-step=410_metrics_1_1.0_exp19_HAA_P1_S410_K8_s46.json"
case_lacks "G6 an EMPTY record is not mistaken for a result" 0 "SKIP: exp19_HAA_P1_S410_K8_s46" "${OKENV[@]}" ARMS=P1
# BF's record carries the orbit fields, and its filename carries the _a4 suffix
# eval_FLAC derives — the resume path must predict the same name.
write_record "$CKDIR_BF" "epoch=999-step=1000" "exp19_HAA_BF_S1000_K1_s42" "_fa_invariant_a4" \
             fa_invariant 42 "$K1_CFG" '[0.0,90.0,180.0,270.0]' 64
case_run "G7 a matching BF record (with the _a4 suffix) is skipped" 0 \
         "SKIP: exp19_HAA_BF_S1000_K1_s42" "${OKENV[@]}" ARMS=BF
write_record "$CKDIR_BF" "epoch=999-step=1000" "exp19_HAA_BF_S1000_K1_s43" "_fa_invariant_a4" \
             fa_invariant 43 "$K1_CFG" '[0.0,90.0,180.0,270.0]' 32
case_run "G8 a BF record with the WRONG orbit cap is re-run (announcement 06)" 0 \
         "RERUN: exp19_HAA_BF_S1000_K1_s43" "${OKENV[@]}" ARMS=BF
# Of P1's five fixture records exactly ONE matches this protocol (s42); the other
# four are the wrong method, the wrong seed, the wrong split, or empty. The count
# must therefore be 1 skipped / 19 to run — if the resume check trusted filenames
# it would read 5 and 15, which is precisely the failure this section exists for.
expect "G9 exactly the MATCHING record is counted as done (1 of 5 present)" \
  "$(grep -qF "queue: 20 cells planned for arms 'P1' (1 already recorded, 19 to run)" \
      <<<"$(env "${OKENV[@]}" ARMS=P1 bash "$EVALSH" 2>&1)" && echo 1)"

echo "--- H. accept path and reporting ---"
case_run "H1 the DRY accept path reaches the boundary" 0 "DRY_RUN: all gates passed" "${OKENV[@]}"
case_run "H2 source pins are reported"  0 "source pins OK (7 files"  "${OKENV[@]}"
case_run "H3 a single-arm queue is 20 cells" 0 "queue: 20 cells planned for arms 'YAW'" "${OKENV[@]}" ARMS=YAW
expect "H4 the registered grid size is not env-overridable" \
  "$(! grep -qE 'EXPECTED_CELLS:-|SEEDS:-|STEPS_GRID:-' "$EVALSH" && echo 1)"
expect "H5 the completeness report recounts from the ARTIFACTS, not the loop" \
  "$(grep -qF 'cells complete (registered grid' "$EVALSH" \
     && grep -qF 'MISSING+=("$NAME")' "$EVALSH" && echo 1)"
expect "H6 an incomplete grid exits 1" \
  "$(grep -A3 'INCOMPLETE — missing or protocol-mismatched cells' "$EVALSH" | grep -qF 'exit 1' && echo 1)"

echo
echo "=== ${PASS} passed, ${FAIL} failed ==="
echo "log: ${GUARDLOG}"
[ "$FAIL" -eq 0 ]
