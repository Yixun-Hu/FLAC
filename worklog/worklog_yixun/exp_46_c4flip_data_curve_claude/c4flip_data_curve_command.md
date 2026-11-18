# exp_46 c4flip_data_curve — commands (A6000 box: EVALUATION AND PUBLICATION ONLY; every launch is appended here AT LAUNCH TIME; failed / superseded runs stay, marked)

Conventions: **tree 46** `~/codespace/exp-46-campaign` = a clean, detached checkout of the frozen launch commit `X46`
(`CAMPAIGN_ROOT`; never committed in, never moved while the campaign lasts: `cmd_run` requires HEAD == the E0
receipt's head); **publication worktree** `~/codespace/exp-46-publish` (branch `exp-46-c4flip-data-curve`), from which
every records commit (`Y46`, the results `R46`) is made and pushed on Yixun's go. exp_46 trains nothing here (no A6000
training path: the supervisor refuses SMOKE / SMOKE_DDP / FULL on this box; every training runs on neuronic, from
`c4flip_data_curve_neuronic_commands.md`). Card locks are exp_38's (`~/codespace/.exp38_card<i>.lock`, the only
intentional sharing). `E=worklog/worklog_yixun/exp_46_c4flip_data_curve_claude`. The order of the campaign (L2 → pin →
L1 → union verify → L3 → L4) is the neuronic sheet's, pending Yixun D2. Nothing below has been run yet.

```bash
# 0. tree 46 at X46 (once, after L4): the links, the environment receipt, E0 (L4b; before any SMOKE)
set -e                  # fail fast: a failed cd, check or tool stops every step after it
X46=<the frozen launch commit X46>
git -C ~/codespace/FLAC fetch origin exp-46-c4flip-data-curve
git -C ~/codespace/FLAC worktree add --detach ~/codespace/exp-46-campaign $X46
cd ~/codespace/exp-46-campaign || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flac
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1; unset PYTHONOPTIMIZE PYTHONPATH
E=worklog/worklog_yixun/exp_46_c4flip_data_curve_claude
ln -s /media/diskstation/yixunhu/FLAC/AcousticRooms AcousticRooms
ln -s /home/yixunhu/codespace/FLAC/weights/AGREE weights/AGREE
ln -s /home/yixunhu/codespace/FLAC/weights/FLAC/VAE.safetensors weights/FLAC/VAE.safetensors
ln -s /home/yixunhu/codespace/FLAC/weights/FLAC/VAE.ckpt weights/FLAC/VAE.ckpt
python -B $E/exp46_nas_publish.py provision     # Codex code r3 finding 7: the NEW exp_46 checkpoint and dump roots on
#                                                 the NAS, once (their parents must be the mounted cifs NAS: refused
#                                                 otherwise; created through the isolation guard; idempotent)
python -B $E/exp46_registry.py untracked --platform a6000
python -B $E/exp46_eval.py env-receipt --write
python -B $E/exp46_eval.py prepare-e0          # sha256 + bytes of the 4 E0 references; creates eval_<closure16>
python -B $E/exp46_eval.py run --gpu 1 --cells de_E0a_K8_s42 de_E0b_K8_s42 de_E0c_K8_s42 de_E0d_K8_s42 \
                                            de_E0a_K1_s42 de_E0b_K1_s42 de_E0c_K1_s42 de_E0d_K1_s42
python -B $E/exp46_eval.py e0              # -> the sealed $E/c4flip_data_curve_e0_receipt.json (exit 0 = PASS only)
python -B $E/exp46_eval.py e0-check        # (one command per line: set -e never stops inside an && list)

# 1. the records commit Y46 from the publication worktree (tree 46 stays at X46): the E0 receipt + sidecar and, per E0
#    cell, its metrics JSON, stream sidecar and eval receipt (exactly 24 files), byte for byte, at the same paths --
#    the registered Y46 rule (test_exp46_provenance.py): that subject, those 24 ADDED files outside the exp_46 folder
#    and nothing else, closure(Y46) == closure(X46)
#    The publication worktree is created HERE, once, as the launch branch at X46 (Codex code r2 finding 7) -- its HEAD,
#    branch, upstream and a clean tree verified before anything is copied into it; set -e (above) and `|| exit 1`
#    stop every later step on a failed cd or check, so nothing is ever staged or committed in tree 46
git -C ~/codespace/FLAC worktree add -b exp-46-c4flip-data-curve ~/codespace/exp-46-publish $X46
cd ~/codespace/exp-46-publish || exit 1
git branch --set-upstream-to=origin/exp-46-c4flip-data-curve
test "$(git rev-parse HEAD)" = "$X46"
test "$(git rev-parse --abbrev-ref HEAD)" = exp-46-c4flip-data-curve
test "$(git rev-parse origin/exp-46-c4flip-data-curve)" = "$X46"
test -z "$(git status --porcelain --untracked-files=no)"
cp ~/codespace/exp-46-campaign/$E/c4flip_data_curve_e0_receipt.json{,.sha256} $E/
for g in E0a E0b E0c E0d; do mkdir -p outputs_FLAC/exp46_c4flip_data_curve/eval/$g
  cp ~/codespace/exp-46-campaign/outputs_FLAC/exp46_c4flip_data_curve/eval/$g/*_metrics_1_1.0_de_${g}_K[18]_s42*.json \
     outputs_FLAC/exp46_c4flip_data_curve/eval/$g/; done          # metrics + stream + receipt: 6 per gate, 24 in all
git add -f $E/c4flip_data_curve_e0_receipt.json{,.sha256} outputs_FLAC/exp46_c4flip_data_curve/eval/E0{a,b,c,d}
git diff --cached --name-only            # exactly the E0 receipt + sidecar and the 24 E0-cell files, nothing else
git commit -m "exp_46 records: E0 PASS at $X46"     # the registered E0-publication subject (full 40-hex X46) -> Y46
python -B $E/exp46_registry.py closure      # == the E0 receipt's closure_sha256: closure(Y46) == closure(X46)
#    push on Yixun's go; then the neuronic sheet (SMOKEs, FULLs, NAS publication)

# 2. (L8) per neuronic-trained run, once published to the NAS (neuronic sheet step 4): 10 cells each, 40 in all
cd ~/codespace/exp-46-campaign || exit 1
RUN=exp46_C4Ff025_l40_2x32       # then exp46_C4Ff050_l40_2x32, exp46_C4Ff075_l40_2x32, exp46_C4Ff100_l40_2x32
python -B $E/exp46_eval.py prepare $RUN          # TRAINING_COMPLETE (loss record; f100: its delta verification) at the E0 closure; farm copies
#    two lanes, one per card (Codex code r3 finding 5): each lane's PID is kept and waited on EXPLICITLY (a bare `wait`
#    returns 0 whatever the lanes did, and set -e never sees a background job's failure); both lanes are always reaped
#    (a lane's cells are independent: the other lane finishes and stays valid), then a failure of either stops the
#    sheet before anything is collected; an interrupted sheet stops both lanes
python -B $E/exp46_eval.py run --gpu 0 --cells ${RUN}_K8_s42 ${RUN}_K8_s43 ${RUN}_K8_s44 ${RUN}_K8_s45 ${RUN}_K8_s46 &
K8=$!
python -B $E/exp46_eval.py run --gpu 1 --cells ${RUN}_K1_s42 ${RUN}_K1_s43 ${RUN}_K1_s44 ${RUN}_K1_s45 ${RUN}_K1_s46 &
K1=$!
trap 'kill $K8 $K1 2>/dev/null' INT TERM
rc8=0
wait $K8 || rc8=$?
rc1=0
wait $K1 || rc1=$?
trap - INT TERM
if [ $rc8 -ne 0 ] || [ $rc1 -ne 0 ]; then echo "EXP46 EVAL LANES FAILED: K8 rc $rc8, K1 rc $rc1" >&2; exit 1; fi
python -B $E/exp46_collect.py            # after all four runs: all 40 cells + the E0 receipt admitted OFFLINE, else exit 2
python -B $E/exp46_collect.py verify     # every file the results JSON names re-hashes

# 3. the exp_46 results commit R46 from the publication worktree (tree 46 stays at X46): the results and every farm
#    file the results JSON names, byte for byte at the same paths, verified there before the commit
git -C ~/codespace/exp-46-publish pull --ff-only
python -B $E/exp46_collect.py materialise --dest ~/codespace/exp-46-publish     # identical-or-absent copies, re-hashed
cd ~/codespace/exp-46-publish || exit 1
test "$(git rev-parse --abbrev-ref HEAD)" = exp-46-c4flip-data-curve
python -B $E/exp46_collect.py verify     # the publication worktree: the same sha256 check before the commit
git add -f $E/c4flip_data_curve_results.{json,md} outputs_FLAC/exp46_c4flip_data_curve/eval
git diff --cached --name-only        # the results and the farm evidence only
git commit -m "exp_46 results publication: 40 C4F cells"     # -> R46; pushed on Yixun's go; then neuronic step 5
```
