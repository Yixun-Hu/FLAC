# exp_38 data_efficiency — commands (every launch is appended here AT LAUNCH TIME; failed / superseded runs stay, marked)

Conventions: dev worktree `~/codespace/exp-38-data-efficiency` (branch `exp-38-data-efficiency`); **campaign tree** `~/codespace/exp-38-campaign` (a detached worktree at the pushed campaign pin `<PIN>`, never committed in while anything runs); conda env `flac`; `E=worklog/worklog_yixun/exp_38_data_efficiency_claude`. Nothing below has been launched for real yet (2026-11-07): only the rungs marked DONE.

## Done (Coder, 2026-11-07, validation ladder)
```bash
# rung 3 — subset audit v2 + verify (CPU, read-only AR tree)
python $E/exp38_subset_audit.py --out $E/data_efficiency_subset_audit.json        # PASS, 39 s
python $E/exp38_subset_audit.py --verify $E/data_efficiency_subset_audit.json     # PASS
# rung 4 — CPU init digest (identical for every arm: 378440d0…)
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= python $E/exp38_init_digest.py --run de_bf_f050 --json <scratch>/init.json
# rung 5 — every launch path dry-run: $E/data_efficiency_2026-11-07_21-36-46_dry_run_all_paths.log + …_dry_run_sbatch.log
# code-review r1 fix round (2026-11-07 22:18) — subset audit v3 (realpath dropped from the bound map, HIGH 8)
python $E/exp38_subset_audit.py --out $E/data_efficiency_subset_audit.json        # PASS (v2 kept as *_v2_superseded_r1.json)
# code-review r2 fix round (2026-11-07 23:51 -> 2026-11-08 00:05) — the AR content manifest (BLOCKING 3): 445,414 files,
# 17.87 GB, per-file sha256 grouped into 729 room groups; pinned sha256(text) 369bd4fc… in exp38_registry.py
python $E/exp38_content_manifest.py --build <scratch>/ar_content_manifest.txt.gz --threads 16   # 14 min over CIFS
python $E/exp38_content_manifest.py --verify $E/data_efficiency_ar_content_manifest.txt.gz        # every byte re-hashed
# rung 6 — GPU-0 single-rank smokes (Yixun-allowed, ≤ 15 min; GPU 1 untouched)
MODE=SMOKE RUN=de_yaw_f025 SMOKE_GPU=0 SMOKE_STEPS=20 bash $E/exp38_train_local.sh   # 21-07-13 FAIL (smoke loss floor 50, fixed) -> 21-15-35 PASS
MODE=SMOKE RUN=de_bf_f025  SMOKE_GPU=0 SMOKE_STEPS=20 bash $E/exp38_train_local.sh   # 21-10-54 PASS
```

## Campaign on THIS box (primary design D2-a) — to run after Yixun's decisions and the code reviews
Every training mode runs through `exp38_train_local.sh` → `exec exp38_supervise.py`, which holds both card
locks (passed down to train.py and to a detached lock holder that keeps them until NO process of the run's
group remains), launches train.py FROM an immutable closure snapshot under
`~/codespace/exp-38-snapshots/<run>_<mode>_<ts>/`, owns its process group (TERM/INT/HUP forwarded, 120 s
grace then KILL) and seals its receipt only after every end audit passes. Every evaluation cell runs FROM
the campaign's evaluation snapshot `~/codespace/exp-38-snapshots/eval_<closure16>/` (created once by
`prepare-e0`). The campaign identity is the **closure digest** (`exp38_registry.py closure`); the campaign
tree's HEAD stays frozen for the whole campaign (receipts and logs written there are untracked; they are
copied to the dev worktree and committed from there — with the farm's E0 / result cells force-added, which
the offline table validator re-admits).
```bash
# 0. the immutable campaign tree at the pushed pin (once)
git -C ~/codespace/FLAC fetch origin exp-38-data-efficiency
git -C ~/codespace/FLAC worktree add --detach ~/codespace/exp-38-campaign <PIN>
cd ~/codespace/exp-38-campaign
ln -s /media/diskstation/yixunhu/FLAC/AcousticRooms AcousticRooms           # the exact registered targets
ln -s /home/yixunhu/codespace/FLAC/weights/AGREE weights/AGREE
ln -s /home/yixunhu/codespace/FLAC/weights/FLAC/VAE.safetensors weights/FLAC/VAE.safetensors
ln -s /home/yixunhu/codespace/FLAC/weights/FLAC/VAE.ckpt weights/FLAC/VAE.ckpt
source ~/miniconda3/etc/profile.d/conda.sh && conda activate flac
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1; unset PYTHONOPTIMIZE PYTHONPATH
E=worklog/worklog_yixun/exp_38_data_efficiency_claude; PIN=$(git rev-parse HEAD)
python $E/exp38_registry.py untracked --platform a6000                     # UNTRACKED PASS (links by content)
python $E/exp38_content_manifest.py --verify $E/data_efficiency_ar_content_manifest.txt.gz   # ~15 min; FULL repeats it

# 1. environment receipt (sealed, once) + E0 (both cards idle — the main session sequences this through
#    ~/codespace/claude_gpu_yield/ first; ≈ 2–3 h)
python $E/exp38_eval.py env-receipt --write
python $E/exp38_eval.py prepare-e0          # sha256 + bytes of the 4 references; creates the evaluation snapshot
python $E/exp38_eval.py run --gpu 0 --cells de_E0a_K8_s42 de_E0b_K8_s42 de_E0c_K8_s42 de_E0d_K8_s42 &
python $E/exp38_eval.py run --gpu 1 --cells de_E0a_K1_s42 de_E0b_K1_s42 de_E0c_K1_s42 de_E0d_K1_s42; wait
python $E/exp38_eval.py e0            # -> the sealed $E/data_efficiency_e0_receipt.json; FAIL = stop, ask Yixun (D6)
python $E/exp38_eval.py e0-check      # every later consumer re-runs this (incl. re-admitting the 8 E0 cells)

# 2. per run, in the order de_bf_f025, de_yaw_f025, de_bf_f050, de_yaw_f050, de_bf_f075, de_yaw_f075:
RUN=de_bf_f025
MODE=DRY_RUN   RUN=$RUN bash $E/exp38_train_local.sh                                   # gates + exact commands
MODE=SMOKE_DDP RUN=$RUN EXPECT_SHA=$PIN bash $E/exp38_train_local.sh                   # ≈ 5–15 min -> sealed
#    $E/data_efficiency_smoke_${RUN}_a6000_<ts>.json (the NEWEST receipt of the run must be a full PASS; its log
#    and snapshot are re-opened by FULL, so keep both)
MODE=FULL      RUN=$RUN EXPECT_SHA=$PIN setsid nohup bash $E/exp38_train_local.sh \
  > $E/data_efficiency_$(date +%Y-%m-%d_%H-%M-%S)_${RUN}_FULL_driver.out 2>&1 < /dev/null &
#    ≈ 15 min of AR content verification first, then B-F ≈ 61–77 h exclusive, YAW ≈ 22 h. Records:
#    $E/at_launch_${RUN}_a6000_FULL_<ts>.json (authoritative; + the run-dir copy), the train log, <NAS run dir>/receipts/
#    (the E0 / SMOKE / environment receipts the run was admitted under + the train log) and the sealed
#    <NAS run dir>/TRAINING_COMPLETE.json on success. Stop it only with `kill -TERM <supervisor_pid>` (in the record):
#    the supervisor stops ITS group and keeps the .exp38_lock crash marker; a crashed run is archived and re-run
#    from step 0 (there is no resume).
python $E/exp38_eval.py prepare $RUN          # TRAINING_COMPLETE (every field + retained records) at the E0 closure,
#                                              re-hash + bytes, full-state audit; farm copies for the offline validator
python $E/exp38_eval.py run --gpu 0 --cells ${RUN}_K8_s42 ${RUN}_K8_s43 ${RUN}_K8_s44 ${RUN}_K8_s45 ${RUN}_K8_s46 &
python $E/exp38_eval.py run --gpu 1 --cells ${RUN}_K1_s42 ${RUN}_K1_s43 ${RUN}_K1_s44 ${RUN}_K1_s45 ${RUN}_K1_s46; wait
#    a failed / interrupted cell keeps its <metrics stem>.claim dir (archive it with the partial outputs to retry)
python $E/exp38_collect.py            # -> $E/data_efficiency_results.{md,json}; exit 1 while cells are pending
python $E/exp38_collect.py --partial  #    (publishes the admitted runs with PENDING rows, exit 0)
```

## Neuronic (sensitivity platform only — D2-b / D2-c): see `data_efficiency_neuronic_commands.md`
