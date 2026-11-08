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
# rung 6 — GPU-0 single-rank smokes (Yixun-allowed, ≤ 15 min; GPU 1 untouched)
MODE=SMOKE RUN=de_yaw_f025 SMOKE_GPU=0 SMOKE_STEPS=20 bash $E/exp38_train_local.sh   # 21-07-13 FAIL (smoke loss floor 50, fixed) -> 21-15-35 PASS
MODE=SMOKE RUN=de_bf_f025  SMOKE_GPU=0 SMOKE_STEPS=20 bash $E/exp38_train_local.sh   # 21-10-54 PASS
```

## Campaign on THIS box (primary design D2-a) — to run after Yixun's decisions and the code reviews
```bash
# 0. the immutable campaign tree at the pushed pin (once)
git -C ~/codespace/FLAC fetch origin exp-38-data-efficiency
git -C ~/codespace/FLAC worktree add --detach ~/codespace/exp-38-campaign <PIN>
cd ~/codespace/exp-38-campaign
ln -s /media/diskstation/yixunhu/FLAC/AcousticRooms AcousticRooms
ln -s /home/yixunhu/codespace/FLAC/weights/AGREE weights/AGREE
ln -s /home/yixunhu/codespace/FLAC/weights/FLAC/VAE.safetensors weights/FLAC/VAE.safetensors
ln -s /home/yixunhu/codespace/FLAC/weights/FLAC/VAE.ckpt weights/FLAC/VAE.ckpt
source ~/miniconda3/etc/profile.d/conda.sh && conda activate flac
E=worklog/worklog_yixun/exp_38_data_efficiency_claude; PIN=$(git rev-parse HEAD)

# 1. environment receipt + E0 (both cards idle — sequence through ~/codespace/claude_gpu_yield/ first; ≈ 2–3 h)
PYTHONNOUSERSITE=1 python $E/exp38_eval.py env-receipt --write
python $E/exp38_eval.py prepare-e0
PYTHONNOUSERSITE=1 python $E/exp38_eval.py run --gpu 0 --cells de_E0a_K8_s42 de_E0b_K8_s42 de_E0c_K8_s42 de_E0d_K8_s42 &
PYTHONNOUSERSITE=1 python $E/exp38_eval.py run --gpu 1 --cells de_E0a_K1_s42 de_E0b_K1_s42 de_E0c_K1_s42 de_E0d_K1_s42; wait
python $E/exp38_eval.py e0            # -> $E/data_efficiency_e0_receipt.json (+ .sha256); FAIL = stop, ask Yixun (D6)

# 2. per run, in the order de_bf_f025, de_yaw_f025, de_bf_f050, de_yaw_f050, de_bf_f075, de_yaw_f075:
RUN=de_bf_f025
MODE=SMOKE_DDP RUN=$RUN EXPECT_SHA=$PIN bash $E/exp38_train_local.sh            # ≈ 5–15 min, receipt
MODE=FULL      RUN=$RUN EXPECT_SHA=$PIN setsid nohup bash $E/exp38_train_local.sh \
  > $E/data_efficiency_$(date +%Y-%m-%d_%H-%M-%S)_${RUN}_FULL_driver.out 2>&1 < /dev/null &
#    B-F ≈ 61–77 h exclusive, YAW ≈ 22 h; the driver log + at_launch_<run>_<ts>.txt record everything
python $E/exp38_eval.py prepare $RUN
python $E/exp38_eval.py run --gpu 0 --cells ${RUN}_K8_s42 ${RUN}_K8_s43 ${RUN}_K8_s44 ${RUN}_K8_s45 ${RUN}_K8_s46 &
python $E/exp38_eval.py run --gpu 1 --cells ${RUN}_K1_s42 ${RUN}_K1_s43 ${RUN}_K1_s44 ${RUN}_K1_s45 ${RUN}_K1_s46; wait
python $E/exp38_collect.py            # -> $E/data_efficiency_results.{json,md}
```
Receipts written in the campaign tree (`data_efficiency_env_receipt_a6000.json`, `data_efficiency_e0_receipt.json`, `data_efficiency_smoke_*.json`, logs) are untracked there; they are copied into the dev worktree and committed from there (records-only commits leave the campaign closure digest unchanged).

## Neuronic (sensitivity platform only — D2-b / D2-c): see `data_efficiency_neuronic_commands.md`
