# exp_12 mem_probe — parameters / configuration

**Status: PRE-LAUNCH DRAFT** (review N7 fix) — becomes the launch record when `mem_probe_command.md` gains the exact `sbatch` command, job ID, and full worker SHA at submission. **Acceptance criterion (unambiguous):** either (a) rc=0 AND the PL marker `` `Trainer.fit` stopped: `max_steps=5` reached. `` in the train log AND a validated UUID-bound peak-VRAM number, or (b) fail-honest OOM: exact `CUDA out of memory`/`OutOfMemoryError` excerpt + validated peak-at-failure. Anything else = distinct non-OOM failure, reported as such.

**Model config:** `FLAC_AR_exp12_memprobe.json` — byte-copy of canonical `src/configs/model_configs/FLAC/AR/FLAC_AR.json` (sha256 `f3eafef4456666e4705ddaf35540f6b9f1f746189814cec000bac794ba2a7ec9`, cmp-asserted at runtime). **Config diff vs canonical: NONE.** Notably: vanilla conditioning (no `cond_method`/`frame_avg_angles` keys), both `ViTCoordinates` conditioners WITHOUT `gradient_checkpointing` (key absent ⇒ dataclass default False, asserted fail-closed at launch).

**Training invocation (paper configuration on L40):**
- 1 GPU (`--num-gpus 1`, Slurm `--gres=gpu:l40:1`) — no DDP (strategy default `auto`, single process)
- micro-batch **64**, `--accum-batches 1` ⇒ effective batch 64 = BN batch 64, single-GPU (paper path; no SyncBN — flag not passed, Trainer kwargs byte-identical to release)
- precision `bf16-mixed` (defaults.ini, no flag)
- `--max-steps 5`, `--checkpoint-every 10000` (⇒ no checkpoints; storage-light)
- `--seed 42`, `--num-workers 6`, dataset `src/configs/dataset_configs/AR/train/acousticroom_train.json`, pretransform `weights/FLAC/VAE.safetensors`, `HF_HUB_OFFLINE=1`
- identity: `--name FLAC_exp12_memprobe --experiment-name exp12_memprobe --save-dir outputs_FLAC/exp12_memprobe`; logger wandb (identity-gated; disclosed fallback `none`)

**Slurm:** partition `all`, job-name `exp12-mem-probe`, cpus 10, mem 40G, time 00:40:00; stdout → `slurm_exp12-mem-probe_<jobid>.out` in this folder; teed run log `mem_probe_<TS>_S5.log`.

**Measurement:** 0.5 s `nvidia-smi` poller (allocated GPU only) → `mem_probe_<TS>_vram.csv`; peak = max over samples. On OOM: exact error excerpt printed from the log; no retry/fallback (Yixun mandate).

**OOM / abort policy (verbatim mandate):** on OOM stop and report exact error + peak memory. Forbidden: batch reduction, gradient checkpointing, gradient accumulation, multi-GPU.
