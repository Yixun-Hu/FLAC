# exp_12 mem_probe — results

## Verdict: **OOM — the paper configuration does NOT fit on one L40 (46 GB).**

**Run:** Slurm job **3637984** (partition `all`, `gres=gpu:l40:1`, NVIDIA L40 `GPU-ab93458e-b01b-d555-f082-d122225ccbc7`, 46,068 MiB), commit-bound to `4c095ae517aa5f9e8e5dd847d40199ce4d98c9f0`, elapsed 2:48, deliberate exit code 3 (OOM classification path). All fail-closed gates passed before training: config pin (`f3eafef4…7ec9`, canonical == probe copy), semantic gate (vanilla conditioning; `use_ema: true`; ViT `gradient_checkpointing` absent/False in `source_vit` and `context_poses_vit`), single-node/single-task/single-GPU shape, wandb identity `yh4742@princeton.edu`.

**Configuration measured (paper path):** canonical `FLAC_AR.json` **verbatim — config diff vs canonical: NONE**; micro-batch **64**, accum **1**, `bf16-mixed`, 1 GPU single-process (no DDP), no SyncBN, no ViT gradient checkpointing, seed 42, workers 6, `--max-steps 5`.

**Outcome:** OOM on the FIRST training batch (Epoch 0, step 0/4550) — no optimizer step completed.

**Peak GPU memory (validated, UUID-bound 0.5 s poll, 228 samples): 45,437 MiB / 46,068 MiB (98.6% of the card)** at failure.

**Exact error (verbatim from `mem_probe_2026-08-05_16-29-28_jid3637984_S5_train.log`):**

```
OutOfMemoryError: CUDA out of memory. Tried to allocate 98.00 MiB. GPU 0 has a total
capacity of 44.39 GiB of which 23.31 MiB is free. Including non-PyTorch memory, this
process has 44.36 GiB memory in use. Of the allocated memory 36.83 GiB is allocated
by PyTorch, and 7.03 GiB is reserved by PyTorch but unallocated. If reserved but
unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
to avoid fragmentation.
```

raised inside `trainer.fit` (train.py:230 → PL 2.1.0 fit path), i.e. during the first forward/backward at micro-batch 64.

**Mandate compliance:** probe stopped and reported; **no** batch reduction, **no** gradient checkpointing, **no** accumulation, **no** extra GPUs. P1 job 3637217 untouched (verified running before and after: separate node allocation, separate job).

**Artifacts (this folder):** `slurm_exp12-mem-probe_3637984.out` (gates + classification), `mem_probe_2026-08-05_16-29-28_jid3637984_S5_train.log` (full train log incl. traceback), `mem_probe_2026-08-05_16-29-28_jid3637984_vram.csv` (raw poll samples), `mem_probe_command.md` (exact submission), `mem_probe_params_set_up.md` (full configuration).
