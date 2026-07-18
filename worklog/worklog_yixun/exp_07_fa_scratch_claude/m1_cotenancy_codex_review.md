Both files need small corrections before ship.

### FILE 1 — REQUEST-CHANGES

- [Lines 44–47](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/m1_ddp_fit_probe.sh:44): unit arithmetic is inconsistent. M0 reported **38.53 GiB**, while `nvidia-smi` returns MiB: `(38.53 + 1.5 + 4) × 1024 = 45,086.72 MiB`, not 44,000 MiB.  
  One-line fix: `MIN_FREE_MB="${MIN_FREE_MB:-45087}"` and describe the components as GiB/MiB consistently.

Everything else passes: `pipefail` makes the query assignment return the `nvidia-smi` failure; empty/non-numeric output aborts; the disclosure is appropriate; the diff changes only the intended gate/disclosure block.

### FILE 2 — REQUEST-CHANGES

- [Lines 67, 74–77](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/m1_rankfit_proxy.sh:67): a failed sampler leaves `peak=0`, producing false `49,140 MiB` headroom, and the script prints the decision rule without actually classifying the result.  
  One-line fixes: add `[ "$peak" -gt 0 ] 2>/dev/null || { echo "VRAM sampler failed"; exit 4; }`; then branch explicitly on `head_mb >= 2000`.

- [Lines 17–18, 56, 82](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/m1_rankfit_proxy.sh:17): `flac`/flash-DiT is assumed, not enforced, yet the OOM verdict claims it.  
  One-line fix before the assertion: `[ "${CONDA_DEFAULT_ENV:-}" = flac ] || { echo "conda env flac required"; exit 2; }`.

- [Line 85](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/m1_rankfit_proxy.sh:85): `timeout -k` can return 137 after KILL, not only 124. It still aborts safely, but misdiagnoses the timeout.  
  One-line fix: `if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then ...`.

Inference is otherwise sound as a lower-memory proxy, not a strict mathematical bound. My estimated incremental DDP+SyncBN cost is roughly **0.35–0.8 GiB typical, ≤1 GiB plausible**: ~246 MiB gradient buckets, tens-to-low-hundreds MiB NCCL, negligible SyncBN statistics, plus transient/allocator slack. Thus the stated **1–1.5 GiB allowance is conservative**, and a 2,000 MiB cutoff is reasonable. “OOM → cannot fit” is operationally strong but should ideally read “strongly rules out” because allocator layout and rank-specific batches prevent a formal guarantee.

`49140 - peak` is correct MiB arithmetic when the sample is valid. GPU 0 is protected: the training process and workers inherit `CUDA_VISIBLE_DEVICES=1`; the preflight assertion is CPU-only. Six workers add only accepted CPU/I/O contention. `bash -n` passes both scripts.