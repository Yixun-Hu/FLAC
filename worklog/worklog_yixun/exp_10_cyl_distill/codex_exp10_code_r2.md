## Findings

1. **Blocking — the hashed training manifest is not bound to the dataset actually loaded.** `real_run_gates()` hashes canonical `data/AR/train.json`, but `--dataset-config` remains unrestricted and is later loaded verbatim at [distill_cyl.py:241](/home/yixunhu/codespace/exp-10-cyl-distill/worklog/worklog_yixun/exp_10_cyl_distill/distill_cyl.py:241). A simulated valid real-run gate accepted an arbitrary dataset configuration, so the run can train on different data while the canonical manifest check passes.

2. **Blocking — probe/synthetic no-resume refusal is rank-asymmetric.** These modes bypass `real_run_gates()`, and the remaining log check is guarded by `rank == 0` at [distill_cyl.py:235](/home/yixunhu/codespace/exp-10-cyl-distill/worklog/worklog_yixun/exp_10_cyl_distill/distill_cyl.py:235). With an existing log, rank 0 exits while rank 1 continues toward DDP work or the probe barrier. The successful-probe path itself correctly barriers all ranks.

3. **Blocking to the requested test criterion — the LR floor test is not exact.** [test_distill_cyl.py:56](/home/yixunhu/codespace/exp-10-cyl-distill/worklog/worklog_yixun/exp_10_cyl_distill/tests/test_distill_cyl.py:56) uses an absolute tolerance rather than equality. A `floor + 5e-19` mutant survived every LR assertion while producing `lr_at(9999) != floor`. The implementation itself currently returns exactly `1e-6`.

Verified otherwise:

- Accumulation and `no_sync()` placement are correct. CPU autograd checks matched the direct 32-sample mean with zero numerical delta for `(16,2,1)`, `(8,2,2)`, and `(4,2,4)`.
- Clip/step occur once; non-finite detection is inside the micro loop before backward; the later loss all-reduce is logging-only.
- Mode exclusivity, required suffixes, no-grad one-batch probe, and checkpoint suppression are present.
- Embedded train/config hashes match their files; package-proper resolves to `301731b…`; reviewed HEAD is `bece0b9f65d442869e9ee75be3b0708bc5e876f7`.
- Loss/2, sum-to-mean, prefix-count, and all ±1 gate-window mutants were killed. The impulses at indices 799 and 9799 kill the earlier-shift mutants.
- `lr_at(499) == 1e-4`, `lr_at(500) < 1e-4`, and `lr_at(9999) == 1e-6`.
- Twelve read-only-compatible tests passed; the sole `tmp_path` test was exercised equivalently in memory because the sandbox has no writable temporary directory.
- `extract_teacher.py` is blob-identical across the review diff; teacher construction remains strict.

NOT CLEARED