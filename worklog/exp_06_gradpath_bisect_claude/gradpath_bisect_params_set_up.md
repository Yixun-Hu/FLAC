# Params — exp_06_gradpath_bisect

Code `197c49a` (post lr-schedule round) · GPU 0 shared envelope · env rir2rir (torch 2.7.0+cu126).

| Stage | Params |
|---|---|
| S1 | evals of existing R1b/V1p ckpts at steps 200/400/600; K=8 seed 42 full split, exp_01 protocol |
| S2 arms | finetune_cond.py: cond_method vanilla, --freeze-bn, batch 4 × accum 32 (eff. 128), 625 opt steps, ckpt-every 200, seed 42, bf16-mixed, clip 0.0, use_ema off; lr per arm {5e-7, [5e-6=V1p anchor], 2e-5, 4.2e-5, 5e-5+--lr-schedule inverse-restart} |
| S2 screens | eval_FLAC K=8 seed 42 full split (SCREENING ONLY) |
| S3.1 | git diff vs upstream/master (github.com/AmandineBtto/FLAC) |
| S3.2 | s3_probes.py: 300 train-split RIRs (review-corrected), paired aug (p=1.0) vs raw via repo metric stack |
