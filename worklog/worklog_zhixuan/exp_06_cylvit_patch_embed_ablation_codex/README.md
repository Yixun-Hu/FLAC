# Experiment 06: CylViT patch-embedding ablation

This experiment changes only the CylViT patch-embedding mechanism while keeping
the C16 geometry encoder, FLAC architecture, optimizer, scheduler, data and
training budget matched.

## Compared variants

- `linear` (the requested MLP baseline): the existing non-overlapping
  `patchify -> LayerNorm(1536) -> Linear(1536, 512) -> LayerNorm(512)` path
  (791,040 patch-embedding parameters).
- `cnn`: each non-overlapping `[3, 16, 32]` patch is reshaped into the batch
  dimension and processed independently by one shared four-layer CNN with widths
  `3 -> 34 -> 44 -> 96 -> 512`, total per-patch stride `(16, 32)`, pointwise
  channel LayerNorm and GELU after the first three layers (791,050 parameters).

The CNN uses zero padding inside each patch and never mixes neighboring patches.
Shared weights and height-major patch ordering keep the output token grid exactly
equivariant to the 16 panorama-aligned yaw rotations of C16. The two patch
embeddings differ by only 10 parameters.

## Initialization and training contract

- "From scratch" applies to the geometry ViT branch, not to all of FLAC.
- All shape-compatible non-geometry weights are loaded from
  `weights/FLAC/FLAC_EMA.ckpt`, including the DiT, other conditioners and frozen
  VAE/pretransform.
- Both geometry branches are initialized once from seed 42. Every common tensor
  (transformer body, output projection and token-pooling projection) is copied
  exactly from the linear variant to the CNN variant; only the patch stem is
  allowed to differ.
- `source_vit` and `context_poses_vit` continue to share one ViT instance.
- VAE parameters remain frozen. The ViT, DiT and all other trainable
  conditioners are jointly optimized, matching the original FLAC training
  code.
- AdamW: learning rate `5e-5`, betas `(0.9, 0.999)`, weight decay `1e-3`.
- InverseLR: `inv_gamma=1,000,000`, `power=0.5`, `warmup=0.99`.
- BF16 mixed precision; micro-batch 4 and gradient accumulation 16 on each
  independent GPU, giving effective batch size 64 per variant.
- CFG dropout `0.1`, log-SNR timestep sampling, DiT EMA enabled from step 0,
  and gradient clipping disabled.
- The original seen-room K=8 validation split is evaluated every 2,500
  optimizer steps (full split, batch size 4).

The first run stops at 30,000 optimizer steps and saves sparse milestones at
5k, 10k, 20k and 30k plus a rolling `last.ckpt`. It is a resumable comparison
gate; the upstream training driver budgets 1,000,000 steps, so 30k should not be
reported as a converged final result without checking the learning curves.

## Evaluation contract

At 30k, clean Table-1 evaluation uses the full unseen AcousticRooms split,
`steps=1`, `cfg_scale=1`, both K=1 and K=8, and generation seeds 42--46. Report
T60, C50, EDT, R@1/R@5/R@10 and FD_G (`FD` in the JSON output) as mean and
standard deviation across generation seeds.

The C16 yaw diagnostic uses the same checkpoint at all 16 angles
`n * 22.5 degrees`. The first paired sweep uses K=1 and generation seed 42 and
reports each metric, its delta from yaw 0, mean absolute delta, worst
degradation and standard deviation over angles.

## Entry points

- `prepare_matched_initializations.py`: builds and audits the paired starting
  checkpoints.
- `train_patch_ablation.py`: one-variant Lightning training driver.
- `run_train_30k_one.sh`: one GPU / one variant launcher.
- `run_train_30k_dual_gpu.sh`: linear on GPU 0 and CNN on GPU 1, in parallel.
- `run_eval_one.sh`, `run_eval_suite_dual_gpu.sh`, `summarize_eval.py`: 30k
  Table-1 and C16 evaluation workflow.
- `prepare_eval_subset.py`, `plot_c16_comparison.py`: deterministic 544-item
  diagnostic subset preparation and C16 comparison plots. The committed subset
  manifest makes the diagnostic sample selection auditable.

Checkpoints, metric JSONs and raw runtime logs are intentionally excluded from
this branch; they are generated artifacts rather than source code.

The paired initialization artifacts are stored under
`outputs_FLAC/exp06_cylvit_pe_matched_initializations/`. Training outputs use
`outputs_FLAC/exp06_cylvit_pe_linear_trainS42/` and
`outputs_FLAC/exp06_cylvit_pe_cnn_patchlocal_trainS42/`; the 30k checkpoints are
named `step=000030000.ckpt`. The older
`outputs_FLAC/exp06_cylvit_pe_cnn_trainS42/` directory is retained as a historical
artifact of the incorrect whole-panorama CNN stem. Those CNN checkpoints must not
be resumed or evaluated as the corrected patch-local variant.

From the FLAC directory, the foreground dual-GPU launcher is:

```bash
bash worklog/worklog_zhixuan/exp_06_cylvit_patch_embed_ablation_codex/run_train_30k_dual_gpu.sh
```

To validate the complete launcher/model/init path without creating a
dataloader, touching a GPU, writing a training checkpoint or starting
training, prefix the same command with `DRY_RUN=1`.
