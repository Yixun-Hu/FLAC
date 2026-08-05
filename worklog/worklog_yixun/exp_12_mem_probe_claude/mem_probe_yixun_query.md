# Yixun's queries — exp_12_mem_probe

## Query 1 (2026-08-05, mid-turn commission)

### Verbatim

> I need you to run a new experiment: Create a new isolated memory probe on Neuronic using the configuration closest to the paper:
>
> - 1× L40
> - batch size 64
> - gradient accumulation 1
> - BF16
> - no DDP
> - no SyncBN
> - no gradient checkpointing in either ViT conditioner
> - canonical FLAC_AR configuration
>
> Run only 2–5 training steps and record peak GPU memory. Do not modify or stop the currently running P1 job 3637217.
>
> If the probe OOMs, stop and report the exact error and peak memory. Do not silently reduce the batch size, enable gradient checkpointing, add gradient accumulation, or use multiple GPUs, because those would change the paper configuration.
>
> Keep this experiment isolated with separate config, output, log, W&B, and worklog names. Report the exact config diff, sbatch file, job ID, result, and peak VRAM.

### Summary

A bounded single-GPU Slurm probe answering: does the paper training configuration (micro-batch 64 on ONE GPU, no SyncBN, no ViT gradient checkpointing, bf16-mixed) fit in an L40's 46 GB, and at what peak VRAM? Fail-honest on OOM — the probe's job is to measure, not to make the configuration fit.

### Assumption / hypothesis (context, recorded by Planner)

The exp_07 config-identity audit established the released recipe achieved BN-batch 64 via micro-64 on a single large-memory GPU (H100-class) without SyncBN; this fork's SyncBN-64 DDP recipe (32×2) exists because micro-64 was assumed not to fit on the available GPUs. This probe measures that assumption directly on the L40.

### Why this experiment needs to run

Peak-VRAM ground truth for the paper configuration on L40 hardware decides whether future arms could use the paper's exact single-GPU recipe (eliminating the SyncBN/DDP deviation entirely) or must keep the SyncBN-64 workaround. The answer (fit + headroom, or OOM + exact shortfall) is decision-grade either way.
