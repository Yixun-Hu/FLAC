# exp_16 — della_vanilla_repro — driving queries (Yixun)

## Query 1 — 2026-08-11 (session `vanilla-flac-repro`, della)

### Verbatim

> I need you to read @worklog/experiment_SOP.md as your rule in this session to conduct experiments. And The first experiment I want you to do is to reproduce the vanilla FLAC using their @FLAC_pdf.md reported training recipe: "Our DiT model consists of 12 transformer blocks with 8 heads and a hidden width of 256. We train it using a flow matching objective using a learning rate of 5 ×10−5, AdamW optimizer [53] and a batch size of 64 on a single H100 GPU. We use an Exponential Moving Average (EMA) of the model weights during training and BF16 precision.". Before everything started, I want to notice that in this della gpu, you should put only light code inside ~/codespace/FLAC, and other heavy checkpoints/logs/data/weights inside /scratch/gpfs/BLANCHETTE/yh4742, you can ls /scratch/gpfs/BLANCHETTE/yh4742/FLAC to see what is currently inside. I ln the `models` folder under ~/codespace/FLAC to `models` under /scratch/gpfs/BLANCHETTE/yh4742/FLAC/models to put your dinov3-vits16-pretrain-lvd1689m weights under it. And HF cache under /scratch/.../hf_cache = shared HF cache, and /scratch/.../conda_envs = environments (which I have installed the flac conda environment). For the output_FLAC folder and the weights folder, since those are previously git tracked, you can put json files under it, but please put those checkpoints under the /scratch/gpfs/BLANCHETTE/yh4742/FLAC/checkpoints. And I have finish the dinov3-vit16 pretrained weights download (into /scratch/gpfs/BLANCHETTE/yh4742/FLAC/models/dinov3-vits16-pretrain-lvd1689m/), I think maybe you need to modify some of the code to acoomodate this. And if you need to modify the code to acomodate to the della cluster usage rule (~ usage and /scratch usage), please checkout a new branch from check-equivariance-necessity and name it della-flac-chequity, and on the new branch please conduct the first experiment of training the vanilla FLAC on H100. Before training, please first eval @README.md (40-49) FLAC on the official checkpoint EMA to verify the evaluation makes sense.

### Summary

1. Follow `worklog/experiment_SOP.md` for all experiment work this session.
2. Experiment: **reproduce vanilla FLAC training** with the paper's reported recipe (DiT 12 blocks / 8 heads / width 256, flow matching, lr 5e-5, AdamW, batch 64, single H100, EMA, BF16) on della.
3. Della storage discipline: light code in `~/codespace/FLAC`; heavy artifacts (checkpoints/logs/data/weights) under `/scratch/gpfs/BLANCHETTE/yh4742`. Training checkpoints specifically to `/scratch/gpfs/BLANCHETTE/yh4742/FLAC/checkpoints`. JSONs may stay in the repo's `outputs_FLAC/` and `weights/`.
4. DINOv3 ViT-S/16 weights are pre-downloaded at `/scratch/gpfs/BLANCHETTE/yh4742/FLAC/models/dinov3-vits16-pretrain-lvd1689m/`, reachable via the repo's `models` symlink; code may need modification to load them locally.
5. All della-accommodation code changes go on a new branch `della-flac-chequity` cut from `check-equivariance-necessity`.
6. **Before training**: run the README lines 40–49 evaluation of the official `FLAC_EMA.ckpt` to verify the evaluation pipeline on this machine.

### Assumption / hypothesis

The published FLAC recipe, run on della's H100 with this repo's configs, reproduces a model statistically comparable to the released `FLAC_EMA.ckpt` (as measured on the full published eval splits). The della environment (Slurm, offline compute nodes, scratch storage) can be accommodated without touching the training math.

### Why the experiment needs to run

This fork's equivariance line (exp_02→exp_15) so far compares against the *released* checkpoint or short screens. A full-budget vanilla reproduction on della (a) validates the entire della training pipeline end-to-end before any method experiments run here, (b) establishes the training noise floor for full-budget comparisons (how far a faithful re-run lands from the release), and (c) provides a locally-trained vanilla anchor for future full-budget method arms.
