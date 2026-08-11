# exp_16 della_vanilla_repro — params & environment setup

## Environment (rung 0, executed 2026-08-11 ~16:30 EDT, login node della9)

- Conda env: `/scratch/gpfs/BLANCHETTE/yh4742/conda_envs/flac` (python 3.10; torch 2.7.0, pytorch_lightning 2.1.0, transformers 4.57.0 — full snapshot in `pip_freeze_2026-08-11.txt`).
- **Repairs (review B1):** `pip install "setuptools<81" pytest` → setuptools **80.10.2** (restores `pkg_resources` for `k_diffusion→clip`), pytest **9.1.1** (+ deps exceptiongroup 1.3.1, iniconfig 2.3.0, pluggy 1.6.0, pygments 2.20.0, tomli 2.4.1). Before this, `import train` died with `ModuleNotFoundError: pkg_resources`.
- **No flash-attn** in the env (math-attention fallback; release used flash-attn 2.7.4.post1 — disclosed irreproducibility factor; exp_01's anchor numbers also ran without it).
- **HF cache populated (review B2/B3):** `HF_HOME=/scratch/gpfs/BLANCHETTE/yh4742/hf_cache hf download facebook/dinov3-vits16-pretrain-lvd1689m` → snapshot `114c1379950215c8b35dfcd4e90a5c251dde0d32` (6 files). Previously the cache entry was a 94 KB `refs/`-only stub.
- **Offline-load proof (rung 0.4), all with `HF_HUB_OFFLINE=1` on CPU:**
  - hub id via cache (conditioner call site, `src/models/conditioners.py:458`): `DINOv3ViTModel`, 21.60M params ✓
  - local dir `models/dinov3-vits16-pretrain-lvd1689m` (resolver target): `DINOv3ViTModel`, 21.60M ✓
  - AGREE-style `AutoModel.from_pretrained(..., device_map="auto")` (`AGREE/AGREE/hf_model.py:30`): ✓
  - **cache snapshot and `models/` dir state_dicts are bit-identical** (`torch.equal` over all keys) — the two load paths cannot diverge in weights.
- Import smoke: `python -c "import train, eval_FLAC"` from repo root → OK.
- Reviewer tooling: `codex-cli 0.147.0` installed by Yixun mid-session (`~/.local/bin/codex`) — **Reviewer switches back to OpenAI Codex `gpt-5.6-sol` at xhigh from the first code round onward**; the plan review remains attributed to the declared Opus 5 fallback.

## Storage layout (per Yixun's della mandate)

- Released weights moved to `/scratch/gpfs/BLANCHETTE/yh4742/FLAC/weights/` (FLAC.ckpt, FLAC_EMA.ckpt, FLAC_HAA.ckpt, VAE.ckpt, VAE.safetensors, AGREE_AR.pt, AGREE_fullAR.pt, AGREE_fullHAA.pt); file-level symlinks left in `weights/FLAC/`, `weights/AGREE/` so all repo-relative config/command paths still resolve. Repo `weights/` now 23 KB + JSONs.
- Dataset: repo `AcousticRooms` → `/scratch/gpfs/BLANCHETTE/yh4742/datasets/AcousticRooms` (symlink).
- ViT weights: repo `models` → `/scratch/gpfs/BLANCHETTE/yh4742/FLAC/models` (Yixun's symlink).
- Training checkpoints will go to `--save-dir /scratch/gpfs/BLANCHETTE/yh4742/FLAC/checkpoints/exp16_vanilla_repro`.
- W&B: `WANDB_MODE=offline`, `WANDB_DIR=/scratch/gpfs/BLANCHETTE/yh4742/FLAC/wandb`.

## Eval protocol constants (announcement 05 — stated, not defaulted)

`--cond-method vanilla · --cond-autocast default · --rotate-deg 0.0 (fixed mode) · --cfg-scale 1.0 · --steps 1 · batch 64 · num-workers 4 (eval defaults, matching exp_01) · --record-stream · --expected-stream-count 6337 (unseen) / 6217 (seen)`. No `--store_predictions` (README deviation, disclosed: predictions land beside the ckpt; not needed for metrics). No `--allow-partial-load`.

## Training run parameters

Filled at Phase-2 launch (command, effective seed incl. SLURM_PROCID term, node, GRES, QOS, `--time`, measured smoke rate). Inherited defaults pinned in plan §2: `gradient_clip_val 0.0`, `num_sanity_val_steps 0`, `log_every_n_steps 100`, `save_top_k -1`, `accum_batches 1`, `precision bf16-mixed`, seed 42.
