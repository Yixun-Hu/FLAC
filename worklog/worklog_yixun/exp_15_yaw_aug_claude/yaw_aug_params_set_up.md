# exp_15 yaw_aug — parameters & setup (written at launch, 2026-08-12)

## Arm: YAWAUG (the only trained arm of this experiment)

- **Model/config:** `worklog/worklog_yixun/exp_15_yaw_aug_claude/FLAC_AR_YAWAUG.json` — byte-copy of exp_11's `FLAC_AR_VANCKPT.json` (sha `733ca52b…`, VANL's registry-pinned config) + exactly `training.yaw_aug = {"enabled": true, "img_w": 512, "seed": 42}` (byte-level-tested). Vanilla conditioning (no `cond_method`), EMA on, ViT grad-ckpt on both conditioners, ViT img_w == 512 == yaw_aug.img_w (gated).
- **Augmentation:** per-sample random yaw, fresh draw per visit, uniform over 512 columns; counter-based seeds — offsets are a pure function of (42, global_step, global_rank, index) via a keyed 32-bit bijection (collision-free over the 40k×8 domain, tested); physically-consistent rotation via `rotate_scene_metadata` (depth roll + 4 pose fields; GT RIR/context audio untouched); `training_step` only; schema guards fail-closed.
- **Recipe (= exp_11 rung 8×8, = VANL control):** torchrun 1×8 L40 · micro 8 × accum 1 = eff 64 = SyncBN batch · seed 42 · bf16-mixed · workers 6 · no val · grad-clip 0 · ckpt every 2500 · max-steps 40000 · AdamW 5e-5/(0.9,0.999)/wd 1e-3 · InverseLR(1e6,0.5,0.99) · dataset `acousticroom_train.json` (291,210 items) · VAE sha `8d82159e…`.
- **Slurm:** `--gres=gpu:l40:8 --cpus-per-task=64 --mem=108G --time=24:00:00`, partition all, single node.
- **Pin:** commit `5368108` (pushed; EXPECT_SHA); content-scoped closure (train.py, defaults.ini, src minus src/tests, data/AR, kit+helpers+configs+admission record+allowlist).

## Control: VANL (reused, exp_11)

`outputs_FLAC/exp11_VANL/…/epoch=8-step=40000.ckpt`, admission record `yaw_aug_control_admission.json` (ckpt sha `1095f493…`, step 40000, EMA mirror exact, embedded config == pinned config canonically). Historical recipe-matched control — code-state delta bounded by the pin-diff allowlist (production diff vs `81ddac3` = factory/diffusion/yaw helpers + tests only; train.py + defaults.ini byte-unchanged).

## Gates active at launch (all fail-closed)

Commit binding + content-scoped drift · allowlist vs control commit · semantic config gate (yaw_aug literal block) · accum==1 · rung/topology pins · VRAM floor · VAE sha · env versions + VANL-manifest cross-check · banner watcher (`yaw_aug ENABLED img_w=512 seed=42`, exact line, before step evidence) · run-owned tool snapshot (hashes manifested) · completion audit at round-2 rigor (ckpt sha/step/embedded-config/EMA-mirror; registry `final_ckpt_sha256`) · storage-light smoke + parsed/bound acceptance record required for production submission. Exit classes: 8 banner, 9 completion audit, 10 smoke, 11 measured-FAIL.

## Smoke (precedes production, same rung)

SMOKE=1: 30 steps, checkpoint interval beyond max (none written, asserted), nvidia-smi VRAM sampler, acceptance record `yaw_aug_smoke_acceptance.json` (rate floor 0.945 = 0.9×VANL 1.05 steps/s; VRAM ceiling = rung floor; all epilogue checks folded in; atomic publish last). Known measurement caveat: rate = PL's last-reported it/s at 30 steps — mild warmup-ramp risk; evidence preserved in the record either way.
