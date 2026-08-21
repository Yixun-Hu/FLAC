# HAA finetune recipe — handoff for the cylindrical-dinov3-SSL arm

**Branch: `exp17-yawaug-scratch` (origin, up to date). Everything below lives in
`worklog/worklog_yixun/exp_19_haa_finetune_claude/` unless pathed otherwise.**
Written 2026-08-21 by the Fable 5 seat for the session adding the CYL-SSL arm.

## The recipe (identical for every arm; enforced, not remembered)

Released HAA recipe: 1,000 steps, batch 16 × accum 4 (eff 64), single GPU,
AdamW 5e-6 + InverseLR(1e6,0.5,0.99), VAE frozen (`weights/FLAC/VAE.safetensors`),
`--val-every 10 --checkpoint-every 10`, seed 42, bf16-mixed, weights-only init
via `--pretrained-ckpt-path`. Full registered table: `plan_haa_finetune.md` §0.

## The moving parts

| What | Where |
|---|---|
| Launcher (all gates + the exact train.py argv) | `haa_ft_launch.sh` — `ARM=<X> GPU=<0|1> MODE=FULL EXPECT_SHA=$(git rev-parse HEAD) bash haa_ft_launch.sh` |
| Eval driver (2 ckpts × 2 K × 5 seeds per arm, per-arm protocol) | `haa_ft_eval.sh` — `ARMS="<X>" EXPECT_SHA=… bash haa_ft_eval.sh` |
| EMA init extraction (wrapped PL ckpt → bare init) | `python -m src.tools.extract_ema_weights --ckpt-path <AR-40k.ckpt> --out outputs_FLAC/exp19_inits/HAA_init_<X>.ckpt`; append `sha256sum` line to `exp19_init_shas.txt` (the launcher refuses unmanifested inits) |
| Aggregation (paper + pooled conventions) | `exp19_aggregate.py` (extend NAMES/EXPECTED_CM for the new arm) |
| Existing arm configs (templates) | `FLAC_HAA_finetune_{BF,YAW,CYL}.json`; **CYL is your template** — stock + 8 byte-inserted deltas mirroring the arm's own AR config |

## To add the SSL arm (mirror the CYL precedent exactly)

1. Find the SSL arm's AR-40k checkpoint + its AR training config (likely the
   exp-12 lineage: `exp12B_ssl*` in `~/codespace/exp-12-arms/outputs_FLAC/` or
   NAS `checkpoints/exp12_cyl_dinov3_arms/`). **The HAA config's deltas must
   mirror THAT config's conditioner/training keys** (implementation/gauge/
   grad-ckpt/cond_method/orbit — CYL-noSSL was fa_invariant on the trivial
   orbit `[0.0]`; verify the SSL arm's own, do not assume).
2. Byte-insert those deltas into a copy of the stock
   `src/configs/model_configs/FLAC/HAA/FLAC_HAA_finetune.json`; add the
   contract to `src/tests/test_exp19_haa_arm_configs.py` (forward construction
   + stripped equality + orbit pinned BY VALUE — see the CYL tests).
3. Add the arm to `haa_ft_launch.sh` (arm list, config map, config sha pin,
   init map, contract branch, probe policy) and `haa_ft_eval.sh` (arm list,
   config map, protocol branch with the arm's OWN orbit — `record_matches`
   distinguishes fa arms by ORBIT, not method). New arms are OPT-IN; the
   registered 60-cell grid stays untouched.
4. Eval protocol = the arm's own AR protocol (announcement 05), bf16, cfg 1.0,
   steps 1, `--record-per-scene`, seeds 42–46, steps {410,1000}, K {1,8}.

## Traps that already bit us (do not rediscover)

- **Concurrent edits to the two driver scripts**: three sessions have extended
  them; sed patterns silently no-op when the anchor moved. Assert
  `count == 1` per replacement (our coder was bitten twice).
- `EXPECT_SHA` must equal HEAD at launch — commit everything FIRST, launch
  second, and don't commit between arming an auto-chain and its firing.
- fa-family training may exceed a bare A6000 (BF OOM'd by 36 MiB with the C4
  orbit); grad-ckpt in the arm config is the fix (allclose-grade equivalent).
- `eval_FLAC.py` writes metric JSONs to `dirname(--ckpt-path)`.
- NAS (`/media/diskstation`) is CIFS: rsync needs `--inplace`; and per Yixun's
  2026-08-20 mandate all training checkpoints archive to
  `/media/diskstation/yixunhu/FLAC/checkpoints/<experiment>/` (verify count +
  per-file bytes + sampled shas BEFORE deleting local).
- HAA data root is a symlink to the NAS; the launcher's inventory gate checks
  the split's first/last files exist — leave `HAA_md.py` untouched (B4).

## Published comparators (paper convention, ckpt-1000, 5 eval seeds)

See `results_haa_ft_five_arm_final.md` (P1/YNA/BNA/YAW/BF) and
`results_haa_ft_steps_curve.md`; CYL-noSSL rows land ~2026-08-21 evening under
`results_haa_ft_six_arm_final.md` (same aggregator, `--table` extended).
