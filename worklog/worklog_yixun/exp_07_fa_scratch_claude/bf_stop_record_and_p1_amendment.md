# B-F stop record (futility, 2026-07-23) & P1 plan amendment

## B-F stop record

| Item | Value |
|---|---|
| Decision | Yixun 2026-07-23 ~16:45: **"Stop now → P1 immediately"** at the criterion-justified early futility review (pre-registered point was 50k; the criterion — EDT AND R@1 worse than the matched 8×8 anchor by >2× eval σ — was met by ~an order of magnitude with a flat slope over 4 screen points) |
| Stopped at | global step **40,249** (epoch 8, 3,849/4,550; loss 0.626 last-line) |
| Last saved ckpt | `outputs_FLAC/exp07_BF/FLAC_exp07_BF/exp07_BF/checkpoints/epoch=8-step=40000.ckpt` (249 unsaved steps ≈ 1 h discarded) |
| Screens on record | 10k/20k/30k/40k EMA+online + cfg0 conditioning-lift probe (worklog 07-20→07-23 entries) |
| Endpoint extrapolation (recorded pre-stop) | T60 ~9.5–10 / C50 ~2.0 / EDT ~78–80 / R@1 ~1 at 67.5k |
| wandb | run `qtz2o9xx` will show "crashed" (process kill) — that is this deliberate stop, not a failure |
| Kill method | TaskStop of the launch task after ckpt 40000 confirmed on disk; ranks verified gone; GPUs cleared to the third-party 6.5 GB tenants |
| Resume (if ever) | needs a `--ckpt-path` variant of `bf_scratch_launch.sh` (fresh-launch only today); all 2.5k-cadence ckpts intact |

**Standing scientific read at stop:** conditioning ACTIVE (cfg0 lift ≥ B-V absolute on error metrics) but the whole trajectory converges to a ~2× worse operating point at this budget; attribution recipe-vs-fa is OPEN → P1.

## P1 plan amendment (supersedes plan_bv_parity.md §1 recipe; tiers/statistics §§ carry over)

**Recipe = EXACTLY B-F's (single-delta contract):** `FLAC_AR_BVp1.json` = BV **semantic copy** (formatting differs; parsed-object identity) + the 2 `gradient_checkpointing` keys → **BVp1 vs BF differ ONLY by `cond_method` + `frame_avg_angles`** (diff-asserted at every launch). 32/GPU × 2 GPUs × accum 1 (eff 64), SyncBN (BN=64), ViT grad-ckpt ON (recipe identity; gradient-equivalent — not guaranteed bitwise CUDA trajectories), env `flac`, seed 42, 67.5k steps, ckpt every 2,500, wandb `FLAC_exp07_P1/exp07_P1`, launch `p1_ddp_launch.sh` (mirrors the SHIP'd `bf_scratch_launch.sh` + config-contract pre-flight).

**Estimands (priority-ordered, amended):**
1. **ATTRIBUTION (primary, new):** P1's 10k/20k/30k/40k screens vs B-F's at matched steps (single-delta → the fa_invariant effect) and vs the 8×8 B-V anchor at matched steps (the recipe effect). Early read expected by ~1.5 d (10k+20k).
   - If P1 ≈ 8×8 anchor → recipe innocent → B-F's ~2× gap is **fa-from-scratch itself** → Route-1 conclusion: fa needs fine-tune (exp_08 path), from-scratch inefficient.
   - If P1 ≈ B-F's curve → recipe-dominated → B-F's numbers reinterpreted; fa possibly fine; recipe (SyncBN-64/micro-32/DDP at this LR schedule) needs revisiting for BOTH arms.
   - Intermediate → **sequential decomposition** (review precision): P1-vs-8×8 = the bundled recipe/environment effect within vanilla (NOT SyncBN/micro individually — no factorial cells); P1-vs-BF = the total FA-arm effect (training + FA screen-time conditioning); interaction folds into the FA-at-new-recipe term.
2. **Micro/BN-parity (original P1 purpose):** P1 endpoint vs the 8×8 B-V endpoint/late-curve statistic (plan tiers: PARITY/STRONG/DIRECTIONAL/NULL, R@1 required) — now at SyncBN-64, *closer to the release's BN-64 than the originally planned 32×2 single-GPU arm*.
3. **Released-parity (secondary):** endpoint vs Table-1 under the tiered σ_c gate.

**Discipline:** hard aborts only; screens per 10k (EMA+online, flac env); futility logic mirrors B-F's (criterion + slope, reviewable early on the same terms).

**Post-launch review (`p1_kit_codex_review.md`): live run VALID.** Deferred fix before any REUSE of `p1_ddp_launch.sh`: replace the diff-count contract gate with exact parsed-object assertions (count-windows can pass a concretely wrong config at the edges). DO NOT edit the script while its bash wrapper runs (incremental-read hazard) — apply at P1 end. Reviewer note: launch pip-freeze hash delta vs B-F reduces to an unrelated editable cylindrical-DINO repo HEAD change; training-relevant deps matched.
