# exp-09 Stage D — launch records (frozen 2026-07-25 20:55:40)

## Pins (records freeze)
- **Checkpoint:** `outputs_FLAC/exp09_cylNoSSL/FLAC_exp09_cylNoSSL/exp09_cylNoSSL/checkpoints/epoch=14-step=67500.ckpt` (C2 final, step 67,500; rc=0 at 2026-07-25 20:41:52)
  sha256 `8d8ac56f334c7f648700c823f73afab68ec4b02bd06870a24563685258f1080a`
- **EXPECT_PACKAGE_SHA** = `4ea1971ff70dc45f1f84361fc7d5a9ab9455153b` (cylindrical-dinov3 HEAD; package-proper
  src/cylindrical_dinov3 unchanged since 301731b — verified; NO commits to that repo during the eval window)
- **EXPECT_EXP09_SHA** = the worktree HEAD of THE COMMIT THAT ADDS THESE RECORDS (captured at launch;
  every driver invocation passes both pins; online-variant FULL pin gate executes per-invocation at the driver)
- **Model config:** `worklog/worklog_yixun/exp_09_cyl_no_ssl/FLAC_AR_exp09_online_eval.json` (variant=online, auto-detected)
- **Dataset configs:** K=8 `src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json` (max_context 8);
  K=1 `..._unseeneval_1.json` (max_context 1) — the exp_01/Table-1 protocol pair
- **Frozen MIN_FREE_MB:** 26355 (c1_frozen_min_free.txt; exact-match gate in the driver)
- **External log dir:** `/home/yixunhu/codespace/cylindrical-dinov3/worklog/worklog_yixun/exp_06_flac_no_ssl_claude/d_eval_logs`
- **References:** `d_records/references.json` — validated against gate_thresholds_to_verdicts (d1 contextual
  advisory + bands; d2 expects; the committed Stage-A A2b artifact joins to exactly {11.25,45,90,180,270})

## Registered run matrix (17 GPU evals; fresh unique eval-names ⇒ unique artifact stems)
- **D1 (metrics, rot0, no predictions):** K∈{1,8} × seeds {42,43,44,45,46} — eval names `exp09_D1_K<k>_s<seed>`.
  GPU0 = K=1 arm, GPU1 = K=8 arm (sequential per GPU via d1_gpu*.sh).
- **D2 (seed 42, EVAL_STORE_PREDS=1):** K=1 × rot {0,45,90,180,270} (`exp09_D2_K1`, rot-suffixed by eval_FLAC);
  K=8 × rot {0,90} (`exp09_D2_K8`). GPU0 = K=1, GPU1 = K=8 via d2_gpu*.sh.
  D2 launches ONLY after the records/amendment Codex review clears (D1 does not use the amendment).
- **CPU e2e cells (after D2):** compare_predictions.py (exp_02 tool, path-pinned) rot-α vs rot-0 per K —
  cells exactly {1:[45,90,180,270], 8:[90]} → `d_records/e2e/` JSONs.
- **Manifests (post-eval, path-backed only):** `d1_manifest.json` {K:{seeds:{id:path}}};
  `flatness_index.json` {K:{angle:metrics-json-path}} incl. rot-0 reference; `e2e_index.json` {K:{angle:compare-json-path}}.
- **Adapter/aggregate:** gate_thresholds_to_verdicts.py --references d_records/references.json --d1/--d2-* →
  fresh out-dir `d_records/verdicts_<ts>/` → aggregate_gate.py over the verdict JSONs → SSL-verdict writeup.

## Standing obligations (D-tool r4) — discharge
1. matched-mode comparators or contextual mode → **contextual** (P1 pending; decision 2026-07-25 registered in exp_06 worklog)
2. path-backed manifests → all three manifests are seed_id/angle → PATH maps (inline values rejected by the CLI)
3. fresh unique out-dirs → per-run unique eval names; verdicts under a fresh timestamped dir; adapter refuses non-empty
4. producer angle-grid pinning → references pin {11.25,45,90,180,270} (cond) and the rot0-free e2e/flatness matrix verbatim
5. online-variant full-gate execution at the driver → every invocation embeds assert_arm_configs_exp09 on the ACTUAL config with variant=online + both SHA pins

## Disclosed amendment (for the records review)
`d_eval_driver.sh` gains env-gated `EVAL_STORE_PREDS=1` → appends `--store_predictions` (the cleared driver could
not produce the prediction bundles the registered e2e cells consume — gap found at D records). Default OFF (D1 shape
byte-identical to the cleared behavior); +2 DRY_RUN tests (present-when-set / absent-by-default); 19/19 driver tests pass.

## D2 conditioning disposition (registered)
The Stage-A A2b artifact (official weights) is the registered --d2-cond input: the gauge is parameter-free ⇒
conditioning-level equivariance is architectural (exp_03), weight-independent; the TRAINED checkpoint's equivariance
is gated directly by d2_end_to_end (≤0.00931) + d2_flatness (H-A3 constants) on its own predictions.
Pre-launch probe: d2_conditioning PASS, max rel-err 3.987e-06.
