# exp_21 sweep manifest (registered pre-launch, 2026-08-23)

GPU hold lifted by Yixun ("my senior finish"). SINGLE-COMMIT RULE: every cell below runs from THIS commit; no commits until all 25 cells finish. Dataset config `src/configs/dataset_configs/RAF/eval/raf_mappingA.json` (1,152 items, generations prepare 0de97c5a1c12 / depth 21a8ec5fc9bd); model config `src/configs/model_configs/FLAC/RAF/FLAC_RAF_finetune.json`; all cells: `--cfg-scale 1.0 --steps 1 --batch-size 64 --rotate-deg 0 --cond-autocast default --record-stream --record-per-item --record-per-scene --expected-stream-count 1152 --eval-name exp21_mA_<ARM>_seed<SEED> --seed <42..46>`.

| Arm | ckpt | cond flags | GPU |
|---|---|---|---|
| P1 | ar_40k_endpoints/P1/epoch=8-step=40000.ckpt | `--cond-method vanilla` | 0 |
| YAW | ar_40k_endpoints/YAW/… | `--cond-method vanilla` | 0 |
| BV | ar_40k_endpoints/BV/… | `--cond-method vanilla` | 0 |
| finetuned | checkpoints/exp19_raf_finetune/FLAC_RAF/exp19_raf_finetune_1000/checkpoints/epoch=142-step=1000.ckpt | `--cond-method vanilla` (labelled TRANSFER row; Mapping-H-trained) | 0 |
| BF | ar_40k_endpoints/BF/… | `--cond-method fa_invariant` (default C₄ angles, fwd-cap 64) | 1 |

Vanilla-smoke rung = P1 seed 42 (verified before the rest launches); FA rung = BF seed 42 (first BF cell). Stats: `mappingA_stats` paired ingestion, primary + minus-flagged(26) rows, placement-clustered.

## Re-stamp (2026-08-23): sweep base moves to the per-item-callback fix
Smoke fail-closed pre-artifact (dataset_name=None; would otherwise have scored per-item rows on the AR 8,000-sample window — a wrong-number bug caught as a wrong-string). Fix `ab6b1bc`/ledger `3f2d88f`. SWEEP BASE = the commit AFTER this line; co-tenancy with the peer campaign per agreement; cell list/flags unchanged.
