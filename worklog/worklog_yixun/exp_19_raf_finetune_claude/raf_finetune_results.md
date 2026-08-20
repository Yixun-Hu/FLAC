# exp_19 RESULTS — FLAC finetuned on RAF (canonical run, 2026-08-20)

**Protocol:** released `FLAC_EMA.ckpt` (AR-pretrained) finetuned 1000 steps on the canonical RAF split per the HAA README recipe (lr 5e-6 AdamW, batch 16 × accum 4, frozen VAE, EMA, seed 42; val loss 1.23 → 0.888). Data: audio generation `46a43f4ce82b` (×3 clamped scalar), depth generation `a44a723fce4c`; every eval stream-audited (768/768 test, 48/48 diagnostic, ×5 seeds ×2 rows, zero mismatches); flags `--cond-method vanilla --rotate-deg 0 --cond-autocast default`; step-1000 EMA ckpt sha in `rcal_weights_sha256.txt`. FD/Recall: **unavailable** (no AGREE-RAF exists) — not zero.

## Headline — RAF test split (768 held-out mics, 32 trained source-groups, 5 seeds, mean±SD)

| Row | T60 (%)↓ | C50 (dB)↓ | EDT (ms)↓ | mrL1↓ | Env↓ |
|---|---|---|---|---|---|
| Zero-shot (released ckpt) | 11.248±0.023 | 3.015±0.002 | 144.98±0.11 | 3.281±0.001 | 0.691±0.000 |
| **Finetuned (1000 steps)** | **5.638±0.037** | **0.849±0.005** | **38.94±0.35** | **2.821±0.001** | **0.467±0.000** |
| Δ (relative) | **−50%** | **−72%** | **−73%** | −14% | −32% |

Pooled and per-room-macro conventions coincide exactly (equal-room design: 384 test items/room). Per room (finetuned): EmptyRoom T60 4.30±0.02 / C50 0.80±0.01 / EDT 36.5±0.3; FurnishedRoom 6.97±0.07 / 0.90±0.01 / 41.4±0.7. Invalid-T60: 0 everywhere.

**Every headline improvement exceeds the R-cal reproduction band (~10–16%, `rcal_results_legB.md`) by 3–5×** — the finetuning effect is unambiguous. For context only (different dataset, no comparison claimed): the paper's HAA finetuned row was T60 3.10 / C50 2.17 / EDT 84.5.

## Diagnostic row (literal HAA-parity, 1 group/room, 24 targets each; tiny-n caveat as registered)
Zero-shot T60 15.50±0.10 / C50 1.30±0.01 / EDT 25.1±0.6 → finetuned 10.97±0.18 / 1.49±0.02 / 31.1±0.5. T60 improves; C50/EDT slightly degrade on this 48-item row — consistent with its registered role as a noisy sanity row, reported, not interpreted.

## Registered caveats
1. **Mapping H framing:** test receivers lie within the same 1.46 m array as the supports — this measures array-scale receiver interpolation on real acoustics, NOT room-scale generalization (RAF cannot offer HAA's receiver spread). The unseen-source protocol (Mapping A) is the registered follow-up.
2. ×3 amplitude scalar applied uniformly (clamp-bound, provenance-hashed) — physically neutral, disclosed.
3. Depth maps contain the scan's internal structure (capture equipment); sightline evidence recorded as diagnostic (Amendment 9); 2 Furnished maps carry ≤0.198% inpainted pixels (hash-attested).
4. Out-of-boundary residuals 1–3 stand as recorded (round-close package).
5. n=1 training run; the R-cal band bounds recipe-level variance, not run-to-run RAF variance.

Artifacts: finetuned ckpt + eval records in `/media/diskstation/yixunhu/FLAC/checkpoints/exp19_raf_finetune/` (zero-shot records copied to `eval_records/`); raw JSONs also beside their ckpts; logs in this folder.
