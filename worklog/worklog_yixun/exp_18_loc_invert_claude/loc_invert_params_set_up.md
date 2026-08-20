# loc_invert_params_set_up — exp_18 (started at R-1 launch, 2026-08-19_21:19:20 EDT; appended per run)

## Code state
- Branch `localization-exp`, launch HEAD `30f26d1` (rounds r1–r5b closed; 504 tests green; CLI parity vs eval_FLAC = 0.0 bitwise).
- Env: conda `flac`, torch 2.7.0+cu126, PL 2.1.0, transformers 4.57.0, **flash_attn ABSENT (fallback attention — recorded in every provenance)**, setuptools pinned <81, pytest 9.1.1. Box: mae-cab-lab-server, 2×A6000.

## Registered protocol constants (plan Rev 3 + 3.1; approved 2026-08-19)
- Headline ckpt: released `weights/FLAC/FLAC_EMA.ckpt` sha256 f3d47e9edd8dfc10… (weights_source resolves "online" — pre-flattened EMA export, pinned by test).
- Scorer: `weights/AGREE/AGREE_AR.pt` sha256 b664d5c09f74685f… (train-split-only), **deterministic VAE-mean readout**; `AGREE_fullAR.pt` (3a13243d6c6a1108…) = labelled leaky diagnostic only.
- Generation: rectified flow, steps=1, cfg_scale=1.0, --cond-method vanilla --rotate-deg 0 --cond-autocast default, matmul precision medium; noise bank keyed (seed, query_id, k), shared across candidates.
- K=8 samples/candidate (flac runs); agg=LME; τ selected on the seen split from {0.02,0.05,0.1,0.2,0.5} by pooled MEAN e_loc, smallest-τ tie-break (provisional τ=0.1 for smoke/oracle rows only — K=1/oracle aggregation is τ-invariant).
- **Pinned loader parallelism (O8, binds the context draws): --batch-size 4 --num-workers 4, shuffle=False, for EVERY exp_18 run.** Seeds: 42 (primary), 43, 44. Paired oracle-vs-FLAC comparisons additionally require row-level context-fingerprint equality between same-seed runs (verified at analysis).
- Unseen split pins: file sha256 9a9d817abc3e19f4…, 6,337 identities / 17 rooms, room-node-map sha256 38c07598fc070cff…; candidate authority = frozen metadata manifest (hash in provenance); wav floor ≥ 10,240 samples (score-inert suffix rationale).
- R0 smoke identities: BY RULE the first 4 identities of the canonical seen-split enumeration (shuffle=False); actual ids echoed in R0's log + summary and appended here at R0 launch.
- Known data facts: all 17 unseen rooms M=10; LRH_idx_30 S10 = metadata-only (expected readback WARNING; oracle eligibility −1 there); no second measurement channel ⇒ identity-oracle = sanity-only.

## Run R-1a (readback gate) — launched 2026-08-19_21:19:20
- Command in loc_invert_command.md; mode readback, unseen config, CPU. Non-registered diagnostic gate.

## Run R-1b (measured-RIR oracle + baselines) — launched 2026-08-19_21:19:20
- --score-source gt_rir (K forced 1), seed 42, device cuda:1, pinned loader (4/4), unseen config, AGREE_AR scorer, no FLAC ckpt (refused by design).
