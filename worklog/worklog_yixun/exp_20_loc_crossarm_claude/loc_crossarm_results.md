# loc_crossarm_results — exp_20 (FINAL, 2026-08-23 23:00 EDT)

## Campaign: 18/18 registered cells, all gates green
3 arms (P1 vanilla / BF FA / YAW @ 40k, admitted ckpts) × 2 regimes (K_ctx=8, K_ctx=1) × 3 seeds (42/43/44); freeze `a92ff5d`; identity gate 6,337/6,337 every cell; zero .partial residue; dumps + inline metrics per cell on the NAS; **pairing gate PROVEN for all 6 (regime, seed) cells across all three arms** (query streams, context digests, noise keys equal).

## Macro top-1 (3-seed mean ± SD)
| arm | K_ctx=8 | K_ctx=1 | ctx-member K8 | ctx-member K1 | macro mean e (K8/K1) |
|---|---|---|---|---|---|
| P1 vanilla | 0.4948 ± 0.0002 | 0.4936 ± 0.0011 | 0.386 | 0.049 | 1.118 / 1.128 m |
| **BF FA** | **0.5087 ± 0.0007** | **0.5049 ± 0.0015** | 0.366 | 0.047 | 1.076 / 1.086 m |
| YAW | 0.4997 ± 0.0015 | 0.4980 ± 0.0004 | 0.381 | 0.049 | 1.099 / 1.114 m |
(References: retrieval control 0.689 at K8 / 0.107 at K1; info-matched chance 0.490 / 0.111. Every arm reproduces exp_18's sparse-regime reversal.)

## REGISTERED CONFIRMATORY VERDICT (top-1, paired per-query 3-seed means, 17-room clustered, Holm over exactly 4)
| contrast | Δtop-1 | 95% CI | p_raw | p_adj | verdict |
|---|---|---|---|---|---|
| **BF vs P1 @ K8** | **+0.0182** | [+0.0086, +0.0260] | 0.0002 | 0.0008 | **REJECTED (BF superior)** |
| **BF vs P1 @ K1** | **+0.0155** | [+0.0074, +0.0214] | 0.0006 | 0.0018 | **REJECTED (BF superior)** |
| YAW vs P1 @ K8 | +0.0079 | [+0.0009, +0.0137] | 0.0298 | 0.0596 | ns after Holm |
| YAW vs P1 @ K1 | +0.0068 | [+0.0003, +0.0133] | 0.039 | 0.0596 | ns after Holm |
Supportive e_loc deltas point the same way (BF −0.090 m p≈0.017 both regimes; YAW −0.060 m ns). Full precision: `outputs_loc/exp20/exp20_confirmatory_holm.json`.

## Reading
**The C₄-equivariant (frame-averaged) FLAC carries confirmed-more invertible source-position information than matched-step vanilla, in BOTH context regimes** — the effect composes with the context-invariance property. Yaw augmentation moves the same direction at roughly 40% of the FA effect but does not survive the registered correction (raw p<0.05 both regimes; a suggestive, not confirmed, effect). Ordering BF > YAW > P1 holds in all 18 cells; every per-seed replicate agrees to ±0.002.
