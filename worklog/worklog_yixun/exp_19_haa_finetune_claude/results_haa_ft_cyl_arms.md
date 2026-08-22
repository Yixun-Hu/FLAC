# exp_19 HAA finetune — CYL + CYLSSL rows (the first VALID cylindrical HAA numbers)

*2026-08-22, exp-12 session. Both arms trained on the registered released-HAA recipe
(1,000 steps, batch 16 × accum 4, AdamW 5e-6, seed 42, bf16-mixed) AFTER the cylindrical
routing port (`093f388`) — the earlier "CYL" run was quarantined for silently building a
vanilla architecture. Runtime verified per arm from the train logs: CYL prints
`azimuth_mode=full, prefix_mode=strip`; CYLSSL prints `azimuth_mode=lowband,
prefix_mode=m0_registers` + its init carries the SSL backbone. Eval: 2 ckpts × 2 K ×
5 seeds per arm, each arm's own protocol (fa_invariant, trivial orbit, `_a1`), evaluator
pinned @ d2f94cae (exp_21's, verified protocol-inert for these methods).*

- **CYL** init: EMA of BB-CYL AR-40k (exp-09 backbone; `HAA_init_CYL.ckpt`, 49ca5250…)
- **CYLSSL** init: EMA of exp-12 arm B DDP AR-42.5k (`HAA_init_CYLSSL.ckpt`, 77949de5…) —
  the true AR-40k was destroyed in the 08-19 disk emergency; +2,500 AR steps, disclosed.
- Final FT losses: CYL train 0.262 / val 0.544 · CYLSSL train 0.252 / val 0.536.

## Seven-arm endpoint table (paper convention, ckpt-1000, 5 eval seeds)

## K = 8 (paper convention)

| Method | T60↓ | C50↓ | EDT↓ | R@1↑ | R@5↑ | R@10↑ | FD↓ |
|---|---|---|---|---|---|---|---|
| Vanilla FLAC (P1→HAA) | 3.4130 ± 0.0127 | 2.2016 ± 0.0123 | 84.994 ± 0.404 | 5.184 ± 0.334 | 19.167 ± 0.239 | 31.693 ± 0.399 | 0.5778 ± 0.0009 |
| Yaw-Aug init, aug OFF in FT | 3.3910 ± 0.0325 | 2.0956 ± 0.0107 | 77.256 ± 0.285 | 4.761 ± 0.367 | 18.543 ± 0.392 | 30.024 ± 0.304 | 0.5728 ± 0.0010 |
| FA(B-F) init, vanilla FT | 3.6259 ± 0.0433 | 2.2470 ± 0.0125 | 93.964 ± 0.334 | 3.954 ± 0.338 | 14.167 ± 0.224 | 23.947 ± 0.368 | 0.5952 ± 0.0016 |
| Yaw-Aug, aug ON in FT | 4.0921 ± 0.0371 | 2.7772 ± 0.0155 | 91.755 ± 0.372 | 4.133 ± 0.176 | 16.100 ± 0.571 | 27.335 ± 0.191 | 0.5887 ± 0.0012 |
| Per-angle FA (B-F→HAA) | 4.8924 ± 0.0219 | 3.2102 ± 0.0119 | 113.034 ± 0.396 | 3.959 ± 0.448 | 15.734 ± 0.381 | 26.811 ± 0.989 | 0.6032 ± 0.0011 |
| Cyl-DINOv3 no-SSL (AR-40k→HAA) | 5.4110 ± 0.0230 | 3.4421 ± 0.0209 | 119.502 ± 0.720 | 4.100 ± 0.440 | 16.268 ± 0.238 | 27.652 ± 0.489 | 0.6035 ± 0.0008 |
| Cyl-DINOv3 SSL (AR-42.5k→HAA) | 4.8187 ± 0.0305 | 3.1032 ± 0.0129 | 113.097 ± 0.510 | 4.070 ± 0.358 | 15.832 ± 0.429 | 26.376 ± 0.584 | 0.5859 ± 0.0015 |

## K = 1 (paper convention)

| Method | T60↓ | C50↓ | EDT↓ | R@1↑ | R@5↑ | R@10↑ | FD↓ |
|---|---|---|---|---|---|---|---|
| Vanilla FLAC (P1→HAA) | 3.6167 ± 0.0585 | 2.2541 ± 0.0161 | 87.950 ± 0.919 | 5.075 ± 0.421 | 19.068 ± 0.698 | 31.223 ± 0.209 | 0.5645 ± 0.0015 |
| Yaw-Aug init, aug OFF in FT | 3.5361 ± 0.0297 | 2.0999 ± 0.0035 | 79.716 ± 0.689 | 4.611 ± 0.459 | 18.186 ± 0.360 | 30.553 ± 0.706 | 0.5629 ± 0.0012 |
| FA(B-F) init, vanilla FT | 3.8225 ± 0.0515 | 2.2333 ± 0.0212 | 97.786 ± 0.820 | 3.491 ± 0.346 | 13.737 ± 0.551 | 24.031 ± 0.318 | 0.5779 ± 0.0028 |
| Yaw-Aug, aug ON in FT | 4.2378 ± 0.0545 | 2.7752 ± 0.0247 | 93.889 ± 0.406 | 4.035 ± 0.426 | 16.269 ± 0.447 | 27.199 ± 0.512 | 0.5751 ± 0.0029 |
| Per-angle FA (B-F→HAA) | 5.1197 ± 0.0603 | 3.2028 ± 0.0187 | 116.180 ± 1.419 | 3.568 ± 0.363 | 15.786 ± 0.699 | 27.115 ± 0.481 | 0.5834 ± 0.0022 |
| Cyl-DINOv3 no-SSL (AR-40k→HAA) | 5.5113 ± 0.0472 | 3.4607 ± 0.0220 | 122.045 ± 0.951 | 3.867 ± 0.111 | 16.248 ± 0.411 | 27.433 ± 0.529 | 0.5827 ± 0.0021 |
| Cyl-DINOv3 SSL (AR-42.5k→HAA) | 4.9688 ± 0.0521 | 3.0780 ± 0.0165 | 114.248 ± 1.461 | 3.919 ± 0.336 | 15.802 ± 0.605 | 26.661 ± 0.724 | 0.5794 ± 0.0029 |


## Reading (K=8, paper convention)

1. **SSL clearly helps the cylindrical backbone on HAA transfer**: CYLSSL beats CYL no-SSL
   on T60 (4.82 vs 5.41, −11%), C50 (3.10 vs 3.44, −10%), EDT (113.1 vs 119.5, −5%) and FD
   (0.5859 vs 0.6035), with retrieval a wash. Same direction at K=1.
2. **CYLSSL lands at Per-angle-FA (B-F) level**: statistically indistinguishable from BF on
   T60/EDT/R@k and slightly better on C50 (3.10 vs 3.21) and FD — i.e. the SSL cylindrical
   backbone transfers about as well as the strongest fa_invariant arm previously measured.
3. **Neither cylindrical arm approaches the vanilla-family arms on HAA** (P1 3.41 T60 /
   YNA 3.39): the fa/cylindrical family's HAA transfer gap (already visible for BF) stands.
4. Caveats: eval-seed σ only (one FT run per arm); CYLSSL's init is +2.5k AR steps vs
   CYL's 40k; CYL vs CYLSSL also differ by the C3/C4 knobs (lowband + m0_registers), so
   "SSL" here means the full arm-B treatment (knobs + SSL), not SSL in isolation.
