# exp_35 orbitrir_yaw_evidence — results (PARTIAL)

Evaluator pin `5b065801e9d1c1844461ec84cd0558702c714326`; campaign digest `da82d0121413b792e2f42b94026b884d371d995c17e2dcc111c7edd58f8650c1`; blocks CTL, BF-pin, BF-dup, OFF-P1, OFF-BF, OFF-CANON, CANON-45, E2-FAIL. Written by `depth_canon_collect.py --exp35`; never hand-edited.

external_reference: the exp_25 results `worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_results.json` (sha256 `ec3fcd08e3d210524b50246d9bfdf084e72835432a5d06450085e0cad12eb6e0`, pin `d6ca307a85b9b0a75ec9d47f4d237e63a3db527e`; its rows are quoted, never paired with this campaign's cells); 24 exp_25 cells verified offline at `d6ca307a85b9b0a75ec9d47f4d237e63a3db527e` (verify_cell(pin=d6ca307a, dump_check='receipt', check_link=False)); the stream registry `9ec31f32a9d89cf9ffa90938d7f1198658af2eafa3b53675605d9fb925365a24`; the A6 CSV `0ae554c09f45ce89b8957cf75a3d15e44f4173f4f8987497dbbebf16bbfb23da` (E2 only; its 45-degree rows enter table (b) on an E2 PASS).

**PARTIAL:** 68 cell(s) not run (exp35_P1_OFF_S40000_K1_s42_th135, exp35_P1_OFF_S40000_K1_s42_th225, exp35_P1_OFF_S40000_K1_s43_th135, exp35_P1_OFF_S40000_K1_s43_th225 ...).

## PARTIAL — E1 — the control cells (exact; a hard stop)

- CTL K1: equal
- CTL K8: equal

## PARTIAL — E2 — decides the 45-degree source only

PASS — |new - A6 CSV| <= T60 0.001, C50 0.0005, EDT 0.005, FD 0.0005, RIR_to_GT_RIR_R@1 0.2, RIR_to_GT_RIR_R@5 0.2, RIR_to_GT_RIR_R@10 0.2 (all 14).

- K1 T60: new 9.4861, A6 9.4859, |diff| 0.0002 <= 0.001: True
- K1 C50: new 1.0547, A6 1.0547, |diff| 0 <= 0.0005: True
- K1 EDT: new 41.236, A6 41.2371, |diff| 0.0011 <= 0.005: True
- K1 FD: new 0.3284, A6 0.3284, |diff| 0 <= 0.0005: True
- K1 R@1: new 5.1917, A6 5.2391, |diff| 0.0474 <= 0.2: True
- K1 R@5: new 15.6857, A6 15.7014, |diff| 0.0157 <= 0.2: True
- K1 R@10: new 23.6547, A6 23.7494, |diff| 0.0947 <= 0.2: True
- K8 T60: new 8.1899, A6 8.189, |diff| 0.0009 <= 0.001: True
- K8 C50: new 0.9804, A6 0.9805, |diff| 0.0001 <= 0.0005: True
- K8 EDT: new 38.8107, A6 38.8102, |diff| 0.0005 <= 0.005: True
- K8 FD: new 0.3333, A6 0.3332, |diff| 0.0001 <= 0.0005: True
- K8 R@1: new 5.3338, A6 5.3495, |diff| 0.0157 <= 0.2: True
- K8 R@5: new 16.4747, A6 16.3484, |diff| 0.126 <= 0.2: True
- K8 R@10: new 24.3333, A6 24.2386, |diff| 0.0947 <= 0.2: True

## PARTIAL — (a) theta = 0, unseen, mean ± sd over seeds 42–46, item-weighted

| method (source) | K | T60 | C50 | EDT | R@1 | R@5 | R@10 |
|---|---|---|---|---|---|---|---|
| FLAC (external_reference.exp25.table.P1) | 1 | 10.287 ± 0.026 (n 5) | 1.0884 ± 0.0088 (n 5) | 43.437 ± 0.398 (n 5) | 4.971 ± 0.154 (n 5) | 15.105 ± 0.118 (n 5) | 22.525 ± 0.210 (n 5) |
| FLAC (external_reference.exp25.table.P1) | 8 | 8.993 ± 0.011 (n 5) | 1.0093 ± 0.0035 (n 5) | 40.650 ± 0.103 (n 5) | 5.208 ± 0.105 (n 5) | 15.471 ± 0.196 (n 5) | 23.450 ± 0.059 (n 5) |
| CANON (external_reference.exp25.table.CANON) | 1 | 9.788 ± 0.040 (n 5) | 1.1049 ± 0.0049 (n 5) | 44.785 ± 0.353 (n 5) | 5.059 ± 0.221 (n 5) | 15.534 ± 0.207 (n 5) | 23.153 ± 0.225 (n 5) |
| CANON (external_reference.exp25.table.CANON) | 8 | 8.418 ± 0.009 (n 5) | 1.0272 ± 0.0019 (n 5) | 42.449 ± 0.105 (n 5) | 5.192 ± 0.098 (n 5) | 15.894 ± 0.116 (n 5) | 23.901 ± 0.171 (n 5) |
| OrbitRIR (exp_35 BF-pin T s42-46) | 1 | 9.543 ± 0.054 (n 5) | 1.0559 ± 0.0041 (n 5) | 41.754 ± 0.349 (n 5) | 5.132 ± 0.148 (n 5) | 16.039 ± 0.249 (n 5) | 23.724 ± 0.152 (n 5) |
| OrbitRIR (exp_35 BF-pin T s42-46) | 8 | 8.202 ± 0.017 (n 5) | 0.9778 ± 0.0015 (n 5) | 38.794 ± 0.074 (n 5) | 5.394 ± 0.047 (n 5) | 16.459 ± 0.046 (n 5) | 24.194 ± 0.151 (n 5) |
| fa scratch B-F @40k (equivariant, 59% budget) — B-F HISTORICAL, quoted, reference only | 1 | 9.543 ± 0.054 | 1.0559 ± 0.0040 | 41.754 ± 0.347 | 5.166 ± 0.166 | 16.071 ± 0.241 | 23.721 ± 0.150 |
| fa scratch B-F @40k (equivariant, 59% budget) — B-F HISTORICAL, quoted, reference only | 8 | 8.202 ± 0.017 | 0.9778 ± 0.0015 | 38.793 ± 0.074 | 5.387 ± 0.075 | 16.456 ± 0.038 | 24.198 ± 0.164 |

B-F HISTORICAL: quoted from model_comparison.md (exp_07 B-F @40k, fa_invariant eval, legacy per-angle loop, its own historical evaluator pin) -- frozen for exp_35 as the exp_25 results quoted it; not re-measured.

## PARTIAL — (b) 45 degrees, unseen, mean ± sd over seeds 42–46, item-weighted

| method (source) | K | T60 | C50 | EDT | R@1 | R@5 | R@10 |
|---|---|---|---|---|---|---|---|
| FLAC (A6 CSV 'Vanilla P1@40k' 45 deg s42-46 (E2 PASS)) | 1 | 10.972 ± 0.058 (n 5) | 1.1435 ± 0.0068 (n 5) | 45.037 ± 0.328 (n 5) | 3.901 ± 0.078 (n 5) | 13.189 ± 0.076 (n 5) | 19.899 ± 0.101 (n 5) |
| FLAC (A6 CSV 'Vanilla P1@40k' 45 deg s42-46 (E2 PASS)) | 8 | 9.715 ± 0.005 (n 5) | 1.0713 ± 0.0021 (n 5) | 42.390 ± 0.092 (n 5) | 4.081 ± 0.071 (n 5) | 13.495 ± 0.112 (n 5) | 20.388 ± 0.112 (n 5) |
| CANON (exp_25 C4+45 th45 s42 (external, d6ca307a) + exp_35 CANON-45 s43-46) | 1 | 9.805 (n 1) | 1.1049 (n 1) | 44.288 (n 1) | 5.097 (n 1) | 15.623 (n 1) | 23.276 (n 1) |
| CANON (exp_25 C4+45 th45 s42 (external, d6ca307a) + exp_35 CANON-45 s43-46) | 8 | 8.429 (n 1) | 1.0255 (n 1) | 42.454 (n 1) | 5.160 (n 1) | 15.954 (n 1) | 23.812 (n 1) |
| OrbitRIR (A6 CSV 'FA B-F@40k' 45 deg s42-46 (E2 PASS)) | 1 | 11.019 ± 0.052 (n 5) | 1.1649 ± 0.0025 (n 5) | 43.787 ± 0.210 (n 5) | 3.740 ± 0.101 (n 5) | 12.182 ± 0.112 (n 5) | 19.009 ± 0.269 (n 5) |
| OrbitRIR (A6 CSV 'FA B-F@40k' 45 deg s42-46 (E2 PASS)) | 8 | 9.640 ± 0.006 (n 5) | 1.0905 ± 0.0022 (n 5) | 40.901 ± 0.050 (n 5) | 3.835 ± 0.105 (n 5) | 12.365 ± 0.054 (n 5) | 19.523 ± 0.194 (n 5) |

Pooled rows, per point (seed, campaign, pin, attention backend):
- CANON K1: s42 exp25 @d6ca307a85b9 (sdpa (plan §3b: exp_25's backend; its schema-1 receipts do not record it))
- CANON K8: s42 exp25 @d6ca307a85b9 (sdpa (plan §3b: exp_25's backend; its schema-1 receipts do not record it))

## PARTIAL — (b) random yaw, unseen, mean ± sd over seeds 42–46, item-weighted

| method (source) | K | T60 | C50 | EDT | R@1 | R@5 | R@10 |
|---|---|---|---|---|---|---|---|
| FLAC (exp_25 P1-pin R s42-46 (external, d6ca307a)) | 1 | 11.033 ± 0.035 (n 5) | 1.1714 ± 0.0107 (n 5) | 46.613 ± 0.476 (n 5) | 4.248 ± 0.228 (n 5) | 13.779 ± 0.146 (n 5) | 20.944 ± 0.194 (n 5) |
| FLAC (exp_25 P1-pin R s42-46 (external, d6ca307a)) | 8 | 9.741 ± 0.028 (n 5) | 1.0960 ± 0.0088 (n 5) | 43.911 ± 0.196 (n 5) | 4.510 ± 0.239 (n 5) | 14.262 ± 0.166 (n 5) | 21.663 ± 0.313 (n 5) |
| CANON (exp_25 CANON R s42-46 (external, d6ca307a)) | 1 | 9.788 ± 0.040 (n 5) | 1.1049 ± 0.0049 (n 5) | 44.783 ± 0.352 (n 5) | 5.062 ± 0.241 (n 5) | 15.522 ± 0.211 (n 5) | 23.188 ± 0.227 (n 5) |
| CANON (exp_25 CANON R s42-46 (external, d6ca307a)) | 8 | 8.419 ± 0.010 (n 5) | 1.0273 ± 0.0019 (n 5) | 42.450 ± 0.103 (n 5) | 5.211 ± 0.090 (n 5) | 15.903 ± 0.116 (n 5) | 23.904 ± 0.163 (n 5) |
| OrbitRIR (exp_35 BF-pin R s42-46) | 1 | 10.263 ± 0.067 (n 5) | 1.1121 ± 0.0049 (n 5) | 42.968 ± 0.400 (n 5) | 4.431 ± 0.104 (n 5) | 14.114 ± 0.158 (n 5) | 21.474 ± 0.225 (n 5) |
| OrbitRIR (exp_35 BF-pin R s42-46) | 8 | 8.883 ± 0.060 (n 5) | 1.0342 ± 0.0027 (n 5) | 40.102 ± 0.161 (n 5) | 4.548 ± 0.162 (n 5) | 14.755 ± 0.121 (n 5) | 22.108 ± 0.204 (n 5) |

## PARTIAL — (c) Tier 3 spread, item-weighted (CANON / P1-pin / YAW-pin quoted from the exp_25 JSON) and the B-F verdict

**Tier 3 B-F: FAIL** — Delta_C4(BF) <= 0.1 x Delta_C4(P1-pin, the exp_25 JSON) and |Delta_R(BF)| <= sd(m_Z BF), every metric incl. FD, both K, item-weighted. pass_c4 fails: none; pass_r fails: K1 T60, K1 C50, K1 EDT, K1 FD, K1 R@1, K1 R@5, K1 R@10, K8 T60, K8 C50, K8 EDT, K8 FD, K8 R@1, K8 R@5, K8 R@10.

| arm (source) | K | metric | Delta_C4 | 45-deg probe | Delta_R | sd(m_Z) |
|---|---|---|---|---|---|---|
| CANON (external_reference.exp25.tier3.CANON) | 1 | T60 | 0 | -0.0003 | -3.553e-16 | 0.04006 |
| CANON (external_reference.exp25.tier3.CANON) | 1 | C50 | 0 | 0 | 6e-05 | 0.004882 |
| CANON (external_reference.exp25.tier3.CANON) | 1 | EDT | 0 | 0.0074 | -0.0022 | 0.353 |
| CANON (external_reference.exp25.tier3.CANON) | 1 | FD | 0 | -0.0001 | -4e-05 | 0.0005404 |
| CANON (external_reference.exp25.tier3.CANON) | 1 | R@1 | 0 | 0 | 0.0032 | 0.2208 |
| CANON (external_reference.exp25.tier3.CANON) | 1 | R@5 | 0 | -0.0158 | -0.01262 | 0.2071 |
| CANON (external_reference.exp25.tier3.CANON) | 1 | R@10 | 0 | 0.0473 | 0.03474 | 0.2252 |
| CANON (external_reference.exp25.tier3.CANON) | 8 | T60 | 0 | -0.0003 | 0.00052 | 0.009158 |
| CANON (external_reference.exp25.tier3.CANON) | 8 | C50 | 0 | 0.0001 | 6e-05 | 0.001862 |
| CANON (external_reference.exp25.tier3.CANON) | 8 | EDT | 0 | 0.0066 | 0.00176 | 0.1047 |
| CANON (external_reference.exp25.tier3.CANON) | 8 | FD | 0 | 0 | 0 | 0.0001225 |
| CANON (external_reference.exp25.tier3.CANON) | 8 | R@1 | 0 | 0 | 0.01892 | 0.0979 |
| CANON (external_reference.exp25.tier3.CANON) | 8 | R@5 | 0 | 0.0315 | 0.00948 | 0.1162 |
| CANON (external_reference.exp25.tier3.CANON) | 8 | R@10 | 0 | -0.0158 | 0.00316 | 0.1705 |
| P1 (external_reference.exp25.tier3.P1) | 1 | T60 | 0.9223 | 0.6827 | 0.7455 | 0.02649 |
| P1 (external_reference.exp25.tier3.P1) | 1 | C50 | 0.1331 | 0.0564 | 0.08292 | 0.008799 |
| P1 (external_reference.exp25.tier3.P1) | 1 | EDT | 6.498 | 1.493 | 3.176 | 0.3984 |
| P1 (external_reference.exp25.tier3.P1) | 1 | FD | 0.0059 | 0.0094 | 0.00478 | 0.0004278 |
| P1 (external_reference.exp25.tier3.P1) | 1 | R@1 | 0.647 | -1.341 | -0.7227 | 0.1542 |
| P1 (external_reference.exp25.tier3.P1) | 1 | R@5 | 0.7101 | -1.673 | -1.326 | 0.1178 |
| P1 (external_reference.exp25.tier3.P1) | 1 | R@10 | 1.073 | -2.841 | -1.581 | 0.2101 |
| P1 (external_reference.exp25.tier3.P1) | 8 | T60 | 0.9033 | 0.7308 | 0.7489 | 0.01085 |
| P1 (external_reference.exp25.tier3.P1) | 8 | C50 | 0.1337 | 0.0632 | 0.08668 | 0.003527 |
| P1 (external_reference.exp25.tier3.P1) | 8 | EDT | 6.807 | 1.837 | 3.262 | 0.1027 |
| P1 (external_reference.exp25.tier3.P1) | 8 | FD | 0.0074 | 0.0105 | 0.00578 | 0.000228 |
| P1 (external_reference.exp25.tier3.P1) | 8 | R@1 | 0.5523 | -1.184 | -0.6975 | 0.1047 |
| P1 (external_reference.exp25.tier3.P1) | 8 | R@5 | 0.7259 | -2.13 | -1.209 | 0.196 |
| P1 (external_reference.exp25.tier3.P1) | 8 | R@10 | 1.089 | -3.156 | -1.786 | 0.05903 |
| YAW (external_reference.exp25.tier3.YAW) | 1 | T60 | 0.0607 | 0.0531 | 0.02988 | 0.05224 |
| YAW (external_reference.exp25.tier3.YAW) | 1 | C50 | 0.0115 | 0.0068 | 0.00996 | 0.005073 |
| YAW (external_reference.exp25.tier3.YAW) | 1 | EDT | 0.386 | 0.5974 | 0.2924 | 0.3794 |
| YAW (external_reference.exp25.tier3.YAW) | 1 | FD | 0.0008 | -0.0008 | -0.00032 | 0.0004087 |
| YAW (external_reference.exp25.tier3.YAW) | 1 | R@1 | 0.1262 | 0.0474 | -0.1957 | 0.09134 |
| YAW (external_reference.exp25.tier3.YAW) | 1 | R@5 | 0.426 | 0.2367 | -0.1767 | 0.2555 |
| YAW (external_reference.exp25.tier3.YAW) | 1 | R@10 | 0.5996 | 0.5365 | -0.2525 | 0.165 |
| YAW (external_reference.exp25.tier3.YAW) | 8 | T60 | 0.0744 | 0.0483 | 0.01498 | 0.01457 |
| YAW (external_reference.exp25.tier3.YAW) | 8 | C50 | 0.0127 | 0.0058 | 0.01008 | 0.001647 |
| YAW (external_reference.exp25.tier3.YAW) | 8 | EDT | 0.3492 | 0.6161 | 0.3032 | 0.05797 |
| YAW (external_reference.exp25.tier3.YAW) | 8 | FD | 0.0012 | -0.0008 | -0.00026 | 0.0003033 |
| YAW (external_reference.exp25.tier3.YAW) | 8 | R@1 | 0.3471 | -0.1421 | -0.243 | 0.03633 |
| YAW (external_reference.exp25.tier3.YAW) | 8 | R@5 | 0.2999 | 0.142 | -0.3093 | 0.2208 |
| YAW (external_reference.exp25.tier3.YAW) | 8 | R@10 | 0.2367 | 0.0789 | -0.243 | 0.1332 |
| BF (tier3.BF.arms.BF) | 1 | T60 | 0.0017 | 1.567 | 0.7203 | 0.05414 |
| BF (tier3.BF.arms.BF) | 1 | C50 | 0.0002 | 0.1083 | 0.05618 | 0.004063 |
| BF (tier3.BF.arms.BF) | 1 | EDT | 0.0031 | 2.251 | 1.214 | 0.3486 |
| BF (tier3.BF.arms.BF) | 1 | FD | 0 | 0.0056 | 0.00184 | 0.0004099 |
| BF (tier3.BF.arms.BF) | 1 | R@1 | 0.0158 | -1.357 | -0.7006 | 0.1482 |
| BF (tier3.BF.arms.BF) | 1 | R@5 | 0.0473 | -3.44 | -1.925 | 0.2486 |
| BF (tier3.BF.arms.BF) | 1 | R@10 | 0.0158 | -4.276 | -2.25 | 0.152 |
| BF (tier3.BF.arms.BF) | 8 | T60 | 0.0007 | 1.452 | 0.6808 | 0.01703 |
| BF (tier3.BF.arms.BF) | 8 | C50 | 0.0001 | 0.112 | 0.05642 | 0.001528 |
| BF (tier3.BF.arms.BF) | 8 | EDT | 0.0043 | 2.093 | 1.308 | 0.07355 |
| BF (tier3.BF.arms.BF) | 8 | FD | 0.0001 | 0.0065 | 0.00206 | 0.0001673 |
| BF (tier3.BF.arms.BF) | 8 | R@1 | 0.0315 | -1.546 | -0.8458 | 0.04654 |
| BF (tier3.BF.arms.BF) | 8 | R@5 | 0.0316 | -4.15 | -1.704 | 0.046 |
| BF (tier3.BF.arms.BF) | 8 | R@10 | 0.0315 | -4.513 | -2.086 | 0.1511 |

## PARTIAL — Tier 2-C4 — OrbitRIR's invariance claim (scored dumps vs the B-F composition control)

**PASS** (the registered p99 verdict: PASS) — every C4 pair p99(relL2) <= 3 x its matched B-F control p99 and <= 0.01, the pooled C4 p99 <= 3 x the pooled control p99; alarm > 0.1; T-dup max relL2 == 0 and dump sha256 equal; 45-degree / random-yaw: descriptive.

| cell | control | control p99 | relL2 p99 | relL2 max | matched | cap |
|---|---|---|---|---|---|---|
| exp35_BF_C4_S40000_K1_s42_th90 | d128 | 0.005276 | 0.006147 | 0.03057 | True | True |
| exp35_BF_C4_S40000_K1_s42_th180 | d256 | 0.00571 | 0.006331 | 0.02233 | True | True |
| exp35_BF_C4_S40000_K1_s42_th270 | d384 | 0.004747 | 0.00625 | 0.03062 | True | True |
| exp35_BF_C4_S40000_K8_s42_th90 | d128 | 0.005276 | 0.005984 | 0.01154 | True | True |
| exp35_BF_C4_S40000_K8_s42_th180 | d256 | 0.00571 | 0.006208 | 0.01391 | True | True |
| exp35_BF_C4_S40000_K8_s42_th270 | d384 | 0.004747 | 0.005939 | 0.0125 | True | True |

Pooled C4: PASS (p99 0.006189, control 0.005707). T-dup determinism: PASS (exp35_BF_T_S40000_K1_s42_dup: max 0, dump sha256 equal True; exp35_BF_T_S40000_K8_s42_dup: max 0, dump sha256 equal True). Alarm: none. eps_cond (B-F, fp32; reported, no verdict): {'value': 0.3639002344719973, 'argmax': {'key': 'd256', 'id': 'context_poses', 'position': 7}, 'verdict': None}.

Descriptive only (outside the claim, no verdict) — 45 degrees and random yaw vs theta = 0:
- exp35_BF_C4_S40000_K1_s42_th45: relL2 p99 0.4917, max 1.077; relLinf max 2.748
- exp35_BF_C4_S40000_K8_s42_th45: relL2 p99 0.4723, max 1.084; relLinf max 2.823
- exp35_BF_R_S40000_K1_s42: relL2 p99 0.4353, max 1.126; relLinf max 1.661
- exp35_BF_R_S40000_K1_s43: relL2 p99 0.4151, max 1.11; relLinf max 2.049
- exp35_BF_R_S40000_K1_s44: relL2 p99 0.4243, max 1.095; relLinf max 3.378
- exp35_BF_R_S40000_K1_s45: relL2 p99 0.4251, max 0.9094; relLinf max 3.129
- exp35_BF_R_S40000_K1_s46: relL2 p99 0.4265, max 1.107; relLinf max 2.076
- exp35_BF_R_S40000_K8_s42: relL2 p99 0.4226, max 1.089; relLinf max 1.8
- exp35_BF_R_S40000_K8_s43: relL2 p99 0.4184, max 1.087; relLinf max 2.74
- exp35_BF_R_S40000_K8_s44: relL2 p99 0.417, max 1.105; relLinf max 3.309
- exp35_BF_R_S40000_K8_s45: relL2 p99 0.4226, max 1.081; relLinf max 2.934
- exp35_BF_R_S40000_K8_s46: relL2 p99 0.4307, max 1.1; relLinf max 1.88
