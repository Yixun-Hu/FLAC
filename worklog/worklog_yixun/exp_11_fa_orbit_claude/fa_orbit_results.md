# exp_11 fa_orbit — Registered results

*Cluster session (Fable 5), 2026-08-12. All derived numbers regenerate via `exp11_mechanism_readout.py` (Codex-approved, zero findings) and `gen_model_comparison.py`; raw metric JSONs + SHA manifests are committed (`NEURONIC_EXP11_METRICS_SHA256SUMS.txt`). Measurement pin `89f24cd`; single training seed 42 per arm (disclosed throughout); eval seeds 42–46; full unseen split (6337 items / 17 rooms); own-protocol evals per convention.*

## R1 (primary): does a finer frame-averaging orbit beat C4? — **NO**

5-seed paired deltas vs C4L at 40k, K=8 (mean ± 95% paired-t CI, df=4; positive = worse on ↓-metrics):

| Arm | T60 ↓ | EDT ↓ | R@1 (co-primary) |
|---|---|---|---|
| C8 | **+0.299 ± 0.005** | +1.461 ± 0.042 | n.s. |
| C16 | **+0.929 ± 0.023** | +3.854 ± 0.093 | n.s. |
| C32 | **+0.732 ± 0.006** | +2.716 ± 0.060 | n.s. |

Both co-primary families resolve cleanly: the K8 T60 degradation is significant for every arm (Holm-corrected within arm); K8 R@1 is null everywhere. The trend is **non-monotone** — an inverted-U peaking at C16 (adjacent-orbit deltas: C8−C4L +0.299, C16−C8 +0.630, C32−C16 −0.197 T60) — so this is not simple "more averaging = more damage" saturation. Absolute rows live in `model_comparison.md` (all four arms, both K, batched-era fa eval).

**Bridge:** C4L (this recipe) vs historical legacy-loop C4 at 40k: T60 +0.212 (K8) — the recipe/execution bridge is material, which is why all R1 comparisons are within-recipe against C4L only.

## q9: frame averaging vs vanilla under the *same* recipe — **REVERSED**

VANL (vanilla, exact exp_11 recipe, one pin, single delta = frame averaging on/off), 40k, 5 seeds:

- VANL: K8 `8.048±0.003 / 1.0219±0.0014 / 37.319±0.090 / R@1 4.949±0.075`; K1 `9.303±0.039 / 1.0912±0.0055 / 39.802±0.310 / 4.750±0.195`.
- Paired fa−vanilla (C4L−VANL): **K8 T60 +0.366±0.011, EDT +4.180±0.091 (fa worse); C50 −0.012±0.001 (fa better, small); R@1 +0.170±0.165 (fa better, marginal)**; K1 concordant (T60 +0.458±0.048, EDT +4.224±0.195, retrieval n.s.).

Under the legacy 2-GPU recipe, fa beat vanilla (exp_07 single-delta; exp_10 A4 fair grid 6/0 best-of). Under the L40 / 8×8 / SyncBN-64 / grad-ckpt / batched-orbit recipe, vanilla wins the acoustic parameters decisively. **The fa advantage is recipe-contingent, not intrinsic.**

## R3 (yaw-flatness): the equivariance design works exactly as intended

Delta vs own 0° reference (s42, K8, 40k): every in-group rotation sits at the invariance floor for every arm (|ΔT60| ≤ 0.001, |ΔEDT| ≤ 0.011). Off-group residuals shrink ~an order of magnitude per refinement: C4L rot45 T60 +1.334 / EDT +3.234 / R@1 −1.231; C8 worst off-group T60 +0.104; C16 and C32 flat at every tested offset (|ΔT60| ≤ 0.026). Claim scope: tested offsets {5.625°, 11.25°, 22.5°, 45°}.

## R2 (mechanism 2×2): the θ=0 cost is training-side, plus a saturating eval-side toll unique to C4-training

Decomposition `Cn/aₙ − C4L/a₄ = eval + train + interaction` (s42, K8, 40k; full off-diagonal sets validated):

| n | T60 total | eval | train | interaction |
|---|---|---|---|---|
| 8 | +0.301 | +0.456 | +0.408 | −0.563 |
| 16 | +0.944 | +0.451 | +1.024 | −0.531 |
| 32 | +0.725 | +0.453 | +0.737 | −0.465 |

(EDT concordant and larger: train +2.50/+4.66/+3.12.) The training-side term dominates and carries the inverted-U. The eval-side term is constant in n — evaluating the C4-trained model under ANY finer orbit costs ~+0.45 T60 / ~1.0 EDT / ~−0.45 R@1, saturating already at a8 — and the **legacy exp_07 B-F checkpoint replicates it era-clean** (+0.490/+0.481/+0.482 vs its own batched-era a4 baseline, same pin).

Full cross matrix (T60 vs own protocol): finer-trained arms are near-robust to any eval-orbit change (worst +0.107, shrinking with training orbit); only C4-trained pays the large toll, and only in the finer direction.

## Economics & integrity

- Futility gates at 20k/30k: no arm stopped; all four reached 40k.
- Eval-protocol flags matched training per arm throughout (fa arms `--cond-method fa_invariant --frame-avg-angles <orbit>`; VANL vanilla); execution era labeled per row (batched vs legacy-loop).
- Disclosed recipe deltas vs paper config: L40 rung 8×8 (BN=64 preserved via SyncBN), uniform ViT grad-ckpt, batched orbit averaging with chunk-shared train-mode RoPE rescale draws (eval-mode equivalence fp32-gated: rel ≤ 5e-6).
- Registered 67.5k extension option: superseded — Yixun commissioned 100k extensions of all four fa arms + VANL (in flight; jobs 3684149–53).

## In flight (not part of the registered readout)

100k continuation of all five arms with rolling 5-seed dual-K trajectory evals every 2500 steps (>40k), reported in the trajectory figures with mean ± sd bands once validated under the `traj` contract.
