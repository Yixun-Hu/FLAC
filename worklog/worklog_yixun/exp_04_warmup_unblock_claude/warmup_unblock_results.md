# Results — exp_04_warmup_unblock

Full 6337-item unseen split (announcement 01); 5 seeds (42–46) per eval; gate = exp_01 baseline (K=1 T60 9.969±0.039, C50 1.046±0.006, EDT 39.95±0.37; K=8 8.609±0.012 / 0.968±0.003 / 37.10±0.07); code `6a6a421`.

## W1 — warmup control (R1b recipe + lr warmup 0→5e-6 over 200 steps): **GATE FAIL**

| K | T60 | C50 | EDT | R@1 | vs R1b (no warmup) |
|---|---|---|---|---|---|
| 1 | 10.485±0.070 (6.4σ) | 1.077±0.010 (2.6σ) | 42.96±0.19 (7.2σ) | 6.81 (0.06σ PASS) | ΔT60 +0.02, ΔEDT −0.31 |
| 8 | 9.206±0.009 (39.6σ) | 0.994±0.003 (5.9σ) | 40.17±0.05 (36.5σ) | 7.09 (0.23σ PASS) | ΔT60 +0.01, ΔEDT −0.34 |

W1 ≡ R1b within seed noise on every metric → **Adam-transient hypothesis falsified** (warmup engagement runtime-verified via the probe's exact lr ramp). Two differently-conditioned runs converge to the same regressed optimum.

## W0 — lr=0 null control (only BN running stats can move): **GATE FAIL (partial regression)**

| K | T60 | C50 | EDT | R@1 |
|---|---|---|---|---|
| 1 | 10.132±0.054 (2.4σ) | **1.050 (0.4σ PASS)** | 41.10±0.10 (3.0σ) | 6.58 (1.1σ PASS) |
| 8 | 8.823±0.007 (15.4σ) | **0.966 (0.6σ PASS)** | 38.39±0.01 (19.0σ) | 6.88 (1.4σ PASS) |

Zero gradient steps; ~30% of the full fine-tune T60/EDT damage appears anyway. BN buffers drift only when incoming statistics differ from stored running stats ⇒ **the conditioning-data statistics from this repo's loader differ from the released checkpoint's training data — data-pipeline drift proven gradient-free.**

## Attribution ledger (across exp_03 + exp_04)

| Mechanism | Evidence | Share of T60/EDT damage |
|---|---|---|
| Data-statistics drift via BN (RIR encoder) | W0: regression at lr=0 | **~30%** |
| Gradient adaptation toward the drifted optimum | R1b ≡ W1 (warmup-independent convergence) | **~70%** |
| Adam second-moment transient | W1 ≡ R1b | dead |
| EMA-vs-online weights | exp_03 corrected diagnostic | ≤15% (subsumed) |
| Effective-batch noise | R1→R1b delta | partial (~35–40% recovery), insufficient |
| Learning-rate magnitude | 5e-6 constant = 8× below original final lr | excluded |

Signature throughout: energy-decay metrics (T60/EDT) damaged, C50 partially, retrieval (R@1/5/10) never — the drift specifically perturbs the decay-envelope statistics of the reference-RIR conditioning path.

## Not run (registered stop)

W2–W4b (fa_invariant fine-tune + H3/H2 verdicts) — uninterpretable until a vanilla control passes the gate. H1 remains proven at the infrastructure level (exp_03).
