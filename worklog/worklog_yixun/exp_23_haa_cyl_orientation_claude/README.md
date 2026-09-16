# exp_23 — CylDINO no-SSL on HAA with an orientation field (arm CYLORI)

**Query (Yixun, 2026-09-14, /goal):** "Is there a way to make our CylDINO no-SSL 40k fine tune on
HAA has better results than Vanilla FLAC baseline?" Mid-investigation Yixun added the hypothesis:
*"physically add an orientation-relative channel to the cylindrical dinov3 encoder ... in the real
world, the speaker is actually having the direction information."*

**Seat:** Claude Fable 5.1, main session. LEAN mode (exp_12 precedent): results-first, no review
rounds. Everything below is reproducible from this folder.

## 1. Diagnosis (no training) — why the cylindrical arm loses on HAA

Registered exp_19 rows (ckpt-1000, K=8, paper convention): P1 T60 3.413 vs CYL 5.411.

1. **Hypotheses ruled out.**
   - *Optimisation budget.* Val-loss trajectories (parsed from the exp_19 train logs): every arm
     bottoms out at step ~530–620 and overfits afterwards; CYL min 0.509 @530 vs P1 0.495 @570,
     val@1000 0.544 vs 0.514. A 3 % loss gap cannot explain a 59 % T60 gap.
   - *Panorama convention mismatch (mirror / azimuth offset) between AR and HAA.* `AR_md.py` and
     `HAA_md.py` share the same column→azimuth formula and `max_value`; the gauge's β offset is
     the same on both. Geometric audit (`convention_audit.py`): receiver positions fall inside the
     room reconstructed from each depth panorama under the formula convention — HAA complexBase
     0.89 inside vs 0.38 mirrored / 0.37 rolled; AR controls pick the formula everywhere (12
     panoramas). Vertical: HAA raw row 0 = floor, `flipud` → row 0 = up = AR. **No mismatch.**
2. **Per-room breakdown localises the gap** (K=8, ckpt-1000, 5 seeds; T60 / EDT):

   | room | P1 | CYL | BF | CYLSSL | YNA |
   |---|---|---|---|---|---|
   | classroomBase | 3.98 / 108.7 | 3.96 / 117.0 | 3.66 / 113.4 | 4.62 / 110.7 | 3.99 / 98.6 |
   | complexBase | 2.84 / 86.1 | 3.43 / 112.3 | 3.07 / 105.4 | 2.89 / 122.6 | 3.01 / 76.6 |
   | hallwayBase | 3.43 / 96.9 | **8.85 / 171.8** | 7.95 / 159.6 | 6.94 / 146.7 | 3.18 / 87.0 |

   Classroom is a tie. The hallway (1 m × 16 m corridor; receivers in front of AND behind the
   speaker along one axis) carries the gap, and every rotation-invariant arm (CYL, CYLSSL, BF)
   fails there while every orientation-aware arm (P1, YNA) does not.
3. **The loudspeaker is strongly directional** (`directivity_probe.py`, ground-truth RIRs,
   distance-corrected direct-path energy vs receiver azimuth about the speaker):

   | room | front sectors (−135°…−45°) | back sectors (45°…135°) | contrast |
   |---|---|---|---|
   | hallwayBase | log10(high·r²) ≈ +0.33 | ≈ −1.94 | ~23 dB |
   | classroomBase | ≈ −0.07 | ≈ −2.4 | ~23 dB |
   | complexBase | ≈ −0.3 | ≈ −2.7 | ~24 dB |
   | dampenedBase | ≈ −0.37 | ≈ −3.0 | ~26 dB |

   The speaker faces **−y (world frame) in all four Base rooms**. A yaw-invariant geometry
   encoder cannot represent "receiver direction relative to the speaker's facing", which is
   exactly what a directional source makes matter; the vanilla encoder gets it for free from the
   world-frame layout of a fixed-orientation dataset. (The DiT also receives raw xyz through the
   `source` dist-embedder in both arms, so the deficit is specifically in the geometry features.)

## 2. The intervention — a covariant orientation field

Append the loudspeaker facing direction **f** (world-frame unit vector, `md['facing']`) as a
second XYZ triple of the geometry-encoder input: `[P − X(u,v) | s·f]`, `s = 2.7` (the training
receivers' reflection-map channel RMS, so the new channels live on the same scale as the old).
The cylindrical gauge rotates **both** triples per column, so the model sees
`Rz(−θ_u) f = (cos(θ_f−θ_u), sin(θ_f−θ_u), 0)` — the column's azimuth *relative to the facing*.
Under a global yaw, **f** turns with the world, the panorama rolls, and the intertwining identity
still holds ⇒ **exact yaw equivariance is preserved** (test: 1.6e-6 on the widened backbone).

Implementation (all default-inert; arm CYL rebuilds byte-identically):
- `cylindrical_dinov3` @ 9be5216: gauge rotates every XYZ triple; `widen_input_channels()`
  appends zero-initialised input channels to the patch conv (108 package tests pass).
- FLAC @ f5fe317: `GeometryConditioner(orientation_field, orientation_scale)` concatenates the
  field; `MultiConditioner` passes `facing` through; `rotate_scene_metadata` rotates `facing`.
- `HAA_md_ori.py` = stock `HAA_md.py` + `md['facing']` from `haa_speaker_facing.json`.
- Init `HAA_init_CYLORI.ckpt` (sha `0c1c37e8…`) = exp_19's `HAA_init_CYL.ckpt` with the two
  patch-conv weights widened (3→6 in-channels, zeros). **Step 0 of CYLORI is bit-identical to
  step 0 of CYL** on a real HAA sample (`smoke_construct.py`, max|Δ| = 0.0), so arm CYL is the
  exact control ("same run with the field switched off").
- Recipe: the exp_19 registered HAA recipe verbatim (1,000 steps, 16×accum 4, AdamW 5e-6 +
  InverseLR, VAE frozen, val/ckpt every 10, seed 42, bf16-mixed). Eval: CYL's own protocol
  (fa_invariant, trivial orbit, cap 64, bf16, cfg 1.0, steps 1, per-scene), K∈{8,1} × seeds 42–46
  at ckpt-1000 (+410), plus the K=8 seed-42 steps curve.

## 3. Status

- 2026-09-14 20:29 EDT: FULL launched on GPU 0 (`chain_full_then_eval.sh`; pid 1897484; wandb
  `FLAC_exp23_HAA_CYLORI/peedunwm`), evals auto-chained. Results: `results_cylori.md` (pending).
- 2026-09-15 01:27 EDT: CYLORI chain DONE (FULL rc=0, 100 ckpts; 28/28 eval cells rc=0). **Results:
  `results_cylori.md`.** Headline (ckpt-1000, K=8, paper): T60 5.411 → **4.835** (P1 3.413), C50
  3.442 → **3.044** (2.202), EDT 119.5 → **108.2** (85.0), R@1 4.10 → 4.31 (5.18), R@10 27.65 → 28.97
  (31.69), FD 0.6035 → 0.5969 (0.5778). The facing field improves every metric at both K and both
  registered endpoints, acts where predicted (hallway T60 8.85 → 7.23, EDT 172 → 147), and its
  T60 curve is still descending through step 800 while vanilla peaks at 410 — consistent with the
  new zero-init channels being learning-rate-limited at 5e-6. It closes ~⅓ of the gap to vanilla
  but does NOT overtake it.
- 2026-09-15 ~01:29 EDT: follow-on **CYLORI27** (identical except `orientation_scale` 27 = 10× the
  effective learning speed of the new weights under Adam; ckpt every 50) auto-launched on GPU 0
  (`chain_cylori27.sh`), evals chained.
- 2026-09-15 05:58 EDT: **CYLORI27 chain DONE** (FULL rc=0, 20 ckpts; 18/18 cells rc=0). ckpt-1000,
  paper convention — K=8: T60 **3.533** (P1 3.413, CYL 5.411), C50 **2.158** (P1 2.202 → CylDINO
  better), EDT **85.8** (85.0), R@1 4.96 (5.18), R@5 18.77 (19.17), R@10 31.45 (31.69), FD 0.5814
  (0.5778). K=1: T60 3.680 (3.617), C50 **2.231** (2.254 → better), EDT 91.0 (88.0), R@1 4.68 (5.08),
  R@5 **19.16** (19.07 → better), R@10 **31.44** (31.22 → better), FD 0.5655 (0.5645). Per room the
  hallway is essentially repaired (T60 4.02 vs P1 3.43; from 8.85) and the complex room is now better
  than vanilla (2.51 vs 2.84). **Parity with vanilla at the registered endpoint** (wins 4 of 12 core
  cells, within 1–4 % elsewhere) from a clean sweep of 12/12 losses by 13–59 %. Curve: T60 3.50 @600,
  3.55 @1000 (vanilla's checkpoint-selected best 2.95 @410 remains better). Caveats: one training
  seed per arm; `orientation_scale` is a tuned constant (2.7 → partial, 27 → parity; monotone); the
  facing direction is supplied from the dataset (estimated from the RIRs; same in all rooms).
- 2026-09-15 09:4x EDT: Yixun: **"co-tenant when free"** → `gpu_watch_seed43.sh` armed: launches
  `chain_seed43.sh` (CYLORI27 seed 43 → P1 seed 43, ckpt/100, ckpt-1000 grids) on the first card
  with ≥ 8 GB free on three consecutive 60-s polls (GPU 1 checked first). Both cards held 46 GB
  xRIR jobs at arming time (GPU 1 ETA ~Sep 16 09:00, GPU 0 ~Sep 17 05:00).
- 2026-09-16 10:11 EDT: watcher launched the seed-43 chain on GPU 1 (CYLORI27 seed 43 first).
- 2026-09-16 12:04 EDT: **STOPPED at Yixun's request** ("Please stop at the earliest checkpoint of
  exp_23, I will leave the GPU to another experiment"): chain + watcher cancelled, waited for the
  step-500 checkpoint (written 12:03:41, full size), SIGINT to train.py at 12:04:07 (Lightning
  KeyboardInterrupt; SIGTERM follow-up at +90 s), GPU 1 free at 12:05:5x. `exp23_HAA_CYLORI27_s43`
  holds checkpoints 100–500 (unevaluated; resumable with `--ckpt-path` on the step-500 file). The
  P1 seed-43 run never started. The two-seed robustness check is therefore **incomplete**; the
  headline (seed-42) tables stand as published in `results_cylori.md`. Nothing deleted.
