# yaw_gen_params_set_up — exp_14 campaign configuration (written at launch)

**Status:** DRAFT pending `_full` review verdict; finalized (pin filled) at first submission. Conditional launch approval: Yixun 2026-08-11T21:39 EDT ("Once the ladder passes, launch approve").

## Arms (all step 40,000, training seed 42, 8×L40 micro-8 recipe; sha256 in `exp14_ckpt_expect.json`, cross-verified against exp_11's audited registry)

| Arm | Conditioning | Eval protocol |
|---|---|---|
| VANL | vanilla | `--cond-method vanilla` |
| C4L | fa_invariant C4 | `--cond-method fa_invariant --frame-avg-angles 0,90,180,270` |
| C8 | fa_invariant C8 | 8 angles (k·45°) |
| C16 | fa_invariant C16 | 16 angles (k·22.5°) |
| C32 | fa_invariant C32 | 32 angles (k·11.25°) |

## Protocol constants (every cell)

Full published unseen split (announcement 01; K=8 `acousticroom_unseeneval.json`, K=1 `acousticroom_unseeneval_1.json`, expected stream count **6337** enforced), EMA weights, cfg 1.0, 1 diffusion step, bf16 conditioning autocast, `--batch-size 64 --num-workers 4`, `--record-stream`, `--record-per-scene` (**expected_scenes 10** — the release family grouping; the split spans 17 physical rooms, but `AR_md.py` sets `md['scene']` to the room FAMILY, so the metric callback's per-scene mean is over 10 families). Slurm per cell: 1× L40, 10 CPU, 32 GB, 3 h limit. Node excludes: ECC-flaky list (neu301/303/305/306/317/319/322/332).

## Grid (106 cells, one campaign pin)

- **Z** (θ=0 reference): 5 arms × K∈{1,8} × seeds 42–46 = 50
- **R** (random yaw): same × `--rotate-mode random --rotate-seed <eval seed>` = 50
- **V** (validity, s42/K8, fixed): C4L/C8/C16/C32/VANL @90° + C4L@45° = 6

## Estimand & analysis (pre-registered; plan Rev 2 §4 + worklog rulings 2026-08-11)

- Primary: absolute `m_R` (H-P endpoint C32 vs VANL, K=8). Secondary: paired Δ (H-M), sanity H-S.
- Co-primaries: **T60% (scene-mean over the 10 room families — release grouping)** + **RIR_to_GT_RIR_R@1 (split-level)**; Holm-2 per hypothesis; paired-t df=4, 95% CI.
- Aggregation routing: acoustic family (T60/C50/EDT) = scene-mean from `by_scene`; retrieval + FD = split-level flat. `RIR_to_geom_R@k` quarantined (rotated-gallery confound), descriptive only.
- Gates before any H-readout: G1 in-group floor (≤0.5·σ̂_Z), G2 VANL@90 positive control (≥5·σ̂_T60), G3 golden seed-42 assignment, G4 stream-hash equalities. G5 exp_11 reproduction = check only.

## Launch sequence (resequenced 2026-08-11T22:24 per full review — G1/G2 need Z data and gate the Z→R transition, not the V rung)

1. Campaign pin = reviewed, remotely-reachable post-FX SHA (filled at submission: `____`), file `yaw_gen_campaign_pin`, committed + pushed before any job. Full pre-flight = the review's launch-readiness checklist (adopted verbatim, `yaw_gen_codex_code_full_review.md`).
2. **Rung 1:** C4L@90 alone → SCREENRESULT + valid metrics/stream/screenmeta + timing ≤ 2× the 23-min conf baseline (doubles as the per-scene compute-cost probe).
3. **Rung 2:** `WAVE=vctl` (C4L@90 deduped) → all six V cells individually VALID.
4. **Rung 3:** C32 rgen s42/K8 probe → VALID + **G3 PASS** + timing.
5. **Rung 4:** `WAVE=zref` → all 50 Z cells → **G1/G2 PASS + available G4 PASS**; VANL table regen fires here.
6. **Rung 5:** `WAVE=rgen` (probe deduped) → all cells → **G1–G4 PASS** before any H-readout.
   G1/G2 PENDING at rungs 1–3 is expected. Yixun pre-approval (21:39): each wave proceeds automatically as the prior rung's criteria pass; any FAIL/INVALID → halt + triage, no launch.

## Cost estimate

Measured cell times 13–36 min (K/arm-dependent) ⇒ ~30–45 GPU-h total; queue-excluded service ~4–6 h at ≤16 concurrent.
