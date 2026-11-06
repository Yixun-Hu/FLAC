# exp_35 OrbitRIR yaw evidence — phase (i) report to Yixun, r3 (2026-11-06)

Written by the Planner (Claude Opus 5.5) at phase (i)'s end and sent to Yixun in the session; the phase-(ii) launch kit binds the
six fields at the bottom (plan §15). r3 SUPERSEDES r2 (`orbitrir_yaw_evidence_phase_i_report_to_yixun_r2.md`, sha256 `868ed39a553125ef2376780f487e864c6fc7ff34a835817a0d272b18ecba7d5b`; its section 5 and ETA field assumed GPU 0 free at the s43 training's end) and r2 superseded r1 (`orbitrir_yaw_evidence_phase_i_report_to_yixun.md`, sha256 `d497cd0237ba923cbc0510957a45304797de0eadb0be865cff045608cdb92bb7`): sections 1-3 are r1's unchanged; section 4 (the phase-(ii) ETA) is re-derived with every number labelled (Codex P13 HIGH 1), and the ETA field changes accordingly.

1. **Phase (i) complete** at 11:12:33 EST: 33 / 33 chunks verified (CTL x2, B-F T x10, R x10, T-dup x2, C4 x8, the
   probe), no stop latch; 20:20 EST Nov 5 -> 11:12 EST Nov 6, every cell shared on GPU 1 with the s43 training;
   the supervisor released its GPU 1 reservation and exited.
2. **The s43 training's slowdown** (the read-only guard, 9039 steps since our first spawn): mean 5.886 s/step vs
   4.260 unshared (x1.38); worst 500-step window 6.058 (limit 9.712), worst 100-step 6.70 (limit 9.950);
   about 4.1 h of training time lost (the s43 table job's own cost was about 4.9 h). s43 at step 36980, now
   unshared: it ends about 14:45-15:00 EST.
3. **Verdicts** (the frozen collector, --exp35 --partial; only the phase-(ii) cells are missing, Tier 2-C4 and Tier 3 are complete):
   E1 equal at K1 and K8; E2 PASS 14 / 14 (45 degrees from the A6 CSV); Tier 2-C4 PASS (OrbitRIR's invariance claim: every C4
   pair within 3x its control and 0.01, pooled p99 0.00619 vs 0.00571; T-dup exact; no alarm); Tier 3 B-F FAIL as
   pre-registered (pass_c4 holds on all 14 checks, pass_r fails on all 14: B-F is C4-invariant only). T60 K1 / K8 at theta 0:
   OrbitRIR 9.543 / 8.202, CANON 9.788 / 8.418, FLAC 10.287 / 8.993; under random yaw: OrbitRIR 10.263 / 8.883, CANON 9.788 / 8.419,
   FLAC 11.033 / 9.741.
4. **Phase (ii)**: 68 cells (P1 and OrbitRIR at 135 / 225 degrees: 10 K1 + 10 K8 each; CANON at 135 / 225 degrees and at 45 degrees s43-46:
   14 K1 + 14 K8), free cards only, through the reviewed kit. The per-cell times, from the pinned table
   `orbitrir_yaw_evidence_2026-11-06_05-35-38_cell_times.txt` (sha256 `92ceaf66...`; group MEDIANS, minutes, K1 / K8; inputs: the two
   exp_25 runner logs at git HEAD and the phase-(i) snapshot, all digest-pinned in the notebook):
   - OBSERVED on a free card: exp_25 YAW eval (vanilla, with dumps; n 4 / 5) 11.9 / 12.6;
     exp_25 CANON TRAJ (no dumps; n 35 / 34) 13.5 / 14.8 (one free-card YAW K8 cell took 20.5).
   - OBSERVED shared: exp_25 YAW eval beside an xRIR job 17.5 / 18.4; phase-(i) P1 beside the s43
     training 20.2 / 25.2; phase-(i) OrbitRIR beside it 25.2 / 30.3.
   - PROXY (no free-card OrbitRIR or P1 cell exists): P1 = the free-card YAW medians; CANON = the CANON TRAJ medians + 0 to 1 min for
     its dumps (an ASSUMPTION); OrbitRIR = its shared medians divided by a sharing ratio, either YAW shared / YAW alone or P1(s43) /
     YAW alone: 14.8-17.1 (K1) and 15.2-20.7 (K8).
   - FORECAST (Planner arithmetic): the 68 cells at those values plus 0.5-1 min per cell for staging and checks (an ASSUMPTION) =
     16.3-18.6 h on one free card, i.e. about 16-19 h (center about 17.5 h); about half on two cards once the second joins.
5. **Card plan** (FORECASTS, revised at 12:54 EST): since 11:38 EST Yixun's eqtail s43 training (`~/codespace/flac-fork-attn-eqtail`,
   resumed from s43's step-35,000 checkpoint to 40,000) runs on BOTH cards beside the s43 training (the read-only observation
   `orbitrir_yaw_evidence_2026-11-06_12-54-18_card_observation.txt`). The s43 training now runs at about 7.1 s/step and ends about 17:10;
   the eqtail training (about 8.7 s/step while shared; 3,668 of epoch 7 at 12:54) then runs on, beside the s43 table job on GPU 1, and
   ends about 20:15-21:45; its script then evaluates on GPU 1. GPU 0 therefore frees about 20:15-21:45; GPU 1 only after the s43 table
   job and the eqtail evaluation. (A, the default) one lane on GPU 0 when it frees: start about 20:30-22:00, done about 13:00-17:00 EST
   Nov 7 (16-19 h on one card, under a day); GPU 1 goes to the exp_22 pre-flight when it frees. (B) a second lane would have to be in the
   single launch and GPU 1 is busy far longer: not useful now. The kit refuses `reserve` while any compute process is on GPU 0.

```
exp35-phase-ii h35 5b065801e9d1c1844461ec84cd0558702c714326
exp35-phase-ii phase-i-state-sha256 7b6cfc8cc8f2108e1a26d8fd3e7e7bccabb39988c5317925becb3dc0d9bca705
exp35-phase-ii lanes 0
exp35-phase-ii second-lane-decision none
exp35-phase-ii eta about 16-19 h on one free card (Planner forecast from proxies, center about 17.5 h); start about 20:30-22:00 EST Nov 6 when GPU 0 frees (his eqtail s43 training ends); done about 13:00-17:00 EST Nov 7
exp35-phase-ii prior-runs none
```
