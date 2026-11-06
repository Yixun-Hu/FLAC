# exp_35 OrbitRIR yaw evidence — phase (i) report to Yixun (2026-11-06)

Written by the Planner (Claude Opus 5.5) at phase (i)'s end and sent to Yixun in the session; the phase-(ii) launch kit binds the
six fields at the bottom (plan §15).

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
4. **Phase (ii)**: 68 cells (P1, OrbitRIR and CANON at 135 / 225 degrees, CANON at 45 degrees s43-46), free cards only, through the
   reviewed kit. Per cell about 12-15 min (P1, CANON) and 16-20 min (OrbitRIR) on a free card (measured: exp_25 cells alone
   11.9-14.8 min; phase-(i) cells shared 20-30 min); one card about 18-19 h (range 17-24 h), two cards about 10 h.
5. **Card plan**: (A, the default) one lane on GPU 0 from about 15:00 EST, done about 09:00-10:00 EST Nov 7; GPU 1 left to the
   exp_22 pre-flight after the s43 table job. (B) both cards (GPU 1 joins after the table job): done about 02:00-04:00 EST Nov 7;
   exp_22 waits until then. Under a day on one card: (A) runs when s43 ends unless Yixun decides otherwise.

```
exp35-phase-ii h35 5b065801e9d1c1844461ec84cd0558702c714326
exp35-phase-ii phase-i-state-sha256 7b6cfc8cc8f2108e1a26d8fd3e7e7bccabb39988c5317925becb3dc0d9bca705
exp35-phase-ii lanes 0
exp35-phase-ii second-lane-decision none
exp35-phase-ii eta 18-19 h on one free card (range 17-24 h); start about 15:00 EST Nov 6 after s43 ends; done about 09:00-10:00 EST Nov 7
exp35-phase-ii prior-runs none
```
