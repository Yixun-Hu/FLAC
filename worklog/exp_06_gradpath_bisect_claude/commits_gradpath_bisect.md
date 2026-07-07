# Commits — exp_06_gradpath_bisect

Base: `803c851` (end of exp_05).

```
9ae0e1b exp_06: scaffold — gradient-path bisection (query incl. lr hypothesis, notebook)
62fcc53 exp_06: plan — S1 dynamics (free), S2 lr axis (5 arms incl. schedule-faithful), S3 lineage audit
5dfe539 exp_06: plan review (APPROVE-WITH-CHANGES x6) + revision — awaiting approval
f15b109 exp_06: approved; S1 + S3.1 launch commands
d6bce3c exp_06: S3.1 — code lineage ELIMINATED (upstream diff trivial); suspects narrow to lr + augmentation
ed34c6c exp_06: tests for --lr-schedule (RED)
7a6272c exp_06: --lr-schedule inverse-restart (GREEN)
2b1ac6a exp_06: lrsched review (REQUEST-CHANGES x2: warmup guard, restart pin)
fe06947 exp_06: warmup/inverse-restart guard + restart-semantics pin (review fix)
197c49a exp_06: lrsched round closed (fix fe06947 re-verified); L5 unblocked
99fb722 exp_06: S3.2 — truncation dead; augmentation partial-EDT only; T60 narrows to lr axis or data lineage
bdfd1c5 exp_06: S1 readout — fast convergence to a T60-worse optimum; checkpoint-selection hypothesis registered; L-arm commands
9b01b94 SOP: rename to experiment_SOP.md; add HTML results-visualization component (artifact 13)
3d3a586 exp_06: L1 screen — mid-descent, no recovery at low lr
ff17ed4 backfill: HTML results pages for exp_01..exp_05 (SOP artifact 13)
e7b006f exp_02: panorama + RIR rotation visualizations; SOP: universal review coverage rule
1cafd02 exp_02: RIR figure rebuilt around dB envelopes (clear P0/P180 separation); captions synced
e8217b9 review fixes: exp_05 page 3-repeat mean, s3_probes train-split-only rerun, exp_03 source line; consolidated review saved
499553e exp_03: method note — group averaging for hard yaw-invariant conditioning
f04e6d4 exp_06: notebook — remote reconciled via force-with-lease; env stays rir2rir per Yixun
e87fbc0 exp_06: S2 verdict + results + params — lr monotone-worse; registered stop
```

*(this file's own commit: child of the last SHA above)*
