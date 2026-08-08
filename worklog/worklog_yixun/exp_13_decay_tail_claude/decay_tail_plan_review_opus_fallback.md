# Plan review — exp_13 decay_tail (Reviewer seat: Claude Opus 5 max effort, SOP fallback — Codex 401-down; swap declared)
# Verdict: REQUEST-CHANGES — do not launch. (2026-08-08)

B1 BLOCKER (fatal, silent): the scheduler swap is a NO-OP on warm resume. InverseLR stores inv_gamma/power/warmup/final_lr as instance attributes; torch LRScheduler.state_dict() serializes them; load_state_dict (= __dict__.update) CLOBBERS the new config's values with the checkpoint's old ones (measured in env flac: post-restore inv_gamma reverts to 1e6/0.5/0.99; lr after 10k steps 4.7727e-5, not 1.28e-5). The prior "counters only" reading of load_state_dict was one-directional. Required: experiment-local src/tools/retune_lr_state.py producing a COPY of the anchor with lr_schedulers[0].{inv_gamma:30000, power:1.0} and optimizer_states[0].param_groups[0].lr = 1.2765957446808513e-5 (else step 87,501 runs at 4.79e-5); TDD red test = hyperparameter-clobber reproduction; probe + launcher must read the LIVE optimizer lr post-restore (±1e-12) and print sched.inv_gamma/power — analytic assertion is exactly what cannot see this bug. Plan must state the treatment is delivered by checkpoint rewrite, not config.

B2 BLOCKER: DT1 as drafted fires 66% under the null (joint rate 1/8 in the untreated band; 1-(7/8)^8=0.656) and its thresholds are anchor-DRAW-matching (T60≤8.40 passes 1/8 band points), not band-typical. Required: qualifier-RATE test — exact binomial vs p0=1/8, one-sided; ≥4/8 tail qualifiers for p<0.05 — and relabel; note DT1 scores single-seed s42 screens while the reference is 5-seed.

B3 BLOCKER: DT3 doubly biased toward "shrink" (window length: P1 SD varies 0.244-0.373 by window choice alone; cadence: denser sampling lowers SD under autocorrelation); comparator step set unpinned (no 82.5k screen exists); ddof/metrics unpinned. Required: DT3-primary = tail subsampled to {90000,92500,95000,97500} vs P1's SAME steps (matched window/cadence/lineage; T60 SD 0.244, EDT SD 0.839, ddof=1), report SD ratio + CI (n=4, wide — say so); 1250-grid full-tail SD as supplementary, flagged biased. P1's 87.5k→100k segment is a free matched control arm — use it for DT1's null rate too (4 untreated draws).

B4 MAJOR: (1) assert_arm_configs.py is a no-op for exp_13 and prints the OPPOSITE scheduler assertion into the launch log — keep for ViT pin, add an exp_13 assert building the dtail config and checking configure_optimizers() returns InverseLR(30000,1.0,warmup). (2) RESTART mode: exp_13's own ckpts embed the dtail config — INITIAL asserts embedded==BVp1, RESTART asserts embedded==dtail (else every crash-restart aborts). (3) Add the POSITIVE byte-exact scheduler-block assert on the dtail config (the negative equal-except-scheduler check passes for any scheduler). (4) Live post-restore lr gate (see B1). Also add a df floor check (volume at 95%).

B5 MAJOR: `final_lr_ratio` does not exist (TypeError crash risk at configure_optimizers via **kwargs); warmup retention unspecified. Write the scheduler block verbatim: {"type":"InverseLR","config":{"inv_gamma":30000,"power":1.0,"warmup":0.99}}.

B6 MINOR: lr(97,500)=1.1764706e-5 (1.15e-5 is 100k, never reached); exact boundary lr 1.2765957e-5 = 0.266× (not "quarter" precisely).

B7 MINOR: budget omits ~2.4h of evals (8 screens + 10-confirm at ~8 min each) — training ETA and results-complete ETA must be stated separately; results land past the 10h approval window (disclose).

B8 MINOR: pin the screen protocol tuple: EMA weights, K=8, s42, cfg 1.0, steps 1, full 6,337/17 split, per-scene mean (the P1 comparator's protocol).

Verified correct: EMA 63.2% turnover at 10k (update_every=1 checked); step arithmetic; anchor provenance bd3fc7db… independently attested; storage 5.8GB vs 201GB free; contract triangle passes (but see B4.3).
