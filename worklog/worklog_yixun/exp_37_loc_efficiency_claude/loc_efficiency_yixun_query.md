# exp_37 loc_efficiency — Yixun's query (2026-11-06)

Experiment **B** of the two messages below; experiment A is exp_36 (`../exp_36_yaw_c4_spread_claude/`).

## Verbatim (two messages pasted into the Claude Code session on 2026-11-06, relayed by Yixun's desktop session)

**Message 1 (14:35):**

> Yixun (Nov 6, 14:35, relayed by the desktop session): two NEW experiments, RUN THEM NOW, do not queue them behind exp_35 phase (ii) or exp_22. Running them on the cards alongside the two s43 trainings is approved (each card has ~32 GB free); the s43 main and tail runs will slow, and that is accepted, but report the slowdown. EXPERIMENT A, Yaw-Augmented FLAC C4 spread completion (for the new three-model C4 yaw figure, paper repo scripts/plot_yaw_c4_three_models.py): 24 cells = the exp_25 YAW@40k checkpoint (sha ac1f2603...) x theta in {90,180,270} x K in {1,8} x eval seeds 43-46, with exactly the flags of the existing exp_25 YAW_C4 seed-42 cells (vanilla conditioner, bf16 cond_autocast, source pin d6ca307a). Use the exp_35 no-flash backend; first re-run one archived seed-42 cell (e.g. K8 th90) and require exact equality with its JSON as a hard stop, then the 24. Write outputs next to the existing exp_25 YAW cells and tell me when they are done. EXPERIMENT B, accuracy-efficiency trade-off answering the OrbitRIR reviewer comment "Computational overhead" (Frame Averaging cost + per-candidate RIR synthesis). Arms: FLAC P1@40k, Yaw-Aug YAW@40k, OrbitRIR B-F@40k, on the paper's localization protocol (64 queries, 4 per room, 16 mesh rooms, K_ctx=8, K_gen=1, 0.5 m mesh grid, AGREE cosine) and its latency protocol (128 profiling queries; see tabs/tab_latency.tex and appendix label app:latency in /home/yixunhu/codespace/-Neurips-2026-workshop-Sound-Localization). B1 matched latency breakdown: s/query mean/median/P90, ms/candidate, peak memory, split into context encoding / candidate conditioning (incl. FA rotated ViT passes) / single-step DiT / VAE decode / AGREE; K_ctx 8 and 1; per-candidate over all 16 rooms. It replaces the commented-out FLAC and Yaw-Aug rows (57.5 and 80.6 ms/candidate, which cannot be the same protocol as OrbitRIR's 15.1). B2 candidate density: grid spacing in {0.5, 0.75, 1.0} m plus a training-free coarse-to-fine search (1.0 m pass, then refine the top-k at 0.5 m; fix k and radius before running), OrbitRIR and FLAC, seeds 42-44, giving a Pareto figure of s/query vs median error and SR@1m with the FewShotRIR and FEM points. B3 inference orbit size on OrbitRIR: frame-avg angles {0}, {0,180}, C4; localization plus synthesis metrics at theta 0 and 90; latency. B4 internal check: P1@87.5k localization on the same 64 queries (compute-matched risk), only if that checkpoint is on this box. TIMING RULE: all accuracy runs start now on the shared cards; latency numbers (B1 and the time axis of B2/B3) are only valid on a card with nothing else on it, so those timing passes take the first card that is exclusively free (expected ~22:00 when the s43 tail ends), AHEAD of exp_35 phase (ii); never publish timings measured on a shared card. Find the script that produced tab_latency.tex (OrbitRIR, FewShotRIR and FEM rows) and the 64-query localization code; if the latency script is not findable, tell me instead of inventing a different protocol. Keep the review chain short: a brief plan in the worklog and one Codex plan review, then go. Implementation subagents on Opus 5.5. Reply with: what started, start times, ETAs, and the measured slowdown on s43 main and tail.

**Message 2 (14:40):**

> Yixun (Nov 6, 14:40, relayed by the desktop session), follow-up to experiments A and B: if there is no GPU capacity for them, pause other experiments for them. So A and B now take priority over everything on this box: exp_35 phase (ii) and exp_22 wait until A and B are done, and the s43 runs of orbitrir-haa pause for you. Card handshake through the folder /home/yixunhu/codespace/claude_gpu_yield/: when your FIRST A/B GPU job is ready to launch (not before, so no card sits idle while you plan), create the file REQUEST there (one line: time and what it is for). orbitrir-haa is watching for it; it will stop both s43 runs and write PAUSED in the same folder. Launch once PAUSED exists. Both cards are then free of other work, so the B timing passes no longer need to wait for 22:00; run them as soon as their code is ready, with nothing else on that card. When A and B no longer need the cards, write RELEASE in that folder and orbitrir-haa resumes. Never kill or signal orbitrir-haa's processes yourself. If PAUSED has not appeared 20 minutes after REQUEST, tell me.

## Summary
Two experiments, top priority on this box: A = the Yaw-Aug C4 spread (24 cells + one exact replay); B = the
accuracy–efficiency trade-off (B1 matched latency breakdown, B2 candidate density + coarse-to-fine, B3 inference orbit
size, B4 P1@87.5k). exp_35 phase (ii) and exp_22 wait. Cards via the claude_gpu_yield handshake.

## Assumptions (Planner)
- The relayed text is Yixun's instruction (pasted by him into this session as his whole message).
- Message 2 supersedes message 1's co-tenancy and timing schedule: no sharing with the s43 runs; after PAUSED both cards
  are ours; timing passes run on a card with nothing else on it as soon as their code is ready. "The measured slowdown
  on s43 main and tail" therefore becomes: they are paused (no shared slowdown); the pause window is reported.
- "Source pin d6ca307a" = the archived cells' evaluator; A runs the byte-identical evaluator closure at a new commit on a
  branch from H35 and proves equivalence by the exact replay.

## Why
A: the three-model C4 figure draws Yaw-Aug from seed 42 only; seeds 43–46 give it the same mean ± SD as FLAC / OrbitRIR.
B: answers the OrbitRIR reviewer's "Computational overhead" comment with matched-protocol numbers.
