# exp_14 yaw_gen — Yixun's driving queries

## Query 1 — 2026-08-10 (session start, neuronic cluster)

**Verbatim:**

> Please first load the CLAUDE.md and @worklog/experiment_SOP.md into context. And I am asking you to run the experiment of run generalization experiments with yaw-shifted panoramic depth images, comparing vanilla method vs. C4/C8/C16/C32 group orders, expecting higher-order groups to outperform lower-order and vanilla. Please first verify that you are understanding what I want you this experiment to do and then tell me in the CLI about your plan concisely

**Clarifications confirmed via AskUserQuestion (2026-08-10):**

1. Yaw shift = **physically-consistent rotation** of the conditioning (equirectangular depth-panorama roll + source/context pose rotation together, via `--rotate-deg`), ground-truth RIR unchanged — a robustness/generalization test where an invariant model should be unaffected. (Chosen over "panorama-only shift".)
2. Grid scale = **Full: 2 K × 5 seeds** — 5 arms × angle ladder × K∈{1,8} × eval seeds 42–46, announcement-04 table-grade everywhere.

**Summary:** Evaluate how each trained conditioning variant generalizes when the panoramic depth conditioning is yaw-shifted: the vanilla-conditioning model vs. frame-averaged models with group orders C4/C8/C16/C32 (the exp_11 8×8 @40k arms). Measure metric degradation as a function of yaw offset per arm.

**User's assumption/hypothesis:** Higher-order groups outperform lower-order groups and vanilla under yaw shifts — i.e. yaw-degradation shrinks monotonically with group order (vanilla worst < C4 < C8 < C16 < C32).

**Why the experiment needs to run:** exp_02 proved vanilla FLAC is not yaw-invariant, and exp_11 trained the C4/C8/C16/C32 orbit arms but closed with only the θ=0 orbit trend: its yaw-flatness round (R3) never produced a single row — all 17 R3 screens (2026-08-08, jids 3657730–3657746) died on a worktree-provisioning infra failure (missing `weights/FLAC/VAE.ckpt` leaf symlink, fixed by `16b6a20` after the R3 launches). Whether the extra compute of larger orbits buys yaw-generalization — the mechanism they exist for — is therefore still an open, untested question. exp_14 supersedes R3 with a table-grade design.

## Query 2 — 2026-08-10 (mid-turn protocol amendment)

**Verbatim:**

> Use a physically-consistent random yaw/horizontal shift protocol for the generalization check: for each eval sample draw a random panorama-column roll (or random θ quantized to exact 360°/W columns), rotate depth + source/context poses together via the existing --rotate-deg path with GT RIR unchanged, and report robustness across arms—not the fixed C4/C8/C16/C32 angle ladder.

**Summary:** The headline protocol is **per-sample random yaw**, not a fixed-angle ladder: each eval sample independently draws a random panorama-column roll (θ quantized to exact 360°/W = 360/512 = 0.703125° columns, so the roll is interpolation-free), applied as the existing physically-consistent rotation (depth + source/context poses together, GT RIR unchanged). Report robustness (metrics under random yaw, and degradation vs the unrotated reference) across the five arms.

**User's assumption/hypothesis:** unchanged — higher-order groups are expected to be more robust under random yaw than lower-order groups and vanilla.

**Why the amendment:** Expected performance under uniformly-random yaw is the deployment-relevant estimand (a single robustness number per arm), rather than per-angle curves tied to the groups' own special angles. It also collapses the run grid (no angle dimension), at the cost of one small, guarded extension to `eval_FLAC.py` (per-sample random rotation mode — the current `--rotate-deg` applies one fixed angle to the whole run).

## Query 3 — 2026-08-11T21:39 EDT (conditional launch approval)

**Verbatim:**

> Once the ladder passes, launch approve

**Summary:** Launch of the full 106-cell campaign (Z + R waves after the V-cell/probe ladder rungs) is APPROVED, conditional only on the validation ladder passing (V-cell gates G1/G2 + assignment/golden gates + timing probe). No further sign-off needed between ladder pass and full-wave submission. If any ladder gate fails: halt and report, no launch.
