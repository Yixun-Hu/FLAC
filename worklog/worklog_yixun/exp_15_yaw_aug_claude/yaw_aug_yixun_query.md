# exp_15 yaw_aug — Yixun's driving queries

## Query 1 — 2026-08-10 (neuronic cluster session)

**Verbatim:**

> Read the @worklog/experiment_SOP.md and CLAUDE.md for rule loading. Run a new experiment that trains vanilla FLAC (not fa_invariant) on the standard 291k AR train split with a random horizontal yaw scene shift applied to every training sample, and compare it against a matched no-yaw-aug vanilla control on the full Table-1 eval — write the SOP plan and wait for my approval before coding or launching.

**Mid-turn follow-up (verbatim):**

> If you have any questions/designs need me to verify that you are align with my intention, please tell me

**Summary:** Train a vanilla-conditioned FLAC model (NOT `fa_invariant`) on the standard AR train split (`data/AR/train.json` — 291,210 items / 243 rooms; the same split every arm in this program uses) with a **random horizontal yaw scene shift applied to every training sample** (data augmentation), and compare it against a matched vanilla control trained without the augmentation, on the full Table-1 evaluation protocol (full unseen split, per announcement 01). Plan first; no code or launches before approval.

**Design decisions confirmed via AskUserQuestion (2026-08-10 23:30 EDT):**

1. **Control arm = reuse exp_11 VANL@40k** (`outputs_FLAC/exp11_VANL`, job 3661520, 8×L40 × micro-batch 8 = eff. 64, SyncBN, training seed 42, 40k steps, sha-pinned in exp_11's committed `arm_launch_registry.json`, `epoch=8-step=40000.ckpt` verified on disk). No fresh control run.
2. **Training recipe/budget = match exp_11's rung**: 8 GPU × micro 8 (eff. 64 = BN batch), 40k steps, training seed 42, grad-ckpt ON, this cluster — keeping the whole arm family (VANL, C4L, C8, C16, C32, YAWAUG) mutually comparable at matched steps and seed.
3. **Eval scope = Table-1 (θ=0) headline + random-yaw robustness secondary**: full unseen split (6,337 items / 17 rooms), K∈{1,8}, 5 eval seeds, EMA, per-scene mean, `--cond-method vanilla` for both arms (announcement 05), PLUS per-sample random-yaw cells for both arms (exp_14's protocol) to measure the robustness the augmentation exists to buy.
4. **Augmentation policy = fresh draw per visit, uniform over all 512 columns**: every time a training sample is loaded, draw a new uniform column offset d ∈ {0…511} from a dedicated generator (global RNG untouched); physically-consistent rotation (equirectangular depth-panorama roll + all pose fields rotated together by the same yaw; GT RIR and context audio unchanged; angle quantized to exact 360°/512 columns — interpolation-free, matching exp_02/exp_14 conventions).

**User's assumption/hypothesis:** Random-yaw training augmentation is the classical data-side alternative to architectural equivariance (the FA arms). The comparison against a matched no-aug vanilla control tests (a) what the augmentation costs or buys on the clean Table-1 protocol, and (b — confirmed secondary readout) whether it buys yaw robustness.

**Why the experiment needs to run:** exp_02 proved vanilla FLAC is not yaw-invariant; exp_11 trained the C4–C32 FA-orbit arms and found headline θ=0 metrics *degrade* with orbit size at matched steps; exp_14 (planned) measures robustness of those trained arms. The missing arm of the augmentation-vs-architecture question is a model trained with yaw *augmentation* and no architectural invariance — the standard baseline any equivariance claim must beat. No such arm exists in the program.

## Query 2 — 2026-08-11 (launch authorization)

**Verbatim:**

> Proceed once smoke is clean — no extra gate needed

**Summary:** The 40k training launch is authorized to proceed autonomously as soon as the validation ladder (integrative full review + SMOKE rung on 8×L40) passes — no additional human approval gate before the real submission. Pre-launch report still posted for the record (params/command/acceptance criteria per SOP), but it is informational, not blocking.

## Query 3 — 2026-08-13 (chunked-training directive)

**Verbatim:**

> Right now I need you to do this: Currently the computing resources are really limited, so our job is hard to backfilled. The strategy I need you to help us training is to write a watchdog and split exp_15 to multiple jobs (for example, one job only train 2500 steps and then closed and next job resume from the previous 2500 steps checkpoints and trained based on that), please verify that current code support checkpoint resume (if not, you should write code support). Using this I think we don't need to wait for a such a long time.

**Summary:** Replace the single 24 h job with a chain of short jobs (~2500 steps each) that resume from the previous leg's checkpoint, supervised by a watchdog, so each leg backfills into small scheduling gaps. Verify (or build) checkpoint-resume support.

**Verification result (2026-08-13):** resume IS supported — the kit's RESTART mode (`--resume/--expected-step`, exp_11 Q10 lineage, proven on real 40k→100k legs), restart preflight, 40k cap, F6 registry leg-provenance, and the r1 counter-based yaw draws are resume-exact by design (tested). Missing and to be built: per-leg step budgets, short per-leg time pins, self-chaining submission. Design decision (login-node rules forbid persistent daemons): the "watchdog" = each leg's epilogue submits the next leg after its completion audit passes (cluster-native, session-independent), plus the session Monitor as supervisor/alerter with manual resubmission as the recovery path.
