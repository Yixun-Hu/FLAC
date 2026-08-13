**Reviewer:** OpenAI Codex `gpt-5.6-sol`, reasoning effort xhigh, codex-cli, read-only sandbox · seat per SOP §Roles · 2026-08-12**

## Verdict: REQUEST-CHANGES

Four prior findings are substantively resolved. Two launch-safety issues remain blocking.

1. **RESOLVED — cap arithmetic and implicated topology.** C4 chunks only the three non-identity angles ([yaw_rotation.py:491](/home/yixunhu/codespace/FLAC/src/data/yaw_rotation.py:491)); line 501 gives:

   - micro-32/cap-32: `32 // 32 = 1` → three separate chunks.
   - micro-32/cap-96: `96 // 32 = 3` → one chunk containing all three.
   - exp_11 micro-8/cap-64: `64 // 8 = 8`, truncated by three remaining angles → one 3-angle chunk. Exp_11’s micro-8 rung is pinned at [fa_orbit_worklog.md:119](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_worklog.md:119), with default cap 64 at [yaw_rotation.py:45](/home/yixunhu/codespace/FLAC/src/data/yaw_rotation.py:45).

   Thus Rev 2 reaches the same **3/3 non-identity draw-sharing topology**, but [plan line 21](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_14_fa_drawshare_claude/plan_fa_drawshare.md:21) must not say it “matches the configuration exp_11 actually ran”: micro-32/cap-96 has a 96-row GEMM versus exp_11’s micro-8/cap-64 producing a 24-row GEMM. Likewise, equal sample rows do not mean equal walltime or memory.

2. **RESOLVED — intervention identity.** The claim is correctly limited to the operational chunk policy, including forward count, GEMM shape, and RNG displacement—not isolated RoPE covariance ([lines 10–12](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_14_fa_drawshare_claude/plan_fa_drawshare.md:10)).

3. **RESOLVED — one-seed inference and null language.** One training seed and the role of evaluation seeds are explicit ([lines 8–12](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_14_fa_drawshare_claude/plan_fa_drawshare.md:8)); DS1 repeats the caveat ([line 35](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_14_fa_drawshare_claude/plan_fa_drawshare.md:35)); `NO-EFFECT-OBSERVED` is explicitly not equivalence ([line 40](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_14_fa_drawshare_claude/plan_fa_drawshare.md:40)). This is honest for the seed-42 trajectory.

4. **REMAINING — config propagation/resume/evaluation contract. BLOCKING.** The proposed API can preserve default behavior, but the plan is incomplete:

   - The actual config handoff is in [factory.py:152](/home/yixunhu/codespace/FLAC/src/training/factory.py:152), yet `factory.py` is absent from the artifact list at [plan line 48](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_14_fa_drawshare_claude/plan_fa_drawshare.md:48).
   - [train.py:21](/home/yixunhu/codespace/FLAC/train.py:21) only writes `model_config`. On resume, the wrapper is reconstructed from the current JSON at [train.py:160](/home/yixunhu/codespace/FLAC/train.py:160) before PL loads `ckpt_path` at [train.py:230](/home/yixunhu/codespace/FLAC/train.py:230). Therefore “survives resume” at [plan line 27](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_14_fa_drawshare_claude/plan_fa_drawshare.md:27) is false without a fail-closed embedded-config/current-config comparison.
   - Evaluation currently calls `invariant_conditioning` without a cap at [eval_FLAC.py:1005](/home/yixunhu/codespace/FLAC/eval_FLAC.py:1005) and records the module default at [eval_FLAC.py:590](/home/yixunhu/codespace/FLAC/eval_FLAC.py:590). Rev 2 never explicitly pins both arms to one evaluation cap or distinguishes training cap from evaluation cap.
   - TDD at [line 28](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_14_fa_drawshare_claude/plan_fa_drawshare.md:28) still needs: boolean rejection, factory/direct-wrapper propagation, resume mismatch rejection, explicit common-cap evaluation, and applied-cap provenance. DDP itself is safe if every rank is bound to one immutable config hash; no state synchronization is otherwise required.

5. **RESOLVED — DS2/DS3 interpretation.** DS2 is now a contextual cross-era check with a commit-parity audit ([line 37](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_14_fa_drawshare_claude/plan_fa_drawshare.md:37)); DS3 is contextual ([line 38](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_14_fa_drawshare_claude/plan_fa_drawshare.md:38)). A5 no longer sets outcome sufficiency.

6. **REMAINING — sequencing gate; other protocol pieces resolved. BLOCKING.** The from-scratch launcher is correct ([line 31](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_14_fa_drawshare_claude/plan_fa_drawshare.md:31)), evaluation flags are explicit ([line 23](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_14_fa_drawshare_claude/plan_fa_drawshare.md:23)), the cap-96 fit probe is a hard no-fallback gate ([line 30](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_14_fa_drawshare_claude/plan_fa_drawshare.md:30)), and metric-driven stopping is prohibited ([line 44](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_14_fa_drawshare_claude/plan_fa_drawshare.md:44)).

   However, [line 44](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_14_fa_drawshare_claude/plan_fa_drawshare.md:44) advances directly from DS-PA to DS-CS3. It must require the DS-PA 40k admission/parity audit first and stop before spending the second six days if provenance/config/chunk-policy parity fails. Metric divergence from July alone should remain contextual, not an abort.

The ETD arithmetic is reasonable: `40,000 / 0.079 = 5.86 d/arm`; two arms plus 1.5 days is about 13.2 days, making August 26 plausible if work starts August 12–13. Record steady-state cap-96 throughput during the fit probe and rebase the ETD before launch.

Also correct the false `Result — launched (planning)` entry at [fa_drawshare_worklog.md:7](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_14_fa_drawshare_claude/fa_drawshare_worklog.md:7), and replace the shared-checkout “announce before pushing” language at [plan line 29](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_14_fa_drawshare_claude/plan_fa_drawshare.md:29) with a safe-boundary or isolated pinned-worktree requirement.

Read-only review; no files, environment, or jobs were changed.