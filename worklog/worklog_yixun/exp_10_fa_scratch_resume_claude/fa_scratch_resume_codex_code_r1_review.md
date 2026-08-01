Verdict: **SHIP — no blocker to the 15-step probe or subsequent launch.**

- [bf_resume_launch.sh](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/bf_resume_launch.sh:90) — **SHIP**
  - Exact canonical anchor path: lines 90–107.
  - Restart namespace and `MAXSTEPS > EXPECTED_STEP`: lines 108–120.
  - Runtime SHA-256 pin: lines 155–168.
  - Exact embedded-config equality, warm optimizer state, scheduler, and EMA: lines 175–223.
  - Exp_09 reset/selection/strip machinery is functionally absent; remaining mentions are explanatory comments only.
  - Contract, environment, VRAM, W&B, DINO pin, manifest, and training arguments retain reference behavior: lines 68–76, 133–153, 226–272.

- [bf_resume_launch_guardtests.sh](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/bf_resume_launch_guardtests.sh:50) — **SHIP**
  - Pin swap is acceptable for this serialized pre-launch exercise: backup/verified restoration at lines 50–79, immediate restoration at 166–176, and final anchor-integrity check at 241–252.
  - Do **not** add an unrestricted `PIN_FILE` override; that would let production launches replace the root of trust. A test-only override would need an enforced no-training mode.
  - Committed evidence reports 26/26, including valid INITIAL and RESTART paths.

- [bf40k_anchor.sha256](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/bf40k_anchor.sha256:1) — **SHIP**
  - Digest exactly matches the current 40k anchor: `5319feb4…42328`.

`bash -n` and `git diff --check` pass; worktree is clean; `outputs_FLAC/exp10_BF` is absent. No files modified.