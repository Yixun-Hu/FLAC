# exp_22 — ORBITRIR: clean commit-by-commit port of the frame-averaging (B-F) method — Yixun's queries

## Query 1 — 2026-08-29 (session: Fable 5, A6000 box)

**Verbatim:**
> Could you please create a new codespace under ~/codespace/ORBITRIR and first put the master branch code inside https://github.com/AmandineBtto/FLAC into it? And then let me know when you done. After this, I want you to add on the ORBITRIR codespace commit by commit with frame averaging method (exp_03 and exp_07). The intention for this is to clean the code for the exp_07 BF method.

**Summary:** (1) Clone upstream `AmandineBtto/FLAC` `master` into `~/codespace/ORBITRIR`. (2) On top of that pristine tree, re-add — commit by commit — the frame-averaging method as it was built in exp_03 (fa_invariant conditioning) and run in exp_07 (B-F from-scratch training), producing a clean codebase for the B-F method without the rest of the fork's experiment scaffolding.

**Yixun's assumption / intent:** the B-F method is separable from the other experiments' code (CylViT, warmup, freeze-BN, lr-schedule, fine-tune driver, batched orbit, yaw-aug, ARE, fa_cartesian…) and can be rebuilt as a small linear commit history on upstream master.

**Why it needs to run:** the FLAC fork's branch carries ~20 experiments' worth of code; a reader (or a release) of the B-F method needs only the ~10 code deltas that define it. A clean port also re-verifies which lines are load-bearing (the tests come along).

**Context question that preceded it (same session):** "what code had I modified to run exp_07 BF?" — answered from `exp_07_fa_scratch_claude/commits_fa_scratch.md` + git: no new FA code was written in exp_07; B-F = exp_03's fa_invariant conditioning + exp_07's `--max-steps`, `--sync-batchnorm`, ViT gradient checkpointing, and the `FLAC_AR_BF.json` config.

## Query 2 — 2026-08-29 ~14:50 EDT — plan approval

**Verbatim:** "Go with the recommendations, and first do C1, after my approval, do C2"

**Summary:** Plan Rev 3 approved with every recommendation (D1 include C1 full 19-file scope; D3 `FLAC_AR_FA.json` + strip exp refs; D4 bookkeeping here; D5 exclude fine-tune; D6 as-run loop; D7 four review rounds; D8 trim guard tests; D9 drop `yaw_transform_consistency`; D10 pin `revision` in both configs; D11 shared dispatcher; D12 two-cell acceptance). **Gate added by Yixun:** C1 is done first and presented; C2 starts only after his explicit approval of C1.
