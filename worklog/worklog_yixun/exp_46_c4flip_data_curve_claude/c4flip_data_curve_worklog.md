# exp_46 c4flip_data_curve — worklog

## 2026-11-17 12:30 EST — experiment opened; plan r0 (coordinating session)

- Origin: Yixun's queries 6–8 (`c4flip_data_curve_yixun_query.md`); C1–C5 approved 11:30 EST.
- Branch `exp-46-c4flip-data-curve` cut from `exp-45-rf-data-curve` at `90d0202f`; worktree
  `/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve`. exp_45's Coder is fixing its code-round-3 findings on the exp_45
  branch; the exp_46 records commits are rebased onto that fixed tip (`X45f`) before L0.
- Facts G1–G9 checked on neuronic (plan §0): the training path refuses flip + offset_step (`factory.py` 97–101,
  `diffusion.py` 305–308); the draw order already composes (offsets → grid snap → rotate → flip bits from the same
  generator); the banner composes; 4 tests assert the refusal; no C4 + flip config or model exists; 232 GiB free.
- Plan r0 written (`plan_c4flip_data_curve.md`, a delta to exp_45 plan r3). Next: Codex plan review round 1.
