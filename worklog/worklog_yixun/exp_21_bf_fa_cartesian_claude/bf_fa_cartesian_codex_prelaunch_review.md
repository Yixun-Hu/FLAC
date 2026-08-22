**Reviewer:** OpenAI Codex (gpt-5.6-sol, `codex exec`, read-only sandbox, reasoning=xhigh) · **Date:** 2026-08-22 (consolidated pre-launch review)

## Verdict: BLOCKING

### BLOCKING findings

1. **The rung-4 probe does not test the registered BF16 conditioning path and is not fail-closed against training-config drift.**

   [bfc_cap_parity_probe.py:130](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bfc_cap_parity_probe.py:130) runs under `no_grad` without autocast, while the registered evaluation explicitly uses BF16 conditioning at [exp21_protocol.py:257](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/exp21_protocol.py:257). The probe therefore establishes FP32 eval-mode parity, not parity under the protocol that will produce the results.

   It also merely prints the configured method/cap at [bfc_cap_parity_probe.py:156](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bfc_cap_parity_probe.py:156), then invokes hard-coded `DEFAULT_FRAME_ANGLES`, cap 32, and cap 64 at [bfc_cap_parity_probe.py:133](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bfc_cap_parity_probe.py:133). A drifted training method, angle list, or cap can still print `RUNG-4 VERDICT: PASS`.

   `conditioner.eval()` at [bfc_cap_parity_probe.py:174](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bfc_cap_parity_probe.py:174) is appropriate for the intended cap-equivalence measurement, but it deliberately cannot prove training-mode equivalence: train-mode RoPE draws, BF16 mixed precision, gradients, and activation checkpointing are absent. Add exact config assertions and run the cap/C4 comparisons under the registered BF16 autocast and matmul settings. Do not reinterpret the result as train-mode allclose.

2. **The required ≥200-step co-tenant rate probe is still absent.**

   The approved plan requires a window over at least 200 real steps at [plan_bf_fa_cartesian.md:121](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/plan_bf_fa_cartesian.md:121), and the integrative review repeats that requirement at [bf_fa_cartesian_codex_code_full_review.md:94](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bf_fa_cartesian_codex_code_full_review.md:94). The only rate evidence is the 25-step smoke, which the worklog itself correctly calls “NOT steady-state evidence” at [bf_fa_cartesian_worklog.md:215](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bf_fa_cartesian_worklog.md:215). Consequently, the pre-launch evidence package is incomplete.

### NIT findings

- [bfc_cap_parity_probe.py:14](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bfc_cap_parity_probe.py:14) incorrectly calls batch 32 the smallest batch where the effective cap plans differ. With three nonzero C4 angles, batch 11 already yields cap-32 chunks `{2,1}` versus cap-64 `{3}`.

- The readback is not fully fail-closed: [bfc_cap_parity_probe.py:117](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bfc_cap_parity_probe.py:117) checks only `context_audio.shape[0]`; its full `(8,1,9600)` shape, dtype, and finiteness are merely printed. [bfc_cap_parity_probe.py:142](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bfc_cap_parity_probe.py:142) compares output tensors but not masks. Scene breadth is also printed rather than asserted.

- The dormant fallback at [bfc_cap_parity_probe.py:188](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bfc_cap_parity_probe.py:188) can label differences up to `1e-3` PASS without demonstrating that such an error is harmless. The committed run did not exercise this fallback.

- [bf_fa_cartesian_params_set_up.md:10](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bf_fa_cartesian_params_set_up.md:10) incorrectly describes a “one-sample train tail.” The smoke loaded 291,210 records; batch 32 leaves 10 globally, or 5 per rank under the two-rank split. `drop_last=True` still correctly avoids it.

- The command-manifest timestamps at [bf_fa_cartesian_command.md:5](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bf_fa_cartesian_command.md:5) and [bf_fa_cartesian_command.md:11](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bf_fa_cartesian_command.md:11) say approximately 09:45/10:00 EDT, contradicting the 03:50/03:53 logs and the timestamp erratum at [bf_fa_cartesian_worklog.md:217](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bf_fa_cartesian_worklog.md:217).

- The command manifest delegates evaluation to the driver at [bf_fa_cartesian_command.md:23](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bf_fa_cartesian_command.md:23) without explicitly recording the critical conditioning flags or effective train/full-eval/tail chunk plans, contrary to announcements 05 and 06. Also, “NOT YET LAUNCHED (added at launch time)” is self-contradictory because those commands were pre-entered.

- The last displayed smoke `train/lr` is `1.11e-5` at [bf_fa_cartesian_2026-08-22_03-50-03_smoke.log:147](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bf_fa_cartesian_2026-08-22_03-50-03_smoke.log:147), not the `1.07e-5` endpoint recorded in the worklog.

### Confirmed evidence

- The probe uses the genuine training dataset configuration and the same `create_dataloader_from_config` entry point as `train.py`. The committed batch was seeded, shuffled, and spanned 9 of 10 training families.
- Pose tensors are correctly pinned to float32 by `AR_md.py`; the actual stored depth read back as float64 because the NumPy array is not downcast there. The geometry conditioner later converts depth and coordinates to float32 before DINO.
- The committed rung-4 log passed the strict `1e-5` threshold: cap parity worst `1.058e-6`, C4 worst `1.319e-6`.
- The smoke log supports 25 completed steps, two-rank DDP, BF16 mixed precision, rc=0, no OOM/NaN, and no `.ckpt` files. The post-exit NCCL/TCPStore warnings are nonfatal shutdown warnings and do not contradict rc=0.