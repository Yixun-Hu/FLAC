# exp_22 Codex review tail — rounds r9x → r9z7 (launch-provenance hardening, consolidated)
Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh, read-only static. Dates: 2026-08-29.

## Round r9x

1. **RESOLVED** — Matched gate evidence is separated from explicitly retired evidence across JSON, Markdown, and all 16 NPZs. [offgrid_probe_report.md:64](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/offgrid_probe_report.md:64)

4. **PARTIALLY** — Capture failures become false-clean/empty values; missing live SHA/GPU checks are skipped, yet `environment_verified=true` is returned. [meshgrid_offgrid_probe.py:192](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:192), [meshgrid_offgrid_probe.py:318](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:318), [meshgrid_offgrid_probe.py:363](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:363)

**FINAL verdict — REJECT.** Minimal blocking set: **#4 only**. Fail closed on unavailable live SHA/status/GPU capture, resolve the executing UUID with `CUDA_VISIBLE_DEVICES`, persist the live comparison, then rerun emit-then-run. No other result-corrupting findings. Static/read-only review only.
## Round r9z

**Residual #4: PARTIALLY RESOLVED.**  
**FINAL verdict: REJECT.**

Minimal blocking set:

1. Record-emission `git status` failures still become “clean.” `build_launch_record()` discards `git_status_capture`, dirty state is not required, and `bool(None)` reaches the clean pass: [meshgrid_offgrid_probe.py:154](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:154), [line 440](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:440), [line 367](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:367). This violates r9x’s named-refusal requirement.

2. The GPU axis resolves only the live `CUDA_VISIBLE_DEVICES` mapping, then checks membership in all recorded GPU UUIDs. It never resolves and compares the record-time physical GPU. Changing visibility between emission and execution can move `cuda:0` to another installed card while still passing 4/4: [line 397](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:397), [line 408](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:408), [line 412](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:412).

Probe v6’s particular artifacts are internally consistent—SHA `084244b…`, recorded/live clean, `CUDA_VISIBLE_DEVICES=null`, four persisted passes, canonical exit, and byte-identical output/worklog copies: [v6 log:60](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_r1_offgrid_v6.log:60), [report JSON](/home/yixunhu/codespace/FLAC/outputs_loc/exp22/r1_offgrid_probe_P1_v6/offgrid_probe_report.json:1). No numerical-result inconsistency was found. Nevertheless, the shipped canonical-admission contract remains fail-open, so probe v6 cannot stand as canonical under the prior review standard.

Static read-only review only; no writes, tests, installs, environment changes, or GPU activity.
## Round r9z3

# FINAL — exp22-r9z3: REJECT

1. **PARTIALLY RESOLVED.** Capture failures and non-boolean `dirty` now fail closed, and admission no longer coerces `dirty`. However, emission checks `value is None`, not non-emptiness, at [meshgrid_offgrid_probe.py:490](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:490). Thus a successful `git_sha=""` or `hostname=""` capture can still reach the writer. The claimed empty-value regression actually supplies `git_sha=None` at [test_loc_meshgrid_offgrid_probe.py:3254](/home/yixunhu/codespace/FLAC/src/tests/test_loc_meshgrid_offgrid_probe.py:3254). Admission later rejects these records, so this did not corrupt v7, but the stated emission contract remains incomplete.

2. **PARTIALLY RESOLVED.** Designation equality replaced membership correctly at [meshgrid_offgrid_probe.py:405](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:405). Blocking defect: with `CUDA_VISIBLE_DEVICES` unset, the resolver assumes sorted `nvidia-smi`/NVML indices are CUDA logical ordinals at [meshgrid_offgrid_probe.py:230](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:230). CUDA enumeration is controlled by `CUDA_DEVICE_ORDER`, defaulting to `FASTEST_FIRST`; NVIDIA also states that NVML indices may not correlate with CUDA indices. [CUDA documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-device-order), [NVML documentation](https://docs.nvidia.com/deploy/archive/R510/nvml-api/group__nvmlDeviceQueries.html)

Emission and admission reuse the same inferred mapping, so both can agree on the wrong UUID. V7 used the vulnerable unset case at [loc_meshgrid_r1_offgrid_v7.log:58](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_r1_offgrid_v7.log:58). Its `environment_verified=4/4` therefore proves resolver self-consistency, not which physical card executed.

Minimal blocking set:

- Enforce genuinely non-empty captured facts and test refusal through `write_launch_record`.
- Bind the selected device’s actual CUDA-runtime UUID, or refuse canonical emission unless an unambiguous UUID-valued `CUDA_VISIBLE_DEVICES` is used; then rerun v7 from the clean fixed commit.

The v7 files and worklog mirrors otherwise match and report canonical completion, but the designated-card provenance is not established. Static review only; nothing modified or executed.
## Round r9z5

## FINAL verdict: REJECT

1. **PARTIALLY RESOLVED.** The implementation correctly rejects blank text after stripping, empty GPU lists, and invalid booleans before writing ([implementation](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:502)). However, the required regression is incomplete: `""`, `"   "`, and `gpus=[]` are tested only through `build_launch_record`; the subsequent file-absence assertion is vacuous because that path was never passed in. Only `hostname=""` actually traverses `write_launch_record` ([test](/home/yixunhu/codespace/FLAC/src/tests/test_loc_meshgrid_offgrid_probe.py:3194)).

2. **RESOLVED.** Emission and admission obtain the normalized UUID from `torch.cuda.get_device_properties(ordinal).uuid` and require equality; unavailable runtime identity refuses emission. NVML/order remain contextual, legacy inference helpers are deleted, and permutation/different-card tests cover the contract ([implementation](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:237), [tests](/home/yixunhu/codespace/FLAC/src/tests/test_loc_meshgrid_offgrid_probe.py:3260)).

Probe v8 itself shows no result corruption: `canonical=true`, environment verification `4/4`, runtime UUID equality, complete 16/16 publication, and byte-identical tracked mirrors ([report](/home/yixunhu/codespace/FLAC/outputs_loc/exp22/r1_offgrid_probe_P1_v8/offgrid_probe_report.json:1), [log](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_r1_offgrid_v8.log:149)).

Minimal blocking set: route every named blank/whitespace/empty-GPU case through `write_launch_record` and assert its target remains absent after each refusal. No production-code change or v8 rerun is indicated if this remains a test-only correction.

Static review only; no tests executed.
## Round r9z7

### exp22-r9z7 — FINAL sign-off

- Routing item: **RESOLVED**
- New result-corrupting findings: **None**
- FINAL verdict: **APPROVE**

Commit `41720e1` provides the required non-vacuous, same-path `write_launch_record` regression coverage and changes tests only. No production change or probe-v8 rerun is indicated. Probe v8 remains canonical; the exp_22 P1 §2 control set is complete and canonical.

Read-only static review; nothing executed or modified.