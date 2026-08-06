# Code re-review — exp_11 batched-orbit fix round (8094d60..10c41e1)

**Reviewer:** OpenAI Codex (gpt-5.6-sol, reasoning effort xhigh, codex-cli 0.146.0, `codex exec`) · **Sandbox:** `--sandbox danger-full-access` (bwrap unavailable); read-only instruction, tree verified clean post-review · **Date:** 2026-08-06

# Code re-review — exp_11_fa_orbit batched-orbit fix round

**Reviewed commits:** `8094d60`, `3075294`, `deeaddf`, `b117acb`, `10c41e1`  
**Approval record:** `f54d2ac`  
**Verdict:** **REJECT — 3 NEW BLOCKING, 3 NEW NIT**

The production batching implementation honors approved option 2: the arithmetic remains ordered identically, train-mode RoPE draws are chunk-shared, every new arm—including C4L—uses that path, C4L is designated the sole inferential comparator, and historical C4 is labeled legacy-loop. The redesigned fp32 gate itself is numerically sound. The submitted wrapper, however, cannot complete successfully, and two provenance/fail-closed defects can admit invalid probe evidence.

## Prior findings 1–9

| # | Status | Re-review evidence |
|---|---|---|
| 1 | **CLOSED** | The recipe change is disclosed precisely: identical averaging arithmetic but fewer chunk-shared train-mode RoPE draws, applied to C4L and all arms, with historical rows labeled legacy-loop ([yaw_rotation.py:36](/n/fs/gatrdp/codespace/FLAC/src/data/yaw_rotation.py:36)). Input-angle accumulation is still ordered without reassociation ([yaw_rotation.py:362](/n/fs/gatrdp/codespace/FLAC/src/data/yaw_rotation.py:362)). The option-2 approval is recorded ([fa_orbit_yixun_query.md:77](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_yixun_query.md:77)). C4L/C8/C16/C32 all select `fa_invariant` and route through the same implementation ([diffusion.py:205](/n/fs/gatrdp/codespace/FLAC/src/training/diffusion.py:205)). Chunk plans and stochastic-forward counts are pinned ([test_invariant_conditioning.py:607](/n/fs/gatrdp/codespace/FLAC/src/tests/test_invariant_conditioning.py:607), [test_invariant_conditioning.py:704](/n/fs/gatrdp/codespace/FLAC/src/tests/test_invariant_conditioning.py:704)). No per-angle reproduction is required under approved option 2. |
| 2 | **PARTIALLY-CLOSED** | The probe now enters real train mode with gradients, medium matmul precision, bf16 autocast, and B8 C4/C32 qualification; it separately covers evaluation schedules ([fa_orbit_equiv_probe.py:237](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_equiv_probe.py:237), [fa_orbit_equiv_probe.py:291](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_equiv_probe.py:291)). It correctly does not demand train-mode equality under option 2. It remains partial because the CUDA/bf16 path is not fail-closed and the gradient assertion accepts any parameter gradient; see NEW-2 and NIT-1. |
| 3 | **PARTIALLY-CLOSED** | Direct dataset indexing fixes the invalid `persistent_workers`/zero-worker construction and obtains eight records deterministically ([fa_orbit_equiv_probe.py:190](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_equiv_probe.py:190)). The emitted identifiers are not exact record identifiers, however; see NEW-3. |
| 4 | **CLOSED** | `deviation()` rejects non-finite tensors and computes `max_abs`, normwise `rel_norm`, and diagnostic `rel_max` with an absolute floor ([fa_orbit_equiv_probe.py:91](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_equiv_probe.py:91)). `verdict()` requires both ViT IDs and enforces both `rel_norm ≤ 1e-6` and `max_abs ≤ 1e-5` for gated fp32 eval cells ([fa_orbit_equiv_probe.py:111](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_equiv_probe.py:111)). The 14-cell plan comprises 12 gated eval cells plus two non-equivalence train qualification cells ([fa_orbit_equiv_probe.py:82](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_equiv_probe.py:82)). |
| 5 | **NOT-CLOSED** | The intended strict checks are present, but the wrapper always aborts while capturing `PIPESTATUS`, and it validates only a nonempty sample-ID token. See NEW-1 and NEW-3. |
| 6 | **NOT-CLOSED** | Correctly documented as outside the probe ([fa_orbit_equiv_probe.py:26](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_equiv_probe.py:26)), but no batched-path 8×8 real forward/backward/optimizer/checkpoint-recompute remeasurement exists yet. This remains a launch precondition. |
| 7 | **PARTIALLY-CLOSED** | Metrics and prediction sidecars now record execution, cap, and source SHA ([eval_FLAC.py:69](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:69), [eval_FLAC.py:111](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:111)); the C4L/legacy-loop decision is explicit. Metrics still omit batch size/sample count, so their B64/B1 schedule is not reconstructible. More importantly, the prediction comparator does not validate `orbit_execution`, cap, or source SHA—its guarded keys stop at `cond_autocast` ([compare_predictions.py:152](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_02_yaw_noninvariance_claude/compare_predictions.py:152)). Historical-versus-batched comparison therefore remains fail-open. |
| 8 | **CLOSED** | Exact production plans `[8,24]`, `[8,56]`, `[8,64,56]`, `[8,64,64,64,56]`, and `[3,63,30]` are pinned, and the mask sentinel proves masks come from the base pass ([test_invariant_conditioning.py:607](/n/fs/gatrdp/codespace/FLAC/src/tests/test_invariant_conditioning.py:607), [test_invariant_conditioning.py:637](/n/fs/gatrdp/codespace/FLAC/src/tests/test_invariant_conditioning.py:637)). |
| 9 | **CLOSED** | B=64 is accepted, B>64 fails loudly, and documentation says input-angle order ([yaw_rotation.py:365](/n/fs/gatrdp/codespace/FLAC/src/data/yaw_rotation.py:365), [test_invariant_conditioning.py:654](/n/fs/gatrdp/codespace/FLAC/src/tests/test_invariant_conditioning.py:654)). |

## NEW findings

### NEW-1 — BLOCKING: the wrapper always aborts after the probe pipeline

At [fa_orbit_equiv_probe.sbatch:98](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_equiv_probe.sbatch:98):

```bash
PROBE_RC="${PIPESTATUS[0]}"; TEE_RC="${PIPESTATUS[1]}"
```

The first assignment is itself a command and replaces `PIPESTATUS` with a one-element array. With `set -u`, the second expansion raises `PIPESTATUS[1]: unbound variable`; a direct shell reproduction exited 127. Thus even a green probe and successful `tee` can never reach result parsing.

Capture the array atomically before indexing, for example:

```bash
PIPE_RC=("${PIPESTATUS[@]}")
PROBE_RC="${PIPE_RC[0]}"
TEE_RC="${PIPE_RC[1]}"
```

### NEW-2 — BLOCKING: CUDA/bf16 qualification is fail-open

The probe silently chooses CPU when `torch.cuda.is_available()` is false ([fa_orbit_equiv_probe.py:291](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_equiv_probe.py:291)). In that case, a requested bf16 cell explicitly disables autocast ([fa_orbit_equiv_probe.py:244](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_equiv_probe.py:244)) but is still printed as `bf16` ([fa_orbit_equiv_probe.py:313](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_equiv_probe.py:313)). The wrapper verifies that `nvidia-smi` sees an L40, not that PyTorch can initialize CUDA or that autocast produced bf16 work.

Require CUDA in the probe, record `device=cuda` in `EQUIVPROBE`, and have the wrapper parse it. A CUDA initialization failure must abort rather than becoming a mislabeled CPU/fp32 qualification.

### NEW-3 — BLOCKING: “exact sample IDs” are eight identical scene labels

The probe records `scene` before `id` or index ([fa_orbit_equiv_probe.py:218](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_equiv_probe.py:218)). Independent execution loaded eight distinct records but emitted:

```text
Cafe,Cafe,Cafe,Cafe,Cafe,Cafe,Cafe,Cafe
```

The dataset already exposes exact `idx`, `path`, and `relpath` fields ([dataset.py:269](/n/fs/gatrdp/codespace/FLAC/src/data/dataset.py:269)). The wrapper then checks only that `sample_ids` is nonempty ([fa_orbit_equiv_probe.sbatch:106](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_equiv_probe.sbatch:106)), not that it contains the registered eight identities.

Emit stable exact identifiers such as `idx:relpath`, pin their expected digest/list, and validate it in the wrapper.

### NEW-4 — NIT: the stochastic contract test simplifies the real DINO draw topology

`StochasticGeometry` takes one draw for an entire geometry-conditioner call ([test_invariant_conditioning.py:671](/n/fs/gatrdp/codespace/FLAC/src/tests/test_invariant_conditioning.py:671)). The real context conditioner calls the shared DINO backbone once per context coordinate ([conditioners.py:282](/n/fs/gatrdp/codespace/FLAC/src/models/conditioners.py:282)); each call independently executes the train-mode RoPE draw. The test pins cross-angle sharing and chunk count adequately for option 2, but its “one draw per forward” wording should distinguish geometry-conditioner forwards from DINO forwards.

Also, the gradient test’s claim that “every rotated chunk contributes” is stronger than its assertion: a nonzero aggregate scale gradient could come from the base path alone ([test_invariant_conditioning.py:733](/n/fs/gatrdp/codespace/FLAC/src/tests/test_invariant_conditioning.py:733)).

### NEW-5 — NIT: bf16 measurements are not represented in the machine summary

Only non-bf16 results enter `results` ([fa_orbit_equiv_probe.py:307](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_equiv_probe.py:307)); consequently `rec_rel_norm` and `rec_max_abs` summarize train-mode fp32, not bf16 ([fa_orbit_equiv_probe.py:322](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_equiv_probe.py:322)). Bf16 remains visible in the durable verbose log, so measured-not-gated is acceptable, but explicit bf16 summary fields would make the record auditable.

The approved eight-record repetition used to fill B64 ([fa_orbit_equiv_probe.py:303](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_equiv_probe.py:303)) and diagnostic-only `rel_max` are acceptable.

### NEW-6 — NIT: vanilla evaluations are labeled as batched-orbit executions

Both provenance builders unconditionally record `orbit_execution="batched"` and the cap, even when `cond_method="vanilla"` executes no orbit ([eval_FLAC.py:83](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:83), [eval_FLAC.py:118](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:118)). Use `None`/`not_applicable` for vanilla to avoid false provenance.

## `eval_FLAC.py` behavior

Numerical evaluation behavior is unchanged beyond metadata:

- No CLI option, output path, seed, dataloader, conditioning, noise, sampler, or metric code changed.
- The `fa_invariant` call remains the same production call ([eval_FLAC.py:295](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:295)).
- The commit adds imports, `source_sha()`, and three serialized fields only.
- `source_sha()` executes after evaluation during record construction and does not consume model RNG.

The remaining problem is provenance completeness/validation, not changed predictions or metrics.

## Verification performed

- All five requested commits and the subsequent option-2 approval commit were inspected.
- `git diff --check d4164e8..10c41e1`: passed.
- Python AST parsing and `bash -n`: passed.
- Independent relevant suite: **119 passed, 4 warnings**.
- Committed evidence reports **174 passed, 4 warnings** ([pytest log:1](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-06_14-34-04_pytest_batched.log:1)).
- No tracked files changed during review; the same three pre-existing untracked entries remained.

## Remaining launch preconditions

The expected sequence is correct but must be amended with the blocking probe fixes:

1. Fix NEW-1 through NEW-3, complete the missing evaluation-provenance validation, obtain re-review, and push the resulting SHA.
2. Run the corrected equivalence/qualification probe green on one L40 from that exact pushed SHA, preserving the durable log and exact sample identities.
3. Spot remeasure **C4L, C8, C16, and C32** at 8×8 through the real train path, including backward, optimizer step, and checkpoint recomputation, with per-GPU memory and throughput evidence.
4. Land and push a new pin commit containing the batched-path measurements, memory floor, expected rates, and arm time limits.
5. Re-run the exact eight-GPU `SMOKE=1` workflow from that pin commit.
6. Obtain final independent sign-off on the fixes, green probe, four-arm spot evidence, new pins, and smoke evidence.
7. Before confirmatory evaluation or aggregation, make the metrics schedule reconstructible and enforce `orbit_execution`/cap/source-SHA compatibility; historical rows must enter only as explicitly audited `legacy-loop`, never as C4L-equivalent evidence.

**No C4L/C8/C16/C32 arm is authorized to launch from `f54d2ac`.**
