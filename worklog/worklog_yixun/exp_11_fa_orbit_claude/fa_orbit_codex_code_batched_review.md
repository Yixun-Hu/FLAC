# Code review — exp_11 batched-orbit round (commits 1479304, d4164e8)

**Reviewer:** OpenAI Codex (gpt-5.6-sol, reasoning effort xhigh, codex-cli 0.146.0, `codex exec`) · **Sandbox:** `--sandbox danger-full-access` (bwrap unavailable); read-only instruction, tree verified clean post-review · **Date:** 2026-08-06 · *(reviewer's self-identification line below retained verbatim)*

# Code review — exp_11_fa_orbit BATCHED-ORBIT round

**Reviewer:** OpenAI Codex (GPT-5, API invocation, read-only review) · **Date:** 2026-08-06  
**Commits reviewed:** `1479304d52f75ccb2b31292946efb41568941854`, `d4164e8f462f57f3e4d772a25ef3830d64d2935c`

## Verdict

**REJECT — 7 BLOCKING, 2 NIT**

The deterministic angle-major stacking, slicing, accumulation, and base-mask handling are correct. The production DINOv3 training path, however, is not stateless: it performs a random RoPE position rescale once per DINO forward. Batching changes both those random values and subsequent CUDA RNG state. The proposed probe explicitly switches the conditioner to evaluation mode, so it disables the behavior that breaks equivalence; it also cannot currently construct its dataloader.

**Launch authorization: none.**

## Findings

### 1. BLOCKING — Training-mode DINOv3 makes batching semantically non-equivalent

The pinned DINOv3 config has `pos_embed_rescale: 2.0` ([config.json:23](/n/fs/gatrdp/hf_cache/hub/models--facebook--dinov3-vits16-pretrain-lvd1689m/snapshots/114c1379950215c8b35dfcd4e90a5c251dde0d32/config.json:23)). In training mode, every DINO forward draws a fresh random scale ([modeling_dinov3_vit.py:123](/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/transformers/models/dinov3_vit/modeling_dinov3_vit.py:123), [modeling_dinov3_vit.py:163](/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/transformers/models/dinov3_vit/modeling_dinov3_vit.py:163)). `GeometryConditioner` invokes DINO once for every source/context coordinate ([conditioners.py:282](/n/fs/gatrdp/codespace/FLAC/src/models/conditioners.py:282)).

With the normal one-source/eight-context stack, each conditioner call therefore makes nine independently augmented DINO forwards. For C32 at training batch 8:

- Loop: base plus 31 angle calls = \(32 \times 9 = 288\) random RoPE draws.
- Batched: base plus four chunks `(64,64,64,56)` = \(5 \times 9 = 45\) draws.

Each batched draw is also shared across several angles that previously received independent draws. This changes the conditioner output distribution and leaves a different CUDA RNG state before timestep/noise generation. It is not floating-point reassociation.

A direct read-only check of the pinned module confirmed that repeated train-mode RoPE calls consume RNG and change the embeddings; evaluation-mode calls do neither.

There is no other identified state hazard in this exact stack: DINO has no BatchNorm, `attention_dropout=0`, `drop_path_rate=0`, and uses LayerNorm; its coordinate cache is immutable. The RIR ResNet BatchNorm remains confined to the unchanged base pass. The random RoPE augmentation is nevertheless sufficient to reject equivalence.

**Fix:** either:

1. Batch the DINO body while generating and applying one RoPE augmentation per original `(angle, ViT id, context index)` invocation in the original RNG order, including preservation of the post-conditioning RNG state; or
2. Obtain explicit approval to disable/redefine the augmentation, document that as a recipe change rather than “same math,” and retain the contemporaneous C4L control.

Add train-mode regressions covering outputs, RNG advancement, and gradients.

### 2. BLOCKING — The probe disables the exact training behavior it must test

The probe calls `cond.eval()` ([fa_orbit_equiv_probe.py:70](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_equiv_probe.py:70)) and runs both paths under `torch.no_grad()` ([fa_orbit_equiv_probe.py:110](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_equiv_probe.py:110)). This:

- disables the stochastic RoPE rescale;
- bypasses the non-reentrant activation-checkpoint path, which requires training mode and gradients ([conditioners.py:180](/n/fs/gatrdp/codespace/FLAC/src/models/conditioners.py:180));
- does not compare gradients, whose batch-reduction order can differ even when forward values agree;
- uses batch 4 rather than the pinned per-rank training batch 8;
- tests only C4/C32, omitting the distinct C8/C16 chunk plans;
- omits training’s `torch.set_float32_matmul_precision("medium")` setting ([train.py:94](/n/fs/gatrdp/codespace/FLAC/train.py:94)).

**Fix:** run a training-mode, grad-enabled probe at B8 for C4/C8/C16/C32 with the actual bf16-mixed and fp32-medium settings. Restore identical CPU/CUDA RNG and model state before each side, compare post-call RNG states, outputs, and representative parameter gradients. Separately test the evaluation schedules B64 and the final B1 batch.

### 3. BLOCKING — The real-data probe currently cannot run

`real_samples()` requests `num_workers=0` ([fa_orbit_equiv_probe.py:80](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_equiv_probe.py:80)), but the shared factory always passes `persistent_workers=True` ([dataset.py:405](/n/fs/gatrdp/codespace/FLAC/src/data/dataset.py:405)). The installed PyTorch immediately raises:

```text
ValueError: persistent_workers option needs num_workers > 0
```

The samples are also not the deterministic “first samples”: shuffle defaults to true, the training config enables augmentations, and NumPy/Python RNGs are not seeded.

**Fix:** construct a deterministic loader with `shuffle=False` and a valid worker configuration, seed Python/NumPy/Torch, disable irrelevant audio augmentation, record exact sample identifiers, and require exactly eight successfully loaded records. Add tests for this probe path per the SOP; none currently exist for the new executable.

### 4. BLOCKING — The numerical acceptance test is fail-open and its tolerances are not defensible as written

The loop and batched implementations perform the tensor additions in the same order. The differences arise inside differently shaped forward kernels—and currently from stochastic state—not from an orbit-sum reassociation. The claims at [yaw_rotation.py:347](/n/fs/gatrdp/codespace/FLAC/src/data/yaw_rotation.py:347), [test_invariant_conditioning.py:550](/n/fs/gatrdp/codespace/FLAC/src/tests/test_invariant_conditioning.py:550), and the probe preamble are therefore incorrect.

Additional acceptance problems:

- r2 §B7 requested maximum absolute and relative tolerances, but only relative error affects the verdict; maximum absolute error is merely printed.
- Elementwise `abs_err / abs(reference)` is ill-conditioned near zero.
- `2e-3` is below bf16’s unit roundoff near one (\(2^{-8}\approx3.9\times10^{-3}\)) and has no measured justification for a maximum elementwise statistic.
- Missing both ViT ids yields an empty result, but `cells` still increments and the probe can pass with zero errors.
- Non-finite outputs are not rejected; `max(0.0, NaN)` can suppress the NaN and retain zero.

**Fix:** require the exact two ids and all finite tensors for every expected cell. Pre-register both scale-aware maximum-absolute and normwise-relative bounds; if an elementwise relative metric remains, combine it with an absolute floor rather than dividing by values near zero. Include forward and gradient comparisons and fail on any missing/non-finite measurement.

### 5. BLOCKING — The sbatch wrapper is not fail-closed

The wrapper uses `set -uo pipefail`, then discards several command statuses:

- failure of `git status` can appear as empty drift;
- the environment/version command and `sha256sum` are informational only;
- the training pipeline records only `PIPESTATUS[0]`, discarding `tee` failure ([fa_orbit_equiv_probe.sbatch:63](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_equiv_probe.sbatch:63));
- it accepts any one `EQUIVPROBE` line without verifying exact cardinality, config hash, sample count, cell count, or `verdict=PASS`;
- unlike the real launcher, it does not run the pinned DINO snapshot/weight-hash gate. Offline mode alone does not bind external-cache identity.

**Fix:** check every provenance command explicitly; capture and require both Python and `tee` status zero; require exactly one strictly parsed result with expected config hash, sample IDs/count, cell set, finite metrics, and `PASS`; run the same DINO revision/weight-hash gate as the arm launcher.

### 6. BLOCKING — Peak memory must be requalified on a real backward step

The cap rationale calls 64 the “effective global batch,” but it is a **per-rank** conditioner batch. At the pinned 8×8 rung, an ordinary rank sees eight samples while a batched orbit call sees up to 64.

The final retained graph is still proportional to all orbit samples, but batching changes peak structure:

- one active DINO call/workspace grows from B8 to B64;
- the source graph remains live while eight context-coordinate DINO calls execute;
- up to 64 rotated full-resolution panoramas and their stacked tensors coexist;
- slices retain the shared batched-output graph until diffusion backward.

This can raise transient peak above the sequential C32 measurement of 30,817 MiB even if total retained activation volume is similar.

**Fix/gate:** run real bf16 training forward, backward, optimizer, and checkpointed recomputation at 8×8, especially C32, with per-UUID allocated/reserved/external peaks. Do not substitute the eval/no-grad equivalence probe. Reduce the cap and repeat if the existing margin is not preserved.

### 7. BLOCKING for confirmatory evaluation — Evaluation protocol identity is currently silent

`eval_FLAC.py` calls the same changed function ([eval_FLAC.py:263](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:263)), but its metrics metadata records only method, angles, and autocast—not loop versus batched execution, cap, or source SHA.

At the default evaluation B64, the batched path degenerates to one angle per call for the first 6,336 items, so those calls match the old grouping. The final item of the 6,337-item split has B1 and is regrouped. Thus the practical historical-C4 effect is narrowly scoped but nonzero, and the current B4/init-only probe does not bound downstream prediction or metric changes.

Using contemporaneous C4L as the inferential comparator makes this acceptable if disclosed. Silently comparing new rows with historical loop-evaluated C4 rows is not acceptable.

**Fix:** record an orbit-execution version, cap, evaluation batch schedule, and source SHA in metric/prediction provenance and validate them. Either re-evaluate the historical checkpoint of record under the batched implementation or label historical rows as legacy-loop and reserve inferential comparisons for C4L. Any direct cross-protocol claim needs full-checkpoint/common-noise prediction parity or re-evaluation; the present conditioner tolerance is insufficient.

### 8. NIT — The call-count test was weakened beyond the intended contract

Changing from `C` calls to a sample-count contract is reasonable, but `calls >= 1` ([test_invariant_conditioning.py:299](/n/fs/gatrdp/codespace/FLAC/src/tests/test_invariant_conditioning.py:299)) does not pin the optimization. A regression to many small calls could pass while restoring multi-day latency.

**Fix:** assert exact production batch plans, including the base call:

- C4/B8: `[8, 24]`
- C8/B8: `[8, 56]`
- C16/B8: `[8, 64, 56]`
- C32/B8: `[8, 64, 64, 64, 56]`
- C32/B3 boundary: `[3, 63, 30]`

Also add a call-varying mask sentinel proving returned masks come specifically from the base pass.

### 9. NIT — The advertised 64-sample cap is not universal

For `batch > 64`, `max(1, 64 // batch)` becomes one angle, so the resulting forward still exceeds 64 samples ([yaw_rotation.py:363](/n/fs/gatrdp/codespace/FLAC/src/data/yaw_rotation.py:363)). The implementation also preserves supplied angle-list order, not necessarily “ascending” order.

**Fix:** either reject batches above the cap, or document the cap as a target that cannot split an angle. Say “input angle order” unless sorted angles are validated.

## Verified behavior

- For B≤64, chunks contain whole angles; no boundary can split an angle.
- Stacking is angle-major, and slicing `out[k*B:(k+1)*B]` maps back to the correct angle and original sample order.
- The accumulation sequence is exactly base, angle 1, angle 2, … in supplied order across chunk boundaries. Chunk boundaries do not reassociate the additions.
- The base pass remains unchanged. Only `base[i][0]` is replaced; all masks and non-ViT outputs remain from angle zero.
- The actual DINO path contains no BatchNorm and no active dropout/drop-path. No mutable position-cache defect was found. Autocast’s weight cache does not introduce model state, although batch-dependent kernels can change rounding.
- `git diff --check`, Python compilation, and shell syntax passed. `src/tests/test_invariant_conditioning.py` passed: **31 tests in 15.52 s**. These deterministic fake-conditioner tests do not cover finding 1.

## Updated launch-preconditions delta

The earlier sequential-orbit P0 report remains useful background, but its VRAM, throughput, wall-time, and manifest pins do not authorize the batched path.

1. **Repair semantic equivalence** from finding 1; add train-mode output/RNG/gradient tests and exact chunk-plan/mask tests.
2. **Repair and re-review the probe and sbatch wrapper**, including deterministic real data, external DINO binding, finite fail-closed metrics, actual B8/all-orbit coverage, and gradient/RNG checks.
3. **Run the corrected equivalence probe** on an L40 from the final reviewed and pushed SHA; log its exact command and durable output.
4. **Spot re-measure the real training path at 8×8**, preferably C4L/C8/C16/C32 so both the speed curve and per-arm wall limits are measured. C32 must complete forward, backward, optimizer step, and checkpoint recomputation without OOM/NaN.
5. **Land a new pin commit** replacing the sequential-path P0/spot binding, free-memory floor, expected rates, and per-arm time limits. Revalidate rather than assume the 8×8 rung and 64-sample cap.
6. **Re-run the exact multi-GPU `SMOKE=1` path** from that pin commit, preserving proof of eight ranks/devices, bf16 + gradient checkpointing, completed optimizer steps, readable full-state checkpoint, W&B identity, dual durable logs, clean classification, and no stale processes.
7. **Record the evaluation-protocol decision** before confirmatory evaluation: batched provenance plus either historical-C4 re-evaluation or explicit legacy-loop labeling with C4L as the sole inferential comparator.
8. **Obtain final independent sign-off** on the fix, probe result, spot evidence, re-pins, and smoke; then commit/push the reviewed SHA and record pre-launch acceptance criteria.

**Final verdict: REJECT. No C4L/C8/C16/C32 arm may launch from `d4164e8`.**
