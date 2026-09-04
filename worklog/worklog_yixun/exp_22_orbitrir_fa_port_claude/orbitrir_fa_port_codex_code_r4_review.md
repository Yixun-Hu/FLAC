**Reviewer:** OpenAI Codex (gpt-5.6-sol, `codex exec`, read-only sandbox) · **Date:** 2026-09-03
**VERDICT:** REQUEST-CHANGES

Review scope is pinned to `3d14856..da370c2`. While the review was running, `main` advanced through two later README/assets commits to `16d4459`; those commits are not covered by this verdict.

## Findings

1. **Minor — `src/tests/test_vit_gradient_checkpointing_dinov3.py:72-73,162-182`**

   - **What:** The module calls `os.environ.setdefault()` during import. By the time `_offline_hf` creates its `MonkeyPatch`, the original value is already `"1"`, so `mp.undo()` restores `"1"` rather than removing the variable.
   - **Why:** `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` leak into later test modules, contradicting the documented “then undo it” behavior and making the suite order-dependent.
   - **Prescribed fix:** Remove the import-time environment mutation and defer `create_multi_conditioner_from_conditioning_config`’s import into `_build()`, which runs after `_offline_hf` has installed reversible environment and module-constant patches. Add an assertion that previously absent variables are absent after fixture teardown.

2. **Minor — `src/tests/test_fa_config.py:90`**

   - **What:** Shared leaves are compared using ordinary Python `!=`. Consequently, changes such as JSON `true → 1`, `false → 0`, or `1 → 1.0` compare equal and are not reported as a fifth delta. Scalar-list comparison has the same type-equivalence hole.
   - **Why:** The current files really do differ by exactly four semantic additions—the raw config diff confirms it—but the permanent test is not fully type-strict despite claiming to reject all other drift.
   - **Prescribed fix:** Compare typed leaf values, including each scalar-list element, for example `(type(value), value)`, while retaining conditioner-ID keying and order-sensitive scalar lists.

3. **Minor — `README.md:199-200,216-218`**

   - **What:** Two arithmetic explanations are inaccurate:
     - An angle is exact when `angle × panorama_width / 360` is integral, not when the angle “divide[s] the panorama width evenly.”
     - `32 × 2 = 64` is the global **scene** batch. `RIRConditioner.forward()` reshapes `[B,N,C,T]` to `[B×N,C,T]`, so its ResNet BatchNorm tensor batch is `64×K` across two ranks, not 64.
   - **Why:** The training command is correct, but these statements misdescribe why the C4 orbit is exact and what SyncBN aggregates.
   - **Prescribed fix:** State the integral-column-shift criterion and describe 64 as the global scene batch; note that the context-audio encoder additionally folds the `K` context dimension into its BatchNorm input.

4. **Minor — `README.md:267-268`**

   - **What:** The precomputed-conditioning call matches the function signatures, but it is only reliable under an unstated single-sample CUDA assumption. `generate_diffusion_cond()` independently creates noise using its default `batch_size=1` and `device="cuda"`.
   - **Why:** If `metadata` contains a normal multi-item batch, or `device` is not CUDA, conditioning and noise have incompatible batch/device contracts.
   - **Prescribed fix:** Pass `batch_size=len(metadata), device=device` to `model.generate()`, or explicitly label the snippet as single-sample CUDA.

5. **Nit — `src/configs/model_configs/FLAC/AR/FLAC_AR_FA.json:62-171`**

   - **What:** Seventeen added lines contain trailing whitespace, and the new file lacks a final newline.
   - **Why:** Both scoped and full-port `git diff --check` commands exit 2, violating the SOP’s mandatory static gate.
   - **Prescribed fix:** Normalize whitespace identically in the vanilla/FA twins, retain the four semantic deltas, add the final newline, and rerun `git diff --check`.

## Focus items

### (a) Training flags

- `_as_bool("") → False` is acceptable as the carried, reference-compatible opt-in convention. The rule should be that recorded commands spell `true` or `false` literally; empty shell-variable expansions must not be used for protocol-defining flags.
- The SyncBN guard lives at the shared Trainer-kwargs boundary. It protects both direct callers and `main()` and fires before `pl.Trainer` construction, though after dataset/model setup.
- With SyncBN off, no `sync_batchnorm` key is added. The remaining 14-key dictionary and object passthroughs match the pre-change Trainer call.
- C8 `train.py` and `defaults.ini` are byte-identical to `e85ebde`; final `defaults.ini` matches `f59f5a4`, and final `train.py` differs from it only by the declared attribution reword.

### (b) Gradient checkpointing

The adapter is structurally correct for the pinned real backbone: twelve DINOv3 layers, shared-backbone idempotency, non-reentrant checkpoint execution, eval/no-grad bypass, unchanged state dictionaries, and recorded bit-exact gradients over 212 tensors.

The stub/real split does not directly cover mixed precision, DDP, the multi-context `context_poses_vit` loop, or four orbit forwards followed by one backward. The planned three-step, two-GPU smoke is the correct coverage for those integration surfaces.

The module-scoped conditioner fixtures do not currently drift BatchNorm because every executed forward restricts `only_ids` to `source_vit`, and cleanup restores mode/grad state. If a future test executes `context_audio` in train mode, it must snapshot/restore named buffers or use an isolated fixture; otherwise ResNet running statistics will leak across tests. The present import-time environment leak is Finding 1.

### (c) C11 configuration

The checked-in configs themselves have exactly four semantic additions. Conditioning entries keyed by ID and whole-list comparison are appropriate; Finding 2 is the remaining type-equivalence hole.

All four relevant ViT blocks in `FLAC_AR.json` and `FLAC_AR_FA.json` use revision `114c1379950215c8b35dfcd4e90a5c251dde0d32`, and that exact snapshot contains both cached config and weights offline.

Leaving `_S`, `_AllCA`, `_InContext`, and `_VAECtxt` unpinned is acceptable under D10: they are not participants in this vanilla/FA identity run. They should be pinned in a separate change before any of those variants becomes part of a reproducibility-sensitive experiment.

### (d) README

The method routing, guard inheritance/override behavior, bf16 conditioning requirement, rotation ordering, default suffix forms, and raw-conditioner warning agree with the code. The precomputed API arguments exist, subject to Finding 4’s batch/device contract. The two arithmetic descriptions needing correction are Finding 3.

### (e) Completeness and execution readiness

No approved C1–C12 method surface is missing: portability, yaw transforms, cylindrical features, frame averaging, shared dispatch, guarded evaluation, max-step and SyncBN flags, gradient checkpointing, the FA config, revision pinning, and API documentation are present.

Execution evidence checked statically:

- The full unseen split contains 6,337 RIR entries.
- The B-F checkpoint is 723,922,667 bytes and its SHA-256 exactly matches `5319feb4af874624859e87105ddd8ab06d4b449769d1e054f712b2b1c0542328`.
- The K=8 and K=1 reference JSONs contain the stated acceptance values.
- The pinned DINOv3 snapshot is complete in the local cache.
- The trained-as guard will refuse explicit vanilla evaluation of the embedded FA checkpoint before model construction unless the override flag is present.

Environmental prerequisites remain: `ORBITRIR/AcousticRooms` and `ORBITRIR/weights` have not yet been linked, and this review shell cannot communicate with an NVIDIA driver. Those are launch-host/setup conditions, not port defects.

## Execution-phase gate

**NO-GO currently.** R4 remains open under the SOP until these findings are fixed and rechecked. After a focused fix review, creation of the planned data/weights links, and use of a functioning two-GPU host, the production path is a **GO** for the storage-light smoke, pinned K={8,1} acceptance cells, and mandatory guard-refusal negative control.

## Commands run

No Python, tests, installs, environment modification, or file writes were performed. Read-only commands included:

```bash
cat worklog/experiment_SOP.md
rg --files worklog/worklog_yixun/announcement
tail -n +1 worklog/worklog_yixun/announcement/{01_*,02_*,03_*,04_*,05_*,06_*}.md
wc -l <plan/worklog/R1-R3 files>
cat <plan, worklog, R1, R2, R3, commits files>
sed -n ... <worklog>
nl -ba ... | sed -n ...    # full changed files in bounded chunks

git -C /home/yixunhu/codespace/ORBITRIR status --short --branch
git -C /home/yixunhu/codespace/ORBITRIR log --oneline --decorate 3d14856..da370c2
git -C /home/yixunhu/codespace/ORBITRIR diff 3d14856..da370c2
git -C /home/yixunhu/codespace/ORBITRIR diff --name-status 3d14856..da370c2
git -C /home/yixunhu/codespace/ORBITRIR diff --stat 3d14856..da370c2
git -C /home/yixunhu/codespace/ORBITRIR diff --check 3d14856..da370c2
git -C /home/yixunhu/codespace/ORBITRIR diff --check ead8bbd..da370c2
git -C /home/yixunhu/codespace/ORBITRIR show <each C8-C12 SHA>
git -C /home/yixunhu/codespace/ORBITRIR show da370c2:<changed path>
git -C /home/yixunhu/codespace/ORBITRIR diff --stat da370c2..16d4459

cmp --silent <(git -C FLAC show <reference>:<path>) <(git -C ORBITRIR show <target>:<path>)
diff -u <(git -C FLAC show <reference>:<path>) <(git -C ORBITRIR show <target>:<path>)
diff -u ORBITRIR/.../FLAC_AR.json ORBITRIR/.../FLAC_AR_FA.json

rg -n <method/config/test patterns> /home/yixunhu/codespace/ORBITRIR
find ~/.cache/huggingface/hub/models--facebook--dinov3-vits16-pretrain-lvd1689m ...
cat <cached DINOv3 config.json>
ls -ld <dataset/weights/checkpoint paths>
sha256sum <B-F 40k checkpoint>
rg -o 'hybrid_IR\.wav' data/AR/unseen_eval.json | wc -l
tail -n +1 <K=8 and K=1 reference metric JSONs>
df -h /home/yixunhu/codespace/{ORBITRIR,FLAC}
nvidia-smi --query-gpu=...   # failed: no accessible NVIDIA driver
```

A `jq` count was attempted but `jq` is not installed; the split count was instead obtained with `rg`/`wc`.