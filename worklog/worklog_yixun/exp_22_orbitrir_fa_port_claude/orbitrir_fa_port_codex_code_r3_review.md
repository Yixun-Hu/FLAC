**Reviewer:** OpenAI Codex (gpt-5.6-sol, `codex exec`, read-only sandbox) · **Date:** 2026-09-03

**VERDICT:** REQUEST-CHANGES

## Findings

1. **BLOCKING — [eval_FLAC.py:192](/home/yixunhu/codespace/ORBITRIR/eval_FLAC.py:192), [yaw_rotation.py:357](/home/yixunhu/codespace/ORBITRIR/src/data/yaw_rotation.py:357)**

   - **What:** `trained_conditioning()` falls back to the supplied model config whenever an embedded `model_config` or `training` block is malformed or missing. Only the complete absence of `model_config` identifies a bare checkpoint. Consequently, an FA checkpoint containing `model_config=None`, `training=None`, or a partial embedded config can be “certified” as vanilla by the external JSON. Tests at [test_protocol_guard.py:389](/home/yixunhu/codespace/ORBITRIR/src/tests/test_protocol_guard.py:389) explicitly enshrine this unsafe fallback. Angle normalization also accepts malformed representations such as numeric strings, mapping keys, booleans, and non-finite numbers because it blindly applies `float()`.
   - **Why:** This defeats the core trained-as guard and can recreate the exact silent off-protocol evaluation it is intended to prevent. `eval_pl` additionally reaches model construction before rejecting some invalid-but-matching protocols.
   - **Prescribed fix:** Fall back only when the checkpoint has no `model_config` key. If that key exists, require a dict containing a dict `training` block; likewise require a valid external training block for genuinely bare checkpoints. Preserve “missing `cond_method` inside a valid training dict means vanilla.” Validate both trained and runtime protocols before comparison: valid method, list/tuple angles, finite numeric elements excluding booleans, non-empty/zero-first FA orbit. Apply the same validation before `eval_pl` model construction. Add adversarial tests for malformed embedded structures and string/dict/bool/NaN angles.

2. **Major — [eval_FLAC.py:395](/home/yixunhu/codespace/ORBITRIR/eval_FLAC.py:395)**

   - **What:** EMA selection overlays any available `diffusion_ema.ema_model.*` keys on top of the complete online `model.*` state. A partial EMA copy therefore produces a hybrid EMA/online model whose key union is complete, so `check_load_integrity()` reports no missing or stray keys. A wrapped checkpoint declaring `use_ema=true` but containing no EMA copy similarly falls back silently to online weights.
   - **Why:** D13’s helper always raises when it sees a mismatch, but this preprocessing hides the mismatch before the helper sees it.
   - **Prescribed fix:** For an embedded wrapped checkpoint selecting EMA, require the EMA key set to be present and complete. Remove online `model.*` keys before installing mapped EMA keys, or explicitly compare the mapped EMA set with the expected/online model-key set. Continue permitting genuinely bare exported checkpoints. Add tests for zero and one-key-short EMA copies through the real preprocessing/load path.

3. **Major — [eval_FLAC.py:220](/home/yixunhu/codespace/ORBITRIR/eval_FLAC.py:220)**

   - **What:** `effective_frame_angles()` normalizes supplied angles before checking the conditioning method. Thus two vanilla protocols containing different, unused angle lists are classified as conflicting. With the override flag, `conditioning_override` becomes true even though no numerical behavior differed.
   - **Why:** This contradicts the CLI and `frame_angles_record()` contract that angles are ignored under vanilla, and violates “override true only when a real conflict was permitted.”
   - **Prescribed fix:** Return `None` immediately unless `cond_method == "fa_invariant"`, then normalize/default the orbit. Test differing extraneous vanilla angles in both evaluation entry points and verify no error and `conditioning_override == false`.

4. **Major — [eval_FLAC.py:49](/home/yixunhu/codespace/ORBITRIR/eval_FLAC.py:49), [eval_FLAC.py:546](/home/yixunhu/codespace/ORBITRIR/eval_FLAC.py:546)**

   - **What:** Output identity remains collision-prone. Python’s default `g` format retains only six significant digits, so distinct rotation requests can share a suffix; large integer-valued angles such as `1000000.0` also become scientific notation rather than the claimed legacy integer spelling. Separately, only orbit cardinality is encoded, so default C4 and a reordered C4 override produce the same path.
   - **Why:** Different protocols using the same `eval_name` can overwrite both metrics and predictions. Shared record fields cannot protect the artifact that was overwritten. The mandated `g` spelling therefore does not meet its stated collision-free property.
   - **Prescribed fix:** Re-adjudicate the token as integer-special-case plus an exact round-trip float representation, rejecting non-finite rotations. Preserve existing common integer/default-C4 names. Include a canonical orbit token or digest for non-default orbits. Test negative/scientific inputs, high-precision values around a panorama-column threshold, large integers, and same-length reordered orbits.

5. **Minor — [eval_pl.py:142](/home/yixunhu/codespace/ORBITRIR/eval_pl.py:142)**

   - **What:** The extended `eval_pl` record omits `trained_frame_avg_angles` and `conditioning_override`, although the stated C7c schema says records gain both fields.
   - **Why:** The two evaluation entry points produce inconsistent provenance schemas, and the trained orbit cannot be inspected directly from an `eval_pl` JSON.
   - **Prescribed fix:** Add normalized `trained_frame_avg_angles` and `"conditioning_override": false`, then assert both for default and custom FA orbits.

6. **Nit — [yaw_rotation.py:382](/home/yixunhu/codespace/ORBITRIR/src/data/yaw_rotation.py:382)**

   - **What:** The dispatcher docstring still says evaluation will use it “in a later commit,” although C7b now does so.
   - **Prescribed fix:** Reword it to describe the current shared training/evaluation dispatch.

## Adversarial-focus results

- **(a) Guard:** Valid embedded inheritance, bare-checkpoint fallback, explicit matches, reordered-orbit rejection, and normal override disclosure are wired correctly. Findings 1 and 3 cover the malformed-authority/type bypass and false vanilla conflict. The intended `None` sentinel flow itself is sound.

- **(b) Load integrity:** No `--allow-partial-load`/`allow_partial_load` remains, and direct missing or non-whitelisted unexpected keys always raise. Finding 2 is the remaining silent path because EMA merging masks incompleteness before checking.

- **(c) Paths and records:** Negative suffixes are filesystem-safe; scientific notation is also syntactically safe. Precision, legacy-shape, and same-cardinality-orbit collisions remain per Finding 4. Within `eval_FLAC`, metrics and prediction metadata carry matching runtime/trained/source/override values.

- **(d) One-batch fake:** It genuinely exercises real rotation, real dispatch, real `invariant_conditioning`, ordering, rotated metadata, and conditioner call counts. It does **not** exercise real `get_conditioning_inputs` propagation—the fake returns `{}`—a real model, state-dict/EMA preprocessing, non-empty conditioning kwargs reaching sampling, default/bf16 autocast, metrics, or the `v`/`rf_denoiser` branches. Only `rectified_flow` runs, and patching `src.inference.sampling.sample_discrete_euler` is the correct target for the local import.

- **(e) `eval_pl`:** The guard is placed before model, dataloader, and Trainer construction. The early checkpoint assertion, full CPU load, and immediate `del` are coherent. Importing helpers from `eval_FLAC` currently introduces no cycle or import-time CLI execution. Findings 1 and 5 cover validation and record completeness.

- **(f) R4:** The current pre-R4 state is expected: `train.py` still hardcodes one million steps, SyncBN/gradient-checkpointing surfaces and `FLAC_AR_FA.json` are absent, and README/smoke/acceptance work remains. R4 must ensure the DINO revision is pinned in both relevant ViT blocks and, per D10, reconciled between `FLAC_AR.json` and the four-leaf-delta FA config. Run the smoke and pinned B-F@40k K={8,1} two-cell acceptance only after these R3 fixes and R4 land. None of these planned absences is a finding.

## Parity and diff commands run

```bash
git -C /home/yixunhu/codespace/ORBITRIR log --oneline bffe709..becaef7
git -C /home/yixunhu/codespace/ORBITRIR diff bffe709..becaef7
git -C /home/yixunhu/codespace/ORBITRIR diff --name-status bffe709..becaef7
git -C /home/yixunhu/codespace/ORBITRIR diff --stat bffe709..becaef7
git -C /home/yixunhu/codespace/ORBITRIR diff --check bffe709..becaef7

git show becaef7:eval_FLAC.py
git show becaef7:eval_pl.py
git show becaef7:src/data/yaw_rotation.py
git show becaef7:src/tests/test_eval_paths.py
git show becaef7:src/tests/test_protocol_guard.py

diff -u <(git -C /home/yixunhu/codespace/FLAC show f59f5a4:eval_FLAC.py) \
        <(git -C /home/yixunhu/codespace/ORBITRIR show becaef7:eval_FLAC.py)
diff -u <(git -C /home/yixunhu/codespace/FLAC show f59f5a4:eval_pl.py) \
        <(git -C /home/yixunhu/codespace/ORBITRIR show becaef7:eval_pl.py)
diff -u <(git -C /home/yixunhu/codespace/FLAC show f59f5a4:src/data/yaw_rotation.py) \
        <(git -C /home/yixunhu/codespace/ORBITRIR show becaef7:src/data/yaw_rotation.py)
diff -u <(git -C /home/yixunhu/codespace/FLAC show f59f5a4:src/tests/test_eval_paths.py) \
        <(git -C /home/yixunhu/codespace/ORBITRIR show becaef7:src/tests/test_eval_paths.py)

cmp --silent <(git -C /home/yixunhu/codespace/FLAC show 0bd5da0:eval_pl.py) \
             <(git -C /home/yixunhu/codespace/FLAC show f59f5a4:eval_pl.py)
git -C /home/yixunhu/codespace/FLAC diff --stat 0bd5da0..f59f5a4 \
    -- eval_FLAC.py eval_pl.py src/data/yaw_rotation.py
git -C /home/yixunhu/codespace/FLAC diff --name-status 0bd5da0..f59f5a4 \
    -- eval_FLAC.py eval_pl.py src/data/yaw_rotation.py
git -C /home/yixunhu/codespace/FLAC show 0bd5da0:eval_FLAC.py
git -C /home/yixunhu/codespace/FLAC show f59f5a4:eval_FLAC.py
```

The production parity differences were limited to the enumerated deviations. I did not execute Python or the test suite; the reported state remains 110 passed. The worktree remained clean, with `main` ahead of `origin/main` by the three reviewed commits.