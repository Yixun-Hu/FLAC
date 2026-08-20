REQUEST-CHANGES

1. **T3 — CRITICAL** — [RAF_md.py:113](/home/yixunhu/codespace/exp-19-raf-finetune/src/configs/dataset_configs/custom_metadata/RAF_md.py:113), [RAF_md.py:158](/home/yixunhu/codespace/exp-19-raf-finetune/src/configs/dataset_configs/custom_metadata/RAF_md.py:158), [prepare_data.py:1035](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/prepare_data.py:1035).  
   **Problem:** Publication verification is silently disabled unless `RAF_REQUIRE_PUBLICATION=1`; no artifact records whether it ran, and no canonical run script sets it. When enabled, it looks for the prepare marker at the runtime root, while production writes that marker under `split_dir`, so a real canonical publication cannot pass. It also verifies only the prepare marker—not combined prepare+depth—and never checks canonical status, rooms, parameters, or pinned digest.  
   **Fix:** Make verification mandatory for RAF canonical configs, call `verify_combined_publication` using the actual split/runtime roots, and validate exact canonical roots/rooms, `"canonical": true`, registered parameters, and digest `9288181b…`. Use an explicit test-only opt-out if needed.

2. **T2 — HIGH** — [prepare_data.py:371](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/prepare_data.py:371), [render_depth.py:773](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/render_depth.py:773), [render_depth.py:863](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/render_depth.py:863).  
   **Problem:** Canonical preparation omits registered `seed=0` from both validation and marker identity; an in-memory probe with `seed=999` returned no deviations. More seriously, canonical rendering does not enforce the two-room set and permits arbitrary `--floor-tol`, so a Furnished-only render or a tolerance that disables the vertical gate can still publish `"canonical": true`.  
   **Fix:** Define and enforce complete canonical parameter identities for both commands before I/O, including seed, exact rooms, and the registered floor tolerance; bind them to their markers.

3. **T4 — HIGH** — [publish.py:267](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/publish.py:267), [publish.py:325](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/publish.py:325).  
   **Problem:** Empty roots and transaction-produced duplicates are rejected, but combined completeness is defined by the caller-supplied `rooms`. Passing only `["EmptyRoom"]` makes a one-room depth marker “complete”; the included test explicitly accepts this. Expected-root lists are also reduced to sets, hiding duplicate expectations.  
   **Fix:** Make canonical combined verification internally require exactly both registered rooms and roots, reject duplicate/empty caller inputs, and validate marker provenance rather than only manifests.

4. **T7 — HIGH** — [render_depth.py:377](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/render_depth.py:377), [render_depth.py:424](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/render_depth.py:424).  
   **Problem:** The raw miss mask remains optional. If absent, `mask_verified=None`, and `audit_ok` accepts that because it only rejects `False`. A forged/stale zero-miss report with empty coordinates, the public empty-set hash, and the correct ray count still passes `depth_qa`; the hostile probe returned `passed=True`.  
   **Fix:** Require raw-mask verification for QA (`mask_verified is True`) and derive count, coordinates, rate, and hash directly from that mask.

5. **T5 — HIGH** — [prepare_data.py:960](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/prepare_data.py:960), [render_depth.py:845](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/render_depth.py:845), [render_depth.py:863](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/render_depth.py:863).  
   **Problem:** The CLI supplies `position[HEIGHT_AXIS]`—already transformed by the candidate gauge—as the allegedly independent tracked RAF height. Raw RAF Y is not retained in runtime metadata. A consistently wrong identity gauge passed the vertical check in a zero-origin box because nadir and the substituted horizontal coordinate were both 2 m. The unit test passes raw `1.5` manually and therefore does not exercise production wiring.  
   **Fix:** Persist the raw RAF Y height, bind it to the pose digest, and pass that independent value to `real_mesh_qa`; add an end-to-end candidate-gauge test through the CLI wiring.

6. **New r4 defect — LOW** — [render_depth.py:616](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/render_depth.py:616), [render_depth.py:658](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/render_depth.py:658), [render_depth.py:708](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/render_depth.py:708).  
   **Problem:** `delta` initially holds bearing degrees, then is overwritten with vertical metres. Canonical QA therefore writes the vertical delta into `bearing_delta_deg` and any bearing warning.  
   **Fix:** Use separate `bearing_delta_deg` and `vertical_delta_m` variables.

Verified closed without further findings: T1’s pinned record is exactly `9288181be62…`, and both recorded pose-file digests match the mounted corpus; T8 restores the pre-S4 AR/HAA whole-batch repetition while keeping RAF per-item, with unequal-update oracles implemented outside the callback. T6, T9, and the substantive T10 text corrections are also closed.

- **RESIDUAL-1 — MEDIUM** — [readback_audit.py:373](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/readback_audit.py:373).  
  **Problem:** The 43-GB audio corpus is not content-bound; only pose indexes and counts are. This is explicitly outside Amendment 6.  
  **Fix:** Optional future full/selected-audio hash manifest.

- **RESIDUAL-2 — LOW** — [readback_audit.py:366](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/readback_audit.py:366), [RAF_md.py:170](/home/yixunhu/codespace/exp-19-raf-finetune/src/configs/dataset_configs/custom_metadata/RAF_md.py:170).  
  **Problem:** No inode/fstat hardening, signing, or per-item rehashing protects against a malicious local actor.  
  **Fix:** Optional descriptor metadata checks, signed markers, and per-item hashes if the threat model expands.

- **RESIDUAL-3 — MEDIUM** — [render_depth.py:92](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/render_depth.py:92).  
  **Problem:** A globally consistent horizontal permutation/chirality remains render-undetectable and is pinned only by derivation.  
  **Fix:** Acquire an independently surveyed landmark or compass bearing.