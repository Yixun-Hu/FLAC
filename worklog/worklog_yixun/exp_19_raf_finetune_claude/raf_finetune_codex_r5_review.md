REQUEST-CHANGES

1. **CRITICAL** — [RAF_md.py:193](/home/yixunhu/codespace/exp-19-raf-finetune/src/configs/dataset_configs/custom_metadata/RAF_md.py:193), [RAF_md.py:203](/home/yixunhu/codespace/exp-19-raf-finetune/src/configs/dataset_configs/custom_metadata/RAF_md.py:203)  
   **Problem:** The pointer does not close the authentication loop. The verifier never requires `pointer["output_dir"]` to equal the runtime root being consumed, so a stale/copied pointer at tree A can redirect verification to valid tree B while `RAF_md` subsequently loads A. It also passes `canonical=bool(pointer["canonical"])`, allowing a registered RAF config to accept a non-canonical publication. The positive test actually exercises this fail-open case with `canonical=False` at [test_raf_md.py:622](/home/yixunhu/codespace/exp-19-raf-finetune/src/tests/test_raf_md.py:622). The pointer is included in the prepare manifest, but currently the verifier can authenticate B’s manifest rather than A’s.  
   **Fix:** In production mode require `canonical is True`, require the resolved/same-file `output_dir` to be the `dataset_folder` being loaded, and always invoke combined verification with `canonical=True`. Add negative tests for a non-canonical pointer and a copied/relocated pointer targeting another valid tree.

2. **HIGH** — [RAF_md.py:31](/home/yixunhu/codespace/exp-19-raf-finetune/src/configs/dataset_configs/custom_metadata/RAF_md.py:31), [dataset.py:335](/home/yixunhu/codespace/exp-19-raf-finetune/src/data/dataset.py:335), [dataset.py:358](/home/yixunhu/codespace/exp-19-raf-finetune/src/data/dataset.py:358)  
   **Problem:** Publication verification still runs inside `SampleDataset.__getitem__`’s catch-all. Its distinctive `ValueError` is swallowed and replaced with recursive random substitution. A RAF-only config eventually surfaces an unrelated `RecursionError`; a mixed dataset can silently substitute another dataset’s item. The r5 tests call `get_custom_metadata` directly, so they do not verify the production failure path.  
   **Fix:** Preflight RAF publication during dataloader/dataset construction, outside the substitution handler, or raise a dedicated exception that `SampleDataset` explicitly re-raises. Test the real `create_dataloader_from_config`/`SampleDataset` path with an invalid pointer.

3. **HIGH** — [publish.py:376](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/publish.py:376), [RAF_md.py:210](/home/yixunhu/codespace/exp-19-raf-finetune/src/configs/dataset_configs/custom_metadata/RAF_md.py:210), [test_raf_publish.py:687](/home/yixunhu/codespace/exp-19-raf-finetune/src/tests/test_raf_publish.py:687)  
   **Problem:** “Marker provenance validation” trusts the producer’s `canonical_parameters: true` boolean. Neither exact prepare/render parameter dictionaries nor the markers’ readback digests are validated. The new positive oracle explicitly accepts canonical markers containing no `parameters` or `readback_record` at all. Thus Amendment 7’s registered identities are not consumer-verified.  
   **Fix:** Compare each marker’s complete parameter payload against its exact per-kind registered identity and require the pinned digest in both markers and the pointer. Add independent negative tests for missing and altered parameters/digests.

4. **MEDIUM** — [render_depth.py:346](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/render_depth.py:346), [render_depth.py:817](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/render_depth.py:817)  
   **Problem:** Canonical render identity remains incomplete. A lower `--max-miss-rate` is deliberately accepted while publishing `canonical:true`, despite Amendment 7 registering `DEFAULT_MAX_MISS_RATE`; the new test locks in that deviation at [test_raf_render_depth.py:1253](/home/yixunhu/codespace/exp-19-raf-finetune/src/tests/test_raf_render_depth.py:1253). Also, `--rx-sightline-receivers` is parsed but never used, recorded, or identity-checked. Prepare’s effective canonical flags are otherwise covered; `--crosscheck-sample` is inert when mandatory full crosscheck is active.  
   **Fix:** Enforce the Amendment 7 miss-cap identity exactly. Either remove the unused receiver-count flag or wire it through `real_mesh_qa`/`rx_sightline_check`, record it, and reject non-default canonical values.

5. **HIGH** — [render_depth.py:282](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/render_depth.py:282), [render_depth.py:316](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/render_depth.py:316), [render_depth.py:430](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/render_depth.py:430)  
   **Problem:** Mask-derived QA remains optional: `depth_qa(..., miss_report=None)` leaves `misses=None`, and `misses is None or misses["audit_ok"]` passes. The production CLI currently supplies the mask, but the mandatory QA contract itself is still fail-open. Additionally, the declared `miss_rate` is never checked against or replaced by the mask-derived rate, despite the r5 docstring requiring agreement.  
   **Fix:** Require a miss report whose `mask_verified is True` for every passing QA result, including zero misses. Serialize authoritative mask-derived count/rate/hash values and reject conflicting declarations. Add a negative test for an entirely absent report, not merely a report with `miss_mask` removed.

Verified closed: `_RAF_MD_TEST_MODE` is not reachable from the registered JSON config path; raw RAF Y is published from untransformed `g["tx_xyz"][1]`, hard-required by the CLI, and the candidate-gauge test does call `render_depth.main`; no transformed height reaches production `real_mesh_qa`. The bearing/vertical delta shadowing is also closed. No metric implementation or AR/HAA oracle was changed in r5, but the new publication/identity tests above encode self-declared booleans rather than independent identities.

- **RESIDUAL-1 — MEDIUM** — [readback_audit.py:373](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/readback_audit.py:373)  
  **Problem:** The 43-GB audio corpus remains outside content binding.  
  **Fix:** Optional future full or sampled audio-hash manifest.

- **RESIDUAL-2 — LOW** — [readback_audit.py:366](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/readback_audit.py:366), [RAF_md.py:191](/home/yixunhu/codespace/exp-19-raf-finetune/src/configs/dataset_configs/custom_metadata/RAF_md.py:191)  
  **Problem:** No signing, inode/fstat hardening, or per-item rehashing protects against a malicious local actor.  
  **Fix:** Optional hardening if the threat model expands.

- **RESIDUAL-3 — MEDIUM** — [render_depth.py:92](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/render_depth.py:92)  
  **Problem:** A globally consistent horizontal permutation/chirality remains render-undetectable.  
  **Fix:** Obtain an independently surveyed landmark or compass bearing.

The archived “486 passed” run was not re-executed because strict read-only mode precludes pytest’s temporary/cache writes; the committed implementations and oracles were independently inspected. The worktree remained unchanged.