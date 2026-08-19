**Reviewer:** OpenAI Codex `gpt-5.6-sol` at Extra High reasoning (codex-cli 0.147.0, `codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort=xhigh`) · **Date:** 2026-08-19

*Round marker: r4 (focused fix review / launch gate). Verdict REQUEST-CHANGES — launches remain held. Body verbatim.*

---

**Reviewer:** OpenAI Codex (GPT-5 API workspace agent, read-only review) · **Date:** 2026-08-19 · **Round:** r4

# Verdict: REQUEST-CHANGES

R-1/R0 are **not launch-cleared**. The candidate/control paths are materially stronger, but R0’s CUDA timings are not wall-correct—especially on the planned GPU 1—and the automated R-1 readback can pass after the corrected M=10 invariant or depth data has regressed.

## Part 1

| Checklist item | Status | Verification |
|---|---|---|
| F1 — frozen manifest | **RESOLVED** | Candidate nodes/coordinates are frozen once, hashed, and consumed from memory by `process_query`; no live per-query candidate enumeration remains. |
| F2 — reviewed entry points | **PARTIALLY** | All four entry points exist, but the probe timing and readback gates have correctness gaps described below. |
| F3 — context fail-open | **RESOLVED** | Context shapes/counts are checked before generation and full control evidence is required before publication. |
| F4 — registration gate | **PARTIALLY** | Locked fields match Rev 3.1 §4 and committed bytes are checked, but symbolic/unrelated commits are still accepted as a “SHA.” |
| F5 — cell names/overwrite | **PARTIALLY** | Primary run artifacts refuse clobbering, but the three new auxiliary modes unconditionally overwrite fixed filenames. |
| F6 — device provenance | **RESOLVED** | Requested device, resolved index, name, capability, and UUID are recorded from the requested CUDA index. |
| F7 — early validation | **PARTIALLY** | Numeric/frame-angle checks moved early, but registration presence is checked after checkpoint deserialization and context/output validation remains after model loading. |
| Part-1 partial — registration-SHA commit validation | **PARTIALLY** | A commit object containing byte-identical content is required, but `HEAD`, tags, abbreviated refs, descendants, and unrelated/dangling commits remain acceptable. |
| Part-1 partial — finite frame-average angles | **RESOLVED** | NaN/±Inf are rejected by argparse before file or model work. |

## New findings

1. **HIGH — R0’s GPU timing numbers are not wall-correct.**

   `_sync(device)` ignores `device` and calls `torch.cuda.synchronize()` without an index. For the explicitly planned GPU-1 R0, this may synchronize GPU 0 while work remains queued on GPU 1. In addition, scoring stops its timer without any synchronization; the later `.cpu()` performs the actual wait outside the scoring interval. There is also no leading synchronization, so outstanding context-control work from the preceding query can spill into the next query’s conditioning time. The reported total excludes `context_evidence`, despite that being part of every scored query. Peak-memory reset/read themselves use the correct explicit device and encompass the evaluation path. Anchors: [eval_localization.py:227](/home/yixunhu/codespace/FLAC/eval_localization.py:227), [eval_localization.py:308](/home/yixunhu/codespace/FLAC/eval_localization.py:308), [eval_localization.py:335](/home/yixunhu/codespace/FLAC/eval_localization.py:335), [eval_localization.py:1300](/home/yixunhu/codespace/FLAC/eval_localization.py:1300).

   **Fix:** synchronize the resolved device before and after each timed interval, including scoring; record context-control time and a separately synchronized whole-query wall time. Add a CUDA-index test that launches work on `cuda:1` and proves the measured interval waits for that device.

2. **HIGH — `--mode readback` does not enforce the corrected R-1 data invariants.**

   The mode reports candidate nodes but never requires 17 unseen rooms with M=10. In particular, deleting LRH’s metadata-only S10 would produce nine metadata nodes matching its nine WAV nodes and still pass—reintroducing the exact candidate-count error that Rev 3.1 corrected. Depth files are checked only for existence, never loaded or checked for `(256,512)`, dtype, or finiteness. Only one WAV per room is decoded, and it is checked for sample rate/nonempty/finiteness but not registered channel/length shape. The current integration test proves today’s snapshot, not the runtime gate. LRH’s known metadata-only source is correctly classified as a warning, but all such anomalies receive that classification. Anchors: [eval_localization.py:1837](/home/yixunhu/codespace/FLAC/eval_localization.py:1837), [eval_localization.py:1874](/home/yixunhu/codespace/FLAC/eval_localization.py:1874), [eval_localization.py:1877](/home/yixunhu/codespace/FLAC/eval_localization.py:1877), [test_eval_localization.py:2269](/home/yixunhu/codespace/FLAC/src/tests/test_eval_localization.py:2269).

   **Fix:** for the registered unseen split, machine-check the expected room/node map—M=10 everywhere and the explicit LRH S10 warning—then load and validate every referenced depth map and validate registered WAV shape on a defined readback sample or all split records.

3. **MEDIUM — The new diagnostic modes reintroduce silent overwrites.**

   Readback, scorer-noise, and reaggregate each open a fixed `{eval_name}_*.json` with `"w"` and ignore `--overwrite`. A failed R-1 can erase prior passing evidence, and different scorer-noise inputs or R1 row sets collide. The primary run stem also omits cell-defining fields such as model/dataset/checkpoint hashes, steps, CFG scale, and loader settings; it refuses rather than overwrites, but distinct cells cannot coexist. Anchors: [eval_localization.py:815](/home/yixunhu/codespace/FLAC/eval_localization.py:815), [eval_localization.py:1921](/home/yixunhu/codespace/FLAC/eval_localization.py:1921), [eval_localization.py:2032](/home/yixunhu/codespace/FLAC/eval_localization.py:2032), [eval_localization.py:2050](/home/yixunhu/codespace/FLAC/eval_localization.py:2050).

   **Fix:** use one atomic, no-clobber report writer for every mode and content-address auxiliary stems with dataset/scorer/input hashes and relevant protocol fields.

4. **MEDIUM — Registration verifies a committish, not an immutable registration SHA/ancestor.**

   Subprocess invocation is shell-safe: arguments are passed as arrays, and `git show` output is byte-compared, so encoding or newline drift correctly refuses. Detached HEAD works; a shallow clone works only when the named object is present and otherwise fails closed. However, `HEAD`, a branch/tag, an abbreviated ref, or an unrelated/dangling commit passes if it contains the bytes. No ancestry check is performed, so the pre-registration ordering requested by the full review is not established. The Rev 3.1 locked-field list is otherwise complete, including seeds through a separate check, and `"tbd"` is restricted correctly. Anchors: [eval_localization.py:1767](/home/yixunhu/codespace/FLAC/eval_localization.py:1767), [eval_localization.py:1780](/home/yixunhu/codespace/FLAC/eval_localization.py:1780), [eval_localization.py:1798](/home/yixunhu/codespace/FLAC/eval_localization.py:1798).

   **Fix:** resolve and record the full 40/64-hex object ID, require the supplied value to be an immutable SHA, require it to be an ancestor of the executing HEAD, and explicitly reject manifest paths outside the repository.

5. **MEDIUM — The “before model loads” ordering remains incomplete.**

   `main()` calls `load_and_validate_artifacts`, including `torch.load` of the checkpoint, before checking that registration flags exist. Full registration verification is before AGREE/model construction, but `resolve_context_k` and output-collision refusal happen only after both models and the dataloader are loaded. The test replaces `load_and_validate_artifacts` with a harmless lambda, so it does not cover the actual checkpoint-read ordering. Anchors: [eval_localization.py:1474](/home/yixunhu/codespace/FLAC/eval_localization.py:1474), [eval_localization.py:1549](/home/yixunhu/codespace/FLAC/eval_localization.py:1549), [test_eval_localization.py:2341](/home/yixunhu/codespace/FLAC/src/tests/test_eval_localization.py:2341).

   **Fix:** load dataset/config JSON first; validate registration, manifest hash, context configuration, and output targets; only then deserialize the checkpoint for ARE validation and construct either model.

6. **MEDIUM — Scorer-noise’s seen-split hold is bypassable and its draw stream is unreproduced.**

   The default file selection is correctly limited to a seen config, and pairwise-cosine and versus-mean statistics are mathematically correct. But explicit `--noise-wavs` are returned without checking membership in that seen split, allowing unseen RIRs while a seen config is supplied. The sampled draws are also taken from an unseeded global RNG, despite `--seed` existing, and the report records no draw seed. Anchors: [eval_localization.py:1947](/home/yixunhu/codespace/FLAC/eval_localization.py:1947), [eval_localization.py:1990](/home/yixunhu/codespace/FLAC/eval_localization.py:1990), [eval_localization.py:2015](/home/yixunhu/codespace/FLAC/eval_localization.py:2015).

   **Fix:** require explicit WAVs to resolve inside the configured seen split, seed immediately before sampled draws, and record that seed.

No further bug was found in reaggregation: K′ is the generation-order prefix, the largest supported K′ is selected, LME minimizes pooled mean dev error with smallest-τ tie-breaking, and the exact float codec has one definition. The AST test adequately prevents the exact “append functions after the guard” recurrence, although it could assert the precise comparator and `main()` call more strictly. The frozen manifest uses sorted-key JSON and deterministic Python float rendering for an identical manifest object; it is conservatively path-dependent because `dataset_root` is hashed. No r4 semantic change to conditioning, sampling, decoding, noise ordering, or waveform values was found; the added synchronizations are value-neutral, consistent with the reported exact parity rerun.

## Launch assessment

R-1/R0 remain held. Before launch, correct and re-review target-device/end-to-end timing, strengthen readback so the M=10/LRH and depth/WAV invariants are executable failures, and apply no-clobber semantics to every new mode; the registration ancestry/early-order and seen-only scorer-noise gaps should close in the same r4 loop. Then rerun the focused CPU tests plus target-GPU timing validation, retain the reported exact parity result, confirm the dataset is immutable, and commit the exact first-N R0 identities and unique output cells. Per the read-only constraint, I ran no pytest or GPU workload; static inspection and `git diff --check` were clean.