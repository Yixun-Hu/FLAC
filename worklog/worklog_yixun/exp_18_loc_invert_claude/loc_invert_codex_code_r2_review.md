**Reviewer:** OpenAI Codex `gpt-5.6-sol` at Extra High reasoning (codex-cli 0.147.0, `codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort=xhigh`) · **Date:** 2026-08-19

*Round marker: r2 (agree_embed — registered scorer readout). Reviewed at HEAD `56664b1`, commits `56d321a`…`ec496f3`.*

---

**Verdict: APPROVE-WITH-CHANGES**

No BLOCKER/HIGH defect was found in the registered readout itself. The MEDIUM test/fail-closed gaps should be addressed before r2 closes.

1. **MEDIUM — Integration tests silently skip when invoked from the wrong CWD.**  
   Asset detection uses CWD-relative paths at [test_loc_agree_embed.py:296](/home/yixunhu/codespace/FLAC/src/tests/test_loc_agree_embed.py:296). Thus a checkout with all assets can report four skips instead of exercising the intended CWD guard.  
   **Fix:** detect assets using a repo-root path derived from `__file__`; when assets exist but CWD is wrong, explicitly fail or allow `load_agree_audio` to raise. Missing assets may still skip.

2. **MEDIUM — The registered real-model batch-invariance and CUDA RNG checks are absent.**  
   Batch invariance is tested only on the stub at [test_loc_agree_embed.py:225](/home/yixunhu/codespace/FLAC/src/tests/test_loc_agree_embed.py:225); the real integration test merely repeats the same B=2 call at [test_loc_agree_embed.py:334](/home/yixunhu/codespace/FLAC/src/tests/test_loc_agree_embed.py:334). RNG tests snapshot only the CPU generator at [test_loc_agree_embed.py:352](/home/yixunhu/codespace/FLAC/src/tests/test_loc_agree_embed.py:352).  
   **Fix:** add the contracted real-model B=8 versus eight B=1 comparison, plus a CUDA-conditional test that snapshots CPU and `torch.cuda.get_rng_state_all()` around mean readout. It should also show that the CUDA sampled path changes the target CUDA generator.

3. **MEDIUM — The configurable CWD guard can validate a different config from the model actually loaded.**  
   [agree_embed.py:131](/home/yixunhu/codespace/FLAC/src/localization/agree_embed.py:131) reads `config_name`, but the reused loader at `metric_callback.py:434` always constructs `dinoV3`. Passing another valid config therefore checks the wrong `pretrained` path. Absolute nonexistent paths also bypass the early guard at line 136.  
   **Fix:** remove the parameter or refuse anything except `AGREE_CONFIG_NAME`, and require `os.path.isfile(pretrained)` for both relative and absolute paths. Add a valid-but-non-dino config regression test.

4. **NIT — Train-mode refusal checks only the parent module and is untested.**  
   [agree_embed.py:77](/home/yixunhu/codespace/FLAC/src/localization/agree_embed.py:77) accepts `model.eval(); model.audio.train()` because the parent remains `training=False`. No test covers the advertised refusal.  
   **Fix:** reject if any scorer/audio submodule is training, and add that child-train regression.

5. **NIT — The preprocessing parity fixture copies the dependency expressions instead of traversing them.**  
   [test_loc_agree_embed.py:71](/home/yixunhu/codespace/FLAC/src/tests/test_loc_agree_embed.py:71) correctly mirrors today’s route, but would remain green if `MetricCallback` or `Retrieval` later changed.  
   **Fix:** add a lightweight spy/fake AGREE test that actually invokes the callback slice and `Retrieval.compute_audio_features`.

### What the round gets right

The registered mean readout at [agree_embed.py:82](/home/yixunhu/codespace/FLAC/src/localization/agree_embed.py:82) is faithful: `layers → chunk(2, dim=1)[0] → flatten → project → float32 L2-normalize`. The first chunk is non-contiguous for B>1, but `reshape` preserves its logical NCT ordering; I verified it is exactly equal to the stock sampled tensor’s contiguous `view` layout. The real checkpoint’s audio parameters are all float32, so moving `.float()` before normalization introduces no actual dtype drift. Its SHA-256 is exactly `b664d5c09f74685fc9121f4b1496642d601489a338769fe52161a0b7912c72f4`.

Preprocessing also matches the real flow: observed RIRs are already clamped by `dataset.py:303`; generated RIRs are clamped by `eval_FLAC.py:1313`; both reach the first-8000 slice and then Retrieval’s 10,240 padding. Intermediate right-padding for shape alignment cannot affect the retained first 8,000 samples.

The mean call graph contains no RNG operation; only `VAEBottleneck → vae_sample → randn_like` samples, and mean bypasses it. The near-zero-stdev test alone would not distinguish accidental sampling at `atol=1e-3`, but the CPU RNG-state test and deterministic large-stdev repeat do, so the stub suite collectively discriminates correctly. B=1 and exact 8,000/10,240 boundaries are covered; float64 input is covered, while integer input remains a low-value unpinned edge.

I could not independently rerun pytest because the enforced read-only environment provides no writable temporary directory; pytest failed before collection. Static checks, checkpoint inspection, SHA verification, and layout arithmetic were read-only.