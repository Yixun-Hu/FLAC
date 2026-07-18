Overall verdict: **REQUEST-CHANGES**.

### Blocking finding

- **P0 — checkpointing is a no-op with the pinned dependency.** [conditioners.py:205](/home/yixunhu/codespace/FLAC/src/models/conditioners.py:205) calls the correct HF API, but [`transformers==4.57.0`](https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/dinov3_vit/modeling_dinov3_vit.py#L487-L530) merely sets `gradient_checkpointing=True`; DINOv3’s forward loop calls each layer directly and never invokes `_gradient_checkpointing_func`. Thus tests see `is_gradient_checkpointing=True` while memory use is unchanged; micro-32 will still OOM.

### Per-file verdicts

- **REQUEST-CHANGES — [src/models/conditioners.py:205](/home/yixunhu/codespace/FLAC/src/models/conditioners.py:205).** Make checkpointing real: upgrade/pin a verified Transformers release whose DINOv3 layers actually checkpoint, or add an explicit layer-checkpoint adapter. The present “fail-closed” check only verifies API presence and therefore fails open for this backbone.

- **REQUEST-CHANGES — [src/tests/test_vit_gradient_checkpointing.py:188](/home/yixunhu/codespace/FLAC/src/tests/test_vit_gradient_checkpointing.py:188).** Lines 188–204 test flags/partial metadata, not checkpoint execution. Lines 244–264 prove only initialization identity, not gradient identity. Add a forward/backward test that records actual checkpoint invocation/recomputation and compares ON/OFF parameter gradients; retain the exact `use_reentrant=False` call-site test at lines 207–237. A two-rank full-stack fit/memory probe remains mandatory.

- **SHIP — [FLAC_AR_BF.json:87](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json:87) and [line 103](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json:103).** Exact diff verified: two checkpointing keys plus `cond_method` and `frame_avg_angles`. Factory forwarding is valid and confined to `ViTCoordinates`.

### Remaining checks

- Once checkpointing is real, non-reentrant checkpointing is supported with DDP, `find_unused_parameters=True`, and repeated checkpointing without the reentrant limitations ([PyTorch DDP contract](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)).
- Double enable on the shared backbone is idempotent; repeated forwards contribute normally to one backward.
- SyncBN is on the separately run audio conditioner, not inside checkpointed DINO layers. EMA tracks `diffusion.model`, not the conditioner. CFG dropout occurs after conditioning.
- Eval calls `module.eval()` at [eval_FLAC.py:212](/home/yixunhu/codespace/FLAC/eval_FLAC.py:212); a correct HF implementation bypasses checkpointing in eval. No inference numerics change or warning expected. The online-eval copy does not need the key.
- JSON parsing and `diff --check` passed. Focused pytest could not execute in this read-only sandbox because PyTorch/pytest require a writable temporary directory.