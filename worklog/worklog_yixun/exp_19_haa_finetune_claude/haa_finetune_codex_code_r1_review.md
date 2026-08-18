**Reviewer:** OpenAI Codex (`gpt-5.6-sol`, xhigh, read-only sandbox) · **Date:** 2026-08-17 · **Round:** exp_19 r1 (extraction tool + arm configs + R1 probe)

Reviewed at HEAD `bc55a3e`. Reviewer independently re-ran the extraction+config suites (54 passed; probe suite uncollectable in its py3.13 env — torchaudio absent, env untouched).

---

Overall verdict: **REQUEST-CHANGES**. The configs are sound, but the initialization and FA gate do not yet provide the required fail-closed assurance.

## Blocking findings

1. **The extraction round-trip test is self-fulfilling, not faithful to `train.py:148`.**  
   [test_extract_ema_weights.py:146](/home/yixunhu/codespace/FLAC/src/tests/test_extract_ema_weights.py:146) constructs a module directly from the extracted weights’ own keys, shapes, and dtypes, then loads those same weights at [line 260](/home/yixunhu/codespace/FLAC/src/tests/test_extract_ema_weights.py:260). Almost any internally consistent but incorrect state dict passes. The approved plan required `create_model_from_config(config)` and a strict load into that independent real target. The real-artifact tests at [line 465](/home/yixunhu/codespace/FLAC/src/tests/test_extract_ema_weights.py:465) compare key sets only, not shapes, dtypes, or tensor values.

2. **EMA/live compatibility checks only names, allowing silent dtype conversion.**  
   [extract_ema_weights.py:177](/home/yixunhu/codespace/FLAC/src/tools/extract_ema_weights.py:177) checks only symmetric key difference. An EMA tensor with the correct key but wrong dtype passes extraction; PyTorch strict loading is key/shape-strict but can cast source tensors into the target dtype. That can create a silently wrong initialization. Validate that both values are tensors and that shape, dtype, and layout match the corresponding live DiT tensor before substitution. CPU placement is handled correctly by `map_location="cpu"`, and `weights_only=True` is correctly explicit.

3. **“Never overwrite” is not atomic.**  
   [extract_ema_weights.py:127](/home/yixunhu/codespace/FLAC/src/tools/extract_ema_weights.py:127) checks existence, but [line 194](/home/yixunhu/codespace/FLAC/src/tools/extract_ema_weights.py:194) later opens the destination through `torch.save`, which can overwrite a file created between those operations. The same TOCTOU window applies after the same-file check. Use exclusive destination creation or a same-directory temporary file followed by a no-replace publication operation.

4. **The real FA gate uses the same rotation implementation as both subject and oracle.**  
   The outer orbit uses `rotate_scene_metadata` at [probe:192](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/probe_haa_fa_invariance.py:192); the inner `invariant_conditioning` uses that same primitive. Consequently, a shared sign/convention error remains algebraically group-invariant and passes. The wrong-sign test is numerically non-vacuous, but only because its test-only inner average deliberately uses a different transform at [test_probe:230](/home/yixunhu/codespace/FLAC/src/tests/test_probe_haa_fa_invariance.py:230). It does not demonstrate that the CLI can catch a shared production error.

5. **The CLI does not load the actual arm initialization’s conditioner weights.**  
   `_build_stack` creates a fresh model at [probe:264](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/probe_haa_fa_invariance.py:264), but there is no checkpoint argument or strict checkpoint load. Therefore the gate probes a newly constructed stack, not the conditioner tensors carried into HAA-BF. Because detection of the documented pose-linear blind spot depends on learned nonlinear panorama/pose coupling, this distinction is material. The gate should load the extracted BF init through the real consumer path before measuring.

## Non-blocking findings

- Loss/discriminator classification uses substring matching at [extract_ema_weights.py:154](/home/yixunhu/codespace/FLAC/src/tools/extract_ema_weights.py:154) and [line 158](/home/yixunhu/codespace/FLAC/src/tools/extract_ema_weights.py:158). A future legitimate conditioner/pretransform key containing `losses` or `discriminator` would be silently dropped. Restrict dropping to known namespaces or validate the final result against the real target model.

- The YAW test duplicates exp_17’s insertion literal at [test_exp19_haa_arm_configs.py:75](/home/yixunhu/codespace/FLAC/src/tests/test_exp19_haa_arm_configs.py:75) instead of deriving it from exp_17’s config/test. The current block does match exp_17, but source drift would not be detected.

- The CLI returns only each condition entry’s tensor and discards its mask at [probe:307](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/probe_haa_fa_invariance.py:307), despite the core supporting full `[tensor, mask]` structures. Masks appear structurally invariant today, but measuring the full real output would better match the stated contract.

## Config assessment

- **BF config: SHIP.** It is stock plus exactly `cond_method: fa_invariant` and float angles `[0, 90, 180, 270]` at [BF config:170](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/FLAC_HAA_finetune_BF.json:170). No fourth delta.
- **YAW config: SHIP.** It is stock plus exactly the exp_17 block at [YAW config:170](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/FLAC_HAA_finetune_YAW.json:170). No FA contamination.
- The stock SHA is correct: `3639a9face84d13bcbb8f4472e78970c8e045952337f11b4f77d8798f786ba80`.
- `strict_diff` is genuinely type-strict at [test_exp19_haa_arm_configs.py:128](/home/yixunhu/codespace/FLAC/src/tests/test_exp19_haa_arm_configs.py:128).

## Test non-vacuity gaps

Tests that remain green if their corresponding assurance is removed or bypassed include:

- Extraction “round trip”: deleting the actual `model.load_state_dict` call still leaves its key/value assertions green; more importantly, the generated target cannot detect wrong target shapes/dtypes.
- `_detached` behavior, explicit `weights_only=True`, CPU mapping, exclusive creation, nested output creation, and non-tensor state entries have no direct mutation guards.
- `test_byte_comparison_would_catch_newline_drift` remains green if the actual arm-file equality tests are removed; it only compares two in-memory byte strings.
- The YAW treatment tests remain green if exp_17 changes because they rely on a copied literal.
- All probe core tests remain green if the CLI stops loading or invoking the real stack. There is no `main()` test with a fake stack proving build → real conditioning → measurement → refusal/pass wiring.
- The panorama-validity and invariant-field tests call `yaw_rotation` directly and remain green if the probe stops using those semantics.
- The pose-linear and wrong-sign tests genuinely exercise the mathematical core, but not the production CLI’s shared-transform blind spot.

## Per-file verdicts

| File | Verdict |
|---|---|
| `src/tools/extract_ema_weights.py` | **REQUEST-CHANGES** |
| `src/tests/test_extract_ema_weights.py` | **REQUEST-CHANGES** |
| `FLAC_HAA_finetune_BF.json` | **SHIP** |
| `FLAC_HAA_finetune_YAW.json` | **SHIP** |
| `src/tests/test_exp19_haa_arm_configs.py` | **SHIP**, with non-blocking YAW-source hardening |
| `probe_haa_fa_invariance.py` | **REQUEST-CHANGES** |
| `src/tests/test_probe_haa_fa_invariance.py` | **REQUEST-CHANGES** |

Fresh verification: the extraction and config suites passed, **54 passed**. The probe suite could not collect in the current Python 3.13 environment because `torchaudio` is absent; I did not alter the environment. The checked-in Python 3.10 run records all 80 tests passing, but that does not resolve the coverage gaps above.
