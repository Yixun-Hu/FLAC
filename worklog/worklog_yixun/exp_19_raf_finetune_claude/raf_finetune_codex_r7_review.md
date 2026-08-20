REQUEST-CHANGES

1. **BLOCKER — the registered ×3 cannot be derived from the registered trained-only population.**  
   [prepare_data.py:1141](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/prepare_data.py:1141), [prepare_data.py:898](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/prepare_data.py:898)  
   The committed audit shows trained maxima of 0.15027 and 0.16574; 0.24990 belongs to test capture `025937`, not training. Consequently the implementation derives ×5, then correctly aborts when that test WAV reaches 1.249524, as confirmed by [prep_canonical2_20260820_150834.log:8](/home/yixunhu/codespace/exp-19-raf-finetune/worklog/worklog_yixun/exp_19_raf_finetune_claude/prep_canonical2_20260820_150834.log:8). No code can simultaneously satisfy “trained supports only,” the stated formula, and registered ×3 with these data.  
   **Fix:** re-adjudicate the measured basis/formula. Do not use `025937` while claiming trained-only derivation. Once reconciled, assert the canonical scalar equals the registered result before writing WAVs.

2. **HIGH — derivation is per-room, not one global scalar.**  
   [prepare_data.py:1117](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/prepare_data.py:1117), [prepare_data.py:1142](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/prepare_data.py:1142), [prepare_data.py:1217](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/prepare_data.py:1217)  
   `derive_amplitude_scalar` is called independently inside the room loop. If room results differ, WAVs receive different scalars and the marker stores an unlabelled sorted list, directly contradicting Amendment 9’s one global scalar.  
   **Fix:** collect room-qualified trained IDs for both rooms first, derive once from their global maximum, and pass that same scalar/provenance to every room. Add a two-room unequal-peak test.

3. **HIGH — the marker identity omits the required derivation provenance.**  
   [prepare_data.py:1213](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/prepare_data.py:1213), [publish.py:338](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/publish.py:338)  
   The audit and splits record contain derivation hashes, but the prepare marker’s canonical parameter identity contains only ceiling and scalar. Its consumer therefore cannot verify that the registered trained-ID union produced the scalar. Manifest hashing merely binds the producer’s claim; it does not validate that claim against the registered derivation identity.  
   **Fix:** add the room-qualified global derivation-set hash—and preferably formula/version and count—to the marker and `CANONICAL_PREPARE_PARAMS`.

4. **HIGH — canonical scale QA remains bypassable by an absent or mistyped reference root.**  
   [render_depth.py:848](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/render_depth.py:848), [render_depth.py:902](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/render_depth.py:902), [render_depth.py:831](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/render_depth.py:831)  
   When neither HAA nor AR is readable, `scale_checked` is false and `(plausible or not scale_checked)` passes. The reference path/availability is not part of canonical identity. Thus an operator path error disables one of Amendment 9’s remaining hard gauge gates while still allowing canonical publication.  
   **Fix:** require the pinned HAA reference to be available in canonical mode, bind its fingerprint in render identity, and require both `scale_checked` and `scale_plausible`.

5. **MEDIUM — the added tests overstate what the 525-test suite proves.**  
   [test_raf_readback.py:898](/home/yixunhu/codespace/exp-19-raf-finetune/src/tests/test_raf_readback.py:898), [test_raf_readback.py:975](/home/yixunhu/codespace/exp-19-raf-finetune/src/tests/test_raf_readback.py:975), [test_raf_readback.py:1003](/home/yixunhu/codespace/exp-19-raf-finetune/src/tests/test_raf_readback.py:1003), [test_raf_render_depth.py:884](/home/yixunhu/codespace/exp-19-raf-finetune/src/tests/test_raf_render_depth.py:884)  
   Tests use one room and homogeneous synthetic peaks, so they miss both critical scalar failures. The republication test compares two generations produced by the same patched code, not against the committed pre-Amendment generation; a deterministic split change would pass. The “hard gates” test asserts only vertical and containment, despite claiming bounds and scale too.  
   **Fix:** add global two-room and loud-nontrain cases, pin the four committed split-file hashes, test missing canonical scale references, and add negative bounds/scale assertions.

Verified: scaling occurs once per selected WAV from the raw `archived` source, so normal reruns do not compound it; context and targets read the same scaled files; clipping and −60 dB checks fail closed. The four tracked split JSONs are currently byte-identical between `fee0981` and `bc8bc9d`, but the new test does not protect that baseline.

I did not execute pytest because the strict no-write constraint excludes its temporary-file activity; review was static plus existing artifact/log evidence.