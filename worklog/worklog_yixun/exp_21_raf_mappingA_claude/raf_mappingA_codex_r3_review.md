# VERDICT: REQUEST-CHANGES

## Findings

1. **Q1 — High — P1 refusal is incomplete.**  
   [prepare_mappingA.py:524](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/prepare_mappingA.py:524) checks only rooms passed through `--rooms`. An `EmptyRoom`-only run therefore accepts `--output-dir <H>/FurnishedRoom` or a descendant. The transaction at [prepare_mappingA.py:1380](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/prepare_mappingA.py:1380) can then replace that Mapping-H room tree. A read-only extracted-function probe confirmed both paths are accepted. Derive protected rooms from Mapping-H’s publication pointer, not the A-run subset. Equal, ancestor, outside, selected-room, and pre-write ordering otherwise close correctly.

2. **Q2 — High — P3 still accepts mismatched evaluation cells.**  
   The identity tuple at [mappingA_stats.py:37](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/mappingA_stats.py:37) omits registered controls already recorded in the metrics artifact—`frame_avg_angles`, `frame_avg_fwd_cap`, `cond_autocast`, `batch_size`, execution/source provenance—despite their protocol significance documented at [eval_FLAC.py:1217](/home/yixunhu/codespace/exp-21-raf-mapping-a/eval_FLAC.py:1217). Thus BF seeds run under different C₄ angles, cap, precision, or batching can still pool.

   Cross-arm checking is narrower still: [mappingA_stats.py:42](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/mappingA_stats.py:42) compares only dataset digest, prepare generation, and input stream. It does not require shared `steps`, `cfg_scale`, `are_lambda`, or rotation protocol. A read-only probe confirmed `assert_paired` accepts otherwise matched 1-step and 8-step arms.

   Corpus identity is also incomplete: [RAF_A_md.py:57](/home/yixunhu/codespace/exp-21-raf-mapping-a/src/configs/dataset_configs/custom_metadata/RAF_A_md.py:57) records only the prepare generation even though the gate attests separate prepare and depth generations. A depth republish can therefore occur between arms without changing their recorded publication identity.

3. **Q3 — Medium — arm labels remain unaudited and can collapse provenance.**  
   The caller-supplied label remains accepted at [mappingA_stats.py:205](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/mappingA_stats.py:205). Two distinct arms may receive the same label; then the dictionaries at [mappingA_stats.py:347](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/mappingA_stats.py:347) overwrite one arm’s identity and seed-variability entry. Require distinct labels and retain identities positionally; registered P1/YAW/BF labels additionally need an explicit identity-to-label registry if those names are asserted.

## Verified closures

- **P2:** fail-closed assignment requirement, target/context row enforcement, finite match fields, and publication-contained offline assignments are present.
- **N1 residual:** the canonical renderer marker is consumed untouched and byte-checked in [test_mappingA_render.py:257](/home/yixunhu/codespace/exp-21-raf-mapping-a/src/tests/test_mappingA_render.py:257).
- Registered deviations are sound: `+inf` margin semantics, metadata-absence renderer guard, per-eval hashing, and pinned seeds 42–46.

All changed Python files parsed successfully. Full pytest was not rerun because the available interpreter lacks pytest/numpy/librosa and installs/writes were prohibited. No files were changed.