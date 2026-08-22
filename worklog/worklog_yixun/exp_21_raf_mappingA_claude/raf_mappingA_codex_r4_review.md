# VERDICT: REQUEST-CHANGES

## Finding

**R1 — High — Q1’s empty-protected-list refusal is bypassed for the derived output path.**

[prepare_mappingA.py:516](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/prepare_mappingA.py:516) returns `<H>/mappingA` before the empty-list refusal at line 533 or any protected-room check runs. Read-only probes confirmed:

- `resolve_output_dir(<H>/FurnishedRoom, H, [EmptyRoom, FurnishedRoom])` refuses correctly.
- `resolve_output_dir(<H>/mappingA, H, [])` refuses correctly.
- `resolve_output_dir(None, H, [])` incorrectly accepts `<H>/mappingA`.
- `resolve_output_dir(None, H, ["mappingA"])` also accepts, deriving an output exactly atop the protected room.

The regression at [test_mappingA_prepare.py:1075](/home/yixunhu/codespace/exp-21-raf-mapping-a/src/tests/test_mappingA_prepare.py:1075) covers only an explicit output path. Move the empty-list refusal before the default-path return and pass the derived path through the same room-disjointness checks.

## Verified closures

- **Q2 closed:** both publication generations are recorded and required; all 19 identity fields are held constant within an arm; all 13 shared fields—including fatal `source_sha`—refuse cross-arm mismatches. The six implemented contrast fields, including `cond_method`, remain per-arm-free as Amendment 3.1 requires.
- **Q3 closed:** archived P1/BF/YAW/BV digests match the registry; the live RAF finetune checkpoint hashes exactly to `6dfc2b2e…`; wrong registered-label assertions and duplicate arm labels refuse; `arm_identities` is positional.
- No other r4 regression found. All seven changed Python files parsed successfully and the diff passes `git diff --check`.

No files were changed. Full pytest was unavailable without NumPy and would also violate the strict no-write constraint through temporary fixtures.