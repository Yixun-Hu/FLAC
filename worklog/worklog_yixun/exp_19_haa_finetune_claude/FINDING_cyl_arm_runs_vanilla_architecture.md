# FINDING (2026-08-21, exp-12 session): the CYL HAA arm is running a VANILLA architecture

*Left untracked deliberately — committing could break an armed EXPECT_SHA eval chain.
From the Fable 5 seat working the CYL-SSL handoff. Evidence below is reproducible.*

## The defect

Branch `exp17-yawaug-scratch`'s `src/models/conditioners.py` has **no
`cylindrical_dinov3` branch** (`git grep cylindrical_dinov3 -- src/models/conditioners.py`
→ 0 hits). The `ViTCoordinates` builder falls through to plain
`AutoModel.from_pretrained(...)`, silently ignoring the config keys
`implementation: cylindrical_dinov3` and `gauge: cylindrical_xyz`. The exp-09/exp-12
cylindrical construction lives only on those experiment branches; it was never merged
into this lineage.

**The live CYL FULL run (launched 10:31, pid 89597) proves it**: its train log prints

```
Loading ViT model from facebook/dinov3-vits16-pretrain-lvd1689m...
```

— the vanilla banner (1 occurrence; the cylindrical banner "Loading cylindrical_dinov3
ViT … (gauge=cylindrical_xyz …)" appears 0 times; compare any exp-09/exp-12 log).

## Why nothing caught it

- The cylindrical AR checkpoint's EMA weights **load into vanilla DINOv3 without error**
  — the cylindrical model adds no parameters (gauge and RoPE changes are buffer/hook
  level), so the state dicts are key-compatible. No strict-load failure.
- The launcher gate and the contract tests validate **the config JSON**, not the runtime
  routing: they assert the config *says* `cylindrical_dinov3`, which it does. Nothing
  asserts the constructed model *is* cylindrical.
- The one runtime probe that would have failed — the R1 rotation-invariance probe —
  is **skipped for the CYL arm** (`R1 probe: SKIPPED`).

## Consequences

- The "CYL" arm being trained/evaluated today is **cylindrical-trained weights running
  through a vanilla DINOv3 forward**: full-spectrum non-integer RoPE, no XYZ gauge, and
  vanilla `pooler_output` (CLS token) where the cylindrical model pools the patch mean.
  Its HAA numbers do not measure the cylindrical method. The six-arm table's CYL rows
  are invalid as labeled.
- The planned **CYL-SSL arm hits the identical defect** (plus its `azimuth_mode` /
  `prefix_mode` keys would also be ignored, and the process has no PYTHONPATH to the
  cylindrical package at all).
- `HAA_init_CYL.ckpt` itself is fine — the extracted weights are valid; only the runtime
  architecture is wrong. A re-run after the code port can reuse it.

## Proposed fix (exp-12 session, pending Yixun's go)

1. Port the exp-12-arms `conditioners.py` cylindrical block (implementation routing,
   gauge, `azimuth_mode`/`prefix_mode`, fail-closed `ssl_ckpt`) onto this branch.
2. Launcher: export `PYTHONPATH=/home/yixunhu/codespace/cylindrical-dinov3/src` for
   cylindrical arms (package pinned by SHA, consumed via PYTHONPATH per convention).
3. Add a RUNTIME contract test: construct from the CYL config and assert the ViT is
   `CylindricalDINOv3ViTModel`, and a roll-equivariance smoke on the pooled output.
4. Re-run CYL FULL (+ its evals); then add CYL-SSL per the handoff.
   Driver-script edits only after the currently-running launcher/eval processes exit.
