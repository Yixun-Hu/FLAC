# exp_19 code round CLOSED — Codex APPROVE (r6 pass, 2026-08-20 ~05:15 EDT)

## The round in numbers
- **6 Codex review passes** (r1 code review + 5 re-reviews), **6 fix rounds**, **8 contract amendments**, final verdict **APPROVE, no open in-boundary findings**.
- **511 tests passing** (1 opt-in CUDA skip) across 9 RAF suites; every metric/loader consumer suite green; AR/HAA behavior byte-identical throughout (incl. the T8 legacy-arithmetic restoration that PREVENTED a silent record shift).
- Total upstream (shared-file) diff: `eval_FLAC.py` (+139/−21, additive stream IDs), `metric_callback.py`/`RT60.py` (RAF policy + macro block), `src/data/dataset.py` (+19, narrow RAF-only re-raise).

## What stands ready
- `data/RAF/`: prepare (canonical identities, staged atomic publish, splits, amplitude audit), depth renderer (miss-cap+inpaint policy, mask-derived QA, mesh-independent vertical gauge gate, sightline checks), readback audit (pinned record gate), publish transaction/verifier.
- `RAF_md.py` with mandatory publication verification; RAF configs; pinned canonical readback record (`9288181b…`, gauge `(X,Z,Y)`, quat `xyzw` per RAF docs, T60 headline confirmed).
- **R-cal PASSED**: eval pipeline exactly calibrated (Leg A vs paper: C50/EDT macro within seed noise; T60 pooled +2.5%); recipe reproduction in-band (Leg B; ~10–16% T60/EDT band documented, quoted beside any smaller RAF delta).

## Residuals for Yixun (out of the registered threat model; recorded, not implemented)
1. **MEDIUM** — 43-GB raw audio corpus not content-hashed (pose indexes + counts are). Optional future manifest.
2. **LOW** — no signing/fstat/per-item rehash vs a malicious local actor.
3. **MEDIUM** — a globally consistent horizontal axis permutation/chirality is provably render-undetectable; pinned by derivation (documented Metashape convention + literal unit-tested matrix + verified vertical axis). Empirical evidence would need a surveyed landmark/compass bearing in the RAF rooms.

## Operator notes (before the canonical run)
- Upstream footgun (pre-existing, documented in a test): a corrupted SUPPORT capture ⇒ unbounded substitution loop in `SampleDataset`.
- Canonical sequence: prep dry-run → canonical prep (splits committed once) → 42 depth renders → smoke finetune (~20 steps) → 1000-step finetune → 5-seed evals ×2 rows (zero-shot + finetuned). GPU need: 1×A6000, primary window after exp_18's 09:00–12:00 reservation.
- Optional: second R-cal training seed (~3.5 h GPU) to bound reproduction variance.
