# exp_19 — commit ledger
(chronological; branch exp17-yawaug-scratch)
- `bc55a3e` round 1: EMA extraction tool + arm configs + R1 probe (80 TDD tests)
- (r1 review archived) `haa_finetune_codex_code_r1_review.md`
- `ff28a5a` rounds 2+fix: launcher + 135 guardtests + all r1 blockers closed
- (r2 review archived) `haa_finetune_codex_code_r2_review.md`
- fix batch 2 (six r2 blockers) — folded into ff28a5a lineage
- `…` probe num_workers fix + PIN bump; `--dtype float64` + adjudication
- R1 gate re-parameterized float64@1e-7 (Yixun ruling; fp64 evidence 5.8e-14 ×3 inits)
- HAA relocation to NAS + prepare_data (splits byte-match pins) + 3 EMA inits
- P1 FULL + eval (20 cells); YAW FULL + eval (20); BF grad-ckpt restoration
  (4-delta contract) + FULL + eval (20); YNA arm + FULL + eval (20)
- curve eval (24 cells); persisted aggregator (byte-reproduces published tables)
- closure review: ZERO blocking (`haa_finetune_codex_closure_review.md`)
Full SHAs: `git log --oneline --grep exp_19`.
