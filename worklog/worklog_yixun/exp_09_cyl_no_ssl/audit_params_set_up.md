# Params — exp-09 Stage A blessed audit run (recorded BEFORE launch)

Cleared code: A-code r6 "CLEARED FOR THE AUDIT RUN" at `f150c40` (6 review rounds;
production byte-identical from `b4bbec6`). Protocol: plan Rev 4.1 §1 (A1, A2a–A2d,
TOTAL three-way audit_status). Sample: Cafe/Cafe_idx_0 S001_R0044 from the registered
dataset instance `/home/yixunhu/codespace/xRIR_code/data` (equivalence to the live
FLAC checkout's data is NOT claimed — r1 adjudication). Geometry 256×512. CPU FP32
eager; env flac. Angles α_j = 2π·16j/512, j ∈ {1,4,8,16,24}.

Expectations (fail-closed, all seven):
- fingerprint: `81038cc90f3f295277016e2a8981867ed752b9a081183a8a9541204a892cad5b`
- package (cylindrical-dinov3) SHA: **`1f2c015905980a070c01a9aebce68bdebe00dbd2`** (repo FROZEN from records
  freeze until post-launch)
- package path prefix: `/home/yixunhu/codespace/cylindrical-dinov3/src/cylindrical_dinov3/`
- worktree SHA: **the SHA of the commit landing this records file** (self-referential
  by construction — the records live in the worktree; resolved at launch as
  `git rev-parse HEAD` and RE-VERIFIED in the result review to equal the records
  commit; `--expect-clean-worktree` guarantees the run executed exactly that commit)
- clean worktree: required
- checkpoint revision: `114c1379950215c8b35dfcd4e90a5c251dde0d32`
- weights sha256: `4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d`

Acceptance: exit 0 + audit_status=valid_pass ⇒ gauge-ON baseline; exit 2 +
valid_convention_failure ⇒ gauge-OFF + separate investigation; exit 3 ⇒ fix
infrastructure and re-run (no gauge decision). Result review before the decision.
