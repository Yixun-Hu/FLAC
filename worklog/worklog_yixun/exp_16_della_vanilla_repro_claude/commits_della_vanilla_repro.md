# commits — exp_16 della_vanilla_repro

Complete and kept complete through HEAD (plan Rev 3 §4), oldest first, from the
branch point `e603947` (`della-flac-chequity`, cut from `check-equivariance-necessity`).

- `0a0cd23` — exp_16 scaffold: query, env-survey worklog (release ckpt pins the 67.5k-step budget; config parity audit), plan Rev 1; CLAUDE.md /init refresh.
- `ddca540` — plan review round 1 (Opus 5 fallback, REVISE, 7 blocking) + plan Rev 2 + worklog entry. *(was the stale "(this commit)" placeholder in this file)*
- `81a3c44` — .gitignore: `/models` + `/AcousticRooms` — dir-only rules do not match symlinks.
- `f69a3f8` — TDD red: `src/tests/test_vit_local_resolution.py`, 8 tests pinning local ViT-snapshot resolution before the implementation exists.
- `dacd4d1` — TDD green: `resolve_vit_model_path` + ViTCoordinates call-site wiring; hub id maps onto the local `models/` snapshot, resolved path logged with weights provenance.
- `34a1535` — rung-0 record (env repair, HF-cache populate, offline load proofs, storage moves), Rev-2 approval + reviewer switch to Codex, round-A worklog + params + pip freeze.
- `c9e4d9a` — hardware amendment: Phase 2 on A100 80GB (H100 QOS-gated on della; Yixun-approved); codex-on-della mechanics (bwrap impossible, GitHub-connector review; push-first rule).
- `f38a804` — codex round-A fixes: hermetic priority tests (competing roots), `_default_local_root()` refactor, dot-basename guard, symlink cases, exact log assertion.
- `18d94ad` — round A closed: Codex (gpt-5.6-sol xhigh) REVISE → `f38a804` fixes → 17/17 re-verified; review artifact + loop record.
- `60edd0a` — Slurm kit 1/2: `della_eval.sbatch` + `della_submit.sh` — Phase-1 calibration cells, gated and recorded.
- `85a928b` — Slurm kit 2/2: `della_train.sbatch` — the Phase-2 reproduction leg, resume-aware and smoke-priced.
- `95253d9` — round B implementation worklog (gate hardening, resume glob, jid log suffix).
- `e41dfec` — codex round-B fixes: env isolation, opt-in resume, held-submit transaction, content-scoped closure gate, write-free dry runs.
- `2bddbcb` — round B closed: Codex REVISE (5 blocking) → `e41dfec` fixes → Planner re-verified DRYRUN matrix; review artifact + loop record.
- `85c02bc` — LAUNCH record: Phase-1 cells 12267442-4 + smoke 12267445 (command manifest + acceptance criteria).
- `2a504fe` — smoke PASSED: 4550 steps/epoch live-confirms batch 64; 0.58 steps/s → Phase-2 ETA 33 h, gpu-medium `2-06:00:00`.
- `4873813` — full integrative review (Codex REVISE: launch-procedure only) + plan Rev 3: A100 consistency, exact gate JSON paths + PHASE1_PASS interlock spec, no-requeue policy + interruption runbook.
- `<this commit>` — codex round-C fixes: `#SBATCH --no-requeue` in both drivers, `PHASE1_PASS.md` interlock on non-smoke `train` submissions, this ledger completed through HEAD.
- `ac734e9` — Rev 4 plan amendment + query 2 + resume-verification worklog.
- `90079a7` — round D: della_chain.sbatch + della_chain_submit.sh (self-locating legs, two-strike watchdog, chain manifest).
- `428cb14` — round D review artifact (Codex REVISE).
- `bd9f1c1` — round D fix loop: fail-closed squeue, checkpoint-indexed reseed, transaction lock, atomic stamp.
