# exp_19 worklog

## 2026-08-17T12:45:00-04:00 — plan approved; round 1 coding dispatched (Opus seat)

- **Approvals:** B1 = 1,000 steps (410 available via ckpt/10); B2 = include HAA-YAW; B3 = single-GPU recipe, two arms parallel; B4 = HAA_md.py untouched.
- **Planner analysis:** yaw-rotation gauge verified consistent on HAA by code reading (column→azimuth is standard CCW, matching rotate_scene_metadata's assumption; the deliberate minus is vertical/translation only; FA's C4 orbit is sign-closed regardless). Empirical R1 probe stays as a launch gate.
- **Coder round 1 (Opus 5 max, per SOP):** extract_ema_weights.py (+6-case TDD), FLAC_HAA_finetune_{BF,YAW}.json by byte-insertion with contract tests (stock sha pinned; BF deltas verbatim from FLAC_AR_BF.json: cond_method fa_invariant + frame_avg_angles [0,90,180,270]; YAW = exp_17 treatment block, width pin 512 verified), probe_haa_fa_invariance.py (pure core + CLI, synthetic non-vacuous tests).
- Facts pinned for the round: load_ckpt_state_dict at src/models/utils.py:23; train.py:139–147 prefix-strip contract; HAA stock config use_ema true, ViT img_w 512×2; all HAA room folders present on disk.

## 2026-08-17T13:5x — Codex r1: REQUEST-CHANGES (configs SHIP; tool+probe fix batch queued)

- Blocking: (1) extraction round-trip test self-fulfilling — must strict-load into `create_model_from_config(stock)` and check shapes/dtypes on real artifacts; (2) EMA substitution checks names only — add tensor-type/shape/dtype/layout validation vs the live DiT tensor; (3) TOCTOU on refuse-overwrite — exclusive create / tmp+no-replace publish; (4) **probe subject==oracle**: a shared gauge error is algebraically invariant, so the R1 gate certifies pipeline/shape/mask consistency ONLY — gauge correctness rests on the planner's code-reading + FA's sign-closure; docstrings/claims to be demoted accordingly (accepted as a documented structural limit, per Codex's own analysis the wrong-sign TEST is non-vacuous only via a test-only inner transform); (5) CLI probes a fresh stack — add `--init-ckpt` to load the BF init's LEARNED conditioner weights through the real consumer path.
- Non-blocking queued: drop-namespace restriction; YAW literal derived from exp_17's module instead of copied; masks included in probe outputs.
- Fix batch will be dispatched to the coder seat after its round-2 (launcher) delivery, to avoid mid-round file collisions.

## 2026-08-18T01:10 — r2 archived; fix batch 2 dispatched; HAA relocation in progress; FULL PREAPPROVED

- **Yixun preapproves the HAA finetune FULL runs; deliverable = final results tomorrow.** Storage question answered: after the HAA move, local free ≈ 316 GiB — all three arms' checkpoints retained without deletion, B3 parallelism restored.
- Codex r2: r1 blockers 1/2/3/5 FIXED; 4 stands as the demoted disclosure. Extraction+tests SHIP; probe SHIPs under demoted scope. Launcher/guardtests REQUEST-CHANGES (6 blocking: HEAD+split-file binding; whole-run per-arm/per-GPU locks; suite-vs-production config-mutation race; non-fatal trap restoration; non-dry reject poisoning; init re-hash at consumption) + floors clamp + symlink/inventory gate. Fix batch 2 dispatched to the coder seat with concrete remedies for each.
- **HAA→NAS move:** first rsync FAILED silently (dirs only, 0 file bytes; my `tail -3` had discarded the per-file errors — measurement lesson recorded). Probes now pass (small + 1 GiB write-readback sha-verified, 108 MB/s real). Re-run with full logging in flight (~35 min expected). Swap to symlink only after count+byte verification.
- Codex r2 confirmed the symlink plan is dataloader-safe (json_scandir joins lexically; no pins under HAA/); the launcher additionally gains a resolved-root + split-inventory gate (fix batch 2 item 8).

## 2026-08-18T02:15 — R1 ADJUDICATED (fp64 5.8e-14 → PASS); all three arms rolling

- **R1 timeline:** production gate (fp32@1e-5) tripped at 3.242e-05 on BF → per pre-registration, STOPPED and reported. Diagnostics: three unrelated inits (P1/BF/YAW) gap within 15% of each other (3.18/3.24/2.83e-05) → weight-independent precision noise hypothesis; sign-flip control is 1.9e-02 (600×). Yixun ruled: fp64 re-measurement decides, rule pre-committed (<1e-7 proceed). **fp64 result: 5.806e-14 / 5.940e-14 / 5.917e-14** — seven orders under the adjudication threshold, thirteen under the fp32 reading. Ruling fires: arms proceed. Gate re-parameterized to float64@1e-7 (stricter in meaning: exactness to fp64 rounding), evidence + reasoning in-file and here.
- **Probe extensions this cycle (coder seat):** num_workers=1 (repo loader hardcodes persistent_workers=True); --dtype float64 (widen AFTER strict load; fp-only metadata cast; fixed a .float() output cast that would have re-measured fp32 rounding and passed for the wrong reason). 40 probe tests green. PIN_probe bumped twice, L7 pairing verified working.
- **In flight:** P1 FULL (GPU0, since 01:45, ~1000 epochs×3 micro-batches, ETA ~04:00); BF FULL (GPU1, launched 02:1x after the recalibrated gate); YAW auto-launcher armed behind P1's clean completion (EXPECT_SHA pinned). Round-3 eval driver (60 cells) dispatched to the coder seat.
- Process notes: my first BF dispatch went out BEFORE the gate edit landed (edit assertion failed on a variable-based invocation; bash block kept going) — it self-refused at the fp32 gate exactly as designed; re-dispatched after the correct edit. Also the third pipeline-rc mismeasurement of the night is recorded (echo rc after a pipe measures tail's rc) — standing lesson: PIPESTATUS or capture-then-check, always.

## 2026-08-20T12:15 — checkpoint archive COMPLETE (disk rescue step 1)

- All 500 finetune ckpts (5 arms × 100) archived to `/media/diskstation/yixunhu/FLAC/checkpoints/exp19_haa_finetune/<ARM>/` per Yixun's 2026-08-20 NAS-storage mandate; per-arm verification (count + per-file bytes + 4 sampled sha256 incl. endpoints) passed before every local deletion. Local keeps {410,1000} per arm + all metric JSONs. Local free: 29 → 503 GiB.
- AcousticRooms NAS copy verified earlier today (615,265 files, file-bytes identical, sampled shas OK); step 3 (symlink swap + local delete, 27 GiB) FROZEN until Yixun's explicit order after his Arm-B finishes (~08-21 03:00).
- Observed under NAS `checkpoints/` (not ours, untouched): `exp12_cyl_dinov3_arms`, `exp19_raf_finetune`, `exp19_rcal_haa_repro` — another session appears to be working RAF; flagged to Yixun re exp_20 planning + numbering collision.
