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
