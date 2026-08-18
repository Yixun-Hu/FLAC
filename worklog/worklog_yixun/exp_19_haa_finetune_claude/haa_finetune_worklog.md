# exp_19 worklog

## 2026-08-17T12:45:00-04:00 — plan approved; round 1 coding dispatched (Opus seat)

- **Approvals:** B1 = 1,000 steps (410 available via ckpt/10); B2 = include HAA-YAW; B3 = single-GPU recipe, two arms parallel; B4 = HAA_md.py untouched.
- **Planner analysis:** yaw-rotation gauge verified consistent on HAA by code reading (column→azimuth is standard CCW, matching rotate_scene_metadata's assumption; the deliberate minus is vertical/translation only; FA's C4 orbit is sign-closed regardless). Empirical R1 probe stays as a launch gate.
- **Coder round 1 (Opus 5 max, per SOP):** extract_ema_weights.py (+6-case TDD), FLAC_HAA_finetune_{BF,YAW}.json by byte-insertion with contract tests (stock sha pinned; BF deltas verbatim from FLAC_AR_BF.json: cond_method fa_invariant + frame_avg_angles [0,90,180,270]; YAW = exp_17 treatment block, width pin 512 verified), probe_haa_fa_invariance.py (pure core + CLI, synthetic non-vacuous tests).
- Facts pinned for the round: load_ckpt_state_dict at src/models/utils.py:23; train.py:139–147 prefix-strip contract; HAA stock config use_ema true, ViT img_w 512×2; all HAA room folders present on disk.
