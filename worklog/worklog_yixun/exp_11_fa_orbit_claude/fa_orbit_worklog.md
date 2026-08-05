# Lab notebook — exp_11_fa_orbit

## 2026-08-05T15:40:00-04:00 — scaffold + environment reconnaissance

- **Goal** — scaffold exp_11 (orbit-size sweep C8/C16/C32 for fa_invariant conditioning, commissioned by Yixun 2026-08-05) and establish the ground truth of the machine/repo state before planning.
- **Version Control** — branch `check-equivariance-necessity`, base commit `b9e38ce` (model_comparison: fa-scratch@40k vanilla-eval rows). Working tree clean except untracked `AcousticRooms` symlink.
- **Result** — `passed` (scaffold). Reconnaissance findings, all verified from disk:
  - **Plumbing already exists end-to-end**: `training.frame_avg_angles` is read by `src/training/factory.py:76` → `LightningDiffusionModel` (train side), and `eval_FLAC.py --frame-avg-angles` (eval side, records `a{n}` in metric-JSON filenames). `invariant_conditioning` (src/data/yaw_rotation.py:221) accepts arbitrary angle tuples (angles[0] must be 0.0); the orbit loop is sequential (one `only_ids` ViT pass per non-identity angle, accumulate + average) so wall-clock scales ~linearly in |orbit| while non-ViT conditioners run once.
  - **Exactness geometry**: `rotate_scene_metadata` quantises to integer panorama-column rolls (W=512). C8=64 px, C16=32 px, C32=16 px — all exact AND all aligned to the DINOv3 16-px patch grid (C32 is the finest patch-aligned subgroup; C64 would be sub-patch).
  - **Measured C4 fa step rate (exp_10, DDP 2×L40, MB32)**: ckpt mtimes 55k→65k = 2,500 steps per ~7.19–7.55 h ⇒ **~0.095 steps/s** (slower than the 0.14 planning figure; node cotenancy suspected).
  - **exp_07 B-F ckpts 2.5k–40k all on disk** (`outputs_FLAC/exp07_BF/.../checkpoints/`) — a C4 fa-protocol screen curve below 40k can be backfilled cheaply (the exp_07 S10000–S30000 screens were the retracted mismatched-protocol evals and are NOT usable as fa comparators).
  - **Storage**: /n/fs/gatrdp at 48% (859 GB free); 3 arms × 16 ckpts × ~724 MB ≈ 35 GB — no quota risk.
- **Analysis** — the experiment is config-only on the training path (new model-config JSONs + a parameterized launch script); no model code changes needed. The dominant planning question is GPU cost: with ViT passes ~5/6 of the C4 fa step cost, projected per-arm cost to 40k steps is ~6.5–9.4 d (C8), ~13–18.5 d (C16), ~26–37 d (C32) on 2 GPUs each — a staging/budget decision that belongs to Yixun.
- **Next** — write `plan_fa_orbit.md`; Codex plan review; present for approval. No implementation before approval (SOP gate).

## 2026-08-05T15:40:00-04:00 — cross-experiment state found during reconnaissance (context, not exp_11 actions)

- **exp_10 B-F resume is STALLED at step 65,000/67,500**: last ckpt written 2026-08-05T09:33; no training process alive; all 8 GPUs idle at 15:27. Logged in exp_10's worklog; restart decision deferred to Yixun.
- **Untracked-file wipe ~2026-08-04T23:17**: every tracked file's mtime is 23:17 and exp_10's tee'd train log plus all pre-Aug-5 `wandb/run-*` dirs are gone (checkpoints under `outputs_FLAC/` survived). The exp_10 process itself survived the wipe and died later, cause not diagnosable without its log.
- **Cotenant session on this node**: a second Claude session (checkout `/n/fs/gatrdp/codespace/cylindrical-dinov3`, working in this FLAC repo) launched `FLAC_exp02_P1rerun` probes and a committed 2-GPU 87.5k-step run at 14:49–14:54 today (wandb dirs in this repo). Its GPUs read idle at 15:27 — treat 2 GPUs as presumptively claimed; launch-time VRAM gates (bf-launch family) will verify reality.

## 2026-08-05T16:25:00-04:00 — plan Rev 1 → Codex review (REJECT, 11 BLOCKING + 3 NIT) → Rev 2

- **Goal** — SOP plan-review loop before surfacing for approval.
- **Command / Validation** — `codex exec`, gpt-5.6-sol @ xhigh, full reviewer briefing (SOP + announcements + exp_07/10 context + implementation surfaces). NOTE: bwrap sandbox unavailable on this host (`max_user_namespaces=0`) → `--sandbox danger-full-access` with read-only instruction; tree verified clean post-review. Review saved: `fa_orbit_codex_plan_review.md`.
- **Result** — `passed` (loop closed at plan level): verdict REJECT; all 14 findings addressed in Rev 2 (per-finding changelog = plan §9). Substantive design changes: contemporaneous C4L bridge control arm (comparator switch), seed-paired statistics with K8 T60/R@1 co-primaries, pre-registered per-arm 2×2 mechanism cells, numeric trend estimates with NOT-ESTIMABLE rules, economic futility gates (FUTILITY-STOPPED excluded from inference), fail-closed row-provenance validator, D3 bf16-invariance-floor probe, D1 demoted to fully-specified cost-prior, R3 redesign, B-V-extend compute claim withdrawn (Planner error, acknowledged), R4 exploratory compute-frame cell, corrected cost algebra prose, Slurm-based launches (environment change discovered mid-planning: this is a 32-node Slurm cluster; org policy + current practice require the scheduler).
- **Analysis** — the review materially strengthened the design; the C4L bridge in particular converts a historical cross-run comparison into a same-environment controlled sweep at modest cost (~160–235 GPU-h).
- **Next** — present Rev 2 + staging options to Yixun for approval. Implementation (TDD rounds) only after sign-off.
