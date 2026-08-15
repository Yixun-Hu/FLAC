# Plan review — exp_17 yaw_aug_a6000
**Reviewer:** main session, Claude Opus 5 (1M), max effort · 2026-08-15 · read-only verification against the live repo (not a reading-only review)

## Verdict: APPROVE WITH ONE BLOCKER — the design is sound; §2.3's source pin does not exist.

### Verified TRUE against the code (each checked, not assumed)
1. **Config schema** `training.yaw_aug = {enabled, img_w, seed}` — exactly the keys `_parse_yaw_aug_config` accepts (`factory.py:9-51`); identical to exp_15's shipped `FLAC_AR_YAWAUG.json`.
2. **Draw key** — `diffusion.py:86-87` documents literally `(yaw_aug.seed, global_step, global_rank, within-batch index)`, counter-based, no stream state. Plan §2.2 is verbatim correct, including resume-exactness and RNG isolation.
3. **Rotation semantics** — `rotate_scene_metadata` rolls `depth` horizontally and rotates exactly the four pose fields (`source`, `source_vit`, `context_poses`, `context_poses_vit`). "All four pose fields" is right.
4. **`img_w: 512` is correct** — `AR_md.py:51` builds the panorama via `convert_equirect_to_camera_coord(..., 256, 512)`, so integer column offsets are exact rolls with no interpolation.
5. **Physics claim is right.** The augmentation rotates the *coordinate frame*, not the scene: panorama columns and all listener-centric pose vectors rotate together, so the physical configuration — and therefore the 1-channel omni RIR and the context audio — is genuinely unchanged. This is the same premise exp_02 validated.
6. **Config base** — exp_15's arm config equals BVp1 after deleting `yaw_aug`, with zero non-training diffs. The "BVp1 + one block" construction is proven to work; exp_17 can mirror it.
7. **Launcher flags exist** — `--precision` (default `bf16-mixed`) and `--accum-batches` (default 1) are real `train.py` flags.
8. **§6.2's premise now holds.** P1 and B-F rotated five-seed grids ARE on disk: P1 {90:20, 45:20, 180:10, 270:10}, B-F {90:11, 45:10, 180:10, 270:10} cells. The three-method figure is buildable.
9. **§10's premise holds** — 58 P1/B-F metric JSONs written since 2026-08-14 18:00 (incl. `curve0_VAN_S10000_K1_s42`), i.e. the curve workers are real.
10. **§2.3's P1 caveat is accurate** — P1 did resume from the 32,500 checkpoint after an interruption, so it is not a single-source trajectory. The plan's refusal to claim bitwise pairing is correct.

### BLOCKER
**B1 — the source pin `58d0b887` does not exist in this repository.** `git rev-parse`, `git cat-file` and `git log --all | grep '^58d0b'` all find nothing. §2.3 is therefore unexecutable as written. Resolve before implementation, choosing one:
 (a) supply the correct SHA (it may be a cluster-side commit that never landed on this remote), or
 (b) **build from current HEAD** — viable, because `yaw_aug` IS present at HEAD in both `factory.py` and `diffusion.py`. But then §2.3's stated *purpose* (excluding exp_14/exp_16 training-path changes) is defeated and must be re-argued: those changes are default-preserving by construction (absent `frame_avg_max_fwd_samples` / absent `are_lambda` → the literal pre-change call, proven by regression tests incl. RNG-stream parity), and exp_16's `RandomTimeShift`/`SampleDataset` contract publishes an already-drawn value without changing the waveform or consuming extra RNG. That argument is defensible — but it must be written down, not assumed.

### Recommended improvements (non-blocking)
- **R1 — name the regularization confound.** A yaw-augmented model may do better on rotated inputs merely from greater effective data diversity, not from yaw-specific structure. The 0° cells you already collect are the discriminator: pure regularization should help at 0° too, while yaw-specific learning should show a much larger gain at 90/180/270 than at 0°. Register that reading in §7 before seeing the numbers.
- **R2 — eval flags.** `--frame-avg-angles` is inert under `--cond-method vanilla`; harmless, but if evaluation runs at HEAD, announcements 05/06 also want `--frame-avg-max-fwd-samples 64` declared and the chunk plan recorded, even for a vanilla arm.
- **R3 — smoke-gate threshold.** §3 estimates 43–50 h from "bounded augmentation overhead" but sets no abort rule. Pre-register a threshold (e.g. measured steps/s implying >55 h ⇒ stop and re-plan rather than launch).
- **R4 — GPU-occupancy rule is stricter than practice.** §10 forbids starting while any unowned process holds a card, but Yixun's `exp12B`/`exp12C` occupy both cards (~5.9 GB each, ~42 GB free) into 8/15–8/16. Under a literal reading exp_17 can never start. The program's standing policy is co-tenancy with a free-VRAM floor and disclosure; align §10 with it or state the intended wait explicitly.
- **R5 — record BOTH SHAs** (training source and evaluation source) in every eval manifest if training is pinned and evaluation runs at HEAD.
