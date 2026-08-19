# exp_17 — closing analysis (reliability + interpretation)

**Analysis by Claude Fable 5; earlier rounds (code + three Codex reviews +
training-side artifacts) by the Opus 5 seat as by-lined per artifact. Closure
review 2026-08-19: zero blocking findings across the exp_17 drivers.**

## Verdict

Training-time random yaw augmentation (online per-sample re-orientation,
uniform over 512 columns) on the 2×A6000 legacy rung, single delta vs P1:

1. **Rotational flatness, delivered from step 2,500 onward:** C4 spread at
   40k (K=8) ΔT60 0.075 / ΔEDT 0.349 vs P1's 0.897 / 6.738 (5-seed control
   orbit) — 12–19×. The full 16-checkpoint control trajectory shows P1's
   anisotropy GROWING with training (ΔT60 0.1→0.90; ΔEDT jumps at 10k) while
   Yaw-Aug never develops it. Invariance is not learned late; it is never lost.
2. **Zero quality cost at the training frame — in fact a regularization gain:**
   0° 5-seed rows beat P1 on T60 both K (7.965 vs 8.993 K=8) and edge FA on
   T60/R@5; FA keeps C50/EDT. FD beats FA, matches P1.
3. **Flatness is continuous in angle, not a C4 artifact:** at 45° (off the
   training orbit; exp_07 A6's negative-control angle) Yaw-Aug sits essentially
   inside its own C4 band (worst excursion EDT +0.27) where P1 degrades
   (T60 +0.72). Exact-C4 FA does not guarantee this off-orbit. ⚠️ 45° rows are
   seed-42 only unless decision (d) upgrades them to 5 seeds.
4. **Residual orientation preference exists but is ~0.01 dB-scale** (C50 at 0°
   1.0132 vs ~1.023 rotated) — an order below P1's 0.13 anisotropy; disclosed
   because B-F's equality is constructional while ours is statistical.

## Reliability

- Single training seed per arm; eval = 5 seeds on all published endpoint rows,
  seed-42 single on trajectory/45° cells (labeled as such everywhere).
- Global-mean estimand throughout (verified identical to the flat key every
  model_comparison raw JSON carries); per-scene convention NOT used here.
- The training run belongs to the exp-17-yawaug-a6000 worktree (grad-ckpt ON,
  single-delta vs P1); completion certified by the read-only audit
  (exp17_full_audit.py) — endpoint marker with Lightning's real framing, 16/16
  cadence checkpoints, whole-line treatment banner, 2 ranks, finite losses.
- Our own launcher/guardtests are RETIRED (r3 blockers listed in-file); the
  grad-ckpt-OFF variant was stood down as a disclosed numerical confound.
- Retractions made and recorded during the campaign: my "grad-ckpt off saves
  19 h" rate claim (warm-up-window measurement error); the "bitwise identical"
  grad-ckpt evidence overstatement (corrected to allclose-grade, twice).
- Accepted debt: roteval-r1 non-blocking findings (audit stale-log stitching,
  save-dir suffix ambiguity, guard H/I vacuity, argval token boundary) +
  closure-review driver debts — none touches a published number.

## Open per Yixun

- (c) 45° probe delivered, acceptance pending; (d) 45° 5-seed upgrade undecided.
- model_comparison.md rows staged in the generator; regeneration is cluster-only
  (exp_11 validator pins cluster-absolute paths) — publish on next cluster regen.
