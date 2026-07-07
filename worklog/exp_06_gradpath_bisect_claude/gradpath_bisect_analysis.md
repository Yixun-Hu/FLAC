# Analysis — exp_06_gradpath_bisect

**Author:** Fable 5 (Planner) · **Date:** 2026-07-06

## Is the result reliable?

**Yes.** Screens were pre-registered as ordering-only with thresholds on both metrics Yixun asked about; the documented K=8 effect sizes (14–62σ in prior 5-seed gates) make single-seed ordering decisive at the observed separations (ΔT60 ≥ 0.15 between adjacent arms vs single-eval noise ~0.01); every prediction was registered before its data (two were falsified and are reported as such); the L5 arm ran on the reviewed `--lr-schedule` implementation with restart semantics pinned by execution-derived tests; S3 numbers survived a consolidated Codex review (train-split correction applied, conclusion unchanged).

## Outcome — both commissioning questions answered

**Q1 (why can't T60/EDT be recovered by fine-tuning?)** Because nothing is being "damaged" in the optimizer sense: S1 shows the model *converging*, fast (~75–80% of the T60 shift inside 200 optimizer steps), toward the optimum of the objective we are actually training — and S2 shows that optimum lies monotonically *farther* from the released point the more lr-distance is covered. The gradient at the released weights points away from the released behavior, robustly, under the exact upstream code (S3.1). Fine-tuning cannot recover T60/EDT for the same reason water cannot flow uphill: the released weights are not near any optimum of this objective on the data/environment we possess.

**Q2 (does lr matter — original used a scheduler?)** Tested across two orders of magnitude plus the faithful InverseLR restart: **no setting recovers the gate, and damage is monotone-increasing in lr** (9.09 → 10.10 T60 at K=8 from 5e-7 to the restart). The 5e-6 constant choice of exp_03–05 was, in hindsight, *conservative* — the original recipe restarted is the most destructive option. The lr deviation is definitively exonerated.

## What remains (outside recipe space)

With code lineage eliminated, augmentation 10× too small for T60, and every optimizer-side mechanism falsified across exp_03–06, the surviving explanations are: **(i) dataset-version lineage** (our AcousticRooms copy vs the release's training data), **(ii) library/environment numerics** (torch/torchaudio versions at training time), **(iii) source-side checkpoint selection or unshipped training config**. None is testable from the shipped artifacts alone; (i) could be probed by checksumming against an authoritative manifest, (iii) only by contacting the FLAC authors.

## Consequence for the project (decision standing since exp_05, now fully justified)

- **Route A — matched comparison** (fa_invariant+freeze-bn vs vanilla+freeze-bn, identical recipe, both arms carrying the now-fully-characterized regression): measures FA's marginal effect cleanly and yields the full-split H1/H2 rotation-sweep verdicts on a fine-tuned model. ~13 h GPU. The exp_02 sanity-check story completes.
- **Route B — from-scratch fa_invariant training**: the only route to absolute Table-1 claims (H3) the evidence supports; all infrastructure reviewed and test-pinned.
- Not recommended: further recipe archaeology (exhausted) or author-independent data forensics before Route A (poor information-per-hour).

**Recommendation: Route A now; Route B as the planned follow-up if Table-1 numbers are required.**

## Bottom line

exp_06 converts the last open "why" into a closed mechanism map: fast convergence to a genuinely different optimum, lr-monotone, code-identical, augmentation-minor — the released checkpoint's superiority on T60/EDT is not reproducible from the shipped artifacts, and the project's path forward no longer depends on recovering it.
