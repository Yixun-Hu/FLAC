**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-10-23 · **Round:** plan r1

**Verdict:** REVISE

## BLOCKING findings

1. **The operator is not equivariant on its residual-tie path, and the proposed audit cannot certify that the path is absent.**

   The circular-profile lexicographic rule is equivariant whenever it produces a unique candidate. The relative-source-azimuth key is also equivariant in exact arithmetic. However, the final `min(cands)` fallback is explicitly frame-dependent: after a roll, “smallest column” need not co-rotate. Returning a residual flag does not restore invariance, and the training wrapper does not retain or report that flag.

   More importantly, the planned full audit examines 9,650 depth maps, while residual resolution depends on the target `source`. A depth-only audit cannot determine the residual count for the 291,210 training examples or 6,337 unseen examples.

   Concrete fix:

   - Fail closed on an unresolved tie instead of selecting the smallest column.
   - First compare the complete recovered panorama rooted at each candidate, not only `P[w]`; this eliminates profile-periodic ambiguities when the underlying depth image is not periodic.
   - Audit every distinct `(depth panorama, target source)` pair used by the train and evaluation splits before FULL.
   - Add exhaustive roll-equivariance tests for periodic profiles with nondegenerate sources, source azimuths at the `±π`/modulo boundary, and the degenerate-source case.
   - If any genuinely symmetric joint input remains, define the method’s restricted domain and abort on it. A deterministic equivariant selector cannot exist for an input with a nontrivial stabilizer.

2. **Checkpoint/config protocol binding is not fail-closed enough to prevent an announcement-05 failure.**

   The plan requires agreement with the checkpoint’s embedded config only “when present” ([plan §2.2](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_25_depth_canon_claude/plan_depth_canon.md:58)). Canonicalization adds no learned parameter, so a CANON checkpoint loads perfectly under a P1 config and vice versa. The embedded config is therefore the only artifact-level proof of how the weights were trained.

   Concrete fix:

   - Whenever canonicalization is in play—trained on, applied, or explicitly mismatched—require `ckpt["model_config"]`; absence must be fatal.
   - Type-strictly compare the embedded and file `training.canonicalize` blocks before mutating `training_config` or constructing the model, following the existing ARE binding pattern in [eval_FLAC.py](/home/yixunhu/codespace/FLAC/eval_FLAC.py:1254).
   - `--allow-canon-mismatch` may authorize only a known trained/applied mismatch. It must not authorize a missing or contradictory embedded config.
   - Record separate `canonicalize_trained`, `canonicalize_applied`, `canonicalize_mismatch`, checkpoint-config hash, and file-config hash fields.
   - Add tests for missing embedded config, absent-vs-false, type mismatch, file/embedded disagreement, and both diagnostic directions.

3. **The planned invariance evidence and acceptance threshold do not establish “round-off-level output invariance.”**

   `|Δ_R| < one across-seed SD` is much looser than round-off and could admit scientifically meaningful degradation. Aggregate T60/C50/EDT/retrieval metrics can also agree while individual generated RIRs differ substantially.

   The evaluation cells omit `--store_predictions`, violating announcement 08. Moreover, the current implementation stores predictions before clamping and padding ([eval_FLAC.py](/home/yixunhu/codespace/FLAC/eval_FLAC.py:1433)), while the scored tensors are produced afterward ([eval_FLAC.py](/home/yixunhu/codespace/FLAC/eval_FLAC.py:1436)). Thus the existing flag does not save “exactly-as-scored” waveforms.

   Concrete fix:

   - Save post-decode, post-clamp, post-padding waveforms for every generative cell, with item/order provenance and checksums.
   - Put `--record-stream --expected-stream-count 6337` on paired θ=0, C4/45°, and random-yaw cells—not only the random cells—and require identical input/context hashes for every comparison.
   - Add paired, item-level waveform statistics such as max and quantiles of relative L2 and L∞, plus exact-equality counts where applicable.
   - Add a duplicate θ=0 cell to measure evaluator repeatability; judge rotated-vs-zero against that floor.
   - Replace the one-SD rule with pre-registered conditioning- and waveform-level tolerances. Aggregate metric spreads should be secondary corroboration.
   - Test that stored tensors are exactly those passed to `metric_callback.update_metrics`.

## HIGH findings

4. **“Exact depth recovery” overstates what is recovered from the point cloud.**

   `z/(-sin φ)` is bitwise stable under the present CPU yaw transform because the rotation matrix leaves `z` unchanged, and `H=256` has no zero denominator. But it is not an exact inverse of raw depth construction: the probe itself reports only 91% bitwise equality and maximum error `8.9e-16`. That is excellent numerical agreement, but not exact recovery of the original `.npy`.

   Because the requested treatment is defined by the raw deepest pixel, the full audit must compare raw-depth and recovered-depth maximum candidate sets and selected columns over all depth maps. Rename this to “bitwise yaw-stable recovered-depth proxy” unless full equality is actually established. Also reject odd heights or any grid containing a zero denominator.

5. **The comparison design does not support a random-yaw comparison against all three named comparators.**

   The R block evaluates only CANON. A P1 R block is optional, while available exp_14/15 random-yaw results use a different Neuronic recipe. There is no matched random-yaw B-F or A6000 Yaw-Aug block in this plan. Since comparator reruns are explicitly out of scope, the result must be framed as:

   - CANON internal Z-versus-random invariance;
   - historical fixed-angle C4/45° comparisons against P1/B-F/Yaw-Aug;
   - cross-recipe random-yaw results only as contextual evidence.

   Do not describe the random-yaw result as a four-arm comparison. Add an artifact ledger naming the exact checkpoint, evaluator pin, protocol, seeds, aggregation, and raw files for every reused comparator cell.

6. **The living comparison table currently contains the wrong Yaw-Aug recipe for the proposed four-row story.**

   The current `model_comparison.md` Yaw-Aug row is exp_15 Neuronic (`10.067/8.503`), not the exp_17 matched-A6000 row (`9.397/7.965`). The plan adds only a CANON row spec, so “CANON becomes a fourth row” would mix recipes or omit the intended Yaw-Aug comparator.

   Either ingest the exp_17 raw five-seed artifacts through a validated generator row, or keep a separately generated matched-A6000 four-arm table. Never silently use the exp_15 row as the named comparator.

7. **“Single delta versus P1” remains a historical matched-recipe claim, not a strict causal pair.**

   The JSON delta is clean and the effective recipe matches. However, P1 was trained through older—and reportedly multiple—source commits, while CANON will use the current tree. An initialization hash and disabled-path test do not prove equality of a 40k CUDA trajectory.

   Retain “single intended config/algorithm delta against a historical matched-recipe control,” perform a static training-path diff audit against the P1 launch/resume commits, and soften H-Q conclusions. CANON≈B-F would be *consistent with* invariance explaining part of B-F’s gain; it would not make that attribution causal. Likewise, CANON<P1 would not uniquely identify orbit averaging because canonicalization introduces discontinuity and a source-dependent tie rule.

8. **Several load-bearing tests are still missing.**

   Add tests for:

   - periodic-profile plus source-key equivariance across all 512 rolls and wrap-boundary azimuths;
   - residual ties raising instead of silently selecting an absolute column;
   - raw-depth versus recovered-depth selected-column agreement;
   - exact shapes (`source [3]`, `source_vit [1,3]`, contexts `[N,3]`), floating dtype, finite poses, nonempty dimensions, and even-height/nonzero-sine requirements;
   - an `evaluate_model` integration spy proving rotation → canonicalization → conditioning order and residual-count propagation;
   - checkpoint-embedded protocol binding;
   - filename injectivity across canonicalized/noncanonical, mismatch, fixed/random, FA/vanilla, and ARE combinations for both metrics and predictions;
   - exact scored-waveform persistence and paired stream identity.

   Do not make critical recovery tests dataset-optional: a float64 256×512 `.npy` exceeds 1 MiB once the header is included, so the proposed fixtures will likely be skipped. Generate deterministic scalar depth tensors and pass them through the real AR converter instead.

9. **The launcher plan needs run-owned checkpoint and NAS-storage gates.**

   The exp_17 r3 remedies are correctly carried forward, but the planned launcher also needs to:

   - require an empty, run-specific FULL namespace;
   - check mount identity, writability, and free space on the NAS save filesystem, not the repository filesystem;
   - ensure every accepted checkpoint was created by this invocation;
   - reject stale/foreign endpoint files and enforce exactly the expected cadence set;
   - bind smoke evidence to the final reviewed HEAD and rerun smoke after any review-driven source change.

## NIT findings

10. The source-azimuth fallback makes this a joint geometry-plus-query canonicalizer in periodic-profile cases, not purely “depth canonicalization.” Name and disclose that distinction.

11. Canonicalization is one conditioner pass and far cheaper than B-F, but it is not literally zero-cost: it scans, rolls, and rotates an `H×W` point cloud. Report measured preprocessing latency rather than “zero inference cost.”

## Verified as correct

- The roll direction and angle sign are consistent. AR columns increase with azimuth; positive `α` rolls by positive `dj` and applies `Rz(+α)`. Therefore `dj=(W−j*) mod W` places column `j*` at column zero while preserving physical consistency.
- `z=-D sin φ` in `AR_md.py`, so `D=z/(-sin φ)` is the correct recovery formula for the current `H=256` grid.
- The circular lexicographic key is equivariant whenever it selects a unique rooted profile.
- Applying canonicalization before conditioning in training, validation, and wrapper test paths is correct. Applying it in `eval_FLAC.py` after `apply_rotation_plan` and before conditioning is also the correct deployment ordering.
- Co-rotating all four pose fields and leaving mono RIR/context audio unchanged is physically sound.
- Mutual exclusion with Yaw-Aug and FA avoids an undeclared multi-treatment arm.
- The matched config, optimizer, batch/SyncBN, precision, seed, 40k budget, EMA, and gradient-checkpointing recipe is correctly traced to the intended P1/Yaw-Aug A6000 setup.
- The planning probe is sound as an exploratory, read-only CPU probe for tie prevalence and z-channel stability. It is not a proof of the proposed tie-break, population prevalence, raw-depth exactness, or cross-backend bitwise behavior.