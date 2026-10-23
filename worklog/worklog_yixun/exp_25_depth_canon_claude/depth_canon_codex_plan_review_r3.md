**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-10-23 · **Round:** plan r3

**Verdict:** REVISE

## Per-finding resolution

| Round-2 finding | Status | Plan section | Assessment / remaining gap |
|---|---|---|---|
| **B1 — stabilizer / C-4** | **PARTIALLY RESOLVED** | §2.1 Steps C-3–C-4; §6 tests 3–4; §7 rung 3 | The ordered `(r_i, z_i, rel_az_i)` sequence is equivariant in exact arithmetic and preserves original row order, so pose/audio pairing is no longer lost. Fail-closed `StabilizedTieError` plus the zero-image-survivor gate makes the registered runs safe. However, equality of this key does not prove literal equality of every accepted canonical input; see HIGH 1. |
| **B2 — Tier-2 thresholds** | **RESOLVED** | §1 Tier 2; §6 test 21b; §9 | Defensible as a pre-registered **practical, distributional waveform-invariance** criterion: p99 relL2 must be within 3× the matched stagewise composition-control p99 and ≤1e-2; the maximum 1e-1 threshold is correctly an alarm rather than being misrepresented as round-off; the duplicate cell is separately an exact determinism requirement; both relL2 and relL∞ denominators are pinned. It does not establish uniform per-item or round-off-level waveform equality, and the final wording must retain that scope. |
| **H1 — embedded-only binding trigger** | **RESOLVED** | §2.2 protocol binding; §6 test 18 | “In play” now includes an embedded-only `canonicalize` key of any value. Embedded/file blocks are compared type-strictly before construction, missing embedded config fails closed, and an unnecessary `--allow-canon-mismatch` is fatal. The three added cases close the identified escape paths. |
| **H2 — rung-3 audit** | **RESOLVED** | §7 rung 3 | The audit is now a pre-FULL gate requiring zero full-image-key survivors over every referenced train/unseen/seen panorama, with path/hash coverage. The argument is sound: a cyclic roll of a non-periodic image remains non-periodic, and C-4 requires at least two candidates to survive the rooted-image key. Dynamic context enumeration is therefore unnecessary when the gate passes. |
| **H3 — atomic completion** | **PARTIALLY RESOLVED** | §5 Atomic cell completion; §6 test 21 | Resume now skips only a marker-bearing, collector-verified cell, fixing the original metrics-only skip problem. But the specified ordering—rename the directory into place, then write `COMPLETE`—still has a crash window that can strand an unmarked, non-empty final directory which the rerun refuses. See HIGH 2. |
| **Nit — unreachable `same_column`** | **RESOLVED** | §2.1 Step C; §6 test 3; §8 pseudocode | It is now classified from the pixel-level global-maximum set before the unique-column return and is reachable. |
| **Nit — cell total** | **RESOLVED** | §5; §13 | The total is correctly 230: 110 endpoint/diagnostic cells plus 120 non-duplicated trajectory cells. |

## Remaining findings, ranked

No BLOCKING findings.

### HIGH 1 — C-4’s full-input identity statement remains too strong

The ordered context key fixes the row-permutation and audio-pairing defect, but equality does not prove identical canonical inputs under the declared schema:

- The selector keys `source` and `context_poses`, while the input contract permits independently valued `source_vit` and `context_poses_vit`. The current AR loader copies/aliases these fields, but §2.1 does not enforce that relationship.
- A pose with `0 < r < 1e-6` is treated as azimuth-degenerate even though different candidate rotations produce genuinely different XY coordinates. That difference is not merely float32 composition round-off and can exceed test 4’s `atol=1e-6`.

This does not endanger registered runs because C-4 raises and the zero-survivor gate makes it unreachable. Nevertheless, the claimed theorem and candidate-output-equality test are false for some inputs accepted by the plan.

Concrete fix: remove the unnecessary “candidate inputs are identical” claim and the stabilized candidate-output-equality assertion; require only `StabilizedTieError` for every roll. Alternatively, enforce equality between base and `*_vit` pose fields and use exact `r == 0` for an identity claim.

### HIGH 2 — Completion publication is not actually atomic

Section 5 says to rename the staged directory into its final location and then write `COMPLETE`. A crash between those operations leaves:

- a non-empty final directory;
- no completion marker;
- a resume predicate that correctly chooses rerun;
- an evaluator that refuses the non-empty final directory.

Removing only the temporary directory does not repair this state.

Concrete fix: construct and validate the complete cell in a same-filesystem staging directory, write `COMPLETE` inside that staging directory last, then atomically rename the already-complete directory to the final path. Alternatively, explicitly quarantine/remove an unmarked final directory before rerun. Test the crash immediately after rename.

### HIGH 3 — The training-path audit names only the initial P1 commit, although the 40k artifact crossed a resume

`depth_canon_training_path_diff_audit.md` says P1 “trained at `e50d098`.” The exp_07 record says P1 had four legs and that the initial run died around 32.5k before a full-state resume; therefore the 40k checkpoint was produced after a later launch commit.

The scientific conclusion is probably unchanged: `git diff e50d098 1b56e7a -- train.py defaults.ini src` is empty. But the audit must establish this using the actual commit of every leg contributing to the 40k checkpoint, not silently attribute the entire lineage to its first launch commit.

Concrete fix: enumerate each pre-40k P1 leg, its logged HEAD and checkpoint interval, then show training-path hashes/diffs are identical across those commits. If an exact leg HEAD cannot be recovered, disclose that provenance limitation.

### NIT — B-F fixed-angle wording contradicts the ledger

Plan §1 says B-F has historical fixed-angle cells from exp_07/17, while the new comparator ledger correctly states that there are no B-F@40k rotated cells on this box and cites exp_08 A-F—not the B-F endpoint—for method-level FA evidence.

Fix: say B-F contributes its historical θ=0 quality row plus structural/method-level FA evidence from A-F; do not call those B-F endpoint fixed-angle measurements.

### NIT — Clarify the Tier-2 alarm consequence

The criterion is acceptable as p99 relL2 practical invariance, but §1 should say explicitly that exceeding the 0.1 maximum alarm suspends an unqualified invariance conclusion pending item-level triage, even though it is not mechanically part of the p99 PASS predicate. Keep relL∞ descriptive unless a separate acceptance threshold is registered.

## What I verified as correct

- The context key’s original-row ordering preserves each pose row’s association with the untouched `context_audio` row.
- The zero-image-survivor gate is sufficient to make C-4 unreachable under both training inputs and evaluation rolls.
- The binding trigger catches CANON-checkpoint/P1-file/no-flag evaluations before model construction.
- The composition control exercises conditioning, velocity, latent, and decoded waveform under fixed noise; the duplicate cell is correctly separated from this control.
- The `same_column` repair and 230-cell accounting are correct.
- The training audit’s curated eight changed training-path files do total 1,665 insertions and 25 deletions; the AR data/model core is unchanged, and the AcousticRooms metric branches remain behaviorally unchanged.
- Comparator checkpoint hashes match the NAS manifest and local copies. The exp_17 endpoint and C4 values quoted in the ledger match the committed branch artifacts.
- No files, environments, packages, tests, training jobs, evaluations, or GPU commands were modified or run.