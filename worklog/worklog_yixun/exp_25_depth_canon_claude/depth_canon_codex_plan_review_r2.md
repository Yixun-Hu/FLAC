**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-10-23 · **Round:** plan r2

**Verdict:** REVISE

## Per-finding resolution

| R1 finding | Status | Resolving plan sections | Assessment / remaining gap |
|---|---|---|---|
| **B1 — residual tie path and insufficient audit** | **PARTIALLY RESOLVED** | §2.1 Steps C-1–C-4; §6 tests 3–7; §7 rung 3 | The full rooted-image key, typed tie classes, counts, and conditional joint audit are substantial improvements. However, C-4’s “same canonical input” argument is false for the specified sorted context-azimuth multiset; details below. `min(C)` also remains a non-equivariant selector, making §6 test 4 impossible for stabilized inputs. |
| **B2 — checkpoint/config binding** | **PARTIALLY RESOLVED** | §2.2 protocol binding; §6 tests 16–18; §8 `eval_FLAC.py` plan | Type-strict block comparison before mutation/model construction, narrow mismatch authorization, provenance fields, hashes, and `_canon`/`_canonmis` tokens are sound. The remaining trigger gap is that “in play” is defined from the file config, flag, or allow flag—but not an embedded-only canonicalization declaration. A CANON checkpoint paired with a P1 file and no flags could therefore take the legacy no-op branch. |
| **B3 — invariance evidence and waveform artifacts** | **PARTIALLY RESOLVED** | §1 Tiers 1–3; §2.2 sidecar/dump; §5; §6 tests 19–21 | Full stream recording, exact-as-scored waveform dumps, item-level comparisons, canonical-depth hashes, and duplicate cells resolve most of B3. The Tier-2 thresholds do not support the phrase “round-off-level invariance”: p99 1% and especially maximum 10% waveform relative error are scientifically material, and the duplicate cell is a determinism floor rather than a rotation-composition numerical floor. |
| **H4 — “exact depth recovery” overclaim** | **RESOLVED** | §2.1 Step A; §6 tests 1 and 9; §7 rung 3a | Correctly renamed to a bitwise yaw-stable recovered-depth proxy; its raw-depth discrepancy is disclosed; odd-height/zero-sine cases fail closed; the real-data audit compares raw and proxy candidates/selections over all maps. |
| **H5 — unsupported four-arm random-yaw comparison** | **RESOLVED** | §1 comparator estimands; §5 P1-pin/YAW-pin blocks and ledger; §9 | Random-yaw claims are now limited to CANON/P1/YAW at one pin. B-F is explicitly excluded from random yaw and retained only as historical fixed-angle evidence. |
| **H6 — wrong Yaw-Aug table row** | **RESOLVED** | §0; §5 YAW-pin; §8 generator change; §13 H6 | The plan re-measures the matched-A6000 YAW endpoint and adds a raw-artifact-backed generator row, avoiding substitution of exp_15’s Neuronic row. This resolution is contingent on A1 being approved; the plan correctly blocks later work pending that decision. |
| **H7 — historical rather than strict causal pair** | **RESOLVED** | §1 Arms/H-Q; §7 rung 4; §9 | The claim is now “single intended config/algorithm delta against a historical matched-recipe control,” with a training-path diff audit and appropriately non-causal interpretations. |
| **H8 — missing load-bearing tests** | **PARTIALLY RESOLVED** | §6 tests 1–22 | Nearly all requested coverage was added. Remaining defects are the impossible stabilized-selector equivariance assertion, lack of a pair-aware full-input stabilizer test, the embedded-only binding case, and the internally unreachable `same_column` test/classification. |
| **H9 — run-owned checkpoints and NAS gates** | **RESOLVED** | §4 launcher; §6 launcher guards; §10 | Empty run namespace, NAS identity/write/free-space checks, launch-time ownership gate, exact cadence, HEAD/manifest binding, and smoke invalidation after reviewed source changes are all specified. |
| **N10 — method naming** | **RESOLVED** | §2.1 Exactness statement; §9 | The plan now discloses that periodic-image cases use a joint geometry-plus-query canonicalizer. |
| **N11 — “zero cost” wording** | **RESOLVED** | §2.1 Exactness statement; §4 wall-clock; §7 smoke | The plan no longer claims literal zero cost and requires measured preprocessing latency. |

## Remaining and new findings

### BLOCKING — C-4 does not establish a stabilizer of the actual paired model input

What C-4 establishes is narrower than claimed:

- Equality of the full rooted scalar-depth images establishes that the recovered `D_canon` is bitwise identical.
- A surviving source tie implies the source is azimuth-degenerate, because any nondegenerate source distinguishes two different candidate anchors in exact arithmetic.
- Equality of the **sorted context azimuth multiset** establishes only that the context angles form a rotation-invariant multiset.

It does **not** establish equality of the canonical input:

- The key discards context radius and height, so the same azimuth multiset can attach different `(r,z)` values to those angles.
- Candidate rotations do not reorder context rows. Thus the ordered `context_poses` and `context_poses_vit` tensors generally differ even when their azimuth sets agree after a permutation.
- `context_audio` remains in its original row order. If a geometric symmetry permutes context rows, the associated audio rows are not co-permuted. The set of paired `(pose_i, audio_i)` observations therefore changes unless the corresponding audio rows also happen to be identical.
- The current FLAC model embeds pose and audio modalities separately and concatenates their cross-attention tokens. Under the stronger condition that the **full pose-token multiset** is merely permuted, current cross-attention may be insensitive to that permutation. But the proposed key does not prove that stronger condition, and it certainly does not prove literal input identity.

Moreover, `min(C)` is not equivariant on a stabilized orbit. For example, with candidates `{0,256}`, rolling by 300 gives `{300,44}`; `min` changes from the co-rotated candidate 300 to 44. Therefore §6 test 4’s assertion

`j*(roll_d(x)) = (j*(x)+d) mod W`

cannot hold for the stabilized fixture. Only the **canonicalized output** could remain invariant, and only after a valid full-input stabilizer proof.

Concrete fix:

- Prefer an ordered, pair-preserving context key: compare each original row’s full relative tuple `(r_i, z_i, wrapped_relative_azimuth_i)` in row order. Because audio remains in that same order, equality then preserves pose/audio pairing.
- Alternatively, treat contexts as a set of pairs, but then derive a stable pair identity and permute `context_poses`, `context_poses_vit`, and `context_audio` together.
- For C-4, directly compare every candidate-canonicalized field: scalar depth bitwise, 3-D depth and all four pose tensors under registered float tolerances, and untouched fields—including `context_audio`—exactly.
- Change §6 test 4 to require selector equivariance only for non-stabilized cases. For stabilized cases, test candidate-output and all-roll canonical-output equivalence.
- The safest policy remains fail-closed on C-4. Because the audit expects zero periodic proxy images, fail-closed should not threaten the real 43-hour run if that expectation is correct.

### BLOCKING — Tier-2 thresholds are not a defensible “round-off-level” standard

Repository evidence gives useful scale: exp_03 measured conditioning relative error `4.9e-8`, latent error `3.7e-7`, and decoder-amplified waveform relative L2 around `4.6e-4` in fp32 and roughly `1.1e-3` under autocast. Exp_08 later registered a full-split bf16 aggregate floor of `0.00931`. Those results make the proposed median `1e-3` and perhaps p99 `1e-2` plausible as exploratory practical thresholds, but they provide no item-level-tail justification, and a maximum relative L2 of `0.1` is not round-off-level invariance.

The duplicate θ=0 cell should be exactly reproducible and mainly detects RNG/context/order drift. It does not exercise the direct-versus-composed rotation path and therefore cannot measure rotation-composition error.

Concrete fix:

- Add a fixed-noise, stagewise numerical-composition control comparing direct canonical metadata against `rotate(d)` followed by candidate-dependent canonical rotation.
- Measure conditioning tensors, one-step model/velocity output, sampled latent before decode, and final decoded waveform. This mirrors exp_03’s successful amplification diagnosis.
- Pre-register an explicit rule against the corresponding composition-control distribution, such as a fixed multiplier plus a hard absolute cap. Keep the duplicate cell as a separate determinism requirement.
- Define relative L2/L∞ denominators and their near-zero floor exactly.
- If the current `median/p99/max` limits are retained, call them **practical waveform invariance**, not round-off-level invariance; treat `max ≤ 0.1` as an alarm bound, not a scientific PASS criterion.

### HIGH — The binding trigger must include an embedded-only declaration

Section 2.2 should define canonicalization as “in play” when **either** the checkpoint’s embedded config or the file config contains a `training.canonicalize` block, or when either CLI flag is supplied. Otherwise the most dangerous mismatch—CANON checkpoint plus P1 file, no flags—can escape before comparison because the weights load structurally.

Also require `--allow-canon-mismatch` to correspond to an actual `trained != applied` mismatch; an unnecessary allow flag should fail rather than silently record an ordinary cell.

Add explicit tests for:

- embedded enabled, file absent, no flags;
- embedded false block, file absent, no flags;
- `allow=True` when trained equals applied.

The proposed provenance fields and filename tokens are otherwise sufficient and injective for the stated combinations.

### HIGH — Rung 3 is conditionally sufficient, not sufficient in every branch

The depth-first audit is logically efficient: C-4 is unreachable unless multiple candidates survive the full rooted-image key. Therefore, if the runtime-exact recovered proxy has zero such images across a hash/path ledger covering every panorama referenced by train, unseen, and seen, the depth-only pass certifies a stabilized count of zero for every possible source/context draw.

If even one proxy-periodic image survives, however, §7 rung 3b cannot truthfully enumerate “every `(panorama, source, context set)` triple” from the split JSONs. `AR_md.py` samples context sources and their order dynamically with `np.random.choice` on every load; the context set is not stored in those split files. One offline draw does not certify training or all evaluation seeds.

Concrete fix: make zero image-key survivors a pre-FULL gate. If survivors exist, either fail closed, enumerate every possible ordered context draw for affected targets, or limit certification to the actual recorded evaluation streams and runtime-observed training counts.

### HIGH — New scored-waveform artifacts need an atomic completion contract

Section 5’s driver resumes by “skip-if-exists,” while a cell now has several required artifacts: metrics, stream sidecar, optional canonical sidecar, waveform array, checksum, and manifest. A crash after metrics but before waveform finalization could cause an incomplete cell to be skipped permanently; a partially populated dump directory then conflicts with the “refuses a non-empty directory” rule.

Use a temporary per-cell directory, validate every artifact and checksum, then write/rename a completion marker last. Resume should skip only a fully verified cell. Extend filename/directory injectivity tests to the scored-waveform dump path, not only legacy metrics and prediction names.

### NIT — Two internal accounting/specification errors

- `same_column` is unreachable in the §8 pseudocode because `P[w]` produces unique column indices and `len(cands)==1` already returns `none`. Either classify multiple globally maximal pixels in the sole winning column before that return, or remove `same_column` and its test.
- The §5 blocks total **230** generative cells, not 232: 110 endpoint/diagnostic cells plus 120 new trajectory cells. The storage estimate remains approximately 60 GB.

## Verified as correct

- The full rooted-image comparison is the right strengthening: equal rooted images at two distinct candidates imply exact periodicity of the recovered scalar panorama.
- The source-relative-azimuth key is equivariant in exact arithmetic and, for a nondegenerate source, distinguishes distinct anchors. The modulo placement correctly handles the ±π boundary.
- The roll direction and rotation sign remain correct.
- `z/(-sin φ)` matches `AR_md.py`’s construction and is correctly described as a yaw-stable proxy rather than an exact raw-depth inverse.
- Rotation → canonicalization → conditioning is the correct evaluation order; canonicalization before conditioning at all three wrapper sites is also correct.
- Co-rotating all four pose fields while leaving the mono target and context RIR waveforms unchanged is physically correct.
- The B2 type-strict comparison, pre-model placement, mismatch-record fields, config hashes, and filename-token design are otherwise sound.
- Saving the post-decode, post-clamp, post-padding tensor passed to `update_metrics` satisfies announcement 08; leaving legacy `--store_predictions` unchanged is appropriate.
- Stream recording and expected-count checks are now applied to fixed, random, and duplicate paired cells.
- The random-yaw comparison framing, YAW re-measurement, historical B-F labeling, historical-control language, raw/proxy audit, NAS gates, checkpoint cadence, and measured-cost disclosure all correctly address their respective round-1 findings.