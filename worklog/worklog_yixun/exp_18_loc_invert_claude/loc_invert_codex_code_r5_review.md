**Reviewer:** OpenAI Codex `gpt-5.6-sol` at Extra High reasoning (codex-cli 0.147.0, `codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort=xhigh`) · **Date:** 2026-08-19

*Round marker: r5 (launch gate v2). Verdict REQUEST-CHANGES — single residual (H2: split digest + wav floor); 5/6 RESOLVED; peer import compatibility for `6c0a16e` confirmed. Body verbatim.*

---

**Reviewer:** OpenAI Codex (GPT-5 API workspace agent, read-only review) · **Date:** 2026-08-19 · **Round:** r5

# Verdict: REQUEST-CHANGES

Five findings are resolved. H2 remains partially open because the executable R-1 gate still accepts a noncanonical split and finite, mono but truncated WAVs.

| Finding | Status | Verification |
|---|---|---|
| H1 — GPU timing | **RESOLVED** | Timers synchronize the resolved CUDA index before and after every interval; scoring’s device transfer, context evidence, and synchronized wall time are included. |
| H2 — readback teeth | **PARTIALLY** | M=10/17 rooms, LRH S10, every referenced depth map, and representative WAV decoding are enforced, but exact split identity/count and registered WAV length are not. |
| M3 — auxiliary overwrites | **RESOLVED** | All report modes use atomic no-clobber publication; the accepted primary-stem deviation presents no correctness hole because provenance identifies the cell and collisions refuse. |
| M4 — registration committish | **RESOLVED** | Registration requires a full lowercase object ID, an in-worktree byte-identical manifest, commit ancestry of HEAD, and records the resolved SHA. |
| M5 — validation ordering | **RESOLVED** | Dataset/model configuration, context, registration, hashes, candidate manifest, and output collision checks precede checkpoint deserialization and model/dataloader construction. |
| M6 — scorer noise | **RESOLVED** | Scorer-noise requires a seen configuration, realpath-binds explicit WAVs to its enumeration, seeds immediately before stochastic draws, and records the seed. |

## Blocking H2 residual

The gate reads its authority from the referenced split JSON, then checks only `17` rooms and `10` nodes per room—not the registered 6,337 identities, exact room/node map, or immutable split digest. Consequently, a truncated or same-shaped substituted split can pass, and the later identity audit merely agrees with that same altered authority. Anchors: [eval_localization.py:75](/home/yixunhu/codespace/FLAC/eval_localization.py:75), [eval_localization.py:1942](/home/yixunhu/codespace/FLAC/eval_localization.py:1942), [eval_localization.py:2027](/home/yixunhu/codespace/FLAC/eval_localization.py:2027), [eval_localization.py:2043](/home/yixunhu/codespace/FLAC/eval_localization.py:2043).

Likewise, WAV validation accepts any finite mono waveform with at least one sample; it does not enforce the observed registered shape `(1, 64542)` or even the minimum samples consumed by the evaluation pipeline. A truncated nonempty RIR therefore passes R-1 and can change oracle, context, or query scores. Anchor: [eval_localization.py:1994](/home/yixunhu/codespace/FLAC/eval_localization.py:1994).

**Required fix:** pin and check the canonical unseen identity-stream/split digest plus `6,337` count and exact room/node map, carry that digest into registration, and enforce the registered WAV length on the defined readback sample. Add same-count wrong-room/node, truncated-split, and finite nonempty truncated-WAV fixtures.

## Peer import compatibility

Confirmed for `6c0a16e`: `sample_target_id`, `canonical_stream_hash`, `source_sha`, `orbit_provenance`, `resolve_cond_autocast`, `check_load_integrity`, `resolve_are_from_checkpoint`, and `resolve_weights_source` are behaviorally unchanged. `sample_context_ids` adds only the RAF `context_capture_ids` branch; AR metadata lacks that key and follows the previous float32 validation and six-decimal rendering exactly.

## New findings

None beyond the remaining H2 correctness gap.

Non-gating nits: scorer-noise stems omit the selected WAV set/`--noise-wav-count`; `total_wall` excludes row construction/JSONL writing; no-clobber checking is not an interprocess-exclusive claim.

## Launch assessment

R-1/R0/R1 remain held because R-1 can still certify data that violate the registered full-split or WAV-length contract. The supplied 492-test, exact-parity, real-readback, and S10-deletion evidence validates the implemented portions, and no generation-path regression or AR import change was found. After the focused H2 fix and a green real readback, the other five findings require no further changes and the existing parity evidence remains applicable.