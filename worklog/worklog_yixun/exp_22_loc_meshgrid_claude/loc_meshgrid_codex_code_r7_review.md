**Reviewer:** OpenAI Codex `gpt-5.6-sol` at Extra High reasoning (codex-cli 0.147.0, read-only) · **Date:** 2026-08-25

*Round exp22-r7 (engine). REJECT/launch blocked: 7 BLOCKERs + 1 MAJOR, with a complete post-fix launch spec (probe sweep + balanced Cafe/rest shard split + merge census). Substantial mechanics verified incl. exact cost-gate recount. Body verbatim.*

---

## Verdict

**REJECT / launch blocked.** Current `localization-exp@113ac94` clears substantial I1 mechanics, but neither the registered throughput probe nor P1 is authorized yet.

### Blocking findings

- **[BLOCKER CRN] Registered noise policy is not shipped.** The binding ruling says CRN, but [`NOISE_KEY_POLICY`](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_engine.py:81) remains `per_candidate`; the driver inherits it at [localize_meshgrid.py:83](/home/yixunhu/codespace/FLAC/localize_meshgrid.py:83), and the test explicitly pins the wrong default at [test_loc_meshgrid_engine.py:55](/home/yixunhu/codespace/FLAC/src/tests/test_loc_meshgrid_engine.py:55). Set the production policy to `shared_across_candidates` and refuse `per_candidate` in a registered run. The CRN implementation itself is correct when explicitly selected.

- **[BLOCKER GT] The engine reads ground truth.** [`assert_receiver_consistent`](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_engine.py:634) reconstructs truth from `md["source"]`; the pass copies that field at [meshgrid_engine.py:1260](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_engine.py:1260). Earlier, D1 verification calls `context_record`, which also reads `source` at [meshgrid_queries.py:145](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_queries.py:145). Remove all engine-side target access. The D1 position/query/context checks plus verified G1 query mapping are sufficient. Add a sentinel test proving `source` cannot be accessed.

- **[BLOCKER RESUME] Unknown rows can be adopted, and row contents are not digest-protected.** On `--resume`, a missing binding is silently created before existing rows are scanned at [localize_meshgrid.py:264](/home/yixunhu/codespace/FLAC/localize_meshgrid.py:264). Thus rows without provenance can be legitimized. Also, [`verify_query_artifact`](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_engine.py:805) authenticates the similarity sidecar but not the row’s predictions, oracle, candidate indices, or binding identity. A changed `prediction_xyz` or `e_oracle` can still be skipped. Require an existing valid binding for every resume, stamp each row with its binding digest, and verify row identity against the expected D1/G1 query.

- **[BLOCKER BINDING/BATCHING] Result-affecting configuration is incomplete and advisory provenance is not durable.** `cond_autocast` is absent from [`build_run_binding`](/home/yixunhu/codespace/FLAC/localize_meshgrid.py:191), while the dataset config is stored only as a pathname, not a hash. A resumed run can therefore mix different conditioner arithmetic or an edited dataset config. Changes to `batch_rows`/`source_chunk` are printed only to stdout; neither rows nor the final summary preserve which queries used which batching. Bind `cond_autocast` and the dataset-config SHA; persist every advisory change and stamp effective batching per row.

- **[BLOCKER DETERMINISM] Advisory tolerance semantics are not defined end-to-end.** Currently bit-exact are: keyed noise rows, K-prefix slicing, and cache memoization at matched batching. The real parity log only measured conditioner-token drift up to `3.90625e-3`; there is no registered similarity/score tolerance, no argmax-stability condition, and no real fixed-batching replay test. Fixed-batching scoring is intended to be deterministic through the mean AGREE readout, but it is not test-pinned. Define the accepted score tolerance and add fixed-batching replay plus changed-batching bounded-difference tests.

- **[BLOCKER PROBE] The diagnostic cannot independently substantiate the cost decision.** [`write_probe_records`](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_engine.py:985) omits the binding digest, checkpoint/config/manifests, batching values, top-level context timing, and conditioner-row count. Source-cache time is repeated per query without a receiver/group identity. The prior smoke also used the invalid per-candidate noise policy. The registered artifact must carry immutable provenance and enough counts/timings to project separately over 71,172,320 waveforms, 5,337 contexts, and 966,147 source rows.

- **[BLOCKER SHARDING] `--rooms` and a census-gated merge are absent.** The parser at [localize_meshgrid.py:61](/home/yixunhu/codespace/FLAC/localize_meshgrid.py:61) has no room filter, and no merge tool exists. This is the binding in-scope addition.

- **[MAJOR DUMP] `--dump-cases` is self-authorizing.** Any JSON list extends the allowed set at [localize_meshgrid.py:167](/home/yixunhu/codespace/FLAC/localize_meshgrid.py:167); its computed digest is discarded. Bind the registered case-list digest, and make resume verification check any waveform sidecar named by a row.

### Checks that pass

- The runtime order currently builds the actual scorer/model before the D1 helper reseeds, then verifies the registered RNG digest before iteration. Per-record context fingerprints and audio digests are checked before conditioning. However, the driver-level ordering is not test-pinned.
- `S = 0.1 * (logsumexp(s/0.1) - log K)`, `S_mean`, and global-index lexicographic tie-breaking are correctly implemented.
- Z-band indices address the room-global lexicographically ordered base bank correctly.
- The float16 per-sample sidecar limitation is clearly disclosed; aggregate scores remain exact float32 hex.
- Independent recount: **16 rooms, 540 receiver groups, 8,896,540 candidate-query pairs, exactly 966,147 union rows**.
- Probe mode emits no score-bearing row or sidecar; it only computes similarities internally for timing.
- Ordinary row/sidecar publication is sidecar-first and atomic.

## Required `--rooms`/merge contract

The filter must accept canonical room IDs, reject duplicates/unknown/empty selections, enter the strict shard binding, and still iterate and verify the complete D1 stream so context draws remain unchanged. Unselected rooms must not load G1 manifests or condition/embed observations.

The merge must publish into a fresh directory only after verifying:

- identical strict base bindings and pinned production advisory values;
- disjoint declared room sets;
- exactly the 16 registered rooms and 5,337 expected query identities/positions;
- no duplicate or extra rows;
- every row/sidecar/binding digest;
- totals of 8,896,540 candidate-query pairs, 966,147 source rows, and 71,172,320 generated waveforms.

## Conditional launch calls

These are **post-fix specifications; do not run them against current HEAD**.

Registered probe sweep:

```bash
for batch_rows in 64 128 256; do
  python localize_meshgrid.py \
    --ckpt-path weights/exp20/P1_40k.ckpt \
    --agree-ckpt weights/AGREE/AGREE_fullAR.pt \
    --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
    --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json \
    --context-manifest outputs_loc/exp22/d1_context_manifest.json \
    --audit-report outputs_loc/exp22/g1_audit/geometry_audit_report.json \
    --branch z_band \
    --device cuda:0 \
    --probe 1 \
    --probe-room Bathrooms/Bathrooms_idx_14 \
    --probe-stem "P1_CRN_br${batch_rows}" \
    --out-dir "outputs_loc/exp22/i1_probe_P1_br${batch_rows}" \
    --noise-policy shared_across_candidates \
    --seed 42 --tau 0.1 --num-samples 8 --k-prefixes 1 4 8 \
    --steps 1 --cfg-scale 1.0 --cond-autocast default \
    --source-chunk 16 --batch-rows "${batch_rows}"
done
```

Select and pin one production `batch_rows` only if the corrected, provenance-bound projection is ≤175 GPU-hours.

The balanced P1 split should be:

- GPU 0: Cafe only — 4,559,398 candidate-query pairs.
- GPU 1: remaining 15 rooms — 4,337,142 pairs.

Both full-pass calls are conditional on the new `--rooms` implementation and use separate output directories. A subsequent mandatory merge must satisfy the census above. Initial launches must omit `--resume`; restarts may add it only after the resume fixes land.

No files, environments, packages, or GPU state were changed. Tests were not executed because of the strict read-only/no-write constraint.