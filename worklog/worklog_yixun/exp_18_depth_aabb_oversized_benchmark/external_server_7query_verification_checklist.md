# External-server seven-query verification checklist

Use this checklist before merging the external-server Cafe/Auditorium results
into the formal 112-query FEM--OMP table.

## 1. Frozen-protocol identity

The internal canonical hashes of the three frozen protocol files must be:

```text
selection: 67bdd25f3df704bc1c57558e7cb68cfaa5d9e60758f2c70e87a59eddc33bcfa9
context:   b757da281dcde3ffc310aac67279a240dac5cb1ff1d9966bf918f69c4dde6f58
geometry:  ae09d9cf9416866d09dea498a1f8467e952866db8b1c914ed0bea6a75e06cf9a
```

These are the values of the JSON documents' internal `sha256` fields. They are
canonical payload hashes and do not necessarily equal `sha256sum file.json`.

## 2. Query identity and frozen candidate count

For every result, verify `query_index`, `query_id`, `room`, `receiver_id`, and
`candidate_count` against the frozen selection manifest.

| Query | Room | Candidate count | Candidate-indices SHA-256 |
|---:|---|---:|---|
| 715 | Cafe_idx_1 | 5296 | `bcf112a9c78877e8494725fabdafdf20eb5181f452ae8a1f61e1705661afb0f1` |
| 30 | Cafe_idx_1 | 5296 | `54c8c7f0b11328b48cd5b2c7b7076778a647a6bf91a743f33141e207b7f8846a` |
| 42 | Cafe_idx_1 | 5296 | `5aebf857bed581a0c141bf06ffcfd14fb017ea3a6832e40d89ca45946ec32d8f` |
| 535 | Cafe_idx_1 | 5296 | `a2da9eed2a3005ef254e18ad4745ee92e2dd147fc0c6f7f74091981482ea7bf3` |
| 917 | Cafe_idx_1 | 5295 | `53d0d736723114bd4225aa848b0516d03a025685f902cdc6ae640b045e1291bb` |
| 3800 | Auditorium_idx_1 | 3723 | `2b0a0f777292c5aae6074532c20c6b584eb39bb276a6672df76700533588603f` |
| 3841 | Auditorium_idx_1 | 3723 | `6ac3d4b3d1e97bb8102a95a4482a54e5ebc65e4fb289bd2f7532cc48309eeae5` |

Because the OMP result archive stores candidate coordinates rather than their
original indices, the strongest candidate-set check is to reconstruct the
frozen candidates from the same geometry audit and confirm exact equality with
the `candidates` array in the scores archive.

## 3. Result-artifact hashes

Each query must provide these three files:

```text
query_XXXXX_depth_aabb_result.json
query_XXXXX_depth_aabb_scores.npz
query_XXXXX_depth_aabb_mesh.npz
```

Run:

```bash
sha256sum query_XXXXX_depth_aabb_scores.npz
sha256sum query_XXXXX_depth_aabb_mesh.npz
```

The values must equal `arrays_sha256` and `mesh_sha256`, respectively, in the
corresponding result JSON. Also hash the file referenced by `depth_path` and
confirm that it equals `depth_sha256`.

## 4. Result JSON self-hash

The `sha256` field in a result JSON is the canonical hash of the JSON payload
after removing the `sha256` field itself. It is not the byte-level output of
`sha256sum result.json`.

```python
import hashlib
import json

path = "query_00715_depth_aabb_result.json"
payload = json.load(open(path))
expected = payload.pop("sha256")
actual = hashlib.sha256(
    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()

print("PASS" if actual == expected else "FAIL", actual, expected)
```

Repeat this check for all seven result JSON files.

## 5. Method and numerical consistency

Every result must satisfy:

```text
method == fem_sabine_depth_aabb
context_count == 8
coverage_protocol.strict_gate_passed == true
fem_audit.frequency_count == 102
fem_audit.solver_profile.frequency_count == 102
fem_audit.maximum_allowed_edge_m == 0.22
fem_audit.maximum_relative_solver_residual < 7e-13
```

For each `query_XXXXX_depth_aabb_scores.npz`, also verify:

- The `candidates` and `scores` lengths equal `candidate_count`.
- `argmax(scores) == metrics.prediction_index`.
- `metrics.prediction_index == sparse_recovery.support[0]`.
- `metrics.prediction_global` equals the selected candidate coordinate.
- The Euclidean distance from `prediction_global` to `source_global` reproduces
  `metrics.localization_error_m`.
- The arrays and reported scalar metrics contain no `NaN` or infinite values.

## 6. FEM--AGREE response-cache check

This section is not required for merging the FEM--OMP table. It matters only
if these queries will later be rescored with AGREE.

Check whether each query also has:

```text
query_XXXXX_response.npz
```

An OMP `scores.npz` contains candidate coordinates, scalar OMP scores, and
frequency coordinates; it does not contain the full complex FEM response
needed by AGREE. If a response-cache audit JSON is present, verify:

```text
method == fem_sabine_depth_aabb_response_cache
response_file_sha256 == sha256sum(query_XXXXX_response.npz)
source_fem_result_sha256 == the OMP result JSON's internal sha256
source_mesh_sha256 == the OMP result JSON's mesh_sha256
```

## Merge gate

**Status: PASS (user-confirmed on 2026-08-30).**

The seven-query external slice is accepted for the formal FEM--OMP aggregation.
Preserve the external-server artifacts in a separate source directory and do
not overwrite a locally produced duplicate. Query `715` is retained as a
cross-server replication check and is counted only once.
