# Localization inference latency: repeat 1 result

Scope: 16 rooms / 128 frozen queries / 1 learned-model timing repeat(s). Every method uses K_ctx=8 and K_gen=1; FEM has 112 observed successful core timings and 16 measured strict-failure/random-candidate fallback timings.
For each method and query, the reported latency is the repeat-1 value; table statistics are then computed across frozen queries.
AGREE/OMP scoring and candidate selection are included. Input loading, candidate filtering, evaluation metrics, and serialization are excluded.

| Method | Mean [s/query] | Median | P90 | Min--max |
|---|---:|---:|---:|---:|
| Vanilla FLAC | 6.456 | 2.104 | 31.742 | 0.338--48.613 |
| OrbitRIR / FA-BF FLAC | 10.919 | 3.719 | 52.829 | 0.734--90.835 |
| Yaw-Augmented FLAC | 6.389 | 2.073 | 32.398 | 0.346--46.097 |
| Few-ShotRIR | 1.263 | 0.400 | 6.457 | 0.057--9.217 |
| FEM--OMP (Depth-AABB) | 695.461 | 70.496 | 1547.088 | 0.017--10214.125 |

| Method | Dataset-wide mean [ms/candidate] | Query median | Query P90 |
|---|---:|---:|---:|
| Vanilla FLAC | 8.923 | 9.556 | 10.410 |
| OrbitRIR / FA-BF FLAC | 15.091 | 16.539 | 20.351 |
| Yaw-Augmented FLAC | 8.830 | 9.177 | 10.640 |
| Few-ShotRIR | 1.745 | 1.769 | 1.907 |
| FEM--OMP (Depth-AABB) | 961.245 | 408.057 | 655.186 |
