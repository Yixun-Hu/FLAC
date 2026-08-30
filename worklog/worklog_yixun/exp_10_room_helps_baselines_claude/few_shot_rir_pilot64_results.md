# Few-ShotRIR-Waveform AR localization pilot

> Historical batch-1 result. The aligned two-batch, 128-query result is in
> `few_shot_rir_128_results/summary.md`.

## Scope and identity

- Frozen seed-42 pilot: 64 queries, four queries per room, 16 rooms, 46,301 query-candidate pairs.
- Acoustic contexts: nested `K_ctx={1,8}`.
- Candidate score: cosine similarity from the frozen AGREE waveform encoder.
- Few-ShotRIR checkpoint: `best-00100000.ckpt`, SHA-256 `f1c83309b7821f6998ed8f754fb6e35fb2a871af836e47240fc101fdb9ae2119`.
- Model configuration SHA-256: `236b791a4d5f9595c1ab76a8219d8a1fbbc8c2c2da19a26438b55e9c973012fd`.
- Pilot manifest SHA-256: `6eeeec401c3f63e47ab446b4b43efe2f1db260bad9decf74de423d0872b659a2`.
- Evaluation run SHA-256: `834a5d0701e09ddafcc123bfd5e17bac7f6f02ad6cd6604169b541d7ea495bbd`.
- All query JSON/NPZ artifacts passed their content-hash, identity, shape, and finite-value checks.

This is a room-stratified diagnostic pilot, not the complete 5,337-query evaluation.

## Primary localization results

| Method | Context / samples | Mean error (m) | Median error (m) | Success@0.5 | Success@1.0 | Oracle-normalized@0.5 |
|---|---:|---:|---:|---:|---:|---:|
| Few-ShotRIR-Waveform | `K_ctx=1` | 3.151 | 1.735 | 0.047 | 0.234 | 0.188 |
| Few-ShotRIR-Waveform | `K_ctx=8` | 3.109 | 1.517 | 0.031 | 0.250 | 0.141 |
| Random candidate | — | 3.209 | 2.320 | 0.078 | 0.188 | 0.141 |
| Vanilla FLAC | `N_ctx=8, K_gen=1` | 1.748 | 1.009 | 0.156 | 0.500 | 0.234 |
| FA-BF FLAC | `N_ctx=8, K_gen=1` | 1.816 | 0.928 | 0.281 | 0.531 | 0.406 |

The FLAC rows are from the existing exp_09 run on the exact same pilot identities and candidate arrays. `K_ctx` and `K_gen` are different axes; the fairest available forward-model comparison is Few-ShotRIR at `K_ctx=8` against FLAC at `N_ctx=8, K_gen=1`.

## Interpretation

Few-ShotRIR shows weak source-location signal but does not establish an advantage over random selection. At `K_ctx=8`, its paired mean error difference from random is -0.100 m; a deterministic 20,000-draw room bootstrap gives a 95% interval of `[-0.588, 0.410]` m. The interval crosses zero. It beats random on 38 queries, ties on one, and loses on 25, while its strict 0.5 m success rate is lower than random.

It is clearly weaker than the existing Vanilla FLAC pilot. The paired mean error penalty at `K_ctx=8` versus Vanilla `K_gen=1` is +1.361 m, with a room-bootstrap 95% interval of `[0.462, 2.551]` m. Few-ShotRIR loses this paired comparison on 44 of 64 queries.

Increasing acoustic context from one to eight does not produce a consistent improvement. Mean error changes by only -0.042 m; 17 queries improve, 28 retain equal localization error, and 19 worsen. Success@0.5 decreases from 4.7% to 3.1%.

Candidate scores are not completely uninformative: the mean per-query Spearman correlation between AGREE score and negative source distance is 0.235 for `K_ctx=1` and 0.227 for `K_ctx=8`. This is positive but too weak for reliable top-1 localization.

The largest failures occur in `Auditorium_idx_1` and `Cafe_idx_1`, whose `K_ctx=8` room-mean errors are 11.111 m and 11.279 m. Excluding these two rooms as a diagnostic only, Few-ShotRIR still has 1.954 m mean error versus 2.096 m random, 1.142 m Vanilla, and 1.072 m FA-BF. Therefore room scale explains part, but not all, of the gap.

## Runtime and implementation correction

The 64 queries required 186.2 seconds of measured query work on one RTX A6000, averaging 2.91 seconds per query; large candidate grids dominate runtime.

The first real-data preflight exposed an adapter bug: pilot receiver identifiers such as `R006` were incorrectly coerced to integers only while serializing a completed result. The runner now preserves the frozen identifier verbatim, and the baseline integration tests cover this case. No failed preflight query artifact or score entered the formal 64-query output directory.
