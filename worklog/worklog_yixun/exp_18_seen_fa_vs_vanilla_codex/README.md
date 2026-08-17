# exp18 — exp07 FA vs Vanilla on FLAC seen

This worklog evaluates the EMA weights in the two 40k checkpoints on the full
FLAC seen split (6,217 items / 131 rooms):

- FA: `exp07_BF`, `fa_invariant`, C4 angles `0,90,180,270`.
- Vanilla: `exp07_P1`, vanilla conditioning.
- K in `{1, 8}` and evaluation seeds `42..46` (five generations from each one
  training checkpoint), `cfg=1.0`, one sampling step, bf16 conditioning.
- Reported dispersion is the sample standard deviation (`ddof=1`).

The standard FLAC loader replaces target waveforms whose peak level is below
`-60 dB` with another seeded sample.  The full seen split contains such files
(for example split index 1194 is `-60.77 dB`), so the newer optional assignment
stream guard is incompatible with the historical/Table A.2 evaluation path.
These runs preserve the canonical loader behavior and still validate that every
record reports all 6,217 evaluation positions; the split/config is not reduced.

The two local-GPU shards keep each paired `(K, seed)` FA/vanilla comparison on
the same physical A6000:

```bash
python run_seen_eval.py --gpu 0 --shard 0
python run_seen_eval.py --gpu 1 --shard 1
python aggregate_seen_results.py
```
