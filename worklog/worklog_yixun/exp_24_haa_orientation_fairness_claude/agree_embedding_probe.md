# Inside the retrieval metric: fidelity vs distinctiveness (HAA, ckpt-1000, K = 8, seed 42)

*AGREE audio embeddings (unit norm) of each arm's stored predictions and of the ground truth, the evaluator's own encoder and crop. Ranking is within room (the paper convention averages the per-room rankings).*

| arm | R@1 | R@5 | R@10 | fidelity cos(pred,own GT) | margin to the nearest other receiver | shared-offset share of the embedding error |
|---|---|---|---|---|---|---|
| FLAC | 5.13 | 19.20 | 31.62 | 0.5479 | -0.1694 | 21.6 % |
| FLAC+cue | 4.84 | 18.50 | 30.66 | 0.5430 | -0.1711 | 20.9 % |
| CylDINO+cue | 4.96 | 18.57 | 31.43 | 0.5433 | -0.1676 | 21.4 % |

After subtracting each arm's own shared offset from every prediction embedding (re-normalised):

| arm | R@1 | R@5 | R@10 | fidelity |
|---|---|---|---|---|
| FLAC | 5.01 | 20.83 | 32.96 | 0.6228 |
| FLAC+cue | 4.75 | 19.35 | 32.15 | 0.6157 |
| CylDINO+cue | 5.43 | 21.58 | 33.19 | 0.6188 |

## Per room (R@10 / fidelity / margin)

| arm | classroomBase | complexBase | dampenedBase | hallwayBase |
|---|---|---|---|---|
| FLAC | 28.1 / 0.596 / -0.147 | 41.4 / 0.584 / -0.169 | 22.7 / 0.463 / -0.232 | 34.3 / 0.548 / -0.129 |
| FLAC+cue | 24.4 / 0.589 / -0.152 | 46.5 / 0.592 / -0.153 | 18.7 / 0.448 / -0.251 | 33.1 / 0.543 / -0.129 |
| CylDINO+cue | 27.9 / 0.609 / -0.139 | 43.4 / 0.576 / -0.165 | 22.7 / 0.452 / -0.240 | 31.7 / 0.536 / -0.127 |