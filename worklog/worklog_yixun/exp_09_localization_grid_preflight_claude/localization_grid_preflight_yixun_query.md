# Yixun's queries — exp_09_localization_grid_preflight

## Query 1 — establish the localization task

### Verbatim

> 读一下这个experiment和NeuriPs_Workshop/acoustic_localization_brief.pdf，明确一下我们的任务

The attached experiment text is preserved verbatim at the original attachment path:

`/home/zhixuanzhao/.codex/attachments/dc7d2f84-35f1-43ad-b742-b8bf9ae2c754/pasted-text.txt`

Its driving request is **Experiment 1: Inverting Vanilla FLAC for Source Localization**: use frozen Vanilla FLAC as an analysis-by-synthesis forward model, compare generated RIRs to one held-out observed RIR with the frozen AGREE acoustic encoder, and recover the source by candidate search on the full AcousticRooms unseen-room split.

### Summary

Determine whether a pretrained forward acoustic model contains enough source-position information to localize a held-out source without training a localization network.

### Assumption / hypothesis

Candidate locations near the true source should generate RIRs whose frozen AGREE acoustic embeddings are more similar to the observed RIR than candidates elsewhere in the room.

### Why this experiment needs to run

Forward RIR metrics do not establish invertibility. This experiment directly measures whether FLAC preserves candidate-specific spatial evidence that can be recovered at inference time.

## Query 2 — scoring precedence and initial candidate interpretation

### Verbatim

> 我还是有几个地方需要明确，一个是loss function，以pdf为主，另一个是怎么划分候选点有明确定义吗

> 先按照用metadata 已有 source coordinates做吧，如果没有区分度再更换三维密集网络

> 除了score参考pdf，其他一律遵循experiment文本

### Summary

The PDF controls the candidate score: cosine similarity in frozen AGREE acoustic space, aggregated across stochastic samples by log-mean-exp. All non-score protocol choices follow the experiment text unless superseded by a later explicit decision.

### Assumption / hypothesis

The score is not a training loss: FLAC and AGREE remain frozen, and localization is inference-only.

## Query 3 — superseding candidate-set decision

### Verbatim

> 又和学长明确了一下，改成划分网格选点，不用一定包含ground truth，我们讨论一下如何划分网格

### Agreed discussion outcome

- Use an isotropic three-dimensional grid in the room's global coordinate frame: `dx = dy = dz = 0.5 m`.
- Anchor the lattice independently of the query and ground truth by snapping the mesh AABB inward to global `0.5 m` multiples.
- **Superseded 2026-08-20 by Yixun after measured B7:** retain ray-parity-valid room-air candidates at least `0.20 m` from room surfaces and at least `0.5 m` from the known receiver. Surface clearance is separated from physical validity.
- Do **not** insert the continuous ground truth into the grid.
- Measure error to the continuous metadata ground truth and report the per-query grid-oracle floor `min_c ||c - x*||`.
- Convert each global candidate to FLAC's receiver-relative coordinate only at conditioning time: `q = c - x_r`.

This supersedes the earlier metadata-source-bank candidate decision and the attachment's statement that the ground truth must be included.

### Assumption / hypothesis

At `0.5 m` spacing, the ideal cubic quantization radius is `sqrt(3) * 0.5 / 2 = 0.433 m`, below the `0.5 m` success threshold, while remaining roughly eight times cheaper than a `0.25 m` three-dimensional grid.

## Query 4 — execute

### Verbatim

> 我同意，执行一下

### Summary

Proceed under `NeuriPs_Workshop`, using the experiment SOP. The isolated implementation branch/worktree is `localization-exp`, based on `origin/check-equivariance-necessity` at `ecb83523c4ae8c60d4cd5f0ae3e562f2a84f1fa9`.

## Query 5 — missing-mesh scope decision

### Verbatim

> 缺失mesh就先注明但不纳入测试

### Summary

Record `ListeningRoom_idx_2` as an upstream geometry-asset omission and exclude that room from this mesh-based preflight. Do not construct a depth fallback and do not substitute another mesh. The existing full split remains the source manifest, but the test runner filters exactly this one room in memory and records the exclusion explicitly.

### Consequence

`ListeningRoom_idx_2` accounts for 1,000 of the 6,337 unseen queries. The exp_09 test denominator is therefore exactly **5,337 queries in 16 rooms**. Results must be labeled “mesh-available preflight subset,” not the complete published unseen-room protocol; no other room/query may be removed.
