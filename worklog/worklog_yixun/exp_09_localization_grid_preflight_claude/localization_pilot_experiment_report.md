# Experimental Protocol Report: Mesh-Grid RIR-Based Source Localization

**Experiment:** 64-query pilot on AcousticRooms unseen rooms  
**Model arms:** Vanilla FLAC and FA-BF FLAC  
**Status:** Pre-registered protocol; no localization-quality result has been read  
**Date:** 2026-08-21

## 1. Objective

This experiment tests whether a pretrained acoustic world model can be inverted for three-dimensional source localization without training a separate localization network. Given an observed room impulse response (RIR) at a known receiver position, the model generates candidate RIRs at a mesh-derived grid of possible source positions. A frozen acoustic embedding model then compares the generated RIRs with the observation, and the highest-scoring candidate is returned as the source estimate.

The pilot compares two frozen 40k checkpoints:

1. Vanilla FLAC;
2. FA-BF FLAC, using the released frame-averaged conditioning path.

The experiment is inference-only. There is no localization training stage and no localization loss function.

## 2. Dataset and pilot sampling

The source pool is the AcousticRooms unseen-room evaluation split, which contains 6,337 query RIRs from 17 rooms. `ListeningRoom_idx_2` is excluded because its official 3-D geometry mesh is unavailable. No substitute geometry is used. The resulting mesh-available pool contains 5,337 query RIRs from 16 rooms.

For this pilot, four target query RIRs are sampled without replacement from each included room using query-selection seed 42 in a separate RNG stream. This produces

\[
N_{\mathrm{query}} = 16\ \mathrm{rooms}\times4\ \mathrm{queries/room}=64
\]

base localization tasks. The same 64 queries, context manifest, candidate grids, generation seeds, and scoring procedure are shared by both model arms.

A query is one held-out target RIR

\[
h_{\mathrm{obs}}=h(\mathbf{x}_s^\star,\mathbf{x}_r),
\]

where the receiver coordinate \(\mathbf{x}_r\) is known and the continuous source coordinate \(\mathbf{x}_s^\star\) is used only for evaluation. The target RIR and target source are excluded from the FLAC context.

## 3. Few-shot room context: \(N_{\mathrm{ctx}}\)

Each localization query uses the standard released FLAC AcousticRooms context width

\[
N_{\mathrm{ctx}}=8.
\]

The eight context entries are RIRs from other source positions recorded at the same receiver, together with their source coordinates. They are selected through the original FLAC global NumPy sampling path and then frozen in a content-hashed manifest. If fewer than eight eligible RIRs exist, the released loader samples with replacement, so the tensor always has width eight but may contain duplicate context entries. The observed target RIR is never included.

`N_ctx` is the number of observed few-shot context RIRs. It is independent of `K_gen`, the number of stochastic RIRs generated at each candidate location.

## 4. Three-dimensional candidate grid

Candidate positions are defined in the global AcousticRooms metadata coordinate system using the official 3-D mesh for each room. The grid does not depend on the target source coordinate.

### 4.1 Base lattice

An isotropic room-global lattice is constructed with spacing

\[
\Delta x=\Delta y=\Delta z=0.5\ \mathrm{m}.
\]

For each coordinate axis, lattice values are integer multiples of 0.5 m within the mesh axis-aligned bounding box:

\[
x_i\in
\left[
\left\lceil\frac{a_i^{\min}}{0.5}\right\rceil0.5,
\left\lfloor\frac{a_i^{\max}}{0.5}\right\rfloor0.5
\right].
\]

### 4.2 Physical-validity filters

A base lattice point is retained only when all applicable conditions hold:

- it is classified as room free space by strict-majority odd ray parity over 31 fixed, non-axis-aligned ray directions;
- its distance to the nearest mesh surface is at least 0.20 m, with a numerical tolerance of \(10^{-4}\) m;
- its distance to the known receiver is at least 0.50 m;
- its distance to every selected context-source coordinate is at least 0.25 m, preventing an effectively duplicated context location;
- its height remains valid under the room mesh and lies within the pre-approved context-derived band
  \[
  z\in[\min z_{\mathrm{ctx}}-0.5,\max z_{\mathrm{ctx}}+0.5]\ \mathrm{m}.
  \]

The geometry audit verified that this height-band rule introduces no additional query with grid-oracle error above 0.5 m relative to the full-height grid. Every included query has a nonempty candidate set.

The continuous ground-truth position is neither inserted into nor snapped onto the grid. Therefore, the target may lie between grid points.

## 5. Conditional RIR generation

For each query and candidate \(\mathbf{x}_s^{(m)}\), FLAC conditions on the candidate source position, the known receiver, the depth panorama, and the fixed eight-RIR room context:

\[
\widehat h_{m,k}
\sim p_\theta
\left(
h\mid\mathbf{x}_s^{(m)},\mathbf{x}_r,\mathcal O_{N_{\mathrm{ctx}}=8}
\right).
\]

Global candidate and context coordinates are converted to receiver-relative coordinates only at the model boundary. All candidates for one query share the same observation, receiver, context RIRs, context poses, and depth panorama; only the candidate source coordinate and stochastic generation noise vary.

The sampler follows the released FLAC evaluation path: one-step discrete-Euler rectified-flow sampling with `cfg_scale=1.0`. Both FLAC checkpoints and the downstream AGREE encoder remain frozen in evaluation mode.

## 6. Stochastic generation count: \(K_{\mathrm{gen}}\)

The experiment reports three generation settings:

\[
K_{\mathrm{gen}}\in\{1,4,8\}.
\]

`K_gen` is the number of stochastic RIR samples generated for each candidate; it is not the number of context RIRs. Samples are nested and counter-seeded:

- \(K_{\mathrm{gen}}=1\): sample 0;
- \(K_{\mathrm{gen}}=4\): samples 0--3;
- \(K_{\mathrm{gen}}=8\): samples 0--7.

Thus, eight RIRs are generated once per candidate and the smaller settings reuse prefixes of the same sequence. Reporting all three settings costs one \(K_{\mathrm{gen}}=8\) generation run rather than three independent runs.

The 64 base queries yield 192 query--`K_gen` readouts per model arm and 384 readouts across the two arms. These are repeated evaluations of 64 localization tasks, not 384 independently sampled target queries.

## 7. Candidate scoring and prediction

The observed and generated RIRs are encoded by the same frozen AGREE acoustic encoder \(E_a\). For candidate \(m\) and generation sample \(k\), the similarity is

\[
s_{m,k}=
\operatorname{cos}
\left(
E_a(h_{\mathrm{obs}}),E_a(\widehat h_{m,k})
\right).
\]

Following the score definition in `acoustic_localization_brief.pdf`, the stochastic samples are aggregated using log-mean-exp:

\[
S(\mathbf{x}_s^{(m)})
=
\tau\log
\left[
\frac{1}{K_{\mathrm{gen}}}
\sum_{k=1}^{K_{\mathrm{gen}}}
\exp\left(\frac{s_{m,k}}{\tau}\right)
\right],
\qquad \tau=0.1.
\]

The source estimate is

\[
\widehat{\mathbf{x}}_s
=
\underset{\mathbf{x}_s^{(m)}\in\mathcal C}{\arg\max}
\ S(\mathbf{x}_s^{(m)}).
\]

Scores are accumulated in float32 using a numerically stable `logsumexp`. Ties are resolved by the first candidate in the fixed lexicographic global-grid order. Any softmax temperature used for visualization does not affect the prediction.

## 8. Evaluation metrics

For every query, model arm, and `K_gen` setting, the following quantities are reported:

- raw localization error:
  \[
  e_{\mathrm{loc}}=\|\widehat{\mathbf{x}}_s-\mathbf{x}_s^\star\|_2;
  \]
- grid-oracle error:
  \[
  e_{\mathrm{oracle}}=
  \min_{\mathbf{c}\in\mathcal C}\|\mathbf{c}-\mathbf{x}_s^\star\|_2;
  \]
- excess error above grid resolution:
  \[
  e_{\mathrm{excess}}=
  \max(0,e_{\mathrm{loc}}-e_{\mathrm{oracle}});
  \]
- raw and oracle-normalized success rates at 0.5 m and 1.0 m;
- mean and median localization errors, summarized first within each room so rooms with more available RIRs do not dominate the result;
- generation throughput, peak GPU memory, and end-to-end runtime.

A deterministic room-matched random-candidate baseline is evaluated on the identical candidate sets. Score heatmaps show the valid grid, receiver, continuous target, predicted candidate, and normalized candidate scores. Because the pilot contains only four queries per room, its quality metrics are treated as diagnostic evidence rather than a replacement for the complete 5,337-query evaluation.

## 9. Leakage and fairness controls

The following controls are fixed before quality evaluation:

1. The target RIR and target source are absent from the few-shot context.
2. Ground truth is not used to construct, filter, insert, or snap candidate points.
3. Both model arms receive byte-identical contexts and candidate manifests.
4. Both arms use the same target-query subset and nested stochastic sample identities.
5. The observed and generated RIRs use the same mono, 22.05 kHz AGREE preprocessing path.
6. Checkpoints must load strictly with zero missing or unexpected parameters.
7. A room without its authoritative mesh is documented and excluded rather than approximated.

## 10. Expected pilot scale

The seed-42 pilot manifest contains exactly 46,301 query--candidate pairs per stochastic sample. Applying the measured real cached-engine rates to the formal runner gives approximately 0.75 GPU-hours for Vanilla and 0.84 GPU-hours for FA-BF at nested \(K_{\mathrm{gen}}=8\), including model startup. The serial two-arm estimate is approximately 1.59 GPU-hours; a 10% operational reserve gives 1.75 GPU-hours. The result is lower than a query-proportional estimate because equal-per-room sampling gives large and small rooms equal weight instead of inheriting the full split's query distribution.

## 11. Interpretation

The main question is whether candidate rankings produced by Vanilla FLAC contain enough source-dependent acoustic information to outperform the room-matched random baseline, and whether FA-BF changes localization accuracy under the same observations and candidate geometry. Improvements with larger `K_gen` would indicate that multiple stochastic forward samples provide a more reliable estimate of candidate compatibility. Failure at all three settings would suggest that generated RIRs preserve broad room acoustics but not sufficiently discriminative source-specific cues in the frozen AGREE embedding space.
