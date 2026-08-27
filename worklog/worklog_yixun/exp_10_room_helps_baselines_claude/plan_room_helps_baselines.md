# Review specification — material-blind Room Helps baselines for FLAC localization

**Experiment:** `exp_10_room_helps_baselines`  
**Author:** OpenAI Codex (Planner / Analyst)  
**Date:** 2026-08-23  
**Status:** CORE IMPLEMENTED — user authorized implementation on 2026-08-23; training, mesh repair/tetrahedralization, and formal runs have not started  
**Base commit:** `f6388cf1c0813061ae9afd522a4867e5bdd5c19b`

## 0. Audit verdict at a glance

The two proposed baselines are frozen and their tested core implementation is now present:

| Item | Frozen decision |
|---|---|
| Learned baseline | `Few-ShotRIR-Waveform`, direct mono waveform prediction |
| Physics baseline | `FEM-Sabine`, uniform Sabine-equivalent boundary, 80–300 Hz |
| Material/RGB access | Neither baseline receives RGB or explicit material assignments |
| Geometry/context/coordinates | Same data and coordinate conventions as FLAC |
| Training objective | RIR reconstruction only; no localization loss |
| Initialization | Few-ShotRIR-Waveform is trained from scratch |
| Context counts | Nested `K_ctx ∈ {1, 8}` |
| Selection rule | Few-ShotRIR: frozen AGREE cosine; FEM: Room Helps pulse-source stacked complex OMP |
| Candidate set and metrics | Byte-identical to the existing FLAC localization protocol |

Two claims are deliberately **not** made:

1. This is not a strict reproduction of the original Few-ShotRIR architecture, because RGB is removed, AcousticRooms is monaural, the coordinate/context conventions change, and the decoder predicts a waveform rather than only log magnitude.
2. `FEM-Sabine + Room-Helps-OMP` is not a strict reproduction of the paper’s physical model: it replaces sound-hard/material-known boundaries with one context-estimated Sabine impedance and evaluates the frozen exp_09 candidate grid rather than FEM mesh nodes. Its selection rule does retain the paper’s pulse-source multifrequency sparse recovery.

Those deviations are necessary consequences of the user-approved material-blind AR adaptation. They must appear in any paper or result table.

## 1. Scientific question

Given an observed AcousticRooms RIR at a fixed receiver and a physically valid 3-D candidate source grid, how accurately can we localize the source when the candidate RIR is supplied by:

1. a deterministic few-shot learned forward model without RGB or material labels; or
2. a low-frequency FEM forward model whose unknown boundary materials are compressed into one Sabine-equivalent absorption coefficient?

Few-ShotRIR remains analysis-by-synthesis through AGREE:

\[
\hat{\mathbf x}
=\arg\max_{\mathbf x\in\mathcal C_q}
\operatorname{cos}
\left(E_{\mathrm{AGREE}}(h_{\mathrm{obs}}),
E_{\mathrm{AGREE}}(\hat h(\mathbf x))\right),
\]

where `C_q` is the frozen candidate set for query `q`, `h_obs` is the observed target RIR, and every method changes only the predicted candidate RIR `h_hat(x)`.

FEM uses the Room Helps pulse-source special case instead. Let `y` be the observed complex RIR spectrum at the exact 80–300 Hz DFT bins and let column `d_j` contain the FEM response at those bins for candidate `j`. Since an RIR is the response to a unit impulse, the source coefficient is shared across frequency. For the one-source AR task, the first OMP step is

\[
\hat j = \arg\max_j
\frac{|d_j^H y|^2}{\|d_j\|_2^2\,\|y\|_2^2}.
\]

Both arms retain the same candidate identities, stable tie breaking, and localization metrics, but they intentionally do not share the selection score.

## 2. Scope and non-goals

### 2.1 In scope

- AcousticRooms monaural RIR localization at 22,050 Hz.
- The existing mesh-valid 0.5 m three-dimensional candidate lattice.
- `K_ctx = 1` and `K_ctx = 8` acoustic observations.
- A direct-waveform Few-ShotRIR adaptation trained once from scratch with variable context count.
- A material-blind FEM baseline using room geometry, context-estimated T60, and a spatially uniform Sabine boundary.
- Frozen AGREE waveform scoring for Few-ShotRIR; Room Helps multifrequency sparse recovery for FEM; common localization/error reporting.

### 2.2 Out of scope

- RGB, semantic surface labels, explicit material assignments, or AcousticRooms simulation material files.
- A learned localization head or localization-aware loss.
- Fine-tuning either baseline on unseen evaluation rooms.
- Query-specific fitting of model, boundary, T60, or hyperparameters from the target RIR. The frozen one-step FEM sparse solve is the authorized selector, not calibration.
- Per-room material calibration, frequency-dependent absorption fitting, or impedance inversion.
- Multi-source localization. There is exactly one active source in each query.
- Claiming a faithful reproduction of either source paper.

## 3. Common data and localization protocol

### 3.1 Data identity

- Dataset: the processed `single_channel_ir_1` AcousticRooms release.
- Audio: mono PCM, 22,050 Hz.
- Target waveform: beginning-preserving pad/crop to 10,240 samples.
- Context waveform: beginning-preserving pad/crop to 9,600 samples.
- No direct-path onset realignment. The propagation delay relative to sample zero is retained.
- Target and context identities come from the existing frozen localization manifest; model arms never redraw contexts.

The official AR files are described as normalized and trimmed, but their start is not normalized to the direct arrival. Local inspection confirms that the geometric propagation delay is retained. Therefore any implementation that shifts each direct arrival to sample zero is a protocol violation and removes a useful localization cue.

### 3.2 Context-count convention

`K` is ambiguous in the current project, so this experiment uses two explicit names:

- `K_ctx ∈ {1, 8}`: number of acoustic context RIRs.
- `K_gen`: number of stochastic RIR generations per candidate.

For every query, `K_ctx=1` uses the first member of the exact same ordered eight-context manifest used by `K_ctx=8`. It is not independently sampled. Both new baselines are deterministic, so `K_gen=1`. The primary FLAC comparator also uses `K_gen=1`; larger FLAC `K_gen` values remain FLAC-only inference ablations and do not replace the paired primary comparison.

### 3.3 Coordinates and geometry

- Receiver and source metadata remain in AcousticRooms global coordinates for candidate construction and error measurement.
- At each model boundary, source and context coordinates are transformed exactly once using the existing FLAC receiver-relative convention.
- Geometry input is the same FLAC depth-derived geometric representation and preprocessing. No RGB channels are added.
- Candidate grids, receiver/context exclusion radii, z-band rule, ordering, hashes, and oracle errors are inherited unchanged from `exp_09_localization_grid_preflight`.
- The truth is continuous and is never inserted into or snapped onto the grid.

### 3.4 Current room coverage

The official AcousticRooms geometry release does not contain `ListeningRoom_idx_2.obj`, although that room contributes 1,000 queries to the 6,337-query unseen split. Until an authoritative geometry is supplied and audited:

- the paired FEM comparison covers exactly the already approved mesh-available subset: **5,337 queries in 16 rooms**;
- no AABB, convex hull, nearest-room mesh, or depth-derived substitute may be used for the missing room;
- Few-ShotRIR-Waveform may be run separately on all 6,337 queries as a supplementary result, but the paired cross-method headline table must use identical 16-room query coverage;
- results must be labeled “mesh-available AcousticRooms unseen subset,” not “full unseen split.”

## 4. Baseline A — Few-ShotRIR-Waveform

### 4.1 What is retained from Few-ShotRIR

- A set of sparse environment observations is encoded into an implicit acoustic memory.
- A transformer-style context encoder aggregates the observation set.
- A query-conditioned decoder attends to that memory to predict the RIR at an arbitrary source/receiver pair.
- The model is trained end-to-end using only RIR reconstruction targets.
- An energy-decay matching term remains part of the training objective.

### 4.2 AR adaptation

The original Few-ShotRIR uses RGB-D, binaural echoes, egocentric context poses, and a log-magnitude decoder. The frozen AR adaptation is:

1. **Delete the RGB branch.** There is no zero-filled RGB tensor and no learned placeholder token.
2. **Geometry input:** use the same depth-derived geometry tensor and normalization as FLAC, encoded by a baseline-owned trainable encoder.
3. **Acoustic context:** use the same ordered FLAC context waveforms and the same 9,600-sample representation.
4. **Coordinates:** use the same FLAC receiver-relative source, context-source, and receiver conventions.
5. **Output:** predict one mono waveform `h_hat ∈ R^10240` directly.
6. **Initialization:** all trainable baseline weights are randomly initialized. No Few-ShotRIR, FLAC, VAE, or AGREE weights initialize the predictor. AGREE remains frozen and evaluation-only.
7. **One model:** train a single variable-context model; during training draw `K_ctx` from 1 through 8 and mask unused context slots. Evaluate the same checkpoint at `K_ctx=1` and `K_ctx=8`.

“Same input as FLAC” means byte-identical data and coordinate preprocessing, not copying FLAC’s learned conditioner weights.

### 4.3 Waveform decoder and timing contract

- Output shape: `[B, 1, 10240]`, dtype `float32`.
- Sample rate: 22,050 Hz.
- Output time origin: identical to the target file’s sample zero.
- No Griffin–Lim, predicted phase angle, target phase, or post-hoc onset alignment.
- Final waveform enters the existing common clamp/preparation path before AGREE scoring.

The decoder architecture may use 1-D upsampling/residual blocks, but it must output the samples directly rather than producing a magnitude spectrogram and reconstructing phase afterward.

### 4.4 Training loss

There is no localization loss. The frozen loss family is:

\[
\mathcal L_{\mathrm{FSW}}
=\lambda_t\mathcal L_{\mathrm{wave}}
+\lambda_s\mathcal L_{\mathrm{MRSTFT}}
+\lambda_d\mathcal L_{\mathrm{EDC}}.
\]

- `L_wave`: sample-aligned waveform reconstruction, robust L1/Charbonnier rather than pure L2.
- `L_MRSTFT`: multi-resolution spectral convergence plus log-magnitude error, computed from the predicted and target waveforms.
- `L_EDC`: differentiable Schroeder backward-energy-decay matching, masking target all-zero padded tails.

Loss weights, STFT windows, optimizer, schedule, model width, and training budget are implementation hyperparameters. They must be registered before training and selected using training/validation reconstruction metrics only. Unseen localization accuracy and AGREE localization scores cannot tune them.

### 4.5 Alignment and augmentation rule

The baseline must not independently shift target arrival times. The FLAC training loader’s optional random delay augmentation is not automatically inherited, because a delay applied only to the supervised target would contradict source/receiver geometry. Any time-shift augmentation requires a separate approved rule that transforms all timing-dependent signals coherently; the initial implementation leaves it disabled.

### 4.6 Expected strengths and limitations

Strengths:

- exact waveform interface to AGREE;
- learns frequency-dependent and spatial effects from context without explicit materials;
- preserves direct-path delay as a localization cue.

Limitations:

- exact high-frequency phase may be underdetermined from sparse material-blind context;
- sample-level loss can overemphasize direct sound and strong early peaks;
- direct waveform prediction is a larger departure from the published Few-ShotRIR output head;
- deterministic prediction cannot represent FLAC’s conditional RIR distribution.

## 5. Baseline B — FEM-Sabine

### 5.1 Interpretation

For each query, FEM constructs a complex room-transfer dictionary over the same candidate source locations and exact 80–300 Hz bins. Because AR provides one-source RIRs, localization is a one-support complex sparse-recovery problem.

Room Helps §3.2 allows frequency-dependent coefficients with a common spatial support and scores OMP atoms by aggregating goodness across frequencies. With only one microphone, that general MMV formulation is degenerate: every nonzero scalar dictionary column spans the scalar observation at its own frequency. AR RIRs are responses to a known unit impulse, so the implementation follows the paper’s §3.3 pulse-source case: vertically stack all complex frequency equations, enforce one frequency-independent sparse source vector, and apply complex OMP. The physical boundary is still the AR-adapted Sabine approximation, so the method remains named `FEM-Sabine`, not a strict Room Helps reproduction.

### 5.2 Acoustic air domain

The FEM domain is the connected acoustic air volume containing the receiver.

- `V`: volume of that connected tetrahedral air domain.
- `S`: total physical air-contact boundary area: walls, floor, ceiling, and closed obstacle/furniture surfaces represented in the repaired acoustic boundary.
- Mesh-repair seams and artificial internal faces are not counted in `S`.
- Disconnected sealed voids not connected to the receiver’s air domain are excluded from both `V` and `S`.

The surface mesh must be audited and repaired deterministically before tetrahedralization. Repair may close topological defects but may not use target RIRs or target locations.

### 5.3 T60 estimate and Sabine boundary

Use the exact FLAC acoustic-context waveforms and the same AR RT60 call used by the project: `pyroomacoustics.experimental.measure_rt60(..., decay_db=20)` at 22,050 Hz. (`decay_db=30` applies to the HAA branch; the earlier draft incorrectly generalized it to AR.)

- `K_ctx=1`: use the one valid context estimate.
- `K_ctx=8`: compute each context estimate independently and take the median of valid estimates.
- Invalid estimates are reported. If no context yields a valid estimate, the query fails closed rather than reading the target RIR or a room label.

The Sabine-equivalent uniform absorption is

\[
A=\frac{0.161V}{T_{60}}, \qquad
\alpha_{\mathrm{Sabine}}=\frac{A}{S}
=\frac{0.161V}{S T_{60}}.
\]

`alpha_Sabine` is spatially uniform and frequency-independent. Numerically invalid or out-of-range raw values are logged; the solver uses a preregistered physical clip only for stability, and the clip rate is reported. No surface-specific or material-category parameters are fitted.

For a locally reacting boundary, use a deterministic normal-incidence mapping:

\[
|R|=\sqrt{1-\alpha},\qquad
z_n=\frac{1+|R|}{1-|R|},
\]

followed by the corresponding Robin boundary term. The reflection phase is fixed to zero; it is not learned from evaluation data.

### 5.4 Volume mesh and solver

- Tetrahedral elements: first-order `P1`.
- Maximum edge length: `h_max = 0.22 m`, corresponding to at least approximately five elements per wavelength at 300 Hz for `c=343 m/s`.
- Air constants: one frozen project-wide `c` and density `rho`; no room-dependent air-property fitting.
- Frequency-domain equation:

\[
\left(K-k^2M+B_\alpha(\omega)\right)p_\omega=q_\omega.
\]

- Frequency range: 80–300 Hz inclusive.
- Frequency grid: positive DFT bins of a 10,240-sample, 22,050 Hz waveform,

\[
\Delta f=\frac{22050}{10240}\approx2.1533\text{ Hz}.
\]

- Solve at every DFT bin inside the band; do not interpolate from a sparse hand-selected frequency set.
- Use acoustic reciprocity: solve from the fixed receiver at each frequency and evaluate the field at all candidate source locations.
- Candidate/source and receiver loads use a documented, mesh-consistent interpolation rule; nearest-node snapping is prohibited unless its error bound passes a preregistered gate.

### 5.5 Pulse-source stacked sparse recovery

For each `K_ctx` setting:

1. compute the target RIR’s complex DFT at the exact FEM bins without onset shifting;
2. form `D ∈ C^(F×N)` by transposing the FEM candidate responses, where `F` is the frequency count and `N` the frozen candidate count;
3. run complex OMP on `y = Df + e` with known `source_count=1`;
4. score every first-step atom by its normalized projection energy `|d_j^H y|²/(||d_j||² ||y||²)`;
5. choose the stable first maximum in frozen candidate order and report the complex least-squares coefficient and relative residual.

No FEM unit-gain calibration is needed for selection: multiplying every dictionary atom by the same nonzero scalar cancels in the normalized projection and is absorbed by the OMP least-squares coefficient. Optional IFFT waveform construction remains only a forward-model diagnostic, not the localization criterion.

The free-field and simple-box validation must still show that the complex phase convention corresponds to a direct arrival near `distance/c`. No candidate-specific phase correction, circular shift, magnitude-only comparison, or AGREE embedding is allowed in the FEM selector.

### 5.6 Expected strengths and limitations

Strengths:

- uses explicit room geometry and low-frequency wave propagation;
- needs no RGB, semantic label, or explicit material assignment;
- uniform Sabine estimation is transparent and reproducible;
- reciprocity amortizes each frequency solve across all candidates.

Limitations:

- one scalar T60 cannot reproduce spatially or frequency-varying boundary absorption;
- mesh repair can change `V`, `S`, and modal structure;
- 80–300 Hz discards most of the measured RIR bandwidth;
- the stacked pulse model assumes the normalized AR RIR behaves like a common unit-impulse source spectrum across selected frequencies;
- low-frequency FEM cost and memory can be high for large rooms;
- boundary/model mismatch can decorrelate complex phase more severely than magnitude-only methods.

The band limitation and pulse-spectrum assumption are intentional baseline constraints, not implementation bugs.

## 6. Selection rules and common metrics

Few-ShotRIR must call the existing FLAC path, not reimplement an approximate scorer:

1. waveform shape `[B, 1, 10240]`, float32, 22,050 Hz;
2. common clamp to `[-1,1]` where the existing path applies it;
3. frozen `AGREE.encode_audio(..., normalize=True)`;
4. cosine similarity between observed and candidate embeddings;
5. deterministic stable argmax in the frozen lexicographic candidate order.

The Few-ShotRIR observed embedding is computed once per query. No AGREE gradient enters its forward model, and it is not trained to maximize AGREE or localization performance.

FEM must not load AGREE. It consumes the same observed RIR, but only its exact 80–300 Hz complex bins, and applies the Room Helps stacked pulse-source OMP specified in §5.5. Both arms use the identical exp_09 candidate array, stable first-maximum rule, continuous truth, oracle calculation, and error metrics.

## 7. Reporting protocol

### 7.1 Primary localization readouts

For every method and `K_ctx`:

- mean and median Euclidean localization error `e_loc`;
- success at 0.5 m and 1.0 m;
- mean and median grid-oracle error `e_oracle`;
- excess error `max(0, e_loc - e_oracle)`;
- oracle-normalized success at 0.5 m and 1.0 m;
- room-macro aggregation and 95% room-bootstrap confidence intervals;
- runtime per query, per candidate, and per generated waveform/FEM dictionary atom.

### 7.2 Required controls and diagnostics

- deterministic random-candidate baseline on the identical candidate manifest;
- Vanilla FLAC and FA-FLAC on the same query/context/candidate identities;
- identical `K_ctx=1` nesting for every arm;
- Few-ShotRIR-Waveform reconstruction diagnostics: waveform, MR-STFT, EDC, T60/EDT/C50;
- FEM diagnostics: mesh sizes/quality, raw and clipped Sabine alpha, valid T60 count, solver residual, direct-arrival timing, OMP support/coefficient, normalized projection-score distribution, and sparse residual;
- AGREE embedding-norm/cosine distributions for real, FLAC, and Few-ShotRIR-Waveform only;
- explicit coverage table showing the `ListeningRoom_idx_2` exclusion.

### 7.3 Interpretation rule

A lower FEM result cannot by itself establish that room physics is unhelpful, because it conflates the Sabine boundary approximation, mesh repair, 80–300 Hz truncation, complex phase mismatch, and the pulse-spectrum assumption. Conclusions must be phrased as performance of the specified `FEM-Sabine + Room-Helps-OMP` baseline.

Likewise, Few-ShotRIR-Waveform measures the AR-adapted few-shot architecture, not the original paper’s RGB-D/binaural/magnitude model.

## 8. Leakage and fairness rules

- Unseen target RIRs are localization observations only: AGREE input for Few-ShotRIR and the right-hand-side complex spectrum for FEM sparse recovery. They cannot estimate T60, boundaries, losses, or hyperparameters.
- Context RIRs may condition Few-ShotRIR and estimate FEM T60, exactly as specified.
- No explicit AcousticRooms material library, simulation boundary assignment, or semantic surface mapping is loaded.
- No RGB is loaded by either new baseline.
- No method receives the target location except after prediction for metric computation.
- Candidate and context manifests are content-hashed and shared across arms.
- Model selection uses training/validation reconstruction criteria, never unseen localization score.

## 9. Implementation and TDD rounds

The user authorized implementation after reviewing the method choices. R1–R5 core code is now implemented test-first under `src/tests/`. No model training, production mesh repair/tetrahedralization, full FEM solve, or formal localization run was launched. Independent review and compute gates remain pre-launch requirements.

### R0 — data/parity audit

Completed findings and remaining audit outputs:

- [x] Reuse the existing frozen localization loader for byte-identical target/context waveform paths.
- [x] Audit AR trimming/padding and confirm that direct-path delay remains relative to file sample zero.
- [x] Pin nested `K_ctx=1/8` manifests and exact-prefix behavior.
- [x] Reuse the exp_09 audit showing all 16 authoritative surface meshes are non-watertight and non-edge-manifold, hence not directly tetrahedralizable.
- [ ] Produce a bounded element-count/memory/runtime estimate at `h_max=0.22 m` before a production mesh or full solver run.

No full training or FEM sweep may launch at R0.

### R1 — shared deterministic baseline interface

Implemented files:

- `src/baselines/protocol.py`
- `src/tests/test_baseline_protocol.py`

Test-first contracts:

- method output must be float32 `[B,1,10240]`;
- sample rate and finite-value checks fail closed;
- K=1 is an exact prefix of K=8;
- candidate ordering/hash is unchanged;
- Few-ShotRIR AGREE preprocessing is bit-identical to the current FLAC scorer;
- FEM cannot reach AGREE and instead receives exact complex DFT observations for Room Helps recovery.

### R2 — Few-ShotRIR-Waveform

Implemented files:

- `src/baselines/few_shot_rir_waveform.py`
- `src/training/few_shot_rir_waveform.py`
- `src/configs/model_configs/baselines/FewShotRIR_Waveform_AR.json`
- `src/configs/dataset_configs/AR/train/acousticroom_train_few_shot_waveform.json`
- `src/tests/test_few_shot_rir_waveform.py`
- `src/tests/test_few_shot_training.py`

Test-first contracts:

- RGB is absent from the accepted batch schema;
- K masking and permutation behavior are correct;
- receiver-relative coordinate conversion matches FLAC;
- direct waveform output shape/dtype/range;
- loss terms are finite, differentiable, and zero on identical signals;
- EDC ignores padded all-zero tails;
- no AGREE/localization loss or frozen FLAC parameter appears in the training graph;
- one optimizer step updates baseline weights from random initialization.

### R3 — T60/Sabine/mesh primitives

Implemented files:

- `src/baselines/fem_sabine.py`
- `src/tests/test_fem_sabine.py`

Test-first contracts:

- RT60 wrapper numerically matches the project’s existing FLAC metric call;
- K=8 valid-estimate median and all-invalid fail-closed behavior;
- analytic-box `V` and `S` recovery;
- exclusion of artificial/internal repair faces;
- Sabine alpha and impedance mapping against hand-computed examples;
- physical clipping is logged and deterministic;
- no material file or semantic label is accepted by the API.

### R4 — FEM assembly, solve, and waveform adapter

Implemented files:

- `src/baselines/fem_solver.py`
- `src/baselines/fem_pipeline.py`
- `src/baselines/room_helps_sparse.py`
- `src/tests/test_fem_solver.py`
- `src/tests/test_fem_pipeline.py`
- `src/tests/test_room_helps_sparse.py`

Test-first contracts:

- analytic 1-D/simple-box modal sanity checks;
- sparse matrix dimensions, symmetry, and boundary-term sign;
- direct solve residual threshold;
- reciprocity agrees with source-by-source solves;
- exact 80–300 Hz DFT-bin selection;
- conjugate symmetry produces a real 10,240-sample waveform;
- free-field direct-arrival timing agrees with `distance/c` within a preregistered tolerance;
- no candidate-specific gain or circular shift;
- stacked complex pulse-source OMP recovers common support and uses stable candidate-order ties;
- deterministic output across batching and cache use.

### R5 — integration and reporting

Implemented files:

- `src/localization/baseline_runner.py` and `src/localization/baseline_experiment.py`, without changing the frozen FLAC execution path;
- `localize_baseline.py` method dispatch;
- `src/tests/test_baseline_runner.py` and `src/tests/test_baseline_experiment.py`;
- implementation status and commands under this exp_10 directory. Result, analysis, and HTML artifacts await authorized real runs.

Integration tests must prove identical query/context/candidate identities, the frozen method-specific selection rule, and resume-safe artifact generation.

## 10. Pre-launch gates

Core implementation was authorized by the user. Expensive training and full runs additionally require:

1. all permanent tests green;
2. independent review of every code round and one integrative review;
3. small synthetic forward/solve passes;
4. real-data readback passes;
5. Few-ShotRIR one-step and overfit-one-batch checks pass;
6. one small-room FEM result passes residual, reciprocity, timing, and memory gates;
7. bounded full-split compute/storage projection is presented to Yixun;
8. the exact room/query coverage and missing-mesh limitation are accepted;
9. all run parameters and acceptance thresholds are written before launch.

## 11. Reviewer sign-off checklist

The reviewer should explicitly accept, reject, or amend each item:

- [ ] Direct waveform output is an acceptable Few-ShotRIR adaptation.
- [ ] Deleting RGB while keeping FLAC geometry/context/coordinates is scientifically interpretable.
- [ ] Training one variable-context model and evaluating nested `K_ctx={1,8}` is acceptable.
- [ ] Reconstruction-only Few-ShotRIR loss and frozen evaluation-only AGREE prevent localization leakage.
- [ ] The Sabine `V`, `S`, T60 aggregation, and uniform impedance definitions are acceptable.
- [ ] The AR-RIR pulse-source assumption and complex stacked OMP are an acceptable adaptation of Room Helps §3.3.
- [ ] The 80–300 Hz band and complex phase/model-mismatch limitations are labeled strongly enough.
- [ ] FEM correctly bypasses AGREE while retaining identical candidates and localization metrics.
- [ ] The 16-room/5,337-query restriction is acceptable until authoritative missing geometry exists.
- [ ] The planned TDD, parity, leakage, and compute gates are sufficient before expensive runs.

## 12. Primary references

1. S. Majumder et al., “Few-Shot Audio-Visual Learning of Environment Acoustics,” NeurIPS 2022: <https://proceedings.neurips.cc/paper_files/paper/2022/file/113ae3a9762ca2168f860a8501d6ae25-Paper-Conference.pdf>
2. I. Dokmanić and M. Vetterli, “Room Helps: Acoustic Localization with Finite Elements,” ICASSP 2012: <https://dokmanic.ece.illinois.edu/assets/pdf/Dokmanic2012ii.pdf>
3. AcousticRooms official data and geometry release: <https://github.com/facebookresearch/AcousticRooms>
4. Existing project localization protocol: `worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/plan_localization_grid_preflight.md`
5. Existing waveform scorer implementation: `src/localization/engine.py` and `src/metrics/modules/Retrieval.py`

## 13. Decision record

The original requested review outcomes were:

- **APPROVED:** this document becomes the implementation contract;
- **APPROVED WITH AMENDMENTS:** list exact section edits, then revise before coding;
- **NOT APPROVED:** identify the method-level issue that must be reopened.

Yixun subsequently instructed “先把代码落实一下,” which is recorded as authorization to implement the agreed source code and tests. It does not authorize an expensive training run, destructive mesh repair, dependency installation, or formal evaluation. Those remain gated by Section 10.
