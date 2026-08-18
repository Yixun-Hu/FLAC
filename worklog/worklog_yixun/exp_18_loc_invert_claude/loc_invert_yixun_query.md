# exp_18 loc_invert — Yixun's driving query

## Query 1 (2026-08-18, session `localization-exp`, verbatim)

> Currently we are going to use the FLAC and FA B-F method, FLAC-data-augmentation method, and potentially, cylindrical-dinov3vit FLAC to do the RIR-based localization experiment. Given an observed RIR at a known receiver pose, we score candidate source locations by the similarity between the observation and RIRs sampled from the acoustic world model. Currently I already have the checkpoints for those model, but I haven't rsync those checkpoints here. But before that, please prepare the preflight experiment now. Please checkout from the existing check-equivariance-necessity branch to a new branch:`localization-exp`. And following @worklog/experiment_SOP.md (and you can modify the corresponding working directory inside this SOP) to conduct this preflight localization experiment 1:

> ### Experiment 1: Inverting Vanilla FLAC for Source Localization
>
> **Objective.** We first test whether a pretrained Vanilla FLAC model contains sufficient source-position information to support localization without training an additional localization network. Given a held-out RIR observed at a known receiver position, we use FLAC as a forward acoustic simulator and recover the unknown source location through analysis-by-synthesis inference.
>
> **Experimental setup.** We use the unseen-room split of AcousticRooms. For each test instance, we retain the standard few-shot room context O used by FLAC and select a held-out target RIR h_obs = h(x_s*, x_r), where x_r is the known receiver position and x_s* is hidden from the localization algorithm. Importantly, h_obs is excluded from the few-shot context to prevent target leakage.
>
> **Three-dimensional candidate locations.** AcousticRooms samples source locations throughout the valid 3-D free space of each room. We therefore do not constrain candidate sources to a common height. For each localization query, we construct a candidate set C = {x_s^(1), …, x_s^(M)}, x_s^(m) ∈ R^3, using the valid source coordinates provided in the room metadata. The ground-truth source is included in the candidate set, while its observed RIR, and optionally all RIRs sharing the same source, are excluded from the FLAC context to prevent leakage.
>
> Given an observed RIR h_obs, a known receiver pose x_r, and room context c, we query the pretrained FLAC model at every candidate location. For a stochastic model, we generate K samples per candidate and define S(x_s) = max_k sim(E_a(h_obs), E_a(ĥ_k(x_s, x_r, c))). The predicted source location is x̂_s = argmax_{x_s ∈ C} S(x_s). This evaluates whether a forward acoustic model can be inverted through candidate-based inference without training an additional localization network. All candidates share the same receiver pose and room context.
>
> **Analysis-by-synthesis inference.** For every candidate x_s^(m), Vanilla FLAC generates K possible RIRs: ĥ_{m,k} ~ p_θ(h | x_s^(m), x_r, O), k = 1…K. We compare the generated RIRs with h_obs using the frozen acoustic branch of AGREE: s_{m,k} = cos(E_a(h_obs), E_a(ĥ_{m,k})). Because FLAC is stochastic, we aggregate the K samples using a log-mean-exp score: S(x_s^(m)) = τ log[(1/K) Σ_k exp(s_{m,k}/τ)]. The estimated source position is x̂_s = argmax_m S(x_s^(m)). We freeze FLAC and AGREE throughout the experiment; localization is performed entirely at inference time.
>
> **Localization heatmap.** For visualization, the candidate scores are converted into a normalized spatial score map: P(x_s^(m) | h_obs) = exp(S(x_s^(m))/T) / Σ_x exp(S(x)/T). This normalization is used only for visualization and is not interpreted as a calibrated posterior probability. Each heatmap shows: the room boundary and valid candidate region; the known receiver position; the ground-truth source x_s*; the predicted source x̂_s; the normalized FLAC similarity score over candidate locations. We will show representative sharp-success, ambiguous, and failure cases.
>
> **Evaluation.** The primary metric is Euclidean localization error e_loc = ||x̂_s − x_s*||₂. Across the unseen-room test set, we report: median localization error in meters; mean localization error; success within 0.5 m; success within 1.0 m; random-candidate performance as a lower baseline. As a diagnostic upper bound, we can also replace FLAC-generated RIRs with ground-truth candidate RIRs when available. This separates limitations of the AGREE matching space from errors introduced by the forward generator.
>
> **Research question.** The experiment answers: Can a generative forward acoustic model be inverted, without localization training, to recover the source position of a held-out RIR in an unseen room? A successful result would be a heatmap whose dominant mode lies near x_s*, together with localization error substantially below the random-candidate baseline. A diffuse or systematically displaced heatmap would indicate that Vanilla FLAC's generated RIRs match global room acoustics but do not preserve sufficiently discriminative source-specific spatial cues.

## Summary

Build and run an analysis-by-synthesis source-localization evaluation that inverts a frozen, pretrained Vanilla FLAC: for each held-out unseen-room RIR (known receiver, hidden source), generate K RIRs at each valid candidate source position from the room metadata (same receiver, same few-shot context, target excluded from context), score each candidate by AGREE audio-branch cosine similarity to the observation aggregated with log-mean-exp, and predict the argmax candidate. Report median/mean localization error, success@0.5 m/1.0 m against a random-candidate lower baseline and a GT-RIR oracle upper bound, plus per-query heatmaps. exp_18 is the preflight for a later cross-arm comparison (Vanilla vs FA B-F vs yaw-aug vs cylindrical-DINOv3 FLAC); those arms' checkpoints will be rsynced later.

## Assumption / hypothesis

Yixun's working hypothesis: FLAC's generated RIRs preserve source-position-specific spatial cues (not just global room acoustics), so candidate-based inversion should localize substantially better than the random-candidate baseline in unseen rooms — making FLAC usable as an acoustic world model for downstream localization, and setting up the question of whether the equivariant/augmented arms invert better.

## Why the experiment needs to run

It is the preflight gate for the localization workstream: it establishes the protocol (candidate construction, leakage exclusions, scorer, aggregation, metrics, baselines), the code, and the Vanilla FLAC reference row that the FA B-F / yaw-aug / cyl-DINOv3 arms will be compared against once their checkpoints arrive.
