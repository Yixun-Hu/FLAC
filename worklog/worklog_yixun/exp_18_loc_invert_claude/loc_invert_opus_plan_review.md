# exp_18 loc_invert — PLAN review

**Reviewer:** Claude Opus 5 (max effort, declared fallback — OpenAI Codex unauthenticated on this box, 401), invoked via Claude Code Agent tool by the Fable 5 planner session · **Date:** 2026-08-18

**Reviewed artifact:** `plan_loc_invert.md` (DRAFT, 2026-08-18) · **Round:** PLAN only, no code exists.
**Context loaded before judging:** `worklog/experiment_SOP.md`; announcements 01–06; `loc_invert_yixun_query.md`; `loc_invert_worklog.md`; `exp_01_.../reproduce_flac_table1_results.md` + its run log + `_command.md`; repo-root `CLAUDE.md` (Metrics + eval-protocol warning); and the code the plan builds on — `eval_FLAC.py` (`evaluate_model`, `sample_context_ids`, `resolve_weights_source`, `resolve_cond_autocast`, `check_load_integrity`, `source_sha`, `orbit_provenance`), `src/configs/dataset_configs/custom_metadata/AR_md.py`, `src/models/conditioners.py`, `src/models/dit.py`, `src/models/diffusion.py`, `src/metrics/metric_callback.py`, `src/metrics/modules/Retrieval.py`, `AGREE/AGREE/model.py`, `AGREE/AGREE/audio_model.py`, `AGREE/AGREE/model_configs/dinoV3.json`, `src/data/dataset.py`, `src/inference/sampling.py`, the AR eval configs and `data/AR/*.json`.
**Out of scope for this round (not flagged as gaps):** FA B-F / yaw-aug / cylindrical arms, training/finetuning, cross-arm comparison tables, HAA. Dataset/checkpoint absence is taken as given.

---

## Verdict

**REQUEST-CHANGES.**

The engineering discipline of this plan is high — reuse of `eval_FLAC`'s loading/dispatch/provenance, the smoke quarantine, the pre-registered heatmap rule, the exact random baseline, the honest admission that the identity oracle is trivial. But four findings are of the kind that make the produced numbers uninterpretable rather than merely imprecise, and they are all in the *method*, not the code:

1. the K=8 context covers all-but-one of the non-GT candidates, so the task is largely solvable without acoustics and the plan's only lower baseline is ~5× too weak;
2. the scorer `E_a` is **stochastic** (AGREE's audio tower ends in a sampling VAE bottleneck), which the plan does not know — it invalidates the identity-oracle claim, perturbs every similarity, and couples the AGREE RNG to the diffusion noise stream;
3. there is no control proving the candidate coordinate changes the generated RIR at all, so "diffuse heatmap" and "plumbing bug" are indistinguishable outcomes;
4. the one experiment that can run *before* the FLAC checkpoints arrive — a measured-RIR power gate — is not in the run matrix, and the substitute the plan proposes (a second `single_channel_ir_*` channel) does not exist anywhere in this repo.

Plus a headline-aggregation convention that contradicts exp_01's own established finding. All fixes below are cheap and none require new eval configs.

---

## Findings

### BLOCKER 1 — The K=8 context covers all but one non-GT candidate: the task is largely solvable with no acoustics, and the registered lower baseline is ~5× too weak

`AR_md.get_ir_and_location_for_other_sources` (AR_md.py:90–106) draws `max_context = 8` references from the *other* source nodes **at the same receiver**: `remain_src_node = all_src_node − {target}`, then `np.random.choice(..., 8, replace=False)`. On the plan's own candidate-set assumption (§2.2 "Expected M ≈ 10"; `_worklog.md` item 8), and verified from the split — `data/AR/unseen_eval.json` has 17 rooms with **10 distinct sources in 16 of them and 9 in one** — this means:

- exactly **one** non-GT candidate is absent from the context (10-source rooms), and **zero** in the 9-source room (there, `len(valid)=8` so the draw takes all of them);
- the GT is absent by construction, always.

Consequences:

- **A "pick uniformly among the candidates absent from the context" heuristic scores ≈ 50 % top-1 — and 100 % in the 9-source room — using no acoustics at all.** The plan's only lower baseline is uniform-over-C at ~10 %. "Localization error substantially below the random-candidate baseline" (§1) is therefore close to unfalsifiable as a claim about FLAC.
- **The FLAC scores are themselves confounded.** The eight in-context candidates are precisely the ones whose true RIR at this receiver the model can copy; the two absent ones are the only ones it must extrapolate. That distinction is *perfectly correlated with the answer*, so any measured accuracy mixes "source-position modelling" with "which candidate is missing from my context".

**Fix (all announcement-01-compliant, all cheap):**

1. **Record the covariate.** Per query, log which candidate nodes are in the drawn context — match `context_poses` rows against the driver's own candidate camera coordinates (both already computed; see HIGH 6 for exactness). Report top-1 / e_loc split by candidate-in-context vs absent.
2. **Add the context-absence baseline** (uniform over the candidates absent from the context) as a *required* lower baseline next to uniform-over-C, and report the method's lift over *it*. It is exactly computable, no Monte Carlo, like the existing one.
3. **Sweep K_ctx with the existing configs.** Announcement 01 explicitly permits changing K only by switching between `acousticroom_*eval_{1,4,8}.json`. At K_ctx = 1, eight of nine non-GT candidates are absent and the absence baseline falls to ~1/9; the K_ctx = 1/4/8 trend is the direct measurement of how much of the result is context-copying. Pre-register which K_ctx is the headline (I recommend K_ctx = 1 for the headline claim, K_ctx = 8 reported alongside since it is the deployment protocol).
4. **Handle the 9-source room explicitly** — report it separately with the note that the absence heuristic is exact there; do not let it silently inflate a macro average.
5. If the readback shows the room *directory* holds more sources than the split's 10, M grows and the confound weakens proportionally — that is a measurement to report, not an assumption to make.

### BLOCKER 2 — `E_a` is a stochastic function; the plan treats it as a deterministic embedding

AGREE's audio branch is not a deterministic encoder. `AGREE.encode_audio` → `self.audio(...)` → `OobleckEncoder.forward` → `self.bottleneck.encode(x)` → `VAEBottleneck.encode` → `vae_sample`, which does

```python
latents = torch.randn_like(mean) * stdev + mean      # AGREE/AGREE/audio_model.py, vae_sample
```

with **no `self.training` guard** and no deterministic path. (`model_configs/dinoV3.json` sets `audio_cfg.model_name = "VAE"`, so this is the tower that is actually built.) Therefore:

- **§2.6's identity-oracle claim is false**: "for candidate = GT this IS h_obs (cos = 1)" — it is not 1, because the two embeddings of the same waveform are different draws. The whole "so the plain oracle is only a pipeline sanity check" paragraph rests on a property the code does not have.
- Every `s_{m,k}` carries a scorer-noise term that the plan's aggregation model (§2.4) does not represent, and which is *not* reduced by increasing K (it is per-embed, not per-generation).
- Each `encode_audio` call consumes the **global** CUDA RNG. `eval_FLAC` draws its diffusion noise from that same global stream (`noise = torch.randn(...)`, eval_FLAC.py:1277). Interleaving embeds with sampling — which the plan's per-query loop does — makes the generations depend on how many embeds happened before them, i.e. **not reproducible from `--seed` alone**.
- `loading_AGREE_model` (metric_callback.py:432–445) never calls `.eval()` or `.to(device)`; it returns a CPU module in train mode. The plan's `load_agree_audio` is described as a "thin wrapper" over it.

**Fix — register the embedding protocol explicitly in §2.4:**

- Draw the diffusion noise from a **dedicated `torch.Generator`** seeded per query, so the sampler stream is immune to AGREE's draws. `eval_FLAC` already uses exactly this pattern for the rotation draws (`rot_generator`, eval_FLAC.py:1242) and states the reason.
- Make `E_a` deterministic **driver-side, without touching release code**: seed the global RNG to a fixed constant immediately before each `encode_audio` call, so every waveform is embedded with the same ε (common random numbers ⇒ `sim(E_a(h), E_a(h)) = 1` exactly and the identity oracle recovers its intended meaning); or average R ≥ 8 draws. Pick one, register it, and unit-test that `E_a(h)` is reproducible across calls.
- `load_agree_audio` must `.to(device).eval()`; test asserts `model.training is False` and the parameter device.
- Declare that this differs from `Retrieval.compute_audio_features`'s (unseeded, stochastic) convention, and measure the **scorer-noise floor** at the smoke rung: the spread of `sim(E_a(h), E_a(h))` over repeated draws, reported next to the between-candidate score spread.

### BLOCKER 3 — No wiring/sensitivity control: nothing proves the candidate coordinate changes the generated RIR

`GeometryConditioner.forward` (conditioners.py:283–301) feeds the ViT `c = (coord[:, i, :, None, None] − depth_coord) / max_value`, so a candidate change *does* re-enter the network — but the plan never checks that the **scores** respond. Given exp_01's measured FLAC R@1 ≈ 7 % (6337-way retrieval, K=8), a score map that is flat in the candidate coordinate is a live scientific outcome, and it is indistinguishable from a plumbing bug (`candidate_metadata` writing the wrong key, `source_vit` shaped `[3]` instead of `[1,3]`, every candidate silently receiving the base `source`).

**Fix — pre-register two controls and one statistic, all nearly free:**

- **Constant-source control:** all M candidates conditioned on one fixed coordinate. Top-1 must be chance and the between-candidate score spread must collapse to the sampling/scorer noise floor. A failure here is a bug, before any conclusion is drawn.
- **Common random numbers across candidates:** use the *same* K noise draws for every candidate within a query. This is a paired design (strictly lower variance on the between-candidate score differences, no bias on the argmax), it costs nothing, and it makes the control above a crisp equality check — identical noise + identical conditioning must give a bit-identical waveform. §2.3's "fresh noise each" should be replaced by this, with the per-query generator seed logged.
- **Power statistic:** per query, log between-candidate score spread vs within-candidate (over-k) spread, and pre-register a smoke-rung GO/NO-GO on it. If the between-candidate signal sits inside the noise, the full sweep is unpowered and should not be launched as-is.

### BLOCKER 4 — The measured-RIR power gate is missing, and the proposed non-trivial oracle does not exist in this repo

§2.6's non-trivial oracle is conditioned on "a different `single_channel_ir_*` channel, if present". Grepping every dataset config, the README and `baselines/`: **only `single_channel_ir_1` is referenced, 10 occurrences, and no other channel folder appears anywhere**. Planning the only non-trivial upper bound on a folder no config has ever named is planning to have no upper bound.

There is an always-available substitute, and it is strictly more informative:

> **Measured-impostor test.** For each query, score every candidate with its **measured** RIR at the same receiver: `s_m = sim(E_a(h_obs), E_a(h(s_m, x_r)))`. The m = GT term is the ceiling (exactly 1 under the BLOCKER-2 fix); the m ≠ GT terms are the impostor distribution. If AGREE cannot separate a source's own measured RIR from the measured RIRs of the other sources at the same receiver, **no generator can**, and the generative sweep is unpowered.

This needs only the dataset + `AGREE_*.pt` + `weights/FLAC/VAE.ckpt` — **no FLAC checkpoint** — so it can run the moment the rsync lands, potentially days before the FLAC weights. It also yields the single most interpretable diagnostic the experiment can produce: *does the FLAC-generated GT RIR beat the real RIRs of the wrong candidates?*

**Fix:** promote it to run **R0.5** in §5, ahead of R1; pre-register a GO/NO-GO threshold (measured-RIR top-1 ≥ a stated value, plus a stated impostor margin) and gate R1/R2 on it. Keep the identity oracle as the pipeline sanity check it is, and drop the second-channel variant to "if the rsynced dataset turns out to ship one".

### HIGH 5 — Headline aggregation contradicts exp_01's established finding, and the random baseline is computed under a different convention than the headline

§2.6: *"Aggregation follows the paper convention: per-room means first, then average over the 17 rooms; pooled-over-samples values reported as secondaries."* exp_01 established the opposite for AR. From `reproduce_flac_table1_results.md`, "Protocol note":

> `eval_FLAC.py` builds its metric callback with `eval_per_scene=False` for AR … these numbers are the release script's standard **all-sample aggregate**. Given every metric lands within ~1σ of Table 1, this is evidently the aggregation behind the paper's AR table; the CLAUDE.md per-scene note applies to the HAA path.

All six metrics reproduced within 1σ under the all-sample aggregate. The plan inverts primary and secondary and attributes the inverted choice to "the paper convention".

Second, internal inconsistency: the random baseline (§2.6) takes *"median from the pooled per-candidate distance distribution (weight 1/M per query)"* — pooled — while the headline median is per-room macro. Mean and success@r are linear and survive either convention; **medians do not**, so as written the baseline median and the method median are different estimands.

Third, the grouping key: `AR_md` sets `md['scene'] = scene_name` = `rel_path.split("/")[-3]`, which is the **scene type** (10 of them, e.g. `Cafe`), not the room (`Cafe_idx_1`, 17 of them). Announcement 01 states "6337 items in 17 unseen rooms (10 scene types)". Reusing `md['scene']` for "per-room" aggregation silently aggregates over 10 groups.

**Fix:** register the **all-sample/pooled** aggregate as the headline (matching exp_01 and the AR protocol), per-room macro as a labelled secondary; compute *every* baseline under *both* conventions (for the macro variant the random baseline's median is the per-room weighted-mixture median, then averaged — still exact, no Monte Carlo); and define the room key explicitly as `scene_name/scene_id` parsed from `relpath`, never `md['scene']`.

### HIGH 6 — No per-query cross-check that the driver's candidate geometry equals the release loader's

§2.2 makes `find_pair_metadata` "naming-tolerant (`S010` vs `S0010`)". Tolerance that silently disagrees with `AR_md.get_receiver_source_location` — which builds `"S00" + str(src) + "_R00" + str(rec) + ".json"` — is a way to produce candidate coordinates from a *different* metadata file than the one the item's own `source` came from, with nothing detecting it.

Note also that `_worklog.md` item 3's inference is not established. The split's **wav** names are `%03d` (verified: unseen sources 1–10, receivers 1–100, *all* three digits, `S010`/`R100` present, no `S0010`), while the train split's wav names follow the release `S00`+n convention (`S0070_R0075_hybrid_IR.wav`, and 125 of 243 train rooms mix 3- and 4-digit widths exactly as that convention predicts). So wav naming and metadata-JSON naming are separate namespaces, and the release rule evidently *does* resolve on the unseen split — exp_01 evaluated all 6,337 items in ~14 min with no substitution storm, which would be impossible if `get_receiver_source_location` failed for every receiver ≥ 10. Tolerance is fine as a **fallback**; it must not become the primary lookup, and it must not mask a divergence.

**Fix:** resolve by the release rule first, fall back only with a logged warning, and add the one assert that subsumes the whole class of errors:

> per query, assert `project_to_camera(rec_loc, cand_xyz[gt]) == md['source']` — **exact float32 equality after the same cast chain**, not `allclose` — and that `rec_loc` agrees across the pair files used. Hard-fail on mismatch.

That single check catches: wrong metadata file, wrong receiver, wrong coordinate frame, a rotated/translated reimplementation, and a substituted item. The §4.2 parity test against `AR_md.get_3d_point_camera_coord` should likewise assert exact equality, not `allclose`, because candidate/context membership (BLOCKER 1) is decided by 6-decimal fingerprint strings.

### HIGH 7 — `sample_context_ids` is a near-vacuous leakage assert and should not be presented as the integrity control

§2.1 promises "a context fingerprint via `eval_FLAC.sample_context_ids` must not contain the GT source position … hard-fails on violation". What that function actually fingerprints is `context_poses` — **positions only**, rendered at 6 decimals, read from the very tensor whose construction defines the exclusion. Since `get_ir_and_location_for_other_sources` removes the target node *before* the draw, the assert can essentially never fire; it is a regression guard on `AR_md`, not evidence about leakage. It says nothing about audio leakage, and nothing about the real leakage-shaped risk, which is BLOCKER 1 (the near-total context coverage of the non-GT candidates). Presenting it in §7 as the leading integrity control overstates the plan's protection.

Two mechanical notes for the implementation: the function is fail-closed on dtype and shape (raises unless `context_poses` is exactly float32 `[K,3]`, with a documented sensitivity — float16 rendering changed 5,032 strings on the real unseen split), and its docstring requires it be read **before** any metadata manipulation.

**Fix:** keep the assert, relabel it in §7 as a regression guard on the draw, and add the informative record it makes possible — the per-query candidate-membership vector required by BLOCKER 1's fix.

### HIGH 8 — The protocol constants that determine the context draw are not pinned

§2.1 says "`batch_size` for iteration only; `pl.seed_everything(seed)` fixes the context draws". Not quite: the draw happens inside dataloader workers via `np.random.choice`, and `pl.seed_everything(seed, workers=True)` seeds each worker from (base seed, worker id). Which item receives which draw therefore depends on **`num_workers` and `batch_size`**. Changing either changes the contexts, hence the queries, hence every number — and would silently break the cross-arm query pairing that this preflight exists to establish.

**Fix:** register `batch_size`, `num_workers`, `shuffle=False`, `drop_last=false` as protocol constants in `_params_set_up.md`, write them into the summary JSON, and emit a run-level digest over the ordered per-query context fingerprints (the `canonical_stream_hash` serialization already fixed in `eval_FLAC.py:490` is the right one to reuse) so two runs/arms can be *proven* to have seen identical queries.

### HIGH 9 — Three numeric conventions are unspecified, and each one changes every similarity

`eval_FLAC` fixes all three; the plan names none:

1. **Clamping.** `eval_FLAC.py:1313` clamps decoded fakes to `[−1, 1]` *before* metrics, and `SampleDataset.__getitem__` (dataset.py:303) clamps the reals. The plan never says what the localization driver does before `embed_rirs`. Un-clamped fakes go into AGREE as a different signal.
2. **matmul precision.** `evaluate_model` sets `torch.set_float32_matmul_precision('medium')` globally (eval_FLAC.py:1130). Omitting it changes the generations.
3. **Autocast scope and value.** `eval_FLAC` wraps **only** the conditioner call in `cond_autocast_ctx()` and runs sampling/decoding outside autocast. §2.3 lists `--cond-autocast` as a flag but registers no value, which is exactly the pattern announcement 05 exists to forbid.

Also: §2.3 says "EMA weights" as though guaranteed. `resolve_weights_source` exists precisely because a checkpoint lacking `diffusion_ema.ema_model.*` (or a config with `use_ema: false`) silently evaluates the online weights.

**Fix:** in §2.3, register: clamp fakes to `[−1,1]` before embedding and do **not** re-clamp `h_obs` (the loader already did); `torch.set_float32_matmul_precision('medium')`; `--cond-autocast default`; `--frame-avg-max-fwd-samples 64` recorded even though it is inert for vanilla (announcement 06 binds the later arms and `orbit_provenance` records it); and write `weights_source` from `resolve_weights_source` into the summary JSON alongside `source_sha`, the ckpt sha256 and the dataset config.

### HIGH 10 — No non-generative baseline, so a positive result cannot be attributed to FLAC

The context handed to the model already contains eight **measured** RIRs at this receiver with known source positions. Two obvious non-generative predictors exist: (a) rank candidates by the similarity of `h_obs` to each context reference and predict from the reference positions; (b) the absence heuristic of BLOCKER 1. The repo also ships KNN / LinearInterp / RdnAcross / RdnSame in `baselines/eval_baselines.py`. Against uniform-random alone, a positive result cannot be attributed to the generative model rather than to the context.

**Fix:** add a context-retrieval baseline row (AGREE-only, no generation, minutes of compute) to §5 as a required control.

### MEDIUM 11 — The metric suite is nearly degenerate on a discrete M ≈ 10 set, and so is the dev-selection objective

With the GT in C and candidates metres apart, `e_loc = 0` **iff** top-1 is correct. So success@0.5 m and success@1.0 m will very likely be the same number as top-1 accuracy, and the median e_loc is a step function of top-1 (0 whenever top-1 > 50 %, a wrong-pick quantile otherwise). §2.5 then selects τ/agg/K′ by *"minimizing dev median e_loc"* — an objective that is literally constant over large regions of the grid, so the "winning triple" would be decided by an unspecified tie-break.

**Fix:** register **top-1 accuracy** as the primary metric (it is the sufficient statistic here) with mean e_loc and MRR as the continuous secondaries; keep median/success@r as reported-but-derived. Change the dev objective to dev mean e_loc (or top-1) with a **declared deterministic tie-break** (e.g. smallest τ, then agg order `lme < mean < max`, then largest K′). Report the empirical inter-candidate distance distribution per room so success@r is interpretable at all.

### MEDIUM 12 — Statistical validity: 6,337 queries are 17 clusters, not 6,337 independent trials

Queries share rooms (≈373 per room), candidate sets and source positions; each source is the GT ~37 times per room. A binomial CI over 6,337 would be far too tight. And the three seeds vary the *context draw as well as* the generation noise (the same convention exp_01 used for its 5 seeds) — worth stating, since it means the bars are generation+context variance, not sampling variance.

**Fix:** pre-register a **room-clustered bootstrap** (resample the 17 rooms with replacement) for every headline number, and pre-register the comparison against each baseline as a **paired per-query** test (method error vs that query's own exact baseline expectation), not a two-sample comparison.

### MEDIUM 13 — Compute basis cites the wrong exp_01 run, and the conditioner is recomputed M-fold on context that is identical across candidates

§9's basis, "exp_01 full unseen eval ≈ 6.5 min (batch 64, A6000)", is exp_01's **K = 1** run. In the same log (`reproduce_flac_table1_2026-07-04_18:28:50.log`) the five **K = 8** runs took **14:12 – 14:25**. Localization uses the K_ctx = 8 config, so the basis is off by ~2.2× before anything else.

More significant: `GeometryConditioner.forward` loops over the coordinate axis, so one item costs **1 ViT forward for `source_vit` + 8 for `context_poses_vit` = 9**. Calling the conditioner once with M = 10 candidate dicts (§2.3) costs **8 × M = 80 context ViT forwards for context that is bit-identical across all candidates** — ~90 forwards per query where 18 suffice. Cross-checking the two exp_01 timings gives ≈ 10.6 ms per ViT-forward-per-item, i.e. ≈ 0.95 s/query of pure redundant conditioning.

**Fix:** choose explicitly and say which —

- (a) keep the naive path and budget honestly: ≈ 1.0–1.5 s/query ⇒ **2–3 h per full-split seed**, and a campaign (dev + 3 unseen seeds + oracle + baselines) of **~10–15 h**, not "1–4 h per run"; or
- (b) compute the `context_*` conditioning **once per query** and expand across candidates, with a pre-registered equivalence test (naive vs cached scores allclose at a declared tolerance on a fixed 20-query set). Worth ~5×.

Either way, put the **campaign** total in §9, not only the per-run figure, and keep the R0 timing gate.

### MEDIUM 14 — Asset manifest is incomplete; the smoke run cannot start with what §3 / §8.5 lists

Constructing the AGREE model requires two things the plan does not list:

1. **`weights/FLAC/VAE.ckpt` at a path relative to CWD.** `AGREE/AGREE/model_configs/dinoV3.json` sets `audio_cfg.pretrained = "weights/FLAC/VAE.ckpt"`, and `OobleckEncoder.__init__` calls `load_pretrained` **at construction**; the constructor raises without it. (The AGREE checkpoint then overwrites those weights — the file is needed anyway.)
2. **The HF DINOv3 backbone** `facebook/dinov3-vits16-pretrain-lvd1689m` — AGREE's *vision* tower is instantiated by `CLIP(**config)` even though only `encode_audio` is used, and FLAC's own conditioners need the same backbone. This box has **no `~/.cache/huggingface` and `HF_HOME` is unset**, and that HF repo is gated.

**Fix:** add `weights/FLAC/VAE.ckpt`, the AGREE `.pt`s, and a warmed HF cache (or `HF_HOME` pointing at the rsynced cache + a valid token) to §8.5's rsync manifest; make the driver fail fast on all of them *before* touching the dataset; and state that it must be invoked from the repo root.

### MEDIUM 15 — Silent-substitution policy is under-specified and collides with announcement 01

§2.1: substitutions are "logged + excluded, count reported (expected 0)". If any occur, the headline is computed on fewer than 6,337 items — a subset, which announcement 01 forbids for headline numbers.

**Fix:** make it fail-closed — a non-zero substitution count **aborts** the run (the item is re-requested, or the cause is fixed), never silently drops. Also state the audit's mechanics: `sample_target_id` returns `'<idx>|<relpath>'` and the check is positional against the `json_scandir` enumeration, which requires `shuffle=False` **and** `drop_last=false`; assert both from the loaded config rather than assuming them.

### MEDIUM 16 — The smoke run must not touch the unseen split

§5's R0 is "2 rooms, `--smoke --max-queries 4`" with no split named. Drawn from the unseen split, it is a pre-registration peek at test data, however small.

**Fix:** pin R0 — and the tiny-synthetic and real-data-readback rungs — to the **seen** split. The unseen split is opened exactly once, at R2, after the (τ, agg, K) registration is committed.

### MEDIUM 17 — Pre-registration must be auditable, not merely asserted

§2.5 says the winning triple is registered "BEFORE the unseen run". Nothing makes that ordering checkable afterwards.

**Fix:** commit `_params_set_up.md` carrying the registered (τ, agg, K, K_ctx, aggregation convention, GO/NO-GO thresholds) and record **that commit SHA** in the R2 launch manifest and in R2's summary JSON — the ordering is then provable from git alone. Same treatment for the heatmap case-selection rule (§2.7), which is otherwise a promise rather than a record.

### MEDIUM 18 — Missing tests on the parts most likely to be wrong

The §4 test list is good on geometry and aggregation and silent on the rest. Add:

- **log-sum-exp numerical stability at the smallest registered τ.** τ = 0.02 means `exp(1/0.02) = e^50`; the contract must be `S = τ·logsumexp(s/τ) − τ·log K`, never a naive mean of exponentials. Test against a float64 reference at τ = 0.02 and assert no overflow. (§4.3's current tests only probe the τ→0⁺ / τ→∞ limits.)
- **Offline-vs-online consistency.** Re-aggregating the JSONL's logged sims with the registered (τ, agg, K) must reproduce the JSONL's own `S_m` to a declared tolerance. This is what makes §2.4's "τ/K sweeps are pure re-aggregation" *true* rather than assumed — and it forces the plan to register the **serialized precision of the sims** (use `repr`/17 significant digits; 4-decimal rounding would visibly move a τ = 0.02 sweep).
- **`embed_rirs` equivalence** against `Retrieval.compute_audio_features` on a stub. Note the plan's §2.4 wording is inaccurate: `compute_audio_features` **pads only, it never crops** (Retrieval.py:46–47). Cropping is nonetheless the right choice, because `OobleckEncoder.forward` does `latents.view(B, -1)` into `self.project = Linear(320, 512)` sized from `audio_length = 10240` — a longer input is a hard shape error, not a silent difference. State the divergence and test both branches. (Also: do not copy `compute_audio_features`'s `feats.cpu().squeeze()`, which collapses the batch dim at B = 1.)
- **`E_a` determinism** under the BLOCKER-2 protocol; **`.eval()` + device** on the loaded AGREE model.
- **Candidate-in-context detection** (BLOCKER 1) and the **room grouping key** (17 rooms, not 10 scenes — HIGH 5).

### NIT 19 — `candidate_metadata`'s deepcopy is wasteful

§4.2 deep-copies the full query metadata per candidate to replace two keys — that copies a `[3, 256, 512]` depth tensor and 8 × 9,600 context waveforms M times per query. A shallow `dict(base_md)` with the two keys rebound is equally non-mutating and ~free; keep the "base_md not mutated" test unchanged.

### NIT 20 — The heatmap can draw the actual room boundary

§2.7 uses the candidate/receiver extent as a room-region proxy. The query already carries a full 3-D point cloud at the receiver — `md['depth']`, `[3, 256, 512]`, listener-centred (`AR_md.convert_equirect_to_camera_coord`). Projecting it to x–y gives the real room silhouette, which is literally what the user asked for ("the room boundary and valid candidate region"), at no extra cost.

### NIT 21 — fp16 sensitivity check

The registered protocol runs the conditioner under fp16 autocast, and the only thing distinguishing two candidates is a 3-vector entering the ViT as `(coord − depth)/max_value`. A one-off `--cond-autocast off` rerun on the smoke/dev subset, reported as a labelled diagnostic, cheaply rules out "the ranking is an fp16 artifact".

### NIT 22 — Checkpoint-of-record framing

§8.1 recommends the exp07_P1 87.5k anchor. The user's spec says "a pretrained Vanilla FLAC model". The released `FLAC_EMA.ckpt` is the published model, is what exp_01 verified against Table 1 within 1σ on all six metrics at both K, and is reproducible by any reader — the stronger substrate for the claim "a *pretrained* Vanilla FLAC can/cannot be inverted". Recommend it as the headline arm, with the program checkpoint as a second labelled row if cheap.

### NIT 23 — R1's cost is optional

R1 spends a full 6,217-item generation pass purely to register (τ, agg, K). Since the sweep is offline, a `dev`-stamped subsample would register the same triple for a fraction of the cost. If the full dev run is kept, say why in §5 — it also doubles as the seen-vs-unseen contrast, which is worth reporting in its own right.

---

## What the plan gets right

Specific to this plan, not generic praise:

- **Reuse over reimplementation.** Routing the driver through `evaluate_model`'s loading path — EMA remap, `check_load_integrity`, cond-method dispatch, `resolve_cond_autocast`, `source_sha`, `orbit_provenance` — is the correct call, and the stated reason (so the FA/yaw-aug/cyl arms later run under their own protocols per announcement 05) is exactly the lesson exp_09 paid for.
- **The smoke quarantine is real, not decorative.** `--max-queries` refusing to run without `--smoke`, plus a `smoke` stamp in both the filename and the record so it can never be aggregated into a headline file, is the right shape of guard for announcement 01 — much better than a convention.
- **Per-sample `s_{m,k}` logging.** Correct instinct, and it genuinely makes the τ/agg/K′ sweep a re-aggregation rather than a regeneration (given the precision and consistency-test fixes in MEDIUM 18). It is what turns "we tried five τ values" from test-set tuning into a labelled sensitivity table.
- **Dev-split registration of the aggregator hyperparameters** before the unseen split is opened — the right structure, needing only the auditability fix (MEDIUM 17) and a non-degenerate objective (MEDIUM 11).
- **The exact random baseline** instead of Monte Carlo, computed over each query's own candidate set so room geometry is respected. (It just needs a companion that is not so easy to beat — BLOCKER 1.)
- **Honesty about the identity oracle.** §2.6 flags the triviality up front rather than shipping a fake 100 % upper bound. The reasoning is right even though the stated mechanism (cos = 1) turns out not to hold (BLOCKER 2).
- **Taking the silent-substitution hazard seriously.** `SampleDataset.__getitem__`'s `return self[random.randrange(len(self))]` is a genuine landmine in this repo, and planning a per-item identity audit against the split enumeration — rather than trusting item counts — is the correct response.
- **Node parsing planned as numeric and naming-tolerant, with a parity test against `AR_md.get_3d_point_camera_coord`.** The right two instincts (do not trust the release string builder; prove the reimplementation matches), needing only exactness and the GT cross-check (HIGH 6).
- **Per-function test list, in `src/tests/`, before implementation** — announcement 02 satisfied in form and in substance; the §4 tables are contracts, not a checklist.
- **The explicit "not touched" list** (§4.7) keeping every release file and all of `src/data/`, `src/models/`, `src/training/` out of the diff, with the new code purely additive.
- **Pre-registered heatmap case selection** (largest top-2 margin / smallest margin / largest error) instead of hand-picking — exactly the discipline the user's "representative sharp-success, ambiguous, failure cases" invites someone to violate.

---

## Suggested minimum bar for re-review

A revised plan clears this round if it: (1) adds the context-membership covariate, the absence baseline and a K_ctx arm from the existing configs; (2) registers a deterministic embedding protocol and a dedicated noise generator; (3) adds the constant-source control, common random numbers and the power statistic with a smoke-rung GO/NO-GO; (4) promotes the measured-impostor power gate to run before R1 and drops the second-channel dependency; (5) fixes the aggregation convention to exp_01's all-sample primary with a correctly-matched baseline and a real 17-room key; and (6) pins clamping, matmul precision, `--cond-autocast`, `weights_source`, `batch_size` and `num_workers` in §2.3/§2.1. The remaining MEDIUM/NIT items can ride into the first Coder round.
