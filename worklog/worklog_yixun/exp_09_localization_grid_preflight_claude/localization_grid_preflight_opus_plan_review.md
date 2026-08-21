# Plan review — exp_09_localization_grid_preflight

**Reviewer:** UNAVAILABLE — Anthropic Claude was not reached (Claude Code 2.1.237 native CLI, read-only tools, `--model opus --effort max`) · **Date:** 2026-08-20

**Review status:** NOT RUN — AUTHENTICATION BLOCKED

The SOP-mandated review invocation failed before a model session was created:

```text
Failed to authenticate: OAuth session expired and could not be refreshed
```

Immediate `claude auth status` evidence:

```json
{
  "loggedIn": false,
  "authMethod": "none",
  "apiProvider": "firstParty"
}
```

No plan verdict or findings were produced. This file records the failed infrastructure attempt only and **must not** be treated as plan approval. Per reviewer reciprocity in `worklog/experiment_SOP.md`, the Codex Planner did not substitute itself or another OpenAI model for the unavailable Claude reviewer.

Recovery: authenticate the installed Claude Code extension/native CLI, then rerun the read-only prompt in `localization_grid_preflight_opus_plan_review_prompt.md` with `--model opus --effort max`. Replace this unavailable record only by appending the actual dated review below it, preserving this failed-attempt provenance.

---

# Plan review — exp_09_localization_grid_preflight (actual review)

**Reviewer:** Anthropic Claude Opus 5 (`claude-opus-5`, Claude Code 2.1.237 VS Code extension, interactive session, read-only inspection + read-only verification scripts) · **Date:** 2026-08-20

**Verdict:** `REQUEST-CHANGES`

> **Invocation deviation, stated explicitly per SOP.** The SOP names "Claude Opus 4.8 at max effort" as the Claude-family reviewer, and the recovery note above prescribes `--print --model opus --effort max` through the native CLI. This review was instead produced by **Opus 5 in an interactive Claude Code session** at Yixun's direct request (「根据SOP进行一下审核」). Opus 5 is the strongest available Claude tier, so the "strongest model of the reviewing family" requirement is satisfied and reviewer reciprocity holds (Planner = Codex, Reviewer = Claude). The *invocation method* differs from the plan's stated recovery procedure; it is recorded here rather than silently substituted. No file outside this experiment folder was modified.

Context loaded before judging: `worklog/experiment_SOP.md`; announcements 01/02/03; the exp_09 query, worklog, plan and review brief; `acoustic_localization_brief.pdf` (all 3 pages); exp_07/exp_08 plans, audits and analyses; and the live source — `eval_FLAC.py`, `src/configs/dataset_configs/custom_metadata/AR_md.py`, `src/models/conditioners.py`, `src/metrics/modules/Retrieval.py`, `AGREE/AGREE/model.py`, `AGREE/AGREE/audio_model.py`, `src/configs/model_configs/FLAC/AR/FLAC_AR_InContext.json`, `src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json`, `src/tests/`.

Findings marked **[measured]** were verified by running read-only scripts against the real split, the real metadata, and the official OBJ meshes at `/home/zhixuanzhao/projects/rir2rir/third_party/AcousticRooms/room_mesh_obj_format`. Per-room numbers below are exact for the full split where stated, and computed on the first 120 queries per room where a per-query sweep was needed (noted inline).

---

## Summary

The plan's frozen protocol, score definition, leakage controls and TDD decomposition are sound and faithful to Eq. (3) of the brief. Three verified facts, however, mean the plan as written cannot produce the headline metric it pre-registers:

1. its own candidate-validity rule makes **21.4 % of the 5,337 queries unwinnable at the 0.5 m success threshold before FLAC runs**, on a premise about the dataset that is factually false;
2. **520 queries (8.2 %) cannot supply the 8 unique contexts** the plan's D1 contract requires, and the plan's fail-closed rule would drop them, breaking the pre-registered denominator;
3. the compute gate is deferred to ladder rung 8 with **no pre-registered fallback**, while §5 forbids changing the grid afterwards — a deadlock, on a workload measured at **≥25.3 M candidate evaluations** before masking.

All three are fixable inside the current design; none requires abandoning the approach. Details, evidence and required corrections follow.

---

## Blocking findings

### B1 — The 1.0 m context-source exclusion destroys the grid-oracle floor for 21.4 % of queries, and its stated justification is false **[measured]**

*Plan §1.2, query-valid mask; risk #5.*

The plan excludes candidates within `1.0 m` of any selected context source, justified as "matches the dataset's published inter-source separation." Measured minimum inter-source distance per room, over the 16 mesh-available unseen rooms (exact, all sources):

```
0.20  0.40  0.50  0.50  0.51  0.51  0.71  0.71  0.72  0.73  1.00  1.02  1.02  1.52  4.03  4.13
```

Ten of sixteen rooms have sources closer than 1.0 m; the minimum is **0.20 m**. There is no 1.0 m inter-source separation in this data, so a 1.0 m exclusion ball around 8 context sources routinely deletes exactly the lattice cell containing the target.

Grid-oracle error `min_c ||c − x*||` under each candidate rule (first 120 queries/room, weighted to the 5,337-query subset):

| context clearance | queries with oracle error > 0.5 m |
|---|---|
| none | **0.0 %** |
| 0.25 m | **0.0 %** |
| 0.50 m | 3.5 % |
| **1.00 m (plan as written)** | **21.4 %** |

Per room at 1.0 m: `Bathrooms_idx_18` **100 %**, `Bathrooms_idx_14` 90 %, `Bedrooms_idx_33` 90 %, `Bedrooms_idx_18` 81 %, `MeetingRoom_idx_32` 59 %, `Apartments_idx_42` 20 %, `LivingRoomsWithHallway_idx_25` 11 %, `Restaurants_idx_22` 8 %. The 0.5 m receiver clearance alone causes **0 %** damage in every room — the harm is entirely attributable to the context rule.

These figures ignore the 0.5 m mesh-surface clearance, which can only remove further candidates; they are lower bounds. In the small rooms (`Bathrooms_idx_18` is 3.2 × 2.0 × 2.8 m, 144 raw lattice points) the combination of surface clearance, receiver clearance and eight 1.0 m balls may leave the **per-query candidate set empty**. The plan's G1 acceptance criterion only checks that the *per-room base* grid is nonempty, so this failure mode is not caught.

**Required correction.** Replace the 1.0 m context clearance with a numerical-duplicate guard at **0.25 m** (half a lattice step), which achieves the plan's stated purpose — no lattice point can coincide with a context source position — at **0.0 %** measured oracle damage; or drop the rule entirely and rely on the fact that a context source is never the target. If Yixun prefers to keep a larger radius, it must be pre-registered together with the per-room oracle table above and the headline must be reported as oracle-conditional. Additionally, add a per-**query** fail-closed assertion that the valid candidate set is nonempty and that `e_oracle` is finite, and emit the per-room oracle distribution from the G1 audit *before* any generation runs.

### B2 — 520 queries cannot supply 8 unique contexts; the plan's D1 contract and fail-closed rule would break the 5,337 denominator **[measured]**

*Plan §1.1 and §3/D1 ("exactly eight unique contexts from the other nine", "missing context fails closed rather than replacement sampling").*

Counting the actual `single_channel_ir_1` directory against the split (exact, all 6,337 queries):

```
valid same-receiver context sources available:  6 → 91 queries
                                                7 → 429
                                                8 → 5263
                                                9 → 554
520 of 6,337 queries (8.2 %) have fewer than 8 — all of them in Cafe_idx_1
```

`Cafe_idx_1` holds 922 of the 5,337 in-scope queries and is missing 78 of its 1,000 IR files, so 56 % of that room's queries have only 6 or 7 same-receiver contexts. The released `get_ir_and_location_for_other_sources` (`AR_md.py:103-106`) handles this by silently falling back to `np.random.choice(..., replace=True)` — duplicated context RIRs. The plan forbids that fallback *and* requires exactly eight, so it would hard-fail on 520 queries and the fixed 5,337 denominator would be unreachable.

A second, related parity issue: that function builds the candidate path as `f"S00{node}"` while the IR files are named `S010_...` for source 10. For `node = 10` the constructed name is `S0010_R0xx_hybrid_IR.wav`, which never exists. **Source S010 is therefore permanently excluded from every context pool in the released evaluation path** — including the exp_01/exp_02 numbers. Consequences: for 9 of 10 targets the context *set* is fully determined (all 8 remaining sources are taken; only the order is random), so the plan's "sort the nine non-target source IDs and choose eight with a deterministic RNG" describes a selection freedom that does not exist for 90 % of queries, and the metadata lookup convention (`S0010_R00100.json`, which *is* the real metadata filename format) differs from the IR convention.

**Required correction.** Pin the context policy explicitly in the plan, choosing one and stating the consequence:
(a) reproduce the released pool exactly, including the `S00{node}` quirk, and define a deterministic, documented policy for the 520 short queries (e.g. use all available contexts and pad the conditioning mask, or deterministically duplicate as upstream does) — preferred for parity with exp_01/02; or
(b) fix the padding bug so S010 becomes eligible, and record it as an explicit, reviewed deviation from the baseline evaluation path with its own regression test.
Either way, D1's test list must be corrected: "exactly eight unique contexts from the other nine" is false as a general invariant, and a test asserting it will encode the bug. Add a test that the context-count histogram over the full split matches the measured `{6: 91, 7: 429, 8: 5263, 9: 554}` (or its 16-room restriction), so any silent change to the pool is caught.

### B3 — No pre-registered compute fallback, and the conditioning path costs 9 ViT forwards per candidate instead of 1 **[measured]**

*Plan §1.2 (uncapped lattice), §4.8, §5, risk #2.*

Raw lattice sizes from the official OBJ AABBs at 0.5 m spacing, summed over the in-scope queries:

```
Cafe_idx_1        9,996 pts × 922 q  =  9.22 M
Auditorium_idx_1 13,824 pts × 1000 q = 13.82 M
… all 16 rooms                       = 25,312,262 candidate evaluations
```

At `K = 4` that is **≈101 M conditional generations** before free-space masking (masking will cut it, but two rooms carry 91 % of the load). The plan places the cost projection at ladder rung 8 and says "stop and ask" — but §5 simultaneously rules "changing 0.5 m grid spacing after reading quality results" out of scope. Reaching rung 8 and finding the cost unacceptable therefore leaves no pre-registered move, and any reduction chosen at that point is post-hoc by construction.

Worse, the plan's premise that candidates "share receiver, depth panorama, context RIRs, context poses" implies the shared conditioning is computed once. In the released path it is not. `GeometryConditioner.forward` (`src/models/conditioners.py:194-224`) loops over the coordinate axis and runs the ViT once per coordinate:

```python
for i in range(coord.shape[1]):
    c = (coord[:, i, :, None, None] - depth_coord) / self.max_value   # [B, 3, H, W]
    c = self.vit(c)
```

`FLAC_AR_InContext.json` instantiates **two** `ViTCoordinates` conditioners on DINOv3 ViT-S/16 at 256 × 512: `source_vit` (N = 1) and `context_poses_vit` (N = 8). So a naive per-candidate call costs **9 DINOv3 forwards**, of which 8 — the context ones, plus `context_poses` and `context_audio` — are candidate-invariant. Note also that the source ViT input is `depth_panorama − candidate`, i.e. genuinely candidate-dependent, so the panorama itself cannot be cached as a feature; only the context branch can.

**Required correction.**
1. Make the query-invariant conditioning cache an explicit, tested I1 contract: compute `context_poses_vit`, `context_poses`, `context_audio` once per query, recompute only `source_vit` and `source` per candidate, and add a test asserting the cached path produces bit-identical prepend-conditioning tokens to the uncached path on a small case. This is a ~9× saving and is required for the run to be affordable at all.
2. Move the cost projection **before** I1: after G1 the true per-room valid-candidate counts are known from geometry alone at negligible cost. Gate implementation of the engine on that number.
3. Pre-register the reduction ladder **now**, in priority order, before any localization quality is observed — e.g. (i) restrict the z-lattice to the observation-derived source-height band (see N1, ~3.5× in tall rooms, no leakage), (ii) coarse-to-fine search with a pre-registered refinement rule, (iii) reduce `K`, (iv) as a last resort a room-level protocol change, which conflicts with announcement 01 and needs Yixun's explicit sign-off.

### B4 — Validation-ladder rung 2 cannot pass today: the permanent regression suite is red **[measured]**

*Plan §4.2 ("new localization tests plus all existing `src/tests/`"); announcement 02 ("existing tests are permanent regression assets: they must keep passing in every later experiment").*

```
python -m pytest src/tests -q
  → ERROR collecting src/tests/test_eval_paths.py
    ModuleNotFoundError: No module named 'compare_predictions'
python -m pytest src/tests -q --ignore=src/tests/test_eval_paths.py
  → 90 passed in 17.87s
```

`src/tests/test_eval_paths.py:36` still resolves the exp_02 comparator at the **pre-announcement-03 path**:

```python
_EXP02_DIR = Path(__file__).resolve().parents[2] / "worklog" / "exp_02_yaw_noninvariance_claude"
```

The 2026-07-12 namespace migration moved that directory to `worklog/worklog_yixun/exp_02_yaw_noninvariance_claude/`. Announcement 03 records that five worklog scripts were converted to a `.git` marker-walk during the migration; this test module was missed, so the whole module has been uncollectable since then. The plan's I1 round explicitly plans to *extend* this file, which would build on a module that does not import.

**Required correction.** Fix the path as a "round 0" commit before any exp_09 code — preferably with the same marker-walk approach used for the migrated scripts, so the test is layout-proof — and record the green rung-2 baseline in `_worklog.md` before G1 opens.

### B5 — AGREE input parity is underspecified, and a reviewed reference implementation already exists

*Plan §1.4 and risk #4.*

The plan says only "encode … through frozen `AGREE.encode_audio(..., normalize=True)`". That is not enough to make `s = cos(E_a(h_obs), E_a(ĥ))` well-posed, because:

- `AudioResNet18.forward` (`AGREE/AGREE/audio_model.py:48-53`) applies `Spectrogram(n_fft=1024, win_length=512, hop_length=256)` and then a ResNet-18 with global pooling — the embedding depends on the input length through the frame count, so `h_obs` and `ĥ` **must** be the same length or the cosine is comparing different geometries;
- `eval_FLAC.py` clamps generated audio (`fakes = fakes.clamp(-1.0, 1.0)`) before it ever reaches a metric, while the observed RIR is whatever the dataloader produced;
- `src/metrics/modules/Retrieval.py:44-50` is the **already-reviewed** preprocessing used for the FLAC retrieval metric — pad to 10 240 then `encode_audio(h, normalize=True)` — and `FLAC_AR_InContext.json` sets `sample_size: 10240` at `sample_rate: 22050` with `random_crop: false`, so the dataloader's `h_obs` is the first 0.464 s.

**Required correction.** Pin in §1.4: mono `[B, 1, 10240]`, float32, generated audio clamped to `[-1, 1]` exactly as `eval_FLAC.py` does, observed RIR taken from the dataloader's `sample_size` crop (not re-read from disk with a different convention), and **reuse `Retrieval.compute_audio_features` (or a shared extraction of it) rather than reimplementing** — with a test asserting the localization path and the metric path return identical embeddings for the same waveform. Add the byte/number audit the plan's risk #4 already promises, as a rung-4 item with recorded numbers.

### B6 — The base mesh-clearance mask has no numerical tolerance at a boundary that is exactly critical **[measured]**

> **Absorbed into B7.** B6 was written against the AABB evidence and treats 0.5 m as correct-but-boundary-critical. B7 measures the same rule against the actual mesh and shows the threshold itself is wrong, so an eps tolerance alone does not fix it. B6 stands only as the eps-hygiene requirement; the threshold decision is B7's.

*Plan §1.2. The eps tolerance is stated only for the query-valid mask, not the base room-valid mask.*

Minimum distance from a real source to the room AABB faces, per room, is **exactly 0.50 m** in `Bathrooms_idx_14`, `Bathrooms_idx_18` and `LivingRoomsWithHallway_idx_30`, and 0.53–1.13 m elsewhere. The dataset's placement rule is evidently "≥ 0.50 m", so the plan's 0.5 m surface-clearance threshold sits exactly on the placement boundary. A strict `distance >= 0.5` against a raycast unsigned distance, in float32, will non-deterministically delete candidates in precisely the shell where sources live — and these are the smallest rooms, already the worst affected by B1.

**Required correction.** Apply the same explicit `distance + eps >= threshold` tolerance to the base room-valid mask, state the eps, and make the G1 audit report, per room, how many *real* metadata source anchors would survive their own mask. Any room where a real source fails its own validity test is a fail-closed condition, not a warning.

### B7 — The geometry backend the base mask depends on does not work on these meshes, and the 0.5 m clearance excludes source positions the dataset itself uses **[measured]**

*(plan §1.2 "Base room-valid mask"; §1.3 primary backend; supersedes N6, which understated this as a tolerance issue)*

Two measured facts about `§1.2`'s `occupancy AND distance >= 0.5 m` mask:

**(a) Open3D occupancy is semantically inverted here, and the meshes violate its precondition.** All **16/16** included OBJs report `is_watertight() == False` **and** `is_edge_manifold() == False`. `RaycastingScene.compute_occupancy` classifies **~100 % of the real source and receiver metadata anchors as "inside a solid"**. This is not a defect in Open3D: occupancy tests containment inside a closed surface, and a room shell's air volume *is* inside its shell. So `occupancy == 0` — the plan's "inside the room free space" — actually selects the space **outside** the room. Nor can the sign simply be flipped: because the meshes are non-manifold, inverted occupancy still misclassifies 10 % of real anchors in `Apartments_idx_50` and 4-8 % in `Office_idx_10`, `LivingRoomsWithHallway_idx_30/_25` and `Restaurants_idx_24`. The plan's own fail-closed acceptance criterion #2 would catch this — and would then fail all 16 rooms, killing the experiment at G1 with no fallback.

**(b) The 0.5 m clearance is justified against the wrong reference.** The claim "matching AcousticRooms' published source-placement clearance" holds only against the room **AABB** (measured minimum exactly 0.50 m). Against the **mesh surface** — which is what §1.2 actually specifies — the measured global minimum real source-to-surface distance is **0.232 m** (`Restaurants_idx_22`), and **7 of 16 rooms** contain a real source closer than 0.5 m to a surface. Receivers sit 0.10-0.19 m from surfaces. A 0.5 m clearance therefore excludes positions the dataset itself uses as sources. Measured cost, with the B1 correction (receiver 0.5 m, context 0.25 m) already applied and validity by distance only:

| Surface clearance | queries with `e_oracle > 0.5 m` | candidates over the 5,337 subset |
|---|---|---|
| 0.20 m | **0.0 %** | 17.9 M |
| 0.25 m | **0.0 %** | 17.1 M |
| 0.30 m | 0.6 % | 15.1 M |
| **0.50 m (planned)** | **3.3 %** | 12.0 M |

The 30 % candidate saving costs 3.3 % of queries outright, concentrated in `LivingRoomsWithHallway_idx_30` (24.2 %), `MeetingRoom_idx_20` (13.3 %), `Bedrooms_idx_33` (12.5 %), `Office_idx_10` (11.7 %), `Office_idx_11` (10.0 %).

**Why it matters.** The mask conflates two different jobs: *physical validity* (do not place a source inside a wall or a solid), which is genuinely required but whose correct test is occupancy/ray-parity rather than distance; and an *in-distribution prior* (FLAC never saw a source 5 cm from a wall), which is a modelling choice whose threshold must be measured, not asserted. With occupancy unusable, distance is forced to carry both jobs alone — which is exactly why the 0.5 m value became load-bearing, and it is the wrong value. Distance-only validity also over-includes: a point deep inside thick furniture or a wall cavity can be >0.2 m from every surface and still be invalid.

**Required correction.** (1) Replace the occupancy backend with a method robust to non-watertight, non-manifold geometry — multi-direction ray-parity majority voting is the standard choice — and keep the plan's anchor test as its acceptance criterion: *every* real source and receiver anchor in all 16 rooms must classify as free space before G1 closes. (2) Separate the validity mask from the clearance prior in both the plan text and the code. (3) Re-derive the clearance from the measured distribution rather than asserting it: **0.20 m**, strictly below the observed 0.232 m minimum, with the per-room source-to-surface distance table published in `_params_set_up.md`. (4) Re-run the `e_oracle` gate from B1 under the corrected mask and confirm 0 queries exceed 0.5 m.

**Revised compute figure for B3.2.** Under the corrected masks the real candidate volume is **~17.1 M candidates** over the 5,337-query subset (mean 3,212 per query), i.e. **~68.6 M generations at `K = 4`** — a firmer number than the ">= 25.3 M before masking" AABB bound quoted in B3, and it does *not* shrink the problem: `Cafe_idx_1` (7,372 candidates/query) and `Auditorium_idx_1` (9,853) still dominate, so the B3.2 post-G1 cost gate and the pre-registered reduction ladder remain mandatory.

---

## Non-blocking recommendations

**N1 — Restrict the z-lattice to the observation-derived source-height band.** Measured source heights are `z_s ∈ [0.99, 2.20] m` across all 16 rooms (mostly 1.2–1.7 m), while room heights are 2.5–4.3 m. A full-height lattice therefore spends most of its candidates at heights where FLAC has never seen a source — pure OOD compute that also inflates 3-D error if the model is z-insensitive. The 8 context source poses are **part of the observation `O`** and are already fed to the model, so deriving a z-band from the context sources (padded by one lattice step) leaks nothing about the target and is defensible in the paper. In `Auditorium_idx_1` this cuts 7 z-levels to ~2. Report the band per query in the manifest.

**N2 — The 3-D grid deviates from the brief's own §4, and the brief must be updated.** `acoustic_localization_brief.pdf` §4 states "Candidates lie on a common valid 2-D grid at source height." Yixun's Query 3 supersedes this with an isotropic 3-D grid, and the 3-D choice is the *better* science — a 2-D-at-source-height grid leaks the target's height into the candidate set. But the plan should say so explicitly and flag that the paper's protocol sentence and the reserved Table 1 caption now need editing, otherwise the manuscript and the experiment disagree.

**N3 — exp_09 fills only the diagnostic column of Table 1.** The brief's §4 Metrics says "The primary split is unseen-room plus random yaw; canonical coordinates are diagnostic." exp_09 is canonical-only (yaw is out of scope per §5). That is a fine preflight, but the plan should state that the "Med. err. rand." and "random-yaw degradation" entries of Table 1 cannot be produced by exp_09, so nobody later mistakes the canonical number for the primary one.

**N4 — Report a mean-aggregated score as a zero-cost co-diagnostic.** At `τ = 0.1` and the cosine spreads typical of RIR embeddings, `S` is effectively `max_k s`. A max-like statistic's expected winner value grows with the number of candidates, and per-room candidate counts differ by ~96× (144 in `Bathrooms_idx_18` vs 13,824 in `Auditorium_idx_1`). This injects a room-dependent bias into a room-bootstrapped headline. Recording `mean_k s` alongside `S` costs nothing (the `s[x,k]` are already computed) and lets the analysis show the verdict is not an artifact of the aggregation. Also pre-register a sanity check that `τ` is commensurate with the empirical spread of `s`, measured on a debug slice only.

**N5 — Add an off-grid ground-truth score probe as a diagnostic control.** Generate at the continuous `x*_s` (off-grid, not inserted into the candidate set) for a pre-registered handful of queries and check that its score exceeds every grid candidate's. This separates "the score is uninformative" from "the grid is too coarse" — the single most useful piece of information if the headline comes out negative — and it does not violate the no-GT-insertion rule because it never enters the argmax.

**N6 — Elevate oracle-normalized success to co-primary.** *(the eps-tolerance point formerly raised here is superseded by B7, which shows the threshold itself is wrong.)* Measured per-room median oracle error ranges from **0.000 m** (`Apartments_idx_50`/`_42`, where sources sit at exactly z = 1.50 m and the global 0.5 m lattice happens to contain the truth exactly) to 0.28 m (`Cafe_idx_1`). With that heterogeneity, a room-bootstrapped `success@0.5 m` is driven substantially by lattice-alignment luck. Report raw and oracle-normalized success side by side in `_results.md`, not the latter as a footnote.

**N7 — Add a real-vs-generated embedding calibration diagnostic.** AGREE was trained on real RIRs; `ĥ` is VAE-decoded and clamped. Report the distribution of `cos(E_a(h_obs), E_a(h_real,other))` against `cos(E_a(h_obs), E_a(ĥ))` so the analysis can tell a domain gap from a localization failure.

**N8 — `cfg_scale = 1.0` is correctly pre-registered for parity** with exp_01/02 (their metric filenames `..._metrics_1_1.0_...` confirm `steps=1, cfg=1.0`). If a CFG sweep is wanted, designate it before headline results and label it diagnostic; CFG is not among the quantities the brief freezes, so it is the one knob that could be explored without breaking the pre-registration — but only if decided now.

**N9 — Open3D is not installed in the interpreter that actually runs this repo. [measured]** The 2026-08-20T11:56 worklog entry concluded "System Python has Open3D 0.19.0; no new mesh dependency is required." That is true of `/usr/bin/python3` (0.19.0) but **not** of `/home/zhixuanzhao/projects/Frame_Average/FLAC-vanilla/.venv/bin/python`, which is the `python` on `PATH`, holds torch/pytorch-lightning, and is the interpreter that runs `src/tests/` — there `import open3d` raises `ModuleNotFoundError`. G1's geometry code and its pytest tests must run in that venv, so the dependency has to be installed there and pinned in `pyproject.toml` before the geometry round opens.

---

## SOP compliance

**Correct:**
- Reviewer reciprocity was honoured: the Codex Planner did **not** substitute itself or another OpenAI model when Claude was unreachable, and the failed attempt was recorded with its exact error rather than dressed up as a verdict. This is exactly what the SOP asks for and it is the reason this review is possible at all.
- Scaffold order (query → worklog → plan → plan review) matches the SOP sequencing; artifact names match the SOP templates; no implementation code exists ahead of approval; the working tree is clean.
- Announcement 01's full-split rule was not quietly bent: the `ListeningRoom_idx_2` exclusion is an explicit user decision, is labeled "mesh-available preflight subset", and its 1,000-query cost is stated. I independently confirmed the missing asset — `room_mesh_obj_format/ListeningRoom/` contains only `idx_0.obj` and `idx_1.obj` — and that the remaining 16 rooms all have meshes, and that the split arithmetic 6,337 − 1,000 = 5,337 is exact.

**To fix:**
- `commits_localization_grid_preflight.md` row 3 still reads `_(this commit)_`; it must carry the real SHA `35373a5`. Rows 2–3 use short SHAs where row 1 uses the full hash — make them consistent.
- `_worklog.md` has no **Version Control** entry recording SHAs `3760c86` and `35373a5`; the SOP requires every SHA inline. The 12:08 missing-mesh entry describes the change but not the commit that carried it.
- The plan's §4 ladder should gain the rung-2 green baseline as an explicit precondition (see B4) and the post-G1 cost gate (see B3.2).

---

## Checklist for user approval

Implementation may open once all of the following are true:

1. **B1** — context clearance reduced to 0.25 m (or removed), or the 1.0 m rule explicitly re-affirmed by Yixun with the 21.4 % oracle-loss table attached to the plan; per-query nonempty-candidate and finite-oracle assertions added.
2. **B2** — context policy pinned to one documented option, the 520 short-context queries handled deterministically without breaking the 5,337 denominator, and the D1 test list corrected (the "eight of nine" invariant removed, the measured histogram asserted instead).
3. **B3** — query-invariant conditioning cache made an I1 contract with a bit-identity test; the cost projection moved to a post-G1 gate; the reduction ladder pre-registered in writing before any quality number is seen.
4. **B4** — `src/tests/test_eval_paths.py` path fixed and the full 91-module suite green, recorded in `_worklog.md` as the rung-2 baseline.
5. **B5** — AGREE preprocessing pinned (mono, 10 240 samples, clamped, dataloader crop) and bound to `Retrieval.compute_audio_features` with an equality test.
6. **B6** — explicit eps tolerance on the base mask (threshold value itself decided under item 7).
7. **B7** — occupancy backend replaced with a non-watertight-robust method and validated against all real anchors in all 16 rooms; validity mask separated from the clearance prior; clearance re-derived to 0.20 m from the measured 0.232 m minimum; B1's `e_oracle` gate re-run under the corrected mask.
8. **N1/N2/N3** folded into the plan text as pre-registered protocol statements (z-band rule, 3-D-vs-brief deviation, canonical-only scope), and **N4/N5/N6** added to the pre-registered reporting set — all before the first generation runs.
9. Commits ledger and worklog SHA bookkeeping brought up to date; **N9** Open3D installed in the venv that runs `src/tests/` and pinned in `pyproject.toml`.

Items 1–3 and 7 change what the experiment measures and must be settled with Yixun. Items 4–6 and 9 are mechanical. Nothing in this review requires a different *method* — the analysis-by-synthesis design, the frozen protocol, the score and the TDD decomposition are all sound — but B7 does require a different *geometry backend* than the one §1.3 names.
