# exp_25 depth_canon — Yixun's driving query

## Query 1 — 2026-10-23 (two parts)

**Verbatim:**

> I need you to find the FA FLAC's previous code (running the results of "[LaTeX table `tab:rir`: FLAC / \method / \YawAugFLAC at K=1 and K=8 — T60 10.287/9.543/9.397 … 8.993/8.202/7.965, C50, EDT, R@1/5/10, mean ± sd over seeds 42–46]") and then help me run anothe experiment as the baseline of orbitrir (FA FLAC): canonicalization experiment. The experiment of how to do that is rather than training vanilla FLAC using the original panoroma depth image, before putting this panoroma depth image into the FLAC model, first find the deepest pixel and always put it on the most left side (horizontally cyclic shift), when evaluation on the unseen rooms, do this kind of horizontally cyclic shift as well. This is what is called as canonicalization everytime before the model input to make the output rir invariant. And the invariant test would be shift the panoroma depth image horizontally cyclicly, and since we have this kind of canonicalization, even if the test cases would be horizontally shifted, the model will still see the same depth image since we will preprocess the shifted panoro depth image (deepest pixel to the most left-hand side) so by this means, the model will achieve the invariance. Please do this experiment.

**Mid-turn follow-up (verbatim):**

> Previously, we have the C8, C16, C32 experiment run on the neuronic cluster, where was the experiment logging for that?

**Summary.** (1) Locate the code and records that produced the three rows of the paper table (vanilla FLAC, FA FLAC = \method = OrbitRIR, Yaw-Aug FLAC; 2×A6000, 40k steps, 5 eval seeds). (2) Design and run a new baseline arm for OrbitRIR: *input canonicalization* — before every model input (training AND unseen-room evaluation) cyclically shift the depth panorama horizontally so that the deepest pixel's column becomes the leftmost column; co-rotate the poses so the scene stays physically consistent. (3) Invariance test: horizontally cyclic-shift the evaluation panoramas; because the canonicalization is applied after the shift, the model should see the same canonical input and produce the same RIR. (4) Locate the neuronic C8/C16/C32 records.

**Assumptions surfaced to Yixun (plan §11).** The arm follows the P1 / Yaw-Aug 2×A6000 recipe exactly (single delta vs P1) so it slots into the same table; the "deepest pixel" is the global maximum of the per-pixel depth, with an equivariant tie-break rule because exact ties at the maximum occur in ≈5 % of AR panoramas (plan §2.1, D1); the canonical shift co-rotates the pose vectors (otherwise the rolled panorama and the poses would describe different scenes).

**Hypothesis (Yixun's).** Canonicalization makes the output RIR invariant to horizontal cyclic shifts of the input panorama by construction, at zero inference cost — a baseline that OrbitRIR (frame averaging) must be judged against.

**Why this experiment needs to run.** The table has an architectural invariance method (FA / OrbitRIR) and a data-side method (Yaw-Aug) but no *preprocessing* invariance baseline. Without it, FA's gains over vanilla cannot be attributed between "invariance per se" and "orbit averaging".
