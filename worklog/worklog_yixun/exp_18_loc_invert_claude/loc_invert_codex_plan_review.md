**Reviewer:** OpenAI Codex `gpt-5.6-sol` at Extra High reasoning (codex-cli 0.147.0, `codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort=xhigh`, model confirmed served from run log) · **Date:** 2026-08-18

*Planner note: the body below is Codex's verbatim output. Its self-written first line ("GPT-5, API workspace agent") under-specifies the invocation; the header above is the authoritative identity per SOP. Reviewed artifact: `plan_loc_invert.md` at commit `4f3658e` (Rev 1).*

---

**Reviewer:** OpenAI Codex (GPT-5, API workspace agent, read-only review) · **Date:** 2026-08-18

**Verdict: REQUEST-CHANGES**

1. **BLOCKER — Target exclusion reveals strong candidate information that the proposed random baseline ignores.**  
   The context contains eight sources explicitly chosen from sources other than the target ([AR_md.py](/home/yixunhu/codespace/FLAC/src/configs/dataset_configs/custom_metadata/AR_md.py:90)). Since their coordinates are visible in `context_poses`, the target is guaranteed to lie in `C \ unique(context sources)`. With roughly ten candidates, this commonly reduces the plausible set to about two; a context-aware random guess can therefore approach 50% top-1 rather than the proposed 10%. In a nine-source case it could reveal the target outright. Excluding the target remains required, but success against uniform-over-C random would not establish source information in FLAC.  
   **Fix:** retain uniform-over-C as the literal baseline, but add an exact context-conditioned baseline uniform over candidates absent from `context_poses`; compute all localization metrics under it, report the eligible-set-size distribution, mark every candidate’s context membership, and report how often FLAC predicts a context source. The scientific success criterion must require improvement over this information-matched baseline.

2. **BLOCKER — Silent substitutions cannot be “logged + excluded” in a headline full-split run.**  
   The plan proposes excluding substituted rows ([plan §2.1](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_18_loc_invert_claude/plan_loc_invert.md:15)), but that silently converts the mandated 6,337-item evaluation into a reduced, potentially selected split. `SampleDataset` really does recursively return another random item on silence or any exception ([dataset.py](/home/yixunhu/codespace/FLAC/src/data/dataset.py:268)); existing `eval_FLAC.verify_stream_positions` correctly treats this as fatal ([eval_FLAC.py](/home/yixunhu/codespace/FLAC/eval_FLAC.py:617)).  
   **Fix:** before GPU generation, hard-fail on the first position/index/relpath mismatch. A headline artifact may be written only after proving exactly 6,337 expected identities, no duplicates or omissions, and 17 room IDs. Record the split hash. Smoke runs may diagnose substitutions, but full runs must be repaired and rerun.

3. **HIGH — The primary estimand is internally inconsistent.**  
   The plan names median localization error as primary, then says to take per-room means and average them. That operation is neither a median nor the requested across-query median. The prior grounding also says AR’s reproduced release protocol used all-sample aggregation; the per-scene note applies to HAA ([exp_01 results](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_01_reproduce_flac_table1_claude/reproduce_flac_table1_results.md)). Additionally, `AR_md` stores only the scene type in `md["scene"]`, producing 10 labels rather than the 17 physical rooms.  
   **Fix:** define the primary as the pooled median over the 6,337 query errors, matching the user’s specification. Construct a physical `room_id = scene_name/scene_id`; report clearly labelled pooled and equal-room macro statistics, including “mean of per-room medians” if desired. Apply identical weighting to both random baselines. Report metrics per seed, seed mean±SD separately from a 17-room clustered-bootstrap CI; three-seed SD must not be presented as a confidence interval.

4. **HIGH — Dev selection can replace the user-specified LME/K=8 method.**  
   The detailed query and summary specify log-mean-exp aggregation, while the plan lets mean, max, or `K′<8` become the registered primary ([plan §2.5](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_18_loc_invert_claude/plan_loc_invert.md:31)). That changes the experiment rather than tuning its stated hyperparameter.  
   **Fix:** keep LME with K=8 as the registered primary and select only τ on the full seen split—or pre-register τ directly. Treat mean, max, and `K′∈{1,2,4}` solely as labelled sensitivity analyses. Define a deterministic dev tie-break before running.

5. **HIGH — Announcement 05’s protocol declaration is incomplete.**  
   The proposed CLI omits `--rotate-deg`, and the plan never pins concrete values for `--cond-autocast` or the frame-angle field, despite claiming full compliance.  
   **Fix:** include and record at minimum `--cond-method vanilla`, `--rotate-deg 0`, `--cond-autocast default`, and an explicit ignored/not-applicable frame-angle value in the plan, params, every smoke/full command, and every output row. Either implement rotation using the existing metadata rotation path or fail closed on any nonzero value.

6. **HIGH — The declared AGREE preprocessing does not match the code it cites, and freezing is underspecified.**  
   `Retrieval.compute_audio_features` pads short inputs but does not crop them ([Retrieval.py](/home/yixunhu/codespace/FLAC/src/metrics/modules/Retrieval.py:44)). Moreover, the AR metric callback first truncates inputs to 8,000 samples and only then passes them to Retrieval, which pads them back to 10,240 ([metric_callback.py](/home/yixunhu/codespace/FLAC/src/metrics/metric_callback.py:272)). The planned “pad/crop to 10,240 exactly as Retrieval” is therefore false. The reused loader also does not itself call `.eval()` or freeze parameters.  
   **Fix:** explicitly choose the scorer protocol. For parity with established AR retrieval, use first-8,000 then zero-pad to 10,240 and test embedding equality against the actual Retrieval path. If full 10,240 is intentional, label it as a new protocol. In either case call `.to(device).eval().requires_grad_(False)`, use inference mode, test batch-size invariance, and record the AGREE checkpoint SHA-256.

7. **HIGH — Candidate membership is inferred from RIR filenames rather than defined from metadata.**  
   The plan obtains node IDs from the room’s audio directory and only then reads coordinates. That assumes the RIR files enumerate exactly the valid metadata source set—an assumption that cannot yet be checked because the dataset is absent. It also complicates the oracle in rooms with missing source/receiver pairs.  
   **Fix:** enumerate unique valid source IDs and `src_loc` values from the room metadata, as specified by the user; assert coordinate consistency, uniqueness, and GT membership. Cross-check the metadata and RIR node sets during readback and fail on unexplained differences. Missing measured RIRs may reduce only the diagnostic oracle’s explicitly reported eligibility set; they must never change FLAC’s candidate set or headline denominator.

8. **HIGH — Numerical parity and TDD coverage are insufficient for a second inference driver.**  
   `evaluate_model`’s loading and generation path is monolithic, so “reuse its loading path” is not presently an actionable reuse boundary. A mocked conditioning-threading test will not catch EMA, objective, conditioner tiling, decode, or candidate/sample-order divergence. The driver and heatmap sections also do not enumerate their new functions and test contracts as required by announcement 02.  
   **Fix:** either extract a reviewed shared inference helper or specify the exact imported/copied boundary and add a one-query numerical parity test against `eval_FLAC` with identical checkpoint, metadata, noise, and vanilla flags. Fail closed on unsupported checkpoint/objective modes. Add named contracts/tests for driver helpers, candidate×sample layout, serialization, case selection, and plotting, followed by the SOP’s per-round and full integrative reviews.

9. **HIGH — The compute estimate and fit-probe waiver are unsupported.**  
   The statement `M×K ≤ 80 ≪ batch 64` is mathematically wrong: 80 exceeds 64 and requires two batches. Relative to exp_01’s 100 batches, this design generates 80× as many samples and, with per-query tail batches, can require up to roughly 127× as many sampler launches. Direct scaling of the stated 6.5-minute basis gives roughly 8.7–13.7 hours per sweep before accounting for which exp_01 costs dominate; nothing yet supports 1–4 hours. The script also has no declared two-GPU execution strategy.  
   **Fix:** make R0 include a maximum-M, K=8 batch-64 fit probe with peak-memory and component timing. Update the budget from that measurement. Declare whether the two GPUs run seeds concurrently or use deterministic full-split shards; if sharded, implement a merge gate proving disjoint union equals all 6,337 identities. Do not waive validation rung 6.

10. **MEDIUM — Randomness is not stable to candidate ordering, batching, sharding, or resume.**  
    “Fresh noise each” tied only to global RNG means a candidate can receive different samples if ordering or batch boundaries change. That makes candidate ranking and offline K′ comparisons unnecessarily execution-dependent.  
    **Fix:** register a deterministic noise mapping, preferably one K-noise bank keyed by `(seed, query_id, k)` and reused across candidates as common random numbers, or independent noise additionally keyed by candidate ID. Add permutation and one-batch-versus-two-batch equivalence tests. Resume must reproduce the recorded context fingerprint and noise keys.

11. **NIT — The visualization and smoke descriptions overclaim their contents.**  
    Candidate/receiver extent is not a room boundary, display temperature T is unspecified, and with `shuffle=False`, the first four queries do not necessarily span the advertised two rooms.  
    **Fix:** derive a boundary from depth/metadata or label the plot honestly as candidate extent; pre-register T and the seed used for case selection. Select explicit smoke identities from two rooms or change the run-matrix description.

### What the plan gets right

The plan correctly uses the complete existing unseen split, preserves one receiver/context across every candidate, keeps the candidate search three-dimensional and discrete, anticipates the `S010` naming hazard, and uses a train-split-only AGREE checkpoint for the clean claim. It also correctly recognizes that the same-pair GT-RIR oracle is an identity sanity check rather than a meaningful upper bound. Logging every per-sample similarity, quarantining smoke artifacts, selecting display cases by rules rather than hand, and gating execution on data/checkpoint arrival are all sound foundations.