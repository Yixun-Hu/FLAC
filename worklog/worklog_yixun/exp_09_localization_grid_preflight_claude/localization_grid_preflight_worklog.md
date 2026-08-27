# Lab notebook — exp_09_localization_grid_preflight

## 2026-08-20T11:56:06-04:00 — scaffold and pre-plan audit

- **Goal** — scaffold the mesh-valid three-dimensional candidate-grid preflight for frozen Vanilla FLAC localization, under the experiment SOP and entirely inside `NeuriPs_Workshop` for program, worktree, logs, and results.
- **Hypothesis** — a deterministic `0.5 m` free-space grid can cover continuous held-out source locations closely enough for the `0.5 m` success metric without inserting ground truth.
- **Version Control** — branch `localization-exp`; `base_commit=ecb83523c4ae8c60d4cd5f0ae3e562f2a84f1fa9`; worktree `/home/zhixuanzhao/projects/Frame_Average/NeuriPs_Workshop/localization-exp`; source worktree had unrelated uncommitted changes and was not modified.
- **Command / Validation** — read `worklog/experiment_SOP.md` and all three `worklog/worklog_yixun/announcement/*.md`; inspected the full unseen split, AR metadata loader, geometry conditioner, evaluation sampler, AGREE acoustic encoder, exp_07/08 plans/results/analyses, checkpoint inventory, official AcousticRooms repository at `3c87318a0188e1b441fc75846d54b487ca215fbb`, and local Open3D availability.
- **Result** — `partial`. The existing full split is 6,337 queries in 17 rooms; 16/17 unseen rooms have an official OBJ in the checked official repository. `ListeningRoom_idx_2` is in the full unseen split but the official `room_mesh_obj_format/ListeningRoom/` contains only `idx_0` and `idx_1`. The README says all rooms have meshes, so this is an upstream data inconsistency, not permission to drop the room. System Python has Open3D 0.19.0; no new mesh dependency is required. The Claude executable is available through the installed VS Code extension at version 2.1.237, so the mandated cross-family plan review can run.
- **Analysis** — deleting `ListeningRoom_idx_2` would violate the full-split announcement. The plan therefore separates a fail-closed mesh audit from the full headline launch and proposes a clearly labeled, separately validated depth-panorama fallback only if the missing mesh cannot be supplied. The fallback may not enter a headline run without explicit approval in the reviewed plan.
- **Next** — write the per-file/TDD plan, run Claude Opus at max effort for the independent plan review, revise all blocking findings, then surface the reviewed plan and the missing-mesh decision to Yixun before implementation.

## 2026-08-20T11:58:00-04:00 — checkpoint and protocol input identity

- **Goal** — pin the read-only model inputs before planning code paths.
- **Command / Validation** — `sha256sum` over the clean 40k model checkpoints, AGREE checkpoint, and VAE weights.
- **Result** — `passed`: Vanilla clean EMA `P1_40k_clean_hybrid_EMA.ckpt` = `da12748586912c5fe9683a6d27b2507ff13c0a89c458abcbdc63aecd4f35c643`; FA-BF clean EMA `BF_40k_clean_hybrid_EMA.ckpt` = `0f61277f45367fb0e75d7ee70c0627b8948a23eb62be58f13fce91662551557a`; AGREE `AGREE_fullAR.pt` = `3a13243d6c6a11082697592c2c5db84790d37859451df2963eb51d655b23c787`; VAE `VAE.safetensors` = `8d82159eec35210198246f449bec6561fc19b514922f340a17515050daf7f0b9`.
- **Analysis** — exp_09's first localization arm is Vanilla, but the generic inference surface will retain the already-reviewed `vanilla` / `fa_invariant` conditioner dispatch so the later FA-BF comparison uses exactly the same candidate/query/score implementation.
- **Next** — plan review.

## 2026-08-20T12:00:50-04:00 — mandatory Claude plan-review attempt blocked by authentication

- **Goal** — obtain the SOP-mandated independent cross-family plan review with the strongest Claude Opus model at max effort before implementation or user approval.
- **Command / Validation** — invoked the installed native CLI `/home/zhixuanzhao/.vscode-server/extensions/anthropic.claude-code-2.1.237-linux-x64/resources/native-binary/claude --print --model opus --effort max --permission-mode plan --tools Read,Grep,Glob,Bash --no-session-persistence --max-turns 30 --output-format text ...`; then ran `claude auth status`.
- **Acceptance criteria** — a real Claude response that begins with the required exact reviewer identity, supplies a verdict, and reviews the fully briefed plan without editing files.
- **Result** — `blocked`: the review invocation exited 1 with `Failed to authenticate: OAuth session expired and could not be refreshed`; `auth status` returned `{"loggedIn": false, "authMethod": "none", "apiProvider": "firstParty"}`. No reviewer model was reached and no review verdict exists. The failed attempt is recorded in `localization_grid_preflight_opus_plan_review.md` and must not be treated as approval.
- **Analysis** — this is an authentication/infrastructure blocker, not a plan or implementation defect. SOP reviewer reciprocity forbids substituting Codex for Claude when the main Planner is Codex. Implementation remains unopened.
- **Next** — Yixun reauthenticates Claude Code (the VS Code Claude extension or the native CLI), then rerun the same read-only Opus/max plan review, revise all findings, and surface the reviewed plan for explicit approval.

## 2026-08-20T12:01:46-04:00 — planning state committed

- **Goal** — preserve a clean, auditable planning boundary while the external review gate waits for authentication.
- **Version Control** — branch `localization-exp`; `base_commit=ecb83523c4ae8c60d4cd5f0ae3e562f2a84f1fa9`; planning/scaffold commit `fe220ade27afeb5dad7331d19a56c660ac47ab57`; changed files are the five exp_09 query/plan/worklog/review-brief/review-attempt Markdown artifacts only; no source or test code exists yet.
- **Command / Validation** — `git diff --check`; `git commit -m "exp_09: scaffold 3d localization grid plan"`.
- **Result** — `passed`; commit created and the branch is one commit ahead of `origin/check-equivariance-necessity` before this bookkeeping commit.
- **Analysis** — the source tree remains byte-identical to the base; stopping here obeys both the SOP user-approval gate and the cross-family review rule.
- **Next** — authenticate and rerun the Opus review.

## 2026-08-20T12:08:02-04:00 — Yixun resolves the missing-mesh scope

- **Goal** — resolve the known `ListeningRoom_idx_2` geometry gap without inventing an unapproved reconstruction backend.
- **Trigger** — Yixun: “缺失mesh就先注明但不纳入测试”.
- **Command / Validation** — parsed the existing `data/AR/unseen_eval.json` without writing a new eval config: full manifest = 6,337 queries / 17 rooms; `ListeningRoom_idx_2` = 1,000 queries; exact remaining mesh-available scope = 5,337 queries / 16 rooms.
- **Change** — query, plan, and independent-review brief now exclude exactly `ListeningRoom_idx_2`; delete the proposed depth fallback from exp_09; require the runner/report to record the excluded room and 1,000-query count. No source code changed.
- **Result** — `passed` as a scope decision. This is an explicit experiment-specific exception to the full-split standing directive, not permission for further subsampling.
- **Analysis** — exp_09 can answer the preflight question on the mesh-available subset, but its results cannot be presented as the complete 6,337-query unseen-room protocol. A future full-protocol model comparison still requires the missing authoritative mesh or a separately approved common geometry rule.
- **Next** — rerun the mandatory Opus plan review after Claude authentication is restored, then surface the revised plan for approval.

## 2026-08-20T13:05:00-04:00 — independent Claude plan review delivered (REQUEST-CHANGES)

- **Goal** — close the SOP-mandated cross-family plan-review gate that the 12:00:50 entry recorded as authentication-blocked.
- **Change** — appended the actual review below the preserved failure record in `localization_grid_preflight_opus_plan_review.md`; the "UNAVAILABLE" provenance block is untouched. No source, test, plan or config file modified.
- **Command / Validation** — reviewer was Claude Opus 5 (`claude-opus-5`) in an interactive Claude Code 2.1.237 session at Yixun's direct request, not the planned `--print --model opus --effort max` native-CLI invocation; the deviation is stated in the review header. Read-only verification run against the real assets: full-split context-availability census; per-room inter-source distance census; grid-oracle sweep over context-clearance thresholds {0, 0.25, 0.5, 1.0} m (first 120 queries/room, weighted to 5,337); AABB lattice-size census from the official OBJs; source/receiver height census; source-to-AABB clearance census; `python -m pytest src/tests -q`.
- **Result** — `passed` as a gate action; verdict `REQUEST-CHANGES` with 6 blocking findings, 8 recommendations, 3 SOP bookkeeping items.
  - B1 — the 1.0 m context-source exclusion makes **21.4 %** of the 5,337 queries unwinnable at 0.5 m (100 % in `Bathrooms_idx_18`); its premise is false, measured minimum inter-source distance is **0.20 m** and 10/16 rooms are under 1.0 m. At 0.25 m clearance the damage is 0.0 %.
  - B2 — **520 queries (8.2 %, all `Cafe_idx_1`)** have fewer than 8 same-receiver contexts (histogram `{6:91, 7:429, 8:5263, 9:554}`); plan's fail-closed "eight of nine" rule would drop them. Separately, `AR_md.py`'s `f"S00{node}"` path construction permanently excludes source `S010` from every context pool in the released eval path.
  - B3 — **25,312,262** raw candidate evaluations before masking (`Cafe_idx_1` + `Auditorium_idx_1` = 91 %); `GeometryConditioner` runs **9 DINOv3 ViT-S/16 forwards per candidate**, 8 of them candidate-invariant and cacheable; no pre-registered compute fallback exists while §5 forbids post-hoc grid changes.
  - B4 — `src/tests/test_eval_paths.py` is uncollectable (stale pre-announcement-03 path `worklog/exp_02_...`); **90 pass, 1 module errors**, so ladder rung 2 is red today.
  - B5 — AGREE input parity underspecified; `Retrieval.compute_audio_features` is the existing reviewed reference (pad to 10240 + `encode_audio(normalize=True)`), and `eval_FLAC.py` clamps generated audio.
  - B6 — base mesh-clearance mask lacks the eps tolerance stated for the query masks, at a boundary that is exactly critical (real sources sit at exactly 0.50 m from the AABB in three rooms).
  - Verified-correct: `ListeningRoom_idx_2.obj` genuinely absent while the other 16 rooms have meshes; 6,337 − 1,000 = 5,337 exact; Eq. (3) ↔ plan's `tau*(logsumexp(s/tau) − log K)` identical; `q = c − x_r` matches `get_3d_point_camera_coord`; `steps=1, cfg=1.0` matches exp_01/02 artifacts.
- **Analysis** — infrastructure, not a defect: the earlier block was authentication, and the plan itself is methodologically sound. B1–B3 are the substantive ones — each would have silently corrupted or stalled the headline after expensive compute, and each was invisible from the plan text alone; they surfaced only by measuring the real split, metadata and meshes. B4/B6 are mechanical. Nothing found requires a different method.
- **Next** — Planner revises against the 8-item approval checklist; B1/B2/B3 need Yixun's decision because they change what is measured. Implementation stays closed until the revised plan is approved.

## 2026-08-20T13:45:00-04:00 — review artifact amended (duplicate copy removed, N9 added)

- **Goal** — correct a bookkeeping error in the review artifact written at 13:05 and fold in one finding that copy had missed.
- **Change** — `localization_grid_preflight_opus_plan_review.md`: a second, redundant copy of the same REQUEST-CHANGES review had been appended below the 13:05 review; it was removed so the file holds exactly one review under the preserved authentication-failure record. The one finding unique to the removed copy was ported in as **N9** and added to checklist item 8.
- **Command / Validation** — `head -n 227` truncation + in-place insert; file now 229 lines, one `# Plan review … (actual review)` heading, N1–N9 present. No change to any verdict, blocking finding, or measured number.
- **Result** — `passed`. Verdict remains **REQUEST-CHANGES**; blocking set is unchanged at B1–B6; non-blocking set is now N1–N9.
- **Analysis** — N9: the 11:56 entry's dependency conclusion ("System Python has Open3D 0.19.0; no new mesh dependency is required") is true of `/usr/bin/python3` but false of the FLAC-vanilla venv that is `python` on `PATH` and runs `src/tests/`, where `import open3d` raises `ModuleNotFoundError`. G1's geometry code and its tests run in that venv, so Open3D must be installed there and pinned in `pyproject.toml` before the geometry round opens.
- **Next** — unchanged: Planner revises against the approval checklist; B1/B2/B3 need Yixun's decision because they change what is measured. Review artifact and this notebook are uncommitted in the working tree.

## 2026-08-20T22:18:01-04:00 — Planner closes review findings in the revised plan

- **Goal** — address every Opus 5 `REQUEST-CHANGES` item before asking Yixun to authorize implementation.
- **Change** — revised `plan_localization_grid_preflight.md` only; preserved the reviewer-authored review and prior append-only notebook entries. B1: 1.0 m exclusion replaced by 0.25 m duplicate guard plus query nonempty/oracle gates. B2: selected parity option (a), preserving the S010 eligible-pool quirk and deterministic replacement sampling for all 520 short Cafe queries. B3: query/receiver-candidate caches, conditional context-derived z-band, post-G1 user cost gate, and global `K=4 -> 2 -> 1`/168-GPU-hour ladder. B4/N9: added R0 for the stale test path, full green baseline, and Open3D pin/runtime install. B5: pinned shared Retrieval preprocessing and clamp/equality audit. B6: fixed `eps=1e-4 m` and real-source-anchor survival. N1–N8: added the 3-D/PDF precedence, canonical-only claim boundary, mean-score diagnostic, fixed 16-query off-grid/calibration probes, raw/oracle/excess co-primary metrics, and no CFG sweep.
- **Version Control** — branch `localization-exp`; base `ecb83523c4ae8c60d4cd5f0ae3e562f2a84f1fa9`; prior planning commits `fe220ade27afeb5dad7331d19a56c660ac47ab57`, `3760c869b94119fa83cd28223f8f8da8422f3d41`, `35373a50f1d903f19b1b97947975e1de8dfe2267`; current review/plan revision pending commit. No source, test, config, dependency, or environment mutation has occurred.
- **Command / Validation** — read the actual review completely; `git diff --check`; searched the revised plan for stale 1.0 m/unique-eight/no-fallback statements and corrected them.
- **Result** — `fix_ready`; all B1–B6 and N1–N9 map to explicit plan text and tests. The revised approval authorizes only R0→D1→G1; the workflow stops again for the exact post-G1 geometry/cost gate before I1.
- **Analysis** — the recommended choices preserve baseline context parity and the agreed 0.5 m 3-D lattice while removing measured oracle damage and denominator loss. Expensive generation is still not authorized. The 168-GPU-hour threshold is a pre-quality resource ceiling; if even cached K=1 exceeds it, the experiment blocks for resources instead of silently changing candidates.
- **Next** — commit the review/revised plan bookkeeping and ask Yixun to approve the bounded R0→D1→G1 phase.

## 2026-08-20T22:19:40-04:00 — reviewed-plan approval boundary committed

- **Goal** — freeze the exact plan presented for Yixun's approval.
- **Version Control** — review/revision commit `c5b7b5a9a28c23550bf8f63b9325f3965ce1a456`; changed only the Opus review artifact, revised plan, append-only worklog, and commits ledger. No implementation code, tests, config, dependency, or environment changed.
- **Command / Validation** — `git diff --check`; `git commit -m "exp_09: revise plan after Opus review"`.
- **Result** — `passed`; branch `localization-exp` reached a clean four-commit-ahead approval boundary before the final bookkeeping commit.
- **Analysis** — the `REQUEST-CHANGES` review is preserved verbatim and the plan contains an explicit response for every B1–B6/N1–N9 item.
- **Next** — Yixun approves or amends the bounded R0→D1→G1 phase; only then may implementation begin.

## 2026-08-20 — Yixun approves R0→D1→G1 with original global-RNG context amendment

- **Goal** — resolve the remaining context-selection choice and open the bounded geometry/cost phase.
- **Decision** — replace query-local hashed context selection with the original exp_01 K=8 loader protocol: seed 42, batch 64, four workers, no shuffle, full 6,337-query split, and the released `AR_md.py` global/per-worker NumPy choice path. Materialize and hash the context manifest before excluding the 1,000-query missing-mesh room; all candidates and model arms reuse it.
- **Rationale** — this preserves the actual FLAC/FA-FLAC input path, including the S010 quirk and with-replacement fallback, while a frozen manifest still guarantees candidate/arm pairing. Filtering before materialization is prohibited because it would change worker RNG consumption.
- **Authorization** — Yixun directed: “改用原版全局随机抽样，其他按照md文档进行修改，修改以后分析GPU成本.” This opens R0→D1→G1 and the post-G1 cost analysis only. Large-scale generation remains blocked on the resulting cost gate.
- **Next** — commit the protocol amendment, then begin R0 test-first.

## 2026-08-20T22:52:14-04:00 — R0 permanent-suite/runtime baseline

- **Goal** — restore the permanent test suite before localization code and put Open3D in the interpreter that actually runs FLAC tests.
- **Red evidence** — `/home/zhixuanzhao/projects/Frame_Average/FLAC-vanilla/.venv/bin/python -m pytest src/tests -q` failed collection at `src/tests/test_eval_paths.py` with `ModuleNotFoundError: compare_predictions`. The same interpreter reported Open3D missing while holding torch 2.7.0+cu126, torchaudio 2.7.0+cu126, PyTorch Lightning 2.1.0, NumPy 1.23.5, and pytest 9.1.1.
- **Change** — `test_eval_paths.py` now walks to a `.git` file-or-directory marker and resolves `worklog/worklog_yixun/exp_02_yaw_noninvariance_claude`; `pyproject.toml` pins `open3d==0.19.0`. Installed that exact version into the recorded FLAC venv while every command remained launched from the NeuriPs_Workshop worktree.
- **Green evidence** — complete permanent suite: `120 passed, 1 skipped, 11 warnings in 30.04s`; explicit runtime readback: interpreter `/home/zhixuanzhao/projects/Frame_Average/FLAC-vanilla/.venv/bin/python`, Open3D `0.19.0`; `git diff --check` passed.
- **Independent-review attempt** — authenticated Claude Code 2.1.238 native CLI, model alias `opus`, read-only/plan permission. Two max-effort requests (full and narrowed R0 diff) and one low-effort minimal retry each ended with `Request timed out` without a verdict or file mutation. This is preserved as reviewer infrastructure failure, not represented as review approval.
- **Result** — implementation/tests are green; independent R0 review remains pending because the reviewer service returned no result. Continue bounded TDD work without launching generation, and retry/consolidate Opus review before interpreting or launching I1.
- **Next** — commit R0, then begin D1 test-first global-RNG manifest construction.

## 2026-08-20 — D1 red: query/context manifest contracts

- **Goal** — freeze the observable contracts before implementing global-RNG context materialization.
- **Tests added** — released split order; S010 path quirk; full-before-filter manifest guard; protocol/hash round-trip; exact real-data eligible-count histograms; target-pose-only candidate cloning.
- **Red evidence** — `python -m pytest src/tests/test_localization_ar_queries.py -q` fails collection with `ModuleNotFoundError: src.localization`, as expected before the new package exists.
- **Next** — commit the red tests, then implement the smallest query/manifest layer that makes them green.

## 2026-08-20 — D1 green: original exp_01 global-RNG context manifest

- **Change** — added immutable split/query parsing and hashed manifest contracts; refactored the released `AR_md.py` selection block into `select_other_source_ir_paths` without changing its set iteration, S010 filename construction, broad exception fallback, or `np.random.choice` calls; added an opt-in `record_paths` output that is absent from ordinary configs. Added a full-loader materializer and CLI under `src/localization/` and `tools/`.
- **Protocol executed** — existing K=8 unseen-eval config; complete 6,337-query order; seed 42; batch 64; four persistent workers; no shuffle; original audio/metadata/depth loader. The sandboxed attempt failed only because local multiprocessing sockets were forbidden; the identical approved command then completed from this worktree.
- **Manifest** — `context_manifest_exp01_seed42.json`, 6,337 records, SHA-256 `b757da281dcde3ffc310aac67279a240dac5cb1ff1d9966bf918f69c4dde6f58`. A second independent full pass produced the same SHA.
- **Census** — full eligible histogram `{6:91,7:429,8:5263,9:554}`; post-materialization mesh subset 5,337 records with `{6:91,7:429,8:4363,9:454}`; exactly 1,000 `ListeningRoom_idx_2` exclusions; all 520 short queries are in `Cafe_idx_1`, have width eight, and necessarily contain duplicate draws. Context paths, order, global coordinates, receiver coordinates, split hash, and protocol are frozen.
- **Validation** — localization D1 tests `5 passed`; complete permanent suite `125 passed, 1 skipped, 11 warnings in 30.58s`; `py_compile` and `git diff --check` passed.
- **Review cadence** — per Yixun's instruction to reduce reviewer-md time, no additional per-microcommit reviewer call is made. D1/G1 code, tests, audit, and cost evidence will be presented as one concentrated review unit.
- **Next** — commit D1 green, then begin G1 geometry tests and mesh audit.

## 2026-08-20 — G1 red: geometry contracts

- **Tests added** — invalid spacing; negative-coordinate lattice snapping; exact lexicographic 3-D lattice; synthetic watertight-box occupancy/surface clearance and chunk identity; inclusive receiver/context/z boundaries with `eps=1e-4`; finite nonempty oracle; global z-band fallback; missing/malformed mesh failure.
- **Red evidence** — `python -m pytest src/tests/test_localization_geometry.py -q` fails collection with `ModuleNotFoundError: src.localization.geometry`, as expected.
- **Next** — commit red, implement the bounded Open3D primitives, then run the synthetic suite before touching official meshes.

## 2026-08-20 — G1 implementation and diagnostic cost gate

- **Change** — implemented deterministic 0.5 m global lattice snapping, Open3D occupancy/unsigned-distance masking, receiver/context/z-band query masks, finite grid oracle, and the global z-band selection rule in `src/localization/geometry.py`; added the official-mesh audit CLI. No FLAC/FA-FLAC generation ran.
- **Validation** — geometry suite `9 passed`; complete permanent suite `134 passed, 1 skipped, 11 warnings in 31.31s`. Lattice order/boundaries, watertight-box classification, chunk identity, oracle gates, and fail-closed mesh loading are pinned.
- **Real diagnostic audit** — full approved 5,337-query/16-room subset, frozen context SHA `b757da281dcde3ffc310aac67279a240dac5cb1ff1d9966bf918f69c4dde6f58`; missing `ListeningRoom_idx_2` remains exactly 1,000 excluded queries. Fast cost mode deferred expensive mesh-topology diagnostics but did not alter any candidate mask. Audit SHA `46c087b243d939010d2796274ea2ce553b147777b0065de346e62141c3cb67e9`.
- **Counts** — z-band passes the pre-registered no-new-unwinnable decision rule and reduces full-height `10,465,069` pairs to `6,094,936`; all 5,337 candidate sets are nonempty/finite; chosen oracle mean/median/max `0.2477/0.2408/0.8684 m`, with 160 queries above 0.5 m. Cache counts are 636,963 unique receiver-candidate branches and 42,696 context ViT forwards.
- **Blocking finding** — geometry gate **FAIL**: 13 unique metadata source anchors across 7 rooms fail the declared inside-and-0.5 m-surface-clearance predicate; 8 unique receiver anchors across 5 rooms fail mesh occupancy. This falsifies the plan-review anchor assumption. The threshold was not relaxed and generation remains prohibited.
- **GPU analysis** — exp_01's five full K-context=8 Vanilla runs give a median 7.193 generated RIR/s. At 6,094,936 pairs this is 235.4/470.7/941.5 GPU-hours for score `K=1/2/4`; even an optimistic 10 RIR/s gives 169.3 hours at K=1. The cache-enabled K=1 engine must exceed 10.078 RIR/s to satisfy 168 GPU-hours, but its exact probe cannot open before the geometry protocol is resolved and I1 is authorized. Full arithmetic and storage bounds are in `gpu_cost_analysis.md`.
- **Review cadence** — per Yixun's request, no additional reviewer-md cycle was inserted before producing these decision-critical counts.
- **Result / Next** — G1 primitives are green, but the real-data geometry gate is red. Stop for Yixun's geometry-protocol decision; do not open I1 or large generation.

## 2026-08-20 — G1 diagnostic commit

- **Version Control** — committed geometry primitives, audit tool, exact JSON/Markdown audit, GPU-cost analysis, and notebook evidence as `30c3bd32f3c12891521dcfc0a2dff7d4134b92fb` (`exp_09 G1: audit geometry and GPU cost`).
- **Boundary** — this is a diagnostic completion, not a passed real-geometry gate and not authorization for I1/generation.

## 2026-08-20T14:20:00-04:00 — B7 added: geometry backend unusable, 0.5 m clearance mis-derived

- **Goal** — answer Yixun's question "基础格点为什么一定要满足到最近 mesh 表面的距离至少 0.5 m？" by measuring the rule instead of reasoning about it.
- **Hypothesis** — the 0.5 m surface clearance is a distribution-matching prior rather than a validity constraint, and its stated justification was verified only against the room AABB, not the mesh.
- **Command / Validation** — Open3D 0.19.0 under `/usr/bin/python3`: `RaycastingScene.compute_distance` / `compute_occupancy` over all 16 included OBJs, all real source/receiver metadata anchors, and the 0.5 m AABB lattice; `e_oracle` and candidate counts recomputed per room on the first 120 queries per room with receiver 0.5 m + context 0.25 m applied.
- **Result** — `passed` (finding confirmed). **All 16/16 meshes are non-watertight and non-edge-manifold**, and `compute_occupancy` labels **~100 % of real source and receiver anchors "inside a solid"** — the plan's `occupancy == 0` free-space test selects the space *outside* the room, and inverting it still misclassifies 10 % of real anchors in `Apartments_idx_50` and 4-8 % in four other rooms. Separately, the global minimum real source-to-**mesh-surface** distance is **0.232 m** (`Restaurants_idx_22`), with **7/16 rooms** holding a source closer than 0.5 m to a surface; the AABB-based minimum is exactly 0.50 m, which is why B6 read the rule as merely boundary-critical. Oracle cost of the clearance, corrected masks applied: 0.20 m → 0.0 % unwinnable / 17.9 M candidates; 0.25 m → 0.0 % / 17.1 M; 0.30 m → 0.6 % / 15.1 M; **0.50 m → 3.3 % / 12.0 M**.
- **Change** — `localization_grid_preflight_opus_plan_review.md`: added **B7** (blocking), marked B6 absorbed into it, renumbered the approval checklist to 9 items, and revised B3's compute figure from the ">= 25.3 M pre-masking" AABB bound to the measured **~17.1 M candidates / ~68.6 M generations at K=4**. Verdict unchanged: REQUEST-CHANGES, now 7 blocking + 9 non-blocking.
- **Analysis** — the mask conflates physical validity (needs occupancy/ray-parity, currently broken) with an in-distribution prior (needs a measured threshold, currently asserted). With occupancy unusable the distance threshold silently carries both jobs, which is what made 0.5 m load-bearing and wrong. The plan's own fail-closed anchor criterion #2 would have caught this at G1 — and would have failed all 16 rooms with no fallback, so it is better caught here.
- **Next** — Planner: replace the backend with multi-direction ray-parity majority voting validated against every real anchor, split validity from clearance, re-derive the clearance to 0.20 m, and re-run B1's `e_oracle` gate under the corrected mask. Still uncommitted.

## 2026-08-20 — Yixun approves 0.20 m; B7 geometry gate closes

- **Decision** — Yixun directed: “把基础格点为到最近 mesh 表面的距离改为0.20 m吧，这样就不会有问题了，然后再来讨论计算成本的问题.” The source-surface prior is now 0.20 m; grid spacing remains 0.5 m, receiver exclusion remains 0.5 m, and context duplicate guard remains 0.25 m.
- **Preserved reviewer input** — the newly written B7 review/worklog amendment is retained. Because all 16 meshes are non-watertight/non-manifold, changing only the distance would leave the previous occupancy backend invalid; implementation therefore also follows B7's required deterministic multi-direction ray-parity majority rule.
- **TDD** — red collection failed on the new `SURFACE_CLEARANCE_METERS` contract; green geometry suite is `11 passed`. Synthetic nested shell/obstacle points prove room-air/solid/outside parity, exact 0.20 m+eps behavior, and chunk identity.
- **Implementation** — split physical validity (`classify_free_space`: 31 frozen Fibonacci-sphere directions, odd-intersection strict majority) from the 0.20 m clearance prior. The audit records the direction count/hash and validates sources with both rules while receivers require physical validity only.
- **Strict real audit** — `geometry_gate=PASS`; final audit SHA `ae09d9cf9416866d09dea498a1f8467e952866db8b1c914ed0bea6a75e06cf9a`. All metadata source and receiver anchors pass in all 16 rooms; minimum source surface distance is 0.231947 m. All 5,337 query grids are nonempty/finite; chosen z-band oracle mean/median/max is 0.2116/0.2408/0.4123 m and `e_oracle>0.5 m` is 0/5,337. Full suite: `136 passed, 1 skipped, 11 warnings in 31.26s`.
- **Exact cost** — chosen z-band work is 8,891,826 query-candidate pairs, up from 6,094,936 under the rejected 0.50 m/occupancy mask; cache keys are 966,728 receiver-candidate branches plus 42,696 context ViT forwards. Historical uncached K-context=8 rate projects score K=1 at 343.4 GPU-h; the cache-oriented K-context=1 proxy spans 157.5–281.0 GPU-h with median 226.8. Meeting 168 GPU-h requires at least 14.702 end-to-end RIR/s, so K=1 is plausible but must be measured; K=2/4 are not currently plausible.
- **Boundary** — geometry is now green. I1/generation stays closed until Yixun accepts the revised cost and authorizes the cache-enabled no-quality probe.

## 2026-08-20 — 0.20 m geometry revision committed

- **Version Control** — committed implementation, tests, reviewer B7 amendment, frozen parameters, regenerated audit, and revised cost evidence as `4867078b73f4fdd0bcdb1a27a6e1f4eaba65ea6f` (`exp_09 G1: adopt 0.20m ray-parity geometry`).

## 2026-08-21 — real cached Vanilla and FA-BF no-quality throughput gate

- **Authorization / boundary** — Yixun directed: “用真实缓存引擎做小规模吞吐测试测定准确所需时间”. Only bounded throughput generation was opened. The probe writes timing/provenance JSON and explicitly saves no localization score, ranking, or quality value.
- **Implementation** — added frozen FLAC/AGREE loading, exact shared Retrieval audio encoding, generated clamp, deterministic batch-invariant candidate/sample seeds, real rectified-flow + VAE + AGREE scoring, runtime projection, candidate reconstruction/hash guard, and branch-cache primitives. Vanilla caches query context and receiver-candidate source branches. FA-BF follows the released cylindrical+C4 dependency graph: query-cache `context_poses_vit/context_audio`, receiver-candidate cache `source/source_vit`, and candidate-dependent `context_poses.dphi` recomputed inside the timed generation batch.
- **Offline strict load** — the gated DINOv3 architecture is instantiated from the checked-in byte-equivalent config SHA `fbf772f80cf673be7c8f59d44853b814a67317827063273844a36fcdc456da1d`; the full FLAC/AGREE checkpoints then replace every tensor. Both 40k FLAC probes report `0 missing, 0 stray unexpected, 0 whitelisted`; AGREE uses `strict=True`. The correct architecture file is `FLAC_AR.json` SHA `f3eafef4456666e4705ddaf35540f6b9f1f746189814cec000bac794ba2a7ec9`. `FLAC_AR_InContext.json` was fail-closed because its 256-dimensional global embedding mismatches the checkpoint's 512-dimensional layer.
- **Fail-fast probe corrections** — pre-generation attempts caught an incorrect worklog input path, the training-time AGREE relative VAE bootstrap, and a missing context channel axis. A later identity diagnostic showed full-batch vs batch-one mixed-precision kernel drift; the gate now requires shape-matched branch identity and separately records the full-vectorized drift. FA unit tests then caught that cylindrical `context_poses.dphi` depends on the candidate, leading to the correct three-layer cache above rather than an invalid optimistic reuse.
- **Cache integrity** — every cached branch is bit-identical to its shape-matched uncached reference for both arms. Full-vectorized masks remain bit-identical; the maximum token difference from mixed-precision batch-shape changes is transparently recorded as `0.00390625`. Candidate noise is bit-identical across batch partitions. No quality value was read.
- **Final protocol** — RTX A6000; exact frozen context/audit hashes `b757da281dcde3ffc310aac67279a240dac5cb1ff1d9966bf918f69c4dde6f58` / `ae09d9cf9416866d09dea498a1f8467e952866db8b1c914ed0bea6a75e06cf9a`; 512 real candidates; source batch 64; generation batches 128/256/512; one warm-up plus eight measured batches/size. Batch 512 peaks at 7.19 GB.
- **Measured result** — Vanilla: 141.732 generated scores/s, 713.546 source-cache candidates/s, 70.34 GPU-hours at `K=4`. FA-BF: 139.099 generated scores/s, 145.935 C4 source-cache candidates/s, 73.52 GPU-hours at `K=4`. Serial two-arm total is **143.86 GPU-hours / 5.99 days**. Replacing winning rates with each arm's slowest individual measured batch gives **157.32 GPU-hours / 6.56 days**; an operational 10% reserve is 173.05 hours / 7.21 days. Two matched GPUs give nominal 3.06 days wall time.
- **Decision** — both the nominal and slowest-measured serial projections are below the pre-registered 168 GPU-hour line, so the ladder selects and freezes **`K=4`** before quality. The 10% reserve is scheduling guidance, not a post-hoc K change.
- **Evidence** — final canonical timing SHAs: Vanilla `e60f8ead63b0fcf8c8522d7adafc84484852324d832c86a7c68a47bbcc979ca4`; FA-BF `f59434cd31abe62fb3b055bd68841da1d7f9cd6322283077fc48d7fd993c2532`. Detailed file hashes and conservative arithmetic are in `throughput_probe_analysis.md`.
- **Validation** — localization engine tests `7 passed`; complete permanent suite `143 passed, 1 skipped, 11 warnings in 31.21s`; `py_compile` and `git diff --check` pass.
- **Remaining gate** — this closes the no-quality cost decision only. A bounded one-query then one-room smoke must still validate streamed aggregation, tail batches, output/resume hashes, and observed wall-clock overhead before a full quality run.

## 2026-08-21 — real throughput gate committed

- **Version Control** — committed the real cache engine, offline strict-load architecture, tests, five no-quality repeat/final timing JSONs, measured cost analysis, frozen `K=4` plan amendment, and notebook evidence as `c35553db319914057055736c883c9cf7f0bfe0f8` (`exp_09: measure real cached localization throughput`).

## 2026-08-21 — Yixun replaces the three K settings with 1/4/8

- **Decision** — Yixun directed: “三个变成K=1，4，8”. This explicitly supersedes the provisional single-K selection and the `4 -> 2 -> 1` runtime fallback. The reported stochastic settings are now fixed at `K∈{1,4,8}`.
- **Nested execution** — generate one counter-seeded sequence of eight RIRs per query/candidate. K=1 aggregates sample 0, K=4 aggregates samples 0–3, and K=8 aggregates samples 0–7. This supplies all three paired readouts for the cost of K=8 and prevents different K values from receiving unrelated randomness.
- **CPU-only reprojection** — no new GPU generation and no quality read. Applying the final measured rates to the exact 8,891,826 pairs gives two-arm totals 38.31/143.86/284.59 GPU-hours for K=1/4/8. The requested nested run is governed by K=8: 140.05 hours Vanilla + 144.54 hours FA-BF = **284.59 GPU-hours / 11.86 serial days**. Slowest-measured-batch bound: **311.52 hours / 12.98 days**; plus 10% operations reserve: 342.67 hours / 14.28 days.
- **Budget boundary** — K=8 exceeds the earlier 168-hour full-execution ceiling. This entry freezes the requested K values but does not infer authorization for the larger full generation. A renewed compute decision is required after presenting the updated cost.
- **Change** — `SCORE_SAMPLE_COUNTS=(1,4,8)` is the code contract; the probe reports all three projections and no longer chooses one K. Plan, parameters, cost analysis, and `throughput_projection_k1_k4_k8.json` now reflect the override.
- **Version Control** — committed the nested K contract, CPU-only reprojection, plan/cost amendments, and test update as `8276478a50c16878a27431a164e1637957de1c6d` (`exp_09: set nested score samples to 1 4 8`).

## 2026-08-21 — 64-query room-stratified pilot implementation and smoke gate

- **Authorization** — Yixun fixed a pilot of four target queries from each of the 16 mesh-available rooms, retaining `N_ctx=8` and reporting nested `K_gen={1,4,8}`, then directed implementation and safe execution after verification.
- **Frozen pilot** — room-stratified sampling without replacement, independent NumPy PCG64 seed 42; 64 target queries, 16 rooms, and exactly 46,301 query-candidate pairs. Manifest SHA-256: `6eeeec401c3f63e47ab446b4b43efe2f1db260bad9decf74de423d0872b659a2`.
- **Implementation** — added a strict pilot manifest, stable log-mean-exp score, candidate metrics/random baseline, a two-arm CUDA runner, atomic per-query JSON/NPZ artifacts, content-hash validation, fail-closed parameter drift, batch-invariant nested samples, resume, complete-pilot aggregation, and exact launch commands. All outputs are constrained to this NeuriPs_Workshop worktree.
- **CPU/geometry validation** — all 64 selected masks were rebuilt from the 16 official meshes and matched their frozen index hashes and counts. Complete suite: `152 passed, 1 skipped, 11 warnings`; `py_compile`, CLI parsing, and `git diff --check` passed.
- **Exact cost** — measured-rate projection for 46,301 pairs at K=8: Vanilla 0.75 GPU-h; FA-BF 0.84 GPU-h; serial total 1.59 GPU-h, or 1.75 GPU-h with a 10% operational reserve.
- **GPU smoke** — one paired real query (`Apartments_idx_42`, 176 candidates) completed in 10.8 s Vanilla and 11.9 s FA-BF after strict checkpoint loads. Both artifacts have finite `[176,8]` similarities, bit-identical candidate arrays, shared random baseline, `N_ctx=8`, and K prefixes 1/4/8. Vanilla run SHA `0007814e135043e57959a7e0b2161e14095d13d015db02e9f97a3e41261b1a9e`; FA-BF run SHA `3df9a80afad7050f807cce3e53cd1839f844bd3049f2e4d7e112804304b48b7e`. Reopening the Vanilla command validated and skipped the completed query as `resume`.
- **Gate result** — pass. The exact `run_pilot_commands.sh` may now launch the resumable formal two-arm pilot on an idle A6000.

## 2026-08-24T02:50:30-04:00 — real-RIR diagnostic upper bound opened

- **Goal** — execute Query 6's diagnostic upper bound and corresponding visualization without changing the completed Vanilla/FA-BF pilot artifacts.
- **Hypothesis** — exact-target retrieval will be rank 1 by identity, while target-to-hardest-negative margin, softmax mass, entropy, and negative distance will reveal whether frozen AGREE is sharp or ambiguous on real same-receiver RIR banks.
- **Scope** — both completed, non-overlapping 64-query pilots (128 targets / 16 rooms); sparse metadata source bank at each known receiver; frozen `AGREE_fullAR.pt`; mono 22.05 kHz, 10,240-sample reviewed audio path; visualization `T=0.1` only.
- **Acceptance criteria** — all 128 targets appear exactly once; every bank contains the exact query path and metadata coordinate; all scores are finite; self cosine is numerically 1; target rank is reported rather than assumed; artifacts distinguish the identity ceiling from the dense-grid geometric oracle; existing model results remain byte-untouched.
- **Result** — `in_progress`; the control was already pre-registered in the approved exp_09 plan. TDD starts with bank construction, score diagnostics, and aggregate contracts before the AGREE run.
- **Next** — add red tests, implement the bounded diagnostic, validate on synthetic and one real bank, then launch with an exact recorded command.

## 2026-08-24 — real-RIR upper bound: scoring correction, execution, and visualization

- **TDD / initial implementation** — the first collection was red with `ModuleNotFoundError: src.localization.real_rir_oracle`; bank discovery, metadata-coordinate validation, ambiguity summaries, pilot joining, deterministic case selection, report rendering, and visualization joins were then implemented. The initial focused suite reached 7 passing tests and the localization-related suite reached 40 passing tests.
- **Independent review** — authenticated Claude Code 2.1.241, requested `opus` at max effort with read-only plan tools, returned `Request timed out` after approximately 155 seconds and no verdict. The attempt is preserved as unavailable, not approval. The final-code coverage limitation is explicit in `localization_grid_preflight_opus_code_real_rir_oracle_review.md`.
- **First real smoke / runtime red** — one exact real bank encoded successfully, then report selection failed because it required four cases. The one-query boundary is now pinned by a test; the focused suite became 8 passing tests.
- **Critical scoring finding** — two initially unseeded smokes produced different margins for the same waveform bank. Source inspection confirmed `AGREE/AGREE/audio_model.py::vae_sample` calls `torch.randn_like` even in `eval()` mode. Dotting the target candidate feature with itself would force cosine 1 and would not match the current localization score. The identity implementation was rejected before the formal run.
- **Corrected protocol** — replace each FLAC-generated RIR with the released real candidate RIR; encode the observed RIR once and independently encode each candidate eight times with fixed, role-separated BLAKE2-derived per-query seeds; reuse nested prefixes `K={1,4,8}` and the exact stable `tau=0.1` log-mean-exp scorer. The target candidate and observation are byte-identical waveforms but not a reused feature. Visualization softmax uses `T=0.1` and is explicitly uncalibrated.
- **Reproducibility smoke** — two consecutive single-query GPU 1 runs produced identical scientific SHA `6106cc285abc9b465292279206d3676a65cc51a393213d70d5bf78a10e42df53` and identical K=1/4/8 values. For `Apartments_idx_42/S006_R006`, all K rank the target first; K=8 target margin is 0.264356.
- **Formal scope** — union of seed-42 batch 1 and seed-43 batch 2: 128 unique queries, exactly 64 per batch, 16 rooms, 116 unique same-receiver real-RIR banks, 9/10/10 min/median/max candidates. GPU 1 was selected after a read-only shared-device check; existing processes were not interrupted. The run completed in 16.75 seconds after model startup.
- **Formal result** — scientific SHA `bc8cbffaa1107647e9ebac8ffeddc3dd0f6226ca03801a5d1064df55d9b5a1cb`. Target R@1, median/mean error, and success@0.5/1.0 m are respectively `1.000`, `0.000/0.000 m`, and `1.000/1.000` at each K=1/4/8. Mean/median target margin is 0.280153/0.251478 at K=1, 0.280096/0.251480 at K=4, and 0.280104/0.251411 at K=8. At primary K=8, p10/p90 margin is 0.086530/0.519977, mean uncalibrated target mass is 0.828180, mean normalized entropy is 0.239317, and mean hardest-negative distance is 2.556224 m.
- **Context against completed dense-grid pilots (K=8, same 128 target queries but different candidate set)** — combined Vanilla has median/mean error 0.986859/1.962437 m and success@0.5/1.0 m 0.164062/0.507812; FA-BF has 1.058661/1.940405 m and 0.148438/0.484375; deterministic random has 2.201885/3.020251 m and 0.046875/0.195312. This large gap shows the frozen AGREE space can robustly retrieve the exact real target among sparse same-receiver RIRs; it does not by itself identify how much of the dense-grid gap is generator error versus off-grid/candidate-discretization error.
- **Visualization** — aggregate PNG SHA `13254d7ec4388d3a9359755ffb9b287c96a503c5e8ee20dbc1266da4acc05660` at 2205×1554; four-case PNG SHA `4aa552ed88217282f46345656d9619372f8838a12364bf9aed99fd744ef173cb` at 2562×2898; visualization manifest SHA `f40dbd1feec95122c63088c54e3b0ec2dfae8b3f77cde01b1118dcbeaf9b9b74`. Cases are deterministic: maximum margin, minimum margin, maximum entropy, and median-margin-nearest, without hand selection.
- **Readback / isolation** — payload and visualization canonical hashes validate; raw `[candidate,8]` similarities are finite; every target path/coordinate matches the bank; all K scores numerically reproduce from the raw similarities; formal query 1 is byte-for-byte equal to the reproducibility smoke row. No file under `pilot_results/` or `pilot_results_batch2/` was created or modified after this task opened.
- **Final validation** — `42 passed` across localization-related and real-RIR tests; `py_compile` passed for all three new modules; `git diff --check` passed. Warnings are pre-existing package deprecations and sandbox NVML warnings.
- **Interpretation** — the diagnostic upper bound is achieved on this sparse bank, and K has negligible effect. AGREE is not the observed bottleneck for exact-real-RIR self-retrieval. The result is intentionally labeled a sparse ground-truth-RIR control, not a dense-grid upper bound, unseen-material test, or independent held-out localization benchmark.
