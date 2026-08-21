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
