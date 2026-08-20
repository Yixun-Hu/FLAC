# exp_19 raf_finetune — working notebook

Branch `localization-exp` (shared with exp_18; this experiment is driven by the SECOND session on mae-cab-lab-server — the tmux peer `localization-exp [453dd7]` owns exp_18 r1 and its files are not touched from here).

## 2026-08-19T16:14:00-0400 — Scaffold + state reconciliation

- **Goal** — Open exp_19 per Yixun's 2026-08-19 directive (verbatim in `raf_finetune_yixun_query.md`); reconcile the two-session state first so no in-flight work is disturbed.
- **Result** — passed:
  1. **exp_18 is peer-owned and live**: plan Rev 3 approved (`ab20700`), impl contracts assembled (`5067be5`), r1 TDD cycles 1–11 committed through `e8b6d49` (16:10), full `pytest src/tests` sweep running at scaffold time. The "plan review go" half of Yixun's message was already satisfied yesterday (`e71df84`, `20586ad`); nothing re-run.
  2. **This box**: `weights/FLAC` + `weights/AGREE` downloaded 15:48 (log in exp_18 folder); `AcousticRooms/` present as root-owned mount-style dirs being populated (peer's dataset landing — not ours); disk 318 G free (RAF RIR set ≈ 21.6 GB fits; Eyeful Tower visual data sized at recon); `aws` CLI present at `~/.local/bin/aws` (RAF distributes via S3 sync).
  3. Peer session notified via cross-session message that exp_19 is claimed by this session (numbering collision guard).
- **Next** — Yixun's answers to the three scoping questions (sequencing, dataset acquisition route, eval scope); then recon (RAF repo layout, mesh/pose formats, channel/sample-rate verification) → `plan_raf_finetune.md` → Codex review → approval gate, per SOP.

## 2026-08-19T16:48:30-0400 — Scoping answers, mesh download in flight, HAA-template recon

- **Goal** — Act on Yixun's three scoping answers (start now in parallel; RAF already downloading to `/media/diskstation/yixunhu/raf_dataset` (his process, still unzipping) with meshes left to me; paper-parity finetune scope).
- **Change** — Room meshes enumerated from the RAF S3 index pages and downloading in background to `raf_dataset/3d_models/{EmptyRoom,FurnishedRoom}/`: per room `mesh.obj` (~215/212 MB, Metashape export, X-front/Y-up/Z-left, y=0 ground, meters) + `mesh.mtl` + `mesh.jpg` (~21 MB). Expected sizes recorded in the download script for post-hoc verification.
- **Command / Validation** — `curl -sS --fail --retry 3` per file; size check vs index listing on completion. RIR archive state at recon: `archived/` holds multi-part zips (`raf_emptyroom.z01..` @1 GiB each, ~43 GB total) with `EmptyRoom/` + `FurnishedRoom/` mid-extraction (his unzip, running from another machine — NOT touched from here).
- **Result** — download in flight; recon of the HAA template complete (`data/HAA/prepare_data.py`, splits, `HAA_md.py`):
  1. HAA prep = 48 kHz → librosa-resample to 22,050 → per-index WAVs; few-shot = **12 train RIRs/room**; val/test = complements. Direct template for `data/RAF/prepare_data.py`.
  2. **Structural delta**: HAA has ONE fixed speaker per room (`speaker_xyz` per scene ⇒ a single pre-rendered `{scene}_depth_image.npy` at the source suffices). RAF has per-sample 6DoF source AND listener poses ⇒ depth panoramas must be **rendered from the OBJ mesh per position** (equirect 256×512), and the render-position convention (source, HAA-style, vs listener, AR-style) plus RAF→FLAC coordinate mapping (RAF: X-front/Y-up/Z-left) are plan-level decisions.
  3. Few-shot split design for RAF (how many train RIRs, how sampled from the dense grid) is a plan-level registered choice; HAA's 12/room is the precedent.
- **Next** — When unzip lands: data readback (RIR folder schema, channel count, sample rate, pose format). Meanwhile: draft `plan_raf_finetune.md` with data-dependent facts explicitly deferred to a readback rung (exp_18 pattern), then Codex review → Yixun approval.

## 2026-08-19T16:53:10-0400 — Data readback (EmptyRoom), meshes verified, plan Rev 1 drafted, Codex review fired

- **Goal** — Turn the landed EmptyRoom data + HAA-recipe recon into `plan_raf_finetune.md` Rev 1.
- **Result** — passed:
  1. Meshes: all 6 files byte-exact vs the S3 index (EmptyRoom obj 214,624,032 / jpg 21,607,805 / mtl 175; Furnished obj 211,966,369 / jpg 21,346,537 / mtl 175).
  2. EmptyRoom readback: 47,484 captures (`data/<id>/{rir.wav,tx_pos.txt,rx_pos.txt}`), RIR mono/48 kHz/float32/1.5 s (peak ~0.01), tx = quat(4)+xyz(3), rx = xyz, 1,319 unique tx lines ⇒ ~36 rx/tx — the HAA-style same-source-other-receivers context relation exists in RAF.
  3. HAA mapping decoded (`HAA_md.py`): source-centered frame, listener into the `source` slot, context = other receivers of same source from the train pool, depth at source. Plan adopts it per-source for RAF.
  4. Plan Rev 1 written (10 sections, 6 open decisions incl. AGREE-RAF absence ⇒ no FD/retrieval in v1, open3d install approval, split constants). Codex review launched in background with the no-install clause; output → `raf_finetune_codex_plan_review.md`.
- **Next** — Fold Codex findings → Rev 2 → surface plan + open decisions to Yixun for approval. No implementation before approval.

## 2026-08-19T17:10:03-0400 — Codex review REQUEST-CHANGES (16 findings) → deep-recon → plan Rev 2

- **Goal** — Fold the Codex plan review (`raf_finetune_codex_plan_review.md`, C1–C16: 14 blocking, 2 should-fix) into Rev 2.
- **Command / Validation** — Code-verified before folding: C4 (fail-closed `SUPPORTED_DATASETS` at `metric_callback.py:96`, 9600-vs-8000 branch at `:114`), C9 (README does say `--accum-batches 4` — Rev 1 omission real), C7/C8 (`--record-stream/--expected-stream-count/--record-per-scene/--frame-avg-angles` all exist in fork `eval_FLAC.py`). New measurements driving the redesign: EmptyRoom = rigid 36-mic array (bbox 1.11×1.24×1.11 m, identical every group) over **139 placements × median 9 tx poses**; every tx-pose group has exactly 36 captures; rx essentially never repeats (47,481/47,484 unique); placements re-occupied sub-cm.
- **Result** — Rev 2 written. Disposition: C1→§3/§5 (group-unit splits, N_g=16 arm + literal HAA-parity diagnostic row + Arm B deferred); C2→§3 (full-pose grouping, q≡−q, orientation-drop labelled approximation, readback audit); C3→§4 (literal registered matrix, non-circular hand-specified-ray oracle, no flipud copying); C4→§7 (explicit RAF metric policy + tests); C5→§2 (onset regression, T60 validity gate); C6→§5 (disjoint splits, final-ckpt preregistration, published counts); C7→§6/§9 (stream audit everywhere, context-ID recording); C8→§6 with **one declared deviation**: `--frame-avg-angles` registered none/n-a under vanilla (fa mechanism; pinning 0,90,180,270 would contradict announcement-05 protocol matching) — cross-checked at smoke vs `orbit_provenance`; C9→§6 (accum 4, bf16, 5 seeds 42–46 paired); C10→§8 (runtime metadata root, schema, conditioner-pass contract test); C11→§6 (eval context deterministic from capture id ⇒ seeds vary diffusion noise only); C12→§8 (float32 subtype declared divergence, fixed-scalar rule); C13→§8 (depth QA suite, miss policy, cache); C14→§7 (language + unavailable-not-zero); C15→§11 (v1 rescoped small: ~42 renders, 768-item test, ckpt cadence 100 declared divergence); C16→§9 (R-cal HAA reproduction, Zenodo acquisition = decision).
- **Analysis** — The review reshaped the experiment: uniform 36-capture groups make literal HAA parity statistically empty (24 test/room), so v1 = 16-group arm + literal-parity diagnostic row, honestly framed as array-scale interpolation; the AR-style unseen-source mapping (placement-based, median 9 tx ⇒ K=8 feasible) is surfaced to Yixun as the stronger follow-up.
- **Next** — Surface Rev 2 + 6 open decisions to Yixun (approval gate). No implementation before approval.

## 2026-08-19T18:42:28-0400 — APPROVAL: plan Rev 2 approved by Yixun ("approve the plan", 2026-08-19 ~18:40 EDT; the 08-20 date in the committed version of this line was a stamp error, corrected here)

- **Decision record** — Blanket approval adopts the recommendation-first option of every §10 decision: (1) Mapping H is v1; Mapping A (unseen-source) becomes its own later experiment; (2) N_g=16 groups/room, 12/24 per group, 4 val groups, farthest-point selection; (3) **open3d install into the `flac` env AUTHORIZED**; (4) normalization rule: none unless readback shows off-scale, else one train-support-derived scalar; (5) **HAA Zenodo download for R-cal AUTHORIZED**; (6) 5 eval seeds 42–46 paired, T60-demotion rule as registered.
- **Next** — Implementation contracts → Opus 5 max-effort Coder (TDD, commit-per-cycle) → consolidated Codex code review → fix round; open3d install + HAA acquisition started in parallel; readback rung fires when FurnishedRoom finishes unzipping.

## 2026-08-19T19:09:21-0400 — R-cal data landed; FurnishedRoom complete; readback pre-checks

- **Goal** — Progress the approved pipeline while the Coder runs (cycle 6/12 at 19:05).
- **Result** —
  1. **open3d 0.19.0** installed into `flac` (approved decision 3), raycast smoke exact; no pinned core packages changed.
  2. **HAA base rooms downloaded** (4 zips, 29.5 GB, byte-exact vs Zenodo, log `haa_download_2026-08-19.log`); extract→released-`prepare_data.py`→runtime-layout chain running in background (log `haa_prepare_2026-08-19.log`); only `RIRs.npy`/`xyzs.npy` extracted per README.
  3. **FurnishedRoom extraction COMPLETE** (Yixun's process finished 18:29): 39,132 captures, ids 000000–039131, same schema + same all_rx trailing-line off-by-one as EmptyRoom.
  4. **Group-invariant pre-check:** FurnishedRoom = 1,086 unique tx poses — **1,085 groups of exactly 36 + ONE group of 72** (one source pose captured twice, presumably at two array placements). Handling rule (within plan §2's recorded-deviation clause): **v1 selection eligibility = exactly-36 groups only**; the 72-group goes to the explicit reserve list with its anomaly recorded in `raf_splits_record.json`. To be echoed in the prep run's output and the results caveats.
- **Next** — Coder report → consolidated Codex code review → fix round; then formal readback rung (both rooms) → prep run → depth renders → smoke; R-cal HAA finetune queued behind the chain's completion.

## 2026-08-19T19:34:04-0400 — Cycle 13 landed; D1 verified on the real corpus; consolidated Codex code review fired

- **Result** —
  1. Cycle 13 (`1d1af86` + ledger `980bcd7`): rx trailing-sentinel rule per Amendment 1, conjunctive conditions, fail-closed everywhere else; suite 190 → **204 passed**.
  2. **Read-only real-corpus check:** `load_room_index` now accepts both rooms — EmptyRoom 47,484 / FurnishedRoom 39,132 captures, sentinel dropped+flagged in both. D1 CLOSED.
  3. R-cal Leg A restart running (per-seed names, Amendment 2); Leg B training ~step-mid.
  4. Consolidated Codex code review of r1 (cycles 1–13, commits `8ac6fad..980bcd7`) launched at xhigh with the no-install clause; output → `raf_finetune_codex_code_review.md`. Round r1 does NOT close before its verdict + fix round.
- **Next** — Fold review findings (fix round via the Coder if REQUEST-CHANGES); assemble Leg-A 5-seed table vs the paper HAA numbers when the rerun lands.

## 2026-08-19T21:30:17-0400 — Readback rung COMPLETE: audit passed, gauge+quat PINNED, canonical record committed

- **Result** —
  1. Measurement pass (400 onset samples/room, 200-capture crosschecks, seed 0): onset-vs-distance PASS both rooms (EmptyRoom slope-ratio 1.19 R²=0.879; Furnished 1.18 R²=0.844; constant delays 1.64/2.07 ms recorded); **T30 crop-invalidation 0/400 both rooms ⇒ T60 stays headline**; amplitude in-scale (no scalar); sentinel + exactly-36 (+ the known 72-group) confirmed.
  2. **Pins (Amendment 4):** gauge `RAF_TO_PIPELINE:(X,Z,Y)` from real-mesh evidence (sub-degree landmark bearings ×2 positions, nadir≈height ≤0.10 m across rooms, containment/bounds/sightline pass); quaternion **xyzw from the RAF release docs verbatim** ("real-part last format: xyzW") — v1 remains grouping-only.
  3. Real-mesh discoveries → Amendment 4 render policy (≤0.1% miss cap + recorded inpainting; bearing-tie applicability; floor tol 0.15 m); Coder implementing.
  4. Pinned canonical record at `data/RAF/raf_readback_record.json` (verdict passed, adjudication carried) — the artifact the prepare/render publish gates require.
- **Next** — Coder render-policy cycle → peer 'R-1/R0 done' → R8/R10 → Codex r2 re-review → Leg B eval → morning package to Yixun (canonical-prep go/no-go).

## 2026-08-19T22:19:54-0400 — R-cal Leg B training COMPLETE; 5-seed repro eval launched
- **Result** — Leg B reached global step 1000 exactly (final val/avg_loss 0.527, converged from ≥0.9; wandb offline run `m15pp1x7`); `epoch=999-step=1000.ckpt` saved and sha256-recorded (`1e153447…71fe25c`). The background wrapper's -1 exit is the harness reaping the tee, not a training failure — verified from the log's completed wandb summary. Wall time 18:44→~22:15 (~3.5 h; the recipe idles the GPU heavily: 3 micro-batches/epoch + val every 10 + 724 MB ckpt writes every 10).
- **Change** — Leg B eval launched on GPU 0 (train slot, now free; GPU 1 left to the peer's R-1/R0): step-1000 ckpt × seeds 42–46, per-seed eval-names (`exp19_rcal_repro_seed<seed>`, Amendment 2 rule), stream-audited 1282, same rcal eval config (fullHAA scorer).
- **Next** — Assemble Leg-B-vs-Leg-A reproduction table on completion (~22:40); R8/R10 land on the peer's 'R-1/R0 done'; then Codex r2 re-review.

## 2026-08-19T22:55:54-0400 — MIGRATION: exp_19 moved to its own worktree/branch (Yixun directive)

- **Goal** — Per Yixun (~22:50 EDT): exp_19 gets its own worktree, branched from `check-equivariance-necessity` (the same base `localization-exp` branched from, tip `6170007`).
- **Change** — Worktree `~/codespace/exp-19-raf-finetune`, branch `raf-finetune-exp`. State-port commits: `5f3e4a7` (all exp_19-owned files @ localization-exp `e23c45e`), `2f500d3` (test suite — missed by the first port because the glob expanded in the new worktree; caught immediately), `18c9aa9` (the three shared files whose entire localization-exp divergence is exp_19-authored: `eval_FLAC.py` R3, `metric_callback.py`/`RT60.py` cycle-12 RAF policy — verified via git diff/log before porting).
- **Validation** — `pytest src/tests/test_raf_*.py` in the new worktree: **326 passed, 7 skipped** (identical to localization-exp). Runtime scaffolding: `HAA -> /media/diskstation/yixunhu/HAA_processed`, per-file weight symlinks into the tracked `weights/FLAC/` (+`AGREE`), fresh `outputs_FLAC/`. R-cal artifacts (step-1000 ckpt + 15 eval/stream JSONs) archived to `/media/diskstation/yixunhu/exp19_rcal_artifacts/` — the originals stay in the FLAC checkout's gitignored outputs.
- **Consequences** — (i) Development history (`138dc26`..`b4c0ac1` + R-cal bookkeeping) remains on `localization-exp`; `commits_raf_finetune.md` is the map; new work ledgers new SHAs on THIS branch. (ii) The src/metrics freeze protocol with the peer session is DISSOLVED — R8/R10 land here only, never on `localization-exp`. (iii) exp_19's already-landed shared-file deltas on `localization-exp` (R3 + cycle-12) stay there — additive, peer-verified (parity 0.0, 492 green); the peer may keep or revert them independently. (iv) The exp_19 folder on `localization-exp` is frozen with a marker.
- **Next** — R8/R10 into THIS worktree (Coder) → Codex r2 re-review against THIS tree → canonical prep go/no-go package to Yixun.

## 2026-08-20T01:39:51-0400 — CODE ROUND CLOSED: Codex APPROVE at r6; package assembled

Final pass APPROVE, no open in-boundary findings; 3 residuals recorded (`raf_finetune_round_close.md`). Round totals: 6 review passes, 6 fix rounds, Amendments 1–8, 511 tests. Awaiting Yixun: canonical-prep GO + residual acknowledgment + optional second R-cal seed.

## 2026-08-20T11:42:56-0400 — Yixun directive: checkpoint storage convention

All training checkpoints go under `/media/diskstation/yixunhu/FLAC/checkpoints/<proper-name>/` (established convention — exp12_cyl_dinov3_arms already present). Applied: R-cal artifacts moved `exp19_rcal_artifacts` → `checkpoints/exp19_rcal_haa_repro/` (21 files incl. step-1000 ckpt); `checkpoints/exp19_raf_finetune/` pre-created — the canonical finetune will `--save-dir` there directly (cadence-100 makes CIFS writes cheap; ~7 GB worst case, pruned to final after eval). Noted read-only: `FLAC/exp19_ckpts/{BF,P1}` appearing 11:17–11:30 = Yixun's own arm-checkpoint rsync (B-F, exp07_P1) — his, untouched. The rcal manifest's artifact pointer updated by this entry (former exp19_rcal_artifacts path superseded).

## 2026-08-20T11:54:51-0400 — GO: Yixun approved the canonical run ("go for the exp_19 canaical-run")

Blanket GO adopts the package recommendations: residuals 1–3 ACCEPTED AS RECORDED; second R-cal seed SKIPPED. Sequence begins: prep dry-run → canonical prep (splits committed once; runtime → NAS RAF_processed) → 42 depth renders → smoke (GPU, announced to peer, co-tenant OK) → 1000-step finetune (--save-dir /media/diskstation/yixunhu/FLAC/checkpoints/exp19_raf_finetune per the storage directive) → 5-seed evals ×2 rows + diagnostic row.

## 2026-08-20T17:52:36-0400 — CANONICAL RUN COMPLETE — results committed
Zero-shot → finetuned on the 768-item test row: T60 11.25→5.64 (−50%), C50 3.02→0.85 (−72%), EDT 145→38.9 (−73%), all ≫ the R-cal band. Full record: `raf_finetune_results.md`. Run artifacts on NAS per the storage directive.
