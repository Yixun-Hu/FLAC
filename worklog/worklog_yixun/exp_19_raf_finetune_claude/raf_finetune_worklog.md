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

## 2026-08-19T18:42:28-0400 — APPROVAL: plan Rev 2 approved by Yixun ("approve the plan", 2026-08-20)

- **Decision record** — Blanket approval adopts the recommendation-first option of every §10 decision: (1) Mapping H is v1; Mapping A (unseen-source) becomes its own later experiment; (2) N_g=16 groups/room, 12/24 per group, 4 val groups, farthest-point selection; (3) **open3d install into the `flac` env AUTHORIZED**; (4) normalization rule: none unless readback shows off-scale, else one train-support-derived scalar; (5) **HAA Zenodo download for R-cal AUTHORIZED**; (6) 5 eval seeds 42–46 paired, T60-demotion rule as registered.
- **Next** — Implementation contracts → Opus 5 max-effort Coder (TDD, commit-per-cycle) → consolidated Codex code review → fix round; open3d install + HAA acquisition started in parallel; readback rung fires when FurnishedRoom finishes unzipping.
