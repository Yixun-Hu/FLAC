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
