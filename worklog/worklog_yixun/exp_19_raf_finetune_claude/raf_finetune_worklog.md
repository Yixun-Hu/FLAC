# exp_19 raf_finetune — working notebook

Branch `localization-exp` (shared with exp_18; this experiment is driven by the SECOND session on mae-cab-lab-server — the tmux peer `localization-exp [453dd7]` owns exp_18 r1 and its files are not touched from here).

## 2026-08-19T16:14:00-0400 — Scaffold + state reconciliation

- **Goal** — Open exp_19 per Yixun's 2026-08-19 directive (verbatim in `raf_finetune_yixun_query.md`); reconcile the two-session state first so no in-flight work is disturbed.
- **Result** — passed:
  1. **exp_18 is peer-owned and live**: plan Rev 3 approved (`ab20700`), impl contracts assembled (`5067be5`), r1 TDD cycles 1–11 committed through `e8b6d49` (16:10), full `pytest src/tests` sweep running at scaffold time. The "plan review go" half of Yixun's message was already satisfied yesterday (`e71df84`, `20586ad`); nothing re-run.
  2. **This box**: `weights/FLAC` + `weights/AGREE` downloaded 15:48 (log in exp_18 folder); `AcousticRooms/` present as root-owned mount-style dirs being populated (peer's dataset landing — not ours); disk 318 G free (RAF RIR set ≈ 21.6 GB fits; Eyeful Tower visual data sized at recon); `aws` CLI present at `~/.local/bin/aws` (RAF distributes via S3 sync).
  3. Peer session notified via cross-session message that exp_19 is claimed by this session (numbering collision guard).
- **Next** — Yixun's answers to the three scoping questions (sequencing, dataset acquisition route, eval scope); then recon (RAF repo layout, mesh/pose formats, channel/sample-rate verification) → `plan_raf_finetune.md` → Codex review → approval gate, per SOP.
