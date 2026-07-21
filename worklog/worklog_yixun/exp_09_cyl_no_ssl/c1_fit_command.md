# C1 fit-probe — command (recorded BEFORE launch)

**Yixun C1 go:** 2026-07-21, verbatim "start exp_06/FLAC exp-09's GPU training"
(recorded in the exp_06 worklog @ cyl `3e416db`). **Co-tenant with the live B-F ranks
855996/856706** (~15.9 GiB each; free at records: GPU0 32,663 / GPU1 32,551 MiB — both
≥ the 21,900 MiB bootstrap gate). B-F is never signalled or touched; the fail-closed
gates own the memory side; the mutual-slowdown flag is in the exp_06 worklog.

Frozen pins: **EXPECT_PACKAGE_SHA = `3e416db1b6933dd842a3667432ff21436e7089ca`** (cylindrical-dinov3
frozen at the go-recording commit; no cyl commits until post-C1);
**EXPECT_EXP09_SHA = the commit landing THIS records file** (audit-precedent
self-referential convention: resolved at launch as `git rev-parse HEAD`, re-verified
post-hoc to equal this records commit; `--expect-clean-worktree` semantics are built
into the pin gate's scoped clean-tree checks).

```bash
cd /home/yixunhu/codespace/exp-09-cyl-dinov3-no-ssl
export EXPECT_PACKAGE_SHA=3e416db1b6933dd842a3667432ff21436e7089ca
export EXPECT_EXP09_SHA=$(git rev-parse HEAD)   # = this records commit; verified post-hoc
bash worklog/worklog_yixun/exp_09_cyl_no_ssl/c1_fit.sh \
  /home/yixunhu/codespace/cylindrical-dinov3/worklog/worklog_yixun/exp_06_flac_no_ssl_claude
```
(`set -o pipefail` verbatim inside the script; tee log + peak JSON land in the exp_06
folder, OUTSIDE this worktree.)

Acceptance: exit 0; pin gate ALL PASS (both SHAs, scoped clean trees, config delta,
class/gauge/eager, cond_method fa_invariant [0.0]); sampler ≥1 valid sample per GPU;
peak JSON with per-GPU peaks + derived_gate_mib = max(peaks) + 4,096. Then the records
step FREEZES `c1_frozen_min_free.txt` = derived_gate_mib.

## CORRECTION (attempts 1–2 → provisioning)

Attempt 1: my launch shell lacked the `flac` env (bare `python` → no torchaudio) AND my
outer `| tail` pipe masked the abort code (no pipefail in MY wrapper — the standing
lesson, self-inflicted; the registered command itself was clean). Attempt 2 (env
active, pipefail set): pin gate ALL PASS; the training child died fail-closed at
`weights/FLAC/VAE.safetensors` — the worktree lacks the live checkout's UNTRACKED
runtime assets. **Provisioning: read-only symlinks** `weights` + `AcousticRooms` →
the live FLAC checkout (reads only; concurrent with B-F's own reads; both links sit
OUTSIDE the scoped clean-tree pathspecs so the pin gates stay valid; writes go to the
worktree-local `outputs_FLAC/`/`wandb/`). Failed-attempt peak JSON/log kept
(timestamped names). Attempt 3 = the registered command re-run with the env active.
