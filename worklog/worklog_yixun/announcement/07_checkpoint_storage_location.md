# 07 — All training checkpoints live under /media/diskstation/yixunhu/FLAC/checkpoints/

**Logged:** 2026-08-20 (from Yixun, during exp_18 R2)

## Original instruction (verbatim)

> later on, all the training checkpoints you should store on /media/diskstatioin/yixunhu/FLAC/checkpoints/ folder (please name your folder under checkpoints/ folder using an appropriate and proper name)

(Path as typed contains "diskstatioin"; the actual mount, verified, is `/media/diskstation/yixunhu/FLAC/checkpoints/` — same disk that hosts `AcousticRooms/` and `HAA/`.)

## Rule

- Every training checkpoint this session's experiments produce (or receive by rsync for evaluation) is stored under `/media/diskstation/yixunhu/FLAC/checkpoints/`, never only in the repo-local gitignored `outputs_FLAC/`/`weights/` (those may hold working copies/symlinks).
- One properly-named subfolder per experiment/arm: `checkpoints/exp<NN>_<exp name>[_<arm>]/` (e.g. `checkpoints/exp20_loc_crossarm_yawaug/`). No loose files at the top level.
- Applies prospectively: exp_18 is inference-only and produces no training checkpoints; the first users will be the cross-arm localization checkpoints (VANL/FA B-F/YAWAUG/cyl rsyncs) and any future training runs. (`exp19_ckpts/` predates this rule and belongs to the other session.)

## Why

The NAS is the shared, machine-independent store; repo-local output dirs are gitignored, machine-local, and have already diverged across the two boxes. A single canonical checkpoints tree with per-experiment names keeps provenance auditable across sessions.
