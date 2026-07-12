# 03 — Per-user worklog namespace (standing directive, Yixun 2026-07-12)

All experiment logging lives in `worklog/worklog_<username>/` — for this project `<username>` = `yixun` → `worklog/worklog_yixun/`. Everything that previously sat directly under `worklog/` (announcements, `exp_<NN>_<exp name>_claude/` folders, archives) now sits inside the user namespace, e.g. `worklog/worklog_yixun/exp_07_fa_scratch_claude/`.

**Exception:** `worklog/experiment_SOP.md` stays directly under `worklog/` — it is the shared/portable SOP, not user logging.

Yixun's instruction (verbatim): "rename the worklog to worklog_<user name> and for me, the user name is yixun. and put worklog_yixun inside worklog folder. And update this rule inside experiment_SOP.md: all previous logging should be changed to logged inside worklog_<username>, for example, exp_<NN>_<exp name>_claude/ should be worklog_<username>/exp_<NN>_<exp name>_claude. But for @worklog/experiment_SOP.md , you need to put it under worklog, not inside worklog_yixun."

Migration executed 2026-07-12 via `git mv` (history preserved). Five worklog scripts that computed the repo root by fixed directory depth were converted to a `.git` marker-walk (layout-proof); the fail-closed exp_07 gate re-ran green at the new depth with an unchanged init-identity hash. The in-flight B-V training's tee log followed its inode through the move (verified growing).
