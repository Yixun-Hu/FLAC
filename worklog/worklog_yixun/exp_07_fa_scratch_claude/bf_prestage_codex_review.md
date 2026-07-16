Overall: **REQUEST-CHANGES** — File 1 blocks launch.

- **FILE 1 — REQUEST-CHANGES**
  - [L34–35](</home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/bf_scratch_launch.sh:34>): GPU guard fails open if `nvidia-smi` errors; without `set -e`, `BUSY=""` is treated as free. Explicitly check the query’s status.
  - [L54](</home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/bf_scratch_launch.sh:54>): run the pin/assert gate with `HF_HUB_OFFLINE=1`; it checks the cache before constructing models, whose `from_pretrained` call can otherwise contact Hub and mutate the cache after validation.
  - [L66](</home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/bf_scratch_launch.sh:66>): extra `--precision bf16-mixed` flag versus the proven B‑V command. It equals [the default](</home/yixunhu/codespace/FLAC/defaults.ini:33>) but violates exact flag identity; remove it.
  - Wandb gate, process-substitution logging, remaining manifest, and EMA config pass.

- **FILE 2 — SHIP**
  - [L9–40](</home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/bf_screen.sh:9>) normalize identically to `bvext_screen.sh` after expected BF path/config/name substitutions. No drift.

- **FILE 3 — SHIP**
  - Exact diff is solely [L159](</home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF_online_eval.json:159>): `training.use_ema: true → false`.
  - Correct eval selector; same position as B‑V online config L159.