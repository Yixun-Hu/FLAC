**SHIP — no blocking findings.**

- [Line 18](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/bvext_screen.sh:18): exact `step=${S}.ckpt` suffix; `7000` does not match `70000`. Paths/expansions are correctly quoted.
- [Lines 23–37](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/bvext_screen.sh:23): whole block appends stdout+stderr directly to a regular file. No downstream reader, so the prior `tail -0` SIGPIPE cannot recur.
- [Lines 25–36](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/bvext_screen.sh:25): evals run sequentially; each status is captured immediately. Arguments are accepted by `eval_FLAC.py`.
- [Lines 40–44](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/bvext_screen.sh:40): any nonzero eval marker fails; absence of any successful eval also fails.
- GPU exposure is confined to each eval process on GPU 1; evaluations are not concurrent. Only normal resource contention remains, previously measured as negligible with ample headroom. Metrics use distinct names and do not modify the checkpoint.