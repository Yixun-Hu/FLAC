**Reviewer:** OpenAI Codex (`gpt-5.6-sol`, xhigh, read-only sandbox) · **Date:** 2026-08-19 · **Round:** consolidated closure review (exp_17 extras/rotseeds drivers + exp_19 curve driver, aggregator, YNA edits)

**Final closure verdict: ZERO blocking findings — no published exp_17/exp_19 number requires retraction or recomputation.**

---

### 1. `yaw_aug_a6000_extras_run.sh` — Non-blocking debt

Protocol is correct at [lines 61–76](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_extras_run.sh:61): correct seen/unseen K configs, vanilla, bf16, fixed rotation, cfg 1.0, one diffusion step, seeds 42–46, and distinct eval names.

Artifact audit found exactly the expected 20 records, with every embedded dataset, K, seed, rotation, checkpoint step, EMA source, sample count, conditioning method, and autocast field matching.

Non-blocking debt:

- The resume predicate trusts a filename glob without parsing the record at [lines 91–93](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_extras_run.sh:91).
- Completion counts arbitrary non-stream JSONs and accepts `>=20`, rather than validating the exact 20-cell set, at [lines 101–103](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_extras_run.sh:101).
- The foreign model config and resolved checkpoint are not byte-hash pinned at [lines 30, 47–50](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_extras_run.sh:30).

These paths could suppress a future rerun incorrectly, but did not affect the published artifacts.

### 2. `yaw_aug_a6000_rotseeds_run.sh` — Non-blocking debt

The 24-cell grid and protocol at [lines 23–35](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_rotseeds_run.sh:23) match the reviewed exp_17 grid.

All 24 expected records exist exactly once. Their embedded metadata validates rotations 90/180/270, seeds 43–46, both K configs, vanilla/bf16, cfg 1.0, one step, EMA, and the 40k checkpoint. The evaluator and dataset-config bytes were identical between the extras and rotseed source SHAs.

Non-blocking debt:

- Filename-only resume at [line 25](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_rotseeds_run.sh:25).
- The checkpoint gate proves only that the existing symlink resolves, not its expected target or hash, at [line 19](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_rotseeds_run.sh:19).
- Completion counts matching filenames rather than validating JSON contents at [lines 41–43](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_rotseeds_run.sh:41).

### 3. `haa_ft_curve_eval.sh` — Non-blocking debt

The executed 24-cell grid is correct: three arms × eight intermediate steps. It uses exactly one checkpoint per arm/step at [lines 24–27](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/haa_ft_curve_eval.sh:24), the correct per-arm conditioning at [lines 18–23](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/haa_ft_curve_eval.sh:18), and the same K=8/seed-42/bf16/per-scene/cfg/steps contract as the reviewed driver at [lines 33–38](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/haa_ft_curve_eval.sh:33).

All 24 raw records passed full metadata validation. The 410/1000 records supplied by the main grid also passed. All source SHAs used byte-identical `eval_FLAC.py`, metric callback, HAA K configs, and arm configs.

Non-blocking debt:

- Filename-only skip at [lines 29–31](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/haa_ft_curve_eval.sh:29).
- No evaluator/config hashes are pinned.
- Successful exit checks child return codes but never performs an exact 24-record, protocol-aware final recount at [lines 43–44](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/haa_ft_curve_eval.sh:43).
- The no-op loop at [line 17](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/haa_ft_curve_eval.sh:17) is harmless clutter.

### 4. `exp19_aggregate.py` — Non-blocking, changes recommended before reuse

The aggregation math is correct:

- Per-room values are averaged across rooms at [lines 42–46](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/exp19_aggregate.py:42).
- Only T60 excludes `dampenedBase`.
- Seed means and sample standard deviations are computed correctly at [lines 48–53](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/exp19_aggregate.py:48).

Reproduction result:

- Three-arm endpoint: all numerical rows reproduce.
- Four-arm endpoint: all numerical rows reproduce.
- Steps curve: all published values reproduce.
- Two-arm file: the step-1000 paper rows reproduce, but the script cannot reproduce that file’s step-410 or pooled tables because endpoint rendering is hard-coded to step 1000 at [lines 58–64](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/exp19_aggregate.py:58).
- Formatting and some method labels differ, so the HEAD claim of reproducing every published table “byte-for-byte” is not literally true.

The protocol-refusal contract is incomplete:

- Selection trusts arm, step, K, and seed encoded in paths/filenames at [lines 24–30](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/exp19_aggregate.py:24).
- It validates only `cond_method` and `cond_autocast` at [lines 33–36](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/exp19_aggregate.py:33). It does not validate embedded seed, dataset/K, eval name, checkpoint/step, cfg scale, diffusion steps, rotation, EMA source, sample count, BF angles/cap, room set, or finite metrics.
- Record count alone does not prove one record per requested seed; duplicates could replace a missing seed while preserving the count.
- Curve mode converts any `SystemExit`, including a protocol mismatch, into `—` and exits successfully at [lines 75–80](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/exp19_aggregate.py:75).

The actual selected records were independently checked against all those omitted fields and were clean, so this does not invalidate a published number.

### 5. `haa_ft_launch.sh` YNA additions — Ship; no published-number finding

YNA correctly:

- Resolves to the exact stock configuration at [lines 179–186](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/haa_ft_launch.sh:179).
- Redirects only its initializer to `HAA_init_YAW.ckpt` at [lines 185–186](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/haa_ft_launch.sh:185).
- Shares P1’s stock-byte pin and contract at [lines 289–293](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/haa_ft_launch.sh:289) and [383–389](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/haa_ft_launch.sh:383).
- Skips the rotation probe appropriately because its finetuning config is vanilla at [lines 503–505](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/haa_ft_launch.sh:503).
- Revalidates the manifest-pinned initializer immediately before consumption at [lines 618–620](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/haa_ft_launch.sh:618).

The actual YNA launch log proves stock config, YAW initializer SHA `6b11a7d4…`, correct training argv, endpoint reached, finite loss, both checkpoints, and final rc 0. The edit does not weaken P1, BF, or YAW gates.

Non-blocking debt: no YNA-specific guardtest coverage was added.

### 6. `haa_ft_eval.sh` YNA additions — Ship; minor documentation debt

YNA is admitted at [lines 82–86](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/haa_ft_eval.sh:82) and uses the stock model config at [lines 181–187](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/haa_ft_eval.sh:181). Existing dispatch correctly gives it vanilla conditioning and no orbit suffix or flags. The stock config remains byte-pinned at [line 169](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/haa_ft_eval.sh:169).

The actual YNA queue completed 20/20, and all 20 raw records passed full protocol validation.

Non-blocking debt: comments/error text and the `EXPECTED_CELLS=60` label remain three-arm-specific at [lines 5–16](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/haa_ft_eval.sh:5), [line 76](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/haa_ft_eval.sh:76), and [line 342](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_19_haa_finetune_claude/haa_ft_eval.sh:342). This produced the misleading but harmless log text “20/20 … registered grid 60.”

Final closure verdict: **zero blocking findings; no exp_17/exp_19 published number requires retraction or recomputation.** Only the aggregator’s fail-closed validation and full-reproduction claim merit substantive closure debt.
