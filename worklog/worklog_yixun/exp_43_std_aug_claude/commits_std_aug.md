# commits_std_aug — exp_43 commit ledger (branch `exp-43-std-aug`, base `6a51f741`)

| # (plan §6) | SHA | Content | Pushed |
|---|---|---|---|
| 0 | `def92ceb` | scaffold: query log (Queries 1–5), worklog, approved plan r1, plan r0 in `plan_history/`, Codex plan round 1 artifacts (prompt, review, transcript) | no |
| 6a | `e1288a92` | verbatim copies (`cmp`-identical) of the exp_40 training kit: `exp43_kit.py`, `exp43_train_local.sh`, `exp43_evidence_driver.py`, `src/tests/test_exp43_kit.py` — no edits | no |
| baseline (§4.2) | `581bf072` | regression-baseline tool `exp43_regression_baseline.py` (`record` / `compare`), its tests (76), the full CPU suite record at `6a51f741` (`exp43_regression_baseline.json`: 6966 nodes — 7 failed (CPU-only: 3 flash-attention, 4 `torch.load` without `map_location`), 6928 passed, 30 skipped, 1 xfailed, + 2 subtests), the terminal log and the JUnit file | no |
| history (§5.4) | `09ef0b12` | W&B history extractor `exp43_train_history.py` (`extract`: offline run file → `step, train/loss, lr` CSV + sidecar; wandb 0.15.4 pinned) and its tests (35 + 1 opt-in) | no |
| 1 | `ba932892` | mirror operator `src/data/yaw_mirror.py` (`check_mirror_metadata`, `draw_mirror_flips`, `mirror_scene_metadata`) + `src/tests/test_yaw_mirror.py` (108 tests) | no |
| 3 | `cd7053f1` | factory key `training.yaw_aug.flip` in `src/training/factory.py` (+18 / −2) + `src/tests/test_yaw_aug_flip_factory.py` (41 tests) | no |
| 4 | `1e0f3d8c` | wrapper flip path in `src/training/diffusion.py` (+44 / −2; keyword-only `yaw_aug_flip`, banner `flip=0.5`), `src/tests/test_yaw_aug_flip.py` (45 tests), the disclosed F12 edit of `src/tests/test_yaw_aug_c16.py` | no |
| 5 | `44ba2854` | arm config `FLAC_AR_YAWAUG_FLIP.json` (exp_15 bytes + `"flip": true`), `src/tests/test_yaw_aug_flip_config.py`, flip-path golden: capture tool, fixture (record `03e9a9e2…`, captured at `1e0f3d8c`), `src/tests/test_exp43_yaw_aug_flip_golden.py` | no |
