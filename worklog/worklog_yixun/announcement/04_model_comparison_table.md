# Announcement 04 — model_comparison.md is the living results table (2026-08-03)

Yixun directive: maintain `worklog/worklog_yixun/model_comparison.md` as the cross-experiment model results table (both K, all six metrics, mean ± std over 5 eval seeds, full unseen split), and **on EVERY model-results update (new 5-seed block, new gate, new checkpoint of record): regenerate the table (`python3 worklog/worklog_yixun/gen_model_comparison.py`), commit, and push immediately.**

Rules: numbers are never hand-edited — the generator aggregates from raw per-seed metric JSONs (row specs = glob patterns in the generator); single-seed screens never enter the table; each arm is evaluated under its own protocol (fa arms with `--cond-method fa_invariant`); new models = new row spec in the generator, same commit.
