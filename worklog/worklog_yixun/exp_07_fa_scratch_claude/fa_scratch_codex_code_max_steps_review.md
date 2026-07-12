**Reviewer:** gpt-5.6-sol xhigh · codex-cli 0.144.1 · `codex exec` read-only · 2026-07-11

**Verdict: REQUEST-CHANGES**

### Medium

- The mandated Trainer-boundary regression test is missing. Tests only validate `build_trainer_kwargs()` ([test_train_max_steps.py:85](/home/yixunhu/codespace/FLAC/src/tests/test_train_max_steps.py:85)); all eight would still pass if the actual call at [train.py:178](/home/yixunhu/codespace/FLAC/train.py:178) stopped using the helper or restored `max_steps=1000000`. Add a test that monkeypatches `pl.Trainer` and asserts the CLI override reaches its constructor—most cleanly through a small tested trainer-construction wrapper.

### Low

- Canonical HAA guidance remains actively stale: [CLAUDE.md:133](/home/yixunhu/codespace/FLAC/CLAUDE.md:133) still instructs manual source editing and forbids a CLI flag. The comments at [train.py:42](/home/yixunhu/codespace/FLAC/train.py:42) and [defaults.ini:50](/home/yixunhu/codespace/FLAC/defaults.ini:50) do not resolve that contradiction. Update CLAUDE.md to prescribe `--max-steps 1000`.

No kwargs-faithfulness or prefigure-consumer defect found: every old Trainer key/value and `**val_args` tail is preserved; prefigure produces an integer `--max-steps`; `finetune_cond.py` uses independent argparse with default **2000**, not 625; the existing test `conftest.py` guards against stale installed `src`.

**Single Most Valuable Change:** Add the mocked `pl.Trainer` boundary test so reverting or bypassing the new plumbing fails.