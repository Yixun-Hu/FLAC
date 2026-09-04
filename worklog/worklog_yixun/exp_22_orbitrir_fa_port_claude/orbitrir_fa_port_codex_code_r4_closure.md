**Reviewer:** OpenAI Codex (gpt-5.6-sol, `codex exec`, read-only sandbox) · **Date:** 2026-09-04

# R4 STILL OPEN

All five original R4 findings are fixed as prescribed. Three new documentation/anonymity findings remain:

1. **Minor — [README.md:267](/home/yixunhu/codespace/ORBITRIR/README.md:267)**  
   C15 removed the AGREE training block containing `cd AGREE`, but the surviving evaluation block still runs `python -m AGREE_train.main` and uses `../data` and `../weights`. From the repository-root context used elsewhere, `AGREE_train` is not a top-level module.  
   **Prescribed fix:** Add `cd AGREE` immediately before the command, or otherwise make the required working directory explicit and the block self-contained.

2. **Minor — [src/tests/test_portability.py:16](/home/yixunhu/codespace/ORBITRIR/src/tests/test_portability.py:16)**  
   The tracked portability test retains two author-name literals: `/home/amandine` and `/media/amandine`. These are not covered by the accepted weights-URL exception.  
   **Prescribed fix:** Replace the name-specific sentinels with anonymous generic prefixes such as `/home/` and `/media/`, preserving or strengthening the absolute-path guard.

3. **Nit — stale HAA references after the purge**  
   [test_train_max_steps.py:4](/home/yixunhu/codespace/ORBITRIR/src/tests/test_train_max_steps.py:4) still claims the README contains an HAA finetuning recipe; [yaw_rotation.py:180](/home/yixunhu/codespace/ORBITRIR/src/data/yaw_rotation.py:180) refers to the deleted HAA metadata module; [train.py:78](/home/yixunhu/codespace/ORBITRIR/train.py:78) and [defaults.ini:53](/home/yixunhu/codespace/ORBITRIR/defaults.ini:53) retain recipe-specific HAA comments. Literal searches for the deleted filenames and paths were otherwise clean; remaining AGREE/metrics HAA branches are still-present generic functionality, not dangling imports.  
   **Prescribed fix:** Neutralize these four comments/docstrings to describe generic short-run overrides and the current AR metadata pipeline.

## Original five findings

All are closed:

- DINOv3 tests no longer mutate offline variables during import; the factory import is deferred and teardown restores both variables to their prior values.
- Config leaves and scalar-list members are compared as `(type-name, value)` pairs.
- README uses the integral-column-shift criterion and correctly distinguishes the 64-scene global batch from the context encoder’s folded `64×K` BatchNorm input.
- The generation example passes `batch_size=len(metadata), device=device`.
- Both config twins have normalized whitespace/final newlines; `git diff --check ead8bbd..834dfa1` is clean and their only content differences are the four declared FA additions.

The complete `eval_baselines.py` AR path remains intact: import, argparse restriction, dataset construction, metrics callback, split naming, four-item batch unpacking, loop, and output paths are preserved. `pyproject.toml` remains structurally installable: `orbitrir` is valid, authors are optional, dependencies are unchanged, and setuptools discovery remains configured. The new overview asset is a valid 4000×1791 PNG. The recorded—not rerun—suite state is **243/243 passed at `834dfa1`**.

## Execution-phase gate

**NO-GO** while R4 remains open. After the three focused documentation/anonymity fixes are committed and reverified, the implementation is otherwise a GO for the smoke run, pinned two-cell acceptance, and mandatory guard negative control, subject to the previously identified launch-host data/weights links and functioning two-GPU environment.

## Anonymity-sweep hits

- [download_weights.sh:1](/home/yixunhu/codespace/ORBITRIR/download_weights.sh:1) — `AmandineBtto/FLAC` — **known and accepted open item**, pending the owner’s anonymous weights URL.
- [download_weights.sh:2](/home/yixunhu/codespace/ORBITRIR/download_weights.sh:2) — `AmandineBtto/AGREE` — **known and accepted open item**.
- [src/tests/test_portability.py:16](/home/yixunhu/codespace/ORBITRIR/src/tests/test_portability.py:16) — `/home/amandine` and `/media/amandine` — new Finding 2.
- No remaining hits for `Brunetto`, the old project-page URL, or arXiv `2603.19176`.

## Commands run

```bash
sed -n '1,260p' <R4-review>

git -C /home/yixunhu/codespace/ORBITRIR status --short --branch
git -C /home/yixunhu/codespace/ORBITRIR rev-parse HEAD da370c2 9ce4c4a 16d4459 73d30eb 834dfa1
git -C /home/yixunhu/codespace/ORBITRIR log --oneline --decorate --reverse da370c2..834dfa1
git -C /home/yixunhu/codespace/ORBITRIR diff --name-status da370c2..834dfa1
git -C /home/yixunhu/codespace/ORBITRIR diff --stat da370c2..834dfa1
git -C /home/yixunhu/codespace/ORBITRIR diff --check da370c2..834dfa1
git -C /home/yixunhu/codespace/ORBITRIR diff --check ead8bbd..834dfa1
git -C /home/yixunhu/codespace/ORBITRIR show --format=fuller --find-renames <each of 9ce4c4a,16d4459,73d30eb,834dfa1>
git -C /home/yixunhu/codespace/ORBITRIR show --format=fuller --name-status 834dfa1
git -C /home/yixunhu/codespace/ORBITRIR diff --word-diff=plain 834dfa1^ 834dfa1 -- baselines/eval_baselines.py
git -C /home/yixunhu/codespace/ORBITRIR diff 834dfa1^..834dfa1 -- pyproject.toml

nl -ba <README, eval_baselines.py, pyproject.toml, affected tests>
sed -n <bounded ranges> <README/code/config/worklog files>
diff -u <FLAC_AR.json> <FLAC_AR_FA.json>
file assets/fa_cond.png
strings assets/fa_cond.png | rg <HAA/anonymity patterns>
find <repository/AGREE package-layout paths>
stat -c '%n %s bytes' <README-referenced files and configs>
rg -n <CLI, HAA, deleted-path, section, generation-signature, and anonymity patterns>
git -C /home/yixunhu/codespace/ORBITRIR ls-files -z | xargs -0 rg --with-filename -n -i <patterns>
git -C /home/yixunhu/codespace/ORBITRIR ls-files <deleted HAA path patterns>
```

No Python, tests, installs, environment changes, or repository writes were performed.