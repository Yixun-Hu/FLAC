**Reviewer:** OpenAI Codex (gpt-5.6-sol, `codex exec`, read-only sandbox) · **Date:** 2026-08-31
**VERDICT:** REQUEST-CHANGES

## Findings

1. **Major — [.gitignore:174](/home/yixunhu/codespace/ORBITRIR/.gitignore:174)**

   - **What:** The four runtime-directory patterns are unanchored. In particular, `HAA/` also ignores untracked files beneath the tracked `data/HAA/` directory.
   - **Why:** New splits, preparation utilities, or other reproducibility assets under `data/HAA/` can silently disappear from `git status`. `git check-ignore -v --no-index data/HAA/review_probe.json` confirmed the match comes from `.gitignore:175`. The file also lacks a final newline.
   - **Prescribed fix:** Change the patterns to `/AcousticRooms/`, `/HAA/`, `/weights/`, and `/outputs_FLAC/`, and add the final newline.

2. **Minor — [README.md:103](/home/yixunhu/codespace/ORBITRIR/README.md:103), [README.md:194](/home/yixunhu/codespace/ORBITRIR/README.md:194), [README.md:227](/home/yixunhu/codespace/ORBITRIR/README.md:227), [README.md:266](/home/yixunhu/codespace/ORBITRIR/README.md:266), [README.md:290](/home/yixunhu/codespace/ORBITRIR/README.md:290)**

   - **What:** The AR baseline example passes unsupported `--output-dir`; the parser defines `--out-dir`. Several command blocks also contain missing continuation backslashes or backslashes followed by spaces, while lines 140, 158, and 227 have unnecessary terminal continuations.
   - **Why:** Copying these examples either produces an argparse error or executes only the first fragment and treats subsequent options as shell commands. In the `eval_pl.py` block, repairing continuation while retaining `--store-predictions False` would additionally enable prediction storage because `"False"` is truthy.
   - **Prescribed fix:** Use `--out-dir`, normalize every affected shell block so only non-final lines end in a space-free `\`, remove unnecessary final continuations, and remove `--store-predictions False` from the README until C7c supplies correct Boolean CLI semantics.

3. **Nit — [FLAC_AR.json:60](/home/yixunhu/codespace/ORBITRIR/src/configs/model_configs/FLAC/AR/FLAC_AR.json:60)**

   - **What:** The changed Hub-ID lines retain trailing spaces at lines 60 and 76 in all five AR model variants and `FLAC_HAA_finetune.json`.
   - **Why:** `git diff --check ead8bbd..dd43998` reports all twelve lines, so the SOP’s mandatory static whitespace check is not clean.
   - **Prescribed fix:** Strip those trailing spaces in the R1 follow-up and record them as formatting-only deviations from `0bd5da0`; re-run `git diff --check`.

## Deferred-item adjudication

1. **(i) Fix in an R1 follow-up now.** Anchor all four ignore patterns and terminate `.gitignore` with a newline. This is Finding 1.

2. **(ii) Fix in an R1 follow-up now.** Correct the baseline flag and normalize the README commands comprehensively. Remove the misleading `--store-predictions False` example without changing `eval_pl.py` yet. This is Finding 2.

3. **(iii) Defer as planned to C7a.** Keeping `_rot{int(deg)}` in C2 preserves the required byte parity and registered naming. C7a should introduce collision-free formatting and tests proving values such as `37.3` and `37.9` produce distinct paths.

4. **(iv) Defer the code fix as planned to C7c.** That commit already owns `eval_pl.py` protocol/CLI work. It should implement unambiguous Boolean semantics rather than accepting the truthy string `"False"`.

## Parity verification

- `eval_FLAC.py` is byte-identical to the fork base. Command run, exit status 0:

  ```bash
  cmp --silent <(git -C /home/yixunhu/codespace/FLAC show 0bd5da0:eval_FLAC.py) <(git -C /home/yixunhu/codespace/ORBITRIR show dd43998:eval_FLAC.py)
  ```

- The yaw module has only the enumerated omissions and permitted docstring edits. Command run:

  ```bash
  diff -u <(git -C /home/yixunhu/codespace/FLAC show f59f5a4:src/data/yaw_rotation.py) <(git -C /home/yixunhu/codespace/ORBITRIR show dd43998:src/data/yaw_rotation.py)
  ```

  Its output contains only omission of `DEFAULT_FRAME_ANGLES`, `wrap_angle`, `cylindrical_pose_features`, `invariant_conditioning`, and `yaw_transform_consistency`, plus the two approved docstring changes.

- I also ran `cmp --silent` against `0bd5da0` separately for all 19 C1 configuration files; all matched byte-for-byte.

No C2 implementation defect was found in the rotation sign, quantisation, pose/depth co-rotation, or non-mutation behavior. Per instruction, I did not execute Python or the test suite; the recorded post-C2 state is 11 passed.