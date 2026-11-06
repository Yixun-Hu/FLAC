**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-05 · **Round:** code + plan A7-C3 (exp_22 R2: the PIN2 weights-layout fix 44c2241b → 233e2bbb, plan A7 r6, the worktree script)

**Verdict:** SHIP

No BLOCKING or HIGH findings. The branch may be re-merged into `localization-exp`; that new merge commit becomes PIN2 under A7 r6. `b722d5f0` remains superseded.

## Findings

- **NIT 1 — generic worktree-path guard is weaker than its safety claim.** `exp22_pin2_worktree.sh:19-20` allows a path beneath `/home/yixunhu/codespace/FLAC/`; `git worktree add` could then create a nested worktree inside the main checkout, contrary to lines 9-10’s “never writes a file of the main checkout.” The prescribed `/home/yixunhu/codespace/exp-22-pin2` is safely outside it. Optional fix: canonicalize `WT` with `realpath -m` and reject equality with or descendants of `MAIN`.

- **NIT 2 — `verify` validates dynamic links, not the launcher’s complete required-asset set.** `exp22_pin2_worktree.sh:23-25,53-62` can report `VERIFY PASS` if, for example, `VAE.ckpt` has disappeared from the main checkout—then it simply drops out of `ENTRIES`. It also does not explicitly require `AcousticRooms/single_channel_ir_1`. This is non-blocking because the launcher independently refuses both at `exp22_r2c_preflight.sh:83-89`, and the real-worktree hand check proves both currently exist. Optional fix: add explicit `REQUIRED_WEIGHT_ASSETS` and `REQUIRED_ROOM_DIRS` loops mirroring the launcher.

## F1 — RESOLVED

The fix implements A7 r6 correctly:

- `exp22_r2c_preflight.sh:82` requires `weights/` to be a directory and not a symbolic link.
- Lines 83-86 require every registered weight to be a regular file and `-ef`-identical to the corresponding main-checkout file. A missing target, dangling link, ordinary copy, or link to another file cannot pass vacuously.
- The six-entry enumeration at lines 53-54 is complete:

  - The authoritative line list supplies exactly `YAW_40k.ckpt`, `BF_40k.ckpt`, and `P1_40k.ckpt`.
  - `FLAC_AR.json:166` supplies `weights/AGREE/AGREE_fullAR.pt`.
  - `dinoV3.json:20`, `AGREE/AGREE/model.py:126-138`, and `audio_model.py:176-187` show that every AGREE construction opens `weights/FLAC/VAE.ckpt`.
  - The selected real-engine T tests use `_FLAC_CKPT = weights/FLAC/FLAC_EMA.ckpt` at `test_eval_localization.py:994`.
  - The selected `_tiny_module` tests at `test_eval_localization.py:1904-1932` are random-init and use no pretrained weight.
  - `AGREE_AR.pt` belongs to the unselected scorer-noise integration tests at `test_eval_localization.py:2774-2804`; it is not required by the eleven registered nodes.
  - The DINOv3 cache is not under `weights/` and remains independently checked at launcher line 91.
  - The `resolve_agree_checkpoint` basename fallback does not create an omitted admissible path: the launcher requires the configured `weights/AGREE/AGREE_fullAR.pt`, so a passing launch resolves through the configured path.

`AcousticRooms/single_channel_ir_1` at lines 55-56 closes the missing real-data directory check.

## F2 — PASS

The tests exercise the relevant real topology rather than passing solely through their own fake:

- `conftest.py:226-258` can commit a tracked file under `weights/`.
- `test_exp22_r2c_preflight_sh.py:43-63` creates a distinct stand-in main checkout and per-entry links while retaining the sandbox’s tracked `weights/FLAC` file.
- Lines 145-198 cover the whole-directory link, missing target behind a linked directory, missing per-file links, byte-identical copy, wrong-file link, missing VAE, and missing `single_channel_ir_1`.
- Lines 201-237 pin the passing tracked-directory layout, real repository tracking under `weights/`, launcher-array equality, and the derived six-asset set.

The static enumeration test is necessarily narrower than the full call-graph proof, but the independent source trace above confirms its assumptions.

## F3 — PASS

For the prescribed worktree and move procedure, the script is correct:

- `create` writes the detached worktree plus Git’s `.git/worktrees/` metadata; it does not modify a file in the main checkout.
- `verify` checks PIN2, detached HEAD, tracked cleanliness, a real `weights/`, the two top-level links, all 23 current weight-entry links and their inode identity, and exact untracked-entry equality (`exp22_pin2_worktree.sh:43-63`).
- The real worktree is detached at superseded `b722d5f0`, has zero tracked-status lines, contains the ten tracked metric JSON files, and presently satisfies all six new `-ef` checks plus `single_channel_ir_1`.
- Because `b722d5f0 → 233e2bbb` has no non-worklog delta and no tracked-weight change, `git checkout --detach <new-PIN2>` preserves the untracked links. Running `verify <new-PIN2> ~/codespace/exp-22-pin2` afterward is the correct transition.

The two script hardening nits above do not invalidate this procedure.

## F4 — RESOLVED

A7 r6 and the implementation agree. The added `AcousticRooms/single_channel_ir_1` check is within the amendment’s “worktree’s asset layout” scope and enforces an already registered T input; it changes no line, argument, estimand, threshold, or evidence rule. No addendum is required.

## F5 — PASS

Regression/freeze audit passed:

- `git diff --name-only b722d5f0 233e2bbb -- ':(exclude)worklog'` is empty.
- Against `9847b45d`, the non-worklog delta is exactly the four PIN2 paths, with hashes `0d0e52ca…`, `bbd2847d…`, `d61a3c02…`, and `6422eae5…`.
- `localize_meshgrid.py` and `src/localization/` are identical to `701926e5`.
- The detector, capture/launch protocol, supervisor, line list, T admission rule, and backend rule are unchanged except for the reviewed layout checks.
- `git diff --check 44c2241b 233e2bbb -- '*.py' '*.sh' '*.json'` is clean.
- Both commits contain the expected Opus trailer.

Audit qualification: the stated “production +10/−5, tests +72/−15” counts are not raw Git counts. `git diff --numstat` reports launcher `+21/−5`, test module `+85/−13`, and `conftest.py` `+4/−3`. The smaller numbers appear to use an undocumented comment/structural-line filter; this is bookkeeping only.

## F6 — SHIP

Re-merge `exp-22-a7-noflash` into `localization-exp` and designate that merge commit as PIN2. Then move the existing detached worktree to it and run the Planner’s `verify` mode before any launcher use.

## Static probes performed

I used only read-only probes: `git log/show/rev-list/status/ls-files/diff --name-status/--numstat/--check/--exit-code`, `git grep`, `sed`, `jq`, `sha256sum`, `readlink -f`, and direct `test -d`, `test -L`, `test -f`, and `test -ef` expressions.

Verified exact hashes for the Planner script and four archived logs, including the new hand-check log `d63efd3d…`; verified the c19 records as `4c074f2b…`, `444372ab…`, and `bfb912ab…`. I did not execute any test, launcher, detector, supervisor, worktree script, GPU command, or reviewed reproduction script, and did not inspect reservations or processes.
