**Reviewer:** OpenAI Codex `gpt-5.6-sol` at Extra High reasoning (codex-cli 0.147.0, read-only) · **Date:** 2026-08-20

*Round marker: r7 (waveform dump + replay). APPROVE-WITH-CHANGES; R2b GO under 3 conditions; replay GO. Body verbatim.*

---

**Verdict: APPROVE-WITH-CHANGES.** No launch-blocking result-corruption defect was found. The conditions concern safe NAS directory use and later R4 composition.

- `[PASS — dump exactness]` The dumped `pred` is the same decoded-and-clamped tensor passed to the scorer: generation clamps at [eval_localization.py:370](/home/yixunhu/codespace/FLAC/eval_localization.py:370), scoring consumes it at [eval_localization.py:374](/home/yixunhu/codespace/FLAC/eval_localization.py:374), and dumping snapshots that same object at [eval_localization.py:1480](/home/yixunhu/codespace/FLAC/eval_localization.py:1480). The float32 cast matches the scorer’s own preprocessing cast at [agree_embed.py:49](/home/yixunhu/codespace/FLAC/src/localization/agree_embed.py:49). Real observations are pad-cropped and clamped by the loader before scoring. No inference-mode, dtype, copy, or ordering hazard found.

- `[PASS — NPZ hashing]` In the current Linux/NumPy path, `np.savez(pred=…, obs=…)` is byte-deterministic for identical arrays: fixed entry order, uncompressed storage, and fixed 1980 ZIP timestamps. The real smoke confirms this and its recorded SHA values match the NAS files. Treat the SHA as an artifact-integrity digest, not a guaranteed canonical hash across unrelated platforms/NumPy implementations.

- `[LOW — replay preflight]` Exact float-hex comparison is correct, missing queries abort, and `_replay` makes the output artifact stem distinct at [eval_localization.py:879](/home/yixunhu/codespace/FLAC/eval_localization.py:879) and [eval_localization.py:2605](/home/yixunhu/codespace/FLAC/eval_localization.py:2605). I independently found 6,337 unique query IDs in each of the three R2 row files. Protocol mismatches are not explicitly compared against source-row provenance; some therefore fail only on the first differing similarity. For registered R2, the pre-model registration gate at [eval_localization.py:1698](/home/yixunhu/codespace/FLAC/eval_localization.py:1698) removes most risk, and a late mismatch cannot publish final replay artifacts. This is wasted work, not silent corruption. A later hardening should preflight reference cardinality/uniqueness and provenance.

- `[MEDIUM — NAS overwrite semantics]` Fresh directories are safe: each NPZ uses same-directory temporary-write plus replace, and the waveform manifest is published only after every query and end gate at [eval_localization.py:1589](/home/yixunhu/codespace/FLAC/eval_localization.py:1589) and [eval_localization.py:1642](/home/yixunhu/codespace/FLAC/eval_localization.py:1642). A crash during a fresh run leaves files but no completion manifest. However, `--overwrite` on a previously completed directory is not transaction-safe: a crash can leave the old manifest beside a mixture of newly replaced and old NPZ files. Two concurrent processes sharing one nominally empty directory can also race. Use a new, seed-specific cell directory for every launch, never share it, and do not use `--overwrite` on a completed dump.

- `[PASS — decode-all]` It genuinely covers every present split file and applies the loader’s `is_silence` criterion at [eval_localization.py:2239](/home/yixunhu/codespace/FLAC/eval_localization.py:2239) and [eval_localization.py:2275](/home/yixunhu/codespace/FLAC/eval_localization.py:2275), so it catches the globally near-silent position-1194 class. Runtime is linear over 6,337 small WAV reads with only per-room metadata retained—reasonable for a CIFS preflight and negligible beside generation. The in-loop identity gate remains authoritative because readback tests the full WAV rather than exactly emulating the loader’s first-10,240 window.

- `[PASS — scoring regression]` The generation, clamp, embed, and similarity lines all predate r7; r7 only retains the already-scored tensor conditionally and copies it after scoring. Without dumping, the old path is unchanged. The pre-r7 parity evidence therefore remains applicable.

- `[R4-COMPOSITION GUARD]` Exact AGREE-sim replay authenticates only the scorer-visible first 8,000 samples; it cannot independently prove that samples 8,000–9,600 used by future R4 metrics match the historical unsaved waveform. When R4 is composed, dump and metrics must consume one immutable float32 snapshot of `outcome["wavs"]`, with no in-place metric operations or re-decoding, and the tail should be described as deterministic replay rather than independently verified historical bytes.

Launch calls:

- **R2b with dumps:** GO after the pending GPU test and one real full `--readback-decode-all` pass are green; use fresh per-seed NAS cell directories and no `--overwrite`.
- **R2 replay back-fill:** GO under the original R2 registration manifest/SHA, batch-size 4, workers 4, preferably the original device mapping, with fresh per-seed replay dump directories and hard abort on any mismatch.

Review scope is pinned to `d6e531d`; concurrent `08d4755`/`837ed46` do not modify the reviewed r7 files. Per the strict read-only instruction, I did not save a review file or rerun tests.