# Codex code review — exp_03, round: cyl_pose (TDD cycles 1–2)

**Reviewer:** OpenAI Codex, model `gpt-5.5` at Extra High (`xhigh`) reasoning effort (codex-cli 0.142.5, `codex exec`, read-only sandbox, context-briefed per SOP) · **Date:** 2026-07-05
**Target:** commits `031852d` (RED) + `a56e5e7` (GREEN)

**Verdict: REQUEST-CHANGES**

1. **Medium: all-degenerate fallback does not honor the `eps` contract.**  
   [src/data/yaw_rotation.py](/home/yixunhu/codespace/FLAC/src/data/yaw_rotation.py:116) picks the largest-radius pose whenever candidates exist, even if every candidate has `r < eps`. The approved plan says all poses degenerate ⇒ `Delta phi == 0`. With tiny nonzero radii below `1e-6`, current code returns arbitrary nonzero `dphi` from undefined azimuths. The test only covers exactly-zero all-degenerate poses at [src/tests/test_yaw_symmetry.py](/home/yixunhu/codespace/FLAC/src/tests/test_yaw_symmetry.py:141), so it misses this branch. Fix by requiring `all_r.max() >= eps` before using the largest-r azimuth; otherwise set `phi_ref = None` and add a below-eps nonzero all-degenerate test.

Notes: `wrap_angle` is correct for the requested boundary convention: `-pi`, `pi`, `3pi`, and `-3pi` map to `+pi`, and `-0.0` maps to `+0.0`; tensor dtype is preserved, including bf16/fp16 in the smoke check. Feature order is `(r, z, dphi)`. `rotate_scene_metadata` default behavior is code-identical to the pre-change path except for the optional `pose_keys` loop input, and the restricted path leaves unlisted pose tensors untouched. Targeted tests passed with `pytest -s -p no:cacheprovider src/tests/test_yaw_symmetry.py -q` (`10 passed`); normal capture was blocked by the read-only sandbox lacking a usable temp directory.

Safe to keep building on these functions? **No, fix the all-degenerate `eps` fallback first.**
---
**Disposition (Fable 5):** Blocking per round-closure rule. Fix dispatched to the Opus coder: require max r >= eps before using the largest-r azimuth (else Δφ ≡ 0), plus a below-eps nonzero all-degenerate invariance test. Note: review ran 4 attempts — attempts 1–3 stalled on a stdin-blocking infra issue (codex exec in background shells needs `< /dev/null`), diagnosed via live monitor; logged as infrastructure, not a code bug.
