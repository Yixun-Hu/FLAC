**Reviewer:** OpenAI Codex, GPT-5 (Codex API workspace agent, read-only sandbox; runtime build/reasoning setting not exposed) · **Date:** 2026-08-14  
**Reviewed:** HEAD `6956cbc4b0b2f1fbb88d8b4fd9213b2f77a80f72`

**Overall verdict: REQUEST-CHANGES — do not start `MODE=FULL` at this HEAD.**

1. **BLOCKER — FULL is not pinned to 40,000 steps.** `40000/2500` are only overridable defaults at [are_launch.sh:120](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_16_are_port_claude/are_launch.sh:120) and [are_launch.sh:143](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_16_are_port_claude/are_launch.sh:143). `MODE=FULL MAXSTEPS=1000 CHECKPOINT_EVERY=1` passes the implemented checks, and readback accepts any checkpoint in `(0, MAXSTEPS]` at [are_launch.sh:591](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_16_are_port_claude/are_launch.sh:591). Pin FULL/RESTART to endpoint 40,000, FULL cadence 2,500, and require successful FULL readback `global_step == 40000`. Add adversarial guards; the existing test at [are_launch_guardtests.sh:180](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_16_are_port_claude/are_launch_guardtests.sh:180) only rejects 1,000 because its chosen cadence writes nothing.

2. **BLOCKER — eval add-back is not bound to the checkpoint’s embedded config.** Eval reads the external JSON and checkpoint separately at [eval_FLAC.py:1021](/home/yixunhu/codespace/FLAC/eval_FLAC.py:1021), then resolves λ/anchor solely from that external JSON at [eval_FLAC.py:1069](/home/yixunhu/codespace/FLAC/eval_FLAC.py:1069). A checkpoint trained with one anchor can therefore be evaluated with another while loading cleanly and recording plausible but false provenance. Type-strictly compare the checkpoint’s embedded `model_config` with the file—or source the anchor from the embedded config—before allowing only the explicit λ-dose override. The round-trip test at [test_are_lambda_config.py:552](/home/yixunhu/codespace/FLAC/src/tests/test_are_lambda_config.py:552) proves embedding, not eval enforcement.

3. **HIGH — “guard-test-only” dirty bypass is production-callable, and the fingerprint omits launch-defining inputs.** Any real caller may set `ALLOW_DIRTY_TREATMENT=1` at [are_launch.sh:323](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_16_are_port_claude/are_launch.sh:323); the guard suite explicitly treats this as accepted at [are_launch_guardtests.sh:386](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_16_are_port_claude/are_launch_guardtests.sh:386). Remove the production bypass. Also, `TREATMENT_PATHS` at [stamp_evidence.py:68](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_16_are_port_claude/stamp_evidence.py:68) omits `are_launch.sh`, the dataset config, and `data/AR/train.json`; dirty edits to those can change the run without invalidating evidence. The `are_fit` kind and calibration/VAE SHA checks themselves are correct.

4. **HIGH — the critical algebra regression test covers `u`, not `x_t`.** Tests at [test_are_lambda_config.py:330](/home/yixunhu/codespace/FLAC/src/tests/test_are_lambda_config.py:330) inspect only `targets`; the forbidden implementation “target from residual, noised input from original z” would pass. [Line 326](/home/yixunhu/codespace/FLAC/src/tests/test_are_lambda_config.py:326) is also tautological. Capture the model input and prove both `x_t` and `u` derive from the same residual, with a mutation/non-vacuity case. The full λ=0 step parity and λ=1 negative control at [lines 603–670](/home/yixunhu/codespace/FLAC/src/tests/test_are_lambda_config.py:603) are substantive, not vacuous.

5. **HIGH — time-shift publication is fail-open.** [dataset.py:262](/home/yixunhu/codespace/FLAC/src/data/dataset.py:262) defaults a missing `last_shift` to zero, defeating the downstream hard-error contract if augmentation ordering/publication regresses. Require the attribute explicitly. Existing tests manually supply a shift at [test_are_anchor.py:485](/home/yixunhu/codespace/FLAC/src/tests/test_are_anchor.py:485); add an integration test through the real `RandomTimeShift` proving waveform displacement, metadata value, and unchanged RNG stream.

**Per-file verdicts**

- `src/data/are_anchor.py` — **SHIP**
- `src/training/factory.py` — **SHIP**
- `src/training/diffusion.py` — **SHIP**; current algebra is correct at [diffusion.py:642](/home/yixunhu/codespace/FLAC/src/training/diffusion.py:642), [682](/home/yixunhu/codespace/FLAC/src/training/diffusion.py:682), and [687](/home/yixunhu/codespace/FLAC/src/training/diffusion.py:687).
- `eval_FLAC.py` — **REQUEST-CHANGES**
- `src/data/dataset.py` — **REQUEST-CHANGES**
- `src/data/utils.py` — **SHIP**
- `src/tests/test_are_anchor.py` — **REQUEST-CHANGES**
- `src/tests/test_are_lambda_config.py` — **REQUEST-CHANGES**
- `FLAC_AR_ARE.json` — **SHIP**; exactly BVp1 plus the two ARE keys.
- `calibrate_delta.py` / `are_calibration.json` — **SHIP**: raw median `0.024175 < 0.5`, therefore R1 correctly yields `δ̂=0`; `A_g=0.394574` is the H=32 ℓ2 statistic; R2–R4 are clear.
- `are_launch.sh` — **REQUEST-CHANGES**
- `are_launch_guardtests.sh` — **REQUEST-CHANGES**
- `stamp_evidence.py` — **REQUEST-CHANGES**
- `plan_are_port.md` / params wording — **REQUEST-CHANGES**: P1 is a historical recipe comparator, not a contemporaneous bit-identical training trajectory; retain the disclosure already present in params.

Static AST checks and both shell syntax checks passed. The latest recorded launcher guard run is 67/67, but it lacks the adversarial cases above. Full pytest was not rerun because the active review interpreter lacks Lightning/torchaudio, and I did not switch or modify environments. No files or jobs were changed.