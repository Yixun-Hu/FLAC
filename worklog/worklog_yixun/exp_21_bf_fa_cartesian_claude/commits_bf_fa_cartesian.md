# commits_bf_fa_cartesian (exp_21)

| SHA | Description |
|---|---|
| `ebb8166` | exp_21: TDD test suite for fa_cartesian_conditioning (27 tests; red-phase verified) — src/tests/test_fa_cartesian.py (+542) |
| `eeed40e` | exp_21: fa_cartesian_conditioning — full-C4 Cartesian frame averaging over all four POSE_KEYS, fail-closed depth/id contracts — src/data/yaw_rotation.py (+160, additive) |
| `328ee5e` | exp_21: r1-nit tests — shared-projection autograd regression + edge-contract pins — test_fa_cartesian.py (+133) |
| `be1e758` | exp_21: TDD red phase — dispatch/factory/config-parity suite — test_fa_cartesian_dispatch.py (+683) |
| `718aff9` | exp_21: training dispatch + widened yaw-aug guards + FLAC_AR_BFC.json (D5 cap 32) — diffusion.py, factory.py (+50/−17 non-test) |
| `c0ada57` | exp_21: r2 review fixes — isolate factory yaw-aug pin (mutation-verified), correct C512-vs-C4 rationale, red-phase docstring |
| `913e60b` | exp_21: TDD red phase — eval-side suite (suffix, provenance schema, guard) — test_fa_cartesian_eval.py (+574) |
| `61de4aa` | exp_21: eval_FLAC fa_cartesian first-class — constants + 4 widened sites + dispatch elif (+51/−11) |
| `36faab9` | exp_21: eval_pl fail-closed fa_cartesian guard before construction (+43) |
| `645d8d4` | exp_21: real-batch dispatch call-shape pins (self-caught empty-dataloader vacuity) (+121/−1) |
| `a15c6fd` | exp_21: TDD red — r3 nits n1/n2/n3 (full fa_invariant call-shape pin, eval_pl allowlist, over-cap pre-flight) — test_fa_cartesian_eval.py (+129/−12); 7 failed / 28 passed |
| `0353954` | exp_21: r3 nits green — eval_pl PERMITTED_COND_METHODS allowlist (+41/−24), eval_FLAC batch-vs-cap pre-flight naming --batch-size/--frame-avg-max-fwd-samples (+21) |
| `8c31bc2` | exp_21: bfc_launch.sh — the BFC training launcher: B-F's recipe flag-for-flag + modern gates + init-identity audit + DRY_RUN (+379) |
| `74f1c2c` | exp_21: bfc_launch_guardtests.sh — 67 cases over every gate branch and the LAUNCH-CMD flag pins (+283); caught the grad-ckpt check ordering |
| `c91e329` | exp_21: TDD red — model-comparison table staging — test_exp21_table_gate.py (+518); 45 failed / 3 passed / 11 errors |
| `746a157` | exp_21: table staging green — exp21_validate_cell.py (+193), gen_model_comparison.py BFC rows + admission gate + two-K transaction (+222/−4, additive) |
| `23888ad` | exp_21: round-4 commit log + full-suite evidence (1999 passed; the one failure is exp_11/exp_15's pre-existing registry drift) |
| `21d423b` | exp_21: r4 BLOCKING 1 — the registered manifest is pinned, not defaulted; SMOKE=1 is the one sanctioned short mode; DRY_RUN/SMOKE fail closed outside {0,1}; guardtests gain 11 rejection cases + exact 39-token argv comparison (65/0) |
| `a8c2a5b` | exp_21: r4 BLOCKING 2+3 — split-derived per-scene evidence + exact-step/one-ckpt identity in the table gate |
| `58c99bb` | exp_21: r5 BLOCKING 1+2 — trained-as binding (embedded model_config, type-strict, before model/GPU construction) + streamed ckpt_sha256; both recorded beside what they prove (eval_FLAC +211/−9; test_exp21_ckpt_binding.py +380) |
| `08e3e08` | exp_21: r5 BLOCKING 3+4 — required stream sidecar w/ re-run positional check, required+uniform digest, exp21_protocol.py campaign definition (+404), bfc_eval_driver.sh (+162), D6 comparator rows + cross-arm one-pin transaction (gen +272/−12); fixes is_exp21_row mislabelling the repin rows legacy-loop |
| `58c99bb` | exp_21: F1+F2 — trained-as binding before construction + streamed ckpt_sha256 in all records |
| `08e3e08` | exp_21: F3+F4 — durable full-split proof + D6 protocol module/driver/repin rows/cross-arm pin transaction |
| `5a256b1` | exp_21: r5 nits — guardtest-only floor bypass, docstring fixes, commits-log dedupe |
| `cc13656` | exp_21: r5 evidence logs |
| `41f11b0` | exp_21: r5 re-review BLOCKING 1-4 — cross-arm gate made transactional (all six arm×K rows, five seeds each), per-(K,seed) input identity recomputed from sidecar preimages, reviewed comparator digests pinned + enforced across both K, preflight fails closed on digest mismatch, `.fullmatch()` for both 64-hex checks (gen +125/−23, protocol +50/−13, validator +42/−2, eval_FLAC +6/−2; tests +218/−30) |
