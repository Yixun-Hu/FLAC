# Commands — exp_07_fa_scratch

Every command lands here at launch time (SOP). Training/eval launch templates will be appended when runs are commissioned.

## Config-identity audit round (2026-07-10 → 11; code `a3e8cf5`-era worktree, pre-commit)

```bash
# ckpt probe v1 (superseded)  → fa_scratch_2026-07-10_23:35:28_ckpt_probe.log
python worklog/exp_07_fa_scratch_claude/probe_released_ckpt.py 2>&1 | tee worklog/exp_07_fa_scratch_claude/fa_scratch_2026-07-10_23:35:28_ckpt_probe.log

# ⚠ PROVENANCE DEVIATION (post-hoc record; flagged by the gpt-5.6-sol review, Blocking 1):
# the "embedded model_config vs repo" diff appended to the v1 log was produced by an
# INLINE python heredoc (not a checked-in script) run as:
#   python - <<'PY' ... torch.load FLAC.ckpt; flatten; diff vs FLAC_AR.json ... PY 2>&1 | tee -a <v1 log>
# Deviation from the universal-review + command-at-launch rules. Remediation: the diff
# logic now lives IN probe_released_ckpt.py (v2) and the v2 log supersedes the v1 log.

# arm configs (BV byte-copy; BF +2 training keys) + JSON diffs
cp src/configs/model_configs/FLAC/AR/FLAC_AR.json worklog/exp_07_fa_scratch_claude/FLAC_AR_BV.json
python - <<'PY'   # (inline, generative one-liner recorded verbatim; output committed as FLAC_AR_BF.json)
import json, collections
cfg = json.load(open("src/configs/model_configs/FLAC/AR/FLAC_AR.json"), object_pairs_hook=collections.OrderedDict)
cfg["training"]["cond_method"] = "fa_invariant"
cfg["training"]["frame_avg_angles"] = [0.0, 90.0, 180.0, 270.0]
json.dump(cfg, open("worklog/exp_07_fa_scratch_claude/FLAC_AR_BF.json", "w"), indent=4)
PY
diff <(python3 -m json.tool worklog/exp_07_fa_scratch_claude/FLAC_AR_BV.json) <(python3 -m json.tool src/configs/model_configs/FLAC/AR/FLAC_AR.json)
diff <(python3 -m json.tool worklog/exp_07_fa_scratch_claude/FLAC_AR_BV.json) <(python3 -m json.tool worklog/exp_07_fa_scratch_claude/FLAC_AR_BF.json)

# arm asserts v1 (superseded) → fa_scratch_2026-07-10_23:43:26_arm_asserts.log
python worklog/exp_07_fa_scratch_claude/assert_arm_configs.py 2>&1 | tee worklog/exp_07_fa_scratch_claude/fa_scratch_2026-07-10_23:43:26_arm_asserts.log

# ckpt probe v2 (canonical: all counter phases + in-file config diff + DINOv3 pin)
python worklog/exp_07_fa_scratch_claude/probe_released_ckpt.py 2>&1 | tee worklog/exp_07_fa_scratch_claude/fa_scratch_2026-07-10_23:59:06_ckpt_probe_v2.log

# arm asserts v2 (canonical: factory wiring via configure_optimizers + seeded init-identity)
python worklog/exp_07_fa_scratch_claude/assert_arm_configs.py 2>&1 | tee worklog/exp_07_fa_scratch_claude/fa_scratch_2026-07-11_00:00:17_arm_asserts_v2.log
# (an intermediate v2 run at 23:59:25 caught the InverseLR step-0 warmup lr 5e-7 — assert
#  corrected to the closed-form expectation; that log retained as the red→green record)

# BatchNorm rebuttal evidence (Medium 3): 20 BatchNorm2d modules under context_audio.net.cnn.*
python - <<'PY'
import sys, os; sys.path.insert(0, os.getcwd())
import json, torch
from src.models.factory import create_model_from_config
m = create_model_from_config(json.load(open("worklog/exp_07_fa_scratch_claude/FLAC_AR_BV.json")))
print(len([n for n,mod in m.named_modules() if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm)]))
PY

# arm asserts v3 (v2 + fail-closed DINOv3 pin gate assert_vit_pin(); superseded by v4)
python worklog/exp_07_fa_scratch_claude/assert_arm_configs.py 2>&1 | tee worklog/exp_07_fa_scratch_claude/fa_scratch_2026-07-11_00:11:31_arm_asserts_v3.log

# arm asserts v4 (CANONICAL: pin gate env-resolved via huggingface_hub.constants.HF_HUB_CACHE
# + explicit raises surviving `python -O` — reverify2 fixes) with green + 2 red fail-closed tests:
python worklog/exp_07_fa_scratch_claude/assert_arm_configs.py 2>&1 | tee -a <v4 log>            # green, exit 0
HF_HUB_CACHE=$(mktemp -d) python worklog/exp_07_fa_scratch_claude/assert_arm_configs.py         # red 1: empty cache -> RuntimeError, exit 1
HF_HUB_CACHE=$(mktemp -d) python -O worklog/exp_07_fa_scratch_claude/assert_arm_configs.py      # red 2: -O still raises, exit 1
# -> worklog/exp_07_fa_scratch_claude/fa_scratch_2026-07-11_00:16:07_arm_asserts_v4.log

# consolidated review (first gpt-5.6-sol use) + focused re-verify + terse fix-verify
~/.local/bin/codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort=xhigh \
  --output-last-message worklog/exp_07_fa_scratch_claude/fa_scratch_codex_code_audit_probes_review.md "<context-briefed prompt>" < /dev/null
~/.local/bin/codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort=xhigh \
  --output-last-message worklog/exp_07_fa_scratch_claude/fa_scratch_codex_code_audit_probes_reverify.md "<fix-list prompt>" < /dev/null
~/.local/bin/codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort=xhigh \
  --output-last-message worklog/exp_07_fa_scratch_claude/fa_scratch_codex_code_audit_probes_reverify2.md "<residual fix-list prompt>" < /dev/null
```
