"""exp_45 (forked from test_exp38_command_sheets.py) — the two command sheets stay well-formed and carry the current
launch / evaluation / publication commands (plan_rf_data_curve.md r3 §3.2 item: command sheets analogous to
exp_38's and covered by this test)."""
import importlib.util
import os
import re

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(REPO, "worklog", "worklog_yixun", "exp_45_rf_data_curve_claude")
_spec = importlib.util.spec_from_file_location("exp45_registry", os.path.join(EXP, "exp45_registry.py"))
reg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reg)
SHEETS = {
    "rf_data_curve_command.md": [   # the A6000 box: E0, the bridge, the evaluation cells -- no training
        "git -C ~/codespace/FLAC worktree add --detach ~/codespace/exp-45-campaign $X",
        "python $E/exp45_eval.py env-receipt --write", "python $E/exp45_eval.py prepare-e0",
        "python $E/exp45_eval.py run --gpu 1 --cells de_E0a_K8_s42 de_E0b_K8_s42 de_E0c_K8_s42 de_E0d_K8_s42",
        "python $E/exp45_eval.py e0 && python $E/exp45_eval.py e0-check",
        "git add -f $E/rf_data_curve_e0_receipt.json{,.sha256} outputs_FLAC/exp45_rf_data_curve/eval/E0{a,b,c,d}",
        "python $E/exp45_eval.py prepare-bridge", "python $E/exp45_eval.py run --gpu 1 --cells exp45_RF100bridge_K8_s42",
        "python $E/exp45_eval.py bridge --bundle", "python $E/exp45_eval.py prepare $RUN", "python $E/exp45_collect.py"],
    "rf_data_curve_neuronic_commands.md": [
        "git worktree add --detach /n/fs/gatrdp/codespace/exp-45-launch $Y",
        "$PY $E/exp45_registry.py e0-check --platform neuronic", "$PY $E/exp45_registry.py untracked --platform neuronic",
        "$PY $E/exp45_registry.py dry-run --platform neuronic",
        "$PY $E38/exp38_subset_audit.py --verify $E38/data_efficiency_subset_audit.json",
        "$PY $E38/exp38_content_manifest.py --verify $E38/data_efficiency_ar_content_manifest.txt.gz",
        "exp45_submit_neuronic.py submit --run $RUN --mode SMOKE", "exp45_submit_neuronic.py submit --run $RUN --mode FULL",
        "exp45_submit_neuronic.py reconcile", "exp45_submit_neuronic.py abandon",
        "training-check $RUN --run-dir $DST/$RUN.incoming --platform neuronic", "--closure", "--e0-sha",
        "$PY $E/exp45_table.py import-exp38 --commit $B38 --generator-root /n/fs/gatrdp/codespace/FLAC",
        "$PY $E/exp45_table.py build --exp38-commit $B38",
        "$PY worklog/worklog_yixun/gen_model_comparison.py --repo-root /n/fs/gatrdp/codespace/FLAC"],
}


@pytest.mark.parametrize("name", sorted(SHEETS))
def test_the_sheet_is_well_formed_and_current(name):
    text = open(os.path.join(EXP, name)).read()
    fences = [ln for ln in text.split("\n") if ln.startswith("```")]
    assert len(fences) % 2 == 0 and fences, "unbalanced code fences"
    blocks = re.findall(r"```bash\n(.*?)\n```", text, flags=re.S)
    assert blocks and all("```" not in b for b in blocks)
    joined = "\n".join(blocks)
    for cmd in SHEETS[name]:
        assert cmd in joined, cmd
    outside = re.sub(r"```bash\n.*?\n```", "", text, flags=re.S)
    assert not re.search(r"(^|\n)(MODE=|python \$E/|\$PY \$E/)", outside), "a command outside a code block"
    assert "--resume" not in text and "RESUME_STEP=" not in text
    assert "exp38_eval.py" not in joined and "exp38_submit_neuronic.py" not in joined   # the fork's own kit only


def _blocks(name):
    return "\n".join(re.findall(r"```bash\n(.*?)\n```", open(os.path.join(EXP, name)).read(), flags=re.S))


def test_d1_the_a6000_sheet_trains_nothing():
    assert reg.TRAINING_PLATFORM == "neuronic"
    blocks = _blocks("rf_data_curve_command.md")
    assert "MODE=FULL" not in blocks and "MODE=SMOKE_DDP" not in blocks and "exp45_submit_neuronic" not in blocks


def test_d1_the_neuronic_sheet_submits_every_campaign_run_smoke_then_full():
    blocks = _blocks("rf_data_curve_neuronic_commands.md")
    runs = re.search(r'^RUNS="([^"]+)"', blocks, flags=re.M).group(1).split()
    assert sorted(runs) == sorted(reg.campaign_runs()) and len(runs) == 3
    loops = re.findall(r"^for RUN in ([^;\n]+); do\n(.*?)^done$", blocks, flags=re.M | re.S)
    full = [r for head, body in loops if "--mode FULL" in body for r in head.split()]
    smoke = [r for head, body in loops if "--mode SMOKE" in body for r in head.split()]
    assert sorted(full) == sorted(runs) and smoke == ["$RUNS"]
    assert blocks.index("--mode SMOKE") < blocks.index("--mode FULL")
