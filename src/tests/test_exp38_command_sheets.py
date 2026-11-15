"""exp_38 — the two command sheets stay well-formed and carry the current launch / publication commands
(Codex code r3 NIT 2: the A6000 sheet was once corrupted by an edit script)."""
import importlib.util
import os
import re

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(REPO, "worklog", "worklog_yixun", "exp_38_data_efficiency_claude")
_spec = importlib.util.spec_from_file_location("exp38_registry", os.path.join(EXP, "exp38_registry.py"))
reg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reg)
SHEETS = {
    "data_efficiency_command.md": [   # D2 = neuronic: this box runs E0 + the evaluation cells only
        "git -C ~/codespace/exp-38-campaign checkout --detach $PIN",
        "mv $E/data_efficiency_e0_receipt.json{,.sha256} ~/codespace/exp-38-archive/",
        "python $E/exp38_eval.py env-receipt --check", "python $E/exp38_eval.py prepare-e0",
        "python $E/exp38_eval.py run --gpu 1 --cells de_E0a_K8_s42 de_E0b_K8_s42 de_E0c_K8_s42 de_E0d_K8_s42",
        "python $E/exp38_eval.py e0 ", "python $E/exp38_eval.py e0-check",
        "python $E/exp38_content_manifest.py --verify $E/data_efficiency_ar_content_manifest.txt.gz",
        "python $E/exp38_eval.py prepare $RUN", "python $E/exp38_collect.py"],
    "data_efficiency_neuronic_commands.md": [
        "exp38_submit_neuronic.py submit --run $RUN --mode SMOKE", "exp38_submit_neuronic.py submit --run $RUN --mode FULL",
        "exp38_submit_neuronic.py reconcile", "exp38_submit_neuronic.py abandon",
        "training-check $RUN --run-dir $DST/$RUN.incoming --platform neuronic", "--closure", "--e0-sha",
        "data_efficiency_e0_receipt.json{,.sha256}", "outputs_FLAC/exp38_data_efficiency/eval/E0{a,b,c,d}",
        "exp38_content_manifest.py --verify"],
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
    assert "--resume" not in text and "RESUME_STEP=" not in text.replace("RESUME_STEP=2500", "")


def _blocks(name):
    return "\n".join(re.findall(r"```bash\n(.*?)\n```", open(os.path.join(EXP, name)).read(), flags=re.S))


def test_d2_the_a6000_sheet_trains_nothing():
    assert reg.TRAINING_PLATFORM == "neuronic"
    assert "MODE=FULL" not in _blocks("data_efficiency_command.md")
    assert "MODE=SMOKE_DDP" not in _blocks("data_efficiency_command.md")


def test_d2_the_neuronic_sheet_submits_every_campaign_run_smoke_then_full():
    blocks = _blocks("data_efficiency_neuronic_commands.md")
    runs = re.search(r'^RUNS="([^"]+)"', blocks, flags=re.M).group(1).split()
    assert sorted(runs) == sorted(reg.campaign_runs()) and len(runs) == 7 and "de_p1_f025" in runs
    loops = re.findall(r"^for RUN in ([^;\n]+); do\n(.*?)^done$", blocks, flags=re.M | re.S)
    full = [r for head, body in loops if "--mode FULL" in body for r in head.split()]
    smoke = [r for head, body in loops if "--mode SMOKE" in body for r in head.split()]
    assert sorted(full) == sorted(runs) and smoke == ["$RUNS"]
