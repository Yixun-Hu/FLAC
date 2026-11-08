"""exp_38 — the two command sheets stay well-formed and carry the current launch / publication commands
(Codex code r3 NIT 2: the A6000 sheet was once corrupted by an edit script)."""
import os
import re

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(REPO, "worklog", "worklog_yixun", "exp_38_data_efficiency_claude")
SHEETS = {
    "data_efficiency_command.md": [
        "python $E/exp38_eval.py env-receipt --write", "python $E/exp38_eval.py prepare-e0",
        "python $E/exp38_eval.py e0", "python $E/exp38_content_manifest.py --verify $E/data_efficiency_ar_content_manifest.txt.gz",
        "MODE=SMOKE_DDP RUN=$RUN EXPECT_SHA=$PIN bash $E/exp38_train_local.sh",
        "MODE=FULL      RUN=$RUN EXPECT_SHA=$PIN setsid nohup bash $E/exp38_train_local.sh",
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
