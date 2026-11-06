"""exp_22 A7 (Codex A7-C1 NIT 1): every tools source -- code, tests, data -- ends with exactly one newline and carries
no trailing whitespace (what ``git diff --check`` reports for source files)."""
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = sorted(n for n in os.listdir(HERE) if n.endswith((".py", ".sh", ".json")))


def test_the_tools_sources_are_listed():
    assert {"exp22_r2c_supervisor.py", "test_exp22_r2c_card_rule.py", "exp22_r2c_preflight.sh",
            "exp22_r2c_preflight_lines.json", "conftest.py"} <= set(SOURCES)


@pytest.mark.parametrize("name", SOURCES)
def test_one_final_newline_and_no_trailing_whitespace(name):
    with open(os.path.join(HERE, name), "rb") as f:
        raw = f.read()
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n"), f"{name}: not exactly one final newline"
    assert not [i for i, ln in enumerate(raw.split(b"\n"), 1) if ln != ln.rstrip()], f"{name}: trailing whitespace"
