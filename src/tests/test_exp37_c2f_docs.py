"""exp_37 final fix round (Codex code review 3, N-02): the c2f documentation states the binding policy (plan §12 D-a /
D-c and the fix-round-2 c2f-parity choice): the c2f point's accuracy comes from SEQUENTIAL c2f runs for all three
seeds (the seed-42 timing pass with its replay, the seed-43 / 44 accuracy runs b2a_c2f_*); the parent-derived offline
readout of the 0.5 m runs is a parity cross-check / sensitivity row only, and seeds 43 / 44 are accuracy-only (never
timing repeats). CPU. Written by the exp_37 Coder (Claude Opus 5.5, max effort), final fix round: fixed after review
3, not re-reviewed (cap).
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _docstring(path):
    text = (REPO / path).read_text()
    start = text.index('"""')
    return " ".join(text[start + 3:text.index('"""', start + 3)].split())


def test_the_c2f_module_names_the_sequential_accuracy_of_record():
    doc = _docstring("src/localization/exp37_c2f.py")
    assert "ACCURACY of record is the offline readout" not in doc
    assert "sequential c2f runs for all three seeds" in doc and "cross-check" in doc


def test_the_aggregator_describes_the_three_seed_sequential_c2f_policy():
    doc = _docstring("tools/exp37_aggregate_b2b3.py")
    assert "43 / 44 = repeats" not in doc and "0.5 m runs give the 0.5 m, nested-1.0 m and c2f readouts" not in doc
    assert "sequential c2f runs for all three seeds" in doc and "accuracy-only" in doc and "parity" in doc
