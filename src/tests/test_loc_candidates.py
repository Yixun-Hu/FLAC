"""Tests for ``src.localization.candidates`` (exp_18 loc_invert, round 1).

Written test-first (announcement 02). The contracts under test are
``loc_invert_impl_contracts.md`` §4.2 plus the Rev 3 §4 deltas: numeric,
naming-tolerant id parsing (the wav namespace and the metadata-file namespace
are separate -- ``S008_R089_hybrid_IR.wav`` vs ``S008_R0089.json`` -- so every
lookup matches on parsed numeric identity, never on a reconstructed fixed
format), metadata pair JSONs as the candidate authority, and a shallow-copy
metadata variant that swaps only ``source``/``source_vit``.
"""
import pytest

from src.localization.candidates import parse_ir_filename


# --------------------------------------------------------------------------- #
# parse_ir_filename
# --------------------------------------------------------------------------- #
def test_parse_ir_filename_standard():
    assert parse_ir_filename("S008_R089_hybrid_IR.wav") == (8, 89)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("S000_R000_hybrid_IR.wav", (0, 0)),
        ("S010_R0010_hybrid_IR.wav", (10, 10)),      # differing zero-padding
        ("S00100_R7_hybrid_IR.wav", (100, 7)),
        ("S0_R0123_hybrid_IR.wav", (0, 123)),
    ],
)
def test_parse_ir_filename_padding_variants(name, expected):
    """Digit counts vary between rooms/nodes; identity is the parsed integer."""
    assert parse_ir_filename(name) == expected


def test_parse_ir_filename_accepts_full_path():
    got = parse_ir_filename("/data/AR/single_channel_ir_1/Cafe/Cafe_idx_1/S008_R089_hybrid_IR.wav")
    assert got == (8, 89)


@pytest.mark.parametrize(
    "bad",
    [
        "S008_R089.wav",              # not an IR file name
        "S008_R089_hybrid_IR.json",   # wrong extension
        "R089_S008_hybrid_IR.wav",    # swapped roles
        "S008_hybrid_IR.wav",         # missing receiver
        "SXXX_R089_hybrid_IR.wav",    # non-numeric
        "",
    ],
)
def test_parse_ir_filename_malformed_raises(bad):
    with pytest.raises(ValueError):
        parse_ir_filename(bad)
