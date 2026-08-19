"""Candidate-set construction for exp_18 (analysis-by-synthesis source localization).

The AcousticRooms wav namespace (``S008_R089_hybrid_IR.wav``) and the metadata
namespace (``S008_R0089.json``) use different zero-padding conventions, so every
identity here is the *parsed integer* node id; file lookups scan the directory
listing and match numerically instead of reconstructing one fixed name format.
"""
import os
import re

_IR_NAME_RE = re.compile(r"^S(\d+)_R(\d+)_hybrid_IR\.wav$")


def parse_ir_filename(name):
    """Return ``(src_node, rec_node)`` parsed from an AR IR file name.

    ``name`` may be a bare file name or a full path. Raises ``ValueError`` for
    anything that is not an ``S<digits>_R<digits>_hybrid_IR.wav`` name.
    """
    base = os.path.basename(str(name))
    match = _IR_NAME_RE.match(base)
    if match is None:
        raise ValueError(f"not an AR IR file name: {name!r}")
    return int(match.group(1)), int(match.group(2))
