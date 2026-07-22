"""exp-09 D-stage: pin-gate config-VARIANT delta logic (Codex D-tool F2), CPU-only.

The D eval driver embeds ``assert_arm_configs_exp09.py`` on EVERY invocation, bound to the
ACTUAL config the eval loads. Because the online-eval config's registered delta vs B-F
differs from the base config's (grad-ckpt dropped + use_ema flipped), the gate takes a
``--config-variant {base,online}`` and validates the delta set for that variant.

These tests exercise ONLY the pure JSON delta logic (``expected_config_delta`` /
``assert_config_delta``) — no model construction, no HF cache, no weights. CUDA disabled at
import by the module itself.
"""
import json
import sys
from pathlib import Path

import pytest

_EXP09_DIR = Path(__file__).resolve().parents[1]
if str(_EXP09_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP09_DIR))

gate = pytest.importorskip("assert_arm_configs_exp09")

_BASE = json.loads((_EXP09_DIR / "FLAC_AR_exp09.json").read_text())
_ONLINE = json.loads((_EXP09_DIR / "FLAC_AR_exp09_online_eval.json").read_text())


def test_expected_delta_base_has_no_removed_or_use_ema():
    added, changed, removed = gate.expected_config_delta("base")
    assert set(changed) == {"training.frame_avg_angles"}
    assert removed == set()
    assert set(added) == set(gate.REGISTERED_ADDED_VALUES)


def test_expected_delta_online_adds_use_ema_and_gradckpt_removed():
    added, changed, removed = gate.expected_config_delta("online")
    assert set(changed) == {"training.frame_avg_angles", "training.use_ema"}
    assert changed["training.use_ema"] == (True, False)
    assert removed == {
        "model.conditioning.configs[1].config.gradient_checkpointing",
        "model.conditioning.configs[2].config.gradient_checkpointing",
    }


def test_expected_delta_unknown_variant_raises():
    with pytest.raises(RuntimeError):
        gate.expected_config_delta("bogus")


def test_base_config_passes_base_variant():
    gate.assert_config_delta(_BASE, "base")           # must not raise


def test_online_config_passes_online_variant():
    gate.assert_config_delta(_ONLINE, "online")       # must not raise


def test_base_config_fails_online_variant():
    """Binding the base config as 'online' is a mismatch (no grad-ckpt removed / no use_ema flip)."""
    with pytest.raises(RuntimeError):
        gate.assert_config_delta(_BASE, "online")


def test_online_config_fails_base_variant():
    """Binding the online config as 'base' is a mismatch (unexpected removed + use_ema changed)."""
    with pytest.raises(RuntimeError):
        gate.assert_config_delta(_ONLINE, "base")


def test_default_variant_is_base_backward_compatible():
    """Existing callers (c1_fit/c1_smoke/exp09_launch) pass no variant => base (unchanged)."""
    gate.assert_config_delta(_BASE)                   # default 'base', must not raise


def test_config_variant_cli_flag_present():
    import argparse
    # the flag is registered with the two valid choices
    p = gate.main
    src = Path(gate.__file__).read_text()
    assert "--config-variant" in src and "--config" in src
    assert 'choices=("base", "online")' in src
    assert callable(p) and argparse  # sanity
