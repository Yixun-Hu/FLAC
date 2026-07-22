"""exp-09 D-stage: FLAC_AR_exp09_online_eval.json config-delta test.

The eval-time variant is derived from ``FLAC_AR_exp09.json`` by applying EXACTLY the
same delta that ``FLAC_AR_BF_online_eval.json`` applies to ``FLAC_AR_BF.json`` (the
B-F pattern, verified structurally in this repo):

    * REMOVE ``gradient_checkpointing`` (value ``true``) from the ``source_vit``
      conditioner (configs[1].config) and the ``context_poses_vit`` conditioner
      (configs[2].config) -- eval has no backward pass, so checkpointing is dropped;
    * FLIP ``training.use_ema`` true -> false -- the "online" (non-EMA) weights variant
      (the EMA pass reuses the base config; see exp09_screen.sh / bf_screen.sh).

Nothing else may change: the cylindrical backbone (implementation/gauge), the
fa_invariant[0.0] one-pass conditioning, and every other field stay byte-identical.
CPU-only; pure JSON structural comparison (no model construction).
"""
import json
from pathlib import Path

_EXP09_DIR = Path(__file__).resolve().parents[1]
_BASE = _EXP09_DIR / "FLAC_AR_exp09.json"
_ONLINE = _EXP09_DIR / "FLAC_AR_exp09_online_eval.json"

# The registered eval-time delta (BF pattern applied to exp-09).
_REMOVED_KEYS = {
    "model.conditioning.configs[1].config.gradient_checkpointing",
    "model.conditioning.configs[2].config.gradient_checkpointing",
}
_CHANGED = {"training.use_ema": (True, False)}


def _flatten(obj, prefix=""):
    flat = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            flat.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            flat.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        flat[prefix] = obj
    return flat


def _load(p):
    return json.loads(p.read_text())


def test_base_config_present():
    assert _BASE.exists(), "FLAC_AR_exp09.json (base) must exist"


def test_online_eval_config_present():
    assert _ONLINE.exists(), "FLAC_AR_exp09_online_eval.json must be created (D-stage deliverable)"


def test_online_eval_is_valid_json_object():
    obj = _load(_ONLINE)
    assert isinstance(obj, dict) and obj.get("model_type") == "diffusion_cond"


def test_online_eval_delta_is_exactly_registered():
    """The ONLY differences from the base config are the two removed
    gradient_checkpointing keys and the use_ema flip -- nothing else."""
    base = _flatten(_load(_BASE))
    online = _flatten(_load(_ONLINE))

    removed = {k: base[k] for k in base if k not in online}
    added = {k: online[k] for k in online if k not in base}
    changed = {k: (base[k], online[k]) for k in base if k in online and base[k] != online[k]}

    assert set(removed) == _REMOVED_KEYS, f"unexpected removed keys: {removed}"
    assert all(v is True for v in removed.values()), f"removed keys were not true: {removed}"
    assert added == {}, f"online-eval must not ADD keys: {added}"
    assert changed == _CHANGED, f"unexpected value changes: {changed}"


def test_online_eval_drops_grad_checkpointing_on_both_vit_conditioners():
    online = _load(_ONLINE)
    for idx in (1, 2):
        cfg = online["model"]["conditioning"]["configs"][idx]["config"]
        assert "gradient_checkpointing" not in cfg, (
            f"configs[{idx}].config must DROP gradient_checkpointing in the eval variant")


def test_online_eval_flips_use_ema_false():
    assert _load(_ONLINE)["training"]["use_ema"] is False
    assert _load(_BASE)["training"]["use_ema"] is True  # guard the flip is a real flip


def test_online_eval_preserves_cylindrical_backbone_and_fa_invariant():
    """The eval variant must NOT disturb the no-SSL cylindrical backbone or the
    fa_invariant[0.0] one-pass conditioning (that would silently change what is evaluated)."""
    online = _load(_ONLINE)
    assert online["training"]["cond_method"] == "fa_invariant"
    assert online["training"]["frame_avg_angles"] == [0.0]
    for idx in (1, 2):
        vit = online["model"]["conditioning"]["configs"][idx]["config"]["ViT"]
        assert vit["implementation"] == "cylindrical_dinov3"
        assert vit["gauge"] == "cylindrical_xyz"
        assert vit["from_scratch"] is False
