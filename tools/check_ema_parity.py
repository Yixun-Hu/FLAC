"""
EMA/raw-load parity gate (reliability diagnostic).

Replicates eval_FLAC.py:35-53's EXACT state_dict transform for both
step=145000 checkpoints and reports, per model:
  - does the ckpt carry diffusion_ema.ema_model.* keys?
  - does the EMA branch fire (use_ema in cfg AND ema keys present)?
  - after load_state_dict(strict=False): #missing / #unexpected for the
    model.* (DiT) params -> a large miss = silent wrong/partial load.

If one model loads EMA and the other loads raw, every cross-model number
is invalid. If the baseline loads with many missing model.* keys, even the
within-baseline 3-way control is built on a mis-loaded net.
"""
import json
import os
import sys

# tools/ scripts get tools/ on sys.path[0], which would import the STALE
# pip-installed site-packages/src (predates the fused_pose conditioner).
# Entry points (eval_FLAC.py/train.py) run from the repo root and use repo
# src/. Match them so this gate exercises the SAME code the eval used.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.models.factory import create_model_from_config

CKPTS = {
    "baseline FLAC_AR": (
        "src/configs/model_configs/FLAC/AR/FLAC_AR.json",
        "outputs_FLAC/FLAC_AR_baseline_short/FLAC_AR_baseline_short_training/checkpoints/epoch=15-step=145000.ckpt",
    ),
    "V3 FLAC_AR_arbRIR_v0": (
        "src/configs/model_configs/FLAC/AR/FLAC_AR_arbRIR_v0.json",
        "outputs_FLAC/FLAC_arbRIR_v0/FLAC_arbRIR_v0_training/checkpoints/epoch=15-step=145000.ckpt",
    ),
}


def transform_like_eval_flac(ckpt_path, training_config):
    """Byte-for-byte the eval_FLAC.py:35-49 logic."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    n_diffusion = sum(1 for k in state_dict if k.startswith("diffusion."))
    n_ema = sum(1 for k in state_dict if k.startswith("diffusion_ema.ema_model."))

    for key in list(state_dict.keys()):
        if key.startswith("diffusion."):
            new_key = key.replace("diffusion.", "")
            state_dict[new_key] = state_dict.pop(key)

    ema_fired = False
    if training_config.get("use_ema", False) and any(
        k.startswith("diffusion_ema.ema_model.") for k in state_dict.keys()
    ):
        ema_fired = True
        for key in list(state_dict.keys()):
            if key.startswith("diffusion_ema.ema_model."):
                new_key = key.replace("diffusion_ema.ema_model.", "model.")
                state_dict[new_key] = state_dict.pop(key)
    return state_dict, n_diffusion, n_ema, ema_fired


def main():
    verdicts = []
    for name, (mc_path, ckpt_path) in CKPTS.items():
        cfg = json.load(open(mc_path))
        tcfg = cfg["training"]
        sd, n_diff, n_ema, ema_fired = transform_like_eval_flac(ckpt_path, dict(tcfg))

        model = create_model_from_config(cfg)
        res = model.load_state_dict(sd, strict=False)
        model_params = {n for n, _ in model.named_parameters()}
        missing_model = [k for k in res.missing_keys if k in model_params]

        print(f"\n=== {name} ===")
        print(f"  ckpt: {ckpt_path.split('/')[-1]}")
        print(f"  use_ema(cfg)         : {tcfg.get('use_ema')}")
        print(f"  raw 'diffusion.*'    : {n_diff}")
        print(f"  'diffusion_ema.*'    : {n_ema}")
        print(f"  EMA branch fired     : {ema_fired}  <-- must be True")
        print(f"  load missing (any)   : {len(res.missing_keys)}")
        print(f"  load missing (param) : {len(missing_model)}  <-- must be ~0")
        print(f"  load unexpected      : {len(res.unexpected_keys)}")
        if missing_model[:5]:
            print(f"  e.g. missing params  : {missing_model[:5]}")
        verdicts.append((name, ema_fired, len(missing_model)))

    print("\n=== PARITY VERDICT ===")
    all_ema = all(v[1] for v in verdicts)
    all_clean = all(v[2] <= 2 for v in verdicts)  # tolerate <=2 (e.g. buffers)
    same = len({v[1] for v in verdicts}) == 1
    print(f"  both EMA-loaded      : {all_ema}")
    print(f"  EMA decision matches : {same} (both {verdicts[0][1]})")
    print(f"  both load cleanly    : {all_clean}")
    ok = all_ema and all_clean and same
    print(f"\n  GATE: {'PASS - comparison is valid, proceed' if ok else 'FAIL - results not trustworthy, STOP'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
