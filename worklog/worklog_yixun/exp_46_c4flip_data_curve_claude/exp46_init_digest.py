#!/usr/bin/env python3
"""exp_46 (forked from exp_45's exp45_init_digest.py at X45f, from exp_38's) — the registered seed-42 initial-state digest of an arm
(exp_38 plan r2 §10 B3; exp_46 plan r3 §3.2: names / registry only).

Builds the training wrapper on the CPU in train.py's exact order (float32 matmul precision,
``pl.seed_everything(42, workers=True)`` -> dataloader construction -> ``create_model_from_config``
-> VAE ``load_state_dict`` -> ``create_training_wrapper_from_config``) and prints the
``src.training.run_evidence.state_dict_digest`` of the wrapper plus the Sobol and RNG digests.
The in-process evidence of every training rank must report the same state digest (the start
gate); YAW, B-F and P1 share it because their ``.model`` sections are canonically identical -- and C4F, the
registered exp_46 prediction (plan r3 §3.2 "Init digest": exp_38's values), verified at L3.

    python exp46_init_digest.py --run exp46_C4Ff025_l40_2x32 [--json out.json]

Written by the exp_38 Coder (Claude Opus 5.5).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import exp46_registry as reg  # noqa: E402


def init_digest(run, repo=None):
    repo = repo or reg.REPO
    sys.path.insert(0, repo)
    os.chdir(repo)
    import pytorch_lightning as pl
    import torch
    from src.data.dataset import create_dataloader_from_config
    from src.models import create_model_from_config
    from src.models.utils import load_ckpt_state_dict
    from src.training import create_training_wrapper_from_config
    from src.training import run_evidence as rev

    arm, tag, rung = reg.parse_run_id(run)
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(reg.SEED, workers=True)
    model_config = json.load(open(reg.model_config_rel(arm, rung)))
    dataset_config = json.load(open(reg.FRACS[tag]["dataset_config"]))
    create_dataloader_from_config(dataset_config, batch_size=reg.RUNGS[rung]["micro"], num_workers=6,
                                  sample_rate=model_config["sample_rate"], sample_size=model_config["sample_size"],
                                  audio_channels=model_config.get("audio_channels", 1))
    model = create_model_from_config(model_config)
    model.pretransform.load_state_dict(load_ckpt_state_dict(reg.VAE_PATH))
    wrapper = create_training_wrapper_from_config(model_config, model)
    sd = wrapper.state_dict()
    return {"run": run, "state_dict_sha256": rev.state_dict_digest(sd), "n_state_tensors": len(sd),
            "sobol": rev.sobol_state(wrapper.rng), "rng_after_build": rev.rng_digests()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", required=True)
    p.add_argument("--json")
    a = p.parse_args(argv)
    if a.json:              # the destination FIRST (Codex code r1 finding 2): resolved, isolation-checked, never clobbered
        try:                # -- before the model is built
            reg.check_isolation(a.json, "the init-digest record")
        except reg.GateError as err:
            print(f"EXP46 INIT DIGEST REFUSED: {err}", file=sys.stderr)
            return 2
        if os.path.lexists(a.json):
            print(f"refusing to overwrite {a.json}", file=sys.stderr)
            return 2
    out = init_digest(a.run)
    text = json.dumps(out, indent=1, sort_keys=True)
    print(text)
    if a.json:
        with open(a.json, "x") as fh:                     # exclusive: a file created meanwhile is never overwritten
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
