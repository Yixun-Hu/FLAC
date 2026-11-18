#!/usr/bin/env python3
"""exp_46 (forked from exp_45's exp45_init_digest.py at X45f, from exp_38's) — the registered seed-42 initial-state digest of an arm
(exp_38 plan r2 §10 B3; exp_46 plan r3 §3.2: names / registry only).

Builds the training wrapper on the CPU in train.py's exact order (float32 matmul precision,
``pl.seed_everything(42, workers=True)`` -> dataloader construction -> ``create_model_from_config``
-> VAE ``load_state_dict`` -> ``create_training_wrapper_from_config``) and prints the
``src.training.run_evidence.state_dict_digest`` of the wrapper plus the Sobol and RNG digests.
The in-process evidence of every training rank must report the same state digest (the start
gate); YAW, B-F and P1 share it because their ``.model`` sections are canonically identical -- and C4F, the
registered exp_46 prediction (plan r3 §3.2 "Init digest": exp_38's values), verified at L3: the command compares
the measurement with the prediction (prediction_problems) and exits 0 only when every value agrees, 1 on any
disagreement (Codex code r1 finding 5), 2 when it refuses its --json destination.

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


def prediction(run):
    """exp_46's REGISTERED prediction for what init_digest() measures on the CPU (plan r3 §3.2 "Init digest": exp_38's
    values): the wrapper's state digest and tensor count, the untouched Sobol engine, and the RNG digests after the build
    = exp_38's single-topology fit-start python / numpy / torch_cpu digests (exp_38's recorded L3 measurement)."""
    return {"run": run, "state_dict_sha256": reg.INIT_STATE_SHA256, "n_state_tensors": reg.N_STATE_TENSORS,
            "sobol": reg.INIT_SOBOL,
            "rng_after_build": {k: reg.RNG_AT_FIT_START["single"][k] for k in ("python", "numpy", "torch_cpu")}}


def prediction_problems(measured, run):
    """Every disagreement of a measurement with the prediction for ``run`` (exact keys, values and types); a
    torch_cuda digest (present only when CUDA was initialised) must be the registered one."""
    want = prediction(run)
    if not isinstance(measured, dict) or set(measured) != set(want):
        return [f"measured keys {sorted(measured) if isinstance(measured, dict) else measured!r} != {sorted(want)}"]
    p = [f"{k}: measured {measured[k]!r} != predicted {want[k]!r}" for k in ("run", "state_dict_sha256",
                                                                               "n_state_tensors", "sobol")
         if reg.canonical(measured[k]) != reg.canonical(want[k]) or type(measured[k]) is not type(want[k])]
    rng = dict(want["rng_after_build"])
    if isinstance(measured["rng_after_build"], dict) and "torch_cuda" in measured["rng_after_build"]:
        rng["torch_cuda"] = reg.RNG_AT_FIT_START["single"]["torch_cuda"]
    if measured["rng_after_build"] != rng:
        p.append(f"rng_after_build: measured {measured['rng_after_build']!r} != predicted {rng!r}")
    return p


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", required=True)
    p.add_argument("--json")
    a = p.parse_args(argv)
    if a.json:              # the destination FIRST (Codex code r1 finding 2): resolved, isolation-checked, never clobbered
        # -- before the model is built -- and ABSOLUTE, anchored to the caller's cwd: init_digest() changes the cwd to
        # the repo, so a relative string would name another place at write time; this one path is checked, re-checked
        # just before the write and written (Codex code r2 finding 3)
        a.json = os.path.abspath(a.json)
        try:
            reg.check_isolation(a.json, "the init-digest record")
        except reg.GateError as err:
            print(f"EXP46 INIT DIGEST REFUSED: {err}", file=sys.stderr)
            return 2
        if os.path.lexists(a.json):
            print(f"refusing to overwrite {a.json}", file=sys.stderr)
            return 2
    out = init_digest(a.run)
    problems = prediction_problems(out, a.run)            # L3 is fail-closed (Codex code r1 finding 5)
    rec = dict(out, prediction="PASS" if not problems else "FAIL", prediction_problems=problems)
    text = json.dumps(rec, indent=1, sort_keys=True)
    print(text)
    print("INIT DIGEST PREDICTION " + ("PASS" if not problems else "FAIL: " + " | ".join(problems)))
    if a.json:
        try:                # the same absolute path, re-checked (a component aliased meanwhile refuses); exclusive and
            reg.check_isolation(a.json, "the init-digest record")       # not following a final link: an entry
            fd = os.open(a.json, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o664)   # created meanwhile
        except (reg.GateError, OSError) as err:                                                 # is never replaced
            print(f"EXP46 INIT DIGEST REFUSED: {err}", file=sys.stderr)
            return 2
        with os.fdopen(fd, "w") as fh:
            fh.write(text + "\n")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
