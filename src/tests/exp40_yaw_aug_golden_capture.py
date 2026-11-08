"""exp_40 yaw_aug_c16 -- capture side of the yaw_aug ENABLED-path golden.

The exp_40 arm differs from the reference Yaw-Aug arms (exp_15 neuronic 8x8, exp_17
A6000 2x32) in exactly one thing: the set of column offsets. Everything else -- the
counter-based draw, the rotation, the training step -- must be the code those arms
trained with. This module records, from a GIVEN source tree, what the yaw_aug
enabled path does:

* ``tiny``: one whole ``training_step`` of a tiny CPU diffusion_cond model (real
  factories, real DiT, flash-attn neutralised) with ``training.yaw_aug`` enabled,
  for a few (global_step, global_rank) cases: the per-sample column offsets that
  reached ``rotate_scene_metadata``, the metadata and conditioning that reached the
  model, the loss (float.hex) and the global RNG digests either side;
* ``real``: the offsets ``_apply_yaw_aug`` draws at the production width 512 and
  seed 42, for both reference rungs (8 ranks x 8 = exp_15, 2 ranks x 32 = exp_17)
  over a fixed set of global steps, plus a digest of the rotated metadata of one
  case and a check that the global RNG streams were left untouched.

It runs against the CURRENT checkout when imported (the pytest replays it), or
against an exported historical tree when run as a script::

    python src/tests/exp40_yaw_aug_golden_capture.py capture --src-root <tree> --out <json>
    python src/tests/exp40_yaw_aug_golden_capture.py assemble --pin <sha>=<json> ... \
        --out src/tests/fixtures/exp40_yaw_aug_enabled_golden.json

where ``<tree>`` is ``git archive <sha> src | tar -x -C <tree>``. ``capture`` puts
``<tree>`` first on ``sys.path`` and refuses to run if ``src`` resolves elsewhere;
``assemble`` refuses unless every pin's record is identical. The pytest
(``test_yaw_aug_c16.py``) only READS the committed fixture -- it never needs git.

Written by the exp_40 Coder seat (Claude Opus 5.5).
"""
import argparse
import hashlib
import json
import os
import random
import sys
import types

GOLDEN_SEED = 20261107
YAW_SEED = 42
TINY_IMG_W = 16
TINY_N = 4
TINY_CASES = ((0, 0), (7, 0), (7, 1), (2500, 3))      # (global_step, global_rank)
REAL_IMG_W = 512
REAL_STEPS = (0, 1, 2, 3, 1000, 2499, 2500, 39999)
RUNGS = {"8x8": (8, 8), "2x32": (2, 32)}               # name -> (world size, micro-batch)
_MD_TENSOR_KEYS = ("source", "source_vit", "context_poses", "context_poses_vit",
                   "context_audio", "depth", "padding_mask")


def tiny_config(img_w=TINY_IMG_W, yaw_aug=None):
    """The exp_15 golden harness's tiny model, plus an optional ``training.yaw_aug``."""
    cfg = {
        "model_type": "diffusion_cond",
        "sample_size": 64,
        "sample_rate": 22050,
        "audio_channels": 1,
        "model": {
            "conditioning": {
                "configs": [
                    {"id": "source", "type": "dist_embedder",
                     "config": {"num_freqs": 4, "max_freq": 4, "ch_dim": 1, "include_in": True}},
                    {"id": "context_poses", "type": "dist_embedder",
                     "config": {"num_freqs": 4, "max_freq": 4, "ch_dim": 1, "include_in": True}},
                ],
                "cond_dim": 32,
            },
            "diffusion": {
                "cross_attention_cond_ids": ["context_poses"],
                "global_cond_ids": ["source"],
                "type": "dit",
                "diffusion_objective": "rectified_flow",
                "config": {
                    "io_channels": 4, "embed_dim": 64, "depth": 1, "num_heads": 1,
                    "cond_token_dim": 32, "global_cond_dim": 32,
                    "transformer_type": "continuous_transformer",
                    "global_cond_type": "adaLN",
                },
            },
            "io_channels": 4,
        },
        "training": {
            "timestep_sampler": "uniform",
            "cfg_dropout_prob": 0.0,
            "use_ema": False,
            "optimizer_configs": {
                "diffusion": {"optimizer": {"type": "AdamW",
                    "config": {"lr": 5e-6, "betas": [0.9, 0.999], "weight_decay": 1e-3}}}
            },
        },
    }
    block = {"enabled": True, "img_w": img_w, "seed": YAW_SEED}
    block.update(yaw_aug or {})
    cfg["training"]["yaw_aug"] = block
    return cfg


def _mods():
    import numpy as np
    import torch
    import src.models.transformer as transformer
    import src.training.diffusion as tdiff
    from src.data import yaw_rotation as yr
    from src.models.factory import create_model_from_config
    from src.training.factory import create_training_wrapper_from_config
    transformer.flash_attn_func = None      # CPU: the installed kernel is CUDA-only
    return types.SimpleNamespace(np=np, torch=torch, tdiff=tdiff, yr=yr,
                                 create_model=create_model_from_config,
                                 create_wrapper=create_training_wrapper_from_config)


def build_wrapper(m, cfg):
    return m.create_wrapper(cfg, m.create_model(cfg))


def attach_stub_trainer(wrapper, global_step, global_rank):
    wrapper.log_dict = lambda *a, **k: None
    wrapper.trainer = types.SimpleNamespace(
        optimizers=[types.SimpleNamespace(param_groups=[{"lr": 5e-6}])],
        global_step=global_step, global_rank=global_rank)


def make_md(m, seed, img_w, height=4):
    g = m.torch.Generator().manual_seed(seed)
    randn = m.torch.randn
    return {
        "source": randn(3, generator=g),
        "source_vit": randn(3, generator=g),
        "context_poses": randn(2, 3, generator=g),
        "context_poses_vit": randn(2, 3, generator=g),
        "context_audio": randn(2, 8, generator=g),
        "depth": randn(3, height, img_w, generator=g),
        "padding_mask": m.torch.ones(64),
        "scene": f"scene_{seed}",
    }


def tensor_digest(t):
    arr = t.detach().cpu().contiguous().numpy()
    return {"shape": list(t.shape), "dtype": str(t.dtype),
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest()}


def rng_digest(m):
    h = hashlib.sha256()
    h.update(repr(random.getstate()).encode("utf-8"))
    st = m.np.random.get_state()
    h.update(str(st[0]).encode("utf-8"))
    h.update(m.np.asarray(st[1], dtype=m.np.uint32).tobytes())
    h.update(repr(tuple(st[2:])).encode("utf-8"))
    h.update(m.torch.get_rng_state().numpy().tobytes())
    return h.hexdigest()


def metadata_digests(metadata):
    return [{k: tensor_digest(md[k]) for k in _MD_TENSOR_KEYS if k in md} for md in metadata]


class RotationSpy:
    """Wraps ``tdiff.rotate_scene_metadata``; records the column offset per call."""

    def __init__(self, m):
        self.m, self.calls, self._orig = m, [], None

    def __enter__(self):
        self._orig = self.m.tdiff.rotate_scene_metadata

        def spy(md, alpha, img_w):
            self.calls.append((self.m.yr.yaw_column_shift(alpha, img_w), float(alpha).hex()))
            return self._orig(md, alpha, img_w)

        self.m.tdiff.rotate_scene_metadata = spy
        return self

    def __exit__(self, *exc):
        self.m.tdiff.rotate_scene_metadata = self._orig
        return False


def capture_tiny_case(m, step, rank, yaw_aug=None):
    random.seed(GOLDEN_SEED)
    m.np.random.seed(GOLDEN_SEED)
    m.torch.manual_seed(GOLDEN_SEED)
    wrapper = build_wrapper(m, tiny_config(TINY_IMG_W, yaw_aug))
    attach_stub_trainer(wrapper, step, rank)
    seen = {}
    original = wrapper._compute_conditioning

    def cond_spy(metadata):
        seen["metadata"] = metadata_digests(metadata)
        out = original(metadata)
        seen["conditioning"] = {k: tensor_digest(v[0]) for k, v in sorted(out.items())}
        return out

    wrapper._compute_conditioning = cond_spy
    g = m.torch.Generator().manual_seed(9000)
    reals = m.torch.randn(TINY_N, 4, 64, generator=g)
    metadata = [make_md(m, s, TINY_IMG_W) for s in range(TINY_N)]
    before_inputs = metadata_digests(metadata)
    rng_before = rng_digest(m)
    with RotationSpy(m) as spy:
        loss = wrapper.training_step((reals, metadata), 0)
    rng_after = rng_digest(m)
    assert metadata_digests(metadata) == before_inputs, "training_step mutated its input metadata"
    return {
        "offsets": [c[0] for c in spy.calls],
        "alpha_hex": [c[1] for c in spy.calls],
        "metadata_into_conditioner": seen["metadata"],
        "conditioning": seen["conditioning"],
        "loss_hex": float(loss.item()).hex(),
        "rng_before": rng_before,
        "rng_after": rng_after,
    }


def capture_real(m, yaw_aug=None):
    """Offsets of ``_apply_yaw_aug`` at width 512, seed 42, for both reference rungs."""
    random.seed(GOLDEN_SEED)
    m.np.random.seed(GOLDEN_SEED)
    m.torch.manual_seed(GOLDEN_SEED)
    wrapper = build_wrapper(m, tiny_config(REAL_IMG_W, yaw_aug))
    out = {}
    probe = None
    rng_before = rng_digest(m)
    for name, (world, micro) in RUNGS.items():
        metadata = [make_md(m, 100 + s, REAL_IMG_W, height=2) for s in range(micro)]
        per_step = {}
        for step in REAL_STEPS:
            ranks = []
            for rank in range(world):
                attach_stub_trainer(wrapper, step, rank)
                with RotationSpy(m) as spy:
                    rotated = wrapper._apply_yaw_aug(metadata)
                ranks.append([c[0] for c in spy.calls])
                if name == "2x32" and step == 2500 and rank == 1:
                    probe = metadata_digests(rotated)
            per_step[str(step)] = ranks
        out[name] = per_step
    rng_after = rng_digest(m)
    return {"offsets": out, "rotated_probe_2x32_step2500_rank1": probe,
            "global_rng_untouched": rng_before == rng_after}


def capture_all(tiny_yaw_aug=None, real_yaw_aug=None):
    """The whole record. ``*_yaw_aug`` override fields of the enabled block (exp_40's
    ``offset_step``); the golden itself is captured with both ``None`` -- the
    reference arms' block ``{enabled, img_w, seed}``, nothing else."""
    m = _mods()
    return {
        "tiny": {f"step{s}_rank{r}": capture_tiny_case(m, s, r, tiny_yaw_aug)
                 for s, r in TINY_CASES},
        "real": capture_real(m, real_yaw_aug),
    }


def canonical_sha256(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _capture_cmd(args):
    root = os.path.realpath(args.src_root)
    sys.path.insert(0, root)
    import src
    got = os.path.realpath(os.path.dirname(src.__file__))
    if got != os.path.join(root, "src"):
        raise SystemExit(f"src resolved to {got}, not under --src-root {root}")
    import torch
    record = capture_all()
    payload = {"src_root": root, "torch": torch.__version__, "record": record,
               "record_sha256": canonical_sha256(record)}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"record_sha256 {payload['record_sha256']}  ({args.out})")
    return 0


def _assemble_cmd(args):
    """Combine per-pin captures into the fixture; refuse unless every record is equal."""
    pins, record, torch_versions = {}, None, set()
    for item in args.pin:
        sha, _, path = item.partition("=")
        if len(sha) != 40 or not path:
            raise SystemExit(f"--pin wants <40-hex sha>=<capture json>, got {item!r}")
        with open(path) as f:
            payload = json.load(f)
        got = canonical_sha256(payload["record"])
        if got != payload["record_sha256"]:
            raise SystemExit(f"{path}: record_sha256 does not match its record")
        if record is None:
            record = payload["record"]
        elif canonical_sha256(record) != got:
            raise SystemExit(f"{path}: the record at {sha} differs from the first pin's")
        pins[sha] = got
        torch_versions.add(payload["torch"])
    with open(os.path.abspath(__file__), "rb") as f:
        tool_sha = hashlib.sha256(f.read()).hexdigest()
    fixture = {
        "_meta": {
            "experiment": "exp_40",
            "purpose": "yaw_aug ENABLED-path golden: the reference arms' training-step "
                       "behaviour (offset_step ABSENT), captured at their source pins",
            "capture_tool": "src/tests/exp40_yaw_aug_golden_capture.py",
            "capture_tool_sha256": tool_sha,
            "pins": pins,
            "record_sha256": canonical_sha256(record),
            "torch": sorted(torch_versions),
            "master_seed": GOLDEN_SEED, "yaw_seed": YAW_SEED,
            "tiny_cases": [list(c) for c in TINY_CASES], "real_steps": list(REAL_STEPS),
            "rungs": {k: list(v) for k, v in RUNGS.items()},
        },
        "record": record,
    }
    with open(args.out, "w") as f:
        json.dump(fixture, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"wrote {args.out}: {len(pins)} pins, record_sha256 {fixture['_meta']['record_sha256']}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture", help="capture from an exported source tree")
    c.add_argument("--src-root", required=True, help="an exported source tree holding src/")
    c.add_argument("--out", required=True)
    a = sub.add_parser("assemble", help="combine equal per-pin captures into the fixture")
    a.add_argument("--pin", action="append", required=True, help="<40-hex sha>=<capture json>")
    a.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return _capture_cmd(args) if args.cmd == "capture" else _assemble_cmd(args)


if __name__ == "__main__":
    sys.exit(main())
