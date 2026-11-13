"""exp_43 std_aug -- capture side of the yaw_aug FLIP-path regression golden (plan §3.3).

A pin against DRIFT of the flip path, not an equivalence claim: the flip path exists only from
exp_43 on, so there is no reference arm to be equal to (the flip-ABSENT path is pinned by
exp_40's golden, replayed in ``test_yaw_aug_flip.py``). It records, from a GIVEN source tree,
what ``training.yaw_aug = {enabled, img_w, seed: 42, flip: true}`` does:

* ``tiny``: one whole ``training_step`` of exp_40's tiny CPU diffusion_cond harness (real
  factories, real DiT, flash-attn neutralised) for exp_40's (global_step, global_rank) cases:
  the column offsets that reached ``rotate_scene_metadata``, the flip bits drawn, the sample
  indices actually passed to ``mirror_scene_metadata``, the metadata and conditioning that
  reached the model, the loss (float.hex) and the global RNG digests either side;
* ``real``: at width 512 and seed 42, for both reference rungs (8 x 8, 2 x 32) over exp_40's
  global steps: per (step, rank) the offsets, bits and applied indices of ``_apply_yaw_aug``,
  a digest of one augmented batch, the totals, and whether the global RNG was left untouched.

The harness helpers (tiny config, synthetic metadata, digests, stub trainer, flash scope) are
exp_40's ``exp40_yaw_aug_golden_capture.py``, loaded by path; the fixture binds the bytes of
BOTH tools. Regenerate deliberately (never to "fix" a red test)::

    python src/tests/exp43_yaw_aug_flip_golden_capture.py capture --out <fixture>
    python src/tests/exp43_yaw_aug_flip_golden_capture.py capture --repo <git repo> --sha <40-hex> --out <fixture>

Without ``--sha`` the CURRENT checkout is captured and the record names ``git HEAD``, whether
the flip closure differs from it, and the closure files' sha256. With ``--sha`` the tool
exports ``git archive <sha> src`` ITSELF into a temporary directory, puts it first on
``sys.path``, refuses if ``src`` resolves elsewhere, and records the commit and the digest of
its ``git ls-tree -r <sha> -- src`` (exp_40's provenance rule). The pytest
(``test_exp43_yaw_aug_flip_golden.py``) only READS the fixture and replays the record.

Written by the exp_43 Coder seat (Claude Opus 5.5).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS_PATH = os.path.join(_HERE, "exp40_yaw_aug_golden_capture.py")
FLIP_BLOCK = {"flip": True}
CLOSURE = ("src/data/yaw_mirror.py", "src/data/yaw_rotation.py", "src/data/yaw_offset_step.py",
           "src/training/diffusion.py", "src/training/factory.py")
PROBE = ("2x32", 2500, 1)                       # (rung, step, rank) of the batch digest


def _load_harness():
    spec = importlib.util.spec_from_file_location("exp40_harness_for_exp43_flip_golden", HARNESS_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


h = _load_harness()


class FlipSpy:
    """Wraps ``tdiff.rotate_scene_metadata`` / ``draw_mirror_flips`` / ``mirror_scene_metadata``:
    the column offset and output of every rotation (call order = sample order), the drawn bits,
    and the index of the rotated sample each mirror call received (by object identity)."""

    def __init__(self, m):
        self.m = m
        self.offsets, self.alpha_hex, self._rotated, self.flips, self.applied = [], [], [], [], []
        self._orig = None

    def __enter__(self):
        td = self.m.tdiff
        self._orig = (td.rotate_scene_metadata, td.draw_mirror_flips, td.mirror_scene_metadata)
        rot, draw, mir = self._orig

        def rot_spy(md, alpha, img_w):
            out = rot(md, alpha, img_w)
            self.offsets.append(self.m.yr.yaw_column_shift(alpha, img_w))
            self.alpha_hex.append(float(alpha).hex())
            self._rotated.append(out)
            return out

        def draw_spy(n, generator):
            bits = draw(n, generator)
            self.flips.extend(int(b) for b in bits.tolist())
            return bits

        def mir_spy(md, img_w):
            hits = [i for i, r in enumerate(self._rotated) if r is md]
            if len(hits) != 1:
                raise RuntimeError("mirror_scene_metadata received an object that is not "
                                   "exactly one rotated sample")
            self.applied.append(hits[0])
            return mir(md, img_w)

        td.rotate_scene_metadata, td.draw_mirror_flips, td.mirror_scene_metadata = rot_spy, draw_spy, mir_spy
        return self

    def __exit__(self, *exc):
        td = self.m.tdiff
        td.rotate_scene_metadata, td.draw_mirror_flips, td.mirror_scene_metadata = self._orig
        return False


def capture_tiny_case(m, step, rank):
    random.seed(h.GOLDEN_SEED)
    m.np.random.seed(h.GOLDEN_SEED)
    m.torch.manual_seed(h.GOLDEN_SEED)
    wrapper = h.build_wrapper(m, h.tiny_config(h.TINY_IMG_W, FLIP_BLOCK))
    h.attach_stub_trainer(wrapper, step, rank)
    seen = {}
    original = wrapper._compute_conditioning

    def cond_spy(metadata):
        seen["metadata"] = h.metadata_digests(metadata)
        out = original(metadata)
        seen["conditioning"] = {k: h.tensor_digest(v[0]) for k, v in sorted(out.items())}
        return out

    wrapper._compute_conditioning = cond_spy
    g = m.torch.Generator().manual_seed(9000)
    reals = m.torch.randn(h.TINY_N, 4, 64, generator=g)
    metadata = [h.make_md(m, s, h.TINY_IMG_W) for s in range(h.TINY_N)]
    before_inputs = h.metadata_digests(metadata)
    rng_before = h.rng_digest(m)
    with FlipSpy(m) as spy:
        loss = wrapper.training_step((reals, metadata), 0)
    rng_after = h.rng_digest(m)
    if h.metadata_digests(metadata) != before_inputs:
        raise RuntimeError("training_step mutated its input metadata")
    return {
        "offsets": spy.offsets,
        "alpha_hex": spy.alpha_hex,
        "flips": spy.flips,
        "applied": spy.applied,
        "metadata_into_conditioner": seen["metadata"],
        "conditioning": seen["conditioning"],
        "loss_hex": float(loss.item()).hex(),
        "rng_before": rng_before,
        "rng_after": rng_after,
    }


def capture_real(m):
    """``_apply_yaw_aug`` at width 512, seed 42, flip on, for both reference rungs."""
    random.seed(h.GOLDEN_SEED)
    m.np.random.seed(h.GOLDEN_SEED)
    m.torch.manual_seed(h.GOLDEN_SEED)
    wrapper = h.build_wrapper(m, h.tiny_config(h.REAL_IMG_W, FLIP_BLOCK))
    out, probe, samples, flipped = {}, None, 0, 0
    rng_before = h.rng_digest(m)
    for name, (world, micro) in h.RUNGS.items():
        metadata = [h.make_md(m, 100 + s, h.REAL_IMG_W, height=2) for s in range(micro)]
        per_step = {}
        for step in h.REAL_STEPS:
            ranks = []
            for rank in range(world):
                h.attach_stub_trainer(wrapper, step, rank)
                with FlipSpy(m) as spy:
                    augmented = wrapper._apply_yaw_aug(metadata)
                ranks.append({"offsets": spy.offsets, "flips": spy.flips, "applied": spy.applied})
                samples += len(spy.flips)
                flipped += len(spy.applied)
                if (name, step, rank) == PROBE:
                    probe = h.metadata_digests(augmented)
            per_step[str(step)] = ranks
        out[name] = per_step
    rng_after = h.rng_digest(m)
    return {"per_step": out, "augmented_probe_2x32_step2500_rank1": probe,
            "totals": {"samples": samples, "flipped": flipped},
            "global_rng_untouched": rng_before == rng_after}


def capture_all():
    m = h._mods()
    with h.flash_disabled(m):
        return {
            "tiny": {f"step{s}_rank{r}": capture_tiny_case(m, s, r) for s, r in h.TINY_CASES},
            "real": capture_real(m),
        }


def tool_sha256(path=None):
    with open(path or os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _git(repo, *args, text=True):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, check=True, text=text).stdout


def _working_tree_source(repo):
    head = _git(repo, "rev-parse", "HEAD").strip()
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *CLOSURE], cwd=repo).returncode != 0
    untracked = _git(repo, "ls-files", "--others", "--exclude-standard", "--", *CLOSURE).split()
    closure = {}
    for rel in CLOSURE:
        with open(os.path.join(repo, rel), "rb") as f:
            closure[rel] = hashlib.sha256(f.read()).hexdigest()
    return {"kind": "working-tree", "head": head, "closure_differs_from_head": bool(dirty or untracked),
            "closure_untracked": sorted(untracked), "closure_sha256": closure}


def _commit_source(repo, sha):
    closure = {rel: hashlib.sha256(_git(repo, "show", f"{sha}:{rel}", text=False)).hexdigest()
               for rel in CLOSURE}
    return {"kind": "commit", "sha": sha, "source_tree_digest": h.source_tree_digest(repo, sha),
            "closure_sha256": closure}


def _capture_cmd(args):
    if (args.sha is None) != (args.repo is None):
        raise SystemExit("--sha and --repo go together (a commit capture needs both)")
    if args.sha is None:
        repo = os.path.realpath(os.path.join(_HERE, "..", ".."))
        source = _working_tree_source(repo)
        sys.dont_write_bytecode = True
        sys.path.insert(0, repo)              # a script run puts src/tests, not the repo, first
        import src
        got = os.path.realpath(os.path.dirname(src.__file__))
        if got != os.path.join(repo, "src"):
            raise SystemExit(f"src resolved to {got}, not this checkout's {repo}/src")
        import torch
        record = capture_all()
    else:
        if not re.fullmatch(r"[0-9a-f]{40}", args.sha):
            raise SystemExit(f"--sha must be a full 40-hex commit id, got {args.sha!r}")
        with tempfile.TemporaryDirectory(prefix="exp43_flip_golden_") as root:
            tar = _git(args.repo, "archive", "--format=tar", args.sha, "--", "src", text=False)
            subprocess.run(["tar", "-x", "-C", root], input=tar, check=True)
            sys.dont_write_bytecode = True
            sys.path.insert(0, root)
            import src
            got = os.path.realpath(os.path.dirname(src.__file__))
            if got != os.path.join(os.path.realpath(root), "src"):
                raise SystemExit(f"src resolved to {got}, not the export of {args.sha}")
            import torch
            record = capture_all()
        source = _commit_source(args.repo, args.sha)
    fixture = {
        "_meta": {
            "experiment": "exp_43",
            "purpose": "yaw_aug FLIP-path regression golden (training.yaw_aug.flip=true): a pin "
                       "against drift of the flip path, NOT an equivalence claim",
            "capture_tool": "src/tests/exp43_yaw_aug_flip_golden_capture.py",
            "capture_tool_sha256": tool_sha256(),
            "harness_tool": "src/tests/exp40_yaw_aug_golden_capture.py",
            "harness_tool_sha256": tool_sha256(HARNESS_PATH),
            "source": source,
            "record_sha256": h.canonical_sha256(record),
            "python": sys.version.split()[0], "torch": torch.__version__,
            "master_seed": h.GOLDEN_SEED, "yaw_seed": h.YAW_SEED, "flip_block": dict(FLIP_BLOCK),
            "tiny_cases": [list(c) for c in h.TINY_CASES], "real_steps": list(h.REAL_STEPS),
            "rungs": {k: list(v) for k, v in h.RUNGS.items()},
        },
        "record": record,
    }
    with open(args.out, "w") as f:
        json.dump(fixture, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"wrote {args.out}: record_sha256 {fixture['_meta']['record_sha256']} ({source['kind']})")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture", help="capture the flip-path record into a fixture file")
    c.add_argument("--repo", default=None)
    c.add_argument("--sha", default=None)
    c.add_argument("--out", required=True)
    return _capture_cmd(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
