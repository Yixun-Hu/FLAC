#!/usr/bin/env python3
"""exp-09 C1 forward-counter probe + smoke-log verification helpers
(integrative-review finding 2, items 3 & 4).

This is the runtime C1 EVIDENCE toolkit the smoke step consumes. It has two jobs:

1. **Backbone forward counter** (``count`` subcommand / :func:`probe_counts`). A RUNTIME
   counter — it hooks the SHARED DINOv3 backbone and counts how many times it is called
   while producing the exp-09 ``fa_invariant[0.0]`` conditioning for one batch, and it
   asserts that count against the ``vanilla`` count for identical metadata.

   Pass criterion (review-corrected): for the REAL dataset ``K = 8`` context poses per
   sample, the backbone fires **NINE** times per batch — ``1`` source + ``K`` context —
   with **zero extra frame-average passes** (``n_fa == n_vanilla == 1 + K``). The count is
   structural: it is ``1 + K`` regardless of batch size, because the ViTCoordinates
   conditioner batches every sample's source into one call and every sample's k-th context
   pose into one call. CPU-testable via the tiny-scene machinery: ``K = 3 => 4``,
   ``K = 8 => 9``. The C1 records pin ``K = 8 => 9``.

2. **Smoke-log verification** (``verify-log`` subcommand / :func:`verify_log`). A FINITE
   loss must have been logged, and the **SUSTAINED** throughput must clear the floor
   ``>= 0.0395 steps/s`` (plan §3 = 0.5 x the B-F anchor 0.079). The gate is the sustained
   rate = completed steps / elapsed wall time — NOT the max instantaneous progress tick
   (integrative-review r2 blocker 2: one fast tick amid many slow ticks must not pass a slow
   smoke). c1_smoke.sh times the run with bash ``SECONDS`` and passes the achieved step count
   (authoritative); absent that, the final cumulative ``N/M [elapsed<...]`` is parsed from the
   log. The max instantaneous rate (from ``it/s`` / ``s/it`` ticks) is serialised as a
   DESCRIPTIVE field only.

CPU-only, no GPU required (the forward count is device-independent — it counts module
calls). Emits atomic finite JSON. Exits nonzero on any miss.
"""
import argparse
import json
import math
import os
import re
import sys
import tempfile
import typing as tp
from pathlib import Path

# Make the FLAC worktree importable (`import src.*`) regardless of CWD. This file lives at
# <worktree>/worklog/worklog_yixun/exp_09_cyl_no_ssl/forward_counter_probe.py -> parents[3].
_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

OUTPUT_TMP_PREFIX = ".forward_counter_probe."
OUTPUT_TMP_SUFFIX = ".tmp"

# plan §3: throughput floor = 0.5 x B-F anchor 0.079 steps/s (one-sided).
DEFAULT_MIN_STEPS_PER_S = 0.0395


# ============================================================================================
# Atomic serialisation (mirrors the sibling atomic writers so they never drift).
# ============================================================================================
def atomic_write_json(path: str, record: tp.Mapping[str, tp.Any]) -> None:
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=OUTPUT_TMP_PREFIX, suffix=OUTPUT_TMP_SUFFIX)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(record, fh, allow_nan=False, indent=2, sort_keys=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


# ============================================================================================
# Section A — backbone forward counter.
# ============================================================================================
def count_backbone_forwards(mc, run_fn) -> int:
    """Count calls to the SHARED backbone object while ``run_fn`` executes. Mirrors the
    Stage-B integration test's counter: register a forward hook on the one backbone the two
    GeometryConditioners share, run, and return the call count."""
    import src.data.yaw_rotation  # noqa: F401  (ensures src is importable before we hook)
    geoms = [c for c in mc.conditioners.values()
             if getattr(c, "name", None) == "GeometryConditioner"]
    if not geoms:
        raise RuntimeError("no GeometryConditioner in the multi-conditioner — cannot count")
    vit = geoms[0].vit
    # every GeometryConditioner must share ONE backbone object (Stage-B invariant); hook it once.
    for g in geoms[1:]:
        if g.vit is not vit:
            raise RuntimeError("GeometryConditioners do not share ONE backbone — cannot count")
    state = {"n": 0}
    handle = vit.register_forward_hook(lambda *_: state.__setitem__("n", state["n"] + 1))
    try:
        run_fn()
    finally:
        handle.remove()
    return state["n"]


def _context_k(metadata: tp.Sequence[tp.Mapping[str, tp.Any]]) -> int:
    """The number K of context poses per sample (must be identical across the batch)."""
    if not metadata:
        raise ValueError("empty metadata batch")
    ks = set()
    for m in metadata:
        cp = m["context_poses_vit"]
        ks.add(int(cp.shape[0]))
    if len(ks) != 1:
        raise ValueError(f"inconsistent context K across the batch: {sorted(ks)}")
    return ks.pop()


def probe_counts(mc, metadata, device="cpu", angles=(0.0,)) -> tp.Dict[str, tp.Any]:
    """Run the ``vanilla`` and the ``fa_invariant`` paths for one batch and count backbone
    forwards for each. ``expected = 1 + K``. ``pass`` requires ``n_fa == n_vanilla ==
    expected`` — i.e. the fa_invariant path adds NO extra frame-average backbone passes."""
    from src.data import yaw_rotation as yr
    mc.eval()
    K = _context_k(metadata)
    expected = 1 + K

    n_vanilla = count_backbone_forwards(mc, lambda: mc(metadata, device))
    n_fa = count_backbone_forwards(
        mc, lambda: yr.invariant_conditioning(mc, metadata, device, tuple(angles)))

    return {
        "K": K,
        "expected_backbone_calls_per_batch": expected,
        "n_vanilla": n_vanilla,
        "n_fa_invariant": n_fa,
        "no_extra_frame_passes": n_fa == n_vanilla,
        "pass": (n_fa == n_vanilla == expected),
    }


# --- tiny-scene machinery (production-usable so the C1 probe can fabricate a K-batch when a
#     real dataloader is not wired in; identical shape to the Stage-B integration fixtures). ---
def _consistent_depth(H: int, W: int):
    import torch
    j = torch.arange(W, dtype=torch.float32)
    theta = (j + 0.5) * 2.0 * math.pi / W - math.pi
    i = torch.arange(H, dtype=torch.float32)
    el = (i + 0.5) * math.pi / H - math.pi / 2.0
    theta_g = theta.view(1, W).expand(H, W)
    el_g = el.view(H, 1).expand(H, W)
    d = 3.0 + 1.0 * torch.sin(theta_g) + 0.5 * torch.sin(2.0 * theta_g)
    x = d * torch.cos(el_g) * torch.cos(theta_g)
    y = d * torch.cos(el_g) * torch.sin(theta_g)
    z = d * torch.sin(el_g)
    return torch.stack([x, y, z], dim=0).contiguous()


def fabricate_scene_batch(n_samples: int = 2, *, H: int = 32, W: int = 128,
                          k: int = 8, seed: int = 0) -> list:
    """A structurally-real K-context batch (source + K context poses + depth panorama).
    The backbone-call count depends only on this structure (1 source + K context), so a
    fabricated K=8 batch produces the same NINE-call evidence as a real dataloader batch."""
    import torch

    def _one(s):
        g = torch.Generator().manual_seed(s)
        return {
            "source": torch.randn(3, generator=g),
            "source_vit": torch.randn(1, 3, generator=g),
            "context_poses": torch.randn(k, 3, generator=g),
            "context_poses_vit": torch.randn(k, 3, generator=g),
            "context_audio": torch.randn(k, 1, 256, generator=g),
            "depth": _consistent_depth(H, W),
        }
    return [_one(seed + s) for s in range(n_samples)]


def build_conditioner(config_path: str):
    """Build the multi-conditioner from an exp-09 model config JSON (the real
    FLAC_AR_exp09.json in C1, a tiny config in tests)."""
    from src.models.conditioners import create_multi_conditioner_from_conditioning_config
    with open(config_path) as f:
        cfg = json.load(f)
    return create_multi_conditioner_from_conditioning_config(cfg["model"]["conditioning"])


# ============================================================================================
# Section B — smoke-log verification (finite loss + throughput floor).
# ============================================================================================
_IT_PER_S_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*it/s")
_S_PER_IT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*s/it")
_LOSS_RE = re.compile(
    r"(?:train/)?loss\s*=\s*([+-]?(?:nan|inf|infinity|[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?))",
    re.IGNORECASE,
)


def parse_throughput_steps_per_s(log_text: str) -> tp.List[float]:
    """Every throughput reading in ``log_text`` as steps/s. Handles BOTH progress-bar units:
    ``X.XXit/s`` -> ``X.XX`` and ``X.XXs/it`` -> ``1/X.XX`` (a sub-1/s run prints ``s/it``).
    With accum=1 (the pinned rung) one progress iteration == one optimizer step."""
    rates: tp.List[float] = []
    for m in _IT_PER_S_RE.finditer(log_text):
        rates.append(float(m.group(1)))
    for m in _S_PER_IT_RE.finditer(log_text):
        v = float(m.group(1))
        if v > 0:
            rates.append(1.0 / v)
    return rates


def best_throughput_steps_per_s(log_text: str) -> tp.Optional[float]:
    """The MAX instantaneous progress-bar rate, or None. DESCRIPTIVE ONLY (integrative-review r2
    blocker 2): this is NOT the acceptance statistic — a single fast tick amid many slow ticks
    must NOT pass a slow smoke. The gate uses the SUSTAINED rate (see :func:`verify_log`)."""
    rates = parse_throughput_steps_per_s(log_text)
    return max(rates) if rates else None


# tqdm/Lightning progress line: ``... N/M [<elapsed><<remaining>, rate ...]`` where <elapsed> is the
# CUMULATIVE wall time since the bar started and N is the completed iteration count.
_PROGRESS_RE = re.compile(r"(\d+)\s*/\s*\d+\s*\[\s*(\d+(?::\d{2})*)\s*<")


def _parse_hms(t: str) -> tp.Optional[int]:
    """Seconds from a tqdm elapsed field: ``H:MM:SS`` / ``MM:SS`` / ``SS`` (base-60 fold)."""
    try:
        nums = [int(p) for p in t.split(":")]
    except ValueError:
        return None
    secs = 0
    for n in nums:
        secs = secs * 60 + n
    return secs


def parse_sustained_from_log(log_text: str) -> tp.Optional[tp.Tuple[int, int]]:
    """The FINAL cumulative ``(completed_steps, elapsed_seconds)`` from the LAST progress line
    (``N/M [elapsed<remaining, ...]``). Returns None if no progress line is present or the values
    are non-positive. This yields the SUSTAINED rate ``completed_steps / elapsed`` — robust to a
    single fast tick, since it uses the run's own end-of-run cumulative accounting."""
    matches = list(_PROGRESS_RE.finditer(log_text))
    if not matches:
        return None
    m = matches[-1]
    steps = int(m.group(1))
    elapsed = _parse_hms(m.group(2))
    if steps <= 0 or elapsed is None or elapsed <= 0:
        return None
    return steps, elapsed


def parse_finite_losses(log_text: str) -> tp.Dict[str, tp.Any]:
    """Find every logged ``loss=<v>`` / ``train/loss=<v>`` and classify finiteness.
    Returns counts + the list of non-finite tokens (nan/inf). A run with NO loss line is
    NOT ok (nothing was verified)."""
    found: tp.List[str] = []
    bad: tp.List[str] = []
    for m in _LOSS_RE.finditer(log_text):
        tok = m.group(1)
        found.append(tok)
        try:
            v = float(tok)
        except ValueError:
            v = float("nan")
        if not math.isfinite(v):
            bad.append(tok)
    return {
        "n_loss_samples": len(found),
        "n_non_finite": len(bad),
        "non_finite_tokens": bad[:10],
        "all_finite": len(found) >= 1 and len(bad) == 0,
    }


def verify_log(log_text: str, min_steps_per_s: float = DEFAULT_MIN_STEPS_PER_S,
               sustained_steps: tp.Optional[int] = None,
               sustained_wall_s: tp.Optional[float] = None) -> tp.Dict[str, tp.Any]:
    """Verify a C1 smoke training log: (a) at least one FINITE loss was logged, and (b) the
    SUSTAINED throughput clears the floor. The gate is the SUSTAINED rate = completed steps /
    elapsed wall time (integrative-review r2 blocker 2), NEVER the max instantaneous tick:

      * if ``sustained_steps`` and ``sustained_wall_s`` are given (c1_smoke.sh times the run with
        bash ``SECONDS`` and passes the achieved step count), that AUTHORITATIVE cumulative rate is
        used — it cannot be inflated by one fast tick and is pessimistic-by-construction (includes
        startup), so it fails closed;
      * else the final cumulative ``(completed_steps, elapsed)`` is parsed from the LAST progress
        line.

    ``max_observed_steps_per_s`` is serialised as a DESCRIPTIVE field only. Fail-closed: a missing
    loss line, or an absent/unparseable sustained rate, both fail."""
    losses = parse_finite_losses(log_text)
    max_observed = best_throughput_steps_per_s(log_text)   # DESCRIPTIVE ONLY — not the gate

    sustained: tp.Optional[float] = None
    source: tp.Optional[str] = None
    if sustained_steps and sustained_wall_s and sustained_wall_s > 0:
        sustained = float(sustained_steps) / float(sustained_wall_s)
        source = "wall_clock_seconds"
    else:
        parsed = parse_sustained_from_log(log_text)
        if parsed is not None:
            steps, elapsed = parsed
            sustained = steps / elapsed
            source = "log_final_cumulative"

    throughput_ok = sustained is not None and sustained >= min_steps_per_s
    finite_loss_ok = bool(losses["all_finite"])
    return {
        "finite_loss_ok": finite_loss_ok,
        "loss": losses,
        "sustained_steps_per_s": sustained,       # THE GATE
        "sustained_source": source,
        "max_observed_steps_per_s": max_observed,  # descriptive only (would false-pass a slow run)
        "min_steps_per_s": min_steps_per_s,
        "throughput_ok": throughput_ok,
        "pass": finite_loss_ok and throughput_ok,
    }


# ============================================================================================
# CLI.
# ============================================================================================
def _cmd_count(args) -> int:
    import torch
    torch.manual_seed(args.seed)
    mc = build_conditioner(args.config)
    # The backbone-call count is STRUCTURAL (1 source + K context), independent of the data
    # values, so a fabricated K=8 batch through the REAL backbone (built from FLAC_AR_exp09.json)
    # is valid C1 evidence for the real dataset's K=8 => 9 criterion (review: "parametrize by K;
    # the C1 records will pin 9"). --k selects the dataset's context count (8 for AcousticRooms).
    metadata = fabricate_scene_batch(args.batch_size, k=args.k, seed=args.seed)
    rec = probe_counts(mc, metadata, device=args.device)
    rec["generated_by"] = "forward_counter_probe.py count"
    rec["source"] = "fabricated_structural_batch"
    if args.expect is not None:
        rec["expect_override"] = args.expect
        rec["pass"] = bool(rec["pass"] and rec["n_fa_invariant"] == args.expect)
    if args.out:
        atomic_write_json(args.out, rec)
    print(f"forward_counter_probe count: K={rec['K']} expected={rec['expected_backbone_calls_per_batch']} "
          f"n_vanilla={rec['n_vanilla']} n_fa={rec['n_fa_invariant']} pass={rec['pass']}")
    return 0 if rec["pass"] else 1


def _cmd_verify_log(args) -> int:
    with open(args.log, errors="replace") as f:
        text = f.read()
    rec = verify_log(text, args.min_steps_per_s,
                     sustained_steps=args.sustained_steps, sustained_wall_s=args.sustained_wall_s)
    rec["generated_by"] = "forward_counter_probe.py verify-log"
    rec["log"] = args.log
    if args.out:
        atomic_write_json(args.out, rec)
    print(f"forward_counter_probe verify-log: finite_loss_ok={rec['finite_loss_ok']} "
          f"sustained_steps_per_s={rec['sustained_steps_per_s']} ({rec['sustained_source']}; "
          f"floor {rec['min_steps_per_s']}) max_observed={rec['max_observed_steps_per_s']} "
          f"throughput_ok={rec['throughput_ok']} pass={rec['pass']}")
    return 0 if rec["pass"] else 1


def main(argv: tp.Optional[tp.Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="exp-09 C1 forward-counter probe + smoke-log verifier")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("count", help="count backbone forwards per batch (expect 1+K; real K=8 -> 9)")
    pc.add_argument("--config", required=True, help="exp-09 model config JSON (e.g. FLAC_AR_exp09.json)")
    pc.add_argument("--k", type=int, default=8, help="context K (AcousticRooms max_context=8; default 8)")
    pc.add_argument("--batch-size", type=int, default=2)
    pc.add_argument("--device", default="cpu")
    pc.add_argument("--seed", type=int, default=0)
    pc.add_argument("--expect", type=int, default=None, help="cross-check observed count against this")
    pc.add_argument("--out", default=None, help="atomic JSON output path")
    pc.set_defaults(func=_cmd_count)

    pv = sub.add_parser("verify-log", help="verify finite loss + SUSTAINED throughput floor in a smoke log")
    pv.add_argument("--log", required=True, help="C1 smoke training log path")
    pv.add_argument("--min-steps-per-s", type=float, default=DEFAULT_MIN_STEPS_PER_S)
    pv.add_argument("--sustained-steps", type=int, default=None,
                    help="authoritative completed step count (c1_smoke passes the achieved steps)")
    pv.add_argument("--sustained-wall-s", type=float, default=None,
                    help="authoritative training wall time in seconds (c1_smoke passes bash SECONDS)")
    pv.add_argument("--out", default=None, help="atomic JSON output path")
    pv.set_defaults(func=_cmd_verify_log)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
