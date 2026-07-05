"""Offline Metric-1 (invariance gap) comparison between two stored prediction tensors.

Both ``.pt`` files are produced by ``eval_FLAC.py --store_predictions`` (the
``decoded_samples_all`` tensor, saved at ``eval_FLAC.py:192-194``): a single
``[N, 1, T]`` float32 CPU tensor of *decoded, un-clamped* RIR waveforms in
dataloader order. A reference run (``P_ref``, unrotated baseline) and an
alternative run (``P_alt``, yaw-rotated conditioning) share dataset config,
batch size and seed, so sample ``i`` corresponds across the two files.

Reported quantities:

* Waveform gap: per-sample mean-absolute-difference and relative L2
  ``||alt - ref|| / (||ref|| + 1e-8)`` over the common minimum length, averaged
  over the dataset.
* Acoustic gap: T60 / C50 / EDT from :class:`AcousticMetricsCallback`, treating
  ``P_ref`` as the ground truth and ``P_alt`` as the prediction (e.g. a relative
  T60 error of ``alt`` w.r.t. ``ref``). FD / retrieval / env stay disabled so no
  AGREE checkpoint is needed.
* Exact-match: max absolute difference between the two tensors, the
  rot0-vs-baseline determinism control (should be ~0 for identical runs).
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

# The repo ships a `src` package that also has a stale copy installed in
# site-packages; prepend the real repo root (three levels up:
# worklog/exp_02_.../compare_predictions.py -> repo root) so `src.*` resolves
# to the working tree, not the stale install.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.metrics.metric_callback import AcousticMetricsCallback  # noqa: E402


def _to_float(value: Any) -> Any:
    """Convert a scalar tensor to a Python float, passing other types through."""
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value.tolist()
    return value


def waveform_gap(ref: torch.Tensor, alt: torch.Tensor, batch_size: int) -> Dict[str, float]:
    """Per-sample waveform mean-abs-diff and relative L2, averaged over the dataset.

    Parameters
    ----------
    ref, alt : torch.Tensor
        Prediction tensors ``[N, 1, T]`` (already trimmed to a common length).
    batch_size : int
        Mini-batch size used to bound memory.

    Returns
    -------
    dict
        ``mean_abs_diff``, ``mean_rel_l2`` and ``max_abs_diff`` over all samples.
    """
    n = ref.shape[0]
    mad_sum, rel_l2_sum, max_abs = 0.0, 0.0, 0.0
    for start in range(0, n, batch_size):
        r = ref[start:start + batch_size].float().flatten(1)
        a = alt[start:start + batch_size].float().flatten(1)
        diff = a - r
        mad_sum += diff.abs().mean(dim=1).sum().item()
        rel_l2_sum += (diff.norm(dim=1) / (r.norm(dim=1) + 1e-8)).sum().item()
        max_abs = max(max_abs, diff.abs().max().item())
    return {
        "mean_abs_diff": mad_sum / n,
        "mean_rel_l2": rel_l2_sum / n,
        "max_abs_diff": max_abs,
    }


def acoustic_gap(
    ref: torch.Tensor, alt: torch.Tensor, batch_size: int, sample_rate: int, device: str
) -> Dict[str, Any]:
    """T60 / C50 / EDT of ``alt`` w.r.t. ``ref`` via AcousticMetricsCallback.

    ``P_ref`` is passed as the ground truth and ``P_alt`` as the prediction, so
    the returned errors quantify how far the rotated run drifts from baseline.
    FD / retrieval / env are disabled; no AGREE checkpoint is required.
    """
    callback = AcousticMetricsCallback(
        dataset_name="AcousticRooms",
        sample_rate=sample_rate,
        sample_size=sample_rate,
        audio_channels=1,
        eval_per_scene=False,
        device=device,
        eval_T60=True,
        eval_C50=True,
        eval_EDT=True,
        AGREE_ckpt=None,
    )
    n = ref.shape[0]
    for start in range(0, n, batch_size):
        pred = alt[start:start + batch_size].float().to(device)
        gt = ref[start:start + batch_size].float().to(device)
        # scene=None follows the non-per-scene AR path; T60 then includes every sample.
        callback.update_metrics("test", pred, gt, scene=None)
    metrics = callback.compute_metrics("test")
    return {key: _to_float(val) for key, val in metrics.items()}


def _load_raw(path: str):
    """Load a prediction file, supporting both storage formats.

    Returns ``(tensor, meta)`` where ``meta`` is ``None`` for the legacy bare
    tensor file and the sidecar dict for the newer
    ``{"predictions": tensor, "meta": {...}}`` format written by
    ``eval_FLAC.py --store_predictions``. The tensor is returned un-normalised.
    """
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "predictions" in obj:
        return obj["predictions"], obj.get("meta")
    return obj, None


def _normalise(tensor: Any, path: str) -> torch.Tensor:
    """Validate a prediction tensor and normalise it to ``[N, 1, T]``."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{path} did not contain a tensor (got {type(tensor)})")
    if tensor.dim() == 2:  # [N, T] -> [N, 1, T]
        tensor = tensor.unsqueeze(1)
    if tensor.dim() != 3:
        raise ValueError(f"Expected a 2D/3D prediction tensor, got shape {tuple(tensor.shape)}")
    return tensor


def load_predictions(path: str) -> torch.Tensor:
    """Load a stored prediction tensor and normalise it to ``[N, 1, T]``.

    Accepts both the legacy bare-tensor file and the new sidecar dict
    ``{"predictions": tensor, "meta": {...}}``; any meta is ignored here (use
    :func:`load_prediction_meta` / :func:`guard_meta` for the consistency check).
    """
    tensor, _meta = _load_raw(path)
    return _normalise(tensor, path)


def load_prediction_meta(path: str):
    """Return the sidecar meta dict for a prediction file, or ``None`` (legacy)."""
    _tensor, meta = _load_raw(path)
    return meta


# Keys that must match for the comparison to be meaningful: dataset_config /
# seed / batch_size so sample i corresponds across the two files, plus
# cond_method / frame_avg_angles so both runs used the same conditioning method
# (a vanilla-vs-fa_invariant or different-angle-set gap is a method difference,
# not a Metric-1 invariance gap). ``rotate_deg`` is intentionally EXEMPT:
# comparing a rotated run against an unrotated baseline is this tool's purpose.
_GUARD_KEYS = ("dataset_config", "seed", "batch_size", "cond_method", "frame_avg_angles")


def guard_meta(meta_ref, meta_alt, ref_path: str = "ref", alt_path: str = "alt") -> None:
    """Hard-error when two runs' sidecar meta make their predictions incomparable.

    When BOTH files carry meta, a mismatch on any of :data:`_GUARD_KEYS`
    (``dataset_config`` / ``seed`` / ``batch_size`` -- sample correspondence --
    plus ``cond_method`` / ``frame_avg_angles`` -- method comparability) makes
    the Metric-1 comparison meaningless -> :class:`ValueError`. ``rotate_deg``
    is deliberately not guarded: rotated-vs-unrotated is the point of the tool.
    When only one side (or neither) carries meta the check is skipped with a
    warning (legacy bare-tensor interop, so old exp_02 artifacts still compare).
    """
    if meta_ref is None and meta_alt is None:
        return
    if meta_ref is None or meta_alt is None:
        present = alt_path if meta_ref is None else ref_path
        print(
            f"WARNING: only {present} carries prediction meta; skipping the "
            "meta-consistency check (legacy bare-tensor interop)."
        )
        return
    mismatches = [
        f"{key}: ref={meta_ref.get(key)!r} vs alt={meta_alt.get(key)!r}"
        for key in _GUARD_KEYS
        if meta_ref.get(key) != meta_alt.get(key)
    ]
    if mismatches:
        raise ValueError(
            "Refusing to compare mismatched prediction runs: "
            + "; ".join(mismatches) + ". Re-run both with the same "
            "dataset_config, seed, batch_size, cond_method and frame_avg_angles "
            "(only rotate_deg may differ)."
        )


def compare(
    ref_path: str, alt_path: str, batch_size: int, sample_rate: int, device: str, allow_trim: bool = False
) -> Dict[str, Any]:
    """Run the full comparison and return the assembled results dict."""
    ref_raw, ref_meta = _load_raw(ref_path)
    alt_raw, alt_meta = _load_raw(alt_path)
    # Refuse to compare runs whose meta disagrees on any _GUARD_KEYS entry
    # (both present); single-sided meta warns only (legacy bare-tensor interop).
    guard_meta(ref_meta, alt_meta, ref_path, alt_path)
    ref = _normalise(ref_raw, ref_path)
    alt = _normalise(alt_raw, alt_path)
    if tuple(ref.shape) != tuple(alt.shape):
        if not allow_trim:
            raise ValueError(
                f"Shape mismatch: ref {tuple(ref.shape)} vs alt {tuple(alt.shape)}. "
                "Pass --allow-trim to trim to the common length instead of erroring."
            )
        if ref.shape[0] != alt.shape[0]:
            raise ValueError(f"Sample count mismatch: ref has {ref.shape[0]}, alt has {alt.shape[0]}")
        print(
            f"WARNING: shape mismatch ref {tuple(ref.shape)} vs alt {tuple(alt.shape)}; "
            "trimming both to the common length (tail differences dropped)."
        )

    common_len = min(ref.shape[-1], alt.shape[-1])
    ref = ref[..., :common_len].contiguous()
    alt = alt[..., :common_len].contiguous()

    results: Dict[str, Any] = {
        "ref_path": ref_path,
        "alt_path": alt_path,
        "num_samples": int(ref.shape[0]),
        "common_length": int(common_len),
        "waveform_gap": waveform_gap(ref, alt, batch_size),
        "acoustic_gap": acoustic_gap(ref, alt, batch_size, sample_rate, device),
    }
    return results


def print_table(results: Dict[str, Any]) -> None:
    """Print a readable summary of the comparison to stdout."""
    print("=" * 60)
    print("Metric-1 invariance-gap comparison (alt vs. ref)")
    print("=" * 60)
    print(f"ref: {results['ref_path']}")
    print(f"alt: {results['alt_path']}")
    print(f"samples: {results['num_samples']}   common length: {results['common_length']}")
    print("-" * 60)
    print("Waveform gap:")
    for key, val in results["waveform_gap"].items():
        print(f"  {key:<16s} {val:.6g}")
    print("Acoustic gap (P_ref as GT, P_alt as prediction):")
    for key, val in results["acoustic_gap"].items():
        pretty = f"{val:.6g}" if isinstance(val, float) else str(val)
        print(f"  {key:<16s} {pretty}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", type=str, required=True, help="Reference (baseline) predictions .pt")
    parser.add_argument("--alt", type=str, required=True, help="Alternative (rotated) predictions .pt")
    parser.add_argument("--out", type=str, required=True, help="Output JSON path")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for metric computation",
    )
    parser.add_argument("--sample-rate", type=int, default=22050, help="Audio sample rate (Hz)")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size to bound memory")
    parser.add_argument(
        "--allow-trim",
        action="store_true",
        help="If ref/alt shapes differ, trim both to the common length instead of erroring",
    )
    args = parser.parse_args()

    results = compare(args.ref, args.alt, args.batch_size, args.sample_rate, args.device, args.allow_trim)
    print_table(results)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
