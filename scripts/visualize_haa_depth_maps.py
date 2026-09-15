#!/usr/bin/env python3
"""Render the HAA panorama depth-map arrays as PNG images."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_depth(path: Path, flip_vertical: bool) -> np.ndarray:
    depth = np.load(path)
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2-D depth map, got {depth.shape} in {path}")

    # HAA stores the vertical axis opposite to the equirectangular convention
    # used by this repository (see HAA_md.py), so flip it for visualization.
    return np.flipud(depth) if flip_vertical else depth


def style_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, color="white", fontsize=15, pad=8)
    ax.set_xlabel("Panorama x (pixels)", color="#d1d5db")
    ax.set_ylabel("Panorama y (pixels)", color="#d1d5db")
    ax.tick_params(colors="#9ca3af", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#4b5563")


def add_depth_map(fig: plt.Figure, ax: plt.Axes, depth: np.ndarray, title: str):
    image = ax.imshow(depth, cmap="turbo", origin="upper", aspect="equal")
    style_axis(ax, title)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Depth", color="#d1d5db")
    colorbar.ax.tick_params(colors="#d1d5db", labelsize=8)
    colorbar.outline.set_edgecolor("#6b7280")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/HAA/depth_maps"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/HAA_depth_maps"))
    parser.add_argument(
        "--raw-orientation",
        action="store_true",
        help="Do not apply the vertical flip used by the repository's HAA loader.",
    )
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"No .npy files found in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    depths = [load_depth(path, flip_vertical=not args.raw_orientation) for path in paths]
    titles = [path.stem.removesuffix("_depth_image") for path in paths]

    rows = (len(paths) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(16, 4.5 * rows), squeeze=False,
                             constrained_layout=True)
    fig.patch.set_facecolor("#111827")
    fig.suptitle("HAA Panorama Depth Maps", color="white", fontsize=22,
                 fontweight="bold")
    for ax, depth, title in zip(axes.flat, depths, titles):
        add_depth_map(fig, ax, depth, title)
    for ax in axes.flat[len(paths):]:
        ax.set_visible(False)
    fig.savefig(args.output_dir / "depth_maps_contact_sheet.png", dpi=180,
                facecolor=fig.get_facecolor())
    plt.close(fig)

    for path, depth, title in zip(paths, depths, titles):
        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        fig.patch.set_facecolor("#111827")
        add_depth_map(fig, ax, depth, title)
        fig.savefig(args.output_dir / f"{path.stem}.png", dpi=180,
                    facecolor=fig.get_facecolor())
        plt.close(fig)


if __name__ == "__main__":
    main()
