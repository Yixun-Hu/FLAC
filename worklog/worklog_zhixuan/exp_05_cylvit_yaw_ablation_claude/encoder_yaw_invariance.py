"""Phase 1 encoder-only yaw invariance probe for SimpleViT vs CylViT.

Run from the FLAC repository root:

    python worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude/encoder_yaw_invariance.py

The default probe uses deterministic synthetic geometry so Phase 1 can run even
when the full AcousticRooms tree is unavailable. It exercises the same yaw group
action as eval_FLAC.py: roll panorama columns and rotate x/y vector channels and
pose vectors together.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.yaw_rotation import rotate_scene_metadata
from src.models.conditioners import create_multi_conditioner_from_conditioning_config


EXP_DIR = Path("worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude")
SIMPLE_CONFIG = Path("src/configs/model_configs/FLAC/AR/FLAC_AR_SimpleViT.json")
CYL_CONFIG = Path("src/configs/model_configs/FLAC/AR/FLAC_AR_CylViT.json")


def _load_conditioner_config(config_path: Path) -> dict:
    with open(config_path) as f:
        model_config = json.load(f)
    return model_config["model"]["conditioning"]


def _build_conditioner(config_path: Path, device: str):
    conditioner = create_multi_conditioner_from_conditioning_config(_load_conditioner_config(config_path))
    return conditioner.eval().to(device)


def _synthetic_metadata(seed: int, img_h: int, img_w: int, n_context: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    depth = torch.randn(3, img_h, img_w, generator=generator)
    source = torch.randn(3, generator=generator)
    context = torch.randn(n_context, 3, generator=generator)
    return {
        "source": source,
        "source_vit": source.unsqueeze(0),
        "context_poses": context,
        "context_poses_vit": context,
        "depth": depth,
        "scene": f"synthetic_{seed}",
    }


def _angle_to_columns(angle_deg: float, img_w: int) -> int:
    return int(round(math.radians(angle_deg) * img_w / (2.0 * math.pi))) % img_w


def _embedding(conditioner, metadata: dict, key: str, device: str) -> torch.Tensor:
    with torch.no_grad():
        out = conditioner([metadata], device, only_ids=(key,))
    return out[key][0].detach().float().flatten().cpu()


def _metrics(base: torch.Tensor, rotated: torch.Tensor) -> dict:
    delta = rotated - base
    return {
        "max_abs": float(delta.abs().max()),
        "l2": float(torch.linalg.vector_norm(delta)),
        "cosine_distance": float(1.0 - F.cosine_similarity(base[None], rotated[None]).item()),
    }


def run_probe(args) -> list[dict]:
    torch.manual_seed(args.seed)
    device = args.device
    metadata = _synthetic_metadata(args.seed, args.img_h, args.img_w, args.n_context)

    models = {
        "simple_vit": _build_conditioner(SIMPLE_CONFIG, device),
        "cyl_vit": _build_conditioner(CYL_CONFIG, device),
    }
    keys = ("source_vit", "context_poses_vit") if args.include_context else ("source_vit",)
    angles = [float(a) for a in args.angles.split(",")]

    rows = []
    for model_name, conditioner in models.items():
        for key in keys:
            base = _embedding(conditioner, metadata, key, device)
            for angle in angles:
                rotated_md = rotate_scene_metadata(metadata, math.radians(angle), args.img_w)
                rotated = _embedding(conditioner, rotated_md, key, device)
                row = {
                    "model": model_name,
                    "conditioner_key": key,
                    "angle_deg": angle,
                    "roll_columns": _angle_to_columns(angle, args.img_w),
                }
                row.update(_metrics(base, rotated))
                rows.append(row)
    return rows


def write_outputs(rows: list[dict], out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    json_path = out_prefix.with_suffix(".json")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


def print_summary(rows: list[dict]) -> None:
    grouped = {}
    for row in rows:
        key = (row["model"], row["conditioner_key"])
        grouped.setdefault(key, []).append(row)

    print("\nSummary by model/key:")
    for (model, cond_key), group in grouped.items():
        max_abs = max(r["max_abs"] for r in group)
        mean_l2 = sum(r["l2"] for r in group) / len(group)
        mean_cos = sum(r["cosine_distance"] for r in group) / len(group)
        print(
            f"  {model:10s} {cond_key:17s} "
            f"max_abs={max_abs:.4e} mean_l2={mean_l2:.4e} mean_cos={mean_cos:.4e}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--img-h", type=int, default=256)
    parser.add_argument("--img-w", type=int, default=512)
    parser.add_argument("--n-context", type=int, default=4)
    parser.add_argument(
        "--angles",
        default="22.5,45,67.5,90,112.5,135,157.5,180,202.5,225,247.5,270,292.5,315,337.5,5,10,15",
    )
    parser.add_argument("--include-context", action="store_true")
    parser.add_argument("--out-prefix", default=str(EXP_DIR / "encoder_yaw_invariance_synthetic"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_probe(args)
    print_summary(rows)
    write_outputs(rows, Path(args.out_prefix))


if __name__ == "__main__":
    main()
