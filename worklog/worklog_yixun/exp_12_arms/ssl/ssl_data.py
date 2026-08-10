"""exp_12 arm B -- native cylindrical SSL: data pipeline.

The backbone consumes an ego-centred displacement field, exactly as
`GeometryConditioner.forward` builds it (src/models/conditioners.py):

    field = (src_loc - rec_loc)[:, None, None] - points(depth[rec])        # [3, H, W]

with `max_value = 1` in the production config, so there is no rescaling. `points()` is
`AR_md.convert_equirect_to_camera_coord`; this module reproduces it with a cached unit-ray
grid (~10x faster) and `test_ssl.py` asserts exact agreement with the production function.

CORPUS = THE 243 TRAIN ROOMS ONLY. The 17 `unseen_eval` rooms share scene names with train
rooms (`Cafe_idx_0` train vs `Cafe_idx_1` unseen) but no room id, so the guard is on room
ids and it is enforced at index-build time AND re-checked at Dataset construction.

A sample is ONE ROOM. Its views are different (receiver, source) entries of that room:
two 256x512 globals and `n_local` 128x256 locals. Different receivers in the same room are
the physically meaningful positive pair -- room-level acoustics (T60) is what the
downstream conditioner must capture.

NOT AUGMENTED WITH AZIMUTH ROLL, deliberately: the backbone is *exactly* roll-equivariant,
so a rolled view produces an identically-permuted patch field and a bit-identical patch
mean. Roll augmentation contributes zero gradient here; it is a no-op that would only burn
GPU time. `test_ssl.py::test_roll_is_a_literal_noop_for_the_losses` proves it.
"""

from __future__ import annotations

import json
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

H_FULL, W_FULL = 256, 512
PATCH = 16


# --------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------
def unit_ray_grid(img_h: int = H_FULL, img_w: int = W_FULL) -> torch.Tensor:
    """[3, H, W] unit direction per equirect pixel -- the depth-independent factor of
    AR_md.convert_equirect_to_camera_coord (same pixel-centre convention, same axes)."""
    phi, theta = torch.meshgrid(torch.arange(img_h), torch.arange(img_w), indexing="ij")
    theta_map = (theta + 0.5) * 2.0 * np.pi / img_w - np.pi
    phi_map = (phi + 0.5) * np.pi / img_h - np.pi / 2
    return torch.stack(
        [
            torch.cos(phi_map) * torch.cos(theta_map),
            torch.cos(phi_map) * torch.sin(theta_map),
            -torch.sin(phi_map),
        ],
        dim=0,
    )


_RAY_CACHE: dict[tuple[int, int], torch.Tensor] = {}


def rays(img_h: int, img_w: int) -> torch.Tensor:
    key = (img_h, img_w)
    if key not in _RAY_CACHE:
        _RAY_CACHE[key] = unit_ray_grid(img_h, img_w)
    return _RAY_CACHE[key]


def depth_to_points(depth: torch.Tensor) -> torch.Tensor:
    """[H, W] depth -> [3, H, W] point cloud in the receiver frame."""
    return depth.unsqueeze(0) * rays(depth.shape[-2], depth.shape[-1]).to(depth.dtype)


def displacement_field(depth: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """The production ViT input: (query - surface_point), [3, H, W]."""
    return query.view(3, 1, 1) - depth_to_points(depth)


def downsample_depth(depth: torch.Tensor, factor: int) -> torch.Tensor:
    """Lower-resolution panorama = area-averaged depth (still a full 360 deg field)."""
    if factor == 1:
        return depth
    h, w = depth.shape
    return depth.view(h // factor, factor, w // factor, factor).mean(dim=(1, 3))


def elevation_flip(field: torch.Tensor) -> torch.Tensor:
    """Mirror the room through the horizontal plane: flip rows AND negate z.

    An exact symmetry of the equirect layout (row v <-> H-1-v maps elevation to its
    negative), so the flipped field is a valid geometry of the mirrored room."""
    out = torch.flip(field, dims=(-2,)).clone()
    out[2] = -out[2]
    return out


# --------------------------------------------------------------------------------------
# corpus index
# --------------------------------------------------------------------------------------
@dataclass
class RoomIndex:
    scene: str
    room: str
    receivers: list[int]                     # receiver ids that have a depth panorama
    rec_loc: dict[int, list[float]]          # world position per receiver
    src_loc: dict[int, list[float]]          # world position per source
    pairs: list[tuple[int, int]]             # (src, rec) pairs from the TRAIN manifest


def _parse_pair(fname: str) -> tuple[int, int]:
    """'S001_R0044_hybrid_IR.wav' -> (src=1, rec=44). Mirrors AR_md's parsing."""
    stem = fname.split("/")[-1].split(".")[0]
    parts = stem.split("_")
    return int(parts[0][1:]), int(parts[1][1:])


def _meta_path(meta_root: str, scene: str, room: str, src: int, rec: int) -> str:
    # AR_md.get_receiver_source_location's exact spelling.
    return os.path.join(meta_root, scene, room, f"S00{src}_R00{rec}.json")


def build_index(
    dataset_root: str,
    manifest_path: str,
    forbidden_manifest_path: str | None = None,
    max_workers: int = 32,
) -> list[RoomIndex]:
    """Index the SSL corpus from a manifest.

    `forbidden_manifest_path` (unseen_eval.json) is a HARD guard: any room appearing in it
    is refused, not silently skipped, so an accidental manifest swap cannot leak the
    evaluation rooms into pretraining.
    """
    manifest = json.load(open(manifest_path))
    forbidden: set[tuple[str, str]] = set()
    if forbidden_manifest_path is not None:
        f = json.load(open(forbidden_manifest_path))
        forbidden = {(s, r) for s in f for r in f[s]}

    depth_root = os.path.join(dataset_root, "depth_map")
    meta_root = os.path.join(dataset_root, "metadata")

    jobs: list[tuple[str, str, list[str]]] = []
    for scene in sorted(manifest):
        for room in sorted(manifest[scene]):
            if (scene, room) in forbidden:
                raise ValueError(
                    f"REFUSE: room {scene}/{room} is in the held-out manifest "
                    f"{forbidden_manifest_path}; it must never enter SSL pretraining."
                )
            jobs.append((scene, room, manifest[scene][room]))

    def one(job) -> RoomIndex | None:
        scene, room, files = job
        ddir = os.path.join(depth_root, scene, room)
        if not os.path.isdir(ddir):
            return None
        receivers = sorted(int(f[:-4]) for f in os.listdir(ddir) if f.endswith(".npy"))
        rset = set(receivers)
        pairs = [p for p in (_parse_pair(f) for f in files) if p[1] in rset]
        if not pairs:
            return None

        # Every json holds BOTH src_loc and rec_loc, so a greedy cover over the pairs
        # recovers every position with ~max(n_src, n_rec) reads instead of one per pair.
        src_loc: dict[int, list[float]] = {}
        rec_loc: dict[int, list[float]] = {}
        for src, rec in pairs:
            if src in src_loc and rec in rec_loc:
                continue
            path = _meta_path(meta_root, scene, room, src, rec)
            if not os.path.exists(path):
                continue
            m = json.load(open(path))
            src_loc.setdefault(src, m["src_loc"])
            rec_loc.setdefault(rec, m["rec_loc"])
        pairs = [(s, r) for (s, r) in pairs if s in src_loc and r in rec_loc]
        if not pairs:
            return None
        return RoomIndex(scene, room, sorted({r for _, r in pairs}), rec_loc, src_loc, pairs)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        out = [r for r in ex.map(one, jobs) if r is not None]
    return out


def save_index(index: list[RoomIndex], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    blob = [
        {
            "scene": r.scene,
            "room": r.room,
            "receivers": r.receivers,
            "rec_loc": {str(k): v for k, v in r.rec_loc.items()},
            "src_loc": {str(k): v for k, v in r.src_loc.items()},
            "pairs": r.pairs,
        }
        for r in index
    ]
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(blob, fh)
    os.replace(tmp, path)


def load_index(path: str) -> list[RoomIndex]:
    blob = json.load(open(path))
    return [
        RoomIndex(
            b["scene"],
            b["room"],
            b["receivers"],
            {int(k): v for k, v in b["rec_loc"].items()},
            {int(k): v for k, v in b["src_loc"].items()},
            [tuple(p) for p in b["pairs"]],
        )
        for b in blob
    ]


# --------------------------------------------------------------------------------------
# iBOT block masking (wrap-around in azimuth, so the mask law is roll-equivariant too)
# --------------------------------------------------------------------------------------
def block_mask(
    n_h: int,
    n_w: int,
    ratio: tuple[float, float],
    rng: random.Random,
    max_tries: int = 40,
) -> torch.Tensor:
    """BEiT/iBOT-style block mask on the patch grid, [n_h, n_w] bool."""
    n = n_h * n_w
    target = int(n * rng.uniform(*ratio))
    mask = torch.zeros(n_h, n_w, dtype=torch.bool)
    if target <= 0:
        return mask
    filled, tries = 0, 0
    while filled < target and tries < max_tries:
        tries += 1
        area = rng.uniform(0.05, 0.35) * target + 4
        aspect = math.exp(rng.uniform(math.log(0.3), math.log(3.3)))
        h = max(1, min(n_h, int(round(math.sqrt(area * aspect)))))
        w = max(1, min(n_w, int(round(math.sqrt(area / aspect)))))
        top = rng.randint(0, n_h - h)
        left = rng.randint(0, n_w - 1)
        cols = [(left + i) % n_w for i in range(w)]        # wraps the seam
        before = int(mask.sum())
        mask[top : top + h, cols] = True
        filled += int(mask.sum()) - before
    return mask


# --------------------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------------------
class RoomViewDataset(Dataset):
    """One item = one room, rendered as 2 global + n_local views from distinct entries."""

    def __init__(
        self,
        index: list[RoomIndex],
        dataset_root: str,
        n_local: int = 4,
        local_factor: int = 2,
        mask_ratio: tuple[float, float] = (0.1, 0.5),
        mask_prob: float = 0.5,
        elev_flip_prob: float = 0.5,
        forbidden_rooms: set[tuple[str, str]] | None = None,
        seed: int = 0,
    ):
        if forbidden_rooms:
            leaked = [(r.scene, r.room) for r in index if (r.scene, r.room) in forbidden_rooms]
            if leaked:
                raise ValueError(f"REFUSE: held-out rooms present in the SSL index: {leaked[:5]}")
        self.index = index
        self.depth_root = os.path.join(dataset_root, "depth_map")
        self.n_local = n_local
        self.local_factor = local_factor
        self.mask_ratio = mask_ratio
        self.mask_prob = mask_prob
        self.elev_flip_prob = elev_flip_prob
        self.seed = seed
        self.n_h, self.n_w = H_FULL // PATCH, W_FULL // PATCH

    def __len__(self) -> int:
        return len(self.index)

    def _depth(self, scene: str, room: str, rec: int) -> torch.Tensor:
        a = np.load(os.path.join(self.depth_root, scene, room, f"{rec}.npy"))
        return torch.from_numpy(a).float()

    def _view(self, room: RoomIndex, pair: tuple[int, int], factor: int) -> torch.Tensor:
        src, rec = pair
        depth = downsample_depth(self._depth(room.scene, room.room, rec), factor)
        query = torch.tensor(room.src_loc[src], dtype=torch.float32) - torch.tensor(
            room.rec_loc[rec], dtype=torch.float32
        )
        return displacement_field(depth, query)

    def __getitem__(self, i: int) -> dict:
        room = self.index[i]
        rng = random.Random((self.seed * 1_000_003 + i * 7919 + torch.initial_seed()) % (2**31))

        # Prefer distinct receivers across views: different vantage points of one room are
        # the positive pair that carries real signal.
        by_rec: dict[int, list[tuple[int, int]]] = {}
        for s, r in room.pairs:
            by_rec.setdefault(r, []).append((s, r))
        recs = list(by_rec)
        rng.shuffle(recs)
        need = 2 + self.n_local
        chosen = [rng.choice(by_rec[recs[k % len(recs)]]) for k in range(need)]

        flip = rng.random() < self.elev_flip_prob
        globals_, locals_ = [], []
        for k, pair in enumerate(chosen):
            factor = 1 if k < 2 else self.local_factor
            v = self._view(room, pair, factor)
            if flip:
                v = elevation_flip(v)
            (globals_ if k < 2 else locals_).append(v)

        masks = []
        for _ in range(2):
            if rng.random() < self.mask_prob:
                masks.append(block_mask(self.n_h, self.n_w, self.mask_ratio, rng))
            else:
                masks.append(torch.zeros(self.n_h, self.n_w, dtype=torch.bool))

        return {
            "globals": torch.stack(globals_),                     # [2, 3, 256, 512]
            "locals": torch.stack(locals_) if locals_ else torch.zeros(0),
            "masks": torch.stack(masks).flatten(1),               # [2, n_h*n_w] bool
            "room_id": i,
        }
