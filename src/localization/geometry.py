"""Compatibility adapter for Zhixuan's exp_09 visualizer.

`/home/yixunhu/codespace/visualize_candidate_grid_case.py` (authored against the
Frame_Average checkout's ``src.localization.geometry``) expects tuple/mask
returns and a scene object carrying ``aabb_min``/``aabb_max``/``path``. This
module maps that interface onto exp_22's registered implementation in
``meshgrid_geometry`` — no geometry logic lives here, and the clearance
arguments the visualizer passes are ASSERTED equal to the registered constants
rather than honored as overrides. Visualization-only; batched into the next
review round per universal coverage.
"""
import numpy as np

from . import meshgrid_geometry as mg

def build_lattice(aabb_min, aabb_max, spacing=0.5):
    spacing = np.asarray(spacing, dtype=np.float64).reshape(-1)
    if spacing.size == 3:
        assert np.allclose(spacing, spacing[0]), "registered lattice is isotropic"
        spacing = spacing[0]
    else:
        spacing = float(spacing[0]) if spacing.size else 0.5
    return mg.build_lattice(aabb_min, aabb_max, float(spacing))


class _SceneShim:
    def __init__(self, scene, path):
        self._scene = scene
        self.path = path
        lo, hi = mg.scene_aabb(scene)
        self.aabb_min = np.asarray(lo, dtype=np.float64)
        self.aabb_max = np.asarray(hi, dtype=np.float64)


def load_raycast_scene(path, compute_topology=None):  # signature per visualizer
    return _SceneShim(mg.load_raycast_scene(str(path)), str(path))


def classify_mesh_candidates(scene_shim, points, clearance):
    assert float(clearance) == float(mg.SURFACE_CLEARANCE), clearance
    out = mg.classify_mesh_candidates(scene_shim._scene, points, clearance=clearance)
    return out["valid"], out


def filter_query_candidates(candidates, receiver, context_sources,
                            receiver_clearance=None, context_clearance=None,
                            z_band=None, eps=mg.EPS):
    if receiver_clearance is not None:
        assert float(receiver_clearance) == float(mg.RECEIVER_MIN_DISTANCE)
    if context_clearance is not None:
        assert float(context_clearance) == float(mg.CONTEXT_GUARD_RADIUS)
    out = mg.filter_query_candidates(candidates, receiver,
                                     context_sources=context_sources,
                                     z_band=z_band, eps=eps)
    return out["mask"]
