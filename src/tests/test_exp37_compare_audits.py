"""exp_37 round B-c: the audit comparator (``tools/exp37_compare_audits.py``) -- two geometry audits are equal except
for the host-dependent path fields (top-level mesh_root / context_manifest, every room's mesh_path) and the payload
sha256 derived from them. Used to prove the parameterized audit tool reproduces the frozen 0.5 m audit on this box.
Written by the exp_37 Coder (Claude Opus 5.5, max effort), round B-c.
"""
import copy

from tools import exp37_compare_audits as cmp

AUDIT = {"mesh_root": "/a", "context_manifest": "c.json", "sha256": "x", "grid_spacing_m": [0.5] * 3,
         "rooms": {"R": {"mesh_path": "/a/S/R.obj", "base_points_sha256": "b", "z_oracle": {"max": 0.4}}},
         "queries": [{"index": 1, "z_count": 3}]}


def test_only_the_host_path_fields_and_the_sha_are_ignored():
    other = copy.deepcopy(AUDIT)
    other.update(mesh_root="/b", context_manifest="/abs/c.json", sha256="y")
    other["rooms"]["R"]["mesh_path"] = "/b/S/R.obj"
    assert cmp.audit_differences(AUDIT, other) == []


def test_any_other_difference_is_reported_with_its_path():
    other = copy.deepcopy(AUDIT)
    other["rooms"]["R"]["z_oracle"]["max"] = 0.41
    other["queries"][0]["z_count"] = 4
    other["grid_spacing_m"] = [0.75] * 3
    assert cmp.audit_differences(AUDIT, other) == ["grid_spacing_m", "queries[0].z_count", "rooms.R.z_oracle.max"]
    del other["rooms"]["R"]["base_points_sha256"]
    assert "rooms.R.base_points_sha256" in cmp.audit_differences(AUDIT, other)
