"""exp_36, cycle 11: the Codex code review I HIGH 1 follow-ups of the publication (the set selector landed in cycle 10):
the PARSED CLI publishes the OFF set's 12 destinations into its own manifest, the two sets never share a default
manifest path, and a manifest written by later code keeps the artifacts' own provenance -- the pin H36_I (461c58e0) and
its campaign digest -- apart from the publishing checkout's HEAD.

Written by the exp_36 Coder (Claude Opus 5.5, max effort) per plan_yaw_c4_spread.md r0 + A1 + Codex code review I.
"""
import json

import test_depth_canon_collect as T
from test_exp36_publish import _cli, pub, pub_off, publish  # noqa: F401 -- the publication worlds
from test_exp36_replay_check import H36, world  # noqa: F401

C = T.C
PUBLISHER = "f" * 40


def test_cli_publishes_the_off_set_into_its_own_manifest(pub_off, tmp_path, monkeypatch, capsys):
    assert _cli(pub_off, tmp_path, monkeypatch, "--exp36-set", "OFF") == 0
    assert "EXP36 PUBLISHED 12 file(s) (set OFF)" in capsys.readouterr().out
    doc = json.load(open(pub_off["out"]))
    assert doc["set"] == "OFF" and len(doc["entries"]) == 12 and doc["status"] == "complete"


def test_the_default_manifest_paths_differ_per_set():
    a, off = C.EXP36_PUBLICATION.format("A"), C.EXP36_PUBLICATION.format("OFF")
    assert a != off and off.endswith("yaw_c4_spread_publication_manifest_OFF.json")
    assert a.startswith(C.EXP36_DIR) and sorted(C.EXP36_PUBLISH_SETS) == ["A", "OFF"]
    assert len(C.EXP36_PUBLISH_SETS["A"]) == 24 and len(C.EXP36_PUBLISH_SETS["OFF"]) == 6


def test_the_manifest_keeps_the_artifact_pin_apart_from_the_publishing_head(pub):
    pub["heads"][C.REPO] = PUBLISHER
    status, doc = publish(pub)
    assert status == "published", doc.get("reasons")
    assert (doc["pin"], doc["campaign_digest"]) == (H36, T.DIGEST)                # the artifacts' provenance
    assert doc["publisher"] == {"repo": C.REPO, "head": PUBLISHER}                # the code that copied them
    assert json.load(open(pub["out"]))["publisher"] == doc["publisher"]
