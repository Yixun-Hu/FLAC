"""exp_38 — the AR content binding (exp38_content_manifest.py; Codex code r2 BLOCKING 3).

The file set is the one AR_md reads for the largest split (target wav, its metadata JSON under AR_md's
own ``S00{s}_R00{r}`` name rebuild, the receiver's depth panorama); the manifest groups per-file
hashes per room; verification re-hashes every byte: a same-name / different-byte file, a missing
file, an extra or missing listed file, and a changed manifest all refuse. Plus the committed real
manifest's identity and coverage.
"""
import gzip
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(REPO, "worklog", "worklog_yixun", "exp_38_data_efficiency_claude")
sys.path.insert(0, EXP)
import exp38_content_manifest as cm  # noqa: E402

reg = cm.reg


def _toy(tmp_path, monkeypatch):
    split = {"Office": {"Office_idx_1": ["S001_R002_hybrid_IR.wav", "S0010_R002_hybrid_IR.wav", "S003_R005_hybrid_IR.wav"]},
             "Cafe": {"Cafe_idx_9": ["S002_R001_hybrid_IR.wav"]}}
    sp = tmp_path / "split.json"
    sp.write_text(json.dumps(split))
    monkeypatch.setitem(reg.FRACS[cm.CONTENT_SPLIT_TAG], "split", str(sp))
    root = tmp_path / "AR"
    for scene, rooms in split.items():
        for room, names in rooms.items():
            for n in names:
                s, r = int(n.split("_")[0][1:]), int(n.split("_")[1][1:])
                for rel, data in ((f"single_channel_ir_1/{scene}/{room}/{n}", b"wav" + n.encode()),
                                  (f"metadata/{scene}/{room}/S00{s}_R00{r}.json", b"{}"),
                                  (f"depth_map/{scene}/{room}/{r}.npy", b"depth" + str(r).encode())):
                    (root / os.path.dirname(rel)).mkdir(parents=True, exist_ok=True)
                    (root / rel).write_bytes(data)
    return root


def test_the_file_set_is_what_ar_md_reads(tmp_path, monkeypatch):
    _toy(tmp_path, monkeypatch)
    files = cm.content_files()
    assert "metadata/Office/Office_idx_1/S0010_R002.json" in files          # AR_md's "S00" + str(int) rebuild
    assert "metadata/Office/Office_idx_1/S0010_R0002.json" not in files
    assert [f for f in files if f.startswith("depth_map/Office")] == ["depth_map/Office/Office_idx_1/2.npy",
                                                                       "depth_map/Office/Office_idx_1/5.npy"]
    assert len(files) == 4 + 4 + 3


def test_build_then_verify_and_every_mutation(tmp_path, monkeypatch):
    root = _toy(tmp_path, monkeypatch)
    files = cm.content_files()
    hashes = cm.hash_files(str(root), files, threads=4)
    man = str(tmp_path / "m.txt.gz")
    digest = cm.write_manifest(man, cm.manifest_text(hashes))
    assert cm.verify(man, str(root), want_sha=digest) == []
    assert cm.read_manifest(man)[1] == digest and len(cm.read_manifest(man)[0]) == 6   # 2 rooms x 3 sub-trees
    with pytest.raises(reg.GateError):
        cm.write_manifest(man, "x")                                        # written once
    assert cm.verify(man, str(root), want_sha="0" * 64)                    # not the pinned manifest
    wav = root / "single_channel_ir_1/Office/Office_idx_1/S003_R005_hybrid_IR.wav"
    orig = wav.read_bytes()
    wav.write_bytes(orig[:-1] + bytes([orig[-1] ^ 1]))                     # same name, same size, other bytes
    assert any("differ" in p for p in cm.verify(man, str(root), want_sha=digest))
    wav.write_bytes(orig)
    (root / "depth_map/Cafe/Cafe_idx_9/1.npy").unlink()
    assert any("unreadable" in p for p in cm.verify(man, str(root), want_sha=digest))
    split = json.load(open(reg.FRACS[cm.CONTENT_SPLIT_TAG]["split"]))
    split["Cafe"]["Cafe_idx_9"].append("S004_R001_hybrid_IR.wav")          # the split implies another file
    json.dump(split, open(reg.FRACS[cm.CONTENT_SPLIT_TAG]["split"], "w"))
    assert any("groups" in p for p in cm.verify(man, str(root), want_sha=digest))


def test_the_manifest_is_byte_stable(tmp_path, monkeypatch):
    root = _toy(tmp_path, monkeypatch)
    text = cm.manifest_text(cm.hash_files(str(root), cm.content_files(), threads=2))
    cm.write_manifest(str(tmp_path / "a.gz"), text)
    cm.write_manifest(str(tmp_path / "b.gz"), text)
    assert (tmp_path / "a.gz").read_bytes() == (tmp_path / "b.gz").read_bytes()


def test_the_committed_manifest_is_the_pinned_one_and_covers_the_split():
    path = reg._abs(reg.AR_CONTENT_MANIFEST)
    rows, digest = cm.read_manifest(path, reg.AR_CONTENT_SHA256)
    files = cm.content_files()
    assert digest == reg.AR_CONTENT_SHA256 and sum(v[1] for v in rows.values()) == len(files) == 445414
    assert sorted(rows) == sorted({cm.group_of(f) for f in files}) and len(rows) == 729
    for tag in ("025", "050"):                                              # the nested splits are covered
        sub = json.load(open(reg._abs(reg.FRACS[tag]["split"])))
        names = {f"single_channel_ir_1/{s}/{r}/{n}" for s, rooms in sub.items() for r, ns in rooms.items() for n in ns}
        assert names <= set(files)
    assert gzip.decompress(open(path, "rb").read()).decode().count("\n") == 729
