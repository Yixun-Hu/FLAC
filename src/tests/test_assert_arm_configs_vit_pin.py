"""exp_25 rB-F1 A2 (Codex rA HIGH H5): the DINOv3 initializer gate of
``worklog/worklog_yixun/exp_07_fa_scratch_claude/assert_arm_configs.py`` hash-pins EVERY file
the loader consumes from the sole snapshot -- ``AutoModel.from_pretrained`` reads
``config.json`` as well as ``model.safetensors`` (a compatible config edit, e.g. dropout or the
layer-norm eps, would keep the weights loadable and change the run).

``assert_vit_pin`` and the ``VIT_*`` pins are extracted from the gate script by AST (importing
the script would build both arms). CPU only; the synthetic caches live in tmp.

Written by the exp_25 Coder seat (Claude Opus 5.5, max effort), round rB-F1.
"""
import ast
import hashlib
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GATE = os.path.join(REPO, "worklog", "worklog_yixun", "exp_07_fa_scratch_claude", "assert_arm_configs.py")
REPO_ID = "models--facebook--dinov3-vits16-pretrain-lvd1689m"


def _load():
    tree = ast.parse(open(GATE).read())
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name == "assert_vit_pin") or (
        isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) and t.id.startswith("VIT_") for t in n.targets))]
    ns = {}
    exec(compile(ast.Module(body=ast.parse("import hashlib, os").body + keep, type_ignores=[]), GATE, "exec"), ns)
    return ns


def _cache(tmp, files, rev=None, extra_snapshot=False):
    """A HF hub cache laid out like the real one: blobs + snapshot symlinks."""
    ns = _load()
    rev = rev or ns["VIT_REV"]
    hub = tmp / "hub" / REPO_ID
    (hub / "blobs").mkdir(parents=True)
    snap = hub / "snapshots" / rev
    snap.mkdir(parents=True)
    for name, data in files.items():
        blob = hub / "blobs" / hashlib.sha256(data).hexdigest()
        blob.write_bytes(data)
        (snap / name).symlink_to(blob)
    if extra_snapshot:
        (hub / "snapshots" / ("0" * 40)).mkdir()
    return str(tmp / "hub")


FILES = {"config.json": b'{"hidden_size": 384, "layer_norm_eps": 1e-05}', "model.safetensors": b"weights"}


def _pinned(ns):
    ns["VIT_FILES"] = {k: hashlib.sha256(v).hexdigest() for k, v in FILES.items()}
    return ns


def test_the_pins_name_every_consumed_file_with_the_measured_digests():
    ns = _load()
    assert sorted(ns["VIT_FILES"]) == ["config.json", "model.safetensors"]
    assert ns["VIT_FILES"]["config.json"] == "9481247be9f95a134a5599402b4bfc838eecdf9a7fffbf4debd1c70ec213898b"
    assert ns["VIT_FILES"]["model.safetensors"] == ns["VIT_SHA256"] == \
        "4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d"


def test_a_synthetic_snapshot_with_the_pinned_bytes_passes(tmp_path, capsys):
    ns = _pinned(_load())
    ns["assert_vit_pin"](hub_cache=_cache(tmp_path, FILES))
    assert "ViT pin OK" in capsys.readouterr().out


@pytest.mark.parametrize("files, kw, needle", [
    ({**FILES, "config.json": FILES["config.json"].replace(b"1e-05", b"1e-06")}, {}, "config.json"),
    ({**FILES, "model.safetensors": b"other weights"}, {}, "model.safetensors"),
    ({"model.safetensors": FILES["model.safetensors"]}, {}, "files"),
    ({**FILES, "preprocessor_config.json": b"{}"}, {}, "files"),
    (FILES, {"extra_snapshot": True}, "snapshots"),
], ids=["tampered_config", "tampered_weights", "config_missing", "unpinned_extra_file", "two_snapshots"])
def test_any_other_snapshot_content_refuses(tmp_path, files, kw, needle):
    ns = _pinned(_load())
    with pytest.raises(RuntimeError, match=needle):
        ns["assert_vit_pin"](hub_cache=_cache(tmp_path, files, **kw))


def test_a_dangling_snapshot_link_refuses(tmp_path):
    ns = _pinned(_load())
    cache = _cache(tmp_path, FILES)
    os.unlink(os.path.join(cache, REPO_ID, "blobs", hashlib.sha256(FILES["config.json"]).hexdigest()))
    with pytest.raises(RuntimeError, match="config.json"):
        ns["assert_vit_pin"](hub_cache=cache)


def test_the_real_cache_passes_when_present():
    from huggingface_hub.constants import HF_HUB_CACHE
    if not os.path.isdir(os.path.join(HF_HUB_CACHE, REPO_ID)):
        pytest.skip("no DINOv3 cache on this machine")
    _load()["assert_vit_pin"]()
