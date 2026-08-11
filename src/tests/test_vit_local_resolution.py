"""Tests for local-snapshot resolution of the ViT backbone id (exp_16, della).

Why this exists: della's compute nodes are OFFLINE, so any ``from_pretrained``
that has to reach the hub fails there. The released FLAC checkpoint shows the
paper run itself loaded its ViT from a *local directory*
(``./Models/dinov3-vits16-pretrain-lvd1689m`` in the checkpoint's embedded
``model_config``), while this repo's JSON configs carry the portable hub id
``facebook/dinov3-vits16-pretrain-lvd1689m``. ``resolve_vit_model_path`` maps the
hub id onto the repo-level ``models/`` snapshot (a symlink into scratch on della)
*at load time*, so the JSON configs stay byte-unchanged and checkpoints keep
embedding the portable hub id.

Surfaces under test:

* pure resolver: existing dir passes through as ``explicit-dir``; hub id resolves
  under an explicit ``local_root`` (``local-root-arg``), under
  ``$FLAC_LOCAL_MODEL_ROOT`` (``env-root``), or under ``<repo_root>/models``
  (``repo-root``); no snapshot anywhere -> ``passthrough`` with the input
  unchanged.
* root derivation is anchored on THIS MODULE'S FILE, never the CWD: a decoy
  ``./models/<basename>`` in the process CWD must never be picked up.
* call sites in ``create_multi_conditioner_from_conditioning_config``: BOTH the
  ``from_scratch`` (``AutoConfig.from_pretrained`` -> ``AutoModel.from_config``)
  and the normal (``AutoModel.from_pretrained``) branches receive the RESOLVED
  path, and the load line names the original id, the resolved path and the rule
  that fired (the run log's only ViT-weights provenance).

No network, no GPU, no DINOv3 weights: the transformers entry points are
monkeypatched at the ``src.models.conditioners`` module level, and the "snapshot"
directories are ``tmp_path`` fakes. The repo-root/``src/tests`` conftest prepends
this checkout to ``sys.path`` so ``src.*`` resolves here, not to a stale
``pip install .`` copy.
"""
import hashlib
import os
import types

import pytest
from torch import nn

import src.models.conditioners as conditioners
from src.models.conditioners import (
    create_multi_conditioner_from_conditioning_config,
    resolve_vit_model_path,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ENV_VAR = "FLAC_LOCAL_MODEL_ROOT"

# The real backbone id carried by every FLAC_AR*.json.
_HUB_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
_HUB_BASENAME = "dinov3-vits16-pretrain-lvd1689m"

# A backbone id that exists nowhere on disk (used where "no snapshot" is the point).
_ABSENT_ID = "facebook/flac-test-no-such-vit-snapshot"

# The id used to drive the call-site tests: distinctive, so resolution can only
# come from the fake root the test controls.
_FAKE_ID = "facebook/flac-test-fake-vit"
_FAKE_BASENAME = "flac-test-fake-vit"

_HIDDEN = 8


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every test states its own root rules; never inherit an ambient one."""
    monkeypatch.delenv(_ENV_VAR, raising=False)


# --------------------------------------------------------------------------------------
# 1-5: the pure resolver
# --------------------------------------------------------------------------------------

def test_existing_dir_returned_unchanged(tmp_path):
    """A path that is already a directory is a deliberate explicit choice: it is
    returned byte-identically, tagged ``explicit-dir``, with no root search."""
    d = tmp_path / "some-local-vit"
    d.mkdir()
    assert resolve_vit_model_path(str(d)) == (str(d), "explicit-dir")


def test_hub_id_resolves_to_local_snapshot(tmp_path):
    """A hub id whose basename exists under the explicit ``local_root`` resolves to
    that directory, tagged ``local-root-arg``."""
    models_root = tmp_path / "models_root"
    snapshot = models_root / _HUB_BASENAME
    snapshot.mkdir(parents=True)
    assert resolve_vit_model_path(_HUB_ID, local_root=str(models_root)) == (
        str(snapshot),
        "local-root-arg",
    )


def test_hub_id_without_snapshot_unchanged(tmp_path):
    """No snapshot under any root -> the input passes through unchanged so the
    normal hub/cache behavior (and its offline error) still applies."""
    empty_root = tmp_path / "empty_root"
    empty_root.mkdir()
    assert resolve_vit_model_path(_ABSENT_ID, local_root=str(empty_root)) == (
        _ABSENT_ID,
        "passthrough",
    )


def test_env_var_root_wins_over_repo_root(tmp_path, monkeypatch):
    """$FLAC_LOCAL_MODEL_ROOT is consulted before <repo_root>/models, so an operator
    can point a job at a scratch snapshot without touching the checkout."""
    env_root = tmp_path / "scratch_models"
    snapshot = env_root / _HUB_BASENAME
    snapshot.mkdir(parents=True)
    monkeypatch.setenv(_ENV_VAR, str(env_root))

    resolved, source = resolve_vit_model_path(_HUB_ID)
    assert (resolved, source) == (str(snapshot), "env-root")
    assert resolved != os.path.join(_REPO_ROOT, "models", _HUB_BASENAME)


def test_repo_root_derived_from_file_not_cwd(tmp_path, monkeypatch):
    """The repo root comes from ``conditioners.__file__``, never the CWD: running
    from a directory that contains its own decoy ``models/<basename>`` must NOT
    resolve to that decoy. On a checkout that has the real ``models/`` snapshot
    (della), the repo snapshot is returned with tag ``repo-root``."""
    decoy = tmp_path / "models" / _HUB_BASENAME
    decoy.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    resolved, source = resolve_vit_model_path(_HUB_ID)
    assert resolved != str(decoy), "resolver used the CWD instead of the repo root"

    repo_snapshot = os.path.join(_REPO_ROOT, "models", _HUB_BASENAME)
    if os.path.isdir(repo_snapshot):
        assert (resolved, source) == (repo_snapshot, "repo-root")
    else:  # checkout without the della symlink: still must not use the CWD
        assert (resolved, source) == (_HUB_ID, "passthrough")


# --------------------------------------------------------------------------------------
# 6-8: the call sites in create_multi_conditioner_from_conditioning_config
# --------------------------------------------------------------------------------------

class _StubViT(nn.Module):
    """Minimal stand-in for the DINOv3 backbone: enough for the ViTCoordinates
    branch (``.config.hidden_size``, ``.parameters()``) and nothing more."""

    def __init__(self, hidden_size: int = _HIDDEN):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=hidden_size)
        self.lin = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):  # pragma: no cover - never called in these tests
        return x


class _Recorder:
    """Records the first positional arg of each patched entry point."""

    def __init__(self):
        self.from_pretrained_args = []
        self.from_config_args = []


def _patch_transformers(monkeypatch):
    """Patch AutoModel/AutoConfig as looked up inside src.models.conditioners."""
    rec = _Recorder()
    stub = _StubViT()

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(name_or_path, *args, **kwargs):
            rec.from_pretrained_args.append(name_or_path)
            return stub

        @staticmethod
        def from_config(config, *args, **kwargs):
            return stub

    class _FakeAutoConfig:
        @staticmethod
        def from_pretrained(name_or_path, *args, **kwargs):
            rec.from_config_args.append(name_or_path)
            return types.SimpleNamespace(hidden_size=_HIDDEN)

    monkeypatch.setattr(conditioners, "AutoModel", _FakeAutoModel)
    monkeypatch.setattr(conditioners, "AutoConfig", _FakeAutoConfig)
    return rec


def _fake_root_with_snapshot(tmp_path, monkeypatch, safetensors_bytes=None):
    """Create <tmp>/fake_models/<_FAKE_BASENAME> and point $FLAC_LOCAL_MODEL_ROOT at
    it, so the call-site tests control resolution without depending on the real
    ``models/`` symlink. Returns the snapshot path."""
    root = tmp_path / "fake_models"
    snapshot = root / _FAKE_BASENAME
    snapshot.mkdir(parents=True)
    if safetensors_bytes is not None:
        (snapshot / "model.safetensors").write_bytes(safetensors_bytes)
    monkeypatch.setenv(_ENV_VAR, str(root))
    return str(snapshot)


def _conditioning(from_scratch=None):
    """Smallest conditioning block that exercises the ViTCoordinates branch."""
    vit = {"hf_model_name_or_path": _FAKE_ID}
    if from_scratch is not None:
        vit["from_scratch"] = from_scratch
    return {
        "cond_dim": _HIDDEN,
        "configs": [
            {"id": "source_vit", "type": "ViTCoordinates", "config": {"ViT": vit}},
        ],
    }


def test_callsite_from_pretrained_uses_resolved_path(tmp_path, monkeypatch):
    """from_scratch absent -> AutoModel.from_pretrained is called with the RESOLVED
    local snapshot path, not the hub id (this is what makes an offline node work)."""
    snapshot = _fake_root_with_snapshot(tmp_path, monkeypatch)
    rec = _patch_transformers(monkeypatch)

    mc = create_multi_conditioner_from_conditioning_config(_conditioning())

    assert rec.from_pretrained_args == [snapshot]
    assert rec.from_config_args == []
    assert isinstance(mc.conditioners["source_vit"].vit, _StubViT)


def test_callsite_from_config_uses_resolved_path(tmp_path, monkeypatch):
    """from_scratch true -> AutoConfig.from_pretrained (the only path that touches
    disk/hub in that branch) also receives the RESOLVED path."""
    snapshot = _fake_root_with_snapshot(tmp_path, monkeypatch)
    rec = _patch_transformers(monkeypatch)

    create_multi_conditioner_from_conditioning_config(_conditioning(from_scratch=True))

    assert rec.from_config_args == [snapshot]
    assert rec.from_pretrained_args == []


def test_callsite_logs_resolved_path(tmp_path, monkeypatch, capsys):
    """The load line names BOTH the original id and the resolved path plus the rule
    that fired; when the resolution redirected and a model.safetensors is present,
    its byte size and sha256 prefix are logged as weights provenance."""
    payload = b"flac-exp16-provenance-probe" * 4
    snapshot = _fake_root_with_snapshot(tmp_path, monkeypatch, safetensors_bytes=payload)
    _patch_transformers(monkeypatch)

    create_multi_conditioner_from_conditioning_config(_conditioning())

    out = capsys.readouterr().out
    assert _FAKE_ID in out
    assert snapshot in out
    assert "env-root" in out
    assert str(len(payload)) in out
    assert hashlib.sha256(payload).hexdigest()[:16] in out
