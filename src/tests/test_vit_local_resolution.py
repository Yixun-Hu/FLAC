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

* resolution PRIORITY, pinned with COMPETING roots so each relation is falsifiable
  on any machine (no reliance on della's real ``models/`` symlink): explicit dir >
  ``local_root`` arg > ``$FLAC_LOCAL_MODEL_ROOT`` > ``<repo_root>/models`` >
  passthrough. The repo-root rule is redirected in tests by monkeypatching
  ``_default_local_root``; the production derivation itself is pinned separately.
* root derivation is anchored on THIS MODULE'S FILE, never the CWD: a decoy
  ``./models/<basename>`` in the process CWD must never be picked up.
* hostile/edge basenames (``.``/``..``) never escape a root, snapshot dirs that are
  SYMLINKS resolve normally (that is della's actual shape), and a BROKEN symlink is
  skipped rather than returned.
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
# the pure resolver: priority pinned with COMPETING roots (hermetic on any machine)
# --------------------------------------------------------------------------------------

def _snapshot(parent):
    """Create ``<parent>/<_HUB_BASENAME>`` (parents included) and return its path."""
    d = os.path.join(str(parent), _HUB_BASENAME)
    os.makedirs(d)
    return d


def _competing_roots(tmp_path, monkeypatch, local=True, env=True, repo=True):
    """Build all three roots, each optionally *holding a snapshot of the same
    basename*, so every priority relation is falsifiable: a resolver that swapped
    two rules, dropped one, or searched roots before an explicit dir would return a
    different path. ``$FLAC_LOCAL_MODEL_ROOT`` is set and the repo-root rule is
    redirected at ``_default_local_root``, so the outcome never depends on whether
    this checkout has della's real ``models/`` symlink.

    Returns ``(roots, snapshots)`` keyed by the source tag each root should yield.
    """
    roots, snapshots = {}, {}
    for tag, populated in (("local-root-arg", local), ("env-root", env), ("repo-root", repo)):
        root = tmp_path / tag
        root.mkdir()
        roots[tag] = str(root)
        if populated:
            snapshots[tag] = _snapshot(root)
    monkeypatch.setenv(_ENV_VAR, roots["env-root"])
    monkeypatch.setattr(conditioners, "_default_local_root", lambda: roots["repo-root"])
    return roots, snapshots


def test_existing_dir_beats_all_roots(tmp_path, monkeypatch):
    """An input that is already a directory is a deliberate explicit choice: it wins
    even when all three roots hold a competing snapshot of the same basename."""
    roots, snapshots = _competing_roots(tmp_path, monkeypatch)
    explicit = _snapshot(tmp_path / "explicit")

    assert resolve_vit_model_path(explicit, local_root=roots["local-root-arg"]) == (
        explicit,
        "explicit-dir",
    )
    assert explicit not in snapshots.values()


def test_local_root_arg_beats_env_root(tmp_path, monkeypatch):
    """Both the explicit ``local_root`` and $FLAC_LOCAL_MODEL_ROOT hold the snapshot:
    the argument wins (a caller's explicit root outranks the ambient environment)."""
    roots, snapshots = _competing_roots(tmp_path, monkeypatch)
    assert resolve_vit_model_path(_HUB_ID, local_root=roots["local-root-arg"]) == (
        snapshots["local-root-arg"],
        "local-root-arg",
    )


def test_env_root_beats_repo_root(tmp_path, monkeypatch):
    """Both the env root and <repo_root>/models hold the snapshot: the env var wins,
    so an operator can point a job at a scratch snapshot without touching the
    checkout. (Repo root is a synthetic directory here, not della's models/.)"""
    roots, snapshots = _competing_roots(tmp_path, monkeypatch, local=False)
    assert resolve_vit_model_path(_HUB_ID) == (snapshots["env-root"], "env-root")


def test_repo_root_fires_when_it_is_the_only_snapshot(tmp_path, monkeypatch):
    """With local_root absent and the env root present but EMPTY, the repo-root rule
    fires — and it is never the CWD: the process runs from a directory holding its
    own decoy ``models/<basename>``, which must be ignored."""
    roots, snapshots = _competing_roots(tmp_path, monkeypatch, local=False, env=False)
    cwd = tmp_path / "elsewhere"
    decoy = _snapshot(cwd / "models")
    monkeypatch.chdir(cwd)

    resolved, source = resolve_vit_model_path(_HUB_ID)
    assert (resolved, source) == (snapshots["repo-root"], "repo-root")
    assert resolved != decoy, "resolver used the CWD instead of the repo root"


def test_default_local_root_is_repo_models_independent_of_cwd(tmp_path, monkeypatch):
    """The production repo-root derivation itself (unpatched): ``<repo_root>/models``
    computed from this module's file, identical before and after a chdir."""
    expected = os.path.join(_REPO_ROOT, "models")
    assert conditioners._default_local_root() == expected
    monkeypatch.chdir(tmp_path)
    assert conditioners._default_local_root() == expected


def test_hub_id_without_snapshot_unchanged(tmp_path, monkeypatch):
    """No snapshot under ANY root -> the input passes through unchanged, so normal
    hub/cache behavior (and its offline error) still applies."""
    roots, _ = _competing_roots(tmp_path, monkeypatch, local=False, env=False, repo=False)
    assert resolve_vit_model_path(_HUB_ID, local_root=roots["local-root-arg"]) == (
        _HUB_ID,
        "passthrough",
    )


@pytest.mark.parametrize("weird", ["facebook/..", "facebook/.", "models/..", "a/b/.."])
def test_dot_basenames_never_escape_a_root(tmp_path, monkeypatch, weird):
    """``os.path.join(root, "..")`` is a real directory, so an unguarded resolver
    would "resolve" a dot basename to the ROOT'S PARENT. Such ids are unresolvable
    by construction: they must pass through untouched."""
    roots, _ = _competing_roots(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    assert resolve_vit_model_path(weird, local_root=roots["local-root-arg"]) == (
        weird,
        "passthrough",
    )


def test_bare_dot_input_is_returned_verbatim(tmp_path, monkeypatch):
    """A bare ``.``/``..`` input is itself an existing directory, so the explicit-dir
    rule returns it VERBATIM — nothing is joined against a root, so no root is
    escaped there either (the loader receives exactly what the config said)."""
    roots, _ = _competing_roots(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    for literal in (".", ".."):
        assert resolve_vit_model_path(literal, local_root=roots["local-root-arg"]) == (
            literal,
            "explicit-dir",
        )


def test_symlinked_snapshot_resolves(tmp_path, monkeypatch):
    """della's snapshot is reached through symlinks (``models`` -> scratch), so a
    snapshot entry that IS a symlink to a real directory must resolve normally."""
    roots, _ = _competing_roots(tmp_path, monkeypatch, local=False, env=False, repo=False)
    real = tmp_path / "real_snapshot_on_scratch"
    real.mkdir()
    link = os.path.join(roots["local-root-arg"], _HUB_BASENAME)
    os.symlink(str(real), link)

    assert resolve_vit_model_path(_HUB_ID, local_root=roots["local-root-arg"]) == (
        link,
        "local-root-arg",
    )


def test_broken_symlink_snapshot_is_skipped(tmp_path, monkeypatch):
    """A dangling snapshot symlink (scratch wiped under the link) is NOT a snapshot:
    resolution continues to the next root, and to passthrough when none is valid —
    the broken path is never returned."""
    roots, snapshots = _competing_roots(tmp_path, monkeypatch, local=False)
    broken = os.path.join(roots["local-root-arg"], _HUB_BASENAME)
    os.symlink(str(tmp_path / "was-wiped"), broken)

    assert resolve_vit_model_path(_HUB_ID, local_root=roots["local-root-arg"]) == (
        snapshots["env-root"],
        "env-root",
    )

    monkeypatch.delenv(_ENV_VAR)
    monkeypatch.setattr(conditioners, "_default_local_root", lambda: str(tmp_path / "gone"))
    assert resolve_vit_model_path(_HUB_ID, local_root=roots["local-root-arg"]) == (
        _HUB_ID,
        "passthrough",
    )


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
    """The load line carries the EXACT ``{original} -> {resolved} [{tag}]`` triple
    (not the three tokens scattered across the log); when the resolution redirected
    and a model.safetensors is present, its byte size and sha256 prefix are logged
    as the run log's only ViT-weights provenance."""
    payload = b"flac-exp16-provenance-probe" * 4
    snapshot = _fake_root_with_snapshot(tmp_path, monkeypatch, safetensors_bytes=payload)
    _patch_transformers(monkeypatch)

    create_multi_conditioner_from_conditioning_config(_conditioning())

    out = capsys.readouterr().out
    assert f"{_FAKE_ID} -> {snapshot} [env-root]" in out
    assert f"{len(payload)} bytes" in out
    assert f"sha256:{hashlib.sha256(payload).hexdigest()[:16]}" in out


def test_callsite_without_safetensors_logs_no_provenance(tmp_path, monkeypatch, capsys):
    """A snapshot directory with no model.safetensors (config-only dir, the
    from_scratch case) must not crash the load: the provenance helper returns None
    on OSError and the call site prints the resolution line only."""
    snapshot = _fake_root_with_snapshot(tmp_path, monkeypatch)
    _patch_transformers(monkeypatch)

    assert conditioners._vit_weights_provenance(snapshot) is None

    create_multi_conditioner_from_conditioning_config(_conditioning())

    out = capsys.readouterr().out
    assert f"{_FAKE_ID} -> {snapshot} [env-root]" in out
    assert "sha256:" not in out
    assert "model.safetensors" not in out
