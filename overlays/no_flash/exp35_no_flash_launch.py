#!/usr/bin/env python3
"""exp_35 no-flash launcher (plan_orbitrir_yaw_evidence.md r2 §11 B1, r3 §12 H4).

Runs ONE target -- ``eval_FLAC.py`` or the B-F composition probe -- in THIS interpreter with the
FLAC attention backend pinned to torch SDPA, fail-closed, in this order:

  (1) ``sys.dont_write_bytecode = True``; ``PYTHONDONTWRITEBYTECODE=1`` must be in the environment;
  (2) refused: anything under ``overlays/no_flash/`` but the overlay and this launcher (a
      ``__pycache__``, a ``*.pyc``, a ``flash_attn/`` package, ...), any ``__init__.py`` or bytecode
      under ``overlays/``, and a ``flash_attn`` module / package at the repo root or in any path
      entry ahead of site-packages;
  (3) ``sys.path`` = [the overlay dir, the repo root, the interpreter's own entries minus every other
      occurrence of those two, every inherited PYTHONPATH entry and the cwd]; children inherit
      ``PYTHONPATH=<overlay dir>`` only;
  (4) ``find_spec("flash_attn")`` is the overlay file (a source module whose bytes hash to the pin),
      and importing it raises exactly the overlay's ImportError;
  (5) ``src.models.transformer`` is ``<repo>/src/models/transformer.py`` (``src`` is ``<repo>/src``)
      and its ``flash_attn_func`` is None, so its SDPA branch runs;
  (6) ONE sentinel line: the asserted backend, the overlay's and this file's sha256, the target;
  (7) ``runpy.run_path(target, run_name="__main__")`` with the original argv.

Any failed check prints ``NO-FLASH LAUNCH REFUSED`` and exits 86 before the target runs.
``--preflight`` stops after (6). Never imported by anything (it refuses bytecode next to itself).

Written by the exp_35 Coder (Claude Opus 5.5, max effort) per plan_orbitrir_yaw_evidence.md r2/r3.
"""
import hashlib
import importlib
import importlib.machinery
import importlib.util
import os
import runpy
import site
import sys

OVERLAY_DIR = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.dirname(os.path.dirname(OVERLAY_DIR))
OVERLAY_FILE = os.path.join(OVERLAY_DIR, "flash_attn.py")
OVERLAY_SHA256 = "0d0e52ca693e236a2e0a44cd61f61539fdf995cff5132d1f621181a0f3e7c023"   # plan §3b
OVERLAY_MESSAGE = "exp35 no-flash overlay: flash_attn disabled, attention backend = torch SDPA"
ALLOWED = ("exp35_no_flash_launch.py", "flash_attn.py")
SENTINEL = "EXP35_NO_FLASH_LAUNCHER"
BACKEND = "sdpa"
PREFLIGHT = "--preflight"
REFUSED = 86


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def flash_candidates(directory):
    """The entries of ``directory`` the import system could bind to the name ``flash_attn``."""
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    return sorted(n for n in names if n == "flash_attn" or n.startswith("flash_attn."))


def layout_reasons(overlay_dir=OVERLAY_DIR, repo=REPO):
    """(2) on disk: the overlay dir holds exactly the two committed files; nothing under overlays/
    is a package marker or bytecode; the repo root holds no flash_attn shadow."""
    out = []
    try:
        names = sorted(os.listdir(overlay_dir))
    except OSError as err:
        return [f"overlays/no_flash ({overlay_dir}) is unreadable ({err})"]
    if names != sorted(ALLOWED):
        out.append(f"overlays/no_flash holds {names}, not exactly {sorted(ALLOWED)} (bytecode, a package or "
                   "any other entry could shadow or bypass the pinned overlay)")
    for root, dirs, files in os.walk(os.path.dirname(overlay_dir)):
        bad = [d for d in dirs if d == "__pycache__"] + [f for f in files if f == "__init__.py"
                                                         or f.endswith((".pyc", ".pyo"))]
        if bad:
            out.append(f"overlays holds {bad} in {root} (no package marker, no bytecode)")
    hits = flash_candidates(repo)
    if hits:
        out.append(f"the repo root {repo} holds {hits}: a flash_attn shadow")
    return out


def build_path(current, pythonpath, overlay_dir=OVERLAY_DIR, repo=REPO):
    """(3) [overlay dir, repo root] + the interpreter's own entries, minus every other occurrence of
    those two, every inherited PYTHONPATH entry and '' (the cwd)."""
    real = os.path.realpath
    drop = {real(overlay_dir), real(repo)} | {real(p) for p in (pythonpath or "").split(os.pathsep) if p}
    return [overlay_dir, repo] + [p for p in current if p and real(p) not in drop]


def path_shadow_reasons(entries, overlay_dir=OVERLAY_DIR):
    """(2) on the built path: no flash_attn candidate ahead of the first site-packages entry, except
    the overlay itself."""
    sites = {os.path.realpath(p) for p in site.getsitepackages() + [site.getusersitepackages()]}
    out = []
    for entry in entries:
        real = os.path.realpath(entry)
        if real in sites:
            break
        hits = [] if real == os.path.realpath(overlay_dir) else flash_candidates(entry)
        if hits:
            out.append(f"the path entry {entry} holds {hits} ahead of site-packages: a flash_attn shadow")
    return out


def spec_reasons(overlay_file=OVERLAY_FILE):
    """(4) the name resolves to the pinned overlay source file and nothing else."""
    if "flash_attn" in sys.modules:
        return [f"flash_attn is already imported ({getattr(sys.modules['flash_attn'], '__file__', None)!r}) "
                "before the launcher ran"]
    spec = importlib.util.find_spec("flash_attn")
    if spec is None:
        return ["find_spec('flash_attn') finds nothing: the overlay is not on the path"]
    out = []
    if not spec.origin or os.path.realpath(spec.origin) != os.path.realpath(overlay_file):
        out.append(f"find_spec('flash_attn').origin is {spec.origin!r}, not the overlay {overlay_file}")
    if spec.submodule_search_locations is not None or type(spec.loader) is not importlib.machinery.SourceFileLoader:
        out.append(f"flash_attn resolves to {type(spec.loader).__name__} / a package, not the overlay source module")
    got = sha256_file(overlay_file)
    if got != OVERLAY_SHA256:
        out.append(f"the overlay {overlay_file} hashes to {got}, not the pinned {OVERLAY_SHA256}")
    return out


def import_reasons():
    """(4) importing flash_attn raises exactly the overlay's ImportError and leaves nothing behind."""
    try:
        importlib.import_module("flash_attn")
    except ImportError as err:
        if str(err) != OVERLAY_MESSAGE:
            return [f"importing flash_attn raised {err!r}, not the overlay's ImportError"]
        return ["flash_attn stayed in sys.modules after its import failed"] if "flash_attn" in sys.modules else []
    return ["importing flash_attn SUCCEEDED: a real flash_attn is reachable"]


def transformer_reasons(repo=REPO):
    """(5) this checkout's transformer, on its SDPA branch (``flash_attn_func is None``): every file on its
    import chain REALLY is the file at its lexical path under the canonical ``repo`` -- a symlinked file,
    or a symlinked ``src`` / ``src/models`` directory, leading out of the checkout is refused."""
    try:
        mod = importlib.import_module("src.models.transformer")
    except Exception as err:                       # noqa: BLE001 -- an import failure is a refusal
        return [f"importing src.models.transformer failed ({type(err).__name__}: {err})"]
    out = []
    for name, rel in (("src", ("src", "__init__.py")), ("src.models", ("src", "models", "__init__.py")),
                      ("src.models.transformer", ("src", "models", "transformer.py"))):
        want = os.path.join(repo, *rel)            # lexical: REPO is canonical, nothing on this side resolves
        got = os.path.realpath(getattr(sys.modules.get(name), "__file__", None) or "")
        if got != want:
            out.append(f"{name} is {got}, not this checkout's {want}")
    if getattr(mod, "flash_attn_func", "MISSING") is not None:
        out.append("src.models.transformer.flash_attn_func is not None: the flash-attention branch would run")
    return out


def sentinel_line(target, overlay_sha256, launcher_sha256):
    return (f"{SENTINEL} attention_backend={BACKEND} overlay_sha256={overlay_sha256} "
            f"launcher_sha256={launcher_sha256} target={target}")


def _refuse(reasons):
    sys.stdout.flush()
    print("NO-FLASH LAUNCH REFUSED (nothing was run):\n" + "".join(f"  - {r}\n" for r in reasons),
          file=sys.stderr, end="", flush=True)
    return REFUSED


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    sys.dont_write_bytecode = True                                                  # (1)
    if not argv:
        return _refuse(["usage: exp35_no_flash_launch.py (--preflight | TARGET [ARG ...])"])
    target, rest = argv[0], argv[1:]
    reasons = [] if os.environ.get("PYTHONDONTWRITEBYTECODE") == "1" else [
        "PYTHONDONTWRITEBYTECODE=1 is required in the environment"]
    if target == PREFLIGHT and rest:
        reasons.append("--preflight takes no further argument")
    if target != PREFLIGHT and not os.path.isfile(target):
        reasons.append(f"the target {target!r} is not a file")
    path = build_path(sys.path, os.environ.get("PYTHONPATH"))                       # (3)
    reasons += layout_reasons() + path_shadow_reasons(path)                         # (2)
    if reasons:
        return _refuse(reasons)
    sys.path[:] = path
    os.environ["PYTHONPATH"] = OVERLAY_DIR
    importlib.invalidate_caches()
    reasons = spec_reasons() or import_reasons()                                    # (4)
    reasons = reasons or transformer_reasons()                                      # (5)
    if reasons:
        return _refuse(reasons)
    print(sentinel_line(target, sha256_file(OVERLAY_FILE), sha256_file(os.path.realpath(__file__))),
          flush=True)                                                               # (6)
    sys.stderr.flush()
    if target == PREFLIGHT:
        return 0
    sys.argv = [target] + rest                                                      # (7)
    runpy.run_path(target, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
