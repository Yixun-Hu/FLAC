#!/usr/bin/env python3
"""exp_46 -- the staged, no-clobber NAS publication of a COMPLETE neuronic run (L7; Codex code r2 finding 8).

``stage`` creates ``<dest root>/<run>.incoming`` EXCLUSIVELY before anything is copied (os.mkdir: an existing entry --
a directory, a file or a symlink, dangling or not -- refuses, and so does an existing ``<dest root>/<run>``); the
dest root itself must be a real, isolated directory (never created here: a missing mount refuses). rsync then fills
the staging directory and exp_46's own admission (sha256sum -c, ``exp46_registry.py training-check``) runs on it.
``publish`` re-validates the staging directory (a real directory below the dest root, never a link, holding the
sealed run's TRAINING_COMPLETE.json and TRAINING_COMMITTED.json) and moves it to ``<dest root>/<run>`` through a
NO-REPLACE rename (renameat2 RENAME_NOREPLACE): a final path created meanwhile refuses -- it is never replaced and
the staging directory is never nested inside it (as ``mv`` would). A filesystem without RENAME_NOREPLACE falls back
to os.rename after a fresh lexists check: a directory rename never nests and never replaces a file, a link or a
non-empty directory.

    python -B exp46_nas_publish.py stage   --dest-root <dir> --run <run id>
    python -B exp46_nas_publish.py publish --dest-root <dir> --run <run id>

Exit 0 = done (the path is printed), 2 = refused. Written by the exp_46 Coder (Claude Opus 5.5).
"""
import argparse
import ctypes
import errno
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import exp46_registry as reg  # noqa: E402

RENAME_NOREPLACE = 1             # <linux/fs.h>
AT_FDCWD = -100


def _paths(dest_root, run):
    reg.parse_run_id(run)
    root = os.path.abspath(dest_root)
    if os.path.islink(root) or not os.path.isdir(root):
        raise reg.GateError(f"dest root {root} is not a real directory (it is never created here)")
    reg.check_isolation(root, "the NAS publication root")
    return root, os.path.join(root, f"{run}.incoming"), os.path.join(root, run)


def stage(dest_root, run):
    """Create the staging directory exclusively, before any copy; refused if it or the final path exists."""
    root, incoming, final = _paths(dest_root, run)
    for p in (final, incoming):
        if os.path.lexists(p):
            raise reg.GateError(f"{p} exists (a staging or published run is never reused): refused")
        reg.check_destination(p, root, "the NAS publication")
    os.mkdir(incoming)                  # exclusive: EEXIST for any entry created meanwhile, a symlink included
    return incoming


def _rename_noreplace(src, dst):
    libc = ctypes.CDLL(None, use_errno=True)
    fn = getattr(libc, "renameat2", None)
    if fn is not None:
        if fn(AT_FDCWD, os.fsencode(src), AT_FDCWD, os.fsencode(dst), RENAME_NOREPLACE) == 0:
            return "renameat2"
        err = ctypes.get_errno()
        if err == errno.EEXIST:
            raise FileExistsError(errno.EEXIST, "the published path exists", dst)
        if err not in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
            raise OSError(err, os.strerror(err), src)
    if os.path.lexists(dst):            # no RENAME_NOREPLACE here: a fresh check, then a directory rename (it never
        raise FileExistsError(errno.EEXIST, "the published path exists", dst)    # nests, never replaces a file / link /
    os.rename(src, dst)                                                          # non-empty directory)
    return "rename"


def publish(dest_root, run):
    """Move the admitted staging directory to the published path, never replacing or nesting."""
    root, incoming, final = _paths(dest_root, run)
    if os.path.islink(incoming) or not os.path.isdir(incoming):
        raise reg.GateError(f"{incoming} is not a real staging directory: refused")
    if os.path.lexists(final):
        raise reg.GateError(f"{final} exists: refused, never replaced or nested; {incoming} kept")
    for p in (incoming, final):
        reg.check_destination(p, root, "the NAS publication")
    missing = [f for f in (reg.TRAINING_RECEIPT, reg.COMMIT_MARKER) if not os.path.isfile(os.path.join(incoming, f))]
    if missing:
        raise reg.GateError(f"{incoming} is not a sealed run (missing {missing}): refused")
    try:
        how = _rename_noreplace(incoming, final)
    except FileExistsError:
        raise reg.GateError(f"{final} exists (created meanwhile): refused, never replaced or nested; {incoming} kept")
    reg._fsync_dir(root)
    return final, how


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cmd", choices=("stage", "publish"))
    p.add_argument("--dest-root", required=True)
    p.add_argument("--run", required=True)
    a = p.parse_args(argv)
    try:
        out = stage(a.dest_root, a.run) if a.cmd == "stage" else publish(a.dest_root, a.run)[0]
    except (reg.GateError, OSError, ValueError) as err:
        print(f"EXP46 NAS PUBLICATION REFUSED: {err}", file=sys.stderr)
        return 2
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
