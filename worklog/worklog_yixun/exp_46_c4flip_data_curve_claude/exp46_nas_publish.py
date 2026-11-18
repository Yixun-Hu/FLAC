#!/usr/bin/env python3
"""exp_46 -- the staged, no-clobber NAS publication of a COMPLETE neuronic run (L7; Codex code r2 finding 8).

``stage`` creates ``<dest root>/<run>.incoming`` EXCLUSIVELY before anything is copied (os.mkdir: an existing entry --
a directory, a file or a symlink, dangling or not -- refuses, and so does an existing ``<dest root>/<run>``); the
dest root itself must be a real, isolated directory (never created here: a missing mount refuses). rsync then fills
the staging directory and exp_46's own admission (sha256sum -c, ``exp46_registry.py training-check``) runs on it.
``publish`` re-validates the staging directory (a real directory below the dest root, never a link, holding the
sealed run's TRAINING_COMPLETE.json and TRAINING_COMMITTED.json) and moves it to ``<dest root>/<run>`` through a
NO-REPLACE rename (renameat2 RENAME_NOREPLACE): a final path created meanwhile refuses -- it is never replaced and
the staging directory is never nested inside it (as ``mv`` would). There is NO fallback (Codex code r3 finding 4): a
box or filesystem without RENAME_NOREPLACE refuses (a lexists check followed by os.rename would let a competitor's
empty directory, created in between, be replaced). Where it runs: L7 publishes from the A6000 box into its NAS mount
(/media/diskstation/..., registered as cifs in STORAGE_FS["a6000"]); the Linux CIFS client implements
RENAME_NOREPLACE (cifs_rename2: no-replace is its natural behaviour), so refusing costs no publication there; a libc
without the renameat2 wrapper (glibc < 2.28) gets the raw syscall.

    python -B exp46_nas_publish.py stage   --dest-root <dir> --run <run id>
    python -B exp46_nas_publish.py publish --dest-root <dir> --run <run id>
    python -B exp46_nas_publish.py provision     # the two exp_46 NAS roots, once (Codex code r3 finding 7)

Exit 0 = done (the path is printed), 2 = refused. Written by the exp_46 Coder (Claude Opus 5.5).
"""
import argparse
import ctypes
import errno
import json
import os
import platform
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import exp46_registry as reg  # noqa: E402

RENAME_NOREPLACE = 1             # <linux/fs.h>
AT_FDCWD = -100
SYS_RENAMEAT2 = {"x86_64": 316, "aarch64": 276}       # the raw syscall number, for a libc without the wrapper


def _paths(dest_root, run):
    reg.parse_run_id(run)
    root = reg.resolve_dotdot(dest_root)       # as the kernel will address it (Codex code r3 finding 1)
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


class NoReplaceUnsupported(OSError):
    """renameat2(RENAME_NOREPLACE) is not available on this box or filesystem."""


def _renameat2(src, dst):
    """renameat2(AT_FDCWD, src, AT_FDCWD, dst, RENAME_NOREPLACE): libc's wrapper (glibc >= 2.28), else the raw syscall
    (an older libc). 0 on success, else the errno."""
    libc = ctypes.CDLL(None, use_errno=True)
    s, d = os.fsencode(src), os.fsencode(dst)
    fn = getattr(libc, "renameat2", None)
    if fn is not None:
        rc = fn(AT_FDCWD, s, AT_FDCWD, d, RENAME_NOREPLACE)
    else:
        nr, sc = SYS_RENAMEAT2.get(platform.machine()), getattr(libc, "syscall", None)
        if nr is None or sc is None:
            return errno.ENOSYS
        rc = sc(ctypes.c_long(nr), ctypes.c_long(AT_FDCWD), ctypes.c_char_p(s), ctypes.c_long(AT_FDCWD),
                ctypes.c_char_p(d), ctypes.c_long(RENAME_NOREPLACE))
    return 0 if rc == 0 else ctypes.get_errno()


def _rename_noreplace(src, dst):
    """The no-replace move, or a refusal -- never a plain rename (Codex code r3 finding 4: a lexists check followed by
    os.rename lets a competitor's empty directory, created in between, be replaced). EEXIST -> FileExistsError;
    EINVAL / ENOSYS / EOPNOTSUPP (no RENAME_NOREPLACE here) -> NoReplaceUnsupported."""
    err = _renameat2(src, dst)
    if err == 0:
        return "renameat2"
    if err == errno.EEXIST:
        raise FileExistsError(errno.EEXIST, "the published path exists", dst)
    if err in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
        raise NoReplaceUnsupported(err, f"renameat2 RENAME_NOREPLACE unsupported here ({os.strerror(err)})", dst)
    raise OSError(err, os.strerror(err), src)


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
    except NoReplaceUnsupported as err:
        raise reg.GateError(f"{err.strerror}: refused -- no fallback rename (it could replace a competitor's directory); "
                            f"{incoming} kept; tell Yixun")
    reg._fsync_dir(root)
    return final, how


def nas_roots():
    """The two NEW exp_46 namespaces on the NAS: the checkpoint root (the A6000 save root -- L7's dest root and the
    evaluator's run dirs) and the scored-waveform dump root. Read from their single sources (registry, evaluator)."""
    import exp46_eval as ev
    return (("the exp_46 checkpoint root", reg.PLATFORMS["a6000"]["save_root"]), ("the exp_46 dump root", ev.DUMP_ROOT))


def provision():
    """Create both exp_46 NAS namespaces once, before their first use (Codex code r3 finding 7). Each PARENT must
    already be a real directory on the registered A6000 NAS filesystem (STORAGE_FS["a6000"]): a missing mount refuses --
    nothing is ever created on a local disk. Each exp_46 directory is created through the isolation guard below its
    parent (mkdir_below: never inside or aliasing a protected root, never through a link); an existing real directory
    is accepted (idempotent), anything else at that path refuses. Both are checked before either is created."""
    plan = []
    for what, path in nas_roots():
        parent = os.path.dirname(reg.resolve_dotdot(path))
        if os.path.islink(parent) or not os.path.isdir(parent):
            raise reg.GateError(f"{parent} (the parent of {what}) is missing or a link: the NAS is not mounted -- refused")
        fstype, mnt = reg.mount_fstype(parent)
        if fstype not in reg.STORAGE_FS["a6000"]:
            raise reg.GateError(f"{parent} is on {fstype} ({mnt}), not the registered NAS filesystem "
                                f"{reg.STORAGE_FS['a6000']}: refused")
        reg.check_destination(path, parent, what)
        if os.path.lexists(path) and (os.path.islink(path) or not os.path.isdir(path)):
            raise reg.GateError(f"{path} exists but is not a real directory: refused")
        plan.append((what, path, parent))
    out = {}
    for what, path, parent in plan:
        created = not os.path.lexists(path)
        if created:
            reg.mkdir_below(path, parent, what)
        out[what] = {"path": path, "created": created}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cmd", choices=("stage", "publish", "provision"))
    p.add_argument("--dest-root")
    p.add_argument("--run")
    a = p.parse_args(argv)
    if a.cmd != "provision" and not (a.dest_root and a.run):
        p.error(f"{a.cmd} needs --dest-root and --run")
    try:
        if a.cmd == "provision":
            out = json.dumps(provision(), indent=1, sort_keys=True)
        else:
            out = stage(a.dest_root, a.run) if a.cmd == "stage" else publish(a.dest_root, a.run)[0]
    except (reg.GateError, OSError, ValueError) as err:
        print(f"EXP46 NAS PUBLICATION REFUSED: {err}", file=sys.stderr)
        return 2
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
