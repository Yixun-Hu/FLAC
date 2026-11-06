"""exp_22 A7 (plan §9 A7 r2 (vii), r3 (a) (i)): ONE real run of the pinned no-flash launcher on THIS checkout (CPU; the
real src/models/transformer.py, so torch imports): its log proves the back-to-back pair -> the dynamic sentinel ->
the target's own output, in that order, under the backend rule; and every registered localize_meshgrid.py argument
vector reaches the REAL localize_meshgrid.parse_args through runpy exactly as a direct run sees it, and parses to the
governing line's namespace except --out-dir (the argv / launch-record identity: strip_launch_flags is unchanged)."""
import json
import os
import subprocess
import sys

from conftest import C, load

E = load("exp22_r2c_evidence")
S = load("exp22_r2c_supervisor")
REPO = os.path.abspath(os.path.join(os.path.dirname(E.__file__), "..", "..", "..", ".."))
MARK = "A7 REAL TARGET OUTPUT"
TARGET = r'''
import json, os, sys
print("A7 REAL TARGET OUTPUT", flush=True)
import localize_meshgrid as L
from src.localization import meshgrid_offgrid_probe as P
import src.models.transformer as T
out = {"name": __name__, "argv": sys.argv, "cwd": os.getcwd(), "flash_none": T.flash_attn_func is None,
       "sets": []}
for args, governing in json.load(open(os.environ["A7_ARGSETS"])):
    a, g = vars(L.parse_args(args)), vars(L.parse_args(governing))
    out["sets"].append({"stripped": P.strip_launch_flags(args), "differs": sorted(k for k in a if a[k] != g[k]),
                        "parsed": {k: sorted(v) if isinstance(v, (set, frozenset)) else str(v) for k, v in a.items()}})
with open(os.environ["A7_REPORT"], "w") as f:
    json.dump(out, f)
'''


def test_the_pinned_launcher_on_this_checkout_pair_sentinel_target_and_argv_identity(tmp_path):
    lines = E.expand_lines(json.load(open(S.LINES_JSON)), run="20261106T130501", index=0, worktree=REPO, py=C.PY,
                           pytest_main=C.PYTEST_MAIN)
    sets = []
    for ln in lines:
        if ln["target"] == "localize_meshgrid.py":
            args, i = ln["argv"][3:], ln["argv"].index("--out-dir") - 3
            gov = args[:i + 1] + ["outputs_loc/exp22/r2_probe_BF" if ln["kind"] == "probe" else "outputs_loc/exp22/r2_gates"]
            sets.append((args, gov + args[i + 2:]))
    assert len(sets) == 7
    target, argsets = tmp_path / "a7_target.py", tmp_path / "argsets.json"
    target.write_text(TARGET)
    argsets.write_text(json.dumps(sets))
    gv = lines[1]
    env = dict(gv["env"], CUDA_VISIBLE_DEVICES="", A7_ARGSETS=str(argsets))     # the wrapped env, CPU only
    reports, logs = {}, {}
    for how in ("launcher", "direct"):
        report, log = tmp_path / f"{how}.json", tmp_path / f"{how}.log"
        argv = ([C.PY, E.LAUNCHER, str(target)] if how == "launcher" else [C.PY, str(target)]) + sets[0][0]
        e = dict(env, A7_REPORT=str(report)) if how == "launcher" else dict(
            {k: v for k, v in env.items() if k != "PYTHONPATH"}, A7_REPORT=str(report), PYTHONPATH=REPO)
        with open(log, "wb") as f:
            p = subprocess.run(argv, cwd=REPO, env=e, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                               timeout=900)
        logs[how] = log.read_text()
        assert p.returncode == 0, logs[how][-3000:]
        reports[how] = json.loads(report.read_text())
    text, lines_ = logs["launcher"], logs["launcher"].splitlines()
    assert E.backend_log_reasons(text, str(target)) == [], E.backend_log_reasons(text, str(target))
    ov, fb = lines_.index(E.OVERLAY_LINE), lines_.index(E.FALLBACK_LINE)
    sent, mark = lines_.index(E.sentinel_line(str(target))), lines_.index(MARK)
    assert fb == ov + 1 < sent < mark                                     # pair -> dynamic sentinel -> target output
    assert E.backend_log_reasons(text, "localize_meshgrid.py")              # the sentinel names THIS target only
    via, direct = reports["launcher"], reports["direct"]
    assert via["name"] == "__main__" and via["cwd"] == REPO and via["flash_none"] is True
    assert via["argv"] == [str(target)] + sets[0][0] and via["argv"][1:] == direct["argv"][1:]
    assert via["sets"] == direct["sets"]                                    # runpy changes nothing the driver sees
    for (args, gov), got in zip(sets, via["sets"]):
        assert got["stripped"] == args and got["differs"] == ["out_dir"]   # identical arguments but the out-dir
        assert got["parsed"]["out_dir"].startswith("outputs_loc/exp22/r2_")
    assert sorted(os.listdir(os.path.join(REPO, "overlays", "no_flash"))) == ["exp35_no_flash_launch.py", "flash_attn.py"]
