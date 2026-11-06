"""exp_35 (Planner, Claude Opus 5.5): READ-ONLY per-cell wall-time table for the phase-(ii) ETA. Sources: (1) the exp_25
campaign runner logs (each "RUN <cell> on GPU <g> (log <path>)" line; the cell log's birth -> mtime = the cell's wall time;
the "shared with pid ..." line just above a RUN names a co-tenant); (2) a phase-(i) supervisor state snapshot (per chunk
started -> ended; every phase-(i) cell shared GPU 1 with Yixun's s43 training). Prints one row per (campaign, arm, K, dumps,
co-tenant) with n, median and max minutes. Writes nothing.
  python3 -B exp35_cell_times.py <exp_25 folder> <state snapshot> <campaign log> [<campaign log> ...]"""
import datetime, json, os, re, statistics, sys

E, SNAP, LOGS = sys.argv[1], sys.argv[2], sys.argv[3:]
rows = []
for L in LOGS:
    lines = open(L, encoding="utf-8", errors="replace").read().splitlines()
    for i, ln in enumerate(lines):
        m = re.match(r"\s+RUN (\S+) on GPU (\d) \(log (\S+)\)$", ln)
        if not m:
            continue
        cell, gpu, rel = m.groups()
        shared = any("shared with pid" in lines[j] and f"GPU {gpu}:" in lines[j] for j in range(max(0, i - 4), i))
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(E))), rel) if not rel.startswith("/") else rel
        try:
            st = os.stat(path)
        except OSError:
            continue
        birth = getattr(st, "st_birthtime", None)
        if birth is None:
            import subprocess
            birth = float(subprocess.run(["stat", "-c", "%W", path], capture_output=True, text=True).stdout.strip() or 0)
        if not birth:
            continue
        mins = (st.st_mtime - birth) / 60
        arm = cell.split("_")[1]
        k = re.search(r"_K(\d)_", cell + "_").group(1)
        block = "TRAJ" if "TRAJ" in cell else ("dup" if cell.endswith("_dup") else "eval")
        rows.append((os.path.basename(L)[12:31], arm, "K" + k, block, "shared" if shared else "alone", mins))
snap = json.load(open(SNAP))
for c in snap["chunks"]:
    if c.get("kind") == "cell" and c.get("started") and c.get("ended"):
        t0 = datetime.datetime.fromisoformat(c["started"]); t1 = datetime.datetime.fromisoformat(c["ended"])
        arm = c["cell"].split("_")[1]; k = re.search(r"_K(\d)_", c["cell"] + "_").group(1)
        rows.append(("exp35 phase (i)", arm, "K" + k, "eval", "shared (his s43)" if c.get("shared") else "alone", (t1 - t0).total_seconds() / 60))
groups = {}
for r in rows:
    groups.setdefault(r[:5], []).append(r[5])
print(f"{'campaign':22s} {'arm':6s} {'K':3s} {'block':5s} {'co-tenant':17s} {'n':>3s} {'median':>7s} {'max':>7s}")
for key in sorted(groups):
    v = groups[key]
    print(f"{key[0]:22s} {key[1]:6s} {key[2]:3s} {key[3]:5s} {key[4]:17s} {len(v):3d} {statistics.median(v):7.1f} {max(v):7.1f}")
