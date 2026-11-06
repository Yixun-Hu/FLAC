"""exp_35 (Planner, Claude Opus 5.5): cross-check watcher v5's literal frozen plan against (1) H35's chunk_plan(), extracted
with ast from `git show 5b065801:<supervisor>` and executed ALONE in an empty namespace (SEEDS taken from the source's own
literal, nothing else imported or run), and (2) a copy of the live supervisor state. Read-only; prints one verdict line.
  python3 -B exp35_plan_literal_check.py <watcher v5> <state json>"""
import ast, json, subprocess, sys

SUP = "worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_supervisor.py"
src = subprocess.run(["git", "-C", "/home/yixunhu/codespace/FLAC", "show", f"5b065801:{SUP}"], capture_output=True, text=True, check=True).stdout
tree = ast.parse(src)
fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "chunk_plan"]
seeds = [n for n in tree.body if isinstance(n, ast.Assign) and [getattr(t, "id", None) for t in n.targets] == ["SEEDS"]]
assert len(fn) == 1 and len(seeds) == 1, "chunk_plan / SEEDS not found exactly once"
ns = {"SEEDS": ast.literal_eval(seeds[0].value)}
exec(compile(ast.Module(body=fn, type_ignores=[]), "chunk_plan@5b065801", "exec"), ns)
h35 = [[c["n"], c["kind"], c["cell"], c["gate"]] for c in ns["chunk_plan"]()]
lines = [ln for ln in open(sys.argv[1], encoding="utf-8") if ln.startswith("PLAN='")]
assert len(lines) == 1, "the watcher has no single PLAN='...' line"
literal = json.loads(lines[0].strip()[len("PLAN='"):-1])
state = [[c["n"], c["kind"], c["cell"], c["gate"]] for c in json.load(open(sys.argv[2]))["chunks"]]
print(f"SEEDS {ns['SEEDS']}; chunk_plan() rows {len(h35)}; literal rows {len(literal)}; state rows {len(state)}")
print(f"literal == H35 chunk_plan(): {literal == h35}; state == H35 chunk_plan(): {state == h35}")
sys.exit(0 if literal == h35 == state else 1)
