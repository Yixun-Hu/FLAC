#!/usr/bin/env python3
"""exp_35 commit-size bookkeeping (the "< 200 changed CODE lines per commit" rule of CLAUDE.md /
the Coder brief): for one commit, the added + removed lines that are code -- not blank, not a
comment-only line, not inside a module / class / function docstring -- split into code (outside
src/tests/), tests (src/tests/) and data (*.json, *.md, *.log, *.sha256, *.csv). Read-only (git show).

Usage: exp35_count_code_lines.py <commit> [<commit> ...]   (one table row per commit)

Written by the exp_35 Coder (Claude Opus 5.5, max effort).
"""
import ast
import re
import subprocess
import sys

DATA = (".json", ".md", ".log", ".sha256", ".csv", ".txt")


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout


def code_lines(path, text):
    """1-based numbers of the code lines of ``text`` (the file ``path`` at one revision)."""
    lines = text.splitlines()
    skip = set()
    if path.endswith(".py"):
        try:
            tree = ast.parse(text)
        except SyntaxError:
            tree = None
        for node in ast.walk(tree) if tree else ():
            body = getattr(node, "body", None)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and body \
                    and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                skip.update(range(body[0].lineno, body[0].end_lineno + 1))
    return {i for i, ln in enumerate(lines, 1) if ln.strip() and not ln.strip().startswith("#") and i not in skip}


def changed(commit, path):
    """``(added code lines, removed code lines)`` of ``path`` in ``commit`` (vs its first parent)."""
    old = _git("show", f"{commit}^:{path}") if _git("cat-file", "-t", f"{commit}^:{path}").strip() else ""
    new = _git("show", f"{commit}:{path}") if _git("cat-file", "-t", f"{commit}:{path}").strip() else ""
    old_code, new_code = code_lines(path, old), code_lines(path, new)
    added = removed = 0
    for hunk in re.finditer(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", _git(
            "diff", "-U0", f"{commit}^", commit, "--", path), re.M):
        o, oc, n, nc = int(hunk[1]), int(hunk[2] or 1), int(hunk[3]), int(hunk[4] or 1)
        removed += sum(1 for i in range(o, o + oc) if i in old_code)
        added += sum(1 for i in range(n, n + nc) if i in new_code)
    return added, removed


def main(argv):
    print("| commit | code +/- | tests +/- | data +/- | files |")
    print("|---|---|---|---|---|")
    for commit in argv:
        tot = {"code": [0, 0], "tests": [0, 0], "data": [0, 0]}
        files = [f for f in _git("diff", "--name-only", f"{commit}^", commit).splitlines() if f]
        for f in files:
            kind = "data" if f.endswith(DATA) else "tests" if f.startswith("src/tests/") else "code"
            a, r = changed(commit, f)
            tot[kind][0] += a
            tot[kind][1] += r
        cells = " | ".join(f"+{v[0]} / -{v[1]} (= {v[0] + v[1]})" for v in tot.values())
        print(f"| `{_git('rev-parse', '--short=8', commit).strip()}` | {cells} | {len(files)} |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
