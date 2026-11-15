"""exp_45 (forked from test_exp38_command_sheets.py) — the two command sheets stay well-formed and carry the current
launch / evaluation / publication commands (plan_rf_data_curve.md r3 §3.2 item: command sheets analogous to
exp_38's and covered by this test)."""
import importlib.util
import os
import re

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(REPO, "worklog", "worklog_yixun", "exp_45_rf_data_curve_claude")
_spec = importlib.util.spec_from_file_location("exp45_registry", os.path.join(EXP, "exp45_registry.py"))
reg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reg)
SHEETS = {
    "rf_data_curve_command.md": [   # the A6000 box: E0, the bridge, the evaluation cells -- no training
        "git -C ~/codespace/FLAC worktree add --detach ~/codespace/exp-45-campaign $X",
        "python $E/exp45_eval.py env-receipt --write", "python $E/exp45_eval.py prepare-e0",
        "python $E/exp45_eval.py run --gpu 1 --cells de_E0a_K8_s42 de_E0b_K8_s42 de_E0c_K8_s42 de_E0d_K8_s42",
        "python $E/exp45_eval.py e0 && python $E/exp45_eval.py e0-check",
        "git add -f $E/rf_data_curve_e0_receipt.json{,.sha256} outputs_FLAC/exp45_rf_data_curve/eval/E0{a,b,c,d}",
        "python $E/exp45_eval.py prepare-bridge", "python $E/exp45_eval.py run --gpu 1 --cells exp45_RF100bridge_K8_s42",
        "python $E/exp45_eval.py bridge --bundle", "python $E/exp45_eval.py prepare $RUN", "python $E/exp45_collect.py",
        # the exp_45 results commit: copied + verified into the publication worktree (Codex code r1 finding 12)
        "python $E/exp45_collect.py verify", "git -C ~/codespace/exp-45-publish pull --ff-only",
        "python $E/exp45_collect.py materialise --dest ~/codespace/exp-45-publish",
        'git commit -m "exp_45 results publication: 30 RF cells + the RF-100 bridge"'],
    "rf_data_curve_neuronic_commands.md": [
        "git worktree add --detach /n/fs/gatrdp/codespace/exp-45-launch $Y",
        "$PY $E/exp45_registry.py e0-check --platform neuronic", "$PY $E/exp45_registry.py untracked --platform neuronic",
        "$PY $E/exp45_registry.py dry-run --platform neuronic",
        "$PY $E38/exp38_subset_audit.py --verify $E38/data_efficiency_subset_audit.json",
        "$PY $E38/exp38_content_manifest.py --verify $E38/data_efficiency_ar_content_manifest.txt.gz",
        "exp45_submit_neuronic.py submit --run $RUN --mode SMOKE", "exp45_submit_neuronic.py submit --run $RUN --mode FULL",
        "exp45_submit_neuronic.py reconcile", "exp45_submit_neuronic.py abandon",
        "training-check $RUN --run-dir $DST/$RUN.incoming --platform neuronic", "--closure", "--e0-sha",
        # L8 (Codex code r1 finding 12): fetch both branches, the build checkout at R45, sha256-verify there, copy +
        # verify into the generator root, fetch the exp_38 bundle descending from its pushed records commit e5a012fa
        "G=/n/fs/gatrdp/codespace/FLAC", "git -C $G fetch origin exp-45-rf-data-curve exp-38-data-efficiency",
        "git -C $G merge-base --is-ancestor $Y $R45",
        "git -C $G worktree add --detach $B $R45",
        'test "$(git rev-parse HEAD)" = "$R45"', "$PY $E/exp45_collect.py verify",
        "$PY $E/exp45_collect.py materialise --dest $G", "$PY $E/exp45_collect.py verify --root $G",
        "git -C $G fetch origin exp-38-data-efficiency",
        "git -C $G merge-base --is-ancestor e5a012fa62705627104bcd9892643edab07aa1e7 $B38",
        "$PY $E/exp45_table.py import-exp38 --commit $B38 --generator-root $G",
        "$PY $E/exp45_table.py build --exp38-commit $B38",
        "$PY worklog/worklog_yixun/gen_model_comparison.py --repo-root $G --out $B/worklog/worklog_yixun/model_comparison.md",
        # L1 (Codex code r2 finding 14): the two-parent comparison with all three junit inputs and checkout roots
        "$PY $E/exp45_regression_compare.py --merged $L1/merged.junit.xml --exp38 $L1/parent_3593f34c.junit.xml "
        "--exp43 $L1/parent_4a7e3b3a.junit.xml --merged-root $M --exp38-root $P38 --exp43-root $P43"],
}


@pytest.mark.parametrize("name", sorted(SHEETS))
def test_the_sheet_is_well_formed_and_current(name):
    text = open(os.path.join(EXP, name)).read()
    fences = [ln for ln in text.split("\n") if ln.startswith("```")]
    assert len(fences) % 2 == 0 and fences, "unbalanced code fences"
    blocks = re.findall(r"```bash\n(.*?)\n```", text, flags=re.S)
    assert blocks and all("```" not in b for b in blocks)
    joined = "\n".join(blocks)
    for cmd in SHEETS[name]:
        assert cmd in joined, cmd
    outside = re.sub(r"```bash\n.*?\n```", "", text, flags=re.S)
    assert not re.search(r"(^|\n)(MODE=|python \$E/|\$PY \$E/)", outside), "a command outside a code block"
    assert "--resume" not in text and "RESUME_STEP=" not in text
    assert "exp38_eval.py" not in joined and "exp38_submit_neuronic.py" not in joined   # the fork's own kit only


def test_l8_makes_the_published_evidence_available_and_verified_before_any_use():
    """Codex code r1 finding 12: on neuronic every consumer of the published evidence runs only after its root
    received and verified it -- fetch -> build checkout at R45 -> verify -> copy into G -> verify G -> regen #1; the exp_38
    bundle is fetched and checked to descend from exp_38's pushed records commit before it is imported; G is the plan
    N6 main checkout and its checkout is never moved (no checkout / pull / merge / reset in G)."""
    import subprocess
    b = _blocks("rf_data_curve_neuronic_commands.md")
    l8 = b[b.index("G=/n/fs/gatrdp/codespace/FLAC"):]
    order = ["git -C $G fetch origin exp-45-rf-data-curve exp-38-data-efficiency", "git -C $G merge-base --is-ancestor $Y $R45",
             "git -C $G worktree add --detach $B $R45", "$PY $E/exp45_collect.py verify  ",
             "$PY $E/exp45_collect.py materialise --dest $G", "$PY $E/exp45_collect.py verify --root $G",
             "$PY worklog/worklog_yixun/gen_model_comparison.py --repo-root $G", "git -C $G fetch origin exp-38-data-efficiency",
             "git -C $G merge-base --is-ancestor e5a012fa62705627104bcd9892643edab07aa1e7 $B38",
             "$PY $E/exp45_table.py import-exp38 --commit $B38 --generator-root $G", "$PY $E/exp45_table.py build --exp38-commit $B38"]
    pos = [l8.index(cmd) for cmd in order]
    assert pos == sorted(pos), [order[i] for i in range(len(pos)) if pos[i] != sorted(pos)[i]]
    assert l8.rindex("gen_model_comparison.py --repo-root $G") > l8.index("exp45_table.py build")      # regen #2 last
    for verb in ("checkout", "pull", "merge ", "reset", "switch", "stash"):
        assert f"git -C $G {verb}" not in l8, verb
    assert "/n/fs/gatrdp/codespace/FLAC" not in l8.replace("G=/n/fs/gatrdp/codespace/FLAC", "")      # only via $G
    # e5a012fa is exp_38's pushed D7-A records commit and descends from the merged tip 3593f34c (EXP38_BASE)
    git = lambda *a: subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True)       # noqa: E731
    assert git("cat-file", "-t", "e5a012fa62705627104bcd9892643edab07aa1e7").stdout.strip() == "commit"
    assert git("merge-base", "--is-ancestor", "3593f34cad4e6e4415a572e93d16ee83465f8ae8",
               "e5a012fa62705627104bcd9892643edab07aa1e7").returncode == 0
    a6 = _blocks("rf_data_curve_command.md")
    seq = ["git -C ~/codespace/exp-45-publish pull --ff-only",
           "python $E/exp45_collect.py materialise --dest ~/codespace/exp-45-publish",
           "cd ~/codespace/exp-45-publish\npython $E/exp45_collect.py verify     # the publication worktree",
           "git add -f $E/rf_data_curve_results.{json,md}", 'git commit -m "exp_45 results publication']
    assert [a6.index(c) for c in seq] == sorted(a6.index(c) for c in seq)


MAIN = "/n/fs/gatrdp/codespace/FLAC"            # the plan-N6 main checkout (the generator root G)
# the only command forms either sheet may run against MAIN: they read it, add .git worktree / fetch metadata, or
# write UNTRACKED files only (the materialisers refuse a tracked destination); everything else that names MAIN, or
# runs with MAIN as its cwd, is a write path into main's tracked tree and fails this test
MAIN_FORMS = (r"G=/n/fs/gatrdp/codespace/FLAC",
              r"SRC=/n/fs/gatrdp/codespace/FLAC/outputs_FLAC/exp45_rf_data_curve/\$RUN",   # read by the L7 rsync
              r"cd /n/fs/gatrdp/codespace/FLAC && git fetch origin exp-45-rf-data-curve",
              r"git worktree add --detach /n/fs/gatrdp/codespace/exp-45-launch \$Y",
              r'ln -s "\$\(readlink -f /n/fs/gatrdp/codespace/FLAC/[\w/.]+\)" [\w/.]+',
              r"git -C \$G fetch origin [\w. -]+", r"git -C \$G merge-base --is-ancestor \S+ \S+",
              r"git -C \$G worktree add --detach \$B \$R45",
              r"\$PY \$E/exp45_collect\.py materialise --dest \$G(\s+#.*)?",
              r"\$PY \$E/exp45_collect\.py verify --root \$G(\s+#.*)?",
              r"\$PY \$E/exp45_table\.py import-exp38 --commit \$B38 --generator-root \$G(\s+#.*)?",
              r"\$PY worklog/worklog_yixun/gen_model_comparison\.py --repo-root \$G --out \$B/worklog/worklog_yixun/"
              r"model_comparison\.md(\s+#.*)?")


def _commands(text):
    """Every command of a sheet's bash blocks (continuations joined, comments and blank lines dropped)."""
    out = []
    for block in re.findall(r"```bash\n(.*?)\n```", text, flags=re.S):
        buf = ""
        for ln in block.split("\n"):
            s = ln.strip()
            if not buf and (not s or s.startswith("#")):
                continue
            buf = (buf + " " + s) if buf else s
            if buf.endswith("\\"):
                buf = buf[:-1].rstrip()
                continue
            out.append(re.sub(r"\s+#\s.*$", "", buf))          # an inline comment is not part of the command
            buf = ""
    return out


@pytest.mark.parametrize("name", sorted(SHEETS))
def test_no_sheet_command_can_write_a_tracked_path_of_the_main_checkout(name):
    """Codex code r2 finding 1: neither regeneration writes main's tracked table (the generator reads its evidence
    from G and writes its table to B via --out), and every other command touching the plan-N6 main checkout is a
    registered read-only / metadata / untracked-only form -- none moves, edits, commits or copies into G."""
    cwd_main = False                                     # is MAIN the shell's working directory?
    for cmd in _commands(open(os.path.join(EXP, name)).read()):
        if cmd.startswith("cd "):
            cwd_main = cmd.split()[1] == MAIN
        if MAIN in cmd or "$G" in cmd or "${G}" in cmd or cwd_main:
            assert any(re.fullmatch(f, cmd) for f in MAIN_FORMS), f"{name}: {cmd}"
        if "gen_model_comparison.py" in cmd:
            assert "--out $B/" in cmd and "--out $G" not in cmd, cmd


def test_the_l1_comparison_command_is_complete():
    """Codex code r2 finding 14: the neuronic sheet runs the registered L1 -- the same full CPU suite on the merged tree
    and on both parents' worktrees (at exp_38's tip and exp_43's pin), then exp45_regression_compare.py with the three
    junit inputs and all three checkout roots."""
    b = _blocks("rf_data_curve_neuronic_commands.md")
    assert re.search(r"git -C \$M worktree add --detach \$P38 3593f34cad4e6e4415a572e93d16ee83465f8ae8", b)
    assert re.search(r"git -C \$M worktree add --detach \$P43 4a7e3b3a948388434bc102b8bd7e7aba94648d71", b)
    runs = re.findall(r"--junitxml=\$L1/(\S+)\.junit\.xml", b)
    assert sorted(runs) == ["merged", "parent_3593f34c", "parent_4a7e3b3a"]
    cmp = [c for c in _commands(open(os.path.join(EXP, "rf_data_curve_neuronic_commands.md")).read())
           if "exp45_regression_compare.py" in c]
    assert len(cmp) == 1
    args = cmp[0].split()
    for flag, val in (("--merged", "$L1/merged.junit.xml"), ("--exp38", "$L1/parent_3593f34c.junit.xml"),
                      ("--exp43", "$L1/parent_4a7e3b3a.junit.xml"), ("--merged-root", "$M"), ("--exp38-root", "$P38"),
                      ("--exp43-root", "$P43")):
        assert args[args.index(flag) + 1] == val, flag
    import importlib.util
    spec = importlib.util.spec_from_file_location("rc_sheet", os.path.join(EXP, "exp45_regression_compare.py"))
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    tail = args[args.index("$E/exp45_regression_compare.py") + 1:]
    assert rc.main(tail) == 2                            # parsed (the placeholder inputs are unreadable: exit 2)


def _blocks(name):
    return "\n".join(re.findall(r"```bash\n(.*?)\n```", open(os.path.join(EXP, name)).read(), flags=re.S))


def test_d1_the_a6000_sheet_trains_nothing():
    assert reg.TRAINING_PLATFORM == "neuronic"
    blocks = _blocks("rf_data_curve_command.md")
    assert "MODE=FULL" not in blocks and "MODE=SMOKE_DDP" not in blocks and "exp45_submit_neuronic" not in blocks


def test_d1_the_neuronic_sheet_submits_every_campaign_run_smoke_then_full():
    blocks = _blocks("rf_data_curve_neuronic_commands.md")
    runs = re.search(r'^RUNS="([^"]+)"', blocks, flags=re.M).group(1).split()
    assert sorted(runs) == sorted(reg.campaign_runs()) and len(runs) == 3
    loops = re.findall(r"^for RUN in ([^;\n]+); do\n(.*?)^done$", blocks, flags=re.M | re.S)
    full = [r for head, body in loops if "--mode FULL" in body for r in head.split()]
    smoke = [r for head, body in loops if "--mode SMOKE" in body for r in head.split()]
    assert sorted(full) == sorted(runs) and smoke == ["$RUNS"]
    assert blocks.index("--mode SMOKE") < blocks.index("--mode FULL")
