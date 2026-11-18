"""exp_46 — the two command sheets (fix round 1, Codex code r1 finding 10 / open item 7): the neuronic sheet
(c4flip_data_curve_neuronic_commands.md: L2 / L1 / L3 / L4, the four SMOKEs and FULLs with f100's extra hour, C3 and
recovery, the staged NAS publication through exp_46's own admission, the verified materialisation / import and both
regenerations writing outside main's tracked tree) and the A6000 sheet (c4flip_data_curve_command.md: E0, the Y46
records commit, the 40 evaluations and the results commit -- evaluation and publication ONLY). They stay well-formed,
name exp_46's own kit and registered paths, and keep the campaign order that is pending Yixun's D2.

Modelled on exp_45's test_exp45_command_sheets.py (exp_45 history, not copied: it tests exp_45's own sheets).

Written by the exp_46 Coder (Claude Opus 5.5).
"""
import importlib.util
import os
import re
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(REPO, "worklog", "worklog_yixun", "exp_46_c4flip_data_curve_claude")
sys.path.insert(0, EXP)
import exp46_registry as reg  # noqa: E402

NEURONIC, A6000 = "c4flip_data_curve_neuronic_commands.md", "c4flip_data_curve_command.md"
SHEETS = {
    A6000: [   # the A6000 box: E0, Y46, the 40 evaluation cells, R46 -- no training
        "git -C ~/codespace/FLAC worktree add --detach ~/codespace/exp-46-campaign $X46",
        "python $E/exp46_eval.py env-receipt --write", "python $E/exp46_eval.py prepare-e0",
        "python $E/exp46_eval.py run --gpu 1 --cells de_E0a_K8_s42 de_E0b_K8_s42 de_E0c_K8_s42 de_E0d_K8_s42",
        "python $E/exp46_eval.py e0 && python $E/exp46_eval.py e0-check",
        "git add -f $E/c4flip_data_curve_e0_receipt.json{,.sha256} outputs_FLAC/exp46_c4flip_data_curve/eval/E0{a,b,c,d}",
        'git commit -m "exp_46 records: E0 PASS at $X46"', "python $E/exp46_eval.py prepare $RUN",
        "python $E/exp46_collect.py", "python $E/exp46_collect.py verify",
        "git -C ~/codespace/exp-46-publish pull --ff-only",
        "python $E/exp46_collect.py materialise --dest ~/codespace/exp-46-publish",
        'git commit -m "exp_46 results publication: 40 C4F cells"'],
    NEURONIC: [
        "$PY $E/exp46_content_manifest.py --build $E/exp46_f100_delta_manifest.txt.gz",
        'git commit -m "exp_46 L2: pin the f100 delta manifest <sha256(text)>"',
        "$PY -B $E38/exp38_subset_audit.py --verify $E38/data_efficiency_subset_audit.json",
        "$PY -B $E38/exp38_content_manifest.py --verify $E38/data_efficiency_ar_content_manifest.txt.gz",
        "$PY $E/exp46_content_manifest.py --verify $E/exp46_f100_delta_manifest.txt.gz",
        "$PY $E/exp46_registry.py check exp46_C4Ff100_l40_2x32 --platform neuronic",
        "$PY $E/exp46_init_digest.py --run $RUN --json",
        "git worktree add --detach /n/fs/gatrdp/codespace/exp-46-launch $Y46",
        "$PY $E/exp46_registry.py e0-check --platform neuronic", "$PY $E/exp46_registry.py untracked --platform neuronic",
        "$PY $E/exp46_registry.py dry-run --platform neuronic",
        "exp46_submit_neuronic.py submit --run $RUN --mode SMOKE", "exp46_submit_neuronic.py submit --run $RUN --mode FULL",
        "exp46_submit_neuronic.py reconcile", "exp46_submit_neuronic.py abandon",
        "training-check $RUN --run-dir $DST/$RUN.incoming --platform neuronic", "--closure", "--e0-sha",
        "G=/n/fs/gatrdp/codespace/FLAC", "git -C $G fetch origin exp-46-c4flip-data-curve exp-38-data-efficiency",
        "git -C $G merge-base --is-ancestor $Y46 $R46", "git -C $G worktree add --detach $B $R46",
        'test "$(git rev-parse HEAD)" = "$R46"', "$PY $E/exp46_collect.py verify",
        "$PY $E/exp46_collect.py materialise --dest $G", "$PY $E/exp46_collect.py verify --root $G",
        "git -C $G merge-base --is-ancestor e5a012fa62705627104bcd9892643edab07aa1e7 $B38",
        "$PY $E/exp46_table.py import-exp38 --commit $B38 --generator-root $G",
        "$PY $E/exp46_table.py build --exp38-commit $B38",
        "$PY worklog/worklog_yixun/gen_model_comparison.py --repo-root $G --out $B/worklog/worklog_yixun/model_comparison.md",
        "$PY $E/exp46_regression_compare.py --merged $L1/merged.junit.xml --exp38 $L1/parent_3593f34c.junit.xml "
        "--exp43 $L1/parent_4a7e3b3a.junit.xml --merged-root $M --exp38-root $P38 --exp43-root $P43"],
}


def _text(name):
    return open(os.path.join(EXP, name)).read()


def _blocks(name):
    return "\n".join(re.findall(r"```bash\n(.*?)\n```", _text(name), flags=re.S))


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
def test_the_sheet_is_well_formed_current_and_names_only_exp46s_kit(name):
    text = _text(name)
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
    assert not re.search(r"exp45_\w+\.py", joined), "exp_45's kit"
    assert "exp38_eval.py" not in joined and "exp38_submit_neuronic.py" not in joined and "exp38_supervise.py" not in joined
    for script in sorted(set(re.findall(r"\$E/(exp46_\w+\.py)", joined))):          # every exp_46 tool it calls exists
        assert os.path.isfile(os.path.join(EXP, script)), script


def test_the_campaign_order_is_the_recommended_one_and_marked_pending_yixun_d2():
    """The coordinator's recommendation (Codex code r1 open item 10), still Yixun's decision D2: final code round -> L2
    delta build -> pin commit -> definitive L1 -> L2 union verification before any f100 SMOKE -> L3 -> L4 freeze X46."""
    text, b = _text(NEURONIC), _blocks(NEURONIC)
    assert "pending Yixun D2" in text and b.count("pending Yixun D2") >= 2 and "Yixun's D1" in text
    order = ["$PY $E/exp46_content_manifest.py --build $E/exp46_f100_delta_manifest.txt.gz",
             'git commit -m "exp_46 L2: pin the f100 delta manifest',
             "--junitxml=$L1/merged.junit.xml", "$PY $E/exp46_regression_compare.py --merged",
             "$PY $E/exp46_content_manifest.py --verify $E/exp46_f100_delta_manifest.txt.gz",
             "$PY $E/exp46_init_digest.py --run $RUN", "# L4: freeze X46", "Y46=<", "--mode SMOKE", "--mode FULL"]
    pos = [b.index(c) for c in order]
    assert pos == sorted(pos), [order[i] for i in range(len(pos)) if pos[i] != sorted(pos)[i]]
    union = [c for c in _commands(text) if "exp46_content_manifest.py --verify" in c]
    assert len(union) == 1 and "exp38_subset_audit.py --verify" in union[0] and "exp38_content_manifest.py --verify" in union[0]
    assert union[0].startswith("sbatch ") and "--output=$M/$E/c4flip_data_curve_l2c_union_verify_" in union[0]
    build = [c for c in _commands(text) if "exp46_content_manifest.py --build" in c]
    assert len(build) == 1 and build[0].startswith("sbatch ") and "--cpus-per-task=2" in build[0]
    assert reg.F100_DELTA_MANIFEST == f"{reg.EXP_DIR}/exp46_f100_delta_manifest.txt.gz"


def test_the_l1_comparison_runs_every_node_with_no_deselection_and_is_complete():
    """L1 (Codex code r1 open item 1): the SAME full CPU suite (src/tests, every node -- no --deselect, no -k / -m
    selection: L0's deselections are NOT carried forward) on the exp_46 tip and on both parents' worktrees at exp_38's
    tip and exp_43's pin, then exp46_regression_compare.py with the three junit inputs and all three checkout roots."""
    b = _blocks(NEURONIC)
    assert re.search(r"git -C \$M worktree add --detach \$P38 3593f34cad4e6e4415a572e93d16ee83465f8ae8", b)
    assert re.search(r"git -C \$M worktree add --detach \$P43 4a7e3b3a948388434bc102b8bd7e7aba94648d71", b)
    suite = re.search(r'^SUITE="([^"]+)"', b, flags=re.M).group(1)
    assert suite.endswith("-m pytest -q -p no:cacheprovider src/tests") and " -I " in suite
    for flag in ("--deselect", " -k ", " -m not", "--ignore", "-x"):
        assert flag not in suite, flag
    runs = re.findall(r"--junitxml=\$L1/(\S+)\.junit\.xml", b)
    assert sorted(runs) == ["merged", "parent_3593f34c", "parent_4a7e3b3a"]
    assert all(f'"cd {root} && $SUITE --junitxml=' in b for root in ("$M", "$P38", "$P43"))
    cmp = [c for c in _commands(_text(NEURONIC)) if "exp46_regression_compare.py" in c]
    assert len(cmp) == 1
    args = cmp[0].split()
    for flag, val in (("--merged", "$L1/merged.junit.xml"), ("--exp38", "$L1/parent_3593f34c.junit.xml"),
                      ("--exp43", "$L1/parent_4a7e3b3a.junit.xml"), ("--merged-root", "$M"), ("--exp38-root", "$P38"),
                      ("--exp43-root", "$P43"), ("--out", "$L1/regression_compare.json")):
        assert args[args.index(flag) + 1] == val, flag
    spec = importlib.util.spec_from_file_location("rc_sheet_exp46", os.path.join(EXP, "exp46_regression_compare.py"))
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    tail = args[args.index("$E/exp46_regression_compare.py") + 1:]
    assert rc.main(tail) == 2                            # parsed (the placeholder inputs are unreadable: exit 2)


def test_the_f100_full_gets_one_more_hour_and_every_full_is_one_monolithic_job():
    b = _blocks(NEURONIC)
    loops = re.findall(r"^for RUN in ([^;\n]+); do\n(.*?)^done$", b, flags=re.M | re.S)
    full = {r: body for head, body in loops if "--mode FULL" in body for r in head.split()}
    assert sorted(full) == sorted(reg.campaign_runs())
    for run, body in full.items():
        times = set(re.findall(r"--time (<[^>]+>|\S+)", body))
        assert times == ({"<computed + 1 h>"} if run == "exp46_C4Ff100_l40_2x32" else {"<computed>"}), (run, times)
    assert "f100 gets ONE MORE HOUR (+ 1 h)" in b and "never clamp, never chain" in b and "7-00:00:00" in b


def test_the_neuronic_sheet_submits_every_campaign_run_smoke_then_full():
    b = _blocks(NEURONIC)
    runs = re.search(r'^RUNS="([^"]+)"', b, flags=re.M).group(1).split()
    assert sorted(runs) == sorted(reg.campaign_runs()) and len(runs) == 4
    loops = re.findall(r"^for RUN in ([^;\n]+); do\n(.*?)^done$", b, flags=re.M | re.S)
    smoke = [r for head, body in loops if "--mode SMOKE" in body for r in head.split()]
    assert smoke == ["$RUNS"] and b.index("--mode SMOKE") < b.index("--mode FULL")
    assert "offset_step=128" in b and "4,550" in b                    # the C4F banner and f100's loader length


def test_the_a6000_sheet_evaluates_and_publishes_only():
    assert reg.TRAINING_PLATFORM == "neuronic"
    b = _blocks(A6000)
    for forbidden in ("exp46_submit_neuronic", "exp46_supervise", "exp46_train.sbatch", "--mode SMOKE", "--mode FULL",
                      "MODE=", "sbatch", "train.py", "torchrun"):
        assert forbidden not in b, forbidden
    cells = set(re.findall(r"\$\{RUN\}_K([18])_s(4[2-6])", b))
    assert cells == {(k, s) for k in "18" for s in ("42", "43", "44", "45", "46")}          # 10 per run, 4 runs: 40
    assert "40 in all" in b and "EVALUATION AND PUBLICATION ONLY" in _text(A6000)


def test_the_y46_commit_is_exactly_the_registered_e0_publication():
    """The A6000 sheet's Y46 commit is the provenance test's E0 rule: its exact subject (a full 40-hex X46) and exactly
    the E0 receipt + sidecar inside the exp_46 folder and the four E0 gates' farm folders."""
    b = _blocks(A6000)
    subj = re.search(r'git commit -m "(exp_46 records: E0 PASS at \$X46)"', b).group(1)
    assert re.fullmatch(r"exp_46 records: E0 PASS at [0-9a-f]{40}", subj.replace("$X46", "0123456789abcdef" * 2 + "01234567"))
    adds = [c for c in _commands(_text(A6000)) if c.startswith("git add -f")]
    assert adds[0] == ("git add -f $E/c4flip_data_curve_e0_receipt.json{,.sha256} "
                       "outputs_FLAC/exp46_c4flip_data_curve/eval/E0{a,b,c,d}")
    assert reg.E0_RECEIPT == f"{reg.EXP_DIR}/c4flip_data_curve_e0_receipt.json"
    assert b.index('git commit -m "exp_46 records: E0 PASS at $X46"') < b.index("python $E/exp46_eval.py prepare $RUN")


def test_exp38s_tools_always_run_without_bytecode():
    """exp_38's two read-only tools are run with -B (Codex code r1 finding 1: nothing is written into exp_38's folder;
    exp_46's own tools disable bytecode themselves)."""
    for name in SHEETS:
        for cmd in _commands(_text(name)):
            for m in re.finditer(r"(\$PY|python)( -B)? \$E38/exp38_\w+\.py", cmd):
                assert m.group(2) == " -B", (name, cmd[:120])


MAIN = "/n/fs/gatrdp/codespace/FLAC"            # the main checkout (the generator root G)
MAIN_FORMS = (r"G=/n/fs/gatrdp/codespace/FLAC",
              r"SRC=/n/fs/gatrdp/codespace/FLAC/outputs_FLAC/exp46_c4flip_data_curve/\$RUN",   # read by the L7 rsync
              r"cd /n/fs/gatrdp/codespace/FLAC && git fetch origin exp-46-c4flip-data-curve",
              r"git worktree add --detach /n/fs/gatrdp/codespace/exp-46-launch \$Y46",
              r'ln -s "\$\(readlink -f /n/fs/gatrdp/codespace/FLAC/[\w/.]+\)" [\w/.]+',
              r"git -C \$G fetch origin [\w. -]+", r"git -C \$G merge-base --is-ancestor \S+ \S+",
              r"git -C \$G worktree add --detach \$B \$R46",
              r"\$PY \$E/exp46_collect\.py materialise --dest \$G(\s+#.*)?",
              r"\$PY \$E/exp46_collect\.py verify --root \$G(\s+#.*)?",
              r"\$PY \$E/exp46_table\.py import-exp38 --commit \$B38 --generator-root \$G(\s+#.*)?",
              r"\$PY worklog/worklog_yixun/gen_model_comparison\.py --repo-root \$G --out \$B/worklog/worklog_yixun/"
              r"model_comparison\.md(\s+#.*)?")


@pytest.mark.parametrize("name", sorted(SHEETS))
def test_no_sheet_command_can_write_a_tracked_path_of_the_main_checkout(name):
    """Both regenerations read their evidence from G and write the table into B (--out); every other command touching
    the main checkout is a registered read-only / metadata / untracked-only form."""
    cwd_main = False
    for cmd in _commands(_text(name)):
        if cmd.startswith("cd "):
            cwd_main = cmd.split()[1] == MAIN
        if MAIN in cmd or "$G" in cmd or "${G}" in cmd or cwd_main:
            assert any(re.fullmatch(f, cmd) for f in MAIN_FORMS), f"{name}: {cmd}"
        if "gen_model_comparison.py" in cmd:
            assert "--out $B/" in cmd and "--out $G" not in cmd, cmd


def test_l8_makes_the_published_evidence_available_and_verified_before_any_use():
    b = _blocks(NEURONIC)
    l8 = b[b.index("G=/n/fs/gatrdp/codespace/FLAC"):]
    order = ["git -C $G fetch origin exp-46-c4flip-data-curve exp-38-data-efficiency",
             "git -C $G merge-base --is-ancestor $Y46 $R46", "git -C $G worktree add --detach $B $R46",
             "$PY $E/exp46_collect.py verify  ", "$PY $E/exp46_collect.py materialise --dest $G",
             "$PY $E/exp46_collect.py verify --root $G", "$PY worklog/worklog_yixun/gen_model_comparison.py --repo-root $G",
             "git -C $G fetch origin exp-38-data-efficiency",
             "git -C $G merge-base --is-ancestor e5a012fa62705627104bcd9892643edab07aa1e7 $B38",
             "$PY $E/exp46_table.py import-exp38 --commit $B38 --generator-root $G",
             "$PY $E/exp46_table.py build --exp38-commit $B38"]
    pos = [l8.index(c) for c in order]
    assert pos == sorted(pos), [order[i] for i in range(len(pos)) if pos[i] != sorted(pos)[i]]
    assert l8.rindex("gen_model_comparison.py --repo-root $G") > l8.index("exp46_table.py build")      # regen #2 last
    for verb in ("checkout", "pull", "merge ", "reset", "switch", "stash"):
        assert f"git -C $G {verb}" not in l8, verb


def test_c3_recovery_and_the_nas_admission_are_stated():
    text, b = _text(NEURONIC), _blocks(NEURONIC)
    assert "C3 (whole-campaign" in text and "read exp_45's reservation" in text and "Recovery:" in text
    assert "--exclude .exp46_tools" in b and "sha256sum -c CHECKPOINTS.sha256" in b and "exp46_registry.py training-check" in b
    assert b.index("training-check") < b.index("mv $DST/$RUN.incoming $DST/$RUN")
