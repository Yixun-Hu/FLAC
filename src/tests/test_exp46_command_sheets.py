"""exp_46 — the two command sheets (fix round 1, Codex code r1 finding 10 / open item 7): the neuronic sheet
(c4flip_data_curve_neuronic_commands.md: L2 / L1 / L3 / L4, the four SMOKEs and FULLs with f100's extra hour, C3 and
recovery, the staged NAS publication through exp_46's own admission, the verified materialisation / import and both
regenerations writing outside main's tracked tree) and the A6000 sheet (c4flip_data_curve_command.md: E0, the Y46
records commit, the 40 evaluations and the results commit -- evaluation and publication ONLY). They stay well-formed,
name exp_46's own kit and registered paths, and keep the campaign order Yixun approved on 2026-11-18 (D2).

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
        "python -B $E/exp46_eval.py env-receipt --write", "python -B $E/exp46_eval.py prepare-e0",
        "python -B $E/exp46_eval.py run --gpu 1 --cells de_E0a_K8_s42 de_E0b_K8_s42 de_E0c_K8_s42 de_E0d_K8_s42",
        "python -B $E/exp46_eval.py e0 ", "python -B $E/exp46_eval.py e0-check",
        "git add -f $E/c4flip_data_curve_e0_receipt.json{,.sha256} outputs_FLAC/exp46_c4flip_data_curve/eval/E0{a,b,c,d}",
        'git commit -m "exp_46 records: E0 PASS at $X46"', "python -B $E/exp46_eval.py prepare $RUN",
        "python -B $E/exp46_collect.py", "python -B $E/exp46_collect.py verify",
        "git -C ~/codespace/exp-46-publish pull --ff-only",
        "python -B $E/exp46_collect.py materialise --dest ~/codespace/exp-46-publish",
        'git commit -m "exp_46 results publication: 40 C4F cells"',
        "git -C ~/codespace/FLAC worktree add -b exp-46-c4flip-data-curve ~/codespace/exp-46-publish $X46",
        'test "$(git rev-parse HEAD)" = "$X46"', "test -z \"$(git status --porcelain --untracked-files=no)\""],
    NEURONIC: [
        "$PY -B $E/exp46_content_manifest.py --build $E/exp46_f100_delta_manifest.txt.gz",
        'git commit -m "exp_46 L2: pin the f100 delta manifest <sha256(text)>"',
        "$PY -B $E38/exp38_subset_audit.py --verify $E38/data_efficiency_subset_audit.json",
        "$PY -B $E38/exp38_content_manifest.py --verify $E38/data_efficiency_ar_content_manifest.txt.gz",
        "$PY -B $E/exp46_content_manifest.py --verify $E/exp46_f100_delta_manifest.txt.gz",
        "$PY -B $E/exp46_registry.py check exp46_C4Ff100_l40_2x32 --platform neuronic",
        "$PY -B $E/exp46_init_digest.py --run $RUN --json",
        "git -C /n/fs/gatrdp/codespace/FLAC worktree add --detach /n/fs/gatrdp/codespace/exp-46-launch $Y46",
        "$PY -B $E/exp46_registry.py e0-check --platform neuronic", "$PY -B $E/exp46_registry.py untracked --platform neuronic",
        "$PY -B $E/exp46_registry.py dry-run --platform neuronic",
        "exp46_submit_neuronic.py submit --run $RUN --mode SMOKE", "exp46_submit_neuronic.py submit --run $RUN --mode FULL",
        "exp46_submit_neuronic.py reconcile", "exp46_submit_neuronic.py abandon", "exp46_submit_neuronic.py release --run $RUN",
        "exp46_nas_publish.py stage --dest-root $DST --run $RUN", "exp46_nas_publish.py publish --dest-root $DST --run $RUN",
        "git -C $M push origin exp-46-c4flip-data-curve",
        "training-check $RUN --run-dir $DST/$RUN.incoming --platform neuronic", "--closure", "--e0-sha",
        "G=/n/fs/gatrdp/codespace/FLAC", "git -C $G fetch origin exp-46-c4flip-data-curve exp-38-data-efficiency",
        "git -C $G merge-base --is-ancestor $Y46 $R46", "git -C $G worktree add --detach $B $R46",
        'test "$(git rev-parse HEAD)" = "$R46"', "$PY -B $E/exp46_collect.py verify",
        "$PY -B $E/exp46_collect.py materialise --dest $G", "$PY -B $E/exp46_collect.py verify --root $G",
        "git -C $G merge-base --is-ancestor e5a012fa62705627104bcd9892643edab07aa1e7 $B38",
        "$PY -B $E/exp46_table.py import-exp38 --commit $B38 --generator-root $G",
        "$PY -B $E/exp46_table.py build --exp38-commit $B38",
        "$PY -B worklog/worklog_yixun/gen_model_comparison.py --repo-root $G --out $B/worklog/worklog_yixun/model_comparison.md",
        "$PY -B $E/exp46_regression_compare.py --merged $L1/merged.junit.xml --exp38 $L1/parent_3593f34c.junit.xml "
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
    assert not re.search(r"(^|\n)(MODE=|python (-B )?\$E/|\$PY (-B )?\$E/)", outside), "a command outside a code block"
    assert "--resume" not in text and "RESUME_STEP=" not in text
    assert not re.search(r"exp45_\w+\.py", joined), "exp_45's kit"
    assert "exp38_eval.py" not in joined and "exp38_submit_neuronic.py" not in joined and "exp38_supervise.py" not in joined
    for script in sorted(set(re.findall(r"\$E/(exp46_\w+\.py)", joined))):          # every exp_46 tool it calls exists
        assert os.path.isfile(os.path.join(EXP, script)), script


def test_the_campaign_order_is_the_recommended_one_and_marked_approved_by_yixun_d2():
    """The coordinator's recommendation (Codex code r1 open item 10), approved by Yixun 2026-11-18 (D2): final code round -> L2
    delta build -> pin commit -> definitive L1 -> L2 union verification before any f100 SMOKE -> L3 -> L4 freeze X46."""
    text, b = _text(NEURONIC), _blocks(NEURONIC)
    approved = "approved by Yixun 2026-11-18 (D2)"                    # D2 decided (Yixun 2026-11-18): no "pending" left
    assert approved in text and b.count(approved) >= 2 and "Yixun's D1" in text
    assert "pending Yixun D2" not in text and "pending Yixun D2" not in _text(A6000) and approved in _text(A6000)
    order = ["$PY -B $E/exp46_content_manifest.py --build $E/exp46_f100_delta_manifest.txt.gz",
             'git commit -m "exp_46 L2: pin the f100 delta manifest',
             "--junitxml=$L1/merged.junit.xml", "$PY -B $E/exp46_regression_compare.py --merged",
             "$PY -B $E/exp46_content_manifest.py --verify $E/exp46_f100_delta_manifest.txt.gz",
             "$PY -B $E/exp46_init_digest.py --run $RUN", "$PY -B $E/exp46_registry.py dry-run --platform neuronic",
             "git -C $M push origin exp-46-c4flip-data-curve", "Y46=<", "--mode SMOKE --time 01:00:00\n",
             "--mode FULL --time <computed>\n"]
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
    assert suite.endswith("-m pytest -q -p no:cacheprovider --continue-on-collection-errors src/tests") and " -B -I " in suite
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


def test_the_neuronic_sheet_exports_exp38s_curated_dinov3_hub_for_every_step():
    """Yixun 2026-11-19, option 1a: the variables block exports HF_HUB_CACHE = exp_38's curated two-file DINOv3 hub
    (reg.EXP38_HF_HUB, the hub_cache exp_38's neuronic launch records name; read-only) next to the unchanged HF_HOME and
    HF_HUB_OFFLINE=1, before any step, and nothing in the sheet sets, drops or overrides it: L1's suites, L3 and every
    SMOKE / FULL job (sbatch passes the environment) see the snapshot vit_pin_check pins."""
    lines = re.search(r"```bash\n(.*?)\n```", _text(NEURONIC), flags=re.S).group(1).split("\n")
    code = [ln for ln in lines if not ln.lstrip().startswith("#")]
    sets = [ln for ln in code if "HF_HUB_CACHE" in ln or "HF_HOME" in ln]
    want = (f"export PYTHONNOUSERSITE=1 HF_HOME=/n/fs/gatrdp/hf_cache HF_HUB_CACHE={reg.EXP38_HF_HUB} HF_HUB_OFFLINE=1; "
            "unset PYTHONOPTIMIZE PYTHONPATH")
    assert sets == [want], sets
    i = lines.index(want)
    why = lines[i - 1]
    assert why.startswith("#") and all(w in why for w in ("exp_38", "two-file", "hub_cache", "read-only")), why
    assert i < min(j for j, ln in enumerate(lines) if ln.startswith(("cd ", "sbatch ", "$PY ", "git ")))
    assert not re.search(r"--export=NONE|env -i|-u HF_|unset [^\n#]*HF_", "\n".join(code))
    suite = re.search(r'^SUITE="([^"]+)"', "\n".join(lines), flags=re.M).group(1)
    assert "HF_" not in suite and reg.EXP38_HF_HUB == "/n/fs/gatrdp/exp38_hf_hub"


def _submits(body, mode):
    """The REAL submissions of ``mode`` in a loop body (L4's and the inspection dry-runs submit nothing)."""
    return [ln for ln in body.split("\n") if f"--mode {mode}" in ln and "--dry-run" not in ln]


def test_the_f100_full_gets_one_more_hour_and_every_full_is_one_monolithic_job():
    b = _blocks(NEURONIC)
    loops = re.findall(r"^for RUN in ([^;\n]+); do\n(.*?)^done$", b, flags=re.M | re.S)
    full = {r: body for head, body in loops if _submits(body, "FULL") for r in head.split()}
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
    smoke = [r for head, body in loops if _submits(body, "SMOKE") for r in head.split()]
    first = {m: min(b.index(ln) for head, body in loops for ln in _submits(body, m)) for m in ("SMOKE", "FULL")}
    assert smoke == ["$RUNS"] and first["SMOKE"] < first["FULL"]
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
    assert b.index('git commit -m "exp_46 records: E0 PASS at $X46"') < b.index("python -B $E/exp46_eval.py prepare $RUN")


PY_CALL = re.compile(r"(?<![\w/=.-])(\$PY|python)((?:\s+-[A-Za-z]+)*)(?=\s)")    # an interpreter call and its flags


def test_every_python_invocation_of_both_sheets_runs_without_bytecode():
    """Codex code r1 finding 1, r2 finding 2: EVERY python invocation of both sheets carries -B -- exp_38's read-only
    tools, the shared generator (both regenerations: it imports exp_38's evaluator and registry from G), exp_46's own
    tools and the inline helpers alike -- and the L1 suite runs isolated (-B -I)."""
    calls = []
    for name in SHEETS:
        for cmd in _commands(_text(name)):
            for m in PY_CALL.finditer(cmd):
                calls.append((name, m.group(1), m.group(2).split(), cmd))
    assert len([c for c in calls if c[0] == NEURONIC]) >= 25 and len([c for c in calls if c[0] == A6000]) >= 10
    assert [(n, cmd[:120]) for n, _, flags, cmd in calls if "-B" not in flags] == []
    regen = [flags for n, _, flags, cmd in calls if "gen_model_comparison.py" in cmd]
    assert regen == [["-B"], ["-B"]]
    assert [flags[:2] for n, _, flags, cmd in calls if "-m pytest" in cmd] == [["-B", "-I"]]
    assert len([c for c in calls if "$E38/exp38_" in c[3]]) >= 2


MAIN = "/n/fs/gatrdp/codespace/FLAC"            # the main checkout (the generator root G)
MAIN_FORMS = (r"G=/n/fs/gatrdp/codespace/FLAC",
              r"SRC=/n/fs/gatrdp/codespace/FLAC/outputs_FLAC/exp46_c4flip_data_curve/\$RUN",   # read by the L7 rsync
              r"git -C /n/fs/gatrdp/codespace/FLAC fetch origin exp-46-c4flip-data-curve",
              r"git -C /n/fs/gatrdp/codespace/FLAC worktree add --detach /n/fs/gatrdp/codespace/exp-46-launch \$Y46",
              r'ln -s "\$\(readlink -f /n/fs/gatrdp/codespace/FLAC/[\w/.]+\)" [\w/.]+',
              r"git -C \$G fetch origin [\w. -]+", r"git -C \$G merge-base --is-ancestor \S+ \S+",
              r"git -C \$G worktree add --detach \$B \$R46",
              r"\$PY -B \$E/exp46_collect\.py materialise --dest \$G(\s+#.*)?",
              r"\$PY -B \$E/exp46_collect\.py verify --root \$G(\s+#.*)?",
              r"\$PY -B \$E/exp46_table\.py import-exp38 --commit \$B38 --generator-root \$G(\s+#.*)?",
              r"\$PY -B worklog/worklog_yixun/gen_model_comparison\.py --repo-root \$G --out \$B/worklog/worklog_yixun/"
              r"model_comparison\.md(\s+#.*)?",
              r"SAVE=/n/fs/gatrdp/codespace/FLAC/outputs_FLAC/exp46_c4flip_data_curve",      # read by the recovery check
              r"for f in \$E38/ref_cells/REF_MANIFEST\.sha256 \$E38/data_efficiency_e0a_decision\.json "
              r"\$E38/data_efficiency_e0a_decision\.json\.sha256; do cmp \$B/\$f \$G/\$f \|\| exit 1; done",
              r"\(cd \$G/\$E38/ref_cells && sha256sum -c --quiet REF_MANIFEST\.sha256\)")


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
             "git -C $G merge-base --is-ancestor e5a012fa62705627104bcd9892643edab07aa1e7 $B38",
             "$PY -B $E/exp46_table.py import-exp38 --commit $B38 --generator-root $G",
             "do cmp $B/$f $G/$f || exit 1; done", "(cd $G/$E38/ref_cells && sha256sum -c --quiet REF_MANIFEST.sha256)",
             "$PY -B $E/exp46_collect.py verify  ", "$PY -B $E/exp46_collect.py materialise --dest $G",
             "$PY -B $E/exp46_collect.py verify --root $G", "$PY -B worklog/worklog_yixun/gen_model_comparison.py --repo-root $G",
             "$PY -B $E/exp46_table.py build --exp38-commit $B38"]
    pos = [l8.index(c) for c in order]
    assert pos == sorted(pos), [order[i] for i in range(len(pos)) if pos[i] != sorted(pos)[i]]
    assert l8.rindex("gen_model_comparison.py --repo-root $G") > l8.index("exp46_table.py build")      # regen #2 last
    for verb in ("checkout", "pull", "merge ", "reset", "switch", "stash"):
        assert f"git -C $G {verb}" not in l8, verb


def test_exp38s_static_dependencies_reach_g_verified_before_the_materialisation():
    """Codex code r2 finding 5: collect.materialise only VERIFIES exp_38's static dependencies of the exp_46 rows in G
    (generator_static's exp_38-owned files: the E0a decision + sidecar, the reference manifest + its 68 cells) and
    never copies them. The kit's verified path brings them in first -- the exp_38 import (every file sha256-hashed into
    the sealed record, copied identical-or-absent, re-hashed) -- then the sheet checks exactly those files against B's
    tracked copies before materialising, with a stated recovery (stop; never copy exp_38's files by hand)."""
    import exp46_collect as col
    import exp46_table as tbl
    import inspect
    text, b, cmds = _text(NEURONIC), _blocks(NEURONIC), _commands(_text(NEURONIC))
    owned = sorted(r for r in col.generator_static() if r.startswith(reg.EXP38_DIR + "/"))
    assert owned == sorted([reg.E0A_DECISION, reg.E0A_DECISION + ".sha256", reg.REF_MANIFEST]
                           + [f"{reg.REF_DIR}/{r}" for r in reg.ref_set()])
    src = inspect.getsource(tbl.exp38_import_files)                  # the verified import copies exactly these
    for needle in ("ref_cells/REF_MANIFEST.sha256", "reg.ref_set()", "data_efficiency_e0a_decision.json"):
        assert needle in src, needle
    cmp = [c for c in cmds if c.startswith("for f in ") and "do cmp $B/$f $G/$f || exit 1" in c][0]
    checked = cmp[len("for f in "):cmp.index(";")].split()
    assert sorted(x.replace("$E38", reg.EXP38_DIR) for x in checked) == sorted(
        [reg.REF_MANIFEST, reg.E0A_DECISION, reg.E0A_DECISION + ".sha256"])
    assert "(cd $G/$E38/ref_cells && sha256sum -c --quiet REF_MANIFEST.sha256)" in cmds       # the 68 cells it lists
    assert reg.REF_DIR == f"{reg.EXP38_DIR}/ref_cells" and reg.REF_MANIFEST == f"{reg.REF_DIR}/REF_MANIFEST.sha256"
    imp = cmds.index("$PY -B $E/exp46_table.py import-exp38 --commit $B38 --generator-root $G")
    assert imp < cmds.index(cmp) < cmds.index("$PY -B $E/exp46_collect.py materialise --dest $G")
    assert "STOP -- never copy exp_38's files by hand" in b


def test_c3_recovery_and_the_nas_admission_are_stated():
    text, b = _text(NEURONIC), _blocks(NEURONIC)
    assert "C3 (whole-campaign" in text and "read exp_45's reservation" in text and "Recovery:" in text
    assert "--exclude .exp46_tools" in b and "sha256sum -c CHECKPOINTS.sha256" in b and "exp46_registry.py training-check" in b
    assert b.index("training-check") < b.index("exp46_nas_publish.py publish --dest-root $DST --run $RUN")


def test_the_nas_publication_stages_exclusively_and_publishes_without_replacing():
    """Codex code r2 finding 8: the staging directory is created exclusively by the tested helper BEFORE rsync, the
    admission runs on the staged copy, and the final move is the helper's no-replace publication -- no plain
    existence test, no ordinary mv (which nests the staging dir in a final path created meanwhile)."""
    cmds = _commands(_text(NEURONIC))
    i = cmds.index("python -B $E/exp46_nas_publish.py stage --dest-root $DST --run $RUN")
    chain = cmds[i - 1:i + 5]                  # one command per line (fix round 3: set -e stops at any refusal)
    assert chain[0] == "python -B $E/exp46_nas_publish.py provision"
    assert chain[2].startswith("rsync -a --partial --exclude .exp46_tools neuronic:$SRC/ $DST/$RUN.incoming/")
    assert chain[3] == "(cd $DST/$RUN.incoming && sha256sum -c CHECKPOINTS.sha256)"
    assert chain[4].startswith("python -B $E/exp46_registry.py training-check $RUN --run-dir $DST/$RUN.incoming")
    assert chain[5] == "python -B $E/exp46_nas_publish.py publish --dest-root $DST --run $RUN"
    assert len([c for c in cmds if "rsync " in c]) == 1
    b = _blocks(NEURONIC)
    assert "mv $DST" not in b and "test ! -e $DST" not in b
    assert os.path.isfile(os.path.join(EXP, "exp46_nas_publish.py"))


def test_l4_dry_runs_every_launch_path_of_the_four_runs_before_the_x46_freeze():
    """Codex code r2 finding 6: L4's dry-runs are executable commands, not a comment -- the registry's four-run dry-run
    and every run's SMOKE and FULL submitter dry-runs -- after L3 and before the push that freezes X46 (and before the
    A6000 E0 / Y46)."""
    cmds = _commands(_text(NEURONIC))
    b = _blocks(NEURONIC)
    l3 = max(i for i, c in enumerate(cmds) if "exp46_init_digest.py --run $RUN" in c)
    push = cmds.index("git -C $M push origin exp-46-c4flip-data-curve")
    assert cmds[push + 1].startswith("X46=$(git -C $M rev-parse HEAD)")
    l4 = cmds[l3 + 2:push]                                                          # after the L3 loop
    assert cmds[l3 + 1] == "done" and l4[:2] == ["cd $M", 'test -z "$(git status --porcelain --untracked-files=no)"']
    assert "$PY -B $E/exp46_registry.py dry-run --platform neuronic" in l4
    loop = re.search(r"^for RUN in \$RUNS; do\n(.*?)^done$", b[b.index("dry-run --platform neuronic"):], re.M | re.S)
    body = loop.group(1)
    assert b.index("dry-run --platform neuronic") + loop.start() < b.index("git -C $M push origin")
    for mode, t in (("SMOKE", "01:00:00"), ("FULL", "7-00:00:00")):
        assert f"$PY -B $E/exp46_submit_neuronic.py submit --run $RUN --mode {mode} --time {t} --dry-run" in body
    assert not _submits(body, "SMOKE") and not _submits(body, "FULL")             # dry-runs only: nothing submitted
    assert re.search(r'^RUNS="([^"]+)"', b, flags=re.M).group(1).split() == list(reg.campaign_runs())
    assert push < next(i for i, c in enumerate(cmds) if c.startswith("Y46=<"))


def test_recovery_keeps_a_sealed_run_and_abandons_only_an_unsealed_attempt():
    """Codex code r2 finding 9: the recovery tells a SEALED run (its commit marker) from an unsealed attempt; a sealed
    run is kept and only its stranded storage hold is freed with `release --run`; abandon is for unsealed attempts only
    (and refuses a sealed run itself)."""
    cmds = _commands(_text(NEURONIC))
    b = _blocks(NEURONIC)
    check = next(i for i, c in enumerate(cmds) if "TRAINING_COMMITTED.json" in c and "SEALED" in c)
    rel = cmds.index("$PY -B $E/exp46_submit_neuronic.py release --run $RUN")
    ab = next(i for i, c in enumerate(cmds) if c.startswith("$PY -B $E/exp46_submit_neuronic.py abandon "))
    rec = next(i for i, c in enumerate(cmds) if c.startswith("$PY -B $E/exp46_submit_neuronic.py reconcile "))
    assert check < rel < rec < ab
    assert reg.COMMIT_MARKER == "TRAINING_COMMITTED.json" and "SAVE=" + reg.PLATFORMS["neuronic"]["save_root"] in cmds
    assert "SEALED: KEEP the run" in b and "it REFUSES a sealed run" in b


def test_the_a6000_publication_worktree_is_created_at_x46_and_is_the_only_place_that_commits():
    """Codex code r2 finding 7: the A6000 sheet creates ~/codespace/exp-46-publish itself, as the launch branch at X46,
    verifies its HEAD, branch, upstream and a clean tree before anything is copied in, runs fail-fast (set -e; every cd
    `|| exit 1`), and stages / commits ONLY there -- never in tree 46 (the campaign checkout)."""
    cmds = _commands(_text(A6000))
    pub, camp = "~/codespace/exp-46-publish", "~/codespace/exp-46-campaign"
    add = cmds.index(f"git -C ~/codespace/FLAC worktree add -b exp-46-c4flip-data-curve {pub} $X46")
    assert cmds.index("set -e") < cmds.index(f"cd {camp} || exit 1") < add
    assert all(c.endswith("|| exit 1") for c in cmds if c.startswith("cd "))
    first_cd = cmds.index(f"cd {pub} || exit 1")
    checks = ['test "$(git rev-parse HEAD)" = "$X46"', 'test "$(git rev-parse --abbrev-ref HEAD)" = exp-46-c4flip-data-curve',
              'test "$(git rev-parse origin/exp-46-c4flip-data-curve)" = "$X46"',
              'test -z "$(git status --porcelain --untracked-files=no)"']
    assert add < first_cd < min(cmds.index(c) for c in checks)
    first_copy = next(i for i, c in enumerate(cmds) if c.startswith("cp "))
    assert max(cmds.index(c) for c in checks) < first_copy
    assert not [c for c in cmds if re.match(r"test .* && test ", c)]          # (set -e never stops inside an && list)
    cwd, staged = None, []
    for c in cmds:
        if c.startswith("cd "):
            cwd = c.split()[1]
        m = re.match(r"git (?:-C (\S+) )?(add|commit)\b", c)
        if m:
            staged.append(m.group(1) or cwd)
    assert staged and set(staged) == {pub}, staged


def test_the_regeneration_writes_no_bytecode_into_exp38s_folder(tmp_path):
    """Codex code r2 finding 2 (HIGH): the shared generator's exp_38 path (validate_exp38_cell -> its exp_38 loader)
    imports exp_38's evaluator and registry from the generator root. Run as its own process on a copied tree -- no
    other validator first -- with the interpreter flags of the sheet's regeneration commands, it writes no bytecode
    into exp_38's folder; the same call without -B does (the check is live, not vacuous)."""
    import shutil
    import subprocess
    flags = {tuple(m.group(2).split()) for c in _commands(_text(NEURONIC)) if "gen_model_comparison.py" in c
             for m in [PY_CALL.search(c)]}
    assert flags == {("-B",)}
    gen_rel = "worklog/worklog_yixun/gen_model_comparison.py"
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import gen_model_comparison as g; "
            "print(g.validate_exp38_cell([sys.argv[2]], repo_root=sys.argv[3], expected_run='de_yaw_f025', "
            "expected_k=8))")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONDONTWRITEBYTECODE", "PYTHONPATH")}
    for run_flags, want in ((list(flags.pop()), False), ([], True)):
        root = tmp_path / ("tree_" + ("_".join(run_flags) or "plain").strip("-"))
        (root / ".git").mkdir(parents=True)
        (root / reg.EXP38_DIR).mkdir(parents=True)
        for name in sorted(os.listdir(os.path.join(REPO, reg.EXP38_DIR))):
            if name.endswith(".py"):
                shutil.copyfile(os.path.join(REPO, reg.EXP38_DIR, name), root / reg.EXP38_DIR / name)
        shutil.copyfile(os.path.join(REPO, gen_rel), root / gen_rel)
        cell = root / "outputs_FLAC" / "exp38_data_efficiency" / "eval" / "de_yaw_f025" / "x_metrics.json"
        cell.parent.mkdir(parents=True)
        cell.write_text("{}")
        res = subprocess.run([sys.executable, *run_flags, "-c", code, str(root / os.path.dirname(gen_rel)), str(cell),
                              str(root)], cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=300)
        assert res.returncode == 0 and "False" in res.stdout, res.stderr[-2000:]          # the loader ran, then refused
        cache = [a for a, ds, fs in os.walk(str(root / reg.EXP38_DIR)) if "__pycache__" in ds]
        assert bool(cache) is want, (run_flags, cache)


# ------------------------------------------------------------------ fix round 3: the gates themselves (Codex code r3)
STUB_SBATCH = r'''#!/bin/bash
# stand-in sbatch: logs the call and runs the --wrap command here, as the job would (output to --output); a job whose
# wrap names $SBATCH_KILL is "killed" -- not run, status 1 (what `sbatch --wait` reports for a signalled job)
wait=0; wrap=""; out=/dev/null
for a in "$@"; do case "$a" in --wait) wait=1;; --output=*) out="${a#--output=}";; esac; done
while [ $# -gt 0 ]; do [ "$1" = "--wrap" ] && wrap="$2"; shift; done
echo "sbatch $wrap" >> "$STUB/calls.log"
if [ -n "$SBATCH_KILL" ] && [[ "$wrap" == *"$SBATCH_KILL"* ]]; then rc=1; else bash -c "$wrap" > "${out//%j/4242}" 2>&1; rc=$?; fi
[ $wait = 1 ] && exit $rc
echo "Submitted batch job 4242"
'''
STUB_GIT = r'''#!/bin/bash
echo "git $*" >> "$STUB/calls.log"
[ "$1" = "-C" ] && shift 2
case "$1" in
  push) touch "$STUB/PUSHED";;
  rev-parse) echo 0123456789abcdef0123456789abcdef01234567;;
  worktree) [ "$2" = add ] && mkdir -p "$4/weights/FLAC";;
esac
exit 0
'''
STUB_PY = r'''#!/bin/bash
# stand-in interpreter: logs the call; each tool exits with the status the test injects for it
echo "py $*" >> "$STUB/calls.log"
for a in "$@"; do case "$a" in --junitxml=*) echo '<testsuites/>' > "${a#--junitxml=}";; esac; done
case "$*" in
  *"-m pytest"*) exit ${PYTEST_RC:-0};;
  *exp46_regression_compare.py*) exit ${COMPARE_RC:-0};;
  *"exp46_content_manifest.py --build"*) exit ${BUILD_RC:-0};;
  *"exp46_content_manifest.py --verify"*) exit ${VERIFY_RC:-0};;
  *exp38_subset_audit.py*|*exp38_content_manifest.py*) exit ${AUDIT_RC:-0};;
  *"exp46_registry.py check"*) exit ${CHECK_RC:-0};;
  *exp46_init_digest.py*) exit ${L3_RC:-0};;
  *"exp46_registry.py dry-run"*) exit ${DRYRUN_RC:-0};;
  *"--mode FULL"*) exit ${FULL_DRY_RC:-0};;
  *"--mode SMOKE"*) exit ${SMOKE_DRY_RC:-0};;
esac
exit 0
'''
LADDER_END = "X46=$(git -C $M rev-parse HEAD)"
LADDER_VARS = {"PY=/n/fs/gatrdp/envs/flac/bin/python;": "PY=$STUB/py;",
               "M=/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve\n": "M=$STUB/m\n",
               "P38=/n/fs/gatrdp/codespace/exp-46-l1-parent-3593f34c; P43=/n/fs/gatrdp/codespace/exp-46-l1-parent-4a7e3b3a":
               "P38=$STUB/p38; P43=$STUB/p43"}


def _ladder(tmp_path, **inject):
    """The neuronic sheet's L2a ... L4 ladder, VERBATIM up to the X46 line, run as the bash script the sheet says it is
    -- only its interpreter, checkout and parent paths point at stand-ins (sbatch / git / python, which run each job's
    wrap and exit with the injected status). Returns (exit status, the call log, whether the push ran)."""
    import subprocess
    text = _text(NEURONIC)
    block = re.search(r"```bash\n(.*?)\n```", text, flags=re.S).group(1)
    script = block[:block.index(LADDER_END) + len(LADDER_END)]
    for old, new in LADDER_VARS.items():
        assert script.count(old) == 1, old
        script = script.replace(old, new)
    stub = tmp_path / "stub"
    (stub / "bin").mkdir(parents=True)
    for path, body in ((stub / "bin" / "sbatch", STUB_SBATCH), (stub / "bin" / "git", STUB_GIT), (stub / "py", STUB_PY)):
        path.write_text(body)
        path.chmod(0o755)
    (stub / "m" / reg.EXP_DIR).mkdir(parents=True)
    (stub / "m" / "weights" / "FLAC").mkdir(parents=True)
    env = dict(os.environ, STUB=str(stub), PATH=f"{stub / 'bin'}:{os.environ['PATH']}", **inject)
    res = subprocess.run(["bash", "-c", script], cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=300)
    return res.returncode, (stub / "calls.log").read_text(), (stub / "PUSHED").exists()


LADDER_FAILURES = {"the L2a delta build": {"BUILD_RC": "1"},
                   "an L1 suite killed (no rc marker)": {"SBATCH_KILL": "merged.junit.xml"},
                   "an L1 session broken (pytest rc 2)": {"PYTEST_RC": "2"},
                   "the L1 comparison": {"COMPARE_RC": "1"},
                   "the L2c exp_38 audit": {"AUDIT_RC": "1"},
                   "the L2c delta verification": {"VERIFY_RC": "1"},
                   "the f100 static check": {"CHECK_RC": "1"},
                   "an L3 disagreement": {"L3_RC": "1"},
                   "the registry dry-run": {"DRYRUN_RC": "1"},
                   "a SMOKE submitter dry-run": {"SMOKE_DRY_RC": "2"},
                   "a FULL submitter dry-run": {"FULL_DRY_RC": "2"}}


@pytest.mark.parametrize("case", sorted(LADDER_FAILURES))
def test_the_neuronic_ladder_stops_at_every_failed_gate_before_the_x46_push(tmp_path, case):
    """Codex code r3 finding 6: the sheet's own L2a ... L4 lines, executed with a single failure injected at one gate,
    stop there -- the script exits non-zero and the push that freezes X46 never runs (Slurm steps block on the job's
    status: sbatch --wait; L1 judges each suite's recorded rc; set -eo pipefail)."""
    rc, calls, pushed = _ladder(tmp_path, **LADDER_FAILURES[case])
    assert rc != 0 and not pushed and " push " not in calls, (case, rc, calls[-500:])


def test_the_neuronic_ladder_reaches_the_push_only_when_every_gate_passed(tmp_path):
    """With every gate passing (L1 suites with failing nodes -- pytest rc 1 -- are judged by the comparison, not
    refused), the same lines run in the registered order and end with the push."""
    rc, calls, pushed = _ladder(tmp_path, PYTEST_RC="1")
    assert rc == 0 and pushed, calls[-800:]
    order = ["exp46_content_manifest.py --build", "git commit -m exp_46 L2: pin", "-m pytest", "exp46_regression_compare.py",
             "exp38_subset_audit.py", "exp46_content_manifest.py --verify", "exp46_registry.py check",
             "exp46_init_digest.py", "exp46_registry.py dry-run", "--mode FULL", "push origin"]
    pos = [calls.index(n) for n in order]
    assert pos == sorted(pos), [order[i] for i in range(len(pos)) if pos[i] != sorted(pos)[i]]
    lines = calls.split("\n")
    assert len([ln for ln in lines if ln.startswith("sbatch ")]) == 1 + 3 + 1 + len(reg.campaign_runs())  # L2a, L1, L2c, L3
    assert len([ln for ln in lines if ln.startswith("py ") and "exp46_init_digest.py" in ln]) == len(reg.campaign_runs())
    assert len([ln for ln in lines if ln.startswith("py ") and "--dry-run" in ln]) == 2 * len(reg.campaign_runs())


def test_the_a6000_eval_lanes_are_waited_on_by_pid_and_either_failure_stops_the_sheet(tmp_path):
    """Codex code r3 finding 5: the sheet's own lane lines, run with failing stand-in lanes (a shell function in place of
    the evaluator): each lane is waited on by its PID, the remaining lane is always reaped (it finishes; its cells are
    independent), and a failure of either lane stops the sheet before anything after it -- a bare `wait` would
    have returned 0."""
    import subprocess
    b = _blocks(A6000)
    start = b.index("python -B $E/exp46_eval.py run --gpu 0 --cells ${RUN}_K8_s42")
    end = b.index("\n", b.index("EXP46 EVAL LANES FAILED"))
    lanes = b[start:end]
    assert not re.search(r"(^|[;&]\s*)wait\s*($|;)", lanes, flags=re.M), "a bare wait"
    assert "wait $K8" in lanes and "wait $K1" in lanes and "K8=$!" in lanes and "K1=$!" in lanes
    prelude = ('set -e\nE=e\nRUN=r\npython() {\n'
               '  case "$*" in\n'
               '    *"--gpu 0"*) sleep "$K8_SLEEP"; echo k8 >> "$T/done"; return "$K8_RC";;\n'
               '    *"--gpu 1"*) sleep "$K1_SLEEP"; echo k1 >> "$T/done"; return "$K1_RC";;\n'
               '  esac\n}\n')
    for k8, k1, s8, s1 in ((0, 0, 0, 0), (3, 0, 0, 1), (0, 4, 1, 0), (5, 6, 0, 0)):
        t = tmp_path / f"lanes_{k8}_{k1}"
        t.mkdir()
        env = dict(os.environ, T=str(t), K8_RC=str(k8), K1_RC=str(k1), K8_SLEEP=str(s8), K1_SLEEP=str(s1))
        res = subprocess.run(["bash", "-c", prelude + lanes + "\necho AFTER\n"], env=env, capture_output=True, text=True,
                             timeout=120)
        ok = k8 == 0 and k1 == 0
        assert (res.returncode == 0) is ok and ("AFTER" in res.stdout) is ok, (k8, k1, res.returncode, res.stderr)
        assert sorted((t / "done").read_text().split()) == ["k1", "k8"], (k8, k1)     # both lanes reaped
        assert ok or f"K8 rc {k8}, K1 rc {k1}" in res.stderr


def test_both_new_nas_roots_are_provisioned_before_their_first_use():
    """Codex code r3 finding 7: the A6000 sheet provisions the exp_46 checkpoint and dump roots (the helper verifies the
    NAS mount, creates only those two through the isolation guard, refuses a missing mount) in step 0, before E0 and
    any evaluation; the neuronic L7 provisions (idempotently) before it stages into the checkpoint root."""
    a6, ne = _commands(_text(A6000)), _commands(_text(NEURONIC))
    prov = "python -B $E/exp46_nas_publish.py provision"
    assert a6.count(prov) == 1 and ne.count(prov) == 1
    assert a6.index(prov) < a6.index("python -B $E/exp46_eval.py prepare-e0")
    assert a6.index(prov) < min(i for i, c in enumerate(a6) if " run --gpu " in c)
    assert ne.index(prov) < ne.index("python -B $E/exp46_nas_publish.py stage --dest-root $DST --run $RUN")
