"""exp_21 round 5, integrative-review BLOCKING 4: the D6 comparator machinery.

D6 (plan §5, APPROVED by Yixun) re-evaluates BOTH comparators at the CURRENT
evaluator pin, because the historical B-F@40k and P1@40k rows were measured at a
different pin, under the legacy per-angle orbit executor, before
``--cond-autocast bf16`` and before the per-scene and stream provenance existed.
Subtracting across that gap would fold the whole evaluator shift into the arm's
effect, and ``model_comparison.md`` already marks legacy-loop and batched rows
non-interchangeable.

What is pinned here:

* the CAMPAIGN is one definition (``exp21_protocol.py``) — the driver that runs
  the cells and the table that admits them read the same module, so a flag
  cannot be right in one and wrong in the other;
* the per-arm TRAINED-AS contracts, which are three different shapes read off
  the real artifacts, not one template;
* the driver's dry inventory, token by token for one cell of each arm;
* and the cross-arm transaction: BFC minus a comparator is a paired delta only
  if both were measured at ONE pin, so the table refuses to render them as
  paired-comparable otherwise.

THE COUNT IS 34, NOT 35. The round-5 brief said "10 BFC + 5 grid + 20
comparator". The approved plan §5 says the invariance grid's 0-degree member IS
the registered K=8/seed-42 cell ("14 unique BFC cells — 10 registered + 4 extra
grid angles, the K8/s42/0 cell shared"), and after this round's finding-3 fix
that cell carries ``--record-stream`` like every registered cell, so its stream
is available to the grid. Re-running it under a second eval-name would spend a
GPU-hour to publish a second measurement of one thing. The plan governs; the
discrepancy is disclosed here and in the driver's header.
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXPDIR = REPO / "worklog" / "worklog_yixun" / "exp_21_bf_fa_cartesian_claude"
DRIVER = EXPDIR / "bfc_eval_driver.sh"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def P():
    return _load(EXPDIR / "exp21_protocol.py", "exp21_protocol")


@pytest.fixture(scope="module")
def gen():
    return _load(REPO / "worklog" / "worklog_yixun" / "gen_model_comparison.py",
                 "gen_model_comparison")


PH = "E"        # stand-in epoch for BFC, which has not trained yet


# --------------------------------------------------------------------------- #
# 1. the inventory
# --------------------------------------------------------------------------- #
def test_the_campaign_is_thirty_four_cells(P):
    cells = P.inventory()
    assert len(cells) == 34
    kinds = {}
    for cell in cells:
        kinds.setdefault((cell.arm, cell.kind), []).append(cell)
    assert len(kinds[("BFC", "registered")]) == 10
    assert len(kinds[("BFC", "grid")]) == 4
    assert len(kinds[("BFre", "comparator")]) == 10
    assert len(kinds[("P1re", "comparator")]) == 10


def test_the_grid_shares_the_registered_zero_degree_cell(P):
    """Plan §5. The 0-degree member is the registered K8/s42 cell, which now
    carries --record-stream — so it is not re-run under a second name."""
    grid = [c for c in P.inventory() if c.kind == "grid"]
    assert sorted(c.rotate_deg for c in grid) == [45.0, 90.0, 180.0, 270.0]
    assert all(c.k == 8 and c.seed == 42 for c in grid)
    registered = [c for c in P.inventory() if c.kind == "registered"]
    assert any(c.k == 8 and c.seed == 42 and c.rotate_deg == 0.0 for c in registered)


def test_every_cell_is_unique(P):
    """Two cells sharing an eval-name would overwrite one another's artifact:
    build_output_paths adds neither K nor the seed itself."""
    names = [P.eval_name(c.arm, c.k, c.seed, c.rotate_deg) for c in P.inventory()]
    assert len(set(names)) == len(names)


# --------------------------------------------------------------------------- #
# 2. the commands — token vectors, one cell per arm
# --------------------------------------------------------------------------- #
def test_bfc_command_token_vector(P):
    assert P.command("BFC", 8, 42, placeholder=PH) == [
        "python", "eval_FLAC.py",
        "--model-config",
        "worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/FLAC_AR_BFC.json",
        "--dataset-config",
        "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json",
        "--ckpt-path",
        "outputs_FLAC/exp21_BFC/FLAC_exp21_BFC/exp21_BFC/checkpoints/"
        "epoch=E-step=40000.ckpt",
        "--cond-method", "fa_cartesian",
        "--frame-avg-angles", "0,90,180,270",
        "--frame-avg-max-fwd-samples", "64",
        "--rotate-mode", "fixed", "--rotate-deg", "0",
        "--cond-autocast", "bf16",
        "--batch-size", "64", "--cfg-scale", "1.0", "--steps", "1",
        "--record-per-scene",
        "--record-stream", "--expected-stream-count", "6337",
        "--seed", "42", "--eval-name", "exp21_BFC_S40000_K8_s42",
    ]


def test_bfre_command_token_vector(P):
    """B-F re-evaluated: every token identical to BFC's except the model config,
    the checkpoint and --cond-method. That is what makes the delta the mechanism."""
    assert P.command("BFre", 8, 42) == [
        "python", "eval_FLAC.py",
        "--model-config",
        "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json",
        "--dataset-config",
        "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json",
        "--ckpt-path",
        "outputs_FLAC/exp07_BF/FLAC_exp07_BF/exp07_BF/checkpoints/"
        "epoch=8-step=40000.ckpt",
        "--cond-method", "fa_invariant",
        "--frame-avg-angles", "0,90,180,270",
        "--frame-avg-max-fwd-samples", "64",
        "--rotate-mode", "fixed", "--rotate-deg", "0",
        "--cond-autocast", "bf16",
        "--batch-size", "64", "--cfg-scale", "1.0", "--steps", "1",
        "--record-per-scene",
        "--record-stream", "--expected-stream-count", "6337",
        "--seed", "42", "--eval-name", "exp21_BFre_S40000_K8_s42",
    ]


def test_p1re_command_token_vector(P):
    """P1 runs NO orbit, so it carries no frame-average flags at all: passing
    them would record a protocol it did not execute and make P1's row look
    protocol-compatible with the frame-averaged arms."""
    assert P.command("P1re", 1, 46) == [
        "python", "eval_FLAC.py",
        "--model-config",
        "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json",
        "--dataset-config",
        "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json",
        "--ckpt-path",
        "outputs_FLAC/exp07_P1/FLAC_exp07_P1/exp07_P1/checkpoints/"
        "epoch=8-step=40000.ckpt",
        "--cond-method", "vanilla",
        "--rotate-mode", "fixed", "--rotate-deg", "0",
        "--cond-autocast", "bf16",
        "--batch-size", "64", "--cfg-scale", "1.0", "--steps", "1",
        "--record-per-scene",
        "--record-stream", "--expected-stream-count", "6337",
        "--seed", "46", "--eval-name", "exp21_P1re_S40000_K1_s46",
    ]


def test_the_grid_cell_only_adds_the_rotation(P):
    base = P.command("BFC", 8, 42, placeholder=PH)
    rot = P.command("BFC", 8, 42, 45.0, placeholder=PH)
    assert len(base) == len(rot)
    diff = [(a, b) for a, b in zip(base, rot) if a != b]
    assert diff == [("0", "45"),
                    ("exp21_BFC_S40000_K8_s42", "exp21_BFC_S40000_K8_s42_rot45")]


def test_every_command_carries_the_announcement_05_flag_set(P):
    for cell in P.inventory():
        argv = P.command(cell.arm, cell.k, cell.seed, cell.rotate_deg, placeholder=PH)
        for flag in ("--cond-autocast", "--record-per-scene", "--record-stream",
                     "--expected-stream-count", "--rotate-mode", "--seed",
                     "--eval-name", "--cond-method"):
            assert flag in argv, (cell, flag)
        assert argv[argv.index("--cond-autocast") + 1] == "bf16"
        assert argv[argv.index("--expected-stream-count") + 1] == "6337"


@pytest.mark.parametrize("arm,k,seed", [("BFC", 9, 42), ("BFC", 8, 99),
                                        ("NOPE", 8, 42)])
def test_an_unregistered_cell_is_refused(P, arm, k, seed):
    with pytest.raises((ValueError, KeyError)):
        P.command(arm, k, seed, placeholder=PH)


def test_metrics_paths_are_eval_FLACs_own_rule(P):
    """The protocol module predicts where each artifact lands (the driver's
    resume gate depends on it). Pinned against build_output_paths itself."""
    sys.path.insert(0, str(REPO))
    import eval_FLAC
    for cell in P.inventory():
        got = P.metrics_path(cell.arm, cell.k, cell.seed, cell.rotate_deg,
                             placeholder=PH)
        want = eval_FLAC.build_output_paths(
            P.resolve_ckpt(cell.arm, placeholder=PH), steps=1, cfg_scale=1.0,
            eval_name=P.eval_name(cell.arm, cell.k, cell.seed, cell.rotate_deg),
            cond_method=P.ARMS[cell.arm]["cond_method"],
            rotate_deg=cell.rotate_deg, n_angles=4,
        )["metrics"]
        assert got == want, cell


def test_the_comparator_artifacts_cannot_collide_with_what_is_already_there(P):
    """The comparator cells land in exp_07's checkpoint directories, because
    eval_FLAC writes beside the checkpoint it reads. Checked against the real
    directories: no existing file carries any of these names."""
    for cell in P.inventory():
        if cell.arm == "BFC":
            continue
        path = P.metrics_path(cell.arm, cell.k, cell.seed, cell.rotate_deg)
        assert not os.path.exists(path), f"would overwrite {path}"


# --------------------------------------------------------------------------- #
# 3. the per-arm trained-as contracts
# --------------------------------------------------------------------------- #
def test_the_three_arms_declare_three_different_training_shapes(P):
    """Not one template: these were read off the artifacts. B-F's checkpoint
    predates ``frame_avg_max_fwd_samples`` entirely, and P1's embedded config
    names no ``cond_method`` at all (the factory default IS vanilla)."""
    assert P.ARMS["BFC"]["expected_training"] == {
        "cond_method": "fa_cartesian",
        "frame_avg_angles": [0.0, 90.0, 180.0, 270.0],
        "frame_avg_max_fwd_samples": 32}
    assert P.ARMS["BFre"]["expected_training"]["cond_method"] == "fa_invariant"
    assert P.ARMS["BFre"]["expected_training"]["frame_avg_max_fwd_samples"] is P.ABSENT
    assert P.ARMS["P1re"]["expected_training"] == {
        "cond_method": "vanilla",
        "frame_avg_angles": P.ABSENT,
        "frame_avg_max_fwd_samples": P.ABSENT}


@pytest.mark.parametrize("arm,cfg_name", [
    ("BFC", "worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/FLAC_AR_BFC.json"),
    ("BFre", "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json"),
    ("P1re", "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json"),
])
def test_each_arms_own_config_satisfies_its_contract(P, arm, cfg_name):
    """The contracts are not a second opinion: each arm's real training config
    — the file its run was launched with — must satisfy its own expectation."""
    cfg = json.loads((REPO / cfg_name).read_text())
    assert P.check_embedded_training(cfg, arm) == []


def test_an_arms_contract_rejects_the_other_arms_configs(P):
    """Mutation resistance across arms: if the expectations were vacuous, every
    config would satisfy every arm."""
    configs = {
        "BFC": json.loads((REPO / "worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude"
                           / "FLAC_AR_BFC.json").read_text()),
        "BFre": json.loads((REPO / "worklog/worklog_yixun/exp_07_fa_scratch_claude"
                            / "FLAC_AR_BF.json").read_text()),
        "P1re": json.loads((REPO / "worklog/worklog_yixun/exp_07_fa_scratch_claude"
                            / "FLAC_AR_BVp1.json").read_text()),
    }
    for arm in ("BFC", "BFre", "P1re"):
        for other, cfg in configs.items():
            reasons = P.check_embedded_training(cfg, arm)
            assert (reasons == []) == (other == arm), (arm, other, reasons)


@pytest.mark.parametrize("payload", [None, {}, [], "cfg", {"training": None},
                                     {"training": []}])
def test_a_checkpoint_with_no_usable_config_fails_every_arm(P, payload):
    for arm in ("BFC", "BFre", "P1re"):
        assert P.check_embedded_training(payload, arm)


def test_a_cap_key_that_should_not_exist_is_refused(P):
    """ABSENT means absent. A B-F checkpoint declaring a training cap is not the
    historical artifact this row publishes."""
    cfg = {"training": {"cond_method": "fa_invariant",
                        "frame_avg_angles": [0.0, 90.0, 180.0, 270.0],
                        "frame_avg_max_fwd_samples": 32}}
    reasons = P.check_embedded_training(cfg, "BFre")
    assert any("frame_avg_max_fwd_samples" in r for r in reasons), reasons


def test_type_strictness(P):
    """``1`` is not ``1.0`` — the distinction the factory itself enforces."""
    cfg = {"training": {"cond_method": "fa_cartesian",
                        "frame_avg_angles": [0, 90, 180, 270],
                        "frame_avg_max_fwd_samples": 32}}
    assert any("frame_avg_angles" in r for r in P.check_embedded_training(cfg, "BFC"))


def test_arm_profiles_come_from_the_same_registry_as_the_commands(P):
    for arm in P.ARM_ORDER:
        prof = P.arm_profile(arm)
        assert prof["cond_method"] == P.ARMS[arm]["cond_method"]
        assert prof["eval_prefix"] == P.ARMS[arm]["eval_prefix"]
        if arm == "P1re":
            assert prof["frame_avg_angles"] is None
            assert prof["frame_avg_fwd_cap"] is None
            assert prof["orbit_execution"] == "n/a"
        else:
            assert prof["frame_avg_angles"] == [0.0, 90.0, 180.0, 270.0]
            assert prof["frame_avg_fwd_cap"] == 64
            assert prof["orbit_execution"] == "batched"


# --------------------------------------------------------------------------- #
# 4. the driver
# --------------------------------------------------------------------------- #
def _run(env_extra, args=()):
    env = dict(os.environ)
    env.update(env_extra)
    return subprocess.run(["bash", str(DRIVER), *args], cwd=str(REPO), env=env,
                          capture_output=True, text=True)


def test_the_driver_parses():
    assert subprocess.run(["bash", "-n", str(DRIVER)], capture_output=True).returncode == 0


def test_dry_run_lists_the_whole_campaign_and_nothing_else():
    out = _run({"DRY_RUN": "1", "PLACEHOLDER": PH})
    assert out.returncode == 0, out.stderr
    lines = [l for l in out.stdout.splitlines() if l.strip()]
    assert len(lines) == 34
    assert all(l.startswith("python eval_FLAC.py ") for l in lines)
    assert len({l for l in lines}) == 34


def test_dry_run_matches_the_protocol_module_exactly(P):
    """The driver RESTATES NO FLAG — it prints what the module builds."""
    out = _run({"DRY_RUN": "1", "PLACEHOLDER": PH})
    lines = [l for l in out.stdout.splitlines() if l.strip()]
    want = [" ".join(P.command(c.arm, c.k, c.seed, c.rotate_deg, placeholder=PH))
            for c in P.inventory()]
    assert lines == want


@pytest.mark.parametrize("arm,count", [("BFC", 14), ("BFre", 10), ("P1re", 10)])
def test_dry_run_can_be_restricted_to_one_arm(arm, count):
    out = _run({"DRY_RUN": "1", "PLACEHOLDER": PH, "ARM": arm})
    assert out.returncode == 0, out.stderr
    lines = [l for l in out.stdout.splitlines() if l.strip()]
    assert len(lines) == count
    assert all(f"--eval-name exp21_{arm}_" in l for l in lines)


@pytest.mark.parametrize("env", [{"DRY_RUN": "2"}, {"DRY_RUN": "yes"},
                                 {"DRY_RUN": "1", "ARM": "NOPE"},
                                 {"DRY_RUN": "1", "ARM": "bfc"}])
def test_a_malformed_mode_fails_closed(env):
    """Never a silent fallback to the default mode: the modes are the only thing
    standing between a dry inventory and 34 GPU-hours."""
    out = _run(env)
    assert out.returncode == 2, out.stdout


def test_the_dry_run_writes_nothing(tmp_path):
    """A dry run must be safe to execute anywhere, at any time — including while
    the arm is still training."""
    before = {p: p.stat().st_mtime_ns for p in EXPDIR.iterdir() if p.is_file()}
    out = _run({"DRY_RUN": "1", "PLACEHOLDER": PH})
    assert out.returncode == 0
    after = {p: p.stat().st_mtime_ns for p in EXPDIR.iterdir() if p.is_file()}
    assert before == after


# --------------------------------------------------------------------------- #
# 5. the table: comparator rows and the cross-arm transaction
# --------------------------------------------------------------------------- #
COMPARATOR_LABELS = {
    "BFre": "B-F @40k re-eval at the exp_21 pin (D6 paired comparator)",
    "P1re": "P1 @40k re-eval at the exp_21 pin (D6 paired comparator)",
}


def test_the_comparator_rows_are_registered_as_two_K_pairs(gen):
    specs = [s for s in gen.ROWS if len(s) > 4 and s[4] == "exp21c"]
    assert len(specs) == 4
    assert {s[2] for s in specs} == {1, 8}
    assert {s[0] for s in specs} == set(COMPARATOR_LABELS.values())
    protos = {s[0]: s[1] for s in specs}
    assert protos[COMPARATOR_LABELS["BFre"]] == "fa eval (repin)"
    assert protos[COMPARATOR_LABELS["P1re"]] == "vanilla eval (repin)"


def test_the_row_labels_agree_with_the_protocol_module(gen, P):
    for arm, label in COMPARATOR_LABELS.items():
        assert P.ARMS[arm]["row_label"] == label
        assert gen.EXP21C_ARM_OF_LABEL[label] == arm


def test_the_comparator_globs_match_the_filenames_the_driver_produces(gen, P):
    """A row whose glob matches nothing renders pending forever, which is
    indistinguishable from an evaluation that never ran."""
    import fnmatch
    specs = {(s[0], s[2]): s[3] for s in gen.ROWS if len(s) > 4 and s[4] == "exp21c"}
    for cell in P.inventory():
        if cell.arm == "BFC":
            continue
        rel = os.path.relpath(P.metrics_path(cell.arm, cell.k, cell.seed),
                              P.repo_root())
        pats = specs[(COMPARATOR_LABELS[cell.arm], cell.k)]
        assert any(fnmatch.fnmatch(rel, p) for p in pats), (rel, pats)


def test_the_comparator_globs_do_not_catch_the_stream_sidecar(gen, P):
    import fnmatch
    specs = {(s[0], s[2]): s[3] for s in gen.ROWS if len(s) > 4 and s[4] == "exp21c"}
    for cell in P.inventory():
        if cell.arm == "BFC":
            continue
        rel = os.path.relpath(P.metrics_path(cell.arm, cell.k, cell.seed),
                              P.repo_root())
        sidecar = rel[: -len(".json")] + ".stream.json"
        pats = specs[(COMPARATOR_LABELS[cell.arm], cell.k)]
        assert not any(fnmatch.fnmatch(sidecar, p) for p in pats), sidecar


def test_no_historical_row_absorbs_a_comparator_cell(gen, P):
    """The comparator artifacts land INSIDE exp_07's directories, where the
    historical B-F and P1 rows' globs also look. If one of those globs matched a
    new cell, a legacy row would silently average this round's numbers into
    itself — and this round must not touch that history at all."""
    import fnmatch
    historical = [s for s in gen.ROWS
                  if (len(s) <= 4 or s[4] not in ("exp21", "exp21c"))]
    for cell in P.inventory():
        if cell.arm == "BFC":
            continue
        rel = os.path.relpath(P.metrics_path(cell.arm, cell.k, cell.seed),
                              P.repo_root())
        for spec in historical:
            for pat in spec[3]:
                assert not fnmatch.fnmatch(rel, pat), (rel, spec[0], pat)


class TestCrossArmTransaction:
    def _cells(self, tmp_path, pins):
        """One landed cell per arm, each at the pin ``pins[arm]``."""
        cells, status = {}, {}
        labels = {"BFC": "BFC C4-Cartesian FA @40k (exp_21)", **COMPARATOR_LABELS}
        for arm, pin in pins.items():
            path = tmp_path / f"{arm}.json"
            path.write_text(json.dumps({"source_sha": pin}))
            cells[(labels[arm], 8)] = [str(path)]
            status[(labels[arm], 8)] = True
        return cells, status

    def test_nothing_landed_is_pending_not_a_failed_transaction(self, gen):
        withheld, problems, notes = gen.check_exp21_cross_arm({}, {})
        assert (withheld, problems, notes) == (set(), [], [])

    def test_one_pin_across_the_three_arms_publishes_and_says_so(self, gen, tmp_path):
        cells, status = self._cells(tmp_path, {"BFC": "a" * 40, "BFre": "a" * 40,
                                               "P1re": "a" * 40})
        withheld, problems, notes = gen.check_exp21_cross_arm(cells, status)
        assert withheld == set() and problems == []
        assert len(notes) == 1
        note = notes[0]
        assert "paired" in note and ("a" * 12) in note
        # ...and the note is where the historical rows are declared contextual,
        # since their specs are NOT edited by this round
        assert "CONTEXTUAL ONLY" in note
        assert "fa scratch B-F @40k" in note and "P1 vanilla @40k" in note

    def test_a_comparator_at_another_pin_withholds_every_arm(self, gen, tmp_path):
        """This is the reading D6 exists to make safe: a cross-pin subtraction
        would fold the evaluator shift into the arm's effect."""
        cells, status = self._cells(tmp_path, {"BFC": "a" * 40, "BFre": "b" * 40,
                                               "P1re": "a" * 40})
        withheld, problems, notes = gen.check_exp21_cross_arm(cells, status)
        assert problems and "MORE THAN ONE evaluator pin" in problems[0]
        assert withheld == set(cells)
        assert notes == []

    def test_a_blocked_cell_does_not_contribute_a_pin(self, gen, tmp_path):
        """A cell its own row gate already refused is not evidence of anything,
        so it must not be able to fail the pin check a second time (or, worse,
        to satisfy it)."""
        cells, status = self._cells(tmp_path, {"BFC": "a" * 40, "BFre": "b" * 40})
        status[(COMPARATOR_LABELS["BFre"], 8)] = False
        withheld, problems, notes = gen.check_exp21_cross_arm(cells, status)
        assert problems == [] and withheld == set()


def test_the_repin_rows_are_labelled_batched_never_legacy_loop(gen):
    """The re-evaluations run the BATCHED orbit executor — they are produced by
    THIS evaluator. Their cells live under outputs_FLAC/exp07_*/ because
    eval_FLAC writes beside the checkpoint it reads, so a namespace test keyed on
    an output DIRECTORY would miss them and the legacy-loop migration would
    append a false execution label to a row nobody touched."""
    specs = [s for s in gen.ROWS if len(s) > 4 and s[4] == "exp21c"]
    for label, proto, _k, pats, _c in specs:
        assert gen.is_batched_orbit_row(pats) is True, label
        for evidence_ready in (False, True):
            got = gen.protocol_label(proto, True, evidence_ready)
            assert "legacy-loop" not in got, (label, got)
        # ...and they are not claimed by another experiment's validator
        assert gen.is_exp11_row(pats) is False
        assert gen.is_exp14_row(pats) is False


def test_the_campaign_namespace_claims_nothing_else(gen):
    """`is_exp21_row` widened to the whole campaign; it must still not sweep in
    any other experiment's row."""
    for spec in gen.ROWS:
        contract = spec[4] if len(spec) > 4 else "table"
        assert gen.is_exp21_row(spec[3]) == (contract in ("exp21", "exp21c")), spec[0]
