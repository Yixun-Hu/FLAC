"""exp_21 (bf_fa_cartesian) — the model-comparison TABLE staging (plan §3g).

Two row specs (K=1, K=8) for BFC@40k under the ``fa_cartesian eval`` protocol,
plus the admission validator that decides whether those rows may carry numbers.

Where these tests live and why: every experiment that added rows to
``gen_model_comparison.py`` staged its tests in its OWN file — exp_15's in
``test_yaw_aug_collect.py``, exp_17's in ``test_exp17_rotation_table.py`` —
while ``test_gen_model_comparison_gate.py`` stayed the generator's own
regression file. This mirrors that convention rather than growing the shared
file.

The contract being pinned (plan §3g, r2 plan-review blocker 4): a fa_cartesian
row publishes only when its five cells ARE the registered evaluation — the exact
dataset config for that K, the full 6,337-item split, seeds {42..46} with no
duplicate, EMA weights, bf16 conditioning autocast, batch 64, cfg 1.0, 1 step,
unrotated, ``cond_method`` fa_cartesian over the C4 orbit at eval cap 64, the
step-40000 checkpoint, and ONE evaluator pin across the cell. Anything else
renders BLOCKED, never as a number: an unprovable row is exactly the failure
announcement 05 exists to prevent, and the numbers here would sit beside B-F's
and P1's in the paper's comparison table.

Nothing under ``outputs_FLAC/`` needs to exist: an empty cell is *pending*, which
is how the generator has always rendered a registered-but-unlanded row, and both
BFC rows are registered in advance exactly like exp_10's endpoint rows and
exp_15's YAWAUG pair.
"""
import importlib.util
import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GEN_PATH = REPO / "worklog" / "worklog_yixun" / "gen_model_comparison.py"
VAL_PATH = (REPO / "worklog" / "worklog_yixun" / "exp_21_bf_fa_cartesian_claude"
            / "exp21_validate_cell.py")

LABEL = "BFC C4-Cartesian FA @40k (exp_21)"
CONTRACT = "exp21"
PROTO = "fa_cartesian eval"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen():
    return _load(GEN_PATH, "gen_model_comparison")


@pytest.fixture(scope="module")
def V():
    return _load(VAL_PATH, "exp21_validate_cell")


# --------------------------------------------------------------------------- #
# fixtures: a registered cell, exactly as eval_FLAC.py writes it
# --------------------------------------------------------------------------- #
METRICS = {"T60": 8.5, "C50": 0.97, "EDT": 37.0,
           "RIR_to_GT_RIR_R@1": 5.4, "RIR_to_GT_RIR_R@5": 16.1,
           "RIR_to_GT_RIR_R@10": 23.3}
DATASET = {8: "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json",
           1: "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json"}
CKPT = ("outputs_FLAC/exp21_BFC/FLAC_exp21_BFC/exp21_BFC/checkpoints/"
        "epoch=19-step=40000.ckpt")
PIN = "c" * 40


def record(k=8, seed=42, **over):
    """The metrics record ``build_metrics_record`` writes for a registered cell.

    Key order and field names follow eval_FLAC.build_metrics_record; the values
    are the plan §5 template's."""
    rec = {
        "metrics": dict(METRICS),
        "ckpt_path": CKPT,
        "rotate_deg": 0.0,
        "cond_method": "fa_cartesian",
        "frame_avg_angles": [0.0, 90.0, 180.0, 270.0],
        "cond_autocast": "bf16",
        "orbit_execution": "batched",
        "frame_avg_fwd_cap": 64,
        "source_sha": PIN,
        "batch_size": 64,
        "n_samples": 6337,
        "dataset_config": DATASET[k],
        "seed": seed,
        "cfg_scale": 1.0,
        "steps": 1,
        "eval_name": f"exp21_BFC_S40000_K{k}_s{seed}",
        "weights_source": "ema",
        "device": "cuda",
    }
    rec.update(over)
    return rec


def write_cell(dirpath, k=8, seed=42, **over):
    """One per-seed JSON at the path the row glob expects."""
    os.makedirs(dirpath, exist_ok=True)
    rec = record(k, seed, **over)
    name = (f"epoch=19-step=40000_metrics_1_1.0_"
            f"exp21_BFC_S40000_K{k}_s{seed}_fa_cartesian_a4.json")
    path = os.path.join(dirpath, name)
    with open(path, "w") as fh:
        json.dump(rec, fh)
    return path


def full_cell(tmp_path, k=8, **over):
    d = tmp_path / f"K{k}"
    return [write_cell(str(d), k=k, seed=s, **over) for s in (42, 43, 44, 45, 46)]


# --------------------------------------------------------------------------- #
# 1. the row specs — registered, additive, correctly labelled
# --------------------------------------------------------------------------- #
class TestRowSpecs:
    def _exp21_specs(self, gen):
        return [s for s in gen.ROWS if len(s) > 4 and s[4] == CONTRACT]

    def test_both_K_rows_are_registered_under_the_exp21_contract(self, gen):
        specs = self._exp21_specs(gen)
        assert len(specs) == 2
        assert {s[2] for s in specs} == {1, 8}
        assert {s[0] for s in specs} == {LABEL}
        assert {s[1] for s in specs} == {PROTO}

    def test_the_registration_is_additive(self, gen):
        """Every earlier experiment's specs are still there, unmoved: this file
        has concurrent writers on other machines, so an edit that reorders or
        drops a spec is a merge hazard as much as a correctness one."""
        labels = [r[0] for r in gen.ROWS]
        for expected in ("exp_11 baseline", "exp_14 Z", "exp_15", "exp_17",
                         "released FLAC_EMA (exp_01 repro)"):
            assert any(expected in l for l in labels), expected
        # ...and the exp_21 rows are at the END, appended, not interleaved
        assert [len(s) > 4 and s[4] == CONTRACT for s in gen.ROWS][-2:] == [True, True]

    def test_the_globs_find_the_registered_filenames(self, gen, tmp_path):
        """The spec's glob and the name eval_FLAC actually writes must agree —
        a row that silently matches nothing renders *pending* forever, which is
        indistinguishable from an evaluation that never ran."""
        import glob as globmod
        ckpts = (tmp_path / "outputs_FLAC" / "exp21_BFC" / "FLAC_exp21_BFC"
                 / "exp21_BFC" / "checkpoints")
        for k in (1, 8):
            for seed in (42, 43, 44, 45, 46):
                write_cell(str(ckpts), k=k, seed=seed)
        for spec in self._exp21_specs(gen):
            k, pats = spec[2], spec[3]
            found = sorted(set(sum(
                (globmod.glob(os.path.join(str(tmp_path), p), recursive=True)
                 for p in pats), [])))
            assert len(found) == 5, (k, found)
            assert all(f"_K{k}_s" in os.path.basename(f) for f in found)

    def test_the_glob_does_not_catch_the_stream_sidecar(self, gen, tmp_path):
        """``--record-stream`` (the §5 invariance grid) writes a
        ``<stem>.stream.json`` beside the metrics file. A glob that also matched
        it would hand the validator ten files for a five-seed cell — the exact
        bug the exp_11 comment warns about."""
        import glob as globmod
        ckpts = (tmp_path / "outputs_FLAC" / "exp21_BFC" / "run" / "checkpoints")
        for seed in (42, 43, 44, 45, 46):
            path = write_cell(str(ckpts), k=8, seed=seed)
            Path(path[: -len(".json")] + ".stream.json").write_text("{}")
            Path(path + ".screenmeta.json").write_text("{}")
        spec = next(s for s in self._exp21_specs(gen) if s[2] == 8)
        found = sorted(set(sum(
            (globmod.glob(os.path.join(str(tmp_path), p), recursive=True)
             for p in spec[3]), [])))
        assert len(found) == 5, found

    def test_the_row_is_labelled_batched_never_legacy_loop(self, gen):
        """The orbit-execution label. BFC runs the BATCHED executor, and the
        generator appends ``(legacy-loop)`` to any ``fa`` row it does not
        recognise as batched once the exp_10 evidence returns — which would put a
        false execution label on this row without touching it."""
        spec = next(s for s in self._exp21_specs(gen) if s[2] == 8)
        pats = spec[3]
        assert gen.is_batched_orbit_row(pats) is True
        for evidence_ready in (False, True):
            got = gen.protocol_label(PROTO, gen.is_batched_orbit_row(pats), evidence_ready)
            assert got == "fa_cartesian eval (batched)", (evidence_ready, got)
        # ...and it is NOT claimed by another experiment's validator
        assert gen.is_exp11_row(pats) is False
        assert gen.is_exp14_row(pats) is False


# --------------------------------------------------------------------------- #
# 2. the admission validator — one test per registered requirement
# --------------------------------------------------------------------------- #
class TestAdmission:
    def test_a_registered_five_seed_cell_validates(self, gen, tmp_path):
        for k in (1, 8):
            ok, problems = gen.validate_exp21_cell(full_cell(tmp_path, k=k), expected_k=k)
            assert ok, problems

    def test_an_empty_cell_is_pending_not_a_failure(self, gen):
        """A registered-but-unlanded row renders *pending*; the generator has
        always treated an absent cell that way and this row is registered in
        advance."""
        ok, problems = gen.validate_exp21_cell([], expected_k=8)
        assert ok and problems == []
        line, blocked = gen.render_row(LABEL, PROTO, 8, [], contract=CONTRACT)
        assert blocked is False
        assert "pending (0/5 seeds on disk)" in line

    @pytest.mark.parametrize("over,fragment", [
        ({"cond_method": "fa_invariant"}, "cond_method"),
        ({"cond_method": "vanilla"}, "cond_method"),
        ({"frame_avg_angles": [0.0, 180.0]}, "frame_avg_angles"),
        ({"frame_avg_angles": None}, "frame_avg_angles"),
        ({"frame_avg_fwd_cap": 32}, "frame_avg_fwd_cap"),
        ({"frame_avg_fwd_cap": None}, "frame_avg_fwd_cap"),
        ({"orbit_execution": "loop"}, "orbit_execution"),
        ({"cond_autocast": "default"}, "cond_autocast"),
        ({"batch_size": 32}, "batch_size"),
        ({"n_samples": 6336}, "n_samples"),
        ({"cfg_scale": 3.0}, "cfg_scale"),
        ({"steps": 4}, "steps"),
        ({"rotate_deg": 45.0}, "rotate_deg"),
        ({"weights_source": "online"}, "weights_source"),
        ({"dataset_config": "src/configs/dataset_configs/AR/eval/"
                            "acousticroom_seeneval.json"}, "dataset_config"),
        ({"ckpt_path": "outputs_FLAC/exp21_BFC/x/epoch=18-step=37500.ckpt"}, "step=40000"),
    ])
    def test_a_protocol_deviation_blocks_the_row(self, gen, tmp_path, over, fragment):
        """Each of these is a run that would produce plausible numbers under a
        different protocol — the announcement-05 failure mode. The row must
        refuse rather than publish them beside B-F's and P1's."""
        files = full_cell(tmp_path, k=8)
        rec = json.load(open(files[0]))
        rec.update(over)
        json.dump(rec, open(files[0], "w"))
        ok, problems = gen.validate_exp21_cell(files, expected_k=8)
        assert ok is False
        assert any(fragment in p for p in problems), (fragment, problems)

    def test_a_randomly_rotated_cell_is_refused(self, gen, tmp_path):
        """exp_14-style random rotation nulls ``rotate_deg`` and adds
        ``rotate_mode``. The registered row is the FIXED, unrotated cell, and a
        null angle must not read as 'not 45 degrees, therefore fine'."""
        files = full_cell(tmp_path, k=8)
        rec = json.load(open(files[0]))
        rec.update({"rotate_deg": None, "rotate_mode": "random", "rotate_seed": 42})
        json.dump(rec, open(files[0], "w"))
        ok, problems = gen.validate_exp21_cell(files, expected_k=8)
        assert ok is False
        assert any("rotate" in p for p in problems), problems

    def test_the_K_in_the_evidence_must_be_the_K_the_row_declares(self, gen, tmp_path):
        """A renamed payload is not a re-measurement (exp_14 round-3 review B2):
        K=1 evidence under the K=8 row would publish one context size's numbers
        under another's label."""
        ok, problems = gen.validate_exp21_cell(full_cell(tmp_path, k=1), expected_k=8)
        assert ok is False
        assert any("K" in p for p in problems), problems

    def test_the_dataset_config_must_be_the_one_registered_for_that_K(self, gen, tmp_path):
        """K is carried by the eval NAME; the split that actually ran is carried
        by the dataset config. Both must agree, or a K=8 row can be produced from
        the K=1 split."""
        files = full_cell(tmp_path, k=8)
        rec = json.load(open(files[0]))
        rec["dataset_config"] = DATASET[1]
        json.dump(rec, open(files[0], "w"))
        ok, problems = gen.validate_exp21_cell(files, expected_k=8)
        assert ok is False
        assert any("dataset_config" in p for p in problems), problems

    def test_an_absolute_dataset_path_into_this_checkout_is_accepted(self, gen, tmp_path):
        """Operators paste absolute paths. The registered split is an identity,
        not a spelling, so a path that ends in the registered relative path is
        the same split."""
        files = full_cell(tmp_path, k=8)
        rec = json.load(open(files[0]))
        rec["dataset_config"] = str(REPO / DATASET[8])
        json.dump(rec, open(files[0], "w"))
        ok, problems = gen.validate_exp21_cell(files, expected_k=8)
        assert ok, problems

    def test_four_seeds_are_not_a_row(self, gen, tmp_path):
        files = full_cell(tmp_path, k=8)[:4]
        ok, problems = gen.validate_exp21_cell(files, expected_k=8)
        assert ok is False
        assert any("seed" in p for p in problems), problems

    def test_a_duplicated_seed_is_not_five_seeds(self, gen, tmp_path):
        """Five files with a duplicate would otherwise be averaged as five
        independent draws (exp_14's counting bug)."""
        files = full_cell(tmp_path, k=8)
        files = files[:4] + [files[0]]
        ok, problems = gen.validate_exp21_cell(files, expected_k=8)
        assert ok is False
        assert any("seed" in p for p in problems), problems

    def test_a_foreign_seed_is_refused(self, gen, tmp_path):
        files = full_cell(tmp_path, k=8)[:4]
        files.append(write_cell(str(tmp_path / "K8"), k=8, seed=47))
        ok, problems = gen.validate_exp21_cell(files, expected_k=8)
        assert ok is False
        assert any("seed" in p for p in problems), problems

    def test_the_cell_must_share_one_evaluator_pin(self, gen, tmp_path):
        """Five cells measured at different commits are not a row; they are five
        measurements wearing one label."""
        files = full_cell(tmp_path, k=8)
        rec = json.load(open(files[0]))
        rec["source_sha"] = "d" * 40
        json.dump(rec, open(files[0], "w"))
        ok, problems = gen.validate_exp21_cell(files, expected_k=8)
        assert ok is False
        assert any("pin" in p or "source_sha" in p for p in problems), problems

    def test_an_unknown_evaluator_pin_is_refused(self, gen, tmp_path):
        """``source_sha`` falls back to the string 'unknown' when git is
        unavailable. A row whose provenance is literally unknown cannot be
        published as a measured one."""
        files = full_cell(tmp_path, k=8, source_sha="unknown")
        ok, problems = gen.validate_exp21_cell(files, expected_k=8)
        assert ok is False
        assert any("unknown" in p for p in problems), problems

    @pytest.mark.parametrize("payload,fragment", [
        ({"T60": float("nan")}, "finite"),
        ({"T60": float("inf")}, "finite"),
        ({"T60": None}, "finite"),
    ])
    def test_a_non_finite_metric_blocks_the_row(self, gen, tmp_path, payload, fragment):
        """A cell can satisfy every provenance rule and still carry a NaN, which
        would propagate into a published mean. It is checked HERE rather than in
        agg_files so the row renders BLOCKED instead of raising mid-regeneration
        and taking every other experiment's row down with it."""
        files = full_cell(tmp_path, k=8)
        rec = json.load(open(files[0]))
        rec["metrics"].update(payload)
        json.dump(rec, open(files[0], "w"))
        ok, problems = gen.validate_exp21_cell(files, expected_k=8)
        assert ok is False
        assert any(fragment in p for p in problems), problems

    def test_a_missing_metric_blocks_the_row(self, gen, tmp_path):
        files = full_cell(tmp_path, k=8)
        rec = json.load(open(files[0]))
        rec["metrics"].pop("RIR_to_GT_RIR_R@10")
        json.dump(rec, open(files[0], "w"))
        ok, problems = gen.validate_exp21_cell(files, expected_k=8)
        assert ok is False
        assert any("R@10" in p for p in problems), problems

    def test_an_unreadable_cell_fails_closed(self, gen, tmp_path):
        files = full_cell(tmp_path, k=8)
        Path(files[0]).write_text("{not json")
        ok, problems = gen.validate_exp21_cell(files, expected_k=8)
        assert ok is False
        assert any("unreadable" in p for p in problems), problems

    def test_an_unparseable_eval_name_fails_closed(self, gen, tmp_path):
        files = full_cell(tmp_path, k=8)
        rec = json.load(open(files[0]))
        rec["eval_name"] = "exp21_BFC_whatever"
        json.dump(rec, open(files[0], "w"))
        ok, problems = gen.validate_exp21_cell(files, expected_k=8)
        assert ok is False
        assert any("eval_name" in p for p in problems), problems

    def test_a_rotation_grid_cell_cannot_stand_in_for_the_table_cell(self, gen, tmp_path):
        """§5's invariance grid writes ``..._s42_rot45`` cells of the SAME
        checkpoint. They are a negative control, never a model row."""
        files = full_cell(tmp_path, k=8)
        rec = json.load(open(files[0]))
        rec["eval_name"] = "exp21_BFC_S40000_K8_s42_rot45"
        rec["rotate_deg"] = 45.0
        json.dump(rec, open(files[0], "w"))
        ok, problems = gen.validate_exp21_cell(files, expected_k=8)
        assert ok is False


# --------------------------------------------------------------------------- #
# 3. rendering + the two-K transaction
# --------------------------------------------------------------------------- #
class TestRendering:
    def test_a_valid_cell_renders_numbers(self, gen, tmp_path):
        line, blocked = gen.render_row(LABEL, PROTO, 8, full_cell(tmp_path, k=8),
                                       contract=CONTRACT)
        assert blocked is False
        assert "8.500" in line and "0.9700" in line       # T60 and the 4-dp C50

    def test_an_invalid_cell_renders_BLOCKED_never_numbers(self, gen, tmp_path):
        files = full_cell(tmp_path, k=8)
        rec = json.load(open(files[0]))
        rec["cond_autocast"] = "default"
        json.dump(rec, open(files[0], "w"))
        line, blocked = gen.render_row(LABEL, PROTO, 8, files, contract=CONTRACT)
        assert blocked is True
        assert "BLOCKED" in line
        assert "8.500" not in line

    def test_the_flat_split_metrics_are_what_publishes(self, gen, tmp_path):
        """The comparator estimand is the FLAT split-level metric (plan §5): B-F
        and P1 rows are flat, so BFC must be too. A ``by_scene`` block may ride
        along (the §5 template passes --record-per-scene) without changing what
        this row prints."""
        files = full_cell(tmp_path, k=8,
                          by_scene={"Office": {"T60": 99.0}}, per_scene_schema=1)
        values, n = gen.agg_files(files)
        assert n == 5
        assert values["T60"][0] == pytest.approx(METRICS["T60"])

    def test_no_evidence_at_all_is_not_a_withheld_transaction(self, gen):
        """Both rows registered in advance and nothing landed yet: that is
        *pending*, not a failed transaction. The gate must stay silent until the
        first cell exists, exactly like the Q9 and exp_14 gates."""
        assert gen.check_exp21_round({(LABEL, 1): [], (LABEL, 8): []},
                                     {(LABEL, 1): False, (LABEL, 8): False}) == []

    def test_one_K_alone_is_withheld(self, gen, tmp_path):
        """A lone K=8 row invites a K=8-only comparison against rows that carry
        both."""
        cells = {(LABEL, 8): full_cell(tmp_path, k=8), (LABEL, 1): []}
        status = {(LABEL, 8): True, (LABEL, 1): False}
        problems = gen.check_exp21_round(cells, status)
        assert problems
        assert any("K=1" in p for p in problems), problems

    def test_a_present_but_invalid_partner_withholds_the_pair(self, gen, tmp_path):
        """Counting files is not validating them (exp_14 round-3 closure A4/B2):
        a five-file K=1 block that did not validate must not let K=8 publish
        beside its refusal."""
        cells = {(LABEL, 8): full_cell(tmp_path, k=8),
                 (LABEL, 1): full_cell(tmp_path, k=1)}
        status = {(LABEL, 8): True, (LABEL, 1): False}
        problems = gen.check_exp21_round(cells, status)
        assert problems
        assert any("did not validate" in p for p in problems), problems

    def test_the_two_K_rows_must_share_one_evaluator_pin(self, gen, tmp_path):
        """Per-cell validation proves each K internally; it cannot see that the
        two blocks were measured at different commits. The paired K=1/K=8 reading
        would then carry whatever moved between those pins."""
        cells = {(LABEL, 8): full_cell(tmp_path, k=8),
                 (LABEL, 1): full_cell(tmp_path, k=1, source_sha="e" * 40)}
        status = {(LABEL, 8): True, (LABEL, 1): True}
        problems = gen.check_exp21_round(cells, status)
        assert any("pin" in p for p in problems), problems

    def test_both_K_valid_publishes(self, gen, tmp_path):
        cells = {(LABEL, 8): full_cell(tmp_path, k=8),
                 (LABEL, 1): full_cell(tmp_path, k=1)}
        status = {(LABEL, 8): True, (LABEL, 1): True}
        assert gen.check_exp21_round(cells, status) == []

    def test_other_experiments_are_unaffected(self, gen):
        """This gate is exp_21-scoped: another machine regenerating the table
        must not be blocked by a contract that says nothing about its rows."""
        assert gen.check_exp21_round({}, {}) == []


# --------------------------------------------------------------------------- #
# 4. the validator module itself (reused by the eval driver in a later round)
# --------------------------------------------------------------------------- #
class TestValidatorModule:
    def test_the_registered_protocol_is_stated_once(self, V):
        assert V.SEEDS == (42, 43, 44, 45, 46)
        assert V.STEP == 40000
        assert V.COND_METHOD == "fa_cartesian"
        assert V.FRAME_AVG_ANGLES == [0.0, 90.0, 180.0, 270.0]
        assert V.FRAME_AVG_FWD_CAP == 64
        assert V.EXPECTED_COUNT == 6337
        assert V.BATCH_SIZE == 64
        assert V.COND_AUTOCAST == "bf16"
        assert V.WEIGHTS_SOURCE == "ema"
        assert V.DATASET_CONFIG == DATASET

    def test_eval_name_round_trips(self, V):
        for k in (1, 8):
            for seed in V.SEEDS:
                cell = V.parse_eval_name(V.eval_name(k, seed))
                assert (cell.step, cell.k, cell.seed) == (40000, k, seed)

    def test_the_eval_name_matches_the_plan_template(self, V):
        assert V.eval_name(8, 42) == "exp21_BFC_S40000_K8_s42"
        assert V.eval_name(1, 46) == "exp21_BFC_S40000_K1_s46"

    @pytest.mark.parametrize("name", [
        "exp21_BFC_S40000_K8_s42_rot45",     # the invariance grid, not a row
        "exp21_BFC_S37500_K8_s42",           # a band-context screen
        "exp21_BFC_K8_s42",
        "exp11_C4L_q9_S40000_s42_K8",        # another experiment entirely
        "",
    ])
    def test_a_foreign_eval_name_is_refused(self, V, name):
        with pytest.raises(ValueError):
            V.parse_eval_name(name)

    def test_validate_metrics_record_accepts_the_registered_record(self, V):
        cell = V.parse_eval_name("exp21_BFC_S40000_K8_s42")
        assert V.validate_metrics_record(record(8, 42), cell) == []

    def test_validate_metrics_record_names_the_field_it_refuses(self, V):
        cell = V.parse_eval_name("exp21_BFC_S40000_K8_s42")
        reasons = V.validate_metrics_record(record(8, 42, batch_size=16), cell)
        assert len(reasons) == 1
        assert "batch_size" in reasons[0] and "16" in reasons[0]

    def test_the_record_seed_must_match_its_own_eval_name(self, V):
        """The seed is in two places; disagreeing means the file was renamed."""
        cell = V.parse_eval_name("exp21_BFC_S40000_K8_s42")
        reasons = V.validate_metrics_record(record(8, 42, seed=43), cell)
        assert any("seed" in r for r in reasons), reasons
