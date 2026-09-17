"""End-to-end test of the comparison-table generator over a data-curve import fixture.

exp_14 "data_curve", plan §9 / codex finding M7: the DEFERRED end-to-end test that must
pass before the first data-curve arm is published to
``worklog/worklog_yixun/model_comparison.md``.

The kit that produces the evidence lives in another checkout, and what it can prove there
is only that the row spec it prints literal-parses as one of the generator's own four-field
``ROWS`` entries and carries no forbidden substring. Everything downstream of that string
-- globbing, aggregation, validator routing, protocol labelling, the markdown write -- is
``gen_model_comparison.py``, which lives HERE. So this module drives that authoritative
generator over a fixture repo root holding a complete two-arm import directory and pins:

* five files per K are found and aggregated (``n`` is 5, not 10 and not 6: neither the
  ``MANIFEST.sha256`` the importer writes beside the cells nor the prediction bundle is
  swallowed by the trailing ``*`` of the registered glob);
* the rendered ``mean ± sd`` equal the hand-computed values at the printed precision;
* the cyl rows carry the generator's fa-eval protocol label and the van rows the vanilla
  one, and NEITHER arm is routed to the exp_14 yaw validator or the exp_11 orbit validator
  -- both of which would render the row BLOCKED on an eval name they were never written to
  parse (the exp_09 protocol error's cousin: the row would look refused for the wrong
  reason);
* no row renders BLOCKED, WITHHELD or pending, and the emitted markdown carries the labels.

INJECTION. The generator's row table is the module-level ``ROWS`` constant -- it accepts no
row list or config on its CLI -- so these tests monkeypatch ``G.ROWS`` with the
importer-style specs. The tracked default table is therefore never edited, and an autouse
guard asserts the committed ``model_comparison.md`` is byte-unchanged by every test in this
module (this is a shared checkout).

The five "documented trap" tests at the end record generator behaviour that is NOT what a
reader would assume; each names the importer contract that is the only thing protecting the
published row from it.
"""
import glob as globmod
import hashlib
import importlib.util
import json
import os

import pytest


_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # src/tests/ -> src/ -> repo root
_GEN_PY = os.path.join(_REPO_ROOT, "worklog", "worklog_yixun", "gen_model_comparison.py")
_TRACKED_TABLE = os.path.join(_REPO_ROOT, "worklog", "worklog_yixun", "model_comparison.md")


def _load_generator():
    spec = importlib.util.spec_from_file_location("gen_model_comparison", _GEN_PY)
    assert spec is not None and spec.loader is not None, f"cannot load {_GEN_PY}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # the write lives behind main(), not import
    return mod


G = _load_generator()
DEFAULT_ROWS = list(G.ROWS)               # the tracked table, captured before any patch


# --------------------------------------------------------------------------- #
# the importer's contract, restated (its module lives in the kit checkout)
# --------------------------------------------------------------------------- #
#: printed verbatim by src.tools.data_curve.import_cells.row_spec(arm, "025", K)
IMPORTER_ROW_SPECS = [
    ("data-curve cyl f=25% @40k (exp_14 cyl-dinov3)", "fa eval", 1,
     ["outputs_FLAC/data_curve_import/dc_cyl_f025/*_K1_s4[2-6]*.json"]),
    ("data-curve cyl f=25% @40k (exp_14 cyl-dinov3)", "fa eval", 8,
     ["outputs_FLAC/data_curve_import/dc_cyl_f025/*_K8_s4[2-6]*.json"]),
    ("data-curve van f=25% @40k (exp_14 cyl-dinov3)", "vanilla eval", 1,
     ["outputs_FLAC/data_curve_import/dc_van_f025/*_K1_s4[2-6]*.json"]),
    ("data-curve van f=25% @40k (exp_14 cyl-dinov3)", "vanilla eval", 8,
     ["outputs_FLAC/data_curve_import/dc_van_f025/*_K8_s4[2-6]*.json"]),
]
SEEDS = (42, 43, 44, 45, 46)
KS = (1, 8)
CKPT_STEM = "epoch=8-step=40000"          # names.MAX_STEPS = 40000
ARMS = ("cyl", "van")
#: names.ARM_COND_METHOD, and the eval-name suffix eval_FLAC appends for it with one angle
ARM_COND_METHOD = {"cyl": "fa_invariant", "van": "vanilla"}
ARM_SUFFIX = {"cyl": "_fa_invariant_a1", "van": ""}

#: What the generator PRINTS for each arm's cells. Both strings are observations, not
#: preferences: the cyl arm's importer label is "fa eval", and protocol_label() rewrites it
#: to "fa eval (legacy-loop)" once the exp_10 endpoint evidence is on disk -- which it is in
#: the real checkout, so the second string is the one a published data-curve row will carry.
CYL_LABEL_EVIDENCE_ABSENT = "fa eval"
CYL_LABEL_EVIDENCE_PRESENT = "fa eval (legacy-loop)"
VAN_LABEL = "vanilla eval"

#: Per-seed increment. Five evenly spaced values, so for every metric of every cell
#: mean = base + 1.0 and the SAMPLE sd is exactly
#:     sqrt(((-1)^2 + (-0.5)^2 + 0 + 0.5^2 + 1^2) / 4) = sqrt(0.625) = 0.7905694150...
#: i.e. "0.791" at the table's 3-decimal precision and "0.7906" at C50's 4-decimal one.
SEED_OFFSETS = (0.0, 0.5, 1.0, 1.5, 2.0)
SD_3DP = "0.791"
SD_4DP = "0.7906"

#: (arm, K) -> the per-metric base, i.e. the seed-42 value. Deliberately distinct in every
#: cell so a swapped arm or a swapped K cannot render the expected numbers.
BASES = {
    ("cyl", 1): {"T60": 7.0, "C50": 0.9, "EDT": 30.0, "R@1": 4.0, "R@5": 14.0, "R@10": 22.0},
    ("cyl", 8): {"T60": 8.0, "C50": 1.0, "EDT": 31.0, "R@1": 5.0, "R@5": 15.0, "R@10": 23.0},
    ("van", 1): {"T60": 11.0, "C50": 2.0, "EDT": 40.0, "R@1": 1.0, "R@5": 6.0, "R@10": 11.0},
    ("van", 8): {"T60": 12.0, "C50": 2.5, "EDT": 41.0, "R@1": 2.0, "R@5": 7.0, "R@10": 12.0},
}
#: The cell text the table must print, computed BY HAND from BASES (mean = base + 1.0) and
#: the closed-form sd above -- never by re-running the generator's own aggregation.
EXPECTED_CELLS = {
    ("cyl", 1): ["8.000 ± 0.791", "1.9000 ± 0.7906", "31.000 ± 0.791",
                 "5.000 ± 0.791", "15.000 ± 0.791", "23.000 ± 0.791"],
    ("cyl", 8): ["9.000 ± 0.791", "2.0000 ± 0.7906", "32.000 ± 0.791",
                 "6.000 ± 0.791", "16.000 ± 0.791", "24.000 ± 0.791"],
    ("van", 1): ["12.000 ± 0.791", "3.0000 ± 0.7906", "41.000 ± 0.791",
                 "2.000 ± 0.791", "7.000 ± 0.791", "12.000 ± 0.791"],
    ("van", 8): ["13.000 ± 0.791", "3.5000 ± 0.7906", "42.000 ± 0.791",
                 "3.000 ± 0.791", "8.000 ± 0.791", "13.000 ± 0.791"],
}


def _label(arm):
    return f"data-curve {arm} f=25% @40k (exp_14 cyl-dinov3)"


def metrics_basename(arm, k, seed):
    """The basename eval_FLAC writes for one data-curve cell (names.metrics_json_path)."""
    return (f"{CKPT_STEM}_metrics_1_1.0_dc_{arm}_f025_K{k}_s{seed}{ARM_SUFFIX[arm]}.json")


def metrics_record(arm, k, seed_index):
    """One cell's metrics JSON, in the schema the data-curve evaluator writes.

    The generator only reads ``metrics``, but the rest of the record is what the importer
    gates on, so the fixture carries it: the four conditioning flags plus ``ckpt_sha256``,
    the digest that binds a record to the checkpoint bytes it was scored from. ``FD`` and
    the geom-retrieval keys are present and unused -- the table prints six of the eleven.
    """
    base, bump = BASES[(arm, k)], SEED_OFFSETS[seed_index]
    return {
        "metrics": {
            "T60": base["T60"] + bump,
            "Invalid T60": 0.0,
            "C50": base["C50"] + bump,
            "EDT": base["EDT"] + bump,
            "FD": 0.3215,
            "RIR_to_GT_RIR_R@1": base["R@1"] + bump,
            "RIR_to_GT_RIR_R@5": base["R@5"] + bump,
            "RIR_to_GT_RIR_R@10": base["R@10"] + bump,
            "RIR_to_geom_R@1": 3.8662,
            "RIR_to_geom_R@5": 13.2397,
            "RIR_to_geom_R@10": 20.1515,
        },
        "ckpt_path": f"checkpoints/exp14_data_curve/dc_{arm}_f025/{CKPT_STEM}.ckpt",
        "rotate_deg": 0.0,
        "cond_method": ARM_COND_METHOD[arm],
        # eval_FLAC records the angles only for the frame-averaged path; --frame-avg-angles
        # is "0" on this campaign, i.e. the identity pass alone.
        "frame_avg_angles": [0.0] if arm == "cyl" else None,
        "cond_autocast": "bf16",
        "ckpt_sha256": "c" * 64,
    }


def _import_dir(root, arm):
    return os.path.join(str(root), "outputs_FLAC", "data_curve_import", f"dc_{arm}_f025")


def build_fixture_root(tmp_path, exp10_evidence, arms=ARMS):
    """A fixture repo root holding a complete two-arm import directory.

    ``exp10_evidence`` plants the two exp_10 endpoint JSONs the generator's deferred
    label migration keys on: with them ON disk (the real checkout's state)
    ``protocol_label`` rewrites every non-batched fa row to ``(legacy-loop)``, and without
    them it leaves the label alone. Both states are exercised, because the string the
    published row will carry is the first one.
    """
    root = tmp_path / "maintree"
    (root / "worklog" / "worklog_yixun").mkdir(parents=True)   # where main() writes
    for arm in arms:
        target = _import_dir(root, arm)
        os.makedirs(target)
        for k in KS:
            for index, seed in enumerate(SEEDS):
                with open(os.path.join(target, metrics_basename(arm, k, seed)), "w") as fh:
                    json.dump(metrics_record(arm, k, index), fh)
        # exactly what import_cells installs beside the ten cells...
        with open(os.path.join(target, "MANIFEST.sha256"), "w") as fh:
            fh.write(f"# exp_14 data-curve import -- dc_{arm}_f025\n")
        # ...and the prediction bundle each cell is proved by, which shares the stem
        with open(os.path.join(target, f"{CKPT_STEM}_predictions_1_1.0_dc_{arm}"
                                       f"_f025_K8_s42{ARM_SUFFIX[arm]}.pt"), "wb") as fh:
            fh.write(b"not a metrics file")
    if exp10_evidence:
        endpoint = os.path.join(str(root), "outputs_FLAC", "exp10_BF")
        os.makedirs(endpoint, exist_ok=True)
        for name in ("x_exp10_BF67_K1_s42.json", "x_exp10_BF67_K8_s43.json"):
            with open(os.path.join(endpoint, name), "w") as fh:
                json.dump(metrics_record("van", 8, 0), fh)
    return root


def written_table(root):
    with open(os.path.join(str(root), "worklog", "worklog_yixun",
                           "model_comparison.md")) as fh:
        return fh.read()


def data_rows(text):
    """``[[cell, ...], ...]`` for every rendered data row of a generated table.

    A LIST, not a dict keyed by (label, K): the tracked table registers several labels
    twice at one K under different protocols (the exp_03/exp_04/exp_06 online-vs-EMA pairs),
    so a (label, K) dict would silently collapse twelve of its rows."""
    rows = []
    for line in text.splitlines():
        if not line.startswith("|") or set(line) <= set("|- "):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 5 or cells[0] == "Model":
            continue
        try:
            int(cells[2]), int(cells[3])
        except ValueError:
            continue                        # header / footnote lines
        rows.append(cells)
    return rows


def find_row(text, label, k):
    """The one rendered row with this label and K."""
    hits = [cells for cells in data_rows(text)
            if cells[0] == label and cells[2] == str(k)]
    assert len(hits) == 1, f"{label!r} K={k}: {len(hits)} rows rendered"
    return hits[0]


def assert_no_refusals(text):
    """No DATA ROW is blocked, withheld or pending -- and no footnote says one was.

    Checked per row, never over the whole document: the static disclosure header explains
    what a **BLOCKED** row is, so a substring search over the file always "finds" one."""
    for cells in data_rows(text):
        rendered = " | ".join(cells)
        for state in ("BLOCKED", "WITHHELD", "pending"):
            assert state not in rendered, f"{state} in {rendered[:120]}"
    assert "BLOCKED by row validation" not in text
    assert "WITHHELD:" not in text


@pytest.fixture(autouse=True)
def tracked_table_is_untouched():
    """This is a shared checkout: no test here may rewrite the published table.

    ``G.ROWS`` is monkeypatched and every run is pointed at a fixture root, so the only way
    the committed file could change is a bug in one of those two -- which is exactly the
    kind of bug that would be discovered by someone else's ``git status``."""
    before = hashlib.sha256(open(_TRACKED_TABLE, "rb").read()).hexdigest()
    yield
    after = hashlib.sha256(open(_TRACKED_TABLE, "rb").read()).hexdigest()
    assert before == after, ("a test rewrote the tracked "
                            "worklog/worklog_yixun/model_comparison.md")


@pytest.fixture(autouse=True)
def restore_rows():
    """Leave ``G.ROWS`` exactly as the tracked file defines it, whatever a test did."""
    yield
    G.ROWS = list(DEFAULT_ROWS)


# --------------------------------------------------------------------------- #
# 1. the row spec the importer emits is one of THIS generator's rows
# --------------------------------------------------------------------------- #
def test_importer_row_specs_are_the_generators_four_field_form():
    """``(label, protocol, K, [glob])`` -- the shape main() unpacks as ``spec[:4]``."""
    for spec in IMPORTER_ROW_SPECS:
        assert len(spec) == 4, spec
        label, proto, k, pats = spec
        assert isinstance(label, str) and label
        assert proto in ("fa eval", "vanilla eval")
        assert k in KS
        assert isinstance(pats, list) and pats and all(isinstance(p, str) for p in pats)
        # a fifth field would select a contract; these rows take the default "table" one
        contract = spec[4] if len(spec) > 4 else "table"
        assert contract == "table"


def test_registered_specs_cover_both_arms_and_both_k():
    assert {(label, k) for label, _p, k, _g in IMPORTER_ROW_SPECS} == {
        (_label(arm), k) for arm in ARMS for k in KS}


def test_data_curve_rows_are_claimed_by_no_experiment_specific_branch():
    """The routing question, asked of the generator's own predicates.

    ``is_exp14_row`` claims any row whose pattern contains ``exp14_`` for the yaw campaign
    and ``is_exp11_row`` any row naming ``exp11_``; either would send a data-curve cell to a
    validator that cannot parse ``dc_cyl_f025_K8_s42`` and render the row BLOCKED. The
    import directory is named ``data_curve_import`` precisely so that neither fires."""
    for _label_, _proto, _k, pats in IMPORTER_ROW_SPECS:
        assert G.is_exp14_row(pats) is False, pats
        assert G.is_exp11_row(pats) is False, pats
        assert G.is_batched_orbit_row(pats) is False, pats
        for pattern in pats:
            assert "exp14_" not in pattern and "exp11_" not in pattern


def test_the_registered_glob_matches_exactly_the_five_cells_of_its_k(tmp_path):
    """n=5, established on the filesystem the generator globs.

    The pattern ends in ``*.json``, so anything else ``.json``-suffixed sitting beside the
    cells would join them (the exp_11 rows carry the same warning about their sidecars).
    What the importer installs beside them is ``MANIFEST.sha256`` and the ``.pt`` bundle --
    neither matches -- so the cell is exactly its five seeds."""
    root = build_fixture_root(tmp_path, exp10_evidence=True)
    for arm in ARMS:
        for k in KS:
            pattern, = [p for lbl, _pr, kk, [p] in IMPORTER_ROW_SPECS
                        if lbl == _label(arm) and kk == k]
            hits = sorted(globmod.glob(os.path.join(str(root), pattern), recursive=True))
            assert [os.path.basename(h) for h in hits] == [
                metrics_basename(arm, k, seed) for seed in SEEDS], (arm, k, hits)
            assert len(hits) == G.MIN_SEEDS == 5


# --------------------------------------------------------------------------- #
# 2. the protocol label each arm renders
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("evidence_ready, cyl_expected", [
    (False, CYL_LABEL_EVIDENCE_ABSENT),
    (True, CYL_LABEL_EVIDENCE_PRESENT),
])
def test_protocol_label_of_each_arm(evidence_ready, cyl_expected):
    """The exact strings, in both states of the deferred label migration.

    A data-curve row is not a batched-orbit row, so it can never render ``(batched)``; once
    the exp_10 endpoint evidence is on disk -- as it is in the real checkout -- the
    migration appends ``(legacy-loop)`` to the cyl arm's ``fa eval``. That claim is true of
    the evidence: the arm is scored by the per-angle ``invariant_conditioning`` loop, which
    with ``--frame-avg-angles 0`` runs the identity pass alone. The vanilla arm carries no
    orbit label in either state."""
    assert G.protocol_label("fa eval", False, evidence_ready) == cyl_expected
    assert "batched" not in G.protocol_label("fa eval", False, evidence_ready)
    assert G.protocol_label("vanilla eval", False, evidence_ready) == VAN_LABEL


# --------------------------------------------------------------------------- #
# 3. the end-to-end run: real globbing, real aggregation, real markdown
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("exp10_evidence, cyl_label", [
    (False, CYL_LABEL_EVIDENCE_ABSENT),
    (True, CYL_LABEL_EVIDENCE_PRESENT),
])
def test_the_generator_renders_a_two_arm_import_directory(tmp_path, monkeypatch,
                                                          exp10_evidence, cyl_label):
    """plan §9 / M7: the authoritative generator, over a fixture import directory."""
    root = build_fixture_root(tmp_path, exp10_evidence=exp10_evidence)
    monkeypatch.setattr(G, "ROWS", list(IMPORTER_ROW_SPECS))
    assert G.main(["--repo-root", str(root)]) == 0
    text = written_table(root)
    assert len(data_rows(text)) == 4, data_rows(text)
    for arm in ARMS:
        for k in KS:
            cells = find_row(text, _label(arm), k)
            assert cells[1] == (cyl_label if arm == "cyl" else VAN_LABEL), cells
            assert cells[3] == "5", f"{arm} K={k} aggregated {cells[3]} files, not 5"
            assert cells[4:10] == EXPECTED_CELLS[(arm, k)], (arm, k, cells)
    # ...and nothing in the round was refused, deferred or half-published
    assert_no_refusals(text)
    for arm in ARMS:
        assert _label(arm) in text


def test_a_single_arm_publishes_on_its_own(tmp_path, monkeypatch):
    """No transaction gate reaches these rows: the exp_11 two-K gate, the Q9 four-cell
    round, the exp_14 per-arm pair and the exp_15 pair are all keyed on a contract or a
    pattern no data-curve row carries. Publishing the cyl arm before the van arm exists is
    therefore a complete, non-aborting regeneration -- which is what announcement 04's
    per-arm update needs, and also what nobody should mistake for a paired comparison."""
    root = build_fixture_root(tmp_path, exp10_evidence=True, arms=("cyl",))
    monkeypatch.setattr(G, "ROWS", [spec for spec in IMPORTER_ROW_SPECS
                                    if spec[0] == _label("cyl")])
    assert G.main(["--repo-root", str(root)]) == 0
    text = written_table(root)
    assert {(cells[0], cells[2]) for cells in data_rows(text)} == {
        (_label("cyl"), "1"), (_label("cyl"), "8")}
    assert_no_refusals(text)
    for k in KS:
        assert find_row(text, _label("cyl"), k)[4:10] == EXPECTED_CELLS[("cyl", k)]


def test_an_incomplete_cell_renders_pending_not_a_number(tmp_path, monkeypatch):
    """Four seeds is not a row: MIN_SEEDS = 5, and a four-seed mean must never print."""
    root = build_fixture_root(tmp_path, exp10_evidence=True)
    os.remove(os.path.join(_import_dir(root, "cyl"), metrics_basename("cyl", 8, 46)))
    monkeypatch.setattr(G, "ROWS", list(IMPORTER_ROW_SPECS))
    assert G.main(["--repo-root", str(root)]) == 0
    text = written_table(root)
    short = find_row(text, _label("cyl"), 8)
    assert short[3] == "4"
    assert "pending (4/5 seeds on disk)" in short[4]
    assert " ± " not in " | ".join(short), "a four-seed mean was published"
    assert find_row(text, _label("cyl"), 1)[4:10] == EXPECTED_CELLS[("cyl", 1)]  # unaffected


# --------------------------------------------------------------------------- #
# 4. the routing claim, proved directly
# --------------------------------------------------------------------------- #
def test_no_experiment_specific_validator_is_ever_invoked(tmp_path, monkeypatch):
    """A round with nothing BLOCKED is only indirect evidence of correct routing.

    A gate could also have been entered and passed for reasons that say nothing about a
    data-curve cell. So the three per-experiment cell gates are replaced by sentinels that
    raise: a complete two-arm round that still renders its numbers cannot have touched any
    of them, which is what "the default ``table`` contract" has to mean."""
    root = build_fixture_root(tmp_path, exp10_evidence=True)

    def forbidden(name):
        def _raise(*args, **kwargs):
            raise AssertionError(f"{name} was invoked for a data-curve row")
        return _raise

    for gate in ("validate_exp11_cell", "validate_exp14_cell", "validate_exp15_cell"):
        monkeypatch.setattr(G, gate, forbidden(gate))
    monkeypatch.setattr(G, "ROWS", list(IMPORTER_ROW_SPECS))
    assert G.main(["--repo-root", str(root)]) == 0
    text = written_table(root)
    assert_no_refusals(text)
    for arm in ARMS:
        for k in KS:
            assert find_row(text, _label(arm), k)[4:10] == EXPECTED_CELLS[(arm, k)]


# --------------------------------------------------------------------------- #
# 5. documented traps: behaviour a reader would not assume
# --------------------------------------------------------------------------- #
def test_trap_exp14_in_a_row_glob_is_claimed_by_the_yaw_campaign():
    """The trap the importer's ``assert_no_forbidden_substring`` exists for.

    ``is_exp14_row`` is a substring test over the row's GLOB, so an import directory named
    ``exp14_*`` -- the announcement-07 form the NAS path uses -- would hand a data-curve row
    to the yaw campaign's label and validator."""
    assert G.is_exp14_row(["outputs_FLAC/exp14_data_curve/dc_cyl_f025/*_K8_s4[2-6]*.json"])
    assert G.is_batched_orbit_row(["outputs_FLAC/exp14_data_curve/*_K8_s4[2-6]*.json"])
    assert G.is_exp11_row(["outputs_FLAC/exp11_C8/**/*exp11_C8_conf_S40000*.json"])


def test_trap_exp14_in_a_glob_publishes_a_false_batched_label(tmp_path, monkeypatch):
    """...and it does so SILENTLY, with numbers.

    main() derives the protocol label from the PATTERN (``is_batched_orbit_row(pats)``) but
    routes the validator on the BASENAMES. A row whose glob says ``exp14_`` while its files
    do not therefore renders a numeric, unblocked line claiming ``fa eval (batched)`` -- a
    provenance the data-curve evidence does not have and cannot get, since its evaluator
    predates the batched orbit. Nothing in the generator catches this; the only guard is the
    kit refusing to emit such a pattern."""
    root = build_fixture_root(tmp_path, exp10_evidence=True)
    renamed = os.path.join(str(root), "outputs_FLAC", "data_curve_import", "exp14_dc_cyl")
    os.rename(_import_dir(root, "cyl"), renamed)
    monkeypatch.setattr(G, "ROWS", [
        (_label("cyl"), "fa eval", 8,
         ["outputs_FLAC/data_curve_import/exp14_dc_cyl/*_K8_s4[2-6]*.json"])])
    assert G.main(["--repo-root", str(root)]) == 0
    cells = find_row(written_table(root), _label("cyl"), 8)
    assert cells[1] == "fa eval (batched)", cells          # the false claim
    assert cells[3] == "5" and cells[4:10] == EXPECTED_CELLS[("cyl", 8)]


def test_trap_exp14_in_a_basename_routes_the_row_to_the_yaw_validator(tmp_path):
    """The other half of the same trap: render_row's exp_14 branch reads the BASENAMES.

    A cell file whose name carries ``exp14_`` is handed to ``exp14_validate_cell``, which
    was written for eval names like ``exp14_C8_zref_S40000_s42_K8`` and cannot speak for a
    data-curve cell -- so the row renders BLOCKED, i.e. refused for the wrong reason."""
    root = build_fixture_root(tmp_path, exp10_evidence=True)
    target = _import_dir(root, "cyl")
    files = []
    for seed in SEEDS:
        source = os.path.join(target, metrics_basename("cyl", 8, seed))
        renamed = source.replace("dc_cyl_f025_K8", "exp14_dc_cyl_K8")
        os.rename(source, renamed)
        files.append(renamed)
    line, blocked = G.render_row(_label("cyl"), "fa eval", 8, sorted(files),
                                 repo_root=str(root))
    assert blocked is True
    assert "BLOCKED — row validation failed:" in line
    assert " ± " not in line


def test_trap_a_stray_json_without_metrics_aborts_the_whole_regeneration(tmp_path,
                                                                        monkeypatch):
    """A ``.json`` sidecar in the import directory does not block one row -- it raises.

    ``agg_files`` raises ``ValueError`` for a payload it cannot print, and ``render_row``
    does NOT catch it (its docstring says otherwise), so main() dies with a traceback and
    writes nothing. Fail-closed, but the operator sees a crash rather than a BLOCKED row.
    The importer installs no ``.json`` beside the cells, and must not start."""
    root = build_fixture_root(tmp_path, exp10_evidence=True)
    sidecar = os.path.join(_import_dir(root, "cyl"),
                           metrics_basename("cyl", 8, 42) + ".screenmeta.json")
    with open(sidecar, "w") as fh:
        json.dump({"commit": "z" * 40}, fh)
    monkeypatch.setattr(G, "ROWS", list(IMPORTER_ROW_SPECS))
    with pytest.raises(ValueError, match="no metrics object to aggregate"):
        G.main(["--repo-root", str(root)])
    assert not os.path.isfile(os.path.join(str(root), "worklog", "worklog_yixun",
                                           "model_comparison.md"))


def test_trap_a_second_json_payload_is_averaged_into_the_cell(tmp_path, monkeypatch):
    """And a stray ``.json`` that DOES carry metrics is worse: it publishes.

    Six files render a six-file mean under a five-seed row's label, with ``n`` the only
    tell. Nothing downstream counts seeds for a ``table``-contract row (the exp_11, exp_14
    and exp_15 contracts each do, but no data-curve row reaches them), so the import
    directory holding exactly the ten cells is the whole guarantee."""
    root = build_fixture_root(tmp_path, exp10_evidence=True)
    duplicate = os.path.join(_import_dir(root, "cyl"),
                             metrics_basename("cyl", 8, 42).replace(".json", "_copy.json"))
    with open(duplicate, "w") as fh:
        json.dump(metrics_record("cyl", 8, 0), fh)
    monkeypatch.setattr(G, "ROWS", list(IMPORTER_ROW_SPECS))
    assert G.main(["--repo-root", str(root)]) == 0
    cells = find_row(written_table(root), _label("cyl"), 8)
    assert cells[3] == "6", cells
    assert cells[4:10] != EXPECTED_CELLS[("cyl", 8)]       # a six-file mean, not the cell's
    assert " ± " in cells[4]                               # ...and it published anyway


# --------------------------------------------------------------------------- #
# 6. smoke: the tracked row table is unchanged and still renders
# --------------------------------------------------------------------------- #
def test_the_default_row_table_still_renders(tmp_path):
    """Nothing about the existing rows changes: the committed ``ROWS`` renders end to end.

    Run against an empty fixture root, every registered row is evidence-free, so this pins
    the row COUNT and that no spec raises -- the two things a new row family could break."""
    root = tmp_path / "maintree"
    (root / "worklog" / "worklog_yixun").mkdir(parents=True)
    assert G.ROWS == DEFAULT_ROWS, "a previous test leaked its row patch"
    assert G.main(["--repo-root", str(root)]) == 0
    text = written_table(root)
    rows = data_rows(text)
    assert len(rows) == len(DEFAULT_ROWS), (len(rows), len(DEFAULT_ROWS))
    present = {(cells[0], cells[1], cells[2]) for cells in rows}
    for spec in DEFAULT_ROWS:
        label, proto, k, pats = spec[:4]
        proto = G.protocol_label(proto, G.is_batched_orbit_row(pats),
                                 G.exp10_evidence_present(str(root)))
        assert (label, proto, str(k)) in present, spec[0]
    for arm in ARMS:
        assert _label(arm) not in text, "a data-curve row leaked into the tracked table"
