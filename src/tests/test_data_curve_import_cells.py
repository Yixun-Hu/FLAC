"""Tests for exp_14's announcement-04 importer (plan §9 D9, findings r2-5 and r3-3).

``model_comparison.md`` is regenerated from raw per-seed metric JSONs that live inside the
FLAC checkout, so publishing an exp_14 arm means physically copying ten files out of the
NAS run directory and registering a glob for them. Three things go wrong quietly there:

* **The glob claims the wrong block.** ``gen_model_comparison.py::is_exp14_row`` hands any
  row whose pattern contains ``exp14_`` to the exp_14 *yaw* campaign -- other protocol
  label, other validator. No emitted path, basename or row may carry that substring.
* **The copied cells are not the run's.** A metrics JSON alone carries neither seed nor
  dataset config, so a cell left over from an earlier checkpoint, an earlier protocol or a
  reduced split reads as a finished result. The import therefore gates on the *pair*: the
  metrics record (checkpoint path + digest + four conditioning flags) and its sibling
  prediction bundle (full 6,337-item split, K-appropriate config, same seed and protocol).
* **The copy itself is silently wrong.** Every destination file is re-hashed after the
  copy and recorded beside its source path in ``MANIFEST.sha256``.

Refusal is all-or-nothing: an import that cannot validate all ten cells copies none of
them, because a half-published arm is exactly what a glob would average without noticing.

CPU-only; the fixtures store 2-item prediction tensors instead of 6,337-item ones.
"""
import ast
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess

import pytest
import torch

from src.tools.data_curve import import_cells, names

#: Fixture split size. The real contract is names.N_ITEMS_UNSEEN (6,337); a bundle of that
#: shape is 260 MB, so the tests pass the expectation in explicitly instead.
N_ITEMS = 2
BASE_METRICS = {
    "T60": 9.0, "Invalid T60": 0.0, "C50": 1.0, "EDT": 40.0, "FD": 0.32,
    "RIR_to_GT_RIR_R@1": 5.0, "RIR_to_GT_RIR_R@5": 15.0, "RIR_to_GT_RIR_R@10": 23.0,
    "RIR_to_geom_R@1": 3.8, "RIR_to_geom_R@5": 13.0, "RIR_to_geom_R@10": 20.0,
}


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fin:
        digest.update(fin.read())
    return digest.hexdigest()


def make_run(tmp_path, arm="cyl", tag="025", drop=(), metrics_patch=None,
             bundle_patch=None, ckpt_bytes=b"checkpoint bytes", step=names.MAX_STEPS,
             n_items=N_ITEMS):
    """A run on a fake NAS: one checkpoint, ten metric JSONs, ten ``n_items``-item bundles.

    ``step`` builds the same internally consistent artifacts around a checkpoint that is
    NOT the final one -- the case a ``--ckpt-path`` override used to let through.
    """
    nas = tmp_path / "nas"
    run_dir = nas / names.run_id(arm, tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = str(run_dir / f"epoch=8-step={step}.ckpt")
    with open(ckpt, "wb") as fout:
        fout.write(ckpt_bytes)
    digest = sha256_of(ckpt)
    angles = [0.0] if arm == "cyl" else None
    for K in names.K_VALUES:
        for seed in names.SEEDS:
            if (K, seed) in drop:
                continue
            record = {
                "metrics": dict(BASE_METRICS), "ckpt_path": ckpt,
                "rotate_deg": names.ROTATE_DEG,
                "cond_method": names.ARM_COND_METHOD[arm], "frame_avg_angles": angles,
                "cond_autocast": names.COND_AUTOCAST, "ckpt_sha256": digest,
            }
            record.update((metrics_patch or {}).get((K, seed), {}))
            with open(names.metrics_json_path(ckpt, arm, tag, K, seed), "w") as fout:
                json.dump(record, fout)
            meta = {
                "dataset_config": names.EVAL_DATASET_CONFIGS[K], "seed": seed,
                "n_samples": n_items, "n_items": n_items,
                "cond_method": names.ARM_COND_METHOD[arm], "frame_avg_angles": angles,
                "rotate_deg": names.ROTATE_DEG, "batch_size": 8,
                "cond_autocast": names.COND_AUTOCAST, "ckpt_path": ckpt,
                "eval_name": names.eval_name(arm, tag, K, seed), "steps": names.EVAL_STEPS,
                "cfg_scale": names.EVAL_CFG_SCALE, "stored_after_clamp_pad": True,
                "artifact_contract": names.PREDICTIONS_ARTIFACT_CONTRACT,
                "ckpt_sha256": digest,
            }
            meta.update((bundle_patch or {}).get((K, seed), {}))
            torch.save({"predictions": torch.zeros(n_items, 1, names.SAMPLE_LEN),
                        "meta": meta},
                       names.predictions_pt_path(ckpt, arm, tag, K, seed))
    return str(nas), ckpt, digest


def run_import(tmp_path, arm="cyl", tag="025", expect_sha=None, n_items=N_ITEMS, **kwargs):
    nas, ckpt, digest = make_run(tmp_path, arm, tag, n_items=n_items, **kwargs)
    checkout = tmp_path / "flac"
    checkout.mkdir(exist_ok=True)
    result, violations = import_cells.import_run(
        nas, arm, tag, str(checkout), expect_sha or digest, expect_n=n_items)
    return result, violations, str(checkout), ckpt


# ------------------------------------------------------------------------- happy path
def test_a_complete_run_is_copied_into_the_tracked_import_directory(tmp_path):
    result, violations, checkout, _ = run_import(tmp_path)
    assert violations == []
    assert result["dest_dir"] == os.path.join(
        checkout, "outputs_FLAC", "data_curve_import", "dc_cyl_f025")
    assert len(result["files"]) == len(names.K_VALUES) * len(names.SEEDS) == 10
    for entry in result["files"]:
        assert os.path.exists(entry["dest"])
        assert sha256_of(entry["dest"]) == entry["sha256"] == sha256_of(entry["source"])


def test_the_manifest_records_a_digest_and_its_source_for_every_file(tmp_path):
    result, _, _, _ = run_import(tmp_path)
    manifest = open(result["manifest"]).read()
    for entry in result["files"]:
        assert entry["sha256"] in manifest
        assert entry["source"] in manifest             # where the bytes came from
        assert os.path.basename(entry["dest"]) in manifest
    # The un-commented half is a plain `sha256sum -c` list rooted at the FLAC checkout.
    checkable = [ln for ln in manifest.splitlines() if ln and not ln.startswith("#")]
    assert len(checkable) == 11                     # ten cells + the attestation (round V)
    assert len([ln for ln in checkable if ln.endswith(".json")]) == 10
    assert len([ln for ln in checkable if ln.endswith("/ATTESTATION.txt")]) == 1
    assert all(ln.split("  ", 1)[1].startswith("outputs_FLAC/data_curve_import/")
               for ln in checkable)


def test_the_row_spec_names_the_arms_own_protocol_and_the_import_glob(tmp_path):
    cyl, _, _, _ = run_import(tmp_path, arm="cyl", tag="025")
    van, _, _, _ = run_import(tmp_path / "v", arm="van", tag="075")
    assert len(cyl["row_specs"]) == 2
    assert any('"fa eval", 8,' in row for row in cyl["row_specs"])
    assert any('"vanilla eval", 1,' in row for row in van["row_specs"])
    assert any('"outputs_FLAC/data_curve_import/dc_cyl_f025/*_K8_s4[2-6]*.json"' in row
               for row in cyl["row_specs"])
    assert all("f=25%" in row for row in cyl["row_specs"])
    assert all("f=75%" in row for row in van["row_specs"])


def test_no_emitted_path_row_or_basename_carries_the_forbidden_substring(tmp_path):
    result, _, _, _ = run_import(tmp_path)
    emitted = [result["dest_dir"], result["manifest"], result["attestation"]] \
        + result["row_specs"] \
        + [entry["dest"] for entry in result["files"]]
    assert all(names.FORBIDDEN_SUBSTRING not in text for text in emitted)


# ------------------------------------------------------------------------- refusals
def _nothing_was_copied(result, checkout):
    root = os.path.join(checkout, "outputs_FLAC", "data_curve_import")
    assert result["files"] == []
    assert not os.path.exists(root) or os.listdir(root) == []


def test_a_missing_seed_refuses_the_whole_import(tmp_path):
    result, violations, checkout, _ = run_import(tmp_path, drop=[(8, 45)])
    assert any("s45" in v for v in violations)
    _nothing_was_copied(result, checkout)


@pytest.mark.parametrize("field, value", [
    ("cond_method", "vanilla"),
    ("frame_avg_angles", None),
    ("rotate_deg", 45.0),
    ("cond_autocast", "off"),
])
def test_a_cell_scored_under_another_protocol_refuses_the_import(tmp_path, field, value):
    result, violations, checkout, _ = run_import(
        tmp_path, metrics_patch={(1, 43): {field: value}})
    assert any(field in v for v in violations)
    _nothing_was_copied(result, checkout)


def test_a_cell_naming_another_checkpoint_refuses_the_import(tmp_path):
    result, violations, checkout, _ = run_import(
        tmp_path, metrics_patch={(8, 42): {"ckpt_path": "/elsewhere/epoch=9-step=40000.ckpt"}})
    assert any("ckpt_path" in v for v in violations)
    _nothing_was_copied(result, checkout)


def test_a_digest_the_launcher_did_not_validate_refuses_the_import(tmp_path):
    result, violations, checkout, _ = run_import(tmp_path, expect_sha="b" * 64)
    assert any("ckpt_sha256" in v or "sha256" in v for v in violations)
    _nothing_was_copied(result, checkout)


def test_a_bundle_from_a_reduced_split_refuses_the_import(tmp_path):
    # The metrics JSON carries neither seed nor n_items: only the bundle can prove the
    # cell covered the whole 6,337-item unseen split (announcement 01).
    result, violations, checkout, _ = run_import(
        tmp_path, bundle_patch={(8, 44): {"n_items": N_ITEMS - 1, "n_samples": N_ITEMS - 1}})
    assert any("n_items" in v for v in violations)
    _nothing_was_copied(result, checkout)


def test_a_bundle_naming_the_other_K_s_dataset_config_refuses_the_import(tmp_path):
    result, violations, checkout, _ = run_import(
        tmp_path, bundle_patch={(8, 46): {"dataset_config": names.EVAL_DATASET_CONFIGS[1]}})
    assert any("dataset_config" in v for v in violations)
    _nothing_was_copied(result, checkout)


def test_a_missing_prediction_bundle_refuses_the_import(tmp_path):
    result, violations, checkout, ckpt = run_import(tmp_path)          # first, a good one
    os.remove(names.predictions_pt_path(ckpt, "cyl", "025", 1, 42))
    nas = os.path.dirname(os.path.dirname(ckpt))
    result, violations = import_cells.import_run(
        nas, "cyl", "025", str(tmp_path / "flac2"), sha256_of(ckpt), expect_n=N_ITEMS)
    assert any("K1 s42" in v for v in violations)
    _nothing_was_copied(result, str(tmp_path / "flac2"))


def test_an_unexpected_ckpt_sha256_argument_is_a_caller_error(tmp_path):
    nas, _, _ = make_run(tmp_path)
    with pytest.raises(ValueError):
        import_cells.import_run(nas, "cyl", "025", str(tmp_path), "not-a-digest",
                                expect_n=N_ITEMS)


# ------------------------------------------------------------------------------ CLI
def test_cli_imports_and_prints_the_row_specs(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(names, "N_ITEMS_UNSEEN", N_ITEMS)
    nas, _, digest = make_run(tmp_path)
    checkout = tmp_path / "flac"
    checkout.mkdir()
    rc = import_cells.main(["--nas-root", nas, "--arm", "cyl", "--tag", "025",
                            "--flac-checkout", str(checkout),
                            "--expect-ckpt-sha256", digest])
    assert rc == 0
    printed = capsys.readouterr().out
    assert '"fa eval", 8,' in printed
    assert "MANIFEST.sha256" in printed
    assert "ATTESTATION.txt" in printed


def test_cli_refuses_and_exits_non_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(names, "N_ITEMS_UNSEEN", N_ITEMS)
    nas, _, digest = make_run(tmp_path, drop=[(1, 46)])
    checkout = tmp_path / "flac"
    checkout.mkdir()
    rc = import_cells.main(["--nas-root", nas, "--arm", "cyl", "--tag", "025",
                            "--flac-checkout", str(checkout),
                            "--expect-ckpt-sha256", digest])
    assert rc != 0


# ------------------------------------------------- the override cannot dodge "final"
def test_a_checkpoint_that_is_not_the_final_one_is_refused(tmp_path):
    # Internally consistent in every way -- the artifacts hang off it, the digest matches,
    # the protocol is right -- and still not the 40,000-step checkpoint this arm reports.
    nas, ckpt, digest = make_run(tmp_path, step=2500)
    checkout = tmp_path / "flac"
    checkout.mkdir()
    result, violations = import_cells.import_run(nas, "cyl", "025", str(checkout), digest,
                                                 expect_n=N_ITEMS, ckpt=ckpt)
    assert any(str(names.MAX_STEPS) in v for v in violations)
    _nothing_was_copied(result, str(checkout))


def test_an_override_that_is_not_the_discovered_final_checkpoint_is_refused(tmp_path):
    nas, ckpt, digest = make_run(tmp_path, step=2500)
    final = os.path.join(os.path.dirname(ckpt), f"epoch=8-step={names.MAX_STEPS}.ckpt")
    with open(final, "wb") as fout:
        fout.write(b"the real final checkpoint")
    checkout = tmp_path / "flac"
    checkout.mkdir()
    result, violations = import_cells.import_run(nas, "cyl", "025", str(checkout), digest,
                                                 expect_n=N_ITEMS, ckpt=ckpt)
    assert any("final" in v for v in violations)
    _nothing_was_copied(result, str(checkout))


# --------------------------------------------- installation is one directory rename
def test_the_destination_appears_whole_manifest_included(tmp_path):
    result, violations, checkout, _ = run_import(tmp_path)
    assert violations == []
    assert result["installed"] == "new"
    assert sorted(os.listdir(result["dest_dir"])) == sorted(
        [os.path.basename(e["dest"]) for e in result["files"]]
        + [import_cells.MANIFEST_BASENAME, "ATTESTATION.txt"])
    # no staging directory survives anywhere under the import root
    root = os.path.join(checkout, "outputs_FLAC", "data_curve_import")
    assert [name for name in os.listdir(root) if name.startswith(".")] == []


@pytest.mark.parametrize("victim", ["os.rename", "manifest", "attestation"])
def test_an_install_that_fails_half_way_leaves_no_destination(tmp_path, monkeypatch, victim):
    nas, _, digest = make_run(tmp_path)
    checkout = tmp_path / "flac"
    checkout.mkdir()
    boom = lambda *a, **k: (_ for _ in ()).throw(OSError("disk went away"))
    if victim == "os.rename":
        monkeypatch.setattr(import_cells.os, "rename", boom)
    elif victim == "manifest":
        monkeypatch.setattr(import_cells, "_write_manifest", boom)
    else:
        monkeypatch.setattr(import_cells, "_write_attestation", boom)
    result, violations = import_cells.import_run(nas, "cyl", "025", str(checkout), digest,
                                                 expect_n=N_ITEMS)
    assert violations
    assert not os.path.exists(result["dest_dir"])
    root = os.path.join(str(checkout), "outputs_FLAC", "data_curve_import")
    assert not os.path.exists(root) or os.listdir(root) == []


def test_a_destination_holding_a_stale_file_is_refused(tmp_path):
    nas, _, digest = make_run(tmp_path)
    checkout = tmp_path / "flac"
    dest = checkout / "outputs_FLAC" / "data_curve_import" / "dc_cyl_f025"
    dest.mkdir(parents=True)
    stale = dest / "epoch=7-step=40000_metrics_1_1.0_dc_cyl_f025_K8_s42_fa_invariant_a1.json"
    stale.write_text("{}")
    result, violations = import_cells.import_run(nas, "cyl", "025", str(checkout), digest,
                                                 expect_n=N_ITEMS)
    assert any("already exists" in v for v in violations)
    assert stale.read_text() == "{}"          # nothing of theirs was touched


def test_an_identical_rerun_is_idempotent(tmp_path):
    nas, _, digest = make_run(tmp_path)
    checkout = str(tmp_path / "flac")
    os.makedirs(checkout)
    first, violations = import_cells.import_run(nas, "cyl", "025", checkout, digest,
                                                expect_n=N_ITEMS)
    assert violations == [] and first["installed"] == "new"
    before = sorted((name, sha256_of(os.path.join(first["dest_dir"], name)))
                    for name in os.listdir(first["dest_dir"]))
    second, violations = import_cells.import_run(nas, "cyl", "025", checkout, digest,
                                                 expect_n=N_ITEMS)
    assert violations == []
    assert second["installed"] == "identical"
    assert sorted((name, sha256_of(os.path.join(second["dest_dir"], name)))
                  for name in os.listdir(second["dest_dir"])) == before


def test_the_row_specs_parse_as_the_generators_own_four_tuples():
    # Plan §9 D9 also asks for an end-to-end test that RUNS gen_model_comparison.py over a
    # fixture import dir; that one belongs to the localization-exp checkout (see the
    # import_cells module docstring). This one imports nothing from the generator and only
    # proves the emitted text is a well-formed ROWS entry.
    row = import_cells.row_spec("cyl", "050", 8)
    label, protocol, K, patterns = ast.literal_eval(row.rstrip(","))
    assert (protocol, K) == ("fa eval", 8)
    assert isinstance(label, str) and isinstance(patterns, list)
    assert patterns == ["outputs_FLAC/data_curve_import/dc_cyl_f050/*_K8_s4[2-6]*.json"]


def test_an_import_hashes_the_checkpoint_exactly_once(tmp_path, monkeypatch):
    nas, ckpt, digest = make_run(tmp_path)
    checkout = tmp_path / "flac"
    checkout.mkdir()
    calls = []
    by_path, by_fd = import_cells._sha256, import_cells._digest_of

    def counting(path):
        if os.path.normpath(path) == os.path.normpath(ckpt):
            calls.append(path)
        return by_path(path)

    def counting_fd(fileobj):
        if os.path.normpath(getattr(fileobj, "name", "")) == os.path.normpath(ckpt):
            calls.append(fileobj.name)
        return by_fd(fileobj)

    monkeypatch.setattr(import_cells, "_sha256", counting)
    monkeypatch.setattr(import_cells, "_digest_of", counting_fd)
    monkeypatch.setattr(names, "file_sha256", counting)
    result, violations = import_cells.import_run(nas, "cyl", "025", str(checkout), digest,
                                                 expect_n=N_ITEMS)
    assert violations == [] and len(result["files"]) == 10
    assert len(calls) == 1


def test_a_checkpoint_swapped_mid_import_publishes_nothing(tmp_path, monkeypatch):
    nas, _, digest = make_run(tmp_path)
    checkout = tmp_path / "flac"
    checkout.mkdir()
    monkeypatch.setattr(import_cells, "_file_identity", lambda path: (1, 2, 3, 5))
    result, violations = import_cells.import_run(nas, "cyl", "025", str(checkout), digest,
                                                 expect_n=N_ITEMS)
    assert any("changed while" in v for v in violations)
    _nothing_was_copied(result, str(checkout))


def test_a_checkpoint_replaced_between_the_stat_and_the_hash_publishes_nothing(
        tmp_path, monkeypatch):
    # codex D3-fix2 finding 2, the importer's half: it hashed the pathname and stat-ed it
    # afterwards, so hash(A) -> stat(B) -> stat(B) published ten cells scored from a
    # checkpoint that is no longer there. The identity now comes from the descriptor the
    # digest was read through.
    nas, ckpt, digest = make_run(tmp_path)
    checkout = tmp_path / "flac"
    checkout.mkdir()
    real = import_cells._digest_of

    def swap_then_hash(fileobj):
        if os.path.normpath(getattr(fileobj, "name", "")) == os.path.normpath(ckpt):
            os.remove(ckpt)
            with open(ckpt, "wb") as fout:
                fout.write(b"a different checkpoint entirely")
        return real(fileobj)

    monkeypatch.setattr(import_cells, "_digest_of", swap_then_hash)
    result, violations = import_cells.import_run(nas, "cyl", "025", str(checkout), digest,
                                                 expect_n=N_ITEMS)
    assert any("changed while" in v for v in violations)
    _nothing_was_copied(result, str(checkout))


def test_a_checkpoint_rewritten_under_the_hash_publishes_nothing(tmp_path, monkeypatch):
    nas, ckpt, digest = make_run(tmp_path)
    checkout = tmp_path / "flac"
    checkout.mkdir()
    real = import_cells._digest_of

    def rewrite_then_hash(fileobj):
        if os.path.normpath(getattr(fileobj, "name", "")) == os.path.normpath(ckpt):
            with open(ckpt, "r+b") as fout:
                fout.write(b"XX")
                fout.truncate(7)
        return real(fileobj)

    monkeypatch.setattr(import_cells, "_digest_of", rewrite_then_hash)
    result, violations = import_cells.import_run(nas, "cyl", "025", str(checkout), digest,
                                                 expect_n=N_ITEMS)
    assert any("rewritten while it was being hashed" in v for v in violations)
    _nothing_was_copied(result, str(checkout))


# ------------------------------------------- row labels (codex T finding F7 / N6)
#: Typed out, never derived: the label used to end "(exp_14 cyl-dinov3)" for BOTH arms,
#: which reads as if the stock baseline were CylDINOv3. Each arm now names its own backbone.
EXPECTED_ROWS = {
    ("cyl", "025", 1):
        '("data-curve cyl f=25% @40k (exp_14 data-curve, CylDINO)", "fa eval", 1, '
        '["outputs_FLAC/data_curve_import/dc_cyl_f025/*_K1_s4[2-6]*.json"]),',
    ("van", "075", 8):
        '("data-curve van f=75% @40k (exp_14 data-curve, stock DINOv3-S)", "vanilla eval", 8, '
        '["outputs_FLAC/data_curve_import/dc_van_f075/*_K8_s4[2-6]*.json"]),',
}


@pytest.mark.parametrize("arm, tag, K", sorted(EXPECTED_ROWS))
def test_each_arm_s_row_names_its_own_backbone(arm, tag, K):
    assert import_cells.row_spec(arm, tag, K) == EXPECTED_ROWS[(arm, tag, K)]


@pytest.mark.parametrize("K", names.K_VALUES)
@pytest.mark.parametrize("tag", sorted(names.FRACTIONS))
@pytest.mark.parametrize("arm", names.ARMS)
def test_every_row_is_a_four_field_entry_with_a_truthful_backbone(arm, tag, K):
    row = import_cells.row_spec(arm, tag, K)
    assert names.assert_no_forbidden_substring(row) == row
    label, protocol, k, patterns = ast.literal_eval(row.rstrip(","))
    pct = {"025": 25, "050": 50, "075": 75}[tag]
    backbone = {"cyl": "CylDINO", "van": "stock DINOv3-S"}[arm]
    assert label == f"data-curve {arm} f={pct}% @40k (exp_14 data-curve, {backbone})"
    assert protocol == {"cyl": "fa eval", "van": "vanilla eval"}[arm]
    assert k == K
    assert patterns == [f"outputs_FLAC/data_curve_import/dc_{arm}_f{tag}/*_K{K}_s4[2-6]*.json"]
    assert "cyl-dinov3" not in row
    if arm == "van":                        # the stock arm's description names no CylDINO
        assert "cyl" not in label.split("(", 1)[1].lower()


# ------------------------------------- the attestation file (codex T finding F6 / M5)
# The metrics JSON alone does not say which seed, K, split or protocol it was scored under;
# the importer checks the pair before copying, and ATTESTATION.txt keeps what it found
# beside the cells. Round V-F1 (codex V finding 1): every section names WHERE its values
# come from -- configured, caller-supplied, verified, re-read, recorded, not verified -- and
# every expectation below is rebuilt from the fixture (paths typed out, digests re-hashed,
# item counts and shapes taken from what make_run wrote), never by asking the importer: an
# overclaim has to be typed here before it can pass.
ATTESTATION = "ATTESTATION.txt"
#: The repository the tests (and the importer) run from: src/tests -> src -> root.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_BY_K = {1: "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json",
               8: "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json"}
CONTRACT = ("clamped/padded callback input "
            "(float32 cast and 8000-sample crop are scoring-internal)")
#: A fixed import time, so the stamp is predictable (the importer reads the clock once).
STAMP_TIME = "2026-11-02T16:00:00-05:00"
#: The one header form: ``[<name> (<tag>: <detail>)]``.
HEADER = re.compile(r"^\[(?P<name>[^()\[\]]+) \((?P<tag>[^:()]+): (?P<detail>.+)\)\]$")
#: Where a section's values may come from -- typed out, the attestation's whole vocabulary.
TAGS = ("configured", "caller-supplied", "verified", "re-read", "recorded", "not verified")
HEADERS = {
    "run": "[run (configured: --arm/--tag and code constants; no run contract, split file, "
           "model config or training log is read)]",
    "final checkpoint": "[final checkpoint (verified: C1, C2)]",
    "expected checkpoint digest":
        "[expected checkpoint digest (caller-supplied: --expect-ckpt-sha256)]",
    "rows": "[rows (configured: --arm/--tag and code constants)]",
    "validation contract": "[validation contract (verified: the checks import_run and "
                           "validate_cells made; every cell below passed all of them)]",
    "limits": "[limits (not verified: what no check of this import establishes)]",
    "import stamp": "[import stamp (recorded: when and by which code the first import ran; "
                    "not a check)]",
}
#: The production split size, typed: what "full" means however a test patches names.
FULL_SPLIT = 6337
SOURCE_SHA_KEY = "record_source_sha (as recorded, unvalidated)"


def cell_headers(K, seed):
    """A cell's three sections: what was verified, what was re-read, what was recorded."""
    return (f"[cell K{K} s{seed} (verified: C3-C13 passed for this cell)]",
            f"[cell K{K} s{seed} bundle (re-read: prediction_bundle read once after "
            "validation -- hashed and loaded from the same bytes; nothing compared)]",
            f"[cell K{K} s{seed} recorded (recorded: copied or hashed; compared with "
            "nothing)]")


def full_no(n):
    return (f"no (fixture override expect_n={n}: C7 checked {n} items; the full unseen "
            f"split is {FULL_SPLIT})")


def full_yes(n):
    return (f"yes (validate_cells: C7 with names.N_ITEMS_UNSEEN = {n} items -- the full "
            "unseen split -- and C9)")


def git_head_and_tree():
    """The importer's own repository, asked directly: HEAD and whether tracked files differ."""
    def git(*args):
        return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True,
                              check=True).stdout
    dirty = git("status", "--porcelain", "--untracked-files=no").strip()
    return (git("rev-parse", "HEAD").strip(),
            "clean" if not dirty else "modified (tracked files differ from HEAD)")


def parse_attestation(text):
    """``{section name: {key: value}}`` -- an independent reader of the plain-text format.

    Every header must be ``[<name> (<tag>: <detail>)]`` with a tag from TAGS: a header that
    does not say where its values come from fails here instead of being read.
    """
    sections, current = {}, None
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        if line.startswith("["):
            match = HEADER.match(line)
            assert match and match["tag"] in TAGS, f"no provenance tag: {line!r}"
            assert match["name"] not in sections, f"duplicate section {line!r}"
            current = sections[match["name"]] = {}
            continue
        key, sep, value = line.partition(": ")
        assert sep and current is not None, f"not a 'key: value' line: {line!r}"
        assert key not in current, f"duplicate key {key!r} in {line!r}"
        current[key] = value
    return sections


def read_text(result):
    with open(os.path.join(result["dest_dir"], ATTESTATION)) as fin:
        return fin.read()


def read_attestation(result):
    return parse_attestation(read_text(result))


def assert_block(text, block):
    """``block`` -- a header and its lines -- appears verbatim, once, and ends a section."""
    lines = text.splitlines()
    assert lines.count(block[0]) == 1, f"{block[0]!r} is not exactly one line"
    start = lines.index(block[0])
    assert lines[start:start + len(block)] == block
    assert start + len(block) == len(lines) or lines[start + len(block)] == ""


def expected_header_lines(arm, tag, ckpt):
    """[run], [final checkpoint], [expected checkpoint digest], [rows] -- typed out."""
    pct, frac = {"025": (25, "0.25"), "050": (50, "0.5"), "075": (75, "0.75")}[tag]
    backbone = {"cyl": "CylDINO", "van": "stock DINOv3-S"}[arm]
    protocol = {"cyl": "fa eval", "van": "vanilla eval"}[arm]
    flags = {"cyl": "cond_method=fa_invariant frame_avg_angles=[0.0]",
             "van": "cond_method=vanilla frame_avg_angles=null"}[arm]
    label = f"data-curve {arm} f={pct}% @40k (exp_14 data-curve, {backbone})"
    rows = [f'row_K{K}: ("{label}", "{protocol}", {K}, ["outputs_FLAC/data_curve_import/'
            f'dc_{arm}_f{tag}/*_K{K}_s4[2-6]*.json"]),' for K in (1, 8)]
    return [HEADERS["run"], f"run_id: dc_{arm}_f{tag}", f"arm: {arm}", f"backbone: {backbone}",
            f"fraction: {frac}", f"fraction_tag: {tag}",
            f"imported_to: outputs_FLAC/data_curve_import/dc_{arm}_f{tag}", "",
            HEADERS["final checkpoint"], f"final_checkpoint: {ckpt}",
            f"final_checkpoint_sha256: {sha256_of(ckpt)}", "",
            HEADERS["expected checkpoint digest"], f"expect_ckpt_sha256: {sha256_of(ckpt)}",
            "syntax: verified (64 lowercase hex characters, names._check_ckpt_sha256)",
            "equals_final_checkpoint_bytes: verified (C2)",
            "provenance: not verified (see [limits] N3)", "",
            HEADERS["rows"], f"label: {label}", f"protocol: {protocol}",
            f"protocol_flags: {flags} rotate_deg=0.0 cond_autocast=bf16", *rows]


def expected_cell_lines(arm, tag, K, seed, ckpt, n=N_ITEMS, full=None, source_sha="absent"):
    """One cell's three sections, line for line, from the fixture's own known values."""
    run_dir, ckpt_stem = os.path.dirname(ckpt), os.path.basename(ckpt)[:-len(".ckpt")]
    stem = f"1_1.0_dc_{arm}_f{tag}_K{K}_s{seed}" + ("_fa_invariant_a1" if arm == "cyl" else "")
    record = os.path.join(run_dir, f"{ckpt_stem}_metrics_{stem}.json")
    bundle = os.path.join(run_dir, f"{ckpt_stem}_predictions_{stem}.pt")
    verified, reread, recorded = cell_headers(K, seed)
    return [
        verified, f"K: {K}", f"seed: {seed}", f"eval_name: dc_{arm}_f{tag}_K{K}_s{seed}",
        f"checkpoint: {ckpt}", f"checkpoint_sha256: {sha256_of(ckpt)}",
        f"metrics_record: {record}", f"metrics_record_sha256: {sha256_of(record)}",
        f"imported_copy: outputs_FLAC/data_curve_import/dc_{arm}_f{tag}/"
        f"{os.path.basename(record)}",
        f"prediction_bundle: {bundle}",
        "cond_method: " + {"cyl": "fa_invariant", "van": "vanilla"}[arm],
        "frame_avg_angles: " + {"cyl": "[0.0]", "van": "null"}[arm],
        "rotate_deg: 0.0", "cond_autocast: bf16", "steps: 1", "cfg_scale: 1.0",
        f"bundle_artifact_contract: {json.dumps(CONTRACT)}",
        f"dataset_config: {CONFIG_BY_K[K]}",
        f"dataset_config_matches_K: yes (C9: meta.dataset_config == {CONFIG_BY_K[K]}, the "
        f"full unseen-eval config of K={K})",
        f"full_unseen_contract_passed: {full or full_no(n)}",
        "item_identity_verified: no (see [limits] N1)", "",
        reread, f"prediction_bundle_sha256: {sha256_of(bundle)}", f"meta_n_items: {n}",
        f"meta_n_samples: {n}", f"predictions_shape: ({n}, 1, 10240)",
        "predictions_finite: yes", f"meta_dataset_config: {CONFIG_BY_K[K]}", "",
        recorded,
        f"dataset_config_sha256: {sha256_of(os.path.join(REPO, CONFIG_BY_K[K]))}",
        f"{SOURCE_SHA_KEY}: {source_sha}",
        f"importer_git_sha: {git_head_and_tree()[0]}", f"imported_at: {STAMP_TIME}",
    ]


@pytest.mark.parametrize("arm, tag", [("cyl", "025"), ("van", "050")])
def test_the_attestation_states_every_line_from_the_fixture(tmp_path, monkeypatch, arm, tag):
    monkeypatch.setattr(import_cells, "_now_iso", lambda: STAMP_TIME)
    result, violations, _, ckpt = run_import(tmp_path, arm=arm, tag=tag)
    assert violations == []
    text = read_text(result)
    assert_block(text, expected_header_lines(arm, tag, ckpt))
    for K in (1, 8):
        for seed in range(42, 47):
            assert_block(text, expected_cell_lines(arm, tag, K, seed, ckpt))


def test_every_section_header_names_where_its_values_come_from(tmp_path):
    result, _, _, _ = run_import(tmp_path)
    headers = [ln for ln in read_text(result).splitlines() if ln.startswith("[")]
    expected = [HEADERS[name] for name in ("run", "final checkpoint",
                                           "expected checkpoint digest", "rows",
                                           "validation contract", "limits")]
    for K in (1, 8):
        for seed in range(42, 47):
            expected += cell_headers(K, seed)
    assert headers == expected + [HEADERS["import stamp"]]
    tags = {HEADER.match(h)["name"]: HEADER.match(h)["tag"] for h in headers}
    assert set(tags.values()) == set(TAGS)
    assert sorted(n for n, t in tags.items() if t == "configured") == ["rows", "run"]
    assert [n for n, t in tags.items() if t == "caller-supplied"] == \
        ["expected checkpoint digest"]
    for tag in TAGS:                       # the opening comment spells out the vocabulary
        assert f"\n#   {tag} -- " in read_text(result)


def test_the_stock_arm_s_attestation_names_its_own_backbone_and_protocol(tmp_path):
    result, _, _, _ = run_import(tmp_path, arm="van", tag="075")
    sections = read_attestation(result)
    assert sections["run"]["backbone"] == "stock DINOv3-S"
    assert sections["rows"]["label"] == \
        "data-curve van f=75% @40k (exp_14 data-curve, stock DINOv3-S)"
    assert sections["rows"]["protocol"] == "vanilla eval"
    assert sections["rows"]["protocol_flags"] == \
        "cond_method=vanilla frame_avg_angles=null rotate_deg=0.0 cond_autocast=bf16"
    assert "vanilla / null / 0.0 / bf16" in sections["validation contract"]["C5"]


def test_a_record_s_source_sha_is_labelled_as_recorded_and_unvalidated(tmp_path):
    # codex V finding 1: any string in the record's source_sha is published (here a fake
    # '0123abcd'), so it must never read as something the import checked.
    result, violations, _, _ = run_import(
        tmp_path, metrics_patch={(8, 44): {"source_sha": "0123abcd"}})
    assert violations == []
    sections = read_attestation(result)
    assert sections["cell K8 s44 recorded"][SOURCE_SHA_KEY] == "0123abcd"
    assert sections["cell K1 s44 recorded"][SOURCE_SHA_KEY] == "absent"
    assert [name for name, body in sections.items()
            if any(key.startswith("record_source_sha") for key in body)] == \
        [f"cell K{K} s{seed} recorded" for K in (1, 8) for seed in range(42, 47)]
    assert "\nrecord_source_sha: " not in read_text(result)


def test_a_production_size_import_says_the_full_contract_passed(tmp_path, monkeypatch):
    monkeypatch.setattr(names, "N_ITEMS_UNSEEN", N_ITEMS)
    nas, _, digest = make_run(tmp_path)
    checkout = tmp_path / "flac"
    checkout.mkdir()
    result, violations = import_cells.import_run(nas, "cyl", "025", str(checkout), digest)
    assert violations == []
    sections = read_attestation(result)
    for K in (1, 8):
        for seed in range(42, 47):
            assert sections[f"cell K{K} s{seed}"]["full_unseen_contract_passed"] == \
                full_yes(N_ITEMS)
            assert sections[f"cell K{K} s{seed} bundle"]["meta_n_items"] == str(N_ITEMS)


def test_c2_credits_the_caller_and_no_round_v_overclaim_survives(tmp_path):
    result, _, _, _ = run_import(tmp_path)
    text = read_text(result)
    contract = parse_attestation(text)["validation contract"]
    assert list(contract) == [f"C{i}" for i in range(1, 14)]
    assert "to the caller-supplied --expect-ckpt-sha256 (syntax + equality verified; " \
           "provenance of the expected digest is not verified by the importer)" \
           in contract["C2"]
    assert "step=40000" in contract["C1"] and "record.ckpt_sha256" in contract["C4"]
    assert "fa_invariant / [0.0] / 0.0 / bf16" in contract["C5"]
    assert f"== {N_ITEMS} and predictions" in contract["C7"]
    assert f"({N_ITEMS}, 1, 10240) -- counts and shape, not which items" in contract["C7"]
    assert "meta.seed" in contract["C8"] and "meta.eval_name" in contract["C8"]
    assert CONFIG_BY_K[1] in contract["C9"] and CONFIG_BY_K[8] in contract["C9"]
    assert json.dumps(CONTRACT) in contract["C11"] and "meta.steps == 1" in contract["C11"]
    assert "meta.ckpt_sha256" in contract["C12"]
    assert "launcher" not in text and "full_unseen_split" not in text
    assert "or hashed by that import" not in text           # the round-V blanket sentence


def test_the_limits_section_names_what_no_check_establishes(tmp_path):
    result, _, _, _ = run_import(tmp_path)
    limits = read_attestation(result)["limits"]
    assert list(limits) == [f"N{i}" for i in range(1, 7)]
    assert limits["N1"].startswith("item identity") and \
        "C7 checks counts and shape only" in limits["N1"]
    assert "no run contract, split file, model config or training log is read" in limits["N2"]
    assert "--expect-ckpt-sha256" in limits["N3"] and "not its provenance" in limits["N3"]
    assert limits["N4"].startswith("record_source_sha")
    assert "nothing compares the two reads" in limits["N5"]
    assert limits["N6"].startswith("that dataset_config_sha256 is the file the evaluator read")


@pytest.mark.parametrize("n", [N_ITEMS, 3])
def test_a_fixture_override_states_its_real_item_count_and_never_yes(tmp_path, n):
    # codex V5: round V checked only which branch rendered. The bundles hold n items, so
    # the file says n -- and never that the full-split contract passed (C7 checked n).
    result, violations, _, _ = run_import(tmp_path, n_items=n)
    assert violations == []
    sections = read_attestation(result)
    for K in (1, 8):
        for seed in range(42, 47):
            passed = sections[f"cell K{K} s{seed}"]["full_unseen_contract_passed"]
            assert passed == full_no(n) and not passed.startswith("yes")
            bundle = sections[f"cell K{K} s{seed} bundle"]
            assert (bundle["meta_n_items"], bundle["meta_n_samples"],
                    bundle["predictions_shape"], bundle["predictions_finite"]) == \
                (str(n), str(n), f"({n}, 1, 10240)", "yes")
    assert f"== {n} and predictions" in sections["validation contract"]["C7"]


def test_a_production_import_of_a_reduced_split_is_refused_and_attests_nothing(tmp_path):
    # No fixture override: 2-item bundles fail the full split, so nothing at all is written.
    nas, _, digest = make_run(tmp_path)
    checkout = tmp_path / "flac"
    checkout.mkdir()
    result, violations = import_cells.import_run(nas, "cyl", "025", str(checkout), digest)
    assert any(f"meta.n_items is 2, expected {FULL_SPLIT}" in v for v in violations)
    assert any(f"predictions shape is (2, 1, 10240), expected ({FULL_SPLIT}, 1, 10240)" in v
               for v in violations)
    assert result["attestation"] is None and result["installed"] is None
    assert glob.glob(os.path.join(glob.escape(str(tmp_path)), "**", ATTESTATION),
                     recursive=True) == []
    _nothing_was_copied(result, str(checkout))


def test_the_bundle_section_states_what_the_reread_found_not_what_the_contract_required(
        tmp_path, monkeypatch):
    # READ, not derived from the contract: a bundle replaced after validation is stated as
    # it now is -- nothing compares the two reads ([limits] N5).
    nas, ckpt, digest = make_run(tmp_path)
    checkout = tmp_path / "flac"
    checkout.mkdir()
    victim = names.predictions_pt_path(ckpt, "cyl", "025", 8, 45)
    real_validate = import_cells.validate_cells

    def validate_then_replace(*args, **kwargs):
        files, violations = real_validate(*args, **kwargs)
        predictions = torch.zeros(5, 1, names.SAMPLE_LEN)
        predictions[0, 0, 0] = float("nan")
        torch.save({"predictions": predictions,
                    "meta": {"n_items": 5, "dataset_config": "elsewhere/other.json"}}, victim)
        return files, violations

    monkeypatch.setattr(import_cells, "validate_cells", validate_then_replace)
    result, violations = import_cells.import_run(nas, "cyl", "025", str(checkout), digest,
                                                 expect_n=N_ITEMS)
    assert violations == []
    sections = read_attestation(result)
    assert sections["cell K8 s45 bundle"] == {
        "prediction_bundle_sha256": sha256_of(victim), "meta_n_items": "5",
        "meta_n_samples": "absent", "predictions_shape": "(5, 1, 10240)",
        "predictions_finite": "no", "meta_dataset_config": "elsewhere/other.json"}
    assert sections["cell K8 s45"]["full_unseen_contract_passed"] == full_no(N_ITEMS)
    assert sections["cell K8 s44 bundle"]["meta_n_items"] == str(N_ITEMS)


def test_a_bundle_without_n_samples_says_absent(tmp_path):
    # check_bundle compares meta.n_samples only when present; the file must not invent one.
    nas, ckpt, digest = make_run(tmp_path)
    path = names.predictions_pt_path(ckpt, "cyl", "025", 1, 42)
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    del bundle["meta"]["n_samples"]
    torch.save(bundle, path)
    checkout = tmp_path / "flac"
    checkout.mkdir()
    result, violations = import_cells.import_run(nas, "cyl", "025", str(checkout), digest,
                                                 expect_n=N_ITEMS)
    assert violations == []
    sections = read_attestation(result)
    assert sections["cell K1 s42 bundle"]["meta_n_samples"] == "absent"
    assert sections["cell K1 s43 bundle"]["meta_n_samples"] == str(N_ITEMS)


def test_the_manifest_lists_the_attestation_and_every_line_verifies(tmp_path):
    result, _, checkout, _ = run_import(tmp_path)
    attestation = os.path.join(result["dest_dir"], ATTESTATION)
    assert result["attestation"] == attestation
    assert import_cells.ATTESTATION_BASENAME == ATTESTATION
    with open(result["manifest"]) as fin:
        checkable = [ln for ln in fin.read().splitlines() if ln and not ln.startswith("#")]
    assert f"{sha256_of(attestation)}  outputs_FLAC/data_curve_import/dc_cyl_f025/" \
           f"{ATTESTATION}" in checkable
    for line in checkable:
        digest, rel = line.split("  ", 1)
        assert sha256_of(os.path.join(checkout, rel)) == digest
    if shutil.which("sha256sum"):           # what a later reader actually runs
        done = subprocess.run(["sha256sum", "--check", "--strict", "--quiet",
                               result["manifest"]], cwd=checkout, capture_output=True)
        assert done.returncode == 0, done.stdout + done.stderr


def test_the_row_globs_match_the_ten_cells_and_never_the_attestation(tmp_path):
    result, _, checkout, _ = run_import(tmp_path)
    dest = result["dest_dir"]
    cells = sorted(os.path.join(dest, os.path.basename(e["dest"])) for e in result["files"])
    assert os.path.exists(os.path.join(dest, ATTESTATION))
    matched = []
    for row in result["row_specs"]:
        _, _, K, patterns = ast.literal_eval(row.rstrip(","))
        hits = sorted(glob.glob(os.path.join(glob.escape(checkout), patterns[0]),
                                recursive=True))
        assert len(hits) == 5 and all(f"_K{K}_" in os.path.basename(h) for h in hits)
        matched += hits
    assert sorted(matched) == cells
    assert sorted(glob.glob(os.path.join(glob.escape(dest), "*.json"))) == cells
    assert names.FORBIDDEN_SUBSTRING not in ATTESTATION


def _snapshot(directory):
    return {name: sha256_of(os.path.join(directory, name)) for name in os.listdir(directory)}


def test_an_existing_attestation_that_differs_is_refused_and_nothing_changes(tmp_path):
    nas, _, digest = make_run(tmp_path)
    checkout = str(tmp_path / "flac")
    os.makedirs(checkout)
    first, violations = import_cells.import_run(nas, "cyl", "025", checkout, digest,
                                                expect_n=N_ITEMS)
    assert violations == []
    attestation = os.path.join(first["dest_dir"], ATTESTATION)
    with open(attestation) as fin:
        text = fin.read()
    with open(attestation, "w") as fout:            # someone else's attestation
        fout.write(text.replace("seed: 43\n", "seed: 99\n", 1))
    before = _snapshot(first["dest_dir"])
    second, violations = import_cells.import_run(nas, "cyl", "025", checkout, digest,
                                                 expect_n=N_ITEMS)
    assert any("already exists" in v and ATTESTATION in v for v in violations)
    assert second["installed"] is None
    assert _snapshot(first["dest_dir"]) == before


def test_a_destination_without_an_attestation_is_refused_and_nothing_changes(tmp_path):
    # e.g. a directory installed by the pre-round-V importer: never merged into.
    nas, _, digest = make_run(tmp_path)
    checkout = str(tmp_path / "flac")
    os.makedirs(checkout)
    first, _ = import_cells.import_run(nas, "cyl", "025", checkout, digest, expect_n=N_ITEMS)
    os.remove(os.path.join(first["dest_dir"], ATTESTATION))
    before = _snapshot(first["dest_dir"])
    _, violations = import_cells.import_run(nas, "cyl", "025", checkout, digest,
                                            expect_n=N_ITEMS)
    assert any("already exists" in v and ATTESTATION in v for v in violations)
    assert _snapshot(first["dest_dir"]) == before


# ------------------------------------------- the import stamp (time + importer identity)
def test_the_import_stamp_names_the_importer_its_tree_and_the_time(tmp_path, monkeypatch):
    monkeypatch.setattr(import_cells, "_now_iso", lambda: STAMP_TIME)
    result, violations, _, _ = run_import(tmp_path)
    assert violations == []
    head, tree = git_head_and_tree()
    assert read_attestation(result)["import stamp"] == {
        "imported_at": STAMP_TIME, "importer": "src.tools.data_curve.import_cells",
        "importer_git_sha": head, "importer_tracked_tree": tree, "importer_worktree": REPO}


def test_the_clock_is_local_iso_8601_to_the_second():
    stamp = import_cells._now_iso()
    assert len(stamp) == len(STAMP_TIME) and stamp[10] == "T" and stamp[-6] in "+-"


def test_an_identical_rerun_keeps_the_first_import_s_stamp_byte_for_byte(tmp_path,
                                                                         monkeypatch):
    times = iter(["2026-11-02T10:00:00-05:00", "2026-11-03T11:11:11-05:00"])
    monkeypatch.setattr(import_cells, "_now_iso", lambda: next(times))
    nas, _, digest = make_run(tmp_path)
    checkout = str(tmp_path / "flac")
    os.makedirs(checkout)
    first, _ = import_cells.import_run(nas, "cyl", "025", checkout, digest, expect_n=N_ITEMS)
    before = _snapshot(first["dest_dir"])
    # a later rerun, from a later importer commit: the directory is what it would publish
    monkeypatch.setattr(import_cells, "_importer_identity", lambda: {
        "importer": "src.tools.data_curve.import_cells", "importer_git_sha": "f" * 40,
        "importer_tracked_tree": "clean", "importer_worktree": "/elsewhere"})
    second, violations = import_cells.import_run(nas, "cyl", "025", checkout, digest,
                                                 expect_n=N_ITEMS)
    assert violations == [] and second["installed"] == "identical"
    assert _snapshot(first["dest_dir"]) == before
    stamp = read_attestation(first)["import stamp"]
    assert stamp["imported_at"] == "2026-11-02T10:00:00-05:00"
    assert stamp["importer_git_sha"] == git_head_and_tree()[0]


def test_without_git_the_stamp_says_so_and_the_import_still_happens(tmp_path, monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("[Errno 2] No such file or directory: 'git'")

    monkeypatch.setattr(import_cells.subprocess, "run", no_git)
    result, violations, _, _ = run_import(tmp_path)
    assert violations == [] and result["installed"] == "new"
    stamp = read_attestation(result)["import stamp"]
    assert stamp["importer_git_sha"].startswith("unavailable (FileNotFoundError")
    assert stamp["importer_tracked_tree"] == "unavailable"


@pytest.mark.parametrize("vanishes", ["bundle", "record", "garbled bundle"])
def test_an_artifact_that_changes_after_validation_raises_and_installs_nothing(
        tmp_path, monkeypatch, vanishes):
    # The attestation reads two things validation did not keep: the bundle (to hash and
    # re-read it) and the staged record (for source_sha). If either is gone or garbled by
    # then, the import raises -- and nothing, not even a staging directory, is left behind.
    nas, ckpt, digest = make_run(tmp_path)
    checkout = tmp_path / "flac"
    checkout.mkdir()
    real_validate = import_cells.validate_cells

    def validate_then_lose_a_bundle(*args, **kwargs):
        files, violations = real_validate(*args, **kwargs)
        os.remove(names.predictions_pt_path(ckpt, "cyl", "025", 8, 45))
        return files, violations

    def garbled(path):
        raise json.JSONDecodeError("Expecting value", "", 0)

    def validate_then_garble_a_bundle(*args, **kwargs):
        files, violations = real_validate(*args, **kwargs)
        with open(names.predictions_pt_path(ckpt, "cyl", "025", 8, 45), "wb") as fout:
            fout.write(b"not a prediction bundle")
        return files, violations

    if vanishes == "bundle":
        monkeypatch.setattr(import_cells, "validate_cells", validate_then_lose_a_bundle)
        expected = OSError
    elif vanishes == "garbled bundle":
        monkeypatch.setattr(import_cells, "validate_cells", validate_then_garble_a_bundle)
        expected = ValueError
    else:
        monkeypatch.setattr(import_cells, "_recorded_source_sha", garbled)
        expected = ValueError
    with pytest.raises(expected):
        import_cells.import_run(nas, "cyl", "025", str(checkout), digest, expect_n=N_ITEMS)
    root = os.path.join(str(checkout), "outputs_FLAC", "data_curve_import")
    assert not os.path.exists(root) or os.listdir(root) == []
