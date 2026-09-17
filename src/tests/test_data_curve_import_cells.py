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
import hashlib
import json
import os

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
             bundle_patch=None, ckpt_bytes=b"checkpoint bytes", step=names.MAX_STEPS):
    """A run on a fake NAS: one checkpoint, ten metric JSONs, ten bundles.

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
                "n_samples": N_ITEMS, "n_items": N_ITEMS,
                "cond_method": names.ARM_COND_METHOD[arm], "frame_avg_angles": angles,
                "rotate_deg": names.ROTATE_DEG, "batch_size": 8,
                "cond_autocast": names.COND_AUTOCAST, "ckpt_path": ckpt,
                "eval_name": names.eval_name(arm, tag, K, seed), "steps": names.EVAL_STEPS,
                "cfg_scale": names.EVAL_CFG_SCALE, "stored_after_clamp_pad": True,
                "artifact_contract": names.PREDICTIONS_ARTIFACT_CONTRACT,
                "ckpt_sha256": digest,
            }
            meta.update((bundle_patch or {}).get((K, seed), {}))
            torch.save({"predictions": torch.zeros(N_ITEMS, 1, names.SAMPLE_LEN),
                        "meta": meta},
                       names.predictions_pt_path(ckpt, arm, tag, K, seed))
    return str(nas), ckpt, digest


def run_import(tmp_path, arm="cyl", tag="025", expect_sha=None, **kwargs):
    nas, ckpt, digest = make_run(tmp_path, arm, tag, **kwargs)
    checkout = tmp_path / "flac"
    checkout.mkdir(exist_ok=True)
    result, violations = import_cells.import_run(
        nas, arm, tag, str(checkout), expect_sha or digest, expect_n=N_ITEMS)
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
    assert len(checkable) == 10
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
    emitted = [result["dest_dir"], result["manifest"]] + result["row_specs"] \
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
        + [import_cells.MANIFEST_BASENAME])
    # no staging directory survives anywhere under the import root
    root = os.path.join(checkout, "outputs_FLAC", "data_curve_import")
    assert [name for name in os.listdir(root) if name.startswith(".")] == []


@pytest.mark.parametrize("victim", ["os.rename", "manifest"])
def test_an_install_that_fails_half_way_leaves_no_destination(tmp_path, monkeypatch, victim):
    nas, _, digest = make_run(tmp_path)
    checkout = tmp_path / "flac"
    checkout.mkdir()
    boom = lambda *a, **k: (_ for _ in ()).throw(OSError("disk went away"))
    if victim == "os.rename":
        monkeypatch.setattr(import_cells.os, "rename", boom)
    else:
        monkeypatch.setattr(import_cells, "_write_manifest", boom)
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
    real = import_cells._sha256

    def counting(path):
        if os.path.normpath(path) == os.path.normpath(ckpt):
            calls.append(path)
        return real(path)

    monkeypatch.setattr(import_cells, "_sha256", counting)
    monkeypatch.setattr(names, "file_sha256", counting)
    result, violations = import_cells.import_run(nas, "cyl", "025", str(checkout), digest,
                                                 expect_n=N_ITEMS)
    assert violations == [] and len(result["files"]) == 10
    assert len(calls) == 1


def test_a_checkpoint_swapped_mid_import_publishes_nothing(tmp_path, monkeypatch):
    nas, _, digest = make_run(tmp_path)
    checkout = tmp_path / "flac"
    checkout.mkdir()
    identities = iter([(1, 2, 3, 4), (1, 2, 3, 5)])
    monkeypatch.setattr(import_cells, "_file_identity", lambda path: next(identities))
    result, violations = import_cells.import_run(nas, "cyl", "025", str(checkout), digest,
                                                 expect_n=N_ITEMS)
    assert any("changed while" in v for v in violations)
    _nothing_was_copied(result, str(checkout))
