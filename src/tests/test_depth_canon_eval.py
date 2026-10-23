"""exp_25 depth_canon round 3a -- the evaluator (``eval_FLAC.py``): filename token,
record provenance, checkpoint protocol binding, canonical sidecar, metadata
preparation and the scored-waveform dump (plan_depth_canon.md v8 §2.2, §5, §6
tests 16-21, 21b).

Written by the exp_25 Coder seat (Claude Opus, max effort) per plan_depth_canon.md v8 §2.2/§5/§6/§8.

CPU only. Synthetic depth goes through the REAL AR_md converter
(``test_depth_canon``'s helpers); the legacy byte pins are the exp_14 golden
fixture and ``test_eval_paths``'s exp_01 names, reused verbatim.
"""
import copy
import itertools
import json

import pytest

import eval_FLAC
from src.data.yaw_rotation import TIE_CLASSES
from test_exp14_fixed_mode_snapshot import MATRIX, META, _record_kwargs

CKPT = "weights/FLAC/FLAC_EMA.ckpt"
SHA_A, SHA_B = "a" * 64, "b" * 64


# --------------------------------------------------------------------------- #
# 16. canon_suffix + build_output_paths
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("applied, mismatch, token", [
    (False, False, ""), (True, False, "_canon"), (False, True, "_canonmis"),
    (True, True, "_canon_canonmis")])
def test_canon_suffix_tokens(applied, mismatch, token):
    assert eval_FLAC.canon_suffix(applied, mismatch) == token


@pytest.mark.parametrize("bad", [1, 0, None, "true"])
def test_canon_suffix_takes_literal_bools_only(bad):
    with pytest.raises(ValueError, match="bool"):
        eval_FLAC.canon_suffix(bad, False)
    with pytest.raises(ValueError, match="bool"):
        eval_FLAC.canon_suffix(False, bad)


@pytest.mark.parametrize("case", MATRIX, ids=lambda c: f'{c["cond_method"]}_{c["rotate_deg"]}')
def test_explicit_non_canonical_paths_are_the_golden_bytes(case):
    paths = eval_FLAC.build_output_paths(
        META["ckpt_path"], META["steps"], META["cfg_scale"], META["eval_name"],
        cond_method=case["cond_method"], rotate_deg=case["rotate_deg"],
        n_angles=case["n_angles"], canonicalize=False, canon_mismatch=False)
    assert paths == case["output_paths"]


def test_exp01_names_are_unchanged_with_explicit_defaults():
    paths = eval_FLAC.build_output_paths(CKPT, 1, 1.0, "exp01_unseen_K1_seed42",
                                         canonicalize=False, canon_mismatch=False)
    assert paths == {
        "metrics": "weights/FLAC/FLAC_EMA_metrics_1_1.0_exp01_unseen_K1_seed42.json",
        "predictions": "weights/FLAC/FLAC_EMA_predictions_1_1.0_exp01_unseen_K1_seed42.pt"}


def test_canon_token_sits_after_the_method_and_before_rotation_and_are():
    paths = eval_FLAC.build_output_paths(
        CKPT, 1, 1.0, "e", cond_method="fa_invariant", rotate_deg=90.0, are_lambda=0.5,
        canonicalize=True, canon_mismatch=True)
    assert paths["metrics"] == \
        "weights/FLAC/FLAC_EMA_metrics_1_1.0_e_fa_invariant_a4_canon_canonmis_rot90_are0p5.json"
    paths = eval_FLAC.build_output_paths(CKPT, 1, 1.0, "e", rotate_mode="random",
                                         rotate_seed=42, canon_mismatch=True)
    assert paths["predictions"] == "weights/FLAC/FLAC_EMA_predictions_1_1.0_e_canonmis_rotrand42.pt"


ROTATIONS = [dict(rotate_deg=0.0), dict(rotate_deg=90.0),
             dict(rotate_mode="random", rotate_seed=42)]
COMBOS = list(itertools.product([False, True], [False, True], range(len(ROTATIONS)),
                                ["vanilla", "fa_invariant"], [None, 0.0, 0.5]))


def _combo_paths(applied, mismatch, rot, cond_method, lam):
    return eval_FLAC.build_output_paths(
        CKPT, 1, 1.0, "cell", cond_method=cond_method, are_lambda=lam,
        canonicalize=applied, canon_mismatch=mismatch, **ROTATIONS[rot])


def test_names_are_injective_over_every_protocol_combination():
    """canonical x mismatch x fixed/random x vanilla/FA x ARE: 72 cells, 72 names each."""
    paths = [_combo_paths(*c) for c in COMBOS]
    for kind in ("metrics", "predictions"):
        assert len({p[kind] for p in paths}) == len(COMBOS) == 72
    for p in paths:  # metrics and predictions differ only by kind + extension
        assert p["metrics"][:-5].replace("_metrics_", "_") == \
            p["predictions"][:-3].replace("_predictions_", "_")


# --------------------------------------------------------------------------- #
# 17. build_metrics_record(..., canon_provenance=None)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", MATRIX, ids=lambda c: f'{c["cond_method"]}_{c["rotate_deg"]}')
def test_record_without_provenance_is_the_golden_record(case, monkeypatch):
    monkeypatch.setattr(eval_FLAC, "source_sha", lambda: META["source_sha_stub"])
    args = (META["metrics"], META["ckpt_path"], case["rotate_deg"], case["cond_method"],
            case["frame_avg_angles"])
    explicit = eval_FLAC.build_metrics_record(*args, canon_provenance=None, **_record_kwargs(case))
    default = eval_FLAC.build_metrics_record(*args, **_record_kwargs(case))
    assert explicit == default and list(explicit) == list(default) == case["metrics_record_keys"]
    assert json.dumps(explicit, indent=4) == case["metrics_record_json_indent4"]


def _prov(applied=True, trained=True, **over):
    prov = {"canonicalize_trained": trained, "canonicalize_applied": applied,
            "canonicalize_mismatch": trained != applied, "canonicalize_img_w": 512,
            "canonicalize_tiebreak": "image" if applied else None,
            "canonicalize_tie_counts": ({"none": 6000, "same_column": 100,
                                         "cross_column_image": 237} if applied else None),
            "model_config_sha256_file": SHA_A, "model_config_sha256_embedded": SHA_B}
    prov.update(over)
    return prov


@pytest.mark.parametrize("applied, trained", [(True, True), (False, True), (True, False),
                                              (False, False)])
def test_record_with_provenance_appends_the_eight_fields(applied, trained):
    base = eval_FLAC.build_metrics_record({"T60": 1.0}, CKPT, 0.0, "vanilla", None)
    rec = eval_FLAC.build_metrics_record({"T60": 1.0}, CKPT, 0.0, "vanilla", None,
                                         canon_provenance=_prov(applied, trained))
    assert list(rec) == list(base) + list(eval_FLAC.CANON_RECORD_KEYS)
    assert len(eval_FLAC.CANON_RECORD_KEYS) == 8
    assert {k: rec[k] for k in base} == base
    for key in ("canonicalize_trained", "canonicalize_applied", "canonicalize_mismatch"):
        assert type(rec[key]) is bool
    assert rec["canonicalize_mismatch"] is (trained != applied)
    assert type(rec["canonicalize_img_w"]) is int
    if applied:
        assert rec["canonicalize_tiebreak"] == "image"
        assert list(rec["canonicalize_tie_counts"]) == list(TIE_CLASSES)
    else:
        assert rec["canonicalize_tiebreak"] is None and rec["canonicalize_tie_counts"] is None
    assert rec["model_config_sha256_file"] == SHA_A
    assert rec["model_config_sha256_embedded"] == SHA_B
    json.loads(json.dumps(rec))


@pytest.mark.parametrize("over", [
    {"canonicalize_mismatch": True}, {"canonicalize_applied": 1}, {"canonicalize_img_w": True},
    {"canonicalize_img_w": 0}, {"canonicalize_tiebreak": None}, {"canonicalize_tie_counts": None},
    {"canonicalize_tie_counts": {"none": 1}}, {"canonicalize_tie_counts": {
        "none": 1, "same_column": -1, "cross_column_image": 0}},
    {"model_config_sha256_file": "A" * 64}, {"model_config_sha256_embedded": None},
    {"extra_key": 1}])
def test_malformed_provenance_is_refused(over):
    with pytest.raises(ValueError, match="canon"):
        eval_FLAC.build_metrics_record({"T60": 1.0}, CKPT, 0.0, "vanilla", None,
                                       canon_provenance=_prov(**over))


def test_record_does_not_alias_the_provenance_dict():
    prov = _prov()
    before = copy.deepcopy(prov)
    rec = eval_FLAC.build_metrics_record({}, CKPT, 0.0, "vanilla", None, canon_provenance=prov)
    rec["canonicalize_tie_counts"]["none"] += 1
    assert prov == before


# --------------------------------------------------------------------------- #
# 18. resolve_canonicalize_protocol (+ the two config hashes)
# --------------------------------------------------------------------------- #
ON, OFF = {"canonicalize": {"enabled": True, "img_w": 512}}, {"canonicalize": {"enabled": False}}


def _model_cfg(training=None, widths=(512, 512)):
    vits = [{"id": f"vit{i}", "type": "ViTCoordinates", "config": {"ViT": {"img_h": 256, "img_w": w}}}
            for i, w in enumerate(widths)]
    return {"model": {"conditioning": {"configs": [
        {"id": "source", "type": "dist_embedder", "config": {}}] + vits}},
            "training": copy.deepcopy(training) if training is not None else {}}


def _resolve(flag=False, file_t=None, emb_t=None, allow=False, cond="vanilla", widths=(512, 512)):
    return eval_FLAC.resolve_canonicalize_protocol(
        flag, copy.deepcopy(file_t), copy.deepcopy(emb_t), allow, cond,
        file_model_cfg=_model_cfg(file_t, widths))


# --------------------------------------------------------------------------- #
# 18 (helpers). CanonProtocol, type-strict diff, ViT width, config digest
# --------------------------------------------------------------------------- #
def test_legacy_protocol_constant():
    assert eval_FLAC.CanonProtocol._fields == (
        "in_play", "trained", "applied", "mismatch", "img_w", "file_block", "embedded_block")
    assert eval_FLAC.LEGACY_CANON_PROTOCOL == (False, None, False, False, None, None, None)


@pytest.mark.parametrize("a, b, needle", [
    ({"enabled": True, "img_w": 512}, {"enabled": True, "img_w": 512}, None),
    ({"enabled": 1}, {"enabled": True}, "x.enabled: type int != bool"),
    ({"img_w": 512.0}, {"img_w": 512}, "x.img_w: type float != int"),
    ({"enabled": True}, {"enabled": True, "img_w": 512}, "key sets differ (['img_w'])"),
    ({"a": [1, 2]}, {"a": [1]}, "x.a: length 2 != 1"), ({"a": [1, {"b": 2}]}, {"a": [1, {"b": 3}]},
                                                        "x.a[1].b: 2 != 3"),
    ({"a": None}, {"a": False}, "type NoneType != bool")])
def test_canon_strict_diff(a, b, needle):
    got = eval_FLAC._canon_strict_diff(a, b, "x")
    assert got is None if needle is None else needle in got


@pytest.mark.parametrize("widths, want", [((512, 512), 512), ((32,), 32), ((512, 256), "disagree on img_w"),
                                          ((), "no ViT conditioner"), ((512.0, 512.0), "positive int"),
                                          ((True,), "positive int"), ((0,), "positive int")])
def test_canon_vit_img_w(widths, want):
    cfg = _model_cfg(None, widths)
    if isinstance(want, int):
        assert eval_FLAC.canon_vit_img_w(cfg) == want
    else:
        with pytest.raises(ValueError, match=want):
            eval_FLAC.canon_vit_img_w(cfg)
    for empty in (None, {}, {"model": {"conditioning": {"configs": [None, {"config": None}]}}}):
        with pytest.raises(ValueError, match="no ViT conditioner"):
            eval_FLAC.canon_vit_img_w(empty)


def test_model_config_sha256_is_the_canonical_json_digest():
    import hashlib
    cfg = {"b": [1, 2.0, True], "a": {"y": None, "x": "s"}}
    want = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert eval_FLAC.model_config_sha256(cfg) == want
    assert eval_FLAC.model_config_sha256({"a": {"x": "s", "y": None}, "b": [1, 2.0, True]}) == want
    assert eval_FLAC.model_config_sha256({"a": 1}) != eval_FLAC.model_config_sha256({"a": 1.0}) \
        != eval_FLAC.model_config_sha256({"a": True})
    assert eval_FLAC.model_config_sha256(None) is None


@pytest.mark.parametrize("file_t, emb_t", [({}, {}), ({}, None), (None, None), ({"x": 1}, {})])
def test_not_in_play_is_the_legacy_tuple(file_t, emb_t):
    legacy = eval_FLAC.CanonProtocol(False, None, False, False, None, None, None)
    assert _resolve(False, file_t, emb_t) == legacy == eval_FLAC.LEGACY_CANON_PROTOCOL


@pytest.mark.parametrize("kw, want", [
    (dict(flag=True, file_t=ON, emb_t=ON), (True, True, False)),           # the CANON table cell
    (dict(file_t=OFF, emb_t=OFF), (False, False, False)),                   # in play, nothing applied
    (dict(file_t=ON, emb_t=ON, allow=True), (True, False, True)),          # D-i: trained, not applied
    (dict(flag=True, file_t={}, emb_t={}, allow=True), (False, True, True)),  # D-ii: P1 + canon
    (dict(flag=True, file_t=OFF, emb_t=OFF, allow=True), (False, True, True))])
def test_accepted_protocols(kw, want):
    p = _resolve(**kw)
    assert p.in_play is True and (p.trained, p.applied, p.mismatch) == want
    assert p.img_w == 512 and type(p.img_w) is int
    file_t, emb_t = kw["file_t"], kw["emb_t"]
    assert p.file_block == file_t.get("canonicalize") and p.embedded_block == emb_t.get("canonicalize")


@pytest.mark.parametrize("kw, needle", [
    (dict(flag=True, file_t={}, emb_t={}), "TRAINED with canonicalize=False"),                      # flag without any config
    (dict(file_t=ON, emb_t=ON), "TRAINED with canonicalize=True"),                                 # config without flag
    (dict(flag=True, file_t=ON, emb_t=None), "no embedded model_config"),                   # missing embedded config
    (dict(file_t=ON, emb_t=None), "no embedded model_config"),
    (dict(flag=True, file_t={}, emb_t=None, allow=True), "no embedded model_config"),
    (dict(file_t=OFF, emb_t={}), "present only in"),                                 # absent vs {enabled:false}
    (dict(file_t={}, emb_t=ON), "present only in"),                                  # embedded-only, no flags
    (dict(file_t={}, emb_t=OFF), "present only in"),
    (dict(flag=True, file_t=ON, emb_t={"canonicalize": {"enabled": 1, "img_w": 512}}), "type int != bool"),
    (dict(flag=True, file_t=ON, emb_t={"canonicalize": {"enabled": True, "img_w": 512.0}}), "type float != int"),
    (dict(flag=True, file_t=ON, emb_t={"canonicalize": {"enabled": True}}), "key sets differ"),
    (dict(file_t=OFF, emb_t=ON, allow=True), r"disagree \(type-strict\)"),                # disagree, allow given
    (dict(flag=True, file_t=OFF, emb_t=ON, allow=True), r"disagree \(type-strict\)"),
    (dict(flag=True, file_t=ON, emb_t=ON, allow=True), "unnecessary"),      # allow, trained == applied
    (dict(file_t={}, emb_t={}, allow=True), "unnecessary"),
    (dict(file_t=OFF, emb_t=OFF, allow=True), "unnecessary"),
    (dict(flag=True, file_t=ON, emb_t=ON, cond="fa_invariant"), "fa_invariant"),
    (dict(flag=True, file_t={}, emb_t={}, allow=True, cond="fa_invariant"), "fa_invariant"),
    (dict(flag=True, file_t=ON, emb_t=ON, widths=(256, 256)), "!= the ViT img_w 256"),     # block vs ViT width
    (dict(flag=True, file_t={}, emb_t={}, allow=True, widths=(512, 256)), "disagree on img_w"),
    (dict(flag=True, file_t=ON, emb_t=ON, widths=()), "no ViT conditioner"),
    (dict(flag=True, file_t={}, emb_t={}, allow=True, widths=(512.0, 512.0)), "ViT img_w must be a positive int"),
    (dict(flag=True, file_t={"canonicalize": {"enabled": True, "img_w": 512, "seed": 1}},
          emb_t={"canonicalize": {"enabled": True, "img_w": 512, "seed": 1}}), "unknown key"),
    (dict(flag=1, file_t=ON, emb_t=ON), "--canonicalize must be a bool"),
    (dict(file_t=ON, emb_t=ON, allow=0), "--allow-canon-mismatch must be a bool")])
def test_refused_protocols(kw, needle):
    with pytest.raises(ValueError, match=needle):
        _resolve(**kw)


def test_mismatch_message_names_both_sides():
    with pytest.raises(ValueError) as err:
        _resolve(file_t=ON, emb_t=ON)
    assert "canonicalize=True" in str(err.value) and "canonicalize=False" in str(err.value)


def test_protocol_blocks_are_copies():
    file_t, emb_t = copy.deepcopy(ON), copy.deepcopy(ON)
    p = eval_FLAC.resolve_canonicalize_protocol(True, file_t, emb_t, False, "vanilla",
                                                file_model_cfg=_model_cfg(ON))
    file_t["canonicalize"]["img_w"] = emb_t["canonicalize"]["img_w"] = 1
    assert p.file_block == p.embedded_block == ON["canonicalize"]
