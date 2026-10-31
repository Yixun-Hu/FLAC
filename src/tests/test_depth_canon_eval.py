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
import hashlib
import itertools
import json
import os

import pytest
import torch

import eval_FLAC
from src.data import yaw_rotation as yr
from src.data.yaw_rotation import TIE_CLASSES
from test_depth_canon import _cloud, _scalar_depth  # synthetic depth via the real AR_md
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
        flag, copy.deepcopy(file_t), copy.deepcopy(emb_t), allow, cond_method=cond,
        file_model_cfg=_model_cfg(file_t, widths))


def test_cond_method_and_file_model_cfg_are_keyword_only():
    """r3a hardening: the positional sixth-argument shape was easy to miscall."""
    with pytest.raises(TypeError):
        eval_FLAC.resolve_canonicalize_protocol(True, ON, ON, False, "vanilla",
                                                file_model_cfg=_model_cfg(ON))
    with pytest.raises(TypeError):
        eval_FLAC.resolve_canonicalize_protocol(True, ON, ON, False)        # cond_method required


@pytest.mark.parametrize("cfg", [None, [], "cfg"])
def test_in_play_needs_the_file_model_config(cfg):
    with pytest.raises(ValueError, match="file_model_cfg"):
        eval_FLAC.resolve_canonicalize_protocol(True, ON, ON, False, cond_method="vanilla",
                                                file_model_cfg=cfg)
    assert eval_FLAC.resolve_canonicalize_protocol(              # not in play: never consulted
        False, {}, {}, False, cond_method="vanilla", file_model_cfg=cfg) == eval_FLAC.LEGACY_CANON_PROTOCOL


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


@pytest.mark.parametrize("widths", [(512, 256), (512, 512)], ids=["disagreeing", "agreeing"])
def test_canon_vit_img_w_refuses_duplicate_conditioner_ids(widths):
    """r3a NIT: widths keyed by id let a duplicate id hide a disagreement (512 then 256 -> 256)."""
    cfg = _model_cfg(None, widths)
    for entry in cfg["model"]["conditioning"]["configs"][1:]:
        entry["id"] = "context_poses_vit"
    with pytest.raises(ValueError, match="duplicate conditioner id"):
        eval_FLAC.canon_vit_img_w(cfg)


def test_model_config_sha256_is_the_canonical_json_digest():
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
    p = eval_FLAC.resolve_canonicalize_protocol(True, file_t, emb_t, False, cond_method="vanilla",
                                                file_model_cfg=_model_cfg(ON))
    file_t["canonicalize"]["img_w"] = emb_t["canonicalize"]["img_w"] = 1
    assert p.file_block == p.embedded_block == ON["canonicalize"]


# --------------------------------------------------------------------------- #
# 20. CanonStream + the .canon.json sidecar
# --------------------------------------------------------------------------- #
W16 = 32  # small panoramas (16 x 32) through the real AR_md converter
PEAKS = {"none": [(3, 9)], "same_column": [(2, 5), (13, 5)],
         "cross_column_image": [(4, 31), (4, 0)]}
CLASSES = ["none", "cross_column_image", "same_column", "none", "cross_column_image"]


def _eval_md(i, tie_class="none", period=None):
    d = _scalar_depth(16, W16, seed=100 + i, period=period,
                      peaks=[(h, (w + 3 * i) % W16) for h, w in PEAKS[tie_class]])
    s = torch.tensor([1.3, -0.7 + 0.1 * i, 0.2])
    ctx = torch.tensor([[0.9, 1.1, -0.3], [-1.2, 0.4 + 0.05 * i, 0.1]])
    return {"depth": _cloud(d), "source": s, "source_vit": s[None], "context_poses": ctx,
            "context_poses_vit": ctx, "context_audio": torch.zeros(2, 1, 8),
            "padding_mask": torch.ones(8), "scene": "Toy/Toy_idx_0", "idx": i,
            "relpath": f"Toy/Toy_idx_0/S00{i}_R001_hybrid_IR.wav"}


def _batch(n=5):
    return [_eval_md(i, CLASSES[i % len(CLASSES)]) for i in range(n)]


def _canon_stream_of(batch, roll=0):
    stream = eval_FLAC.CanonStream()
    for md in batch:
        md = yr.rotate_scene_metadata(md, yr.offsets_to_radians([roll], W16)[0], W16) if roll else md
        md_canon, dj, tie_class = yr.canonicalize_scene_metadata(md, W16)
        stream.record(md, md_canon, dj, tie_class)
    return stream


def test_canon_stream_rows_are_the_per_position_tuples():
    batch = _batch()
    stream = _canon_stream_of(batch)
    assert len(stream) == 5
    for i, (md, row) in enumerate(zip(batch, stream.position_tuples())):
        md_canon, dj, tie_class = yr.canonicalize_scene_metadata(md, W16)
        assert row == [i, eval_FLAC.sample_target_id(md), dj, tie_class,
                       yr.canonical_depth_sha256(md_canon["depth"])]
        assert tie_class == CLASSES[i]
    assert stream.target_ids == [eval_FLAC.sample_target_id(md) for md in batch]
    assert stream.tie_counts() == {"none": 2, "same_column": 1, "cross_column_image": 2}
    assert list(stream.tie_counts()) == list(TIE_CLASSES)


def test_canon_hash_is_recomputable_from_the_preimages():
    stream = _canon_stream_of(_batch())
    lines = "\n".join(json.dumps(t, sort_keys=True, separators=(",", ":"))
                      for t in stream.position_tuples())
    assert stream.canon_hash() == hashlib.sha256(lines.encode("utf-8")).hexdigest()
    rec = json.loads(json.dumps(eval_FLAC.build_canon_record(stream, W16)))
    assert rec == {"schema_version": 1, "img_w": W16, "tiebreak": "image", "stream_count": 5,
                   "tie_counts": stream.tie_counts(), "positions": stream.position_tuples(),
                   "canon_hash": stream.canon_hash(),
                   "canonical_depth_hash": stream.canonical_depth_hash()}
    assert eval_FLAC.canonical_stream_hash(rec["positions"]) == rec["canon_hash"]
    assert eval_FLAC.canonical_stream_hash([[p[1], p[4]] for p in rec["positions"]]) == \
        rec["canonical_depth_hash"]                    # recomputable from the same preimages


@pytest.mark.parametrize("roll", [1, 8, 16, 31])
def test_canonical_depth_hashes_are_equal_across_a_rolled_copy(roll):
    base, rolled = _canon_stream_of(_batch()), _canon_stream_of(_batch(), roll)
    for a, b in zip(base.position_tuples(), rolled.position_tuples()):
        assert a[:2] == b[:2] and a[3:] == b[3:]       # position, target, class, depth sha
        assert b[2] == (a[2] - roll) % W16             # the residual shift co-rotates
    assert base.canon_hash() != rolled.canon_hash()    # dj is part of the cell digest
    assert base.canonical_depth_hash() == rolled.canonical_depth_hash()   # Tier 1, one line


@pytest.mark.parametrize("edit", ["reorder", "target", "one_hash"])
def test_canonical_depth_hash_is_sensitive_to_order_targets_and_each_hash(edit):
    base = _canon_stream_of(_batch())
    changed = copy.deepcopy(base)
    if edit == "reorder":
        changed.rows[0], changed.rows[1] = changed.rows[1], changed.rows[0]
    elif edit == "target":
        changed.rows[2][1] = "999|single_channel_ir_1/Other/Other_idx_0/S001_R001_hybrid_IR.wav"
    else:
        changed.rows[3][4] = "0" * 64
    assert changed.canonical_depth_hash() != base.canonical_depth_hash()


def test_canon_sidecar_path():
    m = "/o/epoch=8-step=40000_metrics_1_1.0_cell_canon.json"
    assert eval_FLAC.canon_sidecar_path(m) == "/o/epoch=8-step=40000_metrics_1_1.0_cell_canon.canon.json"
    assert eval_FLAC.canon_sidecar_path(m) != eval_FLAC.stream_sidecar_path(m)


@pytest.mark.parametrize("dj, tie_class", [(-1, "none"), (True, "none"), (1.0, "none"),
                                           (0, "stabilized"), (0, None)])
def test_canon_stream_refuses_malformed_rows(dj, tie_class):
    md = _eval_md(0)
    md_canon, _, _ = yr.canonicalize_scene_metadata(md, W16)
    with pytest.raises(ValueError, match="canon stream"):
        eval_FLAC.CanonStream().record(md, md_canon, dj, tie_class)


def test_canon_record_refuses_a_shift_outside_the_grid():
    stream = _canon_stream_of(_batch(2))
    with pytest.raises(ValueError, match="img_w"):
        eval_FLAC.build_canon_record(stream, 4)
    with pytest.raises(ValueError, match="img_w"):
        eval_FLAC.build_canon_record(stream, True)


# --------------------------------------------------------------------------- #
# 19. prepare_eval_metadata (rotation -> canonicalization) + build_canon_provenance
# --------------------------------------------------------------------------- #
FIXED0 = eval_FLAC.resolve_rotation_plan("fixed", 0.0, None)
FIXED90 = eval_FLAC.resolve_rotation_plan("fixed", 90.0, None)   # 8 columns of 32
RANDOM7 = eval_FLAC.resolve_rotation_plan("random", 0.0, 7)


def _prepare(batch, plan=FIXED0, stream=None, canon_stream=None, canonicalize=True, seed=7):
    gen = torch.Generator().manual_seed(seed) if plan.is_random else None
    if plan.is_random and stream is None:
        stream = eval_FLAC.RotationStream()
    if canonicalize and canon_stream is None:
        canon_stream = eval_FLAC.CanonStream()
    out = eval_FLAC.prepare_eval_metadata(batch, plan, gen, stream, canonicalize, W16, canon_stream)
    return out.cond, stream, canon_stream


def _prepare_full(batch, plan=FIXED0, canonicalize=True, seed=7):
    """prepare_eval_metadata's whole result: (cond, metric) frames."""
    gen = torch.Generator().manual_seed(seed) if plan.is_random else None
    stream = eval_FLAC.RotationStream() if plan.is_random else None
    canon = eval_FLAC.CanonStream() if canonicalize else None
    return eval_FLAC.prepare_eval_metadata(batch, plan, gen, stream, canonicalize, W16, canon)


def test_prepare_returns_the_canonical_conditioning_and_the_pre_canonical_metric_frame():
    """r3a BLOCKING (decision A4): only the conditioner sees the canonical frame."""
    batch = _batch()
    prepared = _prepare_full(batch, FIXED90)
    assert isinstance(prepared, eval_FLAC.PreparedMetadata) and prepared.metric is not prepared.cond
    rotated = eval_FLAC.apply_rotation_plan(batch, FIXED90)
    for metric, rot, cond in zip(prepared.metric, rotated, prepared.cond):
        assert all(torch.equal(metric[k], rot[k]) for k in ("depth",) + yr.POSE_KEYS)
        assert _j_star(cond) == 0
    assert any(_j_star(md) != 0 for md in prepared.metric)


@pytest.mark.parametrize("plan", [FIXED0, FIXED90, RANDOM7], ids=["z", "rot90", "random"])
def test_disabled_metric_frame_is_the_conditioning_list_itself(plan):
    batch = _batch()
    prepared = _prepare_full(batch, plan, canonicalize=False)
    assert prepared.metric is prepared.cond
    assert (prepared.cond is batch) is (plan is FIXED0)


@pytest.mark.parametrize("plan", [FIXED0, FIXED90, RANDOM7], ids=["z", "rot90", "random"])
def test_rotation_runs_first_then_each_sample_is_canonicalized(plan, monkeypatch):
    calls, rotated_out = [], []
    real_rot, real_canon = eval_FLAC.apply_rotation_plan, eval_FLAC.canonicalize_scene_metadata

    def rot_spy(*a, **k):
        calls.append("rotate")
        rotated_out.extend(real_rot(*a, **k))
        return list(rotated_out)

    def canon_spy(md, img_w):
        calls.append("canon")
        assert md is rotated_out[calls.count("canon") - 1] and img_w == W16
        return real_canon(md, img_w)

    monkeypatch.setattr(eval_FLAC, "apply_rotation_plan", rot_spy)
    monkeypatch.setattr(eval_FLAC, "canonicalize_scene_metadata", canon_spy)
    batch = _batch()
    before = copy.deepcopy(batch)
    out, _, canon = _prepare(batch, plan)
    assert calls == ["rotate"] + ["canon"] * 5
    for md_in, md_out, row in zip(rotated_out, out, canon.position_tuples()):
        want, dj, tie_class = real_canon(md_in, W16)
        assert row[2:4] == [dj, tie_class]
        assert yr.canonical_column(yr.recover_depth_proxy(md_out["depth"]))[0] == 0
        for key in want:
            assert torch.equal(want[key], md_out[key]) if torch.is_tensor(want[key]) \
                else want[key] == md_out[key]
    for md, ref in zip(batch, before):                 # the loader's dicts are not mutated
        assert all(torch.equal(md[k], ref[k]) if torch.is_tensor(ref[k]) else md[k] == ref[k]
                   for k in ref)


def test_disabled_is_the_rotation_alone():
    batch = _batch()
    out, _, canon = _prepare(batch, FIXED0, canonicalize=False)
    assert out is batch and canon is None
    out90, _, _ = _prepare(batch, FIXED90, canonicalize=False)
    want = eval_FLAC.apply_rotation_plan(batch, FIXED90)
    assert all(torch.equal(a["depth"], b["depth"]) for a, b in zip(out90, want))


@pytest.mark.parametrize("plan", [FIXED90, RANDOM7], ids=["rot90", "random"])
def test_canonical_depth_hashes_match_the_unrotated_cell(plan):
    _, _, base = _prepare(_batch(), FIXED0)
    _, rot_stream, rolled = _prepare(_batch(), plan, stream=eval_FLAC.RotationStream())
    assert [r[4] for r in base.position_tuples()] == [r[4] for r in rolled.position_tuples()]
    assert base.tie_counts() == rolled.tie_counts() == {"none": 2, "same_column": 1,
                                                       "cross_column_image": 2}
    offsets = rot_stream.offsets
    assert [(b[2] - d) % W16 for b, d in zip(base.position_tuples(), offsets)] == \
        [r[2] for r in rolled.position_tuples()]
    assert rot_stream.target_ids == rolled.target_ids     # the two sidecars join by position


def test_positions_run_on_across_batches():
    canon = eval_FLAC.CanonStream()
    _prepare(_batch(3), canon_stream=canon)
    _prepare([_eval_md(i + 3, CLASSES[i + 3]) for i in range(2)], canon_stream=canon)
    assert [r[0] for r in canon.position_tuples()] == [0, 1, 2, 3, 4]
    assert canon.target_ids == [eval_FLAC.sample_target_id(md) for md in _batch()]


@pytest.mark.parametrize("mutate, needle", [
    (lambda md: md.update(depth=md["depth"].to("meta")), r"metadata\[1\]\['depth'\] is on meta"),
    (lambda md: md.update(depth=md["depth"].float()), r"metadata\[1\]\['depth'\] is torch.float32"),
    (lambda md: md.pop("source_vit"), r"^canonicalize: metadata\[1\]: .*source_vit"),
    (lambda md: md.update(depth=md["depth"][:, :, :16]), r"^canonicalize: metadata\[1\]: depth width")])
def test_contract_violations_are_refused_with_the_wrapper_prefix(mutate, needle):
    batch = _batch(2)
    mutate(batch[1])
    with pytest.raises(ValueError, match=needle) as err:
        _prepare(batch)
    assert str(err.value).startswith("canonicalize: metadata[1]")


def test_operator_errors_keep_their_cause_and_ties_propagate_untouched():
    batch = _batch(2)
    batch[1].pop("source")
    with pytest.raises(ValueError) as err:
        _prepare(batch)
    assert isinstance(err.value.__cause__, ValueError)
    periodic = [_eval_md(0), _eval_md(1, period=W16 // 2)]
    with pytest.raises(yr.StabilizedTieError) as tie:
        _prepare(periodic)
    assert type(tie.value) is yr.StabilizedTieError and tie.value.sample["idx"] == 1


@pytest.mark.parametrize("kw, needle", [
    (dict(canonicalize=True, img_w=W16, canon_stream=None), "CanonStream"),
    (dict(canonicalize=False, img_w=None, canon_stream="stream"), "without canonicalization"),
    (dict(canonicalize=1, img_w=W16, canon_stream="stream"), "bool"),
    (dict(canonicalize=True, img_w=32.0, canon_stream="stream"), "img_w")])
def test_prepare_argument_contract(kw, needle):
    if kw["canon_stream"] == "stream":
        kw["canon_stream"] = eval_FLAC.CanonStream()
    with pytest.raises(ValueError, match=needle):
        eval_FLAC.prepare_eval_metadata(_batch(1), FIXED0, None, None, kw["canonicalize"],
                                        kw["img_w"], kw["canon_stream"])


def test_build_canon_provenance():
    file_cfg, emb_cfg = _model_cfg(ON), _model_cfg(ON)
    emb_cfg["training"]["use_ema"] = True                 # the two digests are independent
    _, _, canon = _prepare(_batch())
    applied = _resolve(True, ON, ON)
    prov = eval_FLAC.build_canon_provenance(applied, canon, file_cfg, emb_cfg)
    assert prov == {"canonicalize_trained": True, "canonicalize_applied": True,
                    "canonicalize_mismatch": False, "canonicalize_img_w": 512,
                    "canonicalize_tiebreak": "image", "canonicalize_tie_counts": canon.tie_counts(),
                    "model_config_sha256_file": eval_FLAC.model_config_sha256(file_cfg),
                    "model_config_sha256_embedded": eval_FLAC.model_config_sha256(emb_cfg)}
    assert prov["model_config_sha256_file"] != prov["model_config_sha256_embedded"]
    eval_FLAC.build_metrics_record({}, CKPT, 0.0, "vanilla", None, canon_provenance=prov)
    diag = eval_FLAC.build_canon_provenance(_resolve(False, ON, ON, allow=True), None,
                                            file_cfg, file_cfg)
    assert (diag["canonicalize_applied"], diag["canonicalize_mismatch"],
            diag["canonicalize_tiebreak"], diag["canonicalize_tie_counts"]) == (False, True, None, None)
    eval_FLAC.build_metrics_record({}, CKPT, 0.0, "vanilla", None, canon_provenance=diag)
    assert eval_FLAC.build_canon_provenance(eval_FLAC.LEGACY_CANON_PROTOCOL, None, file_cfg,
                                            None) is None
    with pytest.raises(ValueError, match="CanonStream"):
        eval_FLAC.build_canon_provenance(applied, None, file_cfg, emb_cfg)
    with pytest.raises(ValueError, match="CanonStream"):
        eval_FLAC.build_canon_provenance(_resolve(False, ON, ON, allow=True), canon, file_cfg, emb_cfg)


# --------------------------------------------------------------------------- #
# 21. ScoredWaveformDump (announcement 08) + atomic cell completion (plan §5)
# --------------------------------------------------------------------------- #
FILES = ["COMPLETE", "manifest.json", "waveforms.npy", "waveforms.sha256"]
HASH = "c" * 64


def _scored(b, t=16, seed=0):
    return torch.randn(b, 1, t, generator=torch.Generator().manual_seed(seed)).clamp(-1, 1)


def _filled(tmp_path, sizes=(3, 2), t=16):
    dump = eval_FLAC.ScoredWaveformDump(str(tmp_path / "cell"), sum(sizes))
    batches = [_scored(b, t, seed=i) for i, b in enumerate(sizes)]
    for fakes in batches:
        dump.append(fakes)
    return dump, batches


def test_the_dump_is_exactly_the_appended_tensors_in_stream_order(tmp_path):
    import numpy as np
    dump, batches = _filled(tmp_path)
    manifest = dump.finalize(HASH, "ckpt_metrics_1_1.0_cell_canon", {"ckpt_path": "/o/x.ckpt"})
    arr = np.load(tmp_path / "cell" / "waveforms.npy")
    assert arr.dtype == np.float32 and arr.shape == (5, 16)
    assert torch.equal(torch.from_numpy(arr), torch.cat([b[:, 0, :] for b in batches]))
    assert sorted(p.name for p in (tmp_path / "cell").iterdir()) == FILES
    assert not list(tmp_path.glob("cell.staging-*"))
    sha = hashlib.sha256((tmp_path / "cell" / "waveforms.npy").read_bytes()).hexdigest()
    assert (tmp_path / "cell" / "waveforms.sha256").read_text() == f"{sha}  waveforms.npy\n"
    on_disk = json.loads((tmp_path / "cell" / "manifest.json").read_text())
    assert on_disk == manifest
    assert {k: on_disk[k] for k in ("schema_version", "n", "dtype", "shape", "sha256", "input_hash",
                                    "metrics_stem", "ckpt_path")} == {
        "schema_version": 1, "n": 5, "dtype": "float32", "shape": [5, 16], "sha256": sha,
        "input_hash": HASH, "metrics_stem": "ckpt_metrics_1_1.0_cell_canon", "ckpt_path": "/o/x.ckpt"}
    assert on_disk["source_sha"] == eval_FLAC.source_sha()
    assert eval_FLAC.ScoredWaveformDump.is_complete(str(tmp_path / "cell"))


def test_append_stores_a_cpu_copy_of_the_float32_scored_tensor(tmp_path):
    dump = eval_FLAC.ScoredWaveformDump(str(tmp_path / "cell"), 2)
    fakes = _scored(2)
    rows = dump.append(fakes)
    assert rows.dtype == torch.float32 and rows.device.type == "cpu"
    assert torch.equal(rows, fakes[:, 0, :])
    fakes.zero_()                                    # a later in-place change is not dumped
    assert not torch.equal(rows, fakes[:, 0, :])
    for bad in (torch.zeros(2, 2, 16), torch.zeros(2, 16), [0.0], torch.zeros(2, 1, 8)):
        with pytest.raises(ValueError, match="scored"):
            dump.append(bad)


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16, torch.float16])
def test_append_refuses_a_non_float32_scored_tensor(tmp_path, dtype):
    """r3a NIT: a silent cast would weaken 'exactly the tensor handed to update_metrics'."""
    dump = eval_FLAC.ScoredWaveformDump(str(tmp_path / "cell"), 2)
    with pytest.raises(ValueError, match="float32"):
        dump.append(_scored(2).to(dtype))


@pytest.mark.parametrize("setup", [
    lambda d: (d.mkdir(), (d / "waveforms.npy").write_text("x")),   # non-empty directory
    lambda d: d.mkdir(),                                            # EMPTY directory (r3a NIT)
    lambda d: d.write_text("x"),                                    # a file
    lambda d: d.symlink_to(d.parent)], ids=["non_empty", "empty", "file", "symlink"])
def test_any_existing_final_path_is_refused_before_any_work_and_kept(tmp_path, setup):
    final = tmp_path / "cell"
    setup(final)
    before = sorted(p.name for p in tmp_path.iterdir())
    with pytest.raises(ValueError, match="already exists"):
        eval_FLAC.ScoredWaveformDump(str(final), 1)
    assert sorted(p.name for p in tmp_path.iterdir()) == before and os.path.lexists(final)


def test_a_missing_dump_root_is_refused_and_never_created(tmp_path):
    with pytest.raises(ValueError, match="dump root"):
        eval_FLAC.ScoredWaveformDump(str(tmp_path / "missing_root" / "cell"), 1)
    assert not (tmp_path / "missing_root").exists()


@pytest.mark.parametrize("parts", [("r.staging-12", "cell"), ("r", "cell.staging-7")],
                         ids=["root", "cell"])
def test_a_reserved_staging_component_is_refused(tmp_path, parts):
    (tmp_path / parts[0]).mkdir()
    with pytest.raises(ValueError, match="reserved"):
        eval_FLAC.ScoredWaveformDump(str(tmp_path / parts[0] / parts[1]), 1)


@pytest.mark.parametrize("kw, needle", [
    (dict(input_hash=None), "input_hash"), (dict(input_hash="C" * 64), "input_hash"),
    (dict(metrics_stem=""), "metrics_stem"), (dict(extra_meta={"sha256": "x"}), "extra_meta")])
def test_finalize_argument_contract(tmp_path, kw, needle):
    dump, _ = _filled(tmp_path)
    args = dict(input_hash=HASH, metrics_stem="stem", extra_meta=None)
    args.update(kw)
    with pytest.raises(ValueError, match=needle):
        dump.finalize(**args)
    assert not (tmp_path / "cell").exists() and not list(tmp_path.glob("cell.staging-*"))


def test_finalize_refuses_a_short_dump_and_a_final_dir_that_filled_up(tmp_path):
    dump = eval_FLAC.ScoredWaveformDump(str(tmp_path / "cell"), 6)
    dump.append(_scored(5))
    with pytest.raises(ValueError, match="expected 6"):
        dump.finalize(HASH, "stem")
    dump, _ = _filled(tmp_path)
    (tmp_path / "cell").mkdir()                       # appeared after construction, even empty
    with pytest.raises(ValueError, match="already exists"):
        dump.finalize(HASH, "stem")
    assert not list(tmp_path.glob("cell.staging-*")) and (tmp_path / "cell").is_dir()


def test_a_crash_at_the_rename_leaves_a_marked_staging_dir_and_no_final_dir(tmp_path, monkeypatch):
    dump, _ = _filled(tmp_path)

    def crash(*a, **k):
        raise OSError("simulated crash immediately before the rename")

    monkeypatch.setattr(eval_FLAC.os, "rename", crash)
    with pytest.raises(OSError, match="simulated"):
        dump.finalize(HASH, "stem")
    assert not (tmp_path / "cell").exists()
    assert sorted(p.name for p in (tmp_path / f"cell.staging-{eval_FLAC.os.getpid()}").iterdir()) \
        == FILES
    assert not eval_FLAC.ScoredWaveformDump.is_complete(str(tmp_path / "cell"))


def test_a_failed_marker_write_leaves_no_marker_and_no_final_dir(tmp_path, monkeypatch):
    dump, _ = _filled(tmp_path)

    def crash(staging, sha256):
        raise OSError("simulated crash at the marker write")

    monkeypatch.setattr(eval_FLAC, "_write_complete_marker", crash)
    with pytest.raises(OSError, match="simulated"):
        dump.finalize(HASH, "stem")
    staging = tmp_path / f"cell.staging-{eval_FLAC.os.getpid()}"
    assert not (tmp_path / "cell").exists() and not (staging / "COMPLETE").exists()
    assert sorted(p.name for p in staging.iterdir()) == FILES[1:]


def test_verification_failure_writes_no_marker(tmp_path, monkeypatch):
    dump, _ = _filled(tmp_path)
    real_sha = eval_FLAC._file_sha256
    calls = []

    def lying_sha(path):                       # the re-hash disagrees with the first hash
        calls.append(path)
        return real_sha(path) if len(calls) == 1 else "0" * 64

    monkeypatch.setattr(eval_FLAC, "_file_sha256", lying_sha)
    with pytest.raises(RuntimeError, match="verif"):
        dump.finalize(HASH, "stem")
    assert len(calls) == 2 and not (tmp_path / "cell").exists()
    assert not (tmp_path / f"cell.staging-{eval_FLAC.os.getpid()}" / "COMPLETE").exists()


@pytest.mark.parametrize("corrupt", ["one_bit", "float64", "short"])
def test_a_reread_that_differs_from_the_appended_rows_writes_no_marker(tmp_path, monkeypatch, corrupt):
    dump, _ = _filled(tmp_path)
    real_load = eval_FLAC.np.load

    def corrupting_load(path, **kw):
        arr = real_load(path, **kw)
        if corrupt == "one_bit":
            arr = arr.copy()
            arr.view(eval_FLAC.np.uint32)[0, 0] ^= 1          # flip the lowest mantissa bit
        return {"float64": lambda: arr.astype("float64"), "short": lambda: arr[:-1]}.get(
            corrupt, lambda: arr)()

    monkeypatch.setattr(eval_FLAC.np, "load", corrupting_load)
    with pytest.raises(RuntimeError, match="verification"):
        dump.finalize(HASH, "stem")
    assert not (tmp_path / "cell").exists()
    assert not (tmp_path / f"cell.staging-{eval_FLAC.os.getpid()}" / "COMPLETE").exists()


def test_is_complete_verifies_the_whole_cell(tmp_path):
    """r3a HIGH: the evaluator-side predicate verifies every artifact of the cell."""
    import shutil
    import numpy as np
    is_complete = eval_FLAC.ScoredWaveformDump.is_complete
    dump, _ = _filled(tmp_path)
    dump.finalize(HASH, "stem")
    cell = tmp_path / "cell"
    assert is_complete(str(cell)) and is_complete(str(cell), expected_n=5)
    assert not is_complete(str(cell), expected_n=4) and not is_complete(str(tmp_path / "absent"))

    def manifest(d, **over):
        m = json.loads((d / "manifest.json").read_text())
        m.update(over)
        (d / "manifest.json").write_text(json.dumps(m))
    variants = {
        "empty_manifest": lambda d: (d / "manifest.json").write_text("{}"),
        "list_manifest": lambda d: (d / "manifest.json").write_text("[]"),
        "broken_manifest": lambda d: (d / "manifest.json").write_text("{not json"),
        "missing_array": lambda d: (d / "waveforms.npy").unlink(),
        "wrong_n": lambda d: manifest(d, n=6),
        "wrong_shape": lambda d: manifest(d, shape=[5, 17]),
        "wrong_dtype": lambda d: manifest(d, dtype="float64"),
        "wrong_schema": lambda d: manifest(d, schema_version=2),
        "bad_manifest_hash": lambda d: manifest(d, sha256="0" * 64),
        "bad_input_hash": lambda d: manifest(d, input_hash="x"),
        "no_stem": lambda d: manifest(d, metrics_stem=""),
        "bad_checksum": lambda d: (d / "waveforms.sha256").write_text("0" * 64 + "  waveforms.npy\n"),
        "missing_checksum": lambda d: (d / "waveforms.sha256").unlink(),
        "bad_marker_hash": lambda d: (d / "COMPLETE").write_text("0" * 64 + "\n"),
        "missing_marker": lambda d: (d / "COMPLETE").unlink(),
        "array_altered": lambda d: np.save(d / "waveforms.npy", np.zeros((5, 16), np.float32)),
    }
    for name, edit in variants.items():
        shutil.copytree(cell, tmp_path / name)
        edit(tmp_path / name)
        assert not is_complete(str(tmp_path / name)), name
    for bad in (-1, 2.0, True):
        with pytest.raises(ValueError, match="expected_n"):
            is_complete(str(cell), expected_n=bad)


@pytest.mark.parametrize("version", [True, 1.0, "1"], ids=["json_true", "float_1.0", "string_1"])
def test_is_complete_schema_version_is_type_strict(tmp_path, version):
    """Codex r3a-F1 re-verification NIT: ``true`` and ``1.0`` compare equal to the int 1 in
    Python, so the schema check must be ``type(v) is int and v == SCORED_DUMP_SCHEMA_VERSION``."""
    dump, _ = _filled(tmp_path)
    dump.finalize(HASH, "stem")
    cell = tmp_path / "cell"
    assert eval_FLAC.ScoredWaveformDump.is_complete(str(cell))
    m = json.loads((cell / "manifest.json").read_text())
    m["schema_version"] = version
    (cell / "manifest.json").write_text(json.dumps(m))
    assert json.loads((cell / "manifest.json").read_text())["schema_version"] == version
    assert not eval_FLAC.ScoredWaveformDump.is_complete(str(cell))
    assert not eval_FLAC.ScoredWaveformDump.is_complete(str(cell), expected_n=5)


def test_finalize_reads_the_checksum_sidecar_back_before_the_marker(tmp_path, monkeypatch):
    dump, _ = _filled(tmp_path)

    def wrong_sidecar(staging, sha256):
        with open(eval_FLAC.os.path.join(staging, "waveforms.sha256"), "w") as f:
            f.write("0" * 64 + "  waveforms.npy\n")
    monkeypatch.setattr(eval_FLAC, "_write_checksum_sidecar", wrong_sidecar)
    with pytest.raises(RuntimeError, match="verification"):
        dump.finalize(HASH, "stem")
    assert not (tmp_path / "cell").exists()
    assert not (tmp_path / f"cell.staging-{eval_FLAC.os.getpid()}" / "COMPLETE").exists()


def test_dump_directory_names_are_derived_from_the_metrics_stem_and_injective():
    names = {eval_FLAC.scored_waveform_dir("/nas/dumps", _combo_paths(*c)["metrics"]) for c in COMBOS}
    assert len(names) == len(COMBOS)
    p = _combo_paths(True, False, 0, "vanilla", None)["metrics"]
    assert eval_FLAC.scored_waveform_dir("/nas/dumps", p) == \
        "/nas/dumps/FLAC_EMA_metrics_1_1.0_cell_canon"


# --------------------------------------------------------------------------- #
# 19 / 21. evaluate_model wiring: the REAL eval loop on a toy CPU stack
# --------------------------------------------------------------------------- #
ON32 = {"canonicalize": {"enabled": True, "img_w": W16}}


class _Loader:
    def __init__(self, batches):
        self.batches, self.dataset = batches, [None] * sum(len(b[1]) for b in batches)

    def __iter__(self):
        return iter(self.batches)


def _run(monkeypatch, tmp_path, training, embedded="same", raise_at_model=False, events=None,
         **eval_kw):
    import types
    import src.inference.sampling as sampling
    cfg = _model_cfg(training, widths=(W16, W16))
    cfg.update(model_type="diffusion_cond", sample_size=16, sample_rate=22050, audio_channels=1)
    cfg["training"]["use_ema"] = False
    (tmp_path / "model.json").write_text(json.dumps(cfg))
    (tmp_path / "dataset.json").write_text(json.dumps({"datasets": [{"id": "toy"}]}))
    ckpt = {"state_dict": {}}
    if embedded is not None:
        ckpt["model_config"] = copy.deepcopy(cfg) if embedded == "same" else embedded
    torch.save(ckpt, str(tmp_path / "toy.ckpt"))
    batch = _batch()
    loader = _Loader([(torch.zeros(3, 1, 16), batch[:3]), (torch.zeros(2, 1, 16), batch[3:])])
    seen = types.SimpleNamespace(cond=[], scored=[], appended=[], scenes=[], metric_depth=[],
                                 loader=loader)

    log = (lambda e: events.append(e)) if events is not None else (lambda e: None)

    def model_factory(cfg):
        log("model")
        if raise_at_model:
            pytest.fail("evaluate_model reached model construction")
        return types.SimpleNamespace(load_state_dict=lambda sd, strict=False: ([], []))

    module = types.SimpleNamespace(device="cpu", diffusion=types.SimpleNamespace(
        model=types.SimpleNamespace(diffusion_objective="rectified_flow"), pretransform=None,
        conditioner=lambda md, device: log("cond") or seen.cond.append(md) or {}, io_channels=1,
        dist_shift=None,
        get_conditioning_inputs=lambda cond: {}))
    module.eval = module.requires_grad_ = module.to = lambda *a: module
    callback = types.SimpleNamespace(
        compute_metrics=lambda split: {"T60": 1.0},
        update_metrics=lambda stage, pred, reals, scenes, **k: (
            log("score"), seen.scored.append(pred), seen.scenes.append(scenes),
            seen.metric_depth.append(k["depth"])))
    real_append = eval_FLAC.ScoredWaveformDump.append
    monkeypatch.setattr(eval_FLAC.ScoredWaveformDump, "append",
                        lambda self, t: log("append") or seen.appended.append(t) or real_append(self, t))
    monkeypatch.setattr(eval_FLAC, "create_model_from_config", model_factory)
    monkeypatch.setattr(eval_FLAC, "create_training_wrapper_from_config", lambda cfg, m: module)
    monkeypatch.setattr(eval_FLAC, "create_dataloader_from_config", lambda *a, **k: loader)
    monkeypatch.setattr(eval_FLAC, "create_metric_callback_from_config", lambda *a, **k: callback)
    monkeypatch.setattr(sampling, "sample_discrete_euler", lambda model, noise, steps, **kw: noise)
    eval_FLAC.evaluate_model(str(tmp_path / "model.json"), str(tmp_path / "dataset.json"),
                             str(tmp_path / "toy.ckpt"), steps=1, cfg_scale=1.0, batch_size=3,
                             device="cpu", eval_name="cell", seed=42, cond_autocast="off", **eval_kw)
    return cfg, ckpt, seen


def _j_star(md):
    return yr.canonical_column(yr.recover_depth_proxy(md["depth"]))[0]


def test_canon_cell_end_to_end(monkeypatch, tmp_path, capsys):
    import numpy as np
    (tmp_path / "dumps").mkdir()
    cfg, ckpt, seen = _run(monkeypatch, tmp_path, ON32, canonicalize=True, record_stream=True,
                           expected_stream_count=5, dump_scored_waveforms=str(tmp_path / "dumps"))
    assert "canonicalize: trained=True applied=True mismatch=False img_w=32" in capsys.readouterr().out
    stem = "toy_metrics_1_1.0_cell_canon"
    rec = json.loads((tmp_path / f"{stem}.json").read_text())
    assert list(rec)[-8:] == list(eval_FLAC.CANON_RECORD_KEYS)
    assert (rec["canonicalize_trained"], rec["canonicalize_applied"], rec["canonicalize_mismatch"],
            rec["canonicalize_img_w"], rec["canonicalize_tiebreak"]) == (True, True, False, 32, "image")
    assert rec["canonicalize_tie_counts"] == {"none": 2, "same_column": 1, "cross_column_image": 2}
    assert rec["model_config_sha256_file"] == eval_FLAC.model_config_sha256(cfg)
    assert rec["model_config_sha256_embedded"] == eval_FLAC.model_config_sha256(ckpt["model_config"])
    assert all(_j_star(md) == 0 for batch in seen.cond for md in batch)   # canonical BEFORE cond
    stream = json.loads((tmp_path / f"{stem}.stream.json").read_text())
    canon = json.loads((tmp_path / f"{stem}.canon.json").read_text())
    assert [p[1] for p in canon["positions"]] == [t[1] for t in stream["input_tuples"]]
    assert eval_FLAC.canonical_stream_hash(canon["positions"]) == canon["canon_hash"]
    final = tmp_path / "dumps" / stem
    assert eval_FLAC.ScoredWaveformDump.is_complete(str(final))
    assert [a is s for a, s in zip(seen.appended, seen.scored)] == [True, True]
    dumped = torch.from_numpy(np.load(final / "waveforms.npy"))
    assert torch.equal(dumped, torch.cat([s[:, 0, :] for s in seen.scored]))
    manifest = json.loads((final / "manifest.json").read_text())
    assert manifest["input_hash"] == stream["input_hash"] and manifest["metrics_stem"] == stem
    assert manifest["metrics_sha256"] == hashlib.sha256((tmp_path / f"{stem}.json").read_bytes()).hexdigest()
    assert manifest["canon_sidecar_sha256"] == \
        hashlib.sha256((tmp_path / f"{stem}.canon.json").read_bytes()).hexdigest()


def test_legacy_cell_is_untouched(monkeypatch, tmp_path, capsys):
    _, _, seen = _run(monkeypatch, tmp_path, {})
    assert "canonicalize:" not in capsys.readouterr().out
    rec = json.loads((tmp_path / "toy_metrics_1_1.0_cell.json").read_text())
    assert not [k for k in rec if k.startswith(("canonicalize", "model_config_sha256"))]
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "dataset.json", "model.json", "toy.ckpt", "toy_metrics_1_1.0_cell.json"]
    assert [seen.cond[0], seen.cond[1]] == [seen.loader.batches[0][1], seen.loader.batches[1][1]]
    for (_, batch_md), depth in zip(seen.loader.batches, seen.metric_depth):  # legacy metric inputs
        assert all(torch.equal(a, b) for a, b in zip(depth, _metric_inputs(batch_md)))


def _metric_inputs(md_list):
    """What evaluate_model hands update_metrics as geometry, from a given metadata list."""
    return [(md["depth"][:3] - md["source"][:, None, None]).unsqueeze(0).float() for md in md_list]


@pytest.mark.parametrize("kw", [dict(rotate_deg=90.0), dict(rotate_mode="random", rotate_seed=7)],
                         ids=["rot90", "random"])
def test_metric_geometry_is_the_post_rotation_pre_canonical_frame(monkeypatch, tmp_path, kw):
    """r3a BLOCKING (A4): the AGREE retrieval gallery must be arm-independent."""
    _, _, seen = _run(monkeypatch, tmp_path, ON32, canonicalize=True, **kw)
    plan = eval_FLAC.resolve_rotation_plan(kw.get("rotate_mode", "fixed"), kw.get("rotate_deg", 0.0),
                                           kw.get("rotate_seed"), 42)
    gen = torch.Generator().manual_seed(plan.rotate_seed) if plan.is_random else None
    stream = eval_FLAC.RotationStream() if plan.is_random else None
    assert len(seen.metric_depth) == len(seen.cond) == 2
    for (_, batch_md), cond, depth, scenes in zip(seen.loader.batches, seen.cond, seen.metric_depth,
                                                  seen.scenes):
        rotated = eval_FLAC.apply_rotation_plan(list(batch_md), plan, gen, stream)
        want = _metric_inputs(rotated)
        assert len(depth) == len(want) and all(torch.equal(a, b) for a, b in zip(depth, want))
        assert scenes == [md["scene"] for md in rotated]
        assert all(_j_star(md) == 0 for md in cond)                  # conditioner: canonical
        assert any(not torch.equal(a, b) for a, b in zip(_metric_inputs(cond), want))


def test_are_addback_receives_the_post_rotation_pre_canonical_list(monkeypatch, tmp_path):
    """Codex r3a-F1 re-verification NIT (i): with canonicalization ON and a mocked ARE
    bank, ``apply_are_addback`` must receive the METRIC list (post-rotation,
    pre-canonicalization), never the canonical list the conditioner sees. CANON + ARE is
    not a registered cell; this pins the route before that composition is ever used."""
    import types
    bank = types.SimpleNamespace(cfg=types.SimpleNamespace(early_frames=1))
    received = []
    monkeypatch.setattr(eval_FLAC, "resolve_are_from_checkpoint",
                        lambda emb, file_cfg, lam: (0.5, "config", {"toy": True}))
    monkeypatch.setattr(eval_FLAC, "build_anchor_bank", lambda *a, **k: bank)

    def spy(latents, bank_, lam, metadata, device):
        assert bank_ is bank and lam == 0.5
        received.append(metadata)
        return latents
    monkeypatch.setattr(eval_FLAC, "apply_are_addback", spy)
    _, _, seen = _run(monkeypatch, tmp_path, ON32, canonicalize=True, rotate_deg=90.0)
    plan = eval_FLAC.resolve_rotation_plan("fixed", 90.0, None, 42)
    assert len(received) == len(seen.cond) == 2
    for (_, batch_md), got, cond in zip(seen.loader.batches, received, seen.cond):
        want = eval_FLAC.apply_rotation_plan(list(batch_md), plan)
        assert got is not cond and len(got) == len(want)
        assert all(torch.equal(g["depth"], w["depth"]) and torch.equal(g["source"], w["source"])
                   for g, w in zip(got, want))
        assert all(_j_star(md) == 0 for md in cond)
        assert any(not torch.equal(g["depth"], c["depth"]) for g, c in zip(got, cond))


def test_trained_but_not_applied_diagnostic(monkeypatch, tmp_path):
    _, _, seen = _run(monkeypatch, tmp_path, ON32, allow_canon_mismatch=True)
    rec = json.loads((tmp_path / "toy_metrics_1_1.0_cell_canonmis.json").read_text())
    assert (rec["canonicalize_applied"], rec["canonicalize_mismatch"], rec["canonicalize_tiebreak"],
            rec["canonicalize_tie_counts"]) == (False, True, None, None)
    assert not list(tmp_path.glob("*.canon.json"))
    assert any(_j_star(md) != 0 for batch in seen.cond for md in batch)


@pytest.mark.parametrize("training, kw, needle", [
    (ON32, {}, "TRAINED with canonicalize=True"),
    ({}, dict(canonicalize=True), "TRAINED with canonicalize=False"),
    (ON32, dict(canonicalize=True, record_stream=True, dump_scored_waveforms="TAKEN"), "already exists")])
def test_protocol_and_dump_refusals_come_before_model_construction(
        monkeypatch, tmp_path, training, kw, needle):
    if kw.get("dump_scored_waveforms") == "TAKEN":
        (tmp_path / "dumps" / "toy_metrics_1_1.0_cell_canon").mkdir(parents=True)
        (tmp_path / "dumps" / "toy_metrics_1_1.0_cell_canon" / "x").write_text("x")
        kw["dump_scored_waveforms"] = str(tmp_path / "dumps")
    with pytest.raises(ValueError, match=needle):
        _run(monkeypatch, tmp_path, training, raise_at_model=True, **kw)
    assert not list(tmp_path.glob("toy_metrics_*"))


def test_a_dump_without_a_stream_is_refused_before_any_file_work(tmp_path):
    with pytest.raises(ValueError, match="stream"):
        eval_FLAC.evaluate_model(str(tmp_path / "m.json"), str(tmp_path / "d.json"),
                                 str(tmp_path / "c.ckpt"), steps=1, cfg_scale=1.0, device="cpu",
                                 dump_scored_waveforms=str(tmp_path))


@pytest.mark.parametrize("kw", [dict(rotate_mode="random", rotate_seed=3), dict(record_stream=False)],
                         ids=["random_mode_alone", "fixed_without_stream"])
def test_a_dump_needs_the_published_stream_sidecar(tmp_path, kw):
    """r3a: random mode accumulates a stream but publishes no sidecar -- not enough."""
    with pytest.raises(ValueError, match="--record-stream"):
        eval_FLAC.evaluate_model(str(tmp_path / "m.json"), str(tmp_path / "d.json"),
                                 str(tmp_path / "c.ckpt"), steps=1, cfg_scale=1.0, device="cpu",
                                 dump_scored_waveforms=str(tmp_path), **kw)
    assert list(tmp_path.iterdir()) == []                 # refused before any file work


def test_the_loop_routes_through_prepare_eval_metadata_and_dumps_what_it_scores(monkeypatch, tmp_path):
    """Call order and counts on the REAL functions (replaces an inspect.getsource check)."""
    events = []

    def spy(name, fn, around=False):
        def wrapper(*a, **k):
            events.append(name + (">" if around else ""))
            out = fn(*a, **k)
            if around:
                events.append(name + "<")
            return out
        monkeypatch.setattr(eval_FLAC, name, wrapper)
    spy("resolve_canonicalize_protocol", eval_FLAC.resolve_canonicalize_protocol)
    spy("build_output_paths", eval_FLAC.build_output_paths)
    spy("prepare_eval_metadata", eval_FLAC.prepare_eval_metadata, around=True)
    spy("apply_rotation_plan", eval_FLAC.apply_rotation_plan)
    (tmp_path / "dumps").mkdir()
    _run(monkeypatch, tmp_path, ON32, events=events, canonicalize=True, record_stream=True,
         rotate_deg=90.0, dump_scored_waveforms=str(tmp_path / "dumps"))
    setup = [e for e in events if e in ("resolve_canonicalize_protocol", "build_output_paths", "model")]
    assert setup == ["resolve_canonicalize_protocol", "build_output_paths", "model"]
    loop = [e for e in events if e not in setup]
    assert loop == ["prepare_eval_metadata>", "apply_rotation_plan", "prepare_eval_metadata<",
                    "cond", "append", "score"] * 2         # two batches; rotation only inside prepare


# --------------------------------------------------------------------------- #
# CLI: each new flag parses AND reaches evaluate_model (behavioural, real script)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("flags, needle", [
    (["--canonicalize"], "no embedded model_config"),
    (["--allow-canon-mismatch"], "no embedded model_config"),
    (["--dump-scored-waveforms", "/nonexistent_root"], "--dump-scored-waveforms needs")])
def test_cli_flags_reach_evaluate_model(tmp_path, flags, needle):
    import os
    import subprocess
    import sys
    from pathlib import Path
    (tmp_path / "m.json").write_text(json.dumps({"model_type": "diffusion_cond",
                                                 "training": {"use_ema": False}}))
    torch.save({"state_dict": {}}, str(tmp_path / "c.ckpt"))      # no embedded model_config
    proc = subprocess.run(
        [sys.executable, "eval_FLAC.py", "--model-config", str(tmp_path / "m.json"),
         "--dataset-config", str(tmp_path / "d.json"), "--ckpt-path", str(tmp_path / "c.ckpt"),
         "--device", "cpu"] + flags,
        capture_output=True, cwd=str(Path(eval_FLAC.__file__).resolve().parent),
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert proc.returncode == 1 and b"unrecognized arguments" not in proc.stderr
    assert needle.encode() in proc.stderr, proc.stderr.decode()[-2000:]


# --------------------------------------------------------------------------- #
# rB-F2 F2-6 (Codex B2 HIGH, B4): the record names the EXACT bytes torch.load deserialized
# --------------------------------------------------------------------------- #
LOAD = {"ckpt_sha256_loaded": SHA_A, "ckpt_bytes_loaded": 724_000_123}


@pytest.mark.parametrize("case", MATRIX, ids=lambda c: f'{c["cond_method"]}_{c["rotate_deg"]}')
def test_record_without_a_load_digest_is_still_the_golden_record(case, monkeypatch):
    """``ckpt_load=None`` (the default) leaves every legacy record byte-identical."""
    monkeypatch.setattr(eval_FLAC, "source_sha", lambda: META["source_sha_stub"])
    args = (META["metrics"], META["ckpt_path"], case["rotate_deg"], case["cond_method"],
            case["frame_avg_angles"])
    explicit = eval_FLAC.build_metrics_record(*args, ckpt_load=None, **_record_kwargs(case))
    assert list(explicit) == case["metrics_record_keys"]
    assert json.dumps(explicit, indent=4) == case["metrics_record_json_indent4"]


def test_the_load_digest_follows_the_base_keys():
    base = eval_FLAC.build_metrics_record({"T60": 1.0}, CKPT, 0.0, "vanilla", None, canon_provenance=_prov())
    rec = eval_FLAC.build_metrics_record({"T60": 1.0}, CKPT, 0.0, "vanilla", None, canon_provenance=_prov(),
                                         ckpt_load=dict(LOAD))
    keys = list(base)
    at = keys.index("device") + 1
    assert list(rec) == keys[:at] + list(eval_FLAC.CKPT_LOAD_KEYS) + keys[at:]
    assert tuple(eval_FLAC.CKPT_LOAD_KEYS) == ("ckpt_sha256_loaded", "ckpt_bytes_loaded")
    assert {k: rec[k] for k in eval_FLAC.CKPT_LOAD_KEYS} == LOAD


@pytest.mark.parametrize("bad", [{"ckpt_sha256_loaded": SHA_A}, dict(LOAD, extra=1),
                                 dict(LOAD, ckpt_sha256_loaded="A" * 64), dict(LOAD, ckpt_sha256_loaded="a" * 63),
                                 dict(LOAD, ckpt_bytes_loaded=0), dict(LOAD, ckpt_bytes_loaded=True),
                                 dict(LOAD, ckpt_bytes_loaded=1.0), "not a dict"])
def test_a_malformed_load_digest_is_refused(bad):
    with pytest.raises(ValueError, match="ckpt"):
        eval_FLAC.build_metrics_record({"T60": 1.0}, CKPT, 0.0, "vanilla", None, ckpt_load=bad)


def test_read_checkpoint_exact_hashes_the_returned_buffer(tmp_path):
    p = tmp_path / "x.ckpt"
    p.write_bytes(b"\x00checkpoint bytes\xff" * 1000)
    buf, load = eval_FLAC.read_checkpoint_exact(str(p))
    assert buf == p.read_bytes()
    assert load == {"ckpt_sha256_loaded": hashlib.sha256(buf).hexdigest(), "ckpt_bytes_loaded": len(buf)}


def test_evaluate_model_records_the_digest_of_the_bytes_it_deserialized(monkeypatch, tmp_path):
    """evaluate_model reads the checkpoint ONCE into memory, hashes THAT buffer and hands the same
    buffer to torch.load (map_location='cpu', no other argument -- the call is otherwise unchanged).
    The spy rewrites the file on disk after the read: the record still names what was deserialized."""
    import io
    real_load, seen = torch.load, []

    def spy(f, *args, **kwargs):
        seen.append((type(f), args, kwargs))
        if isinstance(f, io.BytesIO):
            seen.append(("bytes", hashlib.sha256(f.getvalue()).hexdigest(), len(f.getvalue())))
            with open(tmp_path / "toy.ckpt", "ab") as out:      # the file changes AFTER the read
                out.write(b"appended after the evaluator read the file")
        return real_load(f, *args, **kwargs)
    monkeypatch.setattr(torch, "load", spy)
    _run(monkeypatch, tmp_path, {})
    rec = json.loads((tmp_path / "toy_metrics_1_1.0_cell.json").read_text())
    assert seen[0] == (io.BytesIO, (), {"map_location": "cpu"}), seen[0]
    _, sha, n = seen[1]
    assert (rec["ckpt_sha256_loaded"], rec["ckpt_bytes_loaded"]) == (sha, n)
    now = (tmp_path / "toy.ckpt").read_bytes()
    assert hashlib.sha256(now[:n]).hexdigest() == sha and len(now) > n     # the deserialized prefix, not the file now
    assert list(rec).index("ckpt_sha256_loaded") == list(rec).index("device") + 1


def test_the_legacy_cell_record_gains_only_the_load_digest(monkeypatch, tmp_path):
    _run(monkeypatch, tmp_path, {})
    rec = json.loads((tmp_path / "toy_metrics_1_1.0_cell.json").read_text())
    data = (tmp_path / "toy.ckpt").read_bytes()
    assert rec["ckpt_sha256_loaded"] == hashlib.sha256(data).hexdigest()
    assert rec["ckpt_bytes_loaded"] == len(data)
    base = eval_FLAC.build_metrics_record({"T60": 1.0}, rec["ckpt_path"], 0.0, "vanilla", None)
    assert [k for k in rec if k not in base] == list(eval_FLAC.CKPT_LOAD_KEYS)
