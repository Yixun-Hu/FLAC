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
                   "canon_hash": stream.canon_hash()}
    assert eval_FLAC.canonical_stream_hash(rec["positions"]) == rec["canon_hash"]


@pytest.mark.parametrize("roll", [1, 8, 16, 31])
def test_canonical_depth_hashes_are_equal_across_a_rolled_copy(roll):
    base, rolled = _canon_stream_of(_batch()), _canon_stream_of(_batch(), roll)
    for a, b in zip(base.position_tuples(), rolled.position_tuples()):
        assert a[:2] == b[:2] and a[3:] == b[3:]       # position, target, class, depth sha
        assert b[2] == (a[2] - roll) % W16             # the residual shift co-rotates
    assert base.canon_hash() != rolled.canon_hash()    # dj is part of the cell digest


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
    return out, stream, canon_stream


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
