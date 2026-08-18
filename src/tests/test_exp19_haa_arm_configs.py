"""exp_19 — the HAA finetune arm configs, delta by delta.

Three arms finetune on HAA from three 40k AR inits. The recipe (steps, batch, LR,
schedule, metrics, the HAA panorama convention) is the released one and must be
**identical across arms** — the only thing under test is the AR pretraining
policy each arm inherits, plus, for two of them, the matching training-time
treatment:

  * **HAA-P1**  — vanilla. Consumes ``src/configs/model_configs/FLAC/HAA/
    FLAC_HAA_finetune.json`` **directly**; no copy exists, so it cannot drift.
  * **HAA-BF**  — frame-averaged. Stock + exactly B-F's two AR training deltas.
  * **HAA-YAW** — yaw-augmented. Stock + exactly exp_17's ``training.yaw_aug``.

Everything here is byte-level and forward-constructed: the expected arm file is
built from the stock file's own bytes plus the registered insertion, then compared
wholesale. Subtraction ("assert the diff is small") cannot see a reordered block
or a re-encoded newline; ``read_text`` cannot see a CRLF rewrite at all, because
universal-newline decoding erases it. The technique is lifted from
``test_yaw_aug_a6000_arm_config.py``, which exists because exp_17 needed the same
guarantee.

Why the deltas must match B-F **verbatim** rather than merely "be frame
averaging": announcement 05 — the eval protocol must match how the checkpoint was
trained, and a checkpoint trained on a different orbit (say C8, or angles not
starting at the identity) is not comparable to the B-F row it is meant to inherit
from. The angle list is therefore compared against the *live* exp_07 B-F config,
not against a remembered literal.

Written by the exp_19 coder seat (Claude Opus 5, max effort).
"""
import hashlib
import json
from pathlib import Path

import pytest

from src.tests.test_yaw_aug_a6000_arm_config import INSERTED_BYTES as EXP17_YAW_INSERTED_BYTES
from src.training.factory import _parse_yaw_aug_config


_REPO = Path(__file__).resolve().parents[2]

STOCK_CONFIG = _REPO / "src/configs/model_configs/FLAC/HAA/FLAC_HAA_finetune.json"
EXP19 = _REPO / "worklog/worklog_yixun/exp_19_haa_finetune_claude"
BF_CONFIG = EXP19 / "FLAC_HAA_finetune_BF.json"
YAW_CONFIG = EXP19 / "FLAC_HAA_finetune_YAW.json"

# exp_07's pair: the AR configs P1 and B-F were trained with. The BF-minus-BVp1
# delta is recomputed from these at test time (see the module docstring).
AR_BVP1 = _REPO / "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json"
AR_BF = _REPO / "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json"

# The stock config's content, pinned. Without it the stock file and both arm files
# could drift TOGETHER and every "arm == stock + deltas" assertion below would
# still pass while all three had moved off the released HAA recipe. Recorded
# 2026-08-17 from the file the P1 arm consumes directly.
STOCK_SHA256 = "3639a9face84d13bcbb8f4472e78970c8e045952337f11b4f77d8798f786ba80"

# The end of the stock file: the close of "training" and of the root object. Both
# insertions go immediately before it, so every other byte is untouched.
TRAILER_BYTES = b'\n    }\n}'

# B-F's two AR training deltas, byte for byte (asserted equal to the live
# exp_07 delta below).
BF_INSERTED_BYTES = (
    b',\n'
    b'        "cond_method": "fa_invariant",\n'
    b'        "frame_avg_angles": [\n'
    b'            0.0,\n'
    b'            90.0,\n'
    b'            180.0,\n'
    b'            270.0\n'
    b'        ]'
)

# exp_17's treatment block, byte for byte — IMPORTED from exp_17's own contract
# test rather than retyped (Codex exp_19 r1, non-blocking finding). A copied
# literal would keep agreeing with itself after exp_17 moved; this way a drift
# there fails the forward-construction test here, which is the point of claiming
# "exactly the exp_17 treatment".
YAW_INSERTED_BYTES = EXP17_YAW_INSERTED_BYTES

EXPECTED_YAW_BLOCK = {"enabled": True, "img_w": 512, "seed": 42}
EXPECTED_FRAME_ANGLES = [0.0, 90.0, 180.0, 270.0]


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _bytes(path):
    assert path.is_file(), f"config not found: {path}"
    return path.read_bytes()


@pytest.fixture(scope="module")
def stock_bytes():
    return _bytes(STOCK_CONFIG)


@pytest.fixture(scope="module")
def bf_bytes():
    return _bytes(BF_CONFIG)


@pytest.fixture(scope="module")
def yaw_bytes():
    return _bytes(YAW_CONFIG)


@pytest.fixture(scope="module")
def stock(stock_bytes):
    return json.loads(stock_bytes.decode())


@pytest.fixture(scope="module")
def bf(bf_bytes):
    return json.loads(bf_bytes.decode())


@pytest.fixture(scope="module")
def yaw(yaw_bytes):
    return json.loads(yaw_bytes.decode())


def strict_diff(a, b, path="root"):
    """First type-then-value difference between two parsed JSON objects, or None.

    Lifted from ``test_yaw_aug_a6000_arm_config.py`` (same contract, same reason:
    ``1 == True`` and ``0 == 0.0`` in Python, so plain equality cannot see a type
    drift that the trainer would see). Module level so the non-vacuity guard below
    exercises the very function the contract tests rely on.
    """
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        if set(a) != set(b):
            return f"{path}: key sets differ ({set(a) ^ set(b)})"
        for k in a:
            found = strict_diff(a[k], b[k], f"{path}.{k}")
            if found:
                return found
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            found = strict_diff(x, y, f"{path}[{i}]")
            if found:
                return found
        return None
    return None if a == b else f"{path}: {a!r} != {b!r}"


# --------------------------------------------------------------------------- #
# 1. the baseline itself
# --------------------------------------------------------------------------- #
def test_the_stock_config_is_the_pinned_one(stock_bytes):
    """Coordinated-drift guard: the released HAA recipe must not have moved.

    It is also the P1 arm's config *as run*, so a change here silently changes
    one arm of the experiment.
    """
    actual = hashlib.sha256(stock_bytes).hexdigest()
    assert actual == STOCK_SHA256, (
        f"the stock HAA finetune config has changed (sha256 {actual}); every "
        "'arm == stock + deltas' assertion below is meaningless — and the P1 arm "
        "is no longer the released recipe — until this pin is re-derived"
    )


def test_p1_has_no_copied_config(stock):
    """P1 consumes the stock file directly; a copy is a drift surface.

    Two files that must stay identical eventually will not, and the divergence
    would be invisible in a launcher that names them separately.
    """
    strays = sorted(p.name for p in EXP19.glob("FLAC_HAA_finetune_P1*.json"))
    assert strays == [], (
        f"unexpected P1 config copies in the exp_19 folder: {strays}; the P1 arm "
        "must point at src/configs/model_configs/FLAC/HAA/FLAC_HAA_finetune.json"
    )
    # …and the stock file must itself be a clean vanilla arm.
    assert "cond_method" not in stock["training"]
    assert "yaw_aug" not in stock["training"]


def test_the_insertion_point_is_unambiguous(stock_bytes):
    """Both arms are built by inserting before the same unique trailer."""
    assert stock_bytes.count(TRAILER_BYTES) == 1
    assert stock_bytes.endswith(TRAILER_BYTES)


# --------------------------------------------------------------------------- #
# 2. forward byte construction
# --------------------------------------------------------------------------- #
def test_bf_arm_is_the_stock_plus_exactly_the_two_fa_deltas(bf_bytes, stock_bytes):
    prefix = stock_bytes[: -len(TRAILER_BYTES)]
    expected = prefix + BF_INSERTED_BYTES + TRAILER_BYTES
    assert bf_bytes == expected, (
        "FLAC_HAA_finetune_BF.json is not the stock HAA config plus exactly "
        "cond_method + frame_avg_angles — the arm is no longer the registered "
        "treatment"
    )


def test_yaw_arm_is_the_stock_plus_exactly_the_yaw_aug_block(yaw_bytes, stock_bytes):
    prefix = stock_bytes[: -len(TRAILER_BYTES)]
    expected = prefix + YAW_INSERTED_BYTES + TRAILER_BYTES
    assert yaw_bytes == expected, (
        "FLAC_HAA_finetune_YAW.json is not the stock HAA config plus exactly the "
        "exp_17 yaw_aug block"
    )


def test_byte_comparison_would_catch_newline_drift(stock_bytes):
    """Non-vacuity guard for the two tests above: prove the comparison is strict.

    Entirely in memory; no file is touched.
    """
    prefix = stock_bytes[: -len(TRAILER_BYTES)]
    genuine = prefix + BF_INSERTED_BYTES + TRAILER_BYTES
    crlf = genuine.replace(b"\n", b"\r\n")
    assert crlf != genuine, "the drift fixture is a no-op"
    assert crlf.decode().splitlines() == genuine.decode().splitlines(), (
        "text-level comparison cannot tell these apart — which is why the "
        "assertions above are on bytes"
    )


# --------------------------------------------------------------------------- #
# 3. type-strict semantic equality
# --------------------------------------------------------------------------- #
def test_removing_the_fa_deltas_leaves_a_type_strict_copy_of_the_stock(bf, stock):
    stripped = json.loads(json.dumps(bf))
    stripped["training"].pop("cond_method")
    stripped["training"].pop("frame_avg_angles")
    assert strict_diff(stripped, stock) is None


def test_removing_yaw_aug_leaves_a_type_strict_copy_of_the_stock(yaw, stock):
    stripped = json.loads(json.dumps(yaw))
    stripped["training"].pop("yaw_aug")
    assert strict_diff(stripped, stock) is None


def test_the_strict_comparator_rejects_a_type_drift_that_plain_equality_misses(stock):
    """Non-vacuity guard: ``strict_diff`` must be stricter than ``==``.

    ``mask_padding_dropout`` is ``0.0`` in the stock config; rewriting it as the
    int ``0`` leaves the objects ``==``-equal while changing the type the trainer
    would see.
    """
    drifted = json.loads(json.dumps(stock))
    assert isinstance(drifted["training"]["mask_padding_dropout"], float)
    drifted["training"]["mask_padding_dropout"] = 0
    assert drifted == stock, "plain equality cannot see the type change"
    found = strict_diff(drifted, stock)
    assert found is not None and "mask_padding_dropout" in found, found


# --------------------------------------------------------------------------- #
# 4. the deltas ARE B-F's / exp_17's, not merely similar
# --------------------------------------------------------------------------- #
def test_the_fa_delta_equals_exp07_BF_minus_BVp1_byte_for_byte():
    """The registered insertion is exp_07's own delta, recomputed from the files.

    If the arm's frame-averaging recipe differed from B-F's in any way — a fifth
    angle, an integer ``90`` where B-F has ``90.0``, a reordering — the HAA row
    would not be the transfer of the AR row it claims to continue.
    """
    bv = _bytes(AR_BVP1)
    bf_ar = _bytes(AR_BF)
    assert bv.endswith(TRAILER_BYTES) and bf_ar.endswith(TRAILER_BYTES)
    prefix = bv[: -len(TRAILER_BYTES)]
    assert bf_ar.startswith(prefix), (
        "exp_07's B-F is no longer BVp1 + a trailing insertion; the delta cannot "
        "be recomputed and BF_INSERTED_BYTES must be re-derived by hand"
    )
    ar_delta = bf_ar[len(prefix): -len(TRAILER_BYTES)]
    assert ar_delta == BF_INSERTED_BYTES


def test_the_bf_arm_values_match_exp07_BF(bf):
    """Same check at the parsed level, with float/int strictness."""
    ar = json.loads(_bytes(AR_BF).decode())["training"]
    assert bf["training"]["cond_method"] == ar["cond_method"] == "fa_invariant"
    assert strict_diff(bf["training"]["frame_avg_angles"], ar["frame_avg_angles"]) is None
    assert bf["training"]["frame_avg_angles"] == EXPECTED_FRAME_ANGLES
    assert all(isinstance(a, float) for a in bf["training"]["frame_avg_angles"])
    assert bf["training"]["frame_avg_angles"][0] == 0.0, (
        "invariant_conditioning requires angles[0] == 0.0 (the identity pass)"
    )


def test_the_yaw_delta_is_exp17s_own_literal_not_a_copy_of_it():
    """The YAW arm inherits exp_17's treatment; the bytes come from exp_17.

    Imported, so an edit to ``FLAC_AR_YAWAUG_A6000.json``'s registered block
    breaks the forward construction above instead of leaving two literals that
    quietly disagree. The parsed content is asserted here as well, so a drift is
    reported as "exp_17's block changed" rather than as an opaque byte mismatch.
    """
    assert YAW_INSERTED_BYTES is EXP17_YAW_INSERTED_BYTES
    parsed = json.loads(b'{"training": {"x": 0' + YAW_INSERTED_BYTES + b'}}')
    assert parsed["training"]["yaw_aug"] == EXPECTED_YAW_BLOCK, (
        "exp_17's registered treatment block has changed; the exp_19 YAW arm is "
        "no longer inheriting the same treatment"
    )


def test_the_yaw_block_is_exactly_the_registered_treatment(yaw):
    block = yaw["training"]["yaw_aug"]
    assert block == EXPECTED_YAW_BLOCK
    assert isinstance(block["enabled"], bool), "enabled must be a literal boolean"
    assert isinstance(block["img_w"], int) and not isinstance(block["img_w"], bool)
    assert isinstance(block["seed"], int) and not isinstance(block["seed"], bool)


# --------------------------------------------------------------------------- #
# 5. the real parsers accept each arm and enable exactly one treatment
# --------------------------------------------------------------------------- #
def test_the_factory_enables_the_treatment_for_the_yaw_arm(yaw):
    kwargs = _parse_yaw_aug_config(yaw["training"])
    assert kwargs.get("yaw_aug_enabled") is True, kwargs
    assert kwargs.get("yaw_aug_img_w") == 512
    assert kwargs.get("yaw_aug_seed") == 42


@pytest.mark.parametrize("name", ["stock", "bf"])
def test_the_factory_enables_no_augmentation_for_the_other_arms(request, name):
    cfg = request.getfixturevalue(name)
    assert _parse_yaw_aug_config(cfg["training"]) == {}


def test_the_bf_cond_method_is_one_the_wrapper_accepts(bf):
    """``DiffusionCondTrainingWrapper`` raises on anything else (diffusion.py:195)."""
    assert bf["training"]["cond_method"] in ("vanilla", "fa_invariant")


@pytest.mark.parametrize("name", ["stock", "bf", "yaw"])
def test_every_arm_keeps_ema_on(request, name):
    """The exp_19 inits are EMA weights and the HAA rows will be EMA rows too.

    ``use_ema: false`` anywhere would make ``extract_ema_weights`` inapplicable to
    that arm's own output and change what the eval loads (eval_FLAC.py:1159).
    """
    assert request.getfixturevalue(name)["training"]["use_ema"] is True


# --------------------------------------------------------------------------- #
# 6. the width pin (one checker, exercised from both sides)
# --------------------------------------------------------------------------- #
def check_width_pin(cfg):
    """Raise unless the augmentation width matches every ViT's panorama width.

    ONE checker, called by the pin test AND by its mutation guards — a guard that
    re-implements the comparison would keep passing if the pin itself were
    deleted (exp_17 review finding). The failure it guards is silent: the
    augmentation rolls the depth panorama by an integer number of COLUMNS, so a
    width mismatch rotates the conditioning by the wrong angle while training
    proceeds without error.
    """
    aug_w = cfg["training"]["yaw_aug"]["img_w"]
    vit_ws = {
        c["config"]["ViT"]["img_w"]
        for c in cfg["model"]["conditioning"]["configs"]
        if c["type"] == "ViTCoordinates"
    }
    if vit_ws != {aug_w}:
        raise AssertionError(
            f"yaw_aug.img_w={aug_w} but ViT widths are {vit_ws}: the augmentation "
            "would roll the panorama by the wrong number of columns and training "
            "would proceed without error"
        )


def test_pin_augmentation_width_matches_the_vit_panorama_width(yaw):
    check_width_pin(yaw)


@pytest.mark.parametrize(
    "mutate, side",
    [
        (lambda c: c["training"]["yaw_aug"].__setitem__("img_w", 256), "aug side"),
        (lambda c: [v["config"]["ViT"].__setitem__("img_w", 256)
                    for v in c["model"]["conditioning"]["configs"]
                    if v["type"] == "ViTCoordinates"], "ViT side"),
        (lambda c: c["model"]["conditioning"]["configs"][1]["config"]["ViT"]
                    .__setitem__("img_w", 256), "one ViT only"),
    ],
)
def test_a_width_mismatch_is_caught_by_the_pin(yaw, mutate, side):
    mutated = json.loads(json.dumps(yaw))
    mutate(mutated)
    with pytest.raises(AssertionError, match="wrong number of columns"):
        check_width_pin(mutated)


def test_the_width_checker_passes_the_unmutated_arm(yaw):
    """Non-vacuity: the guard above must not be passing for a trivial reason."""
    check_width_pin(yaw)


def test_both_haa_vit_conditioners_are_512_wide(stock):
    """The pin's premise, stated: HAA's panorama is 512 columns on both ViTs."""
    ws = [c["config"]["ViT"]["img_w"] for c in stock["model"]["conditioning"]["configs"]
          if c["type"] == "ViTCoordinates"]
    assert ws == [512, 512], ws


# --------------------------------------------------------------------------- #
# 7. the combination that is not a registered arm
# --------------------------------------------------------------------------- #
def test_a_config_carrying_both_treatments_is_refused(yaw):
    """``fa_invariant`` + ``yaw_aug`` is untested and must fail closed.

    It is also not one of the three arms: exp_19 has one treatment per arm by
    construction, so a file with both is a mistake, never a fourth arm.
    """
    training = json.loads(json.dumps(yaw))["training"]
    training["cond_method"] = "fa_invariant"
    with pytest.raises(ValueError, match="fa_invariant"):
        _parse_yaw_aug_config(training)


def test_neither_arm_file_carries_the_other_arm_key(bf, yaw):
    """One treatment per arm, asserted on the files as written."""
    assert "yaw_aug" not in bf["training"]
    assert "cond_method" not in yaw["training"]
    assert "frame_avg_angles" not in yaw["training"]
