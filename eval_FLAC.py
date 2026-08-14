import os
import argparse
import contextlib
import hashlib
import json
import math
import subprocess
from typing import List, NamedTuple, Optional
from tqdm import tqdm
import torch
import pytorch_lightning as pl

from src.data.are_anchor import (apply_anchor_addback, build_anchor_bank,
                                 compute_are_anchors)
from src.data.dataset import create_dataloader_from_config
from src.data.yaw_rotation import (rotate_scene_metadata, invariant_conditioning,
                                   draw_yaw_offsets, offsets_to_radians,
                                   yaw_column_shift,
                                   DEFAULT_FRAME_ANGLES, ORBIT_EXECUTION,
                                   FRAME_AVG_MAX_FWD_SAMPLES, resolve_frame_avg_cap)
from src.models import create_model_from_config
from src.training import create_training_wrapper_from_config, create_metric_callback_from_config


def rot_suffix(rotate_deg):
    """The injective filename suffix for a yaw offset: ``5.625 -> '_rot5p625'``.

    Integer rotations keep their historical byte-identical form (exp_02's
    ``_rot180`` artifacts must still resolve); fractional ones would collide
    under ``int()`` -- R3 evaluates 5.625 deg, which used to land on ``_rot5``
    alongside 5 deg -- so they get a decimal-safe form (round-4 review B3).

    Exposed as its own function because the R3 EVAL NAME needs the same
    rendering as the filename: five rotation rows of one cell otherwise share an
    eval name, and the identity that distinguishes them lives only in a field.
    """
    return '' if float(rotate_deg) == 0.0 else '_rot' + rot_token(rotate_deg)


def rot_token(rotate_deg):
    """The decimal-safe rendering of a yaw offset, with no empty case.

    ``rot_suffix`` renders 0 as the empty string so unrotated filenames stay
    byte-identical to the legacy ones. An eval NAME cannot do that -- `rot` with
    nothing after it is not a name -- so the R3 naming uses this instead:
    ``0 -> '0'``, ``5.625 -> '5p625'``.
    """
    d = float(rotate_deg)
    return str(int(d)) if d.is_integer() else repr(d).replace('.', 'p')


ROTATE_MODES = ('fixed', 'random')


class RotationPlan(NamedTuple):
    """The resolved yaw-rotation protocol for one evaluation run.

    ``mode='fixed'``  -- the legacy behaviour: every sample is rotated by the same
    ``rotate_deg`` (0.0 meaning "not rotated at all"); ``rotate_seed`` is None.

    ``mode='random'`` -- exp_14's estimand: each sample independently draws a
    panorama-column offset from a generator seeded with ``rotate_seed``, so no
    single angle describes the run and ``rotate_deg`` is None (recorded as JSON
    ``null``), not 0.0 -- an unrotated run and a randomly-rotated one must never
    be readable as the same protocol.
    """
    mode: str
    rotate_deg: Optional[float]
    rotate_seed: Optional[int]

    @property
    def is_random(self):
        return self.mode == 'random'


def resolve_rotation_plan(rotate_mode='fixed', rotate_deg=0.0, rotate_seed=None,
                          eval_seed=None):
    """Resolve (and validate) the rotation protocol before anything else runs.

    Every failure here is a HARD error, never a silent precedence rule --- this is
    the announcement-05 trap (eval-protocol flags are part of the experiment): a
    mismatched flag produces plausible-looking, catastrophically wrong numbers.

    - ``random`` with a non-zero ``rotate_deg``: a fixed angle and a per-sample
      draw are mutually exclusive protocols, so there is no correct winner.
    - ``fixed`` with an explicit ``rotate_seed``: silently ignoring it would let a
      manifest record a rotation seed that influenced nothing (review B4).
    - ``random`` with no seed anywhere: the assignment would be unreproducible.

    In ``random`` mode an omitted ``rotate_seed`` resolves to the EVAL seed
    (Yixun 2026-08-10), so a cell's rotation assignment is a function of the seed
    it already reports.
    """
    if rotate_mode not in ROTATE_MODES:
        raise ValueError(
            f"Unknown rotate_mode: {rotate_mode!r}; valid options: {list(ROTATE_MODES)}."
        )
    if rotate_mode == 'fixed':
        if rotate_seed is not None:
            raise ValueError(
                f"rotate_seed={rotate_seed!r} (--rotate-seed) is meaningless with "
                "rotate_mode='fixed' and must not be silently ignored: a fixed rotation "
                "draws nothing. Pass --rotate-mode random to use it."
            )
        return RotationPlan('fixed', float(rotate_deg), None)

    if float(rotate_deg) != 0.0:
        raise ValueError(
            f"rotate_mode='random' is incompatible with rotate_deg={rotate_deg!r} "
            "(--rotate-deg): a per-sample random yaw and a single fixed angle are "
            "different protocols. Drop --rotate-deg (or use --rotate-mode fixed)."
        )
    resolved = rotate_seed if rotate_seed is not None else eval_seed
    if resolved is None:
        raise ValueError(
            "--rotate-mode random needs a rotation seed: pass --rotate-seed, or an "
            "eval --seed for it to default to. An unseeded assignment is not reproducible."
        )
    return RotationPlan('random', None, int(resolved))


def rotation_token(rotate_mode='fixed', rotate_deg=0.0, rotate_seed=None):
    """The identity token for a run's rotation: ``'45'`` / ``'5p625'`` / ``'rand42'``.

    Fixed angles keep :func:`rot_token`'s rendering unchanged. A random-mode run
    is identified by its RESOLVED rotation seed, because that seed --- not any
    angle --- is what determines the assignment.
    """
    if rotate_mode == 'random':
        if rotate_seed is None:
            raise ValueError("random-mode naming requires a resolved rotate_seed")
        return f'rand{int(rotate_seed)}'
    return rot_token(rotate_deg)


def rotation_suffix(rotate_mode='fixed', rotate_deg=0.0, rotate_seed=None):
    """The filename suffix for a run's rotation; ``''`` for the unrotated case.

    Fixed mode is byte-identical to :func:`rot_suffix` (legacy artifacts must
    still resolve). Random mode appends ``_rotrand<seed>``, which is injective
    both across rotation seeds and against every fixed token: a fixed token
    always starts with a digit or ``'-'``, so it can never render as ``rand...``
    (review B5 --- reusing one eval name across rotation seeds must not overwrite).
    """
    if rotate_mode == 'random':
        return '_rot' + rotation_token(rotate_mode, rotate_deg, rotate_seed)
    return rot_suffix(rotate_deg)


ARE_LAMBDA_KEY = 'are_lambda'
ARE_ANCHOR_KEY = 'are_anchor'


def validate_are_lambda(value, who):
    """One place that decides whether a lambda is admissible, for both sources."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{who} must be a number in [0, 1], got {value!r}")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{who} must be in [0, 1], got {value!r}")
    return float(value)


def resolve_are_lambda(training_config, are_lambda=None):
    """``(lambda_or_None, source)`` — which anchor weight this evaluation applies.

    Announcement 05 governs this: an eval flag is part of the experiment
    definition, and a mismatch between the flag and how the checkpoint was
    trained produces plausible, catastrophically wrong numbers. So:

    * no flag -> the arm's OWN declared ``training.are_lambda`` (``None`` for
      every arm already in the record, which is how their records stay
      byte-identical);
    * an explicit ``--are-lambda`` OVERRIDES it, including ``0.0`` — that is the
      AR3 sweep, and 0.0 must be distinguishable from "not an ARE run";
    * a flag on an arm with no ``training.are_anchor`` block is a HARD error, not
      a silent no-op: without the calibrated constants there is no ``A`` to add
      back, and recording a lambda that influenced nothing is exactly the failure
      announcement 05 exists to prevent.

    Whichever wins is recorded in the metrics row together with its source.
    """
    training_config = training_config or {}
    declared = training_config.get(ARE_LAMBDA_KEY, None)
    has_anchor = isinstance(training_config.get(ARE_ANCHOR_KEY, None), dict)

    if are_lambda is None:
        if declared is None:
            return None, None
        lam = validate_are_lambda(declared, f"model config training.{ARE_LAMBDA_KEY}")
    else:
        lam = validate_are_lambda(are_lambda, "--are-lambda")

    if not has_anchor:
        raise ValueError(
            f"an ARE lambda ({lam}) was requested but the model config declares no "
            f"training.{ARE_ANCHOR_KEY} block, so the analytic anchor cannot be "
            "computed. Point --model-config at the ARE arm's config.")
    return lam, ("model_config" if are_lambda is None else "cli")


def are_suffix(are_lambda=None):
    """``''`` / ``'_are0'`` / ``'_are0p5'`` / ``'_are1'`` — the lambda's filename token.

    Empty for a non-ARE run, so every legacy artifact name is byte-identical.
    Non-empty otherwise, because AR3 evaluates ONE checkpoint at three lambdas and
    they must not overwrite one another's metrics file.
    """
    return '' if are_lambda is None else '_are' + rot_token(are_lambda)


def apply_are_addback(latents, bank, are_lambda, metadata, device):
    """``z_hat + lambda * A_query`` for one batch, before the decode.

    ``metadata`` is the POST-rotation metadata: ``r`` is rotation-invariant and the
    LOS lookup direction co-rotates with the panorama, so the anchor is the same
    under a rotated and an unrotated evaluation. That invariance is a design goal
    of the port (plan §2, yaw note), which is why the add-back sits here rather
    than being computed from a pre-rotation copy.
    """
    return apply_anchor_addback(latents, compute_are_anchors(bank, metadata, device),
                                are_lambda)


def build_output_paths(
    ckpt_path,
    steps,
    cfg_scale,
    eval_name,
    cond_method='vanilla',
    rotate_deg=0.0,
    n_angles=4,
    rotate_mode='fixed',
    rotate_seed=None,
    are_lambda=None,
):
    """Construct the metrics-JSON and predictions-.pt output paths for one run.

    Pure (no filesystem access): both paths sit in ``ckpt_path``'s directory,
    named ``<ckpt>_<kind>_<steps>_<cfg>_<eval_name><method><rot>.<ext>``.

    The vanilla + ``rotate_deg == 0`` name is byte-identical to the legacy
    ``eval_FLAC`` output, so exp_01 / exp_02 artifacts reproduce exactly. A
    non-vanilla ``cond_method`` inserts ``_<cond_method>_a<n_angles>`` and a
    non-zero ``rotate_deg`` appends ``_rot<int(rotate_deg)>``. Both suffixes now
    also land on the predictions name -- fixing the exp_02 bug where two
    ``rotate_deg`` values sharing an ``eval_name`` overwrote one predictions file.

    ``rotate_mode='random'`` (exp_14) replaces the angle suffix with
    ``_rotrand<rotate_seed>``; the fixed-mode names are unchanged down to the byte.

    Returns
    -------
    dict
        ``{'metrics': <...>.json, 'predictions': <...>.pt}``.
    """
    ckpt_name = os.path.basename(ckpt_path).replace('.ckpt', '')
    directory = os.path.dirname(ckpt_path)
    method_suffix = '' if cond_method == 'vanilla' else f'_{cond_method}_a{n_angles}'
    rot = rotation_suffix(rotate_mode, rotate_deg, rotate_seed)
    stem = f'{steps}_{cfg_scale}_{eval_name}{method_suffix}{rot}{are_suffix(are_lambda)}'
    return {
        'metrics': os.path.join(directory, f'{ckpt_name}_metrics_{stem}.json'),
        'predictions': os.path.join(directory, f'{ckpt_name}_predictions_{stem}.pt'),
    }


# Decimal places used to fingerprint a context-source position. Reference sources
# in AR sit metres apart, so 1e-6 m never merges two of them, and a fixed-width
# decimal string is stable across machines in a way float repr is not.
CONTEXT_ID_PRECISION = 6

# Version of the context-fingerprint rule (precision, field order, dtype pin).
# Recorded next to every fingerprint we persist: the rendering is stable only
# under the assumptions asserted in sample_context_ids, so a future change to any
# of them must be visible as a schema bump rather than as silently incomparable
# hashes. Codex code review N4 measured the fragility on the real unseen split:
# rendering the same positions as float64 changed two six-decimal strings and
# float16 changed 5,032 of them.
CONTEXT_FINGERPRINT_SCHEMA = 1


class StreamRow(NamedTuple):
    """One position of the evaluation stream, as it was actually evaluated."""
    position: int
    target_id: str
    context_ids: List[str]
    img_w: int
    offset: Optional[int]
    dataset_idx: Optional[int] = None


def sample_target_id(md):
    """Identity of the target item at one stream position: ``'<idx>|<relpath>'``.

    ``SampleDataset.__getitem__`` silently substitutes a *random other item* when
    a file fails to load or is silent (``return self[random.randrange(len(self))]``),
    and the substituted sample carries its OWN ``idx``/``relpath``. Recording both
    is therefore what makes the stream auditable: a hash over offsets alone would
    prove the RNG sequence and nothing about which item received which offset.

    Falls back to the absolute ``path`` when ``relpath`` is absent (it is only set
    when the file sits under a configured root). An item with neither is a
    hard error: an unidentifiable position makes the whole audit worthless.
    """
    idx = md.get('idx', None)
    rel = md.get('relpath', None) or md.get('path', None)
    if rel is None:
        raise ValueError(
            "cannot record stream identity: metadata carries neither 'relpath' nor "
            "'path'. The assignment audit (plan §3.3) requires a per-item identity."
        )
    return f'{idx}|{rel}'


def sample_context_ids(md):
    """Ordered fingerprint of the reference sources this sample actually drew.

    ``AR_md.get_custom_metadata`` picks the K context sources with
    ``np.random.choice`` per sample and does NOT expose their paths, so the
    strongest identity that reaches ``eval_FLAC`` is each source's 3D position
    (``context_poses``, listener-relative, one row per reference in draw order).
    Distinct sources sit metres apart, so a fixed 6-decimal rendering identifies
    the draw exactly while being byte-stable across machines.

    Read BEFORE any rotation: ``rotate_scene_metadata`` rewrites ``context_poses``,
    and a fingerprint taken afterwards would encode the rotation instead of the
    item (and could never match its unrotated pair).

    The rendering is FAIL-CLOSED on its own assumptions (review N4). A six-decimal
    string is only a stable identity while the tensor is exactly what the AR loader
    produces --- ``torch.vstack`` of float32 ``[K, 3]`` positions --- and the
    measured sensitivity is not academic: on the real unseen split, rendering the
    same positions as float64 changes two strings and float16 changes 5,032. A
    silently different dtype would therefore produce hashes that mismatch across
    machines for no scientific reason, so it raises instead.
    """
    poses = md.get('context_poses', None)
    if poses is None:
        return []
    if not isinstance(poses, torch.Tensor):
        raise ValueError(
            f"context_poses must be a torch.Tensor to be fingerprinted, got "
            f"{type(poses).__name__}."
        )
    if poses.dtype != torch.float32:
        raise ValueError(
            f"context_poses must be float32 to be fingerprinted, got {poses.dtype}: "
            "the six-decimal rendering is not dtype-stable (review N4), so a "
            "different dtype would silently produce non-comparable hashes."
        )
    if poses.dim() != 2 or poses.shape[0] < 1 or poses.shape[1] != 3:
        raise ValueError(
            f"context_poses must have shape [K, 3] with K >= 1 to be fingerprinted, "
            f"got {tuple(poses.shape)}."
        )
    if not bool(torch.isfinite(poses).all()):
        raise ValueError(
            "context_poses must be finite to be fingerprinted: a NaN or Inf position "
            "renders to a string that cannot identify a reference source."
        )
    out = []
    for row in poses:
        vals = []
        for v in row:
            val = float(v)
            if val == 0.0:
                val = 0.0  # normalise -0.0, whose rendering would differ
            vals.append(f'{val:.{CONTEXT_ID_PRECISION}f}')
        out.append(','.join(vals))
    return out


def canonical_stream_hash(tuples):
    """sha256 over an ordered tuple stream, in the pre-registered serialization.

    One JSON array per tuple (``sort_keys=True``, ``separators=(",", ":")``),
    LF-joined, UTF-8 (plan §3.3, review B5). Fixed here rather than left to
    ``json`` defaults because this string is the cross-machine contract: two cells
    are declared rotation-matched iff these digests agree, so whitespace or key
    ordering may never depend on the writer.
    """
    payload = "\n".join(
        json.dumps(list(t), sort_keys=True, separators=(",", ":")) for t in tuples
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RotationStream:
    """Accumulates the §3.3 assignment record: what was evaluated, in what order,
    with which rotation.

    Two digests are derived from it, and they answer different questions:

    * ``input_hash`` -- ``(i, target_id, context_ids, img_w)``: *which* items and
      reference draws the cell saw. Independent of the rotation, so it is the
      pairing key between a Z (unrotated) cell and its R (rotated) partner, and
      the cross-arm equality check.
    * ``assignment_hash`` -- ``(i, target_id, d_i)``: which item received which
      offset. Equal across arms iff they were rotated identically.
    """

    def __init__(self):
        self.rows = []

    def __len__(self):
        return len(self.rows)

    def record(self, md, offset, img_w):
        """Append one position. ``md`` must be the PRE-rotation metadata."""
        img_w = int(img_w)
        if self.rows and img_w != self.rows[0].img_w:
            raise ValueError(
                f"img_w changed mid-stream: {self.rows[0].img_w} -> {img_w}. The "
                "column grid defines the offsets, so it cannot vary within a run."
            )
        idx = md.get('idx', None)
        row = StreamRow(
            position=len(self.rows),
            target_id=sample_target_id(md),
            context_ids=sample_context_ids(md),
            img_w=img_w,
            offset=None if offset is None else int(offset),
            dataset_idx=None if idx is None else int(idx),
        )
        self.rows.append(row)
        return row

    @property
    def img_w(self):
        return self.rows[0].img_w if self.rows else None

    @property
    def target_ids(self):
        """Per-position target identities, for comparison against a split manifest."""
        return [r.target_id for r in self.rows]

    @property
    def offsets(self):
        return [r.offset for r in self.rows]

    def input_tuples(self):
        return [[r.position, r.target_id, r.context_ids, r.img_w] for r in self.rows]

    def assignment_tuples(self):
        return [[r.position, r.target_id, r.offset] for r in self.rows]

    def input_hash(self):
        return canonical_stream_hash(self.input_tuples())

    def assignment_hash(self):
        return canonical_stream_hash(self.assignment_tuples())


# Version of the .stream.json payload below. Bumped whenever a field's meaning
# changes, so a collector never reads an old sidecar under a new contract.
STREAM_SCHEMA_VERSION = 1


def stream_sidecar_path(metrics_path):
    """``<...>_metrics_<stem>.json`` -> ``<...>_metrics_<stem>.stream.json``.

    A SEPARATE file, not extra keys in the metrics record: fixed-mode records and
    their paths are frozen (review B4), and the tuple stream is ~6,337 entries --
    orders of magnitude larger than the record it would otherwise swamp.
    """
    base = metrics_path[:-len('.json')] if metrics_path.endswith('.json') else metrics_path
    return base + '.stream.json'


def build_stream_record(plan, stream):
    """The ``.stream.json`` payload: the assignment audit WITH its preimages.

    A digest alone is unfalsifiable evidence --- it can declare two cells matched,
    but when they do not match it cannot say where or why, and it cannot be
    recomputed by anyone who does not already trust the writer (review B2). This
    payload therefore carries the full ordered canonical tuples and the
    per-position offsets, from which both hashes are recomputable, plus the
    schema versions needed to read them under the right contract.

    ``rotate_seed`` is null in fixed mode (nothing was drawn) and ``rotate_deg``
    is null in random mode (no single angle describes the run) --- the same
    asymmetry the metrics record uses, so the two never disagree.
    """
    return {
        "schema_version": STREAM_SCHEMA_VERSION,
        "fingerprint_schema": CONTEXT_FINGERPRINT_SCHEMA,
        "rotate_mode": plan.mode,
        "rotate_seed": plan.rotate_seed,
        "rotate_deg": plan.rotate_deg,
        "img_w": stream.img_w,
        "stream_count": len(stream),
        "input_tuples": stream.input_tuples(),
        "offsets": stream.offsets,
        "assignment_tuples": stream.assignment_tuples(),
        "input_hash": stream.input_hash(),
        "assignment_hash": stream.assignment_hash(),
    }


def verify_stream_positions(stream):
    """Substitution guard, positionally: stream position ``i`` must be dataset item ``i``.

    Evaluation runs with ``shuffle=False`` over a sequential sampler, so this
    holds by construction --- unless ``SampleDataset.__getitem__`` substituted a
    random other item after a load or silence failure, which is exactly the event
    the assignment audit exists to catch. A hash can only reveal that by
    comparison against another cell; this catches it in the cell that suffered it.

    Raises ``RuntimeError`` naming the FIRST offending position. An item without
    an ``idx`` cannot be checked and is not silently accepted.
    """
    for row in stream.rows:
        if row.dataset_idx is None:
            raise RuntimeError(
                f"rotation stream position {row.position} has no dataset 'idx', so it "
                "cannot be checked against its position. Refusing to report metrics."
            )
        if row.dataset_idx != row.position:
            raise RuntimeError(
                f"rotation stream position {row.position} carries dataset idx "
                f"{row.dataset_idx} (target {row.target_id!r}): the dataset served a "
                "different item than the sequential sampler asked for -- "
                "SampleDataset substitutes a random item on a load/silence failure. "
                "Refusing to report metrics."
            )


def verify_stream_count(stream, dataset_count, expected=None):
    """Substitution guard: one stream position per split item, and the right split.

    Two independent checks, because they catch different failures:

    * ``len(stream) == dataset_count`` --- a short stream means items were dropped,
      a long one that they were served twice. This compares the run against
      itself, so it can only detect an inconsistency, never a wrong split.
    * ``... == expected`` --- the campaign's PRE-REGISTERED item count (6,337 for
      the published unseen split, announcement 01). Without it the first check is
      tautological: a zero-item or subsampled dataset is perfectly self-consistent
      and used to produce a valid-looking artifact (review B1). Supplied by the
      launch kit; omitting it leaves only the weaker check.

    Either failure FAILS the cell rather than reporting a number (plan §3.3).
    """
    got = len(stream)
    if dataset_count is None:
        raise RuntimeError(
            "cannot verify the rotation stream: the dataset item count is unknown."
        )
    dataset_count = int(dataset_count)
    if got != dataset_count:
        raise RuntimeError(
            f"rotation stream has {got} positions but the dataset holds {dataset_count} "
            "items: the evaluated split does not match the configured one (dropped, "
            "duplicated or substituted items). Refusing to report metrics."
        )
    if expected is None:
        return
    expected = int(expected)
    if expected <= 0:
        raise RuntimeError(
            f"expected_stream_count must be a positive item count, got {expected}."
        )
    if dataset_count != expected:
        raise RuntimeError(
            f"rotation stream and dataset agree on {got} items, but the pre-registered "
            f"split has {expected}: this cell evaluated a different split than the "
            "campaign declares (announcement 01 forbids new or subsampled eval "
            "configurations). Refusing to report metrics."
        )


def _batch_img_w(metadata):
    """Panorama width for a batch, from the first sample's depth map."""
    if "depth" not in metadata[0]:
        raise ValueError(
            "yaw rotation needs the depth panorama to define the column grid, but "
            "this sample's metadata has no 'depth' key."
        )
    return int(metadata[0]["depth"].shape[-1])


def apply_rotation_plan(metadata, plan, generator=None, stream=None):
    """Apply one batch's yaw rotation, per the resolved :class:`RotationPlan`.

    Fixed mode reproduces the legacy eval-loop branch exactly: nothing at all
    happens at ``rotate_deg == 0`` (the same list object is returned, and no
    ``depth`` key is required), and a non-zero angle rotates every sample by it.
    Passing a ``stream`` in fixed mode (``--record-stream``) additionally records
    each position with the run's CONSTANT column shift --- 0 for an unrotated Z
    cell --- taken from :func:`yaw_column_shift`, the same rule the rotation
    itself applies. Recording never changes what is returned.

    Random mode draws the WHOLE batch's offsets first --- one contiguous slice of
    the dedicated generator's stream, in item order --- and only then does any
    per-sample work. That ordering is what makes the assignment independent of
    how the loader chunks the split. Each sample's PRE-rotation identity is
    recorded into ``stream`` before it is rotated, so the audit describes the item
    rather than the rotation.

    Both ``generator`` and ``stream`` are required in random mode: a missing
    generator would silently draw from the global RNG (perturbing the
    evaluation's own sampling noise), and a missing stream would run an
    unauditable assignment. Neither is used in fixed mode.
    """
    if plan.is_random:
        if generator is None or stream is None:
            raise ValueError(
                "random rotation mode requires both a dedicated generator and a "
                "RotationStream: without them the assignment is neither isolated "
                "from the global RNG nor auditable."
            )
        if len(metadata) == 0:
            return metadata
        img_w = _batch_img_w(metadata)
        # Draw the whole batch up front, in stream order, BEFORE any per-sample work.
        offsets = draw_yaw_offsets(len(metadata), img_w, generator).tolist()
        angles = offsets_to_radians(offsets, img_w)
        rotated = []
        for md, offset, alpha in zip(metadata, offsets, angles):
            stream.record(md, offset, img_w)
            rotated.append(rotate_scene_metadata(md, alpha, img_w))
        return rotated

    # Fixed mode. With no stream and no angle there is nothing to do at all --
    # not even a depth lookup, which the legacy branch never made.
    if stream is None and plan.rotate_deg == 0.0:
        return metadata
    if len(metadata) == 0:
        return metadata
    img_w = _batch_img_w(metadata)
    alpha_rad = math.radians(plan.rotate_deg)
    if stream is not None:
        dj = yaw_column_shift(alpha_rad, img_w)
        for md in metadata:
            stream.record(md, dj, img_w)
    if plan.rotate_deg != 0.0:
        return [rotate_scene_metadata(md, alpha_rad, img_w) for md in metadata]
    return metadata


def resolve_weights_source(training_config, state_dict_keys):
    """Which weights an evaluation will actually use: ``'ema'`` or ``'online'``.

    ``evaluate_model`` swaps in EMA weights only when the config asks for them AND
    the checkpoint carries ``diffusion_ema.ema_model.*`` entries; otherwise it
    silently evaluates the online weights. A screen that merely *asserts* EMA in a
    sidecar can therefore be wrong, so the record states what actually happened
    (round-4 review B1)."""
    wants_ema = bool((training_config or {}).get("use_ema", False))
    has_ema = any(str(k).startswith("diffusion_ema.ema_model.") for k in state_dict_keys)
    return "ema" if (wants_ema and has_ema) else "online"


def source_sha():
    """Short-circuited ``git rev-parse HEAD`` for provenance; ``'unknown'`` if git
    is unavailable. Provenance must never be able to break an evaluation."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL, timeout=10,
        )
        return out.decode().strip() or "unknown"
    except Exception:
        return "unknown"


def orbit_provenance(cond_method, frame_avg_max_fwd_samples=None):
    """``(orbit_execution, frame_avg_fwd_cap)`` for a conditioning method.

    A vanilla evaluation executes NO orbit, so labelling it ``batched`` would be
    false provenance and would make a vanilla row look protocol-compatible with a
    batched frame-averaged row.

    The recorded cap is the one the run ACTUALLY applied. ``None`` -> the module
    default (64), which is what every historical row carries and what exp_14 pins
    as the common EVALUATION cap for both arms: each arm's TRAINING cap is a
    property of its checkpoint, and the comparison is between training policies,
    never between evaluation policies (plan §3.1)."""
    if cond_method == "fa_invariant":
        return ORBIT_EXECUTION, resolve_frame_avg_cap(frame_avg_max_fwd_samples)
    return "n/a", None


def _rotation_provenance(record, rotate_mode, rotate_seed, input_hash,
                         assignment_hash, stream_count, img_w):
    """Add exp_14's random-rotation provenance to a record -- IN RANDOM MODE ONLY.

    Fixed-mode records must stay byte-identical to every row already committed
    (review B4), so nothing is added, removed or reordered there. In random mode
    the keys are APPENDED (a collector reads both generations with one schema) and
    ``rotate_deg`` is overwritten in place with ``None``: no single angle describes
    the run, and recording 0.0 would make a randomly-rotated cell indistinguishable
    from an unrotated one.
    """
    if rotate_mode not in ROTATE_MODES:
        raise ValueError(
            f"Unknown rotate_mode: {rotate_mode!r}; valid options: {list(ROTATE_MODES)}."
        )
    if rotate_mode != 'random':
        return record
    record["rotate_deg"] = None
    record["rotate_mode"] = rotate_mode
    record["rotate_seed"] = rotate_seed
    record["input_hash"] = input_hash
    record["assignment_hash"] = assignment_hash
    record["stream_count"] = stream_count
    record["img_w"] = img_w
    return record


# Version of the per-scene payload below (exp_14 round-3 fix R3F1). Bumped when
# the meaning of a field changes, so a collector never reads an old block under a
# new contract.
PER_SCENE_SCHEMA = 1


def split_per_scene_metrics(metrics_dict):
    """``(flat_metrics, by_scene_or_None)``.

    ``AcousticMetricsCallback.compute_metrics`` returns the per-scene map nested
    INSIDE the metrics dict under ``by_scene``. The record keeps ``metrics`` flat
    and numeric -- every existing consumer reads ``record["metrics"][key]`` as a
    number, and the exp_11 row validator pins that key set exactly -- so the map
    is lifted out here and recorded beside it instead.
    """
    if "by_scene" not in metrics_dict:
        return dict(metrics_dict), None
    flat = {k: v for k, v in metrics_dict.items() if k != "by_scene"}
    return flat, metrics_dict["by_scene"]


def resolve_metrics_payload(computed, record_per_scene=False):
    """``(metrics_dict, by_scene)`` — what the record will carry.

    FROZEN UNLESS ASKED (round-3 closure A1/B1). The callback emits ``by_scene``
    on its own for some configurations — HAA turns per-scene metrics on
    independently of this flag (``metric_callback.py``) — so lifting the block
    unconditionally rewrote records that never asked for it: the legacy NESTED
    ``metrics["by_scene"]`` disappeared and three top-level keys appeared. That
    is exactly the re-keying of historical rows the byte-compat contract exists
    to prevent, and it was invisible to the snapshot matrix, whose payloads carry
    no per-scene block at all.

    Without the flag the callback result is passed through untouched. With it,
    the block is lifted — and a callback that produced none is an error, because
    the run asked for the estimand and did not measure it.
    """
    if not record_per_scene:
        return computed, None
    metrics_dict, by_scene = split_per_scene_metrics(computed)
    if by_scene is None:
        raise RuntimeError(
            "--record-per-scene was requested but the metric callback returned no "
            "per-scene block; the per-scene estimand was not measured."
        )
    return metrics_dict, by_scene


def _per_scene_provenance(record, by_scene):
    """Add the per-scene block -- ONLY when the run actually recorded one.

    Fixed-mode legacy records stay byte-identical (review B4, still binding), so
    nothing is added, removed or reordered unless ``--record-per-scene`` asked
    for it. An EMPTY map is refused rather than recorded: a run that asked for
    per-scene results and produced none did not measure the estimand, and
    recording ``{}`` would publish that silence as a fact.
    """
    if by_scene is None:
        return record
    if not isinstance(by_scene, dict) or not by_scene:
        raise ValueError(
            f"per-scene recording produced {by_scene!r}: a run that asked for "
            "per-scene results and produced none did not measure the per-scene "
            "estimand, so there is nothing to record."
        )
    record["by_scene"] = by_scene
    record["per_scene_schema"] = PER_SCENE_SCHEMA
    record["scene_count"] = len(by_scene)
    return record


def _are_provenance(record, are_lambda, are_lambda_source, are_anchor):
    """Add exp_16's anchor provenance -- ONLY when an anchor was actually applied.

    Same conditional discipline as ``_rotation_provenance``: a non-ARE record must
    stay byte-identical to every row already committed, and an ARE record must
    name the lambda it applied AND where that lambda came from, because
    ``--are-lambda`` can override the config (the AR3 sweep).
    """
    if are_lambda is None:
        return record
    record[ARE_LAMBDA_KEY] = are_lambda
    record["are_lambda_source"] = are_lambda_source
    record[ARE_ANCHOR_KEY] = are_anchor
    return record


def build_metrics_record(metrics_dict, ckpt_path, rotate_deg, cond_method, frame_avg_angles,
                         cond_autocast='default', batch_size=None, n_samples=None,
                         dataset_config=None, seed=None, cfg_scale=None, steps=None,
                         eval_name=None, weights_source=None, device=None,
                         rotate_mode='fixed', rotate_seed=None, input_hash=None,
                         assignment_hash=None, stream_count=None, img_w=None,
                         by_scene=None, frame_avg_max_fwd_samples=None,
                         are_lambda=None, are_lambda_source=None, are_anchor=None):
    """Assemble the dict written to the metrics JSON.

    Extends the legacy ``{metrics, ckpt_path, rotate_deg}`` record with
    ``cond_method``, ``frame_avg_angles`` (the frame-average angles used for
    ``fa_invariant``; ``None`` for vanilla), ``cond_autocast``, and the ORBIT
    EXECUTION provenance (exp_11): which implementation produced the
    conditioning (``batched`` vs the legacy per-angle ``loop``), its per-forward
    sample cap, and the source SHA. Rows produced by different orbit executions
    are not interchangeable — the batched path changes the train-mode
    augmentation schedule and regroups the evaluation tail batch — so this is
    what makes "legacy-loop" a checkable label rather than a footnote.

    Under ``rotate_mode='random'`` (exp_14) the record additionally carries the
    §3.3 assignment provenance and nulls ``rotate_deg``; in fixed mode it is
    byte-identical to the legacy record. See :func:`_rotation_provenance`.

    ``frame_avg_max_fwd_samples`` is the cap this evaluation applied; ``None``
    records the module default, i.e. the value every row already in the record
    was produced under (exp_14 pins it to 64 explicitly on both arms).
    """
    execution, cap = orbit_provenance(cond_method, frame_avg_max_fwd_samples)
    record = {
        "metrics": metrics_dict,
        "ckpt_path": ckpt_path,
        "rotate_deg": rotate_deg,
        "cond_method": cond_method,
        "frame_avg_angles": frame_avg_angles,
        "cond_autocast": cond_autocast,
        "orbit_execution": execution,
        "frame_avg_fwd_cap": cap,
        "source_sha": source_sha(),
        # the batched path regroups the split's TAIL batch, so the schedule that
        # produced a row must be reconstructible from the row itself
        "batch_size": batch_size,
        "n_samples": n_samples,
        # ...and so must the rest of the runtime protocol: which split ran, how
        # many items were actually evaluated, and which weights were used. Omitted
        # values stay explicitly None rather than a plausible-looking guess
        # (round-4 review B2).
        "dataset_config": dataset_config,
        "seed": seed,
        "cfg_scale": cfg_scale,
        "steps": steps,
        "eval_name": eval_name,
        "weights_source": weights_source,
        "device": device,
    }
    record = _rotation_provenance(record, rotate_mode, rotate_seed, input_hash,
                                  assignment_hash, stream_count, img_w)
    record = _per_scene_provenance(record, by_scene)
    return _are_provenance(record, are_lambda, are_lambda_source, are_anchor)


def resolve_cond_autocast(mode):
    """Map a ``--cond-autocast`` mode to ``(enabled, dtype)`` for the conditioning call.

    ``'default'`` -> ``(True, None)``: ``torch.amp.autocast(device)`` with no explicit
    dtype, i.e. the torch per-device default (fp16 on cuda) -- byte-identical to the
    exp_01/exp_02 protocol. ``'bf16'`` -> ``(True, torch.bfloat16)``: matches
    ``finetune_cond``'s bf16-mixed training precision. ``'off'`` -> ``(False, None)``:
    no autocast (fp32), for exactness measurements.
    """
    modes = {"default": (True, None), "bf16": (True, torch.bfloat16), "off": (False, None)}
    if mode not in modes:
        raise ValueError(f"Unknown cond_autocast: {mode!r}; valid options: {sorted(modes)}.")
    return modes[mode]


def build_predictions_meta(dataset_config_path, seed, n_samples, cond_method,
                           frame_avg_angles, rotate_deg, batch_size, cond_autocast,
                           rotate_mode='fixed', rotate_seed=None, input_hash=None,
                           assignment_hash=None, stream_count=None, img_w=None,
                           frame_avg_max_fwd_samples=None):
    """Sidecar meta saved by ``--store_predictions`` (read by the exp_02 comparator
    guard). Carries the same orbit-execution provenance as the metrics record:
    at the default evaluation batch the batched path degenerates to one angle per
    call for every full batch, but the split's tail batch is regrouped, so a
    prediction set must name the execution that produced it.

    exp_14 does not store predictions, but a stored set must still name the
    rotation assignment that produced it, so the random-mode provenance is added
    here on exactly the same conditional terms as in the metrics record (review
    B5): nothing changes unless ``--store_predictions`` AND random mode are both
    in play."""
    execution, cap = orbit_provenance(cond_method, frame_avg_max_fwd_samples)
    meta = {
        "dataset_config": dataset_config_path,
        "seed": seed,
        "n_samples": n_samples,
        "cond_method": cond_method,
        "frame_avg_angles": frame_avg_angles,
        "rotate_deg": rotate_deg,
        "batch_size": batch_size,
        "cond_autocast": cond_autocast,
        "orbit_execution": execution,
        "frame_avg_fwd_cap": cap,
        "source_sha": source_sha(),
    }
    return _rotation_provenance(meta, rotate_mode, rotate_seed, input_hash,
                                assignment_hash, stream_count, img_w)


# Unexpected-key prefixes a PL-wrapper checkpoint legitimately leaves after
# evaluate_model's 'diffusion.' strip: EMA copy/bookkeeping and loss-module buffers.
# Verified against outputs_FLAC/ft_vanilla/epoch=0-step=2000.ckpt (1279 keys):
# all 213 leftovers are diffusion_ema.* (212) + losses.* (1). Exported/bare
# checkpoints (FLAC_EMA.ckpt, *_ft.ckpt: 1066 keys) must load with zero of both.
LOAD_WHITELIST_PREFIXES = ("diffusion_ema.", "losses.")


def check_load_integrity(missing, unexpected, allow_partial_load=False,
                         whitelist_prefixes=LOAD_WHITELIST_PREFIXES):
    """Report ``load_state_dict(strict=False)`` results and fail on real mismatches.

    Missing keys are never acceptable (an un-initialized model weight silently
    corrupts metrics); unexpected keys are tolerated only under
    ``whitelist_prefixes``. Any other mismatch raises ``RuntimeError`` unless
    ``allow_partial_load`` is set, which downgrades it to a printed warning.
    """
    missing = list(missing)
    stray = [k for k in unexpected if not k.startswith(whitelist_prefixes)]
    n_benign = len(list(unexpected)) - len(stray)
    print(f"Checkpoint load: {len(missing)} missing, {len(stray)} stray unexpected, "
          f"{n_benign} whitelisted wrapper-leftover keys.")
    if not (missing or stray):
        return
    msg = (f"Checkpoint did not load cleanly: {len(missing)} missing keys "
           f"(first: {missing[:5]}), {len(stray)} stray unexpected keys "
           f"(first: {stray[:5]}). Wrong model-config/checkpoint pairing? "
           "Re-export the checkpoint, or pass --allow-partial-load to continue anyway.")
    if allow_partial_load:
        print("WARNING (--allow-partial-load): " + msg)
    else:
        raise RuntimeError(msg)


def evaluate_model(
    model_config_path,
    dataset_config_path,
    ckpt_path,
    steps,
    cfg_scale,
    batch_size=64,
    num_workers=6,
    eval_name='FLAC_eval', 
    device='cuda' if torch.cuda.is_available() else 'cpu',
    seed=42,
    store_predictions=False,
    rotate_deg=0.0,
    cond_method='vanilla',
    frame_avg_angles=None,
    frame_avg_max_fwd_samples=None,
    cond_autocast='default',
    allow_partial_load=False,
    rotate_mode='fixed',
    rotate_seed=None,
    expected_stream_count=None,
    record_stream=False,
    record_per_scene=False,
    are_lambda=None,
):
    # Fail fast on an unknown cond_method (the CLI is guarded by argparse
    # choices, but programmatic callers would otherwise silently run vanilla
    # while filenames/meta record the unknown method).
    if cond_method not in ('vanilla', 'fa_invariant'):
        raise ValueError(
            f"Unknown cond_method: {cond_method!r}; valid options: 'vanilla', 'fa_invariant'."
        )

    # The EVALUATION chunk plan, validated before any file/model work for the same
    # reason as the rotation protocol below: a misconfigured cap must never reach a
    # GPU. It is a separate, explicitly pinned quantity from the cap a checkpoint
    # was TRAINED under (exp_14 plan §3.1) -- both arms are evaluated at one common
    # cap, and the value is recorded in the metrics row.
    try:
        frame_avg_max_fwd_samples = resolve_frame_avg_cap(frame_avg_max_fwd_samples)
    except ValueError as err:                      # name the operator-facing knob
        raise ValueError(
            f"--frame-avg-max-fwd-samples (frame_avg_max_fwd_samples): {err}"
        ) from err

    # Resolve (and validate) the rotation protocol before any file/model work:
    # a misconfigured rotation must never reach a GPU (see resolve_rotation_plan).
    rotation_plan = resolve_rotation_plan(rotate_mode, rotate_deg, rotate_seed, seed)

    # An expectation that cannot be checked must not be quietly accepted: a plain
    # fixed-mode run accumulates no stream, so there is nothing to count.
    accumulate_stream = rotation_plan.is_random or bool(record_stream)
    if expected_stream_count is not None and not accumulate_stream:
        raise ValueError(
            "expected_stream_count (--expected-stream-count) is only checkable when an "
            "assignment stream is accumulated, i.e. under --rotate-mode random or "
            "--record-stream. Passing it otherwise would record an expectation that "
            "was never verified."
        )

    # Fail fast on an unknown cond_autocast too (full-review condition C1).
    ac_enabled, ac_dtype = resolve_cond_autocast(cond_autocast)

    def cond_autocast_ctx():
        """Fresh autocast context for one conditioning call (same semantics per batch)."""
        if not ac_enabled:
            return contextlib.nullcontext()
        if ac_dtype is None:
            return torch.amp.autocast(device)  # per-device default: exp_01/02 protocol
        return torch.amp.autocast(device, dtype=ac_dtype)

    torch.set_float32_matmul_precision('medium')

    # Resolve the C4 frame-average angles (only used when cond_method == 'fa_invariant').
    if frame_avg_angles is None:
        frame_avg_angles = DEFAULT_FRAME_ANGLES
    frame_avg_angles = tuple(float(a) for a in frame_avg_angles)

    # Load configurations
    with open(model_config_path) as f:
        model_config = json.load(f)

    training_config = model_config.get('training', None)
    
    # Load ckpt
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['state_dict']
    for key in list(state_dict.keys()):
        if key.startswith('diffusion.'):
            new_key = key.replace('diffusion.', '')
            state_dict[new_key] = state_dict.pop(key)
    
    # Use EMA weights if available (the resolved source is recorded, not assumed)
    weights_source = resolve_weights_source(training_config, state_dict.keys())
    if training_config.get("use_ema", False) and any(k.startswith('diffusion_ema.ema_model.') for k in state_dict.keys()):
        print('Using EMA model')
        for key in list(state_dict.keys()):
            if key.startswith('diffusion_ema.ema_model.'):
                new_key = key.replace('diffusion_ema.ema_model.', 'model.')
                state_dict[new_key] = state_dict.pop(key)
        training_config['use_ema'] = False

    # Build model; assert the checkpoint actually loaded (full-review condition C2).
    model = create_model_from_config(model_config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    check_load_integrity(missing, unexpected, allow_partial_load)

    model_type = model_config.get('model_type', None)
    assert model_type is not None, 'model_type must be specified in model config'

    model_config['training'] = training_config
    module = create_training_wrapper_from_config(model_config, model)
    module.eval().requires_grad_(False)
    module.to(device)
    
    with torch.amp.autocast(device):
        model = module.diffusion.model

    if module.diffusion.pretransform is not None:
            samples = model_config["sample_size"] // module.diffusion.pretransform.downsampling_ratio
    else:
        samples = model_config["sample_size"]

    # exp_16 (ARE): the sampler emits the RESIDUAL, so the query anchor is added
    # back to the latent before the decode. The bank is built once, from the same
    # frozen VAE the arm trained against, and its worst-case frame check runs here
    # -- before any GPU sampling -- rather than mid-loop.
    are_lambda, are_lambda_source = resolve_are_lambda(training_config, are_lambda)
    are_bank, are_anchor_cfg = None, None
    if are_lambda is not None:
        are_anchor_cfg = dict(training_config[ARE_ANCHOR_KEY])
        are_bank = build_anchor_bank(module.diffusion.pretransform,
                                     model_config["sample_rate"],
                                     model_config["sample_size"], are_anchor_cfg)
        print(f"ARE add-back ENABLED: lambda={are_lambda} (source: {are_lambda_source}), "
              f"anchor={are_anchor_cfg}, kept latent frames 0..{are_bank.cfg.early_frames - 1}")

    # Fix seed
    if isinstance(seed, str):
        seed = int(seed)
    pl.seed_everything(seed, workers=True)

    # Dataloader Eval
    with open(dataset_config_path) as f:
        dataset_config = json.load(f)

    eval_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=batch_size,
        num_workers=num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 1),
        shuffle=False
    )

    # Metrics 
    # per_scene=True makes the callback accumulate a second, per-scene set of
    # metric objects and return them under `by_scene` (exp_14 round-3 fix R3F1).
    # Off by default: the legacy record shape is frozen.
    metric_callback = create_metric_callback_from_config(
        model_config, dataset_id=dataset_config['datasets'][0]['id'],
        per_scene=record_per_scene)

    # Yaw-rotation state. The random draw lives on its OWN cpu generator so it
    # cannot advance the global RNG that seeds the diffusion noise -- otherwise a
    # rotated cell and its paired unrotated cell would not share sampling noise.
    rot_generator = None
    rot_stream = RotationStream() if accumulate_stream else None
    if rotation_plan.is_random:
        rot_generator = torch.Generator(device='cpu')
        rot_generator.manual_seed(rotation_plan.rotate_seed)
        print(f"Random yaw rotation: per-sample offsets, rotation seed "
              f"{rotation_plan.rotate_seed}")

    # Eval
    c=0
    if store_predictions:
        decoded_samples = []

    with torch.no_grad():
        for batch in tqdm(eval_dl, desc="Evaluating"):
            reals, metadata = batch
            reals = reals.to(device)

            # Optional yaw rotation: physically rotate the conditioning (depth
            # panorama + source/context poses) while leaving context_audio and the
            # target RIR fixed. Fixed mode = one angle for the whole run; random
            # mode = an independent per-sample draw. See src/data/yaw_rotation.py.
            metadata = apply_rotation_plan(metadata, rotation_plan, rot_generator,
                                           rot_stream)

            with cond_autocast_ctx():
                if cond_method == 'fa_invariant':
                    # Route-1 symmetrized conditioning: cylindrical pose invariants
                    # + C4 frame average of the ViT depth path. Applied AFTER the
                    # optional --rotate-deg above (that composition is the sanity check).
                    conditioning = invariant_conditioning(
                        module.diffusion.conditioner, metadata, module.device, frame_avg_angles,
                        max_fwd_samples=frame_avg_max_fwd_samples,
                    )
                else:
                    conditioning = module.diffusion.conditioner(metadata, module.device)
            cond_inputs = module.diffusion.get_conditioning_inputs(conditioning)

            noise = torch.randn([reals.shape[0], module.diffusion.io_channels, samples]).to(module.device)

            if hasattr(model, "diffusion_objective"):
                objective = model.diffusion_objective
            else:
                objective = getattr(model, "objective", "rectified_flow")

            if objective == "v":
                from src.inference.sampling import sample
                fakes = sample(model, noise, steps, 0, **cond_inputs, cfg_scale=cfg_scale, dist_shift=module.diffusion.dist_shift, batch_cfg=True)
            elif objective == "rectified_flow":
                from src.inference.sampling import sample_discrete_euler
                fakes = sample_discrete_euler(model, noise, steps, **cond_inputs, cfg_scale=cfg_scale, dist_shift=module.diffusion.dist_shift, batch_cfg=True, disable_tqdm=True)
            elif objective == "rf_denoiser":
                from src.inference.sampling import sample_flow_pingpong
                logsnr = torch.linspace(-6, 2, steps+1).to(module.device)
                sigmas = torch.sigmoid(-logsnr)
                sigmas[0] = 1.0
                sigmas[-1] = 0.0
                fakes = sample_flow_pingpong(model, noise, sigmas=sigmas, **cond_inputs, cfg_scale=cfg_scale, dist_shift=module.diffusion.dist_shift, batch_cfg=True, disable_tqdm=True)
            else:
                raise ValueError(f"Unknown diffusion objective: {objective}")

            # exp_16 (ARE): + lambda * A_query, on the LATENT, before the decode
            if are_bank is not None:
                fakes = apply_are_addback(fakes, are_bank, are_lambda, metadata,
                                          module.device)

            # Decode
            if module.diffusion.pretransform is not None:
                fakes = module.diffusion.pretransform.decode(fakes)
            
            if store_predictions:
                decoded_samples.append(fakes.cpu())
            
            # Clamp and pad if necessary
            fakes = fakes.clamp(-1.0, 1.0) 
            if fakes.shape != reals.shape:
                if fakes.shape[-1] < reals.shape[-1]:
                    fakes = torch.nn.functional.pad(fakes, (0, reals.shape[-1] - fakes.shape[-1]))
                else:
                    reals = torch.nn.functional.pad(reals, (0, fakes.shape[-1] - reals.shape[-1]))
    
            # Compute metrics
            scene_list = [md["scene"] for md in metadata]
            depth_list = [md["depth"] if 'depth' in md else None for md in metadata]
            query_list = [md["source"] if 'source' in md else None for md in metadata]
            depthMinusSource_list = [(d[:3, :, :] - source_pose[:, None, None]).unsqueeze(0).float().to(device) for d, source_pose in zip(depth_list, query_list)]
            metric_callback.update_metrics("test", fakes, reals, scene_list, depth=depthMinusSource_list)
            c += reals.shape[0]
    

    # Assignment integrity (plan §3.3). The dataset silently substitutes items on
    # a load/silence failure, so a random cell only reports numbers once its
    # stream is proven to be one position per split item; the check runs BEFORE
    # the metrics are computed or written.
    rotation_provenance = {}
    if rot_stream is not None:
        dataset = getattr(eval_dl, 'dataset', None)
        verify_stream_count(rot_stream, None if dataset is None else len(dataset),
                            expected_stream_count)
        verify_stream_positions(rot_stream)
    if rotation_plan.is_random:
        rotation_provenance = {
            "rotate_mode": rotation_plan.mode,
            "rotate_seed": rotation_plan.rotate_seed,
            "input_hash": rot_stream.input_hash(),
            "assignment_hash": rot_stream.assignment_hash(),
            "stream_count": len(rot_stream),
            "img_w": rot_stream.img_w,
        }
        print(f"Rotation assignment: {rotation_provenance['stream_count']} positions, "
              f"input_hash {rotation_provenance['input_hash']}, "
              f"assignment_hash {rotation_provenance['assignment_hash']}")

    # Compute and print metrics
    metrics_dict, by_scene = resolve_metrics_payload(
        metric_callback.compute_metrics("test"), record_per_scene)
    if by_scene is not None:
        print(f"Per-scene metrics recorded for {len(by_scene)} scene(s)")
    for metric_name, metric_value in metrics_dict.items():
        if metric_name == 'T60' or 'to' in metric_name:
            metric_name += ' (%)'
        elif metric_name == 'EDT':
            metric_name += ' (ms)'
        elif metric_name == 'C50':
            metric_name += ' (dB)'
        print('Test/' + metric_name, metric_value)
    
    # Save metrics in a file
    output_paths = build_output_paths(
        ckpt_path, steps, cfg_scale, eval_name,
        cond_method=cond_method, rotate_deg=rotate_deg, n_angles=len(frame_avg_angles),
        rotate_mode=rotation_plan.mode, rotate_seed=rotation_plan.rotate_seed,
        are_lambda=are_lambda,
    )
    frame_angles_record = list(frame_avg_angles) if cond_method == 'fa_invariant' else None
    metrics_to_save = build_metrics_record(
        metrics_dict, ckpt_path, rotate_deg, cond_method, frame_angles_record,
        cond_autocast=cond_autocast, batch_size=batch_size, n_samples=c,
        dataset_config=dataset_config_path, seed=seed, cfg_scale=cfg_scale,
        steps=steps, eval_name=eval_name, weights_source=weights_source,
        device=str(device), by_scene=by_scene,
        frame_avg_max_fwd_samples=frame_avg_max_fwd_samples,
        are_lambda=are_lambda, are_lambda_source=are_lambda_source,
        are_anchor=are_anchor_cfg, **rotation_provenance,
    )
    path2save = output_paths['metrics']
    with open(path2save, 'w') as f:
        json.dump(metrics_to_save, f, indent=4)

    print(f"Metrics saved to {path2save}")

    # Assignment sidecar (opt-in). Written only after every stream check above has
    # passed, so its existence is itself evidence the cell validated.
    if record_stream:
        path2save_stream = stream_sidecar_path(path2save)
        with open(path2save_stream, 'w') as f:
            json.dump(build_stream_record(rotation_plan, rot_stream), f, indent=2)
        print(f"Assignment stream saved to {path2save_stream}")

    if store_predictions:
        decoded_samples_all = torch.cat(decoded_samples, dim=0)
        path2save_preds = output_paths['predictions']
        preds_bundle = {
            "predictions": decoded_samples_all,
            "meta": build_predictions_meta(
                dataset_config_path, seed, int(decoded_samples_all.shape[0]),
                cond_method, frame_angles_record, rotate_deg, batch_size, cond_autocast,
                frame_avg_max_fwd_samples=frame_avg_max_fwd_samples, **rotation_provenance,
            ),
        }
        torch.save(preds_bundle, path2save_preds)
        print(f"Decoded samples saved to {path2save_preds}")

    print("Evaluation complete!")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--dataset-config", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--cfg-scale", type=float, default=1.0, help="Classifier-free guidance scale")
    parser.add_argument("--steps", type=int, default=1, help="Number of diffusion steps")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the evaluation on")
    parser.add_argument("--eval-name", type=str, default='', help="Name of the evaluation run (optional)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for evaluation")
    parser.add_argument("--store_predictions", action='store_true', help="Whether to store predictions or not")
    parser.add_argument("--rotate-deg", type=float, default=0.0, help="Yaw-rotate the conditioning (depth + poses) by this many degrees before eval; 0 disables (default). Fixed mode only.")
    parser.add_argument("--rotate-mode", type=str, default="fixed", choices=["fixed", "random"], help="Yaw-rotation protocol: 'fixed' (default) rotates every sample by --rotate-deg; 'random' (exp_14) draws an independent panorama-column offset per sample from a generator seeded with --rotate-seed. The two are mutually exclusive: 'random' with a non-zero --rotate-deg is an error.")
    parser.add_argument("--rotate-seed", type=int, default=None, help="Seed for the per-sample random yaw draw; defaults to --seed. Only valid with --rotate-mode random (passing it in fixed mode is an error, never a silent no-op).")
    parser.add_argument("--record-stream", action='store_true', help="Write a <metrics-stem>.stream.json sidecar carrying the full per-position assignment audit (canonical input tuples, offsets, both hashes). Works in fixed mode too, which is how an unrotated Z cell gets an input_hash to be paired against. The metrics record and its path are unaffected.")
    parser.add_argument("--record-per-scene", action='store_true', help="Record a `by_scene` block in the metrics JSON: per-scene, per-metric values plus the scene ids (exp_14). This is what makes the PER-SCENE mean -- the plan's estimand and this repo's headline convention -- computable from the artifact. Off by default; without it the record is byte-identical to every row already committed.")
    parser.add_argument("--expected-stream-count", type=int, default=None, help="Pre-registered number of items in the evaluated split (e.g. 6337 for the AR unseen split). When given, the run fails unless the dataset AND the accumulated assignment stream both hold exactly this many items; without it, the stream can only be checked against the dataset it came from.")
    parser.add_argument("--cond-method", type=str, default="vanilla", choices=["vanilla", "fa_invariant"], help="Conditioning method: 'vanilla' (single conditioner pass) or 'fa_invariant' (cylindrical pose invariants + C4 ViT frame average). Composes with --rotate-deg (rotation applied first).")
    parser.add_argument("--frame-avg-angles", type=str, default=",".join(str(int(a)) for a in DEFAULT_FRAME_ANGLES), help="Comma-separated yaw angles in degrees for fa_invariant frame averaging; the first must be 0. Ignored when --cond-method vanilla.")
    parser.add_argument("--frame-avg-max-fwd-samples", type=int, default=64, help="Largest number of samples in ONE fa_invariant frame-averaging forward. With C4 and a per-forward batch B this makes max(1, cap // B) angles share a chunk -- and, under train-mode DINOv3, one RoPE rescale draw (announcement 06). Default 64 = the value every evaluation in the record ran under; exp_14 pins it explicitly on both arms. Ignored when --cond-method vanilla.")
    parser.add_argument("--cond-autocast", type=str, default="default", choices=["default", "bf16", "off"], help="Autocast mode for the conditioning call: 'default' = torch per-device default dtype (fp16 on cuda; the exp_01/exp_02 protocol), 'bf16' = bfloat16 (matches finetune_cond's bf16-mixed training), 'off' = no autocast (fp32, for exactness measurements).")
    parser.add_argument("--allow-partial-load", action='store_true', help="Continue with a warning when the checkpoint does not load cleanly (missing or non-whitelisted unexpected keys) instead of raising.")
    parser.add_argument("--are-lambda", type=float, default=None, help="exp_16 ARE: weight of the analytic direct-sound anchor added back to the sampled latent before decoding. Omit to use the arm's OWN training.are_lambda from --model-config (None for every non-ARE arm, which leaves the run and its record byte-identical to today). Passing it OVERRIDES the config -- including 0.0, which is the AR3 eval-time sweep and is recorded as such. Requires the model config to declare training.are_anchor; the applied value and its source are written into the metrics record and into the output filename.")
    args = parser.parse_args()

    if args.store_predictions:
        print('Warning: Storing predictions can use a lot of memory.')

    frame_avg_angles = tuple(float(a) for a in args.frame_avg_angles.split(","))

    evaluate_model(
        args.model_config,
        args.dataset_config,
        args.ckpt_path,
        cfg_scale=args.cfg_scale,
        steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        eval_name=args.eval_name,
        seed=args.seed,
        store_predictions=args.store_predictions,
        rotate_deg=args.rotate_deg,
        cond_method=args.cond_method,
        frame_avg_angles=frame_avg_angles,
        frame_avg_max_fwd_samples=args.frame_avg_max_fwd_samples,
        cond_autocast=args.cond_autocast,
        allow_partial_load=args.allow_partial_load,
        rotate_mode=args.rotate_mode,
        rotate_seed=args.rotate_seed,
        expected_stream_count=args.expected_stream_count,
        record_stream=args.record_stream,
        record_per_scene=args.record_per_scene,
        are_lambda=args.are_lambda,
    )
