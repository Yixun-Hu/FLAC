"""Tests for ``eval_localization.py`` (exp_18 loc_invert, round 3 driver).

Written test-first (announcement 02). Contracts: ``loc_invert_impl_contracts.md``
§4.5, serving plan Rev 3 §2. The driver owns the fail-closed identity audit
(§2.1), the keyed noise bank (§2.3), the candidate-major query layout, the JSONL
row/summary records and the ``eval_FLAC`` parity harness (C8); everything the
``src.localization`` modules already provide is reused, never re-derived.
"""
import hashlib

import pytest

import eval_localization as el


# --------------------------------------------------------------------------- #
# fixtures: a split whose identities can be built without loading any audio
# --------------------------------------------------------------------------- #
class _FakeDataset:
    """Stand-in for ``SampleDataset``: the ordered file list plus its roots."""

    def __init__(self, filenames, root_paths):
        self.filenames = list(filenames)
        self.root_paths = list(root_paths)

    def __len__(self):
        return len(self.filenames)


_ROOT = "/data/AcousticRooms"
_FILES = [
    f"{_ROOT}/single_channel_ir_1/Cafe/Cafe_idx_1/S00{s}_R003_hybrid_IR.wav" for s in (0, 7)
] + [f"{_ROOT}/single_channel_ir_1/Bedrooms/Bedrooms_idx_2/S002_R0011_hybrid_IR.wav"]


def _md(idx, filename, root=_ROOT):
    import os
    return {"idx": idx, "path": filename, "relpath": os.path.relpath(filename, root)}


def _loader(dataset, batch_size=2):
    """Mimic the eval dataloader: yields ``(reals, [md, ...])`` batches in order."""
    metadata = [_md(i, f) for i, f in enumerate(dataset.filenames)]
    for start in range(0, len(metadata), batch_size):
        chunk = metadata[start:start + batch_size]
        yield (None, chunk)


# --------------------------------------------------------------------------- #
# expected_split_identities / split_hash / audit_split_identities  (unit a)
# --------------------------------------------------------------------------- #
def test_expected_split_identities_mirrors_the_loader_without_reading_audio():
    dataset = _FakeDataset(_FILES, [_ROOT])
    expected = el.expected_split_identities(dataset)
    assert expected == [
        "0|single_channel_ir_1/Cafe/Cafe_idx_1/S000_R003_hybrid_IR.wav",
        "1|single_channel_ir_1/Cafe/Cafe_idx_1/S007_R003_hybrid_IR.wav",
        "2|single_channel_ir_1/Bedrooms/Bedrooms_idx_2/S002_R0011_hybrid_IR.wav",
    ]


def test_expected_split_identities_falls_back_to_the_absolute_path():
    dataset = _FakeDataset(_FILES[:1], ["/somewhere/else"])
    assert el.expected_split_identities(dataset) == [f"0|{_FILES[0]}"]


def test_split_hash_is_the_pinned_serialization():
    """LF-joined identities, UTF-8, sha256 -- pinned so the recorded split hash is
    comparable across machines and runs."""
    ids = ["0|a.wav", "1|b.wav"]
    assert el.split_hash(ids) == hashlib.sha256("0|a.wav\n1|b.wav".encode("utf-8")).hexdigest()
    assert el.split_hash(ids) != el.split_hash(list(reversed(ids)))       # order matters


def test_audit_split_identities_clean_pass_returns_the_split_hash():
    dataset = _FakeDataset(_FILES, [_ROOT])
    expected = el.expected_split_identities(dataset)
    assert el.audit_split_identities(_loader(dataset), expected) == el.split_hash(expected)


def test_audit_split_identities_aborts_on_an_injected_substitution():
    """SampleDataset silently substitutes a random other item on a load failure;
    the first such position must abort the run (plan §2.1)."""
    dataset = _FakeDataset(_FILES, [_ROOT])
    expected = el.expected_split_identities(dataset)

    def substituted():
        for reals, batch in _loader(dataset):
            patched = [dict(md) for md in batch]
            for md in patched:
                if md["idx"] == 1:
                    md["idx"], md["relpath"] = 2, "single_channel_ir_1/X/X_idx_0/S001_R001_hybrid_IR.wav"
            yield (reals, patched)

    with pytest.raises(SystemExit):
        el.audit_split_identities(substituted(), expected)


def test_audit_split_identities_aborts_on_a_short_or_long_stream():
    dataset = _FakeDataset(_FILES, [_ROOT])
    expected = el.expected_split_identities(dataset)
    short = [(None, [_md(0, _FILES[0])])]
    with pytest.raises(SystemExit):
        el.audit_split_identities(short, expected)
    with pytest.raises(SystemExit):
        el.audit_split_identities(_loader(dataset), expected[:2])


def test_audit_split_identities_accepts_a_plain_metadata_iterable():
    dataset = _FakeDataset(_FILES, [_ROOT])
    expected = el.expected_split_identities(dataset)
    stream = [_md(i, f) for i, f in enumerate(dataset.filenames)]
    assert el.audit_split_identities(stream, expected) == el.split_hash(expected)


# --------------------------------------------------------------------------- #
# build_noise_bank  (unit b) -- plan §2.3: deterministic, keyed by
# (seed, query_id, k), shared across candidates (common random numbers, C10),
# and never drawn from the global stream.
# --------------------------------------------------------------------------- #
import torch                                    # noqa: E402
from src.localization.scoring import noise_key  # noqa: E402

_LATENT = (4, 16)


def test_build_noise_bank_shape_and_dtype():
    bank = el.build_noise_bank(42, "q0", 8, _LATENT)
    assert tuple(bank.shape) == (8, 4, 16)
    assert bank.dtype == torch.float32 and bank.device.type == "cpu"


def test_build_noise_bank_is_keyed_by_scoring_noise_key():
    """Pins the bank to scoring.noise_key: sample k is exactly what a generator
    seeded with noise_key(seed, query_id, k) draws."""
    bank = el.build_noise_bank(42, "Cafe/Cafe_idx_1/S008_R089", 3, _LATENT)
    for k in range(3):
        g = torch.Generator().manual_seed(noise_key(42, "Cafe/Cafe_idx_1/S008_R089", k))
        assert torch.equal(bank[k], torch.randn(_LATENT, generator=g))


def test_build_noise_bank_is_reproducible_and_query_specific():
    a = el.build_noise_bank(42, "q0", 4, _LATENT)
    assert torch.equal(a, el.build_noise_bank(42, "q0", 4, _LATENT))      # resume-safe
    assert not torch.equal(a, el.build_noise_bank(42, "q1", 4, _LATENT))
    assert not torch.equal(a, el.build_noise_bank(43, "q0", 4, _LATENT))
    assert not torch.equal(a[0], a[1])                                    # k is load-bearing


def test_build_noise_bank_never_touches_the_global_rng():
    before = torch.random.get_rng_state()
    el.build_noise_bank(42, "q0", 8, _LATENT)
    assert torch.equal(torch.random.get_rng_state(), before)


def test_build_noise_bank_prefix_is_stable_in_k():
    """A K=8 bank starts with the K=4 bank: enlarging K cannot renumber samples."""
    small = el.build_noise_bank(42, "q0", 4, _LATENT)
    large = el.build_noise_bank(42, "q0", 8, _LATENT)
    assert torch.equal(large[:4], small)


def test_build_noise_bank_rejects_bad_arguments():
    with pytest.raises(ValueError):
        el.build_noise_bank(42, "q0", 0, _LATENT)
    with pytest.raises(ValueError):
        el.build_noise_bank(42, "q0", 4, (4,))
    with pytest.raises(ValueError):
        el.build_noise_bank(42, "q0", 4, (4, 0))
