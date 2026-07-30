"""S1: fail-closed dual-prefix extraction (plan R2-3)."""
import pytest, torch
from conftest import load_mod

et = load_mod("extract_teacher")


def sd_fixture(n=3, tamper=None, drop=None, extra=False):
    sd = {}
    for i in range(n):
        t = torch.randn(2, 2)
        sd[f"{et.P_SRC}k{i}"] = t
        sd[f"{et.P_CTX}k{i}"] = t.clone() if tamper != i else t + 1.0
    if drop is not None:
        del sd[f"{et.P_CTX}k{drop}"]
    if extra:
        sd[f"{et.P_CTX}k_extra"] = torch.zeros(1)
    sd["diffusion.something_else"] = torch.zeros(1)
    return sd


def test_happy_extract():
    out = et.extract(sd_fixture(), n_keys=3)
    assert set(out) == {"k0", "k1", "k2"}


def test_tensor_mismatch_refused():
    with pytest.raises(SystemExit) as e:
        et.extract(sd_fixture(tamper=1), n_keys=3)
    assert "shared-backbone premise FALSE" in str(e.value)


def test_count_mismatch_refused():
    with pytest.raises(SystemExit) as e:
        et.extract(sd_fixture(drop=2), n_keys=3)
    assert "key counts" in str(e.value)


def test_suffix_set_mismatch_refused():
    with pytest.raises(SystemExit) as e:
        et.extract(sd_fixture(drop=2, extra=True), n_keys=3)
    assert "suffix sets" in str(e.value)
