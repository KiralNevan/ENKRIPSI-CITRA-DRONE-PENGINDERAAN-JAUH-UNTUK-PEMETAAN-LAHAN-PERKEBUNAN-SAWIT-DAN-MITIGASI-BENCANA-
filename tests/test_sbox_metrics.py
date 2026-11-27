import numpy as np
import pytest

from src.aes.constants import AES_SBOX, inverse_sbox
from src.sbox.metrics import evaluate_sbox, is_balanced, is_bijective


def test_aes_sbox_known_metrics():
    m = evaluate_sbox(AES_SBOX)
    assert m["nl"] == 112
    assert m["bic_nl"] == 112
    assert m["lp"] == pytest.approx(0.015625)
    assert m["dap"] == pytest.approx(0.015625)
    assert m["sac"] == pytest.approx(0.5048828125)
    assert m["bic_sac"] == pytest.approx(0.5046037946428571)
    assert m["balanced"] is True
    assert m["bijective"] is True


def test_inverse_sbox():
    inv = inverse_sbox(AES_SBOX)
    x = np.arange(256, dtype=np.uint8)
    assert np.array_equal(inv[AES_SBOX], x)
    assert np.array_equal(AES_SBOX[inv], x)


def test_random_permutation_is_bijective_balanced():
    rng = np.random.default_rng(42)
    sbox = rng.permutation(256).astype(np.uint8)
    assert is_bijective(sbox)
    assert is_balanced(sbox)
    m = evaluate_sbox(sbox)
    assert 70 <= m["nl"] <= 112
    assert 0.0 <= m["lp"] <= 1.0
    assert 0.0 <= m["dap"] <= 1.0


def test_non_bijective_rejected():
    bad = np.zeros(256, dtype=np.uint8)
    with pytest.raises(ValueError):
        evaluate_sbox(bad)
