import numpy as np
import pytest

from src.aes.constants import AES_SBOX
from src.aes.modes import cipher_to_image, decrypt_image_bytes, encrypt_image_array
from src.image.metrics import entropy, image_metrics, npcr_uaci


def test_entropy_uniform():
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    assert 7.9 < entropy(img) <= 8.0


def test_npcr_uaci_identical_and_different():
    a = np.zeros((16, 16), dtype=np.uint8)
    assert npcr_uaci(a, a) == (0.0, 0.0)
    b = a.copy()
    b[0, 0] = 1
    npcr, uaci = npcr_uaci(a, b)
    assert npcr == pytest.approx(100.0 / (16 * 16))
    assert uaci == pytest.approx(100.0 / 255.0 / (16 * 16))


def test_image_roundtrip_lossless():
    rng = np.random.default_rng(11)
    img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    key = bytes(range(16))
    iv = bytes(range(16, 32))
    for mode in ("ECB", "CBC"):
        cipher, meta = encrypt_image_array(img, key, mode, AES_SBOX, iv)
        decrypted = decrypt_image_bytes(cipher, key, img.shape, mode, AES_SBOX, iv)
        cipher_img = cipher_to_image(cipher, img.shape)
        assert np.array_equal(decrypted, img)
        m = image_metrics(img, cipher_img, decrypted)
        assert m["mse"] == 0.0
        assert m["hash_match"] is True
        assert m["entropy_cipher"] > 7.9
