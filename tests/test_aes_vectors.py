import os

import numpy as np
import pytest
from Crypto.Cipher import AES as RefAES

from src.aes.constants import AES_SBOX, inverse_sbox
from src.aes.core import decrypt_blocks, encrypt_blocks, expand_key
from src.aes.modes import decrypt_bytes, encrypt_bytes, pkcs7_pad


KEY = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
IV = bytes.fromhex("101112131415161718191a1b1c1d1e1f")


def test_fips_197_c1():
    plaintext = bytes.fromhex("00112233445566778899aabbccddeeff")
    expected = bytes.fromhex("69c4e0d86a7b0430d8cdb78070b4c55a")
    rk = expand_key(KEY, AES_SBOX)
    ct = encrypt_blocks(
        np.frombuffer(plaintext, dtype=np.uint8).reshape(1, 16), rk, AES_SBOX
    ).tobytes()
    assert ct == expected
    pt = decrypt_blocks(
        np.frombuffer(ct, dtype=np.uint8).reshape(1, 16), rk, inverse_sbox(AES_SBOX)
    ).tobytes()
    assert pt == plaintext


@pytest.mark.parametrize("mode", ["ECB", "CBC"])
def test_ecb_cbc_match_pycryptodome(mode):
    data = os.urandom(256)
    padded = pkcs7_pad(data)
    cipher, _ = encrypt_bytes(data, KEY, mode, AES_SBOX, IV)
    if mode == "ECB":
        ref = RefAES.new(KEY, RefAES.MODE_ECB).encrypt(padded)
    else:
        ref = RefAES.new(KEY, RefAES.MODE_CBC, IV).encrypt(padded)
    assert cipher == ref
    assert decrypt_bytes(cipher, KEY, mode, AES_SBOX, IV) == data


@pytest.mark.parametrize("mode", ["ECB", "CBC"])
def test_custom_sbox_roundtrip(mode):
    rng = np.random.default_rng(123)
    sbox = rng.permutation(256).astype(np.uint8)
    data = os.urandom(4096)
    cipher, _ = encrypt_bytes(data, KEY, mode, sbox, IV)
    assert decrypt_bytes(cipher, KEY, mode, sbox, IV) == data


def test_pkcs7_padding():
    data = os.urandom(31)
    padded = pkcs7_pad(data)
    assert len(padded) == 32
    assert padded[-1] == 1
