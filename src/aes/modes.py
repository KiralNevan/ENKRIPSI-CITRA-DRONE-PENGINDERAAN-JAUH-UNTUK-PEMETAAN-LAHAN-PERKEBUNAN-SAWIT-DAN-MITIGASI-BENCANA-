"""Mode operasi AES (ECB/CBC), padding, dan helper enkripsi citra."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from .constants import (
    AES_SBOX,
    INV_SHIFT_IDX,
    MUL2,
    MUL3,
    MUL9,
    MUL11,
    MUL13,
    MUL14,
    SHIFT_IDX,
    inverse_sbox,
)
from .core import (
    cbc_decrypt_blocks,
    cbc_encrypt_blocks,
    decrypt_blocks,
    encrypt_blocks,
    expand_key,
)


@dataclass
class ImageMeta:
    shape: tuple[int, ...]
    original_len: int
    padded_len: int
    padding_len: int
    mode: str
    algorithm: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "original_len": self.original_len,
            "padded_len": self.padded_len,
            "padding_len": self.padding_len,
            "mode": self.mode,
            "algorithm": self.algorithm,
        }


def pkcs7_pad(data: bytes, block_size: int = 16) -> bytes:
    n = block_size - (len(data) % block_size)
    return data + bytes([n]) * n


def pkcs7_unpad(data: bytes) -> bytes:
    if not data:
        raise ValueError("Data kosong")
    n = data[-1]
    if n < 1 or n > 16:
        raise ValueError("Padding PKCS#7 tidak valid")
    if data[-n:] != bytes([n]) * n:
        raise ValueError("Padding PKCS#7 tidak konsisten")
    return data[:-n]


def derive_key(text: str) -> bytes:
    """Turunkan kunci 16 byte dari teks (demo, bukan KDF produksi)."""
    return hashlib.sha256(text.encode("utf-8")).digest()[:16]


def random_iv() -> bytes:
    return os.urandom(16)


def _as_blocks(data: bytes) -> np.ndarray:
    if len(data) % 16 != 0:
        raise ValueError("Panjang data harus kelipatan 16 byte")
    return np.frombuffer(data, dtype=np.uint8).reshape(-1, 16).copy()


def encrypt_bytes(
    data: bytes,
    key: bytes,
    mode: str = "ECB",
    sbox: np.ndarray = AES_SBOX,
    iv: bytes | None = None,
) -> tuple[bytes, ImageMeta]:
    mode = mode.upper()
    if mode not in {"ECB", "CBC"}:
        raise ValueError("Mode harus ECB atau CBC")
    if len(key) != 16:
        raise ValueError("Kunci AES-128 harus 16 byte")

    padded = pkcs7_pad(data)
    blocks = _as_blocks(padded)
    round_keys = expand_key(key, sbox)

    if mode == "ECB":
        cipher_blocks = encrypt_blocks(blocks, round_keys, sbox)
    else:
        if iv is None:
            iv = random_iv()
        if len(iv) != 16:
            raise ValueError("IV harus 16 byte")
        cipher_blocks = cbc_encrypt_blocks(
            blocks,
            round_keys,
            np.asarray(sbox, dtype=np.uint8).reshape(256),
            SHIFT_IDX,
            MUL2,
            MUL3,
            np.frombuffer(iv, dtype=np.uint8).copy(),
        )

    cipher = cipher_blocks.tobytes()
    meta = ImageMeta(
        shape=(),
        original_len=len(data),
        padded_len=len(padded),
        padding_len=len(padded) - len(data),
        mode=mode,
    )
    return cipher, meta


def decrypt_bytes(
    cipher: bytes,
    key: bytes,
    mode: str = "ECB",
    sbox: np.ndarray = AES_SBOX,
    iv: bytes | None = None,
) -> bytes:
    mode = mode.upper()
    if mode not in {"ECB", "CBC"}:
        raise ValueError("Mode harus ECB atau CBC")
    if len(key) != 16:
        raise ValueError("Kunci AES-128 harus 16 byte")
    if len(cipher) % 16 != 0:
        raise ValueError("Ciphertext harus kelipatan 16 byte")

    blocks = _as_blocks(cipher)
    round_keys = expand_key(key, sbox)
    inv = inverse_sbox(sbox)

    if mode == "ECB":
        plain_blocks = decrypt_blocks(blocks, round_keys, inv)
    else:
        if iv is None or len(iv) != 16:
            raise ValueError("IV 16 byte diperlukan untuk dekripsi CBC")
        plain_blocks = cbc_decrypt_blocks(
            blocks,
            round_keys,
            inv,
            INV_SHIFT_IDX,
            MUL9,
            MUL11,
            MUL13,
            MUL14,
            np.frombuffer(iv, dtype=np.uint8).copy(),
        )

    return pkcs7_unpad(plain_blocks.tobytes())


def encrypt_image_array(
    image: np.ndarray,
    key: bytes,
    mode: str = "ECB",
    sbox: np.ndarray = AES_SBOX,
    iv: bytes | None = None,
    algorithm: str = "AES",
) -> tuple[bytes, dict[str, Any]]:
    image = np.ascontiguousarray(image, dtype=np.uint8)
    raw = image.tobytes()
    cipher, meta = encrypt_bytes(raw, key, mode, sbox, iv)
    meta.shape = tuple(image.shape)
    meta.algorithm = algorithm
    return cipher, meta.to_dict()


def decrypt_image_bytes(
    cipher: bytes,
    key: bytes,
    shape: tuple[int, ...],
    mode: str = "ECB",
    sbox: np.ndarray = AES_SBOX,
    iv: bytes | None = None,
) -> np.ndarray:
    raw = decrypt_bytes(cipher, key, mode, sbox, iv)
    expected = int(np.prod(shape))
    if len(raw) < expected:
        raise ValueError("Data hasil dekripsi lebih pendek dari ukuran citra")
    arr = np.frombuffer(raw[:expected], dtype=np.uint8)
    return arr.reshape(shape).copy()


def cipher_to_image(cipher: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """Tampilkan ciphertext sebagai citra (ambil byte pertama sesuai ukuran)."""
    expected = int(np.prod(shape))
    if len(cipher) < expected:
        raise ValueError("Ciphertext lebih pendek dari ukuran citra")
    arr = np.frombuffer(cipher[:expected], dtype=np.uint8)
    return arr.reshape(shape).copy()
