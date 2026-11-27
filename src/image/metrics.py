"""Metrik kualitas enkripsi citra: entropy, NPCR, UACI, korelasi, MSE, hash."""

from __future__ import annotations

import numpy as np

from .io import sha256_image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(np.uint8)
    if arr.shape[2] >= 3:
        return (
            0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        ).astype(np.uint8)
    return arr[:, :, 0].astype(np.uint8)


def entropy(image: np.ndarray) -> float:
    arr = np.asarray(image, dtype=np.uint8).reshape(-1)
    counts = np.bincount(arr, minlength=256).astype(np.float64)
    p = counts[counts > 0] / arr.size
    return float(-np.sum(p * np.log2(p)))


def entropy_channels(image: np.ndarray) -> list[float]:
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 2:
        return [entropy(arr)]
    return [entropy(arr[:, :, c]) for c in range(arr.shape[2])]


def _as_uint8_flat(x) -> np.ndarray:
    if isinstance(x, (bytes, bytearray, memoryview)):
        return np.frombuffer(x, dtype=np.uint8).reshape(-1)
    return np.asarray(x, dtype=np.uint8).reshape(-1)


def npcr_uaci(c1, c2) -> tuple[float, float]:
    a = _as_uint8_flat(c1)
    b = _as_uint8_flat(c2)
    n = min(a.size, b.size)
    a = a[:n]
    b = b[:n]
    diff = a != b
    npcr = float(diff.mean() * 100.0)
    uaci = float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))) / 255.0 * 100.0)
    return npcr, uaci


def correlation_coefficients(
    image: np.ndarray, sample: int = 5000, seed: int = 42
) -> dict[str, float]:
    gray = to_grayscale(image)
    h, w = gray.shape
    rng = np.random.default_rng(seed)

    def corr(pairs: np.ndarray) -> float:
        if pairs.size == 0:
            return 0.0
        a = pairs[:, 0].astype(np.float64)
        b = pairs[:, 1].astype(np.float64)
        if a.std() == 0 or b.std() == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    # Horizontal
    n = min(sample, h * max(w - 1, 0))
    if n > 0:
        ys = rng.integers(0, h, size=n)
        xs = rng.integers(0, w - 1, size=n)
        horiz = np.column_stack([gray[ys, xs], gray[ys, xs + 1]])
    else:
        horiz = np.empty((0, 2), dtype=np.uint8)

    # Vertical
    n = min(sample, max(h - 1, 0) * w)
    if n > 0:
        ys = rng.integers(0, h - 1, size=n)
        xs = rng.integers(0, w, size=n)
        vert = np.column_stack([gray[ys, xs], gray[ys + 1, xs]])
    else:
        vert = np.empty((0, 2), dtype=np.uint8)

    # Diagonal
    n = min(sample, max(h - 1, 0) * max(w - 1, 0))
    if n > 0:
        ys = rng.integers(0, h - 1, size=n)
        xs = rng.integers(0, w - 1, size=n)
        diag = np.column_stack([gray[ys, xs], gray[ys + 1, xs + 1]])
    else:
        diag = np.empty((0, 2), dtype=np.uint8)

    return {"h": corr(horiz), "v": corr(vert), "d": corr(diag)}


def mse(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    return float(np.mean((x - y) ** 2))


def image_metrics(plain: np.ndarray, cipher: np.ndarray, decrypted: np.ndarray) -> dict:
    plain = np.asarray(plain, dtype=np.uint8)
    cipher = np.asarray(cipher, dtype=np.uint8)
    decrypted = np.asarray(decrypted, dtype=np.uint8)
    corr_plain = correlation_coefficients(plain)
    corr_cipher = correlation_coefficients(cipher)
    return {
        "entropy_plain": entropy(plain),
        "entropy_cipher": entropy(cipher),
        "entropy_plain_channels": entropy_channels(plain),
        "entropy_cipher_channels": entropy_channels(cipher),
        "corr_plain_h": corr_plain["h"],
        "corr_plain_v": corr_plain["v"],
        "corr_plain_d": corr_plain["d"],
        "corr_cipher_h": corr_cipher["h"],
        "corr_cipher_v": corr_cipher["v"],
        "corr_cipher_d": corr_cipher["d"],
        "mse": mse(plain, decrypted),
        "hash_plain": sha256_image(plain),
        "hash_decrypted": sha256_image(decrypted),
        "hash_match": bool(np.array_equal(plain, decrypted)),
    }
