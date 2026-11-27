"""Implementasi AES-128 dengan S-box yang dapat diganti.

Bagian inti enkripsi/dekripsi blok di-JIT dengan Numba agar enkripsi citra
512x512 tetap responsif. Jika Numba tidak tersedia, modul tetap dapat
diimpor, tetapi fungsi akan gagal saat dipanggil; gunakan NumPy fallback
di modes.py bila perlu.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .constants import (
    AES_SBOX,
    INV_SHIFT_IDX,
    MUL2,
    MUL3,
    MUL9,
    MUL11,
    MUL13,
    MUL14,
    RCON,
    SHIFT_IDX,
)

# Numba lebih nyaman bekerja dengan int32 daripada uint8 untuk operasi XOR.
_SHIFT = SHIFT_IDX.astype(np.int64)
_INV_SHIFT = INV_SHIFT_IDX.astype(np.int64)
_M2 = MUL2.astype(np.int32)
_M3 = MUL3.astype(np.int32)
_M9 = MUL9.astype(np.int32)
_M11 = MUL11.astype(np.int32)
_M13 = MUL13.astype(np.int32)
_M14 = MUL14.astype(np.int32)


@njit(cache=True)
def _encrypt_block(block, round_keys, sbox, shift_idx, mul2, mul3):
    s = np.empty(16, dtype=np.int32)
    for i in range(16):
        s[i] = block[i] ^ round_keys[0, i]

    for rnd in range(1, 10):
        for i in range(16):
            s[i] = sbox[s[i]]

        tmp = s.copy()
        for i in range(16):
            s[i] = tmp[shift_idx[i]]

        for c in range(4):
            a0 = s[4 * c + 0]
            a1 = s[4 * c + 1]
            a2 = s[4 * c + 2]
            a3 = s[4 * c + 3]
            s[4 * c + 0] = mul2[a0] ^ mul3[a1] ^ a2 ^ a3
            s[4 * c + 1] = a0 ^ mul2[a1] ^ mul3[a2] ^ a3
            s[4 * c + 2] = a0 ^ a1 ^ mul2[a2] ^ mul3[a3]
            s[4 * c + 3] = mul3[a0] ^ a1 ^ a2 ^ mul2[a3]

        for i in range(16):
            s[i] = s[i] ^ round_keys[rnd, i]

    for i in range(16):
        s[i] = sbox[s[i]]

    tmp = s.copy()
    for i in range(16):
        s[i] = tmp[shift_idx[i]]

    out = np.empty(16, dtype=np.uint8)
    for i in range(16):
        out[i] = s[i] ^ round_keys[10, i]
    return out


@njit(cache=True)
def _decrypt_block(block, round_keys, inv_sbox, inv_shift_idx, mul9, mul11, mul13, mul14):
    s = np.empty(16, dtype=np.int32)
    for i in range(16):
        s[i] = block[i] ^ round_keys[10, i]

    for rnd in range(9, 0, -1):
        tmp = s.copy()
        for i in range(16):
            s[i] = tmp[inv_shift_idx[i]]

        for i in range(16):
            s[i] = inv_sbox[s[i]]

        for i in range(16):
            s[i] = s[i] ^ round_keys[rnd, i]

        for c in range(4):
            a0 = s[4 * c + 0]
            a1 = s[4 * c + 1]
            a2 = s[4 * c + 2]
            a3 = s[4 * c + 3]
            s[4 * c + 0] = mul14[a0] ^ mul11[a1] ^ mul13[a2] ^ mul9[a3]
            s[4 * c + 1] = mul9[a0] ^ mul14[a1] ^ mul11[a2] ^ mul13[a3]
            s[4 * c + 2] = mul13[a0] ^ mul9[a1] ^ mul14[a2] ^ mul11[a3]
            s[4 * c + 3] = mul11[a0] ^ mul13[a1] ^ mul9[a2] ^ mul14[a3]

    tmp = s.copy()
    for i in range(16):
        s[i] = tmp[inv_shift_idx[i]]

    for i in range(16):
        s[i] = inv_sbox[s[i]]

    out = np.empty(16, dtype=np.uint8)
    for i in range(16):
        out[i] = s[i] ^ round_keys[0, i]
    return out


@njit(cache=True)
def ecb_encrypt_blocks(blocks, round_keys, sbox, shift_idx, mul2, mul3):
    n = blocks.shape[0]
    out = np.empty((n, 16), dtype=np.uint8)
    for i in range(n):
        out[i, :] = _encrypt_block(blocks[i], round_keys, sbox, shift_idx, mul2, mul3)
    return out


@njit(cache=True)
def ecb_decrypt_blocks(blocks, round_keys, inv_sbox, inv_shift_idx, mul9, mul11, mul13, mul14):
    n = blocks.shape[0]
    out = np.empty((n, 16), dtype=np.uint8)
    for i in range(n):
        out[i, :] = _decrypt_block(
            blocks[i], round_keys, inv_sbox, inv_shift_idx, mul9, mul11, mul13, mul14
        )
    return out


@njit(cache=True)
def cbc_encrypt_blocks(blocks, round_keys, sbox, shift_idx, mul2, mul3, iv):
    n = blocks.shape[0]
    out = np.empty((n, 16), dtype=np.uint8)
    prev = np.empty(16, dtype=np.uint8)
    for j in range(16):
        prev[j] = iv[j]

    for i in range(n):
        x = np.empty(16, dtype=np.uint8)
        for j in range(16):
            x[j] = blocks[i, j] ^ prev[j]
        y = _encrypt_block(x, round_keys, sbox, shift_idx, mul2, mul3)
        out[i, :] = y
        prev = y
    return out


@njit(cache=True)
def cbc_decrypt_blocks(
    blocks, round_keys, inv_sbox, inv_shift_idx, mul9, mul11, mul13, mul14, iv
):
    n = blocks.shape[0]
    out = np.empty((n, 16), dtype=np.uint8)
    for i in range(n):
        y = _decrypt_block(
            blocks[i], round_keys, inv_sbox, inv_shift_idx, mul9, mul11, mul13, mul14
        )
        if i == 0:
            for j in range(16):
                out[i, j] = y[j] ^ iv[j]
        else:
            for j in range(16):
                out[i, j] = y[j] ^ blocks[i - 1, j]
    return out


def expand_key(key: bytes | bytearray | np.ndarray, sbox: np.ndarray) -> np.ndarray:
    """Ekspansi kunci AES-128 menjadi 11 round key berukuran 16 byte.

    SubWord memakai S-box yang diberikan, sehingga S-box GA juga berlaku
    pada key schedule.
    """
    if isinstance(key, np.ndarray):
        key_arr = np.asarray(key, dtype=np.uint8).reshape(-1)
    else:
        key_arr = np.frombuffer(bytes(key), dtype=np.uint8)
    if key_arr.size != 16:
        raise ValueError("AES-128 membutuhkan kunci 16 byte")

    sbox = np.asarray(sbox, dtype=np.uint8).reshape(256)
    w = np.zeros((44, 4), dtype=np.uint8)
    w[:4] = key_arr.reshape(4, 4)

    for i in range(4, 44):
        temp = w[i - 1].copy()
        if i % 4 == 0:
            temp = temp[[1, 2, 3, 0]]
            temp = sbox[temp]
            temp[0] = np.uint8(temp[0] ^ RCON[i // 4 - 1])
        w[i] = w[i - 4] ^ temp

    return np.ascontiguousarray(w.reshape(11, 16))


def encrypt_blocks(blocks: np.ndarray, round_keys: np.ndarray, sbox: np.ndarray) -> np.ndarray:
    return ecb_encrypt_blocks(
        np.ascontiguousarray(blocks, dtype=np.uint8),
        np.ascontiguousarray(round_keys, dtype=np.uint8),
        np.asarray(sbox, dtype=np.uint8).reshape(256),
        _SHIFT,
        _M2,
        _M3,
    )


def decrypt_blocks(
    blocks: np.ndarray, round_keys: np.ndarray, inv_sbox: np.ndarray
) -> np.ndarray:
    return ecb_decrypt_blocks(
        np.ascontiguousarray(blocks, dtype=np.uint8),
        np.ascontiguousarray(round_keys, dtype=np.uint8),
        np.asarray(inv_sbox, dtype=np.uint8).reshape(256),
        _INV_SHIFT,
        _M9,
        _M11,
        _M13,
        _M14,
    )


def warm_up() -> None:
    """Kompilasi fungsi Numba sekali di awal (dipakai Streamlit)."""
    key = bytes(range(16))
    rk = expand_key(key, AES_SBOX)
    blocks = np.zeros((1, 16), dtype=np.uint8)
    encrypt_blocks(blocks, rk, AES_SBOX)
    from .constants import inverse_sbox

    decrypt_blocks(blocks, rk, inverse_sbox(AES_SBOX))
