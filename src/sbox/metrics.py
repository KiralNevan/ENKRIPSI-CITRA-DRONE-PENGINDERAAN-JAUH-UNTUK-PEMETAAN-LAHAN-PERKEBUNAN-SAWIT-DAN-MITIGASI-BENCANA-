"""Metrik kriptografis S-box 8-bit.

Semua metrik dihitung eksak (bukan aproksimasi):
- Nonlinearity (NL) dan Linear Probability (LP) memakai 255 komponen
  output mask dan Fast Walsh-Hadamard Transform.
- BIC-NL memakai 28 mask berbobot 2.
- SAC / BIC-SAC memakai matriks dependensi.
- DAP memakai difference distribution table.
"""

from __future__ import annotations

import numpy as np

PARITY = np.array([bin(i).count("1") & 1 for i in range(256)], dtype=np.uint8)
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)
_X = np.arange(256, dtype=np.int64)
_BIT_POS = np.arange(8, dtype=np.int32)


def fwht_batch(a: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard Transform untuk batch baris."""
    a = np.asarray(a, dtype=np.int32).copy()
    m, n = a.shape
    h = 1
    while h < n:
        a = a.reshape(m, -1, 2, h)
        x = a[:, :, 0, :].copy()
        y = a[:, :, 1, :].copy()
        a[:, :, 0, :] = x + y
        a[:, :, 1, :] = x - y
        a = a.reshape(m, n)
        h *= 2
    return a


def is_bijective(sbox: np.ndarray) -> bool:
    sbox = np.asarray(sbox, dtype=np.uint8).reshape(256)
    return bool(np.array_equal(np.sort(sbox), np.arange(256, dtype=np.uint8)))


def is_balanced(sbox: np.ndarray) -> bool:
    sbox = np.asarray(sbox, dtype=np.uint8).reshape(256)
    bits = (sbox[:, None] >> _BIT_POS) & 1
    return bool(np.all(bits.sum(axis=0) == 128))


def evaluate_sbox(sbox: np.ndarray) -> dict[str, float | int | bool]:
    sbox = np.asarray(sbox, dtype=np.uint8).reshape(256)
    if not is_bijective(sbox):
        raise ValueError("S-box harus bijective (permutasi 0..255)")

    # ---- Walsh spectrum untuk semua 255 output mask ----
    masks = np.arange(1, 256, dtype=np.int32)
    components = PARITY[(masks[:, None] & sbox[None, :])]
    signs = 1 - 2 * components.astype(np.int32)
    walsh = fwht_batch(signs)
    max_abs = np.max(np.abs(walsh), axis=1)
    nl_per_mask = 128 - (max_abs // 2)

    nl = int(nl_per_mask.min())
    bic_nl = int(nl_per_mask[POPCOUNT[masks] == 2].min())
    lp = float(np.max((max_abs.astype(np.float64) / 256.0) ** 2))

    # ---- SAC ----
    sac_matrix = np.zeros((8, 8), dtype=np.float64)
    for i in range(8):
        flipped = sbox[_X ^ (1 << i)]
        diff = sbox ^ flipped
        diff_bits = (diff[:, None].astype(np.int32) >> _BIT_POS) & 1
        sac_matrix[i] = diff_bits.mean(axis=0)

    # ---- BIC-SAC ----
    bic_sac_values = []
    for j in range(8):
        for k in range(j + 1, 8):
            f = ((sbox >> j) ^ (sbox >> k)) & 1
            vals = []
            for i in range(8):
                diff = f ^ f[_X ^ (1 << i)]
                vals.append(float(diff.mean()))
            bic_sac_values.append(float(np.mean(vals)))
    bic_sac_values_arr = np.asarray(bic_sac_values, dtype=np.float64)

    # ---- DAP ----
    dap_max = 0
    for a in range(1, 256):
        diff = sbox ^ sbox[_X ^ a]
        counts = np.bincount(diff, minlength=256)
        dap_max = max(dap_max, int(counts.max()))
    dap = dap_max / 256.0

    fixed_points = int(np.sum(sbox == np.arange(256, dtype=np.uint8)))
    opposite_fixed_points = int(np.sum(sbox == (np.arange(256, dtype=np.uint8) ^ 0xFF)))

    return {
        "nl": nl,
        "sac": float(sac_matrix.mean()),
        "sac_max_dev": float(np.max(np.abs(sac_matrix - 0.5))),
        "bic_nl": bic_nl,
        "bic_sac": float(bic_sac_values_arr.mean()),
        "bic_sac_max_dev": float(np.max(np.abs(bic_sac_values_arr - 0.5))),
        "lp": lp,
        "dap": dap,
        "balanced": is_balanced(sbox),
        "bijective": True,
        "fixed_points": fixed_points,
        "opposite_fixed_points": opposite_fixed_points,
    }


def sbox_to_grid(sbox: np.ndarray) -> np.ndarray:
    return np.asarray(sbox, dtype=np.uint8).reshape(16, 16)
