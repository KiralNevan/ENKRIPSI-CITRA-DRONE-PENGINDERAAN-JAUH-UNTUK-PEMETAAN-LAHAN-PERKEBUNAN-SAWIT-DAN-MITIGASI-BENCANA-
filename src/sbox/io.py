"""Penyimpanan dan pemuatan S-box serta hasil GA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def save_sbox_csv(sbox: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = np.asarray(sbox, dtype=np.uint8).reshape(16, 16)
    pd.DataFrame(grid).to_csv(path, index=False, header=False)
    return path


def load_sbox_csv(path: str | Path) -> np.ndarray:
    path = Path(path)
    grid = pd.read_csv(path, header=None).to_numpy(dtype=np.uint8)
    if grid.shape != (16, 16):
        raise ValueError(f"CSV S-box harus 16x16, ditemukan {grid.shape}")
    sbox = grid.reshape(256)
    if len(np.unique(sbox)) != 256:
        raise ValueError("S-box pada CSV tidak bijective")
    return sbox


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=default)
    return path


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
