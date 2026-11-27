"""Utilitas citra: load, resize/crop, simpan, dan hash."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str | Path, size: tuple[int, int] | None = (512, 512)) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size is not None:
        # Resize dengan mempertahankan rasio, lalu center-crop.
        target_w, target_h = size
        w, h = img.size
        scale = max(target_w / w, target_h / h)
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        left = (img.width - target_w) // 2
        top = (img.height - target_h) // 2
        img = img.crop((left, top, left + target_w, top + target_h))
    return np.asarray(img, dtype=np.uint8)


def save_image(image: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 2:
        Image.fromarray(arr, mode="L").save(path)
    else:
        Image.fromarray(arr, mode="RGB").save(path)
    return path


def image_to_png_bytes(image: np.ndarray) -> bytes:
    arr = np.asarray(image, dtype=np.uint8)
    mode = "L" if arr.ndim == 2 else "RGB"
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    return buf.getvalue()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_image(image: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(image, dtype=np.uint8).tobytes())


def preprocess_dataset(raw_dir: str | Path = "data/raw", out_dir: str | Path = "data/processed") -> dict[str, Path]:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "lena": raw_dir / "lena.jpg",
        "mandrill": raw_dir / "mandrill.jpg",
        "remote": raw_dir / "remote_kalimantan.jpg",
    }
    outputs: dict[str, Path] = {}
    for name, src in mapping.items():
        if not src.exists():
            continue
        img = load_image(src, (512, 512))
        dst = out_dir / f"{name}.png"
        save_image(img, dst)
        outputs[name] = dst
    return outputs
