"""Eksperimen citra penginderaan jauh (NASA Landsat Kalimantan)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.aes.modes import cipher_to_image, decrypt_image_bytes, encrypt_image_array
from src.image.io import load_image, save_image
from src.image.metrics import image_metrics, npcr_uaci
from src.sbox.io import load_sbox_csv
from src.aes.constants import AES_SBOX

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
METRICS = RESULTS / "metrics"
KEY = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
IV = bytes.fromhex("101112131415161718191a1b1c1d1e1f")


def _load_sboxes() -> dict[str, np.ndarray]:
    boxes = {"AES standar": AES_SBOX}
    ga_path = RESULTS / "sbox" / "best_sbox.csv"
    if ga_path.exists():
        boxes["S-box GA"] = load_sbox_csv(ga_path)
    return boxes


def _time_runs(fn, runs: int = 5):
    fn()
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return float(np.mean(times)), float(np.std(times, ddof=1)) if runs > 1 else 0.0


def _save_histograms(plain: np.ndarray, cipher: np.ndarray, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for c in range(3):
        axes[0, c].hist(plain[:, :, c].ravel(), bins=256, range=(0, 256), color="tab:green", alpha=0.8)
        axes[0, c].set_title(f"Plaintext kanal {c}")
        axes[1, c].hist(cipher[:, :, c].ravel(), bins=256, range=(0, 256), color="tab:red", alpha=0.8)
        axes[1, c].set_title(f"Ciphertext kanal {c}")
    fig.tight_layout()
    fig.savefig(FIGURES / f"hist_{name}.png", dpi=150)
    plt.close(fig)


def run(size: int = 512) -> pd.DataFrame:
    METRICS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    src = ROOT / "data" / "raw" / "remote_kalimantan.jpg"
    if not src.exists():
        raise FileNotFoundError(f"Citra remote sensing tidak ditemukan: {src}")
    plain = load_image(src, (size, size))
    save_image(plain, ROOT / "data" / "processed" / f"remote_{size}.png")
    shape = plain.shape
    boxes = _load_sboxes()
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    for _sbox in boxes.values():
        for _mode in ("ECB", "CBC"):
            _c, _ = encrypt_image_array(dummy, KEY, _mode, _sbox, IV)
            decrypt_image_bytes(_c, KEY, dummy.shape, mode=_mode, sbox=_sbox, iv=IV)
    rows = []

    print(f"=== remote {shape} ===")
    for algo_name, sbox in boxes.items():
        for mode in ("ECB", "CBC"):
            label = f"remote_{size}_{algo_name}_{mode}".replace(" ", "_")
            print(f"  {algo_name} {mode}...", end="", flush=True)
            cipher, _ = encrypt_image_array(plain, KEY, mode=mode, sbox=sbox, iv=IV)
            decrypted = decrypt_image_bytes(cipher, KEY, shape, mode=mode, sbox=sbox, iv=IV)
            cipher_img = cipher_to_image(cipher, shape)

            def enc_once():
                return encrypt_image_array(plain, KEY, mode=mode, sbox=sbox, iv=IV)

            def dec_once():
                c, _ = enc_once()
                return decrypt_image_bytes(c, KEY, shape, mode=mode, sbox=sbox, iv=IV)

            t_enc, t_enc_std = _time_runs(enc_once)
            t_dec, t_dec_std = _time_runs(dec_once)
            m = image_metrics(plain, cipher_img, decrypted)

            plain2 = plain.copy()
            cy, cx = 0, 0
            plain2[cy, cx, 0] ^= 1
            cipher2, _ = encrypt_image_array(plain2, KEY, mode=mode, sbox=sbox, iv=IV)
            npcr_p, uaci_p = npcr_uaci(cipher, cipher2)

            key2 = bytes([KEY[0] ^ 0x01]) + KEY[1:]
            cipher_key, _ = encrypt_image_array(plain, key2, mode=mode, sbox=sbox, iv=IV)
            npcr_k, uaci_k = npcr_uaci(cipher, cipher_key)

            row = {
                "image": "remote_kalimantan",
                "size": f"{shape[0]}x{shape[1]}",
                "algorithm": algo_name,
                "mode": mode,
                "entropy": m["entropy_cipher"],
                "entropy_plain": m["entropy_plain"],
                "npcr": npcr_p,
                "uaci": uaci_p,
                "npcr_plain": npcr_p,
                "uaci_plain": uaci_p,
                "npcr_key": npcr_k,
                "uaci_key": uaci_k,
                "corr_plain_h": m["corr_plain_h"],
                "corr_plain_v": m["corr_plain_v"],
                "corr_plain_d": m["corr_plain_d"],
                "corr_cipher_h": m["corr_cipher_h"],
                "corr_cipher_v": m["corr_cipher_v"],
                "corr_cipher_d": m["corr_cipher_d"],
                "time_enc_s": t_enc,
                "time_enc_std": t_enc_std,
                "time_dec_s": t_dec,
                "time_dec_std": t_dec_std,
                "mse": m["mse"],
                "hash_match": m["hash_match"],
            }
            rows.append(row)
            save_image(cipher_img, FIGURES / f"cipher_{label}.png")
            save_image(decrypted, FIGURES / f"decrypted_{label}.png")
            _save_histograms(plain, cipher_img, label)
            print(f" H={row['entropy']:.4f} NPCR={row['npcr']:.2f}% UACI={row['uaci']:.2f}%")

    df = pd.DataFrame(rows)
    df.to_csv(METRICS / f"remote_results_{size}.csv", index=False)
    (METRICS / f"remote_meta_{size}.json").write_text(
        json.dumps({"size": size, "key_hex": KEY.hex(), "iv_hex": IV.hex()}, indent=2),
        encoding="utf-8",
    )
    print(f"Disimpan: {METRICS / f'remote_results_{size}.csv'}")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    run(args.size)
