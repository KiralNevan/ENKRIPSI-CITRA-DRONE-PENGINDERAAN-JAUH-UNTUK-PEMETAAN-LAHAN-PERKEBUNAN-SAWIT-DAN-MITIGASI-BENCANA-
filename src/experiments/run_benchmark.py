"""Eksperimen benchmark: Lena & Mandrill, AES standar vs S-box GA, ECB/CBC."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.aes.constants import AES_SBOX
from src.aes.modes import cipher_to_image, decrypt_image_bytes, encrypt_image_array
from src.image.io import load_image, preprocess_dataset, save_image
from src.image.metrics import image_metrics, npcr_uaci
from src.sbox.io import load_sbox_csv

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
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


def _encrypt(image: np.ndarray, key: bytes, mode: str, sbox: np.ndarray):
    return encrypt_image_array(image, key, mode=mode, sbox=sbox, iv=IV)


def _decrypt(cipher: bytes, shape, key: bytes, mode: str, sbox: np.ndarray):
    return decrypt_image_bytes(cipher, key, shape, mode=mode, sbox=sbox, iv=IV)


def _time_runs(fn, runs: int = 5):
    # Warm-up
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
        axes[0, c].hist(plain[:, :, c].ravel(), bins=256, range=(0, 256), color="tab:blue", alpha=0.8)
        axes[0, c].set_title(f"Plaintext kanal {c}")
        axes[1, c].hist(cipher[:, :, c].ravel(), bins=256, range=(0, 256), color="tab:red", alpha=0.8)
        axes[1, c].set_title(f"Ciphertext kanal {c}")
        axes[0, c].set_xlim(0, 255)
        axes[1, c].set_xlim(0, 255)
    fig.tight_layout()
    fig.savefig(FIGURES / f"hist_{name}.png", dpi=150)
    plt.close(fig)


def _save_corr_scatter(plain: np.ndarray, cipher: np.ndarray, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, img, title in zip(axes, [plain, cipher], ["Plaintext", "Ciphertext"]):
        g = img[:, :, 0].astype(np.int32)
        n = 3000
        rng = np.random.default_rng(42)
        ys = rng.integers(0, g.shape[0], n)
        xs = rng.integers(0, g.shape[1] - 1, n)
        ax.scatter(g[ys, xs], g[ys, xs + 1], s=2, alpha=0.35)
        ax.set_title(title)
        ax.set_xlabel("pixel(x)")
        ax.set_ylabel("pixel(x+1)")
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 255)
    fig.tight_layout()
    fig.savefig(FIGURES / f"corr_scatter_{name}.png", dpi=150)
    plt.close(fig)


def run() -> pd.DataFrame:
    METRICS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    processed = {
        "lena": DATA / "processed" / "lena.png",
        "mandrill": DATA / "processed" / "mandrill.png",
    }
    if not all(p.exists() for p in processed.values()):
        preprocess_dataset(DATA / "raw", DATA / "processed")

    boxes = _load_sboxes()
    # Warm-up Numba untuk semua kombinasi S-box/mode sebelum pengukuran waktu.
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    for _sbox in boxes.values():
        for _mode in ("ECB", "CBC"):
            encrypt_image_array(dummy, KEY, _mode, _sbox, IV)
            decrypt_image_bytes(
                encrypt_image_array(dummy, KEY, _mode, _sbox, IV)[0],
                KEY,
                dummy.shape,
                mode=_mode,
                sbox=_sbox,
                iv=IV,
            )
    rows = []

    for image_name, path in processed.items():
        if not path.exists():
            print(f"Lewati {image_name}: {path} tidak ada")
            continue
        plain = load_image(path, None)
        print(f"\n=== {image_name} {plain.shape} ===")
        shape = plain.shape

        for algo_name, sbox in boxes.items():
            for mode in ("ECB", "CBC"):
                label = f"{image_name}_{algo_name}_{mode}".replace(" ", "_")
                print(f"  {algo_name} {mode}...", end="", flush=True)

                cipher, meta = _encrypt(plain, KEY, mode, sbox)
                decrypted = _decrypt(cipher, shape, KEY, mode, sbox)
                cipher_img = cipher_to_image(cipher, shape)

                def enc_once():
                    return _encrypt(plain, KEY, mode, sbox)

                def dec_once():
                    c, _ = enc_once()
                    return _decrypt(c, shape, KEY, mode, sbox)

                t_enc, t_enc_std = _time_runs(enc_once)
                t_dec, t_dec_std = _time_runs(dec_once)

                m = image_metrics(plain, cipher_img, decrypted)

                # Sensitivitas plaintext: ubah satu piksel
                plain2 = plain.copy()
                cy, cx = 0, 0
                if plain2.ndim == 3:
                    plain2[cy, cx, 0] ^= 1
                else:
                    plain2[cy, cx] ^= 1
                cipher2, _ = _encrypt(plain2, KEY, mode, sbox)
                npcr_p, uaci_p = npcr_uaci(cipher, cipher2)

                # Sensitivitas kunci: balik satu bit
                key2 = bytes([KEY[0] ^ 0x01]) + KEY[1:]
                cipher_key, _ = _encrypt(plain, key2, mode, sbox)
                npcr_k, uaci_k = npcr_uaci(cipher, cipher_key)

                row = {
                    "image": image_name,
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
                    "ciphertext_sha256": __import__("hashlib").sha256(cipher).hexdigest(),
                }
                rows.append(row)

                save_image(cipher_img, FIGURES / f"cipher_{label}.png")
                save_image(decrypted, FIGURES / f"decrypted_{label}.png")
                _save_histograms(plain, cipher_img, label)
                if mode == "CBC" and algo_name in ("AES standar", "S-box GA"):
                    _save_corr_scatter(plain, cipher_img, label)

                print(
                    f" H={row['entropy']:.4f} NPCR={row['npcr']:.2f}% "
                    f"UACI={row['uaci']:.2f}% enc={t_enc:.3f}s"
                )

    df = pd.DataFrame(rows)
    df.to_csv(METRICS / "benchmark_results.csv", index=False)

    # Metadata eksperimen
    meta = {
        "key_hex": KEY.hex(),
        "iv_hex": IV.hex(),
        "images": list(processed.keys()),
        "algorithms": list(boxes.keys()),
        "modes": ["ECB", "CBC"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (METRICS / "benchmark_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nDisimpan: {METRICS / 'benchmark_results.csv'}")
    return df


if __name__ == "__main__":
    run()
