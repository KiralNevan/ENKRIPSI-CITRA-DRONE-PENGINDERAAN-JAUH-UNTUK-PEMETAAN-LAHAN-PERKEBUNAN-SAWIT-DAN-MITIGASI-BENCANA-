"""Buat figur untuk laporan dan aplikasi Streamlit."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.sbox.io import load_sbox_csv
from src.sbox.metrics import sbox_to_grid

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
SBOX_DIR = RESULTS / "sbox"
METRICS = RESULTS / "metrics"


def _save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=160)
    plt.close(fig)


def ga_convergence() -> None:
    files = sorted(SBOX_DIR.glob("ga_history_seed_*.csv"))
    if not files:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for path in files:
        df = pd.read_csv(path)
        seed = path.stem.replace("ga_history_seed_", "")
        ax.plot(df["generation"], df["best_fitness"], label=f"seed {seed}", linewidth=1.5)
    ax.set_xlabel("Generasi")
    ax.set_ylabel("Fitness terbaik")
    ax.set_title("Konvergensi GA")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    _save(fig, "ga_convergence.png")


def sbox_heatmap() -> None:
    path = SBOX_DIR / "best_sbox.csv"
    if not path.exists():
        return
    grid = sbox_to_grid(load_sbox_csv(path))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid, cmap="viridis", interpolation="nearest")
    ax.set_title("S-box 16x16 hasil GA")
    ax.set_xlabel("Kolom")
    ax.set_ylabel("Baris")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Nilai")
    _save(fig, "sbox_heatmap.png")


def sbox_metrics_chart() -> None:
    path = METRICS / "sbox_metrics.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    metrics = ["nl", "bic_nl", "sac", "bic_sac", "lp", "dap"]
    titles = ["Nonlinearity", "BIC-NL", "SAC", "BIC-SAC", "LP", "DAP"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    algorithms = df["algorithm"].tolist()
    for ax, metric, title in zip(axes.ravel(), metrics, titles):
        values = df[metric].to_numpy(dtype=float)
        colors = ["tab:gray", "tab:blue", "tab:orange", "tab:green"][: len(values)]
        ax.bar(algorithms, values, color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(values):
            if np.isfinite(v):
                ax.text(i, v, f"{v:.4g}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Perbandingan metrik S-box", fontsize=14)
    _save(fig, "sbox_metrics.png")


def benchmark_charts() -> None:
    path = METRICS / "benchmark_results.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df["label"] = df["image"] + "\n" + df["algorithm"] + "\n" + df["mode"]
    for metric, title, fname in [
        ("entropy", "Entropy ciphertext", "benchmark_entropy.png"),
        ("npcr", "NPCR (%)", "benchmark_npcr.png"),
        ("uaci", "UACI (%)", "benchmark_uaci.png"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(df["label"], df[metric], color="tab:blue")
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", alpha=0.3)
        _save(fig, fname)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(df))
    width = 0.25
    ax.bar(x - width, df["corr_cipher_h"], width, label="H")
    ax.bar(x, df["corr_cipher_v"], width, label="V")
    ax.bar(x + width, df["corr_cipher_d"], width, label="D")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], fontsize=7)
    ax.set_title("Korelasi piksel ciphertext")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "benchmark_correlation.png")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(df["label"], df["time_enc_s"], color="tab:green")
    ax.set_title("Waktu enkripsi (detik)")
    ax.set_ylabel("detik")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "benchmark_time.png")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    ga_convergence()
    sbox_heatmap()
    sbox_metrics_chart()
    benchmark_charts()
    print(f"Figur disimpan di {FIGURES}")


if __name__ == "__main__":
    main()
