"""Jalankan GA untuk beberapa seed, simpan S-box terbaik dan ringkasan metrik."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.aes.constants import AES_SBOX
from src.sbox.ga import GAConfig, genetic_algorithm
from src.sbox.io import save_json, save_sbox_csv
from src.sbox.metrics import evaluate_sbox


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="2026,2027,2028,2029,2030")
    parser.add_argument("--population", type=int, default=80)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--stagnation", type=int, default=100)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out = Path(args.out)
    sbox_dir = out / "sbox"
    metrics_dir = out / "metrics"
    sbox_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    runs = []
    for seed in seeds:
        print(f"\n=== GA seed {seed} ===")
        cfg = GAConfig(
            population_size=args.population,
            generations=args.generations,
            stagnation=args.stagnation,
            seed=seed,
        )
        result = genetic_algorithm(cfg, verbose=True)
        run = {
            "seed": seed,
            "sbox": result["sbox"],
            "metrics": result["metrics"],
            "fitness": result["fitness"],
            "history": result["history"],
            "config": result["config"],
            "generations_run": result["generations_run"],
            "elapsed_seconds": result["elapsed_seconds"],
            "cache_size": result["cache_size"],
        }
        runs.append(run)
        save_json(run, sbox_dir / f"ga_seed_{seed}.json")
        save_sbox_csv(result["sbox"], sbox_dir / f"ga_seed_{seed}.csv")
        pd.DataFrame(result["history"]).to_csv(
            sbox_dir / f"ga_history_seed_{seed}.csv", index=False
        )

    best = max(runs, key=lambda r: r["fitness"])
    save_sbox_csv(best["sbox"], sbox_dir / "best_sbox.csv")
    save_json(
        {
            "seed": best["seed"],
            "fitness": best["fitness"],
            "metrics": best["metrics"],
            "config": best["config"],
            "generations_run": best["generations_run"],
            "elapsed_seconds": best["elapsed_seconds"],
            "sbox": best["sbox"],
        },
        sbox_dir / "best_sbox.json",
    )

    summary_rows = []
    for run in runs:
        m = run["metrics"]
        summary_rows.append(
            {
                "seed": run["seed"],
                "fitness": run["fitness"],
                "nl": m["nl"],
                "sac": m["sac"],
                "bic_nl": m["bic_nl"],
                "bic_sac": m["bic_sac"],
                "lp": m["lp"],
                "dap": m["dap"],
                "balanced": m["balanced"],
                "bijective": m["bijective"],
                "fixed_points": m["fixed_points"],
                "generations_run": run["generations_run"],
                "elapsed_seconds": run["elapsed_seconds"],
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(sbox_dir / "ga_summary.csv", index=False)

    aes_m = evaluate_sbox(AES_SBOX)
    # Baris ringkas GA yang mudah dipakai di laporan/app
    metric_rows = [
        {
            "algorithm": "AES standar",
            "nl": aes_m["nl"],
            "sac": aes_m["sac"],
            "bic_nl": aes_m["bic_nl"],
            "bic_sac": aes_m["bic_sac"],
            "lp": aes_m["lp"],
            "dap": aes_m["dap"],
            "sac_max_dev": aes_m["sac_max_dev"],
            "bic_sac_max_dev": aes_m["bic_sac_max_dev"],
            "fixed_points": aes_m["fixed_points"],
        },
        {
            "algorithm": "GA terbaik",
            "nl": best["metrics"]["nl"],
            "sac": best["metrics"]["sac"],
            "bic_nl": best["metrics"]["bic_nl"],
            "bic_sac": best["metrics"]["bic_sac"],
            "lp": best["metrics"]["lp"],
            "dap": best["metrics"]["dap"],
            "sac_max_dev": best["metrics"]["sac_max_dev"],
            "bic_sac_max_dev": best["metrics"]["bic_sac_max_dev"],
            "fixed_points": best["metrics"]["fixed_points"],
        },
        {
            "algorithm": "GA rata-rata",
            "nl": float(summary["nl"].mean()),
            "sac": float(summary["sac"].mean()),
            "bic_nl": float(summary["bic_nl"].mean()),
            "bic_sac": float(summary["bic_sac"].mean()),
            "lp": float(summary["lp"].mean()),
            "dap": float(summary["dap"].mean()),
            "sac_max_dev": float("nan"),
            "bic_sac_max_dev": float("nan"),
            "fixed_points": float("nan"),
        },
        {
            "algorithm": "GA simpangan baku",
            "nl": float(summary["nl"].std(ddof=1)),
            "sac": float(summary["sac"].std(ddof=1)),
            "bic_nl": float(summary["bic_nl"].std(ddof=1)),
            "bic_sac": float(summary["bic_sac"].std(ddof=1)),
            "lp": float(summary["lp"].std(ddof=1)),
            "dap": float(summary["dap"].std(ddof=1)),
            "sac_max_dev": float("nan"),
            "bic_sac_max_dev": float("nan"),
            "fixed_points": float("nan"),
        },
    ]
    pd.DataFrame(metric_rows).to_csv(metrics_dir / "sbox_metrics.csv", index=False)

    print("\n=== Ringkasan GA ===")
    print(summary.to_string(index=False))
    print(f"\nS-box terbaik: seed {best['seed']}, fitness {best['fitness']:.6f}")
    print(json.dumps(best["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
