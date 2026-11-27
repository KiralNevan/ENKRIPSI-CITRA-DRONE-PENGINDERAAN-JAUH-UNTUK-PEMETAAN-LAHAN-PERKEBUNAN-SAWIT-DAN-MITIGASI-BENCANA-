"""Algoritma genetika untuk konstruksi S-box 16x16 (permutasi 0..255).

Kromosom berupa permutasi sehingga bijective dan balanced terjamin.
Fitness menggabungkan enam kriteria S-box dengan bobot ternormalisasi.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from .metrics import evaluate_sbox


@dataclass
class GAConfig:
    population_size: int = 80
    generations: int = 500
    tournament_size: int = 3
    crossover_rate: float = 0.85
    mutation_rate: float = 0.08
    elitism: int = 2
    stagnation: int = 50
    seed: int = 2026
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "nl": 0.30,
            "sac": 0.20,
            "bic_nl": 0.20,
            "bic_sac": 0.15,
            "lp": 0.075,
            "dap": 0.075,
        }
    )
    nl_ref: float = 112.0
    lp_ref: float = 0.0625
    dap_ref: float = 0.0625

    def __post_init__(self) -> None:
        if abs(sum(self.weights.values()) - 1.0) > 1e-9:
            raise ValueError("Total bobot fitness harus 1.0")
        if self.elitism >= self.population_size:
            raise ValueError("Elitism harus lebih kecil dari ukuran populasi")


def fitness_from_metrics(
    m: dict,
    weights: dict[str, float],
    nl_ref: float = 112.0,
    lp_ref: float = 0.0625,
    dap_ref: float = 0.0625,
) -> float:
    s_nl = min(float(m["nl"]) / nl_ref, 1.0)
    s_sac = max(0.0, 1.0 - 2.0 * abs(float(m["sac"]) - 0.5))
    s_bic_nl = min(float(m["bic_nl"]) / nl_ref, 1.0)
    s_bic_sac = max(0.0, 1.0 - 2.0 * abs(float(m["bic_sac"]) - 0.5))
    s_lp = max(0.0, 1.0 - float(m["lp"]) / lp_ref)
    s_dap = max(0.0, 1.0 - float(m["dap"]) / dap_ref)
    return float(
        weights["nl"] * s_nl
        + weights["sac"] * s_sac
        + weights["bic_nl"] * s_bic_nl
        + weights["bic_sac"] * s_bic_sac
        + weights["lp"] * s_lp
        + weights["dap"] * s_dap
    )


def random_permutation(rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(256).astype(np.uint8)


def ox_crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Order crossover (OX) untuk permutasi."""
    n = len(p1)
    a, b = sorted(rng.choice(n, size=2, replace=False).tolist())
    child = np.full(n, -1, dtype=np.int16)
    used = np.zeros(n, dtype=bool)

    child[a : b + 1] = p1[a : b + 1]
    used[p1[a : b + 1]] = True

    pos = (b + 1) % n
    for gene in np.concatenate([p2[b + 1 :], p2[: b + 1]]):
        gene = int(gene)
        if not used[gene]:
            child[pos] = gene
            used[gene] = True
            pos = (pos + 1) % n
    return child.astype(np.uint8)


def mutate(sbox: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    child = sbox.copy()
    n_swaps = int(rng.integers(1, 4))
    for _ in range(n_swaps):
        i, j = rng.choice(256, size=2, replace=False)
        child[i], child[j] = child[j], child[i]
    return child


def tournament_select(
    population: np.ndarray, fitness: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    idx = rng.integers(0, len(population), size=k)
    best = idx[np.argmax(fitness[idx])]
    return population[best].copy()


def genetic_algorithm(
    config: GAConfig | None = None,
    verbose: bool = True,
    callback: Callable[[int, dict], None] | None = None,
) -> dict:
    cfg = config or GAConfig()
    rng = np.random.default_rng(cfg.seed)

    population = np.array(
        [random_permutation(rng) for _ in range(cfg.population_size)], dtype=np.uint8
    )
    cache: dict[bytes, tuple[float, dict]] = {}

    def evaluate(ind: np.ndarray) -> tuple[float, dict]:
        key = ind.tobytes()
        if key not in cache:
            metrics = evaluate_sbox(ind)
            cache[key] = (
                fitness_from_metrics(
                    metrics,
                    cfg.weights,
                    nl_ref=cfg.nl_ref,
                    lp_ref=cfg.lp_ref,
                    dap_ref=cfg.dap_ref,
                ),
                metrics,
            )
        return cache[key]

    start = time.perf_counter()
    fitness = np.array([evaluate(ind)[0] for ind in population], dtype=np.float64)
    history: list[dict] = []
    best_ever = float(fitness.max())
    best_ever_sbox = population[int(np.argmax(fitness))].copy()
    best_ever_metrics = evaluate(best_ever_sbox)[1]
    stagnant = 0
    generations_run = 0

    for gen in range(1, cfg.generations + 1):
        order = np.argsort(fitness)[::-1]
        best_idx = int(order[0])
        best_f = float(fitness[best_idx])
        best_m = evaluate(population[best_idx])[1]

        if best_f > best_ever + 1e-12:
            best_ever = best_f
            best_ever_sbox = population[best_idx].copy()
            best_ever_metrics = best_m
            stagnant = 0
        else:
            stagnant += 1

        history.append(
            {
                "generation": gen,
                "best_fitness": best_f,
                "avg_fitness": float(fitness.mean()),
                "best_nl": int(best_m["nl"]),
                "best_sac": float(best_m["sac"]),
                "best_bic_nl": int(best_m["bic_nl"]),
                "best_bic_sac": float(best_m["bic_sac"]),
                "best_lp": float(best_m["lp"]),
                "best_dap": float(best_m["dap"]),
            }
        )
        generations_run = gen

        if verbose and (gen == 1 or gen % 10 == 0 or gen == cfg.generations):
            print(
                f"[seed {cfg.seed}] gen {gen:4d}/{cfg.generations} "
                f"best={best_f:.5f} avg={fitness.mean():.5f} "
                f"NL={best_m['nl']} SAC={best_m['sac']:.4f} "
                f"BIC-NL={best_m['bic_nl']} LP={best_m['lp']:.5f} DAP={best_m['dap']:.5f}"
            )
        if callback is not None:
            callback(gen, history[-1])

        if stagnant >= cfg.stagnation:
            if verbose:
                print(f"[seed {cfg.seed}] berhenti: {stagnant} generasi tanpa perbaikan")
            break

        new_population: list[np.ndarray] = [
            population[int(order[i])].copy() for i in range(cfg.elitism)
        ]
        while len(new_population) < cfg.population_size:
            p1 = tournament_select(population, fitness, cfg.tournament_size, rng)
            p2 = tournament_select(population, fitness, cfg.tournament_size, rng)
            if rng.random() < cfg.crossover_rate:
                child = ox_crossover(p1, p2, rng)
            else:
                child = p1.copy()
            if rng.random() < cfg.mutation_rate:
                child = mutate(child, rng)
            new_population.append(child)

        population = np.array(new_population, dtype=np.uint8)
        fitness = np.array([evaluate(ind)[0] for ind in population], dtype=np.float64)

    elapsed = time.perf_counter() - start
    return {
        "sbox": best_ever_sbox,
        "metrics": best_ever_metrics,
        "fitness": best_ever,
        "history": history,
        "config": asdict(cfg),
        "generations_run": generations_run,
        "elapsed_seconds": elapsed,
        "cache_size": len(cache),
    }
