"""Targeted random-restart audit for the strongest reported emulator."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from asymptotic_markov_renewal_kl import cross_kl_rate, quantum_kernel
from optimize_phase_cross_kl import nelder_mead
from symmetric_emission_frontier import (
    symmetric_classical_model,
    symmetric_quantum_model,
)


AFFINITY = 60.0
PHASES = 15
DT = 0.01
DEAD_STEPS = 10
RANDOM_SEED = 20260906
PERTURBATION_SCALES = (0.05, 0.15, 0.30)
MAX_ITERATIONS = 300


def load_reference() -> tuple[np.ndarray, float]:
    rows = list(
        csv.DictReader(Path("symmetric_emission_frontier.csv").open(encoding="utf-8"))
    )
    row = next(
        item
        for item in rows
        if int(item["phases_per_macrostate"]) == PHASES
        and float(item["hidden_cycle_affinity"]) == AFFINITY
    )
    rates = np.array([float(value) for value in row["forward_rates"].split()])
    return np.log(rates), float(row["asymptotic_kl_rate"])


def main() -> None:
    reference, reference_value = load_reference()
    quantum_data = quantum_kernel(symmetric_quantum_model(4.0, DT), DEAD_STEPS)
    reverse_fraction = float(np.exp(-AFFINITY / (2.0 * PHASES)))
    lower, upper = np.log(1e-3), np.log(100.0)
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for restart, scale in enumerate(PERTURBATION_SCALES, start=1):
        cache: dict[tuple[float, ...], float] = {}

        def objective(log_forward: np.ndarray) -> float:
            clipped = np.clip(log_forward, lower, upper)
            key = tuple(np.round(clipped, 8))
            if key not in cache:
                model = symmetric_classical_model(
                    DT, PHASES, clipped, reverse_fraction
                )[0]
                cache[key] = cross_kl_rate(
                    quantum_data, model, DEAD_STEPS, DT
                )
            return cache[key]

        start = np.clip(
            reference + rng.normal(0.0, scale, size=reference.size), lower, upper
        )
        initial_value = objective(start)
        point, final_value, iterations = nelder_mead(
            objective, start, max_iterations=MAX_ITERATIONS
        )
        rows.append(
            {
                "restart": restart,
                "random_seed": RANDOM_SEED,
                "log_rate_perturbation_sd": scale,
                "initial_kl_rate": initial_value,
                "final_kl_rate": final_value,
                "difference_from_reported": final_value - reference_value,
                "iterations": iterations,
                "objective_evaluations": len(cache),
                "forward_rates": " ".join(
                    f"{value:.12g}" for value in np.exp(np.clip(point, lower, upper))
                ),
            }
        )
        print(
            f"restart={restart} scale={scale:.2f} "
            f"initial={initial_value:.12g} final={final_value:.12g} "
            f"delta={final_value-reference_value:+.3e} evaluations={len(cache)}",
            flush=True,
        )

    output = Path("optimization_restart_audit.csv")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"reported={reference_value:.12g}")
    print(f"restart range={min(row['final_kl_rate'] for row in rows):.12g}--"
          f"{max(row['final_kl_rate'] for row in rows):.12g}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
