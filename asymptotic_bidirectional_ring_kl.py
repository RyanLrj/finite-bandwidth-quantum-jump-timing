"""Asymptotic KL for a classical hidden ring with backward transitions.

This is a stronger classical alternative than the directed phase clock: every
forward edge keeps its own rate and all reverse edges share a fitted rate.
The fitted reverse rate diagnoses whether recurrence/detailed-balance-like
motion helps reproduce the quantum marked waiting-time kernel.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from asymptotic_markov_renewal_kl import cross_kl_rate, quantum_kernel
from classical_memory_cross_kl import classical_split_model, quantum_split_model
from optimize_phase_cross_kl import nelder_mead


def bidirectional_ring_model(dt: float, phases: int, log_rates: np.ndarray):
    rates = np.exp(log_rates)
    if len(rates) != 2 * phases + 1:
        raise ValueError("expected one forward rate per edge and one shared reverse rate")
    forward = rates[:-1]
    reverse = float(rates[-1])
    size = 1 + 2 * phases

    def event(target: int, source: int, rate: float) -> np.ndarray:
        result = np.zeros((size, size), dtype=float)
        result[target, source] = rate
        return result

    cold_states = tuple(range(1, 1 + phases))
    hot_states = tuple(range(1 + phases, 1 + 2 * phases))
    ring = cold_states + hot_states
    cold_emission = sum(
        (event(0, state, 1.0) for state in cold_states), np.zeros((size, size))
    )
    hot_emission = sum(
        (event(0, state, 0.8) for state in hot_states), np.zeros((size, size))
    )
    thermal = (
        cold_emission,
        event(cold_states[0], 0, np.exp(-2.0)),
        hot_emission,
        event(hot_states[0], 0, 0.8 * np.exp(-1.25)),
    )

    work = np.zeros((size, size), dtype=float)
    for index, source in enumerate(ring):
        target = ring[(index + 1) % len(ring)]
        work += event(target, source, float(forward[index]))
        work += event(source, target, reverse)
    work -= np.diag(work.sum(axis=0))
    return classical_split_model(dt, 0, work, thermal)[0]


def directional_to_ring_order(rates: np.ndarray, phases: int) -> np.ndarray:
    """Convert [cold internal, hot internal, two cross] to ring-edge order."""
    cold_internal = rates[: phases - 1]
    hot_internal = rates[phases - 1 : 2 * phases - 2]
    cold_to_hot, hot_to_cold = rates[-2:]
    return np.concatenate(
        (cold_internal, [cold_to_hot], hot_internal, [hot_to_cold])
    )


def optimize(dt, dead_steps, phases, quantum_data, starts):
    lower, upper = np.log(1e-5), np.log(300.0)
    cache = {}

    def objective(log_rates):
        clipped = np.clip(log_rates, lower, upper)
        key = tuple(np.round(clipped, 8))
        if key in cache:
            return cache[key]
        model = bidirectional_ring_model(dt, phases, clipped)
        value = cross_kl_rate(quantum_data, model, dead_steps, dt)
        cache[key] = value
        return value

    answers = []
    for index, start in enumerate(starts, start=1):
        point, value, iterations = nelder_mead(
            objective, np.log(start), max_iterations=180
        )
        fitted = np.exp(np.clip(point, lower, upper))
        answers.append((value, fitted, iterations))
        print(
            f"phases={phases} start={index}/{len(starts)} "
            f"KLrate={value:.12g} reverse={fitted[-1]:.12g}",
            flush=True,
        )
    answers.sort(key=lambda item: item[0])
    point, value, iterations = nelder_mead(
        objective, np.log(answers[0][1]), max_iterations=240
    )
    answers.append((value, np.exp(np.clip(point, lower, upper)), iterations))
    answers.sort(key=lambda item: item[0])
    return answers[0], len(cache)


def main() -> None:
    dt, dead_steps = 0.01, 10
    quantum = quantum_split_model(4.0, dt, 0)[0]
    quantum_data = quantum_kernel(quantum, dead_steps)
    directional_rows = {
        int(row["phases_per_macrostate"]): row
        for row in csv.DictReader(
            Path("asymptotic_directional_phase_kl.csv").open(encoding="utf-8")
        )
    }

    rows = []
    for phases in (2, 3, 4, 5):
        source = directional_rows[phases]
        directional_rates = np.array(
            [float(value) for value in source["rates"].split()]
        )
        forward = directional_to_ring_order(directional_rates, phases)
        starts = (
            np.concatenate((forward, [0.05])),
            np.concatenate((forward, [0.8])),
        )
        best, evaluations = optimize(dt, dead_steps, phases, quantum_data, starts)
        forward_best, reverse_best = best[1][:-1], float(best[1][-1])
        directional_kl = float(source["directional_clock_kl_rate"])
        affinity = float(np.sum(np.log(forward_best / reverse_best)))
        rows.append(
            {
                "classical_states": 1 + 2 * phases,
                "phases_per_macrostate": phases,
                "parameter_count": 2 * phases + 1,
                "directional_kl_rate": directional_kl,
                "bidirectional_kl_rate": best[0],
                "relative_reduction_from_directional": 1.0 - best[0] / directional_kl,
                "shared_reverse_rate": reverse_best,
                "cycle_affinity": affinity,
                "forward_rates": " ".join(f"{value:.12g}" for value in forward_best),
                "objective_evaluations": evaluations,
            }
        )
        print(
            f"BEST states={1+2*phases} directional={directional_kl:.12g} "
            f"bidirectional={best[0]:.12g} reduction={1-best[0]/directional_kl:.3%} "
            f"reverse={reverse_best:.6g} affinity={affinity:.6g}",
            flush=True,
        )

    output = Path("asymptotic_bidirectional_ring_kl.csv")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
