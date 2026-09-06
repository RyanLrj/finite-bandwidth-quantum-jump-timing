"""Finite-entropy-production classical frontier for the renewal-record KL.

Every hidden work edge and every thermal edge has a nonzero reverse partner.
The reverse fraction controls circulation of the hidden phase ring.  The reset
mixing controls how sharply an observed absorption localises the hidden phase:
zero approaches the original pure reset, while one gives a uniform reset.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from asymptotic_bidirectional_ring_kl import directional_to_ring_order
from asymptotic_markov_renewal_kl import cross_kl_rate, quantum_kernel
from classical_memory_cross_kl import classical_split_model, quantum_split_model
from optimize_phase_cross_kl import nelder_mead


def finite_ep_model(
    dt: float,
    phases: int,
    log_forward: np.ndarray,
    reverse_fraction: float,
    reset_mixing: float,
):
    forward = np.exp(log_forward)
    if len(forward) != 2 * phases:
        raise ValueError("expected one forward rate per hidden-ring edge")
    if not (0.0 < reverse_fraction <= 1.0):
        raise ValueError("reverse_fraction must lie in (0,1]")
    if not (0.0 < reset_mixing <= 1.0):
        raise ValueError("reset_mixing must lie in (0,1]")
    size = 1 + 2 * phases

    def event(target: int, source: int, rate: float) -> np.ndarray:
        result = np.zeros((size, size), dtype=float)
        result[target, source] = rate
        return result

    cold = tuple(range(1, 1 + phases))
    hot = tuple(range(1 + phases, 1 + 2 * phases))
    ring = cold + hot

    # A convex mixture of a phase-localised and uniform absorption reset.
    weights = np.full(phases, reset_mixing / phases)
    weights[0] += 1.0 - reset_mixing
    cold_up_rates = np.exp(-2.0) * weights
    hot_up_rates = 0.8 * np.exp(-1.25) * weights
    cold_down_rate = 1.0
    hot_down_rate = 0.8

    thermal = (
        sum((event(0, state, cold_down_rate) for state in cold), np.zeros((size, size))),
        sum(
            (event(state, 0, float(rate)) for state, rate in zip(cold, cold_up_rates)),
            np.zeros((size, size)),
        ),
        sum((event(0, state, hot_down_rate) for state in hot), np.zeros((size, size))),
        sum(
            (event(state, 0, float(rate)) for state, rate in zip(hot, hot_up_rates)),
            np.zeros((size, size)),
        ),
    )

    work = np.zeros((size, size), dtype=float)
    work_pairs = []
    for index, source in enumerate(ring):
        target = ring[(index + 1) % len(ring)]
        fwd = float(forward[index])
        rev = reverse_fraction * fwd
        work += event(target, source, fwd)
        work += event(source, target, rev)
        work_pairs.append((source, target, fwd, rev))
    work -= np.diag(work.sum(axis=0))

    thermal_pairs = []
    for state, up in zip(cold, cold_up_rates):
        thermal_pairs.append((state, 0, cold_down_rate, float(up)))
    for state, up in zip(hot, hot_up_rates):
        thermal_pairs.append((state, 0, hot_down_rate, float(up)))

    model = classical_split_model(dt, 0, work, thermal)[0]
    return model, work, thermal, work_pairs, thermal_pairs


def stationary_generator(generator: np.ndarray) -> np.ndarray:
    matrix = generator.copy()
    right = np.zeros(generator.shape[0])
    matrix[-1, :] = 1.0
    right[-1] = 1.0
    return np.linalg.solve(matrix, right)


def entropy_production(work, thermal, work_pairs, thermal_pairs):
    thermal_generator = sum(thermal)
    thermal_generator -= np.diag(thermal_generator.sum(axis=0))
    probability = stationary_generator(work + thermal_generator)

    work_total = 0.0
    work_medium = 0.0
    for source, target, fwd, rev in work_pairs:
        forward_flux = fwd * probability[source]
        reverse_flux = rev * probability[target]
        current = forward_flux - reverse_flux
        work_total += current * np.log(forward_flux / reverse_flux)
        work_medium += current * np.log(fwd / rev)

    thermal_total = 0.0
    thermal_medium = 0.0
    for excited, ground, down, up in thermal_pairs:
        down_flux = down * probability[excited]
        up_flux = up * probability[ground]
        current = down_flux - up_flux
        thermal_total += current * np.log(down_flux / up_flux)
        thermal_medium += current * np.log(down / up)

    return {
        "work_total_ep_rate": float(work_total),
        "thermal_total_ep_rate": float(thermal_total),
        "total_ep_rate": float(work_total + thermal_total),
        "work_medium_entropy_rate": float(work_medium),
        "thermal_medium_entropy_rate": float(thermal_medium),
        "medium_total_rate": float(work_medium + thermal_medium),
        "stationary_balance_error": float(
            work_total + thermal_total - work_medium - thermal_medium
        ),
        "minimum_stationary_probability": float(probability.min()),
    }


def optimize_point(
    dt,
    dead_steps,
    phases,
    reverse_fraction,
    reset_mixing,
    quantum_data,
    starts,
):
    lower, upper = np.log(1e-3), np.log(100.0)
    cache = {}

    def objective(log_forward):
        clipped = np.clip(log_forward, lower, upper)
        key = tuple(np.round(clipped, 8))
        if key in cache:
            return cache[key]
        model = finite_ep_model(
            dt,
            phases,
            clipped,
            reverse_fraction,
            reset_mixing,
        )[0]
        value = cross_kl_rate(quantum_data, model, dead_steps, dt)
        cache[key] = value
        return value

    start_values = [(objective(np.log(start)), start) for start in starts]
    start_values.sort(key=lambda item: item[0])
    point, value, iterations = nelder_mead(
        objective, np.log(start_values[0][1]), max_iterations=120
    )
    point, value, refine_iterations = nelder_mead(
        objective, np.clip(point, lower, upper), max_iterations=100
    )
    rates = np.exp(np.clip(point, lower, upper))
    return value, rates, iterations + refine_iterations, len(cache)


def main():
    dt, dead_steps, phases = 0.01, 10, 2
    reverse_values = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
    reset_values = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
    quantum = quantum_split_model(4.0, dt, 0)[0]
    quantum_data = quantum_kernel(quantum, dead_steps)

    directional_row = next(
        row
        for row in csv.DictReader(
            Path("asymptotic_directional_phase_kl.csv").open(encoding="utf-8")
        )
        if int(row["phases_per_macrostate"]) == phases
    )
    directional_rates = np.array(
        [float(value) for value in directional_row["rates"].split()]
    )
    base = directional_to_ring_order(directional_rates, phases)
    solutions = {}
    rows = []

    for reverse_index, reverse_fraction in enumerate(reverse_values):
        for reset_index, reset_mixing in enumerate(reset_values):
            starts = [base]
            if reset_index > 0:
                starts.append(solutions[(reverse_fraction, reset_values[reset_index - 1])])
            if reverse_index > 0:
                starts.append(solutions[(reverse_values[reverse_index - 1], reset_mixing)])
            kl_rate, rates, iterations, evaluations = optimize_point(
                dt,
                dead_steps,
                phases,
                reverse_fraction,
                reset_mixing,
                quantum_data,
                tuple(starts),
            )
            solutions[(reverse_fraction, reset_mixing)] = rates
            model_data = finite_ep_model(
                dt,
                phases,
                np.log(rates),
                reverse_fraction,
                reset_mixing,
            )
            ep = entropy_production(*model_data[1:])
            row = {
                "classical_states": 1 + 2 * phases,
                "phases_per_macrostate": phases,
                "reverse_fraction": reverse_fraction,
                "reset_mixing": reset_mixing,
                "hidden_cycle_affinity": 2 * phases * np.log(1.0 / reverse_fraction),
                "asymptotic_kl_rate": kl_rate,
                **ep,
                "forward_rates": " ".join(f"{value:.12g}" for value in rates),
                "iterations": iterations,
                "objective_evaluations": evaluations,
            }
            rows.append(row)
            print(
                f"delta={reverse_fraction:5g} lambda={reset_mixing:5g} "
                f"KL={kl_rate:.9g} EP={ep['total_ep_rate']:.9g} "
                f"balance={ep['stationary_balance_error']:.2e}",
                flush=True,
            )

    output = Path("asymptotic_finite_ep_frontier.csv")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
