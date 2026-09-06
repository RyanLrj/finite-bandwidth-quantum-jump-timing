"""Exact asymptotic record KL using the reset-induced Markov-renewal kernel."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from classical_memory_cross_kl import classical_split_model, quantum_split_model
from optimize_phase_cross_kl import nelder_mead


def generalized_phase_model(dt: float, phases: int, log_rates: np.ndarray):
    """Directed phase clock with shared internal rates and two completion rates."""
    rates = np.exp(log_rates)
    if len(rates) != phases + 1:
        raise ValueError("expected phases-1 internal rates plus two completion rates")
    internal = rates[: phases - 1]
    cold_to_hot, hot_to_cold = rates[-2:]
    size = 1 + 2 * phases

    def event(target: int, source: int, rate: float) -> np.ndarray:
        result = np.zeros((size, size), dtype=float)
        result[target, source] = rate
        return result

    cold_states = tuple(range(1, 1 + phases))
    hot_states = tuple(range(1 + phases, 1 + 2 * phases))
    cold_emission = sum((event(0, state, 1.0) for state in cold_states), np.zeros((size, size)))
    hot_emission = sum((event(0, state, 0.8) for state in hot_states), np.zeros((size, size)))
    thermal = (
        cold_emission,
        event(cold_states[0], 0, np.exp(-2.0)),
        hot_emission,
        event(hot_states[0], 0, 0.8 * np.exp(-1.25)),
    )

    work = np.zeros((size, size), dtype=float)
    for index, rate in enumerate(internal):
        work += event(cold_states[index + 1], cold_states[index], rate)
        work += event(hot_states[index + 1], hot_states[index], rate)
    work += event(hot_states[0], cold_states[-1], cold_to_hot)
    work += event(cold_states[0], hot_states[-1], hot_to_cold)
    work -= np.diag(work.sum(axis=0))
    return classical_split_model(dt, 0, work, thermal)[0]


def add_maps(maps, state):
    result = np.zeros_like(state)
    for mapping in maps:
        result += mapping(state)
    return result


def reset_states(model):
    base = sum(model.stationary_blocks)
    resets = []
    for mapping in model.thermal_maps:
        state = mapping(base)
        probability = model.trace(state)
        if probability <= 0:
            raise RuntimeError("thermal reset channel has zero stationary weight")
        resets.append(state / probability)
    return resets


def quantum_kernel(model, dead_steps: int, tail_tolerance: float = 1e-14):
    resets = reset_states(model)
    kernels = []
    mean_bins = np.zeros(4)
    transition = np.zeros((4, 4))
    tails = []
    lengths = []
    for previous, reset in enumerate(resets):
        state = reset
        for _ in range(dead_steps):
            state = add_maps(model.full_maps, state)
        probabilities = []
        for live_no_clicks in range(200_000):
            row = np.array(
                [model.trace(mapping(state)) for mapping in model.thermal_maps]
            )
            probabilities.append(row)
            transition[previous] += row
            mean_bins[previous] += row.sum() * (
                dead_steps + 1 + live_no_clicks
            )
            state = add_maps(model.hidden_live_maps, state)
            if model.trace(state) < tail_tolerance:
                break
        else:
            raise RuntimeError("waiting-time tail did not converge")
        kernels.append(np.asarray(probabilities))
        tails.append(model.trace(state))
        lengths.append(len(probabilities))
    transition /= transition.sum(axis=1, keepdims=True)
    a = transition.T - np.eye(4)
    b = np.zeros(4)
    a[-1, :] = 1.0
    b[-1] = 1.0
    embedded_stationary = np.linalg.solve(a, b)
    return kernels, mean_bins, transition, embedded_stationary, tails, lengths


def fixed_length_kernel(model, dead_steps: int, lengths):
    resets = reset_states(model)
    if resets[0].ndim == 1:
        size = resets[0].size

        def as_matrix(mapping):
            columns = []
            for index in range(size):
                basis = np.zeros(size)
                basis[index] = 1.0
                columns.append(mapping(basis))
            return np.column_stack(columns)

        no_matrix = sum(
            (as_matrix(mapping) for mapping in model.hidden_live_maps),
            np.zeros((size, size)),
        )
        full_matrix = sum(
            (as_matrix(mapping) for mapping in model.full_maps),
            np.zeros((size, size)),
        )
        thermal_matrices = tuple(as_matrix(mapping) for mapping in model.thermal_maps)
        thermal_rows = np.vstack(
            [np.ones(size) @ matrix for matrix in thermal_matrices]
        )
        blind_matrix = np.linalg.matrix_power(full_matrix, dead_steps)
        eigenvalues, eigenvectors = np.linalg.eig(no_matrix)
        inverse = np.linalg.inv(eigenvectors)
        left = thermal_rows @ eigenvectors
        kernels = []
        for previous, reset in enumerate(resets):
            state = blind_matrix @ reset
            right = inverse @ state
            coefficients = left * right[np.newaxis, :]
            powers = eigenvalues[np.newaxis, :] ** np.arange(
                lengths[previous]
            )[:, np.newaxis]
            probabilities = np.real_if_close(
                powers @ coefficients.T, tol=1000
            ).real
            kernels.append(np.maximum(probabilities, 0.0))
        return kernels

    kernels = []
    for previous, reset in enumerate(resets):
        state = reset
        for _ in range(dead_steps):
            state = add_maps(model.full_maps, state)
        probabilities = []
        for _ in range(lengths[previous]):
            probabilities.append(
                np.array(
                    [
                        model.trace(mapping(state))
                        for mapping in model.thermal_maps
                    ]
                )
            )
            state = add_maps(model.hidden_live_maps, state)
        kernels.append(np.asarray(probabilities))
    return kernels


def cross_kl_rate(quantum_data, classical_model, dead_steps: int, dt: float):
    q_kernels, q_mean_bins, _, q_stationary, tails, lengths = quantum_data
    c_kernels = fixed_length_kernel(classical_model, dead_steps, lengths)
    kl_by_previous = np.zeros(4)
    for previous in range(4):
        q = q_kernels[previous]
        c = c_kernels[previous]
        supported = q > 1e-300
        if np.any(c[supported] <= 0):
            return np.inf
        kl_by_previous[previous] = float(
            np.sum(q[supported] * np.log(q[supported] / c[supported]))
        )
    kl_per_cycle = float(q_stationary @ kl_by_previous)
    mean_cycle_time = float(q_stationary @ q_mean_bins) * dt
    return kl_per_cycle / mean_cycle_time


def optimize_dimension(
    phases: int,
    quantum_data,
    dead_steps: int,
    dt: float,
    starts: tuple[np.ndarray, ...],
):
    lower, upper = np.log(1e-4), np.log(200.0)
    cache = {}

    def objective(log_rates):
        clipped = np.clip(log_rates, lower, upper)
        key = tuple(np.round(clipped, 9))
        if key in cache:
            return cache[key]
        model = generalized_phase_model(dt, phases, clipped)
        value = cross_kl_rate(quantum_data, model, dead_steps, dt)
        cache[key] = value
        return value

    answers = []
    for index, start in enumerate(starts, start=1):
        point, value, iterations = nelder_mead(
            objective, np.log(start), max_iterations=90
        )
        rates = np.exp(np.clip(point, lower, upper))
        answers.append((value, rates, iterations))
        print(
            f"phases={phases} start={index}/{len(starts)} "
            f"KLrate={value:.12g} rates={rates}",
            flush=True,
        )
    answers.sort(key=lambda item: item[0])
    point, value, iterations = nelder_mead(
        objective, np.log(answers[0][1]), max_iterations=100
    )
    answers.append((value, np.exp(np.clip(point, lower, upper)), iterations))
    answers.sort(key=lambda item: item[0])
    return answers[0], answers, len(cache)


def main() -> None:
    dt = 0.01
    dead_steps = 10
    quantum = quantum_split_model(4.0, dt, 0)[0]
    quantum_data = quantum_kernel(quantum, dead_steps)
    _, mean_bins, transition, stationary, tails, lengths = quantum_data
    print(f"embedded click-label stationary distribution: {stationary}")
    print(f"mean inter-click time: {stationary @ mean_bins * dt:.12g}")
    print(f"kernel lengths: {lengths}; maximum omitted tail={max(tails):.3e}")
    print(f"embedded transition row errors: {np.max(np.abs(transition.sum(axis=1)-1)):.3e}")

    starts_by_phases = {
        1: (
            np.array([8.88888889, 8.88888889]),
            np.array([3.0, 4.0]),
        ),
        2: (
            np.array([4.31663448, 2.82399503, 3.53811770]),
            np.array([10.0, 1.0, 1.0]),
        ),
        3: (
            np.array([6.0, 6.0, 2.8, 3.5]),
            np.array([2.0, 5.0, 3.0, 4.0]),
        ),
        4: (
            np.array([8.0, 8.0, 8.0, 6.8, 9.2]),
            np.array([5.0, 7.0, 9.0, 6.0, 8.0]),
        ),
        5: (
            np.array([10.0, 10.0, 10.0, 10.0, 8.2, 11.5]),
            np.array([6.0, 8.0, 10.0, 12.0, 7.5, 10.0]),
        ),
    }
    rows = []
    for phases, starts in starts_by_phases.items():
        best, answers, evaluations = optimize_dimension(
            phases, quantum_data, dead_steps, dt, starts
        )
        row = {
            "classical_states": 1 + 2 * phases,
            "phases_per_excited_macrostate": phases,
            "dt": dt,
            "dead_steps": dead_steps,
            "dead_time": dead_steps * dt,
            "asymptotic_kl_rate": best[0],
            "rates": " ".join(f"{value:.12g}" for value in best[1]),
            "iterations": best[2],
            "objective_evaluations": evaluations,
        }
        rows.append(row)
        print(
            f"BEST states={row['classical_states']} "
            f"asymptotic_KLrate={best[0]:.12g} rates={best[1]} "
            f"evaluations={evaluations}",
            flush=True,
        )

    output = Path("asymptotic_markov_renewal_kl.csv")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
