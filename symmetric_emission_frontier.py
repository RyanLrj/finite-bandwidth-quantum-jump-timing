"""Exact fixed-affinity frontier for a qutrit with equal emission rates."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from asymptotic_finite_ep_frontier import entropy_production
from asymptotic_markov_renewal_kl import cross_kl_rate, quantum_kernel
from classical_memory_cross_kl import classical_split_model
from exact_finite_horizon_kl import RecordModel
from optimize_phase_cross_kl import nelder_mead
from pair_correlation_expansion import matrix_exponential
from passive_deadtime_quantum_filter import apply, matrix_sqrt_positive
from petz_deadtime_kl import stationary_blocks_from_maps


def stationary_vector_map(mapping: np.ndarray) -> np.ndarray:
    matrix = mapping.astype(complex) - np.eye(mapping.shape[0])
    right = np.zeros(mapping.shape[0], dtype=complex)
    matrix[-1, :] = 1.0
    right[-1] = 1.0
    return np.real_if_close(np.linalg.solve(matrix, right), tol=1000).real


def symmetric_quantum_model(
    omega: float,
    dt: float,
    emission_rate: float = 0.9,
    cold_gap: float = 2.0,
    hot_gap: float = 1.25,
):
    size = 3
    hamiltonian = np.zeros((size, size), dtype=complex)
    hamiltonian[1, 2] = hamiltonian[2, 1] = omega / 2.0
    values, vectors = np.linalg.eigh(hamiltonian)
    half_unitary = (
        vectors * np.exp(-1j * values * dt / 2.0)
    ) @ vectors.conj().T

    def operator(target: int, source: int, rate: float) -> np.ndarray:
        result = np.zeros((size, size), dtype=complex)
        result[target, source] = np.sqrt(dt * rate)
        return result

    middle_jumps = (
        operator(0, 1, emission_rate),
        operator(1, 0, emission_rate * np.exp(-cold_gap)),
        operator(0, 2, emission_rate),
        operator(2, 0, emission_rate * np.exp(-hot_gap)),
    )
    rate_effect = sum(jump.conj().T @ jump for jump in middle_jumps)
    middle_no_jump = matrix_sqrt_positive(np.eye(size) - rate_effect)
    no_jump = half_unitary @ middle_no_jump @ half_unitary
    jumps = tuple(half_unitary @ jump @ half_unitary for jump in middle_jumps)
    kraus = (no_jump,) + jumps
    superoperator = sum(np.kron(k.conj(), k) for k in kraus)
    matrix = superoperator - np.eye(size * size)
    right = np.zeros(size * size, dtype=complex)
    matrix[-1, :] = np.eye(size).reshape(-1, order="F")
    right[-1] = 1.0
    rho = np.linalg.solve(matrix, right).reshape((size, size), order="F")
    rho = (rho + rho.conj().T) / 2.0
    maps = tuple((lambda k: (lambda state: apply(k, state)))(k) for k in kraus)
    trace = lambda state: float(np.trace(state).real)
    blocks = stationary_blocks_from_maps((maps[0],), maps[1:], trace, rho, 0)
    return RecordModel((maps[0],), maps[1:], maps, trace, blocks)


def symmetric_classical_model(
    dt: float,
    phases: int,
    log_forward: np.ndarray,
    reverse_fraction: float,
    emission_rate: float = 0.9,
    cold_gap: float = 2.0,
    hot_gap: float = 1.25,
):
    forward = np.exp(log_forward)
    size = 1 + 2 * phases

    def event(target: int, source: int, rate: float) -> np.ndarray:
        result = np.zeros((size, size), dtype=float)
        result[target, source] = rate
        return result

    cold = tuple(range(1, 1 + phases))
    hot = tuple(range(1 + phases, 1 + 2 * phases))
    ring = cold + hot
    cold_up = emission_rate * np.exp(-cold_gap) / phases
    hot_up = emission_rate * np.exp(-hot_gap) / phases
    thermal = (
        sum((event(0, state, emission_rate) for state in cold), np.zeros((size, size))),
        sum((event(state, 0, cold_up) for state in cold), np.zeros((size, size))),
        sum((event(0, state, emission_rate) for state in hot), np.zeros((size, size))),
        sum((event(state, 0, hot_up) for state in hot), np.zeros((size, size))),
    )
    work = np.zeros((size, size), dtype=float)
    work_pairs = []
    for index, source in enumerate(ring):
        target = ring[(index + 1) % len(ring)]
        fwd = float(forward[index])
        rev = reverse_fraction * fwd
        work += event(target, source, fwd) + event(source, target, rev)
        work_pairs.append((source, target, fwd, rev))
    work -= np.diag(work.sum(axis=0))
    thermal_pairs = [
        *((state, 0, emission_rate, cold_up) for state in cold),
        *((state, 0, emission_rate, hot_up) for state in hot),
    ]
    model = classical_split_model(dt, 0, work, thermal)[0]
    return model, work, thermal, work_pairs, thermal_pairs


def load_seeds():
    rows = []
    for path in (
        "finite_ep_fixed_affinity_scaling.csv",
        "finite_ep_fixed_affinity_extension.csv",
        "finite_ep_fixed_affinity_large_state.csv",
    ):
        rows.extend(csv.DictReader(Path(path).open(encoding="utf-8")))
    return {
        (float(row["hidden_cycle_affinity"]), int(row["phases_per_macrostate"])):
        np.array([float(value) for value in row["forward_rates"].split()])
        for row in rows
    }


def optimize_point(dt, dead_steps, phases, affinity, quantum_data, starts):
    reverse_fraction = float(np.exp(-affinity / (2.0 * phases)))
    lower, upper = np.log(1e-3), np.log(100.0)
    cache = {}

    def objective(log_forward):
        clipped = np.clip(log_forward, lower, upper)
        key = tuple(np.round(clipped, 8))
        if key in cache:
            return cache[key]
        model = symmetric_classical_model(
            dt, phases, clipped, reverse_fraction
        )[0]
        value = cross_kl_rate(quantum_data, model, dead_steps, dt)
        cache[key] = value
        return value

    ranked = [(objective(np.log(start)), start) for start in starts]
    ranked.sort(key=lambda item: item[0])
    point, value, iterations = nelder_mead(
        objective, np.log(ranked[0][1]), max_iterations=160
    )
    point, value, refine_iterations = nelder_mead(
        objective, np.clip(point, lower, upper), max_iterations=140
    )
    rates = np.exp(np.clip(point, lower, upper))
    return value, rates, iterations + refine_iterations, len(cache)


def main():
    omega, dt, dead_steps = 4.0, 0.01, 10
    quantum = symmetric_quantum_model(omega, dt)
    quantum_data = quantum_kernel(quantum, dead_steps)
    seeds = load_seeds()
    rows = []
    for affinity in (20.0, 40.0, 60.0):
        previous = None
        for phases in (5, 10, 15):
            starts = [
                seeds[(affinity, phases)],
                np.full(2 * phases, 1.3 * phases),
            ]
            if previous is not None:
                from finite_ep_state_scaling import expand_rates

                starts.append(expand_rates(previous[1], previous[0], phases))
            kl_rate, rates, iterations, evaluations = optimize_point(
                dt, dead_steps, phases, affinity, quantum_data, tuple(starts)
            )
            reverse_fraction = float(np.exp(-affinity / (2.0 * phases)))
            model_data = symmetric_classical_model(
                dt, phases, np.log(rates), reverse_fraction
            )
            ep = entropy_production(*model_data[1:])
            rows.append(
                {
                    "classical_states": 1 + 2 * phases,
                    "phases_per_macrostate": phases,
                    "hidden_cycle_affinity": affinity,
                    "reverse_fraction": reverse_fraction,
                    "asymptotic_kl_rate": kl_rate,
                    **ep,
                    "forward_rates": " ".join(f"{value:.12g}" for value in rates),
                    "iterations": iterations,
                    "objective_evaluations": evaluations,
                }
            )
            previous = (phases, rates)
            print(
                f"A={affinity:g} p={phases} states={1+2*phases} "
                f"KL={kl_rate:.12g} EP={ep['total_ep_rate']:.9g}",
                flush=True,
            )
    output = Path("symmetric_emission_frontier.csv")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
