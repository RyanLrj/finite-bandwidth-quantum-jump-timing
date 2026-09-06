"""Exact record KL from the quantum process to classical memory models."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from exact_finite_horizon_kl import finite_horizon_kl
from marked_record_path_kl import RecordModel
from pair_correlation_expansion import matrix_exponential
from petz_deadtime_kl import stationary_blocks_from_maps
from passive_deadtime_quantum_filter import apply, matrix_sqrt_positive


def stationary_vector_map(mapping: np.ndarray) -> np.ndarray:
    a = mapping.astype(complex) - np.eye(mapping.shape[0])
    b = np.zeros(mapping.shape[0], dtype=complex)
    a[-1, :] = 1.0
    b[-1] = 1.0
    answer = np.linalg.solve(a, b)
    return np.real_if_close(answer, tol=1000).real


def quantum_split_model(omega: float, dt: float, dead_steps: int) -> tuple[RecordModel, np.ndarray]:
    size = 3
    h = np.zeros((size, size), dtype=complex)
    h[1, 2] = h[2, 1] = omega / 2.0
    values, vectors = np.linalg.eigh(h)
    half_unitary = (vectors * np.exp(-1j * values * dt / 2.0)) @ vectors.conj().T

    def operator(target: int, source: int, rate: float) -> np.ndarray:
        result = np.zeros((size, size), dtype=complex)
        result[target, source] = np.sqrt(dt * rate)
        return result

    middle_jumps = (
        operator(0, 1, 1.0),
        operator(1, 0, np.exp(-2.0)),
        operator(0, 2, 0.8),
        operator(2, 0, 0.8 * np.exp(-1.25)),
    )
    rate_effect = sum(jump.conj().T @ jump for jump in middle_jumps)
    middle_no_jump = matrix_sqrt_positive(np.eye(size) - rate_effect)
    no_jump = half_unitary @ middle_no_jump @ half_unitary
    jumps = tuple(half_unitary @ jump @ half_unitary for jump in middle_jumps)
    kraus = (no_jump,) + jumps
    superoperator = sum(np.kron(k.conj(), k) for k in kraus)
    a = superoperator - np.eye(size * size)
    b = np.zeros(size * size, dtype=complex)
    a[-1, :] = np.eye(size).reshape(-1, order="F")
    b[-1] = 1.0
    rho = np.linalg.solve(a, b).reshape((size, size), order="F")
    rho = (rho + rho.conj().T) / 2.0
    maps = tuple((lambda k: (lambda state: apply(k, state)))(k) for k in kraus)
    trace = lambda state: float(np.trace(state).real)
    blocks = stationary_blocks_from_maps((maps[0],), maps[1:], trace, rho, dead_steps)
    return RecordModel((maps[0],), maps[1:], maps, trace, blocks), np.diag(rho).real


def classical_split_model(
    dt: float,
    dead_steps: int,
    work_generator: np.ndarray,
    thermal_rates: tuple[np.ndarray, ...],
) -> tuple[RecordModel, np.ndarray]:
    size = work_generator.shape[0]
    half_work = np.asarray(matrix_exponential(work_generator * dt / 2.0), dtype=float)
    outgoing = sum(thermal_rates).sum(axis=0)
    middle_no_event = np.eye(size) - dt * np.diag(outgoing)
    no_event = half_work @ middle_no_event @ half_work
    thermal = tuple(half_work @ (dt * event) @ half_work for event in thermal_rates)
    full = no_event + sum(thermal)
    stationary = stationary_vector_map(full)

    def wrap(matrix: np.ndarray):
        return lambda state: matrix @ state

    no_map = wrap(no_event)
    thermal_maps = tuple(wrap(matrix) for matrix in thermal)
    trace = lambda state: float(state.sum())
    blocks = stationary_blocks_from_maps((no_map,), thermal_maps, trace, stationary, dead_steps)
    return RecordModel((no_map,), thermal_maps, (no_map,) + thermal_maps, trace, blocks), stationary


def three_state_split_model(dt: float, dead_steps: int, work_rate: float) -> tuple[RecordModel, np.ndarray]:
    size = 3

    def event(target: int, source: int, rate: float) -> np.ndarray:
        result = np.zeros((size, size), dtype=float)
        result[target, source] = rate
        return result

    thermal = (
        event(0, 1, 1.0),
        event(1, 0, np.exp(-2.0)),
        event(0, 2, 0.8),
        event(2, 0, 0.8 * np.exp(-1.25)),
    )
    work_generator = event(2, 1, work_rate) + event(1, 2, work_rate)
    work_generator -= np.diag(work_generator.sum(axis=0))
    return classical_split_model(dt, dead_steps, work_generator, thermal)


def phase_instrument(
    dt: float,
    rates: tuple[float, float, float],
) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...], np.ndarray]:
    first, cold_to_hot, hot_to_cold = rates
    size = 5
    cold_down, cold_up = 1.0, np.exp(-2.0)
    hot_down, hot_up = 0.8, 0.8 * np.exp(-1.25)

    def event(target: int, source: int, rate: float) -> np.ndarray:
        result = np.zeros((size, size), dtype=float)
        result[target, source] = dt * rate
        return result

    thermal = (
        event(0, 1, cold_down) + event(0, 2, cold_down),
        event(1, 0, cold_up),
        event(0, 3, hot_down) + event(0, 4, hot_down),
        event(3, 0, hot_up),
    )
    work = (
        event(2, 1, first),
        event(3, 2, cold_to_hot),
        event(4, 3, first),
        event(1, 4, hot_to_cold),
    )
    outgoing = sum(thermal + work)
    no_event = np.eye(size) - np.diag(outgoing.sum(axis=0))
    if no_event.diagonal().min() < 0:
        raise ValueError("time step is too large for these hidden rates")
    full = no_event + sum(thermal + work)
    a = full - np.eye(size)
    b = np.zeros(size)
    a[-1, :] = 1.0
    b[-1] = 1.0
    stationary = np.linalg.solve(a, b)
    return no_event, work, thermal, stationary


def phase_record_model(
    dt: float,
    dead_steps: int,
    rates: tuple[float, float, float],
) -> RecordModel:
    first, cold_to_hot, hot_to_cold = rates
    size = 5

    def event(target: int, source: int, rate: float) -> np.ndarray:
        result = np.zeros((size, size), dtype=float)
        result[target, source] = rate
        return result

    thermal = (
        event(0, 1, 1.0) + event(0, 2, 1.0),
        event(1, 0, np.exp(-2.0)),
        event(0, 3, 0.8) + event(0, 4, 0.8),
        event(3, 0, 0.8 * np.exp(-1.25)),
    )
    work_generator = (
        event(2, 1, first)
        + event(3, 2, cold_to_hot)
        + event(4, 3, first)
        + event(1, 4, hot_to_cold)
    )
    work_generator -= np.diag(work_generator.sum(axis=0))
    return classical_split_model(dt, dead_steps, work_generator, thermal)[0]


def aggregate(state: np.ndarray) -> np.ndarray:
    return np.array((state[0], state[1] + state[2], state[3] + state[4]))


def fit_equal_stage_to_discrete_quantum(omega: float, dt: float) -> float:
    _, target = quantum_split_model(omega, dt, 0)
    logs = np.linspace(np.log(0.05), np.log(1.0 / dt - 1.1), 5000)
    best = None
    for log_rate in logs:
        rate = float(np.exp(log_rate))
        model = phase_record_model(dt, 0, (rate, rate, rate))
        state = next(iter(model.stationary_blocks))
        error = float(np.linalg.norm(aggregate(state) - target))
        if best is None or error < best[0]:
            best = (error, rate)
    assert best is not None
    lower = np.log(best[1]) - 0.02
    upper = np.log(best[1]) + 0.02
    # Golden-section refinement of aggregate-population mismatch.
    ratio = (np.sqrt(5.0) - 1.0) / 2.0

    def objective(log_rate: float) -> float:
        rate = float(np.exp(log_rate))
        model = phase_record_model(dt, 0, (rate, rate, rate))
        state = next(iter(model.stationary_blocks))
        return float(np.linalg.norm(aggregate(state) - target))

    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    for _ in range(100):
        if objective(left) < objective(right):
            upper, right = right, left
            left = upper - ratio * (upper - lower)
        else:
            lower, left = left, right
            right = lower + ratio * (upper - lower)
    return float(np.exp(0.5 * (lower + upper)))


def main() -> None:
    omega, dt = 4.0, 0.02
    equal_rate = fit_equal_stage_to_discrete_quantum(omega, dt)
    cubic_rates = (8.15514029, 0.11191284, 0.46783069)
    _, quantum_target = quantum_split_model(omega, dt, 0)
    work_logs = np.linspace(np.log(0.05), np.log(40.0), 4000)
    best_work = None
    for log_rate in work_logs:
        rate = float(np.exp(log_rate))
        _, state = three_state_split_model(dt, 0, rate)
        error = float(np.linalg.norm(state - quantum_target))
        if best_work is None or error < best_work[0]:
            best_work = (error, rate)
    assert best_work is not None
    three_state_rate = best_work[1]
    print(f"discrete population-fitted equal stage rate: {equal_rate:.12g}")
    print(f"discrete population-fitted three-state work rate: {three_state_rate:.12g}")

    cases = ((0, 6), (0, 8), (0, 10), (5, 8), (5, 12), (10, 12), (10, 20))
    rows: list[dict[str, float | str]] = []
    for dead_steps, length in cases:
        quantum, _ = quantum_split_model(omega, dt, dead_steps)
        three_state, _ = three_state_split_model(dt, dead_steps, three_state_rate)
        candidates = (
            ("classical_3_markov", three_state),
            (
                "classical_5_equal_phase",
                phase_record_model(dt, dead_steps, (equal_rate, equal_rate, equal_rate)),
            ),
            (
                "classical_5_cubic_phase",
                phase_record_model(dt, dead_steps, cubic_rates),
            ),
        )
        for name, candidate in candidates:
            kl, normalization_error, words = finite_horizon_kl(
                quantum, candidate, dead_steps, length
            )
            row = {
                "candidate": name,
                "dt": dt,
                "dead_steps": dead_steps,
                "dead_time": dead_steps * dt,
                "length": length,
                "duration": length * dt,
                "quantum_to_classical_kl": kl,
                "kl_rate": kl / (length * dt),
                "normalization_error": normalization_error,
                "supported_words": words,
            }
            rows.append(row)
            print(
                f"{name:27s} D={dead_steps:2d} n={length:2d} "
                f"KL={kl:.9g} rate={kl/(length*dt):.9g} words={words}"
            )

    output = Path("classical_memory_cross_kl.csv")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
