"""Ordered two-jump expansion for small detector dead time.

Computes c_{nu,mu}^{(n)}(0) = Tr[J_mu L^n J_nu rho_ss] for
the resonantly driven qutrit and its stationary-population/current-matched
classical model.  It then compares the predicted quantum--classical quadratic
dead-time difference with exact augmented-stationary calculations.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from deadtime_exact_rates import exact_rates as quantum_deadtime_rates
from matched_classical_deadtime import classical_exact_rates


N = 3


def vec(a: np.ndarray) -> np.ndarray:
    return a.reshape(-1, order="F")


def lr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.kron(b.T, a)


def dissipator(jump: np.ndarray) -> np.ndarray:
    eye = np.eye(N, dtype=complex)
    jj = jump.conj().T @ jump
    return lr(jump, jump.conj().T) - 0.5 * (lr(jj, eye) + lr(eye, jj))


def quantum_continuous(omega: float) -> tuple[np.ndarray, tuple[np.ndarray, ...], np.ndarray]:
    h = np.zeros((N, N), dtype=complex)
    h[1, 2] = h[2, 1] = omega / 2.0
    eye = np.eye(N, dtype=complex)
    liouvillian = -1j * (lr(h, eye) - lr(eye, h))

    def jump(target: int, source: int, rate: float) -> np.ndarray:
        result = np.zeros((N, N), dtype=complex)
        result[target, source] = np.sqrt(rate)
        return result

    operators = (
        jump(0, 1, 1.0),
        jump(1, 0, np.exp(-2.0)),
        jump(0, 2, 0.8),
        jump(2, 0, 0.8 * np.exp(-1.25)),
    )
    for operator in operators:
        liouvillian += dissipator(operator)
    jump_maps = tuple(lr(operator, operator.conj().T) for operator in operators)

    a = liouvillian.copy()
    b = np.zeros(N * N, dtype=complex)
    a[-1, :] = vec(np.eye(N))
    b[-1] = 1.0
    rho = np.linalg.solve(a, b)
    return liouvillian, jump_maps, rho


def classical_continuous(target: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...], np.ndarray, float]:
    cold_up = np.exp(-2.0)
    hot_up = 0.8 * np.exp(-1.25)
    work_rate = (target[1] - target[0] * cold_up) / (target[2] - target[1])

    def event(target_state: int, source_state: int, rate: float) -> np.ndarray:
        result = np.zeros((N, N), dtype=float)
        result[target_state, source_state] = rate
        return result
    thermal = (
        event(0, 1, 1.0),
        event(1, 0, cold_up),
        event(0, 2, 0.8),
        event(2, 0, hot_up),
    )
    work = (event(2, 1, work_rate), event(1, 2, work_rate))
    generator = sum(thermal + work)
    generator -= np.diag(generator.sum(axis=0))
    return generator, thermal, target, float(work_rate)


def correlation_sums(
    generator: np.ndarray,
    jump_maps: tuple[np.ndarray, ...],
    stationary: np.ndarray,
    trace_row: np.ndarray,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    entropy = np.array([2.0, -2.0, 1.25, -1.25])
    matrices = []
    for order in range(3):
        derivative = np.zeros((4, 4), dtype=float)
        power = np.linalg.matrix_power(generator, order)
        for nu in range(4):
            post = jump_maps[nu] @ stationary
            for mu in range(4):
                derivative[nu, mu] = float(np.real(trace_row @ jump_maps[mu] @ power @ post))
        matrices.append(derivative)
    weighted = [float(np.sum(matrix * entropy[np.newaxis, :])) for matrix in matrices]
    return weighted[0], weighted[1], weighted[2], matrices[0], matrices[1]


def matrix_exponential(a: np.ndarray) -> np.ndarray:
    """Matrix exponential via eigendecomposition for the small generators here."""
    values, vectors = np.linalg.eig(a)
    inverse = np.linalg.inv(vectors)
    result = (vectors * np.exp(values)) @ inverse
    return np.real_if_close(result, tol=1000)


def continuous_deadtime_rate(
    generator: np.ndarray,
    jump_maps: tuple[np.ndarray, ...],
    trace_row: np.ndarray,
    tau: float,
) -> float:
    """Exact stationary weighted click rate for continuous nonparalyzable dead time.

    If r is the unnormalised live-state block, stationarity gives
        [L-J + exp(L tau) J] r = 0,
    while total normalisation is Tr(r) + tau Tr(J r) = 1.
    """
    entropy = np.array([2.0, -2.0, 1.25, -1.25])
    total_jump = sum(jump_maps)
    live_generator = generator - total_jump
    return_map = matrix_exponential(generator * tau)
    balance = live_generator + return_map @ total_jump
    normalisation = trace_row + tau * (trace_row @ total_jump)
    a = np.asarray(balance, dtype=complex).copy()
    b = np.zeros(generator.shape[0], dtype=complex)
    a[-1, :] = normalisation
    b[-1] = 1.0
    live = np.linalg.solve(a, b)
    rates = np.array(
        [float(np.real(trace_row @ jump_map @ live)) for jump_map in jump_maps]
    )
    return float(rates @ entropy)


def deadtime_series_coefficients(
    generator: np.ndarray,
    jump_maps: tuple[np.ndarray, ...],
    stationary: np.ndarray,
    trace_row: np.ndarray,
    max_order: int,
) -> np.ndarray:
    """Taylor coefficients of the exact continuous dead-time visible rate."""
    entropy = np.array([2.0, -2.0, 1.25, -1.25])
    total_jump = sum(jump_maps)
    weighted_jump = sum(
        entropy[index] * (trace_row @ jump_map)
        for index, jump_map in enumerate(jump_maps)
    )
    state_coefficients = [stationary]
    factorial = 1.0
    generator_power = np.eye(generator.shape[0], dtype=generator.dtype)
    powers_over_factorial = [generator_power]
    for order in range(1, max_order + 1):
        factorial *= order
        generator_power = generator_power @ generator
        powers_over_factorial.append(generator_power / factorial)
    for order in range(1, max_order + 1):
        coefficient = np.zeros_like(stationary, dtype=complex)
        for k in range(1, order + 1):
            coefficient -= (
                powers_over_factorial[k - 1]
                @ total_jump
                @ state_coefficients[order - k]
                / k
            )
        state_coefficients.append(coefficient)
    return np.array(
        [float(np.real(weighted_jump @ state)) for state in state_coefficients]
    )


def main() -> None:
    omega = 4.0
    quantum_l, quantum_jumps, quantum_rho = quantum_continuous(omega)
    quantum_pop = np.array([quantum_rho[i + N * i].real for i in range(N)])
    classical_l, classical_jumps, classical_p, work_rate = classical_continuous(quantum_pop)
    q = correlation_sums(quantum_l, quantum_jumps, quantum_rho, vec(np.eye(N)))
    c = correlation_sums(classical_l, classical_jumps, classical_p, np.ones(N))

    # sigma_visible(tau) = sigma - S0*tau - S1*tau^2/2 + ... plus
    # overlap corrections. The latter have identical zero-time statistics in
    # the matched models, so the leading predicted difference is based on S1.
    predicted_difference_quadratic = -0.5 * (q[1] - c[1])
    entropy = np.array([2.0, -2.0, 1.25, -1.25])
    q_total_jump = sum(quantum_jumps)
    c_total_jump = sum(classical_jumps)
    q_weighted_jump = sum(
        entropy[index] * (vec(np.eye(N)) @ jump_map)
        for index, jump_map in enumerate(quantum_jumps)
    )
    c_weighted_jump = sum(
        entropy[index] * (np.ones(N) @ jump_map)
        for index, jump_map in enumerate(classical_jumps)
    )
    q_triple = float(np.real(q_weighted_jump @ q_total_jump @ q_total_jump @ quantum_rho))
    c_triple = float(np.real(c_weighted_jump @ c_total_jump @ c_total_jump @ classical_p))
    q_quadratic = q_triple - 0.5 * q[1]
    c_quadratic = c_triple - 0.5 * c[1]
    print(f"continuous stationary populations: {quantum_pop}")
    print(f"matched work rate: {work_rate:.12g}")
    print(f"weighted pair sum S0 quantum={q[0]:.12g} classical={c[0]:.12g}")
    print(f"weighted derivative S1 quantum={q[1]:.12g} classical={c[1]:.12g}")
    print(f"weighted second derivative S2 quantum={q[2]:.12g} classical={c[2]:.12g}")
    print(f"three-jump overlap T0 quantum={q_triple:.12g} classical={c_triple:.12g}")
    print(f"individual tau^2 coefficients quantum={q_quadratic:.12g} classical={c_quadratic:.12g}")
    print(f"predicted coefficient of sigma_q-sigma_c at tau^2: {predicted_difference_quadratic:.12g}")

    # The continuous-time renewal equation avoids ambiguity from assigning a
    # physical dead time to a finite number of discrete bins.
    continuous_taus = np.array([0.0, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2])
    continuous_rows = []
    for tau in continuous_taus:
        quantum_rate = continuous_deadtime_rate(
            quantum_l, quantum_jumps, vec(np.eye(N)), float(tau)
        )
        classical_rate = continuous_deadtime_rate(
            classical_l, classical_jumps, np.ones(N), float(tau)
        )
        difference = quantum_rate - classical_rate
        continuous_rows.append((tau, quantum_rate, classical_rate, difference))
    fit_rows_continuous = [row for row in continuous_rows if 0 < row[0] <= 2e-3]
    x_continuous = np.array([row[0] for row in fit_rows_continuous])
    y_continuous = np.array([row[3] for row in fit_rows_continuous])
    design_continuous = np.column_stack((x_continuous**2, x_continuous**3))
    fit_continuous = np.linalg.lstsq(design_continuous, y_continuous, rcond=None)[0]
    print(
        "continuous exact fit difference = "
        f"{fit_continuous[0]:.12g} tau^2 + {fit_continuous[1]:.12g} tau^3"
    )
    print("continuous tau difference difference/tau^2")
    for tau, _, _, difference in continuous_rows:
        scaled = difference / tau**2 if tau else np.nan
        print(f"{tau:12.5g} {difference:16.9g} {scaled:16.9g}")

    dt = 0.001
    dead_steps_values = (0, 1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50)
    rows = []
    for dead_steps in dead_steps_values:
        quantum = quantum_deadtime_rates(omega, dt, dead_steps)
        classical = classical_exact_rates(omega, dt, dead_steps)
        tau = dead_steps * dt
        difference = quantum["sigma_visible_clicks_exact"] - classical["sigma_visible_classical"]
        rows.append(
            {
                "dt": dt,
                "dead_steps": dead_steps,
                "dead_time": tau,
                "sigma_quantum": quantum["sigma_visible_clicks_exact"],
                "sigma_classical": classical["sigma_visible_classical"],
                "quantum_minus_classical": difference,
                "difference_over_tau2": difference / tau**2 if tau else np.nan,
            }
        )
    fit_rows = [row for row in rows if 0 < row["dead_time"] <= 0.02]
    x = np.array([row["dead_time"] for row in fit_rows])
    y = np.array([row["quantum_minus_classical"] for row in fit_rows])
    design = np.column_stack((x**2, x**3))
    fit = np.linalg.lstsq(design, y, rcond=None)[0]
    print(f"exact-scan fit difference = {fit[0]:.12g} tau^2 + {fit[1]:.12g} tau^3")
    print("tau difference difference/tau^2")
    for row in rows:
        print(
            f"{row['dead_time']:7.4f} {row['quantum_minus_classical']:13.7g} "
            f"{row['difference_over_tau2']:13.7g}"
        )

    output = Path("pair_correlation_expansion_scan.csv")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
