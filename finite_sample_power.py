"""Finite-sample discrimination scales for the equal-loss benchmark.

The log-likelihood calculation treats complete marked renewal cycles.  Its
scaled cumulant generating function is the Perron root of the tilted embedded
transition matrix.  The Hoeffding calculation uses the bounded Fourier witness
reported in the eighteenth-round analysis.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from asymptotic_markov_renewal_kl import fixed_length_kernel, quantum_kernel
from symmetric_emission_frontier import symmetric_classical_model, symmetric_quantum_model


def load_candidate():
    rows = list(csv.DictReader(Path("symmetric_emission_frontier.csv").open(encoding="utf-8")))
    row = next(
        item for item in rows
        if float(item["hidden_cycle_affinity"]) == 60.0
        and int(item["phases_per_macrostate"]) == 15
    )
    rates = np.array([float(value) for value in row["forward_rates"].split()])
    return symmetric_classical_model(
        0.01, 15, np.log(rates), float(row["reverse_fraction"]), 0.9
    )[0]


def scgf(s, q_kernels, c_kernels):
    tilted = np.zeros((4, 4), dtype=float)
    for previous in range(4):
        q = q_kernels[previous]
        c = c_kernels[previous]
        log_ratio = np.log(q / c)
        tilted[previous] = np.sum(q * np.exp(s * log_ratio), axis=0)
    values = np.linalg.eigvals(tilted)
    root = values[int(np.argmax(np.real(values)))]
    return float(np.log(root).real)


def main():
    dt, dead_steps = 0.01, 10
    quantum = symmetric_quantum_model(4.0, dt, 0.9)
    q_data = quantum_kernel(quantum, dead_steps)
    q_kernels, mean_bins, _, stationary, _, lengths = q_data
    classical = load_candidate()
    c_kernels = fixed_length_kernel(classical, dead_steps, lengths)

    mean_time = float(stationary @ mean_bins) * dt
    d_cycle = 0.0
    independent_second = 0.0
    for previous in range(4):
        q = q_kernels[previous]
        c = c_kernels[previous]
        ell = np.log(q / c)
        d_cycle += stationary[previous] * float(np.sum(q * ell))
        independent_second += stationary[previous] * float(np.sum(q * ell**2))
    independent_variance = independent_second - d_cycle**2

    # Five-point derivatives of the exact Markov-additive SCGF.
    h = 2.0e-4
    km2, km1 = scgf(-2 * h, q_kernels, c_kernels), scgf(-h, q_kernels, c_kernels)
    kp1, kp2 = scgf(h, q_kernels, c_kernels), scgf(2 * h, q_kernels, c_kernels)
    mean_scgf = (km2 - 8 * km1 + 8 * kp1 - kp2) / (12 * h)
    var_scgf = (-kp2 + 16 * kp1 - 30 * scgf(0.0, q_kernels, c_kernels) + 16 * km1 - km2) / (12 * h**2)

    summary = list(csv.DictReader(Path("symmetric_emission_analysis_summary.csv").open(encoding="utf-8")))[0]
    delta = float(summary["best_channel_resolved_fourier_gap"])
    rows = []
    for sigma, two_sided_alpha in ((3.0, 2.699796e-3), (5.0, 5.733031e-7)):
        # Gaussian expected-LLR separation under Q. This is an asymptotic power scale,
        # not a non-asymptotic guarantee.
        n_gaussian = sigma**2 * var_scgf / d_cycle**2
        # Distribution-free bound for a [-1,1] witness and a midpoint threshold.
        n_hoeffding = 8.0 * np.log(2.0 / two_sided_alpha) / delta**2
        rows.append(
            {
                "sigma_level": sigma,
                "two_sided_alpha": two_sided_alpha,
                "gaussian_llr_cycles": n_gaussian,
                "gaussian_llr_time": n_gaussian * mean_time,
                "hoeffding_fourier_cycles": n_hoeffding,
                "hoeffding_fourier_time": n_hoeffding * mean_time,
            }
        )

    for beta in (0.1, 0.05, 0.01, 0.001):
        # Stein exponent benchmark for fixed type-I error in the asymptotic regime.
        rows.append(
            {
                "sigma_level": "Stein",
                "two_sided_alpha": beta,
                "gaussian_llr_cycles": np.log(1.0 / beta) / d_cycle,
                "gaussian_llr_time": np.log(1.0 / beta) / (d_cycle / mean_time),
                "hoeffding_fourier_cycles": "",
                "hoeffding_fourier_time": "",
            }
        )

    output = Path("finite_sample_power.csv")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    diagnostics = {
        "mean_cycle_time": mean_time,
        "kl_per_cycle_direct": d_cycle,
        "kl_rate": d_cycle / mean_time,
        "scgf_first_derivative": mean_scgf,
        "scgf_asymptotic_variance_per_cycle": var_scgf,
        "independent_cycle_variance": independent_variance,
        "markov_correlation_variance_correction": var_scgf - independent_variance,
        "fourier_expectation_gap": delta,
    }
    diagnostic_path = Path("finite_sample_diagnostics.csv")
    with diagnostic_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=diagnostics.keys())
        writer.writeheader()
        writer.writerow(diagnostics)
    print(diagnostics)
    for row in rows:
        print(row)
    print(f"wrote {output} and {diagnostic_path}")


if __name__ == "__main__":
    main()
