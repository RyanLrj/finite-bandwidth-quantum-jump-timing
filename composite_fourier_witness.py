"""Audit one fixed Fourier witness against the nine optimized resource points.

This is a finite candidate-set certificate, not a continuum/class-wide proof.
The phases are fixed from the strongest tested candidate (31 states, A=60)
and then reused without adaptation.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from asymptotic_markov_renewal_kl import fixed_length_kernel, quantum_kernel
from symmetric_emission_frontier import symmetric_classical_model, symmetric_quantum_model


def build_model(row):
    phases = int(row["phases_per_macrostate"])
    rates = np.array([float(value) for value in row["forward_rates"].split()])
    return symmetric_classical_model(
        0.01, phases, np.log(rates), float(row["reverse_fraction"]), 0.9
    )[0]


def pair_moments(q_kernels, c_kernels, stationary, lengths, frequency):
    moments = np.zeros((4, 4), dtype=complex)
    for previous in range(4):
        times = (11 + np.arange(lengths[previous])) * 0.01
        difference = q_kernels[previous] - c_kernels[previous]
        moments[previous] = stationary[previous] * (
            np.exp(1j * frequency * times) @ difference
        )
    return moments


def main():
    frequency = 4.0548
    q_data = quantum_kernel(symmetric_quantum_model(4.0, 0.01, 0.9), 10)
    q_kernels, mean_bins, _, stationary, _, lengths = q_data
    mean_time = float(stationary @ mean_bins) * 0.01
    rows = list(csv.DictReader(Path("symmetric_emission_frontier.csv").open(encoding="utf-8")))
    reference = next(
        row for row in rows
        if int(row["phases_per_macrostate"]) == 15
        and float(row["hidden_cycle_affinity"]) == 60.0
    )
    reference_kernel = fixed_length_kernel(build_model(reference), 10, lengths)
    reference_moments = pair_moments(
        q_kernels, reference_kernel, stationary, lengths, frequency
    )
    phases = np.exp(-1j * np.angle(reference_moments))

    output_rows = []
    for row in rows:
        c_kernels = fixed_length_kernel(build_model(row), 10, lengths)
        moments = pair_moments(q_kernels, c_kernels, stationary, lengths, frequency)
        fixed_gap = float(np.real(np.sum(phases * moments)))
        adaptive_gap = float(np.sum(np.abs(moments)))
        output_rows.append(
            {
                "classical_states": int(row["classical_states"]),
                "hidden_cycle_affinity": float(row["hidden_cycle_affinity"]),
                "fixed_witness_expectation_gap": fixed_gap,
                "fixed_witness_pinsker_rate": fixed_gap**2 / (2 * mean_time),
                "candidate_adaptive_gap": adaptive_gap,
                "candidate_adaptive_pinsker_rate": adaptive_gap**2 / (2 * mean_time),
                "exact_kl_rate": float(row["asymptotic_kl_rate"]),
            }
        )

    output = Path("composite_fourier_witness.csv")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)
    print("minimum fixed gap", min(row["fixed_witness_expectation_gap"] for row in output_rows))
    for row in output_rows:
        print(row)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
