"""Spectral, KL, and Fourier-moment analysis of the equal-loss benchmark."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from asymptotic_markov_renewal_kl import cross_kl_rate, fixed_length_kernel, quantum_kernel
from symmetric_emission_frontier import symmetric_classical_model, symmetric_quantum_model


LABELS = ("cold_down", "cold_up", "hot_down", "hot_up")


def target_mode(matrix, frequency=4.0):
    values = np.linalg.eigvals(matrix)
    positive = values[np.imag(values) > 1e-8]
    return positive[int(np.argmin(np.abs(np.imag(positive) - frequency)))]


def main():
    omega, dt, dead_steps, gamma = 4.0, 0.01, 10, 0.9
    rows = list(
        csv.DictReader(Path("symmetric_emission_frontier.csv").open(encoding="utf-8"))
    )
    quantum = symmetric_quantum_model(omega, dt, gamma)
    q_data = quantum_kernel(quantum, dead_steps)
    spectral_rows = []
    models = {}
    for row in rows:
        phases = int(row["phases_per_macrostate"])
        affinity = float(row["hidden_cycle_affinity"])
        reverse_fraction = float(row["reverse_fraction"])
        rates = np.array([float(value) for value in row["forward_rates"].split()])
        model_data = symmetric_classical_model(
            dt, phases, np.log(rates), reverse_fraction, gamma
        )
        models[(affinity, phases)] = model_data[0]
        ring = model_data[1][1:, 1:]
        killed = ring - gamma * np.eye(2 * phases)
        ring_mode = target_mode(ring, omega)
        killed_mode = target_mode(killed, omega)
        bound = (
            1.0 / np.tan(np.pi / (2 * phases))
            * np.tanh(affinity / (4 * phases))
        )
        spectral_rows.append(
            {
                "hidden_cycle_affinity": affinity,
                "phases_per_macrostate": phases,
                "ring_bound": bound,
                "ring_mode_real": float(np.real(ring_mode)),
                "ring_mode_imaginary": float(np.imag(ring_mode)),
                "ring_coherence_ratio": float(np.imag(ring_mode) / -np.real(ring_mode)),
                "bound_saturation": float(
                    (np.imag(ring_mode) / -np.real(ring_mode)) / bound
                ),
                "killed_mode_real": float(np.real(killed_mode)),
                "killed_mode_imaginary": float(np.imag(killed_mode)),
                "killed_coherence_ratio": float(
                    np.imag(killed_mode) / -np.real(killed_mode)
                ),
                "real_shift_error": float(
                    np.real(killed_mode) - (np.real(ring_mode) - gamma)
                ),
                "imaginary_shift_error": float(np.imag(killed_mode) - np.imag(ring_mode)),
                "theorem_decay_lower_bound_at_mode_frequency": float(
                    gamma + np.imag(killed_mode) / bound
                ),
                "actual_killed_decay": float(-np.real(killed_mode)),
            }
        )
        print(
            f"A={affinity:g} p={phases} ring={ring_mode:.6g} "
            f"killed={killed_mode:.6g} saturation="
            f"{spectral_rows[-1]['bound_saturation']:.3%}",
            flush=True,
        )

    spectral_path = Path("symmetric_emission_spectral_audit.csv")
    with spectral_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=spectral_rows[0].keys())
        writer.writeheader()
        writer.writerows(spectral_rows)

    # Strongest tested fixed-resource candidate: affinity 60, 31 states.
    classical = models[(60.0, 15)]
    q_kernels, q_mean_bins, q_transition, q_stationary, _, lengths = q_data
    c_data = quantum_kernel(classical, dead_steps)
    c_transition = c_data[2]
    c_kernels = fixed_length_kernel(classical, dead_steps, lengths)
    mean_time = float(q_stationary @ q_mean_bins) * dt
    label_rate = 0.0
    timing_rate = 0.0
    pair_rows = []
    for previous in range(4):
        for following in range(4):
            p_label = q_transition[previous, following]
            c_label = c_transition[previous, following]
            label_piece = p_label * np.log(p_label / c_label)
            q_values = q_kernels[previous][:, following]
            c_values = c_kernels[previous][:, following]
            supported = q_values > 1e-300
            timing_piece = float(
                np.sum(
                    q_values[supported]
                    * np.log(
                        (q_values[supported] / p_label)
                        / (c_values[supported] / c_label)
                    )
                )
            )
            weighted_label = q_stationary[previous] * label_piece / mean_time
            weighted_timing = q_stationary[previous] * timing_piece / mean_time
            label_rate += weighted_label
            timing_rate += weighted_timing
            pair_rows.append(
                {
                    "previous_label": LABELS[previous],
                    "following_label": LABELS[following],
                    "label_kl_rate_contribution": weighted_label,
                    "timing_kl_rate_contribution": weighted_timing,
                    "total_kl_rate_contribution": weighted_label + weighted_timing,
                }
            )

    joint_tv = 0.0
    for previous in range(4):
        joint_tv += 0.5 * q_stationary[previous] * float(
            np.sum(np.abs(q_kernels[previous] - c_kernels[previous]))
        )
    def fourier_gap(frequency):
        gap = 0.0
        for previous in range(4):
            count = lengths[previous]
            times = (dead_steps + 1 + np.arange(count)) * dt
            phase = np.exp(1j * frequency * times)
            difference = q_kernels[previous] - c_kernels[previous]
            moments = phase @ difference
            gap += q_stationary[previous] * float(np.sum(np.abs(moments)))
        return gap

    frequencies = np.linspace(0.0, 8.0, 1601)
    best_frequency = 0.0
    best_gap = -1.0
    for frequency in frequencies:
        gap = fourier_gap(frequency)
        if gap > best_gap:
            best_gap = gap
            best_frequency = float(frequency)
    # Refine the coarse 0.005 grid locally to quote a stable frequency and bound.
    for frequency in np.linspace(
        max(0.0, best_frequency - 0.01), min(8.0, best_frequency + 0.01), 401
    ):
        gap = fourier_gap(frequency)
        if gap > best_gap:
            best_gap = gap
            best_frequency = float(frequency)

    direct_rate = cross_kl_rate(q_data, classical, dead_steps, dt)
    dominant_timing = sum(
        row["timing_kl_rate_contribution"]
        for row in pair_rows
        if row["previous_label"].endswith("_up")
        and row["following_label"].endswith("_down")
    )
    summary = {
        "hidden_cycle_affinity": 60.0,
        "classical_states": 31,
        "mean_quantum_cycle_time": mean_time,
        "label_kl_rate": label_rate,
        "conditional_timing_kl_rate": timing_rate,
        "total_kl_rate": direct_rate,
        "timing_fraction": timing_rate / direct_rate,
        "absorption_emission_timing_rate": dominant_timing,
        "absorption_emission_fraction_of_timing": dominant_timing / timing_rate,
        "joint_cycle_total_variation": joint_tv,
        "pinsker_rate_from_total_variation": 2.0 * joint_tv**2 / mean_time,
        "best_fourier_frequency": best_frequency,
        "best_channel_resolved_fourier_gap": best_gap,
        "pinsker_rate_from_fourier_gap": best_gap**2 / (2.0 * mean_time),
        "fourier_bound_fraction_of_exact_kl": (
            best_gap**2 / (2.0 * mean_time) / direct_rate
        ),
    }
    summary_path = Path("symmetric_emission_analysis_summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)
    pair_path = Path("symmetric_emission_decomposition_pairs.csv")
    with pair_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=pair_rows[0].keys())
        writer.writeheader()
        writer.writerows(pair_rows)
    fourier_rows = []
    for previous in range(4):
        count = lengths[previous]
        times = (dead_steps + 1 + np.arange(count)) * dt
        phase = np.exp(1j * best_frequency * times)
        difference = q_kernels[previous] - c_kernels[previous]
        moments = phase @ difference
        for following in range(4):
            weighted = q_stationary[previous] * moments[following]
            fourier_rows.append(
                {
                    "previous_label": LABELS[previous],
                    "following_label": LABELS[following],
                    "frequency": best_frequency,
                    "weighted_moment_real": float(np.real(weighted)),
                    "weighted_moment_imaginary": float(np.imag(weighted)),
                    "optimal_phase_gap_contribution": float(np.abs(weighted)),
                }
            )
    fourier_path = Path("symmetric_emission_fourier_pairs.csv")
    with fourier_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fourier_rows[0].keys())
        writer.writeheader()
        writer.writerows(fourier_rows)
    print(summary)
    print(
        f"wrote {spectral_path}, {summary_path}, {pair_path}, and {fourier_path}"
    )


if __name__ == "__main__":
    main()
