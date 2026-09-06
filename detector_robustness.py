"""Cycle-marginal robustness under explicit detector post-processing channels.

These channels act on complete marked cycles: Gaussian timestamp jitter,
symmetric label confusion, common background contamination, and heralded
cycle retention.  They do not model unheralded missed jumps, which destroy the
renewal segmentation and require a separate hidden-record likelihood.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from asymptotic_markov_renewal_kl import fixed_length_kernel, quantum_kernel
from finite_sample_power import load_candidate
from symmetric_emission_frontier import symmetric_quantum_model


DT = 0.01
DEAD_STEPS = 10


def padded_joint():
    q_data = quantum_kernel(symmetric_quantum_model(4.0, DT, 0.9), DEAD_STEPS)
    q_kernels, mean_bins, _, stationary, _, lengths = q_data
    c_kernels = fixed_length_kernel(load_candidate(), DEAD_STEPS, lengths)
    width = max(lengths)
    q = np.zeros((4, 4, width))
    c = np.zeros_like(q)
    for previous in range(4):
        count = lengths[previous]
        q[previous, :, :count] = stationary[previous] * q_kernels[previous].T
        c[previous, :, :count] = stationary[previous] * c_kernels[previous].T
    q /= q.sum()
    c /= c.sum()
    mean_time = float(stationary @ mean_bins) * DT
    return q, c, mean_time


def gaussian_jitter(distribution, sigma_time):
    if sigma_time <= 0:
        return distribution.copy()
    sigma = sigma_time / DT
    radius = max(1, int(np.ceil(5.0 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    answer = np.zeros_like(distribution)
    for previous in range(4):
        for following in range(4):
            answer[previous, following] = np.convolve(
                distribution[previous, following], kernel, mode="same"
            )
    return answer / answer.sum()


def confuse_labels(distribution, error_probability):
    if error_probability <= 0:
        return distribution.copy()
    channel = np.full((4, 4), error_probability / 3.0)
    np.fill_diagonal(channel, 1.0 - error_probability)
    # channel[observed, true], independently on the two endpoint labels.
    answer = np.einsum("av,bw,vwt->abt", channel, channel, distribution)
    return answer / answer.sum()


def apply_channel(q, c, sigma_time=0.0, label_error=0.0, contamination=0.0):
    qn = confuse_labels(gaussian_jitter(q, sigma_time), label_error)
    cn = confuse_labels(gaussian_jitter(c, sigma_time), label_error)
    if contamination > 0:
        common = 0.5 * (qn + cn)
        qn = (1.0 - contamination) * qn + contamination * common
        cn = (1.0 - contamination) * cn + contamination * common
    return qn / qn.sum(), cn / cn.sum()


def metrics(q, c, mean_time, retention=1.0):
    supported = q > 0
    kl_cycle = float(np.sum(q[supported] * np.log(q[supported] / c[supported])))
    tv = 0.5 * float(np.sum(np.abs(q - c)))
    times = (DEAD_STEPS + 1 + np.arange(q.shape[-1])) * DT
    best_gap, best_frequency = -1.0, 0.0
    for frequency in np.linspace(3.5, 4.6, 221):
        moments = np.sum((q - c) * np.exp(1j * frequency * times), axis=2)
        gap = float(np.sum(np.abs(moments)))
        if gap > best_gap:
            best_gap, best_frequency = gap, float(frequency)
    return {
        "kl_per_retained_cycle": kl_cycle,
        "information_rate_per_original_time": retention * kl_cycle / mean_time,
        "total_variation": tv,
        "best_fourier_frequency": best_frequency,
        "best_fourier_gap": best_gap,
        "fourier_pinsker_rate_per_original_time": retention * best_gap**2 / (2 * mean_time),
    }


def main():
    q, c, mean_time = padded_joint()
    rows = []

    def add(name, sigma_time, label_error, contamination, retention):
        qn, cn = apply_channel(q, c, sigma_time, label_error, contamination)
        rows.append(
            {
                "scenario": name,
                "timestamp_jitter": sigma_time,
                "label_error_probability": label_error,
                "common_contamination": contamination,
                "heralded_cycle_retention": retention,
                **metrics(qn, cn, mean_time, retention),
            }
        )

    add("ideal", 0.0, 0.0, 0.0, 1.0)
    add("mild_combined", 0.01, 0.01, 0.01, 0.9)
    add("moderate_combined", 0.03, 0.05, 0.05, 0.7)
    add("severe_combined", 0.08, 0.10, 0.10, 0.5)
    for value in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
        add("jitter_sweep", value, 0.0, 0.0, 1.0)
    for value in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3):
        add("label_error_sweep", 0.0, value, 0.0, 1.0)
    for value in (0.01, 0.02, 0.05, 0.1, 0.2, 0.4):
        add("contamination_sweep", 0.0, 0.0, value, 1.0)
    for value in (0.9, 0.8, 0.7, 0.5, 0.3):
        add("retention_sweep", 0.0, 0.0, 0.0, value)

    output = Path("detector_robustness.csv")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows[:4]:
        print(row)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
