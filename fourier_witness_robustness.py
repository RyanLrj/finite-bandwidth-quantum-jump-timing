"""Robustness audit for the fixed channel-resolved Fourier witness.

The audit separates three possible sources of selectivity:

1. the classical resource point used to set the channel phases;
2. the small offset between the optimized frequency and the physical drive;
3. calibration errors in the frozen channel phases.

All evaluations use the same nine optimized classical kernels reported in the
manuscript.  The calculation is therefore a finite candidate-set audit, not a
certificate over a continuum of hidden Markov models.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from asymptotic_markov_renewal_kl import fixed_length_kernel, quantum_kernel
from composite_fourier_witness import build_model, pair_moments
from symmetric_emission_frontier import symmetric_quantum_model


DT = 0.01
DEAD_STEPS = 10
GAMMA = 0.9
DRIVE_FREQUENCY = 4.0
NOMINAL_FREQUENCY = 4.0548
RANDOM_SEED = 20260907
PHASE_ERROR_SCALES = (0.05, 0.10, 0.20)
PHASE_TRIALS = 100


def identifier(row: dict[str, str]) -> str:
    return f"N{int(row['classical_states'])}_A{int(float(row['hidden_cycle_affinity']))}"


def frozen_phases(moments: np.ndarray) -> np.ndarray:
    return np.exp(-1j * np.angle(moments))


def expectation_gap(phases: np.ndarray, moments: np.ndarray) -> float:
    return float(np.real(np.sum(phases * moments)))


def write_rows(path: str, rows: list[dict[str, object]]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = list(
        csv.DictReader(Path("symmetric_emission_frontier.csv").open(encoding="utf-8"))
    )
    q_data = quantum_kernel(
        symmetric_quantum_model(DRIVE_FREQUENCY, DT, GAMMA), DEAD_STEPS
    )
    q_kernels, mean_bins, _, stationary, _, lengths = q_data
    mean_time = float(stationary @ mean_bins) * DT
    kernels = {
        identifier(row): fixed_length_kernel(build_model(row), DEAD_STEPS, lengths)
        for row in rows
    }

    def moments_for(candidate: str, frequency: float) -> np.ndarray:
        return pair_moments(
            q_kernels, kernels[candidate], stationary, lengths, frequency
        )

    candidate_ids = [identifier(row) for row in rows]
    nominal_moments = {
        candidate: moments_for(candidate, NOMINAL_FREQUENCY)
        for candidate in candidate_ids
    }

    anchor_rows: list[dict[str, object]] = []
    anchor_minima: dict[str, float] = {}
    off_anchor_gaps: list[float] = []
    for anchor in candidate_ids:
        phases = frozen_phases(nominal_moments[anchor])
        anchor_gaps = []
        for candidate in candidate_ids:
            gap = expectation_gap(phases, nominal_moments[candidate])
            anchor_gaps.append(gap)
            if candidate != anchor:
                off_anchor_gaps.append(gap)
            anchor_rows.append(
                {
                    "anchor": anchor,
                    "evaluated_candidate": candidate,
                    "frequency": NOMINAL_FREQUENCY,
                    "expectation_gap": gap,
                    "pinsker_rate": gap**2 / (2.0 * mean_time),
                    "is_anchor": int(candidate == anchor),
                }
            )
        anchor_minima[anchor] = min(anchor_gaps)

    reference = "N31_A60"
    reference_phases = frozen_phases(nominal_moments[reference])
    frequency_rows: list[dict[str, object]] = []
    for frequency in np.arange(3.8, 4.3000001, 0.005):
        gaps = [
            expectation_gap(reference_phases, moments_for(candidate, float(frequency)))
            for candidate in candidate_ids
        ]
        frequency_rows.append(
            {
                "frequency": float(frequency),
                "minimum_gap_over_candidates": min(gaps),
                "maximum_gap_over_candidates": max(gaps),
                "all_nine_positive": int(min(gaps) > 0.0),
            }
        )

    drive_moments = {
        candidate: moments_for(candidate, DRIVE_FREQUENCY)
        for candidate in candidate_ids
    }
    drive_phases = frozen_phases(drive_moments[reference])
    drive_gaps = [
        expectation_gap(drive_phases, drive_moments[candidate])
        for candidate in candidate_ids
    ]

    rng = np.random.default_rng(RANDOM_SEED)
    phase_rows: list[dict[str, object]] = []
    for scale in PHASE_ERROR_SCALES:
        for trial in range(1, PHASE_TRIALS + 1):
            errors = rng.normal(0.0, scale, size=reference_phases.shape)
            perturbed = reference_phases * np.exp(1j * errors)
            gaps = [
                expectation_gap(perturbed, nominal_moments[candidate])
                for candidate in candidate_ids
            ]
            phase_rows.append(
                {
                    "random_seed": RANDOM_SEED,
                    "phase_error_sd_radians": scale,
                    "trial": trial,
                    "minimum_gap_over_candidates": min(gaps),
                    "all_nine_positive": int(min(gaps) > 0.0),
                }
            )

    positive_frequencies = [
        row["frequency"] for row in frequency_rows if row["all_nine_positive"]
    ]
    summary_rows: list[dict[str, object]] = [
        {
            "quantity": "minimum_gap_over_all_anchor_candidate_pairs",
            "value": min(row["expectation_gap"] for row in anchor_rows),
            "detail": "9 anchors x 9 evaluated candidates at omega=4.0548",
        },
        {
            "quantity": "minimum_leave_anchor_out_gap",
            "value": min(off_anchor_gaps),
            "detail": "anchor candidate excluded from each test set",
        },
        {
            "quantity": "smallest_anchor_worst_case_gap",
            "value": min(anchor_minima.values()),
            "detail": "minimum across the nine anchor-specific worst cases",
        },
        {
            "quantity": "largest_anchor_worst_case_gap",
            "value": max(anchor_minima.values()),
            "detail": "maximum across the nine anchor-specific worst cases",
        },
        {
            "quantity": "minimum_gap_at_physical_drive_frequency",
            "value": min(drive_gaps),
            "detail": "N31_A60 phases recomputed at omega=Omega=4",
        },
        {
            "quantity": "positive_frequency_interval_lower",
            "value": min(positive_frequencies) if positive_frequencies else float("nan"),
            "detail": "scan step 0.005 with N31_A60 phases frozen at omega=4.0548",
        },
        {
            "quantity": "positive_frequency_interval_upper",
            "value": max(positive_frequencies) if positive_frequencies else float("nan"),
            "detail": "scan step 0.005 with N31_A60 phases frozen at omega=4.0548",
        },
        {
            "quantity": "minimum_gap_over_frequency_scan",
            "value": min(row["minimum_gap_over_candidates"] for row in frequency_rows),
            "detail": "worst case over nine candidates and omega in [3.8,4.3]",
        },
    ]
    for scale in PHASE_ERROR_SCALES:
        subset = [
            row
            for row in phase_rows
            if row["phase_error_sd_radians"] == scale
        ]
        summary_rows.append(
            {
                "quantity": f"minimum_gap_phase_error_sd_{scale:.2f}",
                "value": min(row["minimum_gap_over_candidates"] for row in subset),
                "detail": f"{PHASE_TRIALS} fixed-seed trials",
            }
        )

    write_rows("fourier_witness_anchor_audit.csv", anchor_rows)
    write_rows("fourier_witness_frequency_audit.csv", frequency_rows)
    write_rows("fourier_witness_phase_audit.csv", phase_rows)
    write_rows("fourier_witness_robustness_summary.csv", summary_rows)
    for row in summary_rows:
        print(f"{row['quantity']}: {row['value']} ({row['detail']})")


if __name__ == "__main__":
    main()
