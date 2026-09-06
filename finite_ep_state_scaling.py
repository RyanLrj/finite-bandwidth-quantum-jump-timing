"""Hidden-state scaling of the finite-EP mixed-reset classical model."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from asymptotic_bidirectional_ring_kl import directional_to_ring_order
from asymptotic_finite_ep_frontier import entropy_production, finite_ep_model, optimize_point
from asymptotic_markov_renewal_kl import quantum_kernel
from classical_memory_cross_kl import quantum_split_model


def expand_rates(rates: np.ndarray, old_phases: int, new_phases: int) -> np.ndarray:
    """Interpolate each half-ring in log rate while preserving mean traversal time."""
    halves = []
    old_x = (np.arange(old_phases) + 0.5) / old_phases
    new_x = (np.arange(new_phases) + 0.5) / new_phases
    for half in (rates[:old_phases], rates[old_phases:]):
        interpolated = np.exp(np.interp(new_x, old_x, np.log(half)))
        old_time = float(np.sum(1.0 / half))
        new_time = float(np.sum(1.0 / interpolated))
        interpolated *= new_time / old_time
        halves.append(interpolated)
    return np.concatenate(halves)


def main():
    dt, dead_steps = 0.01, 10
    reverse_fraction, reset_mixing = 0.001, 1.0
    quantum = quantum_split_model(4.0, dt, 0)[0]
    quantum_data = quantum_kernel(quantum, dead_steps)
    directional_rows = {
        int(row["phases_per_macrostate"]): row
        for row in csv.DictReader(
            Path("asymptotic_directional_phase_kl.csv").open(encoding="utf-8")
        )
    }
    known_11 = next(
        row
        for row in csv.DictReader(
            Path("asymptotic_finite_ep_lowdelta11.csv").open(encoding="utf-8")
        )
        if float(row["reverse_fraction"]) == reverse_fraction
    )
    known_11_rates = np.array(
        [float(value) for value in known_11["forward_rates"].split()]
    )

    rows = []
    previous_rates = None
    previous_phases = None
    for phases in (2, 3, 4, 5, 6, 7):
        starts = [np.full(2 * phases, 1.3 * phases)]
        if phases in directional_rows:
            directed = np.array(
                [float(value) for value in directional_rows[phases]["rates"].split()]
            )
            starts.append(directional_to_ring_order(directed, phases))
        if previous_rates is not None:
            starts.append(expand_rates(previous_rates, previous_phases, phases))
        if phases == 5:
            starts.append(known_11_rates)
        kl_rate, rates, iterations, evaluations = optimize_point(
            dt,
            dead_steps,
            phases,
            reverse_fraction,
            reset_mixing,
            quantum_data,
            tuple(starts),
        )
        previous_rates, previous_phases = rates, phases
        model_data = finite_ep_model(
            dt,
            phases,
            np.log(rates),
            reverse_fraction,
            reset_mixing,
        )
        ep = entropy_production(*model_data[1:])
        rows.append(
            {
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
        )
        print(
            f"phases={phases} states={1+2*phases} KL={kl_rate:.12g} "
            f"EP={ep['total_ep_rate']:.9g}",
            flush=True,
        )

    output = Path("finite_ep_state_scaling.csv")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
