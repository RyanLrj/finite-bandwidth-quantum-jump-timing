"""Exact stationary visible entropy rates for a passive dead-time detector."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from passive_deadtime_quantum_filter import N, apply, make_instrument, stationary_unmonitored, theoretical_entropy_rate


def stationary_augmented(omega: float, dt: float, dead_steps: int) -> tuple[list[np.ndarray], int, float]:
    inst = make_instrument(omega, dt)
    steady = stationary_unmonitored(inst)
    blocks = [np.zeros((N, N), dtype=complex) for _ in range(dead_steps + 1)]
    blocks[0] = steady

    for iteration in range(1, 2_000_001):
        new = [np.zeros((N, N), dtype=complex) for _ in range(dead_steps + 1)]
        # Live: no event remains live; any detected jump starts dead time.
        new[0] += apply(inst.no_jump, blocks[0])
        if dead_steps == 0:
            for jump in inst.jumps:
                new[0] += apply(jump, blocks[0])
        else:
            for jump in inst.jumps:
                new[dead_steps] += apply(jump, blocks[0])

        # Refractory: all physical alternatives are hidden and countdown falls.
        for remaining in range(1, dead_steps + 1):
            target = remaining - 1
            new[target] += apply(inst.no_jump, blocks[remaining])
            for jump in inst.jumps:
                new[target] += apply(jump, blocks[remaining])

        norm = sum(np.trace(block).real for block in new)
        new = [block / norm for block in new]
        error = sum(np.linalg.norm(a - b) for a, b in zip(new, blocks))
        blocks = new
        if error < 1e-14:
            return blocks, iteration, error
    raise RuntimeError("augmented stationary iteration did not converge")


def exact_rates(omega: float, dt: float, dead_steps: int) -> dict[str, float]:
    inst = make_instrument(omega, dt)
    blocks, iterations, residual = stationary_augmented(omega, dt, dead_steps)
    live = blocks[0]
    detected_probs = np.array([np.trace(apply(jump, live)).real for jump in inst.jumps])
    rho_total = sum(blocks)
    physical_from_total = theoretical_entropy_rate(inst, rho_total, dt)
    physical_unmonitored = theoretical_entropy_rate(inst, stationary_unmonitored(inst), dt)
    visible = float(detected_probs @ inst.entropy / dt)
    click_rate = float(detected_probs.sum() / dt)
    return {
        "omega_drive": omega,
        "dt": dt,
        "dead_steps": float(dead_steps),
        "dead_time": dead_steps * dt,
        "sigma_physical": physical_unmonitored,
        "sigma_physical_augmented": physical_from_total,
        "sigma_visible_clicks_exact": visible,
        "visible_fraction_exact": visible / physical_unmonitored,
        "detected_click_rate_exact": click_rate,
        "live_probability": float(np.trace(live).real),
        "iterations": float(iterations),
        "iteration_residual": residual,
        "reduced_state_error": float(np.linalg.norm(rho_total - stationary_unmonitored(inst))),
    }


def main() -> None:
    omega = 4.0
    dt = 0.005
    dead_steps_values = (0, 1, 2, 5, 10, 20, 50, 100, 200)
    rows = [exact_rates(omega, dt, dead_steps) for dead_steps in dead_steps_values]
    output = Path("deadtime_exact_rates.csv")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("dead_time physical visible fraction click_rate live_probability")
    for row in rows:
        print(
            f"{row['dead_time']:9.4g} {row['sigma_physical']:10.7g} "
            f"{row['sigma_visible_clicks_exact']:11.7g} {row['visible_fraction_exact']:9.3%} "
            f"{row['detected_click_rate_exact']:10.6g} {row['live_probability']:10.6g}"
        )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
