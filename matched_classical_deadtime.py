"""Classical population model matched to the coherent qutrit steady state.

The coherent 1<->2 coupling is replaced by symmetric incoherent transitions.
Their rate is chosen so the classical model has the same stationary
populations and hence the same mean hot/cold currents as the quantum model.
Both models are then passed through the identical nonparalyzable dead-time
detector.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from deadtime_exact_rates import exact_rates as quantum_exact_rates
from passive_deadtime_quantum_filter import make_instrument, stationary_unmonitored


def matched_model(omega: float, dt: float) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray, float]:
    quantum_inst = make_instrument(omega, dt)
    quantum_rho = stationary_unmonitored(quantum_inst)
    target = np.diag(quantum_rho).real

    cold_down = 1.0
    cold_up = np.exp(-2.0)
    hot_down = 0.8
    hot_up = 0.8 * np.exp(-1.25)
    # Stationarity of level 1 fixes the symmetric work-transition rate.
    work_rate = (target[1] * cold_down - target[0] * cold_up) / (target[2] - target[1])
    if work_rate <= 0:
        raise RuntimeError("matched incoherent work rate is not positive")

    def event(target_state: int, source_state: int, rate: float) -> np.ndarray:
        result = np.zeros((3, 3), dtype=float)
        result[target_state, source_state] = dt * rate
        return result

    thermal = [
        event(0, 1, cold_down),
        event(1, 0, cold_up),
        event(0, 2, hot_down),
        event(2, 0, hot_up),
    ]
    work = [event(2, 1, work_rate), event(1, 2, work_rate)]
    outgoing = sum(thermal + work)
    no_event = np.eye(3) - np.diag(outgoing.sum(axis=0))
    if no_event.diagonal().min() < 0:
        raise RuntimeError("time bin is too large for the classical instrument")
    full = no_event + sum(thermal + work)
    if np.linalg.norm(full.sum(axis=0) - 1.0) > 1e-13:
        raise RuntimeError("classical instrument is not stochastic")
    return no_event, thermal, work, target, float(work_rate)


def stationary_augmented(omega: float, dt: float, dead_steps: int) -> tuple[list[np.ndarray], float, int]:
    no_event, thermal, work, target, work_rate = matched_model(omega, dt)
    blocks = [np.zeros(3, dtype=float) for _ in range(dead_steps + 1)]
    blocks[0] = target.copy()
    hidden_map = no_event + sum(work)
    full_map = hidden_map + sum(thermal)
    for iteration in range(1, 2_000_001):
        new = [np.zeros(3, dtype=float) for _ in range(dead_steps + 1)]
        new[0] += hidden_map @ blocks[0]
        if dead_steps == 0:
            new[0] += sum(channel @ blocks[0] for channel in thermal)
        else:
            new[dead_steps] += sum(channel @ blocks[0] for channel in thermal)
        for remaining in range(1, dead_steps + 1):
            new[remaining - 1] += full_map @ blocks[remaining]
        norm = sum(block.sum() for block in new)
        new = [block / norm for block in new]
        error = sum(np.linalg.norm(a - b) for a, b in zip(new, blocks))
        blocks = new
        if error < 1e-14:
            return blocks, work_rate, iteration
    raise RuntimeError("classical augmented iteration did not converge")


def classical_exact_rates(omega: float, dt: float, dead_steps: int) -> dict[str, float]:
    no_event, thermal, work, target, _ = matched_model(omega, dt)
    blocks, work_rate, iterations = stationary_augmented(omega, dt, dead_steps)
    entropy = np.array([2.0, -2.0, 1.25, -1.25])
    live = blocks[0]
    detected = np.array([(channel @ live).sum() for channel in thermal])
    physical = float(np.array([(channel @ target).sum() for channel in thermal]) @ entropy / dt)
    visible = float(detected @ entropy / dt)
    total = sum(blocks)
    return {
        "omega_drive_matched": omega,
        "dt": dt,
        "dead_steps": float(dead_steps),
        "dead_time": dead_steps * dt,
        "matched_work_rate": work_rate,
        "sigma_physical_classical": physical,
        "sigma_visible_classical": visible,
        "visible_fraction_classical": visible / physical,
        "detected_click_rate_classical": float(detected.sum() / dt),
        "live_probability_classical": float(live.sum()),
        "population_match_error": float(np.linalg.norm(total - target)),
        "iterations": float(iterations),
    }


def main() -> None:
    omega = 4.0
    dt = 0.005
    dead_steps_values = tuple(range(0, 81, 2)) + (100, 150, 200)
    rows: list[dict[str, float]] = []
    for dead_steps in dead_steps_values:
        classical = classical_exact_rates(omega, dt, dead_steps)
        quantum = quantum_exact_rates(omega, dt, dead_steps)
        rows.append(
            classical
            | {
                "sigma_physical_quantum": quantum["sigma_physical"],
                "sigma_visible_quantum": quantum["sigma_visible_clicks_exact"],
                "visible_fraction_quantum": quantum["visible_fraction_exact"],
                "quantum_minus_classical_visible": quantum["sigma_visible_clicks_exact"] - classical["sigma_visible_classical"],
            }
        )

    output = Path("matched_classical_deadtime_scan.csv")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    _, _, _, target, work_rate = matched_model(omega, dt)
    print(f"matched populations: {target}")
    print(f"matched symmetric work rate: {work_rate:.12g}")
    print("dead_time quantum_visible classical_visible difference")
    for row in rows:
        if int(row["dead_steps"]) in (0, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200):
            print(
                f"{row['dead_time']:9.4g} {row['sigma_visible_quantum']:15.8g} "
                f"{row['sigma_visible_classical']:17.8g} "
                f"{row['quantum_minus_classical_visible']:12.5g}"
            )

    for label in ("quantum", "classical"):
        key = f"sigma_visible_{label}"
        crossing = next((row for row in rows if row[key] < 0), None)
        if crossing:
            print(f"first scanned {label} sign reversal at dead_time={crossing['dead_time']:.6g}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
