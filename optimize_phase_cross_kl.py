"""Directly minimize exact finite-window KL over a 5-state phase model."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from classical_memory_cross_kl import phase_record_model, quantum_split_model
from exact_finite_horizon_kl import finite_horizon_kl


def nelder_mead(objective, start: np.ndarray, max_iterations: int = 80):
    dimension = len(start)
    simplex = [start.copy()]
    for index in range(dimension):
        point = start.copy()
        point[index] += 0.35
        simplex.append(point)
    values = [objective(point) for point in simplex]

    for iteration in range(max_iterations):
        order = np.argsort(values)
        simplex = [simplex[index] for index in order]
        values = [values[index] for index in order]
        if np.std(values) < 1e-13 and max(
            np.linalg.norm(point - simplex[0]) for point in simplex[1:]
        ) < 2e-5:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        reflected = centroid + (centroid - simplex[-1])
        reflected_value = objective(reflected)
        if values[0] <= reflected_value < values[-2]:
            simplex[-1], values[-1] = reflected, reflected_value
            continue
        if reflected_value < values[0]:
            expanded = centroid + 2.0 * (reflected - centroid)
            expanded_value = objective(expanded)
            if expanded_value < reflected_value:
                simplex[-1], values[-1] = expanded, expanded_value
            else:
                simplex[-1], values[-1] = reflected, reflected_value
            continue
        contracted = centroid + 0.5 * (simplex[-1] - centroid)
        contracted_value = objective(contracted)
        if contracted_value < values[-1]:
            simplex[-1], values[-1] = contracted, contracted_value
            continue
        best = simplex[0]
        simplex = [best] + [best + 0.5 * (point - best) for point in simplex[1:]]
        values = [values[0]] + [objective(point) for point in simplex[1:]]

    order = np.argsort(values)
    best_index = int(order[0])
    return simplex[best_index], values[best_index], iteration + 1


def optimize_case(dead_steps: int, length: int, dt: float = 0.02):
    quantum, _ = quantum_split_model(4.0, dt, dead_steps)
    lower, upper = np.log(1e-3), np.log(100.0)
    cache: dict[tuple[float, ...], float] = {}

    def objective(log_rates: np.ndarray) -> float:
        clipped = np.clip(log_rates, lower, upper)
        key = tuple(np.round(clipped, 10))
        if key in cache:
            return cache[key]
        rates = tuple(float(value) for value in np.exp(clipped))
        candidate = phase_record_model(dt, dead_steps, rates)
        kl, _, _ = finite_horizon_kl(quantum, candidate, dead_steps, length)
        cache[key] = kl
        return kl

    starts = (
        np.log(np.array([8.15514029, 0.11191284, 0.46783069])),
        np.log(np.array([15.37599448, 15.37599448, 15.37599448])),
    )
    answers = []
    for start_index, start in enumerate(starts, start=1):
        answer, value, iterations = nelder_mead(objective, start)
        answers.append((value, np.exp(np.clip(answer, lower, upper)), iterations))
        print(
            f"  completed start {start_index}/{len(starts)} for D={dead_steps}, n={length}: "
            f"KL={value:.9g}",
            flush=True,
        )
    answers.sort(key=lambda item: item[0])
    refined_point, refined_value, refined_iterations = nelder_mead(
        objective, np.log(answers[0][1]), max_iterations=120
    )
    answers.append(
        (
            refined_value,
            np.exp(np.clip(refined_point, lower, upper)),
            refined_iterations,
        )
    )
    print(
        f"  refinement for D={dead_steps}, n={length}: KL={refined_value:.9g}",
        flush=True,
    )
    answers.sort(key=lambda item: item[0])
    best = answers[0]
    return best, answers, len(cache)


def main() -> None:
    dt = 0.02
    cases = ((5, 12), (10, 20))
    rows = []
    for dead_steps, length in cases:
        best, answers, evaluations = optimize_case(dead_steps, length, dt)
        print(
            f"D={dead_steps} n={length} best_KL={best[0]:.12g} "
            f"rate={best[0]/(length*dt):.12g} rates={best[1]} "
            f"iterations={best[2]} evaluations={evaluations}"
        )
        for rank, answer in enumerate(answers, start=1):
            print(f"  start-rank {rank}: KL={answer[0]:.12g} rates={answer[1]}")
        rows.append(
            {
                "dt": dt,
                "dead_steps": dead_steps,
                "dead_time": dead_steps * dt,
                "length": length,
                "duration": length * dt,
                "optimized_kl": best[0],
                "optimized_kl_rate": best[0] / (length * dt),
                "first_stage_rate": best[1][0],
                "cold_to_hot_rate": best[1][1],
                "hot_to_cold_rate": best[1][2],
                "objective_evaluations": evaluations,
            }
        )

    output = Path("optimized_phase_cross_kl.csv")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
