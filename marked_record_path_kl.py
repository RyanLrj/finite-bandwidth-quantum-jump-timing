"""Time-reversal KL rate of dead-time-filtered marked jump records.

Compares the coherently driven qutrit with its population- and current-matched
classical model.  Record reversal reverses temporal order and exchanges each
thermal emission label with its absorption partner.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from deadtime_exact_rates import stationary_augmented as quantum_stationary_augmented
from matched_classical_deadtime import matched_model, stationary_augmented as classical_stationary_augmented
from passive_deadtime_quantum_filter import apply, make_instrument


State = np.ndarray
Map = Callable[[State], State]


@dataclass(frozen=True)
class RecordModel:
    hidden_live_maps: tuple[Map, ...]
    thermal_maps: tuple[Map, ...]
    full_maps: tuple[Map, ...]
    trace: Callable[[State], float]
    stationary_blocks: tuple[State, ...]


def add_to(blocks: dict[int, State], key: int, value: State) -> None:
    if key in blocks:
        blocks[key] = blocks[key] + value
    else:
        blocks[key] = value.copy()


def quantum_model(omega: float, dt: float, dead_steps: int) -> RecordModel:
    inst = make_instrument(omega, dt)
    stationary, _, _ = quantum_stationary_augmented(omega, dt, dead_steps)
    no_map = lambda rho: apply(inst.no_jump, rho)
    thermal = tuple((lambda jump: (lambda rho: apply(jump, rho)))(jump) for jump in inst.jumps)
    return RecordModel(
        hidden_live_maps=(no_map,),
        thermal_maps=thermal,
        full_maps=(no_map,) + thermal,
        trace=lambda rho: float(np.trace(rho).real),
        stationary_blocks=tuple(stationary),
    )


def classical_model(omega: float, dt: float, dead_steps: int) -> RecordModel:
    no_event, thermal_matrices, work_matrices, _, _ = matched_model(omega, dt)
    stationary, _, _ = classical_stationary_augmented(omega, dt, dead_steps)

    def wrap(matrix: np.ndarray) -> Map:
        return lambda probability: matrix @ probability

    hidden = tuple(wrap(matrix) for matrix in (no_event, *work_matrices))
    thermal = tuple(wrap(matrix) for matrix in thermal_matrices)
    return RecordModel(
        hidden_live_maps=hidden,
        thermal_maps=thermal,
        full_maps=hidden + thermal,
        trace=lambda probability: float(probability.sum()),
        stationary_blocks=tuple(stationary),
    )


def initial_filter(model: RecordModel) -> dict[int, State]:
    return {index: block.copy() for index, block in enumerate(model.stationary_blocks) if model.trace(block) > 1e-16}


def symbol_probabilities(blocks: dict[int, State], model: RecordModel) -> np.ndarray:
    probs = np.zeros(5, dtype=float)
    for remaining, state in blocks.items():
        if remaining == 0:
            probs[0] += sum(model.trace(mapping(state)) for mapping in model.hidden_live_maps)
            for channel, mapping in enumerate(model.thermal_maps):
                probs[channel + 1] += model.trace(mapping(state))
        else:
            probs[0] += sum(model.trace(mapping(state)) for mapping in model.full_maps)
    probs = np.maximum(probs, 0.0)
    return probs / probs.sum()


def filter_symbol(
    blocks: dict[int, State],
    model: RecordModel,
    symbol: int,
    dead_steps: int,
) -> tuple[dict[int, State], float]:
    new: dict[int, State] = {}
    for remaining, state in blocks.items():
        if remaining == 0:
            if symbol == 0:
                for mapping in model.hidden_live_maps:
                    add_to(new, 0, mapping(state))
            else:
                add_to(new, dead_steps, model.thermal_maps[symbol - 1](state))
        elif symbol == 0:
            for mapping in model.full_maps:
                add_to(new, remaining - 1, mapping(state))
    likelihood = sum(model.trace(state) for state in new.values())
    if likelihood <= 1e-300:
        return {}, 0.0
    return {remaining: state / likelihood for remaining, state in new.items()}, likelihood


def sample_record(
    model: RecordModel,
    dead_steps: int,
    steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    blocks = initial_filter(model)
    record = np.empty(steps, dtype=np.int8)
    log_likelihood = 0.0
    for index in range(steps):
        probs = symbol_probabilities(blocks, model)
        symbol = int(rng.choice(5, p=probs))
        blocks, likelihood = filter_symbol(blocks, model, symbol, dead_steps)
        record[index] = symbol
        log_likelihood += np.log(likelihood)
    return record, float(log_likelihood)


def log_likelihood(record: np.ndarray, model: RecordModel, dead_steps: int) -> float:
    blocks = initial_filter(model)
    total = 0.0
    for symbol_value in record:
        blocks, likelihood = filter_symbol(blocks, model, int(symbol_value), dead_steps)
        if likelihood == 0.0:
            return -np.inf
        total += np.log(likelihood)
    return float(total)


def reverse_record(record: np.ndarray) -> np.ndarray:
    # 0=no click; 1<->2 cold emission/absorption; 3<->4 hot emission/absorption.
    conjugate = np.array([0, 2, 1, 4, 3], dtype=np.int8)
    return conjugate[record[::-1]]


def estimate_model(
    kind: str,
    model: RecordModel,
    omega: float,
    dt: float,
    dead_steps: int,
    steps: int,
    seed: int,
) -> dict[str, float | str]:
    record, forward = sample_record(model, dead_steps, steps, np.random.default_rng(seed))
    backward = log_likelihood(reverse_record(record), model, dead_steps)
    rate = (forward - backward) / (steps * dt)
    click_rate = float(np.count_nonzero(record) / (steps * dt))
    return {
        "kind": kind,
        "omega": omega,
        "dt": dt,
        "dead_steps": float(dead_steps),
        "dead_time": dead_steps * dt,
        "steps": float(steps),
        "path_kl_rate": float(rate),
        "click_rate": click_rate,
        "forward_log_likelihood": forward,
        "reverse_log_likelihood": backward,
        "seed": float(seed),
    }


def main() -> None:
    omega = 4.0
    dt = 0.005
    steps = 30_000
    replicas = 3
    dead_steps_values = (0, 10, 30, 50, 100)
    rows: list[dict[str, float | str]] = []
    for kind_index, kind in enumerate(("quantum", "classical")):
        for dead_steps in dead_steps_values:
            # Construct the augmented stationary model once per parameter set;
            # Monte Carlo replicas reuse it.
            model = quantum_model(omega, dt, dead_steps) if kind == "quantum" else classical_model(omega, dt, dead_steps)
            for replica in range(replicas):
                seed = 44000 + kind_index * 10000 + dead_steps * 10 + replica
                rows.append(estimate_model(kind, model, omega, dt, dead_steps, steps, seed))

    output = Path("marked_record_path_kl_scan.csv")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("kind dead_time mean_KL stderr click_rate")
    for kind in ("quantum", "classical"):
        for dead_steps in dead_steps_values:
            group = [row for row in rows if row["kind"] == kind and int(float(row["dead_steps"])) == dead_steps]
            values = np.array([float(row["path_kl_rate"]) for row in group])
            clicks = np.array([float(row["click_rate"]) for row in group])
            print(
                f"{kind:9s} {dead_steps*dt:9.4g} {values.mean():11.6g} "
                f"{values.std(ddof=1)/np.sqrt(replicas):9.3g} {clicks.mean():10.6g}"
            )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
