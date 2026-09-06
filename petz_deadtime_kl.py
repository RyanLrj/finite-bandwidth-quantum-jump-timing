"""Detector-channel KL contraction using stationary Petz reversal.

The forward and stationary-reversed physical instruments are both passed
through the same causal nonparalyzable dead-time channel.  This is the proper
data-processing comparison; reversing an already filtered record would also
reverse the detector's causal recovery mechanism and can create a spurious
arrow of time.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from marked_record_path_kl import (
    RecordModel,
    filter_symbol,
    initial_filter,
    log_likelihood,
    sample_record,
)
from matched_classical_deadtime import matched_model
from passive_deadtime_quantum_filter import apply, make_instrument, stationary_unmonitored


def positive_sqrt_and_inverse(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh((rho + rho.conj().T) / 2.0)
    if values.min() <= 1e-12:
        raise RuntimeError("stationary state is not full rank")
    sqrt = (vectors * np.sqrt(values)) @ vectors.conj().T
    inverse = (vectors * (1.0 / np.sqrt(values))) @ vectors.conj().T
    return sqrt, inverse


def stationary_blocks_from_maps(
    hidden: tuple,
    thermal: tuple,
    trace,
    initial_state: np.ndarray,
    dead_steps: int,
) -> tuple[np.ndarray, ...]:
    full = hidden + thermal
    blocks = [np.zeros_like(initial_state) for _ in range(dead_steps + 1)]
    blocks[0] = initial_state.copy()
    for _ in range(2_000_000):
        new = [np.zeros_like(initial_state) for _ in range(dead_steps + 1)]
        for mapping in hidden:
            new[0] += mapping(blocks[0])
        if dead_steps == 0:
            for mapping in thermal:
                new[0] += mapping(blocks[0])
        else:
            for mapping in thermal:
                new[dead_steps] += mapping(blocks[0])
        for remaining in range(1, dead_steps + 1):
            for mapping in full:
                new[remaining - 1] += mapping(blocks[remaining])
        norm = sum(trace(block) for block in new)
        new = [block / norm for block in new]
        error = sum(np.linalg.norm(a - b) for a, b in zip(new, blocks))
        blocks = new
        if error < 1e-14:
            return tuple(blocks)
    raise RuntimeError("stationary detector state did not converge")


def quantum_pair(omega: float, dt: float, dead_steps: int) -> tuple[RecordModel, RecordModel]:
    inst = make_instrument(omega, dt)
    rho = stationary_unmonitored(inst)
    sqrt, inverse = positive_sqrt_and_inverse(rho)
    forward_kraus = (inst.no_jump,) + inst.jumps
    reverse_raw = tuple(sqrt @ k.conj().T @ inverse for k in forward_kraus)
    completeness = sum(k.conj().T @ k for k in reverse_raw)
    if np.linalg.norm(completeness - np.eye(3)) > 2e-10:
        raise RuntimeError("Petz reversed quantum instrument is not trace preserving")

    def model(kraus: tuple[np.ndarray, ...]) -> RecordModel:
        maps = tuple((lambda k: (lambda state: apply(k, state)))(k) for k in kraus)
        trace = lambda state: float(np.trace(state).real)
        blocks = stationary_blocks_from_maps((maps[0],), maps[1:], trace, rho, dead_steps)
        return RecordModel((maps[0],), maps[1:], maps, trace, blocks)

    # Raw Petz map of a forward emission is the corresponding backward
    # absorption. Reorder paired channels so both models use the same physical
    # alphabet: cold down/up and hot down/up.
    reverse_labeled = (
        reverse_raw[0],
        reverse_raw[2],
        reverse_raw[1],
        reverse_raw[4],
        reverse_raw[3],
    )
    return model(forward_kraus), model(reverse_labeled)


def classical_pair(omega: float, dt: float, dead_steps: int) -> tuple[RecordModel, RecordModel]:
    no_event, thermal_matrices, work_matrices, stationary, _ = matched_model(omega, dt)
    forward_matrices = (no_event, *work_matrices, *thermal_matrices)
    diagonal = np.diag(stationary)
    inverse = np.diag(1.0 / stationary)
    reverse_raw = tuple(diagonal @ matrix.T @ inverse for matrix in forward_matrices)
    if np.linalg.norm(sum(reverse_raw).sum(axis=0) - 1.0) > 1e-12:
        raise RuntimeError("reversed classical instrument is not stochastic")
    # no event; swap the two hidden work directions; swap each observed
    # thermal emission/absorption pair to restore the common physical labels.
    reverse_matrices = (
        reverse_raw[0],
        reverse_raw[2],
        reverse_raw[1],
        reverse_raw[4],
        reverse_raw[3],
        reverse_raw[6],
        reverse_raw[5],
    )

    def model(matrices: tuple[np.ndarray, ...]) -> RecordModel:
        maps = tuple((lambda matrix: (lambda state: matrix @ state))(matrix) for matrix in matrices)
        hidden = maps[:3]
        thermal = maps[3:]
        trace = lambda state: float(state.sum())
        blocks = stationary_blocks_from_maps(hidden, thermal, trace, stationary, dead_steps)
        return RecordModel(hidden, thermal, maps, trace, blocks)

    return model(forward_matrices), model(reverse_matrices)


def estimate_models(
    kind: str,
    forward_model: RecordModel,
    reverse_model: RecordModel,
    omega: float,
    dt: float,
    dead_steps: int,
    steps: int,
    seed: int,
) -> dict[str, float | str]:
    record, forward_ll = sample_record(forward_model, dead_steps, steps, np.random.default_rng(seed))
    reverse_ll = log_likelihood(record, reverse_model, dead_steps)
    return {
        "kind": kind,
        "omega": omega,
        "dt": dt,
        "dead_steps": float(dead_steps),
        "dead_time": dead_steps * dt,
        "steps": float(steps),
        "petz_output_kl_rate": float((forward_ll - reverse_ll) / (steps * dt)),
        "click_rate": float(np.count_nonzero(record) / (steps * dt)),
        "forward_log_likelihood": forward_ll,
        "reverse_log_likelihood": reverse_ll,
        "seed": float(seed),
    }


def main() -> None:
    omega = 4.0
    dt = 0.005
    steps = 30_000
    replicas = 2
    dead_steps_values = (0, 30, 100)
    rows: list[dict[str, float | str]] = []
    for kind_index, kind in enumerate(("quantum", "classical")):
        for dead_steps in dead_steps_values:
            forward_model, reverse_model = (
                quantum_pair(omega, dt, dead_steps)
                if kind == "quantum"
                else classical_pair(omega, dt, dead_steps)
            )
            for replica in range(replicas):
                rows.append(
                    estimate_models(
                        kind,
                        forward_model,
                        reverse_model,
                        omega,
                        dt,
                        dead_steps,
                        steps,
                        88000 + 10000 * kind_index + 10 * dead_steps + replica,
                    )
                )
    output = Path("petz_deadtime_kl_scan.csv")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("kind dead_time mean_KL stderr click_rate")
    for kind in ("quantum", "classical"):
        for dead_steps in dead_steps_values:
            group = [r for r in rows if r["kind"] == kind and int(float(r["dead_steps"])) == dead_steps]
            values = np.array([float(r["petz_output_kl_rate"]) for r in group])
            clicks = np.array([float(r["click_rate"]) for r in group])
            print(
                f"{kind:9s} {dead_steps*dt:9.4g} {values.mean():11.6g} "
                f"{values.std(ddof=1)/np.sqrt(replicas):9.3g} {clicks.mean():10.6g}"
            )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
