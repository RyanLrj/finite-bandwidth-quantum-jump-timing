"""Exact finite-horizon KL audit for the classical dead-time HMM.

All allowed observed symbol strings are enumerated, so the result has no Monte
Carlo error.  This is intended to validate the stationary-reversal labels and
the detector-channel data-processing construction before returning to quantum
trajectory simulations.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from marked_record_path_kl import RecordModel, filter_symbol, initial_filter
from petz_deadtime_kl import classical_pair, quantum_pair


def finite_horizon_kl(
    forward: RecordModel,
    reverse: RecordModel,
    dead_steps: int,
    length: int,
) -> tuple[float, float, int]:
    # Entries are (forward filter, reverse filter, forward probability,
    # reverse probability). Filters are normalized after every symbol.
    frontier = [(initial_filter(forward), initial_filter(reverse), 1.0, 1.0)]
    for _ in range(length):
        following = []
        for f_blocks, r_blocks, f_prefix, r_prefix in frontier:
            for symbol in range(5):
                f_new, f_likelihood = filter_symbol(f_blocks, forward, symbol, dead_steps)
                if f_likelihood == 0.0:
                    continue
                r_new, r_likelihood = filter_symbol(r_blocks, reverse, symbol, dead_steps)
                if r_likelihood == 0.0:
                    raise RuntimeError("forward-supported word has zero reverse probability")
                following.append(
                    (f_new, r_new, f_prefix * f_likelihood, r_prefix * r_likelihood)
                )
        frontier = following

    total_p = sum(item[2] for item in frontier)
    total_q = sum(item[3] for item in frontier)
    kl = sum(p * np.log(p / q) for _, _, p, q in frontier)
    return float(kl), float(max(abs(total_p - 1.0), abs(total_q - 1.0))), len(frontier)


def main() -> None:
    omega = 4.0
    dt = 0.02
    lengths = (2, 4, 6, 8, 10, 12)
    dead_steps_values = (0, 1, 2, 3, 5, 10, 20)
    rows: list[dict[str, float]] = []
    for kind, pair_builder in (("quantum", quantum_pair), ("classical", classical_pair)):
        for dead_steps in dead_steps_values:
            forward, reverse = pair_builder(omega, dt, dead_steps)
            for length in lengths:
                kl, normalization_error, words = finite_horizon_kl(
                    forward, reverse, dead_steps, length
                )
                rows.append(
                    {
                        "kind": kind,
                        "dt": dt,
                        "dead_steps": float(dead_steps),
                        "dead_time": dead_steps * dt,
                        "length": float(length),
                        "duration": length * dt,
                        "kl": kl,
                        "kl_rate": kl / (length * dt),
                        "normalization_error": normalization_error,
                        "supported_words": float(words),
                    }
                )

    output = Path("exact_finite_horizon_kl.csv")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("kind dead_steps length KL KL_rate normalization_error words")
    for row in rows:
        if int(row["length"]) in (4, 8, 12):
            print(
                f"{str(row['kind']):9s} {int(row['dead_steps']):10d} {int(row['length']):6d} "
                f"{row['kl']:11.7g} {row['kl_rate']:11.7g} "
                f"{row['normalization_error']:.2e} {int(row['supported_words']):7d}"
            )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
