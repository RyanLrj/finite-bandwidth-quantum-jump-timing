"""Verify the stored headline outputs and public-release file boundary."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def one_row(name: str) -> dict[str, str]:
    with (ROOT / name).open(newline="", encoding="utf-8") as handle:
        return next(csv.DictReader(handle))


def close(actual: float, expected: float, tolerance: float = 5e-10) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{actual} differs from expected {expected}")


def main() -> None:
    summary = one_row("symmetric_emission_analysis_summary.csv")
    diagnostics = one_row("finite_sample_diagnostics.csv")
    frontier = list(
        csv.DictReader(
            (ROOT / "symmetric_emission_frontier.csv").open(
                newline="", encoding="utf-8"
            )
        )
    )
    best = min(frontier, key=lambda row: float(row["asymptotic_kl_rate"]))

    close(float(best["asymptotic_kl_rate"]), 0.0244603020531)
    close(float(best["total_ep_rate"]), 11.1700804549)
    close(float(summary["timing_fraction"]), 0.995350391128)
    close(float(summary["best_channel_resolved_fourier_gap"]), 0.107860938022)
    close(float(summary["pinsker_rate_from_fourier_gap"]), 0.002919520342)
    close(float(summary["mean_quantum_cycle_time"]), 1.99244748967)
    close(float(diagnostics["scgf_asymptotic_variance_per_cycle"]), 0.05075400632)

    forbidden = {".tex", ".bib", ".pdf", ".doc", ".docx"}
    leaked = [
        path.name
        for path in ROOT.rglob("*")
        if path.suffix.lower() in forbidden
        and "figures" not in path.relative_to(ROOT).parts
        and ".git" not in path.relative_to(ROOT).parts
    ]
    if leaked:
        raise AssertionError(f"manuscript or submission files found: {leaked}")

    print("All stored headline values match and no manuscript files are present.")


if __name__ == "__main__":
    main()
