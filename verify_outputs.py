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
    restarts = list(
        csv.DictReader(
            (ROOT / "optimization_restart_audit.csv").open(
                newline="", encoding="utf-8"
            )
        )
    )
    robustness = {
        row["quantity"]: row
        for row in csv.DictReader(
            (ROOT / "fourier_witness_robustness_summary.csv").open(
                newline="", encoding="utf-8"
            )
        )
    }

    close(float(best["asymptotic_kl_rate"]), 0.0244603020531)
    close(float(best["total_ep_rate"]), 11.1700804549)
    close(float(summary["timing_fraction"]), 0.995350391128)
    close(float(summary["best_channel_resolved_fourier_gap"]), 0.107860938022)
    close(float(summary["pinsker_rate_from_fourier_gap"]), 0.002919520342)
    close(float(summary["mean_quantum_cycle_time"]), 1.99244748967)
    close(float(diagnostics["scgf_asymptotic_variance_per_cycle"]), 0.05075400632)
    expected_restarts = (0.0244765463, 0.0244743092, 0.0246873927)
    if len(restarts) != len(expected_restarts):
        raise AssertionError("restart audit must contain three rows")
    for row, expected in zip(restarts, expected_restarts):
        close(float(row["final_kl_rate"]), expected)
        if float(row["final_kl_rate"]) <= float(best["asymptotic_kl_rate"]):
            raise AssertionError("a restart unexpectedly improves the stored frontier")

    close(
        float(robustness["minimum_gap_over_all_anchor_candidate_pairs"]["value"]),
        0.107775240750,
    )
    close(
        float(robustness["minimum_leave_anchor_out_gap"]["value"]),
        0.107775240750,
    )
    close(
        float(robustness["minimum_gap_at_physical_drive_frequency"]["value"]),
        0.107597354209,
    )
    close(
        float(robustness["minimum_gap_over_frequency_scan"]["value"]),
        0.0922891101545,
    )
    close(
        float(robustness["minimum_gap_phase_error_sd_0.20"]["value"]),
        0.100877188412,
    )

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
