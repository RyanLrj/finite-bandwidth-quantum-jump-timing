"""Passive dead-time filtering of thermal quantum-jump records.

Four bath-labelled jump channels of a coherently driven qutrit are monitored.
After a detected click the detector is blind for a fixed number of time bins;
hidden jumps do not extend this dead time (nonparalyzable detector).  The
detector never feeds back on the qutrit.

Along with the conditional density blocks, the filter propagates their first
moment weighted by accumulated bath entropy.  This yields the model-aware
posterior mean of physical bath entropy conditioned on the imperfect record.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


N = 3


@dataclass(frozen=True)
class Instrument:
    no_jump: np.ndarray
    jumps: tuple[np.ndarray, ...]
    entropy: np.ndarray
    h0: np.ndarray
    h: np.ndarray


def matrix_sqrt_positive(a: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((a + a.conj().T) / 2.0)
    if values.min() < -1e-12:
        raise ValueError("matrix is not positive")
    return (vectors * np.sqrt(np.maximum(values, 0.0))) @ vectors.conj().T


def make_instrument(omega: float, dt: float) -> Instrument:
    energies = np.array([0.0, 1.0, 2.5])
    h0 = np.diag(energies).astype(complex)
    # Resonant rotating frame: h generates the coherently driven 1<->2
    # transition, while h0 retains the laboratory bare energies used for heat
    # accounting.  Including h0 again in this frame would spuriously make the
    # drive off resonant and obscure its time-reversal protocol.
    h = np.zeros((N, N), dtype=complex)
    h[1, 2] = h[2, 1] = omega / 2.0

    def transition(target: int, source: int, rate: float) -> np.ndarray:
        result = np.zeros((N, N), dtype=complex)
        result[target, source] = np.sqrt(dt * rate)
        return result

    beta_c, beta_h = 2.0, 0.5
    cold_down, hot_down = 1.0, 0.8
    jumps = (
        transition(0, 1, cold_down),
        transition(1, 0, cold_down * np.exp(-beta_c * 1.0)),
        transition(0, 2, hot_down),
        transition(2, 0, hot_down * np.exp(-beta_h * 2.5)),
    )
    # Environment entropy per event: emission to a bath is positive.
    entropy = np.array([beta_c * 1.0, -beta_c * 1.0, beta_h * 2.5, -beta_h * 2.5])
    rate_effect = sum(j.conj().T @ j for j in jumps)
    unitary_values, unitary_vectors = np.linalg.eigh(h)
    unitary = (unitary_vectors * np.exp(-1j * unitary_values * dt)) @ unitary_vectors.conj().T
    no_jump = unitary @ matrix_sqrt_positive(np.eye(N) - rate_effect)
    completeness = no_jump.conj().T @ no_jump + rate_effect
    if np.linalg.norm(completeness - np.eye(N)) > 1e-12:
        raise RuntimeError("Kraus instrument is not trace preserving")
    return Instrument(no_jump=no_jump, jumps=jumps, entropy=entropy, h0=h0, h=h)


def apply(kraus: np.ndarray, rho: np.ndarray) -> np.ndarray:
    return kraus @ rho @ kraus.conj().T


def stationary_unmonitored(inst: Instrument) -> np.ndarray:
    kraus = (inst.no_jump,) + inst.jumps
    superop = sum(np.kron(k.conj(), k) for k in kraus)
    a = superop - np.eye(N * N)
    trace_row = np.eye(N).reshape(-1, order="F")
    a[-1, :] = trace_row
    b = np.zeros(N * N, dtype=complex)
    b[-1] = 1.0
    rho = np.linalg.solve(a, b).reshape((N, N), order="F")
    return (rho + rho.conj().T) / 2.0


def theoretical_entropy_rate(inst: Instrument, rho_ss: np.ndarray, dt: float) -> float:
    probabilities = np.array([np.trace(apply(k, rho_ss)).real for k in inst.jumps])
    return float(probabilities @ inst.entropy / dt)


def total_trace(blocks: list[np.ndarray]) -> float:
    return float(sum(np.trace(block).real for block in blocks))


def observed_update(
    rho_blocks: list[np.ndarray],
    moment_blocks: list[np.ndarray],
    inst: Instrument,
    symbol: int,
    dead_steps: int,
) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    """Apply one observed symbol: 0=no click, 1..4=detected channel."""
    new_rho = [np.zeros((N, N), dtype=complex) for _ in range(dead_steps + 1)]
    new_moment = [np.zeros((N, N), dtype=complex) for _ in range(dead_steps + 1)]

    # Live detector. A physical jump is visible and therefore selected by symbol.
    if symbol == 0:
        new_rho[0] += apply(inst.no_jump, rho_blocks[0])
        new_moment[0] += apply(inst.no_jump, moment_blocks[0])
    else:
        channel = symbol - 1
        jump = inst.jumps[channel]
        new_rho[dead_steps] += apply(jump, rho_blocks[0])
        new_moment[dead_steps] += apply(jump, moment_blocks[0])
        new_moment[dead_steps] += inst.entropy[channel] * apply(jump, rho_blocks[0])

    # Blind detector. Every physical alternative appears as no click.
    if symbol == 0:
        for remaining in range(1, dead_steps + 1):
            target = remaining - 1
            new_rho[target] += apply(inst.no_jump, rho_blocks[remaining])
            new_moment[target] += apply(inst.no_jump, moment_blocks[remaining])
            for channel, jump in enumerate(inst.jumps):
                mapped_rho = apply(jump, rho_blocks[remaining])
                new_rho[target] += mapped_rho
                new_moment[target] += apply(jump, moment_blocks[remaining])
                new_moment[target] += inst.entropy[channel] * mapped_rho

    likelihood = total_trace(new_rho)
    if likelihood <= 0.0:
        raise RuntimeError("zero-probability observed symbol")
    new_rho = [block / likelihood for block in new_rho]
    new_moment = [block / likelihood for block in new_moment]
    return new_rho, new_moment, likelihood


def symbol_probabilities(rho_blocks: list[np.ndarray], inst: Instrument, dead_steps: int) -> np.ndarray:
    probs = np.zeros(5, dtype=float)
    probs[0] += np.trace(apply(inst.no_jump, rho_blocks[0])).real
    for channel, jump in enumerate(inst.jumps):
        probs[channel + 1] += np.trace(apply(jump, rho_blocks[0])).real
    for remaining in range(1, dead_steps + 1):
        probs[0] += np.trace(apply(inst.no_jump, rho_blocks[remaining])).real
        probs[0] += sum(np.trace(apply(jump, rho_blocks[remaining])).real for jump in inst.jumps)
    probs = np.maximum(probs, 0.0)
    return probs / probs.sum()


def simulate_filter(
    omega: float,
    dt: float,
    dead_steps: int,
    steps: int,
    seed: int,
) -> dict[str, float]:
    inst = make_instrument(omega, dt)
    steady = stationary_unmonitored(inst)
    sigma_theory = theoretical_entropy_rate(inst, steady, dt)
    # For a nonparalyzable detector the refractory counter is known exactly
    # from the observed record, so no distribution over counter states is
    # needed. Hidden clicks do not reset or extend it.
    rho = steady.copy()
    moment = np.zeros((N, N), dtype=complex)
    remaining = 0
    rng = np.random.default_rng(seed)
    visible_entropy = 0.0
    clicks = 0
    log_likelihood = 0.0

    for _ in range(steps):
        probs = np.zeros(5, dtype=float)
        no_rho = apply(inst.no_jump, rho)
        jump_rhos = [apply(jump, rho) for jump in inst.jumps]
        if remaining == 0:
            probs[0] = np.trace(no_rho).real
            for channel, mapped in enumerate(jump_rhos):
                probs[channel + 1] = np.trace(mapped).real
        else:
            probs[0] = np.trace(no_rho).real + sum(np.trace(mapped).real for mapped in jump_rhos)
        probs = np.maximum(probs, 0.0)
        probs /= probs.sum()
        symbol = int(rng.choice(5, p=probs))

        no_moment = apply(inst.no_jump, moment)
        jump_moments = [apply(jump, moment) for jump in inst.jumps]
        if remaining == 0 and symbol:
            channel = symbol - 1
            new_rho = jump_rhos[channel]
            new_moment = jump_moments[channel] + inst.entropy[channel] * jump_rhos[channel]
            remaining = dead_steps
        elif symbol == 0:
            if remaining == 0:
                new_rho = no_rho
                new_moment = no_moment
            else:
                new_rho = no_rho + sum(jump_rhos)
                new_moment = no_moment.copy()
                for channel in range(len(inst.jumps)):
                    new_moment += jump_moments[channel] + inst.entropy[channel] * jump_rhos[channel]
                remaining -= 1
        else:
            raise RuntimeError("click emitted while detector is refractory")

        likelihood = float(np.trace(new_rho).real)
        rho = new_rho / likelihood
        moment = new_moment / likelihood
        log_likelihood += np.log(likelihood)
        if symbol:
            visible_entropy += float(inst.entropy[symbol - 1])
            clicks += 1

    duration = steps * dt
    posterior_entropy = float(np.trace(moment).real)
    return {
        "omega_drive": omega,
        "dt": dt,
        "dead_steps": float(dead_steps),
        "dead_time": dead_steps * dt,
        "steps": float(steps),
        "duration": duration,
        "sigma_theory": sigma_theory,
        "sigma_visible_clicks": visible_entropy / duration,
        "sigma_posterior_mean": posterior_entropy / duration,
        "visible_fraction": visible_entropy / (duration * sigma_theory),
        "posterior_fraction": posterior_entropy / (duration * sigma_theory),
        "detected_click_rate": clicks / duration,
        "log_likelihood_rate": log_likelihood / duration,
        "trace_error": abs(float(np.trace(rho).real) - 1.0),
    }


def main() -> None:
    # Stronger coherent drive raises the entropy signal without changing the
    # detector mechanism, improving the cancellation-dominated Monte Carlo
    # estimate of net bath entropy.
    omega = 4.0
    dt = 0.005
    steps = 40_000
    dead_steps_values = (0, 1, 5, 20, 100)
    replicas = 4
    rows: list[dict[str, float]] = []
    for dead_steps in dead_steps_values:
        for replica in range(replicas):
            rows.append(simulate_filter(omega, dt, dead_steps, steps, 12000 + 100 * dead_steps + replica))

    output = Path("passive_deadtime_quantum_filter_scan.csv")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("dead_time theory visible_mean visible_se posterior_mean posterior_se click_rate")
    for dead_steps in dead_steps_values:
        group = [row for row in rows if int(row["dead_steps"]) == dead_steps]
        visible = np.array([row["sigma_visible_clicks"] for row in group])
        posterior = np.array([row["sigma_posterior_mean"] for row in group])
        clicks = np.array([row["detected_click_rate"] for row in group])
        print(
            f"{dead_steps*dt:9.4g} {group[0]['sigma_theory']:10.6g} "
            f"{visible.mean():12.6g} {visible.std(ddof=1)/np.sqrt(replicas):10.3g} "
            f"{posterior.mean():14.6g} {posterior.std(ddof=1)/np.sqrt(replicas):10.3g} "
            f"{clicks.mean():10.5g}"
        )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
