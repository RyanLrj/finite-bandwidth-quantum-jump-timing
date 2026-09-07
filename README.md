# Finite-bandwidth quantum-jump timing

Reproducibility code and numerical data for a study of thermodynamic limits on
classical emulation of finite-bandwidth quantum-jump timing.

This repository contains only source code, numerical CSV files, and
reproducibility metadata. It intentionally does not contain the manuscript,
the Supplemental Material, or submission files.

## Environment

The final numerical audit used Python 3.12.14 on Windows with:

```text
numpy==2.3.5
Pillow==12.3.0
```

Create an isolated environment and install the dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate the environment using the command appropriate for your platform.

## Reproduce the reported analyses

Run commands from the repository root. The following sequence uses the stored
optimized frontier and regenerates the headline analysis tables and figures:

```bash
python symmetric_emission_analysis.py
python optimization_restart_audit.py
python composite_fourier_witness.py
python fourier_witness_robustness.py
python finite_sample_power.py
python detector_robustness.py
python paper_figures.py
python verify_outputs.py
```

The expensive fixed-resource optimization can be rerun separately:

```bash
python symmetric_emission_frontier.py
```

This command overwrites `symmetric_emission_frontier.csv`. The optimization is
deterministic for the supplied starting points, but runtime depends on the
machine. The stored CSV preserves the best-found candidates used in the final
analysis.

## Headline outputs

The verification script checks the following stored values:

- smallest KL rate found on the nine-point resource grid: 0.0244603020531;
- entropy-production rate of that candidate: 11.170081;
- conditional-timing fraction of KL: 99.535%;
- fixed Fourier expectation gap: 0.107860938022;
- Fourier Pinsker rate: 0.002919520342;
- mean complete-cycle time: 1.99244748967;
- Markov-additive variance per cycle: 0.05075400632.

These numbers describe the specified 31-state, affinity-60 candidate. They are
not a certified global optimum over every hidden stochastic model.

The three 300-iteration perturbation restarts converge to KL rates between
0.0244743092 and 0.0246873927. None improves the stored candidate. This is a
local seed-sensitivity check, not a global-optimality certificate.

The Fourier-witness audit constructs phases from each of the nine resource
points and evaluates every witness against all nine candidates. All 81 gaps
are positive, with minimum 0.107775241. The minimum remains 0.107597354 at the
physical frequency 4, 0.092289110 over the frequency interval 3.8--4.3, and
0.100877188 in 100 fixed-seed phase-error trials at standard deviation 0.20
radians. These are finite candidate-set checks, not a continuum certificate.

## File guide

- `symmetric_emission_frontier.py`: fixed-resource classical optimization.
- `optimization_restart_audit.py`: fixed-seed perturbation restart audit for
  the best-fitting reported candidate.
- `symmetric_emission_analysis.py`: KL decomposition and spectral audit.
- `composite_fourier_witness.py`: one frozen witness over nine candidates.
- `fourier_witness_robustness.py`: nine-anchor, frequency, and phase-error
  audit for the frozen witness.
- `finite_sample_power.py`: likelihood and bounded-statistic sample scales.
- `detector_robustness.py`: defined detector post-processing channels.
- `paper_figures.py`: regenerates figures from the CSV files.
- supporting modules: exact instruments, matrix propagation, optimization,
  and entropy-production calculations used by the headline scripts.

## Citation

Please cite the versioned repository below (no DOI has yet been assigned):

> Ruijie Lyu, *Finite-bandwidth quantum-jump timing: reproducibility code and
> numerical data*, version 1.2.0 (2026),
> https://github.com/RyanLrj/finite-bandwidth-quantum-jump-timing.

## License

The code and data are released under the MIT License. Copyright 2026 Ruijie
Lyu.
