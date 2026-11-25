# Numerical verification

This directory contains the case manifest, accepted-step histories, and scripts for the numerical verification in Section 5.4 of the manuscript. The reference data support Figure 2 (mesh and time refinement), Table 1 (equilibrium cap errors), and Figure 3 (transient contact-line motion).

## Redraw from the reference data

Install the Python environment described in the [repository README](../README.md). From the repository root, run:

```sh
python verification/tools/postprocess_studies.py \
  --reference verification/reference --output figures/verification
python verification/tools/analyze_transients.py \
  --series verification/reference/transient_series.csv \
  --sharp verification/reference/sharp_trajectories.csv --output figures/verification
```

The equilibrium table uses the rows with `model=ac` and `variant=main` in [equilibrium_study.csv](reference/equilibrium_study.csv). The redraw uses the saved phase-field differences and transient series and requires no MFEM build.

## Rerun the simulations and compute the results

Build all executables using the [repository README](../README.md). The 2D solvers in this checkout include the verification changes. From the repository root, list or dry-run the cases:

```sh
python verification/run_verification.py --list
python verification/run_verification.py --include '^space_ac_n32$' --dry-run
```

Regenerate the 59 manifest cases and the eight independent sharp trajectories:

```sh
python verification/run_verification.py --output Saves/verification-rerun
python sharp/sharp_ac.py --n 1024 --stretch 0.96 --output-dir Saves/sharp-rerun
```

The phase-field runner uses one MPI rank per case. Use `--include` to select a subset and `--workers` to run independent cases concurrently. Choose a fresh output directory for a rerun. Each case writes `run.json`, the solver log, an accepted-step history, and geometry diagnostics. The manifest specifies which cases write ParaView fields.

```sh
python verification/tools/postprocess_studies.py \
  --runs Saves/verification-rerun --output figures/verification-rerun
python verification/tools/analyze_transients.py \
  --runs Saves/verification-rerun --sharp Saves/sharp-rerun/trajectories.csv \
  --output figures/verification-rerun
```

The first command computes successive field differences and equilibrium errors from the saved fields. The second computes the trajectory and wall-law comparisons from the accepted-step geometry. The resulting `phase_field_refinement.pdf` and `transient_trajectories.pdf` are the two verification figures in the manuscript. The resulting `equilibrium_study.csv` provides the values in Table 1. The [sharp solver instructions](../sharp/README.md) also document the independent spatial and temporal controls.

## Settings and archived data

- The new tests use the half-domain `[0,1] × [0,1]`, conforming Q1 elements, Young angle 45 degrees, initial liquid half-area 0.125, and unit surface tension and bulk friction. The phase field has a dynamic wetting condition on the substrate and homogeneous Neumann conditions on the other faces. The Cahn–Hilliard chemical potential has zero normal flux on every face.
- The refinement and transient cases interpolate a tanh profile of signed distance to a parabola continued through the wall. The equilibrium cases use the area-matched circular cap. A tapered bulk nodal adjustment matches the target mass of 0.75 while keeping the zero contour fixed. The spatial and temporal refinement cases are compared at time 0.02. The transient cases record the wall and apex zero crossings and the angle at each accepted step.
- [Run manifest](run_manifest.json) gives the commands, parameters, executable digests, and recorded environment. [Verification histories](verification_histories.csv) and [original histories](original_histories.csv) include the initial row and every accepted step. The `dt` column is the accepted interval; the energy-decay rate is the energy decrease divided by that interval. [Archived figure inputs](original_figures/) cover the six original runs.
- [Transient metrics](transient_metrics.csv) report trajectory discrepancies and wall-law residuals. The primary comparison window is 0.02 to 0.186, with a check starting at 0.04. Local quadratic fits to uniformly sampled wall and apex positions give their velocities. The residual evaluates the measured wall speed against the angle-dependent, Young-angle calibrated, and nominal constant friction laws.
- [Solver patch](contact_line_2d_verification.patch) records the verification changes relative to solver commit `4658d3ce079825eaa697c6ed6c1df661be66a3c0`. Those changes are integrated in the current source. [MFEM patch](mfem-local.patch) records the local header change at MFEM commit `c31510289f23ee29292086a6ed12531b54f767e4`. [Checksums](SHA256SUMS.txt) cover the archived inputs and results.

Numerical reruns can vary slightly with the compiler, solver library versions, and MPI implementation. The archived inputs preserve the results used for the manuscript plots.
