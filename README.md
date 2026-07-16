# MFEM Contact-Line Dynamics

This repository contains MFEM implementations for phase-field contact-line
dynamics and the minimal code needed to reproduce the two- and
three-dimensional numerical results in Section 5 of the accompanying droplet
manuscript. It implements the conservative Allen--Cahn scheme in 2D and 3D
and the Cahn--Hilliard scheme in 2D. It does not depend on SLOTH.

Generated VTK/ParaView data are intentionally excluded: a complete local run
occupies about 85 GB, while every paper result can be regenerated from the
sources and commands below.

## Requirements

- C++17 compiler
- MPI
- MFEM configured with MPI, Hypre, METIS, and SuiteSparse
- Python 3.10 or newer for figure generation

The reported runs used MFEM 4.9.1 at commit
`c31510289f23ee29292086a6ed12531b54f767e4`, Open MPI 5.0.7, and an
Apple Clang 21 compiler. The local uncommitted MFEM change was unrelated to
these conforming H1 solvers. Other recent MFEM builds may work, but the commit
above is the tested version.

Clone and configure MFEM first. Then configure and build this repository:

```bash
cp Make.user.example Make.user
# Edit Make.user so MFEM_DIR points to the configured MFEM tree.
make -j4
```

`Make.user`, compiled binaries, logs, and all generated results are ignored.
To install the plotting environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Quick verification

The smoke tests exercise the 2D Allen--Cahn solver in serial and with two MPI
ranks, and the 3D solver with two ranks:

```bash
make run-smoke
make run-smoke-mpi
make run-smoke-3d
```

Each test writes `time_specialized.csv` under `Saves/`. Successful runs should
finish without a Newton failure and report energy and conserved mass histories.

## Reproduce the paper simulations

Run all commands from the repository root. `NP` is the MPI rank count suitable
for the available machine. The 2D Cahn--Hilliard direct solver is serial; its
SuiteSparse/UMFPACK factorization is the non-scalable solver described in the
paper.

```bash
NP=12

# 2D conservative Allen--Cahn, Figures 5.1--5.4
mpirun -np "$NP" ./build/ac2d_conservative_wetting -theta 45
mpirun -np "$NP" ./build/ac2d_conservative_wetting -theta 135

# 2D Cahn--Hilliard, Figures 5.1--5.4
./build/ch2d_conservative_wetting -theta 45
./build/ch2d_conservative_wetting -theta 135

# 3D conservative Allen--Cahn, Figures 5.5--5.8
mpirun -np "$NP" ./build/ac3d_conservative_wetting -theta 45
mpirun -np "$NP" ./build/ac3d_conservative_wetting -theta 135
```

The defaults are the manuscript parameters: Q1 elements on a `128 x 128` mesh
in 2D and a `128 x 128 x 128` mesh in 3D, `delta=0.01`, target `dt=0.01`,
`sigma_lg=xi=zeta=1`, and stopping threshold `1e-7`. The Young angle is set by
`-theta`. The 2D runs save the common snapshot times `0.02`, `0.2`, and `1.0`.
The 3D simulations are computationally and storage intensive; use the smoke
tests first.

The solvers derive deterministic output names from their parameters. The six
production directories are:

```text
Saves/ac2d_conservative_wetting_theta45_nx128_ny128_delta0p01_dt0p01
Saves/ac2d_conservative_wetting_theta135_nx128_ny128_delta0p01_dt0p01
Saves/ch2d_conservative_wetting_theta45_nx128_ny128_delta0p01_dt0p01
Saves/ch2d_conservative_wetting_theta135_nx128_ny128_delta0p01_dt0p01
Saves/ac3d_conservative_wetting_theta45_nx128_ny128_nz128_delta0p01_dt0p01
Saves/ac3d_conservative_wetting_theta135_nx128_ny128_nz128_delta0p01_dt0p01
```

Use `-h` on an executable to list every numerical, solver, output, and
time-stepping option.

## Reproduce the paper figures

Set short names for the output directories:

```bash
AC45=Saves/ac2d_conservative_wetting_theta45_nx128_ny128_delta0p01_dt0p01
AC135=Saves/ac2d_conservative_wetting_theta135_nx128_ny128_delta0p01_dt0p01
CH45=Saves/ch2d_conservative_wetting_theta45_nx128_ny128_delta0p01_dt0p01
CH135=Saves/ch2d_conservative_wetting_theta135_nx128_ny128_delta0p01_dt0p01
AC3D45=Saves/ac3d_conservative_wetting_theta45_nx128_ny128_nz128_delta0p01_dt0p01
AC3D135=Saves/ac3d_conservative_wetting_theta135_nx128_ny128_nz128_delta0p01_dt0p01
mkdir -p figures
```

Generate all 2D paper figures:

```bash
python scripts/plot_mfem_2d_paper_figures.py \
  --ac45-dir "$AC45" --ch45-dir "$CH45" \
  --ac135-dir "$AC135" --ch135-dir "$CH135" \
  --out-dir figures
```

Generate the 3D cross-sections, energy plot, and reflected surface comparison:

```bash
python scripts/plot_ac3d_symmetric_xz_slice.py "$AC3D45" --theta 45 \
  --times 0.02 0.2 1.0 --quarter \
  --out-base figures/ac3d_theta45_xz_quarter_cut_times
python scripts/plot_ac3d_symmetric_xz_slice.py "$AC3D135" --theta 135 \
  --times 0.02 0.2 1.0 --quarter \
  --out-base figures/ac3d_theta135_xz_quarter_cut_times
python scripts/plot_ac3d_energy_decay.py \
  --case45 "$AC3D45" --case135 "$AC3D135" \
  --out-base figures/ac3d_energy_decay_angles
python scripts/plot_ac3d_surface_pair.py \
  --case45 "$AC3D45" --case135 "$AC3D135" \
  --out-base figures/ac3d_surface_pair_theta45_theta135
```

Each plotting command emits both PDF and PNG files. PyVista reads the generated
ParaView collections and extracts the zero level set; the CSV histories supply
energy, mass, time, and iteration diagnostics.

## Repository contents

```text
src/          three standalone MFEM solvers
scripts/      only the scripts used to generate manuscript figures
makefile      build and smoke-test targets
```

## Citation

If this code contributes to published work, please cite the accompanying
manuscript:

> *Phase-Field Models, Sharp Interface Limits, and Numerical Schemes for
> Contact Line Dynamics*. arXiv preprint, forthcoming (2026).

The author list, arXiv identifier, and journal DOI will be added here when they
become available.

## License

The source code is available under the [MIT License](LICENSE).
