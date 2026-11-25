# MFEM Contact-Line Dynamics

This repository accompanies *Phase-Field Models, Sharp Interface Limits, and Numerical Schemes for Contact Line Dynamics*. It contains the conservative Allen–Cahn solvers in two and three dimensions, the two-dimensional Cahn–Hilliard solver, and the independent sharp-interface Allen–Cahn reference. The instructions below reproduce the numerical experiments, figures, and equilibrium table in the manuscript.

## Build

Use a C++17 compiler, GNU Make, and MFEM configured with MPI, Hypre, METIS, and SuiteSparse. The recorded verification build used MFEM 4.9.1 at `c31510289f23ee29292086a6ed12531b54f767e4`, Open MPI 5.0.7, Hypre 2.33.0, METIS 4.0, and SuiteSparse 7.4.0 with Apple Accelerate.

Clone this repository and MFEM as sibling directories:

```sh
git clone --branch main https://github.com/gridfunction/mfem-contact-line-dynamics.git
git clone https://github.com/mfem/mfem.git
cd mfem
git checkout c31510289f23ee29292086a6ed12531b54f767e4
git apply ../mfem-contact-line-dynamics/verification/mfem-local.patch
```

With the dependency libraries already installed, set `HYPRE_DIR`, `METIS_DIR`, and `SUITESPARSE_DIR` to their installation or build directories. The following GNU Make configuration uses the include layout of the recorded build. For a distribution-provided SuiteSparse installation, set the include flag to the directory containing `umfpack.h`. Use `MFEM_USE_METIS_5=YES` with METIS 5.

```sh
make config MFEM_USE_MPI=YES MFEM_USE_METIS=YES MFEM_USE_METIS_5=NO \
  MFEM_USE_SUITESPARSE=YES HYPRE_DIR="$HYPRE_DIR" METIS_DIR="$METIS_DIR" \
  SUITESPARSE_DIR="$SUITESPARSE_DIR" \
  SUITESPARSE_OPT="-I$SUITESPARSE_DIR/include/suitesparse"
make -j4
MFEM_BUILD="$(pwd)"

cd ../mfem-contact-line-dynamics
make -j4 MFEM_DIR="$MFEM_BUILD"
```

An existing Make-built MFEM installation can also be used. Set `MFEM_DIR` in a local `Make.user` or pass it on the command line as above. The 2D verification source changes are already included on this branch.

For plotting, postprocessing, and the sharp reference, install the Python dependencies in a Python 3.12 virtual environment:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

The following smoke tests exercise serial and parallel 2D Allen–Cahn and parallel 3D Allen–Cahn. Use the verification case below for a serial Cahn–Hilliard smoke test.

```sh
make MFEM_DIR="$MFEM_BUILD" run-smoke run-smoke-mpi run-smoke-3d
python verification/run_verification.py --output Saves/verification-smoke \
  --include '^space_ch_n32$'
```

## Wetting and dewetting examples

Run the original six examples and generate their figures from the repository root:

```sh
bash scripts/reproduce_examples.sh --output Saves/reproduced_examples
```

The script uses 8 MPI ranks for 2D Allen–Cahn, 1 for 2D Cahn–Hilliard, and 12 for 3D Allen–Cahn, matching the saved production outputs. The 2D Cahn–Hilliard driver uses GMRES with a cached serial UMFPACK factorization as its preconditioner. Each model starts with a rectangular drop or block and relaxes toward the prescribed Young angle of 45 or 135 degrees. The script records the mesh and time-stepping options. It also supports `--only 2d`, `--only 3d`, `--stage simulate`, `--stage figures`, and `--dry-run`.

The two 3D runs take several hours and generated about 51 GB of ParaView fields in the recorded runs. To redraw the original figures from the saved results without running MFEM, use:

```sh
python verification/original_figures/redraw.py --output figures/original
```

The [archived figure inputs](verification/original_figures/) and [original step histories](verification/original_histories.csv) are included.

## Numerical verification

The [verification instructions](verification/README.md) cover the 59-run manifest, refinement and equilibrium postprocessing, and the transient comparisons. The [sharp-interface setup](sharp/README.md) documents the standalone solver, both friction laws, and the reference refinement checks.

## Citation

See the [accompanying preprint](https://arxiv.org/abs/2607.17410).

## License

The source code is available under the [MIT License](LICENSE).
