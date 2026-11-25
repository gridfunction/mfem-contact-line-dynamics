# Phase-field numerical verification

These files accompany *Phase-Field Models, Sharp Interface Limits, and Numerical Schemes for Contact Line Dynamics*. The CSV histories contain the initial row and every accepted step for the six original examples and the 59 verification runs.

| File | Contents |
| --- | --- |
| [original_histories.csv](original_histories.csv) | Six original runs, including the 3D Allen–Cahn examples |
| [verification_histories.csv](verification_histories.csv) | Spatial, temporal, diffuse-thickness and transient studies, their controls, and four 2D reproduction spot checks |
| [run_manifest.json](run_manifest.json) | Run identifiers, parameters, commands, original history checksums, and executable checksums for the verification runs |
| [transient_metrics.csv](transient_metrics.csv) | Transient trajectory errors, wall-law residuals, and time-window sensitivity for the cases cited in the manuscript and response |
| [contact_line_2d_verification.patch](contact_line_2d_verification.patch) | Source changes relative to the solver baseline |
| [mfem-local.patch](mfem-local.patch) | The local MFEM header change recorded for the verification build |
| [run_verification.py](run_verification.py) | Standard-library Python runner for the manifest cases |
| [SHA256SUMS.txt](SHA256SUMS.txt) | Checksums of the files in this folder |

The two history tables retain the source precision of every listed field. The `dt` column is the accepted interval. `dt_retries` counts rejected Newton trials before that accepted interval. `mass` is the integral of the phase field, and `energy_decay_rate` is the energy change from the preceding accepted state divided by `dt`, with a positive value for decay. The initial row has `dt=0` and `nan` for the decay rate.

## Source and build

The new computations used the phase-field solver at commit `4658d3ce079825eaa697c6ed6c1df661be66a3c0` with the included solver patch. The build used MFEM 4.9.1 at `c31510289f23ee29292086a6ed12531b54f767e4` with the included MFEM diff, Open MPI 5.0.7, Hypre 2.33.0, METIS 4.0, SuiteSparse 7.4.0, and Apple Accelerate. Apple Clang 21.0.0 on arm64 compiled the C++17 code with `-O3`. The MFEM diff adds an accessor for nonconforming face interpolation. These conforming Q1 solvers do not call that accessor. [Solver baseline](https://github.com/gridfunction/mfem-contact-line-dynamics/commit/4658d3ce079825eaa697c6ed6c1df661be66a3c0) · [MFEM baseline](https://github.com/mfem/mfem/commit/c31510289f23ee29292086a6ed12531b54f767e4)

From a fresh solver checkout, set `VERIFICATION` to the absolute path of this data folder and prepare the source.

```bash
git checkout 4658d3ce079825eaa697c6ed6c1df661be66a3c0
git apply "$VERIFICATION/contact_line_2d_verification.patch"
cp Make.user.example Make.user
# Set MFEM_DIR in Make.user to your configured MFEM directory.
make -j2 build/ac2d_conservative_wetting build/ch2d_conservative_wetting
```

Configure MFEM with MPI, Hypre, METIS and SuiteSparse. Apply `mfem-local.patch` at the stated MFEM revision before building MFEM to reproduce the recorded source state. The verification cases ran on a single MPI rank. The 2D Cahn–Hilliard cases used the default UMFPACK-backed serial preconditioner. The manifest records the command arguments, executable SHA-256 digests, and process environment overrides. The runner needs Python 3.9 or later.

```bash
python3 "$VERIFICATION/run_verification.py" --list
python3 "$VERIFICATION/run_verification.py" --source . --output runs \
  --include '^space_ac_n32$' --dry-run
python3 "$VERIFICATION/run_verification.py" --source . --output runs \
  --include '^space_ac_n32$'
```

Omit `--include` to run all 59 cases. Choose a separate output directory for a rerun. The solver writes each case's accepted-step history and geometry diagnostics. Its visualization option regenerates the field files.

## Original runs

The six original histories come from the saved production data. The PVTU component counts indicate 8 MPI ranks for 2D Allen–Cahn, 1 for 2D Cahn–Hilliard, and 12 for 3D Allen–Cahn. The exact original executables remain unverified. Four new 2D spot checks reproduced the accepted-step schedules through the first shared snapshot at time 0.02. The largest absolute energy and mass differences over the compared rows were `4.72e-9` and `7.92e-13`.

## Transient metrics

The metric table compares uniformly sampled histories at time spacing 0.002. The two start times are 0.02 and 0.04. The wall speed and apex speed are derivatives of the wall and symmetry-axis zero crossings, estimated with local quadratic fits over half-widths 0.0125 and 0.0075. The `*_window_change` columns report the RMS differences between the two derivatives. The fit uses at least five available samples when possible. Trajectory comparisons to the sharp model are given for Allen–Cahn, with the contact angle in degrees. [Independent sharp-interface solver and setup](https://github.com/gridfunction/mfem-contact-line-dynamics/tree/b80fd5e42426c2db3ef80bacf324ffc8f44a2eac/sharp)

The wall-law residuals use the measured contact-line speed and dynamic angle. With unit surface tension, the driving force is `cos(theta_Y) - cos(theta_CL)`. The `angle`, `constant_calibrated`, and `constant_nominal` columns use friction `zeta sin(theta_CL)`, `zeta sin(theta_Y)`, and `zeta`, respectively. The profile-adjusted residual replaces `cos(theta_Y)` by `cos(theta_Y)/q` where `q = sqrt(2) delta |grad(phi)|` is evaluated at the wall crossing. `rms_profile_scale_error` measures the RMS deviation of `q` from one.
