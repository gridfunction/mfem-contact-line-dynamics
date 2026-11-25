#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Reproduce the six original numerical examples and their paper figures.

Usage: bash scripts/reproduce_examples.sh [options]

  --output DIR                     Output root, default Saves/reproduced_examples
  --only all|2d|3d                 Select the spatial dimension, default all
  --stage all|simulate|figures     Select simulations, figures, or both
  --dry-run                        Print commands without creating files
  -h, --help                       Show this help

CONTACT_LINE_MPIEXEC and CONTACT_LINE_PYTHON select the launchers.
The cases use 8 MPI ranks for 2D Allen–Cahn, one for 2D Cahn–Hilliard,
and 12 for 3D Allen–Cahn.
EOF
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output="$repo/Saves/reproduced_examples"
dimension=all
stage=all
dry_run=false
mpiexec=${CONTACT_LINE_MPIEXEC:-mpirun}
python=${CONTACT_LINE_PYTHON:-python3}

while (($#)); do
  case "$1" in
    --output|--only|--stage)
      (($# >= 2)) || die "$1 needs a value"
      case "$1" in
        --output) output=$2 ;;
        --only) dimension=$2 ;;
        --stage) stage=$2 ;;
      esac
      shift 2
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) die "unknown option $1" ;;
  esac
done

case "$dimension" in
  all|2d|3d) ;;
  *) die "--only must be all, 2d, or 3d" ;;
esac
case "$stage" in
  all|simulate|figures) ;;
  *) die "--stage must be all, simulate, or figures" ;;
esac
[[ -n "$output" ]] || die "--output must be a directory"
[[ "$output" = /* ]] || output="$PWD/$output"

ac45="$output/ac2d_conservative_wetting_theta45_nx128_ny128_delta0p01_dt0p01"
ac135="$output/ac2d_conservative_wetting_theta135_nx128_ny128_delta0p01_dt0p01"
ch45="$output/ch2d_conservative_wetting_theta45_nx128_ny128_delta0p01_dt0p01"
ch135="$output/ch2d_conservative_wetting_theta135_nx128_ny128_delta0p01_dt0p01"
ac3d45="$output/ac3d_conservative_wetting_theta45_nx128_ny128_nz128_delta0p01_dt0p01"
ac3d135="$output/ac3d_conservative_wetting_theta135_nx128_ny128_nz128_delta0p01_dt0p01"
figures2d="$output/figures/2d"
figures3d="$output/figures/3d"

run() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  if ! "$dry_run"; then
    "$@"
  fi
}

require_new() {
  local path
  for path in "$@"; do
    [[ ! -e "$path" && ! -L "$path" ]] || die "output already exists at $path"
  done
}

require_case() {
  local path=$1 collection=$2
  [[ -s "$path/time_specialized.csv" && -s "$path/$collection/$collection.pvd" ]] ||
    die "missing history or ParaView collection in $path"
}

simulate() {
  local solver=$1 ranks=$2 angle=$3 final_time=$4 cadence=$5 preconditioner=$6 case_path=$7
  local command=("./build/${solver}_conservative_wetting")
  if ((ranks > 1)); then
    command=("$mpiexec" -np "$ranks" "${command[@]}")
  fi
  command+=(-theta "$angle" -nx 128 -ny 128 -o 1 -delta 0.01 -dt 0.01
    -dtmin 0.0001 -tf "$final_time" -edrtol 1e-7 -adapt-dt
    -ntol 1e-10 -nmax 20 -dt-retries 8 -pc "$preconditioner"
    -oe "$cadence" -vis -out "$case_path")
  if [[ "$solver" = ac3d ]]; then
    command+=(-nz 128)
  fi
  run mkdir "$case_path"
  run "${command[@]}"
}

cd "$repo"
if [[ "$stage" != figures ]]; then
  if [[ "$dimension" != 3d ]]; then
    require_new "$ac45" "$ac135" "$ch45" "$ch135"
  fi
  if [[ "$dimension" != 2d ]]; then
    require_new "$ac3d45" "$ac3d135"
  fi
  if ! "$dry_run"; then
    command -v "$mpiexec" >/dev/null || die "missing MPI launcher $mpiexec"
    if [[ "$dimension" != 3d ]]; then
      [[ -x build/ac2d_conservative_wetting && -x build/ch2d_conservative_wetting ]] ||
        die "build both 2D solvers first"
    fi
    if [[ "$dimension" != 2d ]]; then
      [[ -x build/ac3d_conservative_wetting ]] || die "build the 3D solver first"
    fi
  fi
fi

if [[ "$stage" != simulate ]]; then
  if [[ "$dimension" != 3d ]]; then
    require_new "$figures2d"
  fi
  if [[ "$dimension" != 2d ]]; then
    require_new "$figures3d"
  fi
  if ! "$dry_run"; then
    "$python" -c 'import matplotlib, numpy, pyvista' ||
      die "install the Python figure requirements with $python"
  fi
fi

if [[ "$stage" != figures ]]; then
  run mkdir -p "$output"
  if [[ "$dimension" != 3d ]]; then
    # The large positive cadence retains pinned output and supports the baseline solver.
    simulate ac2d 8 45 4 1000000000 amg "$ac45"
    simulate ac2d 8 135 4 1000000000 amg "$ac135"
    simulate ch2d 1 45 3.44 0 direct "$ch45"
    simulate ch2d 1 135 2.39 0 direct "$ch135"
  fi
  if [[ "$dimension" != 2d ]]; then
    simulate ac3d 12 45 10 10 amg "$ac3d45"
    simulate ac3d 12 135 10 10 amg "$ac3d135"
  fi
fi

if [[ "$stage" != simulate ]]; then
  if ! "$dry_run"; then
    if [[ "$dimension" != 3d ]]; then
      require_case "$ac45" ConservativeWettingAC
      require_case "$ac135" ConservativeWettingAC
      require_case "$ch45" ConservativeWettingCH
      require_case "$ch135" ConservativeWettingCH
    fi
    if [[ "$dimension" != 2d ]]; then
      require_case "$ac3d45" ConservativeWettingAC3D
      require_case "$ac3d135" ConservativeWettingAC3D
    fi
  fi
  run mkdir -p "$output/figures"
  if [[ "$dimension" != 3d ]]; then
    run mkdir "$figures2d"
    run "$python" scripts/plot_mfem_2d_paper_figures.py \
      --ac45-dir "$ac45" --ch45-dir "$ch45" \
      --ac135-dir "$ac135" --ch135-dir "$ch135" --paper-layout --out-dir "$figures2d"
  fi
  if [[ "$dimension" != 2d ]]; then
    run mkdir "$figures3d"
    run "$python" scripts/plot_ac3d_symmetric_xz_slice.py "$ac3d45" --theta 45 \
      --times 0.02 0.2 1.0 --quarter --paper-layout \
      --out-base "$figures3d/ac3d_theta45_xz_quarter_cut_times"
    run "$python" scripts/plot_ac3d_symmetric_xz_slice.py "$ac3d135" --theta 135 \
      --times 0.02 0.2 1.0 --quarter --paper-layout \
      --out-base "$figures3d/ac3d_theta135_xz_quarter_cut_times"
    run "$python" scripts/plot_ac3d_energy_decay.py --case45 "$ac3d45" --case135 "$ac3d135" \
      --out-base "$figures3d/ac3d_energy_decay_angles"
    run "$python" scripts/plot_ac3d_surface_pair.py --case45 "$ac3d45" --case135 "$ac3d135" \
      --out-base "$figures3d/ac3d_surface_pair_theta45_theta135"
  fi
fi
