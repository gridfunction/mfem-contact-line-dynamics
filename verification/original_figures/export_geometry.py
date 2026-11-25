#!/usr/bin/env python3
"""Extract the figure inputs from the six original MFEM ParaView collections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from redraw import CASES, HERE, plots2d, slice3d
from plot_ac3d_final_surface import zero_contour_branch

import matplotlib.pyplot as plt
import numpy as np


def locate_run(saves: Path, case: str) -> Path:
    model, theta = case.split("_theta")
    grid = "nx128_ny128_nz128" if model == "ac3d" else "nx128_ny128"
    return saves / f"{model}_conservative_wetting_theta{theta}_{grid}_delta0p01_dt0p01"


def contour_snapshot(time: float, segments) -> dict:
    return {"time": time, "segments": [segment.tolist() for segment in segments if len(segment) >= 2]}


def extract_2d(case: str, run: Path) -> dict:
    model = "AC" if case.startswith("ac2d") else "CH"
    times = [(t, plots2d.file_at_time(run, model, t)) for t in plots2d.PINNED_TIMES]
    times.append(plots2d.final_file(run, model))
    snapshots = [contour_snapshot(t, plots2d.contour_segments(path)) for t, path in times]
    return {"version": 1, "case": case, "snapshots": snapshots}


def extract_3d(case: str, run: Path) -> dict:
    final_step, _, _ = slice3d.final_step_time_mass(run)
    steps = [slice3d.step_for_time(run, t) for t in plots2d.PINNED_TIMES] + [final_step]
    snapshots = []
    for step in steps:
        time, mass = slice3d.step_time_mass(run, step)
        grid, phase = slice3d.read_pvtu(run, step)
        x, z, field = slice3d.extract_y0_grid(grid, phase)
        del grid, phase
        fig, ax = plt.subplots()
        segments = ax.contour(x, z, field, levels=[0.0]).allsegs[0]
        plt.close(fig)
        record = contour_snapshot(time, segments)
        record["mass"] = mass
        snapshots.append(record)
    radius, height = zero_contour_branch(x, z, field)
    meridian = np.column_stack((radius, height)).tolist()
    return {"version": 1, "case": case, "snapshots": snapshots, "terminal_meridian": meridian}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("saves", type=Path, help="Original repository's Saves directory")
    parser.add_argument("--output", type=Path, default=HERE / "data")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        run = locate_run(args.saves, case)
        extract = extract_3d if case.startswith("ac3d") else extract_2d
        data = extract(case, run)
        path = args.output / f"{case}.json"
        path.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
        print(f"{case}: {path.stat().st_size:,} bytes", flush=True)


if __name__ == "__main__":
    main()
