#!/usr/bin/env python3
"""Compare wall-history angle to a contour fit and to the saved Q1 field."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from postprocess_verification import geometric_metrics, read_snapshot


def compare(run, spec, history, time):
    stored = min(history, key=lambda row: abs(float(row["time"]) - time))
    if abs(float(stored["time"]) - time) > 1.0e-9:
        raise ValueError(f"No accepted phase history at time {time} for {run}")
    snap = read_snapshot(run, at=time)
    geom = geometric_metrics(snap.curve, snap.grid.h, spec["delta"])
    x = geom["contact_radius"]
    _, gx, gy = snap.grid.evaluate(np.array([x]), np.array([0.0]))
    gx, gy = float(gx[0, 0]), float(gy[0, 0])
    norm = math.hypot(gx, gy)
    theta_vtk = math.degrees(math.acos(gy / norm))
    theta_log = float(stored["theta_cl_deg"])
    return {
        "case": run.name,
        **{key: spec[key] for key in ("model", "delta", "n", "theta0", "zeta")},
        "time": time,
        "contact_radius": x,
        "angle_gradient_log": theta_log,
        "angle_gradient_vtk": theta_vtk,
        "angle_contour_fit": geom["theta_cl_deg"],
        "fit_window": geom["theta_window"],
        "fit_window_spread": geom["theta_window_spread_deg"],
        "gradient_fit_difference": theta_log - geom["theta_cl_deg"],
        "gradient_vtk_log_difference": theta_vtk - theta_log,
        "contact_vtk_log_difference": x - float(stored["x_cl"]),
        "apex_vtk_log_difference": geom["apex_y"] - float(stored["apex_y"]),
        "wall_scale_vtk_log_difference": math.sqrt(2) * spec["delta"] * norm - float(stored["wall_profile_scale"]),
        "maximum_height_minus_apex": geom["cap_height"] - geom["apex_y"],
    }


def run(runs, output, times):
    comparisons = []
    for run in sorted(runs.glob("dynamic*")):
        specification = run / "run.json"
        if not specification.exists():
            continue
        try:
            metadata = json.loads(specification.read_text())
        except json.JSONDecodeError:
            continue  # The driver may be updating its record for an active run.
        if metadata.get("exit_code") != 0:
            continue
        with (run / "geometry.csv").open() as file:
            history = list(csv.DictReader(file))
        comparisons.extend(compare(run, metadata["case"], history, time) for time in times)
    if not comparisons:
        raise RuntimeError("No completed runs with comparable geometry were found")
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--times", type=float, nargs="+", default=[0.02, 0.2])
    args = parser.parse_args()
    run(args.runs, args.output, args.times)
