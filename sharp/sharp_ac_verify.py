#!/usr/bin/env python3
"""Check the four spatial levels and a tighter BDF run of the sharp reference."""

import argparse
import csv
from pathlib import Path

import numpy as np

from sharp_ac import SharpAllenCahn


FIELDS = ("a", "theta_deg", "v_cl", "v_apex", "area", "height", "energy")
INTERVALS = (128, 256, 512, 1024)


def read_data(path):
    with (path / "trajectories.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    keys = [(float(r["time"]), r["theta0_deg"], r["zeta"], r["friction"]) for r in rows]
    values = np.array([[float(r[col]) for col in FIELDS] for r in rows])
    if not keys or len(keys) != len(set(keys)) or not np.isfinite(values).all():
        raise ValueError(f"Invalid trajectory data in {path}")
    return keys, values


def exact_cap_residual(n, stretch):
    model = SharpAllenCahn(n, stretch=stretch)
    theta = np.deg2rad(45.0)
    radius = np.sqrt(2.0 * model.area / (theta - np.sin(theta) * np.cos(theta)))
    a = radius * np.sin(theta)
    y = np.sqrt(radius**2 - (a * model.s[:-1]) ** 2) - radius * np.cos(theta)
    rate, velocities, measured_theta = model.rates(np.r_[y, a], 1.0, "angle")
    return max(abs(velocities)), abs(measured_theta - 45.0), abs(rate[-1])


def verify(args):
    datasets = [read_data(path) for path in args.space]
    keys = datasets[-1][0]
    tight_keys, tight = read_data(args.tight)
    if any(dataset_keys != keys for dataset_keys, values in datasets) or tight_keys != keys:
        raise ValueError("All runs must use the same cases and exact output times.")
    mask = np.array([key[0] >= args.after_time for key in keys])
    if not mask.any():
        raise ValueError("No output times remain after the chosen start of the comparison.")
    pairs = [
        (f"space_{left_n}_{right_n}", left[1], right[1])
        for left_n, right_n, left, right in zip(INTERVALS, INTERVALS[1:], datasets, datasets[1:])
    ]
    pairs.append(("time_512_standard_tight", datasets[-2][1], tight))

    with args.output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("comparison", "start_time", *(f"max_abs_{field}" for field in FIELDS)))
        for label, left, right in pairs:
            difference = np.max(np.abs(left[mask] - right[mask]), axis=0)
            writer.writerow((label, args.after_time, *difference))
            print(label, difference)

    reference = datasets[-1][1]
    grouped_energy = {}
    for key, row in zip(keys, reference):
        grouped_energy.setdefault(key[1:], []).append(row[-1])
    print("Maximum area error", np.max(abs(reference[:, 4] - 0.125)))
    print("Maximum a, theta, height", reference[:, [0, 1, 5]].max(axis=0))
    print("Largest signed energy increment", max(np.diff(rows).max() for rows in grouped_energy.values()))
    for n in INTERVALS:
        print("Stationary cap residual", n, *exact_cap_residual(n, args.stretch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", nargs=4, type=Path, required=True, help="Runs at N=128,256,512,1024")
    parser.add_argument("--tight", type=Path, required=True, help="Run at N=512 with tighter BDF tolerances")
    parser.add_argument("--after-time", type=float, default=0.02)
    parser.add_argument("--stretch", type=float, default=0.96)
    parser.add_argument("--output", type=Path, required=True)
    verify(parser.parse_args())
