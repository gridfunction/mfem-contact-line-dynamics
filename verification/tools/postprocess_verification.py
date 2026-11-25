#!/usr/bin/env python3
"""Measure 2D Q1 MFEM snapshots and compare them on a common quadrature grid."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

import numpy as np
import pyvista as pv


TIME_TOLERANCE = 1.0e-8
MERGE_TOLERANCE = 1.0e-10


@dataclass(frozen=True)
class Q1Grid:
    x: np.ndarray
    y: np.ndarray
    phi: np.ndarray  # Row index follows y; column index follows x.

    @classmethod
    def from_vtk(cls, mesh):
        points = np.asarray(mesh.points, dtype=float)
        phase = np.asarray(mesh.point_data["phi"], dtype=float)
        if phase.shape != (len(points),) or not np.all(np.isfinite(phase)):
            raise ValueError("The VTK point data must contain one finite phi per node")
        if np.ptp(points[:, 2]) > MERGE_TOLERANCE:
            raise ValueError("Expected a two-dimensional Cartesian mesh in the x-y plane")

        x, ix = np.unique(np.round(points[:, 0], 12), return_inverse=True)
        y, iy = np.unique(np.round(points[:, 1], 12), return_inverse=True)
        if len(x) < 2 or len(y) < 2 or mesh.n_cells != (len(x) - 1) * (len(y) - 1):
            raise ValueError("Expected one Q1 cell per rectangle of a Cartesian mesh")
        if not np.all(np.isin(mesh.celltypes, [pv.CellType.QUAD, pv.CellType.LAGRANGE_QUADRILATERAL])):
            raise ValueError("Expected four-corner quadrilateral cells")
        if not np.all(np.diff(mesh.offset) == 4):
            raise ValueError("This quadrature reconstruction supports Q1 data only")

        index = iy * len(x) + ix
        count = np.bincount(index, minlength=len(x) * len(y))
        if np.any(count == 0):
            raise ValueError("One or more Cartesian mesh nodes are missing")
        values = np.bincount(index, weights=phase, minlength=len(count)) / count
        if not np.allclose(values[index], phase, atol=1.0e-10, rtol=0.0):
            raise ValueError("Repeated VTK nodes have inconsistent phi values")
        return cls(x, y, values.reshape(len(y), len(x)))

    @property
    def h(self):
        return float(max(np.max(np.diff(self.x)), np.max(np.diff(self.y))))

    @property
    def domain_area(self):
        return float(np.ptp(self.x) * np.ptp(self.y))

    def integral(self):
        q = (self.phi[:-1, :-1] + self.phi[1:, :-1] + self.phi[:-1, 1:] + self.phi[1:, 1:]) / 4
        return float(np.einsum("i,j,ij->", np.diff(self.y), np.diff(self.x), q))

    def evaluate(self, x, y):
        """Return value and both derivatives on the tensor product of x and y."""
        ix = np.clip(np.searchsorted(self.x, x, side="right") - 1, 0, len(self.x) - 2)
        iy = np.clip(np.searchsorted(self.y, y, side="right") - 1, 0, len(self.y) - 2)
        hx = self.x[ix + 1] - self.x[ix]
        hy = (self.y[iy + 1] - self.y[iy])[:, None]
        u = (x - self.x[ix]) / hx
        v = (y - self.y[iy])[:, None] / hy
        f00 = self.phi[np.ix_(iy, ix)]
        f10 = self.phi[np.ix_(iy, ix + 1)]
        f01 = self.phi[np.ix_(iy + 1, ix)]
        f11 = self.phi[np.ix_(iy + 1, ix + 1)]
        f0 = f00 + u * (f10 - f00)
        f1 = f01 + u * (f11 - f01)
        return f0 + v * (f1 - f0), ((1 - v) * (f10 - f00) + v * (f11 - f01)) / hx, (f1 - f0) / hy


def phase_errors(test, reference):
    """Integrate squared Q1 value and gradient differences on the union mesh."""
    axes = []
    for test_axis, ref_axis in [(test.x, reference.x), (test.y, reference.y)]:
        if not np.allclose(test_axis[[0, -1]], ref_axis[[0, -1]], atol=MERGE_TOLERANCE, rtol=0):
            raise ValueError("Phase error reference and sample must have the same domain")
        axes.append(np.unique(np.concatenate([test_axis, ref_axis])))
    x, y = axes
    dx = np.diff(x) / 2
    dy = np.diff(y) / 2
    x_mid = (x[:-1] + x[1:]) / 2
    y_mid = (y[:-1] + y[1:]) / 2
    # Both grids are bilinear on each rectangle of their union. The two-point
    # rule integrates both the squared difference and squared derivatives exactly.
    l2, h1_semi = 0.0, 0.0
    gauss = [-1 / math.sqrt(3), 1 / math.sqrt(3)]
    for sx in gauss:
        xq = x_mid + sx * dx
        for sy in gauss:
            yq = y_mid + sy * dy
            for first in range(0, len(yq), 128):
                stop = first + 128
                ft, fx_t, fy_t = test.evaluate(xq, yq[first:stop])
                fr, fx_r, fy_r = reference.evaluate(xq, yq[first:stop])
                l2 += np.einsum("i,j,ij->", dy[first:stop], dx, (ft - fr) ** 2)
                h1_semi += np.einsum("i,j,ij->", dy[first:stop], dx, (fx_t - fx_r) ** 2 + (fy_t - fy_r) ** 2)
    return {"phi_l2_error": math.sqrt(l2), "phi_h1_seminorm_error": math.sqrt(h1_semi)}


def half_drop_contour(mesh, grid):
    """Retain the unique component that connects the substrate to the symmetry axis."""
    if not np.allclose([grid.x[0], grid.y[0]], [0, 0], atol=MERGE_TOLERANCE):
        raise ValueError("Expected symmetry at x=0 and substrate at y=0")
    contour = mesh.contour([0.0], scalars="phi")
    points = np.asarray(contour.points[:, :2], dtype=float)
    points, canonical = np.unique(np.round(points, 12), axis=0, return_inverse=True)

    neighbors = {}
    lines = contour.lines
    index = 0
    while index < len(lines):
        count = lines[index]
        line = canonical[lines[index + 1:index + 1 + count]]
        index += count + 1
        for first, second in zip(line[:-1], line[1:]):
            if first == second:
                continue
            neighbors.setdefault(first, set()).add(second)
            neighbors.setdefault(second, set()).add(first)

    candidates = []
    for foot in neighbors:
        if len(neighbors[foot]) != 1 or abs(points[foot, 1]) > MERGE_TOLERANCE:
            continue
        path = [foot]
        seen = {foot}
        while True:
            next_nodes = neighbors[path[-1]] - seen
            if len(next_nodes) != 1:
                break
            next_node = next_nodes.pop()
            path.append(next_node)
            seen.add(next_node)
        end = path[-1]
        if len(path) > 1 and len(neighbors[end]) == 1 and abs(points[end, 0]) <= MERGE_TOLERANCE:
            candidates.append(points[path])
    if len(candidates) != 1:
        raise ValueError(f"Expected one zero contour between the substrate and symmetry axis, found {len(candidates)}")
    return candidates[0]


def arc_lengths(curve):
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]


def resample_curve(curve, count, maximum=None):
    length = arc_lengths(curve)
    s = np.linspace(0, length[-1] if maximum is None else maximum, count)
    return s, np.column_stack([np.interp(s, length, curve[:, dim]) for dim in range(2)])


def angle_from_tangent(curve, window):
    # Fit x(s), y(s) on equally spaced arclength points, anchored at the wall
    # crossing. This tangent orientation distinguishes obtuse and acute drops.
    s, samples = resample_curve(curve, 33, window)
    u = s / window
    fit = np.linalg.lstsq(np.column_stack([u, u**2, u**3]), samples - samples[0], rcond=None)[0]
    tx, ty = fit[0]
    return math.degrees(math.atan2(ty, -tx))


def geometric_metrics(curve, h, delta=None, fit_window=None):
    length = arc_lengths(curve)[-1]
    if fit_window is None:
        fit_window = max(4 * h, 2 * delta if delta else 0)
    if fit_window <= 0 or length <= 0:
        raise ValueError("A contact angle fit requires a positive arclength window")
    fit_window = min(fit_window, 0.25 * length)
    ax = curve[:, 0]
    ay = curve[:, 1]
    angles = [angle_from_tangent(curve, fit_window * factor) for factor in [0.75, 1, 1.25]]
    if not 0 < angles[1] < 180:
        raise ValueError("The fitted liquid contact angle must be between 0 and 180 degrees")
    return {
        "contact_radius": float(ax[0]),
        "cap_height": float(np.max(ay)),
        "apex_y": float(ay[-1]),
        "theta_cl_deg": angles[1],
        "theta_window": fit_window,
        "theta_window_spread_deg": max(angles) - min(angles),
        "contour_area_half": float(abs(np.dot(ax[:-1], ay[1:]) - np.dot(ay[:-1], ax[1:])) / 2),
    }


def point_to_curve_squared(points, curve):
    start = curve[:-1]
    direction = np.diff(curve, axis=0)
    norm_sq = np.einsum("ij,ij->i", direction, direction)
    distances = np.empty(len(points))
    for first in range(0, len(points), 128):
        stop = first + 128
        offsets = points[first:stop, None] - start[None]
        t = np.clip(np.einsum("ijk,jk->ij", offsets, direction) / norm_sq, 0, 1)
        residual = offsets - t[:, :, None] * direction
        distances[first:stop] = np.min(np.einsum("ijk,ijk->ij", residual, residual), axis=1)
    return distances


def contour_errors(curve, reference, samples=1025):
    # Include original vertices in the maximum to retain sharp local features.
    _, curve_dense = resample_curve(curve, samples)
    _, ref_dense = resample_curve(reference, samples)
    forward = point_to_curve_squared(np.vstack([curve_dense, curve]), reference)
    backward = point_to_curve_squared(np.vstack([ref_dense, reference]), curve)
    return {
        "contour_hausdorff": math.sqrt(max(np.max(forward), np.max(backward))),
        "contour_rms": math.sqrt(0.5 * (np.mean(forward[:samples]) + np.mean(backward[:samples]))),
    }


def sharp_half_cap(theta_deg, half_area, sigma=1.0):
    theta = math.radians(theta_deg)
    if not 0 < theta < math.pi or half_area <= 0:
        raise ValueError("A circular cap needs positive area and a contact angle in (0, 180)")
    shape_factor = theta - math.sin(theta) * math.cos(theta)
    radius = math.sqrt(2 * half_area / shape_factor)
    a = radius * math.sin(theta)
    height = radius * (1 - math.cos(theta))
    angle = np.linspace(theta, 0, 1025)
    curve = np.column_stack([radius * np.sin(angle), radius * (np.cos(angle) - math.cos(theta))])
    return curve, {"contact_radius": a, "cap_height": height, "theta_cl_deg": theta_deg,
                   "energy": sigma * radius * shape_factor}


def csv_history(path):
    with path.open(newline="") as stream:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    if not rows:
        raise ValueError(f"Empty history: {path}")
    return rows


def snapshot_entries(run):
    collections = sorted(run.glob("ConservativeWetting*/ConservativeWetting*.pvd"))
    if len(collections) != 1:
        raise ValueError(f"Expected one ParaView collection under {run}, found {len(collections)}")
    pvd = collections[0]
    entries = {}
    for dataset in ET.parse(pvd).getroot().iter("DataSet"):
        time = float(dataset.attrib["timestep"])
        path = pvd.parent / dataset.attrib["file"]
        if time in entries and entries[time] != path:
            raise ValueError(f"Ambiguous snapshot at time {time} in {pvd}")
        entries[time] = path
    if not entries:
        raise ValueError(f"No saved snapshots in {pvd}")
    return sorted(entries.items())


@dataclass
class Snapshot:
    run: Path
    time: float
    grid: Q1Grid
    curve: np.ndarray
    history: dict
    initial_mass: float
    accepted: list

    def statistics(self, delta=None, fit_window=None):
        geometry = geometric_metrics(self.curve, self.grid.h, delta, fit_window)
        return {
            "run": self.run.name, "time": self.time, "nx": len(self.grid.x) - 1, "ny": len(self.grid.y) - 1,
            "h": self.grid.h, "mass_initial": self.initial_mass, "mass": self.history["mass"],
            "mass_drift": self.history["mass"] - self.initial_mass, "vtk_mass": self.grid.integral(),
            "max_mass_drift": max((abs(row["mass"] - self.initial_mass) for row in self.accepted), default=0.0),
            "energy": self.history["energy"], "energy_decay_rate": self.history["energy_decay_rate"],
            "max_energy_increase": max((max(0.0, -row["energy_decay_rate"] * row["dt"])
                                        for row in self.accepted), default=0.0),
            "dt_last": self.history["dt"],
            "dt_min": min((row["dt"] for row in self.accepted), default=0.0),
            "dt_max": max((row["dt"] for row in self.accepted), default=0.0),
            "dt_retries": int(sum(row["dt_retries"] for row in self.accepted)),
            **geometry,
        }


def read_snapshot(run, at="final"):
    run = Path(run)
    entries = snapshot_entries(run)
    if at == "final":
        time, path = entries[-1]
    else:
        target = float(at)
        time, path = min(entries, key=lambda entry: abs(entry[0] - target))
        if abs(time - target) > TIME_TOLERANCE:
            raise ValueError(f"No snapshot at t={target} in {run}; nearest collection time is {time}")
    history = csv_history(run / "time_specialized.csv")
    if at == "final" and abs(history[-1]["time"] - time) > TIME_TOLERANCE:
        raise ValueError(f"Final CSV time {history[-1]['time']} disagrees with snapshot time {time} in {run}")
    row = min(history, key=lambda entry: abs(entry["time"] - time))
    if abs(row["time"] - time) > TIME_TOLERANCE:
        raise ValueError(f"No accepted step corresponding to snapshot t={time} in {run}")
    mesh = pv.read(path)
    grid = Q1Grid.from_vtk(mesh)
    curve = half_drop_contour(mesh, grid)
    accepted = [item for item in history if 0 < item["time"] <= row["time"]]
    return Snapshot(run, row["time"], grid, curve, row, history[0]["mass"], accepted)


def equilibrium_row(case, run, theta_deg, delta, at="final", area=None, sigma=1.0, fit_window=None):
    snapshot = read_snapshot(run, at)
    row = {"case": case, "delta": delta, "theta_y_deg": theta_deg, **snapshot.statistics(delta, fit_window)}
    mass_area = (snapshot.grid.domain_area - snapshot.initial_mass) / 2
    reference_area = mass_area if area is None else area
    sharp_curve, sharp = sharp_half_cap(theta_deg, reference_area, sigma)
    row.update({"area_from_initial_mass": mass_area, "cap_reference_area_half": reference_area})
    for key, value in sharp.items():
        error = row[key] - value
        row.update({f"{key}_reference": value, f"{key}_signed_error": error, f"{key}_error": abs(error)})
    row.update(contour_errors(snapshot.curve, sharp_curve))
    return row


RATE_KEYS = ["phi_l2_error", "phi_h1_seminorm_error", "energy_error", "contact_radius_error",
             "cap_height_error", "theta_cl_deg_error", "contour_hausdorff", "contour_rms"]


def refinement_rows(case, reference, samples, at, axis, delta=None, fit_window=None):
    reference = read_snapshot(reference, at)
    test = [(float(size), read_snapshot(run, at)) for size, run in samples]
    test.sort(reverse=True, key=lambda item: item[0])
    if len({size for size, _ in test}) != len(test) or min(size for size, _ in test) <= 0:
        raise ValueError("Refinement sizes must be positive and unique")
    if fit_window is None:
        coarsest_h = max([reference.grid.h] + [sample.grid.h for _, sample in test])
        fit_window = max(4 * coarsest_h, 2 * delta if delta else 0)
    baseline = reference.statistics(delta, fit_window)
    rows = []
    for size, sample in test:
        if abs(sample.time - reference.time) > TIME_TOLERANCE:
            raise ValueError("Refinement comparisons require the same physical time")
        row = {"case": case, "axis": axis, "size": size, "delta": delta, "reference": reference.run.name,
               "reference_h": reference.grid.h, "reference_dt_max": baseline["dt_max"],
               **sample.statistics(delta, fit_window), **phase_errors(sample.grid, reference.grid),
               **contour_errors(sample.curve, reference.curve)}
        for key in ["energy", "contact_radius", "cap_height", "theta_cl_deg"]:
            row[f"{key}_error"] = abs(row[key] - baseline[key])
        for metric in RATE_KEYS:
            rate_key = metric.replace("_error", "") + "_rate"
            if rows and min(rows[-1][metric], row[metric]) > 0:
                row[rate_key] = math.log(rows[-1][metric] / row[metric]) / math.log(rows[-1]["size"] / size)
            else:
                row[rate_key] = None
        rows.append(row)
    return rows


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    equilibrium = commands.add_parser("equilibrium", help="Compare saved droplets to exact circular caps")
    equilibrium.add_argument("--run", action="append", nargs=4, required=True,
                             metavar=("CASE", "RUN_DIR", "YOUNG_DEG", "DELTA"))
    equilibrium.add_argument("--at", default="final", help="Snapshot time or final")
    equilibrium.add_argument("--area", type=float, help="Reference half-area; default uses initial diffuse mass")
    equilibrium.add_argument("--sigma", type=float, default=1.0)
    refine = commands.add_parser("rates", help="Compare fixed-time samples to a finer diffuse reference")
    refine.add_argument("--case", required=True)
    refine.add_argument("--axis", required=True, choices=["h", "dt"])
    refine.add_argument("--reference", type=Path, required=True)
    refine.add_argument("--sample", action="append", nargs=2, required=True, metavar=("SIZE", "RUN_DIR"))
    refine.add_argument("--at", required=True, type=float)
    refine.add_argument("--delta", type=float, required=True)
    series = commands.add_parser("series", help="Independently fit geometry at all saved snapshot times")
    series.add_argument("--run", type=Path, required=True)
    series.add_argument("--delta", type=float)
    for command in [equilibrium, refine, series]:
        command.add_argument("--fit-window", type=float, help="One-sided arclength for the tangent fit")
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "equilibrium":
        rows = [equilibrium_row(case, run, float(theta), float(delta), args.at, args.area, args.sigma,
                                args.fit_window) for case, run, theta, delta in args.run]
    elif args.command == "rates":
        rows = refinement_rows(args.case, args.reference, args.sample, args.at, args.axis, args.delta,
                               args.fit_window)
    else:
        rows = [read_snapshot(args.run, time).statistics(args.delta, args.fit_window)
                for time, _ in snapshot_entries(args.run)]
    write_rows(args.output, rows)


if __name__ == "__main__":
    main()
