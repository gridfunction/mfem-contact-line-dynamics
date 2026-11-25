#!/usr/bin/env python3
"""Revolve the final x-z zero contour and compare with the sharp spherical cap."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def history_rows(case_dir: Path) -> list[dict[str, str]]:
    with (case_dir / "time_specialized.csv").open(newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            if row.get("newton_wall_time") is None:
                continue
            rows.append(row)
        return rows


def final_step_time_mass(case_dir: Path) -> tuple[int, float, float]:
    rows = history_rows(case_dir)
    if not rows:
        raise RuntimeError(f"No history rows in {case_dir}")
    row = rows[-1]
    return int(row["step"]), float(row["time"]), float(row["mass"])


def step_time_mass(case_dir: Path, step: int) -> tuple[float, float]:
    rows = history_rows(case_dir)
    for row in rows:
        if int(row["step"]) == step:
            return float(row["time"]), float(row["mass"])
    if rows:
        last = rows[-1]
        time = float(last["time"]) + (step - int(last["step"])) * float(last["dt"])
        return time, float(last["mass"])
    return float(step) * 0.01, 0.875


def step_for_time(case_dir: Path, target: float) -> int:
    rows = history_rows(case_dir)
    row = min(rows, key=lambda r: abs(float(r["time"]) - target))
    return int(row["step"])


def sharp_cap_parameters(theta_deg: float, mass: float, domain_volume: float = 1.0):
    quarter_liquid_volume = 0.5 * (domain_volume - mass)
    full_volume = 4.0 * quarter_liquid_volume
    theta = math.radians(theta_deg)
    c = math.cos(theta)
    radius = (3.0 * full_volume / (math.pi * (1.0 - c) ** 2 * (2.0 + c))) ** (1.0 / 3.0)
    center_z = -radius * c
    contact_radius = radius * math.sin(theta)
    return radius, center_z, contact_radius, full_volume


def read_pvtu(case_dir: Path, step: int):
    import vtk

    pvtu = case_dir / "ConservativeWettingAC3D" / f"Cycle{step:06d}" / "data.pvtu"
    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(str(pvtu))
    reader.Update()
    grid = reader.GetOutput()
    phi = grid.GetPointData().GetArray("phi")
    if phi is None:
        raise RuntimeError(f"No point-data array named 'phi' in {pvtu}")
    return grid, phi


def extract_y0_grid(grid, phi, y_tol: float = 1.0e-10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values: dict[tuple[float, float], list[float]] = {}
    y_min = min(grid.GetPoint(i)[1] for i in range(grid.GetNumberOfPoints()))
    for i in range(grid.GetNumberOfPoints()):
        x, y, z = grid.GetPoint(i)
        if abs(y - y_min) <= y_tol:
            key = (round(x, 12), round(z, 12))
            values.setdefault(key, []).append(phi.GetValue(i))
    if not values:
        raise RuntimeError("Could not extract the y=0 symmetry slice.")

    xs = sorted({key[0] for key in values})
    zs = sorted({key[1] for key in values})
    x_index = {x: i for i, x in enumerate(xs)}
    z_index = {z: i for i, z in enumerate(zs)}
    field = np.empty((len(zs), len(xs)))
    for (x, z), vals in values.items():
        field[z_index[z], x_index[x]] = float(np.mean(vals))

    return np.asarray(xs), np.asarray(zs), field


def zero_contour_branch(xs: np.ndarray, zs: np.ndarray, field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fig, ax = plt.subplots()
    cs = ax.contour(xs, zs, field, levels=[0.0])
    segments = cs.allsegs[0]
    plt.close(fig)
    if not segments:
        raise RuntimeError("No phi=0 contour found on the y=0 slice.")
    seg = max(segments, key=len)
    r = np.maximum(seg[:, 0], 0.0)
    z = seg[:, 1]
    keep = (r >= -1.0e-12) & (z >= -1.0e-12)
    r = r[keep]
    z = z[keep]
    return r, z


def revolve_curve(r: np.ndarray, z: np.ndarray, ntheta: int = 180):
    theta = np.linspace(0.0, 2.0 * math.pi, ntheta)
    rr, tt = np.meshgrid(r, theta)
    zz, _ = np.meshgrid(z, theta)
    return rr * np.cos(tt), rr * np.sin(tt), zz


def exact_cap_surface(radius: float, center_z: float, contact_radius: float):
    theta_y = math.acos(-center_z / radius)
    alpha = np.linspace(0.5 * math.pi - theta_y, 0.5 * math.pi, 140)
    theta = np.linspace(0.0, 2.0 * math.pi, 180)
    aa, tt = np.meshgrid(alpha, theta)
    rr = radius * np.cos(aa)
    x = rr * np.cos(tt)
    y = rr * np.sin(tt)
    z = center_z + radius * np.sin(aa)
    return x, y, z


def contact_rim(contact_radius: float):
    theta = np.linspace(0.0, 2.0 * math.pi, 360)
    x = contact_radius * np.cos(theta)
    y = contact_radius * np.sin(theta)
    z = np.zeros_like(theta)
    return x, y, z


def make_plot(case_dir: Path, theta: float, out_base: Path,
              step_override: int | None = None) -> None:
    if step_override is None:
        step, time, mass = final_step_time_mass(case_dir)
    else:
        step = step_override
        time, mass = step_time_mass(case_dir, step)
    radius, center_z, contact_radius, full_volume = sharp_cap_parameters(theta, mass)
    grid, phi = read_pvtu(case_dir, step)
    xs, zs, field = extract_y0_grid(grid, phi)
    r_num, z_num = zero_contour_branch(xs, zs, field)
    xn, yn, zn = revolve_curve(r_num, z_num)
    xe, ye, ze = exact_cap_surface(radius, center_z, contact_radius)
    rim_x, rim_y, rim_z = contact_rim(contact_radius)

    fig = plt.figure(figsize=(5.8, 4.7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xe, ye, ze, color="0.74", edgecolor="black", linewidth=0.12,
                    alpha=0.38, rstride=10, cstride=14, shade=False)
    ax.plot(rim_x, rim_y, rim_z, color="black", lw=0.95, alpha=0.85)
    ax.plot_surface(xn, yn, zn, color="tab:blue", edgecolor="none",
                    alpha=0.60, rstride=2, cstride=2, shade=True)

    lim = max(0.82, 1.05 * contact_radius)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0.0, 0.6)
    ax.set_box_aspect((1.0, 1.0, 0.45))
    ax.view_init(elev=24, azim=-55)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_title(rf"$\theta_Y={theta:g}^\circ$, final $t={time:.2f}$", pad=8, fontsize=11)
    ax.tick_params(labelsize=8)
    ax.grid(False)
    ax.legend(
        handles=[
            Patch(facecolor="tab:blue", edgecolor="none", alpha=0.66, label="AC"),
            Patch(facecolor="0.74", edgecolor="black", alpha=0.38, label="sharp interface"),
        ],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        fontsize=9,
    )

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"step={step} time={time:.12g} mass={mass:.12g} full_volume={full_volume:.12g}")
    print(f"sharp cap R={radius:.12g} z0={center_z:.12g} contact_radius={contact_radius:.12g}")
    print(f"saved: {out_base.with_suffix('.png')}")
    print(f"saved: {out_base.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--theta", type=float, required=True)
    parser.add_argument("--out-base", type=Path, required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--time", type=float, default=None)
    args = parser.parse_args()
    step = args.step
    if args.time is not None:
        step = step_for_time(args.case_dir, args.time)
    make_plot(args.case_dir, args.theta, args.out_base, step)


if __name__ == "__main__":
    main()
