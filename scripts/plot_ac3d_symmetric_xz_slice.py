#!/usr/bin/env python3
"""Plot a full x-z droplet cross-section from a symmetric 3D quarter-domain run."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def history_rows(case_dir: Path) -> list[dict[str, str]]:
    history = case_dir / "time_specialized.csv"
    with history.open(newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            if row.get("newton_wall_time") is None:
                continue
            rows.append(row)
        return rows


def final_step_time_mass(case_dir: Path) -> tuple[int, float, float]:
    rows = history_rows(case_dir)
    if not rows:
        raise RuntimeError(f"No rows found in {history}")
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


def read_pvtu(case_dir: Path, step: int):
    import vtk

    pvtu = case_dir / "ConservativeWettingAC3D" / f"Cycle{step:06d}" / "data.pvtu"
    if not pvtu.exists():
        pvtu = case_dir / "ConservativeWettingAC" / f"Cycle{step:06d}" / "data.pvtu"
    if not pvtu.exists():
        raise FileNotFoundError(f"Could not find final PVTU for step {step} under {case_dir}")

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
        raise RuntimeError("Could not extract a y=0 symmetry slice.")

    xs = sorted({key[0] for key in values})
    zs = sorted({key[1] for key in values})
    x_index = {x: i for i, x in enumerate(xs)}
    z_index = {z: i for i, z in enumerate(zs)}
    field = np.empty((len(zs), len(xs)))
    for (x, z), vals in values.items():
        field[z_index[z], x_index[x]] = float(np.mean(vals))

    return np.asarray(xs), np.asarray(zs), field


def mirror_x(xs: np.ndarray, field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    neg_x = -xs[:0:-1]
    full_x = np.concatenate([neg_x, xs])
    full_field = np.concatenate([field[:, :0:-1], field], axis=1)
    return full_x, full_field


def sharp_cap(theta_deg: float, mass: float, domain_volume: float = 1.0, n: int = 600):
    quarter_liquid_volume = 0.5 * (domain_volume - mass)
    full_volume = 4.0 * quarter_liquid_volume
    theta = math.radians(theta_deg)
    c = math.cos(theta)
    factor = (1.0 - c) ** 2 * (2.0 + c)
    radius = (3.0 * full_volume / (math.pi * factor)) ** (1.0 / 3.0)
    center_z = -radius * c
    alpha = np.linspace(0.5 * math.pi - theta, 0.5 * math.pi + theta, n)
    x = radius * np.cos(alpha)
    z = np.maximum(center_z + radius * np.sin(alpha), 0.0)
    return x, z, radius, center_z, full_volume


def sharp_cap_quarter(theta_deg: float, mass: float, domain_volume: float = 1.0, n: int = 600):
    quarter_liquid_volume = 0.5 * (domain_volume - mass)
    full_volume = 4.0 * quarter_liquid_volume
    theta = math.radians(theta_deg)
    c = math.cos(theta)
    factor = (1.0 - c) ** 2 * (2.0 + c)
    radius = (3.0 * full_volume / (math.pi * factor)) ** (1.0 / 3.0)
    center_z = -radius * c
    alpha = np.linspace(0.5 * math.pi - theta, 0.5 * math.pi, n)
    x = radius * np.cos(alpha)
    z = np.maximum(center_z + radius * np.sin(alpha), 0.0)
    return x, z, radius, center_z, full_volume


def make_plot(case_dir: Path, theta: float, out_base: Path, step_override: int | None = None) -> None:
    if step_override is None:
        step, time, mass = final_step_time_mass(case_dir)
    else:
        step = step_override
        time, mass = step_time_mass(case_dir, step)
    grid, phi = read_pvtu(case_dir, step)
    xs, zs, quarter_phi = extract_y0_grid(grid, phi)
    full_x, full_phi = mirror_x(xs, quarter_phi)
    x_grid, z_grid = np.meshgrid(full_x, zs)

    sharp_x, sharp_z, radius, center_z, full_volume = sharp_cap(theta, mass)

    fig, ax = plt.subplots(figsize=(6.0, 3.8), constrained_layout=True)
    levels = np.linspace(-1.0, 1.0, 41)
    im = ax.contourf(x_grid, z_grid, full_phi, levels=levels, cmap="RdBu_r", extend="both")
    ax.contour(x_grid, z_grid, full_phi, levels=[0.0], colors="tab:blue", linewidths=1.8)
    ax.plot(sharp_x, sharp_z, color="black", lw=1.5, label="sharp cap")
    ax.axhline(0.0, color="0.25", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(0.0, 0.75)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(rf"$\theta_Y={theta:g}^\circ$, $t={time:.2f}$")
    ax.legend(frameon=False, loc="upper right")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(r"$\phi$")

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=300)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)

    print(f"case: {case_dir}")
    print(f"step={step} time={time:.12g}")
    print(f"mass={mass:.12g} full sharp volume={full_volume:.12g}")
    print(f"sharp cap: R={radius:.12g} z0={center_z:.12g}")
    print(f"saved: {out_base.with_suffix('.png')}")
    print(f"saved: {out_base.with_suffix('.pdf')}")


def positive_sharp_branch(sharp_x: np.ndarray, sharp_z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = sharp_x >= -1.0e-12
    x = sharp_x[mask]
    z = sharp_z[mask]
    order = np.argsort(x)
    return x[order], z[order]


def plot_initial_quarter_rectangle(ax: plt.Axes) -> None:
    x = np.array([0.0, 0.5, 0.5, 0.0, 0.0])
    z = np.array([0.0, 0.0, 0.25, 0.25, 0.0])
    ax.plot(x, z, color="#c7352f", lw=1.45)


def make_times_plot(case_dir: Path, theta: float, times: list[float], out_base: Path,
                    quarter: bool = False, append_last: bool = True,
                    steps_override: list[int] | None = None, *, paper_layout: bool = False) -> None:
    steps = steps_override if steps_override is not None else [step_for_time(case_dir, t) for t in times]
    if append_last:
        final_step, _, _ = final_step_time_mass(case_dir)
        if final_step not in steps:
            steps.append(final_step)

    snapshots = []
    for step in steps[:4]:
        time, mass = step_time_mass(case_dir, step)
        grid, phi = read_pvtu(case_dir, step)
        snapshots.append((time, mass, extract_y0_grid(grid, phi)))
    draw_times_plot(theta, snapshots, draw_slice_zero, out_base, quarter, paper_layout=paper_layout)
    print(f"steps: {steps}")


def draw_slice_zero(ax, data, quarter: bool) -> None:
    xs, zs, field = data
    if not quarter:
        xs, field = mirror_x(xs, field)
    x_grid, z_grid = np.meshgrid(xs, zs)
    ax.contour(x_grid, z_grid, field, levels=[0.0], colors="tab:blue", linewidths=1.6)


def draw_times_plot(theta: float, snapshots, draw_zero, out_base: Path, quarter: bool = False,
                    *, paper_layout: bool = False) -> None:

    if paper_layout:
        fig, axes = plt.subplots(1, 4, figsize=(10.8, 2.65))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.25))
    axes = axes.ravel()
    handles = [
        Line2D([0], [0], color="tab:blue", lw=1.6),
        Line2D([0], [0], color="#c7352f", lw=1.45),
        Line2D([0], [0], color="black", lw=1.35),
    ]
    labels = ["AC", "initial profile", "stationary sharp interface"]
    for ax, (time, mass, data) in zip(axes, snapshots):
        if quarter:
            sharp_x, sharp_z, _, _, _ = sharp_cap_quarter(theta, mass)
        else:
            sharp_x, sharp_z, _, _, _ = sharp_cap(theta, mass)

        draw_zero(ax, data, quarter)
        if quarter:
            plot_initial_quarter_rectangle(ax)
        ax.plot(sharp_x, sharp_z, color="black", lw=1.35)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0.0, 0.85) if quarter else ax.set_xlim(-0.85, 0.85)
        ax.set_ylim(0.0, 0.75)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(rf"$t={time:.2f}$", fontsize=10)
        ax.grid(True, alpha=0.18, lw=0.55)
        ax.tick_params(labelsize=8.5)
    for ax in axes[len(snapshots):]:
        ax.axis("off")
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               fontsize=9.5, handlelength=2.8)
    if paper_layout:
        fig.subplots_adjust(left=0.035, right=0.995, bottom=0.11, top=0.76, wspace=0.16)
    else:
        fig.subplots_adjust(left=0.055, right=0.99, bottom=0.065, top=0.855, wspace=0.08, hspace=0.30)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_base.with_suffix('.png')}")
    print(f"saved: {out_base.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--theta", type=float, required=True)
    parser.add_argument("--out-base", type=Path, required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--times", type=float, nargs="*", default=None)
    parser.add_argument("--steps", type=int, nargs="*", default=None)
    parser.add_argument("--quarter", action="store_true",
                        help="Plot only the simulated quarter-domain x-z cutline.")
    parser.add_argument("--no-last", action="store_true",
                        help="Do not append the latest/final saved time to a time-series panel.")
    parser.add_argument("--paper-layout", action="store_true", help="Use the manuscript's one-row snapshot layout.")
    args = parser.parse_args()
    if args.times or args.steps:
        make_times_plot(args.case_dir, args.theta, args.times or [], args.out_base,
                        args.quarter, append_last=not args.no_last,
                        steps_override=args.steps, paper_layout=args.paper_layout)
    else:
        make_plot(args.case_dir, args.theta, args.out_base, args.step)


if __name__ == "__main__":
    main()
