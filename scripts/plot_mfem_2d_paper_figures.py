#!/usr/bin/env python3
"""Rebuild the Section 5 two-dimensional figures from MFEM AC/CH runs."""

from __future__ import annotations

import argparse
import csv
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PINNED_TIMES = [0.02, 0.2, 1.0]


def read_history(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty history file: {path}")
    data: dict[str, np.ndarray] = {}
    for key in rows[0].keys():
        values = []
        for row in rows:
            value = row[key]
            values.append(float(value) if value.lower() != "nan" else np.nan)
        data[key] = np.asarray(values)
    return data


def collection_name(model: str) -> str:
    return "ConservativeWettingAC" if model == "AC" else "ConservativeWettingCH"


def pvd_entries(run_dir: Path, model: str) -> list[tuple[float, Path]]:
    name = collection_name(model)
    pvd = run_dir / name / f"{name}.pvd"
    if not pvd.exists():
        raise FileNotFoundError(pvd)
    root = ET.parse(pvd).getroot()
    entries: list[tuple[float, Path]] = []
    for dataset in root.iter("DataSet"):
        time = float(dataset.attrib["timestep"])
        rel = Path(dataset.attrib["file"])
        entries.append((time, pvd.parent / rel))
    return sorted(entries, key=lambda item: item[0])


def file_at_time(run_dir: Path, model: str, target: float) -> Path:
    entries = pvd_entries(run_dir, model)
    times = np.asarray([item[0] for item in entries], dtype=float)
    idx = int(np.argmin(np.abs(times - target)))
    if abs(times[idx] - target) > 1.0e-8:
        raise FileNotFoundError(
            f"{model} run {run_dir} has no snapshot at t={target}; "
            f"nearest is t={times[idx]}"
        )
    return entries[idx][1]


def final_file(run_dir: Path, model: str) -> tuple[float, Path]:
    entries = pvd_entries(run_dir, model)
    if not entries:
        raise FileNotFoundError(f"no ParaView entries in {run_dir}")
    return entries[-1]


def contour_segments(pvtu_path: Path, level: float = 0.0) -> list[np.ndarray]:
    import pyvista as pv

    grid = pv.read(pvtu_path)
    contour = grid.contour([level], scalars="phi")
    points = np.asarray(contour.points)
    lines = np.asarray(contour.lines)
    raw_segments: list[np.ndarray] = []
    i = 0
    while i < len(lines):
        npts = int(lines[i])
        ids = lines[i + 1 : i + 1 + npts]
        if npts >= 2:
            seg = points[ids, :2]
            if np.nanmax(seg[:, 1]) > 1.0e-10:
                raw_segments.append(seg)
        i += npts + 1
    return stitch_segments(raw_segments)


def stitch_segments(raw_segments: list[np.ndarray], tol: float = 1.0e-10) -> list[np.ndarray]:
    """Join VTK two-point contour fragments into longer polylines."""
    edges: list[tuple[np.ndarray, np.ndarray]] = []
    for seg in raw_segments:
        for i in range(len(seg) - 1):
            edges.append((seg[i], seg[i + 1]))
    if not edges:
        return []

    def key(point: np.ndarray) -> tuple[int, int]:
        return (int(round(float(point[0]) / tol)), int(round(float(point[1]) / tol)))

    adjacency: dict[tuple[int, int], list[int]] = {}
    for idx, (p0, p1) in enumerate(edges):
        adjacency.setdefault(key(p0), []).append(idx)
        adjacency.setdefault(key(p1), []).append(idx)

    unused = set(range(len(edges)))
    polylines: list[np.ndarray] = []
    while unused:
        component_keys = [key(edges[next(iter(unused))][0])]
        seen_keys = set(component_keys)
        for vertex_key in component_keys:
            for edge_idx in adjacency.get(vertex_key, []):
                if edge_idx not in unused:
                    continue
                for point in edges[edge_idx]:
                    point_key = key(point)
                    if point_key not in seen_keys:
                        seen_keys.add(point_key)
                        component_keys.append(point_key)

        endpoints = [vertex_key for vertex_key in seen_keys if len(adjacency.get(vertex_key, [])) == 1]
        current_key = endpoints[0] if endpoints else next(iter(seen_keys))
        current_point = None
        polyline: list[np.ndarray] = []

        while True:
            next_edge = None
            for edge_idx in adjacency.get(current_key, []):
                if edge_idx in unused:
                    next_edge = edge_idx
                    break
            if next_edge is None:
                break
            unused.remove(next_edge)
            p0, p1 = edges[next_edge]
            if key(p0) == current_key:
                first, second = p0, p1
            else:
                first, second = p1, p0
            if current_point is None:
                polyline.append(first)
            polyline.append(second)
            current_point = second
            current_key = key(second)

        if len(polyline) >= 2:
            polylines.append(np.vstack(polyline))
    return polylines


def plot_segments(ax: plt.Axes, segments: list[np.ndarray], **kwargs) -> None:
    for j, seg in enumerate(segments):
        style = dict(kwargs)
        style["label"] = kwargs.get("label") if j == 0 else None
        ax.plot(seg[:, 0], seg[:, 1], **style)


def sharp_half_cap(theta_deg: float, half_area: float = 0.125) -> tuple[np.ndarray, np.ndarray]:
    theta = math.radians(theta_deg)
    full_area = 2.0 * half_area
    radius = math.sqrt(full_area / (theta - math.sin(theta) * math.cos(theta)))
    center_y = -radius * math.cos(theta)
    alpha = np.linspace(0.5 * math.pi - theta, 0.5 * math.pi + theta, 1200)
    x_full = radius * np.cos(alpha)
    y_full = center_y + radius * np.sin(alpha)
    mask = x_full >= -1.0e-12
    return np.maximum(x_full[mask], 0.0), np.maximum(y_full[mask], 0.0)


def plot_initial_rectangle(ax: plt.Axes) -> None:
    x = np.array([0.0, 0.5, 0.5, 0.0, 0.0])
    y = np.array([0.0, 0.0, 0.25, 0.25, 0.0])
    ax.plot(x, y, color="#c7352f", lw=1.45, label="initial profile")


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_interface_figure(theta: float, ac_dir: Path, ch_dir: Path, out_dir: Path,
                          *, paper_layout: bool = False) -> None:
    ac_final_time, ac_final_path = final_file(ac_dir, "AC")
    ch_final_time, ch_final_path = final_file(ch_dir, "CH")
    snapshots: list[tuple[str, Path, Path]] = []
    for time in PINNED_TIMES:
        if time < min(ac_final_time, ch_final_time) - 1.0e-10:
            snapshots.append((rf"$t={time:.2f}$", file_at_time(ac_dir, "AC", time), file_at_time(ch_dir, "CH", time)))
    snapshots.append(
        (
            rf"final: AC $t={ac_final_time:.2f}$, CH $t={ch_final_time:.2f}$",
            ac_final_path,
            ch_final_path,
        )
    )

    panels = [(title, contour_segments(ac), contour_segments(ch)) for title, ac, ch in snapshots]
    draw_interface_figure(theta, panels, out_dir, paper_layout=paper_layout)


def draw_interface_figure(theta: float, snapshots, out_dir: Path, *, paper_layout: bool = False) -> None:
    x_cap, y_cap = sharp_half_cap(theta)
    if paper_layout:
        fig, axes = plt.subplots(1, 4, figsize=(10.8, 2.54))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.05))
    for ax, (title, ac_segments, ch_segments) in zip(axes.ravel(), snapshots):
        plot_segments(ax, ac_segments, color="tab:blue", lw=1.6, label="AC")
        plot_segments(
            ax,
            ch_segments,
            color="tab:orange",
            lw=2.15,
            ls=(0, (3.5, 2.0)),
            label="CH",
        )
        ax.plot(x_cap, y_cap, color="black", lw=1.35, label="stationary sharp interface")
        plot_initial_rectangle(ax)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0.0, 0.85)
        ax.set_ylim(0.0, 0.60)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.18, lw=0.55)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=8.5)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=9.5,
        handlelength=2.8,
    )
    if paper_layout:
        fig.subplots_adjust(left=0.035, right=0.995, bottom=0.11, top=0.76, wspace=0.16)
    else:
        fig.subplots_adjust(left=0.055, right=0.99, bottom=0.065, top=0.855, wspace=0.08, hspace=0.30)
    save_figure(fig, out_dir / f"ac_ch_theta{int(theta)}_interfaces")


def make_energy_figure(cases: dict[float, dict[str, Path]], out_dir: Path) -> None:
    draw_energy_figure(read_case_histories(cases), out_dir)


def read_case_histories(cases):
    return {angle: {model: read_history(path / "time_specialized.csv") for model, path in runs.items()}
            for angle, runs in cases.items()}


def draw_energy_figure(histories, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.5), constrained_layout=True)
    colors = {"AC": "tab:blue", "CH": "tab:orange"}
    linestyles = {"AC": "-", "CH": (0, (3.5, 2.0))}
    for row, theta in enumerate((45.0, 135.0)):
        ax_energy, ax_rate = axes[row]
        for model in ("AC", "CH"):
            h = histories[theta][model]
            time = h["time"]
            energy = h["energy"]
            rate = h["energy_decay_rate"]
            mask = np.isfinite(rate) & (rate > 0.0)
            label = model
            ax_energy.plot(time, energy, color=colors[model], ls=linestyles[model], lw=1.5, label=label)
            ax_rate.semilogy(
                time[mask],
                rate[mask],
                color=colors[model],
                ls=linestyles[model],
                lw=1.5,
                label=label,
            )
        ax_energy.set_title(rf"$\theta_Y={theta:.0f}^\circ$: total energy", fontsize=10)
        ax_rate.set_title(rf"$\theta_Y={theta:.0f}^\circ$: energy decay rate", fontsize=10)
        ax_energy.set_xlabel("time")
        ax_rate.set_xlabel("time")
        ax_energy.set_ylabel(r"$E(\phi_h^n)$")
        ax_rate.set_ylabel(r"$(E^{n-1}-E^n)/\Delta t$")
        ax_rate.axhline(1.0e-7, color="0.35", ls=":", lw=1.0)
        for ax in (ax_energy, ax_rate):
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(loc="best", fontsize=8, frameon=False)
    save_figure(fig, out_dir / "ac_ch_energy_decay_angles")


def make_mass_figure(cases: dict[float, dict[str, Path]], out_dir: Path) -> None:
    draw_mass_figure(read_case_histories(cases), out_dir)


def draw_mass_figure(histories, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.2), constrained_layout=True)
    colors = {"AC": "tab:blue", "CH": "tab:orange"}
    for row, theta in enumerate((45.0, 135.0)):
        for col, model in enumerate(("AC", "CH")):
            ax = axes[row, col]
            h = histories[theta][model]
            time = h["time"]
            mass = h["mass"]
            drift = mass - mass[0]
            ax.plot(time, drift, color=colors[model], lw=1.4)
            ax.set_title(rf"{model}, $\theta_Y={theta:.0f}^\circ$", fontsize=10)
            ax.set_xlabel("time")
            ax.set_ylabel(r"$(\phi_h^n-\phi_h^0,1)_\Omega$")
            ax.grid(True, alpha=0.25)
            max_abs = max(float(np.max(np.abs(drift))), 1.0e-14)
            ax.set_ylim(-1.15 * max_abs, 1.15 * max_abs)
    save_figure(fig, out_dir / "ac_ch_mass_evolution_angles")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ac45-dir", type=Path, required=True)
    parser.add_argument("--ch45-dir", type=Path, required=True)
    parser.add_argument("--ac135-dir", type=Path, required=True)
    parser.add_argument("--ch135-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("figs"))
    parser.add_argument("--paper-layout", action="store_true", help="Use the manuscript's one-row snapshot layout.")
    args = parser.parse_args()

    cases = {
        45.0: {"AC": args.ac45_dir, "CH": args.ch45_dir},
        135.0: {"AC": args.ac135_dir, "CH": args.ch135_dir},
    }
    make_interface_figure(45.0, args.ac45_dir, args.ch45_dir, args.out_dir, paper_layout=args.paper_layout)
    make_interface_figure(135.0, args.ac135_dir, args.ch135_dir, args.out_dir, paper_layout=args.paper_layout)
    make_energy_figure(cases, args.out_dir)
    make_mass_figure(cases, args.out_dir)


if __name__ == "__main__":
    main()
