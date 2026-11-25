#!/usr/bin/env python3
"""Redraw all eight original figures from archived histories and contours."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib.contour import ContourSet
import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))

import plot_ac3d_energy_decay as energy3d
import plot_ac3d_surface_pair as surface3d
import plot_ac3d_symmetric_xz_slice as slice3d
import plot_mfem_2d_paper_figures as plots2d


ANGLES = (45, 135)
CASES = tuple(f"{model}_theta{theta}" for model in ("ac2d", "ch2d", "ac3d") for theta in ANGLES)


def read_published_histories(path: Path) -> dict:
    columns = ("time", "energy", "mass", "energy_decay_rate")
    by_case = {case: {key: [] for key in columns} for case in CASES}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            for key in columns:
                by_case[row["case"]][key].append(float(row[key]))
    if any(not data["time"] for data in by_case.values()):
        raise ValueError(f"Incomplete original histories: {path}")
    return {case: {key: np.asarray(values) for key, values in history.items()}
            for case, history in by_case.items()}


def read_geometry(data_dir: Path) -> dict:
    geometry = {}
    for case in CASES:
        path = data_dir / f"{case}.json"
        data = json.loads(path.read_text())
        if data["case"] != case or data["version"] != 1 or len(data["snapshots"]) != 4:
            raise ValueError(f"Unexpected geometry metadata: {path}")
        for snapshot in data["snapshots"]:
            snapshot["segments"] = [np.asarray(seg, dtype=float) for seg in snapshot["segments"]]
        geometry[case] = data
    return geometry


def draw_2d(geometry, histories, output: Path) -> None:
    history_by_angle = {}
    for theta in ANGLES:
        ac_name, ch_name = f"ac2d_theta{theta}", f"ch2d_theta{theta}"
        phase_ac = geometry[ac_name]["snapshots"]
        phase_ch = geometry[ch_name]["snapshots"]
        snapshots = []
        for ac, ch in zip(phase_ac[:-1], phase_ch[:-1]):
            if ac["time"] != ch["time"]:
                raise ValueError(f"Mismatched 2D snapshot times at Young angle {theta}")
            snapshots.append((rf"$t={ac['time']:.2f}$", ac["segments"], ch["segments"]))
        ac, ch = phase_ac[-1], phase_ch[-1]
        title = rf"final: AC $t={ac['time']:.2f}$, CH $t={ch['time']:.2f}$"
        snapshots.append((title, ac["segments"], ch["segments"]))
        plots2d.draw_interface_figure(theta, snapshots, output, paper_layout=True)
        history_by_angle[theta] = {"AC": histories[ac_name], "CH": histories[ch_name]}
    plots2d.draw_energy_figure(history_by_angle, output)
    plots2d.draw_mass_figure(history_by_angle, output)


def draw_archived_zero(ax, segments, quarter: bool) -> None:
    if not quarter:
        raise ValueError("Archived contour paths cover the simulated quarter domain")
    ContourSet(ax, [0.0], [segments], colors="tab:blue", linewidths=1.6)


def draw_3d(geometry, histories, output: Path) -> None:
    energy_histories = []
    meridians = []
    for theta in ANGLES:
        name = f"ac3d_theta{theta}"
        data = geometry[name]
        slices = [(record["time"], record["mass"], record["segments"]) for record in data["snapshots"]]
        slice3d.draw_times_plot(theta, slices, draw_archived_zero,
                                output / f"ac3d_theta{theta}_xz_quarter_cut_times", quarter=True, paper_layout=True)
        hist = histories[name]
        energy_histories.append({"time": hist["time"], "energy": hist["energy"], "rate": hist["energy_decay_rate"]})
        last = data["snapshots"][-1]
        radius, height = np.asarray(data["terminal_meridian"], dtype=float).T
        meridians.append((last["time"], last["mass"], radius, height))
    energy3d.draw_energy_figure(*energy_histories, output / "ac3d_energy_decay_angles")
    surface3d.draw_surface_pair(*meridians, output / "ac3d_surface_pair_theta45_theta135")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("figures/original"))
    args = parser.parse_args()
    geometry = read_geometry(HERE / "data")
    histories = read_published_histories(HERE.parent / "original_histories.csv")
    draw_2d(geometry, histories, args.output)
    draw_3d(geometry, histories, args.output)


if __name__ == "__main__":
    main()
