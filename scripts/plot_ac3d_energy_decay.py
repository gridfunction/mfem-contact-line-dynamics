#!/usr/bin/env python3
"""Plot 3D Allen--Cahn energy and energy-decay histories."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_history(case_dir: Path) -> dict[str, np.ndarray]:
    rows = []
    with (case_dir / "time_specialized.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("newton_wall_time") is None:
                continue
            try:
                rows.append(
                    (
                        float(row["time"]),
                        float(row["energy"]),
                        float(row["energy_decay_rate"]),
                    )
                )
            except ValueError:
                continue
    data = np.asarray(rows, dtype=float)
    return {"time": data[:, 0], "energy": data[:, 1], "rate": data[:, 2]}


def make_plot(case45: Path, case135: Path, out_base: Path) -> None:
    histories = [
        (r"$\theta_Y=45^\circ$", read_history(case45), "tab:blue"),
        (r"$\theta_Y=135^\circ$", read_history(case135), "tab:orange"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    for row, (label, hist, color) in enumerate(histories):
        mask = hist["time"] > 0.0
        ax_energy, ax_rate = axes[row]
        ax_energy.plot(hist["time"][mask], hist["energy"][mask], color=color, lw=1.6)
        rate_mask = mask & np.isfinite(hist["rate"]) & (hist["rate"] > 0.0)
        ax_rate.semilogy(hist["time"][rate_mask], hist["rate"][rate_mask], color=color, lw=1.6)
        ax_rate.axhline(1.0e-7, color="0.25", lw=1.0, ls=":")
        ax_energy.set_title(label + ": free energy", fontsize=10)
        ax_rate.set_title(label + ": energy decay rate", fontsize=10)
        ax_energy.set_xlabel("time")
        ax_rate.set_xlabel("time")
        ax_energy.set_ylabel(r"$E(\phi_h^n)$")
        ax_rate.set_ylabel(r"$(E^{n-1}-E^n)/\Delta t$")
        for ax in (ax_energy, ax_rate):
            ax.grid(True, which="both", alpha=0.22, lw=0.55)
            ax.tick_params(labelsize=8.5)
    fig.subplots_adjust(left=0.095, right=0.98, bottom=0.105, top=0.93, wspace=0.34, hspace=0.46)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_base.with_suffix('.png')}")
    print(f"saved: {out_base.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case45", type=Path, required=True)
    parser.add_argument("--case135", type=Path, required=True)
    parser.add_argument("--out-base", type=Path, required=True)
    args = parser.parse_args()
    make_plot(args.case45, args.case135, args.out_base)


if __name__ == "__main__":
    main()
