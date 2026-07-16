#!/usr/bin/env python3
"""Plot two reconstructed 3D Allen--Cahn interfaces against sharp caps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_ac3d_final_surface import (  # noqa: E402
    contact_rim,
    exact_cap_surface,
    extract_y0_grid,
    final_step_time_mass,
    read_pvtu,
    revolve_curve,
    sharp_cap_parameters,
    step_for_time,
    step_time_mass,
    zero_contour_branch,
)


def panel(ax, case_dir: Path, theta: float, step: int | None, title_prefix: str) -> None:
    if step is None:
        step, time, mass = final_step_time_mass(case_dir)
    else:
        time, mass = step_time_mass(case_dir, step)
    radius, center_z, contact_radius, _ = sharp_cap_parameters(theta, mass)
    grid, phi = read_pvtu(case_dir, step)
    xs, zs, field = extract_y0_grid(grid, phi)
    r_num, z_num = zero_contour_branch(xs, zs, field)
    xn, yn, zn = revolve_curve(r_num, z_num)
    xe, ye, ze = exact_cap_surface(radius, center_z, contact_radius)
    rim_x, rim_y, rim_z = contact_rim(contact_radius)

    ax.plot_surface(xe, ye, ze, color="0.74", edgecolor="black", linewidth=0.11,
                    alpha=0.38, rstride=10, cstride=14, shade=False)
    ax.plot(rim_x, rim_y, rim_z, color="black", lw=0.9, alpha=0.85)
    ax.plot_surface(xn, yn, zn, color="tab:blue", edgecolor="none",
                    alpha=0.60, rstride=2, cstride=2, shade=True)

    lim = max(0.82, 1.05 * contact_radius)
    zmax = 0.6
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0.0, zmax)
    # Match physical axis scales so spherical caps are not visually stretched.
    ax.set_box_aspect((2.0 * lim, 2.0 * lim, zmax))
    ax.view_init(elev=24, azim=-55)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_title(rf"{title_prefix}: $\theta_Y={theta:g}^\circ$, $t={time:.2f}$", fontsize=9, pad=7)
    ax.tick_params(labelsize=7)
    ax.grid(False)


def make_plot(case45: Path, case135: Path, step135: int | None, out_base: Path) -> None:
    fig = plt.figure(figsize=(7.2, 3.55))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    panel(ax1, case45, 45.0, None, "final")
    panel(ax2, case135, 135.0, step135, "final" if step135 is None else "snapshot")
    fig.legend(
        handles=[
            Patch(facecolor="tab:blue", edgecolor="none", alpha=0.66, label="AC"),
            Patch(facecolor="0.74", edgecolor="black", alpha=0.38, label="stationary sharp interface"),
        ],
        frameon=False,
        loc="upper center",
        ncol=2,
        fontsize=9,
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.84, wspace=0.00)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out_base.with_suffix('.png')}")
    print(f"saved: {out_base.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case45", type=Path, required=True)
    parser.add_argument("--case135", type=Path, required=True)
    parser.add_argument("--step135", type=int, default=None)
    parser.add_argument("--time135", type=float, default=None)
    parser.add_argument("--out-base", type=Path, required=True)
    args = parser.parse_args()
    step135 = args.step135
    if args.time135 is not None:
        step135 = step_for_time(args.case135, args.time135)
    make_plot(args.case45, args.case135, step135, args.out_base)


if __name__ == "__main__":
    main()
