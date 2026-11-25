#!/usr/bin/env python3
"""Summarize the two-dimensional discretization and equilibrium studies."""

from __future__ import annotations

import argparse
from functools import lru_cache
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullFormatter
import numpy as np

from postprocess_verification import (
    RATE_KEYS, contour_errors, csv_history, phase_errors, read_snapshot, sharp_half_cap, write_rows,
)
from run_metadata import load_completed_cases


STYLES = {"ac": ("#0072B2", "o", "-", "Allen–Cahn"), "ch": ("#D55E00", "D", "--", "Cahn–Hilliard")}


def loaded_cases(root, study, allow_incomplete=False):
    prefixes = {"all": ("space_", "time_", "equilibrium_"), "rates": ("space_", "time_"),
                "equilibrium": ("equilibrium_",)}
    return load_completed_cases(root, prefixes[study], allow_incomplete)


class VerifiedRuns:
    def __init__(self, root, specs):
        self.root = root
        self.specs = specs

    @lru_cache(maxsize=None)
    def snapshot(self, name):
        spec = self.specs[name]
        snap = read_snapshot(self.root / name, "final")
        if len(snap.grid.x) != spec["n"] + 1 or len(snap.grid.y) != spec["n"] + 1:
            raise ValueError(f"Snapshot grid disagrees with metadata: {name}")
        if spec["fixed_time"]:
            dt = spec["dt"]
            if abs(snap.time - spec["final_time"]) > 1e-8 or any(
                row["dt_retries"] or abs(dt - row["dt"]) > 1e-12 for row in snap.accepted
            ):
                raise ValueError(f"Rate run did not take the requested fixed time steps: {name}")
        return snap

    def differences(self, name, reference, fit_window):
        snap, ref = self.snapshot(name), self.snapshot(reference)
        if abs(snap.time - ref.time) > 1e-8:
            raise ValueError(f"Phase differences require common time: {name} and {reference}")
        spec = self.specs[name]
        stat = snap.statistics(spec["delta"], fit_window)
        reference_stat = ref.statistics(spec["delta"], fit_window)
        row = {"model": spec["model"], "delta": spec["delta"], "dt_requested": spec["dt"],
               "reference": reference, "reference_h": ref.grid.h, "reference_dt": self.specs[reference]["dt"],
               **stat, **phase_errors(snap.grid, ref.grid), **contour_errors(snap.curve, ref.curve)}
        for metric in ["energy", "contact_radius", "cap_height", "theta_cl_deg"]:
            row[metric + "_error"] = abs(stat[metric] - reference_stat[metric])
        return row


def rate_rows(runs, study, comparison):
    rows = []
    for model in STYLES:
        prefix = f"{study}_{model}_"
        names = [name for name in runs.specs if name.startswith(prefix)]
        if study == "time" and f"space_{model}_n128" in runs.specs:
            names.append(f"space_{model}_n128")  # This run has dt = 0.00125.
        key = (lambda name: 1 / runs.specs[name]["n"]) if study == "space" else (
            lambda name: runs.specs[name]["dt"])
        names.sort(reverse=True, key=key)
        model_rows = []
        for i, name in enumerate(names[:-1]):
            reference = names[-1] if comparison == "finest" else names[i + 1]
            if comparison == "pairwise" and not math.isclose(key(name), 2 * key(reference)):
                continue
            fit_window = 0.08  # Fixed arclength across all refinements at delta = 0.04.
            row = {"study": study, "comparison": comparison, "size": key(name),
                   **runs.differences(name, reference, fit_window)}
            for metric in RATE_KEYS:
                order_key = metric.replace("_error", "") + "_rate"
                if model_rows and min(model_rows[-1][metric], row[metric]) > 0:
                    previous = model_rows[-1]
                    row[order_key] = math.log(previous[metric] / row[metric]) / math.log(previous["size"] / row["size"])
                else:
                    row[order_key] = None
            model_rows.append(row)
        rows.extend(model_rows)
    return rows


def rate_controls(runs):
    rows = []
    for model in STYLES:
        for n in (128, 256):
            baseline = f"space_{model}_n{n}"
            control = f"space_check_{model}_n{n}"
            if baseline in runs.specs and control in runs.specs:
                rows.append({"control": "half_dt_at_fixed_h", **runs.differences(baseline, control, 0.08)})
        coarse = f"space_check_{model}_n128"
        fine = f"space_check_{model}_n256"
        if coarse in runs.specs and fine in runs.specs:
            rows.append({"control": "half_h_at_fixed_dt", **runs.differences(coarse, fine, 0.08)})
    return rows


def equilibrium_study(runs):
    rows = []
    for name, spec in runs.specs.items():
        if not name.startswith("equilibrium_"):
            continue
        kind = name.split("_")[1]
        if kind in STYLES:
            kind = "main"
        snap = runs.snapshot(name)
        delta = spec["delta"]
        # The same physical window at fixed delta for the main run and controls.
        stat = snap.statistics(delta, 2 * delta)
        area = (snap.grid.domain_area - stat["mass_initial"]) / 2
        curve, exact = sharp_half_cap(spec["theta_y"], area)
        row = {"model": spec["model"], "variant": kind, "delta": delta, "h_over_delta": stat["h"] / delta,
               "dt_requested": spec["dt"], "theta_y_deg": spec["theta_y"], "area_from_mass": area, **stat}
        for metric, value in exact.items():
            row[f"{metric}_reference"] = value
            row[f"{metric}_signed_error"] = row[metric] - value
            row[f"{metric}_error"] = abs(row[metric] - value)
        row.update(contour_errors(snap.curve, curve))
        tail = [item for item in csv_history(runs.root / name / "geometry.csv") if snap.time - 0.2 <= item["time"]]
        row["terminal_window_duration"] = tail[-1]["time"] - tail[0]["time"] if tail else float("nan")
        for metric in ["x_cl", "apex_y"]:
            value = [item[metric] for item in tail if math.isfinite(item[metric])]
            row[f"terminal_window_{metric}_range"] = max(value) - min(value) if len(value) >= 2 else float("nan")
        rows.append(row)

    indexed = {(row["model"], row["delta"], row["variant"]): row for row in rows}
    controls = []
    for row in rows:
        if row["variant"] == "main":
            continue
        base = indexed.get((row["model"], row["delta"], "main"))
        if base is None:
            continue
        comparison = {"model": row["model"], "variant": row["variant"], "delta": row["delta"],
                      "run": row["run"], "baseline": base["run"], "time": row["time"], "time_baseline": base["time"],
                      "h": row["h"], "h_baseline": base["h"], "dt_requested": row["dt_requested"],
                      "dt_requested_baseline": base["dt_requested"]}
        for metric in ["energy", "contact_radius", "cap_height", "theta_cl_deg"]:
            comparison[metric + "_change"] = row[metric] - base[metric]
        comparison.update(contour_errors(runs.snapshot(row["run"]).curve, runs.snapshot(base["run"]).curve))
        controls.append(comparison)
    agreement = []
    for base in rows:
        if base["model"] != "ac":
            continue
        other = indexed.get(("ch", base["delta"], base["variant"]))
        if other is None:
            continue
        comparison = {"variant": base["variant"], "delta": base["delta"], "h": base["h"],
                      "dt_requested": base["dt_requested"], "ac_run": base["run"], "ch_run": other["run"],
                      "ac_time": base["time"], "ch_time": other["time"]}
        for metric in ["energy", "contact_radius", "cap_height", "theta_cl_deg"]:
            comparison[metric + "_absolute_difference"] = abs(base[metric] - other[metric])
        ac = runs.snapshot(base["run"])
        ch = runs.snapshot(other["run"])
        comparison.update(phase_errors(ac.grid, ch.grid))
        comparison.update(contour_errors(ac.curve, ch.curve))
        agreement.append(comparison)
    return rows, controls, agreement


def style_plots():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9.5,
        "axes.linewidth": 0.7, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7, "pdf.fonttype": 42,
        "axes.axisbelow": True,
    })


def label_panel(ax, letter, title):
    ax.set_title(f"{letter}  {title}", loc="left", weight="semibold", pad=7)
    ax.grid(axis="y", which="major", color="#e9ecf0", linewidth=0.7)


def plot_rates(rows, path):
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.65), layout="constrained")
    panels = [
        ("space", "phi_l2_error", "Spatial refinement", r"$\|\phi_h-\phi_{h/2}\|_{L^2}$", 2),
        ("space", "phi_h1_seminorm_error", "Gradient refinement", r"$|\phi_h-\phi_{h/2}|_{H^1}$", 1),
        ("time", "phi_l2_error", "Time refinement", r"$\|\phi_{\Delta t}-\phi_{\Delta t/2}\|_{L^2}$", 1),
    ]
    for i, (ax, (study, metric, title, ylabel, guide)) in enumerate(zip(axes, panels)):
        selected = [row for row in rows if row["study"] == study]
        if not selected:
            continue
        for model, (color, marker, line, label) in STYLES.items():
            group = sorted((row for row in selected if row["model"] == model), key=lambda item: item["size"])
            if group:
                ax.loglog([row["size"] for row in group], [row[metric] for row in group],
                          color=color, marker=marker, linestyle=line, linewidth=1.1, markersize=4.2,
                          markerfacecolor=color if model == "ac" else "white", label=label)
        end = max(row["size"] for row in selected)
        start = min(row["size"] for row in selected)
        ymax = max(row[metric] for row in selected if row["size"] == end)
        guide_x = np.array([start, end])
        ax.loglog(guide_x, 1.8 * ymax * (guide_x / end) ** guide,
                  color="#646d75", linestyle=":", linewidth=1.0, label=f"Slope {guide}")
        if study == "space":
            sizes = sorted({row["size"] for row in selected})
            ax.set_xticks(sizes, [f"1/{round(1 / size)}" for size in sizes])
        else:
            sizes = sorted({row["size"] for row in selected})
            ax.set_xticks(sizes, [f"{size * 1e3:g}" for size in sizes], rotation=20)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(r"Mesh width $h$" if study == "space" else r"Time step $\Delta t$ ($10^{-3}$)")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, loc="upper left" if i == 0 else "best", handlelength=1.7)
        label_panel(ax, "abc"[i], title)
    fig.savefig(path)
    plt.close(fig)


def plot_equilibrium(rows, path):
    fig, axes = plt.subplots(2, 3, figsize=(7.05, 4.2), layout="constrained")
    panels = [
        ("contact_radius_error", "Footprint radius", r"$|a-a_*|$"),
        ("cap_height_error", "Cap height", r"$|H-H_*|$"),
        ("theta_cl_deg_error", "Contact angle", r"$|\theta_{\mathrm{CL}}-\theta_Y|$ (deg)"),
        ("contour_hausdorff", "Interface distance", "Sampled Hausdorff distance"),
        ("energy_signed_error", "Free energy", r"$E-E_*$"),
    ]
    for i, (ax, (metric, title, ylabel)) in enumerate(zip(axes.ravel(), panels)):
        ax.set_xscale("log")
        if metric == "energy_signed_error":
            ax.axhline(0, color="#69737c", linewidth=0.8)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        else:
            ax.set_yscale("log")
        for model, (color, marker, line, label) in STYLES.items():
            for variant in ["main", "mesh", "step", "fine"]:
                group = sorted((row for row in rows if row["model"] == model and row["variant"] == variant),
                               key=lambda item: item["delta"])
                if group:
                    ax.plot([row["delta"] for row in group], [row[metric] for row in group],
                            color=color, marker={"step": "+", "fine": "s"}.get(variant, marker),
                            linestyle={"main": line, "fine": "-."}.get(variant, ":"), linewidth=1.0,
                            markersize=5.1 if model == "ch" else 4.0,
                            markerfacecolor=color if variant == "main" and model == "ac" else "none",
                            zorder=4 if variant == "step" else 3)
        sizes = sorted({row["delta"] for row in rows})
        ax.set_xticks(sizes, [f"{size:g}" for size in sizes])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(r"Interface thickness $\delta$")
        ax.set_ylabel(ylabel)
        label_panel(ax, "abcde"[i], title)
    legend = axes.ravel()[-1]
    legend.axis("off")
    styles = [Line2D([], [], color=color, marker=marker, linestyle=line, label=label, markersize=4,
                     markerfacecolor=color if model == "ac" else "none")
              for model, (color, marker, line, label) in STYLES.items()]
    controls = [
        Line2D([], [], color="#545c61", marker="o", linestyle="-", label="Base resolution", markersize=4),
        Line2D([], [], color="#545c61", marker="o", markerfacecolor="none", linestyle=":",
               label=r"Half mesh width", markersize=4),
        Line2D([], [], color="#545c61", marker="+", linestyle="none", label=r"Half target $\Delta t$", markersize=5),
    ]
    if any(row["variant"] == "fine" for row in rows):
        controls.append(Line2D([], [], color="#545c61", marker="s", markerfacecolor="none", linestyle="-.",
                               label=r"Finer mesh, larger $\Delta t$", markersize=4))
    legend.legend(handles=styles + controls, loc="center left", frameon=False, handlelength=2.1, labelspacing=0.75)
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--runs", type=Path, help="Regenerated solver runs with saved field snapshots")
    inputs.add_argument("--reference", type=Path, help="Directory with the published plot-input CSVs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study", default="all", choices=["all", "rates", "equilibrium"])
    parser.add_argument("--allow-incomplete", action="store_true", help="Generate provisional figures from completed cases")
    args = parser.parse_args()
    if args.reference is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        style_plots()
        if args.study != "equilibrium":
            source = args.reference / "refinement_pairwise.csv"
            rows = np.atleast_1d(np.genfromtxt(source, names=True, delimiter=",", dtype=None, encoding=None))
            plot_rates(rows, args.output / "phase_field_refinement.pdf")
        if args.study != "rates":
            source = args.reference / "equilibrium_study.csv"
            rows = np.atleast_1d(np.genfromtxt(source, names=True, delimiter=",", dtype=None, encoding=None))
            plot_equilibrium(rows, args.output / "diffuse_interface_refinement.pdf")
        return
    specs, missing = loaded_cases(args.runs, args.study, args.allow_incomplete)
    if missing:
        print("Omitting incomplete cases: " + ", ".join(missing), flush=True)
    runs = VerifiedRuns(args.runs, specs)
    rate_sets = {comparison: [row for study in ["space", "time"] for row in rate_rows(runs, study, comparison)]
                 for comparison in ["finest", "pairwise"]}
    rate_checks = rate_controls(runs)
    equilibria, equilibrium_checks, agreement = equilibrium_study(runs)
    outputs = {"refinement_finest.csv": rate_sets["finest"], "refinement_pairwise.csv": rate_sets["pairwise"],
               "refinement_controls.csv": rate_checks, "equilibrium_study.csv": equilibria,
               "equilibrium_controls.csv": equilibrium_checks, "equilibrium_model_agreement.csv": agreement}
    for filename, rows in outputs.items():
        if rows:
            write_rows(args.output / filename, rows)
    style_plots()
    if rate_sets["pairwise"]:
        plot_rates(rate_sets["pairwise"], args.output / "phase_field_refinement.pdf")
    if equilibria:
        plot_equilibrium(equilibria, args.output / "diffuse_interface_refinement.pdf")


if __name__ == "__main__":
    main()
