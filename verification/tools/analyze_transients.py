"""Compare diffuse contact-line trajectories with a refined sharp AC reference."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.interpolate import PchipInterpolator

from run_metadata import load_completed_cases


RUN_FIELDS = ("name", "model", "n", "delta", "dt", "theta0", "zeta")


def load_csv(path):
    return np.atleast_1d(np.genfromtxt(path, names=True, delimiter=",", encoding=None, dtype=None))


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def local_velocity(time, values, sample_time, half_width=.0125):
    """Differentiate positions with a local quadratic in physical time."""
    velocity = np.empty_like(sample_time)
    for i, center in enumerate(sample_time):
        offset = time - center
        count = min(5, len(time))
        width = max(half_width, 1.01 * np.partition(abs(offset), count - 1)[count - 1])
        window = abs(offset) < width
        scaled = offset[window] / width
        weights = np.sqrt(np.maximum(0, 1 - abs(scaled)**3)**3)
        slope = np.polynomial.polynomial.polyfit(scaled, values[window], 2, w=weights)[1]
        velocity[i] = slope / width
    return velocity


def read_diffuse(path, run):
    fields = load_csv(path / "geometry.csv")
    for name in ("time", "x_cl", "theta_cl_deg", "apex_y", "wall_profile_scale"):
        if not np.all(np.isfinite(fields[name])):
            raise ValueError(f"Nonfinite {name} in {path}")
    if np.min(np.diff(fields["time"])) <= 0 or abs(fields["time"][-1] - run["final_time"]) > 1e-10:
        raise ValueError(f"Incomplete/non-increasing history: {path}")
    sample_time = np.round(np.arange(0, run["final_time"] + .0001, .002), 10)
    data = {"time": sample_time}
    for source, target in (("x_cl", "a"), ("theta_cl_deg", "theta"), ("apex_y", "height"),
                           ("wall_profile_scale", "profile_scale")):
        data[target] = PchipInterpolator(fields["time"], fields[source])(sample_time)
    data["v_cl"] = local_velocity(fields["time"], fields["x_cl"], sample_time)
    data["v_apex"] = local_velocity(fields["time"], fields["apex_y"], sample_time)
    data["v_cl_short"] = local_velocity(fields["time"], fields["x_cl"], sample_time, .0075)
    data["v_apex_short"] = local_velocity(fields["time"], fields["apex_y"], sample_time, .0075)
    theta = np.radians(data["theta"])
    cos_y = np.cos(np.radians(run["theta_y"]))
    force = cos_y - np.cos(theta)
    drag = run["zeta"] * data["v_cl"]
    data["force"] = force
    data["residual_angle"] = np.sin(theta) * drag - force
    data["residual_constant_calibrated"] = np.sin(np.radians(run["theta_y"])) * drag - force
    data["residual_constant_nominal"] = drag - force
    data["residual_profile_adjusted"] = np.sin(theta) * drag - (
        cos_y / data["profile_scale"] - np.cos(theta)
    )
    return data


def read_saved_series(path):
    fields = load_csv(path)
    for name in np.unique(fields["name"]):
        rows = fields[fields["name"] == name]
        run = {key: rows[key][0] for key in RUN_FIELDS}
        run["final_time"] = rows["time"][-1]
        data = {key: rows[key] for key in rows.dtype.names if key not in RUN_FIELDS}
        yield run, data


def get_reference(reference, run, friction, times):
    result = reference[(reference["theta0_deg"] == run["theta0"]) &
                       (reference["zeta"] == run["zeta"]) &
                       (reference["friction"] == friction)]
    return {
        label: PchipInterpolator(result["time"], result[field])(times)
        for label, field in (("a", "a"), ("theta", "theta_deg"), ("v_apex", "v_apex"))
    }


def summarize(run, data, reference):
    results = []
    for start in (.02, .04):
        use = (data["time"] >= start) & (data["time"] <= run["final_time"] - .0125)
        row = {key: run[key] for key in RUN_FIELDS}
        row.update(t_start=start, t_end=float(data["time"][use][-1]))
        for key in ("force", "residual_angle", "residual_constant_calibrated",
                    "residual_constant_nominal", "residual_profile_adjusted"):
            row[f"rms_{key}"] = rms(data[key][use])
        row["rms_profile_scale_error"] = rms(data["profile_scale"][use] - 1)
        row["rms_v_cl_window_change"] = rms(data["v_cl"][use] - data["v_cl_short"][use])
        row["rms_v_apex_window_change"] = rms(data["v_apex"][use] - data["v_apex_short"][use])
        for law in ("angle", "constant_calibrated"):
            sharp = get_reference(reference, run, law, data["time"][use]) if run["model"] == "ac" else {}
            for field in ("a", "theta", "v_apex"):
                row[f"sharp_{law}_rms_{field}"] = rms(data[field][use] - sharp[field]) if sharp else ""
        results.append(row)
    return results


def style_axes(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(color="#d9dee3", linewidth=.45, alpha=.7)
    axis.tick_params(labelsize=8, length=2.5, color="#56616b")


def plot_trajectories(selected, reference, destination):
    colors = {"ac": "#08769a", "ch": "#b45328"}
    fields = [("a", r"Half-width $a$"), ("theta", r"Angle $\theta_{\rm CL}$ (°)"),
              ("v_apex", r"Apex speed $V$")]
    parameter_pairs = [(30, .5), (30, 2), (70, .5), (70, 2)]
    fig, axes = plt.subplots(3, 4, figsize=(10.4, 6.0), sharex=True, layout="constrained")
    for col, (angle, zeta) in enumerate(parameter_pairs):
        axes[0, col].set_title(fr"$\theta_0={angle}^\circ$, $\zeta={zeta:g}$", fontsize=10, pad=7)
        matching = [(run, data) for run, data in selected if run["theta0"] == angle and run["zeta"] == zeta]
        for row, (name, label) in enumerate(fields):
            axis = axes[row, col]
            style_axes(axis)
            if col == 0:
                axis.set_ylabel(label, fontsize=9)
            if row == 2:
                axis.set_xlabel("Time", fontsize=9)
            for law, style in (("angle", "-"), ("constant_calibrated", "--")):
                for_ref = {"theta0": angle, "zeta": zeta}
                time = np.linspace(.02, .2, 181)
                sharp = get_reference(reference, for_ref, law, time)
                axis.plot(time, sharp[name], color="#24292f", linestyle=style, linewidth=1.2)
            for run, data in sorted(matching, key=lambda p: (-p[0]["delta"], p[0]["model"])):
                use = (data["time"] >= .02) & (data["time"] <= (.188 if name == "v_apex" else .2))
                axis.plot(
                    data["time"][use], data[name][use], color=colors[run["model"]],
                    linestyle="-" if run["delta"] == .01 else ":",
                    linewidth=1.2, alpha=.96,
                )
            axis.set_xlim(.02, .2)
    labels = [
        Line2D([0], [0], color="#24292f", label="Sharp AC, angle-dependent"),
        Line2D([0], [0], color="#24292f", linestyle="--", label="Sharp AC, constant"),
        Line2D([0], [0], color=colors["ac"], label="Diffuse AC"),
        Line2D([0], [0], color=colors["ch"], label="Diffuse CH"),
        Line2D([0], [0], color="#64717c", label=r"Diffuse $\delta=0.01$"),
        Line2D([0], [0], color="#64717c", linestyle=":", label=r"Diffuse $\delta=0.02$"),
    ]
    fig.legend(handles=labels, loc="outside upper center", ncol=3, fontsize=8, frameon=False)
    fig.savefig(destination)
    plt.close(fig)


def plot_residuals(results, destination):
    colors = {"angle": "#08769a", "constant_calibrated": "#b45328"}
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3), sharey=True, layout="constrained")
    for axis, model in zip(axes, ("ac", "ch")):
        style_axes(axis)
        subset = [row for row in results if row["model"] == model and row["t_start"] == .02
                  and row["name"].startswith(f"dynamic_{model}_")]
        xs = [(30, .5), (30, 2), (70, .5), (70, 2)]
        for i, (theta, zeta) in enumerate(xs):
            for law, dx in (("angle", -.14), ("constant_calibrated", .14)):
                items = {row["delta"]: row[f"rms_residual_{law}"]
                         for row in subset if row["theta0"] == theta and row["zeta"] == zeta}
                axis.plot([i + dx] * len(items), list(items.values()), color=colors[law], linewidth=.6)
                for delta, value in items.items():
                    axis.scatter(i + dx, value, marker="o" if delta == .01 else "s",
                                 color=colors[law], s=23, zorder=3)
        axis.set_title({"ac": "Allen–Cahn", "ch": "Cahn–Hilliard"}[model], fontsize=10)
        axis.set_xticks(range(len(xs)), [fr"${a}^\circ$, {z:g}" for a, z in xs], fontsize=8)
        axis.set_xlabel(r"Initial angle $\theta_0$ and friction $\zeta$", fontsize=9)
        axis.set_yscale("log")
    axes[0].set_ylabel("RMS contact-line force residual", fontsize=9)
    fig.legend(handles=[
        Line2D([0], [0], color=colors["angle"], label="Angle-dependent law"),
        Line2D([0], [0], color=colors["constant_calibrated"], label="Calibrated constant law"),
        Line2D([0], [0], color="#64717c", marker="o", linestyle="", label=r"$\delta=0.01$"),
        Line2D([0], [0], color="#64717c", marker="s", linestyle="", label=r"$\delta=0.02$"),
    ], loc="outside upper center", ncol=2, fontsize=8, frameon=False)
    fig.savefig(destination)
    plt.close(fig)


def write_csv(path, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--runs", type=Path, help="Regenerated solver runs with geometry histories")
    inputs.add_argument("--series", type=Path, help="Published transient plot-input CSV")
    parser.add_argument("--sharp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true", help="Plot provisional results from completed cases")
    args = parser.parse_args()
    reference = load_csv(args.sharp)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.runs is not None:
        completed, missing = load_completed_cases(args.runs, ("dynamic_",), args.allow_incomplete)
        if missing:
            print("Omitting incomplete cases: " + ", ".join(missing), flush=True)
        datasets = ((run, read_diffuse(args.runs / name, run)) for name, run in completed.items())
    else:
        datasets = read_saved_series(args.series)
    metrics, cases, series = [], [], []
    for run, data in datasets:
        metrics.extend(summarize(run, data, reference))
        compact = {key: run[key] for key in RUN_FIELDS}
        series.extend(compact | dict(zip(data, row)) for row in zip(*data.values()))
        if run["name"].startswith(f"dynamic_{run['model']}_"):
            cases.append((run, data))
    if not metrics:
        raise RuntimeError("No complete transient runs found")
    write_csv(args.output / "transient_metrics.csv", metrics)
    write_csv(args.output / "transient_series.csv", series)
    plot_trajectories(cases, reference, args.output / "transient_trajectories.pdf")
    plot_residuals(metrics, args.output / "contact_line_residuals.pdf")


if __name__ == "__main__":
    main()
