#!/usr/bin/env python3
"""Independent sharp Allen--Cahn reference for a symmetric 2D droplet.

The map x = a(t)*s puts the evolving graph y = u(s,t) on 0 <= s <= 1.
Here a is the contact coordinate. Let g = sqrt(1 + (u_s/a)^2). In the
interior, V = (u_t - s*a'*u_s/a)/g and curvature is -u_ss/(a^2*g^3).
The bulk law is xi*V = -sigma*curvature + Lambda. Choose Lambda so the
Simpson approximation of a*integral(u ds) has zero time derivative.
Centered differences use reflection at the symmetry axis. A second-order
one-sided slope at the right wall supplies the dynamic contact angle and
the prescribed kinetic law determines a'. The moving endpoint has u = 0.
SciPy BDF integrates the resulting stiff method-of-lines system. Optional
stretching with s = q + stretch*sin(pi*q)/pi concentrates nodes at the wall
for steep but non-overhanging profiles. Differences use adjacent physical
node spacings, and Simpson quadrature uses the mapping Jacobian in q.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


KINETIC_LAWS = ("angle", "constant_calibrated")
TRAJECTORY_FIELDS = (
    "time", "theta0_deg", "zeta", "friction", "a", "theta_deg", "v_cl",
    "v_apex", "area", "height", "energy",
)
PROFILE_FIELDS = (
    "time", "theta0_deg", "zeta", "friction", "s", "x", "y", "v_normal",
)


class SharpAllenCahn:
    def __init__(self, n, stretch=0.0, area=0.125, theta_y_deg=45.0, sigma=1.0, xi=1.0):
        if n < 4 or n % 2:
            raise ValueError("The number of graph intervals must be even and at least four.")
        if not 0.0 <= stretch < 1.0:
            raise ValueError("The stretching factor must be in [0, 1).")
        q = np.linspace(0.0, 1.0, n + 1)
        self.s = q + stretch * np.sin(np.pi * q) / np.pi
        jacobian = 1.0 + stretch * np.cos(np.pi * q)
        self.ds = np.diff(self.s)
        self.left_weight = self.ds[1:] / (self.ds[:-1] + self.ds[1:])
        self.curvature_weight = 2.0 / (self.ds[:-1] + self.ds[1:])
        d1 = 1.0 - self.s[-2]
        d2 = 1.0 - self.s[-3]
        self.wall_slope_weights = np.array([d1 / d2, -d2 / d1]) / (d2 - d1)
        self.weights = np.ones(n + 1)
        self.weights[1:n:2] = 4.0
        self.weights[2:n:2] = 2.0
        self.weights *= q[1] / 3.0 * jacobian
        self.area = area
        self.sigma = sigma
        self.xi = xi
        self.cos_y = np.cos(np.deg2rad(theta_y_deg))
        self.sin_y = np.sin(np.deg2rad(theta_y_deg))

    def initial_parabola(self, theta0_deg):
        slope = np.tan(np.deg2rad(theta0_deg))
        profile = 1.0 - self.s**2
        a = np.sqrt(2.0 * self.area / (slope * (self.weights @ profile)))
        height = a * slope / 2.0
        return np.r_[height * profile[:-1], a]

    def geometry(self, state):
        a = state[-1]
        u = np.zeros_like(state)
        u[:-1] = state[:-1]
        secants = np.diff(u) / self.ds
        u_s = self.left_weight * secants[:-1] + (1.0 - self.left_weight) * secants[1:]
        wall_slope = self.wall_slope_weights @ u[-3:-1]
        slope = np.r_[0.0, u_s, wall_slope] / a
        metric = np.sqrt(1.0 + slope**2)
        u_xx = np.r_[2.0 * secants[0] / self.ds[0], self.curvature_weight * np.diff(secants)] / a**2
        return a, u, slope, metric, u_xx

    def rates(self, state, zeta, friction):
        a, u, slope, metric, u_xx = self.geometry(state)
        cos_theta = 1.0 / metric[-1]
        sin_theta = -slope[-1] / metric[-1]
        sin_kinetic = sin_theta if friction == "angle" else self.sin_y
        a_dot = self.sigma * (self.cos_y - cos_theta) / (zeta * sin_kinetic)
        curvature_velocity = self.sigma / self.xi * u_xx / metric[:-1] ** 3
        transport = self.s[:-1] * a_dot * slope[:-1]
        base_rate = transport + metric[:-1] * curvature_velocity
        weights = self.weights[:-1]
        pressure_velocity = -(a_dot / a * (weights @ u[:-1]) + weights @ base_rate) / (
            weights @ metric[:-1]
        )
        u_dot = base_rate + pressure_velocity * metric[:-1]
        normal_velocity = np.r_[curvature_velocity + pressure_velocity, a_dot * sin_theta]
        return np.r_[u_dot, a_dot], normal_velocity, np.rad2deg(np.arctan(-slope[-1]))

    def integrate(self, theta0_deg, zeta, friction, times, rtol, atol, max_step):
        if friction not in KINETIC_LAWS:
            raise ValueError(f"Unknown friction law: {friction}")
        result = solve_ivp(
            lambda _, state: self.rates(state, zeta, friction)[0],
            (times[0], times[-1]),
            self.initial_parabola(theta0_deg),
            method="BDF",
            t_eval=times,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        if not result.success:
            raise RuntimeError(result.message)
        return result.y.T

    def observables(self, state, zeta, friction):
        a, u, _, metric, _ = self.geometry(state)
        rate, normal_velocity, theta_deg = self.rates(state, zeta, friction)
        area = a * (self.weights @ u)
        energy = self.sigma * a * ((self.weights @ metric) - self.cos_y)
        return a, theta_deg, rate[-1], normal_velocity[0], area, u[0], energy


def run(args):
    if args.end <= 0 or args.output_dt <= 0:
        raise ValueError("End time and output interval must be positive.")
    if args.profile_points < 2:
        raise ValueError("At least two points are needed for a saved profile.")
    profile_times = args.profile_times
    if profile_times is None:
        profile_times = sorted({t for t in (0.02, 0.1, 0.2, args.end) if t <= args.end})
    if any(t < 0 or t > args.end for t in profile_times):
        raise ValueError("Profile times must lie in the integration interval.")
    if any(theta <= 0 or theta >= 90 for theta in args.theta0):
        raise ValueError("The graph reference uses acute initial contact angles.")
    if any(zeta <= 0 for zeta in args.zeta):
        raise ValueError("Contact-line friction must be positive.")
    regular_times = np.unique(np.round(np.r_[np.arange(0.0, args.end, args.output_dt), args.end], 12))
    profile_times = np.asarray(profile_times)
    times = np.unique(np.round(np.r_[regular_times, profile_times], decimals=12))
    regular_indices = np.searchsorted(times, np.round(regular_times, decimals=12))
    profile_indices = np.searchsorted(times, np.round(profile_times, decimals=12))
    profile_s = np.linspace(0.0, 1.0, args.profile_points)

    model = SharpAllenCahn(args.n, stretch=args.stretch)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "trajectories.csv").open("w", newline="") as trajectory_file, (
        args.output_dir / "profiles.csv"
    ).open("w", newline="") as profile_file:
        history = csv.writer(trajectory_file)
        profiles = csv.writer(profile_file)
        history.writerow(TRAJECTORY_FIELDS)
        profiles.writerow(PROFILE_FIELDS)
        for theta0 in args.theta0:
            for zeta in args.zeta:
                for friction in args.friction:
                    states = model.integrate(
                        theta0, zeta, friction, times, args.rtol, args.atol, args.max_step
                    )
                    for index in regular_indices:
                        row = model.observables(states[index], zeta, friction)
                        history.writerow((times[index], theta0, zeta, friction, *row))
                    for index in profile_indices:
                        state = states[index]
                        a, u, _, _, _ = model.geometry(state)
                        _, velocity, _ = model.rates(state, zeta, friction)
                        profile_y = np.interp(profile_s, model.s, u)
                        profile_v = np.interp(profile_s, model.s, velocity)
                        for s, y, v in zip(profile_s, profile_y, profile_v):
                            profiles.writerow((times[index], theta0, zeta, friction, s, a * s, y, v))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--stretch", type=float, default=0.0)
    parser.add_argument("--theta0", type=float, nargs="+", default=[30, 70])
    parser.add_argument("--zeta", type=float, nargs="+", default=[0.5, 2])
    parser.add_argument("--friction", choices=KINETIC_LAWS, nargs="+", default=list(KINETIC_LAWS))
    parser.add_argument("--end", type=float, default=0.5)
    parser.add_argument("--output-dt", type=float, default=0.002)
    parser.add_argument("--profile-times", type=float, nargs="*")
    parser.add_argument("--profile-points", type=int, default=101)
    parser.add_argument("--rtol", type=float, default=1.0e-9)
    parser.add_argument("--atol", type=float, default=1.0e-11)
    parser.add_argument("--max-step", type=float, default=0.005)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
