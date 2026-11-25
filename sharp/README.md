# Sharp-interface Allen–Cahn reference

This standalone Python solver generates the sharp trajectories used in the two-dimensional transient comparison. It evolves a nonoverhanging, symmetric droplet by area-preserving curvature flow with a moving contact line.

## Model and initial conditions

The right half of the droplet has area $A=1/8$. Its footprint is $0\le x\le a(t)$ on the flat wall $y=0$. The Young angle is $\theta_Y=45^\circ$, and the surface tension $\sigma$ and bulk dissipation coefficient $\xi$ are both one. For each initial angle $\theta_0=30^\circ,70^\circ$, the initial graph is

$$
h(x,0)=\frac{a_0\tan\theta_0}{2}
\left(1-\frac{x^2}{a_0^2}\right), \qquad
a_0=\sqrt{\frac{3A}{\tan\theta_0}}.
$$

The interior evolves by $\xi V=-\sigma H+\Lambda$, with $\Lambda$ chosen to preserve $A$. The wall speed is determined by

$$
\zeta b(\theta_{\mathrm{CL}})\dot a
=\sigma(\cos\theta_Y-\cos\theta_{\mathrm{CL}}).
$$

We use $\zeta=0.5,2$, with $b(\theta_{\mathrm{CL}})=\sin\theta_{\mathrm{CL}}$ for the angle-dependent law and $b=\sin\theta_Y$ for the calibrated constant-friction law.

## Discretization

The footprint map $x=a(t)s$ fixes the computational interval $0\le s\le1$. The height $u(s,t)=h(a(t)s,t)$ satisfies $u_s(0,t)=0$ and $u(1,t)=0$. The solver uses centered three-point differences for the interior graph derivatives and a one-sided quadratic wall slope for $\theta_{\mathrm{CL}}$. The spacing is clustered near the wall by $s=q+0.96\sin(\pi q)/\pi$ on a uniform $q$-grid. The multiplier enforces a zero time derivative of $a\int_0^1 u\,ds$, evaluated by Simpson quadrature with the map Jacobian. SciPy's BDF method uses relative tolerance $10^{-9}$, absolute tolerance $10^{-11}$, and maximum step $0.005$.

## Reproduce the reference

From this directory, run the following command with NumPy and SciPy installed. The tested versions were NumPy 2.5.3 and SciPy 1.18.1.

    python sharp_ac.py --n 1024 --stretch 0.96 --output-dir sharp_ac_1024

The command produces all eight sharp trajectories through $t=0.5$, with output spacing $0.002$. The paper uses the time window $0.02\le t\le0.186$ for the root-mean-square comparisons. The `trajectories.csv` file contains footprint, contact angle, contact-line speed, apex normal velocity, area, height, and energy. The `profiles.csv` file contains sampled interfaces and normal velocities.

To repeat the sharp-reference refinement checks, generate three coarser spatial runs and a tighter time integration.

    python sharp_ac.py --n 128 --stretch 0.96 --output-dir sharp_ac_128
    python sharp_ac.py --n 256 --stretch 0.96 --output-dir sharp_ac_256
    python sharp_ac.py --n 512 --stretch 0.96 --output-dir sharp_ac_512
    python sharp_ac.py --n 512 --stretch 0.96 --rtol 1e-11 --atol 1e-13 \
        --max-step 0.0025 --output-dir sharp_ac_512_tight
    python sharp_ac_verify.py --space sharp_ac_128 sharp_ac_256 sharp_ac_512 sharp_ac_1024 \
        --tight sharp_ac_512_tight --output verification.csv
