# ddGPE

Package to simulate coupled exciton and photon driven-dissipative Gross–Pitaevskii equations with a dark excitonic reservoir, using a first-order split-step spectral scheme and CuPy.

## Installation

Clone the repository and install the package:

```bash
git clone https://github.com/Quantum-Optics-LKB/ddGPE.git
cd ddGPE
pip install .
```

Running simulations requires a working CuPy installation and a compatible CUDA GPU.

## Physical model

The fields are evolved in the frame rotating at the pump angular frequency, with energies divided by hbar. With `apply_reservoir=True`, the deterministic model is

$$
\begin{aligned}
i\partial_t\psi_X &=
\left[\omega_X-\omega_p+g_0\left(|\psi_X|^2+n_r\right)
-i\frac{\gamma_X+\gamma_{\mathrm{in}}}{2}\right]\psi_X
+\Omega\psi_C,\\
i\partial_t\psi_C &=
\left[\omega_C(-i\nabla)-\omega_p+U_C(\mathbf r)
-i\frac{\gamma_C+\gamma_{\mathrm{in}}+v_\gamma(\mathbf r)}{2}\right]\psi_C
+\Omega\psi_X+F(\mathbf r,t),\\
\partial_t n_r &=
\gamma_{\mathrm{in}}\left(|\psi_X|^2+|\psi_C|^2\right)-\gamma_r n_r.
\end{aligned}
$$

Here the bare decay rates `gamma_X` and `gamma_C` refer to constructor inputs, before conversion losses are added. The linear propagation uses the Hopfield approximation described below.

| Symbol | Code parameter or array | Meaning |
|---|---|---|
| $\psi_X$, $\psi_C$ | `phi[0]`, `phi[1]` | Exciton and photon fields |
| $n_r$ | `den_reservoir` | Reservoir density, not a field amplitude |
| $g_0$ | `g0` | Exciton interaction coefficient; also used for the reservoir-induced exciton shift |
| $\Omega$ | `rabi` | Off-diagonal exciton–photon coupling |
| $\gamma_X$, $\gamma_C$ | Input `gamma_exc`, `gamma_cav` | Bare coherent density decay rates |
| $\gamma_{\mathrm{in}}$ | `gamma_res` | Coherent-to-reservoir conversion rate |
| $\gamma_r$ | `gamma_res_decay` | Reservoir decay rate, set to the original input `gamma_exc` |
| $U_C$ | `potential` | Cavity potential energy divided by hbar |
| $v_\gamma$ | `v_gamma` | Additional photon density loss at the boundaries |

The reservoir shifts only the exciton energy directly. Its effect on photons follows through exciton–photon coupling. There is no reservoir-dependent coherent gain, reservoir diffusion, or exciton potential in this implementation.

Conversion adds `gamma_res` to both coherent decay rates. The reservoir decay remains equal to the original `gamma_exc` input:

```text
self.gamma_exc = input gamma_exc + gamma_res
self.gamma_cav = input gamma_cav + gamma_res
reservoir decay = input gamma_exc
```

Thus conversion removes coherent density and supplies reservoir density at matching rates in the differential model.

With `apply_reservoir=False`, the extra coherent losses and reservoir update are disabled. The nonlinear kernel still uses the supplied reservoir density: a nonzero `initial_state[2]` acts as a frozen exciton shift. The default initial reservoir is zero.

## Dispersion, drive, and rotating frame

The exciton dispersion is flat, while the photon dispersion is

$$
\omega_C(\mathbf k)=\omega_C(0)\sqrt{1+|\mathbf k|^2/k_z^2}.
$$

The pump frequency is defined using the bare, lossless lower-polariton energy at zero momentum:

$$
\begin{aligned}
\omega_{\mathrm{LP}}(0)&=
\frac{\omega_X+\omega_C(0)}{2}
-\frac12\sqrt{[\omega_C(0)-\omega_X]^2+4\Omega^2},\\
\omega_p&=\omega_{\mathrm{LP}}(0)+\mathrm{detuning}.
\end{aligned}
$$

Positive `detuning` places the pump above this reference. For a finite pump wavevector, the effective detuning from the LP resonance also includes the dispersion shift at that wavevector.

The cavity drive is the sum of pump and probe amplitudes:

```python
F = F_pump * F_pump_r * F_pump_t + F_probe * F_probe_r * F_probe_t
```

These profiles multiply field amplitudes; intensity is proportional to the squared modulus. A probe factor `exp(-1j * omega_probe * t)` denotes a frequency offset above the pump for positive `omega_probe`.

## Units and conventions

| Quantity | Units |
|---|---|
| Time | ps |
| Position | micrometres |
| Fields | inverse micrometres |
| Field and reservoir densities | inverse square micrometres |
| `omega_exc`, `omega_cav`, `rabi`, `detuning`, decay rates | inverse ps (angular-frequency convention) |
| `g0` | square micrometres per ps |
| `potential` | inverse ps |
| Drive amplitude, for dimensionless profiles | inverse micrometres per ps |
| Wavevector | inverse micrometres (angular wavevector) |

Convert energies in meV to evolution coefficients by dividing by hbar in meV ps. A physical potential `V` must be supplied as `V / hbar`; an interaction energy shift is `hbar * g0 * density`.

The term `-1j * gamma / 2` gives amplitude decay `exp(-gamma*t/2)` and density decay `exp(-gamma*t)`. The corresponding isolated-mode energy linewidth (FWHM) is `hbar * gamma`.

The lossless resonant Rabi splitting is `2 * hbar * rabi`. In the thesis convention with off-diagonal coupling `hbar * Omega_R / 2`, use `rabi = Omega_R / 2`.

## Lower-polariton mapping and reservoir fraction

For a predominantly LP field at low momentum, let `X02 = |X_0|^2` be its exciton fraction and `C02 = 1 - X02` its photon fraction. The code uses the phase convention

```text
psi_LP = -X * psi_X + C * psi_C
psi_UP =  C * psi_X + X * psi_C
```

Projection with approximately constant Hopfield coefficients gives

```text
g_LP = g0 * X02**2
g_res_LP = g0 * X02
F_LP = sqrt(C02) * F
bare gamma_LP ≈ X02 * input gamma_exc + C02 * input gamma_cav
additional gamma_LP = gamma_res
```

The model fixes the reservoir coupling and decay through these relations; they are not independent constructor inputs.

At steady state in this approximation:

```text
n_res / n_LP = gamma_res / input gamma_exc
beta = (g_res_LP * n_res) / (g_LP * n_LP)
     = gamma_res / (X02 * input gamma_exc)
```

`examples/physical_constants.py` targets `beta = 0.49` by calculating the zero-momentum exciton fraction and setting

```python
delta_cx = omega_cav - omega_exc
X02 = 0.5 * (1 + delta_cx / (delta_cx**2 + 4 * rabi**2)**0.5)
gamma_res = 0.49 * X02 * gamma_exc
```

This is a steady-state, low-momentum calibration, not a constraint on transient or strongly mixed LP/UP states.

These conventions correspond to the LP reservoir model in Eq. (1.91) of [K. Guerrero-Feuillet, *Rotational superradiance in a polariton quantum fluid* (2026)](https://theses.hal.science/tel-05677012v1), with thesis `gamma_in` mapped to code `gamma_res`, and thesis reservoir decay mapped to the input `gamma_exc`. The measured reservoir blueshift ratio is discussed in thesis section 3.1.7.

## Model approximations

- The linear step uses complex polariton eigenfrequencies with lossless Hopfield coefficients. This neglects the small change in composition due to unequal component losses; it becomes exact for equal losses.
- The solver retains both branches and the full photon dispersion. The scalar LP model follows in the low-momentum, predominantly LP regime.
- The reservoir evolves dynamically. Comparing with the thesis Bogoliubov treatment, which neglects reservoir-density perturbations, requires that additional approximation.
- The equations above describe deterministic evolution. Optional noise is added after `t_noise`; its current increments scale with `dt` rather than `sqrt(dt)`, so it is not a correctly time-step-scaled continuous white-noise discretization.
