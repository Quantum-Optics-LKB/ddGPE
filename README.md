# ddGPE
Package to simulate the coupled driven-dissipative or generalized Gross-Pitaevskii equation in the presence of a dark excitonic reservoir. It uses slit-step spectral scheme. 

## Installation
clone the repository:

```bash
git clone https://github.com/Quantum-Optics-LKB/ddGPE.git
cd ddGPE
```

pip install the package:

```bash
pip install .
```
## Physical situation
The dynamics of coupled photon and exciton fields in a polariton microcavity is given by the following coupled equations:

$$
\begin{aligned} i\hbar\frac{d\psi_X}{dt}(\textbf{r}, t) &= \big(E_X(\textbf{k}) + V_X(\textbf{r}) + \hbar g_X (\psi_X|\psi_X(\textbf{r}, t)|^2 + n_r(\textbf{r}, t)) + \hbar\Omega_R\psi_C(\textbf{r}, t) - i\hbar\frac{\gamma_X+\gamma_{inc}}{2}\big)\psi_X(\textbf{r}, t) \\ i\hbar\frac{d\psi_C}{dt}(\textbf{r}, t) &= \big(E_C(\textbf{k}) + V_C(\textbf{r}) + \hbar\Omega_R\psi_X(\textbf{r}, t) - i\hbar\frac{\gamma_C+\gamma_{inc}}{2}\big)\psi_C(\textbf{r}, t) + \hbar F(\textbf{r},t) \\ \frac{dn_r}{dt}(\textbf{r}, t) &= -\gamma_{r}n_r+\gamma_{inc}(|\psi_X(\textbf{r}, t)|^2+|\psi_C(\textbf{r}, t)|^2) \end{aligned}
$$

where

- $\psi_X$ is the exciton field
- $\psi_C$ is the cavity field
- $V_X$ is the exciton potential
- $V_C$ is the cavity potential
- $g_X$ is the exciton interaction energy
- $\Gamma_X$ is the exciton losses coefficient
- $\Gamma_C$ is the cavity losses coefficient
- $\Omega_R$ is the Rabi coupling between excitons and photons
- $F$ are the optical fields sent on the cavity
