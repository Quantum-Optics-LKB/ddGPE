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
\begin{aligned}
i\hbar \frac{\partial \psi_X(\mathbf{r}, t)}{\partial t} &= 
\Big[ E_X(\mathbf{k}) + V_X(\mathbf{r}) 
+ \hbar g_X \big( |\psi_X(\mathbf{r}, t)|^2 + n_r(\mathbf{r}, t) \big) 
- i\hbar \frac{\gamma_X + \gamma_{\mathrm{inc}}}{2} \Big]\psi_X(\mathbf{r}, t)
+ \hbar \Omega_R\, \psi_C(\mathbf{r}, t), \\[6pt]
i\hbar \frac{\partial \psi_C(\mathbf{r}, t)}{\partial t} &= 
\Big[ E_C(\mathbf{k}) + V_C(\mathbf{r}) 
- i\hbar \frac{\gamma_C + \gamma_{\mathrm{inc}}}{2} \Big]\psi_C(\mathbf{r}, t)
+ \hbar \Omega_R\, \psi_X(\mathbf{r}, t)
+ \hbar F(\mathbf{r}, t), \\[6pt]
\frac{\partial n_r(\mathbf{r}, t)}{\partial t} &= 
- \gamma_r\, n_r(\mathbf{r}, t)
+ \gamma_{\mathrm{inc}}\big( |\psi_X(\mathbf{r}, t)|^2 + |\psi_C(\mathbf{r}, t)|^2 \big).
\end{aligned}
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
