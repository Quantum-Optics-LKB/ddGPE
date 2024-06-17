The class solves the coupled driven-dissipative or generalized Gross-Pitaevskii equation in the presence of a dark excitonic reservoir.

$$
i\hbar\frac{d\psi_X}{dt}(\textbf{r}, t)=\big(E_X(\textbf{k}) + V_X(\textbf{r}) + \hbar g_X (\psi_X|\psi_X(\textbf{r}, t)|^2 + n_r(\textbf{r}, t))+ \hbar\Omega_R\psi_C(\textbf{r}, t) - i\hbar\frac{\gamma_X+\gamma_{inc}}{2}\big)\psi_X(\textbf{r}, t)
$$

$$
i\hbar\frac{d\psi_C}{dt}(\textbf{r}, t)=\big(E_C(\textbf{k}) + V_C(\textbf{r}) + \hbar\Omega_R\psi_X(\textbf{r}, t) - i\hbar\frac{\gamma_C+\gamma_{inc}}{2}\big)\psi_C(\textbf{r}, t) + \hbar F(\textbf{r},t)
$$

$$
\frac{dn_r}{dt}(\textbf{r}, t)=-\gamma_{r}n_r+\gamma_{inc}(|\psi_X(\textbf{r}, t)|^2+|\psi_C(\textbf{r}, t)|^2)
$$
