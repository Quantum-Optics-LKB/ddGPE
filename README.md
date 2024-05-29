Checkpoint for the class.

Contains all the functions that have been created and the most recent version of the class as of 16/04/2024. It now allows to run simulations in parallel. 
You may find examples to run in the file simu_examples.py and an example of analysis for the output of those runs in analysis_examples.py . You only need to uncomment the example you wish to run and change the directory path.

The class solves the coupled driven-dissipative or generalized Gross-Pitaevskii equation.

$$
i\hbar\frac{d\psi_X(\textbf{r}, t)}{dt}=(E_X(\textbf{k}) + V_X(\textbf{r}) + \hbar g_X \psi_X|\psi_X(\textbf{r}, t)|^2+ \hbar\Omega_R\psi_C(\textbf{r}, t) - i\hbar\frac{\Gamma_X}{2})\psi_X(\textbf{r}, t)
$$

$$
i\hbar\frac{d\psi_C(\textbf{r}, t)}{dt}=(E_C(\textbf{k}) + V_C(\textbf{r}) + \hbar\Omega_R\psi_X(\textbf{r}, t) - i\hbar\frac{\Gamma_C}{2})\psi_C(\textbf{r}, t) + \hbar F_p(\textbf{r},t)
$$
