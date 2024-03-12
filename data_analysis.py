import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib
from ggpe2d import ggpe
from ggpe2d import tophat
from ggpe2d import gaussian
from ggpe2d import vortex_beam
from ggpe2d import shear_layer
from ggpe2d import plane_wave
from ggpe2d import ring
from ggpe2d import radial_expo
from ggpe2d import to_turning_point
from ggpe2d import bistab_cycle
from ggpe2d import turn_on_pump
from ggpe2d import tempo_probe
from analysis_functions import load_raw_data
from analysis_functions import config_plots
from analysis_functions import polariton_fields
from analysis_functions import stationary_polariton_fields
from analysis_functions import movies
from analysis_functions import plot_gnLP_vs_I
from analysis_functions import plot_density
import os
import cv2
import configparser

# Load data and plotting parameters
directory = "/home/stagios/Oscar/LEON/DATA/Polaritons/2024_ManasOscar/dispersion"
folder = directory + "/data_set_tophat80_beg_noisy_Fpump2"

cav_field_txy, exc_field_txy, cav_stationary_xy, exc_stationary_xy, hopfield_coefs, F_t = load_raw_data(folder)
cav_field_txy, exc_field_txy, cav_stationary_xy, exc_stationary_xy, hopfield_coefs, F_t = cp.asarray(cav_field_txy), cp.asarray(exc_field_txy), cp.asarray(cav_stationary_xy), cp.asarray(exc_stationary_xy), cp.asarray(hopfield_coefs), cp.asarray(F_t)

config_plots()

#--------------------------------------------------------------------------------

# Import parameters values from parameters.txt file created when running the simulation
#%%
config = configparser.ConfigParser()
config.read(folder+"/parameters.txt")

h_bar = float(config.get("parameters", "h_bar"))
h_bar_SI = float(config.get("parameters", "h_bar_SI"))
c = float(config.get("parameters", "c"))
eV_to_J = float(config.get("parameters", "eV_to_J"))
rabi = float(config.get("parameters", "rabi (div by 2hbar)"))/2/h_bar
g0 = float(config.get("parameters", "g0 (div by hbar)"))/h_bar
gamma_exc = float(config.get("parameters", "gamma_exc (div by hbar)"))/h_bar
gamma_ph = float(config.get("parameters", "gamma_ph (div by hbar)"))/h_bar
gamma_LP = float(config.get("parameters", "gamma_LP (div by hbar)"))/h_bar
omega_exc = float(config.get("parameters", "omega_exc (div by hbar)"))/h_bar
omega_cav = float(config.get("parameters", "omega_cav (div by hbar)"))/h_bar
n_cav = float(config.get("parameters", "n_cav"))
k_z = float(config.get("parameters", "k_z"))
t_min = float(config.get("parameters", "t_min"))
t_obs = float(config.get("parameters", "t_obs"))
t_noise = float(config.get("parameters", "t_noise"))
t_probe = float(config.get("parameters", "t_probe"))
t_stationary = float(config.get("parameters", "t_stationary"))
t_max = float(config.get("parameters", "t_max"))
dt_frame = float(config.get("parameters", "dt_frame"))
nmax_1 = int(config.get("parameters", "nmax_1"))
nmax_2 = int(config.get("parameters", "nmax_2"))
long_1 = float(config.get("parameters", "long_1"))
long_2 = float(config.get("parameters", "long_2"))
F_pump = float(config.get("parameters", "F_pump"))
F_probe = float(config.get("parameters", "F_probe"))
corr = float(config.get("parameters", "corr"))
detuning = float(config.get("parameters", "detuning (div by hbar)"))/h_bar
omega_probe = float(config.get("parameters", "omega_probe"))
Pump_profile = config.get("parameters", "Pump_profile")
Probe_profile = config.get("parameters", "Probe_profile")
#%%

k_1 = cp.linspace(-2*cp.pi/long_1*nmax_1/2, 2*cp.pi/long_1*(nmax_1/2-1), nmax_1)
k_2 = cp.linspace(-2*cp.pi/long_2*nmax_2/2, 2*cp.pi/long_2*(nmax_2/2-1), nmax_2)
K_1, K_2 = cp.meshgrid(k_1, k_2)
x_1 = cp.linspace(-long_1/2, +long_1/2, nmax_1, dtype = float)
x_2 = cp.linspace(-long_2/2, +long_2/2, nmax_2, dtype = float)
X, Y = cp.meshgrid(x_1, x_2)
R = cp.hypot(X,Y)

delta = omega_cav - omega_exc # (meV/h_bar)
C02 = np.sqrt(delta**2 + 4*rabi**2) - delta
C02 /= 2*np.sqrt(delta**2 + 4*rabi**2)
X02 = 1 - C02
g_LP = g0*X02**2
#g_LP = 0.5*(X02**2*0.003+X02*np.sqrt(X02)*np.sqrt(C02)*0.0001)
#print(g_LP) #0.001796071745929218


#---------------------------------------------------------------------------

# Now create functions analyzing data, plotting observables, etc... when it works put them in another file and import them from there (maybe even from class file??)

side=100 #even number
window = (nmax_1//2-side/2, nmax_1//2+side/2, nmax_2//2-side/2, nmax_2//2+side/2)
# window = (0, nmax_1, 0, nmax_2)

LP_t_x_y, UP_t_x_y, LP_w_kx_ky, UP_w_kx_ky = polariton_fields(cav_field_txy, exc_field_txy, hopfield_coefs, window = window)

LP_stat_x_y, UP_stat_x_y, LP_stat_kx_ky, UP_stat_kx_ky = stationary_polariton_fields(cav_stationary_xy, exc_stationary_xy, hopfield_coefs, window = window)

fluctuations_LP = cp.zeros(LP_w_kx_ky.shape, dtype = cp.complex64)
fluctuations_LP[:] = LP_w_kx_ky[:] - LP_stat_kx_ky

# movies(folder, LP_t_x_y)
# movies(folder, fluctuations_LP)

# plot_gnLP_vs_I(folder, LP_t_x_y, F_t, R, g_LP, gamma_exc, gamma_ph, X02, C02, detuning = None, theoretical = False) #theoretical = False because it does not work aha

#Plotting stationary densities of LP

plot_density(folder, ("$k_x$", k_1[window[0]:window[1]+1]), ("$k_y$", k_2[window[2]:window[3]+1]), ("stationary_LP_kspace_log", LP_stat_kx_ky))
plot_density(folder, ("$x$", x_1[window[0]:window[1]+1]), ("$y$", x_2[window[2]:window[3]+1]), ("stationary_LP_rspace", LP_stat_x_y))

omega_list = 2*cp.pi*cp.fft.fftshift(cp.fft.fftfreq(LP_w_kx_ky.shape[0], dt_frame))

plot_density(folder, ("$k_x$", k_1[window[0]:window[1]+1]), ("$\omega$", omega_list), ("w_kx_ky=0", fluctuations_LP[::-1,:,fluctuations_LP.shape[1]//2]), norm="log")
plot_density(folder, ("$k_y$", k_1[window[0]:window[1]+1]), ("$\omega$", omega_list), ("w_kx=0_ky", LP_w_kx_ky[::-1,LP_w_kx_ky.shape[1]//2,:]), norm="log")
plot_density(folder, ("$k_x$", k_1[window[0]:window[1]+1]), ("$k_y$", k_2[window[2]:window[3]+1]), ("w="+str(omega_list[omega_list.shape[0]//3*2])+"_kx_ky", LP_w_kx_ky[omega_list.shape[0]//3*2,:,:]), norm="log")

time_plot = np.array([i*dt_frame for i in range(len(cav_field_txy))])
F_intensity = cp.abs(F_t)**2
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(time_plot, LP_t_x_y[:,LP_t_x_y.shape[1]//2, LP_t_x_y.shape[2]//2].get(), label = "$n_{LP}(0,0)$", color = "r")
ax2.plot(time_plot, F_intensity.get(), label = "$I(0,0)$", color = "b", linestyle = "--")
ax1.set_xlabel("Time (ps)")
ax1.set_ylabel("$n_{LP}$")
ax2.set_ylabel("$I$")
fig.legend()
fig.savefig(folder+"/laser_LP_densities.png")
plt.close("all")

