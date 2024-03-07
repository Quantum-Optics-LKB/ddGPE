import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib
from ddGPE_fork_oscar.ggpe2d import ggpe
from ddGPE_fork_oscar.ggpe2d import tophat
from ddGPE_fork_oscar.ggpe2d import gaussian
from ddGPE_fork_oscar.ggpe2d import vortex_beam
from ddGPE_fork_oscar.ggpe2d import shear_layer
from ddGPE_fork_oscar.ggpe2d import plane_wave
from ddGPE_fork_oscar.ggpe2d import ring
from ddGPE_fork_oscar.ggpe2d import radial_expo
from ddGPE_fork_oscar.ggpe2d import to_turning_point
from ddGPE_fork_oscar.ggpe2d import bistab_cycle
from ddGPE_fork_oscar.ggpe2d import turn_on_pump
from ddGPE_fork_oscar.ggpe2d import tempo_probe
from observables import config_plots
from observables import polariton_fields
from observables import stationary_polariton_fields
from observables import movies
from observables import plot_gnLP_vs_I
import os
import cv2

def load_raw_data(folder):
    
    #Load raw data as numpy arrays
    mean_cav_t_x_y = np.load(folder+"/raw_arrays/mean_cav_t_x_y.npy")
    mean_exc_t_x_y = np.load(folder+"/raw_arrays/mean_exc_t_x_y.npy")
    stationary_cav_x_y = np.load(folder+"/raw_arrays/stationary_cav_x_y.npy")
    stationary_exc_x_y = np.load(folder+"/raw_arrays/stationary_exc_x_y.npy")
    hopfield_coefs = np.load(folder+"/raw_arrays/hopfield_coefs.npy")
    F_t = np.load(folder+"/raw_arrays/F_t.npy")
    
    return mean_cav_t_x_y, mean_exc_t_x_y, stationary_cav_x_y, stationary_exc_x_y, hopfield_coefs, F_t

# Load data and plotting parameters
directory = "/home/stagios/Oscar/LEON/DATA/Polaritons/2024_ManasOscar/bistability"
folder = directory + "/data_set_tophat"

cav_field_txy, exc_field_txy, cav_stationary_xy, exc_stationary_xy, hopfield_coefs, F_t = load_raw_data(folder)
cav_field_txy, exc_field_txy, cav_stationary_xy, exc_stationary_xy, hopfield_coefs, F_t = cp.asarray(cav_field_txy), cp.asarray(exc_field_txy), cp.asarray(cav_stationary_xy), cp.asarray(exc_stationary_xy), cp.asarray(hopfield_coefs), cp.asarray(F_t)

config_plots()

#--------------------------------------------------------------------------------

# Some parameers that you need for the observables, think how to make it so we don't need to define them here

long_1, long_2 = 256, 256
nmax = 256**1
nmax_1, nmax_2 = nmax, nmax

x_1 = cp.linspace(-long_1/2, +long_1/2, nmax_1, dtype = float)
x_2 = cp.linspace(-long_2/2, +long_2/2, nmax_2, dtype = float)
X, Y = cp.meshgrid(x_1, x_2)
R = cp.hypot(X,Y)


h_bar = 0.654 # (meV*ps)
rabi = 5.07/2/h_bar # (meV/h_bar) linear coupling (Rabi split)
g0 = (1e-2) /h_bar  # (frequency/density) (meV/hbar)/(1/um^2) nonlinear coupling constant 
#g0 = 0.003/h_bar
omega_exc = 1484.44 /h_bar # (meV/h_bar) exciton energy measured from the cavity energy #-0.5
omega_cav = 1482.76 /h_bar # (meV/h_bar) cavity energy at k=0 
delta = omega_cav - omega_exc # (meV/h_bar)
C02 = np.sqrt(delta**2 + 4*rabi**2) - delta
C02 /= 2*np.sqrt(delta**2 + 4*rabi**2)
X02 = 1 - C02
g_LP = g0*X02**2
#g_LP = 0.5*(X02**2*0.003+X02*np.sqrt(X02)*np.sqrt(C02)*0.0001)
#print(g_LP) #0.001796071745929218
detuning = 0.17/h_bar
gamma_exc, gamma_ph = 0.07 /h_bar, 0.07 /h_bar # (meV/h_bar) exc and ph linewidth 1microeV 69microeV  original value 0.07/h_bar

#---------------------------------------------------------------------------

# Now create functions analyzing data, plotting observables, etc... when it works put them in another file and import them from there (maybe even from class file??)


LP_t_x_y, UP_t_x_y, LP_w_kx_ky, UP_w_kx_ky = polariton_fields(cav_field_txy, exc_field_txy, hopfield_coefs)

LP_stat_x_y, UP_stat_x_y, LP_stat_kx_ky, UP_stat_kx_ky = stationary_polariton_fields(cav_stationary_xy, exc_stationary_xy, hopfield_coefs)

movies(folder, LP_t_x_y)

plot_gnLP_vs_I(folder, LP_t_x_y, F_t, R, g_LP, gamma_exc, gamma_ph, X02, C02, detuning, theoretical = False) #theoretical = False because it does not work aha