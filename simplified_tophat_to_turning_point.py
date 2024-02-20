import datetime
import json
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib
from ggpe2d import ggpe
from skimage.restoration import unwrap_phase
import os
import scipy
#from azim_avg import mean_azim
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cupyx.scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
import scipy.constants as const
#from utilitiesf import create_gif
#from Velocity import velocity
import cProfile
import pstats
import cv2

matplotlib.rcParams['figure.figsize'] = [10, 10]
matplotlib.rcParams["legend.loc"] = 'upper right' 
matplotlib.rcParams['axes.labelsize'] = 18
matplotlib.rcParams['axes.titlesize'] = 20
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)
matplotlib.rc('legend', fontsize = 16) 

def save_data_giant_vortex(folder):

    mean_cav_x_y_t = cp.asnumpy(simu.mean_cav_x_y_t)
    mean_exc_x_y_t = cp.asnumpy(simu.mean_exc_x_y_t)
    F_t = cp.asnumpy(simu.F_t)
    
    mean_cav_t_x_y = np.einsum('xyt->txy', mean_cav_x_y_t)
    mean_exc_t_x_y = np.einsum('xyt->txy', mean_exc_x_y_t)

    mean_exc = mean_exc_t_x_y[:, :, :]
    mean_cav = mean_cav_t_x_y[:, :, :]
    
    pol = mean_cav_t_x_y * np.sqrt(C02) - mean_exc_t_x_y * np.sqrt(X02)
    
    plt.savefig(folder + "/field_tot")
    
    size=(nmax_1, nmax_2)
    fps=15
    
    out_dens = cv2.VideoWriter(folder +"/dens_evolution.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
    out_phase = cv2.VideoWriter(folder +"/phase_evolution.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
    max_dens = np.amax(np.abs(pol)**2)
    for i in range(len(pol)):
        dens = np.array(np.abs(pol[i])**2*255/max_dens, dtype = np.uint8)
        phase = np.array(np.angle(pol[i]), dtype = np.uint8)
        out_dens.write(dens)
        out_phase.write(phase)
    out_dens.release()
    out_phase.release()
    
    string_name = ""
    np.save(folder + "/pol.npy", pol)
    np.save(folder + "/F_t.npy", F_t)



h_bar = 0.654 # (meV*ps)
c = 2.9979*1e2 # (um/ps)
eV_to_J = 1.60218*1e-19
h_bar_SI = 1.05457182*1e-34

#Microcavity parameters
rabi = 5.07/2/h_bar # (meV/h_bar) linear coupling (Rabi split)
g0 = (1e-2) /h_bar  # (frequency/density) (meV/hbar)/(1/um^2) nonlinear coupling constant 
gamma_exc, gamma_ph = 0.07 /h_bar, 0.07 /h_bar # (meV/h_bar) exc and ph linewidth 1microeV 69microeV
omega_exc = 1484.44 /h_bar # (meV/h_bar) exciton energy measured from the cavity energy #-0.5
omega_cav = 1482.76 /h_bar # (meV/h_bar) cavity energy at k=0 
delta = omega_cav - omega_exc # (meV/h_bar)
C02 = np.sqrt(delta**2 + 4*rabi**2) - delta
C02 /= 2*np.sqrt(delta**2 + 4*rabi**2)
X02 = 1 - C02
g_LP = g0*X02**2
print("g_LP="+str(g_LP))
n_cav = 3.54
k_z = 27 # (1/µm) n_cav*omega_cav/c

gamma_LP = X02 * gamma_exc + C02 * gamma_ph

# Time parameters
t_min = 0 # (ps) initial time of evolution
t_obs = 0 # (ps) time from which the observation starts
t_noise = 1e9
t_probe = 1e9
t_stationary = 1750
t_max = 100 # (ps) final time of evolution
dt_frame = 1/(1) #cst/delta_E avec delta_E la résolution en énergie en meV
n_frame = int((t_max-t_obs)/dt_frame)+1
print("dt_frame is %s"%(dt_frame))
print("n_frame is %s"%(n_frame))

nmax = 256**1

nmax_1, nmax_2 = nmax, nmax
long_1, long_2 = 256, 256

if (long_1/nmax)**2<g0/gamma_ph:
    print("WARNING: TWA NOT VALID")

F = 1
corr = 0.0 #0.35
detuning = 0.17/h_bar
noise = 0
omega_probe=0

tempo_type = "to_turning_pt"



simu = ggpe(nmax_1, nmax_2, long_1, long_2, tempo_type, t_max, t_stationary, t_obs, t_probe, t_noise, dt_frame, gamma_exc, 
        gamma_ph, noise, g0, detuning, omega_probe, omega_exc, omega_cav, rabi, k_z)


m_LP = simu.m_LP
R = simu.R
THETA = simu.THETA
X = simu.X
Y = simu.Y
pump_radius = 60



simu.tophat(F, pump_radius)




folder_DATA =  "/home/stagios/Oscar/LEON/DATA/Polaritons/2024_ManasOscar/first_runs"
#string_name = "_noise=%s_dx%s_dt=%s"%(str(round(noise,5)),str(round(long_1/nmax_1,5)),str(round(dt_frame,5)))
string_name="_tophat_turning_point_basischange"
#string_name = "_k=%s_detuning=%s_F=%s"%(str(round(kx,3)),str(round(detuning,3)),str(round(F,3)))

try:
    os.mkdir(folder_DATA)
except:
    print("folder already created")

folder_DATA += "/data_set" + string_name

print("/data_set" + string_name)

try:
    os.mkdir(folder_DATA)
except:
    print("folder already created")
    
simu.evolution()
save_data_giant_vortex(folder_DATA)