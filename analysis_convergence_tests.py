import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from analysis_functions import load_raw_data
from analysis_functions import config_plots
from analysis_functions import polariton_fields
from analysis_functions import stationary_polariton_fields
from analysis_functions import movies
from analysis_functions import plot_gnLP_vs_I
from analysis_functions import plot_density
import configparser
from tqdm import tqdm
import cv2

    
cp.cuda.Device(0).use()

# Load data and plotting parameters
directory = "/home/stagios/Oscar/LEON/DATA/Polaritons/2024_ManasOscar/convergence_tests/aliasing_tests"
folder = directory + "/data_set_noisy_tophat80_256x256_dtframe=dt"

initial_state = False
saved_frame = True


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
gamma_cav = float(config.get("parameters", "gamma_cav (div by hbar)"))/h_bar
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
Nx = int(config.get("parameters", "Nx"))
Ny = int(config.get("parameters", "Ny"))
Lx = float(config.get("parameters", "Lx"))
Ly = float(config.get("parameters", "Ly"))
F_pump = float(config.get("parameters", "F_pump"))
F_probe = float(config.get("parameters", "F_probe"))
detuning = float(config.get("parameters", "detuning (div by hbar)"))/h_bar
omega_probe = float(config.get("parameters", "omega_probe"))
pump_spatial_profile = config.get("parameters", "Pump_spatial_profile")
pump_temporal_profile = config.get("parameters", "Pump_temporal_profile")
probe_spatial_profile = config.get("parameters", "Probe_spatial_profile")
probe_temporal_profile = config.get("parameters", "Probe_temporal_profile")
if initial_state:
    path_ic = config.get("parameters", "initial_state_path")
else:
    path_ic = None
if saved_frame:
    t_save = float(config.get("parameters", "t_save"))

# config.read(folder_bis+"/parameters.txt")

# h_bar_bis = float(config.get("parameters", "h_bar"))
# h_bar_SI_bis = float(config.get("parameters", "h_bar_SI"))
# c_bis = float(config.get("parameters", "c"))
# eV_to_J_bis = float(config.get("parameters", "eV_to_J"))
# rabi_bis = float(config.get("parameters", "rabi (div by 2hbar)"))/2/h_bar
# g0_bis = float(config.get("parameters", "g0 (div by hbar)"))/h_bar
# gamma_exc_bis = float(config.get("parameters", "gamma_exc (div by hbar)"))/h_bar
# gamma_ph_bis = float(config.get("parameters", "gamma_ph (div by hbar)"))/h_bar
# gamma_LP_bis = float(config.get("parameters", "gamma_LP (div by hbar)"))/h_bar
# omega_exc_bis = float(config.get("parameters", "omega_exc (div by hbar)"))/h_bar
# omega_cav_bis = float(config.get("parameters", "omega_cav (div by hbar)"))/h_bar
# n_cav_bis = float(config.get("parameters", "n_cav"))
# k_z_bis = float(config.get("parameters", "k_z"))
# t_min_bis = float(config.get("parameters", "t_min"))
# t_obs_bis = float(config.get("parameters", "t_obs"))
# t_noise_bis = float(config.get("parameters", "t_noise"))
# t_probe_bis = float(config.get("parameters", "t_probe"))
# t_stationary_bis = float(config.get("parameters", "t_stationary"))
# t_max_bis = float(config.get("parameters", "t_max"))
# dt_frame_bis = float(config.get("parameters", "dt_frame"))
# nmax_1_bis = int(config.get("parameters", "nmax_1"))
# nmax_2_bis = int(config.get("parameters", "nmax_2"))
# long_1_bis = float(config.get("parameters", "long_1"))
# long_2_bis = float(config.get("parameters", "long_2"))
# F_pump_bis = float(config.get("parameters", "F_pump"))
# F_probe_bis = float(config.get("parameters", "F_probe"))
# corr_bis = float(config.get("parameters", "corr"))
# detuning_bis = float(config.get("parameters", "detuning (div by hbar)"))/h_bar
# omega_probe_bis = float(config.get("parameters", "omega_probe"))
# Pump_profile_bis = config.get("parameters", "Pump_profile")
# Probe_profile_bis = config.get("parameters", "Probe_profile")
#%%
X, delta_X = cp.linspace(-Lx/2, Lx/2, Nx, retstep = True, dtype=np.float64)
Y, delta_Y = cp.linspace(-Ly/2, Ly/2, Ny, retstep = True, dtype=np.float64)
XX, YY = cp.meshgrid(X, Y)
R = cp.hypot(XX, YY)
THETA = cp.angle(XX + 1j*YY)
Kx = 2 * np.pi * cp.fft.fftfreq(Nx, delta_X)
Ky = 2 * np.pi * cp.fft.fftfreq(Ny, delta_Y)
Kx = cp.fft.fftshift(Kx)
Ky = cp.fft.fftshift(Ky)
Kxx, Kyy = cp.meshgrid(Kx, Ky)

delta = omega_cav - omega_exc # (meV/h_bar)
C02 = np.sqrt(delta**2 + 4*rabi**2) - delta
C02 /= 2*np.sqrt(delta**2 + 4*rabi**2)
X02 = 1 - C02
g_LP = g0*X02**2

side1, side2 = 80, 80 #even number                                              #full window, for movies mainly
window = (Nx//2-side1/2, Nx//2+side1/2, Ny//2-side2/2, Ny//2+side2/2)

cav_field_txy, exc_field_txy, cav_stationary_xy, exc_stationary_xy, hopfield_coefs, F_t = load_raw_data(folder, path_ic)
LP_t_x_y, UP_t_x_y, LP_w_kx_ky, UP_w_kx_ky = polariton_fields(cav_field_txy, exc_field_txy, hopfield_coefs, window = window, dx = delta_X, dy = delta_Y, omega_exc = omega_exc, omega_cav = omega_cav, rabi = rabi, detuning = detuning, k_z = k_z)
LP_stat_x_y, UP_stat_x_y, LP_stat_kx_ky, UP_stat_kx_ky = stationary_polariton_fields(cav_stationary_xy, exc_stationary_xy, hopfield_coefs, window = window, dx = delta_X, dy = delta_Y, omega_exc = omega_exc, omega_cav = omega_cav, rabi = rabi, detuning = detuning, k_z = k_z)

fluctuations_LP = cp.zeros(LP_w_kx_ky.shape, dtype = cp.complex64)
fluctuations_LP = LP_w_kx_ky - LP_stat_kx_ky

# cav_tsave_xy = cp.asarray(np.load(folder + "/raw_arrays/cav_x_y_t%s.npy" %(str(int(t_save)))))
# exc_tsave_xy = cp.asarray(np.load(folder + "/raw_arrays/exc_x_y_t%s.npy" %(str(int(t_save)))))
# hopfield_coefs = cp.asarray(np.load(folder + "/raw_arrays/hopfield_coefs.npy"))

# LP_tsave_kx_ky = cp.zeros(cav_tsave_xy.shape, dtype=cp.complex64)
# Xk = hopfield_coefs[0, :, :]
# Ck = hopfield_coefs[1, :, :]
# cav_tsave_kx_ky = cp.fft.fftn(cav_tsave_xy, axes = (-2, -1))
# exc_tsave_kx_ky = cp.fft.fftn(exc_tsave_xy, axes = (-2, -1))
# LP_tsave_kx_ky[..., :, :] = -1 * Xk[:, :] * exc_tsave_kx_ky[..., :, :] + Ck[:, :] * cav_tsave_kx_ky[..., :, :] 
# LP_tsave_x_y = cp.fft.ifftn(LP_tsave_kx_ky, axes = (-2, -1))
# LP_tsave_kx_ky[..., :, :] = cp.fft.fftshift(LP_tsave_kx_ky[..., :, :], axes = (-2, -1))




#Phase and density movies:
# movies(folder, LP_t_x_y[:,:,:], title = "LP")
# movies(folder, UP_t_x_y[:,:,:], title = "UP")
# movies(folder, cav_field_txy[:,:,:], title = "cav")
# movies(folder, exc_field_txy[:,:,:], title = "exc")


#Plot bistability cycle:
# plot_gnLP_vs_I(folder, LP_t_x_y[:,:,:], F_t, R, g_LP, gamma_exc, gamma_cav, X02, C02, detuning = None, theoretical = False) #theoretical = False because it does not work aha

#Plotting stationary densities of LP:
# plot_density(folder, ("$k_x$", Kx[window[0]:window[1]+1]), ("$k_y$", Ky[window[2]:window[3]+1]), ("stationary_LP_kspace", LP_stat_kx_ky[:,:]))
# plot_density(folder, ("$x$", X[window[0]:window[1]+1]), ("$y$", Y[window[2]:window[3]+1]), ("stationary_LP_rspace", LP_stat_x_y))
# plot_density(folder, ("$x$", X[LP_stat_x_y.shape[0]//2-80:LP_stat_x_y.shape[0]//2+81]), ("$y$", Y[LP_stat_x_y.shape[1]//2-80:LP_stat_x_y.shape[1]//2+81]), ("stationary_LP_rspace_160x160", LP_stat_x_y[LP_stat_x_y.shape[0]//2-80:LP_stat_x_y.shape[0]//2+81, LP_stat_x_y.shape[1]//2-80:LP_stat_x_y.shape[1]//2+81]))
# plot_density(folder, ("$k_x$", Kx[LP_stat_kx_ky.shape[0]//2-80:LP_stat_kx_ky.shape[0]//2+81]), ("$k_y$", Ky[LP_stat_kx_ky.shape[1]//2-80:LP_stat_kx_ky.shape[1]//2+81]), ("stationary_LP_kspace_160x160", LP_stat_kx_ky[LP_stat_kx_ky.shape[0]//2-80:LP_stat_kx_ky.shape[0]//2+81, LP_stat_kx_ky.shape[1]//2-80:LP_stat_kx_ky.shape[1]//2+81]))

# plot_density(folder, ("$k_x$", Kx[window[0]:window[1]+1]), ("$k_y$", Ky[window[2]:window[3]+1]), ("t1800_LP_kspace", LP_tsave_kx_ky[:,:]))
# plot_density(folder, ("$x$", X[window[0]:window[1]+1]), ("$y$", Y[window[2]:window[3]+1]), ("t1800_LP_rspace", LP_tsave_x_y))
# plot_density(folder, ("$x$", X[LP_tsave_x_y.shape[0]//2-320:LP_tsave_x_y.shape[0]//2+321]), ("$y$", Y[LP_tsave_x_y.shape[1]//2-320:LP_tsave_x_y.shape[1]//2+321]), ("t1800_LP_rspace_640x640", LP_tsave_x_y[LP_tsave_x_y.shape[0]//2-320:LP_tsave_x_y.shape[0]//2+321, LP_tsave_x_y.shape[1]//2-320:LP_tsave_x_y.shape[1]//2+321]))
# plot_density(folder, ("$k_x$", Kx[LP_tsave_kx_ky.shape[0]//2-320:LP_tsave_kx_ky.shape[0]//2+321]), ("$k_y$", Ky[LP_tsave_kx_ky.shape[1]//2-320:LP_tsave_kx_ky.shape[1]//2+321]), ("t1800_LP_kspace_640x640", LP_tsave_kx_ky[LP_tsave_kx_ky.shape[0]//2-320:LP_tsave_kx_ky.shape[0]//2+321, LP_tsave_kx_ky.shape[1]//2-320:LP_tsave_kx_ky.shape[1]//2+321]))


#Plotting time evolution of avg LP density around center and laser intensity
# time_plot = np.array([i*dt_frame for i in range(len(cav_field_txy))])
# F_intensity = cp.abs(F_t)**2
# LP_density = cp.abs(LP_t_x_y)**2
# avg_density = cp.zeros(len(F_t))
# radius=15
# disk = cp.zeros((Nx, Ny))
# disk[R < radius] += 1
# for i in range(len(F_t)):
#     avg_density[i] += cp.average(LP_density[i], axis=(-2, -1), weights = disk)
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# ax1.plot(time_plot, avg_density.get(), label = "$n_{LP}(0,0)$", color = "r")
# ax2.plot(time_plot, F_intensity.get(), label = "$I(0,0)$", color = "b", linestyle = "--")
# ax1.set_xlabel("Time (ps)")
# ax1.set_ylabel("$\psi_{LP}$")
# ax2.set_ylabel("$I$")
# fig.legend()
# fig.savefig(folder+"/laser_LP_densities.png")
# plt.close("all")

# disk = cp.zeros((Nx, Ny))
# radius = 15
# disk[R < radius] += 1
# avg_density_t1800 = cp.average(cp.abs(LP_tsave_x_y)**2, axis=(-2, -1), weights = disk)
# print(avg_density_t1800)


# #Avg relative error between fields ar t and t+dt
# print("frames: "+str(cav_field_txy.shape[0]-35))
# cav_rel_error = cp.zeros((cav_field_txy.shape[0]-35, Nx-20, Ny-20), dtype = cp.complex64)
# exc_rel_error = cp.zeros((cav_field_txy.shape[0]-35, Nx-20, Ny-20), dtype = cp.complex64)
# cav_avg_rel_error = cp.zeros(cav_field_txy.shape[0]-35, dtype = cp.complex64)
# exc_avg_rel_error = cp.zeros(cav_field_txy.shape[0]-35, dtype = cp.complex64)

# for i in range(cav_field_txy.shape[0]-35):
#     cav_rel_error[i] = (cav_field_txy[i+1, 10:-10, 10:-10] - cav_field_txy[i, 10:-10, 10:-10])/cav_field_txy[i, 10:-10, 10:-10]
#     exc_rel_error[i] = (exc_field_txy[i+1, 10:-10, 10:-10] - exc_field_txy[i, 10:-10, 10:-10])/exc_field_txy[i, 10:-10, 10:-10]

# for i in range(cav_field_txy.shape[0]-35):    
#     cav_avg_rel_error[i] = cp.average(cav_rel_error[i], axis = (-2, -1))
#     exc_avg_rel_error[i] = cp.average(exc_rel_error[i], axis = (-2, -1))

# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# ax1.plot(np.arange(1800, 1800+0.0078125*(cav_field_txy.shape[0]-35), 0.0078125), np.real(cav_avg_rel_error.get()), label = "Real part", color = "r")
# ax2.plot(np.arange(1800, 1800+0.0078125*(cav_field_txy.shape[0]-35), 0.0078125), np.imag(cav_avg_rel_error.get()), label = "Imaginary part", color = "b", linestyle = "--")
# ax1.set_xlabel("Time (ps)")
# fig.legend()
# fig.savefig(folder+"/cav_avg_relative_errors_251frames_noborder.png")

# fig2, ax12 = plt.subplots()
# ax22 = ax12.twinx()
# ax12.plot(np.arange(1800, 1800+0.0078125*(cav_field_txy.shape[0]-35), 0.0078125), np.real(exc_avg_rel_error.get()), label = "Real part", color = "r")
# ax22.plot(np.arange(1800, 1800+0.0078125*(cav_field_txy.shape[0]-35), 0.0078125), np.imag(exc_avg_rel_error.get()), label = "Imaginary part", color = "b", linestyle = "--")
# ax12.set_xlabel("Time (ps)")
# fig2.legend()
# fig2.savefig(folder+"/exc_avg_relative_errors_frames_251noborder.png")


# #Avg relative error between argument's gradient of LP at t and t+dt
# arg_LP = cp.angle(LP_t_x_y)
# grad_arg_LP_1, grad_arg_LP_2 = cp.gradient(arg_LP, axis = (-2, -1))
# grad_arg_field = cp.zeros((2, LP_t_x_y.shape[0], Nx-20,Ny-20), dtype = cp.complex64)
# grad_arg_field[0] = grad_arg_LP_1[:,10:-10, 10:-10]
# grad_arg_field[1] = grad_arg_LP_2[:,10:-10, 10:-10]

# modulus_LP = cp.zeros((LP_t_x_y.shape[0], Nx-20, Ny-20), dtype = cp.complex64)
# modulus_LP = cp.sqrt(grad_arg_field[0]**2+grad_arg_field[1]**2)

# modulus_rel_error = cp.zeros((LP_t_x_y.shape[0]-35, Nx-20, Ny-20), dtype = cp.complex64)
# modulus_avg_rel_error = cp.zeros(LP_t_x_y.shape[0]-35, dtype = cp.complex64)

# dot_rel_error = cp.zeros((LP_t_x_y.shape[0]-35, Nx-20, Ny-20), dtype = cp.complex64)
# dot_avg_rel_error = cp.zeros(LP_t_x_y.shape[0]-35, dtype = cp.complex64)

# for i in range(LP_t_x_y.shape[0]-35):
#     modulus_rel_error[i] = (modulus_LP[i+1] - modulus_LP[i])/modulus_LP[i]
#     dot_rel_error[i] = (grad_arg_field[0][i+1]*grad_arg_field[0][i] + grad_arg_field[1][i+1]*grad_arg_field[1][i])/(modulus_LP[i+1]*modulus_LP[i])
# for i in range(LP_t_x_y.shape[0]-35):    
#     modulus_avg_rel_error[i] = cp.average(modulus_rel_error[i], axis = (-2, -1))
#     dot_avg_rel_error[i] = cp.average(dot_rel_error[i], axis = (-2, -1))

# plt.figure()
# plt.plot(np.arange(1800, 1800+0.0078125*(LP_t_x_y.shape[0]-35), 0.0078125), modulus_avg_rel_error.get())
# plt.xlabel("Time (ps)")
# plt.ylabel("Average relative error modulus of gradient of arg")
# plt.savefig(folder+"/modulus_avg_relative_errors_frames_251noborder.png")

# plt.figure()
# plt.plot(np.arange(1800, 1800+0.0078125*(LP_t_x_y.shape[0]-35), 0.0078125), dot_avg_rel_error.get())
# plt.xlabel("Time (ps)")
# plt.ylim(0.9999, 1.0001)
# plt.ylabel("Average relative error projection of gradient of arg")
# plt.savefig(folder+"/projection_avg_relative_errors_frames_251noborder.png")


#Plot dispersions:
omega_list = 2*cp.pi*cp.fft.fftshift(cp.fft.fftfreq(fluctuations_LP.shape[-3], dt_frame))
fluctuations_LP[...,:,fluctuations_LP.shape[-2]//2,fluctuations_LP.shape[-1]//2] = 0
fluctuations_LP[...,omega_list.shape[0]//2,:,:] = 0
plot_density(folder, ("$k_x$",Kx[window[0]:window[1]+1]), ("$\omega$", omega_list[omega_list.shape[0]//2-40:omega_list.shape[0]//2+40]), ("zoomedandupsidedown_w_kx_ky=0", fluctuations_LP[..., omega_list.shape[0]//2-40:omega_list.shape[0]//2+40:,fluctuations_LP.shape[-2]//2,:]), norm="log") #careful with the order of the slicins, it is not intuitive imo but this is the correct way
plot_density(folder, ("$k_y$", Ky[window[2]:window[3]+1]), ("$\omega$", omega_list[omega_list.shape[0]//2-40:omega_list.shape[0]//2+40]), ("zoomedandupsidedown_w_kx=0_ky", fluctuations_LP[..., omega_list.shape[0]//2-40:omega_list.shape[0]//2+40:,:,fluctuations_LP.shape[-1]//2]), norm="log")
plot_density(folder, ("$k_x$",Kx[window[0]:window[1]+1]), ("$\omega$", omega_list), ("w_kx_ky=0", fluctuations_LP[..., ::-1,fluctuations_LP.shape[-2]//2,:]), norm="log") #careful with the order of the slicins, it is not intuitive imo but this is the correct way
plot_density(folder, ("$k_y$", Ky[window[2]:window[3]+1]), ("$\omega$", omega_list), ("w_kx=0_ky", fluctuations_LP[..., ::-1,:,fluctuations_LP.shape[-1]//2]), norm="log")
