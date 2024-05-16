import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import analysis_functions as af
import configparser

# In this file you may find the analysis of the 3 simulation examples that can be found in this folder. 
# You can run them by uncommenting the corresponding code, completing the directories and running this file.

#------------------------------------------------------------------------------------------
#EXAMPLE 1: bistability cycle of a tophat pump:

# Load data and plotting parameters
directory = "/home" #complete with your directory
folder = directory + "/data_set_bistab_cycle_tophat80"

initial_state = False

af.config_plots()

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
#%%

X, delta_X = cp.linspace(-Lx/2, Lx/2, Nx, retstep = True, dtype=np.float64)
Y, delta_Y = cp.linspace(-Ly/2, Ly/2, Ny, retstep = True, dtype=np.float64)
XX, YY = cp.meshgrid(X, Y)
Kx = 2 * np.pi * cp.fft.fftfreq(Nx, delta_X)
Ky = 2 * np.pi * cp.fft.fftfreq(Ny, delta_Y)
R = cp.hypot(XX, YY)
delta = omega_cav - omega_exc # (meV/h_bar)
C02 = np.sqrt(delta**2 + 4*rabi**2) - delta
C02 /= 2*np.sqrt(delta**2 + 4*rabi**2)
X02 = 1 - C02
g_LP = g0*X02**2


cav_field_txy, exc_field_txy, cav_stationary_xy, exc_stationary_xy, hopfield_coefs, F_t = af.load_raw_data(folder, path_ic)
LP_t_x_y = af.polariton_fields(cav_field_txy, exc_field_txy, hopfield_coefs, only_LP = True, only_rspace = True)
LP_stat_x_y, LP_stat_kx_ky = af.stationary_polariton_fields(cav_stationary_xy, exc_stationary_xy, hopfield_coefs, only_LP = True, only_rspace = False)


#Phase and density movies:
af.movies(folder, LP_t_x_y)

#Plot bistability cycle:
af.plot_gnLP_vs_I(folder, LP_t_x_y, F_t, R, g_LP, gamma_exc, gamma_cav, X02, C02, h_bar = h_bar, detuning = detuning * h_bar, theoretical = False) #theoretical = False because it does not work yet aha

#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
#EXAMPLE 2: stationary state of a tophat pump and saving initial condition for later use


# # Load data and plotting parameters
# directory = "/home" #complete with your directory
# folder = directory + "/data_set_stationary_state_at_turning_point_tophat80"

# initial_state = False

# af.config_plots()

# #%%
# config = configparser.ConfigParser()
# config.read(folder+"/parameters.txt")

# h_bar = float(config.get("parameters", "h_bar"))
# h_bar_SI = float(config.get("parameters", "h_bar_SI"))
# c = float(config.get("parameters", "c"))
# eV_to_J = float(config.get("parameters", "eV_to_J"))
# rabi = float(config.get("parameters", "rabi (div by 2hbar)"))/2/h_bar
# g0 = float(config.get("parameters", "g0 (div by hbar)"))/h_bar
# gamma_exc = float(config.get("parameters", "gamma_exc (div by hbar)"))/h_bar
# gamma_cav = float(config.get("parameters", "gamma_cav (div by hbar)"))/h_bar
# omega_exc = float(config.get("parameters", "omega_exc (div by hbar)"))/h_bar
# omega_cav = float(config.get("parameters", "omega_cav (div by hbar)"))/h_bar
# n_cav = float(config.get("parameters", "n_cav"))
# k_z = float(config.get("parameters", "k_z"))
# t_min = float(config.get("parameters", "t_min"))
# t_obs = float(config.get("parameters", "t_obs"))
# t_noise = float(config.get("parameters", "t_noise"))
# t_probe = float(config.get("parameters", "t_probe"))
# t_stationary = float(config.get("parameters", "t_stationary"))
# t_max = float(config.get("parameters", "t_max"))
# dt_frame = float(config.get("parameters", "dt_frame"))
# Nx = int(config.get("parameters", "Nx"))
# Ny = int(config.get("parameters", "Ny"))
# Lx = float(config.get("parameters", "Lx"))
# Ly = float(config.get("parameters", "Ly"))
# F_pump = float(config.get("parameters", "F_pump"))
# F_probe = float(config.get("parameters", "F_probe"))
# detuning = float(config.get("parameters", "detuning (div by hbar)"))/h_bar
# omega_probe = float(config.get("parameters", "omega_probe"))
# pump_spatial_profile = config.get("parameters", "Pump_spatial_profile")
# pump_temporal_profile = config.get("parameters", "Pump_temporal_profile")
# probe_spatial_profile = config.get("parameters", "Probe_spatial_profile")
# probe_temporal_profile = config.get("parameters", "Probe_temporal_profile")
# if initial_state:
#     path_ic = config.get("parameters", "initial_state_path")
# else:
#     path_ic = None
# #%%

# X, delta_X = cp.linspace(-Lx/2, Lx/2, Nx, retstep = True, dtype=np.float64)
# Y, delta_Y = cp.linspace(-Ly/2, Ly/2, Ny, retstep = True, dtype=np.float64)
# XX, YY = cp.meshgrid(X, Y)
# Kx = 2 * np.pi * cp.fft.fftfreq(Nx, delta_X)
# Ky = 2 * np.pi * cp.fft.fftfreq(Ny, delta_Y)
# R = cp.hypot(XX, YY)
# delta = omega_cav - omega_exc # (meV/h_bar)
# C02 = np.sqrt(delta**2 + 4*rabi**2) - delta
# C02 /= 2*np.sqrt(delta**2 + 4*rabi**2)
# X02 = 1 - C02
# g_LP = g0*X02**2


# cav_field_txy, exc_field_txy, cav_stationary_xy, exc_stationary_xy, hopfield_coefs, F_t = af.load_raw_data(folder, path_ic)
# LP_t_x_y = af.polariton_fields(cav_field_txy, exc_field_txy, hopfield_coefs, only_LP = True, only_rspace = True)
# LP_stat_x_y, LP_stat_kx_ky = af.stationary_polariton_fields(cav_stationary_xy, exc_stationary_xy, hopfield_coefs, only_LP = True, only_rspace = False)


# #Phase and density movies:
# af.movies(folder, LP_t_x_y)

# #Plot bistability cycle:
# af.plot_gnLP_vs_I(folder, LP_t_x_y, F_t, R, g_LP, gamma_exc, gamma_cav, X02, C02, h_bar = h_bar, detuning = detuning * h_bar, theoretical = False) #theoretical = False because it does not work yet aha

# #Plotting time evolution of the average LP density around the center and laser intensity (create a function for this)
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


#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------

#EXAMPLE 3: tophat pump from initial condition, with noise and probe at resonance: seeing the dispersion relation

# # Load data and plotting parameters
# directory = "/home" #complete with your directory
# folder = directory + "/data_set_dispersion_and_probe_k05_w05"

# initial_state = True

# af.config_plots()

# #%%
# config = configparser.ConfigParser()
# config.read(folder+"/parameters.txt")

# h_bar = float(config.get("parameters", "h_bar"))
# h_bar_SI = float(config.get("parameters", "h_bar_SI"))
# c = float(config.get("parameters", "c"))
# eV_to_J = float(config.get("parameters", "eV_to_J"))
# rabi = float(config.get("parameters", "rabi (div by 2hbar)"))/2/h_bar
# g0 = float(config.get("parameters", "g0 (div by hbar)"))/h_bar
# gamma_exc = float(config.get("parameters", "gamma_exc (div by hbar)"))/h_bar
# gamma_cav = float(config.get("parameters", "gamma_cav (div by hbar)"))/h_bar
# omega_exc = float(config.get("parameters", "omega_exc (div by hbar)"))/h_bar
# omega_cav = float(config.get("parameters", "omega_cav (div by hbar)"))/h_bar
# n_cav = float(config.get("parameters", "n_cav"))
# k_z = float(config.get("parameters", "k_z"))
# t_min = float(config.get("parameters", "t_min"))
# t_obs = float(config.get("parameters", "t_obs"))
# t_noise = float(config.get("parameters", "t_noise"))
# t_probe = float(config.get("parameters", "t_probe"))
# t_stationary = float(config.get("parameters", "t_stationary"))
# t_max = float(config.get("parameters", "t_max"))
# dt_frame = float(config.get("parameters", "dt_frame"))
# Nx = int(config.get("parameters", "Nx"))
# Ny = int(config.get("parameters", "Ny"))
# Lx = float(config.get("parameters", "Lx"))
# Ly = float(config.get("parameters", "Ly"))
# F_pump = float(config.get("parameters", "F_pump"))
# F_probe = float(config.get("parameters", "F_probe"))
# detuning = float(config.get("parameters", "detuning (div by hbar)"))/h_bar
# omega_probe = float(config.get("parameters", "omega_probe"))
# pump_spatial_profile = config.get("parameters", "Pump_spatial_profile")
# pump_temporal_profile = config.get("parameters", "Pump_temporal_profile")
# probe_spatial_profile = config.get("parameters", "Probe_spatial_profile")
# probe_temporal_profile = config.get("parameters", "Probe_temporal_profile")
# if initial_state:
#     path_ic = config.get("parameters", "initial_state_path")
# else:
#     path_ic = None
# #%%

# X, delta_X = cp.linspace(-Lx/2, Lx/2, Nx, retstep = True, dtype=np.float64)
# Y, delta_Y = cp.linspace(-Ly/2, Ly/2, Ny, retstep = True, dtype=np.float64)
# Kx = cp.fft.fftshift(2 * np.pi * cp.fft.fftfreq(Nx, delta_X))
# Ky = cp.fft.fftshift(2 * np.pi * cp.fft.fftfreq(Ny, delta_Y))


# cav_field_txy, exc_field_txy, cav_stationary_xy, exc_stationary_xy, hopfield_coefs, F_t = af.load_raw_data(folder, path_ic)
# LP_t_x_y = af.polariton_fields(cav_field_txy, exc_field_txy, hopfield_coefs, only_LP = True, only_rspace = True)
# LP_stat_x_y, LP_stat_kx_ky = af.stationary_polariton_fields(cav_stationary_xy, exc_stationary_xy, hopfield_coefs, only_LP = True, only_rspace = False)

# #Zoom in a region of homogeneous density
# window = (Nx//2-40, Nx//2+41, Ny//2-40, Ny//2+41)
# fluctuations_LP_txy = cp.zeros(LP_t_x_y[:, window[0]:window[1], window[2]:window[3]].shape, dtype = cp.complex64)
# fluctuations_LP = cp.zeros(LP_t_x_y[:, window[0]:window[1], window[2]:window[3]].shape, dtype = cp.complex64)
# fluctuations_LP_txy = LP_t_x_y[:,window[0]:window[1], window[2]:window[3]] - LP_stat_x_y[window[0]:window[1], window[2]:window[3]]
# fluctuations_LP = cp.fft.fftshift(cp.fft.fftn(fluctuations_LP_txy, axes = (-3,-2,-1)), axes = (-3,-2,-1))
# omega_list = 2*cp.pi*cp.fft.fftshift(cp.fft.fftfreq(fluctuations_LP.shape[-3], dt_frame))
# k_1_list = 2*cp.pi*cp.fft.fftshift(cp.fft.fftfreq(fluctuations_LP.shape[-2], (X[window[1]]-X[window[0]])/(window[1]-window[0]-1))) 
# k_2_list = 2*cp.pi*cp.fft.fftshift(cp.fft.fftfreq(fluctuations_LP.shape[-1], (X[window[3]]-X[window[2]])/(window[3]-window[2]-1)))


# #Plotting stationary densities of LP:
# af.plot_density(folder, ("$k_x$", Kx), ("$k_y$", Ky), ("stationary_LP_kspace", LP_stat_kx_ky))
# af.plot_density(folder, ("$x$", X), ("$y$", Y), ("stationary_LP_rspace", LP_stat_x_y))

# #Plotting dispersions 
# fluctuations_LP[...,:,fluctuations_LP.shape[-2]//2,fluctuations_LP.shape[-1]//2] = 0
# fluctuations_LP[...,omega_list.shape[0]//2,:,:] = 0
# af.plot_density(folder, ("$k_x$",k_1_list), ("$\omega$", omega_list), ("w_kx_ky=0", fluctuations_LP[..., ::-1,fluctuations_LP.shape[-2]//2,:]), norm="log") #careful with the order of the slicins, it is not intuitive imo but this is the correct way
# af.plot_density(folder, ("$k_y$", k_2_list), ("$\omega$", omega_list), ("w_kx=0_ky", fluctuations_LP[..., ::-1,:,fluctuations_LP.shape[-1]//2]), norm="log")
# af.plot_density(folder, ("$k_x$", k_1_list), ("$k_y$", k_2_list), ("w="+str(omega_list[omega_list.shape[0]//3*2])+"_kx_ky", fluctuations_LP[..., omega_list.shape[0]//3*2,:,:]), norm="log")

#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------

#EXAMPLE 4: same as 3 but with parallelized simulations

# # Load data and plotting parameters
# directory = "/home" #complete with your directory
# folder = directory + "/data_set_2k_2w_in_parallel"

# initial_state = True

# af.config_plots()

# #%%
# config = configparser.ConfigParser()
# config.read(folder+"/parameters.txt")

# h_bar = float(config.get("parameters", "h_bar"))
# h_bar_SI = float(config.get("parameters", "h_bar_SI"))
# c = float(config.get("parameters", "c"))
# eV_to_J = float(config.get("parameters", "eV_to_J"))
# rabi = float(config.get("parameters", "rabi (div by 2hbar)"))/2/h_bar
# g0 = float(config.get("parameters", "g0 (div by hbar)"))/h_bar
# gamma_exc = float(config.get("parameters", "gamma_exc (div by hbar)"))/h_bar
# gamma_cav = float(config.get("parameters", "gamma_cav (div by hbar)"))/h_bar
# omega_exc = float(config.get("parameters", "omega_exc (div by hbar)"))/h_bar
# omega_cav = float(config.get("parameters", "omega_cav (div by hbar)"))/h_bar
# n_cav = float(config.get("parameters", "n_cav"))
# k_z = float(config.get("parameters", "k_z"))
# t_min = float(config.get("parameters", "t_min"))
# t_obs = float(config.get("parameters", "t_obs"))
# t_noise = float(config.get("parameters", "t_noise"))
# t_probe = float(config.get("parameters", "t_probe"))
# t_stationary = float(config.get("parameters", "t_stationary"))
# t_max = float(config.get("parameters", "t_max"))
# dt_frame = float(config.get("parameters", "dt_frame"))
# Nx = int(config.get("parameters", "Nx"))
# Ny = int(config.get("parameters", "Ny"))
# Lx = float(config.get("parameters", "Lx"))
# Ly = float(config.get("parameters", "Ly"))
# F_pump = float(config.get("parameters", "F_pump"))
# F_probe = float(config.get("parameters", "F_probe"))
# detuning = float(config.get("parameters", "detuning (div by hbar)"))/h_bar
# omega_probe = float(config.get("parameters", "omega_probe"))
# pump_spatial_profile = config.get("parameters", "Pump_spatial_profile")
# pump_temporal_profile = config.get("parameters", "Pump_temporal_profile")
# probe_spatial_profile = config.get("parameters", "Probe_spatial_profile")
# probe_temporal_profile = config.get("parameters", "Probe_temporal_profile")
# if initial_state:
#     path_ic = config.get("parameters", "initial_state_path")
# else:
#     path_ic = None
# #%%

# X, delta_X = cp.linspace(-Lx/2, Lx/2, Nx, retstep = True, dtype=np.float64)
# Y, delta_Y = cp.linspace(-Ly/2, Ly/2, Ny, retstep = True, dtype=np.float64)
# Kx = cp.fft.fftshift(2 * np.pi * cp.fft.fftfreq(Nx, delta_X))
# Ky = cp.fft.fftshift(2 * np.pi * cp.fft.fftfreq(Ny, delta_Y))



# cav_field_txy, exc_field_txy, cav_stationary_xy, exc_stationary_xy, hopfield_coefs, F_t = af.load_raw_data(folder, path_ic)
# LP_t_x_y = af.polariton_fields(cav_field_txy, exc_field_txy, hopfield_coefs, only_LP = True, only_rspace = True)
# LP_stat_x_y, LP_stat_kx_ky = af.stationary_polariton_fields(cav_stationary_xy, exc_stationary_xy, hopfield_coefs, only_LP = True, only_rspace = False)

# window = (Nx//2-40, Nx//2+41, Ny//2-40, Ny//2+41)
# fluctuations_LP_txy = cp.zeros(LP_t_x_y[..., :, window[0]:window[1], window[2]:window[3]].shape, dtype = cp.complex64)
# fluctuations_LP = cp.zeros(LP_t_x_y[..., :, window[0]:window[1], window[2]:window[3]].shape, dtype = cp.complex64)
# fluctuations_LP_txy = LP_t_x_y[..., :,window[0]:window[1], window[2]:window[3]] - LP_stat_x_y[window[0]:window[1], window[2]:window[3]]
# fluctuations_LP = cp.fft.fftshift(cp.fft.fftn(fluctuations_LP_txy, axes = (-3,-2,-1)), axes = (-3,-2,-1))
# omega_list = 2*cp.pi*cp.fft.fftshift(cp.fft.fftfreq(fluctuations_LP.shape[-3], dt_frame))
# k_1_list = 2*cp.pi*cp.fft.fftshift(cp.fft.fftfreq(fluctuations_LP.shape[-2], (X[window[1]]-X[window[0]])/(window[1]-window[0]-1))) #in 1/um PROBABLY TO CHECK ARGH
# k_2_list = 2*cp.pi*cp.fft.fftshift(cp.fft.fftfreq(fluctuations_LP.shape[-1], (X[window[3]]-X[window[2]])/(window[3]-window[2]-1)))


# #Plotting stationary densities of LP:
# af.plot_density(folder, ("$k_x$", Kx), ("$k_y$", Ky), ("stationary_LP_kspace", LP_stat_kx_ky))
# af.plot_density(folder, ("$x$", X), ("$y$", Y), ("stationary_LP_rspace", LP_stat_x_y))

# #Plotting dispersions 
# fluctuations_LP[...,:,fluctuations_LP.shape[-2]//2,fluctuations_LP.shape[-1]//2] = 0
# fluctuations_LP[...,omega_list.shape[0]//2,:,:] = 0
# af.plot_density(folder, ("$k_x$",k_1_list), ("$\omega$", omega_list), ("w_kx_ky=0", fluctuations_LP[..., ::-1,fluctuations_LP.shape[-2]//2,:]), norm="log") #careful with the order of the slicins, it is not intuitive imo but this is the correct way
# af.plot_density(folder, ("$k_y$", k_2_list), ("$\omega$", omega_list), ("w_kx=0_ky", fluctuations_LP[..., ::-1,:,fluctuations_LP.shape[-1]//2]), norm="log")
# af.plot_density(folder, ("$k_x$", k_1_list), ("$k_y$", k_2_list), ("w="+str(omega_list[omega_list.shape[0]//3*2])+"_kx_ky", fluctuations_LP[..., omega_list.shape[0]//3*2,:,:]), norm="log")