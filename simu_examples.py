import cupy as cp
from ggpe2d import ggpe
import field_creation_functions as fc
import physical_constants as cte
import os
from analysis_functions import load_raw_data
import analysis_functions as af
import numpy as np
import matplotlib.pyplot as plt
import configparser


# In this file you may find 3 different simulation examples. You can run them by uncommenting the corresponding code, completing the directories and running this file.

def save_raw_data(folder,parameters):
    
    #Save experimets parameters
    with open(folder+"/parameters.txt", "w") as f:
        f.write('[parameters]\n')
        f.write('folder: ' + folder)
        f.write('\n')
        f.write('\n'.join('{}: {}'.format(x[0],x[1]) for x in parameters))

    #Import from simu arrays to be saved
    mean_cav_t_x_y = simu.mean_cav_t_x_y
    mean_exc_t_x_y = simu.mean_exc_t_x_y
    mean_den_reservoir_t_x_y = simu.mean_den_reservoir_t_x_y
    stationary_cav_x_y = simu.mean_cav_x_y_stat
    stationary_exc_x_y = simu.mean_exc_x_y_stat
    hopfield_coefs = simu.hopfield_coefs
    F_t = simu.F_t

    #Save data as numpy arrays
    cp.save(folder+"/raw_arrays/mean_cav_t_x_y", mean_cav_t_x_y)
    cp.save(folder+"/raw_arrays/mean_exc_t_x_y", mean_exc_t_x_y)
    cp.save(folder+"/raw_arrays/mean_den_reservoir_t_x_y", mean_den_reservoir_t_x_y)
    if stationary_cav_x_y is not None:
        cp.save(folder+"/raw_arrays/stationary_cav_x_y", stationary_cav_x_y)
    if stationary_exc_x_y is not None:
        cp.save(folder+"/raw_arrays/stationary_exc_x_y", stationary_exc_x_y)
    cp.save(folder+"/raw_arrays/hopfield_coefs", hopfield_coefs)
    cp.save(folder+"/raw_arrays/F_t", F_t)


#------------------------------------------------------------------------------------------
#EXAMPLE 1: bistability cycle of a tophat pump:

# # Laser parameters
# detuning = 0.17/cte.h_bar # (meV/hbar) detuning between the pump and the LP energy
# F_pump = 1.1
# F_probe = 0

# # Grid parameters
# Lx, Ly = 256, 256
# Nx, Ny = 256, 256

# if (Lx/Nx)**2<cte.g0/cte.gamma_cav or (Ly/Ny)**2<cte.g0/cte.gamma_cav:
#     print("WARNING: TWA NOT VALID")
    
# # Time parameters
# t_min = 0 # (ps) initial time of evolution
# t_max = 500 # (ps) final time of evolution
# t_stationary = 1e9
# t_noise = 1e9 # (ps) time from which the noise starts
# t_probe = 1e9 # (ps) time from which the probe starts
# t_obs = 0 # (ps) time from which the observation starts
# dt_frame = 1/(1) #cst/delta_E avec delta_E la résolution/omega_max en énergie en meV // delta_E fixes the window you will see without aliasing in frequencies, delta_E*2pi/2 = nyquist frequency 
# n_frame = int((t_max-t_obs)/dt_frame)+1
# print("dt_frame is %s"%(dt_frame))
# print("n_frame is %s"%(n_frame))
# omega_probe = 0
# k_probe = 0

# simu = ggpe(cte.omega_exc, cte.omega_cav, cte.gamma_exc, cte.gamma_cav, cte.g0, cte.rabi, cte.k_z,
#             detuning, F_pump, F_probe, 
#             t_max, t_stationary, t_obs, dt_frame, t_noise,
#             Lx, Ly, Nx, Ny)

# simu.pump_spatial_profile = fc.tophat(simu.F_pump_r, simu.R, radius = 80)
# simu.pump_temporal_profile = fc.bistab_cycle(simu.F_pump_t, simu.time, simu.t_max)

# #Run simulation and save data
# folder_DATA = "/home/killian/LEON/DATA/Polaritons/Reservoir/simu"
# string_name="/tophat50_turning_point"

# try:
#     os.mkdir(folder_DATA)
# except:
#     print("folder already created")

# folder_DATA += "/data_set" + string_name
# print("/data_set" + string_name)

# try:
#     os.mkdir(folder_DATA)
# except:
#     print("folder already created")
    
# try:
#     os.mkdir(folder_DATA + "/raw_arrays")
# except:
#     print("folder already created")
    
# parameters = [('h_bar',cte.h_bar), ('h_bar_SI', cte.h_bar_SI), ('c', cte.c), ('eV_to_J', cte.eV_to_J), ('n_cav', cte.n_cav), 
#               ('omega_exc (div by hbar)', cte.omega_exc*cte.h_bar), ('omega_cav (div by hbar)', cte.omega_cav*cte.h_bar), ('gamma_exc (div by hbar)', cte.gamma_exc*cte.h_bar), ('gamma_cav (div by hbar)', cte.gamma_cav*cte.h_bar), 
#               ('g0 (div by hbar)', cte.g0*cte.h_bar), ('rabi (div by 2hbar)', cte.rabi*2*cte.h_bar), ('k_z', cte.k_z), ('detuning (div by hbar)', detuning*cte.h_bar), 
#               ('F_pump', F_pump), ('F_probe', F_probe), ('t_min', t_min), ('t_max', t_max), ('t_stationary', t_stationary), ('t_obs', t_obs), ('dt_frame', dt_frame), ('t_noise', t_noise), ('t_probe', t_probe), 
#               ('Nx', Nx), ('Ny', Ny), ('Lx', Lx), ('Ly', Ly),
#               ('omega_probe', omega_probe), ('Pump_spatial_profile', simu.pump_spatial_profile), ('Pump_temporal_profile', simu.pump_temporal_profile), ('Probe_spatial_profile', simu.probe_spatial_profile), ('Probe_temporal_profile', simu.probe_temporal_profile), ("Potential_profile", simu.potential_profile)] 

# simu.evolution()
# save_raw_data(folder_DATA, parameters)

#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
#EXAMPLE 2: stationary state with reservoir of a tophat pump and saving initial condition for later use:

# Laser parameters
detuning = 0.2/cte.h_bar # (meV/hbar) detuning between the pump and the LP energy
F_pump = 0.45
F_probe = 0

# Grid parameters
Lx, Ly = 256, 256
Nx, Ny = 256, 256

if (Lx/Nx)**2<cte.g0/cte.gamma_cav or (Ly/Ny)**2<cte.g0/cte.gamma_cav:
    print("WARNING: TWA NOT VALID")
    
# Time parameters
t_min = 0 # (ps) initial time of evolution
t_max = 1500 # (ps) final time of evolution
t_stationary = 950
t_noise = 1e9 # (ps) time from which the noise starts
t_probe = 1e9 # (ps) time from which the probe starts
t_obs = 0 # (ps) time from which the observation starts
cst = 4
dt_frame = 1/(0.1) #cst/delta_E avec delta_E la résolution/omega_max en énergie en meV // delta_E fixes the window you will see without aliasing in frequencies, delta_E*2pi/2 = nyquist frequency 
n_frame = int((t_max-t_obs)/dt_frame)+1
print("dt_frame is %s"%(dt_frame))
print("n_frame is %s"%(n_frame))
omega_probe = 0
k_probe = 0

apply_reservoir = True
simu = ggpe(cte.omega_exc, cte.omega_cav, cte.gamma_exc, cte.gamma_cav, cte.gamma_res, apply_reservoir, cte.g0, cte.rabi, cte.k_z,
            detuning, F_pump, F_probe, cst,
            t_max, t_stationary, t_obs, dt_frame, t_noise,
            Lx, Ly, Nx, Ny)

simu.pump_spatial_profile = fc.tophat(simu.F_pump_r, simu.R, radius = 75)
simu.pump_temporal_profile = fc.to_turning_point(simu.F_pump_t, simu.time, 400, 400)

#Run simulation and save data
folder_DATA = "/home/killian/LEON/DATA/Polaritons/Reservoir/simu"

if apply_reservoir:
    string_name="tophat50_turning_point"
else:
    string_name="ref_tophat50_turning_point"

folder_DATA += "/data_set_" + string_name
try:
    os.mkdir(folder_DATA)
except:
    print("folder already created")

try:
    os.mkdir(folder_DATA + "/raw_arrays")
except:
    print("folder already created")
    
parameters = [('h_bar',cte.h_bar), ('h_bar_SI', cte.h_bar_SI), ('c', cte.c), ('eV_to_J', cte.eV_to_J), ('n_cav', cte.n_cav), 
              ('omega_exc (div by hbar)', cte.omega_exc*cte.h_bar), ('omega_cav (div by hbar)', cte.omega_cav*cte.h_bar), ('gamma_exc (div by hbar)', cte.gamma_exc*cte.h_bar), ('gamma_cav (div by hbar)', cte.gamma_cav*cte.h_bar), 
              ('g0 (div by hbar)', cte.g0*cte.h_bar), ('rabi (div by 2hbar)', cte.rabi*2*cte.h_bar), ('k_z', cte.k_z), ('detuning (div by hbar)', detuning*cte.h_bar), 
              ('F_pump', F_pump), ('F_probe', F_probe), ('t_min', t_min), ('t_max', t_max), ('t_stationary', t_stationary), ('t_obs', t_obs), ('dt_frame', dt_frame), ('t_noise', t_noise), ('t_probe', t_probe), 
              ('Nx', Nx), ('Ny', Ny), ('Lx', Lx), ('Ly', Ly),
              ('omega_probe', omega_probe), ('Pump_spatial_profile', simu.pump_spatial_profile), ('Pump_temporal_profile', simu.pump_temporal_profile), ('Probe_spatial_profile', simu.probe_spatial_profile), ('Probe_temporal_profile', simu.probe_temporal_profile), ("Potential_profile", simu.potential_profile)] 

simu.evolution()
save_raw_data(folder_DATA, parameters)

initial_state = False

af.config_plots()

#%%
folder = folder_DATA
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
print(C02)
print("ouloulou: ", (1/X02**2))
g_LP = g0*X02**2
print(gamma_cav*C02*h_bar)
print(gamma_cav)
print(C02)


cav_field_txy, exc_field_txy, cav_stationary_xy, exc_stationary_xy, hopfield_coefs, F_t = af.load_raw_data(folder, path_ic)
den_reservoir_txy = np.load(folder+"/raw_arrays/mean_den_reservoir_t_x_y.npy")
LP_t_x_y, LP_w_kx_ky = af.polariton_fields(cav_field_txy, exc_field_txy, hopfield_coefs, only_LP = True)
LP_stat_x_y, LP_stat_kx_ky = af.stationary_polariton_fields(cav_stationary_xy, exc_stationary_xy, hopfield_coefs, only_LP = True)

#show hopfield coefficients
# plt.figure("hopfield coefficients")
# plt.imshow(np.abs(hopfield_coefs[1].get()))
# plt.show()

#Phase and density movies:
af.movies(folder, LP_t_x_y)

#Plot bistability cycle:
af.plot_gnLP_vs_I(folder, LP_t_x_y, F_t, R, g_LP, gamma_exc, gamma_cav, X02, C02, h_bar = h_bar, detuning = detuning * h_bar, theoretical = False) #theoretical = False because it does not work yet aha

#Plotting time evolution of the average LP density around the center and laser intensity (create a function for this)
time_plot = np.array([i*dt_frame for i in range(len(cav_field_txy))])
F_intensity = cp.abs(F_t)**2
LP_density = cp.abs(LP_t_x_y)**2
den_reservoir = np.abs(den_reservoir_txy)
avg_density = cp.zeros(len(F_t))
avg_den_res = np.zeros(len(F_t))
radius = 30
disk = cp.zeros((Nx, Ny))
disk[R < radius] += 1
for i in range(len(F_t)):
    avg_density[i] += cp.average(LP_density[i], axis=(-2, -1), weights = disk)
    avg_den_res[i] += np.average(den_reservoir[i], axis=(-2, -1), weights = disk)
fig, ax1 = plt.subplots()
ax3 = ax1.twinx()
ax1.plot(time_plot, g_LP*avg_density.get(), label = "$gn_{LP}(0,0)$", color = "r")
ax1.plot(time_plot, g0*avg_den_res, label = "$g0n_{res}(0,0)$", color = "b")
ax3.plot(time_plot, F_intensity.get(), label = "$I(0,0)$", color = "k")
ax1.plot(time_plot, (g_LP*avg_density.get()+g0*avg_den_res), label = "$gn_{LP}(0,0)+g0n_{res}$", color = "g")
ax1.axhline(y = detuning, label = "detuning", color = "k", linestyle = "--")
#set hlines
ax1.set_xlabel("Time (ps)")
ax1.set_ylabel("$gn$")
ax3.set_ylabel("$I$")
fig.legend()
#plt.show()
fig.savefig(folder+"/laser_LP_densities.png")
plt.close("all")

ratio = g0*avg_den_res/(g_LP*avg_density.get())
plt.figure("ratio")
plt.plot(ratio)
#plt.show()

X = np.linspace(0, 1, 25)
c = 10
def alpha(x):
    return np.sqrt(c*x**4/(1+c*x**4))
plt.figure("alpha")
plt.plot(X**2, alpha(X))
#plt.show()

#Plotting the stationary state
fig, ax = plt.subplots(1, 2, figsize = (10, 5))
im0 = ax[0].imshow(np.abs(LP_stat_x_y.get())**2, extent = [-Lx/2, Lx/2, -Ly/2, Ly/2], origin = "lower")
ax[0].set_title("LP density")
fig.colorbar(im0, ax = ax[0])
im1 = ax[1].imshow(np.angle(LP_stat_x_y.get()), extent = [-Lx/2, Lx/2, -Ly/2, Ly/2], origin = "lower", cmap = "twilight_shifted")
ax[1].set_title("LP phase")
fig.colorbar(im1, ax = ax[1])
fig.savefig(folder+"/stationary_state.png")
plt.close("all")

#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------

#EXAMPLE 3: tophat pump from initial condition, with noise and probe at resonance

# #Laser parameters
# detuning = 0.17/cte.h_bar # (meV/hbar) detuning between the pump and the LP energy
# F_pump = 1.1
# F_probe = 1e-4

# #Grid parameters
# Lx, Ly = 256, 256
# Nx, Ny = 256, 256

# if (Lx/Nx)**2<cte.g0/cte.gamma_cav or (Ly/Ny)**2<cte.g0/cte.gamma_cav:
#     print("WARNING: TWA NOT VALID")

# # Loading initial condition
# path_ic = "/home/stagios/Oscar/LEON/DATA/Polaritons/2024_ManasOscar/tests_for_repo/newpush" #Complete with your directory
# folder_ic = path_ic + "/data_set_stationary_state_at_turning_point_tophat80"
# cav_ic, exc_ic = load_raw_data(folder_ic, only_stationary = True)
# initial_state = cp.zeros((2, Nx, Ny), dtype = cp.complex64)
# initial_state[0, :, :] = cp.asarray(exc_ic)
# initial_state[1, :, :] = cp.asarray(cav_ic)

# # Time parameters
# t_min = 0 # (ps) initial time of evolution
# t_max = 1050 # (ps) final time of evolution
# t_stationary = 1e9
# t_noise = 0 # (ps) time from which the noise starts
# t_probe = 0 # (ps) time from which the probe starts
# t_obs = 51 # (ps) time from which the observation starts
# dt_frame = 1/(0.5) #cst/delta_E avec delta_E la résolution/omega_max en énergie en meV // delta_E fixes the window you will see without aliasing in frequencies, delta_E*2pi/2 = nyquist frequency 
# n_frame = int((t_max-t_obs)/dt_frame)+1
# print("dt_frame is %s"%(dt_frame))
# print("n_frame is %s"%(n_frame))
# omega_probe = 0.5
# k_probe = 0.5

# simu = ggpe(cte.omega_exc, cte.omega_cav, cte.gamma_exc, cte.gamma_cav, cte.g0, cte.rabi, cte.k_z,
#             detuning, F_pump, F_probe, 
#             t_max, t_stationary, t_obs, dt_frame, t_noise,
#             Lx, Ly, Nx, Ny)

# simu.pump_spatial_profile = fc.tophat(simu.F_pump_r, simu.R, radius = 80)
# simu.probe_spatial_profile += fc.plane_wave(simu.F_probe_r, simu.XX, k_probe)
# simu.probe_temporal_profile = fc.tempo_probe(simu.F_probe_t, omega_probe, t_probe, time=simu.time)


# folder_DATA =  "/home" #Complete with your directory
# string_name="_dispersion_and_probe_k05_w05"

# try:
#     os.mkdir(folder_DATA)
# except:
#     print("folder already created")

# folder_DATA += "/data_set" + string_name
# print("/data_set" + string_name)

# try:
#     os.mkdir(folder_DATA)
# except:
#     print("folder already created")
    
# try:
#     os.mkdir(folder_DATA + "/raw_arrays")
# except:
#     print("folder already created")
    
# parameters = [('h_bar',cte.h_bar), ('h_bar_SI', cte.h_bar_SI), ('c', cte.c), ('eV_to_J', cte.eV_to_J), ('n_cav', cte.n_cav), 
#               ('omega_exc (div by hbar)', cte.omega_exc*cte.h_bar), ('omega_cav (div by hbar)', cte.omega_cav*cte.h_bar), ('gamma_exc (div by hbar)', cte.gamma_exc*cte.h_bar), ('gamma_cav (div by hbar)', cte.gamma_cav*cte.h_bar), 
#               ('g0 (div by hbar)', cte.g0*cte.h_bar), ('rabi (div by 2hbar)', cte.rabi*2*cte.h_bar), ('k_z', cte.k_z), ('detuning (div by hbar)', detuning*cte.h_bar), 
#               ('F_pump', F_pump), ('F_probe', F_probe), ('t_min', t_min), ('t_max', t_max), ('t_stationary', t_stationary), ('t_obs', t_obs), ('dt_frame', dt_frame), ('t_noise', t_noise), ('t_probe', t_probe), 
#               ('Nx', Nx), ('Ny', Ny), ('Lx', Lx), ('Ly', Ly),
#               ('omega_probe', omega_probe), ('Pump_spatial_profile', simu.pump_spatial_profile), ('Pump_temporal_profile', simu.pump_temporal_profile), ('Probe_spatial_profile', simu.probe_spatial_profile), ('Probe_temporal_profile', simu.probe_temporal_profile), ("Potential_profile", simu.potential_profile)] 

# parameters.append(("initial_state_path", folder_ic))

# simu.evolution(initial_state = initial_state)
# save_raw_data(folder_DATA, parameters)

#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------

#EXAMPLE 4: simulating 4 different probe configurations in parallel (w=0.5, k=0.5 and w=1.5, k=1 are resonant)

# #Laser parameters
# detuning = 0.17/cte.h_bar # (meV/hbar) detuning between the pump and the LP energy
# F_pump = 1.1
# F_probe = 1e-4

# #Grid parameters
# Lx, Ly = 256, 256
# Nx, Ny = 256, 256

# if (Lx/Nx)**2<cte.g0/cte.gamma_cav or (Ly/Ny)**2<cte.g0/cte.gamma_cav:
#     print("WARNING: TWA NOT VALID")

# # Loading initial condition
# path_ic = "/home/stagios/Oscar/LEON/DATA/Polaritons/2024_ManasOscar/tests_for_repo/newpush" #Complete with your directory
# folder_ic = path_ic + "/data_set_stationary_state_at_turning_point_tophat80"
# cav_ic, exc_ic = load_raw_data(folder_ic, only_stationary = True)
# initial_state = cp.zeros((2, Nx, Ny), dtype = cp.complex64)
# initial_state[0, :, :] = cp.asarray(exc_ic)
# initial_state[1, :, :] = cp.asarray(cav_ic)

# # Time parameters
# t_min = 0 # (ps) initial time of evolution
# t_max = 1050 # (ps) final time of evolution
# t_stationary = 1e9
# t_noise = 0 # (ps) time from which the noise starts
# t_probe = 0 # (ps) time from which the probe starts
# t_obs = 51 # (ps) time from which the observation starts
# dt_frame = 1/(0.5) #cst/delta_E avec delta_E la résolution/omega_max en énergie en meV // delta_E fixes the window you will see without aliasing in frequencies, delta_E*2pi/2 = nyquist frequency 
# n_frame = int((t_max-t_obs)/dt_frame)+1
# print("dt_frame is %s"%(dt_frame))
# print("n_frame is %s"%(n_frame))
# omega_probe = 0.5
# k_probe = 0.5

# simu = ggpe(cte.omega_exc, cte.omega_cav, cte.gamma_exc, cte.gamma_cav, cte.g0, cte.rabi, cte.k_z,
#             detuning, F_pump, F_probe, 
#             t_max, t_stationary, t_obs, dt_frame, t_noise,
#             Lx, Ly, Nx, Ny)

# simu.pump_spatial_profile = fc.tophat(simu.F_pump_r, simu.R, radius = 80)

# Kx_scan = cp.array([0.5, 1])
# omega_scan = cp.array([0.5, 1.5])
# simu.F_probe_r = cp.ones((Kx_scan.shape[0], 1, simu.Nx, simu.Ny), dtype=cp.complex64)
# simu.F_probe_t = cp.ones((1, omega_scan.shape[0], 1, 1, simu.time.shape[0]), dtype=cp.complex64)
# for i in range(Kx_scan.shape[0]):
#     simu.probe_spatial_profile = fc.plane_wave(simu.F_probe_r[i, :, :, :], simu.XX, Kx_scan[i])
# for j in range(omega_scan.shape[0]):
#     simu.probe_temporal_profile = fc.tempo_probe(simu.F_probe_t[:, j, :, :, :], omega_scan[j], t_probe, simu.time)

# folder_DATA =  "/home" #Complete with your directory
# string_name="_2k_2w_in_parallel"

# try:
#     os.mkdir(folder_DATA)
# except:
#     print("folder already created")

# folder_DATA += "/data_set" + string_name
# print("/data_set" + string_name)

# try:
#     os.mkdir(folder_DATA)
# except:
#     print("folder already created")
    
# try:
#     os.mkdir(folder_DATA + "/raw_arrays")
# except:
#     print("folder already created")
    
# parameters = [('h_bar',cte.h_bar), ('h_bar_SI', cte.h_bar_SI), ('c', cte.c), ('eV_to_J', cte.eV_to_J), ('n_cav', cte.n_cav), 
#               ('omega_exc (div by hbar)', cte.omega_exc*cte.h_bar), ('omega_cav (div by hbar)', cte.omega_cav*cte.h_bar), ('gamma_exc (div by hbar)', cte.gamma_exc*cte.h_bar), ('gamma_cav (div by hbar)', cte.gamma_cav*cte.h_bar), 
#               ('g0 (div by hbar)', cte.g0*cte.h_bar), ('rabi (div by 2hbar)', cte.rabi*2*cte.h_bar), ('k_z', cte.k_z), ('detuning (div by hbar)', detuning*cte.h_bar), 
#               ('F_pump', F_pump), ('F_probe', F_probe), ('t_min', t_min), ('t_max', t_max), ('t_stationary', t_stationary), ('t_obs', t_obs), ('dt_frame', dt_frame), ('t_noise', t_noise), ('t_probe', t_probe), 
#               ('Nx', Nx), ('Ny', Ny), ('Lx', Lx), ('Ly', Ly),
#               ('omega_probe', omega_probe), ('Pump_spatial_profile', simu.pump_spatial_profile), ('Pump_temporal_profile', simu.pump_temporal_profile), ('Probe_spatial_profile', simu.probe_spatial_profile), ('Probe_temporal_profile', simu.probe_temporal_profile), ("Potential_profile", simu.potential_profile)] 

# parameters.append(("initial_state_path", folder_ic))

# simu.evolution(initial_state = initial_state)
# save_raw_data(folder_DATA, parameters)