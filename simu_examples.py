import cupy as cp
from ggpe2d import ggpe
import field_creation_functions as fc
import physical_constants as cte
import os
from analysis_functions import load_raw_data

cp.cuda.Device(1).use()

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
    stationary_cav_x_y = simu.mean_cav_x_y_stat
    stationary_exc_x_y = simu.mean_exc_x_y_stat
    hopfield_coefs = simu.hopfield_coefs
    F_t = simu.F_t

    #Save data as numpy arrays
    cp.save(folder+"/raw_arrays/mean_cav_t_x_y", mean_cav_t_x_y)
    cp.save(folder+"/raw_arrays/mean_exc_t_x_y", mean_exc_t_x_y)
    if stationary_cav_x_y is not None:
        cp.save(folder+"/raw_arrays/stationary_cav_x_y", stationary_cav_x_y)
    if stationary_exc_x_y is not None:
        cp.save(folder+"/raw_arrays/stationary_exc_x_y", stationary_exc_x_y)
    cp.save(folder+"/raw_arrays/hopfield_coefs", hopfield_coefs)
    cp.save(folder+"/raw_arrays/F_t", F_t)


#------------------------------------------------------------------------------------------
#EXAMPLE 1: bistability cycle of a tophat pump:

# Laser parameters
detuning = 0.17/cte.h_bar # (meV/hbar) detuning between the pump and the LP energy
F_pump = 1.1
F_probe = 0

# Grid parameters
Lx, Ly = 256, 256
Nx, Ny = 256, 256

if (Lx/Nx)**2<cte.g0/cte.gamma_cav or (Ly/Ny)**2<cte.g0/cte.gamma_cav:
    print("WARNING: TWA NOT VALID")
    
# Time parameters
t_min = 0 # (ps) initial time of evolution
t_max = 2000 # (ps) final time of evolution
t_stationary = 1e9
t_noise = 1e9 # (ps) time from which the noise starts
t_probe = 1e9 # (ps) time from which the probe starts
t_obs = 0 # (ps) time from which the observation starts
dt_frame = 1/(0.1) #cst/delta_E avec delta_E la résolution/omega_max en énergie en meV // delta_E fixes the window you will see without aliasing in frequencies, delta_E*2pi/2 = nyquist frequency 
n_frame = int((t_max-t_obs)/dt_frame)+1
print("dt_frame is %s"%(dt_frame))
print("n_frame is %s"%(n_frame))
omega_probe = 0
k_probe = 0

simu = ggpe(cte.omega_exc, cte.omega_cav, cte.gamma_exc, cte.gamma_cav, cte.g0, cte.rabi, cte.k_z,
            detuning, F_pump, F_probe, 
            t_max, t_stationary, t_obs, dt_frame, t_noise,
            Lx, Ly, Nx, Ny)

simu.pump_spatial_profile = fc.tophat(simu.F_pump_r, simu.R, radius = 80)
simu.pump_temporal_profile = fc.bistab_cycle(simu.F_pump_t, simu.time, simu.t_max)

#Run simulation and save data
folder_DATA =  "/home/stagios/Oscar/LEON/DATA/Polaritons/2024_ManasOscar/tests_for_repo/new_dt"  #Complete with your directory
string_name="_bistab_cycle_tophat80_fixed_omegamax"

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

#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
#EXAMPLE 2: stationary state of a tophat pump and saving initial condition for later use:

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
# t_max = 2000 # (ps) final time of evolution
# t_stationary = 1750
# t_noise = 1e9 # (ps) time from which the noise starts
# t_probe = 1e9 # (ps) time from which the probe starts
# t_obs = 0 # (ps) time from which the observation starts
# dt_frame = 1/(0.1) #cst/delta_E avec delta_E la résolution/omega_max en énergie en meV // delta_E fixes the window you will see without aliasing in frequencies, delta_E*2pi/2 = nyquist frequency 
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
# simu.pump_temporal_profile = fc.to_turning_point(simu.F_pump_t, simu.time, t_up = 400, t_down = 400)

# #Run simulation and save data
# folder_DATA =  "/home/stagios/Oscar/LEON/DATA/Polaritons/2024_ManasOscar/tests_for_repo/new_dt" #Complete with your directory
# string_name="_stationary_state_at_turning_point_tophat80"

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
# path_ic = "/home/stagios/Oscar/LEON/DATA/Polaritons/2024_ManasOscar/tests_for_repo/new_dt" #Complete with your directory
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


# folder_DATA =  "/home/stagios/Oscar/LEON/DATA/Polaritons/2024_ManasOscar/tests_for_repo/new_dt" #Complete with your directory
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
# path_ic = "/home/stagios/Oscar/LEON/DATA/Polaritons/2024_ManasOscar/tests_for_repo/new_dt" #Complete with your directory
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

# folder_DATA =  "/home/stagios/Oscar/LEON/DATA/Polaritons/2024_ManasOscar/tests_for_repo/new_dt" #Complete with your directory
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