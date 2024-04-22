import numpy as np
import cupy as cp
from ggpe2d import ggpe
import field_creation_functions as fc
import physical_constants as cte
import os
import matplotlib.pyplot as plt
from analysis_functions import load_raw_data
import time



cp.cuda.Device(0).use()

def save_raw_data(folder,parameters,t_save):
    
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
    mean_cav_t_save = simu.mean_cav_t_save
    mean_exc_t_save = simu.mean_exc_t_save
    hopfield_coefs = simu.hopfield_coefs
    F_t = simu.F_t

    #Save data as numpy arrays
    if mean_cav_t_x_y is not None:
        cp.save(folder+"/raw_arrays/mean_cav_t_x_y", mean_cav_t_x_y)
    if mean_exc_t_x_y is not None:
        cp.save(folder+"/raw_arrays/mean_exc_t_x_y", mean_exc_t_x_y)
    if stationary_cav_x_y is not None:
        cp.save(folder+"/raw_arrays/stationary_cav_x_y", stationary_cav_x_y)
    if stationary_exc_x_y is not None:
        cp.save(folder+"/raw_arrays/stationary_exc_x_y", stationary_exc_x_y)
    if mean_cav_t_save is not None:
        cp.save(folder+"/raw_arrays/cav_x_y_t"+str(t_save), mean_cav_t_save)
    if mean_exc_t_save is not None:
        cp.save(folder+"/raw_arrays/exc_x_y_t"+str(t_save), mean_exc_t_save)
    cp.save(folder+"/raw_arrays/hopfield_coefs", hopfield_coefs)
    cp.save(folder+"/raw_arrays/F_t", F_t)
    

# Time parameters
t_min = 0 # (ps) initial time of evolution
t_max = 1810 # (ps) final time of evolution
t_stationary = 1800
t_noise = 1700 # (ps) time from which the noise starts
t_probe = 1e9 # (ps) time from which the probe starts
t_obs = 1800 # (ps) time from which the observation starts
dt_frame = 0.0076236447109110025 #1/(2) #cst/delta_E avec delta_E la résolution/omega_max en énergie en meV // delta_E fixes the window you will see without aliasing in frequencies, delta_E*2pi/2 = nyquist frequency 
n_frame = int((t_max-t_obs)/dt_frame)+1
print("dt_frame is %s"%(dt_frame))
print("n_frame is %s"%(n_frame))

#Laser parameters
detuning = 0.17/cte.h_bar # (meV/hbar) detuning between the pump and the LP energy
F_pump = 1.1
F_probe = 0

#Grid parameters
Lx, Ly = 256, 256
Nx, Ny = 256, 256

if (Lx/Nx)**2<cte.g0/cte.gamma_cav or (Ly/Ny)**2<cte.g0/cte.gamma_cav:
    print("WARNING: TWA NOT VALID")

#Load class with the simulation parameters
simu = ggpe(cte.omega_exc, cte.omega_cav, cte.gamma_exc, cte.gamma_cav, cte.g0, cte.rabi, cte.k_z,
            detuning, F_pump, F_probe, 
            t_max, t_stationary, t_obs, dt_frame, t_noise,
            Lx, Ly, Nx, Ny)

#Build pump, probe and potential fields: ----------------------------------------------------------

k_pump = 0.5
# # Intensity profile of pump field, default is constant all over grid.
simu.pump_spatial_profile = fc.tophat(simu.F_pump_r, simu.R, radius = 80)

# # Temporal modulation of the pump
simu.pump_temporal_profile = fc.to_turning_point(simu.F_pump_t, simu.time, t_up = 400, t_down = 400)


#Opening directories for saving data and simulation parameters: -------------------------

folder_DATA =  "/home/stagios/Oscar/LEON/DATA/Polaritons/2024_ManasOscar/convergence_tests/aliasing_tests"
string_name="_noisy_tophat80_256x256_dtframe=dt"

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
    
    
omega_probe = 0
#Run the simulation and save the raw data: ----------------------------------------------
parameters = [('h_bar',cte.h_bar), ('h_bar_SI', cte.h_bar_SI), ('c', cte.c), ('eV_to_J', cte.eV_to_J), ('n_cav', cte.n_cav), 
              ('omega_exc (div by hbar)', cte.omega_exc*cte.h_bar), ('omega_cav (div by hbar)', cte.omega_cav*cte.h_bar), ('gamma_exc (div by hbar)', cte.gamma_exc*cte.h_bar), ('gamma_cav (div by hbar)', cte.gamma_cav*cte.h_bar), 
              ('g0 (div by hbar)', cte.g0*cte.h_bar), ('rabi (div by 2hbar)', cte.rabi*2*cte.h_bar), ('k_z', cte.k_z), ('detuning (div by hbar)', detuning*cte.h_bar), 
              ('F_pump', F_pump), ('F_probe', F_probe), ('t_min', t_min), ('t_max', t_max), ('t_stationary', t_stationary), ('t_obs', t_obs), ('dt_frame', dt_frame), ('t_noise', t_noise), ('t_probe', t_probe), 
              ('Nx', Nx), ('Ny', Ny), ('Lx', Lx), ('Ly', Ly),
              ('omega_probe', omega_probe), ('Pump_spatial_profile', simu.pump_spatial_profile), ('Pump_temporal_profile', simu.pump_temporal_profile), ('Probe_spatial_profile', simu.probe_spatial_profile), ('Probe_temporal_profile', simu.probe_temporal_profile), ("Potential_profile", simu.potential_profile)] 
t_save = 1800
parameters.append(("t_save", t_save))

simu.evolution(save_fields_at_time = t_save, save = True)
save_raw_data(folder_DATA, parameters, t_save)

# print(simu.dt)