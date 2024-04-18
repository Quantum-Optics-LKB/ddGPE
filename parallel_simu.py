import numpy as np
import cupy as cp
from ggpe2d import ggpe
import field_creation_functions as fc
import physical_constants as cte
import os
from analysis_functions import load_raw_data
import matplotlib.pyplot as plt

cp.cuda.Device(1).use()

def save_raw_data(folder,parameters, t_save):

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
t_max = 100 # (ps) final time of evolution
t_stationary = 1e9
t_noise = 1e9 # (ps) time from which the noise starts
t_probe = 0 # (ps) time from which the probe starts
t_obs = 0 # (ps) time from which the observation starts
dt_frame = 1/(0.5) #cst/delta_E avec delta_E la résolution/omega_max en énergie en meV // delta_E fixes the window you will see without aliasing in frequencies, delta_E*2pi/2 = nyquist frequency
n_frame = int((t_max-t_obs)/dt_frame)+1
print("dt_frame is %s"%(dt_frame))
print("n_frame is %s"%(n_frame))

#Laser parameters
detuning = 0.17/cte.h_bar # (meV/hbar) detuning between the pump and the LP energy
F_pump = 1.1
F_probe = 1e-4

#Grid parameters
Lx, Ly = 256, 256
Nx, Ny = 256, 256

if (Lx/Nx)**2<cte.g0/cte.gamma_cav or (Ly/Ny)**2<cte.g0/cte.gamma_cav:
    print("WARNING: TWA NOT VALID")

#Import initial condition
path_ic = "/home/stagios/Oscar/LEON/DATA/Polaritons/2024_ManasOscar/initial_conditions"
folder_ic = path_ic + "/data_set_tophat80"
cav_ic, exc_ic = load_raw_data(folder_ic, only_stationary = True)
initial_state = cp.zeros((2, Nx, Ny), dtype = cp.complex64)
initial_state[0,:,:] = cp.asarray(exc_ic)
initial_state[1,:,:] = cp.asarray(cav_ic)

#Load class with the simulation parameters
simu = ggpe(cte.omega_exc, cte.omega_cav, cte.gamma_exc, cte.gamma_cav, cte.g0, cte.rabi, cte.k_z,
            detuning, F_pump, F_probe,
            t_max, t_stationary, t_obs, dt_frame, t_noise,
            Lx, Ly, Nx, Ny)


#Build pump, probe and potential fields: ----------------------------------------------------------

k_pump = 0.5
# # Intensity profile of pump field, default is constant all over grid.
simu.pump_spatial_profile = fc.tophat(simu.F_pump_r, simu.R, radius = 80)
# simu.pump_spatial_profile = fc.gaussian(simu.F_pump_r, simu.R, radius = 80)
# simu.pump_spatial_profile = fc.ring(simu.F_pump_r, simu.R, radius = 60, delta_radius = 10)
# simu.pump_spatial_profile = fc.vortex_beam(simu.F_pump_r, simu.R, simu.THETA, waist = 75, inner_waist = 22, C = 15)
# simu.pump_spatial_profile = fc.effective_1d(simu.F_pump_r, simu.XX, simu.YY, half_width = 128, beg_pump = -128, end_pump = 0, end_support = 128, ratio = 0.5, k_pump = k_pump, along = "y")
# # Adding a phase pattern to the pump field.
# simu.pump_spatial_profile += fc.plane_wave(simu.F_pump_r, simu.XX, kx = k_pump)
# simu.pump_spatial_profile += fc.shear_layer(simu.F_pump_r, simu.XX, simu.YY, kx = 1)
# simu.pump_spatial_profile += fc.radial_expo(simu.F_pump_r, simu.R, simu.THETA, m = 3, p = 1)
# # Temporal modulation of the pump
# simu.pump_temporal_profile = fc.to_turning_point(simu.F_pump_t, simu.time, t_up = 400, t_down = 400)
# simu.pump_temporal_profile = fc.bistab_cycle(simu.F_pump_t, simu.time, simu.t_max)
# simu.pump_temporal_profile = fc.turn_on(simu.F_pump_t, simu.time, t_up = 400)


omega_probe = 0.5
k_probe = 0.5

# simu.F_probe_t = cp.ones((1,2, 1, 1, simu.time.shape[0]), dtype=np.complex64)
# simu.probe_temporal_profile = fc.tempo_probe(simu.F_probe_t[0,0,0,0,:], 0.5, t_probe, time=simu.time)
# simu.probe_temporal_profile = fc.tempo_probe(simu.F_probe_t[0,1,0,0,:], 1, t_probe, time=simu.time)
# simu.F_probe_r = cp.ones((2,1, Nx, Ny), dtype=np.complex64)
# simu.probe_spatial_profile += fc.plane_wave(simu.F_probe_r[0], simu.XX, 0.5)
# simu.probe_spatial_profile += fc.plane_wave(simu.F_probe_r[1], simu.XX, 0.5)

# # Intensity profile of probe field, default is constant all over grid.
# simu.probe_spatial_profile = fc.tophat(simu.F_probe_r, simu.R, radius = 80)
# simu.probe_spatial_profile = fc.gaussian(simu.F_probe_r, simu.R, radius = 80)
# simu.probe_spatial_profile = fc.ring(simu.F_probe_r, simu.R, radius = 60, delta_radius = 10)
# simu.probe_spatial_profile = fc.vortex_beam(simu.F_probe_r, simu.R, simu.THETA, waist = 75, inner_waist = 22, C = 15)
# simu.probe_spatial_profile = fc.effective_1d(simu.F_probe_r, simu.XX, simu.YY, half_width = 128, beg_pump = -128, end_pump = 0, end_support = 128, ratio = 0.5, k_pump = 0.5, along = "y")
# # Adding a phase pattern to the probe field.
# simu.probe_spatial_profile += fc.plane_wave(simu.F_probe_r, simu.XX, k_probe)
# simu.probe_spatial_profile += fc.shear_layer(simu.F_probe_r, simu.XX, simu.YY, kx = 1)
# simu.probe_spatial_profile += fc.radial_expo(simu.F_probe_r, simu.R, simu.THETA, waist = 75, m = 3, p = 0)
# # Temporal modulation of the pump
# simu.probe_temporal_profile = fc.tempo_probe(simu.F_probe_t, omega_probe, t_probe, time=simu.time)
# print("Shape of F_probe_t: ", simu.F_probe_t.shape)
# print("Shape of F_probe_r: ", simu.F_probe_r.shape)

# # Potential profile, default is 0 all over grid.
# simu.potential_profile = fc.gaussian_barrier(simu.potential, simu.XX, simu.YY, h = 0.3905, sigma_y = 15, sigma_x = 1000, off_y = 0)
# simu.potential_profile = fc.tophat_barrier(simu.potential, simu.XX, simu.YY, h = 0.3905, sigma_y = 15, sigma_x = 1000, off_y=0)


# folder_DATA =  "/home/stagios/Oscar/LEON/DATA/Polaritons/2024_ManasOscar/ggpe2d_v2_tests/parallel_evolution"

k_min = 0.5
k_max = 0.5
k_step = 0.1
# Kx_scan = cp.arange(k_min, k_max, k_step)
omega_min = -1
omega_max = 1
omega_step = 0.004
omega_scan = cp.arange(omega_min, omega_max, omega_step)

Kx_scan = cp.array([0.5])
# omega_scan = cp.array([0.5])
simu.F_probe_r = cp.ones((Kx_scan.shape[0], 1, simu.Nx, simu.Ny), dtype=np.complex64)
print(simu.F_probe_r.shape)
simu.F_probe_t = cp.ones((1, omega_scan.shape[0], 1, 1, simu.time.shape[0]), dtype=np.complex64)
print(simu.F_probe_t.shape)
for i in range(Kx_scan.shape[0]):
    simu.probe_spatial_profile = fc.plane_wave(simu.F_probe_r[i, :, :, :], simu.XX, Kx_scan[i])
    simu.probe_spatial_profile = fc.tophat(simu.F_probe_r[i, :, :, :], simu.R, radius = 80)
for j in range(omega_scan.shape[0]):
    simu.probe_temporal_profile = fc.tempo_probe(simu.F_probe_t[:, j, :, :, :], omega_scan[j], t_probe, simu.time)

simu.probe_spatial_profile = "Tophat: radius = 80; Parallelized plane wave, Kx_scan: [start: " + str(Kx_scan[0]) + ", end: " + str(Kx_scan[-1]) + "] ; "
simu.probe_temporal_profile = "Parallelized plane wave, omega_scan [start: " + str(omega_scan[0]) + ", end: " + str(omega_scan[-1]) + "]; "

#Opening directories for saving data and simulation parameters: -------------------------


folder_DATA =  "/home/stagios/Oscar/LEON/DATA/Polaritons/2024_ManasOscar/probe_scan"
string_name="_k05_w-1_1_1x500_4WM"

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

#Run the simulation and save the raw data: ----------------------------------------------
parameters = [('h_bar',cte.h_bar), ('h_bar_SI', cte.h_bar_SI), ('c', cte.c), ('eV_to_J', cte.eV_to_J), ('n_cav', cte.n_cav),
              ('omega_exc (div by hbar)', cte.omega_exc*cte.h_bar), ('omega_cav (div by hbar)', cte.omega_cav*cte.h_bar), ('gamma_exc (div by hbar)', cte.gamma_exc*cte.h_bar), ('gamma_cav (div by hbar)', cte.gamma_cav*cte.h_bar),
              ('g0 (div by hbar)', cte.g0*cte.h_bar), ('rabi (div by 2hbar)', cte.rabi*2*cte.h_bar), ('k_z', cte.k_z), ('detuning (div by hbar)', detuning*cte.h_bar),
              ('F_pump', F_pump), ('F_probe', F_probe), ('t_min', t_min), ('t_max', t_max), ('t_stationary', t_stationary), ('t_obs', t_obs), ('dt_frame', dt_frame), ('t_noise', t_noise), ('t_probe', t_probe), 
              ('Nx', Nx), ('Ny', Ny), ('Lx', Lx), ('Ly', Ly),
              ('omega_probe', omega_probe), ('Pump_spatial_profile', simu.pump_spatial_profile), ('Pump_temporal_profile', simu.pump_temporal_profile), ('Probe_spatial_profile', simu.probe_spatial_profile), ('Probe_temporal_profile', simu.probe_temporal_profile), ("Potential_profile", simu.potential_profile)]

parameters.append(("initial_state_path", folder_ic))
t_save = 85
parameters.append(("t_save", t_save))
parameters.append(("k_min", k_min))
parameters.append(("k_max", k_max))
parameters.append(("k_step", k_step))
parameters.append(("omega_min", omega_min))
parameters.append(("omega_max", omega_max))
parameters.append(("omega_step", omega_step))

simu.evolution(initial_state = initial_state, save_fields_at_time = t_save, save = False)
# simu.evolution()
save_raw_data(folder_DATA, parameters, t_save)