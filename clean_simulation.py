import numpy as np
import cupy as cp
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
import os


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
    cp.save(folder+"/raw_arrays/stationary_cav_x_y", stationary_cav_x_y)
    cp.save(folder+"/raw_arrays/stationary_exc_x_y", stationary_exc_x_y)
    cp.save(folder+"/raw_arrays/hopfield_coefs", hopfield_coefs)
    cp.save(folder+"/raw_arrays/F_t", F_t)
    
def load_raw_data(folder):
    
    #Load raw data as numpy arrays
    mean_cav_t_x_y = np.load(folder+"/raw_arrays/mean_cav_t_x_y.npy")
    mean_exc_t_x_y = np.load(folder+"/raw_arrays/mean_exc_t_x_y.npy")
    stationary_cav_x_y = np.load(folder+"/raw_arrays/stationary_cav_x_y.npy")
    stationary_exc_x_y = np.load(folder+"/raw_arrays/stationary_exc_x_y.npy")
    hopfield_coefs = np.load(folder+"/raw_arrays/hopfield_coefs.npy")
    F_t = np.load(folder+"/raw_arrays/F_t.npy")
    
    return mean_cav_t_x_y, mean_exc_t_x_y, stationary_cav_x_y, stationary_exc_x_y, hopfield_coefs, F_t



#Defining experiment parameters: ----------------------------------------------------------

#Physical constants
h_bar = 0.654 # (meV*ps)
c = 2.9979*1e2 # (um/ps)
eV_to_J = 1.60218*1e-19
h_bar_SI = 1.05457182*1e-34

#Microcavity parameters
rabi = 5.07/2/h_bar # (meV/h_bar) linear coupling (Rabi split)
g0 = (1e-2) /h_bar  # (frequency/density) (meV/hbar)/(1/um^2) nonlinear coupling constant 
#g0 = 0.003/h_bar
gamma_exc, gamma_ph = 0.07 /h_bar, 0.07 /h_bar # (meV/h_bar) exc and ph linewidth 1microeV 69microeV  original value 0.07/h_bar
omega_exc = 1484.44 /h_bar # (meV/h_bar) exciton energy measured from the cavity energy #-0.5
omega_cav = 1482.76 /h_bar # (meV/h_bar) cavity energy at k=0  original value: 1482.76 /h_bar
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
t_obs = 1751 # (ps) time from which the observation starts
t_noise = 0 # (ps) time from which the noise starts
t_probe = 1e9
t_stationary = 1749
t_max = 2750 # (ps) final time of evolution
dt_frame = 1/(0.5) #cst/delta_E avec delta_E la résolution/omega_max en énergie en meV
n_frame = int((t_max-t_obs)/dt_frame)+1
print("dt_frame is %s"%(dt_frame))
print("n_frame is %s"%(n_frame))

# Grid parameters
nmax = 256**1
nmax_1, nmax_2 = nmax, nmax
long_1, long_2 = 256, 256

# Pump and probe parameters
F_pump = 2
F_probe = 0
corr = 0.0 #0.35
detuning = 0.17/h_bar
omega_probe=0
pump_radius = 80
pump_profile = " "
probe_profile = " "

if (long_1/nmax)**2<g0/gamma_ph:
    print("WARNING: TWA NOT VALID")

#----------------------------------------------------------------------------------------

#Load class with the simulation parameters
simu = ggpe(nmax_1, nmax_2, long_1, long_2, t_max, t_stationary, t_obs, t_probe, t_noise, dt_frame, gamma_exc, 
        gamma_ph, g0, detuning, omega_probe, omega_exc, omega_cav, rabi, k_z, F_pump, F_probe)


m_LP = simu.m_LP
R = simu.R
THETA = simu.THETA
X = simu.X
Y = simu.Y

omega_LP0 = simu.LB[simu.nmax_1//2]/h_bar
delta_LP = omega_cav - omega_LP0 

#Build pump and probe fields: ----------------------------------------------------------

#Pump's spacial profile

pump_profile = tophat(simu.F_laser_r, R, radius = pump_radius, pump_profile = pump_profile) #I don't like this way of doing the string thing but I can't find a way 
                                                                                                          #to update the string whithin the function....
#pump_profile = gaussian(simu.F_laser_r, R, radius = pump_radius, pump_profile = pump_profile)

#pump_profile = vortex_beam(simu.F_laser_r, R, THETA, waist = 75, inner_waist = 22, C = 15, pump_profile = pump_profile)

#pump_profile = shear_layer(simu.F_laser_r, X, Y, kx = 1, pump_profile = pump_profile)

#pump_profile = plane_wave(simu.F_laser_r, X=X, kx = 1, pump_profile = pump_profile)


#Pump's temporal profile

pump_profile = to_turning_point(simu.F_laser_t, simu.time, t_up = 400, t_down = 400, pump_profile = pump_profile)

#pump_profile = bistab_cycle(simu.F_laser_t, simu.time, simu.t_max, pump_profile = pump_profile)

#pump_profile = turn_on_pump(simu.F_laser_t, simu.time, t_up = 200, pump_profile = pump_profile)


#Probe's spacial profile

#probe_profile = plane_wave(simu.F_probe_r, X=X, kx = 1, pump_profile = probe_profile)

probe_profile = ring(simu.F_probe_r, R, radius = pump_radius, delta_radius = 20, probe_profile = probe_profile)

#probe_profile = radial_expo(F_probe_r, R, THETA, m_probe = 10, p_probe = 5, probe_profile = probe_profile)


#Probe's temporal profile

tempo_probe(simu.F_probe_t, omega_probe, t_probe, simu.time)

#----------------------------------------------------------------------------------------

#Opening directories for saving data and simulation parameters: -------------------------

parameters=[('h_bar',h_bar), ('h_bar_SI', h_bar_SI), ('c', c), ('eV_to_J', eV_to_J), ('rabi (div by 2hbar)', rabi*2*h_bar), ('g0 (div by hbar)', g0*h_bar), ('gamma_exc (div by hbar)', gamma_exc*h_bar), 
            ('gamma_ph (div by hbar)', gamma_ph*h_bar), ('gamma_LP (div by hbar)', gamma_LP*h_bar), ('omega_exc (div by hbar)', omega_exc*h_bar), ('omega_cav (div by hbar)', omega_cav*h_bar), ('n_cav', n_cav), ('k_z', k_z), ('t_min', t_min), ('t_obs', t_obs), 
            ('t_noise', t_noise), ('t_probe', t_probe), ('t_stationary', t_stationary), ('t_max', t_max), ('dt_frame', dt_frame), ('nmax_1', nmax_1), ('nmax_2', nmax_2), ('long_1', long_1), ('long_2', long_2), ('F_pump', F_pump), ('F_probe', F_probe),
            ('corr', corr), ('detuning (div by hbar)', detuning*h_bar), ('omega_probe', omega_probe), ('Pump_profile', pump_profile), ('Probe_profile', probe_profile)] 

folder_DATA =  "/home/stagios/Oscar/LEON/DATA/Polaritons/2024_ManasOscar/dispersion"
string_name="_tophat80_beg_noisy_Fpump2"

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
    
#----------------------------------------------------------------------------------------

#Run the simulation and save the raw data: ----------------------------------------------
    
simu.evolution()

save_raw_data(folder_DATA, parameters)

#----------------------------------------------------------------------------------------