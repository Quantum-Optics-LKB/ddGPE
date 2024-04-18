import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import analysis_functions as af
import configparser
from tqdm import tqdm
import cv2

    
cp.cuda.Device(1).use()


# Load data and plotting parameters
directory = "/home/stagios/Oscar/LEON/DATA/Polaritons/2024_ManasOscar/probe_scan"
folder = directory + "/data_set_k05_w-1_1_1x500_4WM"

initial_state = True
saved_frame = True
parallel_plane_wave = True


# directory_bis = directory
# folder_bis = directory_bis + "/data_set_tophat80_k2_w1_Fprobe001"

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
if parallel_plane_wave:
    k_min = float(config.get("parameters", "k_min"))
    k_max = float(config.get("parameters", "k_max"))
    k_step = float(config.get("parameters", "k_step"))
    omega_min = float(config.get("parameters", "omega_min"))
    omega_max = float(config.get("parameters", "omega_max"))
    omega_step = float(config.get("parameters", "omega_step"))   


#%%



af.config_plots()


cav_tsave_xy = cp.asarray(np.load(folder + "/raw_arrays/cav_x_y_t%s.npy" %(str(int(t_save)))))
exc_tsave_xy = cp.asarray(np.load(folder + "/raw_arrays/exc_x_y_t%s.npy" %(str(int(t_save)))))
cav_stationary_xy = cp.asarray(np.load(path_ic + "/raw_arrays/stationary_cav_x_y.npy"))
exc_stationary_xy = cp.asarray(np.load(path_ic + "/raw_arrays/stationary_exc_x_y.npy"))
hopfield_coefs = cp.asarray(np.load(folder + "/raw_arrays/hopfield_coefs.npy"))

LP_tsave_kx_ky = cp.zeros(cav_tsave_xy.shape, dtype=cp.complex64)
Xk = hopfield_coefs[0, :, :]
Ck = hopfield_coefs[1, :, :]
cav_tsave_kx_ky = cp.fft.fftn(cav_tsave_xy, axes = (-2, -1))
exc_tsave_kx_ky = cp.fft.fftn(exc_tsave_xy, axes = (-2, -1))
LP_tsave_kx_ky[..., :, :] = -1 * Xk[:, :] * exc_tsave_kx_ky[..., :, :] + Ck[:, :] * cav_tsave_kx_ky[..., :, :] 
LP_tsave_x_y = cp.fft.ifftn(LP_tsave_kx_ky, axes = (-2, -1))
LP_tsave_kx_ky[..., :, :] = cp.fft.fftshift(LP_tsave_kx_ky[..., :, :], axes = (-2, -1))

LP_stat_kx_ky = cp.zeros(cav_stationary_xy.shape, dtype=cp.complex64)
Xk = hopfield_coefs[0, :, :]
Ck = hopfield_coefs[1, :, :]
cav_stationary_kx_ky = cp.fft.fftn(cav_stationary_xy, axes = (-2, -1))
exc_stationary_kx_ky = cp.fft.fftn(exc_stationary_xy, axes = (-2, -1))
LP_stat_kx_ky[..., :, :] = -1 * Xk[:, :] * exc_stationary_kx_ky[..., :, :] + Ck[:, :] * cav_stationary_kx_ky[..., :, :] #you changed the minus, careful with convention
LP_stat_x_y = cp.fft.ifftn(LP_stat_kx_ky, axes = (-2, -1))
LP_stat_kx_ky[..., :, :] = cp.fft.fftshift(LP_stat_kx_ky[..., :, :], axes = (-2, -1))
        
        

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


                                              
side1, side2 = Nx, Ny #even number                                              #full window, for movies mainly
window = (Nx//2-side1/2, Nx//2+side1/2, Ny//2-side2/2, Ny//2+side2/2)
    
fluctuations_LP = LP_tsave_kx_ky - LP_stat_kx_ky

Kx_scan = cp.arange(k_min, k_max, k_step)
omega_scan = cp.arange(omega_min, omega_max, omega_step)

af.scan_output(folder, fluctuations_LP, Kx_scan, omega_scan, side_square_filter = 0.1, Kxx = Kxx, Kyy = Kyy)
af.scan_output_4WM(folder, fluctuations_LP, Kx_scan, omega_scan, side_square_filter = 0.1, Kxx = Kxx, Kyy = Kyy)


        