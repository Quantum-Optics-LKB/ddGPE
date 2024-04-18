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

    
cp.cuda.Device(1).use()

# Load data and plotting parameters
directory = "/home/stagios/Oscar/LEON/DATA/Polaritons/2024_ManasOscar/resonance_setting_time/non_resonant_k1_w05"
folder = directory + "/data_set_tmax_200_dtframe_0.4_tprobe0_tophatprobe"

initial_state = True


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


cav_field_txy, exc_field_txy, cav_stationary_xy, exc_stationary_xy, hopfield_coefs, F_t = load_raw_data(folder, path_ic)
cav_field_txy, exc_field_txy, cav_stationary_xy, exc_stationary_xy, hopfield_coefs, F_t = cp.asarray(cav_field_txy), cp.asarray(exc_field_txy), cp.asarray(cav_stationary_xy), cp.asarray(exc_stationary_xy), cp.asarray(hopfield_coefs), cp.asarray(F_t)

# cav_field_txy_bis, exc_field_txy_bis, cav_stationary_xy_bis, exc_stationary_xy_bis, hopfield_coefs_bis, F_t_bis = load_raw_data(folder_bis)
# cav_field_txy_bis, exc_field_txy_bis, cav_stationary_xy_bis, exc_stationary_xy_bis, hopfield_coefs_bis, F_t_bis = cp.asarray(cav_field_txy_bis), cp.asarray(exc_field_txy_bis), cp.asarray(cav_stationary_xy_bis), cp.asarray(exc_stationary_xy_bis), cp.asarray(hopfield_coefs_bis), cp.asarray(F_t_bis)


config_plots()

#--------------------------------------------------------------------------------



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
#g_LP = 0.5*(X02**2*0.003+X02*np.sqrt(X02)*np.sqrt(C02)*0.0001)
#print(g_LP) #0.001796071745929218


#---------------------------------------------------------------------------

# Now create functions analyzing data, plotting observables, etc... when it works put them in another file and import them from there (maybe even from class file??)

#side1, side2 = 80, 80 #even number                                                      # window to select a homogeneous zone, for the dispersions
side1, side2 = Nx, Ny #even number                                              #full window, for movies mainly
window = (Nx//2-side1/2, Nx//2+side1/2, Ny//2-side2/2, Ny//2+side2/2)
                                                   

LP_t_x_y, UP_t_x_y, LP_w_kx_ky, UP_w_kx_ky = polariton_fields(cav_field_txy, exc_field_txy, hopfield_coefs, window = window, dx = delta_X, dy = delta_Y, omega_exc = omega_exc, omega_cav = omega_cav, rabi = rabi, detuning = detuning, k_z = k_z)
LP_stat_x_y, UP_stat_x_y, LP_stat_kx_ky, UP_stat_kx_ky = stationary_polariton_fields(cav_stationary_xy, exc_stationary_xy, hopfield_coefs, window = window, dx = delta_X, dy = delta_Y, omega_exc = omega_exc, omega_cav = omega_cav, rabi = rabi, detuning = detuning, k_z = k_z)

# LP_t_x_y_bis, UP_t_x_y_bis, LP_w_kx_ky_bis, UP_w_kx_ky_bis = polariton_fields(cav_field_txy_bis, exc_field_txy_bis, hopfield_coefs_bis, window = window)
# LP_stat_x_y_bis, UP_stat_x_y_bis, LP_stat_kx_ky_bis, UP_stat_kx_ky_bis = stationary_polariton_fields(cav_stationary_xy_bis, exc_stationary_xy_bis, hopfield_coefs_bis, window = window)


fluctuations_LP = cp.zeros(LP_w_kx_ky.shape, dtype = cp.complex64)
LP_stat_kx_ky = cp.expand_dims(LP_stat_kx_ky, axis = -3)  
fluctuations_LP = LP_w_kx_ky - LP_stat_kx_ky

LP_t_kx_ky = cp.fft.fftshift(cp.fft.fftn(LP_t_x_y, axes = (-2,-1)), axes = (-2,-1))
fluctuations_LP_t_kx_ky = cp.zeros(LP_t_kx_ky.shape, dtype = cp.complex64)
fluctuations_LP_t_kx_ky = LP_t_kx_ky - LP_stat_kx_ky
fluctuations_LP_t_kx_ky[:,fluctuations_LP_t_kx_ky.shape[-2]//2,fluctuations_LP_t_kx_ky.shape[-1]//2] = 0

# fluctuations_UP = cp.zeros(UP_w_kx_ky.shape, dtype = cp.complex64)
# fluctuations_UP[:] = UP_w_kx_ky[:] - UP_stat_kx_ky
# fluctuations_LP_t_x_y = LP_t_x_y - LP_stat_x_y

# fluctuations_LP_bis = cp.zeros(LP_w_kx_ky_bis.shape, dtype = cp.complex64)
# fluctuations_LP_bis[:] = LP_w_kx_ky_bis[:] - LP_stat_kx_ky_bis
# fluctuations_UP = cp.zeros(UP_w_kx_ky.shape, dtype = cp.complex64)
# fluctuations_UP[:] = UP_w_kx_ky[:] - UP_stat_kx_ky

#axes for plots
time_list = cp.arange(t_min, t_max+dt_frame, dt_frame)
print(time_list.shape)
omega_list = 2*cp.pi*cp.fft.fftshift(cp.fft.fftfreq(fluctuations_LP.shape[-3], dt_frame))
k_1_list = 2*cp.pi*cp.fft.fftshift(cp.fft.fftfreq(fluctuations_LP.shape[-2], (X[Nx//2+side1/2-1]-X[Nx//2-side1/2])/side1)) #in 1/um PROBABLY TO CHECK ARGH
k_2_list = 2*cp.pi*cp.fft.fftshift(cp.fft.fftfreq(fluctuations_LP.shape[-1], (Y[Ny//2+side2/2-1]-Y[Ny//2-side2/2])/side2))

fluctuations_LP[...,:,fluctuations_LP.shape[-2]//2,fluctuations_LP.shape[-1]//2] = 0
fluctuations_LP[...,omega_list.shape[0]//2,:,:] = 0


#Phase and density movies:
# movies(folder, LP_t_x_y[:,:,:])
# movies(folder, fluctuations_LP)
# movies(folder, fluctuations_LP_t_x_y)

# movies(folder, zoom, title="zoom_fluct_LP_kspace", movie = "density")
# movies(folder, fluctuations_LP_t_kx_ky, title="fluct_LP_kspace", movie = "density")

# Anim from pictures of k space as a function of time:
arange = np.arange(0, time_list.shape[0]-50, 1)
max_dens = np.amax(np.abs(fluctuations_LP_t_kx_ky[0:-50,:,:].get()) ** 2)
for timeindex in tqdm(arange):
    plot_density(folder, ("$k_x$", Kx[window[0]:window[1]+1]), ("$k_y$", Ky[window[2]:window[3]+1]), ("fluct_LP_kspace_t"+str(time_list[timeindex]), fluctuations_LP_t_kx_ky[timeindex, :, :]), normalization = max_dens, norm="log")
print("dt_frame = ", time_list[1] - time_list[0])
scan = folder
img = cv2.imread(scan+"/fluct_LP_kspace_t0.0_density.png")
frameSize = (img.shape[1], img.shape[0])
out = cv2.VideoWriter(scan + "/anim.mp4",cv2.VideoWriter_fourcc(*'mp4v'), 10, frameSize)
for n in tqdm(arange):
    img = cv2.imread(scan+"/fluct_LP_kspace_t%s_density.png"%(str(time_list[n])))
    out.write(img)
out.release()

#Zoom around k1:
zoom_k1 = fluctuations_LP_t_kx_ky[:-50, fluctuations_LP_t_kx_ky.shape[-1]//2-5:fluctuations_LP_t_kx_ky.shape[-1]//2+6, fluctuations_LP_t_kx_ky.shape[-2]//2+37:fluctuations_LP_t_kx_ky.shape[-2]//2+48]
arange = np.arange(0, time_list.shape[0]-50, 1)
for timeindex in tqdm(arange):
    plot_density(folder, ("$k_x$", Kx[fluctuations_LP_t_kx_ky.shape[-2]//2+37:fluctuations_LP_t_kx_ky.shape[-2]//2+48]), ("$k_y$", Ky[fluctuations_LP_t_kx_ky.shape[-1]//2-5:fluctuations_LP_t_kx_ky.shape[-1]//2+6]), ("zoom_fluct_LP_kspace_t"+str(time_list[timeindex]), zoom_k1[timeindex, :, :]), normalization = 1, norm="log")
print("dt_frame = ", time_list[1] - time_list[0])
scan = folder
img = cv2.imread(scan+"/zoom_fluct_LP_kspace_t0.0_density.png")
frameSize = (img.shape[1], img.shape[0])
out = cv2.VideoWriter(scan + "/anim_zoom_k1.mp4",cv2.VideoWriter_fourcc(*'mp4v'), 10, frameSize)
for n in tqdm(arange):
    img = cv2.imread(scan+"/zoom_fluct_LP_kspace_t%s_density.png"%(str(time_list[n])))
    out.write(img)
out.release()

avg_zoom_k1 = cp.zeros(time_list.shape[0]-50, dtype = cp.float64)
for i in range(len(time_list)-50):
    avg_zoom_k1[i] += cp.average(cp.abs(zoom_k1[i])**2, axis=(-2, -1))
plt.figure()
plt.xlabel("Time ps")
plt.ylabel("Avg density around $k_x = 0.1$")
plt.plot(time_list[:-50].get(), avg_zoom_k1.get())
plt.savefig(folder+"/avg_zoom_k1.png")




#Plot bistability cycle:
# plot_gnLP_vs_I(folder, LP_t_x_y[:,:,:], F_t, R, g_LP, gamma_exc, gamma_cav, X02, C02, detuning = None, theoretical = False) #theoretical = False because it does not work aha

        
# for i in range(LP_stat_x_y.shape[0]):
#     for j in range(LP_stat_x_y.shape[1]):
#         #Plotting stationary densities of LP:
#         plot_density(folder, ("$k_x$", Kx[window[0]:window[1]+1]), ("$k_y$", Ky[window[2]:window[3]+1]), ("stationary_LP_kspace_i"+str(i)+"_j"+str(j), LP_stat_kx_ky[i,j,0,:,:]))
#         plot_density(folder, ("$x$", X[window[0]:window[1]+1]), ("$y$", Y[window[2]:window[3]+1]), ("stationary_LP_rspace_i"+str(i)+"_j"+str(j), LP_stat_x_y[i,j,:,:]))
#         #Plotting dispersions:
#         plot_density(folder, ("$k_x$",k_1_list), ("$\omega$", omega_list), ("w_kx_ky=0_i"+str(i)+"_j"+str(j), fluctuations_LP[i,j,::-1,fluctuations_LP.shape[1]//2,:]), norm="log") #careful with the order of the slicins, it is not intuitive imo but this is the correct way
#         plot_density(folder, ("$k_y$", k_2_list), ("$\omega$", omega_list), ("w_kx=0_ky_i"+str(i)+"_j"+str(j), fluctuations_LP[i,j,::-1,:,fluctuations_LP.shape[1]//2]), norm="log")
#         plot_density(folder, ("$k_x$", k_1_list), ("$k_y$", k_2_list), ("w="+str(omega_list[omega_list.shape[0]//3*2])+"_kx_ky_i"+str(i)+"_j"+str(j), fluctuations_LP[i,j,omega_list.shape[0]//3*2,:,:]), norm="log")

#Plotting stationary densities of LP:
# plot_density(folder, ("$k_x$", Kx[window[0]:window[1]+1]), ("$k_y$", Ky[window[2]:window[3]+1]), ("stationary_LP_kspace", LP_stat_kx_ky[:,:]))
# plot_density(folder, ("$x$", X[window[0]:window[1]+1]), ("$y$", Y[window[2]:window[3]+1]), ("stationary_LP_rspace", LP_stat_x_y))
# plot_density(folder, ("$x$", X[LP_stat_x_y.shape[0]//2-40:LP_stat_x_y.shape[0]//2+41]), ("$y$", Y[LP_stat_x_y.shape[1]//2-40:LP_stat_x_y.shape[1]//2+41]), ("stationary_LP_rspace_80x80", LP_stat_x_y[LP_stat_x_y.shape[0]//2-40:LP_stat_x_y.shape[0]//2+41, LP_stat_x_y.shape[1]//2-40:LP_stat_x_y.shape[1]//2+41]))

# plot_density(folder, ("$x$", X[window[0]:window[1]+1]), ("$y$", Y[window[2]:window[3]+1]), ("stationary_cav_rspace_fullwindow", cav_stationary_xy))
# plot_density(folder, ("$x$", X[Nx//2-40:Nx//2+41]), ("$y$", Y[Ny//2-40: Ny//2+41]), ("stationary_cav_rspace_window80x80", cav_stationary_xy[Nx//2-40:Nx//2+41, Ny//2-40:Ny//2+41]))

# cav_stationary_kx_ky = cp.fft.fftn(cav_stationary_xy, axes = (-2,-1))
# plot_density(folder, ("$k_x$", Kx[window[0]:window[1]+1]), ("$k_y$", Ky[window[2]:window[3]+1]), ("stationary_cav_kspace_log", cp.fft.fftshift(cav_stationary_kx_ky[:,:])), norm="log")
# plot_density(folder, ("$x$", Kx[window[0]:window[1]+1]), ("$y$", Ky[window[2]:window[3]+1]), ("stationary_exc_rspace", cav_stationary_xy[:,:]))
# exc_stationary_kx_ky = cp.fft.fftn(exc_stationary_xy, axes = (-2,-1))
# plot_density(folder, ("$k_x$", Kx[window[0]:window[1]+1]), ("$k_y$", Ky[window[2]:window[3]+1]), ("stationary_exc_kspace_log", cp.fft.fftshift(exc_stationary_kx_ky[:,:])), norm="log")

# plot_density(folder, ("$x$", X[LP_stat_x_y.shape[0]//2-40:LP_stat_x_y.shape[0]//2+41]), ("$y$", Y[LP_stat_x_y.shape[0]//2-40:LP_stat_x_y.shape[0]//2+41]), ("stationary_cav_rspace_80x80", cav_stationary_xy[LP_stat_x_y.shape[0]//2-40:LP_stat_x_y.shape[0]//2+41, LP_stat_x_y.shape[1]//2-40:LP_stat_x_y.shape[1]//2+41]))
# plot_density(folder, ("$x$", X[LP_stat_x_y.shape[0]//2-40:LP_stat_x_y.shape[0]//2+41]), ("$y$", Y[LP_stat_x_y.shape[0]//2-40:LP_stat_x_y.shape[0]//2+41]), ("stationary_exc_rspace_80x80", cav_stationary_xy[LP_stat_x_y.shape[0]//2-40:LP_stat_x_y.shape[0]//2+41, LP_stat_x_y.shape[1]//2-40:LP_stat_x_y.shape[1]//2+41]))

#Plotting dispersions:

# plot_density(folder, ("$k_x$",k_1_list), ("$\omega$", omega_list), ("w_kx_ky=0", fluctuations_LP[..., ::-1,fluctuations_LP.shape[-2]//2,:]), norm="log") #careful with the order of the slicins, it is not intuitive imo but this is the correct way
# plot_density(folder, ("$k_y$", k_2_list), ("$\omega$", omega_list), ("w_kx=0_ky", fluctuations_LP[..., ::-1,:,fluctuations_LP.shape[-1]//2]), norm="log")
# plot_density(folder, ("$k_x$", k_1_list), ("$k_y$", k_2_list), ("w="+str(omega_list[omega_list.shape[0]//3*2])+"_kx_ky", fluctuations_LP[..., omega_list.shape[0]//3*2,:,:]), norm="log")
# # plot_density(folder, ("$k_x$",k_1_list), ("$\omega$", omega_list), ("w_kx_ky=0", LP_w_kx_ky[..., ::-1,fluctuations_LP.shape[1]//2,:]), norm="log") #careful with the order of the slicins, it is not intuitive imo but this is the correct way
# # plot_density(folder, ("$k_y$", k_2_list), ("$\omega$", omega_list), ("w_kx=0_ky", LP_w_kx_ky[..., ::-1,:,fluctuations_LP.shape[1]//2]), norm="log")
# # plot_density(folder, ("$k_x$", k_1_list), ("$k_y$", k_2_list), ("w="+str(omega_list[omega_list.shape[0]//3*2])+"_kx_ky", LP_w_kx_ky[..., omega_list.shape[0]//3*2,:,:]), norm="log")
# plot_density(folder, ("$k_x$",k_1_list), ("$\omega$", omega_list), ("cav_w_kx_ky=0", cav_field_w_kx_ky[..., ::-1,cav_field_w_kx_ky.shape[1]//2,:]), norm="log") #careful with the order of the slicins, it is not intuitive imo but this is the correct way
# plot_density(folder, ("$k_y$", k_2_list), ("$\omega$", omega_list), ("cav_w_kx=0_ky", cav_field_w_kx_ky[..., ::-1,:,cav_field_w_kx_ky.shape[1]//2]), norm="log")
# plot_density(folder, ("$k_x$",k_1_list), ("$\omega$", omega_list), ("exc_w_kx_ky=0", exc_field_w_kx_ky[..., ::-1,exc_field_w_kx_ky.shape[1]//2,:]), norm="log") #careful with the order of the slicins, it is not intuitive imo but this is the correct way
# plot_density(folder, ("$k_y$", k_2_list), ("$\omega$", omega_list), ("exc_w_kx=0_ky", exc_field_w_kx_ky[..., ::-1,:,exc_field_w_kx_ky.shape[1]//2]), norm="log")



# plot_density(folder, ("$k_x$",k_1_list), ("$\omega$", omega_list), ("lin_w_kx_ky=0", fluctuations_LP[..., ::-1,fluctuations_LP.shape[-1]//2,:]), norm=None, vmax = 400000) #careful with the order of the slicins, it is not intuitive imo but this is the correct way
# plot_density(folder, ("$k_y$", k_2_list), ("$\omega$", omega_list), ("lin_w_kx=0_ky", fluctuations_LP[..., ::-1,:,fluctuations_LP.shape[-1]//2]), norm=None, vmax = 400000)

# plot_density(folder, ("$k_x$",k_1_list), ("$\omega$", omega_list), ("cav_w_kx_ky=0", cav_field_w_kx_ky[..., ::-1,cav_field_w_kx_ky.shape[-1]//2,:]), norm="log") #careful with the order of the slicins, it is not intuitive imo but this is the correct way
# plot_density(folder, ("$k_y$", k_2_list), ("$\omega$", omega_list), ("cav_w_kx=0_ky", cav_field_w_kx_ky[..., ::-1,:,cav_field_w_kx_ky.shape[-1]//2]), norm="log")
# plot_density(folder, ("$k_x$",k_1_list), ("$\omega$", omega_list), ("exc_w_kx_ky=0", exc_field_w_kx_ky[..., ::-1,exc_field_w_kx_ky.shape[-1]//2,:]), norm="log") #careful with the order of the slicins, it is not intuitive imo but this is the correct way
# plot_density(folder, ("$k_y$", k_2_list), ("$\omega$", omega_list), ("exc_w_kx=0_ky", exc_field_w_kx_ky[..., ::-1,:,exc_field_w_kx_ky.shape[-1]//2]), norm="log")







#photon's and exciton's dispersion
# cav_field_txy = cav_field_txy[..., :,window[0]:window[1]+1,window[2]:window[3]+1]
# exc_field_txy = exc_field_txy[..., :,window[0]:window[1]+1,window[2]:window[3]+1]
# cav_stationary_xy = cav_stationary_xy[..., window[0]:window[1]+1,window[2]:window[3]+1]
# exc_stationary_xy = exc_stationary_xy[..., window[0]:window[1]+1,window[2]:window[3]+1]
# cav_field_w_kx_ky = cp.fft.fftn(cav_field_txy, axes = (-3, -2, -1))
# exc_field_w_kx_ky = cp.fft.fftn(exc_field_txy, axes = (-3, -2, -1))
# cav_stationary_kx_ky = cp.fft.fftn(cav_stationary_xy, axes = (-2,-1))
# exc_stationary_kx_ky = cp.fft.fftn(exc_stationary_xy, axes = (-2,-1))

# fluct_cav = cp.fft.fftshift(cav_field_w_kx_ky - cav_stationary_kx_ky)
# plot_density(folder, ("$k_x$",k_1_list), ("$\omega$", omega_list), ("cav_w_kx_ky=0", fluct_cav[..., ::-1,fluct_cav.shape[1]//2,:]), norm="log") #careful with the order of the slicins, it is not intuitive imo but this is the correct way
# plot_density(folder, ("$k_y$", k_2_list), ("$\omega$", omega_list), ("cav_w_kx=0_ky", fluct_cav[..., ::-1,:,fluct_cav.shape[1]//2]), norm="log")
# plot_density(folder, ("$k_x$", k_1_list), ("$k_y$", k_2_list), ("cav_w="+str(omega_list[omega_list.shape[0]//3*2])+"_kx_ky", fluct_cav[..., fluct_cav.shape[0]//3*2,:,:]), norm="log")

# fluct_exc = cp.fft.fftshift(exc_field_w_kx_ky - exc_stationary_kx_ky)
# plot_density(folder, ("$k_x$",k_1_list), ("$\omega$", omega_list), ("exc_w_kx_ky=0", fluct_exc[..., ::-1,fluct_exc.shape[1]//2,:]), norm="log") #careful with the order of the slicins, it is not intuitive imo but this is the correct way
# plot_density(folder, ("$k_y$", k_2_list), ("$\omega$", omega_list), ("exc_w_kx=0_ky", fluct_exc[..., ::-1,:,fluct_exc.shape[1]//2]), norm="log")
# plot_density(folder, ("$k_x$", k_1_list), ("$k_y$", k_2_list), ("exc_w="+str(omega_list[omega_list.shape[0]//3*2])+"_kx_ky", fluct_exc[..., fluct_exc.shape[0]//3*2,:,:]), norm="log")



#For arrays in parallel:
# plot_density(folder, ("$k_x$", Kx[window[0]:window[1]+1]), ("$k_y$", Ky[window[2]:window[3]+1]), ("stationary_LP_kspace", LP_stat_kx_ky[0,:,:]))
# plot_density(folder, ("$x$", X[window[0]:window[1]+1]), ("$y$", Y[window[2]:window[3]+1]), ("stationary_LP_rspace", LP_stat_x_y[:,:]))

# plot_density(folder, ("$x$",X[window[0]:window[1]+1]), ("$y$", Y[window[0]:window[1]+1]), ("t="+str([time_list.shape[0]//3*2])+"_x_y", LP_t_x_y[..., time_list.shape[0]//3*2,:,:])) #careful with the order of the slicins, it is not intuitive imo but this is the correct way

# zoom_around_probe_res = fluctuations_LP[406:439, fluctuations_LP.shape[1]//2, 48:64]
# zoom_around_probe_nonres = fluctuations_LP_bis[406:439, fluctuations_LP.shape[1]//2, 65:85]
# plot_density(folder, ("$k_x$",k_1_list[48:64]), ("$\omega$", omega_list[406:439]), ("w_kx_ky=0_aroundprobe_resonant", zoom_around_probe_res), norm="log") #careful with the order of the slicins, it is not intuitive imo but this is the correct way
# plot_density(folder_bis, ("$k_x$", k_1_list[65:85]), ("$\omega$", omega_list[406:439]), ("w_kx_ky=0_aroundprobe_nonresonant", zoom_around_probe_nonres), norm="log")

# zoom_res_dens=cp.abs(zoom_around_probe_res)**2
# zoom_nonres_dens=cp.abs(zoom_around_probe_nonres)**2
# print(cp.amax(zoom_res_dens))
# print(cp.amax(zoom_nonres_dens))
# print(cp.where(zoom_res_dens==cp.amax(zoom_res_dens)))
# print(cp.where(zoom_nonres_dens==cp.amax(zoom_nonres_dens)))
# print(cp.amax(zoom_res_dens)/cp.amax(zoom_nonres_dens))


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

#plotting slices of cav field
# amplitude = cp.abs(cav_field_txy)**2
# amplitude_stat = cp.abs(cav_stationary_xy)**2
# plt.figure()
# plt.xlabel("$x$")
# plt.ylabel("$|F_{probe_r}|^{2}(y=0)$")
# plt.plot(X[amplitude_stat.shape[1]//2-25:amplitude_stat.shape[1]//2+26].get() , amplitude[1201, amplitude.shape[2]//2, amplitude[1].shape[1]//2-25:amplitude.shape[1]//2+26].get())
# plt.savefig(folder+"/slice_cav_density_t_2400_zoom_x.png")
# plt.close("all")