import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib
import cv2


def load_raw_data(
    folder, 
    path_ic = None, 
    only_stationary = False
):
    """Load raw data as cupy arrays from a folder.
    """
    
    mean_cav_t_x_y = np.load(folder + "/raw_arrays/mean_cav_t_x_y.npy")
    mean_exc_t_x_y = np.load(folder + "/raw_arrays/mean_exc_t_x_y.npy")
    if path_ic is None:
        stationary_cav_x_y = np.load(folder + "/raw_arrays/stationary_cav_x_y.npy")
        stationary_exc_x_y = np.load(folder + "/raw_arrays/stationary_exc_x_y.npy")
    else:
        stationary_cav_x_y = np.load(path_ic + "/raw_arrays/stationary_cav_x_y.npy")
        stationary_exc_x_y = np.load(path_ic + "/raw_arrays/stationary_exc_x_y.npy")
    hopfield_coefs = np.load(folder + "/raw_arrays/hopfield_coefs.npy")
    F_t = np.load(folder + "/raw_arrays/F_t.npy")
    
    if only_stationary == True:
        return cp.asarray(stationary_cav_x_y), cp.asarray(stationary_exc_x_y)
    else:
        return cp.asarray(mean_cav_t_x_y), cp.asarray(mean_exc_t_x_y), cp.asarray(stationary_cav_x_y), cp.asarray(stationary_exc_x_y), cp.asarray(hopfield_coefs), cp.asarray(F_t)

def config_plots():
    """Configuration of the plots
    """
    matplotlib.rcParams['figure.figsize'] = [10, 10]
    matplotlib.rcParams["legend.loc"] = 'upper right' 
    matplotlib.rcParams['axes.labelsize'] = 18
    matplotlib.rcParams['axes.titlesize'] = 20
    matplotlib.rc('xtick', labelsize=18) 
    matplotlib.rc('ytick', labelsize=18)
    matplotlib.rc('legend', fontsize = 16)
    plt.rcParams['text.usetex'] = True

def polariton_fields(
    cav_field_txy, 
    exc_field_txy, 
    hopfield_coefs, 
    window = None, 
    dx = 1,
    dy = 1,
    omega_exc = None,
    omega_cav = None,
    rabi = None,
    detuning = None,
    k_z = None,
    only_LP = False
):
    """Building the LP and UP fields from the photonic and excitonic fields.
    """
    if window == None:
        LP_w_kx_ky = cp.zeros(cav_field_txy.shape, dtype=cp.complex64)
        UP_w_kx_ky = cp.zeros(cav_field_txy.shape, dtype=cp.complex64)
        Xk = hopfield_coefs[0, :, :]
        Ck = hopfield_coefs[1, :, :]
        cav_field_w_kx_ky = cp.fft.fftn(cav_field_txy, axes = (-3,-2,-1))
        exc_field_w_kx_ky = cp.fft.fftn(exc_field_txy, axes = (-3,-2,-1))
        LP_w_kx_ky[..., :, :, :] = -1 * Xk[:, :] * exc_field_w_kx_ky[..., :, :, :] + Ck[:, :] * cav_field_w_kx_ky[..., :, :, :]
        UP_w_kx_ky[..., :, :, :] = Ck[:, :] * exc_field_w_kx_ky[..., :, :, :] + Xk[:, :] * cav_field_w_kx_ky[..., :, :, :]
        LP_t_x_y = cp.fft.ifftn(LP_w_kx_ky, axes = (-3,-2,-1))
        UP_t_x_y = cp.fft.ifftn(UP_w_kx_ky, axes = (-3,-2,-1)) 
        LP_w_kx_ky[..., :, :, :] = cp.fft.fftshift(LP_w_kx_ky[..., :, :, :], axes = (-3,-2,-1))
        UP_w_kx_ky[..., :, :, :] = cp.fft.fftshift(UP_w_kx_ky[..., :, :, :], axes = (-3,-2,-1))
        if only_LP == True:
            return LP_t_x_y, LP_w_kx_ky
        if only_LP == False:
            return LP_t_x_y, UP_t_x_y, LP_w_kx_ky, UP_w_kx_ky
    else:
        cav_field_txy = cav_field_txy[..., :, window[0]:window[1]+1, window[2]:window[3]+1]
        exc_field_txy = exc_field_txy[..., :, window[0]:window[1]+1, window[2]:window[3]+1]
        LP_w_kx_ky = cp.zeros(cav_field_txy.shape, dtype=cp.complex64)
        UP_w_kx_ky = cp.zeros(cav_field_txy.shape, dtype=cp.complex64)
        
        #Different window, different frequencies, different Xk and Ck
        hopfield_coefs_window = cp.zeros((2,)+LP_w_kx_ky.shape[-2:], dtype=cp.complex64)
        kx = cp.fft.fftfreq(cav_field_txy.shape[-2], d=dx)
        ky = cp.fft.fftfreq(cav_field_txy.shape[-1], d=dy)
        Kxx, Kyy = cp.meshgrid(kx, ky)
        omega_LP_0 = (omega_exc + omega_cav) / 2 - 0.5 * cp.sqrt((omega_exc - omega_cav) ** 2 + 4 * rabi ** 2)
        omega_pump = detuning + omega_LP_0
        omega = cp.zeros(hopfield_coefs_window.shape, dtype=np.complex64)
        omega[0, :, :] = omega_exc - omega_pump
        omega[1, :, :] = omega_cav * (cp.sqrt(1 + (Kxx ** 2 + Kyy ** 2) / k_z ** 2)) - omega_pump
        hopfield_coefs_window[0, :, :] =  cp.sqrt((cp.sqrt((omega[1, :, :] - omega[0, :, :]) ** 2 + 4 * rabi ** 2) - (omega[1, :, :] - omega[0, :, :])) / (2 * cp.sqrt((omega[1, :, :] - omega[0, :, :]) ** 2 + 4 * rabi ** 2)))
        hopfield_coefs_window[1, :, :] = cp.sqrt(1 - hopfield_coefs_window[1, :, :] ** 2)
        Xk = hopfield_coefs_window[0, :, :]
        Ck = hopfield_coefs_window[1, :, :]
        
        cav_field_w_kx_ky = cp.fft.fftn(cav_field_txy, axes = (-3, -2, -1))
        exc_field_w_kx_ky = cp.fft.fftn(exc_field_txy, axes = (-3, -2, -1))
        LP_w_kx_ky[..., :, :, :] = -1 * Xk[:, :] * exc_field_w_kx_ky[..., :, :, :] + Ck[:, :] * cav_field_w_kx_ky[..., :, :, :]
        UP_w_kx_ky[..., :, :, :] = Ck[:, :] * exc_field_w_kx_ky[..., :, :, :] + Xk[:, :] * cav_field_w_kx_ky[..., :, :, :]
        LP_t_x_y = cp.fft.ifftn(LP_w_kx_ky, axes = (-3, -2, -1))
        UP_t_x_y = cp.fft.ifftn(UP_w_kx_ky, axes = (-3, -2, -1))
        LP_w_kx_ky[..., :, :, :] = cp.fft.fftshift(LP_w_kx_ky[..., :, :, :], axes = (-3, -2, -1))
        UP_w_kx_ky[..., :, :, :] = cp.fft.fftshift(UP_w_kx_ky[..., :, :, :], axes = (-3, -2, -1))
        if only_LP == True:
            return LP_t_x_y, LP_w_kx_ky
        if only_LP == False:
            return LP_t_x_y, UP_t_x_y, LP_w_kx_ky, UP_w_kx_ky
    

def stationary_polariton_fields(
    cav_stationary_xy, 
    exc_stationary_xy, 
    hopfield_coefs, 
    window = None, 
    dx = 1,
    dy = 1,
    omega_exc = None,
    omega_cav = None,
    rabi = None,
    detuning = None,
    k_z = None,
    only_LP = False
):
    """Building the stationary LP and UP fields from the stationary photonic and excitonic fields.
    """
    if window == None:
        LP_stat_kx_ky = cp.zeros(cav_stationary_xy.shape, dtype=cp.complex64)
        UP_stat_kx_ky = cp.zeros(cav_stationary_xy.shape, dtype=cp.complex64)
        Xk = hopfield_coefs[0, :, :]
        Ck = hopfield_coefs[1, :, :]
        cav_stationary_kx_ky = cp.fft.fftn(cav_stationary_xy, axes = (-2, -1))
        exc_stationary_kx_ky = cp.fft.fftn(exc_stationary_xy, axes = (-2, -1))
        LP_stat_kx_ky[..., :, :] = -1 * Xk[:, :] * exc_stationary_kx_ky[..., :, :] + Ck[:, :] * cav_stationary_kx_ky[..., :, :] #you changed the minus, careful with convention
        UP_stat_kx_ky[..., :, :] = Ck[:, :] * exc_stationary_kx_ky[..., :, :] + Xk[:, :] * cav_stationary_kx_ky[..., :, :]
        LP_stat_x_y = cp.fft.ifftn(LP_stat_kx_ky, axes = (-2, -1))
        UP_stat_x_y = cp.fft.ifftn(UP_stat_kx_ky, axes = (-2, -1))  #do we need to normalize as in split-step??
        LP_stat_kx_ky[..., :, :] = cp.fft.fftshift(LP_stat_kx_ky[..., :, :], axes = (-2, -1))
        UP_stat_kx_ky[..., :, :] = cp.fft.fftshift(UP_stat_kx_ky[..., :, :], axes = (-2, -1))
    else:
        cav_stationary_xy = cav_stationary_xy[..., window[0]:window[1]+1, window[2]:window[3]+1]
        exc_stationary_xy = exc_stationary_xy[..., window[0]:window[1]+1, window[2]:window[3]+1]
        LP_stat_kx_ky = cp.zeros(cav_stationary_xy.shape, dtype=cp.complex64)
        UP_stat_kx_ky = cp.zeros(cav_stationary_xy.shape, dtype=cp.complex64)
        
        #Different window, different frequencies, different Xk and Ck
        hopfield_coefs_window = cp.zeros((2,)+LP_stat_kx_ky.shape[-2:], dtype=cp.complex64)
        kx = cp.fft.fftfreq(cav_stationary_xy.shape[-2], d=dx)
        ky = cp.fft.fftfreq(cav_stationary_xy.shape[-1], d=dy)
        Kxx, Kyy = cp.meshgrid(kx, ky)
        omega_LP_0 = (omega_exc + omega_cav) / 2 - 0.5 * cp.sqrt((omega_exc - omega_cav) ** 2 + 4 * rabi ** 2)
        omega_pump = detuning + omega_LP_0
        omega = cp.zeros(hopfield_coefs_window.shape, dtype=np.complex64)
        omega[0, :, :] = omega_exc - omega_pump
        omega[1, :, :] = omega_cav * (cp.sqrt(1 + (Kxx ** 2 + Kyy ** 2) / k_z ** 2)) - omega_pump
        hopfield_coefs_window[0, :, :] =  cp.sqrt((cp.sqrt((omega[1, :, :] - omega[0, :, :]) ** 2 + 4 * rabi ** 2) - (omega[1, :, :] - omega[0, :, :])) / (2 * cp.sqrt((omega[1, :, :] - omega[0, :, :]) ** 2 + 4 * rabi ** 2)))
        hopfield_coefs_window[1, :, :] = cp.sqrt(1 - hopfield_coefs_window[1, :, :] ** 2)
        Xk = hopfield_coefs_window[0, :, :]
        Ck = hopfield_coefs_window[1, :, :]
        
        cav_stationary_kx_ky = cp.fft.fftn(cav_stationary_xy, axes = (-2, -1))
        exc_stationary_kx_ky = cp.fft.fftn(exc_stationary_xy, axes = (-2, -1))
        LP_stat_kx_ky[..., :, :] = -1 * Xk[:, :] * exc_stationary_kx_ky[..., :, :] + Ck[:, :] * cav_stationary_kx_ky[..., :, :]
        UP_stat_kx_ky[..., :, :] = Ck[:, :] * exc_stationary_kx_ky[..., :, :] + Xk[:, :] * cav_stationary_kx_ky[..., :, :]
        LP_stat_x_y = cp.fft.ifftn(LP_stat_kx_ky[..., :, :], axes = (-2, -1))
        UP_stat_x_y = cp.fft.ifftn(UP_stat_kx_ky[..., :, :], axes = (-2, -1))
        LP_stat_kx_ky[..., :, :] = cp.fft.fftshift(LP_stat_kx_ky[..., :, :], axes = (-2, -1))
        UP_stat_kx_ky[..., :, :] = cp.fft.fftshift(UP_stat_kx_ky[..., :, :], axes = (-2, -1))
        
    if only_LP == True:
        return LP_stat_x_y, LP_stat_kx_ky
    if only_LP == False:
        return LP_stat_x_y, UP_stat_x_y, LP_stat_kx_ky, UP_stat_kx_ky
    

def movies(
    folder, 
    field_txy, 
    title = "",
    movie = "both"
):
    """Create movies of the density and phase of the field.
    """
    field_txy = cp.asnumpy(field_txy)
    size = (field_txy[0].shape)
    fps = 15
    
    if movie == "density":
        out_dens = cv2.VideoWriter(folder +"/"+ title + "_dens_evolution.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
        max_dens = np.amax(np.abs(field_txy) ** 2)
        for i in range(len(field_txy)):
            dens = np.array(np.abs(field_txy[i, ::-1, :]) ** 2 * 255 / max_dens, dtype=np.uint8)
            out_dens.write(dens)
        out_dens.release()
        
    if movie == "phase":
        out_phase = cv2.VideoWriter(folder +"/"+ title + "_phase_evolution.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
        for i in range(len(field_txy)):
            phase = np.array(np.angle(field_txy[i, ::-1, :]), dtype=np.uint8)
            out_phase.write(phase)
        out_phase.release()
        
    if movie == "both":
        out_dens = cv2.VideoWriter(folder +"/"+ title + "_dens_evolution.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
        out_phase = cv2.VideoWriter(folder +"/"+ title + "_phase_evolution.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
        max_dens = np.amax(np.abs(field_txy) ** 2)
        for i in range(len(field_txy)):
            dens = np.array(np.abs(field_txy[i, ::-1, :]) ** 2 * 255 / max_dens, dtype=np.uint8)
            phase = np.array(np.angle(field_txy[i, ::-1, :]), dtype=np.uint8)
            out_dens.write(dens)
            out_phase.write(phase)
        out_dens.release()
        out_phase.release()
    

def plot_gnLP_vs_I(
    folder, 
    LP_t_x_y, 
    F_t, 
    R, 
    g, 
    gamma_exc, 
    gamma_ph, 
    X02, 
    C02, 
    detuning = None, 
    theoretical = False    #did not work, still to correct
):
    """Generate plot of the average density of LP nea the center of the cavity vs the intensity of the field.
    """
    Nx, Ny = LP_t_x_y.shape[-2], LP_t_x_y.shape[-1]
    avg_density = cp.zeros(len(F_t))
    LP_density = cp.abs(LP_t_x_y)**2
    radius=15
    disk = cp.zeros((Nx, Ny))
    disk[R < radius] += 1
    for i in range(len(F_t)):
        avg_density[i] += cp.average(LP_density[i], axis=(-2, -1), weights = disk) * g
    F_intensity = np.abs(F_t.get()) ** 2
    
    plt.figure()
    plt.xlabel("Intensity $I$")
    plt.ylabel("Density $g_{LP}n_{LP}$")
    plt.scatter(F_intensity, avg_density.get())
    if detuning != None:
        plt.hlines(detuning, 0, np.max(F_intensity), colors = "r", linestyles = "dashed", label="Detuning = " + str(detuning))
        plt.legend()
        plt.savefig(folder + "/In_loop_avg_rad" + str(radius) + ".png")
    if theoretical == True:
        I_vs_n = np.array([n * ((detuning - g * n) ** 2 + (gamma_exc * X02 + gamma_ph * C02) ** 2 / 4) for n in np.linspace(0, 0.65, len(F_intensity))])
        plt.scatter(F_intensity, I_vs_n, label="Theoretical curve", color = "k", marker = "x")
        plt.legend()
        plt.savefig(folder + "/In_loop_avg_rad" + str(radius) + "_theory.png")
    else:
        plt.legend()
        plt.savefig(folder + "/In_loop_avg_rad" + str(radius) + ".png")
    plt.close("all")
    
    
def plot_density(
    folder, 
    x, 
    y, 
    field, 
    normalization = 1,
    norm = None,
    vmax = None
):
    """Generate a 2D plot of the density of the field.
    """
    
    if len(field[1].shape) == 2:
        plt.figure()
        plt.xlabel(x[0])
        plt.ylabel(y[0])
        if vmax == None:
            to_plot = (np.abs(field[1].get()) ** 2 / normalization)
            plt.pcolormesh(x[1].get(), y[1].get(), to_plot, norm = norm)
            plt.colorbar()
        else:
            to_plot = (np.abs(field[1].get()) ** 2 / normalization)
            plt.pcolormesh(x[1].get(), y[1].get(), to_plot, norm = norm, vmax = vmax)
            plt.colorbar()
        plt.savefig(folder+"/" + field[0] + "_density.png")
        plt.close("all")
    if len(field[1].shape) == 3: # if we have only one scanning parameter e.g. [k_probe, x, y] or [w_probe, x, y]
        for i in range(field[1].shape[-3]):
            plt.figure()
            plt.xlabel(x[0])
            plt.ylabel(y[0])
            if vmax == None:
                to_plot = (np.abs(field[1][i, :, :].get()) ** 2 / normalization)
                plt.pcolormesh(x[1].get(), y[1].get(), to_plot, norm = norm)
                plt.colorbar()
            else:
                to_plot = (np.abs(field[1][i, :, :].get()) ** 2 / normalization)
                plt.pcolormesh(x[1].get(), y[1].get(), to_plot, norm = norm, vmax = vmax)
                plt.colorbar()
            plt.savefig(folder+"/" + field[0] + "_density_" + str(i) + ".png")
            plt.close("all")
    if len(field[1].shape) == 4: #if we have 2 scanning parameters e.g. [k_probe,w_probe, x, y]
        for i in range(field[1].shape[-4]):
            for j in range(field[1].shape[-3]):
                plt.figure()
                plt.xlabel(x[0])
                plt.ylabel(y[0])
                if vmax == None:
                    to_plot = (np.abs(field[1][i, j, :, :].get()) ** 2 / normalization)
                    plt.pcolormesh(x[1].get(), y[1].get(), to_plot, norm = norm)
                    plt.colorbar()
                else:
                    to_plot = (np.abs(field[1][i, j, :, :].get()) ** 2 / normalization)
                    plt.pcolormesh(x[1].get(), y[1].get(), to_plot, norm = norm, vmax = vmax)
                    plt.colorbar()
                plt.savefig(folder + "/"+field[0] + "_density_" + str(i)+"_" + str(j) + ".png")
                plt.close("all")
                
def scan_output(
    folder,
    fluctuations_LP,
    Kx_scan,
    omega_scan,
    side_square_filter,
    Kxx,
    Kyy,
    kx = 0.5
):  
    if fluctuations_LP.shape[0] > 1:
        output = np.zeros((fluctuations_LP.shape[1], fluctuations_LP.shape[0]))
        for i in range(fluctuations_LP.shape[0]):
            mask = cp.ones(fluctuations_LP.shape[-2:])
            mask[cp.abs(Kxx-Kx_scan[i]) > side_square_filter] = 0
            mask[cp.abs(Kyy) > side_square_filter] = 0
            for j in range(fluctuations_LP.shape[1]):
                avg = cp.average(cp.abs(fluctuations_LP[i,j])**2, axis=(-2, -1), weights = mask)
                output[j,i] = avg
        plt.figure()
        plt.pcolormesh(Kx_scan.get(), omega_scan.get(), output[::,::], norm="log")
        plt.colorbar()
        plt.savefig(folder+"/scan_output_log.png")
        plt.close("all")
        plt.figure()
        plt.pcolormesh(Kx_scan.get(), omega_scan.get(), output[::,::])
        plt.colorbar()
        plt.savefig(folder+"/scan_output.png")
        plt.close("all")
    
    if fluctuations_LP.shape[0] == 1:
        output = np.zeros((fluctuations_LP.shape[1]))
        mask = cp.ones(fluctuations_LP.shape[-2:])
        mask[cp.abs(Kxx - kx) > side_square_filter] = 0
        mask[cp.abs(Kyy) > side_square_filter] = 0
        for j in range(fluctuations_LP.shape[1]):
            avg = cp.average(cp.abs(fluctuations_LP[0,j])**2, axis=(-2, -1), weights = mask)
            output[j] = avg
        plt.figure()
        plt.plot(omega_scan.get(), np.log(output[::]))
        plt.savefig(folder+"/scan_output.png")
        plt.close("all")
    

def scan_output_4WM(
    folder,
    fluctuations_LP,
    Kx_scan,
    omega_scan,
    side_square_filter,
    Kxx,
    Kyy,
    kx = 0.5
):  
    if fluctuations_LP.shape[0] > 1:
        output = np.zeros((fluctuations_LP.shape[1], fluctuations_LP.shape[0]))
        for i in range(fluctuations_LP.shape[0]):
            mask = cp.ones(fluctuations_LP.shape[-2:])
            mask[cp.abs(Kxx + Kx_scan[i]) > side_square_filter] = 0
            mask[cp.abs(Kyy) > side_square_filter] = 0
            for j in range(fluctuations_LP.shape[1]):
                avg = cp.average(cp.abs(fluctuations_LP[i,j])**2, axis=(-2, -1), weights = mask)
                output[-1-j,-1-i] = avg
        plt.figure()
        plt.pcolormesh(Kx_scan.get(), omega_scan.get(), output[::,::], norm="log")
        plt.colorbar()
        plt.savefig(folder+"/scan_output_log_4WM.png")
        plt.close("all")
        plt.figure()
        plt.pcolormesh(Kx_scan.get(), omega_scan.get(), output[::,::])
        plt.colorbar()
        plt.savefig(folder+"/scan_output_4WM.png")
        plt.close("all")
    
    if fluctuations_LP.shape[0] == 1:
        output = np.zeros((fluctuations_LP.shape[1]))
        mask = cp.ones(fluctuations_LP.shape[-2:])
        mask[cp.abs(Kxx + kx) > side_square_filter] = 0
        mask[cp.abs(Kyy) > side_square_filter] = 0
        for j in range(fluctuations_LP.shape[1]):
            avg = cp.average(cp.abs(fluctuations_LP[0,j])**2, axis=(-2, -1), weights = mask)
            output[-1-j] = avg
        plt.figure()
        plt.plot(omega_scan.get(), output[::])
        plt.savefig(folder+"/scan_output_4WM.png")
        plt.close("all")
        plt.figure()
        plt.plot(omega_scan.get(), np.log(output[::]))
        plt.savefig(folder+"/scan_output_log_4WM.png")
        plt.close("all")
   