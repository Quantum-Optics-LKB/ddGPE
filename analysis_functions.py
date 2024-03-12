import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib
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
import cv2

def load_raw_data(folder):
    
    #Load raw data as numpy arrays
    mean_cav_t_x_y = np.load(folder+"/raw_arrays/mean_cav_t_x_y.npy")
    mean_exc_t_x_y = np.load(folder+"/raw_arrays/mean_exc_t_x_y.npy")
    stationary_cav_x_y = np.load(folder+"/raw_arrays/stationary_cav_x_y.npy")
    stationary_exc_x_y = np.load(folder+"/raw_arrays/stationary_exc_x_y.npy")
    hopfield_coefs = np.load(folder+"/raw_arrays/hopfield_coefs.npy")
    F_t = np.load(folder+"/raw_arrays/F_t.npy")
    
    return mean_cav_t_x_y, mean_exc_t_x_y, stationary_cav_x_y, stationary_exc_x_y, hopfield_coefs, F_t

def config_plots():
    matplotlib.rcParams['figure.figsize'] = [10, 10]
    matplotlib.rcParams["legend.loc"] = 'upper right' 
    matplotlib.rcParams['axes.labelsize'] = 18
    matplotlib.rcParams['axes.titlesize'] = 20
    matplotlib.rc('xtick', labelsize=18) 
    matplotlib.rc('ytick', labelsize=18)
    matplotlib.rc('legend', fontsize = 16)
    plt.rcParams['text.usetex'] = True

#Creating LP and UP fields:
def polariton_fields(cav_field_txy, exc_field_txy, hopfield_coefs, window = None):
    if window == None:
        LP_w_kx_ky = cp.zeros(cav_field_txy.shape, dtype=cp.complex64)
        UP_w_kx_ky = cp.zeros(cav_field_txy.shape, dtype=cp.complex64)
        Xk = hopfield_coefs[0,:,:]
        Ck = hopfield_coefs[1,:,:]
        cav_field_w_kx_ky = cp.fft.fftn(cav_field_txy)
        exc_field_w_kx_ky = cp.fft.fftn(exc_field_txy)
        LP_w_kx_ky[:] = Xk*exc_field_w_kx_ky[:] - Ck*cav_field_w_kx_ky[:]
        UP_w_kx_ky[:] = Ck*exc_field_w_kx_ky[:] + Xk*cav_field_w_kx_ky[:]
        LP_t_x_y = cp.fft.ifftn(LP_w_kx_ky)
        UP_t_x_y = cp.fft.ifftn(UP_w_kx_ky) 
        #LP_t_x_y /= np.prod(LP_t_x_y.shape)
        #UP_t_x_y /= np.prod(UP_t_x_y.shape)#do we need to normalize as in split-step??
        LP_w_kx_ky[:] = cp.fft.fftshift(LP_w_kx_ky[:])
        UP_w_kx_ky[:] = cp.fft.fftshift(UP_w_kx_ky[:])
    else:
        cav_field_txy = cav_field_txy[:,window[0]:window[1]+1,window[2]:window[3]+1]
        exc_field_txy = exc_field_txy[:,window[0]:window[1]+1,window[2]:window[3]+1]
        LP_w_kx_ky = cp.zeros(cav_field_txy.shape, dtype=cp.complex64)
        UP_w_kx_ky = cp.zeros(cav_field_txy.shape, dtype=cp.complex64)
        Xk = hopfield_coefs[0,window[0]:window[1]+1,window[2]:window[3]+1]
        Ck = hopfield_coefs[1,window[0]:window[1]+1,window[2]:window[3]+1]
        cav_field_w_kx_ky = cp.fft.fftn(cav_field_txy)
        exc_field_w_kx_ky = cp.fft.fftn(exc_field_txy)
        LP_w_kx_ky[:] = Xk*exc_field_w_kx_ky[:] - Ck*cav_field_w_kx_ky[:]
        UP_w_kx_ky[:] = Ck*exc_field_w_kx_ky[:] + Xk*cav_field_w_kx_ky[:]
        LP_t_x_y = cp.fft.ifftn(LP_w_kx_ky)
        UP_t_x_y = cp.fft.ifftn(UP_w_kx_ky)
        LP_w_kx_ky[:] = cp.fft.fftshift(LP_w_kx_ky[:])
        UP_w_kx_ky[:] = cp.fft.fftshift(UP_w_kx_ky[:])
        
    return LP_t_x_y, UP_t_x_y, LP_w_kx_ky, UP_w_kx_ky

#Creating LP and UP stationary fields:
def stationary_polariton_fields(cav_stationary_xy, exc_stationary_xy, hopfield_coefs, window = None):
    if window == None:
        LP_stat_kx_ky = cp.zeros(cav_stationary_xy.shape, dtype=cp.complex64)
        UP_stat_kx_ky = cp.zeros(cav_stationary_xy.shape, dtype=cp.complex64)
        Xk = hopfield_coefs[0,:,:]
        Ck = hopfield_coefs[1,:,:]
        cav_stationary_kx_ky = cp.fft.fftn(cav_stationary_xy)
        exc_stationary_kx_ky = cp.fft.fftn(exc_stationary_xy)
        LP_stat_kx_ky[:] = Xk*exc_stationary_kx_ky[:] - Ck*cav_stationary_kx_ky[:]
        UP_stat_kx_ky[:] = Ck*exc_stationary_kx_ky[:] + Xk*cav_stationary_kx_ky[:]
        LP_stat_x_y = cp.fft.ifftn(LP_stat_kx_ky)
        UP_stat_x_y = cp.fft.ifftn(UP_stat_kx_ky)  #do we need to normalize as in split-step??
        LP_stat_kx_ky[:] = cp.fft.fftshift(LP_stat_kx_ky[:])
        UP_stat_kx_ky[:] = cp.fft.fftshift(UP_stat_kx_ky[:])
    else:
        cav_stationary_xy = cav_stationary_xy[window[0]:window[1]+1,window[2]:window[3]+1]
        exc_stationary_xy = exc_stationary_xy[window[0]:window[1]+1,window[2]:window[3]+1]
        LP_stat_kx_ky = cp.zeros(cav_stationary_xy.shape, dtype=cp.complex64)
        UP_stat_kx_ky = cp.zeros(cav_stationary_xy.shape, dtype=cp.complex64)
        Xk = hopfield_coefs[0,window[0]:window[1]+1,window[2]:window[3]+1]
        Ck = hopfield_coefs[1,window[0]:window[1]+1,window[2]:window[3]+1]
        cav_stationary_kx_ky = cp.fft.fftn(cav_stationary_xy)
        exc_stationary_kx_ky = cp.fft.fftn(exc_stationary_xy)
        LP_stat_kx_ky[:] = Xk*exc_stationary_kx_ky[:] - Ck*cav_stationary_kx_ky[:]
        UP_stat_kx_ky[:] = Ck*exc_stationary_kx_ky[:] + Xk*cav_stationary_kx_ky[:]
        LP_stat_x_y = cp.fft.ifftn(LP_stat_kx_ky)
        UP_stat_x_y = cp.fft.ifftn(UP_stat_kx_ky)
        LP_stat_kx_ky[:] = cp.fft.fftshift(LP_stat_kx_ky[:])
        UP_stat_kx_ky[:] = cp.fft.fftshift(UP_stat_kx_ky[:])
        
    return LP_stat_x_y, UP_stat_x_y, LP_stat_kx_ky, UP_stat_kx_ky


#Generating the movies:
def movies(folder, field_txy, movie = "both"):
    field_txy = cp.asnumpy(field_txy)
    size=(field_txy[0].shape)
    fps=15
    
    if movie == "density":
        out_dens = cv2.VideoWriter(folder +"/dens_evolution.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
        max_dens = np.amax(np.abs(field_txy)**2)
        for i in range(len(field_txy)):
            dens = np.array(np.abs(field_txy[i])**2*255/max_dens, dtype = np.uint8)
            out_dens.write(dens)
        out_dens.release()
        
    if movie == "phase":
        out_phase = cv2.VideoWriter(folder +"/phase_evolution.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
        for i in range(len(field_txy)):
            phase = np.array(np.angle(field_txy[i]), dtype = np.uint8)
            out_phase.write(phase)
        out_phase.release()
        
    if movie == "both":
        out_dens = cv2.VideoWriter(folder +"/dens_evolution.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
        out_phase = cv2.VideoWriter(folder +"/phase_evolution.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
        max_dens = np.amax(np.abs(field_txy)**2)
        for i in range(len(field_txy)):
            dens = np.array(np.abs(field_txy[i])**2*255/max_dens, dtype = np.uint8)
            phase = np.array(np.angle(field_txy[i]), dtype = np.uint8)
            out_dens.write(dens)
            out_phase.write(phase)
        out_dens.release()
        out_phase.release()
    

#Functions for ploting:

def plot_gnLP_vs_I(folder, LP_t_x_y, F_t, R, g, gamma_exc, gamma_ph, X02, C02, detuning = None, theoretical = True):
    nmax_1, nmax_2 = LP_t_x_y.shape[-2], LP_t_x_y.shape[-1]
    avg_density = cp.zeros(len(F_t))
    LP_density = cp.abs(LP_t_x_y)**2
    radius=15
    disk = cp.zeros((nmax_1, nmax_2))
    disk[R < radius] += 1
    for i in range(len(F_t)):
        avg_density[i] += cp.average(LP_density[i], axis=(-2,-1), weights = disk) * g
    #center_density = LP_density[:,nmax_1//2, nmax_2//2] * g 
    F_intensity = np.abs(F_t.get())**2
    
    plt.figure()
    plt.xlabel("Intensity $I$")
    plt.ylabel("Density $g_{LP}n_{LP}$")
    #plt.scatter(F_intensity, center_density.get())
    plt.scatter(F_intensity, avg_density.get())
    if detuning != None:
        plt.hlines(detuning, 0, np.max(F_intensity), colors = "r", linestyles = "dashed", label="Detuning = "+str(detuning))
        plt.legend()
        plt.savefig(folder+"/In_loop_avg_rad"+str(radius)+".png")
    if theoretical == True:
        I_vs_n = np.array([n*((detuning-g*n)**2+(gamma_exc*X02+gamma_ph*C02)**2/4) for n in np.linspace(0, 0.65, len(F_intensity))])
        plt.scatter(F_intensity, I_vs_n, label="Theoretical curve", color = "k", marker = "x")
        plt.legend()
        plt.savefig(folder+"/In_loop_avg_rad"+str(radius)+"_theory.png")
    else:
        plt.legend()
        plt.savefig(folder+"/In_loop_avg_rad"+str(radius)+".png")
    
    plt.close("all")
    
def plot_density(folder, x, y, field, norm = None):
    plt.figure()
    plt.xlabel(x[0])
    plt.ylabel(y[0])
    plt.pcolormesh(x[1].get(), y[1].get(), np.abs(field[1].get())**2, norm = norm)
    plt.savefig(folder+"/"+field[0]+"_density.png")   
    plt.close("all")
