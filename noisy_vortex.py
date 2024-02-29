import datetime
import json
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
from skimage.restoration import unwrap_phase
import os
import scipy
#from azim_avg import mean_azim
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cupyx.scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
import scipy.constants as const
#from utilitiesf import create_gif
#from Velocity import velocity
import cProfile
import pstats
import cv2

matplotlib.rcParams['figure.figsize'] = [10, 10]
matplotlib.rcParams["legend.loc"] = 'upper right' 
matplotlib.rcParams['axes.labelsize'] = 18
matplotlib.rcParams['axes.titlesize'] = 20
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)
matplotlib.rc('legend', fontsize = 16)
#%%
def save_data_giant_vortex(folder,parameters):

    # mean_cav_x_y_t = cp.asnumpy(simu.mean_cav_x_y_t)
    # mean_exc_x_y_t = cp.asnumpy(simu.mean_exc_x_y_t)
    # F_t = cp.asnumpy(simu.F_t)
    
    # mean_cav_t_x_y = np.einsum('xyt->txy', mean_cav_x_y_t)
    # mean_exc_t_x_y = np.einsum('xyt->txy', mean_exc_x_y_t)
    
    mean_cav_t_x_y = cp.asnumpy(simu.mean_cav_t_x_y)
    mean_exc_t_x_y = cp.asnumpy(simu.mean_exc_t_x_y)
    F_t = cp.asnumpy(simu.F_t)
    
    #----------------oscar's observables----------------
    
    # real space density avg
    
    # density_cav = np.zeros((len(mean_cav_t_x_y),len(mean_cav_t_x_y[0]),len(mean_cav_t_x_y[0][0])), dtype = np.complex64)
    # density_exc = np.zeros((len(mean_cav_t_x_y),len(mean_cav_t_x_y[0]),len(mean_cav_t_x_y[0][0])), dtype = np.complex64)
    
    # density_cav[:,:,:]=np.abs(mean_cav_t_x_y[:,:,:])**2
    # density_exc[:,:,:]=np.abs(mean_exc_t_x_y[:,:,:])**2
    
    # avg_dens_cav_t = np.zeros(len(mean_cav_t_x_y), dtype=np.complex64)
    # avg_dens_exc_t = np.zeros(len(mean_exc_t_x_y), dtype=np.complex64)
    
    # avg_dens_cav_t[:] = np.average(density_cav, axis=(1,2))
    # avg_dens_exc_t[:] = np.average(density_exc, axis=(1,2))
    
    # LP_rspace=np.zeros((len(mean_cav_t_x_y),len(mean_cav_t_x_y[0]),len(mean_cav_t_x_y[0][0])), dtype = np.complex64)
    # UP_rspace=np.zeros((len(mean_cav_t_x_y),len(mean_cav_t_x_y[0]),len(mean_cav_t_x_y[0][0])), dtype = np.complex64)
    
    # for i in range(len(mean_cav_t_x_y)):
        
    #     # # Carlon Zambon's convention
    #     # LP_rspace[i,:,:] = np.sqrt(X02)*mean_exc_t_x_y[i,:,:] + np.sqrt(C02)*mean_cav_t_x_y[i,:,:]
    #     # UP_rspace[i,:,:] = np.sqrt(C02)*mean_exc_t_x_y[i,:,:] - np.sqrt(X02)*mean_cav_t_x_y[i,:,:]
        
    #     # Kilian's code convention
    #     LP_rspace[i,:,:] = np.sqrt(X02)*mean_exc_t_x_y[i,:,:] - np.sqrt(C02)*mean_cav_t_x_y[i,:,:]
    #     UP_rspace[i,:,:] = np.sqrt(C02)*mean_exc_t_x_y[i,:,:] + np.sqrt(X02)*mean_cav_t_x_y[i,:,:]
    
    # avg_dens_LP_rspace_t = np.zeros(len(mean_cav_t_x_y), dtype=np.complex64)
    # avg_dens_UP_rspace_t = np.zeros(len(mean_cav_t_x_y), dtype=np.complex64)
    
    # avg_dens_LP_rspace_t[:] = np.average(np.abs(LP_rspace)**2, axis=(1,2))
    # avg_dens_UP_rspace_t[:] = np.average(np.abs(UP_rspace)**2, axis=(1,2))
        
    # plt.figure()
    # plt.plot(avg_dens_LP_rspace_t, label ='LP')
    # plt.plot(avg_dens_UP_rspace_t, label ='UP')
    # plt.plot(avg_dens_cav_t, label ='cav')
    # plt.plot(avg_dens_exc_t, label ='exc')
    # plt.xlabel("time")
    # plt.ylabel("avg_density")
    # plt.legend()
    # plt.savefig(folder+"/avg_densities_rspace_t_Kconv.png")
    # plt.close("all")
    
    
    # kspace density avg
    
    # density_cav_kspace = np.zeros((len(mean_cav_t_x_y),len(mean_cav_t_x_y[0]),len(mean_cav_t_x_y[0][0])), dtype = np.complex64)
    # density_exc_kspace = np.zeros((len(mean_cav_t_x_y),len(mean_cav_t_x_y[0]),len(mean_cav_t_x_y[0][0])), dtype = np.complex64)
    
    # for i in range(len(mean_cav_t_x_y)):
    #     density_cav_kspace[i] = np.fft.fft2(mean_cav_t_x_y[i])
    #     density_exc_kspace[i] = np.fft.fft2(mean_exc_t_x_y[i])
    
    # Ck=np.sqrt(simu.C2)
    # Xk=np.sqrt(1-simu.C2)
    
    # LP_kspace=np.zeros((len(mean_cav_t_x_y),len(mean_cav_t_x_y[0]),len(mean_cav_t_x_y[0][0])), dtype = np.complex64)
    # UP_kspace=np.zeros((len(mean_cav_t_x_y),len(mean_cav_t_x_y[0]),len(mean_cav_t_x_y[0][0])), dtype = np.complex64)
    
    # for i in range(len(mean_cav_t_x_y)):
        
    #     # # Carlon Zambon's convention
    #     # LP_kspace[i,:,:] = Xk[:,:]*density_exc_kspace[i,:,:] + Ck[:,:]*density_cav_kspace[i,:,:]
    #     # UP_kspace[i,:,:] = Ck[:,:]*density_exc_kspace[i,:,:] - Xk[:,:]*density_cav_kspace[i,:,:]
        
    #     # Kilian's code convention
    #     LP_kspace[i,:,:] = Xk[:,:]*density_exc_kspace[i,:,:] - Ck[:,:]*density_cav_kspace[i,:,:]
    #     UP_kspace[i,:,:] = Ck[:,:]*density_exc_kspace[i,:,:] + Xk[:,:]*density_cav_kspace[i,:,:]
   
    # avg_dens_LP_kspace_t = np.zeros(len(mean_cav_t_x_y), dtype=np.complex64)
    # avg_dens_UP_kspace_t = np.zeros(len(mean_cav_t_x_y), dtype=np.complex64)
    
    # avg_dens_LP_kspace_t[:] = np.average(np.abs(LP_kspace)**2, axis=(1,2))
    # avg_dens_UP_kspace_t[:] = np.average(np.abs(UP_kspace)**2, axis=(1,2))
    
    # plt.figure()
    # plt.plot(avg_dens_LP_kspace_t, label ='LP_k')
    # plt.plot(avg_dens_UP_kspace_t, label ='UP_k')
    # #plt.plot(avg_dens_cav_t, label ='cav')
    # #plt.plot(avg_dens_exc_t, label ='exc')
    # plt.xlabel("time")
    # plt.ylabel("avg_density")
    # plt.legend()
    # plt.savefig(folder+"/avg_densities_kspace_t.png")
    # plt.close("all")
    
    
    #Checking diagonalization
    
    # h0 = cp.asarray(simu.h_lin_0)
    # h0_diag = cp.zeros((2, 2, nmax_1, nmax_2), dtype=np.complex64)
    # h0_diag[0, 0, :, :] = cp.multiply(h0[0,0,:,:], simu.X2[:,:])+cp.multiply(h0[1,1,:,:], simu.C2[:,:])-2*cp.multiply(cp.sqrt(simu.X2[:,:]), cp.sqrt(simu.C2[:,:]))*h0[0,1,:,:]
    # h0_diag[1, 1, :, :] = cp.multiply(h0[0,0,:,:], simu.C2[:,:])+cp.multiply(h0[1,1,:,:], simu.X2[:,:])+2*cp.multiply(cp.sqrt(simu.X2[:,:]), cp.sqrt(simu.C2[:,:]))*h0[0,1,:,:]
    # h0_diag[0, 1, :, :] = cp.multiply(cp.sqrt(simu.X2[:,:]), cp.sqrt(simu.C2[:,:]))*h0[0,0,:,:]-cp.multiply(cp.sqrt(simu.X2[:,:]), cp.sqrt(simu.C2[:,:]))*h0[1,1,:,:]+h0[0,1,:,:]*(simu.X2-simu.C2)
    # h0_diag[1, 0, :, :] = cp.multiply(cp.sqrt(simu.X2[:,:]), cp.sqrt(simu.C2[:,:]))*h0[0,0,:,:]-cp.multiply(cp.sqrt(simu.X2[:,:]), cp.sqrt(simu.C2[:,:]))*h0[1,1,:,:]+h0[0,1,:,:]*(simu.X2-simu.C2)

    # print('BASIS CHANGE MATRIX')
    # print(simu.pol_basis_vector_kspace[:,:,0,0])
    # print(simu.pol_basis_vector_kspace[:,:,1,1])
    # print(simu.pol_basis_vector_kspace[:,:,100,100])
    # print(simu.pol_basis_vector_kspace[:,:,128,128])

    # print("h0")
    # print(h0[:,:,0,0])
    # print(h0[:,:,1,1])
    # print(h0[:,:,100,100])
    # print(h0[:,:,128,128])
    
    # print("h0_diag is ")
    # print(h0_diag[:,:,0,0])
    # print(h0_diag[:,:,1,1])
    # print(h0_diag[:,:,100,100])
    # print(h0_diag[:,:,128,128])
    
    
    # verifying new fields:
    
    
    # F_laser_r = simu.F_laser_r
    # F_laser_t = simu.F_laser_t
    # F_probe_r = simu.F_probe_r
    # F_probe_t = simu.F_probe_t


    # #pump
    # F_laser_phase = cp.angle(F_laser_r)
    # max_F = cp.amax(cp.abs(F_laser_r)**2)
    # F_laser_norm = cp.array(F_laser_r*255/max_F,dtype = cp.uint8)
    
    # plt.figure(num=1)
    # plt.pcolormesh(F_laser_norm.get())
    # plt.savefig(folder+"/intensity_new.png")
    
    # plt.figure(num=2)
    # plt.pcolormesh(F_laser_phase.get())
    # plt.savefig(folder+"/phase_new.png")
    
    
    
    
    # F_laser_phase_prev = cp.array(cp.angle(F_laser_prev))
    # max_F_prev = cp.amax(cp.absolute(F_laser_prev)**2)
    # F_laser_norm_prev = cp.array(F_laser_prev*255/max_F_prev, dtype = cp.uint8)
    
    # plt.figure(num=3)
    # plt.pcolormesh(F_laser_norm_prev.get())
    # plt.savefig(folder+"/intensity_prev.png")
    
    # plt.figure(num=4)
    # plt.pcolormesh(F_laser_phase_prev.get())
    # plt.savefig(folder+"/phase_prev.png")
    
    
    # plt.close("all")
    
    #probe
    # F_probe_phase = cp.angle(F_probe_r)
    # max_F = cp.amax(cp.abs(F_probe_r)**2)
    # F_probe_norm = cp.array(F_probe_r*255/max_F,dtype = cp.uint8)
    
    # plt.figure(num=1)
    # plt.pcolormesh(F_probe_norm.get())
    # plt.savefig(folder+"/intensity_new.png")
    
    # plt.figure(num=2)
    # plt.pcolormesh(F_probe_phase.get())
    # plt.savefig(folder+"/phase_new.png")
    
    
    
    
    # F_probe_phase_prev = cp.array(cp.angle(F_probe_prev))
    # max_F_prev = cp.amax(cp.absolute(F_probe_prev)**2)
    # F_probe_norm_prev = cp.array(F_probe_prev*255/max_F_prev, dtype = cp.uint8)
    
    # plt.figure(num=3)
    # plt.pcolormesh(F_probe_norm_prev.get())
    # plt.savefig(folder+"/intensity_prev.png")
    
    # plt.figure(num=4)
    # plt.pcolormesh(F_probe_phase_prev.get())
    # plt.savefig(folder+"/phase_prev.png")
    
    
    # plt.close("all")
       
    
    

    #---------------------------------------------------------------------------
    
    
    # mean_cav_t_x_y = np.einsum('xyt->txy', mean_cav_x_y_t)
    # mean_exc_t_x_y = np.einsum('xyt->txy', mean_exc_x_y_t)

    # mean_exc = mean_exc_t_x_y[:, :, :]
    # mean_cav = mean_cav_t_x_y[:, :, :]
    
 
    pol = mean_cav_t_x_y * np.sqrt(C02) - mean_exc_t_x_y * np.sqrt(X02)
    
    # pol_bog = pol[:-5,:,:] - pol[0,:,:]
    
    # field_outside = np.sum(cp.asnumpy(simu.R>30)*np.abs(pol_bog[:,:,:])*cp.asnumpy(simu.R<35), axis=(1,2))
    # #field_inside = np.sum(cp.asnumpy(simu.R<35)*np.abs(pol_bog[:,:,:]), axis=(1,2))
    # field_tot = np.sum(np.abs(pol_bog[:,:,:]*cp.asnumpy(simu.R<67)), axis=(1,2))
    # plt.figure()
    # plt.plot(field_outside/field_outside[50])
    # plt.xlabel("time")
    # plt.ylabel("field_ouside")
    # plt.savefig(folder + "/field_outside")
    # # plt.figure()
    # # plt.plot(field_inside/field_outside[50])
    # # plt.savefig(folder + "/field_inside")
    # plt.figure()
    # plt.plot(field_tot)
    #plt.savefig(folder + "/field_tot")
    
    # for k in range(pol_bog.shape[0]):
    #     plt.figure()
    #     plt.plot(mean_azim(np.abs(pol_bog[k,:,:]))[:70])
    #     plt.ylim(0,0.11)
    #     plt.savefig(folder + "/field_bog_r_t=%s.png"%(k))
    #     plt.figure()
    #     plt.plot(mean_azim(np.abs(pol_bog[k,:,:])*cp.asnumpy(simu.R))[:70])
    #     plt.ylim(0,3)
    #     plt.savefig(folder + "/normalised_field_r_t=%s.png"%(k))
    
    # import cv2

    # img_r = cv2.imread(folder+"/normalised_field_r_t=%s.png"%(0))
    # frameSize_r = (img_r.shape[1], img_r.shape[0])
    # out_r = cv2.VideoWriter(folder+"/normalised_field_r_t.mp4",cv2.VideoWriter_fourcc(*'mp4v'), 3, frameSize_r)

    # for k in range(pol_bog.shape[0]):
    #     img_r = cv2.imread(folder+"/normalised_field_r_t=%s.png"%(k))
    #     out_r.write(img_r)

    # out_r.release()
    
    #pol_u = mean_cav_t_x_y * np.sqrt(X02) + mean_exc_t_x_y * np.sqrt(C02)
    
    
    # # # # MOVIE
    
    size=(nmax_1, nmax_2)
    fps=15
    
    out_dens = cv2.VideoWriter(folder +"/dens_evolution.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
    out_phase = cv2.VideoWriter(folder +"/phase_evolution.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
    max_dens = np.amax(np.abs(pol)**2)
    for i in range(len(pol)):
        dens = np.array(np.abs(pol[i])**2*255/max_dens, dtype = np.uint8)
        phase = np.array(np.angle(pol[i]), dtype = np.uint8)
        out_dens.write(dens)
        out_phase.write(phase)
    out_dens.release()
    out_phase.release()
    
    
    
    
    
    #pol_bog_gif = 255*np.abs(pol_bog)/np.amax(np.abs(pol_bog))
    #pol_gif = 255*np.abs(pol)/np.amax(np.abs(pol))
    #F_gif = 255*np.abs(pol)/np.amax(np.abs(exc))
    #mean_cav_gif = 255*np.abs(mean_cav)/np.amax(mean_cav)
    # now = datetime.now()
    # dt_string = now.strftime("%Y%m%d-%H%M%S")
    # # # # string_name = ""
    # # # # np.save(folder + "/pol.npy", pol)
    # # # # np.save(folder + "/F_t.npy", F_t)
    #create_gif(folder + "/den.gif", pol_gif)
    #create_gif(folder + "/den_bog.gif", pol_bog_gif)
    #create_gif(folder +"/mean_cav.gif", mean_cav_gif)
    #create_gif(folder +"/F.gif", F_gif)
    
    # plt.figure()
    # plt.plot(F_t**2)
    # plt.xlabel("|F|2)")
    # plt.savefig(folder_DATA+"/F_t.png")
    
    
    # plt.figure()
    # plt.plot(F_t**2, np.mean(np.abs(pol), axis=(1,2), where = cp.asnumpy(simu.R)<40))
    # plt.xlabel("|F|²")
    # plt.ylabel("n")
    # plt.savefig(folder_DATA+"/bistab.pdf")
    
    # plt.figure()
    # plt.imshow(np.abs(cp.asnumpy(simu.F_laser)))
    # plt.colorbar()
    # plt.savefig(folder_DATA+"/pump_I.png")
    # plt.figure()
    # plt.imshow(np.pi+np.angle(cp.asnumpy(simu.F_laser)))
    # plt.colorbar()
    # plt.savefig(folder_DATA+"/pump_phase.png")
    
    # idx_cut = -50 #Time cut index

    # gn = g_LP*np.abs(pol[idx_cut,:,:])**2
    # cs = np.sqrt(h_bar*g_LP*np.abs(pol[idx_cut,:,:])**2/m_LP)
    # phase = np.angle(pol[idx_cut,:,:])

    # phase_unwrap_1d_x = np.unwrap(phase, axis=1)
    # phase_unwrap_1d_y = np.unwrap(phase, axis=0)

    # velocity_x = np.gradient(phase_unwrap_1d_x, long_2/nmax_2, axis=(-1, -2), edge_order=1)
    
    # vx = velocity_x[0]
    
    # velocity_y = np.gradient(phase_unwrap_1d_y, long_2/nmax_2, axis=(-1, -2), edge_order=1)
    # vy = velocity_y[1]

    # vx = h_bar*vx/m_LP
    # vy = h_bar*vy/m_LP
    
    # eff_detuning = detuning-0.5*m_LP*(vx**2+vy**2)/h_bar
    # vazim = (vy*np.cos(cp.asnumpy(THETA))-vx*np.sin(cp.asnumpy(THETA)))
    # vr = (vx*np.cos(cp.asnumpy(THETA))+vy*np.sin(cp.asnumpy(THETA)))

    # np.save(folder + "/vx.npy", vx)
    # np.save(folder + "/vy.npy", vy)
    # np.save(folder + "/vr.npy", vr)
    # np.save(folder + "/vazim.npy", vazim)
    # np.save(folder + "/gn.npy", gn)
    # np.save(folder + "/cs.npy", cs)
    # np.save(folder + "/eff_detuning.npy", eff_detuning)
    
    # plt.figure()
    # plt.imshow(np.abs(vazim))
    # plt.colorbar()
    # plt.savefig(folder+'/vtheta.png')
    # plt.figure()
    # plt.imshow(np.real(vr))
    # plt.colorbar()
    # plt.savefig(folder+'/vr.png')
    
    # vazim_cut = mean_azim(vazim)
    # vr_cut = mean_azim(vr)
    # cs_cut = mean_azim(cs)
    # vx_cut = mean_azim(vx)
    # vy_cut = mean_azim(vy)
    # gn_cut = mean_azim(gn)
    # eff_detuning_cut = mean_azim(eff_detuning)

    # fig, ax = plt.subplots(1,2)
    # divider0 = make_axes_locatable(ax[0])
    # divider1 = make_axes_locatable(ax[1])
    # cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    # cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    # im0 = ax[0].imshow(gn, cmap="gray")
    # ax[0].set_xlabel("x[µm]")
    # ax[0].set_ylabel("y[µm]")
    # im1 = ax[1].imshow(phase, cmap="twilight_shifted")
    # ax[1].set_xlabel("x[µm]")
    # ax[1].set_ylabel("y[µm]")
    # fig.colorbar(im0, cax=cax0, orientation='vertical')
    # fig.colorbar(im1, cax=cax1, orientation='vertical')
    # fig.tight_layout(pad=2.0)
    # plt.savefig(folder+"/field.png")

    # # plt.figure("pump at r=0")
    # # plt.plot(np.max(F_t_x_y[:,:,:],axis=(1,2)))
    # # plt.savefig(folder+"/pump.png")

    # plt.figure('gn_detuningeff')
    # plt.plot(eff_detuning_cut, label='detuning_eff')
    # plt.plot(gn_cut, label='gn')
    # plt.ylabel("[meV/h_bar]")
    # plt.xlabel("r[µm]")
    # plt.legend()
    # plt.savefig(folder+"/gn.png")

    # plt.figure('v_cs')
    # plt.plot(cs_cut, label ='cs')
    # plt.plot(vr_cut, label='vr')
    # plt.plot(vazim_cut, label='vazim')
    # plt.xlabel("r[µm]")
    # plt.ylabel("[µm/ps]")
    # plt.legend()
    # plt.savefig(folder+"/v_cs.png")
    # plt.close("all")
    #print("coeff is "+str((field_outside[200]-field_outside[50])/field_outside[50]))
    
    
    with open(folder+"/parameters.txt", "w") as f:
        f.write('\n'.join('{}: {}'.format(x[0],x[1]) for x in parameters))
#%%



h_bar = 0.654 # (meV*ps)
c = 2.9979*1e2 # (um/ps)
eV_to_J = 1.60218*1e-19
h_bar_SI = 1.05457182*1e-34

#Microcavity parameters
rabi = 5.07/2/h_bar # (meV/h_bar) linear coupling (Rabi split)
g0 = (1e-2) /h_bar  # (frequency/density) (meV/hbar)/(1/um^2) nonlinear coupling constant 
gamma_exc, gamma_ph = 0.07 /h_bar, 0.07 /h_bar # (meV/h_bar) exc and ph linewidth 1microeV 69microeV
omega_exc = 1484.44 /h_bar # (meV/h_bar) exciton energy measured from the cavity energy #-0.5
omega_cav = 1482.76 /h_bar # (meV/h_bar) cavity energy at k=0 
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
t_obs = 0 # (ps) time from which the observation starts
t_noise = 1e9
t_probe = 1e9
t_stationary = 1750
t_max = 1800 # (ps) final time of evolution
dt_frame = 1/(0.1) #cst/delta_E avec delta_E la résolution en énergie en meV
n_frame = int((t_max-t_obs)/dt_frame)+1
print("dt_frame is %s"%(dt_frame))
print("n_frame is %s"%(n_frame))

nmax = 256**1

nmax_1, nmax_2 = nmax, nmax
long_1, long_2 = 256, 256

if (long_1/nmax)**2<g0/gamma_ph:
    print("WARNING: TWA NOT VALID")


F = 1
F_probe = 1
corr = 0.0 #0.35
detuning = 0.17/h_bar
noise = 0
omega_probe=0

pump_radius = 60


parameters=[('h_bar',h_bar), ('h_bar_SI', h_bar_SI), ('c', c), ('eV_to_J', eV_to_J), ('rabi (div by 2hbar)', rabi*2*h_bar), ('g0 (div by hbar)', g0*h_bar), ('gamma_exc (div by hbar)', gamma_exc*h_bar), 
            ('gamma_ph (div by hbar)', gamma_ph*h_bar), ('omega_exc (div by hbar)', omega_exc*h_bar), ('omega_cav (div by hbar)', omega_cav*h_bar), ('n_cav', n_cav), ('k_z', k_z), ('t_min', t_min), ('t_obs', t_obs), 
            ('t_noise', t_noise), ('t_probe', t_probe), ('t_stationary', t_stationary), ('t_max', t_max), ('dt_frame', dt_frame), ('nmax_1', nmax_1), ('nmax_2', nmax_2), ('long_1', long_1), ('long_2', long_2), ('F', F), ('F_probe', F_probe),
            ('corr', corr), ('detuning (div by hbar)', detuning*h_bar), ('noise', noise), ('omega_probe', omega_probe), ('pump_radius', pump_radius)] 

simu = ggpe(nmax_1, nmax_2, long_1, long_2, t_max, t_stationary, t_obs, t_probe, t_noise, dt_frame, gamma_exc, 
        gamma_ph, noise, g0, detuning, omega_probe, omega_exc, omega_cav, rabi, k_z)


m_LP = simu.m_LP
R = simu.R
THETA = simu.THETA
X = simu.X
Y = simu.Y



#Build laser+probe field

#Pump
tophat(simu.F_laser_r, F, R, radius = pump_radius)
#gaussian(simu.F_laser_r, F, R, radius = pump_radius)
#vortex_beam(F_laser_r, F, R, THETA, waist = 75, inner_waist = 22, C = 15)
#shear_layer(F_laser_r, F, X, Y, kx = 1)
#plane_wave(F_laser_r, F, X, kx = 0.5)
to_turning_point(simu.F_laser_t, simu.time, t_up = 400, t_down = 400)
#bistab_cycle(simu.F_laser_t, simu.time, simu.t_max)
#turn_on_pump(simu.F_laser_t, simu.time, t_up = 200)

#Probe
ring(simu.F_probe_r, F_probe, R, radius = pump_radius, delta_radius = 20)
#radial_expo(F_probe_r, F_probe, R, THETA, m_probe = 10, p_probe = 5)
tempo_probe(simu.F_probe_t, omega_probe, t_probe, simu.time)


folder_DATA =  "/home/stagios/Oscar/LEON/DATA/Polaritons/2024_ManasOscar/first_runs"
#string_name = "_noise=%s_dx%s_dt=%s"%(str(round(noise,5)),str(round(long_1/nmax_1,5)),str(round(dt_frame,5)))
string_name="_comparing_fields"
#string_name = "_k=%s_detuning=%s_F=%s"%(str(round(kx,3)),str(round(detuning,3)),str(round(F,3)))

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
    
    
    
simu.evolution()

# save_data_giant_vortex(folder_DATA, F_laser_r, F_laser_prev, F_probe_r, F_probe_prev)
save_data_giant_vortex(folder_DATA, parameters)

