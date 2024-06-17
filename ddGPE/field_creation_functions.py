import cupy as cp
import cupyx.scipy.signal as signal

#Spatial density profiles:

def tophat(
    F_laser_r: cp.ndarray, 
    R: cp.ndarray, 
    radius: float = 75
):
    """A function to create a tophat spatial mode for the laser pump field

    Args:
        F_laser_r (cp.ndarray): self.F_laser_r as defined in class ggpe, cp.ones((n_max1, n_max2)cp.complex64)
        R (cp.ndarray): array of the distance from the center of the grid
        radius (float, optional): radius of the beam. Defaults to 75.
    """
    F_laser_r[..., R > radius] = 0
    profile = "Tophat, radius = " + str(radius) + " ; "
    return profile
    

def gaussian(
    F_laser_r: cp.ndarray, 
    R: cp.ndarray, 
    radius=75
):
    """A function to create a gaussian spatial mode for the laser pump field

    Args:
        F_laser_r (cp.ndarray): self.F_laser_r as defined in class ggpe, cp.ones((n_max1, n_max2)cp.complex64)
        R (cp.ndarray): array of the distance from the center of the grid
        radius (int, optional): radius (=sqrt(2)*std) of the beam. Defaults to 75.
    """
    F_laser_r[..., :, :] = F_laser_r[..., :, :] * cp.exp(-R[:, :] ** 2 / radius ** 2) #identifying with Gaussian distribution, radius is sqrt(2)*std_deviation
    profile = "Gaussian, radius = " + str(radius) + " ; "
    return profile

def ring(
    F_probe_r: cp.ndarray, 
    R: cp.ndarray,
    radius: float, 
    delta_radius: float
):
    """A function to create a ring spatial mode for the laser probe field

    Args:
        F_probe_r (cp.ndarray): self.F_probe_r as defined in class ggpe, cp.ones((n_max1, n_max2)cp.complex64)
        R (cp.ndarray): array of the distance from the center of the grid
        radius (float): radius of the ring
        delta_radius (float): total width of the ring 
    """
    F_probe_r[..., R > radius + delta_radius / 2] = 0
    F_probe_r[..., R < radius - delta_radius / 2] = 0
    profile = "Ring, radius = " + str(radius) + ", delta_radius = " + str(delta_radius) + " ; "
    return profile
    

#Spatial phase profiles:

def plane_wave(
    F_laser_r: cp.ndarray, 
    XX: cp.ndarray, 
    kx=0.5
):
    """A function to create a plane wave spatial mode for the laser pump field

    Args:
        F_laser_r (cp.ndarray): self.F_laser_r as defined in class ggpe, cp.ones((n_max1, n_max2)cp.complex64)
        XX (cp.ndarray): array of dimensions (n_max1, n_max2) with the x coordinate of each point
        kx (float, optional): magnitude of the wavevector in the x direction. Defaults to 0.5.
    """
    phase = cp.zeros(XX.shape)
    phase = kx * XX[:, :]
    F_laser_r[..., :, :] = F_laser_r[..., :, :] * cp.exp(1j * phase[:, :])
    profile = "Plane wave, kx = " + str(kx) + " ; "
    return profile

def shear_layer(
    F_laser_r: cp.ndarray, 
    XX: cp.ndarray, 
    YY: cp.ndarray, 
    kx: float=1
):
    """A function to create a shear layer spatial mode for the laser pump field

    Args:
        F_laser_r (cp.ndarray): self.F_laser_r as defined in class ggpe, cp.ones((n_max1, n_max2)cp.complex64)
        XX (cp.ndarray): array of dimensions (n_max1, n_max2) with the x coordinate of each point
        YY (cp.ndarray): array of dimensions (n_max1, n_max2) with the y coordinate of each point
        kx (float, optional): magnitude of the wavevector in the x direction. Defaults to 1.
    """
    phase = cp.zeros(XX.shape)
    phase = kx * XX[:, :]
    phase[YY > 0] =- phase[YY > 0]
    F_laser_r[..., :, :] = F_laser_r[..., :, :] * cp.exp(1j * phase[:, :])
    profile = "Shear layer, kx = " + str(kx) + " ; "
    return profile

def radial_expo(
    F_probe_r: cp.ndarray, 
    R: cp.ndarray, 
    THETA: cp.ndarray, 
    m, 
    p
):
    """A function to create a spatial mode with both radial and angular phase velocity for the laser probe field

    Args:
        F_probe_r (cp.ndarray): self.F_probe_r as defined in class ggpe, cp.ones((n_max1, n_max2)cp.complex64)
        R (cp.ndarray): array of the distance from the center of the grid
        THETA (cp.ndarray): array of the angle with respect to the positive x axis
        m (_type_): angular phase velocity
        p (_type_): radial phase velocity
    """
    F_probe_r[..., :, :] = F_probe_r[..., :, :] * cp.exp(1j * p * R[:, :]) * cp.exp(1j * m * THETA[:, :])
    profile = "Radial_expo, p = " + str(p) + ", m = " + str(m) + " ; "
    return profile


#Hybrid (density+phase) spactial profiles:

def vortex_beam(
    F_laser_r: cp.ndarray, 
    R: cp.ndarray, 
    THETA: cp.ndarray, 
    waist=75, 
    inner_waist: float = 22, 
    C: int =15
):
    """A function to create a vortex_beam spatial mode for the laser pump field

    Args:
        F_laser_r (cp.ndarray): self.F_laser_r as defined in class ggpe, cp.ones((n_max1, n_max2)cp.complex64)
        R (cp.ndarray): array of the distance from the center of the grid
        THETA (cp.ndarray): array of the angle with respect to the positive x axis
        waist (float, optional): _description_. Defaults to 75.
        inner_waist (float, optional): radius of the inner waist. Defaults to 22.
        C (int, optional): vorticity (right term???) of the vortex. Defaults to 15.
    """
    F_laser_r[..., :, :] = F_laser_r[..., :, :] * cp.exp(1j * C * THETA[:, :]) * cp.tanh(R[:, :] / inner_waist) ** C
    profile = "Vortex beam, C = " + str(C) + ", inner_waist = " + str(inner_waist) + " ; "
    return profile

def effective_1d(
    F_laser_r: cp.ndarray, 
    XX: cp.ndarray, 
    YY: cp.ndarray, 
    half_width: float = 128, 
    beg_pump: float = -128, 
    end_pump: float = 0, 
    end_support: float = 128, 
    ratio: float = 0.5, 
    k_pump: float = 0.5, 
    along: str = "y"
):
    """A function to create an effective 1D spatial mode for the laser pump field

    Args:
        F_laser_r (cp.ndarray): self.F_laser_r as defined in class ggpe, cp.ones((n_max1, n_max2)cp.complex64)
        XX (cp.ndarray): array of dimensions (n_max1, n_max2) with the x coordinate of each point
        YY (cp.ndarray): array of dimensions (n_max1, n_max2) with the y coordinate of each point
        half_width (float, optional): half width of the pump in the perpendicular direction. Defaults to 128.
        beg_pump (float, optional): position at which the pump field starts. Defaults to -128.
        end_pump (float, optional): position at which the pump field ends. Defaults to 0.
        end_support (float, optional): position aat which the support field ends. Defaults to 128.
        ratio (float, optional): ratio between the pump and support fields intensities. Defaults to 0.5.
        k_pump (float, optional): magnitude of the wavevector in the given direction. Defaults to 1.
        along (str, optional): gives the direction along which the effective 1D profile is created. Defaults to "y".
    """
    if along == "y":
        phase = cp.zeros(XX.shape)
        phase = k_pump * YY[:, :]
        F_laser_r[XX > half_width] = 0
        F_laser_r[XX < -1 * half_width] = 0
        F_laser_r[YY < beg_pump] = 0
        F_laser_r[YY > end_pump] = ratio
        F_laser_r[YY > end_support] = 0
        F_laser_r[..., :, :] = F_laser_r[..., :, :] * cp.exp(1j * phase[:, :])
        profile = "Effective 1D along" + along + ", half_width = " + str(half_width) + ", beg_pump = " + str(beg_pump) + ", end_pump = " + str(end_pump) + ", end_support = " + str(end_support) + ", ratio = " + str(ratio) + ", k_pump = " + str(k_pump) + " ; "
        return profile
    if along == "x":
        phase = cp.zeros(XX.shape)
        phase = k_pump * XX[:, :]
        F_laser_r[YY > half_width] = 0
        F_laser_r[YY < -1 * half_width] = 0
        F_laser_r[XX < beg_pump] = 0
        F_laser_r[XX > end_pump] = ratio
        F_laser_r[XX > end_support] = 0
        F_laser_r[..., :, :] = F_laser_r[..., :, :] * cp.exp(1j * phase[:, :])
        profile = "Effective 1D along " + along + ", half_width = " + str(half_width) + ", beg_pump = " + str(beg_pump) + ", end_pump = " + str(end_pump) + ", end_support = " + str(end_support) + ", ratio = " + str(ratio) + " ; "
        return profile
    


#Temporal intensity profiles:

def to_turning_point(
    F_laser_t: cp.ndarray, 
    time: cp.ndarray, 
    t_up = 400, 
    t_down = 400
):
    """A function to create the to_turning_point temporal evolution of the intensity of the pump field

    Args:
        F_laser_t (cp.ndarray): self.F_pump_t as defined in class ggpe, cp.ones((int(self.t_max//self.dt)), dtype=cp.complex64)
        time (cp.ndarray): array with the value of the time at each discretized step
        t_up (float, optional): time at which we reach the maximum intensity (= 3*F). Defaults to 400.
        t_down (float, optional): time after t_up at which we approach the working point intensity (=F). Defaults to 400.
    """
    F_laser_t[time < t_up] = 3 * cp.exp(-((time[time < t_up] - t_up) / (t_up / 2)) ** 2)
    F_laser_t[time >= t_up] = (1 + 2 * cp.exp(-((time[time >= t_up] - t_up) / t_down) ** 2))
    profile = "Time profile: to turning point, t_up = " + str(t_up) + ", t_down = " + str(t_down) + " "
    return profile


def bistab_cycle(
    F_laser_t: cp.ndarray, 
    time: cp.ndarray, 
    t_max
):
    """A function to create the bistab_cycle temporal evolution of the intensity of the pump field

    Args:
        F_laser_t (cp.ndarray): self.F_pump_t as defined in class ggpe, cp.ones((int(self.t_max//self.dt)), dtype=cp.complex64)
        time (cp.ndarray): array with the value of the time at each discretized step
        t_max (_type_): maximum time of the simulation
    """
    F_laser_t[:] = 4 * cp.exp(-((time[:] - t_max // 2) / (t_max // 4)) ** 2)
    profile = "Time profile: bistab_cycle, t_max = " + str(t_max) + " "
    return profile


def turn_on(
    F_laser_t: cp.ndarray, 
    time: cp.ndarray, 
    t_up=200
):
    """A function to create the turn_on_pump temporal evolution of the intensity of the pump field

    Args:
        F_laser_t (cp.ndarray): self.F_pump_t as defined in class ggpe, cp.ones((int(self.t_max//self.dt)), dtype=cp.complex64)
        time (cp.ndarray):  array with the value of the time at each discretized step
        t_up (int, optional): time taken to reach the maximum intensity (=F). Defaults to 200.
    """
    F_laser_t[time < t_up] = cp.exp(-1 * (time[time < t_up] - t_up)**2 / (t_up / 2))
    F_laser_t[time >= t_up] = 1
    profile = "Time profile: turn_on_pump, t_up = " + str(t_up) + " "
    return profile

#Temporal phase profiles:

def tempo_probe(
    F_probe_t: cp.ndarray, 
    omega_probe: float, 
    t_probe, 
    time: cp.ndarray
):
    """A function to create the spatial evolution of the probe field

    Args:
        F_probe_t (cp.ndarray): self.F_probe_t as defined in class ggpe, cp.ones((int(self.t_max//self.dt)), dtype=cp.complex64)
        omega_probe (float): detuning of the probe with respect to the pumping field
        t_probe (float): time at which we turn on the probe
        time (cp.ndarray): array with the value of the time at each discretized step
    """
    F_probe_t[..., :] = cp.exp(-1j * (time - t_probe) * omega_probe)
    F_probe_t[..., time < t_probe] = 0
    profile = "Time profile: tempo_probe, omega_probe = " + str(omega_probe) + " "
    return profile

def linear_ramp(
    F_probe_t: cp.ndarray,  
    omega_start: float,
    omega_end: float,
    t_probe: float, 
    t_ramp: float,
    time: cp.ndarray    
):
    omega_t = omega_start + (omega_end - omega_start) * (time - t_probe) / t_ramp
    F_probe_t[..., :] = cp.exp(-1j * omega_t)
    F_probe_t[..., time < t_probe] = 0
    profile = "Time profile: linear_ramp, omega_start = " + str(omega_start) + ", omega_end = " + str(omega_end) + ", t_probe = " + str(t_probe) + ", t_ramp = " + str(t_ramp) + " "
    return profile

def step_ramp(
    F_probe_t: cp.ndarray, 
    omega_start: float,
    omega_end: float,
    omega_resol: float,
    t_probe: float, 
    t_ramp: float,	
    time: cp.ndarray
):
    omega_step = (omega_end - omega_start) / omega_resol
    t_step = t_ramp / omega_resol
    omega_t = omega_start + omega_step * (time - t_probe) // t_step
    F_probe_t[..., :] = cp.exp(-1j * omega_t)
    F_probe_t[..., time < t_probe] = 0
    profile = "Time profile: step_ramp, omega_start = " + str(omega_start) + ", omega_end = " + str(omega_end) + ", omega_resol = " + str(omega_resol) + ", t_probe = " + str(t_probe) + ", t_ramp = " + str(t_ramp) + " "
    return profile
    
    

#Potential profiles:

def gaussian_barrier(
    potential: cp.ndarray, 
    XX: cp.ndarray, 
    YY: cp.ndarray, 
    h: float, 
    sigma_y: float, 
    sigma_x:float, 
    off_y: float
):
    """A fucntion to create a gaussian barrier in the potential along the y direction

    Args:
        potential (cp.ndarray): self.potential as defined in class ggpe, cp.zeros((self.Nx, self.Ny), dtype=np.complex64)
        XX (cp.ndarray): array of dimensions (n_max1, n_max2) with the x coordinate of each point
        YY (cp.ndarray): array of dimensions (n_max1, n_max2) with the y coordinate of each point
        h (float): height of the barrier (will be normalised by hbar)
        sigma_y (float): std along the y direction
        sigma_x (float): width of the barrier along the x direction
        off_y (float): position of the barrier along y 

    """
    height_defect = h / 0.654 # (height/h_bar)
    potential += height_defect * cp.exp(-((YY - off_y) / sigma_y) ** 2)
    potential[XX ** 2  > sigma_x ** 2] = 0
    toconv = height_defect * cp.exp(-(XX ** 2 + YY ** 2) / (0.01 * sigma_y) ** 2)
    potential = signal.convolve2d(toconv, potential, boundary='symm', mode='same')
    profile = "Gaussian barrier, h = " + str(h) + ", sigma_y = " + str(sigma_y) + ", sigma_x = " + str(sigma_x) + ", off_y = " + str(off_y) + " ; "
    return profile

def tophat_barrier(
    potential: cp.ndarray, 
    XX: cp.ndarray, 
    YY: cp.ndarray, 
    h: float, 
    sigma_y: float, 
    sigma_x:float, 
    off_y: float
):
    """A fucntion to create a gaussian barrier in the potential along the y direction

    Args:
        potential (cp.ndarray): self.potential as defined in class ggpe, cp.zeros((self.Nx, self.Ny), dtype=np.complex64)
        XX (cp.ndarray): array of dimensions (n_max1, n_max2) with the x coordinate of each point
        YY (cp.ndarray): array of dimensions (n_max1, n_max2) with the y coordinate of each point
        h (float): height of the barrier (will be normalised by hbar)
        sigma_y (float): width of the barrier along the y direction
        sigma_x (float): width of the barrier along the x direction
        off_y (float): position of the barrier along y 

    """
    height_defect = h / 0.654 # (height/h_bar)
    potential[XX ** 2  < sigma_x ** 2] += height_defect
    potential[(YY-off_y) ** 2 < sigma_y ** 2] += height_defect
    profile = "Top hat defect, h = " + str(h) + ", sigma_y = " + str(sigma_y) + ", sigma_x = " + str(sigma_x) + ", off_y = " + str(off_y) + " ; "
    return profile


#Functions for simulations in parallel:
# def tophat_parallel_plane_wave( FOR SOME REASON THIS DOES NOT WORK
#     F_probe_r: cp.ndarray,
#     F_probe_t: cp.ndarray,
#     Kx_scan: cp.ndarray,
#     omega_scan: cp.ndarray,
#     t_probe: float,
#     time: cp.ndarray,
#     radius: float,
#     XX: cp.ndarray,
#     R: cp.ndarray
# ):
    
#     F_probe_r = cp.ones((Kx_scan.shape[0], 1)+F_probe_r.shape, dtype=cp.complex64)
#     print(F_probe_r.shape)
#     F_probe_t = cp.ones((1, omega_scan.shape[0], 1, 1, time.shape[0]))
#     print(F_probe_t.shape)
#     for i in range(Kx_scan.shape[0]):
#         phase = cp.zeros(XX.shape)
#         phase = Kx_scan[i] * XX[:, :]
#         F_probe_r[i, :, :, :] = F_probe_r[i, :, :, :] * cp.exp(1j * phase[:, :])
#         F_probe_r[i, :, R > radius] = 0
#     for j in range(omega_scan.shape[0]):
#         F_probe_t[:, j, :, :, :] = cp.exp(-1j * (time - t_probe) * omega_scan[j])
#         F_probe_t[:, j, :, :, time < t_probe] = 0
#     probe_spatial_profile = "Tophat: radius ="+str(radius)+"; Parallelized plane wave, Kx_scan: [start: " + str(Kx_scan[0]) + ", end: " + str(Kx_scan[-1]) + "] ; "
#     probe_temporal_profile = "Parallelized plane wave, omega_scan [start: " + str(omega_scan[0]) + ", end: " + str(omega_scan[-1]) + "]; "
    
#     return probe_spatial_profile, probe_temporal_profile