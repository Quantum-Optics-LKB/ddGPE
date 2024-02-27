import numpy as np
import scipy
import cupy as cp
from cupy.random import rand, randn
import cupyx.scipy.fftpack as fftpack
import numba
from tqdm import tqdm

def build_fft_plan(A:cp.ndarray) -> list:
        """Builds the FFT plan objects for propagation
        Args:
            A (np.ndarray): Array to transform.
        Returns:
            list: A list containing the FFT plans
        """
        plan_fft = fftpack.get_fft_plan(
            A, shape=(A.shape[-2], A.shape[-1]), axes=(-2,-1), value_type='C2C')
        return plan_fft

@cp.fuse(kernel_name="non_linearity")
def non_linearity(phi1: cp.ndarray, dt: float, g0: float) -> None:
    """A fused kernel to apply non linearity term

    Args:
        phi (cp.ndarray): The field in ph,exc basis
        dt (float): Propagation step in ps
        g0 (float): Interactions
    """
    phi1 *= cp.exp(-1j*dt*g0*cp.abs(phi1)**2)

    
@cp.fuse(kernel_name="probe_excitation")
def laser_excitation(phi2: cp.ndarray, F_laser_r: cp.ndarray, F_laser_t: float, F_probe_r: cp.ndarray, F_probe_t: cp.complex64, dt: float) -> None:
    """A fused kernel to apply the pump and probe excitation terms to the photon field

    Args:
        phi2 (cp.ndarray): Photonic field in ph,exc basis
        F_laser_r (cp.ndarray): Spatial mode of the laser pump field
        F_laser_t (float): Temporal dependency of the laser pump field at corresponding time
        F_probe_r (cp.ndarray): Spatial mode of the probe field
        F_probe_t (cp.complex64): Temporal dependency of the probe field at corresponding time
        dt (float): Propagation step in ps
    """
    phi2 -= F_laser_r*F_laser_t*dt*1j
    phi2 -= F_probe_r*F_probe_t*dt*1j


@cp.fuse(kernel_name="single_particle_pot")
def single_particle_pot(phi2: cp.ndarray, dt: float, v_gamma: float) -> None:
    """A fused kernel to apply single_particle_pot term

    Args:
        phi (cp.ndarray): The field in ph,exc basis
        dt (float): Propagation step in ps
        v_gamma (float): Loss at the edges of the grid
    """
    phi2 *= cp.exp(-dt*0.5*v_gamma) #*cp.exp(-1j*dt*v_single)

@cp.fuse(kernel_name="add_noise")
def add_noise(phi1: cp.ndarray, phi2: cp.ndarray, rand1: cp.ndarray, rand2: cp.ndarray, v_gamma: cp.ndarray, gamma_exc: float, gamma_ph: float,
            dv: float, noise_exc: float, noise_ph: float) -> None:
    """A fused kernel to add gaussian noise

    Args:
        phi (cp.ndarray): The field in ph,exc basis
        dt (float): Propagation step in ps
        v_gamma (float): Loss at the edges of the grid
    """
    phi1 += noise_exc*cp.sqrt(gamma_exc/4*dv)*rand1
    phi2 += noise_ph*cp.sqrt((v_gamma+gamma_ph)/4*dv)*rand2

@cp.fuse(kernel_name="linear_step")
def linear_step(phi1: cp.ndarray, phi2: cp.ndarray, phi_up: cp.ndarray, phi_lp: cp.ndarray, propagator_diag: cp.ndarray, hopfield_coefs: cp.ndarray) -> None:
    """A fused kernel to apply the linear step in the diagonal polariton basis

    Args:
        phi1 (cp.ndarray): exciton field in ph,exc basis (in k-space)
        phi2 (cp.ndarray): photon field in ph,exc basis (in k-space)
        phi_up (cp.ndarray): array to stock UP component of the field in the UP-LP basis
        phi_lp (cp.ndarray): array to stock LP component of the field in the UP-LP basis
        propagator (cp.ndarray): Propagator in the diagonal polariton basis
        hopfield_coefs (cp.ndarray): array with the Hopfield coefficients
    """
    
    cp.multiply(phi1, -1*hopfield_coefs[0,:,:], phi_lp)
    phi_lp += cp.multiply(phi2, hopfield_coefs[1,:,:])
    cp.multiply(phi1, hopfield_coefs[1,:,:], phi_up) 
    phi_up += cp.multiply(phi2, hopfield_coefs[0,:,:])

    cp.multiply(phi_lp, propagator_diag[0, :, :], phi_lp)
    cp.multiply(phi_up, propagator_diag[1, :, :], phi_up)

    cp.multiply(phi_lp, -1*hopfield_coefs[0,:,:], phi1) 
    phi1 += cp.multiply(phi_up, hopfield_coefs[1,:,:])
    cp.multiply(phi_lp, hopfield_coefs[1,:,:], phi2) 
    phi2 += cp.multiply(phi_up, hopfield_coefs[0,:,:])
           
            
def tophat(F_laser_r: cp.ndarray, F: float, R: cp.ndarray, radius: float = 75):
    """A function to create a tophat spatial mode for the laser pump field

    Args:
        F_laser_r (cp.ndarray): self.F_laser_r as defined in class ggpe, cp.ones((n_max1, n_max2)cp.complex64)
        F (float): Intensity of the laser pump field
        R (cp.ndarray): array of the distance from the center of the grid
        radius (float, optional): radius of the beam. Defaults to 75.
    """
    F_laser_r[R > radius] = 0
    F_laser_r[:,:] = F_laser_r[:,:] * F

def gaussian(F_laser_r: cp.ndarray, F: float, R: cp.ndarray, radius=75):
    """A function to create a gaussian spatial mode for the laser pump field

    Args:
        F_laser_r (cp.ndarray): self.F_laser_r as defined in class ggpe, cp.ones((n_max1, n_max2)cp.complex64)
        F (float): Intensity of the laser pump field
        R (cp.ndarray): array of the distance from the center of the grid
        radius (int, optional): radius (=sqrt(2)*std) of the beam. Defaults to 75.
    """
    F_laser_r[:,:] = F_laser_r[:,:] * F * cp.exp(-R[:,:]**2/radius**2) #O: identifying with Gaussian distribution, radius is sqrt(2)*std_deviation

def vortex_beam(F_laser_r: cp.ndarray, F: float, R: cp.ndarray, THETA: cp.ndarray, waist=75, inner_waist: float = 22, C: int =15):
    """A function to create a vortex_beam spatial mode for the laser pump field

    Args:
        F_laser_r (cp.ndarray): self.F_laser_r as defined in class ggpe, cp.ones((n_max1, n_max2)cp.complex64)
        F (float): Intensity of the laser pump field
        R (cp.ndarray): array of the distance from the center of the grid
        THETA (cp.ndarray): array of the angle with respect to the positive x axis
        waist (float, optional): _description_. Defaults to 75.
        inner_waist (float, optional): radius of the inner waist. Defaults to 22.
        C (int, optional): vorticity (right term???) of the vortex. Defaults to 15.
    """
    F_laser_r[:,:] = F_laser_r[:,:] * F * cp.exp(1j*C*THETA[:,:])*cp.tanh(R[:,:]/inner_waist)**C

def shear_layer(F_laser_r: cp.ndarray, F: float, X: cp.ndarray, Y: cp.ndarray, kx: float=1):
    """A function to create a shear layer spatial mode for the laser pump field

    Args:
        F_laser_r (cp.ndarray): self.F_laser_r as defined in class ggpe, cp.ones((n_max1, n_max2)cp.complex64)
        F (float): Intensity of the laser pump field
        X (cp.ndarray): array of dimensions (n_max1, n_max2) with the x coordinate of each point
        Y (cp.ndarray): array of dimensions (n_max1, n_max2) with the y coordinate of each point
        kx (float, optional): magnitude of the wavevector in the x direction. Defaults to 1.
    """
    phase = cp.zeros(F_laser_r.shape)
    phase = kx*X[:,:]
    phase[Y>0]=-phase[Y>0]
    F_laser_r[:,:] = F_laser_r[:,:]*F*cp.exp(1j*phase[:,:])

def plane_wave(F_laser_r: cp.ndarray, F: float, X: cp.ndarray, kx=0.5):
    """A function to create a plane wave spatial mode for the laser pump field

    Args:
        F_laser_r (cp.ndarray): self.F_laser_r as defined in class ggpe, cp.ones((n_max1, n_max2)cp.complex64)
        F (float): Intensity of the laser pump field
        X (cp.ndarray): array of dimensions (n_max1, n_max2) with the x coordinate of each point
        kx (float, optional): magnitude of the wavevector in the x direction. Defaults to 0.5.
    """
    phase = cp.zeros(X.shape)
    phase = kx*X[:,:]
    F_laser_r[:,:] = F_laser_r[:,:] * cp.exp(1j*phase[:,:])

def ring(F_probe_r: cp.ndarray, F_probe: float, R: cp.ndarray, radius: float, delta_radius: float):
    """A function to create a ring spatial mode for the laser probe field

    Args:
        F_probe_r (cp.ndarray): self.F_probe_r as defined in class ggpe, cp.ones((n_max1, n_max2)cp.complex64)
        F_probe (float): Intensity of the laser probe field
        R (cp.ndarray): array of the distance from the center of the grid
        radius (float): radius of the ring
        delta_radius (float): total width of the ring 
    """
    F_probe_r[R>radius+delta_radius/2] = 0
    F_probe_r[R<radius-delta_radius/2] = 0
    F_probe_r[:,:] = F_probe_r[:,:] * F_probe

def radial_expo(F_probe_r: cp.ndarray, F_probe: float, R: cp.ndarray, THETA: cp.ndarray, m_probe, p_probe):
    """A function to create a spatial mode with both radial and angular phase velocity for the laser probe field

    Args:
        F_probe_r (cp.ndarray): self.F_probe_r as defined in class ggpe, cp.ones((n_max1, n_max2)cp.complex64)
        F_probe (float): Intensity of the laser probe field
        R (cp.ndarray): array of the distance from the center of the grid
        THETA (cp.ndarray): array of the angle with respect to the positive x axis
        m_probe (_type_): angular phase velocity
        p_probe (_type_): radial phase velocity
    """
    F_probe_r[:,:] = F_probe_r[:,:] * F_probe * cp.exp(1j*p_probe*R[:,:]) * cp.exp(1j*m_probe*THETA[:,:])


def tempo_probe(F_probe_t: cp.ndarray, omega_probe: float, t_probe, time: cp.ndarray):
    """A function to create the spatial evolution of the probe field

    Args:
        F_probe_t (cp.ndarray): self.F_probe_t as defined in class ggpe, cp.ones((int(self.t_max//self.dt)), dtype=cp.complex64)
        omega_probe (float): detuning of the probe with respect to the pumping field
        t_probe (float): time at which we turn on the probe
        time (cp.ndarray): array with the value of the time at each discretized step
    """
    F_probe_t[time<t_probe] = 0
    F_probe_t[time>=t_probe] = cp.exp(-1j*(time[time>=t_probe]-t_probe)*omega_probe)


def to_turning_point(F_laser_t: cp.ndarray, time: cp.ndarray, t_up = 400, t_down = 400):
    """A function to create the to_turning_point temporal evolution of the intensity of the pump field

    Args:
        F_laser_t (cp.ndarray): self.F_pump_t as defined in class ggpe, cp.ones((int(self.t_max//self.dt)), dtype=cp.complex64)
        time (cp.ndarray): array with the value of the time at each discretized step
        t_up (float, optional): time at which we reach the maximum intensity (= 3*F). Defaults to 400.
        t_down (float, optional): time after t_up at which we approach the working point intensity (=F). Defaults to 400.
    """
    F_laser_t[time<t_up] = 3*cp.exp(-((time[time<t_up]-t_up)/(t_up/2))**2)
    F_laser_t[time>=t_up] = (1 + 2*cp.exp(-((time[time>=t_up]-t_up)/t_down)**2))


def bistab_cycle(F_laser_t: cp.ndarray, time: cp.ndarray, t_max):
    """A function to create the bistab_cycle temporal evolution of the intensity of the pump field

    Args:
        F_laser_t (cp.ndarray): self.F_pump_t as defined in class ggpe, cp.ones((int(self.t_max//self.dt)), dtype=cp.complex64)
        time (cp.ndarray): array with the value of the time at each discretized step
        t_max (_type_): maximum time of the simulation
    """
    F_laser_t[:] = 4*cp.exp(-((time[:]-t_max//2)/(t_max//4))**2)


def turn_on_pump(F_laser_t: cp.ndarray, time: cp.ndarray, t_up=200):
    """A function to create the turn_on_pump temporal evolution of the intensity of the pump field

    Args:
        F_laser_t (cp.ndarray): self.F_pump_t as defined in class ggpe, cp.ones((int(self.t_max//self.dt)), dtype=cp.complex64)
        time (cp.ndarray):  array with the value of the time at each discretized step
        t_up (int, optional): time taken to reach the maximum intensity (=F). Defaults to 200.
    """
    F_laser_t[time<t_up] = cp.exp(((time[time<t_up]-t_up)/(t_up/2))**2)
    F_laser_t[time>=t_up] = 1


class ggpe():

    def __init__(self, nmax_1: int, nmax_2: int, long_1: int, long_2: int, t_max: int, t_stationary: int, t_obs: int, t_probe: int, t_noise: int, dt_frame: float,
        gamma_exc: float, gamma_ph: float, noise: float, g0: float, detuning: float, omega_probe: float, omega_exc: float, omega_cav: float, rabi: float, k_z: float) -> None:

        self.nmax_1 = nmax_1
        self.nmax_2 = nmax_2
        self.long_1 = long_1
        self.long_2 = long_2
        self.dv = long_1*long_2/nmax_1/nmax_2

        self.t_obs = t_obs
        self.t_noise = t_noise
        self.t_stationary = t_stationary
        self.t_probe = t_probe
        self.t_max = t_max
        self.dt_frame = dt_frame
        self.n_frame = int((self.t_max-self.t_obs)/self.dt_frame)+1

        self.rabi = rabi
        self.detuning = detuning

        self.gamma_exc = gamma_exc
        self.gamma_ph = gamma_ph
        self.omega_exc = omega_exc
        self.omega_cav = omega_cav
        self.k_z = k_z
        self.noise_exc = noise
        self.noise_ph = noise
        self.g0 = g0

        self.frame = cp.ones((nmax_1, nmax_2), dtype=cp.complex64)
        self.x_1 = cp.linspace(-long_1/2, +long_1/2, nmax_1, dtype = float)
        self.x_2 = cp.linspace(-long_2/2, +long_2/2, nmax_2, dtype = float)
        self.X, self.Y = cp.meshgrid(self.x_1, self.x_2)
        self.R = cp.hypot(self.X,self.Y)
        self.THETA = cp.angle(self.X+1j*self.Y)

        #Max energy in the system to define dt
        #omega_max = max(omega_cav*(np.sqrt(1e0 + (k_1[0]**2+k_2[0]**2)/k_z**2)) - omega_turning_field, omega_exc - omega_turning_field)
        omega_max = 32 #[meV]
        cst = 4 #increase cst if you see fluctuations
        self.dt = 1/(omega_max*cst)
        
        self.time = cp.arange(0, self.t_max, self.dt)  

        
        self.F_laser_r = cp.ones((nmax_1, nmax_2), dtype=cp.complex64)
        self.F_laser_t = cp.ones((int(self.t_max//self.dt)), dtype=cp.complex64)
        
        self.F_probe_r = cp.ones((nmax_1, nmax_2), dtype=cp.complex64)
        self.F_probe_t = cp.ones((int(self.t_max//self.dt)), dtype=cp.complex64)
        
           

        self.phi = cp.zeros((2, self.nmax_1, self.nmax_2), dtype = cp.complex64) #self.phi[0, :, :] = self.phi1=excitons and self.phi[1, :, :] = self.phi2=photons
        self.phi_pol = cp.zeros((2, self.nmax_1, self.nmax_2), dtype = cp.complex64)


        #Definition of the energies, time step and the linear evolution operator.
        k_1 = np.linspace(-2*np.pi/self.long_1*self.nmax_1/2, 2*np.pi/self.long_1*(self.nmax_1/2-1), self.nmax_1)
        k_2 = np.linspace(-2*np.pi/self.long_2*self.nmax_2/2, 2*np.pi/self.long_2*(self.nmax_2/2-1), self.nmax_2)
        K_1, K_2 = np.meshgrid(k_1, k_2)

        self.gamma = np.zeros((2, self.nmax_2, self.nmax_1), dtype=np.complex64)
        self.gamma[0, :, :] = self.gamma_exc
        self.gamma[1, :, :] = self.gamma_ph

        self.omega_LP_0 = (self.omega_exc + self.omega_cav)/2 - 0.5 * np.sqrt((self.omega_exc - self.omega_cav)**2 + 4*self.rabi**2)
        self.omega_pump = self.detuning + self.omega_LP_0
        omega_turning_field = self.omega_pump
        self.omega_pump *= 0
        self.omega_probe = omega_probe
        self.omega = np.zeros((2, self.nmax_2, self.nmax_1), dtype=np.complex64)
        self.omega[0, :, :] = self.omega_exc - omega_turning_field
        self.omega[1, :, :] = self.omega_cav * (np.sqrt(1 + (K_1**2 + K_2**2)/self.k_z**2)) - omega_turning_field

        #todo: define m_LP from k_z, hbar, C02, outside of the class
        dk = np.abs(k_1[1]-k_1[0])    #O: defining only one dk in this way may give problems if nmax1 != nmax2
        LB = (self.omega[0,:,self.nmax_1//2] + self.omega[1,:,self.nmax_1//2])/2 - 0.5 * \
            np.sqrt((self.omega[1,:,self.nmax_1//2] - self.omega[0,:,self.nmax_1//2])**2 + 4*self.rabi**2)
        h_bar = 0.654 #[meV*ps]
        E_lp_kk = np.gradient(np.gradient(h_bar*LB,dk),dk)
        self.m_LP = h_bar**2/E_lp_kk[self.nmax_2//2]

        self.omega[0, :, :] = np.fft.fftshift(self.omega[0, :, :])
        self.omega[1, :, :] = np.fft.fftshift(self.omega[1, :, :])
        
        self.omega=cp.asarray(self.omega)
        self.gamma=cp.asarray(self.gamma)
        
        

        # Build diagonal propagator
        
        self.propagator_diag = cp.zeros((2, self.nmax_1, self.nmax_2), dtype=cp.complex64)
        self.propagator_diag[0, :, :] = cp.exp(-1j*self.dt*0.5*(self.omega[0, :, :] + self.omega[1, :, :] - 0.5j* (self.gamma[0, :, :] + self.gamma[1, :, :]) - cp.sqrt((self.omega[0, :, :] - self.omega[1, :, :] - 0.5*1j*(self.gamma[1, :, :] - self.gamma[0, :, :]))**2 + 4*self.rabi**2)))
        self.propagator_diag[1, :, :] = cp.exp(-1j*self.dt*0.5*(self.omega[0, :, :] + self.omega[1, :, :] - 0.5j* (self.gamma[0, :, :] + self.gamma[1, :, :]) + cp.sqrt((self.omega[0, :, :] - self.omega[1, :, :] - 0.5*1j*(self.gamma[1, :, :] - self.gamma[0, :, :]))**2 + 4*self.rabi**2)))
        self.propagator_diag = cp.asarray(self.propagator_diag)

        # Hopfield coefficients
        
        self.hopfield_coefs = cp.zeros((2, self.nmax_1, self.nmax_2), dtype=np.complex64)  #self.hopfield_coefs[0,:,:]=Xk and self.hopfield_coefs[1,:,:]=Ck
        self.hopfield_coefs[1,:,:] = cp.sqrt((cp.sqrt((self.omega[1, :, :]-self.omega[0, :, :])**2 + 4*self.rabi**2) - (self.omega[1, :, :]-self.omega[0, :, :]))/(2*cp.sqrt((self.omega[1, :, :]-self.omega[0, :, :])**2 + 4*self.rabi**2)))
        self.hopfield_coefs[0,:,:] = cp.sqrt(1-self.hopfield_coefs[1,:,:]**2)
        
        
        
        self.v_gamma = cp.zeros((self.nmax_2, self.nmax_1), dtype=np.complex64)
        delta_gamma_1 = self.long_1/25
        delta_gamma_2 = self.long_2/25
        gamma_boarder = 20*self.gamma_ph
        id_x_1, id_x_2 = cp.ones(self.X.shape), cp.ones(self.Y.shape)
        A = (cp.exp(-0.5*(cp.multiply(cp.transpose(id_x_1), self.Y+self.long_2/2)**2/delta_gamma_2**2)) +
                cp.exp(-0.5*(self.long_2-(cp.multiply(cp.transpose(id_x_1), self.Y+self.long_2/2)))**2/delta_gamma_2**2))
        B = (cp.exp(-0.5*(cp.multiply(cp.transpose(self.x_1+self.long_1/2), id_x_2)**2/delta_gamma_1**2)) + cp.exp(-0.5*(
            self.long_1 - (cp.multiply(cp.transpose(self.x_1+self.long_1/2), id_x_2)))**2/delta_gamma_1**2))
        self.v_gamma = gamma_boarder*(A+B)/(A*B+1)
        
        
    def split_step(self, plan_fft, k: int) -> None:
        phi1 = self.phi[0, :, :]
        phi2 = self.phi[1, :, :]
        phi_lp = self.phi_pol[0, :, :]
        phi_up = self.phi_pol[1, :, :]
        
        # # REAL_SPACE
        laser_excitation(phi2, self.F_laser_r[:,:], self.F_laser_t[k], self.F_probe_r[:,:], self.F_probe_t[k], self.dt)
        single_particle_pot(phi2, self.dt, self.v_gamma)
        non_linearity(phi1, self.dt, self.g0)

        #FOURIER_SPACE
        plan_fft.fft(phi1, phi1, cp.cuda.cufft.CUFFT_FORWARD)
        plan_fft.fft(phi2, phi2, cp.cuda.cufft.CUFFT_FORWARD)

        linear_step(phi1, phi2, phi_up, phi_lp, self.propagator_diag, self.hopfield_coefs)

        plan_fft.fft(phi1, phi1, cp.cuda.cufft.CUFFT_INVERSE)
        plan_fft.fft(phi2, phi2, cp.cuda.cufft.CUFFT_INVERSE)

        phi1 /= np.prod(phi1.shape)
        phi2 /= np.prod(phi2.shape)
        
        #NOISE
        if k*self.dt >= self.t_noise:
            rand1 = cp.random.randn(self.nmax_2, self.nmax_1) + 1j*cp.random.randn(self.nmax_2, self.nmax_1)
            rand2 = cp.random.randn(self.nmax_2, self.nmax_1) + 1j*cp.random.randn(self.nmax_2, self.nmax_1)
            add_noise(phi1, phi2, rand1, rand2, self.v_gamma, self.gamma_exc, self.gamma_ph, self.dv, self.noise_exc, self.noise_ph)

    def evolution(self, save: bool=True) -> (cp.ndarray):
        phi1 = self.phi[0,:,:]
        phi2 = self.phi[1,:,:]
        stationary = 0
        
        if save:
            self.mean_cav_t_x_y = cp.zeros((self.n_frame, self.nmax_1, self.nmax_2), dtype = np.complex64)
            self.mean_exc_t_x_y = cp.zeros((self.n_frame, self.nmax_1, self.nmax_2), dtype = np.complex64)
            self.mean_cav_x_y_stat = cp.zeros((self.nmax_1, self.nmax_2), dtype = np.complex64)
            self.mean_exc_x_y_stat = cp.zeros((self.nmax_1, self.nmax_2), dtype = np.complex64)
            self.F_t = cp.zeros(self.n_frame, dtype = np.float32)
            r_t = 0
            i_frame = 0
            
        plan_fft = build_fft_plan(cp.zeros((self.nmax_1, self.nmax_2), dtype=np.complex64))
        
        for k in tqdm(range(int(self.t_max//self.dt))):
            self.split_step(plan_fft, k)
            if k*self.dt > self.t_stationary and stationary<1:
                self.mean_cav_x_y_stat = phi2
                self.mean_exc_x_y_stat = phi1
                stationary+=1
            if k*self.dt >= self.t_obs and save:
                r_t += self.dt
                if r_t>=self.dt_frame:
                    self.mean_cav_t_x_y[i_frame,:,:] = phi2
                    self.mean_exc_t_x_y[i_frame,:,:] = phi1
                    self.F_t[i_frame] = cp.max(cp.abs(self.F_laser_t))
                    i_frame += 1
                    r_t = 0