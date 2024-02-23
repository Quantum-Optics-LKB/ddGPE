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

@cp.fuse(kernel_name="resonant_excitation")
def resonant_excitation(phi2: cp.ndarray, F_laser_t: cp.ndarray, dt: float) -> None:
    """A fused kernel to apply resonant_excitation term

    Args:
        phi (cp.ndarray): The field in ph,exc basis
        F_laser_t (float): Excitation laser field
        dt (float): Propagation step in ps
    """
    phi2 -= F_laser_t*dt*1j

@cp.fuse(kernel_name="probe_excitation")
def probe_excitation(phi2: cp.ndarray, F_probe_t: cp.ndarray, dt: float) -> None:
    """A fused kernel to apply probe excitation term

    Args:
        phi (cp.ndarray): The field in ph,exc basis
        F_probe_t (float): Probe field
        dt (float): Propagation step in ps
    """
    phi2 -= F_probe_t*dt*1j

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
           
            

class ggpe():

    def __init__(self, nmax_1: int, nmax_2: int, long_1: int, long_2: int, tempo_type: str, t_max: int, t_stationary: int, t_obs: int, t_probe: int, t_noise: int, dt_frame: float,
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
        self.tempo_type = tempo_type
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

        self.F_laser = cp.ones((nmax_1, nmax_2), dtype=cp.complex64)
        self.F_probe = cp.ones((nmax_1, nmax_2), dtype=cp.complex64)
        
    
        #Max energy in the system to define dt
        #omega_max = max(omega_cav*(np.sqrt(1e0 + (k_1[0]**2+k_2[0]**2)/k_z**2)) - omega_turning_field, omega_exc - omega_turning_field)
        omega_max = 32 #[meV]
        cst = 4 #increase cst if you see fluctuations
        self.dt = 1/(omega_max*cst)
        
        #-------oscar's field-----------
        
        # self.F_laser = cp.ones((self.t_max//self.dt + 1, nmax_1, nmax_2), dtype=cp.complex64)
        # self.F_probe = cp.ones((self.t_max//self.dt + 1, nmax_1, nmax_2), dtype=cp.complex64)
        
        #self.F_laser_t = build_field(self.F_laser, self.F_probe, name="to_turning_point", t_up=400, t_down=400)
        
        
        
        
        
        
        
        
        
        
        
        
        

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

        # #Max energy in the system to define dt
        # #omega_max = max(omega_cav*(np.sqrt(1e0 + (k_1[0]**2+k_2[0]**2)/k_z**2)) - omega_turning_field, omega_exc - omega_turning_field)
        # omega_max = 32 #[meV]
        # cst = 4 #increase cst if you see fluctuations
        # self.dt = 1/(omega_max*cst)
        
        

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

    def tophat(self, F, radius=75):
        self.F_laser[self.R > radius] = 0
        self.F_laser = self.F_laser * F
        #self.F_laser[self.R**2 > radius**2] = 0
        self.pumptype = "tophat"

    def gaussian(self, F, radius=75):
        self.F_laser = self.F_laser * F * cp.exp(-self.R**2/radius**2) #O: identifying with Gaussian distribution, radius is sqrt(2)*std_deviation
        self.pumptype = "gaussian"

    def ring(self, F_probe, radius, delta_radius):
        self.F_probe[self.R>radius+delta_radius/2] = 0
        self.F_probe[self.R<radius-delta_radius/2] = 0
        self.F_probe = self.F_probe * F_probe

    def radial_expo(self, m_probe, p_probe):
        self.F_probe = self.F_probe * cp.exp(1j*p_probe*self.R) * cp.exp(1j*m_probe*self.THETA)

    def tempo_probe(self, t):
        return cp.exp(-1j*t*self.omega_probe)

    def vortex_beam(self, waist=75, inner_waist=22, C=15):
        self.F_laser = self.F_laser * cp.exp(1j*C*self.THETA)\
            *cp.tanh(self.R/inner_waist)**C

    def shear_layer(self, kx: float=1):
        phase = cp.zeros(self.X.shape)
        phase = kx*self.X[:,:]
        phase[self.Y>0]=-phase[self.Y>0]
        self.F_laser = self.F_laser*cp.exp(1j*phase)

    def plane_wave(self, kx=0.5):
        phase = cp.zeros(self.X.shape)
        phase = kx*self.X[:,:]
        self.F_laser = self.F_laser * cp.exp(1j*phase)

    def to_turning_point(self, t, t_up, t_down):
        if t<t_up:
            return 3*cp.exp(-((t-t_up)/(t_up/2))**2)
        else:
            return 1 + 2*cp.exp(-((t-t_up)/t_down)**2)

    def bistab_cycle(self, t):
        return 4*cp.exp(-((t-self.t_max//2)/(self.t_max//4))**2)

    def turn_on_pump(self, t, t_up=200):
        if t<t_up:
            return cp.exp(((t-t_up)/(t_up/2))**2)
        else:
            return 1

    def temp(self, t, name = "to_turning_pt", t_up=400, t_down=400) -> (cp.ndarray):
        if self.tempo_type == "to_turning_pt":
            return self.to_turning_point(t, t_up, t_down)
        if self.tempo_type == "bistab_cycle":
            return self.bistab_cycle(t)
        if self.tempo_type == "turn_on_pump":
            return self.turn_on_pump(t, t_up)
        
        
        
    # WORK IN PROGRESS
    #%%    
    # def build_field(self, pump_type: str = "tophat", probe_type: str = "ring", t_up: int = 400, t_down: int = 400, F: float = 1, radius: float = 75, m_probe: float = 0, p_probe: float = 0, delta_radius: float = 0, waist: float = 75, inner_waist: float = 22, C: float = 15, kx: float=1) -> (cp.ndarray):
    #     """Builds the field in time

    #     Args:
    #         F_laser (cp.ndarray): The pump laser field [t,x,y]
    #         F_probe (cp.ndarray): The probe field [t,x,y]
    #         name (str): Name of the temporal shape for the pump field
    #         t_up (int): 
    #         t_down (int): 

    #     Returns:
    #         cp.ndarray: The field at every time and position [t,x,y]
    #     """
        
    #     #spatial profile
    #     #pump
    #     if pump_type == "tophat":
    #         self.tophat(F,radius)
    #     if pump_type == "gaussian":
    #         self.gaussian(F,radius)
    #     if pump_type == "vortex_beam":
    #         self.vortex_beam(waist, inner_waist, C)
    #     if pump_type == "shear_layer":
    #         self.shear_layer(kx)
    #     if pump_type == "plane_wave":
    #         self.plane_wave(kx)
        
    #     #probe
    #     if probe_type == "ring":
    #         self.ring(F, radius, delta_radius)
    #     if probe_type == "radial_expo":
    #         self.radial_expo(m_probe, p_probe)
        
    #     #temporal profile
    #     #pump
    #     if self.tempo_type == "to_turning_pt":
    #         for k in range(len(self.F_laser)):
    #             if k*self.dt<t_up:
    #                 self.F_laser[k,:,:] = self.F_laser[k,:,:]*3*cp.exp(-((k*self.dt-t_up)/(t_up/2))**2)
    #             else:
    #                 self.F_laser[k,:,:] = self.F_laser[k,:,:] + 2*cp.exp(-((self.dt-t_up)/t_down)**2)
    #     if self.tempo_type == "bistab_cycle":
    #         for k in range(len(self.F_laser)):
    #             self.F_laser[k,:,:] = self.F_laser[k,:,:]*4*cp.exp(-((k*self.dt-self.t_max//2)/(self.t_max//4))**2)
    #     if self.tempo_type == "turn_on_pump":
    #         for k in range(len(self.F_laser)):
    #             if k*self.dt<t_up:
    #                 self.F_laser[k,:,:] = self.F_laser[k,:,:]*cp.exp(((k*self.dt-t_up)/(t_up/2))**2)
        
    #     #probe
        
    #     for k in range(len(self.F_probe)):
    #         if k*self.dt<
    #%%  
        
        

    def split_step(self, plan_fft, k: int) -> None:
        phi1 = self.phi[0, :, :]
        phi2 = self.phi[1, :, :]
        phi_lp = self.phi_pol[0, :, :]
        phi_up = self.phi_pol[1, :, :]
        
        # #REAL_SPACE
        resonant_excitation(phi2, self.F_laser_t, self.dt)
        if k*self.dt>self.t_probe:
           probe_excitation(phi2, self.F_probe_t, self.dt)
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
            self.F_probe_t = self.F_probe * self.tempo_probe(k*self.dt)
            self.F_laser_t = self.F_laser * self.temp(k*self.dt)
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