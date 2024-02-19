import numpy as np
import scipy
import cupy as cp
from cupy.random import rand, randn
#from utilitiesf import update_progress
import cupyx.scipy.fftpack as fftpack
import numba
from tqdm import tqdm

def build_fft_plan(A:np.ndarray) -> list:
        """Builds the FFT plan objects for propagation
        Args:
            A (np.ndarray): Array to transform.
        Returns:
            list: A list containing the FFT plans
        """
        plan_fft = fftpack.get_fft_plan(
            A, shape=A.shape, axes=(0, 1), value_type='C2C')
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

@numba.jit(parallel = True)
def build_propagator(propagator: np.ndarray, h_lin_0: np.ndarray, nmax_1:int, nmax_2: int, dt: float):
    for i in range(nmax_1):
        for j in range(nmax_2):
            propagator[j,i,:,:] = scipy.linalg.expm(-1j*h_lin_0[j,i,:,:]*dt)

class ggpe():
    
    def __init__(self, nmax_1: int, nmax_2: int, long_1: int, long_2: int, tempo_type: str, t_max: int, t_stationary: int, t_obs: int, t_probe: int, t_noise: int, dt_frame: float,
        gamma_exc: float, gamma_ph: float, noise: float, g0: float, detuning: float, omega_probe: float, omega_exc: float, omega_cav: float, rabi: float, k_z: float) -> None:
    
        self.nmax_1 = nmax_1
        self.nmax_2 = nmax_2
        self.long_1 = long_1
        self.long_2 = long_2
        self.dv = long_1*long_2*nmax_1/nmax_2
        
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
        
        self.frame = cp.ones((nmax_2, nmax_1), dtype=np.complex64)
        self.x_1 = cp.linspace(-long_1/2, +long_1/2, nmax_1, dtype = float)
        self.x_2 = cp.linspace(-long_2/2, +long_2/2, nmax_2, dtype = float)
        self.X, self.Y = cp.meshgrid(self.x_1, self.x_2)
        self.R = cp.hypot(self.X,self.Y)
        self.THETA = cp.angle(self.X+1j*self.Y)
        
        self.F_laser = cp.ones((nmax_2, nmax_1), dtype=np.complex64)
        self.F_probe = cp.ones((nmax_2, nmax_1), dtype=np.complex64)
        self.phi1 = cp.zeros((self.nmax_1, self.nmax_2), dtype=np.complex64)
        self.phi2 = cp.zeros((self.nmax_1, self.nmax_2), dtype=np.complex64)
        #-----------------------------------oscar's ideas-----------------------------------
        #self.phi12 = cp.array([self.phi1, self.phi2], dtype=np.complex64) # Merging both wavefunctions in the same object: phi12[0]=phi1, phi12[1]=phi2  
        #or is it better to directly
        #self.phi12 = cp.array([cp.zeros((self.nmax_1, self.nmax_2), cp.zeros((self.nmax_1, self.nmax_2)], dtype=np.complex64)
        
        #or inspired from h_lin_0 def I WILL COMMIT TO THIS CHOICE FOR THE TIME BEING
        #self.phi12 = np.zeros((self.nmax_2, self.nmax_1, 2), dtype = np.complex64) #self.phi12[:, :, 0] = self.phi1 and self.phi12[:, :, 1] = self.phi2
        #-----------------------------------------------------------------------------------
        
        
        #Definition of the enrgies, time step and the linear evolution operator.
        k_1 = np.linspace(-2*np.pi/self.long_1*self.nmax_1/2, 2*np.pi/self.long_1*(self.nmax_1/2-1), self.nmax_1)
        k_2 = np.linspace(-2*np.pi/self.long_2*self.nmax_2/2, 2*np.pi/self.long_2*(self.nmax_2/2-1), self.nmax_2)
        K_1, K_2 = np.meshgrid(k_1, k_2)
        
        self.gamma = np.zeros((self.nmax_2, self.nmax_1, 2), dtype=np.complex64)
        self.gamma[:, :, 0] = self.gamma_exc
        self.gamma[:, :, 1] = self.gamma_ph

        self.omega_LP_0 = (self.omega_exc + self.omega_cav)/2 - 0.5 * np.sqrt((self.omega_exc - self.omega_cav)**2 + 4*self.rabi**2)
        self.omega_pump = self.detuning + self.omega_LP_0
        omega_turning_field = self.omega_pump
        self.omega_pump *= 0
        self.omega_probe = omega_probe
        self.omega = np.zeros((self.nmax_2, self.nmax_1, 2), dtype=np.complex64)
        self.omega[:, :, 0] = self.omega_exc - omega_turning_field
        self.omega[:,:,1] = self.omega_cav * (np.sqrt(1 + (K_1**2 + K_2**2)/self.k_z**2)) - omega_turning_field

        #todo: define m_LP from k_z, hbar, C02, outside of the class
        dk = np.abs(k_1[1]-k_1[0])
        LB = (self.omega[:,self.nmax_1//2,0] + self.omega[:,self.nmax_1//2,1])/2 - 0.5 * \
            np.sqrt((self.omega[:,self.nmax_1//2,1] - self.omega[:,self.nmax_1//2,0])**2 + 4*self.rabi**2)
        h_bar = 0.654 #[meV*ps]
        E_lp_kk = np.gradient(np.gradient(h_bar*LB,dk),dk)
        self.m_LP = h_bar**2/E_lp_kk[self.nmax_2//2]
        
        self.omega[:, :, 0] = np.fft.fftshift(self.omega[:, :, 0])
        self.omega[:, :, 1] = np.fft.fftshift(self.omega[:, :, 1])
        
        C2 = np.sqrt((self.omega[:, :, 1]-self.omega[:, :, 0])**2 + 4*self.rabi**2) + (self.omega[:, :, 1]-self.omega[:, :, 0])
        C2 /= 2*np.sqrt((self.omega[:, :, 1]-self.omega[:, :, 0])**2 + 4*self.rabi**2)
        self.C2 = C2
        
        #Max energy in the system to define dt
        #omega_max = max(omega_cav*(np.sqrt(1e0 + (k_1[0]**2+k_2[0]**2)/k_z**2)) - omega_turning_field, omega_exc - omega_turning_field)
        omega_max = 32 #[meV]
        cst = 4 #increase cst if you see fluctuations
        self.dt = 1/(omega_max*cst)
        
        self.h_lin_0 = np.zeros((self.nmax_2, self.nmax_1, 2, 2), dtype = np.complex64) 
        self.h_lin_0[:, :, 0, 0] = self.omega[:, :, 0] - 0.5*1j*self.gamma[:, :, 0]
        self.h_lin_0[:, :, 1, 1] = self.omega[:, :, 1] - 0.5*1j*self.gamma[:, :, 1]
        self.h_lin_0[:, :, 0, 1] = self.rabi
        self.h_lin_0[:, :, 1, 0] = self.rabi
        
        self.propagator = np.zeros((self.nmax_2, self.nmax_1, 2, 2), dtype=np.complex64)
        build_propagator(self.propagator, self.h_lin_0, self.nmax_1, self.nmax_2, self.dt)
        self.propagator = cp.asarray(self.propagator)
        
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
        self.F_laser = self.F_laser * F
        self.F_laser[self.R**2 > radius**2] = 0
        self.pumptype = "tophat"
        
    def gaussian(self, F, radius=75):
        self.F_laser = self.F_laser * F * cp.exp(-self.R**2/radius**2)
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
            return 4*cp.exp(-((t-t_up)/(t_up/2))**2)
        else:
            return 1 + 3*cp.exp(-((t-t_up)/t_down)**2)

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
    
    def split_step(self, plan_fft, k: int) -> None:
        #REAL_SPACE
        resonant_excitation(self.phi2, self.F_laser_t, self.dt)
        if k*self.dt>self.t_probe:
            probe_excitation(self.phi2, self.F_probe_t, self.dt)
        single_particle_pot(self.phi2, self.dt, self.v_gamma)
        non_linearity(self.phi1, self.dt, self.g0)
        #resonant_excitation(self.phi12[:,:,1], self.F_laser_t, self.dt)
        #if k*self.dt>self.t_probe:
        #    probe_excitation(self.phi12[:,:,1], self.F_probe_t, self.dt)
        #single_particle_pot(self.phi12[:,:,1], self.dt, self.v_gamma)
        #non_linearity(self.phi12[:,:,0], self.dt, self.g0)
        
        #FOURIER_SPACE
        plan_fft.fft(self.phi1, self.phi1, cp.cuda.cufft.CUFFT_FORWARD)
        plan_fft.fft(self.phi2, self.phi2, cp.cuda.cufft.CUFFT_FORWARD)
        cp.multiply(self.phi1, self.propagator[:, :, 0, 0], self.phi1)
        self.phi1 += cp.multiply(self.phi2, self.propagator[:, :, 0, 1])
        cp.multiply(self.phi2, self.propagator[:, :, 1, 1], self.phi2)
        self.phi2 += cp.multiply(self.phi1, self.propagator[:, :, 1, 0])
        plan_fft.fft(self.phi1, self.phi1, cp.cuda.cufft.CUFFT_INVERSE)
        plan_fft.fft(self.phi2, self.phi2, cp.cuda.cufft.CUFFT_INVERSE)
        self.phi1 /= np.prod(self.phi1.shape)
        self.phi2 /= np.prod(self.phi2.shape)
        #plan_fft.fft(self.phi12[:,:,0],self.phi12[:,:,0],cp.cuda.cufft.CUFFT_FORWARD)
        #plan_fft.fft(self.phi12[:,:,1],self.phi12[:,:,1],cp.cuda.cufft.CUFFT_FORWARD)
        #cp.multiply(self.phi12[:,:,0], self.propagator[:, :, 0, 0], self.phi12[:,:,0])
        #self.phi12[:,:,0] += cp.multiply(self.phi12[:,:,1], self.propagator[:, :, 0, 1])
        #cp.multiply(self.phi12[:,:,1], self.propagator[:, :, 1, 1], self.phi12[:,:,1])
        #self.phi12[:,:,1] += cp.multiply(self.phi12[:,:,0], self.propagator[:, :, 1, 0])
        #plan_fft.fft(self.phi12[:,:,0], self.phi12[:,:,0], cp.cuda.cufft.CUFFT_INVERSE)
        #plan_fft.fft(self.phi12[:,:,1], self.phi12[:,:,1], cp.cuda.cufft.CUFFT_INVERSE)
        #self.phi12[:,:,0] /= np.prod(self.phi12[:,:,0].shape)
        #self.phi12[:,:,1] /= np.prod(self.phi12[:,:,1].shape)
        
        #NOISE
        if k*self.dt >= self.t_noise:
            rand1 = cp.random.randn(self.nmax_2, self.nmax_1) + 1j*cp.random.randn(self.nmax_2, self.nmax_1) 
            rand2 = cp.random.randn(self.nmax_2, self.nmax_1) + 1j*cp.random.randn(self.nmax_2, self.nmax_1)
            add_noise(self.phi1, self.phi2, rand1, rand2, self.v_gamma, self.gamma_exc, self.gamma_ph, self.dv, self.noise_exc, self.noise_ph)
            #add_noise(self.phi12[:,:,0], self.phi12[:,:,1], rand1, rand2, self.v_gamma, self.gamma_exc, self.gamma_ph, self.dv, self.noise_exc, self.noise_ph)
    
    def evolution(self, save: bool=True) -> (cp.ndarray):#, cp.ndarray, cp.ndarray):
        stationary = 0
        if save:
            self.mean_cav_x_y_t = cp.zeros((self.nmax_2, self.nmax_1, self.n_frame), dtype = np.complex64)
            self.mean_exc_x_y_t = cp.zeros((self.nmax_2, self.nmax_1, self.n_frame), dtype = np.complex64)
            self.mean_cav_x_y_stat = cp.zeros((self.nmax_2, self.nmax_1), dtype = np.complex64)
            self.mean_exc_x_y_stat = cp.zeros((self.nmax_2, self.nmax_1), dtype = np.complex64)
            self.F_t = cp.zeros(self.n_frame, dtype = np.float32)
            r_t = 0
            i_frame = 0
        plan_fft = build_fft_plan(cp.zeros((self.nmax_1, self.nmax_2), dtype=np.complex64))          
        for k in tqdm(range(int(self.t_max//self.dt))):
            self.F_probe_t = self.F_probe * self.tempo_probe(k*self.dt)
            self.F_laser_t = self.F_laser * self.temp(k*self.dt)
            self.split_step(plan_fft, k)
            if k*self.dt > self.t_stationary and stationary<1:
                self.mean_cav_x_y_stat = self.phi2
                self.mean_exc_x_y_stat = self.phi1
                #self.mean_cav_x_y_stat = self.phi12[:,:,1]
                #self.mean_exc_x_y_stat = self.phi12[:,:,0]
                stationary+=1
            if k*self.dt >= self.t_obs and save:
                r_t += self.dt
                if r_t>=self.dt_frame:
                    self.mean_cav_x_y_t[:,:,i_frame] = self.phi2
                    self.mean_exc_x_y_t[:,:,i_frame] = self.phi1
                    #self.mean_cav_x_y_t[:,:,i_frame] = self.phi12[:,:,1]
                    #self.mean_exc_x_y_t[:,:,i_frame] = self.phi12[:,:,0]
                    self.F_t[i_frame] = cp.max(cp.abs(self.F_laser_t))
                    i_frame += 1
                    r_t = 0