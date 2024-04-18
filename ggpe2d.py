import numpy as np
import cupy as cp
import cupyx.scipy.fftpack as fftpack
from tqdm import tqdm
import kernels


class ggpe():
    
    def __init__(
        self, 
        omega_exc: float, 
        omega_cav: float, 
        gamma_exc: float, 
        gamma_cav: float, 
        g0: float, 
        rabi: float,
        k_z: float, 
        detuning: float,
        F_pump: float,
        F_probe: float,
        t_max: float, 
        t_stationary: float, 
        t_obs: float,
        dt_frame: float,  
        t_noise: float = 1e9, 
        Lx: float = 256,
        Ly: float = 256,
        Nx: int = 256,
        Ny: int = 256
    ) -> object:
        """Instantiates the simulation.
        
        DESCRIPTION TO DO

        Args:
            omega_exc (float): Exciton energy (meV/hbar)
            omega_cav (float): Cavity photon energy at k=0 (meV/hbar)
            gamma_exc (float): Exciton linewidth (meV/hbar)
            gamma_cav (float): Cavity photon linewidth (meV/hbar)
            g0 (float): Nonlinear coupling constant (meV/hbar)/(1/µm^2)
            rabi (float): Linear coupling (Rabi frequency) (meV/hbar)
            k_z (float, optional): Cavity wavenumber (1/µm) defined as n_cav*omega_cav/c. Defaults to 27.
            detuning (float): Laser pump detuning from the lower polariton energy at k=0 (meV/hbar) 
            F_pump (float): Pump field amplitude
            F_probe (float): Probe field amplitude
            t_max (float): Total time of the simulation
            t_stationary (float): Time to reach the stationary state
            t_obs (float): Time to start observing/saving the system state
            dt_frame (float): Time between frames
            t_noise (float, optional): Time to add noise. Defaults to 1e9.
            Lx (float, optional): Length of the cavity in the x direction (µm). Defaults to 256.
            Ly (float, optional): Length of the cavity in the y direction (µm). Defaults to 256.
            Nx (int, optional): Number of points in the x direction. Defaults to 256.
            Ny (int, optional): Number of points in the y direction. Defaults to 256.
        """
        
        #Import kernels
        self.kernels = kernels
        
        #Physical parameters
        self.omega_exc = omega_exc
        self.omega_cav = omega_cav
        self.gamma_exc = gamma_exc
        self.gamma_cav = gamma_cav
        self.g0 = g0
        self.rabi = rabi
        self.detuning = detuning
        self.F_pump = F_pump
        self.F_probe = F_probe
        self.k_z = k_z
        self.Lx = Lx
        self.Ly = Ly
        
        #Time parameters
        self.t_max = t_max
        self.t_stationary = t_stationary
        self.t_obs = t_obs
        self.dt_frame = dt_frame
        self.t_noise = t_noise
        
        #Axes, step and coordinate grids in real space
        self.Nx = Nx
        self.Ny = Ny
        self.X, self.delta_X = cp.linspace(
            -Lx / 2, 
            Lx / 2, 
            Nx, 
            retstep=True, 
            dtype=np.float64
        )
        self.Y, self.delta_Y = cp.linspace(
            -Ly / 2, 
            Ly / 2, 
            Ny, 
            retstep=True, 
            dtype=np.float64
        )
        self.XX, self.YY = cp.meshgrid(self.X, self.Y)
        self.R = cp.hypot(self.XX, self.YY)
        self.THETA = cp.angle(self.XX + 1j * self.YY)
        self.dv = Lx * Ly / (Nx * Ny)
        
        #Axes and coordinate grids in Fourier space
        self.Kx = 2 * np.pi * cp.fft.fftfreq(Nx, self.delta_X)
        self.Ky = 2 * np.pi * cp.fft.fftfreq(Ny, self.delta_Y)
        self.Kxx, self.Kyy = cp.meshgrid(self.Kx, self.Ky)
        
        #Time step defined from the maximum energy, time and frequency grids
        #omega_max = max(omega_cav*(np.sqrt(1e0 + (k_1[0]**2+k_2[0]**2)/k_z**2)) - omega_turning_field, omega_exc - omega_turning_field)
        omega_max = 32 #[meV]                              #TO BE MODIFIED ACCORDING TO THE PARAMETERS, this is just a had hoc value
        cst = 4 #increase cst if you see fluctuations
        self.dt = 1 / (omega_max * cst)
        self.time = cp.arange(0, self.t_max, self.dt)
        self.n_frame = int((self.t_max - self.t_obs) / self.dt_frame) + 1
        self.omega_list = 2 * cp.pi * cp.fft.fftshift(cp.fft.fftfreq(self.n_frame, self.dt_frame))
        
        #Energies and losses in Fourier space
        #Losses (may depend on gamma in general)
        self.gamma = cp.zeros((2, self.Nx, self.Ny), dtype=np.complex64)
        self.gamma[0, :, :] = self.gamma_exc
        self.gamma[1, :, :] = self.gamma_cav
        #Energies
        self.omega_LP_0 = (self.omega_exc + self.omega_cav) / 2 - 0.5 * cp.sqrt((self.omega_exc - self.omega_cav) ** 2 + 4 * self.rabi ** 2)
        self.omega_pump = self.detuning + self.omega_LP_0
        self.omega = cp.zeros((2, self.Nx, self.Ny), dtype=np.complex64)
        self.omega[0, :, :] = self.omega_exc - self.omega_pump
        self.omega[1, :, :] = self.omega_cav * (cp.sqrt(1 + (self.Kxx ** 2 + self.Kyy ** 2) / self.k_z ** 2)) - self.omega_pump
        
        #Hopfield coefficients in Fourier space
        self.hopfield_coefs = cp.zeros((2, self.Nx, self.Ny), dtype=np.complex64)  #self.hopfield_coefs[0,:,:]=Xk and self.hopfield_coefs[1,:,:]=Ck
        self.hopfield_coefs[1, :, :] = cp.sqrt((cp.sqrt((self.omega[1, :, :] - self.omega[0, :, :]) ** 2 + 4 * self.rabi ** 2) - (self.omega[1, :, :] - self.omega[0, :, :])) / (2 * cp.sqrt((self.omega[1, :, :] - self.omega[0, :, :]) ** 2 + 4 * self.rabi ** 2)))
        self.hopfield_coefs[0, :, :] = cp.sqrt(1 - self.hopfield_coefs[1, :, :] ** 2)
        
        # Build diagonal propagator in Fourier space
        self.propagator_diag = cp.zeros((2, self.Nx, self.Ny), dtype=cp.complex64)
        self.propagator_diag[0, :, :] = cp.exp(-1j * self.dt * 0.5 * (self.omega[0, :, :] + self.omega[1, :, :] - 0.5j * (self.gamma[0, :, :] + self.gamma[1, :, :]) - cp.sqrt((self.omega[0, :, :] - self.omega[1, :, :] - 0.5 * 1j * (self.gamma[1, :, :] - self.gamma[0, :, :])) ** 2 + 4 * self.rabi ** 2)))
        self.propagator_diag[1, :, :] = cp.exp(-1j * self.dt * 0.5 * (self.omega[0, :, :] + self.omega[1, :, :] - 0.5j * (self.gamma[0, :, :] + self.gamma[1, :, :]) + cp.sqrt((self.omega[0, :, :] - self.omega[1, :, :] - 0.5 * 1j * (self.gamma[1, :, :] - self.gamma[0, :, :])) ** 2 + 4 * self.rabi ** 2)))
        
        #Potential and losses at boundary in real space
        self.potential = cp.zeros((self.Nx, self.Ny), dtype=np.complex64)
        self.potential_profile = "Default: none"
        self.v_gamma = cp.zeros((self.Nx, self.Ny), dtype=np.complex64)
        delta_gamma_1 = self.Lx / 25
        delta_gamma_2 = self.Ly / 25
        gamma_border = 20 * self.gamma_cav
        id_x_1, id_x_2 = cp.ones(self.XX.shape, dtype = cp.complex64), cp.ones(self.YY.shape, dtype = cp.complex64)
        A = (cp.exp(-0.5 * (cp.multiply(cp.transpose(id_x_1), self.YY + self.Ly / 2) ** 2 / delta_gamma_2 ** 2)) +
                cp.exp(-0.5 * (self.Ly - (cp.multiply(cp.transpose(id_x_1), self.YY + self.Ly / 2))) ** 2 / delta_gamma_2 ** 2))
        B = (cp.exp(-0.5 * (cp.multiply(cp.transpose(self.X + self.Lx / 2), id_x_2) ** 2 / delta_gamma_1 ** 2)) + cp.exp(-0.5 * (
            self.Lx - (cp.multiply(cp.transpose(self.X + self.Lx / 2), id_x_2))) ** 2 / delta_gamma_1 ** 2))
        self.v_gamma = gamma_border * (A + B) / (A * B + 1)
        
        #Fields
        self.phi = None
        self.F_pump_r = cp.ones((Nx, Ny), dtype=cp.complex64)
        self.F_pump_t = cp.ones(self.time.shape, dtype=cp.complex64)
        self.F_probe_r = cp.ones((Nx, Ny), dtype=cp.complex64)
        self.F_probe_t = cp.ones(self.time.shape, dtype=cp.complex64)
        self.pump_spatial_profile = "Default: constant"
        self.probe_spatial_profile = "Default: constant"
        self.pump_temporal_profile = "Default: constant"
        self.probe_temporal_profile = "Default: constant"
        
        
    #FFT plans
    def build_fft_plans(self, A: cp.ndarray):
        """Builds the FFT plan objects for propagation, will transform the last two axes of the array

        Args:
            A (np.ndarray): Array to transform.
        Returns:
             FFT plans
        """
        plan_fft = fftpack.get_fft_plan(
            A,
            axes = (-2, -1),
            value_type="C2C",
        )
        return plan_fft
        
    def split_step(
        self,
        phi: cp.array,
        phi_pol: cp.array,
        plan_fft,
        k: int,   
        t_noise: float      
    ) -> None:
        """Split step function for one propagation step

        Args:
            phi (cp.array): Fields to propagate
            phi_pol (cp.array): Fields in the polariton basis (for storage in intermediary step)
            plan_fft (_type_): List of FFT plan objects
            k (int): Iteration index
            t_noise (float): Time at which the noise starts
        """
        phi_exc = phi[0, ... , :, :]
        phi_cav = phi[1, ... , :, :]
        phi_lp = phi_pol[0, ... , :, :]
        phi_up = phi_pol[1, ... , :, :]
        
        #Real space
        self.kernels.laser_excitation(phi_cav, self.F_pump, self.F_pump_r, self.F_pump_t[k], self.F_probe, self.F_probe_r, self.F_probe_t[...,k], self.dt)
        self.kernels.boundary_losses(phi_cav, self.dt, self.v_gamma)
        self.kernels.apply_potential(phi_cav, self.dt, self.potential)
        self.kernels.non_linearity(phi_exc, self.dt, self.g0)
        
        #Fourier space
        plan_fft.fft(phi_exc, phi_exc, cp.cuda.cufft.CUFFT_FORWARD) # should only transform the last 2 arrays
        plan_fft.fft(phi_cav, phi_cav, cp.cuda.cufft.CUFFT_FORWARD)
        
        self.kernels.linear_step(phi_exc, phi_cav, phi_up, phi_lp, self.propagator_diag[0, :, :], self.propagator_diag[1, :, :], self.hopfield_coefs[0, :, :], self.hopfield_coefs[1, :, :])
        
        plan_fft.fft(phi_exc, phi_exc, cp.cuda.cufft.CUFFT_INVERSE)
        plan_fft.fft(phi_cav, phi_cav, cp.cuda.cufft.CUFFT_INVERSE)
        
        phi_exc *= 1 / np.prod(phi_exc.shape[-2:])
        phi_cav *= 1 / np.prod(phi_cav.shape[-2:])
        
        #Noise
        if k * self.dt >= t_noise:
            rand1 = cp.random.normal(loc = 0, scale = self.dt, size = (self.Nx, self.Ny), dtype = np.float64) + 1j * cp.random.normal(loc = 0, scale = self.dt, size = (self.Nx, self.Ny), dtype = np.float64)
            rand2 = cp.random.normal(loc = 0, scale = self.dt, size = (self.Nx, self.Ny), dtype = np.float64) + 1j * cp.random.normal(loc = 0, scale = self.dt, size = (self.Nx, self.Ny), dtype = np.float64)
            self.kernels.add_noise(phi_exc, phi_cav, rand1, rand2, self.v_gamma, self.gamma_exc, self.gamma_cav, self.dv)
            # check complex normal distribution, circularly symmetric central case in wikipedia. If no mistake we have complex normal distribution 
            # for the random variable Z=X+iY with std gamma (real) iff X and Y have std gamma/2 
        
    def evolution(
        self, 
        initial_state: cp.ndarray = None,
        save_fields_at_time: float = 0,
        save: bool = True
    ) -> (cp.ndarray):
        """Evolve the system for a given time
        
        Args:
            initial_state (cp.ndarray, optional): Initial state of the system. Defaults to None.
            save (bool, optional): Save the data. Defaults to True.
        
        
        """
        print("dt = " + str(self.dt))
        
        if len(self.F_probe_t.shape) == 1 and len(self.F_probe_r.shape) == 2:
            self.phi = cp.zeros((2, self.Nx, self.Ny), dtype=np.complex64)
        elif len(self.F_probe_t.shape) == 1 and len(self.F_probe_r.shape) > 2:
            self.phi = cp.zeros((2, self.F_probe_r.shape[0], 1, self.Nx, self.Ny), dtype=np.complex64)
        elif len(self.F_probe_t.shape) > 1 and len(self.F_probe_r.shape) == 2:
            self.phi = cp.zeros((2, 1, self.F_probe_t.shape[1], self.Nx, self.Ny), dtype=np.complex64)
        else:
            self.phi = cp.zeros((2, self.F_probe_r.shape[0], self.F_probe_t.shape[1], self.Nx, self.Ny), dtype=np.complex64)
        
        stationary = 0
        save_fields = 1
            
        if initial_state is not None:
            self.phi[0, ... , :, :] = initial_state[0, :, :]
            self.phi[1, ... , :, :] = initial_state[1, :, :]
            stationary = 1
                
        self.phi_pol = cp.zeros(self.phi.shape, dtype=np.complex64)
        
        self.mean_cav_t_x_y = None
        self.mean_exc_t_x_y = None
        self.mean_cav_x_y_stat = None
        self.mean_exc_x_y_stat = None
        self.mean_cav_t_save = None
        self.mean_exc_t_save = None
        self.F_t = None
        
        if save and stationary == 0:
            self.mean_cav_t_x_y = cp.zeros(self.phi[1, ... , :, :].shape[0:-2]+(self.n_frame, self.Nx, self.Ny), dtype=np.complex64)
            self.mean_exc_t_x_y = cp.zeros(self.phi[0, ... , :, :].shape[0:-2]+(self.n_frame,self.Nx, self.Ny), dtype=np.complex64)
            self.mean_cav_x_y_stat = cp.zeros(self.phi[1, ... , :, :].shape, dtype = np.complex64)
            self.mean_exc_x_y_stat = cp.zeros(self.phi[0, ... , :, :].shape, dtype = np.complex64)
            self.F_t = cp.zeros(self.n_frame, dtype = np.float32)
            r_t = 0
            i_frame = 0
        if save and stationary == 1:
            self.mean_cav_t_x_y = cp.zeros(self.phi[1, ... , :, :].shape[0:-2]+(self.n_frame, self.Nx, self.Ny), dtype=np.complex64)
            self.mean_exc_t_x_y = cp.zeros(self.phi[0, ... , :, :].shape[0:-2]+(self.n_frame,self.Nx, self.Ny), dtype=np.complex64)
            self.F_t = cp.zeros(self.n_frame, dtype=np.float32)
            r_t = 0
            i_frame = 0
        if save_fields_at_time > 0:
            self.mean_cav_t_save = cp.zeros(self.phi[1, ... , :, :].shape, dtype=np.complex64)
            self.mean_exc_t_save = cp.zeros(self.phi[0, ... , :, :].shape, dtype=np.complex64)
            self.F_t = cp.zeros(self.n_frame, dtype=np.float32)
            save_fields = 0
        
            
        
        plan_fft = self.build_fft_plans(self.phi[0, ... , :, :])
        
        for k in tqdm(range(len(self.time))):
            self.split_step(self.phi, self.phi_pol, plan_fft, k, self.t_noise)
            if k * self.dt > self.t_stationary and stationary < 1:
                print("Saving stationary state: k = "+str(k)+", t = "+str(k * self.dt)+" ps")
                self.mean_cav_x_y_stat = self.phi[1, ... , :, :]
                self.mean_exc_x_y_stat = self.phi[0, ... , :, :]
                stationary += 1
            if k * self.dt >= save_fields_at_time and save_fields < 1:
                self.mean_cav_t_save = self.phi[1, ... , :, :]
                self.mean_exc_t_save = self.phi[0, ... , :, :]
                print("Saving fields state: k = "+str(k)+", t = "+str(k * self.dt)+" ps")
                save_fields += 1
            if k * self.dt >= self.t_obs and save:
                r_t += self.dt
                if r_t >= self.dt_frame:
                    self.mean_cav_t_x_y[..., i_frame, :, :] = self.phi[1, ... , :, :]
                    self.mean_exc_t_x_y[..., i_frame, :, :] = self.phi[0, ... , :, :]
                    self.F_t[i_frame] = cp.max(cp.abs(self.F_pump_t[k]))
                    i_frame += 1
                    r_t = 0