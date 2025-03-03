import cupy as cp

@cp.fuse(kernel_name="laser_excitation")
def laser_excitation(
    phi_cav: cp.ndarray, 
    F_pump: float, 
    F_pump_r: cp.ndarray, 
    F_pump_t: float, 
    F_probe: float, 
    F_probe_r: cp.ndarray, 
    F_probe_t: cp.ndarray, 
    dt: float
) -> None:
    """A fused kernel to apply the pump and probe excitation terms to the photon field

    Args:
        phi_cav (cp.ndarray): Photon field in ph,exc basis
        F_pump (float): Pump field amplitude
        F_pump_r (cp.ndarray): Spatial mode of the laser pump field
        F_pump_t (cp.ndarray): Temporal dependency of the laser pump field at corresponding time
        F_probe (float): Probe field amplitude
        F_probe_r (cp.ndarray): Spatial mode of the probe field
        F_probe_t (cp.ndarray): Temporal dependency of the probe field at corresponding time
        dt (float): Propagation step in ps
    """
    phi_cav = F_pump * F_pump_r * F_pump_t * dt * 1j
    phi_cav = F_probe * F_probe_r * F_probe_t * dt * 1j
    
    
@cp.fuse(kernel_name="boundary_losses")
def boundary_losses(
    phi_cav: cp.ndarray, 
    dt: float, 
    v_gamma: cp.array
) -> None:
    """A fused kernel to apply the potential term

    Args:
        phi_cav (cp.ndarray): Photon field in ph,exc basis
        dt (float): Propagation step in ps
        v_gamma (float): Loss at the edges of the grid and optical defects
    """
    phi_cav *= cp.exp(-dt * 0.5 * v_gamma) 
    
@cp.fuse(kernel_name="apply_potential")
def apply_potential(
    phi_cav: cp.ndarray, 
    dt: float, 
    potential: cp.ndarray
) -> None:
    """A fused kernel to apply the potential term

    Args:
        phi_cav (cp.ndarray): Photon field in ph,exc basis
        dt (float): Propagation step in ps
        potential (cp.ndarray): Potential profile
    """
    phi_cav *= cp.exp(-1j*dt*potential) 


@cp.fuse(kernel_name="non_linearity")
def non_linearity(
    phi_exc: cp.ndarray,
    phi_cav: cp.ndarray,
    den_reservoir: cp.ndarray,
    dt: float,
    g0: float,
    X02: float
) -> None:
    """A fused kernel to apply non linearity term

    Args:
        phi_exc (cp.ndarray): Exciton field in ph,exc basis
        dt (float): Propagation step in ps
        g0 (float): Interaction constant/coupling parameter
    """
    #2 possibilities:
    #phi_exc *= cp.exp(-1j * dt * g0 * (cp.abs(phi_exc) ** 2 + den_reservoir))
    #phi_cav *= cp.exp(-1j * dt * g0 * den_reservoir)
    phi_exc *= cp.exp(-1j * dt * g0 * (cp.abs(phi_exc) ** 2 + den_reservoir))
    
@cp.fuse(kernel_name="linear_step")
def linear_step(
    phi_exc: cp.ndarray,
    phi_cav: cp.ndarray,
    phi_up: cp.ndarray,
    phi_lp: cp.ndarray, 
    propagator_diag_lp: cp.ndarray,
    propagator_diag_up: cp.ndarray,
    X_hop: cp.ndarray,
    C_hop: cp.ndarray,
) -> None:
    """A fused kernel to apply the linear step in the diagonal polariton basis

    Args:
        phi1 (cp.ndarray): Exciton field in ph,exc basis (in k-space)
        phi2 (cp.ndarray): Photon field in ph,exc basis (in k-space)
        phi_up (cp.ndarray): array to stock UP component of the field in the UP-LP basis
        phi_lp (cp.ndarray): array to stock LP component of the field in the UP-LP basis
        propagator (cp.ndarray): Propagator in the diagonal polariton basis
        hopfield_coefs (cp.ndarray): array with the Hopfield coefficients
    """
    
    cp.multiply(phi_exc, -1 * X_hop, phi_lp)
    phi_lp += cp.multiply(phi_cav, C_hop)
    cp.multiply(phi_exc, C_hop, phi_up)
    phi_up += cp.multiply(phi_cav, X_hop)

    cp.multiply(phi_lp, propagator_diag_lp, phi_lp)
    cp.multiply(phi_up, propagator_diag_up, phi_up)

    cp.multiply(phi_lp, -1 * X_hop, phi_exc) 
    phi_exc += cp.multiply(phi_up, C_hop)
    cp.multiply(phi_lp, C_hop, phi_cav) 
    phi_cav += cp.multiply(phi_up, X_hop)

@cp.fuse(kernel_name="reservoir")
def reservoir_losses(
    den_reservoir: cp.ndarray,
    phi_exc: cp.ndarray,
    phi_cav: cp.ndarray,
    dt: float,
    gamma_res: float,
    gamma_exc: float,
    gamma_ph: float,
) -> None:
    """A fused kernel to apply the reservoir losses
    
    Args:
    den_reservoir (cp.ndarray): Dark excitonic reservoir density
    phi_exc (cp.ndarray): Exciton field in ph,exc basis
    phi_cav (cp.ndarray): Photon field in ph,exc basis
    gamma_res (float): Reservoir decay rate
    gamma_exc (float): Exciton decay rate
    gamma_ph (float): Photon decay rate
    """
    den_reservoir += dt * (gamma_res * cp.abs(phi_exc)**2 + gamma_res * cp.abs(phi_cav)**2 - gamma_exc * den_reservoir)
    
@cp.fuse(kernel_name="add_noise")
def add_noise(phi_exc: cp.ndarray,
              phi_cav: cp.ndarray,
              rand1: cp.ndarray,
              rand2: cp.ndarray,
              v_gamma: cp.ndarray,
              gamma_exc: float,
              gamma_ph: float,
              dv: float
) -> None:
    """A fused kernel to add gaussian noise (additive white gaussian noise)

    Args:
        phi_exc (cp.ndarray): Exciton field in ph,exc basis
        phi_cav (cp.ndarray): Photon field in ph,exc basis
        rand1 (cp.ndarray): Random array normally distributed
        rand2 (cp.ndarray): Random array normally distributed
        v_gamma (float): Loss at the edges of the grid
        gamma_exc (float): Exciton decay rate
        gamma_ph (float): Photon decay rate
    """
    
    # phi1 += noise_exc*cp.sqrt(gamma_exc/(4*dv))*rand1
    # phi2 += noise_ph*cp.sqrt((v_gamma+gamma_ph)/(4*dv))*rand2
    phi_exc += cp.sqrt(gamma_exc / (4 * dv)) * rand1
    phi_cav += cp.sqrt((v_gamma + gamma_ph) / (4 * dv)) * rand2