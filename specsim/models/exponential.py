import numpy as np
from ..calculations import calculate_decay, calculate_couplings
from ..peak import Coordinate, Phase

def sim_exponential_1D(
        time_domain_size : int,
        frequency_domain_size : int, 
        consant_time_region_size : int, 
        frequency_pts : float, 
        line_width : Coordinate, 
        cos_mod_values : np.ndarray | None = None, 
        sin_mod_values : np.ndarray | None = None, 
        amplitude : float = 0.0,
        phase : Phase = None,
        scale : float = 1.0
        ) -> np.ndarray:
    """
    Simulate a 1D NMR spectrum with a single exponential decay and J-coupling modulation.

    Parameters
    ----------
    time_domain_size : int
        Number of points in the time domain
    frequency_domain_size : int
        Number of points in the frequency domain
    consant_time_region_size : int
        Number of points in the constant time region (usually 0)
    frequency_pts : float
        Frequency in points
    line_width : Coordinate
        Line width of dimension
    cos_mod_values : numpy.ndarray [1D array] | None
        Cosine modulation frequencies (if any)
    sin_mod_values : numpy.ndarray [1D array] | None
        Sine modulation frequencies (if any)
    amplitude : float
        Amplitude of the signal
    phase : Phase
        Phase p0 and p1 of the signal
    scale : float
        Amplitude scaling factor, Default 1

    Returns
    -------
    np.ndarray [1D array]
        The simulated 1D NMR spectrum
    """
    # For our cases, "useTDDecay" "useTDJmod" are true, and "useTimeScale" is false, so we can leave out this stuff.
    # We don't need sin modulation, so this part can be left out too.

    # If the amplitude is zero, time domain size is zero, or frequency domain size is zero, return a zero array
    if (amplitude == 0) or (time_domain_size == 0) or (frequency_domain_size == 0):
        return np.zeros(time_domain_size, dtype=np.complex64)
    
    simulated_data = np.zeros(time_domain_size, dtype=np.complex64)
    
    # Ensure line width is positive
    line_width_pts = abs(line_width.pts)

    # If ctsize is negative set to zero
    if consant_time_region_size < 0:
        consant_time_region_size = 0

    # Set the constant time region to be bound by the time domain size
    consant_time_region_size = min(consant_time_region_size, time_domain_size)

    # Set the frequency value
    frequency : float = 2.0 * np.pi * (frequency_pts - (1 + frequency_domain_size / 2.0))/ frequency_domain_size

    # Set the line broadening value
    line_broadening : float = -0.5 * line_width_pts * np.pi / (time_domain_size) 

    # Initialize sum
    sum : list[float] = [0.0]

    # Set the phase-dependent values
    phase_delay = phase[1]/360.0
    phase_phi = np.pi*((phase[0] + phase[1])/2)/180.0 # Phi is calculated by taking the average of p0 and p1, then converting to radians

    # -------------------------- Create Time Domain Data ------------------------- # 

    # Change amplitude and for loop range if constant time region is set
    if consant_time_region_size > 0:
        constant_decay_func = lambda x : 1.0
        calculate_decay(simulated_data, 0, consant_time_region_size,
                        phase_phi, phase_delay,
                        frequency, constant_decay_func,
                        amplitude, sum, scale)
        
        decay_func = lambda x : np.exp((1 + x - consant_time_region_size) * line_broadening)
        calculate_decay(simulated_data, consant_time_region_size, time_domain_size,
                        phase_phi, phase_delay,
                        frequency, decay_func,
                        amplitude, sum, scale)
    else:
        decay_func = lambda x : np.exp(x * line_broadening)
        calculate_decay(simulated_data, 0, time_domain_size,
                        phase_phi, phase_delay,
                        frequency, decay_func, 
                        amplitude, sum, scale)

    # ------------------------------ Apply Couplings ----------------------------- #

    # Include coupling only if provided
    
    if type(cos_mod_values) != type(None):
        cos_decay_func = lambda x, y : np.cos(x * y) # Cosine amplitude calculation function
        calculate_couplings(simulated_data, cos_mod_values, time_domain_size, cos_decay_func, True)
    if type(sin_mod_values) != type(None):
        sin_decay_func = lambda x, y : np.sin(x * y) # Sine amplitude calculation function
        calculate_couplings(simulated_data, sin_mod_values, time_domain_size, sin_decay_func, False)

    return simulated_data