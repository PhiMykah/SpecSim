import numpy as np
from ..calculations import calculate_decay, calculate_couplings

def simGaussian1D(
        time_domain_size : int,
        frequency_domain_size : int, 
        consant_time_region_size : int, 
        frequency_pts : float, 
        line_width_pts : float, 
        cos_mod_values : np.ndarray | None = None, 
        sin_mod_values : np.ndarray | None = None, 
        max_amplitude : float = 0.0,
        phase : tuple[float, float] = (0.0, 0.0) 
        ) -> np.ndarray:
    """
    Simulate a 1D NMR spectrum with a single Gaussian decay and J-coupling modulation.

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
    line_width_pts : float
        Line width in points
    cos_mod_values : np.ndarray [1D array] | None
        Cosine modulation frequencies (if any)
    sin_mod_values : np.ndarray [1D array] | None
        Sine modulation frequencies (if any)
    max_amplitude : float
        Maximum amplitude of the signal
    phase : tuple[float, float]
        Phase p0 and p1 of the signal

    Returns
    -------
    np.ndarray [1D array]
        The simulated 1D NMR spectrum
    """
    
    # For our cases, "useTDDecay" "useTDJmod" are true, and "useTimeScale" is false, so we can leave out this stuff.
    # We don't need sin modulation, so this part can be left out too.

    # If the amplitude is zero, return a zero array
    if (max_amplitude == 0):
        return np.zeros(time_domain_size, dtype=np.complex128)
    
    simulated_data = np.zeros(time_domain_size, dtype=np.complex128)
    
    # Ensure line width is positive
    line_width_pts = abs(line_width_pts)

    # If ctsize is negative set to zero
    if consant_time_region_size < 0:
        consant_time_region_size = 0

    # Set the constant time region to be bound by the time domain size
    consant_time_region_size = min(consant_time_region_size, time_domain_size)

    magic_gaussian = np.sqrt(2 / (8 * np.log(2))) # Magic number for Gaussian decay

    # Set the frequency value
    frequency : float = 2.0 * np.pi * (frequency_pts - (1 + frequency_domain_size / 2.0))/ frequency_domain_size

    # Set the line broadening value
    # !!! Make sure order of operations is correct
    line_broadening : float = (magic_gaussian * line_width_pts * np.pi) / time_domain_size

    # Initialize sum
    sum : list[float] = [0.0]

    # Set the phase-dependent values
    phase_delay = phase[1]/360.0
    phase_phi = np.pi*((phase[0] + phase[1])/2)/180.0 # Phi is calculated by taking the average of p0 and p1, then converting to radians

    # -------------------------- Create Time Domain Data ------------------------- # 

    # Change amplitude and for loop range if constant time region is set
    if consant_time_region_size > 0:
        amplitude_func = lambda x : 1.0
        calculate_decay(simulated_data, 0, consant_time_region_size,
                        phase_phi, phase_delay,
                        frequency, amplitude_func,
                        max_amplitude, sum)
        
        amplitude_func = lambda x : np.exp((1 + x - consant_time_region_size) * line_broadening)
        calculate_decay(simulated_data, consant_time_region_size, time_domain_size,
                        phase_phi, phase_delay,
                        frequency, amplitude_func,
                        max_amplitude, sum)
    else:
        amplitude_func = lambda x : np.exp(-1 * ((x * line_broadening) ** 2))
        calculate_decay(simulated_data, 0, time_domain_size,
                        phase_phi, phase_delay,
                        frequency, amplitude_func, 
                        max_amplitude)

    # ------------------------------ Apply Couplings ----------------------------- #

    # Include coupling only if provided

    if type(cos_mod_values) != type(None):
        cos_amplitude_func = lambda x, y : np.cos(x * y) # Cosine amplitude calculation function
        calculate_couplings(simulated_data, cos_mod_values, time_domain_size, cos_amplitude_func, True)
    if type(sin_mod_values) != type(None):
        sin_amplitude_func = lambda x, y : np.sin(x * y) # Sine amplitude calculation function
        calculate_couplings(simulated_data, sin_mod_values, time_domain_size, sin_amplitude_func, False)

    return simulated_data