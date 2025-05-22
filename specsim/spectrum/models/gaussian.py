import numpy as np
from typing import Any, Callable
from ...calculations import calculate_decay, calculate_couplings
from ...datatypes import PointUnits, Phase
from ...peak import Peak

def sim_gaussian_1D(
        peak : Peak,
        dimension : int,
        time_domain_size : int,
        frequency_domain_size : int, 
        frequency : PointUnits, 
        linewidths : list[PointUnits], 
        amplitude : float,
        phases : list[Phase],
        constant_time_region_size : int, 
        cos_mod_values : np.ndarray | None = None, 
        sin_mod_values : np.ndarray | None = None, 
        scale : float = 1.0
        ) -> np.ndarray:
    """
    Simulate a 1D NMR spectrum with a single exponential decay and J-coupling modulation.

    Parameters
    ----------
    peak : Peak
        Current peak to collect additional simulation data from.
    dimension : int
        Current 1D dimension index of simulation
    time_domain_size : int
        Number of points in the time domain
    frequency_domain_size : int
        Number of points in the frequency domain
    frequency : PointUnits
        Frequency in PointUnits
    linewidths : list[PointUnits]
        Linewidths of target dimension
    amplitude : float
        Amplitude of the signal
    phases : list[Phase]
        Phase p0 and p1 of the signal
    constant_time_region_size : int
        Number of points in the constant time region (usually 0)
    cos_mod_values : numpy.ndarray [1D array] | None
        Cosine modulation frequencies (if any)
    sin_mod_values : numpy.ndarray [1D array] | None
        Sine modulation frequencies (if any)
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
    
    simulated_data : np.ndarray[Any, np.dtype[Any]] = np.zeros(time_domain_size, dtype=np.complex64)
    
    # Ensure line width is positive
    line_width_pts = abs(linewidths[0].pts)

    # If ctsize is negative set to zero
    if constant_time_region_size < 0:
        constant_time_region_size = 0

    # Set the constant time region to be bound by the time domain size
    constant_time_region_size = min(constant_time_region_size, time_domain_size)

    magic_gaussian : float = np.sqrt(2 / (8 * np.log(2))) # Magic number for Gaussian decay (~0.6005612)

    # Set the frequency value
    freq : float = 2.0 * np.pi * (frequency.pts - (1 + frequency_domain_size / 2.0)) / frequency_domain_size

    # Set the line broadening value
    # !!! Make sure order of operations is correct
    line_broadening : float = (0.5 * magic_gaussian * line_width_pts * np.pi) / (time_domain_size) 

    # Initialize sum
    sum : list[float] = [0.0]

    # Set the phase-dependent values
    phase : Phase = phases[0]
    phase_delay : float = phase[1]/360.0
    phase_phi : float = np.pi*((phase[0] + phase[1])/2)/180.0 # Phi is calculated by taking the average of p0 and p1, then converting to radians

    # -------------------------- Create Time Domain Data ------------------------- # 

    # Change amplitude and for loop range if constant time region is set
    if constant_time_region_size > 0:
        constant_decay_func : Callable[..., float] = lambda x : 1.0
        calculate_decay(simulated_data, 0, constant_time_region_size,
                        phase_phi, phase_delay,
                        freq, constant_decay_func,
                        amplitude, sum, scale)
        
        decay_func : Callable[..., float] = lambda x : np.exp(((1 + x - constant_time_region_size) * line_broadening) ** 2)
        calculate_decay(simulated_data, constant_time_region_size, time_domain_size,
                        phase_phi, phase_delay,
                        freq, decay_func,
                        amplitude, sum, scale)
    else:
        decay_func : Callable[..., float] = lambda x : np.exp(-1 * ((x * line_broadening) ** 2))
        calculate_decay(simulated_data, 0, time_domain_size,
                        phase_phi, phase_delay,
                        freq, decay_func, 
                        amplitude, sum, scale)

    # ------------------------------ Apply Couplings ----------------------------- #

    # Include coupling only if provided

    if cos_mod_values:
        cos_decay_func : Callable[..., float] = lambda x, y : np.cos(x * y) # Cosine amplitude calculation function
        calculate_couplings(simulated_data, cos_mod_values, time_domain_size, cos_decay_func, True)
    if sin_mod_values:
        sin_decay_func : Callable[..., float] = lambda x, y : np.sin(x * y) # Sine amplitude calculation function
        calculate_couplings(simulated_data, sin_mod_values, time_domain_size, sin_decay_func, False)

    return simulated_data