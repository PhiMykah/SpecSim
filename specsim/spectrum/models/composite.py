import numpy as np
from typing import Any
from .exponential import sim_exponential_1D
from .gaussian import sim_gaussian_1D
from ...datatypes import PointUnits, Phase
from ...peak import Peak
from sys import stderr

def sim_composite_1D(
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
    Simulate a 1D NMR spectrum with a combination of an exponential decay with gaussian decay. 
    Includes J-coupling modulation if provided

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
        Linewidths of target dimension for exponential and gaussian models
    amplitude : float
        Amplitude of the signal
    phases : list[Phase]
        Phase p0 and p1 of the signal for exponential and gaussian models
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
    if len(linewidths) < 2:
        # print("Warning: Less than 2 linewidths provided for composite simulation. Duplicating the first value.", file=stderr)
        linewidths = [linewidths[0]] * 2
    if len(phases) < 2:
        # print("Warning: Less than 2 phases provided for composite simulation. Duplicating the first value.", file=stderr)
        phases = [phases[0]] * 2
    
    # Calculate exponential
    exponential_component : np.ndarray[Any, np.dtype[Any]] = sim_exponential_1D(
        peak, dimension, time_domain_size, frequency_domain_size, frequency, [linewidths[0]], 
        amplitude, [phases[0]], constant_time_region_size, cos_mod_values, sin_mod_values, scale)
    
    # Calculate gaussian
    gaussian_component : np.ndarray[Any, np.dtype[Any]] = sim_gaussian_1D(
        peak, dimension, time_domain_size, frequency_domain_size, frequency, [linewidths[1]], 
        amplitude, [phases[1]], constant_time_region_size, cos_mod_values, sin_mod_values, scale)
    
    # Collect weights if they exist
    if peak.weights is not None and len(peak.weights) < 2:
        exp_weight : float = peak.weights[0][dimension]
        gauss_weight : float = peak.weights[0][dimension]
    elif peak.weights:
        exp_weight : float = peak.weights[0][dimension]
        gauss_weight : float = peak.weights[1][dimension]
    else:
        exp_weight : float = 1.0
        gauss_weight : float = 1.0

    

    # Compute combination of exponential and gaussian
    composite_spectrum : np.ndarray[Any, np.dtype[Any]] = exp_weight * exponential_component + gauss_weight * gaussian_component

    return composite_spectrum