import numpy as np
from .exponential import sim_exponential_1D
from .gaussian import sim_gaussian_1D
from ..peak import Coordinate, Phase

def sim_composite_1D(
    time_domain_size: int,
    frequency_domain_size: int,
    constant_time_region_size: int,
    frequency_pts: float,
    exp_line_width: Coordinate,
    gauss_line_width: Coordinate,
    amplitude: float,
    exp_phase: Phase,
    gauss_phase: Phase,
    gauss_weight: float,
    scale: float = 1.0
    ) -> np.ndarray:
    """
    Simulates a composite 1D spectrum by combining exponential and Gaussian components.

    Parameters
    ----------
    time_domain_size : int
        The size of the time domain.
    frequency_domain_size : int
        The size of the frequency domain.
    constant_time_region_size : int
        The size of the constant time region.
    frequency_pts : float
        The frequency in pts of the peak
    exp_line_width : Coordinate
        The line width for the exponential component.
    gauss_line_width : Coordinate
        The line width for the Gaussian component.
    amplitude : float
        The amplitude of the exponential component.
    exp_phase : Phase
        The phase of the exponential component.
    gauss_phase : Phase
        The phase of the Gaussian component.
    gauss_weight : float
        The weight of the Gaussian component in the composite spectrum.
    scale : float, optional
        A scaling factor applied to both components. Defaults to 1.0.

    Returns
    -------what
    np.ndarray [1D array]
        The composite 1D NMR spectrum.
    """

    exponential_component = sim_exponential_1D(
        time_domain_size, frequency_domain_size, constant_time_region_size,
        frequency_pts, exp_line_width, amplitude=amplitude, phase=exp_phase, scale=scale)
    
    gaussian_component = sim_gaussian_1D(
        time_domain_size, frequency_domain_size, constant_time_region_size,
        frequency_pts, gauss_line_width, amplitude=1.0, phase=gauss_phase, scale=scale)
    
    composite_spectrum = exponential_component + gauss_weight * gaussian_component
    return composite_spectrum