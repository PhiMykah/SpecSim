import numpy as np
from typing import Callable

def calculate_decay(
        simulated_data : np.ndarray, 
        first_data_point : int, 
        last_data_point : int,
        phi : float,
        delay : float,
        frequency : float,
        amplitude_func : Callable[[int], float],
        max_amplitude : float,
        sum : list[float] = [0.0]):
    """
    Calculate the decay of the simulated data.

    Parameters
    ----------
    simulated_data : np.ndarray [1D array]
        The simulated data array to modify
    first_data_point : int
        The first data point of the simulated data array
    last_data_point : int
        The last data point of the simulated data array
    phi : float
        Phase phi value
    delay : float
        Phase delay value
    frequency : float
        Frequency value
    amplitude_func : callable[[int], float]
        Function defining how to calculate the amplitude
    max_amplitude : float
        Maximum amplitude of the signal
    sum : list[float]
        List to store the sum of the amplitudes

    Returns 
    -------
    None
    """
    for i in range(first_data_point, last_data_point):
        # Current amplitude
        amplitude = amplitude_func(i)

        # Calculate new frequency based on the phase dependent values
        new_frequency = phi + (delay + i) * frequency

        # Set the real part of the simulated data
        simulated_data.real[i] = max_amplitude * amplitude * np.cos(new_frequency)
        # Set the imaginary part of the simulated data
        simulated_data.imag[i] = -1 * max_amplitude * amplitude * np.sin(new_frequency)

        # Increment sum
        sum[0] += amplitude