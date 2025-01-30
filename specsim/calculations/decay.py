import numpy as np
from typing import Callable

def calculate_decay(
        simulated_data : np.ndarray, 
        first_data_point : int, 
        last_data_point : int,
        phi : float,
        delay : float,
        frequency : float,
        decay_func : Callable[[int], float],
        amplitude : float,
        sum : list[float] = [0.0],
        scale : float = 1.0):
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
    decay_func : callable[[int], float]
        Function defining how to calculate the amplitude
    amplitude : float
        Maximum amplitude of the signal
    sum : list[float]
        List to store the sum of the amplitudes
    scale : float
        Amplitude scaling factor, Default 1

    Returns 
    -------
    None
    """
    for i in range(first_data_point, last_data_point):
        # Decay curve
        decay_curve = decay_func(i)

        # Calculate new frequency based on the phase dependent values
        new_time = phi + (delay + i) * frequency

        # Set the real part of the simulated data
        simulated_data.real[i] = scale * amplitude * decay_curve * np.cos(new_time)
        # Set the imaginary part of the simulated data
        simulated_data.imag[i] = -1 * scale * amplitude * decay_curve * np.sin(new_time)

        # Increment sum
        sum[0] += amplitude