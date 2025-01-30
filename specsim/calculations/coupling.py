import numpy as np
from typing import Callable

def calculate_couplings(
        simulated_data : np.ndarray,
        modulation_values : np.ndarray,
        data_size : int,
        decay_func : Callable[[int, int], float],
        isCosine : bool = True):
    """
    Calculate the couplings of the simulated data.

    Parameters
    ----------
    simulated_data : np.ndarray [1D array]
        The simulated data array to modify
    modulation_values : np.ndarray [1D array]
        Modulation values (Sin or Cos)
    data_size : int
        The last data point of the simulated data array
    decay_func : callable[[int, int], float]
        Function defining how to calculate the decay
    isCosine : bool
        If True, calculate cosine modulation, else calculate sine modulation

    Returns
    -------
    None
    """
    for j in range(modulation_values.size):
        # Calculate initial modulation value
        modulation = np.pi * modulation_values[j]/(data_size - 1)

        for i in range(data_size):
            # Calculate the decay_curve
            decay_curve = decay_func(modulation, i)

            # Apply the modulation based on the cosine or sine
            if isCosine:
                simulated_data.real[i] *= decay_curve
                simulated_data.imag[i] *= decay_curve
            else:
                real_value = simulated_data.real[i]
                imag_value = simulated_data.imag[i]
                
                simulated_data.real[i] = imag_value * decay_curve
                simulated_data.imag[i] = -1*real_value * decay_curve