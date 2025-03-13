import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

def plot_1D(file_name : str, *arrays : np.ndarray):
    """
    plot a 1D array and save the plot to a file

    Parameters
    ----------
    file_name : str
        Name of the file without the file extension
    array : numpy.ndarray | list[numpy.ndarray] (1D Array)
        1D Array(s) to draw with matplotlib
    """
    plt.figure()
    index = 1
    for array in arrays:
        plt.plot(array, label=f'plot #{index}')
        index += 1
    plt.legend(loc="upper right")
    plt.savefig(Path(file_name).with_suffix('.png'))


def plot_2D(file_name : str, *arrays : np.ndarray):
    """
    plot a 2D array and save the plot to a file

    Parameters
    ----------
    file_name : str
        Name of the file without the file extension
    array : numpy.ndarray | list[numpy.ndarray] (1D Array)
        2D Array(s) to draw with matplotlib
    """
    plt.figure()
    index = 1
    for array in arrays:
        plt.contour(array, label=f'plot #{index}')
        index += 1
    plt.legend(loc="upper right")
    plt.savefig(Path(file_name).with_suffix('.png'))

def adjust_dimensions(simulated_data : np.ndarray, target_shape : tuple) -> np.ndarray:
    """
    Adjust the dimensions of the simulated data to match the dimensions of given shape

    Parameters
    ----------
    simulated_data : numpy.ndarray
        Simulated data to adjust dimensions
    target_shape : tuple
        Target dimensions of the data to modify

    Returns
    -------
    numpy.ndarray
        Trimmed or expanded numpy array based on necessary modification
    """
    if simulated_data.shape == target_shape:
        return simulated_data
    elif simulated_data.size < np.prod(target_shape):
        # If simulated data has fewer elements, pad with zeros
        adjusted_data = np.broadcast_to(simulated_data, target_shape)
        return adjusted_data
    else:
        # If simulated data has more elements, truncate the excess
        return simulated_data.flat[:np.prod(target_shape)].reshape(target_shape)