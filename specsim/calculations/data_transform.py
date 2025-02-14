import numpy as np

def fourier_transform(array : np.ndarray) -> np.ndarray:
    """
    Perform a fourier transform on the inputted array

    Parameters
    ----------
    array : numpy.ndarray
        Time-domain numpy array data

    Returns
    -------
    numpy.ndarray 
        Frequency-domain numpy array data
    """
    return(np.roll(np.flip(np.fft.fftshift(np.fft.fft(array))), 1))

def zero_fill(array : np.ndarray, new_size : int) -> np.ndarray:     
    """
    Zero-fills an input array to a new specified size.

    Parameters
    ----------
    array : numpy.ndarray
        The input array to be zero-filled.
    new_size : int
         The desired size of the output array.

    Returns
    -------
    numpy.ndarray
        The zero-filled array with the specified new size.

    Raises
    ------
    ValueError
        If the new size is not greater than the length of the input array.
    """    
    if new_size <= len(array):
        raise ValueError("New size must be greater than the length of the input array.")
    return np.pad(array, (0, int(new_size - len(array))), 'constant')

def extract_region(array : np.ndarray, first_point : int, last_point : int) -> np.ndarray:
    """
    Extracts a region from a given numpy array between specified points.

    Parameters
    ----------
    array : numpy.ndarray
        The input array from which the region will be extracted.
    first_point : int
        The starting index of the region to be extracted.
    last_point : int
        The ending index of the region to be extracted.

    Returns
    -------
    np.ndarray
        A new array containing the elements from the specified region.
    """
    if first_point == 0 and last_point == 0:
        return array
    if first_point != 0:
        first_point = first_point - 1 
    return array[first_point:last_point + 1]