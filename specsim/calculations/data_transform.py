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
    if first_point == 1:
        first_point = first_point - 1 
    return array[first_point:last_point + 1]

def outer_product_summation(x_axis : list[np.ndarray], y_axis : list[np.ndarray]) -> np.ndarray:
    peak_count = len(x_axis)
    if peak_count != len(y_axis):
        raise ValueError(f"Number of peaks in the x-axis ({len(x_axis)}) does not match the peaks in the y-axis ({len(y_axis)})!")
    
    # Check for complex x_axis
    is_complex_x_axis = True if np.iscomplexobj(x_axis) else False

    x_axis = np.array(x_axis)
    y_axis = np.array(y_axis)
    y_length = y_axis.shape[-1]
    x_length = x_axis.shape[-1]

    if np.iscomplexobj(y_axis):
        interleaved_data = np.zeros(y_axis.shape[:-1] + (y_length * 2,), dtype=y_axis.real.dtype)
        for i in range(len(interleaved_data)):
            interleaved_data[i][0::2] = y_axis[i].real
            interleaved_data[i][1::2] = y_axis[i].imag

        y_axis = interleaved_data
        y_length = y_length * 2
    

    if is_complex_x_axis:
        real_part = np.einsum('ki,kj->ji', x_axis.real, y_axis)
        imag_part = np.einsum('ki,kj->ji', x_axis.imag, y_axis)
        data_result = real_part + 1j * imag_part
    else:
        data_result = np.einsum('ki,kj->ji', x_axis, y_axis)
    
    return data_result