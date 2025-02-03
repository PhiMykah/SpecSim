import numpy as np

def fourierTransform(array : np.ndarray) -> np.ndarray:
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