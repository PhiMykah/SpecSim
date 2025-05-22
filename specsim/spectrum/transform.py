import numpy as np
import numpy.typing as npt
from typing import Any

def outer_product_summation(x_axis : npt.ArrayLike, y_axis : npt.ArrayLike) -> np.ndarray:
    x_axis = np.array(x_axis)
    y_axis = np.array(y_axis)

    peak_count: int = len(x_axis)
    if peak_count != len(y_axis):
        raise ValueError(f"Number of peaks in the x-axis ({len(x_axis)}) does not match the peaks in the y-axis ({len(y_axis)})!")
    
    # Check for complex x_axis
    is_complex_x_axis: bool = True if np.iscomplexobj(x_axis) else False


    y_length : int = y_axis.shape[-1]
    # x_length = x_axis.shape[-1]

    if np.iscomplexobj(y_axis):
        interleaved_data : np.ndarray[Any, np.dtype[Any]] = np.zeros(y_axis.shape[:-1] + (y_length * 2,), dtype=y_axis.real.dtype)
        for i in range(len(interleaved_data)):
            interleaved_data[i][0::2] = y_axis[i].real
            interleaved_data[i][1::2] = y_axis[i].imag

        y_axis = interleaved_data
        y_length = y_length * 2
    

    if is_complex_x_axis:
        real_part : np.ndarray[Any, np.dtype[Any]] = np.einsum('ki,kj->ji', x_axis.real, y_axis)
        imag_part : np.ndarray[Any, np.dtype[Any]] = np.einsum('ki,kj->ji', x_axis.imag, y_axis)
        data_result : np.ndarray[Any, np.dtype[Any]] = real_part + 1j * imag_part
    else:
        data_result = np.einsum('ki,kj->ji', x_axis, y_axis)
    
    return data_result