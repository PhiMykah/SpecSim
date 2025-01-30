import sys
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import nmrPype as pype

from specsim import simExponential1D    # Import Exponential decay model
from specsim import simGaussian1D       # Import Gaussian decay model
from specsim import parseCommandLine   # Import command-line parser
from specsim import Coordinate          # Import coordinate data class
from specsim import Coordinate2D        # Import 2D coordinate data class
from specsim import Spectrum            # Import spectrum data class

"""
1. Gaussian decay model where each peak has adjustable phase.

2. Exponential decay model where each peak has adjustable phase.

3. Composite signal which is a weighted sum of (1) and (2), where the Gaussian and exponential signals have the same frequency, but each have their own adjustable decay and phase.

4. Multiplication by one or more J-coupling terms (cosine-modulation).

To-do:
    Function to compute a decay

    Function to compute a sinusoid of a desired phase

    Function to apply cosine modulation

"""

def getDimensionInfo(data_frame: pype.DataFrame, data_type : str) -> tuple[float, float]:
    """
    Obtain x-dimension and y-dimension information from the data frame.

    Parameters
    ----------
    data_frame : pype.DataFrame
        Target data frame
    data_type : str
        Header key for the data frame

    Returns
    -------
    tuple[float, float]
        x-dimension and y-dimension values
    """
    return (data_frame.getParam(data_type, 1), data_frame.getParam(data_type, 2))

def getTotalSize(data_frame : pype.DataFrame, header_key : str) -> tuple[int, int]:
    """
    Obtain the total size of the data frame.

    Parameters
    ----------
    data_frame : pype.DataFrame
        Target data frame

    header_key : str
        NMR Header key for size

    Returns
    -------
    tuple[int, int]
        Total size of the data frame
    """
    return tuple(map(int, getDimensionInfo(data_frame, header_key)))

# ---------------------------------------------------------------------------- #
#                                 Main Function                                #
# ---------------------------------------------------------------------------- #
def main() -> int:
    command_arguments = parseCommandLine(sys.argv[1:]) 

    data_frame = pype.DataFrame(command_arguments.ft)

    spectral_widths = getDimensionInfo(data_frame, 'NDSW')           # (2998.046875, 1920.000000)
    origins = getDimensionInfo(data_frame, 'NDORIG')                 # (3297.501221, 6221.201172)
    observation_frequencies = getDimensionInfo(data_frame, "NDOBS")  # (598.909973, 60.694000)
    total_time_points = getTotalSize(data_frame, 'NDTDSIZE')         # (512, 64) 
    total_freq_points = getTotalSize(data_frame, 'NDFTSIZE')         # (1024, 0)
    
    scaling_factor = command_arguments.scale

    # user_path = input("Please enter peak table file path: ")
    test_spectrum = Spectrum(Path("data/demo/nlin_time_single.tab"), # Path("data/hsqc/nlin_time.tab") # Path("data/hsqc/master.tab")
                 spectral_widths,
                 origins,
                 observation_frequencies,
                 total_time_points)

    print(test_spectrum)

    frequency_pts = test_spectrum.peaks[0].position.x
    line_width_pts = test_spectrum.peaks[0].linewidths[0]
    amplitude = test_spectrum.peaks[0].intensity
    phase = (-13, 0)

    if test_spectrum.peaks[0].extra_params["X_COSJ"]:
        cos_mod_j = np.array(test_spectrum.peaks[0].extra_params["X_COSJ"])
    else:
        cos_mod_j = None
    
    if test_spectrum.peaks[0].extra_params["X_SINJ"]:
        sin_mod_j = np.array(test_spectrum.peaks[0].extra_params["X_SINJ"])
    else:
        sin_mod_j = None

    exp_simulated_data = simExponential1D(total_time_points[0], total_freq_points[0], 0,
                                          frequency_pts, line_width_pts, 
                                          cos_mod_values=cos_mod_j,
                                          sin_mod_values=sin_mod_j,
                                          amplitude=amplitude, phase=phase,
                                          scale=scaling_factor)
    
    gaus_simulated_data = simGaussian1D(total_time_points[0], total_freq_points[0], 0,
                                        frequency_pts, line_width_pts,
                                        cos_mod_values=cos_mod_j,
                                        sin_mod_values=sin_mod_j,
                                        amplitude=amplitude, phase=phase,
                                        scale=scaling_factor)

    demo_data_exponential : np.ndarray = pype.DataFrame("data/demo/sim_time_single_exp.fid").array[0]
    demo_data_gaussian : np.ndarray = pype.DataFrame("data/demo/sim_time_single_gauss.fid").array[0]

    # Save exponential simulated data plot to file
    plt.figure()
    plt.plot(demo_data_exponential.real, 'tab:orange', label='original method')
    plt.plot(exp_simulated_data.real, 'tab:blue', label='specsim')
    plt.legend(loc="upper right")
    plt.savefig(f'simulated_data_ex.png')

    # Save Gaussian simulated data plot to file
    plt.figure()
    plt.plot(demo_data_gaussian.real, 'tab:orange', label='original method')
    plt.plot(gaus_simulated_data.real, 'tab:blue', label='specsim')
    plt.legend(loc="upper right")
    plt.savefig(f'simulated_data_gx.png')
        

if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    

# --------------------------- Calculating Integral --------------------------- #
# exponential_integral = np.trapezoid(exp_simulated_data)

# ---------------------------------------------------------------------------- #
#                                    Process                                   #
# ---------------------------------------------------------------------------- #
"""
(amplitude_exponential*exponential_function*exponential_phase + amplitude_gaussian*gaussian_function*gaussian_phase)*jcoupling_mod[0]*jcoupling_mod[1]*jcoupling_mod[2]

amplitude_exponential = amplitude of exponential decay
amplitude_gaussian = amplitude of Gaussian decay
exponential_function = exponential decay function
gaussian_function = Gaussian decay function
exponential_phase = phase of exponential decay
gaussian_phase = phase of Gaussian decay
[jcoupling_mod1, jcoupling_mod2, jcoupling_mod3] = J-coupling modulation terms
"""