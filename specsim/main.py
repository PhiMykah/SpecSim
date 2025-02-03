import sys
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import nmrPype as pype

# Command-line parsing  
from specsim import parseCommandLine, SpecSimArgs, getDimensionInfo, getTotalSize

# Simulation Models
from specsim import simExponential1D, simGaussian1D     

# Spectrum Generation
from specsim import Spectrum

# Spectrum Modification
from specsim import fourierTransform

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

# ---------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #

def main() -> int:
    command_arguments = SpecSimArgs(parseCommandLine(sys.argv[1:]))

    data_frame = pype.DataFrame(command_arguments.ft)

    spectral_widths = getDimensionInfo(data_frame, 'NDSW')
    origins = getDimensionInfo(data_frame, 'NDORIG')
    observation_frequencies = getDimensionInfo(data_frame, "NDOBS")
    total_time_points = getTotalSize(data_frame, 'NDTDSIZE')
    total_freq_points = getTotalSize(data_frame, 'NDFTSIZE')
    
    phase = (command_arguments.p0, command_arguments.p1)             # p0 and p1 phases of spectrum
    scaling_factor = command_arguments.scale                         # Scale factor for simulation
    xOffset = command_arguments.xOff                                 # Frequency x offset for simulation

    test_spectrum = Spectrum(Path(command_arguments.tab), # Path("data/hsqc/nlin_time.tab") # Path("data/hsqc/master.tab")
                 spectral_widths,
                 origins,
                 observation_frequencies,
                 total_time_points, 
                 total_freq_points)

    demo_data_exponential : np.ndarray = pype.DataFrame("data/demo/sim_time_single_exp.fid").array[0]
    demo_data_gaussian : np.ndarray = pype.DataFrame("data/demo/sim_time_single_gauss.fid").array[0]

    constant_time_region_size = 0
    num_of_peaks = 0

    exp_simulated_data = test_spectrum.spectralSimulation(simExponential1D, num_of_peaks, constant_time_region_size, phase, xOffset, scaling_factor)

    gaus_simulated_data = test_spectrum.spectralSimulation(simGaussian1D, num_of_peaks, constant_time_region_size, phase, xOffset, scaling_factor)
    
    # Save exponential simulated data plot to file
    plt.figure()
    plt.plot(demo_data_exponential.real, 'tab:orange', label='original method')
    plt.plot(exp_simulated_data[0].real, 'tab:blue', label='specsim')
    plt.legend(loc="upper right")
    plt.savefig(f'simulated_data_ex.png')

    # Save Gaussian simulated data plot to file
    plt.figure()
    plt.plot(demo_data_gaussian.real, 'tab:orange', label='original method')
    plt.plot(gaus_simulated_data[0].real, 'tab:blue', label='specsim')
    plt.legend(loc="upper right")
    plt.savefig(f'simulated_data_gx.png')

    exp_simulated_data_ft = fourierTransform(exp_simulated_data[0])
    gaus_simulated_data_ft = fourierTransform(gaus_simulated_data[0])

    demo_data_exponential_ft = fourierTransform(demo_data_exponential)
    demo_data_gaussian_ft = fourierTransform(demo_data_gaussian)

    # Save exponential simulated data plot to file
    plt.figure()
    plt.plot(demo_data_exponential_ft.real, 'tab:orange', label='original method')
    plt.plot(exp_simulated_data_ft.real, 'tab:blue', label='specsim')
    plt.legend(loc="upper right")
    plt.savefig(f'simulated_data_ex_ft.png')

    # Save Gaussian simulated data plot to file
    plt.figure()
    plt.plot(demo_data_gaussian_ft.real, 'tab:orange', label='original method')
    plt.plot(gaus_simulated_data_ft.real, 'tab:blue', label='specsim')
    plt.legend(loc="upper right")
    plt.savefig(f'simulated_data_gx_ft.png')

# ---------------------------------------------------------------------------- #
#                                 Main Function                                #
# ---------------------------------------------------------------------------- #

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