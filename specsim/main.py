import sys
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

from specsim import simExponential1D    # Import Exponential decay model
from specsim import simGaussian1D       # Import Gaussian decay model
from specsim import parse_commandline   # Import command-line parser
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

# ---------------------------------------------------------------------------- #
#                                 Main Function                                #
# ---------------------------------------------------------------------------- #
def main() -> int:
    spectral_widths = (2998.046875, 1920.000000)
    origins = (3297.501221, 6221.201172)
    observation_frequencies = (598.909973, 60.694000)
    total_points = (307, 128)
    
    user_path = input("Please enter the file path: ")
    
    test_spectrum = Spectrum(Path(user_path), # Path("data/hsqc/nlin_time.tab") # Path("data/hsqc/master.tab")
                 spectral_widths,
                 origins,
                 observation_frequencies,
                 total_points)

    print(test_spectrum)


def old_main() -> int:
    try:
        command_arguments = parse_commandline(sys.argv[1:]) 
        # print(command_arguments)

        exp_simulated_data, = simExponential1D(100, 100, 0, 
                                          10.0, 1.0, 
                                          np.array([1.0, 2.0, 3.0]),
                                          np.array([1.0, 2.0, 3.0]),
                                          1.0, (1.0, 180.0))

        exponential_integral = np.trapezoid(exp_simulated_data)

        gaus_simulated_data = simGaussian1D(100, 100, 0,
                                            10.0, 1.0,
                                            np.array([1.0, 2.0, 3.0]),
                                            np.array([1.0, 2.0, 3.0]),
                                            1.0, (1.0, 180.0))
        
        gaussian_integral = np.trapezoid(gaus_simulated_data)
        
        exp_simulated_data = np.fft.fft(exp_simulated_data)
        exp_simulated_data = np.fft.fftshift(exp_simulated_data)
        exp_simulated_data = np.flip(exp_simulated_data)
        exp_simulated_data = np.roll(exp_simulated_data, 1)

        gaus_simulated_data = np.fft.fft(gaus_simulated_data)
        gaus_simulated_data = np.fft.fftshift(gaus_simulated_data)
        gaus_simulated_data = np.flip(gaus_simulated_data)
        gaus_simulated_data = np.roll(gaus_simulated_data, 1)

        # exp_simulated_data = pype.DataFrame(array = exp_simulated_data)
        # gaus_simulated_data = pype.DataFrame(array = gaus_simulated_data)
        
        # exp_simulated_data.runFunc("FT")
        # gaus_simulated_data.runFunc("FT")
        
        # Save exponential simulated data plot to file
        plt.figure()
        plt.plot(exp_simulated_data.real)
        plt.plot(exp_simulated_data.imag)
        plt.savefig('simulated_data_e.png')

        # Save Gaussian simulated data plot to file
        plt.figure()
        plt.plot(gaus_simulated_data.real)
        plt.plot(gaus_simulated_data.imag)
        plt.savefig('simulated_data_g.png')

    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0

if __name__ == '__main__':
    sys.exit(main())

# (amplitude_exponential*exponential_function*exponential_phase + amplitude_gaussian*gaussian_function*gaussian_phase)*jcoupling_mod[0]*jcoupling_mod[1]*jcoupling_mod[2]
# amplitude_exponential = amplitude of exponential decay
# amplitude_gaussian = amplitude of Gaussian decay
# exponential_function = exponential decay function
# gaussian_function = Gaussian decay function
# exponential_phase = phase of exponential decay
# gaussian_phase = phase of Gaussian decay
# [jcoupling_mod1, jcoupling_mod2, jcoupling_mod3] = J-coupling modulation terms