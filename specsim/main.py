import sys
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import nmrPype as pype

# Command-line parsing  
from specsim import parse_command_line, SpecSimArgs, get_dimension_info, get_total_size

# Simulation Models
from specsim import sim_exponential_1D, sim_gaussian_1D     

# Spectrum Generation
from specsim import Spectrum

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

# ---------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #

def main() -> int:
    # Parse command-line
    command_arguments = SpecSimArgs(parse_command_line(sys.argv[1:]))
    data_frame = pype.DataFrame(command_arguments.ft2) # Spectrum nmrpype format data
    interferogram = pype.DataFrame(command_arguments.ft1) # Interferogram nmrpype format data
    data_fid = pype.DataFrame(command_arguments.fid)  # Full time-domain nmrpype format data
    output_file = command_arguments.out # Output file nmrpype format data

    # --------------------------- Simulation Parameters -------------------------- #
    axis_count = 2
    spectral_widths = get_dimension_info(data_frame, 'NDSW')
    origins = get_dimension_info(data_frame, 'NDORIG')
    observation_frequencies = get_dimension_info(data_frame, "NDOBS")
    total_time_points = get_total_size(data_frame, 'NDTDSIZE')
    total_freq_points = get_total_size(data_frame, 'NDFTSIZE')
    phase = (command_arguments.p0, command_arguments.p1)    # p0 and p1 phases of spectrum
    phases = [phase, phase]                                 # Phase values for x-axis and y-axis
    scaling_factors = [command_arguments.scale, 1.0]          # Time domain x and y scaling values
    offsets = [command_arguments.xOff, 0]                   # Frequency x and y offset for simulation                             
    constant_time_region_sizes = [0,0]                      # Size of x and y constant time regions
    peak_count = 0                                          # Number of peaks to simulate
    domain = "ft1"                                          # Domain of simulation data
    if output_file != None:
        suffix = Path(output_file).suffix
        if suffix:
            domain = suffix                                 # Domain of simulation data    
    else:
        output_file = f"test_sim.{domain}"

    # ------------------------------ Spectrum Class ------------------------------ #

    test_spectrum = Spectrum(Path(command_arguments.tab), # Path("data/hsqc/nlin_time.tab") # Path("data/hsqc/master.tab")
                 spectral_widths,
                 origins,
                 observation_frequencies,
                 total_time_points, 
                 total_freq_points)

    # ------------------------------ Run Simulation ------------------------------ #

    exp_simulated_data = test_spectrum.spectral_simulation(
        sim_exponential_1D, data_frame, axis_count,
        peak_count, domain, constant_time_region_sizes,
        phases, offsets, scaling_factors)

    gaus_simulated_data = test_spectrum.spectral_simulation(
        sim_gaussian_1D, data_frame, axis_count,
        peak_count, domain, constant_time_region_sizes,
        phases, offsets, scaling_factors)

    # ------------------------------ Plot Simulation ----------------------------- #

    # # Save exponential simulated data plot to file
    # plot_1D(f"simulated_data_ex_{domain}", exp_simulated_data[0].real)

    # # Save Gaussian simulated data plot to file
    # plot_1D(f"simulated_data_gx_{domain}", gaus_simulated_data[0].real)

    # if exp_simulated_data.ndim > 1:
    #     if exp_simulated_data.shape[0] >= 2 and exp_simulated_data.shape[1] >= 2:
    #         plot_2D(f"simulated_data_ex_{domain}_full", exp_simulated_data.real)

    # if gaus_simulated_data.ndim > 1:
    #     if exp_simulated_data.shape[0] >= 2 and exp_simulated_data.shape[1] >= 2:
    #         plot_2D(f"simulated_data_gx_{domain}full", gaus_simulated_data.real)

    if domain == "fid":
        output_df = data_fid
    elif domain == "ft1":
        output_df = interferogram
    elif domain == 'ft2':
        output_df = data_frame
    else:
        raise TypeError("Invalid nmrpipe data output format!")

    exp_comparison_data = output_df
    gaus_comparison_data = output_df

    exp_difference = exp_simulated_data - exp_comparison_data.array
    output_df.setArray(exp_simulated_data)
    exp_comparison_data.setArray(exp_difference)

    simulation_model : str = 'ex'
    output_file_path = Path(output_file).stem + f"_{simulation_model}" + f".{domain}"
    pype.write_to_file(output_df, output_file_path, True)
    difference_file_path = Path(output_file).stem + f"_diff" + f"_{simulation_model}" + f".{domain}"
    pype.write_to_file(exp_comparison_data, difference_file_path, True)

    gaus_difference = gaus_simulated_data - gaus_comparison_data.array
    output_df.setArray(gaus_simulated_data)
    gaus_comparison_data.setArray(gaus_difference)

    simulation_model : str = 'gaus'
    output_file_path = Path(output_file).stem + f"_{simulation_model}" + f".{domain}"
    pype.write_to_file(output_df, output_file_path, True)
    difference_file_path = Path(output_file).stem + f"_diff" + f"_{simulation_model}" + f".{domain}"
    pype.write_to_file(gaus_comparison_data, difference_file_path, True)

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