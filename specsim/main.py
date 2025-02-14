import sys
from pathlib import Path
import nmrPype as pype

# Command-line parsing  
from specsim import parse_command_line, SpecSimArgs, get_dimension_info, get_total_size

# Simulation Models
from specsim import sim_exponential_1D, sim_gaussian_1D     

# Spectrum Generation
from specsim import Spectrum

# ---------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #

def main() -> int:
    # --------------------------- Command-line Parsing --------------------------- #

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

    test_spectrum = Spectrum(Path(command_arguments.tab),
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

    # ------------------------------ Save Simulation ----------------------------- #

    if domain == "fid":
        output_df = data_fid
    elif domain == "ft1":
        output_df = interferogram
    elif domain == 'ft2':
        output_df = data_frame
    else:
        raise TypeError("Invalid nmrpipe data output format!")

    # Save exponential decay data using existing dataframe as base
    output_df.setArray(exp_simulated_data)

    # Format output file path for exponential decay
    simulation_model : str = 'ex'
    output_file_path = Path(output_file).stem + f"_{simulation_model}" + f".{domain}"
    pype.write_to_file(output_df, output_file_path, True)

    # Save gaussian decay data using existing dataframe as base
    output_df.setArray(gaus_simulated_data)

    simulation_model : str = 'gaus'
    output_file_path = Path(output_file).stem + f"_{simulation_model}" + f".{domain}"
    pype.write_to_file(output_df, output_file_path, True)

# ---------------------------------------------------------------------------- #
#                                  Entry Point                                 #
# ---------------------------------------------------------------------------- #

if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)