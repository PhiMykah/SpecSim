import sys
from pathlib import Path
import nmrPype as pype

# Command-line parsing  
from specsim import parse_command_line, SpecSimArgs, get_dimension_info, get_total_size

# Simulation Models
from specsim import sim_exponential_1D, sim_gaussian_1D     

# Spectrum Generation
from specsim import Spectrum

# Spectrum Optimization
from specsim import interferogram_optimization

# Model enum
from specsim import Model

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
    xPhase = (command_arguments.xP0, command_arguments.xP1)     # p0 and p1 phases of spectrum for x-axis
    yPhase = (command_arguments.yP0, command_arguments.yP1)     # p0 and p1 phases of spectrum for y-axis
    phases = [xPhase, yPhase]                                   # Phase values for x-axis and y-axis
    scaling_factors = [command_arguments.scale, 1.0]            # Time domain x and y scaling values
    offsets = [command_arguments.xOff, 0]                       # Frequency x and y offset for simulation                             
    constant_time_region_sizes = [0,0]                          # Size of x and y constant time regions
    peak_count = 0                                              # Number of peaks to simulate
    domain = "ft1"                                              # Domain of simulation data

    # Use verbose if both flags are used
    if command_arguments.verb and command_arguments.noverb:
        verbose = True
    else:
        verbose = command_arguments.verb and not command_arguments.noverb

    if verbose:
        print("Simulation Parameters:", file=sys.stderr)
        print(f"Spectral Widths: {spectral_widths}", file=sys.stderr)
        print(f"Origins: {origins}", file=sys.stderr)
        print(f"Observation Frequencies: {observation_frequencies}", file=sys.stderr)
        print(f"Total Time Points: {total_time_points}", file=sys.stderr)
        print(f"Total Frequency Points: {total_freq_points}", file=sys.stderr)
        print(f"Phases: {phases}", file=sys.stderr)
        print(f"Scaling Factors: {scaling_factors}", file=sys.stderr)
        print(f"Offsets: {offsets}", file=sys.stderr)
        print(f"Constant Time Region Sizes: {constant_time_region_sizes}", file=sys.stderr)
        if peak_count == 0:
           print(f"All Peaks Simulated", file=sys.stderr)
        else:
           print(f"Number of Peaks Simulated: {peak_count}", file=sys.stderr)
        print(f"Domain: {domain}", file=sys.stderr)
        print("", file=sys.stderr)
    # -------------------------- Optimization Parameters ------------------------- #

    optimization_method = command_arguments.mode                                    # Optimization method
    trial_count = command_arguments.trials                                          # Maximum number of trials
    trial_step_size = command_arguments.step                                        # Step-size for step-sized based optimizations
    initial_decay = (command_arguments.initXDecay, command_arguments.initYDecay)    # Initial decay values for optimization
    decay_bounds = [(command_arguments.xDecayLB, command_arguments.xDecayUB),       # Bounds of decays in simulation optimization
                    (command_arguments.yDecayLB, command_arguments.yDecayUB)]       # Bounds of decays in simulation optimization
    amplitude_bounds = (command_arguments.ampLB, command_arguments.ampUB)           # Bounds of amplitude in simulation optimization

    optimization_params = {
        "trials" : trial_count,
        "step" : trial_step_size,
        "initDecay" : initial_decay,
        "dxBounds" : decay_bounds[0],
        "dyBounds" : decay_bounds[1],
        "aBounds" : amplitude_bounds,
    }

    # Set domain and output file based on whether one is included
    if output_file != None:
        suffix = Path(output_file).suffix.strip(".")
        output_dir = Path(output_file).parent # Find parent directory of file
        output_dir.mkdir(parents=True, exist_ok=True) # Make folders for file if they don't exist

        if suffix:
            domain = suffix                                 # Domain of simulation data    
    else:
        output_file = f"test_sim.{domain}"

    # Select method based on input 
    template_file = Path(command_arguments.ft1).stem.lower()
    simulation_model = Model.from_filename(template_file)

    if verbose:
        print("Optimization Parameters:", file=sys.stderr)
        print(f"Optimization Method: {optimization_method}", file=sys.stderr)
        print(f"Optimization Model: {simulation_model}", file=sys.stderr)
        print(f"Trial Count: {trial_count}", file=sys.stderr)
        print(f"Trial Step Size: {trial_step_size}", file=sys.stderr)
        print(f"Initial Decay: {initial_decay}", file=sys.stderr)
        print(f"x-Decay Bounds: {decay_bounds[0]}", file=sys.stderr)
        print(f"y-Decay Bounds: {decay_bounds[1]}", file=sys.stderr)
        print(f"Amplitude Bounds: {amplitude_bounds}", file=sys.stderr)
        print("", file=sys.stderr)

    # ------------------------------ Spectrum Class ------------------------------ #

    test_spectrum = Spectrum(Path(command_arguments.tab),
                 spectral_widths,
                 origins,
                 observation_frequencies,
                 total_time_points, 
                 total_freq_points,
                 verbose)

    # ------------------------------ Run Simulation ------------------------------ #

    model_function : callable = Model.function(simulation_model)
    
    simulated_data = test_spectrum.spectral_simulation(
            model_function, data_frame, data_fid, axis_count,
            peak_count, domain, constant_time_region_sizes,
            phases, offsets, scaling_factors
        )

    # ------------------------------ Save Simulation ----------------------------- #

    if domain == "fid":
        output_df = data_fid
    elif domain == "ft1":
        output_df = interferogram
    elif domain == 'ft2':
        output_df = data_frame
    else:
        raise TypeError("Invalid nmrpipe data output format!")

    # Save simulated data using existing dataframe as base
    output_df.setArray(simulated_data)

    # Format output file path
    output_file_path = str(Path(output_file).with_suffix('')) + f"_{str(simulation_model)}" + f".{domain}"
    pype.write_to_file(output_df, output_file_path, True)

    # -------------------------- Simulation Optimization ------------------------- #

    if domain != 'ft1':
        return

    target_data = pype.DataFrame(command_arguments.ft1)
    target_data_fid = pype.DataFrame(command_arguments.fid)
    optimized_output = pype.DataFrame(command_arguments.ft1)
    sim_params = (constant_time_region_sizes, phases, offsets, scaling_factors)
    new_spectrum = interferogram_optimization(test_spectrum, model_function, target_data_fid, target_data, optimization_method, sim_params, optimization_params)
    new_spectrum_data = new_spectrum.spectral_simulation(model_function, optimized_output, data_fid, axis_count,
                                                             peak_count, domain, constant_time_region_sizes,
                                                             phases, offsets, scaling_factors)
    
    # Save gaussian decay data using existing dataframe as base
    optimized_output.setArray(new_spectrum_data)

    output_file_path = str(Path(output_file).with_suffix('')) + "_optimized" + f"_{str(simulation_model)}" + f".{domain}"
    pype.write_to_file(optimized_output, output_file_path, True)
    

# ---------------------------------------------------------------------------- #
#                                  Entry Point                                 #
# ---------------------------------------------------------------------------- #

if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)