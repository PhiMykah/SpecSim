import sys
from itertools import zip_longest
from pathlib import Path
import nmrPype as pype

# Command-line parsing  
from specsim import parse_command_line, SpecSimArgs, get_dimension_info, get_total_size

# Simulation Models
from specsim import sim_exponential_1D, sim_gaussian_1D     

# Spectrum Generation
from specsim import Spectrum

# Spectrum Optimization
from specsim import interferogram_optimization, composite_interferogram_optimization

# Model enum
from specsim import Model

# Phase class
from specsim import Phase

from specsim.debug import errPrint

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

    # Set domain and output file based on whether one is included
    domain = "ft1"                                                  # Domain of simulation data
    if output_file != None:
        suffix = Path(output_file).suffix.strip(".")
        output_dir = Path(output_file).parent # Find parent directory of file
        output_dir.mkdir(parents=True, exist_ok=True) # Make folders for file if they don't exist

        if suffix:
            domain = suffix                                 # Domain of simulation data    
    else:
        output_file = f"test_sim.{domain}"

    # Select method based on input 
    template_file = Path(output_file).stem.lower()
    simulation_model = Model.from_filename(template_file)
    
    # ---------------------------------------------------------------------------- #
    #                                  Simulation                                  #
    # ---------------------------------------------------------------------------- #

    # --------------------------- Simulation Parameters -------------------------- #

    axis_count = 2
    spectral_widths = get_dimension_info(data_frame, 'NDSW')
    origins = get_dimension_info(data_frame, 'NDORIG')
    observation_frequencies = get_dimension_info(data_frame, "NDOBS")
    total_time_points = get_total_size(data_frame, 'NDTDSIZE')
    total_freq_points = get_total_size(data_frame, 'NDFTSIZE')
    
    xPhase = [Phase(p0, p1) for p0, p1 in zip_longest(command_arguments.xP0,
                                                      command_arguments.xP1, 
                                                      fillvalue=0)]  # p0 and p1 phases of spectrum for x-axis for each element
    yPhase = [Phase(p0, p1) for p0, p1 in zip_longest(command_arguments.yP0,
                                                      command_arguments.yP1, 
                                                      fillvalue=0)]  # p0 and p1 phases of spectrum for y-axis first model
    
    # Ensure xPhase and yPhase are the same length by padding with default Phase(0, 0)
    max_length = max(len(xPhase), len(yPhase))
    xPhase.extend([Phase(0, 0)] * (max_length - len(xPhase)))
    yPhase.extend([Phase(0, 0)] * (max_length - len(yPhase)))

    phases = (xPhase, yPhase)                                               # Phase values for x-axis and y-axis
    scaling_factors = [command_arguments.scale, 1.0]                        # Time domain x and y scaling values
    offsets = [command_arguments.xOff, 0]                                   # Frequency x and y offset for simulation                             
    constant_time_region_sizes = [0,0]                                      # Size of x and y constant time regions
    peak_count = 0                                                          # Number of peaks to simulate

    # Use verbose if both flags are used
    if command_arguments.verb and command_arguments.noverb:
        verbose = True
    else:
        verbose = command_arguments.verb and not command_arguments.noverb

    if verbose:
        errPrint("Simulation Parameters:")
        errPrint(f"Spectral Widths: {spectral_widths}")
        errPrint(f"Origins: {origins}")
        errPrint(f"Observation Frequencies: {observation_frequencies}")
        errPrint(f"Total Time Points: {total_time_points}")
        errPrint(f"Total Frequency Points: {total_freq_points}")
        errPrint(f"Phases: X-{phases[0]} | Y-{phases[1]}")
        errPrint(f"Scaling Factors: {scaling_factors}")
        errPrint(f"Offsets: {offsets}")
        errPrint(f"Constant Time Region Sizes: {constant_time_region_sizes}")
        if peak_count == 0:
           errPrint(f"All Peaks Simulated")
        else:
           errPrint(f"Number of Peaks Simulated: {peak_count}")
        errPrint(f"Domain: {domain}")
        errPrint("")

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
            peak_count, domain, constant_time_region_sizes, True,
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
    if str(simulation_model).lower() not in Path(output_file).stem.lower():
        output_file_path = str(Path(output_file).with_suffix('')) + f"_{str(simulation_model)}" + f".{domain}"
    else:
        output_file_path = str(Path(output_file).with_suffix('')) + f".{domain}"
        
    pype.write_to_file(output_df, output_file_path, True)

    # ---------------------------------------------------------------------------- #
    #                                 Optimization                                 #
    # ---------------------------------------------------------------------------- #

    if domain != 'ft1':
        return
    
    # -------------------------- Optimization Parameters ------------------------- #

    optimization_method = command_arguments.mode                                    # Optimization method
    trial_count = command_arguments.trials                                          # Maximum number of trials
    trial_step_size = command_arguments.step                                        # Step-size for step-sized based optimizations

    # if not (len(command_arguments.initXDecay) == len(command_arguments.initYDecay) == parameter_count):
    #     raise ValueError("initXDecay and initYDecay must have the same length and match additional phase")
    
    initial_decay = [(x_decay, y_decay) for x_decay, y_decay in zip_longest(command_arguments.initXDecay, 
                      command_arguments.initYDecay, fillvalue=0)]                                # Initial decay values for optimization
    decay_bounds = [tuple(command_arguments.xDecayBounds),                          # Bounds of x-decays in simulation optimization
                    tuple(command_arguments.yDecayBounds)]                          # Bounds of y-decays in simulation optimization
    amplitude_bounds = tuple(command_arguments.ampBounds)                           # Bounds of amplitude in simulation optimization
    phase_bounds = [tuple(command_arguments.p0Bounds),                              # Bounds of phase p0 in simulation optimization
                    tuple(command_arguments.p1Bounds)]                              # Bounds of phase p1 in simulation optimization
    
    optimization_params = {
        "trials" : trial_count,                         # number of trials to perform
        "step" : trial_step_size,                       # step-size of optimization
        "initDecay" : initial_decay[0],                    # tuple of initial decay values for optimization in Hz
        "initPhase" : phases,                           # tuple of initial phase values for optimiation in Hz
        "dxBounds" : decay_bounds[0],                   # tuple of lower and upper bounds for x-axis decay in Hz
        "dyBounds" : decay_bounds[1],                   # tuple of lower and upper bounds for y-axis decay in Hz
        "aBounds" : amplitude_bounds,                   # tuple of lower and upper bounds for decay
        "p0Bounds" : phase_bounds[0],                   # tuple of lower and upper bounds for phase P0 in degrees
        "p1Bounds" : phase_bounds[1],                   # tuple of lower and upper bounds for phase P0 in degrees
    }

    if verbose:
        errPrint("Optimization Parameters:")
        errPrint(f"Optimization Method: {optimization_method}")
        errPrint(f"Optimization Model: {simulation_model}")
        errPrint(f"Trial Count: {trial_count}")
        errPrint(f"Trial Step Size: {trial_step_size}")
        errPrint(f"Initial Decay: {initial_decay[0]}")
        errPrint(f"Initial Phase:  X-{phases[0]} | Y-{phases[1]}")
        errPrint(f"x-Decay Bounds: {decay_bounds[0]}")
        errPrint(f"y-Decay Bounds: {decay_bounds[1]}")
        errPrint(f"Amplitude Bounds: {amplitude_bounds}")
        errPrint(f"Phase P0 Bounds: {phase_bounds[0]}")
        errPrint(f"Phase P1 Bounds: {phase_bounds[1]}")
        errPrint("")

    # -------------------------- Simulation Optimization ------------------------- #

    target_data = pype.DataFrame(command_arguments.ft1)
    target_data_fid = pype.DataFrame(command_arguments.fid)
    optimized_output = pype.DataFrame(command_arguments.ft1)

    if simulation_model == Model.COMPOSITE:
        optimization_params["initDecay"] = initial_decay                 # tuple of initial decay values for optimization in Hz
        new_spectrum = composite_interferogram_optimization(
            test_spectrum, model_function, target_data_fid, target_data, optimization_method,
            optimization_params, constant_time_region_sizes=constant_time_region_sizes,
                                                offsets=offsets, scaling_factors=scaling_factors)
    else:
        new_spectrum = interferogram_optimization(test_spectrum, model_function, target_data_fid, target_data, optimization_method,
                                                optimization_params, constant_time_region_sizes=constant_time_region_sizes,
                                                offsets=offsets, scaling_factors=scaling_factors)
    new_spectrum_data = new_spectrum.spectral_simulation(model_function, optimized_output, data_fid, axis_count,
                                                             peak_count, domain, constant_time_region_sizes, False,
                                                             offsets=offsets, scaling_factors=scaling_factors)
    
    # Save gaussian decay data using existing dataframe as base
    optimized_output.setArray(new_spectrum_data)

    if str(simulation_model).lower() not in Path(output_file).stem.lower():
        output_file_path = str(Path(output_file).with_suffix('')) + f"_{str(simulation_model)}" + "_optimized" + f".{domain}"
    else:
        output_file_path = str(Path(output_file).with_suffix('')) + "_optimized" +  f".{domain}"

    if command_arguments.basis:
        new_spectrum.create_basis_set(command_arguments.basis, target_data_fid, target_data, 
                                      data_frame, model_function, offsets, scaling_factors, domain)
        
    

# ---------------------------------------------------------------------------- #
#                                  Entry Point                                 #
# ---------------------------------------------------------------------------- #

if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)