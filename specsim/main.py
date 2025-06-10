import sys
from itertools import zip_longest
from pathlib import Path
from typing import Callable, Any
import nmrPype as pype
from numpy import iscomplexobj, dtype, ndarray

# Command-line parsing  
from specsim import (
    parse_command_line, 
    SpecSimArgs, 
    get_dimension_info, 
    get_total_size,
    sim_exponential_1D,
    sim_gaussian_1D,
    Spectrum,
    Domain,
    optimize,
    SimulationModel, 
    get_simulation_model)
from specsim.datatypes import Vector, Phase
from specsim.optimization.optimize import OptMethod, get_optimization_method
from specsim.optimization.params import OptimizationParams

# ---------------------------------------------------------------------------- #
#                                Error Printing                                #
# ---------------------------------------------------------------------------- #

def errPrint(*args, **kwargs) -> None:
    print(*args, file=sys.stderr, **kwargs)

# ---------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #

def main(cmd_args : SpecSimArgs) -> int:
    # --------------------------- Command-line Parsing --------------------------- #

    data_frame = pype.DataFrame(cmd_args.ft2) # Spectrum nmrpype format data
    interferogram = pype.DataFrame(cmd_args.ft1) # Interferogram nmrpype format data
    data_fid = pype.DataFrame(cmd_args.fid)  # Full time-domain nmrpype format data
    output_file : str = cmd_args.out if cmd_args.out is not None else "output_file.ft1" # Output file nmrpype format data

    num_of_dimensions = cmd_args.ndim

    # Set domain and output file based on whether one is included
    file_domain : str = "ft1"                                                  # Domain of simulation data
    if output_file != None:
        suffix : str = Path(output_file).suffix.strip(".")
        output_dir : Path = Path(output_file).parent # Find parent directory of file
        output_dir.mkdir(parents=True, exist_ok=True) # Make folders for file if they don't exist

        if suffix:
            file_domain = suffix                                 # Domain of simulation data    
    else:
        output_file = f"sim.{domain}"
    
    match file_domain:
        case "fid":
            domain : Domain = Domain.FID
        case "ft1":
            domain : Domain = Domain.FT1
        case "ft2":
            domain : Domain = Domain.FT2
        case _:
            domain : Domain = Domain.FID
    
    # Select method based on input 
    model_function : Callable = get_simulation_model(cmd_args.model).value[1]
    
    # ---------------------------------------------------------------------------- #
    #                                  Simulation                                  #
    # ---------------------------------------------------------------------------- #

    # --------------------------- Simulation Parameters -------------------------- #

    spectral_widths: Vector[float] = get_dimension_info(data_frame, 'NDSW', num_of_dimensions)
    origins: Vector[float] = get_dimension_info(data_frame, 'NDORIG', num_of_dimensions)
    observation_frequencies: Vector[float] = get_dimension_info(data_frame, "NDOBS", num_of_dimensions)
    total_time_points: Vector[int] = get_total_size(data_frame, 'NDTDSIZE', num_of_dimensions)
    total_freq_points: Vector[int] = get_total_size(data_frame, 'NDFTSIZE', num_of_dimensions)
    
    xPhase : list[Phase] = [Phase(p0, p1) for p0, p1 in zip_longest(cmd_args.xP0,
                                                      cmd_args.xP1, 
                                                      fillvalue=0)]  # p0 and p1 phases of spectrum for x-axis for each element
    yPhase : list[Phase] = [Phase(p0, p1) for p0, p1 in zip_longest(cmd_args.yP0,
                                                      cmd_args.yP1, 
                                                      fillvalue=0)]  # p0 and p1 phases of spectrum for y-axis first model
    
    # Ensure xPhase and yPhase are the same length by padding with default Phase(0, 0)
    max_length : int = max(len(xPhase), len(yPhase))
    xPhase.extend([Phase(0, 0)] * (max_length - len(xPhase)))
    yPhase.extend([Phase(0, 0)] * (max_length - len(yPhase)))

    # Create a list of vectors where each element is a vector with x and y phases
    phases: list[Vector[Phase]] = [Vector([x, y]) for x, y in zip(xPhase, yPhase)]

    # Ensure scaling_factors and offsets have vector length num_of_dimensions
    if cmd_args.scale is None:
        cmd_args.scale = []
    while len(cmd_args.scale) < num_of_dimensions:
        cmd_args.scale.append(1.0)  # Default scaling factor is 1.0
    while len(cmd_args.offsets) < num_of_dimensions:
        cmd_args.offsets.append(0.0)  # Default offset is 0.0

    scaling_factors: Vector[float] = Vector(cmd_args.scale) # Time domain x and y scaling values
    offsets : Vector[float] = Vector(cmd_args.offsets) # Frequency x and y offset for simulation                             
    constant_time_region_sizes : Vector[int] = Vector([0] * num_of_dimensions) # Size of x and y constant time regions
    peaks_simulated = 0 # Number of peaks to simulate (0 for all)

    # Use verbose if both flags are used
    if cmd_args.verb and cmd_args.noverb:
        verbose = True
    else:
        verbose: bool = cmd_args.verb and not cmd_args.noverb

    if verbose:
        errPrint("Simulation Parameters:")
        errPrint(f"Spectral Widths: {spectral_widths}")
        errPrint(f"Origins: {origins}")
        errPrint(f"Observation Frequencies: {observation_frequencies}")
        errPrint(f"Total Time Points: {total_time_points}")
        errPrint(f"Total Frequency Points: {total_freq_points}")
        errPrint(f"Phases: {phases}")
        errPrint(f"Scaling Factors: {scaling_factors}")
        errPrint(f"Offsets: {offsets}")
        errPrint(f"Constant Time Region Sizes: {constant_time_region_sizes}")
        if peaks_simulated == 0:
           errPrint(f"All Peaks Simulated")
        else:
           errPrint(f"Number of Peaks Simulated: {peaks_simulated}")
        errPrint(f"Domain: {domain}")
        errPrint("")

    # ------------------------------ Spectrum Class ------------------------------ #

    spectrum = Spectrum(Path(cmd_args.tab),
                 spectral_widths,
                 origins,
                 observation_frequencies,
                 total_time_points, 
                 total_freq_points,
                 verbose)

    # ------------------------------ Run Simulation ------------------------------ #

    num_of_dimensions: int = len(spectral_widths)
    simulated_data : ndarray[Any, dtype[Any]] = spectrum.simulate(
        model_function, data_frame, data_fid, interferogram,
        None, num_of_dimensions, peaks_simulated, domain.value,
        constant_time_region_sizes, phases, offsets, scaling_factors)

    # ------------------------------ Save Simulation ----------------------------- #

    if domain == Domain.FID:
        output_df : pype.DataFrame = data_fid
    elif domain == Domain.FT1:
        output_df = interferogram
    elif domain == Domain.FT2:
        output_df = data_frame
    else:
        raise TypeError("Invalid nmrpipe data output format!")

    # Save simulated data using existing dataframe as base
    output_df.setArray(simulated_data)

    # Format output file path
    output_file_path : str = str(Path(output_file).with_suffix('')) + f"_{get_simulation_model(cmd_args.model).value[2]}" + f".{domain.to_string()}"
        
    pype.write_to_file(output_df, output_file_path, True)

    # ---------------------------------------------------------------------------- #
    #                                 Optimization                                 #
    # ---------------------------------------------------------------------------- #

    # -------------------------- Optimization Parameters ------------------------- #

    optimization_method : OptMethod = get_optimization_method(cmd_args.mode) # Optimization method


    initial_decay: list[Vector[float]] = [Vector(x_decay, y_decay) for x_decay, y_decay in zip_longest(cmd_args.initXDecay, 
                      cmd_args.initYDecay, fillvalue=0)]                                # Initial decay values for optimization
    xDecayBounds : tuple[float,float] = (cmd_args.xDecayBounds[0], cmd_args.xDecayBounds[1])
    yDecayBounds : tuple[float,float] = (cmd_args.yDecayBounds[0], cmd_args.yDecayBounds[1])
    decay_bounds: list[Vector[tuple[float, float]]] = [Vector(xDecayBounds, yDecayBounds)]  # Bounds of decays in simulation optimization         

    # Ensure the lengths of decay_bounds, initial_decay, and phases are the same
    max_length = max(len(decay_bounds), len(initial_decay), len(phases))

    # Extend each list to match the maximum length
    decay_bounds.extend([decay_bounds[0]] * (max_length - len(decay_bounds)))
    initial_decay.extend([Vector(0.0, 0.0)] * (max_length - len(initial_decay)))
    phases.extend([Vector([Phase(0, 0), Phase(0, 0)])] * (max_length - len(phases)))

    # Set the parameter count to the maximum length
    parameter_count : int = max_length

    optimization_params = OptimizationParams(
    num_of_dimensions, 
    cmd_args.trials, # Maximum number of trials
    cmd_args.step, # Step-size for step-sized based optimizations
    initial_decay,
    phases,
    decay_bounds,
    (cmd_args.ampBounds[0], cmd_args.ampBounds[1]), # Bounds of amplitude in simulation optimization
    (cmd_args.p0Bounds[0], cmd_args.p0Bounds[1]),  # Bounds of phase p0 in simulation optimization
    (cmd_args.p1Bounds[0], cmd_args.p1Bounds[1]))  # Bounds of phase p1 in simulation optimization

    if verbose:
        errPrint(optimization_params)


    # -------------------------- Simulation Optimization ------------------------- #

    target_data_frame = pype.DataFrame(cmd_args.ft2)
    target_interferogram = pype.DataFrame(cmd_args.ft1)
    target_data_fid = pype.DataFrame(cmd_args.fid)

    match domain:
        case Domain.FID:
            optimized_output = pype.DataFrame(cmd_args.fid)
        case Domain.FT1:
            optimized_output = pype.DataFrame(cmd_args.ft1)
        case Domain.FT2:
            optimized_output = pype.DataFrame(cmd_args.ft2)

    new_spectrum : Spectrum = optimize(spectrum, model_function, target_data_fid, target_interferogram, target_data_frame, domain,
                            optimization_method, optimization_params, parameter_count, offsets=offsets, 
                                  scaling_factor=scaling_factors)

    new_spectrum_data : ndarray[Any, dtype[Any]] = new_spectrum.simulate(
        model_function, target_data_frame, target_interferogram, target_data_fid,
        cmd_args.basis, num_of_dimensions, None, domain.value, None, None, offsets, scaling_factors)
    
    # Convert new spectrum data to float32, supporting both real and imaginary parts
    if iscomplexobj(new_spectrum_data):
        new_spectrum_data = new_spectrum_data.astype('complex64')  # 32-bit real and imaginary
    else:
        new_spectrum_data = new_spectrum_data.astype('float32')  # 32-bit real only
    
    # Save gaussian decay data using existing dataframe as base
    optimized_output.setArray(new_spectrum_data)

    output_file_path = str(Path(output_file).with_suffix('')) + f"_{get_simulation_model(cmd_args.model).value[2]}" + "_optimized" + f".{domain.to_string()}"
    
    pype.write_to_file(optimized_output, output_file_path, True)

    return 0

# ---------------------------------------------------------------------------- #
#                                  Entry Point                                 #
# ---------------------------------------------------------------------------- #

if __name__ == '__main__':
    try:
        sys.exit(main(SpecSimArgs(parse_command_line(sys.argv[1:]))))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)