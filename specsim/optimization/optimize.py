from typing import Callable, Any, Literal

from pytest import param

from ..spectrum.models.composite import sim_composite_1D
from ..spectrum.models.exponential import sim_exponential_1D
from ..spectrum import Domain, ModelFunction, Spectrum
from ..datatypes import Vector
from .params import OptimizationParams
import nmrPype as pype
from enum import Enum
from sys import stderr
import numpy as np
from scipy.optimize import basinhopping, minimize, brute, least_squares, dual_annealing
import copy
from matplotlib import pyplot as plt

class OptMethod(Enum):
    LSQ = 0, 'lsq'
    BASIN = 1, 'basin'
    MIN = 2, 'minimize'
    BRUTE = 3, 'brute'
    DANNEAL = 4, 'danneal'

def get_optimization_method(method : str) -> OptMethod:
    match method:
        case 'lsq':
            return OptMethod.LSQ
        case 'basin':
            return OptMethod.BASIN
        case 'minimize':
            return OptMethod.MIN
        case 'min':
            return OptMethod.MIN
        case 'brute':
            return OptMethod.BRUTE
        case 'danneal':
            return OptMethod.DANNEAL
        case _:
            return OptMethod.LSQ

def optimize(input_spectrum : Spectrum, 
                 model_function : ModelFunction, 
                 fid : pype.DataFrame, 
                 interferogram : pype.DataFrame,
                 data_frame_spectrum : pype.DataFrame,
                 domain : Domain = Domain.FT1,
                 method : str | OptMethod = OptMethod.LSQ,
                 opt_params : OptimizationParams = OptimizationParams(2),
                 parameter_count : int = 1,
                 use_deco : bool = False,
                 **sim_params) -> Spectrum:
    """
    Optimize a Spectrum peak table to match the original spectral data without window functions

    Parameters
    ----------
    input_spectrum : Spectrum
        Starting spectrum to optimize
    model_function : ModelFunction
        Spectral simulation decay model function (e.g exponential, gaussian, composite)
    fid : pype.DataFrame
        Original spectrum time domain data
    interferogram : pype.DataFrame
        Original spectral interferogram data 
    data_frame_spectrum : pype.DataFrame
        Original spectrum data frequency domain data
    domain : Domain
        Domain to model, by default Domain.FT1
    method : str | OptMethod
        Method of optimization (lsq, basin, minimize, brute), by default lsq
    opt_params : OptimizationParams
        Parameters for the optimization, by default OptimizationParams(2)
    parameter_count : int
        Number of linewidths and phases in simulation (used for composite models), by default 1
    use_deco : bool
        Whether or not to defer height calculations to decomposition, by default False
    sim_params : **kwargs
        Additional simulation parameters
    
    Returns
    -------
    Spectrum
        Optimized Spectrum with new peak table
    """
    # Convert method str to enum
    if isinstance(method, str):
        method = get_optimization_method(method)

    # Ensure initial phase values are within bounds
    for ip in opt_params.initial_phase:
        for phase in ip:
            if not (opt_params.p0_bounds[0] <= phase.p0 <= opt_params.p0_bounds[1]):
                print(f"Warning: Initial phase p0 ({phase.p0}) outside of initial bounds. Adjusting to midpoint.", file=stderr) 
                phase.p0 = (opt_params.p0_bounds[0] + opt_params.p0_bounds[1]) / 2
            if not (opt_params.p1_bounds[0] <= phase.p1 <= opt_params.p1_bounds[1]):
                print(f"Warning: Initial phase p1 ({phase.p1}) outside of initial bounds. Adjusting to midpoint.", file=stderr) 
                phase.p1 = (opt_params.p1_bounds[0] + opt_params.p1_bounds[1]) / 2

    if parameter_count < 1:
        parameter_count = 1
    if model_function == sim_exponential_1D or model_function == sim_composite_1D:
        parameter_count = 1
    if model_function == sim_composite_1D:
        parameter_count = 2

    # ---------------------------------------------------------------------------- #
    #                             Collect Initial Guess                            #
    # ---------------------------------------------------------------------------- #

    # ----------------------------------- NOTE ----------------------------------- #
    # NOTE: Modify initial peak values and bounds later on, currently changed as needed

    spectral_widths : Vector[float] = input_spectrum._spectral_widths
    time_size_pts : Vector[int] = input_spectrum._total_time_points
    peak_count : int = len(input_spectrum.peaks)
    num_of_dimensions : int = len(spectral_widths)

    # ------------------------------- Initial Decay ------------------------------ #
    # Ensure the initial decays and spectrum dimensions match
    if not num_of_dimensions == len(opt_params.initial_decay[0]):
        raise ValueError(f"Number of initial decay dimensions {len(opt_params.initial_decay[0])} " \
                         f"does not match Spectrum dimensions {num_of_dimensions}!")
    
    # Collect the initial decay values
    initial_decay : list[float] = []
    for j in range(parameter_count):
        for i in range(num_of_dimensions):
            try:
                pts_value: float = (opt_params.initial_decay[j][i]/spectral_widths[i]) * time_size_pts[i]
            except IndexError:
                pts_value: float = (opt_params.initial_decay[0][i]/spectral_widths[i]) * time_size_pts[i]
            initial_decay += [pts_value] * peak_count

    # ------------------------------- Decay Bounds ------------------------------- #

    # Ensure the decay bounds and spectrum dimensions match
    if not num_of_dimensions == len(opt_params.bounds[0]):
        raise ValueError(f"Number of initial decay dimensions {len(opt_params.bounds[0])} " \
                         f"does not match Spectrum dimensions {num_of_dimensions}!")
    
    # Collect the initial bounds
    lower_decay_bounds : list[float] = []
    upper_decay_bounds : list[float] = []
    bounds_pair : list[tuple[float, float]]  = []

    for j in range(parameter_count):
        for i in range(num_of_dimensions):
            try:
                lower: float = (opt_params.bounds[j][i][0]/spectral_widths[i]) * time_size_pts[i]
                upper: float = (opt_params.bounds[j][i][1]/spectral_widths[i]) * time_size_pts[i]
            except IndexError:
                lower: float = (opt_params.bounds[0][i][0]/spectral_widths[i]) * time_size_pts[i]
                upper: float = (opt_params.bounds[0][i][1]/spectral_widths[i]) * time_size_pts[i]
            lower_decay_bounds += [lower] * peak_count
            upper_decay_bounds += [upper] * peak_count
            bounds_pair += [(lower, upper)] * peak_count

    # ------------------------------- Initial Phase ------------------------------ #

    # Ensure the initial phase and spectrum dimensions match
    if not num_of_dimensions == len(opt_params.initial_phase[0]):
        raise ValueError(f"Number of initial decay dimensions {len(opt_params.initial_phase[0])} " \
                         f"does not match Spectrum dimensions {num_of_dimensions}!")

    # Collect the initial phase
    initial_phase : list[float] = []

    for j in range(parameter_count):
        for i in range(num_of_dimensions):
            try:
                initial_phase += [opt_params.initial_phase[j][i].p0] * peak_count
                initial_phase += [opt_params.initial_phase[j][i].p1] * peak_count
            except:
                initial_phase += [opt_params.initial_phase[0][i].p0] * peak_count
                initial_phase += [opt_params.initial_phase[0][i].p1] * peak_count

    # ------------------------------- Phase Bounds ------------------------------- #
    
    # Collect the initial phase bounds
    lower_p0_bounds : list[float] = [opt_params.p0_bounds[0]] * peak_count
    upper_p0_bounds : list[float] = [opt_params.p0_bounds[1]] * peak_count
    p0_bounds_pair : list[tuple[float, float]] = [opt_params.p0_bounds] * peak_count

    # Ensure the decay bounds and spectrum dimensions match
    if not len(spectral_widths) == len(opt_params.p1_bounds):
        raise ValueError(f"Number of initial decay dimensions {len(opt_params.p1_bounds)} " \
                         f"does not match Spectrum dimensions {num_of_dimensions}!")
    
    # Collect the initial phase bounds
    lower_p1_bounds : list[float] = [opt_params.p1_bounds[0]] * peak_count
    upper_p1_bounds : list[float] = [opt_params.p1_bounds[1]] * peak_count
    p1_bounds_pair : list[tuple[float, float]] = [opt_params.p1_bounds] * peak_count

    if not use_deco:
        # --------------------------------- Amplitude -------------------------------- #

        # Collect the initial peak heights
        initial_peak_heights : list[float] = [peak.intensity for peak in input_spectrum.peaks]

        # ----------------------------- Amplitude Bounds ----------------------------- #

        # Find the largest and smallest peak heights in the initial guess
        min_peak_height : float = min(initial_peak_heights)
        max_peak_height : float = max(initial_peak_heights)

        amplitude_bounds : tuple[float, float] = opt_params.amplitude_bounds

        # Adjust the height bounds if they escape
        if min_peak_height < amplitude_bounds[0]:
            amplitude_bounds = (2 * min_peak_height if min_peak_height < 0 else min_peak_height / 2, amplitude_bounds[1])
        if max_peak_height > amplitude_bounds[1]:
            amplitude_bounds = (amplitude_bounds[0], max_peak_height * 2 if max_peak_height > 0 else max_peak_height / 2)

        amplitude_low_bounds : list[float] = [amplitude_bounds[0]] * peak_count
        amplitude_high_bounds : list[float] =  [amplitude_bounds[1]] * peak_count
        amplitude_bounds_pair : list[tuple[float, float]] = [amplitude_bounds] * peak_count
    else:
        input_spectrum.deco_coefficients = [peak.intensity for peak in input_spectrum.peaks]
        initial_peak_heights = []
        amplitude_low_bounds = []
        amplitude_high_bounds = []
        amplitude_bounds_pair = []

    # ---------------------------------- Weights --------------------------------- #

    initial_weight : list[float] = []
    for j in range(parameter_count):
        for i in range(num_of_dimensions):
            try:
                initial_weight += [opt_params.initial_weight[j][i]] * peak_count
            except IndexError:
                initial_weight += [opt_params.initial_weight[0][i]] * peak_count

    # ------------------------------- Weight Bounds ------------------------------ #

    weight_low_bounds : list[float] = [0.0] * peak_count  * num_of_dimensions * parameter_count

    weight_high_bounds : list[float] = [1.0] * peak_count  * num_of_dimensions * parameter_count

    weight_bounds_pair : list[tuple[float,float]] = [(0.0, 1.0)] * peak_count  * num_of_dimensions * parameter_count

    # -------------------------------- Set Bounds -------------------------------- #
    if method == OptMethod.LSQ:
        lower_bounds : list[float] = lower_decay_bounds \
                     + ((lower_p0_bounds + lower_p1_bounds) * num_of_dimensions) * parameter_count \
                     + amplitude_low_bounds \
                     + weight_low_bounds
        
        upper_bounds : list[float] = upper_decay_bounds \
                     + ((upper_p0_bounds + upper_p1_bounds) * num_of_dimensions) * parameter_count \
                     + amplitude_high_bounds \
                     + weight_high_bounds
        
        optimization_bounds : Any = (lower_bounds, upper_bounds)

    else:

        optimization_bounds : Any = bounds_pair \
                            + ((p0_bounds_pair + p1_bounds_pair) * num_of_dimensions * parameter_count ) \
                            + amplitude_bounds_pair \
                            + weight_bounds_pair

    # --------------------------- Additional Parameters -------------------------- #

    offsets : Vector[float] | None = sim_params.get("offsets")
    constant_time_region_sizes : Vector[int] | None = sim_params.get("constant_time_region_sizes")
    scaling_factors : Vector[float] | None = sim_params.get("scaling_factors")

    # ----------------------------- Run Optimization ----------------------------- #

    initial_params: np.ndarray[Any, np.dtype[np.float32]] = np.concatenate((
        initial_decay,
        initial_phase,
        initial_peak_heights,
        initial_weight
    ))

    optimization_args : tuple = (num_of_dimensions, domain, input_spectrum, model_function, 
                         fid, interferogram, data_frame_spectrum, None, 
                         offsets, constant_time_region_sizes, scaling_factors, parameter_count)
    
    obj_func : Callable = basis_objective_function if use_deco else objective_function

    match method:
        case OptMethod.LSQ:
            verbose: Literal[2] | Literal[0] = 2 if input_spectrum.verbose else 0
            difference_equation : Callable[..., Any] = lambda target, simulated: (target - simulated).flatten()
            optimization_args = (num_of_dimensions, domain, input_spectrum, model_function, 
                                 fid, interferogram, data_frame_spectrum, difference_equation, 
                                 offsets, constant_time_region_sizes, scaling_factors, parameter_count)
            result : Any = least_squares(obj_func, initial_params, '2-point',
                                   optimization_bounds, 'trf', args=optimization_args, 
                                   verbose=verbose, max_nfev=opt_params.trials)
            optimized_params : np.ndarray[Any, np.dtype[np.float32]] = result.x
        case OptMethod.BASIN:
            disp : bool = True if input_spectrum.verbose else False
            result = basinhopping(obj_func, initial_params, niter=opt_params.trials,
                                  stepsize=opt_params.step_size, 
                                  minimizer_kwargs={"method": "L-BFGS-B", "args":optimization_args},
                                  disp=disp)
            optimized_params = result.x
        case OptMethod.MIN:
            disp : bool = True if input_spectrum.verbose else False
            result = minimize(obj_func, initial_params, args=optimization_args,
                              method='SLSQP', options={"disp":disp})
            optimized_params = result.x
        case OptMethod.DANNEAL:
            result = dual_annealing(obj_func, optimization_bounds, optimization_args, maxiter=opt_params.trials)
            optimized_params = result.x
        case _:
            disp : bool = True if input_spectrum.verbose else False
            x0, fval, grid, Jout = brute(obj_func, optimization_bounds, args=optimization_args,
                                         Ns=20, full_output=True, workers=-1, disp=disp)
            optimized_params = x0

    # Unpack optimized params
    peak_count = len(input_spectrum.peaks)

    optimized_spectrum : Spectrum = copy.deepcopy(input_spectrum)

    if not use_deco:
        optimized_decays, optimized_phases, optimized_peak_heights, optimized_weights = unpack_params(peak_count, num_of_dimensions, optimized_params, parameter_count)

        optimized_spectrum.update_peaks(optimized_decays, optimized_phases, optimized_weights, parameter_count, num_of_dimensions, optimized_peak_heights)
    else:
        optimized_decays, optimized_phases, optimized_weights = unpack_deco_params(peak_count, num_of_dimensions, optimized_params, parameter_count)

        optimized_spectrum.update_peaks(optimized_decays, optimized_phases, optimized_weights, parameter_count, num_of_dimensions)

    return optimized_spectrum



def objective_function(params : np.ndarray | list, 
                       num_of_dimensions : int, 
                       domain : Domain,
                       input_spectrum : Spectrum, 
                       model_function : ModelFunction, 
                       fid : pype.DataFrame, 
                       interferogram : pype.DataFrame,
                       data_frame_spectrum : pype.DataFrame,
                       difference_equation : Callable | None = None,
                       offsets : Vector[float] | None = None,
                       constant_time_region_sizes : Vector[int] | None = None,
                       scaling_factors : Vector[float] | None = None,
                       unpack_count : int = 1) -> float:
    """
    Function used for optimizing peak linewidths and heights of spectrum

    Parameters
    ----------
    params : np.ndarray
        List of target parameters being tested (peak decays, peak linewidths, peak heights)
    num_of_dimensions : int
        Number of dimensions to simulate
    input_spectrum : Spectrum
        Input spectrum to generate from 
    model_function : ModelFunction
        Spectral modeling method function (e.g. exponential, gaussian)
    fid : pype.DataFrame
        Original spectrum time domain data
    interferogram : pype.DataFrame
        Original spectral interferogram data 
    data_frame_spectrum : pype.DataFrame
        Original spectrum data frequency domain data
    difference_equation : Callable, optional
        Equation used to calculate difference between both arrays
    offsets : Vector[float] | None, optional
        Offset values of the frequency domain in points for each dimension, 
            by default None
    scaling_factors : Vector[float] | None, optional
        Simulation time domain data scaling factor for each dimension, 
            by default None
    unpack_count : int
        Number of linewidths and phases in simulation (used for composite models), by default 1
    """
    peak_count: int = len(input_spectrum.peaks)

    decay_list, phase_list, peak_height_list, weights = unpack_params(peak_count, num_of_dimensions, params, unpack_count)

    input_spectrum.update_peaks(decay_list, phase_list, weights, unpack_count, num_of_dimensions, peak_height_list)

    simulation : np.ndarray[Any, np.dtype[Any]] = input_spectrum.simulate(model_function, data_frame_spectrum, interferogram,
                                         fid, None, num_of_dimensions, None, domain.value,
                                         constant_time_region_sizes, None, offsets, scaling_factors)
    
    match domain:
        case Domain.FID:
            target : pype.DataFrame = fid
        case Domain.FT1:
            target : pype.DataFrame = interferogram
        case Domain.FT2:
            target : pype.DataFrame = data_frame_spectrum

    if difference_equation is None or not callable(difference_equation):
        difference : Any = np.sum((target.array - simulation) ** 2)
    else:
        difference : Any = difference_equation(target.array, simulation)

    print(f"\rCurrent difference: {difference}")
    return difference

def basis_objective_function(params : np.ndarray | list, 
                       num_of_dimensions : int, 
                       domain : Domain,
                       input_spectrum : Spectrum, 
                       model_function : ModelFunction, 
                       fid : pype.DataFrame, 
                       interferogram : pype.DataFrame,
                       data_frame_spectrum : pype.DataFrame,
                       difference_equation : Callable | None = None,
                       offsets : Vector[float] | None = None,
                       constant_time_region_sizes : Vector[int] | None = None,
                       scaling_factors : Vector[float] | None = None,
                       unpack_count : int = 1) -> float:
    peak_count: int = len(input_spectrum.peaks)

    decay_list, phase_list, weights = unpack_deco_params(peak_count, num_of_dimensions, params, unpack_count)

    input_spectrum.update_peaks(decay_list, phase_list, weights, unpack_count, num_of_dimensions)

    bases = input_spectrum.simulate_basis(model_function, data_frame_spectrum, interferogram, fid, num_of_dimensions, None, domain.value,
                                          constant_time_region_sizes, None, offsets, scaling_factors)
    
    # for i, basis in enumerate(bases):
    #     plt.figure(figsize=(8, 6))
    #     plt.contourf(basis, levels=50, cmap='viridis')
    #     plt.colorbar(label='Intensity')
    #     plt.title(f'Basis {i+1}')
    #     plt.xlabel('Dimension 1')
    #     plt.ylabel('Dimension 2')
    #     plt.show()

    match domain:
        case Domain.FID:
            target : pype.DataFrame = fid
        case Domain.FT1:
            target : pype.DataFrame = interferogram
        case Domain.FT2:
            target : pype.DataFrame = data_frame_spectrum

    synthetic_data, coefficients = pype.Deco.runDeco(target, bases, multiprocessing=True)

    # Normalize coefficients
    coeff_norm : np.floating[Any] = np.linalg.norm(coefficients)
    coefficients = coefficients / coeff_norm
    input_spectrum.deco_coefficients = list(coefficients)

    if difference_equation is None or not callable(difference_equation):
        difference : Any = np.sum((target.array - synthetic_data) ** 2)
    else:
        difference : Any = difference_equation(target.array, synthetic_data)
        
    return difference
    
def unpack_params(peak_count : int, num_of_dimensions : int, params : Any, unpack_count : int = 1) -> tuple[list[float],list[float],list[float],list[float]]:
    """
    Helper function to unpack parameters

    Parameters
    ----------
    peak_count : int
        Number of peaks in the spectrum
    num_of_dimensions : int
        Number of dimensions simulated in the spectrum
    params : list
        Parameters to unpack
    unpack_count : int 
        Number of iterations to unpack

    Returns
    -------
    tuple[list, list, list, list]
        Retunrs the decay_list, phase_list, peak_height_list, and weights

    """
    decay_range : int = peak_count * num_of_dimensions * unpack_count
    decay_list : list[float] = params[:decay_range]

    phase_range : int = decay_range + (peak_count * 2 * num_of_dimensions * unpack_count)
    phase_list : list[float] = params[decay_range:phase_range]
    
    peak_height_list : list[float] = params[phase_range:phase_range+peak_count]

    weights : list[float] = params[phase_range+peak_count:]

    return decay_list, phase_list, peak_height_list, weights
    
def unpack_deco_params(peak_count : int, num_of_dimensions : int, params : Any, unpack_count : int = 1) -> tuple[list[float],list[float],list[float]]:
    """
    Helper function to unpack parameters

    Parameters
    ----------
    peak_count : int
        Number of peaks in the spectrum
    num_of_dimensions : int
        Number of dimensions simulated in the spectrum
    params : list
        Parameters to unpack
    unpack_count : int 
        Number of iterations to unpack

    Returns
    -------
    tuple[list,list,list]
        Returns the decay_list, phase_list, and weights
    """
    decay_range : int = peak_count * num_of_dimensions * unpack_count
    decay_list : list[float] = params[:decay_range]

    phase_range : int = decay_range + (peak_count * 2 * num_of_dimensions * unpack_count)
    phase_list : list[float] = params[decay_range:phase_range]

    weights : list[float] = params[phase_range:]

    return decay_list, phase_list, weights