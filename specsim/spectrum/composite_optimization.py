from .spectrum import Spectrum, ModelFunction
import nmrPype as pype
from ..peak import Peak, Coordinate, Coordinate2D, Phase
from ..debug.verbose import errPrint
import numpy as np
from scipy.optimize import basinhopping, minimize, brute, least_squares

def composite_interferogram_optimization(input_spectrum: Spectrum, composite_function: ModelFunction, input_fid : pype.DataFrame,
                               target_interferogram: pype.DataFrame, method: str = 'lsq', options: dict = {}, 
                               **sim_params) -> Spectrum:
    """
    Optimize a Spectrum peak table to match the original interferogram data without window functions

    Parameters
    ----------
    input_spectrum : Spectrum
        Starting spectrum to optimize
    model_function : ModelFunction
        Spectral simulation decay model function (e.g exponential, gaussian)
    target_interferogram : pype.DataFrame
        Target interferogram to optimize peak table towards
    method : str
        Method of optimization, (lsq, basin, minimize, brute)
    trial_count : int
        Number of trials to run for optimization
    options : dict
        Optimization parameters, by default {}
        --------------------------------------
        "trials" : int
            number of trials to perform
        "step" : float
            step-size of optimization
        "initDecay" : list[tuple[float, float]]
            tuple of initial decay values for optimization in Hz
        "initPhase" : list[list[Phase, Phase]]
            tuple of initial phase values for optimiation in Hz
        "dXBounds": tuple[float, float]
            tuple of lower and upper bounds for x-axis decay in Hz
        "dYBounds": tuple[float, float]
            tuple of lower and upper bounds for y-axis decay in Hz
        "aBounds": tuple[float, float]
            tuple of lower and upper bounds for decay
        "p0Bounds": list[tuple[float, float]]
            tuple of lower and upper bounds for phase P0 in degrees
        "p1Bounds": list[tuple[float, float]]
            tuple of lower and upper bounds for phase P0 in degrees
    sim_params : dict
        Spectral simulation function extra parameters
    Returns
    -------
    Spectrum
        Optimized Spectrum with new peak table
    """
    OPT_METHODS = ['lsq', 'basin', 'minimize', 'brute']
    method = method.lower()
    if method == '' or method not in OPT_METHODS:
        method = 'brute'

    # ----------------------------- Initial Settings ----------------------------- #

    trial_count : int = options.get("trials", 100)
    step_size : float = options.get("step", 0.1)
    initial_decay : list[tuple[float, float]] = options.get("initDecay", [(15, 5), (15,5)])
    initial_phase : list[list[Phase, Phase]] = list(options.get("initPhase", [[Phase(0,0), Phase(0,0)],[Phase(0,0), Phase(0,0)]]))
    decay_bounds_x : tuple[float, float] = options.get("dxBounds", (0, 100))
    decay_bounds_y : tuple[float, float] = options.get("dyBounds", (0, 20))
    amplitude_bounds : tuple[float, float] = options.get("aBounds", (0, 150))
    p0_bounds : list[tuple[float, float]] = options.get("p0Bounds", [(-180, 180), (-180, 180)])
    p1_bounds : list[tuple[float, float]] = options.get("p1Bounds", [(-180, 180), (-180, 180)])
    composite_count = len(initial_decay)

    # ---------------------------------------------------------------------------- #
    #                               Checking Settings                              #
    # ---------------------------------------------------------------------------- #

    # ----------------------- Check Lower and Upper Bounds ----------------------- #
    
    # Ensure that the second element of bounds is greater than the first element
    for i, (low, high) in enumerate([decay_bounds_x, decay_bounds_y, amplitude_bounds, p0_bounds, p1_bounds]):
        if high <= low:
            bound_types = {
                0: "x-decay bounds",
                1: "y-decay bounds",
                2: "amplitude bounds",
                3: "p0 bounds",
                4: "p1 bounds"
            }
            bound_type = bound_types.get(i, "decay")
            
            raise ValueError(f"Upper bound must be greater than lower bound for {bound_type} parameter: "
                             f"lower={low}, upper={high}")
    
    # --------------------------- Check Initial Values --------------------------- #

    # Ensure initial decay values are within bounds
    for i, decay in enumerate(initial_decay):
        if not (decay_bounds_x[0] <= decay[0] <= decay_bounds_x[1]):
            errPrint(f"Warning: Initial decay value for x-axis ({initial_decay[i][0]}) is out of bounds {decay_bounds_x}. Adjusting to midpoint.")
            initial_decay[i] = ((decay_bounds_x[0] + decay_bounds_x[1]) / 2, decay[1])
        if not (decay_bounds_y[0] <= decay[1] <= decay_bounds_y[1]):
            errPrint(f"Warning: Initial decay value for y-axis ({initial_decay[i][1]}) is out of bounds {decay_bounds_y}. Adjusting to midpoint.")
            initial_decay[i] = (decay[0], (decay_bounds_y[0] + decay_bounds_y[1]) / 2)
    
    # Ensure initial phase values are within bounds
    for j, phase in enumerate(initial_phase):
        for i, phase_value in enumerate(phase):
            if not (p0_bounds[0] <= phase_value.p0 <= p0_bounds[1]):
                errPrint(f"Warning: Initial phase value for axis {i+1} P0 ({phase_value.p0}) is out of bounds {p0_bounds}. Adjusting to midpoint.")
                initial_phase[j][i] = Phase((p0_bounds[0] + p0_bounds[1]) / 2, phase_value.p1)
            if not (p1_bounds[0] <= phase_value.p1 <= p1_bounds[1]):
                errPrint(f"Warning: Initial phase value for axis {i+1} P1 ({phase_value.p1}) is out of bounds {p1_bounds}. Adjusting to midpoint.")
                initial_phase[j][i] = Phase(phase_value.p0, (p1_bounds[0] + p1_bounds[1]) / 2)

    # ------------------------------- Initial Guess ------------------------------ #

    # ----------------------------------- NOTE ----------------------------------- #
    # NOTE: Modify initial peak values and bounds later on, currently changed as needed

    spectral_widths = input_spectrum._spectral_widths
    time_size_pts = input_spectrum._total_time_points
    peak_count = len(input_spectrum.peaks)

    starting_x = [(decay[0]/spectral_widths[0]) * time_size_pts[0] for decay in initial_decay]

    starting_y = [(decay[1]/spectral_widths[1]) * time_size_pts[1] for decay in initial_decay]

    # Initial parameter bounds
    lower_bound_x = (decay_bounds_x[0]/spectral_widths[0]) * time_size_pts[0]
    
    upper_bound_x = (decay_bounds_x[1]/spectral_widths[0]) * time_size_pts[0]

    lower_bound_y = (decay_bounds_y[0]/spectral_widths[1]) * time_size_pts[1]
    
    upper_bound_y = (decay_bounds_y[1]/spectral_widths[1]) * time_size_pts[1]

    # Collect initial parameters
    initial_peak_lw_x = [lw for lw in starting_x for _ in range(peak_count)] # Repeat each starting x peak_count times and concatenate
    initial_peak_lw_y = [lw for lw in starting_y for _ in range(peak_count)] # Repeat each starting y peak_count times and concatenate
    initial_phase_x = []
    initial_phase_y = []

    # (x,y) -> (e,g) -> (p0,p1)
    x_axis_phase = initial_phase[0]
    y_axis_phase = initial_phase[1]

    for i, iphase in enumerate(x_axis_phase):
        initial_phase_x += ([iphase.p0] * peak_count) \
                        + ([iphase.p1] * peak_count) # Initial x-axis p0 and p1 phase

    for i, iphase in enumerate(y_axis_phase):
        initial_phase_y += ([iphase.p0] * peak_count) \
                        + ([iphase.p1] * peak_count) # Initial y-axis p0 and p1 phase
        
    initial_peak_heights = [peak.intensity for peak in input_spectrum.peaks] # Initial peak heights

    # Find the largest and smallest peak heights in the initial guess
    min_peak_height = min(initial_peak_heights)
    max_peak_height = max(initial_peak_heights)

    # Adjust the height bounds if they escape
    if min_peak_height < amplitude_bounds[0]:
        amplitude_bounds = (2 * min_peak_height if min_peak_height < 0 else min_peak_height / 2, amplitude_bounds[1])
    if max_peak_height > amplitude_bounds[1]:
        amplitude_bounds = (amplitude_bounds[0], max_peak_height * 2 if max_peak_height > 0 else max_peak_height / 2)

    if method == 'lsq':
        # --------------------------- Least Squares Bounds --------------------------- #

        # Collect lower bounds for x linewidth, y linewidth, lower amplitude, and lower p0 and p1 for both axes
        bounds_lsq_low = ([lower_bound_x] * peak_count * composite_count) \
                       + ([lower_bound_y] * peak_count * composite_count) \
                       + ([amplitude_bounds[0]] * peak_count) \
                       + ([p0_bounds[0]] * peak_count * composite_count) \
                       + ([p1_bounds[0]] * peak_count * composite_count) \
                       + ([p0_bounds[0]] * peak_count * composite_count) \
                       + ([p1_bounds[0]] * peak_count * composite_count) 
        
        # Collect upper bounds for x linewidth, y linewidth, lower amplitude, and lower p0 and p1 for both axes
        bounds_lsq_high = ([upper_bound_x] * peak_count * composite_count) \
                       + ([upper_bound_y] * peak_count * composite_count) \
                       + ([amplitude_bounds[1]] * peak_count) \
                       + ([p0_bounds[1]] * peak_count * composite_count) \
                       + ([p1_bounds[1]] * peak_count * composite_count) \
                       + ([p0_bounds[1]] * peak_count * composite_count) \
                       + ([p1_bounds[1]] * peak_count * composite_count) 
        
        bounds = (bounds_lsq_low, bounds_lsq_high)

    else: 
        # ---------------------------------- Bounds ---------------------------------- #

        x_axis_bounds = [(lower_bound_x, upper_bound_x)] * peak_count # X-axis linewidth bounds

        y_axis_bounds = [(lower_bound_y, upper_bound_y)] * peak_count # Y-axis linewidth bounds

        intensity_bounds = [amplitude_bounds] * peak_count # Peak height bounds (non-negative)

        phase_p0_bounds = [p0_bounds] * peak_count # p0 bounds for x-axis and y-axis

        phase_p1_bounds = [p1_bounds] * peak_count # p1 bounds for x-axis and y-axis

        bounds = x_axis_bounds \
               + y_axis_bounds \
               + intensity_bounds \
               + phase_p0_bounds \
               + phase_p1_bounds \
               + phase_p0_bounds \
               + phase_p1_bounds \
               

    # ----------------------------- Run Optimization ----------------------------- #

    if input_spectrum.verbose:
        errPrint("Original Parameters:")
        errPrint(f"X Line-widths = {initial_peak_lw_x[0]:.3f},...")
        errPrint(f"Y Line-widths = {initial_peak_lw_y[0]:.3f},...")
        errPrint(f"Peak Heights = {initial_peak_heights[0]:.3f},...")
        errPrint(f"X Phase = {(initial_phase_x[0], initial_phase_x[peak_count])}")
        errPrint(f"Y Phase = {(initial_phase_y[0], initial_phase_y[peak_count])}")
        errPrint("")

    initial_params = np.concatenate((
        initial_peak_lw_x,
        initial_peak_lw_y,
        initial_peak_heights,
        initial_phase_x, 
        initial_phase_y
        ))

    
    optimization_args = (composite_count, input_spectrum, composite_function, input_fid, target_interferogram, sim_params)
    if method == 'lsq':
        verbose = 2 if input_spectrum.verbose else 0
        sim_params["difference_equation"] = lambda target, simulated: (target - simulated).flatten()
        optimization_args = (composite_count, input_spectrum, composite_function, input_fid, target_interferogram, sim_params)
        result = least_squares(composite_objective_function, initial_params, '2-point',
                            bounds, 'trf', args=optimization_args, verbose=verbose, max_nfev=trial_count)
    elif method == 'basin':
        disp = True if input_spectrum.verbose else False
        result = basinhopping(composite_objective_function, initial_params, niter=trial_count, stepsize=step_size, 
                        minimizer_kwargs={"args": optimization_args},
                        disp=disp)
    elif method == 'minimize':
        disp = True if input_spectrum.verbose else False
        result = minimize(composite_objective_function, initial_params, 
                        args=optimization_args,
                        method='SLSQP', options={"disp":disp})
    else:
        disp = True if input_spectrum.verbose else False
        x0, fval, grid, Jout = brute(composite_objective_function, bounds, args=optimization_args, Ns=20, full_output=True, workers=-1, disp=disp)
    
    if method == 'brute':
        optimized_params = x0
    else:
        # Extract the optimized parameters
        optimized_params = result.x

    composite_total = composite_count * peak_count
    # Unpack parameters
    optimized_peak_lw_x = optimized_params[:composite_total]  # X-axis peak linewidths
    optimized_peak_lw_y = optimized_params[composite_total:2 * composite_total]  # Y-axis peak linewidths
    optimized_peak_heights = optimized_params[2 * composite_total:2 * composite_total + peak_count]  # Peak heights
    phase_start = 2 * composite_total + peak_count
    optimized_phase_x = (optimized_params[phase_start:phase_start + composite_total],  # X-axis p0 value
               optimized_params[phase_start + composite_total:phase_start + 2 * composite_total])  # X-axis p1 value
    optimized_phase_y = (optimized_params[phase_start + 2 * composite_total:phase_start + 3 * composite_total],  # Y-axis p0 value
               optimized_params[phase_start + 3 * composite_total:])  # Y-axis p1 value
    
    if input_spectrum.verbose:
        errPrint("")
        errPrint("Optimized Parameters:")
        errPrint(f"X Line-widths = {optimized_peak_lw_x}")
        errPrint(f"Y Line-widths = {optimized_peak_lw_y}")
        errPrint(f"Peak Heights = {optimized_peak_heights}")
        errPrint(f"X Phase = {optimized_phase_x}")
        errPrint(f"Y Phase = {optimized_phase_y}")
        errPrint("")

    # ------------------------------- New Spectrum ------------------------------- #

    # Create a copy of the input spectrum with the optimized parameters
    optimized_spectrum = Spectrum(
        input_spectrum.file,
        input_spectrum._spectral_widths,
        input_spectrum._coordinate_origins,
        input_spectrum._observation_freqs,
        input_spectrum._total_time_points,
        input_spectrum._total_freq_points
    )

    # Update peaks with the optimized parameters
    optimized_spectrum.peaks = []
    for i, peak in enumerate(input_spectrum.peaks):
        x_linewidth = Coordinate(float(optimized_peak_lw_x[i]),
                             input_spectrum._spectral_widths[0], input_spectrum._coordinate_origins[0], 
                             input_spectrum._observation_freqs[0], input_spectrum._total_time_points[0])

        y_linewidth = Coordinate(float(optimized_peak_lw_y[i]),
                             input_spectrum._spectral_widths[1], input_spectrum._coordinate_origins[1], 
                             input_spectrum._observation_freqs[1], input_spectrum._total_time_points[1])

        linewidths = Coordinate2D(input_spectrum.peaks[i].linewidths.index, 
                                  x_linewidth, y_linewidth)
        
        xPhase = Phase(optimized_phase_x[0][i], optimized_phase_x[1][i])
        yPhase = Phase(optimized_phase_y[0][i], optimized_phase_y[1][i])

        optimized_peak = Peak(
            peak.position,
            float(optimized_peak_heights[i]),
            linewidths,
            extra_params=peak.extra_params
        )
        optimized_peak.phase = [xPhase, yPhase]
        optimized_spectrum.peaks.append(optimized_peak)
    
    # Save optimized parameters to a text file with comma separated values
    with open("optimized_parameters.csv", "w") as file:
        file.write("X_pos,Y_pos,X_LW_exp,X_LW_gauss,Y_LW_exp,Y_LW_gauss,Peak_H,X_P0_exp,X_P0_gauss,X_P1_exp,X_P1_gauss,Y_P0_exp,Y_P0_gauss,Y_P1_exp,Y_P1_gauss\n")
        for i in range(peak_count):
            file.write(f"{optimized_spectrum.peaks[i].position.x},{optimized_spectrum.peaks[i].position.y}" +
                       f"{optimized_peak_lw_x[i]},{optimized_peak_lw_x[i+peak_count]}," +
                       f"{optimized_peak_lw_y[i]},{optimized_peak_lw_y[i+peak_count]}," +
                       f"{optimized_peak_heights[i]}," +
                       f"{optimized_phase_x[0][i]},{optimized_phase_x[0][i+peak_count]}," +
                       f"{optimized_phase_x[1][i]},{optimized_phase_x[1][i+peak_count]}," +
                       f"{optimized_phase_y[0][i]},{optimized_phase_y[0][i+peak_count]}," +
                       f"{optimized_phase_y[1][i]},{optimized_phase_y[1][i+peak_count]}\n")
            
    return optimized_spectrum

def composite_objective_function(params, composite_count : int,
                                 input_spectrum : Spectrum, model_function : ModelFunction, input_fid : pype.DataFrame, target_interferogram : pype.DataFrame, options : dict = {}) -> np.float32:
    """
    Function used for optimizing peak linewidths and heights of spectrum

    Parameters
    ----------
    params : list
        List of target parameters being tested (peak indices, peak linewidths, peak heights)
    composite_count : int
        Number of composite elements
    input_spectrum : Spectrum
        Input spectrum to generate from 
    model_function : ModelFunction
        Spectral modeling method function (e.g. exponential, gaussian)
    input_fid : nmrPype.DataFrame
        Corresponding fid dataframe to perform functions
    target_interferogram : nmrPype.DataFrame
        Corresponding nmrPype dataframe matching header content
    options : dict
        "difference_equation" : 
        "offsets" : list[int]
        "constant_time_region_sizes" : list[int]
        "scaling_factors" : list[float]
    Returns
    -------
    np.float32
        Difference between simulated data and real data
    """

    peak_count = len(input_spectrum.peaks)
    composite_total = composite_count * peak_count
    # Unpack parameters
    peak_x_lws = params[:composite_total]  # X-axis peak linewidths
    peak_y_lws = params[composite_total:2 * composite_total]  # Y-axis peak linewidths
    peak_heights = params[2 * composite_total:2 * composite_total + peak_count]  # Peak heights
    phase_start = 2 * composite_total + peak_count
    phase_x = (params[phase_start:phase_start + composite_total],  # X-axis p0 value
               params[phase_start + composite_total:phase_start + 2 * composite_total])  # X-axis p1 value
    phase_y = (params[phase_start + 2 * composite_total:phase_start + 3 * composite_total],  # Y-axis p0 value
               params[phase_start + 3 * composite_total:])  # Y-axis p1 value

    # Update peaks with the current parameters
    for i, peak in enumerate(input_spectrum.peaks):
        peak.linewidths.x.pts = peak_x_lws[i]
        peak.linewidths.y.pts = peak_y_lws[i]
        peak.exp_linewidths.x.pts = peak_x_lws[i]
        peak.exp_linewidths.y.pts = peak_y_lws[i]
        if composite_count >= 2:
            peak.gauss_linewidths.x.pts = peak_x_lws[i+peak_count]
            peak.gauss_linewidths.y.pts = peak_y_lws[i+peak_count]

        peak.intensity = peak_heights[i]
        xPhase = Phase(phase_x[0][i], phase_x[1][i])
        yPhase = Phase(phase_y[0][i], phase_y[1][i])

        peak.phase = [xPhase, yPhase]
        peak.phase_exp = [xPhase, yPhase]

        if composite_count >= 2:
            gauss_xPhase = Phase(phase_x[0][i+peak_count], 
                                 phase_x[1][i+peak_count])
            gauss_yPhase = Phase(phase_y[0][i+peak_count],
                                 phase_y[1][i+peak_count])
            
            peak.phase_gauss = [gauss_xPhase, gauss_yPhase]
            
    ctrs = options.get("constant_time_region_sizes", [0,0] )
    offsets = options.get("offsets", [0,0])
    scaling_factors = options.get("scaling_factors", [1.0, 1.0])
    # Generate the simulated interferogram
    simulated_interferogram = input_spectrum.spectral_simulation(model_function, target_interferogram, input_fid, 2, 0, 'ft1', ctrs, False, offsets=offsets, scaling_factors=scaling_factors)

    # Calculate the difference between the target and simulated interferograms
    difference_equation = options.get('difference_equation', lambda target, simulated: np.sum((target - simulated) ** 2))
    difference = difference_equation(target_interferogram.array, simulated_interferogram)

    return difference