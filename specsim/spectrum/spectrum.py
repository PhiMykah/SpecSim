import sys, time, io
import re
from pathlib import Path
from ..peak import Peak, Coordinate, Coordinate2D, Phase
from ..calculations import outer_product_summation, extract_region
from ..models import sim_composite_1D
from ..debug.verbose import errPrint
import numpy as np
from scipy.optimize import basinhopping, minimize, brute, least_squares

import nmrPype as pype
from typing import Callable, Optional, Annotated

type File = str | Path
type ModelFunction = Callable[[int, int, int, float, float, Optional[np.ndarray], Optional[np.ndarray], float, tuple[int, int], float], np.ndarray]

from matplotlib import pyplot as plt

# Axes
X = 0 # First index for x-axis
Y = 1 # Second index for y-axis

# ---------------------------------------------------------------------------- #
#                                Spectrum Class                                #
# ---------------------------------------------------------------------------- #
class Spectrum:
    """
    A class to contain the system of peaks that comprise the spectrum.
    Contains methods for reading from peak table file, simulation, and optimization.

    Attributes
    ----------

    file : Path
        File directory of peak table

    remarks : str
        Remarks from the peak table file

    attributes : dict[str, str]
        Dictionary of peak variables and formats
    
    _spectral_widths : tuple[float, float]
        The x-coord and y-coord spectral widths of the spectrum in Hz. 
        Also known as the sweep width.
    
    _coordinate_origins : tuple[float, float]
        The x-plane and y-plane origin of the coordinate system (in pts)

    _observation_freqs : tuple[float, float]
        The x-plane and y-plane observation frequency of the spectrum

    _total_points : tuple[int, int]
        The total number of points in the spectrum for each dimension
    
    peaks : list[Peak]
        Line-by-line string list of all lines from the tab file

    verbose : bool
        Flag to enable verbose mode
    """
    def __init__(self, peak_table_file : File, 
                 spectral_widths : tuple[float,float], 
                 coordinate_origins : tuple[float,float],
                 observation_freqs : tuple[float,float],
                 total_time_points : tuple[int, int],
                 total_freq_points : tuple[int, int],
                 verbose: bool = False):
        
        if type(peak_table_file) == str:
            peak_table = Path(peak_table_file)
        else:
            peak_table = peak_table_file
        if not peak_table.exists():
            raise IOError(f"Unable to find file: {peak_table}")

        # Initializations
        self.file : Path = peak_table
        self.remarks : str = ""
        self.attributes : dict[str, str] = []
        self._spectral_widths = spectral_widths
        self._coordinate_origins = coordinate_origins
        self._observation_freqs = observation_freqs
        self._total_time_points = total_time_points
        self._total_freq_points = total_freq_points

        self.peaks : list[Peak] = self._read_file()
        self._null_string = '*'
        self._null_value = -666
        self.verbose = verbose

    # ---------------------------------------------------------------------------- #
    #                         Spectral Simulation Functions                        #
    # ---------------------------------------------------------------------------- #

    def spectral_simulation(self, model_function : ModelFunction, 
                            spectrum_data_frame : pype.DataFrame,
                            spectrum_fid : pype.DataFrame,
                            axis_count : int = 2,
                            peaks_simulated : list[int] | int = 0, 
                            domain : str = 'ft1',
                            constant_time_region_sizes : Annotated[list[int], 2] = [0, 0],
                            set_phase_values : bool = True,
                            phase_values : Annotated[list[Phase], 2] = [Phase(0,0),Phase(0,0)],
                            offsets : Annotated[list[int], 2] = [0,0],
                            scaling_factors : Annotated[list[float], 2] = [1.0, 1.0]):
        """
        Simulates a multi-dimensional spectrum based on the provided model function and parameters.

        Parameters
        ----------
        model_function : ModelFunction
            The model function used for spectral simulation.
        spectrum_data_frame : nmrPype.DataFrame
            nmrpype format data to obtain header information from
        spectrum_fid : nmrPype.DataFrame
            nmrpype format data to perform transforms on
        axis_count : int, optional
            Number of axes to simulate, by default 2
        peaks_simulated : list[int] | int, optional
            Number of peaks to simulate or indices of peaks to simulate, by default entire spectrum
                - 0 simulates entire spectrum
        domain : str, optional
            The domain in which the simulation is performed, by default 'ft1'.
                - 'fid' for time-domain
                - 'ft1' for interferrogram
                - 'ft2' for full spectrum
        constant_time_region_sizes : Annotated[list[int], 2], optional
            Total number of points in the constain time regions (pts), by default [0, 0]
        set_phase_values : bool, optional
            Update phase values for each peak to phase_values parameter, by default True
        phase_values : Annotated[list[tuple[int,int]], 2], optional
            multi-axis p0 and p1 phase correction values, by default [Phase(0,0),Phase(0,0)]
        offsets : Annotated[list[int], 2], optional
            multi-axis offset values of the frequency domain in points, by default [0,0]
        scaling_factors : Annotated[list[float], 2], optional
            multi-axis simulation time domain data scaling factor, by default [1.0, 1.0]
        """
        if set_phase_values and (model_function == sim_composite_1D):
            spec_sim_phase_values = phase_values[0]
        else:
            spec_sim_phase_values = phase_values
        if not (axis_count == len(constant_time_region_sizes) == len(spec_sim_phase_values) == len(offsets) == len(scaling_factors)):
            raise ValueError(f"Axis count and parameter lengths do not match: "
                             f"axis_count={axis_count}, " 
                             f"constant_time_region_sizes={len(constant_time_region_sizes)}, "
                             f"phase_values={len(phase_values)}, "
                             f"offsets={len(offsets)}, "
                             f"scaling_factors={len(scaling_factors)}")
        simulations : list[np.ndarray] = []
        # // Debug
        # time_start = time.time()
        for axis in range(axis_count):
            if model_function == sim_composite_1D:
                sim_1D_peaks = self.composite_simulation1D(model_function, peaks_simulated, axis,
                                            constant_time_region_sizes[axis],
                                            set_phase_values,
                                            phase_values[axis], 
                                            offsets[axis],
                                            scaling_factors[axis])
            else: 
                sim_1D_peaks = self.spectral_simulation1D(model_function, peaks_simulated, axis,
                                            constant_time_region_sizes[axis],
                                            set_phase_values,
                                            phase_values[0][axis],
                                            offsets[axis],
                                            scaling_factors[axis])
            simulations.append(sim_1D_peaks)

        process_iterations = 0
        if domain == 'ft1' and axis_count >= 2:
            process_iterations = 1
        if domain == 'ft2' and axis_count >= 2:
            process_iterations = 2
        
        # Collect all the x axis peaks and y axis peaks
        spectral_data : list[list[np.ndarray]] = [*simulations]
        fid = spectrum_fid

        # Iterate through all dimensions in need of processing
        for i in range(process_iterations):
            spectral_data[i] = []
            # Go through all peaks in simulation
            for j in range(len(simulations[i])):
                iteration_df = pype.DataFrame(file=fid.file, header=fid.header, array=fid.array)
                iteration_df.array = simulations[i][j]
                dim = i + 1

                #  Add first point scaling and window function if necessary
                off_param = spectrum_data_frame.getParam("NDAPODQ1", dim)
                end_param = spectrum_data_frame.getParam("NDAPODQ2", dim)
                pow_param = spectrum_data_frame.getParam("NDAPODQ3", dim)
                elb_param = spectrum_data_frame.getParam("NDLB", dim)
                glb_param = spectrum_data_frame.getParam("NDGB", dim)
                goff_param = spectrum_data_frame.getParam("NDGOFF", dim)
                first_point_scale = 1 + spectrum_data_frame.getParam("NDC1", dim)

                iteration_df.runFunc("SP", {"sp_off":off_param, "sp_end":end_param, "sp_pow":pow_param, "sp_elb":elb_param,
                                            "sp_glb":glb_param, "sp_goff":goff_param, "sp_c":first_point_scale})
                # Zero fill if necessary
                iteration_df.runFunc("ZF", {"zf_count":1, 'zf_auto':True})

                # Convert to frequency domain
                iteration_df.runFunc("FT")
                
                # Perform phase correction
                iteration_df.runFunc("PS", {"ps_p0":spec_sim_phase_values[i][0], "ps_p1":spec_sim_phase_values[i][1]})

                # Delete imaginary values
                # simulations[i] = simulations[i].real
                iteration_df.runFunc("DI")
                
                # Extract designated region if necessary
                first_point = int(spectrum_data_frame.getParam("NDX1", dim))
                last_point = int(spectrum_data_frame.getParam("NDXN", dim))
                if first_point and last_point:
                    iteration_df.array = extract_region(iteration_df.array, first_point, last_point)

                # Add peak to list
                spectral_data[i].append(iteration_df.array)

        if axis_count == 2:
            result = outer_product_summation(spectral_data[0], spectral_data[1])

            # // Debug
            # time_end = time.time()
            # peak_count = (len(peaks_simulated) if isinstance(peaks_simulated, list) 
            #             else (len(self.peaks) if peaks_simulated == 0 else peaks_simulated))
            # print(f"Simulation of {peak_count} peak(s) took {time_end-time_start:.3f}s")

            return result
        return spectral_data

    def spectral_simulation1D(self, model_function : ModelFunction,
                           peaks_simulated : list[int] | int = 0, 
                           axis : int = X,
                           constant_time_region_size : int = 0,
                           set_phase_value : bool = True,
                           phase : tuple[int,int] = (0,0),
                           offset : int = 0,
                           scaling_factor : float = 1.0) -> np.ndarray:
        """
        Perform spectral simulation using inputted decay function model

        Parameters
        ----------
        model_function : ModelFunction
            Target modeling function for simulation, e.g. exponential, gaussian
        peaks_simulated : list[int] | int
            Number of peaks to simulate or indices of peaks to simulate. by default 0
                - 0 simulates entire spectrum
        axis : int
            Designated axis to simulate, e.g. 0 for 'x', 1 for 'y'.
        constant_time_region_size : int
            Total number of points in the constain time region (pts), by default 0
        set_phase_values : bool, optional
            Update phase values for each peak to phase_values parameter, by default True
        phase : tuple[int,int]
            p0 and p1 phase correction values, by default (0,0)
        offset : int
            Offset of the frequency domain in points, by default 0
        scaling_factor : float
            Simulation time domain data scaling factor, by default 1.0

        Returns
        -------
        np.ndarray
            Return Simulated Spectrum

        Raises
        ------
        TypeError
            Raises an type error if peaks_simulated is incorrectly inputted
        """
        peak_count = -1
        peak_indices = []
        usePeakCount = True

        # Use number of peaks if input is integer
        if type(peaks_simulated) == int:
            peak_count = peaks_simulated
            usePeakCount = True
        # Use index list if input is list
        elif type(peaks_simulated) == list:
            peak_indices = [i for i in peaks_simulated if i < len(self.peaks)]
            usePeakCount = False
        # Raise error if input does not match integer and list
        else:
            raise TypeError("Incorrect Type for Simulation, use a list of indices or an integer count")
        


        # Set range if number of peaks is input
        if usePeakCount:
            # Ensure nonzero integer count
            if peak_count < 0:
                peak_count = 0

            # Use entire spectrum if peak count is 0 or exceeds maximum
            if peak_count > len(self.peaks) or peak_count == 0:
                peak_count = len(self.peaks)
            peak_indices = range(peak_count)

        spectrum_list = []
        # Iterate through given peaks in the spectrum
        for i in peak_indices:
            spectrum_list.append(self._run_simulation_iteration(model_function, i,
                                            axis, constant_time_region_size, offset, scaling_factor,
                                            set_phase_value, phase))
            
        return np.array(spectrum_list)

    def _run_simulation_iteration(self, model_function : ModelFunction, 
                                  index : int, axis : int, constant_time_region_size : int,
                                  offset : int,
                                  scaling_factor : float,
                                  set_phase_value : bool = True,
                                  phase : Phase = Phase(0,0)
                                  ) -> np.ndarray:
        """
        Run a single iteration of a spectral time domain simulation.

        Parameters
        ----------
        model_function : ModelFunction
            Model simulation function such as exponential or gaussian
        index : int
            Current iteraiton index
        axis : str
            Designated axis to simulate, e.g. 0 for 'x', 1 for 'y'.
        constant_time_region_size : int
            Total number of points in the constain time region (pts)
        phase : tuple[int,int]
            p0 and p1 phase correction values
        offset : int
            Offset of the frequency domain in points
        scaling_factor : float
            Simulation time domain data scaling factor

        Returns
        -------
        np.ndarray
            Simulation slice based on model function
        """

        if axis not in [X, Y]:
            TypeError("Incorrect Axis Type for Simulation, please enter a supported axis type.")

        if set_phase_value:
            if axis == X:
                self.peaks[index].xPhase = phase
            if axis == Y:
                self.peaks[index].yPhase = phase

        # Obtain frequency and add offset if necessary
        frequency_pts = self.peaks[index].position[axis] + offset

        # Collect line width, and intensity
        line_width = self.peaks[index].linewidths[axis]

        amplitude = self.peaks[index].intensity

        # Set modulation axis to X, Y, etc
        cosine_modulation = f"{chr(axis + ord('X'))}_COSJ"
        sine_modulation = f"{chr(axis + ord('X'))}_SINJ"

        # Collect cosine moduation if it exists
        if self.peaks[index].extra_params[cosine_modulation]:
            cos_mod_j = np.array(self.peaks[index].extra_params[cosine_modulation])
        else:
            cos_mod_j = None
        
        # Collect sine modulation if it exists
        if self.peaks[index].extra_params[sine_modulation]:
            sin_mod_j = np.array(self.peaks[index].extra_params[sine_modulation])
        else:
            sin_mod_j = None

        return(
            model_function(self._total_time_points[axis], self._total_freq_points[axis],
                           constant_time_region_size, frequency_pts,
                           line_width,
                           cos_mod_values=cos_mod_j,
                           sin_mod_values=sin_mod_j,
                           amplitude=amplitude, phase=self.peaks[index].phase[axis],
                           scale=scaling_factor))

    
    def composite_simulation1D(self, composite_function,
                            peaks_simulated : list[int] | int = 0, 
                            axis : int = X,
                            constant_time_region_size : int = 0,
                            set_phase_value : bool = True,
                            phase : tuple[int,int] = (0,0),
                            offset : int = 0,
                            scaling_factor : float = 1.0) -> np.ndarray:
        peak_count = -1
        peak_indices = []
        usePeakCount = True

        # Use number of peaks if input is integer
        if type(peaks_simulated) == int:
            peak_count = peaks_simulated
            usePeakCount = True
        # Use index list if input is list
        elif type(peaks_simulated) == list:
            peak_indices = [i for i in peaks_simulated if i < len(self.peaks)]
            usePeakCount = False
        # Raise error if input does not match integer and list
        else:
            raise TypeError("Incorrect Type for Simulation, use a list of indices or an integer count")
        


        # Set range if number of peaks is input
        if usePeakCount:
            # Ensure nonzero integer count
            if peak_count < 0:
                peak_count = 0

            # Use entire spectrum if peak count is 0 or exceeds maximum
            if peak_count > len(self.peaks) or peak_count == 0:
                peak_count = len(self.peaks)
            peak_indices = range(peak_count)

        spectrum_list = []
        # Iterate through given peaks in the spectrum
        for i in peak_indices:
            spectrum_list.append(
                self._run_composite_iteration(composite_function, i, axis, 
                                              constant_time_region_size, offset, scaling_factor, set_phase_value, phase
                                              ))
            
        return np.array(spectrum_list)
    
    def _run_composite_iteration(self, composite_function, 
                                index : int, axis : int, constant_time_region_size : int,
                                offset : int,
                                scaling_factor : float,
                                set_phase_value : bool = True,
                                phase : Phase = Phase(0,0)
                                ) -> np.ndarray:
        """
        Run a single iteration of a spectral time domain simulation composite model.

        Parameters
        ----------
        composite_function
            Model simulation function such as exponential or gaussian
        index : int
            Current iteraiton index
        axis : str
            Designated axis to simulate, e.g. 0 for 'x', 1 for 'y'.
        constant_time_region_size : int
            Total number of points in the constain time region (pts)
        phase : tuple[int,int]
            p0 and p1 phase correction values
        offset : int
            Offset of the frequency domain in points
        scaling_factor : float
            Simulation time domain data scaling factor

        Returns
        -------
        np.ndarray
            Simulation slice based on model function
        """

        if axis not in [X, Y]:
            TypeError("Incorrect Axis Type for Simulation, please enter a supported axis type.")

        if set_phase_value:
            self.peaks[index].phase_exp[axis] = phase[axis]
            self.peaks[index].phase_gauss[axis] = phase[axis]

        # Obtain frequency and add offset if necessary
        frequency_pts = self.peaks[index].position[axis] + offset

        # Collect line width, and intensity
        exp_line_width = self.peaks[index].exp_linewidths[axis]
        gauss_line_width = self.peaks[index].gauss_linewidths[axis]

        amplitude = self.peaks[index].intensity

        # Set modulation axis to X, Y, etc
        # cosine_modulation = f"{chr(axis + ord('X'))}_COSJ"
        # sine_modulation = f"{chr(axis + ord('X'))}_SINJ"

        # # Collect cosine moduation if it exists
        # if self.peaks[index].extra_params[cosine_modulation]:
        #     cos_mod_j = np.array(self.peaks[index].extra_params[cosine_modulation])
        # else:
        #     cos_mod_j = None
        
        # # Collect sine modulation if it exists
        # if self.peaks[index].extra_params[sine_modulation]:
        #     sin_mod_j = np.array(self.peaks[index].extra_params[sine_modulation])
        # else:
        #     sin_mod_j = None

        return(sim_composite_1D(
            self._total_time_points[axis], self._total_freq_points[axis], 
            constant_time_region_size, frequency_pts, exp_line_width, gauss_line_width,
            amplitude, self.peaks[index].phase_exp[axis], self.peaks[index].phase_gauss[axis],
            self.peaks[index].gauss_weights[axis], scaling_factor
            ))
        
    # ---------------------------------------------------------------------------- #
    #                              Peak Table Reading                              #
    # ---------------------------------------------------------------------------- #

    def _read_file(self) -> list[Peak]:
        """
        Reads from peak table and strips unnecessary lines from process
        Note: Assumes that self.file has been populated and is a valid file

        Returns
        -------
        list[Peak]
            List of valid peaks
        """
        file_lines : list[str] = [] # List of each line in file as a string
        with self.file.open('r') as f:
            file_lines = f.readlines()

        attribute_keys : list[str] = []     # Attribute variable names
        attribute_formats : list[str] = []  # Attribute format values
        peaks : list[Peak] = []             # List of peaks 

        # ----------------------------- Header Processing ---------------------------- #

        line_number = 0          # Current line of the file
        is_header_read = False  # Whether the header has been read completely or not

        # Iterate through each line and ensure that information
        # Goes to the right property
        while not is_header_read and file_lines:
            line = file_lines[0]

            # Exit at the end of the file
            if not line:
                break
            
            # Increment line number
            line_number += 1 
            # Remove whitespace
            line = line.strip()

            # Only continue if the line isn't empty
            if not line:
                del file_lines[0]
                continue

            if line.upper().startswith("REMARK") or line.upper().startswith("#"):
                # Collect remarks into a single string
                self.remarks += re.sub("(REMARK|#)( )+", "", line, flags=re.IGNORECASE) + "\n"
                del file_lines[0]
            elif line.upper().startswith("VARS"):
                # Collect variable names
                attributes_string = re.sub("VARS( )+", "", line, flags=re.IGNORECASE)
                attribute_keys = re.split(r"[,|\s]+", attributes_string)
                del file_lines[0]
            elif line.upper().startswith("FORMAT"):
                # Collect matching variable formats
                formats_string = re.sub("FORMAT( )+", "", line, flags=re.IGNORECASE).lstrip("%")
                attribute_formats = re.split(r"[,|\s]+%", formats_string)
                del file_lines[0]
            elif line.upper().startswith("NULLSTRING"):
                self._null_string = re.sub("NULLSTRING( )+", "", line, flags=re.IGNORECASE)
                del file_lines[0]
            elif line.upper().startswith("NULLVALUE"):
                self._null_value = int(re.sub("NULLVALUE( )+", "", line, re.IGNORECASE))
                del file_lines[0]
            else:
                # If the line doesn't match header keywords, assume it's data and exit loop
                is_header_read = True

        # Ensure equal attribute variables and formats to match
        if (attribute_keys and attribute_formats) and len(attribute_keys) == len(attribute_formats):
            # Generate attribute dictionary
            self.attributes = dict(
                map(lambda key,value : (key,value), attribute_keys, attribute_formats))

        # // For debugging
        # errPrint(f"Header Length: {line_number}")

        while file_lines:
            line_number += 1
            # Convert the string to a peak and add it to the list
            new_peak = string_to_peak(
                    file_lines.pop(0), self.attributes, self.file, line_number,
                    (self._spectral_widths, self._coordinate_origins,
                     self._observation_freqs, self._total_freq_points))
            
            if new_peak:  
                peaks.append(new_peak)  

        return peaks
    
    # ---------------------------------------------------------------------------- #
    #                             Peak Table Management                            #
    # ---------------------------------------------------------------------------- #

    def set_peak_phases(self, phases_list : list[Annotated[list[Phase], 2]]):
        if len(phases_list) != len(self.peaks):
            raise ValueError("The amount of phase pairs must be equal to the number of peaks in the spectrum.")
        
        for i, phase in enumerate(phases_list):
            if len(phase) != 2:
                raise ValueError(f"Peak #{i+1} must have a phase for x-axis and y-axis (currently {len(phase)}).")
            self.peaks[i].phase = phase

    # ---------------------------------------------------------------------------- #
    #                                 Magic Methods                                #
    # ---------------------------------------------------------------------------- #

    def __repr__(self):
        if not self.remarks:
            remarks = "None"
        else:
            remarks = self.remarks
        return f"Spectrum({len(self.peaks)} peaks, " \
               f"file={self.file}, " \
               f"remarks={remarks}, " \
               f"attributes={self.attributes}, " \
               f"spectral_widths={self._spectral_widths}, " \
               f"coordinate_origins={self._coordinate_origins}, " \
               f"observation_freqs={self._observation_freqs}, " \
               f"total_time_points={self._total_time_points}, " \
               f"total_freq_points={self._total_freq_points}, " \
               f"verbose={self.verbose})"

# ---------------------------------------------------------------------------- #
#                                 Optimization                                 #
# ---------------------------------------------------------------------------- #

def objective_function(params, input_spectrum : Spectrum, model_function : ModelFunction, input_fid : pype.DataFrame, target_interferogram : pype.DataFrame, options : dict = {}) -> np.float32:
    """
    Function used for optimizing peak linewidths and heights of spectrum

    Parameters
    ----------
    params : list
        List of target parameters being tested (peak indices, peak linewidths, peak heights)
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

    # Unpack parameters
    peak_x_lws = params[:peak_count] # X-axis peak linewidths
    peak_y_lws = params[peak_count:2*peak_count] # Y-axis peak linewidths
    peak_heights = params[2*peak_count:3*peak_count] # Peak heights
    phase_x = (params[3*peak_count:4*peak_count], # X-axis p0 value
               params[4*peak_count:5*peak_count]) # X-axis p1 value
    phase_y = (params[5*peak_count:6*peak_count], # Y-axis p0 value
               params[6*peak_count:])             # Y-axis p1 value

    # Update peaks with the current parameters
    for i, peak in enumerate(input_spectrum.peaks):
        peak.linewidths.x.pts = peak_x_lws[i]
        peak.linewidths.y.pts = peak_y_lws[i]
        peak.intensity = peak_heights[i]
        xPhase = Phase(phase_x[0][i], phase_x[1][i])
        yPhase = Phase(phase_y[0][i], phase_y[1][i])
        peak.phase = [xPhase, yPhase]

    ctrs = options.get("constant_time_region_sizes", [0,0] )
    offsets = options.get("offsets", [0,0])
    scaling_factors = options.get("scaling_factors", [1.0, 1.0])
    # Generate the simulated interferogram
    simulated_interferogram = input_spectrum.spectral_simulation(model_function, target_interferogram, input_fid, 2, 0, 'ft1', ctrs, False, offsets=offsets, scaling_factors=scaling_factors)

    # Calculate the difference between the target and simulated interferograms
    difference_equation = options.get('difference_equation', lambda target, simulated: np.sum((target - simulated) ** 2))
    difference = difference_equation(target_interferogram.array, simulated_interferogram)

    return difference
    
def interferogram_optimization(input_spectrum: Spectrum, model_function: ModelFunction, input_fid : pype.DataFrame,
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
        "initDecay" : tuple[float, float]
            tuple of initial decay values for optimization in Hz
        "initPhase" : tuple[Phase, Phase]
            tuple of initial phase values for optimiation in Hz
        "dXBounds": tuple[float, float]
            tuple of lower and upper bounds for x-axis decay in Hz
        "dYBounds": tuple[float, float]
            tuple of lower and upper bounds for y-axis decay in Hz
        "aBounds": tuple[float, float]
            tuple of lower and upper bounds for decay
        "p0Bounds": tuple[float, float]
            tuple of lower and upper bounds for phase P0 in degrees
        "p1Bounds": 
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
    initial_decay : tuple[float, float] = options.get("initDecay", (15, 5))
    initial_phase : list[Phase, Phase] = list(options.get("initPhase", (Phase(0,0), Phase(0,0))))
    decay_bounds_x : tuple[float, float] = options.get("dxBounds", (0, 100))
    decay_bounds_y : tuple[float, float] = options.get("dyBounds", (0, 20))
    amplitude_bounds : tuple[float, float] = options.get("aBounds", (0, 150))
    p0_bounds : tuple[float, float] = options.get("p0Bounds", (-180, 180))
    p1_bounds : tuple[float, float] = options.get("p1Bounds", (-180, 180))

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
    if not (decay_bounds_x[0] <= initial_decay[0] <= decay_bounds_x[1]):
        errPrint(f"Warning: Initial decay value for x-axis ({initial_decay[0]}) is out of bounds {decay_bounds_x}. Adjusting to midpoint.")
        initial_decay = ((decay_bounds_x[0] + decay_bounds_x[1]) / 2, initial_decay[1])
    if not (decay_bounds_y[0] <= initial_decay[1] <= decay_bounds_y[1]):
        errPrint(f"Warning: Initial decay value for y-axis ({initial_decay[1]}) is out of bounds {decay_bounds_y}. Adjusting to midpoint.")
        initial_decay = (initial_decay[0], (decay_bounds_y[0] + decay_bounds_y[1]) / 2)
    
    # Ensure initial phase values are within bounds
    for i, phase_value in enumerate(initial_phase):
        if not (p0_bounds[0] <= phase_value.p0 <= p0_bounds[1]):
            errPrint(f"Warning: Initial phase value for axis {i+1} P0 ({phase_value.p0}) is out of bounds {p0_bounds}. Adjusting to midpoint.")
            initial_phase[i] = Phase((p0_bounds[0] + p0_bounds[1]) / 2, initial_phase[i].p1)
        if not (p1_bounds[0] <= phase_value.p1 <= p1_bounds[1]):
            errPrint(f"Warning: Initial phase value for axis {i+1} P1 ({phase_value.p1}) is out of bounds {p1_bounds}. Adjusting to midpoint.")
            initial_phase[i] = Phase(initial_phase[i].p0, (p1_bounds[0] + p1_bounds[1]) / 2)

    # ------------------------------- Initial Guess ------------------------------ #

    # ----------------------------------- NOTE ----------------------------------- #
    # NOTE: Modify initial peak values and bounds later on, currently changed as needed

    spectral_widths = input_spectrum._spectral_widths
    time_size_pts = input_spectrum._total_time_points
    peak_count = len(input_spectrum.peaks)

    starting_x = (initial_decay[0]/spectral_widths[0]) * time_size_pts[0]

    starting_y = (initial_decay[1]/spectral_widths[1]) * time_size_pts[1]

    # Initial parameter bounds
    lower_bound_x = (decay_bounds_x[0]/spectral_widths[0]) * time_size_pts[0]
    
    upper_bound_x = (decay_bounds_x[1]/spectral_widths[0]) * time_size_pts[0]

    lower_bound_y = (decay_bounds_y[0]/spectral_widths[1]) * time_size_pts[1]
    
    upper_bound_y = (decay_bounds_y[1]/spectral_widths[1]) * time_size_pts[1]

    # Collect initial parameters
    initial_peak_lw_x = [starting_x] * peak_count # Initial peak x linewidths
    initial_peak_lw_y = [starting_y] * peak_count # Initial peak y linewidths
    initial_phase_x = ([initial_phase[0].p0] * peak_count) \
                    + ([initial_phase[0].p1] * peak_count) # Initial x-axis p0 and p1 phase
    initial_phase_y = ([initial_phase[1].p0] * peak_count) \
                    + ([initial_phase[1].p1] * peak_count) # Initial y-axis p0 and p1 phase
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
        bounds_lsq_low = ([lower_bound_x] * peak_count) \
                       + ([lower_bound_y] * peak_count) \
                       + ([amplitude_bounds[0]] * peak_count) \
                       + ([p0_bounds[0]] * peak_count) \
                       + ([p1_bounds[0]] * peak_count) \
                       + ([p0_bounds[0]] * peak_count) \
                       + ([p1_bounds[0]] * peak_count) 
        
        # Collect upper bounds for x linewidth, y linewidth, lower amplitude, and lower p0 and p1 for both axes
        bounds_lsq_high = ([upper_bound_x] * peak_count) \
                       + ([upper_bound_y] * peak_count) \
                       + ([amplitude_bounds[1]] * peak_count) \
                       + ([p0_bounds[1]] * peak_count) \
                       + ([p1_bounds[1]] * peak_count) \
                       + ([p0_bounds[1]] * peak_count) \
                       + ([p1_bounds[1]] * peak_count) 
        
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
        errPrint(f"X Line-widths = {initial_peak_lw_x[0]}")
        errPrint(f"Y Line-widths = {initial_peak_lw_y[0]:.3f}")
        errPrint(f"Peak Heights = {initial_peak_heights[0]:.3f}")
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

    
    optimization_args = (input_spectrum, model_function, input_fid, target_interferogram, sim_params)
    if method == 'lsq':
        verbose = 2 if input_spectrum.verbose else 0
        sim_params["difference_equation"] = lambda target, simulated: (target - simulated).flatten()
        optimization_args = (input_spectrum, model_function, input_fid, target_interferogram, sim_params)
        result = least_squares(objective_function, initial_params, '2-point',
                            bounds, 'trf', args=optimization_args, verbose=verbose, max_nfev=trial_count)
    elif method == 'basin':
        disp = True if input_spectrum.verbose else False
        result = basinhopping(objective_function, initial_params, niter=trial_count, stepsize=step_size, 
                        minimizer_kwargs={"args": optimization_args},
                        disp=disp)
    elif method == 'minimize':
        disp = True if input_spectrum.verbose else False
        result = minimize(objective_function, initial_params, 
                        args=optimization_args,
                        method='SLSQP', options={"disp":disp})
    else:
        disp = True if input_spectrum.verbose else False
        x0, fval, grid, Jout = brute(objective_function, bounds, args=optimization_args, Ns=20, full_output=True, workers=-1, disp=disp)
    
    if method == 'brute':
        optimized_params = x0
    else:
        # Extract the optimized parameters
        optimized_params = result.x

    # optimized_params = x0
    optimized_peak_lw_x = optimized_params[:peak_count]
    optimized_peak_lw_y = optimized_params[peak_count:2*peak_count]
    optimized_peak_heights = optimized_params[2*peak_count:3*peak_count]
    optimized_phase_x = (optimized_params[3*peak_count:4*peak_count], # X-axis p0 value
                         optimized_params[4*peak_count:5*peak_count]) # X-axis p1 value
    optimized_phase_y = (optimized_params[5*peak_count:6*peak_count], # Y-axis p0 value
                         optimized_params[6*peak_count:])             # Y-axis p1 value
    
    if input_spectrum.verbose:
        errPrint("")
        errPrint("Optimized Parameters:")
        errPrint(f"X Line-widths = {optimized_peak_lw_x}")
        errPrint(f"Y Line-widths = {optimized_peak_lw_y}")
        errPrint(f"Peak Heights = {optimized_peak_heights}")
        errPrint(f"X Phase = {optimized_phase_x}")
        errPrint(f"Y Phase = {optimized_phase_y}")
        errPrint("")
        
    # Save optimized parameters to a text file with comma separated values
    with open("optimized_parameters.csv", "w") as file:
        file.write("X Line-widths,Y Line-widths,Peak Heights,X Phase P0,X Phase P1,Y Phase P0,Y Phase P1\n")
        for i in range(peak_count):
            file.write(f"{optimized_peak_lw_x[i]},{optimized_peak_lw_y[i]},{optimized_peak_heights[i]},"
                       f"{optimized_phase_x[0][i]},{optimized_phase_x[1][i]},"
                       f"{optimized_phase_y[0][i]},{optimized_phase_y[1][i]}\n")

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
    
    return optimized_spectrum

def objective_function(params, input_spectrum : Spectrum, model_function : ModelFunction, input_fid : pype.DataFrame, target_interferogram : pype.DataFrame, options : dict = {}) -> np.float32:
    """
    Function used for optimizing peak linewidths and heights of spectrum

    Parameters
    ----------
    params : list
        List of target parameters being tested (peak indices, peak linewidths, peak heights)
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

    # Unpack parameters
    peak_x_lws = params[:peak_count] # X-axis peak linewidths
    peak_y_lws = params[peak_count:2*peak_count] # Y-axis peak linewidths
    peak_heights = params[2*peak_count:3*peak_count] # Peak heights
    phase_x = (params[3*peak_count:4*peak_count], # X-axis p0 value
               params[4*peak_count:5*peak_count]) # X-axis p1 value
    phase_y = (params[5*peak_count:6*peak_count], # Y-axis p0 value
               params[6*peak_count:])             # Y-axis p1 value

    # Update peaks with the current parameters
    for i, peak in enumerate(input_spectrum.peaks):
        peak.linewidths.x.pts = peak_x_lws[i]
        peak.linewidths.y.pts = peak_y_lws[i]
        peak.intensity = peak_heights[i]
        xPhase = Phase(phase_x[0][i], phase_x[1][i])
        yPhase = Phase(phase_y[0][i], phase_y[1][i])
        peak.phase = [xPhase, yPhase]

    ctrs = options.get("constant_time_region_sizes", [0,0] )
    offsets = options.get("offsets", [0,0])
    scaling_factors = options.get("scaling_factors", [1.0, 1.0])
    # Generate the simulated interferogram
    simulated_interferogram = input_spectrum.spectral_simulation(model_function, target_interferogram, input_fid, 2, 0, 'ft1', ctrs, False, offsets=offsets, scaling_factors=scaling_factors)

    # Calculate the difference between the target and simulated interferograms
    difference_equation = options.get('difference_equation', lambda target, simulated: np.sum((target - simulated) ** 2))
    difference = difference_equation(target_interferogram.array, simulated_interferogram)

    return difference

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
        
    # Save optimized parameters to a text file with comma separated values
    with open("optimized_parameters.csv", "w") as file:
        file.write("X_LW_exp,X_LW_gauss,Y_LW,Peak_H,X_P0_exp,X_P0_gauss,X_P1_exp,X_P1_gauss,Y_P0_exp,Y_P0_gauss,Y_P1_exp,Y_P1_gauss\n")
        for i in range(peak_count):
            file.write(f"{optimized_peak_lw_x[i]},{optimized_peak_lw_x[i+peak_count]}," +
                       f"{optimized_peak_lw_y[i]},{optimized_peak_lw_y[i+peak_count]}," +
                       f"{optimized_peak_heights[i]}," +
                       f"{optimized_phase_x[0][i]},{optimized_phase_x[0][i+peak_count]}," +
                       f"{optimized_phase_x[1][i]},{optimized_phase_x[1][i+peak_count]}," +
                       f"{optimized_phase_y[0][i]},{optimized_phase_y[0][i+peak_count]}," +
                       f"{optimized_phase_y[1][i]},{optimized_phase_y[1][i+peak_count]}\n")

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

# ---------------------------------------------------------------------------- #
#                               Helper Functions                               #
# ---------------------------------------------------------------------------- #
def string_to_peak(peak_string : str,
                   attribute_dict : dict[str,str], 
                   file_name : Path, 
                   line_number : int,
                   spectral_features : tuple[tuple[float, float], 
                                             tuple[float, float], 
                                             tuple[float, float],
                                             tuple[  int,   int]]) -> Peak | None:
    """
    Convert a string of data into a peak class object.
    Utilizes the attribute dictionary for accurate datatype conversion.

    Parameters
    ----------
    peak_string : str
        String of separated values to comprise the peak
    attribute_dict : dict[str,str]
        Dictionary of all the peak attributes and their datatypes
    file : Path
        Original file from which data was read for debugging purposes
    line_number : int
        Line number the data originates from for debugging purposes

    Returns
    -------
    Peak | None
        New peak container based on string data input if created
    """
    peak_data = re.split(r"[,|\s]+", peak_string.strip())

    # If line is empty do not return a peak
    if len(peak_data) == 1 and not peak_data[0]:
        return None
    
    # Ensure that there is enough data for the keys
    if len(attribute_dict) > len(peak_data):
        raise Exception(f"An exception occured reading line {line_number} of the file: {file_name}\nMore VARS than provided data.")

    # Change data to correct datatype
    corrected_peak_data = list(map(format_to_datatype, peak_data, attribute_dict.values()))

    # Create the final dictionary
    data = dict(map(lambda key,value : (key,value), attribute_dict.keys(), corrected_peak_data))
    
    # Initialize x coordinate based on spectral information
    x_coord = Coordinate(data["X_AXIS"], 
                         spectral_features[0][0], spectral_features[1][0],
                         spectral_features[2][0], spectral_features[3][0])
    
    # Initialize y coordinate based on spectral information
    y_coord = Coordinate(data["Y_AXIS"], 
                         spectral_features[0][1], spectral_features[1][1],
                         spectral_features[2][1], spectral_features[3][1])
    
    # Collect x-coordinate and y-coordinate to represent peak position
    peakPosition = Coordinate2D(data["INDEX"], x_coord, y_coord)

    x_linewidth = Coordinate(data["XW"],
                             spectral_features[0][0], spectral_features[1][0],
                             spectral_features[2][0], spectral_features[3][0])

    y_linewidth = Coordinate(data["YW"],
                             spectral_features[0][1], spectral_features[1][1],
                             spectral_features[2][1], spectral_features[3][1])

    linewidths = Coordinate2D(data["INDEX"], x_linewidth, y_linewidth)

    # Check for x and y cosine couplings if they exist
    x_cos_couplings = [coupl for coupl in data if re.fullmatch(r'X_COSJ\d+', coupl)]
    y_cos_couplings = [coupl for coupl in data if re.fullmatch(r'Y_COSJ\d+', coupl)]

    # Check for x and y sine couplings if they exist
    x_sin_couplings = [coupl for coupl in data if re.fullmatch(r'X_SINJ\d+', coupl)]
    y_sin_couplings = [coupl for coupl in data if re.fullmatch(r'Y_SINJ\d+', coupl)]

    x_cosj : list[float] = [] # X-axis cosine j-couplings
    y_cosj : list[float] = [] # Y-axis cosine j-couplings
    x_sinj : list[float] = [] # X-axis sine j-couplings
    y_sinj : list[float] = [] # Y-axis sine j-couplings

    if x_cos_couplings:
        for x_cos_key in x_cos_couplings:
            x_cosj.append(data[x_cos_key])
    if y_cos_couplings:
        for y_cos_key in y_cos_couplings:
            y_cosj.append(data[y_cos_key])
    if x_sin_couplings:
        for x_sin_key in x_sin_couplings:
            x_sinj.append(data[x_sin_key])
    if y_sin_couplings:
        for y_sin_key in y_sin_couplings:
            y_sinj.append(data[y_sin_key])

    extra_params = {'X_COSJ':x_cosj, 'Y_COSJ':y_cosj, 
                    'X_SINJ':x_sinj, 'Y_SINJ':y_sinj}

    # Return peak with position, intensity, and linewidth
    return Peak(peakPosition, data["HEIGHT"], linewidths, 
                extra_params=extra_params)

def format_to_datatype(data : str, format : str):
    """
    Convert data represented by a string into its correct format using
    the provided format string.

    Parameters
    ----------
    data : str
        The data to be converted.
    format : str
        The format string indicating the desired data type.

    Returns
    -------
    any
        The data converted to the appropriate type.
    """
    if ('f' in format.lower()) or ('e' in format.lower()):
        return float(re.sub("[A-DF-Za-df-z]", "", data))
    elif ('d' in format.lower()) or ('i' in format.lower()):
        return int(re.sub("[A-DF-Za-df-z]", "", data))
    if ('s' in format.lower()) or ('c' in format.lower()):
        return data
    return data
    