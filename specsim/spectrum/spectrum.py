import sys
import re
from pathlib import Path
from ..peak import Peak, Coordinate, Coordinate2D
from ..calculations import fourier_transform, zero_fill, extract_region, hypercomplex_outer_product
import numpy as np
import nmrPype as pype
from typing import Callable, Optional, Annotated

type File = str | Path
type ModelFunction = Callable[[int, int, int, float, float, Optional[np.ndarray], Optional[np.ndarray], float, tuple[int, int], float], np.ndarray]

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
    """
    def __init__(self, peak_table_file : File, 
                 spectral_widths : tuple[float,float], 
                 coordinate_origins : tuple[float,float],
                 observation_freqs : tuple[float,float],
                 total_time_points : tuple[int, int],
                 total_freq_points : tuple[int, int]):
        
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
        self._obervation_freqs = observation_freqs
        self._total_time_points = total_time_points
        self._total_freq_points = total_freq_points

        self.peaks : list[Peak] = self._read_file()
        self._null_string = '*'
        self._null_value = -666

    def spectral_simulation(self, model_function : ModelFunction, 
                            spectrum_data_frame : pype.DataFrame,
                            axis_count : int = 2,
                            peaks_simulated : list[int] | int = 0, 
                            domain : str = 'ft1',
                            constant_time_region_sizes : Annotated[list[int], 2] = [0, 0],
                            phase_values : Annotated[list[tuple[int,int]], 2] = [(0,0),(0,0)],
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
        phase_values : Annotated[list[tuple[int,int]], 2], optional
            multi-axis p0 and p1 phase correction values, by default [(0,0),(0,0)]
        offsets : Annotated[list[int], 2], optional
            multi-axis offset values of the frequency domain in points, by default [0,0]
        scaling_factors : Annotated[list[float], 2], optional
            multi-axis simulation time domain data scaling factor, by default [1.0, 1.0]
        """
        if not (axis_count == len(constant_time_region_sizes) == len(phase_values) == len(offsets) == len(scaling_factors)):
            raise ValueError(f"Axis count and parameter lengths do not match: "
                             f"axis_count={axis_count}, " 
                             f"constant_time_region_sizes={len(constant_time_region_sizes)}, "
                             f"phase_values={len(phase_values)}, "
                             f"offsets={len(offsets)}, "
                             f"scaling_factors={len(scaling_factors)}")
        simulations : list[np.ndarray] = []
        for axis in range(axis_count):
            sim_1D = self.spectral_simulation1D(model_function, peaks_simulated, axis,
                                           constant_time_region_sizes[axis],
                                           phase_values[axis],
                                           offsets[axis],
                                           scaling_factors[axis])
            sim_1D = np.sum(sim_1D, axis=0)
            simulations.append(sim_1D)

        process_iterations = 0
        if domain == 'ft1' and axis_count >= 2:
            process_iterations = 1
        if domain == 'ft2' and axis_count >= 2:
            process_iterations = 2

        for i in range(process_iterations):
            # Zero fill if necessary
            new_size = spectrum_data_frame.getParam("NDFTSIZE", i + 1)
            if new_size > simulations[i].size:
                simulations[i] = zero_fill(simulations[i], new_size)

            # Convert to frequency domain
            simulations[i] = fourier_transform(simulations[i])

            # Extract designated region if necessary
            first_point = int(spectrum_data_frame.getParam("NDX1", i + 1))
            last_point = int(spectrum_data_frame.getParam("NDXN", i + 1))
            if first_point and last_point:
                simulations[i] = extract_region(simulations[i], first_point, last_point)
            
            # Delete imaginary values
            simulations[i] = simulations[i].real

        # Interleave real and imaginary values in indirect dimensions if complex
        if len(simulations) > 1:
            for i in range(1, len(simulations)):
                if np.iscomplexobj(simulations[i]):
                    interleaved_data = np.empty((simulations[i].size * 2,), dtype=simulations[i].real.dtype)
                    interleaved_data[0::2] = simulations[i].real
                    interleaved_data[1::2] = simulations[i].imag
                    simulations[i] = interleaved_data

        if axis_count == 2:
            simulations_2D = np.outer(simulations[1], simulations[0])
            return simulations_2D
        return simulations

    def spectral_simulation1D(self, model_function : ModelFunction,
                           peaks_simulated : list[int] | int = 0, 
                           axis : int = X,
                           constant_time_region_size : int = 0,
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
                                            axis, constant_time_region_size, phase,
                                            offset, scaling_factor))
            
        return np.array(spectrum_list)

    def _run_simulation_iteration(self, model_function : ModelFunction, 
                                  index : int, axis : int, constant_time_region_size : int,
                                  phase : tuple[int,int], 
                                  offset : int, 
                                  scaling_factor : float) -> np.ndarray:
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

        # Obtain frequency and add offset if necessary
        frequency_pts = self.peaks[index].position[axis] + offset

        # Collect line width, and intensity
        line_width_pts = self.peaks[index].linewidths[axis]

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
                           line_width_pts,
                           cos_mod_values=cos_mod_j,
                           sin_mod_values=sin_mod_j,
                           amplitude=amplitude, phase=phase,
                           scale=scaling_factor))

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
        # print(f"Header Length: {line_number}", file=sys.stderr)

        while file_lines:
            line_number += 1
            # Convert the string to a peak and add it to the list
            new_peak = string_to_peak(
                    file_lines.pop(0), self.attributes, self.file, line_number,
                    (self._spectral_widths, self._coordinate_origins,
                     self._obervation_freqs, self._total_time_points))
            
            if new_peak:  
                peaks.append(new_peak)  

        return peaks
    
    # ------------------------------ Magic Methods ----------------------------- #

    def __repr__(self):
        if not self.remarks:
            remarks = "None"
        else:
            remarks = self.remarks
        return f"Spectrum({len(self.peaks)} peaks, " \
               f"file={self.file}, " \
               f"remarks={remarks}, " \
               f"attributes={self.attributes})"


def string_to_peak(peak_string : str,
                   attribute_dict : dict[str,str], 
                   file_name : Path, 
                   line_number : int,
                   spectral_features : tuple[tuple[float, float], 
                                             tuple[float, float], 
                                             tuple[float, float],
                                             tuple[  int,   int]]) -> Peak | None:
    """Convert a string of data into a peak class object.
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
    return Peak(peakPosition, data["HEIGHT"], (data["XW"], data["YW"]), 
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
    