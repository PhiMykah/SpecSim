import sys
import re
from pathlib import Path
from ..peak import Peak, Coordinate, Coordinate2D
import numpy as np
from typing import Callable, Optional

type File = str | Path
type ModelFunction = Callable[[int, int, int, float, float, Optional[np.ndarray], Optional[np.ndarray], float, tuple[int, int], float], np.ndarray]

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


    def spectralSimulation(self, model_function : ModelFunction,
                           peaks_simulated : list[int] | int, 
                           constant_time_region_size : int = 0,
                           phase : tuple[int,int] = (0,0),
                           xOffset : int = 0,
                           scaling_factor : float = 1.0) -> np.ndarray:
        """
        Perform spectral simulation using inputted decay function model

        Parameters
        ----------
        model_function : ModelFunction
            Target modeling function for simulation, e.g. exponential, gaussian
        peaks_simulated : list[int] | int
            Number of peaks to simulate or indices of peaks to simulate. 0 simulates entire spectrum
        constant_time_region_size : int
            Total number of points in the constain time region (pts) (Default 0)
        phase : tuple[int,int]
            p0 and p1 phase values (Default (0,0))
        xOffset : int
            Offset of the frequency domain in points (Default 0)
        scaling_factor : float
            Simulation time domain data scaling factor (Default 1.0)

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
                                            constant_time_region_size, phase,
                                            xOffset, scaling_factor))
            
        return np.array(spectrum_list)


    def _run_simulation_iteration(self, model_function : ModelFunction, 
                                  index : int, constant_time_region_size : int,
                                  phase : tuple[int,int], 
                                  xOffset : int, 
                                  scaling_factor : float) -> np.ndarray:
        """
        Run a single iteration of a spectral time domain simulation.

        Parameters
        ----------
        model_function : ModelFunction
            Model simulation function such as exponential or gaussian
        index : int
            Current iteraiton index
        constant_time_region_size : int
            Total number of points in the constain time region (pts)
        phase : tuple[int,int]
            p0 and p1 phase values
        xOffset : int
            Offset of the frequency domain in points
        scaling_factor : float
            Simulation time domain data scaling factor

        Returns
        -------
        np.ndarray
            Simulation slice based on model function
        """

        # Obtain frequency and add offset if necessary
        frequency_pts = self.peaks[index].position.x + xOffset

        # Collect line width, and intensity
        line_width_pts = self.peaks[index].linewidths[0]
        amplitude = self.peaks[index].intensity

        # Collect cosine moduation if it exists
        if self.peaks[index].extra_params["X_COSJ"]:
            cos_mod_j = np.array(self.peaks[index].extra_params["X_COSJ"])
        else:
            cos_mod_j = None
        
        # Collect sine modulation if it exists
        if self.peaks[index].extra_params["X_SINJ"]:
            sin_mod_j = np.array(self.peaks[index].extra_params["X_SINJ"])
        else:
            sin_mod_j = None

        return(
            model_function(self._total_time_points[0], self._total_freq_points[0],
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

            if line.upper().startswith("REMARK"):
                # Collect remarks into a single string
                self.remarks += re.sub("REMARK( )+", "", line, flags=re.IGNORECASE) + "\n"
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
            peaks.append(
                string_to_peak(
                    file_lines.pop(0), self.attributes, self.file, line_number,
                    (self._spectral_widths, self._coordinate_origins,
                     self._obervation_freqs, self._total_time_points)))  

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
                                             tuple[  int,   int]]) -> Peak:
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
    Peak
        New peak container based on string data input
    """
    peak_data = re.split(r"[,|\s]+", peak_string.strip())

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
    