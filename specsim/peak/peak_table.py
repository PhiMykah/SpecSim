from .peak import Peak
from pathlib import Path
import re
from typing import Any
from ..datatypes import Vector, PointUnits
type File = str | Path

MAX_DIMENSIONS = 4 # Maximum number of dimensions for a peak
AXIS : dict[int, str] = {0:'X', 1:'Y', 2:'Z', 3:'A'} # Axis names for each dimension

def load_peak_table(file : File, 
                    spectral_widths : Vector[float], 
                    coordinate_origins : Vector[float], 
                    observation_frequencies : Vector[float], 
                    total_frequency_points : Vector[int]) -> tuple:
    """
    Reads from peak table and strips unnecessary lines from process

    Parameters
    ----------
    file : File
        The name of the file to load.
    spectral_widths : Vector[float]
        Spectral widths for each dimension.
    coordinate_origins : Vector[float]
        Coordinate origins for each dimension.
    observation_frequencies : Vector[float]
        Observation frequencies for each dimension.
    total_frequency_points : Vector[int]
        Total frequency points for each dimension.

    Returns
    -------
    peaks : list[Peak]
        List of Peak objects created from the file data.
    remarks : str
        Remarks from the file.
    null_string : str
        Null string value from the file.
    null_value : float
        Null value from the file.
    attributes : dict[str, str]
        Dictionary of attributes and their values.
    """
    if isinstance(file, str):
        file = Path(file)

    file_lines : list[str] = [] # List of each line in file as a string

    with file.open('r') as f:
        file_lines = f.readlines()

    attribute_keys : list[str] = []     # Attribute variable names
    attribute_formats : list[str] = []  # Attribute format values
    peaks : list[Peak] = []             # List of peaks 
    remarks : str = ""               # Remarks from the file
    null_string : str = ""          # Null string value from the file
    null_value : int = 0            # Null value from the file
    attributes : dict[str, str] = {} # Dictionary of attributes and their values

    # ----------------------------- Header Processing ---------------------------- #

    line_number = 0          # Current line of the file
    is_header_read = False  # Whether the header has been read completely or not

    # Iterate through each line and ensure that information
    # Goes to the right property
    while not is_header_read and file_lines:
        line : str = file_lines[0]

        # Exit at the end of the file
        if not line:
            break
        
        # Increment line number
        line_number += 1 
        # Remove whitespace
        line : str = line.strip()

        # Only continue if the line isn't empty
        if not line:
            del file_lines[0]
            continue

        if line.upper().startswith("REMARK") or line.upper().startswith("#"):
            # Collect remarks into a single string
            remarks += re.sub("(REMARK|#)( )+", "", line, flags=re.IGNORECASE) + "\n"
            del file_lines[0]
        elif line.upper().startswith("VARS"):
            # Collect variable names
            attributes_string : str = re.sub("VARS( )+", "", line, flags=re.IGNORECASE)
            attribute_keys = re.split(r"[,|\s]+", attributes_string)
            del file_lines[0]
        elif line.upper().startswith("FORMAT"):
            # Collect matching variable formats
            formats_string : str = re.sub("FORMAT( )+", "", line, flags=re.IGNORECASE).lstrip("%")
            attribute_formats = re.split(r"[,|\s]+%", formats_string)
            del file_lines[0]
        elif line.upper().startswith("NULLSTRING"):
            null_string = re.sub("NULLSTRING( )+", "", line, flags=re.IGNORECASE)
            del file_lines[0]
        elif line.upper().startswith("DATA"):
            remarks += re.sub("(DATA|#)( )+", "", line, flags=re.IGNORECASE) + "\n"
            del file_lines[0]
        elif line.upper().startswith("NULLVALUE"):
            null_value = int(re.sub("NULLVALUE( )+", "", line, re.IGNORECASE))
            del file_lines[0]
        else:
            # If the line doesn't match header keywords, assume it's data and exit loop
            is_header_read = True

            # Ensure equal attribute variables and formats to match
        if (attribute_keys and attribute_formats) and len(attribute_keys) == len(attribute_formats):
            # Generate attribute dictionary
            attributes = dict(
                map(lambda key,value : (key,value), attribute_keys, attribute_formats))

        # // For debugging
        # errPrint(f"Header Length: {line_number}")

    while file_lines:
        line_number += 1
        # Convert the string to a peak and add it to the list
        new_peak : Peak | None = _string_to_peak(
                file_lines.pop(0), attributes, file, line_number,
                spectral_widths, coordinate_origins, 
                observation_frequencies, total_frequency_points)
        
        if new_peak:  
            peaks.append(new_peak) 

    return peaks, remarks, null_string, null_value, attributes



def _string_to_peak(peak_string : str,
                    attribute_dict : dict[str,str], 
                    file_name : Path, 
                    line_number : int,
                    spectral_widths : Vector[float], 
                    coordinate_origins : Vector[float], 
                    observation_frequencies : Vector[float], 
                    total_frequency_points : Vector[int]) -> Peak | None:
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
    spectral_widths : Vector[float]
        Spectral widths for each dimension.
    coordinate_origins : Vector[float]
        Coordinate origins for each dimension.
    observation_frequencies : Vector[float]
        Observation frequencies for each dimension.
    total_frequency_points : Vector[int]
        Total frequency points for each dimension.
        
    Returns
    -------
    Peak | None
        New peak container based on string data input if created
    """
    peak_data : list[str | Any] = re.split(r"[,|\s]+", peak_string.strip())

    # If line is empty do not return a peak
    if len(peak_data) == 1 and not peak_data[0]:
        return None
    
    # Ensure that there is enough data for the keys
    if len(attribute_dict) > len(peak_data):
        raise Exception(f"An exception occured reading line {line_number} of the file: {file_name}\nMore VARS than provided data.")

    # Change data to correct datatype
    corrected_peak_data = list(map(_format_to_datatype, peak_data, attribute_dict.values()))

    # Create the final dictionary
    data = dict(map(lambda key,value : (key,value), attribute_dict.keys(), corrected_peak_data))

    # Check that all spectral parameters have the same length
    if not (len(spectral_widths) == len(coordinate_origins) == len(observation_frequencies) == len(total_frequency_points)):
        raise ValueError("Spectral parameter vectors must have the same length.")

    # Set a variable to the minimum between the length of the spectral parameters and the max dimensions
    num_of_dimensions : int = min(len(spectral_widths), MAX_DIMENSIONS)
    
    coords : list[Any] = [] # List of coordinates for the peak position
    for i in range(num_of_dimensions):
        if f"{AXIS[i]}_AXIS" in data:
            # Initialize coordinate based on spectral parameters if it exists
            coords.append(PointUnits(float(data[f"{AXIS[i]}_AXIS"]), 
                                     spectral_widths[i], coordinate_origins[i],
                                     observation_frequencies[i], total_frequency_points[i]))

    # Collect coordinates to represent peak position
    peakPosition = Vector(index=int(data["INDEX"]), *coords)

    lw_values : list[PointUnits] = []
    for i in range(num_of_dimensions):
        if f"{AXIS[i]}W" in data:
            # Initialize linewidth based on spectral parameters if it exists
            lw_values.append(PointUnits(float(data[f"{AXIS[i]}W"]),
                                        spectral_widths[i], coordinate_origins[i],
                                        observation_frequencies[i], total_frequency_points[i]))

    # Collect linewidths from the data for each dimension
    linewidths = Vector(index=int(data["INDEX"]), *lw_values)

    # Check for x and y cosine couplings if they exist
    x_cos_couplings: list[str] = [coupl for coupl in data if re.fullmatch(r'X_COSJ\d+', coupl)]
    y_cos_couplings: list[str] = [coupl for coupl in data if re.fullmatch(r'Y_COSJ\d+', coupl)]

    # Check for x and y sine couplings if they exist
    x_sin_couplings: list[str] = [coupl for coupl in data if re.fullmatch(r'X_SINJ\d+', coupl)]
    y_sin_couplings: list[str] = [coupl for coupl in data if re.fullmatch(r'Y_SINJ\d+', coupl)]

    x_cosj : list[float] = [] # X-axis cosine j-couplings
    y_cosj : list[float] = [] # Y-axis cosine j-couplings
    x_sinj : list[float] = [] # X-axis sine j-couplings
    y_sinj : list[float] = [] # Y-axis sine j-couplings

    if x_cos_couplings:
        for x_cos_key in x_cos_couplings:
            x_cosj.append(float(data[x_cos_key]))
    if y_cos_couplings:
        for y_cos_key in y_cos_couplings:
            y_cosj.append(float(data[y_cos_key]))
    if x_sin_couplings:
        for x_sin_key in x_sin_couplings:
            x_sinj.append(float(data[x_sin_key]))
    if y_sin_couplings:
        for y_sin_key in y_sin_couplings:
            y_sinj.append(float(data[y_sin_key]))

    extra_params: dict[str, Any] = {
        'X_COSJ': x_cosj if x_cosj else None,
        'Y_COSJ': y_cosj if y_cosj else None,
        'X_SINJ': x_sinj if x_sinj else None,
        'Y_SINJ': y_sinj if y_sinj else None
    }

    # Return peak with position, intensity, and linewidth
    return Peak(peakPosition, float(data["HEIGHT"]), num_of_dimensions, linewidths, 
                **extra_params)

def _format_to_datatype(data : str, format : str) -> float | int | str:
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