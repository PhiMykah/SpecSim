from ..datatypes import Vector, PointUnits, Phase
from ..peak import load_peak_table, Peak
from .transform import outer_product_summation
from pathlib import Path
import nmrPype as pype
import numpy as np
from typing import Callable, Any
from enum import Enum
from inspect import Signature, signature
from copy import deepcopy

type File = str | Path
type ModelFunction = Callable[[Peak, int, int, int, PointUnits, list[PointUnits], float, list[Phase], int, np.ndarray | None, np.ndarray | None, float], np.ndarray]

class Domain(Enum):
    FID = 0
    FT1 = 1
    FT2 = 2

    def to_string(self) -> str:
        """
        Converts the domain enum to a string representation.

        Returns
        -------
        str
            String representation of the domain.
        """
        return self.name.lower()

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
    
    spectral_widths : Vector[float]
        Spectral widths for each dimension.

    coordinate_origins : Vector[float]
        Coordinate origins for each dimension.
    observation_frequencies : Vector[float]
        Observation frequencies for each dimension.
    total_frequency_points : Vector[int]
        Total frequency points for each dimension.
    
    peaks : list[Peak]
        Line-by-line string list of all lines from the tab file

    verbose : bool
        Flag to enable verbose mode
    """
    def __init__(self, file : File, spectral_widths : Vector[float], 
                 coordinate_origins: Vector[float], 
                 observation_frequencies: Vector[float], 
                 total_time_points: Vector[int],
                 total_frequency_points: Vector[int], verbose: bool = False) -> None:
        """
        Initialize the Spectrum object.

        Parameters
        ----------
        file : Path
            File directory of peak table
        spectral_widths : Vector[float]
            Spectral widths for each dimension.
        coordinate_origins : Vector[float]
            Coordinate origins for each dimension.
        observation_frequencies : Vector[float]
            Observation frequencies for each dimension.
        total_time_points : Vector[int]
            Total time points for each dimension.
        total_frequency_points : Vector[int]
            Total frequency points for each dimension.
        verbose : bool, optional
            Flag to enable verbose mode (default is False)
        """
        if type(file) == str:
            peak_table_file = Path(file)
        else:
            peak_table_file = file
        if isinstance(peak_table_file, Path) and not peak_table_file.exists():
            raise IOError(f"Unable to find file: {peak_table_file}")
        
        self._file: Path | str = peak_table_file
        
        self._spectral_widths: Vector[float] = spectral_widths
        self._coordinate_origins: Vector[float] = coordinate_origins
        self._observation_frequencies: Vector[float] = observation_frequencies
        self._total_time_points: Vector[int] = total_time_points
        self._total_frequency_points: Vector[int] = total_frequency_points
        peaks, remarks, null_string, null_value, attributes = load_peak_table(file, 
                                                                              self._spectral_widths, self._coordinate_origins,
                                                                              self._observation_frequencies,
                                                                              self._total_frequency_points)
        self._peaks : list[Peak] = peaks
        self._remarks: str = remarks
        self._attributes : dict[str, str] = attributes
        self._null_string : str = null_string
        self._null_value : float = null_value
        self.verbose: bool = verbose
    
    # ---------------------------------------------------------------------------- #
    #                              Spectral Simulation                             #
    # ---------------------------------------------------------------------------- #

    def simulate(self, model_function : ModelFunction, spectrum_data_frame : pype.DataFrame,
                 spectrum_interferogram : pype.DataFrame, 
                 spectrum_fid : pype.DataFrame, basis_set_folder : str | None = None, 
                 dimensions : int = 2,
                 peaks_simulated : list[int] | int | None = None,
                 domain : int = 1, constant_time_region_sizes : Vector[int] | None = None,
                 phase_values : Vector[Phase] | list[Vector[Phase]] | None = None,
                 offsets : Vector[float] | None = None,
                 scaling_factors : Vector[float] | None = None) -> np.ndarray:
        """
        Simulates a multi-dimensional spectrum based on the provided model function and parameters.
        The simulation can be performed in either the time or frequency domain, and can include
        multiple peaks with different phase values, offsets, and scaling factors.

        Parameters
        ----------
        model_function : ModelFunction
            The model function to be used for the simulation.
        spectrum_data_frame : pype.DataFrame
            The data frame to model the simulation on.
        spectrum_interferogram : pype.DataFrame
            The spectrum interferogram to model the simulation on.
        spectrum_fid : pype.DataFrame
            The base fid to perform spectral functions on
        basis_set_folder : str | None
            Folder to designate a basis set, or None if not utilized,
            by default None
        dimensions : int, optional
            Number of dimensions to simulate, by default 2
        peaks_simulated : list[int] | int | None, optional
            Number of peaks simulated or indices of peaks simulated
            None for all peaks, by default None
        domain : int, optional
            Domain of spectral simulation, 
            0 for time domain, 1 for interferogram, and 2 for frequency domain, 
            by default 1
        constant_time_region_sizes : Vector[int] | None, optional
            Number of points in the simulation with a constant time value,
            by default None
        phase_values : Vector[float] | list[Vector[float]] | None, optional
            P0 and P1 phase correction values for each dimension, 
            by default None
        offsets : Vector[float] | None, optional
            Offset values of the frequency domain in points for each dimension, 
            by default None
        scaling_factors : Vector[float] | None, optional
            Simulation time domain data scaling factor for each dimension, 
            by default None

        Returns
        -------
        np.ndarray
            Simulated spectrum data as a numpy array.

        Raises
        ------
        """
        try:
            self._validate_simulation_parameters(model_function, spectrum_data_frame, spectrum_interferogram, 
                                spectrum_fid, dimensions, peaks_simulated, domain,
                                constant_time_region_sizes, phase_values, offsets, scaling_factors)
        except (TypeError, ValueError) as e:
            raise e

        # Perform a 1D simulation for each dimension of the spectrum
        simulations : list[np.ndarray[Any, np.dtype[Any]]] = []
        for dim in range(self._dimensions):
            sim_1D_peaks : np.ndarray[Any, np.dtype[np.float32]] = self.spectral_simulation_1D(
                model_function, dim, self._peaks_simulated, self._constant_time_region_sizes[dim],
                self._phase_values, self._offsets[dim], self._scaling_factors[dim]
            )
            simulations.append(sim_1D_peaks)
        
        process_iterations : int = int(self._domain.value) if dimensions >= 1 else 0

        spectral_data : list = [*simulations]
        fid : pype.DataFrame = spectrum_fid

        # Iterate through all dimensions in need of processing
        for i in range(process_iterations):
            spectral_data[i] = []
            # Go through all peaks in simulation
            for j in range(len(simulations[i])):
                iteration_df = pype.DataFrame(file=fid.file, header=fid.header, array=fid.array)
                iteration_df.array = simulations[i][j]
                dim : int = i + 1

                #  Add first point scaling and window function if necessary
                off_param : float = spectrum_data_frame.getParam("NDAPODQ1", dim)
                end_param : float = spectrum_data_frame.getParam("NDAPODQ2", dim)
                pow_param : float = spectrum_data_frame.getParam("NDAPODQ3", dim)
                elb_param : float = spectrum_data_frame.getParam("NDLB", dim)
                glb_param : float = spectrum_data_frame.getParam("NDGB", dim)
                goff_param : float = spectrum_data_frame.getParam("NDGOFF", dim)
                first_point_scale : float = 1 + spectrum_data_frame.getParam("NDC1", dim)

                iteration_df.runFunc("SP", {"sp_off":off_param, "sp_end":end_param, "sp_pow":pow_param, "sp_elb":elb_param,
                                            "sp_glb":glb_param, "sp_goff":goff_param, "sp_c":first_point_scale})
                # Zero fill if necessary
                iteration_df.runFunc("ZF", {"zf_count":1, 'zf_auto':True})

                # Convert to frequency domain
                iteration_df.runFunc("FT")
                
                # No need to perform phase correction since each peak has been properly phase corrected 

                # Delete imaginary values
                iteration_df.runFunc("DI")
                
                # Extract designated region if necessary
                first_point = int(spectrum_data_frame.getParam("NDX1", dim))
                last_point = int(spectrum_data_frame.getParam("NDXN", dim))
                if first_point and last_point:
                    if first_point != 0 or last_point != 0:
                        if first_point == 1:
                            first_point : int = first_point-1 
                        if first_point <= last_point:
                            iteration_df.array = iteration_df.array[first_point:last_point+1]

                # Add peak to list
                spectral_data[i].append(iteration_df.array)

        if dimensions == 2:
            result : np.ndarray[Any, np.dtype[Any]] = outer_product_summation(spectral_data[0], spectral_data[1])

            self._build_basis_set(basis_set_folder, spectral_data)
            return result
        
        return np.ndarray(spectral_data)


    def spectral_simulation_1D(self, model_function : ModelFunction, dim : int,
                            peaks_simulated : list[int],
                            constant_time_region_size : int,
                            phase_values : list[Vector[Phase]] | None,
                            offset : float,
                            scaling_factor : float) -> np.ndarray:
        """
        Simulates a 1D spectrum based on the provided model function and parameters.
        The simulation can be performed in either the time or frequency domain, and can include
        multiple peaks with different phase values, offsets, and scaling factors.

        Parameters
        ----------
        model_function : ModelFunction
            The model function to be used for the simulation.
        spectrum_data_frame : pype.DataFrame
            The data frame to model the simulation on.
        spectrum_fid : pype.DataFrame
            The base fid to perform spectral functions on
        dim : int
            Dimension of the spectrum to simulate.
        peaks_simulated : list[int]
            Indices of peaks to be simulated.
        domain : Domain
            Domain of spectral simulation.
        constant_time_region_size : int
            Number of points in the simulation with a constant time value.
        phase_values : list[Vector[Phase]]
            List of P0 and P1 phase correction  Vectors.
        offset : int
            Offset value of the frequency domain in points for the dimension.
        scaling_factor : float
            Simulation time domain data scaling factor for the dimension.

        Returns
        -------
        np.ndarray
            Simulated spectrum data as a 1D numpy array.
        """
        spectrum_list : list[Any] = []
        # Iterate over the peaks to be simulated
        for i in peaks_simulated:
            # Get the peak position in the current dimension
            frequency : float = self._peaks[i].position[dim] + offset

            # Get the peak linewidths in the current dimension
            linewidths : list[PointUnits] = []
            for linewidth in self._peaks[i].linewidth:
                linewidths.append(linewidth[dim])

            # Get the peak intensity in the current dimension
            amplitude : float = self._peaks[i].intensity

            # Update phase values if provided
            if phase_values:
                self._peaks[i].phase = phase_values
            
            simulation_phases : list[Phase] = []
            for vector in self._peaks[i].phase:
                simulation_phases.append(vector[dim])

            # Set modulation dim to X, Y, etc
            cosine_modulation : str = f"{chr(dim + ord('X'))}_COSJ"
            sine_modulation : str = f"{chr(dim + ord('X'))}_SINJ"

            # Collect cosine moduation if it exists
            cos_mod_j : np.ndarray[Any, np.dtype[Any]] | None = np.array(self._peaks[i].extra_params.get(cosine_modulation, None)) \
                if self._peaks[i].extra_params.get(cosine_modulation) else None
            
            # Collect sine modulation if it exists
            sin_mod_j : np.ndarray[Any, np.dtype[Any]] | None = np.array(self._peaks[i].extra_params.get(sine_modulation)) \
                if self._peaks[i].extra_params.get(sine_modulation) else None

            spectrum_list.append(
                model_function(self._peaks[i], dim, self._total_time_points[dim], self._total_frequency_points[dim],
                               frequency, linewidths, amplitude, simulation_phases, constant_time_region_size,
                                cos_mod_j, sin_mod_j, scaling_factor)
                )
        # Combine the simulated peaks into a single spectrum
        return np.array(spectrum_list)

    # ---------------------------------------------------------------------------- #
    #                               Helper Functions                               #
    # ---------------------------------------------------------------------------- #

    def _build_basis_set(self, basis_folder : str | None, spectral_data : list[np.ndarray]) -> None:
        """
        Generate a basis set from the data

        Parameters
        ----------
        basis_set_folder : str | None
            Folder to designate a basis set
        spectral_data : list[np.ndarray]
            _description_
        """
        if not basis_folder:
            return 
        
        x_axis : np.ndarray[Any, np.dtype[Any]] = spectral_data[0]
        y_axis : np.ndarray[Any, np.dtype[Any]] = spectral_data[1]    
        y_length : int = y_axis.shape[-1]

        if np.iscomplexobj(y_axis):
            interleaved_data : np.ndarray[Any, np.dtype[Any]] = np.zeros(y_axis.shape[:-1] + (y_length * 2,), dtype=y_axis.real.dtype)
            for i in range(len(interleaved_data)):
                interleaved_data[i][0::2] = y_axis[i].real
                interleaved_data[i][1::2] = y_axis[i].imag

            y_axis = interleaved_data
            y_length = y_length * 2

        # Create the x-axis y-axis pairings
        planes : list[Any] = []
        peak_count : int = len(self._peaks)

        for i in range(peak_count):
            plane : np.ndarray[Any, np.dtype[Any]] = np.outer(y_axis[i], x_axis[i])
            if self._domain == Domain.FT2:
                iteration_df = pype.DataFrame(file=self._spectrum_data_frame.file, header=self._spectrum_data_frame.header, array=plane)
            elif self._domain == Domain.FID:
                iteration_df = pype.DataFrame(file=self._spectrum_fid.file, header=self._spectrum_fid.header, array=plane)
            else:
                iteration_df = pype.DataFrame(file=self._spectrum_interferogram.file, header=self._spectrum_interferogram.header, array=plane)
            
            # Set plane datatype to float32 if float64 and complex64 if complex128
            if plane.dtype == np.float64:
                plane = plane.astype(np.float32)
            elif plane.dtype == np.complex128:
                plane = plane.astype(np.complex64)

            iteration_df.setArray(plane)
            # Ensure output directory exists
            output_dir = Path(basis_folder)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save the plane to a file
            max_digits: int = len(str(peak_count))
            output_file : Path = output_dir / f"{i + 1:0{max_digits}}.{self._domain.to_string()}"

            pype.write_to_file(iteration_df, str(output_file) , True)
            planes.append(plane)


    def _validate_simulation_parameters(self, model_function, spectrum_data_frame, spectrum_interferogram,
                        spectrum_fid, dimensions, peaks_simulated, domain, constant_time_region_sizes,
                        phase_values, offsets, scaling_factors) -> None:
        """
        Helper function to validate simulation parameters.

        Parameters
        ----------
        model_function : ModelFunction
            The model function to be used for the simulation.
        spectrum_data_frame : pype.DataFrame
            The data frame to model the simulation on.
        spectrum_interferogram : pype.DataFrame
            The spectrum interferogram to model the simulation on.
        spectrum_fid : pype.DataFrame
            The base fid to perform spectral functions on.
        dimensions : int
            Number of dimensions to simulate.
        peaks_simulated : list[int] | int | None
            Number of peaks simulated or indices of peaks simulated.
        domain : int
            Domain of spectral simulation.
        constant_time_region_sizes : Vector[int] | None
            Number of points in the simulation with a constant time value.
        phase_values : Vector[float] | list[Vector[float]] | None
            P0 and P1 phase correction values for each dimension.
        offsets : Vector[float] | None
            Offset values of the frequency domain in points for each dimension.
        scaling_factors : Vector[float] | None
            Scaling factors for the simulation.

        Raises
        ------
        TypeError, ValueError
            If any parameter is invalid.
        """
        # ------------------------------ model_function ------------------------------ #
        if not callable(model_function):
            raise TypeError("model_function must be a callable function.")
        # Check the signature of the model_function
        model_function_signature: Signature = signature(model_function)
        expected_parameters: list[str] = [
            'peak', 'dimension', 'time_domain_size', 'frequency_domain_size', 'frequency', 'linewidths',
            'amplitude', 'phases', 'constant_time_region_size', 'cos_mod_values', 'sin_mod_values', 'scale'
        ]
        if list(model_function_signature.parameters.keys()) != expected_parameters:
            raise TypeError(f"model_function must have the following parameters: {', '.join(expected_parameters)}")
        if model_function_signature.return_annotation != np.ndarray:
            raise TypeError("model_function must return a numpy ndarray.")
        
        # ---------------------------- spectrum_data_frame --------------------------- #
        if not isinstance(spectrum_data_frame, pype.DataFrame):
            raise TypeError("spectrum_data_frame must be an nmrPype DataFrame.")
        
        # ------------------------------- spectrum_fid ------------------------------- #
        if not isinstance(spectrum_fid, pype.DataFrame):
            raise TypeError("spectrum_fid must be an nmrPype DataFrame.")
        
        # -------------------------------- dimensions -------------------------------- #
        if isinstance(dimensions, str):
            try:
                dimensions = int(dimensions)
            except ValueError:
                raise ValueError("dimensions must be a positive integer.")
        if not isinstance(dimensions, int) or not (1 <= dimensions <= 4):
            raise ValueError("dimensions must be an integer between 1 and 4.")
        if dimensions > len(self._spectral_widths):
            raise ValueError(f"dimensions must be less than or equal to the number of spectral widths ({len(self._spectral_widths)}).")
        
        # ------------------------------ peaks_simulated ----------------------------- #
        if peaks_simulated is not None:
            if isinstance(peaks_simulated, int):
                if peaks_simulated == 0:
                    peaks_simulated = list(range(len(self._peaks)))
                elif peaks_simulated < 0: 
                    peaks_simulated = list(range(min(abs(peaks_simulated), len(self._peaks))))
                else:
                    peaks_simulated = list(range(min(peaks_simulated, len(self._peaks))))
            if not isinstance(peaks_simulated, list):
                raise TypeError("peaks_simulated must be a list of integers or None.")
            for peak in peaks_simulated:
                if not isinstance(peak, int) or peak < 0 or peak >= len(self._peaks):
                    raise ValueError(f"peaks_simulated must be a list of integers between 0 and {len(self._peaks)-1}.")
        else:
            peaks_simulated = list(range(len(self._peaks)))

        # ---------------------------------- domain ---------------------------------- #
        if not isinstance(domain, int):
            raise ValueError("domain must be an integer: 0 (time), 1 (interferogram), or 2 (frequency).")
        # convert domain to enum using match case
        match domain:
            case 0:
                domain = Domain.FID
            case 1:
                domain = Domain.FT1
            case 2:
                domain = Domain.FT2
            case _:
                raise ValueError("domain must be an integer: 0 (fid), 1 (ft1), or 2 (ft2).")
            
        # ------------------------ constant_time_region_sizes ------------------------ #
        if constant_time_region_sizes is not None:
            if not isinstance(constant_time_region_sizes, Vector) or len(constant_time_region_sizes) != dimensions:
                raise TypeError("constant_time_region_sizes must be a Vector of integers with length equal to dimensions.")
            for size in constant_time_region_sizes:
                if not isinstance(size, int) or size < 0:
                    raise ValueError("constant_time_region_sizes must be a Vector of positive integers.")
        else:
            default_constant_region : list[int] = [0] * dimensions
            constant_time_region_sizes = Vector(default_constant_region)

        # ------------------------------- phase_values ------------------------------- #
        phase_vals : list[Vector[Phase]] | None = None 
        if phase_values is not None:
            if isinstance(phase_values, float) or isinstance(phase_values, int):
                # Clamp value to -180 to 180 degrees
                new_phase: float = max(Phase.MIN, min(Phase.MAX, phase_values))
                vector : list[Phase] = [Phase(new_phase, new_phase)] * dimensions
                phase_vals : list[Vector[Phase]] | None = [Vector(vector)]
            elif isinstance(phase_values, Vector):
                # Ensure vector is of type Phase
                if not all(isinstance(phase, Phase) for phase in phase_values):
                    raise TypeError("phase_values must be a Vector of Phases.")
                # Check that all elements are of the same length as dimensions
                if len(phase_values) != dimensions:
                    raise ValueError(f"phase_values must be a Vector of Phases with length equal to {dimensions}.")
                phase_vals : list[Vector[Phase]] | None = [phase_values]
            elif isinstance(phase_values, list):
                # Check that all elements are of type Vector
                if not all(isinstance(value, Vector) for value in phase_values):
                    raise TypeError("phase_values must be a list of Phase Vectors.")
                # Check that all elements are of the same length as dimensions\
                if not all(len(value) == dimensions for value in phase_values):
                    raise ValueError(f"phase_values must be a list of Phase Vectors with length equal to {dimensions}.")
                # Check that all elements are of type Phase
                if not all(all(isinstance(phase, Phase) for phase in value) for value in phase_values):
                    raise TypeError("phase_values must be a list of Phase Vectors.")
            else:
                raise TypeError("phase_values must be a float, int, Phase Vector, or list of Phase Vectors.")
        else:
            phase_vals : list[Vector[Phase]] | None = None

        # ---------------------------------- offsets --------------------------------- #
        if offsets is not None:
            if not isinstance(offsets, Vector) or len(offsets) != dimensions:
                raise TypeError("offsets must be a Vector of integers with length equal to dimensions.")
            for offset in offsets:
                if not isinstance(offset, (int, float)) or offset < 0:
                    raise ValueError("offsets must be a Vector of positive integers or floats.")
            new_offsets : Vector[float] = offsets
        else:
            zero_vector : list[float] = [0.0] * dimensions
            new_offsets : Vector[float] = Vector(zero_vector)

        # ------------------------------ scaling_factors ----------------------------- #
        if scaling_factors is not None:
            if not isinstance(scaling_factors, Vector) or len(scaling_factors) != dimensions:
                raise TypeError("scaling_factors must be a Vector of floats with length equal to dimensions.")
            for factor in scaling_factors:
                if not isinstance(factor, (int, float)) or factor < 0:
                    raise ValueError("scaling_factors must be a Vector of positive floats.")
        else:
            zero_vector = [1] * dimensions
            scaling_factors = Vector(zero_vector)

        # Set the attributes
        self._model_function : Callable[..., object] = model_function
        self._spectrum_data_frame : pype.DataFrame = spectrum_data_frame
        self._spectrum_interferogram : pype.DataFrame = spectrum_interferogram
        self._spectrum_fid : pype.DataFrame = spectrum_fid
        self._dimensions : int = dimensions
        self._peaks_simulated : list[int] = peaks_simulated
        self._domain : Domain = domain
        self._constant_time_region_sizes : Vector[int] = constant_time_region_sizes
        self._phase_values : list[Vector[Phase]] | None = phase_vals
        self._offsets : Vector[float] = new_offsets
        self._scaling_factors : Vector[float] = scaling_factors

    def update_peaks(self, decay_list : list, 
                     phase_list : list, peak_height_list : list, weights : list, 
                     unpack_count : int, num_of_dimensions : int) -> None:
        """
        Updates the peaks with the provided decay, phase, and intensity parameters.

        Parameters
        ----------
        decay_list : list
            A list of decay values used to update the linewidth of each peak.
        phase_list : list
            A list of phase values used to update the p0 and p1 phase parameters
        peak_height_list : list
            A list of intensity values used to update the intensity of each peak.
        unpack_count : int
            The number of unpacking iterations to perform for each peak.
        num_of_dimensions : int
            The number of dimensions to consider when updating the peaks.

        Returns
        -------
        None
        """
        peak_count : int = len(self.peaks)

        phase_offset : int = 2 * peak_count

        if unpack_count > 1:
            for peak in self.peaks:
                peak.expand_linewidth(unpack_count)
                peak.expand_phase(unpack_count)
                
        # Update peaks with current parameters
        for i, peak in enumerate(self.peaks):
            for p in range(unpack_count):
                for j in range(num_of_dimensions):
                    lw_index : int = i + (j * peak_count) + (2 * p * peak_count)
                    peak.linewidth[p][j].pts = decay_list[lw_index]
                    
                    p0_index : int = i + (j * phase_offset) + (2 * p * phase_offset)
                    p1_index : int = peak_count + i + (j * phase_offset) + (2 * p * phase_offset)
                    peak.phase[p][j].p0 = phase_list[p0_index]
                    peak.phase[p][j].p1 = phase_list[p1_index]
                    
            peak.intensity = peak_height_list[i]

        if weights is None or len(weights) == 0:
            return
        
        for i, peak in enumerate(self.peaks):
            for p in range(unpack_count - 1):
                for j in range(num_of_dimensions):
                    if peak.weights is not None:
                        weight_index : int = i + (j * peak_count) + (2 * p * peak_count)
                        peak.weights[p][j] = weights[weight_index]
    
    # ---------------------------------------------------------------------------- #
    #                              Getters and Setters                             #
    # ---------------------------------------------------------------------------- #

    # ----------------------------------- File ----------------------------------- #
    
    @property
    def file(self) -> Path | str:
        return self._file
    
    @file.setter
    def file(self, file: File) -> None:
        if type(file) == str:
            peak_table_file = Path(file)
        else:
            peak_table_file = file
        if isinstance(peak_table_file, Path) and not peak_table_file.exists():
            raise IOError(f"Unable to find file: {peak_table_file}")
        self._file = peak_table_file
        peaks, remarks, null_string, null_value, attributes = load_peak_table(file, 
                                                                        self._spectral_widths, self._coordinate_origins,
                                                                        self._observation_frequencies,
                                                                        self._total_frequency_points)
        self._peaks = peaks
        self._remarks = remarks
        self._attributes = attributes
        self._null_string = null_string
        self._null_value = null_value
        
    # ----------------------------------- Peaks ---------------------------------- #

    @property
    def peaks(self) -> list[Peak]:
        return self._peaks
    
    @peaks.setter
    def peaks(self, peaks: list[Peak]) -> None:
        if not isinstance(peaks, list):
            raise TypeError("Peaks must be a list of Peak objects.")
        self._peaks = peaks

    # -------------------------------- Remarks ---------------------------------- #

    @property
    def remarks(self) -> str:
        return self._remarks
    
    @remarks.setter
    def remarks(self, remarks: str) -> None:
        if not isinstance(remarks, str):
            raise TypeError("Remarks must be a string.")
        self._remarks = remarks

    # ------------------------------- Attributes -------------------------------- #
    
    @property
    def attributes(self) -> dict[str, str]:
        return self._attributes
    
    @attributes.setter
    def attributes(self, attributes: dict[str, str]) -> None:
        if not isinstance(attributes, dict):
            raise TypeError("Attributes must be a dictionary.")
        self._attributes = attributes

    # -------------------------------- Null String ------------------------------- #

    @property
    def null_string(self) -> str:
        return self._null_string
    
    @null_string.setter
    def null_string(self, null_string: str) -> None:
        if not isinstance(null_string, str):
            raise TypeError("Null string must be a string.")
        self._null_string = null_string

    # -------------------------------- Null Value -------------------------------- #

    @property
    def null_value(self) -> float:
        return self._null_value
    
    @null_value.setter
    def null_value(self, null_value: float) -> None:
        if not isinstance(null_value, (int, float)):
            raise TypeError("Null value must be an integer or float.")
        self._null_value = null_value
    
    # ---------------------------------------------------------------------------- #
    #                                 Magic Methods                                #
    # ---------------------------------------------------------------------------- #

    def __repr__(self) -> str:
        return f"Spectrum(file={self._file}, spectral_widths={self._spectral_widths}, " \
               f"coordinate_origins={self._coordinate_origins}, observation_frequencies={self._observation_frequencies}, " \
               f"total_frequency_points={self._total_frequency_points}, peaks={self._peaks})"
    
    def __str__(self) -> str:
        return f"Spectrum File: {self._file}\n" \
               f"Spectral Widths: {self._spectral_widths}\n" \
               f"Coordinate Origins: {self._coordinate_origins}\n" \
               f"Observation Frequencies: {self._observation_frequencies}\n" \
               f"Total Frequency Points: {self._total_frequency_points}\n" \
               f"Peaks: {len(self._peaks)} peaks\n" \
               f"Remarks: {self._remarks}\n" \
               f"Attributes: {self._attributes}\n" \
               f"Null String: {self._null_string}\n" \
               f"Null Value: {self._null_value}\n" \
               f"Verbose: {self.verbose}\n"
    
    def __deepcopy__(self, memo) -> "Spectrum":
        """
        Creates a deep copy of the current Spectrum object.

        Returns
        -------
        Spectrum
            A new Spectrum object with the same attributes as the current one.
        """

        new_spectrum = Spectrum(
            file=self._file,
            spectral_widths=deepcopy(self._spectral_widths),
            coordinate_origins=deepcopy(self._coordinate_origins),
            observation_frequencies=deepcopy(self._observation_frequencies),
            total_time_points=deepcopy(self._total_time_points),
            total_frequency_points=deepcopy(self._total_frequency_points),
            verbose=self.verbose
        )
        new_spectrum.peaks = deepcopy(self._peaks)
        new_spectrum.remarks = deepcopy(self._remarks)
        new_spectrum.attributes = deepcopy(self._attributes)
        new_spectrum.null_string = deepcopy(self._null_string)
        new_spectrum.null_value = deepcopy(self._null_value)

        return new_spectrum