import pytest
from specsim_rw.spectrum import Spectrum, sim_exponential_1D, sim_gaussian_1D, sim_composite_1D
from specsim_rw.datatypes import Vector, Phase
from specsim_rw.user import get_dimension_info, get_total_size
from typing import Any, Callable
import nmrPype as pype
import numpy as np

DIR = "specsim_rw/tests/data"
file_path : Callable[..., str] = lambda folder, file : f"{DIR}/{folder}/{file}"
DEFAULT = "default"
SINGLE = "single"
DOUBLE = "double"
ZERO = "zero"
OUTPUT = "output"
BASIS = "basis"

def compare_difference(first : np.ndarray, second : np.ndarray, tolerance : float = 1e-6) -> bool:
    scale_factor : float = np.max(first) / np.max(second)
    second_scaled : np.ndarray[Any, np.dtype[Any]] = second * scale_factor
    return np.allclose(first, second_scaled, atol=tolerance)

def test_spectrum_small_simulation() -> None:
    file: str = file_path(DEFAULT, "test_small.tab")
    spectrum_data_frame = pype.DataFrame(file_path(DEFAULT, "test.ft2"))
    spectrum_interferogram = pype.DataFrame(file_path(DEFAULT, "test.ft1"))
    spectrum_fid = pype.DataFrame(file_path(DEFAULT, "test.fid"))

    spectral_widths : Vector[float] = get_dimension_info(spectrum_data_frame, 'NDSW', 2)
    coordinate_origins : Vector[float] = get_dimension_info(spectrum_data_frame, 'NDORIG', 2)
    observation_frequencies : Vector[float] = get_dimension_info(spectrum_data_frame, 'NDOBS', 2)
    total_time_points : Vector[int] = get_total_size(spectrum_data_frame, 'NDTDSIZE', 2)
    total_frequency_points : Vector[int] = get_total_size(spectrum_data_frame, 'NDFTSIZE', 2)
    sample_spectrum = Spectrum(file, spectral_widths, coordinate_origins, observation_frequencies, total_time_points, total_frequency_points)

    model_function : Callable[..., np.ndarray[Any, np.dtype[Any]]] = sim_gaussian_1D
    basis_set_folder : str = file_path(BASIS, "basis_small")
    dimensions : int = 2
    simulation : np.ndarray[Any, np.dtype[Any]] = sample_spectrum.simulate(model_function, spectrum_data_frame,
                             spectrum_interferogram, spectrum_fid,
                             basis_set_folder, dimensions, None, 1, None,
                             Vector(Phase(-13, 0), Phase(0, 0)),
                             Vector(165.0, 0), Vector(3.0518e-05, 1))
    
    # Compare original data to simulation, ensure difference is within standard error
    # assert compare_difference(spectrum_interferogram.array, simulation, 1e-5)

    # Output simulation 
    output = pype.DataFrame(file_path(DEFAULT, "test.ft1"))

    output.setArray(simulation)

    assert not pype.write_to_file(output, file_path(OUTPUT, "sim_ssr_small.ft1"), True)

def test_spectrum_full_simulation() -> None:
    file: str = file_path(DEFAULT, "test.tab")
    spectrum_data_frame = pype.DataFrame(file_path(DEFAULT, "test.ft2"))
    spectrum_interferogram = pype.DataFrame(file_path(DEFAULT, "test.ft1"))
    spectrum_fid = pype.DataFrame(file_path(DEFAULT, "test.fid"))

    spectral_widths : Vector[float] = get_dimension_info(spectrum_data_frame, 'NDSW', 2)
    coordinate_origins : Vector[float] = get_dimension_info(spectrum_data_frame, 'NDORIG', 2)
    observation_frequencies : Vector[float] = get_dimension_info(spectrum_data_frame, 'NDOBS', 2)
    total_time_points : Vector[int] = get_total_size(spectrum_data_frame, 'NDTDSIZE', 2)
    total_frequency_points : Vector[int] = get_total_size(spectrum_data_frame, 'NDFTSIZE', 2)
    sample_spectrum = Spectrum(file, spectral_widths, coordinate_origins, observation_frequencies, total_time_points, total_frequency_points)

    model_function : Callable[..., np.ndarray[Any, np.dtype[Any]]] = sim_gaussian_1D
    basis_set_folder : str = file_path(BASIS, "basis_composite")
    dimensions : int = 2
    simulation : np.ndarray[Any, np.dtype[Any]] = sample_spectrum.simulate(model_function, spectrum_data_frame,
                             spectrum_interferogram, spectrum_fid,
                             basis_set_folder, dimensions, None, 1, None,
                             Vector(Phase(-13, 0), Phase(0, 0)),
                             Vector(165.0, 0), Vector(3.0518e-05, 1))
    
    # Compare original data to simulation, ensure difference is within standard error
    # assert compare_difference(spectrum_interferogram.array, simulation, 1e-5)

    # Output simulation 
    output = pype.DataFrame(file_path(DEFAULT, "test.ft1"))

    output.setArray(simulation)

    assert not pype.write_to_file(output, file_path(OUTPUT, "sim_ssr_full.ft1"), True)

def test_spectrum_composite_simulation() -> None:
    file : str = file_path(DEFAULT, "test.tab")
    spectrum_data_frame = pype.DataFrame(file_path(DEFAULT, "test.ft2"))
    spectrum_interferogram = pype.DataFrame(file_path(DEFAULT, "test.ft1"))
    spectrum_fid = pype.DataFrame(file_path(DEFAULT, "test.fid"))

    spectral_widths : Vector[float] = get_dimension_info(spectrum_data_frame, 'NDSW', 2)
    coordinate_origins : Vector[float] = get_dimension_info(spectrum_data_frame, 'NDORIG', 2)
    observation_frequencies : Vector[float] = get_dimension_info(spectrum_data_frame, 'NDOBS', 2)
    total_time_points : Vector[int] = get_total_size(spectrum_data_frame, 'NDTDSIZE', 2)
    total_frequency_points : Vector[int] = get_total_size(spectrum_data_frame, 'NDFTSIZE', 2)
    sample_spectrum = Spectrum(file, spectral_widths, coordinate_origins, observation_frequencies, total_time_points, total_frequency_points)

    model_function : Callable[..., np.ndarray[Any, np.dtype[Any]]] = sim_composite_1D
    basis_set_folder : str = file_path(BASIS, "basis_full")
    dimensions : int = 2
    simulation : np.ndarray[Any, np.dtype[Any]] = sample_spectrum.simulate(model_function, spectrum_data_frame,
                            spectrum_interferogram, spectrum_fid,
                            basis_set_folder, dimensions, None, 1, None,
                            Vector(Phase(-13, 0), Phase(0, 0)),
                            Vector(165.0, 0), Vector(3.0518e-05, 1))
    
    # Compare original data to simulation, ensure difference is within standard error
    # assert compare_difference(spectrum_interferogram.array, simulation, 1e-5)

    # Output simulation 
    output = pype.DataFrame(file_path(DEFAULT, "test.ft1"))

    output.setArray(simulation)

    assert not pype.write_to_file(output, file_path(OUTPUT, "sim_ssr_composite.ft1"), True)

def test_spectrum_single_simulation() -> None:
    file : str = file_path(SINGLE, "single.tab")
    spectrum_data_frame = pype.DataFrame(file_path(SINGLE, "single.ft2"))
    spectrum_interferogram = pype.DataFrame(file_path(SINGLE, "single.ft1"))
    spectrum_fid = pype.DataFrame(file_path(SINGLE, "single.fid"))

    spectral_widths : Vector[float] = get_dimension_info(spectrum_data_frame, 'NDSW', 2)
    coordinate_origins : Vector[float] = get_dimension_info(spectrum_data_frame, 'NDORIG', 2)
    observation_frequencies : Vector[float] = get_dimension_info(spectrum_data_frame, 'NDOBS', 2)
    total_time_points : Vector[int] = get_total_size(spectrum_data_frame, 'NDTDSIZE', 2)
    total_frequency_points : Vector[int] = get_total_size(spectrum_data_frame, 'NDFTSIZE', 2)
    sample_spectrum = Spectrum(file, spectral_widths, coordinate_origins, observation_frequencies, total_time_points, total_frequency_points)

    model_function : Callable[..., np.ndarray[Any, np.dtype[Any]]] = sim_exponential_1D

    basis_set_folder : str = file_path(BASIS, "basis_single")
    dimensions : int = 2
    simulation : np.ndarray[Any, np.dtype[Any]] = sample_spectrum.simulate(model_function, spectrum_data_frame,
                             spectrum_interferogram, spectrum_fid,
                             basis_set_folder, dimensions, None, 1, None)
    
    # Compare original data to simulation, ensure difference is within standard error
    if spectrum_interferogram.array is not None:
        assert compare_difference(spectrum_interferogram.array, simulation, 1e-5)

    # Output simulation 
    output = pype.DataFrame(file_path(SINGLE, "single.ft1"))

    output.setArray(simulation)

    assert not pype.write_to_file(output, file_path(OUTPUT, "sim_ssr_single.ft1"), True)

def test_spectrum_double_simulation() -> None:
    file : str = file_path(DOUBLE, "double.tab")
    spectrum_data_frame = pype.DataFrame(file_path(DOUBLE, "double.ft2"))
    spectrum_interferogram = pype.DataFrame(file_path(DOUBLE, "double.ft1"))
    spectrum_fid = pype.DataFrame(file_path(DOUBLE, "double.fid"))

    spectral_widths: Vector[float] = get_dimension_info(spectrum_data_frame, 'NDSW', 2)
    coordinate_origins: Vector[float] = get_dimension_info(spectrum_data_frame, 'NDORIG', 2)
    observation_frequencies: Vector[float] = get_dimension_info(spectrum_data_frame, 'NDOBS', 2)
    total_time_points: Vector[int] = get_total_size(spectrum_data_frame, 'NDTDSIZE', 2)
    total_frequency_points: Vector[int] = get_total_size(spectrum_data_frame, 'NDFTSIZE', 2)
    sample_spectrum = Spectrum(file, spectral_widths, coordinate_origins, observation_frequencies, total_time_points, total_frequency_points)

    model_function : Callable[..., np.ndarray[Any, np.dtype[Any]]] = sim_exponential_1D

    basis_set_folder : str = file_path(BASIS, "basis_double")
    dimensions : int = 2
    simulation : np.ndarray[Any, np.dtype[Any]] = sample_spectrum.simulate(model_function, spectrum_data_frame,
                             spectrum_interferogram, spectrum_fid,
                             basis_set_folder, dimensions, None, 1, None)
    
    # Compare original data to simulation, ensure difference is within standard error
    if spectrum_interferogram.array is not None:
        assert compare_difference(spectrum_interferogram.array, simulation, 1e-5)

    # Output simulation 
    output = pype.DataFrame(file_path(DOUBLE, "double.ft1"))

    output.setArray(simulation)

    assert not pype.write_to_file(output, file_path(OUTPUT, "sim_ssr_double.ft1"), True)

def test_spectrum_zero_simulation() -> None:
    file : str = file_path(ZERO, "zero_freq.tab")
    spectrum_data_frame = pype.DataFrame(file_path(ZERO, "zero_freq_exp.ft2"))
    spectrum_interferogram = pype.DataFrame(file_path(ZERO, "zero_freq_exp.ft1"))
    spectrum_fid = pype.DataFrame(file_path(ZERO, "zero_freq_exp.fid"))

    spectral_widths : Vector[float] = get_dimension_info(spectrum_data_frame, 'NDSW', 2)
    coordinate_origins : Vector[float] = get_dimension_info(spectrum_data_frame, 'NDORIG', 2)
    observation_frequencies : Vector[float] = get_dimension_info(spectrum_data_frame, 'NDOBS', 2)
    total_time_points : Vector[int] = get_total_size(spectrum_data_frame, 'NDTDSIZE', 2)
    total_frequency_points : Vector[int] = get_total_size(spectrum_data_frame, 'NDFTSIZE', 2)
    sample_spectrum = Spectrum(file, spectral_widths, coordinate_origins, observation_frequencies, total_time_points, total_frequency_points)

    model_function : Callable[..., np.ndarray[Any, np.dtype[Any]]] = sim_exponential_1D

    basis_set_folder : str = file_path(BASIS, "basis_zero")
    dimensions : int = 2
    simulation : np.ndarray[Any, np.dtype[Any]] = sample_spectrum.simulate(model_function, spectrum_data_frame,
                             spectrum_interferogram, spectrum_fid,
                             basis_set_folder, dimensions, None, 1, None)
    
    # Compare original data to simulation, ensure difference is within standard error
    if spectrum_interferogram.array is not None:
        assert compare_difference(spectrum_interferogram.array, simulation, 1e-4)

    # Output simulation 
    output = pype.DataFrame(file_path(ZERO, "zero_freq_exp.ft1"))

    output.setArray(simulation)

    assert not pype.write_to_file(output, file_path(OUTPUT, "sim_ssr_zero.ft1"), True)

    spectrum_data_frame = pype.DataFrame(file_path(ZERO, "zero_freq_gauss.ft2"))
    spectrum_interferogram = pype.DataFrame(file_path(ZERO, "zero_freq_gauss.ft1"))
    spectrum_fid = pype.DataFrame(file_path(ZERO, "zero_freq_gauss.fid"))

    spectral_widths = get_dimension_info(spectrum_data_frame, 'NDSW', 2)
    coordinate_origins = get_dimension_info(spectrum_data_frame, 'NDORIG', 2)
    observation_frequencies = get_dimension_info(spectrum_data_frame, 'NDOBS', 2)
    total_time_points = get_total_size(spectrum_data_frame, 'NDTDSIZE', 2)
    total_frequency_points = get_total_size(spectrum_data_frame, 'NDFTSIZE', 2)
    sample_spectrum = Spectrum(file, spectral_widths, coordinate_origins, observation_frequencies, total_time_points, total_frequency_points)

    model_function = sim_gaussian_1D

    simulation = sample_spectrum.simulate(model_function, spectrum_data_frame,
                             spectrum_interferogram, spectrum_fid, None, 
                             dimensions, None, 1, None)
    
    if spectrum_interferogram.array is not None:
        assert compare_difference(spectrum_interferogram.array, simulation, 1e-4)

