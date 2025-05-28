from typing import Any, Callable
import pytest
import numpy as np
from specsim.optimization import optimize
from specsim.spectrum import Spectrum, Domain, sim_composite_1D, sim_exponential_1D, sim_gaussian_1D
from specsim.datatypes import Vector, Phase
from specsim.user import get_dimension_info, get_total_size
from specsim.optimization.params import OptimizationParams
import nmrPype as pype

from specsim.optimization.optimize import (
    get_optimization_method, OptMethod, unpack_params, objective_function
)

DIR = "specsim/tests/data"
file_path: Callable[..., str] = lambda folder, file : f"{DIR}/{folder}/{file}"
DEFAULT = "default"
N15HSQC = "n15_hsqc"
P3AK = "p3ak"
BASIS = "basis"
OUTPUT = "output"

@pytest.fixture
def sample_spectrum() -> Spectrum:
    file: str = file_path(DEFAULT, "test_small.tab")
    spectral_widths = Vector(1920.0, 2998.046875)
    coordinate_origins = Vector(6221.201171875, 3297.501220703125)
    observation_frequencies = Vector(60.694000244140625, 598.9099731445312)
    total_time_points = Vector(512, 64)
    total_frequency_points = Vector(1024, 128)
    return Spectrum(file, spectral_widths, coordinate_origins, observation_frequencies, total_time_points, total_frequency_points)

@pytest.fixture
def sample_full_spectrum() -> Spectrum:
    file: str = file_path(DEFAULT, "test.tab")
    spectral_widths = Vector(1920.0, 2998.046875)
    coordinate_origins = Vector(6221.201171875, 3297.501220703125)
    observation_frequencies = Vector(60.694000244140625, 598.9099731445312)
    total_time_points = Vector(512, 64)
    total_frequency_points = Vector(1024, 128)
    return Spectrum(file, spectral_widths, coordinate_origins, observation_frequencies, total_time_points, total_frequency_points)

@pytest.fixture
def sample_data() -> tuple[pype.DataFrame, pype.DataFrame, pype.DataFrame]:
    fid = pype.DataFrame(file_path(DEFAULT, "test.fid"))
    interferogram = pype.DataFrame(file_path(DEFAULT, "test.ft1"))
    spectrum = pype.DataFrame(file_path(DEFAULT, "test.ft2"))

    return fid, interferogram, spectrum

@pytest.fixture
def n15hsqc_spectrum() -> Spectrum:
    data_frame = pype.DataFrame(file_path(N15HSQC, "test.ft2"))
    file : str =  file_path(N15HSQC, "test.tab")
    spectral_widths : Vector[float] = get_dimension_info(data_frame, 'NDSW', 2)
    coordinate_origins : Vector[float] = get_dimension_info(data_frame, 'NDORIG', 2)
    observation_frequencies : Vector[float] = get_dimension_info(data_frame, 'NDOBS', 2)
    total_time_points : Vector[int] = get_total_size(data_frame, 'NDTDSIZE', 2)
    total_frequency_points : Vector[int] = get_total_size(data_frame, 'NDFTSIZE', 2)
    return Spectrum(file, spectral_widths, coordinate_origins, observation_frequencies, total_time_points, total_frequency_points)

@pytest.fixture
def n15hsqc_data() -> tuple[pype.DataFrame, pype.DataFrame, pype.DataFrame]:
    fid = pype.DataFrame(file_path(N15HSQC, "test.fid"))
    interferogram = pype.DataFrame(file_path(N15HSQC, "test.ft1"))
    spectrum = pype.DataFrame(file_path(N15HSQC, "test.ft2"))

    return fid, interferogram, spectrum

@pytest.fixture
def p3ak_spectrum() -> Spectrum:
    data_frame = pype.DataFrame(file_path(P3AK, "p3ak_freq_exp.ft2"))
    file : str =  file_path(P3AK, "test.tab")
    spectral_widths : Vector[float] = get_dimension_info(data_frame, 'NDSW', 2)
    coordinate_origins : Vector[float] = get_dimension_info(data_frame, 'NDORIG', 2)
    observation_frequencies : Vector[float] = get_dimension_info(data_frame, 'NDOBS', 2)
    total_time_points : Vector[int] = get_total_size(data_frame, 'NDTDSIZE', 2)
    total_frequency_points : Vector[int] = get_total_size(data_frame, 'NDFTSIZE', 2)
    return Spectrum(file, spectral_widths, coordinate_origins, observation_frequencies, total_time_points, total_frequency_points)

@pytest.fixture
def p3ak_data() -> tuple[pype.DataFrame, pype.DataFrame, pype.DataFrame, pype.DataFrame, pype.DataFrame, pype.DataFrame]:
    fid = pype.DataFrame(file_path(P3AK, "p3ak_freq_exp.fid"))
    interferogram = pype.DataFrame(file_path(P3AK, "p3ak_freq_exp.ft1"))
    spectrum = pype.DataFrame(file_path(P3AK, "p3ak_freq_exp.ft2"))
    fid_gauss = pype.DataFrame(file_path(P3AK, "p3ak_freq_gauss.fid"))
    interferogram_gauss = pype.DataFrame(file_path(P3AK, "p3ak_freq_gauss.ft1"))
    spectrum_gauss = pype.DataFrame(file_path(P3AK, "p3ak_freq_gauss.ft2"))
    return fid, interferogram, spectrum, fid_gauss, interferogram_gauss, spectrum_gauss

@pytest.fixture
def p3ak_composite_spectrum() -> Spectrum:
    data_frame = pype.DataFrame(file_path(P3AK, "p3ak_freq_comp.ft2"))
    file : str =  file_path(P3AK, "test.tab")
    spectral_widths : Vector[float] = get_dimension_info(data_frame, 'NDSW', 2)
    coordinate_origins : Vector[float] = get_dimension_info(data_frame, 'NDORIG', 2)
    observation_frequencies : Vector[float] = get_dimension_info(data_frame, 'NDOBS', 2)
    total_time_points : Vector[int] = get_total_size(data_frame, 'NDTDSIZE', 2)
    total_frequency_points : Vector[int] = get_total_size(data_frame, 'NDFTSIZE', 2)
    return Spectrum(file, spectral_widths, coordinate_origins, observation_frequencies, total_time_points, total_frequency_points)

@pytest.fixture
def p3ak_composite_data() -> tuple[pype.DataFrame, pype.DataFrame, pype.DataFrame]:
    fid = pype.DataFrame(file_path(P3AK, "p3ak_freq_comp.fid"))
    interferogram = pype.DataFrame(file_path(P3AK, "p3ak_freq_comp.ft1"))
    spectrum = pype.DataFrame(file_path(P3AK, "p3ak_freq_comp.ft2"))
    return fid, interferogram, spectrum

def test_optimization_method() -> None:
    assert OptMethod.LSQ.value[0] == 0
    assert OptMethod.BASIN.value[0] == 1
    assert OptMethod.MIN.value[0] == 2
    assert OptMethod.BRUTE.value[0] == 3

    assert OptMethod.LSQ.value[1] == 'lsq'
    assert OptMethod.BASIN.value[1] == 'basin'
    assert OptMethod.MIN.value[1] == 'minimize'
    assert OptMethod.BRUTE.value[1] == 'brute'

def test_get_optimization_method() -> None:
    assert get_optimization_method('lsq') == OptMethod.LSQ
    assert get_optimization_method('basin') == OptMethod.BASIN
    assert get_optimization_method('minimize') == OptMethod.MIN
    assert get_optimization_method('min') == OptMethod.MIN
    assert get_optimization_method('brute') == OptMethod.BRUTE
    assert get_optimization_method('unknown') == OptMethod.LSQ

def test_unpack_params() -> None:
    peak_count = 2
    num_of_dimensions = 1
    params : list[float] = [1, 2, 0.1, 0.2, 0.5, 0.8, 10, 20]
    decay, phase, height , weight = unpack_params(peak_count, num_of_dimensions, params)
    assert decay == [1,2]
    assert phase == [0.1,0.2, 0.5, 0.8]
    assert height == [10,20]
    assert weight == []

def test_objective_function_basic(sample_spectrum, sample_data) -> None:
    fid, interferogram, data_frame_spectrum = sample_data
    num_of_peaks = 11
    num_of_dimensions = 2
    params: list[int | float] = list(range(0, num_of_peaks)) * num_of_dimensions \
           + ([5.0] * num_of_peaks \
           +  [10.0] * num_of_peaks) * num_of_dimensions \
           + list(range(10, 10+(10*num_of_peaks), 10))

    model_function : Callable[..., np.ndarray[Any, np.dtype[Any]]] = sim_exponential_1D
    result : float = objective_function(params, num_of_dimensions, Domain.FT1, sample_spectrum,
                                model_function, fid, interferogram, data_frame_spectrum, None,
                                None, None, None)

    assert isinstance(result, (float, np.floating, np.ndarray))

def test_optmization(sample_spectrum, sample_data) -> None:
    fid, interferogram, data_frame_spectrum = sample_data
    sample_spectrum.verbose = True
    model_function : Callable[..., np.ndarray[Any, np.dtype[Any]]] = sim_exponential_1D
    num_of_dimensions = 2
    opt_params = OptimizationParams(num_of_dimensions)
    offsets = Vector(165.0, 0)
    scaling_factors = Vector(3.0518e-05, 1)
    opt_params.initial_phase = Vector(Phase(-13, 0), Phase(0, 0))
    optimized_spectrum : Spectrum = optimize(sample_spectrum, model_function, fid,
                                  interferogram, data_frame_spectrum,
                                  Domain.FT1, 'lsq', opt_params, offsets=offsets, 
                                  scaling_factors=scaling_factors)
    
    simulation : np.ndarray[Any, np.dtype[Any]] = optimized_spectrum.simulate(model_function, data_frame_spectrum,
                                             interferogram, fid, None, num_of_dimensions,
                                             None, 1, None, None, offsets, scaling_factors)
    output = pype.DataFrame(file_path(DEFAULT, "test.ft1"))

    output.setArray(simulation)

    assert not pype.write_to_file(output, file_path(OUTPUT, "sim_optimization.ft1"), True)

def test_composite_optmization(sample_spectrum, sample_data) -> None:
    fid, interferogram, data_frame_spectrum = sample_data
    sample_spectrum.verbose = True
    model_function : Callable[..., np.ndarray[Any, np.dtype[Any]]] = sim_composite_1D
    num_of_dimensions = 2
    opt_params = OptimizationParams(num_of_dimensions)
    offsets = Vector(165.0, 0)
    scaling_factor = Vector(3.0518e-05, 1)
    opt_params.initial_phase = Vector(Phase(-13, 0), Phase(0, 0))
    optimized_spectrum : Spectrum = optimize(sample_spectrum, model_function, fid,
                                  interferogram, data_frame_spectrum,
                                  Domain.FT1, 'lsq', opt_params, parameter_count=2, offsets=offsets, 
                                  scaling_factors=scaling_factor)
    
    simulation : np.ndarray[Any, np.dtype[Any]] = optimized_spectrum.simulate(model_function, data_frame_spectrum,
                                             interferogram, fid, None, num_of_dimensions,
                                             None, 1, None, None, offsets, scaling_factor)
    output = pype.DataFrame(file_path(DEFAULT, "test.ft1"))

    output.setArray(simulation)

    assert not pype.write_to_file(output, file_path(OUTPUT, "sim_composite_optimization.ft1"), True)

def test_15nhsqc_optmization(n15hsqc_spectrum, n15hsqc_data) -> None:
    fid, interferogram, data_frame_spectrum = n15hsqc_data
    n15hsqc_spectrum.verbose = True
    model_function : Callable[..., np.ndarray[Any, np.dtype[Any]]] = sim_exponential_1D
    num_of_dimensions = 2
    opt_params = OptimizationParams(num_of_dimensions)
    offsets = Vector(0.0, 0.0)
    scaling_factor = Vector(1.0, 1.0)
    opt_params.initial_phase = Vector(Phase(-39, 0), Phase(-90, 0))
    optimized_spectrum : Spectrum = optimize(n15hsqc_spectrum, model_function, fid,
                                  interferogram, data_frame_spectrum,
                                  Domain.FT1, 'lsq', opt_params, offsets=offsets, 
                                  scaling_factors=scaling_factor)
    
    simulation : np.ndarray[Any, np.dtype[Any]] = optimized_spectrum.simulate(model_function, data_frame_spectrum,
                                             interferogram, fid, None, num_of_dimensions,
                                             None, 1, None, None, offsets, scaling_factor)
    output = pype.DataFrame(file_path(N15HSQC, "test.ft1"))

    output.setArray(simulation)

    assert not pype.write_to_file(output, file_path(OUTPUT, "sim_n15hsqc_optimization.ft1"), True)

def test_p3ak_optimization(p3ak_spectrum, p3ak_data) -> None:
    fid_exp, interferogram_exp, df_spectrum_exp, fid_gauss, interferogram_gauss, df_spectrum_gauss = p3ak_data
    p3ak_spectrum.verbose = True
    exp : Callable[..., np.ndarray[Any, np.dtype[Any]]] = sim_exponential_1D
    gauss : Callable[..., np.ndarray[Any, np.dtype[Any]]] = sim_gaussian_1D
    num_of_dimensions : int = 2
    opt_params = OptimizationParams(num_of_dimensions)
    offsets = Vector(0.0, 0.0)
    scaling_factor = Vector(1.0, 1.0)

    # -------------------------------- Exponential ------------------------------- #
    exp_optimized : Spectrum = optimize(p3ak_spectrum, exp, fid_exp,
                                        interferogram_exp, df_spectrum_exp,
                                        Domain.FT1, 'lsq', opt_params, offsets=offsets, 
                                        scaling_factors=scaling_factor)
    
    exp_simulation : np.ndarray[Any, np.dtype[Any]] = exp_optimized.simulate(exp, df_spectrum_exp,
                                             interferogram_exp, fid_exp, None, num_of_dimensions,
                                             None, 1, None, None, offsets, scaling_factor)
    output = pype.DataFrame(file_path(P3AK, "p3ak_freq_exp.ft1"))

    output.setArray(exp_simulation)

    assert not pype.write_to_file(output, file_path(OUTPUT, "sim_p3ak_exp_optimization.ft1"), True)

    # --------------------------------- Gaussian --------------------------------- #
    gauss_optimized : Spectrum = optimize(p3ak_spectrum, gauss, fid_gauss,
                                        interferogram_gauss, df_spectrum_gauss,
                                        Domain.FT1, 'lsq', opt_params, offsets=offsets, 
                                        scaling_factors=scaling_factor)
    
    gauss_simulation : np.ndarray[Any, np.dtype[Any]] = gauss_optimized.simulate(gauss, df_spectrum_gauss,
                                             interferogram_gauss, fid_gauss, None, num_of_dimensions,
                                             None, 1, None, None, offsets, scaling_factor)
    output = pype.DataFrame(file_path(P3AK, "p3ak_freq_gauss.ft1"))

    output.setArray(gauss_simulation)

    assert not pype.write_to_file(output, file_path(OUTPUT, "sim_p3ak_gauss_optimization.ft1"), True)

def test_p3ak_composite_optimization(p3ak_composite_spectrum, p3ak_composite_data) -> None:
    fid, interferogram, df_spectrum = p3ak_composite_data
    p3ak_composite_spectrum.verbose = True
    model_function : Callable[..., np.ndarray[Any, np.dtype[Any]]] = sim_composite_1D
    num_of_dimensions : int = 2
    opt_params = OptimizationParams(num_of_dimensions, initial_weight=0.25)
    offsets = Vector(0.0, 0.0)
    scaling_factor = Vector(1.0, 1.0)

    optimized_spectrum : Spectrum = optimize(p3ak_composite_spectrum, model_function, fid,
                                        interferogram, df_spectrum,
                                        Domain.FT1, 'lsq', opt_params, offsets=offsets, 
                                        scaling_factors=scaling_factor)
    
    simulation : np.ndarray[Any, np.dtype[Any]] = optimized_spectrum.simulate(model_function, df_spectrum,
                                             interferogram, fid, None, num_of_dimensions,
                                             None, 1, None, None, offsets, scaling_factor)
    output = pype.DataFrame(file_path(P3AK, "p3ak_freq_comp.ft1"))

    output.setArray(simulation)

    assert not pype.write_to_file(output, file_path(OUTPUT, "sim_p3ak_comp_optimization.ft1"), True)