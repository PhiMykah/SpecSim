from ast import Call
import pytest
from pathlib import Path
from specsim_rw.spectrum import Spectrum, Domain, sim_exponential_1D, sim_gaussian_1D
from specsim_rw.datatypes import Vector, PointUnits
from specsim_rw.peak import Peak
from typing import Any, Callable
import nmrPype as pype
import numpy as np

DIR = "specsim_rw/tests/data"
file_path: Callable[..., str] = lambda folder, file : f"{DIR}/{folder}/{file}"
DEFAULT = "default"
BASIS = "basis"

@pytest.fixture
def sample_spectrum() -> Spectrum:
    file: str = file_path(DEFAULT, "test_small.tab")
    spectral_widths = Vector(1920.0, 2998.046875)
    coordinate_origins = Vector(6221.201171875, 3297.501220703125)
    observation_frequencies = Vector(60.694000244140625, 598.9099731445312)
    total_time_points = Vector(512, 64)
    total_frequency_points = Vector(1024, 128)
    return Spectrum(file, spectral_widths, coordinate_origins, observation_frequencies, total_time_points, total_frequency_points)

def test_spectrum_initialization(sample_spectrum : Spectrum) -> None:
    assert sample_spectrum.file == Path(file_path(DEFAULT, "test_small.tab"))
    assert sample_spectrum._spectral_widths == Vector(1920.0, 2998.046875)
    assert sample_spectrum._coordinate_origins == Vector(6221.201171875, 3297.501220703125)
    assert sample_spectrum._observation_frequencies == Vector(60.694000244140625, 598.9099731445312)
    assert sample_spectrum._total_frequency_points == Vector(1024, 128)
    assert sample_spectrum._total_time_points == Vector(512, 64)

def test_file_setter(sample_spectrum : Spectrum) -> None:
    new_file: str = file_path(DEFAULT, "test.tab")
    sample_spectrum.file = new_file
    assert sample_spectrum.file == Path(new_file)

def test_peaks_setter(sample_spectrum : Spectrum) -> None:
    spectral_widths = Vector(1920.0, 2998.046875)
    coordinate_origins = Vector(6221.201171875, 3297.501220703125)
    observation_frequencies = Vector(60.694000244140625, 598.9099731445312)
    total_frequency_points = Vector(1024, 128)
    pu1 = PointUnits(241.689, spectral_widths[0], coordinate_origins[0], observation_frequencies[0], total_frequency_points[0])
    pu2 = PointUnits(66.903, spectral_widths[1], coordinate_origins[1], observation_frequencies[1], total_frequency_points[1])
    pu3 = PointUnits(0.037, spectral_widths[0], coordinate_origins[0], observation_frequencies[0], total_frequency_points[0])
    pu4 = PointUnits(0.009, spectral_widths[1], coordinate_origins[1], observation_frequencies[1], total_frequency_points[1])

    peaks: list[Peak] = [Peak(Vector(pu1, pu2), 100.0, 2, Vector(pu3, pu4))]
    sample_spectrum.peaks = peaks
    assert sample_spectrum.peaks == peaks

def test_remarks_setter(sample_spectrum : Spectrum) -> None:
    remarks = "Test remarks"
    sample_spectrum.remarks = remarks
    assert sample_spectrum.remarks == remarks

def test_attributes_setter(sample_spectrum : Spectrum) -> None:
    attributes: dict[str, str] = {"key": "value"}
    sample_spectrum.attributes = attributes
    assert sample_spectrum.attributes == attributes

def test_null_string_setter(sample_spectrum : Spectrum) -> None:
    null_string = "NULL"
    sample_spectrum.null_string = null_string
    assert sample_spectrum.null_string == null_string

def test_null_value_setter(sample_spectrum : Spectrum) -> None:
    null_value: float = -1.0
    sample_spectrum.null_value = null_value
    assert sample_spectrum.null_value == null_value

def test_repr() -> None:
    file : str = file_path(DEFAULT, "test_small.tab")
    spectral_widths = Vector(1920.0, 2998.046875)
    coordinate_origins = Vector(6221.201171875, 3297.501220703125)
    observation_frequencies = Vector(60.694000244140625, 598.9099731445312)
    total_time_points = Vector(512, 64)
    total_frequency_points = Vector(1024, 128)
    sample_spectrum = Spectrum(file, spectral_widths, coordinate_origins, observation_frequencies, total_time_points, total_frequency_points)
    repr_str = repr(sample_spectrum)
    assert repr_str == f"Spectrum(file={Path(file)}, spectral_widths={spectral_widths}, " \
               f"coordinate_origins={coordinate_origins}, observation_frequencies={observation_frequencies}, " \
               f"total_frequency_points={total_frequency_points}, peaks={sample_spectrum._peaks})"

def test_str(sample_spectrum) -> None:
    file : str = file_path(DEFAULT, "test_small.tab")
    spectral_widths = Vector(1920.0, 2998.046875)
    coordinate_origins = Vector(6221.201171875, 3297.501220703125)
    observation_frequencies = Vector(60.694000244140625, 598.9099731445312)
    total_time_points = Vector(512, 64)
    total_frequency_points = Vector(1024, 128)
    sample_spectrum = Spectrum(file, spectral_widths, coordinate_origins, observation_frequencies, total_time_points, total_frequency_points)
    str_output = str(sample_spectrum)
    assert str_output == f"Spectrum File: {Path(file)}\n" \
               f"Spectral Widths: {spectral_widths}\n" \
               f"Coordinate Origins: {coordinate_origins}\n" \
               f"Observation Frequencies: {observation_frequencies}\n" \
               f"Total Frequency Points: {total_frequency_points}\n" \
               f"Peaks: {len(sample_spectrum._peaks)} peaks\n" \
               f"Remarks: {sample_spectrum._remarks}\n" \
               f"Attributes: {sample_spectrum._attributes}\n" \
               f"Null String: {sample_spectrum._null_string}\n" \
               f"Null Value: {sample_spectrum._null_value}\n" \
               f"Verbose: {sample_spectrum.verbose}\n"

def test_domain_enum() -> None:
    assert Domain.FID.to_string() == "fid"
    assert Domain.FT1.to_string() == "ft1"
    assert Domain.FT2.to_string() == "ft2"

def test_deepcopy(sample_spectrum: Spectrum) -> None:
    copied_spectrum: Spectrum = sample_spectrum.__deepcopy__({})
    assert copied_spectrum is not sample_spectrum
    assert copied_spectrum.file == sample_spectrum.file
    assert copied_spectrum._spectral_widths == sample_spectrum._spectral_widths
    assert copied_spectrum._coordinate_origins == sample_spectrum._coordinate_origins
    assert copied_spectrum._observation_frequencies == sample_spectrum._observation_frequencies
    assert copied_spectrum._total_time_points == sample_spectrum._total_time_points
    assert copied_spectrum._total_frequency_points == sample_spectrum._total_frequency_points
    assert copied_spectrum.peaks == sample_spectrum.peaks
    assert copied_spectrum.remarks == sample_spectrum.remarks
    assert copied_spectrum.attributes == sample_spectrum.attributes
    assert copied_spectrum.null_string == sample_spectrum.null_string
    assert copied_spectrum.null_value == sample_spectrum.null_value
    assert copied_spectrum.verbose == sample_spectrum.verbose

def test_spectrum_simulation(sample_spectrum : Spectrum) -> None:
    model_function : Callable[..., np.ndarray[Any, np.dtype[Any]]] = sim_exponential_1D
    spectrum_data_frame = pype.DataFrame(file_path(DEFAULT, "test.ft2"))
    spectrum_interferogram = pype.DataFrame(file_path(DEFAULT, "test.ft1"))
    spectrum_fid = pype.DataFrame(file_path(DEFAULT, "test.fid"))
    basis_set_folder : str = file_path(BASIS, "basis_exp")
    dimensions : int = 2
    simulation : np.ndarray[Any, np.dtype[Any]] = sample_spectrum.simulate(model_function, spectrum_data_frame,
                             spectrum_interferogram, spectrum_fid,
                             basis_set_folder, dimensions)
    
    assert isinstance(simulation, np.ndarray)

    model_function = sim_gaussian_1D
    basis_set_folder : str = file_path(BASIS, "basis_gauss")
    simulation = sample_spectrum.simulate(model_function, spectrum_data_frame,
                             spectrum_interferogram, spectrum_fid,
                             basis_set_folder, dimensions)
    
    assert isinstance(simulation, np.ndarray)