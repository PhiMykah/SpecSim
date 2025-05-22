import pytest
from specsim.peak import Peak
from specsim.datatypes import Vector, PointUnits, Phase
from typing import Any

# Test parameters
NUM_OF_DIMENSIONS = 2
# Define the position, linewidth, spectral width, origin, observation, and total frequency points for each dimension
pos: list[float] = [241.689, 66.903]
new_pos: list[float] = [217.195, 81.071]
lw: list[float] = [0.037, 0.009]
new_lw: list[float] = [0.023, 0.007]
sw: list[float] = [1920.0, 2998.046875]
origin: list[float] = [6221.201171875, 3297.501220703125]
obs: list[float] = [60.694000244140625, 598.9099731445312]
total_frequency_points: list[int] = [128, 1024]

# Collect the parameters for a position PointUnits object
pu_pos : list[list[Any]] = [[pos[i], sw[i], origin[i], obs[i], total_frequency_points[i]] for i in range(NUM_OF_DIMENSIONS)]
alt_pu_pos : list[list[Any]] = [[new_pos[i], sw[i], origin[i], obs[i], total_frequency_points[i]] for i in range(NUM_OF_DIMENSIONS)]
pu_lw : list[list[Any]] = [[lw[i], sw[i], origin[i], obs[i], total_frequency_points[i]] for i in range(NUM_OF_DIMENSIONS)]
alt_pu_lw : list[list[Any]] = [[new_lw[i], sw[i], origin[i], obs[i], total_frequency_points[i]] for i in range(NUM_OF_DIMENSIONS)]


# Create PointUnits objects for position and linewidth
position = Vector(PointUnits(*pu_pos[0]), PointUnits(*pu_pos[1]))
linewidth = Vector(PointUnits(*pu_lw[0]), PointUnits(*pu_lw[1]))

# Define intensity, phase, weights, and extra parameters
intensity = 10.0
phase = Vector(Phase(0.0), Phase(90.0))
weights = Vector(0.8, 0.2)
extra_params: dict[str, Any] = {"param1": 1.0, "param2": 2.0}

@pytest.fixture
def default_peak() -> Peak:
    return Peak(position, intensity, NUM_OF_DIMENSIONS, linewidth)

@pytest.fixture
def full_peak() -> Peak:
    return Peak(position, intensity, NUM_OF_DIMENSIONS, linewidth, phase, weights, **extra_params)

def test_peak_initialization(full_peak : Peak) -> None:
    peak : Peak = full_peak

    assert peak.position == position
    assert peak.intensity == intensity
    assert peak.linewidth == [linewidth]
    assert peak.phase == [phase]
    assert peak.weights == [weights]
    assert peak.extra_params == extra_params

def test_peak_position(default_peak : Peak) -> None:
    position = Vector(PointUnits(*pu_pos[0]), PointUnits(*pu_pos[1]))
    peak : Peak = default_peak
    new_position = Vector(PointUnits(*alt_pu_pos[0]), PointUnits(*alt_pu_pos[1]))
    peak.position = new_position

    assert peak.position == new_position

    with pytest.raises(ValueError):
        peak.position = Vector(PointUnits(*alt_pu_pos[0]))  # Mismatch in dimensions

def test_peak_intensity(default_peak : Peak) -> None:
    peak : Peak = default_peak

    peak.intensity = 20.0
    assert peak.intensity == 20.0

    with pytest.raises(TypeError):
        peak.intensity = "invalid"  # Non-numeric intensity

def test_peak_linewidth(default_peak : Peak) -> None:
    peak : Peak = default_peak

    new_linewidth = Vector(PointUnits(*alt_pu_lw[0]), PointUnits(*alt_pu_lw[1]))
    peak.linewidth = new_linewidth
    assert peak.linewidth == [new_linewidth]

    with pytest.raises(ValueError):
        peak.linewidth = Vector(PointUnits(*alt_pu_lw[0]))  # Mismatch in dimensions

def test_peak_phase(default_peak : Peak) -> None:
    peak : Peak = default_peak

    new_phase = Vector(Phase(45.0), Phase(90.0))
    peak.phase = new_phase
    assert peak.phase == [new_phase]

    with pytest.raises(ValueError):
        peak.phase = Vector(Phase(45.0))  # Mismatch in dimensions

def test_peak_weights(default_peak : Peak) -> None:
    peak : Peak = default_peak

    new_weights = Vector(0.6, 0.4)
    peak.weights = new_weights
    assert peak.weights == [new_weights]

    with pytest.raises(ValueError):
        peak.weights = Vector(0.6)  # Mismatch in dimensions

def test_peak_repr_and_str() -> None:
    peak = Peak(position, intensity, NUM_OF_DIMENSIONS, linewidth, phase, weights)

    repr_str : str = repr(peak)
    str_str : str = str(peak)

    assert repr_str == f"Peak(position={position}, intensity={intensity}, " \
           f"linewidth={[linewidth]}, phase={[phase]}, weights={[weights]}, " \
           f"extra_params={{}})"
    assert str_str == f"Peak(position={position}, intensity={intensity}, " \
           f"linewidth={[linewidth]}, phase={[phase]}, weights={[weights]})"

def test_peak_len(default_peak : Peak) -> None:
    peak : Peak = default_peak

    assert len(peak) == NUM_OF_DIMENSIONS

def test_peak_equality_operator(default_peak : Peak, full_peak : Peak) -> None:
    peak1 : Peak = full_peak
    peak2 : Peak = Peak(position, intensity, NUM_OF_DIMENSIONS, linewidth, phase, weights, **extra_params)
    peak3: Peak = default_peak
    
    assert peak1 == peak2  # Peaks with identical attributes should be equal
    assert peak1 != peak3  # Peaks with different attributes should not be equal

    # Modify an attribute and check inequality
    peak2.intensity = 15.0
    assert peak1 != peak2

if __name__ == "__main__":
    pytest.main([__file__])