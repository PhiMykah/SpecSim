import pytest
from specsim.optimization.params import OptimizationParams
from specsim.datatypes import Vector, Phase

def test_init_with_defaults() -> None:
    params = OptimizationParams(num_of_dimensions=2)
    assert params.trials == 100
    assert params.step_size == 0.1
    assert isinstance(params.initial_decay, list)
    assert all(isinstance(item, Vector) for item in params.initial_decay)
    assert isinstance(params.initial_phase, list)
    assert all(isinstance(item, Vector) for item in params.initial_phase)
    assert isinstance(params.bounds, list)
    assert all(isinstance(item, Vector) for item in params.bounds)
    assert params.amplitude_bounds == (0.0, 10.0)
    assert params.p0_bounds == (-180.0, 180.0)
    assert params.p1_bounds == (-180.0, 180.0)

def test_invalid_num_of_dimensions_type() -> None:
    with pytest.raises(TypeError):
        OptimizationParams(num_of_dimensions='2') # type: ignore

def test_invalid_num_of_dimensions_value() -> None:
    with pytest.raises(ValueError):
        OptimizationParams(num_of_dimensions=0)
    with pytest.raises(ValueError):
        OptimizationParams(num_of_dimensions=5)

def test_trials_setter_and_type() -> None:
    params = OptimizationParams(1)
    params.trials = 10
    assert params.trials == 10
    params.trials = 5.0
    assert params.trials == 5
    with pytest.raises(ValueError):
        params.trials = "abc"

def test_step_size_setter() -> None:
    params = OptimizationParams(1)
    params.step_size = 0.5
    assert params.step_size == 0.5
    with pytest.raises(ValueError):
        params.step_size = "abc"

def test_initial_decay_custom_and_type() -> None:
    v = Vector([2.0, 1.0])
    params = OptimizationParams(2, initial_decay=v)
    assert params.initial_decay == [v]
    with pytest.raises(TypeError):
        OptimizationParams(2, initial_decay="not_a_vector") # type: ignore
    with pytest.raises(ValueError):
        OptimizationParams(2, initial_decay=Vector([2.0]))

def test_initial_phase_custom_and_type() -> None:
    v = Vector([Phase(0.0, 0.0), Phase(1.0, 1.0)])
    params = OptimizationParams(2, initial_phase=v)
    assert params.initial_phase == [v]
    with pytest.raises(TypeError):
        OptimizationParams(2, initial_phase="not_a_vector") # type: ignore
    with pytest.raises(ValueError):
        OptimizationParams(2, initial_phase=Vector([Phase(0.0, 0.0)]))

def test_bounds_custom_and_type() -> None:
    bounds_pair: list[tuple]  = [(0.0, 10.0), (0.0, 20.0)]
    v = Vector(bounds_pair)
    params = OptimizationParams(2, bounds=v)
    assert params.bounds == [v]
    with pytest.raises(TypeError):
        OptimizationParams(2, bounds="not_a_vector") # type: ignore
    with pytest.raises(ValueError):
        OptimizationParams(2, bounds=Vector([(0.0, 10.0)])) # type: ignore
    with pytest.raises(ValueError):
        OptimizationParams(2, bounds=Vector([(10.0, 0.0), (0.0, 20.0)])) # type: ignore
    with pytest.raises(ValueError):
        OptimizationParams(2, bounds=Vector([(0, 10), (0.0, 20.0)]))  # type: ignore

def test_amplitude_bounds_custom_and_type() -> None:
    params = OptimizationParams(1, amplitude_bounds=(1.0, 5.0))
    assert params.amplitude_bounds == (1.0, 5.0)
    with pytest.raises(TypeError):
        OptimizationParams(1, amplitude_bounds="not_a_tuple") # type: ignore
    with pytest.raises(TypeError):
        OptimizationParams(1, amplitude_bounds=(1, 5))
    with pytest.raises(ValueError):
        OptimizationParams(1, amplitude_bounds=(5.0, 1.0))

def test_p0_bounds_custom_and_type() -> None:
    params = OptimizationParams(1, p0_bounds=(-90.0, 90.0))
    assert params.p0_bounds == (-90.0, 90.0)
    with pytest.raises(TypeError):
        OptimizationParams(1, p0_bounds="not_a_tuple") # type: ignore
    with pytest.raises(TypeError):
        OptimizationParams(1, p0_bounds=(-90, 90))
    with pytest.raises(ValueError):
        OptimizationParams(1, p0_bounds=(90.0, -90.0))

def test_p1_bounds_custom_and_type() -> None:
    params = OptimizationParams(1, p1_bounds=(-90.0, 90.0))
    assert params.p1_bounds == (-90.0, 90.0)
    with pytest.raises(TypeError):
        OptimizationParams(1, p1_bounds="not_a_tuple") # type: ignore
    with pytest.raises(TypeError):
        OptimizationParams(1, p1_bounds=(-90, 90))
    with pytest.raises(ValueError):
        OptimizationParams(1, p1_bounds=(90.0, -90.0))