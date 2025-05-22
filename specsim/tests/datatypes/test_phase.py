import pytest
from specsim_rw.datatypes.phase import Phase

def test_phase_initialization() -> None:
    phase = Phase(10.0, 20.0)
    assert phase.p0 == 10.0
    assert phase.p1 == 20.0

def test_phase_getters_and_setters() -> None:
    phase = Phase()
    phase.p0 = 15.0
    phase.p1 = 25.0
    assert phase.p0 == 15.0
    assert phase.p1 == 25.0

    # Ensure p0 and p1 must be within -180 and 180 degrees
    with pytest.raises(ValueError):
        phase.p0 = 200.0
    with pytest.raises(ValueError):
        phase.p1 = -200.0

def test_phase_repr() -> None:
    phase = Phase(10.0, 20.0)
    assert repr(phase) == "Phase(p0=10.0000, p1=20.0000)"

def test_phase_getitem() -> None:
    phase = Phase(10.0, 20.0)
    assert phase[0] == 10.0
    assert phase[1] == 20.0
    with pytest.raises(IndexError):
        _ = phase[2]

def test_phase_setitem() -> None:
    phase = Phase()
    phase[0] = 30.0
    phase[1] = 40.0
    assert phase.p0 == 30.0
    assert phase.p1 == 40.0
    with pytest.raises(ValueError):
        phase[0] = 200.0
    with pytest.raises(IndexError):
        phase[2] = 50.0

def test_phase_addition() -> None:
    phase1 = Phase(10.0, 20.0)
    phase2 = Phase(5.0, 15.0)

    result : Phase = phase1 + 5.0
    assert result.p0 == 15.0
    assert result.p1 == 25.0

    result : Phase = phase1 + phase2
    assert result.p0 == 15.0
    assert result.p1 == 35.0

def test_phase_subtraction() -> None:
    phase1 = Phase(10.0, 20.0)
    phase2 = Phase(5.0, 15.0)

    result : Phase = phase1 - 5.0
    assert result.p0 == 5.0
    assert result.p1 == 15.0

    result = phase1 - phase2
    assert result.p0 == 5.0
    assert result.p1 == 5.0

def test_phase_multiplication() -> None:
    phase1 = Phase(10.0, 20.0)
    phase2 = Phase(5.0, 15.0)

    result : Phase = phase1 * 2
    assert result.p0 == 20.0
    assert result.p1 == 40.0

    result = phase1 * phase2
    assert result.p0 == 50.0
    assert result.p1 == 300.0

def test_phase_division() -> None:
    phase1 = Phase(10.0, 20.0)
    phase2 = Phase(2.0, 4.0)

    result : Phase = phase1 / 2
    assert result.p0 == 5.0
    assert result.p1 == 10.0

    result = phase1 / phase2
    assert result.p0 == 5.0
    assert result.p1 == 5.0

    with pytest.raises(ZeroDivisionError):
        _ : Phase = phase1 / 0

    phase_zero = Phase(0.0, 0.0)
    with pytest.raises(ZeroDivisionError):
        _ : Phase = phase1 / phase_zero

def test_phase_negation() -> None:
    phase = Phase(10.0, 20.0)
    result : Phase = -phase
    assert result.p0 == -10.0
    assert result.p1 == -20.0

def test_phase_equality() -> None:
    phase1 = Phase(10.0, 20.0)
    phase2 = Phase(10.0, 20.0)
    phase3 = Phase(15.0, 25.0)

    assert phase1 == phase2  # Same values
    assert phase1 != phase3  # Different values
    assert phase1 != 10.0    # Different type
if __name__ == "__main__":
    pytest.main([__file__])