import pytest
from specsim_rw.datatypes.pointunits import PointUnits

spectral_width = 1920.0
origin = 6221.201171875
observation_freq = 60.694000244140625
total_frequency_points = 128
value = 66.903

new_spectral_width = 2998.046875
new_origin = 3297.501220703125
new_observation_freq = 598.9099731445312
new_total_frequency_points = 1024
new_value = 241.689

def test_pointunits_getters() -> None:
    pu = PointUnits(value, spectral_width, origin, observation_freq, total_frequency_points)

    assert pu.pts == value
    assert pu.spectral_width == spectral_width
    assert pu.origin == origin
    assert pu.observation_freq == observation_freq
    assert pu.total_points == total_frequency_points

    delta_between_pts : float = (-1 * spectral_width) / total_frequency_points
    hz : float = origin + (value - total_frequency_points) * delta_between_pts

    assert pu.hz == hz
    assert pu.ppm == (hz / observation_freq)


def test_pointunits_setters() -> None:

    pu = PointUnits(value, spectral_width, origin, observation_freq, total_frequency_points)

    pu.pts = new_value
    pu.spectral_width = new_spectral_width
    pu.origin = new_origin
    pu.observation_freq = new_observation_freq
    pu.total_points = new_total_frequency_points

    assert pu.pts == new_value
    assert pu.spectral_width == new_spectral_width
    assert pu.origin == new_origin
    assert pu.observation_freq == new_observation_freq
    assert pu.total_points == new_total_frequency_points

    new_delta_between_pts : float = (-1 * new_spectral_width) / new_total_frequency_points
    new_hz : float = new_origin + (new_value - new_total_frequency_points) * new_delta_between_pts

    assert pu.hz== new_hz
    assert pu.ppm == (new_hz / new_observation_freq)


def test_pointunits_string_representation() -> None:
    pu = PointUnits(new_value, new_spectral_width, new_origin, new_observation_freq, new_total_frequency_points)

    new_delta_between_pts : float = (-1 * new_spectral_width) / new_total_frequency_points
    new_hz : float = new_origin + (new_value - new_total_frequency_points) * new_delta_between_pts

    assert str(pu) == f"PointUnits(pts={new_value:.4f}, " \
                      f"ppm={(new_hz / new_observation_freq):.4f}, " \
                      f"hz={new_hz:.4f}, " \
                      f"spectral_width={new_spectral_width:.4f}, " \
                      f"origin={new_origin:.4f}, " \
                      f"observation_freq={new_observation_freq:.4f}, " \
                      f"total_points={new_total_frequency_points})"


def test_pointunits_equality() -> None:
    pu = PointUnits(new_value, new_spectral_width, new_origin, new_observation_freq, new_total_frequency_points)

    pu2 = PointUnits(new_value, new_spectral_width, new_origin, new_observation_freq, new_total_frequency_points)
    assert pu == pu2

    assert pu != PointUnits(0, 0, 0, 0, 0)
    assert pu != PointUnits(new_value, new_spectral_width, new_origin, new_observation_freq, new_total_frequency_points + 1)
    assert pu != PointUnits(new_value, new_spectral_width, new_origin, new_observation_freq + 1, new_total_frequency_points)
    assert pu != PointUnits(new_value, new_spectral_width + 1, new_origin, new_observation_freq, new_total_frequency_points)
    assert pu != PointUnits(new_value, new_spectral_width, new_origin + 1, new_observation_freq, new_total_frequency_points)
    assert pu != PointUnits(new_value + 1, new_spectral_width, new_origin, new_observation_freq, new_total_frequency_points)


def test_pointunits_arithmetic_operations() -> None:
    pu = PointUnits(value, spectral_width, origin, observation_freq, total_frequency_points)
    pu2 = PointUnits(new_value, new_spectral_width, new_origin, new_observation_freq, new_total_frequency_points)

    pu3 : PointUnits = pu + 1.0
    assert pu3.pts == value + 1.0

    pu4 : PointUnits = pu - 1.0
    assert pu4.pts == value - 1.0

    pu5 : PointUnits = pu * 2.0
    assert pu5.pts == value * 2.0

    pu6 : PointUnits = pu / 2.0
    assert pu6.pts == value / 2.0

    with pytest.raises(ZeroDivisionError):
        _ : PointUnits = pu / 0.0

    with pytest.raises(ZeroDivisionError):
        _ : PointUnits = pu / PointUnits(0, 0, 0, 0, 0)

    pu7 : PointUnits = pu2 + pu
    assert pu7.pts == value + new_value
    assert pu7.spectral_width == new_spectral_width
    assert pu7.origin == new_origin
    assert pu7.observation_freq == new_observation_freq
    assert pu7.total_points == new_total_frequency_points

    pu8 : PointUnits = pu2 - pu
    assert pu8.pts == new_value - value
    assert pu8.spectral_width == new_spectral_width
    assert pu8.origin == new_origin
    assert pu8.observation_freq == new_observation_freq
    assert pu8.total_points == new_total_frequency_points

    pu9 : PointUnits = pu2 * pu
    assert pu9.pts == new_value * value
    assert pu9.spectral_width == new_spectral_width
    assert pu9.origin == new_origin
    assert pu9.observation_freq == new_observation_freq
    assert pu9.total_points == new_total_frequency_points

    pu10 : PointUnits = pu2 / pu
    assert pu10.pts == new_value / value
    assert pu10.spectral_width == new_spectral_width
    assert pu10.origin == new_origin
    assert pu10.observation_freq == new_observation_freq
    assert pu10.total_points == new_total_frequency_points

if __name__ == "__main__":
    pytest.main([__file__])