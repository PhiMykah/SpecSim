import pytest
from specsim_rw.peak import load_peak_table
from specsim_rw.datatypes.vector import Vector

sample = "specsim_rw/tests/peak/sample.tab"
full_sample = "specsim_rw/tests/peak/sample2.tab"
sw = Vector(1920.0, 2998.046875)
origin = Vector(6221.201171875, 3297.501220703125)
obs = Vector(60.694000244140625, 598.9099731445312)
total_frequency_points = Vector(128, 1024)

def test_load_peak_table_invalid_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_peak_table("non_existent_file.tab", sw, origin, obs, total_frequency_points)
        
def test_load_peak_table() -> None:
    peak_table, remarks, null_string, null_value, attributes = load_peak_table(sample, 
                                                                               sw, origin, obs, 
                                                                               total_frequency_points)
    assert isinstance(peak_table, list)
    assert len(peak_table) > 0

def test_load_peak_table_larger_dataset() -> None:
    peak_table, remarks, null_string, null_value, attributes = load_peak_table(full_sample, 
                                                                               sw, origin, obs, 
                                                                               total_frequency_points)
    assert isinstance(peak_table, list)
    assert len(peak_table) > 0