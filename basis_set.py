from pathlib import Path
from specsim.spectrum import Spectrum
from specsim.peak import Peak, Coordinate, Coordinate2D, Phase
from specsim import get_dimension_info, get_total_size, sim_composite_1D, extract_region
import nmrPype as pype
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_basis_set(spectrum : Spectrum, spectrum_fid : pype.DataFrame, spectrum_interferogram : pype.DataFrame, spectrum_data_frame : pype.DataFrame, domain : str = 'ft1'):
    simulations = []
    axis_count = 2
    for axis in range(axis_count):
        sim_1D_peaks = spectrum.composite_simulation1D(sim_composite_1D, 0, axis, 0, False, offset=165, scaling_factor=3.0518e-05)
        simulations.append(sim_1D_peaks)

    # Collect all the x axis peaks and y axis peaks
    spectral_data : list[list[np.ndarray]] = [*simulations]
    fid = spectrum_fid

    process_iterations = 0
    if domain == 'ft1' and axis_count >= 2:
        process_iterations = 1
    if domain == 'ft2' and axis_count >= 2:
        process_iterations = 2

    # Iterate through all dimensions in need of processing
    for i in range(process_iterations):
        spectral_data[i] = []
        # Go through all peaks in simulation
        for j in range(len(simulations[i])):
            iteration_df = pype.DataFrame(file=fid.file, header=fid.header, array=fid.array)
            iteration_df.array = simulations[i][j]
            dim = i + 1

            #  Add first point scaling and window function if necessary
            off_param = spectrum_data_frame.getParam("NDAPODQ1", dim)
            end_param = spectrum_data_frame.getParam("NDAPODQ2", dim)
            pow_param = spectrum_data_frame.getParam("NDAPODQ3", dim)
            elb_param = spectrum_data_frame.getParam("NDLB", dim)
            glb_param = spectrum_data_frame.getParam("NDGB", dim)
            goff_param = spectrum_data_frame.getParam("NDGOFF", dim)
            first_point_scale = 1 + spectrum_data_frame.getParam("NDC1", dim)

            iteration_df.runFunc("SP", {"sp_off":off_param, "sp_end":end_param, "sp_pow":pow_param, "sp_elb":elb_param,
                                        "sp_glb":glb_param, "sp_goff":goff_param, "sp_c":first_point_scale})
            # Zero fill if necessary
            iteration_df.runFunc("ZF", {"zf_count":1, 'zf_auto':True})

            # Convert to frequency domain
            iteration_df.runFunc("FT")
            
            # Perform phase correction
            iteration_df.runFunc("PS", {"ps_p0":spectrum.peaks[j].phase_exp[i][0], "ps_p1":spectrum.peaks[j].phase_exp[i][1]})

            # Delete imaginary values
            # simulations[i] = simulations[i].real
            iteration_df.runFunc("DI")
            
            # Extract designated region if necessary
            first_point = int(spectrum_data_frame.getParam("NDX1", dim))
            last_point = int(spectrum_data_frame.getParam("NDXN", dim))
            if first_point and last_point:
                iteration_df.array = extract_region(iteration_df.array, first_point, last_point)

            # Add peak to list
            spectral_data[i].append(iteration_df.array)
    
    x_axis = spectral_data[0]
    y_axis = spectral_data[1]    
    y_length = y_axis.shape[-1]

    if np.iscomplexobj(y_axis):
        interleaved_data = np.zeros(y_axis.shape[:-1] + (y_length * 2,), dtype=y_axis.real.dtype)
        for i in range(len(interleaved_data)):
            interleaved_data[i][0::2] = y_axis[i].real
            interleaved_data[i][1::2] = y_axis[i].imag

        y_axis = interleaved_data
        y_length = y_length * 2

    # Create the 151 x-axis y-axis pairings
    planes = []
    for i in range(peak_count):
        plane = np.outer(y_axis[i], x_axis[i])
        iteration_df = pype.DataFrame(file=spectrum_interferogram.file, header=spectrum_interferogram.header, array=plane)
        iteration_df.setArray(plane)
        # Ensure output directory exists
        output_dir = Path("demo/hsqc_basis")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the plane to a file
        max_digits = len(str(peak_count))
        output_file = output_dir / f"{i + 1:0{max_digits}}.{domain}"

        pype.write_to_file(iteration_df, str(output_file) , True)
        planes.append(plane)

    # Optionally, return or store the pairings for further use
    print("Done!")

# ---------------------------------------------------------------------------- #
#                               CSV File Loading                               #
# ---------------------------------------------------------------------------- #
csv_file = 'debug/composite_simulation/test_optimized_parameters.csv'
data = pd.read_csv(csv_file)

# Check the length of each column
column_lengths = data.apply(len)

# Verify if all column lengths are equivalent
if column_lengths.nunique() == 1:
    peak_count = column_lengths.iloc[0]
    print("Column lengths value:", peak_count)
else:
    raise ValueError("Column Lengths are not equivalent!")

# ---------------------------------------------------------------------------- #
#                               Dataframe Loading                              #
# ---------------------------------------------------------------------------- #

data_frame = pype.DataFrame('hsqc/test.ft2')
interferogram = pype.DataFrame('hsqc/test.ft1')
data_fid = pype.DataFrame('hsqc/test.fid')
spectral_widths = get_dimension_info(data_frame, 'NDSW')
origins = get_dimension_info(data_frame, 'NDORIG')
observation_frequencies = get_dimension_info(data_frame, "NDOBS")
total_time_points = get_total_size(data_frame, 'NDTDSIZE')
total_freq_points = get_total_size(data_frame, 'NDFTSIZE')

# ---------------------------------------------------------------------------- #
#                               Spectrum Loading                               #
# ---------------------------------------------------------------------------- #

spectrum = Spectrum(Path('hsqc/nlin.tab'),
            spectral_widths,
            origins,
            observation_frequencies,
            total_time_points, 
            total_freq_points,
            True)


# ---------------------------------------------------------------------------- #
#                            Optimization Parameters                           #
# ---------------------------------------------------------------------------- #

if (peak_count != len(spectrum.peaks)):
    raise ValueError("Peak count for csv does not match tab data!")

for i in range(peak_count):
    x_exp_linewidth = Coordinate(
        data['X_LW_exp'][i], spectral_widths[0],
        origins[0], observation_frequencies[0],
        total_freq_points[0])
    x_gauss_linewidth = Coordinate(
        data['X_LW_gauss'][i], spectral_widths[0],
        origins[0], observation_frequencies[0],
        total_freq_points[0])
    
    y_exp_linewidth = Coordinate(
        data['Y_LW_exp'][i], spectral_widths[1],
        origins[1], observation_frequencies[1],
        total_freq_points[1])
    y_gauss_linewidth = Coordinate(
        data['Y_LW_gauss'][i], spectral_widths[1],
        origins[1], observation_frequencies[1],
        total_freq_points[1])
    
    exp_linewidths = Coordinate2D(i, x_exp_linewidth, y_exp_linewidth)
    gauss_linewidths = Coordinate2D(i, x_gauss_linewidth, y_gauss_linewidth)
    
    intensity = data['Peak_H'][i]

    x_phase_exp = Phase(data['X_P0_exp'][i], data['X_P1_exp'][i])
    x_phase_gauss = Phase(data['X_P0_gauss'][i], data['X_P1_gauss'][i])

    y_phase_exp = Phase(data['Y_P0_exp'][i], data['Y_P1_exp'][i])
    y_phase_gauss = Phase(data['Y_P0_gauss'][i], data['Y_P1_gauss'][i])

    phase_exp = [x_phase_exp, y_phase_exp]
    phase_gauss = [x_phase_gauss, y_phase_gauss]

    spectrum.peaks[i].exp_linewidths = exp_linewidths
    spectrum.peaks[i].gauss_linewidths = gauss_linewidths
    spectrum.peaks[i].phase_exp = phase_exp
    spectrum.peaks[i].phase_gauss = phase_gauss

create_basis_set(spectrum, data_fid, interferogram, data_frame, 'ft1')