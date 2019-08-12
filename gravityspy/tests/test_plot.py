"""Unit test for GravitySpy
"""

__author__ = 'Scott Coughlin <scott.coughlin@ligo.org>'

import os
import matplotlib
matplotlib.use('agg')
from gravityspy.plot.plot import plot_qtransform
from gwpy.timeseries import TimeSeries
from gwpy.segments import Segment

TIMESERIES_PATH = os.path.join(os.path.split(__file__)[0], 'data',
'timeseries', 'scratchy_timeseries_test.h5')
TIMESERIES = TimeSeries.read(TIMESERIES_PATH)

specsgrams = []
plot_time_ranges = [0.5, 1.0, 2.0, 4.0]
plot_normalized_energy_range = [0, 25.5]
detector_name = 'L1'
center_time = 1127700030.877928972
search_q_range = [4, 64]
search_frequency_range = [10, 2048]
specsgrams = []
start_time = center_time

for time_window in plot_time_ranges:
    duration_for_plot = time_window/2
    try:
        outseg = Segment(center_time - duration_for_plot,
                         center_time + duration_for_plot)
        q_scan = TIMESERIES.q_transform(qrange=tuple(search_q_range),
                                        frange=tuple(search_frequency_range),
                                        gps=center_time,
                                        search=0.5, tres=0.002,
                                        fres=0.5, outseg=outseg, whiten=True)
        q_value = q_scan.q
        q_scan = q_scan.crop(center_time-time_window/2,
                             center_time+time_window/2)
    except:
        outseg = Segment(center_time - 2*duration_for_plot,
                         center_time + 2*duration_for_plot)
        q_scan = TIMESERIES.q_transform(qrange=tuple(search_q_range),
                                        frange=tuple(search_frequency_range),
                                        gps=center_time, search=0.5,
                                        tres=0.002,
                                        fres=0.5, outseg=outseg, whiten=True)
        q_value = q_scan.q
        q_scan = q_scan.crop(center_time-time_window/2,
                             center_time+time_window/2)
    specsgrams.append(q_scan)


class TestGravitySpyPlot(object):
    """`TestCase` for the GravitySpy
    """
    def test_plot(self):
        # Plot q_scans
        ind_fig_all, super_fig = plot_qtransform(specsgrams,
                                                 plot_normalized_energy_range,
                                                 plot_time_ranges,
                                                 detector_name,
                                                 start_time)
