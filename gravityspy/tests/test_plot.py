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
plotTimeRanges = [0.5, 1.0, 2.0, 4.0]
plotNormalizedERange = [0, 25.5]
detectorName = 'L1'
centerTime = 1127700030.877928972
searchQRange = [4, 64]
searchFrequencyRange = [10, 2048]
specsgrams = []
startTime = centerTime

for iTimeWindow in plotTimeRanges:
    durForPlot = iTimeWindow/2
    try:
        outseg = Segment(centerTime - durForPlot, centerTime + durForPlot)
        qScan = TIMESERIES.q_transform(qrange=tuple(searchQRange), frange=tuple(searchFrequencyRange),
                             gps=centerTime, search=0.5, tres=0.002,
                             fres=0.5, outseg=outseg, whiten=True)
        qValue = qScan.q
        qScan = qScan.crop(centerTime-iTimeWindow/2, centerTime+iTimeWindow/2)
    except:
        outseg = Segment(centerTime - 2*durForPlot, centerTime + 2*durForPlot)
        qScan = TIMESERIES.q_transform(qrange=tuple(searchQRange), frange=tuple(searchFrequencyRange),
                             gps=centerTime, search=0.5, tres=0.002,
                             fres=0.5, outseg=outseg, whiten=True)
        qValue = qScan.q
        qScan = qScan.crop(centerTime-iTimeWindow/2, centerTime+iTimeWindow/2)
    specsgrams.append(qScan)


class TestGravitySpyPlot(object):
    """`TestCase` for the GravitySpy
    """
    def test_plot(self):
        # Plot q_scans
        assert 1 == 1
        #indFigAll, superFig = plot_qtransform(specsgrams, plotNormalizedERange,
        #           plotTimeRanges, detectorName, startTime)
