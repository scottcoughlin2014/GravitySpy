#!/usr/bin/env python
  
# ---- Import standard modules to the python path.

from collections import OrderedDict                                                                                    
import numpy
from gwpy.timeseries import TimeSeries
from gwpy.plotter import TimeSeriesPlot

class OrderedConfusionMatrices(OrderedDict):
    def plot_line_chart(self, label, users, userID=None):
        plot = TimeSeriesPlot()
        for user, classifications in self.items():
            if user in users:
                matrices = []
                for cid in sorted(classifications.keys()):
                    matrices.append(classifications[cid])
                    timeseries_confusion_matrix = TimeSeries(numpy.vstack(matrices)[:, label])
                    plot.add_timeseries(timeseries_confusion_matrix)
            else:
                continue
        return plot
