from matplotlib import use
use('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from gwpy.plotter import Plot

import os
import numpy as np


def plot_qtransform(specsgrams, plotNormalizedERange, plotTimeRanges,
                    detectorName, startTime, **kwargs):

    # Set some plotting params
    myfontsize = 15
    mylabelfontsize = 20
    myColor = 'k'
    if detectorName == 'H1':
        title = "Hanford"
    elif detectorName == 'L1':
        title = "Livingston"
    elif detectorName == 'V1':
        title = "VIRGO"
    else:
        raise ValueError('You have supplied a detector '
                         'that is unknown at this time.')

    if 1161907217 < startTime < 1164499217:
        title = title + ' - ER10'
    elif startTime > 1164499217:
        title = title + ' - O2a'
    elif 1126400000 < startTime < 1137250000:
        title = title + ' - O1'
    else:
        raise ValueError('Time outside science or engineering run '
                         'or more likely code not updated to reflect '
                         'new science run.')

    indFigAll = []

    for i, spec in enumerate(specsgrams):

        indFig = spec.plot(figsize=[8, 6])

        ax = indFig.gca()
        ax.set_position([0.125, 0.1, 0.775, 0.8])
        ax.set_yscale('log', basey=2)
        ax.set_xscale('linear')
        ax.grid(False)

        xticks = np.linspace(spec.xindex.min().value,
                             spec.xindex.max().value, 5)
        xticklabels = []
        dur = float(plotTimeRanges[i])
        [xticklabels.append(str(i)) for i in np.linspace(-dur/2, dur/2, 5)]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

        ax.set_xlabel('Time (s)', labelpad=0.1, fontsize=mylabelfontsize,
                      color=myColor)
        ax.set_ylabel('Frequency (Hz)', fontsize=mylabelfontsize,
                      color=myColor)
        ax.set_title(title, fontsize=mylabelfontsize, color=myColor)
        ax.title.set_position([.5, 1.05])
        ax.set_ylim(10, 2048)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(axis='y', style='plain')

        plt.tick_params(axis='x', which='major', labelsize=myfontsize)
        plt.tick_params(axis='y', which='major', labelsize=12)

        cbar = indFig.add_colorbar(cmap='viridis', label='Normalized energy',
                                   clim=plotNormalizedERange,
                                   pad="3%", width="5%")
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.yaxis.label.set_size(myfontsize)
        indFigAll.append(indFig)

    # Create one image containing all spectogram grams
    superFig = Plot(figsize=(27, 6))
    superFig.add_subplot(141, projection='timeseries')
    superFig.add_subplot(142, projection='timeseries')
    superFig.add_subplot(143, projection='timeseries')
    superFig.add_subplot(144, projection='timeseries')
    iN = 0

    for iAx, spec in zip(superFig.axes, specsgrams):
        iAx.plot(spec)

        iAx.set_yscale('log', basey=2)
        iAx.set_xscale('linear')

        xticks = np.linspace(spec.xindex.min().value,
                             spec.xindex.max().value, 5)
        xticklabels = []
        dur = float(plotTimeRanges[iN])
        [xticklabels.append(str(i)) for i in np.linspace(-dur/2, dur/2, 5)]
        iAx.set_xticks(xticks)
        iAx.set_xticklabels(xticklabels)

        iAx.set_xlabel('Time (s)', labelpad=0.1, fontsize=mylabelfontsize,
                       color=myColor)
        iAx.set_ylim(10, 2048)
        iAx.yaxis.set_major_formatter(ScalarFormatter())
        iAx.ticklabel_format(axis='y', style='plain')
        iN = iN + 1

        superFig.add_colorbar(ax=iAx, cmap='viridis',
                              label='Normalized energy',
                              clim=plotNormalizedERange,
                              pad="3%", width="5%")

    superFig.suptitle(title, fontsize=mylabelfontsize, color=myColor, x=0.51)

    return indFigAll, superFig
