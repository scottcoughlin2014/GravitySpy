.. _classify:

.. currentmodule:: gravityspy.classify

###################################
Classify a Trigger With Gravity Spy
###################################

============
Introduction
============

Gravity Spy utilizes a trained convolutional nueral net (CNN)
to classify excess noise events in gravitational wave (GW) detector data.
Specifically, the CNN is trained on :py:mod:`gwpy.timeseries.TimeSeries.q_transform`
of graviatational wave data which are a specialized form of spectrograms.

Below we will show what these spectrograms look like and how to use :ref:`classify` to classify a
known instance of excess noise in GW data.

======================
The excess noise event
======================

In the following we demonstrate an example of the *Scratchy* excess noise event. From here on out,
we will refer to these excess noise events as *glitches*.

.. ipython::

    In [1]: from gwpy.timeseries import TimeSeries

    In [2]: from gravityspy.classify import classify

    In [3]: timeseries = TimeSeries.read('data/timeseries/scratchy_timeseries_test.h5')

    In [4]: event_time = 1127700030.877928972

    In [4]: results = classify(event_time=event_time, channel_name='L1:GDS-CALIB_STRAIN',
       ...:                    path_to_cnn='../models/multi_view_classifier.h5',
       ...:                    timeseries=timeseries)

    In [5]: print(results)


Breaking down `classify`
------------------------

It is best here to break down the steps in the above. The process goes as follows:

- perform the q_transform around the provided event time
- plot four different durations or *views* of the q_transform
- utilize the four *views* of the image to create a single **multiview** image to be passed to the CNN

.. ipython::

    In [2]: from gravityspy.utils import utils

    In [3]: config = utils.GravitySpyConfigFile()

    In [3]: specsgrams, q_value = utils.make_q_scans(event_time=event_time, timeseries=timeseries, config=config)

    In [4]: print(specsgrams[0], q_value)

Now we plot all 4 of the spectrograms as png

.. ipython::

    In [1]: from gravityspy.plot import plot_qtransform

    In [2]: plot_time_ranges = config.plot_time_ranges

    In [2]: plot_normalized_energy_range = config.plot_normalized_energy_range

    In [2]: detector_name = 'L1'

    In [2]: ind_fig_all, super_fig = plot_qtransform(specsgrams,
       ...:                                          plot_normalized_energy_range,
       ...:                                          plot_time_ranges,
       ...:                                          detector_name,
       ...:                                          event_time)

    @savefig plot-scratchy-all-durations.png
    In [3]: super_fig
