.. _events:

.. currentmodule:: gravityspy.table

#############################
Classifying A Table of Events
#############################

============
Introduction
============

Gravity Spy is designed to generally classify a while *table* of excess noise events at once.
These `events` are then generally, although not always, uploaded to the Zooniverse website.
In addition, these events are assumed to have originated from the Event Trigger Generator
*Omicron*. With this in mind, the :mod:`Events` class provides a host of methods
that server as wrappers on tables of events to help with a varity of tasks described below.

=================
A Table of Events
=================

doing stuff

.. ipython::

    In [1]: from gwpy.timeseries import TimeSeries

    In [2]: from gravityspy.table import Events

    In [3]: timeseries = TimeSeries.read('data/timeseries/L-L1_SOFTWAREINJ-1173197648-132.h5')

    In [4]: triggers = Events.read('data/omicron/L1-DCH_FAKE_STRAIN_16k_BBH-SEOBNRv3_OMICRON-1173188120-14996.xml.gz', format='ligolw')

    In [4]: results = triggers.classify(path_to_cnn='../models/multi_view_classifier.h5',
       ...:                             timeseries=timeseries, nproc=3)

    In [5]: print(results)
