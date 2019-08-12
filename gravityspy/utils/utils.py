# -*- coding: utf-8 -*-
# Copyright (C) Scott Coughlin (2017-)
#
# This file is part of gravityspy.
#
# gravityspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gravityspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gravityspy.  If not, see <http://www.gnu.org/licenses/>.

from . import log
from ..plot.plot import plot_qtransform
from ..ml import read_image
from ..ml import labelling_test_glitches as label_glitches

from gwpy.timeseries import TimeSeries
from gwpy.segments import Segment
from gwpy.table import GravitySpyTable

import numpy
import h5py
import os
import pandas
import matplotlib.pyplot as plt

class GravitySpyConfigFile(object):
    def __init__(self, sample_frequency=16384, block_time=64,
                 search_frequency_range=(10, 2048),
                 search_q_range=(4, 64), plot_time_ranges=[0.5, 1.0, 2.0, 4.0],
                 plot_normalized_energy_range=(0, 25.5)):

        self.sample_frequency = sample_frequency
        self.block_time = block_time
        self.search_frequency_range = search_frequency_range
        self.search_q_range = search_q_range
        self.plot_time_ranges = plot_time_ranges
        self.plot_normalized_energy_range = plot_normalized_energy_range

def make_q_scans(event_time, **kwargs):
    """Classify triggers in this table

    Parameters:
    ----------

    Returns
    -------
    """
    # Parse Keyword Arguments
    config = kwargs.pop('config', GravitySpyConfigFile())
    timeseries = kwargs.pop('timeseries', None)
    source = kwargs.pop('source', None)
    channel_name = kwargs.pop('channel_name', None)
    frametype = kwargs.pop('frametype', None)
    verbose = kwargs.pop('verbose', False)

    if verbose:
        logger = log.Logger('Gravity Spy: Making Q Scans')

    if (timeseries is None) and (channel_name is None):
        raise ValueError("If not directly passing a timeseries, then "
                         "the user must pass channel_name")

    ###########################################################################
    #                                   Parse Ini File                        #
    ###########################################################################
    sample_frequency = config.sample_frequency
    block_time = config.block_time
    search_frequency_range = config.search_frequency_range
    search_q_range = config.search_q_range
    plot_time_ranges = config.plot_time_ranges
    plot_normalized_energy_range = config.plot_normalized_energy_range

    if verbose:
        logger.info("""
                    You have chosen the following parameters
                    
                        Sample Frequency : {0}
                        Block Time : {1}
                        Frequency Range : {2}
                        Q Range : {3}
                        Spectrogram Colorbar Range: {4}

                    """.format(sample_frequency, block_time,
                               search_frequency_range, search_q_range,
                               plot_time_ranges, plot_normalized_energy_range))

    # find closest sample time to event time
    center_time = (
                   numpy.floor(event_time) +
                   numpy.round((event_time - numpy.floor(event_time)) *
                   sample_frequency) / sample_frequency
                  )

    # determine segment start and stop times
    start_time = round(center_time - block_time / 2)
    stop_time = start_time + block_time

    # Read in the data
    if timeseries:
        data = timeseries.crop(start_time, stop_time, verbose=verbose)
    elif source:
        if verbose:
            logger.info('Reading Data From Source ...')
        data = TimeSeries.read(source=source, channel=channel_name,
                               start=start_time, end=stop_time, verbose=verbose)
    else:
        if verbose:
            logger.info('Fetching Data...')
        data = TimeSeries.get(channel_name, start_time, stop_time,
                              frametype=frametype, verbose=verbose).astype('float64')

    # resample data
    if verbose:
        logger.info('Resampling Data...')
    if data.sample_rate.decompose().value != sample_frequency:
        data = data.resample(sample_frequency)

    # Cropping the results before interpolation to save on time and memory
    # perform the q-transform
    if verbose:
        logger.info('Processing Q Scans...')

    specsgrams = []
    for time_window in plot_time_ranges:
        duration_for_plot = time_window/2
        try:
            outseg = Segment(center_time - duration_for_plot,
                             center_time + duration_for_plot)
            q_scan = data.q_transform(qrange=tuple(search_q_range),
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
            q_scan = data.q_transform(qrange=tuple(search_q_range),
                                      frange=tuple(search_frequency_range),
                                      gps=center_time, search=0.5,
                                      tres=0.002,
                                      fres=0.5, outseg=outseg, whiten=True)
            q_value = q_scan.q
            q_scan = q_scan.crop(center_time-time_window/2,
                                 center_time+time_window/2)
        specsgrams.append(q_scan)

    if verbose:
        logger.info('The most significant q value is {0}'.format(q_value))

    return specsgrams, q_value

def save_q_scans(plot_directory, specsgrams,
                 plot_normalized_energy_range, plot_time_ranges,
                 detector_name, event_time, **kwargs):
    """Classify triggers in this table

    Parameters:
    ----------

    Returns
    -------
    """
    id_string = kwargs.pop('id_string', '{0:.9f}'.format(event_time))
    verbose = kwargs.pop('verbose', False)
    ###########################################################################
    #                           create output directory                       #
    ###########################################################################
    if verbose:
        logger = log.Logger('Gravity Spy: Saving Q Scan')
    # report status
    if not os.path.isdir(plot_directory):
        if verbose:
            logger.info('creating event directory')
        os.makedirs(plot_directory)

    if verbose:
        logger.info('plot_directory:  {0}'.format(plot_directory))

    # Plot q_scans
    if verbose:
        logger.info('Plotting q scans...')
    ind_fig_all, super_fig = plot_qtransform(specsgrams,
                                             plot_normalized_energy_range,
                                             plot_time_ranges,
                                             detector_name,
                                             event_time, **kwargs)

    for idx, ind_fig in enumerate(ind_fig_all):
        dur = float(plot_time_ranges[idx])
        ind_fig.save(os.path.join(
                                  plot_directory,
                                  detector_name + '_' + id_string
                                  + '_spectrogram_' + str(dur) +'.png'
                                 )
                    )

    super_fig.save(os.path.join(plot_directory, id_string + '.png'),)

    plt.close('all')

    return

def label_q_scans(plot_directory, path_to_cnn, **kwargs):
    """Classify triggers in this table

    Parameters:
    ----------

    Returns
    -------
    """
    verbose = kwargs.pop('verbose', False)
    order_of_channels = kwargs.pop('order_of_channels', 'channels_last')
    original_order = kwargs.pop('original_order', False)

    f = h5py.File(path_to_cnn, 'r')
    # load the api gravityspy project cached class
    classes = kwargs.pop('classes',
                         numpy.array(f['/labels/labels']).astype(str).T[0])

    if verbose:
        logger = log.Logger('Gravity Spy: Labelling Images')
    # Since we created the images in a
    # special temporary directory we can run os.listdir to get there full
    # names so we can convert the images into ML readable format.
    list_of_images = [ifile for ifile in os.listdir(plot_directory)
                      if 'spectrogram' in ifile]

    if verbose:
        logger.info('Converting image to ML readable...')

    image_data_for_cnn = pandas.DataFrame()
    for image in list_of_images:
        if verbose:
            logger.info('Converting {0}'.format(image))

        image_data = read_image.read_grayscale(os.path.join(plot_directory, image),
                                               resolution=0.3)
        image_data_for_cnn[image] = [image_data]

    # Now label the image
    if verbose:
        logger.info('Labelling image...')

    scores, ml_label, ids, filename1, filename2, filename3, filename4 = \
         label_glitches.label_glitches(image_data=image_data_for_cnn,
                                       model_name='{0}'.format(path_to_cnn),
                                       image_size=[140, 170],
                                       order_of_channels=order_of_channels,
                                       original_order=original_order,
                                       verbose=verbose)

    labels = numpy.array(classes)[ml_label]

    scores_table = GravitySpyTable(scores, names=classes)

    scores_table['Filename1'] = filename1
    scores_table['Filename2'] = filename2
    scores_table['Filename3'] = filename3
    scores_table['Filename4'] = filename4
    scores_table['gravityspy_id'] = ids
    scores_table['ml_label'] = labels
    scores_table['ml_confidence'] = scores.max(1)

    return scores_table

def label_select_images(filename1, filename2, filename3, filename4,
                        path_to_cnn, **kwargs):
    """Classify triggers in this table

    Parameters:
    ----------

    Returns
    -------
    """
    verbose = kwargs.pop('verbose', False)
    order_of_channels = kwargs.pop('order_of_channels', 'channels_last')
    original_order = kwargs.pop('original_order', False)

    # determine class names
    f = h5py.File(path_to_cnn, 'r')
    classes = kwargs.pop('classes',
                         numpy.array(f['/labels/labels']).astype(str).T[0])

    if verbose:
        logger = log.Logger('Gravity Spy: Labelling Select Images')

    list_of_images_all = [filename1,
                          filename2,
                          filename3,
                          filename4]

    list_of_images_all = zip(list_of_images_all[0], list_of_images_all[1],
                             list_of_images_all[2], list_of_images_all[3])

    if verbose:
        logger.info('Converting image to ML readable...')

    image_data_for_cnn = pandas.DataFrame()

    for list_of_images in list_of_images_all:
        for image in list_of_images:
            image_name = image.split('/')[-1]
            if verbose:
                logger.info('Converting {0}'.format(image_name))

            image_data = read_image.read_grayscale(image,
                                                   resolution=0.3)
            image_data_for_cnn[image_name] = [image_data]

    # Now label the image
    if verbose:
        logger.info('Labelling images...')

    scores, ml_label, ids, _, _, _, _ = \
         label_glitches.label_glitches(image_data=image_data_for_cnn,
                                       model_name='{0}'.format(path_to_cnn),
                                       image_size=[140, 170],
                                       order_of_channels=order_of_channels,
                                       original_order=original_order,
                                       verbose=verbose)

    labels = numpy.array(classes)[ml_label]

    scores_table = GravitySpyTable(scores, names=classes)

    scores_table['gravityspy_id'] = ids
    scores_table['ml_label'] = labels
    scores_table['ml_confidence'] = scores.max(1)

    return scores_table

def get_features_select_images(filename1, filename2, filename3, filename4,
                               path_to_semantic_model, **kwargs):
    """Classify triggers in this table

    Parameters:
    ----------

    Returns
    -------
    """
    verbose = kwargs.pop('verbose', False)
    order_of_channels = kwargs.pop('order_of_channels', 'channels_last')

    # determine class names
    if verbose:
        logger = log.Logger('Gravity Spy: Extracting features select images')

    list_of_images_all = [filename1,
                          filename2,
                          filename3,
                          filename4]

    list_of_images_all = zip(list_of_images_all[0], list_of_images_all[1],
                             list_of_images_all[2], list_of_images_all[3])

    image_data_for_cnn = pandas.DataFrame()

    for list_of_images in list_of_images_all:
        for image in list_of_images:
            image_name = image.split('/')[-1]
            if verbose:
                logger.info('Converting {0}'.format(image_name))

            image_data_r, image_data_g, image_data_b = read_image.read_rgb(image,
                                                                           resolution=0.3)
            image_data_for_cnn[image] = [[image_data_r, image_data_g, image_data_b]]

    features, ids = label_glitches.get_multiview_feature_space(image_data=image_data_for_cnn,
                                       semantic_model_name='{0}'.format(path_to_semantic_model),
                                       image_size=[140, 170],
                                       verbose=verbose,
                                       order_of_channels=order_of_channels)

    scores_table = GravitySpyTable(features, names=numpy.arange(0, features.shape[1]).astype(str))

    scores_table['gravityspy_id'] = ids

    return scores_table

def get_features(plot_directory, path_to_semantic_model, **kwargs):
    """Classify triggers in this table

    Parameters:
    ----------

    Returns
    -------
    """
    verbose = kwargs.pop('verbose', False)
    order_of_channels = kwargs.pop('order_of_channels', 'channels_last')

    if verbose:
        logger = log.Logger('Gravity Spy: Extracting Feature Space')
    # Since we created the images in a
    # special temporary directory we can run os.listdir to get there full
    # names so we can convert the images into ML readable format.
    list_of_images = [ifile for ifile in os.listdir(plot_directory)
                      if 'spectrogram' in ifile]

    if verbose:
        logger.info('Converting image to RGB readable...')

    image_data_for_si = pandas.DataFrame()
    for image in list_of_images:
        if verbose:
            logger.info('Converting {0}'.format(image))

        image_data_r, image_data_g, image_data_b = read_image.read_rgb(os.path.join(plot_directory, image),
                                                                       resolution=0.3)
        image_data_for_si[image] = [[image_data_r, image_data_g, image_data_b]]

    # Now label the image
    if verbose:
        logger.info('Extracting Features of Image...')

    features, ids = label_glitches.get_multiview_feature_space(image_data=image_data_for_si,
                                       semantic_model_name='{0}'.format(path_to_semantic_model),
                                       image_size=[140, 170],
                                       verbose=verbose,
                                       order_of_channels=order_of_channels)

    scores_table = GravitySpyTable(features, names=numpy.arange(0, features.shape[1]).astype(str))

    scores_table['gravityspy_id'] = ids

    return scores_table

def get_deeplayer(plot_directory, path_to_cnn, **kwargs):
    """Classify triggers in this table

    Parameters:
    ----------

    Returns
    -------
    """
    verbose = kwargs.pop('verbose', False)
    f = h5py.File(path_to_cnn, 'r')
    # load the api gravityspy project cached class
    classes = kwargs.pop('classes',
                         numpy.array(f['/labels/labels']).astype(str).T[0])

    if verbose:
        logger = log.Logger('Gravity Spy: Labelling Images')
    # Since we created the images in a
    # special temporary directory we can run os.listdir to get there full
    # names so we can convert the images into ML readable format.
    list_of_images = [ifile for ifile in os.listdir(plot_directory)
                      if 'spectrogram' in ifile]

    if verbose:
        logger.info('Converting image to ML readable...')

    image_data_for_cnn = pandas.DataFrame()
    for image in list_of_images:
        if verbose:
            logger.info('Converting {0}'.format(image))

        image_data = read_image.read_grayscale(os.path.join(plot_directory, image),
                                               resolution=0.3)
        image_data_for_cnn[image] = [image_data]

    # Now label the image
    if verbose:
        logger.info('Labelling image...')

    scores, ml_label, deeplayer, ids, filename1, filename2, filename3, filename4 = \
         label_glitches.get_deeplayer(image_data=image_data_for_cnn,
                                       model_name='{0}'.format(path_to_cnn),
                                       image_size=[140, 170],
                                       verbose=verbose)

    labels = numpy.array(classes)[ml_label]

    scores_table = GravitySpyTable(scores, names=classes)

    scores_table['Filename1'] = filename1
    scores_table['Filename2'] = filename2
    scores_table['Filename3'] = filename3
    scores_table['Filename4'] = filename4
    scores_table['gravityspy_id'] = ids
    scores_table['ml_label'] = labels
    scores_table['ml_confidence'] = scores.max(1)
    scores_table['deeplayer'] = deeplayer

    return scores_table
