from .GS_utils import build_cnn, concatenate_views
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from gravityspy.utils import log
from gwpy.table import EventTable
from gwpy.timeseries import TimeSeries

import numpy as np
import os
from . import read_image
import pandas as pd

'''
By Sara Bahaadini
This function reads the  pickle files of golden_set and train a ML classifier
and write it into a model folder
'''
def fetch_data(ifo, event_time, duration=8, sample_frequency=4096,
               verbose=False, **kwargs):
    """Fetch raw data around a glitch

    Parameters:

        ifo (str):

        event_time (str):

        duration (int, optional):

        sample_frequency (int, optional):

        verbose (bool, optional):

    Returns:

        a `gwpy.timeseries.TimeSeries`
    """
    # find closest sample time to event time
    center_time = (np.floor(event_time) +
                  np.round((event_time - np.floor(event_time)) *
                         sample_frequency) / sample_frequency)

    # determine segment start and stop times
    start_time = round(center_time - duration / 2)
    stop_time = start_time + duration
    frametype = kwargs.pop('frametype', None)
    frametype = '{0}_HOFT_{1}'.format(ifo, frametype)

    try:
        channel_name = '{0}:GDS-CALIB_STRAIN'.format(ifo)
        data = TimeSeries.get(channel_name, start_time,
                              stop_time, frametype=frametype,
                              verbose=verbose).astype('float64')
    except:
        TimeSeries.fetch_open_data(ifo, start_time, stop_time, verbose=verbose)

    if data.sample_rate.decompose().value != sample_frequency:
        data = data.resample(sample_frequency)

    return data


def training_set_raw_data(filename, format, duration=8, sample_frequency=4096,
                          verbose=False, **kwargs):
    """Obtain the raw timeseries for the whole training set

    Parameters:

        filename (str):

        format (str):

        duration (int, optional):

        sample_frequency (int, optional):

        verbose (bool, optional):

    Returns:

        A file containing the raw timeseries data of the training set
    """
    logger = log.Logger('Gravity Spy: Obtaining TimeSeries'
                        ' Data For Trainingset')
    trainingset_table = EventTable.fetch('gravityspy',
                                         'trainingsetv1d1',
                                         columns=['event_time', 'ifo',
                                                  'true_label'])
    for ifo, gps, label in zip(trainingset_table['ifo'],
                               trainingset_table['event_time'],
                               trainingset_table['true_label']):
        logger.info('Obtaining sample {0} with gps {1} from '
                    '{2}'.format(label, gps, ifo))
        data = fetch_data(ifo, gps, duration=duration,
                          sample_frequency=sample_frequency,
                          verbose=verbose, **kwargs)
        logger.info('Writing Sample To File..')
        data.write(filename, format=format,
                   append=True,
                   path='/data/{0}/{1}/'.format(label,gps))


def pickle_trainingset(path_to_trainingset,
                       save_address='pickleddata/trainingset.pkl',
                       verbose=False):
    """Pre-processes the training set images and save to pickle.

    Parameters:

        path_to_trainingset (str):
            Path to trainingset where format of training set folder
            is "somedirectoryname"/"classname/"images"

        save_address (str, optional):
            Defaults to `pickleddata`
            Path to folder you would like to save the pixelated training data

        verbose (bool, optional):
            Defaults to False
            Extra verbosity

    Returns:

        A pickled `pandas.DataFrame`:
            with rows of samples
            and columns containing the pixelated 0.5, 1.0, 2.0,
            and 4.0 duration images as well as a column with the True
            true_label and a column with an ID that uniquely identifies that sample
    """

    logger = log.Logger('Gravity Spy: Pickling '
                        'Trainingset')

    if not os.path.exists(os.path.dirname(save_address)):
        if verbose:
            logger.info('making... ' + os.path.dirname(save_address))
        os.makedirs(os.path.dirname(save_address))

    classes = sorted(os.listdir(path_to_trainingset))
    nb_classes = len(classes)
    logger.info('The number of classes are {0}'.format(nb_classes))
    logger.info('The classes you are pickling are {0}'.format(
          classes))

    data = pd.DataFrame()
    for iclass in classes:
        logger.info('Converting {0} into b/w info'.format(iclass))
        images = sorted(os.listdir(os.path.join(path_to_trainingset, iclass)))
        images = [imageidx for imageidx in images \
                  if 'L1_' in imageidx or 'H1_' in imageidx or 'V1_' in imageidx]
        # Group each sample into sets of 4 different durations
        samples = zip(*(iter(images),) * 4)
        for isample in samples:
            tmpDF = pd.DataFrame()
            for idur in isample:
                if verbose:
                    logger.info('Converting {0}'.format(idur))
                image_data = read_image.read_grayscale(os.path.join(path_to_trainingset,
                                              iclass, idur), resolution=0.3)
                information_on_image = idur.split('_')
                tmpDF[information_on_image[-1]] = [image_data]
            tmpDF['gravityspy_id'] = information_on_image[1]
            tmpDF['true_label'] = iclass
            data = data.append(tmpDF)

        logger.info('Finished converting {0} into b/w info'.format(iclass))

    picklepath = os.path.join(save_address)
    logger.info('Saving pickled data to {0}'.format(picklepath))
    data.to_pickle(picklepath)
    return data


def make_model(data, batch_size=22, nb_epoch=10,
               order_of_channels="channels_last",
               nb_classes=22, fraction_validation=.125, fraction_testing=None,
               best_model_based_validset=0, image_size=[140, 170],
               random_seed=1986, verbose=True):
    """Train a Convultional Neural Net (CNN).

    This module uses `keras <https://keras.io/>`_ to interface
    with the creation of a neural net and currently uses
    `theano <http://deeplearning.net/software/theano/>`_ as the backend
    for doing the heavy gpu lifting.

    The optimizer :class:`keras.optimizers.Adadelta`

    The loss function being optimized is `softmax <https://en.wikipedia.org/wiki/Softmax_function>`_

    Parameters:
        data (str):
            Pickle file containing training set data

        model_name (str, optional):
            Defaults to `multi_view_classifier.h5`
            path to file you would like to save the model to

        batch_size (int, optional):
            Default 22

        nb_epoch (int, optional):
            Default 40

        nb_classes (int, optional):
            Default 22

        fraction_validation (float, optional):
            Default .125

        fraction_testing (float, optional):
            Default None

        all_data_for_train_flag (int, optional):
            Default 1

        best_model_based_validset (int,optional):
            Default 0

        image_size (list, optional):
            Default [140, 170]

        random_seed (int, optional):
            Default 1986

        verbose (bool, optional):
            Default False

    Returns:
        filename:
            A trained Convultional Neural Network
    """
    logger = log.Logger('Gravity Spy: Training '
                        'Model')

    logger.info('Using random seed {0}'.format(random_seed))
    np.random.seed(random_seed)  # for reproducibility

    logger.info('You have selected the follow channel order : {0}'.format(order_of_channels))
    K.set_image_data_format(order_of_channels)

    logger.info('You data set contained {0} samples'.format(len(data)))

    img_rows, img_cols = image_size[0], image_size[1]

    logger.info('The size of the images being trained {0}'.format(image_size))

    classes = sorted(data.true_label.unique())

    if len(classes) != nb_classes:
        raise ValueError('Youre supplied data set does not match the number of'
                         ' classes you said you were training on')

    classes = dict(enumerate(classes))
    classes = dict((str(v),k) for k,v in classes.items())

    logger.info('You have supplied a training set with the following class'
                'idx to str label mapping: {0}'.format(classes))

    logger.info('converting string to idx...')
    data.true_label = data.true_label.apply(lambda x: classes[x])

    logger.info('Selecting samples for validation ...')
    logger.info('You have selected to set aside {0} percent of '
                'images per class for validation.'.format(
                                       fraction_validation * 100))

    validationDF = data.groupby('true_label').apply(
                       lambda x: x.sample(frac=fraction_validation,
                       random_state=random_seed)
                       ).reset_index(drop=True)

    logger.info('Removing validation images from training DF ...')

    data = data.loc[~data.gravityspy_id.isin(
                                    validationDF.gravityspy_id)]

    logger.info('There are now {0} samples remaining'.format(
                                                       len(data)))

    if fraction_testing:
        logger.info('Selecting samples for testing ...')
        logger.info('You have selected to set aside {0} percent of '
                'images per class for validation.'.format(

                                       fraction_testing * 100))
        testingDF = data.groupby('true_label').apply(
                   lambda x: x.sample(frac=fraction_testing,
                             random_state=random_seed)
                   ).reset_index(drop=True)

        logger.info('Removing testing images from training DF ...')

        data = data.loc[~data.gravityspy_id.isin(
                                        testingDF.gravityspy_id)]

        logger.info('There are now {0} samples remaining'.format(
                                                           len(data)))

    if order_of_channels == 'channels_last':
        reshape_order = (-1, img_rows, img_cols, 1)
    elif order_of_channels == 'channels_first':
        reshape_order = (-1, 1, img_rows, img_cols)
    else:
        raise ValueError("Do not understand supplied channel order")
    # concatenate the pixels
    train_set_x_1 = np.vstack(data['0.5.png'].values).reshape(reshape_order)
    validation_x_1 = np.vstack(validationDF['0.5.png'].values).reshape(reshape_order)

    train_set_x_2 = np.vstack(data['1.0.png'].values).reshape(reshape_order)
    validation_x_2 = np.vstack(validationDF['1.0.png'].values).reshape(reshape_order)

    train_set_x_3 = np.vstack(data['2.0.png'].values).reshape(reshape_order)
    validation_x_3 = np.vstack(validationDF['2.0.png'].values).reshape(reshape_order)

    train_set_x_4 = np.vstack(data['4.0.png'].values).reshape(reshape_order)
    validation_x_4 = np.vstack(validationDF['4.0.png'].values).reshape(reshape_order)

    if fraction_testing:
        testing_x_1 = np.vstack(testingDF['0.5.png'].values).reshape(reshape_order)
        testing_x_2 = np.vstack(testingDF['1.0.png'].values).reshape(reshape_order)
        testing_x_3 = np.vstack(testingDF['2.0.png'].values).reshape(reshape_order)
        testing_x_4 = np.vstack(testingDF['4.0.png'].values).reshape(reshape_order)

    # Concatenate the labels
    trainingset_labels = np.vstack(data['true_label'].values)
    validation_labels = np.vstack(validationDF['true_label'].values)
    if fraction_testing:
        testing_labels = np.vstack(testingDF['true_label'].values)

    # Concatenate the name
    trainingset_names = np.vstack(data['gravityspy_id'].values)
    validation_names = np.vstack(validationDF['gravityspy_id'].values)
    if fraction_testing:
        testing_names = np.vstack(testingDF['gravityspy_id'].values)

    # Categorize labels
    trainingset_labels = np_utils.to_categorical(
                                        trainingset_labels, nb_classes)

    validation_labels = np_utils.to_categorical(
                                        validation_labels, nb_classes)
    if fraction_testing:
        testing_labels = np_utils.to_categorical(
                                        testing_labels, nb_classes)

    logger.info('Concatenating multiple views ...')
    concat_train = concatenate_views(train_set_x_1, train_set_x_2,
                            train_set_x_3, train_set_x_4, [img_rows, img_cols], False, order_of_channels)
    concat_valid = concatenate_views(validation_x_1, validation_x_2,
                            validation_x_3, validation_x_4,
                            [img_rows, img_cols], False,order_of_channels)

    if fraction_testing:
        concat_test = concatenate_views(testing_x_1, testing_x_2,
                            testing_x_3, testing_x_4,
                            [img_rows, img_cols], False,order_of_channels)

    cnn1 = build_cnn(img_rows*2, img_cols*2, order_of_channels)
    final_model = Sequential()
    final_model.add(cnn1)
    final_model.add(Dense(nb_classes, activation='softmax'))

    final_model.compile(loss='categorical_crossentropy',
                        optimizer='adadelta',
                        metrics=['accuracy'])

    final_model.fit(concat_train, trainingset_labels,
        batch_size=batch_size, epochs=nb_epoch, verbose=1,
        validation_data=(concat_valid, validation_labels), callbacks=[])

    if fraction_testing:
        score = final_model.evaluate(concat_test, testing_labels, verbose=0)

        logger.info('Test accuracy (last): {0}'.format(score[1]))

        score2 = final_model.evaluate(concat_valid, validation_labels, verbose=0)
        logger.info('valid accuracy (last): {0}'.format(score2[1]))

        score3 = final_model.evaluate(concat_train, trainingset_labels, verbose=0)
        logger.info('Train accuracy (last): {0}'.format(score3[1]))

    else:
        score2 = final_model.evaluate(concat_valid, validation_labels, verbose=0)
        logger.info('valid accuracy (last): {0}'.format(score2[1]))

        score3 = final_model.evaluate(concat_train, trainingset_labels, verbose=0)
        logger.info('Train accuracy (last): {0}'.format(score3[1]))

    return final_model
