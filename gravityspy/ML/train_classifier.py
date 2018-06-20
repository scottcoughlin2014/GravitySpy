from .GS_utils import build_cnn, concatenate_views
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from gravityspy.utils import log
from gwpy.table import EventTable
from gwpy.timeseries import TimeSeries

import numpy as np
import os
import make_pickle_for_linux as make_pickle
import pandas as pd

'''
By Sara Bahaadini
This function reads the  pickle files of golden_set and train a ML classifier
and write it into a model folder
'''
def fetch_data(ifo, eventTime, blockTime=8, samplefrequency=4096):
    """Pre-processes the training set images and save to pickle.

    Parameters:

        path_to_trainingset (str):
            Path to trainingset where format of training set folder
            is "somedirectoryname"/"classname/"images"

        save_address (str, optional):
            Defaults to `pickleddata`
            Path to folder you would like to save the pixelated training data
    Returns:

        A pickled `pandas.DataFrame`:
            with rows of samples
            and columns containing the pixelated 0.5, 1.0, 2.0,
            and 4.0 duration images as well as a column with the True
            Label and a column with an ID that uniquely identifies that sample
    """
    # find closest sample time to event time
    centerTime = np.floor(eventTime) + \
               np.round((eventTime - np.floor(eventTime)) * \
                     sampleFrequency) / sampleFrequency

    # determine segment start and stop times
    startTime = round(centerTime - blockTime / 2)
    stopTime = startTime + blockTime

    try:
        channelName = '{0}:GDS-CALIB_STRAIN'.format(ifo)
        data = TimeSeries.get(channelName, startTime, stopTime).astype('float64')
    except:
        TimeSeries.fetch_open_data(ifo, startTime, stopTime)

    if data.sample_rate.decompose().value != sampleFrequency:
        data = data.resample(sampleFrequency)

    return data


def pickle_training_set_raw_data(save_address='pickleddata/train_raw_data.pkl',
                                 ):
    """Pre-processes the training set images and save to pickle.

    Parameters:

        path_to_trainingset (str):
            Path to trainingset where format of training set folder
            is "somedirectoryname"/"classname/"images"

        save_address (str, optional):
            Defaults to `pickleddata`
            Path to folder you would like to save the pixelated training data
    Returns:

        A pickled `pandas.DataFrame`:
            with rows of samples
            and columns containing the pixelated 0.5, 1.0, 2.0,
            and 4.0 duration images as well as a column with the True
            Label and a column with an ID that uniquely identifies that sample
    """
    os.environ['GWPY_CACHE'] = 1
    image_dataDF = pd.DataFrame()
    trainingset_table = EventTable.fetch('gravityspy',
                                         'trainingsetv1d1')
    for iIfo, iTrigger in zip(trainingset_table['ifo'],
                              trainingset_table['peakGPS']):
        data = fetch_data(iIfo, iTrigger)


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
            Label and a column with an ID that uniquely identifies that sample
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

    image_dataDF = pd.DataFrame()
    for iclass in classes:
        logger.info('Converting {0} into b/w info'.format(iclass))
        images = sorted(os.listdir(os.path.join(path_to_trainingset, iclass)))
        images = [imageidx for imageidx in images \
                  if 'L1_' in imageidx or 'H1_' in imageidx]
        # Group each sample into sets of 4 different durations
        samples = zip(*(iter(images),) * 4)
        for isample in samples:
            tmpDF = pd.DataFrame()
            for idur in isample:
                if verbose:
                    logger.info('Converting {0}'.format(idur))
                image_data = make_pickle.main(os.path.join(path_to_trainingset,
                                              iclass, idur), resolution=0.3)
                information_on_image = idur.split('_')
                tmpDF[information_on_image[-1]] = [image_data]
            tmpDF['uniqueID'] = information_on_image[1]
            tmpDF['Label'] = iclass
            image_dataDF = image_dataDF.append(tmpDF)

        logger.info('Finished converting {0} into b/w info'.format(iclass))

    picklepath = os.path.join(save_address)
    logger.info('Saving pickled data to {0}'.format(picklepath))
    image_dataDF.to_pickle(picklepath)
    return image_dataDF


def make_model(data, model_folder='model', batch_size=22, nb_epoch=10,
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

        model_folder (str, optional):
            Defaults to `model`
            path to folder you would like to save the model

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
        file:
            name `final_model/multi_view_classifier.h5`
            A trained Convultional Neural Network
    """
    logger = log.Logger('Gravity Spy: Training '
                        'Model')

    logger.info('Using random seed {0}'.format(random_seed))
    np.random.seed(random_seed)  # for reproducibility

    logger.info('Loading file {0}'.format(data))

    image_dataDF = pd.read_pickle(data)

    logger.info('You data set contained {0} samples'.format(len(image_dataDF)))

    img_rows, img_cols = image_size[0], image_size[1]

    logger.info('The size of the images being trained {0}'.format(image_size))

    classes = sorted(image_dataDF.Label.unique())

    if len(classes) != nb_classes:
        raise ValueError('Youre supplied data set does not match the number of'
                         ' classes you said you were training on')

    classes = dict(enumerate(classes))
    classes = dict((str(v),k) for k,v in classes.iteritems())

    logger.info('You have supplied a training set with the following class'
                'idx to str label mapping: {0}'.format(classes))

    logger.info('converting string to idx...')
    image_dataDF.Label = image_dataDF.Label.apply(lambda x: classes[x])

    logger.info('Selecting samples for validation ...')
    logger.info('You have selected to set aside {0} percent of '
                'images per class for validation.'.format(
                                       fraction_validation * 100))

    validationDF = image_dataDF.groupby('Label').apply(
                       lambda x: x.sample(frac=fraction_validation,
                       random_state=random_seed)
                       ).reset_index(drop=True)

    logger.info('Removing validation images from training DF ...')

    image_dataDF = image_dataDF.loc[~image_dataDF.uniqueID.isin(
                                    validationDF.uniqueID)]

    logger.info('There are now {0} samples remaining'.format(
                                                       len(image_dataDF)))

    if fraction_testing:
        logger.info('Selecting samples for testing ...')
        logger.info('You have selected to set aside {0} percent of '
                'images per class for validation.'.format(

                                       fraction_testing * 100))
        testingDF = image_dataDF.groupby('Label').apply(
                   lambda x: x.sample(frac=fraction_testing,
                             random_state=random_seed)
                   ).reset_index(drop=True)

        logger.info('Removing testing images from training DF ...')

        image_dataDF = image_dataDF.loc[~image_dataDF.uniqueID.isin(
                                        testingDF.uniqueID)]

        logger.info('There are now {0} samples remaining'.format(
                                                           len(image_dataDF)))

    # concatenate the pixels
    train_set_x_1 = np.vstack(image_dataDF['0.5.png'].as_matrix()).reshape(
                                                     -1, 1, img_rows, img_cols)
    validation_x_1 = np.vstack(validationDF['0.5.png'].as_matrix()).reshape(
                                                     -1, 1, img_rows, img_cols)

    train_set_x_2 = np.vstack(image_dataDF['1.0.png'].as_matrix()).reshape(
                                                     -1, 1, img_rows, img_cols)
    validation_x_2 = np.vstack(validationDF['1.0.png'].as_matrix()).reshape(
                                                     -1, 1, img_rows, img_cols)

    train_set_x_3 = np.vstack(image_dataDF['2.0.png'].as_matrix()).reshape(
                                                     -1, 1, img_rows, img_cols)
    validation_x_3 = np.vstack(validationDF['2.0.png'].as_matrix()).reshape(
                                                     -1, 1, img_rows, img_cols)

    train_set_x_4 = np.vstack(image_dataDF['4.0.png'].as_matrix()).reshape(
                                                     -1, 1, img_rows, img_cols)
    validation_x_4 = np.vstack(validationDF['4.0.png'].as_matrix()).reshape(
                                                     -1, 1, img_rows, img_cols)

    if fraction_testing:
        testing_x_1 = np.vstack(testingDF['0.5.png'].as_matrix()).reshape(
                                                     -1, 1, img_rows, img_cols)
        testing_x_2 = np.vstack(testingDF['1.0.png'].as_matrix()).reshape(
                                                     -1, 1, img_rows, img_cols)
        testing_x_3 = np.vstack(testingDF['2.0.png'].as_matrix()).reshape(
                                                     -1, 1, img_rows, img_cols)
        testing_x_4 = np.vstack(testingDF['4.0.png'].as_matrix()).reshape(
                                                     -1, 1, img_rows, img_cols)

    # Concatenate the labels
    trainingset_labels = np.vstack(image_dataDF['Label'].as_matrix())
    validation_labels = np.vstack(validationDF['Label'].as_matrix())
    if fraction_testing:
        testing_labels = np.vstack(testingDF['Label'].as_matrix())

    # Concatenate the name
    trainingset_names = np.vstack(image_dataDF['uniqueID'].as_matrix())
    validation_names = np.vstack(validationDF['uniqueID'].as_matrix())
    if fraction_testing:
        testing_names = np.vstack(testingDF['uniqueID'].as_matrix())

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
                            train_set_x_3, train_set_x_4, [img_rows, img_cols])
    concat_valid = concatenate_views(validation_x_1, validation_x_2,
                            validation_x_3, validation_x_4,
                            [img_rows, img_cols])

    if fraction_testing:
        concat_test = concatenate_views(testing_x_1, testing_x_2,
                            testing_x_3, testing_x_4,
                            [img_rows, img_cols])

    cnn1 = build_cnn(img_rows*2, img_cols*2)
    final_model = Sequential()
    final_model.add(cnn1)
    final_model.add(Dense(nb_classes, activation='softmax'))

    final_model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    if not os.path.exists(model_folder):
        if verbose:
            logger.info('making... ' + model_folder)
        os.makedirs(model_folder)

    full_model_adr = model_folder

    loss_checker = ModelCheckpoint(os.path.join(full_model_adr,
                                                "best_weights_loss.h5"),
                                   monitor='val_loss', verbose=1,
                                   save_best_only=True, mode='auto',
                                   save_weights_only=True)
    acc_checker = ModelCheckpoint(os.path.join(full_model_adr,
                                               "best_weights_acc.h5"),
                                  monitor='val_acc', verbose=1,
                                  save_best_only=True, mode='auto',
                                  save_weights_only=True)

    if best_model_based_validset:
        callbacks = [acc_checker, loss_checker]
    else:
        callbacks = []

    final_model_adr = os.path.join(full_model_adr, 'final_model')
    if not os.path.exists(final_model_adr):
        logger.info('making... ' + final_model_adr)
        os.makedirs(final_model_adr)
    out_file = open(os.path.join(final_model_adr, 'out.txt'), "w")
    out_file.write(data + '\n')


    final_model.fit(concat_train, trainingset_labels,
        batch_size=batch_size, epochs=nb_epoch, verbose=1,
        validation_data=(concat_valid, validation_labels), callbacks=callbacks)

    if fraction_testing:
        score = final_model.evaluate(concat_test, testing_labels, verbose=0)

        logger.info('Test accuracy (last): {0}'.format(score[1]))
        out_file.write('\n * Test Accuracy (last): %0.4f%% \n' % score[1])

        score2 = final_model.evaluate(concat_valid, validation_labels, verbose=0)
        logger.info('valid accuracy (last): {0}'.format(score2[1]))

        score3 = final_model.evaluate(concat_train, trainingset_labels, verbose=0)
        logger.info('Train accuracy (last): {0}'.format(score3[1]))

    else:
        score2 = final_model.evaluate(concat_valid, validation_labels, verbose=0)
        logger.info('valid accuracy (last): {0}'.format(score2[1]))

        score3 = final_model.evaluate(concat_train, trainingset_labels, verbose=0)
        logger.info('Train accuracy (last): {0}'.format(score3[1]))

    final_model.save(os.path.join(final_model_adr, 'multi_view_classifier.h5'))

    if best_model_based_validset:
        final_model.load_weights(full_model_adr + "/best_weights_acc.h5")
        score = final_model.evaluate(concat_test, cat_test_set_y_1, verbose=0)
        print('Test accuracy (acc):', score[1])
        out_file.write('\n * Test Accuracy (acc): %0.4f%% \n' % score[1])
        score2 = final_model.evaluate(concat_valid, cat_valid_set_y_1, verbose=0)
        print('valid accuracy (acc):', score2[1])
        score3 = final_model.evaluate(concat_train, cat_train_set_y_1, verbose=0)
        print('Train accuracy (acc):', score3[1])

        final_model.load_weights(full_model_adr + "/best_weights_loss.h5")
        score = final_model.evaluate(concat_test, cat_test_set_y_1, verbose=0)
        print('Test accuracy (loss):', score[1])
        out_file.write('\n * Test Accuracy (loss): %0.4f%% \n' % score[1])
        score2 = final_model.evaluate(concat_valid, cat_valid_set_y_1, verbose=0)
        print('valid accuracy (loss):', score2[1])
        score3 = final_model.evaluate(concat_train, cat_train_set_y_1, verbose=0)
        print('Train accuracy (loss):', score3[1])

        return
