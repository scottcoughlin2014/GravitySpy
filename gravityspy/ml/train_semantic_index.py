import keras.backend as K
from GS_utils import (cosine_distance,
                      siamese_acc, eucl_dist_output_shape,
                      contrastive_loss, concatenate_views)
from keras import regularizers
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import Input, Dense, GlobalAveragePooling2D, Lambda
from keras.models import Model,load_model
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop
from itertools import combinations

from gravityspy.utils import log
from .read_image import read_rgb

import numpy
import os
import pandas
import random

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
                        'Trainingset RGB')

    if not os.path.exists(os.path.dirname(save_address)):
        if verbose:
            logger.info('making... ' + os.path.dirname(save_address))
        os.makedirs(os.path.dirname(save_address))

    classes = sorted(os.listdir(path_to_trainingset))
    nb_classes = len(classes)
    logger.info('The number of classes are {0}'.format(nb_classes))
    logger.info('The classes you are pickling are {0}'.format(
          classes))

    data = pandas.DataFrame()
    for iclass in classes:
        logger.info('Converting {0} into RGB info'.format(iclass))
        images = sorted(os.listdir(os.path.join(path_to_trainingset, iclass)))
        images = [imageidx for imageidx in images \
                  if 'L1_' in imageidx or 'H1_' in imageidx or 'V1_' in imageidx]
        # Group each sample into sets of 4 different durations
        samples = zip(*(iter(images),) * 4)
        for isample in samples:
            tmpdf = pandas.DataFrame()
            for idur in isample:
                if verbose:
                    logger.info('Converting {0}'.format(idur))
                image_data_r, image_data_g, image_data_b  = read_rgb(os.path.join(path_to_trainingset,
                                                                                  iclass, idur),
                                                                     resolution=0.3)
                information_on_image = idur.split('_')
                tmpdf[information_on_image[-1]] = [[image_data_r, image_data_g, image_data_b]]
            tmpdf['gravityspy_id'] = information_on_image[1]
            tmpdf['true_label'] = iclass
            data = data.append(tmpdf)

        logger.info('Finished converting {0} into b/w info'.format(iclass))

    picklepath = os.path.join(save_address)
    logger.info('Saving pickled data to {0}'.format(picklepath))
    data.to_pickle(picklepath)
    return data

def make_model(data,
               order_of_channels="channels_last",
               unknown_classes_labels=['Paired_Doves', 'Power_Line',
                                       '1080Lines', '1400Ripples', 'Scratchy',
                                       'Repeating_Blips', 'Helix'],
               known_classes_labels=['1080Lines', '1400Ripples',
                                     'Air_Compressor', 'Blip', 'Chirp',
                                     'Extremely_Loud', 'Helix', 'Koi_Fish',
                                     'Light_Modulation', 'Low_Frequency_Burst',
                                     'Low_Frequency_Lines', 'No_Glitch',
                                     'Paired_Doves', 'Power_Line',
                                     'Repeating_Blips', 'Scattered_Light',
                                     'Scratchy', 'Tomte',
                                     'Violin_Mode', 'Wandering_Line',
                                     'Whistle'],
               train_vgg=False,
               multi_view=True,
               batch_size=22, nb_epoch=50,
               training_steps_per_epoch=1000,
               validation_steps_per_epoch=100,
               image_size=[140, 170],
               random_seed=1986, verbose=True):
    """Train a Semantic Index.

    This module uses `keras <https://keras.io/>`_ to interface
    with the creation of a neural net and currently uses
    `theano <http://deeplearning.net/software/theano/>`_ as the backend
    for doing the heavy gpu lifting.

    The optimizer :class:`keras.optimizers.Adadelta`

    The loss function being optimized is `softmax <https://en.wikipedia.org/wiki/Softmax_function>`_

    Parameters:
        data (`pandas.DataFrame`):
            pandas Data Frame scontianing the RGB values for the training set

        unknown_classes_labels (list, optional):
            Defaults to
            ``['Paired_Doves', 'Power_Line', '1080Lines', '1400Ripples',
               'Scratchy', 'Repeating_Blips', 'Helix']``
            A list of classes to be considered as the unknown
            domain for which clustering will be performed
            once the knowledge from the known classes
            has been trained.

        known_classes_labels (list, optional):
            Defaults to
            ``['1080Lines', '1400Ripples',
               'Air_Compressor', 'Blip', 'Chirp',
               'Extremely_Loud', 'Helix', 'Koi_Fish',
               'Light_Modulation', 'Low_Frequency_Burst',
               'Low_Frequency_Lines', 'No_Glitch',
               'Paired_Doves', 'Power_Line',
               'Repeating_Blips', 'Scattered_Light',
               'Scratchy', 'Tomte',
               'Violin_Mode', 'Wandering_Line',
               'Whistle']``
            A list of classes to be considered as the unknown
            domain for which clustering will be performed
            once the knowledge from the known classes
            has been trained.

        batch_size (int, optional):
            Default 22

        nb_epoch (int, optional):
            Default 50

        training_steps_per_epoch (int, optional):
            Default 1000

        validation_steps_per_epoch (int, optional):
            Default 100

        image_size (list):
            This refers to the shape of the non flattened pixelized image
            array: default = [140, 170]

    Returns:
        semantic_idx_model (`keras.Model`):
            this model gives you a 200 dimensional feature space output

        similarity_model (`keras.Model`)
            this model takes two images as input as tells you if they are the
            same or not
    """
    reglularization = 1e-4
    logger = log.Logger('Gravity Spy: Training '
                        'Semantic Index')

    logger.info('You have selected the follow channel order : {0}'.format(order_of_channels))
    K.set_image_data_format(order_of_channels)

    logger.info('Using random seed {0}'.format(random_seed))
    numpy.random.seed(random_seed)  # for reproducibility

    logger.info('You data set contained {0} samples'.format(len(data)))

    img_rows, img_cols = image_size[0], image_size[1]

    logger.info('The size of the images being trained {0}'.format(image_size))

    # Tell user what classes are known for similarity search training and what
    # are unknown
    logger.info('Selecting images to be considered in the known '
                'domain of samples which are {0}'.format(known_classes_labels))

    logger.info('Selecting images to be considered in the unknown '
                'domain of samples which are {0}'.format(unknown_classes_labels))

    # Create dict matching string labels to idx labels
    logger.info('Removing NOA images from the training set.')

    data = data.loc[data.true_label != 'None_of_the_Above']
    tmp = dict(enumerate(sorted(data.true_label.unique())))
    str_to_idx = dict((str(v),k) for k,v in tmp.items())
    data['idx_label'] = data.true_label.apply(lambda x: str_to_idx[x])

    known_classes_labels_idx = [str_to_idx[v] for v in known_classes_labels]
    unknown_classes_labels_idx = [str_to_idx[v] for v in unknown_classes_labels]

    if order_of_channels == 'channels_last':
        reshape_order = (-1, img_rows, img_cols, 3)
        channels_order = (img_rows, img_cols, 3)
    elif order_of_channels == 'channels_first':
        reshape_order = (-1, 3, img_rows, img_cols)
        channels_order = (3, img_rows, img_cols)
    else:
        raise ValueError("Do not understand supplied channel order")

    if known_classes_labels is None:
        known_df = data.loc[~data.true_label.isin(unknown_classes_labels)]
    else:
        known_df = data.loc[data.true_label.isin(known_classes_labels)]

    unknown_df = data.loc[data.true_label.isin(unknown_classes_labels)]

    known_data_label = known_df['idx_label'].values
    unknown_data_label = unknown_df['idx_label'].values

    known_x_1 = numpy.vstack(known_df['0.5.png'].values).reshape(reshape_order)
    unknown_x_1 = numpy.vstack(unknown_df['0.5.png'].values).reshape(reshape_order)

    known_x_2 = numpy.vstack(known_df['1.0.png'].values).reshape(reshape_order)
    unknown_x_2 = numpy.vstack(unknown_df['1.0.png'].values).reshape(reshape_order)

    known_x_3 = numpy.vstack(known_df['2.0.png'].values).reshape(reshape_order)
    unknown_x_3 = numpy.vstack(unknown_df['2.0.png'].values).reshape(reshape_order)

    known_x_4 = numpy.vstack(known_df['4.0.png'].values).reshape(reshape_order)
    unknown_x_4 = numpy.vstack(unknown_df['4.0.png'].values).reshape(reshape_order)

    if multi_view:
        known_classes = concatenate_views(known_x_1, known_x_2,
                                          known_x_3, known_x_4,
                                          [img_rows, img_cols],
                                          True, order_of_channels)
        unknown_classes = concatenate_views(unknown_x_1, unknown_x_2,
                                            unknown_x_3, unknown_x_4,
                                            [img_rows, img_cols],
                                            True, order_of_channels)
    else:
        # We are only using one duration for the similarity search
        known_classes = known_x_2
        unknown_classes = unknown_x_2

    known_classes = preprocess_input(known_classes)
    unknown_classes = preprocess_input(unknown_classes)

    known_classes = known_classes.astype(numpy.float32)
    unknown_classes = unknown_classes.astype(numpy.float32)

    known_classes_indices_for_metric_learning = [numpy.where(known_data_label == i)[0] for i in known_classes_labels_idx]
    unknown_classes_indices_for_metric_learning = [numpy.where(unknown_data_label == i)[0] for i in unknown_classes_labels_idx]
    # create binary pairs for known classes
    train_generator = create_pairs3_gen(known_classes, known_classes_indices_for_metric_learning,
                                         batch_size)
    valid_generator = create_pairs3_gen(unknown_classes, unknown_classes_indices_for_metric_learning,
                                        batch_size)

    # Create the model
    vgg16 = VGG16(weights='imagenet', include_top=False,
                  input_shape=known_classes.shape[1:])
    x = vgg16.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, kernel_regularizer=regularizers.l2(reglularization))(x)
    x = Dense(200)(x)
    predictions = LeakyReLU(alpha=0.3)(x)

    #Then create the corresponding model
    base_network = Model(inputs=vgg16.input, outputs=predictions)

    if order_of_channels == 'channels_last':
        if multi_view:
            img_cols = 2*img_cols
            img_rows = 2*img_rows
        channels_order = (img_rows, img_cols, 3)
    elif order_of_channels == 'channels_first':
        if multi_view:
            img_cols = 2*img_cols
            img_rows = 2*img_rows
        channels_order = (3, img_rows, img_cols)
    else:
        raise ValueError("Do not understand supplied channel order")

    input_a = Input(shape=channels_order)
    input_b = Input(shape=channels_order)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(cosine_distance,
                      output_shape=eucl_dist_output_shape)(
                          [processed_a, processed_b]
                      )

    similarity_model = Model(inputs=[input_a, input_b], outputs=distance)
    semantic_idx_model = Model(inputs=[input_a], outputs=processed_a)

    if not train_vgg:
        number_of_cnn_from_vgg = 0
        for i in range(len(vgg16.layers)-number_of_cnn_from_vgg):
            vgg16.layers[i].trainable = False

    similarity_model.summary()
    semantic_idx_model.summary()
    rms = RMSprop()

    similarity_model.compile(loss=contrastive_loss, optimizer=rms,
                             metrics=[siamese_acc(0.1), siamese_acc(0.3), siamese_acc(0.4),
                                      siamese_acc(0.5), siamese_acc(0.6),
                                      siamese_acc(0.7), siamese_acc(0.8),
                                      siamese_acc(0.9), siamese_acc(0.925),
                                      siamese_acc(0.95), siamese_acc(0.975),
                                      siamese_acc(0.985),siamese_acc(0.99)])

    # train
    logger.info('training the model ...')

    logger.info('training steps per epoch {0}'.format(training_steps_per_epoch))
    logger.info('validation steps per epoch {0}'.format(validation_steps_per_epoch))

    similarity_model.fit_generator(train_generator,
                                   validation_data=valid_generator,
                                   steps_per_epoch=training_steps_per_epoch,
                                   validation_steps=validation_steps_per_epoch,
                                   epochs=nb_epoch,
                                   verbose=2,
                                   )

    # validation
    logger.info('validating the model')

    logger.info('Known classes')
    res1 = similarity_model.evaluate_generator(train_generator,
                                               training_steps_per_epoch)
    logger.info(res1)

    logger.info(' unknown classes')
    res2 = similarity_model.evaluate_generator(valid_generator,
                                               validation_steps_per_epoch)
    logger.info(res2)

    return semantic_idx_model, similarity_model

def create_pairs3_gen(data, class_indices, batch_size):
    pairs1 = []
    pairs2 = []
    labels = []
    number_of_classes = len(class_indices)
    counter = 0
    while True:
        for d in numpy.random.randint(0, number_of_classes, size=number_of_classes):
            num_images_in_class_d = len(class_indices[d])
            for i in numpy.random.randint(0, num_images_in_class_d, size=num_images_in_class_d):
                counter += 1
                # positive pair
                j = numpy.random.randint(0, high=len(class_indices[d]))
                if j == i:
                    # Pick another image from same class then
                    j = numpy.random.randint(0, high=len(class_indices[d]))
                z1, z2 = class_indices[d][i], class_indices[d][j]

                pairs1.append(data[z1])
                pairs2.append(data[z2])
                labels.append(1)

                # negative pair
                inc = numpy.random.randint(1, high=number_of_classes)
                other_class_id = (d + inc) % number_of_classes
                j = numpy.random.randint(0, high=len(class_indices[other_class_id])-1)
                z1, z2 = class_indices[d][i], class_indices[other_class_id][j]

                pairs1.append(data[z1])
                pairs2.append(data[z2])
                labels.append(0)

                if counter == batch_size:
                    #yield np.array(pairs), np.array(labels)
                    yield [numpy.asarray(pairs1, numpy.float32), numpy.asarray(pairs2, numpy.float32)], numpy.asarray(labels, numpy.int32)
                    counter = 0
                    pairs1 = []
                    pairs2 = []
                    labels = []
