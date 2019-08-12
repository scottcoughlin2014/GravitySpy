""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""
from keras import backend as K
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D, Conv2D

import numpy as np

#4/2/2018
def concatenate_views(image_set1, image_set2, image_set3,
                      image_set4, image_size, rgb_flag,
                      order_of_channels):
    """Create a merged view from a set of 4 views of one sample image

    Parameters:
        image_set1 (array):
            The grayscale downsamples pixels for one duration of one
            of the samples

        image_set2 (array):
            The grayscale downsamples pixels for one duration of one
            of the samples

        image_set3 (array):
            The grayscale downsamples pixels for one duration of one
            of the samples

        image_set4 (array):
            The grayscale downsamples pixels for one duration of one
            of the samples

        image_size (list):
            This refers to the shape of the non flattened pixelized image
            array

        rgb_flag (bool):
            Are you trying to create a merged view of grayscale
            or RGB image renders

    Returns:
        concated_view (array):
            A single merged view of the sample, in the case
            a merged view of the 0.5 1.0 2.0 and 4.0 duration
            omega scans.
    """
    img_rows = image_size[0]
    img_cols = image_size[1]
    if rgb_flag:
        ch = 3
    else:
        ch = 1

    if order_of_channels == 'channels_last':
        concat_images_set = np.zeros((len(image_set1), img_rows * 2, img_cols, ch))
        concat_images_set2 = np.zeros((len(image_set3), img_rows * 2, img_cols, ch))
        concat_row_axis = 0
        concat_col_axis = 2
    elif order_of_channels == 'channels_first':
        concat_images_set = np.zeros((len(image_set1), ch, img_rows * 2, img_cols))
        concat_images_set2 = np.zeros((len(image_set3), ch, img_rows * 2, img_cols))
        concat_row_axis = 1
        concat_col_axis = 3
    else:
        raise ValueError("Do not understand supplied channel order")

    assert len(image_set1) == len(image_set2)
    for i in range(0, len(image_set1)):
        concat_images_set[i, :, :, :] = np.append(image_set1[i, :, :, :], image_set2[i, :, :, :], axis=concat_row_axis)

    assert len(image_set3) == len(image_set4)
    for i in range(0, len(image_set3)):
        concat_images_set2[i, :, :, :] = np.append(image_set3[i, :, :, :], image_set4[i, :, :, :], axis=concat_row_axis)

    out = np.append(concat_images_set,concat_images_set2, axis=concat_col_axis)
    return out

#4/2/2018
def build_cnn(img_rows, img_cols, order_of_channels):
    """This is where we use Keras to build a covolutional neural network (CNN)

    The CNN built here is described in the
    `Table 5 <https://www.sciencedirect.com/science/article/pii/S0020025518301634#tbl0004>`_

    There are 5 layers. For each layer the logic is as follows

    input 2D matrix --> 2D Conv layer with x number of kernels with a 5 by 5 shape
    --> activation layer we use `ReLU <https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`_
    --> MaxPooling of size 2 by 2 all pixels are now grouped into bigger pixels of
    size 2 by 2 and the max pixel of the pixels that make up the 2 by 2 pizels is the
    value of the bigger pixel --> Dropout set to 50 percent. This means each pixel
    at this stage has a 50 percent chance of being set to 0.

    Parameters:
        image_rows (int):
            This refers to the number of rows in the non-flattened image

        image_cols (int):
            This refers to the number of cols in the non-flattened image

    Returns:
        model (`object`):
            a CNN
    """
    W_reg = 1e-4
    print('regularization parameter: ', W_reg)
    if order_of_channels == 'channels_last':
        input_shape = (img_rows, img_cols, 1)
    elif order_of_channels == 'channels_first':
        input_shape = (1, img_rows, img_cols)
    else:
        raise ValueError("Do not understand supplied channel order")
    model = Sequential()
    model.add(Conv2D(16, (5, 5), padding='valid',
              input_shape=input_shape,
              kernel_regularizer=l2(W_reg)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))


    model.add(Conv2D(32, (5, 5), padding='valid', kernel_regularizer=l2(W_reg)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (5, 5), padding='valid', kernel_regularizer=l2(W_reg)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (5, 5), padding='valid', kernel_regularizer=l2(W_reg)))
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, kernel_regularizer=l2(W_reg)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    print (model.summary())
    return model


def cosine_distance(vects):
    """Calculate the cosine distance of an array

    Parameters:

        vect (array):
    """
    x, y = vects
    x = K.maximum(x, K.epsilon())
    y = K.maximum(y, K.epsilon())
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return 1.0 - K.sum(x * y, axis=1, keepdims=True)

def siamese_acc(thred):
    """Calculate simaese accuracy

    Parameters:
        thred (float):
            It is something
    """
    def inner_siamese_acc(y_true, y_pred):
        pred_res = y_pred < thred
        acc = K.mean(K.cast(K.equal(K.cast(pred_res, dtype='int32'), K.cast(y_true, dtype='int32')), dtype='float32'))
        return acc

    return inner_siamese_acc

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def create_pairs3_gen(data, class_indices, batch_size):
    """ Create the pairs

    Parameters:
        data (float):
            It is something
        class_indices (list):
            It is something
        batch_size (int):
            It is something
    """
    pairs1 = []
    pairs2 = []
    labels = []
    number_of_classes = len(class_indices)
    counter = 0
    while True:
        for d in range(len(class_indices)):
            for i in range(len(class_indices[d])):
                counter += 1
                # positive pair
                j = random.randrange(0, len(class_indices[d]))
                z1, z2 = class_indices[d][i], class_indices[d][j]

                pairs1.append(data[z1])
                pairs2.append(data[z2])
                labels.append(1)

                # negative pair
                inc = random.randrange(1, number_of_classes)
                other_class_id = (d + inc) % number_of_classes
                j = random.randrange(0, len(class_indices[other_class_id])-1)
                z1, z2 = class_indices[d][i], class_indices[other_class_id][j]

                pairs1.append(data[z1])
                pairs2.append(data[z2])
                labels.append(0)

                if counter == batch_size:
                    #yield np.array(pairs), np.array(labels)
                    yield [np.asarray(pairs1, np.float32), np.asarray(pairs2, np.float32)], np.asarray(labels, np.int32)
                    counter = 0
                    pairs1 = []
                    pairs2 = []
                    labels = []


def split_data_set(data, fraction_validation=.125, fraction_testing=None,
                   image_size=[140, 170]):
    """Split data set to training validation and optional testing

    Parameters:
        data (str):
            Pickle file containing training set data

        fraction_validation (float, optional):
            Default .125

        fraction_testing (float, optional):
            Default None

        image_size (list, optional):
            Default [140, 170]

    Returns:
        numpy arrays
    """

    img_rows, img_cols = image_size[0], image_size[1]
    validationDF = data.groupby('Label').apply(
                       lambda x: x.sample(frac=fraction_validation,
                       random_state=random_seed)
                       ).reset_index(drop=True)

    data = data.loc[~data.uniqueID.isin(
                                    validationDF.uniqueID)]

    if fraction_testing:
        testingDF = data.groupby('Label').apply(
                   lambda x: x.sample(frac=fraction_testing,
                             random_state=random_seed)
                   ).reset_index(drop=True)

        data = data.loc[~data.uniqueID.isin(
                                        testingDF.uniqueID)]


    # concatenate the pixels
    train_set_x_1 = np.vstack(data['0.5.png'].values).reshape(
                                                     -1, 1, img_rows, img_cols)
    validation_x_1 = np.vstack(validationDF['0.5.png'].values).reshape(
                                                     -1, 1, img_rows, img_cols)

    train_set_x_2 = np.vstack(data['1.0.png'].values).reshape(
                                                     -1, 1, img_rows, img_cols)
    validation_x_2 = np.vstack(validationDF['1.0.png'].values).reshape(
                                                     -1, 1, img_rows, img_cols)

    train_set_x_3 = np.vstack(data['2.0.png'].values).reshape(
                                                     -1, 1, img_rows, img_cols)
    validation_x_3 = np.vstack(validationDF['2.0.png'].values).reshape(
                                                     -1, 1, img_rows, img_cols)

    train_set_x_4 = np.vstack(data['4.0.png'].values).reshape(
                                                     -1, 1, img_rows, img_cols)
    validation_x_4 = np.vstack(validationDF['4.0.png'].values).reshape(
                                                     -1, 1, img_rows, img_cols)

    if fraction_testing:
        testing_x_1 = np.vstack(testingDF['0.5.png'].values).reshape(
                                                     -1, 1, img_rows, img_cols)
        testing_x_2 = np.vstack(testingDF['1.0.png'].values).reshape(
                                                     -1, 1, img_rows, img_cols)
        testing_x_3 = np.vstack(testingDF['2.0.png'].values).reshape(
                                                     -1, 1, img_rows, img_cols)
        testing_x_4 = np.vstack(testingDF['4.0.png'].values).reshape(
                                                     -1, 1, img_rows, img_cols)

    concat_train = concatenate_views(train_set_x_1, train_set_x_2,
                            train_set_x_3, train_set_x_4, [img_rows, img_cols], False)
    concat_valid = concatenate_views(validation_x_1, validation_x_2,
                            validation_x_3, validation_x_4,
                            [img_rows, img_cols], False)

    if fraction_testing:
        concat_test = concatenate_views(testing_x_1, testing_x_2,
                            testing_x_3, testing_x_4,
                            [img_rows, img_cols], False)
    else:
        concat_test = None

    return concat_train, concat_valid, concat_test
