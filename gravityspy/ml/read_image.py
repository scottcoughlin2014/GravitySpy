from skimage import io
from skimage.color import rgb2gray
from skimage.transform import rescale
import numpy as np
import os
from functools import reduce


def read_and_crop_image(filename, x, y):
    """Read in a crop part of image you want to keep

    Parameters
        filename (str):
            the file you would like to pixelize

        x (float, list):
            xrange of pixels to keep

        y (float, list):
            yrange of pixels to keep


    Returns
        image_data (`np.array):
            this images is taken from rgb to gray scale
            and then downsampled by the resolution.
    """
    xmin = x[0]
    xmax = x[1]
    ymin = y[0]
    ymax = y[1]
    image_data = io.imread(filename)
    image_data = image_data[xmin:xmax, ymin:ymax, :3]
    return image_data

def read_grayscale(filename, resolution=0.3, x=[66, 532], y=[105, 671],
                   verbose=False):
    """Convert image from RGB to Gray, downsample

    Parameters
        filename (str):
            the file you would like to pixelize

        resolution (float, optional):
            default: 0.3

        verbose (bool, optional):
            default: False

    Returns
        image_data (`np.array):
            this images is taken from rgb to gray scale
            and then downsampled by the resolution.
    """
    image_data = read_and_crop_image(filename, x=x, y=y)

    image_data = rgb2gray(image_data)
    image_data = rescale(image_data, resolution, mode='constant',
                         preserve_range='True', multichannel=False)

    dim = np.int(reduce(lambda x, y: x * y, image_data.shape))
    image_data = np.reshape(image_data, (dim))
    image_data = np.array(image_data, dtype='f')

    return image_data

def read_rgb(filename, resolution=0.3, x=[66, 532], y=[105, 671],
             verbose=False):
    """Convert image from RGB to Gray, downsample

    Parameters
        filename (str):
            the file you would like to pixelize

        resolution (float, optional):
            default: 0.3

        verbose (bool, optional):
            default: False

    Returns
        image_data (`np.array):
            this images is taken from rgb to gray scale
            and then downsampled by the resolution.
    """
    image_data = read_and_crop_image(filename, x=x, y=y)
    image_data = rescale(image_data, resolution, mode='constant',
                         preserve_range='True', multichannel=True)
    dim = np.int(reduce(lambda x, y: x * y, image_data[:,:,0].shape))
    image_data_r = np.reshape(image_data[:,:,0], (dim))
    image_data_g = np.reshape(image_data[:,:,1], (dim))
    image_data_b = np.reshape(image_data[:,:,2], (dim))

    return image_data_r, image_data_g, image_data_b
