from skimage import io
from skimage.color import rgb2gray
from skimage.transform import rescale
import numpy as np
import getopt, sys, os


def main(filename, resolution=0.3, verbose=False):
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

    np.random.seed(1986)  # for reproducibility

    image_data = io.imread(filename)
    image_data = image_data[66:532, 105:671, :]
    image_data = rgb2gray(image_data)
    image_data = rescale(image_data, resolution, mode='constant',
                         preserve_range='True')
    dim = np.int(reduce(lambda x, y: x * y, image_data.shape))
    image_data = np.reshape(image_data, (dim))
    image_data = np.array(image_data, dtype='f')

    return image_data
