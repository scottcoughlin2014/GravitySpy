"""Unit test for GravitySpy
"""

__author__ = 'Scott Coughlin <scott.coughlin@ligo.org>'

import os
os.environ["KERAS_BACKEND"] = "theano"

import gravityspy.ML.make_pickle_for_linux as make_pickle
import gravityspy.ML.labelling_test_glitches as label_glitches
import gravityspy.ML.train_classifier as train_classifier

import pandas as pd
import numpy

TEST_IMAGES_PATH = os.path.join(os.path.split(__file__)[0], 'data',
'images')
MODEL_PATH = os.path.join(os.path.split(__file__)[0], '..', '..', 'bin')

SCORE = 0.9997797608375549 

FEATURES = numpy.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
136.32681274414062, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 143.1021728515625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 74.27071380615234, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 110.8934326171875, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
142.75534057617188, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 120.54851531982422, 132.58575439453125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 125.55241394042969, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 154.26510620117188, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 34.673343658447266, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

class TestGravitySpyML(object):
    """`TestCase` for the GravitySpy
    """
    def test_label(self):

        list_of_images = []
        for ifile in os.listdir(TEST_IMAGES_PATH):
            if 'spectrogram' in ifile:
                list_of_images.append(ifile)

        image_dataDF = pd.DataFrame()
        for idx, image in enumerate(list_of_images):
            image_data = make_pickle.main(os.path.join(
                                                       TEST_IMAGES_PATH,
                                                       image),
                                          resolution=0.3)

            image_dataDF[image] = [image_data]

        # Now label the image
        scores, MLlabel = label_glitches.label_glitches(
                                                        image_dataDF,
                                                        '{0}'.format(
                                                              MODEL_PATH),
                                                        [140, 170],
                                                        False)

        confidence = float(scores[0][MLlabel])
        assert confidence == SCORE


    def test_feature_space(self):
        list_of_images = []
        for ifile in os.listdir(TEST_IMAGES_PATH):
            if 'spectrogram' in ifile:
                list_of_images.append(ifile)

        # Get ML semantic index image data
        image_dataDF = pd.DataFrame()
        for idx, image in enumerate(list_of_images):
            if '1.0.png' in image:
                image_data = make_pickle.main(os.path.join(TEST_IMAGES_PATH, image), resolution=0.3)
                image_dataDF[image] = [image_data]

        # Determine features
        features = label_glitches.get_feature_space(image_data=image_dataDF,
                                              semantic_model_adr='{0}'.format(MODEL_PATH),
                                              image_size=[140, 170],
                                              verbose=False)

        numpy.testing.assert_array_almost_equal(features, FEATURES, decimal=3)
