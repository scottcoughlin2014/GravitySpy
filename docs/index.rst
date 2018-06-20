.. GravitySpy documentation master file, created by
   sphinx-quickstart on Thu Apr 21 14:05:08 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GravitySpy's documentation!
======================================

`Gravity Spy <https://gravityspy.org>`_ is an innovative citizen-science meets Machine Learning meets gravitational wave physics project. This repository is meant to faciliate the creation of new similar citizen science projects on `Zooniverse <https://zooniverse.org>`_

The module level docstrings attempt to follow the following format : `Google Style Sphinx <http://www.sphinx-doc.org/en/master/ext/example_google.html>`_


Installing GravitySpy
---------------------

The easiest method to install gravityspy is using `pip <https://pip.pypa.io/en/stable/>`_ directly from the `GitHub repository <https://github.com/Gravity-Spy/GravitySpy.git>`_:

Gravity Spy software has been tested on py35 and py27. Work is in progress to have unit tests verify gravity spy on py27 and py34 py35 and py36. 

.. code-block:: bash

   $ pip install git+https://github.com/Gravity-Spy/GravitySpy.git

For more details see :ref:`install`.

Publications
------------

If you use Gravity Spy in your scientific publications or projects, we ask that you acknowlege our work by citing the publications that describe Gravity Spy.

* For general citations and information on Gravity Spy use the methods paper : `Zevin et al. Gravity Spy: Integrating Advanced LIGO Detector Characterization, Machine Learning, and Citizen Science <https://iopscience.iop.org/article/10.1088/1361-6382/aa5cea>`_

* `K. Crowston, & The Gravity Spy Collaboration. Gravity Spy: Humans, machines and the future of citizen science <https://citsci.syr.edu/sites/crowston.syr.edu/files/cpa137-crowstonA.pdf>`_

* `K. Crowston, C. Østerlund, T. Kyoung Lee. Blending machine and human learning processes <https://crowston.syr.edu/sites/crowston.syr.edu/files/training%20v3%20to%20share_0.pdf>`_ 

* `T. Kyoung Lee, K. Crowston, C. Østerlund, & G. Miller. Recruiting messages matter: Message strategies to attract citizen scientists <https://citsci.syr.edu/sites/crowston.syr.edu/files/cpa143-leeA.pdf>`_

* `S. Bahaadini, N. Rohani, S. Coughlin, M. Zevin, V. Kalogera, & A. Katsaggelos. Deep multi-view models for glitch classification <https://arxiv.org/pdf/1705.00034.pdf>`_

* For a thorough discussion of versionn 1.0 of the training set used see: `S. Bahaadini, V. Noroozi, N. Rohani, S. Coughlin, M. Zevin, J. R. Smith, V. Kalogera, & A. Katsaggelos. Machine learning for Gravity Spy: Glitch classification and dataset <https://www.sciencedirect.com/science/article/pii/S0020025518301634>`_

* C. Jackson, C. Østerlund, K. Crowston, M. Harandi, S. Allen, S. Bahaadini, S. Coughlin, V. Kalogera, A. Katsaggelos, S. Larson, N. Rohani, J. Smith, L. Trouille, and M. Zevin. [Making High-Performing Contributors: An Experiment With Training in an Online Production Community], submitted to IEEE Transactions on Learning Technologies, 2018.

* S. Bahaadini, V. Noroozi, N. Rohani, S. Coughlin, M. Zevin, & A. Katsaggelos. DIRECT: Deep DIscRiminative Embedding for ClusTering of LIGO Data, submitted to IEEE International Conference on Image Processing, 2018.

How to classify unlabeled excess noise
--------------------------------------

One of the main product of this package is the command-line executable `wscan`, which takes an excess noise time makes an omega scan of the event and classifies the image.

To run an analysis:

.. code-block:: bash

   $ wscan --inifile my-wini-config-file.ini --eventTime <eventTime> --outDir ./public_html/GravitySpy/Test/ --uniqueID --ID 123abc1234 --HDF5 --runML --pathToModel ../bin

where ``<eventtime>`` is the GPS time stamp assosciated with an Omicron trigger or other time of known excess noise, and ``./my-wini-config-file.ini`` is the path of your configuration file. In the folder Production, is an example of an ini file.

For a full list of command-line argument and options, run

.. code-block:: bash

   $ wscan --help

For more details see :ref:`wscan`.

How to train a new model
------------------------

Another main product of this package is the command-line executable `trainmodel`, which serves as an easy wrapper function for training a new Convolutional Neural Net Model (CNN). 

To train a model:

.. code-block:: bash

   $ THEANO_FLAGS=mode=FAST_RUN,device=cuda,floatX=float32 trainmodel --path-to-trainingset <path-to-trainingset> --number-of-classes <number-of-classes>

where ``<path-to-trainingset>`` is a folder that has the structure `"class"/"sampes"` and ``<number-of-classes>`` is how many different classes there are.

For a full list of command-line argument and options, run

.. code-block:: bash

   $ trainmodel --help

For more details see :ref:`trainmodel`.

The many databases of GravitySpy
--------------------------------

For more details see :ref:`DBs`

Package documentation
---------------------

Please consult these pages for more details on using GravitySpy:

.. toctree::
   :maxdepth: 1

   install/index
   wscan/index
   trainmodel/index
   examples/index
   DBs/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
