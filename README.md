# GravitySpy
This repo contains code for the GravitySpy Citizen Science project.

The gravityspy project contains three components. The code to produce the images of the glitches that users will evalutate. The code that will evaluate user performance and image retireability colloquially known as the Crowd Sourcing code. Fianlly, the code that determines the Machine Learning labelling of an image, speifically a set of values indicating the classifiers confidence that it belongs in a given category.

# Installing Gravity Spy on LIGO Data Grid (LDG) Clusters
## Create virtualEnv
* ./GravitySpy-init $ENVDIR requirements.txt
* pip install git+https://github.com/Gravity-Spy/GravitySpy.git

## Running Example
* . $ENVDIR/bin/activate
* which wscan
* in examples directory run the commands in wscan.param. The resulting label should score should be 0.964064

# ER10/O2 Gravity Spy

# Livingston
* Start Time of ER10: 1161907217
* O1 Start time = 1126400000 September 15, 2015 -- End time = 1137250000 January 18, 2016
# Hanford
* Start Time of ER10: 1163203217
* O1: Start time = 1126400000 September 15, 2015 -- End time = 1137250000 January 18, 2016


[![Build Status](https://travis-ci.org/Gravity-Spy/GravitySpy.svg?branch=master)](https://travis-ci.org/Gravity-Spy/GravitySpy)
[![Coverage Status](https://coveralls.io/repos/github/Gravity-Spy/GravitySpy/badge.svg?branch=master)](https://coveralls.io/github/Gravity-Spy/GravitySpy?branch=master)
