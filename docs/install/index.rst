.. _install:

############
Installation
############

Verified on py27 and py35 series.

=====
Conda
=====

.. code-block:: bash

   conda create --name gravityspy
   source activate gravityspy
   conda install pygpu
   pip install git+https://github.com/Gravity-Spy/GravitySpy.git

==========
virtualenv
==========

If you have used `macports` or `apt` before it is likely you have already
obtained `virtualenv <https://virtualenv.pypa.io/en/latest/>`_

.. code-block:: bash

    virtualenv-X.x ~/opt/GravitySpy-pyXx
    . ~/opt/GravitySpy-pyXx/bin/activate
    python -m pip install --upgrade --quiet pip setuptools
    pip install git+https://github.com/Gravity-Spy/GravitySpy.git


=======================
Pre-Existing VirtualEnv
=======================

On CIT, LLO, and LHO

.. code-block:: bash

   $ source ~gravityspy/.gravityspy_py36
   $ source ~gravityspy/.gravityspy_py27
