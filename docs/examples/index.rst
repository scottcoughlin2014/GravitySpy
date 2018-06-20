.. _examples:

#########################
Querying Gravity Spy data
#########################

============
Introduction
============

We will use `gwpy <https://gwpy.github.io/>`_, the preferred detchar software utility curated by Duncan Macleod.

The method of note is `gwpy.table.EventTable.fetch <https://gwpy.github.io/docs/latest/api/gwpy.table.EventTable.html#gwpy.table.EventTable.fetch>`_

The following example will although you to query the entire gravity spy trianingset. You do *not* need to be on CIT *but* if you do not have lal installed locally it is suggested that you do this on CIT.

Ahead of time, it is encourage to set up your user environment. For LIGO users please see `Gravity Spy Authentication <https://secrets.ligo.org/secrets/144/>`_ for information concerning authentication to access certain Gravity Spy DBs.

============
Installation
============

Very brief installation! Create a virtualenv on the clusters and `pip install`

.. code-block:: bash

   $ virtualenv --system-site-packages ~/LAAC/opt/
   $ . ~/LAAC/opt/bin/activate
   $ pip install --upgrade pip
   $ pip install gwpy sqlalchemy psycopg2 pandas git+https://github.com/duncanmmacleod/ligo.org.git

=======================
Pre-Existing VirtualEnv
=======================

On CIT, LLO and LHO

.. code-block:: bash

   $ source /home/gravityspy/.gravityspy_profile

=======================
The `glitches` database
=======================

.. code-block:: bash

   $ ipython

    >>> from gwpy.table import EventTable
    >>> blips_O1 = EventTable.fetch('gravityspy','glitches',selection='"Label"="Blip" & 1137250000 > "peakGPS" > 1126400000 & "ImageStatus" = "Retired"') 
    >>> koi_fish_O1 = EventTable.fetch('gravityspy','glitches',selection='"Label"="Koi_Fish" & 1137250000 > "peakGPS" > 1126400000 & "ImageStatus" = "Retired"') 
    >>> whistle_O1 = EventTable.fetch('gravityspy','glitches',selection='"Label"="Whistle" & 1137250000 > "peakGPS" > 1126400000 & "ImageStatus" = "Retired" & "ifo" = "L1"')
    >>> koi_fish_O1.write('O1_Koi_Fish.csv')
    >>> blips_O1.write('O1_Blips.csv')
    >>> whistle_O1["peakGPS","peak_frequency", "snr"].write('{0}-triggers-{1}-{2}.csv'.format()) 



=============================
Utilizing `gwpy` `EventTable`
=============================

There are so many great ways to use `EventTable <https://gwpy.github.io/docs/latest/api/gwpy.table.EventTable.html#gwpy.table.EventTable>`_ to make plotting
publication quality plots easy.

Here we mimic the `histogram <https://gwpy.github.io/docs/latest/examples/table/histogram.html?highlight=hist>`_ functionality

.. plot::
    :context: reset
    :include-source:

    >>> import matplotlib.pyplot as plt
    >>> plt.switch_backend('agg')
    >>> from gwpy.table import EventTable
    >>> blips_O2_L1 = EventTable.fetch('gravityspy', 'glitches', selection = ['"Label" = "Blip"', '"peakGPS" > 1137250000', '"Blip" > 0.95', 'ifo=L1'])
    >>> koi_O2_L1 = EventTable.fetch('gravityspy', 'glitches', selection = ['"Label" = "Koi_Fish"', '"peakGPS" > 1137250000', '"Koi_Fish" > 0.95', 'ifo=L1'])
    >>> aftercomiss_koi_l1 = koi_O2_L1[koi_O2_L1['peakGPS']>1178841618]
    >>> beforecomiss_koi_l1 = koi_O2_L1[koi_O2_L1['peakGPS']<1178841618]
    >>> beforecomiss_blips_l1 = blips_O2_L1[blips_O2_L1['peakGPS']<1178841618]
    >>> aftercomiss_blips_l1 = blips_O2_L1[blips_O2_L1['peakGPS']>1178841618]
    >>> plot = aftercomiss_blips_l1.hist('snr', logbins=True, bins=50, histtype='stepfilled', label='After Commissioning')
    >>> ax = plot.gca()
    >>> ax.hist(aftercomiss_koi_l1['snr'], logbins=True, bins=50, histtype='stepfilled', label='After Commissioning Koi')
    >>> ax.hist(beforecomiss_blips_l1['snr'], logbins=True, bins=50, histtype='stepfilled', label='Before Commissioning')
    >>> ax.hist(beforecomiss_koi_l1['snr'], logbins=True, bins=50, histtype='stepfilled', label='Before Commissioning Koi') 
    >>> ax.set_xlabel('Signal-to-noise ratio (SNR)')
    >>> ax.set_ylabel('Rate')                       
    >>> ax.set_title('Blips and Kois before and after comissioning L1')
    >>> ax.autoscale(axis='x', tight=True)                             
    >>> ax.set_xlim([0,1000])             
    >>> plot.add_legend()    

=====================================
Utilizing `hveto` (and soon Karoo GP)
=====================================

One thing you may think to do with this information is to feed them into tools designed to make statements about correlations/couplings with auxiliary channel data. hVeto is one such software.
`hVeto Documenation <https://ldas-jobs.ligo.caltech.edu/~duncan.macleod/hveto/latest/>`_

Go to LLO

.. code-block:: bash

    $ . /home/scoughlin/Project/opt/GravitySpy/bin/activate
    $ ligo-proxy-init albert.einstein@LIGO.ORG
    $ run_hveto.sh L1 Whistle 1126400000 1137250000 


Coming soon ...
`Karoo GP <http://kstaats.github.io/karoo_gp/>`_

==========================
The `trainingset` database
==========================

.. code-block:: bash

   $ ipython

    >>> from gwpy.table import EventTable
    >>> trainingset = EventTable.fetch('gravityspy','trainingsetv1d1')
    >>> trainingset.download(nproc=4, TrainingSet=1, LabelledSamples=1, download_path='TrainingSet')

================
Training a model 
================

.. code-block:: bash

    $ ipython
    >>> from gwpy.table import EventTable
    >>> from astropy.table import vstack
    >>> blips = EventTable.fetch('gravityspy', 'trainingsetv1d1', selection='Label=Blip')
    >>> whistle = EventTable.fetch('gravityspy', 'trainingsetv1d1', selection='Label=Whistle')
    >>> # Downselect to 100 samples (otherwise the download takes too long)
    >>> blips = blips[0:50]
    >>> whistle = whistle[0:50]
    >>> trainingset = vstack([whistle, blips])
    >>> trainingset.download(nproc=4, TrainingSet=1, download_path='TrainingSet')

The download script utilize `gwpy` very nice `Multi-Processing Tool <https://github.com/gwpy/gwpy/blob/develop/gwpy/utils/mp.py>`_. This tool is also currently being used to help speed up the creation of omega scans and turn things into a pythonic only `Omega Scan <https://github.com/scottcoughlin2014/PyOmega>`_. This Pythonic omega scan utilizes the gwpy implementation of `q_transform <https://gwpy.github.io/docs/latest/examples/timeseries/qscan.html?highlight=q_transform>`_ 

At this point we have a folder with Training data. Let's train a model using a GPU and some in house python software `Theano <http://www.deeplearning.net/software/theano/>`_ and `keras <https://keras.io/>`_

You can also use `tensorflow <https://www.tensorflow.org/>`_ backend when using `keras` you control this via the `KERAS_BACKEND` environmental variable.

LSC has some great hardware resources. Marco Cavaglia and Stuart Anderson have placed some useful hardware information (for LIGO members) here: `LIGO GPU Info <https://wiki.ligo.org/MLA/LV_computing_resources>`_

.. code-block:: bash

    $ THEANO_FLAGS=mode=FAST_RUN,device=cuda,floatX=float32 trainmodel --path-to-trainingset=./TrainingSet --number-of-classes=2 --nb-epoch=7 
