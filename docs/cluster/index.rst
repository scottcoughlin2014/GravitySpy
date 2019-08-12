.. _cluster:


#############################
Clustering Gravity Spy Events
#############################

============
Introduction
============

=========================
Cluster a Table Of Events
=========================

stuff

.. ipython::

    In [1]: from gravityspy.table import Events

    In [2]: virgo_images = Events.fetch('gravityspy', 'updated_similarity_index WHERE ifo = \'V1\'')

    In [3]: virgo_glitch_info = Events.fetch('gravityspy', 'glitches WHERE ifo = \'V1\'')

    In [4]: virgo_clusters = virgo_images.cluster(60)

    In [5]: print(virgo_clusters.to_pandas().clusters.value_counts())   

    In [6]: allinfo = Events.from_pandas(virgo_clusters.to_pandas().drop_duplicates(['links_subjects']).merge(virgo_glitch_info.to_pandas(),on=['uniqueID',  'ifo', 'links_subjects']))

    In [7]: allinfo['Label'] = allinfo['clusters']

    In [8]: allinfo['Label'] = allinfo['Label'].astype(str)
