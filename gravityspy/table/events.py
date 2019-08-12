# -*- coding: utf-8 -*-
# Copyright (C) Scott Coughlin (2017-)
#
# This file is part of gravityspy.
#
# gravityspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gravityspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gravityspy.  If not, see <http://www.gnu.org/licenses/>.

from gwtrigfind import find_trigger_files
from gwpy.segments import DataQualityFlag
from gwpy.table import GravitySpyTable
from gwpy.utils import mp as mp_utils
from sklearn.cluster import KMeans
from astropy.table import Column

from ..utils import log
from ..utils import utils
from ..api.project import GravitySpyProject
from ..ml.train_classifier import make_model

import panoptes_client
import numpy
import pandas
import subprocess
import string
import random
import os

class Events(GravitySpyTable):
    """This class provides method for classifying events with gravityspy
    """
    @classmethod
    def read(cls, *args, **kwargs):
        """Classify triggers in this table

        Parameters:
            `gwpy.table.GravitySpyTable`

        Returns:
            `Events` table
        """
        etg = kwargs.pop('etg', 'OMICRON')
        tab = super(Events, cls).read(*args, **kwargs)
        tab = tab.to_pandas()
        if 'gravityspy_id' not in tab.columns:
            tab['gravityspy_id'] = tab.apply(id_generator, 1)
            tab['image_status'] = 'testing'
            tab['data_quality'] =  'no_flag'
            tab['upload_flag'] = 0
            tab['citizen_score'] = 0.0
            tab['links_subjects'] = 0
            tab['url1'] = '' 
            tab['url2'] = '' 
            tab['url3'] = ''
            tab['url4'] = '' 

        if etg == 'OMICRON':
            tab['event_id'] = tab['event_id'].apply(int)
            tab['process_id'] = tab['process_id'].apply(int)

        tab = cls.from_pandas(tab)

        if etg == 'OMICRON':
            tab['event_time'] = (tab['peak_time'] +
                                 (0.000000001)*tab['peak_time_ns'])
            tab['event_time'].format = '%.9f'

        return tab

    @classmethod
    def fetch(cls, *args, **kwargs):
        tab = super(Events, cls).fetch(*args, **kwargs)
        return cls(tab)

    def classify(self, path_to_cnn, **kwargs):
        """Classify triggers in this table

        Parameters:

            path_to_cnn:
                file name of the CNN you would like to use

            **kwargs:
                nproc : number of parallel event times to be processing at once

        Returns:
            `Events` table
        """
        if 'event_time' not in self.keys():
            raise ValueError("This method only works if you have defined "
                             "a column event_time for your "
                             "Event Trigger Generator.")

        # Parse key word arguments
        config = kwargs.pop('config', utils.GravitySpyConfigFile())
        plot_directory = kwargs.pop('plot_directory', 'plots')
        timeseries = kwargs.pop('timeseries', None)
        source = kwargs.pop('source', None)
        channel_name = kwargs.pop('channel_name', None)
        frametype = kwargs.pop('frametype', None)
        verbose = kwargs.pop('verbose', False)
        # calculate maximum number of processes
        nproc = kwargs.pop('nproc', 1)

        # make a list of event times
        inputs = zip(self['event_time'], self['ifo'],
                     self['gravityspy_id'])

        inputs = ((etime, ifo, gid, config, plot_directory,
                   timeseries, source, channel_name, frametype, nproc, verbose)
                  for etime, ifo, gid in inputs)

        # make q_scans
        output = mp_utils.multiprocess_with_queues(nproc,
                                                   _make_single_qscan,
                                                   inputs)

        qvalues = []
        # raise exceptions (from multiprocessing, single process raises inline)
        for f, x in output:
            if isinstance(x, Exception):
                x.args = ('Failed to make q scan at time %s: %s' % (f,
                                                                    str(x)),)
                raise x
            else:
                qvalues.append(x)

        self['q_value'] = qvalues

        results = utils.label_q_scans(plot_directory=plot_directory,
                                      path_to_cnn=path_to_cnn,
                                      verbose=verbose,
                                      **kwargs)

        results = results.to_pandas()
        results['Filename1'] = results['Filename1'].apply(lambda x, y : os.path.join(y, x),
                                                          args=(plot_directory,))
        results['Filename2'] = results['Filename2'].apply(lambda x, y : os.path.join(y, x),
                                                          args=(plot_directory,))
        results['Filename3'] = results['Filename3'].apply(lambda x, y : os.path.join(y, x),
                                                          args=(plot_directory,))
        results['Filename4'] = results['Filename4'].apply(lambda x, y : os.path.join(y, x),
                                                          args=(plot_directory,))


        results = Events.from_pandas(results.merge(self.to_pandas(),
                                                   on=['gravityspy_id']))
        return results

    def to_sql(self, table='glitches_v2d0', engine=None, **kwargs):
        """Obtain omicron triggers to run gravityspy on

        Parameters:
            table (str): name of SQL table
        """
        from sqlalchemy.engine import create_engine
        # connect if needed
        if engine is None:
            conn_kw = {}
            for key in ('db', 'host', 'user', 'passwd', 'server', 'port'):
                try:
                    conn_kw[key] = kwargs.pop(key)
                except KeyError:
                    pass
            engine = create_engine(get_connection_str(**conn_kw))

        self.to_pandas().to_sql(table, engine, index=False, if_exists='append')
        engine.dispose()
        return

    def to_glitch_db(self, table='GSMetadata', engine=None, **kwargs):
        """Obtain omicron triggers to run gravityspy on

        Parameters:
            table (str): name of SQL table
        """
        from sqlalchemy.engine import create_engine

        tab = self.to_pandas()
        def makelink(x):
            # This horrendous thing obtains the public html path for image
            intermediate_path = '/'.join(filter(None,str(x.Filename1).split('/'))[3:-1])
            if x.ifo == 'L1':
                return 'https://ldas-jobs.ligo-la.caltech.edu/~gravityspy/{0}/{1}.png'.format(intermediate_path,x.gravityspy_id)
            elif x.ifo == 'V1':
                return 'https://ldas-jobs.ligo.caltech.edu/~gravityspy/{0}/{1}.png'.format(intermediate_path, x.gravityspy_id)
            else:
                return 'https://ldas-jobs.ligo-wa.caltech.edu/~gravityspy/{0}/{1}.png'.format(intermediate_path, x.gravityspy_id)

        tab['imgUrl'] = tab[['ifo', 'gravityspy_id', 'Filename1']].apply(makelink, axis=1)
        tab['pipeline'] = 'GravitySpy'
        tab['flag'] = 0
        # Get the right columns for glitch db from all the available colums
        tab = tab[['gravityspy_id','ifo','ml_label', 'imgUrl', 'snr', 'amplitude','peak_frequency','central_freq','duration','bandwidth','chisq','chisq_dof','event_time','ml_confidence', 'image_status', 'pipeline', 'citizen_score', 'flag', 'data_quality', 'q_value']]
        tab.columns = ['id', 'ifo', 'label', 'imgUrl', 'snr', 'amplitude', 'peakFreq', 'centralFreq', 'duration', 'bandwidth', 'chisq', 'chisqDof', 'GPStime','confidence', 'imageStatus', 'pipeline', 'citizenScore', 'flag', 'dqflag', 'qvalue']

        # connect if needed
        if engine is None:
            conn_kw = {}
            for key in ('db', 'host', 'user', 'passwd', 'server', 'port'):
                try:
                    conn_kw[key] = kwargs.pop(key)
                except KeyError:
                    pass
            engine = create_engine(get_connection_str(**conn_kw))

        tab.to_sql(con=engine, name=table, if_exists='append',
                   index=False, chunksize=100)
        return

    def update_sql(self, table='glitches_v2d0', engine=None, **kwargs):
        """Obtain omicron triggers to run gravityspy on

        Parameters:
            table (str): name of SQL tabl
        """
        from sqlalchemy.engine import create_engine
        # connect if needed
        if engine is None:
            conn_kw = {}
            for key in ('db', 'host', 'user', 'passwd'):
                try:
                    conn_kw[key] = kwargs.pop(key)
                except KeyError:
                    pass
            engine = create_engine(get_connection_str(**conn_kw))

        ientry = self.to_pandas().to_dict(orient='records')
        for column_dict in ientry:
            sql_command = 'UPDATE {0} SET '.format(table)
            for column_name in column_dict:
                if isinstance(column_dict[column_name], str):
                    sql_command = sql_command + '''\"{0}\" = \'{1}\', '''.format(column_name, column_dict[column_name])
                else:
                    sql_command = sql_command + '''\"{0}\" = {1}, '''.format(column_name, column_dict[column_name])
            sql_command = sql_command[:-2] + ' WHERE \"gravityspy_id\" = \'' + column_dict['gravityspy_id'] + "'"
            engine.execute(sql_command)
        return

    def update_ldvw(self, table='GSMetadata', engine=None, **kwargs):
        """Obtain omicron triggers to run gravityspy on

        Parameters:
            table (str): name of SQL tabl
        """
        from sqlalchemy.engine import create_engine
        # connect if needed
        if engine is None:
            conn_kw = {}
            for key in ('db', 'host', 'user', 'passwd'):
                try:
                    conn_kw[key] = kwargs.pop(key)
                except KeyError:
                    pass
            engine = create_engine(get_connection_str(**conn_kw))

        tab = self.to_pandas()
        # Get the right columns for glitch db from all the available colums
        tab = tab[['gravityspy_id', 'ml_label', 'ml_confidence']]
        tab.columns = ['id', 'label', 'confidence']

        ientry = tab.to_dict(orient='records')
        for column_dict in ientry:
            sql_command = 'UPDATE {0} SET '.format(table)
            for column_name in column_dict:
                if column_name == 'id':
                    continue
                if isinstance(column_dict[column_name], str):
                    sql_command = sql_command + '''{0} = \'{1}\', '''.format(column_name, column_dict[column_name])
                else:
                    sql_command = sql_command + '''{0} = {1}, '''.format(column_name, column_dict[column_name])
            sql_command = sql_command[:-2] + ' WHERE id = \'' + column_dict['id'] + "'"
            engine.execute(sql_command)
        return

    def upload_to_zooniverse(self, subject_set_id=None, project='1104'):
        """Obtain omicron triggers to run gravityspy on

        Parameters:
            subject_set_id (optional, int) : subject set id to upload to

        Returns:
            `Events` table
        """
        for col in self.itercols():
            if col.dtype.kind in 'SU':
                self.replace_column(col.name, col.astype('object'))
        # First filter out images that have already been uploaded
        tab = self[self['upload_flag'] != 1]
        # If you want a specific subject set to be uploaded to then the
        # subjectset column will be updated to reflect which one you want
        if subject_set_id is not None:
            tab['subjectset'] = subject_set_id

        if subject_set_id is None:
            subset_ids = numpy.unique(numpy.array(tab['subjectset'])) 
        else:
            subset_ids = numpy.atleast_1d(numpy.array(subject_set_id))

        for subset_id in subset_ids:
            subjectset = panoptes_client.SubjectSet.find(subset_id)
            subjects = []

            tab1 = tab[tab['subjectset'] == subset_id]

            for fn1, fn2, fn3, fn4, gid, qval in tab1['Filename1', 'Filename2', 'Filename3', 'Filename4', 'gravityspy_id', 'q_value']:
                subject = panoptes_client.Subject()
                subject.links.project = project
                subject.add_location(str(fn1))
                subject.add_location(str(fn2))
                subject.add_location(str(fn3))
                subject.add_location(str(fn4))
                subject.metadata['date'] = '20180825'
                subject.metadata['subject_id'] = str(gid)
                subject.metadata['Filename1'] = fn1.split('/')[-1]
                subject.metadata['Filename2'] = fn2.split('/')[-1]
                subject.metadata['Filename3'] = fn3.split('/')[-1]
                subject.metadata['Filename4'] = fn4.split('/')[-1]
                subject.metadata['q_value'] = str(qval)
                subject.save()
                subjects.append(subject)
                self['links_subjects'][self['gravityspy_id'] == gid] = int(subject.id)
                self['url1'][self['gravityspy_id'] == gid] = subject.raw['locations'][0]['image/png'].split('?')[0]
                self['url2'][self['gravityspy_id'] == gid] = subject.raw['locations'][1]['image/png'].split('?')[0]
                self['url3'][self['gravityspy_id'] == gid] = subject.raw['locations'][2]['image/png'].split('?')[0]
                self['url4'][self['gravityspy_id'] == gid] = subject.raw['locations'][3]['image/png'].split('?')[0]
                self['upload_flag'][self['gravityspy_id'] == gid] = 1
            subjectset.add(subjects)

        return self

    def update_scores(self, path_to_cnn, nproc=1, **kwargs):
        """Obtain omicron triggers to run gravityspy on

        Parameters:
            path_to_cnn (str): filename of model

        Returns:
            `Events` table with columns containing new scores
        """
        if not all(elem in self.keys() for elem in ['Filename1', 'Filename2',
                                                    'Filename3', 'Filename4']):
            raise ValueError("This method only works if the file paths "
                             "of the images of the images are known.")

        results = utils.label_select_images(filename1=self['Filename1'],
                                            filename2=self['Filename2'],
                                            filename3=self['Filename3'],
                                            filename4=self['Filename4'],
                                            path_to_cnn=path_to_cnn, **kwargs)



        return Events(results)

    def update_features(self, path_to_semantic_model, nproc=1, **kwargs):
        """Obtain omicron triggers to run gravityspy on

        Parameters:
            path_to_semantic_model (str): filename of model

        Returns:
            `Events` table with columns containing new scores
        """
        if not all(elem in self.keys() for elem in ['Filename1', 'Filename2',
                                                    'Filename3', 'Filename4']):
            raise ValueError("This method only works if the file paths "
                             "of the images of the images are known.")

        results = utils.get_features_select_images(filename1=self['Filename1'],
                                            filename2=self['Filename2'],
                                            filename3=self['Filename3'],
                                            filename4=self['Filename4'],
                                            path_to_semantic_model=path_to_semantic_model, **kwargs)

        return Events(results)

    def determine_workflow_and_subjectset(self, project_info_pickle):
        """Obtain omicron triggers to run gravityspy on

        Parameters:
            path_to_cnn (str): filename of file with Gravity Spy project info

        Returns:
            `Events` table with columns workflow and subjectset
        """
        if 'ml_confidence' not in self.keys() or 'ml_label' not in self.keys():
            raise ValueError("This method only works if the confidence and label "
                             "of the image in known.")
        gspyproject = GravitySpyProject.load_project_from_cache(
                                                                project_info_pickle
                                                                )

        workflows_for_each_class = gspyproject.get_level_structure(IDfilter='O2')
        # Determine subject set and workflow this should go to.
        level_of_images = []
        subjectset_of_images = []
        for label, confidence in zip(self['ml_label'], self['ml_confidence']):
            for iworkflow in ['1610', '1934', '1935', '7765', '7766', '7767']:
                if label in workflows_for_each_class[iworkflow].keys():
                     if workflows_for_each_class[iworkflow][label][2][1] <= \
                            confidence <= \
                                 workflows_for_each_class[iworkflow][label][2][0]:
                         level_of_images.append(int(workflows_for_each_class[iworkflow][label][0]))
                         subjectset_of_images.append(workflows_for_each_class[iworkflow][label][1])
                         break

        self["workflow"] = level_of_images
        self["subjectset"] = subjectset_of_images

        return self

    def create_collection(self, name=None, private=True,
                          default_subject=None):
        """Obtain omicron triggers to run gravityspy on

        Parameters:
            name (str, optional):
                name of collection
            private (bool, optional):
                would you like this collection to be private or public
            default_subject (int, optional):
                subject id to be the cover image of collection

        Returns:
            `str` url link to the created collection
        """
        if name is None:
            # will name it after the label of event table
            name = self['Label'][0]

        if default_subject is None:
            default_subject = self['links_subjects'][0]

        collection_url = ('https://www.zooniverse.org/'
                          'projects/zooniverse/gravity-spy/collections/')

        with panoptes_client.Panoptes() as client:
            client.connect()

            collection = panoptes_client.Collection()
            collection.links.project = '1104'
            collection.display_name = '{0}'.format(name)
            collection.private = private
            urltmp = collection.save()
            collection_url = collection_url + urltmp['collections'][0]['slug']
            collection.add(list(self['links_subjects']))
            collection.set_default_subject(default_subject)

        return collection_url

    def cluster(self, nclusters, random_state=30):
        """Create new clusters from feature space vectors

        Parameters:

            nclusters (int): how many clusters to try to group
                these triggers into

        Returns:
            `Events` table
        """
        if '0' not in self.columns:
            raise ValueError("You are trying to cluster but you do not have "
                             "the feature space information in this table.")

        features = self.to_pandas().values[:, 0:200]
        kmeans_1 = KMeans(nclusters, random_state=random_state).fit(features)
        clusters = kmeans_1.labels_
        self['clusters'] = clusters

        return self

    @classmethod
    def get_triggers(cls, start, end, channel,
                     dqflag, verbose=True, **kwargs):
        """Obtain omicron triggers to run gravityspy on

        Parameters:

            start (int): start of time to look for triggers
            end (int): end time to look for triggers
            channel (str): channel to look for triggers
            dqflag (str): name of segment during which to keep triggers

        Returns:
            `Events` table
        """
        duration_max = kwargs.pop('duration_max', None)
        duration_min = kwargs.pop('duration_min', None)
        frequency_max = kwargs.pop('frequency_max', 2048)
        frequency_min = kwargs.pop('frequency_min', 10)
        snr_max = kwargs.pop('snr_max', None)
        snr_min = kwargs.pop('snr_min', 7.5)

        detector = channel.split(':')[0]

        logger = log.Logger('Gravity Spy: Fetching Omicron Triggers')

        # Obtain segments that are analysis ready
        analysis_ready = DataQualityFlag.query('{0}:{1}'.format(detector,
                                                                dqflag),
                                              float(start), float(end))

        # Display segments for which this flag is true
        logger.info("Segments for which the {0} Flag "
                    "is active: {1}".format(dqflag, analysis_ready.active))

        # get Omicron triggers
        files = find_trigger_files(channel,'Omicron',
                                   float(start),float(end))

        triggers = cls.read(files, tablename='sngl_burst', format='ligolw')

        logger.info("Number of triggers "
                    "before any filtering: {0}".format(len(triggers)))

        masks = numpy.ones(len(triggers), dtype=bool)

        logger.info("duration filter "
                    "[{0}, {1}]".format(duration_min, duration_max))

        logger.info("frequency filter "
                    "[{0}, {1}]".format(frequency_min, frequency_max))

        logger.info("snr filter "
                    "[{0}, {1}]".format(snr_min, snr_max))

        if not duration_max is None:
            masks &= (triggers['duration'] <= duration_max)
        if not duration_min is None:
            masks &= (triggers['duration'] >= duration_min)
        if not frequency_max is None:
            masks &= (triggers['peak_frequency'] <= frequency_max)
        if not frequency_min is None:
            masks &= (triggers['peak_frequency'] >= frequency_min)
        if not snr_max is None:
            masks &= (triggers['snr'] <= snr_max)
        if not snr_min is None:
            masks &= (triggers['snr'] >= snr_min)

        triggers = triggers[masks]
        # Set peakGPS

        logger.info("Number of triggers after "
                    "snr, frequency, and duration filters "
                    "cuts but before {0} flag filtering: "
                    "{1}".format(dqflag, len(triggers)))

        # Filter the raw omicron triggers against the ANALYSIS READY flag.
        vetoed = triggers['event_time'].in_segmentlist(analysis_ready.active)
        triggers = triggers[vetoed]

        logger.info("Final trigger length: {0}".format(len(triggers)))

        return triggers

    def create_sub(self, channel_name, frame_type,
                   path_to_cnn, plot_directory,
                   delete_images=False,
                   subfile_name='gravityspy.sub',
                   accounting_group_user='scott.coughlin',
                   accounting_group='ligo.dev.o1.detchar.ch_categorization.glitchzoo'):
        """Create a dag file to process all events in table

        Parameters:

            nclusters (int): how many clusters to try to group
                these triggers into

        Returns:
            `Events` table
        """
        # Determine location of executables
        proc = subprocess.Popen(['which', 'wscan'],
                                stdin = subprocess.PIPE,
                                stdout = subprocess.PIPE,
                                stderr = subprocess.PIPE
                                )
        (path_to_wscan, err) = proc.communicate()
        if not path_to_wscan:
            raise ValueError('Cannot locate wscan executable in your path')

        for d in ['logs', 'condor']:
            if not os.path.isdir(d):
                os.makedirs(d)

        with open(subfile_name, 'w') as subfile:
            subfile.write('universe = vanilla\n')
            subfile.write('executable = {0}\n'.format(str(path_to_wscan)))
            subfile.write('\n')
            if delete_images:
                subfile.write('arguments = "--channel-name {0} '
                              '--frametype {1} '
                              '--event-time $(event_time) '
                              '--plot-directory {2} --hdf5 '
                              '--path-to-cnn-model {3} '
                              '--delete-images"\n'.format(channel_name,
                                                          frame_type,
                                                          plot_directory,
                                                          path_to_cnn))
            else:
                subfile.write('arguments = "--channel-name {0} '
                              '--frametype {1} '
                              '--event-time $(event_time) '
                              '--plot-directory {2} --hdf5 '
                              '--path-to-cnn-model {3}"\n'.format(channel_name,
                                                                frame_type,
                                                                plot_directory,
                                                                path_to_cnn))
            subfile.write('getEnv=True\n')
            subfile.write('\n')
            subfile.write('accounting_group_user = {0}\n'.format(accounting_group_user))
            subfile.write('accounting_group = {0}\n'.format(accounting_group))
            subfile.write('\n')
            subfile.write('priority = 0\n')
            subfile.write('request_memory = 5000\n')
            subfile.write('\n')
            subfile.write('error = logs/gravityspy-$(jobNumber).err\n')
            subfile.write('output = logs/gravityspy-$(jobNumber).out\n')
            subfile.write('notification = Error\n')
            subfile.write('queue 1')
            subfile.close()

    def create_dag(self, subfile_name='gravityspy.sub', retry_number=3):
        """Create a dag file to process all events in table

        Parameters:

            nclusters (int): how many clusters to try to group
                these triggers into

        Returns:
            `Events` table
        """
        with open('gravityspy_{0}_{1}.dag'.format(self['peak_time'].min(),
                                                  self['peak_time'].max()),'a+') as dagfile:
            for peak_time, peak_time_ns, event_id, event_time in zip(self['peak_time'],
                                                                     self['peak_time_ns'],
                                                                     self['event_id'],
                                                                     self['event_time']):
                job_number = '{0}{1}{2}'.format(peak_time, peak_time_ns, event_id)
                dagfile.write('JOB {0} {1}\n'.format(job_number, subfile_name))
                dagfile.write('RETRY {0} {1}\n'.format(job_number, retry_number))
                dagfile.write('VARS {0} jobNumber="{0}" event_time="{1}"'.format(job_number, repr(event_time)))
                dagfile.write('\n\n')

def id_generator(x, size=10,
                 chars=(string.ascii_uppercase +
                        string.digits +
                        string.ascii_lowercase)):
    """Obtain omicron triggers run gravityspy on

    Parameters:

        x (str): the item you would like a random id to be generated for
    Returns:
    """
    return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

def get_connection_str(db='gravityspy',
                       host='gravityspy.ciera.northwestern.edu',
                       port='5432',
                       server='postgresql',
                       user=None,
                       passwd=None):
    """Create string to pass to create_engine
    """
    if (not user) or (not passwd):
        user = os.getenv('GRAVITYSPY_DATABASE_USER', None)
        passwd = os.getenv('GRAVITYSPY_DATABASE_PASSWD', None)

    if (not user) or (not passwd):
        raise ValueError('Remember to either pass '
                         'or export GRAVITYSPY_DATABASE_USER '
                         'and export GRAVITYSPY_DATABASE_PASSWD in order '
                         'to access the Gravity Spy Data: '
                         'https://secrets.ligo.org/secrets/144/'
                         ' description is username and secret is password.')

    return '{0}://{1}:{2}@{3}:{4}/{5}'.format(server, user, passwd,
                                              host, port, db)

# define multiprocessing method
def _make_single_qscan(inputs):
    event_time = inputs[0]
    ifo = inputs[1]
    gid = inputs[2]
    config = inputs[3]
    plot_directory = inputs[4]
    timeseries = inputs[5]
    source = inputs[6]
    channel_name = inputs[7]
    frametype = inputs[8]
    nproc = inputs[9]
    verbose = inputs[10]

    # Parse Ini File
    plot_time_ranges = config.plot_time_ranges
    plot_normalized_energy_range = config.plot_normalized_energy_range
    try:
        if timeseries is not None:
            specsgrams, q_value = utils.make_q_scans(event_time=event_time,
                                                     config=config,
                                                     timeseries=timeseries,
                                                     verbose=verbose)
        if source is not None:
            specsgrams, q_value = utils.make_q_scans(event_time=event_time,
                                                     config=config,
                                                     source=source,
                                                     verbose=verbose)
        if channel_name is not None:
            specsgrams, q_value = utils.make_q_scans(event_time=event_time,
                                                     config=config,
                                                     channel_name=channel_name,
                                                     frametype=frametype,
                                                     verbose=verbose)
        utils.save_q_scans(plot_directory, specsgrams,
                           plot_normalized_energy_range, plot_time_ranges,
                           ifo, event_time, id_string=gid, verbose=verbose)

        return event_time, q_value
    except Exception as exc:  # pylint: disable=broad-except
        if nproc == 1:
            raise
        else:
            return event_time, exc
