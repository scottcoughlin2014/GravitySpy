#!/usr/bin/env python

# ---- Import standard modules to the python path.

from panoptes_client import SubjectSet, Project, Workflow
from scipy.sparse import coo_matrix
from gwpy.table import EventTable

import re, pickle
import pandas as pd
import numpy as np

__all__ = ['ZooProject', 'flatten', 'GravitySpyProject',
           'workflow_with_most_answers']

# This function generically flatten a dict
def flatten(d, parent_key='', sep='_'):
    """Parameters
    ----------

    Returns
    -------
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        try:
            items.extend(flatten(v, new_key, sep=sep).items())
        except:
            items.append((new_key, v))
    return dict(items)


def workflow_with_most_answers(db):
    """Parameters
    ----------

    Returns
    -------
    """
    maxcount = max(len(v) for v in db.values())
    return [k for k, v in db.items() if len(v) == maxcount]


class ZooProject(object):
    '''
    `ZooProject` inherits from the Project class
    from the panoptes client and adds some
    wrapper functions to extracts helpful
    project information.
    '''
    def __init__(self, zoo_project_id, workflow_order=None):

        self.zoo_project_id = zoo_project_id
        tmp = Project.find(zoo_project_id)
        self.project_info = flatten(tmp.raw)

        # Determine workflow order
        self.workflow_info = {}
        if workflow_order is None:
            order = self.project_info['configuration_workflow_order']
        else:
            order = workflow_order

        workflows = [int(str(iWorkflow)) for iWorkflow in order]
        self.workflow_order = workflows

        # Save workflow information
        for iWorkflow in workflows:
            tmp1 = Workflow.find(iWorkflow)
            self.workflow_info[str(iWorkflow)] = flatten(tmp1.raw)


    def cache_project(self):
        """Parameters
        ----------

        Returns
        -------
        A dict with keys of workflow IDs and values list
        of golden sets associated with that workflow
        """
        output = open('{0}.pkl'.format(self.zoo_project_id), 'wb')
        # Pickle dictionary using protocol 0.
        pickle.dump(self, output, protocol=2)
        return

    @classmethod
    def load_project_from_cache(cls, cache_file):
        """Parameters
        ----------
        cache_file : `str` needs a '.pkl' extension

        Returns
        -------
        A dict with keys of workflow IDs and values list
        of golden sets associated with that workflow
        """
        with open('{0}'.format(cache_file), 'rb') as pickle_file:
            # Pickle dictionary using protocol 0.
            try:
                return pickle.load(pickle_file, encoding='latin1')
            except:
                return pickle.load(pickle_file)


    def get_golden_subject_sets(self):
        """Parameters
        ----------

        Returns
        -------
        A dict with keys of workflow IDs and values list
        of golden sets associated with that workflow
        """
        workflowGoldenSetDict = {}

        for iWorkflow in self.workflow_info.keys():
            try:
                workflowGoldenSetDict[iWorkflow] = \
                    self.workflow_info[iWorkflow]\
                        ['configuration_gold_standard_sets']
            except:
                # Workflow has no assosciated golden set
                workflowGoldenSetDict[iWorkflow] = []

        self.workflowGoldenSetDict = workflowGoldenSetDict
        return workflowGoldenSetDict


    def get_golden_images(self, workflow=None):
        """Parameters
        ----------

        Returns
        -------
        A dict with keys of workflow IDs and values list
        of golden sets associated with that workflow
        """
        workflowGoldenSetImagesDict = {}
        workflowGoldenSetDict = self.get_golden_subject_sets()

        if workflow:
            workflows = workflow
        else:
            workflows = workflowGoldenSetDict.keys()

        for iWorkflow in workflows:
            goldenImages = {}

            for iGoldenSubjectSet in workflowGoldenSetDict[iWorkflow]:
                tmp = SubjectSet.find(iGoldenSubjectSet)
                tmpSubjects = tmp.subjects

                while True:
                    try:
                        nextSubject = tmpSubjects.next()
                        goldenImages[str(nextSubject.id)] = \
                            [str(nextSubject.raw['metadata']['subject_id']),\
                             str(nextSubject.raw['metadata']['#Label']),
                             int(nextSubject.id)]
                    except:
                        break

            workflowGoldenSetImagesDict[iWorkflow] = goldenImages

        self.workflowGoldenSetImagesDict = workflowGoldenSetImagesDict
        return workflowGoldenSetImagesDict


    def get_answers(self, workflow=None):
        """Parameters
        ----------
        workflow : `int`, optional, default None

        Returns
        -------
        A dict with keys of workflow IDs and values list
        of golden sets associated with that workflow
        """
        # now determine infrastructure of workflows so we know what workflow
        # this image belongs in
        workflowDictAnswers = {}

        if workflow:
            workflows = [str(workflow)]
        else:
            workflows = self.workflow_info.keys()

        # Determine possible answers to the workflows
        for iWorkflow in workflows:
            answerDict = {}
            try:
                answers = self.workflow_info[iWorkflow]['tasks_T1_choicesOrder']
            except:
                answers = self.workflow_info[iWorkflow]['tasks_T0_choicesOrder']

            for answer in answers:
                answerDict[answer] = []
            workflowDictAnswers[iWorkflow] = answerDict

        self.workflowDictAnswers = workflowDictAnswers
        return workflowDictAnswers


    def get_subject_sets_per_workflow(self, workflow=None):
        """Parameters
        ----------
        workflow : `int`, optional, default None

        Returns
        -------
        A dict with keys of workflow IDs and values list
        of golden sets associated with that workflow
        """
        workflowDictSubjectSets = {}
        workflowGoldenSetDict = self.get_golden_subject_sets()

        if workflow is not None:
            workflows = [str(workflow)]
        else:
            workflows = self.workflow_info.keys()

        for iWorkflow in workflows:
            # check if golden set exists for workflow
            goldenset = workflowGoldenSetDict[iWorkflow]
            # Determine subject sets associated with this workflow
            subject_sets_for_workflow = self.workflow_info[iWorkflow]\
                                        ['links_subject_sets']
            subjectset_id = [int(str(iSubject)) \
                            for iSubject in subject_sets_for_workflow]
            subjectset_id = [iSubject for iSubject in subjectset_id\
                            if iSubject not in goldenset]
            workflowDictSubjectSets[iWorkflow] = subjectset_id

        self.workflowDictSubjectSets = workflowDictSubjectSets
        return workflowDictSubjectSets


class GravitySpyProject(ZooProject):
    '''
    `GravitySpyProject` inherits from the `ZooProject` class
    and adds some GravitySpy specific wrapper functions
    '''
    def get_level_structure(self, workflow=None, IDfilter=''):
        """Parameters
        ----------
        workflow : `int`, optional, default None
        IDfilter : `str`, optional, default ''

        Returns
        -------
        A dict with keys of workflow IDs and values list
        of golden sets associated with that workflow
        """
        if hasattr(self, 'level_structure'):
            return self.level_structure

        level_structure = {}
        workflowDictSubjectSets = \
            self.get_subject_sets_per_workflow(workflow=workflow)

        for iworkflow in workflowDictSubjectSets.keys():
            # If it is final workflow level 4 subject sets are also linked
            # so need to filter for level 5 subject sets
            if int(iworkflow) == 7767:
                subjectset_id = [iid for iid in \
                                workflowDictSubjectSets[iworkflow] \
                                if iid not in workflowDictSubjectSets['7766']]
            else:
                subjectset_id = workflowDictSubjectSets[iworkflow]

            # Determine Display names of subject set
            subjectset_displayname_id = {}
            for iSubjectSet in subjectset_id:
                tmp2 = SubjectSet.find(iSubjectSet)
                displayname = str(tmp2.raw['display_name'])
                if IDfilter in displayname:
                    str_tmp = displayname.split(" '")[0].replace(' ','_')
                    subjectset_displayname_id[str_tmp] = \
                        (iworkflow, iSubjectSet,
                            [float(iThres)
                                for iThres
                                    in re.findall("\d+\.\d+", displayname)
                            ]
                        )
            level_structure[iworkflow] = subjectset_displayname_id

        self.level_structure = level_structure
        return level_structure


    def calculate_confusion_matrices(self):
        """Parameters
        ----------

        Returns
        -------
        A dict with keys of workflow IDs and values list
        of golden sets associated with that workflow
        """
        # Load classifications, and golden images from DB
        # Make sure choice is a valid index
        # Make sure to evaluate only logged in users
        # Ignore NONEOFTHEABOVE classificatios when constructing confusion
        # matrix
        # Make sure to the subject classified was a golden image
        query = 'classificationsdev WHERE \"annotations_value_choiceINT\" != \
            -1 AND \"links_user\" != 0 AND \
            \"annotations_value_choiceINT\" != 12 AND \
            CAST(links_subjects AS FLOAT) IN \
            (SELECT \"links_subjects\" FROM goldenimages)'

        columns = ['id', 'links_user', 'links_subjects', 'links_workflow',
                   'annotations_value_choiceINT']
        classifications = EventTable.fetch('gravityspy', query,
                                           columns = columns)

        classifications = classifications.to_pandas()
        classifications = classifications.sort_values('id')
        golden_images = EventTable.fetch('gravityspy', 'goldenimages')
        golden_images_df = golden_images.to_pandas()

        # From answers Dict determine number of classes
        numClasses = len(self.get_answers(workflow=7766).values()[0])

        # merge the golden image DF with th classification (this merge is on
        # links_subject (i.e. the zooID of the image classified)
        image_and_classification = classifications.merge(golden_images_df,
                                                         on=['links_subjects'])

        # This is where the power of pandas comes in...on the fly in very quick
        # order we can fill all users confusion matrices
        # by smartly chosen groupby
        test = image_and_classification.groupby(['links_user',
                                                 'annotations_value_choiceINT',
                                                 'GoldLabel'])
        test = test.count().links_subjects.to_frame().reset_index()

        # Create "Sparse Matrices" and perform a normalization task on them.
        # Afterwards determine if the users diagonal
        # is above the threshold set above
        confusion_matrices = pd.DataFrame()
        for iUser in test.groupby('links_user'):
            columns = iUser[1].annotations_value_choiceINT
            rows = iUser[1]['GoldLabel']
            entry = iUser[1]['links_subjects']
            tmp = coo_matrix((entry, (rows,columns)), shape=(numClasses,
                                                             numClasses))
            conf_divided, a1, a2, a3 = \
                np.linalg.lstsq(np.diagflat(tmp.sum(axis=1)),
                                            tmp.todense())

            conf_dict = {'userID' : iUser[0], 'conf_matrix' : [conf_divided],
                  'alpha' : [np.diag(conf_divided)]}

            confusion_matrices = \
                confusion_matrices.append(pd.DataFrame(
                                                       conf_dict,
                                                       index=[iUser[0]]))

        self.confusion_matrices = confusion_matrices
        return confusion_matrices


    def calculate_confusion_matrices_per_classification(self):
        """Parameters
        ----------

        Returns
        -------
        A dict with keys of workflow IDs and values list
        of golden sets associated with that workflow
        """
        # Load classifications, and golden images from DB
        # Make sure choice is a valid index
        # Make sure to evaluate only logged in users
        # Ignore NONEOFTHEABOVE classificatios when constructing confusion
        # matrix
        # Make sure to the subject classified was a golden image
        query = 'classificationsdev WHERE \"annotations_value_choiceINT\" != \
            -1 AND \"links_user\" != 0 AND \
            \"annotations_value_choiceINT\" != 12 AND \
            CAST(links_subjects AS FLOAT) IN \
            (SELECT \"links_subjects\" FROM goldenimages)'

        columns = ['id', 'links_user', 'links_subjects', 'links_workflow',
                   'annotations_value_choiceINT']
        classifications = EventTable.fetch('gravityspy', query,
                                           columns = columns)

        classifications = classifications.to_pandas()
        classifications = classifications.sort_values('id')
        golden_images = EventTable.fetch('gravityspy', 'goldenimages')
        golden_images_df = golden_images.to_pandas()

        # From answers Dict determine number of classes
        numClasses = len(self.get_answers(workflow=7766).values()[0])

        # merge the golden image DF with th classification (this merge is on
        # links_subject (i.e. the zooID of the image classified)
        image_and_classification = classifications.merge(golden_images_df,
                                                         on=['links_subjects'])

        # groupby users to get there gold classifications
        tmp = image_and_classification.groupby('links_user')[['annotations_value_choiceINT','GoldLabel', 'id']]
        user_confusion_matrices = {}
        for key, item in tmp:
            user_confusion_matrices[key] = {}
            userlabels = tmp.get_group(key)
            rows = []
            cols = []
            entry = []
            for ilabel in userlabels.sort_values('id').itertuples():
                rows.append(ilabel[2])
                cols.append(ilabel[1])
                entry.append(1)
                user_confusion_matrices[key][ilabel[3]] = coo_matrix(
                (entry, (rows, cols)), shape=(numClasses,numClasses))
        return user_confusion_matrices


    def determine_level(self, alpha=None):
        """Parameters
        ----------
        alpha = 1xN vector of proficiency scores
        where N is the number of glitch classes

        Returns
        -------
        A dict with keys of workflow IDs and values list
        of golden sets associated with that workflow
        """
        answers = self.get_answers()
        answers_dict_rev =  dict(enumerate(sorted(answers['7766'].keys())))
        answers_dict = dict((str(v),k) for k,v in answers_dict_rev.items())
        # Determine what indices of the confusion matrix we evaluate for each level based on the answers for that level
        promotion_Level1 = set([answers_dict[answer] for answer in answers['1610'].keys() if answer not in ['NONEOFTHEABOVE']])
        promotion_Level2 = set([answers_dict[answer] for answer in answers['1934'].keys() if answer not in ['NONEOFTHEABOVE']])
        promotion_Level3 = set([answers_dict[answer] for answer in answers['1935'].keys() if answer not in ['NONEOFTHEABOVE']])
        promotion_Level4 = set([answers_dict[answer] for answer in answers['7765'].keys() if answer not in ['NONEOFTHEABOVE']])
        promotion_Level5 = set([answers_dict[answer] for answer in answers['7766'].keys() if answer not in ['NONEOFTHEABOVE']])

        
        level_dict = dict(enumerate(self.workflow_order))
        workflow_level_dict = dict((v, k + 1) for k, v in
                                   level_dict.items())

        promotions = list(zip(range(4, -1,-1), [promotion_Level5, promotion_Level4,
                                           promotion_Level3, promotion_Level2,
                                           promotion_Level1]))

        if alpha is None:
            numClasses = len(answers_dict.keys())
            alpha = .7*np.ones(numClasses)
            alpha[4] = 0.65
            alpha[9] = 0.65

        if not hasattr(self, 'confusion_matrices'):
            self.calculate_confusion_matrices()

        level = []
        for (iuser, ialpha) in zip(self.confusion_matrices.userID,
                                   self.confusion_matrices.alpha):

            proficiencyidx = set(np.where(ialpha > alpha)[0])
            # determine whether a user is proficient at >= number
            # of answers on a level. If yes, the check next level
            # until < at which point you know which level the user
            # should be on

            for ilevel, promo in promotions:
                if promo.issubset(proficiencyidx):
                    if ilevel == 4:
                        curr_level = ilevel + 3
                        curr_workflow = level_dict[ilevel + 2]
                    else:
                        curr_level = ilevel + 2
                        curr_workflow = level_dict[ilevel + 1]
                    break
                else:
                    curr_level = 1
                    curr_workflow = 1610

            level.append([curr_workflow, curr_level, iuser])

        columns = ['curr_workflow', 'curr_level', 'userID']
        return pd.DataFrame(level, columns = columns)


    def check_level_by_classification(self):
        # Obtain workflow order
        order = self.project_info['configuration_workflow_order']
        workflows = [int(str(iWorkflow)) for iWorkflow in order]
        levelWorkflowDict = dict(enumerate(workflows))
        workflowLevelDict = dict((v, k + 1) for k,v in levelWorkflowDict.iteritems())

        query = 'classificationsdev GROUP BY links_user, links_workflow'
        userlevels = EventTable.fetch('gravityspy', query,
                         columns = ['links_user', 'links_workflow'])

        userlevels = userlevels.to_pandas()
        userlevels['Level'] = userlevels.links_workflow.apply(
                                  lambda x: workflowLevelDict[x])

        init_user_levels = userlevels.groupby('links_user').Level.max()

        init_user_levels_dict = {'userID' : init_user_levels.index.tolist(),
                                'workflowInit' : init_user_levels.tolist()}

        userStatusInit = pd.DataFrame(init_user_levels_dict)
        self.userStatusInit = userStatusInit
        return userStatusInit
