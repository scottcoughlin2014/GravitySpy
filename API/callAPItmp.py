from panoptes_client import *

import pandas as pd
import ast
import numpy as np
import os, sys
import ast
import pdb

from sqlalchemy.engine import create_engine


########################
####### Functions ######
#########################

def extract_choiceINT(x):
    try:
        x = label_dict[x]
    except:
        x = -1
    return x

# Function to aggregate data
def lister(x):
    return list(x)

def extract_all(x):
    s = Subject.find(x)
    try:
        y = str(s.raw['metadata']['subject_id'])
    except:
        y = False

    try:
        y1 = int(str(s.raw['links']['subject_sets'][0]))
    except:
        y1 = False

    # The ML posterior is a unicode string so some parsing needs 
    # to happen to make it a numpy array.
    try:
        y2 = ast.literal_eval(str(s.raw['metadata']['#ML_Posterior']))
        y3 = max(y2)
        y4 = np.argmax(y2)

    except:
        y2 = False
        y3 = False
        y4 = -1

    try:
        if 'Gold' == str(s.raw['metadata']['#Type']):
            y5 = 'G'
            y6 = label_dict[str(s.raw['metadata']['#Label']).upper().translate(None,'AEIOUY ()')]
            y2 = -2
            y3 = -2
            y4 = -2
    except:
        y5 = 'T'
        y6 = -1

    return [y,y1,y2,y3,y4,y5,y6]

# Define little funtions to extract the columns of subjectInfo
def get_image_id(x):
    return x[0]

def get_subject_set(x):
    return x[1]

def get_ML_posterior(x):
    return x[2]

def get_ML_label(x):
    return x[4]

def get_ML_confidence(x):
    return x[3]

def get_image_type(x):
    return x[5]

def get_image_label(x):
    return x[6]

def obtain_first_entry(x):
    return x[0]

# Open connection to mysql table
SQL_USER = os.environ['SQL_USER']
SQL_PASS = os.environ['SQL_PASS']
engine = create_engine('mysql://{0}:{1}@localhost/GravitySpy'.format(SQL_USER,SQL_PASS))

# Open classification table and extract most recent classificationID
classificationsInit = pd.read_sql('SELECT * FROM classifications',engine)
classificationsInit = classificationsInit[['subject_set','choiceINT','choice', 'userID','workflow','classificationID','zooID','classification_number','session','timeToClass','imageID','ML_posterior','ML_label','ML_confidence','type','true_label']]

if not np.isnan(classificationsInit.classificationID.max()):
    lastID = classificationsInit.classificationID.max()
else:
    lastID = "16822410"

#lastID = "15994276"
# Classification number by which the promotion was working well
#lastID = "16822410"

# Connect to panoptes and query all classifications done on project 1104 (i.e. GravitySpy)
Panoptes.connect()

try:
    allClassifications = Classification.where(scope='project',project_id='1104',last_id='{0}'.format(lastID),page_size='100')
except:
    ValueError("Query Classifications has failed")

# Initialize some parameters we need
# the userID of the person who classified
userID = []
# The zooniverse ID of the subject they classified
zooID  = []
# The choice the user made
choice = []
# The classificationID of this classification
classificationID = []
# What workflow was this classification done on
workflow = []
# What session was this classification done in
session = []
#How long did this classification take
time      = []

# Loop until no more classifications
for iN in range(0,100):
    try:
        classification = allClassifications.next()
    except:
        break

    try:
        classificationID.append(int(str(classification.id)))
    except:
        classificationID.append(False)

    try:
        userID.append(int(str(classification.links.raw['user'])))
    except:
        userID.append(False)

    try:
        zooID.append(int(str(classification.links.raw['subjects'][0])))
    except:
        zooID.append(False)

    try:
        choice.append(str(classification.raw['annotations'][0]['value'][0]['choice']))
    except:
        choice.append(False)

    try:
        workflow.append(int(str(classification.links.raw['workflow'])))
    except:
        workflow.append(False)

    try:
        session.append(str(classification.metadata['session']))
    except:
        session.append(False)

    try:
        time.append(str(float(str(classification.metadata['finished_at']).split('T')[1].split('Z')[0].replace(':','')) - float(str(classification.metadata['started_at']).split('T')[1].split('Z')[0].replace(':',''))))
    except:
        time.append(False)

if not classificationID:
    raise ValueError('No New Classifications')

# Now we want to make a panda data structure with all this information
classifications = pd.DataFrame({ 'classificationID' : classificationID ,'userID' : userID, 'zooID' : zooID, 'choice' : choice, 'workflow' : workflow, 'session' : session, 'timeToClass' : time})

# Drop rows where anything is False
classifications = classifications[classifications.choice !=False]
classifications = classifications[classifications.userID !=False]
classifications['classification_number'] = classifications.groupby('userID').cumcount() 

############
###########
# These choices are strings and we need to change them to integers in order to run the crowd sourcing classifer (CC for short). This is the thing that evaluates users and images.

label_dict = {'45MHZLGHTMDLTN':6,'LGHTMDLTN':6,'50HZ':0,'RCMPRSSR50HZ':0,'BLP':1,'CHRP':2,'XTRMLLD':3,'HLX':4,'KFSH':5,
              'LWFRQNCBRST':7,'LWFRQNCLN':8,'NGLTCH':10,'DNTSGLTCH':10,'NNFTHBV':9,'PRDDVS':11,'60HZPWRLN':12,'60HZPWRMNS':12,
              'PWRLN60HZ':12,'RPTNGBLPS':13,'SCTTRDLGHT':14,'SCRTCH':15,'TMT':16,'VLNHRMNC500HZ':17,'VLNMDHRMNC500HZ':17,
              'HRMNCS':17,'WNDRNGLN':18,'WHSTL':19}

classifications['choiceINT'] = classifications['choice'].apply(extract_choiceINT)

# Drop rows where anything is False
classifications = classifications[classifications.choiceINT !=-1]
# Must be a label from the new workflows not the old defunct apprentice workflow
classifications = classifications[classifications.workflow!=1479]


# Extract information about the subject the users have classified
classifications['subjectInfo']   = classifications.zooID.apply(extract_all)
classifications['imageID']       = classifications.subjectInfo.apply(get_image_id)
classifications['subject_set']   = classifications.subjectInfo.apply(get_subject_set)
classifications['ML_posterior']  = classifications.subjectInfo.apply(get_ML_posterior)
classifications['ML_label']      = classifications.subjectInfo.apply(get_ML_label)
classifications['ML_confidence'] = classifications.subjectInfo.apply(get_ML_confidence)
classifications['type']          = classifications.subjectInfo.apply(get_image_type)
classifications['true_label']    = classifications.subjectInfo.apply(get_image_label)

# All images must belong to a subject set
classifications = classifications[classifications.subject_set != False]

# All non gold standard images must have a ML_confidence
classifications = classifications[classifications.ML_confidence != False]

# Drop tuple of information
classifications = classifications.drop('subjectInfo',axis=1)

classifications.ML_posterior = classifications.ML_posterior.apply(str)

# Make sure entries that should be ints are
classifications.ML_label         = classifications.ML_label.apply(int)
classifications.choiceINT        = classifications.choiceINT.apply(int)
classifications.classificationID = classifications.classificationID.apply(int)
classifications.true_label       = classifications.true_label.apply(int)
classifications.workflow         = classifications.workflow.apply(int)
classifications.zooID            = classifications.zooID.apply(int)

# Need to recount classification numbers because we appended the tables
#classifications['classification_number'] = classifications.sort_values('classificationID').groupby('userID').cumcount()

# Put this information into a mysql table
try:
    classifications.to_sql(con=engine, name='classificationstmp', if_exists='replace', flavor='mysql')
    classifications.to_sql(con=engine, name='classifications', if_exists='append', flavor='mysql')
except:
    ValueError("Saving classifications failed!")
    #classificationsInit.to_sql(con=engine, name='classifications', if_exists='replace', flavor='mysql')

# Append this new classifications table to the old one
classifications = pd.concat([classificationsInit,classifications],ignore_index=True)

# Make sure a duplicate classification ID does not exist
classifications = classifications.drop_duplicates('classificationID')

############
###########
# At this point we have a nice classifications based data frame. Now we want to turn this into a subject based data frame. This is done below.

### Pivot dataframe to make index imageID and get choice, user_id, and workflow_version ###
image_values = ['subject_set','imageID','ML_posterior','ML_label','ML_confidence','type','true_label','choiceINT','choice', 'userID','workflow','classificationID','zooID','classification_number']

# The image based variable only needs a subset of the columns from the classifications variable
classifications = classifications[image_values]

images       = pd.pivot_table(classifications,index='zooID',values=image_values,aggfunc=lister)

# Some items were made lists that do not need to be
images.ML_confidence = images.ML_confidence.apply(obtain_first_entry)
images.ML_label      = images.ML_label.apply(obtain_first_entry)
images.ML_posterior  = images.ML_posterior.apply(obtain_first_entry)
images.imageID       = images.imageID.apply(obtain_first_entry)
images['type']       = images['type'].apply(obtain_first_entry)
images.true_label    = images.true_label.apply(obtain_first_entry)
images.subject_set   = images.subject_set.apply(obtain_first_entry)

# Recreate column with zooID (hard to apply to the index of a pandas array)
images['zooID']              = images.index.values
images.choice                = images.choice.apply(str)
images.choiceINT             = images.choiceINT.apply(str)
images.classificationID      = images.classificationID.apply(str)
images.userID                = images.userID.apply(str)
images.classification_number = images.classification_number.apply(str)
images.workflow              = images.workflow.apply(str)

# Save images variable to mySQL table
iI = 0
iII = range(0,len(images)-100,100)
for iIT in range(0,len(images)-100,100):
    if iIT == 0:
        images[0:iII[iI+1]].to_sql(con=engine, name='images', if_exists='replace', flavor='mysql',index=False)
    elif iIT != iII[-1]:
        images[iII[iI]:iII[iI+1]].to_sql(con=engine, name='images', if_exists='append', flavor='mysql',index=False)
    else:
        images[iII[iI]:len(images)].to_sql(con=engine, name='images', if_exists='append', flavor='mysql',index=False)
    iI = iI + 1


print(classifications.classificationID.max())
