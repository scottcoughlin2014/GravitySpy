import pandas as pd
import numpy as np
from sqlalchemy.engine import create_engine
import ast
import os
import pdb
from panoptes_client import *

# Load my sql tables of confusion matrices and pp_matrices
SQL_USER = os.environ['SQL_USER']
SQL_PASS = os.environ['SQL_PASS']
engine = create_engine('mysql://{0}:{1}@localhost/GravitySpy'.format(SQL_USER,SQL_PASS))

# Open classification table and extract most recent classificationID
images                 = pd.read_sql('SELECT * FROM images_for_pp',engine)
confusion_matrices     = pd.read_sql('SELECT * FROM confusion_matrices',engine)
currentStatus          = pd.read_sql('SELECT * FROM user_status',engine)

confusion_matrices1    = confusion_matrices.merge(currentStatus,how='left', left_on='userID', right_on='userID')

confusion_matrices1    = confusion_matrices1[confusion_matrices1.promoted_x != confusion_matrices1.promoted_y]

# Connect to panoptes and query all classifications done on project 1104 (i.e. GravitySpy)
Panoptes.connect()
project = Project.find(slug='zooniverse/gravity-spy')

workflow_dict = {"B1":1610, "B2":1934, "B3":1935,"A":2360,"M":2117}

# Determine who is promoted and change their workflow preference
def promote_users(x):
    user = User.find("{0}".format(x.userID))
    new_settings = {"workflow_id": "{0}".format(workflow_dict[x.promoted_x])}
    print(user)
    print(new_settings)
    ProjectPreferences.save_settings(project=project, user=user, settings=new_settings)

confusion_matrices1[confusion_matrices1.promoted_x != 'S'][['userID','promoted_x']].apply(promote_users,axis =1)

def move_image(x):
    # If has received enough labels or enough NOA form the beginner workflows that it has not been retired then we move it into the NOA Apprentice workflow for further analysis. We also remove it from whatever subject set it was in before.
    tmp = SubjectSet.find(5676)
    subject = Subject.find(x.zooID)
    # Add to new subject set
    print(subject)
    #tmp.add(subject)
    # Remove it form subject set it is currently in.
    tmp = SubjectSet.find(x.subject_set)
    print(tmp)
    #tmp.remove(subject)
    return

#images[images.decision == 2][['subject_set','zooID']].apply(move_image,axis=1)

confusion_matrices[['userID','promoted']].to_sql(con=engine, name='user_status', if_exists='replace', flavor='mysql')
