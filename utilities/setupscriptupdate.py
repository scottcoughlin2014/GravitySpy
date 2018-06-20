import optparse,os,string,random,pdb,socket,subprocess
import pandas as pd
import numpy as np
from sqlalchemy.engine import create_engine


def write_dagfile(x):
    with open('gravityspy_update_scores.dag','a+') as dagfile:
        dagfile.write('JOB {0} ./condor/gravityspy.sub\n'.format(x.indexer))
        dagfile.write('RETRY {0} 2\n'.format(x.indexer))
        dagfile.write('VARS {0} jobNumber="{0}" Filename1="{1}" '
                      'Filename2="{2}" Filename3="{3}" '
                      'Filename4="{4}"'.format(
                      x.indexer, x.Filename1, x.Filename2, x.Filename3,
                      x.Filename4))
        dagfile.write('\n\n')


engine = create_engine('postgresql://{0}:{1}@gravityspy.ciera.northwestern.edu:5432/gravityspy'.format(os.environ['QUEST_SQL_USER'],os.environ['QUEST_SQL_PASSWORD']))

# Load classifications, current user DB status and golden images from DB
glitches = pd.read_sql('SELECT "Filename1", "Filename2", "Filename3", "Filename4" FROM glitches WHERE ifo = \'V1\'', engine)
glitches = glitches.loc[~glitches.Filename1.isnull()]
glitches['indexer'] = glitches.index
glitches.apply(write_dagfile, axis=1)
