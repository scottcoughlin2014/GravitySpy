"""Unit test for GravitySpy
"""

__author__ = 'Scott Coughlin <scott.coughlin@ligo.org>'

import os
from gravityspy.API.project import GravitySpyProject


PICKLE_PATH = os.path.join(os.path.split(__file__)[0], 'data',
'API', '1104.pkl')

workflowDictSubjectSets_unit = {'1610': {'Blip': ('1610', 6717, [1.0, 0.998]),
  'Whistle': ('1610', 6795, [1.0, 0.99])},
 '1934': {'Blip': ('1934', 6717, [1.0, 0.998]),
  'Koi_Fish': ('1934', 6731, [1.0, 0.98]),
  'Power_Line': ('1934', 6772, [1.0, 0.998]),
  'Violin_Mode': ('1934', 6902, [1.0, 0.99]),
  'Whistle': ('1934', 6795, [1.0, 0.99])},
 '1935': {'Blip': ('1935', 6717, [1.0, 0.998]),
  'Chirp': ('1935', 6721, [1.0, 0.7]),
  'Koi_Fish': ('1935', 6731, [1.0, 0.98]),
  'Low_Frequency_Burst': ('1935', 6755, [1.0, 0.99995]),
  'No_Glitch': ('1935', 6762, [1.0, 0.9901]),
  'Power_Line': ('1935', 6772, [1.0, 0.998]),
  'Scattered_Light': ('1935', 6779, [1.0, 0.99965]),
  'Violin_Mode': ('1935', 6902, [1.0, 0.99]),
  'Whistle': ('1935', 6795, [1.0, 0.99])},
 '2117': {'1080Lines': ('2117', 8194, [0.5, 0.0]),
  '1400Ripples': ('2117', 8192, [0.5, 0.0]),
  'Air_Compressor': ('2117', 6715, [0.6, 0.0]),
  'Blip': ('2117', 6719, [0.85, 0.0]),
  'Chirp': ('2117', 6723, [0.5, 0.0]),
  'Extremely_Loud': ('2117', 6726, [0.815, 0.0]),
  'Helix': ('2117', 6729, [0.5, 0.0]),
  'Koi_Fish': ('2117', 6733, [0.621, 0.0]),
  'Light_Modulation': ('2117', 6753, [0.9, 0.0]),
  'Low_Frequency_Burst': ('2117', 7070, [0.93, 0.0]),
  'Low_Frequency_Lines': ('2117', 6760, [0.65, 0.0]),
  'No_Glitch': ('2117', 6764, [0.85, 0.0]),
  'None_of_the_Above': ('2117', 6767, [0.5, 0.0]),
  'Paired_Doves': ('2117', 6770, [0.5, 0.0]),
  'Power_Line': ('2117', 6774, [0.86, 0.0]),
  'Repeating_Blips': ('2117', 6777, [0.686, 0.0]),
  'Scattered_Light': ('2117', 6781, [0.96, 0.0]),
  'Scratchy': ('2117', 6784, [0.913, 0.0]),
  'Tomte': ('2117', 6787, [0.7, 0.0]),
  'Violin_Mode': ('2117', 6790, [0.5, 0.0]),
  'Wandering_Line': ('2117', 6793, [0.97, 0.0]),
  'Whistle': ('2117', 6797, [0.6, 0.0])},
 '2360': {'1080Lines': ('2360', 8193, [1.0, 0.5]),
  '1400Ripples': ('2360', 8190, [1.0, 0.5]),
  'Air_Compressor': ('2360', 6714, [1.0, 0.6]),
  'Blip': ('2360', 6718, [0.998, 0.85]),
  'Chirp': ('2360', 6722, [0.7, 0.5]),
  'Extremely_Loud': ('2360', 6725, [1.0, 0.815]),
  'Helix': ('2360', 6728, [1.0, 0.5]),
  'Koi_Fish': ('2360', 6732, [0.98, 0.621]),
  'Light_Modulation': ('2360', 6752, [1.0, 0.9]),
  'Low_Frequency_Burst': ('2360', 7068, [0.99995, 0.93]),
  'Low_Frequency_Lines': ('2360', 6759, [1.0, 0.65]),
  'No_Glitch': ('2360', 6763, [0.9901, 0.85]),
  'None_of_the_Above': ('2360', 6766, [1.0, 0.5]),
  'Paired_Doves': ('2360', 6769, [1.0, 0.5]),
  'Power_Line': ('2360', 6773, [0.998, 0.86]),
  'Repeating_Blips': ('2360', 6776, [1.0, 0.686]),
  'Scattered_Light': ('2360', 6780, [0.99965, 0.96]),
  'Scratchy': ('2360', 6783, [1.0, 0.913]),
  'Tomte': ('2360', 6786, [1.0, 0.7]),
  'Violin_Mode': ('2360', 6789, [0.99, 0.5]),
  'Wandering_Line': ('2360', 6792, [1.0, 0.97]),
  'Whistle': ('2360', 6796, [0.99, 0.6])}}

classes_unit = ['1080Lines',
 '1400Ripples',
 'Air_Compressor',
 'Blip',
 'Chirp',
 'Extremely_Loud',
 'Helix',
 'Koi_Fish',
 'Light_Modulation',
 'Low_Frequency_Burst',
 'Low_Frequency_Lines',
 'No_Glitch',
 'None_of_the_Above',
 'Paired_Doves',
 'Power_Line',
 'Repeating_Blips',
 'Scattered_Light',
 'Scratchy',
 'Tomte',
 'Violin_Mode',
 'Wandering_Line',
 'Whistle']

class GravitSpyTests(object):
    """`TestCase` for the GravitySpy
    """
    def test_structure(self):
        workflowDictSubjectSets = GravitySpyProject.load_project_from_cache(PICKLE_PATH).get_level_structure(IDfilter='O2')
        assert workflowDictSubjectSets_unit == workflowDictSubjectSets
        classes = sorted(workflowDictSubjectSets['2117'].keys())
        assert classes_unit == classes
