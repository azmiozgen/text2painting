import os

import numpy as np


class DataSettings(object):

    def __init__(self):

        self.BASE_PATH = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
        self.DATA_PATH = os.path.abspath(os.path.join(self.BASE_PATH, 'data', 'ara-lat_real'))
        self.LABELS = np.loadtxt(os.path.join(self.BASE_PATH, 'data', 'metadata', 'labels.txt'), dtype='str', delimiter=',')
        self.N_CLASSES = len(self.LABELS)
