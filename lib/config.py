import os

import numpy as np


class Config():

    def __init__(self):

        self.BASE_PATH = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
        self.DATA_PATH = os.path.abspath(os.path.join(self.BASE_PATH, 'data'))

        ## TODO
        self.MEAN = [0.73, 0.74, 0.55]
        self.STD = [0.01, 0.02, 0.02]

        ## TODO
        self.IMAGE_SIZE_WIDTH = 665
        self.IMAGE_SIZE_HEIGHT = 653

        self.HORIZONTAL_FLIPPING = False
        self.RANDOM_ROTATION = False
        self.COLOR_JITTERING = False
        self.RANDOM_CHANNEL_SWAPPING = True
        self.RANDOM_GAMMA = False
        self.RANDOM_GRAYSCALE = False
        self.RANDOM_RESOLUTION = False
