import os

import numpy as np


class Config():

    def __init__(self):

        self.BASE_PATH = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
        self.DATA_PATH = os.path.join(self.BASE_PATH, 'data')
        self.MODEL_PATH = os.path.join(self.BASE_PATH, 'models')
        self.WORD2VEC_MODEL_FILE = os.path.join(self.MODEL_PATH, 'united_word2vec.model')

        self.MEAN = [0.7402, 0.5459, 0.3712]
        self.STD = [0.0134, 0.0180, 0.0185]

        ## TODO
        self.IMAGE_SIZE_WIDTH = 64
        self.IMAGE_SIZE_HEIGHT = 64

        self.HORIZONTAL_FLIPPING = False
        self.RANDOM_ROTATION = False
        self.COLOR_JITTERING = False
        self.RANDOM_CHANNEL_SWAPPING = False
        self.RANDOM_GAMMA = False
        self.RANDOM_GRAYSCALE = False
        self.RANDOM_RESOLUTION = False

        self.WORD_VECTORS_SIMILAR_PAD = True
        self.WORD_VECTORS_SIMILAR_PAD_TOPN = 20
