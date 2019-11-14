import os

import numpy as np


class Config():

    def __init__(self):

        ## Files
        self.BASE_PATH = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
        self.DATA_PATH = os.path.join(self.BASE_PATH, 'data')
        self.MODEL_PATH = os.path.join(self.BASE_PATH, 'models')
        self.WORD2VEC_MODEL_FILE = os.path.join(self.MODEL_PATH, 'united_word2vec.model')

        ## Image shapes
        self.MEAN = [0.5025, 0.5851, 0.4692]
        self.STD = [0.0470, 0.0228, 0.0072]
        self.IMAGE_SIZE_WIDTH = 64
        self.IMAGE_SIZE_HEIGHT = 64
        self.N_CHANNELS = 3

        ## Augmentation options
        self.HORIZONTAL_FLIPPING = False
        self.RANDOM_ROTATION = False
        self.COLOR_JITTERING = False
        self.RANDOM_CHANNEL_SWAPPING = False
        self.RANDOM_GAMMA = False
        self.RANDOM_GRAYSCALE = False
        self.RANDOM_RESOLUTION = False

        ## Word vectors options
        self.LOAD_WORD_VECTORS = True
        self.WORD_VECTORS_SIMILAR_PAD = True
        self.WORD_VECTORS_SIMILAR_PAD_TOPN = 20

        ## GAN options
        self.N_INPUT = 1 * 2000    ## TODO: Depends on wv feature size and sentence size
        self.NGF = 64
        self.NDF = 64

        ## Hyper-params
        self.BATCH_SIZE = 128
        self.N_EPOCHS = 50
        self.LR = 2e-4
        self.BETA = 0.5

        ## Hardware
        self.WORKERS = 0
        self.N_GPUS = 1