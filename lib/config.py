import os

import numpy as np


class Config():

    def __init__(self):

        ## Files
        self.BASE_PATH = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
        self.DATA_PATH = os.path.join(self.BASE_PATH, 'data')
        self.MODEL_PATH = os.path.join(self.BASE_PATH, 'models')
        self.WORD2VEC_MODEL_FILE = os.path.join(self.MODEL_PATH, 'united_small_word2vec.model')

        ## Shapes  !!!  C x W x H = L x S  !!!
        self.SENTENCE_LENGTH = 8
        self.WV_SIZE = 1536
        self.IMAGE_SIZE_WIDTH = 64
        self.IMAGE_SIZE_HEIGHT = 64
        self.N_CHANNELS = 3
        self.MEAN = [0.5025, 0.5851, 0.4692]
        self.STD = [0.0470, 0.0228, 0.0072]

        ## Word2Vec
        self.WV_MIN_COUNT = 5       ## Ignores all words with total frequency lower than this.
        self.WV_WINDOW = 12         ## Maximum distance between the current and predicted word within a sentence.
        # self.WV_SIZE              ## Dimensionality of the word vectors. ## TODO
        self.WV_SAMPLE = 1e-4       ## The threshold for configuring which higher-frequency words are randomly downsampled. EFFECTIVE!
        self.WV_ALPHA = 1e-2        ## Initial learning rate
        self.WV_MIN_ALPHA = 1e-5    ## Minimum learning rate
        self.WV_EPOCHS = 100        ## Training epochs
        self.WV_NEGATIVE = 5        ## If > 0, negative sampling will be used, 
                                    ## the int for negative specifies how many “noise words” should be drawn (usually between 5-20). 
                                    ## If set to 0, no negative sampling is used.

        ## Batch sampler
        self.SHUFFLE_GROUPS = True
        self.GROUP_N_LABELS_RANGES = [-1, 5, 7, 11, 1000]
        self.GROUP_WIDTH_RANGES = [-1, 500, 700, 1000, 100000]
        self.GROUP_HEIGHT_RANGES = [-1, 100000]
        # self.GROUP_HEIGHT_RANGES = [-1, 590, 100000]

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
        self.WORD_VECTORS_SIMILAR_PAD_TOPN = 10

        ## GAN options
        self.N_INPUT = 1 * 2000    ## TODO: Depends on wv feature size and sentence size
        self.NGF = 64
        self.NDF = 64
        self.GAN_LOSS = 'vanilla'   ## One of 'lsgan', 'vanilla', 'wgangp'
        self.LAMBDA_L1 = 10.0

        ## Hyper-params
        self.BATCH_SIZE = 128
        self.N_EPOCHS = 50
        self.LR = 2e-4
        self.BETA = 0.5

        ## Hardware
        self.N_WORKERS = 0
        self.N_GPUS = 1