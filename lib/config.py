import os

import torch

class Config():

    def __init__(self):

        ## Files and names
        self.BASE_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'data')
        self.MODEL_DIR = os.path.join(self.BASE_DIR, 'models')
        self.WORD2VEC_MODEL_DIR = os.path.join(self.MODEL_DIR, 'word2vec')
        self.WORD2VEC_MODEL_FILE = os.path.join(self.WORD2VEC_MODEL_DIR, 'united_word2vec.model')
        self.MODEL_NAME = 'united'
        self.LOG_HEADER = 'Epoch,Iteration,G_loss,D_loss'

        ## Shapes  !!! L X S = C x W x H !!!
        self.SENTENCE_LENGTH = 8
        self.WV_SIZE = 1536
        self.IMAGE_SIZE_WIDTH = 64
        self.IMAGE_SIZE_HEIGHT = 64
        self.N_CHANNELS = 3
        assert self.SENTENCE_LENGTH * self.WV_SIZE == self.N_CHANNELS * self.IMAGE_SIZE_WIDTH * self.IMAGE_SIZE_HEIGHT, \
               "Incompatible shapes {} x {} != {} x {} x {}".format(self.SENTENCE_LENGTH, self.WV_SIZE, self.N_CHANNELS, \
                                                                    self.IMAGE_SIZE_WIDTH, self.IMAGE_SIZE_HEIGHT)

        ## Stats (Change w.r.t stats file under data/)
        self.MEAN = [0.5393, 0.5967, 0.2886]
        self.STD = [0.3146, 0.4590, 0.1506]

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
        self.GROUP_HEIGHT_RANGES = [-1, 590, 100000]

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
        self.WORD_VECTORS_DISSIMILAR_TOPN = 10

        ## GAN options
        self.N_INPUT = self.SENTENCE_LENGTH * self.WV_SIZE
        self.NGF = 64
        self.NDF = 64
        self.GAN_LOSS = 'vanilla'   ## One of 'lsgan', 'vanilla', 'wgangp'
        self.LAMBDA_L1 = 1.0

        ## Hyper-params
        self.BATCH_SIZE = 16
        self.N_EPOCHS = 10
        self.LR = 2e-4
        self.BETA = 0.5
        self.WEIGHT_DECAY = 1e-4

        ## Hardware
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.N_WORKERS = 4
        self.N_GPUS = 1

        ## Logging
        self.N_PRINT_BATCH = 200
        self.N_LOG_BATCH = 200
        self.N_SAVE_MODEL_EPOCHS = 1