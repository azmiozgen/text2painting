import os

import torch

class Config():

    def __init__(self):

        ## Files and names
        self.BASE_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
        SUBSET = 'extreme'
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'data', SUBSET)
        self.MODEL_DIR = os.path.join(self.BASE_DIR, 'models')
        self.WORD2VEC_MODEL_DIR = os.path.join(self.MODEL_DIR, 'word2vec')
        self.WORD2VEC_MODEL_FILE = os.path.join(self.WORD2VEC_MODEL_DIR, SUBSET + '_word2vec.model')
        self.MODEL_NAME = SUBSET
        self.LOG_HEADER = 'Epoch,Iteration,G_loss,D_loss,G_refiner_loss,D_rr_acc,D_rf_acc,D_fr_acc,D_refined_fr_acc'

        ## Shapes
        self.SENTENCE_LENGTH = 4
        self.WV_SIZE = 64        ## Should be same as in data/config.py
        self.IMAGE_WIDTH = 64
        self.IMAGE_HEIGHT = 64
        self.N_CHANNELS = 3
        # assert self.SENTENCE_LENGTH * self.WV_SIZE == self.IMAGE_WIDTH * self.IMAGE_HEIGHT, \
        #        "Incompatible shapes {} x {} != {} x {}".format(self.SENTENCE_LENGTH, self.WV_SIZE, \
        #                                                        self.IMAGE_WIDTH, self.IMAGE_HEIGHT)

        ## Stats (Change w.r.t stats file under data/)
        self.NORMALIZE = True
        self.MEAN = [0.4395, 0.4016, 0.3927]
        self.STD = [0.2658, 0.2378, 0.2782]
        # self.MEAN = [0.4313, 0.6512, 0.5442]
        # self.STD = [0.2159, 0.2929, 0.2476]

        ## Batch sampler
        self.SHUFFLE_GROUPS = True
        # self.GROUP_N_LABELS_RANGES = [-1, 5, 7, 11, 1000]
        self.GROUP_N_LABELS_RANGES = [0, 10000]
        # self.GROUP_WIDTH_RANGES = [-1, 500, 700, 1000, 100000]
        self.GROUP_WIDTH_RANGES = [-1, 100000]
        # self.GROUP_HEIGHT_RANGES = [-1, 590, 100000]
        self.GROUP_HEIGHT_RANGES = [-1, 100000]
        # self.GROUP_HEIGHT_RANGES = [-1, 100000]

        ## Augmentation options
        self.HORIZONTAL_FLIPPING = True
        self.RANDOM_ROTATION = False
        self.COLOR_JITTERING = False
        self.RANDOM_CHANNEL_SWAPPING = True
        self.RANDOM_GAMMA = True
        self.RANDOM_GRAYSCALE = False
        self.RANDOM_RESOLUTION = True

        ## Word vectors options
        self.LOAD_WORD_VECTORS = True
        self.WORD_VECTORS_SIMILAR_PAD = True
        self.WORD_VECTORS_SIMILAR_PAD_TOPN = 2
        self.WORD_VECTORS_SIMILAR_TAKE_SELF = True
        self.WORD_VECTORS_DISSIMILAR_TOPN = 10

        ## GAN options
        self.N_INPUT = self.SENTENCE_LENGTH * self.WV_SIZE
        self.NGF = 256
        self.NDF = 96
        self.GAN_LOSS = 'wgangp'   ## One of 'lsgan', 'vanilla', 'wgangp'
        self.LAMBDA_L1 = 1.0
        self.NORM_LAYER = torch.nn.BatchNorm2d
        self.USE_DROPOUT = True
        self.N_BLOCKS = 1
        self.PADDING_TYPE = 'reflect'
        self.TRAIN_D_TREND = 1    ## e.g. Train D for each 3 epoch, freeze at others
        self.TRAIN_G_TREND = 1    ## e.g. Train G for each 1 epoch, freeze at others
        self.PROB_FLIP_LABELS = 0.00   ## Flip real-fake labels. 0.0 for no flip

        ## Hyper-params
        self.BATCH_SIZE = 4
        self.N_EPOCHS = 1000
        self.LR = 2e-4
        self.BETA = 0.5
        self.WEIGHT_DECAY = 0.0

        ## Hardware
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.N_WORKERS = 4
        self.N_GPUS = 1

        ## Logging
        self.N_PRINT_BATCH = 1
        self.N_LOG_BATCH = 5
        self.N_SAVE_VISUALS_BATCH = 5
        self.N_SAVE_MODEL_EPOCHS = 50
        self.N_GRID_ROW = 8

        ## Misc
        self.FONTS = ['Lato-Medium.ttf', 'FreeMono.ttf', 'LiberationMono-Regular.ttf']
        self.FONT_SIZE = 7
        self.WORDS2IMAGE_N_COLUMN = 2
