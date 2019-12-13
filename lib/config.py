import os

import torch

class Config():

    def __init__(self):

        ## Files and names
        self.BASE_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
        SUBSET = 'united'
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'data', SUBSET)
        self.MODEL_DIR = os.path.join(self.BASE_DIR, 'models')
        self.WORD2VEC_MODEL_DIR = os.path.join(self.MODEL_DIR, 'word2vec')
        self.WORD2VEC_MODEL_FILE = os.path.join(self.WORD2VEC_MODEL_DIR, SUBSET + '_word2vec.model')
        self.MODEL_NAME = SUBSET
        self.LOG_HEADER = 'Epoch,Iteration,G_loss,D_loss,G_refiner_loss,D_decider_loss,D_rr_acc,D_rf_acc,D_fr_acc,D_decider_rr_acc,D_decider_fr_acc'

        ## Shapes
        self.SENTENCE_LENGTH = 4
        self.NOISE_LENGTH = 1
        self.WV_SIZE = 64        ## Should be same as in data/config.py
        self.IMAGE_WIDTH = 64
        self.IMAGE_HEIGHT = 64
        self.N_CHANNELS = 3
        # assert self.SENTENCE_LENGTH * self.WV_SIZE == self.IMAGE_WIDTH * self.IMAGE_HEIGHT, \
        #        "Incompatible shapes {} x {} != {} x {}".format(self.SENTENCE_LENGTH, self.WV_SIZE, \
        #                                                        self.IMAGE_WIDTH, self.IMAGE_HEIGHT)

        ## Stats (Change w.r.t stats file under data/)
        self.NORMALIZE = True
        # self.MEAN = [0.1036, 0.3302, 0.3311]
        # self.STD = [0.1233, 0.2243, 0.3484]
        # self.MEAN = [0.5505, 0.3927, 0.4473]
        # self.STD = [0.2245, 0.2782, 0.3102]
        self.MEAN = [0.9036, 0.7096, 0.8660]
        self.STD = [0.2221, 0.1697, 0.2122]

        ## Batch sampler
        self.SHUFFLE_GROUPS = True
        # self.GROUP_N_LABELS_RANGES = [-1, 5, 7, 11, 1000]
        # self.GROUP_N_LABELS_RANGES = [0, 10000]
        # self.GROUP_WIDTH_RANGES = [-1, 500, 700, 1000, 100000]
        self.GROUP_WIDTH_RANGES = [-1, 100000]
        # self.GROUP_HEIGHT_RANGES = [-1, 590, 100000]
        self.GROUP_HEIGHT_RANGES = [-1, 100000]
        # self.GROUP_HEIGHT_RANGES = [-1, 100000]

        ## Augmentation options
        self.HORIZONTAL_FLIPPING = True
        self.RANDOM_ROTATION = False
        self.COLOR_JITTERING = True
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
        self.N_INPUT = (self.SENTENCE_LENGTH + self.NOISE_LENGTH) * self.WV_SIZE
        self.NGF = 128
        self.NDF = 64
        self.NG_REF_F = 64
        self.ND_DEC_F = 32
        self.GAN_LOSS1 = 'lsgan'    ## One of 'lsgan', 'vanilla', 'wgangp'
        self.GAN_LOSS2 = 'lsgan'    ## One of 'lsgan', 'vanilla', 'wgangp'
        self.LAMBDA_L1 = 100.0
        self.NORM_LAYER = torch.nn.BatchNorm2d
        self.USE_SPECTRAL_NORM = True
        self.USE_DROPOUT = True
        self.N_BLOCKS = 9
        self.PADDING_TYPE = 'reflect'   ## One of 'reflect', 'replicate', 'zero'
        self.TRAIN_D_TREND = 1    ## e.g. Train D for each 3 epoch, freeze at others
        self.TRAIN_G_TREND = 1    ## e.g. Train G for each 1 epoch, freeze at others
        self.PROB_FLIP_LABELS = 0.05   ## Flip real-fake labels. 0.0 for no flip

        ## Hyper-params
        self.BATCH_SIZE = 16
        self.N_EPOCHS = 2000
        self.G_LR = 1e-4
        self.D_LR = 2e-4
        self.G_REFINER_LR = 1e-4
        self.D_DECIDER_LR = 2e-4
        self.LR_DROP_FACTOR = 0.5
        self.LR_DROP_PATIENCE = self.N_EPOCHS // 10
        self.LR_MIN_VAL = 1e-6
        self.BETA = 0.5
        self.WEIGHT_DECAY = 0.0
        self.WEIGHT_INIT = 'kaiming'  ## One of 'normal', 'xavier', 'kaiming', 'orthogonal'
        self.INIT_GAIN = 0.02

        ## Hardware
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.N_WORKERS = 8
        self.N_GPUS = 1

        ## Logging
        # self.N_PRINT_BATCH = 50
        # self.N_LOG_BATCH = 100
        # self.N_SAVE_VISUALS_BATCH = 100
        # self.N_SAVE_MODEL_EPOCHS = 20
        # self.N_GRID_ROW = 8
        self.N_PRINT_BATCH = 20
        self.N_LOG_BATCH = 30
        self.N_SAVE_VISUALS_BATCH = 30
        self.N_SAVE_MODEL_EPOCHS = 100
        self.N_GRID_ROW = 8

        ## Misc
        self.FONTS = ['Lato-Medium.ttf', 'FreeMono.ttf', 'LiberationMono-Regular.ttf']
        self.FONT_SIZE = 7
        self.WORDS2IMAGE_N_COLUMN = 2
