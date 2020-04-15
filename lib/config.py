import os

import torch

class Config():

    def __init__(self):

        ## Files and names
        self.BASE_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
        SUBSET = 'verified'
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'data', SUBSET)
        self.MODEL_DIR = os.path.join(self.BASE_DIR, 'models')
        self.WORD2VEC_MODEL_DIR = os.path.join(self.MODEL_DIR, 'word2vec')
        self.WORD2VEC_MODEL_FILE = os.path.join(self.WORD2VEC_MODEL_DIR, SUBSET + '_word2vec.model')
        self.MODEL_NAME = SUBSET
        self.LOG_HEADER = 'Epoch,Iteration,G_loss,D_loss,G_refiner_loss,D_decider_loss,\
G_refiner2_loss,D_decider2_loss,D_rr_acc,D_rf_acc,D_fr_acc,D_decider_rr_acc,D_decider_fr_acc,D_decider2_rr_acc,D_decider2_fr_acc'

        ## Shapes
        self.SENTENCE_LENGTH = 6
        self.NOISE_LENGTH = 1
        self.WV_SIZE = 64        ## Should be same as in data/config.py
        self.IMAGE_WIDTH_FIRST = 64
        self.IMAGE_HEIGHT_FIRST = 64
        self.IMAGE_WIDTH_SECOND = 128
        self.IMAGE_HEIGHT_SECOND = 128
        self.IMAGE_WIDTH = 256
        self.IMAGE_HEIGHT = 256
        self.N_CHANNELS = 3

        ## Stats (Change w.r.t stats file under data/)
        self.NORMALIZE = True
        self.MEAN = [0.5, 0.5, 0.5]
        self.STD = [0.5, 0.5, 0.5]
        # self.MEAN = [0.5505, 0.3927, 0.4473]
        # self.STD = [0.2245, 0.2782, 0.3102]
        # self.MEAN = [0.485, 0.456, 0.406]
        # self.STD = [0.229, 0.224, 0.225]

        ## Batch sampler
        # self.SHUFFLE_GROUPS = True
        # self.GROUP_N_LABELS_RANGES = [-1, 5, 1000]
        # self.GROUP_N_LABELS_RANGES = [0, 10000]
        # self.GROUP_WIDTH_RANGES = [-1, 500, 100000]
        # self.GROUP_WIDTH_RANGES = [-1, 100000]
        # self.GROUP_HEIGHT_RANGES = [-1, 590, 100000]
        # self.GROUP_HEIGHT_RANGES = [-1, 100000]
        # self.GROUP_HEIGHT_RANGES = [-1, 100000]

        ## Augmentation options
        self.RANDOM_BLURRINESS = True
        self.HORIZONTAL_FLIPPING = False     ## BAD!
        self.RANDOM_ROTATION = False
        self.COLOR_JITTERING = True
        self.RANDOM_CHANNEL_SWAPPING = False
        self.RANDOM_GAMMA = True
        self.RANDOM_GRAYSCALE = False
        self.RANDOM_RESOLUTION = True
        self.ELASTIC_DEFORMATION = False
        self.SHARPENING = True
        self.EQUALIZING = True

        ## Word vectors options
        self.LOAD_WORD_VECTORS = True
        self.WORD_VECTORS_SIMILAR_PAD = True
        self.WORD_VECTORS_SIMILAR_PAD_TOPN = 10
        self.WORD_VECTORS_SIMILAR_TAKE_SELF = False
        self.WORD_VECTORS_DISSIMILAR_TOPN = 10
        self.MIN_WV_SIMILARITY_PROB = 0.50

        ## GAN options
        self.N_INPUT = (self.SENTENCE_LENGTH + self.NOISE_LENGTH) * self.WV_SIZE
        self.NGF = 128
        self.NDF = 128
        self.NG_REF_F = 64
        self.ND_DEC_F = 64
        self.OUT_CHANNELS = 1
        self.GAN_LOSS1 = 'wgangp'    ## One of 'lsgan', 'vanilla', 'wgangp'
        self.GAN_LOSS2 = 'wgangp'    ## One of 'lsgan', 'vanilla', 'wgangp'
        self.LAMBDA_L1 = 100.0
        self.NORM_LAYER = torch.nn.BatchNorm2d
        self.G_DROPOUT = 0.2
        self.D_DROPOUT = 0.65
        self.USE_SPECTRAL_NORM = True
        self.MINIBATCH_DISCRIMINATION = True
        self.N_BLOCKS = 9
        self.PADDING_TYPE = 'reflect'   ## One of 'reflect', 'replicate', 'zero'
        self.TRAIN_D_TREND = 1    ## e.g. Train D for each 3 epoch, freeze at others
        self.TRAIN_G_TREND = 1    ## e.g. Train G for each 1 epoch, freeze at others
        self.PROB_FLIP_LABELS = 0.05   ## Flip real-fake labels. 0.0 for no flip

        ## Hyper-params
        self.BATCH_SIZE = 16
        self.N_EPOCHS = 1000
        self.G_LR = 1e-4
        self.D_LR = 2e-4
        self.G_REFINER_LR = 1e-4
        self.D_DECIDER_LR = 2e-4
        self.G_REFINER2_LR = 1e-4
        self.D_DECIDER2_LR = 2e-4
        self.LR_DROP_FACTOR = 0.5
        self.LR_DROP_PATIENCE = self.N_EPOCHS // 10
        self.LR_MIN_VAL = 1e-5
        self.BETA = 0.5
        self.WEIGHT_DECAY = 1e-5
        self.WEIGHT_INIT = 'kaiming'  ## One of 'normal', 'xavier', 'kaiming', 'orthogonal'
        self.INIT_GAIN = 0.02

        ## Hardware
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.DEVICE = torch.device('cpu')
        self.N_WORKERS = 8
        self.N_GPUS = 1

        ## Logging
        # self.N_PRINT_BATCH = 50
        # self.N_LOG_BATCH = 100
        # self.N_SAVE_VISUALS_BATCH = 100
        # self.N_SAVE_MODEL_EPOCHS = 20
        # self.N_GRID_ROW = 8
        self.N_PRINT_BATCH = 50
        self.N_LOG_BATCH = 50
        self.N_SAVE_VISUALS_BATCH = 50
        self.N_SAVE_MODEL_EPOCHS = 10
        self.N_GRID_ROW = 10

        ## Misc
        self.FONTS = ['Lato-Bold.ttf', 'FreeMonoBold.ttf', 'LiberationMono-Bold.ttf']
        self.FONT_SIZE = 30
        self.WORDS2IMAGE_N_COLUMN = 1
