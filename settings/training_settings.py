import os
from .model_settings import ModelSettings


class TrainingSettings(ModelSettings):

    def __init__(self):
        super(TrainingSettings, self).__init__()

        self.TRAINING_LMDB = os.path.join(self.BASE_PATH, 'data', 'processed', 'lmdb', 'ara-lat_real_training-lmdb')
        self.VALIDATION_LMDB = os.path.join(self.BASE_PATH, 'data', 'processed', 'lmdb', 'ara-lat_real_validation-lmdb')

        self.MODEL = 'resnet18'  # one of : 'resnet18', 'alexnet', 'squeezenet', ''
        self.OPTIMIZER = 'Adam' # one of : 'RMSprop', 'Adam', 'Adadelta', 'SGD' #TODO
        self.LEARNING_RATE = 0.001
        self.LR_DROP_FACTOR = 0.1
        self.LR_DROP_PATIENCE = 7
        self.WEIGHT_DECAY = 0.001 #TODO 5e-4      # use 0 to disable it
        self.CLIP_GRAD_NORM = 0.0 #TODO 10      # max l2 norm of gradient of parameters - use 0 to disable it

        self.SEED = 13
