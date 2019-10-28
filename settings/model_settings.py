from .data_settings import DataSettings


class ModelSettings(DataSettings):

    def __init__(self):
        super(ModelSettings, self).__init__()

        self.MEAN = [0.43501,0.57167,0.39230]
        self.STD = [0.08248,0.11526,0.02106]

        self.IMAGE_SIZE_WIDTH = 192
        self.IMAGE_SIZE_HEIGHT = 72 #~(width * 0.29)

        self.HORIZONTAL_FLIPPING = False
        self.RANDOM_ROTATION = True
        self.COLOR_JITTERING = False
        self.RANDOM_CHANNEL_SWAPPING = True
        self.RANDOM_GAMMA = True
        self.RANDOM_GRAYSCALE = True
        self.RANDOM_RESOLUTION = True
