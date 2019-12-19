import os

class Config():

    def __init__(self):

        ## Files and names
        self.BASE_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
        SUBSET = 'verified'
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'data', SUBSET)
        self.MODEL_DIR = os.path.join(self.BASE_DIR, 'models')
        self.WORD2VEC_MODEL_DIR = os.path.join(self.MODEL_DIR, 'word2vec')
        self.WORD2VEC_MODEL_FILE = os.path.join(self.WORD2VEC_MODEL_DIR, SUBSET + '_word2vec.model')

        ## Data preprocess
        self.MIN_SENTENCE_LENGTH = 4
        self.MAX_SENTENCE_LENGTH = 20
        self.MIN_IMAGE_WIDTH = 50
        self.MIN_IMAGE_HEIGHT = 50
        self.MAX_IMAGE_WIDTH = 2000
        self.MAX_IMAGE_HEIGHT = 2000
        self.MAX_ASPECT_RATIO = 2.0

        ## Word2Vec
        self.WV_MIN_COUNT = 3       ## Ignores all words with total frequency lower than this.
        self.WV_WINDOW = 10         ## Maximum distance between the current and predicted word within a sentence.
        self.WV_SIZE = 64           ## Dimensionality of the word vectors.
        self.WV_SAMPLE = 2e-4       ## The threshold for configuring which higher-frequency words are randomly downsampled. EFFECTIVE!
        self.WV_ALPHA = 1e-2        ## Initial learning rate
        self.WV_MIN_ALPHA = 1e-5    ## Minimum learning rate
        self.WV_EPOCHS = 500        ## Training epochs
        self.WV_NEGATIVE = 20       ## If > 0, negative sampling will be used, 
                                    ## the int for negative specifies how many “noise words” should be drawn (usually between 5-20). 
                                    ## If set to 0, no negative sampling is used.