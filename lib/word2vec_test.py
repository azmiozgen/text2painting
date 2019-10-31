import glob
import os
import re
import string

import numpy as np
import pandas as pd
import multiprocessing

from gensim.models import word2vec, Word2Vec
from gensim.models.phrases import Phrases, Phraser
import spacy

def get_similar_words(model, word, topN=10):
    if model.wv.vocab.get(word):
        return model.wv.most_similar(word, topn=topN)
    else:
        print("Word '{}' is not in vocabulary".format(word))
        return None


def get_vector(model, word):
    if model.wv.vocab.get(word):
        return model[word]
    else:
        print("Word '{}' is not in vocabulary".format(word))
        return None


if __name__ == "__main__":
    ## Parameters
    SUBSET = 'deviant_wiki'
    MODEL_FILENAME = SUBSET + '_word2vec.model'
    MODEL_FILE = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, 'models', MODEL_FILENAME))
    TOP_N = 20

    ## Check model file
    if not os.path.isfile(MODEL_FILE):
        print(MODEL_FILE, 'not found. Exiting')
        exit()

    ## Load model
    print("Loading {}".format(MODEL_FILE))
    word2vec_master = Word2Vec.load(MODEL_FILE)

    ## Test model
    keywords = ['dog', 'cat', 'female', 'male']
    for keyword in keywords:
        similar_words = get_similar_words(word2vec_master, keyword, topN=TOP_N)
        print("Top {} similar words to {}".format(TOP_N, keyword))
        for word in similar_words:
            print('\t', word)

