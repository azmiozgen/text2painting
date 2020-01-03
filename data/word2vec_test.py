import glob
import os
import sys

import numpy as np

from gensim.models import Word2Vec, word2vec


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

    if len(sys.argv) != 2:
        print("Usage: python {} <subset>".format(sys.argv[0]))
        exit()
    SUBSET = sys.argv[1]
    MODEL_FILENAME = SUBSET + '_word2vec.model'
    MODEL_FILE = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, 'models', 'word2vec', MODEL_FILENAME))
    TOP_N = 20

    ## Check model file
    if not os.path.isfile(MODEL_FILE):
        print(MODEL_FILE, 'not found. Exiting')
        exit()

    ## Load model
    print("Loading {}".format(MODEL_FILE))
    word2vec_master = Word2Vec.load(MODEL_FILE)
    print("\tVocabulary shape:", word2vec_master.wv.vectors.shape)

    ## Test model
    keywords = ['dog', 'cat', 'female', 'male', 'horse']
    for keyword in keywords:
        if not word2vec_master.wv.vocab.get(keyword):
            continue
        similar_words = get_similar_words(word2vec_master, keyword, topN=TOP_N)
        print("Top {} similar words to {}".format(TOP_N, keyword))
        for word in similar_words:
            print('\t', word)
