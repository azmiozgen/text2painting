import glob
import os
import re
import string
import sys

import numpy as np
import pandas as pd
import multiprocessing

from gensim.models import word2vec, Word2Vec
from gensim.models.phrases import Phrases, Phraser
import spacy

from config import Config


class StemmingLemmatization():

    def __init__(self, label_sentences):
        self.nlp = spacy.load('en', disable=['ner', 'parser']) # disabling Named Entity Recognition for speed
        self.label_sentences = label_sentences
        self.lookup = {'christmas': 'christmas'}

    def _clean_doc(self, doc):
        txt = []
        for token in doc:
            if token.text in self.lookup:
                txt.append(self.lookup[token.text])
            else:
                if not token.is_stop:
                    txt.append(token.lemma_)
        # txt = [token.lemma_ for token in doc if not token.is_stop]
        if len(txt) > 0:
            return ' '.join(txt)

    def get_corpus(self):
        labels_txt = [self._clean_doc(doc) for doc in self.nlp.pipe(label_sentences, batch_size=500, n_threads=-1)]
        label_corpus = list(map(lambda s:str(s).split(), labels_txt))
        return label_corpus

class Word2VecModelGenerator():

    def __init__(self, label_corpus, config):
        '''
            label_corpus : List of label tokens. Passed from stemmer and lemmatizer preferably.
            model_output_name : String 
        '''

        WORKERS = multiprocessing.cpu_count() - 1
        self.label_corpus = label_corpus
        self.epochs = config.WV_EPOCHS
        self.model_output_file = config.WORD2VEC_MODEL_FILE
        self.model = Word2Vec(
                             min_count=config.WV_MIN_COUNT,
                             window=config.WV_WINDOW,
                             size=config.WV_SIZE,
                             sample=config.WV_SAMPLE,
                             alpha=config.WV_ALPHA,
                             min_alpha=config.WV_MIN_ALPHA,
                             negative=config.WV_NEGATIVE,
                             workers=WORKERS,
                             )

        self.model.build_vocab(
                              label_corpus,
                              update=False,
                              progress_per=1000,
                              keep_raw_vocab=False,
                              trim_rule=None,
                              )

        print("\tVocabulary shape:", self.model.wv.vectors.shape)
        # ## Phraser
        # self.label_corpus = self._phraser()

    def _phraser(self):
        ## Phraser (Find common phrases for low memory consumption at training phase)
        ## NOT REALLY GOOD
        phrases = Phrases(self.label_corpus, min_count=30, progress_per=10000)
        bigram = Phraser(phrases)
        label_corpus_alt = bigram[self.label_corpus]
        print(len(label_corpus_alt.obj.phrasegrams), "phrases found.")
        return label_corpus_alt

    def train(self):
        self.model.train(
                        self.label_corpus,
                        total_examples=self.model.corpus_count,
                        total_words=None,
                        epochs=self.epochs,
                        start_alpha=None,
                        end_alpha=None,
                        word_count=0,
                        queue_factor=2,
                        report_delay=1.0,
                        compute_loss=False,
                        callbacks=(),
                        )

        self.model.save(self.model_output_file)
        if os.path.isfile(self.model_output_file):
            print("Model {} was saved".format(self.model_output_file))
        else:
            print("Model {} saving FAILED!".format(self.model_output_file))

    def get_similar_words(self, word, topN=10):
        if self.model.wv.vocab.get(word):
            return self.model.wv.most_similar(word, topn=topN)
        else:
            print("Word '{}' is not in vocabulary".format(word))
            return None


    def get_vector(self, word):
        if self.model.wv.vocab.get(word):
            return self.model[word]
        else:
            print("Word '{}' is not in vocabulary".format(word))
            return None

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python {} <labels_file>".format(sys.argv[0]))
        exit()
    LABELS_FILE = sys.argv[1]

    if not os.path.isfile(LABELS_FILE):
        print(LABELS_FILE, "not found. Exiting")
        exit()

    config = Config()

    ## Read labels file
    with open(LABELS_FILE, 'r') as f:
        lines = f.readlines()
    lines = list(map(lambda s: s.strip(), lines))

    label_sentences = []
    for line in lines:
        labels = line.split(',')[1:]
        label_sentence = ' '.join(labels)
        label_sentences.append(label_sentence)

    ## Get stemmed and lemmatized label corpus
    print("Stemming and lemmatizing..")
    stemmer_lemmatizer = StemmingLemmatization(label_sentences)
    label_corpus = stemmer_lemmatizer.get_corpus()

    ## Init Word2Vec model
    print("Initializing Word2Vec model..")
    word2vec_master = Word2VecModelGenerator(label_corpus, config)

    ## Train model
    print("Training model..")
    word2vec_master.train()

    # ## Quick-test model
    # TOP_N = 20
    # keywords = ['dog', 'cat', 'female', 'male']
    # for keyword in keywords:
    #     similar_words = word2vec_master.get_similar_words(keyword, topN=TOP_N)
    #     print("Top {} similar words to {}".format(TOP_N, keyword))
    #     for word in similar_words:
    #         print('\t', word)
