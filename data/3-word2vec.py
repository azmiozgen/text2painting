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


class StemmingLemmatization():

    def __init__(self, label_sentences):
        self.nlp = spacy.load('en', disable=['ner', 'parser']) # disabling Named Entity Recognition for speed
        self.label_sentences = label_sentences

    def _clean_doc(self, doc):
        txt = [token.lemma_ for token in doc if not token.is_stop]
        if len(txt) > 0:
            return ' '.join(txt)

    def get_corpus(self):
        labels_txt = [self._clean_doc(doc) for doc in self.nlp.pipe(label_sentences, batch_size=5000, n_threads=-1)]
        label_corpus = list(map(lambda s:str(s).split(), labels_txt))
        return label_corpus

class Word2VecModelGenerator():

    def __init__(self, label_corpus, model_output_name):
        '''
            label_corpus : List of label tokens. Passed from stemmer and lemmatizer preferably.
            model_output_name : String 
        '''
        ## Parameters
        MIN_COUNT = 5       ## Ignores all words with total frequency lower than this.
        WINDOW = 20         ## Maximum distance between the current and predicted word within a sentence.
        SIZE = 2000         ## Dimensionality of the word vectors.
        SAMPLE = 1e-4       ## The threshold for configuring which higher-frequency words are randomly downsampled. EFFECTIVE!
        ALPHA = 1e-2        ## Initial learning rate
        MIN_ALPHA = 1e-5    ## Minimum learning rate
        EPOCHS = 100        ## Training epochs
        NEGATIVE = 5        ## If > 0, negative sampling will be used, 
                            ## the int for negative specifies how many “noise words” should be drawn (usually between 5-20). 
                            ## If set to 0, no negative sampling is used.
        WORKERS = multiprocessing.cpu_count() - 1
        MODEL_OUTPUT_FILENAME = model_output_name + '_word2vec.model'
        MODEL_OUTPUT_FILE = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, 'models', MODEL_OUTPUT_FILENAME))

        self.label_corpus = label_corpus
        self.epochs = EPOCHS
        self.model_output_file = MODEL_OUTPUT_FILE
        self.model = Word2Vec(min_count=MIN_COUNT,
                              window=WINDOW,
                              size=SIZE,
                              sample=SAMPLE,
                              alpha=ALPHA,
                              min_alpha=MIN_ALPHA,
                              negative=NEGATIVE,
                              workers=WORKERS,
                              )

        self.model.build_vocab(label_corpus,
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
        self.model.train(self.label_corpus,
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
        print("Usage: python {} <subset_dir>".format(sys.argv[0]))
        exit()
    SUBSET = sys.argv[1]

    if not os.path.isdir(SUBSET):
        print(SUBSET, "not found. Exiting")
        exit()

    ## Parameters
    LABEL_SENTENCES_FILENAME = 'train_label_sentences.txt'
    LABEL_SENTENCES_FILE = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, 'data', SUBSET, LABEL_SENTENCES_FILENAME))
    TOP_N = 20

    ## Check label sentences file
    if not os.path.isfile(LABEL_SENTENCES_FILE):
        print(LABEL_SENTENCES_FILE, 'not found. Exiting')
        exit()
    else:
        print("Processing {}".format(LABEL_SENTENCES_FILE))

    ## Read label sentences file
    with open(LABEL_SENTENCES_FILE, 'r') as f:
        lines = f.readlines()
    label_sentences = list(map(lambda s: s.strip(), lines))

    ## Get stemmed and lemmatized label corpus
    print("Stemming and lemmatizing..")
    stemmer_lemmatizer = StemmingLemmatization(label_sentences)
    label_corpus = stemmer_lemmatizer.get_corpus()

    ## Init Word2Vec model
    print("Initializing Word2Vec model..")
    word2vec_master = Word2VecModelGenerator(label_corpus, SUBSET)

    ## Train model
    print("Training model..")
    word2vec_master.train()

    # ## Quick-test model
    # keywords = ['dog', 'cat', 'female', 'male']
    # for keyword in keywords:
    #     similar_words = word2vec_master.get_similar_words(keyword, topN=TOP_N)
    #     print("Top {} similar words to {}".format(TOP_N, keyword))
    #     for word in similar_words:
    #         print('\t', word)

