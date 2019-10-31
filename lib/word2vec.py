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

## Parameters
N_CORES = multiprocessing.cpu_count()
LABELS_FILE = "../data/wikiart/labels.txt"

if __name__ == "__main__":
    ## Read lines
    with open(LABELS_FILE, 'r') as f:
        lines = f.readlines()
    lines = list(map(lambda s: s.strip(), lines))
        
    ## Set filename-labels dict and collect all unique labels
    filenames = []
    all_labels_org = []
    all_labels = []
    for line in lines:
        row = line.strip().split(',')
        filename = row[0]
        all_labels_org.append(row[1:])
        labels = ' '.join(row[1:])
        labels = re.sub("[^A-Za-z0-9']+", ' ', str(labels)).lower()
        filenames.append(filename)
        all_labels.append(labels)