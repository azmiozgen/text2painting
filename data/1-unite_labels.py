## Read subset data .csv labels 
## and get united filename-tokens format separated by comma for Word2Vec.
import os
import re
import sys
import warnings

import numpy as np
from PIL import Image

from config import Config

if __name__ == '__main__':

    # SUBSET_LIST = ['deviantart', 'wikiart']
    #SUBSET_LIST = ['deviantart_1', 'deviantart_2', 'deviantart_3', 'deviantart_4']
    SUBSET_LIST = ['deviantart_verified']
    FRACTION = 1.0  ## To prepare smaller dataset, make 1.0 to take all

    if len(sys.argv) != 2:
        print("Usage: python {} <output_dir>".format(sys.argv[0]))
        exit()
    OUTPUT_DIR = sys.argv[1]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'all_labels.csv')
    if os.path.isfile(OUTPUT_FILE):
        print(OUTPUT_FILE, 'exists. Exiting.')
        exit()

    warnings.filterwarnings('error')

    config = Config()

    image_file_counter = 0
    all_lines = []
    for subset in SUBSET_LIST:
        labels_file = os.path.join(subset, 'labels.csv')

        if not os.path.isfile(labels_file):
            print(labels_file, 'not found. Passing')
            continue

        with open(labels_file, 'r') as f:
            lines = f.readlines()

        ## Fraction the lines
        fraction = FRACTION / len(SUBSET_LIST)
        lines = sorted(np.random.choice(lines, int(len(lines) * FRACTION), replace=False))

        for line in lines:
            row = line.strip().split(',')
            image_filename = row[0]
            labels = row[1:]
            image_relative_file = os.path.join('data', subset, 'images', image_filename)  ## Relative to project base
            image_file = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, image_relative_file))

            ## Check image file existence
            if not os.path.isfile(image_file):
                continue

            ## Apply label conditions
            if len(labels) < config.MIN_SENTENCE_LENGTH or len(labels) > config.MAX_SENTENCE_LENGTH:
                continue

            ## Check image integrity
            try:
                img = Image.open(image_file) 
                if img.mode != 'RGB': 
                    continue
            except (Exception, Warning) as e:
                print("Bad image", image_file, e)
                continue

            ## Apply image shape conditions
            w, h = img.size
            if w < config.MIN_IMAGE_WIDTH or w > config.MAX_IMAGE_WIDTH:
                continue
            if h < config.MIN_IMAGE_HEIGHT or w > config.MAX_IMAGE_HEIGHT:
                continue
            if max(w, h) / (min(w, h) + 1e-7) > config.MAX_ASPECT_RATIO:
                continue

            ## Normalize tokens
            label_sentence = str(' '.join(labels))                                       ## First column is image filename
            label_sentence = re.sub(r"[^A-Za-z0-9']+", " ", label_sentence).lower()      ## Only alphanumeric characters
            label_sentence = re.sub(r"\b[a-zA-Z]\b", " ", label_sentence)                ## Replace single letters
            label_sentence = re.sub("   ", " ", label_sentence)                          ## Go one space
            label_sentence = re.sub("  ", " ", label_sentence)                           ## Go one space
            label_sentence = label_sentence.replace(' ', ',')
            all_lines.append(image_relative_file + ',' + label_sentence + '\n')

            image_file_counter += 1


    with open(OUTPUT_FILE, 'w') as f:
        f.writelines(all_lines)
    print(OUTPUT_FILE, 'was written with {} file'.format(image_file_counter))
