## Read subset data .csv labels 
## and get united filename-tokens format separated by comma for Word2Vec.
import os
import re
import sys
import warnings

from PIL import Image

if __name__ == '__main__':

    SUBSET_LIST = ['deviantart', 'wikiart']

    if len(sys.argv) != 2:
        print("Usage: python {} <output_dir>".format(sys.argv[0]))
        exit()
    OUTPUT_DIR = sys.argv[1]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'all_labels.csv')

    warnings.filterwarnings('error')

    if os.path.isfile(OUTPUT_FILE):
        print(OUTPUT_FILE, 'exists. Exiting.')
        exit()

    image_file_counter = 0
    all_lines = []
    for subset in SUBSET_LIST:
        labels_file = os.path.join(subset, 'labels.csv')

        if not os.path.isfile(labels_file):
            print(labels_file, 'not found. Passing')
            continue

        with open(labels_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            row = line.strip().split(',')
            image_filename = row[0]
            image_relative_file = os.path.join('data', subset, 'images', image_filename)  ## Relative to project base
            image_file = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, image_relative_file))

            if not os.path.isfile(image_file):
                continue

            ## Check image integrity
            try:
                img = Image.open(image_file) 
                if img.mode != 'RGB': 
                    img = img.convert('RGB') 
            except (Exception, Warning) as e:
                print("Bad image,", image_file, e)
                continue

            ## Normalize tokens
            label_sentence = str(' '.join(row[1:]))                                      ## First column is image filename
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
