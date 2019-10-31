## Convert .csv label format to sentence format (each line is token series with space between) for Word2Vec.
import os
import re

SUBSET = 'deviantart'
LABELS_FILE = os.path.join(SUBSET, 'labels.csv')
OUTPUT_FILE = os.path.join(SUBSET, 'label_sentences.txt')

if __name__ == '__main__':

    if not os.path.isfile(LABELS_FILE):
        print(LABELS_FILE, 'not found. Exiting')
        exit()

    if os.path.isfile(OUTPUT_FILE):
       print(OUTPUT_FILE, 'exists. Exiting.')
       exit()

    with open(LABELS_FILE, 'r') as f:
        lines = f.readlines()
    
    all_labels = []
    for line in lines:
        row = line.strip().split(',')
        labels = ' '.join(row[1:])           ## First column is image filename
        labels = re.sub(r"[^A-Za-z0-9']+", " ", str(labels)).lower()  ## Only alphanumeric characters
        labels = re.sub(r"\b[a-zA-Z]\b", " ", labels)                ## Replace single letters
        labels = re.sub("   ", " ", labels)                          ## Go one space
        labels = re.sub("  ", " ", labels)                           ## Go one space
        all_labels.append(labels + '\n')

    with open(OUTPUT_FILE, 'w') as f:
        f.writelines(all_labels)
    print(OUTPUT_FILE, 'was written.')
