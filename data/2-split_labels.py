## Read all_labels.csv and split into train-val-test
import os
import random
import sys

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Usage: python {} <output_dir>".format(sys.argv[0]))
        exit()
    OUTPUT_DIR = sys.argv[1]

    if not os.path.isdir(OUTPUT_DIR):
        print(OUTPUT_DIR, 'not found. Exiting.')
        exit()

    ALL_LABELS_FILE = os.path.join(OUTPUT_DIR, 'all_labels.csv')
    TRAIN_OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'train_labels.csv')
    VAL_OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'val_labels.csv')
    TEST_OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'test_labels.csv')

    VAL_RATIO = 0.25
    TEST_RATIO = 0.0
    SEED = 73

    if not os.path.isfile(ALL_LABELS_FILE):
        print(ALL_LABELS_FILE, 'not found. Exiting.')
        exit()

    with open(ALL_LABELS_FILE, 'r') as f:
        lines = f.readlines()
    n_lines = len(lines)

    ## Shuffle lines
    random.seed(SEED)
    random.shuffle(lines)

    ## Set sub lines
    train_size = int(n_lines * (1 - (VAL_RATIO + TEST_RATIO))) + 1
    val_size = int(n_lines * VAL_RATIO) + 1
    test_size = int(n_lines * TEST_RATIO)
    train_lines = lines[:train_size]
    val_lines = lines[train_size:train_size + val_size]
    test_lines = lines[train_size + val_size:]

    ## Write train lines
    with open(TRAIN_OUTPUT_FILE, 'w') as f:
        f.writelines(train_lines)
    print(TRAIN_OUTPUT_FILE, 'was written.')

    ## Write val lines
    with open(VAL_OUTPUT_FILE, 'w') as f:
        f.writelines(val_lines)
    print(VAL_OUTPUT_FILE, 'was written.')

    ## Write test lines
    with open(TEST_OUTPUT_FILE, 'w') as f:
        f.writelines(test_lines)
    print(TEST_OUTPUT_FILE, 'was written.')
    
