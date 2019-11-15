OUTPUT_DIR=${1}

echo
echo    Subsets are being united..
python 1-unite_labels.py ${OUTPUT_DIR}

echo
echo    Train-val-test are being splitted..
python 2-split_labels.py ${OUTPUT_DIR}

echo
echo    Word2Vec started..
python 3-word2vec.py ${OUTPUT_DIR}/train_labels.csv ${OUTPUT_DIR}

echo
echo    Image stats are being extracted..
python 4-get_image_stats.py ${OUTPUT_DIR} train_labels.csv train
python 4-get_image_stats.py ${OUTPUT_DIR} val_labels.csv val
