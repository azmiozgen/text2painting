echo "Subsets are being united.."
python 1-unite_labels.csv

echo "Train-val-test are being splitted.."
python 2-split_labels.csv

echo "Word2Vec started.."
python 3-word2vec.py united

echo "Image stats are being extracted.."
python 3-get_image_stats.py united train_labels.csv train