echo
echo    Subsets are being united..
python 1-unite_labels.py

echo
echo    Train-val-test are being splitted..
python 2-split_labels.py

echo
echo    Word2Vec started..
python 3-word2vec.py united/train_labels.csv united

echo
echo    Image stats are being extracted..
python 4-get_image_stats.py united train_labels.csv train