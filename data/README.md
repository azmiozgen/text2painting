* Make main data directory with 
    
    `bash run.sh verified`

    This will execute all steps 1-4 with labels of `deviantart_verified` and `wikiart_verified`, and create `verified` directory with training and validation labels and data stats that will be used in training.

* To change subset lists (deviantart_verified, wikiart_verified etc.), modify related line in `1-unite_labels.csv`

* `deviantart` and `wikiart` directories contain larger versions of labels and some analysis codes. But labels in these directories are not verified.

* `config.py` contains some `word2vec` and image parameters for data preparation.