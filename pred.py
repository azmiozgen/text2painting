import argparse
import os
import random
import sys
import time

from gensim.models import Word2Vec
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lib.config import Config

def get_word_vector(word2vec_model, word):
    if word2vec_model.wv.vocab.get(word):
        return word2vec_model.wv[word]
    else:
        return None

def check_vector_in_vocab(word2vec_model, word_vector, min_similarity_prob):
    word_vector = np.array(word_vector)
    word, prob = word2vec_model.wv.similar_by_vector(word_vector, topn=1)[0]
    return prob > min_similarity_prob

def clean_wvs(word2vec_model, word_vectors, min_similarity_prob):
    indices = []
    for i, wv in enumerate(word_vectors):
        if check_vector_in_vocab(word2vec_model, wv, min_similarity_prob):
            indices.append(i)
    return word_vectors[indices]

def normalize_wvs(wv_tensor):
    shifted = wv_tensor - wv_tensor.min()
    norm = shifted / ((shifted).max() + 1e-10)
    return norm * 2 - 1

def get_similar_wv_by_vector(word2vec_model, word_vector, word_vectors_similar_pad_topN):
    word_vector = np.array(word_vector)
    _top_similar_words = word2vec_model.wv.similar_by_vector(word_vector, 
                                                             topn=word_vectors_similar_pad_topN + 1)
    top_similar_words = np.array(_top_similar_words)[1:]   ## Do not take word itself (the most similar)
    try:
        similar_word = np.random.choice(top_similar_words[:, 0], 1, replace=False)[0]
    except ValueError:
        similar_word = np.random.choice(top_similar_words[:, 0], 1, replace=True)[0]

    return get_word_vector(word2vec_model, similar_word)

def pad_wvs_with_similar_wvs(word2vec_model, word_vectors, final_length, word_vectors_similar_pad_topN):
    while len(word_vectors) < final_length:
        word_vector = random.choice(word_vectors)
        similar_vector = torch.Tensor(get_similar_wv_by_vector(word2vec_model, word_vector, word_vectors_similar_pad_topN)).unsqueeze(0)
        word_vectors = torch.cat((word_vectors, similar_vector))

    return word_vectors

def pad_wvs_with_noise(word_vectors, final_length):
    pad_length = final_length - len(word_vectors)
    padding_tensor = torch.rand(pad_length, word_vectors.size()[1]) * 2 - 1   ## [-1, 1]
    word_vectors = torch.cat((word_vectors, padding_tensor))

    return word_vectors

def crop_wvs(word_vectors, final_length):
    return word_vectors[torch.randperm(final_length)]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model file to load')
    parser.add_argument('--input', type=str, required=True, help='Input file to load')
    parser.add_argument('--output', type=str, help='Output file to save')
    args = parser.parse_args()
    model_file = args.model
    input_file = args.input
    output_filename = args.output

    CONFIG = Config()
    # CONFIG.DEVICE = torch.device('cpu')

    ## Import model
    model_dir = os.path.dirname(os.path.abspath(model_file))
    model_lib_dir = os.path.join(model_dir, 'lib')
    sys.path.append(model_lib_dir)
    from model import GANModel

    ## Init model with G
    start = time.time()
    print("\nModel initializing..")
    model = GANModel(CONFIG, model_file=model_file, mode='test', reset_lr=False)
    model.G.eval()
    model.G_refiner.eval()
    model.G_refiner2.eval()
    train_G = False

    ## Read input file
    with open(input_file, 'r') as f:
        lines = f.readlines()

    ## Get Word2Vec model
    word2vec_model_file = CONFIG.WORD2VEC_MODEL_FILE
    word2vec_model = Word2Vec.load(word2vec_model_file)

    word_vectors_list = []
    for i, line in enumerate(lines):
        label_sentence = line.split(',')
        word_vectors = []
        for word in label_sentence: 
            vector = get_word_vector(word2vec_model, word)
            if vector is not None:
                word_vectors.append(vector)

        ## If empty, make noisy vector (-1, 1)
        if len(word_vectors) == 0:
            word_vectors = torch.rand(1, word2vec_model.vector_size) * 2.0 - 1.0
        else:
            word_vectors = torch.Tensor(word_vectors)

        word_vectors = normalize_wvs(word_vectors)

        ## Pad or crop true wvs
        if len(word_vectors) < CONFIG.SENTENCE_LENGTH:
            word_vectors = pad_wvs_with_similar_wvs(word2vec_model, word_vectors, CONFIG.SENTENCE_LENGTH, CONFIG.WORD_VECTORS_SIMILAR_PAD_TOPN)
        else:
            word_vectors = crop_wvs(word_vectors, CONFIG.SENTENCE_LENGTH)

        ## Add noise
        if CONFIG.NOISE_LENGTH > 0:
            word_vectors = pad_wvs_with_noise(word_vectors, CONFIG.SENTENCE_LENGTH + CONFIG.NOISE_LENGTH)

        word_vectors_list.append(word_vectors)

    wvs_tensor = torch.stack(word_vectors_list)
    batch_size = wvs_tensor.size()[0]

    ## Forward G
    wvs_flat = wvs_tensor.view(batch_size, -1)
    fake_images = model.forward(model.G, wvs_flat)

    ## Forward G_refiner
    refined1 = model.forward(model.G_refiner, fake_images)
    
    ## Forward G_refiner2
    refined2 = model.forward(model.G_refiner2, refined1)
    print("Output is generated")

    if not output_filename:
        output_grid_filename = "detailed_output.png"
        output_simple_grid_filename = "simple_output.png"
        output_vanilla_filename = "output.png"
    grid_img_pil = model.generate_grid(wvs_tensor.clone(), fake_images.clone(), refined1.clone(), refined2.clone(), refined2.clone(), word2vec_model)
    grid_simple_img_pil = model.generate_grid_simple(wvs_tensor.clone(), refined2.clone(), word2vec_model)
    
    model.save_img_test_grid(grid_img_pil, output_grid_filename, verbose=True)
    model.save_img_test_grid(grid_simple_img_pil, output_simple_grid_filename, verbose=True)
    model.save_img_test_single(refined2.clone(), output_vanilla_filename, kind='refined2', verbose=True)

