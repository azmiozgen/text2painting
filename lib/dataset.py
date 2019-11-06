import random
import sys
from io import BytesIO

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

import glob
import os

from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import normalize, to_tensor

from .utils import ImageUtilities
from .preprocess import pad_image, crop_edges_lr

class TextArtDataLoader(Dataset):
    def __init__(self, subset_list, word2vec_model_file, mode='train'):
        val_ratio = 0.1
        test_ratio = 0.1
        seed = 73
        np.random.seed(seed)
        self.mode = mode

        ## Load Word2Vec model
        self.word2vec_model = Word2Vec.load(word2vec_model_file)

        if not isinstance(subset_list, list):
            subset_list = [subset_list]

        data_dir = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, 'data'))
        label_filename = 'labels.csv'
        label_sentences_filename = 'label_sentences.txt'
        images_dirname = 'images'

        labels_dict = {}
        for subset in subset_list:
            subset_dir = os.path.join(data_dir, subset)
            if not os.path.isdir(subset_dir):
                print(subset_dir, 'not found.')
                continue

            ## Read label_file and get image_filenames
            label_file = os.path.join(subset_dir, label_filename)
            with open(label_file, 'r') as f:
                lines = f.readlines()
            label_lines = list(map(lambda s:s.strip(), lines))

            ## Read label_sentences file
            label_sentences_file = os.path.join(subset_dir, label_sentences_filename)
            with open(label_sentences_file, 'r') as f:
                lines = f.readlines()
            label_sentence_lines = list(map(lambda s:s.strip(), lines))

            assert(len(label_lines) == len(label_sentence_lines))

            ## Make image_file-sentence pairs
            for label_line, label_sentence_line in zip(label_lines, label_sentence_lines):
                image_filename = label_line.split(',')[0]
                image_file = os.path.join(subset_dir, images_dirname, image_filename)
                if os.path.isfile(image_file):
                    labels_dict[image_file] = label_sentence_line.split()

        ## Set subdata(train-test-val) size
        n_image_files = len(labels_dict)
        if self.mode == 'train':
            self.size = int(n_image_files * (1 - (val_ratio + test_ratio)))
        elif self.mode == 'test':
            self.size = int(n_image_files * test_ratio)
        elif self.mode == 'val':
            self.size = int(n_image_files * val_ratio)

        ## Set subdata(train-test-val)
        subkeys = np.random.choice(list(labels_dict.keys()), self.size, replace=False)
        self.labels_dict = {k : labels_dict[k] for k in subkeys}

        ## Set image files list
        self.image_files = list(self.labels_dict.keys())

    def get_word_vector(self, word):
        if self.word2vec_model.wv.vocab.get(word):
            return self.word2vec_model[word]
        else:
            return None

    # def preprocess(self, img):
    #     img_tensor = to_tensor(img)
    #     # img_tensor = normalize(img_tensor, mean=[el.mean() for el in img_tensor], std=[el.std() for el in img_tensor])
    #     return img_tensor

    def load(self, image_file):
        img = Image.open(image_file).convert('RGB')
        img.load()
        return img

    def __getitem__(self, index):
        image_file = self.image_files[index]

        # Load image
        img = self.load(image_file)
        # img = self.preprocess(img)

        ## Get label sentence
        label_sentence = self.labels_dict[image_file]
        word_vectors = []
        for word in label_sentence:
            vector = self.get_word_vector(word)
            if vector is not None:
                word_vectors.append(vector)
        word_vectors = torch.Tensor(word_vectors)

        return img, word_vectors

    def __len__(self):
        return self.size


class AlignCollate(object):
    """Should be a callable (https://docs.python.org/2/library/functions.html#callable), that gets a minibatch
    and returns minibatch."""

    def __init__(self, mode,
                       mean,
                       std,
                       image_size_height,
                       image_size_width,
                       horizontal_flipping=True,
                       random_rotation=True,
                       color_jittering=True,
                       random_grayscale=True,
                       random_channel_swapping=True,
                       random_gamma=True,
                       random_resolution=True):

        self._mode = mode
        assert self._mode in ['train', 'test']

        self.mean = mean
        self.std = std
        self.image_size_height = image_size_height
        self.image_size_width = image_size_width

        self.horizontal_flipping = horizontal_flipping
        self.random_rotation = random_rotation
        self.color_jittering = color_jittering
        self.random_grayscale = random_grayscale
        self.random_channel_swapping = random_channel_swapping
        self.random_gamma = random_gamma
        self.random_resolution = random_resolution

        if self._mode == 'train':
            if self.random_resolution:
                self.random_res = ImageUtilities.image_random_resolution([0.7, 1.3])

            if self.horizontal_flipping:
                self.horizontal_flipper = ImageUtilities.image_random_horizontal_flipper()

            if self.random_rotation:
                self.random_rotator = ImageUtilities.image_random_rotator(interpolation=Image.BILINEAR, random_bg=True)

            if self.color_jittering:
                self.color_jitter = ImageUtilities.image_random_color_jitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15)

            if self.random_gamma:
                self.random_gamma_adjuster = ImageUtilities.image_random_gamma(gamma_range=[0.7, 1.3], gain=1)

            if self.random_channel_swapping:
                self.channel_swapper = ImageUtilities.image_random_channel_swapper(p=0.5)

            if self.random_grayscale:
                self.grayscaler = ImageUtilities.image_random_grayscaler(p=0.5)

        self.resizer = ImageUtilities.image_resizer(self.image_size_height, self.image_size_width)
        self.normalizer = ImageUtilities.image_normalizer(self.mean, self.std)

    def __preprocess(self, image):

        if self._mode == 'train':

            image = image.filter(ImageFilter.GaussianBlur(radius=np.random.rand() * 2.0))
            if np.random.rand() < 0.5:
                image = pad_image(image)
            else:
                image = crop_edges_lr(image)

            if self.random_resolution:
                image = self.random_res(image)

            if self.horizontal_flipping:
                is_flip = random.random() < 0.5
                image = self.horizontal_flipper(image, is_flip)

            if self.random_rotation:
                rot_angle = int(np.random.rand() * 10)
                if np.random.rand() >= 0.5:
                    rot_angle = -1 * rot_angle
                rot_expand = np.random.rand() < 0.5
                image = self.random_rotator(image, rot_angle, rot_expand)

            if self.color_jittering:
                image = self.color_jitter(image)

            if self.random_gamma:
                image = self.random_gamma_adjuster(image)

            if self.random_channel_swapping:
                image = self.channel_swapper(image)

            if self.random_grayscale:
                image = self.grayscaler(image)

        image = self.resizer(image)
        image = self.normalizer(image)

        return image

    def __call__(self, batch):
        images = []
        word_vectors_list = []
        max_sentence_length = 0
        for item in batch:
            img = item[0]
            word_vectors = item[1]
            images.append(self.__preprocess(img))
            word_vectors_list.append(word_vectors)

            ## Find max sentence length
            if len(word_vectors) > max_sentence_length:
                max_sentence_length = len(word_vectors)

        ## Equalize in-batch word vector lengths
        for i, word_vectors in enumerate(word_vectors_list):
            padded_wv = torch.Tensor(np.pad(word_vectors, ((0, max_sentence_length - len(word_vectors)), (0, 0))))
            word_vectors_list[i] = padded_wv

        images = torch.stack(images)
        word_vectors_tensor = torch.stack(word_vectors_list)

        return images, word_vectors_tensor

if __name__ == "__main__":

    WORD2VEC_MODEL_FILE = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, 'models', 'deviant_wiki_word2vec.model'))
    
    ## Test TextArtDataLoader
    train_dataset = TextArtDataLoader(['wikiart', 'deviantart'], word2vec_model_file=WORD2VEC_MODEL_FILE, mode='train')
    print("Size:", len(train_dataset))
    print(train_dataset[0])