import glob
import os
import random
import sys
import time
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter

import pandas as pd
import torch
from gensim.models import Word2Vec
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision.transforms.functional import normalize, to_tensor

from .preprocess import crop_edges_lr, pad_image
from .utils import ImageUtilities


class TextArtDataLoader(Dataset):
    def __init__(self, subset, word2vec_model_file, mode='train'):

        self.mode = mode

        ## Load Word2Vec model
        self.word2vec_model = Word2Vec.load(word2vec_model_file)

        data_dir = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, 'data'))
        subset_dir = os.path.join(data_dir, subset)
        label_filename = '{}_labels.csv'.format(self.mode)
        label_file = os.path.join(subset_dir, label_filename)

        if not os.path.isfile(label_file):
            print(label_file, "not found. Exiting.")
            exit()

        ## Read label_file
        with open(label_file, 'r') as f:
            lines = f.readlines()
        label_lines = list(map(lambda s:s.strip(), lines))

        ## Make image_file-sentence pairs
        self.labels_dict = {}
        for label_line in label_lines:
            image_relative_file = label_line.split(',')[0]
            image_file = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, image_relative_file))

            if os.path.isfile(image_file):
                labels = label_line.split(',')[1:]
                self.labels_dict[image_file] = labels

        ## Set all image files list
        self.image_files = list(self.labels_dict.keys())

    def get_word_vector(self, word):
        if self.word2vec_model.wv.vocab.get(word):
            return self.word2vec_model.wv[word]
        else:
            return None

    # def preprocess(self, img):
    #     img_tensor = to_tensor(img)
    #     # img_tensor = normalize(img_tensor, mean=[el.mean() for el in img_tensor], std=[el.std() for el in img_tensor])
    #     return img_tensor

    def load(self, image_file):
        img = Image.open(image_file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

    def __getitem__(self, index):
        image_file = self.image_files[index]

        #####start = time.time()
        # Load image
        img = self.load(image_file)
        # img = self.preprocess(img)
        #####print("Image load: {:.4f}".format(time.time() - start))

        ## Get label sentence
        label_sentence = self.labels_dict[image_file]
        word_vectors = []

        #####start = time.time()
        for word in label_sentence:
            vector = self.get_word_vector(word)
            if vector is not None:
                word_vectors.append(vector)
        #####print("Get word vector: {:.4f}".format(time.time() - start))

        ## If empty, make zero vector
        if len(word_vectors) == 0:
            word_vectors = torch.zeros(1, self.word2vec_model.vector_size)
        else:
            word_vectors = torch.Tensor(word_vectors)

        return img, word_vectors

    def __len__(self):
        return len(self.image_files)


class AlignCollate(object):
    """Should be a callable (https://docs.python.org/2/library/functions.html#callable), that gets a minibatch
    and returns minibatch."""

    def __init__(self, mode,
                       word2vec_model_file,
                       mean,
                       std,
                       image_size_height,
                       image_size_width,
                       horizontal_flipping=False,
                       random_rotation=False,
                       color_jittering=False,
                       random_grayscale=False,
                       random_channel_swapping=False,
                       random_gamma=False,
                       random_resolution=False,
                       word_vectors_similar_pad=True,
                       word_vectors_similar_pad_topN=10):

        self._mode = mode
        assert self._mode in ['train', 'val']

        ## Load Word2Vec model
        self.word2vec_model = Word2Vec.load(word2vec_model_file)
        self.word_vectors_similar_pad = word_vectors_similar_pad
        self.word_vectors_similar_pad_topN = word_vectors_similar_pad_topN

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

            image = image.filter(ImageFilter.GaussianBlur(radius=np.random.rand() * 1.5))

            ## Pad or crop_lr image with low prob.
            if np.random.rand() < 0.05:
                image = pad_image(image)
            elif np.random.rand() < 0.1:
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
        image = to_tensor(image)
        #image = self.normalizer(image)

        return image

    def _get_word_vector(self, word):
        if self.word2vec_model.wv.vocab.get(word):
            return self.word2vec_model.wv[word]
        else:
            return None

    def _get_similar_wv_by_vector(self, word_vector):
        word_vector = np.array(word_vector)
        _top_similar_words = self.word2vec_model.wv.similar_by_vector(word_vector, topn=self.word_vectors_similar_pad_topN)
        top_similar_words = np.array(_top_similar_words)
        try:
            similar_word = np.random.choice(top_similar_words[:, 0], 1, replace=False)[0]
        except ValueError:
            similar_word = np.random.choice(top_similar_words[:, 0], 1, replace=True)[0]

        return self._get_word_vector(similar_word)

    def _pad_wvs_with_similar_wvs(self, word_vectors, pad_length):
        '''
        Equalize in-batch word vector lengths with random similar words
        '''
        final_length = len(word_vectors) + pad_length
        while len(word_vectors) < final_length:
            word_vector = random.choice(word_vectors)
            similar_vector = torch.Tensor(self._get_similar_wv_by_vector(word_vector)).unsqueeze(0)
            word_vectors = torch.cat((word_vectors, similar_vector))

        return word_vectors
    
    def _pad_wvs_with_noise(self, word_vectors, pad_length):
        '''
        Equalize in-batch word vector lengths with random noise 
        '''
        padding_tensor = torch.rand(pad_length, word_vectors.size()[1]) * 2 - 1   ## [-1, 1]
        word_vectors = torch.cat((word_vectors, padding_tensor))

        return word_vectors


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

        for i, word_vectors in enumerate(word_vectors_list):
            pad_length = max_sentence_length - len(word_vectors)
            if self.word_vectors_similar_pad:
                word_vectors_list[i] = self._pad_wvs_with_similar_wvs(word_vectors, pad_length)
            else:
                word_vectors_list[i] = self._pad_wvs_with_noise(word_vectors, pad_length)

        images = torch.stack(images)
        word_vectors_tensor = torch.stack(word_vectors_list)

        return images, word_vectors_tensor


class ImageBatchSampler(Sampler):
    '''
        Group image files by their image sizes # of labels and sample similar from images.
    '''

    def __init__(self, subset, batch_size, shuffle_groups=True, mode='train'):


        self.batch_size = batch_size
        self.shuffle_groups = shuffle_groups

        ## Grouping ranges
        n_labels_ranges = [-1, 5, 7, 11, 1000]
        width_ranges = [-1, 500, 700, 1000, 100000]
        height_ranges = [-1, 590, 100000]

        data_dir = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, 'data'))
        subset_dir = os.path.join(data_dir, subset)
        labels_filename = '{}_labels.csv'.format(mode)
        labels_file = os.path.join(subset_dir, labels_filename)
        shapes_filename = '{}_image_shapes.csv'.format(mode)
        shapes_file = os.path.join(subset_dir, shapes_filename)

        ## Read labels file
        with open(labels_file, 'r') as f:
            lines = f.readlines()
        label_lines = list(map(lambda s:s.strip(), lines))

        ## Read shapes file
        shape_lines = np.loadtxt(shapes_file, delimiter=',', dtype=str)

        ## Create image_file, n_labels df
        image_n_labels = []
        for label_line in label_lines:
            image_relative_file = label_line.split(',')[0]
            labels = label_line.split(',')[1:]
            n_labels = len(labels)
            image_n_labels.append([image_relative_file, n_labels])
        image_n_labels = np.array(image_n_labels)
        df_image_n_labels = pd.DataFrame(image_n_labels, columns=['image_file', 'n_labels'])

        ## Create image_file, shapes df
        df_image_shapes = pd.DataFrame(shape_lines, columns=['image_file', 'width', 'height'])

        ## Merge two df on image files
        self.df = pd.merge(df_image_shapes, df_image_n_labels, on='image_file')
        self.df = self.df.astype({'image_file': str, 'width': int, 'height': int, 'n_labels': int})
        self.df['index'] = self.df.index

        ## Group batches
        self.groups = []
        df_n_labels_grouped = self.df.groupby(by=pd.cut(self.df['n_labels'], n_labels_ranges))

        ## Group by labels
        for key1, _ in df_n_labels_grouped:
            df1 = df_n_labels_grouped.get_group(key1)
            df_width_grouped = df1.groupby(by=pd.cut(df1['width'], width_ranges))

            ## Group by widths
            for key2, _ in df_width_grouped:
                df2 = df_width_grouped.get_group(key2)
                df_height_grouped = df2.groupby(by=pd.cut(df2['height'], height_ranges))

                ## Group by heights
                for key3, _ in df_height_grouped:
                    group_df = df_height_grouped.get_group(key3)
                    self.groups.append(group_df)

    def _group_batches(self):
        sample_indexes = []
        n_batches = len(self.df.index) // self.batch_size
        group_index = 0
        while n_batches > 0:
            ## If lasts are dropped groups may end
            if group_index == len(self.groups):
                break
            group = np.array(self.groups[group_index])
            if self.shuffle_groups:
                np.random.shuffle(group)
            batch = []
            for sample in group:
                sample_index = sample[-1]    ## Index is last column in df
                batch.append(sample_index)
                if len(batch) == self.batch_size:
                    sample_indexes.extend(batch)
                    n_batches -= 1
                    batch = []
                    continue
            group_index += 1
            ## Uncomment not to drop lasts
            # if batch != []:
            #     sample_indexes.extend(batch)
            #     n_batches -= 1
            #     group_index += 1
        return sample_indexes

    def __iter__(self):
        self.grouped_indexes = self._group_batches()
        return iter(self.grouped_indexes)

    def __len__(self):
        return len(self.df.index)

if __name__ == "__main__":

    WORD2VEC_MODEL_FILE = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, 'models', 'deviant_wiki_word2vec.model'))
    
    ## Test TextArtDataLoader
    train_dataset = TextArtDataLoader(['wikiart', 'deviantart'], word2vec_model_file=WORD2VEC_MODEL_FILE, mode='train')
    print("Size:", len(train_dataset))
    print(train_dataset[0])
