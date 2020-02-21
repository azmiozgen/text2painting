import glob
import os
import random
import sys
import time
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter, ImageOps

import pandas as pd
import torch
from gensim.models import Word2Vec
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision.transforms.functional import normalize, to_tensor

from lib.preprocess import crop_edges_lr, pad_image
from lib.utils import ImageUtilities


class TextArtDataLoader(Dataset):
    def __init__(self, config, kind='train'):

        assert kind in ['train', 'val', 'test']
        self.kind = kind

        self.word2vec_model_file = config.WORD2VEC_MODEL_FILE
        self.load_word_vectors = config.LOAD_WORD_VECTORS

        ## Load Word2Vec model
        self.word2vec_model = Word2Vec.load(self.word2vec_model_file)

        ## Load all word vectors (Not recommended if low on memory)
        if self.load_word_vectors:
            self.word_vectors_dict = {}
            for word, _ in self.word2vec_model.wv.vocab.items():
                self.word_vectors_dict[word] = self.word2vec_model.wv[word]
        else:
            self.word_vectors_dict = None

        data_dir = config.DATA_DIR
        label_filename = '{}_labels.csv'.format(self.kind)
        label_file = os.path.join(data_dir, label_filename)

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
            if self.word_vectors_dict:
                return self.word_vectors_dict[word]
            else:
                return self.word2vec_model.wv[word]
        else:
            return None

    def load(self, image_file):
        img = Image.open(image_file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

    def __getitem__(self, index):
        image_file = self.image_files[index]

        # Load image
        img = self.load(image_file)

        ## Get label sentence
        label_sentence = self.labels_dict[image_file]
        word_vectors = []

        for word in label_sentence: 
            vector = self.get_word_vector(word)
            if vector is not None:
                word_vectors.append(vector)

        ## If empty, make noisy vector (-1, 1)
        if len(word_vectors) == 0:
            word_vectors_tensor = torch.rand(1, self.word2vec_model.vector_size) * 2.0 - 1.0
        else:
            word_vectors_tensor = torch.Tensor(word_vectors)

        return img, word_vectors_tensor

    def __len__(self):
        return len(self.image_files)


class AlignCollate(object):
    """Should be a callable (https://docs.python.org/2/library/functions.html#callable), that gets a minibatch
    and returns minibatch."""

    def __init__(self, config, mode='train'):

        self.mode = mode
        assert self.mode in ['train', 'test']

        self.config = config
        self.word2vec_model_file = config.WORD2VEC_MODEL_FILE
        self.normalize = config.NORMALIZE
        self.mean = config.MEAN
        self.std = config.STD
        self.image_width_first = config.IMAGE_WIDTH_FIRST
        self.image_height_first = config.IMAGE_HEIGHT_FIRST
        self.image_width_second = config.IMAGE_WIDTH_SECOND
        self.image_height_second = config.IMAGE_HEIGHT_SECOND
        self.image_width = config.IMAGE_WIDTH
        self.image_height = config.IMAGE_HEIGHT
        self.random_blurriness = config.RANDOM_BLURRINESS
        self.horizontal_flipping = config.HORIZONTAL_FLIPPING
        self.random_rotation = config.RANDOM_ROTATION
        self.color_jittering = config.COLOR_JITTERING
        self.random_grayscale = config.RANDOM_GRAYSCALE
        self.random_channel_swapping = config.RANDOM_CHANNEL_SWAPPING
        self.random_gamma = config.RANDOM_GAMMA
        self.random_resolution = config.RANDOM_RESOLUTION
        self.elastic_deformation = config.ELASTIC_DEFORMATION
        self.sharpening = config.SHARPENING
        self.equalizing = config.EQUALIZING

        self.sentence_length = config.SENTENCE_LENGTH
        self.noise_length = config.NOISE_LENGTH

        ## Load Word2Vec model
        self.word2vec_model = Word2Vec.load(self.word2vec_model_file)
        self.word_vectors_similar_pad = config.WORD_VECTORS_SIMILAR_PAD
        self.word_vectors_similar_pad_topN = config.WORD_VECTORS_SIMILAR_PAD_TOPN
        self.word_vectors_similar_take_self = config.WORD_VECTORS_SIMILAR_TAKE_SELF
        self.word_vectors_dissimilar_topN = config.WORD_VECTORS_DISSIMILAR_TOPN

        if self.mode == 'train':
            if self.random_resolution:
                self.random_res = ImageUtilities.image_random_resolution([0.75, 1.25])

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

            if self.elastic_deformation:
                self.elastic_deformer = ImageUtilities.image_elastic_deformer(alpha_range=[500, 1300], sigma_range=[9, 10])

        self.resizer_first = ImageUtilities.image_resizer(self.image_height_first, self.image_width_first)
        self.resizer_second = ImageUtilities.image_resizer(self.image_height_second, self.image_width_second)
        self.resizer = ImageUtilities.image_resizer(self.image_height, self.image_width)
        self.normalizer = ImageUtilities.image_normalizer(self.mean, self.std)

    def __preprocess(self, image, stage):
        assert stage in [1, 2, 3], 'Wrong stage number {}. Choose among [1, 2, 3]'.format(stage)
        ## 'stage' for image resizing for the levels of GAN (first stage 64x64, second stage is 128x128, third stage is 256x256)

        if self.mode == 'train':

            if self.random_blurriness:
                image = image.filter(ImageFilter.GaussianBlur(radius=np.random.rand() * 3.0))

            # ## Pad or crop_lr image with low prob.
            # if np.random.rand() < 0.005:
            #     image = pad_image(image)
            # elif np.random.rand() < 0.01:
            #     image = crop_edges_lr(image)

            if self.random_resolution:
                image = self.random_res(image)

            if self.horizontal_flipping:
                is_flip = random.random() < 0.1
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

            if self.elastic_deformation:
                image = self.elastic_deformer(image)

            if self.sharpening:
                if random.random() < 0.2:
                    image = image.filter(ImageFilter.SHARPEN)

            if self.equalizing:
                if random.random() < 0.2:
                    image = ImageOps.equalize(image, mask=None)

        if stage == 1:
            image = self.resizer_first(image)
        elif stage == 2:
            image = self.resizer_second(image)
        elif stage == 3:
            image = self.resizer(image)

        if self.normalize:
            image = self.normalizer(image)  ## Does 'to_tensor'
        else:
            image = to_tensor(image)

        return image

    def _normalize_wvs(self, wv_tensor):
        '''
        Normalize word vectors into [-1, 1]
        '''
        shifted = wv_tensor - wv_tensor.min()
        norm = shifted / ((shifted).max() + 1e-10)
        return norm * 2 - 1

    def _get_word_vector(self, word):
        if self.word2vec_model.wv.vocab.get(word):
            return self.word2vec_model.wv[word]
        else:
            return None

    def _get_word_by_vector(self, word_vector):
        word_vector = np.array(word_vector)
        word, prob = self.word2vec_model.wv.similar_by_vector(word_vector, topn=1)[0]
        if prob > self.config.MIN_WV_SIMILARITY_PROB:
            return word
        else:
            return None

    def _check_vector_in_vocab(self, word_vector):
        word_vector = np.array(word_vector)
        word, prob = self.word2vec_model.wv.similar_by_vector(word_vector, topn=1)[0]
        return prob > self.config.MIN_WV_SIMILARITY_PROB

    def _clean_wvs(self, word_vectors):
        '''
        Clean word vectors by removing words that are not in vocabulary
        '''
        indices = []
        for i, wv in enumerate(word_vectors):
            if self._check_vector_in_vocab(wv):
                indices.append(i)
        return word_vectors[indices]
        # return torch.Tensor(list(filter(self._check_vector_in_vocab, np.array(word_vectors))))

    def _get_similar_wv_by_vector(self, word_vector):
        word_vector = np.array(word_vector)
        _top_similar_words = self.word2vec_model.wv.similar_by_vector(word_vector, 
                                                                      topn=self.word_vectors_similar_pad_topN + 1)
        if self.word_vectors_similar_take_self:
            top_similar_words = np.array(_top_similar_words)
        else:
            top_similar_words = np.array(_top_similar_words)[1:]   ## Do not take word itself (the most similar)
        try:
            similar_word = np.random.choice(top_similar_words[:, 0], 1, replace=False)[0]
        except ValueError:
            similar_word = np.random.choice(top_similar_words[:, 0], 1, replace=True)[0]

        return self._get_word_vector(similar_word)


    def _pad_wvs_with_similar_wvs(self, word_vectors, final_length):
        '''
        Pad word vectors array with random similar word vectors
        '''
        while len(word_vectors) < final_length:
            word_vector = random.choice(word_vectors)
            similar_vector = torch.Tensor(self._get_similar_wv_by_vector(word_vector)).unsqueeze(0)
            word_vectors = torch.cat((word_vectors, similar_vector))

        return word_vectors


    def _pad_wvs_with_noise(self, word_vectors, final_length):
        '''
        Pad word vectors array with random noise
        '''
        pad_length = final_length - len(word_vectors)
        padding_tensor = torch.rand(pad_length, word_vectors.size()[1]) * 2 - 1   ## [-1, 1]
        word_vectors = torch.cat((word_vectors, padding_tensor))

        return word_vectors

    def _crop_wvs(self, word_vectors, final_length):
        '''
        Crop word vectors randomly to be equal to sentence length
        '''
        return word_vectors[torch.randperm(final_length)]

    def _get_dissimilar_wv_by_vector(self, word_vector):
        word_vector = np.array(word_vector)
        topN = self.word_vectors_dissimilar_topN
        _top_dissimilar_words = self.word2vec_model.wv.similar_by_vector(word_vector, 
                                                                         topn=100000000)[-topN:]
        top_dissimilar_words = np.array(_top_dissimilar_words)
        try:
            dissimilar_word = np.random.choice(top_dissimilar_words[:, 0], 1, replace=False)[0]
        except ValueError:
            dissimilar_word = np.random.choice(top_dissimilar_words[:, 0], 1, replace=True)[0]

        return self._get_word_vector(dissimilar_word)

    def _generate_dissimilar_wvs(self, word_vectors):
        fake_word_vectors = []
        for wv in word_vectors:
            fake_wv = self._get_dissimilar_wv_by_vector(wv)
            fake_word_vectors.append(fake_wv)
        
        return fake_word_vectors

    def __call__(self, batch):
        images_first = []
        images_second = []
        images = []
        word_vectors_list = []
        fake_word_vectors_list = []
        for item in batch:
            img = item[0]
            word_vectors = item[1]
            images_first.append(self.__preprocess(img, stage=1))
            images_second.append(self.__preprocess(img, stage=2))
            images.append(self.__preprocess(img, stage=3))

            ## Clean wvs that are not in vocab
            word_vectors = self._clean_wvs(word_vectors)

            if len(word_vectors) == 0:
                word_vectors = torch.rand(1, self.word2vec_model.vector_size) * 2.0 - 1.0
            word_vectors = self._normalize_wvs(word_vectors)

            ## Pad or crop true wvs
            if len(word_vectors) < self.sentence_length:
                if self.word_vectors_similar_pad:
                    word_vectors = self._pad_wvs_with_similar_wvs(word_vectors, self.sentence_length)
                else:
                    word_vectors = self._pad_wvs_with_noise(word_vectors, self.sentence_length)
            else:
                word_vectors = self._crop_wvs(word_vectors, self.sentence_length)

            ## Add noise
            if self.noise_length > 0:
                word_vectors = self._pad_wvs_with_noise(word_vectors, self.sentence_length + self.noise_length)

            word_vectors_list.append(word_vectors)

            ## Get fake wvs
            fake_word_vectors = torch.Tensor(self._generate_dissimilar_wvs(word_vectors))
            fake_word_vectors_list.append(fake_word_vectors)

        images_first_tensor = torch.stack(images_first)
        images_second_tensor = torch.stack(images_second)
        images_tensor = torch.stack(images)
        word_vectors_tensor = torch.stack(word_vectors_list)
        fake_word_vectors_tensor = torch.stack(fake_word_vectors_list)

        return images_first_tensor, images_second_tensor, images_tensor, word_vectors_tensor, fake_word_vectors_tensor


class ImageBatchSamplerAlt(Sampler):
    '''
        Group image files by their groups.
    '''

    def __init__(self, config, kind='train'):

        assert kind in ['train', 'val', 'test']

        self.config = config
        self.batch_size = config.BATCH_SIZE
        self.shuffle_groups = config.SHUFFLE_GROUPS
        width_ranges = config.GROUP_WIDTH_RANGES
        height_ranges = config.GROUP_HEIGHT_RANGES

        data_dir = config.DATA_DIR
        labels_filename = '{}_labels.csv'.format(kind)
        labels_file = os.path.join(data_dir, labels_filename)
        shapes_filename = '{}_image_shapes.csv'.format(kind)
        shapes_file = os.path.join(data_dir, shapes_filename)

        ## Read labels file
        with open(labels_file, 'r') as f:
            lines = f.readlines()
        label_lines = list(map(lambda s:s.strip(), lines))

        ## Read shapes file
        shape_lines = np.loadtxt(shapes_file, delimiter=',', dtype=str)

        ## Create image_file, group df
        image_n_groups = []
        for label_line in label_lines:
            image_relative_file = label_line.split(',')[0]
            group = image_relative_file.split('/')[1].split('_')[-1]
            image_n_groups.append([image_relative_file, group])
        image_n_groups = np.array(image_n_groups)
        df_image_n_groups = pd.DataFrame(image_n_groups, columns=['image_file', 'groups'])

        ## Create image_file, shapes df
        df_image_shapes = pd.DataFrame(shape_lines, columns=['image_file', 'width', 'height'])

        ## Merge two df on image files
        self.df = pd.merge(df_image_shapes, df_image_n_groups, on='image_file')
        self.df = self.df.astype({'image_file': str, 'width': int, 'height': int, 'groups': int})
        self.df['index'] = self.df.index

        ## Group batches
        self.groups = []
        df_groups_grouped = self.df.groupby(by=self.df['groups'])

        ## Group by groups
        for key1, _ in df_groups_grouped:
            df1 = df_groups_grouped.get_group(key1)
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
                # print(sample)
                sample_index = sample[-1]    ## Index is last column in df
                batch.append(sample_index)
                if len(batch) == self.batch_size:  ## Extend only if length equals batch size (drop lasts)
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

class ImageBatchSampler(Sampler):
    '''
        Group image files by their image sizes # of labels and sample similar from images.
    '''

    def __init__(self, config, kind='train'):

        assert kind in ['train', 'test']

        self.config = config
        self.batch_size = config.BATCH_SIZE
        self.shuffle_groups = config.SHUFFLE_GROUPS
        n_labels_ranges = config.GROUP_N_LABELS_RANGES
        width_ranges = config.GROUP_WIDTH_RANGES
        height_ranges = config.GROUP_HEIGHT_RANGES

        data_dir = config.DATA_DIR
        labels_filename = '{}_labels.csv'.format(kind)
        labels_file = os.path.join(data_dir, labels_filename)
        shapes_filename = '{}_image_shapes.csv'.format(kind)
        shapes_file = os.path.join(data_dir, shapes_filename)

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

    from config import Config
    CONFIG = Config()
    DATA_DIR = 'united'
    
    ## Test TextArtDataLoader
    train_dataset = TextArtDataLoader(DATA_DIR, CONFIG, kind='train')
    print("Size:", len(train_dataset))
    print(train_dataset[0])
