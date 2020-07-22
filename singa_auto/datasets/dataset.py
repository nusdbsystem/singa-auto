#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
from functools import singledispatch
import PIL
from PIL import Image
import numpy as np
from typing import List, Any
import os
import traceback
import zipfile
import io
import csv
import pandas
import tempfile
from singa_auto.datasets.dataset_base import ModelDataset, _load_pil_image
from singa_auto.error_code import InvalidDatasetFormatException


class DatasetUtils:
    '''
    Collection of utility methods to help with the loading of datasets.

    Usage of these methods are optional.
    In fact, you are encouraged to rely on your preferred ML libraries' dataset loading methods, or hand-roll your own.

    This should NOT be initiailized outside of the module. Instead,
    import the global ``utils`` instance from the module ``singa_auto.model``
    and use ``utils.dataset``.

    For example:

    ::

        from singa_auto.model import utils
        ...
        def train(self, dataset_path, **kwargs):
            ...
            utils.dataset.load_dataset_of_image_files(dataset_path)
            ...
    '''

    def load_dataset_of_corpus(self, dataset_path, tags=None, split_by='\\n'):
        '''
            Loads dataset for the task ``POS_TAGGING``

            :param str dataset_path: File path of the dataset
            :returns: An instance of ``CorpusDataset``.
        '''
        return CorpusDataset(dataset_path, tags or ['tag'], split_by)

    def load_dataset_of_image_files(self,
                                    dataset_path,
                                    min_image_size=None,
                                    max_image_size=None,
                                    mode='RGB',
                                    if_shuffle=False
                                    ):
        '''
            Loads dataset for the task ``IMAGE_CLASSIFICATION``.
            :param str dataset_path: File path of the dataset
            :param int min_image_size: minimum width *and* height to resize all images to
            :param int max_image_size: maximum width *and* height to resize all images to
            :param str mode: Pillow image mode. Refer to https://pillow.readthedocs.io/en/3.1.x/handbook/concepts.html#concept-modes
            :param bool if_shuffle: Whether to shuffle the dataset
            :returns: An instance of ``ImageFilesDataset``
        '''
        from singa_auto.datasets.image_classification_dataset import ImageDataset4Clf
        return ImageDataset4Clf(
             dataset_path,
             min_image_size=min_image_size,
             max_image_size=max_image_size,
             mode=mode,
             if_shuffle=if_shuffle)

    def load_img_detection_datasets(self,
                                    dataset_path,
                                    dataset_name,
                                    filter_classes=None,
                                    annotation_dataset_path=None,
                                    is_train=True):
        '''
            Loads dataset for the task ``IMAGE_DETECTION OR IMAGE_SEGMENTATION``.

            :param str dataset_path: File path of the dataset
            :returns: An instance of ``PennFudanDataset``
        '''

        from singa_auto.datasets.image_detection_dataset import PennFudanDataset, CocoDataset
        if filter_classes is None:
            # default is to detect person
            filter_classes = ["person"]
        if "coco" in dataset_name.lower():
            assert annotation_dataset_path is not None
            return CocoDataset(dataset_path, annotation_dataset_path, filter_classes, dataset_name, is_train=is_train)
        elif dataset_name.lower() == "pennfudan":
            return PennFudanDataset(dataset_path, is_train=is_train)
        else:
            return None

    def load_dataset_of_audio_files(self, dataset_path, dataset_dir):
        '''
            Loads dataset with type `AUDIO_FILES`.

            :param str dataset_uri: URI of the dataset file
            :returns: An instance of ``AudioFilesDataset``.
        '''
        return AudioFilesDataset(dataset_path, dataset_dir)

    def normalize_images(self, images, mean=None, std=None):
        '''
            Normalize all images.

            If mean `mean` and standard deviation `std` are `None`, they will be computed on the images.

            :param images: (N x width x height x channels) array-like of images to resize or list of <class 'PIL.Image.Image'>
            :param float[] mean: Mean for normalization, by channel
            :param float[] std: Standard deviation for normalization, by channel
            :returns: (images, mean, std)
        '''
        if len(images) == 0:
            return (images, mean, std)

        if isinstance(images[0], PIL.Image.Image):
            images = [np.asarray(ele) for ele in images]
        # Convert to [0, 1]
        images = np.asarray(images) / 255

        if mean is None:
            mean = np.mean(images,
                           axis=(0, 1, 2)).tolist()  # shape = (channels,)
        if std is None:
            std = np.std(images, axis=(0, 1, 2)).tolist()  # shape = (channels,)

        # Normalize all images
        images = (images - mean) / std

        return (images, mean, std)

    def transform_images(self, images, image_size=None, mode=None):
        '''
            Resize or convert a list of N images to another size and/or mode

            :param images: (N x width x height x channels) array-like of images to resize
            :param int image_size: width *and* height to resize all images to
            :param str mode: Pillow image mode to convert all images to. Refer to https://pillow.readthedocs.io/en/3.1.x/handbook/concepts.html#concept-modes
            :returns: output images as a (N x width x height x channels) numpy array
        '''
        images = [
            Image.fromarray(np.asarray(x, dtype=np.uint8)) for x in images
        ]

        if image_size is not None:
            images = [x.resize([image_size, image_size]) for x in images]

        if mode is not None:
            images = [x.convert(mode) for x in images]

        return np.asarray([np.asarray(x) for x in images]), images

    def load_images(self, image_paths: List[Any],  mode: str = 'RGB') -> List[PIL.Image.Image]:
        '''
             Loads multiple images from the local filesystem or image bytes.
             Assumes that images are of the same dimensions.

             :param images: Paths to images
             :param str mode: Pillow image mode to convert all images to. Refer to https://pillow.readthedocs.io/en/3.1.x/handbook/concepts.html#concept-modes
             :returns: Pillow images as a (N x width x height x channels) numpy array
         '''

        @singledispatch
        def load(image_path):
            raise Exception("Un-support img type {}".format(image_path.__class__))

        @load.register(str)
        def _(image_path):
            pil_image = _load_pil_image(image_path, mode=mode)
            return pil_image

        @load.register(bytes)
        def _(image_bytes):
            encoded = io.BytesIO(image_bytes)
            pil_image = Image.open(encoded).convert(mode)
            return pil_image

        assert isinstance(image_paths, list)
        pil_images = []
        for image_path in image_paths:
            pil_images.append(load(image_path))
        images = np.array([np.asarray(x) for x in pil_images])

        return pil_images


class CorpusDataset(ModelDataset):
    '''
    Class that helps loading of dataset for task ``POS_TAGGING``.

    ``tags`` is the expected list of tags for each token in the corpus.
    Dataset samples are grouped as sentences by a delimiter token corresponding to ``split_by``.

    ``tag_num_classes`` is a list of <number of classes for a tag>, in the same order as ``tags``.
    Each dataset sample is [[token, <tag_1>, <tag_2>, ..., <tag_k>]] where each token is a string,
    each ``tag_i`` is an integer from 0 to (k_i - 1) as each token's corresponding class for that tag,
    with tags appearing in the same order as ``tags``.
    '''

    def __init__(self, dataset_path, tags, split_by):
        self.tags = tags
        (self.size, self.tag_num_classes, self.max_token_len, self.max_sent_len, self._sents) = \
            self._load(dataset_path, self.tags, split_by)

    def __getitem__(self, index):
        return self._sents[index]

    def _load(self, dataset_path, tags, split_by):
        sents = []
        tag_num_classes = [0 for _ in range(len(tags))]
        max_token_len = 0
        max_sent_len = 0

        with tempfile.TemporaryDirectory() as d:
            dataset_zipfile = zipfile.ZipFile(dataset_path, 'r')
            dataset_zipfile.extractall(path=d)
            dataset_zipfile.close()

            # Read corpus.tsv, read token by token, and merge them into sentences
            corpus_tsv_path = os.path.join(d, 'corpus.tsv')
            try:
                with open(corpus_tsv_path, mode='r') as f:
                    reader = csv.DictReader(f, dialect='excel-tab')

                    # Read full corpus into memory token by token
                    sent = []
                    for row in reader:
                        token = row['token']
                        del row['token']

                        # Start new sentence upon encountering delimiter
                        if token == split_by:
                            sents.append(sent)
                            sent = []
                            continue

                        token_tags = [int(row[x]) for x in tags]
                        sent.append([token, *token_tags])

                        # Maintain max classes of tags
                        tag_num_classes = [
                            max(x + 1, m)
                            for (x, m) in zip(token_tags, tag_num_classes)
                        ]

                        # Maintain max token length
                        max_token_len = max(len(token), max_token_len)

                    # Maintain max sent length
                    max_sent_len = max(len(sent), max_sent_len)

            except:
                traceback.print_stack()
                raise InvalidDatasetFormatException()

        size = len(sents)

        return (size, tag_num_classes, max_token_len, max_sent_len, sents)


class AudioFilesDataset(ModelDataset):
    '''
    Class that helps loading of dataset with type `AUDIO_FILES`

    ``dataset_path`` is the URI to the dataset.
    ``dataset_dir`` is the directory to store the extracted the Audio Files.
    '''

    def __init__(self, dataset_path, dataset_dir):
        self._dataset_dir = dataset_dir
        self.df = self._load(dataset_path)

    def _load(self, dataset_path):
        '''
            Loading the dataset into a pandas dataframe. Called in the class __init__ method.

            :param str dataset_path: URI to the dataset
            :returns: pandas dataframe with three columns: ``wav_filename``, ``wav_filesize`` and ``transcript``
        '''
        dataset_dir = self._dataset_dir

        dataset_zipfile = zipfile.ZipFile(dataset_path, 'r')
        dataset_zipfile.extractall(path=dataset_dir.name)
        dataset_zipfile.close()

        # Read images.csv, and read image paths & classes
        audios_csv_path = os.path.join(dataset_dir.name, 'audios.csv')

        df = pandas.read_csv(audios_csv_path, encoding='utf-8', na_filter=False)

        return df





