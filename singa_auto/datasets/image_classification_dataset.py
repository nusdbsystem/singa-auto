import random
import os
import traceback
import zipfile
import tempfile
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from singa_auto.datasets.dataset_base import _load_pil_image, ClfModelDataset
import pandas as pd


class ImageDataset4Clf(ClfModelDataset):
    '''
    Class that helps loading of dataset for task ``IMAGE_CLASSIFICATION``.
    ``classes`` is the number of image classes.
    Each dataset example is (image, class) where:
        - Each image is a 3D numpy array (width x height x channels)
        - Each class is an integer from 0 to (k - 1)
    '''

    def __init__(self,
                 dataset_path: str,
                 min_image_size=None,
                 max_image_size=None,
                 mode='RGB',
                 if_shuffle=False):
        self.mode = mode
        self.path = dataset_path
        self.dataset_zipfile = None
        (self._image_names, self._image_classes, self.size, self.classes) = self._extract_zip(self.path)
        self.min_image_size = min_image_size
        self.max_image_size = max_image_size
        self.label_mapper = dict()
        self.image_size = None
        if if_shuffle:
            (self._image_names,
             self._image_classes) = self._shuffle(self._image_names,
                                                  self._image_classes)

    def __getitem__(self, index):
        if index >= self.size:
            raise StopIteration
        try:
            pil_image = self._extract_item(item_path=self._image_names[index])
            (image, image_size) = self._preprocess(pil_image,
                                                   self.min_image_size,
                                                   self.max_image_size)
            if self.image_size is None:
                self.image_size = image_size
            image_class = self._image_classes[index]
            return (image, image_class)

        except:
            raise

    def _preprocess(self, pil_image, min_image_size, max_image_size):
        (width, height) = pil_image.size
        # crop rectangular image into square
        left = (width - min(width, height)) / 2
        right = (width + min(width, height)) / 2
        top = (height - min(width, height)) / 2
        bottom = (height + min(width, height)) / 2
        crop_pil_image = pil_image.crop((left, top, right, bottom))

        # Decide on image size, adhering to min/max, making it square and trying not to stretch it
        image_size = max(min([width, height, max_image_size or width]),
                         min_image_size or 0)

        # Resize all images
        images = crop_pil_image.resize([image_size, image_size])

        return (images, image_size)

    def _extract_item(self, item_path):
        # Create temp directory to unzip to extract 1 item
        with tempfile.TemporaryDirectory() as d:
            extracted_item_path = self.dataset_zipfile.extract(item_path,
                                                               path=d)
            pil_image = _load_pil_image(extracted_item_path, mode=self.mode)

        return pil_image

    def _extract_zip(self, dataset_path):
        self.dataset_zipfile = zipfile.ZipFile(dataset_path, 'r')
        if 'images.csv' in self.dataset_zipfile.namelist():
            # Create temp directory to unzip to extract paths/classes/numbers only,
            # no actual images would be extracted
            with tempfile.TemporaryDirectory() as d:
                # obtain csv file
                for fileName in self.dataset_zipfile.namelist():
                    if fileName.endswith('.csv'):
                        # Extract a single csv file from zip
                        images_csv_path = self.dataset_zipfile.extract(fileName,
                                                                       path=d)
                        break
                try:
                    csv = pd.read_csv(images_csv_path)
                    image_classes = csv[csv.columns[1:]]
                    image_paths = csv[csv.columns[0]]
                except:
                    traceback.print_stack()
                    raise
            num_classes = len(csv[csv.columns[1]].unique())
            num_labeled_samples = len(csv[csv.columns[0]].unique())
            image_classes = tuple(np.array(image_classes).squeeze().tolist())
            image_paths = tuple(image_paths)

        else:
            # make image name list and remove dir from list
            image_paths = [
                x for x in self.dataset_zipfile.namelist()
                if x.endswith('/') == False
            ]
            num_labeled_samples = len(image_paths)
            str_labels = [os.path.dirname(x) for x in image_paths]
            self.str_labels_set = list(set(str_labels))
            num_classes = len(self.str_labels_set)
            image_classes = [self.str_labels_set.index(x) for x in str_labels]
        return (image_paths, image_classes, num_labeled_samples, num_classes)

    def _shuffle(self, images, classes):
        zipped = list(zip(images, classes))
        random.shuffle(zipped)
        (images, classes) = zip(*zipped)
        return (images, classes)

    def get_item(self, index):
        return self.__getitem__(index)

    def get_stat(self):
        x = 0
        for i in range(self.size):
            try:
                image = np.asarray(self.get_item(i)[0])
                mu_i = np.mean(image, axis=(0, 1))
                mu_i = np.expand_dims(mu_i, axis=0)

                if i == 0:
                    x = mu_i
                else:
                    x = np.concatenate((x, mu_i), axis=0)
                if i % 10000 == 0:
                    print(i)

            except Exception as e:
                import traceback
                traceback.print_exc()
                pass
        x = x / 255
        mu = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        return mu, std


class TorchImageDataset(torch.utils.data.Dataset):
    """
    A Pytorch-type encapsulation to support training/evaluation
    """

    def __init__(self,
                 dataset: ImageDataset4Clf,
                 image_scale_size,
                 norm_mean,
                 norm_std,
                 is_train=False):
        self.dataset = dataset
        if is_train:
            self._transform = transforms.Compose([
                transforms.Resize((image_scale_size, image_scale_size)),
                #transforms.RandomCrop(crop_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)
            ])
        else:
            self._transform = transforms.Compose([
                transforms.Resize((image_scale_size, image_scale_size)),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)
            ])

        # initialize parameters for Self-paced Learning (SPL) module
        self._scores = np.zeros(self.dataset.size)
        self._loss_threshold = -0.00001
        # No threshold means all data samples are effective
        self._effective_dataset_size = self.dataset.size
        # equivalent mapping in default i.e.
        # 0 - 0
        # 1 - 1
        # ...
        # N - N
        self._indice_mapping = np.linspace(start=0,
                                           stop=self.dataset.size - 1,
                                           num=self.dataset.size).astype(np.int32)

    def __len__(self):
        return self._effective_dataset_size

    def __getitem__(self, idx):
        """
        return datasample by given idx

        parameters:
            idx: integer number in range [0 .. self._effective_data_size - 1]

        returns:
            NOTE: being different from the standard procedure, the function returns
            tuple that contains RAW datasample index [0 .. self.dataset.size - 1] as
            the first element
        """
        # translate the index to raw index in singa-auto dataset
        idx = self._indice_mapping[idx]

        image, image_class = self.dataset.get_item(idx)
        image_class = torch.tensor(image_class)
        if self._transform:
            image = self._transform(image)
        else:
            image = torch.tensor(image)

        return (idx, image, image_class)

    def update_sample_score(self, indices, scores):
        """
        update the scores for datasamples

        parameters:
            indices: RAW indices for self.dataset
            scores: scores for corresponding data samples
        """
        self._scores[indices] = scores

    def update_score_threshold(self, threshold):
        self._loss_threshold = threshold
        effective_data_mask = self._scores > self._loss_threshold

        self._indice_mapping = np.linspace(
            start=0, stop=self.dataset.size - 1,
            num=self.dataset.size)[effective_data_mask].astype(np.int32)

        self._effective_dataset_size = len(self._indice_mapping)
        print("dataset threshold = {}, the effective sized = {}".format(
            threshold, self._effective_dataset_size))
