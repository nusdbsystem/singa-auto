import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np

class PandaTorchImageDataset(Dataset):
    """
    A Pytorch-type encapsulation for rafiki ImageFilesDataset to support training/evaluation
    """
    def __init__(self, rafiki_dataset, image_scale_size, norm_mean, norm_std, is_train=False):
        self.rafiki_dataset = rafiki_dataset
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
        self.zero_sample_score()
        self._loss_threshold = -0.00001
        # No threshold means all data samples are effective
        self._effective_dataset_size = self.rafiki_dataset.size
        # equivalent mapping in default i.e.
        # 0 - 0
        # 1 - 1
        # ...
        # N - N
        self._indice_mapping = np.linspace(
            start=0, 
            stop=self.rafiki_dataset.size - 1, 
            num=self.rafiki_dataset.size).astype(np.int32)

    def __len__(self):
        return self._effective_dataset_size

    def __getitem__(self, idx):
        """
        return datasample by given idx
        
        parameters:
            idx: integer number in range [0 .. self._effective_data_size - 1]

        returns:
            NOTE: being different from the standard procedure, the function returns
            tuple that contains RAW datasample index [0 .. self.rafiki_dataset.size - 1] as 
            the first element
        """
        # translate the index to raw index in rafiki dataset
        idx = self._indice_mapping[idx]

        image, image_class = self.rafiki_dataset.get_item(idx)
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
            indices: RAW indices for self.rafiki_dataset
            scores: scores for corresponding data samples
        """
        self._scores[indices] = scores

    def zero_sample_score(self):
        self._scores = np.zeros(self.rafiki_dataset.size)

    def update_score_threshold(self, threshold):
        self._loss_threshold = threshold
        effective_data_mask = self._scores > self._loss_threshold
        self._indice_mapping = np.linspace(
            start=0, 
            stop=self.rafiki_dataset.size - 1, 
            num=self.rafiki_dataset.size)[effective_data_mask].astype(np.int32)
        self._effective_dataset_size = len(self._indice_mapping)
        print("dataset threshold = {}, the effective sized = {}".format(threshold, self._effective_dataset_size))
        


        
