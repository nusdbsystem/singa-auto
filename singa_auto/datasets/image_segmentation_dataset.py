from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import os
from copy import deepcopy
from PIL import Image
import torch
from glob import glob


def ImageFetch(train_data_path, split_rate=0.8):
    '''
    load image as PIL.Image into a list for dataloader, split train/val subsets automatically 
    train_data_path: already unzipped dataset folder path
    split_rate: ratio of train/val data
    '''
    folder_name = train_data_path

    image_train = []
    mask_train = []
    image_val = []
    mask_val = []

    # split train and val subsets
    images_folder = os.path.join(folder_name, "image")
    masks_folder = os.path.join(folder_name, "mask")

    if not os.path.isdir(images_folder) or not os.path.isdir(masks_folder):
        print("imges folder or mask folder does not exist, please check the upload file")

    image_list = sorted(glob(os.path.join(images_folder, '*'))) # use sorted list to control train/val split
    num_img = len(image_list)

    train_num = int(num_img * split_rate)
    train_list = image_list[0:train_num]
    val_list = image_list[train_num:]

    # load images and masks from their folders
    for idx, image_name in tqdm(enumerate(train_list), total=len(train_list), desc="load train images......"):
        image_name = image_name.split('/')[-1]

        image_path = os.path.join(images_folder, image_name)
        mask_path = os.path.join(masks_folder, os.path.splitext(image_name)[0] + ".png") # use image name to find the corresponding mask

        image = Image.open(image_path)
        image_train.append(image)

        mask = Image.open(mask_path)
        mask_train.append(mask)

    for idx, image_name in tqdm(enumerate(val_list), total=len(val_list), desc="load validation images......"):
        image_name = image_name.split('/')[-1]

        image_path = os.path.join(images_folder, image_name)
        mask_path = os.path.join(masks_folder, os.path.splitext(image_name)[0] + ".png")

        image = Image.open(image_path)
        image_val.append(image)

        mask = Image.open(mask_path)
        mask_val.append(mask)

    return image_train, mask_train, image_val, mask_val


def trainImageFetch(folder_name):
    '''
    load train image as PIL.Image into a list for dataloader, need train/val subsets split before execution
    folder_name: already unzipped train dataset folder path
    '''
    image_train = []
    mask_train = []

    # load images and masks from their folders
    images_folder = os.path.join(folder_name, "train", "image")
    masks_folder = os.path.join(folder_name, "train", "mask")
    image_list = os.listdir(images_folder)
    for idx, image_name in tqdm(enumerate(image_list), total=len(image_list), desc="load train images......"):
        image_path = os.path.join(images_folder, image_name)
        mask_path = os.path.join(masks_folder, os.path.splitext(image_name)[0] + ".png")

        image = Image.open(image_path)
        image_train.append(image)

        mask = Image.open(mask_path)
        mask_train.append(mask)

    return image_train, mask_train


def valImageFetch(folder_name):
    '''
    load validation image as PIL.Image into a list for dataloader, need train/val subsets split before execution
    folder_name: already unzipped validation dataset folder path
    '''
    image_val = []
    mask_val = []

    images_folder = os.path.join(folder_name, "val", "image")
    masks_folder = os.path.join(folder_name, "val", "mask")

    image_list = os.listdir(images_folder)
    for idx, image_name in tqdm(enumerate(image_list), total=len(image_list), desc="load validation images......"):
        image_path = os.path.join(images_folder, image_name)
        mask_path = os.path.join(masks_folder, os.path.splitext(image_name)[0] + ".png")

        image = Image.open(image_path)
        image_val.append(image)

        mask = Image.open(mask_path)
        mask_val.append(mask)

    return image_val, mask_val


class SegDataset(Dataset):
    '''
    prepare image dataset with certain transforms
    '''
    def __init__(self, image_list, mask_list, transform_img, transform_mask):
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.imagelist = image_list
        self.masklist = mask_list


    def __len__(self):
        return len(self.imagelist)


    def __getitem__(self, idx):
        image = deepcopy(self.imagelist[idx])
        mask = deepcopy(self.masklist[idx])

        image = self.transform_img(image) # apply transform

        mask = self.transform_mask(mask)
        mask = torch.as_tensor(np.array(mask), dtype=torch.int64) # mask transform does not contain to_tensor function

        return image, mask