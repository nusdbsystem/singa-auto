import os

os.environ['CUDA_VISIBLE_DEVICES'] = "4, 5, 6, 7"

import sys
sys.path.append(os.getcwd())

import base64
import json
import logging
import os
import tempfile
import zipfile
from collections.abc import Sequence
from collections import defaultdict
from copy import deepcopy
from io import BytesIO
from typing import List
from glob import glob
from time import time
import requests

import cv2
import numpy as np
import PIL
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler
from torchvision import models
from torchvision.transforms import functional as F
from torchvision.transforms.transforms import Pad, Resize
from tqdm import tqdm
from torch.nn import DataParallel

from singa_auto.model import SegmentationModel, CategoricalKnob, FixedKnob, utils
from singa_auto.model.knob import BaseKnob
# from singa_auto.utils.metrics import do_kaggle_metric

# from singa_auto.datasets.image_segmentation_dataset import *


# dataset fetch
def ImageFetch(img_folder, split_rate=0.9):
    img_train = []
    mask_train = []
    img_val = []
    mask_val = []

    image_folder = os.path.join(img_folder, "image")
    mask_folder = os.path.join(img_folder, "mask")

    img_list = os.listdir(image_folder)
    total_img_num = len(img_list)
    print(f'Total number of images: {total_img_num}')

    train_num = int(total_img_num * split_rate)
    for idx, image_name in tqdm(enumerate(img_list[:train_num]), total=train_num, desc="load train images......"):
        image_path = os.path.join(image_folder, image_name)
        mask_path = os.path.join(mask_folder, image_name.split('.')[0] + ".png")

        image = Image.open(image_path)
        img_train.append(image)

        mask = Image.open(mask_path)
        mask_train.append(mask)
    for idx, image_name in tqdm(enumerate(img_list[train_num:]), total=(total_img_num - train_num), desc="load val images......"):
        image_path = os.path.join(image_folder, image_name)
        mask_path = os.path.join(mask_folder, image_name.split('.')[0] + ".png")

        image = Image.open(image_path)
        img_val.append(image)

        mask = Image.open(mask_path)
        mask_val.append(mask)

    return img_train, mask_train, img_val, mask_val


def trainImageFetch(train_folder):
    image_train = []
    mask_train = []

    # load images and masks from their folders
    images_folder = os.path.join(train_folder, "image")
    masks_folder = os.path.join(train_folder, "mask")

    image_list = os.listdir(images_folder)
    for idx, image_name in tqdm(enumerate(image_list), total=len(image_list), desc="load train images......"):
        image_path = os.path.join(images_folder, image_name)
        mask_path = os.path.join(masks_folder, image_name.split('.')[0] + ".png")

        image = Image.open(image_path)
        image_train.append(image)

        mask = Image.open(mask_path)
        mask_train.append(mask)

    return image_train, mask_train


def valImageFetch(val_folder):
    image_val = []
    mask_val = []

    images_folder = os.path.join(val_folder, "image")
    masks_folder = os.path.join(val_folder, "mask")

    image_list = os.listdir(images_folder)
    for idx, image_name in tqdm(enumerate(image_list), total=len(image_list), desc="load validation images......"):
        image_path = os.path.join(images_folder, image_name)
        mask_path = os.path.join(masks_folder, image_name.split('.')[0] + ".png")

        image = Image.open(image_path)
        image_val.append(image)

        mask = Image.open(mask_path)
        mask_val.append(mask)

    return image_val, mask_val


class SegDataset(Dataset):
    def __init__(self, image_list, mask_list, mode, transform_img, transform_mask, transform_border):
        self.mode = mode
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.transform_border = transform_border
        self.imagelist = image_list
        self.masklist = mask_list


    def __len__(self):
        return len(self.imagelist)


    def __getitem__(self, idx):
        image = deepcopy(self.imagelist[idx])

        if self.mode == 'train':
            mask = deepcopy(self.masklist[idx])
            
            mask_arr = np.array(mask)
            border = cv2.Canny(mask_arr, 0, 0).astype(np.float)
            border /= 255
            border = Image.fromarray(border.astype(np.uint8))
            border_img = self.transform_border(border)
            border = torch.as_tensor(np.array(border_img), dtype=torch.int64)
            # one_hot = torch.cat((torch.zeros_like(border).unsqueeze(0), torch.zeros_like(border).unsqueeze(0))).scatter_(0, border.unsqueeze(0), 1)

            image = self.transform_img(image)

            mask = self.transform_mask(mask)
            mask = torch.as_tensor(np.array(mask), dtype=torch.int64)
            # print(f'after transform mask max: {mask.max()}')

            # image = image.unsqueeze(0)
            # mask = mask.unsqueeze(0)

            return image, mask, border

        elif self.mode == 'val':
            mask = deepcopy(self.masklist[idx])

            mask_arr = np.array(mask)
            border = cv2.Canny(mask_arr, 0, 0).astype(np.float)
            border /= 255
            border = Image.fromarray(border.astype(np.uint8))
            border_img = self.transform_border(border)
            border = torch.as_tensor(np.array(border_img), dtype=torch.int64)
            # one_hot = torch.cat((torch.zeros_like(border).unsqueeze(0), torch.zeros_like(border).unsqueeze(0))).scatter_(0, border.unsqueeze(0), 1)

            image = self.transform_img(image)

            mask = self.transform_mask(mask)
            mask = torch.as_tensor(np.array(mask), dtype=torch.int64)

            # image = image.unsqueeze(0)
            # mask = mask.unsqueeze(0)

            return image, mask, border


# define model
down_feature = defaultdict(list)
filter_list = [i for i in range(6, 9)]


class down_sampling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(down_sampling, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2)


    def forward(self, in_feat):
        x = self.conv(in_feat)
        down_feature[in_feat.device.index].append(x)
        x = self.pool(x)

        return x


class up_sampling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(up_sampling, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channel, out_channel, 2, stride=2)
        self.relu_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )


    def forward(self, in_feat):
        x = self.up_conv(in_feat)
        down_map = down_feature[in_feat.device.index].pop()
        x = torch.cat([x, down_map], dim=1)
        x = self.relu_conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.input_conv = down_sampling(3, 64)
        self.down_list = [down_sampling(2 ** i, 2 ** (i + 1)) for i in filter_list]
        self.down = nn.Sequential(*self.down_list)

        self.last_layer = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.up_init = up_sampling(1024, 512)
        self.up_list = [up_sampling(2 ** (i + 1), 2 ** i) for i in filter_list[::-1]]
        self.up = nn.Sequential(*self.up_list)

        self.output = nn.Conv2d(64, num_classes, 1)
        # self.classifier = nn.Softmax()
        


    def forward(self, in_feat):
        x = self.input_conv(in_feat)
        x = self.down(x)
        x = self.last_layer(x)
        x = self.up_init(x)
        x = self.up(x)
        out = self.output(x)


        # out = self.classifier(x)
        # return out
        return out, x


class BorderUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.unet = UNet(n_class)

        self.border_extraction = nn.Conv2d(64, 2, kernel_size=1, padding=0)

        self.softmax = nn.Softmax2d()
    
    def forward(self, in_feat):
        # regular cnn process
        init_seg, unet_feature = self.unet(in_feat)

        # extract and enhance border
        init_border = self.border_extraction(unet_feature)
        
        # output
        out = self.softmax(init_seg)

        return out, init_border


# pre-process: resize image to the target scale keeping aspect ratio then pad to square
class ResizeSquarePad(Resize, Pad):
    def __init__(self, target_length, interpolation_strategy):
        if not isinstance(target_length, (int, Sequence)):
            raise TypeError("Size should be int or sequence. Got {}".format(type(target_length)))
        if isinstance(target_length, Sequence) and len(target_length) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")

        self.target_length = target_length
        self.interpolation_strategy = interpolation_strategy
        Resize.__init__(self, size=(320, 320), interpolation=self.interpolation_strategy)
        Pad.__init__(self, padding=(0,0,0,0), fill=255, padding_mode="constant")


    def __call__(self, img):
        w, h = img.size
        if w > h:
            self.size = (int(np.round(self.target_length * (h / w))), self.target_length)
            img = Resize.__call__(self, img)

            total_pad = self.size[1] - self.size[0]
            half_pad = total_pad // 2
            self.padding = (0, half_pad, 0, total_pad - half_pad)
            return Pad.__call__(self, img)
        else:
            self.size = (self.target_length, int(np.round(self.target_length * (w / h))))
            img = Resize.__call__(self, img)

            total_pad = self.size[0] - self.size[1]
            half_pad = total_pad // 2
            self.padding = (half_pad, 0, total_pad - half_pad, 0)
            return Pad.__call__(self, img)


# customized loss function
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / \
                    (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class MulticlassDiceLoss(nn.Module):
    """
    requires input(prediction) dimension as [b, c, h, w]
    target(ground truth mask) dimension as [b, 1, h, w] where dimension 2 refers to the class index
    Can convert target to one_hot automatically and support ignore labels (should be in the form of list)
    """

    def __init__(self, ignore_labels=None):
        super(MulticlassDiceLoss, self).__init__()
        self.ignore_labels = ignore_labels

    def forward(self, input, target):

        num_ignore = 0 if self.ignore_labels == None else len(self.ignore_labels)

        n, _, h, w = target.shape[:]

        num_classes = input.shape[1]

        # initialize zeros for one_hot
        zeros = torch.zeros((n, (num_classes + num_ignore), h, w)).to(target.device)

        # decrease ignore labels' indexes into successive integers(eg: convert 0, 1, 2, 255 into 0, 1, 2, 3)
        for i in range(num_ignore):
            target[target == self.ignore_labels[i]] = num_classes + i

        # scatter to one_hot
        one_hot = zeros.scatter_(1, target, 1)

        dice = DiceLoss()
        totalLoss = 0

        # for indexes out of range, not compute corresponding loss
        for i in range(num_classes):
             diceLoss = dice(input[:, i], one_hot[:,i])
             totalLoss += diceLoss

        return totalLoss


logger = logging.getLogger(__name__)


# main process procedure
class SaUNetBorderLoss(SegmentationModel):
    '''
    train UNet
    '''
    @staticmethod
    def get_knob_config():
        return {
            # hyper parameters
            "lr": FixedKnob(1e-3),
            "momentum": FixedKnob(0.9),

            "ignore_index": FixedKnob(255),
            "batch_size": FixedKnob(12),
            "epoch": FixedKnob(1),

            # application parameters
            # "num_classes": FixedKnob(1),
            "fine_size": FixedKnob(512),

        }


    def __init__(self, **knobs):
        super().__init__(**knobs)
        # load knobs
        self._knobs = knobs

        # initiate hyper params
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("self.device", self.device)
        logger.info(self.device)

        self.model = None

        self.fine_size = self._knobs.get("fine_size")

        self.ignore_index = self._knobs.get("ignore_index")

        self.batch_size = self._knobs.get("batch_size")
        self.epoch = self._knobs.get("epoch")

        self.lr = self._knobs.get("lr")
        self.momentum = self._knobs.get("momentum")

        # define preprocessing procedure
        self.transform_img = torchvision.transforms.Compose([
            ResizeSquarePad(self.fine_size, Image.BILINEAR),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])

        self.transform_mask = torchvision.transforms.Compose([
            ResizeSquarePad(self.fine_size, Image.NEAREST)
        ])

        self.transform_border = torchvision.transforms.Compose([
            ResizeSquarePad(self.fine_size, Image.NEAREST)
        ])
            

    def train(self, dataset_path, **kwargs):
        logger.info("Training params: {}".format(json.dumps(kwargs)))


        # extract uploaded zipfile
        dataset_zipfile = zipfile.ZipFile(dataset_path, 'r')

        train_folder = tempfile.TemporaryDirectory()
        folder_name = train_folder.name
        dataset_zipfile.extractall(path=folder_name)

        # load train params from zipfile
        with open(os.path.join(folder_name, 'param.json'),'r') as load_f:
            load_dict = json.load(load_f)
            self.num_classes = load_dict["num_classes"] if "num_classes" in list(load_dict.keys()) else 21 # default class number(21) is the same as voc2012

        # load images from zipfile
        if os.path.isdir(os.path.join(folder_name, "image")):
            print("split train/val subsets...")
            logger.info("split train/val subsets...")
            image_train, mask_train, image_val, mask_val = ImageFetch(folder_name)
            self.num_image = len(image_train)
            print("Total training images : ", self.num_image) 
            logger.info(f"Total training images : {self.num_image}")           
        elif os.path.isdir(os.path.join(folder_name, "train")):
            print("directly load train/val datasets...")
            logger.info("directly load train/val datasets...")
            image_train, mask_train = trainImageFetch(folder_name)
            image_val, mask_val = valImageFetch(folder_name)
            self.num_image = len(image_train)
            print("Total training images : ", self.num_image)
            logger.info(f"Total training images : {self.num_image}")  
        else:
            print("unsupported dataset format!")
            logger.info("unsupported dataset format!")

        # load dataset
        train_data = SegDataset(image_train, mask_train, 'train', self.transform_img, self.transform_mask, self.transform_border)
        val_data = SegDataset(image_val, mask_val, 'val', self.transform_img, self.transform_mask, self.transform_border)

        logger.info("Training the model ResUNet using {}".format(self.device))
        print("Training the model ResUNet using {}".format(self.device))

        # define training and validation data loaders
        train_loader = DataLoader(train_data,
                    shuffle=RandomSampler(train_data), 
                    batch_size=self.batch_size) 

        val_loader = DataLoader(val_data,
                    shuffle=False, 
                    batch_size=self.batch_size) 

        # get the model using our helper function
        self.model = BorderUNet(self.num_classes)
        self.model = DataParallel(self.model)
        self.model.to(self.device)

        self.criterion_ce = nn.CrossEntropyLoss(weight=torch.Tensor([1, 100]), ignore_index=self.ignore_index)
        self.criterion_dice = MulticlassDiceLoss(ignore_labels=[255])

        self.optimizer_ft = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, momentum=self.momentum)
        self.exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_ft, step_size=20, gamma=0.1)

        # start training
        for epoch_ in range(self.epoch):
            train_loss = self._train_one_epoch(train_loader, self.model)
            val_loss, accuracy = self._evaluate(val_loader, self.model)
            self.exp_lr_scheduler.step()

            print('epoch: {} train_loss: {:.3f} val_loss: {:.3f} val_accuracy: {:.3f}'.format(epoch_ + 1, train_loss, val_loss, accuracy))
            logger.info('epoch: {} train_loss: {:.3f} val_loss: {:.3f} val_accuracy: {:.3f}'.format(epoch_ + 1, train_loss, val_loss, accuracy))
    

    def _train_one_epoch(self, train_loader, model):
        '''
        consider as a sub-train function inside singa-auto framework
        '''
        running_loss = 0.0
        data_size = len(train_loader) * self.batch_size

        model.train()

        for inputs, masks, borders in tqdm(train_loader):
            inputs, masks, borders = inputs.to(self.device), masks.long().to(self.device), borders.long().to(self.device)
            self.optimizer_ft.zero_grad()

            init_seg, init_border = model(inputs)

            self.criterion_ce.to(self.device)
            loss_border = self.criterion_ce(init_border, borders)
            loss_seg = self.criterion_dice(init_seg, masks.unsqueeze(1))

            loss = loss_border + loss_seg

            loss.backward()
            self.optimizer_ft.step()

            running_loss += loss.item() * self.batch_size

        epoch_loss = running_loss / data_size
        return epoch_loss


    def _evaluate(self, test_loader, model):
        '''
        validation per epoch
        '''
        running_loss = 0.0
        acc = 0.0
        data_size = len(test_loader) * self.batch_size

        model.eval()

        with torch.no_grad():
            for inputs, masks, borders in test_loader:
                inputs, masks, borders = inputs.to(self.device), masks.long().to(self.device), borders.long().to(self.device)

                outputs, fine_border = self.model(inputs)

                predict = torch.argmax(nn.Softmax(dim=1)(outputs), dim=1) # extract argmax as the final prediction

                # we do not consider the ignore_index
                pure_mask = masks.masked_select(masks.ne(self.ignore_index))
                pure_predict = predict.masked_select(masks.ne(self.ignore_index))

                acc += pure_mask.cpu().eq(pure_predict.cpu()).sum().item()/len(pure_mask) # find the correct pixels
                
                self.criterion_ce.to(self.device)
                loss_border = self.criterion_ce(fine_border, borders)
                loss_seg = self.criterion_dice(outputs, masks.unsqueeze(1))

                loss = loss_seg + loss_border

                running_loss += loss.item() * self.batch_size

        epoch_loss = running_loss / data_size
        accuracy = acc / len(test_loader)
        return epoch_loss, accuracy


    def evaluate(self, val_dataset_path, **kwargs):
        # extract validation datasets
        dataset_zipfile = zipfile.ZipFile(val_dataset_path, 'r')
        val_folder = tempfile.TemporaryDirectory()
        dataset_zipfile.extractall(path=val_folder.name)
        folder_name = val_folder.name

        if os.path.isdir(os.path.join(folder_name, "image")):
            print("split train/val subsets...")
            logger.info("split train/val subsets...")
            image_train, mask_train, X_val, y_val = ImageFetch(folder_name)
            self.num_image = len(X_val)
            print("Total val images : ", self.num_image) 
            logger.info(f"Total val images : {self.num_image}")           
        elif os.path.isdir(os.path.join(folder_name, "train")):
            print("directly load train/val datasets...")
            logger.info("directly load train/val datasets...")
            image_train, mask_train = trainImageFetch(folder_name)
            X_val, y_val = valImageFetch(folder_name)
            self.num_image = len(X_val)
            print("Total val images : ", self.num_image)
            logger.info(f"Total val images : {self.num_image}")  
        else:
            print("unsupported dataset format!")
            logger.info("unsupported dataset format!")

        val_data = SegDataset(X_val, y_val, 'val', self.transform_img, self.transform_mask, self.transform_border)

        val_loader = DataLoader(val_data,
                            shuffle=False,
                            batch_size=self.batch_size)
        # compute MIoU metric(consider as accuracy)
        temp_miou = {}
        for i in range(self.num_classes):
            temp_miou[i] = [0, 0.0]

        self.model.eval()

        with torch.no_grad():
            for inputs, masks, borders in val_loader:
                inputs, masks, borders = inputs.to(self.device), masks.long().to(self.device), borders.long().to(self.device)

                outputs, fine_border = self.model(inputs)

                predict = torch.argmax(nn.Softmax(dim=1)(outputs), dim=1)
                pure_mask = masks.masked_select(masks.ne(255))
                pure_predict = predict.masked_select(masks.ne(255))

                for class_value in pure_mask.unique():
                    valued_mask = pure_mask.masked_select(pure_mask.eq(class_value))
                    real_len = len(valued_mask)
                    
                    valued_predict = pure_predict.masked_select(pure_mask.eq(class_value))
                    cross_len = valued_mask.eq(valued_predict).sum().item()

                    predict_len = len(pure_predict.masked_select(pure_predict.eq(class_value)))

                    temp_miou[class_value.item()][1] += cross_len / (real_len + predict_len - cross_len)
                    temp_miou[class_value.item()][0] += 1

        miou_overall = 0.0
        existed_classes = 0
        for key in temp_miou.keys():
            if temp_miou[key][0] != 0:
                miou_overall += (temp_miou[key][1] / temp_miou[key][0])
                existed_classes += 1
        temp_miou['overall'] = [1, miou_overall / existed_classes]

        for key in temp_miou.keys():
            if temp_miou[key][0] != 0:
                print(f"class {key} accuracy: {temp_miou[key][1] / temp_miou[key][0]}")
        return temp_miou['overall'][1]


    def dump_parameters(self):
        params = {}
        with tempfile.NamedTemporaryFile() as tmp:
            
            # Save whole model to a tempfile
            torch.save(self.model.module.state_dict(), tmp.name)
            # Read from tempfile & encode it to base64 string
            with open(tmp.name, 'rb') as f:
                weight_base64 = f.read()
        params['weight_base64'] = base64.b64encode(weight_base64).decode('utf-8')
        params['num_classes'] = self.num_classes
        return params


    def load_parameters(self, params):
        weight_base64 = params['weight_base64']
        self.num_classes = params['num_classes']

        weight_base64_bytes = base64.b64decode(weight_base64.encode('utf-8'))

        state_dict = torch.load(BytesIO(weight_base64_bytes), map_location=self.device)

        self.model = BorderUNet(self.num_classes)
        self.model.load_state_dict(state_dict)

        self.model = DataParallel(self.model)
        self.model.to(self.device)
        self.model.eval()

    def _get_prediction(self, img):

        image = self.transform_img(img)

        image = image.to(self.device)
        predict, _ = self.model(image.unsqueeze(0))

        predict = predict.squeeze(0)
        predict = nn.Softmax(dim=0)(predict)
        predict = torch.argmax(predict, dim=0)

        # transform result image into original size
        w, h = img.size
        if w > h:
            re_h = int(np.round(self.fine_size * (h / w)))
            total_pad = self.fine_size - re_h
            half_pad = total_pad // 2
            out = predict[half_pad : half_pad + re_h, :]
        else:
            re_w = int(np.round(self.fine_size * (w / h)))
            total_pad = self.fine_size - re_w
            half_pad = total_pad // 2
            out = predict[:, half_pad : half_pad + re_w]

        out = cv2.resize(out.cpu().numpy(), (w, h), interpolation=cv2.INTER_NEAREST)

        return out



    def predict(self, queries: List[List]) -> List[dict]:
        # print(len(queries))
        result = list()

        # depending on different input types, need different conditions
        for idx, img in enumerate(queries):
            print("*" * 30)
            print(type(img))
            if isinstance(img, List):
                print(len(img))
                img = np.array(img[0])
                print(img.shape)
                img_file = Image.fromarray(np.uint8(img))
                print(type(img_file))
            elif isinstance(img, np.ndarray):
                img_file = Image.fromarray(img)
            else:
                img_file = img

            # get prediction
            res_raw = self._get_prediction(img_file)

            # add color palette (we follow the VOC2012 color map and the max num_class is 21)
            res_raw = res_raw.astype(np.uint8)
            res = Image.fromarray(res_raw)
            palette = []
            for i in range(256):
                palette.extend((i, i, i))
            palette[:3*21] = np.array([[0, 0, 0],
                                [128, 0, 0],
                                [0, 128, 0],
                                [128, 128, 0],
                                [0, 0, 128],
                                [128, 0, 128],
                                [0, 128, 128],
                                [128, 128, 128],
                                [64, 0, 0],
                                [192, 0, 0],
                                [64, 128, 0],
                                [192, 128, 0],
                                [64, 0, 128],
                                [192, 0, 128],
                                [64, 128, 128],
                                [192, 128, 128],
                                [0, 64, 0],
                                [128, 64, 0],
                                [0, 192, 0],
                                [128, 192, 0],
                                [0, 64, 128]
                             ], dtype='uint8').flatten()
            res.putpalette(palette)

            name = f"./query_{idx}.png"
            res.save(name)
            full_name = os.path.abspath(name)

            buffered = BytesIO()
            res.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue())


            result.append(img_str.decode('utf-8'))
            
            # result.append(requests.get('http://192.168.100.203:36667/fetch').text)
                
        return result


if __name__ == "__main__":
    import argparse

    from singa_auto.model.dev import test_model_class
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path',
                        type=str,
                        default='/home/zhaozixiao/dataset/pets/datasets.zip',
                        help='Path to train dataset')
    parser.add_argument('--val_path',
                        type=str,
                        default='/home/zhaozixiao/dataset/pets/datasets.zip',
                        help='Path to validation dataset')

    # parser.add_argument('--annotation_dataset_path',
    #                     type=str,
    #                     default='./dataset/voc2012/val2014.zip',
    #                     help='Path to validation dataset')

    # parser.add_argument('--test_path',
    #                     type=str,
    #                     default='/hdd1/PennFudanPed.zip',
    #                     help='Path to test dataset')
    parser.add_argument('--query_path',
                        type=str,
                        default='/home/zhaozixiao/dataset/pets/Persian_120.jpg,/home/zhaozixiao/dataset/pets/pomeranian_159.jpg',
                        help='Path(s) to query image(s), delimited by commas')

    (args, _) = parser.parse_known_args()

    # print(args.query_path.split(','))

    imgs = utils.dataset.load_images(args.query_path.split(','))
    img_nps = []
    for i in imgs:
        img = np.array(i)
        img_nps.append(img)
    
    queries = img_nps
    test_model_class(model_file_path=__file__,
                     model_class='SaUNetBorderLoss',
                     task='IMAGE_SEGMENTATION',
                     dependencies={"torch": "1.6.0+cu101",
                                   "torchvision": "0.7.0+cu101",
                                   "opencv": "3.4.2",
                                   "tqdm": "4.28.0"},
                     train_dataset_path=args.train_path,
                     val_dataset_path=args.val_path,
                     test_dataset_path=None,
                     train_args={"num_classes": 3},
                     queries=img_nps)
