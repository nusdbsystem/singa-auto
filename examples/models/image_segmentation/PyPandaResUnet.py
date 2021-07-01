import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import sys
sys.path.append(os.getcwd())


import base64
import json
import logging
import os
import tempfile
import zipfile
from collections.abc import Sequence
from copy import deepcopy
from io import BytesIO
from typing import List
from glob import glob

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

from singa_auto.model import SegmentationModel, CategoricalKnob, FixedKnob, utils
from singa_auto.model.knob import BaseKnob
# from singa_auto.utils.metrics import do_kaggle_metric

from singa_auto.datasets.image_segmentation_dataset import *


# define model
def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = models.resnet18(pretrained=False)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


# pre-process: resize image to the target scale keeping aspect ratio then pad to square
class ResizeSquarePad(Resize, Pad):
    def __init__(self, target_length, interpolation_strategy):
        if not isinstance(target_length, (int, Sequence)):
            raise TypeError("Size should be int or sequence. Got {}".format(type(target_length)))
        if isinstance(target_length, Sequence) and len(target_length) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")

        self.target_length = target_length
        self.interpolation_strategy = interpolation_strategy
        Resize.__init__(self, size=(512, 512), interpolation=self.interpolation_strategy)
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


logger = logging.getLogger(__name__)


# main process procedure
class PyPandaResUnet(SegmentationModel):
    '''
    train UNet
    '''
    @staticmethod
    def get_knob_config():
        return {
            # hyper parameters
            "lr": FixedKnob(1e-4),
            "ignore_index": FixedKnob(255),
            "batch_size": FixedKnob(4),
            "epoch": FixedKnob(2),

            # application parameters
            # "num_classes": FixedKnob(1),
            "fine_size": FixedKnob(512),

        }


    def __init__(self, **knobs):
        super().__init__(**knobs)
        self._knobs = knobs

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("self.device", self.device)
        logger.info(self.device)

        self.model = None

        self.fine_size = self._knobs.get("fine_size")

        # define preprocessing procedure
        self.transform_img = torchvision.transforms.Compose([
            ResizeSquarePad(self.fine_size, Image.BILINEAR),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])

        self.transform_mask = torchvision.transforms.Compose([
            ResizeSquarePad(self.fine_size, Image.NEAREST)
        ])
            

    def train(self, dataset_path, **kwargs):
        # hyper parameters 
        self.batch_size = self._knobs.get("batch_size")
        self.epoch = self._knobs.get("epoch")
        snapshot = 2 

        self.lr = self._knobs.get("lr")
        self.ignore_index = self._knobs.get("ignore_index")

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
        train_data = SegDataset(image_train, mask_train, self.transform_img, self.transform_mask)
        val_data = SegDataset(image_val, mask_val, self.transform_img, self.transform_mask)

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
        self.model = ResNetUNet(self.num_classes)
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        self.optimizer_ft = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        self.exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_ft, step_size=30, gamma=0.1)

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
        data_size = len(train_loader)

        model.train()

        for inputs, masks in tqdm(train_loader):
            inputs, masks = inputs.to(self.device), masks.long().to(self.device)
            self.optimizer_ft.zero_grad()

            logit = model(inputs)

            loss = self.criterion(logit, masks.squeeze(1)) # cross_entropy loss
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
        data_size = len(test_loader)

        model.eval()

        with torch.no_grad():
            for inputs, masks in test_loader:
                inputs, masks = inputs.to(self.device), masks.long().to(self.device)

                outputs = self.model(inputs)

                predict = torch.argmax(nn.Softmax(dim=1)(outputs), dim=1) # extract argmax as the final prediction

                # we do not consider the ignore_index
                pure_mask = masks.masked_select(masks.ne(self.ignore_index))
                pure_predict = predict.masked_select(masks.ne(self.ignore_index))

                acc += pure_mask.cpu().eq(pure_predict.cpu()).sum().item()/len(pure_mask) # find the correct piixels
                
                loss = self.criterion(outputs.squeeze(1), masks.squeeze(1))           
                running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / data_size
        accuracy = acc / data_size
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

        val_data = SegDataset(X_val, y_val, self.transform_img, self.transform_mask)

        val_loader = DataLoader(val_data,
                            shuffle=False,
                            batch_size=4)
        # compute MIoU metric(consider as accuracy)
        temp_miou = {}
        for i in range(self.num_classes):
            temp_miou[i] = [0, 0.0]

        self.model.eval()

        with torch.no_grad():
            for inputs, masks in val_loader:
                inputs, masks = inputs.to(self.device), masks.long().to(self.device)

                outputs = self.model(inputs)

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
            torch.save(self.model, tmp.name)
            # Read from tempfile & encode it to base64 string
            with open(tmp.name, 'rb') as f:
                weight_base64 = f.read()
        params['weight_base64'] = base64.b64encode(weight_base64).decode('utf-8')
        return params


    def load_parameters(self, params):
        weight_base64 = params['weight_base64']

        weight_base64_bytes = base64.b64decode(weight_base64.encode('utf-8'))

        self.model = torch.load(BytesIO(weight_base64_bytes), map_location=self.device)

    def _get_prediction(self, img):

        image = self.transform_img(img)

        image = image.to(self.device)
        predict = self.model(image.unsqueeze(0))

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



    def predict(self, queries: List[PIL.Image.Image]) -> List[dict]:

        result = list()

        for idx, img in enumerate(queries):
            res_raw = self._get_prediction(img)

            # add color palette (we follow the VOC2012 color map ant the max num_class is 21)
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
            
            result.append(name)

        return result


if __name__ == "__main__":
    import argparse

    from singa_auto.model.dev import test_model_class
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path',
                        type=str,
                        default='/home/taomingyang/dataset/package/voc2012_mini.zip',
                        help='Path to train dataset')
    parser.add_argument('--val_path',
                        type=str,
                        default='/home/taomingyang/dataset/package/voc2012_mini.zip',
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
                        default='/home/taomingyang/git/singa_auto_hub/examples/data/image_segmentaion/2007_000862.jpg,/home/taomingyang/git/singa_auto_hub/examples/data/image_segmentaion/2007_001397.jpg',
                        help='Path(s) to query image(s), delimited by commas')

    (args, _) = parser.parse_known_args()

    # print(args.query_path.split(','))

    queries = utils.dataset.load_images(args.query_path.split(','))
    test_model_class(model_file_path=__file__,
                     model_class='PyPandaResUnet',
                     task='IMAGE_SEGMENTATION',
                     dependencies={"torch": "1.6.0+cu101",
                                   "torchvision": "0.7.0+cu101",
                                   "opencv-python": "4.4.0.46",
                                   "tqdm": "4.28.0"},
                     train_dataset_path=args.train_path,
                     val_dataset_path=args.val_path,
                     test_dataset_path=None,
                     train_args={"num_classes": 21},
                     queries=queries)
