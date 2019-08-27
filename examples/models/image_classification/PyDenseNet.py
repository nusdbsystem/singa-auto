from __future__ import division
from __future__ import print_function
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import pprint
import json
import time
from PIL import Image
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.models.densenet import densenet121
from torchvision.models.resnet import resnet101
from torchvision.models.inception import inception_v3
from torchvision.models.vgg import vgg19_bn
from torchvision.models.alexnet import alexnet
### pydensenet
from torch.utils.data import Dataset, DataLoader
# from panda import *
import sklearn.metrics
import pickle
import base64
#from generic_models.densenet import densenet121
import abc
from typing import Union, Dict, Optional, Any, List

import tempfile
import numpy as np
import json
import argparse

from rafiki.model import BaseModel, FloatKnob, CategoricalKnob, FixedKnob, IntegerKnob, PolicyKnob, utils
from rafiki.model.knob import BaseKnob
from rafiki.constants import ModelDependency
from rafiki.model.dev import test_model_class


KnobConfig = Dict[str, BaseKnob]
Knobs = Dict[str, Any]
Params = Dict[str, Union[str, int, float, np.ndarray]]

class DenseNet121(nn.Module):
    def __init__(self, scratch=False, drop_rate=0, num_classes=2):
        super(DenseNet121, self).__init__()
        self._model = densenet121(pretrained=not scratch, drop_rate=drop_rate)
        num_features = self._model.classifier.in_features
        self._model.classifier = nn.Linear(num_features, num_classes) 

    def forward(self, x):
        return self._model(x)

class PyDenseNet(BaseModel):
    def __init__(self, **knobs):
        super().__init__(**knobs)
        self.__dict__.update(knobs) 
        self._knobs = knobs

        #Parameters not advised by rafiki advisor
        #NOTE: should be dumped/loaded in dump_parameter/load_parameter
        self._image_size = 128

        #The following parameters are determined when training dataset is loaded
        self._normalize_mean = []
        self._normalize_std = []
        self._num_classes = []

    @staticmethod
    def get_knob_config():
        return {
            'max_epochs': FixedKnob(10), 
            'batch_size': CategoricalKnob([4]),
            'max_iter': FixedKnob(20),
            'max_image_size': FixedKnob(32),   ### scale 
            'share_params': CategoricalKnob(['SHARE_PARAMS']),
            ### panda params
            'model':CategoricalKnob(['densenet']),
            'tag':CategoricalKnob(['relabeled']),
            'optimizer':CategoricalKnob(['adam']),
            'save_path':CategoricalKnob(['ckpt_densenet']),
            'workers':FixedKnob(8),
            'seed':FixedKnob(123456),
            'scale':FixedKnob(512),
            'lr':FixedKnob(0.0001), ### learning_rate
            'weight_decay':FixedKnob(0.0),
            'drop_rate':FixedKnob(0.0),
     
            'toy':FixedKnob(True),
            'horizontal_flip':FixedKnob(True),
            'verbose':FixedKnob(True),
            'scratch':FixedKnob(True),
            'train_weighted':FixedKnob(True),
            'valid_weighted':FixedKnob(True)

        }

    def get_peformance_metrics(self, gts, probabilities, use_only_index = None):
        assert(np.all(probabilities >= 0) == True)
        assert(np.all(probabilities <= 1) == True)

        def compute_metrics_for_class(i):  ### i for each pathology
            p, r, t = sklearn.metrics.precision_recall_curve(gts[:, i], probabilities[:, i])
            PR_AUC = sklearn.metrics.auc(r, p)
            ROC_AUC = sklearn.metrics.roc_auc_score(gts[:, i], probabilities[:, i])
            F1 = sklearn.metrics.f1_score(gts[:, i], preds[:, i])
            acc = sklearn.metrics.accuracy_score(gts[:, i], preds[:, i])
            count = np.sum(gts[:, i])
            return PR_AUC, ROC_AUC, F1, acc, count

        PR_AUCs = []
        ROC_AUCs = []
        F1s = []
        accs = []
        counts = []
        preds = probabilities >= 0.5

        classes = [use_only_index] if use_only_index is not None else range(self.num_classes)

        for i in classes: ### i for each pathology
            try:
                PR_AUC, ROC_AUC, F1, acc, count = compute_metrics_for_class(i) ### i for each pathology
            except ValueError:
                continue
            PR_AUCs.append(PR_AUC)
            ROC_AUCs.append(ROC_AUC)
            F1s.append(F1)
            accs.append(acc)
            counts.append(count)
            # print('Class: {!s} Count: {:d} PR AUC: {:.4f} ROC AUC: {:.4f} F1: {:.3f} Acc: {:.3f}'.format(self.pathologies[i], count, PR_AUC, ROC_AUC, F1, acc))

        avg_PR_AUC = np.average(PR_AUCs)
        ### what if count == 0 ???
        avg_ROC_AUC = np.average(ROC_AUCs, weights=counts)  
        avg_F1 = np.average(F1s, weights=counts)

        print('Avg PR AUC: {:.3f}'.format(avg_PR_AUC))
        print('Avg ROC AUC: {:.3f}'.format(avg_ROC_AUC))
        print('Avg F1: {:.3f}'.format(avg_F1))
        return avg_PR_AUC, avg_ROC_AUC, avg_F1, sum(accs)/classes

    def train(self, dataset_path, **kwargs):###  args
        """
        Train the model with given dataset_path

        parameters:
            dataset_path: path to dataset_path
                type: str
            **kwargs:
                optional arguments

        """
        num_classes = 2
        dataset = utils.dataset.load_dataset_of_image_files(
            dataset_path, 
            min_image_size=32, 
            max_image_size=self.max_image_size, 
            mode='RGB', 
            lazy_load=True)
        #stat = dataset.get_stat()
        mu = [0.48233507, 0.48233507, 0.48233507]
        std = [0.07271624, 0.07271624, 0.07271624]
        self._normalize_mean = mu
        self._normalize_std = std

        # construct the model
        self._model = DenseNet121(self._knobs.get("scratch"), self._knobs.get("drop_rate"), num_classes)

        train_dataset = ImageDataset(
            rafiki_dataset=dataset, 
            image_scale_size=128, 
            norm_mean=mu, 
            norm_std=std, 
            is_train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        #Setup Criterion
        if num_classes == 2:
            self.train_criterion = nn.CrossEntropyLoss()
        else:
            self.train_criterion = nn.MultiLabelSoftMarginLoss()

        #Setup Optimizer
        if self.optimizer == "adam":
            optimizer = optim.Adam(
                           filter(lambda p: p.requires_grad, self._model.parameters()),
                           lr=self.lr,
                           weight_decay=self.weight_decay)
        elif self.optimizer == "rmsprop":
            optimizer = optim.RMSprop(
                           filter(lambda p: p.requires_grad, self._model.parameters()),
                           lr=self.lr,
                           weight_decay=self.weight_decay)
        else:
            print("{} is not a valid optimizer.".format(self.optimizer))

        #Setup Learning Rate Scheduler
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, threshold=0.001, factor=0.1)

        try: 
            self._model = self._model.cuda()
        except: 
            pass

        for epoch in range(1, self.max_epochs + 1):
            print("Epoch {}/{}".format(epoch, self.max_epochs))
            print("-" * 10)
            self._model.train()
            batch_losses = []
            for batch_idx, (traindata, batch_classes) in enumerate(train_dataloader):
                inputs, labels = self._transform_data(traindata, batch_classes, train=True)  
                optimizer.zero_grad()
                outputs = self._model(inputs)
                # out = torch.sigmoid(outputs).data.cpu().numpy()
                trainloss = self.train_criterion(outputs, labels)
                trainloss.backward()
                optimizer.step()
                print("Epoch: {:d} Batch: {:d} Train Loss: {:.6f}".format(epoch, batch_idx, trainloss.item()))
                sys.stdout.flush()
                batch_losses.append(trainloss.item())
            train_loss = np.mean(batch_losses)
            print("Training Loss: {:.6f}".format(train_loss))
        

    def evaluate(self, dataset_path):
        dataset = utils.dataset.load_dataset_of_image_files(
            dataset_path, 
            min_image_size=32, 
            max_image_size=self.max_image_size, 
            mode='RGB')

        mu = [0.48233507, 0.48233507, 0.48233507]
        std = [0.07271624, 0.07271624, 0.07271624]

        torch_dataset = ImageDataset(
            rafiki_dataset=dataset,
            image_scale_size=128,
            norm_mean=mu,
            norm_std=std,
            is_train=False
        )

        torch_dataloader = DataLoader(torch_dataset, batch_size=self.batch_size)

        self._model.eval()
        batch_losses = []
        outs = []
        gts = []
        with torch.no_grad():
            for batch_idx, (batch_data, batch_classes) in enumerate(torch_dataloader):
                inputs, labels = self._transform_data(batch_data, batch_classes, train=True)  
                outputs = self._model(inputs)

                loss = self.train_criterion(outputs, labels)

                batch_losses.append(loss.item())

                outs.extend(outputs)
                gts.extend(labels)

        valid_loss = np.mean(batch_losses)

        print("Validation Loss: {:.6f}".format(valid_loss))

        _, epoch_auc, _, acc = self.get_peformance_metrics(gts=gts, probabilities=outs)

        return acc

    def predict(self, queries):
        outs = []
        images = utils.dataset.transform_images(queries, image_size=128, mode='RGB')
        (images, _, _) = utils.dataset.normalize_images(images, self._normalize_mean, self._normalize_std)

        self._model.eval()

        with torch.no_grad():
            pass

        return outs

    def dump_parameters(self):
        params = {}
        # Save model parameters
        model_bytes = pickle.dumps(self._model)
        model_base64 = base64.b64encode(model_bytes).decode('utf-8')
        params['model_base64'] = model_base64
        # Save image size
        params['image_size'] = self._image_size 

        # self._model.save(tmp.name) ### save model  
        return params

    def load_parameters(self, params):
        ### Load model params
        # model_base64 = params['model_base64']
        # model_bytes = base64.b64decode(model_base64.encode('utf-8'))
        # Load image size
        self._image_size = params['image_size']

        ### load model
        # self._model = keras.models.load_model(tmp.name)  
        # if args.model == "densenet":
        model = DenseNet121(params)
        try: model.load_state_dict(torch.load('densenet.ckpt'))  ### tmp.name
        except: model.load_state_dict(torch.load('densenet.ckpt',map_location='cpu'))
        model.eval()
        try: model = model.cuda()
        except: pass
        self._model =model

        ### params for normalization 
        # self._normalize_mean = json.loads(params['normalize_mean'])
        # self._normalize_std = json.loads(params['normalize_std'])

        ### add seed

        lr = self._knobs.get('lr') or self.lr

    def _transform_data(self, data, labels, train=False):
        """
        Send data to GPU
        """
        inputs = data
        labels = labels.type(torch.LongTensor)
        try:
            inputs, labels = inputs.cuda(), labels.cuda()
        except: 
            pass
        inputs = Variable(inputs, requires_grad=False, volatile=not train)
        labels = Variable(labels, requires_grad=False, volatile=not train)
        return inputs, labels

class ImageDataset(Dataset):
    """
    A Pytorch encapsulation for rafiki ImageFilesDataset
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

    def __len__(self):
        return self.rafiki_dataset.size

    def __getitem__(self, idx):
        image, image_class = self.rafiki_dataset.get_item(idx)
        image_class = torch.tensor(image_class)
        if self._transform:
            image = self._transform(image)
        else:
            image = torch.tensor(image)

        return (image, image_class)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='data/train.zip', help='Path to train dataset')
    parser.add_argument('--val_path', type=str, default='data/val.zip', help='Path to validation dataset')
    parser.add_argument('--test_path', type=str, default='data/test.zip', help='Path to test dataset')
    print (os.getcwd())
    parser.add_argument('--query_path', type=str, default='examples/data/image_classification/xray_1.jpeg',
                        help='Path(s) to query image(s), delimited by commas')  ### os.getcwd()  Error of path setting  examples/data/image_classification/xray_1.jpeg
    (args, _) = parser.parse_known_args()

    queries = utils.dataset.load_images(args.query_path.split(',')).tolist()

    '''(model_file_path: str, 
        model_class: str, task: str, 
        dependencies: Dict[str, str], 
        train_dataset_path: str, 
        val_dataset_path: str, 
        test_dataset_path: str = None, 
        budget: Budget = None, 
        train_args: Dict[str, any] = None, 
        queries: List[Any] = None) -> (List[Any], BaseModel):'''
    test_model_class(
        model_file_path=__file__,
        model_class='PyDenseNet',
        task='IMAGE_CLASSIFICATION',
        dependencies={
            ModelDependency.TENSORFLOW: '1.12.0'
        },
        train_dataset_path=args.train_path,
        val_dataset_path=args.val_path,
        test_dataset_path=args.test_path,
        queries=queries
    )  ### model --> py_model_class : dev.py model 
