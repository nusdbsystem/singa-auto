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



'''
acc --> list
drift detect/gmm
bbox
explanation 

'''



KnobConfig = Dict[str, BaseKnob]
Knobs = Dict[str, Any]
Params = Dict[str, Union[str, int, float, np.ndarray]]


class DenseNet121(nn.Module): ### 
    def __init__(self, args):
        super(DenseNet121, self).__init__() ###
        self._model = densenet121(pretrained=not args.scra
    batch_classes
    batchClasses

    val_avg_loss
    valAvgLossch, drop_rate=args.drop_rate)  ### (num_classes=K, drop_rate=drop_rate)  ### pretrained=not self.scratch
        num_ftrs = self._model.classifier.in_features
        self._model.classifier = nn.Linear(num_ftrs, args.
    batch_classes
    batchClasses

    val_avg_loss
    valAvgLossum_classes) 
        # self.config = args

    def forward(self, x):
        return self._model(x)

def weighted_loss(args, preds, target, epoch=1):
    p_count = (args.all_labels == 1).sum(axis=0)
    args.p_count = p_count
    n_count = (args.all_labels == 0).sum(axis=0)
    total = p_count + n_count

    # invert *opposite* weights to obtain weighted loss
    # (positives weighted higher, all weights same across batches, and p_weight + n_weight == 1)
    p_weight = n_count / total
    n_weight = p_count / total

    p_weight_loss = Variable(torch.FloatTensor([p_weight]), requires_grad=False)
    n_weight_loss = Variable(torch.FloatTensor([n_weight]), requires_grad=False)

    weights = target.type(torch.FloatTensor) * (p_weight_loss.expand_as(target)) + \
              (target == 0).type(torch.FloatTensor) * (n_weight_loss.expand_as(target))


    try: weights = weights.cuda()
    except: pass
    loss = 0.0
    target=target.view(preds.shape)
    weights=weights.view(preds.shape)

    for i in range(args.num_classes):
        loss += nn.functional.binary_cross_entropy_with_logits(preds[:, i], target[:, i].t(), weight=weights[:, i].t())
    return loss / args.num_classes


def get_loss( dataset, weighted):
    criterion = nn.MultiLabelSoftMarginLoss()
    def loss(args, preds, target, epoch):
        if weighted:
            return weighted_loss(args, preds, target, epoch=epoch)  ### criterion
        else:
            return criterion(preds, target)
    return loss

### panda to cuda
def transform_data(data, labels, train=False):
    labels=labels
    inputs = data
    labels = labels.type(torch.FloatTensor)
    try: inputs , labels = inputs.cuda(), labels.cuda()
    except: pass
    inputs = Variable(inputs, requires_grad=False, volatile=not train)
    labels = Variable(labels, requires_grad=False, volatile=not train)
    return inputs, labels


class panda_func_densenet121(BaseModel):

    @staticmethod
    def get_knob_config():
        return {
            'max_epochs': FixedKnob(10), 
            # 'learning_rate': FloatKnob(1e-5, 1e-2, is_exp=True),
            'batch_size': CategoricalKnob([4]),
            # 'max_image_size': CategoricalKnob([512]),
            'max_iter': FixedKnob(20),
            # 'kernel': CategoricalKnob(['rbf', 'linear', 'poly']),
            # 'gamma': CategoricalKnob(['scale', 'auto']),
            # 'C': FloatKnob(1e-4, 1e4, is_exp=True),
            # 'max_image_size': CategoricalKnob([16, 32]),
            # 'max_depth': IntegerKnob(1, 32),
            # 'splitter': CategoricalKnob(['best', 'random']),
            # 'criterion': CategoricalKnob(['gini', 'entropy']),
            # 'max_image_size': CategoricalKnob([16, 32])
            # 'trial_epochs': FixedKnob(300),
            # 'lr': FloatKnob(1e-4, 1, is_exp=True),
            # 'lr_decay': FloatKnob(1e-3, 1e-1, is_exp=True),
            # 'opt_momentum': FloatKnob(0.7, 1, is_exp=True),
            # 'opt_weight_decay': FloatKnob(1e-5, 1e-3, is_exp=True),
            # 'batch_size': CategoricalKnob([32, 64, 128]),
            # 'drop_rate': FloatKnob(0, 0.4),
            'max_image_size': FixedKnob(32),   ### scale 
            'share_params': CategoricalKnob(['SHARE_PARAMS']),
            # Affects whether training is shortened by using early stopping
            # 'quick_train': PolicyKnob('EARLY_STOP'), 
            # 'early_stop_train_val_samples': FixedKnob(1024),
            # 'early_stop_patience_epochs': FixedKnob(5),

            ### panda params
            'model':CategoricalKnob(['densenet']),
            'tag':CategoricalKnob(['relabeled']),
            'optimizer':CategoricalKnob(['adam']),
            'save_path':CategoricalKnob(['ckpt_densenet']),
            # 'epochs':FixedKnob(10),
            # 'batch_size':FixedKnob(8),
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




    def __init__(self, **knobs):  ### 
        super().__init__(**knobs) ###
        self.__dict__.update(knobs) 
        ### panda



    def panda_evaluate(self, gts, probabilities, use_only_index = None):
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



    ### with pydensenet load
    ### model_inst.train(train_dataset_path, shared_params=shared_params, **(train_args or {})) ### call by dev.py
    def train(self, dataset_path, **kwargs):###  args

        ### pydensenet load dataset rafiki dataload
        dataset = utils.dataset.load_dataset_of_image_files(dataset_path, min_image_size=32, max_image_size=self.max_image_size, mode='RGB') ### if_shuffle=True  modify here for densenet
        (images, classes) = zip(*[(image, image_class) for (image, image_class) in dataset]) ###
        self.all_labels=np.array(classes)

        ### pydensenet train_val
        train_val_samples = dataset.size // 5 # up to 1/5 of samples for train-val
        (train_images, train_classes) = (images[train_val_samples:], classes[train_val_samples:])
        (train_val_images, train_val_classes) = (images[:train_val_samples], classes[:train_val_samples])

        self.num_classes =   max((np.array([classes[0]])).shape) ###  take the longer edge, need to be modified for densenet
        self._image_size = dataset.image_size ###

        self._model = DenseNet121(self) 
        ### pydensenet normalization
        # Compute normalization params from train data
        norm_mean = np.mean(np.asarray(images) / 255, axis=(0, 1, 2)).tolist()
        norm_std = np.std(np.asarray(images) / 255, axis=(0, 1, 2)).tolist()

        ### get pysensenet train dataset 
        ### pydensenet normalization 
        train_dataset = ImageDataset(train_images, train_classes, dataset.image_size, norm_mean, norm_std, is_train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        train_val_dataset = ImageDataset(train_val_images, train_val_classes, dataset.image_size, norm_mean, norm_std, is_train=False)
        val_dataloader = DataLoader(train_val_dataset, batch_size=self.batch_size)

        ### criteria 
        self.train_criterion = get_loss(train_dataloader, self.train_weighted)
        self.val_criterion = get_loss(val_dataloader, self.valid_weighted)
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

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, threshold=0.001, factor=0.1)
        best_model_wts, best_loss = self._model.state_dict(), float("inf")

        ### panda if args.enable_gm_prior:

        counter = 0
        ### model to cuda
        try: self._model = self._model.cuda()
        except: pass

        start_time = time.time()
        for epoch in range(1, self.max_epochs + 1):
            print("Epoch {}/{}".format(epoch, self.max_epochs))
            print("-" * 10)
            ### panda train_epoch
            ### panda regularization= gm_optimizer ### loader <--train/test set
            self._model.train()
            batch_losses = []
            for (traindata, batch_classes)  in train_dataloader:
                ### panda data to cuda
                inputs ,labels = transform_data(traindata,batch_classes, train=True)  
                optimizer.zero_grad()
                outputs = self._model(inputs)
                # out = torch.sigmoid(outputs).data.cpu().numpy()
                trainloss = self.train_criterion(self, outputs, labels, epoch=epoch)  ### weighted_loss(args, preds, target, epoch=epoch) ### MultiLabelSoftMarginLoss
                trainloss.backward()

                # if args.enable_gm_prior:
     
                optimizer.step()
                # print("Epoch: {:d} Batch: {:d} ({:d}) Train Loss: {:.6f}".format(epoch, self.batch_size, trainloss.item()))
                sys.stdout.flush()
                batch_losses.append(trainloss.item())
            train_loss = np.mean(batch_losses)
            print("Training Loss: {:.6f}".format(train_loss))

            ### panda test_epoch(model, loader, criterion, epoch=1):
            """Returns: (AUC, ROC AUC, F1, validation loss)"""
            outs=self.predict(val_dataloader, epoch=1)

            # ### panda_evaluate
            # _, epoch_auc, _, valid_loss= self.panda_evaluate(self.val_gts, outs) + (self.val_avg_loss,)
            # # scheduler.step(valid_loss)

            # if valid_loss < best_loss:
            #     best_loss = valid_loss
            #     # best_model_wts = self._model.state_dict()
            #     counter = 0
            # else:
            #     counter += 1

            # if counter > 3:
            #     break
            # ### torch save model
            # # torch.save(best_model_wts, os.path.join(args.save_path, "val%f_train%f_epoch%d" % (valid_loss, train_loss, epoch)))
            # # print("Elapsed Time: {}".format(time.time() - start_time))

    def evaluate(self, dataset_path):
        dataset = utils.dataset.load_dataset_of_image_files(dataset_path, min_image_size=32, max_image_size=self.max_image_size, mode='RGB') ### if_shuffle=True  modify here for densenet
        (images, classes) = zip(*[(image, image_class) for (image, image_class) in dataset]) ###

        # Compute normalization params from train data
        norm_mean = np.mean(np.asarray(images) / 255, axis=(0, 1, 2)).tolist()
        norm_std = np.std(np.asarray(images) / 255, axis=(0, 1, 2)).tolist()
        val_dataset = ImageDataset(images, classes, dataset.image_size, norm_mean, norm_std, is_train=False)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size)

        outs=self.predict(val_dataloader, epoch=1)

        ### panda_evaluate
        _, epoch_auc, _, acc= self.panda_evaluate(self.val_gts, outs)


        print (type(acc),acc)
        return acc



    def predict(self, queries,epoch):
        with torch.no_grad():
            self._model.eval()
            test_losses = []
            outs = []
            gts = []
            self.val_gts=None
            for (data, batch_classes) in queries:
                for gt in batch_classes.numpy().tolist():
                    gts.append(gt)
                inputs, labels = transform_data(data, batch_classes, train=False)
                outputs = self._model(inputs)
                loss = self.val_criterion(self, outputs, labels, epoch=epoch)  ### (args, preds, target, epoch=epoch)
                test_losses.append(loss.item())
                out = torch.sigmoid(outputs).data.cpu().numpy()
                outs.extend(out)
            self.val_avg_loss = np.mean(test_losses)
            # print("Validation Loss: {:.6f}".format(avg_loss))
            outs = np.array(outs)
            self.val_gts = np.array(gts).reshape(-1,self.num_classes) ### need to be modified
            
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




### pydensenet ImageDataset
class ImageDataset(Dataset):
    def __init__(self, images, classes, image_size, norm_mean, norm_std, is_train=False):
        self._images = images
        self._classes = classes
        if is_train:
            self._transform = transforms.Compose([
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)
            ])
        else:
            self._transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)
            ])

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image = self._images[idx]
        image_class =  self._classes[idx]

        image_class = torch.tensor(image_class)  ### panda : image_class = image_class.type(torch.FloatTensor)
        if self._transform:
            image = self._transform(Image.fromarray(image))
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
        model_class='panda_func_densenet121',
        task='IMAGE_CLASSIFICATION',
        dependencies={
            ModelDependency.TENSORFLOW: '1.12.0'
        },
        train_dataset_path=args.train_path,
        val_dataset_path=args.val_path,
        test_dataset_path=args.test_path,
        queries=queries
    )  ### model --> py_model_class : dev.py model 
