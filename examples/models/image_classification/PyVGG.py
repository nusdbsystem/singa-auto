from __future__ import division
from __future__ import print_function
import os
import sys
import base64
import abc
import tempfile
import json
import argparse
from typing import Union, Dict, Optional, Any, List

# Rafiki Dependency
from rafiki.model import PandaModel, FloatKnob, CategoricalKnob, FixedKnob, IntegerKnob, PolicyKnob, utils
from rafiki.model.knob import BaseKnob
from rafiki.constants import ModelDependency
from rafiki.model.dev import test_model_class

# PyTorch Dependency
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.models.vgg import vgg11_bn
from torch.utils.data import Dataset, DataLoader

# Misc Third-party Machine-Learning Dependency
import sklearn.metrics
import numpy as np

# Panda Modules Dependency
from rafiki.panda.modelslicing.models import create_sr_scheduler, upgrade_dynamic_layers

KnobConfig = Dict[str, BaseKnob]
Knobs = Dict[str, Any]
Params = Dict[str, Union[str, int, float, np.ndarray]]

class ImageDataset(Dataset):
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

class PyVGG(PandaModel):
    """
    Implementation of PyTorch DenseNet
    """
    def __init__(self, **knobs):
        super().__init__(**knobs)
        self._knobs = knobs

        if torch.cuda.is_available():
            self._use_gpu = True
        else:
            self._use_gpu = False
        
        #Parameters not advised by rafiki advisor
        #NOTE: should be dumped/loaded in dump_parameter/load_parameter
        self._image_size = 128

        #The following parameters are determined when training dataset is loaded
        self._normalize_mean = []
        self._normalize_std = []
        self._num_classes = 2

    def _create_model(self, scratch: bool, num_classes: int):
        model = vgg11_bn(pretrained=not scratch)
        num_features = 4096
        model.classifier[6] = nn.Linear(num_features, num_classes) 
        return model

    @staticmethod
    def get_knob_config():
        return {
            # Learning parameters
            'lr':FixedKnob(0.0001), ### learning_rate
            'weight_decay':FixedKnob(0.0),
            'drop_rate':FixedKnob(0.0),
            'max_epochs': FixedKnob(1), 
            'batch_size': CategoricalKnob([32]),
            'max_iter': FixedKnob(20),
            'optimizer':CategoricalKnob(['adam']),
            'scratch':FixedKnob(True),

            # Data augmentation
            'max_image_size': FixedKnob(32),
            'share_params': CategoricalKnob(['SHARE_PARAMS']),
            'tag':CategoricalKnob(['relabeled']),
            'workers':FixedKnob(8),
            'seed':FixedKnob(123456),
            'scale':FixedKnob(512),
            'horizontal_flip':FixedKnob(True),
     
            # Hyperparameters for PANDA modules
            # Self-paced Learning and Loss Revision
            'enable_spl':FixedKnob(True),

            # Label Adaptation
            'enable_label_adapatation':FixedKnob(True),

            # GM Prior Regularization
            'enable_gm_prior_regularization':FixedKnob(True),
            'gm_prior_regularization_a':FixedKnob(0.001),
            'gm_prior_regularization_b':FixedKnob(0.0001),
            'gm_prior_regularization_alpha':FixedKnob(0.5),
            'gm_prior_regularization_num':FixedKnob(4),
            'gm_prior_regularization_lambda':FixedKnob(0.0001),
            'gm_prior_regularization_upt_freq':FixedKnob(100),
            'gm_prior_regularization_param_upt_freq':FixedKnob(50),

            # Model Slicing
            'enable_model_slicing':FixedKnob(False),
            'model_slicing_rate':CategoricalKnob([0.25, 0.5, 0.75, 1.0])
        }

    def get_peformance_metrics(self, gts: np.ndarray, probabilities: np.ndarray, use_only_index = None):
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

        classes = [use_only_index] if use_only_index is not None else range(self._num_classes)

        for i in classes:
            try:
                PR_AUC, ROC_AUC, F1, acc, count = compute_metrics_for_class(i)
            except ValueError:
                continue
            PR_AUCs.append(PR_AUC)
            ROC_AUCs.append(ROC_AUC)
            F1s.append(F1)
            accs.append(acc)
            counts.append(count)

            print('Class: {:3d} Count: {:6d} PR AUC: {:.4f} ROC AUC: {:.4f} F1: {:.3f} Acc: {:.3f}'.format(i, count, PR_AUC, ROC_AUC, F1, acc))

        avg_PR_AUC = np.average(PR_AUCs)
        avg_ROC_AUC = np.average(ROC_AUCs, weights=counts)  
        avg_F1 = np.average(F1s, weights=counts)

        print('Avg PR AUC: {:.3f}'.format(avg_PR_AUC))
        print('Avg ROC AUC: {:.3f}'.format(avg_ROC_AUC))
        print('Avg F1: {:.3f}'.format(avg_F1))
        return avg_PR_AUC, avg_ROC_AUC, avg_F1, np.mean(accs)

    def train(self, dataset_path: str, shared_params: Optional[Params] = None, **train_args):
        """
        Overide BaseModel.train()
        Train the model with given dataset_path

        parameters:
            dataset_path: path to dataset_path
                type: str
            **kwargs:
                optional arguments
        
        return:
            nothing
        """
        dataset = utils.dataset.load_dataset_of_image_files(
            dataset_path, 
            min_image_size=32, 
            max_image_size=self._knobs.get("max_image_size"), 
            mode='RGB', 
            lazy_load=True)
        self._normalize_mean, self._normalize_std = dataset.get_stat()
        self._num_classes = dataset.classes

        # construct the model
        self._model = self._create_model(
            scratch = self._knobs.get("scratch"),
            num_classes = self._num_classes
        )
        
        train_dataset = ImageDataset(
            rafiki_dataset=dataset, 
            image_scale_size=128, 
            norm_mean=self._normalize_mean, 
            norm_std=self._normalize_std, 
            is_train=True)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self._knobs.get("batch_size"), 
            shuffle=True)

        #Setup Criterion
        if self._num_classes == 2:
            self.train_criterion = nn.CrossEntropyLoss()
        else:
            self.train_criterion = nn.MultiLabelSoftMarginLoss()

        #Setup Optimizer
        if self._knobs.get("optimizer") == "adam":
            optimizer = optim.Adam(
                           filter(lambda p: p.requires_grad, self._model.parameters()),
                           lr=self._knobs.get("lr"),
                           weight_decay=self._knobs.get("weight_decay"))
        elif self._knobs.get("optimizer") == "rmsprop":
            optimizer = optim.RMSprop(
                           filter(lambda p: p.requires_grad, self._model.parameters()),
                           lr=self._knobs.get("lr"),
                           weight_decay=self._knobs.get("weight_decay"))
        else:
            raise NotImplementedError()

        #Setup Learning Rate Scheduler
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            patience=1, 
            threshold=0.001, 
            factor=0.1)

        
        if self._use_gpu:
            self._model = self._model.cuda()
        
        self._model.train()
        for epoch in range(1, self._knobs.get("max_epochs") + 1):
            print("Epoch {}/{}".format(epoch, self._knobs.get("max_epochs")))
            batch_losses = []
            for batch_idx, (traindata, batch_classes) in enumerate(train_dataloader):
                inputs, labels = self._transform_data(traindata, batch_classes, train=True)  
                optimizer.zero_grad()
                outputs = self._model(inputs)
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
            max_image_size=self._knobs.get("max_image_size"), 
            mode='RGB',
            lazy_load=True)

        torch_dataset = ImageDataset(
            rafiki_dataset=dataset,
            image_scale_size=128,
            norm_mean=self._normalize_mean,
            norm_std=self._normalize_std,
            is_train=False
        )

        torch_dataloader = DataLoader(
            torch_dataset, 
            batch_size=self._knobs.get("batch_size"))

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
                outs.extend(torch.sigmoid(outputs).cpu().numpy())
                gts.extend(labels.cpu().numpy())
                print("Batch: {:d}".format(batch_idx))

        valid_loss = np.mean(batch_losses)
        print("Validation Loss: {:.6f}".format(valid_loss))
        gts = np.array(gts)
        outs = np.array(outs)

        # in case that the ground truth has only one dimension 
        # i.e. is size of (N,) with integer elements of 0...C-1, where C is the number of classes
        # the ground truth array has to be "one hot" encoded for evaluating the performance metric
        if len(gts.shape) == 1:
            gts = np.eye(self._num_classes)[gts].astype(np.int64)

        pr_auc, roc_auc, f1, acc = self.get_peformance_metrics(gts=np.array(gts), probabilities=np.array(outs))

        return f1

    def predict(self, queries: List[Any]) -> List[Any]:
        """
        Overide BaseModel.predict()
        Making prediction using queries

        Parameters:
            queries: list of quries
        Return:
            outs: list of numbers indicating scores of classes
        """
        images = utils.dataset.transform_images(queries, image_size=128, mode='RGB')
        (images, _, _) = utils.dataset.normalize_images(images, self._normalize_mean, self._normalize_std)

        if self._use_gpu:
            self._model.cuda()

        self._model.eval()

        # images are size of (B, W, H, C)
        with torch.no_grad():
            try:
                images = torch.FloatTensor(images).permute(0, 3, 1, 2).cuda()
            except Exception:
                images = torch.FloatTensor(images).permute(0, 3, 1, 2)

            outs = self._model(images)
            outs = torch.sigmoid(outs).cpu()

        return outs.tolist()

    def local_explain(self, queries: List[Any], params: Params) -> List[Any]:
        """
        Override PandaModel.local_explain

        Parameters:
            queries: list of queries
            params: parameters 
        
        Return:
            explanations: list of explanations
        """

        return None


    def dump_parameters(self):
        """
        Override BaseModel.dump_parameters
        
        Write PyTorch model's state dict to file, then read it back and encode with base64 encoding.
        The encoded model and the other persistent hyperparameters are returned to Rafiki
        """
        params = {}

        # Save model parameters
        with tempfile.NamedTemporaryFile() as tmp:
            # Save whole model to temp h5 file
            state_dict = self._model.state_dict()
            torch.save(state_dict, tmp.name)
        
            # Read from temp h5 file & encode it to base64 string
            with open(tmp.name, 'rb') as f:
                h5_model_bytes = f.read()

            params['h5_model_base64'] = base64.b64encode(h5_model_bytes).decode('utf-8')

        # Save pre-processing params
        params['image_size'] = self._image_size
        params['normalize_mean'] = json.dumps(self._normalize_mean.tolist())
        params['normalize_std'] = json.dumps(self._normalize_std.tolist())
        params['num_classes'] = self._num_classes

        return params

    def load_parameters(self, params):
        """
        Override BaseModel.load_parameters
        
        Write base64 encoded PyTorch model state dict to temp file and then read it back with torch.load.
        The other persistent hyperparameters are recovered by setting model's private property
        """
        # Load model parameters
        h5_model_base64 = params['h5_model_base64']

        with tempfile.NamedTemporaryFile() as tmp:
            # Convert back to bytes & write to temp file
            h5_model_bytes = base64.b64decode(h5_model_base64.encode('utf-8'))
            with open(tmp.name, 'wb') as f:
                f.write(h5_model_bytes)

            # Load model from temp file
            self._model = DenseNet121(scratch=True, drop_rate=0, num_classes=2)
            self._model.load_state_dict(torch.load(tmp.name))
        
        # Load pre-processing params
        self._image_size = params['image_size']
        self._normalize_mean = np.array(json.loads(params['normalize_mean']))
        self._normalize_std = np.array(json.loads(params['normalize_std']))
        self._num_classes = params['num_classes']

    def _transform_data(self, data, labels, train=False):
        """
        Send data to GPU
        """
        inputs = data
        labels = labels.type(torch.LongTensor)
        if self._use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs = Variable(inputs, requires_grad=False)
        labels = Variable(labels, requires_grad=False)
        return inputs, labels

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

    test_model_class(
        model_file_path=__file__,
        model_class='PyDenseNet',
        task='IMAGE_CLASSIFICATION',
        dependencies={ 
            ModelDependency.TORCH: '1.0.1',
            ModelDependency.TORCHVISION: '0.2.2'
        },
        train_dataset_path=args.train_path,
        val_dataset_path=args.val_path,
        test_dataset_path=args.test_path,
        queries=queries
    )
