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
import traceback
import io
from collections import Counter
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
import numpy
import matplotlib.pyplot as plt
import copy
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.cm as mpl_color_map
# Panda Modules Dependency
from rafiki.panda.modules.explanations.lime.lime import Lime
from rafiki.panda.modules.explanations.gradcam.gradcam import GradCam
from rafiki.panda.modules.mod_modelslicing.models import create_sr_scheduler, upgrade_dynamic_layers
from rafiki.panda.modules.mod_gmreg.gm_prior_optimizer_pytorch import GMOptimizer
from rafiki.panda.modules.mod_driftadapt import LabelDriftAdapter
from rafiki.panda.modules.mod_spl.spl import SPL
from rafiki.panda.modules.mod_mcdropout.mc_dropout import update_model
from rafiki.panda.datasets.PandaTorchImageDataset import PandaTorchImageDataset

KnobConfig = Dict[str, BaseKnob]
Knobs = Dict[str, Any]
Params = Dict[str, Union[str, int, float, np.ndarray]]

class PandaTorchBasicModel(PandaModel):
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
        self.label_mapper = dict()

    @abc.abstractclassmethod
    def _create_model(self, scratch: bool, num_classes: int):
        raise NotImplementedError

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
            'spl_threshold_init':FixedKnob(16.0),
            'spl_mu':FixedKnob(1.3),
            'enable_lossrevise':FixedKnob(False),
            'lossrevise_slop':FixedKnob(2.0),

            # Label Adaptation
            'enable_label_adaptation':FixedKnob(True),

            # GM Prior Regularization
            'enable_gm_prior_regularization':FixedKnob(False),
            'gm_prior_regularization_a':FixedKnob(0.001),
            'gm_prior_regularization_b':FixedKnob(0.0001),
            'gm_prior_regularization_alpha':FixedKnob(0.5),
            'gm_prior_regularization_num':FixedKnob(4),
            'gm_prior_regularization_lambda':FixedKnob(0.0001),
            'gm_prior_regularization_upt_freq':FixedKnob(100),
            'gm_prior_regularization_param_upt_freq':FixedKnob(50),
            
            # Explanation
            'enable_explanation':FixedKnob(False),
            'explanation_gradcam': FixedKnob(True),
            'explanation_lime': FixedKnob(True),

            # Model Slicing
            'enable_model_slicing':FixedKnob(False),
            'model_slicing_groups':FixedKnob(0),
            'model_slicing_rate':FixedKnob(1.0),
            'model_slicing_scheduler_type':FixedKnob('randomminmax'),
            'model_slicing_randnum':FixedKnob(1),

            # MC Dropout
            'enable_mc_dropout':FixedKnob(False)
        }

    def get_peformance_metrics(self, gts: np.ndarray, probabilities: np.ndarray, use_only_index = None):

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
            ### modified here
            # print('Class: {:3d} Count: {:6d} PR AUC: {:.4f} ROC AUC: {:.4f} F1: {:.3f} Acc: {:.3f}'.format(i, count, PR_AUC, ROC_AUC, F1, acc))

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
        # self._normalize_mean = [0.48233507, 0.48233507, 0.48233507]
        # self._normalize_std = [0.07271624, 0.07271624, 0.07271624]

        self._num_classes = dataset.classes
        self.label_mapper = dataset.label_mapper

        # construct the model
        self._model = self._create_model(
            scratch = self._knobs.get("scratch"),
            num_classes = self._num_classes
        )
        if self._knobs.get("enable_mc_dropout"):
            self._model = update_model(self._model)

        if self._knobs.get("enable_model_slicing"):
            self._model = upgrade_dynamic_layers(
                model=self._model, 
                num_groups=self._knobs.get("model_slicing_groups"), 
                sr_in_list=[0.5, 0.75, 1.0])
        
        if self._knobs.get("enable_gm_prior_regularization"):
            self._gm_optimizer = GMOptimizer()
            for name, f in self._model.named_parameters():
                self._gm_optimizer.gm_register(
                    name,
                    f.data.cpu().numpy(),
                    model_name="PyVGG",
                    hyperpara_list=[
                        self._knobs.get("gm_prior_regularization_a"),
                        self._knobs.get("gm_prior_regularization_b"),
                        self._knobs.get("gm_prior_regularization_alpha"),
                        ],
                    gm_num=self._knobs.get("gm_prior_regularization_num"),
                    gm_lambda_ratio_value=self._knobs.get("gm_prior_regularization_lambda"),
                    uptfreq=[
                        self._knobs.get("gm_prior_regularization_upt_freq"),
                        self._knobs.get("gm_prior_regularization_param_upt_freq")]
                )
        
        if self._knobs.get("enable_spl"):
            self._spl = SPL()

        train_dataset = PandaTorchImageDataset(
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
        print("self._num_classes is :   ", self._num_classes)

        self.train_criterion = nn.MultiLabelSoftMarginLoss() ### type(torch.FloatTensor)

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

        if self._knobs.get("enable_model_slicing"):
            sr_scheduler = create_sr_scheduler(
                scheduler_type=self._knobs.get("model_slicing_scheduler_type"),
                sr_rand_num=self._knobs.get("model_slicing_randnum"),
                sr_list=[0.5, 0.75, 1.0],
                sr_prob=None
            )
        
        for epoch in range(1, self._knobs.get("max_epochs") + 1):
            print("Epoch {}/{}".format(epoch, self._knobs.get("max_epochs")))
            batch_losses = []
            for batch_idx, (raw_indices, traindata, batch_classes) in enumerate(train_dataloader):
                inputs, labels = self._transform_data(traindata, batch_classes, train=True)
                optimizer.zero_grad()
                
                if self._knobs.get("enable_model_slicing"):
                    for sr_idx in next(sr_scheduler):
                        self._model.update_sr_idx(sr_idx)
                        outputs = self._model(inputs)
                        trainloss = self.train_criterion(outputs, labels)
                        trainloss.backward()
                else:
                    outputs = self._model(inputs)
                    trainloss = self.train_criterion(outputs, labels)

                    trainloss.backward()

                if self._knobs.get("enable_gm_prior_regularization"):
                    for name, f in self._model.named_parameters():
                        self._gm_optimizer.apply_GM_regularizer_constraint(
                            labelnum=1,
                            trainnum=0,
                            epoch=epoch,
                            weight_decay=self._knobs.get("weight_decay"),
                            f=f,
                            name=name,
                            step=batch_idx
                        )
                
                if self._knobs.get("enable_spl"):
                    train_dataset.update_sample_score(raw_indices, trainloss.detach().cpu().numpy())

                optimizer.step()
                print("Epoch: {:d} Batch: {:d} Train Loss: {:.6f}".format(epoch, batch_idx, trainloss.item()))
                sys.stdout.flush()
                batch_losses.append(trainloss.item())

            train_loss = np.mean(batch_losses)
            print("Training Loss: {:.6f}".format(train_loss))
            if self._knobs.get("enable_spl"):
                train_dataset.update_score_threshold(
                    threshold=self._spl.calculate_threshold_by_epoch(
                        epoch=epoch,
                        threshold_init=self._knobs.get("spl_threshold_init"),
                        mu=self._knobs.get("spl_mu")))
        
    def evaluate(self, dataset_path):
        dataset = utils.dataset.load_dataset_of_image_files(
            dataset_path, 
            min_image_size=32, 
            max_image_size=self._knobs.get("max_image_size"), 
            mode='RGB',
            lazy_load=True)

        torch_dataset = PandaTorchImageDataset(
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

        if self._knobs.get("enable_label_adaptation"):
            self._label_drift_adapter = LabelDriftAdapter(
                model=self._model,
                num_classes=self._num_classes)

        batch_losses = []
        outs = []
        gts = []
        with torch.no_grad():
            for batch_idx, (raw_indices, batch_data, batch_classes) in enumerate(torch_dataloader):
                inputs, labels = self._transform_data(batch_data, batch_classes, train=True)
                outputs = self._model(inputs)
                loss = self.train_criterion(outputs, labels)
                batch_losses.append(loss.item())
                outs.extend(torch.sigmoid(outputs).cpu().numpy())
                gts.extend(labels.cpu().numpy())

                if self._knobs.get("enable_label_adaptation"):
                    self._label_drift_adapter.accumulate_c(outputs, labels)

                print("Batch: {:d}".format(batch_idx))

        if self._knobs.get("enable_label_adaptation"):
            self._label_drift_adapter.estimate_cinv()

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
        print('begin to predict')
        print('mean and std', self._normalize_mean, self._normalize_std)
        ndarray_images, pil_images = utils.dataset.transform_images(queries, image_size=128, mode='RGB')
        (images, _, _) = utils.dataset.normalize_images(ndarray_images, self._normalize_mean, self._normalize_std)

        print(self._use_gpu)
        if self._use_gpu:
            self._model.cuda()
        self._model.eval()
        # images are size of (B, W, H, C)
        with torch.no_grad():
            if self._use_gpu:
                images = torch.FloatTensor(images).permute(0, 3, 1, 2).cuda()
            else:
                images = torch.FloatTensor(images).permute(0, 3, 1, 2)
            
            if self._knobs.get("enable_mc_dropout"):
                print("MC Dropout Enabled")
                trials_n = self._knobs.get("mc_trials_n")
            else:
                trials_n = 1
            outs = list()

            for i in range(trials_n):
                out = self._model(images)
                if self._knobs.get("enable_label_adaptation"):
                    out = self._label_drift_adapter.adapt(out).squeeze()
                    print('trail_num: ', i)
                    print('out is ', out)
                    print('==' * 100)
                else:
                    out = torch.sigmoid(out).cpu().squeeze()
                    print('trail_num: ', i)
                    print('out is ', out)
                    print('==' * 100)
                outs.append(out.numpy())

        result = dict()
        result['out'] = np.asarray(outs).tolist()
        result['explaination'] = {}
        result['mc_dropout'] = []

        if self._knobs.get("enable_explanation"):
            exp = self.local_explain(org_imgs=pil_images, images=ndarray_images, params={})
            if exp:
                result['explaination'] = exp
        if self._knobs.get("enable_mc_dropout"):
            mean_var_eles = dict()
            outs = np.asarray(outs)
            print("mean {}, var {}".format(np.mean(outs, axis=0), np.var(outs, axis=0)))
            label_index = 0
            for mean, var in zip(np.mean(outs, axis=0).squeeze().tolist(), np.var(outs, axis=0).squeeze().tolist()):
                mean_var_ele = dict()
                mean_var_ele['label'] = self.label_mapper[str(label_index)] if self.label_mapper.get(str(label_index)) \
                                                                               is not None else str(label_index)
                mean_var_ele['mean'] = mean
                mean_var_ele['std'] = var
                mean_var_eles['class_{}'.format(label_index)] = mean_var_ele
                label_index += 1

            result['mc_dropout'] = [mean_var_eles]
        return [result]

    def local_explain(self, org_imgs: Image, images: List[Any], params: Params) -> Dict:
        """
        Override PandaModel.local_explain

        Parameters:
            org_imgs: list of PIL.image
            images: list of images(ndarray)
            params: parameters 
        
        Return:
            explanations: list of explanations
        """
        print('begin local_explain')
        enable_gradcam = self._knobs.get("explanation_gradcam")
        enable_lime = self._knobs.get("explanation_lime")
        print('get method:', enable_gradcam, enable_lime)

        explanation = dict()
        explanation['lime_img'] = ''
        explanation['gradcam_img'] = ''

        if enable_lime:
            try:
                self._lime = Lime(self._model, self._image_size, self._normalize_mean, self._normalize_std, self._use_gpu)
                imgs_explained = self._lime.explain(images)
                explanation['lime_exp'] = imgs_explained.tolist()
                imgs_explained = self.convert_img_to_str(imgs_explained)
                explanation['lime_img'] = imgs_explained
            except:
                traceback.print_exc(file=sys.stdout)

        if enable_gradcam:
            try:
                gc = GradCam(self._model, 'vgg', None)
                (images, _, _) = utils.dataset.normalize_images(images, self._normalize_mean, self._normalize_std)
                images = images.swapaxes(3,1)
                images = images.swapaxes(2,3)
                cam = gc.generate_cam(images)
                combined_gradcam = self.combine_images(org_im=org_imgs[0], activation=cam)
                explanation['gradcam_exp'] = combined_gradcam.tolist()
                combined_gradcam = self.convert_img_to_str(combined_gradcam)
                explanation['gradcam_img'] = combined_gradcam
            except:
                traceback.print_exc(file=sys.stdout)
        return explanation

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
        params['label_mapper'] = json.dumps(self.label_mapper)

        if self._knobs.get("enable_label_adaptation"):
            params[self._label_drift_adapter.get_mod_name()] = self._label_drift_adapter.dump_parameters()

        return params

    def load_parameters(self, params):
        """
        Override BaseModel.load_parameters
        
        Write base64 encoded PyTorch model state dict to temp file and then read it back with torch.load.
        The other persistent hyperparameters are recovered by setting model's private property
        """
        # Load model parameters
        h5_model_base64 = params['h5_model_base64']
        self._image_size = params['image_size']
        self._normalize_mean = np.array(json.loads(params['normalize_mean']))
        self._normalize_std = np.array(json.loads(params['normalize_std']))
        self._num_classes = params['num_classes']
        self.label_mapper = json.loads(params['label_mapper'])

        with tempfile.NamedTemporaryFile() as tmp:
            # Convert back to bytes & write to temp file
            h5_model_bytes = base64.b64decode(h5_model_base64.encode('utf-8'))
            with open(tmp.name, 'wb') as f:
                f.write(h5_model_bytes)

            # Load model from temp file
            self._model = self._create_model(
                scratch=self._knobs.get("scratch"),
                num_classes=self._num_classes
            )
            if self._knobs.get("enable_mc_dropout"):
                self._model = update_model(self._model)
            if self._knobs.get("enable_model_slicing"):
                self._model = upgrade_dynamic_layers(
                    model=self._model, 
                    num_groups=self._knobs.get("model_slicing_groups"), 
                    sr_in_list=[0.5, 0.75, 1.0])
            self._model.load_state_dict(torch.load(tmp.name))
        
        if self._knobs.get("enable_label_adaptation"):
            self._label_drift_adapter = LabelDriftAdapter(
                model=self._model,
                num_classes=self._num_classes
            )
            self._label_drift_adapter.load_parameters(params=params[self._label_drift_adapter.get_mod_name()])

    def _transform_data(self, data, labels, train=False):

        """
        Send data to GPU
        """
        inputs = data
        labels = labels.type(torch.LongTensor)
        one_hot_labels = torch.zeros(labels.shape[0], 2)
        one_hot_labels[range(one_hot_labels.shape[0]), labels.squeeze()] = 1
        one_hot_labels = one_hot_labels.type(torch.FloatTensor)
        if self._use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()

        inputs = Variable(inputs, requires_grad=False)
        one_hot_labels = Variable(one_hot_labels, requires_grad=False)
        return inputs, one_hot_labels

    def combine_images(self,  org_im, activation, colormap_name='hsv'):
        '''
        org_im: PIL.Image, should be the same size with activation
        return:  list
        '''

        color_map = mpl_color_map.get_cmap(colormap_name)
        no_trans_heatmap = color_map(activation)

        heatmap = copy.copy(no_trans_heatmap)
        heatmap[:, :, 3] = 0.4
        heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
        no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

        # Apply heatmap on iamge
        heatmap_on_image = Image.new("RGBA", org_im.size)
        heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
        heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)

        return numpy.asarray(heatmap_on_image)

    def convert_img_to_str(self, arr):
        im = Image.fromarray(arr.astype("uint8"))
        rawBytes = io.BytesIO()
        encoding = 'utf-8'
        im.save(rawBytes, "PNG")
        rawBytes.seek(0)
        return base64.b64encode(rawBytes.read()).decode(encoding)

