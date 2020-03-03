from __future__ import division
from __future__ import print_function
import os
import sys
import argparse
from typing import Union, Dict, Optional, Any, List

# Rafiki Dependency
from rafiki.model import PandaModel, FloatKnob, CategoricalKnob, FixedKnob, IntegerKnob, PolicyKnob, utils
from rafiki.model.knob import BaseKnob
from rafiki.constants import ModelDependency
from rafiki.model.dev import test_model_class
from rafiki.panda.modules.mod_spl.spl import SPL
from rafiki.panda.datasets.PandaTorchImageDataset import PandaTorchImageDataset

# PyTorch Dependency
import torch
import torch.nn as nn
### to localize vgg
# from torchvision.models.vgg import vgg11_bn
from torch.utils.data import Dataset, DataLoader
import torch.utils.model_zoo as model_zoo
import torch.optim as optim
from torch.optim import lr_scheduler
import math


# Misc Third-party Machine-Learning Dependency
import numpy as np

# Panda Modules Dependency
from rafiki.panda.models.PandaTorchBasicModel import PandaTorchBasicModel

KnobConfig = Dict[str, BaseKnob]
Knobs = Dict[str, Any]
Params = Dict[str, Union[str, int, float, np.ndarray]]

### add vgg with selective net
__all__ = [
    'VGG',  'vgg11_bn', 
]

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
}

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(8192, 4096), # 512 * 7 * 7, 4096 (original) size to be decided, 8192, 4096 for cifar10
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self._initialize_weights()

        # Add selection head with respect to the SelectiveNet code
        self.selectionhead = nn.Sequential(
            nn.Linear(8192, 4096), 
            nn.ReLU(False), 
            nn.BatchNorm1d(4096), 
            nn.Linear(4096, 1),
            nn.Sigmoid(), 
        )
        # one output neuron with sigmoid

    def forward(self, x):
        x = self.features(x)
        # 4D to 2D, [BatchSize, 512, 4, 4] to [BatchSize, 8192]
        x = x.view(x.size(0), -1)
        # add slectionhead into forward
        selectionhead = self.selectionhead(x)
        x = self.classifier(x)
        return (x, selectionhead)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model

### end of adding vgg with selective net



class PyPandaVgg(PandaTorchBasicModel):
    """
    Implementation of PyTorch DenseNet
    """
    def __init__(self, **knobs):
        super().__init__(**knobs)

    # overwrite train function
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
        print ('num_class', dataset.classes)

        # construct the model
        self._model = self._create_model(
            scratch = self._knobs.get("scratch"),
            num_classes = self._num_classes
        )

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
        if self._num_classes == 2:
            self.train_criterion = nn.CrossEntropyLoss() # type(torch.LongTensor)
            # add selectionhead loss
            self.selectionhead_criterion = nn.CrossEntropyLoss() 
        else:

            self.train_criterion = nn.MultiLabelSoftMarginLoss() # type(torch.FloatTensor)
            # add selectionhead loss
            self.selectionhead_criterion = nn.MultiLabelSoftMarginLoss() 


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
        
        # SelectiveNet params
        lamda = self._knobs.get("lamda")
        selectionheadloss_weight = self._knobs.get("selectionheadloss_weight")
        target_coverage = self._knobs.get("target_coverage")

        for epoch in range(1, self._knobs.get("max_epochs") + 1):
            print("Epoch {}/{}".format(epoch, self._knobs.get("max_epochs")))
            batch_losses = []
            for batch_idx, (raw_indices, traindata, batch_classes) in enumerate(train_dataloader):
                inputs, labels = self._transform_data(traindata, batch_classes, train=True)  
                optimizer.zero_grad()
                
                if self._knobs.get("enable_model_slicing"):
                    for sr_idx in next(sr_scheduler):
                        self._model.update_sr_idx(sr_idx)
                        # add selection head outputs, selectionhead be a column
                        (outputs, selectionhead) = self._model(inputs)
                        predloss = self.train_criterion(outputs, labels)
                        # apply the Interoir Point Method on labels # same as selectionhead.view(-1, 1).repeat(1,self._num_classes).view(selectionhead.shape[0],-1) * labels
                        interior_point_of_labels = selectionhead * labels
                        auxiliaryhead=outputs
                        empirical_coverage = selectionhead.type(torch.float64).mean()
                        selectionheadloss= self.selectionhead_criterion(interior_point_of_labels, auxiliaryhead) + lamda * (target_coverage - empirical_coverage).clamp(min=0)**2
                        selectionheadloss = torch.tensor(selectionheadloss,dtype=torch.float).cuda()
                        trainloss = selectionheadloss * selectionheadloss_weight + predloss * (1 - selectionheadloss_weight)
                        trainloss.backward()
                else:
                    # add selection head outputs, selectionhead be a column
                    (outputs, selectionhead) = self._model(inputs)
                    predloss = self.train_criterion(outputs, labels)
                    # apply the Interoir Point Method on labels # same as selectionhead.view(-1, 1).repeat(1,self._num_classes).view(selectionhead.shape[0],-1) * labels
                    interior_point_of_labels = selectionhead * labels
                    auxiliaryhead=outputs
                    empirical_coverage = selectionhead.type(torch.float64).mean()
                    selectionheadloss = self.selectionhead_criterion(interior_point_of_labels, auxiliaryhead) + lamda * (target_coverage - empirical_coverage).clamp(min=0)**2
                    selectionheadloss = torch.tensor(selectionheadloss,dtype=torch.float).cuda()
                    trainloss = selectionheadloss * selectionheadloss_weight + predloss * (1 - selectionheadloss_weight)
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
        # SelectiveNet params   
        lamda = self._knobs.get("lamda")
        selectionheadloss_weight = self._knobs.get("selectionheadloss_weight")
        target_coverage = self._knobs.get("target_coverage")
        print('selectionhead: When the selectiionhead is lower than 0.5, the model is prone to make wrong pred')
        print('True means the model makes empirical pred correctly, False otherwise')

        with torch.no_grad():
            for batch_idx, (raw_indices, batch_data, batch_classes) in enumerate(torch_dataloader):
                inputs, labels = self._transform_data(batch_data, batch_classes, train=True) 
                (outputs, selectionhead) = self._model(inputs)
                predloss = self.train_criterion(outputs, labels)
                # loss intergrated with SelectiveNet
                interior_point_of_labels = selectionhead * labels
                auxiliaryhead=outputs
                empirical_coverage = selectionhead.type(torch.float64).mean()
                selectionheadloss= self.selectionhead_criterion(interior_point_of_labels, auxiliaryhead) + lamda * (target_coverage - empirical_coverage).clamp(min=0)**2
                selectionheadloss = torch.tensor(selectionheadloss,dtype=torch.float).cuda()
                loss = selectionheadloss * selectionheadloss_weight + predloss * (1 - selectionheadloss_weight)

                batch_losses.append(loss.item())
                outs.extend(torch.sigmoid(outputs).cpu().numpy())
                gts.extend(labels.cpu().numpy())
                if self._knobs.get("enable_label_adaptation"):
                    self._label_drift_adapter.accumulate_c(outputs, labels)

                print("Batch: {:d}".format(batch_idx))
                print('selectionhead: ', selectionhead[-1].cpu().numpy())
                print('Pred T/F: ', -0.5<labels[-1].cpu().numpy()-outs[-1]<0.5)

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

            (outs, selectionhead) = self._model(images)

            if self._knobs.get("enable_label_adaptation"):
                outs = self._label_drift_adapter.adapt(outs)
            else:
                outs = torch.sigmoid(outs).cpu()

        if self._knobs.get("enable_explanation"):
            self.local_explain(queries)

        print ('This value should be lower than 0.5, if the queries are not from X-Ray dataset')
        print (selectionhead)

        return outs.tolist()


    def _create_model(self, scratch: bool, num_classes: int):
        model = vgg11_bn(pretrained=not scratch)
        num_features = 4096
        # format the last classifier layer
        model.classifier[6] = nn.Linear(num_features, num_classes) 
        print("create model {}".format(model))
        return model

    @staticmethod
    def get_knob_config():
        return {
            # Learning parameters
            'lr':FixedKnob(0.0001), 
            'weight_decay':FixedKnob(0.0),
            'drop_rate':FixedKnob(0.0),
            'max_epochs': FixedKnob(10), # original 5
            'batch_size': CategoricalKnob([96]), # original 32
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
            'enable_label_adaptation':FixedKnob(False),

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
            'explanation_method':FixedKnob('lime'),

            # Model Slicing
            'enable_model_slicing':FixedKnob(False),
            'model_slicing_groups':FixedKnob(0),
            'model_slicing_rate':FixedKnob(1.0),
            'model_slicing_scheduler_type':FixedKnob('randomminmax'),
            'model_slicing_randnum':FixedKnob(1),

            # SelectiveNet
            'selectionheadloss_weight':FixedKnob(0.5),
            'target_coverage':FixedKnob(0.8),
            'lamda':FixedKnob(32)
        }



            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='data/train.zip', help='Path to train dataset')
    parser.add_argument('--val_path', type=str, default='data/val.zip', help='Path to validation dataset')
    parser.add_argument('--test_path', type=str, default='data/test.zip', help='Path to test dataset')
    print (os.getcwd())
    parser.add_argument(
        '--query_path', 
        type=str, 
        default=
        # 'examples/data/image_classification/xray_1.jpeg,examples/data/image_classification/IM-0103-0001.jpeg,examples/data/image_classification/NORMAL2-IM-0023-0001.jpeg',
        # 'examples/data/image_classification/IM-0001-0001.jpeg,examples/data/image_classification/IM-0003-0001.jpeg,examples/data/image_classification/IM-0005-0001.jpeg',
        'examples/data/image_classification/cifar10_test_1.png,examples/data/image_classification/cifar10_test_2.png,examples/data/image_classification/fashion_mnist_test_1.png,examples/data/image_classification/fashion_mnist_test_2.png',
        help='Path(s) to query image(s), delimited by commas')
    (args, _) = parser.parse_known_args()

    queries = utils.dataset.load_images(args.query_path.split(',')).tolist()
    
    test_model_class(
        model_file_path=__file__,
        model_class='PyPandaVgg',
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
