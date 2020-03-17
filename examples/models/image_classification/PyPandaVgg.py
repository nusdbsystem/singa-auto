from __future__ import division
from __future__ import print_function
import os
import argparse
from typing import Union, Dict, Optional, Any, List

# Rafiki Dependency
from rafiki.model import PandaModel, FloatKnob, CategoricalKnob, FixedKnob, IntegerKnob, PolicyKnob, utils
from rafiki.model.knob import BaseKnob
from rafiki.constants import ModelDependency
from rafiki.model.dev import test_model_class

# PyTorch Dependency
import torch.nn as nn
from torchvision.models.vgg import vgg11_bn

# Misc Third-party Machine-Learning Dependency
import numpy as np

# Panda Modules Dependency
from rafiki.panda.models.PandaTorchBasicModel import PandaTorchBasicModel

KnobConfig = Dict[str, BaseKnob]
Knobs = Dict[str, Any]
Params = Dict[str, Union[str, int, float, np.ndarray]]


class PyPandaVgg(PandaTorchBasicModel):
    """
    Implementation of PyTorch DenseNet
    """
    def __init__(self, **knobs):
        super().__init__(**knobs)

    def _create_model(self, scratch: bool, num_classes: int):
        model = vgg11_bn(pretrained=not scratch)
        num_features = 4096
        model.classifier[6] = nn.Linear(num_features, num_classes) 
        print("create model {}".format(model))
        return model

    @staticmethod
    def get_knob_config():
        return {
            # Learning parameters
            'lr':FixedKnob(0.0001), ### learning_rate
            'weight_decay':FixedKnob(0.0),
            'drop_rate':FixedKnob(0.0),
            'max_epochs': FixedKnob(3),
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
            'enable_explanation': FixedKnob(True),
            'explanation_gradcam': FixedKnob(True),
            'explanation_lime': FixedKnob(True),

            # Model Slicing
            'enable_model_slicing':FixedKnob(False),
            'model_slicing_groups':FixedKnob(0),
            'model_slicing_rate':FixedKnob(1.0),
            'model_slicing_scheduler_type':FixedKnob('randomminmax'),
            'model_slicing_randnum':FixedKnob(1),

            # MC Dropout
            'enable_mc_dropout':FixedKnob(True),
            'mc_trials_n':FixedKnob(10)
        }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='data/val.zip', help='Path to train dataset')
    parser.add_argument('--val_path', type=str, default='data/val.zip', help='Path to validation dataset')
    parser.add_argument('--test_path', type=str, default='data/val.zip', help='Path to test dataset')
    print (os.getcwd())
    parser.add_argument(
        '--query_path', 
        type=str, 
        default=
        'examples/data/image_classification/xray_1.png,examples/data/image_classification/fashion_mnist_test_1.png,examples/data/image_classification/cifar10_test_1.png',
        #'examples/data/image_classification/IM-0001-0001.jpeg,examples/data/image_classification/IM-0003-0001.jpeg,examples/data/image_classification/IM-0005-0001.jpeg',
        help='Path(s) to query image(s), delimited by commas')
    (args, _) = parser.parse_known_args()

    queries = utils.dataset.load_images(args.query_path.split(',')).tolist()
    
    test_model_class(
        model_file_path=__file__,
        model_class='PyPandaVgg',
        task='IMAGE_CLASSIFICATION',
        dependencies={ 
            ModelDependency.TORCH: '1.0.1',
            ModelDependency.TORCHVISION: '0.2.2',
            ModelDependency.CV2: '4.2.0.32'
        },
        train_dataset_path=args.train_path,
        val_dataset_path=args.val_path,
        test_dataset_path=args.test_path,
        queries=queries
    )
