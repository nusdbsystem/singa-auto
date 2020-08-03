#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

import abc
import base64
import io
import os
import tempfile
import zipfile
import tarfile
import json
import argparse
from typing import List
import urllib.request

from singa import singa_wrap as singa
from singa import opt
from singa import device
from singa import tensor
from singa import sonnx
from singa import layer
from singa import autograd
import onnx

from singa_auto.model.dev import test_model_class
from singa_auto.constants import ModelDependency
from singa_auto.model import FloatKnob, CategoricalKnob, FixedKnob, utils

from examples.models.image_classification.OnnxModelBase import OnnxModelBase


class MyModel(sonnx.SONNXModel):

    def __init__(self, onnx_model, num_classes=10, image_size=224, num_channels=3):
        super(MyModel, self).__init__(onnx_model)
        self.dimension = 4
        self.num_classes = num_classes
        self.input_size = image_size
        self.num_channels = num_channels
        self.linear = layer.Linear(4096, num_classes)

    def forward(self, *x):
        # if you change to other models, please update the output name here
        y = super(MyModel, self).forward(*x, last_layers=-3)[0]
        y = self.linear(y)
        return y

    def train_one_batch(self, x, y, dist_option, spars):
        out = self.forward(x)
        loss = autograd.softmax_cross_entropy(out, y)
        if dist_option == 'fp32':
            self.optimizer.backward_and_update(loss)
        elif dist_option == 'fp16':
            self.optimizer.backward_and_update_half(loss)
        elif dist_option == 'partialUpdate':
            self.optimizer.backward_and_partial_update(loss)
        elif dist_option == 'sparseTopK':
            self.optimizer.backward_and_sparse_update(loss,
                                                      topK=True,
                                                      spars=spars)
        elif dist_option == 'sparseThreshold':
            self.optimizer.backward_and_sparse_update(loss,
                                                      topK=False,
                                                      spars=spars)
        return out, loss

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer


global_config = {
    'model_url': 'https://github.com/onnx/models/raw/master/vision/classification/vgg/model/vgg16-7.tar.gz',
    'model_path': 'vgg16/vgg16.onnx',
    'device_id': 0,
}


class OnnxVgg16(OnnxModelBase):
    '''
    Implements VGG16 on the ONNX of SINGA for IMAGE_CLASSIFICATION
    '''

    @staticmethod
    def get_knob_config():
        return {
            'max_epochs': FixedKnob(10),
            'learning_rate': FloatKnob(1e-5, 1e-2, is_exp=True),
            'batch_size': CategoricalKnob([16, 32]),
        }

    def __init__(self, **knobs):
        super().__init__(model_url=global_config['model_url'], model_path=global_config[
            'model_path'], singa_model=MyModel, device_id=global_config['device_id'], **knobs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path',
                        type=str,
                        default='data/cifar10_train.zip',
                        help='Path to train dataset')
    parser.add_argument('--val_path',
                        type=str,
                        default='data/cifar10_val.zip',
                        help='Path to validation dataset')
    parser.add_argument('--test_path',
                        type=str,
                        default='data/cifar10_test.zip',
                        help='Path to test dataset')
    parser.add_argument(
        '--query_path',
        type=str,
        default='examples/data/image_classification/0-3096.png',
        help='Path(s) to query image(s), delimited by commas')
    (args, _) = parser.parse_known_args()

    queries = utils.dataset.load_images(args.query_path.split(','))
    test_model_class(model_file_path=__file__,
                     model_class='OnnxVgg16',
                     task='IMAGE_CLASSIFICATION',
                     dependencies={ModelDependency.SINGA: '3.0.0',
                                   ModelDependency.ONNX: '1.15.0'},
                     train_dataset_path=args.train_path,
                     val_dataset_path=args.val_path,
                     test_dataset_path=args.test_path,
                     queries=queries)
