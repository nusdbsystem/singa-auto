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
import tempfile
import numpy as np
import json
import base64
import argparse
import urllib.request
import tarfile
import os
import time
from tqdm import tqdm
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

from singa_auto.model.dev import test_model_class
from singa_auto.constants import ModelDependency
from singa_auto.model import BaseModel, FloatKnob, CategoricalKnob, FixedKnob, utils

from singa import singa_wrap as singa
from singa import opt
from singa import device
from singa import tensor
from singa import sonnx
from singa import layer
from singa import autograd
import onnx


def download_model(url):
    download_dir = '/tmp/'
    with tarfile.open(check_exist_or_download(url), 'r') as t:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(t, path=download_dir)


def check_exist_or_download(url):
    download_dir = '/tmp/'
    name = url.rsplit('/', 1)[-1]
    filename = os.path.join(download_dir, name)
    if not os.path.isfile(filename):
        urllib.request.urlretrieve(url, filename)
    return filename

# Data Augmentation
def augmentation(x, batch_size):
    xpad = np.pad(x, [[0, 0], [0, 0], [4, 4], [4, 4]], 'symmetric')
    for data_num in range(0, batch_size):
        offset = np.random.randint(8, size=2)
        x[data_num, :, :, :] = xpad[data_num, :,
                                    offset[0]:offset[0] + x.shape[2],
                                    offset[1]:offset[1] + x.shape[2]]
        if_flip = np.random.randint(2)
        if (if_flip):
            x[data_num, :, :, :] = x[data_num, :, :, ::-1]
    return x


# Calculate Accuracy
def accuracy(pred, target):
    # y is network output to be compared with ground truth (int)
    y = np.argmax(pred, axis=1)
    a = y == target
    correct = np.array(a, "int").sum()
    return correct


def resize_dataset(x, image_size):
    num_data = x.shape[0]
    dim = x.shape[1]
    X = np.zeros(shape=(num_data, dim, image_size, image_size),
                 dtype=np.float32)
    for n in range(0, num_data):
        for d in range(0, dim):
            X[n, d, :, :] = np.array(Image.fromarray(x[n, d, :, :]).resize(
                (image_size, image_size), Image.BILINEAR),
                dtype=np.float32)
    return X


class OnnxModelBase(BaseModel):
    '''
    Implements base model on the ONNX of SINGA for IMAGE_CLASSIFICATION
    '''

    def __init__(self, model_url, model_path, singa_model, device_id, **knobs):
        super().__init__(**knobs)
        self._knobs = knobs
        self.__dict__.update(knobs)
        self.model_url = model_url
        self.model_path = model_path
        self.singa_model = singa_model
        self.dev = device.create_cuda_gpu_on(device_id)
        self.dev.SetRandSeed(0)
        np.random.seed(0)

    def train(self, dataset_path, **kwargs):
        max_image_size = 224
        batch_size = int(self._knobs.get('batch_size'))
        max_epoch = self._knobs.get('max_epochs')

        # Define plot for loss against epochs
        utils.logger.define_plot('Loss Over Epochs',
                                 ['loss', 'early_stop_val_loss'],
                                 x_axis='epoch')

        dataset = utils.dataset.load_dataset_of_image_files(
            dataset_path,
            min_image_size=32,
            max_image_size=max_image_size,
            mode='RGB')

        self._image_size = max_image_size
        self._batch_size = batch_size
        self._num_classes = dataset.classes
        (images, classes) = zip(*[(image, image_class)
                                  for (image, image_class) in dataset])
        (images, self._normalize_mean,
         self._normalize_std) = utils.dataset.normalize_images(images)
        images = np.transpose(np.asarray(
            images, dtype=np.float32), [0, 3, 1, 2])  # channel first
        classes = np.asarray(classes, dtype=np.int32)

        train_x, val_x, train_y, val_y = train_test_split(
            images, classes, test_size=0.2, random_state=42)

        # compile the model
        tx = tensor.Tensor(
            (batch_size, 3, max_image_size, max_image_size), self.dev,
            tensor.float32)
        ty = tensor.Tensor((batch_size,), self.dev, tensor.int32)
        num_train_batch = train_x.shape[0] // batch_size
        num_val_batch = val_x.shape[0] // batch_size
        idx = np.arange(train_x.shape[0], dtype=np.int32)

        self._model = self._build_model(self._num_classes, max_image_size)
        self._model.compile([tx], is_train=True,
                            use_graph=True, sequential=False)
        # Training and Evaluation Loop
        for epoch in range(max_epoch):
            start_time = time.time()
            np.random.shuffle(idx)

            print('Starting Epoch %d:' % (epoch))
            # Training Phase
            train_correct = np.zeros(shape=[1], dtype=np.float32)
            test_correct = np.zeros(shape=[1], dtype=np.float32)
            train_loss = np.zeros(shape=[1], dtype=np.float32)
            test_loss = np.zeros(shape=[1], dtype=np.float32)

            self._model.train()
            for b in tqdm(range(num_train_batch)):
                # Generate the patch data in this iteration
                x = train_x[idx[b * batch_size:(b + 1) * batch_size]]
                if self._model.dimension == 4:
                    x = augmentation(x, batch_size)
                    if (x.shape[2] != self._model.input_size):
                        x = resize_dataset(x,  self._model.input_size)
                y = train_y[idx[b * batch_size:(b + 1) * batch_size]]

                # Copy the patch data into input tensors
                tx.copy_from_numpy(x)
                ty.copy_from_numpy(y)

                # Train the model
                out, loss = self._model(tx, ty, 'fp32', None)
                train_correct += accuracy(tensor.to_numpy(out), y)
                train_loss += tensor.to_numpy(loss)[0]

            # Evaluation Phase
            for b in tqdm(range(num_val_batch)):
                x = val_x[b * batch_size:(b + 1) * batch_size]
                if self._model.dimension == 4:
                    if (x.shape[2] != self._model.input_size):
                        x = resize_dataset(x, self._model.input_size)
                y = val_y[b * batch_size:(b + 1) * batch_size]
                tx.copy_from_numpy(x)
                ty.copy_from_numpy(y)
                out_test, loss_test = self._model(tx, ty, None, None)
                self._x = tx
                self._y = out_test
                test_correct += accuracy(tensor.to_numpy(out_test), y)
                test_loss += tensor.to_numpy(loss_test)[0]

            self._on_train_epoch_end(epoch=epoch, logs={
                'loss': float(train_loss/num_train_batch),
                'val_loss': float(test_loss/num_val_batch)
            })

        utils.logger.log('Train loss: {}'.format(
            float(train_loss/num_train_batch)))
        utils.logger.log('Train accuracy: {}'.format(
            float(train_correct/num_train_batch)))

    def evaluate(self, dataset_path):
        max_image_size = self._image_size
        batch_size = self._batch_size
        dataset = utils.dataset.load_dataset_of_image_files(
            dataset_path,
            min_image_size=32,
            max_image_size=max_image_size,
            mode='RGB')
        (images, classes) = zip(*[(image, image_class)
                                  for (image, image_class) in dataset])
        (images, self._normalize_mean,
         self._normalize_std) = utils.dataset.normalize_images(images)
        images = np.transpose(np.asarray(
            images, dtype=np.float32), [0, 3, 1, 2])
        classes = np.asarray(classes, dtype=np.int32)

        tx = tensor.Tensor(
            (batch_size, 3, max_image_size, max_image_size), self.dev,
            tensor.float32)
        ty = tensor.Tensor((batch_size,), self.dev, tensor.int32)
        num_batch = images.shape[0] // batch_size
        idx = np.arange(images.shape[0], dtype=np.int32)

        correct = np.zeros(shape=[1], dtype=np.float32)
        loss = np.zeros(shape=[1], dtype=np.float32)

        # Evaluation Phase
        for b in tqdm(range(num_batch)):
            x = images[b * batch_size:(b + 1) * batch_size]
            if self._model.dimension == 4:
                if (x.shape[2] != self._model.input_size):
                    x = resize_dataset(x, self._model.input_size)
            y = classes[b * batch_size:(b + 1) * batch_size]
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            out_test, loss_test = self._model(tx, ty, None, None)
            self._x = tx
            self._y = out_test
            correct += accuracy(tensor.to_numpy(out_test), y)
            loss += tensor.to_numpy(loss_test)[0]

        utils.logger.log('Validation loss: {}'.format(float(loss/num_batch)))

        return float(correct/num_batch)

    def predict(self, queries):
        image_size = self._image_size
        batch_size = self._batch_size
        max_image_size = 224

        images, _ = utils.dataset.transform_images(queries,
                                                image_size=image_size,
                                                mode='RGB')
        # (images, _, _) = utils.dataset.normalize_images(images,
                                                        # self._normalize_mean,
                                                        # self._normalize_std)
        images = np.transpose(np.asarray(
            images, dtype=np.float32), [0, 3, 1, 2])

        tx = tensor.Tensor(
            (batch_size, 3, max_image_size, max_image_size), self.dev,
            tensor.float32)
        ty = tensor.Tensor((batch_size,), self.dev, tensor.int32)
        num_batch = int(np.ceil(images.shape[0] / batch_size))
        idx = np.arange(images.shape[0], dtype=np.int32)
        probs = None

        # Evaluation Phase
        self._model.eval()
        for b in tqdm(range(num_batch)):
            x = images[b * batch_size:(b + 1) * batch_size]
            if self._model.dimension == 4:
                if (x.shape[2] != self._model.input_size):
                    x = resize_dataset(x, self._model.input_size)
            tx.copy_from_numpy(x)
            out = self._model(tx)
            out_probs = tensor.to_numpy(out)
            probs = out_probs if not probs else np.concatenate(
                (probs, out_probs), axis=0)

        return probs.tolist()

    def destroy(self):
        del self._model
        del self._x
        del self._y

    def dump_parameters(self):
        params = {}

        # Save model parameters
        with tempfile.NamedTemporaryFile() as tmp:

            # Save whole model
            model = sonnx.to_onnx([self._x], [self._y])
            onnx.save(model, tmp.name)
            # Read from temp h5 file & encode it to base64 string
            with open(tmp.name, 'rb') as f:
                model_bytes = f.read()

            params['model_bytes'] = base64.b64encode(model_bytes).decode(
                'utf-8')

        # Save pre-processing params
        params['image_size'] = self._image_size
        params['num_classes'] = self._num_classes
        params['normalize_mean'] = json.dumps(self._normalize_mean)
        params['normalize_std'] = json.dumps(self._normalize_std)

        return params

    def load_parameters(self, params):
        # Load model parameters
        model_base64 = params['model_bytes']
        self._image_size = params['image_size']
        self._num_classes = params['num_classes']
        self._batch_size = 1

        with tempfile.NamedTemporaryFile() as tmp:
            # Convert back to bytes & write to temp file
            model_base64 = base64.b64decode(model_base64.encode('utf-8'))
            with open(tmp.name, 'wb') as f:
                f.write(model_base64)

            # Load model from temp file
            onnx_model = onnx.load(tmp.name)
            num_classes = params['num_classes']
            image_size = params['image_size']

            self._model = self.singa_model(onnx_model, self._num_classes, self._image_size )

        # Load pre-processing params
        self._normalize_mean = json.loads(params['normalize_mean'])
        self._normalize_std = json.loads(params['normalize_std'])

    def _on_train_epoch_end(self, epoch, logs):
        loss = logs['loss']
        early_stop_val_loss = logs['val_loss']
        utils.logger.log(loss=loss,
                         early_stop_val_loss=early_stop_val_loss,
                         epoch=epoch)

    def _build_model(self, num_classes, image_size):
        lr = self._knobs.get('learning_rate')

        # read and make onnx model
        download_model(self.model_url)
        onnx_model = onnx.load(os.path.join(
            '/tmp', self.model_path))
        model = self.singa_model(onnx_model, num_classes, image_size)

        model.set_optimizer(
            opt.SGD(lr=lr, momentum=0.9, weight_decay=1e-5))
        return model
