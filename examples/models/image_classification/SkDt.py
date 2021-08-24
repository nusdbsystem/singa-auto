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

from sklearn import tree
import pickle
import base64
import numpy as np
import argparse
import os
import random

from singa_auto.model import ImageClfBase, IntegerKnob, CategoricalKnob, utils
from singa_auto.constants import ModelDependency
from singa_auto.model.dev import test_model_class
from PIL import Image
from io import BytesIO




class SkDt(ImageClfBase):
    '''
    This class defines a decision tree classifier on the MNIST / Fashion-MNIST dataset.
    '''

    @staticmethod
    def get_knob_config():
        return {
            'max_depth': IntegerKnob(1, 32),
            'splitter': CategoricalKnob(['best', 'random']),
            'criterion': CategoricalKnob(['gini', 'entropy'])
        }

    def __init__(self, **knobs):
        self._knobs = knobs
        self.__dict__.update(knobs)

        self._clf = tree.DecisionTreeClassifier(max_depth=self._knobs.get("max_depth"),
                                                criterion=self._knobs.get("criterion"),
                                                splitter=self._knobs.get("splitter"))

    def train(self, dataset_path, work_dir = None, **kwargs):
        dataset = utils.dataset.load_mnist_dataset(dataset_path)
        (images, classes) = zip(*[(np.asarray(image), image_class)
                                for (image, image_class) in dataset])
        
        X = self.image_flatten(images)
        y = classes

        # Training.
        self._clf.fit(X, y)

        # Compute accuracy on the training set.
        preds = self._clf.predict(X)
        accuracy = sum(y == preds) / len(y)
        utils.logger.log('Train accuracy: {}'.format(accuracy))

    def evaluate(self, dataset_path,  work_dir = None, **kwargs):
        dataset = utils.dataset.load_mnist_dataset(dataset_path)
        (images, classes) = zip(*[(np.asarray(image), image_class)
                                for (image, image_class) in dataset])
        X = self.image_flatten(images)
        y = classes
        preds = self._clf.predict(X)
        accuracy = sum(y == preds) / len(y)
        return accuracy

    def predict(self, queries, work_dir = None):
        X = []
        for png_bytes in queries:
            png_image = Image.open(BytesIO(png_bytes))
            png_array = np.asarray(png_image.convert("L"))
            X.append(png_array)
        X = np.asarray(X)
        X = self.image_flatten(X)
        probs = self._clf.predict_proba(X)
        return probs.tolist()

    def dump_parameters(self):
        params = pickle.dumps(self.__dict__)
        return params

    def load_parameters(self, params):
        self.__dict__ = pickle.loads(params)
    
    def image_flatten(self, images):
        X = np.asarray(images)
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        return X

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path',
                        type=str,
                        default='data/fashion_mnist_train.zip',
                        help='Path to train dataset')
    parser.add_argument('--val_path',
                        type=str,
                        default='data/fashion_mnist_val.zip',
                        help='Path to validation dataset')
    parser.add_argument('--test_path',
                        type=str,
                        default='data/fashion_mnist_test.zip',
                        help='Path to test dataset')
    parser.add_argument('--query_path',
                        type=str,
                        default='examples/data/image_classification/fashion_mnist_test_1.png,examples/data/image_classification/fashion_mnist_test_1.png',
                        help='Path(s) to query image(s), delimited by commas')
    (args, _) = parser.parse_known_args()

    query_file_list = args.query_path.split(',')
    queries = [open(fname, 'rb').read() for fname in query_file_list]

    test_model_class(model_file_path=__file__,
                     model_class='SkDt',
                     task='IMAGE_CLASSIFICATION',
                     dependencies={ModelDependency.SCIKIT_LEARN: '0.20.0'},
                     train_dataset_path=args.train_path,
                     val_dataset_path=args.val_path,
                     test_dataset_path=args.test_path,
                     queries=queries)
