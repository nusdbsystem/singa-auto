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

import pickle
import base64
import numpy as np
import argparse
import os
import random

from singa_auto.model import BaseModel, IntegerKnob, utils
from singa_auto.constants import ModelDependency
from singa_auto.model.dev import test_model_class
from PIL import Image
from io import BytesIO

from sklearn.neural_network import MLPRegressor

class MLPBatteryCapacityEstimator(BaseModel):

    '''
    This class defines a MLP battery capacity estimator.
    '''

    @staticmethod
    def get_knob_config():
        return {
            'num_hid_layers': IntegerKnob(2, 4)
        }

    def __init__(self, **knobs):
        self._knobs = knobs
        self.__dict__.update(knobs)

        self.time_interval = 10
        self.time_length = 60

        num_hid_layers = self._knobs.get("num_hid_layers")
        hidden_layer_sizes = [256] * int(num_hid_layers)
        
        self._regr = MLPRegressor(random_state=1, max_iter=100000, hidden_layer_sizes = hidden_layer_sizes)

    def train(self, dataset_path, work_dir = None, **kwargs):

        _, feat_train, tgt_train = self.read_discharge_data(dataset_path)
        self.min_arr, self.max_arr = self.get_min_max_arr(feat_train)
        feat_train = self.min_max_normalisation(feat_train, self.min_arr, self.max_arr)

        # Training.
        self._regr.fit(feat_train, tgt_train)

        # Compute R2 on the training set.
        R2_train = self._regr.score(feat_train, tgt_train)
        utils.logger.log('Train accuracy: {}'.format(R2_train))

    def evaluate(self, dataset_path,  work_dir = None, **kwargs):

        _, feat_eval, tgt_eval = self.read_discharge_data(dataset_path)
        feat_eval = self.min_max_normalisation(feat_eval, self.min_arr, self.max_arr)
        R2_eval = self._regr.score(feat_eval, tgt_eval)
        return R2_eval

    def predict(self, queries, work_dir = None):
        predictions = []
        for data_bytes in queries:
            f = open(work_dir + "/query.csv", "wb")
            f.write(data_bytes)
            f.close()
            _, feat, _ = self.read_discharge_data(work_dir + "/query.csv")
            feat = self.min_max_normalisation(feat, self.min_arr, self.max_arr)
            prediction = self._regr.predict(feat)
            predictions.append(str(np.mean(prediction)))
        return predictions

    def dump_parameters(self):
        params = pickle.dumps(self.__dict__)
        return params

    def load_parameters(self, params):
        self.__dict__ = pickle.loads(params)
    
    # x - data array
    # t - time array
    # d - time interval
    def data_alignment(self, x, t, d):
        x_ali = []
        t_ali = []

        i = 1
        td = 0
        while i < len(t):
            if td >= t[i-1] and td < t[i]:
                k = (x[i] - x[i-1]) * 1.0 / (t[i] - t[i-1])
                v = x[i-1] + k * (td - t[i-1])
                x_ali.append(v)
                t_ali.append(td)
                td = td + d
            else:
                i = i + 1
        return x_ali, t_ali

    def get_time_idx(self, t):
        t_idx = t[0]
        t_idx = t_idx * 12 + t[1]
        t_idx = t_idx * 31 + t[2]
        t_idx = t_idx * 24 + t[3]
        t_idx = t_idx * 60 + t[4]
        t_idx = t_idx * 60 + t[5]
        return t_idx

    def get_min_max_arr(self, feat_data):
        x = np.min(feat_data, axis = 0)
        y = np.max(feat_data, axis = 0)
        return x, y

    def min_max_normalisation(self, feat_data, min_arr, max_arr):
        res = []
        i = -1
        for x in np.array(feat_data).T:
            i = i + 1
            s = max_arr[i] - min_arr[i]
            y = x - min_arr[i]
            if s != 0:
                y = y / s
            res.append(y)
        res = np.array(res).T
        return res

    def read_discharge_data(self, data_file):

        time_interval = self.time_interval
        time_length = self.time_length

        f = open(data_file, "r")
        data_header = ""
        feat_data = []
        tgt_data = []
        for x in f.readlines():
            x = x.replace("\n","")
            if x == "":
                continue
            if data_header == "":
                data_header = x.split(",")
                continue

            if x == "<Start of Discharging>":
                data_matrix = []
                continue

            elif x == "<End of Discharging>":
                data_matrix = np.asarray(data_matrix)
                t = data_matrix[:,data_header.index("Time")]
                if "Capacity" in data_header:
                    target = data_matrix[-1,data_header.index("Capacity")]
                else:
                    target = None
                data_matrix_ali = []

                for i in range(len(data_header)):
                    if data_header[i] == "Capacity":
                        continue
                    elif data_header[i] == "Time":
                        continue
                    else:
                        v = data_matrix.T[i].T.tolist()
                        v_ali, t_ali = self.data_alignment(v, t, time_interval)
                        data_matrix_ali.append(v_ali)
                data_matrix_ali = np.asarray(data_matrix_ali).T
                for i in range(data_matrix_ali.shape[0]-time_length):
                    feature = data_matrix_ali[i:(i+time_length),:]
                    feature = feature.reshape(feature.shape[0] * feature.shape[1])
                    feat_data.append(feature)
                    tgt_data.append(target)
            else:
                x = x.split(",")
                x = [float(p) for p in x]
                data_matrix.append(x)

        return data_header, feat_data, tgt_data




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path',
                        type=str,
                        default='data/B0005_train.csv',
                        help='Path to train dataset')
    parser.add_argument('--val_path',
                        type=str,
                        default='data/B0005_eval.csv',
                        help='Path to validation dataset')
    parser.add_argument('--test_path',
                        type=str,
                        default='data/B0005_eval.csv',
                        help='Path to test dataset')
    parser.add_argument('--query_path',
                        type=str,
                        default='data/B0005_eval.csv,data/B0005_eval.csv',
                        help='Path(s) to query image(s), delimited by commas')
    (args, _) = parser.parse_known_args()

    query_file_list = args.query_path.split(',')
    queries = [open(fname, 'rb').read() for fname in query_file_list]

    test_model_class(model_file_path=__file__,
                     model_class='MLPBatteryCapacityEstimator',
                     task='GENERAL_TASK',
                     dependencies={ModelDependency.SCIKIT_LEARN: '0.20.0'},
                     train_dataset_path=args.train_path,
                     val_dataset_path=args.val_path,
                     test_dataset_path=args.test_path,
                     queries=queries)


