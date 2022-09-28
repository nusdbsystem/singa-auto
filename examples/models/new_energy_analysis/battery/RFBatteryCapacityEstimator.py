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

from singa_auto.model import BaseModel, IntegerKnob, CategoricalKnob, FloatKnob, utils
from singa_auto.constants import ModelDependency
from singa_auto.model.dev import test_model_class
from PIL import Image
from io import BytesIO

from sklearn.ensemble import RandomForestRegressor

class RFBatteryCapacityEstimator(BaseModel):

    '''
    This class defines a Random Forest battery capacity estimator.
    '''

    @staticmethod
    def get_knob_config():
        return {
            'max_depth': IntegerKnob(8, 16),
            'n_estimators': IntegerKnob(8, 32),
            'max_features': CategoricalKnob(['auto', 'sqrt', 'log2']),
            'min_impurity_decrease': FloatKnob(0.0, 0.05)
        }

    def __init__(self, **knobs):
        self._knobs = knobs
        self.__dict__.update(knobs)

        self.time_interval = 10
        self.time_length = 60

        max_depth = self._knobs.get("max_depth")
        n_estimators = self._knobs.get("n_estimators")
        max_features = self._knobs.get("max_features")
        min_impurity_decrease = self._knobs.get("min_impurity_decrease")
        
        self._regr = RandomForestRegressor(n_estimators = n_estimators, max_depth=max_depth, max_features = max_features, min_impurity_decrease = min_impurity_decrease, random_state=0)

    def train(self, dataset_path, work_dir = None, **kwargs):

        _, feat_train, tgt_train = self.read_discharge_data(dataset_path)

        # Training.
        self._regr.fit(feat_train, tgt_train)

        # Compute R2 on the training set.
        R2_train = self._regr.score(feat_train, tgt_train)
        utils.logger.log('Train accuracy: {}'.format(R2_train))

    def evaluate(self, dataset_path,  work_dir = None, **kwargs):

        _, feat_eval, tgt_eval = self.read_discharge_data(dataset_path)
        R2_eval = self._regr.score(feat_eval, tgt_eval)
        return R2_eval

    def predict(self, queries, work_dir = None):
        predictions = []
        for data_bytes in queries:
            f = open(work_dir + "/query.csv", "wb")
            f.write(data_bytes)
            f.close()
            _, feat, _ = self.read_discharge_data(work_dir + "/query.csv")
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
                if not("Capacity" in data_header):
                    is_query = True
                else:
                    is_query = False
                continue

            if "<Start of Discharging>" in x:
                data_matrix = []
                continue

            elif "<End of Discharging>" in x:
                data_matrix = np.asarray(data_matrix)
                t = data_matrix[:,data_header.index("Time")]
                if is_query == True:
                    target = "Unknown"
                else:
                    target = data_matrix[-1,data_header.index("Capacity")]
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
                        default='data/B0005_query.csv,data/B0005_query.csv',
                        help='Path(s) to query image(s), delimited by commas')
    (args, _) = parser.parse_known_args()

    query_file_list = args.query_path.split(',')
    queries = [open(fname, 'rb').read() for fname in query_file_list]

    test_model_class(model_file_path=__file__,
                     model_class='RFBatteryCapacityEstimator',
                     task='GENERAL_TASK',
                     dependencies={ModelDependency.SCIKIT_LEARN: '0.20.0'},
                     train_dataset_path=args.train_path,
                     val_dataset_path=args.val_path,
                     test_dataset_path=args.test_path,
                     #budget={'TIME_HOURS': 0.001},
                     budget={'MODEL_TRIAL_COUNT': 100, 'TIME_HOURS': 1.0},
                     queries=queries)
