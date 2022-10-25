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


import math
import numpy as np
import pandas as pd
from numpy import array
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
#import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error


class RFWindPowerPredictor(BaseModel):

    '''
    This class defines a RF (random forest) wind power predictor.
    '''

    @staticmethod
    def get_knob_config():
        return {
            'n_steps': IntegerKnob(16, 64),
            'max_depth': IntegerKnob(8, 16),
            'n_estimators': IntegerKnob(8, 32),
            'max_features': CategoricalKnob(['auto', 'sqrt', 'log2']),
            'min_impurity_decrease': FloatKnob(0.0, 0.05)
        }

    def __init__(self, **knobs):
        self._knobs = knobs
        self.__dict__.update(knobs)

        # The length of predicted sequence.
        self.prediction_length = 10

        self.n_steps = self._knobs.get("n_steps") # self.n_steps = 32 works well.

        max_depth = self._knobs.get("max_depth")
        n_estimators = self._knobs.get("n_estimators")
        max_features = self._knobs.get("max_features")
        min_impurity_decrease = self._knobs.get("min_impurity_decrease")

        self.model = RandomForestRegressor(n_estimators = n_estimators, max_depth=max_depth, max_features = max_features, min_impurity_decrease = min_impurity_decrease, random_state=0)

    def split_sequence(self, sequence, n_steps):
        """
        Time Series:
        Split the sequences to x with n_steps for input and y for output
        """
        X, y = list(), list()
        for i in range(len(sequence)):
                # find the end of this pattern
                end_ix = i + n_steps
                # check if we are beyond the sequence
                if end_ix > len(sequence)-1:
                        break
                # gather input and output parts of the pattern
                seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
                X.append(seq_x)
                y.append(seq_y)
        return array(X), array(y)

    def load_dataset(self, dataset_path, n_steps):
        """
        Load train and validation Dataset and do corresponding Data Preprocessing
        """
        df = pd.read_csv(dataset_path)

        df_header = list(df.columns.values)
        wind_speed_idx = df_header.index("Wind Speed (km/h)")

        #######################################
        # Data Preprocessing 
        # process into time series model.
        #######################################

        # Wind Speed
        data_seq = np.array(df.iloc[:,wind_speed_idx:wind_speed_idx+1])

        """
        # Normalization
        if norm_mdl == None:
            norm = StandardScaler()
            norm.fit(data_seq)
        else:
            norm = norm_mdl

        data_seq = norm.transform(data_seq)
        """

        data_seq = data_seq.reshape(data_seq.shape[0])
        

        # Process data to n_steps time series model
        ## split into samples
        x, y = self.split_sequence(data_seq, n_steps)
        
        return x, y

    """
    def min_max_normalisation(self, feat_data, speed_min, speed_max):
        s = speed_max - speed_min
        if s != 0:
            y = (np.array(feat_data) - speed_min) / s
        else:
            y = np.array(feat_data)
        return y

    def anti_min_max_normalisation(self, prediction, speed_min, speed_max):
        s = speed_max - speed_min
        if s != 0:
            y = np.array(prediction) * s + speed_min
        else:
            y = np.array(prediction)
        return y
    """

    def train(self, dataset_path, work_dir = None, **kwargs):

        # Load Training Dataset
        x_train, y_train = self.load_dataset(dataset_path = dataset_path, n_steps = self.n_steps)

        # self.speed_min = np.min(x_train)
        # self.speed_max = np.max(x_train)

        # x_train = self.min_max_normalisation(x_train, self.speed_min, self.speed_max)
        # y_train = self.min_max_normalisation(y_train, self.speed_min, self.speed_max)

        # Train a RF model
        self.model.fit(x_train, y_train)

        # Compute R2 on the training set.
        R2_train = self.model.score(x_train, y_train)
        utils.logger.log('Train accuracy: {}'.format(R2_train))



    def evaluate(self, dataset_path,  work_dir = None, **kwargs):

        # Load Dataset
        x_eval, y_eval = self.load_dataset(dataset_path = dataset_path, n_steps = self.n_steps)
       
        # x_eval = self.min_max_normalisation(x_eval, self.speed_min, self.speed_max)
        # y_eval = self.min_max_normalisation(y_eval, self.speed_min, self.speed_max)

        R2_eval = self.model.score(x_eval, y_eval)
        return R2_eval

    def predict(self, queries, work_dir = None):

        predictions = []
        for data_bytes in queries:
            query_path = work_dir + "/query.csv"
            f = open(query_path, "wb")
            f.write(data_bytes)
            f.close()
            df = pd.read_csv(query_path)
            df_header = list(df.columns.values)
            wind_speed_idx = df_header.index("Wind Speed (km/h)")
            data_seq = np.array(df.iloc[:,wind_speed_idx:wind_speed_idx+1])

            data_seq = data_seq.reshape(data_seq.shape[0])
            feat = data_seq[data_seq.shape[0]-self.n_steps : data_seq.shape[0]]
            # feat = self.min_max_normalisation(feat, self.speed_min, self.speed_max)
            feat = feat.tolist()

            prediction = []
            for i in range(self.prediction_length):
                s = self.model.predict([feat])
                prediction.append(s[0])
                feat = feat[1:len(feat)] + [s[0]]
            # prediction = self.anti_min_max_normalisation(prediction, self.speed_min, self.speed_max).tolist() 
            predictions.append(str(prediction))
        return predictions

    def dump_parameters(self):
        params = pickle.dumps(self.__dict__)
        return params

    def load_parameters(self, params):
        self.__dict__ = pickle.loads(params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path',
                        type=str,
                        default='data/wind_train.csv',
                        help='Path to train dataset')
    parser.add_argument('--val_path',
                        type=str,
                        default='data/wind_val.csv',
                        help='Path to validation dataset')
    parser.add_argument('--test_path',
                        type=str,
                        default='data/wind_test.csv',
                        help='Path to test dataset')
    parser.add_argument('--query_path',
                        type=str,
                        default='data/wind_query.csv,data/wind_query.csv',
                        help='Path(s) to query files')

    (args, _) = parser.parse_known_args()

    query_file_list = args.query_path.split(',')
    queries = [open(fname, 'rb').read() for fname in query_file_list]

    test_model_class(model_file_path=__file__,
                     model_class='RFWindPowerPredictor',
                     task='GENERAL_TASK',
                     dependencies={ModelDependency.SCIKIT_LEARN: '0.20.0'},
                     train_dataset_path=args.train_path,
                     val_dataset_path=args.val_path,
                     test_dataset_path=args.test_path,
                     budget={'MODEL_TRIAL_COUNT': 10, 'TIME_HOURS': 1.0},
                     queries=queries)

