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
import pandas as pd
import numpy as np
import json

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from singa_auto.model import BaseModel, IntegerKnob, FloatKnob, CategoricalKnob, logger
from singa_auto.model.dev import test_model_class
from singa_auto.constants import ModelDependency


class TreeReg(BaseModel):
    '''
    Implements a Decision Tree Regressor for regression task using bodyfat dataset.
    '''

    @staticmethod
    def get_knob_config():
        return {
            'criterion': CategoricalKnob(['mse', 'mae']),
            'splitter': CategoricalKnob(['best', 'random']),
            'min_samples_split': IntegerKnob(2, 5),
            'max_features': CategoricalKnob(['auto', 'sqrt']),
            'random_state': IntegerKnob(1, 123),
            'min_impurity_decrease': FloatKnob(0.0, 0.2),
            'min_impurity_split': FloatKnob(1e-07, 1e-03)
        }

    def __init__(self, **knobs):
        self.__dict__.update(knobs)
        self._regressor = self._build_regressor(self.criterion, self.splitter,
                                                self.min_samples_split,
                                                self.max_features,
                                                self.random_state,
                                                self.min_impurity_decrease,
                                                self.min_impurity_split)

    def train(self, dataset_path, features=None, target=None, **kwargs):
        # Record features & target
        self._features = features
        self._target = target

        # Load CSV file as pandas dataframe
        csv_path = dataset_path
        data = pd.read_csv(csv_path)

        # Extract X & y from dataframe
        (X, y) = self._extract_xy(data)

        X = self.prepare_X(X)

        self._regressor.fit(X, y)

        # Compute train root mean square error
        preds = self._regressor.predict(X)

        rmse = np.sqrt(mean_squared_error(y, preds))
        logger.log('Train RMSE: {}'.format(rmse))

    def evaluate(self, dataset_path):
        # Load CSV file as pandas dataframe
        csv_path = dataset_path
        data = pd.read_csv(csv_path)

        # Extract X & y from dataframe
        (X, y) = self._extract_xy(data)

        X = self.prepare_X(X)

        preds = self._regressor.predict(X)

        rmse = np.sqrt(mean_squared_error(y, preds))
        return 1 / rmse

    def predict(self, queries):
        queries = [pd.DataFrame(query, index=[0]) for query in queries]
        data = self.prepare_X(queries)
        result = self._regressor.predict(data)
        return result.tolist()[0]

    def destroy(self):
        pass

    def dump_parameters(self):
        params = {}

        # Put model parameters
        regressor_bytes = pickle.dumps(self._regressor)
        regressor_base64 = base64.b64encode(regressor_bytes).decode('utf-8')
        params['regressor_base64'] = regressor_base64
        params['features'] = json.dumps(self._features)
        if self._target:
            params['target'] = self._target

        return params

    def load_parameters(self, params):
        # Load model parameters
        assert 'regressor_base64' in params
        regressor_base64 = params['regressor_base64']
        regressor_bytes = base64.b64decode(regressor_base64.encode('utf-8'))

        self._regressor = pickle.loads(regressor_bytes)
        self._features = json.loads(params['features'])

        if "target" in params:
            self._target = params['target']
        else:
            self._target = None

    def _extract_xy(self, data):
        features = self._features
        target = self._target

        if features is None:
            X = data.iloc[:, :-1]
        else:
            X = data[features]

        if target is None:
            y = data.iloc[:, -1]
        else:
            y = data[target]

        return (X, y)

    def median_dataset(self, df):
        #replace zero values by median so that 0 will not affect median.
        for col in df.columns:
            df[col].replace(0, np.nan, inplace=True)
            df[col].fillna(df[col].median(), inplace=True)
        return df

    def prepare_X(self, df):
        data = self.median_dataset(df)
        X = StandardScaler().fit_transform(df)
        return X
    def _build_regressor(self, criterion, splitter, min_samples_split,
                         max_features, random_state, min_impurity_decrease,
                         min_impurity_split):
        regressor = DecisionTreeRegressor(
            criterion=criterion,
            splitter=splitter,
            min_samples_split=min_samples_split,
            max_features=max_features,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
        )
        return regressor


if __name__ == '__main__':
    test_model_class(model_file_path=__file__,
                     model_class='TreeReg',
                     task='TABULAR_REGRESSION',
                     dependencies={ModelDependency.SCIKIT_LEARN: '0.20.0'},
                     train_dataset_path='data/bodyfat_train.csv',
                     val_dataset_path='data/bodyfat_val.csv',
                     train_args={
                         'features': [
                             'density', 'age', 'weight', 'height', 'neck',
                             'chest', 'abdomen', 'hip', 'thigh', 'knee',
                             'ankle', 'biceps', 'forearm', 'wrist'
                         ],
                         'target': 'bodyfat'
                     },
                     queries=[{
                         'density': 1.0207,
                         'age': 65,
                         'weight': 224.5,
                         'height': 68.25,
                         'neck': 38.8,
                         'chest': 119.6,
                         'abdomen': 118.0,
                         'hip': 114.3,
                         'thigh': 61.3,
                         'knee': 42.1,
                         'ankle': 23.4,
                         'biceps': 34.9,
                         'forearm': 30.1,
                         'wrist': 19.4
                     }])
