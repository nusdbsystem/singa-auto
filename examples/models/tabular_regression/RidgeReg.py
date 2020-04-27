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

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from singa_auto.model import BaseModel, IntegerKnob, FloatKnob, CategoricalKnob, logger
from singa_auto.model.dev import test_model_class
from singa_auto.constants import ModelDependency

class RidgeReg(BaseModel):
    '''
    Implements a Linear Ridge Regressor for regression task using boston housing price dataset.
    '''
    @staticmethod
    def get_knob_config():
        return {
            'alpha': FloatKnob(0.001, 0.01),
            'normalize': CategoricalKnob([True, False]),
            'copy_X': CategoricalKnob([True, False]),
            'tol': FloatKnob(1e-05, 1e-04),
            'solver': CategoricalKnob(['svd', 'sag']),
            'random_state': IntegerKnob(1, 123)
        }

    def __init__(self, **knobs):
        self.__dict__.update(knobs)
        self._regressor = self._build_regressor(self.alpha, self.normalize, self.copy_X, self.tol, self.solver, self.random_state)


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
            X = data.iloc[:,:-1]
        else:
            X = data[features]

        if target is None:
            y = data.iloc[:,-1]
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
        X = PolynomialFeatures(interaction_only=True).fit_transform(df)
        return X


    def _build_regressor(self, alpha, normalize, copy_X, tol, solver, random_state):
        regressor = Ridge(
            alpha=alpha,
            normalize = normalize,
            copy_X = copy_X,
            tol = tol,
            solver = solver,
            random_state = random_state,
        )
        return regressor

if __name__ == '__main__':
    test_model_class(
        model_file_path=__file__,
        model_class='RidgeReg',
        task='TABULAR_REGRESSION',
        dependencies={
            ModelDependency.SCIKIT_LEARN: '0.20.0'
        },
        train_dataset_path='data/boston_train.csv',
        val_dataset_path='data/boston_val.csv',
        train_args={
            'features': ['CRIM',
                        'ZN',
                        'INDUS',
                        'CHAS',
                        'NOX',
                        'RM',
                        'AGE',
                        'DIS',
                        'RAD',
                        'TAX',
                        'PTRATIO',
                        'B',
                        'LSTAT'],
            'target': 'MEDV'
        },
        queries=[
             {'CRIM': 60.1,
            'ZN': 0.001,
            'INDUS':18.1,
            'CHAS':0,
            'NOX':597,
            'RM':6.23,
            'AGE': 50.0,
            'DIS':1.222,
            'RAD':23,
            'TAX':700,
            'PTRATIO':20.1,
            'B':1.54,
            'LSTAT':11.09}
        ]
    )
