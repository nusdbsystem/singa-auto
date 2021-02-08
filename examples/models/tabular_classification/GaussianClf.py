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

from sklearn.naive_bayes import GaussianNB
# from sklearn.preprocessing import StandardScaler

from singa_auto.model import TabularClfModel, IntegerKnob, FloatKnob, logger
from singa_auto.model.dev import test_model_class
from singa_auto.constants import ModelDependency


class GaussianClf(TabularClfModel):
    '''
    Implements Gaussian Naive Bayes Classifier using heart disease UCI dataset
    '''

    @staticmethod
    def get_knob_config():
        return {
            'var_smoothing': FloatKnob(1e-07, 1e-05),
        }

    def __init__(self, **knobs):
        self._knobs = knobs
        self.__dict__.update(knobs)
        self._clf = self._build_classifier(self._knobs.get("var_smoothing"))


    def train(self, dataset_path, features=None, target=None, **kwargs):
        # Record features & target
        self._features = features
        self._target = target

        # Load CSV file as pandas dataframe
        csv_path = dataset_path
        data = pd.read_csv(csv_path)

        # Extract X & y from dataframe
        (X, y) = self._extract_xy(data)

        # Encode categorical features
        X = self._encoding_categorical_type(X)

        self._clf.fit(X, y)

        # Compute train accuracy
        score = self._clf.score(X, y)
        logger.log('Train accuracy: {}'.format(score))

    def evaluate(self, dataset_path, **kwargs):
        # Load CSV file as pandas dataframe
        csv_path = dataset_path
        data = pd.read_csv(csv_path)

        # Extract X & y from dataframe
        (X, y) = self._extract_xy(data)

        # Encode categorical features
        X = self._encoding_categorical_type(X)

        accuracy = self._clf.score(X, y)
        return accuracy

    def predict(self, queries):
        queries = [pd.DataFrame(query, index=[0]) for query in queries]
        probs = [self._clf.predict_proba(self._features_mapping(query)).tolist()[0] for query in queries]
        return probs

    def destroy(self):
        pass

    def dump_parameters(self):
        params = {}

        # Put model parameters
        clf_bytes = pickle.dumps(self._clf)
        clf_base64 = base64.b64encode(clf_bytes).decode('utf-8')
        params['clf_base64'] = clf_base64
        params['encoding_dict'] = json.dumps(self._encoding_dict)
        params['features'] = json.dumps(self._features)
        params['target'] = self._target

        return params

    def load_parameters(self, params):
        # Load model parameters
        assert 'clf_base64' in params
        clf_base64 = params['clf_base64']
        clf_bytes = base64.b64decode(clf_base64.encode('utf-8'))

        self._clf = pickle.loads(clf_bytes)
        self._encoding_dict = json.loads(params['encoding_dict'])
        self._features = json.loads(params['features'])
        self._target = params['target']

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

    def _encoding_categorical_type(self, cols):
        # Apply label encoding for those categorical columns
        cat_cols = list(
            filter(lambda x: cols[x].dtype == 'object', cols.columns))
        encoded_cols = pd.DataFrame({col: cols[col].astype('category').cat.codes \
            if cols[col].dtype == 'object' else cols[col] for col in cols}, index=cols.index)

        # Recover the missing elements 
        encoded_cols = encoded_cols.replace(to_replace=-1, value=np.nan)

        # Generate the dict that maps categorical features to numerical
        encoding_dict = {col: {cat: n for n, cat in enumerate(cols[col].astype('category'). \
            cat.categories)} for col in cat_cols}
        self._encoding_dict = encoding_dict

        return encoded_cols

    def _features_mapping(self, df):
        # Encode the categorical features with pre saved encoding dict
        cat_cols = list(filter(lambda x: df[x].dtype == 'object', df.columns))
        df_temp = df.copy()
        for col in cat_cols:
            df_temp[col] = df[col].map(self._encoding_dict[col])
        df = df_temp
        return df

    def _build_classifier(self, var_smoothing):
        clf = GaussianNB(var_smoothing=var_smoothing)
        return clf


if __name__ == '__main__':
    test_model_class(model_file_path=__file__,
                     model_class='GaussianClf',
                     task='TABULAR_CLASSIFICATION',
                     dependencies={ModelDependency.SCIKIT_LEARN: '0.20.0'},
                     train_dataset_path='data/heart_train.csv',
                     val_dataset_path='data/heart_val.csv',
                     train_args={
                         'features': [
                             'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                             'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
                             'ca', 'thal'], 
                             'target': 'target'
                     },
                     queries=[{
                         'age': 48,
                         'sex': 1,
                         'cp': 2,
                         'trestbps': 130,
                         'chol': 225,
                         'fbs': 1,
                         'restecg': 1,
                         'thalach': 172,
                         'exang': 1,
                         'oldpeak': 1.7,
                         'slope': 2,
                         'ca': 0,
                         'thal': 3
                     },
                    {
                         'age': 48,
                         'sex': 0,
                         'cp': 2,
                         'trestbps': 130,
                         'chol': 275,
                         'fbs': 0,
                         'restecg': 1,
                         'thalach': 139,
                         'exang': 0,
                         'oldpeak': 0.2,
                         'slope': 2,
                         'ca': 0,
                         'thal': 2
                     },
                    {
                         'age': 58,
                         'sex': 0,
                         'cp': 0,
                         'trestbps': 170,
                         'chol': 225,
                         'fbs': 1,
                         'restecg': 0,
                         'thalach': 146,
                         'exang': 1,
                         'oldpeak': 2.8,
                         'slope': 1,
                         'ca': 2,
                         'thal': 1
                     }])
