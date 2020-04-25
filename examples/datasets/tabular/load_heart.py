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

from examples.datasets.tabular.csv_file import load


# step 1: Install Kaggle API.
# step2: From Rafiki root folder, run: `kaggle datasets download ronitf/heart-disease-uci -p data --unzip`
# to download the `heart.csv` file to `/data` folder.
#)
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def load_heart():
    load(
        dataset_url='data/heart.csv',
        out_train_dataset_path='data/heart_train.csv',
        out_val_dataset_path='data/heart_val.csv'
    )


if __name__ == '__main__':
    load_heart()
