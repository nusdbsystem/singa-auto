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

import pytest
import os
import uuid
import random
import numpy as np

from rafiki.constants import UserType, ModelAccessRight
from rafiki.client import Client
from rafiki.config import SUPERADMIN_EMAIL


superadmin_email = SUPERADMIN_EMAIL
superadmin_password = os.environ['SUPERADMIN_PASSWORD']

# Details for mocks
DATASET_FILE_PATH = 'test/data/dataset.csv'
MODEL_FILE_PATH = 'test/data/Model.py'
MODEL_CLASS = 'Model'

####################################
# General
####################################

@pytest.fixture(scope='session', autouse=True)
def global_setup():
    random.seed(0)
    np.random.seed(0)

def gen():
    return str(uuid.uuid4())

def gen_email():
    return f'{uuid.uuid4()}@rafiki'

####################################
# Users
####################################

@pytest.fixture(scope='module')
def superadmin():
    client = Client()
    client.login(superadmin_email, superadmin_password)
    return client

def make_admin(**kwargs):
    return make_user(UserType.ADMIN, **kwargs)

def make_app_dev(**kwargs):
    return make_user(UserType.APP_DEVELOPER, **kwargs)

def make_model_dev(**kwargs):
    return make_user(UserType.MODEL_DEVELOPER, **kwargs)

# Make a client logged in as new user with a specific user type
def make_user(user_type, email=None, password=None):
    email = email or gen_email()
    password = password or gen()
    client = Client()
    client.login(superadmin_email, superadmin_password)
    client.create_user(email, password, user_type)
    client.login(email, password)
    return client
    

####################################
# Datasets
####################################

def make_dataset(client: Client, task=None):
    name = gen()
    task = task or gen()
    file_path = DATASET_FILE_PATH
    dataset = client.create_dataset(name, task, file_path)
    dataset_id = dataset['id']
    return dataset_id

####################################
# Models
####################################

def make_private_model(**kwargs):
    return make_model(access_right=ModelAccessRight.PRIVATE, **kwargs)

def make_model(task=None, access_right=ModelAccessRight.PUBLIC):
    model_dev = make_model_dev()
    task = task or gen()
    name = gen()
    model_file_path = MODEL_FILE_PATH
    model_class = MODEL_CLASS
    dependencies = {}
    model = model_dev.create_model(name, task, model_file_path, model_class, dependencies, access_right=access_right)
    model_id = model['id']
    return model_id