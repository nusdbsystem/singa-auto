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

from typing import Dict, Any


class BudgetOption:
    GPU_COUNT = 'GPU_COUNT'
    TIME_HOURS = 'TIME_HOURS'
    MODEL_TRIAL_COUNT = 'MODEL_TRIAL_COUNT'
    DIST_WORKERS = 'DIST_WORKERS'


Budget = Dict[BudgetOption, Any]


class InferenceBudgetOption:
    GPU_COUNT = 'GPU_COUNT'


InferenceBudget = Dict[InferenceBudgetOption, Any]

ModelDependencies = Dict[str, str]


class ModelAccessRight:
    PUBLIC = 'PUBLIC'
    PRIVATE = 'PRIVATE'


class InferenceJobStatus:
    STARTED = 'STARTED'
    RUNNING = 'RUNNING'
    ERRORED = 'ERRORED'
    STOPPED = 'STOPPED'


class Tasks:
    IMAGE_DETECTION = 'IMAGE_DETECTION'
    IMAGE_SEGMENTATION = 'IMAGE_SEGMENTATION'
    IMAGE_CLASSIFICATION = 'IMAGE_CLASSIFICATION'
    QUESTION_ANSWERING = 'QUESTION_ANSWERING'
    TEXT_CLASSIFICATION = 'TEXT_CLASSIFICATION'


class TrainJobStatus:
    STARTED = 'STARTED'
    RUNNING = 'RUNNING'
    STOPPED = 'STOPPED'
    ERRORED = 'ERRORED'


class TrialStatus:
    PENDING = 'PENDING'
    RUNNING = 'RUNNING'
    ERRORED = 'ERRORED'
    COMPLETED = 'COMPLETED'


class UserType:
    SUPERADMIN = 'SUPERADMIN'
    ADMIN = 'ADMIN'
    MODEL_DEVELOPER = 'MODEL_DEVELOPER'
    APP_DEVELOPER = 'APP_DEVELOPER'


class ServiceType:
    TRAIN = 'TRAIN'
    ADVISOR = 'ADVISOR'
    PREDICT = 'PREDICT'
    INFERENCE = 'INFERENCE'


class ServiceStatus:
    STARTED = 'STARTED'
    DEPLOYING = 'DEPLOYING'
    RUNNING = 'RUNNING'
    ERRORED = 'ERRORED'
    STOPPED = 'STOPPED'


class ModelDependency:
    TENSORFLOW = 'tensorflow'
    KERAS = 'Keras'
    SCIKIT_LEARN = 'scikit-learn'
    TORCH = 'torch'
    TORCHVISION = 'torchvision'
    SINGA = 'singa'
    XGBOOST = 'xgboost'
    DS_CTCDECODER = 'ds-ctcdecoder'
    NLTK = 'nltk'
    SKLEARN_CRFSUITE = 'sklearn-crfsuite'
    ONNX = 'onnx'


class NodeLabes:
    NodeName = "kubernetes.io/hostname"
    FreeMemorySum = 'samonitor/FreeMemorySum'
    Gpu = 'samonitor/Gpu'
    GpuSummary = 'samonitor/GpuSummary'
    Health = 'samonitor/Health'
    MaxFreeMemory = 'samonitor/MaxFreeMemory'
    Number = 'samonitor/Number'


class RequestsParameters:

    # True: must be provided,
    # False: not necessary

    ####################################
    # User
    ####################################

    USER_CREATE = {'json': {'email': True, 'password': True, 'user_type': True}}

    LOGIN = {'json': {'email': True, 'password': True}}

    USER_BAN = {'json': {'email': True}}

    TOKEN = {'json': {'email': True, 'password': True}}

    ####################################
    # Datasets
    ####################################

    DATASET_POST = {
        'files': {
            'dataset': False
        },
        'data': {
            'name': True,
            'task': True,
            'dataset_url': False
        }
    }

    ####################################
    # Models
    ####################################

    MODEL_CREATE = {
        'files': {
            "model_file_bytes": True,
            "model_pretrained_params_id": False
        },
        'data': {
            'name': True,
            'task': True,
            'dependencies': False,
            'docker_image': False,
            'model_class': True,
            'access_right': False,
        }
    }

    ####################################
    # Train Jobs
    ####################################

    TRAIN_CREATE = {
        'json': {
            'app': True,
            'task': True,
            'train_dataset_id': True,
            'val_dataset_id': True,
            'budget': False,
            'model_ids': False,
            'train_args': False
        }
    }

    TRAIN_GETBY_USER = {'params': {'user_id': True}}

    ####################################
    # Trials
    ####################################

    TRIAL_GET_BEST = {'params': {'type': False, 'max_count': False}}

    ####################################
    # Inference Jobs
    ####################################

    INFERENCE_CREATE = {
        'json': {
            'app': True,
            'app_version': False,
            'budget': False
        }
    }

    INFERENCE_CREATEBY_CHECKOUTPOINT = {
        'json': {
            'model_name': True,
            'budget': False
        }
    }

    INFERENCE_GETBY_USER = {'params': {'user_id': True}}


class ModelType:
    PYTHON_FILE = 'py'
    ZIP_FILE = 'zip'
