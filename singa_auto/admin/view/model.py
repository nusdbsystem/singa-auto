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

import base64
import json
import os
import shutil
import tempfile
import traceback
import uuid

from singa_auto.constants import UserType, RequestsParameters
from flask import jsonify, Blueprint, make_response, g

from singa_auto.meta_store.meta_store import DuplicateModelNameError
from singa_auto.param_store import FileParamStore
from singa_auto.utils.auth import UnauthorizedError, auth
from singa_auto.utils.requests_params import param_check

model_bp = Blueprint('model', __name__)


@model_bp.route('', methods=['POST'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER])
@param_check(required_parameters=RequestsParameters.MODEL_CREATE)
def create_model(auth, params):
    admin = g.admin

    feed_params = dict()
    feed_params['user_id'] = auth['user_id']
    feed_params['name'] = params['name']
    feed_params['task'] = params['task']
    feed_params['model_file_bytes'] = params['model_file_bytes'].read()
    feed_params['model_class'] = params['model_class']

    if 'dependencies' in params and isinstance(params['dependencies'], str):
        feed_params['dependencies'] = json.loads(params['dependencies'])

    if 'docker_image' in params:
        feed_params['docker_image'] = params['docker_image']

    if 'access_right' in params:
        feed_params['access_right'] = params['access_right']

    if 'checkpoint_id' in params and params['checkpoint_id'] is not None:

        # if the checkpoint is not .model file, serialize it first
        if params['checkpoint_id'].filename.split(".")[-1] != 'model':
            h5_model_bytes = params['checkpoint_id'].read()
            checkpoint_id = FileParamStore().save({'h5_model_base64': base64.b64encode(h5_model_bytes).decode('utf-8')})
            feed_params['checkpoint_id'] = checkpoint_id
        # if the model is trained with singa_auto, copy it to params files
        else:
            with tempfile.NamedTemporaryFile() as f:
                file_storage = params['checkpoint_id']
                file_storage.save(f.name)
                file_storage.close()
                checkpoint_id = '{}.model'.format(uuid.uuid4())
                dest_file_path = os.path.join(os.path.join(os.environ['WORKDIR_PATH'],
                                                           os.environ['PARAMS_DIR_PATH']),
                                              checkpoint_id)
                shutil.copyfile(f.name, dest_file_path)
            feed_params['checkpoint_id'] = checkpoint_id
    with admin:
        return jsonify(admin.create_model(**feed_params))


@model_bp.route('/available', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
@param_check()
def get_available_models(auth, params):
    admin = g.admin
    if 'task' in params:
        task = params['task']
    else:
        task = None

    with admin:
        return jsonify(admin.get_available_models(user_id=auth['user_id'], task=task))


@model_bp.route('/<model_id>', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_model(auth, model_id):
    admin = g.admin

    with admin:
        # Non-admins cannot access others' models
        if auth['user_type'] in [UserType.APP_DEVELOPER, UserType.MODEL_DEVELOPER]:
            model = admin.get_model(model_id)
            if auth['user_id'] != model['user_id']:
                raise UnauthorizedError()

        return jsonify(admin.get_model(model_id))


@model_bp.route('/<model_id>', methods=['DELETE'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER])
def delete_model(auth, model_id):
    admin = g.admin

    with admin:
        # Non-admins cannot delete others' models
        if auth['user_type'] in [UserType.MODEL_DEVELOPER]:
            model = admin.get_model(model_id)
            if auth['user_id'] != model['user_id']:
                raise UnauthorizedError()

        return jsonify(admin.delete_model(model_id))


@model_bp.route('/<model_id>/model_file', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER])
def download_model_file(auth, model_id):
    admin = g.admin

    with admin:
        # Non-admins cannot access others' models
        if auth['user_type'] in [UserType.MODEL_DEVELOPER]:
            model = admin.get_model(model_id)
            if auth['user_id'] != model['user_id']:
                raise UnauthorizedError()

        model_file = admin.get_model_file(model_id)

    res = make_response(model_file)
    res.headers.set('Content-Type', 'application/octet-stream')
    return res


# TODO:New METHOD get model of specific task
@model_bp.route('/<task>/available', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_available_models_of_taks(auth, task):
    admin = g.admin
    with admin:
        return jsonify(admin.get_available_models(auth['user_id'], task))


@model_bp.route('/recommended', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
@param_check()
def get_recommend_models(auth, params):
    admin = g.admin
    with admin:
            return jsonify(admin.get_recommend_models(auth['user_id'],
                                                      dataset_id=params['dataset_id']))


@model_bp.errorhandler(Exception)
def handle_error(error):
    traceback.print_exc()
    if type(error) == DuplicateModelNameError:
        return jsonify({'ErrorMsg': 'DuplicateModelNameError'}), 400
