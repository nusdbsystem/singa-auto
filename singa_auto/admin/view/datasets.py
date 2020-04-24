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

import tempfile
from singa_auto.constants import UserType, RequestsParameters
from flask import request, jsonify, Blueprint, g
from singa_auto.utils.auth import auth
from singa_auto.utils.requests_params import param_check

dataset_bp = Blueprint('datasets', __name__)


@dataset_bp.route('', methods=['POST'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
@param_check(required_parameters=RequestsParameters.DATASET_POST)
def create_dataset(auth, params):
    admin = g.admin
    print('params', params)
    # Temporarily store incoming dataset data as file
    with tempfile.NamedTemporaryFile() as f:
        if 'dataset' in request.files:
            # Save dataset data in request body
            file_storage = request.files['dataset']
            file_storage.save(f.name)
            file_storage.close()
        else:
            # Download dataset at URL and save it
            assert 'dataset_url' in params
            r = request.get(params['dataset_url'], allow_redirects=True)
            f.write(r.content)
            del params['dataset_url']

        params['data_file_path'] = f.name

        with admin:
            return jsonify(admin.create_dataset(user_id=auth['user_id'], name=params['name'],
                                                task=params['task'],
                                                data_file_path=params['data_file_path']))


@dataset_bp.route('', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
@param_check()
def get_datasets(auth, params):
    admin = g.admin
    if 'task' in params:
        task = params['task']
    else:
        task = None
    with admin:
        return jsonify(admin.get_datasets(auth['user_id'], task=task))

# TODO:New METHOD Delete Dataset
@dataset_bp.route('/datasets/<id>', methods=['DELETE'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
@param_check()
def del_dataset(auth, id, params):
    admin = g.admin
    with admin:
        # would delete dataset
        return jsonify(admin.del_datasets(auth['user_id'], id, **params))


# TODO:New METHOD get Dataset by ID
@dataset_bp.route('/datasets/<id>', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
@param_check()
def get_dataset(auth, id, params):
    admin = g.admin
    with admin:
        return jsonify(admin.get_dataset_by_id(auth['user_id'], id, **params))
