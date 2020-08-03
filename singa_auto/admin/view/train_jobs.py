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

from singa_auto.constants import UserType, RequestsParameters, BudgetOption
from flask import jsonify, Blueprint, g
from singa_auto.utils.auth import auth
from singa_auto.utils.requests_params import param_check
from singa_auto.error_code import UnauthorizedError

trainjob_bp = Blueprint('trainjob', __name__)


@trainjob_bp.route('', methods=['POST'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
@param_check(required_parameters=RequestsParameters.TRAIN_CREATE)
def create_train_job(auth, params):
    admin = g.admin

    budget = params['budget'] if 'budget' in params else {}
    budget = {BudgetOption.TIME_HOURS: 0.1, BudgetOption.GPU_COUNT: 0, **budget}

    feed_params = {}
    feed_params['user_id'] = auth['user_id']
    feed_params['app'] = params['app']
    feed_params['task'] = params['task']
    feed_params['train_dataset_id'] = params['train_dataset_id']
    feed_params['val_dataset_id'] = params['val_dataset_id']
    feed_params['budget'] = budget

    if "annotation_dataset_id" in params:
        feed_params['annotation_dataset_id'] = params['annotation_dataset_id']
    if 'model_ids' in params:
        feed_params['model_ids'] = params['model_ids']
    if 'train_args' in params:
        feed_params['train_args'] = params['train_args']

    with admin:
        admin._services_manager.service_app_name = params['app']
        # Ensure that datasets are owned by current user
        dataset_attrs = ['train_dataset_id', 'val_dataset_id']
        for attr in dataset_attrs:
            if attr in params:
                dataset_id = params[attr]
                dataset = admin.get_dataset(dataset_id)
                if auth['user_id'] != dataset['owner_id']:
                    raise UnauthorizedError('You have no access to dataset of ID "{}"'.format(dataset_id))

        return jsonify(admin.create_train_job(**feed_params))


@trainjob_bp.route('', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
@param_check()
def get_train_jobs_by_user(auth, params):
    admin = g.admin

    if 'user_id' in params:
        # Non-admins can only get their own jobs
        if auth['user_type'] in [UserType.APP_DEVELOPER, UserType.MODEL_DEVELOPER] \
                and auth['user_id'] != params['user_id']:
            raise UnauthorizedError()

    user_id = auth['user_id']

    with admin:
        return jsonify(admin.get_train_jobs_by_user(user_id=user_id))


@trainjob_bp.route('/<app>', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_train_jobs_by_app(auth, app):
    admin = g.admin

    with admin:
        return jsonify(admin.get_train_jobs_by_app(auth['user_id'], app))


@trainjob_bp.route('/<app>/<app_version>', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_train_job(auth, app, app_version):
    admin = g.admin

    with admin:
        return jsonify(admin.get_train_job(auth['user_id'], app, app_version=int(app_version)))


@trainjob_bp.route('/<app>/<app_version>/stop', methods=['POST'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def stop_train_job(auth, app, app_version):
    admin = g.admin

    with admin:
        return jsonify(admin.stop_train_job(auth['user_id'], app, app_version=int(app_version)))


@trainjob_bp.route('/app', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
@param_check()
def get_train_jobs_by_app_safe(auth, params):
    admin = g.admin

    with admin:
        return jsonify(admin.get_train_jobs_by_app(auth['user_id'], params['app']))


@trainjob_bp.route('/app/app_version', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
@param_check()
def get_train_job_safe(auth, params):
    admin = g.admin

    with admin:
        return jsonify(admin.get_train_job(auth['user_id'], params['app'], app_version=int(params['app_version'])))


@trainjob_bp.route('/app/app_version/stop', methods=['POST'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
@param_check()
def stop_train_job_safe(auth, params):
    admin = g.admin

    with admin:
        return jsonify(admin.stop_train_job(auth['user_id'], params['app'], app_version=int(params['app_version'])))