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

from singa_auto.constants import UserType, RequestsParameters, InferenceBudgetOption
from flask import jsonify, Blueprint, g
from singa_auto.utils.auth import auth
from singa_auto.utils.requests_params import param_check
from singa_auto.error_code import UnauthorizedError

inference_bp = Blueprint('inference', __name__)


@inference_bp.route('', methods=['POST'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
@param_check(required_parameters=RequestsParameters.INFERENCE_CREATE)
def create_inference_job(auth, params):
    admin = g.admin

    budget = params['budget'] if 'budget' in params else {}
    budget = {InferenceBudgetOption.GPU_COUNT: 0, **budget}

    if 'app_version' in params:
        app_version = int(params['app_version'])
    else:
        app_version = -1
    
    with admin:
        admin._services_manager.service_app_name = params['app']
        return jsonify(admin.create_inference_job(user_id=auth['user_id'],
                                                  app=params['app'],
                                                  app_version=app_version,
                                                  budget=budget,
                                                  description=params.get('description', None)
                                                  ))


@inference_bp.route('/checkpoint', methods=['POST'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
@param_check(required_parameters=RequestsParameters.INFERENCE_CREATEBY_CHECKOUTPOINT)
def create_inference_job_by_checkpoint(auth, params):
    admin = g.admin

    budget = params['budget'] if 'budget' in params else {}
    budget = {InferenceBudgetOption.GPU_COUNT: 0, **budget}
    with admin:
        admin._services_manager.service_app_name = params['model_name']
        return jsonify(admin.create_inference_job_by_checkpoint(user_id=auth['user_id'],
                                                                budget=budget,
                                                                model_name=params['model_name'],
                                                                description=params.get('description', None)
                                                                ))


@inference_bp.route('', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
@param_check(required_parameters=RequestsParameters.INFERENCE_GETBY_USER)
def get_inference_jobs_by_user(auth, params):
    admin = g.admin

    assert 'user_id' in params

    # Non-admins can only get their own jobs
    if auth['user_type'] in [UserType.APP_DEVELOPER, UserType.MODEL_DEVELOPER] \
            and auth['user_id'] != params['user_id']:
        raise UnauthorizedError()

    with admin:
        return jsonify(admin.get_inference_jobs_by_user(**params))


@inference_bp.route('/<app>', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_inference_jobs_of_app(auth, app):
    admin = g.admin

    with admin:
        return jsonify(admin.get_inference_jobs_of_app(user_id=auth['user_id'], app=app))


@inference_bp.route('/<app>/<app_version>', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_running_inference_job(auth, app, app_version):
    admin = g.admin

    with admin:
        return jsonify(admin.get_running_inference_job(auth['user_id'], app, app_version=int(app_version)))


@inference_bp.route('/<app>/<app_version>/stop', methods=['POST'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def stop_inference_job(auth, app, app_version=-1):
    admin = g.admin

    with admin:
        return jsonify(admin.stop_inference_job(auth['user_id'], app, app_version=int(app_version)))


@inference_bp.route('/app', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
@param_check()
def get_inference_jobs_of_app_safe(auth, params):
    admin = g.admin

    with admin:
        return jsonify(admin.get_inference_jobs_of_app(user_id=auth['user_id'], app=params['app']))


@inference_bp.route('/app/app_version', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
@param_check()
def get_running_inference_job_safe(auth, params):
    admin = g.admin

    with admin:
        return jsonify(admin.get_running_inference_job(auth['user_id'], params['app'], app_version=int(params['app_version'])))


@inference_bp.route('/app/app_version/stop', methods=['POST'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
@param_check()
def stop_inference_job_safe(auth, params):
    admin = g.admin

    with admin:
        return jsonify(admin.stop_inference_job(auth['user_id'], params['app'], app_version=int(params.get('app_version', -1))))