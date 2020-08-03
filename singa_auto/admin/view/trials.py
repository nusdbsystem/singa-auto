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
from singa_auto.constants import UserType, RequestsParameters
from flask import jsonify, Blueprint, make_response, g
from singa_auto.utils.auth import auth
from singa_auto.utils.requests_params import param_check

trial_bp = Blueprint('trial', __name__)


@trial_bp.route('/trials/<trial_id>/logs', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_trial_logs(auth, trial_id):
    admin = g.admin

    with admin:
        return jsonify(admin.get_trial_logs(trial_id))


@trial_bp.route('/trials/<trial_id>/parameters', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_trial_parameters(auth, trial_id):
    admin = g.admin

    with admin:
        trial_params = admin.get_trial_parameters(trial_id)

    trial_params = pickle.dumps(trial_params)  # Pickle to convert to bytes
    res = make_response(trial_params)
    res.headers.set('Content-Type', 'application/octet-stream')
    return res


@trial_bp.route('/trials/<trial_id>', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
def get_trial(auth, trial_id):
    admin = g.admin

    with admin:
        return jsonify(admin.get_trial(trial_id))


@trial_bp.route('/train_jobs/<app>/<app_version>/trials', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
@param_check(required_parameters=RequestsParameters.TRIAL_GET_BEST)
def get_trials_of_train_job(auth, app, app_version, params):

    admin = g.admin

    # max_count = int(params['max_count']) if 'max_count' in params else 2

    with admin:
        if "type" in params and params.get('type') == 'best':
            # Return best trials by train job
            return jsonify(
                admin.get_best_trials_of_train_job(user_id=auth['user_id'], app=app, app_version=int(app_version),
                                                   ))
        else:
            return jsonify(admin.get_trials_of_train_job(user_id=auth['user_id'], app=app, app_version=int(app_version),
                                                         ))


@trial_bp.route('/train_jobs/app/app_version/trials', methods=['GET'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER, UserType.APP_DEVELOPER])
@param_check(required_parameters=RequestsParameters.TRIAL_GET_BEST)
def get_trials_of_train_job_safe(auth, params):

    admin = g.admin

    # max_count = int(params['max_count']) if 'max_count' in params else 2

    with admin:
        if "type" in params and params.get('type') == 'best':
            # Return best trials by train job
            return jsonify(
                admin.get_best_trials_of_train_job(user_id=auth['user_id'], app=params['app'], app_version=int(params['app_version']),
                                                   ))
        else:
            return jsonify(admin.get_trials_of_train_job(user_id=auth['user_id'], app=params['app'], app_version=int(params['app_version']),
                                                         ))