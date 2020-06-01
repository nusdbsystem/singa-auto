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

from datetime import datetime
from singa_auto.constants import UserType, RequestsParameters
from flask import Blueprint, jsonify, g
from singa_auto.utils.auth import generate_token, auth
from singa_auto.utils.requests_params import param_check
from singa_auto.error_code import UnauthorizedError

user_bp = Blueprint('user', __name__)


@user_bp.route('/users', methods=['POST'])
@auth([UserType.ADMIN])
@param_check(required_parameters=RequestsParameters.USER_CREATE)
def create_user(auth, params):
    admin = g.admin

    # Only superadmins can create admins
    if auth['user_type'] != UserType.SUPERADMIN and \
            params['user_type'] in [UserType.ADMIN, UserType.SUPERADMIN]:
        raise UnauthorizedError()

    with admin:
        return jsonify(admin.create_user(email=params['email'],
                                         password=params['password'],
                                         user_type=params['user_type'],))


@user_bp.route('/users', methods=['GET'])
@auth([UserType.ADMIN])
def get_users(auth):
    admin = g.admin

    with admin:
        return jsonify(admin.get_users())


@user_bp.route('/users', methods=['DELETE'])
@auth([UserType.ADMIN])
@param_check(required_parameters=RequestsParameters.USER_BAN)
def ban_user(auth, params):
    admin = g.admin

    with admin:
        user = admin.get_user_by_email(params['email'])

        if user is not None:
            # Only superadmins can ban admins
            if auth['user_type'] != UserType.SUPERADMIN and \
                    user['user_type'] in [UserType.ADMIN, UserType.SUPERADMIN]:
                raise UnauthorizedError()

            # Cannot ban yourself
            if auth['user_id'] == user['id']:
                raise UnauthorizedError()

        return jsonify(admin.ban_user(email=params['email']))


@user_bp.route('/tokens', methods=['POST'])
@param_check(required_parameters=RequestsParameters.TOKEN)
def generate_user_token(params):
    admin = g.admin

    # Error will be thrown here if credentials are invalid
    with admin:
        user = admin.authenticate_user(**params)

    # User cannot be banned
    if user.get('banned_date') is not None and datetime.now() > user.get('banned_date'):
        raise UnauthorizedError('User is banned')

    token = generate_token(user)

    return jsonify({
        'user_id': user['id'],
        'user_type': user['user_type'],
        'token': token
    })
