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

from flask import jsonify, Blueprint, g
from singa_auto.utils.auth import auth
from singa_auto.utils.requests_params import param_check

events_bp = Blueprint('events', __name__)


@events_bp.route('/actions/stop_all_jobs', methods=['POST'])
@auth([])
def stop_all_jobs(auth):
    admin = g.admin

    with admin:
        train_jobs = admin.stop_all_train_jobs()
        inference_jobs = admin.stop_all_inference_jobs()
        return jsonify({
            'train_jobs': train_jobs,
            'inference_jobs': inference_jobs
        })

####################################
# Internal Events
####################################

@events_bp.route('/event/<name>', methods=['POST'])
@auth([])
@param_check(required_parameters={})
def handle_event(auth, params, name,):
    admin = g.admin
    with admin:
        return jsonify(admin.handle_event(name, **params))



