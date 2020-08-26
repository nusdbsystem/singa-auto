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
import traceback

from flask import Flask, g, jsonify
from flask_cors import CORS

from singa_auto.admin import Admin
from singa_auto.admin.view.events import events_bp
from singa_auto.admin.view.inference_job import inference_bp
from singa_auto.admin.view.user import user_bp
from singa_auto.admin.view.datasets import dataset_bp
from singa_auto.admin.view.model import model_bp
from singa_auto.admin.view.train_jobs import trainjob_bp
from singa_auto.admin.view.trials import trial_bp
from singa_auto.error_code import SingaAutoBaseException, SystemInternalError, ResultSuccess

def create_app():
    app = Flask(__name__)

    app.register_blueprint(user_bp)
    app.register_blueprint(dataset_bp, url_prefix='/datasets')
    app.register_blueprint(model_bp, url_prefix='/models')
    app.register_blueprint(trainjob_bp, url_prefix='/train_jobs')
    app.register_blueprint(trial_bp)
    app.register_blueprint(inference_bp, url_prefix='/inference_jobs')
    app.register_blueprint(events_bp)

    CORS(app)

    @app.before_request
    def requests_context():
        print(hasattr(g, 'admin'))
        if not hasattr(g, 'admin'):
            g.admin = Admin()
            pass

    @app.route('/')
    def index():
        return 'SINGA-Auto Admin is up.'

    @app.errorhandler(SingaAutoBaseException)
    def handle_error(error):
        return jsonify(dict(error))

    @app.errorhandler(500)
    def handle_internal_error(error):
        return jsonify(dict(SystemInternalError())), 500

    @app.after_request
    def after_request_handler(resp):

        # if resp.is_json:
        #     data = resp.get_json()
        #     if isinstance(data, dict):
        #         if 'error_code' and 'message' and 'success' in data:
        #             return resp
        #     result = ResultSuccess(data)
        #     return jsonify(dict(result))
        # else:
        #     return resp
            return resp

    return app
