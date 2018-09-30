from flask import Flask, request, jsonify
import os
import traceback

from rafiki.constants import UserType
from rafiki.utils.auth import generate_token, decode_token, UnauthorizedException, auth
from rafiki.utils.parse import get_request_params

from .admin import Admin

admin = Admin()

app = Flask(__name__)

@app.route('/')
def index():
    return 'Rafiki Admin is up.'

####################################
# Users
####################################

@app.route('/users', methods=['POST'])
@auth([UserType.ADMIN])
def create_user(auth):
    params = get_request_params()
    with admin:
        return jsonify(admin.create_user(**params))

@app.route('/tokens', methods=['POST'])
def generate_user_token():
    params = get_request_params()

    # Error will be thrown here if credentials are invalid
    with admin:
        user = admin.authenticate_user(**params)

    auth = {
        'user_id': user['id'],
        'user_type': user['user_type']
    }
    
    token = generate_token(auth)

    return jsonify({
        'user_id': user['id'],
        'user_type': user['user_type'],
        'token': token
    })

####################################
# Train Jobs
####################################

@app.route('/train_jobs', methods=['POST'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def create_train_job(auth):
    params = get_request_params()
    with admin:
        return jsonify(admin.create_train_job(auth['user_id'], **params))

@app.route('/train_jobs/<app>', methods=['GET'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def get_train_jobs_of_app(auth, app):
    params = get_request_params()
    with admin:
        return jsonify(admin.get_train_jobs_of_app(app, **params))

@app.route('/train_jobs/<app>/<app_version>', methods=['GET'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def get_train_job(auth, app, app_version):
    params = get_request_params()
    with admin:
        return jsonify(admin.get_train_job(app, app_version=int(app_version), **params))

@app.route('/train_jobs/<app>/<app_version>/stop', methods=['POST'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def stop_train_job(auth, app, app_version):
    params = get_request_params()
    with admin:
        return jsonify(admin.stop_train_job(app, app_version=int(app_version), **params))

@app.route('/train_jobs/<app>/<app_version>/trials', methods=['GET'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def get_trials_of_train_job(auth, app, app_version):
    params = get_request_params()

    # Return best trials by train job
    if params.get('type') == 'best':
        del params['type']

        if 'max_count' in params:
            params['max_count'] = int(params['max_count'])

        with admin:
            return jsonify(admin.get_best_trials_of_train_job(
                app, 
                app_version=int(app_version),
                **params
            ))
    
    # Return all trials by train job
    else:
        with admin:
            return jsonify(admin.get_trials_of_train_job(
                app, 
                app_version=int(app_version),
                **params)
            )

@app.route('/train_job_workers/<service_id>/stop', methods=['POST'])
@auth([])
def stop_train_job_worker(auth, service_id):
    params = get_request_params()
    with admin:
        return jsonify(admin.stop_train_job_worker(service_id, **params))

####################################
# Inference Jobs
####################################

@app.route('/inference_jobs', methods=['POST'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def create_inference_jobs(auth):
    params = get_request_params()

    if 'app_version' in params:
        params['app_version'] = int(params['app_version'])

    with admin:
        return jsonify(admin.create_inference_job(auth['user_id'], **params))

@app.route('/inference_jobs/<app>', methods=['GET'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def get_inference_jobs_of_app(auth, app):
    params = get_request_params()
    with admin:
        return jsonify(admin.get_inference_jobs_of_app(app, **params))

@app.route('/inference_jobs/<app>/<app_version>', methods=['GET'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def get_inference_job(auth, app, app_version):
    params = get_request_params()
    with admin:
        return jsonify(admin.get_inference_job(app, app_version=int(app_version), **params))

@app.route('/inference_jobs/<app>/<app_version>/stop', methods=['POST'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER])
def stop_inference_job(auth, app, app_version=-1):
    params = get_request_params()
    with admin:
        return jsonify(admin.stop_inference_job(app, app_version=int(app_version), **params))

####################################
# Models
####################################

@app.route('/models', methods=['POST'])
@auth([UserType.ADMIN, UserType.MODEL_DEVELOPER])
def create_model(auth):
    params = get_request_params()
    model_file_bytes = request.files['model_file_bytes'].read()
    params['model_file_bytes'] = model_file_bytes

    with admin:
        return jsonify(admin.create_model(auth['user_id'], **params))

@app.route('/models', methods=['GET'])
@auth([UserType.ADMIN, UserType.APP_DEVELOPER, UserType.MODEL_DEVELOPER])
def get_models(auth):
    params = get_request_params()

    # Return models by task
    if params.get('task') is not None:
        with admin:
            return jsonify(admin.get_models_of_task(**params))
    
    # Return all models
    else:
        with admin:
            return jsonify(admin.get_models(**params))

# Handle uncaught exceptions with a server error & the error's stack trace (for development)
@app.errorhandler(Exception)
def handle_error(error):
    return traceback.format_exc(), 500