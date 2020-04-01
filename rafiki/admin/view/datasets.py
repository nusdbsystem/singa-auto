import tempfile
from rafiki.constants import UserType, RequestsParameters
from flask import request, jsonify, Blueprint, g
from rafiki.utils.auth import auth
from rafiki.utils.requests_params import param_check

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
