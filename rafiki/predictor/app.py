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

import os
import logging
import tempfile

from flask import Flask, jsonify, request, g
from flask_cors import CORS

from .predictor import Predictor
from ..model import utils

service_id = os.environ['RAFIKI_SERVICE_ID']

logger = logging.getLogger(__name__)
app = Flask(__name__)
CORS(app)


class InvalidQueryFormatError(Exception): pass

def get_predictor() -> Predictor:
    # Allow multiple threads to each have their own instance of predictor
    if not hasattr(g, 'predictor'):
        g.predictor = Predictor(service_id)
    
    return g.predictor

# Extract request params from Flask request
def get_request_params():
    # Get params from body as JSON
    params = request.get_json()

    # If the above fails, get params from body as form data
    if params is None:
        params = request.form.to_dict()

    # Merge in query params
    query_params = {
        k: v
        for k, v in request.args.items()
    }
    params = {**params, **query_params}

    return params

@app.route('/')
def index():
    return 'Predictor is up.'

@app.route('/predict', methods=['POST'])
def predict():
    print('get predication requests')
    try:
        predictor = get_predictor()

        with tempfile.NamedTemporaryFile() as f:
            if 'img' in request.files:
                # Save img data in request body
                try:
                    file_storage = request.files['img']
                    file_storage.save(f.name)
                    query = utils.dataset.load_images([f.name]).tolist()[0]
                    file_storage.close()
                except:
                    return jsonify({'ErrorMsg': 'can not read img'}), 400
            else:
                return jsonify({'ErrorMsg': 'No image provided'}), 400
        predictions = predictor.predict([query])
        assert len(predictions) == 1
        print(predictions)
        return jsonify(predictions[0][0]), 200
    except Exception as e:
        logger.error(str(e))
        print('error', str(e))
        import sys, traceback
        traceback.print_exc(file=sys.stdout)
        return jsonify({'ErrorMsg': 'Server Error'}), 500

    # Either do single prediction or bulk predictions
    # if 'queries' in params:
    #     predictions = predictor.predict(params['queries'])
    #     return jsonify({
    #         'prediction': None,
    #         'predictions': predictions
    #     })
    # else:
    #     predictions = predictor.predict([params['query']])
    #     assert len(predictions) == 1
    #     return jsonify({
    #         'prediction': predictions[0],
    #         'predictions': []
    #     })


