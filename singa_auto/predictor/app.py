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
from flask import Flask, jsonify, g, request
from flask_cors import CORS
from .predictor import Predictor
from singa_auto.model import utils
import traceback
import json

service_id = os.environ['SINGA_AUTO_SERVICE_ID']

logger = logging.getLogger(__name__)
app = Flask(__name__)
CORS(app)


def get_predictor() -> Predictor:
    # Allow multiple threads to each have their own instance of predictor
    if not hasattr(g, 'predictor'):
        g.predictor = Predictor(service_id)

    return g.predictor


@app.route('/')
def index():
    return 'Predictor is up.'


@app.route('/', methods=['POST'])
def predict():

    if request.files.getlist('img'):
        img_stores = request.files.getlist('img')
        img_bytes = [
            img for img in [img_store.read() for img_store in img_stores] if img
        ]
        print("img_stores", img_stores)
        if not img_bytes:
            return jsonify({'ErrorMsg': 'No image provided'}), 400
        print("img_bytes_first 10 bytes", img_bytes[0][:10])
        queries = utils.dataset.load_images(img_bytes)
        print("queries_sizes", len(queries))
    elif request.get_json():
        data = request.get_json()
        queries = [data]
    elif request.data:
        data = json.loads(request.data)
        print(data)
        queries = [data]
    else:
        return jsonify({'ErrorMsg': 'data should be either at files or json payload'}), 400
    try:
        predictor = get_predictor()
        # this queries is type of List[Any]
        predictions = predictor.predict(queries)
        print(type(predictions))
        if isinstance(predictions[0], list):
            # this is only for pandavgg demo as the frontend only accept the dictionary.
            return jsonify(predictions[0][0]), 200
        elif isinstance(predictions[0], dict):
            return jsonify(predictions[0]), 200
        elif isinstance(predictions, list) and isinstance(predictions[0], str):
            # this is only match qa model,
            print("this is only match qa model")
            return predictions[0], 200
        else:
            return jsonify(predictions), 200
    except:
        # for debug,print the error
        traceback.print_exc()
        logging.error(traceback.format_exc())
        return jsonify({'ErrorMsg': 'Server Error'}), 500
