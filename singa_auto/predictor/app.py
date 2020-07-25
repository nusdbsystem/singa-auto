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
from typing import Any, List
from flask import Flask, jsonify, g, request
from flask_cors import CORS
from .predictor import Predictor
from singa_auto.model import utils
import traceback

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
    try:
        if request.files.getlist('img'):
            img_stores = request.files.getlist('img')
            img_bytes = [
                img for img in [img_store.read() for img_store in img_stores] if img
            ]
            if not img_bytes:
                return jsonify({'ErrorMsg': 'No image provided'}), 400
            queries = utils.dataset.load_images(img_bytes)
            print("img_bytes_first 10 bytes", img_bytes[0][:10])
            print("queries_sizes", len(queries))
        elif request.get_json():
            data = request.get_json()
            queries = [data]
        else:
            return jsonify({'ErrorMsg': 'data should be either at files (set "img" as key) or json payload'}), 400
        predictor = get_predictor()
        predictions: List[Any] = predictor.predict(queries)
        return jsonify(predictions), 200
    except:
        # for debug,print the error
        traceback.print_exc()
        logging.error(traceback.format_exc())
        return jsonify({'ErrorMsg': 'Server Error:{}'.format(traceback.format_exc())}
                       ), 500
