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
from flask import Flask, jsonify, g, request, make_response, send_from_directory
import mimetypes
from flask.wrappers import Response
from flask_cors import CORS
from .predictor import Predictor
from singa_auto.model import utils
import traceback
import json
import pickle
import uuid

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
        # Deprecated method to upload data with json or "img" key
        if request.files.getlist('img'):
            img_stores = request.files.getlist('img')
            img_bytes = [
                img for img in [img_store.read() for img_store in img_stores]
                if img
            ]
            if not img_bytes:
                return jsonify({'ErrorMsg': 'No image provided'}), 400
            queries = utils.dataset.load_images(img_bytes)

        elif request.get_json():
            data = request.get_json()
            queries = [data]
        # new
        elif request.form.getlist('input'):
            queries = request.form.getlist('input')
        elif request.files.getlist('input'):
            queries = [obj.read() for obj in request.files.getlist('input')]
        else:
            return jsonify({'ErrorMsg': 'text/file data should be included in form data (set "input" as key)'}), 400

        predictor = get_predictor()
        predictions: List[Any] = predictor.predict(queries)

        # determine the format of returned result
        if len(predictions) > 1 and isinstance(predictions[1], str):
            suffix = predictions[1]
            predictions = predictions[0]
        else:
            suffix = ''

        mime_type = mimetypes.guess_type(f'prefix.{suffix}')[0]
        if mime_type is None:
            mime_type = 'application/octet-stream'

 

        try:
            # Deprecated method to convert predictions to json
            json_predictions = jsonify(predictions)
            res = make_response(json_predictions, 200)
            res.headers.set('Content-Type', 'application/json')
        except:
            # if prediction includes non-jsonify object
            #res = make_response(predictions, 200)


            prediction_dir = os.path.abspath(".") + "/tmp/prediction/"
            os.system("mkdir %s"%prediction_dir)

            prediction_file_id = str(service_id) + "-" + str(uuid.uuid4())
            prediction_result_dir = prediction_dir + prediction_file_id + "/"
            os.system("mkdir %s"%prediction_result_dir)
            os.system("mkdir %s/prediction"%prediction_result_dir)


            prediction_file = prediction_file_id + ".tar"
            os.system("tar -cvf %s/%s -C %s prediction"%(prediction_dir, prediction_file, prediction_result_dir))


            res = make_response(send_from_directory(prediction_dir, prediction_file, as_attachment=True), 200)
            res.headers.set('Content-Type', mime_type)
            res.headers.set('Accept-Encoding','gzip, deflate, br')
        
        return res

    except:
        # for debug,print the error
        traceback.print_exc()
        logging.error(traceback.format_exc())
        return jsonify(
            {'ErrorMsg':
             'Server Error:{}'.format(traceback.format_exc())}), 500
